from __future__ import annotations

import asyncio
from fractions import Fraction

from av import AudioFrame
import numpy as np
import pytest

from mini_arithmetic.realtime_media import (
    BoundedPcmInput,
    MODEL_FRAME_SAMPLES,
    MODEL_SAMPLE_RATE,
    PcmBlock,
    QueuedPcmAudioTrack,
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
    audio_frame_to_model_pcm,
    resample_mono,
)


def test_exact_finite_resampling_contract() -> None:
    source = np.linspace(-0.5, 0.5, WEBRTC_FRAME_SAMPLES, dtype=np.float32)
    down = resample_mono(source, WEBRTC_SAMPLE_RATE, MODEL_SAMPLE_RATE)
    up = resample_mono(down, MODEL_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
    assert down.shape == (MODEL_FRAME_SAMPLES,)
    assert up.shape == (WEBRTC_FRAME_SAMPLES,)
    assert np.isfinite(up).all()
    with pytest.raises(ValueError, match="finite"):
        resample_mono(np.array([np.nan], dtype=np.float32), 24_000, 48_000)


def test_decoded_audio_frame_becomes_mono_24khz() -> None:
    values = np.arange(WEBRTC_FRAME_SAMPLES, dtype=np.int16)[None, :]
    frame = AudioFrame.from_ndarray(values, format="s16", layout="mono")
    frame.sample_rate = WEBRTC_SAMPLE_RATE
    frame.pts = 0
    frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
    pcm = audio_frame_to_model_pcm(frame)
    assert pcm.shape == (MODEL_FRAME_SAMPLES,)
    assert pcm.dtype == np.float32
    assert np.isfinite(pcm).all()


def test_input_queue_is_chronological_and_bounded() -> None:
    async def scenario() -> None:
        queue = BoundedPcmInput(capacity=2)
        for sequence in range(3):
            await queue.put(
                PcmBlock(
                    sequence=sequence,
                    captured_at=float(sequence),
                    received_at=float(sequence) + 0.01,
                    samples=np.zeros((MODEL_FRAME_SAMPLES,), dtype=np.float32),
                )
            )
        assert queue.pending == 2
        assert queue.dropped == 1
        assert (await queue.get()).sequence == 1
        assert (await queue.get()).sequence == 2

    asyncio.run(scenario())


def test_pcm_queues_reject_invalid_timestamps_and_unbounded_blocks() -> None:
    async def scenario() -> None:
        queue = BoundedPcmInput()
        with pytest.raises(ValueError, match="exactly 20 ms"):
            await queue.put(
                PcmBlock(
                    sequence=0,
                    captured_at=0.0,
                    received_at=0.0,
                    samples=np.zeros((MODEL_FRAME_SAMPLES + 1,), dtype=np.float32),
                )
            )
        with pytest.raises(ValueError, match="cannot precede"):
            await queue.put(
                PcmBlock(
                    sequence=0,
                    captured_at=1.0,
                    received_at=0.0,
                    samples=np.zeros((MODEL_FRAME_SAMPLES,), dtype=np.float32),
                )
            )

        track = QueuedPcmAudioTrack()
        with pytest.raises(ValueError, match="exactly 20 ms"):
            track.enqueue(np.zeros((1,), dtype=np.float32), generation=0)

    asyncio.run(scenario())


def test_playback_is_20ms_bounded_and_generation_fenced() -> None:
    async def scenario() -> None:
        async def no_sleep(_delay: float) -> None:
            return None

        track = QueuedPcmAudioTrack(capacity=2, clock=lambda: 0.0, sleep=no_sleep)
        assert track.enqueue(
            np.full((MODEL_FRAME_SAMPLES,), 0.25, dtype=np.float32), generation=0
        )
        first = await track.recv()
        assert first.samples == WEBRTC_FRAME_SAMPLES
        assert first.sample_rate == WEBRTC_SAMPLE_RATE
        assert first.pts == 0
        assert np.max(np.abs(first.to_ndarray())) > 0

        generation = track.interrupt()
        assert generation == 1
        assert not track.enqueue(
            np.ones((MODEL_FRAME_SAMPLES,), dtype=np.float32), generation=0
        )
        second = await track.recv()
        assert second.pts == WEBRTC_FRAME_SAMPLES
        assert np.max(np.abs(second.to_ndarray())) == 0
        assert track.status()["underruns"] == 1

    asyncio.run(scenario())


def test_playback_drops_oldest_without_unbounded_latency() -> None:
    async def scenario() -> None:
        async def no_sleep(_delay: float) -> None:
            return None

        track = QueuedPcmAudioTrack(capacity=2, clock=lambda: 0.0, sleep=no_sleep)
        for amplitude in (0.1, 0.2, 0.3):
            assert track.enqueue(
                np.full((MODEL_FRAME_SAMPLES,), amplitude, dtype=np.float32),
                generation=0,
            )
        assert track.status()["pending_blocks"] == 2
        assert track.status()["dropped_blocks"] == 1
        frame = await track.recv()
        peak = np.max(np.abs(frame.to_ndarray())) / 32_767.0
        assert 0.19 < peak < 0.21

    asyncio.run(scenario())
