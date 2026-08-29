"""Bounded PCM media primitives for the local MLM WebRTC gateway.

These classes transport audio only.  They do not perform speech recognition,
text-to-speech, or model inference, and they never make random weights sound
meaningful.  The contract is fixed at mono float32 24 kHz on the model side and
20 ms mono 48 kHz frames on the WebRTC side.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math
import time
from typing import Protocol

from aiortc import AudioStreamTrack, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler
import numpy as np


MODEL_SAMPLE_RATE = 24_000
WEBRTC_SAMPLE_RATE = 48_000
WEBRTC_FRAME_SAMPLES = 960
MODEL_FRAME_SAMPLES = 480
# Four 20 ms blocks is the hard queue budget used by the readiness gate.
MAX_INPUT_BLOCKS = 4
MAX_OUTPUT_BLOCKS = 4


class Clock(Protocol):
    def __call__(self) -> float: ...


def _finite_mono(samples: np.ndarray) -> np.ndarray:
    value = np.asarray(samples, dtype=np.float32)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1:
        raise ValueError("PCM must be a mono one-dimensional array")
    if value.size == 0:
        raise ValueError("PCM blocks must not be empty")
    if not np.isfinite(value).all():
        raise ValueError("PCM must contain only finite samples")
    return np.clip(value, -1.0, 1.0).astype(np.float32, copy=False)


def resample_mono(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Deterministically resample a finite mono block with exact sample count."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    value = _finite_mono(samples)
    if source_rate == target_rate:
        return value.copy()
    output_size = max(1, int(round(value.size * target_rate / source_rate)))
    source_positions = np.arange(value.size, dtype=np.float64)
    target_positions = (
        np.arange(output_size, dtype=np.float64) * source_rate / target_rate
    )
    target_positions = np.minimum(target_positions, value.size - 1)
    return np.interp(target_positions, source_positions, value).astype(np.float32)


def audio_frame_to_model_pcm(frame: AudioFrame) -> np.ndarray:
    """Convert a decoded WebRTC frame to finite mono float32 24 kHz PCM."""

    if not frame.sample_rate or frame.sample_rate <= 0:
        raise ValueError("Decoded audio frame is missing a sample rate")
    converter = AudioResampler(
        format="fltp",
        layout="mono",
        rate=frame.sample_rate,
    )
    converted = converter.resample(frame)
    if not converted:
        return np.zeros((0,), dtype=np.float32)
    planar = np.concatenate(
        [
            np.asarray(item.to_ndarray(), dtype=np.float32).reshape(-1)
            for item in converted
        ]
    )
    if planar.size == 0:
        return planar
    return resample_mono(planar, frame.sample_rate, MODEL_SAMPLE_RATE)


@dataclass(frozen=True)
class PcmBlock:
    sequence: int
    captured_at: float
    received_at: float
    samples: np.ndarray

    @property
    def ingress_latency_ms(self) -> float:
        return max(0.0, (self.received_at - self.captured_at) * 1_000.0)


class BoundedPcmInput:
    """Chronological audio queue which drops the oldest block under pressure."""

    def __init__(self, capacity: int = MAX_INPUT_BLOCKS) -> None:
        if capacity <= 0:
            raise ValueError("PCM queue capacity must be positive")
        self.capacity = capacity
        self._items: deque[PcmBlock] = deque()
        self._condition = asyncio.Condition()
        self.dropped = 0

    async def put(self, block: PcmBlock) -> None:
        if isinstance(block.sequence, bool) or not isinstance(block.sequence, int):
            raise ValueError("PCM sequence numbers must be integers")
        if block.sequence < 0:
            raise ValueError("PCM sequence numbers must be non-negative")
        if not math.isfinite(block.captured_at) or not math.isfinite(block.received_at):
            raise ValueError("PCM timestamps must be finite")
        if block.received_at < block.captured_at:
            raise ValueError("PCM receive time cannot precede capture time")
        samples = _finite_mono(block.samples)
        if samples.size != MODEL_FRAME_SAMPLES:
            raise ValueError("PCM input blocks must contain exactly 20 ms")
        normalized = PcmBlock(
            sequence=block.sequence,
            captured_at=block.captured_at,
            received_at=block.received_at,
            samples=samples,
        )
        async with self._condition:
            if self._items and normalized.sequence <= self._items[-1].sequence:
                raise ValueError("PCM sequence numbers must increase")
            while len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped += 1
            self._items.append(normalized)
            self._condition.notify()

    async def get(self) -> PcmBlock:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._items))
            return self._items.popleft()

    def clear(self) -> None:
        self._items.clear()

    @property
    def pending(self) -> int:
        return len(self._items)


class MicrophoneConsumer:
    """Read an inbound aiortc audio track into bounded 24 kHz PCM blocks."""

    def __init__(
        self,
        track: MediaStreamTrack,
        queue: BoundedPcmInput,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        if track.kind != "audio":
            raise ValueError("Microphone consumer requires an audio track")
        self.track = track
        self.queue = queue
        self.clock = clock
        self.sequence = 0
        self._buffer = np.zeros((0,), dtype=np.float32)

    async def run(self) -> None:
        try:
            while True:
                frame = await self.track.recv()
                if not isinstance(frame, AudioFrame):
                    raise ValueError("Microphone track returned a non-audio frame")
                received_at = self.clock()
                pcm = audio_frame_to_model_pcm(frame)
                if pcm.size:
                    self._buffer = np.concatenate((self._buffer, pcm))
                while self._buffer.size >= MODEL_FRAME_SAMPLES:
                    block = self._buffer[:MODEL_FRAME_SAMPLES]
                    self._buffer = self._buffer[MODEL_FRAME_SAMPLES:]
                    duration = MODEL_FRAME_SAMPLES / MODEL_SAMPLE_RATE
                    await self.queue.put(
                        PcmBlock(
                            sequence=self.sequence,
                            captured_at=received_at - duration,
                            received_at=received_at,
                            samples=block,
                        )
                    )
                    self.sequence += 1
        except (MediaStreamError, asyncio.CancelledError):
            raise


@dataclass(frozen=True)
class PlaybackBlock:
    generation: int
    samples: np.ndarray


class QueuedPcmAudioTrack(AudioStreamTrack):
    """A bounded generation-fenced 24 kHz PCM source for WebRTC playback."""

    kind = "audio"

    def __init__(
        self,
        *,
        capacity: int = MAX_OUTPUT_BLOCKS,
        clock: Clock = time.monotonic,
        sleep=asyncio.sleep,
    ) -> None:
        super().__init__()
        if capacity <= 0:
            raise ValueError("Playback queue capacity must be positive")
        self.capacity = capacity
        self.clock = clock
        self.sleep = sleep
        self.generation = 0
        self._queue: deque[PlaybackBlock] = deque()
        self._model_buffer = np.zeros((0,), dtype=np.float32)
        self._start: float | None = None
        self._pts = 0
        self.dropped = 0
        self.underruns = 0

    def enqueue(self, samples: np.ndarray, *, generation: int) -> bool:
        value = _finite_mono(samples)
        if value.size != MODEL_FRAME_SAMPLES:
            raise ValueError("Playback blocks must contain exactly 20 ms")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("Playback generation must be a non-negative integer")
        if generation != self.generation:
            return False
        while len(self._queue) >= self.capacity:
            self._queue.popleft()
            self.dropped += 1
        self._queue.append(PlaybackBlock(generation=generation, samples=value))
        return True

    def interrupt(self) -> int:
        self.generation += 1
        self._queue.clear()
        self._model_buffer = np.zeros((0,), dtype=np.float32)
        return self.generation

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        now = self.clock()
        if self._start is None:
            self._start = now
        else:
            deadline = self._start + self._pts / WEBRTC_SAMPLE_RATE
            delay = deadline - now
            if delay > 0:
                await self.sleep(delay)

        while self._queue and self._model_buffer.size < MODEL_FRAME_SAMPLES:
            block = self._queue.popleft()
            if block.generation == self.generation:
                self._model_buffer = np.concatenate((self._model_buffer, block.samples))
        if self._model_buffer.size >= MODEL_FRAME_SAMPLES:
            model_pcm = self._model_buffer[:MODEL_FRAME_SAMPLES]
            self._model_buffer = self._model_buffer[MODEL_FRAME_SAMPLES:]
        else:
            model_pcm = np.zeros((MODEL_FRAME_SAMPLES,), dtype=np.float32)
            self.underruns += 1

        web_pcm = resample_mono(model_pcm, MODEL_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
        if web_pcm.size != WEBRTC_FRAME_SAMPLES:
            raise RuntimeError("Playback resampling violated the 20 ms frame contract")
        int16 = np.rint(np.clip(web_pcm, -1.0, 1.0) * 32_767.0).astype(np.int16)
        frame = AudioFrame.from_ndarray(int16[None, :], format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
        self._pts += WEBRTC_FRAME_SAMPLES
        return frame

    def status(self) -> dict[str, int]:
        queued_samples = sum(item.samples.size for item in self._queue)
        return {
            "generation": self.generation,
            "pending_blocks": len(self._queue),
            "pending_samples": queued_samples + int(self._model_buffer.size),
            "dropped_blocks": self.dropped,
            "underruns": self.underruns,
        }
