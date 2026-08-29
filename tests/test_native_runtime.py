from __future__ import annotations

import asyncio
import hashlib
import io
import json
from pathlib import Path

import av
import numpy as np
import pytest

from mini_arithmetic.native_runtime import (
    DEFAULT_INPUT_CAPACITY,
    DEFAULT_OUTPUT_CAPACITY,
    MINIMUM_VALIDATION_RECORDS,
    NATIVE_CHECKPOINT_FORMAT,
    NATIVE_CHECKPOINT_SCHEMA_VERSION,
    NATIVE_VIDEO_HEIGHT,
    NATIVE_VIDEO_WIDTH,
    NativeBackendResult,
    NativeRuntimeClosed,
    NativeRuntimeFactory,
    NativeVideoFrame,
    ScratchNativeRuntime,
    canonicalize_selected_frame,
    create_untrained_runtime,
    load_native_checkpoint,
)
from mini_arithmetic.realtime_media import MODEL_FRAME_SAMPLES, PcmBlock
from mini_arithmetic.realtime_protocol import SelectedFrame


_POLICIES = {
    ("audio_codec", "reconstruction_snr_db"): ("at_least", 15.0, 16.0),
    ("speech_understanding", "masked_text_accuracy"): (
        "at_least",
        0.80,
        0.90,
    ),
    ("vision", "masked_text_accuracy"): ("at_least", 0.70, 0.80),
    ("vision", "frame_ablation_loss_delta"): ("at_least", 0.05, 0.10),
    ("talker", "model_code_accuracy"): ("at_least", 0.60, 0.70),
    ("conversation", "masked_text_accuracy"): ("at_least", 0.80, 0.90),
    ("conversation", "model_code_accuracy"): ("at_least", 0.60, 0.70),
}
_GATE_METRICS = {
    "audio_codec": ("stage_loss", "reconstruction_snr_db"),
    "speech_understanding": ("stage_loss", "masked_text_accuracy"),
    "vision": ("stage_loss", "masked_text_accuracy", "frame_ablation_loss_delta"),
    "talker": ("stage_loss", "model_code_accuracy"),
    "conversation": ("stage_loss", "masked_text_accuracy", "model_code_accuracy"),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality(gate: str, ready: bool) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for metric in _GATE_METRICS[gate]:
        if metric == "stage_loss":
            direction, threshold, value = "at_most", 2.0, 1.0
        else:
            direction, threshold, value = _POLICIES[(gate, metric)]
        metrics[metric] = {
            "value": value if ready else None,
            "threshold": threshold if ready else None,
            "direction": direction,
            "passed": ready,
        }
    return {
        "stage": gate if ready else None,
        "evaluated_optimizer_steps": 5_000 if ready else None,
        "validation_records": MINIMUM_VALIDATION_RECORDS if ready else 0,
        "minimum_validation_records": MINIMUM_VALIDATION_RECORDS,
        "validation_manifest_sha256": "a" * 64 if ready else None,
        "metrics": metrics,
        "passed": ready,
    }


def _write_checkpoint(
    tmp_path: Path,
    *,
    unready_gate: str | None = None,
    real_weights: bool = False,
) -> tuple[Path, dict[str, object]]:
    source_config = Path(__file__).parents[1] / "configs" / "multimodal-tiny.json"
    config = tmp_path / "multimodal-config.json"
    config.write_bytes(source_config.read_bytes())
    weights = tmp_path / "model.safetensors"
    if real_weights:
        from mini_arithmetic.multimodal import (
            MultimodalConfig,
            create_scratch_multimodal_model,
        )

        model = create_scratch_multimodal_model(MultimodalConfig.from_json(config))
        model.save_weights(str(weights))
    else:
        weights.write_bytes(b"scratch-owned-native-test-weights")
    gate_names = tuple(_GATE_METRICS)
    payload: dict[str, object] = {
        "schema_version": NATIVE_CHECKPOINT_SCHEMA_VERSION,
        "format": NATIVE_CHECKPOINT_FORMAT,
        "checkpoint_id": "native-test-1",
        "scratch_only": True,
        "external_weights": False,
        "model": {
            "config": config.name,
            "weights": weights.name,
            "config_sha256": _digest(config),
            "weights_sha256": _digest(weights),
        },
        "training": {
            "optimizer_steps": 5_000,
            "gates": {
                gate: {
                    "trained": gate != unready_gate,
                    "validation_passed": gate != unready_gate,
                }
                for gate in gate_names
            },
            "quality": {
                gate: _quality(gate, gate != unready_gate) for gate in gate_names
            },
        },
        "runtime": {
            "sample_rate": 24_000,
            "frame_size": 8,
            "video_width": NATIVE_VIDEO_WIDTH,
            "video_height": NATIVE_VIDEO_HEIGHT,
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def _pcm(sequence: int, *, value: float = 0.1, samples: int = 480) -> PcmBlock:
    return PcmBlock(
        sequence=sequence,
        captured_at=100.0 + sequence * 0.02,
        received_at=100.01 + sequence * 0.02,
        samples=np.full((samples,), value, dtype=np.float32),
    )


def _jpeg(value: int = 127) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mjpeg") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuvj420p"
        frame = av.VideoFrame.from_ndarray(
            np.full((48, 64, 3), value, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _selected(
    sequence: int = 0, *, timestamp_ms: int = 1_700_000_000_000
) -> SelectedFrame:
    return SelectedFrame(
        version=1,
        frame_id=f"frame-{sequence}",
        sequence=sequence,
        timestamp_ms=timestamp_ms,
        difference_percent=25.0,
        keyframe=sequence == 0,
        content_type="image/jpeg",
        jpeg=_jpeg(),
    )


class FakeBackend:
    model_loaded = True
    random_init = False
    parameter_count = 123

    def __init__(
        self,
        *,
        output_samples: int = 0,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self._frame_size = 8
        self.output_samples = output_samples
        self.entered = entered
        self.release = release
        self.audio_times: list[float] = []
        self.frames: list[NativeVideoFrame] = []
        self.interruptions: list[int] = []
        self.closed = False

    @property
    def frame_size(self) -> int:
        return self._frame_size

    async def accept_audio(self, block: PcmBlock) -> NativeBackendResult:
        self.audio_times.append(block.captured_at)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        audio = (
            (np.linspace(-0.5, 0.5, self.output_samples, dtype=np.float32),)
            if self.output_samples
            else ()
        )
        return NativeBackendResult(accepted=True, audio=audio)

    async def accept_frame(self, frame: NativeVideoFrame) -> NativeBackendResult:
        self.frames.append(frame)
        return NativeBackendResult(accepted=True)

    async def interrupt(self, generation: int) -> None:
        self.interruptions.append(generation)

    async def close(self) -> None:
        self.closed = True

    def cache_status(self) -> dict[str, int]:
        return {"audio_calls": len(self.audio_times), "video_calls": len(self.frames)}


def test_manifest_requires_scratch_provenance_digests_and_heldout_quality(
    tmp_path: Path,
) -> None:
    manifest, payload = _write_checkpoint(tmp_path)
    loaded = load_native_checkpoint(manifest)
    assert loaded.manifest.voice_trained is True
    assert loaded.manifest.vision_trained is True
    assert loaded.manifest.conversation_ready is True
    assert loaded.manifest.quality["audio_codec"].validation_records == 128

    payload["external_weights"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="scratch-only"):
        load_native_checkpoint(manifest)


def test_manifest_rejects_quality_claim_not_supported_by_measurement(
    tmp_path: Path,
) -> None:
    manifest, payload = _write_checkpoint(tmp_path)
    training = payload["training"]
    assert isinstance(training, dict)
    quality = training["quality"]
    assert isinstance(quality, dict)
    codec = quality["audio_codec"]
    assert isinstance(codec, dict)
    metrics = codec["metrics"]
    assert isinstance(metrics, dict)
    snr = metrics["reconstruction_snr_db"]
    assert isinstance(snr, dict)
    snr["value"] = 14.9
    # Leave the claimed metric/evidence/gate pass flags true: loader recomputes.
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pass flag is inconsistent"):
        load_native_checkpoint(manifest)


def test_manifest_rejects_unbound_validation_evidence(tmp_path: Path) -> None:
    manifest, payload = _write_checkpoint(tmp_path)
    training = payload["training"]
    assert isinstance(training, dict)
    quality = training["quality"]
    assert isinstance(quality, dict)
    codec = quality["audio_codec"]
    assert isinstance(codec, dict)
    codec["validation_manifest_sha256"] = None
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation manifest digest"):
        load_native_checkpoint(manifest)


def test_manifest_digest_and_path_escape_fail_closed(tmp_path: Path) -> None:
    manifest, payload = _write_checkpoint(tmp_path)
    model = payload["model"]
    assert isinstance(model, dict)
    model["weights_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum does not match"):
        load_native_checkpoint(manifest)

    model["weights"] = "../outside.safetensors"
    outside = tmp_path.parent / "outside.safetensors"
    outside.write_bytes(b"x")
    model["weights_sha256"] = _digest(outside)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        load_native_checkpoint(manifest)


def test_untrained_default_is_lightweight_truthful_and_silent() -> None:
    async def scenario() -> None:
        runtime = create_untrained_runtime()
        status = runtime.status()
        assert status["model_loaded"] is False
        assert status["checkpoint_loaded"] is False
        assert status["voice_trained"] is False
        assert status["conversation_ready"] is False
        assert status["can_emit_audio"] is False
        assert status["parameters"] == 0
        assert status["input_capacity"] == DEFAULT_INPUT_CAPACITY == 4
        assert status["output_capacity"] == DEFAULT_OUTPUT_CAPACITY == 4
        assert await runtime.accept_audio(_pcm(0)) is False
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(runtime.next_output(), timeout=0.01)
        await runtime.close()

    asyncio.run(scenario())


def test_unattested_backend_pcm_is_suppressed_even_if_model_returns_it() -> None:
    async def scenario() -> None:
        backend = FakeBackend(output_samples=MODEL_FRAME_SAMPLES)
        runtime = ScratchNativeRuntime(backend)
        assert await runtime.accept_audio(_pcm(0)) is True
        assert runtime.status()["suppressed_outputs"] == 1
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(runtime.next_output(), timeout=0.01)
        await runtime.close()

    asyncio.run(scenario())


def test_explicit_random_model_diagnostic_emits_without_claiming_training() -> None:
    async def scenario() -> None:
        backend = FakeBackend(output_samples=MODEL_FRAME_SAMPLES)
        backend.random_init = True
        runtime = ScratchNativeRuntime(backend, allow_untrained_audio=True)
        status = runtime.status()
        assert status["model_loaded"] is True
        assert status["random_init"] is True
        assert status["untrained_audio_enabled"] is True
        assert status["voice_trained"] is False
        assert status["vision_trained"] is False
        assert status["conversation_ready"] is False
        assert status["can_emit_audio"] is True
        speech = _pcm(0)
        speech_samples = speech.samples.copy()
        speech_samples[::2] *= -1
        speech = PcmBlock(
            sequence=speech.sequence,
            captured_at=speech.captured_at,
            received_at=speech.received_at,
            samples=speech_samples,
        )
        assert await runtime.accept_audio(speech) is False
        assert await runtime.interrupt(1) == 1
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(runtime.next_output(), timeout=0.01)
        assert await runtime.end_turn("turn-1") is True
        output = await asyncio.wait_for(runtime.next_output(), timeout=0.1)
        assert output.samples.shape == (MODEL_FRAME_SAMPLES,)
        await runtime.close()

    asyncio.run(scenario())


def test_random_model_diagnostic_waits_for_vad_turn_boundary() -> None:
    async def scenario() -> None:
        backend = FakeBackend(output_samples=MODEL_FRAME_SAMPLES)
        backend.random_init = True
        runtime = ScratchNativeRuntime(backend, allow_untrained_audio=True)

        assert await runtime.accept_audio(_pcm(0, value=0.0)) is False
        assert backend.audio_times == []
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(runtime.next_output(), timeout=0.01)

        speech = _pcm(1)
        speech_samples = speech.samples.copy()
        speech_samples[::2] *= -1
        speech = PcmBlock(
            sequence=speech.sequence,
            captured_at=speech.captured_at,
            received_at=speech.received_at,
            samples=speech_samples,
        )
        assert await runtime.interrupt(1) == 1
        assert len(backend.audio_times) == 1
        assert await runtime.accept_audio(speech) is True
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(runtime.next_output(), timeout=0.01)
        assert await runtime.end_turn("turn-2") is True
        await asyncio.wait_for(runtime.next_output(), timeout=0.1)
        await runtime.close()

    asyncio.run(scenario())


def test_untrained_audio_opt_in_rejects_nonrandom_backends() -> None:
    with pytest.raises(ValueError, match="random-initialized"):
        ScratchNativeRuntime(FakeBackend(), allow_untrained_audio=True)


def test_incomplete_native_checkpoint_cannot_emit(tmp_path: Path) -> None:
    manifest, _ = _write_checkpoint(tmp_path, unready_gate="speech_understanding")
    checkpoint = load_native_checkpoint(manifest)

    async def scenario() -> None:
        runtime = NativeRuntimeFactory(
            lambda: FakeBackend(output_samples=MODEL_FRAME_SAMPLES),
            checkpoint=checkpoint,
        ).create()
        status = runtime.status()
        assert status["model_loaded"] is True
        assert status["checkpoint_loaded"] is True
        assert status["voice_trained"] is False
        assert status["can_emit_audio"] is False
        await runtime.accept_audio(_pcm(0))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(runtime.next_output(), timeout=0.01)
        await runtime.close()

    asyncio.run(scenario())


def test_trained_checkpoint_streams_only_fixed_20ms_pcm_with_global_sequence(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_checkpoint(tmp_path)
    checkpoint = load_native_checkpoint(manifest)

    async def scenario() -> None:
        backend = FakeBackend(output_samples=2 * MODEL_FRAME_SAMPLES)
        runtime = NativeRuntimeFactory(lambda: backend, checkpoint=checkpoint).create()
        status = runtime.status()
        assert status["model_loaded"] is True
        assert status["voice_trained"] is True
        assert status["vision_trained"] is True
        assert status["conversation_ready"] is True
        assert status["can_emit_audio"] is True
        assert await runtime.accept_audio(_pcm(0)) is True
        first = await runtime.next_output()
        second = await runtime.next_output()
        assert first.samples.shape == second.samples.shape == (MODEL_FRAME_SAMPLES,)
        assert (first.generation, second.generation) == (0, 0)
        assert (first.sequence, second.sequence) == (0, 1)
        assert first.produced_at < second.produced_at

        assert await runtime.interrupt(1) == 1
        assert await runtime.accept_audio(_pcm(1)) is True
        after = await runtime.next_output()
        assert after.generation == 1
        assert after.sequence == 2
        assert backend.interruptions == [1]
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_input_queue_is_globally_bounded_to_four(tmp_path: Path) -> None:
    manifest, _ = _write_checkpoint(tmp_path)
    checkpoint = load_native_checkpoint(manifest)

    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        backend = FakeBackend(entered=entered, release=release)
        runtime = NativeRuntimeFactory(lambda: backend, checkpoint=checkpoint).create()
        tasks = [asyncio.create_task(runtime.accept_audio(_pcm(0)))]
        await entered.wait()
        tasks.extend(
            asyncio.create_task(runtime.accept_audio(_pcm(sequence)))
            for sequence in range(1, 7)
        )
        await asyncio.sleep(0)
        status = runtime.status()
        assert status["input_depth"] == 4
        assert status["dropped_inputs"] == 2
        release.set()
        results = await asyncio.gather(*tasks)
        assert results.count(False) == 2
        await runtime.close()

    asyncio.run(scenario())


def test_selected_jpeg_is_canonical_rgb_and_browser_epoch_is_not_model_time(
    tmp_path: Path,
) -> None:
    canonical = canonicalize_selected_frame(_selected())
    assert canonical.rgb.shape == (NATIVE_VIDEO_HEIGHT, NATIVE_VIDEO_WIDTH, 3)
    assert canonical.rgb.dtype == np.float32
    assert 0.0 <= float(canonical.rgb.min()) <= float(canonical.rgb.max()) <= 1.0
    assert canonical.source_timestamp_ms == 1_700_000_000_000
    assert canonical.session_time_seconds == 0.0

    manifest, _ = _write_checkpoint(tmp_path)
    checkpoint = load_native_checkpoint(manifest)

    class Clock:
        def __init__(self) -> None:
            self.value = 50.0

        def __call__(self) -> float:
            current = self.value
            self.value += 0.025
            return current

    async def scenario() -> None:
        backend = FakeBackend()
        runtime = ScratchNativeRuntime(backend, checkpoint=checkpoint, clock=Clock())
        assert await runtime.accept_frame(_selected()) is True
        assert len(backend.frames) == 1
        assert 0.0 <= backend.frames[0].session_time_seconds < 1.0
        assert backend.frames[0].source_timestamp_ms == 1_700_000_000_000
        assert await runtime.accept_audio(_pcm(0)) is True
        assert 0.0 <= backend.audio_times[0] < 1.0
        await runtime.close()

    asyncio.run(scenario())


def test_invalid_media_and_nonmonotonic_sequences_are_rejected(tmp_path: Path) -> None:
    manifest, _ = _write_checkpoint(tmp_path)
    checkpoint = load_native_checkpoint(manifest)

    async def scenario() -> None:
        runtime = NativeRuntimeFactory(
            lambda: FakeBackend(), checkpoint=checkpoint
        ).create()
        await runtime.accept_audio(_pcm(0))
        with pytest.raises(ValueError, match="sequence must increase"):
            await runtime.accept_audio(_pcm(0))
        bad = _pcm(1)
        bad.samples[0] = np.nan
        with pytest.raises(ValueError, match="finite bounded"):
            await runtime.accept_audio(bad)
        await runtime.accept_frame(_selected(0))
        with pytest.raises(ValueError, match="video sequence"):
            await runtime.accept_frame(_selected(0))
        await runtime.close()

    asyncio.run(scenario())


def test_close_unblocks_output_waiter() -> None:
    async def scenario() -> None:
        runtime = create_untrained_runtime()
        waiter = asyncio.create_task(runtime.next_output())
        await asyncio.sleep(0)
        await runtime.close()
        with pytest.raises(NativeRuntimeClosed):
            await waiter
        assert runtime.status()["closed"] is True

    asyncio.run(scenario())


def test_real_tiny_checkpoint_loads_once_and_sessions_have_private_caches(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_checkpoint(tmp_path, real_weights=True)
    factory = NativeRuntimeFactory.from_checkpoint(manifest)
    first = factory.create("session-1", "connection-1")
    second = factory.create("session-2", "connection-2")
    assert first.status()["parameters"] == second.status()["parameters"]
    assert first.status()["parameters"] > 0
    assert first._backend is not second._backend
    assert first._backend.model is second._backend.model

    async def exercise_and_close() -> None:
        assert await first.accept_audio(_pcm(0)) is True
        output = await asyncio.wait_for(first.next_output(), timeout=2.0)
        assert output.samples.shape == (MODEL_FRAME_SAMPLES,)
        assert first.status()["cache"]["audio_frames_encoded"] == 60
        assert second.status()["cache"]["audio_frames_encoded"] == 0
        await first.close()
        await second.close()

    asyncio.run(exercise_and_close())
