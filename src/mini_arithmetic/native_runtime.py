"""Fail-closed, scratch-only runtime boundary for native realtime MLM sessions.

The runtime deliberately has no ASR, TTS, provider API, or pretrained-model
fallback.  Its normal untrained runtime is lightweight and silent.  A separate,
explicit random-initialization diagnostic may instantiate the native model and
emit its meaningless waveform so the genuine model-to-WebRTC audio path can be
tested before training.  That diagnostic never satisfies any training or
conversation-readiness gate.

One :class:`NativeRuntimeFactory` owns the immutable MLX model weights.  Each
call creates a session with independent codec/thinker/talker caches, queues,
and interruption generation.  Inference against shared MLX weights is
serialized until a measured multi-session scheduler is introduced.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import av
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .multimodal import (
    MultimodalConfig,
    TimedVideo,
    create_scratch_multimodal_model,
)
from .realtime_media import MODEL_FRAME_SAMPLES, MODEL_SAMPLE_RATE, PcmBlock
from .realtime_protocol import MAX_SELECTED_FRAME_BYTES, SelectedFrame


NATIVE_CHECKPOINT_SCHEMA_VERSION = 1
NATIVE_CHECKPOINT_FORMAT = "mlm-native-realtime-v1"
NATIVE_GATE_NAMES = (
    "audio_codec",
    "speech_understanding",
    "vision",
    "talker",
    "conversation",
)
VOICE_GATE_NAMES = (
    "audio_codec",
    "speech_understanding",
    "talker",
    "conversation",
)
NATIVE_VIDEO_WIDTH = 320
NATIVE_VIDEO_HEIGHT = 192
DEFAULT_INPUT_CAPACITY = 4
DEFAULT_OUTPUT_CAPACITY = 4
UNTRAINED_RESPONSE_MAX_SECONDS = 1.5
UNTRAINED_INPUT_PREROLL_SECONDS = 0.25
MAX_AUDIO_BLOCK_SAMPLES = 4_800  # 200 ms at the fixed 24 kHz model rate.
MAX_NATIVE_MANIFEST_BYTES = 64 * 1024
MAX_NATIVE_CONFIG_BYTES = 2 * 1024 * 1024
MAX_DECODED_VIDEO_PIXELS = 4_096 * 4_096
MAX_SESSION_TIME_SECONDS = 24 * 60 * 60

_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MODEL_FIELDS = {"config", "weights", "config_sha256", "weights_sha256"}
_TRAINING_FIELDS = {"optimizer_steps", "gates", "quality"}
_RUNTIME_FIELDS = {
    "sample_rate",
    "frame_size",
    "video_width",
    "video_height",
}
MINIMUM_VALIDATION_RECORDS = 128
MINIMUM_TRAINED_UPDATES = 1_000
_QUALITY_METRICS = {
    "audio_codec": ("stage_loss", "reconstruction_snr_db"),
    "speech_understanding": ("stage_loss", "masked_text_accuracy"),
    "vision": (
        "stage_loss",
        "masked_text_accuracy",
        "frame_ablation_loss_delta",
    ),
    "talker": ("stage_loss", "model_code_accuracy"),
    "conversation": (
        "stage_loss",
        "masked_text_accuracy",
        "model_code_accuracy",
    ),
}
_QUALITY_POLICIES = {
    ("audio_codec", "reconstruction_snr_db"): ("at_least", 15.0),
    ("speech_understanding", "masked_text_accuracy"): ("at_least", 0.80),
    ("vision", "masked_text_accuracy"): ("at_least", 0.70),
    ("vision", "frame_ablation_loss_delta"): ("at_least", 0.05),
    ("talker", "model_code_accuracy"): ("at_least", 0.60),
    ("conversation", "masked_text_accuracy"): ("at_least", 0.80),
    ("conversation", "model_code_accuracy"): ("at_least", 0.60),
}


class NativeRuntimeError(RuntimeError):
    """A runtime or native-checkpoint contract violation."""


class NativeRuntimeClosed(NativeRuntimeError):
    """Raised when a consumer waits for output after its session has closed."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("native checkpoint JSON contains a duplicate field")
        value[key] = item
    return value


def _read_json_object(path: Path, *, maximum_bytes: int) -> dict[str, object]:
    raw = path.read_bytes()
    if not raw or len(raw) > maximum_bytes:
        raise ValueError("native checkpoint JSON is empty or too large")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("native checkpoint JSON contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("native checkpoint is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("native checkpoint must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file(root: Path, value: object, *, suffix: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("native checkpoint file name is invalid")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("native checkpoint files must use relative paths")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("native checkpoint file escapes its directory") from error
    if resolved.suffix != suffix or not resolved.is_file():
        raise FileNotFoundError(f"Native checkpoint file not found: {resolved}")
    return resolved


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"native checkpoint {name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"native checkpoint {name} must be a non-negative integer")
    return value


def _finite_or_none(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"native checkpoint {name} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"native checkpoint {name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class NativeTrainingGate:
    trained: bool
    validation_passed: bool

    @property
    def ready(self) -> bool:
        return self.trained and self.validation_passed


@dataclass(frozen=True, slots=True)
class NativeQualityMetric:
    value: float | None
    threshold: float | None
    direction: str
    passed: bool


@dataclass(frozen=True, slots=True)
class NativeGateQuality:
    stage: str | None
    evaluated_optimizer_steps: int | None
    validation_records: int
    minimum_validation_records: int
    validation_manifest_sha256: str | None
    metrics: Mapping[str, NativeQualityMetric]
    passed: bool


@dataclass(frozen=True, slots=True)
class NativeCheckpointManifest:
    schema_version: int
    format: str
    checkpoint_id: str
    scratch_only: bool
    external_weights: bool
    config_name: str
    weights_name: str
    config_sha256: str
    weights_sha256: str
    optimizer_steps: int
    gates: Mapping[str, NativeTrainingGate]
    quality: Mapping[str, NativeGateQuality]
    sample_rate: int
    frame_size: int
    video_width: int
    video_height: int

    @property
    def voice_trained(self) -> bool:
        return all(
            self.gates[name].ready and self.quality[name].passed
            for name in VOICE_GATE_NAMES
        )

    @property
    def vision_trained(self) -> bool:
        return self.gates["vision"].ready and self.quality["vision"].passed

    @property
    def conversation_ready(self) -> bool:
        return self.voice_trained and self.vision_trained


@dataclass(frozen=True, slots=True)
class LoadedNativeCheckpoint:
    manifest: NativeCheckpointManifest
    manifest_path: Path = field(repr=False)
    config_path: Path = field(repr=False)
    weights_path: Path = field(repr=False)
    config: MultimodalConfig = field(repr=False)


def load_native_checkpoint(path: str | Path) -> LoadedNativeCheckpoint:
    """Validate an MLM-native checkpoint without importing arbitrary code."""

    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Native checkpoint manifest not found: {manifest_path}"
        )
    payload = _read_json_object(manifest_path, maximum_bytes=MAX_NATIVE_MANIFEST_BYTES)
    required = {
        "schema_version",
        "format",
        "checkpoint_id",
        "scratch_only",
        "external_weights",
        "model",
        "training",
        "runtime",
    }
    if set(payload) != required:
        raise ValueError("native checkpoint manifest fields are incompatible")
    if payload["schema_version"] != NATIVE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("native checkpoint schema is incompatible")
    if payload["format"] != NATIVE_CHECKPOINT_FORMAT:
        raise ValueError("native checkpoint format is incompatible")
    checkpoint_id = payload["checkpoint_id"]
    if not isinstance(checkpoint_id, str) or _SAFE_ID.fullmatch(checkpoint_id) is None:
        raise ValueError("native checkpoint ID is invalid")
    if payload["scratch_only"] is not True or payload["external_weights"] is not False:
        raise ValueError("native checkpoint does not attest scratch-only provenance")

    model = payload["model"]
    training = payload["training"]
    runtime = payload["runtime"]
    if not isinstance(model, dict) or set(model) != _MODEL_FIELDS:
        raise ValueError("native checkpoint model fields are incompatible")
    if not isinstance(training, dict) or set(training) != _TRAINING_FIELDS:
        raise ValueError("native checkpoint training fields are incompatible")
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("native checkpoint runtime fields are incompatible")

    optimizer_steps = _positive_integer(training["optimizer_steps"], "optimizer steps")

    root = manifest_path.parent.resolve()
    config_path = _relative_file(root, model["config"], suffix=".json")
    weights_path = _relative_file(root, model["weights"], suffix=".safetensors")
    if config_path.stat().st_size > MAX_NATIVE_CONFIG_BYTES:
        raise ValueError("native model configuration is too large")
    for field_name, resolved in (
        ("config_sha256", config_path),
        ("weights_sha256", weights_path),
    ):
        expected = model[field_name]
        if not isinstance(expected, str) or _HEX64.fullmatch(expected) is None:
            raise ValueError("native checkpoint checksum is invalid")
        if _sha256(resolved) != expected:
            raise ValueError("native checkpoint checksum does not match its file")

    gate_values = training["gates"]
    if not isinstance(gate_values, dict) or set(gate_values) != set(NATIVE_GATE_NAMES):
        raise ValueError("native checkpoint training gates are incompatible")
    gates: dict[str, NativeTrainingGate] = {}
    for name in NATIVE_GATE_NAMES:
        value = gate_values[name]
        if not isinstance(value, dict) or set(value) != {
            "trained",
            "validation_passed",
        }:
            raise ValueError(f"native checkpoint {name} gate is incompatible")
        if not isinstance(value["trained"], bool) or not isinstance(
            value["validation_passed"], bool
        ):
            raise ValueError(f"native checkpoint {name} gate must use booleans")
        gates[name] = NativeTrainingGate(
            trained=value["trained"],
            validation_passed=value["validation_passed"],
        )
    if any(gate.trained for gate in gates.values()) and (
        optimizer_steps < MINIMUM_TRAINED_UPDATES
    ):
        raise ValueError("native checkpoint trained gates require sufficient updates")

    quality_values = training["quality"]
    if not isinstance(quality_values, dict) or set(quality_values) != set(
        NATIVE_GATE_NAMES
    ):
        raise ValueError("native checkpoint quality evidence is incompatible")
    quality: dict[str, NativeGateQuality] = {}
    for gate_name in NATIVE_GATE_NAMES:
        evidence = quality_values[gate_name]
        if not isinstance(evidence, dict) or set(evidence) != {
            "stage",
            "evaluated_optimizer_steps",
            "validation_records",
            "minimum_validation_records",
            "validation_manifest_sha256",
            "metrics",
            "passed",
        }:
            raise ValueError(f"native checkpoint {gate_name} quality is incompatible")
        stage = evidence["stage"]
        if stage is not None and (
            not isinstance(stage, str) or _SAFE_ID.fullmatch(stage) is None
        ):
            raise ValueError(f"native checkpoint {gate_name} quality stage is invalid")
        validation_digest = evidence["validation_manifest_sha256"]
        if stage is None:
            if validation_digest is not None:
                raise ValueError(
                    f"native checkpoint {gate_name} quality digest requires a stage"
                )
        elif (
            not isinstance(validation_digest, str)
            or _HEX64.fullmatch(validation_digest) is None
        ):
            raise ValueError(
                f"native checkpoint {gate_name} validation manifest digest is invalid"
            )
        evaluated_steps_value = evidence["evaluated_optimizer_steps"]
        evaluated_steps = (
            None
            if evaluated_steps_value is None
            else _nonnegative_integer(
                evaluated_steps_value,
                f"{gate_name} evaluated optimizer steps",
            )
        )
        records = _nonnegative_integer(
            evidence["validation_records"], f"{gate_name} validation records"
        )
        minimum_records = _positive_integer(
            evidence["minimum_validation_records"],
            f"{gate_name} minimum validation records",
        )
        if minimum_records != MINIMUM_VALIDATION_RECORDS:
            raise ValueError(
                f"native checkpoint {gate_name} minimum validation records is unsafe"
            )
        metrics_payload = evidence["metrics"]
        required_metrics = set(_QUALITY_METRICS[gate_name])
        if (
            not isinstance(metrics_payload, dict)
            or set(metrics_payload) != required_metrics
        ):
            raise ValueError(
                f"native checkpoint {gate_name} quality metrics are incompatible"
            )
        parsed_metrics: dict[str, NativeQualityMetric] = {}
        for metric_name in _QUALITY_METRICS[gate_name]:
            metric = metrics_payload[metric_name]
            if not isinstance(metric, dict) or set(metric) != {
                "value",
                "threshold",
                "direction",
                "passed",
            }:
                raise ValueError(
                    f"native checkpoint {gate_name} {metric_name} is incompatible"
                )
            direction = metric["direction"]
            if direction not in {"at_least", "at_most"}:
                raise ValueError("native checkpoint quality direction is invalid")
            if not isinstance(metric["passed"], bool):
                raise ValueError("native checkpoint quality pass flag must be boolean")
            measured = _finite_or_none(
                metric["value"], f"{gate_name} {metric_name} value"
            )
            threshold = _finite_or_none(
                metric["threshold"], f"{gate_name} {metric_name} threshold"
            )
            if (measured is None) != (threshold is None):
                raise ValueError("native checkpoint quality value/threshold must align")
            if measured is None:
                computed_pass = False
            elif direction == "at_least":
                computed_pass = measured >= threshold
            else:
                computed_pass = measured <= threshold
            if metric["passed"] is not computed_pass:
                raise ValueError("native checkpoint quality pass flag is inconsistent")
            if metric_name == "stage_loss":
                if direction != "at_most":
                    raise ValueError("native checkpoint stage loss direction is unsafe")
                if measured is not None and (measured < 0.0 or threshold <= 0.0):
                    raise ValueError(
                        "native checkpoint stage loss must be non-negative with a positive threshold"
                    )
            else:
                required_direction, required_threshold = _QUALITY_POLICIES[
                    (gate_name, metric_name)
                ]
                if direction != required_direction or threshold not in {
                    None,
                    required_threshold,
                }:
                    raise ValueError(
                        f"native checkpoint {gate_name} {metric_name} policy is unsafe"
                    )
                if (
                    "accuracy" in metric_name
                    and measured is not None
                    and not (0.0 <= measured <= 1.0)
                ):
                    raise ValueError("native checkpoint accuracy is outside [0, 1]")
            parsed_metrics[metric_name] = NativeQualityMetric(
                value=measured,
                threshold=threshold,
                direction=direction,
                passed=bool(metric["passed"]),
            )
        computed_evidence_pass = (
            stage is not None
            and evaluated_steps == optimizer_steps
            and records >= minimum_records
            and all(metric.passed for metric in parsed_metrics.values())
        )
        measurements_present = any(
            metric.value is not None for metric in parsed_metrics.values()
        )
        if evaluated_steps is None:
            if records != 0 or measurements_present:
                raise ValueError(
                    f"native checkpoint {gate_name} unevaluated quality contains measurements"
                )
        elif stage is None or records <= 0:
            raise ValueError(
                f"native checkpoint {gate_name} evaluated quality lacks a holdout"
            )
        if not isinstance(evidence["passed"], bool) or (
            evidence["passed"] is not computed_evidence_pass
        ):
            raise ValueError(
                f"native checkpoint {gate_name} quality pass flag is inconsistent"
            )
        if gates[gate_name].validation_passed is not (
            gates[gate_name].trained and computed_evidence_pass
        ):
            raise ValueError(
                f"native checkpoint {gate_name} gate is not bound to held-out quality"
            )
        quality[gate_name] = NativeGateQuality(
            stage=stage,
            evaluated_optimizer_steps=evaluated_steps,
            validation_records=records,
            minimum_validation_records=minimum_records,
            validation_manifest_sha256=validation_digest,
            metrics=MappingProxyType(parsed_metrics),
            passed=computed_evidence_pass,
        )

    sample_rate = _positive_integer(runtime["sample_rate"], "sample rate")
    frame_size = _positive_integer(runtime["frame_size"], "frame size")
    video_width = _positive_integer(runtime["video_width"], "video width")
    video_height = _positive_integer(runtime["video_height"], "video height")
    if sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError("native checkpoint sample rate must be 24 kHz")
    if (video_width, video_height) != (NATIVE_VIDEO_WIDTH, NATIVE_VIDEO_HEIGHT):
        raise ValueError("native checkpoint video shape must be 320x192")

    config = MultimodalConfig.from_json(config_path)
    if config.audio.sample_rate != sample_rate or config.audio.frame_size != frame_size:
        raise ValueError("native runtime audio contract does not match model config")
    divisor = config.video.patch_size * config.video.spatial_merge_size
    if video_width % divisor or video_height % divisor:
        raise ValueError("native runtime video shape does not match model patch grid")
    manifest = NativeCheckpointManifest(
        schema_version=NATIVE_CHECKPOINT_SCHEMA_VERSION,
        format=NATIVE_CHECKPOINT_FORMAT,
        checkpoint_id=checkpoint_id,
        scratch_only=True,
        external_weights=False,
        config_name=str(model["config"]),
        weights_name=str(model["weights"]),
        config_sha256=str(model["config_sha256"]),
        weights_sha256=str(model["weights_sha256"]),
        optimizer_steps=optimizer_steps,
        gates=MappingProxyType(gates),
        quality=MappingProxyType(quality),
        sample_rate=sample_rate,
        frame_size=frame_size,
        video_width=video_width,
        video_height=video_height,
    )
    return LoadedNativeCheckpoint(
        manifest=manifest,
        manifest_path=manifest_path,
        config_path=config_path,
        weights_path=weights_path,
        config=config,
    )


@dataclass(frozen=True, slots=True)
class NativeVideoFrame:
    frame_id: str
    sequence: int
    source_timestamp_ms: int
    session_time_seconds: float
    difference_score: float
    keyframe: bool
    rgb: np.ndarray = field(repr=False)


def _jpeg_dimensions(jpeg: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions before asking FFmpeg to allocate a frame."""

    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    offset = 2
    while offset < len(jpeg):
        if jpeg[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(jpeg) and jpeg[offset] == 0xFF:
            offset += 1
        if offset >= len(jpeg):
            break
        marker = jpeg[offset]
        offset += 1
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if marker in {0xD9, 0xDA} or offset + 2 > len(jpeg):
            break
        segment_length = int.from_bytes(jpeg[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(jpeg):
            raise ValueError("selected frame JPEG has an invalid segment")
        if marker in start_of_frame:
            if segment_length < 7:
                raise ValueError("selected frame JPEG has an invalid SOF segment")
            height = int.from_bytes(jpeg[offset + 3 : offset + 5], "big")
            width = int.from_bytes(jpeg[offset + 5 : offset + 7], "big")
            return width, height
        offset += segment_length
    raise ValueError("selected frame JPEG has no image dimensions")


def canonicalize_selected_frame(frame: SelectedFrame) -> NativeVideoFrame:
    """Decode a selected JPEG to normalized, canonical 320x192 RGB."""

    if not isinstance(frame, SelectedFrame):
        raise TypeError("selected video input must be a SelectedFrame")
    if (
        not isinstance(frame.sequence, int)
        or isinstance(frame.sequence, bool)
        or frame.sequence < 0
        or not isinstance(frame.timestamp_ms, int)
        or isinstance(frame.timestamp_ms, bool)
        or frame.timestamp_ms < 0
        or not math.isfinite(frame.difference_percent)
        or not 0.0 <= frame.difference_percent <= 100.0
        or frame.content_type != "image/jpeg"
        or not isinstance(frame.jpeg, bytes)
        or not frame.jpeg
        or len(frame.jpeg) > MAX_SELECTED_FRAME_BYTES
    ):
        raise ValueError("selected frame metadata is invalid")
    if _SAFE_ID.fullmatch(frame.frame_id) is None:
        raise ValueError("selected frame ID is invalid")
    if not frame.jpeg.startswith(b"\xff\xd8") or not frame.jpeg.endswith(b"\xff\xd9"):
        raise ValueError("selected frame is not a complete JPEG")
    source_width, source_height = _jpeg_dimensions(frame.jpeg)
    if (
        source_width <= 0
        or source_height <= 0
        or source_width * source_height > MAX_DECODED_VIDEO_PIXELS
    ):
        raise ValueError("selected frame JPEG dimensions are too large")
    try:
        with av.open(io.BytesIO(frame.jpeg), mode="r", format="mjpeg") as container:
            decoded = list(container.decode(video=0))
    except (av.error.FFmpegError, EOFError, OSError, ValueError) as error:
        raise ValueError("selected frame JPEG could not be decoded") from error
    if len(decoded) != 1:
        raise ValueError("selected frame JPEG must contain exactly one image")
    source = decoded[0]
    if (source.width, source.height) != (source_width, source_height):
        raise ValueError("selected frame JPEG dimensions are inconsistent")
    resized = source.reformat(
        width=NATIVE_VIDEO_WIDTH,
        height=NATIVE_VIDEO_HEIGHT,
        format="rgb24",
    )
    rgb = np.asarray(resized.to_ndarray(), dtype=np.uint8)
    if rgb.shape != (NATIVE_VIDEO_HEIGHT, NATIVE_VIDEO_WIDTH, 3):
        raise ValueError("selected frame JPEG produced an invalid RGB shape")
    normalized = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
    normalized.setflags(write=False)
    return NativeVideoFrame(
        frame_id=frame.frame_id,
        sequence=frame.sequence,
        source_timestamp_ms=frame.timestamp_ms,
        session_time_seconds=0.0,
        difference_score=frame.difference_percent / 100.0,
        keyframe=frame.keyframe,
        rgb=normalized,
    )


@dataclass(frozen=True, slots=True)
class NativeAudioOutput:
    samples: np.ndarray = field(repr=False)
    generation: int
    sequence: int
    produced_at: float


@dataclass(frozen=True, slots=True)
class NativeBackendResult:
    accepted: bool
    audio: tuple[np.ndarray, ...] = ()


@runtime_checkable
class NativeStreamingBackend(Protocol):
    """One session's cached native-model state over immutable shared weights."""

    @property
    def model_loaded(self) -> bool: ...

    @property
    def random_init(self) -> bool: ...

    @property
    def parameter_count(self) -> int: ...

    @property
    def frame_size(self) -> int: ...

    async def accept_audio(self, block: PcmBlock) -> NativeBackendResult: ...

    async def accept_frame(self, frame: NativeVideoFrame) -> NativeBackendResult: ...

    async def interrupt(self, generation: int) -> None: ...

    async def close(self) -> None: ...

    def cache_status(self) -> Mapping[str, int]: ...


class _ObservationBackend:
    """Lightweight media accounting for the honest no-model default."""

    def __init__(self, frame_size: int) -> None:
        self._frame_size = frame_size
        self._tail = np.zeros((0,), dtype=np.float32)
        self._last_frame_sequence = -1
        self._audio_samples = 0

    model_loaded = False
    random_init = False
    parameter_count = 0

    @property
    def frame_size(self) -> int:
        return self._frame_size

    async def accept_audio(self, block: PcmBlock) -> NativeBackendResult:
        self._audio_samples += int(block.samples.size)
        combined = np.concatenate((self._tail, block.samples))
        remainder = combined.size % self._frame_size
        self._tail = combined[-remainder:].copy() if remainder else combined[:0].copy()
        return NativeBackendResult(accepted=False)

    async def accept_frame(self, frame: NativeVideoFrame) -> NativeBackendResult:
        self._last_frame_sequence = frame.sequence
        return NativeBackendResult(accepted=False)

    async def interrupt(self, generation: int) -> None:
        del generation
        self._tail = np.zeros((0,), dtype=np.float32)

    async def close(self) -> None:
        self._tail = np.zeros((0,), dtype=np.float32)

    def cache_status(self) -> Mapping[str, int]:
        return {
            "audio_tail_samples": int(self._tail.size),
            "audio_samples_seen": self._audio_samples,
            "last_video_sequence": self._last_frame_sequence,
        }


class _SharedMlxModel:
    def __init__(self, checkpoint: LoadedNativeCheckpoint) -> None:
        model = create_scratch_multimodal_model(checkpoint.config)
        model.load_weights(str(checkpoint.weights_path), strict=True)
        model.eval()
        mx.eval(model.parameters())
        self.model = model
        self.lock = asyncio.Lock()
        self.parameter_count = model.parameter_count()
        self.random_init = False

    def create_backend(self) -> "_MlxSessionBackend":
        return _MlxSessionBackend(self)


class _SharedRandomMlxModel:
    """One process-owned scratch model with no checkpoint or imported weights."""

    def __init__(self, config: MultimodalConfig, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("random native model seed must be a non-negative integer")
        mx.random.seed(seed)
        model = create_scratch_multimodal_model(config)
        model.eval()
        mx.eval(model.parameters())
        self.model = model
        self.lock = asyncio.Lock()
        self.parameter_count = model.parameter_count()
        self.random_init = True

    def create_backend(self) -> "_MlxSessionBackend":
        return _MlxSessionBackend(self)


class _MlxSessionBackend:
    """Incremental MLX inference with session-private causal caches."""

    model_loaded = True

    def __init__(self, shared: _SharedMlxModel | _SharedRandomMlxModel) -> None:
        self.shared = shared
        self.model = shared.model
        self.random_init = shared.random_init
        self._audio_encoder = self.model.audio_codec.init_stream_state(1)
        self._audio_decoder = self.model.audio_codec.init_stream_state(1)
        self._thinker = self.model.init_thinker_cache(1)
        self._talker = self.model.talker.init_stream_state(1)
        self._audio_frames = 0
        self._video_frames = 0
        self._generation = 0
        self._audio_started = False
        self._last_hidden: mx.array | None = None
        self._last_timeline_time = 0.0

    @property
    def parameter_count(self) -> int:
        return self.shared.parameter_count

    @property
    def frame_size(self) -> int:
        return self.model.config.audio.frame_size

    async def accept_audio(self, block: PcmBlock) -> NativeBackendResult:
        async with self.shared.lock:
            waveform = mx.array(block.samples[None, :, None], dtype=mx.float32)
            encoding = self.model.audio_codec.encode_stream(
                waveform, self._audio_encoder, final=False
            )
            frame_count = int(encoding.codes.shape[1])
            if frame_count == 0:
                return NativeBackendResult(accepted=True)
            if not self._audio_started:
                marker = self.model.thinker.token_embedding(
                    mx.array([[self.model.registry.audio_user_start]], dtype=mx.int32)
                )
                marker_hidden = self._think_embeddings(
                    marker,
                    mx.array([[block.captured_at]], dtype=mx.float32),
                )
                self._last_hidden = marker_hidden[:, -1:, :]
                self._audio_started = True
            quantized = encoding.quantized
            stream_ids = mx.zeros((1, frame_count), dtype=mx.int32)
            raw_audio = self.model.audio_projection(
                quantized
            ) + self.model.audio_stream_embedding(stream_ids)
            frame_duration = self.frame_size / MODEL_SAMPLE_RATE
            start_time = max(block.captured_at, self._last_timeline_time)
            timestamps = mx.array(
                [[start_time + frame * frame_duration for frame in range(frame_count)]],
                dtype=mx.float32,
            )
            hidden = self._think_embeddings(raw_audio, timestamps)
            assert self._last_hidden is not None
            conditioning = mx.concatenate(
                (self._last_hidden, hidden[:, :-1, :]), axis=1
            )
            self._last_hidden = hidden[:, -1:, :]
            generated, self._talker = self.model.talker.generate_stream(
                conditioning, encoding.codes, self._talker
            )
            decoded = self.model.audio_codec.decode_codes_stream(
                generated.codes, self._audio_decoder
            )
            mx.eval(decoded)
            pcm = np.asarray(decoded[0, :, 0], dtype=np.float32).copy()
            self._audio_frames += frame_count
            return NativeBackendResult(accepted=True, audio=(pcm,))

    def _think_embeddings(
        self, raw_embeddings: mx.array, timestamps: mx.array
    ) -> mx.array:
        if timestamps.shape != raw_embeddings.shape[:2]:
            raise ValueError("native streaming timestamps do not match embeddings")
        first_delta = mx.maximum(
            timestamps[:, :1] - self._last_timeline_time,
            mx.zeros((1, 1), dtype=timestamps.dtype),
        )
        deltas = (
            first_delta
            if timestamps.shape[1] == 1
            else mx.concatenate(
                (first_delta, timestamps[:, 1:] - timestamps[:, :-1]), axis=1
            )
        )
        features = mx.stack(
            (timestamps / self.model.config.time_scale_seconds, deltas), axis=-1
        )
        embeddings = raw_embeddings + self.model.time_output(
            nn.silu(self.model.time_input(features))
        )
        hidden, _, _ = self.model._think_from_embeddings(embeddings, self._thinker)
        self._last_timeline_time = max(
            self._last_timeline_time, float(timestamps[0, -1].item())
        )
        return hidden

    async def accept_frame(self, frame: NativeVideoFrame) -> NativeBackendResult:
        async with self.shared.lock:
            frame_time = max(frame.session_time_seconds, self._last_timeline_time)
            events = [
                TimedVideo(
                    frames=mx.array(frame.rgb[None, None, :, :, :]),
                    timestamps=mx.array([frame_time], dtype=mx.float32),
                    difference_scores=mx.array(
                        [frame.difference_score], dtype=mx.float32
                    ),
                )
            ]
            output = self.model.forward_timeline(events, thinker_cache=self._thinker)
            mx.eval(output.hidden)
            self._last_hidden = output.hidden[:, -1:, :]
            self._last_timeline_time = max(self._last_timeline_time, frame_time)
            self._video_frames += 1
            return NativeBackendResult(accepted=True)

    async def interrupt(self, generation: int) -> None:
        async with self.shared.lock:
            self.model.talker.request_interruption(self._talker)
            # Generation-private synthesis caches must never leak old PCM into
            # the next response.  Perception and thinker context remain live.
            self._audio_decoder = self.model.audio_codec.init_stream_state(1)
            self._talker = self.model.talker.init_stream_state(1)
            self._generation = generation

    async def close(self) -> None:
        return None

    def cache_status(self) -> Mapping[str, int]:
        return {
            "audio_encoder_buffer_samples": int(
                self._audio_encoder.encoder_buffer.shape[1]
            ),
            "audio_frames_encoded": self._audio_frames,
            "audio_frames_decoded": int(self._audio_decoder.frames_decoded),
            "thinker_offset": int(self._thinker.offset),
            "thinker_resident_tokens": int(self._thinker.resident_tokens),
            "talker_offset": int(self._talker.offset),
            "video_frames_encoded": self._video_frames,
            "backend_generation": self._generation,
        }


@dataclass(slots=True)
class _PendingInput:
    kind: str
    payload: PcmBlock | NativeVideoFrame
    generation: int
    result: asyncio.Future[bool]


@dataclass(frozen=True, slots=True)
class NativeRuntimeStatus:
    model_loaded: bool
    model_connected: bool
    checkpoint_loaded: bool
    random_init: bool
    untrained_audio_enabled: bool
    voice_trained: bool
    vision_trained: bool
    conversation_ready: bool
    can_emit_audio: bool
    checkpoint_id: str | None
    parameters: int
    sample_rate: int
    codec_frame_size: int
    generation: int
    input_capacity: int
    input_depth: int
    output_capacity: int
    output_depth: int
    dropped_inputs: int
    dropped_outputs: int
    stale_outputs: int
    suppressed_outputs: int
    processing_errors: int
    audio_blocks_received: int
    video_frames_received: int
    closed: bool
    reason: str
    cache: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_loaded": self.model_loaded,
            "model_connected": self.model_connected,
            "checkpoint_loaded": self.checkpoint_loaded,
            "random_init": self.random_init,
            "untrained_audio_enabled": self.untrained_audio_enabled,
            "voice_trained": self.voice_trained,
            "vision_trained": self.vision_trained,
            "conversation_ready": self.conversation_ready,
            "can_emit_audio": self.can_emit_audio,
            "checkpoint_id": self.checkpoint_id,
            "parameters": self.parameters,
            "sample_rate": self.sample_rate,
            "codec_frame_size": self.codec_frame_size,
            "generation": self.generation,
            "input_capacity": self.input_capacity,
            "input_depth": self.input_depth,
            "output_capacity": self.output_capacity,
            "output_depth": self.output_depth,
            "dropped_inputs": self.dropped_inputs,
            "dropped_outputs": self.dropped_outputs,
            "stale_outputs": self.stale_outputs,
            "suppressed_outputs": self.suppressed_outputs,
            "processing_errors": self.processing_errors,
            "audio_blocks_received": self.audio_blocks_received,
            "video_frames_received": self.video_frames_received,
            "closed": self.closed,
            "reason": self.reason,
            "cache": dict(self.cache),
        }


class ScratchNativeRuntime:
    """Bounded, generation-fenced runtime for exactly one conversation."""

    def __init__(
        self,
        backend: NativeStreamingBackend,
        *,
        checkpoint: LoadedNativeCheckpoint | None = None,
        allow_untrained_audio: bool = False,
        input_capacity: int = DEFAULT_INPUT_CAPACITY,
        output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if input_capacity <= 0 or output_capacity <= 0:
            raise ValueError("native runtime queue capacities must be positive")
        if not isinstance(backend, NativeStreamingBackend):
            raise TypeError("backend does not implement NativeStreamingBackend")
        if not isinstance(allow_untrained_audio, bool):
            raise TypeError("allow_untrained_audio must be bool")
        if allow_untrained_audio and not backend.random_init:
            raise ValueError(
                "untrained audio may be enabled only for a random-initialized model"
            )
        self._backend = backend
        self._checkpoint = checkpoint
        self._allow_untrained_audio = allow_untrained_audio
        self.input_capacity = input_capacity
        self.output_capacity = output_capacity
        self._clock = clock
        self._session_origin = float(clock())
        if not math.isfinite(self._session_origin):
            raise ValueError("native runtime clock is not finite")
        self._last_session_time = 0.0
        self._pending: deque[_PendingInput] = deque()
        self._pending_event = asyncio.Event()
        self._output: asyncio.Queue[NativeAudioOutput | None] = asyncio.Queue(
            maxsize=output_capacity
        )
        self._operation_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._generation = 0
        self._output_sequence = 0
        self._last_audio_sequence = -1
        self._last_frame_sequence = -1
        self._last_produced_at = -math.inf
        self._output_tail = np.zeros((0,), dtype=np.float32)
        self._dropped_inputs = 0
        self._dropped_outputs = 0
        self._stale_outputs = 0
        self._suppressed_outputs = 0
        self._processing_errors = 0
        self._audio_received = 0
        self._video_received = 0
        self._untrained_turn_audio: deque[np.ndarray] = deque()
        self._untrained_turn_samples = 0
        self._untrained_release_task: asyncio.Task[None] | None = None
        self._untrained_speech_active = False
        self._untrained_preroll: deque[PcmBlock] = deque()
        self._untrained_preroll_samples = 0

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        input_capacity: int = DEFAULT_INPUT_CAPACITY,
        output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
    ) -> "ScratchNativeRuntime":
        """Single-session convenience loader.

        Gateways should use :meth:`NativeRuntimeFactory.from_checkpoint` once
        at startup so reconnects do not reload a 300M model.
        """

        factory = NativeRuntimeFactory.from_checkpoint(
            path,
            input_capacity=input_capacity,
            output_capacity=output_capacity,
        )
        return factory.create()

    @property
    def generation(self) -> int:
        return self._generation

    def _voice_trained(self) -> bool:
        return self._checkpoint is not None and self._checkpoint.manifest.voice_trained

    def _vision_trained(self) -> bool:
        return self._checkpoint is not None and self._checkpoint.manifest.vision_trained

    def _conversation_ready(self) -> bool:
        return (
            self._backend.model_loaded
            and self._checkpoint is not None
            and self._checkpoint.manifest.conversation_ready
        )

    def _can_emit_audio(self) -> bool:
        return self._backend.model_loaded and (
            self._voice_trained()
            or (self._backend.random_init and self._allow_untrained_audio)
        )

    def snapshot(self) -> NativeRuntimeStatus:
        checkpoint_id = (
            self._checkpoint.manifest.checkpoint_id
            if self._checkpoint is not None
            else None
        )
        if not self._backend.model_loaded:
            reason = "No native model checkpoint is loaded."
        elif self._backend.random_init and self._allow_untrained_audio:
            reason = (
                "Untrained random-model audio diagnostic is enabled; output is "
                "expected to be meaningless."
            )
        elif self._checkpoint is None:
            reason = "Random initialization is not trained and audio is disabled."
        elif not self._voice_trained():
            reason = "Native voice training gates have not passed."
        elif not self._vision_trained():
            reason = "Native voice is gated on, but the vision gate has not passed."
        else:
            reason = "Native audio-video checkpoint gates passed."
        return NativeRuntimeStatus(
            model_loaded=self._backend.model_loaded,
            model_connected=self._backend.model_loaded,
            checkpoint_loaded=self._checkpoint is not None,
            random_init=self._backend.random_init,
            untrained_audio_enabled=(
                self._backend.random_init and self._allow_untrained_audio
            ),
            voice_trained=self._voice_trained(),
            vision_trained=self._vision_trained(),
            conversation_ready=self._conversation_ready(),
            can_emit_audio=self._can_emit_audio(),
            checkpoint_id=checkpoint_id,
            parameters=self._backend.parameter_count,
            sample_rate=MODEL_SAMPLE_RATE,
            codec_frame_size=self._backend.frame_size,
            generation=self._generation,
            input_capacity=self.input_capacity,
            input_depth=len(self._pending),
            output_capacity=self.output_capacity,
            output_depth=self._output.qsize(),
            dropped_inputs=self._dropped_inputs,
            dropped_outputs=self._dropped_outputs,
            stale_outputs=self._stale_outputs,
            suppressed_outputs=self._suppressed_outputs,
            processing_errors=self._processing_errors,
            audio_blocks_received=self._audio_received,
            video_frames_received=self._video_received,
            closed=self._closed,
            reason=reason,
            cache=MappingProxyType(dict(self._backend.cache_status())),
        )

    def status(self) -> dict[str, object]:
        return self.snapshot().to_dict()

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._run_inputs(), name="mlm-native-runtime-input"
            )

    def _drop_oldest_kind(self, kind: str) -> None:
        if len(self._pending) < self.input_capacity:
            return
        item = next(
            (candidate for candidate in self._pending if candidate.kind == kind),
            self._pending[0],
        )
        self._pending.remove(item)
        if not item.result.done():
            item.result.set_result(False)
        self._dropped_inputs += 1

    async def _enqueue(self, kind: str, payload: PcmBlock | NativeVideoFrame) -> bool:
        if self._closed:
            raise NativeRuntimeClosed("native runtime session is closed")
        loop = asyncio.get_running_loop()
        result: asyncio.Future[bool] = loop.create_future()
        self._drop_oldest_kind(kind)
        self._pending.append(
            _PendingInput(
                kind=kind,
                payload=payload,
                generation=self._generation,
                result=result,
            )
        )
        self._ensure_worker()
        self._pending_event.set()
        return await result

    @staticmethod
    def _validated_audio(block: PcmBlock) -> PcmBlock:
        if not isinstance(block, PcmBlock):
            raise TypeError("native audio input must be a PcmBlock")
        if (
            not isinstance(block.sequence, int)
            or isinstance(block.sequence, bool)
            or block.sequence < 0
            or not math.isfinite(block.captured_at)
            or not math.isfinite(block.received_at)
            or block.captured_at < 0.0
            or block.received_at < block.captured_at
        ):
            raise ValueError("native PCM metadata is invalid")
        samples = np.asarray(block.samples, dtype=np.float32)
        if (
            samples.ndim != 1
            or samples.size <= 0
            or samples.size > MAX_AUDIO_BLOCK_SAMPLES
            or not np.isfinite(samples).all()
        ):
            raise ValueError("native PCM must be finite bounded mono audio")
        samples = np.ascontiguousarray(np.clip(samples, -1.0, 1.0))
        samples.setflags(write=False)
        return PcmBlock(
            sequence=block.sequence,
            captured_at=float(block.captured_at),
            received_at=float(block.received_at),
            samples=samples,
        )

    async def accept_audio(self, block: PcmBlock) -> bool:
        validated = self._validated_audio(block)
        if validated.sequence <= self._last_audio_sequence:
            raise ValueError("native PCM sequence must increase")
        self._last_audio_sequence = validated.sequence
        self._audio_received += 1
        session_time = self._relative_session_time()
        normalized_time = PcmBlock(
            sequence=validated.sequence,
            captured_at=session_time,
            received_at=session_time,
            samples=validated.samples,
        )
        if (
            self._backend.random_init
            and self._allow_untrained_audio
            and not self._untrained_speech_active
        ):
            maximum = round(UNTRAINED_INPUT_PREROLL_SECONDS * MODEL_SAMPLE_RATE)
            self._untrained_preroll.append(normalized_time)
            self._untrained_preroll_samples += int(normalized_time.samples.size)
            while self._untrained_preroll and self._untrained_preroll_samples > maximum:
                removed = self._untrained_preroll.popleft()
                self._untrained_preroll_samples -= int(removed.samples.size)
            return False
        return await self._enqueue("audio", normalized_time)

    async def accept_frame(self, frame: SelectedFrame) -> bool:
        canonical = canonicalize_selected_frame(frame)
        if canonical.sequence <= self._last_frame_sequence:
            raise ValueError("native video sequence must increase")
        self._last_frame_sequence = canonical.sequence
        self._video_received += 1
        canonical = replace(
            canonical, session_time_seconds=self._relative_session_time()
        )
        return await self._enqueue("video", canonical)

    def _relative_session_time(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise NativeRuntimeError("native runtime clock is not finite")
        relative = min(max(0.0, now - self._session_origin), MAX_SESSION_TIME_SECONDS)
        relative = max(relative, self._last_session_time)
        self._last_session_time = relative
        return relative

    async def _run_inputs(self) -> None:
        try:
            while not self._closed:
                await self._pending_event.wait()
                while self._pending and not self._closed:
                    item = self._pending.popleft()
                    if item.result.cancelled():
                        continue
                    try:
                        async with self._operation_lock:
                            if item.kind == "audio":
                                assert isinstance(item.payload, PcmBlock)
                                backend_result = await self._backend.accept_audio(
                                    item.payload
                                )
                            else:
                                assert isinstance(item.payload, NativeVideoFrame)
                                backend_result = await self._backend.accept_frame(
                                    item.payload
                                )
                        if self._backend.random_init and self._allow_untrained_audio:
                            self._buffer_untrained_audio(backend_result.audio)
                        else:
                            self._publish_audio(
                                backend_result.audio, generation=item.generation
                            )
                        if not item.result.done():
                            item.result.set_result(backend_result.accepted)
                    except asyncio.CancelledError:
                        if not item.result.done():
                            item.result.set_result(False)
                        raise
                    except Exception as error:
                        self._processing_errors += 1
                        if not item.result.done():
                            item.result.set_exception(error)
                self._pending_event.clear()
        except asyncio.CancelledError:
            return

    def _buffer_untrained_audio(self, chunks: Sequence[np.ndarray]) -> None:
        maximum = round(UNTRAINED_RESPONSE_MAX_SECONDS * MODEL_SAMPLE_RATE)
        for chunk in chunks:
            value = np.asarray(chunk, dtype=np.float32)
            if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
                raise NativeRuntimeError("native backend produced invalid PCM")
            owned = np.ascontiguousarray(np.clip(value, -1.0, 1.0))
            owned.setflags(write=False)
            self._untrained_turn_audio.append(owned)
            self._untrained_turn_samples += int(owned.size)
        while self._untrained_turn_audio and self._untrained_turn_samples > maximum:
            removed = self._untrained_turn_audio.popleft()
            self._untrained_turn_samples -= int(removed.size)

    async def end_turn(self, turn_id: str | None = None) -> bool:
        """Release random diagnostic audio only after the external VAD endpoint."""

        del turn_id
        if self._closed:
            raise NativeRuntimeClosed("native runtime session is closed")
        if not (self._backend.random_init and self._allow_untrained_audio):
            return False
        self._untrained_speech_active = False
        chunks = tuple(self._untrained_turn_audio)
        self._untrained_turn_audio.clear()
        self._untrained_turn_samples = 0
        if not chunks:
            return False
        task = self._untrained_release_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._untrained_release_task = asyncio.create_task(
            self._release_untrained_audio(chunks, self._generation),
            name="mlm-native-random-turn-output",
        )
        return True

    async def _release_untrained_audio(
        self, chunks: Sequence[np.ndarray], generation: int
    ) -> None:
        try:
            combined = np.concatenate(tuple(chunks))
            usable = combined.size - (combined.size % MODEL_FRAME_SAMPLES)
            frame_count = usable // MODEL_FRAME_SAMPLES
            for frame_index, start in enumerate(range(0, usable, MODEL_FRAME_SAMPLES)):
                if self._closed or generation != self._generation:
                    return
                samples = np.ascontiguousarray(
                    combined[start : start + MODEL_FRAME_SAMPLES]
                )
                samples.setflags(write=False)
                now = float(self._clock())
                if not math.isfinite(now):
                    raise NativeRuntimeError("native runtime clock is not finite")
                if now <= self._last_produced_at:
                    now = math.nextafter(self._last_produced_at, math.inf)
                self._last_produced_at = now
                output = NativeAudioOutput(
                    samples=samples,
                    generation=generation,
                    sequence=self._output_sequence,
                    produced_at=now,
                )
                self._output_sequence += 1
                await self._output.put(output)
                if frame_index + 1 < frame_count:
                    await asyncio.sleep(MODEL_FRAME_SAMPLES / MODEL_SAMPLE_RATE)
        except asyncio.CancelledError:
            return

    def _publish_audio(self, chunks: Sequence[np.ndarray], *, generation: int) -> None:
        if not chunks:
            return
        if generation != self._generation:
            self._stale_outputs += len(chunks)
            return
        if not self._can_emit_audio():
            self._suppressed_outputs += len(chunks)
            return
        values = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        if any(
            value.ndim != 1 or value.size == 0 or not np.isfinite(value).all()
            for value in values
        ):
            raise NativeRuntimeError("native backend produced invalid PCM")
        combined = np.concatenate((self._output_tail, *values))
        usable = combined.size - (combined.size % MODEL_FRAME_SAMPLES)
        self._output_tail = combined[usable:].copy()
        for start in range(0, usable, MODEL_FRAME_SAMPLES):
            samples = np.ascontiguousarray(
                np.clip(combined[start : start + MODEL_FRAME_SAMPLES], -1.0, 1.0)
            )
            samples.setflags(write=False)
            now = float(self._clock())
            if not math.isfinite(now):
                raise NativeRuntimeError("native runtime clock is not finite")
            if now <= self._last_produced_at:
                now = math.nextafter(self._last_produced_at, math.inf)
            self._last_produced_at = now
            output = NativeAudioOutput(
                samples=samples,
                generation=generation,
                sequence=self._output_sequence,
                produced_at=now,
            )
            self._output_sequence += 1
            while self._output.full():
                dropped = self._output.get_nowait()
                if dropped is not None:
                    self._dropped_outputs += 1
            self._output.put_nowait(output)

    async def next_output(self) -> NativeAudioOutput:
        if self._closed and self._output.empty():
            raise NativeRuntimeClosed("native runtime session is closed")
        output = await self._output.get()
        if output is None:
            raise NativeRuntimeClosed("native runtime session is closed")
        return output

    def _clear_output(self) -> None:
        while not self._output.empty():
            item = self._output.get_nowait()
            if item is not None:
                self._dropped_outputs += 1
        self._output_tail = np.zeros((0,), dtype=np.float32)

    def _clear_pending(self) -> None:
        while self._pending:
            item = self._pending.popleft()
            if not item.result.done():
                item.result.set_result(False)
            self._dropped_inputs += 1

    async def interrupt(self, generation: int) -> int:
        if self._closed:
            raise NativeRuntimeClosed("native runtime session is closed")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation != self._generation + 1
        ):
            raise ValueError("native interruption generation must increase by one")
        # Advance the fence before waiting for an in-flight MLX operation.
        self._generation = generation
        preroll = tuple(self._untrained_preroll)
        self._untrained_preroll.clear()
        self._untrained_preroll_samples = 0
        self._untrained_speech_active = (
            self._backend.random_init and self._allow_untrained_audio
        )
        self._untrained_turn_audio.clear()
        self._untrained_turn_samples = 0
        release_task = self._untrained_release_task
        if release_task is not None and not release_task.done():
            release_task.cancel()
            await asyncio.gather(release_task, return_exceptions=True)
        self._clear_output()
        self._clear_pending()
        async with self._operation_lock:
            await self._backend.interrupt(generation)
            for block in preroll:
                backend_result = await self._backend.accept_audio(block)
                self._buffer_untrained_audio(backend_result.audio)
        return self._generation

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        release_task = self._untrained_release_task
        if release_task is not None and not release_task.done():
            release_task.cancel()
            await asyncio.gather(release_task, return_exceptions=True)
        self._untrained_turn_audio.clear()
        self._untrained_turn_samples = 0
        self._untrained_speech_active = False
        self._untrained_preroll.clear()
        self._untrained_preroll_samples = 0
        self._clear_pending()
        self._clear_output()
        self._pending_event.set()
        worker = self._worker
        if worker is not None and worker is not asyncio.current_task():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        async with self._operation_lock:
            await self._backend.close()
        if self._output.full():
            self._output.get_nowait()
        self._output.put_nowait(None)


BackendBuilder = Callable[[], NativeStreamingBackend]


class NativeRuntimeFactory:
    """Process-level shared model plus session-level runtime factory."""

    def __init__(
        self,
        builder: BackendBuilder,
        *,
        checkpoint: LoadedNativeCheckpoint | None,
        allow_untrained_audio: bool = False,
        parameter_count: int = 0,
        input_capacity: int = DEFAULT_INPUT_CAPACITY,
        output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
    ) -> None:
        if (
            isinstance(parameter_count, bool)
            or not isinstance(parameter_count, int)
            or parameter_count < 0
        ):
            raise ValueError("native runtime parameter_count must be non-negative")
        self._builder = builder
        self.checkpoint = checkpoint
        self.allow_untrained_audio = allow_untrained_audio
        self.parameter_count = parameter_count
        self.input_capacity = input_capacity
        self.output_capacity = output_capacity

    @classmethod
    def untrained(
        cls,
        *,
        config: MultimodalConfig | None = None,
        input_capacity: int = DEFAULT_INPUT_CAPACITY,
        output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
    ) -> "NativeRuntimeFactory":
        resolved = config or MultimodalConfig()
        return cls(
            lambda: _ObservationBackend(resolved.audio.frame_size),
            checkpoint=None,
            allow_untrained_audio=False,
            parameter_count=0,
            input_capacity=input_capacity,
            output_capacity=output_capacity,
        )

    @classmethod
    def from_random_init(
        cls,
        config: MultimodalConfig,
        *,
        seed: int = 0,
        input_capacity: int = DEFAULT_INPUT_CAPACITY,
        output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
    ) -> "NativeRuntimeFactory":
        """Instantiate MLM's own model at random init for audible diagnostics."""

        shared = _SharedRandomMlxModel(config, seed=seed)
        return cls(
            shared.create_backend,
            checkpoint=None,
            allow_untrained_audio=True,
            parameter_count=shared.parameter_count,
            input_capacity=input_capacity,
            output_capacity=output_capacity,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        input_capacity: int = DEFAULT_INPUT_CAPACITY,
        output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
    ) -> "NativeRuntimeFactory":
        checkpoint = load_native_checkpoint(path)
        shared = _SharedMlxModel(checkpoint)
        return cls(
            shared.create_backend,
            checkpoint=checkpoint,
            allow_untrained_audio=False,
            parameter_count=shared.parameter_count,
            input_capacity=input_capacity,
            output_capacity=output_capacity,
        )

    def create(
        self,
        session_id: str | None = None,
        connection_id: str | None = None,
    ) -> ScratchNativeRuntime:
        # IDs are intentionally not retained: runtime status must not expose
        # user/session identifiers or media-derived content.
        del session_id, connection_id
        return ScratchNativeRuntime(
            self._builder(),
            checkpoint=self.checkpoint,
            allow_untrained_audio=self.allow_untrained_audio,
            input_capacity=self.input_capacity,
            output_capacity=self.output_capacity,
        )

    def __call__(self, session_id: str, connection_id: str) -> ScratchNativeRuntime:
        return self.create(session_id, connection_id)


def create_untrained_runtime(
    *,
    config: MultimodalConfig | None = None,
    input_capacity: int = DEFAULT_INPUT_CAPACITY,
    output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
) -> ScratchNativeRuntime:
    """Create a lightweight, no-model, no-audio-output session."""

    return NativeRuntimeFactory.untrained(
        config=config,
        input_capacity=input_capacity,
        output_capacity=output_capacity,
    ).create()


def load_native_runtime_factory(
    path: str | Path,
    *,
    input_capacity: int = DEFAULT_INPUT_CAPACITY,
    output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
) -> NativeRuntimeFactory:
    """Load shared weights once for use as a gateway ``runtime_factory``."""

    return NativeRuntimeFactory.from_checkpoint(
        path,
        input_capacity=input_capacity,
        output_capacity=output_capacity,
    )


def load_random_native_runtime_factory(
    config_path: str | Path,
    *,
    seed: int = 0,
    input_capacity: int = DEFAULT_INPUT_CAPACITY,
    output_capacity: int = DEFAULT_OUTPUT_CAPACITY,
) -> NativeRuntimeFactory:
    """Load only a local architecture JSON and randomly initialize every weight."""

    resolved = Path(config_path).expanduser().resolve(strict=True)
    if resolved.suffix != ".json" or not resolved.is_file():
        raise ValueError("random native configuration must be a local JSON file")
    payload = _read_json_object(resolved, maximum_bytes=MAX_NATIVE_CONFIG_BYTES)
    config = MultimodalConfig.from_dict(payload)
    return NativeRuntimeFactory.from_random_init(
        config,
        seed=seed,
        input_capacity=input_capacity,
        output_capacity=output_capacity,
    )
