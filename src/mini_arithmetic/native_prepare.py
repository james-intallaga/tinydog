"""Prepare rights-cleared local duplex media for scratch-native MLM training.

This command performs deterministic conversion only.  It accepts canonical
16-bit PCM WAV tracks and optional preselected JPEG frames; it never calls ASR,
TTS, a vision model, an API, or a pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any, Sequence
import tempfile
import wave
import zipfile

import av
import numpy as np

from .multimodal import MultimodalConfig
from .native_data import (
    MAX_NATIVE_SHARD_BYTES,
    NATIVE_DATA_SCHEMA_VERSION,
    NativeShardDataset,
    native_manifest_digest,
)
from .scratch_tokenizer import ScratchTokenizer


NATIVE_PREPARATION_VERSION = 1
CANONICAL_SAMPLE_RATE = 24_000
CANONICAL_CHANNELS = 1
CANONICAL_SAMPLE_WIDTH = 2
CANONICAL_VIDEO_WIDTH = 320
CANONICAL_VIDEO_HEIGHT = 192
MAX_INPUT_JSONL_BYTES = 64 * 1024 * 1024
MAX_INPUT_LINE_BYTES = 256 * 1024
MAX_INPUT_WAV_BYTES = 512 * 1024 * 1024
MAX_INPUT_JPEG_BYTES = 512 * 1024
MAX_MODEL_CONFIG_BYTES = 2 * 1024 * 1024
MAX_SELECTED_FRAMES = 256
MAX_RECORDS = 1_000_000

_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_BANNED_SOURCE = {"unknown", "none", "n/a", "unspecified"}
_BANNED_LICENSE = {
    "unknown",
    "none",
    "n/a",
    "unlicensed",
    "all rights reserved",
}
_REQUIRED_FIELDS = {
    "record_id",
    "split_group",
    "user_wav",
    "model_wav",
    "assistant_text",
    "text_timestamp_seconds",
    "source",
    "license",
    "consent",
    "provenance",
}
_OPTIONAL_FIELDS = {"selected_frames"}
_FRAME_FIELDS = {"path", "timestamp_seconds", "difference_percent"}
_MODEL_CONFIG_FIELDS = {
    "thinker",
    "audio",
    "video",
    "talker",
    "time_scale_seconds",
}
_REQUIRED_PROVENANCE_FIELDS = {"speaker_release", "capture_session"}


@dataclass(frozen=True, slots=True)
class NativePreparationConfig:
    validation_fraction: float = 0.1
    split_seed: str = "mlm-native-split-v1"
    max_duration_seconds: float = 60.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not math.isfinite(float(self.validation_fraction))
            or not 0.0 < float(self.validation_fraction) < 1.0
        ):
            raise ValueError("validation_fraction must be finite and in (0, 1)")
        if (
            not isinstance(self.split_seed, str)
            or not self.split_seed
            or len(self.split_seed.encode("utf-8")) > 1_024
        ):
            raise ValueError("split_seed must be a non-empty bounded string")
        if (
            isinstance(self.max_duration_seconds, bool)
            or not isinstance(self.max_duration_seconds, (int, float))
            or not math.isfinite(float(self.max_duration_seconds))
            or self.max_duration_seconds <= 0.0
        ):
            raise ValueError("max_duration_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class PreparedNativeDataset:
    output_dir: Path
    train_manifest: Path
    validation_manifest: Path
    train_records: int
    validation_records: int
    train_manifest_sha256: str
    validation_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _InputRecord:
    record_id: str
    split_group: str
    user_wav: Path
    model_wav: Path
    assistant_text: str
    text_timestamp_seconds: float
    source: str
    license: str
    provenance: dict[str, Any]
    selected_frames: tuple[tuple[Path, float, float], ...]
    canonical_json_sha256: str


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("native preparation JSON contains a duplicate field")
        value[key] = item
    return value


def _reject_constant(_value: str) -> object:
    raise ValueError("native preparation JSON contains a non-finite number")


def _finite_number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return number


def _audited_text(value: object, name: str, *, banned: set[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"native {name} must be a string")
    text = value.strip()
    if not text or len(text.encode("utf-8")) > 512 or text.casefold() in banned:
        raise ValueError(f"native {name} is missing or not rights-cleared")
    return text


def _resolve_relative_file(
    root: Path,
    value: object,
    *,
    suffixes: set[str],
    maximum_bytes: int,
) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("native input media path is invalid")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("native input media paths must be relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "native input media path escapes the JSONL directory"
        ) from error
    if path.suffix.casefold() not in suffixes or not path.is_file():
        raise FileNotFoundError(f"Native input media file not found: {path}")
    if not 0 < path.stat().st_size <= maximum_bytes:
        raise ValueError("native input media file is empty or too large")
    return path


def _parse_record(value: object, root: Path, line_number: int) -> _InputRecord:
    if (
        not isinstance(value, dict)
        or not _REQUIRED_FIELDS.issubset(value)
        or not set(value).issubset(_REQUIRED_FIELDS | _OPTIONAL_FIELDS)
    ):
        raise ValueError(
            f"native input line {line_number} has missing or unsupported fields"
        )
    record_id = value["record_id"]
    split_group = value["split_group"]
    if not isinstance(record_id, str) or _SAFE_ID.fullmatch(record_id) is None:
        raise ValueError(f"native input line {line_number} record_id is invalid")
    if not isinstance(split_group, str) or _SAFE_ID.fullmatch(split_group) is None:
        raise ValueError(f"native input line {line_number} split_group is invalid")
    if value["consent"] is not True:
        raise ValueError("native voice/video preparation requires affirmative consent")
    source = _audited_text(value["source"], "source", banned=_BANNED_SOURCE)
    license_name = _audited_text(value["license"], "license", banned=_BANNED_LICENSE)
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("native preparation requires non-empty provenance")
    missing_provenance = _REQUIRED_PROVENANCE_FIELDS - set(provenance)
    if missing_provenance:
        raise ValueError(
            "native preparation provenance requires speaker_release and capture_session"
        )
    for field_name in sorted(_REQUIRED_PROVENANCE_FIELDS):
        _audited_text(
            provenance[field_name],
            f"provenance {field_name}",
            banned=_BANNED_SOURCE,
        )
    if "mlm_native_preparation" in provenance:
        raise ValueError("native provenance uses a reserved preparation field")
    assistant_text = value["assistant_text"]
    if (
        not isinstance(assistant_text, str)
        or not assistant_text.strip()
        or len(assistant_text.encode("utf-8")) > 128 * 1024
    ):
        raise ValueError("native assistant_text must be non-empty and bounded")
    timestamp = _finite_number(
        value["text_timestamp_seconds"], "text_timestamp_seconds"
    )
    user_wav = _resolve_relative_file(
        root,
        value["user_wav"],
        suffixes={".wav"},
        maximum_bytes=MAX_INPUT_WAV_BYTES,
    )
    model_wav = _resolve_relative_file(
        root,
        value["model_wav"],
        suffixes={".wav"},
        maximum_bytes=MAX_INPUT_WAV_BYTES,
    )
    raw_frames = value.get("selected_frames", [])
    if not isinstance(raw_frames, list):
        raise ValueError("native selected_frames must be a list")
    if len(raw_frames) > MAX_SELECTED_FRAMES:
        raise ValueError("native selected_frames exceeds the per-record limit")
    selected: list[tuple[Path, float, float]] = []
    selected_paths: set[Path] = set()
    selected_timestamps: set[float] = set()
    last_timestamp = -math.inf
    for item in raw_frames:
        if not isinstance(item, dict) or set(item) != _FRAME_FIELDS:
            raise ValueError("native selected frame fields are incompatible")
        frame_path = _resolve_relative_file(
            root,
            item["path"],
            suffixes={".jpg", ".jpeg"},
            maximum_bytes=MAX_INPUT_JPEG_BYTES,
        )
        frame_time = _finite_number(
            item["timestamp_seconds"], "selected frame timestamp"
        )
        difference = _finite_number(
            item["difference_percent"], "selected frame difference"
        )
        if difference > 100.0:
            raise ValueError("selected frame difference must be at most 100 percent")
        if frame_time < last_timestamp:
            raise ValueError("selected frame timestamps must be non-decreasing")
        if frame_path in selected_paths:
            raise ValueError("native selected frame paths must be unique per record")
        if frame_time in selected_timestamps:
            raise ValueError(
                "native selected frame timestamps must be unique per record"
            )
        last_timestamp = frame_time
        selected_paths.add(frame_path)
        selected_timestamps.add(frame_time)
        selected.append((frame_path, frame_time, difference))
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _InputRecord(
        record_id=record_id,
        split_group=split_group,
        user_wav=user_wav,
        model_wav=model_wav,
        assistant_text=assistant_text,
        text_timestamp_seconds=timestamp,
        source=source,
        license=license_name,
        provenance=dict(provenance),
        selected_frames=tuple(selected),
        canonical_json_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _read_records(path: Path) -> tuple[_InputRecord, ...]:
    resolved = path.expanduser().resolve()
    if (
        not resolved.is_file()
        or not 0 < resolved.stat().st_size <= MAX_INPUT_JSONL_BYTES
    ):
        raise FileNotFoundError(
            f"Native input JSONL not found or too large: {resolved}"
        )
    records: list[_InputRecord] = []
    with resolved.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            if len(raw) > MAX_INPUT_LINE_BYTES:
                raise ValueError(f"native input line {line_number} is too large")
            try:
                value = json.loads(
                    raw.decode("utf-8", errors="strict"),
                    object_pairs_hook=_strict_object,
                    parse_constant=_reject_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"native input line {line_number} is not valid UTF-8 JSON"
                ) from error
            records.append(_parse_record(value, resolved.parent, line_number))
            if len(records) > MAX_RECORDS:
                raise ValueError("native input contains too many records")
    if not records:
        raise ValueError("native input JSONL contains no records")
    ids = [record.record_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("native input record IDs must be unique")
    return tuple(sorted(records, key=lambda record: record.record_id))


def _read_model_config(path: Path) -> tuple[MultimodalConfig, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Native model configuration not found: {path}")
    raw = _read_bounded_media(path, MAX_MODEL_CONFIG_BYTES, "model configuration")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "native model configuration is not valid UTF-8 JSON"
        ) from error
    if not isinstance(payload, dict) or not set(payload).issubset(_MODEL_CONFIG_FIELDS):
        raise ValueError("native model configuration fields are incompatible")
    return MultimodalConfig.from_dict(payload), hashlib.sha256(raw).hexdigest()


def _read_bounded_media(path: Path, maximum_bytes: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"native {name} changed or is not a safe regular file"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"native {name} is empty or too large")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(raw) != before.st_size:
            raise ValueError(f"native {name} changed while it was being read")
        if not raw or len(raw) > maximum_bytes:
            raise ValueError(f"native {name} is empty or too large")
        return raw
    finally:
        os.close(descriptor)


def _read_pcm_wav(path: Path, *, maximum_seconds: float) -> tuple[np.ndarray, str]:
    source = _read_bounded_media(path, MAX_INPUT_WAV_BYTES, "WAV")
    try:
        with wave.open(io.BytesIO(source), "rb") as stream:
            if (
                stream.getcomptype() != "NONE"
                or stream.getnchannels() != CANONICAL_CHANNELS
                or stream.getsampwidth() != CANONICAL_SAMPLE_WIDTH
                or stream.getframerate() != CANONICAL_SAMPLE_RATE
            ):
                raise ValueError(
                    "native WAV must be uncompressed 16-bit PCM, mono, and 24 kHz"
                )
            frames = stream.getnframes()
            if frames <= 0 or frames > int(maximum_seconds * CANONICAL_SAMPLE_RATE):
                raise ValueError("native WAV duration is empty or exceeds the limit")
            raw = stream.readframes(frames)
            if len(raw) != frames * CANONICAL_SAMPLE_WIDTH:
                raise ValueError("native WAV PCM payload is truncated")
            if stream.readframes(1):
                raise ValueError("native WAV declares an inconsistent frame count")
    except (wave.Error, EOFError) as error:
        raise ValueError(f"native WAV could not be decoded: {path.name}") from error
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32_768.0
    return np.ascontiguousarray(pcm), hashlib.sha256(source).hexdigest()


def _read_canonical_jpeg(path: Path) -> tuple[np.ndarray, str]:
    raw = _read_bounded_media(path, MAX_INPUT_JPEG_BYTES, "selected JPEG")
    if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        raise ValueError("native selected frame is not a complete JPEG")
    if _jpeg_dimensions(raw) != (CANONICAL_VIDEO_WIDTH, CANONICAL_VIDEO_HEIGHT):
        raise ValueError("native selected JPEG must already be exactly 320x192")
    try:
        with av.open(io.BytesIO(raw), mode="r", format="mjpeg") as container:
            decoded = list(container.decode(video=0))
    except (av.error.FFmpegError, EOFError, OSError, ValueError) as error:
        raise ValueError(
            f"native selected JPEG could not be decoded: {path.name}"
        ) from error
    if len(decoded) != 1:
        raise ValueError("native selected JPEG must contain exactly one image")
    frame = decoded[0]
    if (frame.width, frame.height) != (
        CANONICAL_VIDEO_WIDTH,
        CANONICAL_VIDEO_HEIGHT,
    ):
        raise ValueError("native selected JPEG must already be exactly 320x192")
    rgb = frame.reformat(format="rgb24").to_ndarray()
    if rgb.shape != (CANONICAL_VIDEO_HEIGHT, CANONICAL_VIDEO_WIDTH, 3):
        raise ValueError("native selected JPEG decoded to an invalid RGB shape")
    return (
        np.ascontiguousarray(rgb.astype(np.float32) / 255.0),
        hashlib.sha256(raw).hexdigest(),
    )


def _jpeg_dimensions(jpeg: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions before allowing FFmpeg to allocate a frame."""

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
            raise ValueError("native selected JPEG has an invalid segment")
        if marker in start_of_frame:
            if segment_length < 7:
                raise ValueError("native selected JPEG has an invalid SOF segment")
            height = int.from_bytes(jpeg[offset + 3 : offset + 5], "big")
            width = int.from_bytes(jpeg[offset + 5 : offset + 7], "big")
            return width, height
        offset += segment_length
    raise ValueError("native selected JPEG has no image dimensions")


def _assistant_tokens(
    tokenizer: ScratchTokenizer, text: str
) -> tuple[np.ndarray, np.ndarray]:
    tokens, boundaries = tokenizer.encode_dialogue_with_boundaries(
        [{"role": "assistant", "content": text}]
    )
    if len(boundaries) != 1 or boundaries[0].role != "assistant":
        raise RuntimeError("scratch tokenizer produced an invalid assistant boundary")
    mask = np.zeros((len(tokens),), dtype=np.float32)
    boundary = boundaries[0]
    mask[boundary.role_offset : boundary.end_turn_offset + 1] = 1.0
    if boundary.end_turn_offset == len(tokens) - 2:
        mask[-1] = 1.0
    return np.asarray(tokens, dtype=np.int32), mask


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_component_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value)
    metadata = json.dumps(
        {"dtype": canonical.dtype.str, "shape": canonical.shape},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            value = np.asarray(arrays[name])
            if value.ndim:
                value = np.ascontiguousarray(value)
            archive.writestr(info, _npy_bytes(value))
    if path.stat().st_size > MAX_NATIVE_SHARD_BYTES:
        raise ValueError("prepared native shard exceeds the reader size limit")


def _split_groups(
    records: tuple[_InputRecord, ...], config: NativePreparationConfig
) -> dict[str, str]:
    groups = sorted({record.split_group for record in records})
    if len(groups) < 2:
        raise ValueError(
            "native deterministic split requires at least two split groups"
        )
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{config.split_seed}\0{group}".encode("utf-8")).digest(),
            group,
        ),
    )
    validation_count = max(
        1,
        min(
            len(groups) - 1,
            int(math.ceil(len(groups) * float(config.validation_fraction))),
        ),
    )
    validation = set(ranked[:validation_count])
    return {group: "validation" if group in validation else "train" for group in groups}


def _record_arrays(
    record: _InputRecord,
    *,
    model_config: MultimodalConfig,
    tokenizer: ScratchTokenizer,
    maximum_seconds: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    user, user_wav_sha256 = _read_pcm_wav(
        record.user_wav, maximum_seconds=maximum_seconds
    )
    model, model_wav_sha256 = _read_pcm_wav(
        record.model_wav, maximum_seconds=maximum_seconds
    )
    if user.shape != model.shape:
        raise ValueError(
            f"native record {record.record_id} duplex WAVs must have equal samples"
        )
    if user.size % model_config.audio.frame_size:
        raise ValueError(
            f"native record {record.record_id} samples must align to codec frame_size"
        )
    duration = user.size / CANONICAL_SAMPLE_RATE
    if record.text_timestamp_seconds > duration + 1e-6:
        raise ValueError(
            f"native record {record.record_id} text timestamp exceeds audio duration"
        )
    tokens, mask = _assistant_tokens(tokenizer, record.assistant_text)
    timeline_tokens = user.size // model_config.audio.frame_size + 2 + int(tokens.size)
    if record.selected_frames:
        timeline_tokens += 2 + len(record.selected_frames) * (
            1 + model_config.video.resampler_tokens
        )
    if timeline_tokens > model_config.thinker.context_length:
        raise ValueError(
            f"native record {record.record_id} exceeds the thinker context length"
        )
    arrays: dict[str, np.ndarray] = {
        "user_waveform": user[None, :, None].astype(np.float32, copy=False),
        "model_waveform": model[None, :, None].astype(np.float32, copy=False),
        "text_tokens": tokens[None, :],
        "text_target_mask": mask[None, :],
        "text_timestamp": np.array(record.text_timestamp_seconds, dtype=np.float32),
    }
    frame_hashes: list[str] = []
    if record.selected_frames:
        frames: list[np.ndarray] = []
        timestamps: list[float] = []
        differences: list[float] = []
        for frame_path, frame_time, difference_percent in record.selected_frames:
            if frame_time > duration + 1e-6:
                raise ValueError(
                    f"native record {record.record_id} frame timestamp exceeds audio duration"
                )
            frame, frame_sha256 = _read_canonical_jpeg(frame_path)
            frames.append(frame)
            timestamps.append(frame_time)
            differences.append(difference_percent / 100.0)
            frame_hashes.append(frame_sha256)
        arrays |= {
            "frames": np.stack(frames, axis=0)[None, :, :, :, :].astype(
                np.float32, copy=False
            ),
            "frame_timestamps": np.asarray(timestamps, dtype=np.float32),
            "frame_difference_scores": np.asarray(differences, dtype=np.float32),
        }
    media_components = [user, model]
    if "frames" in arrays:
        media_components.extend(arrays["frames"][0])
    decoded_media_hashes = [
        _canonical_component_sha256(component) for component in media_components
    ]
    leakage_media_hashes = [
        digest
        for component, digest in zip(
            media_components, decoded_media_hashes, strict=True
        )
        if np.any(component)
    ]
    preparation = {
        "schema_version": NATIVE_PREPARATION_VERSION,
        "input_record_sha256": record.canonical_json_sha256,
        "user_wav_sha256": user_wav_sha256,
        "model_wav_sha256": model_wav_sha256,
        "selected_frame_sha256": frame_hashes,
        "decoded_media_sha256": decoded_media_hashes,
        "leakage_media_sha256": leakage_media_hashes,
        "sample_rate": CANONICAL_SAMPLE_RATE,
        "channels": CANONICAL_CHANNELS,
        "sample_width_bytes": CANONICAL_SAMPLE_WIDTH,
        "samples": int(user.size),
        "split_group": record.split_group,
        "tokenizer_sha256": tokenizer.metadata["tokenizer_sha256"],
        "external_inference": False,
        "pretrained_weights": False,
    }
    return arrays, preparation


def _canonical_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` while refusing even an empty destination."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename_exclusive = libc.renamex_np
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_no_replace = libc.renameat2
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-replace directory publication is unavailable"
            ) from error
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
    elif os.name == "nt":
        os.rename(source, destination)
        return
    else:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "native output destination already exists",
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def prepare_native_dataset(
    input_jsonl: str | Path,
    output_dir: str | Path,
    *,
    model_config_path: str | Path,
    tokenizer_path: str | Path,
    config: NativePreparationConfig | None = None,
) -> PreparedNativeDataset:
    """Atomically prepare deterministic train/validation native shards."""

    options = config or NativePreparationConfig()
    input_path = Path(input_jsonl).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Native output directory already exists: {output}")
    model_path = Path(model_config_path).expanduser().resolve()
    tokenizer = ScratchTokenizer.load(tokenizer_path)
    model_config, model_digest = _read_model_config(model_path)
    if (
        model_config.audio.sample_rate != CANONICAL_SAMPLE_RATE
        or model_config.audio.channels != CANONICAL_CHANNELS
    ):
        raise ValueError("native model config must use mono 24 kHz audio")
    divisor = model_config.video.patch_size * model_config.video.spatial_merge_size
    if CANONICAL_VIDEO_WIDTH % divisor or CANONICAL_VIDEO_HEIGHT % divisor:
        raise ValueError("native model vision grid is incompatible with 320x192")
    if model_config.thinker.vocab_size != tokenizer.vocab_size:
        raise ValueError("native model vocabulary does not match the frozen tokenizer")
    records = _read_records(input_path)
    split_by_group = _split_groups(records, options)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.prepare.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"Native output is already being prepared: {output}"
        ) from error
    staging: Path | None = None
    try:
        if output.exists():
            raise FileExistsError(f"Native output directory already exists: {output}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
        )
        manifest_records: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "validation": [],
        }
        media_splits: dict[str, str] = {}
        split_seed_digest = hashlib.sha256(
            options.split_seed.encode("utf-8")
        ).hexdigest()
        for split in manifest_records:
            (staging / split).mkdir()
        for record in records:
            split = split_by_group[record.split_group]
            arrays, preparation = _record_arrays(
                record,
                model_config=model_config,
                tokenizer=tokenizer,
                maximum_seconds=options.max_duration_seconds,
            )
            leakage_media_hashes = preparation["leakage_media_sha256"]
            if not isinstance(leakage_media_hashes, list):
                raise RuntimeError("native preparation media hashes are invalid")
            for media_digest in leakage_media_hashes:
                if not isinstance(media_digest, str):
                    raise RuntimeError("native preparation media hash is invalid")
                previous_split = media_splits.setdefault(media_digest, split)
                if previous_split != split:
                    raise ValueError(
                        "prepared native train/validation component media overlap"
                    )
            preparation["model_config_sha256"] = model_digest
            preparation["split_seed_sha256"] = split_seed_digest
            preparation["assigned_split"] = split
            shard = staging / split / f"{record.record_id}.npz"
            _write_deterministic_npz(shard, arrays)
            provenance = dict(record.provenance)
            provenance["mlm_native_preparation"] = preparation
            manifest_records[split].append(
                {
                    "record_id": record.record_id,
                    "shard": shard.name,
                    "sha256": _sha256_file(shard),
                    "source": record.source,
                    "license": record.license,
                    "consent": True,
                    "provenance": provenance,
                }
            )
        tokenizer_digest = str(tokenizer.metadata["tokenizer_sha256"])
        for split, split_records in manifest_records.items():
            if not split_records:
                raise ValueError(
                    "native deterministic split produced an empty partition"
                )
            _canonical_json(
                staging / split / "manifest.json",
                {
                    "schema_version": NATIVE_DATA_SCHEMA_VERSION,
                    "split": split,
                    "tokenizer_sha256": tokenizer_digest,
                    "sample_rate": CANONICAL_SAMPLE_RATE,
                    "frame_size": model_config.audio.frame_size,
                    "records": sorted(
                        split_records, key=lambda item: str(item["record_id"])
                    ),
                },
            )
        train = NativeShardDataset(
            staging / "train" / "manifest.json",
            split="train",
            config=model_config,
            tokenizer_sha256=tokenizer_digest,
        )
        validation = NativeShardDataset(
            staging / "validation" / "manifest.json",
            split="validation",
            config=model_config,
            tokenizer_sha256=tokenizer_digest,
        )
        if train.content_hashes & validation.content_hashes:
            raise ValueError("prepared native train/validation media overlap")
        if output.exists():
            raise FileExistsError(
                f"Native output directory appeared during preparation: {output}"
            )
        _publish_directory_no_replace(staging, output)
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
    train_manifest = output / "train" / "manifest.json"
    validation_manifest = output / "validation" / "manifest.json"
    return PreparedNativeDataset(
        output_dir=output,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        train_records=len(manifest_records["train"]),
        validation_records=len(manifest_records["validation"]),
        train_manifest_sha256=native_manifest_digest(train_manifest),
        validation_manifest_sha256=native_manifest_digest(validation_manifest),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Strict native JSONL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--validation-percent", type=float, default=10.0)
    parser.add_argument("--split-seed", default="mlm-native-split-v1")
    parser.add_argument("--max-duration-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = prepare_native_dataset(
        args.input,
        args.output,
        model_config_path=args.model_config,
        tokenizer_path=args.tokenizer,
        config=NativePreparationConfig(
            validation_fraction=args.validation_percent / 100.0,
            split_seed=args.split_seed,
            max_duration_seconds=args.max_duration_seconds,
        ),
    )
    print(
        json.dumps(
            {
                "output": str(result.output_dir),
                "train_manifest": str(result.train_manifest),
                "train_records": result.train_records,
                "train_manifest_sha256": result.train_manifest_sha256,
                "validation_manifest": str(result.validation_manifest),
                "validation_records": result.validation_records,
                "validation_manifest_sha256": result.validation_manifest_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
