"""Audited, rights-aware tensor shards for scratch native-media training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Literal
import zipfile

import mlx.core as mx
import numpy as np

from .multimodal import MultimodalConfig
from .native_objectives import NativeConversationBatch


NATIVE_DATA_SCHEMA_VERSION = 1
MAX_NATIVE_SHARD_BYTES = 128 * 1024 * 1024
MAX_NATIVE_DECOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_NATIVE_MANIFEST_BYTES = 16 * 1024 * 1024
NativeSplit = Literal["train", "validation", "test"]
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RECORD_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_BASE_KEYS = {
    "user_waveform",
    "model_waveform",
    "text_tokens",
    "text_target_mask",
    "text_timestamp",
}
_VIDEO_KEYS = {"frames", "frame_timestamps", "frame_difference_scores"}


@dataclass(frozen=True)
class NativeShardReference:
    record_id: str
    path: Path
    sha256: str
    source: str
    license: str
    consent: bool
    provenance: dict[str, Any]


class NativeShardDataset:
    """Lazy, checksum-bound native tensor batches from one immutable split."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: NativeSplit,
        config: MultimodalConfig,
        tokenizer_sha256: str,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Native manifest not found: {self.manifest_path}")
        raw = self.manifest_path.read_bytes()
        if not raw or len(raw) > MAX_NATIVE_MANIFEST_BYTES:
            raise ValueError("native manifest is empty or too large")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        try:

            def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("native manifest contains duplicate fields")
                    result[key] = value
                return result

            payload = json.loads(
                raw.decode("utf-8", errors="strict"),
                object_pairs_hook=reject_duplicates,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("native manifest contains a non-finite number")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("native manifest is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("native manifest must be a JSON object")
        expected_keys = {
            "schema_version",
            "split",
            "tokenizer_sha256",
            "sample_rate",
            "frame_size",
            "records",
        }
        if set(payload) != expected_keys:
            raise ValueError("native manifest fields are incompatible")
        if payload["schema_version"] != NATIVE_DATA_SCHEMA_VERSION:
            raise ValueError("native manifest schema is incompatible")
        if payload["split"] != split:
            raise ValueError("native manifest split does not match the requested split")
        if not _HEX64.fullmatch(tokenizer_sha256):
            raise ValueError("expected tokenizer checksum must be lowercase SHA-256")
        if payload["tokenizer_sha256"] != tokenizer_sha256:
            raise ValueError("native manifest tokenizer checksum does not match")
        if payload["sample_rate"] != config.audio.sample_rate:
            raise ValueError("native manifest sample rate does not match the model")
        if payload["frame_size"] != config.audio.frame_size:
            raise ValueError("native manifest codec quantum does not match the model")
        records = payload["records"]
        if not isinstance(records, list) or not records:
            raise ValueError("native manifest must contain records")
        self.split = split
        self.config = config
        self.tokenizer_sha256 = tokenizer_sha256
        self.records = tuple(self._parse_record(item) for item in records)
        ids = [item.record_id for item in self.records]
        paths = [item.path for item in self.records]
        digests = [item.sha256 for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("native manifest record IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("native manifest shard paths must be unique")
        if len(digests) != len(set(digests)):
            raise ValueError("native manifest may not duplicate media content")
        self._content_hashes: frozenset[str] | None = None

    def _parse_record(self, value: object) -> NativeShardReference:
        if not isinstance(value, dict):
            raise ValueError("native manifest record must be an object")
        required = {
            "record_id",
            "shard",
            "sha256",
            "source",
            "license",
            "consent",
            "provenance",
        }
        if set(value) != required:
            raise ValueError("native manifest record fields are incompatible")
        record_id = value["record_id"]
        if (
            not isinstance(record_id, str)
            or _SAFE_RECORD_ID.fullmatch(record_id) is None
        ):
            raise ValueError("native record_id is invalid")
        digest = value["sha256"]
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise ValueError("native shard checksum is invalid")
        source = value["source"]
        license_name = value["license"]
        if (
            not isinstance(source, str)
            or not source.strip()
            or len(source) > 512
            or source.strip().casefold() in {"unknown", "none", "n/a", "unspecified"}
        ):
            raise ValueError("native training source is required")
        if (
            not isinstance(license_name, str)
            or not license_name.strip()
            or len(license_name) > 512
            or license_name.strip().casefold()
            in {"unknown", "none", "n/a", "unlicensed", "all rights reserved"}
        ):
            raise ValueError("native training license is required")
        if value["consent"] is not True:
            raise ValueError("native voice/video records require affirmative consent")
        provenance = value["provenance"]
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("native training provenance is required")
        shard_name = value["shard"]
        if not isinstance(shard_name, str) or not shard_name or "\x00" in shard_name:
            raise ValueError("native shard path is invalid")
        root = self.manifest_path.parent
        shard_path = (root / shard_name).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "native shard path escapes its manifest directory"
            ) from error
        if not shard_path.is_file() or shard_path.suffix != ".npz":
            raise FileNotFoundError(f"Native tensor shard not found: {shard_path}")
        if shard_path.stat().st_size > MAX_NATIVE_SHARD_BYTES:
            raise ValueError("native tensor shard is too large")
        return NativeShardReference(
            record_id=record_id,
            path=shard_path,
            sha256=digest,
            source=source,
            license=license_name,
            consent=True,
            provenance=dict(provenance),
        )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def content_hashes(self) -> frozenset[str]:
        """Canonical hashes of decoded native media, independent of NPZ bytes.

        A byte checksum protects each shard in transit, but two NPZ archives
        can contain identical audio/video while having different container
        bytes.  Readiness splits therefore use this decoded-media identity.
        Accessing the property intentionally audits every shard before training.
        """

        if self._content_hashes is None:
            hashes: list[str] = []
            for reference in self.records:
                arrays = self._load_verified_arrays(reference)
                for index in range(arrays["user_waveform"].shape[0]):
                    hashes.extend(_canonical_media_sha256s(arrays, batch_index=index))
            if len(hashes) != len(set(hashes)):
                raise ValueError("native manifest may not duplicate decoded media")
            self._content_hashes = frozenset(hashes)
        return self._content_hashes

    @property
    def shard_hashes(self) -> frozenset[str]:
        return frozenset(item.sha256 for item in self.records)

    @property
    def record_ids(self) -> frozenset[str]:
        return frozenset(item.record_id for item in self.records)

    def load(self, index: int) -> NativeConversationBatch:
        reference = self.records[index]
        arrays = self._load_verified_arrays(reference)
        batch = NativeConversationBatch(
            user_waveform=mx.array(arrays["user_waveform"]),
            model_waveform=mx.array(arrays["model_waveform"]),
            text_tokens=mx.array(arrays["text_tokens"]),
            text_target_mask=mx.array(arrays["text_target_mask"]),
            text_timestamp=float(arrays["text_timestamp"].item()),
            frames=mx.array(arrays["frames"]) if "frames" in arrays else None,
            frame_timestamps=(
                mx.array(arrays["frame_timestamps"])
                if "frame_timestamps" in arrays
                else None
            ),
            frame_difference_scores=(
                mx.array(arrays["frame_difference_scores"])
                if "frame_difference_scores" in arrays
                else None
            ),
        )
        return batch

    def _load_verified_arrays(
        self, reference: NativeShardReference
    ) -> dict[str, np.ndarray]:
        raw = reference.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference.sha256:
            raise ValueError(f"native shard checksum changed for {reference.record_id}")
        try:
            with zipfile.ZipFile(io.BytesIO(raw), mode="r") as container:
                members = container.infolist()
                names = [item.filename for item in members]
                if len(names) != len(set(names)):
                    raise ValueError("native shard contains duplicate archive members")
                if any(item.flag_bits & 0x1 for item in members):
                    raise ValueError("native shard may not contain encrypted tensors")
                if (
                    sum(item.file_size for item in members)
                    > MAX_NATIVE_DECOMPRESSED_BYTES
                ):
                    raise ValueError(
                        "native shard expands beyond the tensor byte limit"
                    )
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                keys = set(archive.files)
                if keys != _BASE_KEYS and keys != _BASE_KEYS | _VIDEO_KEYS:
                    raise ValueError("native shard tensor fields are incompatible")
                arrays = {name: np.array(archive[name], copy=True) for name in keys}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            if isinstance(error, ValueError) and str(error).startswith("native shard"):
                raise
            raise ValueError(
                "native tensor shard could not be decoded safely"
            ) from error
        self._validate_arrays(arrays)
        return arrays

    def _validate_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        user = arrays["user_waveform"]
        model = arrays["model_waveform"]
        tokens = arrays["text_tokens"]
        mask = arrays["text_target_mask"]
        timestamp = arrays["text_timestamp"]
        if user.dtype != np.float32 or model.dtype != np.float32:
            raise ValueError("native waveforms must use float32")
        if (
            user.ndim != 3
            or model.shape != user.shape
            or user.shape[-1] != self.config.audio.channels
        ):
            raise ValueError(
                "native waveforms must be aligned [batch, samples, channels]"
            )
        if user.shape[0] <= 0 or user.shape[1] <= 0:
            raise ValueError("native waveform shard is empty")
        if user.shape[1] % self.config.audio.frame_size:
            raise ValueError("native waveform samples must align to codec frames")
        if not np.isfinite(user).all() or not np.isfinite(model).all():
            raise ValueError("native waveforms contain non-finite samples")
        if (
            np.max(np.abs(user), initial=0.0) > 1.0
            or np.max(np.abs(model), initial=0.0) > 1.0
        ):
            raise ValueError("native waveform samples must be normalized to [-1, 1]")
        if (
            tokens.dtype != np.int32
            or tokens.ndim != 2
            or tokens.shape[0] != user.shape[0]
        ):
            raise ValueError("native text tokens must use int32 [batch, length]")
        if mask.dtype != np.float32 or mask.shape != tokens.shape:
            raise ValueError("native text target mask must be float32 and match tokens")
        if not np.isin(mask, (0.0, 1.0)).all() or float(mask.sum()) <= 0.0:
            raise ValueError("native text target mask must be binary and non-empty")
        if tokens.size and (
            tokens.min() < 0 or tokens.max() >= self.config.thinker.vocab_size
        ):
            raise ValueError("native text tokens are outside the thinker vocabulary")
        if timestamp.shape != () or timestamp.dtype not in {
            np.dtype("float32"),
            np.dtype("float64"),
        }:
            raise ValueError("native text_timestamp must be a scalar float")
        if not math.isfinite(float(timestamp.item())) or float(timestamp.item()) < 0.0:
            raise ValueError("native text_timestamp must be finite and non-negative")
        duration = user.shape[1] / self.config.audio.sample_rate
        if float(timestamp.item()) > duration + 1e-6:
            raise ValueError("native text_timestamp lies outside the audio window")
        timeline_tokens = (
            user.shape[1] // self.config.audio.frame_size + 2 + tokens.shape[1]
        )
        if "frames" not in arrays:
            if timeline_tokens > self.config.thinker.context_length:
                raise ValueError("native record exceeds the thinker context length")
            return
        frames = arrays["frames"]
        times = arrays["frame_timestamps"]
        differences = arrays["frame_difference_scores"]
        if frames.dtype != np.float32 or frames.ndim != 5 or frames.shape[-1] != 3:
            raise ValueError(
                "native frames must use float32 [batch, frames, H, W, RGB]"
            )
        if frames.shape[0] != user.shape[0] or frames.shape[1] <= 0:
            raise ValueError("native frame batch does not match its audio")
        divisor = self.config.video.patch_size * self.config.video.spatial_merge_size
        if frames.shape[2] % divisor or frames.shape[3] % divisor:
            raise ValueError(
                "native frame dimensions do not match the vision patch grid"
            )
        if not np.isfinite(frames).all() or frames.min() < 0.0 or frames.max() > 1.0:
            raise ValueError("native frames must be normalized finite RGB in [0, 1]")
        valid_metadata_shapes = {(frames.shape[1],), frames.shape[:2]}
        if times.dtype != np.float32 or times.shape not in valid_metadata_shapes:
            raise ValueError("native frame timestamps have an incompatible shape")
        if (
            differences.dtype != np.float32
            or differences.shape not in valid_metadata_shapes
        ):
            raise ValueError("native frame differences have an incompatible shape")
        if not np.isfinite(times).all() or np.any(times < 0.0):
            raise ValueError("native frame timestamps must be finite and non-negative")
        if np.any(times > duration + 1e-6):
            raise ValueError("native frame timestamps lie outside the audio window")
        ordered_times = times[None, :] if times.ndim == 1 else times
        if np.any(np.diff(ordered_times, axis=1) < 0.0):
            raise ValueError("native frame timestamps must be non-decreasing")
        if (
            not np.isfinite(differences).all()
            or np.any(differences < 0.0)
            or np.any(differences > 1.0)
        ):
            raise ValueError("native frame differences must be normalized to [0, 1]")
        timeline_tokens += 2 + frames.shape[1] * (
            1 + self.config.video.resampler_tokens
        )
        if timeline_tokens > self.config.thinker.context_length:
            raise ValueError("native record exceeds the thinker context length")


def _canonical_media_sha256s(
    arrays: dict[str, np.ndarray], *, batch_index: int
) -> tuple[str, ...]:
    components: list[np.ndarray] = [
        arrays["user_waveform"][batch_index],
        arrays["model_waveform"][batch_index],
    ]
    if "frames" in arrays:
        components.extend(arrays["frames"][batch_index])
    hashes: list[str] = []
    for raw_value in components:
        value = np.ascontiguousarray(raw_value)
        # Canonical all-zero silence/blank padding carries no speaker or scene
        # identity and is intentionally reusable across held-out partitions.
        if not np.any(value):
            continue
        metadata = json.dumps(
            {"dtype": value.dtype.str, "shape": value.shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(memoryview(value).cast("B"))
        hashes.append(digest.hexdigest())
    return tuple(hashes)


def native_manifest_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest()
