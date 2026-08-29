"""Untrusted community updates for the separately owned floaty layer.

The foundation is deliberately absent from every writable path in this
module.  A contribution is an immutable directory containing a strict JSON
manifest and one fixed-schema safetensors delta.  Contributions enter an
append-only quarantine queue, are checked against an operator-owned contract,
and may be combined into an immutable candidate.  Nothing here promotes a
candidate or changes an active model pointer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat as stat_module
import struct
import sys
import tempfile
import threading
from typing import Any, Iterator, Mapping, Sequence
import uuid

import numpy as np


PROTOCOL_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
DELTA_FILENAME = "floaty-delta.safetensors"
CANDIDATE_WEIGHTS_FILENAME = "floaty.safetensors"
CANDIDATE_REPORT_FILENAME = "report.json"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_TENSOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4}
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "update_id",
        "created_at",
        "foundation_sha256",
        "community_version",
        "community_weights_sha256",
        "architecture_sha256",
        "tokenizer_sha256",
        "floaty_schema_sha256",
        "training_recipe_sha256",
        "delta_file",
        "delta_sha256",
        "delta_bytes",
        "example_count",
    }
)
_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "foundation_sha256",
        "community_version",
        "community_weights_sha256",
        "architecture_sha256",
        "tokenizer_sha256",
        "training_recipe_sha256",
        "floaty_schema_sha256",
        "tensors",
    }
)
_TENSOR_SPEC_KEYS = frozenset({"name", "shape", "dtype"})


class CommunityUpdateError(ValueError):
    """Base error carrying a stable rejection code for queue receipts."""

    code = "community_update_error"


class ManifestError(CommunityUpdateError):
    code = "invalid_manifest"


class UnsafeBundleError(CommunityUpdateError):
    code = "unsafe_bundle"


class ReplayError(CommunityUpdateError):
    code = "replay"


class CompatibilityError(CommunityUpdateError):
    code = "incompatible"


class TensorValidationError(CommunityUpdateError):
    code = "invalid_tensors"


class QueueStateError(CommunityUpdateError):
    code = "invalid_queue_state"


@dataclass(frozen=True, order=True)
class TensorSpec:
    """One operator-approved floaty tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "F32"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _TENSOR_RE.fullmatch(self.name):
            raise ValueError("invalid tensor name")
        if not self.name.startswith("floaty."):
            raise ValueError("floaty tensor names must start with 'floaty.'")
        if ".." in self.name or "/" in self.name or "\\" in self.name:
            raise ValueError("unsafe tensor name")
        if not isinstance(self.shape, tuple) or len(self.shape) > 8:
            raise ValueError("tensor shape must be a tuple with at most 8 dimensions")
        for dimension in self.shape:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError("tensor dimensions must be positive integers")
        if not isinstance(self.dtype, str) or self.dtype not in _DTYPE_BYTES:
            raise ValueError("floaty tensors must use F16, BF16, or F32")

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_count(self) -> int:
        return self.element_count * _DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class CommunityContract:
    """Compatibility values controlled by the foundation owner."""

    foundation_sha256: str
    community_version: str
    community_weights_sha256: str
    architecture_sha256: str
    tokenizer_sha256: str
    training_recipe_sha256: str
    tensor_specs: tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        for name in (
            "foundation_sha256",
            "community_weights_sha256",
            "architecture_sha256",
            "tokenizer_sha256",
            "training_recipe_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.community_version, str) or not _VERSION_RE.fullmatch(
            self.community_version
        ):
            raise ValueError("invalid community_version")
        if not isinstance(self.tensor_specs, tuple) or not self.tensor_specs:
            raise ValueError("tensor_specs must be a non-empty tuple")
        if any(not isinstance(spec, TensorSpec) for spec in self.tensor_specs):
            raise ValueError("tensor_specs must contain only TensorSpec values")
        names = [spec.name for spec in self.tensor_specs]
        if len(set(names)) != len(names):
            raise ValueError("tensor_specs contains duplicate names")

    @classmethod
    def from_floaty_schema(
        cls,
        tensors: Sequence[Mapping[str, Any] | TensorSpec],
        *,
        foundation_sha256: str,
        community_version: str,
        community_weights_sha256: str,
        architecture_sha256: str,
        tokenizer_sha256: str,
        training_recipe_sha256: str,
    ) -> "CommunityContract":
        """Bind a queue contract directly to ``config_manifest()['tensors']``."""

        specs = tensor_specs_from_floaty_schema(tensors)
        return cls(
            foundation_sha256=foundation_sha256,
            community_version=community_version,
            community_weights_sha256=community_weights_sha256,
            architecture_sha256=architecture_sha256,
            tokenizer_sha256=tokenizer_sha256,
            training_recipe_sha256=training_recipe_sha256,
            tensor_specs=specs,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CommunityContract":
        if not isinstance(value, Mapping) or frozenset(value) != _CONTRACT_KEYS:
            raise ValueError("community contract keys do not match schema")
        version = value["schema_version"]
        if isinstance(version, bool) or version != PROTOCOL_VERSION:
            raise ValueError("unsupported community contract schema_version")
        contract = cls.from_floaty_schema(
            value["tensors"],
            foundation_sha256=value["foundation_sha256"],
            community_version=value["community_version"],
            community_weights_sha256=value["community_weights_sha256"],
            architecture_sha256=value["architecture_sha256"],
            tokenizer_sha256=value["tokenizer_sha256"],
            training_recipe_sha256=value["training_recipe_sha256"],
        )
        _require_sha256(value["floaty_schema_sha256"], "floaty_schema_sha256")
        if value["floaty_schema_sha256"] != contract.floaty_schema_sha256:
            raise ValueError("community contract floaty_schema_sha256 is incorrect")
        return contract

    @property
    def floaty_schema_sha256(self) -> str:
        return floaty_schema_sha256(self.tensor_specs)

    @property
    def specs_by_name(self) -> dict[str, TensorSpec]:
        return {spec.name: spec for spec in self.tensor_specs}

    def compatibility_dict(self) -> dict[str, Any]:
        return {
            "foundation_sha256": self.foundation_sha256,
            "community_version": self.community_version,
            "community_weights_sha256": self.community_weights_sha256,
            "architecture_sha256": self.architecture_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "floaty_schema_sha256": self.floaty_schema_sha256,
            "training_recipe_sha256": self.training_recipe_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_VERSION,
            **self.compatibility_dict(),
            "tensors": [
                spec.to_dict()
                for spec in sorted(self.tensor_specs, key=lambda item: item.name)
            ],
        }


@dataclass(frozen=True)
class UpdateLimits:
    """Resource, freshness, influence, and aggregation limits."""

    max_manifest_bytes: int = 64 * 1024
    max_delta_bytes: int = 32 * 1024 * 1024
    max_tensor_count: int = 4096
    max_tensor_elements: int = 50_000_000
    max_example_count: int = 10_000
    # Until authenticated, rate-limited contributor identities exist, every
    # accepted device gets one equal vote.  example_count remains audit data.
    max_weight_per_update: int = 1
    max_delta_l2_norm: float = 100.0
    aggregation_clip_l2_norm: float = 10.0
    max_age: timedelta = timedelta(days=7)
    max_future_skew: timedelta = timedelta(minutes=5)
    min_aggregate_updates: int = 3

    def __post_init__(self) -> None:
        integer_values = (
            self.max_manifest_bytes,
            self.max_delta_bytes,
            self.max_tensor_count,
            self.max_tensor_elements,
            self.max_example_count,
            self.max_weight_per_update,
            self.min_aggregate_updates,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_values
        ):
            raise ValueError("integer update limits must be positive")
        if not math.isfinite(self.max_delta_l2_norm) or self.max_delta_l2_norm <= 0:
            raise ValueError("max_delta_l2_norm must be finite and positive")
        if (
            not math.isfinite(self.aggregation_clip_l2_norm)
            or self.aggregation_clip_l2_norm <= 0
            or self.aggregation_clip_l2_norm > self.max_delta_l2_norm
        ):
            raise ValueError(
                "aggregation clip norm must be positive and no larger than the hard norm limit"
            )
        if self.max_age <= timedelta(0) or self.max_future_skew < timedelta(0):
            raise ValueError("freshness windows are invalid")


@dataclass(frozen=True)
class UpdateManifest:
    schema_version: int
    update_id: str
    created_at: str
    foundation_sha256: str
    community_version: str
    community_weights_sha256: str
    architecture_sha256: str
    tokenizer_sha256: str
    floaty_schema_sha256: str
    training_recipe_sha256: str
    delta_file: str
    delta_sha256: str
    delta_bytes: int
    example_count: int

    def __post_init__(self) -> None:
        if self.schema_version != PROTOCOL_VERSION or isinstance(
            self.schema_version, bool
        ):
            raise ManifestError(f"unsupported schema_version: {self.schema_version!r}")
        if not isinstance(self.update_id, str) or not _IDENTIFIER_RE.fullmatch(
            self.update_id
        ):
            raise ManifestError("invalid update_id")
        _parse_timestamp(self.created_at)
        for name in (
            "foundation_sha256",
            "community_weights_sha256",
            "architecture_sha256",
            "tokenizer_sha256",
            "floaty_schema_sha256",
            "training_recipe_sha256",
            "delta_sha256",
        ):
            try:
                _require_sha256(getattr(self, name), name)
            except ValueError as error:
                raise ManifestError(str(error)) from error
        if not isinstance(self.community_version, str) or not _VERSION_RE.fullmatch(
            self.community_version
        ):
            raise ManifestError("invalid community_version")
        if self.delta_file != DELTA_FILENAME:
            raise UnsafeBundleError(f"delta_file must be exactly {DELTA_FILENAME!r}")
        for name in ("delta_bytes", "example_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ManifestError(f"{name} must be a positive integer")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "UpdateManifest":
        try:
            decoded = payload.decode("utf-8")
            value = json.loads(decoded, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ManifestError("manifest must be valid UTF-8 JSON") from error
        if not isinstance(value, dict):
            raise ManifestError("manifest root must be an object")
        keys = frozenset(value)
        if keys != _MANIFEST_KEYS:
            missing = sorted(_MANIFEST_KEYS - keys)
            extra = sorted(keys - _MANIFEST_KEYS)
            raise ManifestError(
                f"manifest keys do not match schema; missing={missing}, extra={extra}"
            )
        try:
            return cls(**value)
        except TypeError as error:
            raise ManifestError("manifest field types do not match schema") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "update_id": self.update_id,
            "created_at": self.created_at,
            "foundation_sha256": self.foundation_sha256,
            "community_version": self.community_version,
            "community_weights_sha256": self.community_weights_sha256,
            "architecture_sha256": self.architecture_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "floaty_schema_sha256": self.floaty_schema_sha256,
            "training_recipe_sha256": self.training_recipe_sha256,
            "delta_file": self.delta_file,
            "delta_sha256": self.delta_sha256,
            "delta_bytes": self.delta_bytes,
            "example_count": self.example_count,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_json(self.to_dict())

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def content_sha256(self) -> str:
        """Stable replay identity excluding client-selected identity and time."""

        value = self.to_dict()
        value.pop("update_id")
        value.pop("created_at")
        return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class QueueEntry:
    content_sha256: str
    path: Path
    state: str
    manifest: UpdateManifest


@dataclass(frozen=True)
class ValidatedContribution:
    entry: QueueEntry
    tensors: Mapping[str, np.ndarray]
    delta_l2_norm: float


@dataclass(frozen=True)
class AggregationResult:
    candidate_id: str
    candidate_dir: Path
    weights_path: Path
    report_path: Path
    weights_sha256: str
    report: Mapping[str, Any]


def sha256_file(path: str | Path) -> str:
    """Hash one regular file through a no-follow, mutation-checked descriptor."""

    return _hash_regular_path(Path(path).expanduser())


def floaty_schema_sha256(specs: Sequence[TensorSpec]) -> str:
    ordered = [spec.to_dict() for spec in sorted(specs, key=lambda item: item.name)]
    payload = {
        "schema_version": PROTOCOL_VERSION,
        "namespace": "floaty",
        "tensors": ordered,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def tensor_specs_from_floaty_schema(
    tensors: Sequence[Mapping[str, Any] | TensorSpec],
) -> tuple[TensorSpec, ...]:
    """Strictly parse the tensor list emitted by a floaty config manifest."""

    if (
        isinstance(tensors, (str, bytes))
        or not isinstance(tensors, Sequence)
        or not tensors
    ):
        raise ValueError("floaty tensor schema must be a non-empty sequence")
    specs: list[TensorSpec] = []
    for index, value in enumerate(tensors):
        if isinstance(value, TensorSpec):
            specs.append(value)
            continue
        if not isinstance(value, Mapping) or frozenset(value) != _TENSOR_SPEC_KEYS:
            raise ValueError(
                f"floaty tensor schema entry {index} must contain name, shape, and dtype"
            )
        if not isinstance(value["name"], str) or not isinstance(value["dtype"], str):
            raise ValueError(
                f"floaty tensor schema entry {index} has invalid name or dtype"
            )
        shape = value["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in shape
        ):
            raise ValueError(f"floaty tensor schema entry {index} has an invalid shape")
        specs.append(
            TensorSpec(name=value["name"], shape=tuple(shape), dtype=value["dtype"])
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("floaty tensor schema contains duplicate names")
    return tuple(sorted(specs, key=lambda item: item.name))


def load_community_contract(path: str | Path) -> CommunityContract:
    """Load a bounded, strict JSON contract for offline queue commands."""

    try:
        payload = _snapshot_regular_path(
            Path(path).expanduser(),
            max_bytes=1024 * 1024,
            allow_empty=False,
        )
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError("community contract must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ManifestError("community contract root must be an object")
    return CommunityContract.from_dict(value)


def write_floaty_safetensors(
    path: str | Path,
    tensors: Mapping[str, Any],
    specs: Sequence[TensorSpec],
) -> Path:
    """Atomically write a metadata-free, fixed-schema safetensors file once."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        _write_safetensors_file(temporary, tensors, specs)
        _publish_file_no_replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def read_floaty_safetensors(
    path: str | Path,
    specs: Sequence[TensorSpec],
    *,
    expected_sha256: str | None = None,
    limits: UpdateLimits | None = None,
) -> dict[str, np.ndarray]:
    """Load one pinned floaty file from a bounded, no-follow snapshot.

    When supplied, ``expected_sha256`` is checked against the exact same byte
    snapshot that is decoded, rather than by reopening the path.
    """

    settings = limits or UpdateLimits()
    payload = _snapshot_regular_path(
        Path(path).expanduser(),
        max_bytes=settings.max_delta_bytes,
        allow_empty=False,
    )
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_sha256")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise CompatibilityError("floaty weights do not match expected_sha256")
    return _read_safetensors_payload(
        payload,
        specs,
        max_bytes=settings.max_delta_bytes,
        max_tensor_count=settings.max_tensor_count,
        max_elements=settings.max_tensor_elements,
    )


def write_update_bundle(
    destination: str | Path,
    *,
    contract: CommunityContract,
    update_id: str,
    deltas: Mapping[str, Any],
    example_count: int,
    created_at: datetime | None = None,
) -> UpdateManifest:
    """Create an immutable contribution bundle suitable for quarantine."""

    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        delta_path = temporary / DELTA_FILENAME
        _write_safetensors_file(delta_path, deltas, contract.tensor_specs)
        delta_size = delta_path.stat().st_size
        manifest = UpdateManifest(
            schema_version=PROTOCOL_VERSION,
            update_id=update_id,
            created_at=_format_timestamp(created_at or datetime.now(timezone.utc)),
            foundation_sha256=contract.foundation_sha256,
            community_version=contract.community_version,
            community_weights_sha256=contract.community_weights_sha256,
            architecture_sha256=contract.architecture_sha256,
            tokenizer_sha256=contract.tokenizer_sha256,
            floaty_schema_sha256=contract.floaty_schema_sha256,
            training_recipe_sha256=contract.training_recipe_sha256,
            delta_file=DELTA_FILENAME,
            delta_sha256=sha256_file(delta_path),
            delta_bytes=delta_size,
            example_count=example_count,
        )
        _write_new_bytes(
            temporary / MANIFEST_FILENAME, manifest.canonical_bytes + b"\n"
        )
        _fsync_directory(temporary)
        publish_directory_no_replace(temporary, target)
        _fsync_directory(target.parent)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class ContributionQueue:
    """Append-only quarantine and validation ledger for untrusted updates."""

    def __init__(self, root: str | Path, *, limits: UpdateLimits | None = None) -> None:
        requested = Path(root).expanduser()
        if requested.is_symlink():
            raise UnsafeBundleError("queue root cannot be a symlink")
        requested.mkdir(parents=True, exist_ok=True)
        self.root = requested.resolve()
        self.entries_dir = self.root / "entries"
        if self.entries_dir.is_symlink():
            raise UnsafeBundleError("queue entries directory cannot be a symlink")
        self.entries_dir.mkdir(exist_ok=True)
        self.limits = limits or UpdateLimits()
        self._thread_lock = threading.RLock()
        self._lock_path = self.root / ".queue.lock"

    def quarantine(
        self,
        bundle_dir: str | Path,
        *,
        received_at: datetime | None = None,
    ) -> QueueEntry:
        """Copy a bounded, hash-verified bundle into immutable quarantine."""

        manifest, manifest_payload, delta_payload = self._read_external_bundle(
            bundle_dir
        )
        canonical_delta_sha256 = _canonical_safetensors_sha256(
            delta_payload,
            max_bytes=self.limits.max_delta_bytes,
            max_tensor_count=self.limits.max_tensor_count,
            max_elements=self.limits.max_tensor_elements,
        )
        timestamp = received_at or datetime.now(timezone.utc)
        with self._locked():
            self._reject_replay(manifest, canonical_delta_sha256)
            destination = self.entries_dir / manifest.content_sha256
            temporary = Path(
                tempfile.mkdtemp(prefix=".incoming-", dir=self.entries_dir)
            )
            try:
                _write_new_bytes(temporary / MANIFEST_FILENAME, manifest_payload)
                _write_new_bytes(temporary / DELTA_FILENAME, delta_payload)
                _write_new_json(
                    temporary / "receipt.json",
                    {
                        "schema_version": PROTOCOL_VERSION,
                        "content_sha256": manifest.content_sha256,
                        "manifest_sha256": manifest.manifest_sha256,
                        "manifest_file_sha256": hashlib.sha256(
                            manifest_payload
                        ).hexdigest(),
                        "delta_sha256": manifest.delta_sha256,
                        "canonical_delta_sha256": canonical_delta_sha256,
                        "received_at": _format_timestamp(timestamp),
                    },
                )
                events = temporary / "events"
                events.mkdir()
                _write_new_json(
                    events / "0000-quarantined.json",
                    {
                        "schema_version": PROTOCOL_VERSION,
                        "sequence": 0,
                        "state": "quarantined",
                        "at": _format_timestamp(timestamp),
                    },
                )
                _fsync_directory(events)
                _fsync_directory(temporary)
                publish_directory_no_replace(temporary, destination)
                _fsync_directory(self.entries_dir)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return self.get(manifest.content_sha256)

    def get(self, content_sha256: str) -> QueueEntry:
        path = self._entry_path(content_sha256)
        manifest = self._read_stored_manifest(path)
        if manifest.content_sha256 != content_sha256:
            raise UnsafeBundleError(
                "queue entry content hash does not match its directory"
            )
        return QueueEntry(content_sha256, path, self._current_state(path), manifest)

    def state(self, content_sha256: str) -> str:
        return self.get(content_sha256).state

    def list_entries(self, *, state: str | None = None) -> list[QueueEntry]:
        """List immutable entries, optionally selecting one queue state."""

        if state not in {None, "quarantined", "accepted", "rejected"}:
            raise ValueError("unsupported queue state filter")
        entries: list[QueueEntry] = []
        for candidate in sorted(self.entries_dir.iterdir(), key=lambda item: item.name):
            if candidate.name.startswith("."):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise UnsafeBundleError("queue contains an unsafe entry")
            entry = self.get(candidate.name)
            if state is None or entry.state == state:
                entries.append(entry)
        return entries

    def validate(
        self,
        content_sha256: str,
        contract: CommunityContract,
        *,
        now: datetime | None = None,
    ) -> ValidatedContribution:
        """Validate a quarantined entry and append accepted or rejected state."""

        with self._locked():
            entry = self.get(content_sha256)
            if entry.state != "quarantined":
                raise QueueStateError(f"entry is {entry.state}, not quarantined")
            try:
                validated = self._validate_entry(
                    entry, contract, now=now, check_freshness=True
                )
            except CommunityUpdateError as error:
                self._append_event(
                    entry.path,
                    "rejected",
                    {"code": error.code, "reason": str(error)},
                    at=now,
                )
                raise
            self._append_event(
                entry.path,
                "accepted",
                {"delta_l2_norm": validated.delta_l2_norm},
                at=now,
            )
            accepted = self.get(content_sha256)
            return ValidatedContribution(
                accepted, validated.tensors, validated.delta_l2_norm
            )

    def load_accepted(
        self,
        content_sha256: str,
        contract: CommunityContract,
    ) -> ValidatedContribution:
        """Recheck immutable content before aggregation without reapplying age."""

        entry = self.get(content_sha256)
        if entry.state != "accepted":
            raise QueueStateError(f"entry is {entry.state}, not accepted")
        return self._validate_entry(entry, contract, now=None, check_freshness=False)

    def _read_external_bundle(
        self, bundle_dir: str | Path
    ) -> tuple[UpdateManifest, bytes, bytes]:
        requested = Path(bundle_dir).expanduser()
        with _directory_descriptor(requested) as directory_descriptor:
            names = set(os.listdir(directory_descriptor))
            if names != {MANIFEST_FILENAME, DELTA_FILENAME}:
                raise UnsafeBundleError(
                    "bundle must contain exactly manifest.json and floaty-delta.safetensors"
                )
            try:
                manifest_payload = _snapshot_child_descriptor(
                    directory_descriptor,
                    MANIFEST_FILENAME,
                    max_bytes=self.limits.max_manifest_bytes,
                    allow_empty=False,
                )
            except UnsafeBundleError as error:
                if "empty or oversized" in str(error):
                    raise ManifestError("manifest is empty or oversized") from error
                raise
            try:
                delta_payload = _snapshot_child_descriptor(
                    directory_descriptor,
                    DELTA_FILENAME,
                    max_bytes=self.limits.max_delta_bytes,
                    allow_empty=False,
                )
            except UnsafeBundleError as error:
                if "empty or oversized" in str(error):
                    raise ManifestError(
                        "safetensors delta is empty or oversized"
                    ) from error
                raise
        manifest = UpdateManifest.from_bytes(manifest_payload)
        delta_size = len(delta_payload)
        if delta_size != manifest.delta_bytes:
            raise ManifestError("delta_bytes does not match the safetensors file")
        if manifest.example_count > self.limits.max_example_count:
            raise ManifestError("example_count exceeds the contribution limit")
        if hashlib.sha256(delta_payload).hexdigest() != manifest.delta_sha256:
            raise ManifestError("delta_sha256 does not match the safetensors file")
        return manifest, manifest_payload, delta_payload

    def _validate_entry(
        self,
        entry: QueueEntry,
        contract: CommunityContract,
        *,
        now: datetime | None,
        check_freshness: bool,
    ) -> ValidatedContribution:
        manifest = entry.manifest
        expected = contract.compatibility_dict()
        for name, expected_value in expected.items():
            if getattr(manifest, name) != expected_value:
                if name == "community_version":
                    raise CompatibilityError("stale or unknown community_version")
                raise CompatibilityError(f"{name} does not match the active contract")
        if check_freshness:
            current = _as_utc(now or datetime.now(timezone.utc))
            created = _parse_timestamp(manifest.created_at)
            if created < current - self.limits.max_age:
                raise CompatibilityError("update is stale")
            if created > current + self.limits.max_future_skew:
                raise CompatibilityError("update timestamp is too far in the future")
        delta_payload = _snapshot_child(
            entry.path,
            DELTA_FILENAME,
            max_bytes=self.limits.max_delta_bytes,
            allow_empty=False,
        )
        if len(delta_payload) != manifest.delta_bytes:
            raise TensorValidationError(
                "stored delta size changed or exceeds the limit"
            )
        if hashlib.sha256(delta_payload).hexdigest() != manifest.delta_sha256:
            raise TensorValidationError("stored delta content hash changed")
        tensors = _read_safetensors_payload(
            delta_payload,
            contract.tensor_specs,
            max_bytes=self.limits.max_delta_bytes,
            max_tensor_count=self.limits.max_tensor_count,
            max_elements=self.limits.max_tensor_elements,
        )
        canonical_delta_sha256 = _canonical_safetensors_sha256(
            delta_payload,
            max_bytes=self.limits.max_delta_bytes,
            max_tensor_count=self.limits.max_tensor_count,
            max_elements=self.limits.max_tensor_elements,
        )
        receipt = self._read_stored_receipt(entry.path)
        if receipt.get("canonical_delta_sha256") != canonical_delta_sha256:
            raise TensorValidationError("stored semantic delta identity changed")
        norm = _global_l2_norm(tensors.values())
        if not math.isfinite(norm):
            raise TensorValidationError("delta contains non-finite values")
        if norm > self.limits.max_delta_l2_norm:
            raise TensorValidationError("delta L2 norm exceeds the hard outlier limit")
        return ValidatedContribution(entry, tensors, norm)

    def _read_stored_manifest(self, path: Path) -> UpdateManifest:
        with _directory_descriptor(path) as directory_descriptor:
            manifest_payload = _snapshot_child_descriptor(
                directory_descriptor,
                MANIFEST_FILENAME,
                max_bytes=self.limits.max_manifest_bytes,
                allow_empty=False,
            )
            receipt_payload = _snapshot_child_descriptor(
                directory_descriptor,
                "receipt.json",
                max_bytes=self.limits.max_manifest_bytes,
                allow_empty=False,
            )
        manifest = UpdateManifest.from_bytes(manifest_payload)
        receipt = _json_object_from_bytes(receipt_payload, label="queue receipt")
        if (
            receipt.get("manifest_sha256") != manifest.manifest_sha256
            or receipt.get("manifest_file_sha256")
            != hashlib.sha256(manifest_payload).hexdigest()
        ):
            raise UnsafeBundleError("stored manifest differs from its receipt")
        return manifest

    def _read_stored_receipt(self, path: Path) -> dict[str, Any]:
        receipt_payload = _snapshot_child(
            path,
            "receipt.json",
            max_bytes=self.limits.max_manifest_bytes,
            allow_empty=False,
        )
        return _json_object_from_bytes(receipt_payload, label="queue receipt")

    def _entry_path(self, content_sha256: str) -> Path:
        _require_sha256(content_sha256, "content_sha256")
        path = self.entries_dir / content_sha256
        if (
            path.is_symlink()
            or not path.is_dir()
            or path.resolve().parent != self.entries_dir
        ):
            raise FileNotFoundError(content_sha256)
        return path.resolve()

    def _current_state(self, entry_path: Path) -> str:
        events_dir = entry_path / "events"
        if events_dir.is_symlink() or not events_dir.is_dir():
            raise UnsafeBundleError("queue event ledger is missing or unsafe")
        events = sorted(events_dir.glob("[0-9][0-9][0-9][0-9]-*.json"))
        if not events:
            raise UnsafeBundleError("queue event ledger is empty")
        expected_sequence = 0
        state = ""
        for event_path in events:
            if event_path.is_symlink() or not event_path.is_file():
                raise UnsafeBundleError("unsafe queue event")
            event = _read_json_object(event_path)
            if event.get("sequence") != expected_sequence:
                raise UnsafeBundleError("queue event sequence is not contiguous")
            candidate = event.get("state")
            if expected_sequence == 0 and candidate != "quarantined":
                raise UnsafeBundleError("first queue state must be quarantined")
            if expected_sequence == 1 and candidate not in {"accepted", "rejected"}:
                raise UnsafeBundleError("invalid terminal queue state")
            if expected_sequence > 1:
                raise UnsafeBundleError("queue entry has multiple terminal states")
            state = str(candidate)
            expected_sequence += 1
        return state

    def _append_event(
        self,
        entry_path: Path,
        state: str,
        details: Mapping[str, Any],
        *,
        at: datetime | None,
    ) -> None:
        if state not in {"accepted", "rejected"}:
            raise QueueStateError("unsupported queue transition")
        if self._current_state(entry_path) != "quarantined":
            raise QueueStateError("queue transition requires quarantined state")
        _write_new_json(
            entry_path / "events" / f"0001-{state}.json",
            {
                "schema_version": PROTOCOL_VERSION,
                "sequence": 1,
                "state": state,
                "at": _format_timestamp(at or datetime.now(timezone.utc)),
                "details": dict(details),
            },
        )
        _fsync_directory(entry_path / "events")

    def _reject_replay(
        self,
        manifest: UpdateManifest,
        canonical_delta_sha256: str,
    ) -> None:
        for candidate in self.entries_dir.iterdir():
            if candidate.name.startswith("."):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise UnsafeBundleError("queue contains an unsafe entry")
            previous = self._read_stored_manifest(candidate.resolve())
            if previous.update_id == manifest.update_id:
                raise ReplayError("update_id has already been submitted")
            if previous.delta_sha256 == manifest.delta_sha256:
                raise ReplayError("delta content has already been submitted")
            receipt = self._read_stored_receipt(candidate.resolve())
            previous_canonical = receipt.get("canonical_delta_sha256")
            if previous_canonical is None:
                previous_payload = _snapshot_child(
                    candidate.resolve(),
                    DELTA_FILENAME,
                    max_bytes=self.limits.max_delta_bytes,
                    allow_empty=False,
                )
                previous_canonical = _canonical_safetensors_sha256(
                    previous_payload,
                    max_bytes=self.limits.max_delta_bytes,
                    max_tensor_count=self.limits.max_tensor_count,
                    max_elements=self.limits.max_tensor_elements,
                )
            if previous_canonical == canonical_delta_sha256:
                raise ReplayError("semantic delta content has already been submitted")
            if previous.content_sha256 == manifest.content_sha256:
                raise ReplayError("contribution content has already been submitted")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                try:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                except ImportError:
                    pass
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except ImportError:
                    pass
                os.close(descriptor)


def aggregate_candidate(
    queue: ContributionQueue,
    content_hashes: Sequence[str],
    *,
    contract: CommunityContract,
    base_floaty_path: str | Path,
    output_dir: str | Path,
) -> AggregationResult:
    """Create a clipped, influence-capped floaty candidate for manual review.

    This function intentionally writes no active pointer, signature, release,
    or promoted community version.
    """

    unique_hashes = sorted(set(content_hashes))
    if len(unique_hashes) != len(content_hashes):
        raise ReplayError("aggregation input contains a repeated contribution")
    if len(unique_hashes) < queue.limits.min_aggregate_updates:
        raise ValueError(
            f"at least {queue.limits.min_aggregate_updates} accepted contributions are required"
        )

    base_path = Path(base_floaty_path).expanduser()
    if base_path.suffix != ".safetensors":
        raise UnsafeBundleError("base floaty weights must be a real file")
    base_payload = _snapshot_regular_path(
        base_path,
        max_bytes=queue.limits.max_delta_bytes,
        allow_empty=False,
    )
    if hashlib.sha256(base_payload).hexdigest() != contract.community_weights_sha256:
        raise CompatibilityError(
            "base floaty weights do not match the community contract"
        )
    base = _read_safetensors_payload(
        base_payload,
        contract.tensor_specs,
        max_bytes=queue.limits.max_delta_bytes,
        max_tensor_count=queue.limits.max_tensor_count,
        max_elements=queue.limits.max_tensor_elements,
    )
    contributions = [
        queue.load_accepted(identifier, contract) for identifier in unique_hashes
    ]

    accumulators = {
        name: np.zeros(spec.shape, dtype=np.float64)
        for name, spec in contract.specs_by_name.items()
    }
    contribution_reports: list[dict[str, Any]] = []
    total_weight = 0.0
    for contribution in contributions:
        norm = contribution.delta_l2_norm
        clip_scale = (
            min(1.0, queue.limits.aggregation_clip_l2_norm / norm) if norm else 1.0
        )
        raw_weight = contribution.entry.manifest.example_count
        capped_weight = min(raw_weight, queue.limits.max_weight_per_update)
        effective_weight = float(capped_weight)
        for name, tensor in contribution.tensors.items():
            accumulators[name] += np.asarray(tensor, dtype=np.float64) * (
                clip_scale * effective_weight
            )
        total_weight += effective_weight
        contribution_reports.append(
            {
                "content_sha256": contribution.entry.content_sha256,
                "update_id": contribution.entry.manifest.update_id,
                "delta_sha256": contribution.entry.manifest.delta_sha256,
                "delta_l2_norm": norm,
                "clip_scale": clip_scale,
                "reported_example_count": raw_weight,
                "capped_weight": capped_weight,
            }
        )
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise TensorValidationError("aggregate has no finite positive weight")

    candidate_tensors: dict[str, np.ndarray] = {}
    aggregate_delta_norm_parts: list[np.ndarray] = []
    for spec in contract.tensor_specs:
        aggregate_delta = accumulators[spec.name] / total_weight
        candidate = np.asarray(base[spec.name], dtype=np.float64) + aggregate_delta
        if not np.all(np.isfinite(candidate)):
            raise TensorValidationError("candidate contains non-finite weights")
        candidate_tensors[spec.name] = candidate
        aggregate_delta_norm_parts.append(aggregate_delta)
    aggregate_delta_norm = _global_l2_norm(aggregate_delta_norm_parts)

    identity = {
        "schema_version": PROTOCOL_VERSION,
        "foundation_sha256": contract.foundation_sha256,
        "source_community_version": contract.community_version,
        "source_community_weights_sha256": contract.community_weights_sha256,
        "floaty_schema_sha256": contract.floaty_schema_sha256,
        "contributions": unique_hashes,
        "aggregation": {
            "method": "l2-clipped-example-weighted-mean-v1",
            "clip_l2_norm": queue.limits.aggregation_clip_l2_norm,
            "max_weight_per_update": queue.limits.max_weight_per_update,
        },
    }
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    candidate_id = f"floaty-candidate-{identity_sha256[:24]}"
    parent = Path(output_dir).expanduser()
    if parent.is_symlink():
        raise UnsafeBundleError("candidate output directory cannot be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve()
    destination = parent / candidate_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".floaty-candidate-", dir=parent))
    try:
        weights_path = temporary / CANDIDATE_WEIGHTS_FILENAME
        _write_safetensors_file(weights_path, candidate_tensors, contract.tensor_specs)
        weights_sha256 = sha256_file(weights_path)
        report: dict[str, Any] = {
            **identity,
            "candidate_id": candidate_id,
            "candidate_identity_sha256": identity_sha256,
            "created_at": _format_timestamp(datetime.now(timezone.utc)),
            "candidate_weights": CANDIDATE_WEIGHTS_FILENAME,
            "candidate_weights_sha256": weights_sha256,
            "contribution_count": len(contributions),
            "total_capped_weight": total_weight,
            "aggregate_delta_l2_norm": aggregate_delta_norm,
            "contribution_report": contribution_reports,
            "promotion": {
                "status": "manual_review_required",
                "auto_promoted": False,
                "active_pointer_written": False,
                "assigned_community_version": None,
            },
        }
        report_path = temporary / CANDIDATE_REPORT_FILENAME
        _write_new_json(report_path, report)
        _fsync_directory(temporary)
        publish_directory_no_replace(temporary, destination)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return AggregationResult(
        candidate_id=candidate_id,
        candidate_dir=destination.resolve(),
        weights_path=(destination / CANDIDATE_WEIGHTS_FILENAME).resolve(),
        report_path=(destination / CANDIDATE_REPORT_FILENAME).resolve(),
        weights_sha256=weights_sha256,
        report=report,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Offline CLI for quarantine, validation, and candidate aggregation."""

    parser = argparse.ArgumentParser(
        prog="mlm-community-updates",
        description="Manage immutable floaty-layer contribution bundles without promotion.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    quarantine_parser = commands.add_parser("quarantine", help="quarantine one bundle")
    quarantine_parser.add_argument("--queue", required=True, help="queue directory")
    quarantine_parser.add_argument(
        "--bundle", required=True, help="contribution bundle directory"
    )

    validate_parser = commands.add_parser(
        "validate", help="validate one quarantined entry"
    )
    validate_parser.add_argument("--queue", required=True, help="queue directory")
    validate_parser.add_argument(
        "--contract", required=True, help="strict community contract JSON"
    )
    validate_parser.add_argument(
        "--entry", required=True, help="contribution content SHA-256"
    )

    aggregate_parser = commands.add_parser(
        "aggregate", help="create a manual-review candidate from accepted entries"
    )
    aggregate_parser.add_argument("--queue", required=True, help="queue directory")
    aggregate_parser.add_argument(
        "--contract", required=True, help="strict community contract JSON"
    )
    aggregate_parser.add_argument(
        "--base-floaty", required=True, help="active floaty safetensors"
    )
    aggregate_parser.add_argument(
        "--output", required=True, help="immutable candidate parent directory"
    )
    aggregate_parser.add_argument(
        "entries", nargs="+", help="accepted contribution content hashes"
    )

    arguments = parser.parse_args(argv)
    try:
        queue = ContributionQueue(arguments.queue)
        if arguments.command == "quarantine":
            entry = queue.quarantine(arguments.bundle)
            output = {
                "content_sha256": entry.content_sha256,
                "state": entry.state,
                "update_id": entry.manifest.update_id,
            }
        elif arguments.command == "validate":
            contract = load_community_contract(arguments.contract)
            contribution = queue.validate(arguments.entry, contract)
            output = {
                "content_sha256": contribution.entry.content_sha256,
                "state": contribution.entry.state,
                "delta_l2_norm": contribution.delta_l2_norm,
            }
        else:
            contract = load_community_contract(arguments.contract)
            result = aggregate_candidate(
                queue,
                arguments.entries,
                contract=contract,
                base_floaty_path=arguments.base_floaty,
                output_dir=arguments.output,
            )
            output = {
                "candidate_id": result.candidate_id,
                "candidate_dir": str(result.candidate_dir),
                "weights_sha256": result.weights_sha256,
                "promotion_status": result.report["promotion"]["status"],
            }
    except (CommunityUpdateError, OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(_canonical_json(output).decode("utf-8"))
    return 0


def _canonical_safetensors_sha256(
    payload: bytes,
    *,
    max_bytes: int,
    max_tensor_count: int,
    max_elements: int,
) -> str:
    """Hash tensor meaning, independent of safetensors header formatting/order."""

    if len(payload) <= 8 or len(payload) > max_bytes:
        raise TensorValidationError("safetensors file is empty or oversized")
    try:
        header_size = struct.unpack("<Q", payload[:8])[0]
    except struct.error as error:
        raise TensorValidationError("truncated safetensors header") from error
    if header_size < 2 or header_size > min(max_bytes, 8 * 1024 * 1024):
        raise TensorValidationError("invalid safetensors header size")
    data_start = 8 + header_size
    if data_start > len(payload):
        raise TensorValidationError("truncated safetensors header")
    try:
        header = json.loads(
            payload[8:data_start].decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ManifestError) as error:
        raise TensorValidationError("invalid safetensors JSON header") from error
    if not isinstance(header, dict) or not header or "__metadata__" in header:
        raise TensorValidationError(
            "safetensors must contain tensors only, without metadata"
        )
    if len(header) > max_tensor_count:
        raise TensorValidationError("safetensors contains too many tensors")

    parsed: dict[str, tuple[str, list[int], int, int]] = {}
    regions: list[tuple[int, int]] = []
    total_elements = 0
    for name, item in header.items():
        if (
            not isinstance(name, str)
            or not _TENSOR_RE.fullmatch(name)
            or ".." in name
            or "/" in name
            or "\\" in name
        ):
            raise TensorValidationError("safetensors contains an unsafe tensor name")
        if not isinstance(item, dict) or set(item) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise TensorValidationError(f"invalid header entry for {name}")
        dtype = item["dtype"]
        shape = item["shape"]
        offsets = item["data_offsets"]
        if dtype not in _DTYPE_BYTES:
            raise TensorValidationError(f"unsupported tensor dtype for {name}")
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise TensorValidationError(f"invalid tensor shape for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
        ):
            raise TensorValidationError(f"invalid data offsets for {name}")
        element_count = math.prod(shape)
        total_elements += element_count
        if total_elements > max_elements:
            raise TensorValidationError("safetensors has too many tensor elements")
        start, end = offsets
        if (
            start < 0
            or end < start
            or end - start != element_count * _DTYPE_BYTES[dtype]
        ):
            raise TensorValidationError(f"invalid byte length for {name}")
        regions.append((start, end))
        parsed[name] = (dtype, shape, start, end)

    cursor = 0
    for start, end in sorted(regions):
        if start != cursor:
            raise TensorValidationError(
                "tensor data must be contiguous and non-overlapping"
            )
        cursor = end
    if cursor != len(payload) - data_start:
        raise TensorValidationError("safetensors data length does not match its header")

    hasher = hashlib.sha256()
    prefix = b"tinydog.floaty.semantic-delta.v1"
    hasher.update(len(prefix).to_bytes(8, "big"))
    hasher.update(prefix)
    for name in sorted(parsed):
        dtype, shape, start, end = parsed[name]
        descriptor = _canonical_json({"name": name, "dtype": dtype, "shape": shape})
        raw = payload[data_start + start : data_start + end]
        hasher.update(len(descriptor).to_bytes(8, "big"))
        hasher.update(descriptor)
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
    return hasher.hexdigest()


def _read_safetensors_payload(
    payload: bytes,
    specs: Sequence[TensorSpec],
    *,
    max_bytes: int,
    max_tensor_count: int,
    max_elements: int,
) -> dict[str, np.ndarray]:
    if len(payload) <= 8 or len(payload) > max_bytes:
        raise TensorValidationError("safetensors file is empty or oversized")
    if len(specs) > max_tensor_count:
        raise TensorValidationError("floaty schema has too many tensors")
    total_elements = sum(spec.element_count for spec in specs)
    if total_elements > max_elements:
        raise TensorValidationError("floaty schema has too many elements")
    expected = _spec_map(specs)
    try:
        header_size = struct.unpack("<Q", payload[:8])[0]
    except struct.error as error:
        raise TensorValidationError("truncated safetensors header") from error
    if header_size < 2 or header_size > min(max_bytes, 8 * 1024 * 1024):
        raise TensorValidationError("invalid safetensors header size")
    if 8 + header_size > len(payload):
        raise TensorValidationError("truncated safetensors header")
    header_bytes = payload[8 : 8 + header_size]
    try:
        header = json.loads(
            header_bytes.decode("utf-8"), object_pairs_hook=_unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ManifestError) as error:
        raise TensorValidationError("invalid safetensors JSON header") from error
    if not isinstance(header, dict) or "__metadata__" in header:
        raise TensorValidationError(
            "safetensors must contain tensors only, without metadata"
        )
    if len(header) > max_tensor_count:
        raise TensorValidationError("safetensors contains too many tensors")
    for name in header:
        if (
            not _TENSOR_RE.fullmatch(name)
            or ".." in name
            or "/" in name
            or "\\" in name
        ):
            raise TensorValidationError("safetensors contains an unsafe tensor name")
    if set(header) != set(expected):
        unknown = sorted(set(header) - set(expected))
        missing = sorted(set(expected) - set(header))
        if any(not name.startswith("floaty.") for name in unknown):
            raise TensorValidationError("foundation or non-floaty tensor is forbidden")
        raise TensorValidationError(
            f"tensor names do not match the floaty schema; missing={missing}, extra={unknown}"
        )

    regions: list[tuple[int, int, str]] = []
    parsed: dict[str, tuple[TensorSpec, int, int]] = {}
    for name, item in header.items():
        if not isinstance(item, dict) or set(item) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise TensorValidationError(f"invalid header entry for {name}")
        spec = expected[name]
        shape = item["shape"]
        if (
            not isinstance(item["dtype"], str)
            or not isinstance(shape, list)
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int)
                for dimension in shape
            )
            or item["dtype"] != spec.dtype
            or shape != list(spec.shape)
        ):
            raise TensorValidationError(f"dtype or shape mismatch for {name}")
        offsets = item["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
        ):
            raise TensorValidationError(f"invalid data offsets for {name}")
        start, end = offsets
        if start < 0 or end < start or end - start != spec.byte_count:
            raise TensorValidationError(f"invalid byte length for {name}")
        regions.append((start, end, name))
        parsed[name] = (spec, start, end)

    cursor = 0
    for start, end, _ in sorted(regions):
        if start != cursor:
            raise TensorValidationError(
                "tensor data must be contiguous and non-overlapping"
            )
        cursor = end
    data_start = 8 + header_size
    data_size = len(payload) - data_start
    if cursor != data_size:
        raise TensorValidationError("safetensors data length does not match its header")

    tensors: dict[str, np.ndarray] = {}
    for name in sorted(parsed):
        spec, start, end = parsed[name]
        raw = payload[data_start + start : data_start + end]
        if len(raw) != end - start:
            raise TensorValidationError(f"truncated tensor data for {name}")
        tensors[name] = _decode_tensor(raw, spec)
    for name, value in tensors.items():
        if not np.all(np.isfinite(value)):
            raise TensorValidationError(f"non-finite tensor values in {name}")
    return tensors


def _write_safetensors_file(
    path: Path,
    tensors: Mapping[str, Any],
    specs: Sequence[TensorSpec],
) -> None:
    expected = _spec_map(specs)
    if set(tensors) != set(expected):
        unknown = sorted(set(tensors) - set(expected))
        missing = sorted(set(expected) - set(tensors))
        if any(not str(name).startswith("floaty.") for name in unknown):
            raise TensorValidationError("foundation or non-floaty tensor is forbidden")
        raise TensorValidationError(
            f"tensor names do not match schema; missing={missing}, extra={unknown}"
        )
    encoded: dict[str, bytes] = {}
    header: dict[str, Any] = {}
    offset = 0
    for name in sorted(expected):
        spec = expected[name]
        payload = _encode_tensor(tensors[name], spec)
        encoded[name] = payload
        header[name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    header_bytes = _canonical_json(header)
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for name in sorted(encoded):
            stream.write(encoded[name])
        stream.flush()
        os.fsync(stream.fileno())


def _encode_tensor(value: Any, spec: TensorSpec) -> bytes:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise TensorValidationError(
            f"tensor {spec.name} cannot be converted to an array"
        ) from error
    if tuple(array.shape) != spec.shape:
        raise TensorValidationError(f"shape mismatch for {spec.name}")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise TensorValidationError(
            f"tensor {spec.name} must contain real numeric values"
        )
    try:
        finite = np.all(np.isfinite(array))
    except TypeError as error:
        raise TensorValidationError(f"tensor {spec.name} is not numeric") from error
    if not bool(finite):
        raise TensorValidationError(f"non-finite tensor values in {spec.name}")
    if spec.dtype == "F32":
        try:
            cast = np.asarray(array, dtype="<f4")
        except (OverflowError, TypeError, ValueError) as error:
            raise TensorValidationError(
                f"tensor {spec.name} cannot be represented as F32"
            ) from error
        if not np.all(np.isfinite(cast)):
            raise TensorValidationError(f"tensor {spec.name} overflows F32")
        return cast.tobytes(order="C")
    if spec.dtype == "F16":
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                cast = np.asarray(array, dtype="<f2")
        except (OverflowError, TypeError, ValueError) as error:
            raise TensorValidationError(
                f"tensor {spec.name} cannot be represented as F16"
            ) from error
        if not np.all(np.isfinite(cast)):
            raise TensorValidationError(f"tensor {spec.name} overflows F16")
        return cast.tobytes(order="C")
    try:
        values = np.asarray(array, dtype="<f4")
    except (OverflowError, TypeError, ValueError) as error:
        raise TensorValidationError(
            f"tensor {spec.name} cannot be represented as BF16"
        ) from error
    if not np.all(np.isfinite(values)):
        raise TensorValidationError(f"tensor {spec.name} overflows BF16")
    bits = values.view("<u4").copy()
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = bits + rounding_bias
    return (rounded >> np.uint32(16)).astype("<u2").tobytes(order="C")


def _decode_tensor(raw: bytes, spec: TensorSpec) -> np.ndarray:
    if spec.dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(spec.shape).copy()
    if spec.dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(spec.shape).astype(np.float32)
    upper = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
    return upper.view("<f4").reshape(spec.shape).copy()


def _global_l2_norm(tensors: Sequence[np.ndarray] | Any) -> float:
    values = list(tensors)
    if not values:
        return 0.0
    maximum = max(
        float(np.max(np.abs(value))) if value.size else 0.0 for value in values
    )
    if not math.isfinite(maximum):
        return math.inf
    if maximum == 0.0:
        return 0.0
    scaled_sum = 0.0
    for value in values:
        scaled = np.asarray(value, dtype=np.float64) / maximum
        scaled_sum += float(np.sum(scaled * scaled, dtype=np.float64))
    return maximum * math.sqrt(scaled_sum)


def _spec_map(specs: Sequence[TensorSpec]) -> dict[str, TensorSpec]:
    if not specs or any(not isinstance(spec, TensorSpec) for spec in specs):
        raise TensorValidationError("floaty schema must contain TensorSpec values")
    expected = {spec.name: spec for spec in specs}
    if len(expected) != len(specs):
        raise TensorValidationError("floaty schema contains duplicate tensor names")
    return expected


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise ManifestError("created_at must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ManifestError("created_at is not a valid timestamp") from error
    return parsed.astimezone(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = _snapshot_regular_path(path, max_bytes=1024 * 1024, allow_empty=False)
    return _json_object_from_bytes(payload, label=path.name)


def _json_object_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ManifestError) as error:
        raise UnsafeBundleError(f"invalid queue JSON: {label}") from error
    if not isinstance(value, dict):
        raise UnsafeBundleError(f"queue JSON must be an object: {label}")
    return value


@contextmanager
def _directory_descriptor(path: Path) -> Iterator[int]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise UnsafeBundleError(
            f"directory is missing, not a directory, or a symlink: {path}"
        ) from error
    try:
        if not stat_module.S_ISDIR(os.fstat(descriptor).st_mode):
            raise UnsafeBundleError(f"path is not a directory: {path}")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _regular_child_descriptor(directory_descriptor: int, name: str) -> Iterator[int]:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise UnsafeBundleError("unsafe file path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        raise UnsafeBundleError(
            f"file is missing, not regular, or a symlink: {name}"
        ) from error
    try:
        if not stat_module.S_ISREG(os.fstat(descriptor).st_mode):
            raise UnsafeBundleError(f"file is not regular: {name}")
        yield descriptor
    finally:
        os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_child_descriptor(
    directory_descriptor: int,
    name: str,
    *,
    max_bytes: int,
    allow_empty: bool,
) -> bytes:
    if max_bytes <= 0:
        raise ValueError("snapshot byte bound must be positive")
    with _regular_child_descriptor(directory_descriptor, name) as descriptor:
        before = os.fstat(descriptor)
        if before.st_size > max_bytes or (before.st_size == 0 and not allow_empty):
            raise UnsafeBundleError(f"file is empty or oversized: {name}")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise UnsafeBundleError(f"file changed while being read: {name}")
            chunks.append(chunk)
            remaining -= len(chunk)
        extra = os.read(descriptor, 1)
        after = os.fstat(descriptor)
        if extra or _stat_identity(before) != _stat_identity(after):
            raise UnsafeBundleError(f"file changed while being read: {name}")
        return b"".join(chunks)


def _snapshot_child(
    directory: Path,
    name: str,
    *,
    max_bytes: int,
    allow_empty: bool,
) -> bytes:
    with _directory_descriptor(directory) as descriptor:
        return _snapshot_child_descriptor(
            descriptor,
            name,
            max_bytes=max_bytes,
            allow_empty=allow_empty,
        )


def _snapshot_regular_path(path: Path, *, max_bytes: int, allow_empty: bool) -> bytes:
    with _directory_descriptor(path.parent) as directory_descriptor:
        return _snapshot_child_descriptor(
            directory_descriptor,
            path.name,
            max_bytes=max_bytes,
            allow_empty=allow_empty,
        )


def _hash_regular_path(path: Path) -> str:
    digest = hashlib.sha256()
    with _directory_descriptor(path.parent) as directory_descriptor:
        with _regular_child_descriptor(directory_descriptor, path.name) as descriptor:
            before = os.fstat(descriptor)
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(descriptor)
            if _stat_identity(before) != _stat_identity(after):
                raise UnsafeBundleError(f"file changed while being hashed: {path.name}")
    return digest.hexdigest()


def _write_new_json(path: Path, value: Any) -> None:
    _write_new_bytes(path, _canonical_json(value) + b"\n")


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_file_no_replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_file_no_replace(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError as error:
        raise RuntimeError(
            "atomic no-replace file publication is unavailable"
        ) from error
    source.unlink()


def publish_directory_no_replace(source: str | Path, destination: str | Path) -> Path:
    """Atomically move a completed directory while refusing any destination."""

    source = Path(source)
    destination = Path(destination)
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
            -100, source_bytes, -100, destination_bytes, 0x00000001
        )
    elif os.name == "nt":
        os.rename(source, destination)
        return destination
    else:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    if result == 0:
        return destination
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, "destination already exists", str(destination)
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
