"""Private direct-foundation learning from trusted local conversation.

This v2 lineage is deliberately separate from the legacy personal-adapter
registry.  A backend trains and returns a complete portable foundation bundle;
the coordinator publishes an immutable copy and atomically hot-activates it.
No artifact, state file, or protocol in this module can create an upload.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import threading
import time
from typing import Any, Protocol, runtime_checkable
import uuid

from .apprenticeship import ApprenticeshipStore, Lesson
from .personal_training import (
    DialogueExample,
    _atomic_private_json,
    _dialogue_content_sha256,
    _dialogue_example,
    _ensure_private_directory,
    _fsync_directory,
    _is_personal_dialogue_candidate,
    _is_typed_dialogue_lesson,
    _make_private,
    _read_regular_file,
)


PERSONAL_FOUNDATION_FORMAT = "tinydog-personal-foundation-v2"
PERSONAL_FOUNDATION_STATE_FORMAT = "tinydog-personal-foundation-state-v2"
PERSONAL_FOUNDATION_THRESHOLD = 5
PERSONAL_FOUNDATION_MONITORING_EXAMPLES = 1
PERSONAL_FOUNDATION_DEFAULT_REPLAY_EXAMPLES = 8
PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES = 32
PERSONAL_FOUNDATION_REPLAY_SCAN_LIMIT = 2_048
PERSONAL_FOUNDATION_MAX_FILES = 256
PERSONAL_FOUNDATION_MANIFEST_BYTES = 64 * 1024

_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PersonalFoundationError(RuntimeError):
    """A direct personal-foundation boundary failed safely."""


def personal_foundation_backend_contract_sha256(backend: str) -> str:
    """Stable v1 hash for one backend's canonical full-foundation seam."""

    if not isinstance(backend, str) or not _IDENTIFIER_RE.fullmatch(backend):
        raise PersonalFoundationError("backend is invalid")
    return hashlib.sha256(
        f"tinydog-personal-foundation-backend-v1\0{backend}".encode("ascii")
    ).hexdigest()


def personal_foundation_bundle_sha256(bundle_path: str | Path) -> str:
    """Hash every directory and regular file in a complete bundle.

    Framing is versioned and unambiguous: entries are sorted by UTF-8 POSIX
    relative path; directories contribute ``D`` plus path length/path, while
    files contribute ``F`` plus path length/path, byte length, and all bytes.
    Symlinks and special files are rejected and nothing is excluded.
    """

    candidate = Path(bundle_path).expanduser()
    try:
        root_details = candidate.lstat()
    except OSError as error:
        raise PersonalFoundationError(
            "foundation bundle path is not a directory"
        ) from error
    if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
        raise PersonalFoundationError("foundation bundle path is not a directory")
    root = candidate.resolve()
    entries = sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    )
    if len(entries) > PERSONAL_FOUNDATION_MAX_FILES:
        raise PersonalFoundationError("foundation bundle has too many entries")
    digest = hashlib.sha256(b"tinydog-personal-foundation-bundle-v1\0")
    for entry in entries:
        details = entry.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise PersonalFoundationError("foundation bundle cannot contain symlinks")
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        digest.update(b"D" if stat.S_ISDIR(details.st_mode) else b"F")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if stat.S_ISDIR(details.st_mode):
            continue
        if not stat.S_ISREG(details.st_mode):
            raise PersonalFoundationError("foundation bundle contains a special file")
        descriptor = os.open(
            entry,
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != details.st_dev
                or opened.st_ino != details.st_ino
                or opened.st_size != details.st_size
            ):
                raise PersonalFoundationError("foundation bundle changed while hashing")
            digest.update(opened.st_size.to_bytes(8, "big"))
            bytes_read = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                bytes_read += len(chunk)
                digest.update(chunk)
            if bytes_read != opened.st_size:
                raise PersonalFoundationError("foundation bundle changed while hashing")
        finally:
            os.close(descriptor)
    return digest.hexdigest()


def personal_foundation_profile_id(identity: PersonalFoundationIdentity) -> str:
    """Stable key anchored to one initial immutable foundation lineage."""

    identity.validate()
    payload = "\0".join(
        (
            "tinydog-personal-foundation-profile-v1",
            identity.backend,
            identity.config_sha256,
            identity.tokenizer_sha256,
            identity.backend_contract_sha256,
            identity.bundle_sha256,
        )
    ).encode("ascii")
    return f"foundation-{hashlib.sha256(payload).hexdigest()[:24]}"


@dataclass(frozen=True)
class PersonalFoundationIdentity:
    """Complete immutable compatibility identity for one portable bundle."""

    backend: str
    foundation_id: str
    bundle_sha256: str
    weights_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    backend_contract_sha256: str

    def validate(self) -> None:
        for value, label in (
            (self.backend, "backend"),
            (self.foundation_id, "foundation_id"),
        ):
            if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
                raise PersonalFoundationError(f"{label} is invalid")
        for value, label in (
            (self.bundle_sha256, "bundle_sha256"),
            (self.weights_sha256, "weights_sha256"),
            (self.config_sha256, "config_sha256"),
            (self.tokenizer_sha256, "tokenizer_sha256"),
            (self.backend_contract_sha256, "backend_contract_sha256"),
        ):
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise PersonalFoundationError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True)
class PersonalFoundationTrainingConfig:
    enabled: bool = True
    artifact_root: Path = Path("data/personal-foundations")
    maximum_replay_examples: int = PERSONAL_FOUNDATION_DEFAULT_REPLAY_EXAMPLES
    maximum_bundle_bytes: int = 8 * 1024 * 1024 * 1024
    failure_backoff_seconds: float = 300.0
    profile_id: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        project_root: str | Path | None = None,
    ) -> PersonalFoundationTrainingConfig:
        """Build the zero-config local-foundation policy from explicit v2 keys."""

        root = Path(project_root or Path.cwd()).expanduser().resolve()
        enabled_text = os.getenv("MLM_PERSONAL_FOUNDATION_AUTO", "1").strip().lower()
        true_values = {"1", "true", "yes", "on"}
        false_values = {"0", "false", "no", "off"}
        if enabled_text not in true_values | false_values:
            raise ValueError("MLM_PERSONAL_FOUNDATION_AUTO must be true or false")
        configured_root = (
            os.getenv(
                "MLM_PERSONAL_FOUNDATION_DIR",
                "data/personal-foundations",
            ).strip()
            or "data/personal-foundations"
        )
        candidate = Path(configured_root).expanduser()
        artifact_root = candidate if candidate.is_absolute() else root / candidate
        profile_id = os.getenv("MLM_PERSONAL_FOUNDATION_PROFILE_ID", "").strip() or None
        try:
            config = cls(
                enabled=enabled_text in true_values,
                artifact_root=artifact_root.resolve(),
                maximum_replay_examples=int(
                    os.getenv(
                        "MLM_PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES",
                        str(PERSONAL_FOUNDATION_DEFAULT_REPLAY_EXAMPLES),
                    ).strip()
                ),
                maximum_bundle_bytes=int(
                    os.getenv(
                        "MLM_PERSONAL_FOUNDATION_MAX_BUNDLE_BYTES",
                        str(8 * 1024 * 1024 * 1024),
                    ).strip()
                ),
                failure_backoff_seconds=float(
                    os.getenv(
                        "MLM_PERSONAL_FOUNDATION_FAILURE_BACKOFF_SECONDS",
                        "300",
                    ).strip()
                ),
                profile_id=profile_id,
            )
        except ValueError as error:
            raise ValueError(
                "MLM_PERSONAL_FOUNDATION_* numeric setting is invalid"
            ) from error
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise PersonalFoundationError("foundation training enabled flag is invalid")
        try:
            Path(self.artifact_root)
        except TypeError as error:
            raise PersonalFoundationError(
                "foundation artifact root is invalid"
            ) from error
        if (
            type(self.maximum_replay_examples) is not int
            or not 0
            <= self.maximum_replay_examples
            <= PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES
        ):
            raise PersonalFoundationError("foundation replay bound is invalid")
        if type(self.maximum_bundle_bytes) is not int or self.maximum_bundle_bytes <= 0:
            raise PersonalFoundationError("maximum foundation bundle size is invalid")
        if (
            isinstance(self.failure_backoff_seconds, bool)
            or not isinstance(self.failure_backoff_seconds, (int, float))
            or not math.isfinite(float(self.failure_backoff_seconds))
            or self.failure_backoff_seconds < 0
        ):
            raise PersonalFoundationError("foundation failure backoff is invalid")
        if self.profile_id is not None and (
            not isinstance(self.profile_id, str)
            or not _IDENTIFIER_RE.fullmatch(self.profile_id)
        ):
            raise PersonalFoundationError("foundation profile_id is invalid")


@dataclass(frozen=True)
class PersonalFoundationSource:
    """A checksum-identified complete portable bundle available to a backend."""

    bundle_path: Path
    identity: PersonalFoundationIdentity


@dataclass(frozen=True)
class PersonalFoundationTrainingRequest:
    """Exact private data and immutable parent for one direct training run.

    All five ``new_examples`` and every replay example are optimizer data.
    ``monitoring_examples`` is a deterministic subset of the five and exists
    only for clearly labelled in-sample telemetry; it is not a holdout or a
    promotion gate.
    """

    run_id: int
    candidate_foundation_id: str
    parent: PersonalFoundationSource
    new_examples: tuple[DialogueExample, ...]
    monitoring_examples: tuple[DialogueExample, ...]
    replay_examples: tuple[DialogueExample, ...]
    work_dir: Path
    cancellation_event: threading.Event | None = None


@dataclass(frozen=True)
class PersonalFoundationCandidate:
    """Complete portable bundle returned by a backend-specific trainer."""

    bundle_path: Path
    identity: PersonalFoundationIdentity
    parent_foundation_id: str
    parent_bundle_sha256: str
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class PersonalFoundationManifest:
    """Text-free immutable publication metadata for one full checkpoint."""

    identity: PersonalFoundationIdentity
    parent_foundation_id: str | None
    parent_bundle_sha256: str | None
    run_id: int
    new_lesson_ids: tuple[int, ...]
    monitoring_lesson_ids: tuple[int, ...]
    replay_lesson_ids: tuple[int, ...]
    created_at: str
    optimizer_state_included: bool = False

    def validate(self) -> None:
        self.identity.validate()
        if self.run_id == 0:
            if (
                self.parent_foundation_id is not None
                or self.parent_bundle_sha256 is not None
                or self.new_lesson_ids
                or self.monitoring_lesson_ids
                or self.replay_lesson_ids
            ):
                raise PersonalFoundationError("base foundation manifest is invalid")
        else:
            if type(self.run_id) is not int or self.run_id <= 0:
                raise PersonalFoundationError("foundation manifest run_id is invalid")
            if (
                not isinstance(self.parent_foundation_id, str)
                or not _IDENTIFIER_RE.fullmatch(self.parent_foundation_id)
                or not isinstance(self.parent_bundle_sha256, str)
                or not _SHA256_RE.fullmatch(self.parent_bundle_sha256)
            ):
                raise PersonalFoundationError("foundation manifest parent is invalid")
            if len(self.new_lesson_ids) != PERSONAL_FOUNDATION_THRESHOLD:
                raise PersonalFoundationError(
                    "foundation manifest must bind five lessons"
                )
            if len(self.monitoring_lesson_ids) != 1:
                raise PersonalFoundationError(
                    "foundation monitoring manifest is invalid"
                )
            if not set(self.monitoring_lesson_ids) <= set(self.new_lesson_ids):
                raise PersonalFoundationError(
                    "foundation monitor is outside its new batch"
                )
            if set(self.new_lesson_ids) & set(self.replay_lesson_ids):
                raise PersonalFoundationError(
                    "foundation replay overlaps its new batch"
                )
        all_ids = self.new_lesson_ids + self.replay_lesson_ids
        if (
            any(
                type(identifier) is not int or identifier <= 0 for identifier in all_ids
            )
            or len(set(all_ids)) != len(all_ids)
            or len(self.replay_lesson_ids) > PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES
        ):
            raise PersonalFoundationError("foundation manifest lesson IDs are invalid")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise PersonalFoundationError("foundation manifest created_at is invalid")
        if self.optimizer_state_included is not False:
            raise PersonalFoundationError(
                "v2 foundation bundles cannot contain optimizer state"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": PERSONAL_FOUNDATION_FORMAT,
            "visibility": "private-local-only",
            "identity": _identity_dict(self.identity),
            "parent_foundation_id": self.parent_foundation_id,
            "parent_bundle_sha256": self.parent_bundle_sha256,
            "run_id": self.run_id,
            "new_lesson_ids": list(self.new_lesson_ids),
            "monitoring_lesson_ids": list(self.monitoring_lesson_ids),
            "replay_lesson_ids": list(self.replay_lesson_ids),
            "created_at": self.created_at,
            "optimizer_state_included": False,
            "raw_conversation_included": False,
            "uploads_enabled": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PersonalFoundationManifest:
        expected = {
            "format",
            "visibility",
            "identity",
            "parent_foundation_id",
            "parent_bundle_sha256",
            "run_id",
            "new_lesson_ids",
            "monitoring_lesson_ids",
            "replay_lesson_ids",
            "created_at",
            "optimizer_state_included",
            "raw_conversation_included",
            "uploads_enabled",
        }
        if set(value) != expected or value.get("format") != PERSONAL_FOUNDATION_FORMAT:
            raise PersonalFoundationError("foundation manifest fields are incompatible")
        if (
            value.get("visibility") != "private-local-only"
            or value.get("raw_conversation_included") is not False
            or value.get("uploads_enabled") is not False
        ):
            raise PersonalFoundationError(
                "foundation manifest privacy boundary changed"
            )
        manifest = cls(
            identity=_identity_from_dict(value.get("identity")),
            parent_foundation_id=value.get("parent_foundation_id"),
            parent_bundle_sha256=value.get("parent_bundle_sha256"),
            run_id=value.get("run_id"),
            new_lesson_ids=_id_tuple(value.get("new_lesson_ids")),
            monitoring_lesson_ids=_id_tuple(value.get("monitoring_lesson_ids")),
            replay_lesson_ids=_id_tuple(value.get("replay_lesson_ids")),
            created_at=value.get("created_at"),
            optimizer_state_included=value.get("optimizer_state_included"),
        )
        manifest.validate()
        return manifest


@dataclass(frozen=True)
class PersonalFoundationArtifact:
    directory: Path
    bundle_path: Path
    manifest_path: Path
    manifest: PersonalFoundationManifest


@runtime_checkable
class PersonalFoundationTrainer(Protocol):
    @property
    def backend_name(self) -> str: ...

    def train_foundation(
        self,
        request: PersonalFoundationTrainingRequest,
    ) -> PersonalFoundationCandidate: ...


@runtime_checkable
class PersonalFoundationRuntime(Protocol):
    def personal_foundation_source(self) -> PersonalFoundationSource: ...

    def personal_training_target_fits(
        self,
        prompt: str,
        human_response: str,
    ) -> bool: ...

    def prepare_personal_foundation(
        self,
        artifact: PersonalFoundationArtifact,
    ) -> object: ...

    def activate_prepared_personal_foundation(
        self,
        prepared: object,
        identity: PersonalFoundationIdentity,
    ) -> None: ...


class PrivateFoundationRegistry:
    """Profile-isolated immutable full-bundle lineage, separate from adapters."""

    def __init__(
        self,
        root: str | Path,
        *,
        profile_id: str,
        maximum_bundle_bytes: int,
    ) -> None:
        if not _IDENTIFIER_RE.fullmatch(profile_id):
            raise PersonalFoundationError("foundation profile_id is invalid")
        self.profile_id = profile_id
        self.root = Path(root).expanduser().resolve() / "profiles" / profile_id
        self.foundations = self.root / "foundations-v2"
        self.work = self.root / "work-v2"
        self.state_path = self.root / "foundation-state-v2.json"
        self.maximum_bundle_bytes = maximum_bundle_bytes
        for path in (self.root, self.foundations, self.work):
            _ensure_private_directory(path)

    def create_work_dir(self, foundation_id: str) -> Path:
        if not _IDENTIFIER_RE.fullmatch(foundation_id):
            raise PersonalFoundationError("foundation_id is invalid")
        path = Path(tempfile.mkdtemp(prefix=f".{foundation_id}-", dir=self.work))
        _make_private(path, directory=True)
        return path

    def discard_work_dir(self, path: Path) -> None:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.work)
        except ValueError as error:
            raise PersonalFoundationError(
                "foundation work path escaped its profile"
            ) from error
        if resolved.is_dir():
            shutil.rmtree(resolved)

    def clear_stale_work_dirs(self) -> int:
        removed = 0
        for entry in tuple(self.work.iterdir()):
            try:
                details = entry.lstat()
                entry.resolve().relative_to(self.work)
            except (OSError, ValueError):
                continue
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                continue
            shutil.rmtree(entry)
            removed += 1
        return removed

    def initialize_base(self, source: PersonalFoundationSource) -> None:
        source.identity.validate()
        if (
            personal_foundation_bundle_sha256(source.bundle_path)
            != source.identity.bundle_sha256
        ):
            raise PersonalFoundationError("initial foundation bundle hash changed")
        if self.state_path.exists():
            _, _, base, _ = self.read_state()
            if base != source.identity:
                raise PersonalFoundationError(
                    "foundation profile base identity changed"
                )
            self._ensure_base_artifact(source)
            return
        manifest = PersonalFoundationManifest(
            identity=source.identity,
            parent_foundation_id=None,
            parent_bundle_sha256=None,
            run_id=0,
            new_lesson_ids=(),
            monitoring_lesson_ids=(),
            replay_lesson_ids=(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            existing = self.load(source.identity.foundation_id)
        except (FileNotFoundError, OSError, PersonalFoundationError):
            self._remove_foundation_entry(source.identity.foundation_id)
            self._publish_bundle(source.bundle_path, manifest)
        else:
            if existing.manifest != manifest:
                # A publish-before-state crash has a different timestamp but
                # must still be the exact base identity and run-zero shape.
                if (
                    existing.manifest.identity != source.identity
                    or existing.manifest.run_id != 0
                    or existing.manifest.parent_foundation_id is not None
                ):
                    raise PersonalFoundationError(
                        "foundation base publication conflicts with its profile"
                    )
        self.write_state(
            current=source.identity.foundation_id,
            previous=None,
            base=source.identity,
            generation=0,
        )

    def _ensure_base_artifact(self, source: PersonalFoundationSource) -> None:
        try:
            artifact = self.load(source.identity.foundation_id)
        except (FileNotFoundError, OSError, PersonalFoundationError):
            self._remove_foundation_entry(source.identity.foundation_id)
            self._publish_bundle(
                source.bundle_path,
                PersonalFoundationManifest(
                    identity=source.identity,
                    parent_foundation_id=None,
                    parent_bundle_sha256=None,
                    run_id=0,
                    new_lesson_ids=(),
                    monitoring_lesson_ids=(),
                    replay_lesson_ids=(),
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
            return
        if (
            artifact.manifest.identity != source.identity
            or artifact.manifest.run_id != 0
        ):
            raise PersonalFoundationError("foundation base artifact identity changed")

    def _remove_foundation_entry(self, foundation_id: str) -> None:
        if not _IDENTIFIER_RE.fullmatch(foundation_id):
            raise PersonalFoundationError("foundation_id is invalid")
        entry = self.foundations / foundation_id
        if not entry.exists() and not entry.is_symlink():
            return
        details = entry.lstat()
        if stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode):
            shutil.rmtree(entry)
        else:
            entry.unlink()
        _fsync_directory(self.foundations)

    def publish(
        self,
        request: PersonalFoundationTrainingRequest,
        candidate: PersonalFoundationCandidate,
    ) -> PersonalFoundationArtifact:
        validate_foundation_request(request)
        _validate_foundation_candidate(request, candidate)
        manifest = PersonalFoundationManifest(
            identity=candidate.identity,
            parent_foundation_id=candidate.parent_foundation_id,
            parent_bundle_sha256=candidate.parent_bundle_sha256,
            run_id=request.run_id,
            new_lesson_ids=tuple(item.lesson_id for item in request.new_examples),
            monitoring_lesson_ids=tuple(
                item.lesson_id for item in request.monitoring_examples
            ),
            replay_lesson_ids=tuple(item.lesson_id for item in request.replay_examples),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._publish_bundle(candidate.bundle_path, manifest)

    def _publish_bundle(
        self,
        source: Path,
        manifest: PersonalFoundationManifest,
    ) -> PersonalFoundationArtifact:
        manifest.validate()
        destination = self.foundations / manifest.identity.foundation_id
        if destination.exists() or destination.is_symlink():
            raise PersonalFoundationError("foundation identity already exists")
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{manifest.identity.foundation_id}-",
                dir=self.foundations,
            )
        )
        _make_private(temporary, directory=True)
        try:
            bundle_out = temporary / "bundle"
            _copy_bundle_tree(
                source,
                bundle_out,
                maximum_bytes=self.maximum_bundle_bytes,
            )
            if (
                personal_foundation_bundle_sha256(bundle_out)
                != manifest.identity.bundle_sha256
            ):
                raise PersonalFoundationError(
                    "published foundation bundle hash changed"
                )
            _atomic_private_json(temporary / "foundation.json", manifest.to_dict())
            os.rename(temporary, destination)
            _fsync_directory(self.foundations)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return self.load(manifest.identity.foundation_id)

    def load(self, foundation_id: str) -> PersonalFoundationArtifact:
        if not _IDENTIFIER_RE.fullmatch(foundation_id):
            raise PersonalFoundationError("foundation_id is invalid")
        directory = self.foundations / foundation_id
        details = directory.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise PersonalFoundationError("foundation directory is unsafe")
        if {entry.name for entry in directory.iterdir()} != {
            "bundle",
            "foundation.json",
        }:
            raise PersonalFoundationError("foundation directory fields changed")
        manifest_path = directory / "foundation.json"
        try:
            raw = json.loads(
                _read_regular_file(
                    manifest_path,
                    PERSONAL_FOUNDATION_MANIFEST_BYTES,
                )
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PersonalFoundationError(
                "foundation manifest JSON is invalid"
            ) from error
        if not isinstance(raw, Mapping):
            raise PersonalFoundationError("foundation manifest root is invalid")
        manifest = PersonalFoundationManifest.from_dict(raw)
        if manifest.identity.foundation_id != foundation_id:
            raise PersonalFoundationError("foundation directory identity changed")
        bundle_path = directory / "bundle"
        if (
            personal_foundation_bundle_sha256(bundle_path)
            != manifest.identity.bundle_sha256
        ):
            raise PersonalFoundationError("foundation bundle checksum changed")
        return PersonalFoundationArtifact(
            directory=directory,
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    def read_state(
        self,
    ) -> tuple[str, str | None, PersonalFoundationIdentity, int]:
        value = self._read_state_document()
        current = value["current_foundation_id"]
        previous = value["previous_foundation_id"]
        if not isinstance(current, str) or not _IDENTIFIER_RE.fullmatch(current):
            raise PersonalFoundationError("current foundation pointer is invalid")
        if previous is not None and (
            not isinstance(previous, str) or not _IDENTIFIER_RE.fullmatch(previous)
        ):
            raise PersonalFoundationError("previous foundation pointer is invalid")
        base = _identity_from_dict(value["base_identity"])
        generation = value["generation"]
        if type(generation) is not int or generation < 0:
            raise PersonalFoundationError("foundation generation is invalid")
        return current, previous, base, generation

    def pending_retirement(self) -> tuple[str, int | None] | None:
        value = self._read_state_document().get("pending_retirement")
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"mode", "run_id"}:
            raise PersonalFoundationError("pending foundation retirement is invalid")
        mode = value.get("mode")
        run_id = value.get("run_id")
        if mode in {"all", "recover-current-to-base"} and run_id is None:
            return mode, None
        if mode in {"run", "recover-current", "recover-current-to-base"} and (
            type(run_id) is int and run_id > 0
        ):
            return mode, run_id
        raise PersonalFoundationError("pending foundation retirement is invalid")

    def _read_state_document(self) -> Mapping[str, Any]:
        if not self.state_path.exists():
            raise PersonalFoundationError("foundation profile state is missing")
        try:
            value = json.loads(
                _read_regular_file(
                    self.state_path,
                    PERSONAL_FOUNDATION_MANIFEST_BYTES,
                )
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PersonalFoundationError("foundation state JSON is invalid") from error
        required = {
            "format",
            "profile_id",
            "current_foundation_id",
            "previous_foundation_id",
            "base_identity",
            "generation",
        }
        if not isinstance(value, Mapping) or not (
            set(value) == required or set(value) == required | {"pending_retirement"}
        ):
            raise PersonalFoundationError("foundation state fields changed")
        if (
            value["format"] != PERSONAL_FOUNDATION_STATE_FORMAT
            or value["profile_id"] != self.profile_id
        ):
            raise PersonalFoundationError("foundation state profile changed")
        return value

    def write_state(
        self,
        *,
        current: str,
        previous: str | None,
        base: PersonalFoundationIdentity,
        generation: int,
        pending_retirement: tuple[str, int | None] | None = None,
    ) -> None:
        current_artifact = self.load(current)
        previous_artifact = self.load(previous) if previous is not None else None
        base.validate()
        base_artifact = self.load(base.foundation_id)
        if base_artifact.manifest.identity != base:
            raise PersonalFoundationError("foundation base pointer identity changed")
        _validate_lineage_identity(base, current_artifact.manifest.identity)
        if previous_artifact is not None:
            _validate_lineage_identity(base, previous_artifact.manifest.identity)
        if type(generation) is not int or generation < 0:
            raise PersonalFoundationError("foundation generation is invalid")
        pending_value: dict[str, Any] | None = None
        if pending_retirement is not None:
            mode, run_id = pending_retirement
            if not (
                (mode in {"all", "recover-current-to-base"} and run_id is None)
                or (
                    mode in {"run", "recover-current", "recover-current-to-base"}
                    and type(run_id) is int
                    and run_id > 0
                )
            ):
                raise PersonalFoundationError(
                    "pending foundation retirement is invalid"
                )
            pending_value = {"mode": mode, "run_id": run_id}
        _atomic_private_json(
            self.state_path,
            {
                "format": PERSONAL_FOUNDATION_STATE_FORMAT,
                "profile_id": self.profile_id,
                "current_foundation_id": current,
                "previous_foundation_id": previous,
                "base_identity": _identity_dict(base),
                "generation": generation,
                "pending_retirement": pending_value,
            },
        )

    def prune(self) -> int:
        current, previous, base, _ = self.read_state()
        # The immutable base is the last-resort recovery and rollback anchor.
        # It remains present even after multiple successful generations.
        retained = {base.foundation_id, current, previous}
        removed = 0
        for entry in tuple(self.foundations.iterdir()):
            if entry.name in retained:
                continue
            if not _IDENTIFIER_RE.fullmatch(entry.name):
                continue
            try:
                artifact = self.load(entry.name)
            except (OSError, PersonalFoundationError):
                continue
            shutil.rmtree(artifact.directory)
            removed += 1
        if removed:
            _fsync_directory(self.foundations)
        return removed

    def discard_unreferenced(self, foundation_id: str) -> None:
        current, previous, base, _ = self.read_state()
        if foundation_id in {base.foundation_id, current, previous}:
            raise PersonalFoundationError("foundation is still referenced")
        path = self.foundations / foundation_id
        if path.exists():
            shutil.rmtree(self.load(foundation_id).directory)
            _fsync_directory(self.foundations)


def validate_foundation_request(request: PersonalFoundationTrainingRequest) -> None:
    """Shared request contract used by coordinators and backend tests."""

    if type(request.run_id) is not int or request.run_id <= 0:
        raise PersonalFoundationError("run_id must be a positive integer")
    if not _IDENTIFIER_RE.fullmatch(request.candidate_foundation_id):
        raise PersonalFoundationError("candidate_foundation_id is invalid")
    request.parent.identity.validate()
    if len(request.new_examples) != PERSONAL_FOUNDATION_THRESHOLD:
        raise PersonalFoundationError(
            "direct foundation training requires exactly five new examples"
        )
    if len(request.monitoring_examples) != PERSONAL_FOUNDATION_MONITORING_EXAMPLES:
        raise PersonalFoundationError(
            "direct foundation telemetry requires one monitoring example"
        )
    if len(request.replay_examples) > PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES:
        raise PersonalFoundationError("direct foundation replay exceeds its hard bound")
    new_ids = [example.lesson_id for example in request.new_examples]
    monitor_ids = [example.lesson_id for example in request.monitoring_examples]
    replay_ids = [example.lesson_id for example in request.replay_examples]
    if len(set(new_ids)) != len(new_ids):
        raise PersonalFoundationError("new foundation examples must be distinct")
    if not set(monitor_ids) <= set(new_ids):
        raise PersonalFoundationError(
            "monitoring telemetry must be drawn from the five new examples"
        )
    if set(new_ids) & set(replay_ids) or len(set(replay_ids)) != len(replay_ids):
        raise PersonalFoundationError(
            "new and replay foundation examples must be disjoint"
        )
    if not request.work_dir.is_dir():
        raise PersonalFoundationError("foundation work directory does not exist")


def validate_foundation_metrics(metrics: Mapping[str, Any]) -> None:
    """Reject non-finite or text-bearing telemetry before ledger storage."""

    if not isinstance(metrics, Mapping):
        raise PersonalFoundationError("foundation telemetry must be a mapping")
    for key, value in metrics.items():
        if not isinstance(key, str) or not _IDENTIFIER_RE.fullmatch(key):
            raise PersonalFoundationError("foundation telemetry key is invalid")
        if isinstance(value, bool) or value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PersonalFoundationError(
                "foundation telemetry must be finite numeric data"
            )
    _monitoring_summary(metrics)


class PersonalFoundationTrainingCoordinator:
    """Train and hot-activate a complete private foundation every five turns."""

    def __init__(
        self,
        store: ApprenticeshipStore,
        config: PersonalFoundationTrainingConfig,
        *,
        trainer: PersonalFoundationTrainer,
        runtime: PersonalFoundationRuntime,
        recover_interrupted: bool = False,
    ) -> None:
        self.store = store
        self.config = config
        self.trainer = trainer
        self.runtime = runtime
        config.validate()
        if not isinstance(trainer, PersonalFoundationTrainer):
            raise PersonalFoundationError("foundation trainer protocol is incomplete")
        if not isinstance(runtime, PersonalFoundationRuntime):
            raise PersonalFoundationError("foundation runtime protocol is incomplete")
        initial = runtime.personal_foundation_source()
        _validate_source(initial)
        if trainer.backend_name != initial.identity.backend:
            raise PersonalFoundationError(
                "foundation trainer and runtime backends differ"
            )
        self._initial_source = initial
        self.profile_id = config.profile_id or personal_foundation_profile_id(
            initial.identity
        )
        self.registry = PrivateFoundationRegistry(
            config.artifact_root,
            profile_id=self.profile_id,
            maximum_bundle_bytes=config.maximum_bundle_bytes,
        )
        self._mutex = threading.RLock()
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._rerun_requested = False
        self._state = "disabled" if not config.enabled else "idle"
        self._last_error: str | None = None
        self._run_id: int | None = None
        self._last_replay_examples = 0
        self._retry_not_before = 0.0
        self.registry.initialize_base(initial)
        self._legacy_unscoped_adopted = (
            self.store.adopt_unscoped_foundation_pending_lessons(self.profile_id)
        )
        if recover_interrupted:
            self.registry.clear_stale_work_dirs()
            self._recover_interrupted_runs()
        self._reconcile_pending_retirement()
        self._recover_missing_current()
        self._reconcile_current_run_status()
        self._sanitize_previous()
        self._reconcile_incomplete_promotions()
        self.restore_active_foundation()
        self.registry.prune()
        self._consumed_content_hashes = {
            _dialogue_content_sha256(item.problem, item.correct_answer)
            for item in self.store.list_personal_foundation_used_dialogue_lessons(
                self.profile_id,
                limit=None,
            )
        }
        self._restore_failure_backoff()
        self._last_monitoring = self._load_last_monitoring()

    def consider_lesson(self, lesson_id: int) -> bool:
        if type(lesson_id) is not int or lesson_id <= 0:
            raise ValueError("lesson_id must be a positive integer")
        if not self.config.enabled or self._shutdown.is_set():
            return False
        lesson = self.store.get_lesson(lesson_id)
        if (
            lesson is not None
            and lesson.status == "pending"
            and lesson.metadata.get("foundation_profile_id") == self.profile_id
            and _is_personal_dialogue_candidate(lesson)
        ):
            digest = _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
            if digest in self._consumed_content_hashes:
                self.store.quarantine_pending_lesson_ids(
                    [lesson.id],
                    reason="already-consumed-private-foundation-dialogue",
                    foundation_profile_id=self.profile_id,
                )
            elif not self._is_trainable(lesson):
                self.store.quarantine_pending_lesson_ids(
                    [lesson.id],
                    reason="outside-personal-foundation-context",
                    foundation_profile_id=self.profile_id,
                )
        return self.consider_training()

    def consider_training(self) -> bool:
        if not self.config.enabled or self._shutdown.is_set():
            return False
        with self._mutex:
            if self._thread is not None and self._thread.is_alive():
                self._rerun_requested = True
                return False
            if self.registry.pending_retirement() is not None:
                try:
                    self._reconcile_pending_retirement()
                except Exception as error:
                    self._state = "recovery_required"
                    self._last_error = _bounded_error(error)
                    return False
            if time.time() < self._retry_not_before:
                return False
            if (
                self.store.count_trusted_typed_dialogue_lessons(
                    status="pending",
                    foundation_profile_id=self.profile_id,
                )
                < PERSONAL_FOUNDATION_THRESHOLD
            ):
                return False
            self._state = "queued"
            self._last_error = None
            thread = threading.Thread(
                target=self._worker,
                name="tinydog-personal-foundation-trainer",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return True

    def trigger(self, *, force: bool = False) -> bool:
        if force:
            self._retry_not_before = 0.0
        return self.consider_training()

    def wait_for_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._mutex:
                thread = self._thread
            if thread is None or not thread.is_alive():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            thread.join(remaining)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._shutdown.set()
        with self._mutex:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, timeout))
            if thread.is_alive():
                raise PersonalFoundationError(
                    "foundation trainer did not stop; retaining the exclusive lease"
                )

    def status(self) -> dict[str, Any]:
        source = self.runtime.personal_foundation_source()
        source.identity.validate()
        pending_retirement = self.registry.pending_retirement()
        pending = self.store.count_trusted_typed_dialogue_lessons(
            status="pending",
            foundation_profile_id=self.profile_id,
        )
        retry_after = max(0.0, self._retry_not_before - time.time())
        with self._mutex:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "state": "backoff" if retry_after > 0 and not running else self._state,
                "authority": "personal-foundation-local",
                "privacy": "private-local-only",
                "backend": source.identity.backend,
                "profile_id": self.profile_id,
                "pending_scope": "foundation-profile",
                "legacy_unscoped_adoption_policy": "first-v2-profile-atomic",
                "legacy_unscoped_adopted_this_startup": self._legacy_unscoped_adopted,
                "parameter_scope": "full-foundation",
                "foundation_trainable": True,
                "adapter_mode": "legacy-disabled-by-default",
                "uploads_enabled": False,
                "produces_community_updates": False,
                "running": running,
                "recovery_pending": pending_retirement is not None,
                "qualifying_pending": pending,
                "lesson_threshold": PERSONAL_FOUNDATION_THRESHOLD,
                "progress": min(pending, PERSONAL_FOUNDATION_THRESHOLD)
                / PERSONAL_FOUNDATION_THRESHOLD,
                "new_examples_per_run": PERSONAL_FOUNDATION_THRESHOLD,
                "all_new_examples_optimized": True,
                "optimizer_state_continuity": False,
                "optimizer_state_reset_each_run": True,
                "monitoring_examples_per_run": 1,
                "monitoring_is_holdout": False,
                "monitoring_is_promotion_gate": False,
                "quality_metrics_are_telemetry": True,
                "last_monitoring": self._last_monitoring,
                "promotion_policy": "validated-training-always-promotes",
                "maximum_replay_examples_per_run": self.config.maximum_replay_examples,
                "replay_scan_limit": PERSONAL_FOUNDATION_REPLAY_SCAN_LIMIT,
                "last_replay_examples": self._last_replay_examples,
                "historical_used_examples": self.store.count_personal_foundation_used_dialogue_lessons(
                    self.profile_id
                ),
                "active_foundation_id": source.identity.foundation_id,
                "active_bundle_sha256": source.identity.bundle_sha256,
                "rollback_available": self.registry.read_state()[1] is not None
                or source.identity != self.registry.read_state()[2],
                "retry_after_seconds": retry_after,
                "run_id": self._run_id,
                "last_error": self._last_error,
            }

    def restore_active_foundation(self) -> None:
        current, _, base, _ = self.registry.read_state()
        artifact = self.registry.load(current)
        _validate_lineage_identity(base, artifact.manifest.identity)
        prepared = self.runtime.prepare_personal_foundation(artifact)
        self.runtime.activate_prepared_personal_foundation(
            prepared,
            artifact.manifest.identity,
        )
        if (
            self.runtime.personal_foundation_source().identity
            != artifact.manifest.identity
        ):
            raise PersonalFoundationError(
                "runtime did not restore the active foundation"
            )

    def rollback(self) -> dict[str, Any]:
        with self._mutex:
            if self._thread is not None and self._thread.is_alive():
                raise PersonalFoundationError(
                    "cannot roll back during foundation training"
                )
            current, previous, base, generation = self.registry.read_state()
            if current == base.foundation_id:
                raise PersonalFoundationError("the base foundation is already active")
            target_id = previous or base.foundation_id
            failed_previous: str | None = None
            try:
                target = self.registry.load(target_id)
                _validate_lineage_identity(base, target.manifest.identity)
                if target_id != base.foundation_id:
                    if not self._promoted_artifact_matches_ledger(target):
                        self._quarantine_invalid_ancestor_lessons(target)
                        raise PersonalFoundationError(
                            "previous foundation no longer has a promoted ledger run"
                        )
                prepared = self.runtime.prepare_personal_foundation(target)
            except Exception:
                if target_id != base.foundation_id:
                    failed_previous = target_id
                    target_id = base.foundation_id
                target = self.registry.load(target_id)
                prepared = self.runtime.prepare_personal_foundation(target)
            current_artifact = self.registry.load(current)
            current_prepared = self.runtime.prepare_personal_foundation(
                current_artifact
            )
            self.runtime.activate_prepared_personal_foundation(
                prepared,
                target.manifest.identity,
            )
            pending_retirement: tuple[str, int | None]
            if target_id == base.foundation_id:
                pending_retirement = ("all", None)
            elif current_artifact.manifest.run_id > 0:
                pending_retirement = ("run", current_artifact.manifest.run_id)
            else:
                raise PersonalFoundationError("active foundation has no rollback run")
            try:
                # The pending marker makes the filesystem pointer the durable
                # commit record. A crash before the SQLite retirement is
                # completed is repaired on the next exclusive startup without
                # returning intentionally rejected lessons to the queue.
                self.registry.write_state(
                    current=target_id,
                    previous=None,
                    base=base,
                    generation=generation + 1,
                    pending_retirement=pending_retirement,
                )
            except Exception:
                if not self._state_commit_matches(
                    current=target_id,
                    previous=None,
                    base=base,
                    generation=generation + 1,
                    pending_retirement=pending_retirement,
                ):
                    self.runtime.activate_prepared_personal_foundation(
                        current_prepared,
                        current_artifact.manifest.identity,
                    )
                    raise
            self._reconcile_pending_retirement()
            if failed_previous is not None and failed_previous != current:
                self.registry._remove_foundation_entry(failed_previous)
            try:
                self.registry.prune()
            except (OSError, PersonalFoundationError):
                pass
            self._state = "rolled_back"
            return self.status()

    def _worker(self) -> None:
        allow_rerun = True
        try:
            if self.registry.pending_retirement() is not None:
                raise PersonalFoundationError(
                    "foundation retirement must finish before another training run"
                )
            selected = self._select_new_lessons()
            if len(selected) < PERSONAL_FOUNDATION_THRESHOLD:
                return
            selected = selected[:PERSONAL_FOUNDATION_THRESHOLD]
            current, _, base, generation = self.registry.read_state()
            active = self.runtime.personal_foundation_source()
            if active.identity.foundation_id != current:
                raise PersonalFoundationError("runtime and foundation pointer differ")
            run, claimed = self.store.claim_pending_lesson_ids(
                [item.id for item in selected],
                checkpoint_in=current,
                metadata={
                    "authority": "personal-foundation-local",
                    "privacy": "private-local-only",
                    "foundation_profile_id": self.profile_id,
                    "parameter_scope": "full-foundation",
                    "owner_pid": os.getpid(),
                },
                foundation_profile_id=self.profile_id,
            )
            if run is None or len(claimed) != PERSONAL_FOUNDATION_THRESHOLD:
                return
            self._run_id = run.id
            candidate_id = f"foundation-{uuid.uuid4().hex}"
            work_dir: Path | None = None
            published: PersonalFoundationArtifact | None = None
            pointer_committed = False
            try:
                work_dir = self.registry.create_work_dir(candidate_id)
                request = self._request(
                    run.id,
                    candidate_id,
                    self.registry.load(current),
                    claimed,
                    work_dir,
                )
                self._last_replay_examples = len(request.replay_examples)
                self._state = "training"
                candidate = self.trainer.train_foundation(request)
                _validate_foundation_candidate(request, candidate)
                self._state = "validating"
                published = self.registry.publish(request, candidate)
                prepared = self.runtime.prepare_personal_foundation(published)
                self.store.complete_personal_foundation_run(
                    run.id,
                    [item.id for item in claimed],
                    checkpoint_out=str(published.directory),
                    metrics=dict(candidate.metrics),
                    metadata={
                        "authority": "personal-foundation-local",
                        "foundation_profile_id": self.profile_id,
                        "foundation_id": candidate_id,
                        "parent_foundation_id": current,
                        "all_new_examples_optimized": True,
                        "monitoring_is_holdout": False,
                        "quality_gate_applied": False,
                        "uploads_enabled": False,
                    },
                )
                self._last_monitoring = _monitoring_summary(candidate.metrics)
                self._state = "activating"
                self.runtime.activate_prepared_personal_foundation(
                    prepared,
                    candidate.identity,
                )
                after = self.runtime.personal_foundation_source()
                if after.identity != candidate.identity:
                    raise PersonalFoundationError("runtime activation identity changed")
                if self.registry.pending_retirement() is not None:
                    raise PersonalFoundationError(
                        "foundation retirement appeared during training"
                    )
                try:
                    self.registry.write_state(
                        current=candidate_id,
                        previous=current,
                        base=base,
                        generation=generation + 1,
                    )
                except Exception:
                    if not self._promotion_state_committed(
                        current=candidate_id,
                        previous=current,
                        base=base,
                        generation=generation + 1,
                    ):
                        raise
                pointer_committed = True
                self._consumed_content_hashes.update(
                    _dialogue_content_sha256(item.problem, item.correct_answer)
                    for item in claimed
                )
                self._retry_not_before = 0.0
                self._state = "ready"
                try:
                    self.registry.prune()
                except (OSError, PersonalFoundationError):
                    pass
            except Exception as error:
                allow_rerun = False
                if not pointer_committed:
                    compensation_errors = self._compensate_uncommitted_failure(
                        run_id=run.id,
                        published=published,
                        original_error=error,
                    )
                else:
                    compensation_errors = []
                self._state = "failed"
                details = [_bounded_error(error), *compensation_errors]
                self._last_error = _bounded_error(
                    PersonalFoundationError(" | ".join(details))
                )
            finally:
                if work_dir is not None:
                    try:
                        self.registry.discard_work_dir(work_dir)
                    except (OSError, PersonalFoundationError):
                        pass
        except Exception as error:
            allow_rerun = False
            self._state = "failed"
            self._last_error = _bounded_error(error)
        finally:
            has_more = (
                allow_rerun
                and not self._shutdown.is_set()
                and self.store.count_trusted_typed_dialogue_lessons(
                    status="pending",
                    foundation_profile_id=self.profile_id,
                )
                >= PERSONAL_FOUNDATION_THRESHOLD
            )
            with self._mutex:
                rerun = allow_rerun and (self._rerun_requested or has_more)
                self._rerun_requested = False
                if rerun:
                    thread = threading.Thread(
                        target=self._worker,
                        name="tinydog-personal-foundation-trainer",
                        daemon=True,
                    )
                    self._thread = thread
                    thread.start()
                else:
                    self._thread = None

    def _request(
        self,
        run_id: int,
        candidate_id: str,
        parent: PersonalFoundationArtifact,
        claimed: list[Lesson],
        work_dir: Path,
    ) -> PersonalFoundationTrainingRequest:
        examples = tuple(_dialogue_example(item) for item in claimed)
        ordered = tuple(
            sorted(examples, key=lambda item: (item.content_sha256, item.lesson_id))
        )
        monitor = (ordered[-1],)
        replay = self._select_replay(run_id, {item.content_sha256 for item in ordered})
        request = PersonalFoundationTrainingRequest(
            run_id=run_id,
            candidate_foundation_id=candidate_id,
            parent=PersonalFoundationSource(
                parent.bundle_path, parent.manifest.identity
            ),
            new_examples=ordered,
            monitoring_examples=monitor,
            replay_examples=replay,
            work_dir=work_dir,
            cancellation_event=self._shutdown,
        )
        validate_foundation_request(request)
        return request

    def _promotion_state_committed(
        self,
        *,
        current: str,
        previous: str,
        base: PersonalFoundationIdentity,
        generation: int,
    ) -> bool:
        return self._state_commit_matches(
            current=current,
            previous=previous,
            base=base,
            generation=generation,
            pending_retirement=None,
        )

    def _state_commit_matches(
        self,
        *,
        current: str,
        previous: str | None,
        base: PersonalFoundationIdentity,
        generation: int,
        pending_retirement: tuple[str, int | None] | None,
    ) -> bool:
        """Resolve a write that may have committed before reporting failure."""

        try:
            stored = self.registry.read_state()
            pending = self.registry.pending_retirement()
        except (OSError, PersonalFoundationError):
            return False
        return (
            stored == (current, previous, base, generation)
            and pending == pending_retirement
        )

    def _compensate_uncommitted_failure(
        self,
        *,
        run_id: int,
        published: PersonalFoundationArtifact | None,
        original_error: BaseException,
    ) -> list[str]:
        """Independently repair ledger, artifact, and runtime after a failure."""

        retry_at = time.time() + self.config.failure_backoff_seconds
        self._retry_not_before = retry_at
        metadata = {
            "authority": "personal-foundation-local",
            "foundation_profile_id": self.profile_id,
            "retry_not_before_epoch": retry_at,
            "activation_or_publication_failed": True,
        }
        errors: list[str] = []
        latest = None
        try:
            latest = self.store.get_training_status(run_id)
            if latest is not None and latest.status == "promoted":
                self.store.rollback_training_run(
                    run_id,
                    error=_bounded_error(original_error),
                    metadata=metadata,
                )
            elif latest is not None and latest.status == "training":
                self.store.release_training_run(
                    run_id,
                    "failed",
                    error=_bounded_error(original_error),
                    metadata=metadata,
                )
        except Exception as ledger_error:
            try:
                durable = self.store.get_training_status(run_id)
            except Exception as reread_error:
                durable = None
                errors.append(f"ledger reread: {_bounded_error(reread_error)}")
            expected_status = (
                "rolled_back"
                if latest is not None and latest.status == "promoted"
                else "failed"
            )
            compensation_committed = self._ledger_compensation_committed(
                durable,
                expected_status=expected_status,
            )
            if not compensation_committed:
                errors.append(f"ledger compensation: {_bounded_error(ledger_error)}")
            if durable is not None and not compensation_committed:
                try:
                    self.store.update_training_run(
                        run_id,
                        durable.status,
                        error=_bounded_error(original_error),
                        metadata={
                            **metadata,
                            "ledger_compensation_incomplete": True,
                        },
                    )
                except Exception as annotation_error:
                    errors.append(
                        "ledger failure annotation: "
                        f"{_bounded_error(annotation_error)}"
                    )
        if published is not None:
            try:
                self.registry.discard_unreferenced(
                    published.manifest.identity.foundation_id
                )
            except Exception as discard_error:
                errors.append(f"artifact cleanup: {_bounded_error(discard_error)}")
        try:
            self.restore_active_foundation()
            durable_current = self.registry.read_state()[0]
            runtime_current = (
                self.runtime.personal_foundation_source().identity.foundation_id
            )
            if runtime_current != durable_current:
                raise PersonalFoundationError(
                    "runtime still differs from the durable foundation pointer"
                )
        except Exception as restore_error:
            errors.append(f"runtime restoration: {_bounded_error(restore_error)}")
        return errors

    def _ledger_compensation_committed(
        self,
        run: Any | None,
        *,
        expected_status: str,
    ) -> bool:
        """Attest a compensation by rereading both the run and all five rows."""

        if run is None or run.status != expected_status:
            return False
        for lesson_id in run.lesson_ids:
            try:
                lesson = self.store.get_lesson(lesson_id)
            except Exception:
                return False
            if (
                lesson is None
                or lesson.status != "pending"
                or lesson.training_run_id is not None
                or lesson.metadata.get("foundation_profile_id") != self.profile_id
            ):
                return False
        return True

    def _select_new_lessons(self) -> list[Lesson]:
        pending = reversed(
            self.store.list_recent_lessons(
                limit=10_000,
                pending_only=True,
                foundation_profile_id=self.profile_id,
            )
        )
        seen = set(self._consumed_content_hashes)
        selected: list[Lesson] = []
        quarantine: list[int] = []
        for lesson in pending:
            if not _is_personal_dialogue_candidate(lesson):
                continue
            digest = _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
            if digest in seen or not self._is_trainable(lesson):
                quarantine.append(lesson.id)
                continue
            seen.add(digest)
            selected.append(lesson)
        self.store.quarantine_pending_lesson_ids(
            quarantine,
            reason="duplicate-or-ineligible-private-foundation-dialogue",
            foundation_profile_id=self.profile_id,
        )
        return selected

    def _is_trainable(self, lesson: Lesson) -> bool:
        if not _is_typed_dialogue_lesson(self.store, lesson):
            return False
        return bool(
            self.runtime.personal_training_target_fits(
                lesson.problem,
                lesson.correct_answer,
            )
        )

    def _select_replay(
        self,
        run_id: int,
        new_hashes: set[str],
    ) -> tuple[DialogueExample, ...]:
        seen: set[str] = set()
        candidates: list[Lesson] = []
        if self._shutdown.is_set():
            raise PersonalFoundationError("foundation replay selection was cancelled")
        window = self.store.list_personal_foundation_used_dialogue_lessons(
            self.profile_id,
            limit=PERSONAL_FOUNDATION_REPLAY_SCAN_LIMIT,
            run_statuses=("promoted",),
            newest_first=True,
        )
        for index, lesson in enumerate(window):
            if index % 64 == 0 and self._shutdown.is_set():
                raise PersonalFoundationError(
                    "foundation replay selection was cancelled"
                )
            digest = _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
            if digest in seen or digest in new_hashes or not self._is_trainable(lesson):
                continue
            seen.add(digest)
            candidates.append(lesson)

        def score(item: Lesson) -> tuple[str, int]:
            digest = _dialogue_content_sha256(item.problem, item.correct_answer)
            value = hashlib.sha256(
                f"tinydog-foundation-replay-v1\0{self.profile_id}\0{run_id}\0{digest}".encode(
                    "ascii"
                )
            ).hexdigest()
            return value, item.id

        return tuple(
            _dialogue_example(item)
            for item in sorted(candidates, key=score)[
                : self.config.maximum_replay_examples
            ]
        )

    def _recover_interrupted_runs(self) -> None:
        before_id: int | None = None
        while True:
            runs = self.store.list_personal_foundation_runs(
                self.profile_id,
                status="training",
                limit=256,
                before_id=before_id,
            )
            if not runs:
                return
            for run in runs:
                self.store.release_training_run(
                    run.id,
                    "failed",
                    error="Recovered interrupted direct foundation training",
                    metadata={"recovered": True},
                )
            before_id = runs[-1].id

    def _reconcile_incomplete_promotions(self) -> None:
        current, _, _, _ = self.registry.read_state()
        current_run = self.registry.load(current).manifest.run_id
        before_id: int | None = None
        while True:
            runs = self.store.list_personal_foundation_runs(
                self.profile_id,
                status="promoted",
                limit=256,
                before_id=before_id,
            )
            newer = [run for run in runs if run.id > current_run]
            for run in newer:
                foundation_id = run.metadata.get("foundation_id")
                self.store.rollback_training_run(run.id)
                if isinstance(foundation_id, str):
                    try:
                        self.registry.discard_unreferenced(foundation_id)
                    except Exception:
                        pass
            if len(runs) < 256 or len(newer) < len(runs):
                return
            before_id = runs[-1].id

    def _reconcile_current_run_status(self) -> None:
        current, previous, base, generation = self.registry.read_state()
        artifact = self.registry.load(current)
        run_id = artifact.manifest.run_id
        if run_id == 0:
            if current != base.foundation_id:
                raise PersonalFoundationError(
                    "non-base foundation has a base run identity"
                )
            return
        run = self.store.get_training_status(run_id)
        if run is not None and run.status == "promoted":
            if not self._promoted_artifact_matches_ledger(artifact, run=run):
                raise PersonalFoundationError(
                    "active foundation and promoted ledger run differ"
                )
            return
        if run is not None and run.status == "training":
            self.store.release_training_run(
                run.id,
                "failed",
                error="Recovered state pointer without an atomic foundation completion",
                metadata={
                    "authority": "personal-foundation-local",
                    "foundation_profile_id": self.profile_id,
                    "recovered_invalid_current_pointer": True,
                },
            )
        fallback: str | None = None
        unusable_previous = False
        for candidate_id in (previous, base.foundation_id):
            if candidate_id is None or candidate_id == current:
                continue
            try:
                candidate = self.registry.load(candidate_id)
                _validate_lineage_identity(base, candidate.manifest.identity)
                if (
                    candidate_id != base.foundation_id
                    and not self._promoted_artifact_matches_ledger(candidate)
                ):
                    self._quarantine_invalid_ancestor_lessons(candidate)
                    raise PersonalFoundationError(
                        "fallback foundation no longer has a promoted ledger run"
                    )
            except (FileNotFoundError, OSError, PersonalFoundationError):
                if candidate_id == previous:
                    unusable_previous = True
                continue
            fallback = candidate_id
            break
        if fallback is None:
            raise PersonalFoundationError(
                "active foundation ledger mismatch has no verified fallback"
            )
        pending = (
            ("all", None)
            if fallback == base.foundation_id and unusable_previous
            else None
        )
        self.registry.write_state(
            current=fallback,
            previous=None,
            base=base,
            generation=generation + 1,
            pending_retirement=pending,
        )
        if pending is not None:
            self._reconcile_pending_retirement()
        if current != base.foundation_id:
            self.registry._remove_foundation_entry(current)
        if (
            unusable_previous
            and previous is not None
            and previous not in {current, base.foundation_id}
        ):
            self.registry._remove_foundation_entry(previous)

    def _sanitize_previous(self) -> None:
        current, previous, base, generation = self.registry.read_state()
        if previous is None:
            return
        try:
            artifact = self.registry.load(previous)
            _validate_lineage_identity(base, artifact.manifest.identity)
            if (
                previous != base.foundation_id
                and not self._promoted_artifact_matches_ledger(artifact)
            ):
                self._quarantine_invalid_ancestor_lessons(artifact)
                raise PersonalFoundationError(
                    "previous foundation no longer has a promoted ledger run"
                )
        except (FileNotFoundError, OSError, PersonalFoundationError):
            self.registry.write_state(
                current=current,
                previous=None,
                base=base,
                generation=generation + 1,
            )
            if previous not in {current, base.foundation_id}:
                self.registry._remove_foundation_entry(previous)

    def _promoted_artifact_matches_ledger(
        self,
        artifact: PersonalFoundationArtifact,
        *,
        run: Any | None = None,
    ) -> bool:
        manifest = artifact.manifest
        if manifest.run_id <= 0:
            return False
        linked = run or self.store.get_training_status(manifest.run_id)
        if linked is None or linked.status != "promoted":
            return False
        try:
            checkpoint = Path(linked.checkpoint_out).expanduser().resolve()
        except TypeError:
            return False
        if not (
            linked.metadata.get("authority") == "personal-foundation-local"
            and linked.metadata.get("foundation_profile_id") == self.profile_id
            and linked.metadata.get("foundation_id") == manifest.identity.foundation_id
            and set(linked.lesson_ids) == set(manifest.new_lesson_ids)
            and len(linked.lesson_ids) == len(manifest.new_lesson_ids)
            and checkpoint == artifact.directory.resolve()
        ):
            return False
        for lesson_id in manifest.new_lesson_ids:
            lesson = self.store.get_lesson(lesson_id)
            if (
                lesson is None
                or lesson.status != "used"
                or lesson.training_run_id != manifest.run_id
                or lesson.metadata.get("foundation_profile_id") != self.profile_id
            ):
                return False
        return True

    def _quarantine_invalid_ancestor_lessons(
        self,
        artifact: PersonalFoundationArtifact,
    ) -> None:
        """Prevent rejected ancestor data from re-entering this profile's queue."""

        self.store.quarantine_pending_lesson_ids(
            artifact.manifest.new_lesson_ids,
            reason="invalid-private-foundation-ancestor",
            foundation_profile_id=self.profile_id,
        )

    def _reconcile_pending_retirement(self) -> None:
        pending = self.registry.pending_retirement()
        if pending is None:
            return
        mode, run_id = pending
        explicit_metadata = {
            "explicit_user_rollback": True,
            "lessons_requeued": False,
            "authority": "personal-foundation-local",
            "recovered_pending_retirement": True,
        }
        if mode in {"run", "recover-current", "recover-current-to-base"} and (
            run_id is not None
        ):
            assert run_id is not None
            run = self.store.get_training_status(run_id)
            if run is None:
                raise PersonalFoundationError("pending rollback run is missing")
            if (
                run.metadata.get("authority") != "personal-foundation-local"
                or run.metadata.get("foundation_profile_id") != self.profile_id
            ):
                raise PersonalFoundationError("pending rollback run changed profile")
            if run.status == "promoted" and mode == "run":
                try:
                    self.store.retire_promoted_training_run(
                        run.id,
                        metadata=explicit_metadata,
                    )
                except Exception:
                    if not self._retirement_committed((run.id,)):
                        raise
            elif run.status == "promoted":
                try:
                    self.store.rollback_training_run(
                        run.id,
                        metadata={
                            "recovered_missing_foundation": True,
                            "lessons_requeued": True,
                            "authority": "personal-foundation-local",
                        },
                    )
                except Exception:
                    durable = self.store.get_training_status(run.id)
                    if not self._ledger_compensation_committed(
                        durable,
                        expected_status="rolled_back",
                    ):
                        raise
            elif run.status == "rolled_back":
                committed = (
                    self._retirement_committed((run.id,))
                    if mode == "run"
                    else self._ledger_compensation_committed(
                        run,
                        expected_status="rolled_back",
                    )
                )
                if not committed:
                    raise PersonalFoundationError(
                        "pending rollback lesson state is inconsistent"
                    )
            else:
                raise PersonalFoundationError("pending rollback run changed state")
        if mode in {"all", "recover-current-to-base"}:
            while True:
                run_ids = [
                    run.id
                    for run in self.store.list_personal_foundation_runs(
                        self.profile_id,
                        status="promoted",
                        limit=100,
                    )
                ]
                if not run_ids:
                    break
                try:
                    self.store.retire_promoted_training_runs(
                        run_ids,
                        metadata=(
                            explicit_metadata
                            if mode == "all"
                            else {
                                "unrecoverable_foundation_ancestor": True,
                                "lessons_requeued": False,
                                "authority": "personal-foundation-local",
                            }
                        ),
                    )
                except Exception:
                    if not self._retirement_committed(tuple(run_ids)):
                        raise
        current, previous, base, generation = self.registry.read_state()
        try:
            self.registry.write_state(
                current=current,
                previous=previous,
                base=base,
                generation=generation,
                pending_retirement=None,
            )
        except Exception:
            if not self._state_commit_matches(
                current=current,
                previous=previous,
                base=base,
                generation=generation,
                pending_retirement=None,
            ):
                raise

    def _retirement_committed(self, run_ids: tuple[int, ...]) -> bool:
        """Verify explicit retirement without trusting an exception boundary."""

        for run_id in run_ids:
            run = self.store.get_training_status(run_id)
            if run is None or run.status != "rolled_back":
                return False
            for lesson_id in run.lesson_ids:
                lesson = self.store.get_lesson(lesson_id)
                if (
                    lesson is None
                    or lesson.status != "used"
                    or lesson.training_run_id != run_id
                    or lesson.metadata.get("foundation_profile_id") != self.profile_id
                ):
                    return False
        return True

    def _recover_missing_current(self) -> None:
        current, previous, base, generation = self.registry.read_state()
        try:
            self.registry.load(current)
            return
        except (FileNotFoundError, OSError, PersonalFoundationError):
            pass
        fallback: str | None = None
        unusable_previous = False
        for candidate_id in (previous, base.foundation_id):
            if candidate_id is None or candidate_id == current:
                continue
            try:
                candidate = self.registry.load(candidate_id)
                _validate_lineage_identity(base, candidate.manifest.identity)
                if (
                    candidate_id != base.foundation_id
                    and not self._promoted_artifact_matches_ledger(candidate)
                ):
                    self._quarantine_invalid_ancestor_lessons(candidate)
                    raise PersonalFoundationError(
                        "recovery foundation no longer has a promoted ledger run"
                    )
            except (FileNotFoundError, OSError, PersonalFoundationError):
                if candidate_id == previous:
                    unusable_previous = True
                continue
            fallback = candidate_id
            break
        if fallback is None:
            # initialize_base already repaired the exact immutable base, so
            # reaching this point means the profile changed concurrently.
            raise PersonalFoundationError(
                "no verified foundation recovery target exists"
            )
        current_run = self.store.get_personal_foundation_run(self.profile_id, current)
        recovery_mode = (
            "recover-current-to-base"
            if fallback == base.foundation_id and unusable_previous
            else "recover-current"
        )
        pending_recovery = (
            (recovery_mode, None if current_run is None else current_run.id)
            if recovery_mode == "recover-current-to-base" or current_run is not None
            else None
        )
        self.registry.write_state(
            current=fallback,
            previous=None,
            base=base,
            generation=generation + 1,
            pending_retirement=pending_recovery,
        )
        if pending_recovery is not None:
            self._reconcile_pending_retirement()
        if fallback == base.foundation_id and unusable_previous:
            # A corrupt/missing previous checkpoint cannot remain a rollback
            # ancestor. Keep its already-consumed lessons retired; only the
            # newest missing current generation is eligible for crash retry.
            if previous is not None and previous != base.foundation_id:
                self.registry._remove_foundation_entry(previous)
        if current != base.foundation_id:
            self.registry._remove_foundation_entry(current)

    def _restore_failure_backoff(self) -> None:
        runs = self.store.list_personal_foundation_runs(self.profile_id, limit=1)
        if runs:
            value = runs[0].metadata.get("retry_not_before_epoch")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self._retry_not_before = max(self._retry_not_before, float(value))
            # A newer successful or explicit rollback supersedes every older
            # transient failure, including its former backoff.

    def _load_last_monitoring(self) -> dict[str, Any] | None:
        run = self.store.latest_personal_foundation_monitoring_run(self.profile_id)
        if run is None:
            return None
        try:
            return _monitoring_summary(run.metrics)
        except PersonalFoundationError:
            # Legacy or manually altered local telemetry never controls model
            # activation and is omitted rather than echoed through the API.
            return None


def _identity_dict(identity: PersonalFoundationIdentity) -> dict[str, str]:
    identity.validate()
    return {
        "backend": identity.backend,
        "foundation_id": identity.foundation_id,
        "bundle_sha256": identity.bundle_sha256,
        "weights_sha256": identity.weights_sha256,
        "config_sha256": identity.config_sha256,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "backend_contract_sha256": identity.backend_contract_sha256,
    }


def _identity_from_dict(value: Any) -> PersonalFoundationIdentity:
    expected = {
        "backend",
        "foundation_id",
        "bundle_sha256",
        "weights_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "backend_contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PersonalFoundationError("foundation identity fields are incompatible")
    identity = PersonalFoundationIdentity(
        backend=value["backend"],
        foundation_id=value["foundation_id"],
        bundle_sha256=value["bundle_sha256"],
        weights_sha256=value["weights_sha256"],
        config_sha256=value["config_sha256"],
        tokenizer_sha256=value["tokenizer_sha256"],
        backend_contract_sha256=value["backend_contract_sha256"],
    )
    identity.validate()
    return identity


def _id_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise PersonalFoundationError("foundation manifest lesson IDs are invalid")
    return tuple(value)


def _validate_source(source: PersonalFoundationSource) -> None:
    if not isinstance(source, PersonalFoundationSource):
        raise PersonalFoundationError("foundation source has an incompatible type")
    source.identity.validate()
    path = Path(source.bundle_path).expanduser()
    try:
        details = path.lstat()
    except OSError as error:
        raise PersonalFoundationError(
            "foundation source bundle is unavailable"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise PersonalFoundationError("foundation source bundle is unsafe")
    if personal_foundation_bundle_sha256(path) != source.identity.bundle_sha256:
        raise PersonalFoundationError("foundation source bundle checksum changed")


def _validate_lineage_identity(
    base: PersonalFoundationIdentity,
    candidate: PersonalFoundationIdentity,
) -> None:
    base.validate()
    candidate.validate()
    if (
        candidate.backend != base.backend
        or candidate.config_sha256 != base.config_sha256
        or candidate.tokenizer_sha256 != base.tokenizer_sha256
        or candidate.backend_contract_sha256 != base.backend_contract_sha256
    ):
        raise PersonalFoundationError(
            "foundation candidate changed its lineage identity"
        )


def _validate_foundation_candidate(
    request: PersonalFoundationTrainingRequest,
    candidate: PersonalFoundationCandidate,
) -> None:
    validate_foundation_request(request)
    if not isinstance(candidate, PersonalFoundationCandidate):
        raise PersonalFoundationError(
            "foundation trainer returned an incompatible candidate"
        )
    candidate.identity.validate()
    _validate_lineage_identity(request.parent.identity, candidate.identity)
    if candidate.identity.foundation_id != request.candidate_foundation_id:
        raise PersonalFoundationError("foundation candidate ID changed")
    if (
        candidate.parent_foundation_id != request.parent.identity.foundation_id
        or candidate.parent_bundle_sha256 != request.parent.identity.bundle_sha256
    ):
        raise PersonalFoundationError("foundation candidate parent changed")
    candidate_path = Path(candidate.bundle_path).expanduser()
    try:
        details = candidate_path.lstat()
        candidate_path.resolve().relative_to(request.work_dir.resolve())
    except (OSError, ValueError) as error:
        raise PersonalFoundationError(
            "foundation candidate escaped its private work directory"
        ) from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise PersonalFoundationError("foundation candidate bundle is unsafe")
    actual_hash = personal_foundation_bundle_sha256(candidate_path)
    if actual_hash != candidate.identity.bundle_sha256:
        raise PersonalFoundationError("foundation candidate bundle checksum changed")
    if (
        candidate.identity.bundle_sha256 == request.parent.identity.bundle_sha256
        or candidate.identity.weights_sha256 == request.parent.identity.weights_sha256
    ):
        raise PersonalFoundationError(
            "foundation training did not change the checkpoint"
        )
    validate_foundation_metrics(candidate.metrics)
    if (
        candidate.metrics.get("all_foundation_parameters_optimizer_eligible")
        is not True
        or candidate.metrics.get("tensor_scope_full_foundation") is not True
        or candidate.metrics.get("monitoring_used_for_selection") is not False
        or candidate.metrics.get("optimizer_state_continuity") is not False
        or candidate.metrics.get("optimizer_state_reset") is not True
        or candidate.metrics.get("new_examples_optimized")
        != PERSONAL_FOUNDATION_THRESHOLD
        or candidate.metrics.get("monitoring_examples")
        != PERSONAL_FOUNDATION_MONITORING_EXAMPLES
        or candidate.metrics.get("replay_examples") != len(request.replay_examples)
    ):
        raise PersonalFoundationError(
            "foundation candidate did not attest the complete training contract"
        )


def _copy_bundle_tree(
    source: str | Path,
    destination: str | Path,
    *,
    maximum_bytes: int,
) -> None:
    """Copy one complete bundle without following links or special files."""

    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise PersonalFoundationError("maximum foundation bundle size is invalid")
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    try:
        source_details = source_path.lstat()
    except OSError as error:
        raise PersonalFoundationError(
            "foundation bundle source is unavailable"
        ) from error
    if stat.S_ISLNK(source_details.st_mode) or not stat.S_ISDIR(source_details.st_mode):
        raise PersonalFoundationError("foundation bundle source is unsafe")
    source_root = source_path.resolve()
    destination_resolved = destination_path.resolve(strict=False)
    try:
        destination_resolved.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise PersonalFoundationError(
            "foundation bundle destination overlaps its source"
        )
    if destination_path.exists() or destination_path.is_symlink():
        raise PersonalFoundationError("foundation bundle destination already exists")
    destination_path.mkdir(mode=0o700, parents=False)
    _make_private(destination_path, directory=True)
    total_bytes = 0
    total_files = 0
    try:
        entries = sorted(
            source_root.rglob("*"),
            key=lambda item: item.relative_to(source_root).as_posix(),
        )
        if len(entries) > PERSONAL_FOUNDATION_MAX_FILES:
            raise PersonalFoundationError("foundation bundle has too many entries")
        for entry in entries:
            details = entry.lstat()
            relative = entry.relative_to(source_root)
            output = destination_path / relative
            if stat.S_ISLNK(details.st_mode):
                raise PersonalFoundationError(
                    "foundation bundle cannot contain symlinks"
                )
            if stat.S_ISDIR(details.st_mode):
                output.mkdir(mode=0o700)
                _make_private(output, directory=True)
                continue
            if not stat.S_ISREG(details.st_mode):
                raise PersonalFoundationError(
                    "foundation bundle contains a special file"
                )
            total_files += 1
            total_bytes += details.st_size
            if total_bytes > maximum_bytes:
                raise PersonalFoundationError(
                    "foundation bundle exceeds its private size bound"
                )
            input_flags = (
                os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            input_descriptor = -1
            output_descriptor = -1
            try:
                input_descriptor = os.open(entry, input_flags)
                opened = os.fstat(input_descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_dev != details.st_dev
                    or opened.st_ino != details.st_ino
                    or opened.st_size != details.st_size
                ):
                    raise PersonalFoundationError(
                        "foundation bundle changed while copying"
                    )
                output_descriptor = os.open(
                    output,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                    0o600,
                )
                copied = 0
                while True:
                    chunk = os.read(input_descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(output_descriptor, view)
                        if written <= 0:
                            raise PersonalFoundationError(
                                "foundation bundle copy stopped early"
                            )
                        view = view[written:]
                if copied != opened.st_size:
                    raise PersonalFoundationError(
                        "foundation bundle changed while copying"
                    )
                os.fsync(output_descriptor)
            finally:
                if output_descriptor >= 0:
                    os.close(output_descriptor)
                if input_descriptor >= 0:
                    os.close(input_descriptor)
        if total_files == 0:
            raise PersonalFoundationError("foundation bundle cannot be empty")
        for directory in sorted(
            (item for item in destination_path.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(destination_path)
    except Exception:
        if destination_path.exists():
            shutil.rmtree(destination_path)
        raise


def _bounded_error(error: BaseException, *, limit: int = 500) -> str:
    text = " ".join(str(error).split())
    if not text:
        text = error.__class__.__name__
    return text[:limit]


def _monitoring_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values: list[float] = []
    for key in ("baseline_monitoring_loss", "candidate_monitoring_loss"):
        value = metrics.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise PersonalFoundationError(f"foundation telemetry requires finite {key}")
        values.append(float(value))
    baseline, candidate = values
    delta = candidate - baseline
    if not math.isfinite(delta):
        raise PersonalFoundationError("foundation monitoring delta must be finite")
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "improved": candidate < baseline,
        "in_sample": True,
        "promotion_gate": False,
    }
