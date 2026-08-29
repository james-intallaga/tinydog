"""Backend-neutral, local-only conversational adapter apprenticeship.

This module is intentionally independent from MLX, PyTorch, signed releases,
and the community update protocol.  A backend supplies two small interfaces:
an isolated adapter trainer and an atomically swappable inference runtime.
The coordinator owns lesson selection, the exact 4+1 split, local artifact
validation, promotion, restart recovery, and rollback.

Only provenance-checked *typed* human dialogue lessons qualify today.  Raw
PCM, VAD boundaries, and generated audio contain no trusted semantic target
and therefore cannot silently become language-training examples.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from typing import Any, Mapping, Protocol, runtime_checkable
import uuid

from .apprenticeship import ApprenticeshipStore, Lesson


PERSONAL_TRAINING_THRESHOLD = 5
PERSONAL_TRAINING_EXAMPLES = 4
PERSONAL_HOLDOUT_EXAMPLES = 1
PERSONAL_DEFAULT_REPLAY_EXAMPLES = 8
PERSONAL_DEFAULT_RETENTION_EXAMPLES = 2
PERSONAL_MAX_REPLAY_EXAMPLES = 32
PERSONAL_MAX_RETENTION_EXAMPLES = 16
PERSONAL_MAX_TRAINING_CONTEXT_TOKENS = 256
PERSONAL_DIALOGUE_STRUCTURAL_TOKENS = 6
PERSONAL_PENDING_SCAN_LIMIT = 10_000
PERSONAL_ADAPTER_FORMAT = "tinydog-personal-adapter-v1"
PERSONAL_STATE_FORMAT = "tinydog-personal-state-v1"

_MAX_TEXT_BYTES = 8_192
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_ERROR_TEXT = 2_048
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_ARTIFACT_FORMAT_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")


class PersonalTrainingError(RuntimeError):
    """Personal adapter training or activation could not complete safely."""


def personal_dialogue_target_fits(
    tokenizer: Any,
    prompt: str,
    human_response: str,
    *,
    context_length: int,
) -> bool:
    """Whether training can preserve the complete human-authored target.

    Prompts may be compacted to their newest token suffix.  Human targets are
    never truncated, so the fixed BOS/role/turn/EOS structure plus every
    target token must fit the shared bounded personal-training context.
    """

    if (
        type(context_length) is not int
        or not 8 <= context_length <= PERSONAL_MAX_TRAINING_CONTEXT_TOKENS
        or not isinstance(prompt, str)
        or not isinstance(human_response, str)
        or not prompt.strip()
        or not human_response.strip()
    ):
        return False
    try:
        # Encoding the prompt also rejects reserved structural-token literals.
        tokenizer.encode(prompt.strip())
        response_ids = tokenizer.encode(human_response.strip())
    except (TypeError, ValueError):
        return False
    return len(response_ids) + PERSONAL_DIALOGUE_STRUCTURAL_TOKENS <= context_length + 1


def compact_personal_dialogue_prompt(
    tokenizer: Any,
    turns: Any,
    *,
    context_length: int,
) -> list[int]:
    """Bound inference context while preserving the newest turn's suffix.

    Whole old turns are removed first.  If the newest turn alone is too long,
    only its oldest content tokens are discarded; its role, turn boundary,
    and the assistant-generation marker remain structurally intact.
    """

    if type(context_length) is not int or context_length < 4:
        raise PersonalTrainingError("personal inference context is too small")
    if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
        raise TypeError("turns must be a sequence of dialogue turns")
    retained = list(turns)
    while True:
        ids = tokenizer.encode_dialogue(retained, bos=True, eos=False)
        ids.append(tokenizer.role_id("assistant"))
        if len(ids) <= context_length:
            return [int(token) for token in ids]
        if len(retained) > 1:
            retained.pop(0)
            continue
        if not retained:
            raise PersonalTrainingError("conversation exceeds the model context")
        turn = retained[0]
        if isinstance(turn, Mapping):
            role = turn.get("role")
            content = turn.get("content")
        elif (
            isinstance(turn, Sequence)
            and len(turn) == 2
            and not isinstance(turn, (str, bytes))
        ):
            role, content = turn
        else:
            raise TypeError("each dialogue turn must contain role and content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("dialogue role and content must be strings")
        content_budget = context_length - 4
        content_ids = tokenizer.encode(content)
        suffix = content_ids[-content_budget:] if content_budget else []
        return [
            tokenizer.bos_id,
            tokenizer.role_id(role),
            *suffix,
            tokenizer.end_turn_id,
            tokenizer.role_id("assistant"),
        ]


def encode_personal_dialogue_training_target(
    tokenizer: Any,
    prompt: str,
    human_response: str,
    *,
    context_length: int,
) -> tuple[list[int], list[float]]:
    """Encode the shared assistant-role/content/end-turn/EOS SFT contract."""

    if not personal_dialogue_target_fits(
        tokenizer,
        prompt,
        human_response,
        context_length=context_length,
    ):
        raise PersonalTrainingError(
            "a human response exceeds the bounded personal-training context"
        )
    prompt_ids = tokenizer.encode(prompt.strip())
    response_ids = tokenizer.encode(human_response.strip())
    prompt_budget = (
        context_length + 1 - PERSONAL_DIALOGUE_STRUCTURAL_TOKENS - len(response_ids)
    )
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    tokens = [
        tokenizer.bos_id,
        tokenizer.role_id("user"),
        *prompt_ids,
        tokenizer.end_turn_id,
        tokenizer.role_id("assistant"),
        *response_ids,
        tokenizer.end_turn_id,
        tokenizer.eos_id,
    ]
    if len(tokens) - 1 > context_length:
        raise PersonalTrainingError("a dialogue example exceeds the training context")
    mask = [0.0] * (len(tokens) - 1)
    assistant_role_offset = 3 + len(prompt_ids)
    assistant_end_offset = assistant_role_offset + 1 + len(response_ids)
    for target_position in range(assistant_role_offset, assistant_end_offset + 1):
        mask[target_position - 1] = 1.0
    mask[-1] = 1.0
    return tokens, mask


@dataclass(frozen=True)
class PersonalTrainingConfig:
    """Local policy.  No setting can enable upload or foundation training."""

    enabled: bool = True
    artifact_root: Path = Path("data/personal-model")
    minimum_holdout_improvement: float = 0.0
    maximum_replay_examples: int = PERSONAL_DEFAULT_REPLAY_EXAMPLES
    retention_examples: int = PERSONAL_DEFAULT_RETENTION_EXAMPLES
    maximum_retention_loss_increase: float = 0.05
    maximum_adapter_bytes: int = 512 * 1024 * 1024

    @classmethod
    def from_env(
        cls,
        *,
        project_root: str | Path | None = None,
    ) -> "PersonalTrainingConfig":
        root = Path(project_root or Path.cwd()).expanduser().resolve()
        enabled_text = os.getenv("MLM_PERSONAL_TRAINING_AUTO", "1").strip().lower()
        if enabled_text not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("MLM_PERSONAL_TRAINING_AUTO must be true or false")
        configured_root = (
            os.getenv(
                "MLM_PERSONAL_MODEL_DIR",
                "data/personal-model",
            ).strip()
            or "data/personal-model"
        )
        candidate = Path(configured_root).expanduser()
        artifact_root = candidate if candidate.is_absolute() else root / candidate
        improvement_text = os.getenv(
            "MLM_PERSONAL_MIN_HOLDOUT_IMPROVEMENT",
            "0",
        ).strip()
        maximum_text = os.getenv(
            "MLM_PERSONAL_MAX_ADAPTER_BYTES",
            str(512 * 1024 * 1024),
        ).strip()
        replay_text = os.getenv(
            "MLM_PERSONAL_MAX_REPLAY_EXAMPLES",
            str(PERSONAL_DEFAULT_REPLAY_EXAMPLES),
        ).strip()
        retention_examples_text = os.getenv(
            "MLM_PERSONAL_RETENTION_EXAMPLES",
            str(PERSONAL_DEFAULT_RETENTION_EXAMPLES),
        ).strip()
        retention_increase_text = os.getenv(
            "MLM_PERSONAL_MAX_RETENTION_LOSS_INCREASE",
            "0.05",
        ).strip()
        return cls(
            enabled=enabled_text in {"1", "true", "yes", "on"},
            artifact_root=Path(os.path.abspath(artifact_root)),
            minimum_holdout_improvement=float(improvement_text),
            maximum_replay_examples=int(replay_text),
            retention_examples=int(retention_examples_text),
            maximum_retention_loss_increase=float(retention_increase_text),
            maximum_adapter_bytes=int(maximum_text),
        )

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise PersonalTrainingError(
                "personal training enabled flag must be boolean"
            )
        if (
            not math.isfinite(self.minimum_holdout_improvement)
            or self.minimum_holdout_improvement < 0.0
        ):
            raise PersonalTrainingError(
                "minimum holdout improvement must be finite and non-negative"
            )
        if self.maximum_adapter_bytes <= 0:
            raise PersonalTrainingError("maximum adapter size must be positive")
        if (
            type(self.maximum_replay_examples) is not int
            or not 0 <= self.maximum_replay_examples <= PERSONAL_MAX_REPLAY_EXAMPLES
        ):
            raise PersonalTrainingError(
                f"maximum replay examples must be between 0 and {PERSONAL_MAX_REPLAY_EXAMPLES}"
            )
        if (
            type(self.retention_examples) is not int
            or not 0 <= self.retention_examples <= PERSONAL_MAX_RETENTION_EXAMPLES
        ):
            raise PersonalTrainingError(
                f"retention examples must be between 0 and {PERSONAL_MAX_RETENTION_EXAMPLES}"
            )
        if (
            isinstance(self.maximum_retention_loss_increase, bool)
            or not isinstance(self.maximum_retention_loss_increase, (int, float))
            or not math.isfinite(float(self.maximum_retention_loss_increase))
            or self.maximum_retention_loss_increase < 0.0
        ):
            raise PersonalTrainingError(
                "maximum retention loss increase must be finite and non-negative"
            )


@dataclass(frozen=True)
class PersonalRuntimeIdentity:
    """Compatibility identity reported before and after adapter activation."""

    backend: str
    foundation_sha256: str
    adapter_schema_sha256: str
    active_adapter_id: str | None = None

    def validate(self) -> None:
        _validate_identifier(self.backend, "backend")
        _validate_sha256(self.foundation_sha256, "foundation_sha256")
        _validate_sha256(self.adapter_schema_sha256, "adapter_schema_sha256")
        if self.active_adapter_id is not None:
            _validate_identifier(self.active_adapter_id, "active_adapter_id")


@dataclass(frozen=True)
class DialogueExample:
    lesson_id: int
    session_id: str
    prompt: str
    human_response: str
    content_sha256: str


@dataclass(frozen=True)
class PersonalAdapterTrainingRequest:
    """Immutable data handed to a backend-specific isolated trainer."""

    run_id: int
    adapter_id: str
    backend: str
    foundation_sha256: str
    adapter_schema_sha256: str
    parent_adapter_id: str | None
    parent_adapter_path: Path | None
    training_examples: tuple[DialogueExample, ...]
    validation_examples: tuple[DialogueExample, ...]
    work_dir: Path
    replay_examples: tuple[DialogueExample, ...] = ()
    retention_examples: tuple[DialogueExample, ...] = ()
    cancellation_event: threading.Event | None = None


@dataclass(frozen=True)
class PersonalAdapterCandidate:
    """A complete cumulative adapter produced by one backend trainer."""

    artifact_path: Path
    artifact_format: str
    artifact_sha256: str
    backend: str
    foundation_sha256_before: str
    foundation_sha256_after: str
    adapter_schema_sha256: str
    baseline_holdout_loss: float
    candidate_holdout_loss: float
    metrics: Mapping[str, Any]
    baseline_retention_loss: float | None = None
    candidate_retention_loss: float | None = None


@dataclass(frozen=True)
class PersonalAdapterManifest:
    adapter_id: str
    backend: str
    foundation_sha256: str
    adapter_schema_sha256: str
    parent_adapter_id: str | None
    artifact_name: str
    artifact_sha256: str
    artifact_bytes: int
    training_lesson_ids: tuple[int, ...]
    validation_lesson_ids: tuple[int, ...]
    run_id: int
    baseline_holdout_loss: float
    candidate_holdout_loss: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PERSONAL_ADAPTER_FORMAT,
            "visibility": "private-local-only",
            "adapter_id": self.adapter_id,
            "backend": self.backend,
            "foundation_sha256": self.foundation_sha256,
            "adapter_schema_sha256": self.adapter_schema_sha256,
            "parent_adapter_id": self.parent_adapter_id,
            "artifact": {
                "name": self.artifact_name,
                "sha256": self.artifact_sha256,
                "bytes": self.artifact_bytes,
            },
            "partition": {
                "training_lesson_ids": list(self.training_lesson_ids),
                "validation_lesson_ids": list(self.validation_lesson_ids),
            },
            "run_id": self.run_id,
            "evaluation": {
                "baseline_holdout_loss": self.baseline_holdout_loss,
                "candidate_holdout_loss": self.candidate_holdout_loss,
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersonalAdapterManifest":
        if set(payload) != {
            "format",
            "visibility",
            "adapter_id",
            "backend",
            "foundation_sha256",
            "adapter_schema_sha256",
            "parent_adapter_id",
            "artifact",
            "partition",
            "run_id",
            "evaluation",
            "created_at",
        }:
            raise PersonalTrainingError(
                "personal adapter manifest fields are incompatible"
            )
        if (
            payload["format"] != PERSONAL_ADAPTER_FORMAT
            or payload["visibility"] != "private-local-only"
        ):
            raise PersonalTrainingError(
                "personal adapter manifest format is incompatible"
            )
        artifact = payload["artifact"]
        partition = payload["partition"]
        evaluation = payload["evaluation"]
        if not isinstance(artifact, dict) or set(artifact) != {
            "name",
            "sha256",
            "bytes",
        }:
            raise PersonalTrainingError(
                "personal adapter artifact descriptor is invalid"
            )
        if not isinstance(partition, dict) or set(partition) != {
            "training_lesson_ids",
            "validation_lesson_ids",
        }:
            raise PersonalTrainingError("personal adapter partition is invalid")
        if not isinstance(evaluation, dict) or set(evaluation) != {
            "baseline_holdout_loss",
            "candidate_holdout_loss",
        }:
            raise PersonalTrainingError("personal adapter evaluation is invalid")
        manifest = cls(
            adapter_id=payload["adapter_id"],
            backend=payload["backend"],
            foundation_sha256=payload["foundation_sha256"],
            adapter_schema_sha256=payload["adapter_schema_sha256"],
            parent_adapter_id=payload["parent_adapter_id"],
            artifact_name=artifact["name"],
            artifact_sha256=artifact["sha256"],
            artifact_bytes=artifact["bytes"],
            training_lesson_ids=tuple(partition["training_lesson_ids"]),
            validation_lesson_ids=tuple(partition["validation_lesson_ids"]),
            run_id=payload["run_id"],
            baseline_holdout_loss=evaluation["baseline_holdout_loss"],
            candidate_holdout_loss=evaluation["candidate_holdout_loss"],
            created_at=payload["created_at"],
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        _validate_identifier(self.adapter_id, "adapter_id")
        _validate_identifier(self.backend, "backend")
        _validate_sha256(self.foundation_sha256, "foundation_sha256")
        _validate_sha256(self.adapter_schema_sha256, "adapter_schema_sha256")
        if self.parent_adapter_id is not None:
            _validate_identifier(self.parent_adapter_id, "parent_adapter_id")
        if (
            not isinstance(self.artifact_name, str)
            or Path(self.artifact_name).name != self.artifact_name
            or not self.artifact_name.startswith("adapter.")
        ):
            raise PersonalTrainingError("personal adapter artifact name is invalid")
        _validate_sha256(self.artifact_sha256, "artifact_sha256")
        if type(self.artifact_bytes) is not int or self.artifact_bytes <= 0:
            raise PersonalTrainingError("personal adapter artifact size is invalid")
        _validate_partition(self.training_lesson_ids, self.validation_lesson_ids)
        if type(self.run_id) is not int or self.run_id <= 0:
            raise PersonalTrainingError("personal adapter run ID is invalid")
        _validate_losses(self.baseline_holdout_loss, self.candidate_holdout_loss)
        if not isinstance(self.created_at, str) or not self.created_at:
            raise PersonalTrainingError("personal adapter creation time is invalid")


@dataclass(frozen=True)
class PersonalAdapterArtifact:
    directory: Path
    artifact_path: Path
    manifest_path: Path
    manifest: PersonalAdapterManifest


@runtime_checkable
class PersonalAdapterTrainer(Protocol):
    """Backend seam implemented first by MLX and later by PyTorch."""

    @property
    def backend_name(self) -> str: ...

    def train_adapter(
        self,
        request: PersonalAdapterTrainingRequest,
    ) -> PersonalAdapterCandidate: ...


@runtime_checkable
class PersonalAdapterRuntime(Protocol):
    """Inference seam; activation must be atomic from the caller's perspective."""

    def personal_runtime_identity(self) -> PersonalRuntimeIdentity: ...

    def prepare_personal_adapter(self, artifact: PersonalAdapterArtifact) -> object: ...

    def activate_prepared_personal_adapter(
        self,
        prepared: object,
        adapter_id: str,
    ) -> None: ...

    def activate_foundation_only(self) -> None: ...


class PrivateAdapterRegistry:
    """Private, checksum-verified local adapter archive and active pointer."""

    def __init__(self, root: str | Path, *, maximum_adapter_bytes: int) -> None:
        self.root = Path(root).expanduser().resolve()
        self.adapters = self.root / "adapters"
        self.work = self.root / "work"
        self.state_path = self.root / "state.json"
        self.maximum_adapter_bytes = maximum_adapter_bytes
        for directory in (self.root, self.adapters, self.work):
            _ensure_private_directory(directory)

    def create_work_dir(self, adapter_id: str) -> Path:
        _validate_identifier(adapter_id, "adapter_id")
        path = Path(tempfile.mkdtemp(prefix=f".{adapter_id}-", dir=self.work))
        _make_private(path, directory=True)
        return path

    def discard_work_dir(self, path: Path) -> None:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.work)
        except ValueError as error:
            raise PersonalTrainingError(
                "refusing to remove a non-work directory"
            ) from error
        if resolved.exists():
            shutil.rmtree(resolved)

    def clear_stale_work_dirs(self) -> int:
        """Remove only direct, non-symlink work directories under a held lease."""

        removed = 0
        for entry in tuple(self.work.iterdir()):
            try:
                details = entry.lstat()
                resolved = entry.resolve()
                resolved.relative_to(self.work)
            except (OSError, ValueError):
                continue
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                continue
            try:
                shutil.rmtree(resolved)
            except OSError:
                continue
            removed += 1
        return removed

    def prune_unreferenced_adapters(self) -> int:
        """Keep the verified current/previous pair and remove valid stale state."""

        current, previous, _ = self.read_state()
        retained = {identifier for identifier in (current, previous) if identifier}
        removed = 0
        for entry in tuple(self.adapters.iterdir()):
            if entry.name in retained:
                continue
            try:
                details = entry.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                continue
            if entry.name.startswith("."):
                try:
                    resolved = entry.resolve()
                    resolved.relative_to(self.adapters)
                except (OSError, ValueError):
                    continue
                try:
                    shutil.rmtree(resolved)
                except OSError:
                    continue
                removed += 1
                continue
            if not _IDENTIFIER_RE.fullmatch(entry.name):
                continue
            try:
                artifact = self.load(entry.name)
            except (OSError, PersonalTrainingError):
                continue
            try:
                shutil.rmtree(artifact.directory)
            except OSError:
                continue
            removed += 1
        return removed

    def publish(
        self,
        request: PersonalAdapterTrainingRequest,
        candidate: PersonalAdapterCandidate,
    ) -> PersonalAdapterArtifact:
        _validate_candidate(request, candidate)
        source = candidate.artifact_path.expanduser().resolve()
        try:
            source.relative_to(request.work_dir.resolve())
        except ValueError as error:
            raise PersonalTrainingError(
                "trainer artifact escaped its private work directory"
            ) from error
        artifact_name = f"adapter.{candidate.artifact_format}"
        destination = self.adapters / request.adapter_id
        if destination.exists() or destination.is_symlink():
            raise PersonalTrainingError("personal adapter identity already exists")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{request.adapter_id}-", dir=self.adapters)
        )
        _make_private(temporary, directory=True)
        try:
            artifact_path = temporary / artifact_name
            size, digest = _copy_private_regular_file(
                source,
                artifact_path,
                maximum=self.maximum_adapter_bytes,
            )
            if digest != candidate.artifact_sha256:
                raise PersonalTrainingError(
                    "trainer artifact checksum changed before publication"
                )
            manifest = PersonalAdapterManifest(
                adapter_id=request.adapter_id,
                backend=request.backend,
                foundation_sha256=request.foundation_sha256,
                adapter_schema_sha256=request.adapter_schema_sha256,
                parent_adapter_id=request.parent_adapter_id,
                artifact_name=artifact_name,
                artifact_sha256=digest,
                artifact_bytes=size,
                training_lesson_ids=tuple(
                    example.lesson_id for example in request.training_examples
                ),
                validation_lesson_ids=tuple(
                    example.lesson_id for example in request.validation_examples
                ),
                run_id=request.run_id,
                baseline_holdout_loss=float(candidate.baseline_holdout_loss),
                candidate_holdout_loss=float(candidate.candidate_holdout_loss),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            manifest.validate()
            _atomic_private_json(temporary / "manifest.json", manifest.to_dict())
            os.rename(temporary, destination)
            _fsync_directory(self.adapters)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return self.load(request.adapter_id)

    def load(self, adapter_id: str) -> PersonalAdapterArtifact:
        _validate_identifier(adapter_id, "adapter_id")
        directory = self.adapters / adapter_id
        details = directory.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise PersonalTrainingError("personal adapter directory is unsafe")
        manifest_path = directory / "manifest.json"
        payload = _read_regular_file(manifest_path, _MAX_MANIFEST_BYTES)
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PersonalTrainingError(
                "personal adapter manifest is invalid JSON"
            ) from error
        if not isinstance(decoded, dict):
            raise PersonalTrainingError(
                "personal adapter manifest root must be an object"
            )
        manifest = PersonalAdapterManifest.from_dict(decoded)
        if manifest.adapter_id != adapter_id:
            raise PersonalTrainingError("personal adapter directory identity changed")
        artifact_path = directory / manifest.artifact_name
        if {entry.name for entry in directory.iterdir()} != {
            "manifest.json",
            manifest.artifact_name,
        }:
            raise PersonalTrainingError(
                "personal adapter directory contains unexpected files"
            )
        size, digest = _hash_regular_file(
            artifact_path,
            maximum=self.maximum_adapter_bytes,
        )
        if size != manifest.artifact_bytes or digest != manifest.artifact_sha256:
            raise PersonalTrainingError(
                "personal adapter artifact does not match its manifest"
            )
        return PersonalAdapterArtifact(
            directory, artifact_path, manifest_path, manifest
        )

    def read_state(self) -> tuple[str | None, str | None, int]:
        if not self.state_path.exists():
            return None, None, 0
        payload = _read_regular_file(self.state_path, _MAX_MANIFEST_BYTES)
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PersonalTrainingError(
                "personal adapter state is invalid JSON"
            ) from error
        if not isinstance(decoded, dict) or set(decoded) != {
            "format",
            "current_adapter_id",
            "previous_adapter_id",
            "generation",
        }:
            raise PersonalTrainingError(
                "personal adapter state fields are incompatible"
            )
        if decoded["format"] != PERSONAL_STATE_FORMAT:
            raise PersonalTrainingError("personal adapter state format is incompatible")
        current = decoded["current_adapter_id"]
        previous = decoded["previous_adapter_id"]
        generation = decoded["generation"]
        for value, label in ((current, "current"), (previous, "previous")):
            if value is not None:
                _validate_identifier(value, f"{label}_adapter_id")
        if type(generation) is not int or generation < 0:
            raise PersonalTrainingError("personal adapter generation is invalid")
        return current, previous, generation

    def write_state(
        self,
        *,
        current: str | None,
        previous: str | None,
        generation: int,
    ) -> None:
        for value, label in ((current, "current"), (previous, "previous")):
            if value is not None:
                self.load(value)
                _validate_identifier(value, f"{label}_adapter_id")
        if type(generation) is not int or generation < 0:
            raise PersonalTrainingError("personal adapter generation is invalid")
        _atomic_private_json(
            self.state_path,
            {
                "format": PERSONAL_STATE_FORMAT,
                "current_adapter_id": current,
                "previous_adapter_id": previous,
                "generation": generation,
            },
        )

    def discard_unreferenced(self, adapter_id: str) -> None:
        current, previous, _ = self.read_state()
        if adapter_id in {current, previous}:
            raise PersonalTrainingError(
                "refusing to remove a referenced personal adapter"
            )
        destination = self.adapters / adapter_id
        if destination.exists():
            artifact = self.load(adapter_id)
            shutil.rmtree(artifact.directory)


class PersonalTrainingCoordinator:
    """Train and activate one private adapter after every five typed lessons."""

    def __init__(
        self,
        store: ApprenticeshipStore,
        config: PersonalTrainingConfig,
        *,
        trainer: PersonalAdapterTrainer,
        runtime: PersonalAdapterRuntime,
        recover_interrupted: bool = False,
    ) -> None:
        self.store = store
        self.config = config
        self.trainer = trainer
        self.runtime = runtime
        self.config.validate()
        self.registry = PrivateAdapterRegistry(
            config.artifact_root,
            maximum_adapter_bytes=config.maximum_adapter_bytes,
        )
        self._registry_id = hashlib.sha256(
            f"tinydog-personal-registry-v1\0{self.registry.root}".encode("utf-8")
        ).hexdigest()
        self._mutex = threading.RLock()
        self._thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._rerun_requested = False
        self._state = "disabled" if not config.enabled else "idle"
        self._last_error: str | None = None
        self._run_id: int | None = None
        self._adapter_id: str | None = None
        self._last_replay_examples = 0
        self._last_retention_examples = 0
        self._known_ineligible_pending = 0
        self._consumed_content_hashes: set[str] = set()
        self._validate_backend_contract()
        if config.enabled:
            if recover_interrupted:
                self.registry.clear_stale_work_dirs()
                self._recover_interrupted_runs()
            self._recover_unavailable_active_pointer()
            self._reconcile_incomplete_promotions()
            # Rebuild only after recovery has returned interrupted claims and
            # incomplete promotions to pending. Building before reconciliation
            # would incorrectly classify those crash-retry lessons as already
            # consumed.
            self._consumed_content_hashes = {
                _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
                for lesson in self.store.list_personal_consumed_dialogue_lessons(
                    self._registry_id,
                    limit=None,
                )
            }
            self.restore_active_adapter()
            self.registry.prune_unreferenced_adapters()

    def consider_training(self) -> bool:
        if not self.config.enabled or self._shutdown.is_set():
            return False
        with self._mutex:
            if self._thread is not None and self._thread.is_alive():
                self._rerun_requested = True
                return False
            qualifying, _ = self._estimated_pending_counts()
            if qualifying < PERSONAL_TRAINING_THRESHOLD:
                return False
            self._state = "queued"
            self._last_error = None
            self._thread = threading.Thread(
                target=self._worker,
                name="tinydog-personal-trainer",
                daemon=True,
            )
            self._thread.start()
            return True

    def consider_lesson(self, lesson_id: int) -> bool:
        """Classify one newly committed chat lesson before counting it.

        The portable service calls this immediately after its atomic dialogue
        commit. Already-consumed content and targets that cannot fit the
        backend's frozen personal context are durably quarantined, so five
        duplicate or oversized turns cannot start a useless retry cycle.
        Legacy callers may continue to call :meth:`consider_training`; the
        worker performs the same defensive classification over pending rows.
        """

        if type(lesson_id) is not int or lesson_id <= 0:
            raise ValueError("lesson_id must be a positive integer")
        if not self.config.enabled or self._shutdown.is_set():
            return False
        lesson = self.store.get_lesson(lesson_id)
        if lesson is None or lesson.status != "pending":
            return self.consider_training()
        if not _is_personal_dialogue_candidate(lesson):
            return self.consider_training()
        digest = _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
        with self._mutex:
            consumed = digest in self._consumed_content_hashes
        if consumed:
            self.store.quarantine_pending_lesson_ids(
                [lesson.id],
                reason="already-consumed-private-dialogue",
            )
            return self.consider_training()
        if not self._is_trainable_typed_lesson(lesson):
            self.store.quarantine_pending_lesson_ids(
                [lesson.id],
                reason="outside-personal-training-context",
            )
        return self.consider_training()

    def trigger(self, *, force: bool = False) -> bool:
        # Force never bypasses the five distinct, provenance-checked examples.
        del force
        return self.consider_training()

    def shutdown(self, timeout: float = 5.0) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0.0
        ):
            raise ValueError("shutdown timeout must be finite and non-negative")
        self._shutdown.set()
        with self._mutex:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=float(timeout))
            if thread.is_alive():
                raise PersonalTrainingError(
                    "personal trainer did not stop; retaining the exclusive service lease"
                )

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
            thread.join(timeout=remaining)

    def status(self) -> dict[str, Any]:
        qualifying, ineligible = self._estimated_pending_counts()
        identity = self.runtime.personal_runtime_identity()
        identity.validate()
        with self._mutex:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "state": self._state,
                "authority": "personal-local",
                "privacy": "private-local-only",
                "backend": identity.backend,
                "parameter_scope": "adapter-only",
                "foundation_trainable": False,
                "requires_signatures": False,
                "produces_community_updates": False,
                "uploads_enabled": False,
                "running": running,
                "qualifying_pending": qualifying,
                "ineligible_pending": ineligible,
                "quarantined_lessons": self.store.count_lessons(status="ineligible"),
                "lesson_threshold": PERSONAL_TRAINING_THRESHOLD,
                "progress": min(qualifying, PERSONAL_TRAINING_THRESHOLD)
                / PERSONAL_TRAINING_THRESHOLD,
                "training_examples_per_run": PERSONAL_TRAINING_EXAMPLES,
                "holdout_examples_per_run": PERSONAL_HOLDOUT_EXAMPLES,
                "maximum_replay_examples_per_run": self.config.maximum_replay_examples,
                "maximum_retention_examples_per_run": self.config.retention_examples,
                "last_replay_examples": self._last_replay_examples,
                "last_retention_examples": self._last_retention_examples,
                "maximum_retention_loss_increase": (
                    self.config.maximum_retention_loss_increase
                ),
                "historical_used_examples": (
                    self.store.count_personal_used_dialogue_lessons(self._registry_id)
                ),
                "replay_policy": "deterministic-bounded-private-local",
                "retention_policy": "stable-oldest-private-local",
                "accepted_input": "typed-human-dialogue",
                "raw_voice_counts": False,
                "voice_reason": (
                    "Raw audio and VAD boundaries have no trusted transcript or "
                    "human-authored semantic target."
                ),
                "run_id": self._run_id,
                "active_adapter_id": identity.active_adapter_id,
                "last_adapter_id": self._adapter_id,
                "rollback_available": self.registry.read_state()[1] is not None
                or identity.active_adapter_id is not None,
                "live_model_updated": identity.active_adapter_id is not None,
                "last_error": self._last_error,
            }

    def restore_active_adapter(self) -> None:
        """Restore the private active pointer without trusting file names alone."""

        current, _, _ = self.registry.read_state()
        if current is None:
            self.runtime.activate_foundation_only()
        else:
            artifact = self.registry.load(current)
            self._validate_artifact_compatibility(artifact)
            prepared = self.runtime.prepare_personal_adapter(artifact)
            self.runtime.activate_prepared_personal_adapter(prepared, current)
        identity = self.runtime.personal_runtime_identity()
        identity.validate()
        if identity.active_adapter_id != current:
            raise PersonalTrainingError(
                "runtime did not restore the private active adapter"
            )

    def rollback(self) -> dict[str, Any]:
        with self._mutex:
            if self._thread is not None and self._thread.is_alive():
                raise PersonalTrainingError(
                    "cannot roll back while personal training is active"
                )
            current, previous, generation = self.registry.read_state()
            if current is None:
                raise PersonalTrainingError("no personal adapter is active")
            current_artifact = self.registry.load(current)
            if previous is None:
                prepared: object | None = None
                next_previous: str | None = None
            else:
                previous_artifact = self.registry.load(previous)
                self._validate_artifact_compatibility(previous_artifact)
                prepared = self.runtime.prepare_personal_adapter(previous_artifact)
                ancestor = previous_artifact.manifest.parent_adapter_id
                next_previous = None
                if ancestor is not None:
                    try:
                        ancestor_artifact = self.registry.load(ancestor)
                        self._validate_artifact_compatibility(ancestor_artifact)
                    except (FileNotFoundError, OSError, PersonalTrainingError):
                        pass
                    else:
                        next_previous = ancestor
            if prepared is None:
                self.runtime.activate_foundation_only()
            else:
                self.runtime.activate_prepared_personal_adapter(prepared, previous)
            identity = self.runtime.personal_runtime_identity()
            if identity.active_adapter_id != previous:
                self._reactivate(current_artifact)
                raise PersonalTrainingError(
                    "runtime did not activate the rollback target"
                )
            try:
                self.registry.write_state(
                    current=previous,
                    previous=next_previous,
                    generation=generation + 1,
                )
                retire_ids: set[int] = set()
                run = self.store.get_training_status(current_artifact.manifest.run_id)
                if run is not None and run.status == "promoted":
                    retire_ids.add(run.id)
                if next_previous is None:
                    active_run_id = (
                        previous_artifact.manifest.run_id
                        if previous is not None
                        else None
                    )
                    retire_ids.update(
                        candidate.id
                        for candidate in self.store.list_training_runs(
                            status="promoted", limit=10_000
                        )
                        if candidate.id != active_run_id
                        and candidate.metadata.get("authority") == "personal-local"
                        and candidate.metadata.get("personal_registry_id")
                        == self._registry_id
                    )
                if retire_ids:
                    self.store.retire_promoted_training_runs(
                        sorted(retire_ids),
                        metadata={
                            "explicit_user_rollback": True,
                            "lessons_requeued": False,
                        },
                    )
            except Exception:
                self._reactivate(current_artifact)
                self.registry.write_state(
                    current=current,
                    previous=previous,
                    generation=generation,
                )
                raise
            try:
                self.registry.prune_unreferenced_adapters()
            except (OSError, PersonalTrainingError):
                # The pointer, runtime, and retired ledger are already in a
                # consistent state. Cleanup is bounded and retried on startup;
                # it must never undo an explicit rollback commit.
                pass
            self._state = "rolled_back"
            return self.status()

    def _worker(self) -> None:
        allow_rerun = True
        try:
            all_selected = self._select_typed_lessons(limit=None)
            with self._mutex:
                # _select_typed_lessons durably moves every rejected row that
                # it inspected out of pending. Never subtract those rows a
                # second time from the already-updated SQL count.
                self._known_ineligible_pending = 0
            selected = all_selected[:PERSONAL_TRAINING_THRESHOLD]
            if len(selected) != PERSONAL_TRAINING_THRESHOLD:
                return
            identity = self.runtime.personal_runtime_identity()
            identity.validate()
            current, _, generation = self.registry.read_state()
            if identity.active_adapter_id != current:
                raise PersonalTrainingError(
                    "runtime active adapter differs from the private local state"
                )
            run, claimed = self.store.claim_pending_lesson_ids(
                [lesson.id for lesson in selected],
                checkpoint_in=current or "foundation-only",
                metadata={
                    "authority": "personal-local",
                    "privacy": "private-local-only",
                    "backend": identity.backend,
                    "parameter_scope": "adapter-only",
                    "foundation_trainable": False,
                    "owner_pid": os.getpid(),
                    "personal_registry_id": self._registry_id,
                },
            )
            if run is None or len(claimed) != PERSONAL_TRAINING_THRESHOLD:
                return
            self._run_id = run.id
            adapter_id = f"personal-{uuid.uuid4().hex}"
            self._adapter_id = adapter_id
            work_dir: Path | None = None
            published: PersonalAdapterArtifact | None = None
            try:
                work_dir = self.registry.create_work_dir(adapter_id)
                request = self._request(
                    run.id,
                    adapter_id,
                    identity,
                    current,
                    claimed,
                    work_dir,
                )
                with self._mutex:
                    self._last_replay_examples = len(request.replay_examples)
                    self._last_retention_examples = len(request.retention_examples)
                self._state = "training"
                candidate = self.trainer.train_adapter(request)
                _validate_candidate(request, candidate)
                improved = (
                    candidate.candidate_holdout_loss
                    + self.config.minimum_holdout_improvement
                    < candidate.baseline_holdout_loss
                )
                retention_preserved = _retention_preserved(
                    request,
                    candidate,
                    maximum_loss_increase=self.config.maximum_retention_loss_increase,
                )
                gate_metrics = _candidate_gate_metrics(
                    request,
                    candidate,
                    retention_preserved=retention_preserved,
                )
                if not improved or not retention_preserved:
                    # Consume a cleanly rejected batch so the same five turns
                    # cannot create an automatic retry loop.  The next five
                    # human turns form a fresh local experiment.
                    self.store.complete_local_floaty_run(
                        run.id,
                        [lesson.id for lesson in claimed],
                        status="local_rejected",
                        metrics=gate_metrics,
                        metadata={
                            "authority": "personal-local",
                            "privacy": "private-local-only",
                            "reason": (
                                "holdout-did-not-improve"
                                if not improved
                                else "historical-retention-regressed"
                            ),
                            "replay_lesson_ids": [
                                example.lesson_id for example in request.replay_examples
                            ],
                            "retention_lesson_ids": [
                                example.lesson_id
                                for example in request.retention_examples
                            ],
                            "live_model_updated": False,
                        },
                    )
                    self._remember_consumed(claimed)
                    self._state = "rejected"
                    return
                self._state = "validating"
                published = self.registry.publish(request, candidate)
                self._validate_artifact_compatibility(published)
                prepared = self.runtime.prepare_personal_adapter(published)
                # Consume the exact batch before activation.  If activation or
                # pointer publication fails, rollback_training_run requeues it.
                self.store.promote_training_run(
                    run.id,
                    [lesson.id for lesson in claimed],
                    checkpoint_out=str(published.directory),
                    metrics=gate_metrics,
                    metadata={
                        "authority": "personal-local",
                        "privacy": "private-local-only",
                        "adapter_id": adapter_id,
                        "backend": identity.backend,
                        "training_lesson_ids": [
                            example.lesson_id for example in request.training_examples
                        ],
                        "validation_lesson_ids": [
                            example.lesson_id for example in request.validation_examples
                        ],
                        "replay_lesson_ids": [
                            example.lesson_id for example in request.replay_examples
                        ],
                        "retention_lesson_ids": [
                            example.lesson_id for example in request.retention_examples
                        ],
                        "foundation_trainable": False,
                        "produces_community_updates": False,
                    },
                )
                self._state = "activating"
                self.runtime.activate_prepared_personal_adapter(prepared, adapter_id)
                after = self.runtime.personal_runtime_identity()
                after.validate()
                if (
                    after.backend != identity.backend
                    or after.foundation_sha256 != identity.foundation_sha256
                    or after.adapter_schema_sha256 != identity.adapter_schema_sha256
                    or after.active_adapter_id != adapter_id
                ):
                    raise PersonalTrainingError(
                        "runtime identity changed outside the personal adapter surface"
                    )
                self.registry.write_state(
                    current=adapter_id,
                    previous=current,
                    generation=generation + 1,
                )
                self._remember_consumed(claimed)
                try:
                    self.registry.prune_unreferenced_adapters()
                except (OSError, PersonalTrainingError):
                    # State publication is the promotion commit point. Stale
                    # private adapters are harmless and cleanup is retried at
                    # the next exclusive startup.
                    pass
                self._state = "ready"
            except Exception as error:
                allow_rerun = False
                latest = self.store.get_training_status(run.id)
                if latest is not None and latest.status == "promoted":
                    try:
                        self.store.rollback_training_run(run.id)
                    except Exception:
                        pass
                elif latest is not None and latest.status == "training":
                    self.store.release_training_run(
                        run.id,
                        "failed",
                        metadata={"authority": "personal-local"},
                        error=_bounded_error(error),
                    )
                if published is not None:
                    try:
                        self.registry.discard_unreferenced(
                            published.manifest.adapter_id
                        )
                    except Exception:
                        pass
                try:
                    self.restore_active_adapter()
                except Exception:
                    pass
                self._state = "failed"
                self._last_error = "Personal training failed safely; the previous adapter remains active."
            finally:
                try:
                    if work_dir is not None:
                        self.registry.discard_work_dir(work_dir)
                    else:
                        # create_work_dir may fail after mkdir but before it
                        # can return the path. The service lease guarantees no
                        # second personal worker owns another work directory.
                        self.registry.clear_stale_work_dirs()
                except (OSError, PersonalTrainingError):
                    # Validated stale-work cleanup is retried at next exclusive
                    # startup and never changes a model/ledger commit.
                    pass
        except Exception as error:
            allow_rerun = False
            self._state = "failed"
            self._last_error = _bounded_error(error)
        finally:
            has_another_batch = (
                allow_rerun
                and not self._shutdown.is_set()
                and self._estimated_pending_counts()[0] >= PERSONAL_TRAINING_THRESHOLD
            )
            with self._mutex:
                rerun = (
                    allow_rerun
                    and not self._shutdown.is_set()
                    and (self._rerun_requested or has_another_batch)
                )
                self._rerun_requested = False
                if rerun:
                    next_thread = threading.Thread(
                        target=self._worker,
                        name="tinydog-personal-trainer",
                        daemon=True,
                    )
                    self._thread = next_thread
                    # Starting under the mutex avoids a visibility gap where
                    # a request could mistake an assigned-but-unstarted worker
                    # for an idle coordinator and launch a duplicate.
                    next_thread.start()
                else:
                    self._thread = None

    def _request(
        self,
        run_id: int,
        adapter_id: str,
        identity: PersonalRuntimeIdentity,
        current: str | None,
        claimed: list[Lesson],
        work_dir: Path,
    ) -> PersonalAdapterTrainingRequest:
        examples = tuple(_dialogue_example(lesson) for lesson in claimed)
        ordered = sorted(
            examples, key=lambda item: (item.content_sha256, item.lesson_id)
        )
        validation = (ordered[-1],)
        training = tuple(ordered[:-1])
        _validate_partition(
            tuple(example.lesson_id for example in training),
            tuple(example.lesson_id for example in validation),
        )
        replay, retention = self._select_replay_and_retention(run_id)
        parent_path = self.registry.load(current).artifact_path if current else None
        return PersonalAdapterTrainingRequest(
            run_id=run_id,
            adapter_id=adapter_id,
            backend=identity.backend,
            foundation_sha256=identity.foundation_sha256,
            adapter_schema_sha256=identity.adapter_schema_sha256,
            parent_adapter_id=current,
            parent_adapter_path=parent_path,
            training_examples=training,
            validation_examples=validation,
            work_dir=work_dir,
            replay_examples=replay,
            retention_examples=retention,
            cancellation_event=self._shutdown,
        )

    def _select_typed_lessons(
        self, limit: int | None = PERSONAL_TRAINING_THRESHOLD
    ) -> list[Lesson]:
        pending = reversed(
            self.store.list_recent_lessons(
                limit=PERSONAL_PENDING_SCAN_LIMIT,
                pending_only=True,
            )
        )
        selected: list[Lesson] = []
        with self._mutex:
            consumed = set(self._consumed_content_hashes)
        seen: set[str] = set(consumed)
        consumed_duplicates: list[int] = []
        pending_duplicates: list[int] = []
        untrainable: list[int] = []
        for lesson in pending:
            if not _is_personal_dialogue_candidate(lesson):
                continue
            if not self._is_trainable_typed_lesson(lesson):
                untrainable.append(lesson.id)
                continue
            key = _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
            if key in seen:
                if key in consumed:
                    consumed_duplicates.append(lesson.id)
                else:
                    pending_duplicates.append(lesson.id)
                continue
            seen.add(key)
            selected.append(lesson)
            if limit is not None and len(selected) >= limit:
                break
        self.store.quarantine_pending_lesson_ids(
            consumed_duplicates,
            reason="already-consumed-private-dialogue",
        )
        self.store.quarantine_pending_lesson_ids(
            pending_duplicates,
            reason="duplicate-private-dialogue",
        )
        self.store.quarantine_pending_lesson_ids(
            untrainable,
            reason="outside-personal-training-context",
        )
        return selected

    def _remember_consumed(self, lessons: Sequence[Lesson]) -> None:
        digests = {
            _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
            for lesson in lessons
        }
        with self._mutex:
            self._consumed_content_hashes.update(digests)

    def _is_trainable_typed_lesson(self, lesson: Lesson) -> bool:
        if not _is_typed_dialogue_lesson(self.store, lesson):
            return False
        checker = getattr(self.trainer, "personal_training_target_fits", None)
        if not callable(checker):
            return True
        return bool(checker(lesson.problem, lesson.correct_answer))

    def _estimated_pending_counts(self) -> tuple[int, int]:
        raw = self.store.count_trusted_typed_dialogue_lessons(status="pending")
        with self._mutex:
            ineligible = min(raw, self._known_ineligible_pending)
        return max(0, raw - ineligible), ineligible

    def _select_used_typed_lessons(self) -> list[Lesson]:
        """Return this registry's prior consumed dialogue lessons, oldest first."""

        selected: list[Lesson] = []
        seen: set[str] = set()
        history = self.store.list_personal_consumed_dialogue_lessons(
            self._registry_id,
            limit=None,
            run_statuses=("promoted", "local_rejected"),
        )
        for lesson in history:
            if not self._is_trainable_typed_lesson(lesson):
                continue
            digest = _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
            if digest in seen:
                continue
            seen.add(digest)
            selected.append(lesson)
        return selected

    def _select_replay_and_retention(
        self,
        run_id: int,
    ) -> tuple[tuple[DialogueExample, ...], tuple[DialogueExample, ...]]:
        """Select bounded, disjoint history without persisting conversation text.

        The oldest unique examples form a stable retention anchor.  Replay is
        a deterministic run-specific sample from the remaining history, which
        provides coverage over time without an ever-growing training batch.
        """

        history = self._select_used_typed_lessons()
        retention_lessons = history[: self.config.retention_examples]
        retention_ids = {lesson.id for lesson in retention_lessons}
        replay_pool = [lesson for lesson in history if lesson.id not in retention_ids]

        def replay_score(lesson: Lesson) -> tuple[str, int]:
            content = _dialogue_content_sha256(lesson.problem, lesson.correct_answer)
            score = hashlib.sha256(
                (
                    f"tinydog-personal-replay-v1\0{self._registry_id}\0"
                    f"{run_id}\0{content}"
                ).encode("ascii")
            ).hexdigest()
            return score, lesson.id

        replay_lessons = sorted(replay_pool, key=replay_score)[
            : self.config.maximum_replay_examples
        ]
        replay = tuple(_dialogue_example(lesson) for lesson in replay_lessons)
        retention = tuple(_dialogue_example(lesson) for lesson in retention_lessons)
        _validate_history_sets(replay, retention)
        return replay, retention

    def _validate_backend_contract(self) -> None:
        if not isinstance(self.trainer, PersonalAdapterTrainer):
            raise PersonalTrainingError(
                "personal trainer does not implement the backend contract"
            )
        if not isinstance(self.runtime, PersonalAdapterRuntime):
            raise PersonalTrainingError(
                "personal runtime does not implement the backend contract"
            )
        identity = self.runtime.personal_runtime_identity()
        identity.validate()
        if self.trainer.backend_name != identity.backend:
            raise PersonalTrainingError(
                "personal trainer and runtime backends do not match"
            )

    def _reconcile_incomplete_promotions(self) -> None:
        """Undo ledger commits that never reached the durable active pointer.

        Promotion is deliberately committed before runtime activation.  A
        process loss in that small interval therefore leaves a newer promoted
        run than the registry's active adapter.  On restart, those newer runs
        are returned to the lesson queue before the known-good pointer is
        restored.
        """

        current, _, _ = self.registry.read_state()
        current_run_id = 0
        if current is not None:
            active_artifact = self.registry.load(current)
            current_run_id = active_artifact.manifest.run_id
            active_run = self.store.get_training_status(current_run_id)
            if (
                active_run is None
                or active_run.status != "promoted"
                or active_run.metadata.get("authority") != "personal-local"
                or active_run.metadata.get("personal_registry_id") != self._registry_id
            ):
                raise PersonalTrainingError(
                    "private active adapter is not backed by a promoted local run"
                )
        promoted = [
            run
            for run in self.store.list_training_runs(status="promoted", limit=10_000)
            if run.metadata.get("authority") == "personal-local"
            and run.metadata.get("personal_registry_id") == self._registry_id
            and run.id > current_run_id
        ]
        for run in promoted:
            adapter_id = run.metadata.get("adapter_id")
            self.store.rollback_training_run(run.id)
            if isinstance(adapter_id, str) and _IDENTIFIER_RE.fullmatch(adapter_id):
                try:
                    self.registry.discard_unreferenced(adapter_id)
                except (OSError, PersonalTrainingError):
                    # The unreferenced path is never loaded for inference.  A
                    # malformed orphan must not prevent restoration of the
                    # last checksum-verified adapter.
                    pass

    def _recover_unavailable_active_pointer(self) -> None:
        """Fall back safely when the durable current artifact is unavailable.

        Directory fsync makes this unlikely after clean local publication,
        but abrupt power loss or manual damage must not make the service
        unstartable. The exact missing-current run is requeued. A verified
        previous adapter is restored when possible; otherwise older promoted
        history is retired without requeue and recovery stops at foundation.
        """

        current, previous, generation = self.registry.read_state()
        if current is None:
            return
        try:
            active = self.registry.load(current)
            self._validate_artifact_compatibility(active)
        except (FileNotFoundError, OSError, PersonalTrainingError):
            pass
        else:
            return

        fallback: PersonalAdapterArtifact | None = None
        next_previous: str | None = None
        if previous is not None:
            try:
                candidate = self.registry.load(previous)
                self._validate_artifact_compatibility(candidate)
                run = self.store.get_training_status(candidate.manifest.run_id)
                if (
                    run is None
                    or run.status != "promoted"
                    or run.metadata.get("authority") != "personal-local"
                    or run.metadata.get("personal_registry_id") != self._registry_id
                ):
                    raise PersonalTrainingError(
                        "previous private adapter lacks a promoted local run"
                    )
            except (FileNotFoundError, OSError, PersonalTrainingError):
                pass
            else:
                fallback = candidate
                ancestor = candidate.manifest.parent_adapter_id
                if ancestor is not None:
                    try:
                        ancestor_artifact = self.registry.load(ancestor)
                        self._validate_artifact_compatibility(ancestor_artifact)
                    except (FileNotFoundError, OSError, PersonalTrainingError):
                        pass
                    else:
                        next_previous = ancestor

        self.registry.write_state(
            current=fallback.manifest.adapter_id if fallback is not None else None,
            previous=next_previous,
            generation=generation + 1,
        )
        promoted = [
            run
            for run in self.store.list_training_runs(status="promoted", limit=10_000)
            if run.metadata.get("authority") == "personal-local"
            and run.metadata.get("personal_registry_id") == self._registry_id
        ]
        missing_current_runs = [
            run for run in promoted if run.metadata.get("adapter_id") == current
        ]
        for run in missing_current_runs:
            self.store.rollback_training_run(run.id)
        if fallback is None:
            missing_ids = {run.id for run in missing_current_runs}
            stale_ids = [run.id for run in promoted if run.id not in missing_ids]
            if stale_ids:
                self.store.retire_promoted_training_runs(
                    stale_ids,
                    metadata={
                        "missing_active_artifact_recovery": True,
                        "lessons_requeued": False,
                    },
                )

    def _recover_interrupted_runs(self) -> None:
        """Requeue claims only after the caller has obtained the registry lease."""

        for run in self.store.list_training_runs(status="training", limit=10_000):
            if (
                run.metadata.get("authority") != "personal-local"
                or run.metadata.get("personal_registry_id") != self._registry_id
            ):
                continue
            self.store.release_training_run(
                run.id,
                "failed",
                error="Recovered an interrupted private local training run",
                metadata={"recovered": True, "recovered_by_pid": os.getpid()},
            )

    def _validate_artifact_compatibility(
        self, artifact: PersonalAdapterArtifact
    ) -> None:
        identity = self.runtime.personal_runtime_identity()
        manifest = artifact.manifest
        if (
            manifest.backend != identity.backend
            or manifest.foundation_sha256 != identity.foundation_sha256
            or manifest.adapter_schema_sha256 != identity.adapter_schema_sha256
        ):
            raise PersonalTrainingError(
                "personal adapter is incompatible with the active foundation"
            )

    def _reactivate(self, artifact: PersonalAdapterArtifact) -> None:
        prepared = self.runtime.prepare_personal_adapter(artifact)
        self.runtime.activate_prepared_personal_adapter(
            prepared,
            artifact.manifest.adapter_id,
        )


def _is_typed_dialogue_lesson(store: ApprenticeshipStore, lesson: Lesson) -> bool:
    if not _is_personal_dialogue_candidate(lesson):
        return False
    if not store.is_trusted_dialogue_lesson(lesson):
        return False
    for text in (lesson.problem, lesson.correct_answer):
        encoded = text.encode("utf-8")
        if not encoded or len(encoded) > _MAX_TEXT_BYTES or "\x00" in text:
            return False
    return True


def _is_personal_dialogue_candidate(lesson: Lesson) -> bool:
    """Whether a pending row claims the private typed-dialogue contract."""

    return (
        lesson.source == "human-conversation"
        and lesson.verifier == "trusted-human-dialogue-v1"
        and lesson.metadata.get("capture") == "text-chat"
    )


def _dialogue_example(lesson: Lesson) -> DialogueExample:
    if not lesson.session_id:
        raise PersonalTrainingError("dialogue lesson is missing its session identity")
    return DialogueExample(
        lesson_id=lesson.id,
        session_id=lesson.session_id,
        prompt=lesson.problem.strip(),
        human_response=lesson.correct_answer.strip(),
        content_sha256=_dialogue_content_sha256(lesson.problem, lesson.correct_answer),
    )


def _dialogue_content_sha256(prompt: str, response: str) -> str:
    normalized_prompt = prompt.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized_response = response.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(
        f"typed-dialogue-v1\0{normalized_prompt}\0{normalized_response}".encode("utf-8")
    ).hexdigest()


def _validate_candidate(
    request: PersonalAdapterTrainingRequest,
    candidate: PersonalAdapterCandidate,
) -> None:
    if not isinstance(candidate, PersonalAdapterCandidate):
        raise PersonalTrainingError("personal trainer returned an invalid candidate")
    _validate_identifier(candidate.backend, "candidate backend")
    if candidate.backend != request.backend:
        raise PersonalTrainingError("personal trainer returned a different backend")
    if (
        candidate.foundation_sha256_before != request.foundation_sha256
        or candidate.foundation_sha256_after != request.foundation_sha256
    ):
        raise PersonalTrainingError(
            "foundation changed during personal adapter training"
        )
    if candidate.adapter_schema_sha256 != request.adapter_schema_sha256:
        raise PersonalTrainingError("personal trainer changed the adapter schema")
    if not _ARTIFACT_FORMAT_RE.fullmatch(candidate.artifact_format):
        raise PersonalTrainingError("personal trainer artifact format is invalid")
    _validate_sha256(candidate.artifact_sha256, "candidate artifact_sha256")
    _validate_losses(candidate.baseline_holdout_loss, candidate.candidate_holdout_loss)
    _validate_history_sets(request.replay_examples, request.retention_examples)
    core_ids = {
        example.lesson_id
        for example in request.training_examples + request.validation_examples
    }
    history_ids = {
        example.lesson_id
        for example in request.replay_examples + request.retention_examples
    }
    if core_ids & history_ids:
        raise PersonalTrainingError("new and historical personal examples overlap")
    if request.retention_examples:
        _validate_optional_retention_losses(
            candidate.baseline_retention_loss,
            candidate.candidate_retention_loss,
            required=True,
        )
    else:
        _validate_optional_retention_losses(
            candidate.baseline_retention_loss,
            candidate.candidate_retention_loss,
            required=False,
        )
    if not isinstance(candidate.metrics, Mapping):
        raise PersonalTrainingError("personal trainer metrics must be a mapping")
    _validate_partition(
        tuple(example.lesson_id for example in request.training_examples),
        tuple(example.lesson_id for example in request.validation_examples),
    )


def _validate_partition(training: tuple[int, ...], validation: tuple[int, ...]) -> None:
    if (
        len(training) != PERSONAL_TRAINING_EXAMPLES
        or len(validation) != PERSONAL_HOLDOUT_EXAMPLES
    ):
        raise PersonalTrainingError(
            "personal training requires exactly four train and one holdout"
        )
    combined = training + validation
    if any(type(identifier) is not int or identifier <= 0 for identifier in combined):
        raise PersonalTrainingError("personal lesson IDs must be positive integers")
    if len(set(combined)) != PERSONAL_TRAINING_THRESHOLD:
        raise PersonalTrainingError(
            "personal train and holdout lessons must be distinct"
        )


def _validate_history_sets(
    replay: tuple[DialogueExample, ...],
    retention: tuple[DialogueExample, ...],
) -> None:
    if len(replay) > PERSONAL_MAX_REPLAY_EXAMPLES:
        raise PersonalTrainingError("personal replay set exceeds its hard bound")
    if len(retention) > PERSONAL_MAX_RETENTION_EXAMPLES:
        raise PersonalTrainingError("personal retention set exceeds its hard bound")
    examples = replay + retention
    lesson_ids = [example.lesson_id for example in examples]
    content_ids = [example.content_sha256 for example in examples]
    if any(type(identifier) is not int or identifier <= 0 for identifier in lesson_ids):
        raise PersonalTrainingError("historical personal lesson IDs are invalid")
    if len(set(lesson_ids)) != len(lesson_ids) or len(set(content_ids)) != len(
        content_ids
    ):
        raise PersonalTrainingError(
            "personal replay and retention examples must be distinct"
        )


def _validate_optional_retention_losses(
    baseline: float | None,
    candidate: float | None,
    *,
    required: bool,
) -> None:
    if not required:
        if baseline is not None or candidate is not None:
            raise PersonalTrainingError(
                "retention losses were reported without a retention set"
            )
        return
    if baseline is None or candidate is None:
        raise PersonalTrainingError(
            "personal trainer omitted historical retention losses"
        )
    _validate_losses(baseline, candidate)


def _retention_preserved(
    request: PersonalAdapterTrainingRequest,
    candidate: PersonalAdapterCandidate,
    *,
    maximum_loss_increase: float,
) -> bool:
    if not request.retention_examples:
        return True
    assert candidate.baseline_retention_loss is not None
    assert candidate.candidate_retention_loss is not None
    return (
        candidate.candidate_retention_loss
        <= candidate.baseline_retention_loss + maximum_loss_increase
    )


def _candidate_gate_metrics(
    request: PersonalAdapterTrainingRequest,
    candidate: PersonalAdapterCandidate,
    *,
    retention_preserved: bool,
) -> dict[str, Any]:
    return {
        "baseline_holdout_loss": candidate.baseline_holdout_loss,
        "candidate_holdout_loss": candidate.candidate_holdout_loss,
        "baseline_retention_loss": candidate.baseline_retention_loss,
        "candidate_retention_loss": candidate.candidate_retention_loss,
        "new_training_examples": len(request.training_examples),
        "replay_examples": len(request.replay_examples),
        "holdout_examples": len(request.validation_examples),
        "retention_examples": len(request.retention_examples),
        "retention_preserved": retention_preserved,
    }


def _validate_losses(baseline: float, candidate: float) -> None:
    if (
        isinstance(baseline, bool)
        or isinstance(candidate, bool)
        or not isinstance(baseline, (int, float))
        or not isinstance(candidate, (int, float))
        or not math.isfinite(float(baseline))
        or not math.isfinite(float(candidate))
        or baseline < 0.0
        or candidate < 0.0
    ):
        raise PersonalTrainingError(
            "personal holdout losses must be finite and non-negative"
        )


def _validate_identifier(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise PersonalTrainingError(f"{label} is invalid")


def _validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PersonalTrainingError(f"{label} must be a lowercase SHA-256 digest")


def _ensure_private_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise PersonalTrainingError("personal model path is not a safe directory")
    else:
        path.mkdir(parents=True, mode=0o700)
    _make_private(path, directory=True)


def _make_private(path: Path, *, directory: bool) -> None:
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError as error:
        if os.name != "nt":
            raise PersonalTrainingError(
                "could not make personal model artifacts private"
            ) from error


def _copy_private_regular_file(
    source: Path, destination: Path, *, maximum: int
) -> tuple[int, str]:
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, source_flags)
    destination_fd: int | None = None
    try:
        details = os.fstat(source_fd)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_size <= 0
            or details.st_size > maximum
        ):
            raise PersonalTrainingError(
                "personal trainer artifact is not a bounded regular file"
            )
        destination_fd = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        digest = hashlib.sha256()
        written = 0
        while True:
            chunk = os.read(source_fd, min(1024 * 1024, maximum + 1 - written))
            if not chunk:
                break
            written += len(chunk)
            if written > maximum:
                raise PersonalTrainingError(
                    "personal trainer artifact exceeds its size limit"
                )
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                consumed = os.write(destination_fd, view)
                view = view[consumed:]
        if written != details.st_size:
            raise PersonalTrainingError(
                "personal trainer artifact changed while publishing"
            )
        os.fsync(destination_fd)
        _make_private(destination, directory=False)
        return written, digest.hexdigest()
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


def _hash_regular_file(path: Path, *, maximum: int) -> tuple[int, str]:
    payload = _read_regular_file(path, maximum)
    if not payload:
        raise PersonalTrainingError("personal adapter artifact is empty")
    return len(payload), hashlib.sha256(payload).hexdigest()


def _read_regular_file(path: Path, maximum: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size > maximum:
            raise PersonalTrainingError("personal model artifact is unsafe")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != details.st_size:
            raise PersonalTrainingError("personal model artifact changed while reading")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise PersonalTrainingError("personal model metadata is too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _make_private(temporary, directory=False)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    """Persist a completed rename on filesystems supporting directory fsync."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # The file itself was fsynced before replace. Some synthetic and
            # network filesystems do not implement directory fsync; a
            # post-rename durability hint must not turn a visible commit into
            # an application-level rollback.
            pass
    finally:
        os.close(descriptor)


def _bounded_error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[-_MAX_ERROR_TEXT:]
