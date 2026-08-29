"""Local, fail-closed floaty training and community-update packaging.

The client accepts only an owner-signed HybridLM foundation release.  It
snapshots every signed artifact into a private directory, loads the signed
ScratchTokenizer pair, trains only FoundationFloatyLM.floaty, and emits a
fixed-schema delta bundle.  ArithmeticLM and full-model fallbacks do not exist
in this module.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np

from .apprenticeship import ApprenticeshipStore, verify_answer
from .community_updates import (
    CommunityContract,
    UpdateManifest,
    publish_directory_no_replace,
    read_floaty_safetensors,
    sha256_file,
    tensor_specs_from_floaty_schema,
    write_floaty_safetensors,
    write_update_bundle,
)
from .continual import render_verified_lesson
from .community_release import (
    ReleaseArtifact,
    VerifiedCommunityRelease,
    advance_trusted_community_state,
    verify_community_release,
)
from .floaty_training import (
    FloatyCandidateResult,
    FloatyTrainingConfig,
    floaty_training_recipe_sha256,
    run_floaty_candidate,
)
from .foundation_floaty import (
    FLOATY_STATE_VERSION,
    FloatyConfig,
    FloatyState,
    FoundationFloatyLM,
)
from .foundation_release import (
    VerifiedArtifact,
    VerifiedFoundation,
    verify_foundation_manifest,
)
from .hybrid import HybridConfig, HybridLM
from .scratch_tokenizer import (
    METADATA_FILENAME,
    TOKENIZER_FILENAME,
    ScratchTokenizer,
)


COMMUNITY_CLIENT_SCHEMA_VERSION = 1
BASE_FLOATY_FILENAME = "floaty.safetensors"
CONTRACT_FILENAME = "contract.json"
BOOTSTRAP_FILENAME = "bootstrap.json"
_SAFE_UPDATE_PREFIX = "local"


class CommunityClientError(ValueError):
    """A signed-release, local-data, or compatibility check failed."""


@dataclass(frozen=True)
class CommunityBootstrapResult:
    output_dir: Path
    weights_path: Path
    contract_path: Path
    contract: CommunityContract
    foundation_id: str


@dataclass(frozen=True)
class CommunityContributionResult:
    accepted: bool
    candidate: FloatyCandidateResult
    bundle_path: Path | None
    update_manifest: UpdateManifest | None
    training_lesson_ids: tuple[int | str, ...]
    validation_lesson_ids: tuple[int | str, ...]


@dataclass(frozen=True)
class _LoadedFoundation:
    verified: VerifiedFoundation
    model: FoundationFloatyLM
    tokenizer: ScratchTokenizer
    floaty_schema: Mapping[str, Any]


@dataclass(frozen=True)
class _LoadedCommunityRelease:
    verified: VerifiedCommunityRelease
    contract: CommunityContract
    weights: Mapping[str, np.ndarray]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CommunityClientError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise CommunityClientError(f"{label} must contain a JSON object")
    return value


def _snapshot_artifact(
    artifact: VerifiedArtifact | ReleaseArtifact,
    destination: Path,
) -> Path:
    """Copy one no-follow descriptor snapshot and recheck its signed digest."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(artifact.path, flags)
    except OSError as error:
        raise CommunityClientError(
            f"signed {artifact.name} artifact is unavailable"
        ) from error
    digest = hashlib.sha256()
    copied = 0
    try:
        metadata = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != artifact.byte_count
        ):
            raise CommunityClientError(
                f"signed {artifact.name} artifact changed before load"
            )
        with destination.open("xb") as output:
            remaining = artifact.byte_count
            while remaining:
                chunk = os.read(source_descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise CommunityClientError(
                        f"signed {artifact.name} artifact was truncated"
                    )
                output.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                remaining -= len(chunk)
            if os.read(source_descriptor, 1):
                raise CommunityClientError(
                    f"signed {artifact.name} artifact grew during load"
                )
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
    if copied != artifact.byte_count or digest.hexdigest() != artifact.sha256:
        destination.unlink(missing_ok=True)
        raise CommunityClientError(
            f"signed {artifact.name} artifact failed snapshot verification"
        )
    return destination


def _load_hybrid_config(path: Path) -> HybridConfig:
    value = _read_strict_json(path, "signed HybridConfig")
    allowed = {item.name for item in fields(HybridConfig)}
    if set(value) != allowed:
        raise CommunityClientError("signed model config is not an exact HybridConfig")
    schedule = value.get("layer_schedule")
    if not isinstance(schedule, list) or any(
        not isinstance(item, str) for item in schedule
    ):
        raise CommunityClientError("signed HybridConfig layer_schedule is invalid")
    payload = dict(value)
    payload["layer_schedule"] = tuple(schedule)
    try:
        return HybridConfig(**payload)
    except (TypeError, ValueError) as error:
        raise CommunityClientError("signed HybridConfig is invalid") from error


def _floaty_config_from_schema(value: Mapping[str, Any]) -> FloatyConfig:
    required = {
        "schema",
        "foundation_type",
        "foundation_config",
        "adapter_type",
        "rank",
        "alpha_hex",
        "attachments",
        "target_layers",
        "tensors",
    }
    if set(value) != required:
        raise CommunityClientError("signed floaty schema fields are incompatible")
    if value.get("foundation_type") != "mini_arithmetic.hybrid.HybridLM":
        raise CommunityClientError("signed foundation type is not HybridLM")
    if value.get("adapter_type") != "parallel-low-rank-residual":
        raise CommunityClientError("signed floaty adapter type is unsupported")
    attachments = value.get("attachments")
    targets = value.get("target_layers")
    if (
        not isinstance(attachments, list)
        or any(not isinstance(item, str) for item in attachments)
        or not isinstance(targets, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in targets)
        or not isinstance(value.get("alpha_hex"), str)
    ):
        raise CommunityClientError("signed floaty schema values are invalid")
    try:
        alpha = float.fromhex(value["alpha_hex"])
        return FloatyConfig(
            rank=value["rank"],
            alpha=alpha,
            attachments=tuple(attachments),
            target_layers=tuple(targets),
        )
    except (TypeError, ValueError) as error:
        raise CommunityClientError("signed floaty configuration is invalid") from error


@contextmanager
def load_signed_foundation(
    manifest_path: str | Path,
    trusted_public_key: str | Path,
) -> Iterator[_LoadedFoundation]:
    """Verify, privately snapshot, and strictly load one signed Hybrid release."""

    verified = verify_foundation_manifest(manifest_path, trusted_public_key)
    with tempfile.TemporaryDirectory(prefix="tinydog-foundation-") as temporary_name:
        root = Path(temporary_name)
        snapshots = {
            "weights": root / "foundation.safetensors",
            "model_config": root / "hybrid-config.json",
            "tokenizer": root / TOKENIZER_FILENAME,
            "tokenizer_metadata": root / METADATA_FILENAME,
            "floaty_schema": root / "floaty-schema.json",
        }
        for name, destination in snapshots.items():
            _snapshot_artifact(verified.artifacts[name], destination)

        config = _load_hybrid_config(snapshots["model_config"])
        tokenizer = ScratchTokenizer.load(root)
        if tokenizer.vocab_size != config.vocab_size:
            raise CommunityClientError(
                "signed tokenizer and HybridConfig vocabulary differ"
            )
        foundation = HybridLM(config)
        try:
            foundation.load_weights(str(snapshots["weights"]), strict=True)
        except (OSError, ValueError) as error:
            raise CommunityClientError(
                "signed foundation weights do not match HybridConfig"
            ) from error
        mx.eval(foundation.parameters())
        floaty_schema = _read_strict_json(
            snapshots["floaty_schema"], "signed floaty schema"
        )
        model = FoundationFloatyLM(
            foundation,
            _floaty_config_from_schema(floaty_schema),
        )
        if floaty_schema != model.config_manifest():
            raise CommunityClientError(
                "signed floaty schema does not match the loaded foundation"
            )
        model.assert_parameter_partition()
        yield _LoadedFoundation(verified, model, tokenizer, floaty_schema)


@contextmanager
def load_signed_community_release(
    manifest_path: str | Path,
    trusted_public_key: str | Path,
    *,
    foundation_manifest: str | Path,
) -> Iterator[_LoadedCommunityRelease]:
    """Verify and privately snapshot one owner-authorized active floaty release."""

    verified = verify_community_release(
        manifest_path,
        trusted_public_key,
        foundation_manifest=foundation_manifest,
    )
    with tempfile.TemporaryDirectory(prefix="tinydog-community-") as temporary_name:
        root = Path(temporary_name)
        contract_path = _snapshot_artifact(
            verified.artifacts["contract"],
            root / CONTRACT_FILENAME,
        )
        weights_path = _snapshot_artifact(
            verified.artifacts["weights"],
            root / BASE_FLOATY_FILENAME,
        )
        contract = CommunityContract.from_dict(
            _read_strict_json(contract_path, "signed community contract")
        )
        if contract.to_dict() != verified.contract.to_dict():
            raise CommunityClientError(
                "snapshotted contract differs from the signed release"
            )
        if contract.community_weights_sha256 != verified.artifacts["weights"].sha256:
            raise CommunityClientError(
                "signed community weights do not match the contract"
            )
        weights = read_floaty_safetensors(
            weights_path,
            contract.tensor_specs,
            expected_sha256=verified.artifacts["weights"].sha256,
        )
        yield _LoadedCommunityRelease(verified, contract, weights)


def _qualified_floaty_state(model: FoundationFloatyLM) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for name, value in tree_flatten(model.floaty.parameters()):
        converted = value.astype(mx.float32)
        mx.eval(converted)
        state[f"floaty.{name}"] = np.asarray(converted)
    if set(state) != set(model.floaty_parameter_names()):
        raise CommunityClientError(
            "model floaty state does not match its signed whitelist"
        )
    return state


def _import_qualified_floaty_state(
    model: FoundationFloatyLM,
    tensors: Mapping[str, np.ndarray],
) -> None:
    prefix = "floaty."
    if any(not name.startswith(prefix) for name in tensors):
        raise CommunityClientError("community weights contain a non-floaty tensor")
    state = FloatyState(
        format_version=FLOATY_STATE_VERSION,
        foundation_hash=model.foundation_hash(),
        config_hash=model.config_hash(),
        tensors={
            name.removeprefix(prefix): mx.array(value)
            for name, value in tensors.items()
        },
    )
    model.import_floaty_state(state)


def _validate_contract(
    loaded: _LoadedFoundation,
    contract: CommunityContract,
    training_config: FloatyTrainingConfig,
) -> None:
    tokenizer_sha256 = loaded.tokenizer.metadata.get("tokenizer_sha256")
    expected = {
        "foundation_sha256": loaded.model.foundation_hash(),
        "architecture_sha256": loaded.model.config_hash(),
        "tokenizer_sha256": tokenizer_sha256,
        "training_recipe_sha256": floaty_training_recipe_sha256(training_config),
    }
    for name, value in expected.items():
        if not isinstance(value, str) or getattr(contract, name) != value:
            raise CommunityClientError(
                f"community contract {name} does not match signed release"
            )
    signed_contract = CommunityContract.from_floaty_schema(
        loaded.floaty_schema["tensors"],
        foundation_sha256=contract.foundation_sha256,
        community_version=contract.community_version,
        community_weights_sha256=contract.community_weights_sha256,
        architecture_sha256=contract.architecture_sha256,
        tokenizer_sha256=contract.tokenizer_sha256,
        training_recipe_sha256=contract.training_recipe_sha256,
    )
    if signed_contract.to_dict() != contract.to_dict():
        raise CommunityClientError(
            "community contract tensor schema is not owner-signed"
        )


def _lesson_key(record: Any) -> str:
    problem = " ".join(str(getattr(record, "problem", "")).split()).casefold()
    answer = " ".join(str(getattr(record, "correct_answer", "")).split())
    return hashlib.sha256(f"{problem}\0{answer}".encode("utf-8")).hexdigest()


def deterministic_lesson_split(
    lessons: Sequence[Any],
    *,
    foundation_sha256: str,
    validation_fraction: float,
    minimum_training_lessons: int,
    minimum_validation_lessons: int,
    trusted_dialogue_validator: Callable[[Any], bool] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Deduplicate and deterministically reserve an independent held-out set.

    Exact arithmetic remains self-verifying.  A non-arithmetic record is
    eligible only when the owning lesson store revalidates its persisted human
    dialogue provenance; a caller-supplied verifier string is never enough.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    unique: dict[str, Any] = {}
    for lesson in lessons:
        render_verified_lesson(lesson)
        verification = verify_answer(
            str(getattr(lesson, "problem", "")),
            str(getattr(lesson, "correct_answer", "")),
        )
        if verification.expected is None or not verification.is_correct:
            trusted_dialogue = False
            if trusted_dialogue_validator is not None:
                try:
                    trusted_dialogue = trusted_dialogue_validator(lesson) is True
                except Exception:
                    trusted_dialogue = False
            if not trusted_dialogue:
                raise CommunityClientError(
                    "local lesson failed exact arithmetic or trusted human-dialogue verification"
                )
        key = _lesson_key(lesson)
        current = unique.get(key)
        if current is None or int(getattr(lesson, "id", 0)) < int(
            getattr(current, "id", 0)
        ):
            unique[key] = lesson
    required = minimum_training_lessons + minimum_validation_lessons
    if len(unique) < required:
        raise CommunityClientError(
            f"Need at least {required} distinct verified lessons; found {len(unique)}"
        )
    ordered = sorted(
        unique.items(),
        key=lambda item: hashlib.sha256(
            f"{foundation_sha256}:{item[0]}".encode("ascii")
        ).digest(),
    )
    validation_count = max(
        minimum_validation_lessons,
        math.ceil(len(ordered) * validation_fraction),
    )
    validation_count = min(validation_count, len(ordered) - minimum_training_lessons)
    validation = [record for _, record in ordered[:validation_count]]
    training = [record for _, record in ordered[validation_count:]]
    return training, validation


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def initialize_community(
    *,
    foundation_manifest: str | Path,
    trusted_public_key: str | Path,
    output_dir: str | Path,
    community_version: str,
    training_config: FloatyTrainingConfig | None = None,
) -> CommunityBootstrapResult:
    """Create immutable no-op floaty weights and their first strict contract."""

    settings = training_config or FloatyTrainingConfig()
    settings.validate()
    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with load_signed_foundation(foundation_manifest, trusted_public_key) as loaded:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
        )
        try:
            weights = _qualified_floaty_state(loaded.model)
            down_tensors = [
                value
                for name, value in weights.items()
                if name.endswith(".down.weight")
            ]
            if not down_tensors or any(np.any(value != 0.0) for value in down_tensors):
                raise CommunityClientError("initial floaty layer is not an exact no-op")
            weights_path = write_floaty_safetensors(
                temporary / BASE_FLOATY_FILENAME,
                weights,
                tensor_specs_from_floaty_schema(loaded.floaty_schema["tensors"]),
            )
            contract = CommunityContract.from_floaty_schema(
                loaded.floaty_schema["tensors"],
                foundation_sha256=loaded.model.foundation_hash(),
                community_version=community_version,
                community_weights_sha256=sha256_file(weights_path),
                architecture_sha256=loaded.model.config_hash(),
                tokenizer_sha256=loaded.tokenizer.metadata["tokenizer_sha256"],
                training_recipe_sha256=floaty_training_recipe_sha256(settings),
            )
            _write_new_json(temporary / CONTRACT_FILENAME, contract.to_dict())
            _write_new_json(
                temporary / BOOTSTRAP_FILENAME,
                {
                    "schema_version": COMMUNITY_CLIENT_SCHEMA_VERSION,
                    "foundation_id": loaded.verified.foundation_id,
                    "foundation_manifest_sha256": loaded.verified.manifest_sha256,
                    "owner_key_id": loaded.verified.owner_key_id,
                    "community_version": community_version,
                    "community_weights_sha256": contract.community_weights_sha256,
                    "training_recipe_sha256": contract.training_recipe_sha256,
                    "promotion": "not-performed",
                    "signature": "not-created",
                },
            )
            _fsync_directory(temporary)
            publish_directory_no_replace(temporary, destination)
            _fsync_directory(destination.parent)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise
    return CommunityBootstrapResult(
        output_dir=destination,
        weights_path=destination / BASE_FLOATY_FILENAME,
        contract_path=destination / CONTRACT_FILENAME,
        contract=contract,
        foundation_id=loaded.verified.foundation_id,
    )


def train_and_package_update(
    *,
    foundation_manifest: str | Path,
    trusted_public_key: str | Path,
    community_release_manifest: str | Path,
    trust_state_path: str | Path,
    lessons_db: str | Path,
    candidate_output_dir: str | Path,
    bundle_path: str | Path,
    training_config: FloatyTrainingConfig | None = None,
    lesson_limit: int = 128,
    lesson_ids: Sequence[int] | None = None,
    claimed_training_run_id: int | None = None,
    validation_fraction: float = 0.2,
    update_id: str | None = None,
) -> CommunityContributionResult:
    """Train a guarded local adapter delta and package an immutable bundle."""

    settings = training_config or FloatyTrainingConfig()
    settings.validate()
    if (
        isinstance(lesson_limit, bool)
        or not isinstance(lesson_limit, int)
        or lesson_limit <= 0
    ):
        raise ValueError("lesson_limit must be positive")
    selected_lesson_ids: tuple[int, ...] | None = None
    if lesson_ids is not None:
        if claimed_training_run_id is None:
            raise ValueError("claimed_training_run_id is required with lesson_ids")
        if (
            isinstance(claimed_training_run_id, bool)
            or not isinstance(claimed_training_run_id, int)
            or claimed_training_run_id <= 0
        ):
            raise ValueError("claimed_training_run_id must be positive")
        if any(
            type(identifier) is not int or identifier <= 0 for identifier in lesson_ids
        ):
            raise ValueError("lesson_ids must contain distinct positive integers")
        selected_lesson_ids = tuple(lesson_ids)
        if len(set(selected_lesson_ids)) != len(selected_lesson_ids):
            raise ValueError("lesson_ids must contain distinct positive integers")
    elif claimed_training_run_id is not None:
        raise ValueError("claimed_training_run_id requires explicit lesson_ids")
    with load_signed_community_release(
        community_release_manifest,
        trusted_public_key,
        foundation_manifest=foundation_manifest,
    ) as active:
        with load_signed_foundation(foundation_manifest, trusted_public_key) as loaded:
            if (
                active.verified.foundation_manifest_sha256
                != loaded.verified.manifest_sha256
                or active.verified.owner_key_id != loaded.verified.owner_key_id
            ):
                raise CommunityClientError(
                    "community release does not match the pinned signed foundation"
                )
            contract = active.contract
            _validate_contract(loaded, contract, settings)
            base = dict(active.weights)
            _import_qualified_floaty_state(loaded.model, base)
            # Remember a fully loaded and cross-checked release before training.
            # A later local failure may be retried, but must never permit rollback.
            advance_trusted_community_state(trust_state_path, active.verified)
            store = ApprenticeshipStore(lessons_db)
            if selected_lesson_ids is None:
                local_lessons = store.list_recent_lessons(
                    limit=lesson_limit,
                    pending_only=True,
                )
            else:
                local_lessons = []
                for identifier in selected_lesson_ids:
                    lesson = store.get_lesson(identifier)
                    if (
                        lesson is None
                        or lesson.status != "training"
                        or lesson.training_run_id != claimed_training_run_id
                    ):
                        raise CommunityClientError(
                            "explicit lesson is not claimed by the requested training run"
                        )
                    local_lessons.append(lesson)
            training, validation = deterministic_lesson_split(
                local_lessons,
                foundation_sha256=contract.foundation_sha256,
                validation_fraction=validation_fraction,
                minimum_training_lessons=settings.minimum_training_lessons,
                minimum_validation_lessons=settings.minimum_validation_lessons,
                trusted_dialogue_validator=store.is_trusted_dialogue_lesson,
            )
            candidate = run_floaty_candidate(
                loaded.model,
                training,
                validation,
                loaded.tokenizer,
                candidate_output_dir,
                expected_foundation_hash=contract.foundation_sha256,
                expected_config_hash=contract.architecture_sha256,
                expected_tokenizer_sha256=contract.tokenizer_sha256,
                expected_floaty_parameter_names=tuple(
                    spec.name for spec in contract.tensor_specs
                ),
                config=settings,
            )
            training_ids = tuple(
                identifier
                for lesson in training
                if (identifier := getattr(lesson, "id", None)) is not None
            )
            validation_ids = tuple(
                identifier
                for lesson in validation
                if (identifier := getattr(lesson, "id", None)) is not None
            )
            if not candidate.accepted:
                return CommunityContributionResult(
                    accepted=False,
                    candidate=candidate,
                    bundle_path=None,
                    update_manifest=None,
                    training_lesson_ids=training_ids,
                    validation_lesson_ids=validation_ids,
                )
            current = _qualified_floaty_state(loaded.model)
            deltas = {
                name: np.asarray(current[name], dtype=np.float32)
                - np.asarray(base[name], dtype=np.float32)
                for name in current
            }
            base_state = dict(base)
            try:
                manifest = write_update_bundle(
                    bundle_path,
                    contract=contract,
                    update_id=update_id or f"{_SAFE_UPDATE_PREFIX}-{uuid.uuid4().hex}",
                    deltas=deltas,
                    example_count=len(training),
                )
            except Exception:
                _import_qualified_floaty_state(loaded.model, base_state)
                raise
            return CommunityContributionResult(
                accepted=True,
                candidate=candidate,
                bundle_path=Path(bundle_path).expanduser().resolve(),
                update_manifest=manifest,
                training_lesson_ids=training_ids,
                validation_lesson_ids=validation_ids,
            )


def load_training_config(path: str | Path | None) -> FloatyTrainingConfig:
    if path is None:
        return FloatyTrainingConfig()
    value = _read_strict_json(
        Path(path).expanduser().resolve(), "floaty training config"
    )
    allowed = {item.name for item in fields(FloatyTrainingConfig)}
    if set(value) != allowed:
        raise CommunityClientError("training config fields are incompatible")
    try:
        return FloatyTrainingConfig(**value)
    except (TypeError, ValueError) as error:
        raise CommunityClientError("training config is invalid") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlm-community-client",
        description="Initialize or locally train a signed Hybrid foundation's floaty layer.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    recipe = commands.add_parser(
        "recipe-hash", help="print the exact local recipe SHA-256"
    )
    recipe.add_argument("--training-config")
    initialize = commands.add_parser(
        "init",
        help="owner bootstrap: create unsigned no-op community version zero",
    )
    initialize.add_argument("--foundation-manifest", required=True)
    initialize.add_argument("--owner-public-key", required=True)
    initialize.add_argument("--output", required=True)
    initialize.add_argument("--community-version", default="community-0")
    initialize.add_argument("--training-config")
    contribute = commands.add_parser(
        "train-and-package",
        help="train verified local lessons and write one immutable delta bundle",
    )
    contribute.add_argument("--foundation-manifest", required=True)
    contribute.add_argument("--owner-public-key", required=True)
    contribute.add_argument("--community-release", required=True)
    contribute.add_argument("--trust-state", required=True)
    contribute.add_argument("--lessons-db", required=True)
    contribute.add_argument("--candidate-output", required=True)
    contribute.add_argument("--bundle", required=True)
    contribute.add_argument("--training-config")
    contribute.add_argument("--lesson-limit", type=int, default=128)
    contribute.add_argument("--lesson-id", type=int, action="append")
    contribute.add_argument("--claimed-training-run-id", type=int)
    contribute.add_argument("--validation-fraction", type=float, default=0.2)
    contribute.add_argument("--update-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        settings = load_training_config(arguments.training_config)
        if arguments.command == "recipe-hash":
            output: Mapping[str, Any] = {
                "training_recipe_sha256": floaty_training_recipe_sha256(settings)
            }
        elif arguments.command == "init":
            result = initialize_community(
                foundation_manifest=arguments.foundation_manifest,
                trusted_public_key=arguments.owner_public_key,
                output_dir=arguments.output,
                community_version=arguments.community_version,
                training_config=settings,
            )
            output = {
                "foundation_id": result.foundation_id,
                "weights": str(result.weights_path),
                "contract": str(result.contract_path),
                "community_version": result.contract.community_version,
                "promotion": "not-performed",
            }
        else:
            contribution = train_and_package_update(
                foundation_manifest=arguments.foundation_manifest,
                trusted_public_key=arguments.owner_public_key,
                community_release_manifest=arguments.community_release,
                trust_state_path=arguments.trust_state,
                lessons_db=arguments.lessons_db,
                candidate_output_dir=arguments.candidate_output,
                bundle_path=arguments.bundle,
                training_config=settings,
                lesson_limit=arguments.lesson_limit,
                lesson_ids=arguments.lesson_id,
                claimed_training_run_id=arguments.claimed_training_run_id,
                validation_fraction=arguments.validation_fraction,
                update_id=arguments.update_id,
            )
            output = {
                "accepted": contribution.accepted,
                "candidate_report": str(contribution.candidate.report_path),
                "bundle": (
                    str(contribution.bundle_path)
                    if contribution.bundle_path is not None
                    else None
                ),
                "update_id": (
                    contribution.update_manifest.update_id
                    if contribution.update_manifest is not None
                    else None
                ),
                "update_manifest_sha256": (
                    contribution.update_manifest.manifest_sha256
                    if contribution.update_manifest is not None
                    else None
                ),
                "training_lesson_ids": list(contribution.training_lesson_ids),
                "validation_lesson_ids": list(contribution.validation_lesson_ids),
            }
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
