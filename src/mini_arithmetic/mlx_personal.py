"""Private conversational adapters for the MLX Hybrid language model.

This module is the MLX implementation of :mod:`mini_arithmetic.personal_training`.
It consumes only a checksum-verified, scratch-derived :class:`PortableBundle`,
keeps the Hybrid foundation frozen, and trains the small parallel low-rank
``FoundationFloatyLM`` surface.  The resulting safetensors file is cumulative:
when a parent adapter is active, every one of its tensors is loaded before the
next local optimization run.

The runtime keeps one composed model resident and uses Hybrid KV/convolution
caches for generation.  Adapter preparation never mutates the live model;
activation swaps a fully validated tensor state while holding the inference
lock and restores the prior state if anything fails.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from safetensors import safe_open

from .floaty_training import (
    FoundationFloatyTrainingBoundary,
    MaskedLessonBatch,
    make_verified_lesson_batches,
)
from .foundation_floaty import FloatyConfig, FloatyState, FoundationFloatyLM
from .hybrid import HybridLM
from .personal_training import (
    PERSONAL_HOLDOUT_EXAMPLES,
    PERSONAL_MAX_REPLAY_EXAMPLES,
    PERSONAL_MAX_RETENTION_EXAMPLES,
    PERSONAL_MAX_TRAINING_CONTEXT_TOKENS,
    PERSONAL_TRAINING_EXAMPLES,
    DialogueExample,
    PersonalAdapterArtifact,
    PersonalAdapterCandidate,
    PersonalAdapterTrainingRequest,
    PersonalRuntimeIdentity,
    PersonalTrainingError,
    compact_personal_dialogue_prompt,
    personal_dialogue_target_fits,
)
from .personal_foundation import (
    PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES,
    PersonalFoundationArtifact,
    PersonalFoundationCandidate,
    PersonalFoundationIdentity,
    PersonalFoundationSource,
    PersonalFoundationTrainingRequest,
    personal_foundation_backend_contract_sha256,
    personal_foundation_bundle_sha256,
    validate_foundation_metrics,
    validate_foundation_request,
)
from .portable_bundle import (
    PORTABLE_STATE_FORMAT,
    PortableBundle,
    canonical_tensor_schema,
    load_portable_bundle,
    publish_conversational_foundation_bundle,
)
from .scratch_tokenizer import ROLE_NAMES, sanitize_generated_text


MLX_PERSONAL_ADAPTER_FORMAT = "tinydog-mlx-personal-adapter-v1"
MLX_PERSONAL_BACKEND = "mlx"
_ARTIFACT_FORMAT = "safetensors"
_MAX_ADAPTER_TENSORS = 16_384
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


class MLXPersonalError(PersonalTrainingError):
    """An MLX personal adapter failed a local compatibility boundary."""


@dataclass(frozen=True)
class MLXPersonalConfig:
    """Small deterministic optimizer and adapter configuration.

    Twelve single-example updates are intentionally modest.  They make the
    automatic five-turn loop usable on ordinary Apple Silicon while the
    independent fifth lesson still decides whether the candidate is promoted.
    """

    rank: int = 8
    alpha: float = 16.0
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    steps: int = 12
    grad_clip: float = 1.0
    seed: int = 17
    batch_size: int = 1
    context_length: int = 256
    max_tokens: int = 32

    def validate(self) -> None:
        integers = {
            "rank": self.rank,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "max_tokens": self.max_tokens,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.context_length < 8:
            raise ValueError("context_length must be at least eight")
        if self.context_length > PERSONAL_MAX_TRAINING_CONTEXT_TOKENS:
            raise ValueError(
                "context_length exceeds the shared personal-training limit"
            )
        finite = {
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
        }
        for name, value in finite.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if self.alpha <= 0 or self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("alpha, learning_rate, and grad_clip must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")

    def floaty_config(self) -> FloatyConfig:
        self.validate()
        return FloatyConfig(rank=self.rank, alpha=self.alpha)


@dataclass(frozen=True)
class MLXFoundationConfig:
    """Bounded optimizer settings for direct conversational foundation updates.

    The portable default is the bounded ``tiny`` model, while an explicitly
    selected standard bundle uses the same complete-weight recipe. Five new
    human targets contribute once per run; bounded replay may add more updates.
    """

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    steps: int = PERSONAL_TRAINING_EXAMPLES + PERSONAL_HOLDOUT_EXAMPLES
    grad_clip: float = 1.0
    seed: int = 29
    batch_size: int = 1
    context_length: int = 128
    maximum_replay_examples: int = PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES
    disk_reserve_bytes: int = 64 * 1024 * 1024

    def validate(self) -> None:
        for name, value in {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.steps != PERSONAL_TRAINING_EXAMPLES + PERSONAL_HOLDOUT_EXAMPLES:
            raise ValueError(
                "direct foundation training requires exactly five new updates"
            )
        if self.batch_size != 1:
            raise ValueError("direct foundation training batch_size must remain one")
        if (
            isinstance(self.maximum_replay_examples, bool)
            or not isinstance(self.maximum_replay_examples, int)
            or self.maximum_replay_examples < 0
        ):
            raise ValueError("maximum_replay_examples must be a nonnegative integer")
        if self.maximum_replay_examples > PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES:
            raise ValueError("maximum_replay_examples exceeds the shared hard bound")
        if (
            isinstance(self.disk_reserve_bytes, bool)
            or not isinstance(self.disk_reserve_bytes, int)
            or self.disk_reserve_bytes < 0
        ):
            raise ValueError("disk_reserve_bytes must be a nonnegative integer")
        if self.context_length < 8:
            raise ValueError("context_length must be at least eight")
        if self.context_length > PERSONAL_MAX_TRAINING_CONTEXT_TOKENS:
            raise ValueError(
                "context_length exceeds the shared personal-training limit"
            )
        for name, value in {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")


def _default_mlx_foundation_config(bundle: PortableBundle) -> MLXFoundationConfig:
    """Fit the zero-config recipe to a checksum-verified parent context."""

    default = MLXFoundationConfig()
    return MLXFoundationConfig(
        learning_rate=default.learning_rate,
        weight_decay=default.weight_decay,
        steps=default.steps,
        grad_clip=default.grad_clip,
        seed=default.seed,
        batch_size=default.batch_size,
        context_length=min(default.context_length, bundle.config.context_length),
        maximum_replay_examples=default.maximum_replay_examples,
        disk_reserve_bytes=default.disk_reserve_bytes,
    )


def _default_mlx_foundation_runtime_config(
    bundle: PortableBundle,
    *,
    context_length: int | None = None,
) -> MLXPersonalConfig:
    """Create the no-adapter runtime shell without exceeding a small bundle."""

    default = MLXPersonalConfig()
    return MLXPersonalConfig(
        rank=min(default.rank, bundle.config.model_dim - 1),
        alpha=default.alpha,
        learning_rate=default.learning_rate,
        weight_decay=default.weight_decay,
        steps=default.steps,
        grad_clip=default.grad_clip,
        seed=default.seed,
        batch_size=default.batch_size,
        context_length=min(
            default.context_length,
            bundle.config.context_length,
            context_length
            if context_length is not None
            else bundle.config.context_length,
        ),
        max_tokens=default.max_tokens,
    )


@dataclass(frozen=True)
class _LoadedMLXModel:
    bundle: PortableBundle
    model: FoundationFloatyLM
    internal_foundation_sha256: str
    adapter_schema_sha256: str


@dataclass(frozen=True)
class PreparedMLXPersonalAdapter:
    """Fully checked tensor state that is safe to swap into the live runtime."""

    adapter_id: str
    artifact_sha256: str
    tensors: tuple[tuple[str, mx.array], ...]
    runtime_token: object


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_parameters(model: FoundationFloatyLM) -> dict[str, mx.array]:
    return {
        name: value
        for name, value in tree_flatten(model.floaty.parameters())
        if isinstance(value, mx.array)
    }


def _revalidate_bundle(bundle: PortableBundle) -> PortableBundle:
    if type(bundle) is not PortableBundle:
        raise TypeError("bundle must be an exact PortableBundle")
    verified = load_portable_bundle(bundle.root)
    if verified.root != bundle.root or verified.manifest[
        "weights_sha256"
    ] != bundle.manifest.get("weights_sha256"):
        raise MLXPersonalError("portable foundation identity changed")
    if (
        verified.manifest.get("scratch_only") is not True
        or verified.manifest.get("external_weights") is not False
    ):
        raise MLXPersonalError("MLX personal learning requires a scratch-only bundle")
    return verified


def _validate_foundation_tensors(
    foundation: HybridLM,
    weights_path: Path,
    parameter_dtype: str,
) -> None:
    expected = dict(tree_flatten(foundation.parameters()))
    try:
        with safe_open(str(weights_path), framework="numpy") as header:
            if header.metadata() != {"format": PORTABLE_STATE_FORMAT}:
                raise MLXPersonalError(
                    "portable foundation safetensors metadata is incompatible"
                )
            names = tuple(header.keys())
        if len(names) > _MAX_ADAPTER_TENSORS or set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise MLXPersonalError(
                "portable foundation tensor names differ "
                f"(missing={missing[:4]}, extra={extra[:4]})"
            )
        loaded = mx.load(str(weights_path))
        if not isinstance(loaded, dict) or set(loaded) != set(expected):
            raise MLXPersonalError("portable foundation tensor mapping is invalid")
        expected_dtype = mx.bfloat16 if parameter_dtype == "bfloat16" else mx.float32
        checks: list[mx.array] = []
        for name in names:
            value = loaded[name]
            template = expected[name]
            if value.shape != template.shape:
                raise MLXPersonalError(
                    f"portable foundation tensor {name!r} has the wrong shape"
                )
            if value.dtype != expected_dtype:
                raise MLXPersonalError(
                    f"portable foundation tensor {name!r} has the wrong dtype"
                )
            checks.append(mx.all(mx.isfinite(value)))
        mx.eval(checks)
        if not all(bool(check.item()) for check in checks):
            raise MLXPersonalError("portable foundation contains non-finite tensors")
    except MLXPersonalError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise MLXPersonalError("portable foundation safetensors are invalid") from error


def _load_composed_model(
    bundle: PortableBundle,
    config: MLXPersonalConfig,
    *,
    initialization_seed: int | None = None,
) -> _LoadedMLXModel:
    verified = _revalidate_bundle(bundle)
    config.validate()
    if config.context_length > verified.config.context_length:
        raise MLXPersonalError(
            "personal training context exceeds the portable foundation context"
        )
    if config.rank >= verified.config.model_dim:
        raise MLXPersonalError("personal adapter rank must be smaller than model width")
    mx.random.seed(config.seed if initialization_seed is None else initialization_seed)
    foundation = HybridLM(verified.config)
    _validate_foundation_tensors(
        foundation,
        verified.weights_path,
        verified.parameter_dtype,
    )
    try:
        foundation.load_weights(str(verified.weights_path), strict=True)
        mx.eval(foundation.parameters())
    except (OSError, RuntimeError, ValueError) as error:
        raise MLXPersonalError(
            "portable foundation could not be loaded strictly"
        ) from error
    model = FoundationFloatyLM(foundation, config.floaty_config())
    # A no-op adapter must not promote a BF16 foundation's hidden states to
    # FP32.  Matching the portable dtype preserves exact foundation-only
    # inference and keeps consumer-device activation memory bounded.
    if verified.parameter_dtype == "bfloat16":
        model.floaty.set_dtype(mx.bfloat16)
    model.freeze_foundation()
    model.eval()
    mx.eval(model.parameters())
    internal_foundation = model.foundation_hash()
    adapter_schema = model.config_hash()
    return _LoadedMLXModel(
        bundle=verified,
        model=model,
        internal_foundation_sha256=internal_foundation,
        adapter_schema_sha256=adapter_schema,
    )


def _mlx_foundation_forward_smoke(loaded: _LoadedMLXModel) -> None:
    """Force one finite consumer forward before a bundle can be activated."""

    loaded.model.eval()
    logits = loaded.model(mx.array([[loaded.bundle.tokenizer.bos_id]], dtype=mx.int32))
    finite = mx.all(mx.isfinite(logits.astype(mx.float32)))
    mx.eval(logits, finite)
    if tuple(logits.shape) != (1, 1, loaded.bundle.config.vocab_size) or not bool(
        finite.item()
    ):
        raise MLXPersonalError("foundation candidate failed forward validation")


def _metadata_for_candidate(
    request: PersonalAdapterTrainingRequest,
) -> dict[str, str]:
    return {
        "format": MLX_PERSONAL_ADAPTER_FORMAT,
        "backend": MLX_PERSONAL_BACKEND,
        "tensor_scope": "floaty-only",
        "foundation_sha256": request.foundation_sha256,
        "adapter_schema_sha256": request.adapter_schema_sha256,
        "adapter_id": request.adapter_id,
        "parent_adapter_id": request.parent_adapter_id or "",
    }


def _load_adapter_tensors(
    path: Path,
    *,
    model: FoundationFloatyLM,
    foundation_sha256: str,
    adapter_schema_sha256: str,
    adapter_id: str | None,
) -> dict[str, mx.array]:
    if path.is_symlink() or not path.is_file():
        raise MLXPersonalError("personal adapter must be a regular local file")
    expected = _adapter_parameters(model)
    try:
        with safe_open(str(path), framework="numpy") as header:
            metadata = header.metadata()
            if not isinstance(metadata, dict):
                raise MLXPersonalError("personal adapter metadata is missing")
            if set(metadata) != {
                "format",
                "backend",
                "tensor_scope",
                "foundation_sha256",
                "adapter_schema_sha256",
                "adapter_id",
                "parent_adapter_id",
            }:
                raise MLXPersonalError("personal adapter metadata fields differ")
            required = {
                "format": MLX_PERSONAL_ADAPTER_FORMAT,
                "backend": MLX_PERSONAL_BACKEND,
                "tensor_scope": "floaty-only",
                "foundation_sha256": foundation_sha256,
                "adapter_schema_sha256": adapter_schema_sha256,
            }
            if any(metadata.get(name) != value for name, value in required.items()):
                raise MLXPersonalError("personal adapter metadata is incompatible")
            if adapter_id is not None and metadata.get("adapter_id") != adapter_id:
                raise MLXPersonalError("personal adapter identity metadata differs")
            names = tuple(header.keys())
            if len(names) > _MAX_ADAPTER_TENSORS or set(names) != set(expected):
                raise MLXPersonalError("personal adapter tensor names differ")
        loaded = mx.load(str(path))
        if not isinstance(loaded, dict) or set(loaded) != set(expected):
            raise MLXPersonalError("personal adapter tensor mapping is invalid")
        result: dict[str, mx.array] = {}
        finite: list[mx.array] = []
        for name in sorted(names):
            value = loaded[name]
            template = expected[name]
            if value.shape != template.shape or value.dtype != template.dtype:
                raise MLXPersonalError(
                    f"personal adapter tensor {name!r} has an incompatible schema"
                )
            result[name] = mx.array(value)
            finite.append(mx.all(mx.isfinite(value)))
        mx.eval(result, finite)
        if not all(bool(check.item()) for check in finite):
            raise MLXPersonalError("personal adapter contains non-finite tensors")
        return result
    except MLXPersonalError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise MLXPersonalError("personal adapter safetensors are invalid") from error


def _floaty_state(
    loaded: _LoadedMLXModel,
    tensors: Mapping[str, mx.array],
) -> FloatyState:
    return FloatyState(
        format_version=1,
        foundation_hash=loaded.internal_foundation_sha256,
        config_hash=loaded.adapter_schema_sha256,
        tensors=dict(tensors),
    )


def _dialogue_record(example: DialogueExample) -> dict[str, Any]:
    if type(example) is not DialogueExample:
        raise MLXPersonalError("personal trainer received an invalid dialogue example")
    prompt = example.prompt.replace("\r\n", "\n").replace("\r", "\n").strip()
    response = example.human_response.replace("\r\n", "\n").replace("\r", "\n").strip()
    digest = hashlib.sha256(
        f"typed-dialogue-v1\0{prompt}\0{response}".encode("utf-8")
    ).hexdigest()
    if (
        type(example.lesson_id) is not int
        or example.lesson_id <= 0
        or not isinstance(example.session_id, str)
        or not example.session_id
        or not prompt
        or not response
        or digest != example.content_sha256
    ):
        raise MLXPersonalError(
            "personal dialogue content does not match its provenance"
        )
    return {
        "id": example.lesson_id,
        "problem": prompt,
        "correct_answer": response,
        "source": "human-conversation",
        "verifier": "trusted-human-dialogue-v1",
        "metadata": {
            "human_authored_target": True,
            "consent_local_training": True,
        },
    }


def _masked_loss(
    model: FoundationFloatyLM,
    inputs: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
    inactive_logit_bias: mx.array | None = None,
) -> mx.array:
    logits = model(inputs).astype(mx.float32)
    if inactive_logit_bias is not None:
        logits = logits + inactive_logit_bias
    token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    mask = loss_mask.astype(mx.float32)
    denominator = mx.sum(mask)
    return mx.sum(token_losses.astype(mx.float32) * mask) / denominator


def _mean_loss(
    model: FoundationFloatyLM,
    batches: Sequence[MaskedLessonBatch],
    inactive_logit_bias: mx.array | None = None,
) -> float:
    model.eval()
    values: list[float] = []
    for batch in batches:
        packed = mx.array(batch.tokens)
        loss = _masked_loss(
            model,
            packed[:, :-1],
            packed[:, 1:],
            mx.array(batch.loss_mask),
            inactive_logit_bias,
        )
        mx.eval(loss)
        values.append(float(loss.item()))
    return sum(values) / len(values)


def _inactive_text_logit_bias(
    vocab_size: int,
    inactive_ids: Sequence[int],
) -> mx.array:
    if (
        type(vocab_size) is not int
        or vocab_size <= 0
        or not inactive_ids
        or min(inactive_ids) < 0
        or max(inactive_ids) >= vocab_size
        or len(set(inactive_ids)) != len(inactive_ids)
    ):
        raise MLXPersonalError("inactive text-token mask is incompatible")
    values = np.zeros((vocab_size,), dtype=np.float32)
    values[np.asarray(inactive_ids, dtype=np.int64)] = -np.inf
    return mx.array(values)


def _assert_batches_avoid_inactive_targets(
    batches: Sequence[MaskedLessonBatch],
    inactive_ids: Sequence[int],
) -> None:
    inactive = np.asarray(inactive_ids, dtype=np.int64)
    for batch in batches:
        targets = batch.tokens[:, 1:][batch.loss_mask.astype(bool)]
        if np.isin(targets, inactive).any():
            raise MLXPersonalError("personal target uses an inactive text-token ID")


def _tree_finite(tree: Any) -> bool:
    checks = [
        mx.all(mx.isfinite(value))
        for _, value in tree_flatten(tree)
        if isinstance(value, mx.array)
    ]
    mx.eval(checks)
    return all(bool(check.item()) for check in checks)


def _raise_if_cancelled(request: PersonalAdapterTrainingRequest) -> None:
    if request.cancellation_event is not None and request.cancellation_event.is_set():
        raise MLXPersonalError("personal training was cancelled during shutdown")


@dataclass(frozen=True)
class _MLXFoundationTrainingResult:
    tensors: tuple[tuple[str, mx.array], ...]
    baseline_monitoring_loss: float
    candidate_monitoring_loss: float
    metrics: dict[str, Any]
    source_sha256: dict[str, str]
    training_config_sha256: str


@dataclass(frozen=True)
class PreparedMLXPersonalFoundation:
    """Fully loaded immutable bundle awaiting one atomic runtime swap."""

    identity: PersonalFoundationIdentity
    bundle_path: Path
    loaded: _LoadedMLXModel
    runtime_token: object


def _personal_foundation_identity(
    bundle: PortableBundle,
    foundation_id: str,
) -> PersonalFoundationIdentity:
    verified = _revalidate_bundle(bundle)
    identity = PersonalFoundationIdentity(
        backend=MLX_PERSONAL_BACKEND,
        foundation_id=foundation_id,
        bundle_sha256=personal_foundation_bundle_sha256(verified.root),
        weights_sha256=str(verified.manifest["weights_sha256"]),
        config_sha256=str(verified.manifest["config_sha256"]),
        tokenizer_sha256=str(verified.manifest["tokenizer_sha256"]),
        backend_contract_sha256=personal_foundation_backend_contract_sha256(
            MLX_PERSONAL_BACKEND
        ),
    )
    identity.validate()
    return identity


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _foundation_source_digest(examples: Sequence[DialogueExample]) -> str:
    digest = hashlib.sha256()
    digest.update(b"tinydog-direct-foundation-dialogue-v1\0")
    for example in examples:
        _dialogue_record(example)
        digest.update(str(example.lesson_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(example.content_sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _foundation_config_digest(config: MLXFoundationConfig) -> str:
    payload = {
        "backend": MLX_PERSONAL_BACKEND,
        "format": "tinydog-direct-conversation-foundation-v1",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "steps": config.steps,
        "grad_clip": config.grad_clip,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "context_length": config.context_length,
        "maximum_replay_examples": config.maximum_replay_examples,
        "disk_reserve_bytes": config.disk_reserve_bytes,
        "assistant_only_targets": True,
        "new_examples_are_all_trainable": True,
        "inactive_text_logits_masked": True,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_trainable_foundation(
    bundle: PortableBundle,
    config: MLXFoundationConfig,
    *,
    run_id: int,
) -> tuple[PortableBundle, HybridLM]:
    verified = _revalidate_bundle(bundle)
    config.validate()
    if config.context_length > verified.config.context_length:
        raise MLXPersonalError(
            "foundation training context exceeds the portable model context"
        )
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise MLXPersonalError("foundation training run ID is invalid")
    mx.random.seed(config.seed + run_id)
    model = HybridLM(verified.config)
    _validate_foundation_tensors(
        model,
        verified.weights_path,
        verified.parameter_dtype,
    )
    try:
        model.load_weights(str(verified.weights_path), strict=True)
        model.unfreeze()
        model.train()
        mx.eval(model.parameters())
    except (OSError, RuntimeError, ValueError) as error:
        raise MLXPersonalError(
            "portable foundation could not be loaded for direct training"
        ) from error
    expected = set(canonical_tensor_schema(verified.config))
    actual = {name for name, _ in tree_flatten(model.trainable_parameters())}
    if actual != expected:
        raise MLXPersonalError(
            "direct trainer did not unfreeze every foundation tensor"
        )
    return verified, model


def _mlx_foundation_disk_preflight(
    work_dir: Path,
    parent: PortableBundle,
    config: MLXFoundationConfig,
) -> None:
    required = parent.weights_path.stat().st_size * 2 + config.disk_reserve_bytes
    try:
        free = shutil.disk_usage(work_dir).free
    except OSError as error:
        raise MLXPersonalError(
            "could not verify disk space for foundation candidate"
        ) from error
    if free < required:
        raise MLXPersonalError(
            f"direct foundation training needs at least {required} free bytes "
            "for an atomic candidate"
        )


def _clear_mlx_device_cache() -> None:
    try:
        mx.clear_cache()
    except (AttributeError, RuntimeError):
        pass


def _foundation_loss(
    model: HybridLM,
    inputs: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
    inactive_logit_bias: mx.array,
) -> mx.array:
    logits = model(inputs).astype(mx.float32) + inactive_logit_bias
    token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    mask = loss_mask.astype(mx.float32)
    return mx.sum(token_losses.astype(mx.float32) * mask) / mx.sum(mask)


def _mean_foundation_loss(
    model: HybridLM,
    batches: Sequence[MaskedLessonBatch],
    inactive_logit_bias: mx.array,
) -> float:
    model.eval()
    values: list[float] = []
    for batch in batches:
        packed = mx.array(batch.tokens)
        loss = _foundation_loss(
            model,
            packed[:, :-1],
            packed[:, 1:],
            mx.array(batch.loss_mask),
            inactive_logit_bias,
        )
        mx.eval(loss)
        values.append(float(loss.item()))
    return sum(values) / len(values)


def _train_mlx_foundation(
    bundle: PortableBundle,
    *,
    run_id: int,
    new_examples: tuple[DialogueExample, ...],
    monitoring_examples: tuple[DialogueExample, ...],
    replay_examples: tuple[DialogueExample, ...] = (),
    config: MLXFoundationConfig | None = None,
    cancellation_event: threading.Event | None = None,
) -> _MLXFoundationTrainingResult:
    """Train every MLX foundation tensor from one immutable portable parent.

    This helper deliberately returns tensors and measurements but performs no
    promotion. The in-sample monitor is evaluated exactly before and after the
    fixed update sequence, and its value never controls publication.
    """

    selected = config or _default_mlx_foundation_config(_revalidate_bundle(bundle))
    selected.validate()
    required_new = PERSONAL_TRAINING_EXAMPLES + PERSONAL_HOLDOUT_EXAMPLES
    if len(new_examples) != required_new:
        raise MLXPersonalError("direct foundation training requires five new examples")
    if len(monitoring_examples) != PERSONAL_HOLDOUT_EXAMPLES:
        raise MLXPersonalError(
            "direct foundation training requires one monitor example"
        )
    if len(replay_examples) > min(
        PERSONAL_MAX_REPLAY_EXAMPLES, selected.maximum_replay_examples
    ):
        raise MLXPersonalError("direct foundation replay exceeds its hard bound")
    all_examples = new_examples + replay_examples
    if len({example.lesson_id for example in all_examples}) != len(all_examples):
        raise MLXPersonalError("foundation dialogue lesson IDs must be distinct")
    if len({example.content_sha256 for example in all_examples}) != len(all_examples):
        raise MLXPersonalError("foundation dialogue content must be distinct")
    if any(
        example.content_sha256 not in {item.content_sha256 for item in new_examples}
        for example in monitoring_examples
    ):
        raise MLXPersonalError(
            "foundation monitor must be a subset of the five new examples"
        )
    if cancellation_event is not None and cancellation_event.is_set():
        raise MLXPersonalError("foundation training was cancelled during shutdown")

    verified, model = _load_trainable_foundation(bundle, selected, run_id=run_id)
    # Unlike the legacy 4+1 adapter gate, direct foundation consolidation
    # learns all five new human targets. One of those five is also measured
    # before and after as an in-sample monitor, but its score cannot gate
    # publication.
    training_records = [
        _dialogue_record(example) for example in new_examples + replay_examples
    ]
    validation_records = [_dialogue_record(example) for example in monitoring_examples]
    training_batches, _, _ = make_verified_lesson_batches(
        training_records,
        verified.tokenizer,
        batch_size=selected.batch_size,
        context_length=selected.context_length,
        steps=len(training_records),
        seed=selected.seed + run_id,
    )
    validation_batches, _, _ = make_verified_lesson_batches(
        validation_records,
        verified.tokenizer,
        batch_size=1,
        context_length=selected.context_length,
        steps=1,
        seed=selected.seed + run_id + 1,
    )
    inactive_ids = verified.tokenizer.inactive_text_token_ids
    _assert_batches_avoid_inactive_targets(
        (*training_batches, *validation_batches), inactive_ids
    )
    inactive_bias = _inactive_text_logit_bias(
        verified.config.vocab_size,
        inactive_ids,
    )
    baseline_monitoring = _mean_foundation_loss(
        model, validation_batches, inactive_bias
    )
    if not math.isfinite(baseline_monitoring):
        raise MLXPersonalError("foundation baseline evaluation is non-finite")

    expected_names = set(canonical_tensor_schema(verified.config))
    optimizer = optim.AdamW(
        learning_rate=selected.learning_rate,
        weight_decay=selected.weight_decay,
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)

    def loss_fn(
        current: nn.Module,
        inputs: mx.array,
        targets: mx.array,
        mask: mx.array,
    ) -> mx.array:
        if current is not model:
            raise MLXPersonalError("foundation optimizer received another model")
        return _foundation_loss(model, inputs, targets, mask, inactive_bias)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    training_losses: list[float] = []
    assistant_tokens = 0
    tokens_seen = 0
    for batch in training_batches:
        if cancellation_event is not None and cancellation_event.is_set():
            raise MLXPersonalError("foundation training was cancelled during shutdown")
        model.train()
        packed = mx.array(batch.tokens)
        loss, gradients = loss_and_grad(
            model,
            packed[:, :-1],
            packed[:, 1:],
            mx.array(batch.loss_mask),
        )
        gradient_names = {name for name, _ in tree_flatten(gradients)}
        if gradient_names != expected_names:
            raise MLXPersonalError("foundation gradients do not cover every tensor")
        clipped, gradient_norm = optim.clip_grad_norm(gradients, selected.grad_clip)
        mx.eval(loss, clipped, gradient_norm)
        loss_value = float(loss.item())
        norm_value = float(gradient_norm.item())
        if not math.isfinite(loss_value) or not math.isfinite(norm_value):
            raise MLXPersonalError("foundation training produced a non-finite gradient")
        optimizer.update(model, clipped)
        mx.eval(model.parameters(), optimizer.state)
        if not _tree_finite(model.parameters()) or not _tree_finite(optimizer.state):
            raise MLXPersonalError("foundation training produced a non-finite update")
        training_losses.append(loss_value)
        assistant_tokens += int(np.sum(batch.loss_mask))
        tokens_seen += int(batch.tokens.size)

    if cancellation_event is not None and cancellation_event.is_set():
        raise MLXPersonalError("foundation training was cancelled during shutdown")
    candidate_monitoring = _mean_foundation_loss(
        model, validation_batches, inactive_bias
    )
    if not math.isfinite(candidate_monitoring):
        raise MLXPersonalError("foundation candidate evaluation is non-finite")

    tensors = dict(tree_flatten(model.parameters()))
    if set(tensors) != expected_names:
        raise MLXPersonalError("foundation candidate tensor schema differs")
    mx.eval(tensors)
    source_sha256 = {
        "new_conversation": _foundation_source_digest(new_examples),
    }
    if replay_examples:
        source_sha256["bounded_replay"] = _foundation_source_digest(replay_examples)
    return _MLXFoundationTrainingResult(
        tensors=tuple(
            sorted((name, mx.array(value)) for name, value in tensors.items())
        ),
        baseline_monitoring_loss=baseline_monitoring,
        candidate_monitoring_loss=candidate_monitoring,
        metrics={
            "adamw": True,
            "updates": len(training_losses),
            "mean_training_loss": sum(training_losses) / len(training_losses),
            "final_training_loss": training_losses[-1],
            "new_training_examples": len(new_examples),
            "new_monitoring_examples": len(monitoring_examples),
            "monitoring_examples": len(monitoring_examples),
            "new_examples_optimized": len(new_examples),
            "replay_examples": len(replay_examples),
            "assistant_target_tokens": assistant_tokens,
            "tokens_seen": tokens_seen,
            "inactive_text_tokens_masked": len(inactive_ids),
            "monitoring_evaluations": 2,
            "monitor_used_for_selection": False,
            "monitoring_used_for_selection": False,
            "foundation_trainable": True,
            "full_foundation": True,
            "tensor_scope_full_foundation": True,
            "foundation_parameter_count": sum(value.size for value in tensors.values()),
            "gradient_tensor_count": len(expected_names),
            "all_foundation_parameters_optimizer_eligible": True,
            "optimizer_state_continuity": False,
            "optimizer_state_reset": True,
            "output_bfloat16": verified.parameter_dtype == "bfloat16",
        },
        source_sha256=source_sha256,
        training_config_sha256=_foundation_config_digest(selected),
    )


def _publish_mlx_foundation_bundle(
    parent_bundle: PortableBundle,
    result: _MLXFoundationTrainingResult,
    destination: str | Path,
) -> PortableBundle:
    """Atomically publish full trained tensors in the canonical portable format."""

    parent = _revalidate_bundle(parent_bundle)
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=".tinydog-mlx-weights-", dir=target.parent)
    )
    try:
        weights_path = staging / "candidate.safetensors"
        mx.save_safetensors(
            str(weights_path),
            dict(result.tensors),
            metadata={"format": PORTABLE_STATE_FORMAT},
        )
        with weights_path.open("rb") as stream:
            os.fsync(stream.fileno())
        return publish_conversational_foundation_bundle(
            parent,
            weights_path,
            target,
            backend=MLX_PERSONAL_BACKEND,
            training_config_sha256=result.training_config_sha256,
            source_sha256=result.source_sha256,
            steps=int(result.metrics["updates"]),
            tokens_seen=int(result.metrics["tokens_seen"]),
        )
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


class MLXPersonalAdapterTrainer:
    """Isolated MLX floaty trainer implementing ``PersonalAdapterTrainer``."""

    backend_name = MLX_PERSONAL_BACKEND

    def __init__(
        self,
        bundle: PortableBundle,
        *,
        config: MLXPersonalConfig | None = None,
    ) -> None:
        self.bundle = _revalidate_bundle(bundle)
        self.config = config or MLXPersonalConfig()
        self.config.validate()
        self._inactive_text_ids = self.bundle.tokenizer.inactive_text_token_ids

    @classmethod
    def from_bundle(
        cls,
        bundle: PortableBundle,
        *,
        config: MLXPersonalConfig | None = None,
    ) -> "MLXPersonalAdapterTrainer":
        return cls(bundle, config=config)

    def personal_training_target_fits(self, prompt: str, human_response: str) -> bool:
        return personal_dialogue_target_fits(
            self.bundle.tokenizer,
            prompt,
            human_response,
            context_length=self.config.context_length,
        )

    def train_adapter(
        self,
        request: PersonalAdapterTrainingRequest,
    ) -> PersonalAdapterCandidate:
        if type(request) is not PersonalAdapterTrainingRequest:
            raise TypeError("request must be a PersonalAdapterTrainingRequest")
        _raise_if_cancelled(request)
        if type(request.run_id) is not int or request.run_id <= 0:
            raise MLXPersonalError("personal training run ID is invalid")
        if not _IDENTIFIER_RE.fullmatch(request.adapter_id):
            raise MLXPersonalError("personal adapter ID is invalid")
        if request.parent_adapter_id is not None and not _IDENTIFIER_RE.fullmatch(
            request.parent_adapter_id
        ):
            raise MLXPersonalError("parent personal adapter ID is invalid")
        if request.backend != self.backend_name:
            raise MLXPersonalError("personal training request selected another backend")
        if (
            len(request.training_examples) != PERSONAL_TRAINING_EXAMPLES
            or len(request.validation_examples) != PERSONAL_HOLDOUT_EXAMPLES
        ):
            raise MLXPersonalError(
                "MLX personal training requires exactly four plus one"
            )
        if len(request.replay_examples) > PERSONAL_MAX_REPLAY_EXAMPLES:
            raise MLXPersonalError("MLX personal replay exceeds its hard bound")
        if len(request.retention_examples) > PERSONAL_MAX_RETENTION_EXAMPLES:
            raise MLXPersonalError("MLX personal retention set exceeds its hard bound")
        if (request.parent_adapter_id is None) != (request.parent_adapter_path is None):
            raise MLXPersonalError(
                "parent adapter identity and path must appear together"
            )
        if request.work_dir.is_symlink() or not request.work_dir.is_dir():
            raise MLXPersonalError("personal training work directory is unsafe")

        # The initial no-op up matrices and batch order are repeatable for a
        # given immutable ledger run.
        loaded = _load_composed_model(
            self.bundle,
            self.config,
            initialization_seed=self.config.seed + request.run_id,
        )
        model = loaded.model
        portable_foundation = str(loaded.bundle.manifest["weights_sha256"])
        if portable_foundation != request.foundation_sha256:
            raise MLXPersonalError(
                "personal request foundation is not this portable bundle"
            )
        if loaded.adapter_schema_sha256 != request.adapter_schema_sha256:
            raise MLXPersonalError("personal request adapter schema differs")

        if request.parent_adapter_path is not None:
            parent = _load_adapter_tensors(
                request.parent_adapter_path,
                model=model,
                foundation_sha256=request.foundation_sha256,
                adapter_schema_sha256=request.adapter_schema_sha256,
                adapter_id=request.parent_adapter_id,
            )
            model.import_floaty_state(_floaty_state(loaded, parent))

        training_records = [
            _dialogue_record(example)
            for example in request.training_examples + request.replay_examples
        ]
        validation_records = [
            _dialogue_record(example) for example in request.validation_examples
        ]
        retention_records = [
            _dialogue_record(example) for example in request.retention_examples
        ]
        all_examples = (
            request.training_examples
            + request.validation_examples
            + request.replay_examples
            + request.retention_examples
        )
        if len({example.lesson_id for example in all_examples}) != len(all_examples):
            raise MLXPersonalError("personal dialogue lesson IDs must be distinct")
        if len({example.content_sha256 for example in all_examples}) != len(
            all_examples
        ):
            raise MLXPersonalError("personal dialogue content must be distinct")

        training_batches, training_digest, _ = make_verified_lesson_batches(
            training_records,
            loaded.bundle.tokenizer,
            batch_size=self.config.batch_size,
            context_length=self.config.context_length,
            steps=self.config.steps,
            seed=self.config.seed + request.run_id,
        )
        validation_batches, validation_digest, _ = make_verified_lesson_batches(
            validation_records,
            loaded.bundle.tokenizer,
            batch_size=1,
            context_length=self.config.context_length,
            steps=1,
            seed=self.config.seed + request.run_id + 1,
        )
        if retention_records:
            retention_batches, retention_digest, _ = make_verified_lesson_batches(
                retention_records,
                loaded.bundle.tokenizer,
                batch_size=1,
                context_length=self.config.context_length,
                steps=len(retention_records),
                seed=self.config.seed + request.run_id + 2,
            )
        else:
            retention_batches = []
            retention_digest = None
        inactive_logit_bias = _inactive_text_logit_bias(
            loaded.bundle.config.vocab_size,
            self._inactive_text_ids,
        )
        _assert_batches_avoid_inactive_targets(
            (*training_batches, *validation_batches, *retention_batches),
            self._inactive_text_ids,
        )
        boundary = FoundationFloatyTrainingBoundary(
            model,
            expected_foundation_hash=loaded.internal_foundation_sha256,
            expected_config_hash=loaded.adapter_schema_sha256,
            expected_floaty_parameter_names=model.floaty_parameter_names(),
        )
        baseline_loss = _mean_loss(model, validation_batches, inactive_logit_bias)
        if not math.isfinite(baseline_loss):
            raise MLXPersonalError("personal holdout baseline is non-finite")
        baseline_retention = (
            _mean_loss(model, retention_batches, inactive_logit_bias)
            if retention_batches
            else None
        )
        if baseline_retention is not None and not math.isfinite(baseline_retention):
            raise MLXPersonalError("personal retention baseline is non-finite")
        optimizer = optim.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        optimizer.init(boundary.optimizer_parameters)
        mx.eval(optimizer.state)
        boundary.assert_optimizer_state(optimizer.state)

        def loss_fn(
            current_floaty: nn.Module,
            inputs: mx.array,
            targets: mx.array,
            mask: mx.array,
        ) -> mx.array:
            if current_floaty is not boundary.floaty:
                raise MLXPersonalError("optimizer received an unexpected module")
            return _masked_loss(
                model,
                inputs,
                targets,
                mask,
                inactive_logit_bias,
            )

        loss_and_grad = nn.value_and_grad(boundary.floaty, loss_fn)
        training_losses: list[float] = []
        model.train()
        for batch in training_batches:
            _raise_if_cancelled(request)
            model.train()
            packed = mx.array(batch.tokens)
            loss, gradients = loss_and_grad(
                boundary.floaty,
                packed[:, :-1],
                packed[:, 1:],
                mx.array(batch.loss_mask),
            )
            boundary.assert_gradient_whitelist(gradients)
            clipped, gradient_norm = optim.clip_grad_norm(
                gradients, self.config.grad_clip
            )
            mx.eval(loss, clipped, gradient_norm)
            loss_value = float(loss.item())
            norm_value = float(gradient_norm.item())
            if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                raise MLXPersonalError(
                    "personal training produced a non-finite gradient"
                )
            optimizer.update(boundary.floaty, clipped)
            mx.eval(boundary.floaty.parameters(), optimizer.state)
            boundary.assert_foundation_unchanged()
            boundary.assert_optimizer_state(optimizer.state)
            if not _tree_finite(boundary.floaty.parameters()) or not _tree_finite(
                optimizer.state
            ):
                raise MLXPersonalError("personal training produced a non-finite update")
            training_losses.append(loss_value)
        _raise_if_cancelled(request)
        candidate_loss = _mean_loss(model, validation_batches, inactive_logit_bias)
        candidate_retention = (
            _mean_loss(model, retention_batches, inactive_logit_bias)
            if retention_batches
            else None
        )
        boundary.assert_foundation_unchanged()
        foundation_after = model.foundation_hash()
        if foundation_after != loaded.internal_foundation_sha256:
            raise MLXPersonalError("MLX foundation changed during personal training")
        if not math.isfinite(candidate_loss):
            raise MLXPersonalError("personal candidate holdout loss is non-finite")
        if candidate_retention is not None and not math.isfinite(candidate_retention):
            raise MLXPersonalError("personal candidate retention loss is non-finite")

        tensors = _adapter_parameters(model)
        if set(tensors) != set(boundary.floaty_local_parameter_names):
            raise MLXPersonalError(
                "personal export escaped the floaty parameter surface"
            )
        artifact = request.work_dir / "adapter.safetensors"
        _raise_if_cancelled(request)
        if artifact.exists() or artifact.is_symlink():
            raise MLXPersonalError("personal candidate artifact already exists")
        mx.save_safetensors(
            str(artifact),
            tensors,
            metadata=_metadata_for_candidate(request),
        )
        with artifact.open("rb") as stream:
            os.fsync(stream.fileno())
        if os.name != "nt":
            artifact.chmod(0o600)
        artifact_sha256 = _sha256_file(artifact)
        return PersonalAdapterCandidate(
            artifact_path=artifact,
            artifact_format=_ARTIFACT_FORMAT,
            artifact_sha256=artifact_sha256,
            backend=self.backend_name,
            foundation_sha256_before=request.foundation_sha256,
            foundation_sha256_after=request.foundation_sha256,
            adapter_schema_sha256=request.adapter_schema_sha256,
            baseline_holdout_loss=baseline_loss,
            candidate_holdout_loss=candidate_loss,
            baseline_retention_loss=baseline_retention,
            candidate_retention_loss=candidate_retention,
            metrics={
                "optimizer": "mlx-adamw",
                "updates": len(training_losses),
                "mean_training_loss": sum(training_losses) / len(training_losses),
                "final_training_loss": training_losses[-1],
                "training_digest": training_digest,
                "validation_digest": validation_digest,
                "retention_digest": retention_digest,
                "new_training_examples": len(request.training_examples),
                "replay_examples": len(request.replay_examples),
                "retention_examples": len(request.retention_examples),
                "assistant_target_tokens": int(
                    sum(np.sum(batch.loss_mask) for batch in training_batches)
                ),
                "inactive_text_tokens_masked": len(self._inactive_text_ids),
                "parent_adapter_loaded": request.parent_adapter_id is not None,
                "holdout_evaluations": 2,
                "holdout_used_for_selection": False,
                "foundation_trainable": False,
                "tensor_scope": "floaty-only",
            },
        )


class MLXPersonalAdapterRuntime:
    """Resident, cached MLX inference runtime with atomic floaty swaps."""

    def __init__(
        self,
        bundle: PortableBundle,
        *,
        config: MLXPersonalConfig | None = None,
    ) -> None:
        self.config = config or MLXPersonalConfig()
        self.config.validate()
        self._loaded = _load_composed_model(bundle, self.config)
        self._model = self._loaded.model
        self._model.freeze()
        self._model.assert_parameter_partition()
        self._tokenizer = self._loaded.bundle.tokenizer
        self._inactive_text_ids = self._tokenizer.inactive_text_token_ids
        self._generation_blocked_ids = self._tokenizer.generation_blocked_token_ids
        self._generation_logit_bias = _inactive_text_logit_bias(
            self._loaded.bundle.config.vocab_size,
            self._generation_blocked_ids,
        )
        self._first_generation_logit_bias = _inactive_text_logit_bias(
            self._loaded.bundle.config.vocab_size,
            self._tokenizer.first_generation_blocked_token_ids,
        )
        mx.eval(
            self._generation_logit_bias,
            self._first_generation_logit_bias,
        )
        self._foundation_sha256 = str(self._loaded.bundle.manifest["weights_sha256"])
        self._adapter_schema_sha256 = self._loaded.adapter_schema_sha256
        self._active_adapter_id: str | None = None
        self._lock = threading.RLock()
        self._prepare_token = object()
        self._answer_calls = 0
        self._cached_tokens = 0
        base_tensors = {
            name: mx.array(value)
            for name, value in _adapter_parameters(self._model).items()
        }
        mx.eval(base_tensors)
        self._foundation_only = tuple(sorted(base_tensors.items()))

    @classmethod
    def from_bundle(
        cls,
        bundle: PortableBundle,
        *,
        config: MLXPersonalConfig | None = None,
    ) -> "MLXPersonalAdapterRuntime":
        return cls(bundle, config=config)

    def personal_runtime_identity(self) -> PersonalRuntimeIdentity:
        with self._lock:
            return PersonalRuntimeIdentity(
                backend=MLX_PERSONAL_BACKEND,
                foundation_sha256=self._foundation_sha256,
                adapter_schema_sha256=self._adapter_schema_sha256,
                active_adapter_id=self._active_adapter_id,
            )

    def prepare_personal_adapter(
        self,
        artifact: PersonalAdapterArtifact,
    ) -> PreparedMLXPersonalAdapter:
        if type(artifact) is not PersonalAdapterArtifact:
            raise TypeError("artifact must be a PersonalAdapterArtifact")
        manifest = artifact.manifest
        if (
            manifest.backend != MLX_PERSONAL_BACKEND
            or manifest.foundation_sha256 != self._foundation_sha256
            or manifest.adapter_schema_sha256 != self._adapter_schema_sha256
        ):
            raise MLXPersonalError("published personal adapter is incompatible")
        digest = _sha256_file(artifact.artifact_path)
        if digest != manifest.artifact_sha256:
            raise MLXPersonalError("published personal adapter checksum changed")
        tensors = _load_adapter_tensors(
            artifact.artifact_path,
            model=self._model,
            foundation_sha256=self._foundation_sha256,
            adapter_schema_sha256=self._adapter_schema_sha256,
            adapter_id=manifest.adapter_id,
        )
        return PreparedMLXPersonalAdapter(
            adapter_id=manifest.adapter_id,
            artifact_sha256=digest,
            tensors=tuple(sorted(tensors.items())),
            runtime_token=self._prepare_token,
        )

    def activate_prepared_personal_adapter(
        self,
        prepared: object,
        adapter_id: str,
    ) -> None:
        if type(prepared) is not PreparedMLXPersonalAdapter:
            raise MLXPersonalError("prepared adapter has an unexpected MLX type")
        if (
            prepared.adapter_id != adapter_id
            or prepared.runtime_token is not self._prepare_token
        ):
            raise MLXPersonalError("prepared adapter identity differs")
        with self._lock:
            previous = self._model.export_floaty_state()
            previous_id = self._active_adapter_id
            try:
                self._model.import_floaty_state(
                    _floaty_state(self._loaded, dict(prepared.tensors))
                )
                self._model.freeze()
                self._model.eval()
                self._active_adapter_id = adapter_id
            except Exception:
                self._model.import_floaty_state(previous)
                self._model.freeze()
                self._model.eval()
                self._active_adapter_id = previous_id
                raise

    def activate_foundation_only(self) -> None:
        with self._lock:
            if self._active_adapter_id is None:
                self._model.freeze()
                self._model.assert_parameter_partition()
                return
            previous = self._model.export_floaty_state()
            previous_id = self._active_adapter_id
            try:
                self._model.import_floaty_state(
                    _floaty_state(self._loaded, dict(self._foundation_only))
                )
                self._model.freeze()
                self._model.eval()
                self._active_adapter_id = None
            except Exception:
                self._model.import_floaty_state(previous)
                self._model.freeze()
                self._model.eval()
                self._active_adapter_id = previous_id
                raise

    def _prompt_ids(self, turns: Sequence[tuple[str, str]]) -> list[int]:
        return compact_personal_dialogue_prompt(
            self._loaded.bundle.tokenizer,
            turns,
            context_length=self._loaded.bundle.config.context_length,
        )

    def personal_training_target_fits(self, prompt: str, human_response: str) -> bool:
        return personal_dialogue_target_fits(
            self._loaded.bundle.tokenizer,
            prompt,
            human_response,
            context_length=self.config.context_length,
        )

    def answer(
        self,
        turns: Sequence[tuple[str, str]],
        *,
        max_tokens: int = 32,
    ) -> str:
        """Greedily generate a new assistant turn using incremental caches."""

        if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
            raise TypeError("turns must be a sequence of role/content pairs")
        normalized: list[tuple[str, str]] = []
        for turn in turns:
            if (
                not isinstance(turn, Sequence)
                or isinstance(turn, (str, bytes))
                or len(turn) != 2
            ):
                raise TypeError("each dialogue turn must be a role/content pair")
            role, content = turn
            if not isinstance(role, str) or role not in ROLE_NAMES:
                raise ValueError(f"unsupported scratch dialogue role: {role!r}")
            if not isinstance(content, str):
                raise TypeError("dialogue content must be a string")
            normalized.append((role, content))
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        limit = min(max_tokens, self.config.max_tokens)
        prefix = self._prompt_ids(normalized)
        stop_tokens = {
            self._tokenizer.pad_id,
            self._tokenizer.bos_id,
            self._tokenizer.eos_id,
            self._tokenizer.end_turn_id,
            *(self._tokenizer.role_id(role) for role in ROLE_NAMES),
        }
        generated: list[int] = []
        with self._lock:
            self._model.eval()
            logits, cache = self._model.forward_cached(
                mx.array([prefix], dtype=mx.int32)
            )
            cached_tokens = len(prefix)
            mx.eval(logits)
            for index in range(limit):
                next_logits = logits[0, -1].astype(mx.float32) + (
                    self._first_generation_logit_bias
                    if index == 0
                    else self._generation_logit_bias
                )
                token = int(mx.argmax(next_logits).item())
                if token in stop_tokens:
                    break
                generated.append(token)
                if index + 1 == limit:
                    break
                logits, cache = self._model.forward_cached(
                    mx.array([[token]], dtype=mx.int32),
                    cache,
                )
                cached_tokens += 1
                mx.eval(logits)
            self._answer_calls += 1
            self._cached_tokens += cached_tokens
        return sanitize_generated_text(
            self._tokenizer.decode(generated, skip_special_tokens=True)
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": MLX_PERSONAL_BACKEND,
                "device": "apple-silicon",
                "model_size": self._loaded.bundle.model_size,
                "parameters": self._loaded.bundle.config.expected_parameter_count(),
                "adapter_parameters": self._model.floaty_parameter_count(),
                "active_adapter_parameters": (
                    self._model.floaty_parameter_count()
                    if self._active_adapter_id is not None
                    else 0
                ),
                "active_adapter_id": self._active_adapter_id,
                "foundation_sha256": self._foundation_sha256,
                "adapter_schema_sha256": self._adapter_schema_sha256,
                "foundation_trainable": False,
                "adapter_trainable": bool(
                    tree_flatten(self._model.floaty.trainable_parameters())
                ),
                "foundation_frozen": not bool(
                    tree_flatten(self._model.foundation.trainable_parameters())
                ),
                "cached_inference": True,
                "cached_decode": True,
                "cache_scope": "per-answer",
                "answer_calls": self._answer_calls,
                "cached_tokens": self._cached_tokens,
                "scratch_only": True,
                "external_weights": False,
                "inactive_text_tokens_masked": len(self._inactive_text_ids),
                "generation_blocked_text_tokens": len(self._generation_blocked_ids),
            }


class MLXPersonalFoundationTrainer:
    """Direct conversational trainer for the complete MLX foundation."""

    backend_name = MLX_PERSONAL_BACKEND

    def __init__(
        self,
        bundle: PortableBundle,
        *,
        config: MLXFoundationConfig | None = None,
    ) -> None:
        verified = _revalidate_bundle(bundle)
        self.config = config or _default_mlx_foundation_config(verified)
        self.config.validate()
        if self.config.context_length > verified.config.context_length:
            raise ValueError("foundation context exceeds the portable model context")
        self._config_sha256 = str(verified.manifest["config_sha256"])
        self._tokenizer_sha256 = str(verified.manifest["tokenizer_sha256"])

    def train_foundation(
        self,
        request: PersonalFoundationTrainingRequest,
    ) -> PersonalFoundationCandidate:
        if type(request) is not PersonalFoundationTrainingRequest:
            raise TypeError("request must be a PersonalFoundationTrainingRequest")
        validate_foundation_request(request)
        parent_identity = request.parent.identity
        if parent_identity.backend != self.backend_name:
            raise MLXPersonalError("foundation request selected another backend")
        parent = load_portable_bundle(request.parent.bundle_path)
        actual_parent = _personal_foundation_identity(
            parent,
            parent_identity.foundation_id,
        )
        if actual_parent != parent_identity:
            raise MLXPersonalError("foundation request parent identity changed")
        if (
            actual_parent.config_sha256 != self._config_sha256
            or actual_parent.tokenizer_sha256 != self._tokenizer_sha256
        ):
            raise MLXPersonalError("foundation request parent lineage is incompatible")
        if request.work_dir.is_symlink():
            raise MLXPersonalError("foundation work directory must not be a symlink")
        for example in request.new_examples + request.replay_examples:
            if not personal_dialogue_target_fits(
                parent.tokenizer,
                example.prompt,
                example.human_response,
                context_length=self.config.context_length,
            ):
                raise MLXPersonalError(
                    "foundation dialogue target exceeds the bounded context"
                )
        if (
            request.cancellation_event is not None
            and request.cancellation_event.is_set()
        ):
            raise MLXPersonalError("foundation training was cancelled during shutdown")

        candidate_path = request.work_dir / "bundle"
        if candidate_path.exists() or candidate_path.is_symlink():
            raise MLXPersonalError("foundation candidate path already exists")
        _mlx_foundation_disk_preflight(request.work_dir, parent, self.config)
        try:
            result = _train_mlx_foundation(
                parent,
                run_id=request.run_id,
                new_examples=request.new_examples,
                monitoring_examples=request.monitoring_examples,
                replay_examples=request.replay_examples,
                config=self.config,
                cancellation_event=request.cancellation_event,
            )
        except MemoryError as error:
            _clear_mlx_device_cache()
            raise MLXPersonalError(
                "not enough device memory for a full foundation update; "
                "the active parent remains unchanged"
            ) from error
        except RuntimeError as error:
            detail = str(error).casefold()
            if any(
                marker in detail
                for marker in (
                    "out of memory",
                    "not enough memory",
                    "failed to allocate",
                    "resource limit",
                )
            ):
                _clear_mlx_device_cache()
                raise MLXPersonalError(
                    "not enough device memory for a full foundation update; "
                    "the active parent remains unchanged"
                ) from error
            raise
        candidate_bundle = _publish_mlx_foundation_bundle(
            parent,
            result,
            candidate_path,
        )
        if (
            request.cancellation_event is not None
            and request.cancellation_event.is_set()
        ):
            raise MLXPersonalError("foundation training was cancelled during shutdown")
        candidate_identity = _personal_foundation_identity(
            candidate_bundle,
            request.candidate_foundation_id,
        )
        if (
            candidate_identity.config_sha256 != parent_identity.config_sha256
            or candidate_identity.tokenizer_sha256 != parent_identity.tokenizer_sha256
            or candidate_identity.backend_contract_sha256
            != parent_identity.backend_contract_sha256
            or candidate_identity.weights_sha256 == parent_identity.weights_sha256
            or candidate_identity.bundle_sha256 == parent_identity.bundle_sha256
        ):
            raise MLXPersonalError("foundation candidate lineage is incompatible")
        metrics: dict[str, Any] = {
            **result.metrics,
            "baseline_monitoring_loss": result.baseline_monitoring_loss,
            "candidate_monitoring_loss": result.candidate_monitoring_loss,
            "monitoring_loss_delta": (
                result.candidate_monitoring_loss - result.baseline_monitoring_loss
            ),
            "quality_gate_applied": False,
            "integrity_checks_passed": True,
        }
        validate_foundation_metrics(metrics)
        return PersonalFoundationCandidate(
            bundle_path=candidate_bundle.root,
            identity=candidate_identity,
            parent_foundation_id=parent_identity.foundation_id,
            parent_bundle_sha256=parent_identity.bundle_sha256,
            metrics=metrics,
        )


class MLXPersonalFoundationRuntime(MLXPersonalAdapterRuntime):
    """Resident MLX runtime with prepared whole-foundation atomic swaps."""

    def __init__(
        self,
        bundle: PortableBundle,
        *,
        foundation_id: str = "foundation-base",
        config: MLXPersonalConfig | None = None,
    ) -> None:
        verified = _revalidate_bundle(bundle)
        selected_config = config or _default_mlx_foundation_runtime_config(verified)
        super().__init__(verified, config=selected_config)
        self._foundation_runtime_token = object()
        self._active_foundation_identity = _personal_foundation_identity(
            self._loaded.bundle,
            foundation_id,
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: PortableBundle,
        *,
        foundation_id: str = "foundation-base",
        config: MLXPersonalConfig | None = None,
    ) -> "MLXPersonalFoundationRuntime":
        return cls(bundle, foundation_id=foundation_id, config=config)

    def personal_foundation_source(self) -> PersonalFoundationSource:
        with self._lock:
            return PersonalFoundationSource(
                bundle_path=self._loaded.bundle.root,
                identity=self._active_foundation_identity,
            )

    def prepare_personal_adapter(self, artifact: PersonalAdapterArtifact) -> object:
        raise MLXPersonalError(
            "legacy adapters cannot be rebound to a direct-foundation runtime"
        )

    def activate_prepared_personal_adapter(
        self,
        prepared: object,
        adapter_id: str,
    ) -> None:
        raise MLXPersonalError(
            "legacy adapters cannot be activated on a direct-foundation runtime"
        )

    def activate_foundation_only(self) -> None:
        raise MLXPersonalError(
            "legacy adapter rollback is unavailable on a direct-foundation runtime"
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = super().status()
            status.pop("adapter_schema_sha256", None)
            status.update(
                {
                    "active_foundation_id": (
                        self._active_foundation_identity.foundation_id
                    ),
                    "bundle_sha256": self._active_foundation_identity.bundle_sha256,
                    "weights_sha256": self._active_foundation_identity.weights_sha256,
                    "foundation_sha256": (
                        self._active_foundation_identity.weights_sha256
                    ),
                    "adapter_parameters": 0,
                    "active_adapter_parameters": 0,
                    "active_adapter_id": None,
                    "automatic_learning": "full-foundation-every-five-turns",
                    "all_foundation_parameters_optimizer_eligible": True,
                    "optimizer_state_continuity": False,
                    "foundation_trainable": False,
                    "foundation_frozen": True,
                }
            )
            return status

    def prepare_personal_foundation(
        self,
        artifact: PersonalFoundationArtifact,
    ) -> PreparedMLXPersonalFoundation:
        if type(artifact) is not PersonalFoundationArtifact:
            raise TypeError("artifact must be a PersonalFoundationArtifact")
        artifact.manifest.validate()
        identity = artifact.manifest.identity
        identity.validate()
        if identity.backend != MLX_PERSONAL_BACKEND:
            raise MLXPersonalError("published foundation selected another backend")
        bundle = load_portable_bundle(artifact.bundle_path)
        actual = _personal_foundation_identity(bundle, identity.foundation_id)
        if actual != identity:
            raise MLXPersonalError("published foundation identity changed")
        current = self.personal_foundation_source().identity
        if (
            actual.config_sha256 != current.config_sha256
            or actual.tokenizer_sha256 != current.tokenizer_sha256
            or actual.backend_contract_sha256 != current.backend_contract_sha256
        ):
            raise MLXPersonalError("published foundation is runtime-incompatible")
        loaded = _load_composed_model(bundle, self.config)
        if str(loaded.bundle.manifest["weights_sha256"]) != actual.weights_sha256:
            raise MLXPersonalError("prepared foundation weights identity differs")
        _mlx_foundation_forward_smoke(loaded)
        return PreparedMLXPersonalFoundation(
            identity=actual,
            bundle_path=loaded.bundle.root,
            loaded=loaded,
            runtime_token=self._foundation_runtime_token,
        )

    def activate_prepared_personal_foundation(
        self,
        prepared: object,
        identity: PersonalFoundationIdentity,
    ) -> None:
        if type(prepared) is not PreparedMLXPersonalFoundation:
            raise MLXPersonalError("prepared foundation has an unexpected MLX type")
        identity.validate()
        if (
            prepared.runtime_token is not self._foundation_runtime_token
            or prepared.identity != identity
        ):
            raise MLXPersonalError("prepared foundation identity differs")
        with self._lock:
            previous = {
                "loaded": self._loaded,
                "model": self._model,
                "tokenizer": self._tokenizer,
                "inactive_text_ids": self._inactive_text_ids,
                "generation_blocked_ids": self._generation_blocked_ids,
                "generation_logit_bias": self._generation_logit_bias,
                "first_generation_logit_bias": self._first_generation_logit_bias,
                "foundation_sha256": self._foundation_sha256,
                "adapter_schema_sha256": self._adapter_schema_sha256,
                "active_adapter_id": self._active_adapter_id,
                "foundation_only": self._foundation_only,
                "active_foundation_identity": self._active_foundation_identity,
            }
            try:
                loaded = prepared.loaded
                model = loaded.model
                model.freeze()
                model.eval()
                model.assert_parameter_partition()
                tokenizer = loaded.bundle.tokenizer
                generation_blocked_ids = tokenizer.generation_blocked_token_ids
                generation_logit_bias = _inactive_text_logit_bias(
                    loaded.bundle.config.vocab_size,
                    generation_blocked_ids,
                )
                first_generation_logit_bias = _inactive_text_logit_bias(
                    loaded.bundle.config.vocab_size,
                    tokenizer.first_generation_blocked_token_ids,
                )
                base_tensors = {
                    name: mx.array(value)
                    for name, value in _adapter_parameters(model).items()
                }
                mx.eval(
                    generation_logit_bias,
                    first_generation_logit_bias,
                    base_tensors,
                )
                self._loaded = loaded
                self._model = model
                self._tokenizer = tokenizer
                self._inactive_text_ids = tokenizer.inactive_text_token_ids
                self._generation_blocked_ids = generation_blocked_ids
                self._generation_logit_bias = generation_logit_bias
                self._first_generation_logit_bias = first_generation_logit_bias
                self._foundation_sha256 = identity.weights_sha256
                self._adapter_schema_sha256 = loaded.adapter_schema_sha256
                self._active_adapter_id = None
                self._foundation_only = tuple(sorted(base_tensors.items()))
                self._active_foundation_identity = identity
                if self.personal_foundation_source().identity != identity:
                    raise MLXPersonalError("runtime did not activate the foundation")
            except Exception:
                self._loaded = previous["loaded"]
                self._model = previous["model"]
                self._tokenizer = previous["tokenizer"]
                self._inactive_text_ids = previous["inactive_text_ids"]
                self._generation_blocked_ids = previous["generation_blocked_ids"]
                self._generation_logit_bias = previous["generation_logit_bias"]
                self._first_generation_logit_bias = previous[
                    "first_generation_logit_bias"
                ]
                self._foundation_sha256 = previous["foundation_sha256"]
                self._adapter_schema_sha256 = previous["adapter_schema_sha256"]
                self._active_adapter_id = previous["active_adapter_id"]
                self._foundation_only = previous["foundation_only"]
                self._active_foundation_identity = previous[
                    "active_foundation_identity"
                ]
                raise


def create_mlx_personal_trainer(
    bundle: PortableBundle,
    *,
    config: MLXPersonalConfig | None = None,
) -> MLXPersonalAdapterTrainer:
    """Create an isolated trainer which reloads its own foundation per run."""

    return MLXPersonalAdapterTrainer.from_bundle(bundle, config=config)


def create_mlx_personal_runtime(
    bundle: PortableBundle,
    *,
    config: MLXPersonalConfig | None = None,
) -> MLXPersonalAdapterRuntime:
    """Create the resident MLX inference runtime."""

    return MLXPersonalAdapterRuntime.from_bundle(bundle, config=config)


def create_mlx_personal_backend(
    bundle: PortableBundle,
    *,
    rank: int = 8,
    config: MLXPersonalConfig | None = None,
) -> tuple[MLXPersonalAdapterTrainer, MLXPersonalAdapterRuntime]:
    """Build matching isolated trainer and resident runtime for one bundle."""

    if config is not None and rank != 8 and rank != config.rank:
        raise ValueError("rank and explicit MLXPersonalConfig disagree")
    selected = config or MLXPersonalConfig(rank=rank)
    selected.validate()
    return (
        create_mlx_personal_trainer(bundle, config=selected),
        create_mlx_personal_runtime(bundle, config=selected),
    )


def create_mlx_personal_foundation_backend(
    bundle: PortableBundle,
    *,
    foundation_id: str = "foundation-base",
    config: MLXFoundationConfig | None = None,
    runtime_config: MLXPersonalConfig | None = None,
) -> tuple[MLXPersonalFoundationTrainer, MLXPersonalFoundationRuntime]:
    """Build the distinct full-foundation trainer and matching MLX runtime."""

    verified = _revalidate_bundle(bundle)
    selected = config or _default_mlx_foundation_config(verified)
    selected.validate()
    if selected.context_length > verified.config.context_length:
        raise ValueError("foundation context exceeds the portable model context")
    trainer = MLXPersonalFoundationTrainer(verified, config=selected)
    selected_runtime = runtime_config or _default_mlx_foundation_runtime_config(
        verified,
        context_length=selected.context_length,
    )
    runtime = MLXPersonalFoundationRuntime.from_bundle(
        verified,
        foundation_id=foundation_id,
        config=selected_runtime,
    )
    return trainer, runtime


__all__ = [
    "MLX_PERSONAL_ADAPTER_FORMAT",
    "MLXFoundationConfig",
    "MLXPersonalAdapterRuntime",
    "MLXPersonalAdapterTrainer",
    "MLXPersonalFoundationRuntime",
    "MLXPersonalFoundationTrainer",
    "MLXPersonalConfig",
    "MLXPersonalError",
    "PreparedMLXPersonalAdapter",
    "PreparedMLXPersonalFoundation",
    "create_mlx_personal_backend",
    "create_mlx_personal_foundation_backend",
    "create_mlx_personal_runtime",
    "create_mlx_personal_trainer",
]
