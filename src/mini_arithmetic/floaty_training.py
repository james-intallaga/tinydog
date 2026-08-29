"""Fail-closed local training for a foundation-plus-floaty language model.

The optimizer in this module is attached to the floaty module itself, never to
the composed model.  Consequently foundation tensors cannot appear in the
gradient tree or optimizer state.  The boundary also freezes the entire model,
unfreezes only the exact floaty whitelist, and verifies the foundation tensor
identities after every update.

This is an isolated candidate builder.  It does not replace an active runtime
model or submit an update to a community service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np

from .continual import _lesson_content, render_verified_lesson
from .community_updates import publish_directory_no_replace


FLOATY_TRAINING_SCHEMA_VERSION = 1
_SHA256_LENGTH = 64
FLOATY_TRAINING_OBJECTIVE = (
    "scratch-assistant-role-content-end-turn-eos-plus-human-dialogue-v2"
)
_TRUSTED_DIALOGUE_SOURCE = "human-conversation"
_TRUSTED_DIALOGUE_VERIFIER = "trusted-human-dialogue-v1"


class FloatyBoundaryError(RuntimeError):
    """The model does not expose the strict foundation/floaty contract."""


@dataclass(frozen=True)
class FloatyTrainingConfig:
    """Controls a small, local, verified-lesson candidate run."""

    steps: int = 50
    batch_size: int = 4
    context_length: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    minimum_training_lessons: int = 4
    minimum_validation_lessons: int = 1
    minimum_validation_improvement: float = 0.0
    maximum_protected_loss_regression: float = 0.02
    seed: int = 42

    def validate(self) -> None:
        integer_fields = {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "minimum_training_lessons": self.minimum_training_lessons,
            "minimum_validation_lessons": self.minimum_validation_lessons,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        finite_fields = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "minimum_validation_improvement": self.minimum_validation_improvement,
            "maximum_protected_loss_regression": self.maximum_protected_loss_regression,
        }
        for name, value in finite_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0.0 or self.grad_clip <= 0.0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.minimum_validation_improvement < 0.0:
            raise ValueError("minimum_validation_improvement cannot be negative")
        if self.maximum_protected_loss_regression < 0.0:
            raise ValueError("maximum_protected_loss_regression cannot be negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class FloatyCandidateResult:
    """Local artifact and gate decision from :func:`run_floaty_candidate`."""

    run_id: str
    accepted: bool
    rolled_back: bool
    floaty_weights_path: Path | None
    provenance_path: Path
    report_path: Path
    report: Mapping[str, Any]


@dataclass(frozen=True)
class MaskedLessonSequence:
    """One dialogue plus an auditable assistant-content target mask."""

    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    boundaries: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) < 2:
            raise ValueError("Masked lesson sequence needs at least two tokens")
        if len(self.loss_mask) != len(self.token_ids) - 1:
            raise ValueError("Loss mask must align with next-token targets")
        if not any(self.loss_mask):
            raise ValueError("Masked lesson contains no assistant-answer targets")


@dataclass(frozen=True)
class MaskedLessonBatch:
    """Fixed-shape token rows and assistant-only next-token masks."""

    tokens: np.ndarray
    loss_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2 or self.loss_mask.ndim != 2:
            raise ValueError("Masked lesson batch tensors must be rank two")
        if self.loss_mask.shape != (self.tokens.shape[0], self.tokens.shape[1] - 1):
            raise ValueError("Masked lesson batch loss mask has the wrong shape")
        if not np.isin(self.loss_mask, (0.0, 1.0)).all() or not self.loss_mask.any():
            raise ValueError("Masked lesson batch requires binary assistant targets")


def floaty_training_recipe_sha256(config: FloatyTrainingConfig) -> str:
    """Hash the exact local optimization and target-selection contract."""

    config.validate()
    payload = {
        "schema": "tinydog.local-floaty-training.v1",
        "objective": FLOATY_TRAINING_OBJECTIVE,
        "optimizer": "mlx-adamw",
        "gradient_scope": "model.floaty",
        "training_config": asdict(config),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise FloatyBoundaryError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise FloatyBoundaryError(
            f"{label} must be a lowercase SHA-256 digest"
        ) from error
    if value != value.lower():
        raise FloatyBoundaryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _call_digest(model: Any, method_name: str, label: str) -> str:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise FloatyBoundaryError(f"Model is missing required {method_name}()")
    return _sha256(method(), label)


def _parameter_dict(module: nn.Module) -> dict[str, mx.array]:
    return {
        name: value
        for name, value in tree_flatten(module.parameters())
        if isinstance(value, mx.array)
    }


def _trainable_parameter_dict(module: nn.Module) -> dict[str, mx.array]:
    return {
        name: value
        for name, value in tree_flatten(module.trainable_parameters())
        if isinstance(value, mx.array)
    }


class FoundationFloatyTrainingBoundary:
    """Verified view of a composed model with one trainable floaty subtree.

    The model contract is deliberately structural.  A compatible model must
    expose ``foundation`` and ``floaty`` modules plus these methods:

    - ``foundation_hash()`` and ``config_hash()``
    - ``floaty_parameter_names()`` (wrapper-qualified ``floaty.*`` names)
    - ``trainable_parameter_names()`` (the exact same ``floaty.*`` names)
    - ``assert_parameter_partition()`` and ``freeze_foundation()``
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        expected_foundation_hash: str,
        expected_config_hash: str,
        expected_floaty_parameter_names: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise FloatyBoundaryError("Model must be an MLX module")
        foundation = getattr(model, "foundation", None)
        floaty = getattr(model, "floaty", None)
        if not isinstance(foundation, nn.Module) or not isinstance(floaty, nn.Module):
            raise FloatyBoundaryError(
                "Model must expose separate foundation and floaty MLX modules"
            )
        self.model = model
        self.foundation = foundation
        self.floaty = floaty
        self.foundation_hash = _call_digest(model, "foundation_hash", "foundation hash")
        self.config_hash = _call_digest(model, "config_hash", "floaty config hash")
        if self.foundation_hash != _sha256(
            expected_foundation_hash, "expected foundation hash"
        ):
            raise FloatyBoundaryError(
                "Foundation hash does not match the pinned active foundation"
            )
        if self.config_hash != _sha256(
            expected_config_hash, "expected floaty config hash"
        ):
            raise FloatyBoundaryError(
                "Floaty config hash does not match the pinned adapter shape"
            )

        partition_check = getattr(model, "assert_parameter_partition", None)
        freeze_foundation = getattr(model, "freeze_foundation", None)
        names_method = getattr(model, "floaty_parameter_names", None)
        trainable_names_method = getattr(model, "trainable_parameter_names", None)
        if not all(
            callable(method)
            for method in (
                partition_check,
                freeze_foundation,
                names_method,
                trainable_names_method,
            )
        ):
            raise FloatyBoundaryError(
                "Model is missing the strict parameter-partition API"
            )
        model.freeze()
        floaty.unfreeze()
        try:
            freeze_foundation()
            partition_check()
        except Exception as error:
            raise FloatyBoundaryError("Model parameter partition is invalid") from error
        actual_floaty = tuple(_parameter_dict(floaty))
        wrapper_floaty = tuple(f"floaty.{name}" for name in actual_floaty)
        declared_floaty = tuple(names_method())
        if not actual_floaty or len(set(actual_floaty)) != len(actual_floaty):
            raise FloatyBoundaryError(
                "Floaty module must contain unique parameter names"
            )
        if set(declared_floaty) != set(wrapper_floaty):
            raise FloatyBoundaryError(
                "Declared floaty whitelist does not match floaty tensors"
            )
        if expected_floaty_parameter_names is not None:
            expected = tuple(expected_floaty_parameter_names)
            if not expected or len(set(expected)) != len(expected):
                raise FloatyBoundaryError(
                    "Expected floaty whitelist must be non-empty and unique"
                )
            if set(expected) != set(wrapper_floaty):
                raise FloatyBoundaryError(
                    "Floaty tensors do not match the caller-pinned whitelist"
                )

        declared_trainable = tuple(trainable_names_method())
        actual_trainable = tuple(_trainable_parameter_dict(model))
        if set(declared_trainable) != set(wrapper_floaty):
            raise FloatyBoundaryError(
                "Model declared a trainable tensor outside floaty"
            )
        if set(actual_trainable) != set(wrapper_floaty):
            raise FloatyBoundaryError("Model exposed a trainable tensor outside floaty")
        if _trainable_parameter_dict(foundation):
            raise FloatyBoundaryError("Foundation tensors remain trainable")

        foundation_ids = {id(value) for value in _parameter_dict(foundation).values()}
        floaty_ids = {id(value) for value in _parameter_dict(floaty).values()}
        if not foundation_ids:
            raise FloatyBoundaryError("Foundation module contains no parameters")
        if foundation_ids & floaty_ids:
            raise FloatyBoundaryError("Foundation and floaty tensors may not alias")
        self.floaty_local_parameter_names = actual_floaty
        self.floaty_parameter_names = wrapper_floaty
        self.trainable_parameter_names = wrapper_floaty
        optimizer_parameters = getattr(model, "floaty_parameters", None)
        if not callable(optimizer_parameters):
            raise FloatyBoundaryError("Model is missing required floaty_parameters()")
        self.optimizer_parameters = optimizer_parameters()
        if set(name for name, _ in tree_flatten(self.optimizer_parameters)) != set(
            actual_floaty
        ):
            raise FloatyBoundaryError(
                "Floaty optimizer surface does not match its whitelist"
            )
        self._foundation_snapshot = tree_map(
            lambda value: value, foundation.parameters()
        )
        self._foundation_leaves = _parameter_dict(foundation)

    def assert_foundation_unchanged(self) -> None:
        current = _parameter_dict(self.foundation)
        if set(current) != set(self._foundation_leaves):
            raise FloatyBoundaryError(
                "Foundation parameter structure changed during local training"
            )
        if any(
            current[name] is not value
            for name, value in self._foundation_leaves.items()
        ):
            raise FloatyBoundaryError("Foundation tensor changed during local training")
        if _trainable_parameter_dict(self.foundation):
            raise FloatyBoundaryError(
                "Foundation became trainable during local training"
            )

    def restore_foundation(self) -> None:
        self.foundation.update(self._foundation_snapshot)
        self.foundation.freeze()
        mx.eval(self.foundation.parameters())

    def assert_gradient_whitelist(self, gradients: Any) -> tuple[str, ...]:
        names = tuple(name for name, _ in tree_flatten(gradients))
        if set(names) != set(self.floaty_local_parameter_names):
            raise FloatyBoundaryError(
                "Gradient tree contains a tensor outside floaty whitelist"
            )
        return names

    def assert_optimizer_state(self, state: Any) -> tuple[str, ...]:
        names = tuple(name for name, _ in tree_flatten(state))
        parameter_state_names = tuple(
            name for name in names if name not in {"step", "learning_rate"}
        )
        allowed = self.floaty_local_parameter_names
        if any(
            not any(
                name == parameter or name.startswith(f"{parameter}.")
                for parameter in allowed
            )
            for name in parameter_state_names
        ):
            raise FloatyBoundaryError(
                "Optimizer state contains a tensor outside floaty whitelist"
            )
        return parameter_state_names


def _record_value(record: Any, *names: str) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    return None


def _lesson_key(record: Any) -> str:
    problem = _record_value(record, "problem", "question", "prompt")
    answer = _record_value(record, "correct_answer", "verified_answer", "answer")
    canonical = (
        f"{' '.join(str(problem).split()).casefold()}\0{' '.join(str(answer).split())}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lesson_identifier(record: Any) -> int | str | None:
    value = _record_value(record, "id", "lesson_id")
    return (
        value if isinstance(value, (int, str)) and not isinstance(value, bool) else None
    )


def _is_trusted_human_dialogue(record: Any) -> bool:
    metadata = _record_value(record, "metadata")
    return (
        _record_value(record, "source") == _TRUSTED_DIALOGUE_SOURCE
        and _record_value(record, "verifier") == _TRUSTED_DIALOGUE_VERIFIER
        and isinstance(metadata, Mapping)
        and metadata.get("human_authored_target") is True
        and metadata.get("consent_local_training") is True
    )


def _authored_dialogue_text(value: Any, label: str) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError(f"Trusted human dialogue requires nonempty {label}")
    return text


def _training_lesson_content(record: Any) -> tuple[str, str, str, str]:
    if not _is_trusted_human_dialogue(record):
        return _lesson_content(record)
    problem = _authored_dialogue_text(
        _record_value(record, "problem", "question", "prompt"),
        "prompt",
    )
    answer = _authored_dialogue_text(
        _record_value(record, "correct_answer", "verified_answer", "answer"),
        "human response",
    )
    rendered = json.dumps(
        {"kind": "human-dialogue", "prompt": problem, "response": answer},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return rendered, problem, answer, ""


def _compact_dialogue_turns(
    turns: Sequence[tuple[str, str]],
    tokenizer: Any,
    max_length: int,
) -> list[tuple[str, str]]:
    """Trim prompt/review tails while preserving the verified answer exactly."""

    encode_boundaries = tokenizer.encode_dialogue_with_boundaries
    blank_turns = [(role, "") for role, _ in turns]
    blank_ids, _ = encode_boundaries(blank_turns, bos=True, eos=True)
    answer_tokens = list(tokenizer.encode(turns[-1][1]))
    available = max_length - len(blank_ids) - len(answer_tokens)
    if available < 0:
        raise ValueError(
            "Verified assistant answer is too long for the training context"
        )
    context_turns = list(turns[:-1])
    shares = [available // len(context_turns)] * len(context_turns)
    for index in range(available % len(context_turns)):
        shares[index] += 1
    compacted: list[tuple[str, str]] = []
    for (role, content), budget in zip(context_turns, shares, strict=True):
        encoded = list(tokenizer.encode(content))
        if len(encoded) <= budget:
            compacted.append((role, content))
            continue
        if budget == 0:
            compacted.append((role, ""))
            continue
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise ValueError("Tokenizer cannot safely compact a verified dialogue")
        suffix = str(decode(encoded[-budget:]))
        while suffix and len(tokenizer.encode(suffix)) > budget:
            suffix = suffix[1:]
        compacted.append((role, suffix))
    compacted.append(turns[-1])
    encoded, _ = encode_boundaries(compacted, bos=True, eos=True)
    if len(encoded) > max_length:
        raise ValueError("Tokenizer could not compact a verified dialogue safely")
    return compacted


def _fallback_assistant_mask(
    token_ids: Sequence[int], tokenizer: Any
) -> tuple[bool, ...]:
    role_id = getattr(tokenizer, "role_id", None)
    if not callable(role_id):
        raise TypeError("Explicit test-tokenizer fallback requires role_id()")
    assistant = int(role_id("assistant"))
    role_tokens = {int(role_id(role)) for role in ("user", "teacher", "assistant")}
    boundaries = set(role_tokens)
    for name in ("BOS", "EOS", "PAD", "bos_id", "eos_id", "pad_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            boundaries.add(value)
    eos = getattr(tokenizer, "EOS", getattr(tokenizer, "eos_id", None))
    mask = [False] * (len(token_ids) - 1)
    in_assistant = False
    for target_offset in range(1, len(token_ids)):
        target = int(token_ids[target_offset])
        if target in role_tokens:
            in_assistant = target == assistant
            mask[target_offset - 1] = in_assistant
        elif eos is not None and target == int(eos):
            mask[target_offset - 1] = in_assistant
            in_assistant = False
        elif target in boundaries:
            in_assistant = False
        elif in_assistant:
            mask[target_offset - 1] = True
    return tuple(mask)


def _scratch_assistant_mask(
    token_ids: Sequence[int], boundaries: Sequence[Any], tokenizer: Any
) -> tuple[bool, ...]:
    """Mirror the audited scratch SFT role/content/stop-token mask contract."""

    if (
        len(token_ids) < 2
        or token_ids[0] != tokenizer.bos_id
        or token_ids[-1] != tokenizer.eos_id
    ):
        raise ValueError("Verified dialogue is missing its BOS/EOS boundary")
    target_mask = [False] * len(token_ids)
    last_assistant_end: int | None = None
    expected_role_offset = 1
    assistant_count = 0
    for boundary in boundaries:
        role = getattr(boundary, "role", None)
        role_offset = int(getattr(boundary, "role_offset"))
        content_offset = int(getattr(boundary, "content_offset"))
        content_length = int(getattr(boundary, "content_length"))
        end_turn_offset = int(getattr(boundary, "end_turn_offset"))
        if (
            not isinstance(role, str)
            or role_offset != expected_role_offset
            or content_offset != role_offset + 1
            or content_length < 0
            or end_turn_offset != content_offset + content_length
            or end_turn_offset >= len(token_ids) - 1
        ):
            raise ValueError("Verified dialogue boundary lies outside its token record")
        if token_ids[role_offset] != tokenizer.role_id(role):
            raise ValueError("Verified dialogue role boundary does not match tokenizer")
        if token_ids[end_turn_offset] != tokenizer.end_turn_id:
            raise ValueError(
                "Verified dialogue end-turn boundary does not match tokenizer"
            )
        if role == "assistant":
            assistant_count += 1
            for position in range(role_offset, end_turn_offset + 1):
                target_mask[position] = True
            last_assistant_end = end_turn_offset
        expected_role_offset = end_turn_offset + 1
    if expected_role_offset != len(token_ids) - 1:
        raise ValueError("Verified dialogue boundaries do not cover the token record")
    if assistant_count != 1:
        raise ValueError("Verified dialogue must contain exactly one assistant answer")
    if last_assistant_end == len(token_ids) - 2:
        target_mask[-1] = True
    return tuple(target_mask[1:])


def encode_masked_verified_lesson(
    record: Any,
    tokenizer: Any,
    *,
    corrected: bool,
    max_length: int,
    allow_test_tokenizer_fallback: bool = False,
) -> MaskedLessonSequence:
    """Encode one lesson and select only assistant answer-content targets."""

    if not _is_trusted_human_dialogue(record):
        render_verified_lesson(record)
    _, problem, answer, review = _training_lesson_content(record)
    turns = [("user", problem)]
    if corrected:
        turns.append(("teacher", review))
    turns.append(("assistant", answer))
    encode_boundaries = getattr(tokenizer, "encode_dialogue_with_boundaries", None)
    if callable(encode_boundaries):
        token_ids, boundaries = encode_boundaries(turns, bos=True, eos=True)
        if len(token_ids) > max_length:
            turns = _compact_dialogue_turns(turns, tokenizer, max_length)
            token_ids, boundaries = encode_boundaries(turns, bos=True, eos=True)
        return MaskedLessonSequence(
            token_ids=tuple(int(token) for token in token_ids),
            loss_mask=_scratch_assistant_mask(token_ids, boundaries, tokenizer),
            boundaries=tuple(boundaries),
        )
    if not allow_test_tokenizer_fallback:
        raise TypeError(
            "Floaty training requires ScratchTokenizer.encode_dialogue_with_boundaries(); "
            "the legacy fallback is test-only and must be explicitly enabled"
        )
    encode_dialogue = getattr(tokenizer, "encode_dialogue", None)
    if not callable(encode_dialogue):
        raise TypeError("Explicit test-tokenizer fallback requires encode_dialogue()")
    token_ids = [int(token) for token in encode_dialogue(turns, bos=True, eos=True)]
    if len(token_ids) > max_length:
        raise ValueError(
            "Test tokenizer cannot safely compact a long verified dialogue"
        )
    return MaskedLessonSequence(
        token_ids=tuple(token_ids),
        loss_mask=_fallback_assistant_mask(token_ids, tokenizer),
        boundaries=(),
    )


def _encode_lessons(
    records: Sequence[Any],
    tokenizer: Any,
    context_length: int,
    *,
    allow_test_tokenizer_fallback: bool,
) -> tuple[list[MaskedLessonSequence], str, list[int | str]]:
    sequences: list[MaskedLessonSequence] = []
    digest = hashlib.sha256()
    identifiers: list[int | str] = []
    for record in records:
        rendered, _, _, _ = _training_lesson_content(record)
        direct = encode_masked_verified_lesson(
            record,
            tokenizer,
            corrected=False,
            max_length=context_length + 1,
            allow_test_tokenizer_fallback=allow_test_tokenizer_fallback,
        )
        sequences_for_record = [direct]
        if not _is_trusted_human_dialogue(record):
            sequences_for_record.append(
                encode_masked_verified_lesson(
                    record,
                    tokenizer,
                    corrected=True,
                    max_length=context_length + 1,
                    allow_test_tokenizer_fallback=allow_test_tokenizer_fallback,
                )
            )
        for sequence in sequences_for_record:
            sequences.append(sequence)
            values = np.asarray(sequence.token_ids, dtype=np.int32)
            masks = np.asarray(sequence.loss_mask, dtype=np.uint8)
            digest.update(len(sequence.token_ids).to_bytes(8, "big"))
            digest.update(values.tobytes())
            digest.update(masks.tobytes())
        digest.update(hashlib.sha256(rendered.encode("utf-8")).digest())
        identifier = _lesson_identifier(record)
        if identifier is not None:
            identifiers.append(identifier)
    return sequences, digest.hexdigest(), list(dict.fromkeys(identifiers))


def _pack_sequence(
    sequence: MaskedLessonSequence, context_length: int
) -> tuple[np.ndarray, np.ndarray]:
    target_length = context_length + 1
    repeats = math.ceil(target_length / len(sequence.token_ids))
    tokens = (list(sequence.token_ids) * repeats)[:target_length]
    target_flags = [False, *sequence.loss_mask]
    repeated_flags = (target_flags * repeats)[:target_length]
    return (
        np.asarray(tokens, dtype=np.int32),
        np.asarray(repeated_flags[1:], dtype=np.float32),
    )


def make_verified_lesson_batches(
    lessons: Sequence[Any],
    tokenizer: Any,
    *,
    batch_size: int,
    context_length: int,
    steps: int,
    seed: int,
    allow_test_tokenizer_fallback: bool = False,
) -> tuple[list[MaskedLessonBatch], str, list[int | str]]:
    """Validate, encode, and deterministically batch conversational lessons."""

    records = list(lessons)
    if not records:
        raise ValueError("At least one verified lesson is required")
    if batch_size <= 0 or context_length < 2 or steps <= 0:
        raise ValueError("Batch size, context length, and steps are invalid")
    sequences, digest, identifiers = _encode_lessons(
        records,
        tokenizer,
        context_length,
        allow_test_tokenizer_fallback=allow_test_tokenizer_fallback,
    )
    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    cursor = 0
    batches: list[MaskedLessonBatch] = []
    for _ in range(steps):
        rows: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for _ in range(batch_size):
            row, mask = _pack_sequence(
                sequences[order[cursor % len(order)]], context_length
            )
            rows.append(row)
            masks.append(mask)
            cursor += 1
        batches.append(
            MaskedLessonBatch(tokens=np.stack(rows), loss_mask=np.stack(masks))
        )
    return batches, digest, identifiers


def _validate_packed_batches(
    batches: Sequence[MaskedLessonBatch], context_length: int, label: str
) -> list[MaskedLessonBatch]:
    result: list[MaskedLessonBatch] = []
    for batch in batches:
        if not isinstance(batch, MaskedLessonBatch):
            raise TypeError(f"{label} requires MaskedLessonBatch values")
        if batch.tokens.shape[0] <= 0 or batch.tokens.shape[1] != context_length + 1:
            raise ValueError(
                f"{label} batches must have shape [batch, context_length + 1]"
            )
        result.append(batch)
    if not result:
        raise ValueError(f"{label} batches cannot be empty")
    return result


def _masked_next_token_loss(
    model: nn.Module,
    inputs: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
) -> mx.array:
    token_losses = nn.losses.cross_entropy(model(inputs), targets, reduction="none")
    mask = loss_mask.astype(mx.float32)
    denominator = mx.sum(mask)
    return mx.sum(token_losses.astype(mx.float32) * mask) / denominator


def _mean_loss(model: nn.Module, batches: Sequence[MaskedLessonBatch]) -> float:
    model.eval()
    values: list[float] = []
    for batch in batches:
        packed = mx.array(batch.tokens)
        loss = _masked_next_token_loss(
            model,
            packed[:, :-1],
            packed[:, 1:],
            mx.array(batch.loss_mask),
        )
        mx.eval(loss)
        values.append(float(loss.item()))
    return sum(values) / len(values)


def _tree_is_finite(tree: Any) -> bool:
    checks = [
        mx.all(mx.isfinite(value))
        for _, value in tree_flatten(tree)
        if isinstance(value, mx.array)
    ]
    if not checks:
        return True
    mx.eval(checks)
    return all(bool(value.item()) for value in checks)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    _fsync_file(path)


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:12]}"


def run_floaty_candidate(
    model: nn.Module,
    lessons: Sequence[Any],
    validation_lessons: Sequence[Any],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    expected_foundation_hash: str,
    expected_config_hash: str,
    expected_tokenizer_sha256: str | None = None,
    expected_floaty_parameter_names: Sequence[str] | None = None,
    protected_batches: Sequence[MaskedLessonBatch] | None = None,
    allow_test_tokenizer_fallback: bool = False,
    config: FloatyTrainingConfig | None = None,
) -> FloatyCandidateResult:
    """Train an isolated floaty candidate and roll it back when its gate fails.

    ``expected_foundation_hash`` and ``expected_config_hash`` must come from a
    trusted active-model manifest, rather than from an incoming contribution.
    Accepted artifacts contain floaty tensors only.  The caller remains
    responsible for activating or submitting the accepted artifact.
    """

    settings = config or FloatyTrainingConfig()
    settings.validate()
    training_recipe_sha256 = floaty_training_recipe_sha256(settings)
    model_config = getattr(model, "config", None)
    model_context = getattr(model_config, "context_length", None)
    if isinstance(model_context, int) and settings.context_length > model_context:
        raise ValueError("Floaty training context exceeds the foundation context")
    model_vocab = getattr(model_config, "vocab_size", None)
    tokenizer_vocab = getattr(tokenizer, "vocab_size", None)
    if (
        isinstance(model_vocab, int)
        and isinstance(tokenizer_vocab, int)
        and model_vocab != tokenizer_vocab
    ):
        raise ValueError("Tokenizer and foundation vocabulary sizes differ")
    tokenizer_sha256: str | None = None
    if expected_tokenizer_sha256 is not None:
        expected_tokenizer_sha256 = _sha256(
            expected_tokenizer_sha256,
            "expected tokenizer hash",
        )
        metadata = getattr(tokenizer, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise FloatyBoundaryError("Pinned tokenizer must expose signed metadata")
        tokenizer_sha256 = metadata.get("tokenizer_sha256")
        if tokenizer_sha256 != expected_tokenizer_sha256:
            raise FloatyBoundaryError(
                "Tokenizer hash does not match the pinned contract"
            )
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    training_records = list(lessons)
    validation_records = list(validation_lessons)
    if len(training_records) < settings.minimum_training_lessons:
        raise ValueError(
            f"At least {settings.minimum_training_lessons} verified training lessons are required"
        )
    if len(validation_records) < settings.minimum_validation_lessons:
        raise ValueError(
            f"At least {settings.minimum_validation_lessons} verified validation lessons are required"
        )
    overlap = {_lesson_key(record) for record in training_records} & {
        _lesson_key(record) for record in validation_records
    }
    if overlap:
        raise ValueError("Training and validation lessons must be independent")

    boundary = FoundationFloatyTrainingBoundary(
        model,
        expected_foundation_hash=expected_foundation_hash,
        expected_config_hash=expected_config_hash,
        expected_floaty_parameter_names=expected_floaty_parameter_names,
    )
    training_batches, training_digest, training_ids = make_verified_lesson_batches(
        training_records,
        tokenizer,
        batch_size=settings.batch_size,
        context_length=settings.context_length,
        steps=settings.steps,
        seed=settings.seed,
        allow_test_tokenizer_fallback=allow_test_tokenizer_fallback,
    )
    validation_batches, validation_digest, validation_ids = (
        make_verified_lesson_batches(
            validation_records,
            tokenizer,
            batch_size=settings.batch_size,
            context_length=settings.context_length,
            steps=max(1, math.ceil(len(validation_records) / settings.batch_size)),
            seed=settings.seed + 1,
            allow_test_tokenizer_fallback=allow_test_tokenizer_fallback,
        )
    )
    protected = (
        _validate_packed_batches(
            protected_batches,
            settings.context_length,
            "Protected evaluation",
        )
        if protected_batches is not None
        else None
    )

    initial_floaty = tree_map(lambda value: value, boundary.floaty.parameters())
    mx.eval(initial_floaty, boundary.foundation.parameters())
    baseline_validation_loss = _mean_loss(model, validation_batches)
    baseline_protected_loss = (
        _mean_loss(model, protected) if protected is not None else None
    )
    if not math.isfinite(baseline_validation_loss) or (
        baseline_protected_loss is not None
        and not math.isfinite(baseline_protected_loss)
    ):
        raise FloatingPointError("Floaty candidate baseline evaluation is non-finite")
    optimizer = optim.AdamW(
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    optimizer.init(boundary.optimizer_parameters)
    mx.eval(optimizer.state)
    optimizer_state_names = boundary.assert_optimizer_state(optimizer.state)

    def loss_fn(
        current_floaty: nn.Module,
        inputs: mx.array,
        targets: mx.array,
        loss_mask: mx.array,
    ) -> mx.array:
        if current_floaty is not boundary.floaty:
            raise FloatyBoundaryError("Loss received an unexpected floaty module")
        return _masked_next_token_loss(model, inputs, targets, loss_mask)

    loss_and_grad = nn.value_and_grad(boundary.floaty, loss_fn)
    training_losses: list[float] = []
    gradient_names: tuple[str, ...] = ()
    accepted_updates = 0
    model.train()
    try:
        for batch in training_batches:
            packed = mx.array(batch.tokens)
            loss, gradients = loss_and_grad(
                boundary.floaty,
                packed[:, :-1],
                packed[:, 1:],
                mx.array(batch.loss_mask),
            )
            gradient_names = boundary.assert_gradient_whitelist(gradients)
            clipped, gradient_norm = optim.clip_grad_norm(gradients, settings.grad_clip)
            mx.eval(loss, gradients, clipped, gradient_norm)
            loss_value = float(loss.item())
            gradient_value = float(gradient_norm.item())
            if not math.isfinite(loss_value) or not math.isfinite(gradient_value):
                raise FloatingPointError(
                    "Floaty candidate produced a non-finite gradient"
                )
            previous_floaty = tree_map(
                lambda value: value, boundary.floaty.parameters()
            )
            previous_optimizer = tree_map(lambda value: value, optimizer.state)
            optimizer.update(boundary.floaty, clipped)
            mx.eval(boundary.floaty.parameters(), optimizer.state)
            boundary.assert_foundation_unchanged()
            boundary.assert_optimizer_state(optimizer.state)
            if not _tree_is_finite(boundary.floaty.parameters()) or not _tree_is_finite(
                optimizer.state
            ):
                boundary.floaty.update(previous_floaty)
                optimizer.state = previous_optimizer
                mx.eval(boundary.floaty.parameters(), optimizer.state)
                raise FloatingPointError(
                    "Floaty candidate produced a non-finite update"
                )
            training_losses.append(loss_value)
            accepted_updates += 1

        candidate_validation_loss = _mean_loss(model, validation_batches)
        candidate_protected_loss = (
            _mean_loss(model, protected) if protected is not None else None
        )
        boundary.assert_foundation_unchanged()
        if (
            _call_digest(model, "foundation_hash", "foundation hash")
            != boundary.foundation_hash
        ):
            raise FloatyBoundaryError("Foundation hash changed during local training")
        validation_improvement = baseline_validation_loss - candidate_validation_loss
        protected_regression = (
            candidate_protected_loss - baseline_protected_loss
            if candidate_protected_loss is not None
            and baseline_protected_loss is not None
            else None
        )
        finite_candidate = _tree_is_finite(boundary.floaty.parameters()) and all(
            math.isfinite(value)
            for value in (
                candidate_validation_loss,
                *(
                    [candidate_protected_loss]
                    if candidate_protected_loss is not None
                    else []
                ),
            )
        )
        improves_validation = (
            validation_improvement > settings.minimum_validation_improvement
        )
        preserves_protected = (
            protected_regression is None
            or protected_regression <= settings.maximum_protected_loss_regression
        )
        accepted = finite_candidate and improves_validation and preserves_protected
        if not accepted:
            boundary.floaty.update(initial_floaty)
            mx.eval(boundary.floaty.parameters())
        boundary.assert_foundation_unchanged()
    except Exception:
        boundary.floaty.update(initial_floaty)
        boundary.restore_foundation()
        model.freeze()
        boundary.floaty.unfreeze()
        model.freeze_foundation()
        mx.eval(model.parameters())
        raise

    model.freeze()
    boundary.floaty.unfreeze()
    model.freeze_foundation()
    model.assert_parameter_partition()
    mx.eval(model.parameters())
    run_id = _run_id()
    destination = output_root / f"floaty-candidate-{run_id}"
    weights_name = "floaty.safetensors"
    temporary: Path | None = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=".floaty-candidate-", dir=output_root))
        weights_path = temporary / weights_name
        provenance_path = temporary / "provenance.json"
        report_path = temporary / "report.json"
        floaty_sha256: str | None = None
        tensor_metadata = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in _parameter_dict(boundary.floaty).items()
        }
        if accepted:
            floaty_state = _parameter_dict(boundary.floaty)
            if set(floaty_state) != set(boundary.floaty_local_parameter_names):
                raise FloatyBoundaryError(
                    "Export attempted to include a non-whitelisted tensor"
                )
            mx.save_safetensors(
                str(weights_path),
                floaty_state,
                metadata={
                    "schema_version": str(FLOATY_TRAINING_SCHEMA_VERSION),
                    "run_id": run_id,
                    "foundation_sha256": boundary.foundation_hash,
                    "floaty_config_sha256": boundary.config_hash,
                },
            )
            _fsync_file(weights_path)
            floaty_sha256 = _file_sha256(weights_path)
        provenance: dict[str, Any] = {
            "schema_version": FLOATY_TRAINING_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "foundation_sha256": boundary.foundation_hash,
            "floaty_config_sha256": boundary.config_hash,
            "foundation_trainable": False,
            "gradient_scope": "floaty-only",
            "optimizer_scope": "floaty-only",
            "training_recipe_sha256": training_recipe_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "floaty_parameter_names": list(boundary.floaty_parameter_names),
            "floaty_tensors": tensor_metadata,
            "training_lessons_digest": training_digest,
            "validation_lessons_digest": validation_digest,
            "training_lesson_ids": training_ids,
            "validation_lesson_ids": validation_ids,
            "accepted_updates": accepted_updates,
            "floaty_weights": weights_name if accepted else None,
            "floaty_weights_sha256": floaty_sha256,
        }
        report: dict[str, Any] = {
            "schema_version": FLOATY_TRAINING_SCHEMA_VERSION,
            "run_id": run_id,
            "accepted": accepted,
            "rolled_back": not accepted,
            "gate": {
                "foundation_unchanged": True,
                "verified_lesson_batch": True,
                "independent_validation": True,
                "finite_candidate": finite_candidate,
                "improves_validation": improves_validation,
                "preserves_protected_loss": preserves_protected,
                "minimum_validation_improvement": settings.minimum_validation_improvement,
                "maximum_protected_loss_regression": settings.maximum_protected_loss_regression,
            },
            "metrics": {
                "baseline_validation_loss": baseline_validation_loss,
                "candidate_validation_loss": candidate_validation_loss,
                "validation_improvement": validation_improvement,
                "baseline_protected_loss": baseline_protected_loss,
                "candidate_protected_loss": candidate_protected_loss,
                "protected_loss_regression": protected_regression,
                "mean_training_loss": sum(training_losses) / len(training_losses),
                "final_training_loss": training_losses[-1],
            },
            "boundary": {
                "gradient_parameter_names": list(gradient_names),
                "optimizer_state_parameter_names": list(optimizer_state_names),
                "foundation_gradient_parameters": [],
                "foundation_optimizer_parameters": [],
            },
            "lessons": {
                "training": len(training_records),
                "validation": len(validation_records),
            },
            "training_config": asdict(settings),
            "training_recipe_sha256": training_recipe_sha256,
        }
        _write_json(provenance_path, provenance)
        _write_json(report_path, report)
        _fsync_directory(temporary)
        publish_directory_no_replace(temporary, destination)
        _fsync_directory(output_root)
    except Exception:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        boundary.floaty.update(initial_floaty)
        boundary.restore_foundation()
        model.freeze()
        boundary.floaty.unfreeze()
        model.freeze_foundation()
        mx.eval(model.parameters())
        raise

    return FloatyCandidateResult(
        run_id=run_id,
        accepted=accepted,
        rolled_back=not accepted,
        floaty_weights_path=(destination / weights_name) if accepted else None,
        provenance_path=destination / "provenance.json",
        report_path=destination / "report.json",
        report=report,
    )
