"""Guarded candidate training from verified conversational lessons.

This module never replaces the active checkpoint.  It trains and evaluates a
versioned candidate; a coordinator may hot-load it only when the report's gate
marks it as eligible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence
import unicodedata

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .apprenticeship import canonical_practice_key, verify_answer
from .data import ArithmeticCurriculum, make_batch, split_for_key
from .generate import complete_dialogue
from .model import ArithmeticLM, ModelConfig
from .tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class ContinualTrainingConfig:
    """Controls a short, replay-protected candidate training run."""

    steps: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    replay_ratio: float = 0.5
    max_curriculum_level: int = 6
    validation_fraction: float = 0.2
    evaluation_batches: int = 4
    evaluation_new_weight: float = 0.5
    min_new_loss_improvement: float = 1e-4
    max_old_loss_regression: float = 0.05
    context_length: int | None = None
    seed: int = 42

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if not 0.0 <= self.replay_ratio < 1.0:
            raise ValueError("replay_ratio must be in [0, 1)")
        if not 1 <= self.max_curriculum_level <= 6:
            raise ValueError("max_curriculum_level must be between 1 and 6")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.evaluation_batches <= 0:
            raise ValueError("evaluation_batches must be positive")
        if not 0.0 < self.evaluation_new_weight < 1.0:
            raise ValueError("evaluation_new_weight must be in (0, 1)")
        if self.min_new_loss_improvement < 0 or self.max_old_loss_regression < 0:
            raise ValueError("promotion thresholds cannot be negative")
        if self.context_length is not None and self.context_length < 2:
            raise ValueError("context_length must be at least 2")


@dataclass(frozen=True)
class CandidateTrainingResult:
    """Files and gate decision produced by :func:`run_candidate`."""

    run_id: str
    weights_path: Path
    report_path: Path
    promotion_eligible: bool
    trained_lesson_ids: tuple[int | str, ...]
    report: Mapping[str, Any]


def _field(record: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(record, Mapping) and name in record:
            return record[name]
        if hasattr(record, name):
            return getattr(record, name)
    metadata = (
        record.get("metadata", {})
        if isinstance(record, Mapping)
        else getattr(record, "metadata", {})
    )
    if isinstance(metadata, Mapping):
        for name in names:
            if name in metadata:
                return metadata[name]
    return default


def _one_line(value: Any) -> str:
    return _unicode_line(value)


def _unicode_line(value: Any) -> str:
    """Keep teacher language intact while removing sequence-breaking whitespace."""
    return " ".join(
        unicodedata.normalize("NFKC", str(value)).replace("\t", " ").split()
    )


def _printable_ascii(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return "".join(
        char
        if char in {"\n", "\t"} or not unicodedata.category(char).startswith("C")
        else " "
        for char in text
    )


def _is_false(value: Any) -> bool:
    return value is False or (
        isinstance(value, str)
        and value.lower() in {"false", "no", "rejected", "invalid"}
    )


def render_verified_lesson(record: Any) -> str:
    """Render one verified lesson as safe, printable-ASCII LM training text.

    The correct direct response is placed first.  An incorrect first attempt is
    retained only inside an explicitly labelled teacher review, so ordinary
    next-token training does not teach it as the desired answer.
    """

    problem = _field(record, "problem", "question", "prompt")
    answer = _field(record, "correct_answer", "verified_answer", "answer")
    verifier = _field(record, "verifier", "verified_by", default="")
    verified = _field(record, "verified", "is_verified", default=None)
    verification_status = _field(record, "verification_status", default="")
    if _is_false(verified) or _is_false(verification_status):
        raise ValueError("Rejected or unverified lesson cannot be trained")
    if (
        verified is not True
        and not verifier
        and str(verification_status).lower() not in {"verified", "approved", "correct"}
    ):
        raise ValueError("Lesson must include a verifier or explicit verified status")
    if (
        problem is None
        or answer is None
        or not str(problem).strip()
        or not str(answer).strip()
    ):
        raise ValueError("Verified lesson requires a problem and correct_answer")

    problem_text = _one_line(problem)
    answer_text = _one_line(answer)
    lines = [f"Teacher: {problem_text}", f"Student: {answer_text}"]
    skill = _one_line(_field(record, "skill", default=""))
    if skill:
        lines.append(f"Skill: {skill}")
    first_attempt = _one_line(
        _field(
            record, "first_attempt", "student_attempt", "assistant_answer", default=""
        )
    )
    feedback = _one_line(
        _field(record, "feedback", "teacher_feedback", "correction", "hint", default="")
    )
    if first_attempt and first_attempt != answer_text:
        lines.append(
            f"Teacher review: The earlier attempt '{first_attempt}' was incorrect."
        )
    if feedback:
        lines.append(f"Teacher feedback: {feedback}")
    lines.append(f"Verified by: {_one_line(verifier) or 'trusted verifier'}")
    return "\n".join(lines) + "\n"


def encode_verified_lesson(
    record: Any,
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> list[int]:
    """Encode the direct behaviour: user problem, then verified student answer."""
    return _encode_lesson_behaviour(
        record, tokenizer, corrected=False, max_length=max_length
    )


def encode_corrected_lesson(
    record: Any,
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> list[int]:
    """Encode apprenticeship behaviour: problem, teaching, then revised answer."""
    return _encode_lesson_behaviour(
        record, tokenizer, corrected=True, max_length=max_length
    )


def _lesson_content(record: Any) -> tuple[str, str, str, str]:
    rendered = render_verified_lesson(record)
    problem = _unicode_line(_field(record, "problem", "question", "prompt"))
    answer = _unicode_line(
        _field(record, "correct_answer", "verified_answer", "answer")
    )
    first_attempt = _unicode_line(
        _field(
            record, "first_attempt", "student_attempt", "assistant_answer", default=""
        )
    )
    feedback = _unicode_line(
        _field(record, "feedback", "teacher_feedback", "correction", "hint", default="")
    )
    skill = _unicode_line(_field(record, "skill", default=""))
    verifier = (
        _unicode_line(_field(record, "verifier", "verified_by", default=""))
        or "trusted verifier"
    )
    review: list[str] = []
    if first_attempt and first_attempt != answer:
        review.append(f"The earlier attempt {first_attempt!r} was incorrect.")
    if feedback:
        review.append(feedback)
    if skill:
        review.append(f"Skill: {skill}.")
    review.append(f"Verified by: {verifier}.")
    return rendered, problem, answer, " ".join(review)


def _encode_lesson_behaviour(
    record: Any,
    tokenizer: Any,
    *,
    corrected: bool,
    max_length: int | None,
) -> list[int]:
    rendered, problem, answer, review = _lesson_content(record)
    encode_dialogue = getattr(tokenizer, "encode_dialogue", None)
    role_id = getattr(tokenizer, "role_id", None)
    if not callable(encode_dialogue) or not callable(role_id):
        tokens = _token_ids(tokenizer, rendered)
        if max_length is not None and len(tokens) > max_length:
            raise ValueError("Tokenizer cannot safely compact a long verified lesson")
        return tokens
    turns = [("user", problem)]
    if corrected:
        turns.append(("teacher", review))
    turns.append(("assistant", answer))
    if max_length is None:
        return [int(token) for token in encode_dialogue(turns, bos=True, eos=True)]
    if max_length < 6:
        raise ValueError("Lesson context is too short for role-aware training")

    bos = int(getattr(tokenizer, "BOS"))
    eos = int(getattr(tokenizer, "EOS"))
    user_id = int(role_id("user"))
    teacher_id = int(role_id("teacher"))
    assistant_id = int(role_id("assistant"))
    problem_tokens = [int(token) for token in tokenizer.encode(problem)]
    review_tokens = (
        [int(token) for token in tokenizer.encode(review)] if corrected else []
    )
    answer_tokens = [int(token) for token in tokenizer.encode(answer)]
    fixed_size = 4 + len(answer_tokens) + (1 if corrected else 0)
    if fixed_size > max_length:
        raise ValueError("Verified answer is too long for the training context")
    body_budget = max_length - fixed_size
    if corrected and len(problem_tokens) + len(review_tokens) > body_budget:
        review_budget = (
            min(len(review_tokens), max(1, body_budget // 2)) if body_budget else 0
        )
        problem_budget = max(0, body_budget - review_budget)
    else:
        problem_budget = min(len(problem_tokens), body_budget)
        review_budget = min(len(review_tokens), body_budget - problem_budget)
    compact_problem = problem_tokens[-problem_budget:] if problem_budget else []
    compact_review = review_tokens[-review_budget:] if review_budget else []
    tokens = [bos, user_id, *compact_problem]
    if corrected:
        tokens.extend([teacher_id, *compact_review])
    tokens.extend([assistant_id, *answer_tokens, eos])
    return tokens


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    """Use the installed tokenizer API without assuming one concrete class."""

    try:
        encoded = tokenizer.encode(text, bos=True, eos=True)
    except TypeError:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=True)
        except TypeError:
            encoded = tokenizer.encode(text)
            bos = getattr(tokenizer, "BOS", getattr(tokenizer, "bos_token_id", None))
            eos = getattr(tokenizer, "EOS", getattr(tokenizer, "eos_token_id", None))
            encoded = (
                ([bos] if bos is not None else [])
                + list(encoded)
                + ([eos] if eos is not None else [])
            )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    ids = [int(token) for token in encoded]
    if not ids:
        raise ValueError("Tokenizer produced an empty lesson")
    return ids


def make_packed_lesson_batch(
    lesson_texts: Sequence[str],
    tokenizer: Any,
    batch_size: int,
    context_length: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Pack lesson texts into deterministic, fixed-length next-token batches."""

    if not lesson_texts:
        raise ValueError("At least one lesson text is required")
    if batch_size <= 0 or context_length < 2:
        raise ValueError("batch_size must be positive and context_length at least 2")
    sequences = [_token_ids(tokenizer, _printable_ascii(text)) for text in lesson_texts]
    return _make_packed_token_batch(sequences, batch_size, context_length, seed=seed)


def _make_packed_token_batch(
    sequences: Sequence[Sequence[int]],
    batch_size: int,
    context_length: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    if not sequences:
        raise ValueError("At least one lesson sequence is required")
    rng = random.Random(seed)
    target_length = context_length + 1
    batch = np.empty((batch_size, target_length), dtype=np.int32)
    for row in range(batch_size):
        packed: list[int] = []
        while len(packed) < target_length:
            packed.extend(
                int(token) for token in sequences[rng.randrange(len(sequences))]
            )
        batch[row] = packed[:target_length]
    return batch


def _make_selected_token_batch(
    sequences: Sequence[Sequence[int]],
    context_length: int,
) -> np.ndarray:
    """Pack one explicitly selected lesson per row for coverage accounting."""
    if not sequences:
        raise ValueError("At least one selected lesson sequence is required")
    target_length = context_length + 1
    batch = np.empty((len(sequences), target_length), dtype=np.int32)
    for row, sequence in enumerate(sequences):
        packed: list[int] = []
        while len(packed) < target_length:
            packed.extend(int(token) for token in sequence)
        batch[row] = packed[:target_length]
    return batch


def _lm_loss(model: ArithmeticLM, inputs: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.cross_entropy(model(inputs), targets, reduction="mean")


def _mean_loss(model: ArithmeticLM, batches: Sequence[np.ndarray]) -> float:
    model.eval()
    values: list[float] = []
    for batch in batches:
        packed = mx.array(batch)
        loss = _lm_loss(model, packed[:, :-1], packed[:, 1:])
        mx.eval(loss)
        values.append(float(loss.item()))
    model.train()
    return sum(values) / len(values)


def _assistant_mask(batch: np.ndarray, tokenizer: Any) -> np.ndarray:
    role_id = getattr(tokenizer, "role_id", None)
    if not callable(role_id):
        raise ValueError("Assistant-answer evaluation requires a role-aware tokenizer")
    assistant = int(role_id("assistant"))
    boundaries = {
        int(role_id(role)) for role in ("user", "teacher", "memory", "lesson")
    }
    boundaries.update(
        int(token)
        for token in (
            getattr(tokenizer, "BOS", -1),
            getattr(tokenizer, "EOS", -1),
            getattr(tokenizer, "PAD", -1),
        )
    )
    mask = np.zeros((batch.shape[0], batch.shape[1] - 1), dtype=np.float32)
    for row_index, row in enumerate(batch):
        active = False
        for position in range(row.shape[0] - 1):
            current, target = int(row[position]), int(row[position + 1])
            if current == assistant:
                active = True
            elif current in boundaries:
                active = False
            if active and target not in boundaries and target != assistant:
                mask[row_index, position] = 1.0
    if not mask.any():
        raise ValueError("Evaluation batch contains no assistant answer tokens")
    return mask


def _mean_answer_loss(
    model: ArithmeticLM, batches: Sequence[np.ndarray], tokenizer: Any
) -> float:
    model.eval()
    values: list[float] = []
    for batch in batches:
        packed = mx.array(batch)
        token_losses = nn.losses.cross_entropy(
            model(packed[:, :-1]),
            packed[:, 1:],
            reduction="none",
        )
        mask = mx.array(_assistant_mask(batch, tokenizer))
        loss = mx.sum(token_losses * mask) / mx.sum(mask)
        mx.eval(loss)
        values.append(float(loss.item()))
    model.train()
    return sum(values) / len(values)


def _exact_answer_accuracy(
    model: ArithmeticLM,
    records: Sequence[Any],
    tokenizer: Any,
    *,
    max_examples: int = 8,
    corrected: bool = False,
) -> float:
    sample = list(records[:max_examples])
    if not sample:
        raise ValueError("Exact-answer evaluation requires held-out lessons")
    model.eval()
    correct = 0
    for record in sample:
        _, problem, _, review = _lesson_content(record)
        expected = str(_field(record, "correct_answer", "verified_answer", "answer"))
        max_tokens = min(64, max(8, len(expected.encode("utf-8")) + 16))
        turns = [("user", problem)]
        if corrected:
            turns.append(("teacher", review))
        generated = complete_dialogue(
            model,
            tokenizer,
            turns,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        if verify_answer(problem, generated).is_correct:
            correct += 1
    model.train()
    return correct / len(sample)


def _lesson_id(record: Any) -> int | str | None:
    value = _field(record, "id", "lesson_id", default=None)
    return value if isinstance(value, (int, str)) else None


def _split_lessons(
    records: Sequence[Any], validation_fraction: float
) -> tuple[list[tuple[Any, str]], list[tuple[Any, str]], bool]:
    rendered = [(record, render_verified_lesson(record)) for record in records]
    groups: dict[str, list[tuple[Any, str]]] = {}
    for record, text in rendered:
        canonical = "\0".join(
            (
                _unicode_line(
                    _field(record, "problem", "question", "prompt")
                ).casefold(),
                _unicode_line(
                    _field(record, "correct_answer", "verified_answer", "answer")
                ),
            )
        )
        groups.setdefault(hashlib.sha256(canonical.encode()).hexdigest(), []).append(
            (record, text)
        )
    ordered_groups = [groups[key] for key in sorted(groups)]
    if len(ordered_groups) == 1:
        return rendered, rendered, True
    validation_count = max(
        1,
        min(
            len(ordered_groups) - 1,
            math.ceil(len(ordered_groups) * validation_fraction),
        ),
    )
    held_out = [pair for group in ordered_groups[:validation_count] for pair in group]
    training = [pair for group in ordered_groups[validation_count:] for pair in group]
    return training, held_out, False


def _practice_records(record: Any) -> list[dict[str, Any]]:
    variants = _field(record, "practice_variants", default=())
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        return []
    results: list[dict[str, Any]] = []
    for variant in variants:
        if isinstance(variant, Mapping):
            problem = _field(variant, "problem", "question", "prompt")
            answer = _field(variant, "correct_answer", "answer")
            metadata = _field(variant, "metadata", default={})
        elif (
            isinstance(variant, Sequence)
            and not isinstance(variant, (str, bytes))
            and len(variant) >= 2
        ):
            problem, answer = variant[0], variant[1]
            metadata = {}
        else:
            continue
        if not str(problem or "").strip() or not str(answer or "").strip():
            continue
        if isinstance(metadata, Mapping) and metadata.get("generated") is True:
            derived_key = canonical_practice_key(str(problem))
            recorded_key = metadata.get("canonical_key")
            if (
                derived_key is None
                or (recorded_key is not None and str(recorded_key) != derived_key)
                or split_for_key(derived_key) != "train"
            ):
                continue
        results.append(
            {
                "problem": problem,
                "first_attempt": "",
                "correct_answer": answer,
                "feedback": _field(record, "feedback", default=""),
                "skill": _field(record, "skill", default="arithmetic"),
                "source": "generated-practice",
                "verifier": _field(record, "verifier", default="exact-arithmetic-v1"),
                "status": "pending",
                "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
            }
        )
    return results


def _load_model(model_config: ModelConfig, weights_path: Path) -> ArithmeticLM:
    model = ArithmeticLM(model_config)
    model.load_weights(str(weights_path))
    mx.eval(model.parameters())
    return model


def _unique_run_dir(output_dir: Path, digest: str) -> tuple[str, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = f"{stamp}-{digest[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        run_id = base if suffix == 0 else f"{base}-{suffix}"
        run_dir = output_dir / f"candidate-{run_id}"
        try:
            run_dir.mkdir()
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique candidate directory")


def run_candidate(
    active_weights_path: str | Path,
    model_config: ModelConfig | str | Path,
    lessons: Sequence[Any],
    output_dir: str | Path,
    *,
    tokenizer: Any | None = None,
    validation_lessons: Sequence[Any] | None = None,
    config: ContinualTrainingConfig | None = None,
) -> CandidateTrainingResult:
    """Train, evaluate, save, and report a replay-protected candidate.

    ``active_weights_path`` is read only.  Promotion is merely recommended in
    the returned result; replacing or hot-reloading the active model remains a
    coordinator responsibility.
    """

    settings = config or ContinualTrainingConfig()
    settings.validate()
    active_path = Path(active_weights_path).expanduser().resolve()
    if not active_path.is_file():
        raise FileNotFoundError(active_path)
    resolved_model_config = (
        model_config
        if isinstance(model_config, ModelConfig)
        else ModelConfig.from_json(model_config)
    )
    context = settings.context_length or resolved_model_config.context_length
    if context > resolved_model_config.context_length:
        raise ValueError("Training context cannot exceed the model context length")
    active_tokenizer = tokenizer or ArithmeticTokenizer()
    vocab_size = getattr(active_tokenizer, "vocab_size", None)
    if vocab_size is not None and int(vocab_size) != resolved_model_config.vocab_size:
        raise ValueError("Tokenizer and model vocabulary sizes do not match")
    records = list(lessons)
    if not records:
        raise ValueError("At least one verified lesson is required")

    primary_train_pairs, held_out_pairs, validation_overlap = _split_lessons(
        records, settings.validation_fraction
    )
    if validation_lessons is not None:
        explicit_validation = list(validation_lessons)
        if not explicit_validation:
            raise ValueError("validation_lessons cannot be empty")
        held_out_pairs = [
            (record, render_verified_lesson(record)) for record in explicit_validation
        ]
        primary_train_pairs = [
            (record, render_verified_lesson(record)) for record in records
        ]
        train_keys = {
            (
                _unicode_line(
                    _field(record, "problem", "question", "prompt")
                ).casefold(),
                _unicode_line(
                    _field(record, "correct_answer", "verified_answer", "answer")
                ),
            )
            for record, _ in primary_train_pairs
        }
        validation_keys = {
            (
                _unicode_line(
                    _field(record, "problem", "question", "prompt")
                ).casefold(),
                _unicode_line(
                    _field(record, "correct_answer", "verified_answer", "answer")
                ),
            )
            for record, _ in held_out_pairs
        }
        validation_overlap = bool(train_keys & validation_keys)
    train_pairs = list(primary_train_pairs)
    for parent, _ in primary_train_pairs:
        train_pairs.extend(
            (practice, render_verified_lesson(practice))
            for practice in _practice_records(parent)
        )
    train_texts = [text for _, text in train_pairs]
    primary_sequences = [
        sequence
        for record, _ in primary_train_pairs
        for sequence in (
            encode_verified_lesson(record, active_tokenizer, max_length=context + 1),
            encode_corrected_lesson(record, active_tokenizer, max_length=context + 1),
        )
    ]
    practice_sequences = [
        sequence
        for record, _ in train_pairs[len(primary_train_pairs) :]
        for sequence in (
            encode_verified_lesson(record, active_tokenizer, max_length=context + 1),
            encode_corrected_lesson(record, active_tokenizer, max_length=context + 1),
        )
    ]
    train_sequences = [*primary_sequences, *practice_sequences]
    validation_sequences = [
        sequence
        for record, _ in held_out_pairs
        for sequence in (
            encode_verified_lesson(record, active_tokenizer, max_length=context + 1),
            encode_corrected_lesson(record, active_tokenizer, max_length=context + 1),
        )
    ]

    new_eval_batches = [
        _make_packed_token_batch(
            validation_sequences,
            settings.batch_size,
            context,
            seed=settings.seed + 10_000 + index,
        )
        for index in range(settings.evaluation_batches)
    ]
    old_validation = ArithmeticCurriculum(
        seed=settings.seed + 20_000, split="validation"
    )
    old_eval_batches = [
        make_batch(
            old_validation,
            active_tokenizer,
            settings.batch_size,
            context,
            settings.max_curriculum_level,
        )
        for _ in range(settings.evaluation_batches)
    ]

    mx.random.seed(settings.seed)
    candidate = _load_model(resolved_model_config, active_path)
    held_out_records = [record for record, _ in held_out_pairs]
    baseline_new_loss = _mean_answer_loss(candidate, new_eval_batches, active_tokenizer)
    baseline_old_loss = _mean_loss(candidate, old_eval_batches)
    baseline_direct_accuracy = _exact_answer_accuracy(
        candidate, held_out_records, active_tokenizer
    )
    baseline_corrected_accuracy = _exact_answer_accuracy(
        candidate,
        held_out_records,
        active_tokenizer,
        corrected=True,
    )
    baseline_exact_accuracy = (
        baseline_direct_accuracy + baseline_corrected_accuracy
    ) / 2

    optimizer = optim.AdamW(
        learning_rate=settings.learning_rate, weight_decay=settings.weight_decay
    )
    loss_and_grad = nn.value_and_grad(candidate, _lm_loss)
    replay = ArithmeticCurriculum(seed=settings.seed + 30_000, split="train")
    schedule_rng = random.Random(settings.seed + 40_000)
    primary_sequence_count = len(primary_sequences)
    practice_sequence_count = len(train_sequences) - primary_sequence_count
    coverage_target = primary_sequence_count + min(
        primary_sequence_count, practice_sequence_count
    )
    effective_steps = settings.steps
    while True:
        total_rows = effective_steps * settings.batch_size
        new_rows_total = max(1, round(total_rows * (1.0 - settings.replay_ratio)))
        if settings.replay_ratio > 0.0 and total_rows > 1:
            new_rows_total = min(total_rows - 1, new_rows_total)
        if new_rows_total >= coverage_target:
            break
        effective_steps += 1
    schedule = [True] * new_rows_total + [False] * (total_rows - new_rows_total)
    schedule_rng.shuffle(schedule)
    primary_order = list(range(primary_sequence_count))
    practice_order = list(range(primary_sequence_count, len(train_sequences)))
    schedule_rng.shuffle(primary_order)
    schedule_rng.shuffle(practice_order)
    sequence_order = [*primary_order, *practice_order]
    new_sequence_cursor = 0
    training_losses: list[float] = []

    for step in range(effective_steps):
        flags = schedule[step * settings.batch_size : (step + 1) * settings.batch_size]
        new_count = sum(flags)
        replay_count = settings.batch_size - new_count
        parts: list[np.ndarray] = []
        if new_count:
            selected: list[Sequence[int]] = []
            for _ in range(new_count):
                selected.append(
                    train_sequences[
                        sequence_order[new_sequence_cursor % len(sequence_order)]
                    ]
                )
                new_sequence_cursor += 1
            parts.append(_make_selected_token_batch(selected, context))
        if replay_count:
            parts.append(
                make_batch(
                    replay,
                    active_tokenizer,
                    replay_count,
                    context,
                    settings.max_curriculum_level,
                )
            )
        packed_np = np.concatenate(parts, axis=0)
        order = list(range(settings.batch_size))
        schedule_rng.shuffle(order)
        packed = mx.array(packed_np[order])
        loss, gradients = loss_and_grad(candidate, packed[:, :-1], packed[:, 1:])
        gradients, gradient_norm = optim.clip_grad_norm(gradients, settings.grad_clip)
        optimizer.update(candidate, gradients)
        mx.eval(loss, gradient_norm, candidate.parameters(), optimizer.state)
        training_losses.append(float(loss.item()))

    candidate_new_loss = _mean_answer_loss(
        candidate, new_eval_batches, active_tokenizer
    )
    candidate_old_loss = _mean_loss(candidate, old_eval_batches)
    candidate_direct_accuracy = _exact_answer_accuracy(
        candidate, held_out_records, active_tokenizer
    )
    candidate_corrected_accuracy = _exact_answer_accuracy(
        candidate,
        held_out_records,
        active_tokenizer,
        corrected=True,
    )
    candidate_exact_accuracy = (
        candidate_direct_accuracy + candidate_corrected_accuracy
    ) / 2
    new_improvement = baseline_new_loss - candidate_new_loss
    old_regression = candidate_old_loss - baseline_old_loss
    improves_new = new_improvement > settings.min_new_loss_improvement
    preserves_old = old_regression <= settings.max_old_loss_regression
    preserves_exact = (
        candidate_direct_accuracy >= baseline_direct_accuracy
        and candidate_corrected_accuracy >= baseline_corrected_accuracy
    )
    independent_validation = not validation_overlap
    new_weight = settings.evaluation_new_weight
    baseline_mixture_loss = (
        new_weight * baseline_new_loss + (1.0 - new_weight) * baseline_old_loss
    )
    candidate_mixture_loss = (
        new_weight * candidate_new_loss + (1.0 - new_weight) * candidate_old_loss
    )
    preserves_mixture = candidate_mixture_loss <= baseline_mixture_loss
    promotion_eligible = (
        improves_new
        and preserves_old
        and preserves_exact
        and independent_validation
        and preserves_mixture
    )

    digest = hashlib.sha256("\n".join(train_texts).encode()).hexdigest()
    run_id, run_dir = _unique_run_dir(Path(output_dir).expanduser().resolve(), digest)
    weights_path = run_dir / f"candidate-{run_id}.safetensors"
    if weights_path.resolve() == active_path:
        raise RuntimeError("Candidate path unexpectedly resolves to active weights")
    candidate.save_weights(str(weights_path))
    report_path = run_dir / "report.json"
    trained_ids = tuple(
        dict.fromkeys(
            identifier
            for record, _ in primary_train_pairs
            if (identifier := _lesson_id(record)) is not None
        )
    )
    validation_ids = [
        identifier
        for record, _ in held_out_pairs
        if (identifier := _lesson_id(record)) is not None
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_weights": str(active_path),
        "candidate_weights": str(weights_path),
        "promotion_eligible": promotion_eligible,
        "gate": {
            "improves_new": improves_new,
            "preserves_old": preserves_old,
            "preserves_exact_accuracy": preserves_exact,
            "independent_validation": independent_validation,
            "preserves_mixture_loss": preserves_mixture,
            "minimum_new_loss_improvement": settings.min_new_loss_improvement,
            "maximum_old_loss_regression": settings.max_old_loss_regression,
        },
        "metrics": {
            "baseline": {
                "new_loss": baseline_new_loss,
                "old_loss": baseline_old_loss,
                "exact_accuracy": baseline_exact_accuracy,
                "direct_exact_accuracy": baseline_direct_accuracy,
                "corrected_exact_accuracy": baseline_corrected_accuracy,
                "mixture_loss": baseline_mixture_loss,
            },
            "candidate": {
                "new_loss": candidate_new_loss,
                "old_loss": candidate_old_loss,
                "exact_accuracy": candidate_exact_accuracy,
                "direct_exact_accuracy": candidate_direct_accuracy,
                "corrected_exact_accuracy": candidate_corrected_accuracy,
                "mixture_loss": candidate_mixture_loss,
            },
            "new_loss_improvement": new_improvement,
            "old_loss_regression": old_regression,
            "mean_training_loss": sum(training_losses) / len(training_losses),
            "final_training_loss": training_losses[-1],
        },
        "lessons": {
            "provided": len(records),
            "trained": len(primary_train_pairs),
            "practice_variants": len(train_pairs) - len(primary_train_pairs),
            "validated": len(held_out_pairs),
            "validation_overlaps_training": validation_overlap,
            "trained_ids": list(trained_ids),
            "validation_ids": validation_ids,
        },
        "training_rows": {
            "new_lessons": new_rows_total,
            "synthetic_replay": total_rows - new_rows_total,
            "requested_steps": settings.steps,
            "effective_steps": effective_steps,
            "all_primary_lessons_covered": new_sequence_cursor
            >= primary_sequence_count,
            "practice_rows_sampled": min(
                practice_sequence_count,
                max(0, new_sequence_cursor - primary_sequence_count),
            ),
        },
        "model_config": resolved_model_config.to_dict(),
        "training_config": asdict(settings),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return CandidateTrainingResult(
        run_id=run_id,
        weights_path=weights_path,
        report_path=report_path,
        promotion_eligible=promotion_eligible,
        trained_lesson_ids=trained_ids,
        report=report,
    )
