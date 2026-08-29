"""Deterministic staged scratch training for :class:`HybridLM`.

This trainer accepts only random initialization or one of its own complete
scratch-training checkpoints. It deliberately has no option for importing an
external model checkpoint. The default optimizer is conventional AdamW;
Muon, multi-token prediction, reinforcement learning, and other experimental
methods belong in later controlled ablations, not the baseline run.

Samples are never concatenated into one attention stream. Every pretraining
document chunk or SFT conversation occupies its own padded batch row, giving
the model an unambiguous BOS/EOS boundary without requiring a block mask.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
from typing import Any, Literal
import uuid

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from .hybrid import HybridConfig, HybridLM
from .scratch_tokenizer import VOCAB_SIZE, ScratchTokenizer


SCHEMA_VERSION = 1
Objective = Literal["pretrain", "sft"]
DataFormat = Literal["audited_jsonl", "token_shards"]


@dataclass(frozen=True)
class OptimizerConfig:
    """Stable AdamW baseline settings for scratch training."""

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bias_correction: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("AdamW epsilon must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")


@dataclass(frozen=True)
class GuardConfig:
    """Pre- and post-update numerical safety limits."""

    loss_spike_factor: float = 4.0
    loss_spike_warmup_updates: int = 10
    ema_decay: float = 0.98
    max_consecutive_bad_updates: int = 8

    def __post_init__(self) -> None:
        if self.loss_spike_factor <= 1.0:
            raise ValueError("loss_spike_factor must exceed 1")
        if self.loss_spike_warmup_updates < 0:
            raise ValueError("loss_spike_warmup_updates must be non-negative")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.max_consecutive_bad_updates <= 0:
            raise ValueError("max_consecutive_bad_updates must be positive")


@dataclass(frozen=True)
class StageConfig:
    """One deterministic pretraining or conversational SFT stage."""

    name: str
    objective: Objective
    train_path: str
    eval_path: str
    steps: int
    data_format: DataFormat = "audited_jsonl"
    batch_size: int = 4
    sequence_length: int = 512
    peak_learning_rate: float = 3e-4
    warmup_steps: int = 100
    min_learning_rate_ratio: float = 0.1
    eval_every: int = 100
    eval_batches: int = 8
    save_every: int = 100
    report_every: int = 10

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stage name must not be empty")
        if self.objective not in {"pretrain", "sft"}:
            raise ValueError("objective must be 'pretrain' or 'sft'")
        if self.data_format not in {"audited_jsonl", "token_shards"}:
            raise ValueError("data_format must be 'audited_jsonl' or 'token_shards'")
        if not self.train_path or not self.eval_path:
            raise ValueError("each stage requires train_path and eval_path")
        if (
            self.data_format == "audited_jsonl"
            and Path(self.train_path).resolve() == Path(self.eval_path).resolve()
        ):
            raise ValueError(
                "train_path and eval_path must be distinct held-out sources"
            )
        if self.steps <= 0 or self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("steps, batch_size, and sequence_length must be positive")
        if self.peak_learning_rate <= 0.0:
            raise ValueError("peak_learning_rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be smaller than stage steps")
        if not 0.0 < self.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be in (0, 1]")
        if (
            min(self.eval_every, self.eval_batches, self.save_every, self.report_every)
            <= 0
        ):
            raise ValueError(
                "evaluation, saving, and reporting intervals must be positive"
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StageConfig":
        return cls(**payload)


@dataclass(frozen=True)
class ScratchTrainingConfig:
    """JSON-serializable configuration for a full staged scratch run."""

    stages: tuple[StageConfig, ...]
    tokenizer_path: str
    tokenizer_sha256: str
    model: HybridConfig = field(default_factory=HybridConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    output_dir: str = "checkpoints/hybrid-scratch"
    seed: int = 42
    parameter_dtype: Literal["bfloat16", "float32"] = "bfloat16"

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("at least one training stage is required")
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")
        if self.parameter_dtype not in {"bfloat16", "float32"}:
            raise ValueError("parameter_dtype must be 'bfloat16' or 'float32'")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path is required")
        if len(self.tokenizer_sha256) != 64:
            raise ValueError("tokenizer_sha256 must be a 64-character SHA-256 digest")
        try:
            int(self.tokenizer_sha256, 16)
        except ValueError as error:
            raise ValueError("tokenizer_sha256 must be hexadecimal") from error
        if self.model.vocab_size != VOCAB_SIZE:
            raise ValueError(
                f"HybridConfig vocab_size must equal scratch tokenizer size {VOCAB_SIZE}"
            )
        for stage in self.stages:
            if stage.sequence_length > self.model.context_length:
                raise ValueError(
                    f"stage {stage.name!r} sequence_length exceeds model context_length"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScratchTrainingConfig":
        raw_model = dict(payload.get("model", {}))
        if "layer_schedule" in raw_model:
            raw_model["layer_schedule"] = tuple(raw_model["layer_schedule"])
        return cls(
            stages=tuple(StageConfig.from_dict(item) for item in payload["stages"]),
            tokenizer_path=str(payload["tokenizer_path"]),
            tokenizer_sha256=str(payload["tokenizer_sha256"]),
            model=HybridConfig(**raw_model),
            optimizer=OptimizerConfig(**payload.get("optimizer", {})),
            guards=GuardConfig(**payload.get("guards", {})),
            output_dir=str(payload.get("output_dir", "checkpoints/hybrid-scratch")),
            seed=int(payload.get("seed", 42)),
            parameter_dtype=payload.get("parameter_dtype", "bfloat16"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ScratchTrainingConfig":
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("training configuration must be a JSON object")
        return cls.from_dict(payload)

    def write_json(self, path: str | Path) -> None:
        _atomic_write_json(Path(path), self.to_dict())

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TokenizedSample:
    """One isolated sequence and the token positions contributing to loss."""

    tokens: tuple[int, ...]
    target_mask: tuple[bool, ...]
    source_line: int

    def __post_init__(self) -> None:
        if len(self.tokens) < 2 or len(self.tokens) != len(self.target_mask):
            raise ValueError(
                "sample tokens and target_mask must have equal length >= 2"
            )
        if not any(self.target_mask[1:]):
            raise ValueError("sample has no trainable next-token targets")


@dataclass(frozen=True)
class TrainingBatch:
    inputs: mx.array
    targets: mx.array
    loss_mask: mx.array
    loss_tokens: int
    text_logit_bias: mx.array | None = None


def _text_logit_bias(tokenizer: ScratchTokenizer) -> mx.array:
    """Mask tokenizer rows that are not legal text-loss classes."""

    bias = np.zeros((tokenizer.vocab_size,), dtype=np.float32)
    inactive = tokenizer.inactive_text_token_ids
    if inactive:
        bias[np.asarray(inactive, dtype=np.int64)] = -np.inf
    return mx.array(bias)


class JsonlDataset:
    """Eager tokenization of provenance-audited canonical JSONL records."""

    def __init__(
        self,
        path: str | Path,
        objective: Objective,
        tokenizer: ScratchTokenizer,
        sequence_length: int,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.objective = objective
        self.sequence_length = sequence_length
        self.pad_id = tokenizer.pad_id
        self.text_logit_bias = _text_logit_bias(tokenizer)
        if not self.path.is_file():
            raise FileNotFoundError(f"Training source not found: {self.path}")
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.samples: list[TokenizedSample] = []
        for line_number, raw_line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {self.path}:{line_number}: {error}"
                ) from error
            _validate_audited_record(record, line_number)
            self.samples.extend(
                _record_samples(
                    record, objective, tokenizer, sequence_length, line_number
                )
            )
        if not self.samples:
            raise ValueError(f"Training source contains no usable records: {self.path}")

    def __len__(self) -> int:
        return len(self.samples)


def _validate_audited_record(record: object, line_number: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Audited JSONL line {line_number} must be an object")
    required = (
        "type",
        "source",
        "license",
        "provenance",
        "record_id",
        "content_sha256",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"Audited JSONL line {line_number} is missing {missing}")
    if not isinstance(record["source"], str) or not record["source"].strip():
        raise ValueError(f"Audited JSONL line {line_number} has an invalid source")
    if not isinstance(record["license"], str) or not record["license"].strip():
        raise ValueError(f"Audited JSONL line {line_number} has an invalid license")
    if record["provenance"] in (None, "", [], {}):
        raise ValueError(f"Audited JSONL line {line_number} has empty provenance")
    if not isinstance(record["record_id"], str) or not record["record_id"].strip():
        raise ValueError(f"Audited JSONL line {line_number} has an invalid record_id")
    checksum = record["content_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError(
            f"Audited JSONL line {line_number} has an invalid content checksum"
        )
    try:
        int(checksum, 16)
    except ValueError as error:
        raise ValueError(
            f"Audited JSONL line {line_number} has a non-hexadecimal content checksum"
        ) from error
    if record["type"] == "document" and not isinstance(record.get("text"), str):
        raise ValueError(f"Audited document line {line_number} has no text")
    if record["type"] == "dialogue":
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Audited dialogue line {line_number} has no messages")
        if any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            or not message["content"]
            for message in messages
        ):
            raise ValueError(
                f"Audited dialogue line {line_number} has an invalid message"
            )
    if record["type"] not in {"document", "dialogue"}:
        raise ValueError(f"Audited JSONL line {line_number} has an invalid type")
    canonical = (
        str(record["text"])
        if record["type"] == "document"
        else json.dumps(
            [[message["role"], message["content"]] for message in record["messages"]],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if record["content_sha256"] != actual:
        raise ValueError(
            f"Audited JSONL line {line_number} content checksum does not match"
        )


def _record_samples(
    record: dict[str, Any],
    objective: Objective,
    tokenizer: ScratchTokenizer,
    sequence_length: int,
    line_number: int,
) -> list[TokenizedSample]:
    if objective == "sft":
        if record["type"] != "dialogue":
            raise ValueError(f"SFT line {line_number} must be an audited dialogue")
        return [_sft_sample(record, tokenizer, sequence_length, line_number)]
    if record["type"] == "document":
        tokens = tokenizer.encode(str(record["text"]), bos=True, eos=True)
    else:
        tokens = tokenizer.encode_dialogue(record["messages"], bos=True, eos=True)
    return _pretraining_samples_from_tokens(
        tokens, tokenizer, sequence_length, line_number
    )


def _pretraining_samples_from_tokens(
    tokens: list[int] | tuple[int, ...],
    tokenizer: ScratchTokenizer,
    sequence_length: int,
    source_line: int,
) -> list[TokenizedSample]:
    if (
        len(tokens) < 2
        or tokens[0] != tokenizer.bos_id
        or tokens[-1] != tokenizer.eos_id
    ):
        raise ValueError("prepared sequence is missing its BOS/EOS boundary")
    payload = list(tokens[1:-1])
    payload_per_chunk = max(1, sequence_length - 1)
    chunks = [
        payload[index : index + payload_per_chunk]
        for index in range(0, len(payload), payload_per_chunk)
    ]
    if not chunks:
        chunks = [[]]
    samples = []
    for chunk in chunks:
        bounded = (tokenizer.bos_id, *chunk, tokenizer.eos_id)
        target_mask = (False, *(True for _ in range(len(bounded) - 1)))
        samples.append(TokenizedSample(tuple(bounded), tuple(target_mask), source_line))
    return samples


def _sft_sample(
    record: dict[str, Any],
    tokenizer: ScratchTokenizer,
    sequence_length: int,
    line_number: int,
) -> TokenizedSample:
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        raise ValueError(f"SFT line {line_number} must contain a messages list")
    tokens, boundaries = tokenizer.encode_dialogue_with_boundaries(record["messages"])
    if len(tokens) > sequence_length + 1:
        raise ValueError(
            f"SFT line {line_number} has {len(tokens) - 1} targets, exceeding "
            f"sequence_length={sequence_length}; preprocess it into shorter conversations"
        )
    target_mask = _assistant_target_mask(tokens, boundaries, tokenizer)
    return TokenizedSample(tuple(tokens), target_mask, line_number)


def _assistant_target_mask(
    tokens: list[int] | tuple[int, ...],
    boundaries: list[Any],
    tokenizer: ScratchTokenizer,
) -> tuple[bool, ...]:
    if (
        len(tokens) < 2
        or tokens[0] != tokenizer.bos_id
        or tokens[-1] != tokenizer.eos_id
    ):
        raise ValueError("dialogue record is missing its BOS/EOS boundary")
    mask = [False] * len(tokens)
    last_assistant_end: int | None = None
    expected_role_offset = 1
    for boundary in boundaries:
        role = boundary.role if hasattr(boundary, "role") else boundary["role"]
        role_offset = (
            boundary.role_offset
            if hasattr(boundary, "role_offset")
            else int(boundary["role_offset"])
        )
        content_offset = (
            boundary.content_offset
            if hasattr(boundary, "content_offset")
            else int(boundary["content_offset"])
        )
        content_length = (
            boundary.content_length
            if hasattr(boundary, "content_length")
            else int(boundary["content_length"])
        )
        end_turn_offset = (
            boundary.end_turn_offset
            if hasattr(boundary, "end_turn_offset")
            else int(boundary["end_turn_offset"])
        )
        if (
            role_offset != expected_role_offset
            or content_offset != role_offset + 1
            or content_length < 0
            or end_turn_offset != content_offset + content_length
            or end_turn_offset >= len(tokens) - 1
        ):
            raise ValueError("dialogue boundary lies outside its token record")
        if tokens[role_offset] != tokenizer.role_id(role):
            raise ValueError("dialogue role boundary does not match the tokenizer")
        if tokens[end_turn_offset] != tokenizer.end_turn_id:
            raise ValueError("dialogue end-turn boundary does not match the tokenizer")
        if role == "assistant":
            for position in range(role_offset, end_turn_offset + 1):
                mask[position] = True
            last_assistant_end = end_turn_offset
        expected_role_offset = end_turn_offset + 1
    if expected_role_offset != len(tokens) - 1:
        raise ValueError("dialogue boundaries do not cover the complete token record")
    if last_assistant_end == len(tokens) - 2 and tokens[-1] == tokenizer.eos_id:
        mask[-1] = True
    return tuple(mask)


class TokenShardDataset:
    """Read prepared uint16 shards and their auditable record boundaries."""

    def __init__(
        self,
        path: str | Path,
        split: Literal["train", "validation"],
        objective: Objective,
        tokenizer: ScratchTokenizer,
        sequence_length: int,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.objective = objective
        self.sequence_length = sequence_length
        self.pad_id = tokenizer.pad_id
        self.text_logit_bias = _text_logit_bias(tokenizer)
        inactive_token_ids = frozenset(tokenizer.inactive_text_token_ids)
        index_paths = self._index_paths(split)
        digest = hashlib.sha256()
        self.samples: list[TokenizedSample] = []
        source_record = 0
        for index_path in index_paths:
            binary_path = index_path.with_name(
                index_path.name.replace(".idx.json", ".bin")
            )
            if not binary_path.is_file():
                raise FileNotFoundError(f"Token shard is missing: {binary_path}")
            index_bytes = index_path.read_bytes()
            binary_bytes = binary_path.read_bytes()
            digest.update(len(index_bytes).to_bytes(8, "big"))
            digest.update(index_bytes)
            digest.update(len(binary_bytes).to_bytes(8, "big"))
            digest.update(binary_bytes)
            index = json.loads(index_bytes)
            if (
                index.get("format_version") != 1
                or index.get("dtype") != "uint16-le"
                or index.get("split") != split
            ):
                raise ValueError(f"Prepared shard index is incompatible: {index_path}")
            if index.get("tokenizer_sha256") != tokenizer.metadata["tokenizer_sha256"]:
                raise ValueError(
                    f"Prepared shard tokenizer checksum differs: {index_path}"
                )
            shard = np.frombuffer(binary_bytes, dtype="<u2")
            if len(shard) != int(index.get("token_count", -1)):
                raise ValueError(f"Prepared shard token count differs: {index_path}")
            for record in index.get("records", []):
                offset = int(record["offset"])
                length = int(record["length"])
                record_tokens = [
                    int(value) for value in shard[offset : offset + length]
                ]
                if len(record_tokens) != length or any(
                    token < 0 or token >= tokenizer.vocab_size
                    for token in record_tokens
                ):
                    raise ValueError(
                        f"Prepared shard record is out of range: {index_path}"
                    )
                if inactive_token_ids.intersection(record_tokens):
                    raise ValueError(
                        f"Prepared shard record contains an inactive text token: {index_path}"
                    )
                source_record += 1
                if objective == "pretrain":
                    self.samples.extend(
                        _pretraining_samples_from_tokens(
                            record_tokens,
                            tokenizer,
                            sequence_length,
                            source_record,
                        )
                    )
                    continue
                if record.get("record_type") != "dialogue":
                    raise ValueError(
                        "SFT token shards may contain only dialogue records"
                    )
                if len(record_tokens) > sequence_length + 1:
                    raise ValueError(
                        "SFT token-shard conversation exceeds sequence_length"
                    )
                mask = _assistant_target_mask(
                    record_tokens,
                    list(record.get("dialogue_boundaries", [])),
                    tokenizer,
                )
                self.samples.append(
                    TokenizedSample(tuple(record_tokens), mask, source_record)
                )
        self.sha256 = digest.hexdigest()
        if not self.samples:
            raise ValueError(
                f"Prepared token source contains no {split} records: {self.path}"
            )

    def _index_paths(self, split: str) -> list[Path]:
        if self.path.is_file() and self.path.name.endswith(".idx.json"):
            paths = [self.path]
        else:
            shard_dir = (
                self.path / "shards" if (self.path / "shards").is_dir() else self.path
            )
            paths = sorted(shard_dir.glob(f"{split}-*.idx.json"))
        if not paths:
            raise FileNotFoundError(
                f"No prepared {split} shard indexes found at {self.path}"
            )
        return paths

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class DataCursor:
    epoch: int = 0
    index: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DataCursor":
        payload = payload or {}
        return cls(
            epoch=int(payload.get("epoch", 0)), index=int(payload.get("index", 0))
        )


class BoundaryBatcher:
    """Batch isolated rows; never pack two records into one causal sequence."""

    def __init__(
        self,
        dataset: JsonlDataset | TokenShardDataset,
        *,
        batch_size: int,
        sequence_length: int,
        seed: int,
        stream_name: str,
        cursor: DataCursor | None = None,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.stream_name = stream_name
        self.cursor = cursor or DataCursor()
        self.shuffle = shuffle
        if self.cursor.epoch < 0 or not 0 <= self.cursor.index <= len(dataset):
            raise ValueError("data cursor is outside the dataset")

    def _order(self, epoch: int) -> list[int]:
        indices = list(range(len(self.dataset)))
        if not self.shuffle:
            return indices
        prefix = f"{self.seed}:{self.stream_name}:{epoch}:".encode()
        return sorted(
            indices,
            key=lambda index: hashlib.blake2b(
                prefix + str(index).encode(), digest_size=16
            ).digest(),
        )

    def next_batch(self) -> TrainingBatch:
        selected: list[TokenizedSample] = []
        while len(selected) < self.batch_size:
            order = self._order(self.cursor.epoch)
            if self.cursor.index == len(order):
                self.cursor.epoch += 1
                self.cursor.index = 0
                continue
            take = min(self.batch_size - len(selected), len(order) - self.cursor.index)
            selected.extend(
                self.dataset.samples[index]
                for index in order[self.cursor.index : self.cursor.index + take]
            )
            self.cursor.index += take

        inputs = np.full(
            (self.batch_size, self.sequence_length),
            self.dataset.pad_id,
            dtype=np.int32,
        )
        targets = np.full_like(inputs, self.dataset.pad_id)
        loss_mask = np.zeros((self.batch_size, self.sequence_length), dtype=np.float32)
        for row, sample in enumerate(selected):
            target_count = min(len(sample.tokens) - 1, self.sequence_length)
            inputs[row, :target_count] = sample.tokens[:target_count]
            targets[row, :target_count] = sample.tokens[1 : target_count + 1]
            loss_mask[row, :target_count] = sample.target_mask[1 : target_count + 1]
        loss_tokens = int(loss_mask.sum())
        if loss_tokens <= 0:
            raise ValueError("batch has no supervised target tokens")
        return TrainingBatch(
            inputs=mx.array(inputs),
            targets=mx.array(targets),
            loss_mask=mx.array(loss_mask),
            loss_tokens=loss_tokens,
            text_logit_bias=self.dataset.text_logit_bias,
        )


def masked_next_token_loss(model: HybridLM, batch: TrainingBatch) -> mx.array:
    """Compute next-token cross entropy in FP32 over selected targets only."""
    logits = model(batch.inputs).astype(mx.float32)
    text_logit_bias = getattr(batch, "text_logit_bias", None)
    if text_logit_bias is not None:
        if text_logit_bias.shape != (logits.shape[-1],):
            raise ValueError("text_logit_bias does not match the model vocabulary")
        logits = logits + text_logit_bias.astype(mx.float32)
    per_token = nn.losses.cross_entropy(logits, batch.targets, reduction="none")
    denominator = mx.maximum(mx.sum(batch.loss_mask), mx.array(1.0, dtype=mx.float32))
    return mx.sum(per_token * batch.loss_mask) / denominator


def scheduled_learning_rate(stage: StageConfig, zero_based_step: int) -> float:
    """Linear warmup followed by cosine decay to a non-zero floor."""
    if zero_based_step < 0:
        raise ValueError("schedule step must be non-negative")
    peak = stage.peak_learning_rate
    floor = peak * stage.min_learning_rate_ratio
    if stage.warmup_steps and zero_based_step < stage.warmup_steps:
        return peak * (zero_based_step + 1) / stage.warmup_steps
    decay_steps = max(1, stage.steps - stage.warmup_steps - 1)
    progress = min(1.0, max(0.0, (zero_based_step - stage.warmup_steps) / decay_steps))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


class FP32AdamW(optim.Optimizer):
    """AdamW with FP32 moments and master weights for BF16 model parameters."""

    def __init__(self, learning_rate: float, config: OptimizerConfig) -> None:
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.eps = config.eps
        self.weight_decay = config.weight_decay
        self.bias_correction = config.bias_correction

    def init_single(self, parameter: mx.array, state: dict) -> None:
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["master"] = parameter.astype(mx.float32)

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: dict
    ) -> mx.array:
        gradient = gradient.astype(mx.float32)
        learning_rate = self.learning_rate.astype(mx.float32)
        moment = self.beta1 * state["m"] + (1.0 - self.beta1) * gradient
        variance = self.beta2 * state["v"] + (1.0 - self.beta2) * mx.square(gradient)
        master = state["master"] * (1.0 - learning_rate * self.weight_decay)
        if self.bias_correction:
            step = self.step.astype(mx.float32)
            moment_hat = moment / (1.0 - self.beta1**step)
            variance_hat = variance / (1.0 - self.beta2**step)
        else:
            moment_hat, variance_hat = moment, variance
        master = master - learning_rate * moment_hat / (
            mx.sqrt(variance_hat) + self.eps
        )
        state["m"] = moment
        state["v"] = variance
        state["master"] = master
        return master.astype(parameter.dtype)


@dataclass
class TrainerState:
    global_step: int = 0
    stage_index: int = 0
    stage_step: int = 0
    accepted_updates: int = 0
    skipped_updates: int = 0
    tokens_seen: int = 0
    loss_ema: float | None = None
    last_good_loss: float | None = None
    consecutive_bad_updates: int = 0
    best_eval_losses: dict[str, float] = field(default_factory=dict)
    data_cursors: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainerState":
        fields = {
            key: payload[key] for key in cls.__dataclass_fields__ if key in payload
        }
        return cls(**fields)


class MetricsLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **values: object) -> None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        encoded = (
            json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())


class ScratchTrainer:
    """Own a complete random-init staged run and its resumable state."""

    def __init__(
        self,
        config: ScratchTrainingConfig,
        *,
        resume: str | Path | None = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config_fingerprint = config.fingerprint()
        self.tokenizer = ScratchTokenizer.load(config.tokenizer_path)
        self.tokenizer_sha256 = str(self.tokenizer.metadata["tokenizer_sha256"])
        if self.tokenizer_sha256 != config.tokenizer_sha256:
            raise ValueError(
                "loaded tokenizer checksum does not match tokenizer_sha256 in the training config"
            )
        if self.tokenizer.vocab_size != config.model.vocab_size:
            raise ValueError(
                "loaded tokenizer vocabulary does not match HybridConfig.vocab_size"
            )

        self.datasets: dict[
            str,
            tuple[JsonlDataset | TokenShardDataset, JsonlDataset | TokenShardDataset],
        ] = {}
        for stage in config.stages:
            if stage.data_format == "audited_jsonl":
                train_data = JsonlDataset(
                    stage.train_path,
                    stage.objective,
                    self.tokenizer,
                    stage.sequence_length,
                )
                eval_data = JsonlDataset(
                    stage.eval_path,
                    stage.objective,
                    self.tokenizer,
                    stage.sequence_length,
                )
            else:
                train_data = TokenShardDataset(
                    stage.train_path,
                    "train",
                    stage.objective,
                    self.tokenizer,
                    stage.sequence_length,
                )
                eval_data = TokenShardDataset(
                    stage.eval_path,
                    "validation",
                    stage.objective,
                    self.tokenizer,
                    stage.sequence_length,
                )
            if train_data.sha256 == eval_data.sha256:
                raise ValueError(
                    f"stage {stage.name!r} train and held-out data are identical"
                )
            self.datasets[stage.name] = train_data, eval_data
        self.source_hashes = {
            f"{stage.name}:train": self.datasets[stage.name][0].sha256
            for stage in config.stages
        } | {
            f"{stage.name}:eval": self.datasets[stage.name][1].sha256
            for stage in config.stages
        }

        random.seed(config.seed)
        np.random.seed(config.seed)
        mx.random.seed(config.seed)
        self.mlx_rng_key = mx.random.key(config.seed)
        self.model = HybridLM(config.model)
        self.model.set_dtype(
            mx.bfloat16 if config.parameter_dtype == "bfloat16" else mx.float32
        )
        self.model.train()
        first_lr = config.stages[0].peak_learning_rate
        self.optimizer = FP32AdamW(first_lr, config.optimizer)
        self.optimizer.init(self.model.trainable_parameters())
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.loss_and_grad = nn.value_and_grad(self.model, masked_next_token_loss)
        self.state = TrainerState()
        self._last_saved_state_signature: str | None = None
        self.metrics = MetricsLogger(self.output_dir / "metrics.jsonl")
        self._write_or_validate_provenance()
        if resume is not None:
            self.load_checkpoint(resume)

    @property
    def complete(self) -> bool:
        return self.state.stage_index >= len(self.config.stages)

    def _write_or_validate_provenance(self) -> None:
        provenance_path = self.output_dir / "provenance.json"
        record = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scratch_initialization": True,
            "external_weights": False,
            "config_sha256": self.config_fingerprint,
            "source_sha256": self.source_hashes,
            "tokenizer": {
                "path": str(self.tokenizer.artifact_dir.resolve()),
                "sha256": self.tokenizer_sha256,
                "vocab_size": self.tokenizer.vocab_size,
            },
            "model_parameters": self.config.model.expected_parameter_count(),
            "parameter_dtype": self.config.parameter_dtype,
            "sample_boundaries": "one-record-or-document-chunk-per-padded-row",
            "optimizer": {
                "name": "fp32-master-adamw",
                "betas": [self.config.optimizer.beta1, self.config.optimizer.beta2],
                "weight_decay": self.config.optimizer.weight_decay,
                "grad_clip": self.config.optimizer.grad_clip,
            },
            "later_ablations_not_enabled": ["Muon", "multi-token prediction", "RL"],
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "mlx": getattr(mx, "__version__", "unknown"),
            },
        }
        if provenance_path.is_file():
            existing = json.loads(provenance_path.read_text())
            if existing.get("config_sha256") != self.config_fingerprint:
                raise ValueError(
                    "output directory belongs to a different training config"
                )
            if existing.get("source_sha256") != self.source_hashes:
                raise ValueError(
                    "training or held-out data changed since the run began"
                )
            if existing.get("tokenizer", {}).get("sha256") != self.tokenizer_sha256:
                raise ValueError("scratch tokenizer changed since the run began")
        else:
            _atomic_write_json(provenance_path, record)
            _atomic_write_json(self.output_dir / "config.json", self.config.to_dict())

    def _cursor_for(self, stage: StageConfig) -> DataCursor:
        return DataCursor.from_dict(self.state.data_cursors.get(stage.name))

    def _set_cursor(self, stage: StageConfig, cursor: DataCursor) -> None:
        self.state.data_cursors[stage.name] = asdict(cursor)

    def evaluate(self, stage: StageConfig) -> float:
        _, eval_data = self.datasets[stage.name]
        batcher = BoundaryBatcher(
            eval_data,
            batch_size=stage.batch_size,
            sequence_length=stage.sequence_length,
            seed=self.config.seed,
            stream_name=f"{stage.name}:heldout",
            shuffle=False,
        )
        weighted_loss = 0.0
        total_tokens = 0
        was_training = self.model.training
        self.model.eval()
        try:
            for _ in range(stage.eval_batches):
                batch = batcher.next_batch()
                loss = masked_next_token_loss(self.model, batch)
                mx.eval(loss)
                weighted_loss += float(loss.item()) * batch.loss_tokens
                total_tokens += batch.loss_tokens
        finally:
            self.model.train(was_training)
        return weighted_loss / max(1, total_tokens)

    def _guard_reason(
        self,
        loss_value: float,
        gradient_norm: float,
        gradients: object,
    ) -> str | None:
        if not math.isfinite(loss_value):
            return "non_finite_loss"
        if not math.isfinite(gradient_norm) or not _tree_is_finite(gradients):
            return "non_finite_gradient"
        guards = self.config.guards
        if (
            self.state.loss_ema is not None
            and self.state.accepted_updates >= guards.loss_spike_warmup_updates
            and loss_value > self.state.loss_ema * guards.loss_spike_factor
        ):
            return "loss_spike"
        return None

    def _accept_loss(self, value: float) -> None:
        decay = self.config.guards.ema_decay
        self.state.loss_ema = (
            value
            if self.state.loss_ema is None
            else decay * self.state.loss_ema + (1.0 - decay) * value
        )
        self.state.last_good_loss = value
        self.state.consecutive_bad_updates = 0
        self.state.accepted_updates += 1

    def _skip_update(
        self, reason: str, stage: StageConfig, loss: float, grad_norm: float
    ) -> None:
        self.state.skipped_updates += 1
        self.state.consecutive_bad_updates += 1
        self.metrics.write(
            "skipped_update",
            global_step=self.state.global_step,
            stage=stage.name,
            stage_step=self.state.stage_step,
            reason=reason,
            loss=loss if math.isfinite(loss) else None,
            grad_norm=grad_norm if math.isfinite(grad_norm) else None,
        )
        if (
            self.state.consecutive_bad_updates
            >= self.config.guards.max_consecutive_bad_updates
        ):
            self.save_checkpoint()
            raise RuntimeError(
                f"Stopped after {self.state.consecutive_bad_updates} consecutive bad updates"
            )

    def run(self, *, max_steps: int | None = None) -> TrainerState:
        """Train until all stages finish or ``max_steps`` attempts are made."""
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if not (self.output_dir / "last.json").is_file():
            self.save_checkpoint()
        attempts = 0
        while not self.complete and (max_steps is None or attempts < max_steps):
            stage = self.config.stages[self.state.stage_index]
            if self.state.stage_step >= stage.steps:
                self.state.stage_index += 1
                self.state.stage_step = 0
                self.state.loss_ema = None
                self.state.last_good_loss = None
                self.state.consecutive_bad_updates = 0
                continue
            train_data, _ = self.datasets[stage.name]
            batcher = BoundaryBatcher(
                train_data,
                batch_size=stage.batch_size,
                sequence_length=stage.sequence_length,
                seed=self.config.seed,
                stream_name=f"{stage.name}:train",
                cursor=self._cursor_for(stage),
            )
            zero_based_stage_step = self.state.stage_step
            batch = batcher.next_batch()
            self._set_cursor(stage, batcher.cursor)
            learning_rate = scheduled_learning_rate(stage, zero_based_stage_step)
            loss, gradients = self.loss_and_grad(self.model, batch)
            clipped_gradients, gradient_norm_array = clip_grad_norm_fp32(
                gradients, self.config.optimizer.grad_clip
            )
            mx.eval(loss, gradients, clipped_gradients, gradient_norm_array)
            loss_value = float(loss.item())
            gradient_norm = float(gradient_norm_array.item())

            self.state.global_step += 1
            self.state.stage_step += 1
            self.state.tokens_seen += batch.loss_tokens
            attempts += 1
            reason = self._guard_reason(loss_value, gradient_norm, gradients)
            if reason is None:
                previous_parameters = self.model.parameters()
                previous_optimizer_state = tree_map(
                    lambda value: value, self.optimizer.state
                )
                self.optimizer.learning_rate = mx.array(learning_rate, dtype=mx.float32)
                self.optimizer.update(self.model, clipped_gradients)
                mx.eval(self.model.parameters(), self.optimizer.state)
                if not _tree_is_finite(self.model.parameters()) or not _tree_is_finite(
                    self.optimizer.state
                ):
                    self.model.update(previous_parameters)
                    self.optimizer.state = previous_optimizer_state
                    self.optimizer.init(self.model.trainable_parameters())
                    mx.eval(self.model.parameters(), self.optimizer.state)
                    reason = "non_finite_update"
                else:
                    self._accept_loss(loss_value)
            if reason is not None:
                self._skip_update(reason, stage, loss_value, gradient_norm)

            if reason is None and (
                self.state.global_step == 1
                or self.state.stage_step % stage.report_every == 0
            ):
                self.metrics.write(
                    "train",
                    global_step=self.state.global_step,
                    stage=stage.name,
                    stage_step=self.state.stage_step,
                    objective=stage.objective,
                    loss=loss_value,
                    grad_norm=gradient_norm,
                    learning_rate=learning_rate,
                    loss_tokens=batch.loss_tokens,
                    tokens_seen=self.state.tokens_seen,
                    accepted_updates=self.state.accepted_updates,
                    skipped_updates=self.state.skipped_updates,
                )

            is_best = False
            if (
                self.state.stage_step % stage.eval_every == 0
                or self.state.stage_step == stage.steps
            ):
                eval_loss = self.evaluate(stage)
                self.metrics.write(
                    "eval",
                    global_step=self.state.global_step,
                    stage=stage.name,
                    stage_step=self.state.stage_step,
                    loss=eval_loss if math.isfinite(eval_loss) else None,
                    perplexity=(
                        math.exp(min(eval_loss, 50.0))
                        if math.isfinite(eval_loss)
                        else None
                    ),
                )
                previous_best = self.state.best_eval_losses.get(stage.name, math.inf)
                if math.isfinite(eval_loss) and eval_loss < previous_best:
                    self.state.best_eval_losses[stage.name] = eval_loss
                    is_best = True

            stage_complete = self.state.stage_step >= stage.steps
            save_due = self.state.stage_step % stage.save_every == 0
            if stage_complete:
                self.state.stage_index += 1
                self.state.stage_step = 0
                self.state.loss_ema = None
                self.state.last_good_loss = None
                self.state.consecutive_bad_updates = 0
            if save_due or is_best or stage_complete:
                self.save_checkpoint(
                    is_best=is_best, best_stage=stage.name if is_best else None
                )

        if self._last_saved_state_signature != self._state_signature():
            self.save_checkpoint()
        return self.state

    def _state_signature(self) -> str:
        payload = json.dumps(asdict(self.state), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _checkpoint_payload(self) -> dict[str, Any]:
        mlx_global_key = mx.array(mx.random.state[0])
        mx.eval(mlx_global_key)
        return {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": self.config_fingerprint,
            "source_sha256": self.source_hashes,
            "tokenizer_sha256": self.tokenizer_sha256,
            "trainer": asdict(self.state),
            "rng": {
                "python": random.getstate(),
                "numpy": _numpy_rng_to_dict(np.random.get_state()),
                "mlx_explicit_key": self.mlx_rng_key.tolist(),
                "mlx_global_key": mlx_global_key.tolist(),
            },
        }

    def save_checkpoint(
        self,
        *,
        is_best: bool = False,
        best_stage: str | None = None,
    ) -> Path:
        """Write an immutable checkpoint, then atomically advance pointers."""
        temporary = Path(
            tempfile.mkdtemp(prefix=".checkpoint-", dir=self.checkpoint_dir)
        )
        identifier = f"step-{self.state.global_step:09d}-{uuid.uuid4().hex[:12]}"
        destination = self.checkpoint_dir / identifier
        try:
            model_checkpoint = temporary / "model.safetensors"
            optimizer_checkpoint = temporary / "optimizer.safetensors"
            self.model.save_weights(str(model_checkpoint))
            optimizer_arrays = dict(tree_flatten(self.optimizer.state))
            mx.eval(optimizer_arrays)
            mx.save_safetensors(
                str(optimizer_checkpoint),
                optimizer_arrays,
            )
            _fsync_file(model_checkpoint)
            _fsync_file(optimizer_checkpoint)
            _write_json(temporary / "state.json", self._checkpoint_payload())
            _write_json(
                temporary / "manifest.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "scratch_only": True,
                    "external_weights": False,
                    "config_sha256": self.config_fingerprint,
                    "tokenizer_sha256": self.tokenizer_sha256,
                    "global_step": self.state.global_step,
                },
            )
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(self.checkpoint_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        relative = str(destination.relative_to(self.output_dir))
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": relative,
            "global_step": self.state.global_step,
        }
        _atomic_write_json(self.output_dir / "last.json", pointer)
        if is_best:
            _atomic_write_json(
                self.output_dir / "best.json",
                {**pointer, "stage": best_stage},
            )
        self.metrics.write(
            "checkpoint",
            global_step=self.state.global_step,
            checkpoint=relative,
            best=is_best,
            best_stage=best_stage,
        )
        self._last_saved_state_signature = self._state_signature()
        return destination

    def load_checkpoint(self, path: str | Path) -> Path:
        """Resume only a complete checkpoint emitted by this scratch trainer."""
        checkpoint = _resolve_checkpoint_path(Path(path).expanduser().resolve())
        manifest_path = checkpoint / "manifest.json"
        state_path = checkpoint / "state.json"
        model_path = checkpoint / "model.safetensors"
        optimizer_path = checkpoint / "optimizer.safetensors"
        if not all(
            item.is_file()
            for item in (manifest_path, state_path, model_path, optimizer_path)
        ):
            raise ValueError(
                "resume path is not a complete scratch-training checkpoint"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("checkpoint manifest schema is incompatible")
        if (
            manifest.get("scratch_only") is not True
            or manifest.get("external_weights") is not False
        ):
            raise ValueError(
                "external or unverified model checkpoints are not accepted"
            )
        if manifest.get("config_sha256") != self.config_fingerprint:
            raise ValueError("checkpoint training configuration does not match")
        if manifest.get("tokenizer_sha256") != self.tokenizer_sha256:
            raise ValueError(
                "checkpoint tokenizer does not match the loaded scratch tokenizer"
            )
        payload = json.loads(state_path.read_text())
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("checkpoint state schema is incompatible")
        if payload.get("source_sha256") != self.source_hashes:
            raise ValueError(
                "checkpoint data provenance does not match current sources"
            )
        if payload.get("tokenizer_sha256") != self.tokenizer_sha256:
            raise ValueError(
                "checkpoint state tokenizer does not match the loaded tokenizer"
            )

        self.model.load_weights(str(model_path), strict=True)
        flat_optimizer = mx.load(str(optimizer_path))
        if not isinstance(flat_optimizer, dict):
            raise ValueError("optimizer checkpoint is malformed")
        expected_optimizer = dict(tree_flatten(self.optimizer.state))
        _validate_array_state(expected_optimizer, flat_optimizer, "optimizer")
        self.optimizer.state = tree_unflatten(list(flat_optimizer.items()))
        self.optimizer.init(self.model.trainable_parameters())
        self.state = TrainerState.from_dict(payload["trainer"])
        if not 0 <= self.state.stage_index <= len(self.config.stages):
            raise ValueError("trainer stage index is outside the configured stages")
        if self.state.stage_index < len(self.config.stages):
            active_stage = self.config.stages[self.state.stage_index]
            if not 0 <= self.state.stage_step <= active_stage.steps:
                raise ValueError("trainer stage step is outside the active stage")
        rng = payload.get("rng", {})
        random.setstate(_nested_tuple(rng["python"]))
        np.random.set_state(_numpy_rng_from_dict(rng["numpy"]))
        self.mlx_rng_key = mx.array(rng["mlx_explicit_key"], dtype=mx.uint32)
        mlx_words = rng["mlx_global_key"]
        if not isinstance(mlx_words, list) or len(mlx_words) != 2:
            raise ValueError("MLX RNG checkpoint is malformed")
        mx.random.seed((int(mlx_words[0]) << 32) | int(mlx_words[1]))
        mx.eval(self.model.parameters(), self.optimizer.state, self.mlx_rng_key)
        if int(self.optimizer.step.item()) != self.state.accepted_updates:
            raise ValueError("optimizer step does not match accepted update count")
        self.model.train()
        self._last_saved_state_signature = self._state_signature()
        return checkpoint


def _tree_is_finite(tree: object) -> bool:
    leaves = [value for _, value in tree_flatten(tree) if isinstance(value, mx.array)]
    checks = [mx.all(mx.isfinite(value)) for value in leaves]
    if not checks:
        return True
    mx.eval(checks)
    return all(bool(check.item()) for check in checks)


def clip_grad_norm_fp32(grads: object, max_norm: float) -> tuple[object, mx.array]:
    """Clip one global norm while accumulating BF16 gradients safely in FP32."""
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    squared_norm = mx.array(0.0, dtype=mx.float32)
    for _, gradient in tree_flatten(grads):
        squared_norm = squared_norm + mx.sum(mx.square(gradient.astype(mx.float32)))
    norm = mx.sqrt(squared_norm)
    scale = mx.minimum(mx.array(1.0, dtype=mx.float32), max_norm / (norm + 1e-6))
    clipped = tree_map(
        lambda gradient: gradient * scale.astype(gradient.dtype),
        grads,
    )
    return clipped, norm


def _validate_array_state(
    expected: dict[str, mx.array],
    loaded: dict[str, mx.array],
    label: str,
) -> None:
    if set(expected) != set(loaded):
        missing = sorted(set(expected) - set(loaded))
        extra = sorted(set(loaded) - set(expected))
        raise ValueError(f"{label} state keys differ; missing={missing}, extra={extra}")
    for key, expected_value in expected.items():
        loaded_value = loaded[key]
        if (
            expected_value.shape != loaded_value.shape
            or expected_value.dtype != loaded_value.dtype
        ):
            raise ValueError(
                f"{label} state {key!r} shape/dtype mismatch: "
                f"expected {expected_value.shape}/{expected_value.dtype}, "
                f"received {loaded_value.shape}/{loaded_value.dtype}"
            )


def _nested_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _numpy_rng_to_dict(state: tuple) -> dict[str, object]:
    name, keys, position, has_gauss, cached_gaussian = state
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_rng_from_dict(payload: dict[str, object]) -> tuple:
    return (
        str(payload["name"]),
        np.array(payload["keys"], dtype=np.uint32),
        int(payload["position"]),
        int(payload["has_gauss"]),
        float(payload["cached_gaussian"]),
    )


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir() and (path / "state.json").is_file():
        return path
    pointer = path / "last.json" if path.is_dir() else path
    if pointer.suffix == ".safetensors":
        raise ValueError("bare or external weight files cannot resume scratch training")
    if not pointer.is_file():
        raise FileNotFoundError(f"Checkpoint pointer not found: {pointer}")
    payload = json.loads(pointer.read_text())
    target = payload.get("checkpoint")
    if not isinstance(target, str):
        raise ValueError("checkpoint pointer is malformed")
    return (pointer.parent / target).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="ScratchTrainingConfig JSON")
    parser.add_argument(
        "--resume",
        help="Trainer checkpoint directory, last.json, or run directory (never bare weights)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional attempt limit for a bounded run or test",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = ScratchTrainingConfig.from_json(args.config)
    trainer = ScratchTrainer(config, resume=args.resume)
    state = trainer.run(max_steps=args.max_steps)
    print(
        f"scratch training step={state.global_step} "
        f"accepted={state.accepted_updates} skipped={state.skipped_updates} "
        f"complete={trainer.complete}",
        flush=True,
    )


if __name__ == "__main__":
    main()
