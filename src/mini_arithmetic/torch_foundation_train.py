"""Explicit, portable PyTorch training for the complete scratch foundation.

This module is intentionally separate from conversational ``talk-to-train``.
It is the only PyTorch path that makes the foundation parameters trainable,
and it must be started explicitly from the command line.  Input model bundles,
tokenizers, and prepared corpora are checksum/structure verified before use.

Checkpoints contain canonical MLX-layout safetensors, a pickle-free AdamW
state, exact data cursors, and CPU/accelerator RNG state.  The model tensor can
therefore be loaded by either the PyTorch or MLX HybridLM implementation.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from typing import Any, Literal, Sequence
import uuid

import numpy as np

from .community_updates import publish_directory_no_replace
from .model_sizes import MODEL_SIZE_IDS
from .portable_bundle import (
    CONFIG_FILENAME,
    MANIFEST_FILENAME,
    PORTABLE_BUNDLE_FORMAT,
    PORTABLE_STATE_FORMAT,
    PORTABLE_TRAINED_INITIALIZATION_FORMAT,
    PORTABLE_TRAINING_PROVENANCE_FORMAT,
    TOKENIZER_DIRECTORY,
    WEIGHTS_FILENAME,
    PortableBundle,
    create_portable_bundle,
    load_portable_bundle,
)
from .scratch_tokenizer import (
    METADATA_FILENAME,
    ROLE_NAMES,
    TOKENIZER_FILENAME,
    ScratchTokenizer,
)
from .torch_hybrid import HybridLM, require_torch

try:  # Keep the base, backend-neutral installation importable without Torch.
    import torch
    import torch.nn.functional as functional
except Exception as error:  # pragma: no cover - exercised only on torch-free hosts.
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = error
else:
    _TORCH_IMPORT_ERROR = None


CHECKPOINT_FORMAT = "tinydog-torch-foundation-checkpoint-v1"
SCHEMA_VERSION = 1
DataFormat = Literal["audited_jsonl", "token_shards"]
Objective = Literal["foundation", "dialogue"]
DeviceName = Literal["auto", "cpu", "cuda", "xpu", "mps"]
ComputeDtype = Literal["auto", "bfloat16", "float32"]
_MAX_JSON_BYTES = 4 * 1024 * 1024
_CHECKPOINT_NAME_RE = re.compile(r"step-(?P<step>[0-9]{8})-[0-9a-f]{8}")


class FoundationTrainingError(ValueError):
    """The explicit foundation run or one of its artifacts is invalid."""


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


@dataclass(frozen=True)
class OptimizerConfig:
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if not all(
            _is_finite_number(value)
            for value in (
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                self.grad_clip,
            )
        ):
            raise ValueError("AdamW settings must be finite numbers")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("AdamW epsilon must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be positive")


@dataclass(frozen=True)
class StageConfig:
    """One full-foundation pretraining or assistant-only dialogue stage."""

    name: str
    objective: Objective
    train_path: str
    eval_path: str
    steps: int
    data_format: DataFormat = "token_shards"
    batch_size: int = 1
    sequence_length: int = 512
    peak_learning_rate: float = 3e-4
    warmup_steps: int = 100
    min_learning_rate_ratio: float = 0.1
    eval_every: int = 100
    eval_batches: int = 4
    save_every: int = 100
    report_every: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("stage name must not be empty")
        if self.objective not in {"foundation", "dialogue"}:
            raise ValueError("objective must be foundation or dialogue")
        if self.data_format not in {"audited_jsonl", "token_shards"}:
            raise ValueError("data_format must be audited_jsonl or token_shards")
        if (
            not isinstance(self.train_path, str)
            or not self.train_path
            or not isinstance(self.eval_path, str)
            or not self.eval_path
        ):
            raise ValueError("train_path and eval_path are required")
        if (
            self.data_format == "audited_jsonl"
            and Path(self.train_path).resolve() == Path(self.eval_path).resolve()
        ):
            raise ValueError("audited train and evaluation files must be distinct")
        integer_values = (
            self.steps,
            self.batch_size,
            self.sequence_length,
            self.warmup_steps,
            self.eval_every,
            self.eval_batches,
            self.save_every,
            self.report_every,
        )
        if not all(_is_integer(value) for value in integer_values):
            raise ValueError("stage counts must be integers")
        if min(self.steps, self.batch_size, self.sequence_length) <= 0:
            raise ValueError("steps, batch_size, and sequence_length must be positive")
        if (
            not _is_finite_number(self.peak_learning_rate)
            or self.peak_learning_rate <= 0.0
        ):
            raise ValueError("peak_learning_rate must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("warmup_steps must be non-negative and smaller than steps")
        if (
            not _is_finite_number(self.min_learning_rate_ratio)
            or not 0.0 < self.min_learning_rate_ratio <= 1.0
        ):
            raise ValueError("min_learning_rate_ratio must be in (0, 1]")
        if (
            min(self.eval_every, self.eval_batches, self.save_every, self.report_every)
            <= 0
        ):
            raise ValueError("evaluation, save, and report intervals must be positive")


@dataclass(frozen=True)
class FoundationTrainingConfig:
    """Portable, JSON-serializable full-foundation training decisions."""

    stages: tuple[StageConfig, ...]
    output_dir: str = "checkpoints/torch-foundation"
    seed: int = 42
    device: DeviceName = "auto"
    compute_dtype: ComputeDtype = "auto"
    gradient_accumulation_steps: int = 1
    deterministic: bool = True
    checkpoint_keep_last: int = 2
    checkpoint_milestone_every: int = 5_000
    optimizer: OptimizerConfig = OptimizerConfig()

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("at least one training stage is required")
        if len({stage.name for stage in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if not _is_integer(self.seed) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.device not in {"auto", "cpu", "cuda", "xpu", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, xpu, or mps")
        if self.compute_dtype not in {"auto", "bfloat16", "float32"}:
            raise ValueError("compute_dtype must be auto, bfloat16, or float32")
        if (
            not _is_integer(self.gradient_accumulation_steps)
            or self.gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be boolean")
        if not _is_integer(self.checkpoint_keep_last) or self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive")
        if (
            not _is_integer(self.checkpoint_milestone_every)
            or self.checkpoint_milestone_every < 0
        ):
            raise ValueError("checkpoint_milestone_every must be non-negative")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FoundationTrainingConfig:
        allowed = {
            "stages",
            "output_dir",
            "seed",
            "device",
            "compute_dtype",
            "gradient_accumulation_steps",
            "deterministic",
            "checkpoint_keep_last",
            "checkpoint_milestone_every",
            "optimizer",
        }
        extra = set(payload) - allowed
        if extra:
            raise ValueError(f"unknown foundation training fields: {sorted(extra)}")
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, list):
            raise ValueError("stages must be a JSON list")
        stages: list[StageConfig] = []
        for raw in raw_stages:
            if not isinstance(raw, dict):
                raise ValueError("each stage must be a JSON object")
            normalized = dict(raw)
            # Accept the names used by the existing MLX scratch-training plan.
            objective = normalized.get("objective")
            if objective == "pretrain":
                normalized["objective"] = "foundation"
            elif objective == "sft":
                normalized["objective"] = "dialogue"
            stages.append(StageConfig(**normalized))
        raw_optimizer = payload.get("optimizer", {})
        if not isinstance(raw_optimizer, dict):
            raise ValueError("optimizer must be a JSON object")
        # ``bias_correction`` is accepted from the existing MLX config because
        # PyTorch AdamW always applies it.
        raw_optimizer = dict(raw_optimizer)
        bias_correction = raw_optimizer.pop("bias_correction", True)
        if bias_correction is not True:
            raise ValueError("PyTorch AdamW requires bias correction")
        return cls(
            stages=tuple(stages),
            output_dir=payload.get("output_dir", "checkpoints/torch-foundation"),
            seed=payload.get("seed", 42),
            device=payload.get("device", "auto"),
            compute_dtype=payload.get("compute_dtype", "auto"),
            gradient_accumulation_steps=payload.get("gradient_accumulation_steps", 1),
            deterministic=payload.get("deterministic", True),
            checkpoint_keep_last=payload.get("checkpoint_keep_last", 2),
            checkpoint_milestone_every=payload.get("checkpoint_milestone_every", 5_000),
            optimizer=OptimizerConfig(**raw_optimizer),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> FoundationTrainingConfig:
        payload = _read_json(Path(path), "foundation training config")
        return cls.from_dict(payload)

    def fingerprint(self) -> str:
        # Output location is operational, not a mathematical training choice.
        payload = asdict(self)
        payload.pop("output_dir", None)
        payload.pop("checkpoint_keep_last", None)
        payload.pop("checkpoint_milestone_every", None)
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class TokenizedSample:
    tokens: tuple[int, ...]
    target_mask: tuple[bool, ...]
    source_record: int

    def __post_init__(self) -> None:
        if len(self.tokens) < 2 or len(self.tokens) != len(self.target_mask):
            raise ValueError("tokens and target mask must have equal length >= 2")
        if not any(self.target_mask[1:]):
            raise ValueError("sample contains no supervised target")


@dataclass(frozen=True)
class ShardFoundationSample:
    """A lazy bounded view so large prepared shards are not copied into RAM."""

    payload: Any
    bos_id: int
    eos_id: int
    source_record: int

    @property
    def tokens(self) -> tuple[int, ...]:
        return (
            self.bos_id,
            *(int(value) for value in self.payload),
            self.eos_id,
        )

    @property
    def target_mask(self) -> tuple[bool, ...]:
        return (False, *(True for _ in range(len(self.payload) + 1)))


class PreparedDataset:
    """Checksum-bound samples from audited JSONL or prepared uint16 shards."""

    def __init__(
        self,
        path: str | Path,
        *,
        data_format: DataFormat,
        split: Literal["train", "validation"],
        objective: Objective,
        tokenizer: ScratchTokenizer,
        sequence_length: int,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.pad_id = tokenizer.pad_id
        self.samples: list[TokenizedSample | ShardFoundationSample] = []
        self.record_ids: set[str] = set()
        self.content_hashes: set[str] = set()
        # Retain mmap owners for every lazy foundation slice.
        self._mapped_shards: list[Any] = []
        if data_format == "audited_jsonl":
            self.sha256 = self._load_jsonl(
                objective=objective,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
            )
        else:
            self.sha256 = self._load_shards(
                split=split,
                objective=objective,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
            )
        if not self.samples:
            raise FoundationTrainingError(
                f"prepared source contains no usable {split} samples: {self.path}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_jsonl(
        self,
        *,
        objective: Objective,
        tokenizer: ScratchTokenizer,
        sequence_length: int,
    ) -> str:
        if not self.path.is_file():
            raise FileNotFoundError(f"audited source not found: {self.path}")
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise FoundationTrainingError(
                        f"audited JSONL is not UTF-8 at {self.path}:{line_number}"
                    ) from error
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise FoundationTrainingError(
                        f"invalid JSON at {self.path}:{line_number}"
                    ) from error
                _validate_audited_record(record, line_number)
                self.record_ids.add(str(record["record_id"]))
                self.content_hashes.add(str(record["content_sha256"]))
                self.samples.extend(
                    _samples_from_record(
                        record,
                        objective=objective,
                        tokenizer=tokenizer,
                        sequence_length=sequence_length,
                        source_record=line_number,
                    )
                )
        return digest.hexdigest()

    def _load_shards(
        self,
        *,
        split: Literal["train", "validation"],
        objective: Objective,
        tokenizer: ScratchTokenizer,
        sequence_length: int,
    ) -> str:
        if self.path.is_file() and self.path.name.endswith(".idx.json"):
            index_paths = [self.path]
        else:
            shard_dir = (
                self.path / "shards" if (self.path / "shards").is_dir() else self.path
            )
            index_paths = sorted(shard_dir.glob(f"{split}-*.idx.json"))
        if not index_paths:
            raise FileNotFoundError(f"no prepared {split} shards found at {self.path}")
        digest = hashlib.sha256()
        source_record = 0
        tokenizer_sha = tokenizer.metadata["tokenizer_sha256"]
        for index_path in index_paths:
            binary_path = index_path.with_name(
                index_path.name.replace(".idx.json", ".bin")
            )
            if (
                index_path.is_symlink()
                or binary_path.is_symlink()
                or not binary_path.is_file()
            ):
                raise FoundationTrainingError(
                    f"prepared shard is missing or linked: {binary_path}"
                )
            index_bytes = index_path.read_bytes()
            digest.update(len(index_bytes).to_bytes(8, "big"))
            digest.update(index_bytes)
            binary_size = binary_path.stat().st_size
            digest.update(binary_size.to_bytes(8, "big"))
            _update_digest_from_file(digest, binary_path)
            try:
                index = json.loads(index_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FoundationTrainingError(
                    f"invalid shard index: {index_path}"
                ) from error
            if (
                not isinstance(index, dict)
                or index.get("format_version") != 1
                or index.get("dtype") != "uint16-le"
                or index.get("split") != split
                or index.get("tokenizer_sha256") != tokenizer_sha
            ):
                raise FoundationTrainingError(f"incompatible shard index: {index_path}")
            if binary_size % np.dtype("<u2").itemsize:
                raise FoundationTrainingError(
                    f"shard byte length is invalid: {binary_path}"
                )
            if binary_size:
                shard = np.memmap(binary_path, mode="r", dtype="<u2")
                self._mapped_shards.append(shard)
            else:
                shard = np.empty((0,), dtype="<u2")
            if len(shard) != index.get("token_count"):
                raise FoundationTrainingError(
                    f"shard token count differs: {binary_path}"
                )
            records = index.get("records")
            if not isinstance(records, list):
                raise FoundationTrainingError(
                    f"shard records are malformed: {index_path}"
                )
            previous_end = 0
            for record in records:
                if not isinstance(record, dict):
                    raise FoundationTrainingError(
                        f"shard record is malformed: {index_path}"
                    )
                offset = record.get("offset")
                length = record.get("length")
                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or isinstance(length, bool)
                    or not isinstance(length, int)
                    or offset != previous_end
                    or length < 2
                    or offset + length > len(shard)
                ):
                    raise FoundationTrainingError(
                        f"shard record bounds differ: {index_path}"
                    )
                previous_end = offset + length
                tokens = shard[offset:previous_end]
                if bool(np.any(tokens >= tokenizer.vocab_size)):
                    raise FoundationTrainingError(
                        f"shard token is out of range: {index_path}"
                    )
                source_record += 1
                if objective == "foundation":
                    self.samples.extend(
                        _lazy_foundation_shard_samples(
                            tokens,
                            tokenizer,
                            sequence_length,
                            source_record,
                        )
                    )
                    continue
                if record.get("record_type") != "dialogue":
                    raise FoundationTrainingError(
                        "dialogue stages may consume only dialogue shard records"
                    )
                boundaries = record.get("dialogue_boundaries")
                if not isinstance(boundaries, list):
                    raise FoundationTrainingError(
                        "dialogue shard boundaries are missing"
                    )
                self.samples.append(
                    _dialogue_sample(
                        tuple(int(value) for value in tokens),
                        boundaries,
                        tokenizer,
                        sequence_length,
                        source_record,
                    )
                )
            if previous_end != len(shard):
                raise FoundationTrainingError(
                    f"shard has unindexed token data: {binary_path}"
                )
        return digest.hexdigest()


@dataclass
class DataCursor:
    epoch: int = 0
    index: int = 0

    @classmethod
    def from_dict(cls, payload: object) -> DataCursor:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise FoundationTrainingError("data cursor must be an object")
        epoch = payload.get("epoch", 0)
        index = payload.get("index", 0)
        if not _is_integer(epoch) or not _is_integer(index):
            raise FoundationTrainingError("data cursor fields must be integers")
        cursor = cls(epoch=epoch, index=index)
        if cursor.epoch < 0 or cursor.index < 0:
            raise FoundationTrainingError("data cursor cannot be negative")
        return cursor


@dataclass(frozen=True)
class TrainingBatch:
    inputs: Any
    targets: Any
    loss_mask: Any
    loss_tokens: int


class DeterministicBatcher:
    """Hash-shuffled batches with an exactly resumable sample cursor."""

    def __init__(
        self,
        dataset: PreparedDataset,
        *,
        objective: Objective,
        batch_size: int,
        sequence_length: int,
        seed: int,
        stream_name: str,
        cursor: DataCursor | None = None,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.objective = objective
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.seed = seed
        self.stream_name = stream_name
        self.cursor = cursor or DataCursor()
        self.shuffle = shuffle
        if self.cursor.index > len(dataset):
            raise FoundationTrainingError("data cursor is outside the dataset")

    def _order(self, epoch: int) -> list[int]:
        indices = list(range(len(self.dataset)))
        if not self.shuffle:
            return indices
        prefix = f"{self.seed}:{self.stream_name}:{epoch}:".encode("utf-8")
        return sorted(
            indices,
            key=lambda index: hashlib.blake2b(
                prefix + str(index).encode("ascii"), digest_size=16
            ).digest(),
        )

    def _peek(self) -> TokenizedSample | ShardFoundationSample:
        if self.cursor.index == len(self.dataset):
            self.cursor.epoch += 1
            self.cursor.index = 0
        return self.dataset.samples[self._order(self.cursor.epoch)[self.cursor.index]]

    def _advance(self) -> None:
        self.cursor.index += 1

    def next_batch(self, device: Any) -> TrainingBatch:
        inputs = np.full(
            (self.batch_size, self.sequence_length),
            self.dataset.pad_id,
            dtype=np.int64,
        )
        targets = np.full_like(inputs, self.dataset.pad_id)
        loss_mask = np.zeros((self.batch_size, self.sequence_length), dtype=np.float32)
        for row in range(self.batch_size):
            if self.objective == "foundation":
                packed: list[int] = []
                while len(packed) <= self.sequence_length:
                    sample = self._peek()
                    sample_tokens = sample.tokens
                    # The first record contributes N-1 targets. Each later
                    # record contributes N, including the explicit EOS->BOS
                    # document-boundary transition in the packed stream.
                    additional_targets = len(sample_tokens) - (1 if not packed else 0)
                    if packed and additional_targets > self.sequence_length - (
                        len(packed) - 1
                    ):
                        break
                    if not packed and additional_targets > self.sequence_length:
                        raise FoundationTrainingError(
                            "sample exceeds configured sequence length"
                        )
                    self._advance()
                    packed.extend(sample_tokens)
                    if len(packed) - 1 == self.sequence_length:
                        break
                target_count = len(packed) - 1
                inputs[row, :target_count] = packed[:-1]
                targets[row, :target_count] = packed[1:]
                loss_mask[row, :target_count] = 1.0
                continue

            sample = self._peek()
            sample_tokens = sample.tokens
            target_count = len(sample_tokens) - 1
            if target_count > self.sequence_length:
                raise FoundationTrainingError(
                    "sample exceeds configured sequence length"
                )
            self._advance()
            inputs[row, :target_count] = sample_tokens[:-1]
            targets[row, :target_count] = sample_tokens[1:]
            loss_mask[row, :target_count] = sample.target_mask[1:]
        loss_tokens = int(loss_mask.sum())
        if loss_tokens <= 0:
            raise FoundationTrainingError("batch has no supervised tokens")
        return TrainingBatch(
            inputs=torch.from_numpy(inputs).to(device=device),
            targets=torch.from_numpy(targets).to(device=device),
            loss_mask=torch.from_numpy(loss_mask).to(device=device),
            loss_tokens=loss_tokens,
        )


@dataclass
class TrainerState:
    stage_index: int = 0
    stage_step: int = 0
    global_step: int = 0
    tokens_seen: int = 0
    cursors: dict[str, dict[str, int]] | None = None

    def __post_init__(self) -> None:
        if self.cursors is None:
            self.cursors = {}


def scheduled_learning_rate(stage: StageConfig, zero_based_step: int) -> float:
    if not 0 <= zero_based_step < stage.steps:
        raise ValueError("step lies outside stage")
    if stage.warmup_steps and zero_based_step < stage.warmup_steps:
        return stage.peak_learning_rate * (zero_based_step + 1) / stage.warmup_steps
    remaining = stage.steps - stage.warmup_steps
    progress = (
        1.0
        if remaining <= 1
        else (zero_based_step - stage.warmup_steps) / (remaining - 1)
    )
    ratio = stage.min_learning_rate_ratio + (
        1.0 - stage.min_learning_rate_ratio
    ) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return stage.peak_learning_rate * ratio


def _text_cross_entropy_sum(
    logits: Any,
    targets: Any,
    loss_mask: Any,
    active_class_mask: Any,
) -> Any:
    """Compute text CE after making inactive vocabulary rows impossible."""

    if (
        logits.ndim != 3
        or targets.shape != logits.shape[:2]
        or loss_mask.shape != targets.shape
        or active_class_mask.ndim != 1
        or active_class_mask.dtype != torch.bool
        or active_class_mask.shape[0] != logits.shape[-1]
    ):
        raise FoundationTrainingError("text-loss tensors have incompatible shapes")
    supervised = loss_mask > 0
    if bool(supervised.any()) and not bool(
        active_class_mask[targets[supervised]].all()
    ):
        raise FoundationTrainingError("inactive text token is a supervised target")
    masked_logits = logits.masked_fill(
        ~active_class_mask.view(1, 1, -1),
        float("-inf"),
    )
    per_token = functional.cross_entropy(
        masked_logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(loss_mask)
    return (per_token * loss_mask).sum()


class TorchFoundationTrainer:
    """Train every foundation parameter only when explicitly constructed."""

    def __init__(
        self,
        bundle_path: str | Path,
        config: FoundationTrainingConfig,
        *,
        resume: str | Path | None = None,
        device_override: DeviceName | None = None,
    ) -> None:
        require_torch()
        if torch is None:  # Keeps static checkers and unusual import failures honest.
            raise RuntimeError("PyTorch could not be imported") from _TORCH_IMPORT_ERROR
        self.bundle = load_portable_bundle(bundle_path)
        self.config = config
        self.config_fingerprint = config.fingerprint()
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = _resolve_device(device_override or config.device)
        self.compute_dtype = _resolve_compute_dtype(config.compute_dtype, self.device)
        _configure_determinism(config.deterministic)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        _seed_accelerator(self.device, config.seed)

        self.model = HybridLM(self.bundle.config)
        _load_canonical_weights_strict(self.model, self.bundle)
        self.model.to(device=self.device, dtype=torch.float32)
        self._active_text_class_mask = torch.ones(
            self.bundle.config.vocab_size,
            dtype=torch.bool,
            device=self.device,
        )
        inactive_text_ids = torch.tensor(
            self.bundle.tokenizer.inactive_text_token_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._active_text_class_mask[inactive_text_ids] = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        if not all(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError(
                "the explicit foundation trainer did not unfreeze every parameter"
            )
        self.model.train()

        optimizer_config = config.optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.stages[0].peak_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )
        self.datasets: dict[tuple[int, str], PreparedDataset] = {}
        self.source_hashes: dict[str, str] = {}
        for stage_index, stage in enumerate(config.stages):
            if stage.sequence_length > self.bundle.config.context_length:
                raise ValueError(
                    f"stage {stage.name!r} exceeds the model context length"
                )
            for split, path in (
                ("train", stage.train_path),
                ("validation", stage.eval_path),
            ):
                dataset = PreparedDataset(
                    path,
                    data_format=stage.data_format,
                    split=split,
                    objective=stage.objective,
                    tokenizer=self.bundle.tokenizer,
                    sequence_length=stage.sequence_length,
                )
                self.datasets[(stage_index, split)] = dataset
                self.source_hashes[f"{stage.name}:{split}"] = dataset.sha256
            if stage.data_format == "audited_jsonl":
                training = self.datasets[(stage_index, "train")]
                validation = self.datasets[(stage_index, "validation")]
                overlapping_ids = sorted(training.record_ids & validation.record_ids)
                overlapping_content = sorted(
                    training.content_hashes & validation.content_hashes
                )
                if overlapping_ids or overlapping_content:
                    raise FoundationTrainingError(
                        f"stage {stage.name!r} has train/evaluation leakage "
                        f"(record_ids={overlapping_ids[:4]}, "
                        f"content_sha256={overlapping_content[:4]})"
                    )
        self.state = TrainerState()
        self._last_saved_signature: tuple[int, int, int] | None = None
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self._write_provenance_once()
        if resume is not None:
            self.load_checkpoint(resume)

    @property
    def complete(self) -> bool:
        return self.state.stage_index >= len(self.config.stages)

    def _autocast(self) -> Any:
        if self.compute_dtype != torch.bfloat16:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def _cursor(self, stage: StageConfig) -> DataCursor:
        assert self.state.cursors is not None
        return DataCursor.from_dict(self.state.cursors.get(stage.name))

    def _record_cursor(self, stage: StageConfig, cursor: DataCursor) -> None:
        assert self.state.cursors is not None
        self.state.cursors[stage.name] = asdict(cursor)

    def _batcher(
        self, stage_index: int, *, evaluation: bool = False
    ) -> DeterministicBatcher:
        stage = self.config.stages[stage_index]
        return DeterministicBatcher(
            self.datasets[(stage_index, "validation" if evaluation else "train")],
            objective=stage.objective,
            batch_size=stage.batch_size,
            sequence_length=stage.sequence_length,
            seed=self.config.seed,
            stream_name=f"{stage.name}:{'eval' if evaluation else 'train'}",
            cursor=DataCursor() if evaluation else self._cursor(stage),
            shuffle=not evaluation,
        )

    def _loss_sum(self, batch: TrainingBatch) -> Any:
        with self._autocast():
            logits = self.model(batch.inputs)
        return _text_cross_entropy_sum(
            logits,
            batch.targets,
            batch.loss_mask,
            self._active_text_class_mask,
        )

    def evaluate(self, stage_index: int) -> float:
        stage = self.config.stages[stage_index]
        batcher = self._batcher(stage_index, evaluation=True)
        total_loss = 0.0
        total_tokens = 0
        self.model.eval()
        try:
            with torch.no_grad():
                for _ in range(stage.eval_batches):
                    batch = batcher.next_batch(self.device)
                    value = self._loss_sum(batch)
                    total_loss += float(value.detach().cpu())
                    total_tokens += batch.loss_tokens
        finally:
            self.model.train()
        return total_loss / max(1, total_tokens)

    def run(self, *, max_optimizer_steps: int | None = None) -> TrainerState:
        if max_optimizer_steps is not None and max_optimizer_steps < 0:
            raise ValueError("max_optimizer_steps must be non-negative")
        performed = 0
        while not self.complete and (
            max_optimizer_steps is None or performed < max_optimizer_steps
        ):
            stage_index = self.state.stage_index
            stage = self.config.stages[stage_index]
            batcher = self._batcher(stage_index)
            learning_rate = scheduled_learning_rate(stage, self.state.stage_step)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            self.optimizer.zero_grad(set_to_none=True)
            batches = [
                batcher.next_batch(self.device)
                for _ in range(self.config.gradient_accumulation_steps)
            ]
            self._record_cursor(stage, batcher.cursor)
            loss_tokens = sum(batch.loss_tokens for batch in batches)
            loss_sum_value = 0.0
            for batch in batches:
                loss_sum = self._loss_sum(batch)
                if not bool(torch.isfinite(loss_sum).detach().cpu()):
                    raise FloatingPointError("non-finite foundation loss")
                loss_sum_value += float(loss_sum.detach().cpu())
                (loss_sum / loss_tokens).backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.optimizer.grad_clip
            )
            if not bool(torch.isfinite(gradient_norm).detach().cpu()):
                raise FloatingPointError("non-finite foundation gradient norm")
            self.optimizer.step()
            self.state.global_step += 1
            self.state.stage_step += 1
            self.state.tokens_seen += loss_tokens
            performed += 1
            loss_value = loss_sum_value / loss_tokens
            if self.state.global_step % stage.report_every == 0:
                self._metric(
                    "train",
                    stage=stage.name,
                    loss=loss_value,
                    learning_rate=learning_rate,
                    gradient_norm=float(gradient_norm.detach().cpu()),
                    loss_tokens=loss_tokens,
                )
            if self.state.global_step % stage.eval_every == 0:
                self._metric(
                    "evaluation", stage=stage.name, loss=self.evaluate(stage_index)
                )
            stage_finished = self.state.stage_step >= stage.steps
            if stage_finished:
                self.state.stage_index += 1
                self.state.stage_step = 0
            if self.state.global_step % stage.save_every == 0 or stage_finished:
                self.save_checkpoint()
        if performed and self._last_saved_signature != self._state_signature():
            self.save_checkpoint()
        return self.state

    def _state_signature(self) -> tuple[int, int, int]:
        return (self.state.stage_index, self.state.stage_step, self.state.global_step)

    def _state_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trainer": asdict(self.state),
            "training_plan": {
                "stage_count": len(self.config.stages),
                "total_steps": sum(stage.steps for stage in self.config.stages),
            },
            "source_sha256": self.source_hashes,
            "rng": _capture_rng_state(self.device),
            "resolved_device_type": self.device.type,
            "resolved_compute_dtype": _dtype_name(self.compute_dtype),
        }

    def save_checkpoint(self) -> Path:
        """Atomically publish a canonical model and pickle-free resume state."""

        signature = self._state_signature()
        if self._last_saved_signature == signature:
            pointer = _read_json(self.output_dir / "last.json", "checkpoint pointer")
            return (self.output_dir / str(pointer["checkpoint"])).resolve()
        temporary = Path(
            tempfile.mkdtemp(prefix=".foundation-", dir=self.checkpoint_dir)
        )
        identifier = f"step-{self.state.global_step:08d}-{uuid.uuid4().hex[:8]}"
        destination = self.checkpoint_dir / identifier
        try:
            model_path = temporary / "model.safetensors"
            optimizer_path = temporary / "optimizer.safetensors"
            state_path = temporary / "state.json"
            from safetensors.torch import save_file

            canonical = {
                name: value.detach().cpu().contiguous()
                for name, value in self.model.canonical_state_dict().items()
            }
            save_file(
                canonical,
                str(model_path),
                metadata={"format": PORTABLE_STATE_FORMAT},
            )
            optimizer_tensors, optimizer_metadata = _serialize_optimizer(
                self.optimizer, self.model
            )
            save_file(
                optimizer_tensors,
                str(optimizer_path),
                metadata={"format": CHECKPOINT_FORMAT},
            )
            state_payload = self._state_payload()
            state_payload["optimizer"] = optimizer_metadata
            _write_json(state_path, state_payload)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "format": CHECKPOINT_FORMAT,
                "state_format": PORTABLE_STATE_FORMAT,
                "scratch_only": True,
                "external_weights": False,
                "foundation_trainable": True,
                "config_sha256": self.config_fingerprint,
                "bundle_weights_sha256": self.bundle.manifest["weights_sha256"],
                "tokenizer_sha256": self.bundle.tokenizer.metadata["tokenizer_sha256"],
                "model_sha256": _sha256_file(model_path),
                "optimizer_sha256": _sha256_file(optimizer_path),
                "state_sha256": _sha256_file(state_path),
                "global_step": self.state.global_step,
                "training_complete": self.complete,
            }
            _write_json(temporary / "manifest.json", manifest)
            for item in temporary.iterdir():
                _fsync_file(item)
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(self.checkpoint_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(destination.relative_to(self.output_dir)),
            "global_step": self.state.global_step,
        }
        _atomic_write_json(self.output_dir / "last.json", pointer)
        self._last_saved_signature = signature
        self._metric("checkpoint", checkpoint=pointer["checkpoint"])
        self._apply_checkpoint_retention(destination)
        return destination

    def _owned_checkpoint_record(self, candidate: Path) -> tuple[int, bool] | None:
        """Identify only checkpoints created by this exact training run."""

        match = _CHECKPOINT_NAME_RE.fullmatch(candidate.name)
        if match is None or candidate.is_symlink() or not candidate.is_dir():
            return None
        expected_files = {
            "manifest.json",
            "model.safetensors",
            "optimizer.safetensors",
            "state.json",
        }
        try:
            if {item.name for item in candidate.iterdir()} != expected_files:
                return None
            manifest = _read_json(candidate / "manifest.json", "checkpoint manifest")
        except (FileNotFoundError, OSError, FoundationTrainingError):
            return None
        step = int(match.group("step"))
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("format") != CHECKPOINT_FORMAT
            or manifest.get("config_sha256") != self.config_fingerprint
            or manifest.get("bundle_weights_sha256")
            != self.bundle.manifest["weights_sha256"]
            or manifest.get("global_step") != step
            or not isinstance(manifest.get("training_complete"), bool)
        ):
            return None
        return step, bool(manifest["training_complete"])

    def _apply_checkpoint_retention(self, latest: Path) -> None:
        """Keep resume/final/sparse milestones and remove owned older snapshots."""

        records: list[tuple[int, str, Path, bool]] = []
        for candidate in self.checkpoint_dir.iterdir():
            record = self._owned_checkpoint_record(candidate)
            if record is not None:
                step, complete = record
                records.append((step, candidate.name, candidate, complete))
        records.sort(key=lambda item: (item[0], item[1]))
        keep = {
            candidate
            for _, _, candidate, _ in records[-self.config.checkpoint_keep_last :]
        }
        keep.add(latest)
        keep.update(candidate for _, _, candidate, complete in records if complete)
        milestone_every = self.config.checkpoint_milestone_every
        if milestone_every:
            keep.update(
                candidate
                for step, _, candidate, _ in records
                if step > 0 and step % milestone_every == 0
            )
        removed = False
        checkpoint_root = self.checkpoint_dir.resolve()
        for _, _, candidate, _ in records:
            if candidate in keep:
                continue
            if (
                candidate.is_symlink()
                or candidate.resolve().parent != checkpoint_root
                or _CHECKPOINT_NAME_RE.fullmatch(candidate.name) is None
            ):
                raise FoundationTrainingError(
                    "checkpoint retention target became unsafe"
                )
            shutil.rmtree(candidate)
            removed = True
            self._metric(
                "checkpoint_pruned",
                checkpoint=str(candidate.relative_to(self.output_dir)),
            )
        if removed:
            _fsync_directory(self.checkpoint_dir)

    def load_checkpoint(self, path: str | Path) -> Path:
        checkpoint = _resolve_checkpoint(Path(path).expanduser().resolve())
        manifest = _read_json(checkpoint / "manifest.json", "checkpoint manifest")
        state = _read_json(checkpoint / "state.json", "checkpoint state")
        required = (
            checkpoint / "model.safetensors",
            checkpoint / "optimizer.safetensors",
        )
        if not all(item.is_file() for item in required):
            raise FoundationTrainingError("checkpoint tensor files are incomplete")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("format") != CHECKPOINT_FORMAT
            or manifest.get("state_format") != PORTABLE_STATE_FORMAT
            or manifest.get("scratch_only") is not True
            or manifest.get("external_weights") is not False
            or manifest.get("foundation_trainable") is not True
        ):
            raise FoundationTrainingError("checkpoint identity is incompatible")
        if state.get("schema_version") != SCHEMA_VERSION:
            raise FoundationTrainingError("checkpoint state schema is incompatible")
        expected_plan = {
            "stage_count": len(self.config.stages),
            "total_steps": sum(stage.steps for stage in self.config.stages),
        }
        if state.get("training_plan") != expected_plan:
            raise FoundationTrainingError("checkpoint training plan differs")
        expected = {
            "config_sha256": self.config_fingerprint,
            "bundle_weights_sha256": self.bundle.manifest["weights_sha256"],
            "tokenizer_sha256": self.bundle.tokenizer.metadata["tokenizer_sha256"],
            "model_sha256": _sha256_file(checkpoint / "model.safetensors"),
            "optimizer_sha256": _sha256_file(checkpoint / "optimizer.safetensors"),
            "state_sha256": _sha256_file(checkpoint / "state.json"),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise FoundationTrainingError(
                "checkpoint checksum or training identity differs"
            )
        if state.get("source_sha256") != self.source_hashes:
            raise FoundationTrainingError("checkpoint data provenance differs")
        if state.get("resolved_device_type") != self.device.type:
            raise FoundationTrainingError(
                "deterministic optimizer resume requires the same device type; "
                "the canonical model.safetensors remains portable"
            )
        if state.get("resolved_compute_dtype") != _dtype_name(self.compute_dtype):
            raise FoundationTrainingError("checkpoint compute dtype differs")
        _load_checkpoint_model(self.model, checkpoint / "model.safetensors")
        _restore_optimizer(
            self.optimizer,
            self.model,
            checkpoint / "optimizer.safetensors",
            state.get("optimizer"),
        )
        trainer = state.get("trainer")
        if not isinstance(trainer, dict):
            raise FoundationTrainingError("checkpoint trainer state is malformed")
        trainer_fields = ("stage_index", "stage_step", "global_step", "tokens_seen")
        if any(
            not _is_integer(trainer.get(field_name)) for field_name in trainer_fields
        ):
            raise FoundationTrainingError("checkpoint trainer counters are malformed")
        self.state = TrainerState(
            stage_index=trainer["stage_index"],
            stage_step=trainer["stage_step"],
            global_step=trainer["global_step"],
            tokens_seen=trainer["tokens_seen"],
            cursors=trainer.get("cursors"),
        )
        _validate_trainer_state(self.state, self.config)
        assert self.state.cursors is not None
        for stage_index, stage_config in enumerate(self.config.stages):
            if stage_config.name not in self.state.cursors:
                continue
            cursor = DataCursor.from_dict(self.state.cursors[stage_config.name])
            if cursor.index > len(self.datasets[(stage_index, "train")]):
                raise FoundationTrainingError(
                    "checkpoint data cursor exceeds its source"
                )
        if manifest.get("global_step") != self.state.global_step:
            raise FoundationTrainingError(
                "checkpoint manifest step differs from trainer state"
            )
        if manifest.get("training_complete") is not self.complete:
            raise FoundationTrainingError("checkpoint completion marker differs")
        _restore_rng_state(state.get("rng"), self.device)
        self.model.train()
        self._last_saved_signature = self._state_signature()
        return checkpoint

    def _metric(self, event: str, **payload: Any) -> None:
        record = {"event": event, "global_step": self.state.global_step, **payload}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _write_provenance_once(self) -> None:
        path = self.output_dir / "provenance.json"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "explicit_foundation_training": True,
            "invoked_by_talk_to_train": False,
            "scratch_only": True,
            "external_weights": False,
            "bundle": str(self.bundle.root),
            "bundle_weights_sha256": self.bundle.manifest["weights_sha256"],
            "tokenizer_sha256": self.bundle.tokenizer.metadata["tokenizer_sha256"],
            "training_config_sha256": self.config_fingerprint,
            "source_sha256": self.source_hashes,
        }
        if path.exists():
            if _read_json(path, "training provenance") != payload:
                raise FoundationTrainingError(
                    "output directory belongs to a different run"
                )
            return
        _atomic_write_json(path, payload)


def initialize_foundation_bundle(
    destination: str | Path,
    *,
    config_path: str | Path,
    tokenizer_path: str | Path,
    model_size: str,
    seed: int = 0,
    parameter_dtype: Literal["bfloat16", "float32"] | None = None,
) -> PortableBundle:
    """Create random weights around a corpus-trained, frozen tokenizer.

    This is the explicit foundation-training initialization path. Consumer
    bootstrap keeps its small built-in tokenizer so first start remains
    offline and zero-configuration.
    """

    require_torch()
    return create_portable_bundle(
        destination,
        config_path=config_path,
        backend="torch",
        model_size=model_size,
        seed=seed,
        parameter_dtype=parameter_dtype,
        tokenizer_path=tokenizer_path,
    )


def export_trained_portable_bundle(
    bundle_path: str | Path,
    checkpoint_path: str | Path,
    destination: str | Path,
) -> PortableBundle:
    """Publish a trained checkpoint as a new immutable portable bundle.

    The random-initialized parent bundle is never modified. The checkpoint is
    checksum-bound to that parent, strictly loaded into the Torch architecture,
    and copied only after its model/optimizer/state hashes and training
    provenance pass validation.
    """

    require_torch()
    parent = load_portable_bundle(bundle_path)
    checkpoint, manifest, state, parameter_dtype = _verify_export_checkpoint(
        parent, checkpoint_path
    )
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Portable model already exists at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=".tinydog-trained-", dir=target.parent)
    )
    try:
        tokenizer_dir = temporary / TOKENIZER_DIRECTORY
        tokenizer_dir.mkdir()
        shutil.copyfile(parent.root / CONFIG_FILENAME, temporary / CONFIG_FILENAME)
        shutil.copyfile(checkpoint / "model.safetensors", temporary / WEIGHTS_FILENAME)
        shutil.copyfile(
            parent.root / TOKENIZER_DIRECTORY / TOKENIZER_FILENAME,
            tokenizer_dir / TOKENIZER_FILENAME,
        )
        shutil.copyfile(
            parent.root / TOKENIZER_DIRECTORY / METADATA_FILENAME,
            tokenizer_dir / METADATA_FILENAME,
        )
        trainer = state["trainer"]
        source_hashes = state["source_sha256"]
        trained_manifest = {
            "schema_version": 1,
            "format": PORTABLE_BUNDLE_FORMAT,
            "state_format": PORTABLE_STATE_FORMAT,
            "initialization": PORTABLE_TRAINED_INITIALIZATION_FORMAT,
            "model_size": parent.model_size,
            "parameters": parent.config.expected_parameter_count(),
            "parameter_dtype": parameter_dtype,
            "scratch_only": True,
            "external_weights": False,
            "seed": parent.manifest["seed"],
            "created_by_backend": "torch",
            "config_file": CONFIG_FILENAME,
            "config_sha256": _sha256_file(temporary / CONFIG_FILENAME),
            "weights_file": WEIGHTS_FILENAME,
            "weights_sha256": _sha256_file(temporary / WEIGHTS_FILENAME),
            "tokenizer_file": f"{TOKENIZER_DIRECTORY}/{TOKENIZER_FILENAME}",
            "tokenizer_sha256": _sha256_file(tokenizer_dir / TOKENIZER_FILENAME),
            "tokenizer_metadata_file": f"{TOKENIZER_DIRECTORY}/{METADATA_FILENAME}",
            "tokenizer_metadata_sha256": _sha256_file(
                tokenizer_dir / METADATA_FILENAME
            ),
            "training": {
                "format": PORTABLE_TRAINING_PROVENANCE_FORMAT,
                "parent_weights_sha256": parent.manifest["weights_sha256"],
                "training_config_sha256": manifest["config_sha256"],
                "source_sha256": source_hashes,
                "global_step": trainer["global_step"],
                "tokens_seen": trainer["tokens_seen"],
                "training_complete": True,
                "stage_count": state["training_plan"]["stage_count"],
                "total_steps": state["training_plan"]["total_steps"],
            },
        }
        _write_json(temporary / MANIFEST_FILENAME, trained_manifest)
        for item in temporary.rglob("*"):
            if item.is_file():
                _fsync_file(item)
        _fsync_directory(tokenizer_dir)
        _fsync_directory(temporary)
        # Detect any mutation of the parent between verification and publish.
        reverified = load_portable_bundle(parent.root)
        if reverified.manifest["weights_sha256"] != parent.manifest["weights_sha256"]:
            raise FoundationTrainingError("parent bundle changed during export")
        publish_directory_no_replace(temporary, target)
        temporary = None
        _fsync_directory(target.parent)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return load_portable_bundle(target)


def _verify_export_checkpoint(
    parent: PortableBundle,
    checkpoint_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], Literal["bfloat16", "float32"]]:
    checkpoint = _resolve_checkpoint(Path(checkpoint_path).expanduser().resolve())
    if {item.name for item in checkpoint.iterdir()} != {
        "manifest.json",
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
    }:
        raise FoundationTrainingError("checkpoint directory contents differ")
    manifest = _read_json(checkpoint / "manifest.json", "checkpoint manifest")
    state = _read_json(checkpoint / "state.json", "checkpoint state")
    expected_manifest_fields = {
        "schema_version",
        "format",
        "state_format",
        "scratch_only",
        "external_weights",
        "foundation_trainable",
        "config_sha256",
        "bundle_weights_sha256",
        "tokenizer_sha256",
        "model_sha256",
        "optimizer_sha256",
        "state_sha256",
        "global_step",
        "training_complete",
    }
    if set(manifest) != expected_manifest_fields or (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("format") != CHECKPOINT_FORMAT
        or manifest.get("state_format") != PORTABLE_STATE_FORMAT
        or manifest.get("scratch_only") is not True
        or manifest.get("external_weights") is not False
        or manifest.get("foundation_trainable") is not True
    ):
        raise FoundationTrainingError("checkpoint export identity is incompatible")
    if (
        manifest.get("bundle_weights_sha256") != parent.manifest["weights_sha256"]
        or manifest.get("tokenizer_sha256")
        != parent.tokenizer.metadata["tokenizer_sha256"]
    ):
        raise FoundationTrainingError("checkpoint belongs to another parent bundle")
    checks = {
        "model_sha256": checkpoint / "model.safetensors",
        "optimizer_sha256": checkpoint / "optimizer.safetensors",
        "state_sha256": checkpoint / "state.json",
    }
    if any(
        manifest.get(field_name) != _sha256_file(path)
        for field_name, path in checks.items()
    ):
        raise FoundationTrainingError("checkpoint export artifact checksum differs")
    config_sha = manifest.get("config_sha256")
    if not _is_sha256(config_sha):
        raise FoundationTrainingError("checkpoint training config digest is invalid")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise FoundationTrainingError("checkpoint export state schema differs")
    source_hashes = state.get("source_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(
            not isinstance(name, str) or not name or not _is_sha256(value)
            for name, value in source_hashes.items()
        )
    ):
        raise FoundationTrainingError("checkpoint source provenance is invalid")
    trainer = state.get("trainer")
    if not isinstance(trainer, dict):
        raise FoundationTrainingError("checkpoint trainer provenance is missing")
    if any(
        not _is_integer(trainer.get(field_name))
        for field_name in ("stage_index", "stage_step", "global_step", "tokens_seen")
    ):
        raise FoundationTrainingError("checkpoint trainer counters are malformed")
    for field_name in ("global_step", "tokens_seen"):
        value = trainer.get(field_name)
        if not _is_integer(value) or value <= 0:
            raise FoundationTrainingError(f"checkpoint {field_name} is invalid")
    if manifest.get("global_step") != trainer["global_step"]:
        raise FoundationTrainingError("checkpoint export step differs")
    plan = state.get("training_plan")
    if (
        not isinstance(plan, dict)
        or set(plan) != {"stage_count", "total_steps"}
        or not _is_integer(plan.get("stage_count"))
        or plan["stage_count"] <= 0
        or not _is_integer(plan.get("total_steps"))
        or plan["total_steps"] <= 0
        or trainer.get("stage_index") != plan["stage_count"]
        or trainer.get("stage_step") != 0
        or trainer["global_step"] != plan["total_steps"]
        or manifest.get("training_complete") is not True
    ):
        raise FoundationTrainingError(
            "only a checkpoint that completed every configured stage may be exported"
        )

    from safetensors import safe_open

    with safe_open(
        str(checkpoint / "model.safetensors"), framework="pt", device="cpu"
    ) as header:
        if header.metadata() != {"format": PORTABLE_STATE_FORMAT}:
            raise FoundationTrainingError("checkpoint model metadata is not portable")
        dtypes = {header.get_slice(name).get_dtype() for name in header.keys()}
    if dtypes == {"F32"}:
        parameter_dtype: Literal["bfloat16", "float32"] = "float32"
    elif dtypes == {"BF16"}:
        parameter_dtype = "bfloat16"
    else:
        raise FoundationTrainingError(
            "checkpoint model tensors have mixed/unsupported dtypes"
        )
    verifier = HybridLM(parent.config)
    _load_checkpoint_model(verifier, checkpoint / "model.safetensors")
    if verifier.parameter_count() != parent.config.expected_parameter_count():
        raise FoundationTrainingError("checkpoint model parameter count differs")
    return checkpoint, manifest, state, parameter_dtype


def _validate_audited_record(record: object, line_number: int) -> None:
    if not isinstance(record, dict):
        raise FoundationTrainingError(f"audited line {line_number} must be an object")
    required = {
        "type",
        "source",
        "license",
        "provenance",
        "record_id",
        "content_sha256",
    }
    if required - set(record):
        raise FoundationTrainingError(
            f"audited line {line_number} lacks provenance fields"
        )
    if record.get("type") not in {"document", "dialogue"}:
        raise FoundationTrainingError(f"audited line {line_number} has an invalid type")
    for field_name in ("source", "license", "record_id"):
        if (
            not isinstance(record.get(field_name), str)
            or not record[field_name].strip()
        ):
            raise FoundationTrainingError(
                f"audited line {line_number} has invalid {field_name}"
            )
    if record.get("provenance") in (None, "", [], {}):
        raise FoundationTrainingError(
            f"audited line {line_number} has empty provenance"
        )
    if record["type"] == "document":
        if not isinstance(record.get("text"), str) or not record["text"]:
            raise FoundationTrainingError(
                f"audited line {line_number} has no document text"
            )
        canonical = record["text"]
    else:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise FoundationTrainingError(f"audited line {line_number} has no dialogue")
        canonical_messages: list[list[str]] = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ROLE_NAMES
                or not isinstance(message.get("content"), str)
                or not message["content"]
            ):
                raise FoundationTrainingError(
                    f"audited line {line_number} has invalid dialogue"
                )
            canonical_messages.append([message["role"], message["content"]])
        canonical = json.dumps(
            canonical_messages,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    checksum = record.get("content_sha256")
    actual = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()
    if checksum != actual:
        raise FoundationTrainingError(f"audited line {line_number} checksum differs")


def _samples_from_record(
    record: dict[str, Any],
    *,
    objective: Objective,
    tokenizer: ScratchTokenizer,
    sequence_length: int,
    source_record: int,
) -> list[TokenizedSample]:
    if objective == "dialogue":
        if record["type"] != "dialogue":
            raise FoundationTrainingError(
                "dialogue stages may contain only dialogue records"
            )
        tokens, boundaries = tokenizer.encode_dialogue_with_boundaries(
            record["messages"]
        )
        return [
            _dialogue_sample(
                tuple(tokens), boundaries, tokenizer, sequence_length, source_record
            )
        ]
    tokens = (
        tokenizer.encode(record["text"], bos=True, eos=True)
        if record["type"] == "document"
        else tokenizer.encode_dialogue(record["messages"], bos=True, eos=True)
    )
    return _foundation_samples(tuple(tokens), tokenizer, sequence_length, source_record)


def _foundation_samples(
    tokens: Sequence[int],
    tokenizer: ScratchTokenizer,
    sequence_length: int,
    source_record: int,
) -> list[TokenizedSample]:
    if (
        len(tokens) < 2
        or tokens[0] != tokenizer.bos_id
        or tokens[-1] != tokenizer.eos_id
    ):
        raise FoundationTrainingError("foundation record lacks BOS/EOS boundaries")
    payload = list(tokens[1:-1])
    payload_size = max(1, sequence_length - 1)
    chunks = [
        payload[index : index + payload_size]
        for index in range(0, len(payload), payload_size)
    ]
    if not chunks:
        chunks = [[]]
    result: list[TokenizedSample] = []
    for chunk in chunks:
        bounded = (tokenizer.bos_id, *chunk, tokenizer.eos_id)
        result.append(
            TokenizedSample(
                tokens=bounded,
                target_mask=(False, *(True for _ in range(len(bounded) - 1))),
                source_record=source_record,
            )
        )
    return result


def _lazy_foundation_shard_samples(
    tokens: Any,
    tokenizer: ScratchTokenizer,
    sequence_length: int,
    source_record: int,
) -> list[ShardFoundationSample]:
    """Index mmap-backed payload chunks without copying the prepared corpus."""

    if (
        len(tokens) < 2
        or int(tokens[0]) != tokenizer.bos_id
        or int(tokens[-1]) != tokenizer.eos_id
    ):
        raise FoundationTrainingError(
            "foundation shard record lacks BOS/EOS boundaries"
        )
    payload = tokens[1:-1]
    payload_size = max(1, sequence_length - 1)
    if not len(payload):
        return [
            ShardFoundationSample(
                payload=payload,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                source_record=source_record,
            )
        ]
    return [
        ShardFoundationSample(
            payload=payload[start : start + payload_size],
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            source_record=source_record,
        )
        for start in range(0, len(payload), payload_size)
    ]


def _dialogue_sample(
    tokens: Sequence[int],
    boundaries: Sequence[Any],
    tokenizer: ScratchTokenizer,
    sequence_length: int,
    source_record: int,
) -> TokenizedSample:
    if len(tokens) > sequence_length + 1:
        raise FoundationTrainingError(
            "dialogue exceeds sequence_length; prepare shorter turns"
        )
    mask = _assistant_target_mask(tokens, boundaries, tokenizer)
    return TokenizedSample(tuple(tokens), mask, source_record)


def _boundary_value(boundary: Any, name: str) -> Any:
    if isinstance(boundary, dict):
        return boundary.get(name)
    return getattr(boundary, name, None)


def _assistant_target_mask(
    tokens: Sequence[int],
    boundaries: Sequence[Any],
    tokenizer: ScratchTokenizer,
) -> tuple[bool, ...]:
    if (
        len(tokens) < 2
        or tokens[0] != tokenizer.bos_id
        or tokens[-1] != tokenizer.eos_id
    ):
        raise FoundationTrainingError("dialogue lacks BOS/EOS boundaries")
    mask = [False] * len(tokens)
    expected_role_offset = 1
    last_assistant_end: int | None = None
    for boundary in boundaries:
        role = _boundary_value(boundary, "role")
        role_offset = _boundary_value(boundary, "role_offset")
        content_offset = _boundary_value(boundary, "content_offset")
        content_length = _boundary_value(boundary, "content_length")
        end_turn_offset = _boundary_value(boundary, "end_turn_offset")
        if (
            role not in ROLE_NAMES
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    role_offset,
                    content_offset,
                    content_length,
                    end_turn_offset,
                )
            )
            or role_offset != expected_role_offset
            or content_offset != role_offset + 1
            or content_length < 0
            or end_turn_offset != content_offset + content_length
            or end_turn_offset >= len(tokens) - 1
            or tokens[role_offset] != tokenizer.role_id(role)
            or tokens[end_turn_offset] != tokenizer.end_turn_id
        ):
            raise FoundationTrainingError("dialogue boundaries are invalid")
        if role == "assistant":
            for position in range(role_offset, end_turn_offset + 1):
                mask[position] = True
            last_assistant_end = end_turn_offset
        expected_role_offset = end_turn_offset + 1
    if expected_role_offset != len(tokens) - 1:
        raise FoundationTrainingError("dialogue boundaries do not cover the record")
    if last_assistant_end == len(tokens) - 2:
        mask[-1] = True
    if not any(mask[1:]):
        raise FoundationTrainingError("dialogue contains no assistant target")
    return tuple(mask)


def _resolve_device(requested: DeviceName) -> Any:
    if torch is None:
        require_torch()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            requested = "xpu"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    availability = {
        "cuda": lambda: torch.cuda.is_available(),
        "xpu": lambda: hasattr(torch, "xpu") and torch.xpu.is_available(),
        "mps": lambda: torch.backends.mps.is_available(),
        "cpu": lambda: True,
    }
    if requested not in availability or not availability[requested]():
        raise FoundationTrainingError(
            f"requested PyTorch device is unavailable: {requested}"
        )
    return torch.device(requested)


def _resolve_compute_dtype(requested: ComputeDtype, device: Any) -> Any:
    if requested == "float32":
        return torch.float32
    supported = False
    if device.type == "cuda":
        supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    elif device.type == "xpu":
        supported = bool(getattr(torch.xpu, "is_bf16_supported", lambda: True)())
    elif device.type == "cpu":
        # Explicit CPU BF16 is useful on recent server CPUs; auto remains FP32.
        supported = requested == "bfloat16"
    if requested == "bfloat16" and not supported:
        raise FoundationTrainingError(
            f"bfloat16 compute is unavailable on {device.type}"
        )
    return torch.bfloat16 if supported and requested != "float32" else torch.float32


def _configure_determinism(enabled: bool) -> None:
    if torch is None:
        return
    torch.use_deterministic_algorithms(enabled)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def _seed_accelerator(device: Any, seed: int) -> None:
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "xpu" and hasattr(torch.xpu, "manual_seed_all"):
        torch.xpu.manual_seed_all(seed)


def _dtype_name(dtype: Any) -> str:
    return "bfloat16" if dtype == torch.bfloat16 else "float32"


def _load_canonical_weights_strict(model: HybridLM, bundle: PortableBundle) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(bundle.weights_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != PORTABLE_STATE_FORMAT:
        raise FoundationTrainingError("portable model state format differs")
    state = load_file(str(bundle.weights_path), device="cpu")
    result = model.load_canonical_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise FoundationTrainingError("portable model tensor keys differ")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise FoundationTrainingError("portable model contains non-finite weights")


def _load_checkpoint_model(model: HybridLM, path: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != PORTABLE_STATE_FORMAT:
        raise FoundationTrainingError("checkpoint model state format differs")
    state = load_file(str(path), device="cpu")
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise FoundationTrainingError("checkpoint contains non-finite model weights")
    result = model.load_canonical_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise FoundationTrainingError("checkpoint model tensor keys differ")


def _serialize_optimizer(
    optimizer: Any, model: HybridLM
) -> tuple[dict[str, Any], dict[str, Any]]:
    names = [name for name, _ in model.named_parameters()]
    parameters = [parameter for _, parameter in model.named_parameters()]
    tensors: dict[str, Any] = {}
    state_fields: dict[str, list[str]] = {}
    for index, (name, parameter) in enumerate(zip(names, parameters, strict=True)):
        state = optimizer.state.get(parameter, {})
        fields: list[str] = []
        for field_name, value in sorted(state.items()):
            if not isinstance(value, torch.Tensor):
                raise FoundationTrainingError("AdamW state contains a non-tensor value")
            key = f"state.{index}.{field_name}"
            tensors[key] = value.detach().cpu().contiguous()
            fields.append(field_name)
        state_fields[name] = fields
    metadata = {
        "parameter_names": names,
        "state_fields": state_fields,
        "group_learning_rates": [
            float(group["lr"]) for group in optimizer.param_groups
        ],
    }
    return tensors, metadata


def _restore_optimizer(
    optimizer: Any,
    model: HybridLM,
    tensor_path: Path,
    metadata: object,
) -> None:
    from safetensors import safe_open
    from safetensors.torch import load_file

    if not isinstance(metadata, dict):
        raise FoundationTrainingError("optimizer metadata is malformed")
    names_and_parameters = list(model.named_parameters())
    names = [name for name, _ in names_and_parameters]
    if metadata.get("parameter_names") != names:
        raise FoundationTrainingError("optimizer parameter order differs")
    state_fields = metadata.get("state_fields")
    rates = metadata.get("group_learning_rates")
    if not isinstance(state_fields, dict) or not isinstance(rates, list):
        raise FoundationTrainingError("optimizer metadata fields differ")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if (handle.metadata() or {}).get("format") != CHECKPOINT_FORMAT:
            raise FoundationTrainingError("optimizer tensor format differs")
    tensors = load_file(str(tensor_path), device="cpu")
    expected_keys: set[str] = set()
    optimizer.state.clear()
    for index, (name, parameter) in enumerate(names_and_parameters):
        fields = state_fields.get(name)
        if (
            not isinstance(fields, list)
            or not all(isinstance(item, str) for item in fields)
            or len(fields) != len(set(fields))
        ):
            raise FoundationTrainingError("optimizer state field map differs")
        restored: dict[str, Any] = {}
        for field_name in fields:
            key = f"state.{index}.{field_name}"
            expected_keys.add(key)
            if key not in tensors:
                raise FoundationTrainingError("optimizer state tensor is missing")
            value = tensors[key]
            if not bool(torch.isfinite(value).all()):
                raise FoundationTrainingError(
                    "optimizer state contains a non-finite tensor"
                )
            if field_name in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise FoundationTrainingError(
                        "optimizer moment shape or dtype differs"
                    )
                value = value.to(device=parameter.device)
            elif field_name == "step":
                if value.numel() != 1 or float(value.item()) < 0.0:
                    raise FoundationTrainingError("optimizer step tensor is malformed")
            else:
                raise FoundationTrainingError(
                    f"unexpected AdamW state field: {field_name}"
                )
            restored[field_name] = value
        if restored:
            optimizer.state[parameter] = restored
    if set(tensors) != expected_keys:
        raise FoundationTrainingError("optimizer tensor keys differ")
    if len(rates) != len(optimizer.param_groups):
        raise FoundationTrainingError("optimizer group count differs")
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        if not _is_finite_number(rate) or float(rate) < 0.0:
            raise FoundationTrainingError("optimizer learning rate is malformed")
        group["lr"] = float(rate)


def _capture_rng_state(device: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().tolist(),
    }
    if device.type == "cuda":
        payload["torch_accelerator"] = [
            item.cpu().tolist() for item in torch.cuda.get_rng_state_all()
        ]
    elif device.type == "xpu" and hasattr(torch.xpu, "get_rng_state_all"):
        payload["torch_accelerator"] = [
            item.cpu().tolist() for item in torch.xpu.get_rng_state_all()
        ]
    return payload


def _restore_rng_state(payload: object, device: Any) -> None:
    if not isinstance(payload, dict):
        raise FoundationTrainingError("checkpoint RNG state is malformed")
    try:
        random.setstate(_nested_tuple(payload.get("python")))
    except (TypeError, ValueError) as error:
        raise FoundationTrainingError(
            "checkpoint Python RNG state is malformed"
        ) from error
    cpu_state = payload.get("torch_cpu")
    if not isinstance(cpu_state, list):
        raise FoundationTrainingError("checkpoint CPU RNG state is malformed")
    torch.set_rng_state(torch.tensor(cpu_state, dtype=torch.uint8))
    accelerator = payload.get("torch_accelerator")
    if device.type in {"cuda", "xpu"}:
        if not isinstance(accelerator, list):
            raise FoundationTrainingError("checkpoint accelerator RNG state is missing")
        states = [torch.tensor(item, dtype=torch.uint8) for item in accelerator]
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(states)
        else:
            torch.xpu.set_rng_state_all(states)


def _validate_trainer_state(
    state: TrainerState, config: FoundationTrainingConfig
) -> None:
    if (
        min(state.stage_index, state.stage_step, state.global_step, state.tokens_seen)
        < 0
    ):
        raise FoundationTrainingError("trainer state cannot be negative")
    if state.stage_index > len(config.stages):
        raise FoundationTrainingError("trainer stage lies outside the plan")
    if state.stage_index == len(config.stages):
        if state.stage_step != 0:
            raise FoundationTrainingError("completed trainer has a partial stage")
    elif state.stage_step >= config.stages[state.stage_index].steps:
        raise FoundationTrainingError("trainer step lies outside the active stage")
    if not isinstance(state.cursors, dict):
        raise FoundationTrainingError("trainer cursors are malformed")
    expected_global_step = (
        sum(stage.steps for stage in config.stages[: state.stage_index])
        + state.stage_step
    )
    if state.global_step != expected_global_step:
        raise FoundationTrainingError(
            "global step does not match completed stage updates"
        )
    allowed_cursors = {stage.name for stage in config.stages}
    if set(state.cursors) - allowed_cursors:
        raise FoundationTrainingError("trainer state contains an unknown data cursor")
    for cursor in state.cursors.values():
        DataCursor.from_dict(cursor)


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise FoundationTrainingError(f"{label} is missing: {path}") from error
    if not raw or len(raw) > _MAX_JSON_BYTES:
        raise FoundationTrainingError(f"{label} size is invalid")
    try:

        def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise FoundationTrainingError(f"{label} contains duplicate fields")
                result[key] = item
            return result

        value = json.loads(
            raw,
            object_pairs_hook=exact_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                FoundationTrainingError(f"{label} contains a non-finite number: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FoundationTrainingError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise FoundationTrainingError(f"{label} must contain one object")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_canonical_json(payload))


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # Directory handles are not available on every Windows setup.
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest_from_file(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        if path.name != "last.json":
            raise FoundationTrainingError(
                "resume requires a checkpoint directory or last.json"
            )
        root = path.parent
        pointer = _read_json(path, "checkpoint pointer")
    elif path.is_dir() and (path / "last.json").is_file():
        root = path
        pointer = _read_json(path / "last.json", "checkpoint pointer")
    elif path.is_dir():
        return path
    else:
        raise FoundationTrainingError("resume path does not exist")
    relative = pointer.get("checkpoint")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise FoundationTrainingError("checkpoint pointer is malformed")
    checkpoint = (root / relative).resolve()
    if root.resolve() not in checkpoint.parents:
        raise FoundationTrainingError("checkpoint pointer leaves its run directory")
    return checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly train all tinydog.ai foundation weights with PyTorch. "
            "This command is never called by talk-to-train."
        )
    )
    parser.add_argument(
        "--bundle", required=True, help="verified portable model directory"
    )
    parser.add_argument("--config", required=True, help="foundation training JSON plan")
    parser.add_argument(
        "--resume", help="run directory, last.json, or checkpoint directory"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu", "mps"))
    parser.add_argument(
        "--max-steps",
        type=int,
        help="stop after this many optimizer steps (for controlled runs/testing)",
    )
    return parser.parse_args(argv)


def parse_init_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create random scratch weights with a corpus-trained frozen tokenizer."
    )
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--model-config", required=True, help="portable HybridConfig JSON"
    )
    parser.add_argument(
        "--tokenizer", required=True, help="frozen ScratchTokenizer directory"
    )
    parser.add_argument("--model-size", choices=MODEL_SIZE_IDS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parameter-dtype", choices=("bfloat16", "float32"))
    return parser.parse_args(argv)


def init_main(argv: list[str] | None = None) -> int:
    args = parse_init_args(argv)
    bundle = initialize_foundation_bundle(
        args.destination,
        config_path=args.model_config,
        tokenizer_path=args.tokenizer,
        model_size=args.model_size,
        seed=args.seed,
        parameter_dtype=args.parameter_dtype,
    )
    print(
        json.dumps(
            {
                "bundle": str(bundle.root),
                "model_size": bundle.model_size,
                "parameters": bundle.manifest["parameters"],
                "tokenizer_sha256": bundle.manifest["tokenizer_sha256"],
                "weights_sha256": bundle.manifest["weights_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def parse_export_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a verified foundation checkpoint as a new portable bundle."
    )
    parser.add_argument(
        "--bundle", required=True, help="unchanged parent portable bundle"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="run, last.json, or checkpoint"
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="new portable-{size} directory; it must not already exist",
    )
    return parser.parse_args(argv)


def export_main(argv: list[str] | None = None) -> int:
    args = parse_export_args(argv)
    bundle = export_trained_portable_bundle(
        args.bundle,
        args.checkpoint,
        args.destination,
    )
    print(
        json.dumps(
            {
                "bundle": str(bundle.root),
                "model_size": bundle.model_size,
                "initialization": bundle.manifest["initialization"],
                "global_step": bundle.manifest["training"]["global_step"],
                "weights_sha256": bundle.manifest["weights_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = FoundationTrainingConfig.from_json(args.config)
    trainer = TorchFoundationTrainer(
        args.bundle,
        config,
        resume=args.resume,
        device_override=args.device,
    )
    trainer.run(max_optimizer_steps=args.max_steps)
    print(
        json.dumps(
            {
                "complete": trainer.complete,
                "global_step": trainer.state.global_step,
                "tokens_seen": trainer.state.tokens_seen,
                "device": trainer.device.type,
                "compute_dtype": _dtype_name(trainer.compute_dtype),
                "checkpoint": str(trainer.output_dir / "last.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_FORMAT",
    "DataCursor",
    "DeterministicBatcher",
    "FoundationTrainingConfig",
    "FoundationTrainingError",
    "OptimizerConfig",
    "PreparedDataset",
    "StageConfig",
    "TorchFoundationTrainer",
    "TrainingBatch",
    "_assistant_target_mask",
    "export_trained_portable_bundle",
    "initialize_foundation_bundle",
    "scheduled_learning_rate",
]
