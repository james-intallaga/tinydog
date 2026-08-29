"""Resumable scratch training for the native realtime multimodal model.

Only project-created random initialization or checkpoints emitted by this
trainer are accepted.  Runtime-readiness gates remain false for toy runs: a
gate needs a completed stage, enough optimizer updates and held-out records,
and a configured held-out loss threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
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

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np

from .hybrid_train import (
    FP32AdamW,
    OptimizerConfig,
    clip_grad_norm_fp32,
)
from .multimodal import MultimodalConfig, ScratchMultimodalLM
from .native_data import NativeShardDataset
from .native_objectives import (
    NativeConversationBatch,
    NativeLossWeights,
    NativeStage,
    native_conversation_loss,
)


NATIVE_TRAINING_SCHEMA_VERSION = 1
NATIVE_RUNTIME_FORMAT = "mlm-native-realtime-v1"
NATIVE_GATE_NAMES = (
    "audio_codec",
    "speech_understanding",
    "vision",
    "talker",
    "conversation",
)
MIN_GATE_UPDATES = 1_000
MIN_GATE_VALIDATION_RECORDS = 128
MAX_NATIVE_JSON_BYTES = 4 * 1024 * 1024
NATIVE_QUALITY_THRESHOLDS: dict[str, dict[str, tuple[float, str]]] = {
    "audio_codec": {
        "reconstruction_snr_db": (15.0, "at_least"),
    },
    "speech_understanding": {
        "masked_text_accuracy": (0.80, "at_least"),
    },
    "vision": {
        "masked_text_accuracy": (0.70, "at_least"),
        "frame_ablation_loss_delta": (0.05, "at_least"),
    },
    "talker": {
        "model_code_accuracy": (0.60, "at_least"),
    },
    "conversation": {
        "masked_text_accuracy": (0.80, "at_least"),
        "model_code_accuracy": (0.60, "at_least"),
    },
}
_SAFE_STAGE_NAME = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@dataclass(frozen=True)
class NativeStageConfig:
    name: str
    objective: NativeStage
    train_manifest: str
    validation_manifest: str
    steps: int
    max_validation_loss: float
    peak_learning_rate: float = 1e-4
    warmup_steps: int = 10
    min_learning_rate_ratio: float = 0.1
    eval_every: int = 100
    eval_records: int = 16
    save_every: int = 100
    report_every: int = 10

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or _SAFE_STAGE_NAME.fullmatch(self.name) is None
        ):
            raise ValueError("native stage name is invalid")
        if self.objective not in {*NATIVE_GATE_NAMES, "joint"}:
            raise ValueError("unsupported native stage objective")
        if not self.train_manifest or not self.validation_manifest:
            raise ValueError("native stage requires train and validation manifests")
        if (
            Path(self.train_manifest).resolve()
            == Path(self.validation_manifest).resolve()
        ):
            raise ValueError("native train and validation manifests must be distinct")
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps <= 0
        ):
            raise ValueError("native stage steps must be positive")
        if (
            isinstance(self.max_validation_loss, bool)
            or not isinstance(self.max_validation_loss, (int, float))
            or not math.isfinite(self.max_validation_loss)
            or self.max_validation_loss <= 0.0
        ):
            raise ValueError("max_validation_loss must be finite and positive")
        if (
            isinstance(self.peak_learning_rate, bool)
            or not isinstance(self.peak_learning_rate, (int, float))
            or not math.isfinite(self.peak_learning_rate)
            or self.peak_learning_rate <= 0.0
        ):
            raise ValueError("native peak learning rate must be positive")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 0
            or self.warmup_steps >= self.steps
        ):
            raise ValueError(
                "native warmup must be non-negative and smaller than steps"
            )
        if (
            isinstance(self.min_learning_rate_ratio, bool)
            or not isinstance(self.min_learning_rate_ratio, (int, float))
            or not math.isfinite(self.min_learning_rate_ratio)
            or not 0.0 < self.min_learning_rate_ratio <= 1.0
        ):
            raise ValueError("native minimum learning-rate ratio must be in (0, 1]")
        intervals = (
            self.eval_every,
            self.eval_records,
            self.save_every,
            self.report_every,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in intervals
        ):
            raise ValueError("native evaluation/save/report settings must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NativeStageConfig":
        return cls(**value)


@dataclass(frozen=True)
class NativeTrainingConfig:
    model_config: str
    tokenizer_sha256: str
    stages: tuple[NativeStageConfig, ...]
    output_dir: str = "checkpoints/native-scratch"
    seed: int = 42
    parameter_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    allow_runtime_readiness: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if not Path(self.model_config).expanduser().is_file():
            raise FileNotFoundError(
                f"native model config not found: {self.model_config}"
            )
        if len(self.tokenizer_sha256) != 64:
            raise ValueError(
                "native tokenizer_sha256 must contain 64 hexadecimal characters"
            )
        try:
            int(self.tokenizer_sha256, 16)
        except ValueError as error:
            raise ValueError("native tokenizer_sha256 must be hexadecimal") from error
        if self.tokenizer_sha256 != self.tokenizer_sha256.lower():
            raise ValueError("native tokenizer_sha256 must use lowercase hexadecimal")
        if not self.stages:
            raise ValueError("native training needs at least one stage")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("native stage names must be unique")
        gate_objectives = [
            stage.objective
            for stage in self.stages
            if stage.objective in NATIVE_GATE_NAMES
        ]
        if len(gate_objectives) != len(set(gate_objectives)):
            raise ValueError("each native readiness gate may have only one stage")
        if self.parameter_dtype not in {"bfloat16", "float32"}:
            raise ValueError("native parameter_dtype must be bfloat16 or float32")
        if not isinstance(self.allow_runtime_readiness, bool):
            raise ValueError("allow_runtime_readiness must be a boolean")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("native seed must be an integer")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("native output_dir must be a non-empty path")
        for name in ("beta1", "beta2", "eps", "weight_decay", "grad_clip"):
            value = getattr(self.optimizer, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"native optimizer {name} must be finite")
        if type(self.optimizer.bias_correction) is not bool:
            raise ValueError("native optimizer bias_correction must be a boolean")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NativeTrainingConfig":
        required = {"model_config", "tokenizer_sha256", "stages"}
        allowed = required | {
            "output_dir",
            "seed",
            "parameter_dtype",
            "allow_runtime_readiness",
            "optimizer",
        }
        if set(value) - allowed or required - set(value):
            raise ValueError("native training configuration fields are incompatible")
        if not isinstance(value["stages"], list) or not value["stages"]:
            raise ValueError("native training stages must be a non-empty list")
        if not isinstance(value.get("optimizer", {}), dict):
            raise ValueError("native optimizer configuration must be an object")
        return cls(
            model_config=str(value["model_config"]),
            tokenizer_sha256=str(value["tokenizer_sha256"]),
            stages=tuple(NativeStageConfig.from_dict(item) for item in value["stages"]),
            output_dir=str(value.get("output_dir", "checkpoints/native-scratch")),
            seed=int(value.get("seed", 42)),
            parameter_dtype=value.get("parameter_dtype", "bfloat16"),
            allow_runtime_readiness=value.get("allow_runtime_readiness", False),
            optimizer=OptimizerConfig(**value.get("optimizer", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "NativeTrainingConfig":
        config_path = Path(path).expanduser().resolve()
        payload = _read_json_object(config_path, "training configuration")
        # Relative model/data/output paths belong to the training config.
        base = config_path.parent
        payload = dict(payload)
        if not isinstance(payload.get("model_config"), str):
            raise ValueError("native model_config must be a path string")
        if not isinstance(payload.get("stages"), list):
            raise ValueError("native stages must be a list")
        if any(not isinstance(item, dict) for item in payload["stages"]):
            raise ValueError("native stage configurations must be objects")
        if any(
            not isinstance(item.get(name), str)
            for item in payload["stages"]
            for name in ("train_manifest", "validation_manifest")
        ):
            raise ValueError("native stage manifests must be path strings")
        payload["model_config"] = str((base / payload["model_config"]).resolve())
        payload["output_dir"] = str(
            (base / payload.get("output_dir", "checkpoints/native-scratch")).resolve()
        )
        payload["stages"] = [
            {
                **item,
                "train_manifest": str((base / item["train_manifest"]).resolve()),
                "validation_manifest": str(
                    (base / item["validation_manifest"]).resolve()
                ),
            }
            for item in payload["stages"]
        ]
        return cls.from_dict(payload)

    def fingerprint(self) -> str:
        payload = {
            **asdict(self),
            "stages": [asdict(stage) for stage in self.stages],
            "model_config_sha256": _file_sha256(
                Path(self.model_config).expanduser().resolve()
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass
class NativeTrainerState:
    global_step: int = 0
    stage_index: int = 0
    stage_step: int = 0
    accepted_updates: int = 0
    skipped_updates: int = 0
    accepted_updates_by_stage: dict[str, int] = field(default_factory=dict)
    best_validation_losses: dict[str, float] = field(default_factory=dict)
    completed_stages: list[str] = field(default_factory=list)
    data_epochs: dict[str, int] = field(default_factory=dict)
    data_indices: dict[str, int] = field(default_factory=dict)
    quality: dict[str, dict[str, Any]] = field(default_factory=dict)
    gates: dict[str, dict[str, bool]] = field(
        default_factory=lambda: {
            name: {"trained": False, "validation_passed": False}
            for name in NATIVE_GATE_NAMES
        }
    )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NativeTrainerState":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("native checkpoint trainer state fields are incompatible")
        state = cls(**value)
        if set(state.gates) != set(NATIVE_GATE_NAMES):
            raise ValueError("native checkpoint gate state is incompatible")
        return state


@dataclass(frozen=True)
class NativeEvaluation:
    loss: float
    quality: dict[str, Any] | None


def _stage_learning_rate(stage: NativeStageConfig, step: int) -> float:
    peak = stage.peak_learning_rate
    floor = peak * stage.min_learning_rate_ratio
    if stage.warmup_steps and step < stage.warmup_steps:
        return peak * (step + 1) / stage.warmup_steps
    decay_steps = max(1, stage.steps - stage.warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - stage.warmup_steps) / decay_steps))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def _duplex_codec_loss(
    model: ScratchMultimodalLM, batch: NativeConversationBatch
) -> mx.array:
    batch.validate(model)
    user = model.audio_codec(batch.user_waveform).losses.total.astype(mx.float32)
    assistant = model.audio_codec(batch.model_waveform).losses.total.astype(mx.float32)
    return (user + assistant) * 0.5


def _native_stage_loss(
    model: ScratchMultimodalLM,
    batch: NativeConversationBatch,
    objective: NativeStage,
) -> mx.array:
    if objective == "audio_codec":
        return _duplex_codec_loss(model, batch)
    return native_conversation_loss(
        model, batch, NativeLossWeights.for_stage(objective)
    ).total


class NativeScratchTrainer:
    """Train staged native objectives and emit runtime-verifiable checkpoints."""

    def __init__(
        self,
        config: NativeTrainingConfig,
        *,
        resume: str | Path | None = None,
    ) -> None:
        self.config = config
        self.config_sha256 = config.fingerprint()
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.checkpoint_root = self.output_dir / "checkpoints"
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.model_config_path = Path(config.model_config).expanduser().resolve()
        self.model_config = MultimodalConfig.from_json(self.model_config_path)
        self.model_config_sha256 = _file_sha256(self.model_config_path)
        video_divisor = (
            self.model_config.video.patch_size
            * self.model_config.video.spatial_merge_size
        )
        if 320 % video_divisor or 192 % video_divisor:
            raise ValueError(
                "native model vision grid is incompatible with 320x192 realtime frames"
            )
        self.datasets: dict[str, tuple[NativeShardDataset, NativeShardDataset]] = {}
        for stage in config.stages:
            train = NativeShardDataset(
                stage.train_manifest,
                split="train",
                config=self.model_config,
                tokenizer_sha256=config.tokenizer_sha256,
            )
            validation = NativeShardDataset(
                stage.validation_manifest,
                split="validation",
                config=self.model_config,
                tokenizer_sha256=config.tokenizer_sha256,
            )
            self.datasets[stage.name] = train, validation
        all_train_content: set[str] = set()
        all_validation_content: set[str] = set()
        all_train_shards: set[str] = set()
        all_validation_shards: set[str] = set()
        all_train_ids: set[str] = set()
        all_validation_ids: set[str] = set()
        for train, validation in self.datasets.values():
            all_train_content.update(train.content_hashes)
            all_validation_content.update(validation.content_hashes)
            all_train_shards.update(train.shard_hashes)
            all_validation_shards.update(validation.shard_hashes)
            all_train_ids.update(train.record_ids)
            all_validation_ids.update(validation.record_ids)
        if (
            all_train_content & all_validation_content
            or all_train_shards & all_validation_shards
            or all_train_ids & all_validation_ids
        ):
            raise ValueError(
                "native training data leaks into the global validation holdout"
            )
        self.source_hashes = {
            f"{stage.name}:train": self.datasets[stage.name][0].sha256
            for stage in config.stages
        } | {
            f"{stage.name}:validation": self.datasets[stage.name][1].sha256
            for stage in config.stages
        }
        random.seed(config.seed)
        np.random.seed(config.seed)
        mx.random.seed(config.seed)
        self.model = ScratchMultimodalLM(self.model_config)
        self.model.set_dtype(
            mx.bfloat16 if config.parameter_dtype == "bfloat16" else mx.float32
        )
        self.state = NativeTrainerState()
        self.state.quality = self._quality_template()
        self.optimizer: FP32AdamW | None = None
        self._optimizer_stage: str | None = None
        self._last_saved_state_signature: str | None = None
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self._write_provenance()
        if resume is not None:
            self.load_checkpoint(resume)

    @property
    def complete(self) -> bool:
        return self.state.stage_index >= len(self.config.stages)

    def _write_provenance(self) -> None:
        value = {
            "schema_version": NATIVE_TRAINING_SCHEMA_VERSION,
            "scratch_initialization": True,
            "external_weights": False,
            "training_config_sha256": self.config_sha256,
            "tokenizer_sha256": self.config.tokenizer_sha256,
            "source_sha256": self.source_hashes,
            "model_config_sha256": self.model_config_sha256,
            "model_parameters": self.model.parameter_count(),
            "model_config": self.model_config.to_dict(),
            "allow_runtime_readiness": self.config.allow_runtime_readiness,
        }
        path = self.output_dir / "provenance.json"
        if path.exists():
            previous = _read_json_object(path, "training provenance")
            for key in (
                "training_config_sha256",
                "tokenizer_sha256",
                "source_sha256",
                "model_config_sha256",
                "allow_runtime_readiness",
            ):
                if previous.get(key) != value[key]:
                    raise ValueError("native output directory belongs to another run")
        else:
            _atomic_json(path, value)

    def _write_metric(self, event: str, **values: object) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": NATIVE_TRAINING_SCHEMA_VERSION,
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        with self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _configure_stage(self, stage: NativeStageConfig) -> None:
        if self._optimizer_stage == stage.name and self.optimizer is not None:
            return
        self.model.freeze()
        objective = stage.objective
        if objective == "audio_codec":
            modules = (self.model.audio_codec,)
        elif objective == "speech_understanding":
            modules = (
                self.model.thinker,
                self.model.audio_codec,
                self.model.audio_projection,
                self.model.audio_stream_embedding,
                self.model.time_input,
                self.model.time_output,
            )
        elif objective == "vision":
            modules = (
                self.model.thinker,
                self.model.video_encoder,
                self.model.time_input,
                self.model.time_output,
            )
        elif objective == "talker":
            modules = (self.model.audio_codec, self.model.talker)
        elif objective == "conversation":
            modules = (
                self.model.thinker,
                self.model.audio_projection,
                self.model.audio_stream_embedding,
                self.model.time_input,
                self.model.time_output,
                self.model.talker,
            )
        else:
            modules = (self.model,)
        for module in modules:
            module.unfreeze()
        self.optimizer = FP32AdamW(stage.peak_learning_rate, self.config.optimizer)
        self.optimizer.init(self.model.trainable_parameters())
        mx.eval(self.optimizer.state)
        self._optimizer_stage = stage.name

    def _order(self, stage: NativeStageConfig, length: int, epoch: int) -> list[int]:
        prefix = f"{self.config.seed}:{stage.name}:{epoch}:".encode()
        return sorted(
            range(length),
            key=lambda index: hashlib.blake2b(
                prefix + str(index).encode(), digest_size=16
            ).digest(),
        )

    def _next_training_batch(self, stage: NativeStageConfig) -> NativeConversationBatch:
        dataset, _ = self.datasets[stage.name]
        epoch = self.state.data_epochs.get(stage.name, 0)
        index = self.state.data_indices.get(stage.name, 0)
        order = self._order(stage, len(dataset), epoch)
        if index >= len(order):
            epoch += 1
            index = 0
            order = self._order(stage, len(dataset), epoch)
        batch = dataset.load(order[index])
        self.state.data_epochs[stage.name] = epoch
        self.state.data_indices[stage.name] = index + 1
        if stage.objective == "vision" and batch.frames is None:
            raise ValueError("vision stages require selected-frame tensors")
        return batch

    def _quality_template(self) -> dict[str, dict[str, Any]]:
        stages_by_gate = {
            stage.objective: stage
            for stage in self.config.stages
            if stage.objective in NATIVE_GATE_NAMES
        }
        quality: dict[str, dict[str, Any]] = {}
        for gate_name in NATIVE_GATE_NAMES:
            stage = stages_by_gate.get(gate_name)
            metrics: dict[str, dict[str, object]] = {
                "stage_loss": _quality_metric(
                    None,
                    None,
                    "at_most",
                )
            }
            for metric_name, (_threshold, direction) in NATIVE_QUALITY_THRESHOLDS[
                gate_name
            ].items():
                metrics[metric_name] = _quality_metric(None, None, direction)
            quality[gate_name] = {
                "stage": None if stage is None else stage.name,
                "validation_manifest_sha256": (
                    None if stage is None else self.datasets[stage.name][1].sha256
                ),
                "evaluated_optimizer_steps": None,
                "validation_records": 0,
                "minimum_validation_records": MIN_GATE_VALIDATION_RECORDS,
                "metrics": metrics,
                "passed": False,
            }
        return quality

    def _evaluate_stage(self, stage: NativeStageConfig) -> NativeEvaluation:
        _, dataset = self.datasets[stage.name]
        count = min(stage.eval_records, len(dataset))
        total_loss = 0.0
        text_correct = 0
        text_targets = 0
        model_audio_correct = 0
        model_audio_targets = 0
        signal_energy = 0.0
        reconstruction_error = 0.0
        ablation_delta = 0.0
        self.model.eval()
        try:
            for index in range(count):
                batch = dataset.load(index)
                if stage.objective == "vision" and batch.frames is None:
                    raise ValueError(
                        "vision validation requires selected-frame tensors"
                    )
                if stage.objective == "audio_codec":
                    user = self.model.audio_codec(batch.user_waveform)
                    assistant = self.model.audio_codec(batch.model_waveform)
                    loss = (
                        user.losses.total.astype(mx.float32)
                        + assistant.losses.total.astype(mx.float32)
                    ) * 0.5
                    signal = mx.sum(
                        batch.user_waveform.astype(mx.float32) ** 2
                    ) + mx.sum(batch.model_waveform.astype(mx.float32) ** 2)
                    error = mx.sum(
                        (
                            batch.user_waveform.astype(mx.float32)
                            - user.reconstruction.astype(mx.float32)
                        )
                        ** 2
                    ) + mx.sum(
                        (
                            batch.model_waveform.astype(mx.float32)
                            - assistant.reconstruction.astype(mx.float32)
                        )
                        ** 2
                    )
                    mx.eval(loss, signal, error)
                    total_loss += float(loss.item())
                    signal_energy += float(signal.item())
                    reconstruction_error += float(error.item())
                    continue

                result = native_conversation_loss(
                    self.model,
                    batch,
                    NativeLossWeights.for_stage(stage.objective),
                )
                mx.eval(result)
                total_loss += float(result.total.item())
                text_correct += result.text_correct
                text_targets += result.text_targets
                model_audio_correct += result.model_audio_correct
                model_audio_targets += result.audio_targets
                if stage.objective == "vision":
                    assert batch.frames is not None
                    ablated = replace(batch, frames=mx.zeros_like(batch.frames))
                    ablated_result = native_conversation_loss(
                        self.model,
                        ablated,
                        NativeLossWeights.for_stage("vision"),
                    )
                    mx.eval(ablated_result.text)
                    ablation_delta += float(
                        ablated_result.text.item() - result.text.item()
                    )
        finally:
            self.model.train()

        loss_value = total_loss / count
        if stage.objective not in NATIVE_GATE_NAMES:
            return NativeEvaluation(loss=loss_value, quality=None)
        metrics: dict[str, dict[str, object]] = {
            "stage_loss": _quality_metric(
                loss_value, stage.max_validation_loss, "at_most"
            )
        }
        if stage.objective == "audio_codec":
            epsilon = 1e-12
            snr_db = 10.0 * math.log10(
                (signal_energy + epsilon) / (reconstruction_error + epsilon)
            )
            metrics["reconstruction_snr_db"] = _quality_metric(
                snr_db,
                *NATIVE_QUALITY_THRESHOLDS["audio_codec"]["reconstruction_snr_db"],
            )
        if stage.objective in {"speech_understanding", "vision", "conversation"}:
            accuracy = text_correct / text_targets if text_targets else math.nan
            metrics["masked_text_accuracy"] = _quality_metric(
                accuracy,
                *NATIVE_QUALITY_THRESHOLDS[stage.objective]["masked_text_accuracy"],
            )
        if stage.objective == "vision":
            metrics["frame_ablation_loss_delta"] = _quality_metric(
                ablation_delta / count,
                *NATIVE_QUALITY_THRESHOLDS["vision"]["frame_ablation_loss_delta"],
            )
        if stage.objective in {"talker", "conversation"}:
            accuracy = (
                model_audio_correct / model_audio_targets
                if model_audio_targets
                else math.nan
            )
            metrics["model_code_accuracy"] = _quality_metric(
                accuracy,
                *NATIVE_QUALITY_THRESHOLDS[stage.objective]["model_code_accuracy"],
            )
        passed = count >= MIN_GATE_VALIDATION_RECORDS and all(
            metric["passed"] is True for metric in metrics.values()
        )
        return NativeEvaluation(
            loss=loss_value,
            quality={
                "stage": stage.name,
                "validation_manifest_sha256": self.datasets[stage.name][1].sha256,
                "evaluated_optimizer_steps": self.state.accepted_updates,
                "validation_records": count,
                "minimum_validation_records": MIN_GATE_VALIDATION_RECORDS,
                "metrics": metrics,
                "passed": passed,
            },
        )

    def evaluate(self, stage: NativeStageConfig) -> float:
        return self._evaluate_stage(stage).loss

    def _refresh_gate(self, stage: NativeStageConfig, quality: dict[str, Any]) -> None:
        if stage.objective not in NATIVE_GATE_NAMES:
            return
        self.state.quality[stage.objective] = quality
        enough_training = (
            self.config.allow_runtime_readiness
            and stage.name in self.state.completed_stages
            and self.state.accepted_updates_by_stage.get(stage.name, 0)
            >= MIN_GATE_UPDATES
        )
        gate = self.state.gates[stage.objective]
        gate["trained"] = enough_training
        gate["validation_passed"] = enough_training and quality.get("passed") is True

    def _refresh_all_gates(self) -> None:
        for stage in self.config.stages:
            if stage.objective not in NATIVE_GATE_NAMES:
                continue
            evaluation = self._evaluate_stage(stage)
            loss = evaluation.loss
            if math.isfinite(loss):
                previous = self.state.best_validation_losses.get(stage.name, math.inf)
                self.state.best_validation_losses[stage.name] = min(previous, loss)
            assert evaluation.quality is not None
            self._refresh_gate(stage, evaluation.quality)

    def _invalidate_quality_after_update(self) -> None:
        """Invalidate evidence measured against any earlier parameter state."""

        self.state.quality = self._quality_template()
        for gate in self.state.gates.values():
            gate["validation_passed"] = False

    def run(self, *, max_steps: int | None = None) -> NativeTrainerState:
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        attempts = 0
        while not self.complete and (max_steps is None or attempts < max_steps):
            stage = self.config.stages[self.state.stage_index]
            self._configure_stage(stage)
            batch = self._next_training_batch(stage)
            objective = stage.objective

            def loss_fn(current: ScratchMultimodalLM) -> mx.array:
                return _native_stage_loss(current, batch, objective)

            loss_and_grad = nn.value_and_grad(self.model, loss_fn)
            loss, gradients = loss_and_grad(self.model)
            clipped, grad_norm = clip_grad_norm_fp32(
                gradients, self.config.optimizer.grad_clip
            )
            mx.eval(loss, gradients, clipped, grad_norm)
            loss_value = float(loss.item())
            grad_value = float(grad_norm.item())
            self.state.global_step += 1
            self.state.stage_step += 1
            attempts += 1
            if not math.isfinite(loss_value) or not math.isfinite(grad_value):
                self.state.skipped_updates += 1
                self._write_metric(
                    "skipped_update",
                    global_step=self.state.global_step,
                    stage=stage.name,
                    reason="non_finite",
                )
            else:
                assert self.optimizer is not None
                previous_parameters = self.model.parameters()
                previous_optimizer = tree_map(lambda value: value, self.optimizer.state)
                self.optimizer.learning_rate = mx.array(
                    _stage_learning_rate(stage, self.state.stage_step - 1),
                    dtype=mx.float32,
                )
                self.optimizer.update(self.model, clipped)
                mx.eval(self.model.parameters(), self.optimizer.state)
                if not _tree_is_finite(self.model.parameters()) or not _tree_is_finite(
                    self.optimizer.state
                ):
                    self.model.update(previous_parameters)
                    self.optimizer.state = previous_optimizer
                    self.optimizer.init(self.model.trainable_parameters())
                    mx.eval(self.model.parameters(), self.optimizer.state)
                    self.state.skipped_updates += 1
                    self._write_metric(
                        "skipped_update",
                        global_step=self.state.global_step,
                        stage=stage.name,
                        reason="non_finite_update",
                    )
                else:
                    self.state.accepted_updates += 1
                    self.state.accepted_updates_by_stage[stage.name] = (
                        self.state.accepted_updates_by_stage.get(stage.name, 0) + 1
                    )
                    self._invalidate_quality_after_update()
            if (
                self.state.stage_step == 1
                or self.state.stage_step % stage.report_every == 0
            ):
                self._write_metric(
                    "train",
                    global_step=self.state.global_step,
                    stage=stage.name,
                    objective=stage.objective,
                    stage_step=self.state.stage_step,
                    loss=loss_value if math.isfinite(loss_value) else None,
                    grad_norm=grad_value if math.isfinite(grad_value) else None,
                )

            stage_complete = self.state.stage_step >= stage.steps
            eval_due = self.state.stage_step % stage.eval_every == 0 or stage_complete
            latest_eval_loss: float | None = None
            latest_quality: dict[str, Any] | None = None
            if eval_due:
                evaluation = self._evaluate_stage(stage)
                eval_loss = evaluation.loss
                latest_eval_loss = eval_loss
                latest_quality = evaluation.quality
                if stage.objective in NATIVE_GATE_NAMES:
                    assert latest_quality is not None
                    self.state.quality[stage.objective] = latest_quality
                if math.isfinite(eval_loss):
                    previous = self.state.best_validation_losses.get(
                        stage.name, math.inf
                    )
                    self.state.best_validation_losses[stage.name] = min(
                        previous, eval_loss
                    )
                self._write_metric(
                    "validation",
                    global_step=self.state.global_step,
                    stage=stage.name,
                    loss=eval_loss if math.isfinite(eval_loss) else None,
                    threshold=stage.max_validation_loss,
                )

            save_due = self.state.stage_step % stage.save_every == 0
            if stage_complete:
                if stage.name not in self.state.completed_stages:
                    self.state.completed_stages.append(stage.name)
                if latest_eval_loss is None:
                    raise RuntimeError(
                        "native stage completed without held-out evaluation"
                    )
                if stage.objective in NATIVE_GATE_NAMES:
                    if latest_quality is None:
                        raise RuntimeError(
                            "native readiness stage completed without quality evidence"
                        )
                    self._refresh_gate(stage, latest_quality)
                self.state.stage_index += 1
                self.state.stage_step = 0
                self._optimizer_stage = None
                self.optimizer = None
                if stage.objective == "joint":
                    self._refresh_all_gates()
            if save_due or stage_complete:
                self.save_checkpoint()
        if (
            attempts > 0
            and self.state.accepted_updates > 0
            and self._last_saved_state_signature != self._state_signature()
        ):
            self.save_checkpoint()
        return self.state

    def _state_signature(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self.state), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def _validate_state(self) -> None:
        state = self.state

        def require_non_negative_integer(value: object, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"native trainer {label} is invalid")
            return value

        global_step = require_non_negative_integer(state.global_step, "global_step")
        stage_index = require_non_negative_integer(state.stage_index, "stage_index")
        stage_step = require_non_negative_integer(state.stage_step, "stage_step")
        accepted = require_non_negative_integer(
            state.accepted_updates, "accepted_updates"
        )
        skipped = require_non_negative_integer(state.skipped_updates, "skipped_updates")
        if stage_index > len(self.config.stages):
            raise ValueError("native resume stage index is invalid")
        if stage_index == len(self.config.stages):
            if stage_step != 0:
                raise ValueError(
                    "completed native training may not retain a stage step"
                )
        elif stage_step >= self.config.stages[stage_index].steps:
            raise ValueError("native resume stage step is invalid")
        expected_completed = [stage.name for stage in self.config.stages[:stage_index]]
        if state.completed_stages != expected_completed:
            raise ValueError("native completed-stage state is inconsistent")
        expected_global_step = (
            sum(stage.steps for stage in self.config.stages[:stage_index]) + stage_step
        )
        if global_step != expected_global_step or accepted + skipped != global_step:
            raise ValueError("native update counters are inconsistent")

        stage_names = {stage.name for stage in self.config.stages}
        if set(state.accepted_updates_by_stage) - stage_names:
            raise ValueError("native per-stage update counters are incompatible")
        per_stage_total = 0
        for name, value in state.accepted_updates_by_stage.items():
            per_stage_total += require_non_negative_integer(
                value, f"accepted_updates_by_stage[{name!r}]"
            )
        if per_stage_total != accepted:
            raise ValueError("native accepted-update counters are inconsistent")
        for name, value in state.best_validation_losses.items():
            if name not in stage_names or not isinstance(value, (int, float)):
                raise ValueError("native validation-loss state is incompatible")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    "native validation-loss state must be finite and non-negative"
                )

        if set(state.data_epochs) != set(state.data_indices):
            raise ValueError("native data cursor state is incomplete")
        for name, epoch in state.data_epochs.items():
            if name not in stage_names:
                raise ValueError("native data cursor stage is incompatible")
            require_non_negative_integer(epoch, f"data_epochs[{name!r}]")
            index = require_non_negative_integer(
                state.data_indices[name], f"data_indices[{name!r}]"
            )
            if index > len(self.datasets[name][0]):
                raise ValueError("native data cursor index is outside its dataset")

        template = self._quality_template()
        if not isinstance(state.quality, dict) or set(state.quality) != set(
            NATIVE_GATE_NAMES
        ):
            raise ValueError("native quality evidence is incompatible")
        for gate_name, expected_entry in template.items():
            entry = state.quality[gate_name]
            if not isinstance(entry, dict) or set(entry) != {
                "stage",
                "validation_manifest_sha256",
                "evaluated_optimizer_steps",
                "validation_records",
                "minimum_validation_records",
                "metrics",
                "passed",
            }:
                raise ValueError("native quality evidence fields are incompatible")
            if entry["stage"] != expected_entry["stage"]:
                raise ValueError("native quality evidence stage differs")
            if (
                entry["validation_manifest_sha256"]
                != expected_entry["validation_manifest_sha256"]
            ):
                raise ValueError("native quality holdout checksum differs")
            evaluated_steps = entry["evaluated_optimizer_steps"]
            if evaluated_steps is not None:
                evaluated_steps = require_non_negative_integer(
                    evaluated_steps,
                    f"quality[{gate_name!r}].evaluated_optimizer_steps",
                )
            records = require_non_negative_integer(
                entry["validation_records"],
                f"quality[{gate_name!r}].validation_records",
            )
            if entry["minimum_validation_records"] != MIN_GATE_VALIDATION_RECORDS:
                raise ValueError("native quality minimum holdout differs")
            metrics = entry["metrics"]
            expected_metrics = expected_entry["metrics"]
            if not isinstance(metrics, dict) or set(metrics) != set(expected_metrics):
                raise ValueError("native quality metric set is incompatible")
            metric_passes: list[bool] = []
            metric_values_present: list[bool] = []
            for metric_name, expected_metric in expected_metrics.items():
                metric = metrics[metric_name]
                if not isinstance(metric, dict) or set(metric) != {
                    "value",
                    "threshold",
                    "direction",
                    "passed",
                }:
                    raise ValueError("native quality metric fields are incompatible")
                if metric["direction"] != expected_metric["direction"]:
                    raise ValueError("native quality metric direction differs")
                value = metric["value"]
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("native quality metric value is invalid")
                if value is None:
                    required_threshold: float | None = None
                elif metric_name == "stage_loss":
                    stage_for_gate = next(
                        (
                            stage
                            for stage in self.config.stages
                            if stage.objective == gate_name
                        ),
                        None,
                    )
                    if stage_for_gate is None:
                        raise ValueError(
                            "unconfigured native quality metric has a value"
                        )
                    required_threshold = float(stage_for_gate.max_validation_loss)
                else:
                    required_threshold = NATIVE_QUALITY_THRESHOLDS[gate_name][
                        metric_name
                    ][0]
                if metric["threshold"] != required_threshold:
                    raise ValueError("native quality metric threshold differs")
                recomputed = _quality_metric(
                    None if value is None else float(value),
                    metric["threshold"],
                    metric["direction"],
                )["passed"]
                if (
                    type(metric["passed"]) is not bool
                    or metric["passed"] is not recomputed
                ):
                    raise ValueError("native quality metric pass flag is inconsistent")
                metric_passes.append(metric["passed"])
                metric_values_present.append(value is not None)
            if evaluated_steps is None:
                if records != 0 or any(metric_values_present):
                    raise ValueError("unevaluated native quality contains measurements")
            elif expected_entry["stage"] is None or records <= 0:
                raise ValueError("evaluated native quality lacks a configured holdout")
            expected_passed = (
                expected_entry["stage"] is not None
                and evaluated_steps == state.accepted_updates
                and records >= MIN_GATE_VALIDATION_RECORDS
                and all(metric_passes)
            )
            if (
                type(entry["passed"]) is not bool
                or entry["passed"] is not expected_passed
            ):
                raise ValueError("native quality aggregate pass flag is inconsistent")

        if set(state.gates) != set(NATIVE_GATE_NAMES):
            raise ValueError("native checkpoint gate state is incompatible")
        stages_by_objective = {
            stage.objective: stage
            for stage in self.config.stages
            if stage.objective in NATIVE_GATE_NAMES
        }
        for gate_name, gate in state.gates.items():
            if not isinstance(gate, dict) or set(gate) != {
                "trained",
                "validation_passed",
            }:
                raise ValueError("native checkpoint gate fields are incompatible")
            if (
                type(gate["trained"]) is not bool
                or type(gate["validation_passed"]) is not bool
            ):
                raise ValueError("native checkpoint gate values must be booleans")
            if gate["validation_passed"] and not gate["trained"]:
                raise ValueError("native validation gate cannot pass before training")
            stage = stages_by_objective.get(gate_name)
            expected_trained = (
                self.config.allow_runtime_readiness
                and stage is not None
                and stage.name in state.completed_stages
                and state.accepted_updates_by_stage.get(stage.name, 0)
                >= MIN_GATE_UPDATES
            )
            if gate["trained"] is not expected_trained:
                raise ValueError("native trained gate is inconsistent")
            expected_validation = (
                expected_trained and state.quality[gate_name]["passed"]
            )
            if gate["validation_passed"] is not expected_validation:
                raise ValueError("native validation gate is inconsistent")

    def save_checkpoint(self) -> Path:
        if self.state.accepted_updates <= 0:
            raise ValueError(
                "native runtime checkpoint requires at least one accepted update"
            )
        self._validate_state()
        if self.optimizer is None:
            if self._optimizer_stage is not None or self.state.stage_step != 0:
                raise ValueError("native checkpoint optimizer boundary is inconsistent")
        else:
            if self.state.stage_index >= len(self.config.stages):
                raise ValueError(
                    "completed native training may not retain an optimizer"
                )
            active_stage = self.config.stages[self.state.stage_index]
            if self._optimizer_stage != active_stage.name or self.state.stage_step <= 0:
                raise ValueError("native checkpoint optimizer stage is inconsistent")
            mx.eval(self.optimizer.state)
            if int(
                self.optimizer.step.item()
            ) != self.state.accepted_updates_by_stage.get(active_stage.name, 0):
                raise ValueError("native optimizer step is inconsistent")
        identifier = f"native-{self.state.global_step:09d}-{uuid.uuid4().hex[:12]}"
        temporary = Path(tempfile.mkdtemp(prefix=".native-", dir=self.checkpoint_root))
        destination = self.checkpoint_root / identifier
        try:
            config_path = temporary / "multimodal-config.json"
            weights_path = temporary / "model.safetensors"
            optimizer_path = temporary / "optimizer.safetensors"
            config_path.write_text(
                json.dumps(
                    self.model_config.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            self.model.save_weights(str(weights_path))
            if self.optimizer is not None:
                optimizer_arrays = dict(tree_flatten(self.optimizer.state))
                mx.eval(optimizer_arrays)
                mx.save_safetensors(str(optimizer_path), optimizer_arrays)
            _fsync_file(config_path)
            _fsync_file(weights_path)
            if optimizer_path.is_file():
                _fsync_file(optimizer_path)
            config_digest = _file_sha256(config_path)
            weights_digest = _file_sha256(weights_path)
            manifest = {
                "schema_version": 1,
                "format": NATIVE_RUNTIME_FORMAT,
                "checkpoint_id": identifier,
                "scratch_only": True,
                "external_weights": False,
                "model": {
                    "config": config_path.name,
                    "weights": weights_path.name,
                    "config_sha256": config_digest,
                    "weights_sha256": weights_digest,
                },
                "training": {
                    "optimizer_steps": self.state.accepted_updates,
                    "gates": self.state.gates,
                    "quality": self.state.quality,
                },
                "runtime": {
                    "sample_rate": self.model_config.audio.sample_rate,
                    "frame_size": self.model_config.audio.frame_size,
                    "video_width": 320,
                    "video_height": 192,
                },
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            mlx_global_key = mx.array(mx.random.state[0])
            mx.eval(mlx_global_key)
            trainer_state = {
                "schema_version": NATIVE_TRAINING_SCHEMA_VERSION,
                "training_config_sha256": self.config_sha256,
                "tokenizer_sha256": self.config.tokenizer_sha256,
                "source_sha256": self.source_hashes,
                "model_config_sha256": self.model_config_sha256,
                "state": asdict(self.state),
                "optimizer_stage": self._optimizer_stage,
                "optimizer_sha256": (
                    _file_sha256(optimizer_path) if optimizer_path.is_file() else None
                ),
                "rng": {
                    "python": random.getstate(),
                    "numpy": _numpy_rng_to_dict(np.random.get_state()),
                    "mlx_global_key": mlx_global_key.tolist(),
                },
            }
            (temporary / "trainer-state.json").write_text(
                json.dumps(trainer_state, sort_keys=True), encoding="utf-8"
            )
            _fsync_file(temporary / "manifest.json")
            _fsync_file(temporary / "trainer-state.json")
            _fsync_directory(temporary)
            os.replace(temporary, destination)
            _fsync_directory(self.checkpoint_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        _atomic_json(
            self.output_dir / "last.json",
            {
                "schema_version": 1,
                "checkpoint": str(destination.relative_to(self.output_dir)),
                "global_step": self.state.global_step,
            },
        )
        self._write_metric(
            "checkpoint",
            global_step=self.state.global_step,
            checkpoint=str(destination.relative_to(self.output_dir)),
            conversation_ready=all(
                gate["trained"] and gate["validation_passed"]
                for gate in self.state.gates.values()
            ),
        )
        self._last_saved_state_signature = self._state_signature()
        return destination

    def load_checkpoint(self, path: str | Path) -> Path:
        checkpoint = _resolve_checkpoint(Path(path).expanduser().resolve())
        manifest = _read_json_object(checkpoint / "manifest.json", "runtime manifest")
        trainer = _read_json_object(checkpoint / "trainer-state.json", "trainer state")
        _validate_runtime_manifest(manifest, checkpoint, self.model_config)
        expected_trainer_fields = {
            "schema_version",
            "training_config_sha256",
            "tokenizer_sha256",
            "source_sha256",
            "model_config_sha256",
            "state",
            "optimizer_stage",
            "optimizer_sha256",
            "rng",
        }
        if set(trainer) != expected_trainer_fields:
            raise ValueError("native resume trainer-state fields are incompatible")
        if trainer.get("schema_version") != NATIVE_TRAINING_SCHEMA_VERSION:
            raise ValueError("native resume trainer-state schema is incompatible")
        if trainer.get("training_config_sha256") != self.config_sha256:
            raise ValueError("native resume training configuration differs")
        if trainer.get("tokenizer_sha256") != self.config.tokenizer_sha256:
            raise ValueError("native resume tokenizer differs")
        if trainer.get("source_sha256") != self.source_hashes:
            raise ValueError("native resume data provenance differs")
        if trainer.get("model_config_sha256") != self.model_config_sha256:
            raise ValueError("native resume source model configuration differs")
        config_path = checkpoint / "multimodal-config.json"
        weights_path = checkpoint / "model.safetensors"
        if MultimodalConfig.from_json(config_path) != self.model_config:
            raise ValueError("native resume model architecture differs")
        self.model.load_weights(str(weights_path), strict=True)
        self.state = NativeTrainerState.from_dict(trainer["state"])
        self._validate_state()
        if manifest["training"]["optimizer_steps"] != self.state.accepted_updates:
            raise ValueError("native runtime and trainer update counts differ")
        if manifest["training"]["gates"] != self.state.gates:
            raise ValueError("native runtime and trainer gate states differ")
        if manifest["training"]["quality"] != self.state.quality:
            raise ValueError("native runtime and trainer quality evidence differs")
        optimizer_stage = trainer.get("optimizer_stage")
        optimizer_path = checkpoint / "optimizer.safetensors"
        if (
            self.state.stage_index < len(self.config.stages)
            and self.state.stage_step > 0
        ):
            stage = self.config.stages[self.state.stage_index]
            if optimizer_stage != stage.name or not optimizer_path.is_file():
                raise ValueError("native resume optimizer stage is incomplete")
            if _file_sha256(optimizer_path) != trainer.get("optimizer_sha256"):
                raise ValueError("native resume optimizer checksum differs")
            self._configure_stage(stage)
            assert self.optimizer is not None
            loaded = mx.load(str(optimizer_path))
            if not isinstance(loaded, dict):
                raise ValueError("native resume optimizer is malformed")
            expected = dict(tree_flatten(self.optimizer.state))
            if set(expected) != set(loaded):
                raise ValueError("native resume optimizer keys differ")
            for key in expected:
                if (
                    expected[key].shape != loaded[key].shape
                    or expected[key].dtype != loaded[key].dtype
                ):
                    raise ValueError("native resume optimizer tensor differs")
            self.optimizer.state = tree_unflatten(
                [(key, loaded[key]) for key in expected]
            )
            self.optimizer.init(self.model.trainable_parameters())
            mx.eval(self.optimizer.state)
            if int(
                self.optimizer.step.item()
            ) != self.state.accepted_updates_by_stage.get(stage.name, 0):
                raise ValueError("native resume optimizer step differs")
        else:
            if (
                optimizer_stage is not None
                or trainer.get("optimizer_sha256") is not None
            ):
                raise ValueError("native resume contains an unexpected optimizer state")
            if optimizer_path.exists():
                raise ValueError("native resume contains an unexpected optimizer file")
        rng = trainer.get("rng")
        if not isinstance(rng, dict) or set(rng) != {
            "python",
            "numpy",
            "mlx_global_key",
        }:
            raise ValueError("native resume RNG state is incompatible")
        try:
            random.setstate(_nested_tuple(rng["python"]))
            np.random.set_state(_numpy_rng_from_dict(rng["numpy"]))
            mlx_words = rng["mlx_global_key"]
            if not isinstance(mlx_words, list) or len(mlx_words) != 2:
                raise ValueError("native resume MLX RNG state is malformed")
            mx.random.seed((int(mlx_words[0]) << 32) | int(mlx_words[1]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("native resume RNG state is malformed") from error
        mx.eval(
            self.model.parameters(),
            *([self.optimizer.state] if self.optimizer is not None else []),
        )
        self.model.train()
        self._last_saved_state_signature = self._state_signature()
        return checkpoint


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file() and path.name == "last.json":
        pointer = _read_json_object(path, "checkpoint pointer")
        path = _resolve_checkpoint_pointer(path.parent, pointer)
    elif (
        path.is_dir()
        and (path / "last.json").is_file()
        and not (path / "manifest.json").is_file()
    ):
        pointer = _read_json_object(path / "last.json", "checkpoint pointer")
        path = _resolve_checkpoint_pointer(path, pointer)
    required = (
        "manifest.json",
        "trainer-state.json",
        "model.safetensors",
        "multimodal-config.json",
    )
    if not path.is_dir() or not all((path / name).is_file() for name in required):
        raise ValueError("native resume path is not a complete checkpoint")
    return path


def _resolve_checkpoint_pointer(root: Path, pointer: dict[str, Any]) -> Path:
    if set(pointer) != {"schema_version", "checkpoint", "global_step"}:
        raise ValueError("native checkpoint pointer fields are incompatible")
    if pointer["schema_version"] != NATIVE_TRAINING_SCHEMA_VERSION:
        raise ValueError("native checkpoint pointer schema is incompatible")
    relative = pointer["checkpoint"]
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("native checkpoint pointer path is invalid")
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            "native checkpoint pointer escapes its output directory"
        ) from error
    return destination


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > MAX_NATIVE_JSON_BYTES:
            raise ValueError(f"native {label} is empty or too large")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"native {label} contains duplicate fields")
                result[key] = value
            return result

        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError(f"native {label} contains a non-finite number")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"native {label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"native {label} must be a JSON object")
    return value


def _validate_runtime_manifest(
    manifest: dict[str, Any],
    checkpoint: Path,
    model_config: MultimodalConfig,
) -> None:
    expected_root = {
        "schema_version",
        "format",
        "checkpoint_id",
        "scratch_only",
        "external_weights",
        "model",
        "training",
        "runtime",
    }
    if set(manifest) != expected_root:
        raise ValueError("native runtime manifest fields are incompatible")
    if manifest["schema_version"] != 1 or manifest["format"] != NATIVE_RUNTIME_FORMAT:
        raise ValueError("resume checkpoint is not an MLM native v1 checkpoint")
    if manifest["checkpoint_id"] != checkpoint.name:
        raise ValueError("native runtime checkpoint identity differs")
    if (
        manifest["scratch_only"] is not True
        or manifest["external_weights"] is not False
    ):
        raise ValueError("external native weights are not accepted")

    model = manifest["model"]
    if not isinstance(model, dict) or set(model) != {
        "config",
        "weights",
        "config_sha256",
        "weights_sha256",
    }:
        raise ValueError("native runtime model manifest is incompatible")
    if (
        model["config"] != "multimodal-config.json"
        or model["weights"] != "model.safetensors"
    ):
        raise ValueError("native runtime model paths are incompatible")
    if _file_sha256(checkpoint / model["config"]) != model["config_sha256"]:
        raise ValueError("native resume model config checksum differs")
    if _file_sha256(checkpoint / model["weights"]) != model["weights_sha256"]:
        raise ValueError("native resume model weights checksum differs")

    training = manifest["training"]
    if not isinstance(training, dict) or set(training) != {
        "optimizer_steps",
        "gates",
        "quality",
    }:
        raise ValueError("native runtime training manifest is incompatible")
    optimizer_steps = training["optimizer_steps"]
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps <= 0
    ):
        raise ValueError("native runtime optimizer steps must be positive")
    gates = training["gates"]
    if not isinstance(gates, dict) or set(gates) != set(NATIVE_GATE_NAMES):
        raise ValueError("native runtime readiness gates are incompatible")
    for gate in gates.values():
        if not isinstance(gate, dict) or set(gate) != {
            "trained",
            "validation_passed",
        }:
            raise ValueError("native runtime readiness gate fields are incompatible")
        if (
            type(gate["trained"]) is not bool
            or type(gate["validation_passed"]) is not bool
        ):
            raise ValueError("native runtime readiness gate values must be booleans")
        if gate["validation_passed"] and not gate["trained"]:
            raise ValueError("native runtime validation cannot pass before training")
    _validate_runtime_quality(training["quality"], optimizer_steps=optimizer_steps)
    for gate_name, gate in gates.items():
        if gate["validation_passed"] is not (
            gate["trained"] and training["quality"][gate_name]["passed"]
        ):
            raise ValueError("native runtime gate and quality evidence differ")

    expected_runtime = {
        "sample_rate": model_config.audio.sample_rate,
        "frame_size": model_config.audio.frame_size,
        "video_width": 320,
        "video_height": 192,
    }
    if manifest["runtime"] != expected_runtime:
        raise ValueError("native runtime media contract differs")


def _validate_runtime_quality(value: object, *, optimizer_steps: int) -> None:
    if not isinstance(value, dict) or set(value) != set(NATIVE_GATE_NAMES):
        raise ValueError("native runtime quality evidence is incompatible")
    for gate_name in NATIVE_GATE_NAMES:
        entry = value[gate_name]
        if not isinstance(entry, dict) or set(entry) != {
            "stage",
            "validation_manifest_sha256",
            "evaluated_optimizer_steps",
            "validation_records",
            "minimum_validation_records",
            "metrics",
            "passed",
        }:
            raise ValueError("native runtime quality fields are incompatible")
        stage = entry["stage"]
        if stage is not None and (not isinstance(stage, str) or not stage.strip()):
            raise ValueError("native runtime quality stage is invalid")
        validation_digest = entry["validation_manifest_sha256"]
        if stage is None:
            if validation_digest is not None:
                raise ValueError("unconfigured native quality holdout must be null")
        elif (
            not isinstance(validation_digest, str)
            or len(validation_digest) != 64
            or validation_digest != validation_digest.lower()
            or any(
                character not in "0123456789abcdef" for character in validation_digest
            )
        ):
            raise ValueError("native runtime quality holdout checksum is invalid")
        evaluated_steps = entry["evaluated_optimizer_steps"]
        if evaluated_steps is not None and (
            isinstance(evaluated_steps, bool)
            or not isinstance(evaluated_steps, int)
            or evaluated_steps < 0
        ):
            raise ValueError("native runtime quality optimizer step is invalid")
        records = entry["validation_records"]
        if isinstance(records, bool) or not isinstance(records, int) or records < 0:
            raise ValueError("native runtime quality sample count is invalid")
        if entry["minimum_validation_records"] != MIN_GATE_VALIDATION_RECORDS:
            raise ValueError("native runtime quality minimum holdout differs")
        metrics = entry["metrics"]
        expected_names = {"stage_loss", *NATIVE_QUALITY_THRESHOLDS[gate_name]}
        if not isinstance(metrics, dict) or set(metrics) != expected_names:
            raise ValueError("native runtime quality metric set is incompatible")
        metric_passes: list[bool] = []
        metric_values_present: list[bool] = []
        for metric_name, metric in metrics.items():
            if not isinstance(metric, dict) or set(metric) != {
                "value",
                "threshold",
                "direction",
                "passed",
            }:
                raise ValueError(
                    "native runtime quality metric fields are incompatible"
                )
            if metric_name == "stage_loss":
                expected_direction = "at_most"
                threshold = metric["threshold"]
                if threshold is not None and (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or not math.isfinite(float(threshold))
                    or float(threshold) <= 0.0
                ):
                    raise ValueError("native runtime quality loss threshold is invalid")
            else:
                expected_threshold, expected_direction = NATIVE_QUALITY_THRESHOLDS[
                    gate_name
                ][metric_name]
                if metric["threshold"] not in {None, expected_threshold}:
                    raise ValueError("native runtime quality threshold differs")
            if metric["direction"] != expected_direction:
                raise ValueError("native runtime quality direction differs")
            metric_value = metric["value"]
            if metric_value is not None and (
                isinstance(metric_value, bool)
                or not isinstance(metric_value, (int, float))
                or not math.isfinite(float(metric_value))
            ):
                raise ValueError("native runtime quality value is invalid")
            if (metric_value is None) != (metric["threshold"] is None):
                raise ValueError("native runtime quality value/threshold must align")
            recomputed = _quality_metric(
                None if metric_value is None else float(metric_value),
                None if metric["threshold"] is None else float(metric["threshold"]),
                metric["direction"],
            )["passed"]
            if type(metric["passed"]) is not bool or metric["passed"] is not recomputed:
                raise ValueError("native runtime quality metric pass flag differs")
            metric_passes.append(metric["passed"])
            metric_values_present.append(metric_value is not None)
        if evaluated_steps is None:
            if records != 0 or any(metric_values_present):
                raise ValueError("unevaluated native quality contains measurements")
        elif stage is None or records <= 0:
            raise ValueError("evaluated native quality lacks a configured holdout")
        expected_passed = (
            stage is not None
            and evaluated_steps == optimizer_steps
            and records >= MIN_GATE_VALIDATION_RECORDS
            and all(metric_passes)
        )
        if type(entry["passed"]) is not bool or entry["passed"] is not expected_passed:
            raise ValueError("native runtime quality aggregate pass flag differs")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required native file is missing: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_metric(
    value: float | None,
    threshold: float | None,
    direction: str,
) -> dict[str, object]:
    if direction not in {"at_least", "at_most"}:
        raise ValueError("native quality metric direction is incompatible")
    finite_value = (
        float(value) if value is not None and math.isfinite(float(value)) else None
    )
    finite_threshold = (
        float(threshold)
        if threshold is not None and math.isfinite(float(threshold))
        else None
    )
    if finite_value is None:
        finite_threshold = None
    passed = False
    if finite_value is not None and finite_threshold is not None:
        if direction == "at_least":
            passed = finite_value >= finite_threshold
        else:
            passed = finite_value <= finite_threshold
    return {
        "value": finite_value,
        "threshold": finite_threshold,
        "direction": direction,
        "passed": passed,
    }


def _tree_is_finite(tree: object) -> bool:
    leaves = [value for _, value in tree_flatten(tree) if isinstance(value, mx.array)]
    if not leaves:
        return True
    checks = [mx.all(mx.isfinite(value)) for value in leaves]
    mx.eval(checks)
    return all(bool(value.item()) for value in checks)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _numpy_rng_from_dict(payload: object) -> tuple:
    if not isinstance(payload, dict) or set(payload) != {
        "name",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("native NumPy RNG state is incompatible")
    return (
        str(payload["name"]),
        np.array(payload["keys"], dtype=np.uint32),
        int(payload["position"]),
        int(payload["has_gauss"]),
        float(payload["cached_gaussian"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MLM's native audio/video model from random initialization"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    trainer = NativeScratchTrainer(
        NativeTrainingConfig.from_json(args.config), resume=args.resume
    )
    state = trainer.run(max_steps=args.max_steps)
    print(
        json.dumps(
            {
                "global_step": state.global_step,
                "stage_index": state.stage_index,
                "complete": trainer.complete,
                "gates": state.gates,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
