from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import replace

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from mini_arithmetic.native_runtime import load_native_checkpoint
from mini_arithmetic.native_train import (
    NATIVE_GATE_NAMES,
    NativeScratchTrainer,
    NativeStageConfig,
    NativeTrainingConfig,
)
from test_multimodal import tiny_multimodal_config


TOKENIZER_DIGEST = "b" * 64


def _write_manifest(
    root: Path,
    *,
    split: str,
    amplitude: float,
    text_tokens: tuple[int, ...] = (1, 2, 3),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shard = root / f"{split}.npz"
    np.savez(
        shard,
        user_waveform=np.full((1, 24, 1), amplitude, dtype=np.float32),
        model_waveform=np.full((1, 24, 1), -amplitude, dtype=np.float32),
        text_tokens=np.array([text_tokens], dtype=np.int32),
        text_target_mask=np.ones((1, len(text_tokens)), dtype=np.float32),
        text_timestamp=np.array(0.0005, dtype=np.float32),
    )
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "split": split,
        "tokenizer_sha256": TOKENIZER_DIGEST,
        "sample_rate": 24_000,
        "frame_size": 8,
        "records": [
            {
                "record_id": f"{split}-1",
                "shard": shard.name,
                "sha256": digest,
                "source": "consenting test speaker",
                "license": "project-owned-test-fixture",
                "consent": True,
                "provenance": {"fixture": True},
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _config(tmp_path: Path) -> NativeTrainingConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "tiny-model.json"
    model_path.write_text(json.dumps(tiny_multimodal_config().to_dict()))
    train = _write_manifest(tmp_path / "train", split="train", amplitude=0.0)
    validation = _write_manifest(
        tmp_path / "validation", split="validation", amplitude=0.1
    )
    return NativeTrainingConfig(
        model_config=str(model_path),
        tokenizer_sha256=TOKENIZER_DIGEST,
        output_dir=str(tmp_path / "output"),
        parameter_dtype="float32",
        stages=(
            NativeStageConfig(
                name="codec-pilot",
                objective="audio_codec",
                train_manifest=str(train),
                validation_manifest=str(validation),
                steps=2,
                max_validation_loss=100.0,
                peak_learning_rate=1e-3,
                warmup_steps=0,
                eval_every=1,
                eval_records=1,
                save_every=1,
                report_every=1,
            ),
        ),
    )


def test_native_trainer_is_resumable_and_emits_fail_closed_runtime_manifest(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    trainer = NativeScratchTrainer(config)
    state = trainer.run(max_steps=1)
    assert state.global_step == 1
    assert not trainer.complete
    pointer = json.loads((Path(config.output_dir) / "last.json").read_text())
    checkpoint = Path(config.output_dir) / pointer["checkpoint"]
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert set(manifest) == {
        "schema_version",
        "format",
        "checkpoint_id",
        "scratch_only",
        "external_weights",
        "model",
        "training",
        "runtime",
    }
    assert manifest["scratch_only"] is True
    assert manifest["external_weights"] is False
    assert manifest["runtime"] == {
        "sample_rate": 24_000,
        "frame_size": 8,
        "video_width": 320,
        "video_height": 192,
    }
    assert set(manifest["training"]["gates"]) == set(NATIVE_GATE_NAMES)
    assert set(manifest["training"]["quality"]) == set(NATIVE_GATE_NAMES)
    codec_quality = manifest["training"]["quality"]["audio_codec"]
    assert codec_quality["validation_records"] == 1
    assert codec_quality["minimum_validation_records"] == 128
    assert (
        codec_quality["validation_manifest_sha256"]
        == trainer.datasets["codec-pilot"][1].sha256
    )
    assert codec_quality["evaluated_optimizer_steps"] == 1
    assert codec_quality["metrics"]["reconstruction_snr_db"]["threshold"] == 15.0
    assert codec_quality["passed"] is False
    assert not any(
        gate["trained"] or gate["validation_passed"]
        for gate in manifest["training"]["gates"].values()
    )
    loaded = load_native_checkpoint(checkpoint)
    assert loaded.manifest.conversation_ready is False

    resumed = NativeScratchTrainer(config, resume=Path(config.output_dir) / "last.json")
    resumed_state = resumed.run(max_steps=1)
    assert resumed.complete
    assert resumed_state.global_step == 2
    assert "codec-pilot" in resumed_state.completed_stages
    # A two-step, one-record fixture must never be promoted as a trained codec,
    # even though its deliberately loose numerical loss threshold is met.
    assert resumed_state.gates["audio_codec"] == {
        "trained": False,
        "validation_passed": False,
    }


def test_native_stage_schedule_and_config_fingerprint_are_deterministic(
    tmp_path: Path,
) -> None:
    first = _config(tmp_path)
    second = _config(tmp_path)
    assert first.fingerprint() == second.fingerprint()


def test_native_trainer_rejects_decoded_media_reused_in_global_holdout(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "tiny-model.json"
    model_path.write_text(json.dumps(tiny_multimodal_config().to_dict()))
    train = _write_manifest(
        tmp_path / "train",
        split="train",
        amplitude=0.2,
        text_tokens=(1, 2, 3),
    )
    validation = _write_manifest(
        tmp_path / "validation",
        split="validation",
        amplitude=0.2,
        text_tokens=(4, 5, 6),
    )
    assert (
        hashlib.sha256(train.read_bytes()).hexdigest()
        != hashlib.sha256(validation.read_bytes()).hexdigest()
    )
    config = NativeTrainingConfig(
        model_config=str(model_path),
        tokenizer_sha256=TOKENIZER_DIGEST,
        output_dir=str(tmp_path / "output"),
        parameter_dtype="float32",
        stages=(
            NativeStageConfig(
                name="codec",
                objective="audio_codec",
                train_manifest=str(train),
                validation_manifest=str(validation),
                steps=2,
                max_validation_loss=1.0,
                warmup_steps=0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="global validation holdout"):
        NativeScratchTrainer(config)


def test_native_resume_matches_uninterrupted_training_exactly(tmp_path: Path) -> None:
    continuous_config = _config(tmp_path / "continuous")
    continuous = NativeScratchTrainer(continuous_config)
    continuous.run(max_steps=2)

    resumed_config = _config(tmp_path / "resumed")
    partial = NativeScratchTrainer(resumed_config)
    partial.run(max_steps=1)
    resumed = NativeScratchTrainer(
        resumed_config, resume=Path(resumed_config.output_dir) / "last.json"
    )
    resumed.run(max_steps=1)

    continuous_weights = dict(tree_flatten(continuous.model.parameters()))
    resumed_weights = dict(tree_flatten(resumed.model.parameters()))
    assert set(continuous_weights) == set(resumed_weights)
    for name in continuous_weights:
        assert mx.array_equal(
            continuous_weights[name], resumed_weights[name]
        ).item(), name


def test_native_checkpoint_rejects_tampered_optimizer_and_quality(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    trainer = NativeScratchTrainer(config)
    trainer.run(max_steps=1)
    pointer = json.loads((Path(config.output_dir) / "last.json").read_text())
    checkpoint = Path(config.output_dir) / pointer["checkpoint"]

    optimizer = checkpoint / "optimizer.safetensors"
    with optimizer.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="optimizer checksum"):
        NativeScratchTrainer(config, resume=checkpoint)

    # Restore a clean checkpoint, then make its one-record toy evidence claim
    # a pass. The strict runtime recomputation must reject that claim.
    trainer.run(max_steps=0)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training"]["quality"]["audio_codec"]["passed"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="quality pass flag"):
        load_native_checkpoint(checkpoint)


def test_native_stage_freezing_exposes_only_the_intended_modules(
    tmp_path: Path,
) -> None:
    trainer = NativeScratchTrainer(_config(tmp_path))
    base = trainer.config.stages[0]
    expected = {
        "audio_codec": {"audio_codec"},
        "speech_understanding": {
            "thinker",
            "audio_codec",
            "audio_projection",
            "audio_stream_embedding",
            "time_input",
            "time_output",
        },
        "vision": {"thinker", "video_encoder", "time_input", "time_output"},
        "talker": {"audio_codec", "talker"},
        "conversation": {
            "thinker",
            "audio_projection",
            "audio_stream_embedding",
            "time_input",
            "time_output",
            "talker",
        },
        "joint": {
            "thinker",
            "audio_codec",
            "video_encoder",
            "audio_projection",
            "audio_stream_embedding",
            "time_input",
            "time_output",
            "talker",
        },
    }
    for objective, expected_modules in expected.items():
        stage = replace(base, name=f"stage-{objective}", objective=objective)
        trainer._configure_stage(stage)
        actual_modules = {
            name.split(".", 1)[0]
            for name, _ in tree_flatten(trainer.model.trainable_parameters())
        }
        assert actual_modules == expected_modules


def test_explicit_readiness_opt_in_still_cannot_promote_a_toy_run(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path), allow_runtime_readiness=True)
    trainer = NativeScratchTrainer(config)
    state = trainer.run()
    assert state.gates["audio_codec"] == {
        "trained": False,
        "validation_passed": False,
    }
    assert state.quality["audio_codec"]["validation_records"] == 1
    assert state.quality["audio_codec"]["passed"] is False
