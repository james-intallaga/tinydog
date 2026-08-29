from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np
import pytest

from mini_arithmetic.hybrid import CONV, HybridConfig, HybridLM
from mini_arithmetic.hybrid_train import (
    BoundaryBatcher,
    FP32AdamW,
    GuardConfig,
    JsonlDataset,
    OptimizerConfig,
    ScratchTrainer,
    ScratchTrainingConfig,
    StageConfig,
    TokenShardDataset,
    _resolve_checkpoint_path,
    _sft_sample,
    _text_logit_bias,
    clip_grad_norm_fp32,
    masked_next_token_loss,
    scheduled_learning_rate,
)
from mini_arithmetic.prepare_data import write_token_shards
from mini_arithmetic.scratch_tokenizer import (
    VOCAB_SIZE,
    ScratchTokenizer,
    build_scratch_tokenizer,
)


@pytest.fixture(scope="module")
def scratch_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> ScratchTokenizer:
    return build_scratch_tokenizer(
        [
            "Arithmetic dialogue: user asks one plus one and assistant answers two.",
            "A held-out learner asks for a concise, exact mathematical answer.",
            "Numbers 0123456789 and operators + - * / remain structurally clear.",
        ],
        tmp_path_factory.mktemp("hybrid-train-tokenizer"),
    )


def _tiny_model_config() -> HybridConfig:
    return HybridConfig(
        vocab_size=VOCAB_SIZE,
        context_length=32,
        model_dim=8,
        ffn_dim=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        conv_kernel_size=3,
        layer_schedule=(CONV,),
    )


def _content_hash(record_type: str, content: object) -> str:
    canonical = (
        str(content)
        if record_type == "document"
        else json.dumps(
            content, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _document(record_id: str, text: str) -> dict[str, object]:
    return {
        "type": "document",
        "text": text,
        "source": "unit-test",
        "license": "CC0-1.0",
        "provenance": {"fixture": record_id},
        "record_id": record_id,
        "content_sha256": _content_hash("document", text),
    }


def _dialogue(
    record_id: str,
    messages: list[dict[str, str]],
) -> dict[str, object]:
    canonical_messages = [[message["role"], message["content"]] for message in messages]
    return {
        "type": "dialogue",
        "messages": messages,
        "source": "unit-test",
        "license": "CC0-1.0",
        "provenance": {"fixture": record_id},
        "record_id": record_id,
        "content_sha256": _content_hash("dialogue", canonical_messages),
    }


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _stage(
    name: str,
    objective: str,
    train_path: Path,
    eval_path: Path,
    *,
    steps: int = 2,
    data_format: str = "audited_jsonl",
) -> StageConfig:
    return StageConfig(
        name=name,
        objective=objective,
        train_path=str(train_path),
        eval_path=str(eval_path),
        steps=steps,
        data_format=data_format,
        batch_size=1,
        sequence_length=32,
        peak_learning_rate=1e-3,
        warmup_steps=0 if steps == 1 else 1,
        min_learning_rate_ratio=0.1,
        eval_every=1,
        eval_batches=1,
        save_every=1,
        report_every=1,
    )


def _config(
    output: Path,
    stages: tuple[StageConfig, ...],
    tokenizer: ScratchTokenizer,
) -> ScratchTrainingConfig:
    return ScratchTrainingConfig(
        stages=stages,
        tokenizer_path=str(tokenizer.artifact_dir),
        tokenizer_sha256=str(tokenizer.metadata["tokenizer_sha256"]),
        model=_tiny_model_config(),
        output_dir=str(output),
        seed=71,
        parameter_dtype="float32",
    )


def test_config_json_and_warmup_cosine_floor(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(train, [_document("train", "abc")])
    _write_jsonl(heldout, [_document("heldout", "xyz")])
    stage = _stage("pretrain", "pretrain", train, heldout, steps=5)
    config = _config(tmp_path / "run", (stage,), scratch_tokenizer)
    path = tmp_path / "config.json"
    config.write_json(path)

    loaded = ScratchTrainingConfig.from_json(path)
    assert loaded == config
    assert loaded.tokenizer_sha256 == scratch_tokenizer.metadata["tokenizer_sha256"]
    assert loaded.optimizer.beta1 == 0.9
    assert loaded.optimizer.beta2 == 0.95
    assert loaded.optimizer.weight_decay == 0.1
    assert loaded.optimizer.grad_clip == 1.0
    assert scheduled_learning_rate(stage, 0) == stage.peak_learning_rate
    assert scheduled_learning_rate(stage, stage.steps - 1) == (
        stage.peak_learning_rate * stage.min_learning_rate_ratio
    )


def test_sft_masks_only_assistant_and_batches_keep_boundaries(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    sample = _sft_sample(
        _dialogue(
            "sft",
            [
                {"role": "system", "content": "Exact."},
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        ),
        scratch_tokenizer,
        sequence_length=32,
        line_number=1,
    )
    assistant_index = sample.tokens.index(scratch_tokenizer.role_id("assistant"))
    assert not any(sample.target_mask[:assistant_index])
    assert all(sample.target_mask[assistant_index:])
    assert sample.tokens[-2] == scratch_tokenizer.end_turn_id
    assert sample.tokens[-1] == scratch_tokenizer.eos_id

    path = tmp_path / "pretrain.jsonl"
    _write_jsonl(
        path,
        [_document("first", "abc"), _document("second", "uvwxyz")],
    )
    dataset = JsonlDataset(path, "pretrain", scratch_tokenizer, sequence_length=32)
    batch = BoundaryBatcher(
        dataset,
        batch_size=2,
        sequence_length=32,
        seed=3,
        stream_name="boundary-test",
        shuffle=False,
    ).next_batch()
    inputs = np_array(batch.inputs)
    targets = np_array(batch.targets)
    for row in range(2):
        assert inputs[row, 0] == scratch_tokenizer.bos_id
        eos_positions = [
            index
            for index, token in enumerate(targets[row])
            if token == scratch_tokenizer.eos_id
        ]
        assert len(eos_positions) == 1
        assert all(
            token == scratch_tokenizer.pad_id
            for token in targets[row, eos_positions[0] + 1 :]
        )


def test_token_shards_use_frozen_tokenizer_and_assistant_mask(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    train_dialogue = _dialogue(
        "train-dialogue",
        [
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2"},
        ],
    )
    heldout_dialogue = _dialogue(
        "heldout-dialogue",
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
    )
    shard_dir = tmp_path / "prepared" / "shards"
    write_token_shards(
        {
            "train": [train_dialogue],
            "validation": [heldout_dialogue],
            "test": [],
        },
        scratch_tokenizer,
        shard_dir,
        shard_token_limit=1_000,
    )

    dataset = TokenShardDataset(
        tmp_path / "prepared",
        "train",
        "sft",
        scratch_tokenizer,
        sequence_length=32,
    )
    sample = dataset.samples[0]
    assistant = sample.tokens.index(scratch_tokenizer.role_id("assistant"))
    assert not any(sample.target_mask[:assistant])
    assert all(sample.target_mask[assistant:])


def test_fp32_master_adamw_and_bfloat16_safe_global_clip() -> None:
    gradients = {"weight": mx.array([10_000.0, 10_000.0], dtype=mx.bfloat16)}
    clipped, norm = clip_grad_norm_fp32(gradients, 1.0)
    mx.eval(clipped, norm)
    assert mx.isfinite(norm).item()
    assert float(norm.item()) > 10_000.0
    assert mx.all(clipped["weight"] != 0).item()

    mx.random.seed(9)
    model = HybridLM(_tiny_model_config())
    model.set_dtype(mx.bfloat16)
    optimizer = FP32AdamW(1e-3, OptimizerConfig())
    optimizer.init(model.trainable_parameters())
    state = dict(tree_flatten(optimizer.state))
    moment_state = [
        value
        for key, value in state.items()
        if key.endswith(".m") or key.endswith(".v") or key.endswith(".master")
    ]
    assert moment_state
    assert all(value.dtype == mx.float32 for value in moment_state)

    inputs = mx.array([[1, 8, 9, 2, 0, 0, 0, 0]], dtype=mx.int32)
    targets = mx.array([[8, 9, 2, 0, 0, 0, 0, 0]], dtype=mx.int32)
    mask = mx.array([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=mx.float32)
    batch = type(
        "Batch", (), {"inputs": inputs, "targets": targets, "loss_mask": mask}
    )()
    loss_and_grad = nn.value_and_grad(model, masked_next_token_loss)
    loss, grads = loss_and_grad(model, batch)
    grads, _ = clip_grad_norm_fp32(grads, 1.0)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters(), optimizer.state)
    assert all(
        value.dtype == mx.bfloat16
        for _, value in tree_flatten(model.trainable_parameters())
        if mx.issubdtype(value.dtype, mx.floating)
    )


def test_mlx_foundation_loss_excludes_inactive_text_classes(
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    inactive = scratch_tokenizer.inactive_text_token_ids
    visible_id = scratch_tokenizer.encode("A")[0]
    assert inactive and visible_id not in inactive

    class FixedLossModel:
        def __call__(self, inputs):
            logits = mx.zeros((*inputs.shape, VOCAB_SIZE), dtype=mx.float32)
            inactive_peak = mx.arange(VOCAB_SIZE) == inactive[0]
            return logits + inactive_peak * 100.0

    inputs = mx.array([[scratch_tokenizer.bos_id]], dtype=mx.int32)
    targets = mx.array([[visible_id]], dtype=mx.int32)
    mask = mx.array([[1.0]], dtype=mx.float32)
    plain_batch = type(
        "Batch",
        (),
        {"inputs": inputs, "targets": targets, "loss_mask": mask},
    )()
    masked_batch = type(
        "Batch",
        (),
        {
            "inputs": inputs,
            "targets": targets,
            "loss_mask": mask,
            "text_logit_bias": _text_logit_bias(scratch_tokenizer),
        },
    )()
    unmasked = masked_next_token_loss(FixedLossModel(), plain_batch)
    masked = masked_next_token_loss(FixedLossModel(), masked_batch)
    mx.eval(unmasked, masked)

    active_class_count = VOCAB_SIZE - len(inactive)
    assert mx.isfinite(masked).item()
    assert np.isclose(masked.item(), np.log(active_class_count), atol=1e-5)
    assert masked.item() < unmasked.item()


def test_scratch_token_shard_resume_matches_uninterrupted_run(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    prepared = tmp_path / "prepared"
    write_token_shards(
        {
            "train": [
                _document("one", "one"),
                _document("two", "two"),
                _document("three", "three"),
                _document("four", "four"),
            ],
            "validation": [
                _document("five", "five"),
                _document("six", "six"),
            ],
            "test": [],
        },
        scratch_tokenizer,
        prepared / "shards",
        shard_token_limit=10_000,
    )

    full_config = _config(
        tmp_path / "full",
        (
            _stage(
                "pretrain", "pretrain", prepared, prepared, data_format="token_shards"
            ),
        ),
        scratch_tokenizer,
    )
    full = ScratchTrainer(full_config)
    full.run()

    resumed_config = _config(
        tmp_path / "resumed",
        (
            _stage(
                "pretrain", "pretrain", prepared, prepared, data_format="token_shards"
            ),
        ),
        scratch_tokenizer,
    )
    interrupted = ScratchTrainer(resumed_config)
    interrupted.run(max_steps=1)
    resumed = ScratchTrainer(resumed_config, resume=tmp_path / "resumed")
    resumed.run()

    for (_, expected), (_, actual) in zip(
        tree_flatten(full.model.parameters()),
        tree_flatten(resumed.model.parameters()),
        strict=True,
    ):
        assert mx.array_equal(expected, actual).item()
    full_optimizer = dict(tree_flatten(full.optimizer.state))
    resumed_optimizer = dict(tree_flatten(resumed.optimizer.state))
    assert full_optimizer.keys() == resumed_optimizer.keys()
    for key, expected in full_optimizer.items():
        assert mx.array_equal(expected, resumed_optimizer[key]).item()
    assert resumed.state.global_step == full.state.global_step == 2
    assert resumed.state.accepted_updates == full.state.accepted_updates == 2
    assert resumed.complete

    pointer = json.loads((tmp_path / "resumed" / "last.json").read_text())
    checkpoint = tmp_path / "resumed" / pointer["checkpoint"]
    assert {
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
        "manifest.json",
    } <= {path.name for path in checkpoint.iterdir()}
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["scratch_only"]
    assert (
        manifest["tokenizer_sha256"] == scratch_tokenizer.metadata["tokenizer_sha256"]
    )
    assert (tmp_path / "resumed" / "best.json").is_file()


def test_pretraining_then_assistant_only_sft_is_staged(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    pretrain = tmp_path / "pretrain.jsonl"
    pretrain_eval = tmp_path / "pretrain-eval.jsonl"
    sft = tmp_path / "sft.jsonl"
    sft_eval = tmp_path / "sft-eval.jsonl"
    _write_jsonl(
        pretrain,
        [_document("numbers", "numbers"), _document("arithmetic", "arithmetic")],
    )
    _write_jsonl(pretrain_eval, [_document("heldout", "heldout")])
    conversation = _dialogue(
        "conversation",
        [
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2"},
        ],
    )
    heldout_conversation = _dialogue(
        "heldout-conversation",
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
    )
    _write_jsonl(sft, [conversation])
    _write_jsonl(sft_eval, [heldout_conversation])
    config = _config(
        tmp_path / "staged",
        (
            _stage("pretrain", "pretrain", pretrain, pretrain_eval, steps=1),
            _stage("conversation", "sft", sft, sft_eval, steps=1),
        ),
        scratch_tokenizer,
    )
    trainer = ScratchTrainer(config)
    state = trainer.run()

    assert trainer.complete
    assert state.stage_index == 2
    assert state.global_step == 2
    events = [
        json.loads(line)
        for line in (tmp_path / "staged" / "metrics.jsonl").read_text().splitlines()
    ]
    trained_stages = {event["stage"] for event in events if event["event"] == "train"}
    assert trained_stages == {"pretrain", "conversation"}
    provenance = json.loads((tmp_path / "staged" / "provenance.json").read_text())
    assert provenance["external_weights"] is False
    assert (
        provenance["tokenizer"]["sha256"]
        == scratch_tokenizer.metadata["tokenizer_sha256"]
    )
    assert provenance["later_ablations_not_enabled"] == [
        "Muon",
        "multi-token prediction",
        "RL",
    ]


def test_loss_spike_skips_update_and_tokenizer_mismatch_is_rejected(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    _write_jsonl(train, [_document("one", "one"), _document("two", "two")])
    _write_jsonl(heldout, [_document("heldout", "heldout")])
    stage = _stage("guarded", "pretrain", train, heldout, steps=1)
    config = ScratchTrainingConfig(
        stages=(stage,),
        tokenizer_path=str(scratch_tokenizer.artifact_dir),
        tokenizer_sha256=str(scratch_tokenizer.metadata["tokenizer_sha256"]),
        model=_tiny_model_config(),
        guards=GuardConfig(loss_spike_warmup_updates=0),
        output_dir=str(tmp_path / "guarded"),
        seed=19,
        parameter_dtype="float32",
    )
    trainer = ScratchTrainer(config)
    before = dict(tree_flatten(trainer.model.parameters()))
    zero_gradients = tree_map(
        lambda value: mx.zeros_like(value),
        trainer.model.trainable_parameters(),
    )
    trainer.state.loss_ema = 1.0
    trainer.loss_and_grad = lambda _model, _batch: (mx.array(10.0), zero_gradients)
    state = trainer.run(max_steps=1)

    assert state.accepted_updates == 0
    assert state.skipped_updates == 1
    after = dict(tree_flatten(trainer.model.parameters()))
    assert before.keys() == after.keys()
    assert all(
        mx.array_equal(value, after[key]).item() for key, value in before.items()
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "guarded" / "metrics.jsonl").read_text().splitlines()
    ]
    assert any(
        event["event"] == "skipped_update" and event["reason"] == "loss_spike"
        for event in events
    )

    wrong_tokenizer_config = ScratchTrainingConfig(
        stages=(stage,),
        tokenizer_path=str(scratch_tokenizer.artifact_dir),
        tokenizer_sha256="0" * 64,
        model=_tiny_model_config(),
        output_dir=str(tmp_path / "wrong-tokenizer"),
        parameter_dtype="float32",
    )
    with pytest.raises(ValueError, match="loaded tokenizer checksum"):
        ScratchTrainer(wrong_tokenizer_config)

    bare_weights = tmp_path / "external.safetensors"
    bare_weights.write_bytes(b"not a trainer checkpoint")
    with pytest.raises(ValueError, match="bare or external"):
        _resolve_checkpoint_path(bare_weights)


def np_array(value: mx.array) -> np.ndarray:
    return np.array(value)
