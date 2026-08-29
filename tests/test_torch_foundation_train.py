from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

torch = pytest.importorskip("torch")
safe_open = pytest.importorskip("safetensors").safe_open

from mini_arithmetic.hybrid_config import CONV, HybridConfig  # noqa: E402
from mini_arithmetic.portable_bundle import (  # noqa: E402
    PORTABLE_STATE_FORMAT,
    PORTABLE_TRAINED_INITIALIZATION_FORMAT,
    create_portable_bundle,
    load_portable_bundle,
)
from mini_arithmetic.portable import BackendSelection  # noqa: E402
from mini_arithmetic.portable_serve import build_service  # noqa: E402
from mini_arithmetic.prepare_data import write_token_shards  # noqa: E402
from mini_arithmetic.torch_foundation_train import (  # noqa: E402
    DeterministicBatcher,
    FoundationTrainingConfig,
    FoundationTrainingError,
    PreparedDataset,
    StageConfig,
    TorchFoundationTrainer,
    _assistant_target_mask,
    _text_cross_entropy_sum,
    export_trained_portable_bundle,
)


def _tiny_model_config() -> HybridConfig:
    return HybridConfig(
        vocab_size=32_768,
        context_length=12,
        model_dim=4,
        ffn_dim=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        conv_kernel_size=2,
        layer_schedule=(CONV,),
    )


def _portable_bundle(tmp_path: Path):
    config_path = tmp_path / "model.json"
    payload = asdict(_tiny_model_config())
    payload["layer_schedule"] = list(payload["layer_schedule"])
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return create_portable_bundle(
        tmp_path / "bundle",
        config_path=config_path,
        backend="torch",
        model_size="tiny",
        seed=17,
        parameter_dtype="float32",
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


def _dialogue(record_id: str, user: str, assistant: str) -> dict[str, object]:
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    canonical = [[message["role"], message["content"]] for message in messages]
    return {
        "type": "dialogue",
        "messages": messages,
        "source": "unit-test",
        "license": "CC0-1.0",
        "provenance": {"fixture": record_id},
        "record_id": record_id,
        "content_sha256": _content_hash("dialogue", canonical),
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _training_config(
    output: Path,
    train: Path,
    validation: Path,
    *,
    steps: int = 2,
    checkpoint_keep_last: int = 2,
    checkpoint_milestone_every: int = 5_000,
) -> FoundationTrainingConfig:
    return FoundationTrainingConfig(
        stages=(
            StageConfig(
                name="foundation",
                objective="foundation",
                train_path=str(train),
                eval_path=str(validation),
                data_format="audited_jsonl",
                steps=steps,
                batch_size=1,
                sequence_length=8,
                peak_learning_rate=1e-3,
                warmup_steps=1,
                min_learning_rate_ratio=0.2,
                eval_every=1,
                eval_batches=1,
                save_every=1,
                report_every=1,
            ),
        ),
        output_dir=str(output),
        seed=91,
        device="cpu",
        compute_dtype="float32",
        gradient_accumulation_steps=2,
        checkpoint_keep_last=checkpoint_keep_last,
        checkpoint_milestone_every=checkpoint_milestone_every,
    )


def test_foundation_packing_and_dialogue_assistant_only_mask(tmp_path: Path) -> None:
    bundle = _portable_bundle(tmp_path)
    train = tmp_path / "train.jsonl"
    _write_jsonl(train, [_document("a", "a"), _document("b", "b")])
    dataset = PreparedDataset(
        train,
        data_format="audited_jsonl",
        split="train",
        objective="foundation",
        tokenizer=bundle.tokenizer,
        sequence_length=8,
    )
    batcher = DeterministicBatcher(
        dataset,
        objective="foundation",
        batch_size=1,
        sequence_length=8,
        seed=1,
        stream_name="packing",
        shuffle=False,
    )
    batch = batcher.next_batch(torch.device("cpu"))
    # More than one bounded document is placed in the row when it fits.
    assert int((batch.inputs[0] == bundle.tokenizer.bos_id).sum()) >= 2
    assert any(
        int(input_id) == bundle.tokenizer.eos_id
        and int(target_id) == bundle.tokenizer.bos_id
        for input_id, target_id in zip(batch.inputs[0], batch.targets[0], strict=True)
    )
    assert batch.loss_tokens == int(batch.loss_mask.sum())

    tokens, boundaries = bundle.tokenizer.encode_dialogue_with_boundaries(
        [("user", "hello"), ("assistant", "hi")]
    )
    mask = _assistant_target_mask(tokens, boundaries, bundle.tokenizer)
    assistant_offset = tokens.index(bundle.tokenizer.role_id("assistant"))
    assert not any(mask[:assistant_offset])
    assert all(mask[assistant_offset:])


def test_foundation_text_loss_excludes_inactive_classes_with_finite_reference(
    tmp_path: Path,
) -> None:
    bundle = _portable_bundle(tmp_path)
    active = torch.ones(bundle.config.vocab_size, dtype=torch.bool)
    inactive = torch.tensor(bundle.tokenizer.inactive_text_token_ids, dtype=torch.long)
    active[inactive] = False
    logits = torch.zeros((1, 1, bundle.config.vocab_size), dtype=torch.float32)
    logits[..., inactive] = 10_000.0
    target_id = bundle.tokenizer.bos_id
    logits[..., target_id] = 0.75
    targets = torch.tensor([[target_id]], dtype=torch.long)
    loss_mask = torch.ones((1, 1), dtype=torch.float32)

    actual = _text_cross_entropy_sum(logits, targets, loss_mask, active)
    expected = torch.logsumexp(logits[0, 0, active], dim=0) - logits[0, 0, target_id]

    assert bool(torch.isfinite(actual))
    torch.testing.assert_close(actual, expected)
    with pytest.raises(FoundationTrainingError, match="inactive text token"):
        _text_cross_entropy_sum(
            logits,
            inactive[:1].reshape(1, 1),
            loss_mask,
            active,
        )


def test_prepared_token_shards_are_supported(tmp_path: Path) -> None:
    bundle = _portable_bundle(tmp_path)
    prepared = tmp_path / "prepared"
    write_token_shards(
        {
            "train": [_dialogue("train", "hello", "hello there")],
            "validation": [_dialogue("heldout", "goodbye", "see you")],
            "test": [],
        },
        bundle.tokenizer,
        prepared / "shards",
        shard_token_limit=1_000,
    )
    dataset = PreparedDataset(
        prepared,
        data_format="token_shards",
        split="train",
        objective="dialogue",
        tokenizer=bundle.tokenizer,
        sequence_length=32,
    )
    assert len(dataset) == 1
    assistant_offset = dataset.samples[0].tokens.index(
        bundle.tokenizer.role_id("assistant")
    )
    assert not any(dataset.samples[0].target_mask[:assistant_offset])

    foundation_dataset = PreparedDataset(
        prepared,
        data_format="token_shards",
        split="train",
        objective="foundation",
        tokenizer=bundle.tokenizer,
        sequence_length=12,
    )
    assert foundation_dataset.samples[0].__class__.__name__ == "ShardFoundationSample"
    assert foundation_dataset.samples[0].tokens[0] == bundle.tokenizer.bos_id

    shard_config = FoundationTrainingConfig(
        stages=(
            StageConfig(
                name="prepared-foundation",
                objective="foundation",
                train_path=str(prepared),
                eval_path=str(prepared),
                data_format="token_shards",
                steps=2,
                sequence_length=12,
                warmup_steps=1,
                eval_every=1,
                eval_batches=1,
                save_every=1,
                report_every=1,
            ),
        ),
        output_dir=str(tmp_path / "shard-run"),
        device="cpu",
        compute_dtype="float32",
    )
    trainer = TorchFoundationTrainer(bundle.root, shard_config)
    assert len(trainer.datasets[(0, "train")]) >= 1
    assert len(trainer.datasets[(0, "validation")]) >= 1


@pytest.mark.parametrize("overlap", ["record_id", "content_sha256"])
def test_audited_train_evaluation_leakage_is_rejected(
    tmp_path: Path,
    overlap: str,
) -> None:
    bundle = _portable_bundle(tmp_path)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    if overlap == "record_id":
        train_record = _document("shared-id", "training content")
        validation_record = _document("shared-id", "held-out content")
    else:
        train_record = _document("train-id", "duplicated content")
        validation_record = _document("validation-id", "duplicated content")
    _write_jsonl(train, [train_record])
    _write_jsonl(validation, [validation_record])

    with pytest.raises(FoundationTrainingError, match="train/evaluation leakage"):
        TorchFoundationTrainer(
            bundle.root,
            _training_config(tmp_path / "leaky-run", train, validation),
        )


def test_tiny_cpu_training_resume_is_exact_and_checkpoint_is_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _portable_bundle(tmp_path)
    train = tmp_path / "foundation-train.jsonl"
    validation = tmp_path / "foundation-validation.jsonl"
    _write_jsonl(
        train,
        [
            _document("one", "one"),
            _document("two", "two"),
            _document("three", "three"),
        ],
    )
    _write_jsonl(validation, [_document("heldout", "heldout")])

    full = TorchFoundationTrainer(
        bundle.root,
        _training_config(tmp_path / "full", train, validation),
    )
    full.run()

    resume_config = _training_config(tmp_path / "resumed", train, validation)
    interrupted = TorchFoundationTrainer(bundle.root, resume_config)
    interrupted.run(max_optimizer_steps=1)
    with pytest.raises(ValueError, match="completed every configured stage"):
        export_trained_portable_bundle(
            bundle.root,
            tmp_path / "resumed",
            tmp_path / "partial-models" / "portable-tiny",
        )
    assert not (tmp_path / "partial-models" / "portable-tiny").exists()
    resumed = TorchFoundationTrainer(
        bundle.root,
        resume_config,
        resume=tmp_path / "resumed",
    )
    resumed.run()

    assert full.complete and resumed.complete
    assert full.state.global_step == resumed.state.global_step == 2
    assert full.state.tokens_seen == resumed.state.tokens_seen
    for name, expected in full.model.canonical_state_dict().items():
        torch.testing.assert_close(
            expected,
            resumed.model.canonical_state_dict()[name],
            atol=0,
            rtol=0,
        )

    pointer = json.loads((tmp_path / "resumed" / "last.json").read_text())
    checkpoint = tmp_path / "resumed" / pointer["checkpoint"]
    assert {
        "manifest.json",
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
    } == {item.name for item in checkpoint.iterdir()}
    with safe_open(
        str(checkpoint / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        assert handle.metadata()["format"] == PORTABLE_STATE_FORMAT
        assert set(handle.keys()) == set(resumed.model.canonical_state_dict())
    provenance = json.loads((tmp_path / "resumed" / "provenance.json").read_text())
    assert provenance["explicit_foundation_training"] is True
    assert provenance["invoked_by_talk_to_train"] is False

    original_weights_sha = load_portable_bundle(bundle.root).manifest["weights_sha256"]
    served_models = tmp_path / "served-models"
    exported = export_trained_portable_bundle(
        bundle.root,
        tmp_path / "resumed",
        served_models / "portable-tiny",
    )
    assert exported.manifest["initialization"] == PORTABLE_TRAINED_INITIALIZATION_FORMAT
    assert exported.manifest["training"]["global_step"] == 2
    assert exported.manifest["training"]["training_complete"] is True
    assert exported.manifest["training"]["total_steps"] == 2
    assert (
        exported.manifest["training"]["parent_weights_sha256"] == original_weights_sha
    )
    assert exported.manifest["weights_sha256"] != original_weights_sha
    assert (
        load_portable_bundle(bundle.root).manifest["weights_sha256"]
        == original_weights_sha
    )
    with pytest.raises(FileExistsError):
        export_trained_portable_bundle(
            bundle.root,
            tmp_path / "resumed",
            served_models / "portable-tiny",
        )

    selection = BackendSelection(
        requested="torch",
        backend="torch",
        accelerator="cpu",
        ready=True,
        system="TestOS",
        machine="x86_64",
        install_extra="torch",
        reason="bounded export test",
    )
    monkeypatch.setattr(
        "mini_arithmetic.portable_serve.select_backend",
        lambda requested: selection,
    )
    service, lease = build_service(
        root=tmp_path,
        backend="torch",
        model_size="tiny",
        db_path=tmp_path / "served.sqlite3",
        model_root=served_models,
        personal_root=tmp_path / "personal",
        seed=0,
        rank=2,
        max_new_tokens=1,
    )
    try:
        assert service.bundle.root == exported.root
        assert (
            service.bundle.manifest["initialization"]
            == PORTABLE_TRAINED_INITIALIZATION_FORMAT
        )
    finally:
        service.close()
        lease.close()

    with (checkpoint / "model.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        export_trained_portable_bundle(
            bundle.root,
            tmp_path / "resumed",
            tmp_path / "tampered-models" / "portable-tiny",
        )


def test_checkpoint_retention_keeps_latest_final_and_sparse_milestones(
    tmp_path: Path,
) -> None:
    bundle = _portable_bundle(tmp_path)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(
        train,
        [_document("one", "one"), _document("two", "two")],
    )
    _write_jsonl(validation, [_document("heldout", "heldout")])
    run_dir = tmp_path / "retained-run"
    config = _training_config(
        run_dir,
        train,
        validation,
        steps=4,
        checkpoint_keep_last=1,
        checkpoint_milestone_every=2,
    )
    trainer = TorchFoundationTrainer(bundle.root, config)
    unmanaged_export = trainer.checkpoint_dir / "export-artifact"
    unmanaged_export.mkdir()
    (unmanaged_export / "model.safetensors").write_bytes(b"do-not-delete")

    trainer.run()

    checkpoints = sorted(
        path
        for path in trainer.checkpoint_dir.iterdir()
        if path.name.startswith("step-")
    )
    assert [path.name[:13] for path in checkpoints] == [
        "step-00000002",
        "step-00000004",
    ]
    assert unmanaged_export.is_dir()
    assert (unmanaged_export / "model.safetensors").read_bytes() == b"do-not-delete"
    pointer = json.loads((run_dir / "last.json").read_text(encoding="utf-8"))
    final_checkpoint = run_dir / pointer["checkpoint"]
    final_manifest = json.loads(
        (final_checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["training_complete"] is True

    exported = export_trained_portable_bundle(
        bundle.root,
        run_dir,
        tmp_path / "retained-export" / "portable-tiny",
    )
    assert exported.weights_path.is_file()
    assert final_checkpoint.is_dir()


@pytest.mark.skipif(
    not os.environ.get("TINYDOG_MLX_PYTHON"),
    reason="set TINYDOG_MLX_PYTHON to an MLX-capable interpreter",
)
def test_exported_torch_checkpoint_loads_and_forwards_in_mlx(tmp_path: Path) -> None:
    bundle = _portable_bundle(tmp_path)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(train, [_document("train", "hello")])
    _write_jsonl(validation, [_document("heldout", "goodbye")])
    config = _training_config(tmp_path / "run", train, validation)
    trainer = TorchFoundationTrainer(bundle.root, config)
    trainer.run()
    exported = export_trained_portable_bundle(
        bundle.root,
        tmp_path / "run",
        tmp_path / "models" / "portable-tiny",
    )

    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    script = """
import json
import sys
from mini_arithmetic.mlx_personal import MLXPersonalConfig, create_mlx_personal_backend
from mini_arithmetic.portable_bundle import load_portable_bundle

bundle = load_portable_bundle(sys.argv[1])
config = MLXPersonalConfig(rank=2, context_length=12, max_tokens=1)
_, runtime = create_mlx_personal_backend(bundle, config=config)
reply = runtime.answer([('user', 'hi')], max_tokens=1)
print(json.dumps({'backend': runtime.status()['backend'], 'reply_type': type(reply).__name__}))
"""
    result = subprocess.run(
        [os.environ["TINYDOG_MLX_PYTHON"], "-c", script, str(exported.root)],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(result.stdout.strip())
    assert payload == {"backend": "mlx", "reply_type": "str"}
