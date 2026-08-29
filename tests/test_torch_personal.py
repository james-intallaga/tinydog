from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mini_arithmetic.hybrid_config import CONV, HybridConfig  # noqa: E402
from mini_arithmetic.personal_training import (  # noqa: E402
    DialogueExample,
    PersonalAdapterRuntime,
    PersonalAdapterTrainer,
    PersonalAdapterTrainingRequest,
    PrivateAdapterRegistry,
)
from mini_arithmetic.personal_foundation import (  # noqa: E402
    PersonalFoundationArtifact,
    PersonalFoundationError,
    PersonalFoundationManifest,
    PersonalFoundationRuntime,
    PersonalFoundationTrainer,
    PersonalFoundationTrainingRequest,
)
from mini_arithmetic.portable import select_backend  # noqa: E402
from mini_arithmetic.portable_bundle import (  # noqa: E402
    create_portable_bundle,
    load_portable_bundle,
)
from mini_arithmetic.torch_hybrid import HybridLM  # noqa: E402
from mini_arithmetic.torch_personal import (  # noqa: E402
    TorchPersonalAdapter,
    TorchPersonalConfig,
    TorchFoundationConfig,
    TorchPersonalModel,
    _masked_loss,
    _TokenBatch,
    _text_logit_bias,
    create_torch_personal_backend,
    create_torch_personal_foundation_backend,
)


@pytest.fixture(scope="module")
def portable_model(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("torch-personal")
    config = HybridConfig(
        vocab_size=32_768,
        context_length=64,
        model_dim=8,
        ffn_dim=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        conv_kernel_size=2,
        layer_schedule=(CONV,),
    )
    payload = asdict(config)
    payload["layer_schedule"] = list(config.layer_schedule)
    config_path = root / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return create_portable_bundle(
        root / "bundle",
        config_path=config_path,
        backend="torch",
        model_size="tiny",
        seed=29,
        parameter_dtype="float32",
    )


@pytest.fixture(scope="module")
def personal_config() -> TorchPersonalConfig:
    # The high learning rate is intentionally restricted to this 8-wide CPU
    # fixture so the held-out improvement is both quick and deterministic.
    return TorchPersonalConfig(
        rank=2,
        alpha=4.0,
        learning_rate=3e-2,
        steps=12,
        seed=7,
    )


def _example(identifier: int, *, response: str = "Hello, friend!") -> DialogueExample:
    prompt = f"Please greet me number {identifier}."
    digest = hashlib.sha256(
        f"typed-dialogue-v1\0{prompt}\0{response}".encode("utf-8")
    ).hexdigest()
    return DialogueExample(
        lesson_id=identifier,
        session_id="private-session",
        prompt=prompt,
        human_response=response,
        content_sha256=digest,
    )


def _request(
    runtime,
    work_dir: Path,
    *,
    run_id: int,
    adapter_id: str,
    start: int = 1,
    parent_id: str | None = None,
    parent_path: Path | None = None,
    replay_examples: tuple[DialogueExample, ...] = (),
    retention_examples: tuple[DialogueExample, ...] = (),
) -> PersonalAdapterTrainingRequest:
    identity = runtime.personal_runtime_identity()
    work_dir.mkdir(parents=True)
    examples = tuple(_example(index) for index in range(start, start + 5))
    return PersonalAdapterTrainingRequest(
        run_id=run_id,
        adapter_id=adapter_id,
        backend=identity.backend,
        foundation_sha256=identity.foundation_sha256,
        adapter_schema_sha256=identity.adapter_schema_sha256,
        parent_adapter_id=parent_id,
        parent_adapter_path=parent_path,
        training_examples=examples[:4],
        validation_examples=examples[4:],
        work_dir=work_dir,
        replay_examples=replay_examples,
        retention_examples=retention_examples,
    )


def test_only_adapter_tensors_receive_gradients_and_foundation_stays_exact(
    portable_model,
    personal_config: TorchPersonalConfig,
) -> None:
    from safetensors.torch import load_file

    foundation = HybridLM(portable_model.config)
    foundation.load_canonical_state_dict(
        load_file(str(portable_model.weights_path), device="cpu")
    )
    before = {
        name: value.detach().clone() for name, value in foundation.state_dict().items()
    }
    torch.manual_seed(personal_config.seed)
    adapter = TorchPersonalAdapter(portable_model.config, personal_config)
    model = TorchPersonalModel(foundation, adapter)
    model.assert_parameter_boundary()
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=1e-2)

    tokens = torch.tensor(
        [[portable_model.tokenizer.bos_id, portable_model.tokenizer.role_id("user")]],
        dtype=torch.long,
    )
    loss = model(tokens).float().square().mean()
    loss.backward()

    assert all(parameter.grad is None for parameter in foundation.parameters())
    assert any(parameter.grad is not None for parameter in adapter.parameters())
    adapter_before = {
        name: value.detach().clone() for name, value in adapter.state_dict().items()
    }
    optimizer.step()
    assert all(
        torch.equal(before[name], value)
        for name, value in foundation.state_dict().items()
    )
    assert any(
        not torch.equal(adapter_before[name], value)
        for name, value in adapter.state_dict().items()
    )


def test_trainer_returns_improving_cumulative_safetensors_candidate(
    tmp_path: Path,
    portable_model,
    personal_config: TorchPersonalConfig,
) -> None:
    selection = select_backend("torch")
    trainer, runtime = create_torch_personal_backend(
        portable_model,
        selection,
        config=personal_config,
    )
    assert isinstance(trainer, PersonalAdapterTrainer)
    assert isinstance(runtime, PersonalAdapterRuntime)
    foundation_bytes = portable_model.weights_path.read_bytes()
    request = _request(
        runtime,
        tmp_path / "work",
        run_id=1,
        adapter_id="personal-first",
    )

    candidate = trainer.train_adapter(request)

    assert candidate.candidate_holdout_loss < candidate.baseline_holdout_loss
    assert candidate.metrics["assistant_only_targets"] is True
    assert candidate.metrics["foundation_gradients"] is False
    assert candidate.metrics["cumulative_adapter"] is True
    assert candidate.artifact_path.parent == request.work_dir.resolve()
    assert candidate.artifact_path.suffix == ".safetensors"
    assert portable_model.weights_path.read_bytes() == foundation_bytes
    from safetensors.torch import load_file

    state = load_file(str(candidate.artifact_path), device="cpu")
    assert state
    assert all(name.startswith("blocks.") for name in state)
    assert not any("foundation" in name for name in state)


def test_runtime_activation_changes_logits_and_uses_request_local_cache(
    tmp_path: Path,
    portable_model,
    personal_config: TorchPersonalConfig,
) -> None:
    trainer, runtime = create_torch_personal_backend(
        portable_model,
        device="cpu",
        config=personal_config,
    )
    request = _request(
        runtime,
        tmp_path / "work",
        run_id=2,
        adapter_id="personal-live",
    )
    candidate = trainer.train_adapter(request)
    registry = PrivateAdapterRegistry(
        tmp_path / "registry", maximum_adapter_bytes=1_000_000
    )
    artifact = registry.publish(request, candidate)
    prompt = torch.tensor(
        [
            [
                portable_model.tokenizer.bos_id,
                portable_model.tokenizer.role_id("assistant"),
            ]
        ],
        dtype=torch.long,
    )
    with torch.inference_mode():
        before = runtime._active_model(prompt).detach().clone()
    prepared = runtime.prepare_personal_adapter(artifact)
    runtime.activate_prepared_personal_adapter(prepared, artifact.manifest.adapter_id)
    with torch.inference_mode():
        after = runtime._active_model(prompt).detach().clone()

    assert not torch.equal(before, after)
    assert runtime.personal_runtime_identity().active_adapter_id == "personal-live"
    runtime.answer((("user", "Hello"),), max_tokens=3)
    status = runtime.status()
    assert status["cached_decode"] is True
    assert status["cache_scope"] == "per-answer"
    assert status["answer_calls"] == 1
    assert status["cached_tokens"] >= 1
    assert status["foundation_frozen"] is True
    runtime.activate_foundation_only()
    assert runtime.personal_runtime_identity().active_adapter_id is None


def test_trainer_replays_history_and_reports_retention_losses(
    tmp_path: Path,
    portable_model,
    personal_config: TorchPersonalConfig,
) -> None:
    trainer, runtime = create_torch_personal_backend(
        portable_model,
        device="cpu",
        config=personal_config,
    )
    request = _request(
        runtime,
        tmp_path / "replay-work",
        run_id=12,
        adapter_id="personal-replay",
        replay_examples=(_example(20), _example(21)),
        retention_examples=(_example(30), _example(31)),
    )

    candidate = trainer.train_adapter(request)

    assert candidate.baseline_retention_loss is not None
    assert candidate.candidate_retention_loss is not None
    assert candidate.baseline_retention_loss >= 0.0
    assert candidate.candidate_retention_loss >= 0.0
    assert candidate.metrics["new_training_examples"] == 4
    assert candidate.metrics["replay_examples"] == 2
    assert candidate.metrics["retention_examples"] == 2
    assert candidate.metrics["foundation_unchanged"] is True


def test_runtime_keeps_newest_user_suffix_and_training_rejects_oversized_target(
    portable_model,
    personal_config: TorchPersonalConfig,
) -> None:
    _, runtime = create_torch_personal_backend(
        portable_model,
        device="cpu",
        config=personal_config,
    )
    text = "🧪" * 1_000
    content_ids = portable_model.tokenizer.encode(text)
    budget = portable_model.config.context_length - 4
    assert len(content_ids) > budget

    prompt = runtime._prompt_ids((("user", text),))

    assert len(prompt) == portable_model.config.context_length
    assert prompt[0] == portable_model.tokenizer.bos_id
    assert prompt[1] == portable_model.tokenizer.role_id("user")
    assert prompt[2:-2] == content_ids[-budget:]
    assert prompt[-2] == portable_model.tokenizer.end_turn_id
    assert prompt[-1] == portable_model.tokenizer.role_id("assistant")
    assert runtime.personal_training_target_fits("hello", text) is False


def test_inactive_text_logits_cannot_consume_generation_or_ce_mass(
    portable_model,
    personal_config: TorchPersonalConfig,
) -> None:
    _, runtime = create_torch_personal_backend(
        portable_model,
        device="cpu",
        config=personal_config,
    )
    inactive = portable_model.tokenizer.inactive_text_token_ids
    visible_id = portable_model.tokenizer.encode("A")[0]
    blank_id = portable_model.tokenizer.encode(" ")[0]
    control_id = portable_model.tokenizer.encode("\x00")[0]
    assistant_id = portable_model.tokenizer.role_id("assistant")
    assert visible_id not in inactive

    class FixedLogits:
        def forward_cached(self, tokens, cache=None):
            logits = torch.full(
                (*tokens.shape, portable_model.config.vocab_size),
                -20.0,
                device=tokens.device,
            )
            logits[..., inactive[0]] = 140.0
            logits[..., portable_model.tokenizer.unk_id] = 139.0
            logits[..., assistant_id] = 138.0
            logits[..., control_id] = 137.0
            logits[..., portable_model.tokenizer.eos_id] = 136.0
            logits[..., portable_model.tokenizer.end_turn_id] = 135.0
            logits[..., blank_id] = 134.0
            logits[..., visible_id] = 133.0
            if cache is not None:
                logits[..., visible_id] = 150.0
            return logits, object()

    runtime._active_model = FixedLogits()
    generated = runtime.answer((("user", "hello"),), max_tokens=3)
    assert generated == portable_model.tokenizer.decode([visible_id] * 3).strip()
    assert generated
    assert "\x00" not in generated

    class FixedLossModel(torch.nn.Module):
        def forward(self, inputs):
            logits = torch.zeros(
                (*inputs.shape, portable_model.config.vocab_size),
                dtype=torch.float32,
            )
            logits[..., inactive[0]] = 100.0
            return logits

    batch = _TokenBatch(
        inputs=torch.tensor([[portable_model.tokenizer.bos_id]]),
        targets=torch.tensor([[visible_id]]),
        mask=torch.tensor([[1.0]]),
    )
    unmasked = _masked_loss(FixedLossModel(), batch)
    bias = _text_logit_bias(
        portable_model.config.vocab_size,
        inactive,
        device=torch.device("cpu"),
    )
    masked = _masked_loss(FixedLossModel(), batch, bias)
    assert torch.isfinite(masked)
    assert masked < unmasked

    structural = _TokenBatch(
        inputs=batch.inputs,
        targets=torch.tensor([[portable_model.tokenizer.end_turn_id]]),
        mask=batch.mask,
    )
    assert torch.isfinite(_masked_loss(FixedLossModel(), structural, bias))


def test_parent_adapter_survives_restart_and_is_the_next_cumulative_base(
    tmp_path: Path,
    portable_model,
    personal_config: TorchPersonalConfig,
) -> None:
    first_trainer, first_runtime = create_torch_personal_backend(
        portable_model, device="cpu", config=personal_config
    )
    first_request = _request(
        first_runtime,
        tmp_path / "first-work",
        run_id=3,
        adapter_id="personal-parent",
    )
    first_candidate = first_trainer.train_adapter(first_request)
    registry = PrivateAdapterRegistry(
        tmp_path / "registry", maximum_adapter_bytes=1_000_000
    )
    parent = registry.publish(first_request, first_candidate)

    # These instances reload the canonical foundation and parent file; they do
    # not share mutable tensors with the first process-shaped pair.
    next_trainer, restarted_runtime = create_torch_personal_backend(
        portable_model, device="cpu", config=personal_config
    )
    restarted_runtime.activate_prepared_personal_adapter(
        restarted_runtime.prepare_personal_adapter(parent),
        parent.manifest.adapter_id,
    )
    assert (
        restarted_runtime.personal_runtime_identity().active_adapter_id
        == "personal-parent"
    )
    next_request = _request(
        restarted_runtime,
        tmp_path / "next-work",
        run_id=4,
        adapter_id="personal-child",
        start=6,
        parent_id=parent.manifest.adapter_id,
        parent_path=parent.artifact_path,
    )
    child = next_trainer.train_adapter(next_request)

    from safetensors.torch import load_file

    parent_state = load_file(str(parent.artifact_path), device="cpu")
    child_state = load_file(str(child.artifact_path), device="cpu")
    assert set(parent_state) == set(child_state)
    assert any(
        not torch.equal(parent_state[name], child_state[name]) for name in parent_state
    )
    assert child.foundation_sha256_before == parent.manifest.foundation_sha256
    assert child.foundation_sha256_after == parent.manifest.foundation_sha256


def _foundation_request(
    runtime,
    work_dir: Path,
    *,
    run_id: int = 40,
    candidate_id: str = "foundation-candidate",
    replay: tuple[DialogueExample, ...] = (),
    cancelled: threading.Event | None = None,
) -> PersonalFoundationTrainingRequest:
    work_dir.mkdir(parents=True)
    examples = tuple(_example(index) for index in range(201, 206))
    return PersonalFoundationTrainingRequest(
        run_id=run_id,
        candidate_foundation_id=candidate_id,
        parent=runtime.personal_foundation_source(),
        new_examples=examples,
        monitoring_examples=(examples[-1],),
        replay_examples=replay,
        work_dir=work_dir,
        cancellation_event=cancelled,
    )


def _foundation_artifact(
    request: PersonalFoundationTrainingRequest,
    candidate,
) -> PersonalFoundationArtifact:
    return PersonalFoundationArtifact(
        directory=candidate.bundle_path,
        bundle_path=candidate.bundle_path,
        manifest_path=candidate.bundle_path / "personal-foundation.json",
        manifest=PersonalFoundationManifest(
            identity=candidate.identity,
            parent_foundation_id=candidate.parent_foundation_id,
            parent_bundle_sha256=candidate.parent_bundle_sha256,
            run_id=request.run_id,
            new_lesson_ids=tuple(item.lesson_id for item in request.new_examples),
            monitoring_lesson_ids=tuple(
                item.lesson_id for item in request.monitoring_examples
            ),
            replay_lesson_ids=tuple(item.lesson_id for item in request.replay_examples),
            created_at="2026-08-28T00:00:00Z",
            optimizer_state_included=False,
        ),
    )


def test_direct_torch_foundation_trains_all_five_and_publishes_complete_bundle(
    tmp_path: Path,
    portable_model,
) -> None:
    config = TorchFoundationConfig(
        learning_rate=1e-2,
        epochs=1,
        seed=31,
        context_length=32,
        disk_reserve_bytes=0,
    )
    trainer, runtime = create_torch_personal_foundation_backend(
        portable_model,
        device="cpu",
        config=config,
    )
    assert isinstance(trainer, PersonalFoundationTrainer)
    assert isinstance(runtime, PersonalFoundationRuntime)
    parent_bytes = portable_model.weights_path.read_bytes()
    request = _foundation_request(
        runtime,
        tmp_path / "foundation-work",
        replay=(_example(301), _example(302)),
    )

    candidate = trainer.train_foundation(request)
    trained = load_portable_bundle(candidate.bundle_path)

    assert portable_model.weights_path.read_bytes() == parent_bytes
    assert candidate.identity.weights_sha256 != request.parent.identity.weights_sha256
    assert candidate.identity.bundle_sha256 != request.parent.identity.bundle_sha256
    assert candidate.parent_foundation_id == request.parent.identity.foundation_id
    assert candidate.parent_bundle_sha256 == request.parent.identity.bundle_sha256
    assert candidate.metrics["new_examples_optimized"] == 5
    assert candidate.metrics["monitoring_examples"] == 1
    assert candidate.metrics["monitoring_used_for_selection"] is False
    assert candidate.metrics["replay_examples"] == 2
    assert candidate.metrics["updates"] == 7
    assert candidate.metrics["optimizer_state_continuity"] is False
    assert candidate.metrics["optimizer_state_reset"] is True
    assert candidate.metrics["assistant_target_tokens"] > 0
    assert candidate.metrics["inactive_text_tokens_masked"] > 0
    assert (
        trained.manifest["training"]["parent_weights_sha256"]
        == (portable_model.manifest["weights_sha256"])
    )
    assert set(trained.root.iterdir()) == {
        trained.root / "model-config.json",
        trained.root / "model.safetensors",
        trained.root / "portable-model.json",
        trained.root / "tokenizer",
    }


def test_direct_torch_foundation_zero_config_fits_parent_context(
    portable_model,
) -> None:
    trainer, runtime = create_torch_personal_foundation_backend(
        portable_model,
        device="cpu",
    )

    assert trainer.config.context_length == portable_model.config.context_length
    assert runtime.config.context_length == portable_model.config.context_length
    with pytest.raises(ValueError, match="foundation context"):
        create_torch_personal_foundation_backend(
            portable_model,
            device="cpu",
            config=TorchFoundationConfig(
                context_length=portable_model.config.context_length + 1
            ),
        )


def test_direct_torch_foundation_activation_is_atomic_and_can_switch_back(
    tmp_path: Path,
    portable_model,
) -> None:
    config = TorchFoundationConfig(
        learning_rate=1e-2,
        context_length=32,
        disk_reserve_bytes=0,
    )
    trainer, runtime = create_torch_personal_foundation_backend(
        portable_model,
        device="cpu",
        config=config,
    )
    base_source = runtime.personal_foundation_source()
    request = _foundation_request(runtime, tmp_path / "activation-work", run_id=41)
    candidate = trainer.train_foundation(request)
    prepared = runtime.prepare_personal_foundation(
        _foundation_artifact(request, candidate)
    )

    runtime.activate_prepared_personal_foundation(prepared, candidate.identity)
    assert runtime.personal_foundation_source().identity == candidate.identity
    assert runtime.status()["foundation_frozen"] is True
    runtime.answer((("user", "hello"),), max_tokens=1)

    base_artifact = PersonalFoundationArtifact(
        directory=base_source.bundle_path,
        bundle_path=base_source.bundle_path,
        manifest_path=base_source.bundle_path / "base-foundation.json",
        manifest=PersonalFoundationManifest(
            identity=base_source.identity,
            parent_foundation_id=None,
            parent_bundle_sha256=None,
            run_id=0,
            new_lesson_ids=(),
            monitoring_lesson_ids=(),
            replay_lesson_ids=(),
            created_at="2026-08-28T00:00:00Z",
        ),
    )
    previous = runtime.prepare_personal_foundation(base_artifact)
    runtime.activate_prepared_personal_foundation(previous, base_source.identity)
    assert runtime.personal_foundation_source() == base_source


def test_direct_torch_foundation_honors_preflight_cancellation(
    tmp_path: Path,
    portable_model,
) -> None:
    trainer, runtime = create_torch_personal_foundation_backend(
        portable_model,
        device="cpu",
        config=TorchFoundationConfig(context_length=32, disk_reserve_bytes=0),
    )
    cancelled = threading.Event()
    cancelled.set()
    request = _foundation_request(
        runtime,
        tmp_path / "cancelled-work",
        run_id=42,
        cancelled=cancelled,
    )

    with pytest.raises(PersonalFoundationError, match="cancelled"):
        trainer.train_foundation(request)
    assert runtime.personal_foundation_source() == request.parent
