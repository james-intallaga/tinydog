from __future__ import annotations

import hashlib
from pathlib import Path
import threading

import pytest


mx = pytest.importorskip("mlx.core")

from safetensors import safe_open  # noqa: E402

from mini_arithmetic.floaty_training import encode_masked_verified_lesson  # noqa: E402

from mini_arithmetic.mlx_personal import (  # noqa: E402
    MLX_PERSONAL_ADAPTER_FORMAT,
    MLXPersonalAdapterRuntime,
    MLXPersonalAdapterTrainer,
    MLXFoundationConfig,
    MLXPersonalConfig,
    MLXPersonalError,
    _inactive_text_logit_bias,
    _dialogue_record,
    _masked_loss,
    _mlx_foundation_forward_smoke,
    _publish_mlx_foundation_bundle,
    _train_mlx_foundation,
    create_mlx_personal_foundation_backend,
)
from mini_arithmetic.personal_training import (  # noqa: E402
    DialogueExample,
    PersonalAdapterArtifact,
    PersonalAdapterManifest,
    PersonalAdapterTrainingRequest,
    encode_personal_dialogue_training_target,
)
from mini_arithmetic.personal_foundation import (  # noqa: E402
    PersonalFoundationArtifact,
    PersonalFoundationManifest,
    PersonalFoundationTrainingRequest,
    PersonalFoundationRuntime,
    PersonalFoundationTrainer,
)
from mini_arithmetic.portable_bundle import create_portable_bundle  # noqa: E402


def _write_micro_config(path: Path) -> None:
    path.write_text(
        """{
  "vocab_size": 32768,
  "context_length": 24,
  "model_dim": 8,
  "ffn_dim": 16,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "head_dim": 4,
  "conv_kernel_size": 3,
  "norm_eps": 0.00001,
  "rope_theta": 10000.0,
  "layer_schedule": ["conv", "attention"]
}
""",
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def portable_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlx-personal")
    config_path = root / "config.json"
    _write_micro_config(config_path)
    return create_portable_bundle(
        root / "bundle",
        config_path=config_path,
        backend="mlx",
        model_size="tiny",
        seed=19,
        parameter_dtype="float32",
    )


@pytest.fixture(scope="module")
def personal_config() -> MLXPersonalConfig:
    return MLXPersonalConfig(
        rank=2,
        alpha=4.0,
        learning_rate=1e-2,
        steps=2,
        grad_clip=1.0,
        seed=23,
        batch_size=1,
        context_length=12,
        max_tokens=3,
    )


def _example(identifier: int) -> DialogueExample:
    prompt = f"p{identifier}"
    response = f"r{identifier}"
    digest = hashlib.sha256(
        f"typed-dialogue-v1\0{prompt}\0{response}".encode()
    ).hexdigest()
    return DialogueExample(
        lesson_id=identifier,
        session_id="local",
        prompt=prompt,
        human_response=response,
        content_sha256=digest,
    )


def _request(
    root: Path,
    runtime: MLXPersonalAdapterRuntime,
    *,
    run_id: int,
    adapter_id: str,
    parent_id: str | None = None,
    parent_path: Path | None = None,
    replay_examples: tuple[DialogueExample, ...] = (),
    retention_examples: tuple[DialogueExample, ...] = (),
) -> PersonalAdapterTrainingRequest:
    identity = runtime.personal_runtime_identity()
    work = root / f"work-{run_id}"
    work.mkdir()
    return PersonalAdapterTrainingRequest(
        run_id=run_id,
        adapter_id=adapter_id,
        backend="mlx",
        foundation_sha256=identity.foundation_sha256,
        adapter_schema_sha256=identity.adapter_schema_sha256,
        parent_adapter_id=parent_id,
        parent_adapter_path=parent_path,
        training_examples=tuple(_example(index) for index in range(1, 5)),
        validation_examples=(_example(5),),
        work_dir=work,
        replay_examples=replay_examples,
        retention_examples=retention_examples,
    )


def _artifact(
    request: PersonalAdapterTrainingRequest,
    candidate,
) -> PersonalAdapterArtifact:
    manifest = PersonalAdapterManifest(
        adapter_id=request.adapter_id,
        backend="mlx",
        foundation_sha256=request.foundation_sha256,
        adapter_schema_sha256=request.adapter_schema_sha256,
        parent_adapter_id=request.parent_adapter_id,
        artifact_name=candidate.artifact_path.name,
        artifact_sha256=candidate.artifact_sha256,
        artifact_bytes=candidate.artifact_path.stat().st_size,
        training_lesson_ids=tuple(
            example.lesson_id for example in request.training_examples
        ),
        validation_lesson_ids=tuple(
            example.lesson_id for example in request.validation_examples
        ),
        run_id=request.run_id,
        baseline_holdout_loss=candidate.baseline_holdout_loss,
        candidate_holdout_loss=candidate.candidate_holdout_loss,
        created_at="2026-01-01T00:00:00+00:00",
    )
    manifest.validate()
    return PersonalAdapterArtifact(
        directory=request.work_dir,
        artifact_path=candidate.artifact_path,
        manifest_path=request.work_dir / "manifest.json",
        manifest=manifest,
    )


def test_mlx_trainer_exports_only_cumulative_floaty_tensors(
    portable_bundle,
    personal_config,
    tmp_path: Path,
) -> None:
    runtime = MLXPersonalAdapterRuntime.from_bundle(
        portable_bundle,
        config=personal_config,
    )
    trainer = MLXPersonalAdapterTrainer.from_bundle(
        portable_bundle,
        config=personal_config,
    )
    first_request = _request(
        tmp_path,
        runtime,
        run_id=1,
        adapter_id="personal-first",
    )
    first = trainer.train_adapter(first_request)

    assert first.foundation_sha256_before == first.foundation_sha256_after
    assert first.metrics["foundation_trainable"] is False
    assert first.metrics["assistant_target_tokens"] > 0
    with safe_open(str(first.artifact_path), framework="mlx") as tensors:
        assert tensors.metadata()["format"] == MLX_PERSONAL_ADAPTER_FORMAT
        assert tensors.metadata()["tensor_scope"] == "floaty-only"
        assert set(tensors.keys()) == set(runtime._model.exported_floaty_tensor_names())
        assert not any(name.startswith("foundation.") for name in tensors.keys())

    second_request = _request(
        tmp_path,
        runtime,
        run_id=2,
        adapter_id="personal-second",
        parent_id="personal-first",
        parent_path=first.artifact_path,
    )
    second = trainer.train_adapter(second_request)
    assert second.metrics["parent_adapter_loaded"] is True
    with safe_open(str(second.artifact_path), framework="mlx") as tensors:
        assert tensors.metadata()["parent_adapter_id"] == "personal-first"
        assert set(tensors.keys()) == set(runtime._model.exported_floaty_tensor_names())


def test_mlx_runtime_activates_rolls_back_and_generates_with_cache(
    portable_bundle,
    personal_config,
    tmp_path: Path,
) -> None:
    runtime = MLXPersonalAdapterRuntime.from_bundle(
        portable_bundle,
        config=personal_config,
    )
    trainer = MLXPersonalAdapterTrainer.from_bundle(
        portable_bundle,
        config=personal_config,
    )
    request = _request(
        tmp_path,
        runtime,
        run_id=3,
        adapter_id="personal-live",
    )
    candidate = trainer.train_adapter(request)
    prepared = runtime.prepare_personal_adapter(_artifact(request, candidate))
    runtime.activate_prepared_personal_adapter(prepared, "personal-live")

    assert runtime.personal_runtime_identity().active_adapter_id == "personal-live"
    status = runtime.status()
    assert status["cached_inference"] is True
    assert status["foundation_trainable"] is False
    assert status["adapter_trainable"] is False
    assert isinstance(runtime.answer([("user", "hello")], max_tokens=2), str)

    runtime.activate_foundation_only()
    assert runtime.personal_runtime_identity().active_adapter_id is None
    state = runtime._model.export_floaty_state()
    assert all(
        mx.array_equal(value, dict(runtime._foundation_only)[name]).item()
        for name, value in state.tensors.items()
    )


def test_mlx_trainer_replays_history_and_reports_retention_losses(
    portable_bundle,
    personal_config,
    tmp_path: Path,
) -> None:
    runtime = MLXPersonalAdapterRuntime.from_bundle(
        portable_bundle,
        config=personal_config,
    )
    trainer = MLXPersonalAdapterTrainer.from_bundle(
        portable_bundle,
        config=personal_config,
    )
    request = _request(
        tmp_path,
        runtime,
        run_id=31,
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
    assert candidate.metrics["retention_digest"] is not None
    assert candidate.metrics["foundation_trainable"] is False


def test_mlx_runtime_keeps_newest_user_suffix_and_rejects_oversized_target(
    portable_bundle,
    personal_config,
) -> None:
    runtime = MLXPersonalAdapterRuntime.from_bundle(
        portable_bundle,
        config=personal_config,
    )
    text = "🧪" * 1_000
    content_ids = portable_bundle.tokenizer.encode(text)
    budget = portable_bundle.config.context_length - 4
    assert len(content_ids) > budget

    prompt = runtime._prompt_ids((("user", text),))

    assert len(prompt) == portable_bundle.config.context_length
    assert prompt[0] == portable_bundle.tokenizer.bos_id
    assert prompt[1] == portable_bundle.tokenizer.role_id("user")
    assert prompt[2:-2] == content_ids[-budget:]
    assert prompt[-2] == portable_bundle.tokenizer.end_turn_id
    assert prompt[-1] == portable_bundle.tokenizer.role_id("assistant")
    assert runtime.personal_training_target_fits("hello", text) is False


def test_mlx_inactive_logits_cannot_consume_generation_or_ce_mass(
    portable_bundle,
    personal_config,
) -> None:
    runtime = MLXPersonalAdapterRuntime.from_bundle(
        portable_bundle,
        config=personal_config,
    )
    inactive = portable_bundle.tokenizer.inactive_text_token_ids
    visible_id = portable_bundle.tokenizer.encode("A")[0]
    blank_id = portable_bundle.tokenizer.encode(" ")[0]
    control_id = portable_bundle.tokenizer.encode("\x00")[0]
    assistant_id = portable_bundle.tokenizer.role_id("assistant")
    assert visible_id not in inactive

    class FixedLogits:
        def eval(self):
            return None

        def forward_cached(self, tokens, cache=None):
            values = mx.full(
                (*tokens.shape, portable_bundle.config.vocab_size),
                -20.0,
                dtype=mx.float32,
            )
            inactive_peak = mx.arange(portable_bundle.config.vocab_size) == inactive[0]
            unknown_peak = (
                mx.arange(portable_bundle.config.vocab_size)
                == portable_bundle.tokenizer.unk_id
            )
            assistant_peak = (
                mx.arange(portable_bundle.config.vocab_size) == assistant_id
            )
            control_peak = mx.arange(portable_bundle.config.vocab_size) == control_id
            eos_peak = (
                mx.arange(portable_bundle.config.vocab_size)
                == portable_bundle.tokenizer.eos_id
            )
            end_turn_peak = (
                mx.arange(portable_bundle.config.vocab_size)
                == portable_bundle.tokenizer.end_turn_id
            )
            blank_peak = mx.arange(portable_bundle.config.vocab_size) == blank_id
            visible_peak = mx.arange(portable_bundle.config.vocab_size) == visible_id
            values = (
                values
                + inactive_peak * 160.0
                + unknown_peak * 155.0
                + assistant_peak * 150.0
                + control_peak * 145.0
                + eos_peak * 140.0
                + end_turn_peak * 135.0
                + blank_peak * 130.0
                + visible_peak * (170.0 if cache is not None else 125.0)
            )
            return values, object()

    runtime._model = FixedLogits()
    generated = runtime.answer((("user", "hello"),), max_tokens=3)
    assert generated == portable_bundle.tokenizer.decode([visible_id] * 3).strip()
    assert generated
    assert "\x00" not in generated

    class FixedLossModel:
        def __call__(self, inputs):
            values = mx.zeros(
                (*inputs.shape, portable_bundle.config.vocab_size),
                dtype=mx.float32,
            )
            peak = mx.arange(portable_bundle.config.vocab_size) == inactive[0]
            return values + peak * 100.0

    inputs = mx.array([[portable_bundle.tokenizer.bos_id]])
    targets = mx.array([[visible_id]])
    loss_mask = mx.array([[1.0]])
    unmasked = _masked_loss(FixedLossModel(), inputs, targets, loss_mask)
    bias = _inactive_text_logit_bias(portable_bundle.config.vocab_size, inactive)
    masked = _masked_loss(FixedLossModel(), inputs, targets, loss_mask, bias)
    mx.eval(unmasked, masked)
    assert mx.isfinite(masked).item()
    assert masked.item() < unmasked.item()

    structural = _masked_loss(
        FixedLossModel(),
        inputs,
        mx.array([[portable_bundle.tokenizer.end_turn_id]]),
        loss_mask,
        bias,
    )
    mx.eval(structural)
    assert mx.isfinite(structural).item()


def test_torch_contract_tokens_and_mask_exactly_match_mlx_floaty_contract(
    portable_bundle,
    personal_config,
) -> None:
    example = _example(70)
    long_example = DialogueExample(
        lesson_id=example.lesson_id,
        session_id=example.session_id,
        prompt="old prompt words " * 20,
        human_response=example.human_response,
        content_sha256=hashlib.sha256(
            (
                "typed-dialogue-v1\0"
                + ("old prompt words " * 20).strip()
                + "\0"
                + example.human_response
            ).encode()
        ).hexdigest(),
    )
    tokens, mask = encode_personal_dialogue_training_target(
        portable_bundle.tokenizer,
        long_example.prompt,
        long_example.human_response,
        context_length=personal_config.context_length,
    )
    mlx_sequence = encode_masked_verified_lesson(
        _dialogue_record(long_example),
        portable_bundle.tokenizer,
        corrected=False,
        max_length=personal_config.context_length + 1,
    )

    assert tokens == list(mlx_sequence.token_ids)
    assert mask == [float(value) for value in mlx_sequence.loss_mask]
    assistant_role = portable_bundle.tokenizer.role_id("assistant")
    assistant_offset = tokens.index(assistant_role)
    assert mask[assistant_offset - 1] == 1.0


def test_bfloat16_foundation_only_path_remains_an_exact_noop(
    personal_config,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    _write_micro_config(config_path)
    bundle = create_portable_bundle(
        tmp_path / "bundle",
        config_path=config_path,
        backend="mlx",
        model_size="tiny",
        seed=29,
        parameter_dtype="bfloat16",
    )
    runtime = MLXPersonalAdapterRuntime.from_bundle(bundle, config=personal_config)
    tokens = mx.array([[1, 5, 12, 2]], dtype=mx.int32)

    foundation = runtime._model.foundation(tokens)
    composed = runtime._model(tokens)
    mx.eval(foundation, composed)

    assert foundation.dtype == mx.bfloat16
    assert composed.dtype == mx.bfloat16
    assert mx.array_equal(foundation, composed).item()

    trainer = MLXPersonalAdapterTrainer.from_bundle(bundle, config=personal_config)
    request = _request(
        tmp_path,
        runtime,
        run_id=4,
        adapter_id="personal-bfloat",
    )
    candidate = trainer.train_adapter(request)
    state = mx.load(str(candidate.artifact_path))
    assert state and all(value.dtype == mx.bfloat16 for value in state.values())
    prepared = runtime.prepare_personal_adapter(_artifact(request, candidate))
    runtime.activate_prepared_personal_adapter(prepared, "personal-bfloat")
    assert runtime.personal_runtime_identity().active_adapter_id == "personal-bfloat"

    direct = _train_mlx_foundation(
        bundle,
        run_id=5,
        new_examples=tuple(_example(index) for index in range(81, 86)),
        monitoring_examples=(_example(85),),
        config=MLXFoundationConfig(
            learning_rate=1e-3,
            context_length=12,
        ),
    )
    trained_bundle = _publish_mlx_foundation_bundle(
        bundle,
        direct,
        tmp_path / "bfloat-direct-foundation",
    )
    assert trained_bundle.parameter_dtype == "bfloat16"
    trained_state = mx.load(str(trained_bundle.weights_path))
    assert {value.dtype for value in trained_state.values()} == {mx.bfloat16}


def test_direct_mlx_foundation_training_changes_canonical_bundle_and_only_reports_gates(
    portable_bundle,
    tmp_path: Path,
) -> None:
    parent_bytes = portable_bundle.weights_path.read_bytes()
    config = MLXFoundationConfig(
        learning_rate=1e-3,
        steps=5,
        grad_clip=1.0,
        seed=41,
        batch_size=1,
        context_length=12,
    )
    result = _train_mlx_foundation(
        portable_bundle,
        run_id=51,
        new_examples=tuple(_example(index) for index in range(101, 106)),
        monitoring_examples=(_example(105),),
        replay_examples=(_example(106), _example(107)),
        config=config,
    )

    assert result.metrics["foundation_trainable"] is True
    assert result.metrics["full_foundation"] is True
    assert result.metrics["updates"] == 7
    assert result.metrics["new_training_examples"] == 5
    assert result.metrics["new_monitoring_examples"] == 1
    assert result.metrics["new_examples_optimized"] == 5
    assert result.metrics["replay_examples"] == 2
    assert result.metrics["assistant_target_tokens"] > 0
    assert result.metrics["inactive_text_tokens_masked"] > 0
    assert result.metrics["monitor_used_for_selection"] is False
    assert result.metrics["optimizer_state_continuity"] is False
    assert result.metrics["optimizer_state_reset"] is True
    assert result.baseline_monitoring_loss >= 0.0
    assert result.candidate_monitoring_loss >= 0.0

    candidate = _publish_mlx_foundation_bundle(
        portable_bundle,
        result,
        tmp_path / "direct-foundation-candidate",
    )
    assert portable_bundle.weights_path.read_bytes() == parent_bytes
    assert candidate.manifest["initialization"].endswith("foundation-trained-v1")
    assert candidate.manifest["created_by_backend"] == "mlx"
    assert (
        candidate.manifest["training"]["parent_weights_sha256"]
        == (portable_bundle.manifest["weights_sha256"])
    )
    assert (
        candidate.manifest["weights_sha256"]
        != (portable_bundle.manifest["weights_sha256"])
    )
    assert candidate.manifest["training"]["global_step"] == 7
    assert candidate.manifest["training"]["total_steps"] == 7

    with safe_open(str(candidate.weights_path), framework="mlx") as trained:
        with safe_open(str(portable_bundle.weights_path), framework="mlx") as parent:
            names = tuple(trained.keys())
            assert set(names) == set(parent.keys())
            assert any(
                not mx.array_equal(
                    trained.get_tensor(name), parent.get_tensor(name)
                ).item()
                for name in names
            )


def test_direct_mlx_foundation_training_is_bounded_and_cancellable(
    portable_bundle,
) -> None:
    config = MLXFoundationConfig(context_length=12)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(MLXPersonalError, match="cancelled"):
        _train_mlx_foundation(
            portable_bundle,
            run_id=52,
            new_examples=tuple(_example(index) for index in range(111, 116)),
            monitoring_examples=(_example(115),),
            config=config,
            cancellation_event=cancelled,
        )

    with pytest.raises(MLXPersonalError, match="replay exceeds"):
        _train_mlx_foundation(
            portable_bundle,
            run_id=53,
            new_examples=tuple(_example(index) for index in range(121, 126)),
            monitoring_examples=(_example(125),),
            replay_examples=tuple(_example(index) for index in range(130, 163)),
            config=config,
        )


def _foundation_artifact(
    bundle_path: Path,
    identity,
    *,
    parent_id: str | None,
    parent_bundle_sha256: str | None,
    run_id: int,
) -> PersonalFoundationArtifact:
    manifest = PersonalFoundationManifest(
        identity=identity,
        parent_foundation_id=parent_id,
        parent_bundle_sha256=parent_bundle_sha256,
        run_id=run_id,
        new_lesson_ids=(201, 202, 203, 204, 205) if run_id else (),
        monitoring_lesson_ids=(205,) if run_id else (),
        replay_lesson_ids=(),
        created_at="2026-01-01T00:00:00+00:00",
    )
    return PersonalFoundationArtifact(
        directory=bundle_path.parent,
        bundle_path=bundle_path,
        manifest_path=bundle_path.parent / "manifest.json",
        manifest=manifest,
    )


def test_public_mlx_foundation_backend_trains_and_atomically_switches_back(
    portable_bundle,
    tmp_path: Path,
    monkeypatch,
) -> None:
    foundation_config = MLXFoundationConfig(
        learning_rate=1e-3,
        context_length=12,
    )
    trainer, runtime = create_mlx_personal_foundation_backend(
        portable_bundle,
        foundation_id="foundation-original",
        config=foundation_config,
    )
    assert isinstance(trainer, PersonalFoundationTrainer)
    assert isinstance(runtime, PersonalFoundationRuntime)
    parent = runtime.personal_foundation_source()
    initial_status = runtime.status()
    assert initial_status["active_foundation_id"] == "foundation-original"
    assert initial_status["bundle_sha256"] == parent.identity.bundle_sha256
    assert initial_status["adapter_parameters"] == 0
    assert initial_status["active_adapter_parameters"] == 0
    assert "adapter_schema_sha256" not in initial_status
    assert initial_status["all_foundation_parameters_optimizer_eligible"] is True
    assert initial_status["optimizer_state_continuity"] is False
    assert initial_status["foundation_trainable"] is False
    assert initial_status["foundation_frozen"] is True
    with pytest.raises(MLXPersonalError, match="legacy adapters"):
        runtime.prepare_personal_adapter(object())  # type: ignore[arg-type]
    work = tmp_path / "public-work"
    work.mkdir()
    new_examples = tuple(_example(index) for index in range(201, 206))
    request = PersonalFoundationTrainingRequest(
        run_id=61,
        candidate_foundation_id="foundation-candidate",
        parent=parent,
        new_examples=new_examples,
        monitoring_examples=(new_examples[-1],),
        replay_examples=(),
        work_dir=work,
    )
    candidate = trainer.train_foundation(request)
    assert candidate.parent_foundation_id == "foundation-original"
    assert candidate.parent_bundle_sha256 == parent.identity.bundle_sha256
    assert candidate.identity.weights_sha256 != parent.identity.weights_sha256
    assert candidate.metrics["new_examples_optimized"] == 5
    assert candidate.metrics["quality_gate_applied"] is False
    assert candidate.metrics["optimizer_state_continuity"] is False

    candidate_artifact = _foundation_artifact(
        candidate.bundle_path,
        candidate.identity,
        parent_id=candidate.parent_foundation_id,
        parent_bundle_sha256=candidate.parent_bundle_sha256,
        run_id=61,
    )
    smoke_calls: list[str] = []

    def counted_smoke(loaded):
        smoke_calls.append(str(loaded.bundle.root))
        _mlx_foundation_forward_smoke(loaded)

    monkeypatch.setattr(
        "mini_arithmetic.mlx_personal._mlx_foundation_forward_smoke",
        counted_smoke,
    )
    prepared_candidate = runtime.prepare_personal_foundation(candidate_artifact)
    assert smoke_calls == [str(candidate.bundle_path)]
    runtime.activate_prepared_personal_foundation(
        prepared_candidate,
        candidate.identity,
    )
    assert runtime.personal_foundation_source().identity == candidate.identity
    assert runtime.personal_runtime_identity().foundation_sha256 == (
        candidate.identity.weights_sha256
    )

    base_artifact = _foundation_artifact(
        parent.bundle_path,
        parent.identity,
        parent_id=None,
        parent_bundle_sha256=None,
        run_id=0,
    )
    prepared_base = runtime.prepare_personal_foundation(base_artifact)
    runtime.activate_prepared_personal_foundation(prepared_base, parent.identity)
    assert runtime.personal_foundation_source() == parent

    prepared_candidate = runtime.prepare_personal_foundation(candidate_artifact)
    before_failed_swap = runtime.personal_foundation_source()

    def fail_bias(*args, **kwargs):
        raise RuntimeError("synthetic activation failure")

    monkeypatch.setattr(
        "mini_arithmetic.mlx_personal._inactive_text_logit_bias",
        fail_bias,
    )
    with pytest.raises(RuntimeError, match="synthetic activation failure"):
        runtime.activate_prepared_personal_foundation(
            prepared_candidate,
            candidate.identity,
        )
    assert runtime.personal_foundation_source() == before_failed_swap


def test_public_mlx_foundation_zero_config_fits_parent_context(
    portable_bundle,
) -> None:
    trainer, runtime = create_mlx_personal_foundation_backend(portable_bundle)

    assert trainer.config.context_length == portable_bundle.config.context_length
    assert runtime.config.context_length == portable_bundle.config.context_length
    assert runtime.config.rank < portable_bundle.config.model_dim
    with pytest.raises(ValueError, match="foundation context"):
        create_mlx_personal_foundation_backend(
            portable_bundle,
            config=MLXFoundationConfig(
                context_length=portable_bundle.config.context_length + 1
            ),
        )


def test_public_mlx_foundation_preflights_disk_without_mutating_runtime(
    portable_bundle,
    tmp_path: Path,
) -> None:
    trainer, runtime = create_mlx_personal_foundation_backend(
        portable_bundle,
        config=MLXFoundationConfig(
            context_length=12,
            disk_reserve_bytes=10**18,
        ),
    )
    parent = runtime.personal_foundation_source()
    work = tmp_path / "disk-preflight"
    work.mkdir()
    new_examples = tuple(_example(index) for index in range(401, 406))
    request = PersonalFoundationTrainingRequest(
        run_id=62,
        candidate_foundation_id="foundation-no-disk",
        parent=parent,
        new_examples=new_examples,
        monitoring_examples=(new_examples[-1],),
        replay_examples=(),
        work_dir=work,
    )

    with pytest.raises(MLXPersonalError, match="needs at least"):
        trainer.train_foundation(request)
    assert runtime.personal_foundation_source() == parent


def test_public_mlx_foundation_accepts_shared_nine_example_replay_bound(
    portable_bundle,
    tmp_path: Path,
) -> None:
    trainer, runtime = create_mlx_personal_foundation_backend(
        portable_bundle,
        config=MLXFoundationConfig(
            context_length=12,
            disk_reserve_bytes=0,
        ),
    )
    work = tmp_path / "nine-replay"
    work.mkdir()
    new_examples = tuple(_example(index) for index in range(601, 606))
    request = PersonalFoundationTrainingRequest(
        run_id=64,
        candidate_foundation_id="foundation-nine-replay",
        parent=runtime.personal_foundation_source(),
        new_examples=new_examples,
        monitoring_examples=(new_examples[-1],),
        replay_examples=tuple(_example(index) for index in range(701, 710)),
        work_dir=work,
    )

    candidate = trainer.train_foundation(request)
    assert candidate.metrics["new_examples_optimized"] == 5
    assert candidate.metrics["replay_examples"] == 9
    assert candidate.metrics["updates"] == 14


def test_public_mlx_foundation_reports_oom_without_mutating_runtime(
    portable_bundle,
    tmp_path: Path,
    monkeypatch,
) -> None:
    trainer, runtime = create_mlx_personal_foundation_backend(
        portable_bundle,
        config=MLXFoundationConfig(
            context_length=12,
            disk_reserve_bytes=0,
        ),
    )
    parent = runtime.personal_foundation_source()
    work = tmp_path / "oom-preflight"
    work.mkdir()
    new_examples = tuple(_example(index) for index in range(501, 506))
    request = PersonalFoundationTrainingRequest(
        run_id=63,
        candidate_foundation_id="foundation-oom",
        parent=parent,
        new_examples=new_examples,
        monitoring_examples=(new_examples[-1],),
        replay_examples=(),
        work_dir=work,
    )

    def fail_training(*args, **kwargs):
        raise RuntimeError("Metal out of memory")

    monkeypatch.setattr(
        "mini_arithmetic.mlx_personal._train_mlx_foundation",
        fail_training,
    )
    with pytest.raises(MLXPersonalError, match="active parent remains unchanged"):
        trainer.train_foundation(request)
    assert runtime.personal_foundation_source() == parent
