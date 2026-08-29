import hashlib
import json

import mlx.core as mx
import pytest

from mini_arithmetic.continual import (
    ContinualTrainingConfig,
    _split_lessons,
    run_candidate,
)
from mini_arithmetic.coordinator import resolve_active_checkpoint
from mini_arithmetic.model import ArithmeticLM, ModelConfig
from mini_arithmetic.runtime import ModelRuntime
from mini_arithmetic.tokenizer import ArithmeticTokenizer


def _lesson(identifier: int, a: int, b: int, *, practice: bool = False) -> dict:
    record = {
        "id": identifier,
        "problem": f"What is {a} + {b}?",
        "first_attempt": str(a + b - 1),
        "correct_answer": str(a + b),
        "feedback": "Add carefully, then check the result.",
        "skill": "addition",
        "verifier": "exact-arithmetic-v1",
        "status": "pending",
    }
    if practice:
        record["practice_variants"] = [
            {"problem": f"What is {a + 1} + {b + 1}?", "correct_answer": str(a + b + 2)}
        ]
    return record


def _tiny_checkpoint(tmp_path, name: str = "active"):
    tokenizer = ArithmeticTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=24,
        num_layers=1,
        model_dim=16,
        num_heads=4,
        ffn_dim=32,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    mx.random.seed(17)
    model = ArithmeticLM(config)
    mx.eval(model.parameters())
    weights_path = tmp_path / f"{name}.safetensors"
    model.save_weights(str(weights_path))
    return config, config_path, weights_path


def test_failed_runtime_reload_does_not_mutate_active_model(tmp_path) -> None:
    _, config_path, weights_path = _tiny_checkpoint(tmp_path)
    runtime = ModelRuntime(config_path, weights_path)
    active_model = runtime.model
    active_path = runtime.weights_path

    with pytest.raises(RuntimeError):
        runtime.reload(tmp_path / "missing.safetensors")

    assert runtime.model is active_model
    assert runtime.weights_path == active_path


def test_active_pointer_requires_unchanged_config_hash_and_tensor_signature(
    tmp_path,
) -> None:
    config, config_path, initial_path = _tiny_checkpoint(tmp_path, "initial")
    mx.random.seed(18)
    candidate = ArithmeticLM(config)
    mx.eval(candidate.parameters())
    candidate_path = tmp_path / "candidate.safetensors"
    candidate.save_weights(str(candidate_path))
    pointer = tmp_path / "active.json"
    pointer.write_text(
        json.dumps(
            {
                "checkpoint": str(candidate_path),
                "config": str(config_path),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "model_signature": config.to_dict(),
            }
        )
    )
    assert (
        resolve_active_checkpoint(initial_path, pointer, config_path)
        == candidate_path.resolve()
    )

    changed = {**config.to_dict(), "norm_eps": 1e-4}
    config_path.write_text(json.dumps(changed))
    assert (
        resolve_active_checkpoint(initial_path, pointer, config_path)
        == initial_path.resolve()
    )


def test_single_or_duplicate_lesson_has_no_independent_validation(tmp_path) -> None:
    one = _lesson(1, 2, 3)
    training, validation, overlap = _split_lessons([one, dict(one, id=2)], 0.2)
    assert overlap
    assert len(training) == len(validation) == 2

    config, _, active_path = _tiny_checkpoint(tmp_path)
    result = run_candidate(
        active_path,
        config,
        [one],
        tmp_path / "candidates",
        config=ContinualTrainingConfig(
            steps=1,
            batch_size=1,
            replay_ratio=0.0,
            evaluation_batches=1,
            max_old_loss_regression=100.0,
            context_length=24,
        ),
    )
    assert not result.promotion_eligible
    assert not result.report["gate"]["independent_validation"]


def test_training_budget_covers_primary_and_practice_sequences(tmp_path) -> None:
    config, _, active_path = _tiny_checkpoint(tmp_path)
    records = [_lesson(1, 2, 3, practice=True), _lesson(2, 4, 5, practice=True)]
    result = run_candidate(
        active_path,
        config,
        records,
        tmp_path / "candidates",
        validation_lessons=[_lesson(99, 8, 9)],
        config=ContinualTrainingConfig(
            steps=1,
            batch_size=1,
            replay_ratio=0.5,
            evaluation_batches=1,
            max_old_loss_regression=100.0,
            context_length=24,
        ),
    )
    rows = result.report["training_rows"]
    assert rows["effective_steps"] > rows["requested_steps"]
    assert rows["all_primary_lessons_covered"]
    assert rows["practice_rows_sampled"] > 0
    assert result.trained_lesson_ids == (1, 2)
