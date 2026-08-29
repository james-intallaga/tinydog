import json

import mlx.core as mx
import numpy as np
import pytest

from mini_arithmetic.continual import (
    ContinualTrainingConfig,
    _practice_records,
    encode_corrected_lesson,
    encode_verified_lesson,
    make_packed_lesson_batch,
    render_verified_lesson,
    run_candidate,
)
from mini_arithmetic.model import ArithmeticLM, ModelConfig
from mini_arithmetic.tokenizer import ArithmeticTokenizer


def lesson(identifier: int, a: int, b: int) -> dict:
    return {
        "id": identifier,
        "problem": f"What is {a} + {b}?",
        "first_attempt": str(a + b - 1),
        "correct_answer": str(a + b),
        "feedback": "Add both numbers and check the result.",
        "skill": "addition",
        "verifier": "integer calculator",
        "status": "pending",
    }


def test_render_and_pack_verified_lessons() -> None:
    text = render_verified_lesson(lesson(1, 7, 8))
    assert text.startswith("Teacher: What is 7 + 8?\nStudent: 15\n")
    assert "earlier attempt '14' was incorrect" in text
    tokenizer = ArithmeticTokenizer()
    encoded = encode_verified_lesson(lesson(1, 7, 8), tokenizer)
    assert encoded[0] == tokenizer.BOS
    assert encoded.index(tokenizer.USER) < encoded.index(tokenizer.ASSISTANT)
    assert tokenizer.TEACHER not in encoded
    corrected = encode_corrected_lesson(lesson(1, 7, 8), tokenizer)
    assert corrected.index(tokenizer.USER) < corrected.index(tokenizer.TEACHER)
    assert corrected.index(tokenizer.TEACHER) < corrected.index(tokenizer.ASSISTANT)
    first = make_packed_lesson_batch([text], tokenizer, 2, 24, seed=3)
    second = make_packed_lesson_batch([text], tokenizer, 2, 24, seed=3)
    assert first.shape == (2, 25)
    assert first.dtype == np.int32
    np.testing.assert_array_equal(first, second)


def test_rejects_unverified_lesson() -> None:
    record = lesson(1, 1, 2)
    record["verifier"] = ""
    with pytest.raises(ValueError, match="verifier"):
        render_verified_lesson(record)


def test_generated_held_out_practice_is_not_admitted_to_training() -> None:
    record = lesson(1, 21, 38)
    record["practice_variants"] = [
        {
            "problem": "What is 25 * 30?",
            "correct_answer": "750",
            "metadata": {"generated": True, "canonical_key": "mul:25:30"},
        },
        {
            "problem": "What is 24 * 47?",
            "correct_answer": "1128",
            "metadata": {"generated": True, "canonical_key": "mul:24:47"},
        },
        {
            "problem": "What is 25 * 30?",
            "correct_answer": "750",
            "metadata": {"origin": "human"},
        },
    ]

    admitted = _practice_records(record)
    assert [item["problem"] for item in admitted] == [
        "What is 24 * 47?",
        "What is 25 * 30?",
    ]


def test_run_candidate_preserves_active_and_writes_gate_report(tmp_path) -> None:
    tokenizer = ArithmeticTokenizer()
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=24,
        num_layers=1,
        model_dim=16,
        num_heads=4,
        ffn_dim=32,
    )
    mx.random.seed(9)
    active_model = ArithmeticLM(model_config)
    mx.eval(active_model.parameters())
    active_path = tmp_path / "active.safetensors"
    active_model.save_weights(str(active_path))
    active_before = active_path.read_bytes()

    config = ContinualTrainingConfig(
        steps=2,
        batch_size=2,
        learning_rate=1e-3,
        replay_ratio=0.5,
        evaluation_batches=1,
        max_old_loss_regression=100.0,
        seed=7,
    )
    records = [lesson(1, 2, 3), lesson(2, 4, 5), lesson(3, 8, 7)]
    result = run_candidate(
        active_path,
        model_config,
        records,
        tmp_path / "candidates",
        tokenizer=tokenizer,
        validation_lessons=[lesson(99, 6, 8)],
        config=config,
    )

    assert active_path.read_bytes() == active_before
    assert result.weights_path.is_file()
    assert result.weights_path != active_path
    assert result.report_path.is_file()
    assert result.trained_lesson_ids == (1, 2, 3)
    report = json.loads(result.report_path.read_text())
    assert report["candidate_weights"] == str(result.weights_path)
    assert report["lessons"]["trained_ids"] == [1, 2, 3]
    assert report["promotion_eligible"] == (
        report["gate"]["improves_new"] and report["gate"]["preserves_old"]
    )
