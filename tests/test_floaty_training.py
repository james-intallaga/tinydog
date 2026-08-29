import json

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

import mini_arithmetic.floaty_training as floaty_training
from mini_arithmetic.floaty_training import (
    FloatyBoundaryError,
    FloatyTrainingConfig,
    FoundationFloatyTrainingBoundary,
    encode_masked_verified_lesson,
    make_verified_lesson_batches,
    run_floaty_candidate,
)
from mini_arithmetic.foundation_floaty import FloatyConfig, FoundationFloatyLM
from mini_arithmetic.hybrid import CONV, HybridConfig, HybridLM
from mini_arithmetic.model import ArithmeticLM, ModelConfig
from mini_arithmetic.scratch_tokenizer import ScratchTokenizer, build_scratch_tokenizer


class FixedConversationTokenizer:
    """Keep tests focused on the update boundary, not tokenizer quality."""

    BOS = 0
    EOS = 1
    PAD = 2
    _ROLES = {"user": 3, "assistant": 4, "teacher": 5}

    def encode(self, _text: str, **_kwargs: object) -> list[int]:
        return [6]

    def role_id(self, role: str) -> int:
        return self._ROLES[role]

    def encode_dialogue(
        self,
        turns: list[tuple[str, str]],
        *,
        bos: bool,
        eos: bool,
    ) -> list[int]:
        tokens = [self.BOS] if bos else []
        for role, content in turns:
            tokens.extend([self.role_id(role), *self.encode(content)])
        if eos:
            tokens.append(self.EOS)
        return tokens


def lesson(identifier: int, problem: str, answer: str) -> dict[str, object]:
    return {
        "id": identifier,
        "problem": problem,
        "first_attempt": "wrong",
        "correct_answer": answer,
        "feedback": "Use the verified answer.",
        "skill": "conversation",
        "verifier": "local-test-verifier",
    }


def tiny_model(seed: int = 7) -> FoundationFloatyLM:
    config = HybridConfig(
        vocab_size=7,
        context_length=7,
        model_dim=4,
        ffn_dim=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        conv_kernel_size=2,
        layer_schedule=(CONV,),
    )
    mx.random.seed(seed)
    return FoundationFloatyLM(
        HybridLM(config),
        FloatyConfig(rank=1, alpha=1.0),
    )


def training_config(**overrides: object) -> FloatyTrainingConfig:
    values: dict[str, object] = {
        "steps": 12,
        "batch_size": 2,
        "context_length": 7,
        "learning_rate": 5e-2,
        "weight_decay": 0.0,
        "minimum_training_lessons": 2,
        "minimum_validation_lessons": 1,
        "seed": 11,
    }
    values.update(overrides)
    return FloatyTrainingConfig(**values)


def array_snapshot(module: object) -> dict[str, np.ndarray]:
    values = {
        name: np.asarray(value).copy()
        for name, value in tree_flatten(module.parameters())
    }
    return values


@pytest.fixture(scope="module")
def scratch_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> ScratchTokenizer:
    return build_scratch_tokenizer(
        [
            "User asks a problem and the assistant gives a verified answer.",
            "Teacher feedback explains a correction before the assistant retries.",
            "What is 19 + 23? The exact corrected answer is 42.",
        ],
        tmp_path_factory.mktemp("floaty-scratch-tokenizer"),
    )


def test_boundary_rejects_live_arithmetic_model_without_floaty_partition() -> None:
    model = ArithmeticLM(
        ModelConfig(
            vocab_size=4,
            context_length=3,
            num_layers=1,
            model_dim=4,
            num_heads=1,
            ffn_dim=8,
        )
    )
    with pytest.raises(FloatyBoundaryError, match="separate foundation and floaty"):
        FoundationFloatyTrainingBoundary(
            model,
            expected_foundation_hash="0" * 64,
            expected_config_hash="1" * 64,
        )


def test_verified_lesson_batcher_is_deterministic_and_rejects_unverified() -> None:
    tokenizer = FixedConversationTokenizer()
    records = [lesson(1, "Say alpha", "alpha"), lesson(2, "Say beta", "beta")]
    first, first_digest, first_ids = make_verified_lesson_batches(
        records,
        tokenizer,
        batch_size=2,
        context_length=7,
        steps=2,
        seed=4,
        allow_test_tokenizer_fallback=True,
    )
    second, second_digest, second_ids = make_verified_lesson_batches(
        records,
        tokenizer,
        batch_size=2,
        context_length=7,
        steps=2,
        seed=4,
        allow_test_tokenizer_fallback=True,
    )
    assert first_digest == second_digest
    assert first_ids == second_ids == [1, 2]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.tokens, right.tokens)
        np.testing.assert_array_equal(left.loss_mask, right.loss_mask)

    rejected = lesson(3, "Say gamma", "gamma")
    rejected["verifier"] = ""
    with pytest.raises(ValueError, match="verifier"):
        make_verified_lesson_batches(
            [rejected],
            tokenizer,
            batch_size=1,
            context_length=7,
            steps=1,
            seed=4,
            allow_test_tokenizer_fallback=True,
        )


def test_trusted_human_dialogue_uses_human_reply_as_one_direct_target(
    monkeypatch,
) -> None:
    record = {
        "id": 4,
        "problem": "How has your morning been?",
        "first_attempt": "",
        "correct_answer": "Quiet, but good. I made coffee.",
        "feedback": "",
        "skill": "conversation",
        "source": "human-conversation",
        "verifier": "trusted-human-dialogue-v1",
        "metadata": {
            "human_authored_target": True,
            "consent_local_training": True,
        },
    }
    corrected_flags: list[bool] = []
    original = floaty_training.encode_masked_verified_lesson

    def capture(*args: object, **kwargs: object):
        corrected_flags.append(bool(kwargs["corrected"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(floaty_training, "encode_masked_verified_lesson", capture)
    batches, _, identifiers = make_verified_lesson_batches(
        [record],
        FixedConversationTokenizer(),
        batch_size=1,
        context_length=7,
        steps=1,
        seed=4,
        allow_test_tokenizer_fallback=True,
    )

    assert corrected_flags == [False]
    assert identifiers == [4]
    assert batches[0].loss_mask.any()


def test_scratch_boundaries_mask_user_and_teacher_targets(
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    record = lesson(20, "What is 19 + 23?", "42")
    direct = encode_masked_verified_lesson(
        record,
        scratch_tokenizer,
        corrected=False,
        max_length=256,
    )
    direct_targets = np.asarray(direct.token_ids[1:])[np.asarray(direct.loss_mask)]
    np.testing.assert_array_equal(
        direct_targets,
        [
            scratch_tokenizer.role_id("assistant"),
            *scratch_tokenizer.encode("42"),
            scratch_tokenizer.end_turn_id,
            scratch_tokenizer.eos_id,
        ],
    )
    user = next(boundary for boundary in direct.boundaries if boundary.role == "user")
    assert not any(
        direct.loss_mask[target_offset - 1]
        for target_offset in range(user.content_offset, user.end_turn_offset)
    )

    corrected = encode_masked_verified_lesson(
        record,
        scratch_tokenizer,
        corrected=True,
        max_length=256,
    )
    roles = [boundary.role for boundary in corrected.boundaries]
    assert roles == ["user", "teacher", "assistant"]
    teacher = corrected.boundaries[1]
    assistant = corrected.boundaries[2]
    assert teacher.content_length > 0
    assert not any(
        corrected.loss_mask[target_offset - 1]
        for target_offset in range(teacher.content_offset, teacher.end_turn_offset)
    )
    assert corrected.token_ids[assistant.role_offset] == scratch_tokenizer.role_id(
        "assistant"
    )
    assert corrected.loss_mask[assistant.role_offset - 1] is True
    corrected_targets = np.asarray(corrected.token_ids[1:])[
        np.asarray(corrected.loss_mask)
    ]
    np.testing.assert_array_equal(
        corrected_targets,
        [
            scratch_tokenizer.role_id("assistant"),
            *scratch_tokenizer.encode("42"),
            scratch_tokenizer.end_turn_id,
            scratch_tokenizer.eos_id,
        ],
    )


def test_test_tokenizer_fallback_is_explicit() -> None:
    record = lesson(30, "Say alpha", "alpha")
    tokenizer = FixedConversationTokenizer()
    with pytest.raises(TypeError, match="test-only"):
        encode_masked_verified_lesson(
            record,
            tokenizer,
            corrected=True,
            max_length=8,
        )
    encoded = encode_masked_verified_lesson(
        record,
        tokenizer,
        corrected=True,
        max_length=8,
        allow_test_tokenizer_fallback=True,
    )
    assert sum(encoded.loss_mask) == 3


def test_accepted_candidate_exports_only_floaty_and_never_touches_foundation(
    tmp_path,
) -> None:
    model = tiny_model()
    foundation_before = array_snapshot(model.foundation)
    foundation_hash = model.foundation_hash()
    config_hash = model.config_hash()
    result = run_floaty_candidate(
        model,
        [lesson(1, "Say alpha", "alpha"), lesson(2, "Say beta", "beta")],
        [lesson(10, "Say gamma", "gamma")],
        FixedConversationTokenizer(),
        tmp_path,
        expected_foundation_hash=foundation_hash,
        expected_config_hash=config_hash,
        expected_floaty_parameter_names=model.floaty_parameter_names(),
        allow_test_tokenizer_fallback=True,
        config=training_config(),
    )

    assert result.accepted is True
    assert result.rolled_back is False
    assert result.floaty_weights_path is not None
    exported = mx.load(str(result.floaty_weights_path))
    assert set(exported) == set(model.exported_floaty_tensor_names())
    assert all(not name.startswith("foundation.") for name in exported)
    for name, before in foundation_before.items():
        np.testing.assert_array_equal(
            np.asarray(dict(tree_flatten(model.foundation.parameters()))[name]), before
        )
    assert model.foundation_hash() == foundation_hash

    provenance = json.loads(result.provenance_path.read_text())
    report = json.loads(result.report_path.read_text())
    assert provenance["foundation_trainable"] is False
    assert provenance["gradient_scope"] == "floaty-only"
    assert provenance["optimizer_scope"] == "floaty-only"
    assert set(provenance["floaty_parameter_names"]) == set(
        model.floaty_parameter_names()
    )
    assert report["boundary"]["foundation_gradient_parameters"] == []
    assert report["boundary"]["foundation_optimizer_parameters"] == []
    assert set(report["boundary"]["gradient_parameter_names"]) == set(
        model.exported_floaty_tensor_names()
    )
    assert all(
        not name.startswith("foundation.")
        for name in report["boundary"]["optimizer_state_parameter_names"]
    )
    serialized = result.provenance_path.read_text() + result.report_path.read_text()
    assert "Say alpha" not in serialized
    assert "Use the verified answer" not in serialized


def test_rejected_candidate_rolls_floaty_back_and_emits_no_weights(tmp_path) -> None:
    model = tiny_model()
    floaty_before = array_snapshot(model.floaty)
    foundation_before = array_snapshot(model.foundation)
    result = run_floaty_candidate(
        model,
        [lesson(1, "Say alpha", "alpha"), lesson(2, "Say beta", "beta")],
        [lesson(10, "Say gamma", "gamma")],
        FixedConversationTokenizer(),
        tmp_path,
        expected_foundation_hash=model.foundation_hash(),
        expected_config_hash=model.config_hash(),
        allow_test_tokenizer_fallback=True,
        config=training_config(minimum_validation_improvement=1_000_000.0),
    )

    assert result.accepted is False
    assert result.rolled_back is True
    assert result.floaty_weights_path is None
    current_floaty = dict(tree_flatten(model.floaty.parameters()))
    current_foundation = dict(tree_flatten(model.foundation.parameters()))
    for name, before in floaty_before.items():
        np.testing.assert_array_equal(np.asarray(current_floaty[name]), before)
    for name, before in foundation_before.items():
        np.testing.assert_array_equal(np.asarray(current_foundation[name]), before)
    assert json.loads(result.report_path.read_text())["rolled_back"] is True


def test_candidate_rejects_hash_mismatch_before_training(tmp_path) -> None:
    model = tiny_model()
    with pytest.raises(FloatyBoundaryError, match="pinned active foundation"):
        run_floaty_candidate(
            model,
            [lesson(1, "Say alpha", "alpha"), lesson(2, "Say beta", "beta")],
            [lesson(10, "Say gamma", "gamma")],
            FixedConversationTokenizer(),
            tmp_path,
            expected_foundation_hash="0" * 64,
            expected_config_hash=model.config_hash(),
            allow_test_tokenizer_fallback=True,
            config=training_config(),
        )
    assert list(tmp_path.iterdir()) == []


def test_artifact_failure_rolls_candidate_back(tmp_path, monkeypatch) -> None:
    model = tiny_model()
    floaty_before = array_snapshot(model.floaty)

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(floaty_training.mx, "save_safetensors", fail_save)
    with pytest.raises(OSError, match="disk unavailable"):
        run_floaty_candidate(
            model,
            [lesson(1, "Say alpha", "alpha"), lesson(2, "Say beta", "beta")],
            [lesson(10, "Say gamma", "gamma")],
            FixedConversationTokenizer(),
            tmp_path,
            expected_foundation_hash=model.foundation_hash(),
            expected_config_hash=model.config_hash(),
            allow_test_tokenizer_fallback=True,
            config=training_config(),
        )

    current = dict(tree_flatten(model.floaty.parameters()))
    for name, before in floaty_before.items():
        np.testing.assert_array_equal(np.asarray(current[name]), before)
    assert not list(tmp_path.glob(".floaty-candidate-*"))
