from mini_arithmetic.apprenticeship import verify_answer
from mini_arithmetic.data import (
    ArithmeticCurriculum,
    curriculum_level,
    make_batch,
    split_for_key,
)
from mini_arithmetic.tokenizer import ArithmeticTokenizer


def test_stable_disjoint_split() -> None:
    assert split_for_key("add:2:2") == split_for_key("add:2:2")
    for split in ("train", "validation", "test"):
        curriculum = ArithmeticCurriculum(seed=3, split=split)
        for _ in range(20):
            example = curriculum.sample(max_level=6)
            assert split_for_key(example.key) == split


def test_batch_shape_and_curriculum() -> None:
    tokenizer = ArithmeticTokenizer()
    curriculum = ArithmeticCurriculum(seed=1, split="train")
    batch = make_batch(
        curriculum, tokenizer, batch_size=2, context_length=64, max_level=3
    )
    assert batch.shape == (2, 65)
    assert curriculum_level(0, 100) == 1
    assert curriculum_level(99, 100) == 6


def test_fraction_and_percentage_curriculum_answers_are_exact() -> None:
    curriculum = ArithmeticCurriculum(seed=75, split="train")
    for _ in range(500):
        example = curriculum._fractions_percentages()
        assert verify_answer(example.question, example.answer).is_correct
