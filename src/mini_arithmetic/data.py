"""Infinite, exactly verified arithmetic training data."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import random

import numpy as np

from .tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class Example:
    task: str
    question: str
    answer: str
    work: str
    key: str

    def render(self, worked: bool = False) -> str:
        middle = f"Work: {self.work}\n" if worked else ""
        return f"Problem: {self.question}\n{middle}Answer: {self.answer}\n"


SPLIT_BUCKETS = {
    "train": range(0, 90),
    "validation": range(90, 95),
    "test": range(95, 100),
}


def split_for_key(key: str) -> str:
    """Assign a canonical problem to a stable, disjoint data split."""
    bucket = (
        int.from_bytes(hashlib.blake2b(key.encode(), digest_size=2).digest(), "big")
        % 100
    )
    return next(name for name, buckets in SPLIT_BUCKETS.items() if bucket in buckets)


class ArithmeticCurriculum:
    def __init__(self, seed: int = 0, split: str = "train") -> None:
        if split not in SPLIT_BUCKETS:
            raise ValueError(f"Unknown split {split!r}")
        self.rng = random.Random(seed)
        self.split = split

    def sample(self, max_level: int = 6) -> Example:
        level = self.rng.randint(1, max(1, min(max_level, 6)))
        while True:
            example = self._sample_level(level)
            if split_for_key(example.key) == self.split:
                return example

    def _sample_level(self, level: int) -> Example:
        samplers = {
            1: self._small_add_sub,
            2: self._multiplication,
            3: self._large_add_sub,
            4: self._division,
            5: self._mixed_expression,
            6: self._fractions_percentages,
        }
        return samplers[level]()

    def _small_add_sub(self) -> Example:
        a, b = self.rng.randint(0, 99), self.rng.randint(0, 99)
        if self.rng.random() < 0.5:
            return Example(
                "addition",
                f"What is {a} + {b}?",
                str(a + b),
                f"{a} + {b} = {a + b}.",
                f"add:{a}:{b}",
            )
        high, low = max(a, b), min(a, b)
        return Example(
            "subtraction",
            f"Calculate {high} - {low}.",
            str(high - low),
            f"{high} - {low} = {high - low}.",
            f"sub:{high}:{low}",
        )

    def _multiplication(self) -> Example:
        a, b = self.rng.randint(0, 99), self.rng.randint(0, 99)
        product = a * b
        question = self.rng.choice([f"What is {a} * {b}?", f"Multiply {a} by {b}."])
        return Example(
            "multiplication",
            question,
            str(product),
            f"{a} * {b} = {product}.",
            f"mul:{a}:{b}",
        )

    def _large_add_sub(self) -> Example:
        digits = self.rng.randint(3, 8)
        a = self.rng.randint(10 ** (digits - 1), 10**digits - 1)
        b = self.rng.randint(10 ** (digits - 1), 10**digits - 1)
        if self.rng.random() < 0.5:
            result, symbol, task = a + b, "+", "large_addition"
        else:
            a, b = max(a, b), min(a, b)
            result, symbol, task = a - b, "-", "large_subtraction"
        return Example(
            task,
            f"Calculate {a} {symbol} {b}.",
            str(result),
            f"{a} {symbol} {b} = {result}.",
            f"{symbol}:{a}:{b}",
        )

    def _division(self) -> Example:
        divisor = self.rng.randint(1, 99)
        quotient = self.rng.randint(0, 999)
        dividend = divisor * quotient
        question = self.rng.choice(
            [f"What is {dividend} / {divisor}?", f"Divide {dividend} by {divisor}."]
        )
        return Example(
            "exact_division",
            question,
            str(quotient),
            f"{divisor} * {quotient} = {dividend}, so {dividend} / {divisor} = {quotient}.",
            f"div:{dividend}:{divisor}",
        )

    def _mixed_expression(self) -> Example:
        a, b, c = (self.rng.randint(-99, 99) for _ in range(3))
        if self.rng.random() < 0.5:
            result = a + b * c
            expression = f"{a} + {b} * {c}"
            work = f"Multiply first: {b} * {c} = {b*c}; then {a} + {b*c} = {result}."
            key = f"mixed:addmul:{a}:{b}:{c}"
        else:
            result = (a + b) * c
            expression = f"({a} + {b}) * {c}"
            work = f"Parentheses first: {a} + {b} = {a+b}; then {a+b} * {c} = {result}."
            key = f"mixed:group:{a}:{b}:{c}"
        return Example(
            "mixed_expression", f"Evaluate {expression}.", str(result), work, key
        )

    def _fractions_percentages(self) -> Example:
        if self.rng.random() < 0.5:
            a, b = self.rng.randint(1, 20), self.rng.randint(1, 20)
            c, d = self.rng.randint(1, 20), self.rng.randint(1, 20)
            result = Fraction(a, b) + Fraction(c, d)
            answer = (
                str(result.numerator)
                if result.denominator == 1
                else f"{result.numerator}/{result.denominator}"
            )
            work = f"Use a common denominator and reduce: {a}/{b} + {c}/{d} = {answer}."
            return Example(
                "fraction_addition",
                f"Add and simplify: {a}/{b} + {c}/{d}.",
                answer,
                work,
                f"fracadd:{a}:{b}:{c}:{d}",
            )
        percent = self.rng.choice([5, 10, 20, 25, 50, 75])
        base_unit = 100 // math.gcd(100, percent)
        number = base_unit * self.rng.randint(1, 500)
        result = number * percent // 100
        return Example(
            "percentage",
            f"What is {percent}% of {number}?",
            str(result),
            f"{percent}/100 * {number} = {result}.",
            f"percent:{percent}:{number}",
        )


def curriculum_level(step: int, total_steps: int) -> int:
    """Introduce one level at a time, while retaining all earlier skills."""
    if total_steps <= 0:
        return 6
    return min(6, 1 + int(6 * step / total_steps))


def make_batch(
    curriculum: ArithmeticCurriculum,
    tokenizer: ArithmeticTokenizer,
    batch_size: int,
    context_length: int,
    max_level: int,
    worked_probability: float = 0.3,
    dialogue_probability: float = 0.5,
) -> np.ndarray:
    """Pack independently generated examples into fixed-length LM sequences."""
    batch = np.empty((batch_size, context_length + 1), dtype=np.int32)
    for row in range(batch_size):
        tokens: list[int] = []
        while len(tokens) < context_length + 1:
            example = curriculum.sample(max_level=max_level)
            worked = curriculum.rng.random() < worked_probability
            if curriculum.rng.random() < dialogue_probability:
                answer = (
                    f"{example.work}\nAnswer: {example.answer}"
                    if worked
                    else example.answer
                )
                encoded = tokenizer.encode_dialogue(
                    [("user", example.question), ("assistant", answer)],
                    bos=True,
                    eos=True,
                )
            else:
                encoded = tokenizer.encode(example.render(worked), bos=True, eos=True)
            tokens.extend(encoded)
        batch[row] = tokens[: context_length + 1]
    return batch
