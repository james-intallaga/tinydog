"""Measure exact-answer accuracy on a sealed deterministic test stream."""

from __future__ import annotations

import argparse
from collections import Counter
import re

from .data import ArithmeticCurriculum
from .generate import complete
from .model import create_model
from .tokenizer import ArithmeticTokenizer


ANSWER_PATTERN = re.compile(r"Answer:\s*([^\n]+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/100m.json")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--max-level", type=int, default=6)
    args = parser.parse_args()

    tokenizer = ArithmeticTokenizer()
    model = create_model(args.config, args.weights)
    model.eval()
    curriculum = ArithmeticCurriculum(args.seed, split="test")
    correct, totals = Counter(), Counter()
    failures = []
    for _ in range(args.examples):
        example = curriculum.sample(args.max_level)
        prompt = f"Problem: {example.question}\nAnswer:"
        output = complete(model, tokenizer, prompt, max_tokens=64)
        matches = ANSWER_PATTERN.findall(output)
        prediction = matches[-1].strip().rstrip(".") if matches else ""
        totals[example.task] += 1
        if prediction == example.answer:
            correct[example.task] += 1
        elif len(failures) < 10:
            failures.append((example.question, example.answer, prediction))

    total = sum(totals.values())
    total_correct = sum(correct.values())
    print(f"Overall: {total_correct}/{total} = {total_correct/max(total, 1):.1%}")
    for task in sorted(totals):
        print(
            f"{task:20s} {correct[task]:4d}/{totals[task]:4d} = {correct[task]/totals[task]:.1%}"
        )
    if failures:
        print("\nFirst failures:")
        for question, expected, predicted in failures:
            print(f"- {question} expected={expected!r} predicted={predicted!r}")


if __name__ == "__main__":
    main()
