"""Interactive arithmetic chat with a trained MLM checkpoint."""

from __future__ import annotations

import argparse

from .generate import complete
from .model import create_model
from .tokenizer import ArithmeticTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/100m.json")
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    model = create_model(args.config, args.weights)
    model.eval()
    tokenizer = ArithmeticTokenizer()
    print("MLM Arithmetic 100M is running. Type a maths problem, or quit.")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit"}:
            break
        output = complete(
            model, tokenizer, f"Problem: {question}\nAnswer:", max_tokens=64
        )
        answer = output.split("Answer:", 1)[-1].splitlines()[0].strip()
        print(f"MLM> {answer}")


if __name__ == "__main__":
    main()
