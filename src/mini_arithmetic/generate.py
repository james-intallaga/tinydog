"""Generate an answer from a trained arithmetic checkpoint."""

from __future__ import annotations

import argparse

import mlx.core as mx

from .model import create_model
from .tokenizer import ArithmeticTokenizer


def should_stop_generation(
    generated: list[int],
    tokenizer: ArithmeticTokenizer,
    *,
    stop_at_newline: bool = False,
) -> bool:
    """Return whether a byte-level continuation has clearly finished or collapsed.

    Arithmetic answers can legitimately contain long or repeating digit strings,
    so numeric-only suffixes are deliberately exempt from repetition stopping.
    """
    if not generated:
        return False

    last = generated[-1]
    if tokenizer.BYTE_OFFSET <= last < tokenizer.vocab_size:
        byte = last - tokenizer.BYTE_OFFSET
        if stop_at_newline and byte in {10, 13}:
            return True
        # Raw control bytes have no place in a text answer. Tab is harmless and
        # newlines are handled by the caller's chosen policy above.
        if byte < 32 and byte not in {9, 10, 13}:
            return True

    byte_values = [
        token - tokenizer.BYTE_OFFSET
        for token in generated
        if tokenizer.BYTE_OFFSET <= token < tokenizer.vocab_size
    ]
    for period in range(1, min(12, len(byte_values) // 3) + 1):
        suffix = byte_values[-3 * period :]
        unit = suffix[:period]
        if suffix == unit * 3 and not all(48 <= byte <= 57 for byte in suffix):
            return True
    return False


def generate_tokens(
    model,
    tokenizer: ArithmeticTokenizer,
    prefix: list[int],
    max_tokens: int = 128,
    temperature: float = 0.0,
    stop_tokens: set[int] | None = None,
    stop_at_newline: bool = False,
    stop_repetition: bool = True,
) -> list[int]:
    """Generate only the new token IDs following a prepared prefix."""
    tokens = list(prefix)
    generated: list[int] = []
    for _ in range(max_tokens):
        window = tokens[-model.config.context_length :]
        logits = model(mx.array([window]))[0, -1]
        if temperature > 0:
            next_token = int(mx.random.categorical(logits / temperature).item())
        else:
            next_token = int(mx.argmax(logits).item())
        if next_token == tokenizer.EOS or (stop_tokens and next_token in stop_tokens):
            break
        tokens.append(next_token)
        generated.append(next_token)
        if stop_repetition and should_stop_generation(
            generated,
            tokenizer,
            stop_at_newline=stop_at_newline,
        ):
            break
    return generated


def complete(
    model,
    tokenizer: ArithmeticTokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    prefix = tokenizer.encode(prompt, bos=True)
    generated = generate_tokens(model, tokenizer, prefix, max_tokens, temperature)
    return tokenizer.decode([*prefix, *generated])


def complete_dialogue(
    model,
    tokenizer: ArithmeticTokenizer,
    turns: list[tuple[str, str]],
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> str:
    """Generate the student's next turn using dedicated role tokens."""
    prefix = tokenizer.encode_dialogue(turns, bos=True, eos=False)
    prefix.append(tokenizer.ASSISTANT)
    generated = generate_tokens(
        model,
        tokenizer,
        prefix,
        max_tokens,
        temperature,
        stop_tokens={
            tokenizer.PAD,
            tokenizer.BOS,
            tokenizer.USER,
            tokenizer.ASSISTANT,
            tokenizer.TEACHER,
            tokenizer.MEMORY,
            tokenizer.LESSON,
        },
        stop_at_newline=True,
    )
    return tokenizer.decode(generated).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/100m.json")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    model = create_model(args.config, args.weights)
    model.eval()
    prompt = f"Problem: {args.problem}\nAnswer:"
    print(
        complete(
            model, ArithmeticTokenizer(), prompt, args.max_tokens, args.temperature
        )
    )


if __name__ == "__main__":
    main()
