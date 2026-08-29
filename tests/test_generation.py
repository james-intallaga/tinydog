from types import SimpleNamespace
import threading

import mlx.core as mx

from mini_arithmetic.generate import generate_tokens, should_stop_generation
from mini_arithmetic.runtime import ModelRuntime
from mini_arithmetic.tokenizer import ArithmeticTokenizer


def _encoded(text: str) -> list[int]:
    return ArithmeticTokenizer().encode(text)


def test_stops_at_answer_newline_and_nonnumeric_repetition() -> None:
    tokenizer = ArithmeticTokenizer()
    assert should_stop_generation(_encoded("42\n"), tokenizer, stop_at_newline=True)
    assert should_stop_generation(_encoded("abcabcabc"), tokenizer)
    assert should_stop_generation(_encoded("!!!!!!!!"), tokenizer)


def test_repeating_digits_remain_valid_arithmetic_output() -> None:
    tokenizer = ArithmeticTokenizer()
    assert not should_stop_generation(_encoded("111111111111"), tokenizer)
    assert not should_stop_generation(_encoded("121212121212"), tokenizer)


class _ScriptedModel:
    def __init__(self, outputs: list[int], vocab_size: int) -> None:
        self.outputs = iter(outputs)
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(context_length=64)

    def __call__(self, tokens):
        target = next(self.outputs)
        length = int(tokens.shape[1])
        logits = [-1_000.0] * self.vocab_size
        logits[target] = 1_000.0
        return mx.array([[logits] * length])


def test_generation_stops_before_structural_token() -> None:
    tokenizer = ArithmeticTokenizer()
    output = _encoded("42") + [tokenizer.ASSISTANT] + _encoded("ignored")
    model = _ScriptedModel(output, tokenizer.vocab_size)
    generated = generate_tokens(
        model,
        tokenizer,
        [tokenizer.BOS],
        max_tokens=20,
        stop_tokens={tokenizer.ASSISTANT},
    )
    assert tokenizer.decode(generated) == "42"


def test_runtime_caps_oversized_chat_budget(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def fake_complete(_model, _tokenizer, _turns, *, max_tokens, temperature):
        captured["max_tokens"] = max_tokens
        return "4"

    monkeypatch.setattr("mini_arithmetic.runtime.complete_dialogue", fake_complete)
    runtime = object.__new__(ModelRuntime)
    runtime._lock = threading.RLock()
    runtime.model = object()
    runtime.tokenizer = ArithmeticTokenizer()

    assert runtime.answer([("user", "2 + 2")], max_tokens=10_000) == "4"
    assert captured["max_tokens"] == ModelRuntime.MAX_CHAT_NEW_TOKENS
