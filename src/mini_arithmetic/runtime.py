"""Thread-safe live model runtime with checkpoint hot-swapping."""

from __future__ import annotations

from pathlib import Path
import threading
import unicodedata

import mlx.core as mx

from .generate import complete_dialogue
from .model import create_model
from .tokenizer import ArithmeticTokenizer


class ModelRuntime:
    """Keep one active student model available while candidates train elsewhere."""

    MAX_CHAT_NEW_TOKENS = 32

    def __init__(self, config_path: str | Path, weights_path: str | Path) -> None:
        self.config_path = str(config_path)
        self._weights_path = str(weights_path)
        self._lock = threading.RLock()
        self.tokenizer = ArithmeticTokenizer()
        self.model = self.prepare_checkpoint(self._weights_path)

    def _validate_model(self, model) -> None:
        if model.config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"Checkpoint config vocabulary {model.config.vocab_size} does not "
                f"match tokenizer vocabulary {self.tokenizer.vocab_size}"
            )

    def prepare_checkpoint(self, weights_path: str | Path):
        """Fully load and validate a checkpoint without touching live inference."""
        model = create_model(self.config_path, str(weights_path))
        self._validate_model(model)
        model.eval()
        mx.eval(model.parameters())
        return model

    def activate_checkpoint(self, model, weights_path: str | Path) -> None:
        """Swap an already validated model with one lock-protected assignment."""
        self._validate_model(model)
        with self._lock:
            self.model = model
            self._weights_path = str(weights_path)

    @property
    def weights_path(self) -> str:
        with self._lock:
            return self._weights_path

    def answer(
        self,
        turns: list[tuple[str, str]],
        *,
        max_tokens: int = MAX_CHAT_NEW_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        token_budget = max(0, min(int(max_tokens), self.MAX_CHAT_NEW_TOKENS))
        with self._lock:
            answer = complete_dialogue(
                self.model,
                self.tokenizer,
                turns,
                max_tokens=token_budget,
                temperature=temperature,
            )
        return "".join(
            character
            for character in answer
            if character in {"\n", "\t"}
            or not unicodedata.category(character).startswith("C")
        ).strip()

    def reload(self, weights_path: str | Path) -> None:
        """Load separately, then atomically replace the active model object."""
        new_path = str(weights_path)
        prepared = self.prepare_checkpoint(new_path)
        self.activate_checkpoint(prepared, new_path)

    def status(self) -> dict[str, object]:
        with self._lock:
            config = self.model.config
            return {
                "parameters": self.model.parameter_count(),
                "context_length": config.context_length,
                "vocab_size": config.vocab_size,
                "tokenizer": "utf8-byte-apprenticeship-v1",
                "checkpoint": self._weights_path,
                "max_chat_new_tokens": self.MAX_CHAT_NEW_TOKENS,
            }
