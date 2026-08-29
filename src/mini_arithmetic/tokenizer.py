"""A zero-setup UTF-8 byte tokenizer with explicit apprenticeship roles."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


class ArithmeticTokenizer:
    """Represent every UTF-8 byte directly and keep digits atomic.

    A byte vocabulary needs no pretraining corpus and can encode any human or
    teacher-model conversation. Dedicated role IDs let training distinguish a
    user question, the student's response, teaching, retrieved memory, and a
    consolidated lesson without relying on fragile textual delimiters.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    USER = 3
    ASSISTANT = 4
    TEACHER = 5
    MEMORY = 6
    LESSON = 7
    BYTE_OFFSET = 8
    _ROLE_IDS = {
        "user": USER,
        "assistant": ASSISTANT,
        "teacher": TEACHER,
        "memory": MEMORY,
        "lesson": LESSON,
    }

    @property
    def vocab_size(self) -> int:
        return self.BYTE_OFFSET + 256

    def encode(
        self,
        text: str,
        *,
        bos: bool = False,
        eos: bool = False,
        role: str | None = None,
    ) -> list[int]:
        tokens = [byte + self.BYTE_OFFSET for byte in text.encode("utf-8")]
        if role is not None:
            try:
                tokens.insert(0, self._ROLE_IDS[role])
            except KeyError as error:
                raise ValueError(f"Unknown conversation role: {role!r}") from error
        if bos:
            tokens.insert(0, self.BOS)
        if eos:
            tokens.append(self.EOS)
        return tokens

    def encode_dialogue(
        self,
        turns: Sequence[tuple[str, str]],
        *,
        bos: bool = True,
        eos: bool = True,
    ) -> list[int]:
        tokens = [self.BOS] if bos else []
        for role, text in turns:
            tokens.extend(self.encode(text, role=role))
        if eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        raw = bytes(
            int(token) - self.BYTE_OFFSET
            for token in tokens
            if self.BYTE_OFFSET <= int(token) < self.vocab_size
        )
        return raw.decode("utf-8", errors="replace")

    def role_id(self, role: str) -> int:
        try:
            return self._ROLE_IDS[role]
        except KeyError as error:
            raise ValueError(f"Unknown conversation role: {role!r}") from error
