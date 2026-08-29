"""Frozen 32K byte-level BPE tokenizer for scratch-model training.

This module is intentionally separate from :mod:`mini_arithmetic.tokenizer`,
which remains the tokenizer for the currently running apprenticeship model.
The artifact produced here is immutable and checksum-verified when loaded.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import tokenizers
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)


VOCAB_SIZE = 32_768
TOKENIZER_FILENAME = "tokenizer.json"
METADATA_FILENAME = "tokenizer_meta.json"
EXTENSION_SLOT_COUNT = 256

SPECIAL_TOKEN_ITEMS: tuple[tuple[str, str], ...] = (
    ("pad", "<|pad|>"),
    ("bos", "<|bos|>"),
    ("eos", "<|eos|>"),
    ("unk", "<|unk|>"),
    ("system", "<|system|>"),
    ("user", "<|user|>"),
    ("assistant", "<|assistant|>"),
    ("teacher", "<|teacher|>"),
    ("correction", "<|correction|>"),
    ("memory", "<|memory|>"),
    ("work", "<|work|>"),
    ("final", "<|final|>"),
    ("end_turn", "<|end_turn|>"),
)
SPECIAL_TOKENS = dict(SPECIAL_TOKEN_ITEMS)
ROLE_NAMES = frozenset(name for name, _ in SPECIAL_TOKEN_ITEMS[4:-1])
EXTENSION_TOKENS: tuple[str, ...] = tuple(
    f"<|extension_{index:03d}|>" for index in range(EXTENSION_SLOT_COUNT)
)
_CONTROL_TOKEN_LITERAL = re.compile(
    r"<\|(?:"
    + "|".join(re.escape(name) for name, _ in SPECIAL_TOKEN_ITEMS)
    + r"|extension_\d{3}|reserved_\d{5})\|>"
)

# Each item below is a pre-tokenization boundary.  Consequently neither a
# digit nor an arithmetic delimiter can be absorbed into a neighbouring BPE
# merge.  Unicode operators are seeded during training so each is also one ID.
ATOMIC_TOKENS: tuple[str, ...] = tuple("0123456789+-*/%^=<>()[]{}.,:;") + (
    "×",
    "÷",
    "−",
    "±",
    "√",
    "≠",
    "≤",
    "≥",
    "≈",
)


class TokenizerIntegrityError(ValueError):
    """Raised when a frozen tokenizer artifact fails validation."""


@dataclass(frozen=True)
class DialogueBoundary:
    """Offsets that make a dialogue turn auditable in a packed token record."""

    role: str
    role_offset: int
    content_offset: int
    content_length: int
    end_turn_offset: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "role": self.role,
            "role_offset": self.role_offset,
            "content_offset": self.content_offset,
            "content_length": self.content_length,
            "end_turn_offset": self.end_turn_offset,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_pattern() -> str:
    alternatives = "|".join(
        re.escape(item)
        for item in sorted(ATOMIC_TOKENS, key=lambda item: (-len(item), item))
    )
    return f"(?:{alternatives})"


def contains_control_token_literal(text: str) -> bool:
    """Whether ordinary text could impersonate a structural tokenizer ID."""

    return _CONTROL_TOKEN_LITERAL.search(text) is not None


def contains_unsafe_text_control(text: str) -> bool:
    """Whether decoded model text contains a C0, DEL, or C1 control."""

    return any(
        ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in text
    )


def sanitize_generated_text(text: str) -> str:
    """Make untrusted random-model text safe for the dialogue ledger.

    Control bytes are replaced with spaces rather than persisted into the next
    training prompt.  The visible ellipsis is a final defensive fallback; the
    runtimes additionally mask blank/control IDs on their first decode step.
    """

    if not isinstance(text, str):
        raise TypeError("Generated text must be a string")
    cleaned = "".join(
        " " if ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F else character
        for character in text
    )
    # A random model can theoretically spell a reserved role literal from
    # ordinary byte/BPE pieces even though the structural ID itself is masked.
    # Neutralize it before the reply becomes the next turn's frozen prompt.
    cleaned = _CONTROL_TOKEN_LITERAL.sub("[control]", cleaned).strip()
    return cleaned or "…"


def extension_slot_id(slot: int) -> int:
    """Map a versioned protocol extension slot to its stable tokenizer ID."""

    if not isinstance(slot, int) or not 0 <= slot < EXTENSION_SLOT_COUNT:
        raise ValueError(f"Extension slot must be in [0, {EXTENSION_SLOT_COUNT})")
    return len(SPECIAL_TOKEN_ITEMS) + slot


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _coerce_turn(turn: Mapping[str, Any] | Sequence[str]) -> tuple[str, str]:
    if isinstance(turn, Mapping):
        role = turn.get("role")
        content = turn.get("content")
    elif (
        isinstance(turn, Sequence)
        and not isinstance(turn, (str, bytes))
        and len(turn) == 2
    ):
        role, content = turn
    else:
        raise TypeError("Dialogue turns must be (role, content) pairs or mappings")
    if not isinstance(role, str) or role not in ROLE_NAMES:
        raise ValueError(f"Unknown scratch dialogue role: {role!r}")
    if not isinstance(content, str):
        raise TypeError("Dialogue content must be a string")
    return role, content


class ScratchTokenizer:
    """Read-only wrapper around a checksum-verified tokenizer artifact."""

    def __init__(
        self, backend: Tokenizer, metadata: Mapping[str, Any], artifact_dir: Path
    ) -> None:
        self._backend = backend
        self._metadata = dict(metadata)
        self.artifact_dir = artifact_dir
        self._validate()
        self._generation_blocked_token_ids = self._derive_generation_blocked_ids()
        self._first_generation_blocked_token_ids = (
            self._derive_first_generation_blocked_ids()
        )

    @classmethod
    def load(
        cls, path: str | Path, *, verify_checksum: bool = True
    ) -> "ScratchTokenizer":
        artifact = Path(path)
        if artifact.is_dir():
            tokenizer_path = artifact / TOKENIZER_FILENAME
            metadata_path = artifact / METADATA_FILENAME
            artifact_dir = artifact
        else:
            tokenizer_path = artifact
            metadata_path = artifact.with_name(METADATA_FILENAME)
            artifact_dir = artifact.parent
        if not tokenizer_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Expected {TOKENIZER_FILENAME} and {METADATA_FILENAME} in {artifact_dir}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if verify_checksum:
            expected = metadata.get("tokenizer_sha256")
            actual = _sha256_file(tokenizer_path)
            if not isinstance(expected, str) or actual != expected:
                raise TokenizerIntegrityError(
                    f"Tokenizer checksum mismatch: expected {expected!r}, found {actual}"
                )
        return cls(Tokenizer.from_file(str(tokenizer_path)), metadata, artifact_dir)

    def _validate(self) -> None:
        if self._metadata.get("format_version") != 1:
            raise TokenizerIntegrityError(
                "Unsupported scratch tokenizer metadata version"
            )
        if (
            self.vocab_size != VOCAB_SIZE
            or self._metadata.get("vocab_size") != VOCAB_SIZE
        ):
            raise TokenizerIntegrityError(
                f"Scratch tokenizer must have exactly {VOCAB_SIZE:,} IDs"
            )
        if self._backend.normalizer is not None:
            raise TokenizerIntegrityError(
                "Scratch tokenizer must not normalize Unicode"
            )
        declared = self._metadata.get("special_token_ids")
        expected: dict[str, int] = {}
        for expected_id, (name, token) in enumerate(SPECIAL_TOKEN_ITEMS):
            token_id = self._backend.token_to_id(token)
            if token_id != expected_id:
                raise TokenizerIntegrityError(
                    f"Special token {token!r} must have ID {expected_id}, found {token_id}"
                )
            expected[name] = expected_id
        if declared != expected:
            raise TokenizerIntegrityError(
                "Special-token metadata does not match tokenizer IDs"
            )
        extension = self._metadata.get("extension_slots")
        if not isinstance(extension, Mapping):
            raise TokenizerIntegrityError("Missing extension-slot metadata")
        extension_start = extension.get("start_id")
        extension_count = extension.get("count")
        extension_end = extension.get("end_id_exclusive")
        if type(extension_start) is not int or extension_start != len(
            SPECIAL_TOKEN_ITEMS
        ):
            raise TokenizerIntegrityError("Extension ID range has moved")
        if type(extension_count) is not int or extension_count != EXTENSION_SLOT_COUNT:
            raise TokenizerIntegrityError("Extension slot count has changed")
        if (
            type(extension_end) is not int
            or extension_end != extension_start + extension_count
        ):
            raise TokenizerIntegrityError("Extension slot range is inconsistent")
        for slot, token in enumerate(EXTENSION_TOKENS):
            if self._backend.token_to_id(token) != len(SPECIAL_TOKEN_ITEMS) + slot:
                raise TokenizerIntegrityError(
                    f"Extension slot {slot} has an unstable ID"
                )
        base_size = self._metadata.get("base_vocabulary_size")
        reserved_count = self._metadata.get("reserved_token_count")
        if (
            type(base_size) is not int
            or type(reserved_count) is not int
            or not extension_end <= base_size <= self.vocab_size
            or reserved_count < 0
            or base_size + reserved_count != self.vocab_size
        ):
            raise TokenizerIntegrityError("Reserved text-token range is inconsistent")
        for offset in range(reserved_count):
            token = f"<|reserved_{offset:05d}|>"
            if self._backend.token_to_id(token) != base_size + offset:
                raise TokenizerIntegrityError(
                    f"Reserved token {offset} has an unstable ID"
                )
        for item in ATOMIC_TOKENS:
            encoded = self._backend.encode(item, add_special_tokens=False).ids
            if len(encoded) != 1 or self.decode(encoded) != item:
                raise TokenizerIntegrityError(
                    f"Arithmetic token {item!r} is not atomic"
                )

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy so callers cannot mutate the loaded contract."""

        return copy.deepcopy(self._metadata)

    @property
    def inactive_text_token_ids(self) -> tuple[int, ...]:
        """IDs reserved for future modalities or vocabulary padding.

        They are valid embedding rows but are not legal text-generation or
        text-loss classes until a future versioned modality registry assigns
        extension semantics.  Learned BPE IDs and dialogue structural IDs are
        deliberately excluded.
        """

        extension = self._metadata["extension_slots"]
        extension_ids = range(
            int(extension["start_id"]),
            int(extension["end_id_exclusive"]),
        )
        reserved_start = int(self._metadata["base_vocabulary_size"])
        reserved_ids = range(reserved_start, self.vocab_size)
        return tuple((*extension_ids, *reserved_ids))

    @property
    def generation_blocked_token_ids(self) -> tuple[int, ...]:
        """IDs that can never be emitted as ordinary assistant text.

        EOS and end-turn are deliberately absent because they remain valid
        stop signals after the first visible token.
        """

        return self._generation_blocked_token_ids

    @property
    def first_generation_blocked_token_ids(self) -> tuple[int, ...]:
        """IDs suppressed until one safe non-blank text token is emitted."""

        return self._first_generation_blocked_token_ids

    def _derive_generation_blocked_ids(self) -> tuple[int, ...]:
        blocked = set(self.inactive_text_token_ids)
        blocked.update(
            {
                self.pad_id,
                self.bos_id,
                self.unk_id,
                *(self.role_id(role) for role in ROLE_NAMES),
            }
        )
        base_size = int(self._metadata["base_vocabulary_size"])
        for token_id in range(base_size):
            if contains_unsafe_text_control(self.decode([token_id])):
                blocked.add(token_id)
        blocked.discard(self.eos_id)
        blocked.discard(self.end_turn_id)
        return tuple(sorted(blocked))

    def _derive_first_generation_blocked_ids(self) -> tuple[int, ...]:
        blocked = set(self.generation_blocked_token_ids)
        blocked.update({self.eos_id, self.end_turn_id})
        base_size = int(self._metadata["base_vocabulary_size"])
        for token_id in range(base_size):
            decoded = self.decode([token_id])
            if not decoded.strip() or contains_unsafe_text_control(decoded):
                blocked.add(token_id)
        return tuple(sorted(blocked))

    @property
    def pad_id(self) -> int:
        return self.token_id("pad")

    @property
    def bos_id(self) -> int:
        return self.token_id("bos")

    @property
    def eos_id(self) -> int:
        return self.token_id("eos")

    @property
    def unk_id(self) -> int:
        return self.token_id("unk")

    @property
    def end_turn_id(self) -> int:
        return self.token_id("end_turn")

    def token_id(self, name: str) -> int:
        try:
            token = SPECIAL_TOKENS[name]
        except KeyError as error:
            raise ValueError(f"Unknown special token name: {name!r}") from error
        token_id = self._backend.token_to_id(token)
        if token_id is None:  # Guard against an externally replaced artifact.
            raise TokenizerIntegrityError(f"Missing special token {token!r}")
        return token_id

    def role_id(self, role: str) -> int:
        if role not in ROLE_NAMES:
            raise ValueError(f"Unknown scratch dialogue role: {role!r}")
        return self.token_id(role)

    def extension_id(self, slot: int) -> int:
        """Return a stable reserved protocol ID for a future modality.

        A separate, versioned modality registry can assign meanings such as
        ``audio_start`` or ``video_frame`` to these slots without changing the
        frozen tokenizer or moving any learned embedding row.
        """

        return extension_slot_id(slot)

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be a string")
        if contains_control_token_literal(text):
            raise ValueError("Raw text contains a reserved structural-token literal")
        ids = list(self._backend.encode(text, add_special_tokens=False).ids)
        if bos:
            ids.insert(0, self.bos_id)
        if eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self, token_ids: Iterable[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self._backend.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=skip_special_tokens,
        )

    def encode_dialogue(
        self,
        turns: Sequence[Mapping[str, Any] | Sequence[str]],
        *,
        bos: bool = True,
        eos: bool = True,
    ) -> list[int]:
        ids, _ = self.encode_dialogue_with_boundaries(turns, bos=bos, eos=eos)
        return ids

    def encode_dialogue_with_boundaries(
        self,
        turns: Sequence[Mapping[str, Any] | Sequence[str]],
        *,
        bos: bool = True,
        eos: bool = True,
    ) -> tuple[list[int], list[DialogueBoundary]]:
        ids = [self.bos_id] if bos else []
        boundaries: list[DialogueBoundary] = []
        for raw_turn in turns:
            role, content = _coerce_turn(raw_turn)
            role_offset = len(ids)
            ids.append(self.role_id(role))
            content_offset = len(ids)
            content_ids = self.encode(content)
            ids.extend(content_ids)
            end_turn_offset = len(ids)
            ids.append(self.end_turn_id)
            boundaries.append(
                DialogueBoundary(
                    role=role,
                    role_offset=role_offset,
                    content_offset=content_offset,
                    content_length=len(content_ids),
                    end_turn_offset=end_turn_offset,
                )
            )
        if eos:
            ids.append(self.eos_id)
        return ids, boundaries


def build_scratch_tokenizer(
    texts: Iterable[str],
    output_dir: str | Path,
    *,
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = 2,
    overwrite: bool = False,
) -> ScratchTokenizer:
    """Train and freeze a deterministic byte-level BPE tokenizer.

    ``texts`` is consumed once and is not normalized.  Its order therefore
    remains part of the reproducible training artifact and corpus checksum.
    """

    if vocab_size != VOCAB_SIZE:
        raise ValueError(f"The scratch architecture requires vocab_size={VOCAB_SIZE}")
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    destination = Path(output_dir)
    tokenizer_path = destination / TOKENIZER_FILENAME
    metadata_path = destination / METADATA_FILENAME
    if not overwrite and (tokenizer_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Frozen tokenizer artifact already exists in {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    backend = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk"], byte_fallback=True))
    backend.normalizer = None
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(_atomic_pattern()), behavior="isolated"),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, trim_offsets=False, use_regex=True
            ),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    special_added_tokens = [
        AddedToken(token, normalized=False, special=True)
        for _, token in SPECIAL_TOKEN_ITEMS
    ]
    extension_added_tokens = [
        AddedToken(token, normalized=False, special=True) for token in EXTENSION_TOKENS
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=min_frequency,
        show_progress=False,
        special_tokens=special_added_tokens + extension_added_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    corpus_digest = hashlib.sha256()
    text_count = 0

    def training_iterator() -> Iterable[str]:
        nonlocal text_count
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Every tokenizer training item must be a string")
            if contains_control_token_literal(text):
                raise ValueError(
                    "Tokenizer training text contains a reserved structural-token literal"
                )
            encoded = text.encode("utf-8")
            corpus_digest.update(len(encoded).to_bytes(8, "big"))
            corpus_digest.update(encoded)
            text_count += 1
            yield text
        # This deterministic seed is tokenizer engineering, not hidden model
        # training data.  It guarantees that multi-byte maths operators merge
        # to one ID even if an early corpus happens not to contain them.
        repetitions = max(2, min_frequency)
        yield " ".join(item for item in ATOMIC_TOKENS for _ in range(repetitions))

    backend.train_from_iterator(training_iterator(), trainer=trainer)
    if text_count == 0:
        raise ValueError("At least one tokenizer training text is required")

    current_size = backend.get_vocab_size(with_added_tokens=True)
    if current_size > VOCAB_SIZE:
        raise RuntimeError(
            f"Tokenizer trainer produced {current_size} IDs, above the fixed limit"
        )
    reserved_count = VOCAB_SIZE - current_size
    if reserved_count:
        added = backend.add_special_tokens(
            [
                AddedToken(f"<|reserved_{index:05d}|>", normalized=False, special=True)
                for index in range(reserved_count)
            ]
        )
        if added != reserved_count:
            raise RuntimeError("Could not pad tokenizer to its fixed 32K ID space")

    # Validate the in-memory artifact before making it visible to callers.
    if backend.get_vocab_size(with_added_tokens=True) != VOCAB_SIZE:
        raise RuntimeError("Tokenizer does not have the required fixed ID count")
    special_token_ids = {
        name: int(backend.token_to_id(token)) for name, token in SPECIAL_TOKEN_ITEMS
    }
    for expected_id, (name, _) in enumerate(SPECIAL_TOKEN_ITEMS):
        if special_token_ids[name] != expected_id:
            raise RuntimeError("Tokenizer backend assigned unstable special-token IDs")
    for item in ATOMIC_TOKENS:
        item_ids = backend.encode(item, add_special_tokens=False).ids
        if (
            len(item_ids) != 1
            or backend.decode(item_ids, skip_special_tokens=False) != item
        ):
            raise RuntimeError(f"Failed to make arithmetic token {item!r} atomic")

    backend.save(str(tokenizer_path), pretty=True)
    metadata = {
        "format_version": 1,
        "frozen": True,
        "tokenizer_type": "byte_level_bpe",
        "byte_fallback": True,
        "unicode_normalizer": None,
        "vocab_size": VOCAB_SIZE,
        "base_vocabulary_size": current_size,
        "reserved_token_count": reserved_count,
        "min_frequency": min_frequency,
        "training_text_count": text_count,
        "training_text_sha256": corpus_digest.hexdigest(),
        "special_tokens": dict(SPECIAL_TOKEN_ITEMS),
        "special_token_ids": special_token_ids,
        "extension_slots": {
            "start_id": len(SPECIAL_TOKEN_ITEMS),
            "count": EXTENSION_SLOT_COUNT,
            "end_id_exclusive": len(SPECIAL_TOKEN_ITEMS) + EXTENSION_SLOT_COUNT,
            "token_format": "<|extension_NNN|>",
            "allocation": "external_versioned_registry",
        },
        "atomic_tokens": list(ATOMIC_TOKENS),
        "atomic_token_ids": {
            item: backend.encode(item, add_special_tokens=False).ids[0]
            for item in ATOMIC_TOKENS
        },
        "tokenizers_version": tokenizers.__version__,
        "tokenizer_sha256": _sha256_file(tokenizer_path),
    }
    _json_dump(metadata_path, metadata)
    return ScratchTokenizer.load(destination)


def iter_jsonl_training_texts(paths: Sequence[str | Path]) -> Iterable[str]:
    """Yield text fields from audited document/dialogue JSONL files."""

    for raw_path in sorted((Path(path) for path in paths), key=lambda path: str(path)):
        with raw_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {raw_path}:{line_number}: {error}"
                    ) from error
                for required in ("source", "license", "provenance"):
                    if required not in record or record[required] in (None, "", {}):
                        raise ValueError(
                            f"Missing required {required!r} in {raw_path}:{line_number}"
                        )
                record_type = record.get("type")
                if record_type == "document" and isinstance(record.get("text"), str):
                    yield record["text"]
                elif record_type == "dialogue" and isinstance(
                    record.get("messages"), list
                ):
                    for message in record["messages"]:
                        _, content = _coerce_turn(message)
                        yield content
                else:
                    raise ValueError(
                        f"Expected a document or dialogue record in {raw_path}:{line_number}"
                    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="Audited JSONL file(s); normally only prepared train.jsonl",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Tokenizer artifact directory"
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    tokenizer = build_scratch_tokenizer(
        iter_jsonl_training_texts(args.input),
        args.output,
        min_frequency=args.min_frequency,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "artifact": str(tokenizer.artifact_dir),
                "vocab_size": tokenizer.vocab_size,
                "sha256": tokenizer.metadata["tokenizer_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
