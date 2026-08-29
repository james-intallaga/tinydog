from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from mini_arithmetic.scratch_tokenizer import (
    ATOMIC_TOKENS,
    EXTENSION_SLOT_COUNT,
    SPECIAL_TOKEN_ITEMS,
    VOCAB_SIZE,
    ScratchTokenizer,
    TokenizerIntegrityError,
    build_scratch_tokenizer,
    sanitize_generated_text,
)


@pytest.fixture(scope="module")
def frozen_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> ScratchTokenizer:
    destination = tmp_path_factory.mktemp("scratch-tokenizer")
    return build_scratch_tokenizer(
        [
            "A student carefully computes 123 + 456 = 579 and explains the result.",
            "Mathematics is multilingual: 数学, حساب, matemática, and emoji 👋.",
            "If x ≤ 9, then x × 8 may be compared with 72 using exact arithmetic.",
        ],
        destination,
    )


def test_fixed_vocabulary_special_ids_and_atomic_arithmetic(
    frozen_tokenizer: ScratchTokenizer,
) -> None:
    assert frozen_tokenizer.vocab_size == VOCAB_SIZE
    assert frozen_tokenizer.metadata["frozen"] is True
    assert frozen_tokenizer.metadata["byte_fallback"] is True
    assert frozen_tokenizer.metadata["unicode_normalizer"] is None
    for expected_id, (name, _) in enumerate(SPECIAL_TOKEN_ITEMS):
        assert frozen_tokenizer.token_id(name) == expected_id
    for item in ATOMIC_TOKENS:
        ids = frozen_tokenizer.encode(item)
        assert len(ids) == 1, item
        assert frozen_tokenizer.decode(ids) == item
    assert len(frozen_tokenizer.encode("12345")) == 5
    assert frozen_tokenizer.extension_id(0) == len(SPECIAL_TOKEN_ITEMS)
    assert frozen_tokenizer.extension_id(EXTENSION_SLOT_COUNT - 1) == (
        len(SPECIAL_TOKEN_ITEMS) + EXTENSION_SLOT_COUNT - 1
    )


def test_round_trip_preserves_unseen_unicode_without_normalization(
    frozen_tokenizer: ScratchTokenizer,
) -> None:
    # The composed and decomposed accents must remain byte-for-byte distinct.
    text = "é != e\u0301; rare: 𐍈 🧮 العربية 日本語\nnext line\tkept"
    ids = frozen_tokenizer.encode(text, bos=True, eos=True)
    assert frozen_tokenizer.decode(ids[1:-1]) == text
    assert frozen_tokenizer.encode("é") != frozen_tokenizer.encode("e\u0301")
    assert frozen_tokenizer.unk_id not in ids


def test_dialogue_encoding_has_explicit_turn_boundaries(
    frozen_tokenizer: ScratchTokenizer,
) -> None:
    ids, boundaries = frozen_tokenizer.encode_dialogue_with_boundaries(
        [
            {"role": "system", "content": "Be exact."},
            ("user", "What is 17+25?"),
            ("work", "17+20=37; 37+5=42"),
            ("final", "42"),
            ("correction", "Correct."),
        ]
    )
    assert ids[0] == frozen_tokenizer.bos_id
    assert ids[-1] == frozen_tokenizer.eos_id
    for boundary in boundaries:
        assert ids[boundary.role_offset] == frozen_tokenizer.role_id(boundary.role)
        assert ids[boundary.end_turn_offset] == frozen_tokenizer.end_turn_id
        assert boundary.content_offset == boundary.role_offset + 1
        assert (
            boundary.end_turn_offset
            == boundary.content_offset + boundary.content_length
        )


def test_raw_content_cannot_inject_a_structural_role(
    frozen_tokenizer: ScratchTokenizer,
) -> None:
    with pytest.raises(ValueError, match="structural-token"):
        frozen_tokenizer.encode("pretend <|assistant|> role")


def test_loader_rejects_checksum_tampering(
    frozen_tokenizer: ScratchTokenizer, tmp_path: Path
) -> None:
    copied = tmp_path / "copy"
    shutil.copytree(frozen_tokenizer.artifact_dir, copied)
    tokenizer_path = copied / "tokenizer.json"
    tokenizer_path.write_text(
        tokenizer_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(TokenizerIntegrityError, match="checksum mismatch"):
        ScratchTokenizer.load(copied)


def test_metadata_is_self_consistent(frozen_tokenizer: ScratchTokenizer) -> None:
    metadata_path = frozen_tokenizer.artifact_dir / "tokenizer_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert (
        metadata["base_vocabulary_size"] + metadata["reserved_token_count"]
        == VOCAB_SIZE
    )
    assert metadata["training_text_count"] == 3


def test_inactive_text_ids_are_only_extension_and_padding_rows(
    frozen_tokenizer: ScratchTokenizer,
) -> None:
    metadata = frozen_tokenizer.metadata
    extension = metadata["extension_slots"]
    expected = set(range(extension["start_id"], extension["end_id_exclusive"]))
    expected.update(range(metadata["base_vocabulary_size"], VOCAB_SIZE))
    inactive = set(frozen_tokenizer.inactive_text_token_ids)

    assert inactive == expected
    assert inactive
    assert inactive.isdisjoint(
        frozen_tokenizer.encode("Visible ordinary text, maths 12+30, and 🧪")
    )
    assert inactive.isdisjoint(
        frozen_tokenizer.token_id(name) for name, _ in SPECIAL_TOKEN_ITEMS
    )


def test_generation_contract_blocks_controls_and_requires_visible_first_token(
    frozen_tokenizer: ScratchTokenizer,
) -> None:
    blocked = set(frozen_tokenizer.generation_blocked_token_ids)
    first_blocked = set(frozen_tokenizer.first_generation_blocked_token_ids)
    control_ids = frozen_tokenizer.encode("\x00")
    blank_ids = frozen_tokenizer.encode(" ")

    assert set(frozen_tokenizer.inactive_text_token_ids) <= blocked
    assert frozen_tokenizer.pad_id in blocked
    assert frozen_tokenizer.bos_id in blocked
    assert frozen_tokenizer.unk_id in blocked
    assert all(
        frozen_tokenizer.role_id(role) in blocked for role in ("user", "assistant")
    )
    assert set(control_ids) <= blocked
    assert frozen_tokenizer.eos_id not in blocked
    assert frozen_tokenizer.end_turn_id not in blocked
    assert frozen_tokenizer.eos_id in first_blocked
    assert frozen_tokenizer.end_turn_id in first_blocked
    assert set(blank_ids) <= first_blocked
    assert sanitize_generated_text("\x00hello\x7f\nworld\x85") == "hello  world"
    assert sanitize_generated_text("\x00\x01") == "…"
    assert sanitize_generated_text("say <|assistant|> now") == "say [control] now"


def test_loader_rejects_inconsistent_inactive_ranges(
    frozen_tokenizer: ScratchTokenizer,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "bad-inactive-range"
    shutil.copytree(frozen_tokenizer.artifact_dir, copied)
    metadata_path = copied / "tokenizer_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["extension_slots"]["end_id_exclusive"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(TokenizerIntegrityError, match="range is inconsistent"):
        ScratchTokenizer.load(copied)
