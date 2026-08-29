from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from mini_arithmetic.prepare_data import PreparationConfig, prepare_corpus, sha256_file
from mini_arithmetic.scratch_tokenizer import build_scratch_tokenizer


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _document(
    record_id: str, text: str, *, source: str = "fixture"
) -> dict[str, object]:
    return {
        "type": "document",
        "id": record_id,
        "text": text,
        "source": source,
        "license": "CC0-1.0",
        "provenance": {"collection": "unit-test", "row": record_id},
    }


def _dialogue(record_id: str) -> dict[str, object]:
    return {
        "type": "dialogue",
        "id": record_id,
        "messages": [
            {"role": "system", "content": "Explain arithmetic exactly and clearly."},
            {"role": "user", "content": "What is 19 + 23?"},
            {
                "role": "assistant",
                "content": "I combine 19 + 20 + 3, so the final answer is 42.",
            },
            {
                "role": "teacher",
                "content": "Correct; the intermediate decomposition is valid.",
            },
        ],
        "source": "human-tutor-transcript",
        "license": "CC-BY-4.0",
        "provenance": {"session": "fixture-session", "consent": True},
    }


def _unique_text(index: int) -> str:
    words = [
        hashlib.sha256(f"{index}:{part}".encode()).hexdigest()[:12]
        for part in range(12)
    ]
    return "Audited educational record " + " ".join(words) + "."


def _all_output_bytes(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_auditing_filtering_dedup_and_hash_disjoint_splits(tmp_path: Path) -> None:
    base_words = [f"conceptword{index}" for index in range(80)]
    near_words = list(base_words)
    near_words[40] = "onechangedconcept"
    records = [_document(f"unique-{index}", _unique_text(index)) for index in range(60)]
    records.extend(
        [
            _document(
                "control",
                "A control\u0000character is cleaned while useful maths 2+2=4 remains.",
            ),
            _dialogue("dialogue"),
            _document(
                "exact-a",
                "This exact duplicate explains why seven plus five equals twelve.",
            ),
            _document(
                "exact-b",
                "This exact duplicate explains why seven plus five equals twelve.",
                source="second-source",
            ),
            _document("near-a", " ".join(base_words)),
            _document("near-b", " ".join(near_words), source="second-source"),
            {
                "type": "document",
                "id": "missing-provenance",
                "text": "This otherwise acceptable record lacks its required provenance field.",
                "source": "fixture",
                "license": "CC0-1.0",
            },
            _document("template", "boilerplate line\n" * 8),
        ]
    )
    input_path = tmp_path / "raw.jsonl"
    _write_jsonl(input_path, records)
    output = tmp_path / "prepared"
    config = PreparationConfig(
        min_chars=10,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        near_duplicate_threshold=0.85,
        lsh_bands=16,
    )

    manifest = prepare_corpus([input_path], output, config=config)

    assert manifest["counts"]["rejected"] == 2
    assert manifest["rejection_reasons"] == {
        "missing_provenance": 1,
        "templated_repeated_lines": 1,
    }
    assert manifest["counts"]["exact_duplicates_removed"] == 1
    assert manifest["counts"]["near_duplicates_removed"] == 1
    assert all(
        manifest["counts"]["splits"][name] > 0
        for name in ("train", "validation", "test")
    )

    split_hashes: dict[str, set[str]] = {}
    cleaned_control = None
    for split in ("train", "validation", "test"):
        split_records = [
            json.loads(line)
            for line in (output / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        split_hashes[split] = {record["dedup_sha256"] for record in split_records}
        cleaned_control = next(
            (
                record["text"]
                for record in split_records
                if record["record_id"] == "control"
            ),
            cleaned_control,
        )
    assert split_hashes["train"].isdisjoint(split_hashes["validation"])
    assert split_hashes["train"].isdisjoint(split_hashes["test"])
    assert split_hashes["validation"].isdisjoint(split_hashes["test"])
    assert (
        cleaned_control
        == "A control character is cleaned while useful maths 2+2=4 remains."
    )

    dedup_log = [
        json.loads(line) for line in (output / "dedup.jsonl").read_text().splitlines()
    ]
    assert {entry["kind"] for entry in dedup_log} == {"exact", "near"}
    assert manifest["hash_disjoint"] is True

    first_run = _all_output_bytes(output)
    second_manifest = prepare_corpus(
        [input_path], output, config=config, overwrite=True
    )
    assert second_manifest == manifest
    assert _all_output_bytes(output) == first_run


def test_uint16_shards_preserve_record_and_dialogue_boundaries(tmp_path: Path) -> None:
    raw_path = tmp_path / "dialogues.jsonl"
    records = [
        _dialogue("dialogue-boundaries"),
        _document(
            "document-boundaries",
            "A standalone document proves that 12 * 12 = 144 exactly.",
        ),
    ]
    _write_jsonl(raw_path, records)
    tokenizer = build_scratch_tokenizer(
        [
            "Explain arithmetic exactly and clearly.",
            "What is 19 + 23? The final answer is 42.",
            "A standalone document proves multiplication exactly.",
        ],
        tmp_path / "tokenizer",
    )
    output = tmp_path / "prepared"
    config = PreparationConfig(
        min_chars=10,
        train_ratio=1.0,
        validation_ratio=0.0,
        test_ratio=0.0,
        shard_token_limit=30,
    )

    manifest = prepare_corpus([raw_path], output, config=config, tokenizer=tokenizer)

    assert manifest["tokenizer"]["vocab_size"] == 32_768
    found_dialogue = False
    for index_path in sorted((output / "shards").glob("*.idx.json")):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        binary_path = index_path.with_name(index_path.name.replace(".idx.json", ".bin"))
        shard = np.fromfile(binary_path, dtype="<u2")
        assert len(shard) == index["token_count"]
        for record_index in index["records"]:
            start = record_index["offset"]
            stop = start + record_index["length"]
            assert int(shard[start]) == tokenizer.bos_id
            assert int(shard[stop - 1]) == tokenizer.eos_id
            if record_index["record_id"] != "dialogue-boundaries":
                continue
            found_dialogue = True
            for boundary in record_index["dialogue_boundaries"]:
                assert int(shard[start + boundary["role_offset"]]) == tokenizer.role_id(
                    boundary["role"]
                )
                assert (
                    int(shard[start + boundary["end_turn_offset"]])
                    == tokenizer.end_turn_id
                )
    assert found_dialogue

    checksum_lines = (
        (output / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines()
    )
    for line in checksum_lines:
        checksum, relative = line.split("  ", maxsplit=1)
        assert sha256_file(output / relative) == checksum


def test_required_provenance_and_schema_failures_are_logged(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.jsonl"
    input_path.write_text(
        '{"type":"document","text":"long enough but unaudited","source":"x","license":"y"}\n'
        "{not json}\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    manifest = prepare_corpus(
        [input_path],
        output,
        config=PreparationConfig(
            min_chars=5,
            train_ratio=1.0,
            validation_ratio=0.0,
            test_ratio=0.0,
        ),
    )
    assert manifest["counts"]["accepted"] == 0
    assert manifest["rejection_reasons"] == {"invalid_json": 1, "missing_provenance": 1}
