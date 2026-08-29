"""Deterministic, provenance-first preparation for scratch language-model data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .scratch_tokenizer import (
    ROLE_NAMES,
    ScratchTokenizer,
    contains_control_token_literal,
)


SPLIT_NAMES = ("train", "validation", "test")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_LONG_RUN = re.compile(r"(.)\1{31,}", flags=re.DOTALL)


@dataclass(frozen=True)
class PreparationConfig:
    """All decisions that affect deterministic filtering and partitioning."""

    min_chars: int = 20
    max_chars: int = 1_000_000
    train_ratio: float = 0.98
    validation_ratio: float = 0.01
    test_ratio: float = 0.01
    near_duplicate_threshold: float = 0.85
    shingle_size: int = 5
    minhash_permutations: int = 64
    lsh_bands: int = 8
    shard_token_limit: int = 4_000_000

    def validate(self) -> None:
        if self.min_chars < 1 or self.max_chars < self.min_chars:
            raise ValueError("Character limits are invalid")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError(
                "Train, validation, and test ratios must be non-negative and sum to 1"
            )
        if not 0.0 <= self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be between 0 and 1")
        if self.shingle_size < 1:
            raise ValueError("shingle_size must be positive")
        if self.minhash_permutations < 1 or self.lsh_bands < 1:
            raise ValueError("MinHash dimensions must be positive")
        if self.minhash_permutations % self.lsh_bands:
            raise ValueError("minhash_permutations must be divisible by lsh_bands")
        if self.shard_token_limit < 1:
            raise ValueError("shard_token_limit must be positive")


@dataclass(frozen=True)
class Rejection:
    input_path: str
    line: int
    reason: str
    record_id: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    """Remove unsafe controls without applying Unicode normalization."""

    if not isinstance(text, str):
        raise TypeError("Content must be a string")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARACTERS.sub(" ", text)
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned: list[str] = []
    blank_count = 0
    for line in lines:
        if line:
            blank_count = 0
            cleaned.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
    return "\n".join(cleaned).strip()


def _clean_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _clean_json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"Unsupported provenance value: {type(value).__name__}")


def _stable_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _nonempty_metadata(record: Mapping[str, Any], name: str) -> Any:
    if name not in record:
        raise ValueError(f"missing_{name}")
    value = _clean_json_value(record[name])
    if value in (None, "", [], {}):
        raise ValueError(f"missing_{name}")
    if name in {"source", "license"} and not isinstance(value, str):
        raise ValueError(f"invalid_{name}")
    return value


def _quality_reason(text: str, config: PreparationConfig) -> str | None:
    if len(text) < config.min_chars:
        return "too_short"
    if len(text) > config.max_chars:
        return "too_long"
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return "no_visible_content"
    if contains_control_token_literal(text):
        return "reserved_structural_token_literal"
    if text.count("\ufffd") / max(1, len(text)) > 0.01:
        return "excessive_replacement_characters"
    if _LONG_RUN.search(text):
        return "repeated_character_run"

    nonempty_lines = [
        re.sub(r"\s+", " ", line).strip().casefold() for line in text.splitlines()
    ]
    nonempty_lines = [line for line in nonempty_lines if line]
    if len(nonempty_lines) >= 4:
        frequencies = Counter(nonempty_lines)
        if max(frequencies.values()) / len(nonempty_lines) >= 0.5:
            return "templated_repeated_lines"
    if (
        len(nonempty_lines) >= 6
        and len(set(nonempty_lines)) / len(nonempty_lines) < 0.3
    ):
        return "templated_low_line_diversity"
    return None


def _canonical_content(record: Mapping[str, Any]) -> str:
    if record["type"] == "document":
        return str(record["text"])
    return _stable_json(
        [[message["role"], message["content"]] for message in record["messages"]]
    )


def _quality_text(record: Mapping[str, Any]) -> str:
    if record["type"] == "document":
        return str(record["text"])
    return "\n".join(str(message["content"]) for message in record["messages"])


def _normalize_for_dedup(text: str) -> str:
    # This changes only the comparison key; stored training text is untouched.
    return " ".join(text.casefold().split())


def _content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonicalize_record(
    raw: Mapping[str, Any], config: PreparationConfig
) -> dict[str, Any]:
    source = _nonempty_metadata(raw, "source")
    license_name = _nonempty_metadata(raw, "license")
    provenance = _nonempty_metadata(raw, "provenance")
    record_type = raw.get("type")
    if record_type not in {"document", "dialogue"}:
        raise ValueError("invalid_type")

    canonical: dict[str, Any] = {
        "type": record_type,
        "source": source,
        "license": license_name,
        "provenance": provenance,
    }
    if "metadata" in raw:
        metadata = _clean_json_value(raw["metadata"])
        if not isinstance(metadata, Mapping):
            raise ValueError("invalid_metadata")
        canonical["metadata"] = metadata

    if record_type == "document":
        if not isinstance(raw.get("text"), str):
            raise ValueError("invalid_document_text")
        canonical["text"] = clean_text(raw["text"])
    else:
        messages = raw.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("invalid_messages")
        cleaned_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError("invalid_message")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or role not in ROLE_NAMES:
                raise ValueError("invalid_role")
            if not isinstance(content, str):
                raise ValueError("invalid_message_content")
            cleaned_content = clean_text(content)
            if not cleaned_content:
                raise ValueError("empty_message")
            cleaned_messages.append({"role": role, "content": cleaned_content})
        canonical["messages"] = cleaned_messages

    quality_reason = _quality_reason(_quality_text(canonical), config)
    if quality_reason is not None:
        raise ValueError(quality_reason)
    content = _canonical_content(canonical)
    content_sha = _content_sha256(content)
    supplied_id = raw.get("id", raw.get("record_id"))
    if supplied_id is not None and (
        not isinstance(supplied_id, str) or not supplied_id.strip()
    ):
        raise ValueError("invalid_id")
    record_id = (
        supplied_id.strip() if isinstance(supplied_id, str) else f"sha256:{content_sha}"
    )
    canonical["record_id"] = record_id
    canonical["content_sha256"] = content_sha
    canonical["dedup_sha256"] = _content_sha256(_normalize_for_dedup(content))
    return canonical


def _shingles(text: str, width: int) -> frozenset[str]:
    tokens = _TOKEN_PATTERN.findall(_normalize_for_dedup(text))
    if not tokens:
        return frozenset({""})
    if len(tokens) < width:
        return frozenset(tokens)
    return frozenset(
        "\x1f".join(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )


def _minhash(shingles: frozenset[str], permutations: int) -> tuple[int, ...]:
    encoded = [shingle.encode("utf-8") for shingle in shingles]
    signature: list[int] = []
    for permutation in range(permutations):
        salt = permutation.to_bytes(4, "big")
        signature.append(
            min(
                int.from_bytes(
                    hashlib.blake2b(salt + item, digest_size=8).digest(), "big"
                )
                for item in encoded
            )
        )
    return tuple(signature)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


class StableMinHashLSH:
    """Small dependency-free, deterministic MinHash/LSH deduplication index."""

    def __init__(self, config: PreparationConfig) -> None:
        self.config = config
        self._rows = config.minhash_permutations // config.lsh_bands
        self._buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
        self._shingle_sets: list[frozenset[str]] = []
        self._signatures: list[tuple[int, ...]] = []

    def query(self, shingles: frozenset[str]) -> tuple[int | None, float]:
        signature = _minhash(shingles, self.config.minhash_permutations)
        candidates: set[int] = set()
        for band in range(self.config.lsh_bands):
            start = band * self._rows
            candidates.update(
                self._buckets.get((band, signature[start : start + self._rows]), ())
            )
        best_index: int | None = None
        best_score = -1.0
        for index in sorted(candidates):
            score = _jaccard(shingles, self._shingle_sets[index])
            if score > best_score:
                best_index, best_score = index, score
        if best_score >= self.config.near_duplicate_threshold:
            return best_index, best_score
        return None, max(0.0, best_score)

    def add(self, shingles: frozenset[str]) -> int:
        signature = _minhash(shingles, self.config.minhash_permutations)
        index = len(self._shingle_sets)
        self._shingle_sets.append(shingles)
        self._signatures.append(signature)
        for band in range(self.config.lsh_bands):
            start = band * self._rows
            self._buckets[(band, signature[start : start + self._rows])].append(index)
        return index


def _deduplicate(
    records: Sequence[dict[str, Any]], config: PreparationConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    exact: dict[str, int] = {}
    near_index = StableMinHashLSH(config)
    log: list[dict[str, Any]] = []
    for record in records:
        exact_key = str(record["content_sha256"])
        if exact_key in exact:
            kept = accepted[exact[exact_key]]
            log.append(
                {
                    "kind": "exact",
                    "kept_record_id": kept["record_id"],
                    "removed_record_id": record["record_id"],
                    "removed_source": record["source"],
                    "similarity": 1.0,
                }
            )
            continue
        shingles = _shingles(_canonical_content(record), config.shingle_size)
        near_match, score = near_index.query(shingles)
        if near_match is not None:
            kept = accepted[near_match]
            log.append(
                {
                    "kind": "near",
                    "kept_record_id": kept["record_id"],
                    "removed_record_id": record["record_id"],
                    "removed_source": record["source"],
                    "similarity": round(score, 8),
                }
            )
            continue
        accepted_index = len(accepted)
        accepted.append(record)
        exact[exact_key] = accepted_index
        lsh_index = near_index.add(shingles)
        if lsh_index != accepted_index:
            raise RuntimeError("Deduplication indexes diverged")
    return accepted, log


def split_for_hash(content_hash: str, config: PreparationConfig) -> str:
    point = int(content_hash[:16], 16) / 2**64
    if point < config.train_ratio:
        return "train"
    if point < config.train_ratio + config.validation_ratio:
        return "validation"
    return "test"


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(_stable_json(record) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")


def _tokenize_record(
    record: Mapping[str, Any], tokenizer: ScratchTokenizer
) -> tuple[list[int], list[dict[str, int | str]]]:
    if record["type"] == "document":
        return tokenizer.encode(str(record["text"]), bos=True, eos=True), []
    ids, boundaries = tokenizer.encode_dialogue_with_boundaries(record["messages"])
    return ids, [boundary.to_dict() for boundary in boundaries]


def write_token_shards(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer: ScratchTokenizer,
    output_dir: str | Path,
    *,
    shard_token_limit: int,
) -> list[Path]:
    """Write little-endian uint16 shards plus record/role boundary indexes."""

    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("Tokenizer IDs do not fit in uint16")
    shard_dir = Path(output_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for split in SPLIT_NAMES:
        shard_number = 0
        tokens: list[int] = []
        index_records: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal shard_number, tokens, index_records
            stem = f"{split}-{shard_number:05d}"
            binary_path = shard_dir / f"{stem}.bin"
            index_path = shard_dir / f"{stem}.idx.json"
            np.asarray(tokens, dtype="<u2").tofile(binary_path)
            index_payload = {
                "format_version": 1,
                "dtype": "uint16-le",
                "split": split,
                "shard": shard_number,
                "token_count": len(tokens),
                "tokenizer_sha256": tokenizer.metadata["tokenizer_sha256"],
                "records": index_records,
            }
            index_path.write_text(
                _stable_json(index_payload, pretty=True), encoding="utf-8"
            )
            written.extend((binary_path, index_path))
            shard_number += 1
            tokens = []
            index_records = []

        for record in records_by_split.get(split, ()):
            record_tokens, boundaries = _tokenize_record(record, tokenizer)
            if tokens and len(tokens) + len(record_tokens) > shard_token_limit:
                flush()
            offset = len(tokens)
            if any(
                token_id < 0 or token_id >= tokenizer.vocab_size
                for token_id in record_tokens
            ):
                raise RuntimeError("Tokenizer emitted an out-of-range ID")
            tokens.extend(record_tokens)
            index_records.append(
                {
                    "record_id": record["record_id"],
                    "record_type": record["type"],
                    "offset": offset,
                    "length": len(record_tokens),
                    "dialogue_boundaries": boundaries,
                }
            )
        # An empty split still gets an explicit zero-token shard, making the
        # absence visible rather than looking like an interrupted preparation.
        flush()
    return written


def _prepare_destination(destination: Path, *, overwrite: bool) -> None:
    known_files = [destination / f"{split}.jsonl" for split in SPLIT_NAMES] + [
        destination / "dedup.jsonl",
        destination / "rejections.jsonl",
        destination / "manifest.json",
        destination / "CHECKSUMS.sha256",
    ]
    shard_dir = destination / "shards"
    known_files.extend(shard_dir.glob("*.bin") if shard_dir.exists() else ())
    known_files.extend(shard_dir.glob("*.idx.json") if shard_dir.exists() else ())
    existing = [path for path in known_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Prepared artifacts already exist in {destination}")
    if overwrite:
        for path in existing:
            path.unlink()
    destination.mkdir(parents=True, exist_ok=True)


def prepare_corpus(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    config: PreparationConfig | None = None,
    tokenizer: ScratchTokenizer | str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Audit, clean, deduplicate, split, and optionally tokenize JSONL data."""

    active_config = config or PreparationConfig()
    active_config.validate()
    inputs = sorted((Path(path) for path in input_paths), key=lambda path: str(path))
    if not inputs:
        raise ValueError("At least one input JSONL path is required")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    destination = Path(output_dir)
    _prepare_destination(destination, overwrite=overwrite)
    canonical_records: list[dict[str, Any]] = []
    rejections: list[Rejection] = []
    input_line_count = 0
    input_manifest: list[dict[str, Any]] = []
    for path in inputs:
        input_manifest.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                input_line_count += 1
                raw: Any = None
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, Mapping):
                        raise ValueError("record_is_not_an_object")
                    canonical_records.append(_canonicalize_record(raw, active_config))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    if isinstance(error, json.JSONDecodeError):
                        reason = "invalid_json"
                    else:
                        reason = str(error) or type(error).__name__
                    rejections.append(
                        Rejection(
                            input_path=str(path),
                            line=line_number,
                            reason=reason,
                            record_id=(
                                str(raw.get("id"))
                                if isinstance(raw, Mapping)
                                and raw.get("id") is not None
                                else None
                            ),
                            source=(
                                str(raw.get("source"))
                                if isinstance(raw, Mapping)
                                and raw.get("source") is not None
                                else None
                            ),
                        )
                    )

    # Sorting before choosing a duplicate winner makes output independent of
    # input-file order and JSONL record order.
    canonical_records.sort(
        key=lambda record: (
            str(record["dedup_sha256"]),
            str(record["source"]),
            str(record["record_id"]),
        )
    )
    accepted, dedup_log = _deduplicate(canonical_records, active_config)
    records_by_split: dict[str, list[dict[str, Any]]] = {
        name: [] for name in SPLIT_NAMES
    }
    for record in accepted:
        split = split_for_hash(str(record["dedup_sha256"]), active_config)
        output_record = dict(record)
        output_record["split"] = split
        records_by_split[split].append(output_record)
    for records in records_by_split.values():
        records.sort(
            key=lambda record: (str(record["content_sha256"]), str(record["record_id"]))
        )

    artifact_paths: list[Path] = []
    for split in SPLIT_NAMES:
        split_path = destination / f"{split}.jsonl"
        _write_jsonl(split_path, records_by_split[split])
        artifact_paths.append(split_path)
    dedup_path = destination / "dedup.jsonl"
    rejection_path = destination / "rejections.jsonl"
    _write_jsonl(dedup_path, dedup_log)
    _write_jsonl(rejection_path, (rejection.to_dict() for rejection in rejections))
    artifact_paths.extend((dedup_path, rejection_path))

    loaded_tokenizer: ScratchTokenizer | None
    if tokenizer is None:
        loaded_tokenizer = None
    elif isinstance(tokenizer, ScratchTokenizer):
        loaded_tokenizer = tokenizer
    else:
        loaded_tokenizer = ScratchTokenizer.load(tokenizer)
    if loaded_tokenizer is not None:
        artifact_paths.extend(
            write_token_shards(
                records_by_split,
                loaded_tokenizer,
                destination / "shards",
                shard_token_limit=active_config.shard_token_limit,
            )
        )

    exact_count = sum(item["kind"] == "exact" for item in dedup_log)
    near_count = sum(item["kind"] == "near" for item in dedup_log)
    output_inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(
        artifact_paths, key=lambda item: str(item.relative_to(destination))
    ):
        relative = str(path.relative_to(destination))
        entry: dict[str, Any] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.name in {f"{split}.jsonl" for split in SPLIT_NAMES}:
            entry["records"] = len(records_by_split[path.stem])
        if path.suffix == ".bin":
            entry["tokens"] = path.stat().st_size // 2
        output_inventory[relative] = entry

    rejection_reasons = Counter(rejection.reason for rejection in rejections)
    source_counts = Counter(str(record["source"]) for record in accepted)
    license_counts = Counter(str(record["license"]) for record in accepted)
    split_hashes = {
        split: hashlib.sha256(
            "".join(
                str(record["dedup_sha256"]) for record in records_by_split[split]
            ).encode("ascii")
        ).hexdigest()
        for split in SPLIT_NAMES
    }
    manifest: dict[str, Any] = {
        "format_version": 1,
        "deterministic": True,
        "inputs": input_manifest,
        "config": asdict(active_config),
        "counts": {
            "input_records": input_line_count,
            "valid_before_dedup": len(canonical_records),
            "accepted": len(accepted),
            "rejected": len(rejections),
            "exact_duplicates_removed": exact_count,
            "near_duplicates_removed": near_count,
            "splits": {split: len(records_by_split[split]) for split in SPLIT_NAMES},
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "sources": dict(sorted(source_counts.items())),
        "licenses": dict(sorted(license_counts.items())),
        "deduplication": {
            "exact_key": "sha256(cleaned_canonical_content_utf8)",
            "near_method": "stable_minhash_lsh_then_exact_jaccard",
            "global_before_split": True,
        },
        "split_method": "first_64_bits_of_dedup_sha256",
        "split_fingerprint_sha256": split_hashes,
        "hash_disjoint": True,
        "tokenizer": (
            {
                "path": str(loaded_tokenizer.artifact_dir),
                "sha256": loaded_tokenizer.metadata["tokenizer_sha256"],
                "vocab_size": loaded_tokenizer.vocab_size,
            }
            if loaded_tokenizer is not None
            else None
        ),
        "outputs": output_inventory,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(_stable_json(manifest, pretty=True), encoding="utf-8")
    checksum_paths = artifact_paths + [manifest_path]
    checksums_path = destination / "CHECKSUMS.sha256"
    checksums_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(destination)}\n"
            for path in sorted(
                checksum_paths, key=lambda item: str(item.relative_to(destination))
            )
        ),
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, help="Frozen tokenizer directory; enables shards"
    )
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--shard-token-limit", type=int, default=4_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = PreparationConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        near_duplicate_threshold=args.near_duplicate_threshold,
        shard_token_limit=args.shard_token_limit,
    )
    manifest = prepare_corpus(
        args.input,
        args.output,
        config=config,
        tokenizer=args.tokenizer,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
