from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mini_arithmetic.native_data import NativeShardDataset
from test_multimodal import tiny_multimodal_config


TOKENIZER_DIGEST = "a" * 64


def _write_dataset(
    root: Path,
    *,
    split: str = "train",
    consent: bool = True,
    with_video: bool = True,
) -> Path:
    arrays: dict[str, np.ndarray] = {
        "user_waveform": np.full((1, 24, 1), 0.1, dtype=np.float32),
        "model_waveform": np.full((1, 24, 1), 0.2, dtype=np.float32),
        "text_tokens": np.array([[1, 2, 3]], dtype=np.int32),
        "text_target_mask": np.ones((1, 3), dtype=np.float32),
        "text_timestamp": np.array(0.0005, dtype=np.float32),
    }
    if with_video:
        arrays |= {
            "frames": np.stack(
                (
                    np.full((8, 8, 3), 0.3, dtype=np.float32),
                    np.full((8, 8, 3), 0.4, dtype=np.float32),
                ),
                axis=0,
            )[None, ...],
            "frame_timestamps": np.array([0.0, 0.0004], dtype=np.float32),
            "frame_difference_scores": np.array([0.0, 0.5], dtype=np.float32),
        }
    shard = root / "record.npz"
    np.savez(shard, **arrays)
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "split": split,
        "tokenizer_sha256": TOKENIZER_DIGEST,
        "sample_rate": 24_000,
        "frame_size": 8,
        "records": [
            {
                "record_id": "record-1",
                "shard": shard.name,
                "sha256": digest,
                "source": "consenting local speaker",
                "license": "project-owned-test-fixture",
                "consent": consent,
                "provenance": {"capture": "unit-test"},
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_native_dataset_verifies_rights_checksum_and_tensor_contract(
    tmp_path: Path,
) -> None:
    path = _write_dataset(tmp_path)
    dataset = NativeShardDataset(
        path,
        split="train",
        config=tiny_multimodal_config(),
        tokenizer_sha256=TOKENIZER_DIGEST,
    )
    batch = dataset.load(0)
    assert len(dataset) == 1
    assert batch.user_waveform.shape == (1, 24, 1)
    assert batch.frames is not None and batch.frames.shape == (1, 2, 8, 8, 3)
    assert len(dataset.content_hashes) == 4
    assert dataset.shard_hashes == frozenset({dataset.records[0].sha256})


def test_native_dataset_rejects_missing_consent_and_split_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_dataset(tmp_path, consent=False)
    with pytest.raises(ValueError, match="affirmative consent"):
        NativeShardDataset(
            path,
            split="train",
            config=tiny_multimodal_config(),
            tokenizer_sha256=TOKENIZER_DIGEST,
        )

    path = _write_dataset(tmp_path)
    payload = json.loads(path.read_text())
    payload["records"][0]["license"] = "unknown"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="license is required"):
        NativeShardDataset(
            path,
            split="train",
            config=tiny_multimodal_config(),
            tokenizer_sha256=TOKENIZER_DIGEST,
        )
    path = _write_dataset(tmp_path, split="validation")
    with pytest.raises(ValueError, match="requested split"):
        NativeShardDataset(
            path,
            split="train",
            config=tiny_multimodal_config(),
            tokenizer_sha256=TOKENIZER_DIGEST,
        )


def test_native_dataset_rejects_tampering_traversal_and_bad_media_values(
    tmp_path: Path,
) -> None:
    path = _write_dataset(tmp_path)
    dataset = NativeShardDataset(
        path,
        split="train",
        config=tiny_multimodal_config(),
        tokenizer_sha256=TOKENIZER_DIGEST,
    )
    with dataset.records[0].path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum changed"):
        dataset.load(0)

    path = _write_dataset(tmp_path)
    payload = json.loads(path.read_text())
    payload["records"][0]["shard"] = "../outside.npz"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, FileNotFoundError), match="escapes|not found"):
        NativeShardDataset(
            path,
            split="train",
            config=tiny_multimodal_config(),
            tokenizer_sha256=TOKENIZER_DIGEST,
        )

    path = _write_dataset(tmp_path)
    shard = tmp_path / "record.npz"
    np.savez(
        shard,
        user_waveform=np.full((1, 24, 1), 2.0, dtype=np.float32),
        model_waveform=np.zeros((1, 24, 1), dtype=np.float32),
        text_tokens=np.array([[1, 2]], dtype=np.int32),
        text_target_mask=np.ones((1, 2), dtype=np.float32),
        text_timestamp=np.array(0.0, dtype=np.float32),
    )
    payload = json.loads(path.read_text())
    payload["records"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    dataset = NativeShardDataset(
        path,
        split="train",
        config=tiny_multimodal_config(),
        tokenizer_sha256=TOKENIZER_DIGEST,
    )
    with pytest.raises(ValueError, match="normalized"):
        dataset.load(0)
