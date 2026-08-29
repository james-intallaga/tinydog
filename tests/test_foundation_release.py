import hashlib
import json
from pathlib import Path

import pytest
import mlx.core as mx

from mini_arithmetic.foundation_release import (
    create_foundation_manifest,
    generate_owner_keypair,
    verify_foundation_manifest,
)


def _release_files(root: Path) -> None:
    mx.save_safetensors(str(root / "model.safetensors"), {"weight": mx.zeros((2, 2))})
    model_config = {"model_dim": 16}
    (root / "model.json").write_text(json.dumps(model_config))
    (root / "tokenizer.json").write_text(json.dumps({"version": "1.0"}))
    tokenizer_sha256 = hashlib.sha256(
        (root / "tokenizer.json").read_bytes()
    ).hexdigest()
    (root / "metadata.json").write_text(
        json.dumps({"tokenizer_sha256": tokenizer_sha256})
    )
    (root / "floaty.json").write_text(
        json.dumps(
            {
                "foundation_config": model_config,
                "rank": 4,
                "attachments": ["operator"],
            }
        )
    )


def _signed_release(tmp_path: Path):
    root = tmp_path / "release"
    root.mkdir()
    _release_files(root)
    private = tmp_path / "owner.key"
    public = tmp_path / "owner.pub"
    key_id = generate_owner_keypair(private, public)
    manifest = create_foundation_manifest(
        root,
        foundation_id="foundation-f1",
        weights="model.safetensors",
        model_config="model.json",
        tokenizer="tokenizer.json",
        tokenizer_metadata="metadata.json",
        floaty_schema="floaty.json",
        private_key=private,
    )
    return root, private, public, key_id, manifest


def test_owner_signed_foundation_round_trip(tmp_path: Path) -> None:
    root, _, public, key_id, manifest = _signed_release(tmp_path)
    verified = verify_foundation_manifest(manifest, public)
    assert verified.foundation_id == "foundation-f1"
    assert verified.owner_key_id == key_id
    assert verified.artifacts["weights"].path == root / "model.safetensors"
    assert (
        verified.artifacts["weights"].byte_count
        == (root / "model.safetensors").stat().st_size
    )
    assert len(verified.artifacts["weights"].sha256) == 64
    assert verified.artifacts["tokenizer_metadata"].path == root / "metadata.json"
    with pytest.raises(TypeError):
        verified.artifacts["extra"] = verified.artifacts["weights"]
    assert len(verified.manifest_sha256) == 64

    parsed = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(parsed, separators=(",", ":")))
    reformatted = verify_foundation_manifest(manifest, root.parent / "owner.pub")
    assert reformatted.manifest_sha256 == verified.manifest_sha256


def test_foundation_rejects_wrong_owner_and_tampered_artifact(tmp_path: Path) -> None:
    root, _, public, _, manifest = _signed_release(tmp_path)
    other_private = tmp_path / "other.key"
    other_public = tmp_path / "other.pub"
    generate_owner_keypair(other_private, other_public)
    with pytest.raises(ValueError, match="trusted owner"):
        verify_foundation_manifest(manifest, other_public)
    (root / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="safetensors|does not match"):
        verify_foundation_manifest(manifest, public)


def test_owner_key_and_manifest_never_overwrite(tmp_path: Path) -> None:
    root, private, public, _, _ = _signed_release(tmp_path)
    with pytest.raises(FileExistsError):
        generate_owner_keypair(private, public)
    with pytest.raises(FileExistsError):
        create_foundation_manifest(
            root,
            foundation_id="foundation-f1",
            weights="model.safetensors",
            model_config="model.json",
            tokenizer="tokenizer.json",
            tokenizer_metadata="metadata.json",
            floaty_schema="floaty.json",
            private_key=private,
        )


def test_foundation_rejects_artifacts_outside_release_root(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    _release_files(root)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    private = tmp_path / "owner.key"
    public = tmp_path / "owner.pub"
    generate_owner_keypair(private, public)
    with pytest.raises(ValueError, match="inside the foundation root"):
        create_foundation_manifest(
            root,
            foundation_id="foundation-f1",
            weights=outside,
            model_config="model.json",
            tokenizer="tokenizer.json",
            tokenizer_metadata="metadata.json",
            floaty_schema="floaty.json",
            private_key=private,
        )
