import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest

from mini_arithmetic.community_release import (
    advance_trusted_community_state,
    create_community_release_manifest,
    verify_community_release,
)
from mini_arithmetic.community_updates import (
    CommunityContract,
    TensorSpec,
    write_floaty_safetensors,
)
from mini_arithmetic.foundation_release import (
    create_foundation_manifest,
    generate_owner_keypair,
    verify_foundation_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _release_fixture(tmp_path: Path):
    private = tmp_path / "owner.key"
    public = tmp_path / "owner.pub"
    generate_owner_keypair(private, public)

    foundation_root = tmp_path / "foundation"
    foundation_root.mkdir()
    mx.save_safetensors(
        str(foundation_root / "model.safetensors"),
        {"weight": mx.zeros((2, 2))},
    )
    model_config = {"model_dim": 8, "layer_schedule": ["attention"]}
    _json(foundation_root / "model.json", model_config)
    tokenizer = foundation_root / "tokenizer.json"
    _json(tokenizer, {"model": {"type": "BPE"}, "version": "1.0"})
    _json(
        foundation_root / "metadata.json",
        {"tokenizer_sha256": _sha256(tokenizer)},
    )
    spec = TensorSpec("floaty.blocks.0.operator.down.weight", (1, 1), "F32")
    schema = {
        "schema": "tinydog.foundation-floaty.config.v1",
        "foundation_config": model_config,
        "tensors": [spec.to_dict()],
    }
    _json(foundation_root / "floaty.json", schema)
    foundation_manifest = create_foundation_manifest(
        foundation_root,
        foundation_id="foundation-1",
        weights="model.safetensors",
        model_config="model.json",
        tokenizer="tokenizer.json",
        tokenizer_metadata="metadata.json",
        floaty_schema="floaty.json",
        private_key=private,
    )
    foundation = verify_foundation_manifest(foundation_manifest, public)

    active_root = tmp_path / "community-0"
    active_root.mkdir()
    active_weights = active_root / "floaty.safetensors"
    write_floaty_safetensors(
        active_weights,
        {spec.name: np.zeros(spec.shape, dtype=np.float32)},
        (spec,),
    )
    contract = CommunityContract.from_floaty_schema(
        schema["tensors"],
        foundation_sha256=foundation.foundation_sha256,
        community_version="community-0",
        community_weights_sha256=_sha256(active_weights),
        architecture_sha256=foundation.architecture_sha256,
        tokenizer_sha256=foundation.tokenizer_sha256,
        training_recipe_sha256=hashlib.sha256(b"recipe-1").hexdigest(),
    )
    _json(active_root / "contract.json", contract.to_dict())
    return private, public, foundation_manifest, active_root, contract


def _sign_and_verify(
    root: Path,
    *,
    private: Path,
    public: Path,
    foundation_manifest: Path,
    sequence: int = 0,
    previous: str | None = None,
):
    manifest = create_community_release_manifest(
        root,
        sequence=sequence,
        previous_release_sha256=previous,
        foundation_manifest=foundation_manifest,
        trusted_public_key=public,
        private_key=private,
    )
    return verify_community_release(
        manifest,
        public,
        foundation_manifest=foundation_manifest,
    )


def test_owner_signed_community_release_and_trust_state(tmp_path: Path) -> None:
    private, public, foundation_manifest, root, contract = _release_fixture(tmp_path)
    release = _sign_and_verify(
        root,
        private=private,
        public=public,
        foundation_manifest=foundation_manifest,
    )
    assert release.sequence == 0
    assert release.community_version == contract.community_version
    assert release.contract == contract
    assert release.artifacts["weights"].sha256 == contract.community_weights_sha256

    state = tmp_path / "trusted-community.json"
    assert advance_trusted_community_state(state, release) == state
    assert advance_trusted_community_state(state, release) == state
    assert json.loads(state.read_text())["release_sha256"] == release.release_sha256
    with pytest.raises(ValueError, match="expected chain"):
        verify_community_release(
            release.manifest_path,
            public,
            foundation_manifest=foundation_manifest,
            expected_previous_release_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="non-negative"):
        verify_community_release(
            release.manifest_path,
            public,
            foundation_manifest=foundation_manifest,
            minimum_sequence=-1,
        )


def test_release_chain_rejects_rollback_and_fork(tmp_path: Path) -> None:
    private, public, foundation_manifest, root0, _ = _release_fixture(tmp_path)
    release0 = _sign_and_verify(
        root0,
        private=private,
        public=public,
        foundation_manifest=foundation_manifest,
    )
    root1 = tmp_path / "community-1"
    root1.mkdir()
    for name in ("contract.json", "floaty.safetensors"):
        (root1 / name).write_bytes((root0 / name).read_bytes())
    release1 = _sign_and_verify(
        root1,
        private=private,
        public=public,
        foundation_manifest=foundation_manifest,
        sequence=1,
        previous=release0.release_sha256,
    )
    state = tmp_path / "state.json"
    advance_trusted_community_state(state, release0)
    advance_trusted_community_state(state, release1)
    with pytest.raises(ValueError, match="rollback|skips"):
        advance_trusted_community_state(state, release0)

    root2 = tmp_path / "community-fork"
    root2.mkdir()
    for name in ("contract.json", "floaty.safetensors"):
        (root2 / name).write_bytes((root0 / name).read_bytes())
    fork = _sign_and_verify(
        root2,
        private=private,
        public=public,
        foundation_manifest=foundation_manifest,
        sequence=2,
        previous="0" * 64,
    )
    with pytest.raises(ValueError, match="forks"):
        advance_trusted_community_state(state, fork)


def test_release_rejects_wrong_owner_tampering_and_contract_drift(
    tmp_path: Path,
) -> None:
    private, public, foundation_manifest, root, contract = _release_fixture(tmp_path)
    other_private = tmp_path / "other.key"
    other_public = tmp_path / "other.pub"
    generate_owner_keypair(other_private, other_public)
    with pytest.raises(ValueError, match="signing key"):
        create_community_release_manifest(
            root,
            sequence=0,
            foundation_manifest=foundation_manifest,
            trusted_public_key=public,
            private_key=other_private,
        )

    release = _sign_and_verify(
        root,
        private=private,
        public=public,
        foundation_manifest=foundation_manifest,
    )
    with pytest.raises(ValueError, match="trusted owner"):
        verify_community_release(
            release.manifest_path,
            other_public,
            foundation_manifest=foundation_manifest,
        )
    (root / "floaty.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="match|safetensors|invalid"):
        verify_community_release(
            release.manifest_path,
            public,
            foundation_manifest=foundation_manifest,
        )

    drift_root = tmp_path / "drift"
    drift_root.mkdir()
    spec = contract.tensor_specs[0]
    write_floaty_safetensors(
        drift_root / "floaty.safetensors",
        {spec.name: np.zeros(spec.shape, dtype=np.float32)},
        contract.tensor_specs,
    )
    drift = CommunityContract(
        foundation_sha256="f" * 64,
        community_version="community-drift",
        community_weights_sha256=_sha256(drift_root / "floaty.safetensors"),
        architecture_sha256=contract.architecture_sha256,
        tokenizer_sha256=contract.tokenizer_sha256,
        training_recipe_sha256=contract.training_recipe_sha256,
        tensor_specs=contract.tensor_specs,
    )
    _json(drift_root / "contract.json", drift.to_dict())
    with pytest.raises(ValueError, match="foundation weights"):
        create_community_release_manifest(
            drift_root,
            sequence=0,
            foundation_manifest=foundation_manifest,
            trusted_public_key=public,
            private_key=private,
        )


def test_release_identity_is_canonical_and_symlink_is_rejected(tmp_path: Path) -> None:
    private, public, foundation_manifest, root, _ = _release_fixture(tmp_path)
    release = _sign_and_verify(
        root,
        private=private,
        public=public,
        foundation_manifest=foundation_manifest,
    )
    parsed = json.loads(release.manifest_path.read_text())
    release.manifest_path.write_text(json.dumps(parsed, separators=(",", ":")))
    reformatted = verify_community_release(
        release.manifest_path,
        public,
        foundation_manifest=foundation_manifest,
    )
    assert reformatted.release_sha256 == release.release_sha256
    with pytest.raises(FileExistsError):
        create_community_release_manifest(
            root,
            sequence=0,
            foundation_manifest=foundation_manifest,
            trusted_public_key=public,
            private_key=private,
        )
    link = tmp_path / "community-link.json"
    link.symlink_to(release.manifest_path)
    with pytest.raises(ValueError, match="symbolic link"):
        verify_community_release(
            link,
            public,
            foundation_manifest=foundation_manifest,
        )

    state_target = tmp_path / "real-state.json"
    state_target.write_text("{}")
    state_link = tmp_path / "state-link.json"
    state_link.symlink_to(state_target)
    with pytest.raises(ValueError, match="symbolic link"):
        advance_trusted_community_state(state_link, release)


def test_trust_state_serializes_competing_chain_updates(tmp_path: Path) -> None:
    private, public, foundation_manifest, root0, _ = _release_fixture(tmp_path)
    release0 = _sign_and_verify(
        root0,
        private=private,
        public=public,
        foundation_manifest=foundation_manifest,
    )
    candidates = []
    for suffix in ("a", "b"):
        root = tmp_path / f"community-1-{suffix}"
        root.mkdir()
        for name in ("contract.json", "floaty.safetensors"):
            (root / name).write_bytes((root0 / name).read_bytes())
        candidates.append(
            _sign_and_verify(
                root,
                private=private,
                public=public,
                foundation_manifest=foundation_manifest,
                sequence=1,
                previous=release0.release_sha256,
            )
        )
    state = tmp_path / "concurrent-state.json"
    advance_trusted_community_state(state, release0)

    def advance(release):
        try:
            advance_trusted_community_state(state, release)
            return "accepted"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(advance, candidates))
    assert sorted(outcomes) == ["accepted", "rejected"]
