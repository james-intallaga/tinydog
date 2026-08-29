from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mini_arithmetic.hybrid_config import CONV, HybridConfig  # noqa: E402
from mini_arithmetic.portable_bundle import (  # noqa: E402
    MANIFEST_FILENAME,
    PORTABLE_STATE_FORMAT,
    PortableBundle,
    PortableBundleError,
    create_portable_bundle,
    load_portable_bundle,
    publish_conversational_foundation_bundle,
)
from mini_arithmetic.scratch_tokenizer import build_scratch_tokenizer  # noqa: E402
from mini_arithmetic.torch_hybrid import HybridLM  # noqa: E402


def _config_file(tmp_path: Path) -> tuple[Path, HybridConfig]:
    config = HybridConfig(
        vocab_size=32_768,
        context_length=16,
        model_dim=8,
        ffn_dim=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        conv_kernel_size=2,
        layer_schedule=(CONV,),
    )
    payload = asdict(config)
    payload["layer_schedule"] = list(config.layer_schedule)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, config


def _rewrite_weights(
    bundle: PortableBundle,
    state: dict[str, torch.Tensor],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    from safetensors.torch import save_file

    weights_path = bundle.weights_path
    save_file(
        {name: value.contiguous() for name, value in state.items()},
        str(weights_path),
        metadata=metadata,
    )
    _refresh_weights_checksum(bundle)


def _refresh_weights_checksum(bundle: PortableBundle) -> None:
    weights_path = bundle.weights_path
    digest = hashlib.sha256()
    with weights_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest_path = bundle.root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["weights_sha256"] = digest.hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )


def _new_bundle(tmp_path: Path) -> PortableBundle:
    config_path, _ = _config_file(tmp_path)
    return create_portable_bundle(
        tmp_path / "personal-model",
        config_path=config_path,
        backend="torch",
        model_size="tiny",
        seed=23,
        parameter_dtype="float32",
    )


def test_random_portable_bundle_loads_in_torch_without_external_weights(
    tmp_path: Path,
) -> None:
    config_path, config = _config_file(tmp_path)
    bundle = create_portable_bundle(
        tmp_path / "personal-model",
        config_path=config_path,
        backend="torch",
        model_size="tiny",
        seed=17,
        parameter_dtype="float32",
    )

    assert bundle.config == config
    assert bundle.manifest["scratch_only"] is True
    assert bundle.manifest["external_weights"] is False
    assert bundle.manifest["parameters"] == config.expected_parameter_count()

    from safetensors.torch import load_file

    state = load_file(str(bundle.weights_path), device="cpu")
    model = HybridLM(config)
    loaded = model.load_canonical_state_dict(state)
    assert loaded.missing_keys == []
    assert loaded.unexpected_keys == []
    logits = model(torch.tensor([[bundle.tokenizer.bos_id]], dtype=torch.long))
    assert tuple(logits.shape) == (1, 1, config.vocab_size)


def test_portable_bundle_detects_weight_tampering(tmp_path: Path) -> None:
    config_path, _ = _config_file(tmp_path)
    bundle = create_portable_bundle(
        tmp_path / "personal-model",
        config_path=config_path,
        backend="torch",
        model_size="tiny",
        seed=3,
        parameter_dtype="float32",
    )
    with bundle.weights_path.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(PortableBundleError, match="changed"):
        load_portable_bundle(bundle.root)


def test_portable_bundle_rejects_self_consistent_oversized_weights(
    tmp_path: Path,
) -> None:
    bundle = _new_bundle(tmp_path)
    maximum = bundle.config.expected_parameter_count() * 4 + 16 * 1024 * 1024
    with bundle.weights_path.open("r+b") as handle:
        handle.truncate(maximum + 1)
    _refresh_weights_checksum(bundle)

    with pytest.raises(PortableBundleError, match="structural bound"):
        load_portable_bundle(bundle.root)


def test_portable_bundle_accepts_uniform_manifest_bfloat16(tmp_path: Path) -> None:
    config_path, _ = _config_file(tmp_path)
    bundle = create_portable_bundle(
        tmp_path / "personal-model",
        config_path=config_path,
        backend="torch",
        model_size="tiny",
        seed=11,
        parameter_dtype="bfloat16",
    )

    assert load_portable_bundle(bundle.root).parameter_dtype == "bfloat16"


def test_portable_bundle_rejects_uniform_wrong_tensor_dtype(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    bundle = _new_bundle(tmp_path)
    state = {
        name: value.to(dtype=torch.float16)
        for name, value in load_file(str(bundle.weights_path)).items()
    }
    _rewrite_weights(bundle, state, metadata={"format": PORTABLE_STATE_FORMAT})

    with pytest.raises(PortableBundleError, match="dtype differs"):
        load_portable_bundle(bundle.root)


def test_portable_bundle_rejects_mixed_tensor_dtypes(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    bundle = _new_bundle(tmp_path)
    state = load_file(str(bundle.weights_path))
    first_name = next(iter(state))
    state[first_name] = state[first_name].to(dtype=torch.bfloat16)
    _rewrite_weights(bundle, state, metadata={"format": PORTABLE_STATE_FORMAT})

    with pytest.raises(PortableBundleError, match="dtype differs"):
        load_portable_bundle(bundle.root)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_portable_bundle_rejects_non_finite_weights(
    tmp_path: Path,
    non_finite: float,
) -> None:
    from safetensors.torch import load_file

    bundle = _new_bundle(tmp_path)
    state = load_file(str(bundle.weights_path))
    first_name = next(iter(state))
    corrupted = state[first_name].clone()
    corrupted.view(-1)[0] = non_finite
    state[first_name] = corrupted
    _rewrite_weights(bundle, state, metadata={"format": PORTABLE_STATE_FORMAT})

    with pytest.raises(PortableBundleError, match="non-finite"):
        load_portable_bundle(bundle.root)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_portable_bundle_rejects_missing_or_extra_tensor(
    tmp_path: Path,
    mutation: str,
) -> None:
    from safetensors.torch import load_file

    bundle = _new_bundle(tmp_path)
    state = load_file(str(bundle.weights_path))
    if mutation == "missing":
        state.pop(next(iter(state)))
    else:
        state["unexpected.weight"] = torch.ones(1, dtype=torch.float32)
    _rewrite_weights(bundle, state, metadata={"format": PORTABLE_STATE_FORMAT})

    with pytest.raises(PortableBundleError, match="tensor names differ"):
        load_portable_bundle(bundle.root)


def test_portable_bundle_rejects_wrong_tensor_shape(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    bundle = _new_bundle(tmp_path)
    state = load_file(str(bundle.weights_path))
    first_name = next(iter(state))
    state[first_name] = state[first_name].reshape(-1)[:-1]
    _rewrite_weights(bundle, state, metadata={"format": PORTABLE_STATE_FORMAT})

    with pytest.raises(PortableBundleError, match="shape differs"):
        load_portable_bundle(bundle.root)


@pytest.mark.parametrize(
    "metadata",
    [None, {"format": PORTABLE_STATE_FORMAT, "extra": "not-canonical"}],
)
def test_portable_bundle_rejects_noncanonical_safetensors_metadata(
    tmp_path: Path,
    metadata: dict[str, str] | None,
) -> None:
    from safetensors.torch import load_file

    bundle = _new_bundle(tmp_path)
    state = load_file(str(bundle.weights_path))
    _rewrite_weights(bundle, state, metadata=metadata)

    with pytest.raises(PortableBundleError, match="metadata is incompatible"):
        load_portable_bundle(bundle.root)


def test_portable_bundle_can_freeze_a_corpus_trained_tokenizer(tmp_path: Path) -> None:
    config_path, _ = _config_file(tmp_path)
    tokenizer = build_scratch_tokenizer(
        [
            "copper zeppelin conversations should become useful vocabulary " * 8,
            "rights cleared local training text remains private " * 8,
        ],
        tmp_path / "corpus-tokenizer",
    )
    bundle = create_portable_bundle(
        tmp_path / "corpus-model",
        config_path=config_path,
        backend="torch",
        model_size="tiny",
        seed=9,
        parameter_dtype="float32",
        tokenizer_path=tokenizer.artifact_dir,
    )

    assert (
        bundle.tokenizer.metadata["tokenizer_sha256"]
        == tokenizer.metadata["tokenizer_sha256"]
    )
    assert (
        bundle.tokenizer.metadata["training_text_sha256"]
        == tokenizer.metadata["training_text_sha256"]
    )
    assert bundle.tokenizer.metadata["base_vocabulary_size"] > 561
    with pytest.raises(FileExistsError):
        create_portable_bundle(
            bundle.root,
            config_path=config_path,
            backend="torch",
            model_size="tiny",
            tokenizer_path=tokenizer.artifact_dir,
        )


@pytest.mark.parametrize("parameter_dtype", ["float32", "bfloat16"])
def test_conversational_foundation_publication_is_complete_changed_and_immutable(
    tmp_path: Path,
    parameter_dtype: str,
) -> None:
    from safetensors.torch import load_file, save_file

    config_path, _ = _config_file(tmp_path)
    parent = create_portable_bundle(
        tmp_path / "parent",
        config_path=config_path,
        backend="torch",
        model_size="tiny",
        seed=41,
        parameter_dtype=parameter_dtype,  # type: ignore[arg-type]
    )
    parent_manifest = dict(parent.manifest)
    state = load_file(str(parent.weights_path), device="cpu")
    name = next(iter(sorted(state)))
    changed = state[name].clone()
    changed.view(-1)[0] = changed.view(-1)[0] + 1
    state[name] = changed
    candidate_weights = tmp_path / "candidate.safetensors"
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        str(candidate_weights),
        metadata={"format": PORTABLE_STATE_FORMAT},
    )
    recipe_sha256 = hashlib.sha256(b"bounded-conversation-recipe").hexdigest()
    source_sha256 = {
        "lesson-1": hashlib.sha256(b"first private lesson").hexdigest(),
        "lesson-2": hashlib.sha256(b"second private lesson").hexdigest(),
    }

    trained = publish_conversational_foundation_bundle(
        parent,
        candidate_weights,
        tmp_path / "trained",
        backend="torch",
        training_config_sha256=recipe_sha256,
        source_sha256=source_sha256,
        steps=3,
        tokens_seen=27,
    )

    assert load_portable_bundle(parent.root).manifest == parent_manifest
    assert trained.parameter_dtype == parameter_dtype
    assert trained.manifest["weights_sha256"] != parent.manifest["weights_sha256"]
    assert trained.manifest["config_sha256"] == parent.manifest["config_sha256"]
    assert trained.manifest["tokenizer_sha256"] == parent.manifest["tokenizer_sha256"]
    assert trained.manifest["training"] == {
        "format": "tinydog-foundation-provenance-v1",
        "parent_weights_sha256": parent.manifest["weights_sha256"],
        "training_config_sha256": recipe_sha256,
        "source_sha256": source_sha256,
        "global_step": 3,
        "tokens_seen": 27,
        "training_complete": True,
        "stage_count": 1,
        "total_steps": 3,
    }


def test_conversational_foundation_publication_rejects_unchanged_tensor_payload(
    tmp_path: Path,
) -> None:
    bundle = _new_bundle(tmp_path)
    recipe_sha256 = hashlib.sha256(b"recipe").hexdigest()
    source_sha256 = {"lesson": hashlib.sha256(b"lesson").hexdigest()}

    with pytest.raises(PortableBundleError, match="did not change"):
        publish_conversational_foundation_bundle(
            bundle,
            bundle.weights_path,
            tmp_path / "unchanged",
            backend="torch",
            training_config_sha256=recipe_sha256,
            source_sha256=source_sha256,
            steps=1,
            tokens_seen=1,
        )
