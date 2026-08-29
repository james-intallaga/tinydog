from dataclasses import replace
import hashlib
import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from mini_arithmetic.community_updates import TensorSpec
from mini_arithmetic.foundation_floaty import (
    FEED_FORWARD,
    FLOATY_STATE_VERSION,
    OPERATOR,
    FloatyConfig,
    FloatyState,
    FoundationFloatyLM,
)
from mini_arithmetic.hybrid import ATTENTION, CONV, HybridConfig, HybridLM


def tiny_config() -> HybridConfig:
    return HybridConfig(
        vocab_size=41,
        context_length=20,
        model_dim=16,
        ffn_dim=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        conv_kernel_size=3,
        rope_theta=10_000.0,
        layer_schedule=(CONV, ATTENTION),
    )


def clone_foundation(source: HybridLM) -> HybridLM:
    clone = HybridLM(source.config)
    clone.update(source.parameters(), strict=True)
    mx.eval(clone.parameters())
    return clone


def trained_state(model: FoundationFloatyLM, amount: float = 0.02) -> FloatyState:
    state = model.export_floaty_state()
    tensors = {
        name: (
            mx.full(value.shape, amount, dtype=value.dtype)
            if name.endswith("down.weight")
            else mx.array(value)
        )
        for name, value in state.tensors.items()
    }
    return replace(state, tensors=tensors)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"rank": True},
        {"alpha": 0.0},
        {"alpha": float("inf")},
        {"attachments": []},
        {"attachments": ()},
        {"attachments": ("unknown",)},
        {"attachments": (FEED_FORWARD, OPERATOR)},
        {"attachments": (OPERATOR, OPERATOR)},
        {"target_layers": []},
        {"target_layers": ()},
        {"target_layers": (1, 0)},
        {"target_layers": (0, 0)},
        {"target_layers": (-1,)},
    ],
)
def test_floaty_config_is_strict_and_canonical(kwargs) -> None:
    with pytest.raises(ValueError):
        FloatyConfig(**kwargs)


def test_config_rejects_full_rank_and_out_of_range_targets() -> None:
    foundation = HybridLM(tiny_config())
    with pytest.raises(ValueError, match="smaller"):
        FoundationFloatyLM(foundation, FloatyConfig(rank=16))
    with pytest.raises(ValueError, match="outside"):
        FoundationFloatyLM(foundation, FloatyConfig(rank=2, target_layers=(2,)))


def test_exact_parameter_partition_freezes_all_foundation_tensors() -> None:
    foundation = HybridLM(tiny_config())
    config = FloatyConfig(
        rank=2,
        attachments=(OPERATOR,),
        target_layers=(1,),
    )
    model = FoundationFloatyLM(foundation, config)
    partition = model.assert_parameter_partition()

    assert model.floaty_parameter_count() == 2 * 16 * 2
    assert model.floaty_parameter_count() == config.expected_parameter_count(
        model_dim=16,
        num_layers=2,
    )
    assert partition.floaty == (
        "floaty.blocks.1.operator.down.weight",
        "floaty.blocks.1.operator.up.weight",
    )
    assert all(name.startswith("foundation.") for name in partition.foundation)
    assert model.trainable_parameter_names() == partition.floaty
    assert model.exported_floaty_tensor_names() == (
        "blocks.1.operator.down.weight",
        "blocks.1.operator.up.weight",
    )
    assert tree_flatten(model.foundation.trainable_parameters()) == []


def test_partition_remains_safe_when_floaty_is_frozen_and_can_be_rearmed() -> None:
    model = FoundationFloatyLM(HybridLM(tiny_config()), FloatyConfig(rank=2))
    model.freeze()

    partition = model.assert_parameter_partition()
    assert model.trainable_parameter_names() == ()
    with pytest.raises(RuntimeError, match="optimizer surface"):
        model.floaty_parameters()

    state = model.export_floaty_state()
    model.import_floaty_state(state)
    model.freeze_foundation()
    assert model.trainable_parameter_names() == partition.floaty
    assert tree_flatten(model.floaty_parameters())

    model.foundation.unfreeze()
    with pytest.raises(RuntimeError, match="outside|Foundation"):
        model.assert_parameter_partition()


def test_gradients_are_floaty_only() -> None:
    mx.random.seed(91)
    model = FoundationFloatyLM(HybridLM(tiny_config()), FloatyConfig(rank=2))
    tokens = mx.array([[1, 3, 5, 7]], dtype=mx.int32)
    targets = mx.array([[3, 5, 7, 9]], dtype=mx.int32)

    def loss_fn(current: FoundationFloatyLM) -> mx.array:
        return nn.losses.cross_entropy(current(tokens), targets, reduction="mean")

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, gradients)
    gradient_names = tuple(sorted(name for name, _ in tree_flatten(gradients)))

    assert mx.isfinite(loss).item()
    assert gradient_names == model.floaty_parameter_names()
    assert all(name.startswith("floaty.") for name in gradient_names)


def test_untrained_floaty_is_exact_noop_for_full_and_cached_inference() -> None:
    mx.random.seed(92)
    foundation = HybridLM(tiny_config())
    model = FoundationFloatyLM(foundation, FloatyConfig(rank=2))
    tokens = mx.array([[2, 4, 8, 16, 32]], dtype=mx.int32)

    foundation_logits = foundation(tokens)
    floaty_logits = model(tokens)
    cache = model.init_cache(1)
    cached_parts = []
    for index in range(tokens.shape[1]):
        part, cache = model.forward_cached(tokens[:, index : index + 1], cache)
        cached_parts.append(part)
    cached_logits = mx.concatenate(cached_parts, axis=1)
    mx.eval(foundation_logits, floaty_logits, cached_logits)

    assert mx.array_equal(foundation_logits, floaty_logits).item()
    assert mx.allclose(floaty_logits, cached_logits, atol=2e-5, rtol=2e-5).item()
    assert cache.offset == tokens.shape[1]


def test_trained_floaty_composes_in_full_cached_and_direct_block_paths() -> None:
    mx.random.seed(93)
    foundation = HybridLM(tiny_config())
    model = FoundationFloatyLM(foundation, FloatyConfig(rank=2))
    model.import_floaty_state(trained_state(model))
    tokens = mx.array([[6, 12, 18, 24, 30]], dtype=mx.int32)

    foundation_logits = foundation(tokens)
    full_logits = model(tokens)
    first, cache = model.forward_cached(tokens[:, :2])
    second, cache = model.forward_cached(tokens[:, 2:], cache)
    cached_logits = mx.concatenate((first, second), axis=1)

    # ScratchMultimodalLM consumes these three attributes directly.  Its path
    # must see the same floaty-composed blocks as ordinary token inference.
    hidden = model.token_embedding(tokens)
    for block in model.blocks:
        hidden = block(hidden)
    direct_logits = model.final_norm(hidden) @ model.token_embedding.weight.T
    mx.eval(foundation_logits, full_logits, cached_logits, direct_logits)

    assert not mx.array_equal(foundation_logits, full_logits).item()
    assert mx.allclose(full_logits, cached_logits, atol=2e-5, rtol=2e-5).item()
    assert mx.array_equal(full_logits, direct_logits).item()


def test_hashes_bind_exact_foundation_weights_and_resolved_config() -> None:
    mx.random.seed(94)
    first = FoundationFloatyLM(HybridLM(tiny_config()), FloatyConfig(rank=2))
    second = FoundationFloatyLM(
        clone_foundation(first.foundation),
        FloatyConfig(rank=2),
    )

    assert len(first.foundation_hash()) == 64
    assert first.foundation_hash() == second.foundation_hash()
    assert first.config_hash() == second.config_hash()

    second.foundation.token_embedding.weight = (
        second.foundation.token_embedding.weight + mx.array(1e-4)
    )
    mx.eval(second.foundation.parameters())
    assert first.foundation_hash() != second.foundation_hash()

    different_config = FoundationFloatyLM(
        clone_foundation(first.foundation),
        FloatyConfig(rank=2, alpha=8.0),
    )
    assert first.foundation_hash() == different_config.foundation_hash()
    assert first.config_hash() != different_config.config_hash()


def test_canonical_schema_content_and_file_hash_match_config_hash(tmp_path) -> None:
    model = FoundationFloatyLM(
        HybridLM(tiny_config()),
        FloatyConfig(rank=2, attachments=(OPERATOR,), target_layers=(1,)),
    )
    manifest = model.config_manifest()
    assert manifest["tensors"] == [
        {
            "name": "floaty.blocks.1.operator.down.weight",
            "shape": [16, 2],
            "dtype": "F32",
        },
        {
            "name": "floaty.blocks.1.operator.up.weight",
            "shape": [2, 16],
            "dtype": "F32",
        },
    ]
    queue_specs = tuple(
        TensorSpec(
            name=tensor["name"],
            shape=tuple(tensor["shape"]),
            dtype=tensor["dtype"],
        )
        for tensor in manifest["tensors"]
    )
    assert tuple(spec.name for spec in queue_specs) == model.floaty_parameter_names()
    manifest["attachments"].append("caller-mutation")
    manifest["tensors"][0]["shape"][0] = 999
    assert model.config_manifest()["attachments"] == [OPERATOR]
    assert model.config_manifest()["tensors"][0]["shape"] == [16, 2]

    destination = tmp_path / "floaty-schema.json"
    returned = model.write_floaty_schema(destination)
    payload = destination.read_bytes()

    assert returned == destination
    assert json.loads(payload) == model.config_manifest()
    assert hashlib.sha256(payload).hexdigest() == model.config_hash()

    with pytest.raises(FileExistsError):
        model.write_floaty_schema(destination)
    assert destination.read_bytes() == payload

    replacement = FoundationFloatyLM(
        clone_foundation(model.foundation),
        FloatyConfig(rank=2, alpha=8.0, attachments=(OPERATOR,), target_layers=(1,)),
    )
    replacement.write_floaty_schema(destination, overwrite=True)
    assert (
        hashlib.sha256(destination.read_bytes()).hexdigest()
        == replacement.config_hash()
    )


def test_floaty_state_round_trip_exports_no_foundation_tensors() -> None:
    mx.random.seed(95)
    source = FoundationFloatyLM(HybridLM(tiny_config()), FloatyConfig(rank=2))
    source.import_floaty_state(trained_state(source, amount=0.03))
    state = source.export_floaty_state()
    destination = FoundationFloatyLM(
        clone_foundation(source.foundation),
        FloatyConfig(rank=2),
    )
    destination.import_floaty_state(state)

    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    source_logits = source(tokens)
    destination_logits = destination(tokens)
    mx.eval(source_logits, destination_logits)

    assert state.format_version == FLOATY_STATE_VERSION
    assert tuple(sorted(state.tensors)) == source.exported_floaty_tensor_names()
    assert all("foundation" not in name for name in state.tensors)
    assert mx.allclose(source_logits, destination_logits).item()


def test_floaty_import_fails_closed_on_envelope_or_tensor_mismatch() -> None:
    model = FoundationFloatyLM(HybridLM(tiny_config()), FloatyConfig(rank=2))
    state = model.export_floaty_state()
    first_name = sorted(state.tensors)[0]
    first_value = state.tensors[first_name]

    with pytest.raises(TypeError, match="FloatyState"):
        model.import_floaty_state("adapter.safetensors")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="version"):
        model.import_floaty_state(replace(state, format_version=99))
    with pytest.raises(ValueError, match="version"):
        model.import_floaty_state(replace(state, format_version=True))
    with pytest.raises(ValueError, match="foundation hash"):
        model.import_floaty_state(replace(state, foundation_hash="0" * 64))
    with pytest.raises(ValueError, match="config hash"):
        model.import_floaty_state(replace(state, config_hash="0" * 64))

    missing = dict(state.tensors)
    missing.pop(first_name)
    with pytest.raises(ValueError, match="keys mismatch"):
        model.import_floaty_state(replace(state, tensors=missing))

    extra = dict(state.tensors)
    extra["foundation.token_embedding.weight"] = first_value
    with pytest.raises(ValueError, match="keys mismatch"):
        model.import_floaty_state(replace(state, tensors=extra))

    non_string_name = dict(state.tensors)
    non_string_name[7] = first_value  # type: ignore[index]
    with pytest.raises(TypeError, match="names"):
        model.import_floaty_state(replace(state, tensors=non_string_name))

    wrong_shape = dict(state.tensors)
    wrong_shape[first_name] = first_value.reshape(-1)
    with pytest.raises(ValueError, match="shape"):
        model.import_floaty_state(replace(state, tensors=wrong_shape))

    wrong_dtype = dict(state.tensors)
    wrong_dtype[first_name] = first_value.astype(mx.float16)
    with pytest.raises(ValueError, match="dtype"):
        model.import_floaty_state(replace(state, tensors=wrong_dtype))

    non_finite = dict(state.tensors)
    non_finite[first_name] = mx.full(
        first_value.shape, float("nan"), dtype=first_value.dtype
    )
    with pytest.raises(ValueError, match="non-finite"):
        model.import_floaty_state(replace(state, tensors=non_finite))
