import inspect

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mini_arithmetic.hybrid import (
    ATTENTION,
    CONV,
    AttentionCache,
    ConvCache,
    HybridConfig,
    HybridLM,
)


def tiny_config(
    *,
    schedule: tuple[str, ...] = (CONV, ATTENTION, CONV),
) -> HybridConfig:
    return HybridConfig(
        vocab_size=97,
        context_length=24,
        model_dim=32,
        ffn_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        conv_kernel_size=3,
        rope_theta=10_000.0,
        layer_schedule=schedule,
    )


def test_default_schedule_and_exact_parameter_count() -> None:
    config = HybridConfig()
    assert config.layer_schedule == (
        CONV,
        CONV,
        ATTENTION,
        CONV,
        CONV,
        ATTENTION,
        CONV,
        CONV,
        ATTENTION,
        CONV,
        ATTENTION,
        CONV,
        ATTENTION,
        CONV,
        ATTENTION,
        CONV,
    )
    assert config.num_conv_layers == 10
    assert config.num_attention_layers == 6
    assert config.expected_parameter_count() == 320_929_536

    model = HybridLM(config)
    assert model.parameter_count() == 320_929_536


def test_full_sequence_shape_and_training_gradients() -> None:
    mx.random.seed(11)
    model = HybridLM(tiny_config())
    tokens = mx.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    assert logits.shape == (2, 5, 97)

    targets = mx.array([[2, 3, 4, 5, 6], [4, 3, 2, 1, 0]], dtype=mx.int32)

    def loss_fn(current_model: HybridLM) -> mx.array:
        return nn.losses.cross_entropy(current_model(tokens), targets, reduction="mean")

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    flat_gradients = tree_flatten(gradients)
    mx.eval(loss, gradients)
    assert mx.isfinite(loss).item()
    assert flat_gradients
    assert sum(value.size for _, value in flat_gradients) == model.parameter_count()


def test_full_forward_is_causal_without_future_leakage() -> None:
    mx.random.seed(22)
    model = HybridLM(tiny_config())
    prefix = [3, 7, 11, 15]
    first = mx.array([[*prefix, 19, 23, 29]], dtype=mx.int32)
    second = mx.array([[*prefix, 31, 37, 41]], dtype=mx.int32)
    first_logits = model(first)
    second_logits = model(second)
    mx.eval(first_logits, second_logits)

    assert mx.allclose(
        first_logits[:, : len(prefix)], second_logits[:, : len(prefix)], atol=1e-6
    ).item()
    assert not mx.allclose(
        first_logits[:, len(prefix)], second_logits[:, len(prefix)]
    ).item()


def test_model_is_random_init_and_has_no_checkpoint_dependency(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert list(inspect.signature(HybridLM).parameters) == ["config"]

    mx.random.seed(101)
    first = HybridLM(tiny_config(schedule=(CONV, ATTENTION)))
    mx.eval(first.parameters())
    first_weight = first.token_embedding.weight

    mx.random.seed(202)
    second = HybridLM(tiny_config(schedule=(CONV, ATTENTION)))
    mx.eval(second.parameters())
    second_weight = second.token_embedding.weight

    assert not mx.array_equal(first_weight, second_weight).item()
    assert list(tmp_path.iterdir()) == []


def test_incremental_cache_matches_full_logits_and_keeps_both_state_kinds() -> None:
    mx.random.seed(33)
    config = tiny_config()
    model = HybridLM(config)
    tokens = mx.array([[2, 5, 8, 13, 21, 34, 55]], dtype=mx.int32)
    full_logits = model(tokens)

    cache = model.init_cache(batch_size=1)
    cached_parts = []
    for position in range(tokens.shape[1]):
        logits, returned_cache = model.forward_cached(
            tokens[:, position : position + 1], cache
        )
        assert returned_cache is cache
        cached_parts.append(logits)
    cached_logits = mx.concatenate(cached_parts, axis=1)
    mx.eval(full_logits, cached_logits)

    assert cache.offset == tokens.shape[1]
    assert cache.resident_tokens == tokens.shape[1]
    assert mx.allclose(full_logits, cached_logits, atol=2e-5, rtol=2e-5).item()

    for layer_type, layer_cache in zip(
        config.layer_schedule, cache.layers, strict=True
    ):
        if layer_type == ATTENTION:
            assert isinstance(layer_cache, AttentionCache)
            assert layer_cache.keys is not None
            assert layer_cache.values is not None
            assert layer_cache.keys.shape == (
                1,
                config.num_key_value_heads,
                tokens.shape[1],
                config.head_dim,
            )
            assert layer_cache.values.shape == layer_cache.keys.shape
        else:
            assert isinstance(layer_cache, ConvCache)
            assert layer_cache.state is not None
            assert layer_cache.state.shape == (
                1,
                config.conv_kernel_size - 1,
                config.model_dim,
            )


def test_cached_chunk_prefill_matches_full_logits() -> None:
    mx.random.seed(44)
    model = HybridLM(tiny_config(schedule=(ATTENTION, CONV, ATTENTION)))
    tokens = mx.array([[1, 4, 9, 16, 25, 36]], dtype=mx.int32)
    full_logits = model(tokens)
    first, cache = model.forward_cached(tokens[:, :3])
    second, cache = model.forward_cached(tokens[:, 3:], cache)
    chunked_logits = mx.concatenate((first, second), axis=1)
    mx.eval(full_logits, chunked_logits)

    assert cache.offset == tokens.shape[1]
    assert mx.allclose(full_logits, chunked_logits, atol=2e-5, rtol=2e-5).item()


def test_cached_stream_slides_kv_without_resetting_absolute_rope_position() -> None:
    mx.random.seed(45)
    config = tiny_config(schedule=(ATTENTION, CONV, ATTENTION))
    model = HybridLM(config)
    cache = model.init_cache(1)
    outputs = []
    total = config.context_length + 9
    for position in range(total):
        token = mx.array([[position % config.vocab_size]], dtype=mx.int32)
        logits, cache = model.forward_cached(token, cache)
        outputs.append(logits)
    mx.eval(outputs)

    assert cache.offset == total
    assert cache.resident_tokens == config.context_length
    assert all(mx.isfinite(output).all().item() for output in outputs)
    for layer in cache.layers:
        if isinstance(layer, AttentionCache):
            assert layer.keys is not None
            assert layer.keys.shape[2] == config.context_length
