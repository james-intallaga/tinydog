import copy

import pytest

torch = pytest.importorskip("torch")
functional = pytest.importorskip("torch.nn.functional")
torch_hybrid = pytest.importorskip("mini_arithmetic.torch_hybrid")
ATTENTION = torch_hybrid.ATTENTION
CONV = torch_hybrid.CONV
AttentionCache = torch_hybrid.AttentionCache
ConvCache = torch_hybrid.ConvCache
HybridConfig = torch_hybrid.HybridConfig
HybridLM = torch_hybrid.HybridLM


def tiny_config(
    *,
    schedule: tuple[str, ...] = (CONV, ATTENTION, CONV),
    context_length: int = 24,
) -> HybridConfig:
    return HybridConfig(
        vocab_size=97,
        context_length=context_length,
        model_dim=32,
        ffn_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        conv_kernel_size=3,
        rope_theta=10_000.0,
        layer_schedule=schedule,
    )


def test_config_schedule_and_exact_parameter_count() -> None:
    default = HybridConfig()
    assert HybridConfig.__module__ == "mini_arithmetic.hybrid_config"
    assert default.num_conv_layers == 10
    assert default.num_attention_layers == 6
    assert default.expected_parameter_count() == 320_929_536

    config = tiny_config()
    model = HybridLM(config)
    assert model.parameter_count() == config.expected_parameter_count()
    assert "output_head" not in dict(model.named_parameters())


def test_random_embedding_scale_matches_backend_neutral_width_rule() -> None:
    torch.manual_seed(123)
    config = HybridConfig(
        vocab_size=32_768,
        context_length=32,
        model_dim=384,
        ffn_dim=1_024,
        num_attention_heads=6,
        num_key_value_heads=3,
        head_dim=64,
        layer_schedule=(CONV, ATTENTION),
    )
    model = HybridLM(config)
    observed = float(model.token_embedding.weight.std())
    expected = config.model_dim**-0.5

    assert observed == pytest.approx(expected, rel=0.02)
    with torch.inference_mode():
        logits = model(torch.tensor([[1, 2, 3, 4]]))
    assert float(logits.std()) < 3.0
    assert float(logits.abs().max()) < 20.0


def test_full_shape_causality_and_training_gradients() -> None:
    torch.manual_seed(11)
    model = HybridLM(tiny_config())
    tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    logits = model(tokens)
    assert logits.shape == (2, 5, 97)

    prefix = [3, 7, 11, 15]
    first = torch.tensor([[*prefix, 19, 23, 29]])
    second = torch.tensor([[*prefix, 31, 37, 41]])
    first_logits = model(first)
    second_logits = model(second)
    torch.testing.assert_close(
        first_logits[:, : len(prefix)],
        second_logits[:, : len(prefix)],
        atol=1e-6,
        rtol=1e-6,
    )
    assert not torch.allclose(
        first_logits[:, len(prefix)], second_logits[:, len(prefix)]
    )

    targets = torch.tensor([[2, 3, 4, 5, 6], [4, 3, 2, 1, 0]])
    loss = functional.cross_entropy(logits.reshape(-1, 97), targets.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_depthwise_convolution_receives_contiguous_channel_first_input() -> None:
    model = HybridLM(tiny_config(schedule=(CONV, ATTENTION, CONV)))
    observed: list[bool] = []
    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            handles.append(
                module.register_forward_pre_hook(
                    lambda _module, inputs: observed.append(inputs[0].is_contiguous())
                )
            )
    try:
        logits = model(torch.tensor([[1, 2, 3, 4, 5]]))
        logits.sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    assert len(observed) == 2
    assert all(observed)


def test_explicit_mps_safe_depthwise_convolution_matches_native_forward_and_gradients() -> (
    None
):
    torch.manual_seed(13)
    native_inputs = torch.randn(2, 5, 9, requires_grad=True)
    native_weight = torch.randn(5, 1, 3, requires_grad=True)
    explicit_inputs = native_inputs.detach().clone().requires_grad_(True)
    explicit_weight = native_weight.detach().clone().requires_grad_(True)

    native = functional.conv1d(
        native_inputs,
        native_weight,
        groups=native_inputs.shape[1],
    )
    explicit = torch_hybrid._explicit_depthwise_convolution(
        explicit_inputs,
        explicit_weight,
        None,
    )
    torch.testing.assert_close(explicit, native)

    native.square().sum().backward()
    explicit.square().sum().backward()
    torch.testing.assert_close(explicit_inputs.grad, native_inputs.grad)
    torch.testing.assert_close(explicit_weight.grad, native_weight.grad)


def test_heterogeneous_prefill_and_token_decode_match_full_forward() -> None:
    torch.manual_seed(33)
    config = tiny_config(schedule=(ATTENTION, CONV, ATTENTION, CONV))
    model = HybridLM(config).eval()
    tokens = torch.tensor(
        [
            [2, 5, 8, 13, 21, 34, 55],
            [1, 3, 6, 10, 15, 21, 28],
        ]
    )
    with torch.no_grad():
        full_logits = model(tokens)
        first, cache = model.forward_cached(tokens[:, :3])
        second, returned_cache = model.forward_cached(tokens[:, 3:5], cache)
        token_parts = []
        for position in range(5, tokens.shape[1]):
            part, returned_cache = model.forward_cached(
                tokens[:, position : position + 1], returned_cache
            )
            token_parts.append(part)
        cached_logits = torch.cat((first, second, *token_parts), dim=1)

    assert returned_cache is cache
    assert cache.offset == tokens.shape[1]
    assert cache.resident_tokens == tokens.shape[1]
    torch.testing.assert_close(full_logits, cached_logits, atol=2e-5, rtol=2e-5)

    for layer_type, layer_cache in zip(
        config.layer_schedule, cache.layers, strict=True
    ):
        if layer_type == ATTENTION:
            assert isinstance(layer_cache, AttentionCache)
            assert layer_cache.keys is not None
            assert layer_cache.values is not None
            assert layer_cache.keys.shape == (
                2,
                config.num_key_value_heads,
                tokens.shape[1],
                config.head_dim,
            )
        else:
            assert isinstance(layer_cache, ConvCache)
            assert layer_cache.state is not None
            assert layer_cache.state.shape == (
                2,
                config.conv_kernel_size - 1,
                config.model_dim,
            )


def test_cache_slides_but_absolute_rope_offset_keeps_advancing() -> None:
    torch.manual_seed(44)
    config = tiny_config(
        schedule=(ATTENTION, CONV, ATTENTION),
        context_length=6,
    )
    model = HybridLM(config).eval()
    cache = model.init_cache(1)
    with torch.no_grad():
        for position in range(15):
            output, cache = model.forward_cached(
                torch.tensor([[position % config.vocab_size]]), cache
            )
            assert torch.isfinite(output).all()

    assert cache.offset == 15
    assert cache.resident_tokens == config.context_length
    for layer in cache.layers:
        if isinstance(layer, AttentionCache):
            assert layer.keys is not None
            assert layer.values is not None
            assert layer.keys.shape[2] == config.context_length


def test_canonical_state_names_mlx_layout_and_round_trip() -> None:
    torch.manual_seed(55)
    config = tiny_config(schedule=(CONV, ATTENTION))
    source = HybridLM(config).eval()
    canonical = source.canonical_state_dict()

    assert set(canonical) == set(source.state_dict())
    assert canonical["blocks.0.operator.conv.weight"].shape == (
        config.model_dim,
        config.conv_kernel_size,
        1,
    )
    assert source.state_dict()["blocks.0.operator.conv.weight"].shape == (
        config.model_dim,
        1,
        config.conv_kernel_size,
    )
    assert "token_embedding.weight" in canonical
    assert not any("head" in name for name in canonical)

    destination = HybridLM(config).eval()
    result = destination.load_canonical_state_dict(canonical)
    assert not result.missing_keys
    assert not result.unexpected_keys
    tokens = torch.tensor([[1, 4, 9, 16, 25]])
    with torch.no_grad():
        torch.testing.assert_close(source(tokens), destination(tokens), atol=0, rtol=0)

    detached_copy = copy.deepcopy(canonical)
    with torch.no_grad():
        source.token_embedding.weight.add_(1)
    torch.testing.assert_close(
        canonical["token_embedding.weight"],
        detached_copy["token_embedding.weight"],
        atol=0,
        rtol=0,
    )


def test_optional_backend_failure_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr(torch_hybrid, "torch", None)
    monkeypatch.setattr(
        torch_hybrid,
        "_TORCH_IMPORT_ERROR",
        ImportError("simulated missing backend"),
    )
    with pytest.raises(RuntimeError, match="pytorch.org/get-started/locally"):
        torch_hybrid.require_torch()
