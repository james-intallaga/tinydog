"""Cross-platform PyTorch implementation of the scratch hybrid language model.

This module intentionally has no MLX dependency, so it can be imported on
Windows, Linux, and non-Apple devices.  PyTorch itself is optional at package
import time: constructing a model without it raises an installation-oriented
error instead of making unrelated ``mini_arithmetic`` modules unusable.

The architecture mirrors :mod:`mini_arithmetic.hybrid`: dense double-gated
causal depthwise convolutions, grouped-query attention with QK RMSNorm and
full-head RoPE, SwiGLU feed-forwards, and a tied token embedding/output head.

Portable weights use the same canonical tensor names and shapes as the MLX
model.  Linear, embedding, and norm tensors already agree between frameworks.
Only depthwise-convolution kernels are transposed when crossing the boundary:
PyTorch ``[out, 1, kernel]`` becomes MLX ``[out, kernel, 1]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from .hybrid_config import ATTENTION, CONV, DEFAULT_LAYER_SCHEDULE, HybridConfig

try:  # PyTorch is an optional backend, not a package-wide import requirement.
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except Exception as error:  # pragma: no cover - exercised on torch-free hosts.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = error
else:
    _TORCH_IMPORT_ERROR = None


CANONICAL_STATE_FORMAT = "tinydog-hybrid-mlx-layout-v1"


def require_torch() -> None:
    """Raise a clear backend error when PyTorch is unavailable."""

    if torch is None:
        message = (
            "The PyTorch hybrid backend is not installed. Install a PyTorch "
            "build for this device from https://pytorch.org/get-started/locally/ "
            "and then retry."
        )
        raise RuntimeError(message) from _TORCH_IMPORT_ERROR


@dataclass
class AttentionCache:
    """Cached rotary keys and values for one grouped-query attention layer."""

    keys: Any | None = None
    values: Any | None = None


@dataclass
class ConvCache:
    """Rolling gated inputs for one causal depthwise-convolution layer."""

    state: Any | None = None


LayerCache: TypeAlias = AttentionCache | ConvCache


@dataclass
class HybridCache:
    """Incremental attention and convolution state for a model stream."""

    layers: list[LayerCache]
    batch_size: int
    # Absolute RoPE position and bounded resident KV length diverge after the
    # first sliding-window eviction.
    offset: int = 0
    resident_tokens: int = 0

    @classmethod
    def empty(cls, config: HybridConfig, batch_size: int) -> HybridCache:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        layers: list[LayerCache] = [
            AttentionCache() if layer_type == ATTENTION else ConvCache()
            for layer_type in config.layer_schedule
        ]
        return cls(layers=layers, batch_size=batch_size)

    def reset(self) -> None:
        """Clear stored tensors while retaining the model-shaped cache list."""

        self.offset = 0
        self.resident_tokens = 0
        for layer in self.layers:
            if isinstance(layer, AttentionCache):
                layer.keys = None
                layer.values = None
            else:
                layer.state = None

    def reserve(self, new_tokens: int, context_length: int) -> None:
        """Evict the oldest KV entries before appending a new cached chunk."""

        if new_tokens <= 0:
            raise ValueError("cached chunks must contain at least one token")
        if new_tokens > context_length:
            raise ValueError("cached chunk exceeds the model context length")
        keep = context_length - new_tokens
        if self.resident_tokens <= keep:
            return
        for layer in self.layers:
            if isinstance(layer, AttentionCache):
                if (layer.keys is None) != (layer.values is None):
                    raise ValueError(
                        "Attention cache must contain both keys and values"
                    )
                if layer.keys is not None:
                    layer.keys = layer.keys[:, :, -keep:, :] if keep else None
                    layer.values = layer.values[:, :, -keep:, :] if keep else None
        self.resident_tokens = keep


if nn is not None:

    class RMSNorm(nn.Module):
        """Small dependency-free RMSNorm with the same learned shape as MLX."""

        def __init__(self, dimensions: int, eps: float) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dimensions))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            scale = torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + self.eps)
            return inputs * scale * self.weight

    class SwiGLU(nn.Module):
        def __init__(self, config: HybridConfig) -> None:
            super().__init__()
            self.w1 = nn.Linear(config.model_dim, config.ffn_dim, bias=False)
            self.w2 = nn.Linear(config.ffn_dim, config.model_dim, bias=False)
            self.w3 = nn.Linear(config.model_dim, config.ffn_dim, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.w2(functional.silu(self.w1(inputs)) * self.w3(inputs))

    class DoubleGatedShortConv(nn.Module):
        """Dense double-gated causal depthwise convolution with short state."""

        def __init__(self, config: HybridConfig) -> None:
            super().__init__()
            self.model_dim = config.model_dim
            self.kernel_size = config.conv_kernel_size
            self.in_proj = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
            self.conv = nn.Conv1d(
                config.model_dim,
                config.model_dim,
                kernel_size=config.conv_kernel_size,
                groups=config.model_dim,
                bias=False,
            )
            self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)

        def forward(
            self,
            inputs: torch.Tensor,
            cache: ConvCache | None = None,
        ) -> torch.Tensor:
            first_gate, second_gate, values = self.in_proj(inputs).chunk(3, dim=-1)
            gated_values = first_gate * values
            state_length = self.kernel_size - 1

            if cache is None:
                convolution_input = functional.pad(
                    gated_values.transpose(1, 2),
                    (state_length, 0),
                )
            else:
                state = cache.state
                if state is None:
                    state = gated_values.new_zeros(
                        (inputs.shape[0], state_length, self.model_dim)
                    )
                convolution_input_time_major = torch.cat((state, gated_values), dim=1)
                cache.state = convolution_input_time_major[:, -state_length:, :]
                convolution_input = convolution_input_time_major.transpose(1, 2)

            convolution_input = convolution_input.contiguous()
            if convolution_input.device.type == "mps" and torch.is_grad_enabled():
                # Torch 2.5's MPS depthwise Conv1d backward fails on the
                # heterogeneous conv/attention schedule even with contiguous
                # input. A tiny explicit K-term slice sum is mathematically
                # identical for groups=model_dim and keeps gradients for both
                # inputs and convolution weights on supported MPS primitives.
                convolved = _explicit_depthwise_convolution(
                    convolution_input,
                    self.conv.weight,
                    self.conv.bias,
                )
            else:
                convolved = self.conv(convolution_input)
            convolved = convolved.transpose(1, 2)
            return self.out_proj(second_gate * convolved)

    def _explicit_depthwise_convolution(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Depthwise valid Conv1d using only MPS-safe slice/multiply/sum ops."""

        if inputs.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1:
            raise ValueError("explicit depthwise convolution shapes are invalid")
        if inputs.shape[1] != weight.shape[0] or inputs.shape[-1] < weight.shape[-1]:
            raise ValueError("explicit depthwise convolution dimensions differ")
        output_length = inputs.shape[-1] - weight.shape[-1] + 1
        terms = [
            inputs[:, :, offset : offset + output_length]
            * weight[:, 0, offset].reshape(1, -1, 1)
            for offset in range(weight.shape[-1])
        ]
        output = terms[0]
        for term in terms[1:]:
            output = output + term
        if bias is not None:
            output = output + bias.reshape(1, -1, 1)
        return output

    def _apply_rope(
        inputs: torch.Tensor,
        *,
        offset: int,
        theta: float,
    ) -> torch.Tensor:
        """Apply MLX's non-traditional (half-stride) full-head RoPE."""

        feature_dim = inputs.shape[-1]
        half_dim = feature_dim // 2
        if feature_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        frequency_indices = torch.arange(
            0,
            feature_dim,
            2,
            device=inputs.device,
            dtype=torch.float32,
        )
        inverse_frequencies = theta ** (-frequency_indices / feature_dim)
        positions = torch.arange(
            offset,
            offset + inputs.shape[-2],
            device=inputs.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, inverse_frequencies).to(dtype=inputs.dtype)
        cosine = angles.cos()[None, None, :, :]
        sine = angles.sin()[None, None, :, :]
        first_half = inputs[..., :half_dim]
        second_half = inputs[..., half_dim:]
        return torch.cat(
            (
                first_half * cosine - second_half * sine,
                second_half * cosine + first_half * sine,
            ),
            dim=-1,
        )

    class GroupedQueryAttention(nn.Module):
        """Causal GQA with QK RMSNorm and full-head rotary positions."""

        def __init__(self, config: HybridConfig) -> None:
            super().__init__()
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
            self.rope_theta = config.rope_theta
            query_width = self.num_heads * self.head_dim
            key_value_width = self.num_key_value_heads * self.head_dim
            self.q_proj = nn.Linear(config.model_dim, query_width, bias=False)
            self.k_proj = nn.Linear(config.model_dim, key_value_width, bias=False)
            self.v_proj = nn.Linear(config.model_dim, key_value_width, bias=False)
            self.out_proj = nn.Linear(query_width, config.model_dim, bias=False)
            self.q_norm = RMSNorm(config.head_dim, config.norm_eps)
            self.k_norm = RMSNorm(config.head_dim, config.norm_eps)

        def forward(
            self,
            inputs: torch.Tensor,
            *,
            offset: int = 0,
            cache: AttentionCache | None = None,
        ) -> torch.Tensor:
            batch, length, _ = inputs.shape
            queries = self.q_proj(inputs).reshape(
                batch, length, self.num_heads, self.head_dim
            )
            keys = self.k_proj(inputs).reshape(
                batch, length, self.num_key_value_heads, self.head_dim
            )
            values = self.v_proj(inputs).reshape(
                batch, length, self.num_key_value_heads, self.head_dim
            )

            queries = self.q_norm(queries).transpose(1, 2)
            keys = self.k_norm(keys).transpose(1, 2)
            values = values.transpose(1, 2)
            queries = _apply_rope(queries, offset=offset, theta=self.rope_theta)
            keys = _apply_rope(keys, offset=offset, theta=self.rope_theta)

            past_length = 0
            if cache is not None:
                if (cache.keys is None) != (cache.values is None):
                    raise ValueError(
                        "Attention cache must contain both keys and values"
                    )
                if cache.keys is not None:
                    past_length = cache.keys.shape[2]
                    keys = torch.cat((cache.keys, keys), dim=2)
                    values = torch.cat((cache.values, values), dim=2)
                cache.keys = keys
                cache.values = values

            groups = self.num_heads // self.num_key_value_heads
            expanded_keys = keys.repeat_interleave(groups, dim=1)
            expanded_values = values.repeat_interleave(groups, dim=1)
            key_positions = torch.arange(keys.shape[2], device=inputs.device)
            query_limits = past_length + torch.arange(length, device=inputs.device)
            causal_mask = key_positions[None, :] <= query_limits[:, None]
            attended = functional.scaled_dot_product_attention(
                queries,
                expanded_keys,
                expanded_values,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            attended = attended.transpose(1, 2).reshape(batch, length, -1)
            return self.out_proj(attended)

    class HybridBlock(nn.Module):
        def __init__(self, config: HybridConfig, layer_type: str) -> None:
            super().__init__()
            self.layer_type = layer_type
            self.operator_norm = RMSNorm(config.model_dim, config.norm_eps)
            self.operator = (
                GroupedQueryAttention(config)
                if layer_type == ATTENTION
                else DoubleGatedShortConv(config)
            )
            self.ffn_norm = RMSNorm(config.model_dim, config.norm_eps)
            self.feed_forward = SwiGLU(config)

        def forward(
            self,
            inputs: torch.Tensor,
            *,
            offset: int = 0,
            cache: LayerCache | None = None,
        ) -> torch.Tensor:
            normalized = self.operator_norm(inputs)
            if self.layer_type == ATTENTION:
                if cache is not None and not isinstance(cache, AttentionCache):
                    raise TypeError("Attention block received a convolution cache")
                mixed = self.operator(normalized, offset=offset, cache=cache)
            else:
                if cache is not None and not isinstance(cache, ConvCache):
                    raise TypeError("Convolution block received an attention cache")
                mixed = self.operator(normalized, cache=cache)
            hidden = inputs + mixed
            return hidden + self.feed_forward(self.ffn_norm(hidden))

    class HybridLM(nn.Module):
        """Randomly initialized decoder-only PyTorch hybrid language model."""

        def __init__(self, config: HybridConfig | None = None) -> None:
            super().__init__()
            self.config = config or HybridConfig()
            self.token_embedding = nn.Embedding(
                self.config.vocab_size,
                self.config.model_dim,
            )
            # torch.nn.Embedding defaults to N(0, 1), unlike MLX's
            # width-scaled embedding initialization.  With a tied output head
            # that produces enormous random logits and makes a fresh model
            # nearly untrainable.  Keep the scratch distribution explicitly
            # backend-neutral.
            nn.init.normal_(
                self.token_embedding.weight,
                mean=0.0,
                std=self.config.model_dim**-0.5,
            )
            self.blocks = nn.ModuleList(
                HybridBlock(self.config, layer_type)
                for layer_type in self.config.layer_schedule
            )
            self.final_norm = RMSNorm(self.config.model_dim, self.config.norm_eps)

        def _validate_tokens(self, tokens: torch.Tensor, offset: int) -> None:
            if tokens.ndim != 2:
                raise ValueError("tokens must have shape [batch, length]")
            if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
                raise ValueError("tokens must contain a non-empty batch and sequence")
            total_length = offset + tokens.shape[1]
            if total_length > self.config.context_length:
                raise ValueError(
                    f"Sequence end {total_length} exceeds context "
                    f"{self.config.context_length}"
                )

        def _forward(
            self,
            tokens: torch.Tensor,
            *,
            offset: int = 0,
            cache: HybridCache | None = None,
        ) -> torch.Tensor:
            # Cached streams bound resident KV separately and can have absolute
            # RoPE positions beyond the context length.
            self._validate_tokens(tokens, 0 if cache is not None else offset)
            hidden = self.token_embedding(tokens)
            for index, block in enumerate(self.blocks):
                layer_cache = None if cache is None else cache.layers[index]
                hidden = block(hidden, offset=offset, cache=layer_cache)
            hidden = self.final_norm(hidden)
            # Structural tying: no separate output-head parameter exists.
            return functional.linear(hidden, self.token_embedding.weight)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            """Return full causal sequence logits for training or evaluation."""

            return self._forward(tokens)

        def init_cache(self, batch_size: int) -> HybridCache:
            """Create an empty model-shaped incremental inference cache."""

            return HybridCache.empty(self.config, batch_size)

        def forward_cached(
            self,
            tokens: torch.Tensor,
            cache: HybridCache | None = None,
        ) -> tuple[torch.Tensor, HybridCache]:
            """Decode one new token or a heterogeneous prefill chunk.

            The input must contain only tokens not previously processed by this
            cache.  Old attention KV tensors slide out at ``context_length``;
            absolute RoPE positions continue increasing.
            """

            if tokens.ndim != 2:
                raise ValueError("tokens must have shape [batch, length]")
            if cache is None:
                cache = self.init_cache(tokens.shape[0])
            if cache.batch_size != tokens.shape[0]:
                raise ValueError(
                    f"Cache batch size {cache.batch_size} does not match tokens "
                    f"{tokens.shape[0]}"
                )
            if len(cache.layers) != len(self.blocks):
                raise ValueError("Cache layer count does not match model")

            cache.reserve(tokens.shape[1], self.config.context_length)
            logits = self._forward(tokens, offset=cache.offset, cache=cache)
            cache.offset += tokens.shape[1]
            cache.resident_tokens += tokens.shape[1]
            return logits, cache

        def parameter_count(self) -> int:
            """Count all actual trainable parameters."""

            return sum(parameter.numel() for parameter in self.parameters())

        def canonical_state_dict(
            self, *, clone: bool = True
        ) -> dict[str, torch.Tensor]:
            """Return MLX-compatible names and tensor layouts."""

            return canonical_state_dict(self, clone=clone)

        def load_canonical_state_dict(
            self,
            state: Mapping[str, Any],
            *,
            strict: bool = True,
        ) -> Any:
            """Load tensors produced by MLX or :meth:`canonical_state_dict`."""

            return load_canonical_state_dict(self, state, strict=strict)


else:  # Keep imports/config inspection useful even before torch is installed.

    class HybridLM:  # type: ignore[no-redef]
        """Unavailable-model placeholder that reports how to install PyTorch."""

        def __init__(self, config: HybridConfig | None = None) -> None:
            del config
            require_torch()


def _is_depthwise_conv_weight(name: str) -> bool:
    return name.endswith(".operator.conv.weight")


def canonical_state_dict(
    model: HybridLM,
    *,
    clone: bool = True,
) -> dict[str, Any]:
    """Export a framework-neutral state dictionary in the MLX tensor layout.

    Names exactly match ``tree_flatten(mlx_model.parameters())``.  Returned
    tensors remain PyTorch tensors so callers can directly use
    ``safetensors.torch.save_file``; NumPy conversion is intentionally left to
    the caller.  With ``clone=True`` (the default), the mapping is independent
    of subsequent optimizer updates.
    """

    require_torch()
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch HybridLM")
    canonical: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        exported = value.permute(0, 2, 1) if _is_depthwise_conv_weight(name) else value
        exported = exported.detach()
        canonical[name] = exported.clone() if clone else exported
    return canonical


def load_canonical_state_dict(
    model: HybridLM,
    state: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Any:
    """Load the canonical MLX-compatible mapping into a PyTorch model.

    Mapping values may be PyTorch tensors or array-like CPU values.  PyTorch's
    normal strict missing/unexpected-key validation is retained.
    """

    require_torch()
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a PyTorch HybridLM")
    converted: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if _is_depthwise_conv_weight(name):
            if tensor.ndim != 3:
                raise ValueError(
                    f"Canonical convolution tensor {name!r} must be rank 3"
                )
            tensor = tensor.permute(0, 2, 1).contiguous()
        converted[name] = tensor
    return model.load_state_dict(converted, strict=strict)


def create_hybrid_model(config: HybridConfig | None = None) -> HybridLM:
    """Construct a fresh random PyTorch model without reading any checkpoint."""

    require_torch()
    return HybridLM(config)


# Explicit backend-prefixed aliases make mixed-backend call sites unambiguous.
TorchHybridConfig = HybridConfig
TorchHybridCache = HybridCache
TorchHybridLM = HybridLM


__all__ = [
    "ATTENTION",
    "CANONICAL_STATE_FORMAT",
    "CONV",
    "DEFAULT_LAYER_SCHEDULE",
    "AttentionCache",
    "ConvCache",
    "HybridCache",
    "HybridConfig",
    "HybridLM",
    "TorchHybridCache",
    "TorchHybridConfig",
    "TorchHybridLM",
    "canonical_state_dict",
    "create_hybrid_model",
    "load_canonical_state_dict",
    "require_torch",
]
