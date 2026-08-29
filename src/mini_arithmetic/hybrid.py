"""Scratch-only convolution/GQA language model implemented in MLX.

The module is deliberately independent from the live arithmetic model.  It
provides a normal full-sequence forward pass for training and an explicit
incremental cache that carries both attention KV tensors and convolution
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .hybrid_config import (
    ATTENTION as ATTENTION,
    CONV as CONV,
    DEFAULT_LAYER_SCHEDULE as DEFAULT_LAYER_SCHEDULE,
    HybridConfig as HybridConfig,
)


@dataclass
class AttentionCache:
    """Cached rotary keys and values for one GQA layer."""

    keys: mx.array | None = None
    values: mx.array | None = None


@dataclass
class ConvCache:
    """Rolling gated inputs for one causal depthwise convolution layer."""

    state: mx.array | None = None


LayerCache: TypeAlias = AttentionCache | ConvCache


@dataclass
class HybridCache:
    """Incremental state for every sequence-mixing layer in a model."""

    layers: list[LayerCache]
    batch_size: int
    # ``offset`` is the absolute rotary position; ``resident_tokens`` is the
    # bounded attention history still held in memory.  They intentionally
    # diverge after a sliding-window eviction.
    offset: int = 0
    resident_tokens: int = 0

    @classmethod
    def empty(cls, config: HybridConfig, batch_size: int) -> "HybridCache":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        layers: list[LayerCache] = [
            AttentionCache() if layer_type == ATTENTION else ConvCache()
            for layer_type in config.layer_schedule
        ]
        return cls(layers=layers, batch_size=batch_size)

    def reset(self) -> None:
        """Clear all stored arrays while retaining the cache allocation shape."""
        self.offset = 0
        self.resident_tokens = 0
        for layer in self.layers:
            if isinstance(layer, AttentionCache):
                layer.keys = None
                layer.values = None
            else:
                layer.state = None

    def reserve(self, new_tokens: int, context_length: int) -> None:
        """Make room for a new cached chunk by evicting the oldest KV states."""

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


class SwiGLU(nn.Module):
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.model_dim, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.model_dim, bias=False)
        self.w3 = nn.Linear(config.model_dim, config.ffn_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class DoubleGatedShortConv(nn.Module):
    """Double-gated causal depthwise convolution with a fixed short state."""

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

    def __call__(self, x: mx.array, cache: ConvCache | None = None) -> mx.array:
        first_gate, second_gate, values = mx.split(self.in_proj(x), 3, axis=-1)
        gated_values = first_gate * values
        state_length = self.kernel_size - 1

        if cache is None:
            convolution_input = mx.pad(
                gated_values,
                [(0, 0), (state_length, 0), (0, 0)],
            )
        else:
            state = cache.state
            if state is None:
                state = mx.zeros(
                    (x.shape[0], state_length, self.model_dim),
                    dtype=gated_values.dtype,
                )
            convolution_input = mx.concatenate((state, gated_values), axis=1)
            cache.state = convolution_input[:, -state_length:, :]

        convolved = self.conv(convolution_input)
        return self.out_proj(second_gate * convolved)


class GroupedQueryAttention(nn.Module):
    """Causal GQA with QK RMSNorm and full-head rotary positions."""

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        query_width = self.num_heads * self.head_dim
        key_value_width = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.model_dim, query_width, bias=False)
        self.k_proj = nn.Linear(config.model_dim, key_value_width, bias=False)
        self.v_proj = nn.Linear(config.model_dim, key_value_width, bias=False)
        self.out_proj = nn.Linear(query_width, config.model_dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.norm_eps)
        self.rope = nn.RoPE(
            config.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        offset: int = 0,
        cache: AttentionCache | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
        keys = self.k_proj(x).reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        values = self.v_proj(x).reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        )

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            if (cache.keys is None) != (cache.values is None):
                raise ValueError("Attention cache must contain both keys and values")
            if cache.keys is not None:
                keys = mx.concatenate((cache.keys, keys), axis=2)
                values = mx.concatenate((cache.values, values), axis=2)
            cache.keys = keys
            cache.values = values

        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask="causal",
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.out_proj(attended)


class HybridBlock(nn.Module):
    def __init__(self, config: HybridConfig, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.operator_norm = nn.RMSNorm(config.model_dim, eps=config.norm_eps)
        self.operator = (
            GroupedQueryAttention(config)
            if layer_type == ATTENTION
            else DoubleGatedShortConv(config)
        )
        self.ffn_norm = nn.RMSNorm(config.model_dim, eps=config.norm_eps)
        self.feed_forward = SwiGLU(config)

    def __call__(
        self,
        x: mx.array,
        *,
        offset: int = 0,
        cache: LayerCache | None = None,
    ) -> mx.array:
        normalized = self.operator_norm(x)
        if self.layer_type == ATTENTION:
            if cache is not None and not isinstance(cache, AttentionCache):
                raise TypeError("Attention block received a convolution cache")
            mixed = self.operator(normalized, offset=offset, cache=cache)
        else:
            if cache is not None and not isinstance(cache, ConvCache):
                raise TypeError("Convolution block received an attention cache")
            mixed = self.operator(normalized, cache=cache)
        hidden = x + mixed
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class HybridLM(nn.Module):
    """Randomly initialized decoder-only hybrid language model."""

    def __init__(self, config: HybridConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridConfig()
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.model_dim
        )
        self.blocks = [
            HybridBlock(self.config, layer_type)
            for layer_type in self.config.layer_schedule
        ]
        self.final_norm = nn.RMSNorm(self.config.model_dim, eps=self.config.norm_eps)

    def _validate_tokens(self, tokens: mx.array, offset: int) -> None:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, length]")
        if tokens.shape[0] <= 0 or tokens.shape[1] <= 0:
            raise ValueError("tokens must contain a non-empty batch and sequence")
        total_length = offset + tokens.shape[1]
        if total_length > self.config.context_length:
            raise ValueError(
                f"Sequence end {total_length} exceeds context {self.config.context_length}"
            )

    def _forward(
        self,
        tokens: mx.array,
        *,
        offset: int = 0,
        cache: HybridCache | None = None,
    ) -> mx.array:
        # A cached stream may use absolute RoPE positions beyond the resident
        # context window.  The caller bounds resident KV separately.
        self._validate_tokens(tokens, 0 if cache is not None else offset)
        hidden = self.token_embedding(tokens)
        for index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache.layers[index]
            hidden = block(hidden, offset=offset, cache=layer_cache)
        hidden = self.final_norm(hidden)
        # Tying is structural: there is no second output-head parameter tensor.
        return hidden @ self.token_embedding.weight.T

    def __call__(self, tokens: mx.array) -> mx.array:
        """Return full causal sequence logits for training or evaluation."""
        return self._forward(tokens)

    def init_cache(self, batch_size: int) -> HybridCache:
        """Create an empty, model-shaped incremental inference cache."""
        return HybridCache.empty(self.config, batch_size)

    def forward_cached(
        self,
        tokens: mx.array,
        cache: HybridCache | None = None,
    ) -> tuple[mx.array, HybridCache]:
        """Decode a new chunk and update both attention and conv state.

        ``tokens`` must contain only tokens not previously processed by this
        cache.  Passing multiple tokens is supported and applies a causal mask
        within the new chunk.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, length]")
        if cache is None:
            cache = self.init_cache(tokens.shape[0])
        if cache.batch_size != tokens.shape[0]:
            raise ValueError(
                f"Cache batch size {cache.batch_size} does not match tokens {tokens.shape[0]}"
            )
        if len(cache.layers) != len(self.blocks):
            raise ValueError("Cache layer count does not match model")

        cache.reserve(tokens.shape[1], self.config.context_length)
        logits = self._forward(tokens, offset=cache.offset, cache=cache)
        cache.offset += tokens.shape[1]
        cache.resident_tokens += tokens.shape[1]
        return logits, cache

    def parameter_count(self) -> int:
        """Count the model's actual trainable parameter tensors."""
        return sum(value.size for _, value in tree_flatten(self.parameters()))


def create_hybrid_model(config: HybridConfig | None = None) -> HybridLM:
    """Construct a fresh random model without reading configuration or weights."""
    return HybridLM(config)
