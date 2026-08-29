"""Backend-neutral configuration for tinydog's scratch hybrid language model."""

from __future__ import annotations

from dataclasses import dataclass


CONV = "conv"
ATTENTION = "attention"
DEFAULT_LAYER_SCHEDULE = (
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


@dataclass(frozen=True)
class HybridConfig:
    """Shape and numerical settings shared by every tensor backend."""

    vocab_size: int = 32_768
    context_length: int = 32_768
    model_dim: int = 1_024
    ffn_dim: int = 4_608
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 64
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    layer_schedule: tuple[str, ...] = DEFAULT_LAYER_SCHEDULE

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.model_dim <= 0 or self.ffn_dim <= 0:
            raise ValueError("model_dim and ffn_dim must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_attention_heads * self.head_dim != self.model_dim:
            raise ValueError("num_attention_heads * head_dim must equal model_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.conv_kernel_size < 2:
            raise ValueError("conv_kernel_size must be at least 2")
        if not self.layer_schedule:
            raise ValueError("layer_schedule must not be empty")
        invalid = set(self.layer_schedule) - {CONV, ATTENTION}
        if invalid:
            raise ValueError(f"Unsupported layer types: {sorted(invalid)}")

    @property
    def num_layers(self) -> int:
        return len(self.layer_schedule)

    @property
    def num_conv_layers(self) -> int:
        return self.layer_schedule.count(CONV)

    @property
    def num_attention_layers(self) -> int:
        return self.layer_schedule.count(ATTENTION)

    def expected_parameter_count(self) -> int:
        """Return the exact trainable tensor count implied by this config."""

        embedding = self.vocab_size * self.model_dim
        feed_forwards = self.num_layers * 3 * self.model_dim * self.ffn_dim
        convolutions = self.num_conv_layers * (
            3 * self.model_dim * self.model_dim
            + self.model_dim * self.model_dim
            + self.conv_kernel_size * self.model_dim
        )
        query_width = self.num_attention_heads * self.head_dim
        key_value_width = self.num_key_value_heads * self.head_dim
        attention = self.num_attention_layers * (
            self.model_dim * query_width
            + 2 * self.model_dim * key_value_width
            + query_width * self.model_dim
            + 2 * self.head_dim
        )
        block_norms = self.num_layers * 2 * self.model_dim
        final_norm = self.model_dim
        return (
            embedding
            + feed_forwards
            + convolutions
            + attention
            + block_norms
            + final_norm
        )


__all__ = ["ATTENTION", "CONV", "DEFAULT_LAYER_SCHEDULE", "HybridConfig"]
