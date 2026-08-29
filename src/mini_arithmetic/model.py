"""A roughly 100M-parameter decoder-only Transformer implemented in MLX."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 264
    context_length: int = 512
    num_layers: int = 14
    model_dim: int = 768
    num_heads: int = 12
    ffn_dim: int = 2048
    norm_eps: float = 1e-5

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.model_dim % config.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.qkv = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, index].transpose(0, 2, 1, 3) for index in range(3)]
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask="causal"
        )
        return self.output(attended.transpose(0, 2, 1, 3).reshape(batch, length, width))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.model_dim, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.model_dim, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.model_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.model_dim, eps=config.norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.model_dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class ArithmeticLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.model_dim)
        self.blocks = [TransformerBlock(config) for _ in range(config.num_layers)]
        self.final_norm = nn.RMSNorm(config.model_dim, eps=config.norm_eps)

    def __call__(self, tokens: mx.array) -> mx.array:
        length = tokens.shape[1]
        if length > self.config.context_length:
            raise ValueError(
                f"Sequence length {length} exceeds context {self.config.context_length}"
            )
        x = self.token_embedding(tokens) + self.position_embedding(mx.arange(length))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return x @ self.token_embedding.weight.T

    def parameter_count(self) -> int:
        return sum(value.size for _, value in tree_flatten(self.parameters()))


def create_model(
    config_path: str | Path, weights: str | Path | None = None
) -> ArithmeticLM:
    config = ModelConfig.from_json(config_path)
    model = ArithmeticLM(config)
    if weights is not None:
        model.load_weights(str(weights))
    mx.eval(model.parameters())
    return model
