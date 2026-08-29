"""Owner-controlled HybridLM foundation with a world-trainable floaty layer.

The security boundary in this module is structural:

* ``foundation`` contains the complete :class:`~mini_arithmetic.hybrid.HybridLM`
  and is always frozen by the wrapper.
* ``floaty`` contains only small parallel low-rank residual adapters at the two
  explicitly approved block attachment points.
* exported state is an in-memory tensor-only envelope.  This module never
  imports Python objects or loads arbitrary checkpoint formats.

The floaty adapters start as exact no-ops, so wrapping a foundation does not
change either full-sequence or cached inference until an adapter is trained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mini_arithmetic.hybrid import (
    ATTENTION,
    AttentionCache,
    ConvCache,
    HybridCache,
    HybridConfig,
    HybridLM,
)


OPERATOR = "operator"
FEED_FORWARD = "feed_forward"
APPROVED_ATTACHMENTS = (OPERATOR, FEED_FORWARD)
FLOATY_STATE_VERSION = 1


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _framed(hasher: Any, value: bytes) -> None:
    """Hash a byte string with an unambiguous fixed-width length prefix."""

    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _array_bytes(value: mx.array) -> memoryview:
    """Return the exact evaluated tensor representation without dtype casts."""

    # Viewing as bytes also supports bfloat16, which NumPy cannot expose using
    # the regular MLX buffer protocol.
    raw = mx.contiguous(value).view(mx.uint8).reshape(-1)
    mx.eval(raw)
    return memoryview(raw)


def _parameter_names(module: nn.Module) -> tuple[str, ...]:
    return tuple(sorted(name for name, _ in tree_flatten(module.parameters())))


def _parameter_count(module: nn.Module) -> int:
    return sum(value.size for _, value in tree_flatten(module.parameters()))


def _qualified(prefix: str, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}.{name}" for name in names)


def _safetensors_dtype(dtype: mx.Dtype) -> str:
    supported = {
        mx.float16: "F16",
        mx.bfloat16: "BF16",
        mx.float32: "F32",
    }
    try:
        return supported[dtype]
    except KeyError as error:
        raise ValueError(f"Unsupported floaty safetensors dtype: {dtype}") from error


@dataclass(frozen=True)
class FloatyConfig:
    """Strict, canonical configuration for the world-trainable adapters."""

    rank: int = 8
    alpha: float = 16.0
    attachments: tuple[str, ...] = APPROVED_ATTACHMENTS
    target_layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise ValueError("alpha must be a positive finite number")
        if not math.isfinite(float(self.alpha)) or self.alpha <= 0:
            raise ValueError("alpha must be a positive finite number")
        if type(self.attachments) is not tuple or not self.attachments:
            raise ValueError("attachments must be a non-empty tuple")
        if len(set(self.attachments)) != len(self.attachments):
            raise ValueError("attachments must not contain duplicates")
        if (
            tuple(name for name in APPROVED_ATTACHMENTS if name in self.attachments)
            != self.attachments
        ):
            raise ValueError(
                f"attachments must follow the approved order {APPROVED_ATTACHMENTS!r}"
            )
        invalid = set(self.attachments) - set(APPROVED_ATTACHMENTS)
        if invalid:
            raise ValueError(f"Unsupported floaty attachments: {sorted(invalid)}")
        if self.target_layers is not None:
            if type(self.target_layers) is not tuple:
                raise ValueError("target_layers must be a tuple or None")
            if not self.target_layers:
                raise ValueError("target_layers must not be empty")
            if any(type(index) is not int or index < 0 for index in self.target_layers):
                raise ValueError("target_layers must contain non-negative integers")
            if tuple(sorted(set(self.target_layers))) != self.target_layers:
                raise ValueError("target_layers must be unique and sorted")

    def resolved_target_layers(self, num_layers: int) -> tuple[int, ...]:
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        targets = (
            tuple(range(num_layers))
            if self.target_layers is None
            else self.target_layers
        )
        if targets[-1] >= num_layers:
            raise ValueError(
                f"target layer {targets[-1]} is outside a {num_layers}-layer foundation"
            )
        return targets

    def expected_parameter_count(self, *, model_dim: int, num_layers: int) -> int:
        if type(model_dim) is not int or model_dim <= 1:
            raise ValueError("model_dim must be an integer greater than one")
        if self.rank >= model_dim:
            raise ValueError("rank must be smaller than the foundation model dimension")
        targets = self.resolved_target_layers(num_layers)
        # Each attachment has A:[rank, model_dim] and B:[model_dim, rank].
        return len(targets) * len(self.attachments) * 2 * model_dim * self.rank


@dataclass(frozen=True)
class ParameterPartition:
    """Exact, wrapper-qualified parameter names on either side of the boundary."""

    foundation: tuple[str, ...]
    floaty: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return tuple(sorted((*self.foundation, *self.floaty)))


@dataclass(frozen=True)
class FloatyState:
    """Tensor-only community state bound to one exact foundation and config."""

    format_version: int
    foundation_hash: str
    config_hash: str
    tensors: dict[str, mx.array]


class LowRankAdapter(nn.Module):
    """A shape-generic LoRA-style residual branch initialized as a no-op."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if any(
            type(value) is not int or value <= 0
            for value in (input_dim, output_dim, rank)
        ):
            raise ValueError("adapter dimensions and rank must be positive integers")
        if rank >= min(input_dim, output_dim):
            raise ValueError("adapter rank must be smaller than both dimensions")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError("adapter alpha must be a positive finite number")
        if not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("adapter alpha must be a positive finite number")
        self.up = nn.Linear(input_dim, rank, bias=False)
        self.down = nn.Linear(rank, output_dim, bias=False)
        self.down.weight = mx.zeros_like(self.down.weight)
        self.scale = float(alpha) / rank

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.down(self.up(inputs)) * self.scale


class FloatyBlock(nn.Module):
    """Approved adapter branches for one foundation block."""

    def __init__(self, config: FloatyConfig, width: int, *, enabled: bool) -> None:
        super().__init__()
        self.operator = (
            LowRankAdapter(width, width, config.rank, config.alpha)
            if enabled and OPERATOR in config.attachments
            else None
        )
        self.feed_forward = (
            LowRankAdapter(width, width, config.rank, config.alpha)
            if enabled and FEED_FORWARD in config.attachments
            else None
        )


class CommunityFloatyLayer(nn.Module):
    """The complete world-owned parameter surface for one foundation."""

    def __init__(self, foundation_config: HybridConfig, config: FloatyConfig) -> None:
        super().__init__()
        targets = set(config.resolved_target_layers(foundation_config.num_layers))
        self.blocks = [
            FloatyBlock(config, foundation_config.model_dim, enabled=index in targets)
            for index in range(foundation_config.num_layers)
        ]


class _ComposedHybridBlock:
    """Non-parameter proxy preserving the normal HybridBlock call contract."""

    def __init__(self, foundation_block: nn.Module, floaty_block: FloatyBlock) -> None:
        # This intentionally is not an nn.Module.  The owning wrapper already
        # registers both parameter trees exactly once under foundation/floaty.
        self.foundation_block = foundation_block
        self.floaty_block = floaty_block
        self.layer_type = foundation_block.layer_type

    def __call__(
        self,
        inputs: mx.array,
        *,
        offset: int = 0,
        cache: AttentionCache | ConvCache | None = None,
    ) -> mx.array:
        operator_input = self.foundation_block.operator_norm(inputs)
        if self.layer_type == ATTENTION:
            if cache is not None and not isinstance(cache, AttentionCache):
                raise TypeError("Attention block received a convolution cache")
            mixed = self.foundation_block.operator(
                operator_input,
                offset=offset,
                cache=cache,
            )
        else:
            if cache is not None and not isinstance(cache, ConvCache):
                raise TypeError("Convolution block received an attention cache")
            mixed = self.foundation_block.operator(operator_input, cache=cache)
        if self.floaty_block.operator is not None:
            mixed = mixed + self.floaty_block.operator(operator_input)

        residual = inputs + mixed
        feed_forward_input = self.foundation_block.ffn_norm(residual)
        feed_forward = self.foundation_block.feed_forward(feed_forward_input)
        if self.floaty_block.feed_forward is not None:
            feed_forward = feed_forward + self.floaty_block.feed_forward(
                feed_forward_input
            )
        return residual + feed_forward


class FoundationFloatyLM(nn.Module):
    """Compose a frozen owner foundation with a trainable community layer."""

    def __init__(
        self, foundation: HybridLM, floaty_config: FloatyConfig | None = None
    ) -> None:
        super().__init__()
        if type(foundation) is not HybridLM:
            raise TypeError("foundation must be an exact HybridLM instance")
        self.foundation = foundation
        self.floaty_config = floaty_config or FloatyConfig()
        if type(self.floaty_config) is not FloatyConfig:
            raise TypeError("floaty_config must be an exact FloatyConfig instance")
        self.floaty_config.expected_parameter_count(
            model_dim=self.foundation.config.model_dim,
            num_layers=self.foundation.config.num_layers,
        )
        self.floaty = CommunityFloatyLayer(self.foundation.config, self.floaty_config)
        self.freeze_foundation()

    @property
    def config(self) -> HybridConfig:
        """Expose the foundation config for HybridLM-compatible consumers."""

        return self.foundation.config

    @property
    def token_embedding(self) -> nn.Embedding:
        """Expose the frozen shared embedding without registering it twice."""

        return self.foundation.token_embedding

    @property
    def final_norm(self) -> nn.RMSNorm:
        """Expose the frozen final norm without registering it twice."""

        return self.foundation.final_norm

    @property
    def blocks(self) -> tuple[_ComposedHybridBlock, ...]:
        """HybridBlock-compatible views that include floaty residuals.

        The multimodal timeline invokes thinker blocks directly from embedded
        inputs.  These proxies ensure that path and the token path compose the
        exact same floaty layer.
        """

        return tuple(
            _ComposedHybridBlock(foundation_block, floaty_block)
            for foundation_block, floaty_block in zip(
                self.foundation.blocks,
                self.floaty.blocks,
                strict=True,
            )
        )

    def freeze_foundation(self) -> None:
        """Reassert that only the floaty parameter tree can receive gradients."""

        self.foundation.freeze()
        self.floaty.unfreeze()
        self.assert_parameter_partition()

    def parameter_partition(self) -> ParameterPartition:
        foundation = _qualified("foundation", _parameter_names(self.foundation))
        floaty = _qualified("floaty", _parameter_names(self.floaty))
        return ParameterPartition(foundation=foundation, floaty=floaty)

    def foundation_parameter_names(self) -> tuple[str, ...]:
        return self.parameter_partition().foundation

    def floaty_parameter_names(self) -> tuple[str, ...]:
        return self.parameter_partition().floaty

    def exported_floaty_tensor_names(self) -> tuple[str, ...]:
        """Return the unqualified names used by :class:`FloatyState`."""

        return _parameter_names(self.floaty)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, _ in tree_flatten(self.trainable_parameters()))
        )

    def floaty_parameters(self) -> dict[str, Any]:
        """Return the only parameter tree an optimizer is allowed to update."""

        partition = self.assert_parameter_partition()
        parameters = self.floaty.trainable_parameters()
        trainable = _qualified(
            "floaty",
            tuple(sorted(name for name, _ in tree_flatten(parameters))),
        )
        if trainable != tuple(sorted(partition.floaty)):
            raise RuntimeError(
                "The floaty optimizer surface is frozen; call freeze_foundation() first"
            )
        return parameters

    def foundation_parameter_count(self) -> int:
        return _parameter_count(self.foundation)

    def floaty_parameter_count(self) -> int:
        return _parameter_count(self.floaty)

    def parameter_count(self) -> int:
        return self.foundation_parameter_count() + self.floaty_parameter_count()

    def assert_parameter_partition(self) -> ParameterPartition:
        """Fail closed if a foundation tensor becomes trainable or unclassified."""

        partition = self.parameter_partition()
        if set(partition.foundation) & set(partition.floaty):
            raise RuntimeError("Foundation and floaty parameter names overlap")
        actual = tuple(sorted(name for name, _ in tree_flatten(self.parameters())))
        if actual != partition.all:
            raise RuntimeError(
                "Model contains parameters outside the declared partition"
            )
        trainable = set(self.trainable_parameter_names())
        if not trainable.issubset(partition.floaty):
            raise RuntimeError("A parameter outside the floaty partition is trainable")
        if tree_flatten(self.foundation.trainable_parameters()):
            raise RuntimeError("Foundation parameters are not fully frozen")
        expected = self.floaty_config.expected_parameter_count(
            model_dim=self.foundation.config.model_dim,
            num_layers=self.foundation.config.num_layers,
        )
        if self.floaty_parameter_count() != expected:
            raise RuntimeError(
                f"Floaty parameter count {self.floaty_parameter_count()} does not match {expected}"
            )
        return partition

    def config_manifest(self) -> dict[str, Any]:
        """Return a defensive, canonical, plain-JSON config contract."""

        targets = self.floaty_config.resolved_target_layers(
            self.foundation.config.num_layers
        )
        foundation_config = asdict(self.foundation.config)
        foundation_config["layer_schedule"] = list(
            self.foundation.config.layer_schedule
        )
        tensors = [
            {
                "name": f"floaty.{name}",
                "shape": list(value.shape),
                "dtype": _safetensors_dtype(value.dtype),
            }
            for name, value in sorted(tree_flatten(self.floaty.parameters()))
        ]
        manifest = {
            "schema": "tinydog.foundation-floaty.config.v1",
            "foundation_type": "mini_arithmetic.hybrid.HybridLM",
            "foundation_config": foundation_config,
            "adapter_type": "parallel-low-rank-residual",
            "rank": self.floaty_config.rank,
            "alpha_hex": float(self.floaty_config.alpha).hex(),
            "attachments": list(self.floaty_config.attachments),
            "target_layers": list(targets),
            "tensors": tensors,
        }
        # The round trip both guarantees that callers receive only ordinary
        # JSON values and prevents them from mutating any internal structure.
        return json.loads(_canonical_json(manifest))

    def config_hash(self) -> str:
        """Return the exact SHA-256 of the resolved architecture/config contract."""

        return hashlib.sha256(_canonical_json(self.config_manifest())).hexdigest()

    def write_floaty_schema(
        self,
        path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
    ) -> Path:
        """Atomically write the canonical config contract.

        The default no-overwrite path publishes the completed temporary file
        using an atomic hard link, so readers never observe a partial schema
        and an existing signed contract is never silently replaced.
        """

        if type(overwrite) is not bool:
            raise TypeError("overwrite must be a boolean")
        destination = Path(path)
        parent = destination.parent
        if not parent.is_dir():
            raise FileNotFoundError(f"Schema parent directory does not exist: {parent}")
        payload = _canonical_json(self.config_manifest())
        descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)
                temporary.unlink()
            directory_descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def foundation_hash(self) -> str:
        """Hash exact foundation config, tensor names, shapes, dtypes, and bytes."""

        hasher = hashlib.sha256()
        _framed(hasher, b"tinydog.foundation.HybridLM.v1")
        foundation_config = asdict(self.foundation.config)
        foundation_config["layer_schedule"] = list(
            self.foundation.config.layer_schedule
        )
        _framed(hasher, _canonical_json(foundation_config))
        parameters = sorted(tree_flatten(self.foundation.parameters()))
        _framed(hasher, str(len(parameters)).encode("ascii"))
        for name, value in parameters:
            _framed(hasher, name.encode("utf-8"))
            _framed(hasher, str(value.dtype).encode("ascii"))
            _framed(hasher, _canonical_json({"shape": list(value.shape)}))
            _framed(hasher, _array_bytes(value))
        return hasher.hexdigest()

    def export_floaty_state(self) -> FloatyState:
        """Export only floaty tensors in a foundation/config-bound envelope."""

        self.assert_parameter_partition()
        tensors = {
            name: mx.array(value)
            for name, value in sorted(tree_flatten(self.floaty.parameters()))
        }
        mx.eval(tensors)
        return FloatyState(
            format_version=FLOATY_STATE_VERSION,
            foundation_hash=self.foundation_hash(),
            config_hash=self.config_hash(),
            tensors=tensors,
        )

    def import_floaty_state(self, state: FloatyState) -> None:
        """Import an exact tensor envelope; paths and serialized objects are rejected."""

        if type(state) is not FloatyState:
            raise TypeError("state must be an exact FloatyState tensor envelope")
        if (
            type(state.format_version) is not int
            or state.format_version != FLOATY_STATE_VERSION
        ):
            raise ValueError(
                f"Unsupported floaty state version: {state.format_version!r}"
            )
        if state.foundation_hash != self.foundation_hash():
            raise ValueError("Floaty state foundation hash does not match")
        if state.config_hash != self.config_hash():
            raise ValueError("Floaty state config hash does not match")
        if type(state.tensors) is not dict:
            raise TypeError("Floaty state tensors must be a plain dict")
        if any(type(name) is not str for name in state.tensors):
            raise TypeError("Floaty tensor names must be exact strings")

        expected = dict(tree_flatten(self.floaty.parameters()))
        if set(state.tensors) != set(expected):
            missing = sorted(set(expected) - set(state.tensors))
            extra = sorted(set(state.tensors) - set(expected))
            raise ValueError(
                f"Floaty tensor keys mismatch; missing={missing}, extra={extra}"
            )

        validated: list[tuple[str, mx.array]] = []
        for name in sorted(expected):
            value = state.tensors[name]
            template = expected[name]
            if type(value) is not mx.array:
                raise TypeError(
                    f"Floaty tensor {name!r} must be an exact mlx.core.array"
                )
            if value.shape != template.shape:
                raise ValueError(
                    f"Floaty tensor {name!r} shape {value.shape} does not match {template.shape}"
                )
            if value.dtype != template.dtype:
                raise ValueError(
                    f"Floaty tensor {name!r} dtype {value.dtype} does not match {template.dtype}"
                )
            if not mx.isfinite(value).all().item():
                raise ValueError(f"Floaty tensor {name!r} contains non-finite values")
            validated.append((name, mx.array(value)))

        self.floaty.update(tree_unflatten(validated), strict=True)
        mx.eval(self.floaty.parameters())
        self.assert_parameter_partition()

    def _forward(
        self,
        tokens: mx.array,
        *,
        offset: int = 0,
        cache: HybridCache | None = None,
    ) -> mx.array:
        self.foundation._validate_tokens(tokens, 0 if cache is not None else offset)
        hidden = self.token_embedding(tokens)
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                offset=offset,
                cache=None if cache is None else cache.layers[index],
            )

        hidden = self.final_norm(hidden)
        return hidden @ self.token_embedding.weight.T

    def __call__(self, tokens: mx.array) -> mx.array:
        """Return full causal logits from foundation plus floaty adapters."""

        return self._forward(tokens)

    def init_cache(self, batch_size: int) -> HybridCache:
        return self.foundation.init_cache(batch_size)

    def forward_cached(
        self,
        tokens: mx.array,
        cache: HybridCache | None = None,
    ) -> tuple[mx.array, HybridCache]:
        """Decode a new chunk with the same composed model and foundation cache."""

        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, length]")
        if cache is None:
            cache = self.init_cache(tokens.shape[0])
        if cache.batch_size != tokens.shape[0]:
            raise ValueError(
                f"Cache batch size {cache.batch_size} does not match tokens {tokens.shape[0]}"
            )
        if len(cache.layers) != len(self.foundation.blocks):
            raise ValueError("Cache layer count does not match model")

        cache.reserve(tokens.shape[1], self.foundation.config.context_length)
        logits = self._forward(tokens, offset=cache.offset, cache=cache)
        cache.offset += tokens.shape[1]
        cache.resident_tokens += tokens.shape[1]
        return logits, cache
