"""Private conversational adapters for the portable PyTorch backend.

The portable foundation is immutable.  Personal learning is confined to two
small, parallel low-rank residual branches per hybrid block: one beside the
token-mixing operator and one beside the feed-forward network.  Adapter files
contain the complete cumulative adapter state and no executable objects.

This module is deliberately independent of the Apple-only compute backend so
it can be imported and exercised on Windows and Linux.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file

from .hybrid_config import ATTENTION, HybridConfig
from .personal_training import (
    PERSONAL_MAX_REPLAY_EXAMPLES,
    PERSONAL_MAX_RETENTION_EXAMPLES,
    PERSONAL_MAX_TRAINING_CONTEXT_TOKENS,
    DialogueExample,
    PersonalAdapterArtifact,
    PersonalAdapterCandidate,
    PersonalAdapterTrainingRequest,
    PersonalRuntimeIdentity,
    PersonalTrainingError,
    compact_personal_dialogue_prompt,
    encode_personal_dialogue_training_target,
    personal_dialogue_target_fits,
)
from .personal_foundation import (
    PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES,
    PersonalFoundationArtifact,
    PersonalFoundationCandidate,
    PersonalFoundationError,
    PersonalFoundationIdentity,
    PersonalFoundationSource,
    PersonalFoundationTrainingRequest,
    personal_foundation_backend_contract_sha256,
    personal_foundation_bundle_sha256,
    validate_foundation_metrics,
    validate_foundation_request,
)
from .portable import BackendSelection
from .portable_bundle import (
    PORTABLE_STATE_FORMAT,
    PortableBundle,
    load_portable_bundle,
    publish_conversational_foundation_bundle,
)
from .scratch_tokenizer import ROLE_NAMES, ScratchTokenizer, sanitize_generated_text
from .torch_hybrid import AttentionCache, ConvCache, HybridCache, HybridLM


TORCH_PERSONAL_BACKEND = "torch"
TORCH_PERSONAL_STATE_FORMAT = "tinydog-torch-personal-adapter-v1"
TORCH_FOUNDATION_BASE_ID = "foundation-base"
TORCH_FOUNDATION_RECIPE_FORMAT = "tinydog-torch-conversation-foundation-v1"
_APPROVED_ATTACHMENTS = ("operator", "feed_forward")
_SHA256_LENGTH = 64
_VERIFIED_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True)
class TorchPersonalConfig:
    """Small deterministic optimizer and adapter contract."""

    rank: int = 8
    alpha: float = 16.0
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    steps: int = 12
    grad_clip: float = 1.0
    seed: int = 17
    training_context_length: int = 256
    max_tokens: int = 32

    def validate(self, foundation: HybridConfig) -> None:
        if type(self.rank) is not int or not 0 < self.rank < foundation.model_dim:
            raise ValueError("adapter rank must be smaller than the model dimension")
        for name in ("alpha", "learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.alpha <= 0.0 or self.learning_rate <= 0.0 or self.grad_clip <= 0.0:
            raise ValueError("alpha, learning_rate, and grad_clip must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("training_context_length", "max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training_context_length < 8:
            raise ValueError("training_context_length must be at least eight")
        if self.training_context_length > PERSONAL_MAX_TRAINING_CONTEXT_TOKENS:
            raise ValueError(
                "training_context_length exceeds the shared personal-training limit"
            )


@dataclass(frozen=True)
class TorchFoundationConfig:
    """Bounded, deterministic full-foundation conversational recipe."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 1
    grad_clip: float = 1.0
    seed: int = 29
    batch_size: int = 1
    context_length: int = 128
    disk_reserve_bytes: int = 64 * 1024 * 1024

    def validate(self) -> None:
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0.0 or self.grad_clip <= 0.0:
            raise ValueError("learning_rate and grad_clip must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if type(self.epochs) is not int or not 1 <= self.epochs <= 4:
            raise ValueError("epochs must be between one and four")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.batch_size != 1:
            raise ValueError("direct foundation training batch_size must remain one")
        if (
            type(self.context_length) is not int
            or not 8 <= self.context_length <= PERSONAL_MAX_TRAINING_CONTEXT_TOKENS
        ):
            raise ValueError("context_length exceeds the bounded training window")
        if type(self.disk_reserve_bytes) is not int or self.disk_reserve_bytes < 0:
            raise ValueError("disk_reserve_bytes must be a nonnegative integer")


def _default_torch_foundation_config(
    foundation: HybridConfig,
) -> TorchFoundationConfig:
    """Fit the zero-config recipe to a checksum-verified parent context."""

    default = TorchFoundationConfig()
    return TorchFoundationConfig(
        learning_rate=default.learning_rate,
        weight_decay=default.weight_decay,
        epochs=default.epochs,
        grad_clip=default.grad_clip,
        seed=default.seed,
        batch_size=default.batch_size,
        context_length=min(default.context_length, foundation.context_length),
        disk_reserve_bytes=default.disk_reserve_bytes,
    )


class _LowRankBranch(nn.Module):
    def __init__(self, width: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.up = nn.Linear(width, rank, bias=False)
        self.down = nn.Linear(rank, width, bias=False)
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.weight)
        self.scale = float(alpha) / rank

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(inputs)) * self.scale


class _PersonalBlock(nn.Module):
    def __init__(self, width: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.operator = _LowRankBranch(width, rank, alpha)
        self.feed_forward = _LowRankBranch(width, rank, alpha)


class TorchPersonalAdapter(nn.Module):
    """The complete and only mutable parameter surface."""

    def __init__(self, foundation: HybridConfig, config: TorchPersonalConfig) -> None:
        super().__init__()
        config.validate(foundation)
        self.blocks = nn.ModuleList(
            _PersonalBlock(foundation.model_dim, config.rank, config.alpha)
            for _ in range(foundation.num_layers)
        )


class TorchPersonalModel(nn.Module):
    """Compose a frozen ``HybridLM`` with an optional personal adapter."""

    def __init__(
        self,
        foundation: HybridLM,
        adapter: TorchPersonalAdapter | None,
    ) -> None:
        super().__init__()
        if type(foundation) is not HybridLM:
            raise TypeError("foundation must be an exact PyTorch HybridLM")
        self.foundation = foundation
        self.adapter = adapter
        for parameter in self.foundation.parameters():
            parameter.requires_grad_(False)

    @property
    def config(self) -> HybridConfig:
        return self.foundation.config

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.adapter is None:
            return ()
        return tuple(self.adapter.parameters())

    def assert_parameter_boundary(self) -> None:
        if any(parameter.requires_grad for parameter in self.foundation.parameters()):
            raise PersonalTrainingError("a foundation tensor became trainable")
        if self.adapter is not None and not all(
            parameter.requires_grad for parameter in self.adapter.parameters()
        ):
            raise PersonalTrainingError("the personal adapter is not fully trainable")

    def _block(
        self,
        hidden: torch.Tensor,
        index: int,
        *,
        offset: int,
        cache: AttentionCache | ConvCache | None,
    ) -> torch.Tensor:
        block = self.foundation.blocks[index]
        operator_input = block.operator_norm(hidden)
        if block.layer_type == ATTENTION:
            if cache is not None and not isinstance(cache, AttentionCache):
                raise TypeError("attention block received a convolution cache")
            mixed = block.operator(operator_input, offset=offset, cache=cache)
        else:
            if cache is not None and not isinstance(cache, ConvCache):
                raise TypeError("convolution block received an attention cache")
            mixed = block.operator(operator_input, cache=cache)
        if self.adapter is not None:
            mixed = mixed + self.adapter.blocks[index].operator(operator_input)
        residual = hidden + mixed
        ffn_input = block.ffn_norm(residual)
        feed_forward = block.feed_forward(ffn_input)
        if self.adapter is not None:
            feed_forward = feed_forward + self.adapter.blocks[index].feed_forward(
                ffn_input
            )
        return residual + feed_forward

    def _forward(
        self,
        tokens: torch.Tensor,
        *,
        offset: int = 0,
        cache: HybridCache | None = None,
    ) -> torch.Tensor:
        self.foundation._validate_tokens(tokens, 0 if cache is not None else offset)
        hidden = self.foundation.token_embedding(tokens)
        for index in range(len(self.foundation.blocks)):
            layer_cache = None if cache is None else cache.layers[index]
            hidden = self._block(hidden, index, offset=offset, cache=layer_cache)
        hidden = self.foundation.final_norm(hidden)
        return functional.linear(hidden, self.foundation.token_embedding.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._forward(tokens)

    def forward_cached(
        self,
        tokens: torch.Tensor,
        cache: HybridCache | None = None,
    ) -> tuple[torch.Tensor, HybridCache]:
        if tokens.ndim != 2 or not tokens.shape[0] or not tokens.shape[1]:
            raise ValueError("tokens must have non-empty shape [batch, length]")
        if cache is None:
            cache = self.foundation.init_cache(tokens.shape[0])
        if cache.batch_size != tokens.shape[0]:
            raise ValueError("cache batch size does not match tokens")
        if len(cache.layers) != len(self.foundation.blocks):
            raise ValueError("cache layer count does not match model")
        cache.reserve(tokens.shape[1], self.config.context_length)
        logits = self._forward(tokens, offset=cache.offset, cache=cache)
        cache.offset += tokens.shape[1]
        cache.resident_tokens += tokens.shape[1]
        return logits, cache


@dataclass(frozen=True)
class _PreparedAdapter:
    adapter_id: str
    foundation_sha256: str
    adapter_schema_sha256: str
    model: TorchPersonalModel


@dataclass(frozen=True)
class _PreparedTorchFoundation:
    source: PersonalFoundationSource
    model: TorchPersonalModel


@dataclass(frozen=True)
class _TokenBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _tensor_identity_versions(module: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Cheap mutation boundary for a strictly held, immutable module object."""

    tensors = [
        (f"parameter:{name}", value) for name, value in module.named_parameters()
    ] + [(f"buffer:{name}", value) for name, value in module.named_buffers()]
    return tuple(
        (
            name,
            id(value),
            int(value.data_ptr()),
            int(value._version),
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
            bool(value.requires_grad),
        )
        for name, value in tensors
    )


def _verified_bundle(bundle: PortableBundle) -> PortableBundle:
    if not isinstance(bundle, PortableBundle):
        raise TypeError("bundle must be a verified PortableBundle")
    # Verification is repeated because a caller may have held the object while
    # another process changed files beneath it.
    return load_portable_bundle(bundle.root)


def _resolve_device(selected: str | torch.device) -> torch.device:
    if isinstance(selected, torch.device):
        device = selected
    elif selected == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(selected, str) and selected:
        device = torch.device(selected)
    else:
        raise ValueError("device must be 'auto' or a PyTorch device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PersonalTrainingError("the requested CUDA device is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise PersonalTrainingError("the requested MPS device is unavailable")
    if device.type == "xpu" and not (
        hasattr(torch, "xpu") and torch.xpu.is_available()
    ):
        raise PersonalTrainingError("the requested XPU device is unavailable")
    return device


def _selection_device(selection: BackendSelection) -> str:
    if not isinstance(selection, BackendSelection):
        raise TypeError("selection must be a BackendSelection")
    if not selection.ready or selection.backend != TORCH_PERSONAL_BACKEND:
        raise PersonalTrainingError("backend selection is not ready for PyTorch")
    if selection.accelerator == "rocm":
        return "cuda"
    if selection.accelerator in {"cuda", "xpu", "mps", "cpu"}:
        return selection.accelerator
    raise PersonalTrainingError("backend selection has no usable PyTorch device")


def _new_foundation(bundle: PortableBundle, device: torch.device) -> HybridLM:
    if _sha256_file(bundle.weights_path) != bundle.manifest["weights_sha256"]:
        raise PersonalTrainingError("the portable foundation checksum changed")
    state = load_file(str(bundle.weights_path), device="cpu")
    model = HybridLM(bundle.config)
    result = model.load_canonical_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise PersonalTrainingError("portable foundation tensors are incompatible")
    model.to(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _new_adapter(
    foundation: HybridConfig,
    config: TorchPersonalConfig,
    *,
    device: torch.device,
) -> TorchPersonalAdapter:
    adapter = TorchPersonalAdapter(foundation, config)
    return adapter.to(device=device, dtype=torch.float32)


def _adapter_schema(
    foundation: HybridConfig,
    config: TorchPersonalConfig,
) -> tuple[str, tuple[str, ...]]:
    # Isolate initialization so schema inspection cannot perturb training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        adapter = TorchPersonalAdapter(foundation, config)
    state = adapter.state_dict()
    names = tuple(sorted(state))
    payload = {
        "format": TORCH_PERSONAL_STATE_FORMAT,
        "attachments": list(_APPROVED_ATTACHMENTS),
        "adapter": {"rank": config.rank, "alpha": float(config.alpha)},
        "foundation": {
            "model_dim": foundation.model_dim,
            "num_layers": foundation.num_layers,
            "layer_schedule": list(foundation.layer_schedule),
        },
        "tensors": [
            {
                "name": name,
                "shape": list(state[name].shape),
                "dtype": "float32",
            }
            for name in names
        ],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest(), names


def _load_adapter_state(
    adapter: TorchPersonalAdapter,
    path: Path,
    expected_names: tuple[str, ...],
) -> None:
    if not path.is_file() or path.is_symlink():
        raise PersonalTrainingError("personal adapter is not a safe regular file")
    state = load_file(str(path), device="cpu")
    if tuple(sorted(state)) != expected_names:
        raise PersonalTrainingError("personal adapter tensor names are incompatible")
    expected = adapter.state_dict()
    converted: dict[str, torch.Tensor] = {}
    for name in expected_names:
        value = state[name]
        if tuple(value.shape) != tuple(expected[name].shape):
            raise PersonalTrainingError(
                f"personal adapter tensor {name!r} has wrong shape"
            )
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise PersonalTrainingError(f"personal adapter tensor {name!r} is invalid")
        converted[name] = value.to(dtype=torch.float32)
    result = adapter.load_state_dict(converted, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise PersonalTrainingError("personal adapter state is incomplete")


def _typed_content_sha256(example: DialogueExample) -> str:
    prompt = example.prompt.replace("\r\n", "\n").replace("\r", "\n").strip()
    response = example.human_response.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(
        f"typed-dialogue-v1\0{prompt}\0{response}".encode("utf-8")
    ).hexdigest()


def _validate_request(
    request: PersonalAdapterTrainingRequest,
    *,
    foundation_sha256: str,
    schema_sha256: str,
) -> None:
    if request.backend != TORCH_PERSONAL_BACKEND:
        raise PersonalTrainingError(
            "personal training request selected another backend"
        )
    if request.foundation_sha256 != foundation_sha256:
        raise PersonalTrainingError("personal training request foundation changed")
    if request.adapter_schema_sha256 != schema_sha256:
        raise PersonalTrainingError("personal training request adapter schema changed")
    if len(request.training_examples) != 4 or len(request.validation_examples) != 1:
        raise PersonalTrainingError(
            "PyTorch personal training requires exactly four plus one"
        )
    if len(request.replay_examples) > PERSONAL_MAX_REPLAY_EXAMPLES:
        raise PersonalTrainingError("PyTorch personal replay exceeds its hard bound")
    if len(request.retention_examples) > PERSONAL_MAX_RETENTION_EXAMPLES:
        raise PersonalTrainingError(
            "PyTorch personal retention set exceeds its hard bound"
        )
    examples = (
        request.training_examples
        + request.validation_examples
        + request.replay_examples
        + request.retention_examples
    )
    if len({example.lesson_id for example in examples}) != len(examples):
        raise PersonalTrainingError("personal training examples must be distinct")
    if len({example.content_sha256 for example in examples}) != len(examples):
        raise PersonalTrainingError("personal dialogue content must be distinct")
    for example in examples:
        if (
            type(example.lesson_id) is not int
            or example.lesson_id <= 0
            or not example.session_id
            or not example.prompt.strip()
            or not example.human_response.strip()
            or example.content_sha256 != _typed_content_sha256(example)
        ):
            raise PersonalTrainingError(
                "personal dialogue example provenance is invalid"
            )
    if (request.parent_adapter_id is None) != (request.parent_adapter_path is None):
        raise PersonalTrainingError(
            "parent adapter identity and path must appear together"
        )
    work_dir = request.work_dir.resolve()
    if not work_dir.is_dir() or request.work_dir.is_symlink():
        raise PersonalTrainingError("personal trainer work directory is unsafe")


def _raise_if_cancelled(request: PersonalAdapterTrainingRequest) -> None:
    if request.cancellation_event is not None and request.cancellation_event.is_set():
        raise PersonalTrainingError("personal training was cancelled during shutdown")


def _encode_example(
    tokenizer: ScratchTokenizer,
    example: DialogueExample,
    context_length: int,
) -> tuple[list[int], list[float]]:
    return encode_personal_dialogue_training_target(
        tokenizer,
        example.prompt,
        example.human_response,
        context_length=context_length,
    )


def _batch(
    tokenizer: ScratchTokenizer,
    examples: Sequence[DialogueExample],
    context_length: int,
    device: torch.device,
) -> _TokenBatch:
    encoded = [
        _encode_example(tokenizer, example, context_length) for example in examples
    ]
    width = max(len(tokens) - 1 for tokens, _ in encoded)
    inputs = torch.full(
        (len(encoded), width), tokenizer.pad_id, dtype=torch.long, device=device
    )
    targets = torch.full(
        (len(encoded), width), tokenizer.pad_id, dtype=torch.long, device=device
    )
    mask = torch.zeros((len(encoded), width), dtype=torch.float32, device=device)
    for row, (tokens, weights) in enumerate(encoded):
        length = len(tokens) - 1
        inputs[row, :length] = torch.tensor(
            tokens[:-1], dtype=torch.long, device=device
        )
        targets[row, :length] = torch.tensor(
            tokens[1:], dtype=torch.long, device=device
        )
        mask[row, :length] = torch.tensor(weights, dtype=torch.float32, device=device)
    return _TokenBatch(inputs, targets, mask)


def _text_logit_bias(
    vocab_size: int,
    blocked_ids: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if (
        type(vocab_size) is not int
        or vocab_size <= 0
        or not blocked_ids
        or min(blocked_ids) < 0
        or max(blocked_ids) >= vocab_size
        or len(set(blocked_ids)) != len(blocked_ids)
    ):
        raise PersonalTrainingError("inactive text-token mask is incompatible")
    bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    indices = torch.tensor(blocked_ids, dtype=torch.long, device=device)
    return bias.index_fill(0, indices, -torch.inf)


def _mask_inactive_text_logits(
    logits: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    if (
        logit_bias.ndim != 1
        or logit_bias.shape[0] != logits.shape[-1]
        or logit_bias.device != logits.device
        or not logit_bias.dtype.is_floating_point
    ):
        raise PersonalTrainingError("inactive text-token bias is incompatible")
    return logits + logit_bias


def _masked_loss(
    model: TorchPersonalModel,
    batch: _TokenBatch,
    inactive_logit_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    supervised_targets = batch.targets[batch.mask > 0]
    logits = model(batch.inputs).float()
    if inactive_logit_bias is not None:
        # MPS in Torch 2.5 does not implement aten::isneginf.out. This bias is
        # contractually either 0 or -inf, so a simple comparison is equivalent
        # and portable across CPU, CUDA, and MPS.
        if (inactive_logit_bias[supervised_targets] < 0).any().item():
            raise PersonalTrainingError(
                "personal target uses an inactive text-token ID"
            )
        logits = _mask_inactive_text_logits(logits, inactive_logit_bias)
    per_token = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.targets.reshape(-1),
        reduction="none",
    ).reshape_as(batch.mask)
    denominator = batch.mask.sum()
    if denominator.item() <= 0:
        raise PersonalTrainingError("personal batch has no supervised targets")
    return (per_token * batch.mask).sum() / denominator


def _finite_loss(
    model: TorchPersonalModel,
    batch: _TokenBatch,
    inactive_logit_bias: torch.Tensor | None = None,
) -> float:
    model.eval()
    with torch.inference_mode():
        value = float(_masked_loss(model, batch, inactive_logit_bias).detach().cpu())
    if not math.isfinite(value) or value < 0.0:
        raise PersonalTrainingError("personal adapter produced a non-finite loss")
    return value


def _clone_adapter_state(adapter: TorchPersonalAdapter) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().float().contiguous().clone()
        for name, value in adapter.state_dict().items()
    }


def _save_adapter(path: Path, adapter: TorchPersonalAdapter) -> str:
    state = _clone_adapter_state(adapter)
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise PersonalTrainingError("refusing to save a non-finite personal adapter")
    descriptor: str | None = None
    try:
        file_descriptor, descriptor = tempfile.mkstemp(
            prefix=".candidate-", suffix=".safetensors", dir=path.parent
        )
        os.close(file_descriptor)
        save_file(state, descriptor, metadata={"format": TORCH_PERSONAL_STATE_FORMAT})
        os.chmod(descriptor, 0o600)
        os.replace(descriptor, path)
        descriptor = None
    finally:
        if descriptor is not None:
            Path(descriptor).unlink(missing_ok=True)
    return _sha256_file(path)


def _torch_foundation_identity(
    bundle: PortableBundle,
    foundation_id: str,
) -> PersonalFoundationIdentity:
    verified = load_portable_bundle(bundle.root)
    identity = PersonalFoundationIdentity(
        backend=TORCH_PERSONAL_BACKEND,
        foundation_id=foundation_id,
        bundle_sha256=personal_foundation_bundle_sha256(verified.root),
        weights_sha256=str(verified.manifest["weights_sha256"]),
        config_sha256=str(verified.manifest["config_sha256"]),
        tokenizer_sha256=str(verified.manifest["tokenizer_sha256"]),
        backend_contract_sha256=personal_foundation_backend_contract_sha256(
            TORCH_PERSONAL_BACKEND
        ),
    )
    identity.validate()
    return identity


def _raise_if_foundation_cancelled(
    request: PersonalFoundationTrainingRequest,
) -> None:
    if request.cancellation_event is not None and request.cancellation_event.is_set():
        raise PersonalFoundationError(
            "direct foundation training was cancelled during shutdown"
        )


def _validate_torch_foundation_request(
    request: PersonalFoundationTrainingRequest,
    *,
    expected_config_sha256: str,
    expected_tokenizer_sha256: str,
    context_length: int,
    tokenizer: ScratchTokenizer,
) -> PortableBundle:
    if type(request) is not PersonalFoundationTrainingRequest:
        raise TypeError("request must be a PersonalFoundationTrainingRequest")
    validate_foundation_request(request)
    if request.parent.identity.backend != TORCH_PERSONAL_BACKEND:
        raise PersonalFoundationError("foundation request selected another backend")
    if (
        request.parent.identity.backend_contract_sha256
        != personal_foundation_backend_contract_sha256(TORCH_PERSONAL_BACKEND)
    ):
        raise PersonalFoundationError("foundation backend contract changed")
    if request.work_dir.is_symlink():
        raise PersonalFoundationError("foundation work directory must not be a symlink")
    parent = load_portable_bundle(request.parent.bundle_path)
    actual_identity = _torch_foundation_identity(
        parent,
        request.parent.identity.foundation_id,
    )
    if actual_identity != request.parent.identity:
        raise PersonalFoundationError("active parent foundation identity changed")
    if (
        actual_identity.config_sha256 != expected_config_sha256
        or actual_identity.tokenizer_sha256 != expected_tokenizer_sha256
    ):
        raise PersonalFoundationError(
            "active parent foundation lineage is incompatible"
        )
    if context_length > parent.config.context_length:
        raise PersonalFoundationError(
            "foundation training context exceeds the portable model context"
        )
    examples = request.new_examples + request.replay_examples
    if len({example.content_sha256 for example in examples}) != len(examples):
        raise PersonalFoundationError("foundation dialogue content must be distinct")
    for example in examples:
        if example.content_sha256 != _typed_content_sha256(example):
            raise PersonalFoundationError("foundation dialogue provenance is invalid")
        if not personal_dialogue_target_fits(
            tokenizer,
            example.prompt,
            example.human_response,
            context_length=context_length,
        ):
            raise PersonalFoundationError(
                "foundation dialogue target exceeds the bounded context"
            )
    if request.monitoring_examples[0] not in request.new_examples:
        raise PersonalFoundationError(
            "foundation monitoring example must exactly match a new example"
        )
    return parent


def _torch_foundation_recipe_sha256(config: TorchFoundationConfig) -> str:
    payload = {
        "assistant_only_targets": True,
        "backend": TORCH_PERSONAL_BACKEND,
        "batch_size": config.batch_size,
        "context_length": config.context_length,
        "epochs": config.epochs,
        "format": TORCH_FOUNDATION_RECIPE_FORMAT,
        "grad_clip": float(config.grad_clip),
        "inactive_text_logits_masked": True,
        "learning_rate": float(config.learning_rate),
        "new_examples_optimized": 5,
        "optimizer": "adamw",
        "optimizer_state_continuity": False,
        "weight_decay": float(config.weight_decay),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _foundation_example_group_sha256(
    label: str,
    examples: Sequence[DialogueExample],
) -> str:
    digest = hashlib.sha256(
        f"tinydog-direct-foundation-dialogue-v1\0{label}\0".encode("ascii")
    )
    for example in examples:
        digest.update(str(example.lesson_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(example.content_sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _foundation_source_sha256(
    request: PersonalFoundationTrainingRequest,
) -> dict[str, str]:
    sources = {
        "new_conversation": _foundation_example_group_sha256(
            "new", request.new_examples
        )
    }
    if request.replay_examples:
        sources["bounded_replay"] = _foundation_example_group_sha256(
            "replay", request.replay_examples
        )
    return sources


def _foundation_disk_preflight(
    work_dir: Path,
    parent: PortableBundle,
    config: TorchFoundationConfig,
) -> None:
    # The trainer briefly owns a loose canonical state plus the atomically
    # staged complete bundle. Optimizer state stays in memory and is never
    # pickled or written into the artifact.
    required = parent.weights_path.stat().st_size * 2 + config.disk_reserve_bytes
    try:
        free = shutil.disk_usage(work_dir).free
    except OSError as error:
        raise PersonalFoundationError(
            "could not verify disk space for foundation candidate"
        ) from error
    if free < required:
        raise PersonalFoundationError(
            f"direct foundation training needs at least {required} free bytes "
            "for an atomic candidate"
        )


def _save_torch_foundation_weights(
    path: Path,
    model: HybridLM,
    *,
    parameter_dtype: str,
) -> None:
    output_dtype = torch.bfloat16 if parameter_dtype == "bfloat16" else torch.float32
    state: dict[str, torch.Tensor] = {}
    for name, value in model.canonical_state_dict().items():
        exported = value.detach().to(device="cpu", dtype=output_dtype).contiguous()
        if not torch.isfinite(exported.float()).all().item():
            raise PersonalFoundationError(
                f"foundation candidate tensor {name!r} is non-finite"
            )
        state[name] = exported
    descriptor: str | None = None
    try:
        file_descriptor, descriptor = tempfile.mkstemp(
            prefix=".foundation-weights-",
            suffix=".safetensors",
            dir=path.parent,
        )
        os.close(file_descriptor)
        save_file(
            state,
            descriptor,
            metadata={"format": PORTABLE_STATE_FORMAT},
        )
        os.chmod(descriptor, 0o600)
        os.replace(descriptor, path)
        descriptor = None
    finally:
        if descriptor is not None:
            Path(descriptor).unlink(missing_ok=True)


def _clear_torch_device_cache(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def _torch_foundation_forward_smoke(
    model: TorchPersonalModel, tokenizer: ScratchTokenizer
) -> None:
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        tokens = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
        logits = model(tokens)
        if (
            tuple(logits.shape) != (1, 1, model.config.vocab_size)
            or not torch.isfinite(logits.float()).all().item()
        ):
            raise PersonalFoundationError(
                "foundation candidate failed forward validation"
            )


class TorchPersonalTrainer:
    """Deterministic adapter-only AdamW trainer with a private holdout gate."""

    def __init__(
        self,
        bundle: PortableBundle,
        *,
        device: str | torch.device = "auto",
        config: TorchPersonalConfig | None = None,
        _verification_token: object | None = None,
    ) -> None:
        verified = (
            bundle
            if _verification_token is _VERIFIED_CONSTRUCTION_TOKEN
            else _verified_bundle(bundle)
        )
        self._bundle_root = verified.root
        self._foundation_sha256 = str(verified.manifest["weights_sha256"])
        self._foundation_config = verified.config
        self._tokenizer = verified.tokenizer
        self._inactive_text_ids = verified.tokenizer.inactive_text_token_ids
        self._device = _resolve_device(device)
        self._inactive_logit_bias = _text_logit_bias(
            verified.config.vocab_size,
            self._inactive_text_ids,
            device=self._device,
        )
        self.config = config or TorchPersonalConfig()
        self.config.validate(verified.config)
        self._schema_sha256, self._state_names = _adapter_schema(
            verified.config, self.config
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: PortableBundle,
        *,
        device: str | torch.device = "auto",
        config: TorchPersonalConfig | None = None,
    ) -> "TorchPersonalTrainer":
        return cls(bundle, device=device, config=config)

    @property
    def backend_name(self) -> str:
        return TORCH_PERSONAL_BACKEND

    @property
    def adapter_schema_sha256(self) -> str:
        return self._schema_sha256

    def personal_training_target_fits(self, prompt: str, human_response: str) -> bool:
        return personal_dialogue_target_fits(
            self._tokenizer,
            prompt,
            human_response,
            context_length=min(
                self._foundation_config.context_length,
                self.config.training_context_length,
            ),
        )

    def train_adapter(
        self,
        request: PersonalAdapterTrainingRequest,
    ) -> PersonalAdapterCandidate:
        bundle = load_portable_bundle(self._bundle_root)
        _raise_if_cancelled(request)
        _validate_request(
            request,
            foundation_sha256=self._foundation_sha256,
            schema_sha256=self._schema_sha256,
        )
        if bundle.manifest["weights_sha256"] != self._foundation_sha256:
            raise PersonalTrainingError(
                "portable foundation identity changed before training"
            )

        devices = [self._device.index or 0] if self._device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed)
            if self._device.type == "cuda":
                torch.cuda.manual_seed_all(self.config.seed)
            foundation = _new_foundation(bundle, self._device)
            adapter = _new_adapter(bundle.config, self.config, device=self._device)
            if request.parent_adapter_path is not None:
                _load_adapter_state(
                    adapter, request.parent_adapter_path, self._state_names
                )
            model = TorchPersonalModel(foundation, adapter).to(self._device)
            model.assert_parameter_boundary()
            foundation_before = _tensor_identity_versions(foundation)

            context_limit = min(
                bundle.config.context_length, self.config.training_context_length
            )
            train_batches = tuple(
                _batch(
                    self._tokenizer,
                    (example,),
                    context_limit,
                    self._device,
                )
                for example in request.training_examples + request.replay_examples
            )
            holdout_batch = _batch(
                self._tokenizer,
                request.validation_examples,
                context_limit,
                self._device,
            )
            baseline = _finite_loss(model, holdout_batch, self._inactive_logit_bias)
            retention_batch = (
                _batch(
                    self._tokenizer,
                    request.retention_examples,
                    context_limit,
                    self._device,
                )
                if request.retention_examples
                else None
            )
            baseline_retention = (
                _finite_loss(model, retention_batch, self._inactive_logit_bias)
                if retention_batch is not None
                else None
            )
            optimizer = torch.optim.AdamW(
                model.adapter_parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            final_training_loss = math.inf
            for step in range(self.config.steps):
                _raise_if_cancelled(request)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = _masked_loss(
                    model,
                    train_batches[step % len(train_batches)],
                    self._inactive_logit_bias,
                )
                if not torch.isfinite(loss).item():
                    raise PersonalTrainingError(
                        "personal training loss became non-finite"
                    )
                loss.backward()
                if any(
                    parameter.grad is not None for parameter in foundation.parameters()
                ):
                    raise PersonalTrainingError("foundation received a gradient")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.adapter_parameters(), self.config.grad_clip
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                    raise PersonalTrainingError(
                        "personal adapter gradient became non-finite"
                    )
                optimizer.step()
                final_training_loss = float(loss.detach().cpu())
            _raise_if_cancelled(request)
            candidate_loss = _finite_loss(
                model, holdout_batch, self._inactive_logit_bias
            )
            candidate_retention = (
                _finite_loss(model, retention_batch, self._inactive_logit_bias)
                if retention_batch is not None
                else None
            )
            foundation_after = _tensor_identity_versions(foundation)

        if foundation_before != foundation_after:
            raise PersonalTrainingError(
                "foundation tensors changed during adapter training"
            )
        _raise_if_cancelled(request)
        artifact_path = request.work_dir.resolve() / "candidate.safetensors"
        try:
            artifact_path.relative_to(request.work_dir.resolve())
        except ValueError as error:  # Defensive even though the name is fixed.
            raise PersonalTrainingError(
                "personal artifact escaped its work directory"
            ) from error
        artifact_sha256 = _save_adapter(artifact_path, adapter)
        return PersonalAdapterCandidate(
            artifact_path=artifact_path,
            artifact_format="safetensors",
            artifact_sha256=artifact_sha256,
            backend=TORCH_PERSONAL_BACKEND,
            foundation_sha256_before=self._foundation_sha256,
            foundation_sha256_after=self._foundation_sha256,
            adapter_schema_sha256=self._schema_sha256,
            baseline_holdout_loss=baseline,
            candidate_holdout_loss=candidate_loss,
            baseline_retention_loss=baseline_retention,
            candidate_retention_loss=candidate_retention,
            metrics={
                "optimizer": "torch-adamw",
                "steps": self.config.steps,
                "new_training_examples": len(request.training_examples),
                "replay_examples": len(request.replay_examples),
                "holdout_examples": 1,
                "retention_examples": len(request.retention_examples),
                "training_context_length": context_limit,
                "assistant_only_targets": True,
                "inactive_text_tokens_masked": len(self._inactive_text_ids),
                "foundation_gradients": False,
                "foundation_unchanged": True,
                "cumulative_adapter": True,
                "holdout_improved": candidate_loss < baseline,
                "holdout_evaluations": 2,
                "holdout_used_for_selection": False,
                "final_training_loss": final_training_loss,
                "device": str(self._device),
            },
        )


class TorchPersonalRuntime:
    """Atomically swappable adapter runtime with cached token generation."""

    def __init__(
        self,
        bundle: PortableBundle,
        *,
        device: str | torch.device = "auto",
        config: TorchPersonalConfig | None = None,
        _verification_token: object | None = None,
    ) -> None:
        verified = (
            bundle
            if _verification_token is _VERIFIED_CONSTRUCTION_TOKEN
            else _verified_bundle(bundle)
        )
        self._bundle_root = verified.root
        self._foundation_sha256 = str(verified.manifest["weights_sha256"])
        self._device = _resolve_device(device)
        self.config = config or TorchPersonalConfig()
        self.config.validate(verified.config)
        self._schema_sha256, self._state_names = _adapter_schema(
            verified.config, self.config
        )
        self._tokenizer = verified.tokenizer
        self._inactive_text_ids = verified.tokenizer.inactive_text_token_ids
        self._generation_blocked_ids = verified.tokenizer.generation_blocked_token_ids
        self._generation_logit_bias = _text_logit_bias(
            verified.config.vocab_size,
            self._generation_blocked_ids,
            device=self._device,
        )
        self._first_generation_logit_bias = _text_logit_bias(
            verified.config.vocab_size,
            verified.tokenizer.first_generation_blocked_token_ids,
            device=self._device,
        )
        self._model_size = verified.model_size
        self._foundation = _new_foundation(verified, self._device)
        self._foundation_model = TorchPersonalModel(self._foundation, None)
        self._foundation_identity_versions = _tensor_identity_versions(self._foundation)
        self._active_model = self._foundation_model
        self._active_adapter_id: str | None = None
        self._lock = threading.RLock()
        self._answer_calls = 0
        self._cached_tokens = 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            schema_adapter = TorchPersonalAdapter(verified.config, self.config)
        self._adapter_parameter_count = sum(
            parameter.numel() for parameter in schema_adapter.parameters()
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: PortableBundle,
        *,
        device: str | torch.device = "auto",
        config: TorchPersonalConfig | None = None,
    ) -> "TorchPersonalRuntime":
        return cls(bundle, device=device, config=config)

    def personal_runtime_identity(self) -> PersonalRuntimeIdentity:
        with self._lock:
            return PersonalRuntimeIdentity(
                backend=TORCH_PERSONAL_BACKEND,
                foundation_sha256=self._foundation_sha256,
                adapter_schema_sha256=self._schema_sha256,
                active_adapter_id=self._active_adapter_id,
            )

    def prepare_personal_adapter(self, artifact: PersonalAdapterArtifact) -> object:
        identity = self.personal_runtime_identity()
        manifest = artifact.manifest
        if (
            manifest.backend != identity.backend
            or manifest.foundation_sha256 != identity.foundation_sha256
            or manifest.adapter_schema_sha256 != identity.adapter_schema_sha256
        ):
            raise PersonalTrainingError("personal adapter manifest is incompatible")
        if manifest.artifact_sha256 != _sha256_file(artifact.artifact_path):
            raise PersonalTrainingError("personal adapter checksum changed")
        adapter = _new_adapter(
            self._foundation.config, self.config, device=self._device
        )
        _load_adapter_state(adapter, artifact.artifact_path, self._state_names)
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
        model = TorchPersonalModel(self._foundation, adapter).to(self._device)
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise PersonalTrainingError(
                "inference model unexpectedly has trainable tensors"
            )
        return _PreparedAdapter(
            adapter_id=manifest.adapter_id,
            foundation_sha256=self._foundation_sha256,
            adapter_schema_sha256=self._schema_sha256,
            model=model,
        )

    def activate_prepared_personal_adapter(
        self,
        prepared: object,
        adapter_id: str,
    ) -> None:
        if type(prepared) is not _PreparedAdapter:
            raise PersonalTrainingError("prepared personal adapter object is invalid")
        if (
            prepared.adapter_id != adapter_id
            or prepared.foundation_sha256 != self._foundation_sha256
            or prepared.adapter_schema_sha256 != self._schema_sha256
        ):
            raise PersonalTrainingError("prepared personal adapter identity changed")
        with self._lock:
            self._assert_foundation_identity()
            self._active_model = prepared.model
            self._active_adapter_id = adapter_id

    def activate_foundation_only(self) -> None:
        with self._lock:
            self._assert_foundation_identity()
            self._active_model = self._foundation_model
            self._active_adapter_id = None

    def _assert_foundation_identity(self) -> None:
        if any(parameter.requires_grad for parameter in self._foundation.parameters()):
            raise PersonalTrainingError("runtime foundation became trainable")
        if (
            _tensor_identity_versions(self._foundation)
            != self._foundation_identity_versions
        ):
            raise PersonalTrainingError("runtime foundation tensors changed")

    def _prompt_ids(
        self, turns: Sequence[Mapping[str, Any] | Sequence[str]]
    ) -> list[int]:
        return compact_personal_dialogue_prompt(
            self._tokenizer,
            turns,
            context_length=self._foundation.config.context_length,
        )

    def personal_training_target_fits(self, prompt: str, human_response: str) -> bool:
        return personal_dialogue_target_fits(
            self._tokenizer,
            prompt,
            human_response,
            context_length=min(
                self._foundation.config.context_length,
                self.config.training_context_length,
            ),
        )

    def answer(
        self,
        turns: Sequence[Mapping[str, Any] | Sequence[str]],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
            raise TypeError("turns must be a sequence of dialogue turns")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
            raise ValueError("max_tokens must be between 1 and 1024")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0.0
        ):
            raise ValueError("temperature must be finite and non-negative")
        prompt = self._prompt_ids(turns)
        token_limit = min(max_tokens, self.config.max_tokens)
        stop = {
            self._tokenizer.pad_id,
            self._tokenizer.bos_id,
            self._tokenizer.end_turn_id,
            self._tokenizer.eos_id,
            *(self._tokenizer.role_id(role) for role in ROLE_NAMES),
        }
        generated: list[int] = []
        # The lock makes activation atomic relative to an entire answer.  The
        # KV/conv cache remains request-local and can never cross an adapter
        # boundary.
        with self._lock, torch.inference_mode():
            model = self._active_model
            tokens = torch.tensor([prompt], dtype=torch.long, device=self._device)
            logits, cache = model.forward_cached(tokens)
            cached = len(prompt)
            for index in range(token_limit):
                next_logits = _mask_inactive_text_logits(
                    logits[0, -1].float(),
                    (
                        self._first_generation_logit_bias
                        if index == 0
                        else self._generation_logit_bias
                    ),
                )
                if temperature == 0.0:
                    token = int(torch.argmax(next_logits).item())
                else:
                    probabilities = torch.softmax(next_logits / temperature, dim=-1)
                    token = int(torch.multinomial(probabilities, 1).item())
                if token in stop:
                    break
                generated.append(token)
                token_tensor = torch.tensor(
                    [[token]], dtype=torch.long, device=self._device
                )
                logits, cache = model.forward_cached(token_tensor, cache)
                cached += 1
            self._answer_calls += 1
            self._cached_tokens += cached
        return sanitize_generated_text(
            self._tokenizer.decode(generated, skip_special_tokens=True)
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            adapter_parameters = (
                0
                if self._active_model.adapter is None
                else sum(
                    parameter.numel()
                    for parameter in self._active_model.adapter.parameters()
                )
            )
            return {
                "backend": TORCH_PERSONAL_BACKEND,
                "device": str(self._device),
                "model_size": self._model_size,
                "foundation_sha256": self._foundation_sha256,
                "adapter_schema_sha256": self._schema_sha256,
                "active_adapter_id": self._active_adapter_id,
                "scratch_only": True,
                "external_weights": False,
                "inactive_text_tokens_masked": len(self._inactive_text_ids),
                "generation_blocked_text_tokens": len(self._generation_blocked_ids),
                "foundation_frozen": not any(
                    parameter.requires_grad
                    for parameter in self._foundation.parameters()
                ),
                "foundation_trainable": False,
                "foundation_parameters": sum(
                    parameter.numel() for parameter in self._foundation.parameters()
                ),
                "parameters": sum(
                    parameter.numel() for parameter in self._foundation.parameters()
                ),
                "adapter_parameters": self._adapter_parameter_count,
                "active_adapter_parameters": adapter_parameters,
                "cached_decode": True,
                "cached_inference": True,
                "cache_scope": "per-answer",
                "answer_calls": self._answer_calls,
                "cached_tokens": self._cached_tokens,
            }


class TorchPersonalFoundationTrainer:
    """Isolated full-weight conversational trainer for an active bundle."""

    backend_name = TORCH_PERSONAL_BACKEND

    def __init__(
        self,
        bundle: PortableBundle,
        *,
        device: str | torch.device = "auto",
        config: TorchFoundationConfig | None = None,
        _verification_token: object | None = None,
    ) -> None:
        verified = (
            bundle
            if _verification_token is _VERIFIED_CONSTRUCTION_TOKEN
            else _verified_bundle(bundle)
        )
        self.config = config or _default_torch_foundation_config(verified.config)
        self.config.validate()
        if self.config.context_length > verified.config.context_length:
            raise ValueError("foundation context exceeds the portable model context")
        self._device = _resolve_device(device)
        self._config_sha256 = str(verified.manifest["config_sha256"])
        self._tokenizer_sha256 = str(verified.manifest["tokenizer_sha256"])

    def train_foundation(
        self,
        request: PersonalFoundationTrainingRequest,
    ) -> PersonalFoundationCandidate:
        _raise_if_foundation_cancelled(request)
        parent = _validate_torch_foundation_request(
            request,
            expected_config_sha256=self._config_sha256,
            expected_tokenizer_sha256=self._tokenizer_sha256,
            context_length=self.config.context_length,
            tokenizer=load_portable_bundle(request.parent.bundle_path).tokenizer,
        )
        if len(request.replay_examples) > PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES:
            raise PersonalFoundationError("foundation replay exceeds its hard bound")
        _foundation_disk_preflight(request.work_dir, parent, self.config)
        candidate_weights = request.work_dir / "candidate-weights.safetensors"
        candidate_bundle_path = request.work_dir / "candidate-bundle"
        if candidate_weights.exists() or candidate_bundle_path.exists():
            raise PersonalFoundationError("foundation candidate path already exists")

        devices = [self._device.index or 0] if self._device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.config.seed + request.run_id)
                if self._device.type == "cuda":
                    torch.cuda.manual_seed_all(self.config.seed + request.run_id)
                foundation = _new_foundation(parent, self._device)
                for parameter in foundation.parameters():
                    parameter.requires_grad_(True)
                trainable = tuple(foundation.parameters())
                if not trainable or not all(
                    parameter.requires_grad for parameter in trainable
                ):
                    raise PersonalFoundationError(
                        "every foundation parameter must be optimizer-eligible"
                    )
                training_examples = request.new_examples + request.replay_examples
                training_batches = tuple(
                    _batch(
                        parent.tokenizer,
                        (example,),
                        self.config.context_length,
                        self._device,
                    )
                    for example in training_examples
                )
                monitoring_batch = _batch(
                    parent.tokenizer,
                    request.monitoring_examples,
                    self.config.context_length,
                    self._device,
                )
                inactive_ids = parent.tokenizer.inactive_text_token_ids
                inactive_bias = _text_logit_bias(
                    parent.config.vocab_size,
                    inactive_ids,
                    device=self._device,
                )
                baseline_monitoring = _finite_loss(
                    foundation,  # type: ignore[arg-type]
                    monitoring_batch,
                    inactive_bias,
                )
                optimizer = torch.optim.AdamW(
                    trainable,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    foreach=False,
                )
                losses: list[float] = []
                assistant_tokens = 0
                tokens_seen = 0
                gradient_tensors = 0
                for _epoch in range(self.config.epochs):
                    for batch in training_batches:
                        _raise_if_foundation_cancelled(request)
                        foundation.train()
                        optimizer.zero_grad(set_to_none=True)
                        loss = _masked_loss(
                            foundation,  # type: ignore[arg-type]
                            batch,
                            inactive_bias,
                        )
                        if not torch.isfinite(loss).item():
                            raise PersonalFoundationError(
                                "foundation training loss became non-finite"
                            )
                        loss.backward()
                        gradients = tuple(
                            parameter.grad
                            for parameter in trainable
                            if parameter.grad is not None
                        )
                        if not gradients or any(
                            not torch.isfinite(gradient).all().item()
                            for gradient in gradients
                        ):
                            raise PersonalFoundationError(
                                "foundation gradient became non-finite or empty"
                            )
                        gradient_tensors = max(gradient_tensors, len(gradients))
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable,
                            self.config.grad_clip,
                        )
                        if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                            raise PersonalFoundationError(
                                "foundation gradient norm became non-finite"
                            )
                        optimizer.step()
                        if any(
                            not torch.isfinite(parameter).all().item()
                            for parameter in trainable
                        ):
                            raise PersonalFoundationError(
                                "foundation optimizer produced non-finite weights"
                            )
                        losses.append(float(loss.detach().cpu()))
                        assistant_tokens += int(batch.mask.sum().item())
                        tokens_seen += int(batch.inputs.numel())
                _raise_if_foundation_cancelled(request)
                final_monitoring = _finite_loss(
                    foundation,  # type: ignore[arg-type]
                    monitoring_batch,
                    inactive_bias,
                )
                _save_torch_foundation_weights(
                    candidate_weights,
                    foundation,
                    parameter_dtype=parent.parameter_dtype,
                )

            _raise_if_foundation_cancelled(request)
            trained = publish_conversational_foundation_bundle(
                parent,
                candidate_weights,
                candidate_bundle_path,
                backend=TORCH_PERSONAL_BACKEND,
                training_config_sha256=_torch_foundation_recipe_sha256(self.config),
                source_sha256=_foundation_source_sha256(request),
                steps=len(losses),
                tokens_seen=tokens_seen,
            )
            identity = _torch_foundation_identity(
                trained,
                request.candidate_foundation_id,
            )
            metrics: dict[str, Any] = {
                "all_foundation_parameters_optimizer_eligible": True,
                "assistant_target_tokens": assistant_tokens,
                "baseline_monitoring_loss": baseline_monitoring,
                "candidate_monitoring_loss": final_monitoring,
                "epochs": self.config.epochs,
                "final_training_loss": losses[-1],
                "foundation_parameter_count": sum(
                    parameter.numel() for parameter in trainable
                ),
                "gradient_tensor_count": gradient_tensors,
                "inactive_text_tokens_masked": len(inactive_ids),
                "mean_training_loss": sum(losses) / len(losses),
                "monitoring_examples": len(request.monitoring_examples),
                "monitoring_used_for_selection": False,
                "new_examples_optimized": len(request.new_examples),
                "optimizer_state_continuity": False,
                "optimizer_state_reset": True,
                "optimizer_eligible_tensor_count": len(trainable),
                "output_bfloat16": parent.parameter_dtype == "bfloat16",
                "replay_examples": len(request.replay_examples),
                "tensor_scope_full_foundation": True,
                "tokens_seen": tokens_seen,
                "updates": len(losses),
            }
            validate_foundation_metrics(metrics)
            return PersonalFoundationCandidate(
                bundle_path=trained.root,
                identity=identity,
                parent_foundation_id=request.parent.identity.foundation_id,
                parent_bundle_sha256=request.parent.identity.bundle_sha256,
                metrics=metrics,
            )
        except RuntimeError as error:
            text = str(error).casefold()
            if "out of memory" in text or "not enough memory" in text:
                _clear_torch_device_cache(self._device)
                raise PersonalFoundationError(
                    "not enough device memory for a full foundation update; "
                    "the active parent remains unchanged"
                ) from error
            raise
        finally:
            candidate_weights.unlink(missing_ok=True)


class TorchPersonalFoundationRuntime:
    """Hot-swappable complete-foundation runtime with request-local caches."""

    def __init__(
        self,
        bundle: PortableBundle,
        *,
        device: str | torch.device = "auto",
        config: TorchFoundationConfig | None = None,
        _verification_token: object | None = None,
    ) -> None:
        verified = (
            bundle
            if _verification_token is _VERIFIED_CONSTRUCTION_TOKEN
            else _verified_bundle(bundle)
        )
        self.config = config or _default_torch_foundation_config(verified.config)
        self.config.validate()
        if self.config.context_length > verified.config.context_length:
            raise ValueError("foundation context exceeds the portable model context")
        self._device = _resolve_device(device)
        self._base_config_sha256 = str(verified.manifest["config_sha256"])
        self._base_tokenizer_sha256 = str(verified.manifest["tokenizer_sha256"])
        self._backend_contract_sha256 = personal_foundation_backend_contract_sha256(
            TORCH_PERSONAL_BACKEND
        )
        source = PersonalFoundationSource(
            bundle_path=verified.root,
            identity=_torch_foundation_identity(verified, TORCH_FOUNDATION_BASE_ID),
        )
        foundation = _new_foundation(verified, self._device)
        model = TorchPersonalModel(foundation, None).to(self._device)
        model.eval()
        _torch_foundation_forward_smoke(model, verified.tokenizer)
        self._source = source
        self._model = model
        self._tokenizer = verified.tokenizer
        self._generation_logit_bias = _text_logit_bias(
            verified.config.vocab_size,
            verified.tokenizer.generation_blocked_token_ids,
            device=self._device,
        )
        self._first_generation_logit_bias = _text_logit_bias(
            verified.config.vocab_size,
            verified.tokenizer.first_generation_blocked_token_ids,
            device=self._device,
        )
        self._lock = threading.RLock()
        self._answer_calls = 0
        self._cached_tokens = 0

    def personal_foundation_source(self) -> PersonalFoundationSource:
        with self._lock:
            return self._source

    def prepare_personal_foundation(
        self,
        artifact: PersonalFoundationArtifact,
    ) -> object:
        if type(artifact) is not PersonalFoundationArtifact:
            raise TypeError("artifact must be a PersonalFoundationArtifact")
        artifact.manifest.validate()
        verified = load_portable_bundle(artifact.bundle_path)
        identity = _torch_foundation_identity(
            verified,
            artifact.manifest.identity.foundation_id,
        )
        if identity != artifact.manifest.identity:
            raise PersonalFoundationError("foundation artifact identity changed")
        if (
            identity.backend != TORCH_PERSONAL_BACKEND
            or identity.config_sha256 != self._base_config_sha256
            or identity.tokenizer_sha256 != self._base_tokenizer_sha256
            or identity.backend_contract_sha256 != self._backend_contract_sha256
        ):
            raise PersonalFoundationError("foundation artifact lineage is incompatible")
        foundation = _new_foundation(verified, self._device)
        model = TorchPersonalModel(foundation, None).to(self._device)
        model.eval()
        _torch_foundation_forward_smoke(model, verified.tokenizer)
        return _PreparedTorchFoundation(
            source=PersonalFoundationSource(verified.root, identity),
            model=model,
        )

    def activate_prepared_personal_foundation(
        self,
        prepared: object,
        identity: PersonalFoundationIdentity,
    ) -> None:
        if type(prepared) is not _PreparedTorchFoundation:
            raise PersonalFoundationError("prepared foundation object is invalid")
        if prepared.source.identity != identity:
            raise PersonalFoundationError("prepared foundation identity changed")
        identity.validate()
        with self._lock:
            # Assign the complete, already-smoked state as one critical-section
            # update. In-flight answers hold this same lock for their full cache.
            self._model = prepared.model
            self._source = prepared.source

    def personal_training_target_fits(self, prompt: str, human_response: str) -> bool:
        return personal_dialogue_target_fits(
            self._tokenizer,
            prompt,
            human_response,
            context_length=self.config.context_length,
        )

    def _prompt_ids(
        self,
        turns: Sequence[Mapping[str, Any] | Sequence[str]],
    ) -> list[int]:
        return compact_personal_dialogue_prompt(
            self._tokenizer,
            turns,
            context_length=self._model.config.context_length,
        )

    def answer(
        self,
        turns: Sequence[Mapping[str, Any] | Sequence[str]],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        if isinstance(turns, (str, bytes)) or not isinstance(turns, Sequence):
            raise TypeError("turns must be a sequence of dialogue turns")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 1024:
            raise ValueError("max_tokens must be between 1 and 1024")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0.0
        ):
            raise ValueError("temperature must be finite and non-negative")
        prompt = self._prompt_ids(turns)
        stop = {
            self._tokenizer.pad_id,
            self._tokenizer.bos_id,
            self._tokenizer.end_turn_id,
            self._tokenizer.eos_id,
            *(self._tokenizer.role_id(role) for role in ROLE_NAMES),
        }
        generated: list[int] = []
        with self._lock, torch.inference_mode():
            model = self._model
            tokens = torch.tensor([prompt], dtype=torch.long, device=self._device)
            logits, cache = model.forward_cached(tokens)
            cached = len(prompt)
            for index in range(min(max_tokens, 32)):
                next_logits = _mask_inactive_text_logits(
                    logits[0, -1].float(),
                    self._first_generation_logit_bias
                    if index == 0
                    else self._generation_logit_bias,
                )
                if temperature == 0.0:
                    token = int(torch.argmax(next_logits).item())
                else:
                    probabilities = torch.softmax(next_logits / temperature, dim=-1)
                    token = int(torch.multinomial(probabilities, 1).item())
                if token in stop:
                    break
                generated.append(token)
                token_tensor = torch.tensor(
                    [[token]], dtype=torch.long, device=self._device
                )
                logits, cache = model.forward_cached(token_tensor, cache)
                cached += 1
            self._answer_calls += 1
            self._cached_tokens += cached
        return sanitize_generated_text(
            self._tokenizer.decode(generated, skip_special_tokens=True)
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            identity = self._source.identity
            parameters = sum(
                parameter.numel() for parameter in self._model.parameters()
            )
            return {
                "active_foundation_id": identity.foundation_id,
                "adapter_parameters": 0,
                "all_foundation_parameters_optimizer_eligible": True,
                "answer_calls": self._answer_calls,
                "automatic_learning": "full-foundation",
                "backend": TORCH_PERSONAL_BACKEND,
                "bundle_sha256": identity.bundle_sha256,
                "cached_decode": True,
                "cached_inference": True,
                "cached_tokens": self._cached_tokens,
                "device": str(self._device),
                "external_weights": False,
                "foundation_frozen": not any(
                    parameter.requires_grad for parameter in self._model.parameters()
                ),
                "foundation_parameters": parameters,
                "foundation_sha256": identity.weights_sha256,
                "foundation_trainable": False,
                "optimizer_state_continuity": False,
                "parameters": parameters,
                "scratch_only": True,
            }


def create_torch_personal_trainer(
    bundle: PortableBundle,
    *,
    device: str | torch.device = "auto",
    config: TorchPersonalConfig | None = None,
) -> TorchPersonalTrainer:
    """Create an isolated trainer that reloads its own foundation tensors."""

    return TorchPersonalTrainer.from_bundle(bundle, device=device, config=config)


def create_torch_personal_runtime(
    bundle: PortableBundle,
    *,
    device: str | torch.device = "auto",
    config: TorchPersonalConfig | None = None,
) -> TorchPersonalRuntime:
    return TorchPersonalRuntime.from_bundle(bundle, device=device, config=config)


def create_torch_personal_backend(
    bundle: PortableBundle,
    selection: BackendSelection | None = None,
    *,
    rank: int = 8,
    device: str | torch.device | None = None,
    trainer_device: str | torch.device | None = None,
    config: TorchPersonalConfig | None = None,
) -> tuple[TorchPersonalTrainer, TorchPersonalRuntime]:
    """Create matching trainer/runtime instances with no mutable tensor sharing."""

    if config is not None and rank != 8 and rank != config.rank:
        raise ValueError("rank and explicit TorchPersonalConfig disagree")
    selected_config = config or TorchPersonalConfig(rank=rank)
    selected_device: str | torch.device = device or (
        _selection_device(selection) if selection is not None else "auto"
    )
    verified = _verified_bundle(bundle)
    runtime = TorchPersonalRuntime(
        verified,
        device=selected_device,
        config=selected_config,
        _verification_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )
    trainer = TorchPersonalTrainer(
        verified,
        device=selected_device if trainer_device is None else trainer_device,
        config=selected_config,
        _verification_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )
    if (
        trainer.adapter_schema_sha256
        != runtime.personal_runtime_identity().adapter_schema_sha256
    ):
        raise PersonalTrainingError("PyTorch personal backend schema mismatch")
    return trainer, runtime


def create_torch_personal_foundation_trainer(
    bundle: PortableBundle,
    *,
    device: str | torch.device = "auto",
    config: TorchFoundationConfig | None = None,
) -> TorchPersonalFoundationTrainer:
    return TorchPersonalFoundationTrainer(bundle, device=device, config=config)


def create_torch_personal_foundation_runtime(
    bundle: PortableBundle,
    *,
    device: str | torch.device = "auto",
    config: TorchFoundationConfig | None = None,
) -> TorchPersonalFoundationRuntime:
    return TorchPersonalFoundationRuntime(bundle, device=device, config=config)


def create_torch_personal_foundation_backend(
    bundle: PortableBundle,
    selection: BackendSelection | None = None,
    *,
    device: str | torch.device | None = None,
    trainer_device: str | torch.device | None = None,
    config: TorchFoundationConfig | None = None,
) -> tuple[TorchPersonalFoundationTrainer, TorchPersonalFoundationRuntime]:
    """Create isolated full-weight trainer and atomically swappable runtime."""

    selected_device: str | torch.device = device or (
        _selection_device(selection) if selection is not None else "auto"
    )
    verified = _verified_bundle(bundle)
    selected_config = config or _default_torch_foundation_config(verified.config)
    selected_config.validate()
    runtime = TorchPersonalFoundationRuntime(
        verified,
        device=selected_device,
        config=selected_config,
        _verification_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )
    trainer = TorchPersonalFoundationTrainer(
        verified,
        device=selected_device if trainer_device is None else trainer_device,
        config=selected_config,
        _verification_token=_VERIFIED_CONSTRUCTION_TOKEN,
    )
    source = runtime.personal_foundation_source()
    if (
        source.identity.config_sha256 != verified.manifest["config_sha256"]
        or source.identity.tokenizer_sha256 != verified.manifest["tokenizer_sha256"]
    ):
        raise PersonalFoundationError("Torch foundation backend lineage mismatch")
    return trainer, runtime


__all__ = [
    "TORCH_PERSONAL_BACKEND",
    "TORCH_PERSONAL_STATE_FORMAT",
    "TorchPersonalAdapter",
    "TorchPersonalConfig",
    "TorchFoundationConfig",
    "TorchPersonalModel",
    "TorchPersonalFoundationRuntime",
    "TorchPersonalFoundationTrainer",
    "TorchPersonalRuntime",
    "TorchPersonalTrainer",
    "create_torch_personal_backend",
    "create_torch_personal_foundation_backend",
    "create_torch_personal_foundation_runtime",
    "create_torch_personal_foundation_trainer",
    "create_torch_personal_runtime",
    "create_torch_personal_trainer",
]
