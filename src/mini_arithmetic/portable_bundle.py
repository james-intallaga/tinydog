"""Portable scratch-model bundles shared by MLX and PyTorch.

The bundle is deliberately boring: one exact HybridConfig, one frozen
ScratchTokenizer, one canonical safetensors state, and a checksum manifest.
It contains no executable objects and never downloads model weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Literal

import numpy as np

from .community_updates import publish_directory_no_replace
from .hybrid_config import CONV, HybridConfig
from .scratch_tokenizer import (
    METADATA_FILENAME,
    TOKENIZER_FILENAME,
    ScratchTokenizer,
    build_scratch_tokenizer,
)


PORTABLE_BUNDLE_FORMAT = "tinydog-portable-hybrid-v1"
PORTABLE_STATE_FORMAT = "tinydog-hybrid-mlx-layout-v1"
PORTABLE_INITIALIZATION_FORMAT = "tinydog-hybrid-init-v1"
PORTABLE_TRAINED_INITIALIZATION_FORMAT = "tinydog-hybrid-foundation-trained-v1"
PORTABLE_TRAINING_PROVENANCE_FORMAT = "tinydog-foundation-provenance-v1"
MANIFEST_FILENAME = "portable-model.json"
CONFIG_FILENAME = "model-config.json"
WEIGHTS_FILENAME = "model.safetensors"
TOKENIZER_DIRECTORY = "tokenizer"
_MAX_JSON_BYTES = 256 * 1024
_MAX_WEIGHTS_OVERHEAD_BYTES = 16 * 1024 * 1024
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FIELDS = {
    "schema_version",
    "format",
    "state_format",
    "initialization",
    "model_size",
    "parameters",
    "parameter_dtype",
    "scratch_only",
    "external_weights",
    "seed",
    "created_by_backend",
    "config_file",
    "config_sha256",
    "weights_file",
    "weights_sha256",
    "tokenizer_file",
    "tokenizer_sha256",
    "tokenizer_metadata_file",
    "tokenizer_metadata_sha256",
}
_TRAINED_MANIFEST_FIELDS = _MANIFEST_FIELDS | {"training"}
_SEED_TOKENIZER_TEXT = (
    "Hello. I am TinyDog. We learn through a private local conversation.",
    "User and assistant turns may contain any UTF-8 language.",
    "0123456789 + - * / % = ( ) [ ] { } . , : ;",
    "The assistant listens, responds, receives a human example, and practises.",
)


class PortableBundleError(ValueError):
    """A local portable model bundle is malformed or incompatible."""


@dataclass(frozen=True)
class PortableBundle:
    root: Path
    manifest: dict[str, Any]
    config: HybridConfig
    tokenizer: ScratchTokenizer
    weights_path: Path

    @property
    def model_size(self) -> str:
        return str(self.manifest["model_size"])

    @property
    def parameter_dtype(self) -> str:
        return str(self.manifest["parameter_dtype"])


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _fsync_directory_portable(path: Path) -> None:
    """Persist a directory entry where the platform exposes POSIX fsync."""

    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # File bytes were flushed first. Some filesystems do not expose
            # directory fsync, so the durability hint remains best-effort.
            pass
    finally:
        os.close(descriptor)


def _fsync_bundle_tree(root: Path) -> None:
    """Flush every bundle file before its directory is atomically published."""

    directories = [root]
    for path in root.rglob("*"):
        details = path.lstat()
        if stat.S_ISLNK(details.st_mode):
            raise PortableBundleError("portable model staging contains a symlink")
        if stat.S_ISDIR(details.st_mode):
            directories.append(path)
            continue
        if not stat.S_ISREG(details.st_mode):
            raise PortableBundleError("portable model staging contains a special file")
        # Windows' os.fsync delegates to the CRT commit call, which rejects a
        # descriptor opened read-only. These are private staging files that we
        # created, so opening them read/write is safe and preserves the same
        # pre-publication durability boundary on every supported platform.
        flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(
        directories,
        key=lambda item: len(item.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory_portable(directory)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    data = path.read_bytes()
    if not data or len(data) > _MAX_JSON_BYTES:
        raise PortableBundleError(f"{label} has an invalid size")
    try:

        def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise PortableBundleError(f"{label} contains duplicate fields")
                result[key] = item
            return result

        value = json.loads(
            data,
            object_pairs_hook=exact_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                PortableBundleError(f"{label} contains non-finite numbers: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortableBundleError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PortableBundleError(f"{label} must contain one JSON object")
    return value


def load_hybrid_config(path: str | Path) -> HybridConfig:
    """Load an exact backend-neutral HybridConfig JSON file."""

    value = _read_json(Path(path), "HybridConfig")
    allowed = {item.name for item in fields(HybridConfig)}
    if set(value) != allowed:
        missing = sorted(allowed - set(value))
        extra = sorted(set(value) - allowed)
        raise PortableBundleError(
            f"HybridConfig fields differ (missing={missing}, extra={extra})"
        )
    schedule = value.get("layer_schedule")
    if not isinstance(schedule, list) or not all(
        isinstance(item, str) for item in schedule
    ):
        raise PortableBundleError("HybridConfig layer_schedule must be a string list")
    value["layer_schedule"] = tuple(schedule)
    try:
        return HybridConfig(**value)
    except (TypeError, ValueError) as error:
        raise PortableBundleError("HybridConfig values are invalid") from error


def canonical_tensor_schema(config: HybridConfig) -> dict[str, tuple[int, ...]]:
    """Return the exact MLX-layout tensor contract without importing an engine."""

    dimensions = config.model_dim
    query_width = config.num_attention_heads * config.head_dim
    key_value_width = config.num_key_value_heads * config.head_dim
    schema: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (config.vocab_size, dimensions),
    }
    for index, layer_type in enumerate(config.layer_schedule):
        prefix = f"blocks.{index}"
        schema[f"{prefix}.operator_norm.weight"] = (dimensions,)
        if layer_type == CONV:
            schema[f"{prefix}.operator.in_proj.weight"] = (3 * dimensions, dimensions)
            schema[f"{prefix}.operator.conv.weight"] = (
                dimensions,
                config.conv_kernel_size,
                1,
            )
            schema[f"{prefix}.operator.out_proj.weight"] = (dimensions, dimensions)
        else:
            schema[f"{prefix}.operator.q_proj.weight"] = (query_width, dimensions)
            schema[f"{prefix}.operator.k_proj.weight"] = (key_value_width, dimensions)
            schema[f"{prefix}.operator.v_proj.weight"] = (key_value_width, dimensions)
            schema[f"{prefix}.operator.out_proj.weight"] = (dimensions, query_width)
            schema[f"{prefix}.operator.q_norm.weight"] = (config.head_dim,)
            schema[f"{prefix}.operator.k_norm.weight"] = (config.head_dim,)
        schema[f"{prefix}.ffn_norm.weight"] = (dimensions,)
        schema[f"{prefix}.feed_forward.w1.weight"] = (config.ffn_dim, dimensions)
        schema[f"{prefix}.feed_forward.w2.weight"] = (dimensions, config.ffn_dim)
        schema[f"{prefix}.feed_forward.w3.weight"] = (config.ffn_dim, dimensions)
    schema["final_norm.weight"] = (dimensions,)
    if (
        sum(math.prod(shape) for shape in schema.values())
        != config.expected_parameter_count()
    ):
        raise RuntimeError("canonical tensor schema and parameter count diverged")
    return schema


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    file_size = path.stat().st_size
    if file_size < 10:
        raise PortableBundleError("portable model safetensors file is too small")
    with path.open("rb") as handle:
        length_bytes = handle.read(8)
        header_length = int.from_bytes(length_bytes, "little", signed=False)
        if (
            header_length <= 0
            or header_length > _MAX_SAFETENSORS_HEADER_BYTES
            or header_length > file_size - 8
        ):
            raise PortableBundleError(
                "portable model safetensors header size is invalid"
            )
        header_bytes = handle.read(header_length)
    try:

        def exact_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise PortableBundleError(
                        "portable model safetensors header contains duplicate fields"
                    )
                result[key] = item
            return result

        header = json.loads(header_bytes, object_pairs_hook=exact_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PortableBundleError(
            "portable model safetensors header is invalid"
        ) from error
    if not isinstance(header, dict):
        raise PortableBundleError("portable model safetensors header must be an object")
    return header, 8 + header_length, file_size


def _validate_finite_payload(
    path: Path,
    *,
    data_offset: int,
    dtype_name: Literal["F32", "BF16"],
) -> None:
    if dtype_name == "F32":
        word_dtype = np.dtype("<u4")
        exponent_mask = np.uint32(0x7F800000)
    else:
        word_dtype = np.dtype("<u2")
        exponent_mask = np.uint16(0x7F80)
    chunk_bytes = 4 * 1024 * 1024
    chunk_bytes -= chunk_bytes % word_dtype.itemsize
    with path.open("rb") as handle:
        handle.seek(data_offset)
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            if len(chunk) % word_dtype.itemsize:
                raise PortableBundleError("portable model tensor bytes are misaligned")
            words = np.frombuffer(chunk, dtype=word_dtype)
            if bool(np.any(np.bitwise_and(words, exponent_mask) == exponent_mask)):
                raise PortableBundleError("portable model contains non-finite weights")


def _validate_portable_weights(
    path: Path,
    config: HybridConfig,
    parameter_dtype: str,
) -> None:
    """Validate the complete canonical tensor contract without a compute engine."""

    from safetensors import SafetensorError, safe_open

    schema = canonical_tensor_schema(config)
    expected_dtype: Literal["F32", "BF16"] = (
        "BF16" if parameter_dtype == "bfloat16" else "F32"
    )
    try:
        with safe_open(str(path), framework="numpy") as handle:
            if handle.metadata() != {"format": PORTABLE_STATE_FORMAT}:
                raise PortableBundleError(
                    "portable model safetensors metadata is incompatible"
                )
            names = tuple(handle.keys())
            if set(names) != set(schema):
                missing = sorted(set(schema) - set(names))
                extra = sorted(set(names) - set(schema))
                raise PortableBundleError(
                    "portable model tensor names differ "
                    f"(missing={missing[:4]}, extra={extra[:4]})"
                )
            for name, shape in schema.items():
                tensor = handle.get_slice(name)
                if tuple(tensor.get_shape()) != shape:
                    raise PortableBundleError(
                        f"portable model tensor {name!r} shape differs"
                    )
                if tensor.get_dtype() != expected_dtype:
                    raise PortableBundleError(
                        f"portable model tensor {name!r} dtype differs from the manifest"
                    )
    except PortableBundleError:
        raise
    except (OSError, RuntimeError, SafetensorError, TypeError, ValueError) as error:
        raise PortableBundleError("portable model safetensors are invalid") from error

    header, data_offset, file_size = _read_safetensors_header(path)
    if header.get("__metadata__") != {"format": PORTABLE_STATE_FORMAT}:
        raise PortableBundleError("portable model safetensors metadata is incompatible")
    tensor_header = {
        name: value for name, value in header.items() if name != "__metadata__"
    }
    if set(tensor_header) != set(schema):
        raise PortableBundleError("portable model safetensors header names differ")
    item_size = 2 if expected_dtype == "BF16" else 4
    ranges: list[tuple[int, int]] = []
    for name, shape in schema.items():
        entry = tensor_header[name]
        if not isinstance(entry, dict) or set(entry) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            raise PortableBundleError(f"portable model tensor {name!r} header differs")
        offsets = entry.get("data_offsets")
        declared_shape = entry.get("shape")
        if (
            entry.get("dtype") != expected_dtype
            or not isinstance(declared_shape, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in declared_shape
            )
            or declared_shape != list(shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in offsets
            )
        ):
            raise PortableBundleError(
                f"portable model tensor {name!r} header is invalid"
            )
        start, end = offsets
        if start < 0 or end < start or end - start != math.prod(shape) * item_size:
            raise PortableBundleError(
                f"portable model tensor {name!r} byte range differs"
            )
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise PortableBundleError(
                "portable model tensor byte ranges are not contiguous"
            )
        cursor = end
    if data_offset + cursor != file_size:
        raise PortableBundleError("portable model safetensors file length differs")
    _validate_finite_payload(path, data_offset=data_offset, dtype_name=expected_dtype)


def _portable_tensor_payloads_differ(parent: Path, candidate: Path) -> bool:
    """Return whether at least one canonical tensor byte genuinely changed.

    A safetensors file hash also covers JSON header ordering and metadata.  A
    conversational foundation candidate must change model values, not merely
    rewrite an equivalent container, so compare each named tensor's raw range.
    Both files have already passed the same canonical schema/dtype validator.
    """

    parent_header, parent_data, _ = _read_safetensors_header(parent)
    candidate_header, candidate_data, _ = _read_safetensors_header(candidate)
    names = sorted(set(parent_header) - {"__metadata__"})
    if names != sorted(set(candidate_header) - {"__metadata__"}):
        raise PortableBundleError("portable candidate tensor names differ")
    chunk_size = 1024 * 1024
    with parent.open("rb") as parent_handle, candidate.open("rb") as candidate_handle:
        for name in names:
            parent_offsets = parent_header[name]["data_offsets"]
            candidate_offsets = candidate_header[name]["data_offsets"]
            parent_length = parent_offsets[1] - parent_offsets[0]
            candidate_length = candidate_offsets[1] - candidate_offsets[0]
            if parent_length != candidate_length:
                raise PortableBundleError(
                    "portable candidate tensor byte length differs"
                )
            parent_handle.seek(parent_data + parent_offsets[0])
            candidate_handle.seek(candidate_data + candidate_offsets[0])
            remaining = parent_length
            while remaining:
                width = min(chunk_size, remaining)
                if parent_handle.read(width) != candidate_handle.read(width):
                    return True
                remaining -= width
    return False


def _write_initial_weights(
    path: Path,
    config: HybridConfig,
    *,
    backend: Literal["mlx", "torch"],
    seed: int,
    parameter_dtype: Literal["bfloat16", "float32"],
) -> None:
    if backend == "torch":
        try:
            import torch
            from safetensors.torch import save_file
        except (ImportError, OSError) as error:
            raise PortableBundleError("PyTorch and safetensors are required") from error
        from .torch_hybrid import HybridLM

        torch.manual_seed(seed)
        model = HybridLM(config)
        dtype = torch.bfloat16 if parameter_dtype == "bfloat16" else torch.float32
        model.to(device="cpu", dtype=dtype)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.canonical_state_dict().items()
        }
        save_file(
            state,
            str(path),
            metadata={"format": PORTABLE_STATE_FORMAT},
        )
        return

    if backend != "mlx":
        raise PortableBundleError("backend must be mlx or torch")
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
    except (ImportError, OSError) as error:
        raise PortableBundleError("MLX is required for this backend") from error
    from .hybrid import HybridLM

    mx.random.seed(seed)
    model = HybridLM(config)
    model.set_dtype(mx.bfloat16 if parameter_dtype == "bfloat16" else mx.float32)
    tensors = dict(tree_flatten(model.parameters()))
    mx.eval(tensors)
    mx.save_safetensors(
        str(path),
        tensors,
        metadata={"format": PORTABLE_STATE_FORMAT},
    )


def _install_frozen_tokenizer(
    source: str | Path, destination: Path
) -> ScratchTokenizer:
    """Copy one checksum-verified tokenizer without following artifact links."""

    unresolved = Path(source).expanduser()
    if unresolved.is_symlink():
        raise PortableBundleError("frozen tokenizer source must not be a symlink")
    tokenizer = ScratchTokenizer.load(unresolved)
    source_directory = tokenizer.artifact_dir.resolve()
    tokenizer_source = source_directory / TOKENIZER_FILENAME
    metadata_source = source_directory / METADATA_FILENAME
    if any(
        path.is_symlink() or not path.is_file()
        for path in (tokenizer_source, metadata_source)
    ):
        raise PortableBundleError("frozen tokenizer artifacts must be regular files")
    destination.mkdir(parents=False, exist_ok=False)
    (destination / TOKENIZER_FILENAME).write_bytes(tokenizer_source.read_bytes())
    (destination / METADATA_FILENAME).write_bytes(metadata_source.read_bytes())
    installed = ScratchTokenizer.load(destination)
    if installed.metadata["tokenizer_sha256"] != tokenizer.metadata["tokenizer_sha256"]:
        raise PortableBundleError("copied tokenizer identity changed")
    return installed


def create_portable_bundle(
    destination: str | Path,
    *,
    config_path: str | Path,
    backend: Literal["mlx", "torch"],
    model_size: Literal["tiny", "standard"],
    seed: int = 0,
    parameter_dtype: Literal["bfloat16", "float32"] | None = None,
    tokenizer_path: str | Path | None = None,
) -> PortableBundle:
    """Create one new random-init model bundle without external weights."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if backend not in {"mlx", "torch"}:
        raise ValueError("backend must be mlx or torch")
    if model_size not in {"tiny", "standard"}:
        raise ValueError("model_size must be tiny or standard")
    if parameter_dtype not in {None, "bfloat16", "float32"}:
        raise ValueError("parameter_dtype must be bfloat16 or float32")
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Portable model already exists at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    config = load_hybrid_config(config_path)
    if config.vocab_size != 32_768:
        raise PortableBundleError(
            "portable HybridConfig must use the frozen 32K vocabulary"
        )
    dtype = parameter_dtype or ("float32" if model_size == "tiny" else "bfloat16")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=".tinydog-portable-", dir=target.parent)
    )
    try:
        tokenizer_dir = temporary / TOKENIZER_DIRECTORY
        if tokenizer_path is None:
            build_scratch_tokenizer(
                _SEED_TOKENIZER_TEXT,
                tokenizer_dir,
                overwrite=False,
            )
        else:
            _install_frozen_tokenizer(tokenizer_path, tokenizer_dir)
        config_payload = asdict(config)
        config_payload["layer_schedule"] = list(config.layer_schedule)
        config_bytes = _canonical_json(config_payload)
        (temporary / CONFIG_FILENAME).write_bytes(config_bytes)
        weights_path = temporary / WEIGHTS_FILENAME
        _write_initial_weights(
            weights_path,
            config,
            backend=backend,
            seed=seed,
            parameter_dtype=dtype,
        )
        tokenizer_file_path = tokenizer_dir / TOKENIZER_FILENAME
        tokenizer_metadata_path = tokenizer_dir / METADATA_FILENAME
        manifest = {
            "schema_version": 1,
            "format": PORTABLE_BUNDLE_FORMAT,
            "state_format": PORTABLE_STATE_FORMAT,
            "initialization": PORTABLE_INITIALIZATION_FORMAT,
            "model_size": model_size,
            "parameters": config.expected_parameter_count(),
            "parameter_dtype": dtype,
            "scratch_only": True,
            "external_weights": False,
            "seed": seed,
            "created_by_backend": backend,
            "config_file": CONFIG_FILENAME,
            "config_sha256": _sha256_bytes(config_bytes),
            "weights_file": WEIGHTS_FILENAME,
            "weights_sha256": _sha256_file(weights_path),
            "tokenizer_file": f"{TOKENIZER_DIRECTORY}/{TOKENIZER_FILENAME}",
            "tokenizer_sha256": _sha256_file(tokenizer_file_path),
            "tokenizer_metadata_file": f"{TOKENIZER_DIRECTORY}/{METADATA_FILENAME}",
            "tokenizer_metadata_sha256": _sha256_file(tokenizer_metadata_path),
        }
        (temporary / MANIFEST_FILENAME).write_bytes(_canonical_json(manifest))
        _fsync_bundle_tree(temporary)
        publish_directory_no_replace(temporary, target)
        _fsync_directory_portable(target.parent)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return load_portable_bundle(target)


def publish_conversational_foundation_bundle(
    parent_bundle: PortableBundle | str | Path,
    candidate_weights: str | Path,
    destination: str | Path,
    *,
    backend: Literal["mlx", "torch"],
    training_config_sha256: str,
    source_sha256: dict[str, str],
    steps: int,
    tokens_seen: int,
) -> PortableBundle:
    """Atomically publish a complete full-weight conversational candidate.

    The active parent remains immutable.  ``candidate_weights`` must already
    be a complete canonical state with the parent's exact dtype; this function
    verifies the schema, finite payload, tokenizer/config lineage, genuine
    tensor-byte mutation, checksums, and provenance before publishing a new
    portable bundle directory.
    """

    if backend not in {"mlx", "torch"}:
        raise ValueError("backend must be mlx or torch")
    if (
        not isinstance(training_config_sha256, str)
        or _SHA256_RE.fullmatch(training_config_sha256) is None
    ):
        raise ValueError("training_config_sha256 must be a SHA-256 digest")
    if (
        not isinstance(source_sha256, dict)
        or not source_sha256
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            for name, digest in source_sha256.items()
        )
    ):
        raise ValueError("source_sha256 must contain named SHA-256 digests")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(tokens_seen, bool)
        or not isinstance(tokens_seen, int)
        or tokens_seen <= 0
    ):
        raise ValueError("tokens_seen must be a positive integer")

    parent = load_portable_bundle(
        parent_bundle.root
        if isinstance(parent_bundle, PortableBundle)
        else parent_bundle
    )
    unresolved_weights = Path(candidate_weights).expanduser()
    if unresolved_weights.is_symlink():
        raise PortableBundleError("candidate foundation weights must not be a symlink")
    weights_source = unresolved_weights.resolve()
    if not weights_source.is_file():
        raise PortableBundleError("candidate foundation weights are missing")
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Portable model already exists at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=".tinydog-conversation-", dir=target.parent)
    )
    try:
        tokenizer_dir = temporary / TOKENIZER_DIRECTORY
        tokenizer_dir.mkdir()
        shutil.copyfile(parent.root / CONFIG_FILENAME, temporary / CONFIG_FILENAME)
        shutil.copyfile(weights_source, temporary / WEIGHTS_FILENAME)
        shutil.copyfile(
            parent.root / TOKENIZER_DIRECTORY / TOKENIZER_FILENAME,
            tokenizer_dir / TOKENIZER_FILENAME,
        )
        shutil.copyfile(
            parent.root / TOKENIZER_DIRECTORY / METADATA_FILENAME,
            tokenizer_dir / METADATA_FILENAME,
        )
        published_weights = temporary / WEIGHTS_FILENAME
        _validate_portable_weights(
            published_weights,
            parent.config,
            parent.parameter_dtype,
        )
        candidate_sha256 = _sha256_file(published_weights)
        if candidate_sha256 == parent.manifest["weights_sha256"]:
            raise PortableBundleError("candidate foundation weights did not change")
        if not _portable_tensor_payloads_differ(parent.weights_path, published_weights):
            raise PortableBundleError(
                "candidate foundation tensor bytes did not change"
            )
        manifest = {
            "schema_version": 1,
            "format": PORTABLE_BUNDLE_FORMAT,
            "state_format": PORTABLE_STATE_FORMAT,
            "initialization": PORTABLE_TRAINED_INITIALIZATION_FORMAT,
            "model_size": parent.model_size,
            "parameters": parent.config.expected_parameter_count(),
            "parameter_dtype": parent.parameter_dtype,
            "scratch_only": True,
            "external_weights": False,
            "seed": parent.manifest["seed"],
            "created_by_backend": backend,
            "config_file": CONFIG_FILENAME,
            "config_sha256": _sha256_file(temporary / CONFIG_FILENAME),
            "weights_file": WEIGHTS_FILENAME,
            "weights_sha256": candidate_sha256,
            "tokenizer_file": f"{TOKENIZER_DIRECTORY}/{TOKENIZER_FILENAME}",
            "tokenizer_sha256": _sha256_file(tokenizer_dir / TOKENIZER_FILENAME),
            "tokenizer_metadata_file": f"{TOKENIZER_DIRECTORY}/{METADATA_FILENAME}",
            "tokenizer_metadata_sha256": _sha256_file(
                tokenizer_dir / METADATA_FILENAME
            ),
            "training": {
                "format": PORTABLE_TRAINING_PROVENANCE_FORMAT,
                "parent_weights_sha256": parent.manifest["weights_sha256"],
                "training_config_sha256": training_config_sha256,
                "source_sha256": dict(sorted(source_sha256.items())),
                "global_step": steps,
                "tokens_seen": tokens_seen,
                "training_complete": True,
                "stage_count": 1,
                "total_steps": steps,
            },
        }
        (temporary / MANIFEST_FILENAME).write_bytes(_canonical_json(manifest))
        _fsync_bundle_tree(temporary)
        # Full load validates the bundle as a consumer will see it before any
        # directory becomes visible at the final path.
        load_portable_bundle(temporary)
        reverified_parent = load_portable_bundle(parent.root)
        if (
            reverified_parent.manifest["weights_sha256"]
            != parent.manifest["weights_sha256"]
            or reverified_parent.manifest["config_sha256"]
            != parent.manifest["config_sha256"]
            or reverified_parent.manifest["tokenizer_sha256"]
            != parent.manifest["tokenizer_sha256"]
        ):
            raise PortableBundleError(
                "parent bundle changed during candidate publication"
            )
        publish_directory_no_replace(temporary, target)
        _fsync_directory_portable(target.parent)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return load_portable_bundle(target)


def _validate_training_provenance(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "format",
        "parent_weights_sha256",
        "training_config_sha256",
        "source_sha256",
        "global_step",
        "tokens_seen",
        "training_complete",
        "stage_count",
        "total_steps",
    }:
        raise PortableBundleError("trained bundle provenance fields differ")
    if value.get("format") != PORTABLE_TRAINING_PROVENANCE_FORMAT:
        raise PortableBundleError("trained bundle provenance format is invalid")
    for field_name in ("parent_weights_sha256", "training_config_sha256"):
        digest = value.get(field_name)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise PortableBundleError(f"trained bundle {field_name} is invalid")
    source_hashes = value.get("source_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            for name, digest in source_hashes.items()
        )
    ):
        raise PortableBundleError("trained bundle source provenance is invalid")
    if value.get("training_complete") is not True:
        raise PortableBundleError("trained bundle is not a completed foundation run")
    for field_name in ("global_step", "tokens_seen", "stage_count", "total_steps"):
        count = value.get(field_name)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PortableBundleError(f"trained bundle {field_name} is invalid")
    if value["global_step"] != value["total_steps"]:
        raise PortableBundleError("trained bundle step count is incomplete")


def load_portable_bundle(path: str | Path) -> PortableBundle:
    """Verify and open one local bundle without materializing model tensors."""

    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise PortableBundleError("portable model root must not be a symlink")
    root = unresolved.resolve()
    try:
        root_details = root.lstat()
    except FileNotFoundError as error:
        raise PortableBundleError("portable model directory is missing") from error
    if not stat.S_ISDIR(root_details.st_mode):
        raise PortableBundleError("portable model root must be a directory")
    manifest = _read_json(root / MANIFEST_FILENAME, "portable model manifest")
    manifest_fields = set(manifest)
    if (
        manifest_fields != _MANIFEST_FIELDS
        and manifest_fields != _TRAINED_MANIFEST_FIELDS
    ):
        raise PortableBundleError("portable model manifest fields differ")
    initialization = manifest.get("initialization")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != PORTABLE_BUNDLE_FORMAT
        or manifest.get("state_format") != PORTABLE_STATE_FORMAT
        or initialization
        not in {PORTABLE_INITIALIZATION_FORMAT, PORTABLE_TRAINED_INITIALIZATION_FORMAT}
        or manifest.get("scratch_only") is not True
        or manifest.get("external_weights") is not False
    ):
        raise PortableBundleError("portable model manifest identity is invalid")
    if initialization == PORTABLE_INITIALIZATION_FORMAT:
        if manifest_fields != _MANIFEST_FIELDS:
            raise PortableBundleError(
                "random bundle must not claim training provenance"
            )
    else:
        if manifest_fields != _TRAINED_MANIFEST_FIELDS:
            raise PortableBundleError("trained bundle is missing training provenance")
        _validate_training_provenance(manifest.get("training"))
    expected_files = {
        "config_file": CONFIG_FILENAME,
        "weights_file": WEIGHTS_FILENAME,
        "tokenizer_file": f"{TOKENIZER_DIRECTORY}/{TOKENIZER_FILENAME}",
        "tokenizer_metadata_file": f"{TOKENIZER_DIRECTORY}/{METADATA_FILENAME}",
    }
    for field_name, expected in expected_files.items():
        if manifest.get(field_name) != expected:
            raise PortableBundleError(f"portable model {field_name} is invalid")
    for digest_field in (
        "config_sha256",
        "weights_sha256",
        "tokenizer_sha256",
        "tokenizer_metadata_sha256",
    ):
        digest = manifest.get(digest_field)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise PortableBundleError(f"portable model {digest_field} is invalid")
    checks = (
        (CONFIG_FILENAME, "config_sha256"),
        (WEIGHTS_FILENAME, "weights_sha256"),
        (f"{TOKENIZER_DIRECTORY}/{TOKENIZER_FILENAME}", "tokenizer_sha256"),
        (
            f"{TOKENIZER_DIRECTORY}/{METADATA_FILENAME}",
            "tokenizer_metadata_sha256",
        ),
    )
    for relative, digest_field in checks:
        candidate = root / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise PortableBundleError(f"portable model artifact {relative} is missing")
        if _sha256_file(candidate) != manifest.get(digest_field):
            raise PortableBundleError(f"portable model artifact {relative} changed")
    config = load_hybrid_config(root / CONFIG_FILENAME)
    if manifest.get("parameters") != config.expected_parameter_count():
        raise PortableBundleError("portable model parameter count is invalid")
    if manifest.get("model_size") not in {"tiny", "standard"}:
        raise PortableBundleError("portable model size is invalid")
    if manifest.get("parameter_dtype") not in {"bfloat16", "float32"}:
        raise PortableBundleError("portable model dtype is invalid")
    if (
        isinstance(manifest.get("seed"), bool)
        or not isinstance(manifest.get("seed"), int)
        or manifest["seed"] < 0
    ):
        raise PortableBundleError("portable model seed is invalid")
    if manifest.get("created_by_backend") not in {"mlx", "torch"}:
        raise PortableBundleError("portable model creator backend is invalid")
    maximum_weights = (
        config.expected_parameter_count() * 4 + _MAX_WEIGHTS_OVERHEAD_BYTES
    )
    if (root / WEIGHTS_FILENAME).stat().st_size > maximum_weights:
        raise PortableBundleError(
            "portable model weights exceed their structural bound"
        )
    _validate_portable_weights(
        root / WEIGHTS_FILENAME,
        config,
        str(manifest["parameter_dtype"]),
    )
    if {item.name for item in root.iterdir()} != {
        MANIFEST_FILENAME,
        CONFIG_FILENAME,
        WEIGHTS_FILENAME,
        TOKENIZER_DIRECTORY,
    }:
        raise PortableBundleError("portable model directory contains unexpected files")
    tokenizer_directory = root / TOKENIZER_DIRECTORY
    if tokenizer_directory.is_symlink() or {
        item.name for item in tokenizer_directory.iterdir()
    } != {TOKENIZER_FILENAME, METADATA_FILENAME}:
        raise PortableBundleError("portable tokenizer directory is invalid")
    tokenizer = ScratchTokenizer.load(root / TOKENIZER_DIRECTORY)
    if tokenizer.vocab_size != config.vocab_size:
        raise PortableBundleError("portable tokenizer and model vocabulary differ")
    return PortableBundle(
        root=root,
        manifest=manifest,
        config=config,
        tokenizer=tokenizer,
        weights_path=root / WEIGHTS_FILENAME,
    )


def ensure_portable_bundle(
    destination: str | Path,
    **create_options: Any,
) -> tuple[PortableBundle, bool]:
    """Load an existing bundle or create it once; return ``(bundle, created)``."""

    target = Path(destination).expanduser().resolve()
    if target.exists():
        return load_portable_bundle(target), False
    return create_portable_bundle(target, **create_options), True


__all__ = [
    "CONFIG_FILENAME",
    "MANIFEST_FILENAME",
    "PORTABLE_BUNDLE_FORMAT",
    "PORTABLE_INITIALIZATION_FORMAT",
    "PORTABLE_STATE_FORMAT",
    "PORTABLE_TRAINED_INITIALIZATION_FORMAT",
    "PORTABLE_TRAINING_PROVENANCE_FORMAT",
    "TOKENIZER_DIRECTORY",
    "PortableBundle",
    "PortableBundleError",
    "WEIGHTS_FILENAME",
    "canonical_tensor_schema",
    "create_portable_bundle",
    "ensure_portable_bundle",
    "load_hybrid_config",
    "load_portable_bundle",
    "publish_conversational_foundation_bundle",
]
