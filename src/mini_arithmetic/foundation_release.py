"""Owner-signed, immutable foundation release manifests.

Community clients may train approved floaty tensors, but an official
foundation is trusted only when every artifact is hash-bound to a manifest
signed by the owner's pinned Ed25519 key.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
from types import MappingProxyType
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


FOUNDATION_SCHEMA_VERSION = 1
FOUNDATION_FORMAT = "tinydog-foundation-v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ARTIFACT_LIMITS = {
    "weights": 8 * 1024 * 1024 * 1024,
    "model_config": 1024 * 1024,
    "tokenizer": 64 * 1024 * 1024,
    "tokenizer_metadata": 1024 * 1024,
    "floaty_schema": 1024 * 1024,
}
_ROOT_FIELDS = {
    "schema_version",
    "format",
    "foundation_id",
    "created_at",
    "owner_key_id",
    "artifacts",
    "signature",
}
_ARTIFACT_FIELDS = {"path", "sha256", "bytes"}


@dataclass(frozen=True)
class VerifiedArtifact:
    """One signed artifact descriptor safe to re-snapshot before model load."""

    name: str
    path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class VerifiedFoundation:
    manifest_path: Path
    foundation_id: str
    manifest_sha256: str
    owner_key_id: str
    foundation_sha256: str
    architecture_sha256: str
    tokenizer_sha256: str
    artifacts: Mapping[str, VerifiedArtifact]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _strict_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error


def _read_json_object(
    path: Path, *, limit: int, label: str
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file(path, limit=limit, label=label)
    value = _strict_json(raw, label)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, raw


def _read_regular_file(path: Path, *, limit: int, label: str) -> bytes:
    descriptor, metadata = _open_regular_file(path, limit=limit, label=label)
    try:
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != metadata.st_size or os.read(descriptor, 1):
            raise ValueError(f"{label} changed while it was being read")
        return raw
    finally:
        os.close(descriptor)


def _open_regular_file(
    path: Path, *, limit: int, label: str
) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > limit
        ):
            raise ValueError(f"{label} has an invalid size or file type")
        return descriptor, metadata
    except Exception:
        os.close(descriptor)
        raise


def _exclusive_write(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _framed(hasher: Any, payload: bytes) -> None:
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


_SEMANTIC_DTYPES = {
    "F16": "mlx.core.float16",
    "BF16": "mlx.core.bfloat16",
    "F32": "mlx.core.float32",
}


def _semantic_foundation_sha256(
    weights_path: Path,
    *,
    expected_file_sha256: str,
    expected_byte_count: int,
    model_config: Mapping[str, Any],
) -> str:
    """Reproduce FoundationFloatyLM.foundation_hash without allocating MLX tensors."""

    descriptor, metadata = _open_regular_file(
        weights_path,
        limit=_ARTIFACT_LIMITS["weights"],
        label="weights",
    )
    try:
        if metadata.st_size != expected_byte_count:
            raise ValueError(
                "signed foundation weights changed before semantic hashing"
            )
        prefix = os.read(descriptor, 8)
        if len(prefix) != 8:
            raise ValueError("weights must be a complete safetensors file")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 1 or header_size > 64 * 1024 * 1024:
            raise ValueError("weights safetensors header is invalid")
        header_payload = bytearray()
        remaining = header_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            header_payload.extend(chunk)
            remaining -= len(chunk)
        if len(header_payload) != header_size:
            raise ValueError("weights safetensors header is truncated")
        header = _strict_json(bytes(header_payload), "weights safetensors header")
        if not isinstance(header, dict):
            raise ValueError("weights safetensors header must be an object")
        _validate_safetensors_header(
            bytes(header_payload),
            data_size=metadata.st_size - 8 - header_size,
        )

        # Bind the semantic digest to the exact already-signed container bytes
        # while keeping this same descriptor open for the later tensor reads.
        os.lseek(descriptor, 0, os.SEEK_SET)
        file_hasher = hashlib.sha256()
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            file_hasher.update(chunk)
            remaining -= len(chunk)
        if remaining or file_hasher.hexdigest() != expected_file_sha256:
            raise ValueError(
                "signed foundation weights changed before semantic hashing"
            )

        tensors = {
            name: value for name, value in header.items() if name != "__metadata__"
        }
        hasher = hashlib.sha256()
        _framed(hasher, b"tinydog.foundation.HybridLM.v1")
        _framed(hasher, _canonical(dict(model_config)))
        _framed(hasher, str(len(tensors)).encode("ascii"))
        data_start = 8 + header_size
        for name in sorted(tensors):
            value = tensors[name]
            dtype = value["dtype"]
            if dtype not in _SEMANTIC_DTYPES:
                raise ValueError("foundation weights use an unsupported semantic dtype")
            start, end = value["data_offsets"]
            _framed(hasher, name.encode("utf-8"))
            _framed(hasher, _SEMANTIC_DTYPES[dtype].encode("ascii"))
            _framed(hasher, _canonical({"shape": value["shape"]}))
            length = end - start
            hasher.update(length.to_bytes(8, "big"))
            os.lseek(descriptor, data_start + start, os.SEEK_SET)
            remaining = length
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    break
                hasher.update(chunk)
                remaining -= len(chunk)
            if remaining:
                raise ValueError("foundation tensor data is truncated")
        return hasher.hexdigest()
    finally:
        os.close(descriptor)


def _validate_safetensors_header(header_payload: bytes, *, data_size: int) -> None:
    header = _strict_json(header_payload, "weights safetensors header")
    if not isinstance(header, dict) or not header:
        raise ValueError("weights safetensors header must contain tensors")
    ranges: list[tuple[int, int]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            if descriptor is not None and not isinstance(descriptor, dict):
                raise ValueError("weights safetensors metadata is invalid")
            continue
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise ValueError("weights safetensors tensor descriptor is invalid")
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("weights safetensors tensor fields are incompatible")
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if (
            not isinstance(descriptor["dtype"], str)
            or not isinstance(shape, list)
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 0
                for size in shape
            )
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(offset, bool) or not isinstance(offset, int)
                for offset in offsets
            )
        ):
            raise ValueError("weights safetensors tensor shape or offsets are invalid")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ValueError("weights safetensors tensor data is out of bounds")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("weights safetensors file contains no tensors")
    ordered = sorted(ranges)
    if any(previous[1] > current[0] for previous, current in zip(ordered, ordered[1:])):
        raise ValueError("weights safetensors tensor data overlaps")


def _inspect_artifact(path: Path, *, limit: int, label: str) -> tuple[str, int]:
    """Hash and structurally validate one artifact from a single no-follow fd."""

    descriptor, metadata = _open_regular_file(path, limit=limit, label=label)
    digest = hashlib.sha256()
    total_read = 0
    try:
        if label == "weights":
            prefix = os.read(descriptor, 8)
            if len(prefix) != 8:
                raise ValueError("weights must be a complete safetensors file")
            digest.update(prefix)
            total_read += len(prefix)
            header_size = struct.unpack("<Q", prefix)[0]
            if (
                header_size <= 1
                or header_size > 64 * 1024 * 1024
                or 8 + header_size > metadata.st_size
            ):
                raise ValueError("weights safetensors header is invalid")
            header = bytearray()
            remaining_header = header_size
            while remaining_header:
                chunk = os.read(descriptor, min(remaining_header, 1024 * 1024))
                if not chunk:
                    break
                header.extend(chunk)
                digest.update(chunk)
                total_read += len(chunk)
                remaining_header -= len(chunk)
            if len(header) != header_size:
                raise ValueError("weights safetensors header is truncated")
            _validate_safetensors_header(
                bytes(header),
                data_size=metadata.st_size - 8 - header_size,
            )

        chunks: list[bytes] = []
        while total_read < metadata.st_size:
            chunk = os.read(descriptor, min(metadata.st_size - total_read, 1024 * 1024))
            if not chunk:
                break
            digest.update(chunk)
            total_read += len(chunk)
            if label != "weights":
                chunks.append(chunk)
        if total_read != metadata.st_size or os.read(descriptor, 1):
            raise ValueError(f"{label} changed while it was being inspected")
        if label != "weights":
            raw = b"".join(chunks)
            value = _strict_json(raw, label)
            if not isinstance(value, dict):
                raise ValueError(f"{label} must contain a JSON object")
        return digest.hexdigest(), metadata.st_size
    finally:
        os.close(descriptor)


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _resolved_parent_path(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    return candidate.parent.resolve() / candidate.name


def generate_owner_keypair(private_path: str | Path, public_path: str | Path) -> str:
    """Create a new owner keypair without overwriting either destination."""

    private_destination = _resolved_parent_path(private_path, label="private key")
    public_destination = _resolved_parent_path(public_path, label="public key")
    if private_destination == public_destination:
        raise ValueError("private and public key paths must differ")
    if private_destination.exists() or public_destination.exists():
        raise FileExistsError("owner key destination already exists")
    private_key = Ed25519PrivateKey.generate()
    private_raw = private_key.private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    public_raw = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    _exclusive_write(private_destination, private_raw, 0o600)
    try:
        _exclusive_write(public_destination, public_raw, 0o644)
    except Exception:
        private_destination.unlink(missing_ok=True)
        raise
    return _sha256(public_raw)


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    raw = _read_regular_file(
        _resolved_parent_path(path, label="owner private key"),
        limit=32,
        label="owner private key",
    )
    if len(raw) != 32:
        raise ValueError("owner private key must be a raw 32-byte Ed25519 key")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _load_public_key(path: str | Path) -> tuple[Ed25519PublicKey, bytes]:
    raw = _read_regular_file(
        _resolved_parent_path(path, label="trusted owner public key"),
        limit=32,
        label="trusted owner public key",
    )
    if len(raw) != 32:
        raise ValueError("trusted owner public key must be a raw 32-byte Ed25519 key")
    return Ed25519PublicKey.from_public_bytes(raw), raw


def _relative_artifact(
    root: Path,
    path: str | Path,
    *,
    label: str,
) -> tuple[str, str, int]:
    candidate = Path(path)
    requested = root / candidate if not candidate.is_absolute() else candidate
    if requested.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    resolved = requested.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the foundation root") from error
    if relative == Path(".") or ".." in relative.parts:
        raise ValueError(f"{label} path is invalid")
    digest, byte_count = _inspect_artifact(
        resolved,
        limit=_ARTIFACT_LIMITS[label],
        label=label,
    )
    return relative.as_posix(), digest, byte_count


def create_foundation_manifest(
    root: str | Path,
    *,
    foundation_id: str,
    weights: str | Path,
    model_config: str | Path,
    tokenizer: str | Path,
    tokenizer_metadata: str | Path,
    floaty_schema: str | Path,
    private_key: str | Path,
    output: str | Path = "foundation.json",
) -> Path:
    """Hash, sign, and atomically publish one foundation manifest."""

    if not _SAFE_ID.fullmatch(foundation_id):
        raise ValueError("foundation_id must be 1-64 safe identifier characters")
    raw_release_root = Path(root).expanduser()
    if raw_release_root.is_symlink():
        raise ValueError("foundation root cannot be a symbolic link")
    release_root = raw_release_root.resolve()
    if not release_root.is_dir():
        raise ValueError("foundation root must be a real directory")
    artifacts: dict[str, dict[str, str | int]] = {}
    for label, source in (
        ("weights", weights),
        ("model_config", model_config),
        ("tokenizer", tokenizer),
        ("tokenizer_metadata", tokenizer_metadata),
        ("floaty_schema", floaty_schema),
    ):
        relative, digest, byte_count = _relative_artifact(
            release_root,
            source,
            label=label,
        )
        artifacts[label] = {
            "path": relative,
            "sha256": digest,
            "bytes": byte_count,
        }
    signing_key = _load_private_key(private_key)
    public_raw = signing_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    body: dict[str, Any] = {
        "schema_version": FOUNDATION_SCHEMA_VERSION,
        "format": FOUNDATION_FORMAT,
        "foundation_id": foundation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "owner_key_id": _sha256(public_raw),
        "artifacts": artifacts,
    }
    signature = signing_key.sign(_canonical(body))
    manifest = {**body, "signature": base64.b64encode(signature).decode("ascii")}
    output_path = Path(output)
    destination = (
        (release_root / output_path).resolve()
        if not output_path.is_absolute()
        else output_path.resolve()
    )
    try:
        destination.relative_to(release_root)
    except ValueError as error:
        raise ValueError(
            "foundation manifest must remain inside the foundation root"
        ) from error
    _exclusive_write(
        destination,
        json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n",
        0o644,
    )
    return destination


def verify_foundation_manifest(
    manifest_path: str | Path,
    trusted_public_key: str | Path,
) -> VerifiedFoundation:
    """Verify the pinned owner signature and every referenced artifact."""

    path = _resolved_parent_path(manifest_path, label="foundation manifest")
    manifest, _ = _read_json_object(
        path, limit=1024 * 1024, label="foundation manifest"
    )
    if set(manifest) != _ROOT_FIELDS:
        raise ValueError("foundation manifest fields are incompatible")
    if (
        manifest["schema_version"] != FOUNDATION_SCHEMA_VERSION
        or manifest["format"] != FOUNDATION_FORMAT
    ):
        raise ValueError("foundation manifest version or format is incompatible")
    if not isinstance(manifest["foundation_id"], str) or not _SAFE_ID.fullmatch(
        manifest["foundation_id"]
    ):
        raise ValueError("foundation manifest ID is invalid")
    public_key, public_raw = _load_public_key(trusted_public_key)
    key_id = _sha256(public_raw)
    if manifest["owner_key_id"] != key_id:
        raise ValueError("foundation was not signed by the trusted owner key")
    signature_value = manifest["signature"]
    if not isinstance(signature_value, str):
        raise ValueError("foundation signature is invalid")
    try:
        signature = base64.b64decode(signature_value, validate=True)
        body = {key: value for key, value in manifest.items() if key != "signature"}
        public_key.verify(signature, _canonical(body))
    except (ValueError, InvalidSignature) as error:
        raise ValueError("foundation signature verification failed") from error
    artifacts_value = manifest["artifacts"]
    if not isinstance(artifacts_value, dict) or set(artifacts_value) != set(
        _ARTIFACT_LIMITS
    ):
        raise ValueError("foundation artifact set is incompatible")
    resolved_artifacts: dict[str, VerifiedArtifact] = {}
    root = path.parent
    for label, descriptor in artifacts_value.items():
        if not isinstance(descriptor, dict) or set(descriptor) != _ARTIFACT_FIELDS:
            raise ValueError(f"{label} artifact descriptor is incompatible")
        relative = descriptor["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ValueError(f"{label} artifact path is invalid")
        requested = root / relative
        if requested.is_symlink():
            raise ValueError(f"{label} artifact cannot be a symbolic link")
        candidate = requested.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} artifact escapes the foundation root") from error
        if (
            isinstance(descriptor["bytes"], bool)
            or not isinstance(descriptor["bytes"], int)
            or descriptor["bytes"] <= 0
            or not isinstance(descriptor["sha256"], str)
        ):
            raise ValueError(f"{label} artifact descriptor is invalid")
        digest, byte_count = _inspect_artifact(
            candidate,
            limit=_ARTIFACT_LIMITS[label],
            label=label,
        )
        if descriptor["bytes"] != byte_count or descriptor["sha256"] != digest:
            raise ValueError(f"{label} artifact does not match the signed foundation")
        resolved_artifacts[label] = VerifiedArtifact(
            name=label,
            path=candidate,
            sha256=digest,
            byte_count=byte_count,
        )

    def verified_payload(label: str) -> bytes:
        artifact = resolved_artifacts[label]
        payload = _read_regular_file(
            artifact.path,
            limit=_ARTIFACT_LIMITS[label],
            label=label,
        )
        if len(payload) != artifact.byte_count or _sha256(payload) != artifact.sha256:
            raise ValueError(f"{label} changed after foundation verification")
        return payload

    model_config = _strict_json(verified_payload("model_config"), "model_config")
    floaty_schema = _strict_json(verified_payload("floaty_schema"), "floaty_schema")
    tokenizer_metadata = _strict_json(
        verified_payload("tokenizer_metadata"),
        "tokenizer_metadata",
    )
    if not all(
        isinstance(value, dict)
        for value in (model_config, floaty_schema, tokenizer_metadata)
    ):
        raise ValueError("foundation JSON artifacts must contain objects")
    tokenizer_sha256 = resolved_artifacts["tokenizer"].sha256
    if tokenizer_metadata.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("tokenizer metadata does not bind the signed tokenizer")
    if floaty_schema.get("foundation_config") != model_config:
        raise ValueError("floaty schema does not bind the signed model config")
    architecture_sha256 = _sha256(_canonical(floaty_schema))
    foundation_sha256 = _semantic_foundation_sha256(
        resolved_artifacts["weights"].path,
        expected_file_sha256=resolved_artifacts["weights"].sha256,
        expected_byte_count=resolved_artifacts["weights"].byte_count,
        model_config=model_config,
    )
    return VerifiedFoundation(
        manifest_path=path,
        foundation_id=manifest["foundation_id"],
        manifest_sha256=_sha256(_canonical(manifest)),
        owner_key_id=key_id,
        foundation_sha256=foundation_sha256,
        architecture_sha256=architecture_sha256,
        tokenizer_sha256=tokenizer_sha256,
        artifacts=MappingProxyType(resolved_artifacts),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser("keygen", help="Generate the owner's Ed25519 keypair")
    keygen.add_argument("--private", required=True)
    keygen.add_argument("--public", required=True)
    sign = commands.add_parser("sign", help="Sign an immutable foundation release")
    sign.add_argument("--root", required=True)
    sign.add_argument("--foundation-id", required=True)
    sign.add_argument("--weights", required=True)
    sign.add_argument("--model-config", required=True)
    sign.add_argument("--tokenizer", required=True)
    sign.add_argument("--tokenizer-metadata", required=True)
    sign.add_argument("--floaty-schema", required=True)
    sign.add_argument("--private-key", required=True)
    sign.add_argument("--output", default="foundation.json")
    verify = commands.add_parser(
        "verify", help="Verify a foundation against the pinned owner key"
    )
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--public-key", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "keygen":
        print(generate_owner_keypair(args.private, args.public))
    elif args.command == "sign":
        print(
            create_foundation_manifest(
                args.root,
                foundation_id=args.foundation_id,
                weights=args.weights,
                model_config=args.model_config,
                tokenizer=args.tokenizer,
                tokenizer_metadata=args.tokenizer_metadata,
                floaty_schema=args.floaty_schema,
                private_key=args.private_key,
                output=args.output,
            )
        )
    else:
        verified = verify_foundation_manifest(args.manifest, args.public_key)
        print(
            json.dumps(
                {
                    "foundation_id": verified.foundation_id,
                    "manifest_sha256": verified.manifest_sha256,
                    "owner_key_id": verified.owner_key_id,
                    "foundation_sha256": verified.foundation_sha256,
                    "architecture_sha256": verified.architecture_sha256,
                    "tokenizer_sha256": verified.tokenizer_sha256,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
