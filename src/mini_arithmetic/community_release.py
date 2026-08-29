"""Owner-signed active floaty releases with a local anti-rollback chain.

Community members train the floaty layer, but an official active release is a
separate act: the owner signs the exact community contract and active floaty
weights after external evaluation.  This module never trains, aggregates,
evaluates, or promotes a candidate automatically.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from .community_updates import (
    CommunityContract,
    read_floaty_safetensors,
    tensor_specs_from_floaty_schema,
)
from .foundation_release import (
    VerifiedFoundation,
    verify_foundation_manifest,
)


COMMUNITY_RELEASE_SCHEMA_VERSION = 1
COMMUNITY_RELEASE_FORMAT = "tinydog-community-release-v1"
CONTRACT_FILENAME = "contract.json"
WEIGHTS_FILENAME = "floaty.safetensors"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "format",
        "sequence",
        "community_version",
        "foundation_manifest_sha256",
        "previous_release_sha256",
        "created_at",
        "owner_key_id",
        "artifacts",
        "signature",
    }
)
_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_TRUST_STATE_KEYS = frozenset(
    {
        "schema_version",
        "foundation_manifest_sha256",
        "sequence",
        "release_sha256",
    }
)
_TRUST_STATE_THREAD_LOCK = threading.RLock()


@dataclass(frozen=True)
class ReleaseArtifact:
    name: str
    path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class VerifiedCommunityRelease:
    manifest_path: Path
    release_sha256: str
    foundation_manifest_sha256: str
    sequence: int
    community_version: str
    previous_release_sha256: str | None
    owner_key_id: str
    contract: CommunityContract
    artifacts: Mapping[str, ReleaseArtifact]


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _snapshot(path: Path, *, maximum: int, label: str) -> bytes:
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
            or metadata.st_size > maximum
        ):
            raise ValueError(f"{label} has an invalid file type or size")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size or os.read(descriptor, 1):
            raise ValueError(f"{label} changed while it was read")
        return payload
    finally:
        os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_at must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("created_at is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("created_at must use UTC")
    return value


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("owner private key cannot be a symbolic link")
    raw = _snapshot(
        requested.parent.resolve() / requested.name,
        maximum=32,
        label="owner private key",
    )
    if len(raw) != 32:
        raise ValueError("owner private key must be a raw 32-byte Ed25519 key")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _load_public_key(path: str | Path) -> tuple[Ed25519PublicKey, bytes]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("owner public key cannot be a symbolic link")
    raw = _snapshot(
        requested.parent.resolve() / requested.name,
        maximum=32,
        label="owner public key",
    )
    if len(raw) != 32:
        raise ValueError("owner public key must be a raw 32-byte Ed25519 key")
    return Ed25519PublicKey.from_public_bytes(raw), raw


def _relative_file(root: Path, value: str | Path, label: str) -> Path:
    source = Path(value)
    requested = root / source if not source.is_absolute() else source
    if requested.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    candidate = requested.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{label} must stay inside the community release root"
        ) from error
    return candidate


def _artifact_descriptor(
    root: Path, path: Path, label: str, maximum: int
) -> dict[str, Any]:
    payload = _snapshot(path, maximum=maximum, label=label)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(payload),
        "bytes": len(payload),
    }


def _verified_artifact_bytes(
    foundation: VerifiedFoundation,
    name: str,
    *,
    maximum: int,
) -> bytes:
    artifact = foundation.artifacts[name]
    payload = _snapshot(artifact.path, maximum=maximum, label=f"foundation {name}")
    if len(payload) != artifact.byte_count or _sha256(payload) != artifact.sha256:
        raise ValueError(f"signed foundation {name} changed after verification")
    return payload


def _validate_contract_against_foundation(
    contract: CommunityContract,
    foundation: VerifiedFoundation,
) -> None:
    """Bind every compatibility field derivable from the signed foundation."""

    if contract.foundation_sha256 != foundation.foundation_sha256:
        raise ValueError("community contract targets different foundation weights")
    if contract.tokenizer_sha256 != foundation.tokenizer_sha256:
        raise ValueError("community contract targets a different tokenizer")
    if contract.architecture_sha256 != foundation.architecture_sha256:
        raise ValueError("community contract targets a different floaty architecture")

    metadata = _strict_object(
        _verified_artifact_bytes(
            foundation,
            "tokenizer_metadata",
            maximum=1024 * 1024,
        ),
        "foundation tokenizer metadata",
    )
    if metadata.get("tokenizer_sha256") != foundation.tokenizer_sha256:
        raise ValueError("foundation tokenizer metadata does not bind the tokenizer")

    model_config = _strict_object(
        _verified_artifact_bytes(
            foundation,
            "model_config",
            maximum=1024 * 1024,
        ),
        "foundation model config",
    )
    floaty_schema = _strict_object(
        _verified_artifact_bytes(
            foundation,
            "floaty_schema",
            maximum=1024 * 1024,
        ),
        "foundation floaty schema",
    )
    if floaty_schema.get("foundation_config") != model_config:
        raise ValueError("floaty schema does not bind the signed foundation config")
    try:
        schema_specs = tensor_specs_from_floaty_schema(floaty_schema["tensors"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "signed floaty schema has an invalid tensor contract"
        ) from error
    if schema_specs != contract.tensor_specs:
        raise ValueError("community contract tensor schema differs from the foundation")


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_chain_fields(sequence: Any, previous: Any) -> tuple[int, str | None]:
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("community release sequence must be a non-negative integer")
    if sequence == 0:
        if previous is not None:
            raise ValueError("sequence zero cannot have a previous release")
        return sequence, None
    return sequence, _require_sha256(previous, "previous_release_sha256")


def create_community_release_manifest(
    root: str | Path,
    *,
    sequence: int,
    foundation_manifest: str | Path,
    trusted_public_key: str | Path,
    contract: str | Path = CONTRACT_FILENAME,
    weights: str | Path = WEIGHTS_FILENAME,
    private_key: str | Path,
    previous_release_sha256: str | None = None,
    output: str | Path = "community-release.json",
) -> Path:
    """Sign one already-evaluated active floaty contract and weight file."""

    sequence, previous_release_sha256 = _validate_chain_fields(
        sequence, previous_release_sha256
    )
    raw_release_root = Path(root).expanduser()
    if raw_release_root.is_symlink():
        raise ValueError("community release root cannot be a symbolic link")
    release_root = raw_release_root.resolve()
    if not release_root.is_dir():
        raise ValueError("community release root must be a real directory")
    foundation = verify_foundation_manifest(
        foundation_manifest,
        trusted_public_key,
    )
    contract_path = _relative_file(release_root, contract, "contract")
    weights_path = _relative_file(release_root, weights, "weights")
    contract_payload = _snapshot(contract_path, maximum=1024 * 1024, label="contract")
    contract_value = _strict_object(contract_payload, "contract")
    active_contract = CommunityContract.from_dict(contract_value)
    if not _VERSION_RE.fullmatch(active_contract.community_version):
        raise ValueError("community_version is invalid")
    _validate_contract_against_foundation(active_contract, foundation)
    # This consumes one bounded no-follow snapshot and checks the exact schema,
    # dtype, finiteness, and signed base hash before release signing.
    read_floaty_safetensors(
        weights_path,
        active_contract.tensor_specs,
        expected_sha256=active_contract.community_weights_sha256,
    )
    artifacts = {
        "contract": {
            "path": contract_path.relative_to(release_root).as_posix(),
            "sha256": _sha256(contract_payload),
            "bytes": len(contract_payload),
        },
        "weights": _artifact_descriptor(
            release_root,
            weights_path,
            "weights",
            64 * 1024 * 1024,
        ),
    }
    if artifacts["weights"]["sha256"] != active_contract.community_weights_sha256:
        raise ValueError("active weights changed while the release was being signed")
    signing_key = _load_private_key(private_key)
    public_raw = signing_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    if _sha256(public_raw) != foundation.owner_key_id:
        raise ValueError(
            "community release signing key is not the foundation owner key"
        )
    body: dict[str, Any] = {
        "schema_version": COMMUNITY_RELEASE_SCHEMA_VERSION,
        "format": COMMUNITY_RELEASE_FORMAT,
        "sequence": sequence,
        "community_version": active_contract.community_version,
        "foundation_manifest_sha256": foundation.manifest_sha256,
        "previous_release_sha256": previous_release_sha256,
        "created_at": _timestamp(),
        "owner_key_id": _sha256(public_raw),
        "artifacts": artifacts,
    }
    signature = signing_key.sign(_canonical(body))
    manifest = {**body, "signature": base64.b64encode(signature).decode("ascii")}
    destination = _relative_file(release_root, output, "manifest")
    _exclusive_write(
        destination,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n",
    )
    _fsync_directory(release_root)
    return destination


def verify_community_release(
    manifest_path: str | Path,
    trusted_public_key: str | Path,
    *,
    foundation_manifest: str | Path,
    minimum_sequence: int | None = None,
    expected_previous_release_sha256: str | None = None,
) -> VerifiedCommunityRelease:
    """Verify owner signature, active artifacts, compatibility, and chain pins."""

    raw_path = Path(manifest_path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("community release manifest cannot be a symbolic link")
    path = raw_path.resolve()
    payload = _snapshot(path, maximum=1024 * 1024, label="community release manifest")
    manifest = _strict_object(payload, "community release manifest")
    if frozenset(manifest) != _ROOT_KEYS:
        raise ValueError("community release manifest fields are incompatible")
    if (
        manifest["schema_version"] != COMMUNITY_RELEASE_SCHEMA_VERSION
        or manifest["format"] != COMMUNITY_RELEASE_FORMAT
    ):
        raise ValueError("community release version or format is incompatible")
    sequence, previous = _validate_chain_fields(
        manifest["sequence"], manifest["previous_release_sha256"]
    )
    foundation_manifest_sha256 = _require_sha256(
        manifest["foundation_manifest_sha256"], "foundation_manifest_sha256"
    )
    foundation = verify_foundation_manifest(foundation_manifest, trusted_public_key)
    if foundation_manifest_sha256 != foundation.manifest_sha256:
        raise ValueError("community release targets a different foundation manifest")
    if minimum_sequence is not None:
        if (
            isinstance(minimum_sequence, bool)
            or not isinstance(minimum_sequence, int)
            or minimum_sequence < 0
        ):
            raise ValueError("minimum_sequence must be a non-negative integer")
        if sequence < minimum_sequence:
            raise ValueError("community release is a rollback")
    if expected_previous_release_sha256 is not None:
        required_previous = _require_sha256(
            expected_previous_release_sha256,
            "expected_previous_release_sha256",
        )
        if previous != required_previous:
            raise ValueError("community release does not extend the expected chain")
    _validate_timestamp(manifest["created_at"])
    if not isinstance(manifest["community_version"], str) or not _VERSION_RE.fullmatch(
        manifest["community_version"]
    ):
        raise ValueError("community_version is invalid")

    public_key, public_raw = _load_public_key(trusted_public_key)
    owner_key_id = _sha256(public_raw)
    if manifest["owner_key_id"] != owner_key_id:
        raise ValueError("community release was not signed by the trusted owner")
    try:
        signature_value = manifest["signature"]
        if not isinstance(signature_value, str):
            raise ValueError("community release signature is invalid")
        signature = base64.b64decode(signature_value, validate=True)
        body = {key: value for key, value in manifest.items() if key != "signature"}
        public_key.verify(signature, _canonical(body))
    except (InvalidSignature, ValueError) as error:
        raise ValueError("community release signature verification failed") from error

    artifacts_value = manifest["artifacts"]
    if not isinstance(artifacts_value, dict) or set(artifacts_value) != {
        "contract",
        "weights",
    }:
        raise ValueError("community release artifact set is incompatible")
    root = path.parent
    artifacts: dict[str, ReleaseArtifact] = {}
    snapshots: dict[str, bytes] = {}
    for name, maximum in (("contract", 1024 * 1024), ("weights", 64 * 1024 * 1024)):
        descriptor = artifacts_value[name]
        if not isinstance(descriptor, dict) or frozenset(descriptor) != _ARTIFACT_KEYS:
            raise ValueError(f"{name} release artifact descriptor is invalid")
        relative = descriptor["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise ValueError(f"{name} release artifact path is invalid")
        requested_artifact = root / relative
        if requested_artifact.is_symlink():
            raise ValueError(f"{name} release artifact cannot be a symbolic link")
        artifact_path = requested_artifact.resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} release artifact escapes its root") from error
        artifact_payload = _snapshot(artifact_path, maximum=maximum, label=name)
        digest = _sha256(artifact_payload)
        if (
            isinstance(descriptor["bytes"], bool)
            or descriptor["bytes"] != len(artifact_payload)
            or descriptor["sha256"] != digest
        ):
            raise ValueError(f"{name} release artifact does not match its signature")
        snapshots[name] = artifact_payload
        artifacts[name] = ReleaseArtifact(
            name=name,
            path=artifact_path,
            sha256=digest,
            byte_count=len(artifact_payload),
        )

    contract_value = _strict_object(snapshots["contract"], "contract")
    contract = CommunityContract.from_dict(contract_value)
    if contract.community_version != manifest["community_version"]:
        raise ValueError("signed community version does not match the contract")
    if contract.community_weights_sha256 != artifacts["weights"].sha256:
        raise ValueError("signed active weights do not match the contract")
    _validate_contract_against_foundation(contract, foundation)
    read_floaty_safetensors(
        artifacts["weights"].path,
        contract.tensor_specs,
        expected_sha256=artifacts["weights"].sha256,
    )
    return VerifiedCommunityRelease(
        manifest_path=path,
        release_sha256=_sha256(_canonical(manifest)),
        foundation_manifest_sha256=foundation_manifest_sha256,
        sequence=sequence,
        community_version=manifest["community_version"],
        previous_release_sha256=previous,
        owner_key_id=owner_key_id,
        contract=contract,
        artifacts=MappingProxyType(artifacts),
    )


@contextmanager
def _trust_state_lock(destination: Path):
    lock_path = destination.with_name(f".{destination.name}.lock")
    flags = (
        os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    with _TRUST_STATE_THREAD_LOCK:
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise ValueError("community trust-state lock is unsafe") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("community trust-state lock is not a regular file")
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover - non-POSIX fallback
                pass
            os.close(descriptor)


def _advance_trusted_community_state_locked(
    destination: Path,
    release: VerifiedCommunityRelease,
) -> Path:
    if destination.exists():
        current_payload = _snapshot(
            destination,
            maximum=64 * 1024,
            label="community trust state",
        )
        current = _strict_object(current_payload, "community trust state")
        if frozenset(current) != _TRUST_STATE_KEYS or current["schema_version"] != 1:
            raise ValueError("community trust state is incompatible")
        current_foundation = _require_sha256(
            current["foundation_manifest_sha256"],
            "trusted foundation_manifest_sha256",
        )
        current_release = _require_sha256(
            current["release_sha256"], "trusted release_sha256"
        )
        current_sequence = current["sequence"]
        if isinstance(current_sequence, bool) or not isinstance(current_sequence, int):
            raise ValueError("community trust state sequence is invalid")
        if (
            release.sequence == current_sequence
            and release.release_sha256 == current_release
            and release.foundation_manifest_sha256 == current_foundation
        ):
            return destination
        if release.foundation_manifest_sha256 != current_foundation:
            raise ValueError("community release changes the trusted foundation")
        if release.sequence != current_sequence + 1:
            raise ValueError(
                "community release is a rollback or skips the trusted chain"
            )
        if release.previous_release_sha256 != current_release:
            raise ValueError("community release forks the trusted chain")
    elif release.sequence != 0 or release.previous_release_sha256 is not None:
        raise ValueError(
            "a new trust state must start from community release sequence zero"
        )

    state = {
        "schema_version": 1,
        "foundation_manifest_sha256": release.foundation_manifest_sha256,
        "sequence": release.sequence,
        "release_sha256": release.release_sha256,
    }
    payload = json.dumps(state, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def advance_trusted_community_state(
    state_path: str | Path,
    release: VerifiedCommunityRelease,
) -> Path:
    """Atomically remember the newest verified release and reject forks/rollback."""

    if not isinstance(release, VerifiedCommunityRelease):
        raise TypeError("release must be a VerifiedCommunityRelease")
    requested = Path(state_path).expanduser()
    if requested.is_symlink():
        raise ValueError("community trust state cannot be a symbolic link")
    parent = requested.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / requested.name
    with _trust_state_lock(destination):
        if destination.is_symlink():
            raise ValueError("community trust state cannot be a symbolic link")
        return _advance_trusted_community_state_locked(destination, release)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sign = commands.add_parser("sign", help="sign an evaluated active floaty release")
    sign.add_argument("--root", required=True)
    sign.add_argument("--sequence", type=int, required=True)
    sign.add_argument("--foundation-manifest", required=True)
    sign.add_argument("--public-key", required=True)
    sign.add_argument("--contract", default=CONTRACT_FILENAME)
    sign.add_argument("--weights", default=WEIGHTS_FILENAME)
    sign.add_argument("--private-key", required=True)
    sign.add_argument("--previous-release-sha256")
    sign.add_argument("--output", default="community-release.json")
    verify = commands.add_parser("verify", help="verify a signed active floaty release")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--public-key", required=True)
    verify.add_argument("--foundation-manifest", required=True)
    verify.add_argument("--minimum-sequence", type=int)
    verify.add_argument("--expected-previous-release-sha256")
    verify.add_argument("--trust-state")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "sign":
        path = create_community_release_manifest(
            args.root,
            sequence=args.sequence,
            foundation_manifest=args.foundation_manifest,
            trusted_public_key=args.public_key,
            contract=args.contract,
            weights=args.weights,
            private_key=args.private_key,
            previous_release_sha256=args.previous_release_sha256,
            output=args.output,
        )
        output: dict[str, Any] = {"manifest": str(path)}
    else:
        release = verify_community_release(
            args.manifest,
            args.public_key,
            foundation_manifest=args.foundation_manifest,
            minimum_sequence=args.minimum_sequence,
            expected_previous_release_sha256=args.expected_previous_release_sha256,
        )
        if args.trust_state:
            advance_trusted_community_state(args.trust_state, release)
        output = {
            "release_sha256": release.release_sha256,
            "foundation_manifest_sha256": release.foundation_manifest_sha256,
            "sequence": release.sequence,
            "community_version": release.community_version,
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
