"""Strict, memory-only protocol primitives for MLM's loopback realtime gateway.

This module deliberately contains no network server and performs no logging.  In
particular, callers must not log admission tokens, SDP, ICE candidates, or media
payloads when handling :class:`RealtimeProtocolError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
import math
import re
import secrets
from threading import Lock
import time
from typing import Callable, Collection, Mapping
from urllib.parse import urlsplit


REALTIME_PROTOCOL_VERSION = 1
ADMISSION_TTL_MS = 30_000
MAX_SIGNAL_BYTES = 64 * 1024
MAX_SELECTED_FRAME_HEADER_BYTES = 4 * 1024
MAX_SELECTED_FRAME_BYTES = 512 * 1024
MAX_SAFE_INTEGER = 2**53 - 1

_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
_ERROR_CODE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_MAX_TOKEN_BYTES = 8 * 1024
_MAX_TEXT_FIELD_BYTES = 60 * 1024
_MAX_CANDIDATE_BYTES = 16 * 1024


class RealtimeProtocolError(ValueError):
    """A safe-to-display protocol rejection without rejected wire values."""


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _safe_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise RealtimeProtocolError(f"Realtime {name} is invalid.")
    return value


def _bounded_string(
    value: object,
    name: str,
    *,
    maximum_bytes: int = _MAX_TEXT_FIELD_BYTES,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RealtimeProtocolError(f"Realtime {name} is invalid.")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise RealtimeProtocolError(f"Realtime {name} is not valid UTF-8.") from error
    if len(encoded) > maximum_bytes:
        raise RealtimeProtocolError(f"Realtime {name} is too large.")
    return value


def _safe_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > MAX_SAFE_INTEGER
    ):
        raise RealtimeProtocolError(f"Realtime {name} is invalid.")
    return value


def _finite_number(
    value: object, name: str, *, minimum: float, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealtimeProtocolError(f"Realtime {name} is invalid.")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise RealtimeProtocolError(f"Realtime {name} is invalid.")
    return number


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RealtimeProtocolError("Realtime JSON contains a duplicate field.")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> object:
    raise RealtimeProtocolError("Realtime JSON contains a non-finite number.")


def _decode_json(raw: str | bytes, *, maximum_bytes: int) -> object:
    if isinstance(raw, bytes):
        wire = raw
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise RealtimeProtocolError("Realtime JSON is not valid UTF-8.") from error
    elif isinstance(raw, str):
        text = raw
        try:
            wire = raw.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise RealtimeProtocolError("Realtime JSON is not valid UTF-8.") from error
    else:
        raise RealtimeProtocolError("Realtime message must be text.")
    if not wire or len(wire) > maximum_bytes:
        raise RealtimeProtocolError("Realtime message is empty or too large.")
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_reject_nonfinite,
        )
    except RealtimeProtocolError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise RealtimeProtocolError("Realtime message is not valid JSON.") from error


def _require_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RealtimeProtocolError("Realtime message must be a JSON object.")
    return value


def _require_fields(
    value: Mapping[str, object],
    *,
    required: Collection[str],
    optional: Collection[str] = (),
) -> None:
    required_set = set(required)
    keys = set(value)
    if not required_set.issubset(keys):
        raise RealtimeProtocolError("Realtime message is missing a required field.")
    if not keys.issubset(required_set | set(optional)):
        raise RealtimeProtocolError("Realtime message contains an unsupported field.")


def _candidate(value: object) -> dict[str, object]:
    candidate = _require_object(value)
    _require_fields(
        candidate,
        required={"candidate"},
        optional={"sdpMid", "sdpMLineIndex", "usernameFragment"},
    )
    result: dict[str, object] = {
        "candidate": _bounded_string(
            candidate["candidate"],
            "ICE candidate",
            maximum_bytes=_MAX_CANDIDATE_BYTES,
            allow_empty=True,
        )
    }
    for key in ("sdpMid", "usernameFragment"):
        item = candidate.get(key)
        if item is not None:
            result[key] = _bounded_string(item, key, maximum_bytes=1_024)
    line_index = candidate.get("sdpMLineIndex")
    if line_index is not None:
        result["sdpMLineIndex"] = _safe_integer(line_index, "ICE media-line index")
    return result


@dataclass(frozen=True, slots=True)
class AdmissionClaims:
    """Non-secret identity returned after successful one-use admission."""

    session_id: str
    connection_id: str
    expires_at_unix_ms: int


@dataclass(frozen=True, slots=True)
class AdmissionTicket:
    """A freshly issued ticket.  Its representation intentionally hides the token."""

    session_id: str
    connection_id: str
    expires_at_unix_ms: int
    access_token: str = field(repr=False)


class AdmissionTokenStore:
    """Thread-safe, one-use 30-second admission tickets held only in memory.

    Only SHA-256 token digests are retained.  Restarting the process therefore
    invalidates every outstanding ticket, which is desirable for a local gateway.
    """

    def __init__(self, *, clock_ms: Callable[[], int] = _now_ms) -> None:
        self._clock_ms = clock_ms
        self._tickets: dict[bytes, AdmissionClaims] = {}
        self._lock = Lock()

    @staticmethod
    def _digest(token: str) -> bytes:
        return hashlib.sha256(token.encode("ascii")).digest()

    def issue(self, session_id: str, connection_id: str) -> AdmissionTicket:
        session = _safe_identifier(session_id, "session ID")
        connection = _safe_identifier(connection_id, "connection ID")
        now = self._clock_ms()
        expires = now + ADMISSION_TTL_MS
        token = secrets.token_urlsafe(32)
        claims = AdmissionClaims(session, connection, expires)
        digest = self._digest(token)
        with self._lock:
            self._purge_expired_locked(now)
            while digest in self._tickets:  # Defensive against a mocked RNG collision.
                token = secrets.token_urlsafe(32)
                digest = self._digest(token)
            self._tickets[digest] = claims
        return AdmissionTicket(session, connection, expires, token)

    def consume(
        self, access_token: str, session_id: str, connection_id: str
    ) -> AdmissionClaims:
        session = _safe_identifier(session_id, "session ID")
        connection = _safe_identifier(connection_id, "connection ID")
        if (
            not isinstance(access_token, str)
            or _TOKEN.fullmatch(access_token) is None
            or len(access_token) > _MAX_TOKEN_BYTES
        ):
            raise RealtimeProtocolError("Realtime admission was rejected.")
        digest = self._digest(access_token)
        now = self._clock_ms()
        with self._lock:
            self._purge_expired_locked(now)
            claims = self._tickets.get(digest)
            if (
                claims is None
                or claims.session_id != session
                or claims.connection_id != connection
                or claims.expires_at_unix_ms <= now
            ):
                raise RealtimeProtocolError("Realtime admission was rejected.")
            del self._tickets[digest]
        return claims

    def revoke(self, access_token: str) -> None:
        if not isinstance(access_token, str):
            return
        try:
            digest = self._digest(access_token)
        except (UnicodeEncodeError, AttributeError):
            return
        with self._lock:
            self._tickets.pop(digest, None)

    def purge_expired(self) -> int:
        with self._lock:
            return self._purge_expired_locked(self._clock_ms())

    def _purge_expired_locked(self, now_ms: int) -> int:
        expired = [
            digest
            for digest, claims in self._tickets.items()
            if claims.expires_at_unix_ms <= now_ms
        ]
        for digest in expired:
            del self._tickets[digest]
        return len(expired)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._tickets)


def is_loopback_host(host: str) -> bool:
    """Return true only for literal loopback addresses or the exact localhost name."""

    if not isinstance(host, str):
        return False
    candidate = host.strip().lower()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    candidate = candidate.rstrip(".")
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _host_from_authority(authority: str) -> str:
    if (
        not isinstance(authority, str)
        or not authority
        or any(character in authority for character in "/?#@")
    ):
        raise RealtimeProtocolError("Realtime Host header is invalid.")
    try:
        parsed = urlsplit(f"//{authority}")
        port = parsed.port
    except ValueError as error:
        raise RealtimeProtocolError("Realtime Host header is invalid.") from error
    if not parsed.hostname or parsed.username or parsed.password:
        raise RealtimeProtocolError("Realtime Host header is invalid.")
    if port is not None and not 1 <= port <= 65_535:
        raise RealtimeProtocolError("Realtime Host header is invalid.")
    return parsed.hostname


def normalize_origin(origin: str) -> str:
    """Validate and normalize an HTTP(S) browser origin without resolving DNS."""

    if not isinstance(origin, str) or not origin or len(origin) > 2_048:
        raise RealtimeProtocolError("Realtime Origin header is invalid.")
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as error:
        raise RealtimeProtocolError("Realtime Origin header is invalid.") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RealtimeProtocolError("Realtime Origin header is invalid.")
    host = parsed.hostname.lower().rstrip(".")
    if not is_loopback_host(host):
        raise RealtimeProtocolError("Realtime Origin must be loopback.")
    default_port = 80 if parsed.scheme == "http" else 443
    authority_host = f"[{host}]" if ":" in host else host
    authority = (
        authority_host if port in {None, default_port} else f"{authority_host}:{port}"
    )
    return f"{parsed.scheme}://{authority}"


def validate_loopback_request(
    *,
    peer_host: str,
    host_header: str,
    origin: str,
    allowed_origins: Collection[str],
) -> str:
    """Validate the socket peer, Host header, and an explicit browser allowlist.

    Forwarded headers are intentionally not accepted: the MLM gateway is owned by
    this process and is expected to bind directly to a loopback interface.
    """

    if not is_loopback_host(peer_host):
        raise RealtimeProtocolError("Realtime connections are loopback-only.")
    if not is_loopback_host(_host_from_authority(host_header)):
        raise RealtimeProtocolError("Realtime Host must be loopback.")
    normalized = normalize_origin(origin)
    normalized_allowed = {normalize_origin(item) for item in allowed_origins}
    if normalized not in normalized_allowed:
        raise RealtimeProtocolError("Realtime Origin is not allowed.")
    return normalized


_CLIENT_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "join": (
        {"type", "protocolVersion", "accessToken", "sessionId", "connectionId"},
        set(),
    ),
    "ping": ({"type", "clientTimeMs", "connectionId"}, set()),
    "offer": (
        {"type", "connectionId", "requestId", "revision", "sdp", "iceRestart"},
        set(),
    ),
    "answer": (
        {"type", "connectionId", "requestId", "revision", "sdp"},
        set(),
    ),
    "ice": ({"type", "connectionId", "revision", "candidate"}, set()),
    "interrupt": (
        {"type", "connectionId", "playbackGeneration"},
        set(),
    ),
    "playbackSilent": (
        {"type", "connectionId", "playbackGeneration"},
        set(),
    ),
    "turnEnded": ({"type", "connectionId", "turnId"}, set()),
    "firstResponseAudio": ({"type", "connectionId", "turnId"}, set()),
    "close": ({"type", "connectionId"}, set()),
}


def parse_client_signal(raw: str | bytes) -> dict[str, object]:
    """Parse and validate one browser-to-gateway JSON signalling message."""

    message = _require_object(_decode_json(raw, maximum_bytes=MAX_SIGNAL_BYTES))
    signal_type = message.get("type")
    if not isinstance(signal_type, str) or signal_type not in _CLIENT_FIELDS:
        raise RealtimeProtocolError("Realtime signal type is not supported.")
    required, optional = _CLIENT_FIELDS[signal_type]
    _require_fields(message, required=required, optional=optional)

    result: dict[str, object] = {"type": signal_type}
    if signal_type == "join":
        protocol_version = _safe_integer(
            message["protocolVersion"], "protocol version", minimum=1
        )
        if protocol_version != REALTIME_PROTOCOL_VERSION:
            raise RealtimeProtocolError("Realtime protocol version is not supported.")
        token = _bounded_string(
            message["accessToken"],
            "admission token",
            maximum_bytes=_MAX_TOKEN_BYTES,
        )
        if _TOKEN.fullmatch(token) is None:
            raise RealtimeProtocolError("Realtime admission token is invalid.")
        result.update(
            protocolVersion=REALTIME_PROTOCOL_VERSION,
            accessToken=token,
            sessionId=_safe_identifier(message["sessionId"], "session ID"),
            connectionId=_safe_identifier(message["connectionId"], "connection ID"),
        )
        return result

    result["connectionId"] = _safe_identifier(message["connectionId"], "connection ID")
    if signal_type == "ping":
        result["clientTimeMs"] = _safe_integer(message["clientTimeMs"], "client time")
    elif signal_type in {"offer", "answer"}:
        result["requestId"] = _safe_identifier(message["requestId"], "request ID")
        result["revision"] = _safe_integer(message["revision"], "revision", minimum=1)
        result["sdp"] = _bounded_string(message["sdp"], "SDP")
        if signal_type == "offer":
            if not isinstance(message["iceRestart"], bool):
                raise RealtimeProtocolError("Realtime ICE restart flag is invalid.")
            result["iceRestart"] = message["iceRestart"]
    elif signal_type == "ice":
        result["revision"] = _safe_integer(message["revision"], "revision", minimum=1)
        result["candidate"] = _candidate(message["candidate"])
    elif signal_type in {"interrupt", "playbackSilent"}:
        result["playbackGeneration"] = _safe_integer(
            message["playbackGeneration"], "playback generation", minimum=1
        )
    elif signal_type in {"turnEnded", "firstResponseAudio"}:
        result["turnId"] = _safe_identifier(message["turnId"], "turn ID")
    return result


_SERVER_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "joined": ({"type", "connectionId"}, {"heartbeatIntervalMs"}),
    "pong": ({"type", "connectionId"}, set()),
    "answer": (
        {"type", "connectionId", "requestId", "revision", "sdp"},
        set(),
    ),
    "offer": (
        {"type", "connectionId", "requestId", "revision", "sdp"},
        {"iceRestart"},
    ),
    "ice": ({"type", "connectionId", "revision", "candidate"}, set()),
    "restart": ({"type", "connectionId"}, set()),
    "frameAccepted": (
        {"type", "connectionId", "frameId", "sequence"},
        set(),
    ),
    "error": ({"type", "code", "message"}, {"retryable"}),
}


def _validated_server_signal(message: Mapping[str, object]) -> dict[str, object]:
    signal_type = message.get("type")
    if not isinstance(signal_type, str) or signal_type not in _SERVER_FIELDS:
        raise RealtimeProtocolError("Realtime signal type is not supported.")
    required, optional = _SERVER_FIELDS[signal_type]
    _require_fields(message, required=required, optional=optional)
    result: dict[str, object] = {"type": signal_type}
    if signal_type == "error":
        code = message["code"]
        if not isinstance(code, str) or _ERROR_CODE.fullmatch(code) is None:
            raise RealtimeProtocolError("Realtime error code is invalid.")
        result["code"] = code
        result["message"] = _bounded_string(
            message["message"], "error message", maximum_bytes=1_024
        )
        retryable = message.get("retryable")
        if retryable is not None:
            if not isinstance(retryable, bool):
                raise RealtimeProtocolError("Realtime retryable flag is invalid.")
            result["retryable"] = retryable
        return result

    result["connectionId"] = _safe_identifier(message["connectionId"], "connection ID")
    if signal_type == "joined":
        heartbeat = message.get("heartbeatIntervalMs")
        if heartbeat is not None:
            result["heartbeatIntervalMs"] = _safe_integer(
                heartbeat, "heartbeat interval", minimum=1_000
            )
            if result["heartbeatIntervalMs"] > 60_000:
                raise RealtimeProtocolError("Realtime heartbeat interval is invalid.")
    elif signal_type in {"offer", "answer"}:
        result["requestId"] = _safe_identifier(message["requestId"], "request ID")
        result["revision"] = _safe_integer(message["revision"], "revision", minimum=1)
        result["sdp"] = _bounded_string(message["sdp"], "SDP")
        ice_restart = message.get("iceRestart")
        if ice_restart is not None:
            if signal_type != "offer" or not isinstance(ice_restart, bool):
                raise RealtimeProtocolError("Realtime ICE restart flag is invalid.")
            result["iceRestart"] = ice_restart
    elif signal_type == "ice":
        result["revision"] = _safe_integer(message["revision"], "revision", minimum=1)
        result["candidate"] = _candidate(message["candidate"])
    elif signal_type == "frameAccepted":
        result["frameId"] = _safe_identifier(message["frameId"], "frame ID")
        result["sequence"] = _safe_integer(message["sequence"], "frame sequence")
    return result


def serialize_server_signal(message: Mapping[str, object]) -> str:
    """Validate and compactly serialize one gateway-to-browser signal."""

    if not isinstance(message, Mapping):
        raise RealtimeProtocolError("Realtime server signal must be an object.")
    normalized = _validated_server_signal(message)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > MAX_SIGNAL_BYTES:
        raise RealtimeProtocolError("Realtime signal is too large.")
    return encoded


class MonotonicSequenceGuard:
    """Accept only strictly increasing JavaScript-safe integer sequences."""

    def __init__(self, name: str, *, initial: int, minimum: int) -> None:
        self._name = name
        self._value = initial
        self._minimum = minimum

    @property
    def value(self) -> int:
        return self._value

    def advance(self, value: object) -> int:
        candidate = _safe_integer(value, self._name, minimum=self._minimum)
        if candidate <= self._value:
            raise RealtimeProtocolError(
                f"Realtime {self._name} must increase monotonically."
            )
        self._value = candidate
        return candidate

    def require_current(self, value: object) -> int:
        candidate = _safe_integer(value, self._name, minimum=self._minimum)
        if candidate != self._value:
            raise RealtimeProtocolError(f"Realtime {self._name} is stale.")
        return candidate


class RealtimeSequenceGuards:
    """Per-connection negotiation, interruption, and selected-frame guards."""

    def __init__(self) -> None:
        self.revision = MonotonicSequenceGuard("revision", initial=0, minimum=1)
        self.playback = MonotonicSequenceGuard(
            "playback generation", initial=0, minimum=1
        )
        self.playback_acknowledged = MonotonicSequenceGuard(
            "acknowledged playback generation", initial=0, minimum=1
        )
        self.frame = MonotonicSequenceGuard("frame sequence", initial=-1, minimum=0)
        self.frame_acknowledged = MonotonicSequenceGuard(
            "acknowledged frame sequence", initial=-1, minimum=0
        )

    def accept_revision(self, revision: object) -> int:
        return self.revision.advance(revision)

    def require_current_revision(self, revision: object) -> int:
        return self.revision.require_current(revision)

    def accept_playback_generation(self, generation: object) -> int:
        return self.playback.advance(generation)

    def acknowledge_playback_generation(self, generation: object) -> int:
        candidate = _safe_integer(generation, "playback generation", minimum=1)
        if candidate > self.playback.value:
            raise RealtimeProtocolError("Realtime playback acknowledgement is unknown.")
        return self.playback_acknowledged.advance(candidate)

    def accept_frame_sequence(self, sequence: object) -> int:
        return self.frame.advance(sequence)

    def acknowledge_frame_sequence(self, sequence: object) -> int:
        candidate = _safe_integer(sequence, "frame sequence")
        if candidate > self.frame.value:
            raise RealtimeProtocolError("Realtime frame acknowledgement is unknown.")
        return self.frame_acknowledged.advance(candidate)


@dataclass(frozen=True, slots=True)
class SelectedFrame:
    version: int
    frame_id: str
    sequence: int
    timestamp_ms: int
    difference_percent: float
    keyframe: bool
    content_type: str
    jpeg: bytes = field(repr=False)


_FRAME_HEADER_FIELDS = {
    "version",
    "frameId",
    "sequence",
    "timestampMs",
    "differencePercent",
    "keyframe",
    "contentType",
}


def parse_selected_frame_packet(
    packet: bytes | bytearray | memoryview,
) -> SelectedFrame:
    """Parse the MLM selected-frame data-channel wire format.

    The packet is ``uint32_be header_length + UTF-8 JSON header + exact JPEG``.
    The JPEG body itself is capped at 512 KiB, matching the browser producer.
    """

    if not isinstance(packet, (bytes, bytearray, memoryview)):
        raise RealtimeProtocolError("Selected frame packet must be binary.")
    try:
        view = memoryview(packet).cast("B")
    except (TypeError, ValueError) as error:
        raise RealtimeProtocolError("Selected frame packet is invalid.") from error
    if len(view) < 8:
        raise RealtimeProtocolError("Selected frame packet is incomplete.")
    header_length = int.from_bytes(view[:4], byteorder="big", signed=False)
    if not 1 <= header_length <= MAX_SELECTED_FRAME_HEADER_BYTES:
        raise RealtimeProtocolError("Selected frame header size is invalid.")
    jpeg_offset = 4 + header_length
    if jpeg_offset + 4 > len(view):
        raise RealtimeProtocolError("Selected frame packet is incomplete.")
    jpeg_size = len(view) - jpeg_offset
    if jpeg_size > MAX_SELECTED_FRAME_BYTES:
        raise RealtimeProtocolError("Selected frame JPEG is too large.")
    header = _require_object(
        _decode_json(
            bytes(view[4:jpeg_offset]), maximum_bytes=MAX_SELECTED_FRAME_HEADER_BYTES
        )
    )
    _require_fields(header, required=_FRAME_HEADER_FIELDS)
    version = _safe_integer(header["version"], "frame protocol version", minimum=1)
    if version != REALTIME_PROTOCOL_VERSION:
        raise RealtimeProtocolError("Selected frame protocol version is not supported.")
    if header["contentType"] != "image/jpeg":
        raise RealtimeProtocolError("Selected frame content type is invalid.")
    if not isinstance(header["keyframe"], bool):
        raise RealtimeProtocolError("Selected frame keyframe flag is invalid.")
    jpeg = bytes(view[jpeg_offset:])
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise RealtimeProtocolError("Selected frame is not a complete JPEG.")
    return SelectedFrame(
        version=REALTIME_PROTOCOL_VERSION,
        frame_id=_safe_identifier(header["frameId"], "frame ID"),
        sequence=_safe_integer(header["sequence"], "frame sequence"),
        timestamp_ms=_safe_integer(header["timestampMs"], "frame timestamp"),
        difference_percent=_finite_number(
            header["differencePercent"],
            "frame difference percentage",
            minimum=0,
            maximum=100,
        ),
        keyframe=header["keyframe"],
        content_type="image/jpeg",
        jpeg=jpeg,
    )
