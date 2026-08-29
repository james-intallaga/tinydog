"""Bounded, memory-only ingress for future native audio/video inference.

The current arithmetic model does not consume these packets.  Keeping the
transport contract separate lets the browser capture layer be exercised now
without pretending that an untrained multimodal model already exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import re
from threading import Lock
from typing import Mapping


MAX_FRAME_BYTES = 512 * 1024
MAX_AUDIO_BYTES = 1024 * 1024
_SESSION = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_AUDIO_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/wav",
    "audio/x-wav",
    "application/octet-stream",
}


@dataclass(frozen=True)
class FramePacket:
    session_id: str
    timestamp_ms: int
    difference_percent: float
    forced_keyframe: bool
    content_type: str
    sha256: str
    payload: bytes

    def public_metadata(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("payload")
        value["bytes"] = len(self.payload)
        return value


@dataclass(frozen=True)
class AudioPacket:
    session_id: str
    timestamp_ms: int
    sequence: int
    content_type: str
    sha256: str
    payload: bytes

    def public_metadata(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("payload")
        value["bytes"] = len(self.payload)
        return value


def _header(headers: Mapping[str, str], name: str, default: str = "") -> str:
    """Read a header from ordinary or case-insensitive mappings."""

    direct = headers.get(name)
    if direct is not None:
        return str(direct)
    lowered = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lowered:
            return str(value)
    return default


def _session_id(headers: Mapping[str, str]) -> str:
    session_id = _header(headers, "X-MLM-Session").strip()
    if not _SESSION.fullmatch(session_id):
        raise ValueError("X-MLM-Session must be 1-128 safe identifier characters")
    return session_id


def _integer_header(headers: Mapping[str, str], name: str, *, minimum: int = 0) -> int:
    raw = _header(headers, name).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class RealtimeIngress:
    """Thread-safe transient packet queues with strict request validation."""

    def __init__(self, *, frame_capacity: int = 32, audio_capacity: int = 128) -> None:
        if frame_capacity <= 0 or audio_capacity <= 0:
            raise ValueError("Ingress capacities must be positive")
        self._frames: deque[FramePacket] = deque(maxlen=frame_capacity)
        self._audio: deque[AudioPacket] = deque(maxlen=audio_capacity)
        self._lock = Lock()
        self._frame_received = 0
        self._audio_received = 0

    def accept_frame(
        self,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str],
    ) -> FramePacket:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime != "image/jpeg":
            raise ValueError("Video frames must use image/jpeg")
        if not 4 <= len(payload) <= MAX_FRAME_BYTES:
            raise ValueError(f"Frame body must be 4-{MAX_FRAME_BYTES} bytes")
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise ValueError("Frame body is not a complete JPEG")
        difference_raw = _header(headers, "X-MLM-Difference").strip()
        try:
            difference = float(difference_raw)
        except ValueError as error:
            raise ValueError("X-MLM-Difference must be a percentage") from error
        if not 0.0 <= difference <= 100.0:
            raise ValueError("X-MLM-Difference must be between 0 and 100")
        keyframe_raw = _header(headers, "X-MLM-Keyframe", "false").strip().lower()
        if keyframe_raw not in {"true", "false", "1", "0"}:
            raise ValueError("X-MLM-Keyframe must be true or false")
        packet = FramePacket(
            session_id=_session_id(headers),
            timestamp_ms=_integer_header(headers, "X-MLM-Timestamp-Ms"),
            difference_percent=difference,
            forced_keyframe=keyframe_raw in {"true", "1"},
            content_type=mime,
            sha256=hashlib.sha256(payload).hexdigest(),
            payload=bytes(payload),
        )
        with self._lock:
            self._frames.append(packet)
            self._frame_received += 1
        return packet

    def accept_audio(
        self,
        payload: bytes,
        content_type: str,
        headers: Mapping[str, str],
    ) -> AudioPacket:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime not in _AUDIO_TYPES:
            raise ValueError(f"Unsupported audio content type: {mime or 'missing'}")
        if not 1 <= len(payload) <= MAX_AUDIO_BYTES:
            raise ValueError(f"Audio body must be 1-{MAX_AUDIO_BYTES} bytes")
        packet = AudioPacket(
            session_id=_session_id(headers),
            timestamp_ms=_integer_header(headers, "X-MLM-Timestamp-Ms"),
            sequence=_integer_header(headers, "X-MLM-Sequence"),
            content_type=mime,
            sha256=hashlib.sha256(payload).hexdigest(),
            payload=bytes(payload),
        )
        with self._lock:
            self._audio.append(packet)
            self._audio_received += 1
        return packet

    def drain_frames(self, *, limit: int | None = None) -> list[FramePacket]:
        return self._drain(self._frames, limit)

    def drain_audio(self, *, limit: int | None = None) -> list[AudioPacket]:
        return self._drain(self._audio, limit)

    def _drain(self, queue: deque, limit: int | None) -> list:
        if limit is not None and limit < 0:
            raise ValueError("Drain limit must not be negative")
        with self._lock:
            count = len(queue) if limit is None else min(limit, len(queue))
            return [queue.popleft() for _ in range(count)]

    def status(self) -> dict[str, object]:
        with self._lock:
            last_frame = self._frames[-1].public_metadata() if self._frames else None
            last_audio = self._audio[-1].public_metadata() if self._audio else None
            return {
                "ready": True,
                "transport": "http-ingress",
                "webrtc_ready": False,
                "model_connected": False,
                "persistence": "memory-only",
                "frames_received": self._frame_received,
                "audio_chunks_received": self._audio_received,
                "pending_frames": len(self._frames),
                "pending_audio_chunks": len(self._audio),
                "last_frame": last_frame,
                "last_audio": last_audio,
            }
