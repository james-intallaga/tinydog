"""Loopback-only WebRTC gateway for MLM's native realtime runtime.

The gateway owns signalling and media transport. Its normal untrained runtime
is silent. An explicit ``--untrained-config`` diagnostic may return the native
randomly initialized model's own meaningless waveform; this never elevates any
trained or conversation-readiness flag.

Admission tokens, SDP, ICE candidates, and media payloads are deliberately never
logged or placed in URLs.  The default CLI binds only to a literal loopback
address and accepts browser origins from the local MLM interface allowlist.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any, Protocol
import uuid

from aiohttp import WSMsgType, web
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from aiortc.sdp import candidate_from_sdp
import numpy as np

from .realtime_latency import RealtimeLatencyTracker
from .realtime_media import (
    BoundedPcmInput,
    MicrophoneConsumer,
    PcmBlock,
    QueuedPcmAudioTrack,
)
from .realtime_protocol import (
    ADMISSION_TTL_MS,
    MAX_SELECTED_FRAME_BYTES,
    MAX_SELECTED_FRAME_HEADER_BYTES,
    MAX_SIGNAL_BYTES,
    REALTIME_PROTOCOL_VERSION,
    AdmissionTokenStore,
    RealtimeProtocolError,
    RealtimeSequenceGuards,
    SelectedFrame,
    is_loopback_host,
    normalize_origin,
    parse_client_signal,
    parse_selected_frame_packet,
    serialize_server_signal,
    validate_loopback_request,
)


DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8_787
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
DEFAULT_HEARTBEAT_INTERVAL_MS = 5_000
DEFAULT_JOIN_TIMEOUT_SECONDS = 5.0
CLOSE_TIMEOUT_SECONDS = 2.0
SELECTED_FRAME_CHANNEL_LABEL = "mlm-selected-frames-v1"
MAX_ADMISSION_BYTES = 4 * 1024
MAX_PENDING_ADMISSIONS = 64
# A 300M native checkpoint is currently single-conversation by design.  The
# process shares one immutable model while each connection receives isolated
# streaming caches; the cap keeps that first measured resource envelope strict.
MAX_ACTIVE_CONNECTIONS = 1
MAX_PENDING_FRAMES = 2
MAX_PENDING_TELEMETRY = 8
MAX_SEEN_TURN_IDS = 64
MAX_FRAME_PACKET_BYTES = 4 + MAX_SELECTED_FRAME_HEADER_BYTES + MAX_SELECTED_FRAME_BYTES


class NativeAudioOutputLike(Protocol):
    """One runtime-produced 24 kHz mono PCM block."""

    generation: int
    sequence: int
    produced_at: float
    samples: np.ndarray


class NativeSessionRuntime(Protocol):
    """Per-peer native runtime contract consumed by the gateway."""

    def status(self) -> dict[str, object]: ...

    async def accept_audio(self, block: PcmBlock) -> bool: ...

    async def accept_frame(self, frame: SelectedFrame) -> bool: ...

    async def next_output(self) -> NativeAudioOutputLike: ...

    async def interrupt(self, generation: int) -> int: ...

    async def close(self) -> None: ...


RuntimeFactory = Callable[
    [str, str], NativeSessionRuntime | Awaitable[NativeSessionRuntime]
]
PeerFactory = Callable[[], RTCPeerConnection]
_UNTRAINED_FACTORY: RuntimeFactory | None = None


class _UnavailableNativeRuntime:
    """Fail-closed runtime used when no native model runtime can be loaded."""

    def __init__(self) -> None:
        self._outputs: asyncio.Queue[NativeAudioOutputLike] = asyncio.Queue(maxsize=1)
        self._generation = 0

    def status(self) -> dict[str, object]:
        return {
            "model_loaded": False,
            "model_connected": False,
            "checkpoint_loaded": False,
            "voice_trained": False,
            "vision_trained": False,
            "conversation_ready": False,
            "can_emit_audio": False,
            "random_init": False,
            "untrained_audio_enabled": False,
            "generation": self._generation,
        }

    async def accept_audio(self, block: PcmBlock) -> bool:
        del block
        return False

    async def accept_frame(self, frame: SelectedFrame) -> bool:
        del frame
        return False

    async def next_output(self) -> NativeAudioOutputLike:
        return await self._outputs.get()

    async def interrupt(self, generation: int) -> int:
        self._generation = generation
        return generation

    async def close(self) -> None:
        return None


def _default_runtime_factory(
    session_id: str, connection_id: str
) -> NativeSessionRuntime:
    """Create the local scratch runtime without importing any external model.

    The lazy import keeps the transport independently testable.  A missing
    native-runtime module fails closed instead of silently substituting ASR,
    TTS, an API, or pretrained weights.
    """

    global _UNTRAINED_FACTORY
    try:
        from .native_runtime import NativeRuntimeFactory
    except ImportError:
        return _UnavailableNativeRuntime()
    if _UNTRAINED_FACTORY is None:
        _UNTRAINED_FACTORY = NativeRuntimeFactory.untrained()
    return _UNTRAINED_FACTORY(session_id, connection_id)


def checkpoint_runtime_factory(manifest_path: str | Path) -> RuntimeFactory:
    """Build a per-session factory for a verified local native checkpoint.

    The manifest is resolved before the listener starts.  The native loader is
    responsible for scratch provenance, checksum, training-gate, and backend
    validation.  The returned factory shares immutable weights while producing
    isolated stream state; the gateway still admits only one measured realtime
    conversation at a time.
    """

    resolved = Path(manifest_path).expanduser().resolve(strict=True)
    if not (resolved.is_file() or resolved.is_dir()):
        raise ValueError("Native checkpoint must be a local file or directory")
    from .native_runtime import load_native_runtime_factory

    # This performs strict manifest/checksum/gate validation and loads the
    # immutable MLX weights exactly once, before aiohttp starts listening.
    return load_native_runtime_factory(resolved)


def random_runtime_factory(config_path: str | Path, *, seed: int = 0) -> RuntimeFactory:
    """Randomly initialize MLM's native structure without loading weights."""

    resolved = Path(config_path).expanduser().resolve(strict=True)
    from .native_runtime import load_random_native_runtime_factory

    return load_random_native_runtime_factory(resolved, seed=seed)


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    bind_host: str = DEFAULT_BIND_HOST
    port: int = DEFAULT_PORT
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    heartbeat_interval_ms: int = DEFAULT_HEARTBEAT_INTERVAL_MS
    join_timeout_seconds: float = DEFAULT_JOIN_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not is_loopback_host(self.bind_host):
            raise ValueError("Realtime gateway must bind to a literal loopback host")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65_535
        ):
            raise ValueError("Realtime gateway port must be in [1, 65535]")
        if not self.allowed_origins:
            raise ValueError("At least one local browser origin must be allowed")
        normalized = tuple(normalize_origin(item) for item in self.allowed_origins)
        object.__setattr__(self, "allowed_origins", normalized)
        if (
            isinstance(self.heartbeat_interval_ms, bool)
            or not isinstance(self.heartbeat_interval_ms, int)
            or not 1_000 <= self.heartbeat_interval_ms <= 60_000
        ):
            raise ValueError("Heartbeat interval must be in [1000, 60000] ms")
        if (
            isinstance(self.join_timeout_seconds, bool)
            or not isinstance(self.join_timeout_seconds, (int, float))
            or not 0.0 < float(self.join_timeout_seconds) <= ADMISSION_TTL_MS / 1_000
        ):
            raise ValueError(
                "Join timeout must be positive and no longer than admission TTL"
            )

    @property
    def signalling_url(self) -> str:
        host = f"[{self.bind_host}]" if ":" in self.bind_host else self.bind_host
        return f"ws://{host}:{self.port}/realtime"


@dataclass(slots=True)
class GatewayCounters:
    admissions_issued: int = 0
    joins_accepted: int = 0
    audio_blocks_received: int = 0
    audio_blocks_accepted: int = 0
    selected_frames_received: int = 0
    selected_frames_accepted: int = 0
    selected_frames_rejected: int = 0
    selected_frames_dropped: int = 0
    output_blocks_enqueued: int = 0
    output_blocks_rejected: int = 0
    interruptions: int = 0
    closed_input_drops: int = 0
    closed_output_drops: int = 0


def _runtime_flags(runtime: NativeSessionRuntime) -> dict[str, bool]:
    """Read only explicit ``True`` readiness claims from a runtime."""

    try:
        status = runtime.status()
    except Exception:
        status = {}
    if not isinstance(status, Mapping):
        status = {}
    model_loaded = status.get("model_loaded") is True
    model_connected = status.get("model_connected") is True
    checkpoint_loaded = status.get("checkpoint_loaded") is True
    voice_trained = status.get("voice_trained") is True
    vision_trained = status.get("vision_trained") is True
    can_emit_audio = status.get("can_emit_audio") is True
    random_init = status.get("random_init") is True
    untrained_audio_enabled = status.get("untrained_audio_enabled") is True
    declared_conversation_ready = status.get("conversation_ready") is True
    return {
        "model_loaded": model_loaded,
        "model_connected": model_connected,
        "checkpoint_loaded": checkpoint_loaded,
        "voice_trained": voice_trained,
        "vision_trained": vision_trained,
        "can_emit_audio": can_emit_audio,
        "random_init": random_init,
        "untrained_audio_enabled": untrained_audio_enabled,
        "conversation_ready": (
            model_loaded
            and model_connected
            and checkpoint_loaded
            and voice_trained
            and vision_trained
            and can_emit_audio
            and declared_conversation_ready
        ),
    }


def _peer_host(request: web.Request) -> str:
    transport = request.transport
    peer = transport.get_extra_info("peername") if transport is not None else None
    if not isinstance(peer, (tuple, list)) or not peer or not isinstance(peer[0], str):
        raise RealtimeProtocolError("Realtime connections are loopback-only.")
    return peer[0]


def _validate_request_origin(
    request: web.Request, config: GatewayConfig, *, required: bool
) -> None:
    origin = request.headers.get("Origin")
    if not origin:
        if required:
            raise RealtimeProtocolError("Realtime Origin header is required.")
        # A read-only health request from a local process does not normally have
        # a browser Origin.  Supplying a known allowed origin reuses the strict
        # peer and Host checks without weakening admission or WebSocket policy.
        origin = config.allowed_origins[0]
    validate_loopback_request(
        peer_host=_peer_host(request),
        host_header=request.headers.get("Host", ""),
        origin=origin,
        allowed_origins=config.allowed_origins,
    )


def _no_store(response: web.StreamResponse) -> web.StreamResponse:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _json_error(message: str, *, status: int) -> web.Response:
    return _no_store(
        web.json_response(
            {"error": {"code": "realtime_rejected", "message": message}},
            status=status,
        )
    )


def _strict_json_object(raw: bytes) -> dict[str, object]:
    if not raw or len(raw) > MAX_ADMISSION_BYTES:
        raise RealtimeProtocolError("Realtime admission body is empty or too large.")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RealtimeProtocolError("Realtime admission is not valid UTF-8.") from error

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RealtimeProtocolError(
                    "Realtime admission contains a duplicate field."
                )
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=no_duplicates)
    except RealtimeProtocolError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        raise RealtimeProtocolError("Realtime admission is not valid JSON.") from error
    if not isinstance(value, dict):
        raise RealtimeProtocolError("Realtime admission must be a JSON object.")
    return value


def _candidate_from_wire(value: Mapping[str, object]) -> RTCIceCandidate | None:
    candidate_sdp = value.get("candidate")
    if not isinstance(candidate_sdp, str):
        raise RealtimeProtocolError("Realtime ICE candidate is invalid.")
    if candidate_sdp == "":
        return None
    raw = candidate_sdp.removeprefix("candidate:")
    try:
        candidate = candidate_from_sdp(raw)
    except (AssertionError, IndexError, TypeError, ValueError) as error:
        raise RealtimeProtocolError("Realtime ICE candidate is invalid.") from error
    sdp_mid = value.get("sdpMid")
    line_index = value.get("sdpMLineIndex")
    if sdp_mid is None and line_index is None:
        raise RealtimeProtocolError(
            "Realtime ICE candidate is missing its media identity."
        )
    candidate.sdpMid = sdp_mid if isinstance(sdp_mid, str) else None
    candidate.sdpMLineIndex = line_index if isinstance(line_index, int) else None
    return candidate


class RealtimeGatewaySession:
    """One admitted WebSocket, WebRTC peer, and native runtime."""

    def __init__(
        self,
        *,
        gateway: RealtimeGateway,
        websocket: web.WebSocketResponse,
        runtime: NativeSessionRuntime,
        session_id: str,
        connection_id: str,
    ) -> None:
        self.gateway = gateway
        self.websocket = websocket
        self.runtime = runtime
        self.session_id = session_id
        self.connection_id = connection_id
        self.peer = gateway.peer_factory()
        self.audio_input = BoundedPcmInput()
        self.audio_output = QueuedPcmAudioTrack()
        self.guards = RealtimeSequenceGuards()
        self.frame_queue: asyncio.Queue[SelectedFrame] = asyncio.Queue(
            maxsize=MAX_PENDING_FRAMES
        )
        self._tasks: set[asyncio.Task[None]] = set()
        self._send_lock = asyncio.Lock()
        self._microphone_task: asyncio.Task[None] | None = None
        self._microphone_seen = False
        self._frame_channel: Any | None = None
        self._last_output_sequence = -1
        self._pending_interrupts: OrderedDict[int, str] = OrderedDict()
        self._pending_turns: OrderedDict[str, bool] = OrderedDict()
        self._seen_turn_order: deque[str] = deque()
        self._seen_turn_ids: set[str] = set()
        self._active_response_turn: str | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._started = False
        self._fatal_started = False

    @property
    def transport_connected(self) -> bool:
        return self.peer.connectionState == "connected"

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Realtime session was already started")
        self._started = True
        self.peer.addTrack(self.audio_output)

        @self.peer.on("track")
        def on_track(track: Any) -> None:
            if track.kind != "audio":
                track.stop()
                return
            if self._microphone_seen:
                track.stop()
                return
            self._microphone_seen = True
            self._microphone_task = self._spawn(self._consume_microphone(track))

        @self.peer.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            if (
                channel.label != SELECTED_FRAME_CHANNEL_LABEL
                or self._frame_channel is not None
            ):
                channel.close()
                return
            self._frame_channel = channel

            @channel.on("message")
            def on_message(message: object) -> None:
                self._receive_selected_frame(message)

        @self.peer.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if self.peer.connectionState in {"failed", "closed"} and not self._closed:
                await self.websocket.close(code=1011, message=b"media connection ended")

        self._spawn(self._pump_audio())
        self._spawn(self._pump_frames())
        self._spawn(self._pump_output())

    def _spawn(self, coroutine: Awaitable[None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled() or self._closed:
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None and not self._fatal_started:
            self._fatal_started = True
            asyncio.create_task(self._fatal_runtime_error())

    async def _fatal_runtime_error(self) -> None:
        if self._closed:
            return
        try:
            await self.send_signal(
                {
                    "type": "error",
                    "code": "runtime_unavailable",
                    "message": "The local realtime runtime became unavailable.",
                    "retryable": True,
                }
            )
            await self.websocket.close(code=1011, message=b"runtime unavailable")
        except (ConnectionError, RuntimeError):
            return

    async def send_signal(self, signal: Mapping[str, object]) -> None:
        if self.websocket.closed:
            return
        encoded = serialize_server_signal(signal)
        async with self._send_lock:
            if not self.websocket.closed:
                await self.websocket.send_str(encoded)

    async def _consume_microphone(self, track: Any) -> None:
        consumer = MicrophoneConsumer(track, self.audio_input)
        try:
            await consumer.run()
        except MediaStreamError:
            return

    async def _pump_audio(self) -> None:
        while True:
            block = await self.audio_input.get()
            self.gateway.counters.audio_blocks_received += 1
            latency_turn = f"audio:{block.sequence}"
            self.gateway.latency.mark_audio_captured(
                latency_turn,
                block.sequence,
                at_ns=int(block.captured_at * 1_000_000_000),
            )
            self.gateway.latency.observe_queue(
                depth=self.audio_input.pending + 1,
                oldest_age_ms=max(
                    0.0, (time.monotonic() - block.received_at) * 1_000.0
                ),
            )
            accepted = await self.runtime.accept_audio(block)
            if not isinstance(accepted, bool):
                raise TypeError("Native runtime accept_audio must return bool")
            if accepted:
                self.gateway.counters.audio_blocks_accepted += 1
                self.gateway.latency.mark_audio_ingress_ready(
                    latency_turn,
                    block.sequence,
                    at_ns=time.monotonic_ns(),
                )

    def _receive_selected_frame(self, message: object) -> None:
        if self._closed:
            return
        if not isinstance(message, (bytes, bytearray, memoryview)):
            self.gateway.counters.selected_frames_rejected += 1
            if self._frame_channel is not None:
                self._frame_channel.close()
            return
        if len(message) > MAX_FRAME_PACKET_BYTES:
            self.gateway.counters.selected_frames_rejected += 1
            if self._frame_channel is not None:
                self._frame_channel.close()
            return
        try:
            frame = parse_selected_frame_packet(message)
            # The browser intentionally uses an unordered, unreliable channel
            # so a late older frame is normal network behaviour.  Drop it
            # without ACK rather than disabling the healthy channel.
            if frame.sequence <= self.guards.frame.value:
                self.gateway.counters.selected_frames_dropped += 1
                return
            self.guards.accept_frame_sequence(frame.sequence)
        except RealtimeProtocolError:
            self.gateway.counters.selected_frames_rejected += 1
            if self._frame_channel is not None:
                self._frame_channel.close()
            return
        self.gateway.counters.selected_frames_received += 1
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                self.gateway.counters.selected_frames_dropped += 1
        self.frame_queue.put_nowait(frame)

    async def _pump_frames(self) -> None:
        while True:
            frame = await self.frame_queue.get()
            accepted = await self.runtime.accept_frame(frame)
            if not isinstance(accepted, bool):
                raise TypeError("Native runtime accept_frame must return bool")
            if not accepted:
                continue
            self.guards.acknowledge_frame_sequence(frame.sequence)
            self.gateway.counters.selected_frames_accepted += 1
            await self.send_signal(
                {
                    "type": "frameAccepted",
                    "connectionId": self.connection_id,
                    "frameId": frame.frame_id,
                    "sequence": frame.sequence,
                }
            )

    async def _pump_output(self) -> None:
        while True:
            output = await self.runtime.next_output()
            generation = getattr(output, "generation", None)
            sequence = getattr(output, "sequence", None)
            produced_at = getattr(output, "produced_at", None)
            samples = getattr(output, "samples", None)
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                or isinstance(produced_at, bool)
                or not isinstance(produced_at, (int, float))
                or not math.isfinite(float(produced_at))
                or not isinstance(samples, np.ndarray)
            ):
                raise TypeError("Native runtime returned an invalid audio output")
            if sequence <= self._last_output_sequence:
                self.gateway.counters.output_blocks_rejected += 1
                continue
            self._last_output_sequence = sequence
            output_age_ms = (time.monotonic() - float(produced_at)) * 1_000.0
            if (
                output_age_ms < -1.0
                or output_age_ms > self.gateway.latency.gates.queue_age_p95_ms
            ):
                self.gateway.counters.output_blocks_rejected += 1
                continue
            flags = _runtime_flags(self.runtime)
            audible_untrained_diagnostic = (
                flags["model_loaded"]
                and flags["random_init"]
                and flags["untrained_audio_enabled"]
                and flags["can_emit_audio"]
            )
            if not (flags["conversation_ready"] or audible_untrained_diagnostic):
                self.gateway.counters.output_blocks_rejected += 1
                continue
            if self.audio_output.enqueue(samples, generation=generation):
                self.gateway.counters.output_blocks_enqueued += 1
                if bool(np.any(np.abs(samples) > 1e-6)):
                    self._mark_response_output_started()
            else:
                self.gateway.counters.output_blocks_rejected += 1

    def _remember_turn(self, turn_id: str) -> None:
        if turn_id in self._seen_turn_ids:
            raise RealtimeProtocolError("Realtime turn telemetry was replayed.")
        while len(self._seen_turn_order) >= MAX_SEEN_TURN_IDS:
            expired = self._seen_turn_order.popleft()
            self._seen_turn_ids.discard(expired)
        self._seen_turn_order.append(turn_id)
        self._seen_turn_ids.add(turn_id)

    def _mark_response_output_started(self) -> None:
        if self._active_response_turn is not None:
            return
        for turn_id, output_started in self._pending_turns.items():
            if not output_started:
                self._pending_turns[turn_id] = True
                self._active_response_turn = turn_id
                return

    async def handle_signal(self, signal: Mapping[str, object]) -> bool:
        """Handle an authenticated signal; return false for an orderly close."""

        if signal.get("connectionId") != self.connection_id:
            raise RealtimeProtocolError("Realtime connection identity does not match.")
        received_ns = time.monotonic_ns()
        signal_type = signal["type"]
        if signal_type == "ping":
            await self.send_signal({"type": "pong", "connectionId": self.connection_id})
            return True
        if signal_type == "offer":
            revision = self.guards.accept_revision(signal["revision"])
            await self.peer.setRemoteDescription(
                RTCSessionDescription(sdp=str(signal["sdp"]), type="offer")
            )
            answer = await self.peer.createAnswer()
            await self.peer.setLocalDescription(answer)
            local = self.peer.localDescription
            if local is None or not local.sdp:
                raise RealtimeProtocolError("Realtime answer could not be created.")
            await self.send_signal(
                {
                    "type": "answer",
                    "connectionId": self.connection_id,
                    "requestId": signal["requestId"],
                    "revision": revision,
                    "sdp": local.sdp,
                }
            )
            return True
        if signal_type == "ice":
            self.guards.require_current_revision(signal["revision"])
            candidate_value = signal["candidate"]
            if not isinstance(candidate_value, Mapping):
                raise RealtimeProtocolError("Realtime ICE candidate is invalid.")
            await self.peer.addIceCandidate(_candidate_from_wire(candidate_value))
            return True
        if signal_type == "interrupt":
            if len(self._pending_interrupts) >= MAX_PENDING_TELEMETRY:
                raise RealtimeProtocolError(
                    "Too many realtime interruption acknowledgements are pending."
                )
            generation = self.guards.accept_playback_generation(
                signal["playbackGeneration"]
            )
            latency_turn = f"interrupt:{generation}"
            self.gateway.latency.mark_interrupt_requested(
                latency_turn,
                generation,
                at_ns=received_ns,
            )
            self._pending_interrupts[generation] = latency_turn
            while self.audio_output.generation < generation:
                self.audio_output.interrupt()
            if self.audio_output.generation != generation:
                raise RealtimeProtocolError(
                    "Realtime playback generation is inconsistent."
                )
            runtime_status = self.runtime.status()
            runtime_current = (
                runtime_status.get("generation")
                if isinstance(runtime_status, Mapping)
                else None
            )
            if (
                isinstance(runtime_current, bool)
                or not isinstance(runtime_current, int)
                or runtime_current < 0
                or runtime_current > generation
            ):
                raise RealtimeProtocolError(
                    "Native runtime playback generation is inconsistent."
                )
            for next_generation in range(runtime_current + 1, generation + 1):
                runtime_generation = await self.runtime.interrupt(next_generation)
                if runtime_generation != next_generation:
                    raise RealtimeProtocolError(
                        "Native runtime did not fence interrupted audio."
                    )
            self.gateway.counters.interruptions += 1
            return True
        if signal_type == "playbackSilent":
            generation = signal["playbackGeneration"]
            if (
                not self._pending_interrupts
                or next(iter(self._pending_interrupts)) != generation
            ):
                raise RealtimeProtocolError(
                    "Realtime silence acknowledgement is stale or unknown."
                )
            self.guards.acknowledge_playback_generation(generation)
            latency_turn = self._pending_interrupts.pop(generation)
            self.gateway.latency.mark_playback_silent(
                latency_turn,
                generation,
                at_ns=received_ns,
            )
            return True
        if signal_type == "turnEnded":
            if len(self._pending_turns) >= MAX_PENDING_TELEMETRY:
                raise RealtimeProtocolError(
                    "Too many realtime response acknowledgements are pending."
                )
            turn_id = str(signal["turnId"])
            self._remember_turn(turn_id)
            self.gateway.latency.mark_turn_ended(turn_id, at_ns=received_ns)
            self._pending_turns[turn_id] = False
            end_turn = getattr(self.runtime, "end_turn", None)
            if callable(end_turn):
                released = await end_turn(turn_id)
                if not isinstance(released, bool):
                    raise TypeError("Native runtime end_turn must return bool")
            return True
        if signal_type == "firstResponseAudio":
            turn_id = str(signal["turnId"])
            if (
                self._active_response_turn != turn_id
                or self._pending_turns.get(turn_id) is not True
            ):
                raise RealtimeProtocolError(
                    "Realtime response acknowledgement is stale or uncorrelated."
                )
            self.gateway.latency.mark_first_response_audio(turn_id, at_ns=received_ns)
            del self._pending_turns[turn_id]
            self._active_response_turn = None
            return True
        if signal_type == "close":
            return False
        if signal_type in {"join", "answer"}:
            raise RealtimeProtocolError("Realtime signal is not valid in this state.")
        raise RealtimeProtocolError("Realtime signal type is not supported.")

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_resources())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            # aiohttp may cancel a WebSocket request task as the close frame is
            # exchanged.  Resource release continues in its own task and is
            # completed before propagating cancellation to the caller.
            await self._close_task
            raise

    async def _close_resources(self) -> None:
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self.gateway.counters.closed_input_drops += self.audio_input.dropped
        self.gateway.counters.closed_output_drops += self.audio_output.dropped
        self.audio_input.clear()
        self.audio_output.interrupt()
        self.audio_output.stop()
        # Release model-side state before waiting on aiortc teardown.  A peer
        # close handshake must never retain a native session or its buffers.
        try:
            await asyncio.wait_for(self.runtime.close(), timeout=CLOSE_TIMEOUT_SECONDS)
        finally:
            await asyncio.wait_for(self.peer.close(), timeout=CLOSE_TIMEOUT_SECONDS)


class RealtimeGateway:
    """Application state and route handlers for the local gateway."""

    def __init__(
        self,
        *,
        config: GatewayConfig | None = None,
        runtime_factory: RuntimeFactory | None = None,
        peer_factory: PeerFactory = RTCPeerConnection,
        latency: RealtimeLatencyTracker | None = None,
        checkpoint_configured: bool = False,
        checkpoint_preloaded: bool = False,
        untrained_audio_enabled: bool = False,
        untrained_parameters: int = 0,
    ) -> None:
        self.config = config or GatewayConfig()
        self.runtime_factory = runtime_factory or _default_runtime_factory
        self.peer_factory = peer_factory
        self.latency = latency or RealtimeLatencyTracker()
        if not isinstance(checkpoint_configured, bool):
            raise TypeError("checkpoint_configured must be bool")
        if not isinstance(checkpoint_preloaded, bool):
            raise TypeError("checkpoint_preloaded must be bool")
        if checkpoint_preloaded and not checkpoint_configured:
            raise ValueError("A preloaded checkpoint must be configured")
        if not isinstance(untrained_audio_enabled, bool):
            raise TypeError("untrained_audio_enabled must be bool")
        if (
            isinstance(untrained_parameters, bool)
            or not isinstance(untrained_parameters, int)
            or untrained_parameters < 0
        ):
            raise ValueError("untrained_parameters must be non-negative")
        if untrained_audio_enabled and checkpoint_configured:
            raise ValueError(
                "checkpoint and untrained audio modes are mutually exclusive"
            )
        if untrained_audio_enabled != (untrained_parameters > 0):
            raise ValueError(
                "untrained audio mode requires its exact positive parameter count"
            )
        self.checkpoint_configured = checkpoint_configured
        self.checkpoint_preloaded = checkpoint_preloaded
        self.untrained_audio_enabled = untrained_audio_enabled
        self.untrained_parameters = untrained_parameters
        self.tokens = AdmissionTokenStore()
        self.sessions: dict[str, RealtimeGatewaySession] = {}
        self._joining_count = 0
        self.counters = GatewayCounters()

    async def _make_runtime(
        self, session_id: str, connection_id: str
    ) -> NativeSessionRuntime:
        value = self.runtime_factory(session_id, connection_id)
        if inspect.isawaitable(value):
            value = await value
        required = (
            "status",
            "accept_audio",
            "accept_frame",
            "next_output",
            "interrupt",
            "close",
        )
        if not all(callable(getattr(value, name, None)) for name in required):
            raise TypeError("Native runtime does not implement the realtime contract")
        return value

    async def status(self, request: web.Request) -> web.Response:
        try:
            _validate_request_origin(request, self.config, required=False)
        except RealtimeProtocolError as error:
            return _json_error(str(error), status=403)

        self.tokens.purge_expired()
        runtime_flags = [
            _runtime_flags(item.runtime) for item in self.sessions.values()
        ]
        model_connected = any(item["model_connected"] for item in runtime_flags)
        checkpoint_loaded = self.checkpoint_preloaded or any(
            item["checkpoint_loaded"] for item in runtime_flags
        )
        voice_trained = any(item["voice_trained"] for item in runtime_flags)
        vision_trained = any(item["vision_trained"] for item in runtime_flags)
        runtime_conversation_ready = any(
            item["conversation_ready"] for item in runtime_flags
        )
        transport_connected = any(
            item.transport_connected for item in self.sessions.values()
        )
        self.latency.set_stages(
            transport_connected=transport_connected,
            model_connected=model_connected,
            voice_trained=voice_trained,
        )
        latency_status = self.latency.status()
        input_drops = self.counters.closed_input_drops + sum(
            item.audio_input.dropped for item in self.sessions.values()
        )
        output_drops = self.counters.closed_output_drops + sum(
            item.audio_output.dropped for item in self.sessions.values()
        )
        response = web.json_response(
            {
                "service": "mlm-realtime",
                "protocolVersion": REALTIME_PROTOCOL_VERSION,
                "transportReady": True,
                "webrtcReady": True,
                "checkpointConfigured": self.checkpoint_configured,
                "checkpointLoaded": checkpoint_loaded,
                "modelConnected": model_connected,
                "randomInit": self.untrained_audio_enabled,
                "untrainedAudioAvailable": self.untrained_audio_enabled,
                "untrainedAudioEnabled": self.untrained_audio_enabled,
                "untrainedParameters": self.untrained_parameters,
                "voiceTrained": voice_trained,
                "visionTrained": vision_trained,
                "conversationReady": (
                    runtime_conversation_ready and latency_status["ready"] is True
                ),
                "activeConnections": len(self.sessions),
                "connectedPeers": sum(
                    int(item.transport_connected) for item in self.sessions.values()
                ),
                "pendingAdmissions": self.tokens.pending_count,
                "media": {
                    "audioBlocksReceived": self.counters.audio_blocks_received,
                    "audioBlocksAccepted": self.counters.audio_blocks_accepted,
                    "audioQueueDrops": input_drops,
                    "selectedFramesReceived": self.counters.selected_frames_received,
                    "selectedFramesAccepted": self.counters.selected_frames_accepted,
                    "selectedFramesRejected": self.counters.selected_frames_rejected,
                    "selectedFramesDropped": self.counters.selected_frames_dropped,
                    "outputBlocksEnqueued": self.counters.output_blocks_enqueued,
                    "outputBlocksRejected": self.counters.output_blocks_rejected,
                    "outputQueueDrops": output_drops,
                    "interruptions": self.counters.interruptions,
                },
                "latency": latency_status,
            }
        )
        return _no_store(response)

    async def admission(self, request: web.Request) -> web.Response:
        try:
            _validate_request_origin(request, self.config, required=True)
        except RealtimeProtocolError as error:
            return _json_error(str(error), status=403)
        try:
            if (
                request.content_length is not None
                and request.content_length > MAX_ADMISSION_BYTES
            ):
                raise RealtimeProtocolError("Realtime admission body is too large.")
            body = _strict_json_object(await request.read())
            if set(body) != {"sessionId"}:
                raise RealtimeProtocolError("Realtime admission fields are invalid.")
            session_id = body["sessionId"]
            if not isinstance(session_id, str):
                raise RealtimeProtocolError("Realtime session ID is invalid.")
            self.tokens.purge_expired()
            if self.tokens.pending_count >= MAX_PENDING_ADMISSIONS:
                return _json_error(
                    "Too many realtime admissions are pending.", status=429
                )
            connection_id = f"rtc-{uuid.uuid4().hex}"
            ticket = self.tokens.issue(session_id, connection_id)
        except RealtimeProtocolError as error:
            return _json_error(str(error), status=400)
        self.counters.admissions_issued += 1
        response = web.json_response(
            {
                "protocolVersion": REALTIME_PROTOCOL_VERSION,
                "sessionId": ticket.session_id,
                "connectionId": ticket.connection_id,
                "signallingUrl": self.config.signalling_url,
                "accessToken": ticket.access_token,
                "expiresAtUnixMs": ticket.expires_at_unix_ms,
                "heartbeatIntervalMs": self.config.heartbeat_interval_ms,
                "iceServers": [],
                "capabilities": {
                    "duplexAudio": True,
                    "selectedFrameDataChannel": True,
                },
            }
        )
        return _no_store(response)

    async def websocket(self, request: web.Request) -> web.StreamResponse:
        try:
            _validate_request_origin(request, self.config, required=True)
        except RealtimeProtocolError as error:
            return _json_error(str(error), status=403)

        websocket = web.WebSocketResponse(
            autoping=True,
            autoclose=True,
            heartbeat=None,
            max_msg_size=MAX_SIGNAL_BYTES,
            compress=False,
        )
        await websocket.prepare(request)
        session: RealtimeGatewaySession | None = None
        runtime: NativeSessionRuntime | None = None
        try:
            try:
                first = await asyncio.wait_for(
                    websocket.receive(), timeout=self.config.join_timeout_seconds
                )
            except TimeoutError:
                await self._send_socket_error(
                    websocket,
                    code="join_timeout",
                    message="Realtime join timed out.",
                    retryable=True,
                )
                await websocket.close(code=1008, message=b"join timeout")
                return websocket
            if first.type != WSMsgType.TEXT:
                raise RealtimeProtocolError("Realtime join must be a text signal.")
            join = parse_client_signal(first.data)
            if join["type"] != "join":
                raise RealtimeProtocolError("Realtime join must be the first signal.")
            session_id = str(join["sessionId"])
            connection_id = str(join["connectionId"])
            self.tokens.consume(str(join["accessToken"]), session_id, connection_id)
            if len(self.sessions) + self._joining_count >= MAX_ACTIVE_CONNECTIONS:
                raise RealtimeProtocolError(
                    "The local realtime connection limit was reached."
                )
            self._joining_count += 1
            try:
                runtime = await self._make_runtime(session_id, connection_id)
            finally:
                self._joining_count -= 1
            if self.counters.joins_accepted > 0 and not self.sessions:
                self.latency = self.latency.empty_copy()
            session = RealtimeGatewaySession(
                gateway=self,
                websocket=websocket,
                runtime=runtime,
                session_id=session_id,
                connection_id=connection_id,
            )
            if connection_id in self.sessions:
                raise RealtimeProtocolError(
                    "Realtime connection identity is already active."
                )
            self.sessions[connection_id] = session
            self.counters.joins_accepted += 1
            session.start()
            await session.send_signal(
                {
                    "type": "joined",
                    "connectionId": connection_id,
                    "heartbeatIntervalMs": self.config.heartbeat_interval_ms,
                }
            )

            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    signal = parse_client_signal(message.data)
                    if not await session.handle_signal(signal):
                        await websocket.close(code=1000, message=b"client closed")
                        break
                elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED}:
                    break
                elif message.type == WSMsgType.ERROR:
                    break
                else:
                    raise RealtimeProtocolError(
                        "Realtime signalling accepts text only."
                    )
        except RealtimeProtocolError as error:
            await self._send_socket_error(
                websocket,
                code="protocol_error",
                message=str(error),
                retryable=False,
            )
            await websocket.close(code=1002, message=b"protocol error")
        except Exception:
            await self._send_socket_error(
                websocket,
                code="gateway_error",
                message="The local realtime gateway could not complete the request.",
                retryable=True,
            )
            await websocket.close(code=1011, message=b"gateway error")
        finally:
            if session is not None:
                self.sessions.pop(session.connection_id, None)
                try:
                    await session.close()
                except (asyncio.CancelledError, Exception):
                    # The request boundary must not hand runtime exception text
                    # to aiohttp's access/error logger.  Session cleanup itself
                    # is cancellation-shielded and has already been attempted.
                    pass
            elif runtime is not None:
                try:
                    await runtime.close()
                except (asyncio.CancelledError, Exception):
                    pass
        return websocket

    @staticmethod
    async def _send_socket_error(
        websocket: web.WebSocketResponse,
        *,
        code: str,
        message: str,
        retryable: bool,
    ) -> None:
        if websocket.closed:
            return
        try:
            await websocket.send_str(
                serialize_server_signal(
                    {
                        "type": "error",
                        "code": code,
                        "message": message,
                        "retryable": retryable,
                    }
                )
            )
        except (ConnectionError, RuntimeError):
            return

    async def shutdown(self) -> None:
        sessions = tuple(self.sessions.values())
        self.sessions.clear()
        for session in sessions:
            await session.websocket.close(code=1001, message=b"gateway shutdown")
        if sessions:
            await asyncio.gather(
                *(session.close() for session in sessions), return_exceptions=True
            )


GATEWAY_KEY = web.AppKey("mlm_realtime_gateway", RealtimeGateway)


def create_app(
    *,
    config: GatewayConfig | None = None,
    runtime_factory: RuntimeFactory | None = None,
    peer_factory: PeerFactory = RTCPeerConnection,
    latency: RealtimeLatencyTracker | None = None,
    checkpoint_configured: bool = False,
    checkpoint_preloaded: bool = False,
    untrained_audio_enabled: bool = False,
    untrained_parameters: int = 0,
) -> web.Application:
    """Build the aiohttp application without opening a network listener."""

    gateway = RealtimeGateway(
        config=config,
        runtime_factory=runtime_factory,
        peer_factory=peer_factory,
        latency=latency,
        checkpoint_configured=checkpoint_configured,
        checkpoint_preloaded=checkpoint_preloaded,
        untrained_audio_enabled=untrained_audio_enabled,
        untrained_parameters=untrained_parameters,
    )
    app = web.Application(client_max_size=MAX_ADMISSION_BYTES)
    app[GATEWAY_KEY] = gateway
    app.router.add_get("/status", gateway.status)
    app.router.add_post("/admission", gateway.admission)
    app.router.add_get("/realtime", gateway.websocket)

    async def on_shutdown(_app: web.Application) -> None:
        await gateway.shutdown()

    app.on_shutdown.append(on_shutdown)
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MLM's loopback-only native WebRTC gateway"
    )
    parser.add_argument("--host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Local scratch-native checkpoint manifest to serve",
    )
    model_group.add_argument(
        "--untrained-config",
        type=Path,
        help=(
            "Local scratch architecture JSON to random-initialize for an explicit "
            "meaningless-audio diagnostic"
        ),
    )
    parser.add_argument(
        "--untrained-seed",
        type=int,
        default=0,
        help="Non-negative random seed used only with --untrained-config",
    )
    parser.add_argument(
        "--allow-origin",
        action="append",
        dest="allowed_origins",
        help="Allowed local browser origin; repeat to allow more than one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        config = GatewayConfig(
            bind_host=args.host,
            port=args.port,
            allowed_origins=tuple(args.allowed_origins or DEFAULT_ALLOWED_ORIGINS),
        )
    except (RealtimeProtocolError, ValueError) as error:
        raise SystemExit(str(error)) from error
    try:
        if args.untrained_seed < 0:
            raise ValueError("--untrained-seed must be non-negative")
        if args.untrained_config is not None:
            runtime_factory = random_runtime_factory(
                args.untrained_config, seed=args.untrained_seed
            )
        elif args.checkpoint is not None:
            runtime_factory = checkpoint_runtime_factory(args.checkpoint)
        else:
            if args.untrained_seed != 0:
                raise ValueError("--untrained-seed requires --untrained-config")
            runtime_factory = _default_runtime_factory
    except (ImportError, OSError, ValueError) as error:
        raise SystemExit(
            f"Native checkpoint could not be configured: {error}"
        ) from error
    web.run_app(
        create_app(
            config=config,
            runtime_factory=runtime_factory,
            checkpoint_configured=args.checkpoint is not None,
            checkpoint_preloaded=args.checkpoint is not None,
            untrained_audio_enabled=args.untrained_config is not None,
            untrained_parameters=(
                getattr(runtime_factory, "parameter_count", 0)
                if args.untrained_config is not None
                else 0
            ),
        ),
        host=config.bind_host,
        port=config.port,
        access_log=None,
    )


if __name__ == "__main__":
    main()
