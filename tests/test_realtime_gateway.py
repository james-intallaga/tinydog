import asyncio
from dataclasses import dataclass
from fractions import Fraction
import json
import struct
import sys
from types import ModuleType

from aiohttp.test_utils import TestClient, TestServer
from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame
import numpy as np
import pytest

from mini_arithmetic.realtime_gateway import (
    GATEWAY_KEY,
    GatewayConfig,
    _candidate_from_wire,
    build_parser,
    checkpoint_runtime_factory,
    create_app,
)
from mini_arithmetic.realtime_latency import RealtimeLatencyTracker
from mini_arithmetic.realtime_protocol import RealtimeProtocolError


ORIGIN = {"Origin": "http://localhost:3000"}


@dataclass(frozen=True)
class FakeOutput:
    generation: int
    sequence: int
    samples: np.ndarray
    produced_at: float


class FakeRuntime:
    def __init__(
        self,
        *,
        accept_frames: bool = True,
        ready: bool = False,
        untrained_audio: bool = False,
    ) -> None:
        self.accept_frames = accept_frames
        self.ready = ready
        self.untrained_audio = untrained_audio
        self.audio = []
        self.frames = []
        self.interruptions = []
        self.outputs: asyncio.Queue[FakeOutput] = asyncio.Queue(maxsize=2)
        self.audio_event = asyncio.Event()
        self.frame_event = asyncio.Event()
        self.closed = False
        self.generation = 0

    def status(self):
        model_loaded = self.ready or self.untrained_audio
        return {
            "model_loaded": model_loaded,
            "model_connected": model_loaded,
            "checkpoint_loaded": self.ready,
            "voice_trained": self.ready,
            "vision_trained": self.ready,
            "conversation_ready": self.ready,
            "can_emit_audio": self.ready or self.untrained_audio,
            "random_init": self.untrained_audio,
            "untrained_audio_enabled": self.untrained_audio,
            "generation": self.generation,
        }

    async def accept_audio(self, block):
        self.audio.append(block)
        self.audio_event.set()
        return True

    async def accept_frame(self, frame):
        self.frames.append(frame)
        self.frame_event.set()
        return self.accept_frames

    async def next_output(self):
        return await self.outputs.get()

    async def interrupt(self, generation):
        self.interruptions.append(generation)
        self.generation = generation
        return generation

    async def close(self):
        self.closed = True


class SyntheticMicrophoneTrack(AudioStreamTrack):
    def __init__(self) -> None:
        super().__init__()
        self._pts = 0

    async def recv(self):
        await asyncio.sleep(0.02)
        pcm = np.zeros((1, 960), dtype=np.int16)
        frame = AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = 48_000
        frame.pts = self._pts
        frame.time_base = Fraction(1, 48_000)
        self._pts += 960
        return frame


def run(coroutine):
    return asyncio.run(coroutine)


async def open_client(*, runtime_factory=None, **app_options):
    app = create_app(runtime_factory=runtime_factory, **app_options)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def admit(client, session_id="session-1"):
    response = await client.post(
        "/admission", json={"sessionId": session_id}, headers=ORIGIN
    )
    assert response.status == 200
    return await response.json()


async def join(client, admission):
    websocket = await client.ws_connect("/realtime", headers=ORIGIN)
    await websocket.send_json(
        {
            "type": "join",
            "protocolVersion": 1,
            "accessToken": admission["accessToken"],
            "sessionId": admission["sessionId"],
            "connectionId": admission["connectionId"],
        }
    )
    joined = await websocket.receive_json(timeout=2)
    assert joined == {
        "type": "joined",
        "connectionId": admission["connectionId"],
        "heartbeatIntervalMs": 5_000,
    }
    return websocket


def selected_frame_packet(sequence=0, frame_id="frame-0"):
    jpeg = b"\xff\xd8complete-jpeg\xff\xd9"
    header = json.dumps(
        {
            "version": 1,
            "frameId": frame_id,
            "sequence": sequence,
            "timestampMs": 1234,
            "differencePercent": 24.5,
            "keyframe": sequence == 0,
            "contentType": "image/jpeg",
        },
        separators=(",", ":"),
    ).encode()
    return struct.pack(">I", len(header)) + header + jpeg


def test_gateway_configuration_is_loopback_only_and_normalizes_origins():
    config = GatewayConfig(allowed_origins=("http://LOCALHOST:3000/",))
    assert config.allowed_origins == ("http://localhost:3000",)
    assert config.signalling_url == "ws://127.0.0.1:8787/realtime"
    with pytest.raises(ValueError, match="loopback"):
        GatewayConfig(bind_host="0.0.0.0")
    with pytest.raises(RealtimeProtocolError, match="loopback"):
        GatewayConfig(allowed_origins=("https://example.com",))


def test_checkpoint_cli_builds_a_local_native_runtime_factory(tmp_path, monkeypatch):
    manifest = tmp_path / "native-manifest.json"
    manifest.write_text("{}")
    captured = []

    class StubFactory:
        def __call__(self, _session_id, _connection_id):
            return object()

    def load_factory(path):
        captured.append(path)
        return StubFactory()

    module = ModuleType("mini_arithmetic.native_runtime")
    module.load_native_runtime_factory = load_factory
    monkeypatch.setitem(sys.modules, "mini_arithmetic.native_runtime", module)

    args = build_parser().parse_args(["--checkpoint", str(manifest)])
    assert args.checkpoint == manifest
    factory = checkpoint_runtime_factory(args.checkpoint)
    assert isinstance(factory, StubFactory)
    assert captured == [manifest.resolve()]
    with pytest.raises(FileNotFoundError):
        checkpoint_runtime_factory(tmp_path / "missing.json")


def test_configured_checkpoint_status_still_fails_closed_before_it_is_loaded():
    async def scenario():
        client = await open_client(checkpoint_configured=True)
        try:
            response = await client.get("/status")
            status = await response.json()
            assert status["checkpointConfigured"] is True
            assert status["checkpointLoaded"] is False
            assert status["modelConnected"] is False
            assert status["conversationReady"] is False
        finally:
            await client.close()

    run(scenario())


def test_explicit_random_model_audio_is_forwarded_without_readiness_claims():
    async def scenario():
        runtime = FakeRuntime(untrained_audio=True)
        client = await open_client(
            runtime_factory=lambda _s, _c: runtime,
            untrained_audio_enabled=True,
            untrained_parameters=31_004_864,
        )
        try:
            status = await (await client.get("/status")).json()
            assert status["randomInit"] is True
            assert status["untrainedAudioAvailable"] is True
            assert status["untrainedAudioEnabled"] is True
            assert status["untrainedParameters"] == 31_004_864
            assert status["voiceTrained"] is False
            assert status["visionTrained"] is False
            assert status["conversationReady"] is False

            ticket = await admit(client)
            websocket = await join(client, ticket)
            await runtime.outputs.put(
                FakeOutput(
                    generation=0,
                    sequence=0,
                    samples=np.ones((480,), dtype=np.float32) * 0.1,
                    produced_at=asyncio.get_running_loop().time(),
                )
            )
            await asyncio.sleep(0.02)
            gateway = client.server.app[GATEWAY_KEY]
            assert gateway.counters.output_blocks_enqueued == 1
            assert gateway.counters.output_blocks_rejected == 0
            await websocket.close()
        finally:
            await client.close()

    run(scenario())


def test_preloaded_checkpoint_status_does_not_claim_an_active_model_session():
    async def scenario():
        client = await open_client(
            checkpoint_configured=True,
            checkpoint_preloaded=True,
        )
        try:
            response = await client.get("/status")
            status = await response.json()
            assert status["checkpointConfigured"] is True
            assert status["checkpointLoaded"] is True
            assert status["modelConnected"] is False
            assert status["conversationReady"] is False
        finally:
            await client.close()

    run(scenario())


def test_checkpoint_factory_validates_manifest_before_listener_start(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest fields"):
        checkpoint_runtime_factory(tmp_path)


def test_candidate_conversion_requires_bounded_media_identity():
    candidate = _candidate_from_wire(
        {
            "candidate": "candidate:1 1 udp 2122260223 127.0.0.1 54321 typ host",
            "sdpMid": "0",
        }
    )
    assert candidate is not None
    assert candidate.ip == "127.0.0.1"
    assert candidate.sdpMid == "0"
    assert _candidate_from_wire({"candidate": ""}) is None
    with pytest.raises(RealtimeProtocolError, match="media identity"):
        _candidate_from_wire(
            {"candidate": "candidate:1 1 udp 1 127.0.0.1 1234 typ host"}
        )
    with pytest.raises(RealtimeProtocolError, match="invalid"):
        _candidate_from_wire({"candidate": "candidate:not valid", "sdpMid": "0"})


def test_status_is_truthful_and_admission_is_origin_bound_and_no_store():
    async def scenario():
        client = await open_client()
        try:
            status_response = await client.get("/status")
            assert status_response.status == 200
            assert status_response.headers["Cache-Control"] == "no-store"
            status = await status_response.json()
            assert status["transportReady"] is True
            assert status["webrtcReady"] is True
            assert status["checkpointConfigured"] is False
            assert status["checkpointLoaded"] is False
            assert status["modelConnected"] is False
            assert status["voiceTrained"] is False
            assert status["visionTrained"] is False
            assert status["conversationReady"] is False
            assert status["activeConnections"] == 0

            rejected = await client.post(
                "/admission",
                json={"sessionId": "session-1"},
                headers={"Origin": "http://localhost:4000"},
            )
            assert rejected.status == 403
            invalid = await client.post(
                "/admission",
                data=b'{"sessionId":"a","sessionId":"b"}',
                headers={**ORIGIN, "Content-Type": "application/json"},
            )
            assert invalid.status == 400

            ticket = await admit(client)
            assert ticket["protocolVersion"] == 1
            assert ticket["sessionId"] == "session-1"
            assert ticket["connectionId"].startswith("rtc-")
            assert ticket["signallingUrl"] == "ws://127.0.0.1:8787/realtime"
            assert ticket["iceServers"] == []
            assert ticket["capabilities"] == {
                "duplexAudio": True,
                "selectedFrameDataChannel": True,
            }
            assert ticket["accessToken"] not in repr(
                ticket | {"accessToken": "<hidden>"}
            )
        finally:
            await client.close()

    run(scenario())


def test_default_native_runtime_accepts_transport_but_stays_fail_closed():
    async def scenario():
        client = await open_client()
        try:
            ticket = await admit(client, "untrained-session")
            websocket = await join(client, ticket)
            response = await client.get("/status")
            status = await response.json()
            assert status["activeConnections"] == 1
            assert status["modelConnected"] is False
            assert status["checkpointLoaded"] is False
            assert status["voiceTrained"] is False
            assert status["conversationReady"] is False
            await websocket.close()
        finally:
            await client.close()

    run(scenario())


def test_websocket_requires_one_use_join_and_supports_heartbeat_and_interrupt():
    async def scenario():
        runtimes = []

        def factory(_session_id, _connection_id):
            runtime = FakeRuntime()
            runtimes.append(runtime)
            return runtime

        client = await open_client(runtime_factory=factory)
        try:
            ticket = await admit(client)
            websocket = await join(client, ticket)
            await websocket.send_json(
                {
                    "type": "ping",
                    "clientTimeMs": 123,
                    "connectionId": ticket["connectionId"],
                }
            )
            assert await websocket.receive_json(timeout=2) == {
                "type": "pong",
                "connectionId": ticket["connectionId"],
            }
            await websocket.send_json(
                {
                    "type": "interrupt",
                    "connectionId": ticket["connectionId"],
                    "playbackGeneration": 3,
                }
            )
            for _ in range(20):
                if runtimes[0].interruptions:
                    break
                await asyncio.sleep(0.005)
            assert runtimes[0].interruptions == [1, 2, 3]
            await websocket.send_json(
                {
                    "type": "playbackSilent",
                    "connectionId": ticket["connectionId"],
                    "playbackGeneration": 3,
                }
            )

            replay = await client.ws_connect("/realtime", headers=ORIGIN)
            await replay.send_json(
                {
                    "type": "join",
                    "protocolVersion": 1,
                    "accessToken": ticket["accessToken"],
                    "sessionId": ticket["sessionId"],
                    "connectionId": ticket["connectionId"],
                }
            )
            failure = await replay.receive_json(timeout=2)
            assert failure["type"] == "error"
            assert failure["code"] == "protocol_error"
            assert ticket["accessToken"] not in json.dumps(failure)
            await replay.close()

            await websocket.send_json(
                {"type": "close", "connectionId": ticket["connectionId"]}
            )
            await websocket.receive(timeout=2)
            await websocket.close()
            for _ in range(100):
                if runtimes[0].closed:
                    break
                await asyncio.sleep(0.005)
            assert runtimes[0].closed is True
        finally:
            await client.close()

    run(scenario())


def test_turn_telemetry_pending_state_is_strictly_bounded():
    async def scenario():
        client = await open_client(runtime_factory=lambda _s, _c: FakeRuntime())
        try:
            ticket = await admit(client)
            websocket = await join(client, ticket)
            for index in range(9):
                await websocket.send_json(
                    {
                        "type": "turnEnded",
                        "connectionId": ticket["connectionId"],
                        "turnId": f"turn-{index}",
                    }
                )
            failure = await websocket.receive_json(timeout=2)
            assert failure["type"] == "error"
            assert failure["code"] == "protocol_error"
            assert "pending" in failure["message"]
            assert (
                client.server.app[GATEWAY_KEY].latency.status()["trace"][
                    "pending_spans"
                ]
                == 8
            )
        finally:
            await client.close()

    run(scenario())


def test_turn_telemetry_rejects_a_replayed_identifier():
    async def scenario():
        client = await open_client(runtime_factory=lambda _s, _c: FakeRuntime())
        try:
            ticket = await admit(client)
            websocket = await join(client, ticket)
            signal = {
                "type": "turnEnded",
                "connectionId": ticket["connectionId"],
                "turnId": "same-turn",
            }
            await websocket.send_json(signal)
            await websocket.send_json(signal)
            failure = await websocket.receive_json(timeout=2)
            assert failure["type"] == "error"
            assert failure["code"] == "protocol_error"
            assert "replayed" in failure["message"]
        finally:
            await client.close()

    run(scenario())


def test_offer_negotiates_duplex_audio_and_only_accepted_frames_are_acknowledged():
    async def scenario():
        runtimes = []

        def factory(_session_id, _connection_id):
            runtime = FakeRuntime(accept_frames=True, ready=True)
            runtimes.append(runtime)
            return runtime

        client = await open_client(
            runtime_factory=factory,
            latency=RealtimeLatencyTracker(minimum_samples=1),
        )
        peer = RTCPeerConnection()
        microphone = SyntheticMicrophoneTrack()
        channel = peer.createDataChannel(
            "mlm-selected-frames-v1", ordered=False, maxRetransmits=0
        )
        channel_open = asyncio.Event()
        channel.on("open", channel_open.set)
        peer.addTrack(microphone)
        try:
            ticket = await admit(client, "media-session")
            websocket = await join(client, ticket)
            offer = await peer.createOffer()
            await peer.setLocalDescription(offer)
            await websocket.send_json(
                {
                    "type": "offer",
                    "connectionId": ticket["connectionId"],
                    "requestId": "offer-1",
                    "revision": 1,
                    "sdp": peer.localDescription.sdp,
                    "iceRestart": False,
                }
            )
            answer = await websocket.receive_json(timeout=5)
            assert answer["type"] == "answer"
            assert answer["requestId"] == "offer-1"
            assert answer["revision"] == 1
            await peer.setRemoteDescription(
                RTCSessionDescription(sdp=answer["sdp"], type="answer")
            )
            await asyncio.wait_for(channel_open.wait(), timeout=5)
            await asyncio.wait_for(runtimes[0].audio_event.wait(), timeout=5)
            assert runtimes[0].audio
            assert runtimes[0].audio[0].samples.shape == (480,)

            channel.send(selected_frame_packet())
            accepted = await websocket.receive_json(timeout=5)
            assert accepted == {
                "type": "frameAccepted",
                "connectionId": ticket["connectionId"],
                "frameId": "frame-0",
                "sequence": 0,
            }
            assert len(runtimes[0].frames) == 1
            assert runtimes[0].frames[0].jpeg.startswith(b"\xff\xd8")

            runtimes[0].accept_frames = False
            runtimes[0].frame_event.clear()
            channel.send(selected_frame_packet(sequence=1, frame_id="frame-1"))
            await asyncio.wait_for(runtimes[0].frame_event.wait(), timeout=5)
            with pytest.raises(TimeoutError):
                await websocket.receive(timeout=0.05)

            await websocket.send_json(
                {
                    "type": "turnEnded",
                    "connectionId": ticket["connectionId"],
                    "turnId": "turn-1",
                }
            )
            await websocket.send_json(
                {
                    "type": "ping",
                    "clientTimeMs": 456,
                    "connectionId": ticket["connectionId"],
                }
            )
            assert (await websocket.receive_json(timeout=2))["type"] == "pong"
            await runtimes[0].outputs.put(
                FakeOutput(
                    generation=0,
                    sequence=0,
                    samples=np.ones((480,), dtype=np.float32) * 0.1,
                    produced_at=asyncio.get_running_loop().time(),
                )
            )
            gateway = client.server.app[GATEWAY_KEY]
            for _ in range(100):
                if gateway.counters.output_blocks_enqueued:
                    break
                await asyncio.sleep(0.005)
            assert gateway.counters.output_blocks_enqueued == 1

            await websocket.send_json(
                {
                    "type": "firstResponseAudio",
                    "connectionId": ticket["connectionId"],
                    "turnId": "turn-1",
                }
            )
            await websocket.send_json(
                {
                    "type": "interrupt",
                    "connectionId": ticket["connectionId"],
                    "playbackGeneration": 1,
                }
            )
            await websocket.send_json(
                {
                    "type": "playbackSilent",
                    "connectionId": ticket["connectionId"],
                    "playbackGeneration": 1,
                }
            )
            await websocket.send_json(
                {
                    "type": "ping",
                    "clientTimeMs": 789,
                    "connectionId": ticket["connectionId"],
                }
            )
            assert (await websocket.receive_json(timeout=2))["type"] == "pong"

            status_response = await client.get("/status")
            status = await status_response.json()
            assert status["latency"]["metrics"]["audio_ingress_ms"]["count"] >= 1
            assert status["latency"]["metrics"]["interrupt_to_silence_ms"]["count"] == 1
            assert status["latency"]["metrics"]["first_response_audio_ms"]["count"] == 1
            assert status["latency"]["trace"]["pending_spans"] == 0
            assert status["latency"]["ready"] is True
            assert status["conversationReady"] is True

            await websocket.send_json(
                {
                    "type": "firstResponseAudio",
                    "connectionId": ticket["connectionId"],
                    "turnId": "turn-1",
                }
            )
            replay = await websocket.receive_json(timeout=2)
            assert replay["type"] == "error"
            assert replay["code"] == "protocol_error"
        finally:
            microphone.stop()
            await peer.close()
            await client.close()

    run(scenario())


def test_untrained_runtime_output_is_dropped_instead_of_becoming_fake_sound():
    async def scenario():
        runtime = FakeRuntime(ready=False)
        client = await open_client(runtime_factory=lambda _s, _c: runtime)
        try:
            ticket = await admit(client)
            websocket = await join(client, ticket)
            await runtime.outputs.put(
                FakeOutput(
                    generation=0,
                    sequence=0,
                    samples=np.ones((480,), dtype=np.float32),
                    produced_at=asyncio.get_running_loop().time(),
                )
            )
            await asyncio.sleep(0.02)
            gateway = client.server.app[GATEWAY_KEY]
            assert gateway.counters.output_blocks_enqueued == 0
            assert gateway.counters.output_blocks_rejected == 1

            runtime.ready = True
            await runtime.outputs.put(
                FakeOutput(
                    generation=0,
                    sequence=1,
                    samples=np.zeros((480,), dtype=np.float32),
                    produced_at=asyncio.get_running_loop().time(),
                )
            )
            await asyncio.sleep(0.02)
            assert gateway.counters.output_blocks_enqueued == 1
            await runtime.outputs.put(
                FakeOutput(
                    generation=0,
                    sequence=1,
                    samples=np.zeros((480,), dtype=np.float32),
                    produced_at=asyncio.get_running_loop().time(),
                )
            )
            await asyncio.sleep(0.02)
            assert gateway.counters.output_blocks_rejected == 2
            await runtime.outputs.put(
                FakeOutput(
                    generation=0,
                    sequence=2,
                    samples=np.zeros((480,), dtype=np.float32),
                    produced_at=asyncio.get_running_loop().time() - 1.0,
                )
            )
            await asyncio.sleep(0.02)
            assert gateway.counters.output_blocks_enqueued == 1
            assert gateway.counters.output_blocks_rejected == 3
            await websocket.close()
        finally:
            await client.close()

    run(scenario())
