from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from mini_arithmetic.realtime_protocol import (
    ADMISSION_TTL_MS,
    MAX_SELECTED_FRAME_BYTES,
    MAX_SELECTED_FRAME_HEADER_BYTES,
    MAX_SIGNAL_BYTES,
    AdmissionTokenStore,
    MonotonicSequenceGuard,
    RealtimeProtocolError,
    RealtimeSequenceGuards,
    is_loopback_host,
    normalize_origin,
    parse_client_signal,
    parse_selected_frame_packet,
    serialize_server_signal,
    validate_loopback_request,
)


class Clock:
    def __init__(self, now: int = 1_700_000_000_000) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def frame_header(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "version": 1,
        "frameId": "frame-1",
        "sequence": 7,
        "timestampMs": 1_234,
        "differencePercent": 17.25,
        "keyframe": False,
        "contentType": "image/jpeg",
    }
    value.update(overrides)
    return value


def frame_packet(
    header: dict[str, object] | bytes | None = None,
    jpeg: bytes = b"\xff\xd8pixels\xff\xd9",
) -> bytes:
    if header is None:
        header = frame_header()
    encoded = (
        header
        if isinstance(header, bytes)
        else json.dumps(header, separators=(",", ":")).encode()
    )
    return len(encoded).to_bytes(4, "big") + encoded + jpeg


def test_admission_ticket_is_exactly_30_seconds_and_hides_its_secret() -> None:
    clock = Clock()
    store = AdmissionTokenStore(clock_ms=clock)

    ticket = store.issue("session-1", "connection-1")

    assert ticket.expires_at_unix_ms == clock.now + ADMISSION_TTL_MS
    assert ticket.access_token not in repr(ticket)
    assert store.pending_count == 1
    claims = store.consume(ticket.access_token, "session-1", "connection-1")
    assert claims.session_id == "session-1"
    assert claims.connection_id == "connection-1"
    assert store.pending_count == 0


def test_admission_ticket_is_one_use() -> None:
    store = AdmissionTokenStore(clock_ms=Clock())
    ticket = store.issue("session-1", "connection-1")
    store.consume(ticket.access_token, "session-1", "connection-1")

    with pytest.raises(RealtimeProtocolError, match="admission was rejected"):
        store.consume(ticket.access_token, "session-1", "connection-1")


def test_wrong_identity_does_not_reveal_or_destroy_ticket() -> None:
    store = AdmissionTokenStore(clock_ms=Clock())
    ticket = store.issue("session-1", "connection-1")

    with pytest.raises(RealtimeProtocolError) as captured:
        store.consume(ticket.access_token, "session-1", "connection-2")
    assert ticket.access_token not in str(captured.value)
    assert store.consume(ticket.access_token, "session-1", "connection-1")


def test_expired_admission_is_rejected_at_the_boundary_and_purged() -> None:
    clock = Clock()
    store = AdmissionTokenStore(clock_ms=clock)
    ticket = store.issue("session-1", "connection-1")
    clock.now = ticket.expires_at_unix_ms

    with pytest.raises(RealtimeProtocolError, match="admission was rejected"):
        store.consume(ticket.access_token, "session-1", "connection-1")
    assert store.pending_count == 0


def test_admission_revoke_and_expiry_purge() -> None:
    clock = Clock()
    store = AdmissionTokenStore(clock_ms=clock)
    revoked = store.issue("session-1", "connection-1")
    expiring = store.issue("session-2", "connection-2")
    store.revoke(revoked.access_token)
    assert store.pending_count == 1
    clock.now = expiring.expires_at_unix_ms
    assert store.purge_expired() == 1
    assert store.pending_count == 0


def test_only_one_concurrent_consumer_can_use_a_ticket() -> None:
    store = AdmissionTokenStore(clock_ms=Clock())
    ticket = store.issue("session-1", "connection-1")

    def consume() -> bool:
        try:
            store.consume(ticket.access_token, "session-1", "connection-1")
            return True
        except RealtimeProtocolError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: consume(), range(16)))
    assert results.count(True) == 1


@pytest.mark.parametrize(
    "host",
    ["localhost", "LOCALHOST.", "127.0.0.1", "127.23.4.5", "::1", "[::1]"],
)
def test_recognizes_loopback_without_dns_resolution(host: str) -> None:
    assert is_loopback_host(host)


@pytest.mark.parametrize(
    "host", ["example.com", "localhost.example.com", "0.0.0.0", "192.168.1.1", "::"]
)
def test_rejects_non_loopback_hosts(host: str) -> None:
    assert not is_loopback_host(host)


def test_validates_peer_host_origin_and_explicit_allowlist() -> None:
    assert (
        validate_loopback_request(
            peer_host="127.0.0.1",
            host_header="[::1]:8766",
            origin="http://LOCALHOST.:3000/",
            allowed_origins={"http://localhost:3000"},
        )
        == "http://localhost:3000"
    )
    assert normalize_origin("https://[::1]:443") == "https://[::1]"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("peer_host", "192.168.1.2"),
        ("host_header", "example.com:8766"),
        ("origin", "https://example.com"),
        ("origin", "http://localhost.example.com:3000"),
        ("origin", "http://localhost:3000/path"),
        ("origin", "http://user@localhost:3000"),
        ("origin", "http://localhost:3000?token=x"),
    ],
)
def test_rejects_non_loopback_or_malformed_request_parts(
    field: str, value: str
) -> None:
    arguments = {
        "peer_host": "127.0.0.1",
        "host_header": "localhost:8766",
        "origin": "http://localhost:3000",
        "allowed_origins": {"http://localhost:3000"},
    }
    arguments[field] = value
    with pytest.raises(RealtimeProtocolError):
        validate_loopback_request(**arguments)


def test_rejects_a_loopback_origin_that_is_not_explicitly_allowed() -> None:
    with pytest.raises(RealtimeProtocolError, match="not allowed"):
        validate_loopback_request(
            peer_host="::1",
            host_header="localhost:8766",
            origin="http://localhost:3001",
            allowed_origins={"http://localhost:3000"},
        )


@pytest.mark.parametrize(
    ("message", "expected_type"),
    [
        (
            {
                "type": "join",
                "protocolVersion": 1,
                "accessToken": "a" * 43,
                "sessionId": "session-1",
                "connectionId": "connection-1",
            },
            "join",
        ),
        (
            {"type": "ping", "clientTimeMs": 123, "connectionId": "connection-1"},
            "ping",
        ),
        (
            {
                "type": "offer",
                "connectionId": "connection-1",
                "requestId": "offer-1",
                "revision": 1,
                "sdp": "v=0\r\n",
                "iceRestart": False,
            },
            "offer",
        ),
        (
            {
                "type": "answer",
                "connectionId": "connection-1",
                "requestId": "offer-1",
                "revision": 1,
                "sdp": "v=0\r\n",
            },
            "answer",
        ),
        (
            {
                "type": "ice",
                "connectionId": "connection-1",
                "revision": 1,
                "candidate": {
                    "candidate": "candidate:1 1 UDP 1 127.0.0.1 5000 typ host",
                    "sdpMid": "0",
                    "sdpMLineIndex": 0,
                    "usernameFragment": "ufrag",
                },
            },
            "ice",
        ),
        (
            {
                "type": "interrupt",
                "connectionId": "connection-1",
                "playbackGeneration": 1,
            },
            "interrupt",
        ),
        (
            {
                "type": "playbackSilent",
                "connectionId": "connection-1",
                "playbackGeneration": 1,
            },
            "playbackSilent",
        ),
        (
            {
                "type": "turnEnded",
                "connectionId": "connection-1",
                "turnId": "turn-1",
            },
            "turnEnded",
        ),
        (
            {
                "type": "firstResponseAudio",
                "connectionId": "connection-1",
                "turnId": "turn-1",
            },
            "firstResponseAudio",
        ),
        ({"type": "close", "connectionId": "connection-1"}, "close"),
    ],
)
def test_parses_every_browser_signal(
    message: dict[str, object], expected_type: str
) -> None:
    parsed = parse_client_signal(json.dumps(message))
    assert parsed["type"] == expected_type
    assert set(parsed) == set(message)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "[]",
        "{",
        '{"type":"close","type":"ping","connectionId":"c"}',
        '{"type":"ping","clientTimeMs":NaN,"connectionId":"c"}',
        '{"type":"close","connectionId":"c","extra":1}',
        '{"type":"close"}',
        '{"type":"unknown","connectionId":"c"}',
        '"\ud800"',
        b"\xff",
    ],
)
def test_rejects_malformed_or_ambiguous_json_signals(raw: str | bytes) -> None:
    with pytest.raises(RealtimeProtocolError):
        parse_client_signal(raw)


def test_signal_size_is_measured_in_utf8_bytes() -> None:
    with pytest.raises(RealtimeProtocolError, match="too large"):
        parse_client_signal('"' + ("\u00e9" * (MAX_SIGNAL_BYTES // 2)) + '"')


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocolVersion", True),
        ("protocolVersion", 2),
        ("sessionId", "unsafe id"),
        ("connectionId", ""),
        ("accessToken", "secret with spaces"),
    ],
)
def test_rejects_invalid_join_fields(field: str, value: object) -> None:
    message = {
        "type": "join",
        "protocolVersion": 1,
        "accessToken": "a" * 43,
        "sessionId": "session-1",
        "connectionId": "connection-1",
    }
    message[field] = value
    with pytest.raises(RealtimeProtocolError):
        parse_client_signal(json.dumps(message))


def test_rejects_boolean_sequence_and_extra_ice_fields() -> None:
    with pytest.raises(RealtimeProtocolError):
        parse_client_signal(
            json.dumps(
                {
                    "type": "interrupt",
                    "connectionId": "connection-1",
                    "playbackGeneration": True,
                }
            )
        )
    with pytest.raises(RealtimeProtocolError):
        parse_client_signal(
            json.dumps(
                {
                    "type": "ice",
                    "connectionId": "connection-1",
                    "revision": 1,
                    "candidate": {"candidate": "candidate", "password": "no"},
                }
            )
        )


@pytest.mark.parametrize(
    "message",
    [
        {
            "type": "joined",
            "connectionId": "connection-1",
            "heartbeatIntervalMs": 5_000,
        },
        {"type": "pong", "connectionId": "connection-1"},
        {
            "type": "answer",
            "connectionId": "connection-1",
            "requestId": "offer-1",
            "revision": 1,
            "sdp": "v=0\r\n",
        },
        {
            "type": "offer",
            "connectionId": "connection-1",
            "requestId": "offer-2",
            "revision": 2,
            "sdp": "v=0\r\n",
            "iceRestart": True,
        },
        {
            "type": "ice",
            "connectionId": "connection-1",
            "revision": 2,
            "candidate": {"candidate": "", "sdpMid": None, "sdpMLineIndex": None},
        },
        {"type": "restart", "connectionId": "connection-1"},
        {
            "type": "frameAccepted",
            "connectionId": "connection-1",
            "frameId": "frame-1",
            "sequence": 4,
        },
        {
            "type": "error",
            "code": "bad-signal",
            "message": "Rejected.",
            "retryable": False,
        },
    ],
)
def test_serializes_every_browser_server_signal(message: dict[str, object]) -> None:
    encoded = serialize_server_signal(message)
    decoded = json.loads(encoded)
    expected = json.loads(json.dumps(message))
    if message["type"] == "ice":
        expected["candidate"].pop("sdpMid")
        expected["candidate"].pop("sdpMLineIndex")
    assert decoded == expected
    assert ": " not in encoded


@pytest.mark.parametrize(
    "message",
    [
        {"type": "joined", "connectionId": "c", "heartbeatIntervalMs": 999},
        {
            "type": "frameAccepted",
            "connectionId": "c",
            "frameId": "bad id",
            "sequence": 1,
        },
        {"type": "error", "code": "bad code", "message": "Rejected."},
        {"type": "error", "code": "bad", "message": "Rejected.", "retryable": "yes"},
        {"type": "pong", "connectionId": "c", "secret": "no"},
    ],
)
def test_rejects_invalid_server_signals(message: dict[str, object]) -> None:
    with pytest.raises(RealtimeProtocolError):
        serialize_server_signal(message)


def test_generic_monotonic_guard_rejects_replays_gaps_are_allowed() -> None:
    guard = MonotonicSequenceGuard("sample", initial=-1, minimum=0)
    assert guard.advance(0) == 0
    assert guard.advance(9) == 9
    assert guard.require_current(9) == 9
    with pytest.raises(RealtimeProtocolError, match="monotonically"):
        guard.advance(9)
    with pytest.raises(RealtimeProtocolError, match="stale"):
        guard.require_current(8)
    with pytest.raises(RealtimeProtocolError):
        guard.advance(True)


def test_session_guards_reject_stale_negotiation_interrupts_and_frames() -> None:
    guards = RealtimeSequenceGuards()
    assert guards.accept_revision(1) == 1
    assert guards.require_current_revision(1) == 1
    assert guards.accept_revision(3) == 3
    assert guards.accept_playback_generation(1) == 1
    assert guards.acknowledge_playback_generation(1) == 1
    assert guards.accept_frame_sequence(0) == 0
    assert guards.accept_frame_sequence(5) == 5

    with pytest.raises(RealtimeProtocolError):
        guards.accept_revision(2)
    with pytest.raises(RealtimeProtocolError):
        guards.accept_playback_generation(1)
    with pytest.raises(RealtimeProtocolError):
        guards.acknowledge_playback_generation(1)
    with pytest.raises(RealtimeProtocolError):
        guards.accept_frame_sequence(4)


def test_playback_acknowledgements_must_be_known_and_monotonic() -> None:
    guards = RealtimeSequenceGuards()
    with pytest.raises(RealtimeProtocolError, match="unknown"):
        guards.acknowledge_playback_generation(1)
    guards.accept_playback_generation(2)
    assert guards.acknowledge_playback_generation(2) == 2
    guards.accept_playback_generation(4)
    with pytest.raises(RealtimeProtocolError, match="monotonically"):
        guards.acknowledge_playback_generation(2)


def test_frame_acknowledgements_are_known_and_monotonic() -> None:
    guards = RealtimeSequenceGuards()
    guards.accept_frame_sequence(4)
    assert guards.acknowledge_frame_sequence(4) == 4
    with pytest.raises(RealtimeProtocolError, match="monotonically"):
        guards.acknowledge_frame_sequence(4)
    guards.accept_frame_sequence(8)
    with pytest.raises(RealtimeProtocolError, match="unknown"):
        guards.acknowledge_frame_sequence(9)


def test_parses_selected_frame_packet_matching_browser_encoder() -> None:
    parsed = parse_selected_frame_packet(frame_packet())
    assert parsed.frame_id == "frame-1"
    assert parsed.sequence == 7
    assert parsed.timestamp_ms == 1_234
    assert parsed.difference_percent == 17.25
    assert parsed.keyframe is False
    assert parsed.content_type == "image/jpeg"
    assert parsed.jpeg == b"\xff\xd8pixels\xff\xd9"
    assert "pixels" not in repr(parsed)


def test_accepts_the_maximum_selected_frame_jpeg() -> None:
    jpeg = b"\xff\xd8" + (b"x" * (MAX_SELECTED_FRAME_BYTES - 4)) + b"\xff\xd9"
    assert len(parse_selected_frame_packet(frame_packet(jpeg=jpeg)).jpeg) == len(jpeg)


@pytest.mark.parametrize(
    "packet",
    [
        b"",
        b"\x00\x00\x00\x00" + b"\xff\xd8\xff\xd9",
        (MAX_SELECTED_FRAME_HEADER_BYTES + 1).to_bytes(4, "big")
        + b"{}\xff\xd8\xff\xd9",
        b"\x00\x00\x00\x10{}\xff\xd8\xff\xd9",
        frame_packet(header=b"{"),
        frame_packet(header=b'{"version":1,"version":1}'),
        frame_packet(jpeg=b"not-a-jpeg"),
        frame_packet(
            jpeg=b"\xff\xd8" + b"x" * (MAX_SELECTED_FRAME_BYTES - 3) + b"\xff\xd9"
        ),
    ],
)
def test_rejects_malformed_bounded_frame_packets(packet: bytes) -> None:
    with pytest.raises(RealtimeProtocolError):
        parse_selected_frame_packet(packet)


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": True},
        {"version": 2},
        {"frameId": "bad id"},
        {"sequence": -1},
        {"timestampMs": 1.5},
        {"differencePercent": float("nan")},
        {"differencePercent": 101},
        {"keyframe": 1},
        {"contentType": "image/png"},
        {"extra": "field"},
    ],
)
def test_rejects_invalid_selected_frame_headers(overrides: dict[str, object]) -> None:
    with pytest.raises(RealtimeProtocolError):
        parse_selected_frame_packet(frame_packet(frame_header(**overrides)))


def test_rejects_missing_selected_frame_header_field() -> None:
    header = frame_header()
    del header["frameId"]
    with pytest.raises(RealtimeProtocolError, match="missing"):
        parse_selected_frame_packet(frame_packet(header))
