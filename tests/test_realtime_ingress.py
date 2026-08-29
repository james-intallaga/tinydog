import pytest

from mini_arithmetic.realtime_ingress import RealtimeIngress


def headers(**overrides: str) -> dict[str, str]:
    value = {
        "X-MLM-Session": "session-1",
        "X-MLM-Timestamp-Ms": "1234",
        "X-MLM-Difference": "17.25",
        "X-MLM-Keyframe": "false",
        "X-MLM-Sequence": "3",
    }
    value.update(overrides)
    return value


def test_accepts_bounded_frame_and_exposes_metadata_only() -> None:
    ingress = RealtimeIngress(frame_capacity=2)
    first = ingress.accept_frame(b"\xff\xd8one\xff\xd9", "image/jpeg", headers())
    ingress.accept_frame(
        b"\xff\xd8two\xff\xd9",
        "image/jpeg; quality=0.8",
        headers(**{"X-MLM-Difference": "10", "X-MLM-Keyframe": "true"}),
    )
    ingress.accept_frame(b"\xff\xd8three\xff\xd9", "image/jpeg", headers())

    status = ingress.status()
    assert first.difference_percent == 17.25
    assert status["frames_received"] == 3
    assert status["pending_frames"] == 2
    assert status["transport"] == "http-ingress"
    assert status["webrtc_ready"] is False
    assert status["model_connected"] is False
    assert "payload" not in status["last_frame"]
    assert status["last_frame"]["bytes"] == len(b"\xff\xd8three\xff\xd9")


def test_rejects_invalid_frame_contract() -> None:
    ingress = RealtimeIngress()
    with pytest.raises(ValueError, match="image/jpeg"):
        ingress.accept_frame(b"not jpeg", "image/png", headers())
    with pytest.raises(ValueError, match="complete JPEG"):
        ingress.accept_frame(b"not jpeg", "image/jpeg", headers())
    with pytest.raises(ValueError, match="between 0 and 100"):
        ingress.accept_frame(
            b"\xff\xd8x\xff\xd9",
            "image/jpeg",
            headers(**{"X-MLM-Difference": "101"}),
        )


def test_accepts_audio_chunks_and_drains_in_order() -> None:
    ingress = RealtimeIngress(audio_capacity=3)
    first = ingress.accept_audio(b"audio-1", "audio/webm;codecs=opus", headers())
    second = ingress.accept_audio(
        b"audio-2", "audio/ogg", headers(**{"X-MLM-Sequence": "4"})
    )

    assert first.sequence == 3
    assert second.sequence == 4
    assert [item.payload for item in ingress.drain_audio(limit=1)] == [b"audio-1"]
    assert [item.payload for item in ingress.drain_audio()] == [b"audio-2"]
    assert ingress.status()["pending_audio_chunks"] == 0


def test_rejects_missing_session_and_bad_audio_sequence() -> None:
    ingress = RealtimeIngress()
    with pytest.raises(ValueError, match="X-MLM-Session"):
        ingress.accept_audio(b"x", "audio/webm", headers(**{"X-MLM-Session": ""}))
    with pytest.raises(ValueError, match="X-MLM-Sequence"):
        ingress.accept_audio(b"x", "audio/webm", headers(**{"X-MLM-Sequence": "-1"}))
