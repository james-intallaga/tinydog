from __future__ import annotations

import pytest

from mini_arithmetic.realtime_latency import (
    EventKind,
    LatencyGates,
    LatencyMetric,
    RealtimeLatencyTracker,
    nearest_rank_percentile,
)


MS = 1_000_000


def populate(
    tracker: RealtimeLatencyTracker,
    *,
    audio_ms: list[int],
    interrupt_ms: list[int],
    response_ms: list[int],
    queue_depths: list[int] | None = None,
    queue_ages_ms: list[int] | None = None,
) -> None:
    for index, (audio, interrupt, response) in enumerate(
        zip(audio_ms, interrupt_ms, response_ms, strict=True)
    ):
        turn = f"turn-{index}"
        base = index * 2_000 * MS
        tracker.mark_audio_captured(turn, index, at_ns=base)
        tracker.mark_audio_ingress_ready(turn, index, at_ns=base + audio * MS)
        tracker.mark_turn_ended(turn, at_ns=base + 200 * MS)
        tracker.mark_first_response_audio(turn, at_ns=base + (200 + response) * MS)
        tracker.mark_interrupt_requested(turn, index, at_ns=base + 700 * MS)
        tracker.mark_playback_silent(turn, index, at_ns=base + (700 + interrupt) * MS)

    depths = queue_depths if queue_depths is not None else [1] * len(audio_ms)
    ages = queue_ages_ms if queue_ages_ms is not None else [20] * len(audio_ms)
    for index, (depth, age) in enumerate(zip(depths, ages, strict=True)):
        tracker.observe_queue(depth=depth, oldest_age_ms=age, at_ns=index * MS)


def declared_ready(tracker: RealtimeLatencyTracker) -> None:
    tracker.set_stages(
        transport_connected=True,
        model_connected=True,
        voice_trained=True,
    )


def test_nearest_rank_percentile_is_deterministic_and_rejects_bad_input() -> None:
    values = [50.0, 10.0, 40.0, 20.0]
    assert nearest_rank_percentile(values, 0.50) == 20.0
    assert nearest_rank_percentile(values, 0.95) == 50.0
    assert nearest_rank_percentile(values, 1.0) == 50.0
    with pytest.raises(ValueError, match="at least one"):
        nearest_rank_percentile([], 0.95)
    with pytest.raises(ValueError, match="finite"):
        nearest_rank_percentile([float("nan")], 0.95)


def test_empty_tracker_fails_closed_for_stages_samples_and_queue() -> None:
    status = RealtimeLatencyTracker(minimum_samples=3).status()

    assert status["ready"] is False
    assert status["stages"]["transport"]["ready"] is False
    assert status["stages"]["model"]["ready"] is False
    assert status["stages"]["voice_trained"]["ready"] is False
    assert status["metrics"][LatencyMetric.AUDIO_INGRESS.value]["p95_ms"] is None
    assert status["queue"]["pass"] is False


def test_empty_copy_preserves_configuration_without_carrying_old_spans() -> None:
    tracker = RealtimeLatencyTracker(minimum_samples=1, sample_capacity=7)
    tracker.mark_turn_ended("old", at_ns=1)
    clone = tracker.empty_copy()

    assert clone.gates == tracker.gates
    assert clone.minimum_samples == 1
    assert clone.sample_capacity == 7
    status = clone.status()
    assert status["trace"]["pending_spans"] == 0
    assert status["metrics"][LatencyMetric.FIRST_RESPONSE_AUDIO.value]["count"] == 0


def test_complete_trace_passes_inclusive_p95_gates() -> None:
    tracker = RealtimeLatencyTracker(minimum_samples=4)
    populate(
        tracker,
        audio_ms=[10, 20, 40, 50],
        interrupt_ms=[20, 40, 80, 100],
        response_ms=[100, 200, 250, 300],
        queue_depths=[1, 2, 1, 1],
        queue_ages_ms=[10, 20, 80, 100],
    )
    declared_ready(tracker)

    status = tracker.status()
    assert status["ready"] is True
    assert status["metrics"][LatencyMetric.AUDIO_INGRESS.value]["p50_ms"] == 20
    assert status["metrics"][LatencyMetric.AUDIO_INGRESS.value]["p95_ms"] == 50
    assert status["metrics"][LatencyMetric.INTERRUPT_TO_SILENCE.value]["p95_ms"] == 100
    assert status["metrics"][LatencyMetric.FIRST_RESPONSE_AUDIO.value]["p95_ms"] == 300
    assert status["queue"]["oldest_age_p95_ms"] == 100
    assert status["queue"]["net_growth"] == 0
    assert status["trace"]["pass"] is True


@pytest.mark.parametrize(
    ("audio", "interrupt", "response", "failed_stage"),
    [
        ([20, 20, 51], [20, 20, 20], [100, 100, 100], "transport"),
        ([20, 20, 20], [20, 20, 101], [100, 100, 100], "transport"),
        ([20, 20, 20], [20, 20, 20], [100, 100, 301], "model"),
    ],
)
def test_each_latency_gate_fails_at_one_millisecond_over_limit(
    audio: list[int],
    interrupt: list[int],
    response: list[int],
    failed_stage: str,
) -> None:
    tracker = RealtimeLatencyTracker(minimum_samples=3)
    populate(tracker, audio_ms=audio, interrupt_ms=interrupt, response_ms=response)
    declared_ready(tracker)

    status = tracker.status()
    assert status["ready"] is False
    assert status["stages"][failed_stage]["ready"] is False


@pytest.mark.parametrize(
    ("depths", "ages", "reason_fragment"),
    [
        ([1, 1, 1], [20, 20, 101], "age p95"),
        ([1, 5, 1], [20, 20, 20], "maximum depth"),
        ([1, 1, 2], [20, 20, 20], "net growth"),
    ],
)
def test_queue_age_depth_and_growth_are_hard_transport_gates(
    depths: list[int], ages: list[int], reason_fragment: str
) -> None:
    tracker = RealtimeLatencyTracker(minimum_samples=3)
    populate(
        tracker,
        audio_ms=[20, 20, 20],
        interrupt_ms=[20, 20, 20],
        response_ms=[100, 100, 100],
        queue_depths=depths,
        queue_ages_ms=ages,
    )
    declared_ready(tracker)

    status = tracker.status()
    assert status["ready"] is False
    assert status["queue"]["pass"] is False
    assert any(reason_fragment in reason for reason in status["queue"]["reasons"])
    assert status["stages"]["transport"]["ready"] is False


def test_readiness_stages_remain_separate() -> None:
    tracker = RealtimeLatencyTracker(minimum_samples=3)
    populate(
        tracker,
        audio_ms=[20, 20, 20],
        interrupt_ms=[20, 20, 20],
        response_ms=[100, 100, 100],
    )
    tracker.set_stages(
        transport_connected=True,
        model_connected=True,
        voice_trained=False,
    )

    status = tracker.status()
    assert status["stages"]["transport"]["ready"] is True
    assert status["stages"]["model"]["ready"] is True
    assert status["stages"]["voice_trained"]["ready"] is False
    assert status["ready"] is False


def test_incomplete_or_evicted_spans_fail_trace_integrity() -> None:
    tracker = RealtimeLatencyTracker(
        minimum_samples=1,
        pending_capacity=1,
    )
    tracker.mark_audio_captured("first", 0, at_ns=0)
    tracker.mark_audio_captured("second", 0, at_ns=MS)

    status = tracker.status()
    assert status["trace"]["pass"] is False
    assert status["trace"]["pending_spans"] == 1
    assert status["trace"]["dropped_pending_spans"] == 1
    assert status["ready"] is False


def test_samples_and_events_are_bounded_to_latest_values() -> None:
    tracker = RealtimeLatencyTracker(
        sample_capacity=3,
        event_capacity=4,
        minimum_samples=1,
    )
    for index, duration in enumerate([10, 20, 30, 40, 50]):
        turn = f"bounded-{index}"
        base = index * 100 * MS
        tracker.mark_audio_captured(turn, 0, at_ns=base)
        tracker.mark_audio_ingress_ready(turn, 0, at_ns=base + duration * MS)

    metric = tracker.status()["metrics"][LatencyMetric.AUDIO_INGRESS.value]
    assert metric["count"] == 3
    assert metric["p50_ms"] == 40
    assert metric["p95_ms"] == 50
    assert len(tracker.recent_events()) == 4


def test_turn_and_queue_timestamps_must_be_monotonic() -> None:
    tracker = RealtimeLatencyTracker(minimum_samples=1)
    tracker.mark_audio_captured("turn", 0, at_ns=10 * MS)
    with pytest.raises(ValueError, match="finish before"):
        tracker.mark_audio_ingress_ready("turn", 0, at_ns=9 * MS)

    tracker.observe_queue(depth=1, oldest_age_ms=5, at_ns=20 * MS)
    with pytest.raises(ValueError, match="non-decreasing"):
        tracker.observe_queue(depth=1, oldest_age_ms=5, at_ns=19 * MS)


def test_event_log_contains_correlated_turn_markers() -> None:
    tracker = RealtimeLatencyTracker(minimum_samples=1)
    tracker.mark_interrupt_requested("barge-in", 7, at_ns=100 * MS)
    assert tracker.mark_playback_silent("barge-in", 7, at_ns=180 * MS) == 80

    events = tracker.recent_events()
    assert [event.kind for event in events] == [
        EventKind.INTERRUPT_REQUESTED,
        EventKind.PLAYBACK_SILENT,
    ]
    assert all(event.turn_id == "barge-in" for event in events)
    assert all(event.correlation_id == "7" for event in events)


def test_custom_gates_are_validated_and_reported() -> None:
    gates = LatencyGates(queue_max_depth=2, queue_max_growth=1)
    status = RealtimeLatencyTracker(gates=gates, minimum_samples=1).status()
    assert status["gates"]["queue_max_depth"] == 2
    assert status["gates"]["queue_max_growth"] == 1
    with pytest.raises(ValueError, match="non-negative"):
        LatencyGates(audio_ingress_p95_ms=-1)
