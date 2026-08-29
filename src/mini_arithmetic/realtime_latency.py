"""Fail-closed latency and readiness accounting for realtime conversation.

This module measures an execution pipeline; it does not claim that a random or
untrained model can converse.  Callers must translate every timestamp into one
shared monotonic clock domain before recording it.  Browser/gateway clock
synchronisation belongs at the transport boundary, not in this pure tracker.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
import math
import threading
import time
from typing import Any


class LatencyMetric(str, Enum):
    """Durations that gate native realtime conversation readiness."""

    AUDIO_INGRESS = "audio_ingress_ms"
    INTERRUPT_TO_SILENCE = "interrupt_to_silence_ms"
    FIRST_RESPONSE_AUDIO = "first_response_audio_ms"


class EventKind(str, Enum):
    """Monotonic events used to derive the gated durations."""

    AUDIO_CAPTURED = "audio_captured"
    AUDIO_INGRESS_READY = "audio_ingress_ready"
    TURN_ENDED = "turn_ended"
    FIRST_RESPONSE_AUDIO = "first_response_audio"
    INTERRUPT_REQUESTED = "interrupt_requested"
    PLAYBACK_SILENT = "playback_silent"
    QUEUE_OBSERVED = "queue_observed"


@dataclass(frozen=True)
class LatencyGates:
    """Initial super-low-latency gates, expressed in milliseconds."""

    audio_ingress_p95_ms: float = 50.0
    interrupt_to_silence_p95_ms: float = 100.0
    first_response_audio_p95_ms: float = 300.0
    queue_age_p95_ms: float = 100.0
    queue_max_depth: int = 4
    queue_max_growth: int = 0

    def __post_init__(self) -> None:
        for name in (
            "audio_ingress_p95_ms",
            "interrupt_to_silence_p95_ms",
            "first_response_audio_p95_ms",
            "queue_age_p95_ms",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative duration")
        if self.queue_max_depth < 0 or self.queue_max_growth < 0:
            raise ValueError("Queue depth and growth gates must be non-negative")


@dataclass(frozen=True)
class LatencyEvent:
    turn_id: str
    kind: EventKind
    at_ns: int
    correlation_id: str


@dataclass(frozen=True)
class QueueObservation:
    at_ns: int
    depth: int
    oldest_age_ms: float


@dataclass(frozen=True)
class StageState:
    """Evidence supplied by the runtime, separate from measured latency."""

    transport_connected: bool = False
    model_connected: bool = False
    voice_trained: bool = False


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    """Return the deterministic nearest-rank percentile.

    Nearest rank makes latency gates easy to audit: p95 is the element at
    ``ceil(0.95 * n)`` in sorted order.  Empty samples fail instead of quietly
    producing a misleading zero.
    """

    if not values:
        raise ValueError("A percentile requires at least one sample")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency samples must be finite and non-negative")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


class RealtimeLatencyTracker:
    """Track bounded realtime spans and produce a fail-closed readiness status.

    A turn ID identifies one conversational turn.  Audio chunks additionally
    use their sequence as a correlation ID, and interruptions use their
    playback generation.  All buffers have fixed capacities so telemetry
    cannot become an unbounded realtime queue of its own.
    """

    def __init__(
        self,
        *,
        gates: LatencyGates | None = None,
        sample_capacity: int = 512,
        event_capacity: int = 4_096,
        pending_capacity: int = 1_024,
        minimum_samples: int = 20,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if sample_capacity <= 0 or event_capacity <= 0 or pending_capacity <= 0:
            raise ValueError("Telemetry capacities must be positive")
        if minimum_samples <= 0 or minimum_samples > sample_capacity:
            raise ValueError("minimum_samples must be in [1, sample_capacity]")
        self.gates = gates or LatencyGates()
        self.sample_capacity = sample_capacity
        self.event_capacity = event_capacity
        self.pending_capacity = pending_capacity
        self.minimum_samples = minimum_samples
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._samples = {
            metric: deque(maxlen=sample_capacity) for metric in LatencyMetric
        }
        self._events: deque[LatencyEvent] = deque(maxlen=event_capacity)
        self._queue: deque[QueueObservation] = deque(maxlen=sample_capacity)
        self._pending: OrderedDict[tuple[LatencyMetric, str, str], int] = OrderedDict()
        self._last_turn_timestamp: OrderedDict[str, int] = OrderedDict()
        self._last_queue_timestamp: int | None = None
        self._dropped_pending_spans = 0
        self._stages = StageState()

    @staticmethod
    def _identifier(value: object, name: str) -> str:
        identifier = str(value).strip()
        if not identifier or len(identifier) > 128:
            raise ValueError(f"{name} must be 1-128 characters")
        return identifier

    def _timestamp(self, at_ns: int | None) -> int:
        value = self._clock_ns() if at_ns is None else at_ns
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "Realtime timestamps must be non-negative integer nanoseconds"
            )
        return value

    def _append_event(
        self,
        turn_id: str,
        kind: EventKind,
        at_ns: int,
        correlation_id: str,
    ) -> None:
        previous = self._last_turn_timestamp.get(turn_id)
        if previous is not None and at_ns < previous:
            raise ValueError(
                "Events within a turn must use non-decreasing monotonic timestamps"
            )
        self._last_turn_timestamp.pop(turn_id, None)
        self._last_turn_timestamp[turn_id] = at_ns
        while len(self._last_turn_timestamp) > self.event_capacity:
            self._last_turn_timestamp.popitem(last=False)
        self._events.append(LatencyEvent(turn_id, kind, at_ns, correlation_id))

    def _start(
        self,
        metric: LatencyMetric,
        kind: EventKind,
        turn_id: object,
        correlation_id: object,
        at_ns: int | None,
    ) -> None:
        turn = self._identifier(turn_id, "turn_id")
        correlation = self._identifier(correlation_id, "correlation_id")
        timestamp = self._timestamp(at_ns)
        key = (metric, turn, correlation)
        with self._lock:
            if key in self._pending:
                raise ValueError(
                    f"A {metric.value} span with this identity is already open"
                )
            self._append_event(turn, kind, timestamp, correlation)
            if len(self._pending) >= self.pending_capacity:
                self._pending.popitem(last=False)
                self._dropped_pending_spans += 1
            self._pending[key] = timestamp

    def _finish(
        self,
        metric: LatencyMetric,
        kind: EventKind,
        turn_id: object,
        correlation_id: object,
        at_ns: int | None,
    ) -> float:
        turn = self._identifier(turn_id, "turn_id")
        correlation = self._identifier(correlation_id, "correlation_id")
        timestamp = self._timestamp(at_ns)
        key = (metric, turn, correlation)
        with self._lock:
            start = self._pending.get(key)
            if start is None:
                raise ValueError(f"A matching {metric.value} start event is required")
            if timestamp < start:
                raise ValueError("A span cannot finish before it starts")
            self._append_event(turn, kind, timestamp, correlation)
            del self._pending[key]
            elapsed_ms = (timestamp - start) / 1_000_000
            self._samples[metric].append(elapsed_ms)
            return elapsed_ms

    def mark_audio_captured(
        self, turn_id: object, sequence: int, *, at_ns: int | None = None
    ) -> None:
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Audio sequence must be a non-negative integer")
        self._start(
            LatencyMetric.AUDIO_INGRESS,
            EventKind.AUDIO_CAPTURED,
            turn_id,
            sequence,
            at_ns,
        )

    def mark_audio_ingress_ready(
        self, turn_id: object, sequence: int, *, at_ns: int | None = None
    ) -> float:
        return self._finish(
            LatencyMetric.AUDIO_INGRESS,
            EventKind.AUDIO_INGRESS_READY,
            turn_id,
            sequence,
            at_ns,
        )

    def mark_turn_ended(self, turn_id: object, *, at_ns: int | None = None) -> None:
        turn = self._identifier(turn_id, "turn_id")
        self._start(
            LatencyMetric.FIRST_RESPONSE_AUDIO,
            EventKind.TURN_ENDED,
            turn,
            turn,
            at_ns,
        )

    def mark_first_response_audio(
        self, turn_id: object, *, at_ns: int | None = None
    ) -> float:
        """Finish at the first assistant audio sample rendered by the client."""

        turn = self._identifier(turn_id, "turn_id")
        return self._finish(
            LatencyMetric.FIRST_RESPONSE_AUDIO,
            EventKind.FIRST_RESPONSE_AUDIO,
            turn,
            turn,
            at_ns,
        )

    def mark_interrupt_requested(
        self,
        turn_id: object,
        playback_generation: int,
        *,
        at_ns: int | None = None,
    ) -> None:
        if (
            isinstance(playback_generation, bool)
            or not isinstance(playback_generation, int)
            or playback_generation < 0
        ):
            raise ValueError("Playback generation must be a non-negative integer")
        self._start(
            LatencyMetric.INTERRUPT_TO_SILENCE,
            EventKind.INTERRUPT_REQUESTED,
            turn_id,
            playback_generation,
            at_ns,
        )

    def mark_playback_silent(
        self,
        turn_id: object,
        playback_generation: int,
        *,
        at_ns: int | None = None,
    ) -> float:
        return self._finish(
            LatencyMetric.INTERRUPT_TO_SILENCE,
            EventKind.PLAYBACK_SILENT,
            turn_id,
            playback_generation,
            at_ns,
        )

    def observe_queue(
        self,
        *,
        depth: int,
        oldest_age_ms: float,
        at_ns: int | None = None,
    ) -> None:
        """Record model/playback queue depth and age of its oldest item."""

        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("Queue depth must be a non-negative integer")
        age = float(oldest_age_ms)
        if not math.isfinite(age) or age < 0:
            raise ValueError("Queue age must be finite and non-negative")
        if depth == 0 and age != 0:
            raise ValueError("An empty queue must have zero oldest-item age")
        timestamp = self._timestamp(at_ns)
        with self._lock:
            if (
                self._last_queue_timestamp is not None
                and timestamp < self._last_queue_timestamp
            ):
                raise ValueError(
                    "Queue observations must use non-decreasing timestamps"
                )
            self._last_queue_timestamp = timestamp
            self._queue.append(QueueObservation(timestamp, depth, age))
            self._events.append(
                LatencyEvent("queue", EventKind.QUEUE_OBSERVED, timestamp, str(depth))
            )

    def set_stages(
        self,
        *,
        transport_connected: bool,
        model_connected: bool,
        voice_trained: bool,
    ) -> None:
        if not all(
            isinstance(value, bool)
            for value in (transport_connected, model_connected, voice_trained)
        ):
            raise ValueError("Readiness stages must be booleans")
        with self._lock:
            self._stages = StageState(
                transport_connected=transport_connected,
                model_connected=model_connected,
                voice_trained=voice_trained,
            )

    def recent_events(self) -> tuple[LatencyEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def empty_copy(self) -> "RealtimeLatencyTracker":
        """Create an empty tracker with the same auditable configuration.

        Realtime connection epochs must not inherit incomplete correlation
        spans from a closed peer.  A new epoch still has to collect every
        minimum sample again before it can become ready.
        """

        return type(self)(
            gates=self.gates,
            sample_capacity=self.sample_capacity,
            event_capacity=self.event_capacity,
            pending_capacity=self.pending_capacity,
            minimum_samples=self.minimum_samples,
            clock_ns=self._clock_ns,
        )

    @staticmethod
    def _metric_summary(
        samples: Sequence[float], *, gate_ms: float, minimum_samples: int
    ) -> dict[str, Any]:
        count = len(samples)
        enough = count >= minimum_samples
        p50 = nearest_rank_percentile(samples, 0.50) if samples else None
        p95 = nearest_rank_percentile(samples, 0.95) if samples else None
        passed = enough and p95 is not None and p95 <= gate_ms
        reasons: list[str] = []
        if not enough:
            reasons.append(
                f"requires at least {minimum_samples} samples; received {count}"
            )
        elif not passed:
            reasons.append(f"p95 {p95:.3f} ms exceeds {gate_ms:.3f} ms")
        return {
            "count": count,
            "p50_ms": p50,
            "p95_ms": p95,
            "max_ms": max(samples) if samples else None,
            "gate_p95_ms": gate_ms,
            "pass": passed,
            "reasons": reasons,
        }

    def status(self) -> dict[str, Any]:
        """Return JSON-safe stage and metric readiness without optimistic defaults."""

        with self._lock:
            samples = {
                metric: tuple(values) for metric, values in self._samples.items()
            }
            queue = tuple(self._queue)
            stages = self._stages
            pending_count = len(self._pending)
            dropped_pending = self._dropped_pending_spans

        metric_status = {
            LatencyMetric.AUDIO_INGRESS.value: self._metric_summary(
                samples[LatencyMetric.AUDIO_INGRESS],
                gate_ms=self.gates.audio_ingress_p95_ms,
                minimum_samples=self.minimum_samples,
            ),
            LatencyMetric.INTERRUPT_TO_SILENCE.value: self._metric_summary(
                samples[LatencyMetric.INTERRUPT_TO_SILENCE],
                gate_ms=self.gates.interrupt_to_silence_p95_ms,
                minimum_samples=self.minimum_samples,
            ),
            LatencyMetric.FIRST_RESPONSE_AUDIO.value: self._metric_summary(
                samples[LatencyMetric.FIRST_RESPONSE_AUDIO],
                gate_ms=self.gates.first_response_audio_p95_ms,
                minimum_samples=self.minimum_samples,
            ),
        }

        queue_ages = tuple(observation.oldest_age_ms for observation in queue)
        queue_count = len(queue)
        queue_enough = queue_count >= self.minimum_samples
        queue_p50 = nearest_rank_percentile(queue_ages, 0.50) if queue else None
        queue_p95 = nearest_rank_percentile(queue_ages, 0.95) if queue else None
        queue_max_depth = max(
            (observation.depth for observation in queue), default=None
        )
        queue_growth = queue[-1].depth - queue[0].depth if len(queue) >= 2 else None
        queue_reasons: list[str] = []
        if not queue_enough:
            queue_reasons.append(
                f"requires at least {self.minimum_samples} observations; received {queue_count}"
            )
        if queue_p95 is not None and queue_p95 > self.gates.queue_age_p95_ms:
            queue_reasons.append(
                f"age p95 {queue_p95:.3f} ms exceeds {self.gates.queue_age_p95_ms:.3f} ms"
            )
        if queue_max_depth is not None and queue_max_depth > self.gates.queue_max_depth:
            queue_reasons.append(
                f"maximum depth {queue_max_depth} exceeds {self.gates.queue_max_depth}"
            )
        if queue_growth is None:
            queue_reasons.append(
                "at least two observations are required to measure growth"
            )
        elif queue_growth > self.gates.queue_max_growth:
            queue_reasons.append(
                f"net growth {queue_growth} exceeds {self.gates.queue_max_growth}"
            )
        queue_pass = queue_enough and not queue_reasons
        queue_status = {
            "count": queue_count,
            "oldest_age_p50_ms": queue_p50,
            "oldest_age_p95_ms": queue_p95,
            "gate_age_p95_ms": self.gates.queue_age_p95_ms,
            "max_depth": queue_max_depth,
            "gate_max_depth": self.gates.queue_max_depth,
            "net_growth": queue_growth,
            "gate_max_growth": self.gates.queue_max_growth,
            "pass": queue_pass,
            "reasons": queue_reasons,
        }

        trace_reasons: list[str] = []
        if pending_count:
            trace_reasons.append(f"{pending_count} latency spans are incomplete")
        if dropped_pending:
            trace_reasons.append(f"{dropped_pending} pending spans were evicted")
        trace_pass = not trace_reasons

        transport_reasons: list[str] = []
        if not stages.transport_connected:
            transport_reasons.append("transport is not connected")
        for name in (
            LatencyMetric.AUDIO_INGRESS.value,
            LatencyMetric.INTERRUPT_TO_SILENCE.value,
        ):
            if not metric_status[name]["pass"]:
                transport_reasons.extend(
                    f"{name}: {reason}" for reason in metric_status[name]["reasons"]
                )
        if not queue_pass:
            transport_reasons.extend(f"queue: {reason}" for reason in queue_reasons)
        transport_reasons.extend(f"trace: {reason}" for reason in trace_reasons)
        transport_ready = not transport_reasons

        model_reasons: list[str] = []
        if not stages.model_connected:
            model_reasons.append("model is not connected")
        response = metric_status[LatencyMetric.FIRST_RESPONSE_AUDIO.value]
        if not response["pass"]:
            model_reasons.extend(
                f"{LatencyMetric.FIRST_RESPONSE_AUDIO.value}: {reason}"
                for reason in response["reasons"]
            )
        model_reasons.extend(f"trace: {reason}" for reason in trace_reasons)
        model_ready = not model_reasons

        voice_reasons = (
            [] if stages.voice_trained else ["voice checkpoint is not trained"]
        )
        voice_reasons.extend(f"trace: {reason}" for reason in trace_reasons)
        voice_ready = not voice_reasons
        ready = transport_ready and model_ready and voice_ready and trace_pass

        return {
            "ready": ready,
            "clock": "monotonic_ns",
            "minimum_samples": self.minimum_samples,
            "sample_capacity": self.sample_capacity,
            "gates": asdict(self.gates),
            "stages": {
                "transport": {
                    "declared_connected": stages.transport_connected,
                    "ready": transport_ready,
                    "reasons": transport_reasons,
                },
                "model": {
                    "declared_connected": stages.model_connected,
                    "ready": model_ready,
                    "reasons": model_reasons,
                },
                "voice_trained": {
                    "declared_trained": stages.voice_trained,
                    "ready": voice_ready,
                    "reasons": voice_reasons,
                },
            },
            "metrics": metric_status,
            "queue": queue_status,
            "trace": {
                "pass": trace_pass,
                "pending_spans": pending_count,
                "dropped_pending_spans": dropped_pending,
                "reasons": trace_reasons,
            },
        }
