from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import threading

import pytest

import mini_arithmetic.personal_training as personal_training_module
from mini_arithmetic.apprenticeship import ApprenticeshipStore
from mini_arithmetic.personal_training import (
    PERSONAL_HOLDOUT_EXAMPLES,
    PERSONAL_TRAINING_EXAMPLES,
    PERSONAL_TRAINING_THRESHOLD,
    PersonalAdapterArtifact,
    PersonalAdapterCandidate,
    PersonalAdapterTrainingRequest,
    PersonalRuntimeIdentity,
    PersonalTrainingConfig,
    PersonalTrainingCoordinator,
    PersonalTrainingError,
    PrivateAdapterRegistry,
)


FOUNDATION_SHA256 = "1" * 64
ADAPTER_SCHEMA_SHA256 = "2" * 64


class FakeRuntime:
    def __init__(self, *, backend: str = "fake") -> None:
        self.backend = backend
        self.foundation_sha256 = FOUNDATION_SHA256
        self.adapter_schema_sha256 = ADAPTER_SCHEMA_SHA256
        self.active_adapter_id: str | None = None
        self.activations: list[str | None] = []

    def personal_runtime_identity(self) -> PersonalRuntimeIdentity:
        return PersonalRuntimeIdentity(
            backend=self.backend,
            foundation_sha256=self.foundation_sha256,
            adapter_schema_sha256=self.adapter_schema_sha256,
            active_adapter_id=self.active_adapter_id,
        )

    def prepare_personal_adapter(self, artifact: PersonalAdapterArtifact) -> object:
        assert artifact.artifact_path.is_file()
        return artifact.manifest.adapter_id

    def activate_prepared_personal_adapter(
        self,
        prepared: object,
        adapter_id: str,
    ) -> None:
        assert prepared == adapter_id
        self.active_adapter_id = adapter_id
        self.activations.append(adapter_id)

    def activate_foundation_only(self) -> None:
        self.active_adapter_id = None
        self.activations.append(None)


class FailingActivationRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_activation = False

    def activate_prepared_personal_adapter(
        self,
        prepared: object,
        adapter_id: str,
    ) -> None:
        if self.fail_next_activation:
            self.fail_next_activation = False
            self.active_adapter_id = adapter_id
            raise RuntimeError("simulated partial activation failure")
        super().activate_prepared_personal_adapter(prepared, adapter_id)


class FakeTrainer:
    backend_name = "fake"

    def __init__(
        self,
        *,
        baseline_loss: float = 2.0,
        candidate_loss: float = 1.0,
        baseline_retention_loss: float = 1.0,
        candidate_retention_loss: float = 1.0,
        foundation_after: str = FOUNDATION_SHA256,
    ) -> None:
        self.baseline_loss = baseline_loss
        self.candidate_loss = candidate_loss
        self.baseline_retention_loss = baseline_retention_loss
        self.candidate_retention_loss = candidate_retention_loss
        self.foundation_after = foundation_after
        self.requests: list[PersonalAdapterTrainingRequest] = []

    def train_adapter(
        self,
        request: PersonalAdapterTrainingRequest,
    ) -> PersonalAdapterCandidate:
        self.requests.append(request)
        payload = f"private adapter {request.adapter_id}".encode()
        artifact = request.work_dir / "candidate.safetensors"
        artifact.write_bytes(payload)
        return PersonalAdapterCandidate(
            artifact_path=artifact,
            artifact_format="safetensors",
            artifact_sha256=hashlib.sha256(payload).hexdigest(),
            backend=self.backend_name,
            foundation_sha256_before=request.foundation_sha256,
            foundation_sha256_after=self.foundation_after,
            adapter_schema_sha256=request.adapter_schema_sha256,
            baseline_holdout_loss=self.baseline_loss,
            candidate_holdout_loss=self.candidate_loss,
            metrics={"updates": 3},
            baseline_retention_loss=(
                self.baseline_retention_loss if request.retention_examples else None
            ),
            candidate_retention_loss=(
                self.candidate_retention_loss if request.retention_examples else None
            ),
        )


class BlockingFirstTrainer(FakeTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def train_adapter(
        self,
        request: PersonalAdapterTrainingRequest,
    ) -> PersonalAdapterCandidate:
        if not self.requests:
            self.entered.set()
            assert self.release.wait(timeout=5)
        return super().train_adapter(request)


class ContextCheckingTrainer(FakeTrainer):
    def personal_training_target_fits(self, prompt: str, human_response: str) -> bool:
        return bool(prompt.strip()) and len(human_response) <= 32


def _config(tmp_path: Path) -> PersonalTrainingConfig:
    return PersonalTrainingConfig(
        enabled=True,
        artifact_root=tmp_path / "personal-model",
        minimum_holdout_improvement=0.0,
        maximum_adapter_bytes=1024 * 1024,
    )


def _add_dialogue_lessons(
    store: ApprenticeshipStore,
    count: int,
    *,
    session_id: str = "conversation",
    start: int = 0,
    capture: str = "text-chat",
    responses: dict[int, str] | None = None,
) -> list[int]:
    store.create_session(session_id)
    messages = store.list_messages(session_id)
    prior_assistant = next(
        (message for message in reversed(messages) if message.role == "assistant"),
        None,
    )
    lesson_ids: list[int] = []
    for offset in range(start, start + count):
        user = store.add_message(
            session_id,
            "user",
            (responses or {}).get(offset, f"human response {offset}"),
            reply_to_id=prior_assistant.id if prior_assistant else None,
            metadata={"input": "text" if capture == "text-chat" else "audio"},
        )
        lesson = store.add_trusted_dialogue_lesson(
            human_response_message_id=user.id,
            prompt_message_id=prior_assistant.id if prior_assistant else None,
            consent_local_training=True,
            metadata={"capture": capture, "automatic": True},
        )
        lesson_ids.append(lesson.id)
        prior_assistant = store.add_message(
            session_id,
            "assistant",
            f"tiny dog prompt {offset}",
            reply_to_id=user.id,
        )
    return lesson_ids


def test_five_typed_turns_train_exact_four_plus_one_and_activate_locally(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD - 1)
    trainer = FakeTrainer()
    runtime = FakeRuntime()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=runtime,
    )

    assert coordinator.consider_training() is False
    _add_dialogue_lessons(store, 1, start=4)
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert len(trainer.requests) == 1
    request = trainer.requests[0]
    assert len(request.training_examples) == PERSONAL_TRAINING_EXAMPLES
    assert len(request.validation_examples) == PERSONAL_HOLDOUT_EXAMPLES
    assert {
        example.lesson_id
        for example in request.training_examples + request.validation_examples
    } == set(range(1, PERSONAL_TRAINING_THRESHOLD + 1))
    assert runtime.active_adapter_id == request.adapter_id
    run = store.get_training_status(request.run_id)
    assert run is not None and run.status == "promoted"
    assert run.metadata["authority"] == "personal-local"
    status = coordinator.status()
    assert status["live_model_updated"] is True
    assert status["produces_community_updates"] is False
    assert status["requires_signatures"] is False
    assert status["uploads_enabled"] is False

    adapter_dir = tmp_path / "personal-model" / "adapters" / request.adapter_id
    manifest_text = (adapter_dir / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    assert manifest["visibility"] == "private-local-only"
    assert set(entry.name for entry in adapter_dir.iterdir()) == {
        "adapter.safetensors",
        "manifest.json",
    }
    assert "human response" not in manifest_text
    assert "community" not in manifest_text
    assert "signature" not in manifest_text
    if os.name != "nt":
        assert stat_mode(adapter_dir) == 0o700
        assert stat_mode(adapter_dir / "adapter.safetensors") == 0o600
        assert stat_mode(adapter_dir / "manifest.json") == 0o600


def test_raw_voice_markers_do_not_count_without_a_confirmed_transcript(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD, capture="voice-audio")
    trainer = FakeTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=FakeRuntime(),
    )

    assert coordinator.consider_training() is False
    assert trainer.requests == []
    status = coordinator.status()
    assert status["qualifying_pending"] == 0
    assert status["raw_voice_counts"] is False
    assert "transcript" in status["voice_reason"]


def test_status_and_subthreshold_trigger_do_not_run_full_lesson_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD - 1)
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeTrainer(),
        runtime=FakeRuntime(),
    )

    def unexpected_scan(*args: object, **kwargs: object) -> list[object]:
        del args, kwargs
        raise AssertionError(
            "full lesson scan should only run in the fifth-turn worker"
        )

    monkeypatch.setattr(coordinator, "_select_typed_lessons", unexpected_scan)
    status = coordinator.status()
    assert status["qualifying_pending"] == 4
    assert status["progress"] == 0.8
    assert coordinator.consider_training() is False


def test_oversized_legacy_lesson_is_quarantined_and_later_five_train(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    oversized_id = _add_dialogue_lessons(
        store,
        1,
        responses={0: "x" * 33},
    )[0]
    valid_ids = _add_dialogue_lessons(
        store,
        PERSONAL_TRAINING_THRESHOLD,
        start=1,
    )
    trainer = ContextCheckingTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=FakeRuntime(),
    )

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    assert len(trainer.requests) == 1
    request_ids = {
        example.lesson_id
        for example in (
            trainer.requests[0].training_examples
            + trainer.requests[0].validation_examples
        )
    }
    assert request_ids == set(valid_ids)
    oversized = store.get_lesson(oversized_id)
    assert oversized is not None and oversized.status == "ineligible"
    assert (
        oversized.metadata["personal_training_quarantine_reason"]
        == "outside-personal-training-context"
    )
    assert coordinator.consider_training() is False
    status = coordinator.status()
    assert status["ineligible_pending"] == 0
    assert status["quarantined_lessons"] == 1
    assert status["qualifying_pending"] == 0


def test_consumed_duplicates_do_not_count_or_starve_later_unique_turns(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    trainer = FakeTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=FakeRuntime(),
    )

    first_ids = _add_dialogue_lessons(
        store,
        PERSONAL_TRAINING_THRESHOLD,
        session_id="first",
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    assert len(trainer.requests) == 1
    coordinator.shutdown()

    # The consumed-content gate is rebuilt from the private ledger, not only
    # held in process memory.
    trainer = FakeTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=FakeRuntime(),
    )

    duplicate_ids = _add_dialogue_lessons(
        store,
        PERSONAL_TRAINING_THRESHOLD,
        session_id="duplicates",
    )
    for lesson_id in duplicate_ids:
        assert coordinator.consider_lesson(lesson_id) is False
    duplicate_status = coordinator.status()
    assert duplicate_status["qualifying_pending"] == 0
    assert duplicate_status["progress"] == 0.0
    assert duplicate_status["quarantined_lessons"] == 5
    assert all(
        store.get_lesson(lesson_id).status == "ineligible"
        for lesson_id in duplicate_ids
    )
    assert trainer.requests == []

    unique_ids: list[int] = []
    for offset in range(5, 9):
        lesson_id = _add_dialogue_lessons(
            store,
            1,
            session_id="unique",
            start=offset,
        )[0]
        unique_ids.append(lesson_id)
        assert coordinator.consider_lesson(lesson_id) is False
    assert coordinator.status()["qualifying_pending"] == 4
    last_unique = _add_dialogue_lessons(
        store,
        1,
        session_id="unique",
        start=9,
    )[0]
    unique_ids.append(last_unique)
    assert coordinator.consider_lesson(last_unique) is True
    assert coordinator.wait_for_idle()

    assert len(trainer.requests) == 1
    trained_ids = {
        example.lesson_id
        for example in (
            trainer.requests[0].training_examples
            + trainer.requests[0].validation_examples
        )
    }
    assert trained_ids == set(unique_ids)
    assert trained_ids.isdisjoint(first_ids)
    assert trained_ids.isdisjoint(duplicate_ids)
    assert coordinator.status()["qualifying_pending"] == 0


def test_duplicate_before_threshold_is_quarantined_then_next_unique_trains(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    responses = {
        0: "same answer",
        1: "same answer",
        2: "answer b",
        3: "answer c",
        4: "answer d",
    }
    first_five = _add_dialogue_lessons(
        store,
        5,
        responses=responses,
    )
    # The first two bootstrap targets have identical prompt+answer content.
    second_session_duplicate = _add_dialogue_lessons(
        store,
        1,
        session_id="second-bootstrap",
        responses={0: "same answer"},
    )[0]
    # Replace the non-duplicate second item with the exact second bootstrap
    # row, yielding four unique contents among five pending candidates.
    store.quarantine_pending_lesson_ids(
        [first_five[1]],
        reason="test-setup",
    )
    pending_ids = [first_five[0], *first_five[2:], second_session_duplicate]

    trainer = FakeTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=FakeRuntime(),
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    assert trainer.requests == []
    assert store.get_lesson(second_session_duplicate).status == "ineligible"
    assert coordinator.status()["qualifying_pending"] == 4

    final_unique = _add_dialogue_lessons(
        store,
        1,
        session_id="final-unique",
        responses={0: "answer e"},
    )[0]
    assert coordinator.consider_lesson(final_unique) is True
    assert coordinator.wait_for_idle()
    assert len(trainer.requests) == 1
    trained_ids = {
        example.lesson_id
        for example in (
            trainer.requests[0].training_examples
            + trainer.requests[0].validation_examples
        )
    }
    assert trained_ids == {*pending_ids[:-1], final_unique}


def test_bounded_pending_scan_drains_quarantine_then_reaches_older_valid_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    valid_ids = _add_dialogue_lessons(store, 5)
    oversized_ids = _add_dialogue_lessons(
        store,
        5,
        start=5,
        responses={index: "x" * 33 for index in range(5, 10)},
    )
    monkeypatch.setattr(personal_training_module, "PERSONAL_PENDING_SCAN_LIMIT", 5)
    trainer = ContextCheckingTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=FakeRuntime(),
    )

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    assert len(trainer.requests) == 1
    trained_ids = {
        example.lesson_id
        for example in (
            trainer.requests[0].training_examples
            + trainer.requests[0].validation_examples
        )
    }
    assert trained_ids == set(valid_ids)
    assert all(store.get_lesson(item).status == "ineligible" for item in oversized_ids)


def test_leased_startup_recovers_interrupted_personal_claim_and_retries(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    config = _config(tmp_path)
    registry_root = config.artifact_root.expanduser().resolve()
    registry_id = hashlib.sha256(
        f"tinydog-personal-registry-v1\0{registry_root}".encode("utf-8")
    ).hexdigest()
    interrupted, claimed = store.claim_pending_lesson_ids(
        lesson_ids,
        metadata={
            "authority": "personal-local",
            "parameter_scope": "adapter-only",
            "personal_registry_id": registry_id,
        },
    )
    assert interrupted is not None and len(claimed) == PERSONAL_TRAINING_THRESHOLD

    trainer = FakeTrainer()
    runtime = FakeRuntime()
    coordinator = PersonalTrainingCoordinator(
        store,
        config,
        trainer=trainer,
        runtime=runtime,
        recover_interrupted=True,
    )

    recovered = store.get_training_status(interrupted.id)
    assert recovered is not None and recovered.status == "failed"
    assert recovered.metadata["recovered"] is True
    assert all(
        store.get_lesson(identifier).status == "pending" for identifier in lesson_ids
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    assert runtime.active_adapter_id is not None


def test_non_improving_candidate_is_not_activated_and_batch_is_consumed(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    runtime = FakeRuntime()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeTrainer(baseline_loss=1.0, candidate_loss=1.0),
        runtime=runtime,
    )

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert runtime.active_adapter_id is None
    assert coordinator.status()["state"] == "rejected"
    run = store.list_training_runs()[0]
    assert run.status == "local_rejected"
    assert run.metadata["authority"] == "personal-local"
    assert all(store.get_lesson(lesson_id).status == "used" for lesson_id in lesson_ids)
    assert list((tmp_path / "personal-model" / "adapters").iterdir()) == []


def test_replay_is_bounded_and_retention_anchor_is_stable_and_disjoint(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    trainer = FakeTrainer()
    runtime = FakeRuntime()
    config = PersonalTrainingConfig(
        enabled=True,
        artifact_root=tmp_path / "personal-model",
        maximum_replay_examples=3,
        retention_examples=2,
        maximum_retention_loss_increase=0.05,
        maximum_adapter_bytes=1024 * 1024,
    )
    coordinator = PersonalTrainingCoordinator(
        store,
        config,
        trainer=trainer,
        runtime=runtime,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    assert trainer.requests[0].replay_examples == ()
    assert trainer.requests[0].retention_examples == ()

    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD, start=5)
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    second = trainer.requests[1]
    retention_ids = tuple(example.lesson_id for example in second.retention_examples)
    replay_ids = tuple(example.lesson_id for example in second.replay_examples)
    assert retention_ids == (1, 2)
    assert len(replay_ids) == 3
    assert set(replay_ids) == {3, 4, 5}
    assert set(replay_ids).isdisjoint(retention_ids)

    # Selection is stable for a fixed immutable ledger/run identity, while the
    # replay cap remains independent of an ever-growing conversation history.
    replay_again, retention_again = coordinator._select_replay_and_retention(123)
    replay_repeat, retention_repeat = coordinator._select_replay_and_retention(123)
    assert replay_again == replay_repeat
    assert retention_again == retention_repeat
    assert tuple(example.lesson_id for example in retention_again) == (1, 2)
    assert len(replay_again) == 3

    status = coordinator.status()
    assert status["maximum_replay_examples_per_run"] == 3
    assert status["maximum_retention_examples_per_run"] == 2
    assert status["last_replay_examples"] == 3
    assert status["last_retention_examples"] == 2
    assert status["maximum_retention_loss_increase"] == 0.05
    assert status["historical_used_examples"] == 10


def test_historical_retention_regression_rejects_an_improved_new_holdout(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    runtime = FakeRuntime()
    first = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeTrainer(),
        runtime=runtime,
    )
    assert first.consider_training() is True
    assert first.wait_for_idle()
    known_good = runtime.active_adapter_id

    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD, start=5)
    regressing = FakeTrainer(
        baseline_loss=2.0,
        candidate_loss=1.0,
        baseline_retention_loss=1.0,
        candidate_retention_loss=1.051,
    )
    second = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=regressing,
        runtime=runtime,
    )
    assert second.consider_training() is True
    assert second.wait_for_idle()

    assert runtime.active_adapter_id == known_good
    run = store.get_training_status(regressing.requests[0].run_id)
    assert run is not None and run.status == "local_rejected"
    assert run.metadata["reason"] == "historical-retention-regressed"
    assert run.metrics["retention_preserved"] is False
    assert run.metrics["replay_examples"] == 3
    assert run.metrics["retention_examples"] == 2


def test_foundation_mutation_report_fails_closed_and_requeues_lessons(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    runtime = FakeRuntime()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeTrainer(foundation_after="3" * 64),
        runtime=runtime,
    )

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert coordinator.status()["state"] == "failed"
    assert runtime.active_adapter_id is None
    assert all(
        store.get_lesson(lesson_id).status == "pending" for lesson_id in lesson_ids
    )
    assert store.list_training_runs()[0].status == "failed"


def test_second_adapter_builds_on_first_and_rollback_reactivates_first(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    trainer = FakeTrainer()
    runtime = FakeRuntime()
    config = _config(tmp_path)
    coordinator = PersonalTrainingCoordinator(
        store,
        config,
        trainer=trainer,
        runtime=runtime,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    first_id = trainer.requests[0].adapter_id

    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD, start=5)
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    second_request = trainer.requests[1]
    assert second_request.parent_adapter_id == first_id
    assert second_request.parent_adapter_path is not None
    assert second_request.parent_adapter_path.is_file()
    assert runtime.active_adapter_id == second_request.adapter_id

    status = coordinator.rollback()
    assert runtime.active_adapter_id == first_id
    assert status["state"] == "rolled_back"
    second_run = store.get_training_status(second_request.run_id)
    assert second_run is not None and second_run.status == "rolled_back"

    status = coordinator.rollback()
    assert runtime.active_adapter_id is None
    assert status["state"] == "rolled_back"
    first_run = store.get_training_status(trainer.requests[0].run_id)
    assert first_run is not None and first_run.status == "rolled_back"
    assert first_run.metadata["explicit_user_rollback"] is True
    retired_lesson_ids = {
        example.lesson_id
        for request in trainer.requests
        for example in request.training_examples + request.validation_examples
    }
    assert all(
        store.get_lesson(lesson_id).status == "used" for lesson_id in retired_lesson_ids
    )

    # A second rollback walks back to the foundation; it must not toggle the
    # already rolled-back second adapter back into service on restart.
    restarted_runtime = FakeRuntime()
    restarted = PersonalTrainingCoordinator(
        store,
        config,
        trainer=FakeTrainer(),
        runtime=restarted_runtime,
    )
    assert restarted_runtime.active_adapter_id is None
    assert restarted.status()["active_adapter_id"] is None
    assert restarted.consider_training() is False


def test_activation_failure_restores_prior_adapter_and_requeues_batch(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    runtime = FailingActivationRuntime()
    trainer = FakeTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=runtime,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    known_good = runtime.active_adapter_id
    assert known_good is not None

    second_ids = _add_dialogue_lessons(
        store,
        PERSONAL_TRAINING_THRESHOLD,
        start=5,
    )
    runtime.fail_next_activation = True
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert runtime.active_adapter_id == known_good
    assert coordinator.status()["state"] == "failed"
    second_run = store.get_training_status(trainer.requests[1].run_id)
    assert second_run is not None and second_run.status == "rolled_back"
    assert all(
        store.get_lesson(lesson_id).status == "pending" for lesson_id in second_ids
    )


def test_new_coordinator_restores_active_private_adapter(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    trainer = FakeTrainer()
    first_runtime = FakeRuntime()
    config = _config(tmp_path)
    first = PersonalTrainingCoordinator(
        store,
        config,
        trainer=trainer,
        runtime=first_runtime,
    )
    assert first.consider_training() is True
    assert first.wait_for_idle()
    active = first_runtime.active_adapter_id
    assert active is not None

    restarted_runtime = FakeRuntime()
    restarted = PersonalTrainingCoordinator(
        store,
        config,
        trainer=FakeTrainer(),
        runtime=restarted_runtime,
    )
    assert restarted_runtime.active_adapter_id == active
    assert restarted.status()["active_adapter_id"] == active


def test_restart_requeues_promotion_that_never_reached_active_pointer(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    config = _config(tmp_path)
    first_runtime = FakeRuntime()
    first = PersonalTrainingCoordinator(
        store,
        config,
        trainer=FakeTrainer(),
        runtime=first_runtime,
    )
    assert first.consider_training() is True
    assert first.wait_for_idle()
    known_good = first_runtime.active_adapter_id
    assert known_good is not None
    first_run = store.list_training_runs(status="promoted")[0]
    registry_id = first_run.metadata["personal_registry_id"]

    interrupted_ids = _add_dialogue_lessons(
        store,
        PERSONAL_TRAINING_THRESHOLD,
        start=5,
    )
    interrupted, claimed = store.claim_pending_lesson_ids(
        interrupted_ids,
        checkpoint_in=known_good,
        metadata={
            "authority": "personal-local",
            "parameter_scope": "adapter-only",
            "personal_registry_id": registry_id,
        },
    )
    assert interrupted is not None and len(claimed) == PERSONAL_TRAINING_THRESHOLD
    store.promote_training_run(
        interrupted.id,
        interrupted_ids,
        checkpoint_out=str(tmp_path / "personal-model" / "adapters" / "personal-crash"),
        metadata={
            "authority": "personal-local",
            "adapter_id": "personal-crash",
            "personal_registry_id": registry_id,
        },
    )

    restarted_runtime = FakeRuntime()
    retry_trainer = FakeTrainer()
    restarted = PersonalTrainingCoordinator(
        store,
        config,
        trainer=retry_trainer,
        runtime=restarted_runtime,
    )

    recovered = store.get_training_status(interrupted.id)
    assert recovered is not None and recovered.status == "rolled_back"
    assert all(
        store.get_lesson(lesson_id).status == "pending" for lesson_id in interrupted_ids
    )
    assert restarted_runtime.active_adapter_id == known_good
    assert restarted.status()["qualifying_pending"] == 5
    assert restarted.consider_training() is True
    assert restarted.wait_for_idle()
    retry_ids = {
        example.lesson_id
        for example in (
            retry_trainer.requests[0].training_examples
            + retry_trainer.requests[0].validation_examples
        )
    }
    assert retry_ids == set(interrupted_ids)


def test_backend_mismatch_is_rejected_before_training(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    trainer = FakeTrainer()
    runtime = FakeRuntime(backend="another-backend")

    with pytest.raises(PersonalTrainingError, match="backends do not match"):
        PersonalTrainingCoordinator(
            store,
            _config(tmp_path),
            trainer=trainer,
            runtime=runtime,
        )


def test_fifth_turn_arriving_during_closeout_schedules_the_next_batch(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    trainer = BlockingFirstTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=FakeRuntime(),
    )

    assert coordinator.consider_training() is True
    assert trainer.entered.wait(timeout=5)
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD, start=5)
    assert coordinator.consider_training() is False
    trainer.release.set()
    assert coordinator.wait_for_idle(timeout=10)

    assert len(trainer.requests) == 2
    assert all(len(request.training_examples) == 4 for request in trainer.requests)
    assert all(len(request.validation_examples) == 1 for request in trainer.requests)


def test_exclusive_recovery_clears_validated_stale_work_directory(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    registry = PrivateAdapterRegistry(
        config.artifact_root,
        maximum_adapter_bytes=config.maximum_adapter_bytes,
    )
    stale = registry.create_work_dir("personal-stale")
    (stale / "partial.bin").write_bytes(b"partial")

    PersonalTrainingCoordinator(
        ApprenticeshipStore(tmp_path / "lessons.sqlite3"),
        config,
        trainer=FakeTrainer(),
        runtime=FakeRuntime(),
        recover_interrupted=True,
    )

    assert not stale.exists()
    assert list(registry.work.iterdir()) == []


def test_three_promotions_two_rollbacks_cannot_resurrect_pruned_ancestor(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    trainer = FakeTrainer()
    runtime = FakeRuntime()
    config = _config(tmp_path)
    coordinator = PersonalTrainingCoordinator(
        store,
        config,
        trainer=trainer,
        runtime=runtime,
    )
    for start in (0, 5, 10):
        _add_dialogue_lessons(
            store,
            PERSONAL_TRAINING_THRESHOLD,
            start=start,
        )
        assert coordinator.consider_training() is True
        assert coordinator.wait_for_idle()

    first, second, third = (request.adapter_id for request in trainer.requests)
    assert {entry.name for entry in coordinator.registry.adapters.iterdir()} == {
        second,
        third,
    }
    assert not (coordinator.registry.adapters / first).exists()

    coordinator.rollback()
    assert runtime.active_adapter_id == second
    coordinator.rollback()
    assert runtime.active_adapter_id is None
    assert all(run.status == "rolled_back" for run in store.list_training_runs())
    assert all(
        lesson.status == "used" for lesson in store.list_recent_lessons(limit=100)
    )

    restarted = PersonalTrainingCoordinator(
        store,
        config,
        trainer=FakeTrainer(),
        runtime=FakeRuntime(),
        recover_interrupted=True,
    )
    assert restarted.consider_training() is False
    assert restarted.status()["qualifying_pending"] == 0


def test_shutdown_cancels_aware_worker_and_fails_closed_for_stuck_worker(
    tmp_path: Path,
) -> None:
    class CancelAwareTrainer(FakeTrainer):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()

        def train_adapter(self, request: PersonalAdapterTrainingRequest):
            self.entered.set()
            assert request.cancellation_event is not None
            assert request.cancellation_event.wait(timeout=2)
            raise PersonalTrainingError("cancelled")

    store = ApprenticeshipStore(tmp_path / "cancel.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    aware = CancelAwareTrainer()
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path / "aware"),
        trainer=aware,
        runtime=FakeRuntime(),
    )
    assert coordinator.consider_training() is True
    assert aware.entered.wait(timeout=2)
    coordinator.shutdown(timeout=2)
    assert coordinator.wait_for_idle()

    blocked_store = ApprenticeshipStore(tmp_path / "blocked.sqlite3")
    _add_dialogue_lessons(blocked_store, PERSONAL_TRAINING_THRESHOLD)
    blocked = BlockingFirstTrainer()
    stuck = PersonalTrainingCoordinator(
        blocked_store,
        _config(tmp_path / "blocked"),
        trainer=blocked,
        runtime=FakeRuntime(),
    )
    assert stuck.consider_training() is True
    assert blocked.entered.wait(timeout=2)
    with pytest.raises(PersonalTrainingError, match="retaining the exclusive"):
        stuck.shutdown(timeout=0.01)
    blocked.release.set()
    assert stuck.wait_for_idle(timeout=2)


@pytest.mark.parametrize("failure_point", ["create_work_dir", "request"])
def test_post_claim_setup_failure_requeues_in_process_and_cleans_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    store = ApprenticeshipStore(tmp_path / f"{failure_point}.sqlite3")
    lesson_ids = _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path / failure_point),
        trainer=FakeTrainer(),
        runtime=FakeRuntime(),
    )
    if failure_point == "create_work_dir":
        original_create = coordinator.registry.create_work_dir

        def fail_after_mkdir(adapter_id: str) -> Path:
            original_create(adapter_id)
            raise OSError("simulated work-directory failure")

        monkeypatch.setattr(
            coordinator.registry,
            "create_work_dir",
            fail_after_mkdir,
        )
    else:

        def fail_request(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("simulated request-construction failure")

        monkeypatch.setattr(coordinator, "_request", fail_request)

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    assert coordinator.status()["state"] == "failed"
    assert all(store.get_lesson(item).status == "pending" for item in lesson_ids)
    run = store.list_training_runs()[0]
    assert run.status == "failed"
    assert list(coordinator.registry.work.iterdir()) == []


def test_cleanup_failure_after_pointer_commit_does_not_undo_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    trainer = FakeTrainer()
    runtime = FakeRuntime()
    config = _config(tmp_path)
    coordinator = PersonalTrainingCoordinator(
        store,
        config,
        trainer=trainer,
        runtime=runtime,
    )

    def fail_prune() -> int:
        raise OSError("simulated post-pointer cleanup failure")

    monkeypatch.setattr(
        coordinator.registry,
        "prune_unreferenced_adapters",
        fail_prune,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    adapter_id = trainer.requests[0].adapter_id
    assert coordinator.status()["state"] == "ready"
    assert runtime.active_adapter_id == adapter_id
    assert coordinator.registry.read_state()[0] == adapter_id
    run = store.get_training_status(trainer.requests[0].run_id)
    assert run is not None and run.status == "promoted"
    assert all(store.get_lesson(item).status == "used" for item in lesson_ids)

    restarted_runtime = FakeRuntime()
    restarted = PersonalTrainingCoordinator(
        store,
        config,
        trainer=FakeTrainer(),
        runtime=restarted_runtime,
    )
    assert restarted_runtime.active_adapter_id == adapter_id
    assert restarted.status()["qualifying_pending"] == 0


def test_missing_current_artifact_falls_back_and_requeues_only_that_run(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    trainer = FakeTrainer()
    runtime = FakeRuntime()
    config = _config(tmp_path)
    coordinator = PersonalTrainingCoordinator(
        store,
        config,
        trainer=trainer,
        runtime=runtime,
    )
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    first_id = trainer.requests[0].adapter_id

    second_lesson_ids = _add_dialogue_lessons(
        store,
        PERSONAL_TRAINING_THRESHOLD,
        start=5,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()
    second_request = trainer.requests[1]
    shutil.rmtree(coordinator.registry.adapters / second_request.adapter_id)

    restarted_runtime = FakeRuntime()
    restarted = PersonalTrainingCoordinator(
        store,
        config,
        trainer=FakeTrainer(),
        runtime=restarted_runtime,
    )

    assert restarted_runtime.active_adapter_id == first_id
    assert restarted.registry.read_state()[:2] == (first_id, None)
    missing_run = store.get_training_status(second_request.run_id)
    assert missing_run is not None and missing_run.status == "rolled_back"
    assert all(
        store.get_lesson(lesson_id).status == "pending"
        for lesson_id in second_lesson_ids
    )
    assert restarted.status()["qualifying_pending"] == 5


def test_adapter_and_state_publications_request_parent_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, PERSONAL_TRAINING_THRESHOLD)
    coordinator = PersonalTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeTrainer(),
        runtime=FakeRuntime(),
    )
    synced: list[Path] = []
    monkeypatch.setattr(
        personal_training_module,
        "_fsync_directory",
        lambda path: synced.append(path.resolve()),
    )

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert coordinator.registry.adapters in synced
    assert coordinator.registry.root in synced


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
