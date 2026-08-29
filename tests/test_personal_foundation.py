from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any

import pytest

from mini_arithmetic.apprenticeship import ApprenticeshipStore
from mini_arithmetic.personal_foundation import (
    PERSONAL_FOUNDATION_REPLAY_SCAN_LIMIT,
    PERSONAL_FOUNDATION_THRESHOLD,
    PersonalFoundationArtifact,
    PersonalFoundationCandidate,
    PersonalFoundationError,
    PersonalFoundationIdentity,
    PersonalFoundationSource,
    PersonalFoundationTrainingConfig,
    PersonalFoundationTrainingCoordinator,
    PersonalFoundationTrainingRequest,
    personal_foundation_backend_contract_sha256,
    personal_foundation_bundle_sha256,
    personal_foundation_profile_id,
)


BACKEND = "fake"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(
    path: Path,
    *,
    foundation_id: str = "foundation-base",
    weights: bytes = b"base weights",
) -> PersonalFoundationSource:
    path.mkdir(parents=True)
    (path / "model-config.json").write_bytes(b'{"width":8,"layers":1}')
    (path / "weights.safetensors").write_bytes(weights)
    tokenizer = path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_bytes(b'{"vocabulary":"fixed"}')
    identity = PersonalFoundationIdentity(
        backend=BACKEND,
        foundation_id=foundation_id,
        bundle_sha256=personal_foundation_bundle_sha256(path),
        weights_sha256=_sha256(path / "weights.safetensors"),
        config_sha256=_sha256(path / "model-config.json"),
        tokenizer_sha256=_sha256(tokenizer / "tokenizer.json"),
        backend_contract_sha256=personal_foundation_backend_contract_sha256(BACKEND),
    )
    return PersonalFoundationSource(path, identity)


class FakeFoundationRuntime:
    def __init__(self, source: PersonalFoundationSource) -> None:
        self.source = source
        self.activations: list[str] = []
        self.fail_next_candidate_activation = False

    def personal_foundation_source(self) -> PersonalFoundationSource:
        return self.source

    def prepare_personal_foundation(
        self,
        artifact: PersonalFoundationArtifact,
    ) -> object:
        assert artifact.bundle_path.is_dir()
        assert (
            personal_foundation_bundle_sha256(artifact.bundle_path)
            == artifact.manifest.identity.bundle_sha256
        )
        return PersonalFoundationSource(
            artifact.bundle_path,
            artifact.manifest.identity,
        )

    def activate_prepared_personal_foundation(
        self,
        prepared: object,
        identity: PersonalFoundationIdentity,
    ) -> None:
        assert isinstance(prepared, PersonalFoundationSource)
        assert prepared.identity == identity
        if (
            self.fail_next_candidate_activation
            and identity.foundation_id != "foundation-base"
        ):
            self.fail_next_candidate_activation = False
            self.source = prepared
            raise RuntimeError("simulated partial full-foundation activation")
        self.source = prepared
        self.activations.append(identity.foundation_id)

    @staticmethod
    def personal_training_target_fits(prompt: str, human_response: str) -> bool:
        return bool(prompt.strip()) and len(human_response) <= 256


class FakeFoundationTrainer:
    backend_name = BACKEND

    def __init__(self) -> None:
        self.requests: list[PersonalFoundationTrainingRequest] = []

    def train_foundation(
        self,
        request: PersonalFoundationTrainingRequest,
    ) -> PersonalFoundationCandidate:
        self.requests.append(request)
        output = request.work_dir / "candidate-bundle"
        shutil.copytree(request.parent.bundle_path, output)
        weights = output / "weights.safetensors"
        weights.write_bytes(
            weights.read_bytes() + request.candidate_foundation_id.encode()
        )
        identity = PersonalFoundationIdentity(
            backend=BACKEND,
            foundation_id=request.candidate_foundation_id,
            bundle_sha256=personal_foundation_bundle_sha256(output),
            weights_sha256=_sha256(weights),
            config_sha256=request.parent.identity.config_sha256,
            tokenizer_sha256=request.parent.identity.tokenizer_sha256,
            backend_contract_sha256=request.parent.identity.backend_contract_sha256,
        )
        return PersonalFoundationCandidate(
            bundle_path=output,
            identity=identity,
            parent_foundation_id=request.parent.identity.foundation_id,
            parent_bundle_sha256=request.parent.identity.bundle_sha256,
            # Deliberately worse monitoring telemetry proves quality is not a gate.
            metrics={
                "all_foundation_parameters_optimizer_eligible": True,
                "updates": 2,
                "baseline_monitoring_loss": 1.0,
                "candidate_monitoring_loss": 9.0,
                "monitoring_examples": 1,
                "monitoring_used_for_selection": False,
                "new_examples_optimized": 5,
                "optimizer_state_continuity": False,
                "optimizer_state_included": False,
                "optimizer_state_reset": True,
                "replay_examples": len(request.replay_examples),
                "tensor_scope_full_foundation": True,
            },
        )


class CancelAwareFoundationTrainer(FakeFoundationTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()

    def train_foundation(
        self,
        request: PersonalFoundationTrainingRequest,
    ) -> PersonalFoundationCandidate:
        self.requests.append(request)
        self.entered.set()
        assert request.cancellation_event is not None
        assert request.cancellation_event.wait(timeout=5)
        raise RuntimeError("cancelled by local shutdown")


class ContractLyingFoundationTrainer(FakeFoundationTrainer):
    def train_foundation(
        self,
        request: PersonalFoundationTrainingRequest,
    ) -> PersonalFoundationCandidate:
        candidate = super().train_foundation(request)
        return replace(
            candidate,
            metrics={**candidate.metrics, "optimizer_state_continuity": True},
        )


def _config(
    tmp_path: Path,
    *,
    replay: int = 3,
    backoff: float = 3_600,
) -> PersonalFoundationTrainingConfig:
    return PersonalFoundationTrainingConfig(
        artifact_root=tmp_path / "personal-foundations",
        maximum_replay_examples=replay,
        maximum_bundle_bytes=1024 * 1024,
        failure_backoff_seconds=backoff,
    )


def _add_dialogue_lessons(
    store: ApprenticeshipStore,
    count: int,
    *,
    session_id: str = "conversation",
    start: int = 0,
    responses: dict[int, str] | None = None,
    foundation_profile_id: str | None = None,
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
            metadata={"input": "text"},
        )
        lesson = store.add_trusted_dialogue_lesson(
            human_response_message_id=user.id,
            prompt_message_id=prior_assistant.id if prior_assistant else None,
            consent_local_training=True,
            metadata={
                "capture": "text-chat",
                "automatic": True,
                **(
                    {"foundation_profile_id": foundation_profile_id}
                    if foundation_profile_id is not None
                    else {}
                ),
            },
        )
        lesson_ids.append(lesson.id)
        prior_assistant = store.add_message(
            session_id,
            "assistant",
            f"tiny dog prompt {offset}",
            reply_to_id=user.id,
        )
    return lesson_ids


def _coordinator(
    tmp_path: Path,
    store: ApprenticeshipStore,
    *,
    source: PersonalFoundationSource | None = None,
    trainer: FakeFoundationTrainer | None = None,
    runtime: FakeFoundationRuntime | None = None,
    replay: int = 3,
    backoff: float = 3_600,
    recover_interrupted: bool = False,
) -> tuple[
    PersonalFoundationTrainingCoordinator,
    FakeFoundationTrainer,
    FakeFoundationRuntime,
    PersonalFoundationSource,
]:
    actual_source = source or _make_bundle(tmp_path / "source-bundle")
    actual_trainer = trainer or FakeFoundationTrainer()
    actual_runtime = runtime or FakeFoundationRuntime(actual_source)
    coordinator = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path, replay=replay, backoff=backoff),
        trainer=actual_trainer,
        runtime=actual_runtime,
        recover_interrupted=recover_interrupted,
    )
    return coordinator, actual_trainer, actual_runtime, actual_source


def _train_five(
    coordinator: PersonalFoundationTrainingCoordinator,
    store: ApprenticeshipStore,
    *,
    start: int,
) -> PersonalFoundationTrainingRequest:
    _add_dialogue_lessons(
        store,
        PERSONAL_FOUNDATION_THRESHOLD,
        start=start,
        foundation_profile_id=coordinator.profile_id,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle(timeout=10)
    trainer = coordinator.trainer
    assert isinstance(trainer, FakeFoundationTrainer)
    return trainer.requests[-1]


def test_bundle_hash_covers_every_entry_and_rejects_links(tmp_path: Path) -> None:
    source = _make_bundle(tmp_path / "bundle")
    first = source.identity.bundle_sha256
    (source.bundle_path / "tokenizer" / "extra.txt").write_text("new file")
    assert personal_foundation_bundle_sha256(source.bundle_path) != first
    if hasattr(os, "symlink"):
        os.symlink("weights.safetensors", source.bundle_path / "linked")
        with pytest.raises(PersonalFoundationError, match="symlink"):
            personal_foundation_bundle_sha256(source.bundle_path)


def test_every_five_typed_turns_train_the_full_foundation_and_always_promote(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, 4)
    coordinator, trainer, runtime, source = _coordinator(tmp_path, store)
    assert coordinator.consider_training() is False
    _add_dialogue_lessons(
        store,
        1,
        start=4,
        foundation_profile_id=coordinator.profile_id,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle(timeout=10)

    assert len(trainer.requests) == 1
    request = trainer.requests[0]
    assert len(request.new_examples) == 5
    assert len(request.monitoring_examples) == 1
    assert set(request.monitoring_examples) <= set(request.new_examples)
    assert request.replay_examples == ()
    assert runtime.source.identity.foundation_id == request.candidate_foundation_id
    assert runtime.source.identity.bundle_sha256 != source.identity.bundle_sha256
    run = store.get_training_status(request.run_id)
    assert run is not None and run.status == "promoted"
    assert (
        run.metrics["candidate_monitoring_loss"]
        > run.metrics["baseline_monitoring_loss"]
    )
    assert run.metadata["quality_gate_applied"] is False
    assert run.metadata["all_new_examples_optimized"] is True

    artifact = coordinator.registry.load(request.candidate_foundation_id)
    manifest_text = artifact.manifest_path.read_text()
    assert "human response" not in manifest_text
    assert "tiny dog prompt" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["raw_conversation_included"] is False
    assert manifest["uploads_enabled"] is False
    assert len(manifest["new_lesson_ids"]) == 5
    assert manifest["monitoring_lesson_ids"][0] in manifest["new_lesson_ids"]
    status = coordinator.status()
    assert status["parameter_scope"] == "full-foundation"
    assert status["monitoring_is_holdout"] is False
    assert status["monitoring_is_promotion_gate"] is False
    assert status["promotion_policy"] == "validated-training-always-promotes"
    assert status["uploads_enabled"] is False
    assert status["last_monitoring"] == {
        "baseline": 1.0,
        "candidate": 9.0,
        "delta": 8.0,
        "improved": False,
        "in_sample": True,
        "promotion_gate": False,
    }

    coordinator.shutdown()
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=FakeFoundationRuntime(source),
    )
    assert restarted.status()["last_monitoring"] == status["last_monitoring"]


def test_second_generation_uses_bounded_deterministic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, trainer, _, _ = _coordinator(tmp_path, store, replay=2)
    first = _train_five(coordinator, store, start=0)
    original = store.list_personal_foundation_used_dialogue_lessons
    calls: list[tuple[str, int | None, tuple[str, ...] | None, bool]] = []

    def bounded_history(
        profile_id: str,
        *,
        limit: int | None = None,
        run_statuses: tuple[str, ...] | None = None,
        newest_first: bool = False,
    ) -> list[Any]:
        calls.append((profile_id, limit, run_statuses, newest_first))
        return original(
            profile_id,
            limit=limit,
            run_statuses=run_statuses,
            newest_first=newest_first,
        )

    monkeypatch.setattr(
        store,
        "list_personal_foundation_used_dialogue_lessons",
        bounded_history,
    )
    second = _train_five(coordinator, store, start=5)

    assert second.parent.identity.foundation_id == first.candidate_foundation_id
    assert len(second.new_examples) == 5
    assert len(second.replay_examples) == 2
    assert set(item.lesson_id for item in second.new_examples).isdisjoint(
        item.lesson_id for item in second.replay_examples
    )
    assert len(trainer.requests) == 2
    assert calls == [
        (
            coordinator.profile_id,
            PERSONAL_FOUNDATION_REPLAY_SCAN_LIMIT,
            ("promoted",),
            True,
        )
    ]
    assert coordinator.status()["replay_scan_limit"] == 2_048

    history_item = original(
        coordinator.profile_id,
        limit=1,
        run_statuses=("promoted",),
    )[0]

    class CancelMidWindow(list[Any]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            for index, item in enumerate(super().__iter__()):
                if index == 64:
                    coordinator._shutdown.set()
                yield item

    monkeypatch.setattr(
        store,
        "list_personal_foundation_used_dialogue_lessons",
        lambda *args, **kwargs: CancelMidWindow([history_item] * 65),
    )
    with pytest.raises(PersonalFoundationError, match="replay selection was cancelled"):
        coordinator._select_replay(999, set())


def test_duplicate_typed_turn_is_quarantined_and_does_not_count(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, 1, session_id="first", start=0)
    _add_dialogue_lessons(store, 1, session_id="duplicate", start=0)
    _add_dialogue_lessons(store, 4, session_id="unique", start=1)
    coordinator, trainer, _, _ = _coordinator(tmp_path, store)

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle(timeout=10)
    assert len(trainer.requests) == 1
    assert len(trainer.requests[0].new_examples) == 5
    assert store.count_lessons(status="ineligible") == 1
    assert store.count_lessons(status="pending") == 0


def test_foundation_pending_lessons_are_strictly_profile_scoped(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    source_a = _make_bundle(tmp_path / "source-a", weights=b"profile a")
    coordinator_a, trainer_a, _, _ = _coordinator(
        tmp_path,
        store,
        source=source_a,
    )
    a_ids = _add_dialogue_lessons(
        store,
        4,
        session_id="profile-a",
        foundation_profile_id=coordinator_a.profile_id,
    )
    assert coordinator_a.consider_training() is False

    source_b = _make_bundle(tmp_path / "source-b", weights=b"profile b")
    coordinator_b, trainer_b, _, _ = _coordinator(
        tmp_path,
        store,
        source=source_b,
    )
    b_ids = _add_dialogue_lessons(
        store,
        1,
        session_id="profile-b",
        start=100,
        foundation_profile_id=coordinator_b.profile_id,
    )
    assert coordinator_b.profile_id != coordinator_a.profile_id
    assert coordinator_b.status()["qualifying_pending"] == 1
    assert coordinator_b.consider_training() is False
    assert trainer_b.requests == []
    assert all(store.get_lesson(identifier).status == "pending" for identifier in a_ids)

    b_ids.extend(
        _add_dialogue_lessons(
            store,
            4,
            session_id="profile-b",
            start=101,
            foundation_profile_id=coordinator_b.profile_id,
        )
    )
    assert coordinator_b.consider_training() is True
    assert coordinator_b.wait_for_idle(timeout=10)
    assert len(trainer_b.requests) == 1
    assert {item.lesson_id for item in trainer_b.requests[0].new_examples} == set(b_ids)
    assert all(store.get_lesson(identifier).status == "pending" for identifier in a_ids)
    assert trainer_a.requests == []


def test_first_v2_profile_atomically_adopts_legacy_unscoped_pending(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    legacy_ids = _add_dialogue_lessons(store, 4, session_id="legacy")
    coordinator_a, _, _, _ = _coordinator(tmp_path, store)
    assert coordinator_a.status()["legacy_unscoped_adopted_this_startup"] == 4
    assert all(
        store.get_lesson(identifier).metadata["foundation_profile_id"]
        == coordinator_a.profile_id
        for identifier in legacy_ids
    )

    source_b = _make_bundle(tmp_path / "source-b", weights=b"other base")
    coordinator_b, _, _, _ = _coordinator(tmp_path, store, source=source_b)
    assert coordinator_b.status()["legacy_unscoped_adopted_this_startup"] == 0
    assert coordinator_b.status()["qualifying_pending"] == 0


def test_candidate_must_attest_full_scope_and_optimizer_reset(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    trainer = ContractLyingFoundationTrainer()
    coordinator, _, runtime, source = _coordinator(
        tmp_path,
        store,
        trainer=trainer,
    )
    _train_five(coordinator, store, start=0)

    assert store.list_training_runs(limit=1)[0].status == "failed"
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 5
    assert runtime.source.identity == source.identity


def test_activation_failure_requeues_and_persists_backoff(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    source = _make_bundle(tmp_path / "source-bundle")
    runtime = FakeFoundationRuntime(source)
    runtime.fail_next_candidate_activation = True
    coordinator, trainer, runtime, _ = _coordinator(
        tmp_path,
        store,
        source=source,
        runtime=runtime,
    )
    request = _train_five(coordinator, store, start=0)

    run = store.get_training_status(request.run_id)
    assert run is not None and run.status == "rolled_back"
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 5
    assert runtime.source.identity.foundation_id == source.identity.foundation_id
    assert coordinator.status()["state"] == "backoff"
    assert coordinator.consider_training() is False

    coordinator.shutdown()
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=trainer,
        runtime=FakeFoundationRuntime(source),
    )
    assert restarted.status()["state"] == "backoff"
    assert restarted.consider_training() is False

    assert restarted.trigger(force=True) is True
    assert restarted.wait_for_idle(timeout=10)
    assert len(trainer.requests) == 2
    assert restarted.status()["state"] == "ready"

    restarted.shutdown()
    after_success = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=FakeFoundationRuntime(source),
    )
    assert after_success.status()["state"] != "backoff"


def test_promotion_state_commit_then_raise_is_recognized_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, trainer, runtime, source = _coordinator(tmp_path, store)
    original_write_state = coordinator.registry.write_state

    def commit_then_raise(**kwargs: Any) -> None:
        original_write_state(**kwargs)
        if kwargs["current"] != source.identity.foundation_id:
            raise OSError("simulated post-replace promotion error")

    monkeypatch.setattr(coordinator.registry, "write_state", commit_then_raise)
    request = _train_five(coordinator, store, start=0)

    run = store.get_training_status(request.run_id)
    assert run is not None and run.status == "promoted"
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 0
    assert runtime.source.identity.foundation_id == request.candidate_foundation_id
    assert coordinator.registry.read_state()[0] == request.candidate_foundation_id
    assert coordinator.status()["state"] == "ready"
    assert len(trainer.requests) == 1

    coordinator.shutdown()
    restarted_runtime = FakeFoundationRuntime(source)
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=restarted_runtime,
    )
    assert (
        restarted_runtime.source.identity.foundation_id
        == request.candidate_foundation_id
    )
    assert restarted.consider_training() is False


def test_failed_promotion_compensation_restores_runtime_even_if_ledger_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, runtime, source = _coordinator(tmp_path, store)
    original_write_state = coordinator.registry.write_state

    def fail_candidate_pointer(**kwargs: Any) -> None:
        if kwargs["current"] != source.identity.foundation_id:
            raise OSError("simulated state write failure")
        original_write_state(**kwargs)

    def fail_ledger_rollback(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated ledger rollback failure")

    monkeypatch.setattr(coordinator.registry, "write_state", fail_candidate_pointer)
    monkeypatch.setattr(store, "rollback_training_run", fail_ledger_rollback)
    request = _train_five(coordinator, store, start=0)

    durable_current = coordinator.registry.read_state()[0]
    assert durable_current == source.identity.foundation_id
    assert runtime.source.identity.foundation_id == durable_current
    assert coordinator.status()["state"] == "backoff"
    assert "ledger compensation" in (coordinator.status()["last_error"] or "")
    run = store.get_training_status(request.run_id)
    assert run is not None and run.status == "promoted"
    assert run.metadata["ledger_compensation_incomplete"] is True
    assert not (
        coordinator.registry.foundations / request.candidate_foundation_id
    ).exists()


def test_compensation_recognizes_rollback_that_commits_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, runtime, source = _coordinator(tmp_path, store)
    original_write_state = coordinator.registry.write_state
    original_rollback = store.rollback_training_run

    def fail_candidate_pointer(**kwargs: Any) -> None:
        if kwargs["current"] != source.identity.foundation_id:
            raise OSError("simulated pointer failure")
        original_write_state(**kwargs)

    def commit_rollback_then_raise(*args: Any, **kwargs: Any) -> None:
        original_rollback(*args, **kwargs)
        raise OSError("simulated post-commit ledger error")

    monkeypatch.setattr(coordinator.registry, "write_state", fail_candidate_pointer)
    monkeypatch.setattr(store, "rollback_training_run", commit_rollback_then_raise)
    request = _train_five(coordinator, store, start=0)

    run = store.get_training_status(request.run_id)
    assert run is not None and run.status == "rolled_back"
    assert all(
        store.get_lesson(lesson_id).status == "pending"
        and store.get_lesson(lesson_id).training_run_id is None
        for lesson_id in run.lesson_ids
    )
    assert runtime.source.identity == source.identity
    assert "ledger compensation" not in (coordinator.status()["last_error"] or "")


def test_compensation_recognizes_release_that_commits_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, runtime, source = _coordinator(tmp_path, store)
    original_release = store.release_training_run

    def fail_work_dir(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated work directory failure")

    def commit_release_then_raise(*args: Any, **kwargs: Any) -> None:
        original_release(*args, **kwargs)
        raise OSError("simulated post-commit ledger error")

    monkeypatch.setattr(coordinator.registry, "create_work_dir", fail_work_dir)
    monkeypatch.setattr(store, "release_training_run", commit_release_then_raise)
    _add_dialogue_lessons(
        store,
        PERSONAL_FOUNDATION_THRESHOLD,
        foundation_profile_id=coordinator.profile_id,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle(timeout=10)

    run = store.list_training_runs(limit=1)[0]
    assert run.status == "failed"
    assert all(
        store.get_lesson(lesson_id).status == "pending"
        and store.get_lesson(lesson_id).training_run_id is None
        for lesson_id in run.lesson_ids
    )
    assert runtime.source.identity == source.identity
    assert "ledger compensation" not in (coordinator.status()["last_error"] or "")


def test_shutdown_cooperatively_cancels_foundation_training(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, 5)
    source = _make_bundle(tmp_path / "source-bundle")
    trainer = CancelAwareFoundationTrainer()
    coordinator, _, runtime, _ = _coordinator(
        tmp_path,
        store,
        source=source,
        trainer=trainer,
    )
    assert coordinator.consider_training() is True
    assert trainer.entered.wait(timeout=5)

    coordinator.shutdown(timeout=5)
    assert coordinator.wait_for_idle(timeout=0)
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 5
    assert store.list_training_runs(limit=1)[0].status == "failed"
    assert runtime.source.identity == source.identity


def test_promotion_before_pointer_crash_requeues_exact_five_on_restart(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_dialogue_lessons(store, 5)
    coordinator, trainer, _, source = _coordinator(tmp_path, store)
    selected = coordinator._select_new_lessons()
    current, _, _, _ = coordinator.registry.read_state()
    run, claimed = store.claim_pending_lesson_ids(
        [item.id for item in selected],
        checkpoint_in=current,
        metadata={
            "authority": "personal-foundation-local",
            "foundation_profile_id": coordinator.profile_id,
            "parameter_scope": "full-foundation",
        },
    )
    assert run is not None and len(claimed) == 5
    candidate_id = "foundation-crash-candidate"
    work = coordinator.registry.create_work_dir(candidate_id)
    request = coordinator._request(
        run.id,
        candidate_id,
        coordinator.registry.load(current),
        claimed,
        work,
    )
    candidate = trainer.train_foundation(request)
    artifact = coordinator.registry.publish(request, candidate)
    store.complete_personal_foundation_run(
        run.id,
        [item.id for item in claimed],
        checkpoint_out=str(artifact.directory),
        metrics=dict(candidate.metrics),
        metadata={
            "authority": "personal-foundation-local",
            "foundation_profile_id": coordinator.profile_id,
            "foundation_id": candidate_id,
        },
    )
    coordinator.registry.discard_work_dir(work)
    coordinator.shutdown()

    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=FakeFoundationRuntime(source),
    )
    assert store.get_training_status(run.id).status == "rolled_back"
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 5
    assert not (restarted.registry.foundations / candidate_id).exists()
    assert restarted.trigger(force=True) is True
    assert restarted.wait_for_idle(timeout=10)


def test_three_promotions_two_rollbacks_return_to_base_without_resurrection(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, trainer, runtime, source = _coordinator(tmp_path, store)
    _train_five(coordinator, store, start=0)
    _train_five(coordinator, store, start=5)
    _train_five(coordinator, store, start=10)

    foundations = coordinator.registry.foundations
    assert (foundations / source.identity.foundation_id).is_dir()
    assert (
        len(
            [entry for entry in foundations.iterdir() if not entry.name.startswith(".")]
        )
        == 3
    )
    coordinator.rollback()
    coordinator.rollback()
    assert runtime.source.identity == source.identity
    assert (foundations / source.identity.foundation_id).is_dir()
    assert {run.status for run in store.list_training_runs(limit=10)} == {"rolled_back"}
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 0
    coordinator.shutdown()

    restarted_trainer = FakeFoundationTrainer()
    restarted_runtime = FakeFoundationRuntime(source)
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=restarted_trainer,
        runtime=restarted_runtime,
        recover_interrupted=True,
    )
    assert restarted_runtime.source.identity == source.identity
    assert restarted.consider_training() is False
    assert restarted_trainer.requests == []
    assert len(trainer.requests) == 3


def test_explicit_rollback_crash_marker_retires_without_requeue_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, runtime, source = _coordinator(tmp_path, store)
    request = _train_five(coordinator, store, start=0)

    def simulate_crash_after_pointer() -> None:
        raise RuntimeError("simulated crash after rollback pointer")

    monkeypatch.setattr(
        coordinator,
        "_reconcile_pending_retirement",
        simulate_crash_after_pointer,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        coordinator.rollback()
    assert runtime.source.identity == source.identity
    assert coordinator.registry.pending_retirement() == ("all", None)
    assert store.get_training_status(request.run_id).status == "promoted"
    coordinator.shutdown()

    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=FakeFoundationRuntime(source),
        recover_interrupted=True,
    )
    assert restarted.registry.pending_retirement() is None
    assert store.get_training_status(request.run_id).status == "rolled_back"
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 0
    assert restarted.consider_training() is False


def test_explicit_rollback_state_commit_then_raise_keeps_committed_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, runtime, source = _coordinator(tmp_path, store)
    request = _train_five(coordinator, store, start=0)
    original_write_state = coordinator.registry.write_state

    def commit_marker_then_raise(**kwargs: Any) -> None:
        original_write_state(**kwargs)
        if kwargs.get("pending_retirement") is not None:
            raise OSError("simulated post-replace rollback error")

    monkeypatch.setattr(
        coordinator.registry,
        "write_state",
        commit_marker_then_raise,
    )
    status = coordinator.rollback()

    assert runtime.source.identity == source.identity
    assert coordinator.registry.read_state()[0] == source.identity.foundation_id
    assert coordinator.registry.pending_retirement() is None
    assert store.get_training_status(request.run_id).status == "rolled_back"
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 0
    assert status["state"] == "rolled_back"

    coordinator.shutdown()
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=FakeFoundationRuntime(source),
        recover_interrupted=True,
    )
    assert restarted.status()["active_foundation_id"] == source.identity.foundation_id
    assert restarted.consider_training() is False


def test_retirement_commit_then_raise_is_verified_and_marker_is_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, runtime, source = _coordinator(tmp_path, store)
    request = _train_five(coordinator, store, start=0)
    original_retire = store.retire_promoted_training_runs

    def commit_retirement_then_raise(*args: Any, **kwargs: Any) -> None:
        original_retire(*args, **kwargs)
        raise OSError("simulated post-commit retirement error")

    monkeypatch.setattr(
        store,
        "retire_promoted_training_runs",
        commit_retirement_then_raise,
    )
    coordinator.rollback()

    assert runtime.source.identity == source.identity
    assert coordinator.registry.pending_retirement() is None
    assert store.get_training_status(request.run_id).status == "rolled_back"
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 0


def test_unresolved_retirement_marker_blocks_new_training_until_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, trainer, runtime, source = _coordinator(tmp_path, store)
    rejected = _train_five(coordinator, store, start=0)
    original_retire = store.retire_promoted_training_runs

    def fail_retirement(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated durable retirement failure")

    monkeypatch.setattr(
        store,
        "retire_promoted_training_runs",
        fail_retirement,
    )
    with pytest.raises(OSError, match="durable retirement failure"):
        coordinator.rollback()
    assert runtime.source.identity == source.identity
    assert coordinator.registry.pending_retirement() == ("all", None)
    assert store.get_training_status(rejected.run_id).status == "promoted"

    _add_dialogue_lessons(
        store,
        PERSONAL_FOUNDATION_THRESHOLD,
        session_id="after-rejection",
        start=100,
        foundation_profile_id=coordinator.profile_id,
    )
    assert coordinator.consider_training() is False
    assert coordinator.status()["recovery_pending"] is True
    assert len(trainer.requests) == 1

    monkeypatch.setattr(store, "retire_promoted_training_runs", original_retire)
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle(timeout=10)
    assert coordinator.registry.pending_retirement() is None
    assert store.get_training_status(rejected.run_id).status == "rolled_back"
    assert len(trainer.requests) == 2
    assert trainer.requests[-1].replay_examples == ()


def test_rollback_all_is_not_hidden_by_newer_other_profile_runs(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, runtime, source = _coordinator(tmp_path, store)
    request = _train_five(coordinator, store, start=0)
    other_profile = "foundation-other-profile"
    other_run_ids: list[int] = []
    for index in range(101):
        run = store.start_training_run(
            metadata={
                "authority": "personal-foundation-local",
                "foundation_profile_id": other_profile,
                "foundation_id": f"foundation-other-{index}",
            }
        )
        store.update_training_run(run.id, "promoted")
        other_run_ids.append(run.id)

    assert (
        store.list_training_runs(status="promoted", limit=100)[-1].id > request.run_id
    )
    coordinator.rollback()

    assert runtime.source.identity == source.identity
    assert store.get_training_status(request.run_id).status == "rolled_back"
    assert {
        run.id
        for run in store.list_personal_foundation_runs(
            other_profile,
            status="promoted",
            limit=200,
        )
    } == set(other_run_ids)


def test_missing_current_foundation_falls_back_and_requeues_uncommitted_branch(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, _, source = _coordinator(tmp_path, store)
    request = _train_five(coordinator, store, start=0)
    current = coordinator.registry.load(request.candidate_foundation_id)
    shutil.rmtree(current.directory)
    coordinator.shutdown()

    runtime = FakeFoundationRuntime(source)
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=runtime,
    )
    assert runtime.source.identity == source.identity
    assert store.get_training_status(request.run_id).status == "rolled_back"
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 5
    assert restarted.status()["active_foundation_id"] == source.identity.foundation_id


def test_missing_immutable_base_is_repaired_from_validated_initial_source(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, _, source = _coordinator(tmp_path, store)
    request = _train_five(coordinator, store, start=0)
    base = coordinator.registry.load(source.identity.foundation_id)
    shutil.rmtree(base.directory)
    coordinator.shutdown()

    runtime = FakeFoundationRuntime(source)
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=runtime,
    )
    assert restarted.registry.load(source.identity.foundation_id).manifest.run_id == 0
    assert runtime.source.identity.foundation_id == request.candidate_foundation_id
    assert restarted.status()["rollback_available"] is True


def test_missing_current_and_corrupt_previous_fall_back_without_old_resurrection(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, trainer, _, source = _coordinator(tmp_path, store)
    first = _train_five(coordinator, store, start=0)
    second = _train_five(coordinator, store, start=5)
    current, previous, _, _ = coordinator.registry.read_state()
    assert current == second.candidate_foundation_id
    assert previous == first.candidate_foundation_id
    current_artifact = coordinator.registry.load(current)
    previous_artifact = coordinator.registry.load(previous)
    shutil.rmtree(current_artifact.directory)
    previous_weights = previous_artifact.bundle_path / "weights.safetensors"
    previous_weights.write_bytes(previous_weights.read_bytes() + b"corrupt")
    coordinator.shutdown()

    restarted_trainer = FakeFoundationTrainer()
    restarted_runtime = FakeFoundationRuntime(source)
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=restarted_trainer,
        runtime=restarted_runtime,
    )
    assert restarted_runtime.source.identity == source.identity
    first_run = store.get_training_status(first.run_id)
    second_run = store.get_training_status(second.run_id)
    assert first_run is not None and first_run.status == "rolled_back"
    assert second_run is not None and second_run.status == "rolled_back"
    assert all(
        store.get_lesson(identifier).status == "used"
        for identifier in first_run.lesson_ids
    )
    assert all(
        store.get_lesson(identifier).status == "pending"
        for identifier in second_run.lesson_ids
    )
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 5
    assert not (restarted.registry.foundations / previous).exists()
    assert restarted.consider_training() is True
    assert restarted.wait_for_idle(timeout=10)
    assert len(restarted_trainer.requests) == 1
    assert {
        item.lesson_id for item in restarted_trainer.requests[0].new_examples
    } == set(second_run.lesson_ids)
    assert len(trainer.requests) == 2


def test_valid_current_sanitizes_corrupt_previous_then_rolls_back_to_base(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, _, source = _coordinator(tmp_path, store)
    first = _train_five(coordinator, store, start=0)
    second = _train_five(coordinator, store, start=5)
    previous = coordinator.registry.load(first.candidate_foundation_id)
    weights = previous.bundle_path / "weights.safetensors"
    weights.write_bytes(weights.read_bytes() + b"corrupt retained previous")
    coordinator.shutdown()

    runtime = FakeFoundationRuntime(source)
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=runtime,
    )
    current, previous_id, _, _ = restarted.registry.read_state()
    assert current == second.candidate_foundation_id
    assert previous_id is None
    assert runtime.source.identity.foundation_id == second.candidate_foundation_id
    assert not (restarted.registry.foundations / first.candidate_foundation_id).exists()

    restarted.rollback()
    assert runtime.source.identity == source.identity
    assert all(
        run.status == "rolled_back" for run in store.list_training_runs(limit=10)
    )
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 0
    restarted.shutdown()

    final_trainer = FakeFoundationTrainer()
    final = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=final_trainer,
        runtime=FakeFoundationRuntime(source),
        recover_interrupted=True,
    )
    assert final.status()["active_foundation_id"] == source.identity.foundation_id
    assert final.consider_training() is False
    assert final_trainer.requests == []


@pytest.mark.parametrize("invalid_ledger", ["rolled_back", "missing", "mismatched"])
def test_valid_previous_requires_exact_promoted_ledger_linkage(
    tmp_path: Path,
    invalid_ledger: str,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, _, source = _coordinator(tmp_path, store)
    first = _train_five(coordinator, store, start=0)
    second = _train_five(coordinator, store, start=5)
    previous = coordinator.registry.load(first.candidate_foundation_id)
    if invalid_ledger == "rolled_back":
        store.rollback_training_run(first.run_id)
    else:
        manifest = json.loads(previous.manifest_path.read_text())
        manifest["run_id"] = 999_999 if invalid_ledger == "missing" else second.run_id
        previous.manifest_path.write_text(json.dumps(manifest))
    coordinator.shutdown()

    runtime = FakeFoundationRuntime(source)
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=runtime,
    )
    current, previous_id, _, _ = restarted.registry.read_state()
    assert current == second.candidate_foundation_id
    assert previous_id is None
    assert runtime.source.identity.foundation_id == second.candidate_foundation_id
    assert not (restarted.registry.foundations / first.candidate_foundation_id).exists()
    if invalid_ledger == "rolled_back":
        assert all(
            store.get_lesson(example.lesson_id).status == "ineligible"
            for example in first.new_examples
        )

    restarted.rollback()
    assert runtime.source.identity == source.identity
    assert store.count_trusted_typed_dialogue_lessons(status="pending") == 0
    restarted.shutdown()

    final = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=FakeFoundationRuntime(source),
        recover_interrupted=True,
    )
    assert final.consider_training() is False


def test_missing_current_does_not_activate_a_rolled_back_previous(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    coordinator, _, _, source = _coordinator(tmp_path, store)
    first = _train_five(coordinator, store, start=0)
    second = _train_five(coordinator, store, start=5)
    store.rollback_training_run(first.run_id)
    shutil.rmtree(coordinator.registry.load(second.candidate_foundation_id).directory)
    coordinator.shutdown()

    runtime = FakeFoundationRuntime(source)
    restarted = PersonalFoundationTrainingCoordinator(
        store,
        _config(tmp_path),
        trainer=FakeFoundationTrainer(),
        runtime=runtime,
    )
    assert runtime.source.identity == source.identity
    assert restarted.registry.read_state()[0] == source.identity.foundation_id
    assert all(
        store.get_lesson(example.lesson_id).status == "ineligible"
        for example in first.new_examples
    )
    assert all(
        store.get_lesson(example.lesson_id).status == "pending"
        for example in second.new_examples
    )


def test_profile_identity_is_anchored_to_the_complete_base_bundle(
    tmp_path: Path,
) -> None:
    first = _make_bundle(tmp_path / "first", weights=b"first")
    second = _make_bundle(tmp_path / "second", weights=b"second")
    assert first.identity.config_sha256 == second.identity.config_sha256
    assert first.identity.tokenizer_sha256 == second.identity.tokenizer_sha256
    assert personal_foundation_profile_id(
        first.identity
    ) != personal_foundation_profile_id(second.identity)


def test_foundation_config_from_env_has_safe_explicit_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in tuple(os.environ):
        if key.startswith("MLM_PERSONAL_FOUNDATION_"):
            monkeypatch.delenv(key)
    default = PersonalFoundationTrainingConfig.from_env(project_root=tmp_path)
    assert default.enabled is True
    assert default.artifact_root == (tmp_path / "data/personal-foundations").resolve()
    assert default.maximum_replay_examples == 8
    assert default.failure_backoff_seconds == 300

    monkeypatch.setenv("MLM_PERSONAL_FOUNDATION_AUTO", "off")
    monkeypatch.setenv("MLM_PERSONAL_FOUNDATION_DIR", "private/foundations")
    monkeypatch.setenv("MLM_PERSONAL_FOUNDATION_MAX_REPLAY_EXAMPLES", "2")
    monkeypatch.setenv("MLM_PERSONAL_FOUNDATION_MAX_BUNDLE_BYTES", "123456")
    monkeypatch.setenv("MLM_PERSONAL_FOUNDATION_FAILURE_BACKOFF_SECONDS", "45.5")
    monkeypatch.setenv("MLM_PERSONAL_FOUNDATION_PROFILE_ID", "private-profile")
    configured = PersonalFoundationTrainingConfig.from_env(project_root=tmp_path)
    assert configured == PersonalFoundationTrainingConfig(
        enabled=False,
        artifact_root=(tmp_path / "private/foundations").resolve(),
        maximum_replay_examples=2,
        maximum_bundle_bytes=123456,
        failure_backoff_seconds=45.5,
        profile_id="private-profile",
    )


def test_foundation_config_rejects_ambiguous_boolean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLM_PERSONAL_FOUNDATION_AUTO", "sometimes")
    with pytest.raises(ValueError, match="must be true or false"):
        PersonalFoundationTrainingConfig.from_env(project_root=tmp_path)


def test_exact_foundation_completion_rejects_a_changed_claim(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_dialogue_lessons(store, 5)
    run, claimed = store.claim_pending_lesson_ids(
        lesson_ids,
        checkpoint_in="foundation-base",
        metadata={"authority": "personal-foundation-local"},
    )
    assert run is not None and len(claimed) == 5
    with pytest.raises(ValueError, match="differs from its exact claim"):
        store.complete_personal_foundation_run(
            run.id,
            list(reversed(lesson_ids)),
            checkpoint_out="candidate",
            metrics={"loss": 1.0},
            metadata={"foundation_profile_id": "profile"},
        )
    assert store.get_training_status(run.id).status == "training"
