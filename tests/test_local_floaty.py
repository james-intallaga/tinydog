from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from mini_arithmetic.apprenticeship import ApprenticeshipStore
from mini_arithmetic.community_updates import (
    CommunityContract,
    TensorSpec,
    write_update_bundle,
)
from mini_arithmetic.local_floaty import (
    LOCAL_FLOATY_THRESHOLD,
    LocalFloatyAutoConfig,
    LocalFloatyCoordinator,
    LocalFloatyRunRequest,
    LocalFloatyRunResult,
)
from mini_arithmetic.floaty_training import FloatyTrainingConfig


def _configured_local_floaty(tmp_path: Path) -> LocalFloatyAutoConfig:
    foundation_manifest = tmp_path / "foundation-manifest.json"
    owner_public_key = tmp_path / "owner.pub"
    community_release_manifest = tmp_path / "community-release.json"
    foundation_manifest.write_text("{}\n", encoding="utf-8")
    owner_public_key.write_text("test-public-key\n", encoding="utf-8")
    community_release_manifest.write_text("{}\n", encoding="utf-8")
    return LocalFloatyAutoConfig(
        enabled=True,
        foundation_manifest=foundation_manifest,
        owner_public_key=owner_public_key,
        community_release_manifest=community_release_manifest,
        trust_state_path=tmp_path / "trust.json",
        candidate_output_dir=tmp_path / "candidates",
        bundle_output_dir=tmp_path / "outgoing",
        lesson_mode="verified",
    )


def _contract() -> CommunityContract:
    return CommunityContract(
        foundation_sha256="1" * 64,
        community_version="community-test-1",
        community_weights_sha256="2" * 64,
        architecture_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        training_recipe_sha256="5" * 64,
        tensor_specs=(TensorSpec("floaty.test.weight", (1,), "F32"),),
    )


def _add_lessons(
    store: ApprenticeshipStore,
    count: int,
    *,
    start: int = 1,
) -> list[int]:
    lessons = []
    for value in range(start, start + count):
        lessons.append(
            store.add_verified_lesson(
                problem=f"{value} + {value + 1}",
                first_attempt="0",
                correct_answer=str(value + value + 1),
                feedback="Use exact arithmetic.",
                source="human-teacher",
            ).id
        )
    return lessons


def _accepted_result(request: LocalFloatyRunRequest) -> LocalFloatyRunResult:
    write_update_bundle(
        request.bundle_path,
        contract=_contract(),
        update_id=request.update_id,
        deltas={"floaty.test.weight": np.array([0.25], dtype=np.float32)},
        example_count=4,
    )
    report = request.bundle_path.parent / f"{request.update_id}-candidate.json"
    report.write_text("{}\n", encoding="utf-8")
    return LocalFloatyRunResult(
        accepted=True,
        candidate_report=report,
        bundle_path=request.bundle_path.resolve(),
        training_lesson_ids=request.lesson_ids[:4],
        validation_lesson_ids=request.lesson_ids[4:],
    )


def _rejected_result(request: LocalFloatyRunRequest) -> LocalFloatyRunResult:
    report = request.bundle_path.parent / f"{request.update_id}-candidate.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}\n", encoding="utf-8")
    return LocalFloatyRunResult(
        accepted=False,
        candidate_report=report,
        bundle_path=None,
        training_lesson_ids=request.lesson_ids[:4],
        validation_lesson_ids=request.lesson_ids[4:],
    )


def test_four_distinct_lessons_do_not_trigger_local_training(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_lessons(store, LOCAL_FLOATY_THRESHOLD - 1)
    calls: list[LocalFloatyRunRequest] = []
    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=lambda request: calls.append(request) or _accepted_result(request),
    )

    assert coordinator.consider_training() is False
    assert coordinator.wait_for_idle()
    assert calls == []
    assert store.count_pending_lessons() == 4
    assert store.list_training_runs() == []
    assert coordinator.status()["qualifying_pending"] == 4


def test_fifth_distinct_lesson_triggers_once_and_consumes_complete_bundle(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_lessons(store, LOCAL_FLOATY_THRESHOLD - 1)
    calls: list[LocalFloatyRunRequest] = []
    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=lambda request: calls.append(request) or _accepted_result(request),
    )
    assert coordinator.consider_training() is False

    _add_lessons(store, 1, start=20)
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert len(calls) == 1
    assert len(calls[0].lesson_ids) == LOCAL_FLOATY_THRESHOLD
    assert calls[0].bundle_path.is_dir()
    assert store.count_pending_lessons() == 0
    assert all(
        store.get_lesson(identifier).status == "used"
        for identifier in calls[0].lesson_ids
    )
    runs = store.list_training_runs()
    assert len(runs) == 1
    assert runs[0].status == "contributed"
    status = coordinator.status()
    assert status["state"] == "ready"
    assert status["last_bundle"] == calls[0].bundle_path.name
    assert status["live_model_updated"] is False


def test_duplicate_canonical_problem_and_answer_do_not_satisfy_threshold(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_lessons(store, LOCAL_FLOATY_THRESHOLD - 1)
    duplicate = store.add_verified_lesson(
        problem="1 + 2",
        first_attempt="2",
        correct_answer="3.0",
        feedback="Three is the exact answer.",
        source="human-teacher",
    )
    calls: list[LocalFloatyRunRequest] = []
    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=lambda request: calls.append(request) or _accepted_result(request),
    )

    assert store.count_pending_lessons() == LOCAL_FLOATY_THRESHOLD
    assert duplicate.correct_answer == "3"
    assert coordinator.consider_training() is False
    assert calls == []
    assert coordinator.status()["qualifying_pending"] == LOCAL_FLOATY_THRESHOLD - 1
    assert store.list_training_runs() == []


def test_concurrent_consider_calls_invoke_one_runner(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_lessons(store, LOCAL_FLOATY_THRESHOLD)
    entered = threading.Event()
    release = threading.Event()
    calls: list[LocalFloatyRunRequest] = []
    calls_lock = threading.Lock()

    def runner(request: LocalFloatyRunRequest) -> LocalFloatyRunResult:
        with calls_lock:
            calls.append(request)
        entered.set()
        assert release.wait(timeout=5)
        return _accepted_result(request)

    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=runner,
    )
    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(
            executor.map(lambda _: coordinator.consider_training(), range(12))
        )

    assert entered.wait(timeout=5)
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 11
    release.set()
    assert coordinator.wait_for_idle()
    assert len(calls) == 1
    assert store.list_training_runs()[0].status == "contributed"


def test_clean_rejection_consumes_batch_without_looping(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_lessons(store, LOCAL_FLOATY_THRESHOLD)
    calls: list[LocalFloatyRunRequest] = []
    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=lambda request: calls.append(request) or _rejected_result(request),
    )

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert len(calls) == 1
    assert store.count_pending_lessons() == 0
    assert all(
        store.get_lesson(identifier).status == "used" for identifier in lesson_ids
    )
    assert store.list_training_runs()[0].status == "local_rejected"
    assert coordinator.status()["state"] == "rejected"
    assert coordinator.consider_training() is False
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["runner-error", "missing-bundle"])
def test_training_failure_requeues_complete_batch_and_reports_failed(
    tmp_path: Path,
    failure: str,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_lessons(store, LOCAL_FLOATY_THRESHOLD)
    calls: list[LocalFloatyRunRequest] = []

    def runner(request: LocalFloatyRunRequest) -> LocalFloatyRunResult:
        calls.append(request)
        if failure == "runner-error":
            raise RuntimeError("trainer process failed")
        report = request.bundle_path.parent / "candidate.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}\n", encoding="utf-8")
        return LocalFloatyRunResult(
            accepted=True,
            candidate_report=report,
            bundle_path=request.bundle_path.resolve(),
            training_lesson_ids=request.lesson_ids[:4],
            validation_lesson_ids=request.lesson_ids[4:],
        )

    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=runner,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert len(calls) == 1
    assert store.count_pending_lessons() == LOCAL_FLOATY_THRESHOLD
    assert all(
        store.get_lesson(identifier).status == "pending" for identifier in lesson_ids
    )
    run = store.list_training_runs()[0]
    assert run.status == "failed"
    assert run.error
    status = coordinator.status()
    assert status["state"] == "failed"
    assert status["last_error"]


def test_incomplete_train_holdout_partition_fails_and_requeues_batch(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_lessons(store, LOCAL_FLOATY_THRESHOLD)
    calls: list[LocalFloatyRunRequest] = []

    def incomplete_runner(request: LocalFloatyRunRequest) -> LocalFloatyRunResult:
        calls.append(request)
        report = request.bundle_path.parent / "candidate.json"
        return LocalFloatyRunResult(
            accepted=False,
            candidate_report=report,
            bundle_path=None,
            training_lesson_ids=request.lesson_ids[:3],
            validation_lesson_ids=request.lesson_ids[4:],
        )

    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=incomplete_runner,
    )
    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert len(calls) == 1
    assert store.count_pending_lessons() == LOCAL_FLOATY_THRESHOLD
    assert all(
        store.get_lesson(identifier).status == "pending" for identifier in lesson_ids
    )
    run = store.list_training_runs()[0]
    assert run.status == "failed"
    assert "complete lesson batch" in (run.error or "")
    assert coordinator.status()["state"] == "failed"


def test_interrupted_exact_floaty_claim_is_recovered_then_retrained(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_lessons(store, LOCAL_FLOATY_THRESHOLD)
    interrupted, claimed = store.claim_pending_lesson_ids(
        lesson_ids,
        metadata={
            "parameter_scope": "floaty-only",
            "authority": "community",
        },
    )
    assert interrupted is not None
    assert [lesson.id for lesson in claimed] == lesson_ids
    assert store.count_pending_lessons() == 0

    calls: list[LocalFloatyRunRequest] = []
    config = _configured_local_floaty(tmp_path)
    coordinator = LocalFloatyCoordinator(
        store,
        config,
        runner=lambda request: calls.append(request) or _accepted_result(request),
    )

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert store.path.with_name(f".{store.path.name}.local-floaty.lock").is_file()
    assert len(calls) == 1
    assert calls[0].run_id != interrupted.id
    assert set(calls[0].lesson_ids) == set(lesson_ids)
    assert store.count_pending_lessons() == 0
    assert all(
        store.get_lesson(identifier).status == "used" for identifier in lesson_ids
    )
    recovered = store.get_training_status(interrupted.id)
    assert recovered is not None
    assert recovered.status == "failed"
    assert recovered.metadata["recovered"] is True
    assert recovered.error == "Recovered an interrupted local floaty training run"
    replacement = store.get_training_status(calls[0].run_id)
    assert replacement is not None
    assert replacement.status == "contributed"


def test_published_orphan_bundle_is_reconciled_without_retraining(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_lessons(store, LOCAL_FLOATY_THRESHOLD)
    update_id = "local-orphan-bundle"
    interrupted, claimed = store.claim_pending_lesson_ids(
        lesson_ids,
        metadata={
            "parameter_scope": "floaty-only",
            "authority": "community",
            "update_id": update_id,
        },
    )
    assert interrupted is not None
    assert len(claimed) == LOCAL_FLOATY_THRESHOLD
    config = _configured_local_floaty(tmp_path)
    bundle_path = config.bundle_output_dir / update_id
    write_update_bundle(
        bundle_path,
        contract=_contract(),
        update_id=update_id,
        deltas={"floaty.test.weight": np.array([0.25], dtype=np.float32)},
        example_count=4,
    )
    calls: list[LocalFloatyRunRequest] = []
    coordinator = LocalFloatyCoordinator(
        store,
        config,
        runner=lambda request: calls.append(request) or _accepted_result(request),
    )

    assert coordinator.consider_training() is True
    assert coordinator.wait_for_idle()

    assert calls == []
    recovered = store.get_training_status(interrupted.id)
    assert recovered is not None
    assert recovered.status == "contributed"
    assert recovered.metadata["recovered_orphan_bundle"] is True
    assert all(
        store.get_lesson(identifier).status == "used" for identifier in lesson_ids
    )
    assert coordinator.status()["last_bundle"] == update_id


def test_lessons_arriving_during_first_run_form_second_disjoint_batch(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    first_ids = set(_add_lessons(store, LOCAL_FLOATY_THRESHOLD))
    first_entered = threading.Event()
    first_release = threading.Event()
    calls: list[LocalFloatyRunRequest] = []
    calls_lock = threading.Lock()

    def runner(request: LocalFloatyRunRequest) -> LocalFloatyRunResult:
        with calls_lock:
            calls.append(request)
            call_number = len(calls)
        if call_number == 1:
            first_entered.set()
            assert first_release.wait(timeout=5)
        return _accepted_result(request)

    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=runner,
    )
    assert coordinator.consider_training() is True
    assert first_entered.wait(timeout=5)

    second_ids = set(_add_lessons(store, LOCAL_FLOATY_THRESHOLD, start=20))
    first_release.set()
    assert coordinator.wait_for_idle()

    assert len(calls) == 2
    claimed = [set(request.lesson_ids) for request in calls]
    assert claimed[0] == first_ids
    assert claimed[1] == second_ids
    assert claimed[0].isdisjoint(claimed[1])
    assert store.count_pending_lessons() == 0
    assert [run.status for run in store.list_training_runs()] == [
        "contributed",
        "contributed",
    ]


def test_closeout_arrival_hands_off_to_exactly_one_new_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    first_ids = set(_add_lessons(store, LOCAL_FLOATY_THRESHOLD))
    calls: list[LocalFloatyRunRequest] = []
    closeout_scan = threading.Event()
    release_closeout = threading.Event()
    blocked_once = threading.Event()
    coordinator = LocalFloatyCoordinator(
        store,
        _configured_local_floaty(tmp_path),
        runner=lambda request: calls.append(request) or _accepted_result(request),
    )
    original_select = coordinator._select_distinct_pending

    def pause_after_final_insufficient_scan(
        *,
        limit: int | None,
    ) -> list[object]:
        selected = original_select(limit=limit)
        if (
            threading.current_thread().name == "mlm-local-floaty"
            and len(calls) == 1
            and len(selected) < LOCAL_FLOATY_THRESHOLD
            and not blocked_once.is_set()
        ):
            blocked_once.set()
            closeout_scan.set()
            assert release_closeout.wait(timeout=5)
        return selected

    monkeypatch.setattr(
        coordinator,
        "_select_distinct_pending",
        pause_after_final_insufficient_scan,
    )
    assert coordinator.consider_training() is True
    assert closeout_scan.wait(timeout=5)

    second_ids = set(_add_lessons(store, LOCAL_FLOATY_THRESHOLD, start=20))
    # The original worker has already observed an insufficient queue but has
    # not reached its closeout handoff yet.
    assert coordinator.status()["running"] is True
    assert coordinator.consider_training() is False
    release_closeout.set()
    assert coordinator.wait_for_idle()

    assert len(calls) == 2
    assert set(calls[0].lesson_ids) == first_ids
    assert set(calls[1].lesson_ids) == second_ids
    assert calls[0].run_id != calls[1].run_id
    assert store.count_pending_lessons() == 0
    assert len(store.list_training_runs(status="contributed")) == 2


def test_missing_signed_settings_blocks_without_claiming_lessons(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    lesson_ids = _add_lessons(store, LOCAL_FLOATY_THRESHOLD)
    calls: list[LocalFloatyRunRequest] = []
    config = LocalFloatyAutoConfig(
        enabled=True,
        foundation_manifest=None,
        owner_public_key=tmp_path / "missing-owner.pub",
        community_release_manifest=None,
        trust_state_path=tmp_path / "trust.json",
        candidate_output_dir=tmp_path / "candidates",
        bundle_output_dir=tmp_path / "outgoing",
    )
    coordinator = LocalFloatyCoordinator(
        store,
        config,
        runner=lambda request: calls.append(request) or _accepted_result(request),
    )

    assert coordinator.consider_training() is False
    assert calls == []
    assert store.list_training_runs() == []
    assert store.count_pending_lessons() == LOCAL_FLOATY_THRESHOLD
    assert all(
        store.get_lesson(identifier).status == "pending" for identifier in lesson_ids
    )
    status = coordinator.status()
    assert status["state"] == "blocked"
    assert set(status["missing_settings"]) == {
        "MLM_LOCAL_FOUNDATION_MANIFEST",
        "MLM_LOCAL_OWNER_PUBLIC_KEY",
        "MLM_LOCAL_COMMUNITY_RELEASE",
    }


def test_three_train_two_holdout_recipe_is_blocked_before_claim(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "lessons.sqlite3")
    _add_lessons(store, LOCAL_FLOATY_THRESHOLD)
    recipe = FloatyTrainingConfig(
        minimum_training_lessons=3,
        minimum_validation_lessons=2,
    )
    recipe_path = tmp_path / "floaty-training.json"
    recipe_path.write_text(
        json.dumps(asdict(recipe), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    base = _configured_local_floaty(tmp_path)
    config = LocalFloatyAutoConfig(
        enabled=base.enabled,
        foundation_manifest=base.foundation_manifest,
        owner_public_key=base.owner_public_key,
        community_release_manifest=base.community_release_manifest,
        trust_state_path=base.trust_state_path,
        candidate_output_dir=base.candidate_output_dir,
        bundle_output_dir=base.bundle_output_dir,
        training_config_path=recipe_path,
    )
    calls: list[LocalFloatyRunRequest] = []
    coordinator = LocalFloatyCoordinator(
        store,
        config,
        runner=lambda request: calls.append(request) or _accepted_result(request),
    )

    assert coordinator.consider_training() is False
    assert calls == []
    assert store.count_pending_lessons() == LOCAL_FLOATY_THRESHOLD
    assert coordinator.status()["state"] == "blocked"
