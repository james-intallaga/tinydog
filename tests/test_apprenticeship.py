from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
import os
from pathlib import Path
import threading

import pytest

from mini_arithmetic.apprenticeship import (
    ApprenticeshipStore,
    TRUSTED_DIALOGUE_BOOTSTRAP_PROMPT,
    TRUSTED_HUMAN_DIALOGUE_SKILL,
    TRUSTED_HUMAN_DIALOGUE_SOURCE,
    TRUSTED_HUMAN_DIALOGUE_VERIFIER,
    canonical_practice_key,
    evaluate_arithmetic,
    generate_practice_variants,
    parse_correction,
    record_verified_correction,
    verify_answer,
)
from mini_arithmetic.data import split_for_key
from mini_arithmetic.teacher import TeacherClient


def _add_local_training_lessons(
    store: ApprenticeshipStore,
    count: int = 5,
) -> list[int]:
    return [
        store.add_verified_lesson(
            problem=f"{index} + {index + 1}",
            first_attempt="0",
            correct_answer=str(index + index + 1),
            feedback="Use exact arithmetic.",
        ).id
        for index in range(1, count + 1)
    ]


def test_exact_arithmetic_is_restricted_and_verified() -> None:
    assert evaluate_arithmetic("What is 37 + 28?") == Fraction(65)
    assert evaluate_arithmetic("50% of 84") == Fraction(42)
    assert evaluate_arithmetic("Add and simplify: 1/2 + 3/4.") == Fraction(5, 4)
    assert verify_answer("Calculate (2 + 3) * 4.", "The answer is 20.").is_correct
    wrong = verify_answer("37 + 28", "55")
    assert not wrong.is_correct
    assert wrong.expected == "65"
    with pytest.raises(ValueError):
        evaluate_arithmetic("__import__('os').system('echo unsafe')")


def test_store_persists_dialogue_lessons_practice_and_runs(tmp_path) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    session = store.create_session("session-one")
    problem = store.add_message(session.id, "user", "What is 37 + 28?")
    attempt = store.add_message(session.id, "assistant", "55", reply_to_id=problem.id)
    updated_attempt = store.update_message_metadata(
        attempt.id,
        {
            "verification": {"status": "incorrect", "expected": "65"},
            "lesson_saved": True,
        },
    )
    assert updated_attempt.metadata["verification"]["expected"] == "65"
    assert store.list_messages(session.id)[-1].metadata["lesson_saved"] is True
    correction = store.add_message(
        session.id,
        "teacher",
        "You forgot to carry. The answer is 65.",
        reply_to_id=attempt.id,
    )
    lesson = store.add_verified_lesson(
        problem=problem.content,
        first_attempt=attempt.content,
        correct_answer="65",
        feedback=correction.content,
        session_id=session.id,
        problem_message_id=problem.id,
        assistant_message_id=attempt.id,
        correction_message_id=correction.id,
    )
    assert lesson.status == "pending"
    enriched = store.add_verified_lesson(
        problem=problem.content,
        first_attempt=attempt.content,
        correct_answer="65",
        feedback="Human explanation: carry the tens.",
        source="human-teacher",
        session_id=session.id,
        problem_message_id=problem.id,
        assistant_message_id=attempt.id,
        correction_message_id=correction.id,
    )
    assert enriched.id == lesson.id
    assert enriched.source == "human-teacher"
    assert enriched.feedback == "Human explanation: carry the tens."
    assert enriched.correction_message_id == correction.id
    assert store.count_pending_lessons() == 1
    assert store.find_preceding_problem(attempt.id) == problem
    variants = generate_practice_variants(problem.content, count=3, seed=2)
    assert len(variants) == 3
    for variant, answer in variants:
        store.add_practice(lesson.id, variant, answer)
    assert len(store.list_practice(lesson.id)) == 3
    assert store.search_relevant_lessons("addition carrying")[0].id == lesson.id
    assert store.search_relevant_lessons("hello, how are you?") == []
    assert store.search_relevant_lessons("What is your name?") == []
    run, claimed = store.claim_pending_lessons(
        limit=10, checkpoint_in="active.safetensors"
    )
    assert run is not None
    assert [item.id for item in claimed] == [lesson.id]
    assert store.count_pending_lessons() == 0
    store.promote_training_run(
        run.id,
        [lesson.id],
        checkpoint_out="candidate.safetensors",
        metrics={"new_loss": 1.0},
    )
    assert store.count_pending_lessons() == 0
    assert store.get_training_status().status == "promoted"
    rolled_back = store.rollback_training_run(run.id)
    assert rolled_back.status == "rolled_back"
    assert store.count_pending_lessons() == 1
    retry_run, retry_lessons = store.claim_pending_lessons(
        limit=10, checkpoint_in="active.safetensors"
    )
    assert retry_run is not None and [item.id for item in retry_lessons] == [lesson.id]
    store.release_training_run(
        retry_run.id, "rejected", error="gate rejected candidate"
    )
    assert store.count_pending_lessons() == 1
    assert store.get_training_status().status == "rejected"


def test_personal_status_indexes_and_bulk_counts_are_present(tmp_path) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    _add_local_training_lessons(store, count=3)
    with store._connect() as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }

    assert {
        "lessons_dialogue_provenance",
        "lessons_training_status",
        "training_runs_status_id",
    } <= indexes
    assert store.count_lessons() == 3
    assert store.count_lessons(status="pending") == 3


def test_exact_lesson_claim_is_all_or_nothing_under_contention(tmp_path) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    lesson_ids = _add_local_training_lessons(store)

    missing_run, missing_claim = store.claim_pending_lesson_ids(
        [lesson_ids[0], max(lesson_ids) + 1000]
    )
    assert missing_run is None
    assert missing_claim == []
    assert store.count_pending_lessons() == len(lesson_ids)
    assert store.list_training_runs() == []

    barrier = threading.Barrier(2)

    def claim() -> tuple[object | None, list[object]]:
        contender = ApprenticeshipStore(store.path)
        barrier.wait()
        return contender.claim_pending_lesson_ids(lesson_ids)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: claim(), range(2)))

    winners = [(run, lessons) for run, lessons in outcomes if run is not None]
    assert len(winners) == 1
    winner_run, winner_lessons = winners[0]
    assert [lesson.id for lesson in winner_lessons] == lesson_ids
    assert winner_run.lesson_ids == lesson_ids
    assert sum(run is None and lessons == [] for run, lessons in outcomes) == 1
    assert store.count_pending_lessons() == 0


def test_local_floaty_completion_consumes_only_the_exact_claimed_batch(
    tmp_path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    lesson_ids = _add_local_training_lessons(store, count=6)
    run, claimed = store.claim_pending_lesson_ids(
        lesson_ids[:5],
        checkpoint_in="community-release.json",
        metadata={"authority": "local-floaty"},
    )
    assert run is not None
    assert [lesson.id for lesson in claimed] == lesson_ids[:5]

    completed = store.complete_local_floaty_run(
        run.id,
        lesson_ids[:5],
        status="contributed",
        bundle_out="outgoing/local-five",
        metrics={"accepted": True},
    )

    assert completed.status == "contributed"
    assert completed.checkpoint_out == "outgoing/local-five"
    assert completed.metrics == {"accepted": True}
    assert [store.get_lesson(identifier).status for identifier in lesson_ids[:5]] == [
        "used"
    ] * 5
    assert all(
        store.get_lesson(identifier).training_run_id == run.id
        for identifier in lesson_ids[:5]
    )
    untouched = store.get_lesson(lesson_ids[5])
    assert untouched is not None
    assert untouched.status == "pending"
    assert untouched.training_run_id is None


def test_local_floaty_completion_rejects_subset_without_consuming_claim(
    tmp_path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    lesson_ids = _add_local_training_lessons(store)
    run, claimed = store.claim_pending_lesson_ids(
        lesson_ids,
        metadata={"parameter_scope": "floaty-only"},
    )
    assert run is not None
    assert [lesson.id for lesson in claimed] == lesson_ids

    with pytest.raises(ValueError, match="exact claimed batch"):
        store.complete_local_floaty_run(
            run.id,
            lesson_ids[:4],
            status="contributed",
            bundle_out="outgoing/incomplete",
        )

    unchanged = store.get_training_status(run.id)
    assert unchanged is not None
    assert unchanged.status == "training"
    assert unchanged.checkpoint_out is None
    assert store.count_pending_lessons() == 0
    for identifier in lesson_ids:
        lesson = store.get_lesson(identifier)
        assert lesson is not None
        assert lesson.status == "training"
        assert lesson.training_run_id == run.id
        assert lesson.used_at is None


def test_failed_exact_local_claim_requeues_every_lesson(tmp_path) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    lesson_ids = _add_local_training_lessons(store)
    run, claimed = store.claim_pending_lesson_ids(lesson_ids)
    assert run is not None
    assert len(claimed) == 5

    released = store.release_training_run(
        run.id,
        "failed",
        error="local subprocess failed",
    )

    assert released.status == "failed"
    assert released.error == "local subprocess failed"
    assert store.count_pending_lessons() == 5
    for identifier in lesson_ids:
        lesson = store.get_lesson(identifier)
        assert lesson is not None
        assert lesson.status == "pending"
        assert lesson.training_run_id is None


def test_natural_correction_becomes_verified_lesson(tmp_path) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    session = store.create_session()
    problem = store.add_message(session.id, "user", "17 + 28")
    attempt = store.add_message(session.id, "assistant", "35", reply_to_id=problem.id)
    correction_message = store.add_message(session.id, "user", "No, the answer is 45.")
    parsed = parse_correction(correction_message.content, problem.content)
    assert parsed and parsed.verified
    lesson = record_verified_correction(
        store,
        session.id,
        attempt.id,
        correction_message.content,
        correction_message_id=correction_message.id,
    )
    assert lesson is not None
    assert lesson.correct_answer == "45"


def test_store_records_only_linked_consenting_human_dialogue_targets(tmp_path) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    session = store.create_session("talk-to-train")
    first_human = store.add_message(session.id, "user", "Hello, TinyDog.")

    bootstrap = store.add_trusted_dialogue_lesson(
        human_response_message_id=first_human.id,
        consent_local_training=True,
        metadata={"channel": "text"},
    )

    assert bootstrap.problem == TRUSTED_DIALOGUE_BOOTSTRAP_PROMPT
    assert bootstrap.correct_answer == first_human.content
    assert bootstrap.source == TRUSTED_HUMAN_DIALOGUE_SOURCE
    assert bootstrap.verifier == TRUSTED_HUMAN_DIALOGUE_VERIFIER
    assert bootstrap.skill == TRUSTED_HUMAN_DIALOGUE_SKILL
    assert bootstrap.metadata["bootstrap_prompt"] is True
    assert bootstrap.metadata["human_authored_target"] is True
    assert bootstrap.metadata["consent_local_training"] is True
    assert store.is_trusted_dialogue_lesson(bootstrap)

    assistant = store.add_message(
        session.id,
        "assistant",
        "What would you like to talk about?",
        reply_to_id=first_human.id,
    )
    second_human = store.add_message(
        session.id,
        "user",
        "I have been thinking about small language models.",
    )
    linked = store.add_trusted_dialogue_lesson(
        prompt_message_id=assistant.id,
        human_response_message_id=second_human.id,
        consent_local_training=True,
    )

    assert linked.problem == assistant.content
    assert linked.correct_answer == second_human.content
    assert linked.problem_message_id == assistant.id
    assert linked.assistant_message_id == assistant.id
    assert linked.correction_message_id == second_human.id
    assert linked.metadata["bootstrap_prompt"] is False
    assert store.is_trusted_dialogue_lesson(linked)
    duplicate = store.add_trusted_dialogue_lesson(
        prompt_message_id=assistant.id,
        human_response_message_id=second_human.id,
        consent_local_training=True,
    )
    assert duplicate.id == linked.id
    assert store.count_pending_lessons() == 2


def test_private_conversation_database_uses_owner_only_permissions(tmp_path) -> None:
    if os.name == "nt":
        pytest.skip("Windows privacy relies on the user profile ACL")
    directory = tmp_path / "private-conversations"
    directory.mkdir(mode=0o755)
    store = ApprenticeshipStore(directory / "learning.db")

    with store._connect() as connection:
        connection.execute(
            "INSERT INTO sessions(id, created_at, metadata) VALUES ('private', 'now', '{}')"
        )
        store._harden_permissions()
        assert directory.stat().st_mode & 0o777 == 0o700
        assert store.path.stat().st_mode & 0o777 == 0o600
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{store.path}{suffix}")
            if sidecar.exists():
                assert sidecar.stat().st_mode & 0o777 == 0o600

    target = tmp_path / "redirected.db"
    target.touch()
    link = tmp_path / "linked.db"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        ApprenticeshipStore(link)


def test_trusted_dialogue_requires_consent_adjacency_and_untampered_provenance(
    tmp_path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    session = store.create_session("talk-to-train")
    first_human = store.add_message(session.id, "user", "Hello.")
    with pytest.raises(ValueError, match="explicit consent"):
        store.add_trusted_dialogue_lesson(
            human_response_message_id=first_human.id,
            consent_local_training=False,
        )

    bootstrap = store.add_trusted_dialogue_lesson(
        human_response_message_id=first_human.id,
        consent_local_training=True,
    )
    assert not store.is_trusted_dialogue_lesson(
        replace(bootstrap, correct_answer="A forged target")
    )
    assert not store.is_trusted_dialogue_lesson(
        replace(
            bootstrap,
            metadata={
                **bootstrap.metadata,
                "consent_local_training": False,
            },
        )
    )

    assistant = store.add_message(session.id, "assistant", "How are you?")
    skipped_human = store.add_message(session.id, "user", "Quite well.")
    later_human = store.add_message(session.id, "user", "And you?")
    with pytest.raises(ValueError, match="first human response"):
        store.add_trusted_dialogue_lesson(
            prompt_message_id=assistant.id,
            human_response_message_id=later_human.id,
            consent_local_training=True,
        )
    with pytest.raises(ValueError, match="only valid for the first turn"):
        store.add_trusted_dialogue_lesson(
            human_response_message_id=skipped_human.id,
            consent_local_training=True,
        )


def test_generated_practice_variants_are_canonical_train_split() -> None:
    held_out_key = canonical_practice_key("What is 25 * 30?")
    assert held_out_key == "mul:25:30"
    assert split_for_key(held_out_key) == "test"

    variants = generate_practice_variants("Multiply 21 by 38.", count=5, seed=76)
    assert len(variants) == 5
    assert "What is 25 * 30?" not in {problem for problem, _ in variants}
    for problem, answer in variants:
        key = canonical_practice_key(problem)
        assert key is not None
        assert split_for_key(key) == "train"
        assert verify_answer(problem, answer).is_correct

    conversational = generate_practice_variants("What is 200 + 5?", count=5, seed=11)
    assert len(conversational) == 5
    assert all(
        canonical_practice_key(problem) is not None for problem, _ in conversational
    )
    assert all(
        split_for_key(canonical_practice_key(problem)) == "train"
        for problem, _ in conversational
    )


def test_store_rejects_incorrect_human_correction(tmp_path) -> None:
    store = ApprenticeshipStore(tmp_path / "learning.db")
    with pytest.raises(ValueError):
        store.add_verified_lesson(
            problem="2 + 2", first_attempt="5", correct_answer="6"
        )


def test_teacher_is_safely_disabled_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("MLM_TEACHER_PROVIDER", raising=False)
    teacher = TeacherClient.from_env()
    feedback = teacher.diagnose("2 + 2", "5", "4")
    assert not teacher.enabled
    assert not feedback.available
