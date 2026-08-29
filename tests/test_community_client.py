from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import mlx.core as mx
import pytest

from mini_arithmetic.apprenticeship import ApprenticeshipStore
import mini_arithmetic.community_client as community_client
from mini_arithmetic.community_client import (
    deterministic_lesson_split,
    initialize_community,
    load_signed_foundation,
    train_and_package_update,
)
from mini_arithmetic.community_release import create_community_release_manifest
from mini_arithmetic.community_updates import (
    ContributionQueue,
    load_community_contract,
    read_floaty_safetensors,
)
from mini_arithmetic.floaty_training import FloatyTrainingConfig
from mini_arithmetic.foundation_floaty import FloatyConfig, FoundationFloatyLM
from mini_arithmetic.foundation_release import (
    create_foundation_manifest,
    generate_owner_keypair,
)
from mini_arithmetic.hybrid import CONV, HybridConfig, HybridLM
from mini_arithmetic.scratch_tokenizer import (
    METADATA_FILENAME,
    TOKENIZER_FILENAME,
    ScratchTokenizer,
    build_scratch_tokenizer,
)


@pytest.fixture(scope="module")
def scratch_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> ScratchTokenizer:
    return build_scratch_tokenizer(
        [
            "The user asks arithmetic and the assistant answers exactly.",
            "Teacher feedback precedes a corrected assistant answer.",
            "19+23=42, 40+2=42, 84/2=42, and 50-8=42.",
        ],
        tmp_path_factory.mktemp("community-client-tokenizer"),
    )


def training_config() -> FloatyTrainingConfig:
    return FloatyTrainingConfig(
        steps=20,
        batch_size=2,
        context_length=96,
        learning_rate=2e-2,
        weight_decay=0.0,
        minimum_training_lessons=2,
        minimum_validation_lessons=1,
        seed=17,
    )


def signed_release(
    tmp_path: Path,
    tokenizer: ScratchTokenizer,
) -> tuple[Path, Path, Path]:
    release = tmp_path / "release"
    release.mkdir()
    config = HybridConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=96,
        model_dim=8,
        ffn_dim=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        conv_kernel_size=2,
        layer_schedule=(CONV,),
    )
    mx.random.seed(23)
    foundation = HybridLM(config)
    foundation.save_weights(str(release / "foundation.safetensors"))
    config_value = asdict(config)
    config_value["layer_schedule"] = list(config.layer_schedule)
    (release / "hybrid-config.json").write_text(
        json.dumps(config_value, sort_keys=True), encoding="utf-8"
    )
    shutil.copyfile(
        tokenizer.artifact_dir / TOKENIZER_FILENAME, release / TOKENIZER_FILENAME
    )
    shutil.copyfile(
        tokenizer.artifact_dir / METADATA_FILENAME,
        release / METADATA_FILENAME,
    )
    FoundationFloatyLM(
        foundation,
        FloatyConfig(rank=2, alpha=2.0),
    ).write_floaty_schema(release / "floaty-schema.json")
    private_key = tmp_path / "owner.key"
    public_key = tmp_path / "owner.pub"
    generate_owner_keypair(private_key, public_key)
    manifest = create_foundation_manifest(
        release,
        foundation_id="foundation-test-1",
        weights="foundation.safetensors",
        model_config="hybrid-config.json",
        tokenizer=TOKENIZER_FILENAME,
        tokenizer_metadata=METADATA_FILENAME,
        floaty_schema="floaty-schema.json",
        private_key=private_key,
    )
    return manifest, public_key, private_key


def add_lessons(path: Path) -> list[int]:
    store = ApprenticeshipStore(path)
    values = [
        ("19 + 23", "42"),
        ("40 + 2", "42"),
        ("84 / 2", "42"),
        ("50 - 8", "42"),
    ]
    return [
        store.add_verified_lesson(
            problem=problem,
            first_attempt="41",
            correct_answer=answer,
            feedback="Use exact arithmetic, then give the corrected answer.",
        ).id
        for problem, answer in values
    ]


def add_dialogue_lessons(path: Path, count: int = 5) -> list[int]:
    store = ApprenticeshipStore(path)
    session = store.create_session("human-dialogue")
    lesson_ids: list[int] = []
    first_human = store.add_message(session.id, "user", "Hello, TinyDog.")
    lesson_ids.append(
        store.add_trusted_dialogue_lesson(
            human_response_message_id=first_human.id,
            consent_local_training=True,
        ).id
    )
    prior_human = first_human
    for index in range(1, count):
        assistant = store.add_message(
            session.id,
            "assistant",
            f"Tell me something interesting about topic {index}.",
            reply_to_id=prior_human.id,
        )
        prior_human = store.add_message(
            session.id,
            "user",
            f"A thoughtful human response about topic {index}.",
        )
        lesson_ids.append(
            store.add_trusted_dialogue_lesson(
                prompt_message_id=assistant.id,
                human_response_message_id=prior_human.id,
                consent_local_training=True,
            ).id
        )
    return lesson_ids


def test_init_and_train_package_round_trip(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    manifest, public_key, private_key = signed_release(tmp_path, scratch_tokenizer)
    bootstrap = initialize_community(
        foundation_manifest=manifest,
        trusted_public_key=public_key,
        output_dir=tmp_path / "community-0",
        community_version="community-0",
        training_config=training_config(),
    )
    assert bootstrap.weights_path.is_file()
    assert bootstrap.contract_path.is_file()
    contract = load_community_contract(bootstrap.contract_path)
    initial = read_floaty_safetensors(
        bootstrap.weights_path,
        contract.tensor_specs,
        expected_sha256=contract.community_weights_sha256,
    )
    assert all(
        not value.any()
        for name, value in initial.items()
        if name.endswith(".down.weight")
    )
    bootstrap_record = json.loads((bootstrap.output_dir / "bootstrap.json").read_text())
    assert bootstrap_record["promotion"] == "not-performed"
    assert bootstrap_record["signature"] == "not-created"
    active_release = create_community_release_manifest(
        bootstrap.output_dir,
        sequence=0,
        foundation_manifest=manifest,
        trusted_public_key=public_key,
        private_key=private_key,
    )

    lessons_db = tmp_path / "lessons.sqlite3"
    lesson_ids = add_lessons(lessons_db)
    store = ApprenticeshipStore(lessons_db)
    training_run, claimed = store.claim_pending_lesson_ids(lesson_ids)
    assert training_run is not None
    assert [lesson.id for lesson in claimed] == lesson_ids
    trust_state = tmp_path / "trusted-community.json"
    result = train_and_package_update(
        foundation_manifest=manifest,
        trusted_public_key=public_key,
        community_release_manifest=active_release,
        trust_state_path=trust_state,
        lessons_db=lessons_db,
        candidate_output_dir=tmp_path / "local-candidates",
        bundle_path=tmp_path / "outgoing-update",
        training_config=training_config(),
        lesson_ids=lesson_ids,
        claimed_training_run_id=training_run.id,
        update_id="local-test-1",
    )
    assert result.accepted is True
    assert result.bundle_path is not None and result.bundle_path.is_dir()
    assert result.update_manifest is not None
    assert result.update_manifest.example_count == len(result.training_lesson_ids)
    assert set(result.training_lesson_ids).isdisjoint(result.validation_lesson_ids)
    assert set(result.training_lesson_ids) | set(result.validation_lesson_ids) == set(
        lesson_ids
    )
    assert all(
        store.get_lesson(identifier).status == "training" for identifier in lesson_ids
    )
    trusted = json.loads(trust_state.read_text())
    assert trusted["sequence"] == 0
    assert (
        trusted["foundation_manifest_sha256"]
        == bootstrap_record["foundation_manifest_sha256"]
    )

    queue = ContributionQueue(tmp_path / "queue")
    entry = queue.quarantine(result.bundle_path)
    validated = queue.validate(entry.content_sha256, contract)
    assert validated.entry.state == "accepted"
    assert set(validated.tensors) == {spec.name for spec in contract.tensor_specs}
    assert all(name.startswith("floaty.") for name in validated.tensors)


def test_split_is_deterministic_and_independent(tmp_path: Path) -> None:
    database = tmp_path / "lessons.sqlite3"
    add_lessons(database)
    lessons = ApprenticeshipStore(database).list_recent_lessons(
        limit=10, pending_only=True
    )
    first = deterministic_lesson_split(
        lessons,
        foundation_sha256="a" * 64,
        validation_fraction=0.25,
        minimum_training_lessons=2,
        minimum_validation_lessons=1,
    )
    second = deterministic_lesson_split(
        list(reversed(lessons)),
        foundation_sha256="a" * 64,
        validation_fraction=0.25,
        minimum_training_lessons=2,
        minimum_validation_lessons=1,
    )
    assert [[item.id for item in group] for group in first] == [
        [item.id for item in group] for group in second
    ]
    assert {item.problem for item in first[0]}.isdisjoint(
        {item.problem for item in first[1]}
    )


def test_claimed_human_dialogue_has_a_deterministic_training_holdout_split(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dialogue-lessons.sqlite3"
    lesson_ids = add_dialogue_lessons(database)
    store = ApprenticeshipStore(database)
    training_run, claimed = store.claim_pending_lesson_ids(lesson_ids)
    assert training_run is not None
    assert [lesson.id for lesson in claimed] == lesson_ids

    first = deterministic_lesson_split(
        claimed,
        foundation_sha256="d" * 64,
        validation_fraction=0.2,
        minimum_training_lessons=4,
        minimum_validation_lessons=1,
        trusted_dialogue_validator=store.is_trusted_dialogue_lesson,
    )
    second = deterministic_lesson_split(
        list(reversed(claimed)),
        foundation_sha256="d" * 64,
        validation_fraction=0.2,
        minimum_training_lessons=4,
        minimum_validation_lessons=1,
        trusted_dialogue_validator=store.is_trusted_dialogue_lesson,
    )

    assert [[lesson.id for lesson in group] for group in first] == [
        [lesson.id for lesson in group] for group in second
    ]
    assert len(first[0]) == 4
    assert len(first[1]) == 1
    assert set(lesson.id for lesson in first[0]).isdisjoint(
        lesson.id for lesson in first[1]
    )
    assert {lesson.id for group in first for lesson in group} == set(lesson_ids)


def test_community_split_rejects_forged_or_arbitrary_conversation_records(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dialogue-lessons.sqlite3"
    lesson_ids = add_dialogue_lessons(database, count=2)
    store = ApprenticeshipStore(database)
    real = store.get_lesson(lesson_ids[0])
    assert real is not None

    forged = replace(real, correct_answer="This text was never written by the human.")
    with pytest.raises(
        community_client.CommunityClientError, match="trusted human-dialogue"
    ):
        deterministic_lesson_split(
            [forged, store.get_lesson(lesson_ids[1])],
            foundation_sha256="e" * 64,
            validation_fraction=0.5,
            minimum_training_lessons=1,
            minimum_validation_lessons=1,
            trusted_dialogue_validator=store.is_trusted_dialogue_lesson,
        )

    arbitrary = {
        "id": 999,
        "problem": "Pretend prompt",
        "first_attempt": "",
        "correct_answer": "Pretend human response",
        "skill": "conversation",
        "source": "human-conversation",
        "verifier": "trusted-human-dialogue-v1",
        "metadata": {
            "bootstrap_prompt": True,
            "consent_local_training": True,
            "human_authored_target": True,
        },
    }
    with pytest.raises(
        community_client.CommunityClientError, match="trusted human-dialogue"
    ):
        deterministic_lesson_split(
            [arbitrary, store.get_lesson(lesson_ids[1])],
            foundation_sha256="e" * 64,
            validation_fraction=0.5,
            minimum_training_lessons=1,
            minimum_validation_lessons=1,
            trusted_dialogue_validator=store.is_trusted_dialogue_lesson,
        )


def test_signed_tokenizer_metadata_tamper_is_rejected(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    manifest, public_key, _ = signed_release(tmp_path, scratch_tokenizer)
    metadata = manifest.parent / METADATA_FILENAME
    metadata.write_text(metadata.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="signed foundation|does not match"):
        with load_signed_foundation(manifest, public_key):
            pass


def test_signed_non_hybrid_config_has_no_fallback(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    manifest, public_key, _ = signed_release(tmp_path, scratch_tokenizer)
    release = manifest.parent
    private_key = tmp_path / "owner.key"
    manifest.unlink()
    (release / "hybrid-config.json").write_text(
        json.dumps(
            {
                "vocab_size": scratch_tokenizer.vocab_size,
                "context_length": 96,
                "num_layers": 1,
                "model_dim": 8,
                "num_heads": 1,
                "ffn_dim": 16,
                "norm_eps": 1e-5,
            }
        ),
        encoding="utf-8",
    )
    replacement = create_foundation_manifest(
        release,
        foundation_id="foundation-test-legacy-shape",
        weights="foundation.safetensors",
        model_config="hybrid-config.json",
        tokenizer=TOKENIZER_FILENAME,
        tokenizer_metadata=METADATA_FILENAME,
        floaty_schema="floaty-schema.json",
        private_key=private_key,
    )
    with pytest.raises(
        ValueError,
        match="exact HybridConfig|floaty schema does not bind",
    ):
        with load_signed_foundation(replacement, public_key):
            pass


def test_contribution_rejects_unsigned_bootstrap_artifacts(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
) -> None:
    manifest, public_key, _ = signed_release(tmp_path, scratch_tokenizer)
    bootstrap = initialize_community(
        foundation_manifest=manifest,
        trusted_public_key=public_key,
        output_dir=tmp_path / "community-0",
        community_version="community-0",
        training_config=training_config(),
    )
    lessons_db = tmp_path / "lessons.sqlite3"
    add_lessons(lessons_db)
    trust_state = tmp_path / "trusted-community.json"

    with pytest.raises(ValueError, match="community release"):
        train_and_package_update(
            foundation_manifest=manifest,
            trusted_public_key=public_key,
            community_release_manifest=bootstrap.contract_path,
            trust_state_path=trust_state,
            lessons_db=lessons_db,
            candidate_output_dir=tmp_path / "local-candidates",
            bundle_path=tmp_path / "outgoing-update",
            training_config=training_config(),
        )
    assert not trust_state.exists()
    assert not (tmp_path / "outgoing-update").exists()


def test_release_artifact_is_resnapshotted_before_client_load(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, public_key, private_key = signed_release(tmp_path, scratch_tokenizer)
    bootstrap = initialize_community(
        foundation_manifest=manifest,
        trusted_public_key=public_key,
        output_dir=tmp_path / "community-0",
        community_version="community-0",
        training_config=training_config(),
    )
    active_release = create_community_release_manifest(
        bootstrap.output_dir,
        sequence=0,
        foundation_manifest=manifest,
        trusted_public_key=public_key,
        private_key=private_key,
    )
    original_verify = community_client.verify_community_release

    def verify_then_replace(*args: object, **kwargs: object):
        verified = original_verify(*args, **kwargs)
        verified.artifacts["weights"].path.write_bytes(b"replaced after verification")
        return verified

    monkeypatch.setattr(
        community_client,
        "verify_community_release",
        verify_then_replace,
    )
    lessons_db = tmp_path / "lessons.sqlite3"
    add_lessons(lessons_db)
    trust_state = tmp_path / "trusted-community.json"
    with pytest.raises(ValueError, match="changed before load|snapshot verification"):
        train_and_package_update(
            foundation_manifest=manifest,
            trusted_public_key=public_key,
            community_release_manifest=active_release,
            trust_state_path=trust_state,
            lessons_db=lessons_db,
            candidate_output_dir=tmp_path / "local-candidates",
            bundle_path=tmp_path / "outgoing-update",
            training_config=training_config(),
        )
    assert not trust_state.exists()


def test_trust_state_advances_before_local_lessons_are_read(
    tmp_path: Path,
    scratch_tokenizer: ScratchTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, public_key, private_key = signed_release(tmp_path, scratch_tokenizer)
    bootstrap = initialize_community(
        foundation_manifest=manifest,
        trusted_public_key=public_key,
        output_dir=tmp_path / "community-0",
        community_version="community-0",
        training_config=training_config(),
    )
    active_release = create_community_release_manifest(
        bootstrap.output_dir,
        sequence=0,
        foundation_manifest=manifest,
        trusted_public_key=public_key,
        private_key=private_key,
    )
    trust_state = tmp_path / "trusted-community.json"

    class StopAfterTrust:
        def __init__(self, _path: str | Path) -> None:
            assert json.loads(trust_state.read_text())["sequence"] == 0

        def list_recent_lessons(self, *, limit: int, pending_only: bool):
            assert limit > 0 and pending_only is True
            raise RuntimeError("stop after trust advancement")

    monkeypatch.setattr(community_client, "ApprenticeshipStore", StopAfterTrust)
    with pytest.raises(RuntimeError, match="stop after trust advancement"):
        train_and_package_update(
            foundation_manifest=manifest,
            trusted_public_key=public_key,
            community_release_manifest=active_release,
            trust_state_path=trust_state,
            lessons_db=tmp_path / "not-opened.sqlite3",
            candidate_output_dir=tmp_path / "local-candidates",
            bundle_path=tmp_path / "outgoing-update",
            training_config=training_config(),
        )
    assert trust_state.exists()
    assert not (tmp_path / "outgoing-update").exists()
