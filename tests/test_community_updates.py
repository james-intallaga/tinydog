from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import struct

import mlx.core as mx
import numpy as np
import pytest

import mini_arithmetic.community_updates as community_updates
from mini_arithmetic.community_updates import (
    CompatibilityError,
    CommunityContract,
    ContributionQueue,
    DELTA_FILENAME,
    ManifestError,
    QueueStateError,
    ReplayError,
    TensorSpec,
    TensorValidationError,
    UnsafeBundleError,
    UpdateLimits,
    aggregate_candidate,
    load_community_contract,
    main,
    read_floaty_safetensors,
    sha256_file,
    write_floaty_safetensors,
    write_update_bundle,
)
from mini_arithmetic.foundation_floaty import FloatyConfig, FoundationFloatyLM
from mini_arithmetic.hybrid import ATTENTION, HybridConfig, HybridLM


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _specs() -> tuple[TensorSpec, ...]:
    return (
        TensorSpec("floaty.attention.lora_a", (2, 3), "F32"),
        TensorSpec("floaty.attention.lora_b", (3, 2), "F16"),
    )


def _base_and_contract(
    tmp_path: Path,
    *,
    specs: tuple[TensorSpec, ...] | None = None,
    version: str = "community-7",
) -> tuple[Path, CommunityContract]:
    active_specs = specs or _specs()
    base = tmp_path / f"base-{version}.safetensors"
    write_floaty_safetensors(
        base,
        {spec.name: np.zeros(spec.shape, dtype=np.float32) for spec in active_specs},
        active_specs,
    )
    contract = CommunityContract(
        foundation_sha256=_hash("foundation-owner-release-3"),
        community_version=version,
        community_weights_sha256=sha256_file(base),
        architecture_sha256=_hash("architecture-3"),
        tokenizer_sha256=_hash("tokenizer-5"),
        training_recipe_sha256=_hash("local-floaty-recipe-2"),
        tensor_specs=active_specs,
    )
    return base, contract


def _deltas(
    specs: tuple[TensorSpec, ...], value: float = 0.25
) -> dict[str, np.ndarray]:
    return {spec.name: np.full(spec.shape, value, dtype=np.float32) for spec in specs}


def _write_raw_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    header: dict[str, object] = {}
    offset = 0
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for name in sorted(tensors):
            stream.write(tensors[name][2])


def _attacker_bundle(
    path: Path,
    contract: CommunityContract,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    *,
    update_id: str,
    created_at: datetime = NOW,
) -> Path:
    path.mkdir()
    delta = path / DELTA_FILENAME
    _write_raw_safetensors(delta, tensors)
    manifest = {
        "schema_version": 1,
        "update_id": update_id,
        "created_at": created_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        **contract.compatibility_dict(),
        "delta_file": DELTA_FILENAME,
        "delta_sha256": sha256_file(delta),
        "delta_bytes": delta.stat().st_size,
        "example_count": 4,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def test_strict_bound_bundle_moves_from_quarantine_to_accepted(tmp_path: Path) -> None:
    _, contract = _base_and_contract(tmp_path)
    bundle = tmp_path / "update"
    manifest = write_update_bundle(
        bundle,
        contract=contract,
        update_id="device-a.0001",
        deltas=_deltas(contract.tensor_specs),
        example_count=17,
        created_at=NOW,
    )
    assert manifest.foundation_sha256 == contract.foundation_sha256
    assert manifest.community_version == contract.community_version
    assert manifest.community_weights_sha256 == contract.community_weights_sha256
    assert manifest.floaty_schema_sha256 == contract.floaty_schema_sha256

    queue = ContributionQueue(tmp_path / "queue")
    entry = queue.quarantine(bundle, received_at=NOW)
    assert entry.state == "quarantined"
    validated = queue.validate(entry.content_sha256, contract, now=NOW)

    assert validated.entry.state == "accepted"
    assert validated.delta_l2_norm == pytest.approx(np.sqrt(12 * 0.25**2))
    assert [item.content_sha256 for item in queue.list_entries(state="accepted")] == [
        entry.content_sha256
    ]
    assert queue.list_entries(state="quarantined") == []
    assert sorted(item.name for item in (entry.path / "events").iterdir()) == [
        "0000-quarantined.json",
        "0001-accepted.json",
    ]
    with pytest.raises(QueueStateError):
        queue.validate(entry.content_sha256, contract, now=NOW)


def test_manifest_schema_is_exact_and_paths_cannot_escape(tmp_path: Path) -> None:
    _, contract = _base_and_contract(tmp_path)
    bundle = tmp_path / "update"
    write_update_bundle(
        bundle,
        contract=contract,
        update_id="strict-1",
        deltas=_deltas(contract.tensor_specs),
        example_count=2,
        created_at=NOW,
    )
    delta_bytes = (bundle / DELTA_FILENAME).read_bytes()
    manifest_path = bundle / "manifest.json"
    value = json.loads(manifest_path.read_text())
    value["note"] = "schema smuggling"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ManifestError, match="keys do not match"):
        ContributionQueue(tmp_path / "queue-extra").quarantine(bundle)

    value.pop("note")
    value["delta_file"] = "../foundation.safetensors"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(UnsafeBundleError, match="delta_file"):
        ContributionQueue(tmp_path / "queue-path").quarantine(bundle)

    value["delta_file"] = DELTA_FILENAME
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(delta_bytes)
    (bundle / DELTA_FILENAME).unlink()
    (bundle / DELTA_FILENAME).symlink_to(outside)
    with pytest.raises(UnsafeBundleError, match="symlink"):
        ContributionQueue(tmp_path / "queue-symlink").quarantine(bundle)


def test_foundation_tensor_and_wrong_shape_are_rejected_in_quarantine_validation(
    tmp_path: Path,
) -> None:
    _, contract = _base_and_contract(tmp_path)
    good_a = np.zeros((2, 3), dtype="<f4").tobytes()
    good_b = np.zeros((3, 2), dtype="<f2").tobytes()
    foundation = np.zeros((2, 2), dtype="<f4").tobytes()
    bundle = _attacker_bundle(
        tmp_path / "foundation-attack",
        contract,
        {
            "floaty.attention.lora_a": ("F32", (2, 3), good_a),
            "floaty.attention.lora_b": ("F16", (3, 2), good_b),
            "token_embedding.weight": ("F32", (2, 2), foundation),
        },
        update_id="foundation-attack",
    )
    queue = ContributionQueue(tmp_path / "queue")
    entry = queue.quarantine(bundle, received_at=NOW)
    with pytest.raises(TensorValidationError, match="foundation or non-floaty"):
        queue.validate(entry.content_sha256, contract, now=NOW)
    assert queue.state(entry.content_sha256) == "rejected"

    wrong_shape = _attacker_bundle(
        tmp_path / "wrong-shape",
        contract,
        {
            "floaty.attention.lora_a": (
                "F32",
                (3, 2),
                np.zeros((3, 2), dtype="<f4").tobytes(),
            ),
            "floaty.attention.lora_b": ("F16", (3, 2), good_b),
        },
        update_id="wrong-shape",
    )
    second = queue.quarantine(wrong_shape, received_at=NOW)
    with pytest.raises(TensorValidationError, match="dtype or shape"):
        queue.validate(second.content_sha256, contract, now=NOW)
    assert queue.state(second.content_sha256) == "rejected"


def test_stale_and_incompatible_updates_are_terminally_rejected(tmp_path: Path) -> None:
    _, active = _base_and_contract(tmp_path, version="community-8")
    _, old = _base_and_contract(tmp_path, version="community-7")
    queue = ContributionQueue(
        tmp_path / "queue", limits=UpdateLimits(max_age=timedelta(days=1))
    )

    old_bundle = tmp_path / "old-version"
    write_update_bundle(
        old_bundle,
        contract=old,
        update_id="old-version",
        deltas=_deltas(old.tensor_specs),
        example_count=2,
        created_at=NOW,
    )
    old_entry = queue.quarantine(old_bundle, received_at=NOW)
    with pytest.raises(CompatibilityError, match="community_version"):
        queue.validate(old_entry.content_sha256, active, now=NOW)
    assert queue.state(old_entry.content_sha256) == "rejected"

    stale_bundle = tmp_path / "stale-time"
    write_update_bundle(
        stale_bundle,
        contract=active,
        update_id="stale-time",
        deltas=_deltas(active.tensor_specs, 0.5),
        example_count=2,
        created_at=NOW - timedelta(days=2),
    )
    stale_entry = queue.quarantine(stale_bundle, received_at=NOW)
    with pytest.raises(CompatibilityError, match="stale"):
        queue.validate(stale_entry.content_sha256, active, now=NOW)
    assert queue.state(stale_entry.content_sha256) == "rejected"


def test_replay_is_detected_by_update_id_and_tensor_content(tmp_path: Path) -> None:
    _, contract = _base_and_contract(tmp_path)
    queue = ContributionQueue(tmp_path / "queue")
    first = tmp_path / "first"
    write_update_bundle(
        first,
        contract=contract,
        update_id="replay-id",
        deltas=_deltas(contract.tensor_specs),
        example_count=2,
        created_at=NOW,
    )
    queue.quarantine(first, received_at=NOW)
    with pytest.raises(ReplayError, match="update_id"):
        queue.quarantine(first, received_at=NOW)

    renamed = tmp_path / "renamed"
    write_update_bundle(
        renamed,
        contract=contract,
        update_id="different-id",
        deltas=_deltas(contract.tensor_specs),
        example_count=99,
        created_at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(ReplayError, match="delta content"):
        queue.quarantine(renamed, received_at=NOW)


def test_replay_uses_semantic_tensor_identity_not_container_whitespace(
    tmp_path: Path,
) -> None:
    _, contract = _base_and_contract(tmp_path)
    first = tmp_path / "semantic-first"
    second = tmp_path / "semantic-second"
    write_update_bundle(
        first,
        contract=contract,
        update_id="semantic-1",
        deltas=_deltas(contract.tensor_specs),
        example_count=1,
        created_at=NOW,
    )
    write_update_bundle(
        second,
        contract=contract,
        update_id="semantic-2",
        deltas=_deltas(contract.tensor_specs),
        example_count=999,
        created_at=NOW + timedelta(seconds=1),
    )
    delta_path = second / DELTA_FILENAME
    original = delta_path.read_bytes()
    header_size = struct.unpack("<Q", original[:8])[0]
    header = original[8 : 8 + header_size].rstrip() + b" " * 16
    reformatted = struct.pack("<Q", len(header)) + header + original[8 + header_size :]
    delta_path.write_bytes(reformatted)
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["delta_sha256"] = sha256_file(delta_path)
    manifest["delta_bytes"] = len(reformatted)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    queue = ContributionQueue(tmp_path / "semantic-queue")
    queue.quarantine(first, received_at=NOW)
    with pytest.raises(ReplayError, match="semantic delta"):
        queue.quarantine(second, received_at=NOW)


def test_oversize_nonfinite_and_outlier_updates_are_rejected(tmp_path: Path) -> None:
    specs = (TensorSpec("floaty.scalar", (1,), "F32"),)
    _, contract = _base_and_contract(tmp_path, specs=specs)
    normal = tmp_path / "normal"
    write_update_bundle(
        normal,
        contract=contract,
        update_id="oversized",
        deltas={"floaty.scalar": np.array([1], dtype=np.float32)},
        example_count=1,
        created_at=NOW,
    )
    actual_size = (normal / DELTA_FILENAME).stat().st_size
    tiny_queue = ContributionQueue(
        tmp_path / "tiny-queue",
        limits=UpdateLimits(max_delta_bytes=actual_size - 1),
    )
    with pytest.raises(ManifestError, match="oversized"):
        tiny_queue.quarantine(normal)

    nonfinite = _attacker_bundle(
        tmp_path / "nonfinite",
        contract,
        {"floaty.scalar": ("F32", (1,), np.array([np.nan], dtype="<f4").tobytes())},
        update_id="nonfinite",
    )
    queue = ContributionQueue(
        tmp_path / "queue",
        limits=UpdateLimits(max_delta_l2_norm=10, aggregation_clip_l2_norm=2),
    )
    bad = queue.quarantine(nonfinite, received_at=NOW)
    with pytest.raises(TensorValidationError, match="non-finite"):
        queue.validate(bad.content_sha256, contract, now=NOW)
    assert queue.state(bad.content_sha256) == "rejected"

    outlier = tmp_path / "outlier"
    write_update_bundle(
        outlier,
        contract=contract,
        update_id="outlier",
        deltas={"floaty.scalar": np.array([11], dtype=np.float32)},
        example_count=1,
        created_at=NOW,
    )
    large = queue.quarantine(outlier, received_at=NOW)
    with pytest.raises(TensorValidationError, match="outlier"):
        queue.validate(large.content_sha256, contract, now=NOW)
    assert queue.state(large.content_sha256) == "rejected"


def test_clipped_weighted_aggregation_only_creates_manual_review_candidate(
    tmp_path: Path,
) -> None:
    specs = (TensorSpec("floaty.scalar", (1,), "F32"),)
    base = tmp_path / "base.safetensors"
    write_floaty_safetensors(
        base, {"floaty.scalar": np.array([10], dtype=np.float32)}, specs
    )
    contract = CommunityContract(
        foundation_sha256=_hash("foundation"),
        community_version="community-1",
        community_weights_sha256=sha256_file(base),
        architecture_sha256=_hash("architecture"),
        tokenizer_sha256=_hash("tokenizer"),
        training_recipe_sha256=_hash("recipe"),
        tensor_specs=specs,
    )
    limits = UpdateLimits(
        min_aggregate_updates=2,
        max_delta_l2_norm=10,
        aggregation_clip_l2_norm=4,
        max_weight_per_update=2,
    )
    queue = ContributionQueue(tmp_path / "queue", limits=limits)
    accepted: list[str] = []
    for update_id, delta, examples in (("small", 2.0, 1), ("large", 8.0, 50)):
        bundle = tmp_path / update_id
        write_update_bundle(
            bundle,
            contract=contract,
            update_id=update_id,
            deltas={"floaty.scalar": np.array([delta], dtype=np.float32)},
            example_count=examples,
            created_at=NOW,
        )
        entry = queue.quarantine(bundle, received_at=NOW)
        queue.validate(entry.content_sha256, contract, now=NOW)
        accepted.append(entry.content_sha256)

    result = aggregate_candidate(
        queue,
        accepted,
        contract=contract,
        base_floaty_path=base,
        output_dir=tmp_path / "candidates",
    )
    candidate = mx.load(str(result.weights_path))
    # (2 * weight 1 + clipped(8 -> 4) * capped weight 2) / 3 = 10/3.
    assert float(candidate["floaty.scalar"].item()) == pytest.approx(10 + 10 / 3)
    assert result.report["aggregate_delta_l2_norm"] == pytest.approx(10 / 3)
    assert result.report["promotion"] == {
        "status": "manual_review_required",
        "auto_promoted": False,
        "active_pointer_written": False,
        "assigned_community_version": None,
    }
    assert not (tmp_path / "candidates" / "active.json").exists()
    assert all(queue.state(identifier) == "accepted" for identifier in accepted)
    with pytest.raises(FileExistsError):
        aggregate_candidate(
            queue,
            accepted,
            contract=contract,
            base_floaty_path=base,
            output_dir=tmp_path / "candidates",
        )


def test_artifact_writers_never_overwrite_existing_paths(tmp_path: Path) -> None:
    base, contract = _base_and_contract(tmp_path)
    original = base.read_bytes()
    with pytest.raises(FileExistsError):
        write_floaty_safetensors(
            base,
            {spec.name: np.ones(spec.shape) for spec in contract.tensor_specs},
            contract.tensor_specs,
        )
    assert base.read_bytes() == original

    bundle = tmp_path / "immutable-bundle"
    write_update_bundle(
        bundle,
        contract=contract,
        update_id="immutable",
        deltas=_deltas(contract.tensor_specs),
        example_count=1,
        created_at=NOW,
    )
    manifest_before = (bundle / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_update_bundle(
            bundle,
            contract=contract,
            update_id="replacement",
            deltas=_deltas(contract.tensor_specs, 0.9),
            example_count=1,
            created_at=NOW,
        )
    assert (bundle / "manifest.json").read_bytes() == manifest_before


def test_contract_consumes_foundation_floaty_manifest_without_model_reconstruction(
    tmp_path: Path,
) -> None:
    model = FoundationFloatyLM(
        HybridLM(
            HybridConfig(
                vocab_size=32,
                context_length=16,
                model_dim=8,
                ffn_dim=16,
                num_attention_heads=1,
                num_key_value_heads=1,
                head_dim=8,
                conv_kernel_size=3,
                rope_theta=10_000.0,
                layer_schedule=(ATTENTION,),
            )
        ),
        FloatyConfig(rank=2),
    )
    schema = model.config_manifest()["tensors"]
    base = tmp_path / "base-from-schema.safetensors"
    specs = tuple(
        TensorSpec(item["name"], tuple(item["shape"]), item["dtype"]) for item in schema
    )
    write_floaty_safetensors(
        base,
        {spec.name: np.zeros(spec.shape, dtype=np.float32) for spec in specs},
        specs,
    )
    contract = CommunityContract.from_floaty_schema(
        schema,
        foundation_sha256=model.foundation_hash(),
        community_version="community-0",
        community_weights_sha256=sha256_file(base),
        architecture_sha256=model.config_hash(),
        tokenizer_sha256=_hash("tokenizer"),
        training_recipe_sha256=_hash("recipe"),
    )

    assert (
        tuple(spec.name for spec in contract.tensor_specs)
        == model.floaty_parameter_names()
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    assert load_community_contract(contract_path) == contract

    malformed = contract.to_dict()
    malformed["tensors"][0]["foundation_parameter"] = True
    with pytest.raises(ValueError, match="name, shape, and dtype"):
        CommunityContract.from_dict(malformed)


def test_offline_cli_quarantines_and_validates(tmp_path: Path, capsys) -> None:
    base, contract = _base_and_contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    queue_path = tmp_path / "cli-queue"
    entries: list[str] = []
    for index, value in enumerate((0.1, 0.2, 0.3), start=1):
        bundle = tmp_path / f"cli-update-{index}"
        write_update_bundle(
            bundle,
            contract=contract,
            update_id=f"cli-update-{index}",
            deltas=_deltas(contract.tensor_specs, value),
            example_count=3,
            created_at=datetime.now(timezone.utc),
        )
        assert (
            main(["quarantine", "--queue", str(queue_path), "--bundle", str(bundle)])
            == 0
        )
        quarantine_output = json.loads(capsys.readouterr().out)
        assert quarantine_output["state"] == "quarantined"
        entries.append(quarantine_output["content_sha256"])
        assert (
            main(
                [
                    "validate",
                    "--queue",
                    str(queue_path),
                    "--contract",
                    str(contract_path),
                    "--entry",
                    quarantine_output["content_sha256"],
                ]
            )
            == 0
        )
        validate_output = json.loads(capsys.readouterr().out)
        assert validate_output["state"] == "accepted"

    assert (
        main(
            [
                "aggregate",
                "--queue",
                str(queue_path),
                "--contract",
                str(contract_path),
                "--base-floaty",
                str(base),
                "--output",
                str(tmp_path / "cli-candidates"),
                *entries,
            ]
        )
        == 0
    )
    aggregate_output = json.loads(capsys.readouterr().out)
    assert aggregate_output["promotion_status"] == "manual_review_required"
    assert Path(aggregate_output["candidate_dir"]).is_dir()


def test_quarantine_hashes_and_copies_one_no_follow_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, contract = _base_and_contract(tmp_path)
    bundle = tmp_path / "snapshot-update"
    write_update_bundle(
        bundle,
        contract=contract,
        update_id="snapshot-update",
        deltas=_deltas(contract.tensor_specs, 0.4),
        example_count=2,
        created_at=NOW,
    )
    original_delta = (bundle / DELTA_FILENAME).read_bytes()
    real_snapshot = community_updates._snapshot_child_descriptor
    swapped = False

    def snapshot_then_swap(*args, **kwargs):
        nonlocal swapped
        payload = real_snapshot(*args, **kwargs)
        name = args[1] if len(args) > 1 else kwargs["name"]
        if name == DELTA_FILENAME and not swapped:
            swapped = True
            (bundle / DELTA_FILENAME).write_bytes(b"attacker replacement")
        return payload

    monkeypatch.setattr(
        community_updates, "_snapshot_child_descriptor", snapshot_then_swap
    )
    entry = ContributionQueue(tmp_path / "queue").quarantine(bundle, received_at=NOW)

    assert swapped
    assert (entry.path / DELTA_FILENAME).read_bytes() == original_delta
    assert (bundle / DELTA_FILENAME).read_bytes() != original_delta


def test_validation_rejects_a_stored_delta_replaced_by_symlink(tmp_path: Path) -> None:
    _, contract = _base_and_contract(tmp_path)
    bundle = tmp_path / "symlink-update"
    write_update_bundle(
        bundle,
        contract=contract,
        update_id="symlink-update",
        deltas=_deltas(contract.tensor_specs),
        example_count=2,
        created_at=NOW,
    )
    queue = ContributionQueue(tmp_path / "queue")
    entry = queue.quarantine(bundle, received_at=NOW)
    stored_delta = entry.path / DELTA_FILENAME
    outside = tmp_path / "stored-outside.safetensors"
    outside.write_bytes(stored_delta.read_bytes())
    stored_delta.unlink()
    stored_delta.symlink_to(outside)

    with pytest.raises(UnsafeBundleError, match="symlink"):
        queue.validate(entry.content_sha256, contract, now=NOW)
    assert queue.state(entry.content_sha256) == "rejected"


def test_base_and_contract_are_single_snapshot_and_no_follow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    specs = (TensorSpec("floaty.scalar", (1,), "F32"),)
    base = tmp_path / "base.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    write_floaty_safetensors(
        base, {"floaty.scalar": np.array([10], dtype=np.float32)}, specs
    )
    write_floaty_safetensors(
        replacement,
        {"floaty.scalar": np.array([100], dtype=np.float32)},
        specs,
    )
    contract = CommunityContract.from_floaty_schema(
        [spec.to_dict() for spec in specs],
        foundation_sha256=_hash("foundation"),
        community_version="community-snapshot",
        community_weights_sha256=sha256_file(base),
        architecture_sha256=_hash("architecture"),
        tokenizer_sha256=_hash("tokenizer"),
        training_recipe_sha256=_hash("recipe"),
    )
    loaded_base = read_floaty_safetensors(
        base,
        specs,
        expected_sha256=contract.community_weights_sha256,
    )
    assert loaded_base["floaty.scalar"].tolist() == [10.0]
    with pytest.raises(CompatibilityError, match="expected_sha256"):
        read_floaty_safetensors(base, specs, expected_sha256=_hash("wrong-base"))
    queue = ContributionQueue(
        tmp_path / "queue",
        limits=UpdateLimits(min_aggregate_updates=1),
    )
    bundle = tmp_path / "delta"
    write_update_bundle(
        bundle,
        contract=contract,
        update_id="one",
        deltas={"floaty.scalar": np.array([1], dtype=np.float32)},
        example_count=1,
        created_at=NOW,
    )
    entry = queue.quarantine(bundle, received_at=NOW)
    queue.validate(entry.content_sha256, contract, now=NOW)

    real_snapshot = community_updates._snapshot_regular_path
    swapped = False

    def snapshot_then_replace(path, *, max_bytes, allow_empty):
        nonlocal swapped
        payload = real_snapshot(path, max_bytes=max_bytes, allow_empty=allow_empty)
        if Path(path) == base and not swapped:
            swapped = True
            base.write_bytes(replacement.read_bytes())
        return payload

    monkeypatch.setattr(
        community_updates, "_snapshot_regular_path", snapshot_then_replace
    )
    result = aggregate_candidate(
        queue,
        [entry.content_sha256],
        contract=contract,
        base_floaty_path=base,
        output_dir=tmp_path / "candidates",
    )
    candidate = mx.load(str(result.weights_path))
    assert swapped
    assert float(candidate["floaty.scalar"].item()) == pytest.approx(11)

    base_link = tmp_path / "base-link.safetensors"
    base_link.symlink_to(replacement)
    with pytest.raises(UnsafeBundleError, match="symlink"):
        read_floaty_safetensors(base_link, specs)

    contract_path = tmp_path / "contract-real.json"
    contract_path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    contract_link = tmp_path / "contract-link.json"
    contract_link.symlink_to(contract_path)
    with pytest.raises(UnsafeBundleError, match="symlink"):
        load_community_contract(contract_link)
