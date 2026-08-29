from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import wave

import av
import numpy as np
import pytest

import mini_arithmetic.native_prepare as native_prepare_module
from mini_arithmetic.multimodal import MultimodalConfig
from mini_arithmetic.native_data import NativeShardDataset
from mini_arithmetic.native_prepare import (
    NativePreparationConfig,
    prepare_native_dataset,
)
from mini_arithmetic.scratch_tokenizer import ScratchTokenizer, build_scratch_tokenizer
from test_multimodal import tiny_multimodal_config


@pytest.fixture(scope="module")
def native_tokenizer(tmp_path_factory: pytest.TempPathFactory) -> ScratchTokenizer:
    return build_scratch_tokenizer(
        [
            "A consenting speaker asks arithmetic questions and receives exact answers.",
            "The assistant says that seventeen plus twenty-five equals forty-two.",
            "Native duplex speech and selected frames are trained only from owned data.",
        ],
        tmp_path_factory.mktemp("native-prepare-tokenizer"),
    )


def _model_config(root: Path) -> Path:
    payload = tiny_multimodal_config().to_dict()
    payload["thinker"]["vocab_size"] = 32_768
    payload["thinker"]["context_length"] = 256
    payload["audio"]["frame_size"] = 480
    path = root / "multimodal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _wav(path: Path, *, value: int, samples: int = 960, rate: int = 24_000) -> None:
    pcm = np.full((samples,), value, dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm.tobytes())


def _jpeg(path: Path, *, value: int = 127, width: int = 320, height: int = 192) -> None:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mjpeg") as container:
        stream = container.add_stream("mjpeg", rate=1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuvj420p"
        frame = av.VideoFrame.from_ndarray(
            np.full((height, width, 3), value, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    path.write_bytes(buffer.getvalue())


def _records(
    root: Path, *, count: int = 4, with_frames: bool = True
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(count):
        user = root / f"user-{index}.wav"
        model = root / f"model-{index}.wav"
        _wav(user, value=100 + index)
        _wav(model, value=200 + index)
        record: dict[str, object] = {
            "record_id": f"record-{index}",
            "split_group": f"speaker-session-{index}",
            "user_wav": user.name,
            "model_wav": model.name,
            "assistant_text": f"The verified answer for example {index} is {index + 1}.",
            "text_timestamp_seconds": 0.02,
            "source": "project-owned consenting studio recording",
            "license": "project-owned-training-license-v1",
            "consent": True,
            "provenance": {
                "capture_session": f"session-{index}",
                "speaker_release": f"release-{index}",
            },
        }
        if with_frames:
            frame = root / f"selected-{index}.jpg"
            _jpeg(frame, value=120 + index)
            record["selected_frames"] = [
                {
                    "path": frame.name,
                    "timestamp_seconds": 0.01,
                    "difference_percent": 25.0,
                }
            ]
        records.append(record)
    return records


def _jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def test_preparer_builds_reader_compatible_deterministic_rights_bound_splits(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
) -> None:
    config_path = _model_config(tmp_path)
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, _records(tmp_path))
    options = NativePreparationConfig(
        validation_fraction=0.25,
        split_seed="stable-test-seed",
        max_duration_seconds=1.0,
    )
    first = prepare_native_dataset(
        input_path,
        tmp_path / "prepared-a",
        model_config_path=config_path,
        tokenizer_path=native_tokenizer.artifact_dir,
        config=options,
    )
    second = prepare_native_dataset(
        input_path,
        tmp_path / "prepared-b",
        model_config_path=config_path,
        tokenizer_path=native_tokenizer.artifact_dir,
        config=options,
    )

    assert (first.train_records, first.validation_records) == (3, 1)
    assert _tree_bytes(first.output_dir) == _tree_bytes(second.output_dir)
    assert first.train_manifest_sha256 == second.train_manifest_sha256
    assert first.validation_manifest_sha256 == second.validation_manifest_sha256

    model_config = MultimodalConfig.from_json(config_path)
    tokenizer_digest = str(native_tokenizer.metadata["tokenizer_sha256"])
    train = NativeShardDataset(
        first.train_manifest,
        split="train",
        config=model_config,
        tokenizer_sha256=tokenizer_digest,
    )
    validation = NativeShardDataset(
        first.validation_manifest,
        split="validation",
        config=model_config,
        tokenizer_sha256=tokenizer_digest,
    )
    assert not (train.content_hashes & validation.content_hashes)
    batch = train.load(0)
    assert batch.user_waveform.shape == (1, 960, 1)
    assert batch.frames is not None and batch.frames.shape == (1, 1, 192, 320, 3)
    tokens = np.asarray(batch.text_tokens)
    mask = np.asarray(batch.text_target_mask)
    assert tokens[0, 0] == native_tokenizer.bos_id and mask[0, 0] == 0.0
    assistant = int(np.where(tokens[0] == native_tokenizer.role_id("assistant"))[0][0])
    assert mask[0, assistant] == 1.0
    assert tokens[0, -1] == native_tokenizer.eos_id and mask[0, -1] == 1.0

    manifest = json.loads(first.train_manifest.read_text(encoding="utf-8"))
    record = manifest["records"][0]
    assert record["consent"] is True
    assert record["source"] == "project-owned consenting studio recording"
    prep = record["provenance"]["mlm_native_preparation"]
    assert prep["external_inference"] is False
    assert prep["pretrained_weights"] is False
    assert len(prep["user_wav_sha256"]) == 64
    assert prep["tokenizer_sha256"] == tokenizer_digest
    assert (
        prep["model_config_sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.__setitem__("consent", False), "affirmative consent"),
        (lambda record: record.__setitem__("license", "unknown"), "rights-cleared"),
        (
            lambda record: record.__setitem__(
                "provenance", {"note": "missing release"}
            ),
            "speaker_release and capture_session",
        ),
        (lambda record: record.__setitem__("user_wav", "../escape.wav"), "escapes"),
    ],
)
def test_preparer_fails_closed_on_rights_and_path_claims(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
    mutation,
    message: str,
) -> None:
    config_path = _model_config(tmp_path)
    records = _records(tmp_path, count=2, with_frames=False)
    mutation(records[0])
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    output = tmp_path / "prepared"
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        prepare_native_dataset(
            input_path,
            output,
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )
    assert not output.exists()


def test_preparer_rejects_component_media_overlap_across_groups(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
) -> None:
    config_path = _model_config(tmp_path)
    records = _records(tmp_path, count=2, with_frames=False)
    # The assistant track and transcript remain different. Reusing just one
    # source waveform must still be treated as train/validation leakage.
    records[1]["user_wav"] = records[0]["user_wav"]
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    output = tmp_path / "component-overlap"
    with pytest.raises(ValueError, match="component media overlap"):
        prepare_native_dataset(
            input_path,
            output,
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
            config=NativePreparationConfig(validation_fraction=0.5),
        )
    assert not output.exists()


def test_preparer_allows_canonical_silence_across_splits(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
) -> None:
    config_path = _model_config(tmp_path)
    records = _records(tmp_path, count=2, with_frames=False)
    _wav(tmp_path / "model-0.wav", value=0)
    _wav(tmp_path / "model-1.wav", value=0)
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    prepared = prepare_native_dataset(
        input_path,
        tmp_path / "shared-silence",
        model_config_path=config_path,
        tokenizer_path=native_tokenizer.artifact_dir,
        config=NativePreparationConfig(validation_fraction=0.5),
    )
    assert (prepared.train_records, prepared.validation_records) == (1, 1)


def test_preparer_hashes_the_exact_media_snapshot_it_decodes(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _model_config(tmp_path)
    records = _records(tmp_path, count=2, with_frames=False)
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    source = tmp_path / "user-0.wav"
    original_bytes = source.read_bytes()
    expected_hash = hashlib.sha256(original_bytes).hexdigest()
    original_reader = native_prepare_module._read_pcm_wav

    def read_then_mutate(path: Path, *, maximum_seconds: float):
        result = original_reader(path, maximum_seconds=maximum_seconds)
        if path == source:
            _wav(source, value=999)
        return result

    monkeypatch.setattr(native_prepare_module, "_read_pcm_wav", read_then_mutate)
    prepared = prepare_native_dataset(
        input_path,
        tmp_path / "snapshot-bound",
        model_config_path=config_path,
        tokenizer_path=native_tokenizer.artifact_dir,
        config=NativePreparationConfig(validation_fraction=0.5),
    )
    manifests = [prepared.train_manifest, prepared.validation_manifest]
    claims = [
        record
        for manifest_path in manifests
        for record in json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
        if record["record_id"] == "record-0"
    ]
    assert len(claims) == 1
    assert (
        claims[0]["provenance"]["mlm_native_preparation"]["user_wav_sha256"]
        == expected_hash
    )


@pytest.mark.skipif(
    not getattr(native_prepare_module.os, "O_NOFOLLOW", 0),
    reason="platform does not expose O_NOFOLLOW",
)
def test_preparer_rejects_media_replaced_by_symlink_after_manifest_parse(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _model_config(tmp_path)
    records = _records(tmp_path, count=2, with_frames=False)
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    replacement = tmp_path / "replacement.wav"
    _wav(replacement, value=777)
    original_split = native_prepare_module._split_groups

    def split_then_swap(parsed_records, config):
        assignment = original_split(parsed_records, config)
        victim = parsed_records[0].user_wav
        victim.unlink()
        victim.symlink_to(replacement)
        return assignment

    monkeypatch.setattr(native_prepare_module, "_split_groups", split_then_swap)
    with pytest.raises(ValueError, match="safe regular file"):
        prepare_native_dataset(
            input_path,
            tmp_path / "symlink-swap",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
            config=NativePreparationConfig(validation_fraction=0.5),
        )


def test_preparer_rejects_noncanonical_or_misaligned_media(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _model_config(tmp_path)
    records = _records(tmp_path, count=2)
    _wav(tmp_path / "model-0.wav", value=200, samples=480)
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    with pytest.raises(ValueError, match="equal samples"):
        prepare_native_dataset(
            input_path,
            tmp_path / "bad-audio",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )

    records = _records(tmp_path, count=2)
    _jpeg(tmp_path / "selected-0.jpg", width=160, height=96)
    _jsonl(input_path, records)
    with pytest.raises(ValueError, match="exactly 320x192"):
        prepare_native_dataset(
            input_path,
            tmp_path / "bad-frame",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )

    # A small JPEG can declare enormous decoded dimensions.  The preparer must
    # reject its SOF header before FFmpeg is allowed to allocate a frame.
    (tmp_path / "selected-0.jpg").write_bytes(
        b"\xff\xd8\xff\xc0\x00\x0b\x08\xff\xff\xff\xff\x01\x01\x11\x00\xff\xd9"
    )
    monkeypatch.setattr(
        native_prepare_module.av,
        "open",
        lambda *_args, **_kwargs: pytest.fail(
            "decoder was called before the JPEG dimension guard"
        ),
    )
    with pytest.raises(ValueError, match="exactly 320x192"):
        prepare_native_dataset(
            input_path,
            tmp_path / "oversized-frame-header",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )


def test_preparer_rejects_context_overflow_before_decoding_frames(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _model_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["thinker"]["context_length"] = 1
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    records = _records(tmp_path, count=2)
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    monkeypatch.setattr(
        native_prepare_module.av,
        "open",
        lambda *_args, **_kwargs: pytest.fail(
            "decoder was called before the context guard"
        ),
    )
    with pytest.raises(ValueError, match="thinker context length"):
        prepare_native_dataset(
            input_path,
            tmp_path / "context-overflow",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )


def test_preparer_rejects_unbounded_or_unknown_model_config(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
) -> None:
    config_path = _model_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["ignored_security_field"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    records = _records(tmp_path, count=2, with_frames=False)
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    with pytest.raises(ValueError, match="configuration fields"):
        prepare_native_dataset(
            input_path,
            tmp_path / "bad-config",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )

    config_path.write_bytes(b" " * (native_prepare_module.MAX_MODEL_CONFIG_BYTES + 1))
    with pytest.raises(ValueError, match="empty or too large"):
        prepare_native_dataset(
            input_path,
            tmp_path / "oversized-config",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )


def test_preparer_rejects_duplicate_selected_frames(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
) -> None:
    config_path = _model_config(tmp_path)
    records = _records(tmp_path, count=2)
    first_frames = records[0]["selected_frames"]
    assert isinstance(first_frames, list)
    first_frames.append(
        {
            "path": "selected-0.jpg",
            "timestamp_seconds": 0.015,
            "difference_percent": 30.0,
        }
    )
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    with pytest.raises(ValueError, match="paths must be unique"):
        prepare_native_dataset(
            input_path,
            tmp_path / "duplicate-frame",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )


def test_preparer_requires_group_split_and_never_overwrites(
    tmp_path: Path,
    native_tokenizer: ScratchTokenizer,
) -> None:
    config_path = _model_config(tmp_path)
    records = _records(tmp_path, count=2, with_frames=False)
    records[1]["split_group"] = records[0]["split_group"]
    input_path = tmp_path / "records.jsonl"
    _jsonl(input_path, records)
    with pytest.raises(ValueError, match="at least two split groups"):
        prepare_native_dataset(
            input_path,
            tmp_path / "one-group",
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )

    _jsonl(input_path, _records(tmp_path, count=2, with_frames=False))
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_native_dataset(
            input_path,
            output,
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )
    assert marker.read_text(encoding="utf-8") == "mine"

    locked_output = tmp_path / "locked"
    lock = tmp_path / ".locked.prepare.lock"
    lock.write_text("another preparer owns this", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already being prepared"):
        prepare_native_dataset(
            input_path,
            locked_output,
            model_config_path=config_path,
            tokenizer_path=native_tokenizer.artifact_dir,
        )
    assert not locked_output.exists()
    assert lock.read_text(encoding="utf-8") == "another preparer owns this"

    racing_staging = tmp_path / "racing-staging"
    racing_staging.mkdir()
    (racing_staging / "new.txt").write_text("new", encoding="utf-8")
    racing_destination = tmp_path / "racing-destination"
    racing_destination.mkdir()
    with pytest.raises(FileExistsError):
        native_prepare_module._publish_directory_no_replace(
            racing_staging, racing_destination
        )
    assert (racing_staging / "new.txt").read_text(encoding="utf-8") == "new"
    assert racing_destination.is_dir() and not any(racing_destination.iterdir())
