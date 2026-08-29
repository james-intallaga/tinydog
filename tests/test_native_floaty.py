from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from mini_arithmetic.community_updates import TensorSpec
from mini_arithmetic.foundation_floaty import FloatyConfig, FoundationFloatyLM
from mini_arithmetic.hybrid import ATTENTION, CONV, HybridConfig
from mini_arithmetic.multimodal import (
    AudioCodecConfig,
    MultimodalConfig,
    ScratchMultimodalLM,
    TalkerConfig,
    TimedAudio,
    TimedText,
    TimedVideo,
    VideoEncoderConfig,
)
from mini_arithmetic.native_floaty import attach_floaty_thinker


def tiny_native_config() -> MultimodalConfig:
    return MultimodalConfig(
        thinker=HybridConfig(
            vocab_size=300,
            context_length=128,
            model_dim=16,
            ffn_dim=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            conv_kernel_size=3,
            layer_schedule=(CONV, ATTENTION),
        ),
        audio=AudioCodecConfig(
            frame_size=8,
            hidden_dim=16,
            latent_dim=8,
            temporal_kernel_size=3,
            num_codebooks=2,
            codebook_size=16,
        ),
        video=VideoEncoderConfig(
            patch_size=2,
            spatial_merge_size=2,
            vision_dim=8,
            resampler_tokens=4,
            resampler_dim=8,
            resampler_heads=2,
            model_dim=16,
        ),
        talker=TalkerConfig(
            hidden_dim=12,
            num_layers=2,
            temporal_kernel_size=3,
        ),
        time_scale_seconds=10.0,
    )


def flat_parameters(module: nn.Module) -> dict[str, mx.array]:
    return dict(tree_flatten(module.parameters()))


def activate_floaty(model: ScratchMultimodalLM, amount: float = 0.02) -> None:
    assert type(model.thinker) is FoundationFloatyLM
    state = model.thinker.export_floaty_state()
    tensors = {
        name: (
            mx.full(value.shape, amount, dtype=value.dtype)
            if name.endswith("down.weight")
            else mx.array(value)
        )
        for name, value in state.tensors.items()
    }
    model.thinker.import_floaty_state(replace(state, tensors=tensors))


def test_native_attachment_registers_owner_once_and_only_floaty_is_trainable() -> None:
    mx.random.seed(301)
    model = ScratchMultimodalLM(tiny_native_config())
    original = flat_parameters(model)
    original_ids = {id(value) for value in original.values()}
    original_count = model.parameter_count()

    boundary = attach_floaty_thinker(model, FloatyConfig(rank=2))
    partition = boundary.assert_parameter_partition()
    attached = flat_parameters(model)
    owner_ids = {id(attached[name]) for name in partition.foundation}
    floaty_ids = {id(attached[name]) for name in partition.floaty}

    assert type(model.thinker) is FoundationFloatyLM
    assert boundary.model is model
    assert owner_ids == original_ids
    assert owner_ids.isdisjoint(floaty_ids)
    assert len(attached) == len({id(value) for value in attached.values()})
    assert boundary.foundation_parameter_count() == original_count
    assert model.parameter_count() == original_count + boundary.floaty_parameter_count()
    assert boundary.trainable_parameter_names() == partition.floaty
    assert tuple(
        sorted(name for name, _ in tree_flatten(boundary.floaty_parameters()))
    ) == (
        "blocks.0.feed_forward.down.weight",
        "blocks.0.feed_forward.up.weight",
        "blocks.0.operator.down.weight",
        "blocks.0.operator.up.weight",
        "blocks.1.feed_forward.down.weight",
        "blocks.1.feed_forward.up.weight",
        "blocks.1.operator.down.weight",
        "blocks.1.operator.up.weight",
    )

    owner_prefixes = {
        "thinker.foundation",
        "audio_codec",
        "video_encoder",
        "audio_projection",
        "audio_stream_embedding",
        "time_input",
        "time_output",
        "talker",
    }
    assert all(
        any(name.startswith(prefix) for name in partition.foundation)
        for prefix in owner_prefixes
    )
    assert all(name.startswith("thinker.floaty.") for name in partition.floaty)
    assert not tree_flatten(model.audio_codec.trainable_parameters())
    assert not tree_flatten(model.video_encoder.trainable_parameters())
    assert not tree_flatten(model.talker.trainable_parameters())

    report = model.parameter_report()
    assert (
        sum(value for name, value in report.items() if name != "total")
        == report["total"]
    )
    assert report["total"] == model.parameter_count()


def test_attachment_preserves_native_text_timeline_and_cache_apis() -> None:
    mx.random.seed(302)
    model = ScratchMultimodalLM(tiny_native_config())
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    events = [TimedText(timestamp=0.1, tokens=tokens)]
    original_text = model(tokens)
    original_forward_text = model.forward_text(tokens)
    original_timeline = model.forward_timeline(events)
    mx.eval(original_text, original_forward_text, original_timeline)

    attach_floaty_thinker(model, FloatyConfig(rank=2))
    attached_text = model(tokens)
    attached_forward_text = model.forward_text(tokens)
    attached_timeline = model.forward_timeline(events)
    first, cache = model.thinker.forward_cached(tokens[:, :2])
    second, cache = model.thinker.forward_cached(tokens[:, 2:], cache)
    cached = mx.concatenate((first, second), axis=1)
    mx.eval(
        original_text,
        original_forward_text,
        original_timeline,
        attached_text,
        attached_forward_text,
        attached_timeline,
        cached,
    )

    assert mx.array_equal(original_text, attached_text).item()
    assert mx.array_equal(original_forward_text, attached_forward_text).item()
    assert mx.array_equal(
        original_timeline.text_logits,
        attached_timeline.text_logits,
    ).item()
    assert mx.allclose(attached_text, cached, atol=2e-5, rtol=2e-5).item()
    assert cache.offset == tokens.shape[1]

    activate_floaty(model)
    changed_text = model(tokens)
    changed_timeline = model.forward_timeline(events)
    mx.eval(changed_text, changed_timeline)
    assert not mx.array_equal(original_text, changed_text).item()
    assert not mx.array_equal(
        original_timeline.text_logits,
        changed_timeline.text_logits,
    ).item()


def test_native_audio_video_gradients_are_exactly_thinker_floaty() -> None:
    mx.random.seed(303)
    model = ScratchMultimodalLM(tiny_native_config())
    boundary = attach_floaty_thinker(model, FloatyConfig(rank=2))
    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    codes = mx.array([[[1, 2], [3, 4]]], dtype=mx.int32)
    events = [
        TimedText(timestamp=0.0, tokens=tokens),
        TimedVideo(
            frames=mx.random.uniform(shape=(1, 2, 8, 8, 3)),
            timestamps=mx.array([0.1, 0.2]),
            difference_scores=mx.array([0.0, 0.8]),
        ),
        TimedAudio(timestamp=0.3, codes=codes, stream="user"),
    ]

    def loss_fn(current: ScratchMultimodalLM) -> mx.array:
        output = current.forward_timeline(
            events,
            user_codes=codes,
            model_codes=codes,
        )
        assert output.audio is not None
        return mx.mean(output.text_logits) + mx.mean(output.audio.model_logits)

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, gradients)
    gradient_names = tuple(sorted(name for name, _ in tree_flatten(gradients)))

    assert mx.isfinite(loss).item()
    assert gradient_names == boundary.floaty_parameter_names()
    assert all(name.startswith("thinker.floaty.") for name in gradient_names)
    assert not any(
        owner in name
        for name in gradient_names
        for owner in ("audio_codec", "video_encoder", "talker", "time_")
    )

    model.audio_codec.unfreeze()
    with pytest.raises(RuntimeError, match="owner-foundation"):
        boundary.assert_parameter_partition()
    boundary.freeze_foundation()
    assert boundary.trainable_parameter_names() == boundary.floaty_parameter_names()


def test_external_namespace_mapping_is_queue_safe_and_deterministic() -> None:
    model = ScratchMultimodalLM(tiny_native_config())
    boundary = attach_floaty_thinker(
        model,
        FloatyConfig(rank=2, attachments=("operator",), target_layers=(1,)),
    )
    mapping = boundary.community_to_native_parameter_names()
    manifest = boundary.config_manifest()
    tensor_contract = manifest["thinker_floaty"]["tensors"]
    specs = tuple(
        TensorSpec(
            name=tensor["name"],
            shape=tuple(tensor["shape"]),
            dtype=tensor["dtype"],
        )
        for tensor in tensor_contract
    )

    assert mapping == {
        "floaty.blocks.1.operator.down.weight": "thinker.floaty.blocks.1.operator.down.weight",
        "floaty.blocks.1.operator.up.weight": "thinker.floaty.blocks.1.operator.up.weight",
    }
    assert tuple(mapping) == boundary.queue_floaty_parameter_names()
    assert tuple(mapping.values()) == boundary.floaty_parameter_names()
    assert all(spec.name.startswith("floaty.") for spec in specs)
    assert not any(
        forbidden in spec.name
        for spec in specs
        for forbidden in ("thinker.", "audio_codec", "video_encoder", "talker", "time_")
    )
    assert manifest["community_to_native"] == [
        {"community_name": community, "native_name": native}
        for community, native in mapping.items()
    ]

    manifest["community_to_native"][0]["native_name"] = "caller mutation"
    assert boundary.config_manifest()["community_to_native"][0][
        "native_name"
    ].startswith("thinker.floaty.")


def test_native_foundation_hash_includes_all_owner_and_excludes_floaty_bytes() -> None:
    mx.random.seed(304)
    model = ScratchMultimodalLM(tiny_native_config())
    boundary = attach_floaty_thinker(model, FloatyConfig(rank=2))
    initial_foundation_hash = boundary.foundation_hash()
    initial_config_hash = boundary.config_hash()

    activate_floaty(model, amount=0.04)
    assert boundary.foundation_hash() == initial_foundation_hash
    assert boundary.config_hash() == initial_config_hash

    model.audio_projection.weight = model.audio_projection.weight + mx.array(1e-4)
    mx.eval(model.audio_projection.parameters())
    assert boundary.foundation_hash() != initial_foundation_hash
    assert boundary.config_hash() == initial_config_hash


def test_native_attachment_is_idempotent_but_rejects_config_replacement() -> None:
    model = ScratchMultimodalLM(tiny_native_config())
    first = attach_floaty_thinker(model, FloatyConfig(rank=2))
    first_parameter_ids = {
        name: id(value) for name, value in flat_parameters(model).items()
    }
    second = attach_floaty_thinker(model, FloatyConfig(rank=2))

    assert second.thinker is first.thinker
    assert {
        name: id(value) for name, value in flat_parameters(model).items()
    } == first_parameter_ids
    with pytest.raises(ValueError, match="different"):
        attach_floaty_thinker(model, FloatyConfig(rank=3))
