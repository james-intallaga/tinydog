from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from mini_arithmetic.multimodal import ScratchMultimodalLM
from mini_arithmetic.native_objectives import (
    NativeConversationBatch,
    NativeLossWeights,
    _causal_audio_conditioning,
    native_conversation_loss,
)
from test_multimodal import tiny_multimodal_config


def _batch(*, with_video: bool = True) -> NativeConversationBatch:
    frames = mx.random.uniform(shape=(1, 2, 8, 8, 3)) if with_video else None
    return NativeConversationBatch(
        user_waveform=mx.random.uniform(low=-0.5, high=0.5, shape=(1, 24, 1)),
        model_waveform=mx.random.uniform(low=-0.5, high=0.5, shape=(1, 24, 1)),
        text_tokens=mx.array([[1, 7, 8, 2]], dtype=mx.int32),
        text_target_mask=mx.array([[1, 1, 1, 1]], dtype=mx.float32),
        text_timestamp=0.0004,
        frames=frames,
        frame_timestamps=mx.array([0.0001, 0.0003]) if with_video else None,
        frame_difference_scores=mx.array([0.0, 0.7]) if with_video else None,
    )


def test_stage_weights_are_explicit_and_never_all_zero() -> None:
    assert NativeLossWeights.for_stage("audio_codec").codec == 1.0
    assert NativeLossWeights.for_stage("conversation").model_audio == 1.0
    assert NativeLossWeights.for_stage("vision").text == 1.0
    with pytest.raises(ValueError, match="at least one"):
        NativeLossWeights(0.0, 0.0, 0.0, 0.0)


def test_native_joint_loss_is_finite_and_differentiates_all_model_components() -> None:
    mx.random.seed(71)
    model = ScratchMultimodalLM(tiny_multimodal_config())
    batch = _batch(with_video=True)
    weights = NativeLossWeights.for_stage("joint")

    def loss_fn(current: ScratchMultimodalLM) -> mx.array:
        return native_conversation_loss(current, batch, weights).total

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    result = native_conversation_loss(model, batch, weights)
    mx.eval(loss, gradients, result)
    assert mx.isfinite(loss).item()
    assert result.text_targets == 4
    assert result.audio_targets == 6
    assert all(mx.isfinite(value).all().item() for _, value in tree_flatten(gradients))
    gradient_names = {name for name, _ in tree_flatten(gradients)}
    assert any(name.startswith("thinker.") for name in gradient_names)
    assert any(name.startswith("audio_codec.") for name in gradient_names)
    assert any(name.startswith("video_encoder.") for name in gradient_names)
    assert any(name.startswith("talker.") for name in gradient_names)


def test_codec_stage_has_no_language_or_talker_contribution() -> None:
    mx.random.seed(72)
    model = ScratchMultimodalLM(tiny_multimodal_config())
    result = native_conversation_loss(
        model,
        _batch(with_video=False),
        NativeLossWeights.for_stage("audio_codec"),
    )
    mx.eval(result)
    assert mx.allclose(result.total, result.codec).item()


def test_native_batch_requires_aligned_duplex_audio_and_complete_video_metadata() -> (
    None
):
    model = ScratchMultimodalLM(tiny_multimodal_config())
    bad_audio = _batch(with_video=False)
    bad_audio = NativeConversationBatch(
        **{
            **bad_audio.__dict__,
            "model_waveform": mx.zeros((1, 16, 1)),
        }
    )
    with pytest.raises(ValueError, match="same batch and samples"):
        native_conversation_loss(model, bad_audio, NativeLossWeights.for_stage("joint"))

    missing_times = _batch(with_video=True)
    missing_times = NativeConversationBatch(
        **{**missing_times.__dict__, "frame_timestamps": None}
    )
    with pytest.raises(ValueError, match="supplied together"):
        native_conversation_loss(
            model, missing_times, NativeLossWeights.for_stage("joint")
        )


def test_audio_conditioning_uses_only_state_before_each_current_user_frame() -> None:
    hidden = mx.arange(7, dtype=mx.float32).reshape(1, 7, 1)
    kinds = (
        "audio_user_start",
        "audio_user",
        "video_frame",
        "video",
        "audio_user",
        "text",
        "audio_user_end",
    )
    conditioning = _causal_audio_conditioning(hidden, kinds, audio_frames=2)
    mx.eval(conditioning)
    # Frame zero sees only the start marker. Frame one may see intervening
    # video, but never its own user-audio embedding at offset four.
    assert conditioning.tolist() == [[[0.0], [3.0]]]


def test_native_batch_rejects_media_timestamps_outside_its_audio_window() -> None:
    model = ScratchMultimodalLM(tiny_multimodal_config())
    batch = _batch(with_video=False)
    invalid = NativeConversationBatch(**{**batch.__dict__, "text_timestamp": 1.0})
    with pytest.raises(ValueError, match="aligned audio window"):
        invalid.validate(model)
