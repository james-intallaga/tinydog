from __future__ import annotations

import inspect
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from mini_arithmetic.hybrid import ATTENTION, CONV, HybridConfig
from mini_arithmetic.multimodal import (
    AudioCodecConfig,
    DualStreamAudioTalker,
    ModalityTokenRegistry,
    MultimodalConfig,
    ScratchMultimodalLM,
    SelectedFrameVideoEncoder,
    StreamingWaveformCodec,
    TalkerConfig,
    TimedAudio,
    TimedInterruption,
    TimedText,
    TimedVideo,
    VideoEncoderConfig,
)
from mini_arithmetic.scratch_tokenizer import EXTENSION_SLOT_COUNT, extension_slot_id


def tiny_audio_config() -> AudioCodecConfig:
    return AudioCodecConfig(
        frame_size=8,
        hidden_dim=16,
        latent_dim=8,
        temporal_kernel_size=3,
        num_codebooks=2,
        codebook_size=16,
    )


def tiny_video_config() -> VideoEncoderConfig:
    return VideoEncoderConfig(
        patch_size=2,
        spatial_merge_size=2,
        vision_dim=8,
        resampler_tokens=4,
        resampler_dim=8,
        resampler_heads=2,
        model_dim=16,
    )


def tiny_multimodal_config() -> MultimodalConfig:
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
        audio=tiny_audio_config(),
        video=tiny_video_config(),
        talker=TalkerConfig(hidden_dim=12, num_layers=2, temporal_kernel_size=3),
        time_scale_seconds=10.0,
    )


def test_registry_uses_only_frozen_extension_ids_and_config_round_trips(
    tmp_path: Path,
) -> None:
    registry = ModalityTokenRegistry()
    values = list(registry.to_dict().values())[1:]
    assert values == [extension_slot_id(index) for index in range(8)]
    assert all(
        extension_slot_id(0) <= value < extension_slot_id(EXTENSION_SLOT_COUNT - 1) + 1
        for value in values
    )

    config = tiny_multimodal_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    loaded = MultimodalConfig.from_json(path)
    assert loaded == config
    repository_config = Path(__file__).parents[1] / "configs" / "multimodal-tiny.json"
    assert MultimodalConfig.from_json(repository_config) == config


def test_committed_native_pilot_has_the_claimed_exact_parameter_count() -> None:
    path = Path(__file__).parents[1] / "configs" / "multimodal-pilot-31m.json"
    model = ScratchMultimodalLM(MultimodalConfig.from_json(path))
    assert model.thinker.parameter_count() == 29_851_776
    assert model.parameter_count() == 31_004_864


def test_random_codec_shapes_losses_and_codebooks() -> None:
    mx.random.seed(1)
    config = tiny_audio_config()
    codec = StreamingWaveformCodec(config)
    waveform = mx.random.normal((2, 21))
    output = codec(waveform)
    mx.eval(output)

    assert output.encoding.codes.shape == (2, 3, config.num_codebooks)
    assert output.encoding.latents.shape == (2, 3, config.latent_dim)
    assert output.encoding.quantized.shape == output.encoding.latents.shape
    assert output.reconstruction.shape == (2, 21, 1)
    assert output.encoding.padded_samples == 3
    assert mx.isfinite(output.losses.total).item()
    assert output.losses.total.item() >= output.losses.reconstruction.item()
    assert int(mx.min(output.encoding.codes).item()) >= 0
    assert int(mx.max(output.encoding.codes).item()) < config.codebook_size

    def loss_fn(current: StreamingWaveformCodec) -> mx.array:
        return current(waveform).losses.total

    loss, gradients = nn.value_and_grad(codec, loss_fn)(codec)
    mx.eval(loss, gradients)
    assert mx.isfinite(loss).item()
    assert sum(value.size for _, value in tree_flatten(gradients)) == sum(
        value.size for _, value in tree_flatten(codec.parameters())
    )


def test_default_codec_quantum_is_one_twenty_millisecond_media_block() -> None:
    config = AudioCodecConfig()
    assert config.frame_size == 480
    assert config.frame_rate == 50.0


def test_codec_chunked_stream_matches_full_and_preserves_state() -> None:
    mx.random.seed(2)
    config = tiny_audio_config()
    codec = StreamingWaveformCodec(config)
    waveform = mx.random.normal((1, 29, 1))
    full = codec.encode(waveform)

    encode_state = codec.init_stream_state(1)
    parts = []
    starts = (0, 5, 17)
    stops = (5, 17, 29)
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        parts.append(
            codec.encode_stream(
                waveform[:, start:stop, :],
                encode_state,
                final=index == len(starts) - 1,
            )
        )
    streamed_codes = mx.concatenate([part.codes for part in parts], axis=1)
    streamed_latents = mx.concatenate([part.latents for part in parts], axis=1)
    mx.eval(full.codes, full.latents, streamed_codes, streamed_latents)
    assert mx.array_equal(full.codes, streamed_codes).item()
    assert mx.allclose(full.latents, streamed_latents, atol=1e-6).item()
    assert encode_state.encoder_finalized is True
    assert encode_state.samples_seen == waveform.shape[1]
    assert encode_state.frames_encoded == full.codes.shape[1]
    assert encode_state.encoder_buffer.shape[1] == 0

    full_decoded = codec.decode(full)
    decode_state = codec.init_stream_state(1)
    first = codec.decode_codes_stream(full.codes[:, :2, :], decode_state)
    second = codec.decode_codes_stream(full.codes[:, 2:, :], decode_state)
    streamed_decoded = mx.concatenate((first, second), axis=1)[
        :, : waveform.shape[1], :
    ]
    mx.eval(full_decoded, streamed_decoded)
    assert mx.allclose(full_decoded, streamed_decoded, atol=1e-6).item()
    assert decode_state.frames_decoded == full.codes.shape[1]
    assert decode_state.decoder_temporal.values is not None


def test_codec_encoder_and_decoder_do_not_leak_future_frames() -> None:
    mx.random.seed(3)
    config = tiny_audio_config()
    codec = StreamingWaveformCodec(config)
    prefix = mx.random.normal((1, 3 * config.frame_size, 1))
    first_waveform = mx.concatenate(
        (prefix, mx.zeros((1, config.frame_size, 1))), axis=1
    )
    second_waveform = mx.concatenate(
        (prefix, mx.ones((1, config.frame_size, 1))), axis=1
    )
    first_encoding = codec.encode(first_waveform)
    second_encoding = codec.encode(second_waveform)
    mx.eval(first_encoding.latents, second_encoding.latents)
    assert mx.allclose(
        first_encoding.latents[:, :3], second_encoding.latents[:, :3], atol=1e-6
    ).item()

    first_codes = mx.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], dtype=mx.int32)
    second_codes = mx.array([[[1, 2], [3, 4], [9, 10], [11, 12]]], dtype=mx.int32)
    first_state = codec.init_stream_state(1)
    second_state = codec.init_stream_state(1)
    first_audio = codec.decode_codes_stream(first_codes, first_state)
    second_audio = codec.decode_codes_stream(second_codes, second_state)
    mx.eval(first_audio, second_audio)
    prefix_samples = 2 * config.frame_size
    assert mx.allclose(
        first_audio[:, :prefix_samples], second_audio[:, :prefix_samples], atol=1e-6
    ).item()

    with pytest.raises(ValueError, match="audio codebook IDs"):
        codec.quantizer.decode_codes(
            mx.array([[[config.codebook_size, 0]]], dtype=mx.int32)
        )


def test_selected_video_frames_include_metadata_and_remain_frame_causal() -> None:
    mx.random.seed(4)
    config = tiny_video_config()
    encoder = SelectedFrameVideoEncoder(config)
    prefix = mx.random.uniform(shape=(1, 2, 8, 8, 3))
    first_frames = mx.concatenate((prefix, mx.zeros((1, 1, 8, 8, 3))), axis=1)
    second_frames = mx.concatenate((prefix, mx.ones((1, 1, 8, 8, 3))), axis=1)
    timestamps = mx.array([0.0, 0.4, 1.1])
    first = encoder(first_frames, timestamps)
    second = encoder(second_frames, timestamps)
    mx.eval(first, second)

    assert first.embeddings.shape == (1, 3, 4, config.model_dim)
    assert first.timestamps.shape == (1, 3, 4)
    assert first.difference_scores.shape == (1, 3)
    assert mx.allclose(
        first.embeddings[:, :2], second.embeddings[:, :2], atol=1e-6
    ).item()
    assert not mx.allclose(first.embeddings[:, 2], second.embeddings[:, 2]).item()


def test_video_resampler_has_a_fixed_token_budget_across_camera_resolutions() -> None:
    mx.random.seed(41)
    config = tiny_video_config()
    encoder = SelectedFrameVideoEncoder(config)
    small = encoder(mx.random.uniform(shape=(1, 1, 8, 8, 3)), mx.array([0.0]))
    larger = encoder(mx.random.uniform(shape=(1, 1, 12, 16, 3)), mx.array([0.0]))
    mx.eval(small, larger)
    assert small.embeddings.shape == (1, 1, config.resampler_tokens, config.model_dim)
    assert larger.embeddings.shape == small.embeddings.shape
    assert small.tokens_per_frame == config.resampler_tokens
    assert larger.tokens_per_frame == config.resampler_tokens


def test_dual_stream_talker_is_causal_cached_and_interruptible() -> None:
    mx.random.seed(5)
    audio = tiny_audio_config()
    talker = DualStreamAudioTalker(
        thinker_dim=16,
        audio_config=audio,
        config=TalkerConfig(hidden_dim=12, num_layers=2, temporal_kernel_size=3),
    )
    conditioning = mx.random.normal((1, 5, 16))
    user_codes = mx.array([[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]], dtype=mx.int32)
    model_codes = mx.array([[[2, 1], [4, 3], [6, 5], [8, 7], [10, 9]]], dtype=mx.int32)
    changed_conditioning = mx.concatenate(
        (conditioning[:, :3], mx.random.normal((1, 2, 16))), axis=1
    )
    changed_user = mx.concatenate(
        (user_codes[:, :3], mx.array([[[11, 12], [13, 14]]], dtype=mx.int32)), axis=1
    )
    changed_model = mx.concatenate(
        (model_codes[:, :3], mx.array([[[12, 11], [14, 13]]], dtype=mx.int32)), axis=1
    )
    first = talker(conditioning, user_codes, model_codes)
    second = talker(changed_conditioning, changed_user, changed_model)
    mx.eval(first, second)
    assert first.user_logits.shape == (1, 5, audio.num_codebooks, audio.codebook_size)
    assert first.model_logits.shape == first.user_logits.shape
    assert mx.allclose(
        first.model_logits[:, :3], second.model_logits[:, :3], atol=1e-6
    ).item()

    state = talker.init_stream_state(1)
    part_one, state = talker.forward_stream(
        conditioning[:, :2], user_codes[:, :2], model_codes[:, :2], state
    )
    part_two, state = talker.forward_stream(
        conditioning[:, 2:], user_codes[:, 2:], model_codes[:, 2:], state
    )
    cached = mx.concatenate((part_one.model_logits, part_two.model_logits), axis=1)
    mx.eval(first.model_logits, cached)
    assert mx.allclose(first.model_logits, cached, atol=1e-6).item()
    assert state.offset == 5
    assert state.previous_user_codes is not None
    assert state.previous_model_codes is not None

    talker.request_interruption(state)
    assert not state.model_active.item()
    continued, state = talker.forward_stream(
        conditioning[:, :1], user_codes[:, :1], model_codes[:, :1], state
    )
    assert continued.user_logits.shape[1] == 1
    assert not continued.model_active.item()
    talker.resume(state)
    assert state.model_active.item()


def test_talker_training_logits_never_see_same_step_user_codes() -> None:
    mx.random.seed(51)
    audio = tiny_audio_config()
    talker = DualStreamAudioTalker(
        thinker_dim=16,
        audio_config=audio,
        config=TalkerConfig(hidden_dim=12, num_layers=2, temporal_kernel_size=3),
    )
    conditioning = mx.random.normal((1, 5, 16))
    user_codes = mx.array([[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]], dtype=mx.int32)
    model_codes = mx.array([[[2, 1], [4, 3], [6, 5], [8, 7], [10, 9]]], dtype=mx.int32)
    baseline = talker(conditioning, user_codes, model_codes)

    for frame in range(user_codes.shape[1]):
        replacement = (user_codes[:, frame : frame + 1, :] + 7) % audio.codebook_size
        changed_user = mx.concatenate(
            (
                user_codes[:, :frame, :],
                replacement,
                user_codes[:, frame + 1 :, :],
            ),
            axis=1,
        )
        changed = talker(conditioning, changed_user, model_codes)
        mx.eval(baseline, changed)
        # Both training heads at frame t are based on frames < t. The current
        # user target cannot leak through the shared hidden representation.
        assert mx.allclose(
            baseline.user_logits[:, : frame + 1],
            changed.user_logits[:, : frame + 1],
            atol=1e-6,
        ).item()
        assert mx.allclose(
            baseline.model_logits[:, : frame + 1],
            changed.model_logits[:, : frame + 1],
            atol=1e-6,
        ).item()


def test_talker_generate_stream_is_greedy_finite_and_commits_selected_codes() -> None:
    mx.random.seed(52)
    audio = tiny_audio_config()
    talker = DualStreamAudioTalker(
        thinker_dim=16,
        audio_config=audio,
        config=TalkerConfig(hidden_dim=12, num_layers=2, temporal_kernel_size=3),
    )
    conditioning = mx.random.normal((2, 4, 16))
    user_codes = mx.array(
        [
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[8, 7], [6, 5], [4, 3], [2, 1]],
        ],
        dtype=mx.int32,
    )

    first, first_state = talker.generate_stream(
        conditioning, user_codes, talker.init_stream_state(2)
    )
    second, second_state = talker.generate_stream(
        conditioning, user_codes, talker.init_stream_state(2)
    )
    mx.eval(first, second)

    expected_shape = (2, 4, audio.num_codebooks)
    assert first.codes.shape == expected_shape
    assert first.model_logits.shape == (*expected_shape, audio.codebook_size)
    assert first.user_logits.shape == first.model_logits.shape
    assert first.hidden.shape == (2, 4, 12)
    assert mx.array_equal(first.codes, second.codes).item()
    assert mx.array_equal(first.model_logits, second.model_logits).item()
    assert mx.isfinite(first.model_logits).all().item()
    assert mx.isfinite(first.hidden).all().item()
    assert int(mx.min(first.codes).item()) >= 0
    assert int(mx.max(first.codes).item()) < audio.codebook_size
    codec = StreamingWaveformCodec(audio)
    decoded = codec.decode_codes_stream(first.codes, codec.init_stream_state(2))
    mx.eval(decoded)
    assert decoded.shape == (2, 4 * audio.frame_size, audio.channels)
    assert mx.isfinite(decoded).all().item()
    assert first_state.offset == 4
    assert second_state.offset == 4
    assert mx.array_equal(
        first_state.previous_model_codes, first.codes[:, -1, :]
    ).item()
    assert mx.array_equal(first_state.previous_user_codes, user_codes[:, -1, :]).item()


def test_talker_generate_stream_matches_chunks_and_cannot_see_future_user_codes() -> (
    None
):
    mx.random.seed(53)
    audio = tiny_audio_config()
    talker = DualStreamAudioTalker(
        thinker_dim=16,
        audio_config=audio,
        config=TalkerConfig(hidden_dim=12, num_layers=2, temporal_kernel_size=3),
    )
    conditioning = mx.random.normal((1, 6, 16))
    user_codes = mx.array(
        [[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]],
        dtype=mx.int32,
    )

    full, _ = talker.generate_stream(
        conditioning, user_codes, talker.init_stream_state(1)
    )
    chunk_state = talker.init_stream_state(1)
    first_chunk, chunk_state = talker.generate_stream(
        conditioning[:, :2], user_codes[:, :2], chunk_state
    )
    second_chunk, chunk_state = talker.generate_stream(
        conditioning[:, 2:], user_codes[:, 2:], chunk_state
    )
    chunked_codes = mx.concatenate((first_chunk.codes, second_chunk.codes), axis=1)
    chunked_logits = mx.concatenate(
        (first_chunk.model_logits, second_chunk.model_logits), axis=1
    )

    changed_user = mx.concatenate(
        (
            user_codes[:, :3],
            mx.array([[[13, 14], [14, 13], [15, 15]]], dtype=mx.int32),
        ),
        axis=1,
    )
    changed, _ = talker.generate_stream(
        conditioning, changed_user, talker.init_stream_state(1)
    )
    mx.eval(full, chunked_codes, chunked_logits, changed)

    assert mx.array_equal(full.codes, chunked_codes).item()
    assert mx.allclose(full.model_logits, chunked_logits, atol=1e-6).item()
    # User frame 3 is not an input until generation of frame 4, so even its
    # same-step model output is invariant.
    assert mx.array_equal(full.codes[:, :4], changed.codes[:, :4]).item()
    assert mx.allclose(
        full.model_logits[:, :4], changed.model_logits[:, :4], atol=1e-6
    ).item()
    assert chunk_state.offset == 6


def test_talker_generate_stream_rejects_bad_shapes_ids_state_and_sampling() -> None:
    audio = tiny_audio_config()
    talker = DualStreamAudioTalker(
        thinker_dim=16,
        audio_config=audio,
        config=TalkerConfig(hidden_dim=12, num_layers=2, temporal_kernel_size=3),
    )
    conditioning = mx.zeros((1, 2, 16))
    valid_codes = mx.zeros((1, 2, audio.num_codebooks), dtype=mx.int32)

    with pytest.raises(ValueError, match="Only deterministic greedy"):
        talker.generate_stream(
            conditioning,
            valid_codes,
            talker.init_stream_state(1),
            sampling="temperature",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="user audio IDs"):
        talker.generate_stream(
            conditioning,
            mx.full(valid_codes.shape, audio.codebook_size, dtype=mx.int32),
            talker.init_stream_state(1),
        )
    with pytest.raises(ValueError, match="user_codes must have shape"):
        talker.generate_stream(
            conditioning,
            mx.zeros((1, 2), dtype=mx.int32),
            talker.init_stream_state(1),
        )
    bad_state = talker.init_stream_state(1)
    bad_state.previous_model_codes = mx.array(
        [[audio.codebook_size, 0]], dtype=mx.int32
    )
    with pytest.raises(ValueError, match="previous model audio IDs"):
        talker.generate_stream(conditioning, valid_codes, bad_state)


def test_wrapper_preserves_text_api_and_orders_modalities_by_time() -> None:
    mx.random.seed(6)
    config = tiny_multimodal_config()
    model = ScratchMultimodalLM(config)
    text_tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    direct = model(text_tokens)
    original = model.thinker(text_tokens)
    mx.eval(direct, original)
    assert mx.array_equal(direct, original).item()

    audio_codes = mx.array([[[1, 2], [3, 4]]], dtype=mx.int32)
    events = [
        TimedAudio(timestamp=0.4, codes=audio_codes, stream="user"),
        TimedVideo(
            frames=mx.random.uniform(shape=(1, 2, 8, 8, 3)),
            timestamps=mx.array([0.2, 0.5]),
            difference_scores=mx.array([0.0, 0.8]),
        ),
        TimedText(timestamp=0.0, tokens=text_tokens),
        TimedInterruption(timestamp=0.45),
    ]
    output = model.forward_timeline(
        events,
        user_codes=audio_codes,
        model_codes=audio_codes,
    )
    mx.eval(output)
    assert output.text_logits.shape[:2] == output.timeline.embeddings.shape[:2]
    assert output.text_logits.shape[-1] == config.thinker.vocab_size
    assert output.audio is not None
    assert output.audio.model_logits.shape == (1, 2, 2, 16)
    assert mx.all(
        output.timeline.timestamps[:, 1:] >= output.timeline.timestamps[:, :-1]
    ).item()
    controls = {control.name: control.token_id for control in output.timeline.controls}
    assert controls["audio_user_start"] == model.registry.audio_user_start
    assert controls["video_frame"] == model.registry.video_frame
    assert controls["interruption"] == model.registry.interruption
    for control in output.timeline.controls:
        assert output.timeline.kinds[control.offset] == control.name

    report = model.parameter_report()
    assert report["thinker"] == model.thinker.parameter_count()
    assert (
        sum(value for name, value in report.items() if name != "total")
        == report["total"]
    )
    assert report["total"] == model.parameter_count()


def test_multimodal_modules_are_random_init_with_no_checkpoint_dependency(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert list(inspect.signature(ScratchMultimodalLM).parameters) == [
        "config",
        "registry",
    ]
    mx.random.seed(10)
    first = ScratchMultimodalLM(tiny_multimodal_config())
    mx.eval(first.audio_codec.quantizer.codebooks)
    first_codebooks = first.audio_codec.quantizer.codebooks
    mx.random.seed(11)
    second = ScratchMultimodalLM(tiny_multimodal_config())
    mx.eval(second.audio_codec.quantizer.codebooks)
    assert not mx.array_equal(
        first_codebooks, second.audio_codec.quantizer.codebooks
    ).item()
    assert list(tmp_path.iterdir()) == []
