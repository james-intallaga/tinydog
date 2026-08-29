"""Scratch-only losses for native duplex audio/video conversation training.

The live transport must never be treated as training.  This module is the
explicit parameter-learning boundary: audited, time-aligned examples enter as
native waveform/video tensors, and gradients cover the codec, perception
adapters, shared thinker, and causal talker without ASR or TTS targets from a
runtime service.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from .multimodal import (
    AudioCodecLoss,
    ScratchMultimodalLM,
    TimedAudio,
    TimedText,
    TimedVideo,
)


NativeStage = Literal[
    "audio_codec",
    "speech_understanding",
    "vision",
    "talker",
    "conversation",
    "joint",
]


@dataclass(frozen=True)
class NativeLossWeights:
    codec: float
    text: float
    user_audio: float
    model_audio: float

    def __post_init__(self) -> None:
        values = (self.codec, self.text, self.user_audio, self.model_audio)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            for value in values
        ):
            raise ValueError("native loss weights must be finite and non-negative")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one native loss weight must be positive")

    @classmethod
    def for_stage(cls, stage: NativeStage) -> "NativeLossWeights":
        if stage == "audio_codec":
            return cls(codec=1.0, text=0.0, user_audio=0.0, model_audio=0.0)
        if stage == "speech_understanding":
            return cls(codec=0.1, text=1.0, user_audio=0.1, model_audio=0.0)
        if stage == "vision":
            return cls(codec=0.0, text=1.0, user_audio=0.0, model_audio=0.0)
        if stage == "talker":
            return cls(codec=0.1, text=0.0, user_audio=0.1, model_audio=1.0)
        if stage == "conversation":
            return cls(codec=0.0, text=1.0, user_audio=0.1, model_audio=1.0)
        if stage == "joint":
            return cls(codec=0.1, text=1.0, user_audio=0.1, model_audio=1.0)
        raise ValueError(f"unsupported native training stage: {stage}")


@dataclass(frozen=True)
class NativeConversationBatch:
    """One aligned full-duplex training window.

    User and model waveforms cover the same wall-clock samples.  Silence is a
    real waveform target, not a magic token, so overlap and barge-in can be
    learned without padding IDs leaking into the audio heads.
    """

    user_waveform: mx.array
    model_waveform: mx.array
    text_tokens: mx.array
    text_target_mask: mx.array
    text_timestamp: float
    frames: mx.array | None = None
    frame_timestamps: mx.array | None = None
    frame_difference_scores: mx.array | None = None

    def validate(self, model: ScratchMultimodalLM) -> None:
        audio = model.config.audio
        for name, waveform in (
            ("user_waveform", self.user_waveform),
            ("model_waveform", self.model_waveform),
        ):
            expected_channels = audio.channels
            if waveform.ndim == 2 and expected_channels == 1:
                pass
            elif waveform.ndim != 3 or waveform.shape[-1] != expected_channels:
                raise ValueError(
                    f"{name} must have shape [batch, samples] or "
                    f"[batch, samples, {expected_channels}]"
                )
            if waveform.shape[0] <= 0 or waveform.shape[1] <= 0:
                raise ValueError(f"{name} must contain audio")
            if not mx.issubdtype(waveform.dtype, mx.floating):
                raise ValueError(f"{name} must use a floating-point dtype")
            if waveform.shape[1] % audio.frame_size:
                raise ValueError(f"{name} samples must align to codec frames")
            if not bool(mx.all(mx.isfinite(waveform)).item()):
                raise ValueError(f"{name} must contain only finite samples")
            if float(mx.max(mx.abs(waveform)).item()) > 1.0:
                raise ValueError(f"{name} samples must be normalized to [-1, 1]")
        if self.user_waveform.shape[:2] != self.model_waveform.shape[:2]:
            raise ValueError("duplex waveforms must cover the same batch and samples")
        if (
            self.text_tokens.ndim != 2
            or self.text_tokens.shape[0] != self.user_waveform.shape[0]
        ):
            raise ValueError("text_tokens must have shape [batch, length]")
        if self.text_tokens.shape[1] <= 0:
            raise ValueError("text_tokens must not be empty")
        if self.text_target_mask.shape != self.text_tokens.shape:
            raise ValueError("text_target_mask must match text_tokens")
        if not mx.issubdtype(self.text_target_mask.dtype, mx.floating):
            raise ValueError("text_target_mask must use a floating-point dtype")
        if not bool(mx.all(mx.isfinite(self.text_target_mask)).item()):
            raise ValueError("text_target_mask must be finite")
        if not bool(
            mx.all(
                (self.text_target_mask == 0.0) | (self.text_target_mask == 1.0)
            ).item()
        ):
            raise ValueError("text_target_mask must be binary")
        if not mx.issubdtype(self.text_tokens.dtype, mx.integer):
            raise ValueError("text_tokens must use an integer dtype")
        if self.text_tokens.size and (
            int(mx.min(self.text_tokens).item()) < 0
            or int(mx.max(self.text_tokens).item()) >= model.config.thinker.vocab_size
        ):
            raise ValueError("text token IDs are outside the thinker vocabulary")
        if float(mx.sum(self.text_target_mask).item()) <= 0.0:
            raise ValueError("text_target_mask must supervise at least one token")
        duration = self.user_waveform.shape[1] / audio.sample_rate
        if (
            not math.isfinite(self.text_timestamp)
            or self.text_timestamp < 0.0
            or self.text_timestamp > duration + 1e-6
        ):
            raise ValueError("text_timestamp must fall inside the aligned audio window")
        video_values = (
            self.frames,
            self.frame_timestamps,
            self.frame_difference_scores,
        )
        if all(value is None for value in video_values):
            return
        if self.frames is None or self.frame_timestamps is None:
            raise ValueError("frames and frame_timestamps must be supplied together")
        if self.frames.ndim != 5 or self.frames.shape[0] != self.user_waveform.shape[0]:
            raise ValueError(
                "frames must have shape [batch, frames, height, width, RGB]"
            )
        if self.frames.shape[1] <= 0 or self.frames.shape[-1] != 3:
            raise ValueError("frames must contain at least one RGB frame")
        if not mx.issubdtype(self.frames.dtype, mx.floating):
            raise ValueError("frames must use a floating-point dtype")
        if not bool(mx.all(mx.isfinite(self.frames)).item()):
            raise ValueError("frames must be finite")
        if (
            float(mx.min(self.frames).item()) < 0.0
            or float(mx.max(self.frames).item()) > 1.0
        ):
            raise ValueError("frames must be normalized to [0, 1]")
        divisor = model.config.video.patch_size * model.config.video.spatial_merge_size
        if self.frames.shape[2] % divisor or self.frames.shape[3] % divisor:
            raise ValueError("frame dimensions do not match the vision patch grid")
        metadata_shapes = {
            (self.frames.shape[1],),
            (self.frames.shape[0], self.frames.shape[1]),
        }
        if self.frame_timestamps.shape not in metadata_shapes:
            raise ValueError(
                "frame_timestamps must have shape [frames] or [batch, frames]"
            )
        if not mx.issubdtype(self.frame_timestamps.dtype, mx.floating):
            raise ValueError("frame_timestamps must use a floating-point dtype")
        if not bool(mx.all(mx.isfinite(self.frame_timestamps)).item()):
            raise ValueError("frame_timestamps must be finite")
        ordered_times = (
            self.frame_timestamps[None, :]
            if self.frame_timestamps.ndim == 1
            else self.frame_timestamps
        )
        if (
            float(mx.min(ordered_times).item()) < 0.0
            or float(mx.max(ordered_times).item()) > duration + 1e-6
            or not bool(mx.all(ordered_times[:, 1:] >= ordered_times[:, :-1]).item())
        ):
            raise ValueError(
                "frame_timestamps must be ordered inside the aligned audio window"
            )
        if self.frame_difference_scores is not None:
            if self.frame_difference_scores.shape not in metadata_shapes:
                raise ValueError("frame_difference_scores must match selected frames")
            if not mx.issubdtype(self.frame_difference_scores.dtype, mx.floating):
                raise ValueError(
                    "frame_difference_scores must use a floating-point dtype"
                )
            if not bool(mx.all(mx.isfinite(self.frame_difference_scores)).item()):
                raise ValueError("frame_difference_scores must be finite")
            if (
                float(mx.min(self.frame_difference_scores).item()) < 0.0
                or float(mx.max(self.frame_difference_scores).item()) > 1.0
            ):
                raise ValueError("frame_difference_scores must be normalized to [0, 1]")


@dataclass(frozen=True)
class NativeLoss:
    total: mx.array
    codec: mx.array
    text: mx.array
    user_audio: mx.array
    model_audio: mx.array
    text_targets: int
    text_correct: int
    audio_targets: int
    model_audio_correct: int


def _mean_codec_loss(first: AudioCodecLoss, second: AudioCodecLoss) -> mx.array:
    return (first.total.astype(mx.float32) + second.total.astype(mx.float32)) * 0.5


def _masked_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> mx.array:
    losses = nn.losses.cross_entropy(
        logits.astype(mx.float32), targets, reduction="none"
    )
    weights = mask.astype(mx.float32)
    denominator = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=mx.float32))
    return mx.sum(losses * weights) / denominator


def _causal_audio_conditioning(
    output_hidden: mx.array,
    timeline_kinds: tuple[str, ...],
    audio_frames: int,
) -> mx.array:
    """Return the thinker state immediately before each current user frame.

    The native waveforms are aligned on one wall-clock grid.  Conditioning
    every target frame on the final timeline state would expose the entire
    future user utterance (and response text) to early audio predictions.  The
    state immediately preceding each ``audio_user`` unit is the conservative
    causal boundary: it may contain earlier audio/video, never teacher-forced
    response text, the current user frame, or a future user frame.
    """

    positions = [
        index for index, kind in enumerate(timeline_kinds) if kind == "audio_user"
    ]
    if len(positions) != audio_frames or any(position <= 0 for position in positions):
        raise RuntimeError("native timeline audio grid is incompatible")
    return mx.concatenate(
        [output_hidden[:, position - 1 : position, :] for position in positions],
        axis=1,
    )


def native_conversation_loss(
    model: ScratchMultimodalLM,
    batch: NativeConversationBatch,
    weights: NativeLossWeights,
) -> NativeLoss:
    """Compute aligned codec, response-text, and duplex-code objectives.

    Text is predicted after the observed media prefix.  Audio targets are
    teacher-forced only through their *previous* frames; the talker itself
    enforces that current user/model codes cannot leak into same-step logits.
    """

    batch.validate(model)
    user_codec = model.audio_codec(batch.user_waveform)
    model_codec = model.audio_codec(batch.model_waveform)
    if user_codec.encoding.codes.shape != model_codec.encoding.codes.shape:
        raise ValueError("duplex codec streams must produce the same frame grid")

    perception_events: list[TimedAudio | TimedVideo] = [
        TimedAudio(timestamp=0.0, codes=user_codec.encoding.codes, stream="user")
    ]
    if batch.frames is not None and batch.frame_timestamps is not None:
        perception_events.append(
            TimedVideo(
                frames=batch.frames,
                timestamps=batch.frame_timestamps,
                difference_scores=batch.frame_difference_scores,
            )
        )
    events: list[TimedAudio | TimedText | TimedVideo] = [
        *perception_events,
        TimedText(timestamp=batch.text_timestamp, tokens=batch.text_tokens),
    ]
    output = model.forward_timeline(events)

    try:
        text_start = output.timeline.kinds.index("text")
    except ValueError as error:
        raise RuntimeError("native timeline contains no response text") from error
    text_length = batch.text_tokens.shape[1]
    if text_start <= 0 or text_start + text_length > output.text_logits.shape[1]:
        raise RuntimeError("native response text lies outside the timeline logits")
    # The last media/control state predicts the first response token; each
    # response token then predicts the next token in the same assistant span.
    text_logits = output.text_logits[
        :, text_start - 1 : text_start + text_length - 1, :
    ]
    text_loss = _masked_cross_entropy(
        text_logits,
        batch.text_tokens,
        batch.text_target_mask,
    )
    text_predictions = mx.argmax(text_logits, axis=-1)
    text_correct = int(
        mx.sum(
            (text_predictions == batch.text_tokens)
            * batch.text_target_mask.astype(mx.int32)
        ).item()
    )

    audio_mask = mx.ones(user_codec.encoding.codes.shape, dtype=mx.float32)
    zero = mx.array(0.0, dtype=mx.float32)
    user_audio_loss = zero
    model_audio_loss = zero
    model_audio_correct = 0
    if weights.user_audio > 0.0 or weights.model_audio > 0.0:
        # The live runtime conditions the talker on causal media states, not on
        # teacher-forced assistant text.  Use a separate perception timeline so
        # response labels cannot become a training-only shortcut for speech.
        perception_output = model.forward_timeline(perception_events)
        audio_conditioning = _causal_audio_conditioning(
            perception_output.hidden,
            perception_output.timeline.kinds,
            user_codec.encoding.codes.shape[1],
        )
        audio_output = model.talker(
            audio_conditioning,
            user_codec.encoding.codes,
            model_codec.encoding.codes,
        )
        if weights.user_audio > 0.0:
            user_audio_loss = _masked_cross_entropy(
                audio_output.user_logits,
                user_codec.encoding.codes,
                audio_mask,
            )
        if weights.model_audio > 0.0:
            model_audio_loss = _masked_cross_entropy(
                audio_output.model_logits,
                model_codec.encoding.codes,
                audio_mask,
            )
            model_audio_correct = int(
                mx.sum(
                    mx.argmax(audio_output.model_logits, axis=-1)
                    == model_codec.encoding.codes
                ).item()
            )
    codec_loss = _mean_codec_loss(user_codec.losses, model_codec.losses)
    total = (
        weights.codec * codec_loss
        + weights.text * text_loss
        + weights.user_audio * user_audio_loss
        + weights.model_audio * model_audio_loss
    )
    return NativeLoss(
        total=total,
        codec=codec_loss,
        text=text_loss,
        user_audio=user_audio_loss,
        model_audio=model_audio_loss,
        text_targets=int(mx.sum(batch.text_target_mask).item()),
        text_correct=text_correct,
        audio_targets=int(audio_mask.size),
        model_audio_correct=model_audio_correct,
    )
