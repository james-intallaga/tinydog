"""Random-init, scratch-only audio/video scaffold around :class:`HybridLM`.

The code defines trainable interfaces and causal stream state; it does not
contain a trained codec, speech recognizer, voice, vision backbone, or external
checkpoint dependency.  The existing text-only ``HybridLM`` remains unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal, TypeAlias

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .hybrid import HybridCache, HybridConfig, HybridLM
from .scratch_tokenizer import EXTENSION_SLOT_COUNT, extension_slot_id


AudioStreamName: TypeAlias = Literal["user", "model"]


def _validate_discrete_ids(values: mx.array, *, cardinality: int, name: str) -> None:
    """Fail clearly before an embedding/gather sees malformed stream IDs."""

    if not mx.issubdtype(values.dtype, mx.integer):
        raise ValueError(f"{name} must use an integer dtype")
    if values.size and (
        int(mx.min(values).item()) < 0 or int(mx.max(values).item()) >= cardinality
    ):
        raise ValueError(f"{name} IDs must be in [0, {cardinality})")


@dataclass(frozen=True)
class ModalityTokenRegistry:
    """Versioned meanings assigned to the tokenizer's immutable extension IDs."""

    version: int = 1
    audio_user_start: int = field(default_factory=lambda: extension_slot_id(0))
    audio_user_end: int = field(default_factory=lambda: extension_slot_id(1))
    audio_model_start: int = field(default_factory=lambda: extension_slot_id(2))
    audio_model_end: int = field(default_factory=lambda: extension_slot_id(3))
    video_start: int = field(default_factory=lambda: extension_slot_id(4))
    video_frame: int = field(default_factory=lambda: extension_slot_id(5))
    video_end: int = field(default_factory=lambda: extension_slot_id(6))
    interruption: int = field(default_factory=lambda: extension_slot_id(7))

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("Unsupported modality token registry version")
        values = list(self.to_dict().values())[1:]
        if len(set(values)) != len(values):
            raise ValueError("Modality control-token IDs must be unique")
        allowed = {extension_slot_id(slot) for slot in range(EXTENSION_SLOT_COUNT)}
        if any(value not in allowed for value in values):
            raise ValueError("Modality tokens must use frozen tokenizer extension IDs")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AudioCodecConfig:
    sample_rate: int = 24_000
    channels: int = 1
    # One native codec step is one 20 ms media block.  Keeping the codec,
    # WebRTC playout, and interruption quantum aligned avoids an unavoidable
    # 80 ms buffering floor before the model has done any work.
    frame_size: int = 480
    hidden_dim: int = 128
    latent_dim: int = 128
    temporal_kernel_size: int = 3
    num_codebooks: int = 8
    codebook_size: int = 256
    commitment_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.sample_rate != 24_000:
            raise ValueError("The scratch audio contract is fixed at 24 kHz")
        for name in (
            "channels",
            "frame_size",
            "hidden_dim",
            "latent_dim",
            "num_codebooks",
            "codebook_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.temporal_kernel_size < 2:
            raise ValueError("temporal_kernel_size must be at least 2")
        if self.commitment_weight < 0:
            raise ValueError("commitment_weight must be non-negative")

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.frame_size


@dataclass(frozen=True)
class VideoEncoderConfig:
    input_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    vision_dim: int = 128
    resampler_tokens: int = 12
    resampler_dim: int = 256
    resampler_heads: int = 8
    model_dim: int = 1_024

    def __post_init__(self) -> None:
        if self.input_channels != 3:
            raise ValueError("The video scaffold accepts RGB frames")
        for name in (
            "patch_size",
            "spatial_merge_size",
            "vision_dim",
            "resampler_tokens",
            "resampler_dim",
            "resampler_heads",
            "model_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.resampler_dim % self.resampler_heads:
            raise ValueError("resampler_dim must be divisible by resampler_heads")


@dataclass(frozen=True)
class TalkerConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    temporal_kernel_size: int = 3

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.num_layers <= 0:
            raise ValueError("Talker dimensions must be positive")
        if self.temporal_kernel_size < 2:
            raise ValueError("Talker temporal kernel must be at least 2")


@dataclass(frozen=True)
class MultimodalConfig:
    thinker: HybridConfig = field(default_factory=HybridConfig)
    audio: AudioCodecConfig = field(default_factory=AudioCodecConfig)
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    talker: TalkerConfig = field(default_factory=TalkerConfig)
    time_scale_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.video.model_dim != self.thinker.model_dim:
            raise ValueError("Video output dimension must match thinker model_dim")
        if self.time_scale_seconds <= 0:
            raise ValueError("time_scale_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultimodalConfig":
        thinker_value = dict(value.get("thinker", {}))
        if "layer_schedule" in thinker_value:
            thinker_value["layer_schedule"] = tuple(thinker_value["layer_schedule"])
        return cls(
            thinker=HybridConfig(**thinker_value),
            audio=AudioCodecConfig(**dict(value.get("audio", {}))),
            video=VideoEncoderConfig(**dict(value.get("video", {}))),
            talker=TalkerConfig(**dict(value.get("talker", {}))),
            time_scale_seconds=float(value.get("time_scale_seconds", 60.0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "MultimodalConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class CausalConvState:
    values: mx.array | None = None


class CausalTemporalBlock(nn.Module):
    """Pre-normalized causal depthwise temporal convolution with explicit state."""

    def __init__(self, dimension: int, kernel_size: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.norm = nn.RMSNorm(dimension)
        self.conv = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=kernel_size,
            groups=dimension,
            bias=False,
        )
        self.projection = nn.Linear(dimension, dimension, bias=False)

    def __call__(self, x: mx.array, state: CausalConvState | None = None) -> mx.array:
        if x.ndim != 3 or x.shape[-1] != self.dimension:
            raise ValueError(f"Expected [batch, time, {self.dimension}] temporal input")
        if x.shape[1] == 0:
            return x
        normalized = self.norm(x)
        history_length = self.kernel_size - 1
        if state is None:
            convolution_input = mx.pad(
                normalized, [(0, 0), (history_length, 0), (0, 0)]
            )
        else:
            history = state.values
            if history is None:
                history = mx.zeros(
                    (x.shape[0], history_length, self.dimension), dtype=normalized.dtype
                )
            elif history.shape != (x.shape[0], history_length, self.dimension):
                raise ValueError("Causal convolution state has an incompatible shape")
            convolution_input = mx.concatenate((history, normalized), axis=1)
            state.values = convolution_input[:, -history_length:, :]
        return x + self.projection(nn.silu(self.conv(convolution_input)))


@dataclass(frozen=True)
class QuantizerLoss:
    codebook: mx.array
    commitment: mx.array
    total: mx.array


@dataclass(frozen=True)
class AudioEncoding:
    codes: mx.array
    latents: mx.array
    quantized: mx.array
    original_samples: int
    padded_samples: int
    quantizer_loss: QuantizerLoss


@dataclass(frozen=True)
class AudioCodecLoss:
    reconstruction: mx.array
    codebook: mx.array
    commitment: mx.array
    total: mx.array


@dataclass(frozen=True)
class AudioCodecOutput:
    encoding: AudioEncoding
    reconstruction: mx.array
    losses: AudioCodecLoss


@dataclass
class AudioStreamState:
    batch_size: int
    encoder_buffer: mx.array
    encoder_temporal: CausalConvState = field(default_factory=CausalConvState)
    decoder_temporal: CausalConvState = field(default_factory=CausalConvState)
    samples_seen: int = 0
    frames_encoded: int = 0
    frames_decoded: int = 0
    encoder_finalized: bool = False


class ResidualVectorQuantizer(nn.Module):
    """Trainable random residual codebooks with straight-through latents."""

    def __init__(self, config: AudioCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.codebooks = mx.random.normal(
            (config.num_codebooks, config.codebook_size, config.latent_dim)
        ) * (config.latent_dim**-0.5)

    def __call__(self, latents: mx.array) -> tuple[mx.array, mx.array, QuantizerLoss]:
        if latents.ndim != 3 or latents.shape[-1] != self.config.latent_dim:
            raise ValueError("RVQ latents have an incompatible shape")
        if latents.shape[1] == 0:
            codes = mx.zeros(
                (latents.shape[0], 0, self.config.num_codebooks), dtype=mx.int32
            )
            zero = mx.array(0.0, dtype=latents.dtype)
            return latents, codes, QuantizerLoss(zero, zero, zero)

        residual = latents
        quantized_sum = mx.zeros_like(latents)
        code_columns: list[mx.array] = []
        for codebook_index in range(self.config.num_codebooks):
            codebook = self.codebooks[codebook_index]
            distances = (
                mx.sum(residual * residual, axis=-1, keepdims=True)
                - 2 * (residual @ codebook.T)
                + mx.sum(codebook * codebook, axis=-1)
            )
            # Nearest-code selection is discrete.  Keeping its indices outside
            # the VJP lets gradients flow through selected codebook rows and
            # the straight-through latent path without differentiating argmin.
            codes = mx.stop_gradient(mx.argmin(distances, axis=-1))
            selected = mx.take(codebook, codes, axis=0)
            quantized_sum = quantized_sum + selected
            residual = residual - selected
            code_columns.append(codes)

        code_ids = mx.stack(code_columns, axis=-1).astype(mx.int32)
        codebook_loss = mx.mean((quantized_sum - mx.stop_gradient(latents)) ** 2)
        commitment_loss = mx.mean((latents - mx.stop_gradient(quantized_sum)) ** 2)
        total_loss = codebook_loss + self.config.commitment_weight * commitment_loss
        straight_through = latents + mx.stop_gradient(quantized_sum - latents)
        return (
            straight_through,
            code_ids,
            QuantizerLoss(
                codebook=codebook_loss,
                commitment=commitment_loss,
                total=total_loss,
            ),
        )

    def decode_codes(self, codes: mx.array) -> mx.array:
        if codes.ndim != 3 or codes.shape[-1] != self.config.num_codebooks:
            raise ValueError("Audio codes must have shape [batch, frames, codebooks]")
        _validate_discrete_ids(
            codes, cardinality=self.config.codebook_size, name="audio codebook"
        )
        quantized = mx.zeros(
            (codes.shape[0], codes.shape[1], self.config.latent_dim),
            dtype=self.codebooks.dtype,
        )
        for codebook_index in range(self.config.num_codebooks):
            quantized = quantized + mx.take(
                self.codebooks[codebook_index], codes[:, :, codebook_index], axis=0
            )
        return quantized


class StreamingWaveformCodec(nn.Module):
    """Frame-causal 24 kHz waveform codec skeleton with explicit stream state."""

    def __init__(self, config: AudioCodecConfig | None = None) -> None:
        super().__init__()
        self.config = config or AudioCodecConfig()
        frame_width = self.config.frame_size * self.config.channels
        self.encoder_input = nn.Linear(frame_width, self.config.hidden_dim, bias=False)
        self.encoder_temporal = CausalTemporalBlock(
            self.config.hidden_dim, self.config.temporal_kernel_size
        )
        self.encoder_output = nn.Linear(
            self.config.hidden_dim, self.config.latent_dim, bias=False
        )
        self.quantizer = ResidualVectorQuantizer(self.config)
        self.decoder_input = nn.Linear(
            self.config.latent_dim, self.config.hidden_dim, bias=False
        )
        self.decoder_temporal = CausalTemporalBlock(
            self.config.hidden_dim, self.config.temporal_kernel_size
        )
        self.decoder_output = nn.Linear(self.config.hidden_dim, frame_width, bias=False)

    def _coerce_waveform(self, waveform: mx.array) -> mx.array:
        if waveform.ndim == 2 and self.config.channels == 1:
            waveform = waveform[:, :, None]
        if waveform.ndim != 3 or waveform.shape[-1] != self.config.channels:
            raise ValueError(
                f"waveform must have shape [batch, samples, {self.config.channels}]"
            )
        if waveform.shape[0] <= 0:
            raise ValueError("waveform batch must be non-empty")
        return waveform

    def init_stream_state(
        self, batch_size: int, *, dtype: mx.Dtype = mx.float32
    ) -> AudioStreamState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return AudioStreamState(
            batch_size=batch_size,
            encoder_buffer=mx.zeros((batch_size, 0, self.config.channels), dtype=dtype),
        )

    def encode_stream(
        self,
        waveform_chunk: mx.array,
        state: AudioStreamState,
        *,
        final: bool = False,
    ) -> AudioEncoding:
        waveform = self._coerce_waveform(waveform_chunk)
        if state.encoder_finalized:
            raise ValueError("Cannot encode after finalizing an audio stream")
        if state.batch_size != waveform.shape[0]:
            raise ValueError("Audio stream state batch size does not match waveform")
        if state.encoder_buffer.dtype != waveform.dtype:
            state.encoder_buffer = state.encoder_buffer.astype(waveform.dtype)
        combined = mx.concatenate((state.encoder_buffer, waveform), axis=1)
        original_chunk_samples = waveform.shape[1]
        state.samples_seen += original_chunk_samples
        remainder = combined.shape[1] % self.config.frame_size
        padding = (
            (self.config.frame_size - remainder) % self.config.frame_size
            if final
            else 0
        )
        if padding:
            combined_for_frames = mx.pad(combined, [(0, 0), (0, padding), (0, 0)])
        else:
            combined_for_frames = combined
        usable_samples = (
            combined_for_frames.shape[1] if final else combined.shape[1] - remainder
        )
        framed_samples = combined_for_frames[:, :usable_samples, :]
        state.encoder_buffer = (
            mx.zeros((state.batch_size, 0, self.config.channels), dtype=waveform.dtype)
            if final
            else combined[:, usable_samples:, :]
        )
        if final:
            state.encoder_finalized = True

        frame_count = usable_samples // self.config.frame_size
        if frame_count == 0:
            latents = mx.zeros(
                (state.batch_size, 0, self.config.latent_dim), dtype=waveform.dtype
            )
        else:
            frames = framed_samples.reshape(
                state.batch_size,
                frame_count,
                self.config.frame_size * self.config.channels,
            )
            hidden = nn.silu(self.encoder_input(frames))
            hidden = self.encoder_temporal(hidden, state.encoder_temporal)
            latents = self.encoder_output(hidden)
        quantized, codes, quantizer_loss = self.quantizer(latents)
        state.frames_encoded += frame_count
        return AudioEncoding(
            codes=codes,
            latents=latents,
            quantized=quantized,
            original_samples=original_chunk_samples,
            padded_samples=padding,
            quantizer_loss=quantizer_loss,
        )

    def encode(self, waveform: mx.array) -> AudioEncoding:
        coerced = self._coerce_waveform(waveform)
        if coerced.shape[1] <= 0:
            raise ValueError("waveform must contain samples")
        state = self.init_stream_state(coerced.shape[0], dtype=coerced.dtype)
        return self.encode_stream(coerced, state, final=True)

    def decode_latents_stream(
        self, quantized: mx.array, state: AudioStreamState
    ) -> mx.array:
        if quantized.ndim != 3 or quantized.shape[-1] != self.config.latent_dim:
            raise ValueError(
                "quantized audio must have shape [batch, frames, latent_dim]"
            )
        if state.batch_size != quantized.shape[0]:
            raise ValueError("Audio decoder state batch size does not match latents")
        if quantized.shape[1] == 0:
            return mx.zeros(
                (state.batch_size, 0, self.config.channels), dtype=quantized.dtype
            )
        hidden = nn.silu(self.decoder_input(quantized))
        hidden = self.decoder_temporal(hidden, state.decoder_temporal)
        frames = self.decoder_output(hidden).reshape(
            state.batch_size,
            quantized.shape[1],
            self.config.frame_size,
            self.config.channels,
        )
        state.frames_decoded += quantized.shape[1]
        return frames.reshape(
            state.batch_size,
            quantized.shape[1] * self.config.frame_size,
            self.config.channels,
        )

    def decode_codes_stream(self, codes: mx.array, state: AudioStreamState) -> mx.array:
        return self.decode_latents_stream(self.quantizer.decode_codes(codes), state)

    def decode(self, encoding: AudioEncoding) -> mx.array:
        state = self.init_stream_state(
            encoding.quantized.shape[0], dtype=encoding.quantized.dtype
        )
        decoded = self.decode_latents_stream(encoding.quantized, state)
        return decoded[:, : encoding.original_samples, :]

    def loss(self, waveform: mx.array, output: AudioCodecOutput) -> AudioCodecLoss:
        target = self._coerce_waveform(waveform)
        reconstruction = mx.mean((target - output.reconstruction) ** 2)
        quantizer = output.encoding.quantizer_loss
        return AudioCodecLoss(
            reconstruction=reconstruction,
            codebook=quantizer.codebook,
            commitment=quantizer.commitment,
            total=reconstruction + quantizer.total,
        )

    def __call__(self, waveform: mx.array) -> AudioCodecOutput:
        encoding = self.encode(waveform)
        reconstruction = self.decode(encoding)
        provisional = AudioCodecOutput(
            encoding=encoding,
            reconstruction=reconstruction,
            losses=AudioCodecLoss(
                reconstruction=mx.array(0.0),
                codebook=encoding.quantizer_loss.codebook,
                commitment=encoding.quantizer_loss.commitment,
                total=encoding.quantizer_loss.total,
            ),
        )
        return AudioCodecOutput(
            encoding=encoding,
            reconstruction=reconstruction,
            losses=self.loss(waveform, provisional),
        )


@dataclass(frozen=True)
class VideoEncoding:
    embeddings: mx.array
    timestamps: mx.array
    difference_scores: mx.array
    frames: int
    tokens_per_frame: int


class SelectedFrameVideoEncoder(nn.Module):
    """Patch encoder plus a fixed-budget learned-query frame resampler.

    Every selected frame is encoded independently, then reduced to a small,
    configuration-fixed token budget.  The model therefore pays for changes
    in the scene rather than for the camera resolution or the raw patch grid.
    """

    def __init__(self, config: VideoEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or VideoEncoderConfig()
        self.patch_encoder = nn.Conv2d(
            self.config.input_channels,
            self.config.vision_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            bias=False,
        )
        self.metadata_projection = nn.Linear(3, self.config.vision_dim, bias=False)
        merged_width = (
            self.config.vision_dim
            * self.config.spatial_merge_size
            * self.config.spatial_merge_size
        )
        self.block_projection = nn.Linear(
            merged_width, self.config.resampler_dim, bias=False
        )
        self.source_norm = nn.RMSNorm(self.config.resampler_dim)
        self.query_norm = nn.RMSNorm(self.config.resampler_dim)
        self.query_embeddings = mx.random.normal(
            (self.config.resampler_tokens, self.config.resampler_dim)
        ) * (self.config.resampler_dim**-0.5)
        self.query_projection = nn.Linear(
            self.config.resampler_dim, self.config.resampler_dim, bias=False
        )
        self.key_projection = nn.Linear(
            self.config.resampler_dim, self.config.resampler_dim, bias=False
        )
        self.value_projection = nn.Linear(
            self.config.resampler_dim, self.config.resampler_dim, bias=False
        )
        self.resampler_output = nn.Linear(
            self.config.resampler_dim, self.config.model_dim, bias=False
        )
        self.output_norm = nn.RMSNorm(self.config.model_dim)

    def _resample_blocks(self, blocks: mx.array) -> mx.array:
        """Reduce ``[batch*frames, blocks, width]`` to fixed visual tokens."""

        batch_frames, _, width = blocks.shape
        if width != self.config.resampler_dim:
            raise ValueError("Video block projection has an incompatible width")
        heads = self.config.resampler_heads
        head_dim = width // heads
        queries = mx.broadcast_to(
            self.query_embeddings[None, :, :],
            (batch_frames, self.config.resampler_tokens, width),
        )
        queries = self.query_projection(self.query_norm(queries))
        source = self.source_norm(blocks)
        keys = self.key_projection(source)
        values = self.value_projection(source)

        queries = queries.reshape(
            batch_frames, self.config.resampler_tokens, heads, head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_frames, -1, heads, head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch_frames, -1, heads, head_dim).transpose(0, 2, 1, 3)
        weights = mx.softmax(
            (queries @ keys.transpose(0, 1, 3, 2)) * (head_dim**-0.5), axis=-1
        )
        pooled = (
            (weights @ values)
            .transpose(0, 2, 1, 3)
            .reshape(batch_frames, self.config.resampler_tokens, width)
        )
        return self.output_norm(self.resampler_output(pooled))

    def _metadata(
        self,
        frames: mx.array,
        timestamps: mx.array,
        difference_scores: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        batch, frame_count = frames.shape[:2]
        if timestamps.ndim == 1:
            if timestamps.shape[0] != frame_count:
                raise ValueError("timestamp count must match selected frames")
            timestamps = mx.broadcast_to(timestamps[None, :], (batch, frame_count))
        if timestamps.shape != (batch, frame_count):
            raise ValueError("timestamps must have shape [frames] or [batch, frames]")
        deltas = mx.concatenate(
            (
                mx.zeros((batch, 1), dtype=timestamps.dtype),
                timestamps[:, 1:] - timestamps[:, :-1],
            ),
            axis=1,
        )
        if difference_scores is None:
            if frame_count == 1:
                difference_scores = mx.zeros((batch, 1), dtype=frames.dtype)
            else:
                differences = mx.mean(
                    mx.abs(frames[:, 1:, :, :, :] - frames[:, :-1, :, :, :]),
                    axis=(2, 3, 4),
                )
                difference_scores = mx.concatenate(
                    (mx.zeros((batch, 1), dtype=differences.dtype), differences), axis=1
                )
        elif difference_scores.ndim == 1:
            difference_scores = mx.broadcast_to(
                difference_scores[None, :], (batch, frame_count)
            )
        if difference_scores.shape != (batch, frame_count):
            raise ValueError("difference_scores must match selected frame timestamps")
        return timestamps, mx.stack((timestamps, deltas, difference_scores), axis=-1)

    def __call__(
        self,
        frames: mx.array,
        timestamps: mx.array,
        difference_scores: mx.array | None = None,
    ) -> VideoEncoding:
        if frames.ndim != 5 or frames.shape[-1] != self.config.input_channels:
            raise ValueError(
                "frames must have shape [batch, frames, height, width, RGB]"
            )
        batch, frame_count, height, width, channels = frames.shape
        if batch <= 0 or frame_count <= 0:
            raise ValueError("At least one selected frame is required")
        divisor = self.config.patch_size * self.config.spatial_merge_size
        if height % divisor or width % divisor:
            raise ValueError(f"Frame height and width must be divisible by {divisor}")
        timestamps, metadata = self._metadata(frames, timestamps, difference_scores)
        if not mx.all(timestamps[:, 1:] >= timestamps[:, :-1]).item():
            raise ValueError("Selected-frame timestamps must be non-decreasing")
        patches = self.patch_encoder(
            frames.reshape(batch * frame_count, height, width, channels)
        )
        patch_height, patch_width = patches.shape[1:3]
        patches = patches.reshape(
            batch, frame_count, patch_height, patch_width, self.config.vision_dim
        )
        patches = patches + self.metadata_projection(metadata)[:, :, None, None, :]
        merge = self.config.spatial_merge_size
        block_height, block_width = patch_height // merge, patch_width // merge
        blocks = patches.reshape(
            batch,
            frame_count,
            block_height,
            merge,
            block_width,
            merge,
            self.config.vision_dim,
        ).transpose(0, 1, 2, 4, 3, 5, 6)
        blocks = nn.silu(
            self.block_projection(
                blocks.reshape(
                    batch,
                    frame_count,
                    block_height * block_width,
                    merge * merge * self.config.vision_dim,
                )
            )
        )
        embeddings = self._resample_blocks(
            blocks.reshape(
                batch * frame_count,
                block_height * block_width,
                self.config.resampler_dim,
            )
        ).reshape(
            batch,
            frame_count,
            self.config.resampler_tokens,
            self.config.model_dim,
        )
        repeated_timestamps = mx.broadcast_to(
            timestamps[:, :, None],
            (batch, frame_count, self.config.resampler_tokens),
        )
        return VideoEncoding(
            embeddings=embeddings,
            timestamps=repeated_timestamps,
            difference_scores=metadata[:, :, 2],
            frames=frame_count,
            tokens_per_frame=self.config.resampler_tokens,
        )


@dataclass
class TalkerStreamState:
    batch_size: int
    temporal: list[CausalConvState]
    model_active: mx.array
    previous_user_codes: mx.array | None = None
    previous_model_codes: mx.array | None = None
    offset: int = 0


@dataclass(frozen=True)
class TalkerOutput:
    user_logits: mx.array
    model_logits: mx.array
    hidden: mx.array
    model_active: mx.array


@dataclass(frozen=True)
class TalkerGeneration:
    """Autoregressively selected model-audio codes and their causal outputs."""

    codes: mx.array
    user_logits: mx.array
    model_logits: mx.array
    hidden: mx.array
    model_active: mx.array


class DualStreamAudioTalker(nn.Module):
    """Small causal user/model-code talker conditioned by thinker states."""

    def __init__(
        self,
        thinker_dim: int,
        audio_config: AudioCodecConfig,
        config: TalkerConfig | None = None,
    ) -> None:
        super().__init__()
        self.thinker_dim = thinker_dim
        self.audio_config = audio_config
        self.config = config or TalkerConfig()
        # Both streams have a private beginning-of-audio entry. Training inputs
        # are shifted, so neither head can inspect the user or model code it is
        # being trained to predict at the same frame.
        self.user_embeddings = [
            nn.Embedding(audio_config.codebook_size + 1, self.config.hidden_dim)
            for _ in range(audio_config.num_codebooks)
        ]
        self.model_embeddings = [
            nn.Embedding(audio_config.codebook_size + 1, self.config.hidden_dim)
            for _ in range(audio_config.num_codebooks)
        ]
        self.conditioning_projection = nn.Linear(
            thinker_dim, self.config.hidden_dim, bias=False
        )
        self.temporal_blocks = [
            CausalTemporalBlock(
                self.config.hidden_dim, self.config.temporal_kernel_size
            )
            for _ in range(self.config.num_layers)
        ]
        output_width = audio_config.num_codebooks * audio_config.codebook_size
        self.user_head = nn.Linear(self.config.hidden_dim, output_width, bias=False)
        self.model_head = nn.Linear(self.config.hidden_dim, output_width, bias=False)

    def init_stream_state(self, batch_size: int) -> TalkerStreamState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return TalkerStreamState(
            batch_size=batch_size,
            temporal=[CausalConvState() for _ in self.temporal_blocks],
            model_active=mx.ones((batch_size,), dtype=mx.bool_),
        )

    def request_interruption(
        self, state: TalkerStreamState, mask: mx.array | None = None
    ) -> None:
        """Stop selected model-audio streams while user-code processing continues."""

        if mask is None:
            mask = mx.ones((state.batch_size,), dtype=mx.bool_)
        if mask.shape != (state.batch_size,):
            raise ValueError("interruption mask must have shape [batch]")
        state.model_active = mx.where(
            mask, mx.zeros_like(state.model_active), state.model_active
        )

    def resume(self, state: TalkerStreamState, mask: mx.array | None = None) -> None:
        if mask is None:
            mask = mx.ones((state.batch_size,), dtype=mx.bool_)
        if mask.shape != (state.batch_size,):
            raise ValueError("resume mask must have shape [batch]")
        state.model_active = mx.where(
            mask, mx.ones_like(state.model_active), state.model_active
        )

    def _validate_previous_codes(
        self,
        codes: mx.array | None,
        *,
        batch: int,
        name: str,
    ) -> None:
        if codes is None:
            return
        if codes.shape != (batch, self.audio_config.num_codebooks):
            raise ValueError(f"{name} state has an incompatible shape")
        _validate_discrete_ids(
            codes,
            cardinality=self.audio_config.codebook_size,
            name=name,
        )

    def _validate_common_inputs(
        self,
        conditioning: mx.array,
        user_codes: mx.array,
        state: TalkerStreamState | None,
    ) -> tuple[int, int]:
        if user_codes.ndim != 3:
            raise ValueError("user_codes must have shape [batch, frames, codebooks]")
        expected = (
            user_codes.shape[0],
            user_codes.shape[1],
            self.audio_config.num_codebooks,
        )
        if user_codes.shape != expected:
            raise ValueError("user_codes must have shape [batch, frames, codebooks]")
        _validate_discrete_ids(
            user_codes,
            cardinality=self.audio_config.codebook_size,
            name="user audio",
        )
        if conditioning.ndim != 3 or conditioning.shape[0] != expected[0]:
            raise ValueError("conditioning must have shape [batch, time, thinker_dim]")
        if conditioning.shape[-1] != self.thinker_dim:
            raise ValueError("conditioning width does not match the thinker")
        if conditioning.shape[1] not in {1, expected[1]}:
            raise ValueError("conditioning time must be one or match audio frames")
        batch, frames = expected[:2]
        if batch <= 0:
            raise ValueError("Talker input batch must be positive")
        if frames <= 0:
            raise ValueError("Talker input must contain at least one audio frame")
        if state is not None:
            if state.batch_size != batch:
                raise ValueError("Talker stream state batch size does not match input")
            if len(state.temporal) != len(self.temporal_blocks):
                raise ValueError("Talker stream state layer count does not match model")
            if state.model_active.shape != (batch,):
                raise ValueError("Talker model-active mask has an incompatible shape")
            self._validate_previous_codes(
                state.previous_user_codes,
                batch=batch,
                name="previous user audio",
            )
            self._validate_previous_codes(
                state.previous_model_codes,
                batch=batch,
                name="previous model audio",
            )
        return batch, frames

    def _validate_inputs(
        self,
        conditioning: mx.array,
        user_codes: mx.array,
        model_codes: mx.array,
        state: TalkerStreamState | None,
    ) -> tuple[int, int]:
        batch, frames = self._validate_common_inputs(conditioning, user_codes, state)
        if model_codes.shape != user_codes.shape:
            raise ValueError("model_codes must match user_codes")
        _validate_discrete_ids(
            model_codes,
            cardinality=self.audio_config.codebook_size,
            name="model audio",
        )
        return batch, frames

    def _shift_codes(
        self,
        codes: mx.array,
        previous: mx.array | None,
    ) -> mx.array:
        if previous is None:
            previous = mx.full(
                (codes.shape[0], self.audio_config.num_codebooks),
                self.audio_config.codebook_size,
                dtype=mx.int32,
            )
        return mx.concatenate((previous[:, None, :], codes[:, :-1, :]), axis=1)

    def _hidden_from_shifted_codes(
        self,
        conditioning: mx.array,
        shifted_user_codes: mx.array,
        shifted_model_codes: mx.array,
        state: TalkerStreamState | None,
    ) -> mx.array:
        hidden = self.conditioning_projection(conditioning)
        for index in range(self.audio_config.num_codebooks):
            hidden = hidden + self.user_embeddings[index](
                shifted_user_codes[:, :, index]
            )
            hidden = hidden + self.model_embeddings[index](
                shifted_model_codes[:, :, index]
            )
        hidden = hidden / (2 * self.audio_config.num_codebooks + 1) ** 0.5
        for index, block in enumerate(self.temporal_blocks):
            hidden = block(hidden, None if state is None else state.temporal[index])
        return hidden

    def _output_from_hidden(
        self,
        hidden: mx.array,
        model_active: mx.array,
    ) -> TalkerOutput:
        batch, frames, _ = hidden.shape
        logits_shape = (
            batch,
            frames,
            self.audio_config.num_codebooks,
            self.audio_config.codebook_size,
        )
        return TalkerOutput(
            user_logits=self.user_head(hidden).reshape(*logits_shape),
            model_logits=self.model_head(hidden).reshape(*logits_shape),
            hidden=hidden,
            model_active=model_active,
        )

    def _forward(
        self,
        conditioning: mx.array,
        user_codes: mx.array,
        model_codes: mx.array,
        state: TalkerStreamState | None,
    ) -> TalkerOutput:
        batch, frames = self._validate_inputs(
            conditioning, user_codes, model_codes, state
        )
        if conditioning.shape[1] == 1:
            conditioning = mx.broadcast_to(
                conditioning, (batch, frames, self.thinker_dim)
            )
        previous_user = None if state is None else state.previous_user_codes
        previous_model = None if state is None else state.previous_model_codes
        hidden = self._hidden_from_shifted_codes(
            conditioning,
            self._shift_codes(user_codes, previous_user),
            self._shift_codes(model_codes, previous_model),
            state,
        )
        if state is None:
            model_active = mx.ones((batch,), dtype=mx.bool_)
        else:
            state.previous_user_codes = user_codes[:, -1, :]
            state.previous_model_codes = model_codes[:, -1, :]
            state.offset += frames
            model_active = state.model_active
        return self._output_from_hidden(hidden, model_active)

    def __call__(
        self, conditioning: mx.array, user_codes: mx.array, model_codes: mx.array
    ) -> TalkerOutput:
        return self._forward(conditioning, user_codes, model_codes, None)

    def forward_stream(
        self,
        conditioning: mx.array,
        user_codes: mx.array,
        model_codes: mx.array,
        state: TalkerStreamState,
    ) -> tuple[TalkerOutput, TalkerStreamState]:
        """Teacher-forced causal training with explicit stream state."""

        return self._forward(conditioning, user_codes, model_codes, state), state

    def generate_stream(
        self,
        conditioning: mx.array,
        user_codes: mx.array,
        state: TalkerStreamState,
        *,
        sampling: Literal["greedy"] = "greedy",
    ) -> tuple[TalkerGeneration, TalkerStreamState]:
        """Greedily generate model frames using only prior selected codes.

        The loop is frame-autoregressive: a selected model frame is committed
        to ``state`` only after its logits have been produced, and it becomes
        input only to the following frame. No checkpoint or external inference
        service participates in this path.
        """

        if sampling != "greedy":
            raise ValueError("Only deterministic greedy sampling is supported")
        batch, frames = self._validate_common_inputs(conditioning, user_codes, state)
        if conditioning.shape[1] == 1:
            conditioning = mx.broadcast_to(
                conditioning, (batch, frames, self.thinker_dim)
            )

        selected_parts: list[mx.array] = []
        user_logit_parts: list[mx.array] = []
        model_logit_parts: list[mx.array] = []
        hidden_parts: list[mx.array] = []
        for frame in range(frames):
            start_codes = mx.full(
                (batch, 1, self.audio_config.num_codebooks),
                self.audio_config.codebook_size,
                dtype=mx.int32,
            )
            shifted_user = (
                start_codes
                if state.previous_user_codes is None
                else state.previous_user_codes[:, None, :]
            )
            shifted_model = (
                start_codes
                if state.previous_model_codes is None
                else state.previous_model_codes[:, None, :]
            )
            hidden = self._hidden_from_shifted_codes(
                conditioning[:, frame : frame + 1, :],
                shifted_user,
                shifted_model,
                state,
            )
            output = self._output_from_hidden(hidden, state.model_active)
            # Argmax is intrinsically bounded by the validated head width. Do
            # not force a host synchronization here; the selected codes remain
            # a lazy device-side dependency of the next autoregressive frame.
            selected = mx.argmax(output.model_logits, axis=-1).astype(mx.int32)

            # Only the actual greedy choice is fed back. There is no caller-
            # provided placeholder model code on this inference path.
            state.previous_user_codes = user_codes[:, frame, :]
            state.previous_model_codes = selected[:, 0, :]
            state.offset += 1
            selected_parts.append(selected)
            user_logit_parts.append(output.user_logits)
            model_logit_parts.append(output.model_logits)
            hidden_parts.append(output.hidden)

        return (
            TalkerGeneration(
                codes=mx.concatenate(selected_parts, axis=1),
                user_logits=mx.concatenate(user_logit_parts, axis=1),
                model_logits=mx.concatenate(model_logit_parts, axis=1),
                hidden=mx.concatenate(hidden_parts, axis=1),
                model_active=state.model_active,
            ),
            state,
        )


@dataclass(frozen=True)
class TimedText:
    timestamp: float
    tokens: mx.array


@dataclass(frozen=True)
class TimedAudio:
    timestamp: float
    codes: mx.array
    stream: AudioStreamName = "user"


@dataclass(frozen=True)
class TimedVideo:
    frames: mx.array
    timestamps: mx.array
    difference_scores: mx.array | None = None


@dataclass(frozen=True)
class TimedInterruption:
    timestamp: float


TimelineEvent: TypeAlias = TimedText | TimedAudio | TimedVideo | TimedInterruption


@dataclass(frozen=True)
class ControlTokenUse:
    name: str
    token_id: int
    offset: int


@dataclass(frozen=True)
class TimelineEncoding:
    embeddings: mx.array
    timestamps: mx.array
    kinds: tuple[str, ...]
    controls: tuple[ControlTokenUse, ...]


@dataclass(frozen=True)
class MultimodalOutput:
    text_logits: mx.array
    hidden: mx.array
    timeline: TimelineEncoding
    audio: TalkerOutput | None = None


@dataclass(frozen=True)
class _TimelineUnit:
    sort_time: float
    event_order: int
    unit_order: int
    embeddings: mx.array
    timestamps: mx.array
    kinds: tuple[str, ...]
    control_name: str | None = None
    control_id: int | None = None


class ScratchMultimodalLM(nn.Module):
    """Thinker/talker scaffold sharing a normal scratch :class:`HybridLM`."""

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        registry: ModalityTokenRegistry | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MultimodalConfig()
        self.registry = registry or ModalityTokenRegistry()
        registry_ids = list(self.registry.to_dict().values())[1:]
        if max(registry_ids) >= self.config.thinker.vocab_size:
            raise ValueError(
                "Thinker vocabulary does not include modality extension IDs"
            )
        self.thinker = HybridLM(self.config.thinker)
        self.audio_codec = StreamingWaveformCodec(self.config.audio)
        self.video_encoder = SelectedFrameVideoEncoder(self.config.video)
        self.audio_projection = nn.Linear(
            self.config.audio.latent_dim, self.config.thinker.model_dim, bias=False
        )
        self.audio_stream_embedding = nn.Embedding(2, self.config.thinker.model_dim)
        self.time_input = nn.Linear(2, self.config.thinker.model_dim, bias=False)
        self.time_output = nn.Linear(
            self.config.thinker.model_dim, self.config.thinker.model_dim, bias=False
        )
        self.talker = DualStreamAudioTalker(
            self.config.thinker.model_dim, self.config.audio, self.config.talker
        )

    def __call__(self, tokens: mx.array) -> mx.array:
        """Preserve the ordinary text-only ``HybridLM(tokens)`` API."""

        return self.thinker(tokens)

    def forward_text(self, tokens: mx.array) -> mx.array:
        return self.thinker(tokens)

    def init_thinker_cache(self, batch_size: int) -> HybridCache:
        return self.thinker.init_cache(batch_size)

    def _marker_unit(
        self,
        name: str,
        token_id: int,
        timestamp: float,
        batch_size: int,
        event_order: int,
        unit_order: int,
    ) -> _TimelineUnit:
        ids = mx.full((batch_size, 1), token_id, dtype=mx.int32)
        return _TimelineUnit(
            sort_time=timestamp,
            event_order=event_order,
            unit_order=unit_order,
            embeddings=self.thinker.token_embedding(ids),
            timestamps=mx.full((batch_size, 1), timestamp, dtype=mx.float32),
            kinds=(name,),
            control_name=name,
            control_id=token_id,
        )

    def encode_timeline(self, events: Sequence[TimelineEvent]) -> TimelineEncoding:
        if not events:
            raise ValueError("At least one timeline event is required")
        units: list[_TimelineUnit] = []
        batch_candidates: list[int] = []
        for event in events:
            if isinstance(event, TimedText):
                if event.tokens.ndim < 1:
                    raise ValueError("Timed text tokens must include a batch dimension")
                batch_candidates.append(event.tokens.shape[0])
            elif isinstance(event, TimedAudio):
                if event.codes.ndim < 1:
                    raise ValueError("Timed audio codes must include a batch dimension")
                batch_candidates.append(event.codes.shape[0])
            elif isinstance(event, TimedVideo):
                if event.frames.ndim < 1:
                    raise ValueError(
                        "Timed video frames must include a batch dimension"
                    )
                batch_candidates.append(event.frames.shape[0])
        if not batch_candidates:
            raise ValueError(
                "Timeline needs text, audio, or video to establish a batch"
            )
        if any(candidate != batch_candidates[0] for candidate in batch_candidates):
            raise ValueError("All timeline events must share a batch size")
        batch_size: int | None = batch_candidates[0]

        def establish_batch(candidate: int) -> int:
            nonlocal batch_size
            if candidate <= 0:
                raise ValueError("Timeline batches must be non-empty")
            if batch_size is None:
                batch_size = candidate
            elif batch_size != candidate:
                raise ValueError("All timeline events must share a batch size")
            return batch_size

        for event_order, event in enumerate(events):
            if isinstance(event, TimedText):
                if not math.isfinite(event.timestamp):
                    raise ValueError("Timeline timestamps must be finite")
                if event.tokens.ndim != 2 or event.tokens.shape[1] <= 0:
                    raise ValueError(
                        "Timed text tokens must have shape [batch, length]"
                    )
                _validate_discrete_ids(
                    event.tokens,
                    cardinality=self.config.thinker.vocab_size,
                    name="text token",
                )
                batch = establish_batch(event.tokens.shape[0])
                units.append(
                    _TimelineUnit(
                        sort_time=float(event.timestamp),
                        event_order=event_order,
                        unit_order=0,
                        embeddings=self.thinker.token_embedding(event.tokens),
                        timestamps=mx.full(
                            (batch, event.tokens.shape[1]),
                            event.timestamp,
                            dtype=mx.float32,
                        ),
                        kinds=tuple("text" for _ in range(event.tokens.shape[1])),
                    )
                )
            elif isinstance(event, TimedAudio):
                if not math.isfinite(event.timestamp):
                    raise ValueError("Timeline timestamps must be finite")
                if event.stream not in {"user", "model"}:
                    raise ValueError("Audio stream must be 'user' or 'model'")
                if event.codes.ndim != 3 or event.codes.shape[1] <= 0:
                    raise ValueError(
                        "Timed audio codes must have shape [batch, frames, codebooks]"
                    )
                batch = establish_batch(event.codes.shape[0])
                quantized = self.audio_codec.quantizer.decode_codes(event.codes)
                stream_index = 0 if event.stream == "user" else 1
                stream_ids = mx.full(
                    (batch, event.codes.shape[1]), stream_index, dtype=mx.int32
                )
                audio_embeddings = self.audio_projection(
                    quantized
                ) + self.audio_stream_embedding(stream_ids)
                start_name = f"audio_{event.stream}_start"
                end_name = f"audio_{event.stream}_end"
                start_id = getattr(self.registry, start_name)
                end_id = getattr(self.registry, end_name)
                units.append(
                    self._marker_unit(
                        start_name, start_id, event.timestamp, batch, event_order, 0
                    )
                )
                frame_duration = (
                    self.config.audio.frame_size / self.config.audio.sample_rate
                )
                for frame_index in range(event.codes.shape[1]):
                    timestamp = event.timestamp + frame_index * frame_duration
                    units.append(
                        _TimelineUnit(
                            sort_time=timestamp,
                            event_order=event_order,
                            unit_order=frame_index + 1,
                            embeddings=audio_embeddings[
                                :, frame_index : frame_index + 1, :
                            ],
                            timestamps=mx.full((batch, 1), timestamp, dtype=mx.float32),
                            kinds=(f"audio_{event.stream}",),
                        )
                    )
                end_timestamp = event.timestamp + event.codes.shape[1] * frame_duration
                units.append(
                    self._marker_unit(
                        end_name,
                        end_id,
                        end_timestamp,
                        batch,
                        event_order,
                        event.codes.shape[1] + 1,
                    )
                )
            elif isinstance(event, TimedVideo):
                batch = establish_batch(event.frames.shape[0])
                video = self.video_encoder(
                    event.frames, event.timestamps, event.difference_scores
                )
                reference_times = video.timestamps[0, :, 0]
                if not mx.allclose(
                    video.timestamps,
                    mx.broadcast_to(
                        reference_times[None, :, None], video.timestamps.shape
                    ),
                    atol=1e-6,
                ).item():
                    raise ValueError(
                        "Batched video timelines must use the same selected-frame timestamps"
                    )
                start_time = float(reference_times[0].item())
                units.append(
                    self._marker_unit(
                        "video_start",
                        self.registry.video_start,
                        start_time,
                        batch,
                        event_order,
                        0,
                    )
                )
                unit_order = 1
                for frame_index in range(video.frames):
                    timestamp = float(reference_times[frame_index].item())
                    units.append(
                        self._marker_unit(
                            "video_frame",
                            self.registry.video_frame,
                            timestamp,
                            batch,
                            event_order,
                            unit_order,
                        )
                    )
                    unit_order += 1
                    units.append(
                        _TimelineUnit(
                            sort_time=timestamp,
                            event_order=event_order,
                            unit_order=unit_order,
                            embeddings=video.embeddings[:, frame_index, :, :],
                            timestamps=video.timestamps[:, frame_index, :],
                            kinds=tuple("video" for _ in range(video.tokens_per_frame)),
                        )
                    )
                    unit_order += 1
                end_time = float(reference_times[-1].item())
                units.append(
                    self._marker_unit(
                        "video_end",
                        self.registry.video_end,
                        end_time,
                        batch,
                        event_order,
                        unit_order,
                    )
                )
            elif isinstance(event, TimedInterruption):
                if not math.isfinite(event.timestamp):
                    raise ValueError("Timeline timestamps must be finite")
                if batch_size is None:
                    raise ValueError(
                        "An interruption cannot establish the timeline batch size"
                    )
                units.append(
                    self._marker_unit(
                        "interruption",
                        self.registry.interruption,
                        event.timestamp,
                        batch_size,
                        event_order,
                        0,
                    )
                )
            else:
                raise TypeError(f"Unsupported timeline event: {type(event).__name__}")

        units.sort(key=lambda unit: (unit.sort_time, unit.event_order, unit.unit_order))
        raw_embeddings = mx.concatenate([unit.embeddings for unit in units], axis=1)
        timestamps = mx.concatenate([unit.timestamps for unit in units], axis=1)
        deltas = mx.concatenate(
            (
                mx.zeros((timestamps.shape[0], 1), dtype=timestamps.dtype),
                timestamps[:, 1:] - timestamps[:, :-1],
            ),
            axis=1,
        )
        time_features = mx.stack(
            (timestamps / self.config.time_scale_seconds, deltas), axis=-1
        )
        embeddings = raw_embeddings + self.time_output(
            nn.silu(self.time_input(time_features))
        )
        kinds: list[str] = []
        controls: list[ControlTokenUse] = []
        offset = 0
        for unit in units:
            kinds.extend(unit.kinds)
            if unit.control_name is not None and unit.control_id is not None:
                controls.append(
                    ControlTokenUse(unit.control_name, unit.control_id, offset)
                )
            offset += unit.embeddings.shape[1]
        return TimelineEncoding(
            embeddings=embeddings,
            timestamps=timestamps,
            kinds=tuple(kinds),
            controls=tuple(controls),
        )

    def _think_from_embeddings(
        self, embeddings: mx.array, cache: HybridCache | None = None
    ) -> tuple[mx.array, mx.array, HybridCache | None]:
        if (
            embeddings.ndim != 3
            or embeddings.shape[-1] != self.config.thinker.model_dim
        ):
            raise ValueError("Timeline embeddings have an incompatible thinker shape")
        offset = 0 if cache is None else cache.offset
        if cache is None and embeddings.shape[1] > self.config.thinker.context_length:
            raise ValueError("Multimodal timeline exceeds thinker context length")
        if cache is not None:
            if cache.batch_size != embeddings.shape[0]:
                raise ValueError("Thinker cache batch size does not match timeline")
            if len(cache.layers) != len(self.thinker.blocks):
                raise ValueError("Thinker cache layer count does not match model")
            cache.reserve(embeddings.shape[1], self.config.thinker.context_length)
        hidden = embeddings
        for index, block in enumerate(self.thinker.blocks):
            hidden = block(
                hidden,
                offset=offset,
                cache=None if cache is None else cache.layers[index],
            )
        hidden = self.thinker.final_norm(hidden)
        logits = hidden @ self.thinker.token_embedding.weight.T
        if cache is not None:
            cache.offset += embeddings.shape[1]
            cache.resident_tokens += embeddings.shape[1]
        return hidden, logits, cache

    def forward_timeline(
        self,
        events: Sequence[TimelineEvent],
        *,
        thinker_cache: HybridCache | None = None,
        user_codes: mx.array | None = None,
        model_codes: mx.array | None = None,
        talker_state: TalkerStreamState | None = None,
    ) -> MultimodalOutput:
        timeline = self.encode_timeline(events)
        hidden, text_logits, _ = self._think_from_embeddings(
            timeline.embeddings, thinker_cache
        )
        audio_output: TalkerOutput | None = None
        if (user_codes is None) != (model_codes is None):
            raise ValueError("user_codes and model_codes must be supplied together")
        if user_codes is not None and model_codes is not None:
            conditioning = hidden[:, -1:, :]
            if talker_state is None:
                audio_output = self.talker(conditioning, user_codes, model_codes)
            else:
                audio_output, _ = self.talker.forward_stream(
                    conditioning, user_codes, model_codes, talker_state
                )
        return MultimodalOutput(
            text_logits=text_logits,
            hidden=hidden,
            timeline=timeline,
            audio=audio_output,
        )

    def parameter_count(self) -> int:
        return sum(value.size for _, value in tree_flatten(self.parameters()))

    def parameter_report(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(value.size for _, value in tree_flatten(module.parameters()))

        components = {
            "thinker": count(self.thinker),
            "audio_codec": count(self.audio_codec),
            "video_encoder": count(self.video_encoder),
            "talker": count(self.talker),
            "audio_projection": count(self.audio_projection),
            "audio_stream_embedding": count(self.audio_stream_embedding),
            "time_embedding": count(self.time_input) + count(self.time_output),
        }
        components["total"] = self.parameter_count()
        return components


def create_scratch_multimodal_model(
    config: MultimodalConfig | None = None,
) -> ScratchMultimodalLM:
    """Construct fresh random multimodal modules without reading any weights."""

    return ScratchMultimodalLM(config)
