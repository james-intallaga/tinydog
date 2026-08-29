# Scratch-native realtime audio/video MLM

This project now contains a complete **scratch-only architecture, training
boundary, streaming runtime, and local realtime transport** for a native
audio/video model. It does not import pretrained speech, vision, codec, or
language weights and it does not call an outside ASR, TTS, or inference model.

The full example is 325,754,496 parameters:

- a 320,929,536-parameter hybrid convolution/GQA thinker;
- a trainable 24 kHz streaming waveform codec;
- a fixed-token selected-frame vision encoder;
- a causal dual-stream audio talker; and
- audio, modality, and time adapters.

`configs/multimodal-pilot-31m.json` is a 31,004,864-parameter version for
proving the data, objective, checkpoint, and latency path before spending the
resources required by the full model. Both models always begin from this
project's random initialization.

There is deliberately **no trained native checkpoint or media dataset in the
repository**. The live gateway therefore loads a lightweight silent runtime,
reports that the model and trained voice are unavailable, and never plays
random waveform output. Architecture and transport readiness are not described
as learned capability.

## Realtime conversation path

The browser and Python gateway now implement the path that a trained model will
use:

1. The browser obtains a short-lived, one-use, loopback-only admission token.
2. WebSocket signalling establishes a WebRTC peer connection.
3. The microphone is sent as a duplex WebRTC audio track. The gateway decodes
   it into bounded, chronological, mono 24 kHz PCM blocks.
4. Camera snapshots are compared with the last frame the model actually
   accepted. The first frame, a heartbeat, or a frame meeting the configurable
   10–100% difference threshold is sent over an unordered latest-only data
   channel.
5. The gateway canonicalizes accepted JPEGs to 320 × 192 before the native
   runtime sees them. Browser wall-clock timestamps never become model input;
   the runtime maps audio and video to one session-relative monotonic clock.
6. A trained runtime would stream 20 ms PCM responses back as the WebRTC remote
   audio track. Interruptions fence old playback generations and clear unheard
   server output.

The same-origin HTTP media path remains a bounded fallback and is off by
default. It is not silently substituted for WebRTC. The status display keeps
transport, model, trained voice, vision, and end-to-end readiness separate.

## Model components

### Streaming waveform codec

`StreamingWaveformCodec` consumes mono 24 kHz waveform frames. The full
configuration uses 480 samples per frame, or 20 ms. A causal encoder maps each
frame into residual-vector-quantized codes and a causal decoder reconstructs
waveform frames. Streaming state owns incomplete samples, convolution history,
codebook history, counters, and finalization state.

The codec exposes reconstruction, codebook, and commitment objectives. These
are trainable hooks, not evidence that the current random codec produces useful
speech. Runtime audio emission requires a held-out codec SNR gate.

### Difference-selected video

`SelectedFrameVideoEncoder` patchifies canonical RGB frames and uses learned
queries to resample every accepted frame to a fixed number of thinker tokens:
12 in the full model and 8 in the pilot. This prevents camera resolution or
session length from creating an unbounded token rate. Timestamps, deltas, and
frame-difference metadata are encoded on the same causal timeline as audio.

Frame selection stays in the browser for privacy and bandwidth control. The
comparison baseline advances only after a valid `frameAccepted` acknowledgement,
so a dropped frame never becomes the hidden reference frame.

### Thinker and talker

`ScratchMultimodalLM` uses the same hybrid decoder as the text-only scratch
research track. Text, user-audio, model-audio, selected-video, and interruption
boundaries use frozen modality IDs reserved by the scratch tokenizer.

`DualStreamAudioTalker` predicts user and model codec streams causally. Both
training streams are shifted to prevent same-step target leakage. Streaming
state preserves prior user/model codes and interruption state; inference only
commits codes it actually selected. The native runtime gives each session
private queues and caches while sharing one immutable loaded model.

## Scratch training and promotion

`mini_arithmetic.native_data`, `native_objectives`, and `native_train` provide
the staged path:

1. codec reconstruction and quantization;
2. speech understanding with assistant/text masking;
3. selected-frame vision grounding and a frame-ablation check;
4. causal talker prediction;
5. duplex conversation; and
6. low-learning-rate joint tuning.

Media shards are checksum-bound and carry source, consent, rights, speaker, and
conversation provenance. Canonical keys prevent a speaker/conversation family
from leaking between train and validation. Resume state, tokenizer identity,
model configuration, and held-out validation manifests are checksum-bound.

A low training loss alone cannot enable the runtime. A native checkpoint
manifest must prove scratch-only provenance, positive optimizer progress, and
held-out evidence for codec SNR, speech accuracy, vision accuracy plus frame
ablation, talker code accuracy, and duplex text/code accuracy. The loader
rechecks the manifest, files, hashes, configuration, sample rate, frame size,
and every gate before allowing PCM output.

See `configs/native-pilot-training.example.json` for the staged pilot. Its
paths and tokenizer digest are placeholders; no dataset is bundled, and
runtime promotion is disabled in the example.

## Latency and interruption gates

The gateway measures one correlated monotonic timeline rather than combining
browser and server wall clocks. Realtime readiness fails closed until it has at
least 20 complete samples for each required span and all of these hold:

- audio capture-to-runtime-accept p95 at or below 50 ms;
- interrupt-to-confirmed-browser-silence p95 at or below 100 ms;
- end-of-turn-to-first rendered non-silent response audio p95 at or below
  300 ms;
- queue age p95 at or below 100 ms, maximum depth 4, and no sustained growth.

The browser's local voice detector marks candidate turn endings and confirms
rendered non-silent audio. It is transport telemetry, not a learned endpoint
policy. A useful trained model still needs endpoint/interruption behavior in
its conversation curriculum and full-model latency measurement on the target
Mac.

## Clean-room scope and current truth

The architecture uses public high-level ideas such as streaming audio-code
modeling, a Thinker–Talker split, mixed efficient/exact attention, and causal
multimodal timing. The implementation was written for this repository with
ordinary MLX layers. No third-party model code, checkpoint, voice, codec table,
vision backbone, or generated provider output is embedded.

What is ready now: the model structure, causal/cached inference boundary,
strict checkpoint loader, data/objective/trainer infrastructure, WebRTC
gateway, selected-frame protocol, interruption fencing, bounded queues,
browser controls, and fail-closed telemetry.

What is not ready: an audited native training corpus, a checkpoint that passes
all five quality gates, learned endpoint behavior, intelligible speech, or a
measured full-model sub-300 ms conversation loop. Until those exist, the UI
must continue to say `Model · disconnected` and `Trained voice · not trained`.
