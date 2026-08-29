# Classon-inspired MLM realtime transport

Classon was inspected read-only as an engineering reference. No Classon file,
dependency, service, configuration, credential, or data was changed or copied
into MLM. MLM owns this implementation and has no runtime dependency on the
Classon project.

## Implemented separation

The local transport follows the same useful separation of concerns:

- a reliable WebSocket control plane handles admission, join, heartbeat,
  offer/answer, ICE, reconnection, interruption, and correlated latency events;
- an encrypted WebRTC media plane carries the live microphone to the gateway
  and response audio back to the browser; and
- an MLM-specific unordered data channel carries only JPEG frames selected by
  the configurable difference filter.

The Python gateway is implemented in `mini_arithmetic.realtime_gateway`; the
browser client is implemented inside this project. The gateway is loopback-only
and its admission token is short-lived, single-use, session-bound, and never
placed in a URL. Signalling is size-bounded, strictly parsed, revision-fenced,
and protected against stale offers, answers, ICE, reconnect work, and playback
generations.

MLM intentionally rebuilds its peer after signalling failure rather than
trying to preserve a possibly stale peer indefinitely. Peer mutations are
serialized, connection/negotiation generations are checked after asynchronous
browser operations, join/negotiation/connect deadlines are bounded, and
cleanup is lifecycle-fenced.

## Selected-frame extension

Sending an ordinary camera track would transmit every encoded frame and defeat
the requested difference algorithm. The browser instead:

- compares a small RGB sample with the last model-accepted frame;
- selects the first frame, a heartbeat, or a frame meeting the configurable
  10–100% difference threshold;
- keeps only the newest unsent selected frame under backpressure;
- validates SCTP message limits before sending; and
- advances the comparison baseline only for the exact acknowledged sequence.

The gateway validates the binary header, MIME type, size, timestamp,
difference score, frame ID, and monotonic sequence before acknowledging
acceptance. The native runtime then performs a guarded JPEG decode and
canonical 320 × 192 conversion. A bounded same-origin HTTP ingress remains an
explicit fallback, disabled by default.

## Audio, interruption, and latency correlation

Inbound WebRTC audio is decoded and resampled into mono 24 kHz PCM. Runtime
queues are globally bounded; old or overloaded work is rejected rather than
allowed to grow without limit. A trained native runtime can return 20 ms PCM
blocks through the browser's remote audio track.

The browser applies interruption locally first—muting and pausing playback—then
sends the new playback generation. The gateway discards older queued output.
Three strict client telemetry messages close the end-to-end measurement loop:

- `turnEnded{turnId}` after the local mic detector observes a spoken turn and
  endpoint silence;
- `firstResponseAudio{turnId}` only after the browser actually observes
  non-silent, unmuted rendered remote audio; and
- `playbackSilent{playbackGeneration}` only after interruption has taken
  effect in the browser.

The gateway accepts those events only for live connection IDs, expected turns,
valid playback generations, and a response turn for which it already enqueued
non-silent trained-runtime PCM. Events are bounded and replay-protected. This
allows capture-to-accept, interruption-to-silence, and turn-to-render p50/p95
measurement on one gateway monotonic clock.

## Fail-closed capability status

The UI reports separate facts instead of one ambiguous “realtime” badge:

- **Transport** — admission/signalling/WebRTC are available;
- **Model** — a native checkpoint was configured and loaded for this session;
- **Trained voice** — codec, speech, talker, and conversation evidence passed;
- **Vision** — the held-out vision and frame-ablation gate passed; and
- **Conversation ready** — model, all quality gates, real latency samples, and
  queue gates passed together.

With no checkpoint, the current gateway intentionally uses a no-model silent
runtime. It can prove the transport, reconnection, frame acknowledgement, and
telemetry boundary, but it cannot produce sound. A random checkpoint is also
insufficient: the strict manifest and held-out evidence must pass before PCM
emission is enabled.

## Activation and verification gates

The implementation is covered by protocol, security, media, gateway, runtime,
browser lifecycle, reconnect, interruption, backpressure, and telemetry tests.
Actual native conversation still requires all of the following on the target
machine:

- a rights-cleared scratch training/validation corpus with no group leakage;
- a locally trained checkpoint passing codec, speech, vision, talker, and
  duplex conversation thresholds;
- browser permission/connect/stop/reconnect checks with a real mic and speaker;
- at least 20 complete samples for every latency span;
- p95 audio ingress at or below 50 ms, interrupt-to-silence at or below 100 ms,
  and turn-end-to-rendered-audio at or below 300 ms; and
- bounded queue age/depth with no sustained growth.

Until the checkpoint and measured gates pass, `Transport · online` together
with `Model · disconnected` and `Trained voice · not trained` is the correct,
intentional status.
