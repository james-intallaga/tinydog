# tinydog.ai local interface

This Next.js app is the local chat and teaching surface for the Python learning service in the parent folder.

Use the combined launcher from the project root:

```bash
./tinydog.sh start
```

For interface-only development, keep the Python service on `127.0.0.1:8765`, then run:

```bash
npm ci
npm run dev
```

The normal portable service turns every eligible typed reply into one private
dialogue demonstration and checks the five-turn local full-foundation trigger. The server-side
routes proxy chat, history, status, teaching, and one-use
realtime admission requests so local services never become configurable browser
targets. Only the non-secret local chat session ID is kept in browser storage;
WebRTC admission objects and access tokens remain memory-only. Messages and
lessons are persisted by the Python service in SQLite.

Press Command+B on macOS or Ctrl+B on Windows/Linux to open the text box.
Outside a text field, Command+V or Ctrl+V cycles to the next companion; normal
paste remains available while editing text. The pet choice and opaque chat
session identifier are the only values stored in browser local storage.

On Apple Silicon, the camera and voice controls can use the local WebRTC gateway on
`127.0.0.1:8787` when its transport is online. Microphone audio is a live WebRTC
track and model audio plays through the returned track. Camera frames use a
bounded latest-only data channel when their RGB difference from the last frame
the model acknowledged meets the configurable threshold (10–100%), plus
heartbeat frames. Transport, model, and trained-voice status remain separate;
an online WebRTC link does not claim that an untrained model can converse.
When the gateway is deliberately started with random-initialized native audio,
the normal Voice action plays that WebRTC output and the interface labels it as
random and unintelligible. The portable PyTorch Windows/Linux launcher does
not start this still-MLX-only neural media runtime. This diagnostic sound does not change any
trained-model readiness indicator.

A slower HTTP media path remains available only through the explicit fallback
setting. HTTP frame acceptance never advances the model-acknowledged video
baseline. Classon itself is only a read-only design reference and is not a
package or runtime dependency. See
[`../docs/classon-webrtc-port.md`](../docs/classon-webrtc-port.md).

Validation:

```bash
npm test
npm run lint
npm run build
```
