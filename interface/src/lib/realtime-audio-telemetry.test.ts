import assert from "node:assert/strict";
import test from "node:test";
import {
  ResponseAudioDetector,
  rmsLevel,
  VoiceEndpointDetector,
} from "./realtime-audio-telemetry";

test("voice endpoint requires real speech followed by bounded silence", () => {
  const detector = new VoiceEndpointDetector({
    speechThreshold: 0.02,
    silenceThreshold: 0.01,
    minimumSpeechMs: 40,
    endpointSilenceMs: 100,
  });
  assert.equal(detector.observe(0.03, 0), null);
  assert.equal(detector.observe(0.03, 40), "speechStarted");
  assert.equal(detector.observe(0.005, 60), null);
  assert.equal(detector.observe(0.005, 159), null);
  assert.equal(detector.observe(0.005, 160), "turnEnded");
});

test("brief noise and brief pauses do not create false endpoints", () => {
  const detector = new VoiceEndpointDetector({
    speechThreshold: 0.02,
    silenceThreshold: 0.01,
    minimumSpeechMs: 40,
    endpointSilenceMs: 100,
  });
  assert.equal(detector.observe(0.03, 0), null);
  assert.equal(detector.observe(0, 20), null);
  assert.equal(detector.observe(0.03, 100), null);
  assert.equal(detector.observe(0.03, 140), "speechStarted");
  assert.equal(detector.observe(0, 160), null);
  assert.equal(detector.observe(0.02, 220), null);
  assert.equal(detector.observe(0, 240), null);
  assert.equal(detector.observe(0, 340), "turnEnded");
});

test("response detector fires once for the armed turn", () => {
  const detector = new ResponseAudioDetector(0.01, 2);
  detector.arm("turn-1");
  assert.equal(detector.observe(0.02), null);
  assert.equal(detector.observe(0), null);
  assert.equal(detector.observe(0.02), null);
  assert.equal(detector.observe(0.02), "turn-1");
  assert.equal(detector.observe(0.02), null);
});

test("RMS is deterministic and bounds out-of-range samples", () => {
  assert.equal(rmsLevel(new Float32Array()), 0);
  assert.equal(rmsLevel(Float32Array.of(1, -1)), 1);
  assert.equal(rmsLevel(Float32Array.of(2, -2)), 1);
  assert.throws(() => rmsLevel(Float32Array.of(Number.NaN)), /finite/);
});
