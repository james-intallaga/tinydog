import assert from "node:assert/strict";
import test from "node:test";
import {
  clampFrameDifferenceThreshold,
  decideFrameTransmission,
  normalizedFrameDifferencePercent,
} from "./frame-difference";

function pixels(...values: number[]) {
  return new Uint8ClampedArray(values);
}

test("identical frames have zero difference", () => {
  const frame = pixels(12, 34, 56, 255, 78, 90, 123, 255);
  assert.equal(normalizedFrameDifferencePercent(frame, frame), 0);
});

test("black and white frames have 100 percent difference", () => {
  const black = pixels(0, 0, 0, 255, 0, 0, 0, 255);
  const white = pixels(255, 255, 255, 255, 255, 255, 255, 255);
  assert.equal(normalizedFrameDifferencePercent(black, white), 100);
});

test("the metric averages RGB channels and ignores alpha", () => {
  const first = pixels(0, 0, 0, 0);
  const red = pixels(255, 0, 0, 255);
  assert.ok(Math.abs(normalizedFrameDifferencePercent(first, red) - 100 / 3) < 1e-10);
});

test("invalid frame shapes are rejected", () => {
  assert.throws(
    () => normalizedFrameDifferencePercent(pixels(0, 0, 0, 255), pixels(0, 0, 0, 255, 0)),
    RangeError,
  );
  assert.throws(
    () => normalizedFrameDifferencePercent(new Uint8ClampedArray(), new Uint8ClampedArray()),
    RangeError,
  );
});

test("thresholds are constrained to 10 through 100 percent", () => {
  assert.equal(clampFrameDifferenceThreshold(4), 10);
  assert.equal(clampFrameDifferenceThreshold(42), 42);
  assert.equal(clampFrameDifferenceThreshold(140), 100);
  assert.equal(clampFrameDifferenceThreshold(Number.NaN), 10);
});

test("first frames and heartbeat frames are forced keyframes", () => {
  assert.deepEqual(
    decideFrameTransmission({
      differencePercent: 0,
      thresholdPercent: 10,
      nowMs: 1_000,
      lastSentAtMs: null,
      heartbeatMs: 5_000,
      hasLastSentFrame: false,
    }),
    { send: true, keyframe: true, reason: "initial" },
  );

  assert.deepEqual(
    decideFrameTransmission({
      differencePercent: 0,
      thresholdPercent: 10,
      nowMs: 6_000,
      lastSentAtMs: 1_000,
      heartbeatMs: 5_000,
      hasLastSentFrame: true,
    }),
    { send: true, keyframe: true, reason: "heartbeat" },
  );
});

test("changed frames send at the threshold while quieter frames skip", () => {
  assert.equal(
    decideFrameTransmission({
      differencePercent: 10,
      thresholdPercent: 10,
      nowMs: 2_000,
      lastSentAtMs: 1_000,
      heartbeatMs: 5_000,
      hasLastSentFrame: true,
    }).reason,
    "difference",
  );
  assert.deepEqual(
    decideFrameTransmission({
      differencePercent: 9.99,
      thresholdPercent: 10,
      nowMs: 2_000,
      lastSentAtMs: 1_000,
      heartbeatMs: 5_000,
      hasLastSentFrame: true,
    }),
    { send: false, keyframe: false, reason: "unchanged" },
  );
});
