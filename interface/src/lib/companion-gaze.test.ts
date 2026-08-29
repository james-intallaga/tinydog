import assert from "node:assert/strict";
import test from "node:test";
import { advanceSpring, gazeTarget, springIsSettled, type SpringAxis } from "./companion-gaze";

const bounds = { left: 300, top: 200, width: 400, height: 400 };
const viewport = { width: 1200, height: 800 };

test("gaze is neutral at the dog's visual center", () => {
  assert.deepEqual(
    gazeTarget({ x: 500, y: 368 }, bounds, viewport),
    { x: 0, y: 0 },
  );
});

test("gaze follows the pointer while staying softly bounded", () => {
  const lowerRight = gazeTarget({ x: 50_000, y: 50_000 }, bounds, viewport);
  const upperLeft = gazeTarget({ x: -50_000, y: -50_000 }, bounds, viewport);

  assert.ok(lowerRight.x > 0.99 && lowerRight.x < 1);
  assert.ok(lowerRight.y > 0.99 && lowerRight.y < 1);
  assert.ok(upperLeft.x < -0.99 && upperLeft.x > -1);
  assert.ok(upperLeft.y < -0.99 && upperLeft.y > -1);
});

test("spring approaches its target and settles without an abrupt jump", () => {
  let axis: SpringAxis = { position: 0, velocity: 0 };
  axis = advanceSpring(axis, 1, 16.67, 150, 25);
  assert.ok(axis.position > 0 && axis.position < 0.1);

  for (let frame = 0; frame < 180; frame += 1) {
    axis = advanceSpring(axis, 1, 16.67, 150, 25);
  }

  assert.ok(Math.abs(axis.position - 1) < 0.001);
  assert.equal(springIsSettled(axis, 1), true);
});

test("a paused animation does not change spring state", () => {
  const axis = { position: 0.25, velocity: 0.5 };
  assert.equal(advanceSpring(axis, 1, 0, 150, 25), axis);
});
