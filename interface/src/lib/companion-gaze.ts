export type Point = { x: number; y: number };
export type Bounds = { left: number; top: number; width: number; height: number };
export type SpringAxis = { position: number; velocity: number };

export const NEUTRAL_GAZE: Point = Object.freeze({ x: 0, y: 0 });
const SOFT_LIMIT = 1 - Number.EPSILON;

function softlyBound(value: number) {
  return Math.max(-SOFT_LIMIT, Math.min(SOFT_LIMIT, Math.tanh(value)));
}

export function gazeTarget(
  pointer: Point,
  bounds: Bounds,
  viewport: { width: number; height: number },
): Point {
  const centerX = bounds.left + bounds.width / 2;
  const centerY = bounds.top + bounds.height * 0.42;
  const horizontalReach = Math.max(viewport.width * 0.34, bounds.width * 0.72, 1);
  const verticalReach = Math.max(viewport.height * 0.3, bounds.height * 0.66, 1);

  return {
    x: softlyBound((pointer.x - centerX) / horizontalReach),
    y: softlyBound((pointer.y - centerY) / verticalReach),
  };
}

export function advanceSpring(
  axis: SpringAxis,
  target: number,
  deltaMilliseconds: number,
  stiffness: number,
  damping: number,
): SpringAxis {
  const deltaSeconds = Math.min(Math.max(deltaMilliseconds, 0) / 1000, 1 / 30);
  if (deltaSeconds === 0) return axis;

  const acceleration = (target - axis.position) * stiffness - axis.velocity * damping;
  const velocity = axis.velocity + acceleration * deltaSeconds;
  return {
    position: axis.position + velocity * deltaSeconds,
    velocity,
  };
}

export function springIsSettled(axis: SpringAxis, target: number, epsilon = 0.001): boolean {
  return Math.abs(axis.position - target) < epsilon && Math.abs(axis.velocity) < epsilon;
}
