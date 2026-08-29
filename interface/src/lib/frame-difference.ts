export const MIN_FRAME_DIFFERENCE_PERCENT = 10;
export const MAX_FRAME_DIFFERENCE_PERCENT = 100;

export type FrameTransmissionReason =
  | "initial"
  | "difference"
  | "heartbeat"
  | "unchanged";

export type FrameTransmissionDecision = {
  send: boolean;
  keyframe: boolean;
  reason: FrameTransmissionReason;
};

/**
 * Returns normalized mean absolute RGB error as a percentage.
 *
 * For each pixel, the absolute red, green, and blue channel changes are
 * averaged and divided by the maximum possible channel change (255). Alpha is
 * intentionally ignored. Identical frames are 0%; black versus white is 100%.
 */
export function normalizedFrameDifferencePercent(
  current: Uint8ClampedArray,
  previous: Uint8ClampedArray,
): number {
  if (current.length !== previous.length) {
    throw new RangeError("Frames must have the same number of pixels.");
  }
  if (current.length === 0 || current.length % 4 !== 0) {
    throw new RangeError("Frames must contain non-empty RGBA pixel data.");
  }

  let totalDifference = 0;
  for (let index = 0; index < current.length; index += 4) {
    totalDifference += Math.abs(current[index] - previous[index]);
    totalDifference += Math.abs(current[index + 1] - previous[index + 1]);
    totalDifference += Math.abs(current[index + 2] - previous[index + 2]);
  }

  const rgbChannelCount = (current.length / 4) * 3;
  return (totalDifference / (rgbChannelCount * 255)) * 100;
}

export function clampFrameDifferenceThreshold(value: number): number {
  if (!Number.isFinite(value)) return MIN_FRAME_DIFFERENCE_PERCENT;
  return Math.min(
    MAX_FRAME_DIFFERENCE_PERCENT,
    Math.max(MIN_FRAME_DIFFERENCE_PERCENT, value),
  );
}

export function decideFrameTransmission({
  differencePercent,
  thresholdPercent,
  nowMs,
  lastSentAtMs,
  heartbeatMs,
  hasLastSentFrame,
}: {
  differencePercent: number;
  thresholdPercent: number;
  nowMs: number;
  lastSentAtMs: number | null;
  heartbeatMs: number;
  hasLastSentFrame: boolean;
}): FrameTransmissionDecision {
  if (!hasLastSentFrame || lastSentAtMs === null) {
    return { send: true, keyframe: true, reason: "initial" };
  }

  if (nowMs - lastSentAtMs >= Math.max(0, heartbeatMs)) {
    return { send: true, keyframe: true, reason: "heartbeat" };
  }

  if (differencePercent >= clampFrameDifferenceThreshold(thresholdPercent)) {
    return { send: true, keyframe: false, reason: "difference" };
  }

  return { send: false, keyframe: false, reason: "unchanged" };
}
