export type VideoFrameChunk = {
  blob: Blob;
  timestampMs: number;
  differencePercent: number;
  keyframe: boolean;
  sessionId: string;
  signal?: AbortSignal;
};

export type AudioChunk = {
  blob: Blob;
  timestampMs: number;
  sequence: number;
  mimeType: string;
  sessionId: string;
  signal?: AbortSignal;
};

/** Replace these methods later to connect capture to another local realtime ingress. */
export interface RealtimeMediaTransport {
  sendVideoFrame(frame: VideoFrameChunk): Promise<void>;
  sendAudioChunk(chunk: AudioChunk): Promise<void>;
}

export class MediaTransportError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "MediaTransportError";
  }
}

async function postLocalBinary(
  path: "/api/media/frame" | "/api/media/audio",
  blob: Blob,
  headers: Record<string, string | undefined>,
  signal?: AbortSignal,
) {
  const response = await fetch(path, {
    method: "POST",
    body: blob,
    cache: "no-store",
    signal,
    headers: Object.fromEntries(
      Object.entries(headers).filter((entry): entry is [string, string] => entry[1] !== undefined),
    ),
  });

  if (!response.ok) {
    let message = response.status === 501 ? "Local media receiver is not ready." : "Media was not accepted.";
    try {
      const data = (await response.json()) as { error?: unknown };
      if (typeof data.error === "string" && data.error.trim()) message = data.error;
    } catch {
      // A non-JSON backend error is intentionally replaced with a safe message.
    }
    throw new MediaTransportError(message, response.status);
  }
}

export const localRealtimeMediaTransport: RealtimeMediaTransport = {
  async sendVideoFrame(frame) {
    await postLocalBinary("/api/media/frame", frame.blob, {
      "Content-Type": "image/jpeg",
      "X-MLM-Session": frame.sessionId,
      "X-MLM-Timestamp-Ms": Math.round(frame.timestampMs).toString(),
      "X-MLM-Difference": frame.differencePercent.toFixed(4),
      "X-MLM-Keyframe": frame.keyframe ? "true" : "false",
    }, frame.signal);
  },

  async sendAudioChunk(chunk) {
    await postLocalBinary("/api/media/audio", chunk.blob, {
      "Content-Type": chunk.mimeType,
      "X-MLM-Session": chunk.sessionId,
      "X-MLM-Timestamp-Ms": Math.round(chunk.timestampMs).toString(),
      "X-MLM-Sequence": chunk.sequence.toString(),
    }, chunk.signal);
  },
};
