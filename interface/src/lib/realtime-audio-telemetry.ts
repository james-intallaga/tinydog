export type VoiceEndpointEvent = "speechStarted" | "turnEnded" | null;

export type VoiceEndpointConfig = {
  speechThreshold: number;
  silenceThreshold: number;
  minimumSpeechMs: number;
  endpointSilenceMs: number;
};

const DEFAULT_ENDPOINT_CONFIG: VoiceEndpointConfig = {
  speechThreshold: 0.018,
  silenceThreshold: 0.010,
  minimumSpeechMs: 80,
  endpointSilenceMs: 180,
};

function validLevel(level: number) {
  return Number.isFinite(level) && level >= 0 && level <= 1;
}

/** Small, deterministic client VAD used only to timestamp a turn boundary. */
export class VoiceEndpointDetector {
  private speaking = false;
  private speechStartedAt: number | null = null;
  private silenceStartedAt: number | null = null;

  constructor(private readonly config = DEFAULT_ENDPOINT_CONFIG) {
    if (
      !validLevel(config.speechThreshold) ||
      !validLevel(config.silenceThreshold) ||
      config.silenceThreshold >= config.speechThreshold ||
      !Number.isFinite(config.minimumSpeechMs) ||
      config.minimumSpeechMs < 0 ||
      !Number.isFinite(config.endpointSilenceMs) ||
      config.endpointSilenceMs <= 0
    ) {
      throw new Error("Voice endpoint configuration is invalid.");
    }
  }

  observe(level: number, nowMs: number): VoiceEndpointEvent {
    if (!validLevel(level) || !Number.isFinite(nowMs) || nowMs < 0) {
      throw new Error("Voice endpoint observation is invalid.");
    }
    if (!this.speaking) {
      if (level < this.config.speechThreshold) {
        this.speechStartedAt = null;
        return null;
      }
      this.speechStartedAt ??= nowMs;
      if (nowMs - this.speechStartedAt < this.config.minimumSpeechMs) return null;
      this.speaking = true;
      this.silenceStartedAt = null;
      return "speechStarted";
    }

    if (level > this.config.silenceThreshold) {
      this.silenceStartedAt = null;
      return null;
    }
    this.silenceStartedAt ??= nowMs;
    if (nowMs - this.silenceStartedAt < this.config.endpointSilenceMs) return null;
    this.reset();
    return "turnEnded";
  }

  reset() {
    this.speaking = false;
    this.speechStartedAt = null;
    this.silenceStartedAt = null;
  }
}

export class ResponseAudioDetector {
  private consecutive = 0;
  private armedTurnId: string | null = null;

  constructor(
    private readonly threshold = 0.003,
    private readonly requiredSamples = 2,
  ) {
    if (!validLevel(threshold) || threshold <= 0 || requiredSamples < 1) {
      throw new Error("Response-audio detector configuration is invalid.");
    }
  }

  arm(turnId: string) {
    if (!turnId) throw new Error("A response turn ID is required.");
    this.armedTurnId = turnId;
    this.consecutive = 0;
  }

  observe(level: number) {
    if (!validLevel(level)) throw new Error("Response audio level is invalid.");
    if (this.armedTurnId === null) return null;
    this.consecutive = level >= this.threshold ? this.consecutive + 1 : 0;
    if (this.consecutive < this.requiredSamples) return null;
    const turnId = this.armedTurnId;
    this.armedTurnId = null;
    this.consecutive = 0;
    return turnId;
  }

  reset() {
    this.armedTurnId = null;
    this.consecutive = 0;
  }
}

export function rmsLevel(samples: Float32Array) {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (const sample of samples) {
    if (!Number.isFinite(sample)) throw new Error("Audio samples must be finite.");
    const bounded = Math.max(-1, Math.min(1, sample));
    sum += bounded * bounded;
  }
  return Math.sqrt(sum / samples.length);
}
