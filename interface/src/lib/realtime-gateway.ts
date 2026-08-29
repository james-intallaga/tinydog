import type { RealtimeAdmission } from "./realtime-webrtc";

export type RealtimeGatewayStatus = {
  transportReady: boolean;
  checkpointLoaded: boolean;
  modelConnected: boolean;
  voiceTrained: boolean;
  visionTrained: boolean;
  conversationReady: boolean;
  randomInit: boolean;
  untrainedAudioEnabled: boolean;
  untrainedParameters: number;
  activeConnections: number;
};

export const unavailableRealtimeGatewayStatus: RealtimeGatewayStatus = {
  transportReady: false,
  checkpointLoaded: false,
  modelConnected: false,
  voiceTrained: false,
  visionTrained: false,
  conversationReady: false,
  randomInit: false,
  untrainedAudioEnabled: false,
  untrainedParameters: 0,
  activeConnections: 0,
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function parseRealtimeGatewayStatus(value: unknown): RealtimeGatewayStatus {
  if (!isRecord(value)) return unavailableRealtimeGatewayStatus;
  const transportReady = value.transportReady === true && value.webrtcReady === true;
  const checkpointLoaded = value.checkpointLoaded === true;
  const modelConnected = value.modelConnected === true;
  const voiceTrained = value.voiceTrained === true;
  const visionTrained = value.visionTrained === true;
  const advertisedConversationReady = value.conversationReady === true;
  const randomInit = value.randomInit === true;
  const untrainedAudioEnabled = value.untrainedAudioEnabled === true;
  const untrainedParameters = Number.isSafeInteger(value.untrainedParameters) &&
      (value.untrainedParameters as number) > 0
    ? (value.untrainedParameters as number)
    : 0;
  const activeConnections = Number.isSafeInteger(value.activeConnections) &&
      (value.activeConnections as number) >= 0
    ? (value.activeConnections as number)
    : 0;

  return {
    transportReady,
    checkpointLoaded,
    modelConnected,
    voiceTrained,
    visionTrained,
    randomInit,
    untrainedAudioEnabled,
    untrainedParameters,
    // Never elevate a single optimistic flag above the underlying facts.
    conversationReady:
      advertisedConversationReady &&
      transportReady &&
      checkpointLoaded &&
      modelConnected &&
      voiceTrained &&
      visionTrained,
    activeConnections,
  };
}

export async function fetchRealtimeGatewayStatus(signal?: AbortSignal) {
  const response = await fetch("/api/realtime/status", {
    cache: "no-store",
    signal,
  });
  const value: unknown = await response.json();
  if (!response.ok) throw new Error("The realtime transport is offline.");
  return parseRealtimeGatewayStatus(value);
}

export async function requestRealtimeAdmission(
  sessionId: string,
  signal?: AbortSignal,
): Promise<RealtimeAdmission> {
  const response = await fetch("/api/realtime/admission", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId }),
    cache: "no-store",
    signal,
  });
  const value: unknown = await response.json();
  if (!response.ok) {
    const message = isRecord(value) && typeof value.error === "string"
      ? value.error
      : "Realtime admission is unavailable.";
    throw new Error(message);
  }
  return value as RealtimeAdmission;
}
