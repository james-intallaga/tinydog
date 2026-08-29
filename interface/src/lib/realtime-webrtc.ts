/**
 * Classon-inspired browser transport for the MLM-owned local realtime gateway.
 *
 * HTTP media ingress remains an explicit fallback. Gateway admission and the
 * UI report model/voice readiness separately from transport connectivity.
 */

export const REALTIME_PROTOCOL_VERSION = 1 as const;

const INITIAL_RECONNECT_DELAY_MS = 250;
const MAX_RECONNECT_DELAY_MS = 5_000;
const DEFAULT_HEARTBEAT_MS = 5_000;
const MAX_SIGNAL_BYTES = 64 * 1024;
const MAX_SIGNAL_BUFFERED_BYTES = 256 * 1024;
const MAX_OUTBOX_MESSAGES = 64;
const MAX_QUEUED_OPERATIONS = 64;
const OPERATION_TIMEOUT_MS = 10_000;
const ADMISSION_TIMEOUT_MS = 5_000;
const SOCKET_OPEN_TIMEOUT_MS = 5_000;
const JOIN_TIMEOUT_MS = 5_000;
const NEGOTIATION_ANSWER_TIMEOUT_MS = 8_000;
const PEER_CONNECTION_TIMEOUT_MS = 15_000;
const ICE_DISCONNECT_GRACE_MS = 2_000;
const ICE_RESTART_TIMEOUT_MS = 8_000;
const MAX_ICE_RESTARTS = 1;
const MAX_REMOTE_CANDIDATES = 64;
const MAX_LOCAL_ICE_GENERATIONS = 8;
const FRAME_CHANNEL_HIGH_WATER_BYTES = 256 * 1024;
const FRAME_CHANNEL_LOW_WATER_BYTES = 64 * 1024;
const MAX_SELECTED_FRAME_BYTES = 512 * 1024;
const SAFE_ID = /^[A-Za-z0-9._:-]{1,128}$/;

export type RealtimeVoiceState =
  | "idle"
  | "admitting"
  | "signalling"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "failed"
  | "closed";

export type RealtimeAdmission = {
  protocolVersion: typeof REALTIME_PROTOCOL_VERSION;
  sessionId: string;
  connectionId: string;
  signallingUrl: string;
  accessToken: string;
  expiresAtUnixMs: number;
  heartbeatIntervalMs?: number;
  iceServers: RTCIceServer[];
  capabilities?: {
    duplexAudio?: boolean;
    /** MLM extension; Classon itself does not use an RTC data channel. */
    selectedFrameDataChannel?: boolean;
  };
};

export type RealtimeVoiceStatus = {
  state: RealtimeVoiceState;
  connectionId: string | null;
  playbackGeneration: number;
  detail?: string;
};

export type PendingNegotiation = {
  connectionId: string;
  requestId: string;
  revision: number;
};

export type SelectedFramePacket = {
  frameId: string;
  sequence: number;
  timestampMs: number;
  differencePercent: number;
  keyframe: boolean;
  jpeg: ArrayBuffer;
};

export type RealtimeVoiceOptions = {
  admit: () => Promise<RealtimeAdmission>;
  onStatus?: (status: RealtimeVoiceStatus) => void;
  onRemoteAudio?: (stream: MediaStream | null) => void;
  onFrameAccepted?: (frameId: string, timestampMs: number, sequence: number) => void;
  onSelectedFrameFallback?: (frame: SelectedFramePacket, reason: string) => void;
  onSelectedFrameChannelState?: (available: boolean) => void;
  onError?: (message: string) => void;
  webSocketFactory?: (url: string) => WebSocket;
  peerFactory?: (configuration: RTCConfiguration) => RTCPeerConnection;
  now?: () => number;
  random?: () => number;
};

type ClientSignal =
  | {
      type: "join";
      protocolVersion: typeof REALTIME_PROTOCOL_VERSION;
      accessToken: string;
      sessionId: string;
      connectionId: string;
    }
  | { type: "ping"; clientTimeMs: number; connectionId: string }
  | {
      type: "offer";
      connectionId: string;
      requestId: string;
      revision: number;
      sdp: string;
      iceRestart: boolean;
    }
  | {
      type: "answer";
      connectionId: string;
      requestId: string;
      revision: number;
      sdp: string;
    }
  | {
      type: "ice";
      connectionId: string;
      revision: number;
      candidate: RTCIceCandidateInit;
    }
  | {
      type: "interrupt";
      connectionId: string;
      playbackGeneration: number;
    }
  | {
      type: "playbackSilent";
      connectionId: string;
      playbackGeneration: number;
    }
  | { type: "turnEnded"; connectionId: string; turnId: string }
  | { type: "firstResponseAudio"; connectionId: string; turnId: string }
  | { type: "close"; connectionId: string };

type ServerSignal =
  | { type: "joined"; connectionId: string; heartbeatIntervalMs?: number }
  | { type: "pong"; connectionId: string }
  | {
      type: "answer";
      connectionId: string;
      requestId: string;
      revision: number;
      sdp: string;
    }
  | {
      type: "offer";
      connectionId: string;
      requestId: string;
      revision: number;
      sdp: string;
      iceRestart?: boolean;
    }
  | {
      type: "ice";
      connectionId: string;
      revision: number;
      candidate: RTCIceCandidateInit;
    }
  | { type: "restart"; connectionId: string }
  | {
      type: "frameAccepted";
      connectionId: string;
      frameId: string;
      sequence: number;
    }
  | { type: "error"; code: string; message: string; retryable?: boolean };

type EncodedSelectedFrame = {
  frameId: string;
  sequence: number;
  timestampMs: number;
  payload: ArrayBuffer;
  source: SelectedFramePacket;
};

type PendingLocalCandidate = {
  revision: number;
  candidate: RTCIceCandidateInit;
};

type QueuedControl = {
  encoded: string;
  type: ClientSignal["type"];
  connectionId?: string;
  revision?: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function requiredString(record: Record<string, unknown>, key: string) {
  const value = record[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Realtime signal is missing ${key}.`);
  }
  return value;
}

function requiredInteger(record: Record<string, unknown>, key: string) {
  const value = record[key];
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`Realtime signal has an invalid ${key}.`);
  }
  return value as number;
}

function candidateFrom(value: unknown): RTCIceCandidateInit {
  if (!isRecord(value) || typeof value.candidate !== "string") {
    throw new Error("Realtime signal has an invalid ICE candidate.");
  }
  return {
    candidate: value.candidate,
    ...(typeof value.sdpMid === "string" ? { sdpMid: value.sdpMid } : {}),
    ...(Number.isInteger(value.sdpMLineIndex)
      ? { sdpMLineIndex: value.sdpMLineIndex as number }
      : {}),
    ...(typeof value.usernameFragment === "string"
      ? { usernameFragment: value.usernameFragment }
      : {}),
  };
}

export function iceUsernameFragment(candidate: RTCIceCandidateInit) {
  const direct = candidate.usernameFragment?.trim();
  if (direct) return direct;
  const match = /(?:^|\s)ufrag\s+([^\s]+)/i.exec(candidate.candidate ?? "");
  return match?.[1] ?? null;
}

export function iceUsernameFragmentsFromSdp(sdp: string) {
  const fragments = new Set<string>();
  for (const line of sdp.split(/\r?\n/)) {
    if (!line.startsWith("a=ice-ufrag:")) continue;
    const fragment = line.slice("a=ice-ufrag:".length).trim();
    if (fragment) fragments.add(fragment);
  }
  return [...fragments];
}

export function parseServerSignal(raw: string): ServerSignal {
  if (new TextEncoder().encode(raw).byteLength > MAX_SIGNAL_BYTES) {
    throw new Error("Realtime signal is too large.");
  }
  const value: unknown = JSON.parse(raw);
  if (!isRecord(value)) throw new Error("Realtime signal must be an object.");
  const type = requiredString(value, "type");
  switch (type) {
    case "joined":
      return {
        type,
        connectionId: requiredString(value, "connectionId"),
        ...(Number.isSafeInteger(value.heartbeatIntervalMs)
          ? { heartbeatIntervalMs: value.heartbeatIntervalMs as number }
          : {}),
      };
    case "pong":
    case "restart":
      return { type, connectionId: requiredString(value, "connectionId") };
    case "answer":
      return {
        type,
        connectionId: requiredString(value, "connectionId"),
        requestId: requiredString(value, "requestId"),
        revision: requiredInteger(value, "revision"),
        sdp: requiredString(value, "sdp"),
      };
    case "offer":
      return {
        type,
        connectionId: requiredString(value, "connectionId"),
        requestId: requiredString(value, "requestId"),
        revision: requiredInteger(value, "revision"),
        sdp: requiredString(value, "sdp"),
        ...(typeof value.iceRestart === "boolean"
          ? { iceRestart: value.iceRestart }
          : {}),
      };
    case "ice":
      return {
        type,
        connectionId: requiredString(value, "connectionId"),
        revision: requiredInteger(value, "revision"),
        candidate: candidateFrom(value.candidate),
      };
    case "frameAccepted":
      if (!SAFE_ID.test(requiredString(value, "frameId"))) {
        throw new Error("Realtime signal has an invalid frame ID.");
      }
      return {
        type,
        connectionId: requiredString(value, "connectionId"),
        frameId: requiredString(value, "frameId"),
        sequence: requiredInteger(value, "sequence"),
      };
    case "error":
      return {
        type,
        code: requiredString(value, "code"),
        message: requiredString(value, "message"),
        ...(typeof value.retryable === "boolean" ? { retryable: value.retryable } : {}),
      };
    default:
      throw new Error("Realtime signal type is not supported.");
  }
}

function isLoopbackHost(hostname: string) {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]" || hostname === "::1";
}

export function validateIceServers(iceServers: RTCIceServer[]) {
  if (!Array.isArray(iceServers) || iceServers.length > 8) {
    throw new Error("Realtime admission has invalid ICE configuration.");
  }
  let totalUrls = 0;
  for (const server of iceServers) {
    if (!isRecord(server)) {
      throw new Error("Realtime admission has an invalid ICE server.");
    }
    const rawUrls = server.urls;
    const urls = typeof rawUrls === "string"
      ? [rawUrls]
      : Array.isArray(rawUrls) && rawUrls.every((value) => typeof value === "string")
        ? rawUrls
        : [];
    if (urls.length === 0 || urls.length > 8) {
      throw new Error("Realtime admission has invalid ICE server URLs.");
    }
    totalUrls += urls.length;
    if (totalUrls > 16) {
      throw new Error("Realtime admission has too many ICE server URLs.");
    }
    for (const value of urls) {
      if (
        value.length > 2_048 ||
        !/^(?:stun|stuns|turn|turns):[^\s#]+$/i.test(value) ||
        value.includes("@") ||
        /[?&](?:token|auth|secret|credential|password|username|api.?key)=/i.test(value)
      ) {
        throw new Error("Realtime ICE URLs must use bounded STUN or TURN addresses without credentials.");
      }
    }
    if (
      server.username !== undefined &&
      (typeof server.username !== "string" || server.username.length > 1_024)
    ) {
      throw new Error("Realtime ICE username is invalid.");
    }
    if (
      server.credential !== undefined &&
      (typeof server.credential !== "string" || server.credential.length > 8_192)
    ) {
      throw new Error("Realtime ICE credential is invalid.");
    }
    if (server.credentialType !== undefined && server.credentialType !== "password") {
      throw new Error("Realtime ICE credential type is not supported.");
    }
  }
  return iceServers;
}

export function validateRealtimeAdmission(
  admission: RealtimeAdmission,
  nowMs = Date.now(),
): RealtimeAdmission {
  if (admission.protocolVersion !== REALTIME_PROTOCOL_VERSION) {
    throw new Error("Realtime protocol version is not supported.");
  }
  if (!SAFE_ID.test(admission.sessionId) || !SAFE_ID.test(admission.connectionId)) {
    throw new Error("Realtime admission contains an invalid session identity.");
  }
  if (!admission.accessToken || admission.accessToken.length > 8_192) {
    throw new Error("Realtime admission token is missing or too large.");
  }
  if (!Number.isSafeInteger(admission.expiresAtUnixMs) || admission.expiresAtUnixMs <= nowMs) {
    throw new Error("Realtime admission has expired.");
  }
  const url = new URL(admission.signallingUrl);
  if (url.username || url.password) {
    throw new Error("Realtime credentials must not be placed in the signalling URL.");
  }
  if (url.hash) {
    throw new Error("Realtime signalling URLs must not contain fragments.");
  }
  for (const key of url.searchParams.keys()) {
    if (/token|auth|secret|credential|api.?key/i.test(key)) {
      throw new Error("Realtime credentials must not be placed in the signalling URL.");
    }
  }
  if (url.protocol !== "wss:" && !(url.protocol === "ws:" && isLoopbackHost(url.hostname))) {
    throw new Error("Realtime signalling must use WSS, except on loopback.");
  }
  const heartbeat = admission.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_MS;
  if (!Number.isSafeInteger(heartbeat) || heartbeat < 1_000 || heartbeat > 60_000) {
    throw new Error("Realtime heartbeat interval is outside the supported range.");
  }
  validateIceServers(admission.iceServers);
  return { ...admission, heartbeatIntervalMs: heartbeat };
}

export function reconnectDelayMs(attempt: number, randomValue = Math.random()) {
  const safeAttempt = Math.max(0, Math.min(30, Math.floor(attempt)));
  const safeRandom = Math.max(0, Math.min(1, randomValue));
  const exponential = Math.min(
    MAX_RECONNECT_DELAY_MS,
    INITIAL_RECONNECT_DELAY_MS * 2 ** safeAttempt,
  );
  return Math.round(exponential * (0.8 + safeRandom * 0.4));
}

export function heartbeatMissed(nowMs: number, lastPongMs: number, intervalMs: number) {
  return nowMs - lastPongMs > intervalMs * 2.5;
}

export function isCurrentNegotiationAnswer(
  pending: PendingNegotiation | null,
  answer: Pick<PendingNegotiation, "connectionId" | "requestId" | "revision">,
) {
  return pending !== null &&
    pending.connectionId === answer.connectionId &&
    pending.requestId === answer.requestId &&
    pending.revision === answer.revision;
}

export class LatestOnlyValue<T> {
  private value: T | null = null;

  offer(value: T) {
    const replaced = this.value !== null;
    this.value = value;
    return replaced;
  }

  take() {
    const value = this.value;
    this.value = null;
    return value;
  }

  clear() {
    this.value = null;
  }

  get pending() {
    return this.value !== null;
  }

  get current() {
    return this.value;
  }
}

function makeRequestId(prefix: string) {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function encodeFramePacket(frame: SelectedFramePacket) {
  if (!SAFE_ID.test(frame.frameId)) throw new Error("Selected frame has an invalid ID.");
  if (frame.jpeg.byteLength < 4 || frame.jpeg.byteLength > MAX_SELECTED_FRAME_BYTES) {
    throw new Error("Selected frame is outside the supported size range.");
  }
  const jpeg = new Uint8Array(frame.jpeg);
  if (
    jpeg[0] !== 0xff ||
    jpeg[1] !== 0xd8 ||
    jpeg[jpeg.length - 2] !== 0xff ||
    jpeg[jpeg.length - 1] !== 0xd9
  ) {
    throw new Error("Selected frame is not a complete JPEG.");
  }
  if (!Number.isFinite(frame.differencePercent) || frame.differencePercent < 0 || frame.differencePercent > 100) {
    throw new Error("Selected frame has an invalid difference percentage.");
  }
  const header = new TextEncoder().encode(JSON.stringify({
    version: 1,
    frameId: frame.frameId,
    sequence: frame.sequence,
    timestampMs: Math.round(frame.timestampMs),
    differencePercent: frame.differencePercent,
    keyframe: frame.keyframe,
    contentType: "image/jpeg",
  }));
  const packet = new Uint8Array(4 + header.byteLength + jpeg.byteLength);
  new DataView(packet.buffer).setUint32(0, header.byteLength, false);
  packet.set(header, 4);
  packet.set(jpeg, 4 + header.byteLength);
  return packet.buffer;
}

/**
 * Browser half of a single-peer realtime voice conversation.
 *
 * The caller owns activation. Once supplied, microphone tracks are owned and
 * stopped by this transport.
 */
export class RealtimeVoiceTransport {
  private admission: RealtimeAdmission | null = null;
  private socket: WebSocket | null = null;
  private peer: RTCPeerConnection | null = null;
  private audioTransceiver: RTCRtpTransceiver | null = null;
  private duplexAudioEnabled = false;
  private localAudioTrack: MediaStreamTrack | null = null;
  private localStream: MediaStream | null = null;
  private frameChannel: RTCDataChannel | null = null;
  private pendingFrame = new LatestOnlyValue<EncodedSelectedFrame>();
  private fallbackFrame = new LatestOnlyValue<SelectedFramePacket>();
  private sentFrameIds = new Map<string, EncodedSelectedFrame>();
  private lastOfferedFrameSequence = -1;
  private lastAcceptedFrameSequence = -1;
  private pendingNegotiation: PendingNegotiation | null = null;
  private pendingLocalCandidates: PendingLocalCandidate[] = [];
  private localIceUfragRevisions = new Map<string, number>();
  private pendingRemoteCandidates = new Map<number, RTCIceCandidateInit[]>();
  private descriptionSentRevision = -1;
  private remoteDescriptionRevision = -1;
  private operation: Promise<void> = Promise.resolve();
  private queuedOperations = 0;
  private outbox: QueuedControl[] = [];
  private drainTimer: number | null = null;
  private joinRetryTimer: number | null = null;
  private joinTimeoutTimer: number | null = null;
  private reconnectTimer: number | null = null;
  private admissionTimeoutTimer: number | null = null;
  private socketOpenTimeoutTimer: number | null = null;
  private heartbeatTimer: number | null = null;
  private disconnectTimer: number | null = null;
  private restartTimer: number | null = null;
  private negotiationTimeoutTimer: number | null = null;
  private connectionTimeoutTimer: number | null = null;
  private reconnectAttempt = 0;
  private negotiationRevision = 0;
  private iceRestartAttempts = 0;
  private lastPongAt = 0;
  private playbackGeneration = 0;
  private closed = true;
  private lifecycleEpoch = 0;
  private connectionGeneration = 0;
  private joinedGeneration = -1;
  private openingEpoch: number | null = null;
  private state: RealtimeVoiceState = "idle";
  private readonly now: () => number;
  private readonly random: () => number;
  private readonly webSocketFactory: (url: string) => WebSocket;
  private readonly peerFactory: (configuration: RTCConfiguration) => RTCPeerConnection;

  constructor(private readonly options: RealtimeVoiceOptions) {
    this.now = options.now ?? (() => Date.now());
    this.random = options.random ?? (() => Math.random());
    this.webSocketFactory = options.webSocketFactory ?? ((url) => new WebSocket(url));
    this.peerFactory = options.peerFactory ?? ((configuration) => new RTCPeerConnection(configuration));
  }

  connect(audioTrack?: MediaStreamTrack) {
    if (audioTrack && (audioTrack.kind !== "audio" || audioTrack.readyState === "ended")) {
      throw new Error("Realtime voice transport requires a live audio track.");
    }
    if (!this.closed) {
      if (audioTrack && audioTrack !== this.localAudioTrack) {
        void this.setMicrophoneTrack(audioTrack).catch((reason: unknown) => {
          this.reportError(reason, "The microphone track could not be replaced.");
        });
      }
      return;
    }
    this.lifecycleEpoch += 1;
    this.connectionGeneration += 1;
    this.openingEpoch = null;
    if (audioTrack) {
      this.localAudioTrack?.stop();
      this.installLocalAudioTrack(audioTrack);
    }
    this.closed = false;
    this.beginConnect(this.lifecycleEpoch);
  }

  setMicrophoneTrack(track: MediaStreamTrack | null) {
    if (track && (track.kind !== "audio" || track.readyState === "ended")) {
      track.stop();
      return Promise.reject(new Error("Realtime voice transport requires a live audio track."));
    }
    const lifecycleEpoch = this.lifecycleEpoch;
    return this.queue(async () => {
      if (this.closed || lifecycleEpoch !== this.lifecycleEpoch) {
        track?.stop();
        return;
      }
      const previous = this.localAudioTrack;
      if (previous === track) return;
      const connectionGeneration = this.connectionGeneration;
      const transceiver = this.duplexAudioEnabled ? this.audioTransceiver : null;
      this.installLocalAudioTrack(track);
      if (transceiver) {
        try {
          await transceiver.sender.replaceTrack(track);
        } catch (reason) {
          if (
            !this.closed &&
            lifecycleEpoch === this.lifecycleEpoch &&
            connectionGeneration === this.connectionGeneration
          ) {
            this.installLocalAudioTrack(previous);
            track?.stop();
            throw reason;
          }
          previous?.stop();
          return;
        }
      }
      if (this.closed || lifecycleEpoch !== this.lifecycleEpoch) {
        track?.stop();
        return;
      }
      previous?.stop();
    });
  }

  interrupt() {
    this.playbackGeneration += 1;
    const connectionId = this.admission?.connectionId;
    if (connectionId) {
      this.sendControl({
        type: "interrupt",
        connectionId,
        playbackGeneration: this.playbackGeneration,
      });
    }
    this.emitStatus(this.state);
    return this.playbackGeneration;
  }

  /**
   * Report that this browser has made the interrupted generation inaudible.
   * Telemetry is deliberately connection-ephemeral: it is never queued across
   * a signalling reconnect, where its correlation would be stale.
   */
  acknowledgePlaybackSilent(playbackGeneration = this.playbackGeneration) {
    if (
      !Number.isSafeInteger(playbackGeneration) ||
      playbackGeneration < 1 ||
      playbackGeneration > this.playbackGeneration
    ) {
      throw new Error("Playback-silence generation is invalid.");
    }
    return this.sendTelemetryNow({
      type: "playbackSilent",
      connectionId: this.admission?.connectionId ?? "",
      playbackGeneration,
    });
  }

  markTurnEnded(turnId: string) {
    if (!SAFE_ID.test(turnId)) throw new Error("Realtime turn ID is invalid.");
    return this.sendTelemetryNow({
      type: "turnEnded",
      connectionId: this.admission?.connectionId ?? "",
      turnId,
    });
  }

  acknowledgeFirstResponseAudio(turnId: string) {
    if (!SAFE_ID.test(turnId)) throw new Error("Realtime turn ID is invalid.");
    return this.sendTelemetryNow({
      type: "firstResponseAudio",
      connectionId: this.admission?.connectionId ?? "",
      turnId,
    });
  }

  /**
   * Offers a difference-selected JPEG to the optional MLM data channel.
   * At most one unsent frame is retained. Baselines must advance only after
   * the server's frameAccepted signal reaches onFrameAccepted.
   */
  offerSelectedFrame(frame: SelectedFramePacket) {
    if (!Number.isFinite(frame.timestampMs) || frame.timestampMs < 0) {
      throw new Error("Selected frame has an invalid timestamp.");
    }
    if (typeof frame.keyframe !== "boolean") {
      throw new Error("Selected frame has an invalid keyframe flag.");
    }
    if (!Number.isSafeInteger(frame.sequence) || frame.sequence <= this.lastOfferedFrameSequence) {
      throw new Error("Selected frame sequence must increase monotonically.");
    }
    if (this.sentFrameIds.has(frame.frameId) || this.pendingFrame.current?.frameId === frame.frameId) {
      throw new Error("Selected frame IDs must be unique until acknowledged.");
    }
    const replacedFrameId = this.pendingFrame.current?.frameId ?? null;
    const replaced = this.pendingFrame.offer({
      frameId: frame.frameId,
      sequence: frame.sequence,
      timestampMs: frame.timestampMs,
      payload: encodeFramePacket(frame),
      source: { ...frame },
    });
    this.lastOfferedFrameSequence = frame.sequence;
    this.flushFrameChannel();
    return { frameId: frame.frameId, replaced, replacedFrameId };
  }

  takeSelectedFrameFallback() {
    return this.fallbackFrame.take();
  }

  disconnect() {
    this.closed = true;
    this.lifecycleEpoch += 1;
    this.connectionGeneration += 1;
    this.joinedGeneration = -1;
    this.openingEpoch = null;
    this.clearReconnectTimer();
    this.clearAdmissionDeadline();
    this.clearSocketOpenDeadline();
    this.stopHeartbeat();
    this.clearRecoveryTimers();
    this.clearJoinTimers();
    this.stopDrainTimer();
    const connectionId = this.admission?.connectionId;
    if (connectionId && this.socket?.readyState === WebSocket.OPEN) {
      this.sendNow({ type: "close", connectionId });
    }
    this.socket?.close();
    this.socket = null;
    this.closePeer();
    const track = this.localAudioTrack;
    this.installLocalAudioTrack(null);
    track?.stop();
    this.admission = null;
    this.outbox = [];
    this.pendingFrame.clear();
    this.fallbackFrame.clear();
    this.sentFrameIds.clear();
    this.emitStatus("closed");
  }

  get status(): RealtimeVoiceStatus {
    return {
      state: this.state,
      connectionId: this.admission?.connectionId ?? null,
      playbackGeneration: this.playbackGeneration,
    };
  }

  private beginConnect(lifecycleEpoch = this.lifecycleEpoch) {
    if (
      this.closed ||
      lifecycleEpoch !== this.lifecycleEpoch ||
      this.openingEpoch !== null ||
      this.socket
    ) {
      return;
    }
    this.emitStatus(this.reconnectAttempt === 0 ? "admitting" : "reconnecting");
    this.openingEpoch = lifecycleEpoch;
    const admissionRequest = Promise.resolve().then(() => this.options.admit());
    let admissionTimer: number | null = null;
    const admissionDeadline = new Promise<never>((_resolve, reject) => {
      admissionTimer = window.setTimeout(
        () => {
          if (this.admissionTimeoutTimer === admissionTimer) {
            this.admissionTimeoutTimer = null;
          }
          reject(new Error("Realtime admission timed out."));
        },
        ADMISSION_TIMEOUT_MS,
      );
      this.admissionTimeoutTimer = admissionTimer;
    });
    void Promise.race([admissionRequest, admissionDeadline])
      .then((value) => validateRealtimeAdmission(value, this.now()))
      .then((admission) => {
        if (this.closed || lifecycleEpoch !== this.lifecycleEpoch) return;
        if (this.admission?.connectionId !== admission.connectionId) {
          this.negotiationRevision = 0;
          this.pendingNegotiation = null;
          this.pendingLocalCandidates = [];
          this.localIceUfragRevisions.clear();
          this.pendingRemoteCandidates.clear();
          this.descriptionSentRevision = -1;
          this.remoteDescriptionRevision = -1;
        }
        this.admission = admission;
        const generation = ++this.connectionGeneration;
        this.openSocket(admission, lifecycleEpoch, generation);
      })
      .catch((reason: unknown) => {
        if (this.closed || lifecycleEpoch !== this.lifecycleEpoch) return;
        this.reportError(reason, "Realtime admission failed.");
        this.scheduleReconnect(lifecycleEpoch);
      })
      .finally(() => {
        if (admissionTimer !== null) {
          window.clearTimeout(admissionTimer);
          if (this.admissionTimeoutTimer === admissionTimer) {
            this.admissionTimeoutTimer = null;
          }
        }
        if (this.openingEpoch === lifecycleEpoch) this.openingEpoch = null;
      });
  }

  private openSocket(
    admission: RealtimeAdmission,
    lifecycleEpoch: number,
    generation: number,
  ) {
    const socket = this.webSocketFactory(admission.signallingUrl);
    this.socket = socket;
    this.joinedGeneration = -1;
    this.emitStatus("signalling");
    const openTimer = window.setTimeout(() => {
      if (this.socketOpenTimeoutTimer === openTimer) this.socketOpenTimeoutTimer = null;
      if (this.isCurrentConnection(lifecycleEpoch, generation) && this.socket === socket) {
        this.notifyError("Realtime signalling connection timed out.");
        this.recoverConnection(lifecycleEpoch, generation);
      }
    }, SOCKET_OPEN_TIMEOUT_MS);
    this.socketOpenTimeoutTimer = openTimer;
    socket.addEventListener("open", () => {
      if (!this.isCurrentConnection(lifecycleEpoch, generation) || this.socket !== socket) {
        socket.close();
        return;
      }
      this.clearSocketOpenDeadline();
      this.beginJoin(socket, admission, lifecycleEpoch, generation, {
        type: "join",
        protocolVersion: REALTIME_PROTOCOL_VERSION,
        accessToken: admission.accessToken,
        sessionId: admission.sessionId,
        connectionId: admission.connectionId,
      });
    });
    socket.addEventListener("message", (event) => {
      if (
        !this.isCurrentConnection(lifecycleEpoch, generation) ||
        this.socket !== socket ||
        typeof event.data !== "string"
      ) {
        return;
      }
      let signal: ServerSignal;
      try {
        signal = parseServerSignal(event.data);
      } catch (reason) {
        this.reportError(reason, "Realtime signalling message was rejected.");
        socket.close(1002, "invalid signal");
        return;
      }
      this.handleSignal(signal, lifecycleEpoch, generation);
    });
    socket.addEventListener("close", () => {
      if (!this.isCurrentConnection(lifecycleEpoch, generation) || this.socket !== socket) return;
      this.recoverConnection(lifecycleEpoch, generation);
    });
    socket.addEventListener("error", () => {
      if (this.isCurrentConnection(lifecycleEpoch, generation) && this.socket === socket) {
        socket.close();
      }
    });
  }

  private beginJoin(
    socket: WebSocket,
    admission: RealtimeAdmission,
    lifecycleEpoch: number,
    generation: number,
    signal: Extract<ClientSignal, { type: "join" }>,
  ) {
    this.clearJoinTimers();
    const encoded = JSON.stringify(signal);
    if (new TextEncoder().encode(encoded).byteLength > MAX_SIGNAL_BYTES) {
      this.terminalFailure("Realtime join message is too large.");
      return;
    }
    const trySend = () => {
      this.joinRetryTimer = null;
      if (
        !this.isCurrentConnection(lifecycleEpoch, generation, admission.connectionId) ||
        this.socket !== socket ||
        socket.readyState !== WebSocket.OPEN
      ) {
        return;
      }
      if (socket.bufferedAmount > MAX_SIGNAL_BUFFERED_BYTES) {
        this.joinRetryTimer = window.setTimeout(trySend, 25);
        return;
      }
      try {
        socket.send(encoded);
      } catch {
        socket.close();
      }
    };
    this.joinTimeoutTimer = window.setTimeout(() => {
      this.joinTimeoutTimer = null;
      if (
        this.isCurrentConnection(lifecycleEpoch, generation, admission.connectionId) &&
        this.joinedGeneration !== generation
      ) {
        this.recoverConnection(lifecycleEpoch, generation);
      }
    }, JOIN_TIMEOUT_MS);
    trySend();
  }

  private handleSignal(signal: ServerSignal, lifecycleEpoch: number, generation: number) {
    if (!this.isCurrentConnection(lifecycleEpoch, generation)) return;
    if (signal.type === "error") {
      this.notifyError(signal.message);
      if (signal.retryable === false) {
        this.terminalFailure(signal.message);
      } else {
        this.recoverConnection(lifecycleEpoch, generation);
      }
      return;
    }
    const admission = this.admission;
    if (!admission || signal.connectionId !== admission.connectionId) return;
    switch (signal.type) {
      case "joined": {
        this.joinedGeneration = generation;
        this.clearJoinTimers();
        this.reconnectAttempt = 0;
        this.lastPongAt = this.now();
        const advertisedHeartbeat = signal.heartbeatIntervalMs;
        const heartbeat =
          Number.isSafeInteger(advertisedHeartbeat) &&
          (advertisedHeartbeat as number) >= 1_000 &&
          (advertisedHeartbeat as number) <= 60_000
            ? (advertisedHeartbeat as number)
            : admission.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_MS;
        this.startHeartbeat(
          heartbeat,
          lifecycleEpoch,
          generation,
          admission.connectionId,
        );
        this.scheduleDrain();
        this.enqueueConnectionOperation(
          lifecycleEpoch,
          generation,
          () => this.rebuildPeer(false, lifecycleEpoch, generation),
        );
        break;
      }
      case "pong":
        this.lastPongAt = this.now();
        break;
      case "answer":
        this.enqueueConnectionOperation(
          lifecycleEpoch,
          generation,
          () => this.applyAnswer(signal, lifecycleEpoch, generation),
        );
        break;
      case "offer":
        this.enqueueConnectionOperation(
          lifecycleEpoch,
          generation,
          () => this.applyServerOffer(signal, lifecycleEpoch, generation),
        );
        break;
      case "ice":
        this.enqueueConnectionOperation(
          lifecycleEpoch,
          generation,
          () => this.applyRemoteCandidate(signal, lifecycleEpoch, generation),
        );
        break;
      case "restart": {
        const peer = this.peer;
        this.enqueueConnectionOperation(
          lifecycleEpoch,
          generation,
          () => this.restartIce(lifecycleEpoch, generation, peer, true),
        );
        break;
      }
      case "frameAccepted": {
        const sent = this.sentFrameIds.get(signal.frameId);
        if (sent?.sequence === signal.sequence) {
          this.sentFrameIds.delete(signal.frameId);
          if (signal.sequence > this.lastAcceptedFrameSequence) {
            this.lastAcceptedFrameSequence = signal.sequence;
            this.notifyFrameAccepted(signal.frameId, sent.timestampMs, signal.sequence);
          }
        }
        break;
      }
    }
  }

  private async rebuildPeer(
    iceRestart: boolean,
    lifecycleEpoch: number,
    generation: number,
  ) {
    if (!this.isCurrentConnection(lifecycleEpoch, generation) || !this.admission) return;
    this.closePeer();
    this.iceRestartAttempts = 0;
    const admission = this.admission;
    const connectionId = admission.connectionId;
    const peer = this.peerFactory({
      bundlePolicy: "max-bundle",
      iceServers: admission.iceServers,
    });
    this.peer = peer;
    this.pendingLocalCandidates = [];
    this.pendingRemoteCandidates.clear();
    this.descriptionSentRevision = -1;
    this.remoteDescriptionRevision = -1;
    peer.addEventListener("icecandidate", (event) => {
      if (
        !this.isCurrentConnection(lifecycleEpoch, generation, connectionId) ||
        this.peer !== peer ||
        !event.candidate
      ) {
        return;
      }
      const candidate = event.candidate.toJSON();
      const revision = this.localCandidateRevision(candidate);
      if (revision === null) {
        this.notifyError("A local ICE candidate without a matching generation was ignored.");
        return;
      }
      const pending: PendingLocalCandidate = { revision, candidate };
      if (this.descriptionSentRevision === pending.revision) {
        this.sendLocalCandidate(pending, connectionId);
      } else if (this.pendingLocalCandidates.length < MAX_REMOTE_CANDIDATES) {
        this.pendingLocalCandidates.push(pending);
      } else {
        this.fail("Too many pending local ICE candidates.");
      }
    });
    peer.addEventListener("track", (event) => {
      if (
        !this.isCurrentConnection(lifecycleEpoch, generation, connectionId) ||
        this.peer !== peer ||
        event.track.kind !== "audio"
      ) {
        return;
      }
      const stream = event.streams[0] ?? new MediaStream([event.track]);
      this.notifyRemoteAudio(stream);
    });
    peer.addEventListener("connectionstatechange", () => {
      if (
        !this.isCurrentConnection(lifecycleEpoch, generation, connectionId) ||
        this.peer !== peer
      ) {
        return;
      }
      if (peer.connectionState === "connected") {
        this.clearRecoveryTimers();
        this.clearConnectionDeadline();
        this.iceRestartAttempts = 0;
        this.emitStatus("connected");
      } else if (peer.connectionState === "disconnected") {
        this.emitStatus("reconnecting", "Media connection interrupted.");
        this.scheduleIceRestart(
          ICE_DISCONNECT_GRACE_MS,
          lifecycleEpoch,
          generation,
          peer,
        );
      } else if (peer.connectionState === "failed") {
        this.emitStatus("reconnecting", "Media connection failed; recovering.");
        this.scheduleIceRestart(0, lifecycleEpoch, generation, peer);
      }
    });

    this.duplexAudioEnabled = admission.capabilities?.duplexAudio === true;
    const outboundTrack = this.duplexAudioEnabled ? this.localAudioTrack : null;
    this.localStream = outboundTrack ? new MediaStream([outboundTrack]) : null;
    this.audioTransceiver = peer.addTransceiver(outboundTrack ?? "audio", {
      direction: this.duplexAudioEnabled ? "sendrecv" : "recvonly",
      ...(this.localStream ? { streams: [this.localStream] } : {}),
    });
    if (admission.capabilities?.selectedFrameDataChannel) {
      const channel = peer.createDataChannel("mlm-selected-frames-v1", {
        ordered: false,
        maxRetransmits: 0,
      });
      channel.binaryType = "arraybuffer";
      channel.bufferedAmountLowThreshold = FRAME_CHANNEL_LOW_WATER_BYTES;
      channel.addEventListener("open", () => {
        if (this.peer === peer) {
          this.notifySelectedFrameChannelState(true);
          this.flushFrameChannel();
        }
      });
      channel.addEventListener("bufferedamountlow", () => {
        if (this.peer === peer) this.flushFrameChannel();
      });
      channel.addEventListener("close", () => {
        this.disableSelectedFrameChannel(channel, "Selected-frame WebRTC channel closed.");
      });
      channel.addEventListener("error", () => {
        const reason = "Selected-frame WebRTC channel failed; use the HTTP fallback.";
        this.disableSelectedFrameChannel(channel, reason);
        this.notifyError(reason);
      });
      this.frameChannel = channel;
    } else {
      this.notifySelectedFrameChannelState(false);
    }
    await this.createAndSendOffer(
      peer,
      iceRestart,
      lifecycleEpoch,
      generation,
      connectionId,
    );
  }

  private async createAndSendOffer(
    peer: RTCPeerConnection,
    iceRestart: boolean,
    lifecycleEpoch: number,
    generation: number,
    connectionId: string,
  ) {
    if (
      !this.isCurrentConnection(lifecycleEpoch, generation, connectionId) ||
      this.peer !== peer
    ) {
      return;
    }
    this.negotiationRevision += 1;
    const pending = {
      connectionId,
      requestId: makeRequestId(iceRestart ? "restart" : "offer"),
      revision: this.negotiationRevision,
    };
    this.pendingNegotiation = pending;
    const offer = await peer.createOffer({ iceRestart });
    if (
      !this.isCurrentConnection(lifecycleEpoch, generation, connectionId) ||
      this.peer !== peer
    ) {
      return;
    }
    this.registerLocalIceRevision(offer.sdp ?? "", pending.revision);
    await peer.setLocalDescription(offer);
    if (
      !this.isCurrentConnection(lifecycleEpoch, generation, connectionId) ||
      this.peer !== peer
    ) {
      return;
    }
    const sdp = peer.localDescription?.sdp;
    if (!sdp) throw new Error("The browser could not create a realtime offer.");
    this.registerLocalIceRevision(sdp, pending.revision);
    this.purgeObsoleteNegotiationMessages(connectionId);
    if (!this.sendControl({
      type: "offer",
      ...pending,
      sdp,
      iceRestart,
    })) {
      throw new Error("The realtime offer could not be queued.");
    }
    this.descriptionSentRevision = pending.revision;
    this.flushLocalCandidates(pending.revision, connectionId);
    this.startNegotiationDeadline(
      pending,
      peer,
      lifecycleEpoch,
      generation,
    );
    this.startConnectionDeadline(peer, lifecycleEpoch, generation, connectionId);
    this.emitStatus("connecting");
  }

  private async applyAnswer(
    signal: Extract<ServerSignal, { type: "answer" }>,
    lifecycleEpoch: number,
    generation: number,
  ) {
    if (!this.isCurrentConnection(lifecycleEpoch, generation, signal.connectionId)) return;
    const peer = this.peer;
    if (!peer || !isCurrentNegotiationAnswer(this.pendingNegotiation, signal)) return;
    await peer.setRemoteDescription({ type: "answer", sdp: signal.sdp });
    if (
      !this.isCurrentConnection(lifecycleEpoch, generation, signal.connectionId) ||
      this.peer !== peer
    ) {
      return;
    }
    this.remoteDescriptionRevision = signal.revision;
    this.pendingNegotiation = null;
    this.clearNegotiationDeadline();
    await this.flushRemoteCandidates(peer, signal.revision, lifecycleEpoch, generation);
  }

  private async applyServerOffer(
    signal: Extract<ServerSignal, { type: "offer" }>,
    lifecycleEpoch: number,
    generation: number,
  ) {
    if (!this.isCurrentConnection(lifecycleEpoch, generation, signal.connectionId)) return;
    const peer = this.peer;
    if (!peer || signal.revision <= this.negotiationRevision) return;
    this.clearNegotiationDeadline();
    this.pendingLocalCandidates = [];
    this.descriptionSentRevision = -1;
    if (peer.signalingState !== "stable") {
      await peer.setLocalDescription({ type: "rollback" });
      if (
        !this.isCurrentConnection(lifecycleEpoch, generation, signal.connectionId) ||
        this.peer !== peer
      ) {
        return;
      }
    }
    await peer.setRemoteDescription({ type: "offer", sdp: signal.sdp });
    if (
      !this.isCurrentConnection(lifecycleEpoch, generation, signal.connectionId) ||
      this.peer !== peer
    ) {
      return;
    }
    this.remoteDescriptionRevision = signal.revision;
    const answer = await peer.createAnswer();
    if (
      !this.isCurrentConnection(lifecycleEpoch, generation, signal.connectionId) ||
      this.peer !== peer
    ) {
      return;
    }
    this.negotiationRevision = signal.revision;
    this.registerLocalIceRevision(answer.sdp ?? "", signal.revision);
    await peer.setLocalDescription(answer);
    if (
      !this.isCurrentConnection(lifecycleEpoch, generation, signal.connectionId) ||
      this.peer !== peer
    ) {
      return;
    }
    const sdp = peer.localDescription?.sdp;
    if (!sdp) throw new Error("The browser could not answer realtime negotiation.");
    this.registerLocalIceRevision(sdp, signal.revision);
    this.pendingNegotiation = null;
    this.purgeObsoleteNegotiationMessages(signal.connectionId);
    if (!this.sendControl({
      type: "answer",
      connectionId: signal.connectionId,
      requestId: signal.requestId,
      revision: signal.revision,
      sdp,
    })) {
      throw new Error("The realtime answer could not be queued.");
    }
    this.descriptionSentRevision = signal.revision;
    this.flushLocalCandidates(signal.revision, signal.connectionId);
    await this.flushRemoteCandidates(peer, signal.revision, lifecycleEpoch, generation);
  }

  private async applyRemoteCandidate(
    signal: Extract<ServerSignal, { type: "ice" }>,
    lifecycleEpoch: number,
    generation: number,
  ) {
    if (!this.isCurrentConnection(lifecycleEpoch, generation, signal.connectionId)) return;
    const peer = this.peer;
    if (!peer || signal.revision < this.negotiationRevision) return;
    if (this.remoteDescriptionRevision !== signal.revision) {
      const pendingCount = [...this.pendingRemoteCandidates.values()]
        .reduce((count, candidates) => count + candidates.length, 0);
      if (pendingCount >= MAX_REMOTE_CANDIDATES) {
        throw new Error("Too many pending realtime ICE candidates.");
      }
      const candidates = this.pendingRemoteCandidates.get(signal.revision) ?? [];
      candidates.push(signal.candidate);
      this.pendingRemoteCandidates.set(signal.revision, candidates);
      return;
    }
    await peer.addIceCandidate(signal.candidate);
  }

  private async flushRemoteCandidates(
    peer: RTCPeerConnection,
    revision: number,
    lifecycleEpoch: number,
    generation: number,
  ) {
    const candidates = this.pendingRemoteCandidates.get(revision) ?? [];
    this.pendingRemoteCandidates.delete(revision);
    for (const pendingRevision of [...this.pendingRemoteCandidates.keys()]) {
      if (pendingRevision < revision) this.pendingRemoteCandidates.delete(pendingRevision);
    }
    for (const candidate of candidates) {
      if (
        !this.isCurrentConnection(lifecycleEpoch, generation) ||
        this.peer !== peer ||
        this.remoteDescriptionRevision !== revision
      ) {
        return;
      }
      await peer.addIceCandidate(candidate);
    }
  }

  private sendLocalCandidate(pending: PendingLocalCandidate, connectionId: string) {
    if (
      this.admission?.connectionId !== connectionId ||
      pending.revision !== this.negotiationRevision
    ) {
      return;
    }
    this.sendControl({
      type: "ice",
      connectionId,
      revision: pending.revision,
      candidate: pending.candidate,
    });
  }

  private registerLocalIceRevision(sdp: string, revision: number) {
    for (const fragment of iceUsernameFragmentsFromSdp(sdp)) {
      this.localIceUfragRevisions.delete(fragment);
      this.localIceUfragRevisions.set(fragment, revision);
    }
    while (this.localIceUfragRevisions.size > MAX_LOCAL_ICE_GENERATIONS) {
      const oldest = this.localIceUfragRevisions.keys().next().value;
      if (typeof oldest !== "string") break;
      this.localIceUfragRevisions.delete(oldest);
    }
  }

  private localCandidateRevision(candidate: RTCIceCandidateInit) {
    const fragment = iceUsernameFragment(candidate);
    return fragment === null ? null : (this.localIceUfragRevisions.get(fragment) ?? null);
  }

  private flushLocalCandidates(revision: number, connectionId: string) {
    const candidates = this.pendingLocalCandidates.splice(0);
    for (const pending of candidates) {
      if (pending.revision === revision) this.sendLocalCandidate(pending, connectionId);
    }
  }

  private scheduleIceRestart(
    delayMs: number,
    lifecycleEpoch: number,
    generation: number,
    peer: RTCPeerConnection,
  ) {
    if (
      !this.isCurrentConnection(lifecycleEpoch, generation) ||
      this.peer !== peer ||
      this.disconnectTimer !== null
    ) {
      return;
    }
    this.disconnectTimer = window.setTimeout(() => {
      this.disconnectTimer = null;
      this.enqueueConnectionOperation(
        lifecycleEpoch,
        generation,
        () => this.restartIce(lifecycleEpoch, generation, peer, false),
      );
    }, delayMs);
  }

  private async restartIce(
    lifecycleEpoch: number,
    generation: number,
    peer: RTCPeerConnection | null,
    force: boolean,
  ) {
    const connectionId = this.admission?.connectionId;
    if (
      !peer ||
      !connectionId ||
      !this.isCurrentConnection(lifecycleEpoch, generation, connectionId) ||
      this.peer !== peer ||
      (!force && peer.connectionState === "connected")
    ) {
      return;
    }
    if (this.iceRestartAttempts >= MAX_ICE_RESTARTS) {
      await this.rebuildPeer(false, lifecycleEpoch, generation);
      return;
    }
    this.iceRestartAttempts += 1;
    peer.restartIce();
    await this.createAndSendOffer(
      peer,
      true,
      lifecycleEpoch,
      generation,
      connectionId,
    );
    this.restartTimer = window.setTimeout(() => {
      this.restartTimer = null;
      if (
        this.isCurrentConnection(lifecycleEpoch, generation, connectionId) &&
        this.peer === peer &&
        peer.connectionState !== "connected"
      ) {
        this.enqueueConnectionOperation(
          lifecycleEpoch,
          generation,
          () => this.rebuildPeer(false, lifecycleEpoch, generation),
        );
      }
    }, ICE_RESTART_TIMEOUT_MS);
  }

  private readonly handleLocalAudioEnded = () => {
    const track = this.localAudioTrack;
    if (!track || track.readyState !== "ended") return;
    void this.setMicrophoneTrack(null).catch((reason: unknown) => {
      this.reportError(reason, "The ended microphone track could not be detached.");
    });
  };

  private installLocalAudioTrack(track: MediaStreamTrack | null) {
    this.localAudioTrack?.removeEventListener("ended", this.handleLocalAudioEnded);
    this.localAudioTrack = track;
    track?.addEventListener("ended", this.handleLocalAudioEnded);
    this.localStream = track ? new MediaStream([track]) : null;
  }

  private flushFrameChannel() {
    const channel = this.frameChannel;
    if (
      !channel ||
      channel.readyState !== "open" ||
      channel.bufferedAmount > FRAME_CHANNEL_HIGH_WATER_BYTES
    ) {
      return;
    }
    const packet = this.pendingFrame.take();
    if (packet) {
      const maxMessageSize = this.peer?.sctp?.maxMessageSize;
      if (
        typeof maxMessageSize === "number" &&
        maxMessageSize > 0 &&
        packet.payload.byteLength > maxMessageSize
      ) {
        const reason = "Selected frame exceeds the negotiated WebRTC message size.";
        this.pendingFrame.offer(packet);
        this.disableSelectedFrameChannel(channel, reason);
        this.notifyError(reason);
        return;
      }
      if (this.sentFrameIds.size >= 64) {
        const oldest = this.sentFrameIds.keys().next().value;
        if (typeof oldest === "string") this.sentFrameIds.delete(oldest);
      }
      try {
        channel.send(packet.payload);
        this.sentFrameIds.set(packet.frameId, packet);
      } catch (reason) {
        this.pendingFrame.offer(packet);
        const message = reason instanceof Error
          ? reason.message
          : "The selected frame could not be sent.";
        this.disableSelectedFrameChannel(channel, message);
        this.notifyError(message);
      }
    }
  }

  private disableSelectedFrameChannel(channel: RTCDataChannel, reason: string) {
    if (this.frameChannel !== channel) return;
    let latest = this.pendingFrame.current;
    for (const packet of this.sentFrameIds.values()) {
      if (!latest || packet.sequence > latest.sequence) latest = packet;
    }
    if (latest) {
      this.pendingFrame.offer(latest);
      this.fallbackFrame.offer(latest.source);
      this.notifySelectedFrameFallback(latest.source, reason);
    }
    this.sentFrameIds.clear();
    this.frameChannel = null;
    try {
      channel.close();
    } catch {
      // The channel is already unusable and the exact latest frame is preserved.
    }
    this.notifySelectedFrameChannelState(false);
  }

  private closePeer() {
    this.clearRecoveryTimers();
    this.clearPeerDeadlines();
    this.pendingNegotiation = null;
    this.pendingLocalCandidates = [];
    this.localIceUfragRevisions.clear();
    this.pendingRemoteCandidates.clear();
    this.descriptionSentRevision = -1;
    this.remoteDescriptionRevision = -1;
    for (const packet of this.sentFrameIds.values()) {
      const pending = this.pendingFrame.current;
      if (!pending || packet.sequence > pending.sequence) this.pendingFrame.offer(packet);
    }
    this.sentFrameIds.clear();
    this.purgeNegotiationMessages();
    this.frameChannel?.close();
    this.frameChannel = null;
    this.notifySelectedFrameChannelState(false);
    this.audioTransceiver = null;
    this.duplexAudioEnabled = false;
    this.peer?.close();
    this.peer = null;
    this.notifyRemoteAudio(null);
  }

  private startHeartbeat(
    intervalMs: number,
    lifecycleEpoch: number,
    generation: number,
    connectionId: string,
  ) {
    this.stopHeartbeat();
    this.lastPongAt = this.now();
    this.heartbeatTimer = window.setInterval(() => {
      if (!this.isCurrentConnection(lifecycleEpoch, generation, connectionId)) return;
      if (heartbeatMissed(this.now(), this.lastPongAt, intervalMs)) {
        this.socket?.close();
        return;
      }
      this.sendNow({
        type: "ping",
        clientTimeMs: this.now(),
        connectionId,
      });
    }, intervalMs);
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer !== null) {
      window.clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private sendControl(signal: ClientSignal) {
    const encoded = JSON.stringify(signal);
    if (new TextEncoder().encode(encoded).byteLength > MAX_SIGNAL_BYTES) {
      this.fail("Realtime control message is too large.");
      return false;
    }
    const socket = this.socket;
    if (
      this.joinedGeneration === this.connectionGeneration &&
      socket?.readyState === WebSocket.OPEN &&
      socket.bufferedAmount <= MAX_SIGNAL_BUFFERED_BYTES &&
      this.outbox.length === 0
    ) {
      try {
        socket.send(encoded);
        return true;
      } catch {
        socket.close();
        return false;
      }
    }
    if (signal.type === "interrupt") {
      this.outbox = this.outbox.filter((message) => message.type !== "interrupt");
    }
    if (signal.type === "ice") {
      this.outbox = this.outbox.filter(
        (message) =>
          message.type !== "ice" ||
          message.connectionId !== signal.connectionId ||
          message.revision === signal.revision,
      );
    }
    if (this.outbox.length >= MAX_OUTBOX_MESSAGES) {
      this.fail("Realtime control backpressure limit was reached.");
      return false;
    }
    this.outbox.push({
      encoded,
      type: signal.type,
      ...("connectionId" in signal ? { connectionId: signal.connectionId } : {}),
      ...("revision" in signal ? { revision: signal.revision } : {}),
    });
    this.scheduleDrain();
    return true;
  }

  private sendNow(signal: ClientSignal) {
    const socket = this.socket;
    if (socket?.readyState !== WebSocket.OPEN || socket.bufferedAmount > MAX_SIGNAL_BUFFERED_BYTES) {
      return false;
    }
    const encoded = JSON.stringify(signal);
    if (new TextEncoder().encode(encoded).byteLength > MAX_SIGNAL_BYTES) return false;
    try {
      socket.send(encoded);
      return true;
    } catch {
      socket.close();
      return false;
    }
  }

  private sendTelemetryNow(
    signal: Extract<
      ClientSignal,
      { type: "playbackSilent" | "turnEnded" | "firstResponseAudio" }
    >,
  ) {
    const connectionId = this.admission?.connectionId;
    if (
      !connectionId ||
      signal.connectionId !== connectionId ||
      this.joinedGeneration !== this.connectionGeneration ||
      this.state !== "connected"
    ) {
      return false;
    }
    return this.sendNow(signal);
  }

  private scheduleDrain() {
    if (this.drainTimer !== null || this.closed) return;
    this.drainTimer = window.setTimeout(() => {
      this.drainTimer = null;
      const socket = this.socket;
      while (
        this.outbox.length > 0 &&
        this.joinedGeneration === this.connectionGeneration &&
        socket?.readyState === WebSocket.OPEN &&
        socket.bufferedAmount <= MAX_SIGNAL_BUFFERED_BYTES
      ) {
        const message = this.outbox.shift();
        if (message) {
          try {
            socket.send(message.encoded);
          } catch {
            socket.close();
            return;
          }
        }
      }
      if (this.outbox.length > 0) this.scheduleDrain();
    }, 25);
  }

  private stopDrainTimer() {
    if (this.drainTimer !== null) {
      window.clearTimeout(this.drainTimer);
      this.drainTimer = null;
    }
  }

  private scheduleReconnect(lifecycleEpoch = this.lifecycleEpoch) {
    if (
      this.closed ||
      lifecycleEpoch !== this.lifecycleEpoch ||
      this.reconnectTimer !== null
    ) {
      return;
    }
    this.emitStatus("reconnecting");
    const delay = reconnectDelayMs(this.reconnectAttempt, this.random());
    this.reconnectAttempt += 1;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      if (lifecycleEpoch === this.lifecycleEpoch) this.beginConnect(lifecycleEpoch);
    }, delay);
  }

  private clearReconnectTimer() {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private clearAdmissionDeadline() {
    if (this.admissionTimeoutTimer !== null) {
      window.clearTimeout(this.admissionTimeoutTimer);
      this.admissionTimeoutTimer = null;
    }
  }

  private clearSocketOpenDeadline() {
    if (this.socketOpenTimeoutTimer !== null) {
      window.clearTimeout(this.socketOpenTimeoutTimer);
      this.socketOpenTimeoutTimer = null;
    }
  }

  private clearRecoveryTimers() {
    if (this.disconnectTimer !== null) window.clearTimeout(this.disconnectTimer);
    if (this.restartTimer !== null) window.clearTimeout(this.restartTimer);
    this.disconnectTimer = null;
    this.restartTimer = null;
  }

  private startNegotiationDeadline(
    pending: PendingNegotiation,
    peer: RTCPeerConnection,
    lifecycleEpoch: number,
    generation: number,
  ) {
    this.clearNegotiationDeadline();
    this.negotiationTimeoutTimer = window.setTimeout(() => {
      this.negotiationTimeoutTimer = null;
      if (
        this.isCurrentConnection(lifecycleEpoch, generation, pending.connectionId) &&
        this.peer === peer &&
        isCurrentNegotiationAnswer(this.pendingNegotiation, pending)
      ) {
        this.notifyError("Realtime negotiation answer timed out.");
        this.recoverConnection(lifecycleEpoch, generation);
      }
    }, NEGOTIATION_ANSWER_TIMEOUT_MS);
  }

  private startConnectionDeadline(
    peer: RTCPeerConnection,
    lifecycleEpoch: number,
    generation: number,
    connectionId: string,
  ) {
    if (this.connectionTimeoutTimer !== null) {
      window.clearTimeout(this.connectionTimeoutTimer);
    }
    this.connectionTimeoutTimer = window.setTimeout(() => {
      this.connectionTimeoutTimer = null;
      if (
        this.isCurrentConnection(lifecycleEpoch, generation, connectionId) &&
        this.peer === peer &&
        peer.connectionState !== "connected"
      ) {
        this.notifyError("Realtime media connection timed out.");
        this.recoverConnection(lifecycleEpoch, generation);
      }
    }, PEER_CONNECTION_TIMEOUT_MS);
  }

  private clearNegotiationDeadline() {
    if (this.negotiationTimeoutTimer !== null) {
      window.clearTimeout(this.negotiationTimeoutTimer);
      this.negotiationTimeoutTimer = null;
    }
  }

  private clearPeerDeadlines() {
    this.clearNegotiationDeadline();
    this.clearConnectionDeadline();
  }

  private clearConnectionDeadline() {
    if (this.connectionTimeoutTimer !== null) {
      window.clearTimeout(this.connectionTimeoutTimer);
      this.connectionTimeoutTimer = null;
    }
  }

  private clearJoinTimers() {
    if (this.joinRetryTimer !== null) window.clearTimeout(this.joinRetryTimer);
    if (this.joinTimeoutTimer !== null) window.clearTimeout(this.joinTimeoutTimer);
    this.joinRetryTimer = null;
    this.joinTimeoutTimer = null;
  }

  private queue(task: () => void | Promise<void>) {
    if (this.queuedOperations >= MAX_QUEUED_OPERATIONS) {
      return Promise.reject(new Error("Realtime operation queue is full."));
    }
    this.queuedOperations += 1;
    const runTask = () => this.withOperationTimeout(task);
    const run = this.operation.then(runTask, runTask);
    this.operation = run.then(
      () => undefined,
      () => undefined,
    );
    return run.finally(() => {
      this.queuedOperations -= 1;
    });
  }

  private withOperationTimeout(task: () => void | Promise<void>) {
    return new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(
        () => reject(new Error("Realtime media operation timed out.")),
        OPERATION_TIMEOUT_MS,
      );
      Promise.resolve()
        .then(task)
        .then(
          () => {
            window.clearTimeout(timer);
            resolve();
          },
          (reason: unknown) => {
            window.clearTimeout(timer);
            reject(reason);
          },
        );
    });
  }

  private enqueueConnectionOperation(
    lifecycleEpoch: number,
    generation: number,
    task: () => void | Promise<void>,
  ) {
    void this.queue(async () => {
      if (!this.isCurrentConnection(lifecycleEpoch, generation)) return;
      await task();
    }).catch((reason: unknown) => {
      this.reportError(reason, "Realtime media operation failed.");
      this.recoverConnection(lifecycleEpoch, generation);
    });
  }

  private isCurrentConnection(
    lifecycleEpoch: number,
    generation: number,
    connectionId?: string,
  ) {
    return !this.closed &&
      lifecycleEpoch === this.lifecycleEpoch &&
      generation === this.connectionGeneration &&
      (connectionId === undefined || this.admission?.connectionId === connectionId);
  }

  private purgeObsoleteNegotiationMessages(connectionId: string) {
    this.outbox = this.outbox.filter(
      (message) =>
        message.connectionId !== connectionId ||
        !["offer", "answer", "ice"].includes(message.type),
    );
  }

  private purgeNegotiationMessages() {
    this.outbox = this.outbox.filter(
      (message) => !["offer", "answer", "ice"].includes(message.type),
    );
  }

  private recoverConnection(lifecycleEpoch: number, generation: number) {
    if (!this.isCurrentConnection(lifecycleEpoch, generation)) return;
    const socket = this.socket;
    this.socket = null;
    this.connectionGeneration += 1;
    this.joinedGeneration = -1;
    this.clearJoinTimers();
    this.stopHeartbeat();
    this.stopDrainTimer();
    this.clearAdmissionDeadline();
    this.clearSocketOpenDeadline();
    this.outbox = [];
    this.closePeer();
    this.admission = null;
    try {
      socket?.close();
    } catch {
      // The lifecycle generation already fenced every old callback.
    }
    if (!this.closed) this.scheduleReconnect(lifecycleEpoch);
  }

  private terminalFailure(message: string) {
    this.closed = true;
    this.lifecycleEpoch += 1;
    this.connectionGeneration += 1;
    this.joinedGeneration = -1;
    this.openingEpoch = null;
    this.clearReconnectTimer();
    this.clearAdmissionDeadline();
    this.clearSocketOpenDeadline();
    this.clearJoinTimers();
    this.stopHeartbeat();
    this.clearRecoveryTimers();
    this.stopDrainTimer();
    const socket = this.socket;
    this.socket = null;
    this.closePeer();
    const track = this.localAudioTrack;
    this.installLocalAudioTrack(null);
    track?.stop();
    this.admission = null;
    this.outbox = [];
    this.pendingFrame.clear();
    this.fallbackFrame.clear();
    this.sentFrameIds.clear();
    socket?.close();
    this.emitStatus("failed", message);
  }

  private fail(message: string) {
    this.notifyError(message);
    this.emitStatus("failed", message);
    this.socket?.close();
  }

  private reportError(reason: unknown, fallback: string) {
    const message = reason instanceof Error ? reason.message : fallback;
    this.notifyError(message);
  }

  private emitStatus(state: RealtimeVoiceState, detail?: string) {
    this.state = state;
    this.notifyStatus({
      state,
      connectionId: this.admission?.connectionId ?? null,
      playbackGeneration: this.playbackGeneration,
      ...(detail ? { detail } : {}),
    });
  }

  private notifyStatus(status: RealtimeVoiceStatus) {
    try {
      this.options.onStatus?.(status);
    } catch {
      // Application callbacks cannot break transport cleanup or recovery.
    }
  }

  private notifyRemoteAudio(stream: MediaStream | null) {
    try {
      this.options.onRemoteAudio?.(stream);
    } catch {
      // Application callbacks cannot break transport cleanup or recovery.
    }
  }

  private notifyFrameAccepted(frameId: string, timestampMs: number, sequence: number) {
    try {
      this.options.onFrameAccepted?.(frameId, timestampMs, sequence);
    } catch {
      // Application callbacks cannot break transport cleanup or recovery.
    }
  }

  private notifySelectedFrameChannelState(available: boolean) {
    try {
      this.options.onSelectedFrameChannelState?.(available);
    } catch {
      // Application callbacks cannot break transport cleanup or recovery.
    }
  }

  private notifySelectedFrameFallback(frame: SelectedFramePacket, reason: string) {
    try {
      this.options.onSelectedFrameFallback?.(frame, reason);
    } catch {
      // The exact latest fallback frame remains available through takeSelectedFrameFallback.
    }
  }

  private notifyError(message: string) {
    try {
      this.options.onError?.(message);
    } catch {
      // Error reporting must never throw back into the transport.
    }
  }
}
