import assert from "node:assert/strict";
import test from "node:test";
import {
  heartbeatMissed,
  iceUsernameFragment,
  iceUsernameFragmentsFromSdp,
  isCurrentNegotiationAnswer,
  LatestOnlyValue,
  parseServerSignal,
  RealtimeVoiceTransport,
  reconnectDelayMs,
  validateIceServers,
  validateRealtimeAdmission,
  type RealtimeAdmission,
} from "./realtime-webrtc";

type Listener = (event: unknown) => void;

class FakeSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  readyState = FakeSocket.CONNECTING;
  bufferedAmount = 0;
  readonly sent: string[] = [];
  private readonly listeners = new Map<string, Listener[]>();

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    const callback = listener as unknown as Listener;
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback]);
  }

  send(data: string | ArrayBufferLike | Blob | ArrayBufferView) {
    if (this.readyState !== FakeSocket.OPEN) throw new Error("socket is not open");
    this.sent.push(typeof data === "string" ? data : "<binary>");
  }

  open() {
    this.readyState = FakeSocket.OPEN;
    this.emit("open", {});
  }

  server(signal: Record<string, unknown>) {
    this.emit("message", { data: JSON.stringify(signal) });
  }

  close() {
    this.readyState = FakeSocket.CLOSED;
    this.emit("close", {});
  }

  emit(type: string, event: unknown) {
    for (const listener of this.listeners.get(type) ?? []) listener(event);
  }
}

class FakeDataChannel {
  readyState: RTCDataChannelState = "open";
  bufferedAmount = 0;
  bufferedAmountLowThreshold = 0;
  binaryType: BinaryType = "arraybuffer";
  throwOnSend = false;
  readonly sent: ArrayBuffer[] = [];
  private readonly listeners = new Map<string, Listener[]>();

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    const callback = listener as unknown as Listener;
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback]);
  }

  send(data: string | Blob | ArrayBuffer | ArrayBufferView) {
    if (this.throwOnSend) throw new Error("simulated channel failure");
    if (data instanceof ArrayBuffer) this.sent.push(data);
  }

  close() {
    this.readyState = "closed";
  }
}

class FakeTrack {
  readonly kind = "audio";
  readyState: MediaStreamTrackState = "live";
  stopped = false;
  private readonly listeners = new Map<string, Listener[]>();

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    const callback = listener as unknown as Listener;
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback]);
  }

  removeEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    const callback = listener as unknown as Listener;
    this.listeners.set(
      type,
      (this.listeners.get(type) ?? []).filter((candidate) => candidate !== callback),
    );
  }

  stop() {
    this.stopped = true;
    this.readyState = "ended";
    for (const listener of this.listeners.get("ended") ?? []) listener({});
  }
}

class FakeMediaStream {
  constructor(readonly tracks: MediaStreamTrack[] = []) {}
  getTracks() {
    return [...this.tracks];
  }
}

class FakePeer {
  connectionState: RTCPeerConnectionState = "new";
  signalingState: RTCSignalingState = "stable";
  localDescription: RTCSessionDescription | null = null;
  remoteDescription: RTCSessionDescription | null = null;
  readonly sctp = { maxMessageSize: 1024 * 1024 };
  readonly dataChannel = new FakeDataChannel();
  readonly sender = {
    replaceTrack: async () => undefined,
  };
  transceiverSource: MediaStreamTrack | string | null = null;
  transceiverDirection: RTCRtpTransceiverDirection | null = null;
  offerCount = 0;
  answerGate: Promise<RTCSessionDescriptionInit> | null = null;
  closed = false;
  private readonly listeners = new Map<string, Listener[]>();

  addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
    const callback = listener as unknown as Listener;
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), callback]);
  }

  addTransceiver(
    source: MediaStreamTrack | string,
    init?: RTCRtpTransceiverInit,
  ) {
    this.transceiverSource = source;
    this.transceiverDirection = init?.direction ?? null;
    return { sender: this.sender } as unknown as RTCRtpTransceiver;
  }

  createDataChannel() {
    return this.dataChannel as unknown as RTCDataChannel;
  }

  async createOffer() {
    this.offerCount += 1;
    return {
      type: "offer",
      sdp: `v=0\r\na=ice-ufrag:offer-${this.offerCount}\r\n`,
    } as RTCSessionDescriptionInit;
  }

  async createAnswer() {
    if (this.answerGate) return this.answerGate;
    return { type: "answer", sdp: "v=0 answer" } as RTCSessionDescriptionInit;
  }

  async setLocalDescription(description: RTCSessionDescriptionInit) {
    if (description.type === "rollback") {
      this.localDescription = null;
      this.signalingState = "stable";
      return;
    }
    this.localDescription = description as RTCSessionDescription;
  }

  async setRemoteDescription(description: RTCSessionDescriptionInit) {
    this.remoteDescription = description as RTCSessionDescription;
  }

  async addIceCandidate() {}

  restartIce() {}

  emitIce(candidate: RTCIceCandidateInit) {
    for (const listener of this.listeners.get("icecandidate") ?? []) {
      listener({ candidate: { toJSON: () => candidate } });
    }
  }

  emitConnectionState() {
    for (const listener of this.listeners.get("connectionstatechange") ?? []) {
      listener({});
    }
  }

  close() {
    this.closed = true;
    this.connectionState = "closed";
  }
}

function installBrowserFakes() {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: globalThis,
  });
  Object.defineProperty(globalThis, "WebSocket", {
    configurable: true,
    value: FakeSocket,
  });
  Object.defineProperty(globalThis, "MediaStream", {
    configurable: true,
    value: FakeMediaStream,
  });
}

function settle(ms = 0) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

function admission(overrides: Partial<RealtimeAdmission> = {}): RealtimeAdmission {
  return {
    protocolVersion: 1,
    sessionId: "session-1",
    connectionId: "connection-1",
    signallingUrl: "wss://realtime.example.test/v1",
    accessToken: "short-lived-token",
    expiresAtUnixMs: 20_000,
    heartbeatIntervalMs: 5_000,
    iceServers: [{ urls: "stun:stun.example.test" }],
    ...overrides,
  };
}

test("admission permits WSS and loopback WS without URL credentials", () => {
  assert.equal(validateRealtimeAdmission(admission(), 10_000).heartbeatIntervalMs, 5_000);
  assert.equal(
    validateRealtimeAdmission(admission({ signallingUrl: "ws://127.0.0.1:8787/realtime" }), 10_000)
      .connectionId,
    "connection-1",
  );
  assert.throws(
    () => validateRealtimeAdmission(admission({ signallingUrl: "ws://example.test/realtime" }), 10_000),
    /WSS/,
  );
  assert.throws(
    () => validateRealtimeAdmission(admission({ signallingUrl: "wss://example.test/?token=secret" }), 10_000),
    /credentials/,
  );
});

test("expired or unbound admissions are rejected", () => {
  assert.throws(
    () => validateRealtimeAdmission(admission({ expiresAtUnixMs: 9_999 }), 10_000),
    /expired/,
  );
  assert.throws(
    () => validateRealtimeAdmission(admission({ sessionId: "unsafe session" }), 10_000),
    /identity/,
  );
});

test("ICE admission is bounded and keeps credentials out of URLs", () => {
  assert.deepEqual(
    validateIceServers([{
      urls: ["stun:stun.example.test", "turns:turn.example.test?transport=tcp"],
      username: "temporary-user",
      credential: "temporary-password",
    }]),
    [{
      urls: ["stun:stun.example.test", "turns:turn.example.test?transport=tcp"],
      username: "temporary-user",
      credential: "temporary-password",
    }],
  );
  assert.throws(
    () => validateIceServers([{ urls: "https://turn.example.test" }]),
    /STUN or TURN/,
  );
  assert.throws(
    () => validateIceServers([{ urls: "turn:user:password@turn.example.test" }]),
    /without credentials/,
  );
  assert.throws(
    () => validateIceServers(Array.from({ length: 9 }, () => ({ urls: "stun:x" }))),
    /invalid ICE configuration/,
  );
});

test("reconnect delay is exponential, capped, and jittered", () => {
  assert.equal(reconnectDelayMs(0, 0), 200);
  assert.equal(reconnectDelayMs(0, 0.5), 250);
  assert.equal(reconnectDelayMs(3, 0.5), 2_000);
  assert.equal(reconnectDelayMs(20, 0.5), 5_000);
  assert.equal(reconnectDelayMs(20, 1), 6_000);
});

test("heartbeat closes only after more than two and a half intervals", () => {
  assert.equal(heartbeatMissed(13_500, 1_000, 5_000), false);
  assert.equal(heartbeatMissed(13_501, 1_000, 5_000), true);
});

test("late negotiation answers cannot mutate a newer peer", () => {
  const current = { connectionId: "c1", requestId: "r2", revision: 2 };
  assert.equal(isCurrentNegotiationAnswer(current, current), true);
  assert.equal(
    isCurrentNegotiationAnswer(current, { connectionId: "c1", requestId: "r1", revision: 1 }),
    false,
  );
  assert.equal(
    isCurrentNegotiationAnswer(current, { connectionId: "old", requestId: "r2", revision: 2 }),
    false,
  );
});

test("local ICE candidates stay bound to their SDP generation", () => {
  assert.deepEqual(
    iceUsernameFragmentsFromSdp(
      "v=0\r\na=ice-ufrag:first\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=ice-ufrag:second\r\n",
    ),
    ["first", "second"],
  );
  assert.equal(
    iceUsernameFragment({ candidate: "candidate:1 1 UDP 1 127.0.0.1 9 typ host ufrag old" }),
    "old",
  );
  assert.equal(
    iceUsernameFragment({ candidate: "candidate:2", usernameFragment: "new" }),
    "new",
  );
  assert.equal(iceUsernameFragment({ candidate: "candidate:3" }), null);
});

test("latest-only buffer replaces stale selected frames", () => {
  const values = new LatestOnlyValue<number>();
  assert.equal(values.offer(1), false);
  assert.equal(values.offer(2), true);
  assert.equal(values.pending, true);
  assert.equal(values.take(), 2);
  assert.equal(values.take(), null);
});

test("selected-frame extension requires a complete bounded JPEG", () => {
  const transport = new RealtimeVoiceTransport({ admit: async () => admission() });
  assert.throws(
    () => transport.offerSelectedFrame({
      frameId: "frame-1",
      sequence: 1,
      timestampMs: 1,
      differencePercent: 25,
      keyframe: false,
      jpeg: Uint8Array.of(1, 2, 3, 4).buffer,
    }),
    /complete JPEG/,
  );
  assert.deepEqual(
    transport.offerSelectedFrame({
      frameId: "frame-2",
      sequence: 2,
      timestampMs: 2,
      differencePercent: 100,
      keyframe: true,
      jpeg: Uint8Array.of(0xff, 0xd8, 0xff, 0xd9).buffer,
    }),
    { frameId: "frame-2", replaced: false, replacedFrameId: null },
  );
});

test("server signals are bounded and structurally checked", () => {
  assert.deepEqual(
    parseServerSignal(JSON.stringify({
      type: "answer",
      connectionId: "c1",
      requestId: "r1",
      revision: 3,
      sdp: "v=0",
    })),
    { type: "answer", connectionId: "c1", requestId: "r1", revision: 3, sdp: "v=0" },
  );
  assert.throws(
    () => parseServerSignal('{"type":"ice","connectionId":"c1","revision":1}'),
    /ICE/,
  );
  assert.throws(() => parseServerSignal(JSON.stringify({ type: "unknown" })), /not supported/);
});

test("synchronous admission failures enter bounded reconnect instead of escaping", async () => {
  installBrowserFakes();
  const errors: string[] = [];
  const transport = new RealtimeVoiceTransport({
    admit: () => {
      throw new Error("admission unavailable");
    },
    random: () => 0,
    onError: (message) => errors.push(message),
  });

  assert.doesNotThrow(() => transport.connect());
  await settle();
  assert.equal(transport.status.state, "reconnecting");
  assert.match(errors[0], /admission unavailable/);
  transport.disconnect();
});

test("a stale pending admission cannot authenticate a newer lifecycle", async () => {
  installBrowserFakes();
  const pending: Array<(value: RealtimeAdmission) => void> = [];
  const sockets: FakeSocket[] = [];
  const transport = new RealtimeVoiceTransport({
    admit: () => new Promise((resolve) => pending.push(resolve)),
    now: () => 10_000,
    webSocketFactory: () => {
      const socket = new FakeSocket();
      sockets.push(socket);
      return socket as unknown as WebSocket;
    },
  });

  transport.connect();
  await settle();
  assert.equal(pending.length, 1);
  transport.disconnect();
  transport.connect();
  await settle();
  assert.equal(pending.length, 2);
  pending[0](admission({ connectionId: "stale", signallingUrl: "ws://127.0.0.1/stale" }));
  pending[1](admission({ connectionId: "current", signallingUrl: "ws://127.0.0.1/current" }));
  await settle();

  assert.equal(sockets.length, 1);
  assert.equal(transport.status.connectionId, "current");
  transport.disconnect();
});

test("join waits for WebSocket backpressure instead of being dropped", async () => {
  installBrowserFakes();
  const socket = new FakeSocket();
  socket.bufferedAmount = 300 * 1024;
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({ signallingUrl: "ws://127.0.0.1/realtime" }),
    now: () => 10_000,
    webSocketFactory: () => socket as unknown as WebSocket,
  });

  transport.connect();
  await settle();
  socket.open();
  await settle(40);
  assert.equal(socket.sent.length, 0);
  socket.bufferedAmount = 0;
  await settle(40);
  assert.equal(JSON.parse(socket.sent[0]).type, "join");
  transport.disconnect();
});

test("a delayed stale socket close cannot tear down the replacement peer", async () => {
  installBrowserFakes();
  const sockets: FakeSocket[] = [];
  const peers: FakePeer[] = [];
  let admitted = 0;
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({
      connectionId: `connection-${++admitted}`,
      signallingUrl: `ws://127.0.0.1/${admitted}`,
    }),
    now: () => 10_000,
    random: () => 0,
    webSocketFactory: () => {
      const socket = new FakeSocket();
      sockets.push(socket);
      return socket as unknown as WebSocket;
    },
    peerFactory: () => {
      const peer = new FakePeer();
      peers.push(peer);
      return peer as unknown as RTCPeerConnection;
    },
  });

  transport.connect();
  await settle();
  sockets[0].open();
  sockets[0].server({ type: "joined", connectionId: "connection-1", heartbeatIntervalMs: 5_000 });
  await settle();
  assert.equal(peers.length, 1);
  sockets[0].close();
  await settle(220);
  assert.equal(sockets.length, 2);
  sockets[1].open();
  sockets[1].server({ type: "joined", connectionId: "connection-2", heartbeatIntervalMs: 5_000 });
  await settle();
  assert.equal(peers.length, 2);

  sockets[0].emit("close", {});
  assert.equal(peers[1].closed, false);
  assert.equal(transport.status.connectionId, "connection-2");
  transport.disconnect();
});

test("selected-frame send failures preserve only the newest unsent frame", async () => {
  installBrowserFakes();
  const socket = new FakeSocket();
  const peer = new FakePeer();
  peer.dataChannel.throwOnSend = true;
  const fallbackIds: string[] = [];
  const channelStates: boolean[] = [];
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({
      signallingUrl: "ws://127.0.0.1/realtime",
      capabilities: { duplexAudio: true, selectedFrameDataChannel: true },
    }),
    now: () => 10_000,
    webSocketFactory: () => socket as unknown as WebSocket,
    peerFactory: () => peer as unknown as RTCPeerConnection,
    onSelectedFrameFallback: (frame) => fallbackIds.push(frame.frameId),
    onSelectedFrameChannelState: (available) => channelStates.push(available),
  });
  transport.connect();
  await settle();
  socket.open();
  socket.server({ type: "joined", connectionId: "connection-1", heartbeatIntervalMs: 5_000 });
  await settle();

  const first = transport.offerSelectedFrame({
    frameId: "frame-a",
    sequence: 1,
    timestampMs: 1,
    differencePercent: 20,
    keyframe: false,
    jpeg: Uint8Array.of(0xff, 0xd8, 0xff, 0xd9).buffer,
  });
  const second = transport.offerSelectedFrame({
    frameId: "frame-b",
    sequence: 2,
    timestampMs: 2,
    differencePercent: 30,
    keyframe: false,
    jpeg: Uint8Array.of(0xff, 0xd8, 0xff, 0xd9).buffer,
  });
  assert.equal(first.replaced, false);
  assert.deepEqual(fallbackIds, ["frame-a"]);
  assert.equal(channelStates.at(-1), false);
  assert.equal(transport.takeSelectedFrameFallback()?.frameId, "frame-a");
  assert.deepEqual(second, {
    frameId: "frame-b",
    replaced: true,
    replacedFrameId: "frame-a",
  });
  transport.disconnect();
});

test("SCTP message limits preserve the exact frame for HTTP fallback", async () => {
  installBrowserFakes();
  const socket = new FakeSocket();
  const peer = new FakePeer();
  peer.sctp.maxMessageSize = 8;
  const fallbackIds: string[] = [];
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({
      signallingUrl: "ws://127.0.0.1/realtime",
      capabilities: { selectedFrameDataChannel: true },
    }),
    now: () => 10_000,
    webSocketFactory: () => socket as unknown as WebSocket,
    peerFactory: () => peer as unknown as RTCPeerConnection,
    onSelectedFrameFallback: (frame) => fallbackIds.push(frame.frameId),
  });
  transport.connect();
  await settle();
  socket.open();
  socket.server({ type: "joined", connectionId: "connection-1" });
  await settle();

  transport.offerSelectedFrame({
    frameId: "frame-large",
    sequence: 1,
    timestampMs: 10,
    differencePercent: 40,
    keyframe: true,
    jpeg: Uint8Array.of(0xff, 0xd8, 0xff, 0xd9).buffer,
  });
  assert.deepEqual(fallbackIds, ["frame-large"]);
  assert.equal(transport.takeSelectedFrameFallback()?.frameId, "frame-large");
  transport.disconnect();
});

test("a deferred stale server offer cannot mutate a replacement lifecycle", async () => {
  installBrowserFakes();
  const sockets: FakeSocket[] = [];
  const peers: FakePeer[] = [];
  let admitted = 0;
  let resolveAnswer!: (answer: RTCSessionDescriptionInit) => void;
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({
      connectionId: `connection-${++admitted}`,
      signallingUrl: `ws://127.0.0.1/${admitted}`,
    }),
    now: () => 10_000,
    webSocketFactory: () => {
      const socket = new FakeSocket();
      sockets.push(socket);
      return socket as unknown as WebSocket;
    },
    peerFactory: () => {
      const peer = new FakePeer();
      peers.push(peer);
      return peer as unknown as RTCPeerConnection;
    },
  });
  transport.connect();
  await settle();
  sockets[0].open();
  sockets[0].server({ type: "joined", connectionId: "connection-1" });
  await settle();
  peers[0].answerGate = new Promise((resolve) => {
    resolveAnswer = resolve;
  });
  sockets[0].server({
    type: "offer",
    connectionId: "connection-1",
    requestId: "server-offer",
    revision: 2,
    sdp: "v=0 stale server offer",
  });
  await settle();

  transport.disconnect();
  transport.connect();
  resolveAnswer({ type: "answer", sdp: "v=0\r\na=ice-ufrag:stale-answer\r\n" });
  await settle();
  sockets[1].open();
  sockets[1].server({ type: "joined", connectionId: "connection-2" });
  await settle();

  assert.equal(peers.length, 2);
  assert.match(peers[1].localDescription?.sdp ?? "", /offer-1/);
  assert.equal(transport.status.connectionId, "connection-2");
  transport.disconnect();
});

test("selected-frame acknowledgements require the exact sent sequence", async () => {
  installBrowserFakes();
  const socket = new FakeSocket();
  const peer = new FakePeer();
  const accepted: number[] = [];
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({
      signallingUrl: "ws://127.0.0.1/realtime",
      capabilities: { selectedFrameDataChannel: true },
    }),
    now: () => 10_000,
    webSocketFactory: () => socket as unknown as WebSocket,
    peerFactory: () => peer as unknown as RTCPeerConnection,
    onFrameAccepted: (_frameId, _timestampMs, sequence) => accepted.push(sequence),
  });
  transport.connect();
  await settle();
  socket.open();
  socket.server({ type: "joined", connectionId: "connection-1" });
  await settle();
  transport.offerSelectedFrame({
    frameId: "frame-ack",
    sequence: 1,
    timestampMs: 42,
    differencePercent: 20,
    keyframe: false,
    jpeg: Uint8Array.of(0xff, 0xd8, 0xff, 0xd9).buffer,
  });

  socket.server({
    type: "frameAccepted",
    connectionId: "connection-1",
    frameId: "frame-ack",
    sequence: 999,
  });
  assert.deepEqual(accepted, []);
  socket.server({
    type: "frameAccepted",
    connectionId: "connection-1",
    frameId: "frame-ack",
    sequence: 1,
  });
  assert.deepEqual(accepted, [1]);
  transport.disconnect();
});

test("late ICE candidates keep the revision of their own local description", async () => {
  installBrowserFakes();
  const socket = new FakeSocket();
  const peer = new FakePeer();
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({ signallingUrl: "ws://127.0.0.1/realtime" }),
    now: () => 10_000,
    webSocketFactory: () => socket as unknown as WebSocket,
    peerFactory: () => peer as unknown as RTCPeerConnection,
  });
  transport.connect();
  await settle();
  socket.open();
  socket.server({ type: "joined", connectionId: "connection-1" });
  await settle();
  socket.server({ type: "restart", connectionId: "connection-1" });
  await settle();

  peer.emitIce({ candidate: "candidate:old", usernameFragment: "offer-1" });
  peer.emitIce({ candidate: "candidate:new", usernameFragment: "offer-2" });
  const iceSignals = socket.sent
    .map((value) => JSON.parse(value) as Record<string, unknown>)
    .filter((signal) => signal.type === "ice");
  assert.equal(iceSignals.length, 1);
  assert.equal(iceSignals[0].revision, 2);
  assert.deepEqual(iceSignals[0].candidate, {
    candidate: "candidate:new",
    usernameFragment: "offer-2",
  });
  transport.disconnect();
});

test("microphone media stays receive-only unless duplex audio is admitted", async () => {
  installBrowserFakes();
  const socket = new FakeSocket();
  const peer = new FakePeer();
  const track = new FakeTrack();
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({
      signallingUrl: "ws://127.0.0.1/realtime",
      capabilities: { duplexAudio: false },
    }),
    now: () => 10_000,
    webSocketFactory: () => socket as unknown as WebSocket,
    peerFactory: () => peer as unknown as RTCPeerConnection,
  });
  transport.connect(track as unknown as MediaStreamTrack);
  await settle();
  socket.open();
  socket.server({ type: "joined", connectionId: "connection-1" });
  await settle();

  assert.equal(peer.transceiverSource, "audio");
  assert.equal(peer.transceiverDirection, "recvonly");
  assert.equal(track.stopped, false);
  transport.disconnect();
  assert.equal(track.stopped, true);
});

test("interrupts are generation-fenced and sent on the live connection", async () => {
  installBrowserFakes();
  const socket = new FakeSocket();
  const peer = new FakePeer();
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({ signallingUrl: "ws://127.0.0.1/realtime" }),
    now: () => 10_000,
    webSocketFactory: () => socket as unknown as WebSocket,
    peerFactory: () => peer as unknown as RTCPeerConnection,
  });
  transport.connect();
  await settle();
  socket.open();
  socket.server({ type: "joined", connectionId: "connection-1" });
  await settle();
  peer.connectionState = "connected";
  peer.emitConnectionState();

  assert.equal(transport.interrupt(), 1);
  assert.equal(transport.acknowledgePlaybackSilent(1), true);
  assert.equal(transport.markTurnEnded("turn-1"), true);
  assert.equal(transport.acknowledgeFirstResponseAudio("turn-1"), true);
  assert.equal(transport.interrupt(), 2);
  const controls = socket.sent
    .map((value) => JSON.parse(value) as Record<string, unknown>)
    .filter((signal) => [
      "interrupt",
      "playbackSilent",
      "turnEnded",
      "firstResponseAudio",
    ].includes(signal.type as string));
  const interrupts = controls.filter((signal) => signal.type === "interrupt");
  assert.deepEqual(interrupts.map((signal) => signal.playbackGeneration), [1, 2]);
  assert.deepEqual(controls.slice(1, 4), [
    {
      type: "playbackSilent",
      connectionId: "connection-1",
      playbackGeneration: 1,
    },
    { type: "turnEnded", connectionId: "connection-1", turnId: "turn-1" },
    {
      type: "firstResponseAudio",
      connectionId: "connection-1",
      turnId: "turn-1",
    },
  ]);
  assert.equal(transport.status.playbackGeneration, 2);
  assert.throws(() => transport.markTurnEnded("unsafe turn"), /turn ID/);
  assert.throws(() => transport.acknowledgePlaybackSilent(3), /generation/);
  transport.disconnect();
});

test("application callbacks cannot abort transport cleanup", () => {
  installBrowserFakes();
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission(),
    onRemoteAudio: () => {
      throw new Error("application callback failed");
    },
    onStatus: () => {
      throw new Error("application callback failed");
    },
  });
  assert.doesNotThrow(() => transport.disconnect());
  assert.equal(transport.status.connectionId, null);
  assert.equal(transport.status.state, "closed");
});

test("repeated connect replaces and owns the live microphone track", async () => {
  installBrowserFakes();
  const socket = new FakeSocket();
  const first = new FakeTrack();
  const second = new FakeTrack();
  const transport = new RealtimeVoiceTransport({
    admit: async () => admission({ signallingUrl: "ws://127.0.0.1/realtime" }),
    now: () => 10_000,
    webSocketFactory: () => socket as unknown as WebSocket,
  });

  transport.connect(first as unknown as MediaStreamTrack);
  transport.connect(second as unknown as MediaStreamTrack);
  await settle();
  assert.equal(first.stopped, true);
  assert.equal(second.stopped, false);
  transport.disconnect();
  assert.equal(second.stopped, true);
});
