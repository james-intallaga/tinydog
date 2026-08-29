"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { MicVAD as SileroMicVAD } from "@ricky0123/vad-web";
import { AudioLines, ScanEye, SquareTerminal } from "lucide-react";
import {
  clampFrameDifferenceThreshold,
  decideFrameTransmission,
  normalizedFrameDifferencePercent,
} from "@/lib/frame-difference";
import {
  localRealtimeMediaTransport,
  MediaTransportError,
  type RealtimeMediaTransport,
} from "@/lib/media-transport";
import {
  fetchRealtimeGatewayStatus,
  requestRealtimeAdmission,
  unavailableRealtimeGatewayStatus,
  type RealtimeGatewayStatus,
} from "@/lib/realtime-gateway";
import {
  ResponseAudioDetector,
  rmsLevel,
  VoiceEndpointDetector,
} from "@/lib/realtime-audio-telemetry";
import {
  RealtimeVoiceTransport,
  type RealtimeVoiceStatus,
  type SelectedFramePacket,
} from "@/lib/realtime-webrtc";
import styles from "./page.module.css";

const ANALYSIS_WIDTH = 64;
const ANALYSIS_HEIGHT = 36;
const MAX_FRAME_WIDTH = 640;
const MAX_FRAME_HEIGHT = 360;
const JPEG_QUALITY = 0.72;
const AUDIO_TIMESLICE_MS = 750;
const SAFE_SESSION = /^[A-Za-z0-9._:-]{1,128}$/;

type CaptureState = "off" | "requesting" | "capturing" | "live" | "waiting" | "error";
type MicrophoneMode = "webrtc" | "http" | null;
type VadEngine = "off" | "loading" | "silero" | "level-fallback";
export type CompanionState = "idle" | "connecting" | "listening" | "speaking" | "error";

type PendingFrame = {
  sequence: number;
  pixels: Uint8ClampedArray;
  capturedAtMs: number;
};

type QueuedAudio = {
  blob: Blob;
  timestampMs: number;
  sequence: number;
  mimeType: string;
  generation: number;
};

type LevelMonitor = {
  stop: () => void;
  resume: () => Promise<void>;
};

function monitorStreamLevel(
  stream: MediaStream,
  onLevel: (level: number, nowMs: number) => void,
): LevelMonitor | null {
  if (typeof AudioContext === "undefined") return null;
  let context: AudioContext;
  try {
    context = new AudioContext({ latencyHint: "interactive" });
  } catch {
    return null;
  }
  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  analyser.smoothingTimeConstant = 0;
  source.connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  let stopped = false;
  const timer = window.setInterval(() => {
    if (stopped || context.state !== "running" || document.visibilityState === "hidden") {
      return;
    }
    analyser.getFloatTimeDomainData(samples);
    onLevel(rmsLevel(samples), performance.now());
  }, 20);
  return {
    stop: () => {
      if (stopped) return;
      stopped = true;
      window.clearInterval(timer);
      source.disconnect();
      analyser.disconnect();
      void context.close().catch(() => undefined);
    },
    resume: async () => {
      if (!stopped && context.state === "suspended") await context.resume();
    },
  };
}

function frameBlob(canvas: HTMLCanvasElement) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("The camera frame could not be encoded."))),
      "image/jpeg",
      JPEG_QUALITY,
    );
  });
}

function preferredAudioMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/mp4;codecs=mp4a.40.2",
    "audio/webm",
    "audio/mp4",
  ];
  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) ?? "";
}

function readableCaptureState(state: CaptureState) {
  if (state === "requesting") return "Asking permission";
  if (state === "capturing") return "Connecting";
  if (state === "live") return "Live";
  if (state === "waiting") return "Waiting";
  if (state === "error") return "Unavailable";
  return "Off";
}

function captureErrorMessage(reason: unknown, fallback: string) {
  if (reason instanceof DOMException && reason.name === "AbortError") return "";
  return reason instanceof Error ? reason.message : fallback;
}

export default function MediaCapture({
  sessionId,
  transport = localRealtimeMediaTransport,
  minimal = false,
  onCompanionState,
  onOpenTextInput,
}: {
  sessionId: string | null;
  transport?: RealtimeMediaTransport;
  minimal?: boolean;
  onCompanionState?: (state: CompanionState) => void;
  onOpenTextInput?: () => void;
}) {
  const fallbackSessionId = `mlm-media-${useId().replace(/[^A-Za-z0-9._-]/g, "")}`;
  const localSessionId = sessionId && SAFE_SESSION.test(sessionId) ? sessionId : fallbackSessionId;
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [cameraState, setCameraState] = useState<CaptureState>("off");
  const [microphoneState, setMicrophoneState] = useState<CaptureState>("off");
  const [microphoneActive, setMicrophoneActive] = useState(false);
  const [microphoneMode, setMicrophoneMode] = useState<MicrophoneMode>(null);
  const [thresholdPercent, setThresholdPercent] = useState(10);
  const [maxFramesPerSecond, setMaxFramesPerSecond] = useState(2);
  const [heartbeatSeconds, setHeartbeatSeconds] = useState(5);
  const [currentDifference, setCurrentDifference] = useState(0);
  const [sentFrames, setSentFrames] = useState(0);
  const [skippedFrames, setSkippedFrames] = useState(0);
  const [sentAudioChunks, setSentAudioChunks] = useState(0);
  const [droppedAudioChunks, setDroppedAudioChunks] = useState(0);
  const [fallbackFrames, setFallbackFrames] = useState(0);
  const [useHttpFallback, setUseHttpFallback] = useState(false);
  const [gatewayStatus, setGatewayStatus] = useState<RealtimeGatewayStatus>(
    unavailableRealtimeGatewayStatus,
  );
  const [gatewayChecking, setGatewayChecking] = useState(true);
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeVoiceStatus>({
    state: "idle",
    connectionId: null,
    playbackGeneration: 0,
  });
  const [frameChannelAvailable, setFrameChannelAvailable] = useState(false);
  const [remoteAudioAvailable, setRemoteAudioAvailable] = useState(false);
  const [speakerOutputActive, setSpeakerOutputActive] = useState(false);
  const [modelSpeaking, setModelSpeaking] = useState(false);
  const [vadEngine, setVadEngine] = useState<VadEngine>("off");
  const [captureError, setCaptureError] = useState("");
  const realtimeConnected = realtimeStatus.state === "connected";
  const [voiceEndpointDetector] = useState(() => new VoiceEndpointDetector());
  const [responseAudioDetector] = useState(() => new ResponseAudioDetector());

  const mountedRef = useRef(true);
  const speakingTimerRef = useRef<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const microphoneModeRef = useRef<MicrophoneMode>(null);
  const realtimeRef = useRef<RealtimeVoiceTransport | null>(null);
  const remoteAudioRef = useRef<HTMLAudioElement>(null);
  const remoteStreamRef = useRef<MediaStream | null>(null);
  const speakerContextRef = useRef<AudioContext | null>(null);
  const speakerSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const speakerGainRef = useRef<GainNode | null>(null);
  const speakerCompressorRef = useRef<DynamicsCompressorNode | null>(null);
  const resumeAudioTimerRef = useRef<number | null>(null);
  const silenceAckTimerRef = useRef<number | null>(null);
  const gatewayTransportReadyRef = useRef(false);
  const httpFallbackRef = useRef(false);
  const fallbackTransportRef = useRef(transport);
  const cameraGenerationRef = useRef(0);
  const microphoneGenerationRef = useRef(0);
  const microphoneActiveRef = useRef(false);
  const lastAcceptedPixelsRef = useRef<Uint8ClampedArray | null>(null);
  const lastAcceptedAtRef = useRef<number | null>(null);
  const lastAcceptedSequenceRef = useRef(-1);
  const frameSequenceRef = useRef(0);
  const pendingFramesRef = useRef(new Map<string, PendingFrame>());
  const retryNotBeforeRef = useRef(0);
  const frameSendingRef = useRef(false);
  const frameAbortRef = useRef<AbortController | null>(null);
  const audioAbortRef = useRef<AbortController | null>(null);
  const audioSequenceRef = useRef(0);
  const audioInFlightRef = useRef(false);
  const pendingAudioRef = useRef<QueuedAudio | null>(null);
  const microphoneLevelMonitorRef = useRef<LevelMonitor | null>(null);
  const sileroVadRef = useRef<SileroMicVAD | null>(null);
  const remoteLevelMonitorRef = useRef<LevelMonitor | null>(null);
  const turnSequenceRef = useRef(0);
  const pendingResponseTurnRef = useRef<string | null>(null);

  useEffect(() => {
    httpFallbackRef.current = useHttpFallback;
  }, [useHttpFallback]);

  useEffect(() => {
    fallbackTransportRef.current = transport;
  }, [transport]);

  useEffect(() => {
    gatewayTransportReadyRef.current = gatewayStatus.transportReady;
  }, [gatewayStatus.transportReady]);

  useEffect(() => {
    lastAcceptedPixelsRef.current = null;
    lastAcceptedAtRef.current = null;
    pendingFramesRef.current.clear();
  }, [localSessionId]);

  const acceptFrame = useCallback((frameId: string, _timestampMs: number, sequence: number) => {
    const pending = pendingFramesRef.current.get(frameId);
    if (!pending || pending.sequence !== sequence || sequence <= lastAcceptedSequenceRef.current) {
      return;
    }
    lastAcceptedSequenceRef.current = sequence;
    lastAcceptedPixelsRef.current = pending.pixels;
    lastAcceptedAtRef.current = Date.now();
    for (const [pendingId, candidate] of pendingFramesRef.current) {
      if (candidate.sequence <= sequence) pendingFramesRef.current.delete(pendingId);
    }
    if (mountedRef.current) {
      setSentFrames((count) => count + 1);
      setCameraState("live");
      setCaptureError("");
    }
  }, []);

  useEffect(() => {
    if (gatewayStatus.transportReady && cameraStreamRef.current) {
      realtimeRef.current?.connect();
    }
  }, [gatewayStatus.transportReady]);

  const disconnectRemoteSpeaker = useCallback(() => {
    speakerSourceRef.current?.disconnect();
    speakerGainRef.current?.disconnect();
    speakerCompressorRef.current?.disconnect();
    speakerSourceRef.current = null;
    speakerGainRef.current = null;
    speakerCompressorRef.current = null;
    if (mountedRef.current) setSpeakerOutputActive(false);
  }, []);

  const connectRemoteSpeaker = useCallback(async (stream: MediaStream) => {
    disconnectRemoteSpeaker();
    if (typeof AudioContext === "undefined") {
      throw new Error("Web Audio speaker playback is not supported in this browser.");
    }
    let context = speakerContextRef.current;
    if (!context || context.state === "closed") {
      context = new AudioContext({ latencyHint: "interactive" });
      speakerContextRef.current = context;
    }
    if (context.state === "suspended") await context.resume();
    const source = context.createMediaStreamSource(stream);
    const gain = context.createGain();
    const compressor = context.createDynamicsCompressor();
    gain.gain.value = 2;
    compressor.threshold.value = -12;
    compressor.knee.value = 12;
    compressor.ratio.value = 6;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.12;
    source.connect(gain).connect(compressor).connect(context.destination);
    speakerSourceRef.current = source;
    speakerGainRef.current = gain;
    speakerCompressorRef.current = compressor;
    if (mountedRef.current) setSpeakerOutputActive(context.state === "running");
  }, [disconnectRemoteSpeaker]);

  const prepareRemoteSpeaker = useCallback(async () => {
    if (typeof AudioContext === "undefined") return;
    let context = speakerContextRef.current;
    if (!context || context.state === "closed") {
      context = new AudioContext({ latencyHint: "interactive" });
      speakerContextRef.current = context;
    }
    if (context.state === "suspended") await context.resume();
    const stream = remoteStreamRef.current;
    if (stream && !speakerSourceRef.current) await connectRemoteSpeaker(stream);
  }, [connectRemoteSpeaker]);

  const receiveRemoteAudio = useCallback((stream: MediaStream | null) => {
    remoteStreamRef.current = stream;
    const audio = remoteAudioRef.current;
    if (audio) {
      audio.srcObject = stream;
      // Web Audio is the primary output path. Keeping the element muted avoids
      // double playback on browsers where both routes work.
      audio.muted = true;
      if (!stream) audio.pause();
    }
    if (stream) {
      void connectRemoteSpeaker(stream).catch(() => {
        if (mountedRef.current) {
          setCaptureError("Tap Voice again to allow speaker playback.");
        }
      });
    } else {
      disconnectRemoteSpeaker();
    }
    if (mountedRef.current) setRemoteAudioAvailable(stream !== null);
  }, [connectRemoteSpeaker, disconnectRemoteSpeaker]);

  const interruptOutput = useCallback(() => {
    const realtime = realtimeRef.current;
    if (!realtime || !realtimeConnected) return;
    const generation = realtime.interrupt();
    const audio = remoteAudioRef.current;
    if (audio) {
      audio.muted = true;
      audio.pause();
    }
    if (speakerGainRef.current) speakerGainRef.current.gain.value = 0;
    // Muting and pausing are synchronous browser playback operations. Send the
    // acknowledgement only after that state has been applied locally.
    if (silenceAckTimerRef.current !== null) {
      window.clearTimeout(silenceAckTimerRef.current);
      silenceAckTimerRef.current = null;
    }
    let acknowledgementAttempts = 0;
    const acknowledgeSilence = () => {
      silenceAckTimerRef.current = null;
      if (realtime.acknowledgePlaybackSilent(generation)) return;
      acknowledgementAttempts += 1;
      if (acknowledgementAttempts < 4) {
        silenceAckTimerRef.current = window.setTimeout(acknowledgeSilence, 20);
      }
    };
    queueMicrotask(acknowledgeSilence);
    if (resumeAudioTimerRef.current !== null) {
      window.clearTimeout(resumeAudioTimerRef.current);
    }
    resumeAudioTimerRef.current = window.setTimeout(() => {
      resumeAudioTimerRef.current = null;
      const current = remoteAudioRef.current;
      if (!current || !remoteStreamRef.current) return;
      if (speakerGainRef.current) speakerGainRef.current.gain.value = 2;
      void prepareRemoteSpeaker().catch(() => undefined);
    }, 120);
  }, [prepareRemoteSpeaker, realtimeConnected]);

  useEffect(() => {
    microphoneLevelMonitorRef.current?.stop();
    microphoneLevelMonitorRef.current = null;
    const previousSilero = sileroVadRef.current;
    sileroVadRef.current = null;
    if (previousSilero) void previousSilero.destroy().catch(() => undefined);
    voiceEndpointDetector.reset();
    if (!microphoneActive || microphoneMode !== "webrtc") return;
    const stream = microphoneStreamRef.current;
    if (!stream) return;

    let cancelled = false;
    let fallbackMonitor: LevelMonitor | null = null;

    const finishTurn = () => {
      if (cancelled || pendingResponseTurnRef.current !== null) return;
      const turnId = `turn-${++turnSequenceRef.current}-${Date.now()}`;
      if (realtimeRef.current?.markTurnEnded(turnId)) {
        pendingResponseTurnRef.current = turnId;
        responseAudioDetector.arm(turnId);
      }
    };

    const startLevelFallback = () => {
      if (cancelled) return;
      setVadEngine("level-fallback");
      fallbackMonitor = monitorStreamLevel(stream, (level, nowMs) => {
        const event = voiceEndpointDetector.observe(level, nowMs);
        if (event === "speechStarted") interruptOutput();
        else if (event === "turnEnded") finishTurn();
      });
      microphoneLevelMonitorRef.current = fallbackMonitor;
      void fallbackMonitor?.resume().catch(() => undefined);
    };

    queueMicrotask(() => {
      if (!cancelled && mountedRef.current) setVadEngine("loading");
    });
    void import("@ricky0123/vad-web")
      .then(async ({ MicVAD }) => {
        const silero = await MicVAD.new({
          model: "v5",
          baseAssetPath: "/vad/",
          onnxWASMBasePath: "/vad/",
          getStream: async () => stream,
          pauseStream: async () => undefined,
          resumeStream: async () => stream,
          startOnLoad: false,
          positiveSpeechThreshold: 0.5,
          negativeSpeechThreshold: 0.35,
          minSpeechMs: 128,
          preSpeechPadMs: 250,
          redemptionMs: 320,
          onSpeechRealStart: () => {
            if (!cancelled) interruptOutput();
          },
          onSpeechEnd: () => finishTurn(),
        });
        if (cancelled) {
          await silero.destroy();
          return;
        }
        sileroVadRef.current = silero;
        await silero.start();
        if (!cancelled) setVadEngine("silero");
      })
      .catch(() => startLevelFallback());

    return () => {
      cancelled = true;
      if (microphoneLevelMonitorRef.current === fallbackMonitor) {
        microphoneLevelMonitorRef.current = null;
      }
      fallbackMonitor?.stop();
      const silero = sileroVadRef.current;
      sileroVadRef.current = null;
      if (silero) void silero.destroy().catch(() => undefined);
      voiceEndpointDetector.reset();
    };
  }, [
    interruptOutput,
    microphoneActive,
    microphoneMode,
    responseAudioDetector,
    voiceEndpointDetector,
  ]);

  useEffect(() => {
    remoteLevelMonitorRef.current?.stop();
    remoteLevelMonitorRef.current = null;
    responseAudioDetector.reset();
    const pendingTurn = pendingResponseTurnRef.current;
    if (pendingTurn) responseAudioDetector.arm(pendingTurn);
    if (!remoteAudioAvailable) return;
    const stream = remoteStreamRef.current;
    if (!stream) return;
    const monitor = monitorStreamLevel(stream, (level) => {
      // Playback uses the Web Audio graph. The fallback <audio> element stays
      // muted to prevent duplicate sound, so it is not a playback indicator.
      const renderedLevel = speakerOutputActive ? level : 0;
      if (renderedLevel > 0.008) {
        if (mountedRef.current) setModelSpeaking(true);
        if (speakingTimerRef.current !== null) window.clearTimeout(speakingTimerRef.current);
        speakingTimerRef.current = window.setTimeout(() => {
          speakingTimerRef.current = null;
          if (mountedRef.current) setModelSpeaking(false);
        }, 180);
      }
      const turnId = responseAudioDetector.observe(renderedLevel);
      if (!turnId) return;
      if (realtimeRef.current?.acknowledgeFirstResponseAudio(turnId)) {
        if (pendingResponseTurnRef.current === turnId) {
          pendingResponseTurnRef.current = null;
        }
      } else {
        responseAudioDetector.arm(turnId);
      }
    });
    remoteLevelMonitorRef.current = monitor;
    void monitor?.resume().catch(() => undefined);
    return () => {
      if (remoteLevelMonitorRef.current === monitor) remoteLevelMonitorRef.current = null;
      monitor?.stop();
      if (speakingTimerRef.current !== null) {
        window.clearTimeout(speakingTimerRef.current);
        speakingTimerRef.current = null;
      }
      setModelSpeaking(false);
      responseAudioDetector.reset();
    };
  }, [remoteAudioAvailable, responseAudioDetector, speakerOutputActive]);

  const sendFrameFallback = useCallback((frame: SelectedFramePacket, reason: string) => {
    if (!httpFallbackRef.current) {
      if (mountedRef.current) setCaptureError(reason);
      return;
    }
    const controller = new AbortController();
    frameAbortRef.current?.abort();
    frameAbortRef.current = controller;
    void fallbackTransportRef.current.sendVideoFrame({
      blob: new Blob([frame.jpeg], { type: "image/jpeg" }),
      timestampMs: frame.timestampMs,
      differencePercent: frame.differencePercent,
      keyframe: frame.keyframe,
      sessionId: localSessionId,
      signal: controller.signal,
    }).then(() => {
      if (!mountedRef.current || controller.signal.aborted) return;
      setFallbackFrames((count) => count + 1);
      setCameraState("waiting");
      setCaptureError("");
    }).catch((fallbackReason: unknown) => {
      if (!mountedRef.current || controller.signal.aborted) return;
      setCameraState("waiting");
      setCaptureError(captureErrorMessage(fallbackReason, "The HTTP fallback is unavailable."));
    }).finally(() => {
      if (frameAbortRef.current === controller) frameAbortRef.current = null;
    });
  }, [localSessionId]);

  useEffect(() => {
    const realtime = new RealtimeVoiceTransport({
      admit: () => requestRealtimeAdmission(localSessionId),
      onStatus: (status) => {
        if (!mountedRef.current) return;
        setRealtimeStatus(status);
        if (status.state !== "connected") {
          pendingResponseTurnRef.current = null;
          responseAudioDetector.reset();
        }
        if (microphoneModeRef.current === "webrtc") {
          if (status.state === "connected") {
            setMicrophoneState("live");
            setCaptureError("");
          } else if (["failed", "reconnecting"].includes(status.state)) {
            setMicrophoneState("waiting");
          }
        }
      },
      onRemoteAudio: receiveRemoteAudio,
      onFrameAccepted: acceptFrame,
      onSelectedFrameFallback: sendFrameFallback,
      onSelectedFrameChannelState: (available) => {
        if (mountedRef.current) setFrameChannelAvailable(available);
      },
      onError: (message) => {
        if (mountedRef.current) setCaptureError(message);
      },
    });
    realtimeRef.current = realtime;
    setRealtimeStatus(realtime.status);
    if (gatewayTransportReadyRef.current && cameraStreamRef.current) realtime.connect();
    return () => {
      if (realtimeRef.current === realtime) realtimeRef.current = null;
      realtime.disconnect();
      if (microphoneModeRef.current === "webrtc") {
        microphoneModeRef.current = null;
        microphoneActiveRef.current = false;
        microphoneStreamRef.current = null;
        if (mountedRef.current) {
          setMicrophoneActive(false);
          setMicrophoneMode(null);
          setMicrophoneState("off");
        }
      }
    };
  }, [
    acceptFrame,
    localSessionId,
    receiveRemoteAudio,
    responseAudioDetector,
    sendFrameFallback,
  ]);

  useEffect(() => {
    let active = true;
    let requestController: AbortController | null = null;

    async function refreshGateway() {
      requestController?.abort();
      const controller = new AbortController();
      requestController = controller;
      try {
        const status = await fetchRealtimeGatewayStatus(controller.signal);
        if (active) setGatewayStatus(status);
      } catch (reason) {
        if (
          active &&
          !(reason instanceof DOMException && reason.name === "AbortError")
        ) {
          setGatewayStatus(unavailableRealtimeGatewayStatus);
        }
      } finally {
        if (active) setGatewayChecking(false);
      }
    }

    void refreshGateway();
    const interval = window.setInterval(() => void refreshGateway(), 2_000);
    return () => {
      active = false;
      requestController?.abort();
      window.clearInterval(interval);
    };
  }, []);

  const stopCamera = useCallback(() => {
    cameraGenerationRef.current += 1;
    frameAbortRef.current?.abort();
    frameAbortRef.current = null;
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    frameSendingRef.current = false;
    lastAcceptedPixelsRef.current = null;
    lastAcceptedAtRef.current = null;
    pendingFramesRef.current.clear();
    retryNotBeforeRef.current = 0;
    if (!microphoneActiveRef.current) realtimeRef.current?.disconnect();
    if (mountedRef.current) {
      setCameraStream(null);
      setCameraState("off");
      setCurrentDifference(0);
    }
  }, []);

  const stopMicrophone = useCallback(() => {
    microphoneGenerationRef.current += 1;
    microphoneActiveRef.current = false;
    const mode = microphoneModeRef.current;
    microphoneModeRef.current = null;
    audioAbortRef.current?.abort();
    audioAbortRef.current = null;
    pendingAudioRef.current = null;
    microphoneLevelMonitorRef.current?.stop();
    microphoneLevelMonitorRef.current = null;
    voiceEndpointDetector.reset();
    pendingResponseTurnRef.current = null;
    responseAudioDetector.reset();
    disconnectRemoteSpeaker();
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
    mediaRecorderRef.current = null;
    if (mode === "webrtc") {
      void realtimeRef.current?.setMicrophoneTrack(null).catch(() => undefined);
    } else {
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());
    }
    microphoneStreamRef.current = null;
    if (!cameraStreamRef.current) realtimeRef.current?.disconnect();
    if (mountedRef.current) {
      setMicrophoneActive(false);
      setMicrophoneMode(null);
      setMicrophoneState("off");
    }
  }, [disconnectRemoteSpeaker, responseAudioDetector, voiceEndpointDetector]);

  useEffect(() => {
    mountedRef.current = true;
    const remoteAudio = remoteAudioRef.current;
    return () => {
      mountedRef.current = false;
      cameraGenerationRef.current += 1;
      microphoneGenerationRef.current += 1;
      microphoneActiveRef.current = false;
      frameAbortRef.current?.abort();
      audioAbortRef.current?.abort();
      if (resumeAudioTimerRef.current !== null) {
        window.clearTimeout(resumeAudioTimerRef.current);
      }
      if (silenceAckTimerRef.current !== null) {
        window.clearTimeout(silenceAckTimerRef.current);
      }
      cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== "inactive") recorder.stop();
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());
      pendingAudioRef.current = null;
      pendingResponseTurnRef.current = null;
      responseAudioDetector.reset();
      microphoneLevelMonitorRef.current?.stop();
      remoteLevelMonitorRef.current?.stop();
      microphoneLevelMonitorRef.current = null;
      remoteLevelMonitorRef.current = null;
      disconnectRemoteSpeaker();
      const speakerContext = speakerContextRef.current;
      speakerContextRef.current = null;
      if (speakerContext) void speakerContext.close().catch(() => undefined);
      remoteAudio?.pause();
      if (remoteAudio) remoteAudio.srcObject = null;
    };
  }, [disconnectRemoteSpeaker, responseAudioDetector]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !cameraStream) return;
    video.srcObject = cameraStream;
    void video.play().catch(() => {
      if (mountedRef.current) {
        setCameraState("error");
        setCaptureError("The camera preview could not start.");
      }
    });
  }, [cameraStream]);

  useEffect(() => {
    if (!cameraStream || !localSessionId) return;
    const generation = cameraGenerationRef.current;
    let cancelled = false;

    async function sampleAndMaybeSend() {
      const video = videoRef.current;
      const analysisCanvas = analysisCanvasRef.current;
      const outputCanvas = outputCanvasRef.current;
      if (
        cancelled ||
        generation !== cameraGenerationRef.current ||
        !video ||
        !analysisCanvas ||
        !outputCanvas ||
        video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        document.visibilityState === "hidden"
      ) {
        return;
      }
      if (frameSendingRef.current || Date.now() < retryNotBeforeRef.current) {
        setSkippedFrames((count) => count + 1);
        return;
      }

      const videoWidth = Math.max(1, video.videoWidth);
      const videoHeight = Math.max(1, video.videoHeight);
      const scale = Math.min(1, MAX_FRAME_WIDTH / videoWidth, MAX_FRAME_HEIGHT / videoHeight);
      outputCanvas.width = Math.max(1, Math.round(videoWidth * scale));
      outputCanvas.height = Math.max(1, Math.round(videoHeight * scale));
      const outputContext = outputCanvas.getContext("2d");
      const analysisContext = analysisCanvas.getContext("2d", { willReadFrequently: true });
      if (!outputContext || !analysisContext) {
        setCameraState("error");
        setCaptureError("This browser cannot prepare camera frames.");
        return;
      }

      // Compare the same canvas snapshot that will be JPEG-encoded below.
      outputContext.drawImage(video, 0, 0, outputCanvas.width, outputCanvas.height);
      analysisContext.drawImage(outputCanvas, 0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
      const currentPixels = new Uint8ClampedArray(
        analysisContext.getImageData(0, 0, ANALYSIS_WIDTH, ANALYSIS_HEIGHT).data,
      );
      const previousPixels = lastAcceptedPixelsRef.current;
      const differencePercent = previousPixels
        ? normalizedFrameDifferencePercent(currentPixels, previousPixels)
        : 100;
      setCurrentDifference(differencePercent);

      const capturedAtMs = Date.now();
      const decision = decideFrameTransmission({
        differencePercent,
        thresholdPercent,
        nowMs: capturedAtMs,
        lastSentAtMs: lastAcceptedAtRef.current,
        heartbeatMs: heartbeatSeconds * 1_000,
        hasLastSentFrame: previousPixels !== null,
      });
      if (!decision.send) {
        setSkippedFrames((count) => count + 1);
        return;
      }

      const realtimeFrameAvailable =
        gatewayStatus.transportReady && gatewayStatus.modelConnected;
      if (!realtimeFrameAvailable && !useHttpFallback) {
        setCameraState("waiting");
        setCaptureError(
          gatewayStatus.transportReady
            ? "The model is disconnected, so camera frames stay local."
            : "Realtime transport is offline. HTTP fallback is off.",
        );
        setSkippedFrames((count) => count + 1);
        return;
      }

      frameSendingRef.current = true;
      const controller = new AbortController();
      frameAbortRef.current = controller;
      let offeredFrameId: string | null = null;
      try {
        const blob = await frameBlob(outputCanvas);
        if (cancelled || generation !== cameraGenerationRef.current) return;
        const realtime = realtimeRef.current;
        if (realtimeFrameAvailable && realtime) {
          const sequence = frameSequenceRef.current++;
          const frameId = `frame-${sequence}-${capturedAtMs}`;
          offeredFrameId = frameId;
          const jpeg = await blob.arrayBuffer();
          if (cancelled || generation !== cameraGenerationRef.current) return;
          pendingFramesRef.current.set(frameId, {
            sequence,
            pixels: currentPixels,
            capturedAtMs,
          });
          const offered = realtime.offerSelectedFrame({
            frameId,
            sequence,
            timestampMs: capturedAtMs,
            differencePercent,
            keyframe: decision.keyframe,
            jpeg,
          });
          offeredFrameId = null;
          if (offered.replacedFrameId) pendingFramesRef.current.delete(offered.replacedFrameId);
          while (pendingFramesRef.current.size > 64) {
            const oldest = pendingFramesRef.current.keys().next().value;
            if (typeof oldest !== "string") break;
            pendingFramesRef.current.delete(oldest);
          }
          retryNotBeforeRef.current = 0;
          setCameraState(frameChannelAvailable ? "capturing" : "waiting");
          setCaptureError("");
        } else {
          await transport.sendVideoFrame({
            blob,
            timestampMs: capturedAtMs,
            differencePercent,
            keyframe: decision.keyframe,
            sessionId: localSessionId,
            signal: controller.signal,
          });
          if (cancelled || generation !== cameraGenerationRef.current) return;
          // HTTP has no model-consumption acknowledgement, so it never moves
          // the frame-difference baseline.
          retryNotBeforeRef.current = Date.now() + Math.max(1_000, heartbeatSeconds * 1_000);
          setFallbackFrames((count) => count + 1);
          setCameraState("waiting");
          setCaptureError("");
        }
      } catch (reason) {
        if (offeredFrameId) pendingFramesRef.current.delete(offeredFrameId);
        if (cancelled || generation !== cameraGenerationRef.current) return;
        const message = captureErrorMessage(reason, "The frame was not accepted.");
        if (!message) return;
        retryNotBeforeRef.current = Date.now() + Math.max(1_000, heartbeatSeconds * 1_000);
        setCameraState(
          reason instanceof MediaTransportError && [501, 503].includes(reason.status)
            ? "waiting"
            : "error",
        );
        setCaptureError(message);
      } finally {
        if (frameAbortRef.current === controller) frameAbortRef.current = null;
        frameSendingRef.current = false;
      }
    }

    const interval = window.setInterval(
      () => void sampleAndMaybeSend(),
      Math.round(1_000 / Math.max(1, maxFramesPerSecond)),
    );
    void sampleAndMaybeSend();
    return () => {
      cancelled = true;
      window.clearInterval(interval);
      frameAbortRef.current?.abort();
    };
  }, [
    cameraStream,
    frameChannelAvailable,
    gatewayStatus.modelConnected,
    gatewayStatus.transportReady,
    heartbeatSeconds,
    localSessionId,
    maxFramesPerSecond,
    thresholdPercent,
    transport,
    useHttpFallback,
  ]);

  const pumpAudio = useCallback(
    async (initial: QueuedAudio) => {
      if (audioInFlightRef.current) return;
      audioInFlightRef.current = true;
      let next: QueuedAudio | null = initial;
      try {
        while (next !== null) {
          const chunk: QueuedAudio = next;
          next = null;
          if (
            mountedRef.current &&
            microphoneActiveRef.current &&
            chunk.generation === microphoneGenerationRef.current
          ) {
            const controller = new AbortController();
            audioAbortRef.current = controller;
            try {
              await transport.sendAudioChunk({
                blob: chunk.blob,
                timestampMs: chunk.timestampMs,
                sequence: chunk.sequence,
                mimeType: chunk.mimeType,
                sessionId: localSessionId,
                signal: controller.signal,
              });
              if (
                mountedRef.current &&
                microphoneActiveRef.current &&
                chunk.generation === microphoneGenerationRef.current
              ) {
                setSentAudioChunks((count) => count + 1);
                setMicrophoneState("live");
                setCaptureError("");
              }
            } catch (reason) {
              if (
                mountedRef.current &&
                microphoneActiveRef.current &&
                chunk.generation === microphoneGenerationRef.current
              ) {
                const message = captureErrorMessage(reason, "The voice chunk was not accepted.");
                if (message) {
                  setMicrophoneState(
                    reason instanceof MediaTransportError && [501, 503].includes(reason.status)
                      ? "waiting"
                      : "error",
                  );
                  setCaptureError(message);
                }
              }
            } finally {
              if (audioAbortRef.current === controller) audioAbortRef.current = null;
            }
          }
          next = pendingAudioRef.current;
          pendingAudioRef.current = null;
        }
      } finally {
        audioInFlightRef.current = false;
      }
    },
    [localSessionId, transport],
  );

  const queueAudio = useCallback(
    (chunk: QueuedAudio) => {
      if (audioInFlightRef.current) {
        if (pendingAudioRef.current !== null) setDroppedAudioChunks((count) => count + 1);
        pendingAudioRef.current = chunk;
        return;
      }
      void pumpAudio(chunk);
    },
    [pumpAudio],
  );

  async function startCamera() {
    if (!localSessionId || !navigator.mediaDevices?.getUserMedia) {
      setCameraState("error");
      setCaptureError("Camera capture is not supported in this browser.");
      return;
    }
    const generation = ++cameraGenerationRef.current;
    setCameraState("requesting");
    setCaptureError("");
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      if (!mountedRef.current || generation !== cameraGenerationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      cameraStreamRef.current = stream;
      lastAcceptedPixelsRef.current = null;
      lastAcceptedAtRef.current = null;
      retryNotBeforeRef.current = 0;
      setCurrentDifference(0);
      setSentFrames(0);
      setFallbackFrames(0);
      setSkippedFrames(0);
      pendingFramesRef.current.clear();
      setCameraStream(stream);
      setCameraState("capturing");
      if (gatewayStatus.transportReady) realtimeRef.current?.connect();
    } catch {
      stream?.getTracks().forEach((track) => track.stop());
      if (mountedRef.current && generation === cameraGenerationRef.current) {
        setCameraState("error");
        setCaptureError("Camera permission was not granted.");
      }
    }
  }

  async function startMicrophone() {
    if (!localSessionId || !navigator.mediaDevices?.getUserMedia) {
      setMicrophoneState("error");
      setCaptureError("Voice capture is not supported in this browser.");
      return;
    }
    if (!gatewayStatus.transportReady && !useHttpFallback) {
      setMicrophoneState("waiting");
      setCaptureError("Realtime transport is offline. Enable HTTP fallback to send recorded chunks.");
      return;
    }
    if (!gatewayStatus.transportReady && typeof MediaRecorder === "undefined") {
      setMicrophoneState("error");
      setCaptureError("HTTP voice fallback is not supported in this browser.");
      return;
    }
    const generation = ++microphoneGenerationRef.current;
    setMicrophoneState("requesting");
    setCaptureError("");
    let stream: MediaStream | null = null;
    try {
      // Create/resume the speaker graph inside the same user gesture that
      // starts Voice; browsers otherwise may silently suspend remote output.
      await prepareRemoteSpeaker();
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      if (!mountedRef.current || generation !== microphoneGenerationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      if (gatewayStatus.transportReady) {
        const realtime = realtimeRef.current;
        const track = stream.getAudioTracks()[0];
        if (!realtime || !track) throw new Error("Realtime voice could not start.");
        microphoneStreamRef.current = stream;
        microphoneModeRef.current = "webrtc";
        microphoneActiveRef.current = true;
        setSentAudioChunks(0);
        setDroppedAudioChunks(0);
        setMicrophoneActive(true);
        setMicrophoneMode("webrtc");
        setMicrophoneState("capturing");
        setCaptureError("");
        realtime.connect(track);
        void remoteLevelMonitorRef.current?.resume().catch(() => undefined);
        return;
      }

      const mimeType = preferredAudioMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      microphoneStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      microphoneModeRef.current = "http";
      microphoneActiveRef.current = true;
      audioSequenceRef.current = 0;
      pendingAudioRef.current = null;
      setSentAudioChunks(0);
      setDroppedAudioChunks(0);

      recorder.addEventListener("dataavailable", (event) => {
        if (
          event.data.size === 0 ||
          !microphoneActiveRef.current ||
          generation !== microphoneGenerationRef.current
        ) {
          return;
        }
        queueAudio({
          blob: event.data,
          timestampMs: Date.now(),
          sequence: audioSequenceRef.current++,
          mimeType: event.data.type || recorder.mimeType || "audio/webm",
          generation,
        });
      });
      recorder.addEventListener("error", () => {
        if (!mountedRef.current || generation !== microphoneGenerationRef.current) return;
        stopMicrophone();
        setMicrophoneState("error");
        setCaptureError("Voice recording stopped unexpectedly.");
      });
      recorder.start(AUDIO_TIMESLICE_MS);
      setMicrophoneActive(true);
      setMicrophoneMode("http");
      setMicrophoneState("capturing");
    } catch (reason) {
      stream?.getTracks().forEach((track) => track.stop());
      if (mountedRef.current && generation === microphoneGenerationRef.current) {
        microphoneStreamRef.current = null;
        mediaRecorderRef.current = null;
        microphoneActiveRef.current = false;
        microphoneModeRef.current = null;
        setMicrophoneActive(false);
        setMicrophoneMode(null);
        setMicrophoneState("error");
        setCaptureError(
          reason instanceof DOMException && reason.name === "NotAllowedError"
            ? "Microphone permission was not granted."
            : captureErrorMessage(reason, "Realtime voice could not start."),
        );
      }
    }
  }

  const cameraActive = cameraStream !== null;
  const cameraRequesting = cameraState === "requesting";
  const microphoneRequesting = microphoneState === "requesting";
  const conversationActive =
    realtimeConnected && microphoneActive && gatewayStatus.conversationReady;
  const untrainedPlaybackActive =
    realtimeConnected &&
    microphoneActive &&
    gatewayStatus.untrainedAudioEnabled;
  const transportLabel = gatewayChecking
    ? "checking"
    : realtimeConnected
      ? "connected"
      : gatewayStatus.transportReady
        ? "online"
        : "offline";
  const modelLabel = gatewayStatus.modelConnected
    ? "connected"
    : gatewayStatus.checkpointLoaded
      ? "loaded"
      : "not loaded";
  const voiceLabel = gatewayStatus.voiceTrained ? "trained" : "not trained";
  const visionLabel = gatewayStatus.visionTrained ? "trained" : "not trained";
  const readinessLabel = gatewayStatus.conversationReady ? "ready" : "not ready";
  const untrainedParameterLabel = gatewayStatus.untrainedParameters > 0
    ? `${Math.round(gatewayStatus.untrainedParameters / 100_000) / 10}M parameters`
    : "random parameters";
  const mediaSummary = untrainedPlaybackActive
    ? "Untrained sound active · random weights, not speech"
    : conversationActive
      ? "Two-way voice session active"
    : !gatewayStatus.transportReady
      ? "Realtime transport offline"
      : gatewayStatus.untrainedAudioEnabled
        ? "Transport online · random model sound enabled"
      : !gatewayStatus.checkpointLoaded
        ? "Transport online · native checkpoint not loaded"
        : !gatewayStatus.modelConnected
          ? "Native checkpoint loaded · start Voice"
        : !gatewayStatus.voiceTrained
          ? "Model connected · voice weights not trained"
          : !gatewayStatus.visionTrained
            ? "Voice trained · vision weights not trained"
            : "Model trained · realtime latency gates pending";

  const companionState: CompanionState = captureError
    ? "error"
    : microphoneRequesting
      ? "connecting"
      : modelSpeaking
        ? "speaking"
        : microphoneActive
          ? "listening"
          : "idle";

  useEffect(() => {
    onCompanionState?.(companionState);
  }, [companionState, onCompanionState]);

  if (minimal) {
    return (
      <section className={styles.minimalMedia} aria-label="Voice and camera">
        <button
          className={styles.voiceOrb}
          type="button"
          aria-label={microphoneActive ? "Stop voice" : "Start voice"}
          aria-pressed={microphoneActive}
          data-active={microphoneActive || undefined}
          disabled={!localSessionId || microphoneRequesting}
          onClick={() => (microphoneActive ? stopMicrophone() : void startMicrophone())}
        >
          <AudioLines aria-hidden="true" />
        </button>
        <button
          className={styles.cameraOrb}
          type="button"
          aria-label={cameraActive ? "Stop camera" : "Start camera"}
          aria-pressed={cameraActive}
          data-active={cameraActive || undefined}
          disabled={!localSessionId || cameraRequesting}
          onClick={() => (cameraActive ? stopCamera() : void startCamera())}
        >
          <ScanEye aria-hidden="true" />
        </button>
        <button
          className={styles.commandOrb}
          type="button"
          aria-label="Open text input, Command B or Control B"
          onClick={onOpenTextInput}
        >
          <SquareTerminal aria-hidden="true" />
        </button>
        {cameraActive && (
          <video ref={videoRef} className={styles.hiddenMedia} muted playsInline aria-label="Camera preview" />
        )}
        <canvas ref={analysisCanvasRef} className={styles.hiddenMedia} width={ANALYSIS_WIDTH} height={ANALYSIS_HEIGHT} aria-hidden="true" />
        <canvas ref={outputCanvasRef} className={styles.hiddenMedia} aria-hidden="true" />
        <audio ref={remoteAudioRef} className={styles.hiddenMedia} autoPlay playsInline aria-label="tinydog voice response" />
      </section>
    );
  }

  return (
    <section className={styles.mediaPanel} aria-label="Realtime voice and video capture">
      <div className={styles.mediaToolbar}>
        <div className={styles.mediaActions}>
          <button
            className={`${styles.mediaButton} ${cameraActive ? styles.mediaButtonActive : ""}`}
            type="button"
            aria-pressed={cameraActive}
            disabled={!localSessionId || cameraRequesting}
            onClick={() => (cameraActive ? stopCamera() : void startCamera())}
          >
            <span className={styles.mediaIcon} aria-hidden="true">◉</span>
            {cameraRequesting ? "Camera…" : cameraActive ? "Stop camera" : "Camera"}
          </button>
          <button
            className={`${styles.mediaButton} ${microphoneActive ? styles.mediaButtonActive : ""}`}
            type="button"
            aria-pressed={microphoneActive}
            disabled={!localSessionId || microphoneRequesting}
            onClick={() => (microphoneActive ? stopMicrophone() : void startMicrophone())}
          >
            <span className={styles.mediaIcon} aria-hidden="true">●</span>
            {microphoneRequesting ? "Voice…" : microphoneActive ? "Stop voice" : "Voice"}
          </button>
          <button
            className={styles.mediaButton}
            type="button"
            disabled={!realtimeConnected}
            onClick={interruptOutput}
          >
            <span className={styles.mediaIcon} aria-hidden="true">■</span>
            Interrupt
          </button>
        </div>

        <span className={styles.localOnly}>{mediaSummary}</span>

        <details className={styles.mediaSettings}>
          <summary>Settings</summary>
          <div className={styles.settingsGrid}>
            <label>
              Send at
              <output>{thresholdPercent}% difference</output>
              <input
                aria-label="Frame difference threshold"
                type="range"
                min="10"
                max="100"
                step="1"
                value={thresholdPercent}
                onChange={(event) =>
                  setThresholdPercent(clampFrameDifferenceThreshold(event.currentTarget.valueAsNumber))
                }
              />
            </label>
            <label>
              Maximum rate
              <output>{maxFramesPerSecond} frame{maxFramesPerSecond === 1 ? "" : "s"}/sec</output>
              <input
                aria-label="Maximum sampled frames per second"
                type="range"
                min="1"
                max="10"
                step="1"
                value={maxFramesPerSecond}
                onChange={(event) => setMaxFramesPerSecond(event.currentTarget.valueAsNumber)}
              />
            </label>
            <label>
              Keepalive frame
              <output>Every {heartbeatSeconds} sec</output>
              <input
                aria-label="Keepalive frame interval"
                type="range"
                min="1"
                max="30"
                step="1"
                value={heartbeatSeconds}
                onChange={(event) => setHeartbeatSeconds(event.currentTarget.valueAsNumber)}
              />
            </label>
            <label className={styles.fallbackSetting}>
              HTTP fallback
              <output>{useHttpFallback ? "On" : "Off"}</output>
              <input
                aria-label="Use slower HTTP media fallback"
                type="checkbox"
                checked={useHttpFallback}
                onChange={(event) => setUseHttpFallback(event.currentTarget.checked)}
              />
            </label>
            <p>
              Average RGB change on a 64 × 36 sample, compared with the last frame the model
              acknowledged. HTTP fallback is slower and never advances that baseline.
            </p>
          </div>
        </details>
      </div>

      {gatewayStatus.untrainedAudioEnabled && (
        <div className={styles.untrainedAudioNotice} role="status">
          <span>
            <strong>Random sound enabled</strong>
            <small>
              Actual {untrainedParameterLabel} native output. It is random, unintelligible,
              and not a trained voice.
            </small>
          </span>
        </div>
      )}

      <div className={styles.realtimeReadiness} aria-label="Realtime capability status">
        <span data-online={(gatewayStatus.transportReady || realtimeConnected) || undefined}>
          Transport · {transportLabel}
        </span>
        <span data-online={gatewayStatus.modelConnected || undefined}>
          Model · {modelLabel}
        </span>
        <span data-online={gatewayStatus.voiceTrained || undefined}>
          Trained voice · {voiceLabel}
        </span>
        <span data-online={gatewayStatus.visionTrained || undefined}>
          Trained vision · {visionLabel}
        </span>
        <span data-online={gatewayStatus.conversationReady || undefined}>
          Realtime · {readinessLabel}
        </span>
      </div>

      {(cameraActive || microphoneActive || captureError) && (
        <div className={styles.mediaActivity} aria-live="polite">
          {cameraActive && (
            <video
              ref={videoRef}
              className={styles.cameraPreview}
              muted
              playsInline
              aria-label="Camera preview"
            />
          )}
          <div className={styles.captureReadout}>
            {cameraActive && (
              <span>
                Camera · {readableCaptureState(cameraState)} · Δ {currentDifference.toFixed(1)}% · {sentFrames} model accepted · {fallbackFrames} fallback · {skippedFrames} skipped
              </span>
            )}
            {microphoneActive && (
              <span>
                Voice · {readableCaptureState(microphoneState)} · {microphoneMode === "webrtc"
                  ? `WebRTC · ${vadEngine === "silero"
                    ? "Silero VAD"
                    : vadEngine === "loading"
                      ? "loading VAD"
                      : vadEngine === "level-fallback"
                        ? "basic VAD fallback"
                        : "VAD off"}${speakerOutputActive
                    ? " · speaker output active"
                    : remoteAudioAvailable
                      ? " · speaker stream"
                      : ""}${
                    gatewayStatus.untrainedAudioEnabled
                      ? " · random sound enabled"
                      : ""
                  }`
                  : `${sentAudioChunks} chunks sent · ${droppedAudioChunks} stale dropped`}
              </span>
            )}
            {cameraActive && gatewayStatus.transportReady && (
              <span>Selected frames · {frameChannelAvailable ? "WebRTC channel open" : "channel opening"}</span>
            )}
            {captureError && <span className={styles.captureWarning}>{captureError}</span>}
          </div>
        </div>
      )}

      <canvas
        ref={analysisCanvasRef}
        className={styles.captureCanvas}
        width={ANALYSIS_WIDTH}
        height={ANALYSIS_HEIGHT}
        aria-hidden="true"
      />
      <canvas ref={outputCanvasRef} className={styles.captureCanvas} aria-hidden="true" />
      <audio
        ref={remoteAudioRef}
        className={styles.remoteAudio}
        autoPlay
        playsInline
        aria-label="MLM voice response"
      />
    </section>
  );
}
