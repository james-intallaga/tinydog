import assert from "node:assert/strict";
import test from "node:test";
import {
  parseRealtimeGatewayStatus,
  unavailableRealtimeGatewayStatus,
} from "./realtime-gateway";

test("realtime status fails closed when the gateway response is missing", () => {
  assert.deepEqual(parseRealtimeGatewayStatus(null), unavailableRealtimeGatewayStatus);
  assert.deepEqual(parseRealtimeGatewayStatus({ transportReady: true }), {
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
  });
});

test("conversation readiness requires transport, checkpoint, model, voice, and vision", () => {
  assert.equal(parseRealtimeGatewayStatus({
    transportReady: true,
    webrtcReady: true,
    checkpointLoaded: true,
    modelConnected: false,
    voiceTrained: true,
    visionTrained: true,
    conversationReady: true,
  }).conversationReady, false);
  assert.equal(parseRealtimeGatewayStatus({
    transportReady: true,
    webrtcReady: true,
    checkpointLoaded: true,
    modelConnected: true,
    voiceTrained: true,
    visionTrained: false,
    conversationReady: true,
  }).conversationReady, false);
  assert.equal(parseRealtimeGatewayStatus({
    transportReady: true,
    webrtcReady: true,
    checkpointLoaded: true,
    modelConnected: true,
    voiceTrained: true,
    visionTrained: true,
    conversationReady: true,
  }).conversationReady, true);
});

test("gateway counters are bounded to non-negative safe integers", () => {
  assert.equal(parseRealtimeGatewayStatus({ activeConnections: 3 }).activeConnections, 3);
  assert.equal(parseRealtimeGatewayStatus({ activeConnections: -1 }).activeConnections, 0);
  assert.equal(parseRealtimeGatewayStatus({ activeConnections: 1.5 }).activeConnections, 0);
});

test("random-initialized audio remains separate from trained conversation readiness", () => {
  const status = parseRealtimeGatewayStatus({
    transportReady: true,
    webrtcReady: true,
    checkpointLoaded: false,
    modelConnected: true,
    voiceTrained: false,
    visionTrained: false,
    conversationReady: true,
    randomInit: true,
    untrainedAudioEnabled: true,
    untrainedParameters: 31_004_864,
  });

  assert.equal(status.randomInit, true);
  assert.equal(status.untrainedAudioEnabled, true);
  assert.equal(status.untrainedParameters, 31_004_864);
  assert.equal(status.conversationReady, false);
});

test("invalid untrained parameter counts fail closed", () => {
  assert.equal(parseRealtimeGatewayStatus({ untrainedParameters: -1 }).untrainedParameters, 0);
  assert.equal(parseRealtimeGatewayStatus({ untrainedParameters: 1.5 }).untrainedParameters, 0);
  assert.equal(parseRealtimeGatewayStatus({ untrainedParameters: "31004864" }).untrainedParameters, 0);
});
