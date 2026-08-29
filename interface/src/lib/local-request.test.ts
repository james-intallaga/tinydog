import assert from "node:assert/strict";
import test from "node:test";

import {
  LocalJsonBodyError,
  readBoundedJsonObject,
  validateLocalMutation,
  validateLoopbackRequest,
} from "./local-request";

const JSON_TYPES = new Set(["application/json"]);

test("local mutations require exact same-origin requests", () => {
  const evil = new Request("http://localhost:3000/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "text/plain",
      Origin: "https://evil.example",
    },
  });
  assert.equal(validateLocalMutation(evil, JSON_TYPES)?.status, 403);

  const missing = new Request("http://localhost:3000/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  assert.equal(validateLocalMutation(missing, JSON_TYPES)?.status, 403);

  const otherLoopbackName = new Request("http://localhost:3000/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Origin: "http://127.0.0.1:3000",
    },
  });
  assert.equal(validateLocalMutation(otherLoopbackName, JSON_TYPES)?.status, 403);
});

test("same-origin JSON with an optional charset is accepted", () => {
  const request = new Request("http://localhost:3000/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      Origin: "http://localhost:3000",
    },
  });
  assert.equal(validateLocalMutation(request, JSON_TYPES), null);
});

test("same-origin mutations still enforce the route media type", () => {
  const request = new Request("http://localhost:3000/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "text/plain",
      Origin: "http://localhost:3000",
    },
  });
  assert.equal(validateLocalMutation(request, JSON_TYPES)?.status, 415);
});

test("private read routes reject a non-loopback UI host", () => {
  const remote = new Request("http://192.0.2.5:3000/api/history?session_id=secret");
  assert.equal(validateLoopbackRequest(remote)?.status, 403);
  const local = new Request("http://127.0.0.1:3000/api/history?session_id=secret");
  assert.equal(validateLoopbackRequest(local), null);
});

test("bounded JSON accepts an object and rejects oversized input", async () => {
  const accepted = new Request("http://localhost:3000/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "hello" }),
  });
  assert.deepEqual(await readBoundedJsonObject(accepted, 64), { message: "hello" });

  const oversized = new Request("http://localhost:3000/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "x".repeat(100) }),
  });
  await assert.rejects(
    () => readBoundedJsonObject(oversized, 32),
    (reason: unknown) => reason instanceof LocalJsonBodyError && reason.status === 413,
  );
});
