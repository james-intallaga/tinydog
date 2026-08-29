import {
  noStoreGatewayJson,
  proxyRealtimeGateway,
} from "../_gateway";

export const runtime = "nodejs";

const SAFE_SESSION = /^[A-Za-z0-9._:-]{1,128}$/;
const MAX_BODY_BYTES = 512;

export async function POST(request: Request) {
  if (request.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase() !== "application/json") {
    return noStoreGatewayJson({ error: "Realtime admission requires JSON." }, 415);
  }
  const declaredLength = Number(request.headers.get("content-length") ?? "0");
  if (!Number.isFinite(declaredLength) || declaredLength < 0 || declaredLength > MAX_BODY_BYTES) {
    return noStoreGatewayJson({ error: "Realtime admission request is invalid." }, 400);
  }

  let body: string;
  try {
    body = await request.text();
  } catch {
    return noStoreGatewayJson({ error: "Realtime admission request is invalid." }, 400);
  }
  if (new TextEncoder().encode(body).byteLength > MAX_BODY_BYTES) {
    return noStoreGatewayJson({ error: "Realtime admission request is too large." }, 413);
  }

  let value: unknown;
  try {
    value = JSON.parse(body);
  } catch {
    return noStoreGatewayJson({ error: "Realtime admission request is invalid." }, 400);
  }
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    Object.keys(value).length !== 1 ||
    typeof (value as { sessionId?: unknown }).sessionId !== "string" ||
    !SAFE_SESSION.test((value as { sessionId: string }).sessionId)
  ) {
    return noStoreGatewayJson({ error: "Realtime session identity is invalid." }, 400);
  }

  return proxyRealtimeGateway(
    request,
    "/admission",
    JSON.stringify({ sessionId: (value as { sessionId: string }).sessionId }),
  );
}
