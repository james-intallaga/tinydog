const DEFAULT_BACKEND = "http://127.0.0.1:8765";

export class BodyTooLargeError extends Error {}

export function noStoreJson(body: Record<string, unknown>, status: number) {
  return Response.json(body, {
    status,
    headers: { "Cache-Control": "no-store" },
  });
}

export async function readBoundedBody(request: Request, maxBytes: number) {
  const contentLengthHeader = request.headers.get("content-length");
  if (contentLengthHeader !== null) {
    const contentLength = Number(contentLengthHeader);
    if (!Number.isInteger(contentLength) || contentLength < 0) {
      throw new TypeError("Invalid Content-Length header.");
    }
    if (contentLength > maxBytes) throw new BodyTooLargeError();
  }

  if (!request.body) return new Uint8Array();
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      totalBytes += value.byteLength;
      if (totalBytes > maxBytes) {
        await reader.cancel();
        throw new BodyTooLargeError();
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const body = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

export function normalizedContentType(request: Request) {
  return request.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase() ?? "";
}

export function validSessionHeader(request: Request) {
  const sessionId = request.headers.get("x-mlm-session")?.trim() ?? "";
  return {
    valid:
      sessionId.length > 0 &&
      sessionId.length <= 128 &&
      /^[A-Za-z0-9._:-]+$/.test(sessionId),
    value: sessionId,
  };
}

export function numericHeader(
  request: Request,
  name: string,
  { min, max, integer = false }: { min: number; max: number; integer?: boolean },
) {
  const raw = request.headers.get(name)?.trim() ?? "";
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) {
    return null;
  }
  return value;
}

function localBackendUrl(path: "/multimodal/frame" | "/multimodal/audio") {
  const configured = (process.env.MLM_SERVER_URL || DEFAULT_BACKEND).trim();
  const base = configured
    .replace(/\/(?:generate|chat|teach|history|status|multimodal\/(?:frame|audio))\/?$/, "")
    .replace(/\/$/, "");
  const url = new URL(`${base}${path}`);
  const localHosts = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);
  if (!localHosts.has(url.hostname) || !["http:", "https:"].includes(url.protocol) || url.username || url.password) {
    throw new Error("Realtime media is restricted to the local machine.");
  }
  return url;
}

export async function proxyLocalMedia(
  path: "/multimodal/frame" | "/multimodal/audio",
  body: Uint8Array,
  contentType: string,
  metadata: Record<string, string>,
) {
  try {
    const localToken = process.env.TINYDOG_LOCAL_API_TOKEN?.trim();
    const transferableBody = body.buffer.slice(
      body.byteOffset,
      body.byteOffset + body.byteLength,
    ) as ArrayBuffer;
    const response = await fetch(localBackendUrl(path), {
      method: "POST",
      body: transferableBody,
      cache: "no-store",
      headers: {
        "Content-Type": contentType,
        ...(localToken ? { "X-TinyDog-Local-Token": localToken } : {}),
        ...metadata,
      },
      signal: AbortSignal.timeout(10_000),
    });
    return new Response(response.status === 204 ? null : response.body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("content-type") ?? "application/json; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  } catch (reason) {
    return noStoreJson(
      {
        error:
          reason instanceof Error && reason.message.includes("restricted to the local machine")
            ? reason.message
            : "The local media receiver is not available.",
      },
      503,
    );
  }
}
