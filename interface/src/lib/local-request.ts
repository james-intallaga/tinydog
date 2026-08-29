const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

export type LocalMutationRejection = {
  status: 403 | 415;
  error: string;
};

export class LocalJsonBodyError extends Error {
  constructor(
    message: string,
    readonly status: 400 | 413,
  ) {
    super(message);
  }
}

export async function readBoundedJsonObject(
  request: Request,
  maxBytes = 32 * 1024,
): Promise<Record<string, unknown>> {
  const declared = request.headers.get("content-length");
  if (declared !== null) {
    const length = Number(declared);
    if (!Number.isInteger(length) || length < 0) {
      throw new LocalJsonBodyError("The request has an invalid length.", 400);
    }
    if (length > maxBytes) {
      throw new LocalJsonBodyError("The local request is too large.", 413);
    }
  }

  if (!request.body) return {};
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        await reader.cancel();
        throw new LocalJsonBodyError("The local request is too large.", 413);
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    const value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    if (value === null || Array.isArray(value) || typeof value !== "object") {
      throw new TypeError("JSON payload is not an object.");
    }
    return value as Record<string, unknown>;
  } catch {
    throw new LocalJsonBodyError("The local request must contain valid JSON.", 400);
  }
}

export function validateLoopbackRequest(request: Request): LocalMutationRejection | null {
  try {
    const target = new URL(request.url);
    if (
      ["http:", "https:"].includes(target.protocol) &&
      LOOPBACK_HOSTS.has(target.hostname) &&
      !target.username &&
      !target.password
    ) {
      return null;
    }
  } catch {
    // Return the same deliberately generic rejection below.
  }
  return { status: 403, error: "This private local request was rejected." };
}

export function validateLocalMutation(
  request: Request,
  allowedContentTypes: ReadonlySet<string>,
): LocalMutationRejection | null {
  let target: URL;
  let origin: URL;
  try {
    target = new URL(request.url);
    const targetRejection = validateLoopbackRequest(request);
    if (targetRejection) return targetRejection;
    const suppliedOrigin = request.headers.get("origin")?.trim();
    if (!suppliedOrigin) {
      return { status: 403, error: "This local action requires a same-origin request." };
    }
    origin = new URL(suppliedOrigin);
  } catch {
    return { status: 403, error: "This local action has an invalid origin." };
  }

  const safeOrigin =
    ["http:", "https:"].includes(origin.protocol) &&
    LOOPBACK_HOSTS.has(origin.hostname) &&
    !origin.username &&
    !origin.password &&
    origin.origin === target.origin;
  if (!safeOrigin) {
    return { status: 403, error: "This local action was rejected." };
  }

  const contentType =
    request.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase() ?? "";
  if (!allowedContentTypes.has(contentType)) {
    return { status: 415, error: "This local action has an unsupported content type." };
  }
  return null;
}
