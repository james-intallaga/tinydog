import "server-only";

const REALTIME_GATEWAY = "http://127.0.0.1:8787";
const ALLOWED_BROWSER_ORIGINS = new Set([
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

export function browserOrigin(request: Request) {
  const supplied = request.headers.get("origin")?.trim();
  const candidate = supplied || new URL(request.url).origin;
  return ALLOWED_BROWSER_ORIGINS.has(candidate) ? candidate : null;
}

export function noStoreGatewayJson(body: Record<string, unknown>, status: number) {
  return Response.json(body, {
    status,
    headers: { "Cache-Control": "no-store" },
  });
}

export async function proxyRealtimeGateway(
  request: Request,
  path: "/status" | "/admission",
  body?: string,
) {
  const origin = browserOrigin(request);
  if (!origin) return noStoreGatewayJson({ error: "Realtime access is local only." }, 403);

  try {
    const response = await fetch(`${REALTIME_GATEWAY}${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        Origin: origin,
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(3_000),
    });
    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("application/json")) {
      return noStoreGatewayJson({ error: "The realtime gateway returned an invalid response." }, 502);
    }
    return new Response(await response.text(), {
      status: response.status,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return noStoreGatewayJson({ error: "The realtime gateway is not available." }, 503);
  }
}
