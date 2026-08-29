const DEFAULT_BACKEND = "http://127.0.0.1:8765";

function backendUrl(path: string) {
  const configured = (process.env.MLM_SERVER_URL || DEFAULT_BACKEND).trim();
  const base = configured
    .replace(/\/(?:generate|chat|teach|history|status)\/?$/, "")
    .replace(/\/$/, "");
  const url = new URL(`${base}${path}`);
  const localHosts = new Set(["localhost", "127.0.0.1", "::1"]);
  if (
    !localHosts.has(url.hostname) ||
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password
  ) {
    throw new Error("The learning service must stay on this computer.");
  }
  return url;
}

export async function proxyBackend(
  path: string,
  options: {
    method?: "GET" | "POST";
    body?: Record<string, unknown>;
    timeout?: number;
  } = {},
) {
  try {
    const localToken = process.env.TINYDOG_LOCAL_API_TOKEN?.trim();
    const response = await fetch(backendUrl(path), {
      method: options.method ?? "GET",
      headers: {
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(localToken ? { "X-TinyDog-Local-Token": localToken } : {}),
      },
      body: options.body ? JSON.stringify(options.body) : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(options.timeout ?? 30_000),
    });
    if (response.status === 204) return Response.json({ ok: true });

    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.includes("application/json")) {
      return Response.json(
        { error: response.ok ? "The learning service returned an invalid response." : "The learning service failed." },
        { status: response.ok ? 502 : response.status },
      );
    }
    return new Response(await response.text(), {
      status: response.status,
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return Response.json(
      { error: "The local learning service is not available." },
      { status: 503 },
    );
  }
}
