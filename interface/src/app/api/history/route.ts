import { proxyBackend } from "../_backend";
import { validateLoopbackRequest } from "../../../lib/local-request";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const rejection = validateLoopbackRequest(request);
  if (rejection) return Response.json({ error: rejection.error }, { status: rejection.status });
  const sessionId = new URL(request.url).searchParams.get("session_id")?.trim();
  if (!sessionId) return Response.json({ error: "A session ID is required." }, { status: 400 });
  return proxyBackend(`/history?session_id=${encodeURIComponent(sessionId)}`);
}
