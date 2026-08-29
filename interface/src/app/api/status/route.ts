import { proxyBackend } from "../_backend";
import { validateLoopbackRequest } from "../../../lib/local-request";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const rejection = validateLoopbackRequest(request);
  if (rejection) return Response.json({ error: rejection.error }, { status: rejection.status });
  return proxyBackend("/status");
}
