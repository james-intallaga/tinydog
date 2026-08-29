import { proxyBackend } from "../_backend";
import {
  LocalJsonBodyError,
  readBoundedJsonObject,
  validateLocalMutation,
} from "../../../lib/local-request";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const rejection = validateLocalMutation(request, new Set(["application/json"]));
  if (rejection) return Response.json({ error: rejection.error }, { status: rejection.status });
  try {
    const body = await readBoundedJsonObject(request);
    const message = typeof body.message === "string" ? body.message.trim() : "";
    const sessionId = typeof body.session_id === "string" ? body.session_id.trim() : "";
    if (!message) return Response.json({ error: "Enter a message." }, { status: 400 });
    return proxyBackend("/chat", {
      method: "POST",
      body: { message, ...(sessionId ? { session_id: sessionId } : {}) },
      timeout: 60_000,
    });
  } catch (reason) {
    const status = reason instanceof LocalJsonBodyError ? reason.status : 400;
    return Response.json(
      { error: "The message must be valid and reasonably sized JSON." },
      { status },
    );
  }
}
