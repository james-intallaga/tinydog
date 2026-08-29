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
    const sessionId = typeof body.session_id === "string" ? body.session_id.trim() : "";
    const correctedAnswer =
      typeof body.corrected_answer === "string" ? body.corrected_answer.trim() : "";
    const messageId = body.message_id;
    const explanation = typeof body.explanation === "string" ? body.explanation.trim() : "";

    if (
      !sessionId ||
      !correctedAnswer ||
      (typeof messageId !== "string" && typeof messageId !== "number")
    ) {
      return Response.json(
        { error: "A session, model reply, and corrected answer are required." },
        { status: 400 },
      );
    }
    return proxyBackend("/teach", {
      method: "POST",
      body: {
        session_id: sessionId,
        message_id: messageId,
        corrected_answer: correctedAnswer,
        ...(explanation ? { explanation } : {}),
      },
      timeout: 60_000,
    });
  } catch (reason) {
    const status = reason instanceof LocalJsonBodyError ? reason.status : 400;
    return Response.json(
      { error: "The lesson must be valid and reasonably sized JSON." },
      { status },
    );
  }
}
