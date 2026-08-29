import { proxyBackend } from "../_backend";
import {
  LocalJsonBodyError,
  readBoundedJsonObject,
  validateLocalMutation,
} from "../../../lib/local-request";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const rejection = validateLocalMutation(request, new Set(["application/json"]));
  if (rejection) {
    return Response.json({ error: rejection.error }, { status: rejection.status });
  }
  try {
    await readBoundedJsonObject(request, 512);
  } catch (reason) {
    const status = reason instanceof LocalJsonBodyError ? reason.status : 400;
    return Response.json({ error: "The rollback request must be valid JSON." }, { status });
  }
  return proxyBackend("/rollback", {
    method: "POST",
    body: {},
    timeout: 60_000,
  });
}
