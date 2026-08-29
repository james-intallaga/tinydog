import {
  BodyTooLargeError,
  noStoreJson,
  normalizedContentType,
  numericHeader,
  proxyLocalMedia,
  readBoundedBody,
  validSessionHeader,
} from "../../_media";
import { validateLocalMutation } from "../../../../lib/local-request";

export const runtime = "nodejs";

const MAX_FRAME_BYTES = 512 * 1024;

export async function POST(request: Request) {
  const rejection = validateLocalMutation(request, new Set(["image/jpeg"]));
  if (rejection) return noStoreJson({ error: rejection.error }, rejection.status);
  if (normalizedContentType(request) !== "image/jpeg") {
    return noStoreJson({ error: "Camera frames must be JPEG images." }, 415);
  }

  const session = validSessionHeader(request);
  const timestamp = numericHeader(request, "x-mlm-timestamp-ms", {
    min: 0,
    max: Number.MAX_SAFE_INTEGER,
    integer: true,
  });
  const difference = numericHeader(request, "x-mlm-difference", { min: 0, max: 100 });
  const keyframe = request.headers.get("x-mlm-keyframe")?.toLowerCase();
  if (!session.valid || timestamp === null || difference === null || !["true", "false"].includes(keyframe ?? "")) {
    return noStoreJson({ error: "Camera frame metadata is invalid." }, 400);
  }

  try {
    const body = await readBoundedBody(request, MAX_FRAME_BYTES);
    if (body.length < 4 || body[0] !== 0xff || body[1] !== 0xd8 || body[2] !== 0xff) {
      return noStoreJson({ error: "The camera frame is not a valid JPEG image." }, 400);
    }
    return proxyLocalMedia("/multimodal/frame", body, "image/jpeg", {
      ...(session.value ? { "X-MLM-Session": session.value } : {}),
      "X-MLM-Timestamp-Ms": timestamp.toString(),
      "X-MLM-Difference": difference.toString(),
      "X-MLM-Keyframe": keyframe!,
    });
  } catch (reason) {
    if (reason instanceof BodyTooLargeError) {
      return noStoreJson({ error: "The camera frame is too large." }, 413);
    }
    return noStoreJson({ error: "The camera frame could not be read." }, 400);
  }
}
