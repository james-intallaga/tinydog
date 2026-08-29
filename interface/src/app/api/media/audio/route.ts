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

const MAX_AUDIO_BYTES = 1024 * 1024;
const AUDIO_TYPES = new Set([
  "audio/webm",
  "audio/ogg",
  "audio/mp4",
  "audio/wav",
  "audio/x-wav",
]);

export async function POST(request: Request) {
  const rejection = validateLocalMutation(request, AUDIO_TYPES);
  if (rejection) return noStoreJson({ error: rejection.error }, rejection.status);
  const contentType = normalizedContentType(request);
  if (!AUDIO_TYPES.has(contentType)) {
    return noStoreJson({ error: "This voice recording format is not supported." }, 415);
  }

  const session = validSessionHeader(request);
  const timestamp = numericHeader(request, "x-mlm-timestamp-ms", {
    min: 0,
    max: Number.MAX_SAFE_INTEGER,
    integer: true,
  });
  const sequence = numericHeader(request, "x-mlm-sequence", {
    min: 0,
    max: Number.MAX_SAFE_INTEGER,
    integer: true,
  });
  if (!session.valid || timestamp === null || sequence === null) {
    return noStoreJson({ error: "Voice chunk metadata is invalid." }, 400);
  }

  try {
    const body = await readBoundedBody(request, MAX_AUDIO_BYTES);
    if (body.length === 0) return noStoreJson({ error: "The voice chunk is empty." }, 400);
    return proxyLocalMedia("/multimodal/audio", body, request.headers.get("content-type")!, {
      ...(session.value ? { "X-MLM-Session": session.value } : {}),
      "X-MLM-Timestamp-Ms": timestamp.toString(),
      "X-MLM-Sequence": sequence.toString(),
    });
  } catch (reason) {
    if (reason instanceof BodyTooLargeError) {
      return noStoreJson({ error: "The voice chunk is too large." }, 413);
    }
    return noStoreJson({ error: "The voice chunk could not be read." }, 400);
  }
}
