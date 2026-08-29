"""Serve persistent conversation, verified teaching, memory, and live inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import signal
from typing import Any
from urllib.parse import parse_qs, urlparse

from .apprenticeship import (
    ApprenticeshipStore,
    TRUSTED_HUMAN_DIALOGUE_SOURCE,
    TRUSTED_HUMAN_DIALOGUE_VERIFIER,
    canonical_practice_key,
    generate_practice_variants,
    parse_correction,
    record_verified_correction,
    render_lesson_prompt,
    verify_answer,
)
from .coordinator import (
    LearningCoordinator,
    resolve_active_checkpoint,
)
from .local_floaty import LocalFloatyAutoConfig, LocalFloatyCoordinator
from .realtime_ingress import MAX_AUDIO_BYTES, MAX_FRAME_BYTES, RealtimeIngress
from .runtime import ModelRuntime
from .teacher import TeacherClient


LOCAL_API_TOKEN_HEADER = "X-TinyDog-Local-Token"
MAX_JSON_REQUEST_BYTES = 1_000_000
_LOOPBACK_BIND_HOSTS = frozenset({"localhost", "127.0.0.1"})


def _is_loopback_hostname(host: str | None) -> bool:
    if not host:
        return False
    candidate = host.strip().lower().rstrip(".")
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _validated_bind_host(host: str) -> str:
    candidate = host.strip().lower()
    if candidate not in _LOOPBACK_BIND_HOSTS:
        raise ValueError(
            "The legacy learning service may bind only to localhost or 127.0.0.1."
        )
    return candidate


def _loopback_host_header(authority: str) -> bool:
    if (
        not authority
        or len(authority) > 1_024
        or any(character in authority for character in "/?#@")
    ):
        return False
    try:
        parsed = urlparse(f"//{authority}")
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.hostname
        and not parsed.username
        and not parsed.password
        and (port is None or 1 <= port <= 65_535)
        and _is_loopback_hostname(parsed.hostname)
    )


def _loopback_origin(origin: str) -> bool:
    if not origin or len(origin) > 2_048:
        return False
    try:
        parsed = urlparse(origin)
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme in {"http", "https"}
        and parsed.hostname
        and not parsed.username
        and not parsed.password
        and (port is None or 1 <= port <= 65_535)
        and parsed.path in {"", "/"}
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and _is_loopback_hostname(parsed.hostname)
    )


class _LocalOnlyRequestHandler(BaseHTTPRequestHandler):
    """Shared admission and bounded-body rules for the legacy local service."""

    expected_local_api_token = ""

    def _authorized(self, *, expect_json: bool) -> bool:
        peer_host = str(self.client_address[0]) if self.client_address else ""
        if not _is_loopback_hostname(peer_host) or not _loopback_host_header(
            self.headers.get("Host", "")
        ):
            self._json(
                403,
                {"error": "The local learning service rejected this host."},
            )
            return False

        if self.expected_local_api_token:
            supplied = self.headers.get(LOCAL_API_TOKEN_HEADER, "")
            if not hmac.compare_digest(supplied, self.expected_local_api_token):
                self._json(
                    403,
                    {"error": "The local learning request is not authorized."},
                )
                return False
        else:
            origin = self.headers.get("Origin", "").strip()
            if origin and not _loopback_origin(origin):
                self._json(
                    403,
                    {"error": "The local learning service rejected this origin."},
                )
                return False

        if expect_json:
            media_type = (
                self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            )
            if media_type != "application/json":
                self._json(
                    415,
                    {"error": "Content-Type must be application/json."},
                )
                return False
        return True

    def _payload(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Transfer-Encoding is not supported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Content-Length is invalid") from error
        if length <= 0 or length > MAX_JSON_REQUEST_BYTES:
            raise ValueError("A bounded JSON request body is required")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("JSON request body ended early")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _binary_body(self, maximum: int) -> bytes:
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Transfer-Encoding is not supported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Content-Length is invalid") from error
        if length <= 0 or length > maximum:
            raise ValueError(f"Binary request body must be 1-{maximum} bytes")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("Binary request body ended early")
        return body

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _make_loopback_http_server(
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((_validated_bind_host(host), port), handler)
    server.daemon_threads = True
    return server


def _jsonable(value):
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _skill(expression: str | None) -> str:
    if not expression:
        return "arithmetic"
    if "%" in expression:
        return "percentages"
    if "/" in expression:
        return "division and fractions"
    if "*" in expression:
        return "multiplication"
    if "-" in expression:
        return "subtraction"
    if "+" in expression:
        return "addition"
    return "arithmetic"


def _history_turns(messages, *, limit: int = 4) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for message in messages[-limit:]:
        role = (
            message.role
            if message.role in {"user", "assistant", "teacher"}
            else "teacher"
        )
        turns.append((role, message.content[:256]))
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/300m.json")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--db", default="data/apprenticeship.db")
    parser.add_argument("--candidate-dir", default="checkpoints/apprenticeship")
    parser.add_argument(
        "--training-authority",
        choices=("owner", "community"),
        default=os.getenv("MLM_TRAINING_AUTHORITY", "community"),
        help=(
            "community is the safe public default: it can package signed "
            "floaty-only local contributions but never mutates this legacy "
            "runtime; owner explicitly enables its full-model trainer"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=ModelRuntime.MAX_CHAT_NEW_TOKENS,
        help=f"Reply token budget (capped at {ModelRuntime.MAX_CHAT_NEW_TOKENS})",
    )
    args = parser.parse_args()
    if args.training_authority not in {"owner", "community"}:
        parser.error("MLM_TRAINING_AUTHORITY must be 'owner' or 'community'")
    try:
        args.host = _validated_bind_host(args.host)
    except ValueError as error:
        parser.error(str(error))
    configured_api_token = os.environ.get("TINYDOG_LOCAL_API_TOKEN")
    if configured_api_token is not None and not configured_api_token.strip():
        parser.error("TINYDOG_LOCAL_API_TOKEN must not be empty when configured")
    local_api_token = (configured_api_token or "").strip()

    candidate_dir = Path(args.candidate_dir).resolve()
    active_weights = resolve_active_checkpoint(
        args.weights,
        candidate_dir / "active.json",
        args.config,
    )
    store = ApprenticeshipStore(args.db)
    runtime = ModelRuntime(args.config, active_weights)
    teacher = TeacherClient.from_env()
    coordinator = (
        LearningCoordinator(store, runtime, args.config, candidate_dir)
        if args.training_authority == "owner"
        else LocalFloatyCoordinator(
            store,
            LocalFloatyAutoConfig.from_env(project_root=Path.cwd()),
        )
    )
    realtime = RealtimeIngress()
    coordinator.consider_training()

    class Handler(_LocalOnlyRequestHandler):
        expected_local_api_token = local_api_token

        def do_GET(self) -> None:
            if not self._authorized(expect_json=False):
                return
            parsed = urlparse(self.path)
            if parsed.path == "/status":
                self._json(200, self._status())
                return
            if parsed.path == "/history":
                session_id = parse_qs(parsed.query).get("session_id", [""])[0]
                if not session_id:
                    self._json(400, {"error": "session_id is required"})
                    return
                self._json(
                    200,
                    {
                        "session_id": session_id,
                        "messages": store.list_messages(session_id),
                    },
                )
                return
            if parsed.path == "/lessons":
                lessons = [
                    lesson
                    for lesson in store.list_recent_lessons(limit=100)
                    if lesson.source != TRUSTED_HUMAN_DIALOGUE_SOURCE
                    or lesson.verifier != TRUSTED_HUMAN_DIALOGUE_VERIFIER
                ]
                self._json(200, {"lessons": lessons})
                return
            self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if not self._authorized(
                expect_json=parsed.path
                not in {"/multimodal/frame", "/multimodal/audio"}
            ):
                return
            try:
                if parsed.path == "/multimodal/frame":
                    packet = realtime.accept_frame(
                        self._binary_body(MAX_FRAME_BYTES),
                        self.headers.get("Content-Type", ""),
                        self.headers,
                    )
                    self._json(
                        202,
                        {
                            "accepted": True,
                            "kind": "video-frame",
                            "packet": packet.public_metadata(),
                            "model_connected": False,
                        },
                    )
                    return
                if parsed.path == "/multimodal/audio":
                    packet = realtime.accept_audio(
                        self._binary_body(MAX_AUDIO_BYTES),
                        self.headers.get("Content-Type", ""),
                        self.headers,
                    )
                    self._json(
                        202,
                        {
                            "accepted": True,
                            "kind": "audio-chunk",
                            "packet": packet.public_metadata(),
                            "model_connected": False,
                        },
                    )
                    return
                payload = self._payload()
                if parsed.path == "/chat":
                    self._json(200, self._chat(payload))
                    return
                if parsed.path == "/teach":
                    self._json(200, self._teach(payload))
                    return
                if parsed.path == "/train":
                    started = coordinator.trigger(force=True)
                    self._json(
                        202 if started else 200,
                        {"started": started, "learning": coordinator.status()},
                    )
                    return
                if parsed.path == "/rollback":
                    self._json(200, coordinator.rollback())
                    return
                if parsed.path == "/generate":
                    question = str(payload.get("question", "")).strip()
                    if not question:
                        raise ValueError("A question is required")
                    self._json(
                        200,
                        {
                            "answer": runtime.answer(
                                [("user", question)], max_tokens=args.max_new_tokens
                            )
                        },
                    )
                    return
                self._json(404, {"error": "Not found"})
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
            except Exception as error:
                print(f"request failed: {type(error).__name__}: {error}", flush=True)
                self._json(500, {"error": "Request failed."})

        def _chat(self, payload: dict) -> dict:
            text = str(payload.get("message", payload.get("question", ""))).strip()
            if not text:
                raise ValueError("A message is required")
            if len(text.encode("utf-8")) > 8_192:
                raise ValueError("Message is too long")

            requested_session = str(payload.get("session_id", "")).strip() or None
            session = store.create_session(requested_session)
            prior = store.list_messages(session.id)
            last_assistant = next(
                (message for message in reversed(prior) if message.role == "assistant"),
                None,
            )
            preceding = (
                store.find_preceding_problem(last_assistant.id)
                if last_assistant
                else None
            )
            parsed_correction = parse_correction(
                text, problem=preceding.content if preceding else None
            )
            user_message = store.add_message(
                session.id,
                "user",
                text,
                reply_to_id=last_assistant.id if last_assistant else None,
                metadata={"input": "text", "local_training": True},
            )
            dialogue_lesson = store.add_trusted_dialogue_lesson(
                human_response_message_id=user_message.id,
                prompt_message_id=last_assistant.id if last_assistant else None,
                consent_local_training=True,
                metadata={"capture": "text-chat", "automatic": True},
            )

            if parsed_correction is not None and last_assistant is not None:
                lesson = record_verified_correction(
                    store,
                    session.id,
                    last_assistant.id,
                    text,
                    correction_message_id=user_message.id,
                    source="human-conversation",
                )
                if lesson is not None:
                    self._add_practice(lesson)
                    store.update_message_metadata(
                        last_assistant.id,
                        {
                            "lesson_saved": True,
                            "human_taught": True,
                            "lesson_id": lesson.id,
                        },
                    )
                    reply = f"Thank you. I saved the verified correction: {lesson.correct_answer}."
                    reply_message = store.add_message(
                        session.id,
                        "assistant",
                        reply,
                        reply_to_id=user_message.id,
                        metadata={
                            "lesson_id": lesson.id,
                            "lesson_saved": True,
                            "verification": {
                                "status": "taught",
                                "expected": lesson.correct_answer,
                            },
                            "kind": "learning_acknowledgement",
                        },
                    )
                    coordinator.consider_training()
                    return {
                        "session_id": session.id,
                        "message_id": reply_message.id,
                        "answer": reply,
                        "verification": {
                            "status": "taught",
                            "expected": lesson.correct_answer,
                        },
                        "lesson_saved": True,
                        "conversation_lesson_saved": True,
                        "conversation_lesson_id": dialogue_lesson.id,
                        "memory_used": 0,
                        "learning": coordinator.status(),
                    }

            memories = store.search_relevant_lessons(text, limit=1)
            turns: list[tuple[str, str]] = []
            if memories:
                memory_text = "\n".join(
                    render_lesson_prompt(lesson) for lesson in memories
                )
                turns.append(("memory", memory_text[:256]))
            turns.extend(_history_turns(prior))
            turns.append(("user", text))
            answer = runtime.answer(turns, max_tokens=args.max_new_tokens)
            if not answer:
                answer = "I do not know yet. Please teach me the answer."
            assistant = store.add_message(
                session.id, "assistant", answer, reply_to_id=user_message.id
            )

            verification = verify_answer(text, answer)
            teacher_feedback = None
            lesson = None
            if verification.expected is not None:
                if not verification.is_correct:
                    teacher_feedback = teacher.diagnose(
                        text, answer, verification.expected
                    )
                feedback = (
                    "The exact verifier supplied the correct result."
                    if not verification.is_correct
                    else "The answer was verified exactly."
                )
                lesson = store.add_verified_lesson(
                    problem=text,
                    first_attempt=answer,
                    correct_answer=verification.expected,
                    feedback=feedback,
                    skill=_skill(verification.expression),
                    source="exact-verifier",
                    session_id=session.id,
                    problem_message_id=user_message.id,
                    assistant_message_id=assistant.id,
                    metadata={
                        "automatic": True,
                        "teacher_advisory": (
                            _jsonable(teacher_feedback)
                            if teacher_feedback is not None
                            and teacher_feedback.available
                            else None
                        ),
                        "teacher_advisory_trusted_for_training": False,
                    },
                )
                self._add_practice(lesson)
                coordinator.consider_training()

            status = "unverified"
            if verification.expected is not None:
                status = "correct" if verification.is_correct else "incorrect"
            durable_verification = {
                "status": status,
                "expected": verification.expected,
                "attempted": verification.attempted,
                "expression": verification.expression,
                "error": verification.error,
            }
            store.update_message_metadata(
                assistant.id,
                {
                    "verification": durable_verification,
                    "lesson_saved": lesson is not None,
                    "lesson_id": lesson.id if lesson else None,
                    "memory_used": len(memories),
                    "teacher_hint": (
                        teacher_feedback.hint
                        if teacher_feedback is not None and teacher_feedback.available
                        else None
                    ),
                },
            )
            coordinator.consider_training()
            return {
                "session_id": session.id,
                "message_id": assistant.id,
                "answer": answer,
                "verification": durable_verification,
                "lesson_saved": lesson is not None,
                "lesson_id": lesson.id if lesson else None,
                "conversation_lesson_saved": True,
                "conversation_lesson_id": dialogue_lesson.id,
                "memory_used": len(memories),
                "teacher_hint": teacher_feedback.hint
                if teacher_feedback and teacher_feedback.available
                else None,
                "learning": coordinator.status(),
            }

        def _teach(self, payload: dict) -> dict:
            session_id = str(payload.get("session_id", "")).strip()
            message_id = int(payload.get("message_id", 0))
            corrected = str(payload.get("corrected_answer", "")).strip()
            explanation = str(payload.get("explanation", "")).strip()
            if not session_id or not message_id or not corrected:
                raise ValueError(
                    "session_id, message_id, and corrected_answer are required"
                )
            messages = store.list_messages(session_id)
            assistant = next(
                (
                    message
                    for message in messages
                    if message.id == message_id and message.role == "assistant"
                ),
                None,
            )
            if assistant is None:
                raise ValueError("Model message was not found")
            problem_message = store.find_preceding_problem(assistant.id)
            if problem_message is None:
                raise ValueError("The original problem was not found")
            verification = verify_answer(problem_message.content, corrected)
            if verification.expected is None:
                raise ValueError(
                    "This version can permanently learn only automatically verifiable arithmetic"
                )
            if not verification.is_correct:
                raise ValueError(
                    f"The verifier expects {verification.expected}, not {corrected}"
                )
            correction_message = store.add_message(
                session_id,
                "teacher",
                explanation or f"The correct answer is {verification.expected}.",
                reply_to_id=assistant.id,
                metadata={"corrected_answer": verification.expected},
            )
            lesson = store.add_verified_lesson(
                problem=problem_message.content,
                first_attempt=assistant.content,
                correct_answer=verification.expected,
                feedback=correction_message.content,
                skill=_skill(verification.expression),
                source="human-teacher",
                session_id=session_id,
                problem_message_id=problem_message.id,
                assistant_message_id=assistant.id,
                correction_message_id=correction_message.id,
            )
            store.update_message_metadata(
                assistant.id,
                {"lesson_saved": True, "human_taught": True, "lesson_id": lesson.id},
            )
            self._add_practice(lesson)
            coordinator.consider_training()
            return {
                "saved": True,
                "lesson": lesson,
                "verification": {
                    "status": "verified",
                    "expected": verification.expected,
                },
                "learning": coordinator.status(),
            }

        def _add_practice(self, lesson) -> None:
            for problem, answer in generate_practice_variants(
                lesson.problem, count=5, seed=lesson.id
            ):
                canonical_key = canonical_practice_key(problem)
                if canonical_key is None:
                    continue
                store.add_practice(
                    lesson.id,
                    problem,
                    answer,
                    metadata={
                        "generated": True,
                        "canonical_key": canonical_key,
                        "data_split": "train",
                    },
                )

        def _status(self) -> dict:
            teacher_status = teacher.status()
            return {
                "ok": True,
                "model": runtime.status(),
                "teacher": teacher_status,
                "learning": coordinator.status(),
                "realtime": realtime.status(),
                "lessons": {
                    "total": len(store.list_recent_lessons(limit=10_000)),
                    "pending": store.count_pending_lessons(),
                },
            }

    server = _make_loopback_http_server(args.host, args.port, Handler)

    def stop_on_termination(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_termination)
    print(
        f"MLM apprenticeship ready at http://{args.host}:{args.port} "
        f"({runtime.status()['parameters']:,} parameters)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        coordinator.shutdown()


if __name__ == "__main__":
    main()
