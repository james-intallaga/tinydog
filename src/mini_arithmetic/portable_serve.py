"""Serve a private, portable, conversation-trained tinydog.ai model.

The service is deliberately local-only. It creates a random-initialized
portable Hybrid bundle on first use, records typed dialogue in SQLite, and
trains a new immutable foundation checkpoint after every five distinct human
turns. No contribution bundles, signatures, uploads, or pretrained weights
are part of this path. The earlier low-rank adapter learner remains available
only as an explicit compatibility mode.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass, replace
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import socket
import threading
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qs, urlparse

from .apprenticeship import ApprenticeshipStore, TRUSTED_DIALOGUE_BOOTSTRAP_PROMPT
from .personal_foundation import (
    PersonalFoundationError,
    PersonalFoundationTrainingConfig,
    PersonalFoundationTrainingCoordinator,
)
from .personal_training import (
    PersonalTrainingConfig,
    PersonalTrainingCoordinator,
    PersonalTrainingError,
)
from .portable import BackendSelection, select_backend
from .portable_bundle import PortableBundle, ensure_portable_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_MESSAGE_BYTES = 8_192
MAX_SESSION_ID_BYTES = 128
HISTORY_RESPONSE_LIMIT = 64
INFERENCE_HISTORY_LIMIT = 8
DEFAULT_MAX_NEW_TOKENS = 32
LOCAL_API_TOKEN_HEADER = "X-TinyDog-Local-Token"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_LOOPBACK_BIND_HOSTS = frozenset({"localhost", "127.0.0.1"})


class PortableServeError(RuntimeError):
    """The portable local service cannot start safely."""


class ConversationalRuntime(Protocol):
    """Backend-neutral inference surface used by the local HTTP service."""

    def answer(
        self,
        turns: Sequence[tuple[str, str]],
        *,
        max_tokens: int,
    ) -> str: ...

    def status(self) -> dict[str, Any]: ...

    def personal_training_target_fits(
        self,
        prompt: str,
        human_response: str,
    ) -> bool: ...


class LocalLearningCoordinator(Protocol):
    """Common lifecycle shared by foundation and legacy adapter learning."""

    def consider_training(self) -> bool: ...

    def trigger(self, *, force: bool = False) -> bool: ...

    def status(self) -> dict[str, Any]: ...

    def rollback(self) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def choose_model_size(requested: str, selection: BackendSelection) -> str:
    """Choose a portable default without guessing available accelerator memory."""

    if requested not in {"auto", "tiny", "standard"}:
        raise ValueError("model size must be auto, tiny, or standard")
    del selection
    return "tiny" if requested == "auto" else requested


def _resolve_local_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )


def _personal_base_path(
    root: Path,
    command_line_value: str | Path | None,
    configured: PersonalTrainingConfig | PersonalFoundationTrainingConfig,
) -> Path:
    """Resolve an explicit CLI override or honor the private env configuration."""

    if command_line_value is None:
        return configured.artifact_root.expanduser().resolve()
    return _resolve_local_path(root, command_line_value)


def personal_profile_id(identity: Any) -> str:
    """Name a private adapter registry by its exact runtime compatibility."""

    identity.validate()
    payload = (
        "tinydog-personal-profile-v1\0"
        f"{identity.backend}\0{identity.foundation_sha256}\0"
        f"{identity.adapter_schema_sha256}"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:24]


class LocalInstanceLease:
    """Hold one process-wide lock over a private personal-model directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            self.path.parent.chmod(0o700)
        self._handle = self.path.open("a+b")
        self._locked = False
        try:
            self._acquire()
        except Exception:
            self._handle.close()
            raise

    def _acquire(self) -> None:
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"\0")
            self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise PortableServeError(
                "Another tinydog.ai learning service is already using this private model."
            ) from error
        self._locked = True

    def close(self) -> None:
        if self._handle.closed:
            return
        if self._locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    self._handle.seek(0)
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._locked = False
        self._handle.close()

    def __enter__(self) -> "LocalInstanceLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PortableChatService:
    """Backend-independent conversation, ledger, and apprenticeship logic."""

    def __init__(
        self,
        store: ApprenticeshipStore,
        runtime: ConversationalRuntime,
        coordinator: LocalLearningCoordinator,
        *,
        bundle: PortableBundle,
        selection: BackendSelection,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        bundle_created: bool = False,
        learning_mode: str = "foundation",
    ) -> None:
        if not callable(getattr(runtime, "answer", None)) or not callable(
            getattr(runtime, "status", None)
        ):
            raise TypeError(
                "runtime does not implement portable conversation inference"
            )
        if not all(
            callable(getattr(coordinator, name, None))
            for name in (
                "consider_training",
                "trigger",
                "status",
                "rollback",
                "shutdown",
            )
        ):
            raise TypeError("coordinator does not implement local learning lifecycle")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        if learning_mode not in {"foundation", "adapter"}:
            raise ValueError("learning_mode must be foundation or adapter")
        self.store = store
        self.runtime = runtime
        self.coordinator = coordinator
        self.bundle = bundle
        self.selection = selection
        self.max_new_tokens = max_new_tokens
        self.bundle_created = bundle_created
        self.learning_mode = learning_mode
        self.foundation_profile_id: str | None = None
        if learning_mode == "foundation":
            profile_id = getattr(coordinator, "profile_id", None)
            if (
                not isinstance(profile_id, str)
                or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", profile_id) is None
            ):
                raise TypeError(
                    "foundation coordinator does not expose a valid profile identity"
                )
            self.foundation_profile_id = profile_id
        self._chat_lock = threading.RLock()

    def close(self) -> None:
        self.coordinator.shutdown()

    def history(self, session_id: str) -> dict[str, Any]:
        clean = self._session_id(session_id, required=True)
        assert clean is not None
        return {
            "session_id": clean,
            "messages": self.store.list_messages(clean, limit=HISTORY_RESPONSE_LIMIT),
            "limit": HISTORY_RESPONSE_LIMIT,
        }

    def chat(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        # The message ledger and provenance lesson form one logical turn.  A
        # per-service lock prevents two requests for the same browser session
        # from interleaving between those SQLite transactions.
        with self._chat_lock:
            return self._chat_locked(payload)

    def _chat_locked(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        text = str(payload.get("message", payload.get("question", ""))).strip()
        if not text:
            raise ValueError("A message is required")
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_MESSAGE_BYTES or "\x00" in text:
            raise ValueError("Message is too long or contains an invalid character")

        requested_session = self._session_id(payload.get("session_id"), required=False)
        session = self.store.create_session(requested_session)
        prior = self.store.list_messages(session.id, limit=INFERENCE_HISTORY_LIMIT)
        last_assistant = next(
            (message for message in reversed(prior) if message.role == "assistant"),
            None,
        )
        training_prompt = (
            last_assistant.content
            if last_assistant is not None
            else TRUSTED_DIALOGUE_BOOTSTRAP_PROMPT
        )
        target_checker = getattr(self.runtime, "personal_training_target_fits", None)
        if callable(target_checker) and not target_checker(training_prompt, text):
            raise ValueError(
                "Message is too long to preserve as a complete local-training target"
            )
        turns = self._history_turns(prior)
        turns.append(("user", text))

        # Generate before committing the new human turn.  A device/backend
        # failure therefore cannot strand a half-written alternating dialogue.
        answer = self.runtime.answer(turns, max_tokens=self.max_new_tokens).strip()
        if not answer:
            answer = "…"

        user_message, dialogue_lesson, assistant = self.store.add_trusted_dialogue_turn(
            session.id,
            text,
            answer,
            prompt_message_id=last_assistant.id if last_assistant else None,
            human_metadata={
                "input": "text",
                "capture": "text-chat",
                "local_training": True,
                "privacy": "private-local-only",
            },
            lesson_metadata={
                "capture": "text-chat",
                "automatic": True,
                "privacy": "private-local-only",
            },
            assistant_metadata={
                "backend": self.selection.backend,
                "model_generated": True,
            },
            foundation_profile_id=self.foundation_profile_id,
        )
        consider_lesson = getattr(self.coordinator, "consider_lesson", None)
        if callable(consider_lesson):
            consider_lesson(dialogue_lesson.id)
        else:
            # Compatibility for embedders implementing the original minimal
            # coordinator surface.
            self.coordinator.consider_training()
        return {
            "session_id": session.id,
            "message_id": assistant.id,
            "answer": answer,
            "conversation_lesson_saved": True,
            "conversation_lesson_id": dialogue_lesson.id,
            "learning": self.coordinator.status(),
        }

    @staticmethod
    def _history_turns(
        messages: Sequence[Any],
        *,
        limit: int = INFERENCE_HISTORY_LIMIT,
    ) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        for message in messages[-limit:]:
            if message.role not in {"user", "assistant"}:
                continue
            content = message.content
            # Bound prompt assembly before tokenization while preserving UTF-8.
            if len(content.encode("utf-8")) > 2_048:
                content = content.encode("utf-8")[-2_048:].decode("utf-8", "ignore")
            turns.append((message.role, content))
        return turns

    @staticmethod
    def _session_id(value: object, *, required: bool) -> str | None:
        if value is None or value == "":
            if required:
                raise ValueError("session_id is required")
            return None
        if not isinstance(value, str):
            raise ValueError("session_id must be a short local identifier")
        clean = value.strip()
        if (
            not clean
            or len(clean.encode("utf-8")) > MAX_SESSION_ID_BYTES
            or re.fullmatch(r"[A-Za-z0-9._-]+", clean) is None
        ):
            raise ValueError("session_id must be a short local identifier")
        return clean

    def status(self) -> dict[str, Any]:
        runtime_status = self.runtime.status()
        return {
            "ok": True,
            "privacy": {
                "mode": "private-local-only",
                "uploads_enabled": False,
                "community_aggregation": False,
            },
            "portable": {
                **self.selection.as_dict(),
                "model_size": self.bundle.model_size,
                "learning_mode": self.learning_mode,
                "bundle_created_this_start": self.bundle_created,
                "scratch_only": True,
                "external_weights": False,
            },
            "model": runtime_status,
            "learning": self.coordinator.status(),
            "lessons": {
                "total": self.store.count_lessons(),
                "pending": self.store.count_lessons(status="pending"),
            },
            "realtime": {
                "transport": "not-started-by-portable-text-service",
                "model_connected": False,
                "voice_trained": False,
                "conversation_ready": False,
            },
        }


def make_http_server(
    host: str,
    port: int,
    service: PortableChatService,
    *,
    api_token: str | None = None,
) -> ThreadingHTTPServer:
    if host.strip().lower() not in _LOOPBACK_BIND_HOSTS:
        raise PortableServeError(
            "The private learning service may bind only to localhost or 127.0.0.1."
        )
    expected_token = (api_token or "").strip()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if not self._authorized(expect_json=False):
                return
            parsed = urlparse(self.path)
            if parsed.path == "/status":
                self._json(200, service.status())
                return
            if parsed.path == "/history":
                try:
                    session_id = parse_qs(parsed.query).get("session_id", [""])[0]
                    self._json(200, service.history(session_id))
                except ValueError as error:
                    self._json(400, {"error": str(error)})
                return
            self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if not self._authorized(
                expect_json=not parsed.path.startswith("/multimodal/")
            ):
                return
            if parsed.path.startswith("/multimodal/"):
                self._json(
                    501,
                    {"error": "Portable semantic voice training is not available yet."},
                )
                return
            try:
                payload = self._payload()
                if parsed.path == "/chat":
                    self._json(200, service.chat(payload))
                    return
                if parsed.path == "/generate":
                    question = str(payload.get("question", "")).strip()
                    if not question:
                        raise ValueError("A question is required")
                    if (
                        len(question.encode("utf-8")) > MAX_MESSAGE_BYTES
                        or "\x00" in question
                    ):
                        raise ValueError(
                            "Question is too long or contains an invalid character"
                        )
                    answer = service.runtime.answer(
                        [("user", question)], max_tokens=service.max_new_tokens
                    )
                    self._json(200, {"answer": answer})
                    return
                if parsed.path == "/train":
                    started = service.coordinator.trigger(force=True)
                    self._json(
                        202 if started else 200,
                        {"started": started, "learning": service.coordinator.status()},
                    )
                    return
                if parsed.path == "/rollback":
                    self._json(200, service.coordinator.rollback())
                    return
                if parsed.path == "/teach":
                    self._json(
                        400,
                        {
                            "error": "Every typed reply already teaches this private model."
                        },
                    )
                    return
                self._json(404, {"error": "Not found"})
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
            except PersonalTrainingError as error:
                self._json(409, {"error": str(error)})
            except Exception as error:
                print(
                    f"portable request failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                self._json(500, {"error": "The local learning request failed safely."})

        def _authorized(self, *, expect_json: bool) -> bool:
            try:
                host = urlparse(f"//{self.headers.get('Host', '')}").hostname
            except ValueError:
                host = None
            if host is None or host.lower() not in _LOOPBACK_HOSTS:
                self._json(
                    403, {"error": "The local learning service rejected this host."}
                )
                return False

            if expected_token:
                supplied = self.headers.get(LOCAL_API_TOKEN_HEADER, "")
                if not hmac.compare_digest(supplied, expected_token):
                    self._json(
                        403, {"error": "The local learning request is not authorized."}
                    )
                    return False
            else:
                origin = self.headers.get("Origin", "").strip()
                if origin and not self._loopback_origin(origin):
                    self._json(
                        403,
                        {"error": "The local learning service rejected this origin."},
                    )
                    return False

            if expect_json:
                content_type = self.headers.get("Content-Type", "")
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type != "application/json":
                    self._json(415, {"error": "Content-Type must be application/json."})
                    return False
            return True

        @staticmethod
        def _loopback_origin(origin: str) -> bool:
            try:
                parsed = urlparse(origin)
                _ = parsed.port
            except ValueError:
                return False
            return (
                parsed.scheme in {"http", "https"}
                and (parsed.hostname or "").lower() in _LOOPBACK_HOSTS
                and not parsed.username
                and not parsed.password
                and parsed.path in {"", "/"}
                and not parsed.params
                and not parsed.query
                and not parsed.fragment
            )

        def _payload(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("Content-Length is invalid") from error
            if length <= 0 or length > 1_000_000:
                raise ValueError("A bounded JSON request body is required")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

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

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _assert_port_available(host: str, port: int) -> None:
    """Avoid allocating a model when an old local service owns the port."""

    if host.strip().lower() not in _LOOPBACK_BIND_HOSTS:
        raise PortableServeError(
            "The private learning service may bind only to localhost or 127.0.0.1."
        )
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, port))
    except OSError as error:
        raise PortableServeError(
            f"The local address {host}:{port} is already in use. Stop the older service first."
        ) from error


def _backend_components(
    bundle: PortableBundle,
    selection: BackendSelection,
    *,
    rank: int,
    learning_mode: str,
) -> tuple[Any, ConversationalRuntime]:
    if learning_mode not in {"foundation", "adapter"}:
        raise PortableServeError("learning mode must be foundation or adapter")
    try:
        if selection.backend == "torch":
            if learning_mode == "foundation":
                from .torch_personal import create_torch_personal_foundation_backend

                return create_torch_personal_foundation_backend(bundle, selection)
            from .torch_personal import create_torch_personal_backend

            return create_torch_personal_backend(bundle, selection, rank=rank)
        if selection.backend == "mlx":
            if learning_mode == "foundation":
                from .mlx_personal import create_mlx_personal_foundation_backend

                return create_mlx_personal_foundation_backend(bundle)
            from .mlx_personal import create_mlx_personal_backend

            return create_mlx_personal_backend(bundle, rank=rank)
    except (ImportError, OSError, RuntimeError) as error:
        raise PortableServeError(
            f"The {selection.backend} engine could not initialize: {error}"
        ) from error
    raise PortableServeError("No compatible local compute backend is ready")


def build_service(
    *,
    root: Path,
    backend: str,
    model_size: str,
    db_path: str | Path,
    model_root: str | Path,
    personal_root: str | Path | None,
    seed: int,
    rank: int,
    max_new_tokens: int,
    learning_mode: str = "foundation",
) -> tuple[PortableChatService, LocalInstanceLease]:
    selection = select_backend(backend)  # type: ignore[arg-type]
    if not selection.ready or selection.backend not in {"mlx", "torch"}:
        raise PortableServeError(selection.reason)
    resolved_size = choose_model_size(model_size, selection)
    config_path = root / "configs" / f"portable-{resolved_size}.json"
    model_parent = _resolve_local_path(root, model_root)
    model_directory = model_parent / f"portable-{resolved_size}"
    if learning_mode == "foundation":
        base_training_config: (
            PersonalTrainingConfig | PersonalFoundationTrainingConfig
        ) = PersonalFoundationTrainingConfig.from_env(project_root=root)
    elif learning_mode == "adapter":
        base_training_config = PersonalTrainingConfig.from_env(project_root=root)
    else:
        raise PortableServeError("learning mode must be foundation or adapter")
    private_base = _personal_base_path(root, personal_root, base_training_config)
    lease = LocalInstanceLease(private_base / "service.lock")
    try:
        bundle, created = ensure_portable_bundle(
            model_directory,
            config_path=config_path,
            backend=selection.backend,
            model_size=resolved_size,
            seed=seed,
        )
        if bundle.model_size != resolved_size:
            raise PortableServeError(
                "The portable model directory contains another size"
            )
        trainer, runtime = _backend_components(
            bundle,
            selection,
            rank=rank,
            learning_mode=learning_mode,
        )
        store = ApprenticeshipStore(_resolve_local_path(root, db_path))
        if learning_mode == "foundation":
            assert isinstance(base_training_config, PersonalFoundationTrainingConfig)
            foundation_config = replace(
                base_training_config,
                artifact_root=private_base,
            )
            coordinator: LocalLearningCoordinator = (
                PersonalFoundationTrainingCoordinator(
                    store,
                    foundation_config,
                    trainer=trainer,
                    runtime=runtime,
                    recover_interrupted=True,
                )
            )
        else:
            assert isinstance(base_training_config, PersonalTrainingConfig)
            profile = personal_profile_id(runtime.personal_runtime_identity())
            private_directory = private_base / "profiles" / profile
            adapter_config = replace(
                base_training_config,
                artifact_root=private_directory,
            )
            coordinator = PersonalTrainingCoordinator(
                store,
                adapter_config,
                trainer=trainer,
                runtime=runtime,
                recover_interrupted=True,
            )
        service = PortableChatService(
            store,
            runtime,
            coordinator,
            bundle=bundle,
            selection=selection,
            max_new_tokens=max_new_tokens,
            bundle_created=created,
            learning_mode=learning_mode,
        )
        coordinator.consider_training()
        return service, lease
    except Exception:
        lease.close()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a private, portable tinydog.ai conversational learner."
    )
    parser.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    parser.add_argument(
        "--model-size", choices=("auto", "tiny", "standard"), default="auto"
    )
    parser.add_argument("--db", default="data/personal-conversations.db")
    parser.add_argument("--model-root", default="data/models")
    parser.add_argument(
        "--personal-root",
        default=None,
        help="override the private learning-artifact directory configured in .env",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--learning-mode",
        choices=("foundation", "adapter"),
        default=os.environ.get("MLM_PERSONAL_LEARNING_MODE", "foundation"),
        help="train full foundation weights by default; adapter is legacy compatibility",
    )
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    service: PortableChatService | None = None
    lease: LocalInstanceLease | None = None
    server: ThreadingHTTPServer | None = None
    try:
        _assert_port_available(args.host, args.port)
        service, lease = build_service(
            root=PROJECT_ROOT,
            backend=args.backend,
            model_size=args.model_size,
            db_path=args.db,
            model_root=args.model_root,
            personal_root=args.personal_root,
            seed=args.seed,
            rank=args.adapter_rank,
            max_new_tokens=args.max_new_tokens,
            learning_mode=args.learning_mode,
        )
        server = make_http_server(
            args.host,
            args.port,
            service,
            api_token=os.environ.get("TINYDOG_LOCAL_API_TOKEN"),
        )

        def request_stop(*_: object) -> None:
            # BaseServer.shutdown must run from a different thread.
            threading.Thread(target=server.shutdown, daemon=True).start()

        for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            value = getattr(signal, signal_name, None)
            if value is not None:
                signal.signal(value, request_stop)
        status = service.status()
        model = status["model"]
        print(
            "tinydog.ai portable learner ready on "
            f"http://{args.host}:{args.port} "
            f"({model['backend']}/{model['device']}, "
            f"{model['parameters']:,} foundation parameters; "
            f"learning={args.learning_mode})",
            flush=True,
        )
        server.serve_forever(poll_interval=0.25)
        return 0
    except (
        OSError,
        ValueError,
        PortableServeError,
        PersonalFoundationError,
        PersonalTrainingError,
    ) as error:
        print(f"tinydog.ai could not start: {error}", flush=True)
        return 1
    finally:
        if server is not None:
            server.server_close()
        if service is not None:
            service.close()
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConversationalRuntime",
    "LocalInstanceLease",
    "PortableChatService",
    "PortableServeError",
    "build_service",
    "choose_model_size",
    "main",
    "make_http_server",
]
