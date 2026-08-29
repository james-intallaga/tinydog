"""Automatic five-round local training for the community floaty layer.

This coordinator deliberately has no reference to the live inference runtime or
the legacy full-weight trainer.  It claims five distinct verified lessons,
runs the signed community client in a child process, and records either an
immutable outgoing contribution or a clean local gate rejection.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator, Mapping
import uuid

from .apprenticeship import (
    ApprenticeshipStore,
    Lesson,
    TRUSTED_HUMAN_DIALOGUE_SOURCE,
    TRUSTED_HUMAN_DIALOGUE_VERIFIER,
    verify_answer,
)
from .community_client import load_training_config
from .community_updates import (
    DELTA_FILENAME,
    MANIFEST_FILENAME,
    UpdateManifest,
)


LOCAL_FLOATY_THRESHOLD = 5
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_DELTA_BYTES = 32 * 1024 * 1024
_MAX_ERROR_TEXT = 2_048


class LocalFloatyError(RuntimeError):
    """Local automatic floaty training could not complete safely."""


@dataclass(frozen=True)
class LocalFloatyAutoConfig:
    """Pinned inputs and private output locations for automatic local training."""

    enabled: bool
    foundation_manifest: Path | None
    owner_public_key: Path
    community_release_manifest: Path | None
    trust_state_path: Path
    candidate_output_dir: Path
    bundle_output_dir: Path
    training_config_path: Path | None = None
    lesson_mode: str = "conversation"

    @classmethod
    def from_env(
        cls, *, project_root: str | Path | None = None
    ) -> "LocalFloatyAutoConfig":
        root = Path(project_root or Path.cwd()).expanduser().resolve()

        def optional_path(name: str) -> Path | None:
            value = os.getenv(name, "").strip()
            if not value:
                return None
            candidate = Path(value).expanduser()
            combined = candidate if candidate.is_absolute() else root / candidate
            return Path(os.path.abspath(combined))

        def path(name: str, default: str) -> Path:
            value = os.getenv(name, default).strip() or default
            candidate = Path(value).expanduser()
            combined = candidate if candidate.is_absolute() else root / candidate
            return Path(os.path.abspath(combined))

        enabled_text = os.getenv("MLM_LOCAL_FLOATY_AUTO", "1").strip().lower()
        if enabled_text not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("MLM_LOCAL_FLOATY_AUTO must be true or false")
        lesson_mode = (
            os.getenv(
                "MLM_LOCAL_FLOATY_LESSON_MODE",
                "conversation",
            )
            .strip()
            .lower()
        )
        if lesson_mode not in {"conversation", "verified", "mixed"}:
            raise ValueError(
                "MLM_LOCAL_FLOATY_LESSON_MODE must be conversation, verified, or mixed"
            )
        return cls(
            enabled=enabled_text in {"1", "true", "yes", "on"},
            foundation_manifest=optional_path("MLM_LOCAL_FOUNDATION_MANIFEST"),
            owner_public_key=path(
                "MLM_LOCAL_OWNER_PUBLIC_KEY",
                "configs/foundation-owner.pub",
            ),
            community_release_manifest=optional_path("MLM_LOCAL_COMMUNITY_RELEASE"),
            trust_state_path=path(
                "MLM_LOCAL_FLOATY_TRUST_STATE",
                "data/local-community-trust.json",
            ),
            candidate_output_dir=path(
                "MLM_LOCAL_FLOATY_CANDIDATES",
                "checkpoints/local-floaty",
            ),
            bundle_output_dir=path("MLM_LOCAL_FLOATY_BUNDLES", "outgoing"),
            training_config_path=optional_path("MLM_LOCAL_FLOATY_TRAINING_CONFIG"),
            lesson_mode=lesson_mode,
        )

    def missing_settings(self) -> tuple[str, ...]:
        """Return only public setting names, never local paths."""

        if not self.enabled:
            return ("MLM_LOCAL_FLOATY_AUTO",)
        checks = (
            ("MLM_LOCAL_FOUNDATION_MANIFEST", self.foundation_manifest),
            ("MLM_LOCAL_OWNER_PUBLIC_KEY", self.owner_public_key),
            ("MLM_LOCAL_COMMUNITY_RELEASE", self.community_release_manifest),
            ("MLM_LOCAL_FLOATY_TRAINING_CONFIG", self.training_config_path),
        )
        missing: list[str] = []
        for name, candidate in checks:
            if candidate is None:
                if name != "MLM_LOCAL_FLOATY_TRAINING_CONFIG":
                    missing.append(name)
                continue
            try:
                details = candidate.lstat()
            except OSError:
                missing.append(name)
                continue
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                missing.append(name)
        return tuple(missing)

    def validate_recipe(self) -> None:
        if self.lesson_mode not in {"conversation", "verified", "mixed"}:
            raise LocalFloatyError(
                "automatic local training lesson mode must be conversation, verified, or mixed"
            )
        settings = load_training_config(self.training_config_path)
        if (
            settings.minimum_training_lessons != 4
            or settings.minimum_validation_lessons != 1
        ):
            raise LocalFloatyError(
                "automatic local training requires exactly four training lessons "
                "and one held-out lesson"
            )


@dataclass(frozen=True)
class LocalFloatyRunRequest:
    run_id: int
    lesson_ids: tuple[int, ...]
    update_id: str
    bundle_path: Path


@dataclass(frozen=True)
class LocalFloatyRunResult:
    accepted: bool
    candidate_report: Path
    bundle_path: Path | None
    training_lesson_ids: tuple[int, ...] = ()
    validation_lesson_ids: tuple[int, ...] = ()
    update_manifest_sha256: str | None = None


LocalFloatyRunner = Callable[[LocalFloatyRunRequest], LocalFloatyRunResult]


class LocalFloatyCoordinator:
    """Serialize safe five-round local floaty contribution attempts."""

    def __init__(
        self,
        store: ApprenticeshipStore,
        config: LocalFloatyAutoConfig,
        *,
        runner: LocalFloatyRunner | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self._lock_path = store.path.with_name(f".{store.path.name}.local-floaty.lock")
        self._runner = runner or self._run_subprocess
        self._mutex = threading.RLock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._lease_fd: int | None = None
        self._shutdown = threading.Event()
        self._rerun_requested = False
        self._state = "blocked" if self._configuration_problem() else "idle"
        self._run_id: int | None = None
        self._update_id: str | None = None
        self._last_bundle: str | None = None
        self._last_candidate: str | None = None
        self._last_error: str | None = None

    def consider_training(self) -> bool:
        """Start one background worker once five distinct lessons qualify."""

        return self._start_if_ready()

    def trigger(self, *, force: bool = False) -> bool:
        # Force means "check now", never "bypass five verified rounds".
        del force
        return self._start_if_ready()

    def rollback(self) -> dict[str, Any]:
        raise ValueError(
            "Community mode has no authority to change the active foundation"
        )

    def shutdown(self) -> None:
        self._shutdown.set()
        with self._mutex:
            process = self._process
            thread = self._thread
        if process is not None and process.poll() is None:
            _terminate_process(process)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)

    def wait_for_idle(self, timeout: float = 10.0) -> bool:
        """Testing/controlled-shutdown seam; normal requests never block on it."""

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            with self._mutex:
                thread = self._thread
            if thread is None or not thread.is_alive():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            thread.join(timeout=remaining)

    def status(self) -> dict[str, Any]:
        qualifying = len(self._select_distinct_pending(limit=None))
        problem = self._configuration_problem()
        with self._mutex:
            running = self._thread is not None and self._thread.is_alive()
            state = self._state
            if problem and not running:
                state = "blocked"
            return {
                "state": state,
                "authority": "community",
                "parameter_scope": "floaty-only",
                "foundation_trainable": False,
                "accepts_community_updates": False,
                "produces_community_updates": True,
                "running": running,
                "pending_lessons": self.store.count_pending_lessons(),
                "qualifying_pending": qualifying,
                "lesson_threshold": LOCAL_FLOATY_THRESHOLD,
                "lesson_mode": self.config.lesson_mode,
                "progress": min(qualifying, LOCAL_FLOATY_THRESHOLD)
                / LOCAL_FLOATY_THRESHOLD,
                "run_id": self._run_id,
                "update_id": self._update_id,
                "last_bundle": self._last_bundle,
                "last_candidate": self._last_candidate,
                "last_error": self._last_error,
                "rollback_available": False,
                "live_model_updated": False,
                "activation": "contribution-packaged-only",
                "intake_validation": "not-performed",
                "missing_settings": list(problem[1]) if problem else [],
                "reason": problem[0] if problem else None,
            }

    def _start_if_ready(self) -> bool:
        if self._shutdown.is_set():
            return False
        problem = self._configuration_problem()
        with self._mutex:
            if problem:
                self._state = "blocked"
                self._last_error = None
                return False
            if self._thread is not None and self._thread.is_alive():
                self._rerun_requested = True
                return False
            if (
                len(self._select_distinct_pending(limit=LOCAL_FLOATY_THRESHOLD))
                < LOCAL_FLOATY_THRESHOLD
                and not self._has_interrupted_run()
            ):
                if self._state not in {"ready", "rejected", "failed"}:
                    self._state = "idle"
                return False
            self._state = "queued"
            self._last_error = None
            self._thread = threading.Thread(
                target=self._worker,
                name="mlm-local-floaty",
                daemon=True,
            )
            self._thread.start()
            return True

    def _worker(self) -> None:
        allow_rerun = True
        try:
            with _process_lock(self._lock_path) as lease_fd:
                with self._mutex:
                    self._lease_fd = lease_fd
                self._recover_interrupted_runs()
                while not self._shutdown.is_set():
                    selected = self._select_distinct_pending(
                        limit=LOCAL_FLOATY_THRESHOLD
                    )
                    if len(selected) < LOCAL_FLOATY_THRESHOLD:
                        with self._mutex:
                            if self._state == "queued":
                                self._state = "idle"
                        break
                    update_id = f"local-{uuid.uuid4().hex}"
                    ids = tuple(lesson.id for lesson in selected)
                    run, claimed = self.store.claim_pending_lesson_ids(
                        ids,
                        checkpoint_in="owner-signed-community-release",
                        metadata={
                            "authority": "community",
                            "parameter_scope": "floaty-only",
                            "foundation_trainable": False,
                            "update_id": update_id,
                            "owner_pid": os.getpid(),
                        },
                    )
                    if run is None or len(claimed) != LOCAL_FLOATY_THRESHOLD:
                        continue
                    request = LocalFloatyRunRequest(
                        run_id=run.id,
                        lesson_ids=tuple(lesson.id for lesson in claimed),
                        update_id=update_id,
                        bundle_path=self.config.bundle_output_dir / update_id,
                    )
                    with self._mutex:
                        self._state = "training"
                        self._run_id = run.id
                        self._update_id = update_id
                    try:
                        result = self._runner(request)
                        _verify_lesson_partition(request, result)
                        _verify_candidate_report(result.candidate_report)
                        with self._mutex:
                            self._state = "packaging" if result.accepted else "rejected"
                        if result.accepted:
                            manifest = _verify_complete_bundle(request, result)
                            self.store.complete_local_floaty_run(
                                run.id,
                                request.lesson_ids,
                                status="contributed",
                                bundle_out=str(request.bundle_path),
                                metrics={"accepted": True},
                                metadata={
                                    "candidate_report": str(result.candidate_report),
                                    "update_manifest_sha256": manifest.manifest_sha256,
                                    "training_lesson_ids": list(
                                        result.training_lesson_ids
                                    ),
                                    "validation_lesson_ids": list(
                                        result.validation_lesson_ids
                                    ),
                                    "live_model_updated": False,
                                    "intake_validation": "not-performed",
                                },
                            )
                            with self._mutex:
                                self._state = "ready"
                                self._last_bundle = request.bundle_path.name
                                self._last_candidate = result.candidate_report.name
                        else:
                            self.store.complete_local_floaty_run(
                                run.id,
                                request.lesson_ids,
                                status="local_rejected",
                                metrics={"accepted": False},
                                metadata={
                                    "candidate_report": str(result.candidate_report),
                                    "training_lesson_ids": list(
                                        result.training_lesson_ids
                                    ),
                                    "validation_lesson_ids": list(
                                        result.validation_lesson_ids
                                    ),
                                    "live_model_updated": False,
                                    "intake_validation": "not-performed",
                                },
                            )
                            with self._mutex:
                                self._state = "rejected"
                                self._last_candidate = result.candidate_report.name
                    except Exception as error:
                        allow_rerun = False
                        self.store.release_training_run(
                            run.id,
                            "failed",
                            metadata={"parameter_scope": "floaty-only"},
                            error=_bounded_error(error),
                        )
                        with self._mutex:
                            self._state = "failed"
                            self._last_error = "Local floaty training failed; the five lessons were kept for retry."
                        print(
                            f"local floaty training failed: {type(error).__name__}: {error}",
                            flush=True,
                        )
                        break
        except BlockingIOError:
            allow_rerun = False
            with self._mutex:
                self._state = "idle"
                self._last_error = (
                    "Another local trainer is already using this lesson database."
                )
        except Exception as error:
            allow_rerun = False
            with self._mutex:
                self._state = "failed"
                self._last_error = "Local floaty training could not start safely."
            print(
                f"local floaty coordinator failed: {type(error).__name__}: {error}",
                flush=True,
            )
        finally:
            with self._mutex:
                rerun = self._rerun_requested and allow_rerun
                self._rerun_requested = False
                self._process = None
                self._lease_fd = None
                self._thread = None
            if rerun and not self._shutdown.is_set():
                self._start_if_ready()

    def _run_subprocess(self, request: LocalFloatyRunRequest) -> LocalFloatyRunResult:
        if self._shutdown.is_set():
            raise LocalFloatyError("local trainer is shutting down")
        foundation = self.config.foundation_manifest
        release = self.config.community_release_manifest
        if foundation is None or release is None:
            raise LocalFloatyError("signed local floaty inputs are not configured")
        command = [
            sys.executable,
            "-m",
            "mini_arithmetic.community_client",
            "train-and-package",
            "--foundation-manifest",
            str(foundation),
            "--owner-public-key",
            str(self.config.owner_public_key),
            "--community-release",
            str(release),
            "--trust-state",
            str(self.config.trust_state_path),
            "--lessons-db",
            str(self.store.path),
            "--candidate-output",
            str(self.config.candidate_output_dir),
            "--bundle",
            str(request.bundle_path),
            "--lesson-limit",
            str(LOCAL_FLOATY_THRESHOLD),
            "--claimed-training-run-id",
            str(request.run_id),
            "--validation-fraction",
            "0.2",
            "--update-id",
            request.update_id,
        ]
        for lesson_id in request.lesson_ids:
            command.extend(("--lesson-id", str(lesson_id)))
        if self.config.training_config_path is not None:
            command.extend(("--training-config", str(self.config.training_config_path)))
        with self._mutex:
            lease_fd = self._lease_fd
        if lease_fd is None:
            raise LocalFloatyError("local trainer child has no database-bound lease")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=(lease_fd,),
        )
        with self._mutex:
            self._process = process
            stop_now = self._shutdown.is_set()
        if stop_now:
            _terminate_process(process)
            raise LocalFloatyError("local trainer stopped during child startup")
        try:
            self.store.update_training_run(
                request.run_id,
                "training",
                metadata={"worker_pid": process.pid},
            )
            stdout, stderr = process.communicate()
        finally:
            if process.poll() is None:
                _terminate_process(process)
            with self._mutex:
                if self._process is process:
                    self._process = None
        if process.returncode != 0:
            detail = (stderr or stdout).strip()[-_MAX_ERROR_TEXT:]
            raise LocalFloatyError(detail or "community client exited unsuccessfully")
        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            raise LocalFloatyError("community client returned no result")
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError as error:
            raise LocalFloatyError(
                "community client result was not valid JSON"
            ) from error
        if not isinstance(payload, dict) or type(payload.get("accepted")) is not bool:
            raise LocalFloatyError("community client result had an invalid schema")
        candidate = payload.get("candidate_report")
        if not isinstance(candidate, str) or not candidate:
            raise LocalFloatyError("community client omitted its candidate report")
        bundle = payload.get("bundle")
        if bundle is not None and (not isinstance(bundle, str) or not bundle):
            raise LocalFloatyError("community client returned an invalid bundle path")
        training_ids = _result_lesson_ids(payload, "training_lesson_ids")
        validation_ids = _result_lesson_ids(payload, "validation_lesson_ids")
        if set(training_ids) | set(validation_ids) != set(request.lesson_ids):
            raise LocalFloatyError(
                "community client did not account for the complete lesson batch"
            )
        if set(training_ids) & set(validation_ids):
            raise LocalFloatyError(
                "community client reused a lesson across train and holdout"
            )
        manifest_sha256 = (
            _result_sha256(payload, "update_manifest_sha256")
            if payload["accepted"]
            else None
        )
        if (
            not payload["accepted"]
            and payload.get("update_manifest_sha256") is not None
        ):
            raise LocalFloatyError(
                "rejected community client returned a manifest identity"
            )
        return LocalFloatyRunResult(
            accepted=payload["accepted"],
            candidate_report=Path(candidate).expanduser().resolve(),
            bundle_path=Path(bundle).expanduser().resolve()
            if bundle is not None
            else None,
            training_lesson_ids=training_ids,
            validation_lesson_ids=validation_ids,
            update_manifest_sha256=manifest_sha256,
        )

    def _configuration_problem(self) -> tuple[str, tuple[str, ...]] | None:
        missing = self.config.missing_settings()
        if missing:
            if not self.config.enabled:
                return ("Automatic local floaty training is disabled.", missing)
            return (
                "Automatic local floaty training is waiting for its owner-signed release.",
                missing,
            )
        try:
            self.config.validate_recipe()
        except (OSError, ValueError, LocalFloatyError) as error:
            return (str(error), ("MLM_LOCAL_FLOATY_TRAINING_CONFIG",))
        return None

    def _select_distinct_pending(self, *, limit: int | None) -> list[Lesson]:
        pending = list(
            reversed(self.store.list_recent_lessons(limit=10_000, pending_only=True))
        )
        selected: list[Lesson] = []
        seen: set[str] = set()
        for lesson in pending:
            key = self._lesson_key(lesson)
            if key is None:
                continue
            if key in seen:
                continue
            seen.add(key)
            selected.append(lesson)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    def _lesson_key(self, lesson: Lesson) -> str | None:
        mode = self.config.lesson_mode
        is_dialogue_marker = (
            lesson.source == TRUSTED_HUMAN_DIALOGUE_SOURCE
            and lesson.verifier == TRUSTED_HUMAN_DIALOGUE_VERIFIER
        )
        if mode in {"conversation", "mixed"} and is_dialogue_marker:
            if not self.store.is_trusted_dialogue_lesson(lesson):
                return None
            prompt = _normalize_dialogue_text(lesson.problem)
            response = _normalize_dialogue_text(lesson.correct_answer)
            return hashlib.sha256(
                f"dialogue\0{prompt}\0{response}".encode("utf-8")
            ).hexdigest()
        if mode == "conversation":
            return None
        return _verified_arithmetic_lesson_key(lesson)

    def _has_interrupted_run(self) -> bool:
        return any(
            run.metadata.get("parameter_scope") == "floaty-only"
            for run in self.store.list_training_runs(status="training")
        )

    def _recover_interrupted_runs(self) -> None:
        # The cross-process lock is held here. Therefore any matching active
        # ledger row belongs to a worker that no longer owns the local lease.
        for run in self.store.list_training_runs(status="training"):
            if run.metadata.get("parameter_scope") != "floaty-only":
                continue
            update_id = run.metadata.get("update_id")
            lesson_ids = run.lesson_ids
            if (
                isinstance(update_id, str)
                and update_id.startswith("local-")
                and len(lesson_ids) == LOCAL_FLOATY_THRESHOLD
                and all(
                    type(identifier) is int and identifier > 0
                    for identifier in lesson_ids
                )
            ):
                bundle_path = self.config.bundle_output_dir / update_id
                request = LocalFloatyRunRequest(
                    run_id=run.id,
                    lesson_ids=tuple(lesson_ids),
                    update_id=update_id,
                    bundle_path=bundle_path,
                )
                recovered = LocalFloatyRunResult(
                    accepted=True,
                    candidate_report=Path("recovered-orphan"),
                    bundle_path=bundle_path.resolve(),
                    training_lesson_ids=tuple(lesson_ids[:4]),
                    validation_lesson_ids=tuple(lesson_ids[4:]),
                )
                try:
                    manifest = _verify_complete_bundle(request, recovered)
                    self.store.complete_local_floaty_run(
                        run.id,
                        lesson_ids,
                        status="contributed",
                        bundle_out=str(bundle_path),
                        metrics={"accepted": True, "recovered": True},
                        metadata={
                            "recovered_orphan_bundle": True,
                            "update_manifest_sha256": manifest.manifest_sha256,
                            "live_model_updated": False,
                            "intake_validation": "not-performed",
                        },
                    )
                    with self._mutex:
                        self._state = "ready"
                        self._last_bundle = bundle_path.name
                    continue
                except (OSError, ValueError, LocalFloatyError):
                    pass
            self.store.release_training_run(
                run.id,
                "failed",
                error="Recovered an interrupted local floaty training run",
                metadata={"recovered": True},
            )


def _verified_arithmetic_lesson_key(lesson: Lesson) -> str | None:
    verification = verify_answer(lesson.problem, lesson.correct_answer)
    if verification.expected is None or not verification.is_correct:
        # Store insertion already enforces this.  A corrupt historical row must
        # never help satisfy the automatic trigger.
        return None
    problem = " ".join(lesson.problem.split()).casefold()
    return hashlib.sha256(
        f"{problem}\0{verification.expected}".encode("utf-8")
    ).hexdigest()


def _normalize_dialogue_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


@contextmanager
def _process_lock(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise LocalFloatyError("local trainer lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _verify_complete_bundle(
    request: LocalFloatyRunRequest,
    result: LocalFloatyRunResult,
) -> UpdateManifest:
    if result.bundle_path != request.bundle_path.resolve():
        raise LocalFloatyError("accepted result did not return the requested bundle")
    bundle = request.bundle_path
    details = bundle.lstat()
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise LocalFloatyError("accepted bundle is not an immutable directory")
    if {entry.name for entry in bundle.iterdir()} != {
        MANIFEST_FILENAME,
        DELTA_FILENAME,
    }:
        raise LocalFloatyError(
            "accepted bundle contents do not match the update protocol"
        )
    manifest_payload = _read_regular_file(
        bundle / MANIFEST_FILENAME, _MAX_MANIFEST_BYTES
    )
    manifest = UpdateManifest.from_bytes(manifest_payload)
    if manifest.update_id != request.update_id:
        raise LocalFloatyError("accepted bundle update identity changed")
    if manifest.delta_bytes > _MAX_DELTA_BYTES:
        raise LocalFloatyError("accepted bundle delta exceeds the local update limit")
    delta = _read_regular_file(bundle / DELTA_FILENAME, manifest.delta_bytes)
    if len(delta) != manifest.delta_bytes:
        raise LocalFloatyError("accepted bundle delta size changed")
    if hashlib.sha256(delta).hexdigest() != manifest.delta_sha256:
        raise LocalFloatyError("accepted bundle delta checksum changed")
    if (
        result.update_manifest_sha256 is not None
        and result.update_manifest_sha256 != manifest.manifest_sha256
    ):
        raise LocalFloatyError(
            "accepted bundle changed after the training child returned"
        )
    return manifest


def _verify_candidate_report(path: Path) -> None:
    try:
        payload = _read_regular_file(path, 4 * 1024 * 1024)
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise LocalFloatyError(
            "local candidate report is missing or invalid"
        ) from error
    if not isinstance(value, dict):
        raise LocalFloatyError("local candidate report root must be an object")


def _read_regular_file(path: Path, maximum: int) -> bytes:
    if maximum <= 0:
        raise LocalFloatyError("invalid artifact size bound")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size > maximum:
            raise LocalFloatyError("local update artifact is unsafe")
        payload = b""
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
        if len(payload) != details.st_size:
            raise LocalFloatyError("local update artifact changed while reading")
        return payload
    finally:
        os.close(descriptor)


def _bounded_error(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}"
    return text[-_MAX_ERROR_TEXT:]


def _verify_lesson_partition(
    request: LocalFloatyRunRequest,
    result: LocalFloatyRunResult,
) -> None:
    training = set(result.training_lesson_ids)
    validation = set(result.validation_lesson_ids)
    if training & validation:
        raise LocalFloatyError(
            "local candidate reused a lesson across train and holdout"
        )
    if training | validation != set(request.lesson_ids):
        raise LocalFloatyError(
            "local candidate did not account for the complete lesson batch"
        )
    if len(training) != 4 or len(validation) != 1:
        raise LocalFloatyError(
            "local candidate must use four train lessons and one holdout"
        )
    if not result.accepted and result.bundle_path is not None:
        raise LocalFloatyError(
            "rejected local candidate unexpectedly returned a bundle"
        )


def _result_lesson_ids(payload: Mapping[str, Any], name: str) -> tuple[int, ...]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise LocalFloatyError(f"community client omitted {name}")
    if any(type(identifier) is not int or identifier <= 0 for identifier in value):
        raise LocalFloatyError(f"community client returned invalid {name}")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise LocalFloatyError(f"community client returned duplicate {name}")
    return result


def _result_sha256(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LocalFloatyError(f"community client returned invalid {name}")
    return value


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, 15)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, 9)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
        process.wait(timeout=5)
