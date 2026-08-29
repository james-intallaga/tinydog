"""Background lesson consolidation, evaluation, promotion, and rollback safety."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
from typing import Any

from .apprenticeship import ApprenticeshipStore
from .continual import CandidateTrainingResult, ContinualTrainingConfig
from .model import ModelConfig
from .runtime import ModelRuntime


@dataclass(frozen=True)
class CoordinatorConfig:
    lesson_threshold: int = 32
    candidate_steps: int = 20
    training_batch_size: int = 1
    training_context: int = 256
    evaluation_batches: int = 3
    replay_ratio: float = 0.5
    min_new_loss_improvement: float = 1e-4
    max_old_loss_regression: float = 0.08
    max_lessons_per_run: int = 256
    checkpoint_retention: int = 3

    @classmethod
    def from_env(cls) -> "CoordinatorConfig":
        return cls(
            lesson_threshold=max(1, int(os.getenv("MLM_AUTO_TRAIN_LESSONS", "32"))),
            candidate_steps=max(1, int(os.getenv("MLM_CANDIDATE_STEPS", "20"))),
            training_batch_size=max(1, int(os.getenv("MLM_TRAIN_BATCH_SIZE", "1"))),
            training_context=max(16, int(os.getenv("MLM_TRAIN_CONTEXT", "256"))),
            evaluation_batches=max(1, int(os.getenv("MLM_EVALUATION_BATCHES", "3"))),
            min_new_loss_improvement=max(
                0.0,
                float(os.getenv("MLM_MIN_NEW_LOSS_IMPROVEMENT", "0.0001")),
            ),
            checkpoint_retention=max(
                1, int(os.getenv("MLM_CHECKPOINT_RETENTION", "3"))
            ),
        )


def resolve_active_checkpoint(
    initial_weights: str | Path,
    pointer_path: str | Path,
    config_path: str | Path,
) -> Path:
    """Resume the latest promoted checkpoint when its config still matches."""
    initial = Path(initial_weights).expanduser().resolve()
    pointer = Path(pointer_path)
    if not pointer.is_file():
        return initial
    try:
        payload = json.loads(pointer.read_text())
        candidate = Path(payload["checkpoint"]).expanduser().resolve()
        recorded_config = Path(payload["config"]).expanduser().resolve()
        resolved_config = Path(config_path).expanduser().resolve()
        config_hash = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
        signature = ModelConfig.from_json(resolved_config).to_dict()
        if (
            candidate.is_file()
            and recorded_config == resolved_config
            and payload.get("config_sha256") == config_hash
            and payload.get("model_signature") == signature
            and _checkpoint_matches_config(candidate, signature)
        ):
            return candidate
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        OSError,
        struct.error,
    ):
        pass
    return initial


def _checkpoint_matches_config(checkpoint: Path, signature: dict[str, Any]) -> bool:
    """Inspect safetensors metadata without allocating the checkpoint tensors."""
    with checkpoint.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size <= 0 or header_size > 64 * 1024 * 1024:
            return False
        header = json.loads(stream.read(header_size))
    expected = {
        "token_embedding.weight": [signature["vocab_size"], signature["model_dim"]],
        "position_embedding.weight": [
            signature["context_length"],
            signature["model_dim"],
        ],
        "blocks.0.attention.qkv.weight": [
            3 * signature["model_dim"],
            signature["model_dim"],
        ],
        "blocks.0.feed_forward.gate.weight": [
            signature["ffn_dim"],
            signature["model_dim"],
        ],
        f"blocks.{signature['num_layers'] - 1}.attention.qkv.weight": [
            3 * signature["model_dim"],
            signature["model_dim"],
        ],
    }
    return all(
        header.get(name, {}).get("shape") == shape for name, shape in expected.items()
    )


class LearningCoordinator:
    """Owner-only legacy full-model consolidation for ``ArithmeticLM``.

    This coordinator predates the foundation/floaty split.  It is intentionally
    never used for community training: public-safe service startup selects the
    separate ``LocalFloatyCoordinator``, which can package floaty-only updates
    but has no reference to this full-model path.
    """

    def __init__(
        self,
        store: ApprenticeshipStore,
        runtime: ModelRuntime,
        model_config_path: str | Path,
        output_dir: str | Path,
        *,
        config: CoordinatorConfig | None = None,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.model_config_path = str(Path(model_config_path).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pointer_path = self.output_dir / "active.json"
        self.config = config or CoordinatorConfig.from_env()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._state: dict[str, Any] = {
            "state": "idle",
            "last_result": None,
            "error": None,
        }
        self._recover_orphaned_runs()

    def consider_training(self) -> bool:
        return self.trigger(force=False)

    def trigger(self, *, force: bool = False) -> bool:
        pending = self.store.count_pending_lessons()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            if pending == 0 or (not force and pending < self.config.lesson_threshold):
                return False
            self._state = {
                "state": "queued",
                "last_result": self._state.get("last_result"),
                "error": None,
            }
            self._thread = threading.Thread(
                target=self._run, name="mlm-consolidation", daemon=True
            )
            self._thread.start()
            return True

    def _run(self) -> None:
        checkpoint_in = self.runtime.weights_path
        run, lessons = self.store.claim_pending_lessons(
            limit=self.config.max_lessons_per_run,
            checkpoint_in=checkpoint_in,
            metadata={"coordinator": asdict(self.config), "owner_pid": os.getpid()},
        )
        if run is None or not lessons:
            with self._lock:
                self._state = {
                    "state": "idle",
                    "last_result": self._state.get("last_result"),
                    "error": None,
                }
                self._thread = None
            return
        run_id = run.id
        lesson_ids = [lesson.id for lesson in lessons]
        highest_claimed_id = max(lesson_ids)
        lesson_records: list[dict[str, Any]] = []
        for lesson in lessons:
            record = asdict(lesson)
            record["practice_variants"] = [
                {
                    "problem": practice.problem,
                    "correct_answer": practice.answer,
                    "metadata": practice.metadata,
                }
                for practice in self.store.list_practice(lesson.id, limit=5)
            ]
            lesson_records.append(record)
        with self._lock:
            self._state = {
                "state": "training",
                "run_id": run_id,
                "last_result": self._state.get("last_result"),
                "error": None,
            }
        try:
            training_config = ContinualTrainingConfig(
                steps=self.config.candidate_steps,
                batch_size=self.config.training_batch_size,
                context_length=min(
                    self.config.training_context,
                    self.runtime.model.config.context_length,
                ),
                evaluation_batches=self.config.evaluation_batches,
                replay_ratio=self.config.replay_ratio,
                min_new_loss_improvement=self.config.min_new_loss_improvement,
                max_old_loss_regression=self.config.max_old_loss_regression,
            )
            result = self._run_candidate_worker(
                run_id, checkpoint_in, lesson_records, training_config
            )
            final_state = "rejected"
            if result.promotion_eligible:
                prepared = self.runtime.prepare_checkpoint(result.weights_path)
                previous_pointer = (
                    self.pointer_path.read_bytes()
                    if self.pointer_path.is_file()
                    else None
                )
                self._write_pointer(result.weights_path, result.run_id, run_id)
                try:
                    self.store.promote_training_run(
                        run_id,
                        result.trained_lesson_ids,
                        checkpoint_out=str(result.weights_path),
                        metrics=result.report.get("metrics", {}),
                        metadata={
                            "report_path": str(result.report_path),
                            "gate": result.report.get("gate", {}),
                        },
                    )
                except Exception:
                    self._restore_pointer(previous_pointer)
                    raise
                self.runtime.activate_checkpoint(prepared, result.weights_path)
                final_state = "promoted"
            else:
                self.store.release_training_run(
                    run_id,
                    status=final_state,
                    checkpoint_out=str(result.weights_path),
                    metrics=result.report.get("metrics", {}),
                    metadata={
                        "report_path": str(result.report_path),
                        "gate": result.report.get("gate", {}),
                    },
                )
            self._prune_candidate_weights()
            with self._lock:
                self._state = {
                    "state": "idle",
                    "last_result": {
                        "status": final_state,
                        "run_id": run_id,
                        "candidate_run_id": result.run_id,
                        "checkpoint": str(result.weights_path),
                        "metrics": result.report.get("metrics", {}),
                    },
                    "error": None,
                }
        except Exception as error:
            self.store.release_training_run(run_id, status="failed", error=str(error))
            with self._lock:
                self._state = {
                    "state": "idle",
                    "last_result": self._state.get("last_result"),
                    "error": str(error),
                }
        finally:
            pending = self.store.list_recent_lessons(
                limit=self.config.max_lessons_per_run,
                pending_only=True,
            )
            should_requeue = (
                self.store.count_pending_lessons() >= self.config.lesson_threshold
                and any(lesson.id > highest_claimed_id for lesson in pending)
            )
            with self._lock:
                self._thread = None
            if should_requeue:
                self.trigger(force=False)

    def _run_candidate_worker(
        self,
        run_id: int,
        checkpoint_in: str,
        lessons: list[dict[str, Any]],
        training_config: ContinualTrainingConfig,
    ) -> CandidateTrainingResult:
        jobs_dir = self.output_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_path = jobs_dir / f"run-{run_id}.json"
        result_path = jobs_dir / f"run-{run_id}-result.json"
        temporary = job_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "active_weights": checkpoint_in,
                    "model_config": self.model_config_path,
                    "lessons": lessons,
                    "output_dir": str(self.output_dir),
                    "training_config": asdict(training_config),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        temporary.replace(job_path)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mini_arithmetic.worker",
                "--job",
                str(job_path),
                "--result",
                str(result_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._process = process
        self.store.update_training_run(
            run_id,
            status="training",
            metadata={"worker_pid": process.pid, "job_path": str(job_path)},
        )
        stdout, stderr = process.communicate()
        with self._lock:
            self._process = None
        if process.returncode != 0:
            detail = (stderr or stdout or "candidate worker failed").strip()[-2_000:]
            raise RuntimeError(detail)
        if not result_path.is_file():
            raise RuntimeError("Candidate worker did not write a result")
        payload = json.loads(result_path.read_text())
        report_path = Path(payload["report_path"])
        report = json.loads(report_path.read_text())
        job_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        return CandidateTrainingResult(
            run_id=str(payload["run_id"]),
            weights_path=Path(payload["weights_path"]),
            report_path=report_path,
            promotion_eligible=bool(payload["promotion_eligible"]),
            trained_lesson_ids=tuple(payload.get("trained_lesson_ids", ())),
            report=report,
        )

    def shutdown(self) -> None:
        """Stop an isolated candidate worker before the service exits."""
        with self._lock:
            process = self._process
            thread = self._thread
        if process is not None and process.poll() is None:
            process.terminate()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)
        if process is not None and process.poll() is None:
            process.kill()

    def rollback(self) -> dict[str, Any]:
        """Return to the previous active checkpoint and requeue its lessons."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("Cannot roll back while candidate training is active")
        if not self.pointer_path.is_file():
            raise ValueError("No promoted checkpoint has rollback history")
        previous_pointer = self.pointer_path.read_bytes()
        payload = json.loads(previous_pointer)
        history = payload.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError("No previous checkpoint is available")
        target = history.pop()
        target_path = Path(target["checkpoint"]).expanduser().resolve()
        signature = ModelConfig.from_json(self.model_config_path).to_dict()
        if not target_path.is_file() or not _checkpoint_matches_config(
            target_path, signature
        ):
            raise ValueError("Previous checkpoint is missing or incompatible")
        prepared = self.runtime.prepare_checkpoint(target_path)
        current_run_id = payload.get("training_run_id")
        payload.update(
            {
                "checkpoint": str(target_path),
                "candidate_run_id": target.get("candidate_run_id"),
                "training_run_id": target.get("training_run_id"),
                "history": history,
                "rolled_back_from": self.runtime.weights_path,
            }
        )
        self._write_pointer_payload(payload)
        try:
            if isinstance(current_run_id, int):
                self.store.rollback_training_run(current_run_id)
        except Exception:
            self._restore_pointer(previous_pointer)
            raise
        self.runtime.activate_checkpoint(prepared, target_path)
        result = {"status": "rolled_back", "checkpoint": str(target_path)}
        with self._lock:
            self._state = {"state": "idle", "last_result": result, "error": None}
        return result

    def _recover_orphaned_runs(self) -> None:
        for run in self.store.list_training_runs(status="training"):
            owner_pid = run.metadata.get("owner_pid")
            if isinstance(owner_pid, int) and self._pid_is_alive(owner_pid):
                continue
            worker_pid = run.metadata.get("worker_pid")
            if isinstance(worker_pid, int) and self._is_training_worker(worker_pid):
                try:
                    os.kill(worker_pid, 15)
                except OSError:
                    pass
            self.store.release_training_run(
                run.id,
                status="failed",
                error="Recovered an interrupted candidate-training run",
            )

    def _prune_candidate_weights(self) -> None:
        try:
            weights = sorted(
                self.output_dir.glob("candidate-*/*.safetensors"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        active = Path(self.runtime.weights_path).resolve()
        keep = {path.resolve() for path in weights[: self.config.checkpoint_retention]}
        keep.add(active)
        for path in weights:
            if path.resolve() not in keep:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _is_training_worker(pid: int) -> bool:
        if not LearningCoordinator._pid_is_alive(pid):
            return False
        try:
            completed = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return "mini_arithmetic.worker" in completed.stdout

    def _write_pointer(
        self,
        checkpoint: str | Path,
        candidate_run_id: str,
        training_run_id: int,
    ) -> None:
        config_path = Path(self.model_config_path)
        existing: dict[str, Any] = {}
        if self.pointer_path.is_file():
            try:
                parsed = json.loads(self.pointer_path.read_text())
                if isinstance(parsed, dict):
                    existing = parsed
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing_history = existing.get("history", [])
        history = list(existing_history) if isinstance(existing_history, list) else []
        current_checkpoint = str(Path(self.runtime.weights_path).resolve())
        if current_checkpoint != str(Path(checkpoint).resolve()):
            history.append(
                {
                    "checkpoint": current_checkpoint,
                    "candidate_run_id": existing.get("candidate_run_id"),
                    "training_run_id": existing.get("training_run_id"),
                }
            )
        history = history[-self.config.checkpoint_retention :]
        payload = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "config": self.model_config_path,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "model_signature": ModelConfig.from_json(config_path).to_dict(),
            "candidate_run_id": candidate_run_id,
            "training_run_id": training_run_id,
            "history": history,
        }
        self._write_pointer_payload(payload)

    def _write_pointer_payload(self, payload: dict[str, Any]) -> None:
        temporary = self.pointer_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.pointer_path)

    def _restore_pointer(self, previous: bytes | None) -> None:
        temporary = self.pointer_path.with_suffix(".rollback.tmp")
        if previous is None:
            self.pointer_path.unlink(missing_ok=True)
            return
        temporary.write_bytes(previous)
        temporary.replace(self.pointer_path)

    def status(self) -> dict[str, Any]:
        pending = self.store.count_pending_lessons()
        with self._lock:
            state = dict(self._state)
            running = self._thread is not None and self._thread.is_alive()
        rollback_available = False
        if self.pointer_path.is_file():
            try:
                pointer = json.loads(self.pointer_path.read_text())
                rollback_available = bool(pointer.get("history"))
            except (json.JSONDecodeError, OSError):
                pass
        return {
            **state,
            "authority": "owner",
            "parameter_scope": "legacy-full-model-owner-only",
            "foundation_trainable": True,
            "accepts_community_updates": False,
            "running": running,
            "pending_lessons": pending,
            "lesson_threshold": self.config.lesson_threshold,
            "candidate_steps": self.config.candidate_steps,
            "rollback_available": rollback_available,
        }


class CommunityTrainingBoundary:
    """Fail closed instead of exposing the legacy full-model trainer publicly.

    Conversation and verified lessons continue to be stored locally.  Weight
    training is deliberately unavailable here because the live ``ArithmeticLM``
    has no structural foundation/floaty parameter partition.  Community
    clients use ``FoundationFloatyLM`` and the isolated floaty tooling instead.
    """

    def __init__(self, store: ApprenticeshipStore) -> None:
        self.store = store

    def consider_training(self) -> bool:
        return False

    def trigger(self, *, force: bool = False) -> bool:
        del force
        raise ValueError(
            "Community training cannot use the legacy full-model runtime; "
            "use an owner-signed FoundationFloatyLM release and floaty-only trainer"
        )

    def rollback(self) -> dict[str, Any]:
        raise ValueError(
            "Community mode has no authority to change the active foundation"
        )

    def shutdown(self) -> None:
        return

    def status(self) -> dict[str, Any]:
        return {
            "state": "blocked",
            "authority": "community",
            "parameter_scope": "floaty-only",
            "foundation_trainable": False,
            "accepts_community_updates": False,
            "running": False,
            "pending_lessons": self.store.count_pending_lessons(),
            "rollback_available": False,
            "reason": (
                "The live ArithmeticLM has no foundation/floaty partition. "
                "Its full-model trainer is disabled in community mode."
            ),
        }
