"""Durable conversations, exact verification, lessons, memory, and replay state."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import uuid
from typing import Any, Iterable

from .data import split_for_key


TRUSTED_HUMAN_DIALOGUE_VERIFIER = "trusted-human-dialogue-v1"
TRUSTED_HUMAN_DIALOGUE_SOURCE = "human-conversation"
TRUSTED_HUMAN_DIALOGUE_SKILL = "conversation"
TRUSTED_DIALOGUE_BOOTSTRAP_PROMPT = "Start a natural conversation."
_FOUNDATION_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


def _foundation_profile(value: str) -> str:
    if not isinstance(value, str) or _FOUNDATION_PROFILE_RE.fullmatch(value) is None:
        raise ValueError("foundation_profile_id is invalid")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _load(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


@dataclass(frozen=True)
class Session:
    id: str
    created_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Message:
    id: int
    session_id: str
    role: str
    content: str
    reply_to_id: int | None
    created_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Lesson:
    id: int
    session_id: str | None
    problem_message_id: int | None
    assistant_message_id: int | None
    correction_message_id: int | None
    problem: str
    first_attempt: str
    correct_answer: str
    feedback: str
    skill: str
    source: str
    verifier: str
    status: str
    created_at: str
    used_at: str | None
    training_run_id: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Practice:
    id: int
    lesson_id: int
    problem: str
    answer: str
    created_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingRun:
    id: int
    status: str
    lesson_ids: list[int | str]
    checkpoint_in: str | None
    checkpoint_out: str | None
    metrics: dict[str, Any]
    error: str | None
    created_at: str
    updated_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VerificationResult:
    is_correct: bool
    expected: str | None
    attempted: str | None
    expression: str | None
    error: str | None


@dataclass(frozen=True)
class Correction:
    correct_answer: str
    feedback: str
    verified: bool


class ApprenticeshipStore:
    """Small local ledger; each operation owns its SQLite connection."""

    def __init__(self, db_path: str | Path) -> None:
        candidate = Path(db_path).expanduser()
        if candidate.is_symlink():
            raise ValueError(
                "The private conversation database cannot be a symbolic link"
            )
        self.path = candidate.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            self.path.parent.chmod(0o700)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        self._harden_permissions()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        self._harden_permissions()
        return connection

    def _harden_permissions(self) -> None:
        if os.name == "nt":
            return
        self.path.parent.chmod(0o700)
        for candidate in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            try:
                candidate.chmod(0o600)
            except FileNotFoundError:
                continue

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    reply_to_id INTEGER REFERENCES messages(id),
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS messages_session_id ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    problem_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                    assistant_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                    correction_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
                    problem TEXT NOT NULL,
                    first_attempt TEXT NOT NULL,
                    correct_answer TEXT NOT NULL,
                    feedback TEXT NOT NULL DEFAULT '',
                    skill TEXT NOT NULL DEFAULT 'arithmetic',
                    source TEXT NOT NULL,
                    verifier TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    used_at TEXT,
                    training_run_id INTEGER,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS lessons_status ON lessons(status, id);
                CREATE INDEX IF NOT EXISTS lessons_dialogue_provenance
                    ON lessons(correction_message_id, verifier);
                CREATE INDEX IF NOT EXISTS lessons_training_status
                    ON lessons(training_run_id, status);
                CREATE TABLE IF NOT EXISTS practice (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lesson_id INTEGER NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
                    problem TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(lesson_id, problem)
                );
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    lesson_ids TEXT NOT NULL DEFAULT '[]',
                    checkpoint_in TEXT,
                    checkpoint_out TEXT,
                    metrics TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS training_runs_status_id
                    ON training_runs(status, id);
                CREATE INDEX IF NOT EXISTS training_runs_foundation_profile_status_id
                    ON training_runs(
                        json_extract(metadata, '$.authority'),
                        json_extract(metadata, '$.foundation_profile_id'),
                        status,
                        id
                    );
                """
            )
            self._harden_permissions()

    def create_session(
        self, session_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> Session:
        identifier = session_id or uuid.uuid4().hex
        created = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, metadata) VALUES (?, ?, ?)",
                (identifier, created, _dump(metadata)),
            )
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (identifier,)
            ).fetchone()
        return self._session(row)

    def get_session(self, session_id: str) -> Session | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._session(row) if row else None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        reply_to_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        if role not in {"user", "assistant", "teacher", "memory", "system"}:
            raise ValueError(f"Unsupported message role: {role}")
        self.create_session(session_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages(session_id, role, content, reply_to_id, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, reply_to_id, _now(), _dump(metadata)),
            )
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._message(row)

    def list_messages(
        self, session_id: str, *, limit: int | None = None, before_id: int | None = None
    ) -> list[Message]:
        clauses, parameters = ["session_id = ?"], [session_id]
        if before_id is not None:
            clauses.append("id < ?")
            parameters.append(before_id)
        sql = f"SELECT * FROM messages WHERE {' AND '.join(clauses)} ORDER BY id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._message(row) for row in reversed(rows)]

    def update_message_metadata(
        self, message_id: int, metadata: dict[str, Any]
    ) -> Message:
        """Merge durable UI and verification facts into an existing message."""
        with self._connect() as connection:
            current = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            if current is None:
                raise ValueError("Message not found")
            merged = _load(current["metadata"])
            merged.update(metadata)
            connection.execute(
                "UPDATE messages SET metadata = ? WHERE id = ?",
                (_dump(merged), message_id),
            )
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        return self._message(row)

    def find_preceding_problem(self, assistant_message_id: int) -> Message | None:
        with self._connect() as connection:
            assistant = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (assistant_message_id,)
            ).fetchone()
            if not assistant:
                return None
            row = None
            if assistant["reply_to_id"] is not None:
                row = connection.execute(
                    "SELECT * FROM messages WHERE id = ? AND role = 'user'",
                    (assistant["reply_to_id"],),
                ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT * FROM messages WHERE session_id = ? AND role = 'user' AND id < ? ORDER BY id DESC LIMIT 1",
                    (assistant["session_id"], assistant_message_id),
                ).fetchone()
        return self._message(row) if row else None

    def add_verified_lesson(
        self,
        *,
        problem: str,
        first_attempt: str,
        correct_answer: str,
        feedback: str = "",
        skill: str = "",
        source: str = "human",
        verifier: str = "exact-arithmetic-v1",
        session_id: str | None = None,
        problem_message_id: int | None = None,
        assistant_message_id: int | None = None,
        correction_message_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Lesson:
        verification = verify_answer(problem, correct_answer)
        if verification.expected is None or not verification.is_correct:
            raise ValueError(
                verification.error
                or f"Correction is not verified; expected {verification.expected}"
            )
        canonical = verification.expected
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if assistant_message_id is not None:
                existing = connection.execute(
                    "SELECT * FROM lessons WHERE assistant_message_id = ? AND correct_answer = ? ORDER BY id DESC LIMIT 1",
                    (assistant_message_id, canonical),
                ).fetchone()
                if existing:
                    merged_metadata = _load(existing["metadata"])
                    merged_metadata.update(metadata or {})
                    human_teaching = (
                        source.startswith("human") or correction_message_id is not None
                    )
                    has_new_teaching = human_teaching or bool(feedback.strip())
                    if existing["status"] != "training" or not has_new_teaching:
                        next_status = (
                            "pending" if has_new_teaching else existing["status"]
                        )
                        next_feedback = feedback.strip() or existing["feedback"]
                        next_source = (
                            source
                            if human_teaching or existing["source"] == "exact-verifier"
                            else existing["source"]
                        )
                        connection.execute(
                            """UPDATE lessons SET
                               session_id = COALESCE(?, session_id),
                               problem_message_id = COALESCE(?, problem_message_id),
                               correction_message_id = COALESCE(?, correction_message_id),
                               feedback = ?, skill = ?, source = ?, status = ?,
                               used_at = CASE WHEN ? = 'pending' THEN NULL ELSE used_at END,
                               training_run_id = CASE WHEN ? = 'pending' THEN NULL ELSE training_run_id END,
                               metadata = ?
                               WHERE id = ?""",
                            (
                                session_id,
                                problem_message_id,
                                correction_message_id,
                                next_feedback,
                                skill or existing["skill"],
                                next_source,
                                next_status,
                                next_status,
                                next_status,
                                _dump(merged_metadata),
                                existing["id"],
                            ),
                        )
                        updated = connection.execute(
                            "SELECT * FROM lessons WHERE id = ?", (existing["id"],)
                        ).fetchone()
                        return self._lesson(updated)
            cursor = connection.execute(
                """INSERT INTO lessons(
                    session_id, problem_message_id, assistant_message_id, correction_message_id,
                    problem, first_attempt, correct_answer, feedback, skill, source, verifier,
                    status, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    session_id,
                    problem_message_id,
                    assistant_message_id,
                    correction_message_id,
                    problem,
                    first_attempt,
                    canonical,
                    feedback,
                    skill or "arithmetic",
                    source,
                    verifier,
                    _now(),
                    _dump(metadata),
                ),
            )
            row = connection.execute(
                "SELECT * FROM lessons WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._lesson(row)

    def add_trusted_dialogue_lesson(
        self,
        *,
        human_response_message_id: int,
        prompt_message_id: int | None = None,
        consent_local_training: bool,
        metadata: dict[str, Any] | None = None,
    ) -> Lesson:
        """Record a human-authored conversational target with durable provenance.

        A normal target is the first human turn following the latest assistant
        message in the same session.  The first turn of a session may instead
        use the one fixed bootstrap prompt.  Callers cannot supply the answer,
        source, verifier, or trust flags: those values are derived from the
        persisted messages here.
        """

        if consent_local_training is not True:
            raise ValueError("Local dialogue training requires explicit consent")
        if (
            isinstance(human_response_message_id, bool)
            or not isinstance(human_response_message_id, int)
            or human_response_message_id <= 0
        ):
            raise ValueError("human_response_message_id must be a positive integer")
        if prompt_message_id is not None and (
            isinstance(prompt_message_id, bool)
            or not isinstance(prompt_message_id, int)
            or prompt_message_id <= 0
        ):
            raise ValueError("prompt_message_id must be a positive integer")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._insert_trusted_dialogue_lesson(
                connection,
                human_response_message_id=human_response_message_id,
                prompt_message_id=prompt_message_id,
                metadata=metadata,
            )
        return self._lesson(row)

    def add_trusted_dialogue_turn(
        self,
        session_id: str,
        human_content: str,
        assistant_content: str,
        *,
        prompt_message_id: int | None = None,
        human_metadata: dict[str, Any] | None = None,
        lesson_metadata: dict[str, Any] | None = None,
        assistant_metadata: dict[str, Any] | None = None,
        foundation_profile_id: str | None = None,
    ) -> tuple[Message, Lesson, Message]:
        """Atomically save one human target, its lesson, and the model reply.

        This is the portable chat commit boundary.  If provenance validation,
        lesson insertion, or assistant insertion fails, SQLite rolls the
        entire turn back so the next request cannot inherit a half-written
        dialogue.
        """

        identifier = session_id.strip() if isinstance(session_id, str) else ""
        human = human_content.strip() if isinstance(human_content, str) else ""
        assistant = (
            assistant_content.strip() if isinstance(assistant_content, str) else ""
        )
        if not identifier:
            raise ValueError("session_id is required")
        if not human:
            raise ValueError("Dialogue target cannot be empty")
        if not assistant:
            raise ValueError("Assistant dialogue response cannot be empty")
        if prompt_message_id is not None and (
            isinstance(prompt_message_id, bool)
            or not isinstance(prompt_message_id, int)
            or prompt_message_id <= 0
        ):
            raise ValueError("prompt_message_id must be a positive integer")
        if foundation_profile_id is not None:
            _foundation_profile(foundation_profile_id)

        created = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            scoped_lesson_metadata = dict(lesson_metadata or {})
            if foundation_profile_id is not None:
                existing_scope = scoped_lesson_metadata.get("foundation_profile_id")
                if (
                    existing_scope is not None
                    and existing_scope != foundation_profile_id
                ):
                    raise ValueError("lesson foundation profile scope changed")
                scoped_lesson_metadata["foundation_profile_id"] = foundation_profile_id
            connection.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, metadata) VALUES (?, ?, '{}')",
                (identifier, created),
            )
            user_cursor = connection.execute(
                """INSERT INTO messages(
                    session_id, role, content, reply_to_id, created_at, metadata
                ) VALUES (?, 'user', ?, ?, ?, ?)""",
                (
                    identifier,
                    human,
                    prompt_message_id,
                    created,
                    _dump(human_metadata),
                ),
            )
            user_id = int(user_cursor.lastrowid)
            lesson_row = self._insert_trusted_dialogue_lesson(
                connection,
                human_response_message_id=user_id,
                prompt_message_id=prompt_message_id,
                metadata=scoped_lesson_metadata,
            )
            assistant_cursor = connection.execute(
                """INSERT INTO messages(
                    session_id, role, content, reply_to_id, created_at, metadata
                ) VALUES (?, 'assistant', ?, ?, ?, ?)""",
                (
                    identifier,
                    assistant,
                    user_id,
                    _now(),
                    _dump(assistant_metadata),
                ),
            )
            user_row = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (user_id,)
            ).fetchone()
            assistant_row = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (assistant_cursor.lastrowid,)
            ).fetchone()
        return (
            self._message(user_row),
            self._lesson(lesson_row),
            self._message(assistant_row),
        )

    def _insert_trusted_dialogue_lesson(
        self,
        connection: sqlite3.Connection,
        *,
        human_response_message_id: int,
        prompt_message_id: int | None,
        metadata: dict[str, Any] | None,
    ) -> sqlite3.Row:
        response = connection.execute(
            "SELECT * FROM messages WHERE id = ?",
            (human_response_message_id,),
        ).fetchone()
        if response is None or response["role"] != "user":
            raise ValueError("Dialogue target must be an existing human user message")
        human_response = response["content"].strip()
        if not human_response:
            raise ValueError("Dialogue target cannot be empty")

        bootstrap_prompt = prompt_message_id is None
        if bootstrap_prompt:
            prior_dialogue = connection.execute(
                """SELECT id FROM messages
                   WHERE session_id = ? AND id < ? AND role IN ('user', 'assistant')
                   ORDER BY id DESC LIMIT 1""",
                (response["session_id"], human_response_message_id),
            ).fetchone()
            if prior_dialogue is not None:
                raise ValueError(
                    "The fixed dialogue bootstrap is only valid for the first turn"
                )
            prompt_text = TRUSTED_DIALOGUE_BOOTSTRAP_PROMPT
        else:
            prompt = connection.execute(
                "SELECT * FROM messages WHERE id = ?",
                (prompt_message_id,),
            ).fetchone()
            if (
                prompt is None
                or prompt["role"] != "assistant"
                or prompt["session_id"] != response["session_id"]
                or prompt["id"] >= response["id"]
            ):
                raise ValueError(
                    "Dialogue target must follow an assistant prompt in the same session"
                )
            latest_assistant = connection.execute(
                """SELECT id FROM messages
                   WHERE session_id = ? AND role = 'assistant' AND id < ?
                   ORDER BY id DESC LIMIT 1""",
                (response["session_id"], response["id"]),
            ).fetchone()
            first_human_response = connection.execute(
                """SELECT id FROM messages
                   WHERE session_id = ? AND role = 'user' AND id > ? AND id <= ?
                   ORDER BY id ASC LIMIT 1""",
                (response["session_id"], prompt["id"], response["id"]),
            ).fetchone()
            if (
                latest_assistant is None
                or latest_assistant["id"] != prompt["id"]
                or first_human_response is None
                or first_human_response["id"] != response["id"]
            ):
                raise ValueError(
                    "Dialogue target must be the first human response to the latest assistant turn"
                )
            prompt_text = prompt["content"].strip()
            if not prompt_text:
                raise ValueError("Assistant dialogue prompt cannot be empty")

        existing = connection.execute(
            """SELECT * FROM lessons
               WHERE correction_message_id = ? AND verifier = ?
               ORDER BY id ASC LIMIT 1""",
            (human_response_message_id, TRUSTED_HUMAN_DIALOGUE_VERIFIER),
        ).fetchone()
        if existing is not None:
            return existing

        trusted_metadata = dict(metadata or {})
        trusted_metadata.update(
            {
                "bootstrap_prompt": bootstrap_prompt,
                "consent_local_training": True,
                "human_authored_target": True,
                "dialogue_provenance": {
                    "prompt_message_id": prompt_message_id,
                    "response_message_id": human_response_message_id,
                    "version": 1,
                },
            }
        )
        cursor = connection.execute(
            """INSERT INTO lessons(
                session_id, problem_message_id, assistant_message_id,
                correction_message_id, problem, first_attempt, correct_answer,
                feedback, skill, source, verifier, status, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, '', ?, '', ?, ?, ?, 'pending', ?, ?)""",
            (
                response["session_id"],
                prompt_message_id,
                prompt_message_id,
                human_response_message_id,
                prompt_text,
                human_response,
                TRUSTED_HUMAN_DIALOGUE_SKILL,
                TRUSTED_HUMAN_DIALOGUE_SOURCE,
                TRUSTED_HUMAN_DIALOGUE_VERIFIER,
                _now(),
                _dump(trusted_metadata),
            ),
        )
        row = connection.execute(
            "SELECT * FROM lessons WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        if row is None:  # pragma: no cover - SQLite guarantees RETURNING row identity.
            raise RuntimeError("Failed to store the trusted dialogue lesson")
        return row

    def is_trusted_dialogue_lesson(self, lesson: Lesson) -> bool:
        """Revalidate that a dialogue lesson still matches its source messages."""

        if not isinstance(lesson, Lesson):
            return False
        if (
            lesson.verifier != TRUSTED_HUMAN_DIALOGUE_VERIFIER
            or lesson.source != TRUSTED_HUMAN_DIALOGUE_SOURCE
            or lesson.skill != TRUSTED_HUMAN_DIALOGUE_SKILL
            or lesson.first_attempt != ""
            or not lesson.session_id
            or not lesson.problem.strip()
            or not lesson.correct_answer.strip()
            or lesson.metadata.get("human_authored_target") is not True
            or lesson.metadata.get("consent_local_training") is not True
        ):
            return False
        provenance = lesson.metadata.get("dialogue_provenance")
        if not isinstance(provenance, dict) or provenance.get("version") != 1:
            return False
        if provenance.get("response_message_id") != lesson.correction_message_id:
            return False
        if (
            isinstance(lesson.correction_message_id, bool)
            or not isinstance(lesson.correction_message_id, int)
            or lesson.correction_message_id <= 0
        ):
            return False

        with self._connect() as connection:
            response = connection.execute(
                "SELECT * FROM messages WHERE id = ?",
                (lesson.correction_message_id,),
            ).fetchone()
            if (
                response is None
                or response["role"] != "user"
                or response["session_id"] != lesson.session_id
                or response["content"].strip() != lesson.correct_answer
            ):
                return False

            bootstrap_prompt = lesson.metadata.get("bootstrap_prompt") is True
            if bootstrap_prompt:
                if (
                    lesson.problem != TRUSTED_DIALOGUE_BOOTSTRAP_PROMPT
                    or lesson.problem_message_id is not None
                    or lesson.assistant_message_id is not None
                    or provenance.get("prompt_message_id") is not None
                ):
                    return False
                prior_dialogue = connection.execute(
                    """SELECT id FROM messages
                       WHERE session_id = ? AND id < ? AND role IN ('user', 'assistant')
                       ORDER BY id DESC LIMIT 1""",
                    (lesson.session_id, response["id"]),
                ).fetchone()
                return prior_dialogue is None

            prompt_id = provenance.get("prompt_message_id")
            if (
                isinstance(prompt_id, bool)
                or not isinstance(prompt_id, int)
                or prompt_id <= 0
                or lesson.problem_message_id != prompt_id
                or lesson.assistant_message_id != prompt_id
            ):
                return False
            prompt = connection.execute(
                "SELECT * FROM messages WHERE id = ?", (prompt_id,)
            ).fetchone()
            if (
                prompt is None
                or prompt["role"] != "assistant"
                or prompt["session_id"] != lesson.session_id
                or prompt["id"] >= response["id"]
                or prompt["content"].strip() != lesson.problem
            ):
                return False
            latest_assistant = connection.execute(
                """SELECT id FROM messages
                   WHERE session_id = ? AND role = 'assistant' AND id < ?
                   ORDER BY id DESC LIMIT 1""",
                (lesson.session_id, response["id"]),
            ).fetchone()
            first_human_response = connection.execute(
                """SELECT id FROM messages
                   WHERE session_id = ? AND role = 'user' AND id > ? AND id <= ?
                   ORDER BY id ASC LIMIT 1""",
                (lesson.session_id, prompt_id, response["id"]),
            ).fetchone()
        return bool(
            latest_assistant is not None
            and latest_assistant["id"] == prompt_id
            and first_human_response is not None
            and first_human_response["id"] == response["id"]
        )

    def get_lesson(self, lesson_id: int) -> Lesson | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM lessons WHERE id = ?", (lesson_id,)
            ).fetchone()
        return self._lesson(row) if row else None

    def list_recent_lessons(
        self,
        limit: int = 20,
        pending_only: bool = False,
        *,
        foundation_profile_id: str | None = None,
    ) -> list[Lesson]:
        clauses = ["status = 'pending'"] if pending_only else []
        parameters: list[Any] = []
        if foundation_profile_id is not None:
            clauses.append("json_extract(metadata, '$.foundation_profile_id') = ?")
            parameters.append(_foundation_profile(foundation_profile_id))
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM lessons {where} ORDER BY id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._lesson(row) for row in rows]

    def search_relevant_lessons(self, query: str, limit: int = 5) -> list[Lesson]:
        # Human dialogue demonstrations teach the floaty parameters, but they
        # are deliberately not factual retrieval memory.  Injecting an old
        # model utterance as a "problem" and the next human utterance as its
        # "correct answer" would invert an ordinary conversation at inference.
        candidates = [
            lesson
            for lesson in self.list_recent_lessons(limit=250)
            if lesson.source != TRUSTED_HUMAN_DIALOGUE_SOURCE
            or lesson.verifier != TRUSTED_HUMAN_DIALOGUE_VERIFIER
        ]

        def searchable_terms(text: str) -> set[str]:
            terms = set(re.findall(r"[a-z]+|[-+*/%]", text.lower()))
            terms.difference_update(
                {
                    "a",
                    "an",
                    "and",
                    "answer",
                    "are",
                    "by",
                    "calculate",
                    "could",
                    "do",
                    "evaluate",
                    "find",
                    "how",
                    "is",
                    "me",
                    "my",
                    "of",
                    "please",
                    "solve",
                    "the",
                    "to",
                    "was",
                    "were",
                    "what",
                    "your",
                    "you",
                }
            )
            aliases = {
                "+": {"add", "addition", "plus", "sum"},
                "-": {"subtract", "subtraction", "minus", "difference"},
                "*": {"multiply", "multiplication", "times", "product"},
                "/": {"divide", "division", "quotient"},
                "%": {"percent", "percentage"},
            }
            for symbol, words in aliases.items():
                if terms & words:
                    terms.add(symbol)
            return terms

        query_terms = searchable_terms(query)

        def score(lesson: Lesson) -> tuple[int, int]:
            text_terms = searchable_terms(
                f"{lesson.problem} {lesson.skill} {lesson.feedback}"
            )
            return (len(query_terms & text_terms), lesson.id)

        ranked = sorted(candidates, key=score, reverse=True)
        relevant = [lesson for lesson in ranked if score(lesson)[0] > 0]
        # No lexical match means no useful memory. Falling back to recent
        # lessons made greetings carry hundreds of irrelevant byte tokens
        # through every autoregressive generation step.
        return relevant[:limit]

    def count_pending_lessons(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM lessons WHERE status = 'pending'"
                ).fetchone()[0]
            )

    def count_lessons(self, *, status: str | None = None) -> int:
        where, parameters = ("WHERE status = ?", [status]) if status else ("", [])
        with self._connect() as connection:
            return int(
                connection.execute(
                    f"SELECT COUNT(*) FROM lessons {where}",
                    parameters,
                ).fetchone()[0]
            )

    def count_trusted_typed_dialogue_lessons(
        self,
        *,
        status: str,
        foundation_profile_id: str | None = None,
    ) -> int:
        profile_clause = ""
        parameters: list[Any] = [
            status,
            TRUSTED_HUMAN_DIALOGUE_SOURCE,
            TRUSTED_HUMAN_DIALOGUE_VERIFIER,
        ]
        if foundation_profile_id is not None:
            profile_clause = (
                " AND json_extract(metadata, '$.foundation_profile_id') = ?"
            )
            parameters.append(_foundation_profile(foundation_profile_id))
        with self._connect() as connection:
            return int(
                connection.execute(
                    f"""SELECT COUNT(*) FROM lessons
                       WHERE status = ? AND source = ? AND verifier = ?
                       AND json_extract(metadata, '$.capture') = 'text-chat'
                       {profile_clause}""",
                    parameters,
                ).fetchone()[0]
            )

    def adopt_unscoped_foundation_pending_lessons(self, profile_id: str) -> int:
        """Atomically assign legacy unscoped typed pending rows to one v2 profile.

        Concurrent profile startups serialize under ``BEGIN IMMEDIATE``. The
        first one deterministically adopts every qualifying legacy row in ID
        order; later profiles cannot see or claim those rows.
        """

        profile = _foundation_profile(profile_id)
        changed = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """SELECT id, metadata FROM lessons
                   WHERE status = ? AND source = ? AND verifier = ?
                   AND json_extract(metadata, '$.capture') = 'text-chat'
                   AND json_type(metadata, '$.foundation_profile_id') IS NULL
                   ORDER BY id ASC""",
                (
                    "pending",
                    TRUSTED_HUMAN_DIALOGUE_SOURCE,
                    TRUSTED_HUMAN_DIALOGUE_VERIFIER,
                ),
            ).fetchall()
            for row in rows:
                metadata = _load(row["metadata"])
                metadata["foundation_profile_id"] = profile
                metadata["foundation_profile_adoption"] = "legacy-unscoped-v1"
                cursor = connection.execute(
                    """UPDATE lessons SET metadata = ?
                       WHERE id = ? AND status = 'pending'
                       AND json_type(metadata, '$.foundation_profile_id') IS NULL""",
                    (_dump(metadata), row["id"]),
                )
                changed += cursor.rowcount
        return changed

    def count_personal_used_dialogue_lessons(self, registry_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM lessons AS lesson
                       JOIN training_runs AS run ON run.id = lesson.training_run_id
                       WHERE lesson.status = 'used'
                       AND lesson.source = ? AND lesson.verifier = ?
                       AND json_extract(lesson.metadata, '$.capture') = 'text-chat'
                       AND json_extract(run.metadata, '$.authority') = 'personal-local'
                       AND json_extract(run.metadata, '$.personal_registry_id') = ?""",
                    (
                        TRUSTED_HUMAN_DIALOGUE_SOURCE,
                        TRUSTED_HUMAN_DIALOGUE_VERIFIER,
                        registry_id,
                    ),
                ).fetchone()[0]
            )

    def count_personal_foundation_used_dialogue_lessons(self, profile_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """SELECT COUNT(*) FROM lessons AS lesson
                       JOIN training_runs AS run ON run.id = lesson.training_run_id
                       WHERE lesson.status = 'used'
                       AND lesson.source = ? AND lesson.verifier = ?
                       AND json_extract(lesson.metadata, '$.capture') = 'text-chat'
                       AND json_extract(run.metadata, '$.authority') = 'personal-foundation-local'
                       AND json_extract(run.metadata, '$.foundation_profile_id') = ?""",
                    (
                        TRUSTED_HUMAN_DIALOGUE_SOURCE,
                        TRUSTED_HUMAN_DIALOGUE_VERIFIER,
                        profile_id,
                    ),
                ).fetchone()[0]
            )

    def latest_personal_foundation_monitoring_run(
        self,
        profile_id: str,
    ) -> TrainingRun | None:
        """Return the newest v2 run carrying numeric in-sample telemetry."""

        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM training_runs
                   WHERE json_extract(metadata, '$.authority') = ?
                   AND json_extract(metadata, '$.foundation_profile_id') = ?
                   AND json_type(metrics, '$.baseline_monitoring_loss')
                       IN ('integer', 'real')
                   AND json_type(metrics, '$.candidate_monitoring_loss')
                       IN ('integer', 'real')
                   ORDER BY id DESC LIMIT 1""",
                ("personal-foundation-local", profile_id),
            ).fetchone()
        return None if row is None else self._training_run(row)

    def get_personal_foundation_run(
        self,
        profile_id: str,
        foundation_id: str,
    ) -> TrainingRun | None:
        """Resolve one v2 immutable foundation publication in its profile."""

        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM training_runs
                   WHERE json_extract(metadata, '$.authority') = ?
                   AND json_extract(metadata, '$.foundation_profile_id') = ?
                   AND json_extract(metadata, '$.foundation_id') = ?
                   ORDER BY id DESC LIMIT 1""",
                ("personal-foundation-local", profile_id, foundation_id),
            ).fetchone()
        return None if row is None else self._training_run(row)

    def list_personal_foundation_runs(
        self,
        profile_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
        before_id: int | None = None,
    ) -> list[TrainingRun]:
        """Page newest v2 runs without unrelated profiles consuming the limit."""

        profile = _foundation_profile(profile_id)
        if before_id is not None and (
            isinstance(before_id, bool)
            or not isinstance(before_id, int)
            or before_id <= 0
        ):
            raise ValueError("before_id must be a positive integer")
        clauses = [
            "json_extract(metadata, '$.authority') = ?",
            "json_extract(metadata, '$.foundation_profile_id') = ?",
        ]
        parameters: list[Any] = ["personal-foundation-local", profile]
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        if before_id is not None:
            clauses.append("id < ?")
            parameters.append(before_id)
        parameters.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM training_runs
                    WHERE {' AND '.join(clauses)}
                    ORDER BY id DESC LIMIT ?""",
                parameters,
            ).fetchall()
        return [self._training_run(row) for row in rows]

    def list_personal_foundation_used_dialogue_lessons(
        self,
        profile_id: str,
        *,
        limit: int | None = None,
        run_statuses: Iterable[str] | None = None,
        newest_first: bool = False,
    ) -> list[Lesson]:
        statuses = None if run_statuses is None else tuple(dict.fromkeys(run_statuses))
        if statuses is not None and not statuses:
            return []
        status_clause = ""
        parameters: list[Any] = [
            TRUSTED_HUMAN_DIALOGUE_SOURCE,
            TRUSTED_HUMAN_DIALOGUE_VERIFIER,
            profile_id,
        ]
        if statuses is not None:
            status_clause = (
                " AND run.status IN (" + ",".join("?" for _ in statuses) + ")"
            )
            parameters.extend(statuses)
        limit_clause = "" if limit is None else " LIMIT ?"
        if limit is not None:
            parameters.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT lesson.* FROM lessons AS lesson
                    JOIN training_runs AS run ON run.id = lesson.training_run_id
                    WHERE lesson.status = 'used'
                    AND lesson.source = ? AND lesson.verifier = ?
                    AND json_extract(lesson.metadata, '$.capture') = 'text-chat'
                    AND json_extract(run.metadata, '$.authority') = 'personal-foundation-local'
                    AND json_extract(run.metadata, '$.foundation_profile_id') = ?
                    {status_clause}
                    ORDER BY lesson.id {"DESC" if newest_first else "ASC"}{limit_clause}""",
                parameters,
            ).fetchall()
        return [self._lesson(row) for row in rows]

    def list_personal_consumed_dialogue_lessons(
        self,
        registry_id: str,
        *,
        limit: int | None = 10_000,
        run_statuses: Iterable[str] | None = None,
    ) -> list[Lesson]:
        statuses = None if run_statuses is None else tuple(dict.fromkeys(run_statuses))
        if statuses is not None and not statuses:
            return []
        status_clause = ""
        if statuses is not None:
            status_clause = (
                " AND run.status IN (" + ",".join("?" for _ in statuses) + ")"
            )
        limit_clause = "" if limit is None else " LIMIT ?"
        parameters: list[Any] = [
            TRUSTED_HUMAN_DIALOGUE_SOURCE,
            TRUSTED_HUMAN_DIALOGUE_VERIFIER,
            registry_id,
        ]
        if statuses is not None:
            parameters.extend(statuses)
        if limit is not None:
            parameters.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT lesson.* FROM lessons AS lesson
                   JOIN training_runs AS run ON run.id = lesson.training_run_id
                   WHERE lesson.status = 'used'
                   AND lesson.source = ? AND lesson.verifier = ?
                   AND json_extract(lesson.metadata, '$.capture') = 'text-chat'
                   AND json_extract(run.metadata, '$.authority') = 'personal-local'
                   AND json_extract(run.metadata, '$.personal_registry_id') = ?
                   {status_clause}
                   ORDER BY lesson.id ASC{limit_clause}""",
                parameters,
            ).fetchall()
        return [self._lesson(row) for row in rows]

    def quarantine_pending_lesson_ids(
        self,
        lesson_ids: Iterable[int],
        *,
        reason: str,
        foundation_profile_id: str | None = None,
    ) -> int:
        identifiers = list(dict.fromkeys(lesson_ids))
        if not identifiers:
            return 0
        if any(
            type(identifier) is not int or identifier <= 0 for identifier in identifiers
        ):
            raise ValueError("Quarantined lesson IDs must be positive integers")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Quarantine reason is required")
        profile = (
            None
            if foundation_profile_id is None
            else _foundation_profile(foundation_profile_id)
        )
        changed = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for identifier in identifiers:
                row = connection.execute(
                    "SELECT status, metadata FROM lessons WHERE id = ?",
                    (identifier,),
                ).fetchone()
                if row is None or row["status"] != "pending":
                    continue
                metadata = _load(row["metadata"])
                if (
                    profile is not None
                    and metadata.get("foundation_profile_id") != profile
                ):
                    continue
                metadata.update(
                    {
                        "personal_training_eligible": False,
                        "personal_training_quarantine_reason": reason.strip(),
                    }
                )
                cursor = connection.execute(
                    """UPDATE lessons SET status = 'ineligible', metadata = ?
                       WHERE id = ? AND status = 'pending'
                       AND (? IS NULL OR json_extract(metadata, '$.foundation_profile_id') = ?)""",
                    (_dump(metadata), identifier, profile, profile),
                )
                changed += cursor.rowcount
        return changed

    def claim_pending_lessons(
        self,
        *,
        limit: int,
        checkpoint_in: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TrainingRun | None, list[Lesson]]:
        """Atomically reserve pending lessons for one trainer process."""
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM lessons WHERE status = 'pending' ORDER BY id ASC LIMIT ?",
                (max(0, limit),),
            ).fetchall()
            if not rows:
                return None, []
            ids = [int(row["id"]) for row in rows]
            cursor = connection.execute(
                """INSERT INTO training_runs(
                       status, lesson_ids, checkpoint_in, created_at, updated_at, metadata
                   ) VALUES ('training', ?, ?, ?, ?, ?)""",
                (json.dumps(ids), checkpoint_in, now, now, _dump(metadata)),
            )
            run_id = int(cursor.lastrowid)
            placeholders = ",".join("?" for _ in ids)
            connection.execute(
                f"UPDATE lessons SET status = 'training', training_run_id = ? WHERE status = 'pending' AND id IN ({placeholders})",
                [run_id, *ids],
            )
            claimed_rows = connection.execute(
                "SELECT * FROM lessons WHERE training_run_id = ? AND status = 'training' ORDER BY id ASC",
                (run_id,),
            ).fetchall()
            run_row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._training_run(run_row), [self._lesson(row) for row in claimed_rows]

    def claim_pending_lesson_ids(
        self,
        lesson_ids: Iterable[int | str],
        *,
        checkpoint_in: str | None = None,
        metadata: dict[str, Any] | None = None,
        foundation_profile_id: str | None = None,
    ) -> tuple[TrainingRun | None, list[Lesson]]:
        """Atomically claim one exact preselected local-training batch."""

        raw_ids = list(lesson_ids)
        if any(type(identifier) not in {int, str} for identifier in raw_ids):
            raise ValueError("lesson_ids must contain distinct positive integers")
        ids = [int(identifier) for identifier in raw_ids]
        if not ids:
            return None, []
        if len(set(ids)) != len(ids) or any(identifier <= 0 for identifier in ids):
            raise ValueError("lesson_ids must contain distinct positive integers")
        now = _now()
        placeholders = ",".join("?" for _ in ids)
        profile = (
            None
            if foundation_profile_id is None
            else _foundation_profile(foundation_profile_id)
        )
        if (
            profile is not None
            and (metadata or {}).get("foundation_profile_id") != profile
        ):
            raise ValueError("Foundation claim metadata differs from its lesson scope")
        profile_clause = (
            ""
            if profile is None
            else " AND json_extract(metadata, '$.foundation_profile_id') = ?"
        )
        selection_parameters: list[Any] = [*ids]
        if profile is not None:
            selection_parameters.append(profile)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""SELECT * FROM lessons WHERE status = 'pending'
                    AND id IN ({placeholders}){profile_clause}""",
                selection_parameters,
            ).fetchall()
            if len(rows) != len(ids):
                return None, []
            cursor = connection.execute(
                """INSERT INTO training_runs(
                       status, lesson_ids, checkpoint_in, created_at, updated_at, metadata
                   ) VALUES ('training', ?, ?, ?, ?, ?)""",
                (json.dumps(ids), checkpoint_in, now, now, _dump(metadata)),
            )
            run_id = int(cursor.lastrowid)
            connection.execute(
                f"""UPDATE lessons SET status = 'training', training_run_id = ?
                    WHERE status = 'pending' AND id IN ({placeholders})
                    {profile_clause}""",
                [run_id, *selection_parameters],
            )
            claimed_rows = connection.execute(
                """SELECT * FROM lessons WHERE training_run_id = ?
                   AND status = 'training'
                   AND (? IS NULL OR json_extract(metadata, '$.foundation_profile_id') = ?)
                   ORDER BY id ASC""",
                (run_id, profile, profile),
            ).fetchall()
            if len(claimed_rows) != len(ids):
                raise RuntimeError(
                    "Could not atomically claim the complete lesson batch"
                )
            run_row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._training_run(run_row), [self._lesson(row) for row in claimed_rows]

    def mark_lessons_used(
        self, lesson_ids: Iterable[int | str], training_run_id: int | None = None
    ) -> None:
        ids = [int(identifier) for identifier in lesson_ids]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE lessons SET status = 'used', used_at = ?, training_run_id = ? WHERE id IN ({placeholders})",
                [_now(), training_run_id, *ids],
            )

    def add_practice(
        self,
        lesson_id: int,
        problem: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> Practice:
        verification = verify_answer(problem, answer)
        if verification.expected is None or not verification.is_correct:
            raise ValueError("Practice answer is not verified")
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO practice(lesson_id, problem, answer, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (lesson_id, problem, verification.expected, _now(), _dump(metadata)),
            )
            row = connection.execute(
                "SELECT * FROM practice WHERE lesson_id = ? AND problem = ?",
                (lesson_id, problem),
            ).fetchone()
        return self._practice(row)

    def list_practice(
        self, lesson_id: int | None = None, limit: int = 100
    ) -> list[Practice]:
        where, parameters = (
            ("WHERE lesson_id = ?", [lesson_id]) if lesson_id is not None else ("", [])
        )
        parameters.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM practice {where} ORDER BY id DESC LIMIT ?", parameters
            ).fetchall()
        return [self._practice(row) for row in rows]

    def start_training_run(
        self,
        lesson_ids: Iterable[int | str] = (),
        checkpoint_in: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO training_runs(status, lesson_ids, checkpoint_in, created_at, updated_at, metadata) VALUES ('queued', ?, ?, ?, ?, ?)",
                (
                    json.dumps(list(lesson_ids)),
                    checkpoint_in,
                    now,
                    now,
                    _dump(metadata),
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._training_run(row)

    def update_training_run(
        self,
        run_id: int,
        status: str,
        *,
        checkpoint_out: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TrainingRun:
        with self._connect() as connection:
            current = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not current:
                raise ValueError("Training run not found")
            merged_metadata = _load(current["metadata"])
            merged_metadata.update(metadata or {})
            connection.execute(
                """UPDATE training_runs SET status = ?, checkpoint_out = COALESCE(?, checkpoint_out),
                   metrics = ?, error = ?, updated_at = ?, metadata = ? WHERE id = ?""",
                (
                    status,
                    checkpoint_out,
                    _dump(metrics or _load(current["metrics"])),
                    error,
                    _now(),
                    _dump(merged_metadata),
                    run_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._training_run(row)

    def promote_training_run(
        self,
        run_id: int,
        lesson_ids: Iterable[int | str],
        *,
        checkpoint_out: str,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """Commit the promotion ledger and consumed lessons in one transaction."""
        ids = list(dict.fromkeys(int(identifier) for identifier in lesson_ids))
        with self._connect() as connection:
            current = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if current is None:
                raise ValueError("Training run not found")
            merged_metadata = _load(current["metadata"])
            merged_metadata.update(metadata or {})
            now = _now()
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(
                    f"""UPDATE lessons SET status = 'used', used_at = ?, training_run_id = ?
                        WHERE training_run_id = ? AND status = 'training' AND id IN ({placeholders})""",
                    [now, run_id, run_id, *ids],
                )
            connection.execute(
                """UPDATE lessons SET status = 'pending', training_run_id = NULL
                   WHERE training_run_id = ? AND status = 'training'""",
                (run_id,),
            )
            connection.execute(
                """UPDATE training_runs SET status = 'promoted', checkpoint_out = ?,
                   metrics = ?, error = NULL, updated_at = ?, metadata = ? WHERE id = ?""",
                (checkpoint_out, _dump(metrics), now, _dump(merged_metadata), run_id),
            )
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._training_run(row)

    def complete_local_floaty_run(
        self,
        run_id: int,
        lesson_ids: Iterable[int | str],
        *,
        status: str,
        bundle_out: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """Consume one attempted five-round batch without claiming promotion."""

        if status not in {"contributed", "local_rejected"}:
            raise ValueError(
                "Local floaty status must be contributed or local_rejected"
            )
        raw_ids = list(lesson_ids)
        if any(
            type(identifier) is not int or identifier <= 0 for identifier in raw_ids
        ):
            raise ValueError(
                "Local floaty lesson IDs must be distinct positive integers"
            )
        if not raw_ids:
            raise ValueError("A local floaty run must consume at least one lesson")
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError(
                "Local floaty lesson IDs must be distinct positive integers"
            )
        ids = raw_ids
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if current is None or current["status"] != "training":
                raise ValueError("Local floaty training run is not active")
            claimed_ids = json.loads(current["lesson_ids"])
            if (
                not isinstance(claimed_ids, list)
                or any(type(identifier) is not int for identifier in claimed_ids)
                or len(claimed_ids) != len(ids)
                or set(claimed_ids) != set(ids)
            ):
                raise ValueError(
                    "Local floaty completion must include the exact claimed batch"
                )
            merged_metadata = _load(current["metadata"])
            merged_metadata.update(metadata or {})
            now = _now()
            cursor = connection.execute(
                f"""UPDATE lessons SET status = 'used', used_at = ?, training_run_id = ?
                    WHERE training_run_id = ? AND status = 'training'
                    AND id IN ({placeholders})""",
                [now, run_id, run_id, *ids],
            )
            if cursor.rowcount != len(ids):
                raise ValueError("Claimed local lesson batch changed before completion")
            connection.execute(
                """UPDATE lessons SET status = 'pending', training_run_id = NULL
                   WHERE training_run_id = ? AND status = 'training'""",
                (run_id,),
            )
            connection.execute(
                """UPDATE training_runs SET status = ?, checkpoint_out = ?,
                   metrics = ?, error = NULL, updated_at = ?, metadata = ? WHERE id = ?""",
                (
                    status,
                    bundle_out,
                    _dump(metrics),
                    now,
                    _dump(merged_metadata),
                    run_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._training_run(row)

    def complete_personal_foundation_run(
        self,
        run_id: int,
        lesson_ids: Iterable[int],
        *,
        checkpoint_out: str,
        metrics: dict[str, Any],
        metadata: dict[str, Any],
    ) -> TrainingRun:
        """Atomically promote one exact five-turn full-foundation claim."""

        ids = list(lesson_ids)
        if (
            len(ids) != 5
            or any(type(identifier) is not int or identifier <= 0 for identifier in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "Personal foundation completion requires five exact lesson IDs"
            )
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if current is None or current["status"] != "training":
                raise ValueError("Personal foundation training run is not active")
            claimed = json.loads(current["lesson_ids"])
            current_metadata = _load(current["metadata"])
            if (
                claimed != ids
                or current_metadata.get("authority") != "personal-foundation-local"
            ):
                raise ValueError(
                    "Personal foundation completion differs from its exact claim"
                )
            profile = _foundation_profile(current_metadata.get("foundation_profile_id"))
            now = _now()
            cursor = connection.execute(
                f"""UPDATE lessons SET status = 'used', used_at = ?, training_run_id = ?
                    WHERE training_run_id = ? AND status = 'training'
                    AND id IN ({placeholders})
                    AND json_extract(metadata, '$.foundation_profile_id') = ?""",
                [now, run_id, run_id, *ids, profile],
            )
            if cursor.rowcount != len(ids):
                raise ValueError("Personal foundation lesson ownership changed")
            merged = dict(current_metadata)
            merged.update(metadata)
            connection.execute(
                """UPDATE training_runs SET status = 'promoted', checkpoint_out = ?,
                   metrics = ?, error = NULL, updated_at = ?, metadata = ? WHERE id = ?""",
                (checkpoint_out, _dump(metrics), now, _dump(merged), run_id),
            )
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return self._training_run(row)

    def release_training_run(
        self,
        run_id: int,
        status: str,
        *,
        checkpoint_out: str | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TrainingRun:
        """Return claimed lessons to pending when a candidate is rejected or fails."""
        if status not in {"rejected", "failed"}:
            raise ValueError("Released training runs must be rejected or failed")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if current is None:
                raise ValueError("Training run not found")
            merged_metadata = _load(current["metadata"])
            merged_metadata.update(metadata or {})
            connection.execute(
                """UPDATE lessons SET status = 'pending', training_run_id = NULL
                   WHERE training_run_id = ? AND status = 'training'""",
                (run_id,),
            )
            connection.execute(
                """UPDATE training_runs SET status = ?, checkpoint_out = COALESCE(?, checkpoint_out),
                   metrics = ?, error = ?, updated_at = ?, metadata = ? WHERE id = ?""",
                (
                    status,
                    checkpoint_out,
                    _dump(metrics or _load(current["metrics"])),
                    error,
                    _now(),
                    _dump(merged_metadata),
                    run_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._training_run(row)

    def rollback_training_run(
        self,
        run_id: int,
        *,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TrainingRun:
        """Return lessons from one promoted checkpoint to the pending queue."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if current is None or current["status"] != "promoted":
                raise ValueError("Only a promoted training run can be rolled back")
            merged_metadata = _load(current["metadata"])
            merged_metadata.update(metadata or {})
            connection.execute(
                """UPDATE lessons SET status = 'pending', used_at = NULL, training_run_id = NULL
                   WHERE training_run_id = ? AND status = 'used'""",
                (run_id,),
            )
            connection.execute(
                """UPDATE training_runs SET status = 'rolled_back', error = ?,
                   metadata = ?, updated_at = ? WHERE id = ?""",
                (error, _dump(merged_metadata), _now(), run_id),
            )
            row = connection.execute(
                "SELECT * FROM training_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._training_run(row)

    def retire_promoted_training_run(
        self,
        run_id: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingRun:
        """Record an explicit user rollback without replaying its lessons.

        Crash and activation recovery use :meth:`rollback_training_run` and
        deliberately requeue their lessons.  An intentional user rollback is
        different: its exact batch remains consumed so automatic startup
        training cannot recreate the adapter the user just rejected.
        """

        return self.retire_promoted_training_runs(
            [run_id],
            metadata=metadata,
        )[0]

    def retire_promoted_training_runs(
        self,
        run_ids: Iterable[int],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[TrainingRun]:
        """Atomically retire promoted runs while leaving lessons consumed."""

        identifiers = list(dict.fromkeys(run_ids))
        if not identifiers or any(
            type(identifier) is not int or identifier <= 0 for identifier in identifiers
        ):
            raise ValueError("Retired training run IDs must be positive integers")
        retired: list[TrainingRun] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for run_id in identifiers:
                current = connection.execute(
                    "SELECT * FROM training_runs WHERE id = ?",
                    (run_id,),
                ).fetchone()
                if current is None or current["status"] != "promoted":
                    raise ValueError("Only promoted training runs can be retired")
                try:
                    lesson_ids = json.loads(current["lesson_ids"])
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Promoted training run has invalid lesson IDs"
                    ) from error
                if (
                    not isinstance(lesson_ids, list)
                    or not lesson_ids
                    or any(
                        type(identifier) is not int or identifier <= 0
                        for identifier in lesson_ids
                    )
                    or len(set(lesson_ids)) != len(lesson_ids)
                ):
                    raise ValueError("Promoted training run has invalid lesson IDs")
                placeholders = ",".join("?" for _ in lesson_ids)
                linked = connection.execute(
                    f"""SELECT COUNT(*) FROM lessons
                        WHERE training_run_id = ? AND status = 'used'
                        AND id IN ({placeholders})""",
                    [run_id, *lesson_ids],
                ).fetchone()[0]
                if linked != len(lesson_ids):
                    raise ValueError("Promoted training run lesson ownership changed")
                merged_metadata = _load(current["metadata"])
                merged_metadata.update(metadata or {})
                connection.execute(
                    """UPDATE training_runs SET status = 'rolled_back', updated_at = ?,
                       metadata = ? WHERE id = ?""",
                    (_now(), _dump(merged_metadata), run_id),
                )
            placeholders = ",".join("?" for _ in identifiers)
            rows = connection.execute(
                f"SELECT * FROM training_runs WHERE id IN ({placeholders}) ORDER BY id",
                identifiers,
            ).fetchall()
            retired = [self._training_run(row) for row in rows]
        return retired

    def get_training_status(self, run_id: int | None = None) -> TrainingRun | None:
        with self._connect() as connection:
            if run_id is None:
                row = connection.execute(
                    "SELECT * FROM training_runs ORDER BY id DESC LIMIT 1"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM training_runs WHERE id = ?", (run_id,)
                ).fetchone()
        return self._training_run(row) if row else None

    def list_training_runs(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[TrainingRun]:
        where, parameters = (
            ("WHERE status = ?", [status]) if status is not None else ("", [])
        )
        parameters.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM training_runs {where} ORDER BY id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return [self._training_run(row) for row in rows]

    @staticmethod
    def _session(row: sqlite3.Row) -> Session:
        return Session(row["id"], row["created_at"], _load(row["metadata"]))

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(
            row["id"],
            row["session_id"],
            row["role"],
            row["content"],
            row["reply_to_id"],
            row["created_at"],
            _load(row["metadata"]),
        )

    @staticmethod
    def _lesson(row: sqlite3.Row) -> Lesson:
        return Lesson(
            row["id"],
            row["session_id"],
            row["problem_message_id"],
            row["assistant_message_id"],
            row["correction_message_id"],
            row["problem"],
            row["first_attempt"],
            row["correct_answer"],
            row["feedback"],
            row["skill"],
            row["source"],
            row["verifier"],
            row["status"],
            row["created_at"],
            row["used_at"],
            row["training_run_id"],
            _load(row["metadata"]),
        )

    @staticmethod
    def _practice(row: sqlite3.Row) -> Practice:
        return Practice(
            row["id"],
            row["lesson_id"],
            row["problem"],
            row["answer"],
            row["created_at"],
            _load(row["metadata"]),
        )

    @staticmethod
    def _training_run(row: sqlite3.Row) -> TrainingRun:
        try:
            lesson_ids = json.loads(row["lesson_ids"])
        except json.JSONDecodeError:
            lesson_ids = []
        return TrainingRun(
            row["id"],
            row["status"],
            lesson_ids,
            row["checkpoint_in"],
            row["checkpoint_out"],
            _load(row["metrics"]),
            row["error"],
            row["created_at"],
            row["updated_at"],
            _load(row["metadata"]),
        )


_ALLOWED = re.compile(r"^[0-9+\-*/().%\s]+$")


def _format_fraction(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _ast_fraction(node: ast.AST) -> Fraction:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _ast_fraction(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _ast_fraction(node.left), _ast_fraction(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        if isinstance(node.op, ast.Mod):
            if right == 0 or left.denominator != 1 or right.denominator != 1:
                raise ValueError("Modulo requires non-zero integers")
            return Fraction(left.numerator % right.numerator)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 8:
                raise ValueError("Exponent is outside the safe range")
            return left**right.numerator
    raise ValueError("Unsupported arithmetic expression")


def _extract_expression(text: str) -> str:
    normalized = (
        text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    )
    percent = re.search(
        r"(-?\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(-?\d+(?:\.\d+)?)", normalized, re.I
    )
    if percent:
        return f"({percent.group(1)}/100)*{percent.group(2)}"
    word_patterns = [
        (
            r"divide\s+(-?[\d.]+)\s+by\s+(-?[\d.]+)",
            lambda m: f"{m.group(1)}/{m.group(2)}",
        ),
        (
            r"multiply\s+(-?[\d.]+)\s+by\s+(-?[\d.]+)",
            lambda m: f"{m.group(1)}*{m.group(2)}",
        ),
        (
            r"add\s+(-?[\d.]+)\s+(?:and|to)\s+(-?[\d.]+)",
            lambda m: f"{m.group(1)}+{m.group(2)}",
        ),
        (
            r"subtract\s+(-?[\d.]+)\s+from\s+(-?[\d.]+)",
            lambda m: f"{m.group(2)}-{m.group(1)}",
        ),
    ]
    for pattern, builder in word_patterns:
        match = re.search(pattern, normalized, re.I)
        if match:
            return builder(match)
    candidates = re.findall(
        r"(?<![A-Za-z])[-+]?\s*(?:\d+(?:\.\d+)?|\([^A-Za-z]+\))(?:\s*(?:\*\*|[+\-*/%])\s*[-+]?\s*(?:\d+(?:\.\d+)?|\([^A-Za-z]+\)))+",
        normalized,
    )
    if not candidates:
        raise ValueError("No exact arithmetic expression found")
    expression = max(candidates, key=len).strip().rstrip(".?")
    if not _ALLOWED.fullmatch(expression):
        raise ValueError("Expression contains unsupported characters")
    return expression


def evaluate_arithmetic(text: str) -> Fraction:
    expression = _extract_expression(text)
    if len(expression) > 256:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression, mode="eval")
    return _ast_fraction(tree.body)


def _extract_attempt(answer: str) -> Fraction:
    matches = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?", answer)
    if not matches:
        raise ValueError("No numerical answer found")
    candidate = matches[-1].replace(" ", "")
    if "/" in candidate:
        numerator, denominator = candidate.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(candidate)


def verify_answer(problem: str, answer: str) -> VerificationResult:
    try:
        expression = _extract_expression(problem)
        expected_value = _ast_fraction(ast.parse(expression, mode="eval").body)
        expected = _format_fraction(expected_value)
    except (ValueError, SyntaxError, ZeroDivisionError) as error:
        return VerificationResult(False, None, None, None, str(error))
    try:
        attempted_value = _extract_attempt(answer)
        attempted = _format_fraction(attempted_value)
    except (ValueError, ZeroDivisionError) as error:
        return VerificationResult(False, expected, None, expression, str(error))
    return VerificationResult(
        attempted_value == expected_value, expected, attempted, expression, None
    )


def parse_correction(text: str, problem: str | None = None) -> Correction | None:
    match = re.search(
        r"(?:correct\s+answer\s+is|answer\s+is|should\s+be|it(?:'s|\s+is)|equals?)\s*[:=]?\s*(-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+)?)",
        text,
        re.I,
    )
    if not match:
        return None
    answer = match.group(1).replace(" ", "")
    verified = bool(problem and verify_answer(problem, answer).is_correct)
    return Correction(answer, text.strip(), verified)


def canonical_practice_key(problem: str) -> str | None:
    """Map a generated binary problem to its stable curriculum split key.

    Problems inside the built-in curriculum domains deliberately reuse the
    curriculum's exact key format.  Other valid binary arithmetic keeps a
    separate, stable namespace so conversational lessons can still produce
    practice without being mistaken for official held-out examples.
    """

    try:
        expression = _extract_expression(problem)
        tree = ast.parse(expression, mode="eval").body
    except (ValueError, SyntaxError):
        return None
    if not isinstance(tree, ast.BinOp) or not isinstance(
        tree.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        return None
    try:
        left, right = _ast_fraction(tree.left), _ast_fraction(tree.right)
    except ValueError:
        return None
    if left.denominator != 1 or right.denominator != 1:
        return None
    a, b = left.numerator, right.numerator

    def same_large_width() -> bool:
        return a > 0 and b > 0 and 3 <= len(str(a)) == len(str(b)) <= 8

    if isinstance(tree.op, ast.Add):
        if 0 <= a <= 99 and 0 <= b <= 99:
            return f"add:{a}:{b}"
        if same_large_width():
            return f"+:{a}:{b}"
        operation = "add"
    elif isinstance(tree.op, ast.Sub):
        if 0 <= b <= a <= 99:
            return f"sub:{a}:{b}"
        if a >= b and same_large_width():
            return f"-:{a}:{b}"
        operation = "sub"
    elif isinstance(tree.op, ast.Mult):
        if 0 <= a <= 99 and 0 <= b <= 99:
            return f"mul:{a}:{b}"
        operation = "mul"
    else:
        if b == 0:
            return None
        if 1 <= b <= 99 and a >= 0 and a % b == 0 and 0 <= a // b <= 999:
            return f"div:{a}:{b}"
        operation = "div"
    return f"practice:{operation}:{a}:{b}"


def generate_practice_variants(
    problem: str, count: int = 5, seed: int | None = None
) -> list[tuple[str, str]]:
    try:
        expression = _extract_expression(problem)
        tree = ast.parse(expression, mode="eval").body
    except (ValueError, SyntaxError):
        return []
    if not isinstance(tree, ast.BinOp) or not isinstance(
        tree.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        return []
    try:
        left, right = _ast_fraction(tree.left), _ast_fraction(tree.right)
    except ValueError:
        return []
    if left.denominator != 1 or right.denominator != 1:
        return []
    symbol = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(tree.op)]
    rng = random.Random(seed)
    results: list[tuple[str, str]] = []
    seen = {expression.replace(" ", "")}
    target = max(0, count)
    attempts = 0
    while len(results) < target and attempts < max(100, target * 200):
        attempts += 1
        a = left.numerator + rng.randint(-20, 20)
        b = right.numerator + rng.randint(-20, 20)
        if symbol == "/" and b == 0:
            continue
        variant = f"{a} {symbol} {b}"
        if variant.replace(" ", "") in seen:
            continue
        seen.add(variant.replace(" ", ""))
        answer = _format_fraction(evaluate_arithmetic(variant))
        question = f"What is {variant}?"
        key = canonical_practice_key(question)
        if key is None or split_for_key(key) != "train":
            continue
        results.append((question, answer))
    return results


def render_lesson_prompt(lesson: Lesson) -> str:
    feedback = f" Hint: {lesson.feedback}" if lesson.feedback else ""
    return (
        f"Lesson: {lesson.problem} Correct answer: {lesson.correct_answer}.{feedback}"
    )


def record_verified_correction(
    store: ApprenticeshipStore,
    session_id: str,
    assistant_message_id: int,
    correction_text: str,
    correction_message_id: int | None = None,
    source: str = "human",
) -> Lesson | None:
    problem_message = store.find_preceding_problem(assistant_message_id)
    if problem_message is None:
        return None
    correction = parse_correction(correction_text, problem_message.content)
    if correction is None or not correction.verified:
        return None
    assistant = next(
        (
            message
            for message in store.list_messages(session_id)
            if message.id == assistant_message_id
        ),
        None,
    )
    if assistant is None:
        return None
    return store.add_verified_lesson(
        problem=problem_message.content,
        first_attempt=assistant.content,
        correct_answer=correction.correct_answer,
        feedback=correction.feedback,
        source=source,
        session_id=session_id,
        problem_message_id=problem_message.id,
        assistant_message_id=assistant.id,
        correction_message_id=correction_message_id,
    )
