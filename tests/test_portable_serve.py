from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from mini_arithmetic.apprenticeship import ApprenticeshipStore
from mini_arithmetic.hybrid_config import HybridConfig
from mini_arithmetic.personal_training import (
    PersonalRuntimeIdentity,
    PersonalTrainingConfig,
)
from mini_arithmetic.portable import BackendSelection
from mini_arithmetic.portable_bundle import PortableBundle
from mini_arithmetic.portable_serve import (
    LocalInstanceLease,
    PortableChatService,
    PortableServeError,
    choose_model_size,
    make_http_server,
    personal_profile_id,
    _personal_base_path,
)


def test_personal_root_honors_environment_configuration_without_cli_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLM_PERSONAL_MODEL_DIR", "private/custom-model")
    configured = PersonalTrainingConfig.from_env(project_root=tmp_path)

    assert (
        _personal_base_path(tmp_path, None, configured)
        == (tmp_path / "private" / "custom-model").resolve()
    )
    assert (
        _personal_base_path(tmp_path, "cli-model", configured)
        == (tmp_path / "cli-model").resolve()
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.active_adapter_id: str | None = None
        self.turns: list[list[tuple[str, str]]] = []

    def personal_runtime_identity(self) -> PersonalRuntimeIdentity:
        return PersonalRuntimeIdentity(
            backend="torch",
            foundation_sha256="1" * 64,
            adapter_schema_sha256="2" * 64,
            active_adapter_id=self.active_adapter_id,
        )

    def prepare_personal_adapter(self, artifact: object) -> object:
        return artifact

    def activate_prepared_personal_adapter(
        self, prepared: object, adapter_id: str
    ) -> None:
        del prepared
        self.active_adapter_id = adapter_id

    def activate_foundation_only(self) -> None:
        self.active_adapter_id = None

    def answer(
        self,
        turns: list[tuple[str, str]],
        *,
        max_tokens: int,
    ) -> str:
        assert max_tokens == 12
        self.turns.append(list(turns))
        return f"model reply {len(self.turns)}"

    def status(self) -> dict[str, Any]:
        return {
            "backend": "torch",
            "device": "cpu",
            "parameters": 123,
            "foundation_trainable": False,
        }


class FakeCoordinator:
    profile_id = "test-foundation-profile"

    def __init__(self) -> None:
        self.consider_calls = 0
        self.closed = False

    def consider_training(self) -> bool:
        self.consider_calls += 1
        return self.consider_calls == 5

    def trigger(self, *, force: bool = False) -> bool:
        del force
        return self.consider_training()

    def rollback(self) -> dict[str, object]:
        return {"state": "rolled_back"}

    def status(self) -> dict[str, object]:
        return {
            "state": "ready" if self.consider_calls >= 5 else "idle",
            "lesson_threshold": 5,
            "qualifying_pending": self.consider_calls,
            "uploads_enabled": False,
        }

    def shutdown(self) -> None:
        self.closed = True


def _selection(accelerator: str = "cpu") -> BackendSelection:
    return BackendSelection(
        requested="torch",
        backend="torch",
        accelerator=accelerator,  # type: ignore[arg-type]
        ready=True,
        system="TestOS",
        machine="x86_64",
        install_extra="torch",
        reason="test device",
    )


def _bundle(tmp_path: Path) -> PortableBundle:
    return PortableBundle(
        root=tmp_path,
        manifest={"model_size": "tiny", "parameter_dtype": "float32"},
        config=HybridConfig(
            vocab_size=32_768,
            context_length=32,
            model_dim=16,
            ffn_dim=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_schedule=("conv",),
        ),
        tokenizer=None,  # type: ignore[arg-type]
        weights_path=tmp_path / "unused.safetensors",
    )


def test_five_ordinary_typed_turns_become_private_dialogue_lessons(
    tmp_path: Path,
) -> None:
    store = ApprenticeshipStore(tmp_path / "personal.sqlite3")
    runtime = FakeRuntime()
    coordinator = FakeCoordinator()
    service = PortableChatService(
        store,
        runtime,
        coordinator,  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )

    session_id: str | None = None
    for index in range(5):
        payload: dict[str, object] = {"message": f"human turn {index}"}
        if session_id is not None:
            payload["session_id"] = session_id
        response = service.chat(payload)
        session_id = str(response["session_id"])
        assert response["conversation_lesson_saved"] is True

    lessons = list(reversed(store.list_recent_lessons(limit=20)))
    assert len(lessons) == 5
    assert [lesson.correct_answer for lesson in lessons] == [
        f"human turn {index}" for index in range(5)
    ]
    assert lessons[0].problem == "Start a natural conversation."
    assert [lesson.problem for lesson in lessons[1:]] == [
        f"model reply {index}" for index in range(1, 5)
    ]
    assert all(lesson.metadata["capture"] == "text-chat" for lesson in lessons)
    assert all(lesson.metadata["privacy"] == "private-local-only" for lesson in lessons)
    assert all(
        lesson.metadata["foundation_profile_id"] == coordinator.profile_id
        for lesson in lessons
    )
    assert coordinator.consider_calls == 5
    assert runtime.turns[-1][-1] == ("user", "human turn 4")

    status = service.status()
    assert status["privacy"]["uploads_enabled"] is False
    assert status["privacy"]["community_aggregation"] is False
    assert status["learning"]["lesson_threshold"] == 5
    assert status["realtime"]["conversation_ready"] is False

    service.close()
    assert coordinator.closed is True


def test_auto_size_uses_20m_without_guessing_accelerator_memory(
    tmp_path: Path,
) -> None:
    assert choose_model_size("auto", _selection("cpu"), model_root=tmp_path) == "20m"
    for accelerator in ("cuda", "rocm", "xpu", "mps"):
        assert (
            choose_model_size("auto", _selection(accelerator), model_root=tmp_path)
            == "20m"
        )
    (tmp_path / "portable-tiny").mkdir()
    assert choose_model_size("auto", _selection("cpu"), model_root=tmp_path) == "tiny"
    assert choose_model_size("tiny", _selection("cuda")) == "tiny"
    assert choose_model_size("standard", _selection("cuda")) == "standard"
    assert choose_model_size("200m", _selection("cpu")) == "200m"


def test_private_adapter_profiles_are_namespaced_by_runtime_identity() -> None:
    base = PersonalRuntimeIdentity(
        backend="torch",
        foundation_sha256="1" * 64,
        adapter_schema_sha256="2" * 64,
    )
    values = {
        personal_profile_id(base),
        personal_profile_id(
            PersonalRuntimeIdentity(
                backend="mlx",
                foundation_sha256=base.foundation_sha256,
                adapter_schema_sha256=base.adapter_schema_sha256,
            )
        ),
        personal_profile_id(
            PersonalRuntimeIdentity(
                backend=base.backend,
                foundation_sha256="3" * 64,
                adapter_schema_sha256=base.adapter_schema_sha256,
            )
        ),
        personal_profile_id(
            PersonalRuntimeIdentity(
                backend=base.backend,
                foundation_sha256=base.foundation_sha256,
                adapter_schema_sha256="4" * 64,
            )
        ),
    }
    assert len(values) == 4
    assert all(len(value) == 24 for value in values)


def test_private_model_lease_rejects_a_second_local_service(tmp_path: Path) -> None:
    lock_path = tmp_path / "private" / "service.lock"
    first = LocalInstanceLease(lock_path)
    try:
        with pytest.raises(PortableServeError, match="already using"):
            LocalInstanceLease(lock_path)
    finally:
        first.close()

    replacement = LocalInstanceLease(lock_path)
    replacement.close()


def test_private_http_service_rejects_non_loopback_bind(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "personal.sqlite3")
    service = PortableChatService(
        store,
        FakeRuntime(),
        FakeCoordinator(),  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )
    with pytest.raises(PortableServeError, match="bind only"):
        make_http_server("0.0.0.0", 0, service)


def test_message_is_not_committed_when_inference_fails(tmp_path: Path) -> None:
    class FailingRuntime(FakeRuntime):
        def answer(
            self,
            turns: list[tuple[str, str]],
            *,
            max_tokens: int,
        ) -> str:
            del turns, max_tokens
            raise RuntimeError("device failure")

    store = ApprenticeshipStore(tmp_path / "personal.sqlite3")
    service = PortableChatService(
        store,
        FailingRuntime(),
        FakeCoordinator(),  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )

    with pytest.raises(RuntimeError, match="device failure"):
        service.chat({"message": "do not strand me", "session_id": "failure"})
    assert store.list_messages("failure") == []
    assert store.list_recent_lessons(limit=20) == []


def test_untrainable_target_is_rejected_before_inference_or_lesson_commit(
    tmp_path: Path,
) -> None:
    class EligibilityRuntime(FakeRuntime):
        def personal_training_target_fits(
            self,
            prompt: str,
            human_response: str,
        ) -> bool:
            assert prompt == "Start a natural conversation."
            return len(human_response) <= 12

    store = ApprenticeshipStore(tmp_path / "personal.sqlite3")
    runtime = EligibilityRuntime()
    coordinator = FakeCoordinator()
    service = PortableChatService(
        store,
        runtime,
        coordinator,  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )

    with pytest.raises(ValueError, match="complete local-training target"):
        service.chat({"message": "x" * 13, "session_id": "eligibility"})
    assert runtime.turns == []
    assert store.list_messages("eligibility") == []
    assert store.list_recent_lessons(limit=20) == []
    assert coordinator.consider_calls == 0

    accepted = service.chat({"message": "short", "session_id": "eligibility"})
    assert accepted["conversation_lesson_saved"] is True
    assert len(store.list_recent_lessons(limit=20)) == 1


def test_whole_turn_rolls_back_when_lesson_commit_fails(tmp_path: Path) -> None:
    class FailingLessonStore(ApprenticeshipStore):
        fail_lesson = True

        def _insert_trusted_dialogue_lesson(self, *args: Any, **kwargs: Any) -> Any:
            if self.fail_lesson:
                raise RuntimeError("simulated lesson write failure")
            return super()._insert_trusted_dialogue_lesson(*args, **kwargs)

    store = FailingLessonStore(tmp_path / "personal.sqlite3")
    service = PortableChatService(
        store,
        FakeRuntime(),
        FakeCoordinator(),  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )

    with pytest.raises(RuntimeError, match="simulated lesson write failure"):
        service.chat({"message": "first", "session_id": "atomic"})
    assert store.list_messages("atomic") == []
    assert store.list_recent_lessons(limit=20) == []

    store.fail_lesson = False
    response = service.chat({"message": "retry", "session_id": "atomic"})
    assert response["conversation_lesson_saved"] is True
    assert [message.role for message in store.list_messages("atomic")] == [
        "user",
        "assistant",
    ]
    lesson = store.list_recent_lessons(limit=1)[0]
    assert lesson.problem == "Start a natural conversation."


def test_simultaneous_chat_requests_keep_one_session_ledger_consistent(
    tmp_path: Path,
) -> None:
    class BlockingRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def answer(
            self,
            turns: list[tuple[str, str]],
            *,
            max_tokens: int,
        ) -> str:
            if not self.turns:
                self.entered.set()
                assert self.release.wait(timeout=5)
            return super().answer(turns, max_tokens=max_tokens)

    store = ApprenticeshipStore(tmp_path / "personal.sqlite3")
    runtime = BlockingRuntime()
    coordinator = FakeCoordinator()
    service = PortableChatService(
        store,
        runtime,
        coordinator,  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.chat,
            {"message": "first concurrent turn", "session_id": "shared"},
        )
        assert runtime.entered.wait(timeout=5)
        second = executor.submit(
            service.chat,
            {"message": "second concurrent turn", "session_id": "shared"},
        )
        runtime.release.set()
        first.result(timeout=5)
        second.result(timeout=5)

    messages = store.list_messages("shared")
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[2].reply_to_id == messages[1].id
    lessons = list(reversed(store.list_recent_lessons(limit=10)))
    assert len(lessons) == 2
    assert lessons[0].problem == "Start a natural conversation."
    assert lessons[1].problem == messages[1].content
    assert all(store.is_trusted_dialogue_lesson(lesson) for lesson in lessons)


def test_chat_and_history_remain_bounded_for_a_long_session(tmp_path: Path) -> None:
    store = ApprenticeshipStore(tmp_path / "personal.sqlite3")
    runtime = FakeRuntime()
    service = PortableChatService(
        store,
        runtime,
        FakeCoordinator(),  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )

    for index in range(40):
        service.chat({"message": f"turn {index}", "session_id": "long-session"})

    assert len(runtime.turns[-1]) == 9  # eight prior turns plus the new human turn
    history = service.history("long-session")
    assert history["limit"] == 64
    assert len(history["messages"]) == 64
    assert history["messages"][0].content == "turn 8"
    assert history["messages"][-1].role == "assistant"


@pytest.mark.parametrize("session_id", ["x" * 129, "not/a/session", 123])
def test_chat_rejects_unbounded_or_malformed_session_ids(
    tmp_path: Path,
    session_id: object,
) -> None:
    store = ApprenticeshipStore(tmp_path / "personal.sqlite3")
    service = PortableChatService(
        store,
        FakeRuntime(),
        FakeCoordinator(),  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )
    with pytest.raises(ValueError, match="short local identifier"):
        service.chat({"message": "hello", "session_id": session_id})
    assert store.list_recent_lessons(limit=10) == []


def test_http_mutations_require_local_origin_or_launch_capability(
    tmp_path: Path,
) -> None:
    def request(
        port: int,
        *,
        headers: dict[str, str],
        message: str,
    ) -> int:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(
                "POST",
                "/chat",
                body=json.dumps({"message": message, "session_id": "secure"}),
                headers=headers,
            )
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()

    store = ApprenticeshipStore(tmp_path / "personal.sqlite3")
    service = PortableChatService(
        store,
        FakeRuntime(),
        FakeCoordinator(),  # type: ignore[arg-type]
        bundle=_bundle(tmp_path),
        selection=_selection(),
        max_new_tokens=12,
    )

    protected = make_http_server("127.0.0.1", 0, service, api_token="launch-secret")
    protected_thread = threading.Thread(target=protected.serve_forever, daemon=True)
    protected_thread.start()
    protected_port = int(protected.server_address[1])
    try:
        assert (
            request(
                protected_port,
                headers={
                    "Content-Type": "text/plain",
                    "Origin": "https://evil.example",
                },
                message="drive-by lesson",
            )
            == 403
        )
        assert (
            request(
                protected_port,
                headers={"X-TinyDog-Local-Token": "launch-secret"},
                message="wrong media type",
            )
            == 415
        )
        assert (
            request(
                protected_port,
                headers={
                    "Content-Type": "application/json",
                    "X-TinyDog-Local-Token": "launch-secret",
                },
                message="authorized turn",
            )
            == 200
        )
    finally:
        protected.shutdown()
        protected.server_close()
        protected_thread.join(timeout=5)

    assert [message.content for message in store.list_messages("secure")] == [
        "authorized turn",
        "model reply 1",
    ]

    direct = make_http_server("127.0.0.1", 0, service)
    direct_thread = threading.Thread(target=direct.serve_forever, daemon=True)
    direct_thread.start()
    direct_port = int(direct.server_address[1])
    try:
        assert (
            request(
                direct_port,
                headers={
                    "Content-Type": "application/json",
                    "Origin": "https://evil.example",
                },
                message="second drive-by lesson",
            )
            == 403
        )
    finally:
        direct.shutdown()
        direct.server_close()
        direct_thread.join(timeout=5)

    assert len(store.list_recent_lessons(limit=10)) == 1
