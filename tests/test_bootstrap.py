import json
import os
from pathlib import Path

import pytest

from mini_arithmetic.bootstrap import (
    BootstrapError,
    _needs_setup,
    _node_version_supported,
    _configured_model_size,
    _force_process_tree_stop,
    _install_launcher_stop_handlers,
    _LauncherStopRequested,
    _read_dotenv,
    _realtime_command,
    _process_group_options,
    _request_process_tree_stop,
    _restore_launcher_stop_handlers,
    _resolve_setup_backend,
    _runtime_environment,
    _setup_fingerprint,
    build_setup_plan,
    build_start_commands,
    erase_personal_learning,
    reset_personal_learning,
    restore_personal_learning,
    rollback_personal_foundation,
    show_status,
)


class _RunningProcess:
    pid = 4242

    def poll(self):
        return None

    def terminate(self):  # pragma: no cover - fallback must not be reached
        raise AssertionError("unexpected single-process terminate")

    def kill(self):  # pragma: no cover - fallback must not be reached
        raise AssertionError("unexpected single-process kill")


def test_services_launch_and_stop_as_owned_process_trees(monkeypatch) -> None:
    assert _process_group_options(windows=False) == {"start_new_session": True}
    assert int(_process_group_options(windows=True)["creationflags"]) & 0x00000200

    requested: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "mini_arithmetic.bootstrap.os.killpg",
        lambda process_id, sent_signal: requested.append((process_id, sent_signal)),
    )
    _request_process_tree_stop(_RunningProcess(), windows=False)  # type: ignore[arg-type]
    assert requested == [(4242, 15)]

    forced: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        forced.append(tuple(command))

    monkeypatch.setattr("mini_arithmetic.bootstrap.subprocess.run", fake_run)
    _force_process_tree_stop(_RunningProcess(), windows=True)  # type: ignore[arg-type]
    assert forced == [("taskkill", "/PID", "4242", "/T", "/F")]


def test_external_stop_signal_uses_and_restores_orderly_cleanup(monkeypatch) -> None:
    installed: list[tuple[int, object]] = []
    monkeypatch.setattr(
        "mini_arithmetic.bootstrap.signal.getsignal",
        lambda signal_number: f"old-{int(signal_number)}",
    )
    monkeypatch.setattr(
        "mini_arithmetic.bootstrap.signal.signal",
        lambda signal_number, handler: installed.append((int(signal_number), handler)),
    )

    previous = _install_launcher_stop_handlers()
    assert previous
    stop_handler = installed[0][1]
    with pytest.raises(_LauncherStopRequested):
        stop_handler(installed[0][0], None)  # type: ignore[operator]

    installed.clear()
    _restore_launcher_stop_handlers(previous)
    assert installed == [
        (signal_number, previous[signal_number]) for signal_number in previous
    ]


def test_model_size_uses_env_default_but_explicit_cli_wins() -> None:
    settings = {"TINYDOG_MODEL_SIZE": "standard"}
    assert _configured_model_size("auto", settings) == "standard"
    assert _configured_model_size("tiny", settings) == "tiny"
    with pytest.raises(BootstrapError, match="TINYDOG_MODEL_SIZE"):
        _configured_model_size("auto", {"TINYDOG_MODEL_SIZE": "giant"})


def test_runtime_forces_local_ui_telemetry_off(monkeypatch) -> None:
    monkeypatch.setenv("NEXT_TELEMETRY_DISABLED", "0")
    environment = _runtime_environment(
        {"NEXT_TELEMETRY_DISABLED": "0"},
        backend="torch",
        model_size="tiny",
    )
    assert environment["NEXT_TELEMETRY_DISABLED"] == "1"
    assert environment["TINYDOG_BACKEND"] == "torch"
    assert environment["TINYDOG_MODEL_SIZE"] == "tiny"


@pytest.mark.parametrize(
    ("reported", "supported"),
    (
        ("v22.13.0", True),
        ("24.0.0\n", True),
        ("v22.12.1", False),
        ("v20.19.1", False),
        ("v19.9.0", False),
        ("nope", False),
    ),
)
def test_local_screen_requires_node_22_13_or_newer(
    reported: str,
    supported: bool,
) -> None:
    assert _node_version_supported(reported) is supported


def test_setup_plan_installs_only_mlx_bundle_on_apple_path(tmp_path: Path) -> None:
    plan = build_setup_plan(
        tmp_path,
        backend="mlx",
        source_python="python3",
        install_ui=False,
        windows=False,
    )

    assert plan.backend == "mlx"
    assert plan.commands[0] == ("python3", "-m", "venv", str(tmp_path / ".venv"))
    install = plan.commands[1]
    assert f"{tmp_path}[realtime,mlx]" in install
    assert not any("torch" in part for command in plan.commands for part in command)
    assert not any(
        "checkpoint" in part or "weight" in part
        for command in plan.commands
        for part in command
    )


def test_text_only_mlx_setup_does_not_install_realtime_dependencies(
    tmp_path: Path,
) -> None:
    plan = build_setup_plan(
        tmp_path,
        backend="mlx",
        source_python="python3",
        install_ui=False,
        install_realtime=False,
        windows=False,
    )

    assert f"{tmp_path}[mlx]" in plan.commands[1]
    assert "realtime" not in plan.commands[1][-1]


def test_setup_plan_uses_windows_venv_and_optional_torch_index(tmp_path: Path) -> None:
    plan = build_setup_plan(
        tmp_path,
        backend="torch",
        source_python="py",
        torch_index_url="https://download.pytorch.example/whl/rocm",
        install_ui=True,
        windows=True,
    )

    assert plan.python == tmp_path / ".venv" / "Scripts" / "python.exe"
    assert str(tmp_path) in plan.commands[1]
    assert not any("realtime" in part for part in plan.commands[1])
    assert plan.commands[2][-2:] == (
        "--index-url",
        "https://download.pytorch.example/whl/rocm",
    )
    assert plan.commands[-1] == (
        "npm",
        "ci",
        "--omit=dev",
        "--no-audit",
        "--no-fund",
        "--loglevel=error",
    )


def test_backend_setup_selection_is_platform_specific() -> None:
    assert _resolve_setup_backend("auto", system="Darwin", machine="arm64") == "mlx"
    assert _resolve_setup_backend("auto", system="Windows", machine="AMD64") == "torch"
    with pytest.raises(BootstrapError, match="Apple Silicon"):
        _resolve_setup_backend("mlx", system="Linux", machine="x86_64")


def test_portable_start_command_carries_backend_and_size(tmp_path: Path) -> None:
    commands = build_start_commands(
        tmp_path,
        backend="torch",
        model_size="tiny",
        include_realtime=False,
        include_ui=False,
        windows=False,
    )

    assert commands == (
        (
            "learning service",
            tmp_path,
            (
                str(tmp_path / ".venv" / "bin" / "python"),
                "-m",
                "mini_arithmetic.portable_serve",
                "--backend",
                "torch",
                "--model-size",
                "tiny",
            ),
        ),
    )


def test_portable_start_command_selects_exported_model_and_private_state(
    tmp_path: Path,
) -> None:
    command = build_start_commands(
        tmp_path,
        backend="torch",
        model_size="tiny",
        model_root="trained-models",
        personal_root="data/trained-personal",
        db_path="data/trained-conversations.db",
        include_realtime=False,
        include_ui=False,
        windows=False,
    )[0][2]

    assert command[-6:] == (
        "--model-root",
        "trained-models",
        "--personal-root",
        "data/trained-personal",
        "--db",
        "data/trained-conversations.db",
    )


def test_torch_start_does_not_launch_the_mlx_native_voice_runtime(
    tmp_path: Path,
) -> None:
    commands = build_start_commands(
        tmp_path,
        backend="torch",
        include_realtime=True,
        include_ui=True,
        windows=False,
    )

    assert [name for name, _cwd, _command in commands] == [
        "learning service",
        "local screen",
    ]
    assert commands[-1][2][1:3] == ("run", "start")
    assert commands[-1][2] == ("npm", "run", "start")


def test_legacy_start_is_explicit_and_mlx_only(tmp_path: Path) -> None:
    commands = build_start_commands(
        tmp_path,
        backend="mlx",
        legacy_mlx=True,
        config="config.json",
        weights="local.safetensors",
        include_realtime=False,
        include_ui=False,
    )
    assert "mini_arithmetic.serve" in commands[0][2]
    assert "local.safetensors" in commands[0][2]

    with pytest.raises(BootstrapError, match="requires the MLX engine"):
        build_start_commands(tmp_path, backend="torch", legacy_mlx=True)


def test_dotenv_is_read_as_data_and_never_as_shell(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# local only\nexport SIMPLE=yes\nQUOTED='hello world'\n"
        "INVALID LINE\n9BAD=no\nSHELL=$(do-not-run)\n",
        encoding="utf-8",
    )

    assert _read_dotenv(path) == {
        "SIMPLE": "yes",
        "QUOTED": "hello world",
        "SHELL": "$(do-not-run)",
    }
    if os.name != "nt" and path.stat().st_mode & 0o077:
        pytest.fail("the local settings file remained readable by another Unix user")


def test_dotenv_refuses_a_link_instead_of_reading_its_target(tmp_path: Path) -> None:
    private = tmp_path / "private-settings"
    private.write_text("SECRET=do-not-follow\n", encoding="utf-8")
    path = tmp_path / ".env"
    try:
        path.symlink_to(private)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are unavailable on this platform")

    with pytest.raises(BootstrapError, match="regular, unlinked"):
        _read_dotenv(path)


def test_realtime_local_configuration_is_forwarded() -> None:
    base = ("python", "-m", "mini_arithmetic.realtime_gateway")
    assert _realtime_command(
        base, {"MLM_NATIVE_CHECKPOINT": "local/manifest.json"}
    ) == (
        *base,
        "--checkpoint",
        "local/manifest.json",
    )
    assert _realtime_command(
        base,
        {"MLM_UNTRAINED_NATIVE_CONFIG": "pilot.json", "MLM_UNTRAINED_NATIVE_SEED": "7"},
    ) == (*base, "--untrained-config", "pilot.json", "--untrained-seed", "7")


def test_setup_marker_becomes_stale_when_dependencies_change(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").touch()
    (tmp_path / "interface").mkdir()
    (tmp_path / "pyproject.toml").write_text("first", encoding="utf-8")
    marker = _setup_fingerprint(tmp_path, "torch")
    (tmp_path / ".tinydog-setup.json").write_text(json.dumps(marker), encoding="utf-8")

    assert _needs_setup(tmp_path, "auto") is False
    (tmp_path / "pyproject.toml").write_text("second", encoding="utf-8")
    assert _needs_setup(tmp_path, "auto") is True


def test_no_ui_setup_is_reused_only_for_no_ui_start(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").touch()
    (tmp_path / "interface").mkdir()
    marker = _setup_fingerprint(tmp_path, "torch", ui_installed=False)
    (tmp_path / ".tinydog-setup.json").write_text(json.dumps(marker), encoding="utf-8")

    assert _needs_setup(tmp_path, "auto", require_ui=False) is False
    assert _needs_setup(tmp_path, "auto", require_ui=True) is True


def test_text_only_mlx_setup_is_reused_only_without_realtime(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").touch()
    (tmp_path / "interface").mkdir()
    marker = _setup_fingerprint(
        tmp_path, "mlx", ui_installed=True, realtime_installed=False
    )
    (tmp_path / ".tinydog-setup.json").write_text(json.dumps(marker), encoding="utf-8")

    assert _needs_setup(tmp_path, "auto", require_realtime=False) is False
    assert _needs_setup(tmp_path, "auto", require_realtime=True) is True


def test_status_and_rollback_use_only_fixed_local_ui_routes(
    monkeypatch, capsys
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_local(path: str, *, method: str = "GET") -> dict[str, object]:
        calls.append((path, method))
        if path == "/api/status":
            return {
                "learning": {
                    "state": "idle",
                    "qualifying_pending": 3,
                    "lesson_threshold": 5,
                    "parameter_scope": "full-foundation",
                    "active_foundation_id": "foundation-one",
                    "rollback_available": True,
                },
                "model": {"backend": "torch", "device": "cpu"},
            }
        return {"state": "ready", "active_foundation_id": "foundation-base"}

    monkeypatch.setattr("mini_arithmetic.bootstrap._local_ui_json", fake_local)
    assert show_status() == 0
    assert rollback_personal_foundation() == 0
    assert calls == [("/api/status", "GET"), ("/api/rollback", "POST")]
    output = capsys.readouterr().out
    assert "3/5 eligible typed turns" in output
    assert "Rollback complete" in output


def test_personal_reset_is_explicit_private_and_recoverable(
    tmp_path: Path, monkeypatch
) -> None:
    data = tmp_path / "data"
    foundation = data / "personal-foundations"
    adapter = data / "personal-model"
    foundation.mkdir(parents=True)
    adapter.mkdir()
    database = data / "personal-conversations.db"
    database.write_text("private", encoding="utf-8")
    (foundation / "state.json").write_text("foundation", encoding="utf-8")
    (adapter / "state.json").write_text("adapter", encoding="utf-8")
    monkeypatch.setattr("mini_arithmetic.bootstrap._port_is_open", lambda _port: False)

    assert reset_personal_learning(tmp_path, confirmed=False) == 1
    assert database.exists()
    assert reset_personal_learning(tmp_path, confirmed=True) == 0
    assert not database.exists()
    assert not foundation.exists()
    assert not adapter.exists()

    backups = list((data / "reset-backups").glob("personal-reset-*"))
    assert len(backups) == 1
    assert (backups[0] / "data__personal-conversations.db").read_text() == "private"
    assert (backups[0] / "data__personal-foundations" / "state.json").is_file()
    assert (backups[0] / "data__personal-model" / "state.json").is_file()
    assert (backups[0] / "reset-manifest.json").is_file()

    assert restore_personal_learning(tmp_path, confirmed=False) == 1
    assert restore_personal_learning(tmp_path, confirmed=True) == 0
    assert database.read_text(encoding="utf-8") == "private"
    assert (foundation / "state.json").read_text(encoding="utf-8") == "foundation"
    assert (adapter / "state.json").read_text(encoding="utf-8") == "adapter"
    assert not backups[0].exists()


def test_personal_reset_refuses_a_configured_path_containing_backups(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text(
        "MLM_PERSONAL_FOUNDATION_DIR=data\n", encoding="utf-8"
    )
    with pytest.raises(BootstrapError, match="backup directory"):
        reset_personal_learning(tmp_path, confirmed=False)


@pytest.mark.parametrize("operation", ("reset", "erase"))
@pytest.mark.parametrize("link_position", ("final", "parent"))
def test_personal_data_commands_never_follow_in_project_directory_links(
    tmp_path: Path,
    monkeypatch,
    operation: str,
    link_position: str,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    protected = tmp_path / "protected"
    if link_position == "final":
        protected.mkdir()
        private_target = protected
        link = data / "personal-foundations"
        personal_root = None
    else:
        private_target = protected / "personal-foundations"
        private_target.mkdir(parents=True)
        link = data / "linked-parent"
        personal_root = "data/linked-parent/personal-foundations"
    sentinel = private_target / "must-survive.txt"
    sentinel.write_text("unrelated", encoding="utf-8")
    try:
        link.symlink_to(protected, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable on this platform")
    monkeypatch.setattr("mini_arithmetic.bootstrap._port_is_open", lambda _port: False)

    with pytest.raises(BootstrapError, match="linked private-data"):
        if operation == "reset":
            reset_personal_learning(
                tmp_path,
                confirmed=True,
                personal_root=personal_root,
            )
        else:
            erase_personal_learning(
                tmp_path,
                confirmed=True,
                permanent=True,
                personal_root=personal_root,
            )

    assert sentinel.read_text(encoding="utf-8") == "unrelated"
    assert not (data / "reset-backups").exists()


def test_personal_restore_refuses_a_linked_backup_item(
    tmp_path: Path, monkeypatch
) -> None:
    backup = tmp_path / "data" / "reset-backups" / "personal-reset-linked"
    backup.mkdir(parents=True)
    protected = backup / "must-survive.db"
    protected.write_text("private", encoding="utf-8")
    linked = backup / "data__personal-conversations.db"
    try:
        linked.symlink_to(protected)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are unavailable on this platform")
    (backup / "reset-manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "items": [
                    {
                        "original": "data/personal-conversations.db",
                        "stored": linked.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("mini_arithmetic.bootstrap._port_is_open", lambda _port: False)

    with pytest.raises(BootstrapError, match="linked private-data"):
        restore_personal_learning(
            tmp_path,
            backup_name=backup.name,
            confirmed=True,
        )

    assert protected.read_text(encoding="utf-8") == "private"
    assert not (tmp_path / "data" / "personal-conversations.db").exists()


def test_personal_restore_refuses_a_linked_destination_parent(
    tmp_path: Path, monkeypatch
) -> None:
    backup = tmp_path / "data" / "reset-backups" / "personal-reset-linked-parent"
    backup.mkdir(parents=True)
    stored = backup / "custom.db"
    stored.write_text("private", encoding="utf-8")
    protected = tmp_path / "protected"
    protected.mkdir()
    sentinel = protected / "must-survive.txt"
    sentinel.write_text("unrelated", encoding="utf-8")
    linked_parent = tmp_path / "data" / "linked-parent"
    try:
        linked_parent.symlink_to(protected, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable on this platform")
    (backup / "reset-manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "items": [
                    {
                        "original": "data/linked-parent/restored.db",
                        "stored": stored.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("mini_arithmetic.bootstrap._port_is_open", lambda _port: False)

    with pytest.raises(BootstrapError, match="linked private-data"):
        restore_personal_learning(
            tmp_path,
            backup_name=backup.name,
            confirmed=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "unrelated"
    assert stored.read_text(encoding="utf-8") == "private"
    assert not (protected / "restored.db").exists()


@pytest.mark.parametrize("operation", ("reset", "erase"))
@pytest.mark.parametrize(
    "case",
    (
        "source-directory",
        "source-database",
        "base-model-directory",
        "base-model-database",
        "backup-child-directory",
        "database-directory",
        "personal-file",
    ),
)
def test_personal_data_commands_reject_non_private_direct_targets(
    tmp_path: Path,
    monkeypatch,
    operation: str,
    case: str,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    kwargs: dict[str, str]
    if case == "source-directory":
        protected = tmp_path / "src"
        protected.mkdir()
        sentinel = protected / "must-survive.txt"
        sentinel.write_text("source", encoding="utf-8")
        kwargs = {"personal_root": "src"}
    elif case == "source-database":
        protected = tmp_path / "src"
        protected.mkdir()
        sentinel = protected / "must-survive.db"
        sentinel.write_text("source-db", encoding="utf-8")
        kwargs = {"db_path": "src/must-survive.db"}
    elif case == "base-model-directory":
        protected = data / "models"
        protected.mkdir()
        sentinel = protected / "must-survive.txt"
        sentinel.write_text("base", encoding="utf-8")
        kwargs = {"personal_root": "data/models"}
    elif case == "base-model-database":
        protected = data / "models"
        protected.mkdir()
        sentinel = protected / "must-survive.db"
        sentinel.write_text("base-db", encoding="utf-8")
        kwargs = {"db_path": "data/models/must-survive.db"}
    elif case == "backup-child-directory":
        protected = data / "reset-backups" / "not-personal"
        protected.mkdir(parents=True)
        sentinel = protected / "must-survive.txt"
        sentinel.write_text("backup", encoding="utf-8")
        kwargs = {"personal_root": "data/reset-backups/not-personal"}
    elif case == "database-directory":
        protected = data / "not-a-database.db"
        protected.mkdir()
        sentinel = protected / "must-survive.txt"
        sentinel.write_text("directory", encoding="utf-8")
        kwargs = {"db_path": "data/not-a-database.db"}
    else:
        protected = data / "ordinary-file"
        protected.write_text("ordinary", encoding="utf-8")
        sentinel = protected
        kwargs = {"personal_root": "data/ordinary-file"}
    monkeypatch.setattr("mini_arithmetic.bootstrap._port_is_open", lambda _port: False)

    with pytest.raises(BootstrapError):
        if operation == "reset":
            reset_personal_learning(tmp_path, confirmed=True, **kwargs)
        else:
            erase_personal_learning(
                tmp_path,
                confirmed=True,
                permanent=True,
                **kwargs,
            )

    assert sentinel.exists()
    if sentinel.is_file():
        assert sentinel.read_text(encoding="utf-8") in {
            "source",
            "source-db",
            "base",
            "base-db",
            "backup",
            "directory",
            "ordinary",
        }


def test_custom_private_data_under_data_can_reset_restore_and_erase(
    tmp_path: Path, monkeypatch
) -> None:
    personal = tmp_path / "data" / "custom-personal"
    personal.mkdir(parents=True)
    database = tmp_path / "data" / "custom-conversations.sqlite3"
    (personal / "weights").write_text("learned", encoding="utf-8")
    database.write_text("private", encoding="utf-8")
    monkeypatch.setattr("mini_arithmetic.bootstrap._port_is_open", lambda _port: False)

    options = {
        "personal_root": "data/custom-personal",
        "db_path": "data/custom-conversations.sqlite3",
    }
    assert reset_personal_learning(tmp_path, confirmed=True, **options) == 0
    assert not personal.exists()
    assert not database.exists()
    assert restore_personal_learning(tmp_path, confirmed=True) == 0
    assert (personal / "weights").read_text(encoding="utf-8") == "learned"
    assert database.read_text(encoding="utf-8") == "private"
    assert (
        erase_personal_learning(
            tmp_path,
            confirmed=True,
            permanent=True,
            **options,
        )
        == 0
    )
    assert not personal.exists()
    assert not database.exists()


def test_personal_erasure_requires_two_confirmations_and_removes_backups(
    tmp_path: Path, monkeypatch
) -> None:
    data = tmp_path / "data"
    foundation = data / "personal-foundations"
    backups = data / "reset-backups" / "personal-reset-old"
    foundation.mkdir(parents=True)
    backups.mkdir(parents=True)
    database = data / "personal-conversations.db"
    database.write_text("private", encoding="utf-8")
    (foundation / "weights").write_text("learned", encoding="utf-8")
    (backups / "old").write_text("private", encoding="utf-8")
    monkeypatch.setattr("mini_arithmetic.bootstrap._port_is_open", lambda _port: False)

    assert erase_personal_learning(tmp_path, confirmed=True, permanent=False) == 1
    assert database.exists()
    assert erase_personal_learning(tmp_path, confirmed=True, permanent=True) == 0
    assert not database.exists()
    assert not foundation.exists()
    assert not (data / "reset-backups").exists()
