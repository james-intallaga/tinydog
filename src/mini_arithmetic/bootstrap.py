"""Cross-platform setup and process launcher for local tinydog.ai.

Only dependencies and source code are installed. This launcher never fetches
tinydog language-model checkpoints, pretrained language-model weights, or
training conversations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .portable import BackendName, recommended_backend_for_host


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUP_MARKER = ".tinydog-setup.json"
DEFAULT_CONFIG = "configs/300m.json"
DEFAULT_WEIGHTS = "checkpoints/300m/random-apprentice-init.safetensors"
LOCAL_UI_ORIGIN = "http://localhost:3000"
LOCAL_SERVICE_PORTS = (3000, 8765, 8787)


class BootstrapError(RuntimeError):
    """A setup problem with an actionable, non-technical message."""


class _LauncherStopRequested(Exception):
    """A process signal requested orderly child-tree cleanup."""


@dataclass(frozen=True)
class SetupPlan:
    backend: str
    python: Path
    commands: tuple[tuple[str, ...], ...]


def _node_version_supported(version_text: str) -> bool:
    """Return whether a ``node --version`` response satisfies the UI floor."""

    value = version_text.strip().lower()
    if value.startswith("v"):
        value = value[1:]
    pieces = value.split(".")
    if len(pieces) < 2 or not pieces[0].isdigit() or not pieces[1].isdigit():
        return False
    return (int(pieces[0]), int(pieces[1])) >= (22, 13)


def _require_supported_node() -> None:
    node = shutil.which("node.exe" if os.name == "nt" else "node")
    if node is None:
        raise BootstrapError(
            "Node.js 22.13 or newer is needed for the local screen. "
            "Install Node.js, then run setup again."
        )
    try:
        result = subprocess.run(
            (node, "--version"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BootstrapError(
            "Could not verify the installed Node.js version."
        ) from error
    if not _node_version_supported(result.stdout):
        raise BootstrapError(
            "The local screen needs Node.js 22.13 or newer. "
            f"This computer reported {result.stdout.strip() or 'an unknown version'}."
        )


def _venv_python(root: Path, *, windows: bool | None = None) -> Path:
    use_windows = os.name == "nt" if windows is None else windows
    return root / ".venv" / ("Scripts/python.exe" if use_windows else "bin/python")


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _setup_fingerprint(
    root: Path,
    backend: str,
    *,
    ui_installed: bool = True,
    realtime_installed: bool | None = None,
) -> dict[str, object]:
    if realtime_installed is None:
        realtime_installed = backend == "mlx"
    return {
        "format": 1,
        "backend": backend,
        "ui_installed": ui_installed,
        "realtime_installed": realtime_installed,
        "pyproject_sha256": _file_sha256(root / "pyproject.toml"),
        "package_lock_sha256": _file_sha256(root / "interface" / "package-lock.json"),
    }


def _load_marker(root: Path) -> dict[str, object] | None:
    try:
        payload = json.loads((root / SETUP_MARKER).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_marker(root: Path, payload: Mapping[str, object]) -> None:
    marker = root / SETUP_MARKER
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(marker)


def _resolve_setup_backend(
    requested: BackendName, *, system: str | None = None, machine: str | None = None
) -> str:
    if requested == "auto":
        return recommended_backend_for_host(system=system, machine=machine)
    if (
        requested == "mlx"
        and recommended_backend_for_host(system=system, machine=machine) != "mlx"
    ):
        raise BootstrapError(
            "MLX setup is available only on an Apple Silicon Mac. Use --backend torch here."
        )
    return requested


def build_setup_plan(
    root: Path,
    *,
    backend: str,
    source_python: str = sys.executable,
    npm: str = "npm",
    torch_index_url: str | None = None,
    install_ui: bool = True,
    install_realtime: bool = True,
    windows: bool | None = None,
) -> SetupPlan:
    """Build deterministic dependency-only setup commands."""

    python = _venv_python(root, windows=windows)
    commands: list[tuple[str, ...]] = []
    if not python.is_file():
        commands.append((source_python, "-m", "venv", str(root / ".venv")))

    # Torch currently serves the portable text learner only.  Do not make a
    # Windows/Linux installation compile or resolve the unrelated MLX-native
    # WebRTC/audio stack before that backend exists there.
    extras = (
        "realtime,mlx"
        if backend == "mlx" and install_realtime
        else ("mlx" if backend == "mlx" else "")
    )
    install_target = f"{root}[{extras}]" if extras else str(root)
    commands.append(
        (
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-e",
            install_target,
        )
    )
    if backend == "torch":
        torch_command = [str(python), "-m", "pip", "install", "torch>=2.4,<3"]
        if torch_index_url:
            torch_command.extend(("--index-url", torch_index_url))
        commands.append(tuple(torch_command))
    if install_ui:
        commands.append(
            (
                npm,
                "ci",
                "--omit=dev",
                "--no-audit",
                "--no-fund",
                "--loglevel=error",
            )
        )
    return SetupPlan(backend=backend, python=python, commands=tuple(commands))


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(command)


def setup_project(
    root: Path,
    *,
    requested: BackendName = "auto",
    torch_index_url: str | None = None,
    install_ui: bool = True,
    install_realtime: bool = True,
    dry_run: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
) -> SetupPlan:
    if sys.version_info < (3, 10):
        raise BootstrapError("tinydog.ai needs Python 3.10 or newer.")
    backend = _resolve_setup_backend(requested)
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm") if install_ui else None
    if install_ui and npm is None:
        raise BootstrapError(
            "Node.js 22.13 or newer is needed for the local screen. "
            "Install Node.js, then run setup again."
        )
    if install_ui:
        _require_supported_node()

    plan = build_setup_plan(
        root,
        backend=backend,
        npm=npm or "npm",
        torch_index_url=torch_index_url,
        install_ui=install_ui,
        install_realtime=install_realtime,
    )
    print(
        f"Setting up the {backend.upper()} engine. "
        "No tinydog language-model weights will be downloaded."
    )
    for index, command in enumerate(plan.commands):
        print(f"  {_display_command(command)}")
        if dry_run:
            continue
        cwd = (
            root / "interface"
            if install_ui and index == len(plan.commands) - 1
            else root
        )
        runner(command, cwd=cwd, check=True)
    if not dry_run:
        _write_marker(
            root,
            _setup_fingerprint(
                root,
                backend,
                ui_installed=install_ui,
                realtime_installed=install_realtime and backend == "mlx",
            ),
        )
        print("Setup complete.")
    return plan


def _needs_setup(
    root: Path,
    requested: BackendName,
    *,
    require_ui: bool = True,
    require_realtime: bool = True,
) -> bool:
    marker = _load_marker(root)
    if marker is None or not _venv_python(root).is_file():
        return True
    marker_backend = marker.get("backend")
    backend = marker_backend if requested == "auto" else requested
    if backend not in {"mlx", "torch"}:
        return True
    ui_installed = marker.get("ui_installed") is True
    if require_ui and not ui_installed:
        return True
    realtime_installed = marker.get("realtime_installed") is True
    if backend == "mlx" and require_realtime and not realtime_installed:
        return True
    return marker != _setup_fingerprint(
        root,
        str(backend),
        ui_installed=ui_installed,
        realtime_installed=realtime_installed,
    )


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=value settings without executing the file as a script."""

    result: dict[str, str] = {}
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise BootstrapError(
                "The local .env file must be a regular, unlinked file."
            )
        if os.name != "nt" and stat.S_IMODE(metadata.st_mode) & 0o077:
            try:
                path.chmod(stat.S_IMODE(metadata.st_mode) & ~0o077)
            except OSError as error:
                raise BootstrapError(
                    "The local .env file permissions could not be made private."
                ) from error
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return result
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if (
            not separator
            or not key
            or not key.replace("_", "a").isalnum()
            or key[0].isdigit()
        ):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result[key] = value
    return result


def _backend_from_marker(root: Path, requested: BackendName) -> str:
    if requested != "auto":
        return requested
    marker = _load_marker(root)
    backend = marker.get("backend") if marker else None
    if backend not in {"mlx", "torch"}:
        raise BootstrapError(
            "Setup did not record a usable compute engine. Run setup again."
        )
    return str(backend)


def _configured_model_size(requested: str, settings: Mapping[str, str]) -> str:
    """Honor the private `.env` default unless the command line is explicit."""

    if requested not in {"auto", "tiny", "standard"}:
        raise BootstrapError("Model size must be auto, tiny, or standard.")
    if requested != "auto":
        return requested
    configured = settings.get("TINYDOG_MODEL_SIZE", "auto").strip().lower()
    if configured not in {"auto", "tiny", "standard"}:
        raise BootstrapError("TINYDOG_MODEL_SIZE must be auto, tiny, or standard.")
    return configured


def _runtime_environment(
    settings: Mapping[str, str],
    *,
    backend: str,
    model_size: str,
) -> dict[str, str]:
    """Build the private child environment without outbound UI telemetry."""

    environment = dict(os.environ)
    for key, value in settings.items():
        environment.setdefault(key, value)
    environment["TINYDOG_BACKEND"] = backend
    environment["TINYDOG_MODEL_SIZE"] = model_size
    environment["MLM_BACKEND"] = backend
    environment["PYTHONUNBUFFERED"] = "1"
    # Next build/start telemetry is unrelated to the local learner and is
    # disabled unconditionally. Conversation content never leaves the machine.
    environment["NEXT_TELEMETRY_DISABLED"] = "1"
    return environment


def build_start_commands(
    root: Path,
    *,
    backend: str,
    model_size: str = "auto",
    legacy_mlx: bool = False,
    config: str = DEFAULT_CONFIG,
    weights: str = DEFAULT_WEIGHTS,
    model_root: str | None = None,
    personal_root: str | None = None,
    db_path: str | None = None,
    include_realtime: bool = True,
    include_ui: bool = True,
    npm: str = "npm",
    windows: bool | None = None,
) -> tuple[tuple[str, Path, tuple[str, ...]], ...]:
    """Return named process commands without starting them."""

    python = str(_venv_python(root, windows=windows))
    if legacy_mlx:
        if backend != "mlx":
            raise BootstrapError(
                "--legacy-mlx requires the MLX engine on an Apple Silicon Mac."
            )
        service = (
            python,
            "-m",
            "mini_arithmetic.serve",
            "--config",
            config,
            "--weights",
            weights,
        )
    else:
        portable_service = [
            python,
            "-m",
            "mini_arithmetic.portable_serve",
            "--backend",
            backend,
            "--model-size",
            model_size,
        ]
        if model_root:
            portable_service.extend(("--model-root", model_root))
        if personal_root:
            portable_service.extend(("--personal-root", personal_root))
        if db_path:
            portable_service.extend(("--db", db_path))
        service = tuple(portable_service)

    commands: list[tuple[str, Path, tuple[str, ...]]] = [
        ("learning service", root, service)
    ]
    if include_realtime and backend == "mlx":
        commands.append(
            (
                "voice transport",
                root,
                (python, "-m", "mini_arithmetic.realtime_gateway"),
            )
        )
    if include_ui:
        commands.append(
            (
                "local screen",
                root / "interface",
                (npm, "run", "start"),
            )
        )
    return tuple(commands)


def _realtime_command(
    command: tuple[str, ...], env: Mapping[str, str]
) -> tuple[str, ...]:
    checkpoint = env.get("MLM_NATIVE_CHECKPOINT", "").strip()
    untrained_config = env.get("MLM_UNTRAINED_NATIVE_CONFIG", "").strip()
    if checkpoint:
        return command + ("--checkpoint", checkpoint)
    if untrained_config:
        return command + (
            "--untrained-config",
            untrained_config,
            "--untrained-seed",
            env.get("MLM_UNTRAINED_NATIVE_SEED", "0"),
        )
    return command


def _process_group_options(*, windows: bool | None = None) -> dict[str, object]:
    """Launch each service in a tree that the bootstrap can stop as a unit."""

    is_windows = os.name == "nt" if windows is None else windows
    if is_windows:
        return {
            "creationflags": getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0x00000200,
            )
        }
    return {"start_new_session": True}


def _request_process_tree_stop(
    process: subprocess.Popen[object],
    *,
    windows: bool | None = None,
) -> None:
    if process.poll() is not None:
        return
    is_windows = os.name == "nt" if windows is None else windows
    process_id = getattr(process, "pid", None)
    if isinstance(process_id, int) and process_id > 0:
        try:
            if is_windows:
                control_break = getattr(signal, "CTRL_BREAK_EVENT", None)
                if control_break is not None:
                    os.kill(process_id, control_break)
                    return
            else:
                # start_new_session=True makes the child PID its process-group ID.
                os.killpg(process_id, signal.SIGTERM)
                return
        except OSError:
            pass
    process.terminate()


def _force_process_tree_stop(
    process: subprocess.Popen[object],
    *,
    windows: bool | None = None,
) -> None:
    if process.poll() is not None:
        return
    is_windows = os.name == "nt" if windows is None else windows
    process_id = getattr(process, "pid", None)
    if is_windows and isinstance(process_id, int) and process_id > 0:
        try:
            subprocess.run(
                ("taskkill", "/PID", str(process_id), "/T", "/F"),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return
        except (OSError, subprocess.SubprocessError):
            pass
    elif isinstance(process_id, int) and process_id > 0:
        try:
            os.killpg(process_id, signal.SIGKILL)
            return
        except OSError:
            pass
    process.kill()


def _install_launcher_stop_handlers() -> dict[int, object]:
    """Convert external termination into the same cleanup path as Ctrl+C."""

    previous: dict[int, object] = {}

    def request_stop(_signum: int, _frame: object) -> None:
        raise _LauncherStopRequested

    for signal_name in ("SIGTERM", "SIGHUP"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is None:
            continue
        try:
            previous[int(signal_number)] = signal.getsignal(signal_number)
            signal.signal(signal_number, request_stop)
        except ValueError:
            # Embedded callers may invoke the launcher outside the main thread.
            previous.pop(int(signal_number), None)
    return previous


def _restore_launcher_stop_handlers(previous: Mapping[int, object]) -> None:
    for signal_number, handler in previous.items():
        try:
            signal.signal(signal_number, handler)  # type: ignore[arg-type]
        except ValueError:
            pass


def _stop_processes(processes: Sequence[subprocess.Popen[object]]) -> None:
    for process in reversed(processes):
        _request_process_tree_stop(process)
    deadline = time.monotonic() + 5.0
    for process in reversed(processes):
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _force_process_tree_stop(process)


def start_project(
    root: Path,
    *,
    requested: BackendName = "auto",
    model_size: str = "auto",
    legacy_mlx: bool = False,
    config: str = DEFAULT_CONFIG,
    weights: str = DEFAULT_WEIGHTS,
    model_root: str | None = None,
    personal_root: str | None = None,
    db_path: str | None = None,
    include_realtime: bool = True,
    include_ui: bool = True,
    torch_index_url: str | None = None,
    dry_run: bool = False,
) -> int:
    if _needs_setup(
        root,
        requested,
        require_ui=include_ui,
        require_realtime=include_realtime,
    ):
        setup_project(
            root,
            requested=requested,
            torch_index_url=torch_index_url,
            install_ui=include_ui,
            install_realtime=include_realtime,
            dry_run=dry_run,
        )
        if dry_run:
            return 0

    backend = _backend_from_marker(root, requested)
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm") if include_ui else None
    if include_ui and npm is None:
        raise BootstrapError(
            "Node.js is missing. Install Node.js, then start tinydog.ai again."
        )
    if include_ui:
        _require_supported_node()

    dotenv = _read_dotenv(root / ".env")
    selected_model_size = _configured_model_size(model_size, dotenv)
    commands = build_start_commands(
        root,
        backend=backend,
        model_size=selected_model_size,
        legacy_mlx=legacy_mlx,
        config=config,
        weights=weights,
        model_root=model_root,
        personal_root=personal_root,
        db_path=db_path,
        include_realtime=include_realtime,
        include_ui=include_ui,
        npm=npm or "npm",
    )
    if include_realtime and backend != "mlx":
        print(
            "Voice transport is not started on the Torch path yet; "
            "the current scratch-native voice runtime is MLX-only."
        )
    environment = _runtime_environment(
        dotenv,
        backend=backend,
        model_size=selected_model_size,
    )
    if include_ui:
        build_command = (npm or "npm", "run", "build")
        print(f"Building local screen: {_display_command(build_command)}")
        if not dry_run:
            subprocess.run(
                build_command,
                cwd=root / "interface",
                env=environment,
                check=True,
            )
        # Generate the capability after the build so its value cannot be baked
        # into static output.  Only the Python service and server-side Next
        # process receive it at runtime.
        environment["TINYDOG_LOCAL_API_TOKEN"] = secrets.token_urlsafe(32)

    processes: list[subprocess.Popen[object]] = []
    previous_signal_handlers = _install_launcher_stop_handlers()
    try:
        for name, cwd, original_command in commands:
            command = (
                _realtime_command(original_command, environment)
                if name == "voice transport"
                else original_command
            )
            print(f"Starting {name}: {_display_command(command)}")
            if not dry_run:
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=environment,
                        **_process_group_options(),
                    )
                )
        if dry_run:
            return 0

        print("tinydog.ai is starting at http://localhost:3000 — press Ctrl+C to stop.")
        while True:
            for index, process in enumerate(processes):
                code = process.poll()
                if code is not None:
                    name = commands[index][0]
                    raise BootstrapError(
                        f"The {name} stopped unexpectedly (exit code {code})."
                    )
            time.sleep(0.25)
    except (KeyboardInterrupt, _LauncherStopRequested):
        print("\nStopping tinydog.ai…")
        return 0
    finally:
        _stop_processes(processes)
        _restore_launcher_stop_handlers(previous_signal_handlers)


def doctor(root: Path, requested: BackendName) -> int:
    python = _venv_python(root)
    if not python.is_file():
        print("tinydog.ai is not set up yet. Run the setup command first.")
        return 1
    marker = _load_marker(root)
    recorded = marker.get("backend") if marker else None
    backend = (
        recorded if requested == "auto" and recorded in {"mlx", "torch"} else requested
    )
    command = (str(python), "-m", "mini_arithmetic.portable", "--backend", backend)
    return subprocess.run(command, cwd=root, check=False).returncode


def _local_ui_json(path: str, *, method: str = "GET") -> dict[str, object]:
    """Call a fixed same-origin UI route without exposing the service token."""

    if not path.startswith("/api/") or "?" in path or "#" in path:
        raise BootstrapError("The requested local action is invalid.")
    body = b"{}" if method == "POST" else None
    request = Request(
        f"{LOCAL_UI_ORIGIN}{path}",
        data=body,
        method=method,
        headers={
            "Accept": "application/json",
            "Origin": LOCAL_UI_ORIGIN,
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed loopback URL
            payload = response.read(1024 * 1024 + 1)
    except HTTPError as error:
        payload = error.read(1024 * 1024 + 1)
        try:
            detail = json.loads(payload).get("error")
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            detail = None
        raise BootstrapError(
            str(detail)
            if isinstance(detail, str) and detail
            else "The local action failed."
        ) from error
    except (OSError, URLError) as error:
        raise BootstrapError(
            "tinydog.ai is not running. Start it, then try this command again."
        ) from error
    if len(payload) > 1024 * 1024:
        raise BootstrapError("The local service returned too much data.")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError(
            "The local service returned an invalid response."
        ) from error
    if not isinstance(decoded, dict):
        raise BootstrapError("The local service returned an invalid response.")
    return decoded


def show_status(*, as_json: bool = False) -> int:
    payload = _local_ui_json("/api/status")
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    learning = payload.get("learning")
    model = payload.get("model")
    if not isinstance(learning, dict) or not isinstance(model, dict):
        raise BootstrapError("The local status response is incomplete.")
    pending = learning.get("qualifying_pending", 0)
    threshold = learning.get("lesson_threshold", 5)
    print(f"Learning: {learning.get('state', 'unknown')}")
    print(f"Progress: {pending}/{threshold} eligible typed turns")
    print(f"Mode:     {learning.get('parameter_scope', 'unknown')}")
    print(
        f"Model:    {model.get('backend', 'unknown')}/{model.get('device', 'unknown')}"
    )
    active = learning.get("active_foundation_id")
    print(f"Active:   {active if isinstance(active, str) and active else 'local base'}")
    print(
        "Rollback: "
        + (
            "available"
            if learning.get("rollback_available") is True
            else "not available"
        )
    )
    return 0


def rollback_personal_foundation() -> int:
    payload = _local_ui_json("/api/rollback", method="POST")
    state = payload.get("state", "ready")
    active = payload.get("active_foundation_id")
    print(f"Rollback complete. Learning state: {state}.")
    print(
        f"Active foundation: {active if isinstance(active, str) and active else 'local base'}"
    )
    return 0


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.15):
            return True
    except OSError:
        return False


def _require_local_services_stopped() -> None:
    active_ports = [port for port in LOCAL_SERVICE_PORTS if _port_is_open(port)]
    if active_ports:
        raise BootstrapError(
            "Stop tinydog.ai before changing its private files "
            "(a local service is still running)."
        )


def _inside_root(root: Path, candidate: Path) -> Path:
    """Return a lexical in-project path after rejecting every linked component.

    Resolving first would erase the evidence that a destructive target was a
    symlink.  Keep the path anchored to the canonical project root, inspect each
    existing component with ``lstat``, and reject any other filesystem
    redirection (including Windows junctions) before reset/restore/erase acts.
    """

    lexical_root = Path(os.path.abspath(root))
    resolved_root = lexical_root.resolve()
    lexical_candidate = Path(os.path.abspath(candidate))
    try:
        relative = lexical_candidate.relative_to(lexical_root)
    except ValueError as error:
        try:
            # Internal callers can pass a path already anchored to a resolved
            # project root even when ``root`` itself was reached through a link.
            relative = lexical_candidate.relative_to(resolved_root)
        except ValueError:
            raise BootstrapError(
                "Personal-data commands refuse paths outside this project. "
                "Move the files manually."
            ) from error

    safe = resolved_root.joinpath(*relative.parts)
    current = resolved_root
    for part in relative.parts:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            # No deeper component can exist once an ancestor is absent.
            break
        except OSError as error:
            raise BootstrapError(
                "Personal-data path safety could not be verified."
            ) from error
        if stat.S_ISLNK(mode):
            raise BootstrapError(
                "Personal-data commands refuse linked private-data paths."
            )

    try:
        resolved_candidate = safe.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise BootstrapError(
            "Personal-data path safety could not be verified."
        ) from error
    if resolved_candidate != safe:
        raise BootstrapError("Personal-data commands refuse linked private-data paths.")
    return safe


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
    except ValueError:
        try:
            right.relative_to(left)
        except ValueError:
            return False
    return True


def _private_data_target(root: Path, candidate: Path, *, kind: str) -> Path:
    """Validate a user-selectable private database or lineage target."""

    safe = _inside_root(root, candidate)
    data_root = _inside_root(root, root / "data")
    try:
        relative = safe.relative_to(data_root)
    except ValueError as error:
        raise BootstrapError(
            "Personal-data targets must stay under this project's data directory."
        ) from error
    if not relative.parts:
        raise BootstrapError(
            "The complete data directory contains base models and the backup "
            "directory; it is not a personal-data target."
        )

    for protected, label in (
        (data_root / "models", "base-model"),
        (data_root / "reset-backups", "reset-backup"),
    ):
        if _paths_overlap(safe, protected):
            raise BootstrapError(
                f"Personal-data targets must not overlap the {label} directory."
            )

    if kind == "database":
        if safe.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            raise BootstrapError(
                "A private conversation database must end in .db, .sqlite, or .sqlite3."
            )
        if safe.exists() and not safe.is_file():
            raise BootstrapError(
                "The private conversation database path is not a regular file."
            )
    elif kind == "directory":
        if safe.exists() and not safe.is_dir():
            raise BootstrapError(
                "A personal model path must be a directory, not a file."
            )
    else:  # pragma: no cover - internal contract
        raise ValueError("private data target kind is invalid")
    return safe


def _private_database_sidecar(root: Path, candidate: Path) -> Path:
    safe = _inside_root(root, candidate)
    raw = str(safe)
    matching_suffix = next(
        (suffix for suffix in ("-wal", "-shm") if raw.endswith(suffix)), None
    )
    database = Path(raw[: -len(matching_suffix)]) if matching_suffix else safe
    _private_data_target(root, database, kind="database")
    if safe.exists() and not safe.is_file():
        raise BootstrapError("A private database sidecar is not a regular file.")
    return safe


def _personal_reset_targets(
    root: Path,
    *,
    personal_root: str | None = None,
    db_path: str | None = None,
) -> tuple[Path, ...]:
    settings = _read_dotenv(root / ".env")
    configured_foundation = (
        personal_root
        or settings.get(
            "MLM_PERSONAL_FOUNDATION_DIR", "data/personal-foundations"
        ).strip()
    )
    configured_adapter = settings.get(
        "MLM_PERSONAL_MODEL_DIR", "data/personal-model"
    ).strip()
    configured_database = db_path or "data/personal-conversations.db"
    database = _private_data_target(root, root / configured_database, kind="database")
    foundation = _private_data_target(
        root, root / configured_foundation, kind="directory"
    )
    adapter = _private_data_target(root, root / configured_adapter, kind="directory")
    for first, second in (
        (database, foundation),
        (database, adapter),
        (foundation, adapter),
    ):
        if first != second and _paths_overlap(first, second):
            raise BootstrapError("Configured personal-data targets overlap each other.")

    candidates = (
        database,
        _private_database_sidecar(root, Path(f"{database}-wal")),
        _private_database_sidecar(root, Path(f"{database}-shm")),
        foundation,
        adapter,
    )
    protected_backup_root = _inside_root(root, root / "data" / "reset-backups")
    unique: list[Path] = []
    for candidate in candidates:
        resolved = _inside_root(root, candidate)
        try:
            protected_backup_root.relative_to(resolved)
        except ValueError:
            pass
        else:
            raise BootstrapError(
                "Reset refuses a private-data path that contains its backup directory."
            )
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def reset_personal_learning(
    root: Path,
    *,
    confirmed: bool = False,
    personal_root: str | None = None,
    db_path: str | None = None,
) -> int:
    """Move personal chat/training state to a private, recoverable backup."""

    targets = tuple(
        path
        for path in _personal_reset_targets(
            root, personal_root=personal_root, db_path=db_path
        )
        if path.exists()
    )
    if not targets:
        print("tinydog.ai is already at its beginning personal-learning state.")
        return 0
    if not confirmed:
        print(
            "This will reset private conversation history and learned personal weights."
        )
        print("The files will be moved to a private backup, not deleted.")
        print("Stop tinydog.ai, then rerun this command with --yes.")
        return 1
    _require_local_services_stopped()
    targets = tuple(_inside_root(root, path) for path in targets)

    backup_parent = _inside_root(root, root / "data" / "reset-backups")
    backup_parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        backup_parent.chmod(0o700)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = backup_parent / f"personal-reset-{stamp}-{secrets.token_hex(4)}"
    backup.mkdir(mode=0o700)
    moved: list[tuple[Path, Path]] = []
    try:
        for path in targets:
            relative = path.relative_to(root.resolve())
            destination = backup / "__".join(relative.parts)
            path.rename(destination)
            moved.append((path, destination))
        manifest = backup / "reset-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "format": 1,
                    "created_utc": stamp,
                    "items": [
                        {
                            "original": str(original.relative_to(root.resolve())),
                            "stored": destination.name,
                        }
                        for original, destination in moved
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if os.name != "nt":
            manifest.chmod(0o600)
    except OSError as error:
        restore_errors: list[OSError] = []
        for original, destination in reversed(moved):
            try:
                destination.rename(original)
            except OSError as restore_error:
                restore_errors.append(restore_error)
        if not restore_errors:
            try:
                (backup / "reset-manifest.json").unlink(missing_ok=True)
                backup.rmdir()
            except OSError:
                pass
        detail = (
            " Some items could not be restored; keep the backup directory."
            if restore_errors
            else " No personal files were reset."
        )
        raise BootstrapError(f"The private reset could not finish.{detail}") from error
    print(f"Reset complete. {len(moved)} private item(s) moved to:")
    print(backup)
    print(
        "The next start will use the unchanged local base model and a new conversation."
    )
    print(
        "To undo this reset while tinydog.ai is stopped, run the same launcher "
        "with: restore --yes"
    )
    return 0


def _reset_backup(root: Path, requested: str | None) -> Path:
    parent = _inside_root(root, root / "data" / "reset-backups")
    if requested:
        if Path(requested).name != requested or not requested.startswith(
            "personal-reset-"
        ):
            raise BootstrapError("The reset backup name is invalid.")
        candidates = (parent / requested,)
    else:
        candidates = tuple(sorted(parent.glob("personal-reset-*"), reverse=True))
    for raw_candidate in candidates:
        candidate = _inside_root(root, raw_candidate)
        if candidate.is_dir():
            manifest = _inside_root(root, candidate / "reset-manifest.json")
            if manifest.is_file():
                return candidate
    raise BootstrapError("No restorable personal reset backup was found.")


def restore_personal_learning(
    root: Path,
    *,
    backup_name: str | None = None,
    confirmed: bool = False,
) -> int:
    """Restore a complete reset backup without overwriting newer local state."""

    backup = _reset_backup(root, backup_name)
    if not confirmed:
        print(f"This will restore the private backup {backup.name}.")
        print("Stop tinydog.ai, then rerun this command with --yes.")
        return 1
    _require_local_services_stopped()
    try:
        payload = json.loads(
            (backup / "reset-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BootstrapError("The reset backup manifest is invalid.") from error
    if not isinstance(payload, dict):
        raise BootstrapError("The reset backup manifest is invalid.")
    items = payload.get("items")
    if payload.get("format") != 1 or not isinstance(items, list) or not items:
        raise BootstrapError("The reset backup manifest is invalid.")

    planned: list[tuple[Path, Path]] = []
    for item in items:
        if not isinstance(item, dict):
            raise BootstrapError("The reset backup manifest is invalid.")
        original_value = item.get("original")
        stored_value = item.get("stored")
        if (
            not isinstance(original_value, str)
            or not original_value
            or not isinstance(stored_value, str)
            or Path(stored_value).name != stored_value
        ):
            raise BootstrapError("The reset backup manifest is invalid.")
        original_path = Path(original_value)
        if original_path.is_absolute() or ".." in original_path.parts:
            raise BootstrapError("The reset backup manifest is invalid.")
        source = _inside_root(root, backup / stored_value)
        if source.parent != backup or not source.exists():
            raise BootstrapError("The reset backup is incomplete or linked.")
        if source.is_dir():
            destination = _private_data_target(
                root, root / original_path, kind="directory"
            )
        elif source.is_file():
            destination = _private_database_sidecar(root, root / original_path)
        else:
            raise BootstrapError("The reset backup contains an unsafe item.")
        if destination.exists():
            raise BootstrapError(
                "Personal learning files already exist. "
                "Reset them before restoring a backup."
            )
        planned.append((source, destination))

    restored: list[tuple[Path, Path]] = []
    try:
        for source, destination in planned:
            source = _inside_root(root, source)
            destination = _inside_root(root, destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            restored.append((source, destination))
    except OSError as error:
        restore_errors: list[OSError] = []
        for source, destination in reversed(restored):
            try:
                destination.rename(source)
            except OSError as restore_error:
                restore_errors.append(restore_error)
        detail = (
            " Some items could not be returned to the backup."
            if restore_errors
            else " No personal files were restored."
        )
        raise BootstrapError(
            f"The private restore could not finish.{detail}"
        ) from error
    try:
        (backup / "reset-manifest.json").unlink()
        backup.rmdir()
    except OSError:
        pass
    print(f"Restore complete from {backup.name}. Start tinydog.ai again.")
    return 0


def erase_personal_learning(
    root: Path,
    *,
    confirmed: bool = False,
    permanent: bool = False,
    personal_root: str | None = None,
    db_path: str | None = None,
) -> int:
    """Delete portable personal state after two explicit confirmations."""

    if not (confirmed and permanent):
        print(
            "This permanently deletes local chats, learned personal weights, and reset backups."
        )
        print("It keeps the original local base model. This action cannot be undone.")
        print("Stop tinydog.ai, then rerun with --yes --permanent to continue.")
        return 1
    _require_local_services_stopped()
    targets = [
        path
        for path in _personal_reset_targets(
            root, personal_root=personal_root, db_path=db_path
        )
        if path.exists()
    ]
    backup_root = _inside_root(root, root / "data" / "reset-backups")
    if backup_root.exists():
        targets.append(backup_root)
    targets = [_inside_root(root, path) for path in targets]
    failures: list[OSError] = []
    removed = 0
    for path in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except OSError as error:
            failures.append(error)
    if failures:
        raise BootstrapError(
            "Personal erasure was only partly completed. Stop using the app and "
            "inspect the private data directory before restarting."
        )
    print(f"Erasure complete. Removed {removed} private item(s).")
    print("The next start will use the unchanged local base model and an empty ledger.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinydog",
        description="Set up and run a private tinydog.ai learner on this computer.",
    )
    subparsers = parser.add_subparsers(dest="command")
    for command in ("setup", "start"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--backend", choices=("auto", "mlx", "torch"), default="auto"
        )
        child.add_argument(
            "--torch-index-url", help="optional official PyTorch wheel index"
        )
        child.add_argument(
            "--no-ui", action="store_true", help="skip the local web screen"
        )
        child.add_argument(
            "--no-realtime",
            action="store_true",
            help="skip the optional camera/microphone transport",
        )
        child.add_argument(
            "--dry-run",
            action="store_true",
            help="show actions without changing anything",
        )
    start = subparsers.choices["start"]
    start.add_argument(
        "--model-size",
        choices=("auto", "tiny", "standard"),
        default="auto",
        help="choose automatically, a lightweight pilot, or the standard 320.9M model",
    )
    start.add_argument(
        "--legacy-mlx",
        action="store_true",
        help="run the original MLX-only arithmetic service",
    )
    start.add_argument(
        "--model-root",
        help="directory containing portable-tiny or portable-standard",
    )
    start.add_argument(
        "--personal-root",
        help="private lineage directory for the selected foundation",
    )
    start.add_argument("--db", help="private conversation database path")
    start.add_argument("--config", default=DEFAULT_CONFIG, help=argparse.SUPPRESS)
    start.add_argument("--weights", default=DEFAULT_WEIGHTS, help=argparse.SUPPRESS)
    check = subparsers.add_parser("doctor")
    check.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    status = subparsers.add_parser("status", help="show local learning progress")
    status.add_argument("--json", action="store_true", help="print the complete status")
    subparsers.add_parser("rollback", help="return to the previous personal foundation")
    reset = subparsers.add_parser(
        "reset", help="back up and reset personal conversation learning"
    )
    reset.add_argument(
        "--yes", action="store_true", help="confirm the recoverable local reset"
    )
    reset.add_argument("--personal-root", help="private lineage directory to reset")
    reset.add_argument("--db", help="private conversation database to reset")
    restore = subparsers.add_parser(
        "restore", help="restore the newest personal reset backup"
    )
    restore.add_argument(
        "backup", nargs="?", help="optional personal-reset-* backup directory name"
    )
    restore.add_argument(
        "--yes", action="store_true", help="confirm the private backup restore"
    )
    erase = subparsers.add_parser(
        "erase", help="permanently delete all portable personal learning state"
    )
    erase.add_argument("--yes", action="store_true", help="confirm personal erasure")
    erase.add_argument(
        "--permanent",
        action="store_true",
        help="acknowledge that personal files and backups cannot be restored",
    )
    erase.add_argument("--personal-root", help="private lineage directory to erase")
    erase.add_argument("--db", help="private conversation database to erase")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command or "start"
    try:
        if command == "setup":
            setup_project(
                PROJECT_ROOT,
                requested=args.backend,
                torch_index_url=args.torch_index_url,
                install_ui=not args.no_ui,
                install_realtime=not args.no_realtime,
                dry_run=args.dry_run,
            )
            return 0
        if command == "doctor":
            return doctor(PROJECT_ROOT, args.backend)
        if command == "status":
            return show_status(as_json=args.json)
        if command == "rollback":
            return rollback_personal_foundation()
        if command == "reset":
            return reset_personal_learning(
                PROJECT_ROOT,
                confirmed=args.yes,
                personal_root=args.personal_root,
                db_path=args.db,
            )
        if command == "restore":
            return restore_personal_learning(
                PROJECT_ROOT, backup_name=args.backup, confirmed=args.yes
            )
        if command == "erase":
            return erase_personal_learning(
                PROJECT_ROOT,
                confirmed=args.yes,
                permanent=args.permanent,
                personal_root=args.personal_root,
                db_path=args.db,
            )
        return start_project(
            PROJECT_ROOT,
            requested=getattr(args, "backend", "auto"),
            model_size=getattr(args, "model_size", "auto"),
            legacy_mlx=getattr(args, "legacy_mlx", False),
            config=getattr(args, "config", DEFAULT_CONFIG),
            weights=getattr(args, "weights", DEFAULT_WEIGHTS),
            model_root=getattr(args, "model_root", None),
            personal_root=getattr(args, "personal_root", None),
            db_path=getattr(args, "db", None),
            include_realtime=not getattr(args, "no_realtime", False),
            include_ui=not getattr(args, "no_ui", False),
            torch_index_url=getattr(args, "torch_index_url", None),
            dry_run=getattr(args, "dry_run", False),
        )
    except (BootstrapError, OSError, subprocess.CalledProcessError) as error:
        print(f"tinydog.ai could not continue: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
