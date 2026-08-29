"""Backend-neutral device detection for local tinydog.ai runs.

This module imports a compute engine only after platform and package checks.
It remains safe on platforms where MLX cannot be installed, while an Apple
Silicon readiness report also proves that the native MLX runtime can execute a
minimal array operation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import importlib.util
import json
import platform
from types import ModuleType
from typing import Callable, Literal


BackendName = Literal["auto", "mlx", "torch"]
SelectedBackend = Literal["mlx", "torch", "none"]
AcceleratorName = Literal["metal", "cuda", "rocm", "xpu", "mps", "cpu", "none"]


@dataclass(frozen=True)
class BackendSelection:
    """A side-effect-free description of the selected local compute path."""

    requested: BackendName
    backend: SelectedBackend
    accelerator: AcceleratorName
    ready: bool
    system: str
    machine: str
    install_extra: Literal["mlx", "torch"]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _load_module(name: str) -> ModuleType:
    return importlib.import_module(name)


def is_apple_silicon(system: str, machine: str) -> bool:
    return system.casefold() == "darwin" and machine.casefold() in {"arm64", "aarch64"}


def recommended_backend_for_host(
    *, system: str | None = None, machine: str | None = None
) -> Literal["mlx", "torch"]:
    """Return the dependency family setup should install on this host."""

    host_system = system or platform.system()
    host_machine = machine or platform.machine()
    return "mlx" if is_apple_silicon(host_system, host_machine) else "torch"


def _safe_available(container: object, attribute: str = "is_available") -> bool:
    try:
        probe = getattr(container, attribute)
        return bool(probe())
    except (AttributeError, RuntimeError, TypeError):
        return False


def _torch_accelerator(torch: ModuleType) -> tuple[AcceleratorName, str]:
    cuda = getattr(torch, "cuda", None)
    torch_version = getattr(torch, "version", None)
    hip_version = getattr(torch_version, "hip", None)
    if cuda is not None and _safe_available(cuda):
        if hip_version:
            return "rocm", f"PyTorch sees an AMD GPU through ROCm {hip_version}"
        cuda_version = getattr(torch_version, "cuda", None)
        suffix = f" {cuda_version}" if cuda_version else ""
        return "cuda", f"PyTorch sees an NVIDIA GPU through CUDA{suffix}"

    xpu = getattr(torch, "xpu", None)
    if xpu is not None and _safe_available(xpu):
        return "xpu", "PyTorch sees an Intel XPU device"

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and _safe_available(mps):
        return "mps", "PyTorch sees Apple Metal Performance Shaders"

    return (
        "cpu",
        "PyTorch is installed; no supported accelerator is currently available",
    )


def select_backend(
    requested: BackendName = "auto",
    *,
    system: str | None = None,
    machine: str | None = None,
    module_exists: Callable[[str], bool] = _module_exists,
    module_loader: Callable[[str], ModuleType] = _load_module,
) -> BackendSelection:
    """Select MLX on Apple Silicon when present, otherwise an installed Torch.

    ``module_exists`` and ``module_loader`` are injectable so callers can test
    platform combinations without requiring either framework.
    """

    if requested not in {"auto", "mlx", "torch"}:
        raise ValueError("requested backend must be auto, mlx, or torch")

    host_system = system or platform.system()
    host_machine = machine or platform.machine()
    apple_silicon = is_apple_silicon(host_system, host_machine)
    recommendation = recommended_backend_for_host(
        system=host_system, machine=host_machine
    )
    mlx_present = apple_silicon and module_exists("mlx")

    mlx_error: str | None = None
    if requested in {"auto", "mlx"} and mlx_present:
        try:
            mx = module_loader("mlx.core")
            probe = mx.array([0])
            mx.eval(probe)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError) as error:
            mlx_error = str(error)
        else:
            return BackendSelection(
                requested=requested,
                backend="mlx",
                accelerator="metal",
                ready=True,
                system=host_system,
                machine=host_machine,
                install_extra="mlx",
                reason="MLX executed a Metal readiness probe on Apple Silicon",
            )

    if requested == "mlx":
        reason = (
            f"MLX is installed but could not run: {mlx_error}"
            if mlx_error is not None
            else (
                "MLX is not installed"
                if apple_silicon
                else "MLX is supported here only on Apple Silicon macOS"
            )
        )
        return BackendSelection(
            requested=requested,
            backend="none",
            accelerator="none",
            ready=False,
            system=host_system,
            machine=host_machine,
            install_extra="mlx",
            reason=reason,
        )

    if module_exists("torch"):
        try:
            torch = module_loader("torch")
            accelerator, reason = _torch_accelerator(torch)
        except (ImportError, OSError, RuntimeError) as error:
            return BackendSelection(
                requested=requested,
                backend="none",
                accelerator="none",
                ready=False,
                system=host_system,
                machine=host_machine,
                install_extra="torch",
                reason=f"PyTorch is installed but could not be loaded: {error}",
            )
        return BackendSelection(
            requested=requested,
            backend="torch",
            accelerator=accelerator,
            ready=True,
            system=host_system,
            machine=host_machine,
            install_extra="torch",
            reason=reason,
        )

    return BackendSelection(
        requested=requested,
        backend="none",
        accelerator="none",
        ready=False,
        system=host_system,
        machine=host_machine,
        install_extra=recommendation,
        reason=f"No compatible engine is installed; install the [{recommendation}] dependency bundle",
    )


def format_selection(selection: BackendSelection) -> str:
    state = "ready" if selection.ready else "not ready"
    engine = (
        f"{selection.backend}/{selection.accelerator}"
        if selection.backend != "none"
        else "none"
    )
    return "\n".join(
        (
            f"Device:  {selection.system} {selection.machine}",
            f"Engine:  {engine} ({state})",
            f"Details: {selection.reason}",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show the local tinydog.ai compute engine"
    )
    parser.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable status"
    )
    args = parser.parse_args(argv)
    selection = select_backend(args.backend)
    if args.json:
        print(json.dumps(selection.as_dict(), sort_keys=True))
    else:
        print(format_selection(selection))
    return 0 if selection.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
