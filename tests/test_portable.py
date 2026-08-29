from types import SimpleNamespace

import pytest

from mini_arithmetic.portable import (
    format_selection,
    recommended_backend_for_host,
    select_backend,
)


def _torch(
    *, cuda: bool = False, hip: str | None = None, xpu: bool = False, mps: bool = False
):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        xpu=SimpleNamespace(is_available=lambda: xpu),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        version=SimpleNamespace(cuda="12.8" if cuda and not hip else None, hip=hip),
    )


def _mlx():
    return SimpleNamespace(array=lambda value: tuple(value), eval=lambda _value: None)


def test_auto_prefers_mlx_after_an_executable_apple_silicon_probe() -> None:
    loaded: list[str] = []
    selection = select_backend(
        system="Darwin",
        machine="arm64",
        module_exists=lambda name: name == "mlx",
        module_loader=lambda name: (loaded.append(name), _mlx())[1],
    )

    assert selection.ready is True
    assert selection.backend == "mlx"
    assert selection.accelerator == "metal"
    assert loaded == ["mlx.core"]


def test_windows_probe_never_even_checks_for_mlx() -> None:
    checked: list[str] = []

    def exists(name: str) -> bool:
        checked.append(name)
        if name == "mlx":
            raise AssertionError("MLX must not be probed on Windows")
        return name == "torch"

    selection = select_backend(
        system="Windows",
        machine="AMD64",
        module_exists=exists,
        module_loader=lambda _name: _torch(cuda=True),  # type: ignore[arg-type]
    )

    assert selection.backend == "torch"
    assert selection.accelerator == "cuda"
    assert checked == ["torch"]


@pytest.mark.parametrize(
    ("torch", "accelerator"),
    (
        (_torch(cuda=True), "cuda"),
        (_torch(cuda=True, hip="6.4"), "rocm"),
        (_torch(xpu=True), "xpu"),
        (_torch(mps=True), "mps"),
        (_torch(), "cpu"),
    ),
)
def test_torch_accelerator_detection(torch, accelerator: str) -> None:
    selection = select_backend(
        system="Linux",
        machine="x86_64",
        module_exists=lambda name: name == "torch",
        module_loader=lambda _name: torch,
    )

    assert selection.ready is True
    assert selection.backend == "torch"
    assert selection.accelerator == accelerator


def test_auto_falls_back_to_torch_on_apple_when_mlx_is_missing() -> None:
    selection = select_backend(
        system="Darwin",
        machine="arm64",
        module_exists=lambda name: name == "torch",
        module_loader=lambda _name: _torch(mps=True),
    )

    assert selection.backend == "torch"
    assert selection.accelerator == "mps"


def test_missing_backend_reports_host_specific_install_bundle() -> None:
    apple = select_backend(
        system="Darwin",
        machine="arm64",
        module_exists=lambda _name: False,
    )
    linux = select_backend(
        system="Linux",
        machine="x86_64",
        module_exists=lambda _name: False,
    )

    assert apple.ready is False
    assert apple.install_extra == "mlx"
    assert linux.install_extra == "torch"
    assert "not ready" in format_selection(linux)


def test_explicit_mlx_is_rejected_away_from_apple_silicon() -> None:
    selection = select_backend(
        "mlx",
        system="Linux",
        machine="aarch64",
        module_exists=lambda _name: True,
    )

    assert selection.ready is False
    assert selection.backend == "none"
    assert "Apple Silicon" in selection.reason


def test_recommended_backend_uses_host_not_installed_modules() -> None:
    assert recommended_backend_for_host(system="Darwin", machine="arm64") == "mlx"
    assert recommended_backend_for_host(system="Windows", machine="AMD64") == "torch"
    assert recommended_backend_for_host(system="Linux", machine="aarch64") == "torch"


def test_broken_torch_install_is_not_reported_ready() -> None:
    def broken(_name: str):
        raise OSError("missing runtime library")

    selection = select_backend(
        "torch",
        system="Windows",
        machine="AMD64",
        module_exists=lambda _name: True,
        module_loader=broken,
    )

    assert selection.ready is False
    assert "could not be loaded" in selection.reason


def test_broken_mlx_install_is_not_reported_ready() -> None:
    def broken(_name: str):
        raise OSError("missing native MLX runtime")

    selection = select_backend(
        "mlx",
        system="Darwin",
        machine="arm64",
        module_exists=lambda name: name == "mlx",
        module_loader=broken,
    )

    assert selection.ready is False
    assert selection.backend == "none"
    assert "could not run" in selection.reason
