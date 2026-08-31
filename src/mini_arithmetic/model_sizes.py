"""Canonical portable model-size profiles and user-facing aliases.

The public profiles are deliberately named by approximate parameter count.
The earlier ``tiny`` and ``standard`` profiles remain valid immutable
identities so existing bundles and personal learning lineages keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ParameterDType = Literal["bfloat16", "float32"]


@dataclass(frozen=True)
class ModelSizePreset:
    """One immutable portable architecture identity."""

    identifier: str
    parameters: int
    context_length: int
    parameter_dtype: ParameterDType
    description: str
    legacy: bool = False

    @property
    def config_filename(self) -> str:
        return f"portable-{self.identifier}.json"

    @property
    def directory_name(self) -> str:
        return f"portable-{self.identifier}"


PUBLIC_MODEL_SIZE_IDS = ("20m", "50m", "80m", "100m", "200m", "300m")
LEGACY_MODEL_SIZE_IDS = ("tiny", "standard")
MODEL_SIZE_IDS = PUBLIC_MODEL_SIZE_IDS + LEGACY_MODEL_SIZE_IDS
MODEL_SIZE_CHOICES = ("auto",) + MODEL_SIZE_IDS
DEFAULT_NEW_MODEL_SIZE = "20m"

MODEL_SIZE_PRESETS: dict[str, ModelSizePreset] = {
    "20m": ModelSizePreset(
        "20m", 20_003_456, 4_096, "float32", "minimum memory (recommended)"
    ),
    "50m": ModelSizePreset("50m", 49_471_872, 4_096, "float32", "lightweight"),
    "80m": ModelSizePreset(
        "80m", 79_467_520, 4_096, "bfloat16", "balanced lightweight"
    ),
    "100m": ModelSizePreset("100m", 99_819_648, 4_096, "bfloat16", "mid-size"),
    "200m": ModelSizePreset("200m", 199_810_048, 4_096, "bfloat16", "large local"),
    "300m": ModelSizePreset(
        "300m", 297_875_456, 4_096, "bfloat16", "high-memory research"
    ),
    # These names and shapes shipped before the numbered family. They are
    # compatibility identities, not aliases for 20m and 300m.
    "tiny": ModelSizePreset(
        "tiny", 29_851_776, 4_096, "float32", "legacy 29.85M", legacy=True
    ),
    "standard": ModelSizePreset(
        "standard",
        320_929_536,
        32_768,
        "bfloat16",
        "legacy 320.93M",
        legacy=True,
    ),
}


def normalize_model_size(value: str, *, allow_auto: bool = True) -> str:
    """Return one canonical profile ID, accepting friendly numeric forms."""

    normalized = value.strip().lower()
    if normalized.isdigit():
        normalized = f"{normalized}m"
    if allow_auto and normalized == "auto":
        return normalized
    if normalized not in MODEL_SIZE_PRESETS:
        choices = ", ".join(PUBLIC_MODEL_SIZE_IDS)
        raise ValueError(f"model size must be one of: {choices}")
    return normalized


def default_parameter_dtype(model_size: str) -> ParameterDType:
    """Return the audited storage dtype for a portable profile."""

    try:
        return MODEL_SIZE_PRESETS[model_size].parameter_dtype
    except KeyError as error:
        raise ValueError("portable model size is invalid") from error


__all__ = [
    "DEFAULT_NEW_MODEL_SIZE",
    "LEGACY_MODEL_SIZE_IDS",
    "MODEL_SIZE_CHOICES",
    "MODEL_SIZE_IDS",
    "MODEL_SIZE_PRESETS",
    "ModelSizePreset",
    "PUBLIC_MODEL_SIZE_IDS",
    "default_parameter_dtype",
    "normalize_model_size",
]
