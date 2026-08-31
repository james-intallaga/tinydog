from __future__ import annotations

from pathlib import Path

import pytest

from mini_arithmetic.model_sizes import (
    MODEL_SIZE_PRESETS,
    PUBLIC_MODEL_SIZE_IDS,
    default_parameter_dtype,
    normalize_model_size,
)
from mini_arithmetic.portable_bundle import load_hybrid_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("identifier", PUBLIC_MODEL_SIZE_IDS)
def test_numbered_model_size_configs_match_their_public_identity(
    identifier: str,
) -> None:
    preset = MODEL_SIZE_PRESETS[identifier]
    config = load_hybrid_config(PROJECT_ROOT / "configs" / preset.config_filename)

    assert config.expected_parameter_count() == preset.parameters
    assert config.context_length == preset.context_length
    assert config.vocab_size == 32_768
    assert config.head_dim == 64
    assert config.num_attention_heads == config.model_dim // config.head_dim
    assert config.num_attention_heads == 2 * config.num_key_value_heads
    assert default_parameter_dtype(identifier) == preset.parameter_dtype


def test_numbered_model_sizes_increase_strictly() -> None:
    counts = [MODEL_SIZE_PRESETS[value].parameters for value in PUBLIC_MODEL_SIZE_IDS]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)


@pytest.mark.parametrize(
    ("provided", "expected"),
    (("20", "20m"), ("50M", "50m"), (" 100m ", "100m"), ("auto", "auto")),
)
def test_friendly_model_size_forms_are_canonicalized(
    provided: str,
    expected: str,
) -> None:
    assert normalize_model_size(provided) == expected


def test_legacy_model_identities_remain_distinct() -> None:
    assert MODEL_SIZE_PRESETS["tiny"].parameters == 29_851_776
    assert MODEL_SIZE_PRESETS["standard"].parameters == 320_929_536
    assert MODEL_SIZE_PRESETS["tiny"].legacy is True
    assert MODEL_SIZE_PRESETS["standard"].legacy is True
    assert MODEL_SIZE_PRESETS["tiny"].parameters != MODEL_SIZE_PRESETS["20m"].parameters
    assert (
        MODEL_SIZE_PRESETS["standard"].parameters
        != MODEL_SIZE_PRESETS["300m"].parameters
    )
