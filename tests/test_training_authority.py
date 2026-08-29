from __future__ import annotations

import pytest

from mini_arithmetic.apprenticeship import ApprenticeshipStore
from mini_arithmetic.coordinator import CommunityTrainingBoundary


def test_community_authority_never_falls_back_to_full_model_training(tmp_path) -> None:
    store = ApprenticeshipStore(tmp_path / "apprenticeship.db")
    boundary = CommunityTrainingBoundary(store)

    assert boundary.consider_training() is False
    status = boundary.status()
    assert status["state"] == "blocked"
    assert status["authority"] == "community"
    assert status["parameter_scope"] == "floaty-only"
    assert status["foundation_trainable"] is False
    assert status["accepts_community_updates"] is False

    with pytest.raises(ValueError, match="cannot use the legacy full-model runtime"):
        boundary.trigger(force=True)
    with pytest.raises(ValueError, match="no authority"):
        boundary.rollback()
    boundary.shutdown()
