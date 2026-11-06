"""Degenerate estimation inputs give a clear ValidationError."""

import numpy as np
import pytest

from dynecon.core.errors import ValidationError
from dynecon.estimation import hmc, kalman_filter, metropolis


def test_non_finite_state_space_is_rejected() -> None:
    one, y = np.eye(1), np.ones((5, 1))
    with pytest.raises(ValidationError):
        kalman_filter(y, one, one, np.array([[np.nan]]), one, np.zeros(1), one)
    with pytest.raises(ValidationError):
        kalman_filter(y, one, one, one, one, np.array([np.inf]), one)


def test_empty_starting_point_is_rejected() -> None:
    with pytest.raises(ValidationError):
        metropolis(lambda x: 0.0, np.zeros(0), 100)
    with pytest.raises(ValidationError):
        hmc(lambda x: (0.0, np.zeros(0)), np.zeros(0), 100)


def test_bootstrap_rejects_non_finite_data() -> None:
    from dynecon.experiment import bootstrap_ci

    with pytest.raises(ValidationError):
        bootstrap_ci(np.array([1.0, np.nan, 3.0]), np.mean, n_boot=10)
