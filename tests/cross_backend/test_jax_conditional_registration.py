"""Tests for conditional JAX backend registrations."""

from __future__ import annotations

import numpy as np
import pytest

from econkit.config import config
from econkit.numerics.backends.discretization_jax import JAX_AVAILABLE as DISC_JAX_AVAILABLE
from econkit.numerics.backends.jax_backend import JAX_AVAILABLE as INTERP_JAX_AVAILABLE
from econkit.numerics.discretization import rouwenhorst, tauchen
from econkit.numerics.interpolation import linear_interp_1d


def test_interpolation_jax_registration_respects_availability() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 4.0])
    x_new = np.array([0.5, 1.5])
    if INTERP_JAX_AVAILABLE:
        out = linear_interp_1d(x, y, x_new, backend="jax")
        assert out.shape == (2,)
    else:
        original = config.strict_backend
        config.strict_backend = True
        try:
            with pytest.raises(ValueError):
                linear_interp_1d(x, y, x_new, backend="jax")
        finally:
            config.strict_backend = original


def test_discretization_jax_registration_respects_availability() -> None:
    if DISC_JAX_AVAILABLE:
        grid, trans = tauchen(0.9, 0.2, 5, backend="jax")
        assert grid.shape == (5,)
        assert trans.shape == (5, 5)
        grid2, trans2 = rouwenhorst(0.9, 0.2, 5, backend="jax")
        assert grid2.shape == (5,)
        assert trans2.shape == (5, 5)
    else:
        original = config.strict_backend
        config.strict_backend = True
        try:
            with pytest.raises(ValueError):
                tauchen(0.9, 0.2, 5, backend="jax")
            with pytest.raises(ValueError):
                rouwenhorst(0.9, 0.2, 5, backend="jax")
        finally:
            config.strict_backend = original
