"""Cross-backend parity baseline tests.

Economics rationale:
Backend acceleration should preserve economic meaning. Numerical parity tests
protect against backend-specific drifts in continuation-value interpolation.
"""

import numpy as np
import pytest

from econkit.config import config
from econkit.numerics.discretization import rouwenhorst, tauchen
from econkit.numerics.integration import (
    expectation_discrete,
    gauss_hermite_normal,
    gauss_laguerre_exp,
)
from econkit.numerics.interpolation import linear_interp_1d
from econkit.numerics.optimize import brent_max


def test_linear_interp_numpy_numba_parity() -> None:
    pytest.importorskip("numba")

    xp = np.linspace(0.0, 10.0, 11, dtype=np.float64)
    fp = np.sin(xp)
    x = np.linspace(0.25, 9.75, 31, dtype=np.float64)

    y_numpy = linear_interp_1d(x, xp, fp, backend="numpy")
    y_numba = linear_interp_1d(x, xp, fp, backend="numba")

    np.testing.assert_allclose(y_numpy, y_numba, rtol=1e-10, atol=1e-10)


def test_tauchen_backend_selection_matches_numpy_reference() -> None:
    pytest.importorskip("numba")

    grid_numpy, trans_numpy = tauchen(0.9, 0.1, 5, backend="numpy")
    grid_numba, trans_numba = tauchen(0.9, 0.1, 5, backend="numba")

    np.testing.assert_allclose(grid_numpy, grid_numba, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(trans_numpy, trans_numba, rtol=0.0, atol=0.0)


def test_gauss_hermite_and_expectation_match_under_backend_fallback() -> None:
    pytest.importorskip("numba")

    nodes_numpy, weights_numpy = gauss_hermite_normal(7, backend="numpy")
    nodes_numba, weights_numba = gauss_hermite_normal(7, backend="numba")

    np.testing.assert_allclose(nodes_numpy, nodes_numba, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights_numpy, weights_numba, rtol=0.0, atol=0.0)

    values = np.array([1.0, 2.0, 5.0], dtype=np.float64)
    probs = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    e_numpy = expectation_discrete(values, probs, backend="numpy")
    e_numba = expectation_discrete(values, probs, backend="numba")
    assert abs(e_numpy - e_numba) < 1e-14


def test_rouwenhorst_backend_selection_matches_numpy_reference() -> None:
    pytest.importorskip("numba")

    grid_numpy, trans_numpy = rouwenhorst(0.95, 0.05, 7, backend="numpy")
    grid_numba, trans_numba = rouwenhorst(0.95, 0.05, 7, backend="numba")

    np.testing.assert_allclose(grid_numpy, grid_numba, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(trans_numpy, trans_numba, rtol=1e-14, atol=1e-14)


def test_gauss_laguerre_backend_selection_matches_numpy_reference() -> None:
    pytest.importorskip("numba")

    nodes_numpy, weights_numpy = gauss_laguerre_exp(10, backend="numpy")
    nodes_numba, weights_numba = gauss_laguerre_exp(10, backend="numba")

    np.testing.assert_allclose(nodes_numpy, nodes_numba, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(weights_numpy, weights_numba, rtol=0.0, atol=0.0)


def test_strict_backend_mode_rejects_unimplemented_backend() -> None:
    original = config.strict_backend
    config.strict_backend = True
    try:
        with pytest.raises(ValueError):
            brent_max(lambda x: -((x - 1.0) ** 2), 0.0, 2.0, backend="jax")
    finally:
        config.strict_backend = original
