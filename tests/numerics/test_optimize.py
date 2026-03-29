"""Bounded optimization tests.

Economics rationale:
Many policy problems reduce to bounded scalar maximization, for example choosing
consumption/savings under constraints. These tests ensure robust and accurate
argmax recovery for smooth objectives.
"""

import pytest

from econkit.numerics.optimize import brent_max, golden_section_max


def _concave_quadratic(x: float) -> float:
    return -((x - 2.0) ** 2) + 5.0


def test_golden_section_max_recovers_quadratic_peak() -> None:
    res = golden_section_max(_concave_quadratic, a=0.0, b=4.0, tol=1e-10, backend="numpy")
    assert abs(res.x - 2.0) < 1e-5
    assert abs(res.fun - 5.0) < 1e-8


def test_brent_max_recovers_quadratic_peak() -> None:
    res = brent_max(_concave_quadratic, a=0.0, b=4.0, tol=1e-10, backend="numpy")
    assert abs(res.x - 2.0) < 1e-5
    assert abs(res.fun - 5.0) < 1e-8


def test_brent_and_golden_are_close() -> None:
    g = golden_section_max(_concave_quadratic, a=0.0, b=4.0, tol=1e-10, backend="numpy")
    b = brent_max(_concave_quadratic, a=0.0, b=4.0, tol=1e-10, backend="numpy")
    assert abs(g.x - b.x) < 1e-4


def test_numba_backend_optimization_parity() -> None:
    pytest.importorskip("numba")

    g_np = golden_section_max(_concave_quadratic, a=0.0, b=4.0, tol=1e-10, backend="numpy")
    g_nb = golden_section_max(_concave_quadratic, a=0.0, b=4.0, tol=1e-10, backend="numba")
    b_np = brent_max(_concave_quadratic, a=0.0, b=4.0, tol=1e-10, backend="numpy")
    b_nb = brent_max(_concave_quadratic, a=0.0, b=4.0, tol=1e-10, backend="numba")

    assert abs(g_np.x - g_nb.x) < 1e-6
    assert abs(g_np.fun - g_nb.fun) < 1e-8
    assert abs(b_np.x - b_nb.x) < 1e-6
    assert abs(b_np.fun - b_nb.fun) < 1e-8
