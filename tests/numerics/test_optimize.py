"""Bounded optimization tests.

Economics rationale:
Many policy problems reduce to bounded scalar maximization, for example choosing
consumption/savings under constraints. These tests ensure robust and accurate
argmax recovery for smooth objectives.
"""

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
