import numpy as np

from econkit.numerics.interpolation import linear_interp_1d


def test_linear_interp_exact_at_grid_nodes() -> None:
    xp = np.linspace(0.0, 1.0, 6)
    fp = np.sin(xp)
    out = linear_interp_1d(xp, xp, fp, backend="numpy")
    np.testing.assert_allclose(out, fp, rtol=0.0, atol=1e-12)


def test_linear_interp_between_nodes() -> None:
    xp = np.array([0.0, 1.0])
    fp = np.array([0.0, 2.0])
    x = np.array([0.25, 0.5, 0.75])
    out = linear_interp_1d(x, xp, fp, backend="numpy")
    np.testing.assert_allclose(out, np.array([0.5, 1.0, 1.5]))
