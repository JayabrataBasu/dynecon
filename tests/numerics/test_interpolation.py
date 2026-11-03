"""Interpolation: behaviour, property tests and NumPy/Numba parity."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from econkit.core.errors import ValidationError
from econkit.numerics.interpolation import linear_interp_1d

BACKENDS = ["numpy", pytest.param("numba", marks=pytest.mark.numba)]


def test_exact_at_nodes_and_midpoints() -> None:
    xp = np.linspace(0.0, 1.0, 6)
    fp = np.sin(xp)
    np.testing.assert_allclose(linear_interp_1d(xp, xp, fp), fp, atol=1e-14)
    out = linear_interp_1d([0.25, 0.5, 0.75], [0.0, 1.0], [0.0, 2.0])
    np.testing.assert_allclose(out, [0.5, 1.0, 1.5])


def test_extrapolation_modes() -> None:
    xp, fp = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0])
    np.testing.assert_allclose(linear_interp_1d([-1.0, 3.0], xp, fp), [0.0, 4.0])
    np.testing.assert_allclose(
        linear_interp_1d([-1.0, 3.0], xp, fp, extrapolate="linear"), [-1.0, 7.0]
    )


def test_shape_preserved_and_validation() -> None:
    xp = np.linspace(0, 1, 5)
    assert linear_interp_1d(np.zeros((3, 4)), xp, xp).shape == (3, 4)
    assert linear_interp_1d(0.3, xp, xp).shape == ()
    with pytest.raises(ValidationError):
        linear_interp_1d(0.5, [0.0, 0.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValidationError):
        linear_interp_1d(0.5, [0.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValidationError):
        linear_interp_1d(0.5, [0.0, 1.0], [0.0, 1.0], extrapolate="cubic")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["clamp", "linear"])
def test_backend_parity(backend: str, mode: str) -> None:
    if backend == "numba":
        pytest.importorskip("numba")
    rng = np.random.default_rng(0)
    xp = np.sort(rng.uniform(-3, 3, 50))
    xp = xp + np.arange(50) * 1e-3
    fp = rng.standard_normal(50)
    x = rng.uniform(-4, 4, 500)
    ref = linear_interp_1d(x, xp, fp, extrapolate=mode, backend="numpy")
    out = linear_interp_1d(x, xp, fp, extrapolate=mode, backend=backend)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-13)


_nodes = st.lists(st.floats(-100, 100), min_size=2, max_size=30, unique=True).map(sorted)


@settings(deadline=None, max_examples=100, derandomize=True)
@given(nodes=_nodes, data=st.data())
def test_properties(nodes: list[float], data: st.DataObject) -> None:
    xp = np.array(nodes)
    if np.min(np.diff(xp)) < 1e-6:
        return
    fp = np.array(data.draw(st.lists(st.floats(-10, 10), min_size=len(xp), max_size=len(xp))))
    np.testing.assert_allclose(linear_interp_1d(xp, xp, fp), fp, atol=1e-9)
    q = np.array(
        data.draw(st.lists(st.floats(float(xp[0]), float(xp[-1])), min_size=1, max_size=10))
    )
    out = linear_interp_1d(q, xp, fp)
    assert np.all(out <= fp.max() + 1e-9) and np.all(out >= fp.min() - 1e-9)
    perm = np.random.default_rng(0).permutation(q.shape[0])
    np.testing.assert_allclose(linear_interp_1d(q[perm], xp, fp), out[perm], atol=1e-12)
    inc = np.sort(fp)
    assert np.all(np.diff(linear_interp_1d(np.sort(q), xp, inc)) >= -1e-9)
    a, b = 1.7, -0.3
    far = np.array([xp[0] - 5.0, xp[-1] + 5.0])
    np.testing.assert_allclose(
        linear_interp_1d(far, xp, a * xp + b, extrapolate="linear"), a * far + b, atol=1e-6
    )


@pytest.mark.parametrize("backend", BACKENDS + [pytest.param("jax", marks=pytest.mark.jax)])
@pytest.mark.parametrize("mode", ["clamp", "linear"])
def test_columns_match_rowwise_1d(backend: str, mode: str) -> None:
    from econkit.numerics.interpolation import linear_interp_columns

    if backend == "numba":
        pytest.importorskip("numba")
    if backend == "jax":
        pytest.importorskip("jax")
    rng = np.random.default_rng(3)
    xk = np.sort(rng.uniform(-2, 2, (4, 30)), axis=1) + np.arange(30) * 1e-3
    fk = rng.standard_normal((4, 30))
    xq = rng.uniform(-3, 3, (4, 50))
    out = linear_interp_columns(xq, xk, fk, extrapolate=mode, backend=backend)
    for c in range(4):
        np.testing.assert_allclose(
            out[c],
            linear_interp_1d(xq[c], xk[c], fk[c], extrapolate=mode, backend="numpy"),
            atol=1e-12,
        )
    with pytest.raises(ValidationError):
        linear_interp_columns(xq, xk[:3], fk[:3])
    with pytest.raises(ValidationError):
        linear_interp_columns(xq, xk, fk, extrapolate="cubic")
