"""Cross-backend parity baseline tests.

Economics rationale:
Backend acceleration should preserve economic meaning. Numerical parity tests
protect against backend-specific drifts in continuation-value interpolation.
"""

import numpy as np
import pytest

from econkit.numerics.interpolation import linear_interp_1d


def test_linear_interp_numpy_numba_parity() -> None:
    pytest.importorskip("numba")

    xp = np.linspace(0.0, 10.0, 11, dtype=np.float64)
    fp = np.sin(xp)
    x = np.linspace(0.25, 9.75, 31, dtype=np.float64)

    y_numpy = linear_interp_1d(x, xp, fp, backend="numpy")
    y_numba = linear_interp_1d(x, xp, fp, backend="numba")

    np.testing.assert_allclose(y_numpy, y_numba, rtol=1e-10, atol=1e-10)
