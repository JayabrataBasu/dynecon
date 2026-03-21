"""Dispatcher infrastructure tests.

Economics rationale:
Backend routing must be transparent to model authors so economic code remains
stable while numerical implementations evolve across performance layers.
"""

import numpy as np

from econkit.config import config
from econkit.numerics._dispatch import resolve_backend
from econkit.numerics.interpolation import linear_interp_1d


def test_dispatch_resolves_numpy() -> None:
    backend, _ = resolve_backend("linear_interp_1d", backend="numpy")
    assert backend == "numpy"


def test_dispatch_falls_back_to_numpy() -> None:
    config.strict_backend = False
    config.set_backend("jax")
    x = np.array([0.1, 0.2, 0.3])
    xp = np.array([0.0, 1.0])
    fp = np.array([0.0, 1.0])
    out = linear_interp_1d(x, xp, fp)
    assert out.shape == x.shape
    config.set_backend("numpy")
