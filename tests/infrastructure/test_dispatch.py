"""Dispatcher infrastructure tests: routing is transparent to model authors."""

import numpy as np
import pytest

from dynecon.config import config
from dynecon.numerics._dispatch import resolve_backend
from dynecon.numerics._registry import list_registered, register
from dynecon.numerics.interpolation import linear_interp_1d


def test_dispatch_resolves_numpy() -> None:
    backend, _ = resolve_backend("linear_interp_1d", backend="numpy")
    assert backend == "numpy"


def test_global_backend_numpy_routes() -> None:
    config.set_backend("numpy")
    out = linear_interp_1d(np.array([0.1, 0.2]), np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    np.testing.assert_allclose(out, [0.1, 0.2])


def test_duplicate_registration_rejected() -> None:
    with pytest.raises(ValueError):
        register("linear_interp_1d", "numpy", lambda *a: None)


def test_every_registered_numba_op_has_numpy_reference() -> None:
    reg = list_registered()
    for op, backend in reg:
        if backend != "numpy":
            assert (op, "numpy") in reg
