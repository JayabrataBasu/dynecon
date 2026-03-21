"""Discretization tests.

Economics rationale:
Discrete Markov approximations for AR(1) shocks must preserve probability
structure because they determine transition risk in dynamic models.
"""

import numpy as np

from econkit.numerics.discretization import rouwenhorst, tauchen


def test_tauchen_shapes_and_stochastic_rows() -> None:
    grid, transition = tauchen(rho=0.9, sigma_e=0.1, n_states=7, backend="numpy")
    assert grid.shape == (7,)
    assert transition.shape == (7, 7)
    np.testing.assert_allclose(transition.sum(axis=1), np.ones(7), atol=1e-12)


def test_rouwenhorst_shapes_and_stochastic_rows() -> None:
    grid, transition = rouwenhorst(rho=0.95, sigma_e=0.05, n_states=9, backend="numpy")
    assert grid.shape == (9,)
    assert transition.shape == (9, 9)
    np.testing.assert_allclose(transition.sum(axis=1), np.ones(9), atol=1e-12)
