"""Dynecon vs QuantEcon on identical problems (skipped if quantecon is not installed)."""

import warnings

import numpy as np
import pytest

from dynecon.numerics import rouwenhorst, stationary_distribution, tauchen

pytestmark = pytest.mark.crosscheck
qe_markov = pytest.importorskip("quantecon.markov")

CASES = [(rho, sigma, n) for rho in (0.0, 0.5, 0.9) for sigma in (0.05, 0.2) for n in (3, 7, 15)]


@pytest.mark.parametrize(
    ("ours", "theirs"), [("tauchen", "tauchen"), ("rouwenhorst", "rouwenhorst")]
)
@pytest.mark.parametrize(("rho", "sigma", "n"), CASES)
def test_discretization_matches_quantecon(ours, theirs, rho, sigma, n) -> None:
    fn = {"tauchen": tauchen, "rouwenhorst": rouwenhorst}[ours]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quantecon warns about its rouwenhorst API change
        chain = getattr(qe_markov, theirs)(n, rho, sigma)
    grid, P = fn(rho, sigma, n)
    np.testing.assert_allclose(grid, chain.state_values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(P, chain.P, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        stationary_distribution(P), chain.stationary_distributions[0], rtol=0, atol=1e-8
    )
