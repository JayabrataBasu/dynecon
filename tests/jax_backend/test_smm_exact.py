"""Gradient-based SMM recovers known parameters from model-generated moments."""

import numpy as np
import pytest

from dynecon import models

pytestmark = pytest.mark.jax
pytest.importorskip("jax.numpy")
from test_sensitivity import moment_fn  # noqa: E402

from dynecon.estimation.sensitivity import stationary_moments  # noqa: E402
from dynecon.estimation.smm_exact import smm_exact  # noqa: E402


def test_recovers_parameters_partial_equilibrium() -> None:
    truth = {"beta": 0.95, "sigma_e": 0.12}
    names = list(truth)
    start = models.load("consumption_savings", n_a=40, n_z=3, beta=0.93, sigma_e=0.08).compile()
    true_model = start.with_params(**truth)
    m_hat, _ = stationary_moments(true_model, names, moment_fn)
    omega = np.diag((0.05 * np.abs(m_hat)) ** 2)  # any positive-definite matrix
    res = smm_exact(start, names, m_hat, moment_fn, omega, n_obs=5000)
    assert res.converged
    np.testing.assert_allclose(res.estimates, [truth[n] for n in names], rtol=1e-5)
    assert res.loss < 1e-12 and res.df == 1 and np.all(res.std_errors > 0)


def test_recovers_builtin_aiyagari_in_general_equilibrium() -> None:
    from dynecon.estimation.sensitivity import equilibrium_moments
    from dynecon.models import aiyagari

    def moments(mu, c, a_next, a_grid, z, p):  # mean consumption is redundant in equilibrium
        mean_a, mean_c = (mu * a_next).sum(), (mu * c).sum()
        return [mean_a, (mu * a_next**2).sum() - mean_a**2, (mu * c**2).sum() - mean_c**2]

    grid = {"n_a": 40, "n_z": 3, "a_max": 80.0}
    truth = {"beta": 0.96, "sigma_e": 0.2}
    names = list(truth)
    eq = aiyagari.equilibrium(**grid, **truth)
    m_hat, _, _ = equilibrium_moments(eq.model, names, "r", aiyagari.excess_demand, moments)
    omega = np.diag((0.05 * np.abs(m_hat)) ** 2)
    market = (
        "r",
        lambda m: (-m.p.delta + 1e-4, 1.0 / m.p.beta - 1.0 - 1e-4),
        aiyagari.excess_demand,
    )
    start = models.load("aiyagari", **grid, beta=0.95, sigma_e=0.15).compile()
    res = smm_exact(
        start, names, m_hat, moments, omega, n_obs=10_000, equilibrium=market, weights="identity"
    )
    assert res.converged
    np.testing.assert_allclose(res.estimates, [truth[n] for n in names], rtol=1e-4)


def test_extra_starts_keep_the_best_fit() -> None:
    truth = {"beta": 0.95, "sigma_e": 0.12}
    names = list(truth)
    start = models.load("consumption_savings", n_a=40, n_z=3, beta=0.93, sigma_e=0.08).compile()
    m_hat, _ = stationary_moments(start.with_params(**truth), names, moment_fn)
    omega = np.diag((0.05 * np.abs(m_hat)) ** 2)
    one = smm_exact(start, names, m_hat, moment_fn, omega, n_obs=5000, weights="identity")
    many = smm_exact(
        start, names, m_hat, moment_fn, omega, n_obs=5000, weights="identity",
        starts=[[0.9, 0.2], [0.5, 0.01]],  # the second start is far outside the useful region
    )  # fmt: skip
    assert many.loss <= one.loss + 1e-15
    np.testing.assert_allclose(many.estimates, [truth[n] for n in names], rtol=1e-5)
