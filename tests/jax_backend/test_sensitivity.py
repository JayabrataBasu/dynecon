"""Exact stationary-moment derivatives vs finite differences of the NumPy pipeline."""

import numpy as np
import pytest

import dynecon as ek
from dynecon import models
from dynecon.core.errors import ValidationError
from dynecon.simulation import stationary_distribution_young

pytestmark = pytest.mark.jax
pytest.importorskip("jax.numpy")
from dynecon.estimation.sensitivity import stationary_moments  # noqa: E402

NAMES = ["beta", "sigma", "r", "rho", "sigma_e"]
BASE = dict(beta=0.95, sigma=2.0, r=0.02, rho=0.9, sigma_e=0.1)


def moment_fn(mu, c, a_next, a_grid, z, p):
    mean_a = (mu * a_next).sum()
    return [mean_a, (mu * c).sum(), (mu * a_next**2).sum() - mean_a**2]


def numpy_moments(**params):
    model = models.load("consumption_savings", n_a=40, n_z=3, **params).compile()
    sol = ek.solve(model, method="egm", tol=1e-14, max_iter=100_000)
    mu = stationary_distribution_young(sol, tol=1e-15, max_iter=200_000)
    a_next, c = sol.policy["assets_next"], sol.policy["consumption"]
    mean_a = (mu * a_next).sum()
    return np.array([mean_a, (mu * c).sum(), (mu * a_next**2).sum() - mean_a**2])


def test_exact_jacobian_matches_finite_differences() -> None:
    model = models.load("consumption_savings", n_a=40, n_z=3, **BASE).compile()
    m, G = stationary_moments(model, NAMES, moment_fn)
    np.testing.assert_allclose(m, numpy_moments(**BASE), rtol=1e-9)
    for j, name in enumerate(NAMES):
        h = 1e-5 * max(abs(BASE[name]), 0.1)
        up, dn = dict(BASE, **{name: BASE[name] + h}), dict(BASE, **{name: BASE[name] - h})
        fd = (numpy_moments(**up) - numpy_moments(**dn)) / (2 * h)
        np.testing.assert_allclose(G[:, j], fd, rtol=2e-4, atol=1e-6, err_msg=name)


def test_bad_inputs_raise() -> None:
    model = models.load("consumption_savings", n_a=40, n_z=3).compile()
    with pytest.raises(ValidationError):
        stationary_moments(model, ["not_a_parameter"], moment_fn)
    with pytest.raises(ValidationError):
        stationary_moments(model, [], moment_fn)
    big = models.load("consumption_savings", n_a=1000, n_z=7).compile()
    with pytest.raises(ValidationError):
        stationary_moments(big, ["beta"], moment_fn)
