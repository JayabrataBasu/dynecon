"""The quasi-posterior's chain rule (bounds, Jacobian) and its link to smm_exact standard errors."""

import numpy as np
import pytest

from dynecon import models

pytestmark = pytest.mark.jax
pytest.importorskip("jax.numpy")

from dynecon.estimation import smm_exact as mod  # noqa: E402


def test_spread_matches_the_sandwich_standard_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    model = models.load("consumption_savings", n_a=20, n_z=3).compile()
    names = ["beta", "sigma_e"]
    truth = np.array([0.95, 0.12])

    def moments(theta: np.ndarray):  # smooth, identified, with an exact Jacobian
        b, s = theta
        m = np.array([2 * b, s + b, 3 * s * s])
        G = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 6 * s]])
        return m, G

    monkeypatch.setattr(mod, "_evaluate", lambda _m, _n, _f, _e, theta: moments(theta))
    n_obs = 2000
    omega = np.diag([0.02, 0.03, 0.01])
    m_hat, G = moments(truth)
    expected_se = np.sqrt(np.diag(np.linalg.inv(G.T @ np.linalg.inv(omega) @ G)) / n_obs)
    start = model.with_params(beta=0.94, sigma_e=0.13)
    res = mod.quasi_posterior(
        start, names, m_hat, lambda *a: None, omega, n_obs=n_obs, n_draws=1500, n_chains=2,
        n_warmup=300, rng=3,
    )  # fmt: skip
    assert res.divergences == 0 and np.all(res.rhat < 1.05)
    np.testing.assert_allclose(res.mean, truth, atol=3 * expected_se.max())
    np.testing.assert_allclose(res.std, expected_se, rtol=0.2)
    lo, hi = zip(*model.parameters.bounds(names), strict=True)
    assert np.all(res.pooled > lo) and np.all(res.pooled < hi)
