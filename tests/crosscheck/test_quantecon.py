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


@pytest.mark.parametrize(("n_a", "n_y"), [(1, 1), (2, 1), (3, 2)])
def test_kalman_matches_quantecon(n_a, n_y) -> None:
    from quantecon import Kalman, LinearStateSpace
    from scipy.stats import multivariate_normal

    from dynecon.estimation import kalman_filter

    rng = np.random.default_rng(n_a * 10 + n_y)
    A = rng.normal(size=(n_a, n_a))
    A *= 0.9 / np.abs(np.linalg.eigvals(A)).max()
    C, G, Hf = 0.3 * rng.normal(size=(n_a, n_a)), rng.normal(size=(n_y, n_a)), 0.4 * np.eye(n_y)
    Q, R, a0, P0 = C @ C.T, Hf @ Hf.T, np.zeros(n_a), np.eye(n_a)
    x, y = rng.multivariate_normal(a0, P0), []
    for _ in range(60):
        y.append(G @ x + Hf @ rng.normal(size=n_y))
        x = A @ x + C @ rng.normal(size=n_a)
    ours = kalman_filter(np.array(y), G, R, A, Q, a0, P0)
    kal = Kalman(LinearStateSpace(A, C, G, H=Hf), x_hat=a0.copy(), Sigma=P0.copy())
    filtered, loglik = [], 0.0
    for yt in y:
        prior = np.asarray(kal.x_hat).reshape(n_a)
        F = G @ kal.Sigma @ G.T + R
        loglik += multivariate_normal(mean=np.zeros(n_y), cov=F).logpdf(yt - G @ prior)
        kal.prior_to_filtered(yt)
        filtered.append(np.asarray(kal.x_hat).reshape(n_a).copy())
        kal.filtered_to_forecast()
    np.testing.assert_allclose(ours.filtered_state, np.array(filtered), rtol=0, atol=1e-10)
    assert ours.loglik == pytest.approx(loglik, abs=1e-9)


@pytest.mark.parametrize(
    ("sigma", "r", "beta", "a_min"), [(2.0, 0.03, 0.96, 0.0), (3.0, 0.03, 0.96, -1.0)]
)
def test_vfi_matches_quantecon_discrete_dp(sigma, r, beta, a_min) -> None:
    from quantecon.markov import DiscreteDP

    import dynecon as ek
    from dynecon import models
    from dynecon.simulation import stationary_distribution_young

    n_a, n_z = 40, 3
    model = models.load(
        "consumption_savings", n_a=n_a, n_z=n_z, sigma=sigma, r=r, beta=beta, a_min=a_min
    ).compile()
    sol = ek.solve(model, method="vfi", tol=1e-12, max_iter=20000)
    a, income = model.grids["assets"], model.shock_process("income")
    reward = np.full((n_a * n_z, n_a), -np.inf)
    trans = np.zeros((n_a * n_z, n_a, n_a * n_z))
    for i in range(n_a):
        for j in range(n_z):
            cons = (1 + r) * a[i] + income.values[j] - a
            ok = cons > 0
            reward[i * n_z + j, ok] = cons[ok] ** (1 - sigma) / (1 - sigma)
            for k in range(n_a):
                trans[i * n_z + j, k, k * n_z : (k + 1) * n_z] = income.P[j]
    ddp = DiscreteDP(reward, trans, beta)
    res = ddp.solve(method="policy_iteration", epsilon=1e-12, max_iter=2000)
    np.testing.assert_allclose(sol.value, res.v.reshape(n_a, n_z), rtol=0, atol=1e-9)
    np.testing.assert_array_equal(sol.policy["assets_next_index"], res.sigma.reshape(n_a, n_z))
    mu = stationary_distribution_young(sol).ravel()
    assert np.abs(mu @ ddp.controlled_mc(res.sigma).P - mu).max() < 1e-10
