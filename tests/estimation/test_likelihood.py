"""MLE, Kalman filter and MCMC against closed forms."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import econkit as ek
from econkit import models
from econkit.core.errors import ValidationError
from econkit.estimation import (
    bayes_estimate,
    kalman_filter,
    loglik_state_space,
    metropolis,
    mle,
)
from econkit.estimation.bayes import ess, split_rhat
from econkit.estimation.priors import Beta
from econkit.model import Model, Param


def test_mle_ar1_equals_ols_and_known_standard_error() -> None:
    rng = np.random.default_rng(0)
    T, rho_true = 4000, 0.7
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = rho_true * y[t - 1] + rng.standard_normal()
    x, z = y[:-1], y[1:]

    def loglik(th: np.ndarray) -> float:
        rho, sig = th
        r = z - rho * x
        return float(-0.5 * np.sum(np.log(2 * np.pi * sig**2) + (r / sig) ** 2))

    res = mle(loglik, [0.1, 2.0], bounds=[(-0.99, 0.99), (1e-3, 10.0)])
    ols = float(x @ z / (x @ x))
    assert res.converged and abs(res.estimates[0] - ols) < 1e-5
    assert abs(res.std_errors[0] / np.sqrt((1 - ols**2) / T) - 1) < 0.05
    assert "MLE estimates" in res.summary()
    with pytest.raises(ValidationError):
        mle(loglik, [[0.1, 1.0]])


def local_level(q: float, n: int = 200) -> tuple[np.ndarray, ...]:
    Z, T = np.array([[1.0]]), np.array([[1.0]])
    return Z, np.array([[1.0]]), T, np.array([[q]])


def test_local_level_gain_converges_to_steady_state() -> None:
    q = 0.3
    Z, H, T, Q = local_level(q)
    y = np.random.default_rng(1).standard_normal(300)
    res = kalman_filter(y, Z, H, T, Q, np.zeros(1), np.array([[10.0]]))
    pbar = (q + np.sqrt(q**2 + 4 * q)) / 2
    assert abs(res.predicted_cov[-1, 0, 0] - pbar) < 1e-8
    assert abs(res.gains[-1, 0, 0] - pbar / (pbar + 1)) < 1e-8
    assert res.n_obs_used == 300


def test_likelihood_equals_brute_force_multivariate_normal() -> None:
    q, h, P0, a0, n = 0.3, 0.7, 2.0, 0.5, 15
    y = np.random.default_rng(2).standard_normal(n)
    res = kalman_filter(
        y,
        np.array([[1.0]]),
        np.array([[h]]),
        np.array([[1.0]]),
        np.array([[q]]),
        np.array([a0]),
        np.array([[P0]]),
    )
    idx = np.arange(n)
    cov = P0 + q * np.minimum.outer(idx, idx) + h * np.eye(n)
    ref = multivariate_normal(mean=np.full(n, a0), cov=cov).logpdf(y)
    assert abs(res.loglik - ref) < 1e-8


def test_missing_observations_are_skipped() -> None:
    n, q, P0 = 10, 0.4, 1.5
    Z = np.array([[1.0], [1.0]])
    H = np.diag([0.5, 0.8])
    rng = np.random.default_rng(3)
    y = rng.standard_normal((n, 2))
    y[2, 1] = np.nan
    y[5, :] = np.nan
    y[7, 0] = np.nan
    res = kalman_filter(y, Z, H, np.array([[1.0]]), np.array([[q]]), np.zeros(1), np.array([[P0]]))
    t_idx, i_idx = np.where(~np.isnan(y))
    cov = (
        P0
        + q * (np.minimum.outer(t_idx, t_idx))
        + np.diag(np.diag(H)[i_idx])
        * (t_idx[:, None] == t_idx[None, :])
        * (i_idx[:, None] == i_idx[None, :])
    )
    ref = multivariate_normal(mean=np.zeros(t_idx.size), cov=cov).logpdf(y[t_idx, i_idx])
    assert abs(res.loglik - ref) < 1e-8
    assert res.n_obs_used == t_idx.size
    with pytest.raises(ValidationError):
        kalman_filter(y, Z[:1], H, np.eye(1), np.eye(1), np.zeros(1), np.eye(1))


def test_dsge_likelihood_peaks_near_true_persistence() -> None:
    sol = ek.solve(models.load("brock_mirman", problem="dsge").compile())
    sim = ek.simulate(sol, n_periods=400, rng=5, store=["k", "c"])
    data = np.column_stack([sim.paths["k"][:, 0], sim.paths["c"][:, 0]])
    grid = [0.5, 0.7, 0.9, 0.97]
    ll = [
        loglik_state_space(
            ek.solve(models.load("brock_mirman", problem="dsge", rho=r).compile()),
            ["k", "c"],
            data,
            measurement_error=0.002,
        )
        for r in grid
    ]
    assert grid[int(np.argmax(ll))] in (0.9, 0.97)
    with pytest.raises(ValidationError):
        loglik_state_space(sol, ["nope"], data)


def test_metropolis_recovers_bivariate_gaussian() -> None:
    mu = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    prec = np.linalg.inv(cov)

    def logpost(x: np.ndarray) -> float:
        d = x - mu
        return float(-0.5 * d @ prec @ d)

    res = metropolis(logpost, [0.0, 0.0], 5000, n_chains=4, burn_in=1000, rng=0, names=["a", "b"])
    assert res.draws.shape == (4, 5000, 2)
    assert np.all(res.rhat < 1.05)
    assert 0.05 < res.accept_rate < 0.9
    se = np.sqrt(np.diag(cov)) / np.sqrt(res.ess)
    assert np.all(np.abs(res.mean - mu) < 4 * se)
    emp = np.cov(res.pooled.T)
    assert np.all(np.abs(emp - cov) / np.abs(cov) < 0.10)
    assert "Posterior summary" in res.summary()
    with pytest.raises(ValidationError):
        metropolis(logpost, [0.0, 0.0], 5)
    with pytest.raises(ValidationError):
        metropolis(lambda x: float("-inf"), [0.0], 50, n_chains=1)


def test_bayes_estimate_beta_prior_via_logit_transform() -> None:
    m = Model("prior_only")
    m.params(p=Param(0.5, bounds=(0.0, 1.0), free=True, prior=Beta(2.0, 5.0)), beta=Param(0.9))
    m.set_problem(
        "finite_dp",
        reward=lambda q: np.ones((1, 1)),
        transition=lambda q: np.ones((1, 1, 1)),
        discount="beta",
    )
    res = bayes_estimate(
        m.compile(), ["p"], lambda model: 0.0, 6000, n_chains=4, burn_in=1000, rng=1
    )
    assert abs(res.mean[0] - 2 / 7) < 0.01
    assert 0.0 < res.draws.min() and res.draws.max() < 1.0
    assert res.rhat[0] < 1.05


def test_diagnostics_helpers() -> None:
    rng = np.random.default_rng(0)
    iid = rng.standard_normal((4, 2000))
    assert abs(split_rhat(iid) - 1.0) < 0.02
    assert ess(iid) > 0.7 * iid.size
    ar = np.zeros((4, 2000))
    for t in range(1, 2000):
        ar[:, t] = 0.95 * ar[:, t - 1] + rng.standard_normal(4)
    assert ess(ar) < 0.1 * ar.size
    assert split_rhat(np.zeros((2, 10))) == 1.0
