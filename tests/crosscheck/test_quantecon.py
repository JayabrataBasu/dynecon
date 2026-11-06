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


def test_markov_chain_structure_matches_quantecon() -> None:
    from quantecon.markov import MarkovChain as QMC

    from dynecon.numerics import MarkovChain

    rng = np.random.default_rng(3)
    for _ in range(40):
        n = int(rng.integers(2, 9))
        P = rng.random((n, n)) * (rng.random((n, n)) < 0.35)
        P[np.arange(n), rng.integers(0, n, n)] += 0.5  # at least one edge per row
        P /= P.sum(axis=1, keepdims=True)
        ours, theirs = MarkovChain(P), QMC(P)
        key = lambda cs: sorted(tuple(sorted(map(int, c))) for c in cs)  # noqa: E731
        assert key(ours.communication_classes) == key(theirs.communication_classes)
        assert key(ours.recurrent_classes) == key(theirs.recurrent_classes)
        assert ours.is_irreducible == theirs.is_irreducible
        assert ours.is_aperiodic == theirs.is_aperiodic
        a = np.array(sorted(map(tuple, np.round(ours.stationary_distributions, 10))))
        b = np.array(sorted(map(tuple, np.round(theirs.stationary_distributions, 10))))
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_lq_matches_quantecon() -> None:
    from quantecon import LQ as QLQ

    from dynecon.solvers.lq import LQ

    rng = np.random.default_rng(11)
    for beta in (1.0, 0.95):
        for _ in range(10):
            n, k = int(rng.integers(1, 4)), int(rng.integers(1, 3))
            A = rng.normal(size=(n, n)) * 0.5
            B = rng.normal(size=(n, k))
            M = rng.normal(size=(n, n))
            R = M @ M.T + np.eye(n)
            Q = np.eye(k) * 2.0
            N = 0.05 * rng.normal(size=(k, n))
            C = None if beta == 1.0 else 0.2 * rng.normal(size=(n, 2))
            ours, theirs = LQ(Q, R, A, B, C=C, N=N, beta=beta), QLQ(Q, R, A, B, C=C, N=N, beta=beta)
            for a, b in zip(ours.stationary_values(), theirs.stationary_values(), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-9)
            fin, ref = (
                LQ(Q, R, A, B, C=C, N=N, beta=beta, T=12, Rf=2 * np.eye(n)),
                QLQ(Q, R, A, B, C=C, N=N, beta=beta, T=12, Rf=2 * np.eye(n)),
            )
            fin.update_values()
            for _ in range(12):  # QuantEcon steps the recursion once per call
                ref.update_values()
            np.testing.assert_allclose(fin.P, ref.P, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(fin.F, ref.F, rtol=1e-8, atol=1e-9)
            assert abs(fin.d - ref.d) < 1e-8


def test_normal_and_lognormal_quadrature_match_quantecon() -> None:
    from quantecon.quad import qnwlogn, qnwnorm

    from dynecon.numerics import lognormal_nodes, mvnormal_nodes

    mean, cov = np.array([0.3, -0.2]), np.array([[0.5, 0.2], [0.2, 0.8]])
    tests = (lambda x: x[:, 0] * x[:, 1] + x[:, 0] ** 2, lambda x: np.sin(x[:, 0]) + x[:, 1] ** 3)
    for ours, theirs in ((mvnormal_nodes, qnwnorm), (lognormal_nodes, qnwlogn)):
        xo, wo = ours([6, 5], mean, cov)
        xq, wq = theirs([6, 5], mean, cov)
        for f in tests:  # node order differs, so compare integrals of test functions
            assert abs(wo @ f(xo) - wq @ f(xq)) < 1e-9
