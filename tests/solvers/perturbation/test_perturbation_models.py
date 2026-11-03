"""Perturbation against the analytic solutions of Brock-Mirman, NK and the OLG model."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.linalg import solve_discrete_lyapunov

import dynecon as ek
from dynecon import models
from dynecon.core.errors import ConvergenceError, DeterminacyError, ModelSpecificationError
from dynecon.model import Model, Param
from dynecon.models.nk3 import closed_form_coefficients
from dynecon.models.olg2 import euler_steady_state, iterate_capital, steady_state_capital
from dynecon.solvers.perturbation import irf, simulate_linear, theoretical_moments
from dynecon.solvers.perturbation.steady_state import steady_state


def test_brock_mirman_loglinear_coefficients_and_irf() -> None:
    alpha, rho, sig = 0.36, 0.9, 0.02
    sol = ek.solve(models.load("brock_mirman", problem="dsge").compile())
    assert sol.method == "perturbation" and sol.variables == ("k", "z", "c")
    np.testing.assert_allclose(sol.P, [[alpha, 1.0], [0.0, rho]], atol=1e-6)
    np.testing.assert_allclose(sol.F, [[alpha, 1.0]], atol=1e-6)
    np.testing.assert_allclose(sol.Sigma, [[0.0], [sig]])
    resp = irf(sol, "eps_z", size=1.0 / sig, horizon=6)  # unit z-hat shock
    expect_k = [0.0, 1.0, alpha + rho]
    expect_k.append(alpha * expect_k[2] + rho**2)
    np.testing.assert_allclose(resp["k"][:4], expect_k, atol=1e-6)
    np.testing.assert_allclose(resp["c"][0], 1.0, atol=1e-6)


def test_nk_closed_form_with_zero_output_gap_response() -> None:
    p = dict(beta=0.99, kappa=0.15, sigma=1.0, phi_pi=1.5, rho_v=0.5)
    sol = ek.solve(models.load("nk3", phi_x=0.0, **p).compile())
    a, b = closed_form_coefficients(**p)
    assert a < 0 and b < 0  # contractionary policy shock lowers inflation and output
    pi_i, x_i = sol.variables.index("pi") - 1, sol.variables.index("x") - 1
    np.testing.assert_allclose(sol.F[pi_i, 0], a, atol=1e-6)
    np.testing.assert_allclose(sol.F[x_i, 0], b, atol=1e-6)
    resp = irf(sol, 0, horizon=5)
    assert resp["pi"][0] < 0 and resp["x"][0] < 0


@pytest.mark.parametrize(
    "phi_pi,ok", [(0.5, False), (0.9, False), (1.1, True), (1.5, True), (3.0, True)]
)
def test_taylor_principle_determinacy(phi_pi: float, ok: bool) -> None:
    m = models.load("nk3", phi_x=0.0, phi_pi=phi_pi).compile()
    if ok:
        assert ek.solve(m).converged
    else:
        with pytest.raises(DeterminacyError):
            ek.solve(m)


def test_bullard_mitra_with_output_gap_response() -> None:
    kappa, beta = 0.15, 0.99
    phi_pi, phi_x = 0.9, 3.0
    assert kappa * (phi_pi - 1) + (1 - beta) * phi_x > 0
    assert ek.solve(models.load("nk3", phi_pi=phi_pi, phi_x=phi_x).compile()).converged
    phi_x_low = 0.5
    assert kappa * (phi_pi - 1) + (1 - beta) * phi_x_low < 0
    with pytest.raises(DeterminacyError):
        ek.solve(models.load("nk3", phi_pi=phi_pi, phi_x=phi_x_low).compile())


def test_moments_match_ar1_theory_and_simulation() -> None:
    rho, sig = 0.9, 0.02
    sol = ek.solve(models.load("brock_mirman", problem="dsge").compile())
    mom = theoretical_moments(sol, lags=(1, 2))
    zi = sol.variables.index("z")
    np.testing.assert_allclose(mom.covariance[zi, zi], sig**2 / (1 - rho**2), rtol=1e-10)
    np.testing.assert_allclose(mom.autocorrelation(1)[zi], rho, atol=1e-10)
    np.testing.assert_allclose(mom.autocorrelation(2)[zi], rho**2, atol=1e-10)
    np.testing.assert_allclose(mom.variance_decomposition[zi], [1.0])
    np.testing.assert_allclose(mom.std**2, np.diag(mom.covariance))
    sim = ek.simulate(sol, n_periods=20_000, n_agents=10, burn_in=100, rng=3)
    zpaths = sim.paths["z"]
    per_agent = (zpaths**2).mean(axis=0)
    se = per_agent.std(ddof=1) / np.sqrt(per_agent.size)
    assert abs(per_agent.mean() - sig**2 / (1 - rho**2)) < 4 * se
    assert abs(sim.moments(["autocorr_z"])["autocorr_z"] - rho) < 0.01


def test_irf_and_moments_validation_and_store() -> None:
    sol = ek.solve(models.load("nk3").compile())
    with pytest.raises(ek.errors.ValidationError):
        irf(sol, "nope")
    with pytest.raises(ek.errors.ValidationError):
        irf(sol, 5)
    with pytest.raises(ek.errors.ValidationError):
        irf(sol, 0, horizon=0)
    only = simulate_linear(sol, n_periods=10, rng=0, store=["pi"])
    assert list(only.paths) == ["pi"]
    with pytest.raises(ek.errors.ValidationError):
        simulate_linear(sol, n_periods=10, store=["zzz"])
    with pytest.raises(ek.errors.ValidationError):
        simulate_linear(sol, n_periods=0)


def test_rbc_steady_state_and_irf_shape() -> None:
    from dynecon.models.rbc import steady_state as rbc_ss

    sol = ek.solve(models.load("rbc").compile())
    s = rbc_ss(0.33, 0.99, 0.025)
    assert abs(s["k_over_y"] - 0.33 / (1 / 0.99 - 1 + 0.025)) < 1e-12
    assert abs(s["i"] / s["y"] - 0.025 * 0.33 / (1 / 0.99 - 1 + 0.025)) < 1e-12
    np.testing.assert_allclose(sol.steady_state[0], s["k"], rtol=1e-12)
    out = irf(sol, "eps_z", horizon=60)["y"]
    peak = int(np.argmax(out))
    assert peak <= 3 and np.all(np.diff(out[peak:]) < 1e-12)


def test_rbc_full_depreciation_equals_brock_mirman() -> None:
    rbc = ek.solve(
        models.load("rbc", delta=1.0, beta=0.96, alpha=0.36, rho=0.9, sigma_e=0.02).compile()
    )
    bm = ek.solve(models.load("brock_mirman", problem="dsge").compile())
    np.testing.assert_allclose(rbc.P, bm.P, atol=1e-6)
    np.testing.assert_allclose(rbc.F[0], bm.F[0], atol=1e-6)  # consumption policy


def test_rbc_perturbation_matches_egm_near_steady_state() -> None:
    dsge = ek.solve(models.load("rbc").compile())
    egm = ek.solve(models.load("rbc", problem="saving").compile(), method="egm")
    k_ss = dsge.steady_state[0]
    c_pert = dsge.steady_state[2]  # c at (k_ss, z=1)
    f = egm.policy_function("c")
    z_mid = egm.model.shocks["z"].values.shape[0] // 2
    assert abs(float(f(k_ss, z_mid)) / c_pert - 1.0) < 1e-3


def test_olg_steady_state_three_ways_and_stability() -> None:
    alpha, beta, n = 0.3, 0.3, 0.5
    kstar = steady_state_capital(alpha, beta, n)
    assert abs(kstar - euler_steady_state(alpha, beta, n)) < 1e-10
    sol = ek.solve(models.load("olg2", alpha=alpha, beta=beta, n=n).compile())
    assert abs(sol.steady_state[0] - kstar) < 1e-12
    np.testing.assert_allclose(sol.P, [[alpha]], atol=1e-6)  # slope of the law of motion
    for k0 in (0.1 * kstar, kstar, 10 * kstar):
        path = iterate_capital(k0, 200, alpha, beta, n)
        assert abs(path[-1] - kstar) < 1e-10
    assert abs(models.olg2.PARAMS["beta"].value - 0.96**30) < 1e-15


def test_numerical_steady_state_from_guess_and_failure() -> None:
    bm = models.load("brock_mirman", problem="dsge").compile()
    prob = bm.problem
    from dataclasses import replace

    guessed = replace(prob, steady_state=None, guess=lambda p: np.array([0.1, 1.0, 0.1]))
    ss = steady_state(guessed, bm.p)
    np.testing.assert_allclose(ss, prob.steady_state(bm.p), rtol=1e-8)
    broken = replace(
        prob,
        steady_state=None,
        guess=lambda p: np.array([1.0, 1.0, 1.0]),
        equations=lambda yp, y, p: [yp.k + 1.0, yp.z - 1.0, yp.c - 1.0],
    )
    with pytest.raises(ConvergenceError, match="residual"):
        steady_state(broken, bm.p)


def test_dsge_template_validation() -> None:
    def build(**kw):
        m = Model("d")
        m.params(rho=Param(0.5))
        base = dict(
            variables=["x", "y"],
            states=["x"],
            shocks=["e"],
            equations=lambda yp, y, p: [yp.x - p.rho * y.x, y.y - y.x],
            shock_loadings=lambda p: np.array([[1.0]]),
            steady_state=lambda p: np.zeros(2),
        )
        m.set_problem("dsge", **{**base, **kw})
        return m

    assert build().compile().problem.variables == ("x", "y")
    for bad, msg in [
        ({"states": ["y"]}, "states"),
        ({"steady_state": None}, "steady_state"),
        ({"equations": lambda yp, y, p: [yp.x]}, "residual"),
        ({"shock_loadings": lambda p: np.ones((2, 2))}, "shock_loadings"),
        ({"log_variables": ["w"]}, "log_variables"),
        ({"variables": ["x", "x"], "states": ["x"]}, "unique"),
    ]:
        with pytest.raises(ModelSpecificationError, match=msg):
            build(**bad).compile()


@settings(deadline=None, max_examples=30, derandomize=True)
@given(rho=st.floats(0.0, 0.95), sig=st.floats(0.01, 1.0))
def test_lyapunov_covariance_satisfies_equation(rho: float, sig: float) -> None:
    sol = ek.solve(models.load("nk3", rho_v=rho).compile())
    g0 = solve_discrete_lyapunov(sol.P, sol.Sigma @ sol.Sigma.T)
    np.testing.assert_allclose(g0, sol.P @ g0 @ sol.P.T + sol.Sigma @ sol.Sigma.T, atol=1e-10)
    del sig


def test_registry_errors_and_unknown_option() -> None:
    with pytest.raises(ek.errors.ValidationError):
        models.load("nope")
    with pytest.raises(ek.errors.ValidationError):
        models.load("nk3", phi_zzz=1.0)
    with pytest.raises(ek.errors.ValidationError):
        models.register("nk3", lambda: None)  # type: ignore[arg-type, return-value]
    with pytest.raises(ek.errors.ValidationError):
        models.load("brock_mirman", problem="bogus")
    with pytest.raises(ek.errors.ValidationError):
        ek.solve(models.load("nk3").compile(), method="vfi")
