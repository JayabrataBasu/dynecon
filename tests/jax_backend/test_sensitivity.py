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


def test_bad_inputs_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    model = models.load("consumption_savings", n_a=40, n_z=3).compile()
    with pytest.raises(ValidationError):
        stationary_moments(model, ["not_a_parameter"], moment_fn)
    with pytest.raises(ValidationError):
        stationary_moments(model, [], moment_fn)
    monkeypatch.setattr("dynecon.estimation.sensitivity.MAX_STATES", 100)  # 40 x 3 states > 100
    with pytest.raises(ValidationError):
        stationary_moments(model, ["beta"], moment_fn)


def build_ge(**kw):
    """Aiyagari-style economy: wage and capital demand computed from the parameter view."""
    from dynecon.model import Model
    from dynecon.model.specs import AR1Shock, Control, Param, State
    from dynecon.models.utility import crra, crra_marginal, crra_marginal_inverse

    m = Model("ge")
    m.params(
        beta=Param(0.95, (0.9, 0.985), True, "discount"),
        sigma=Param(2.0, (0.5, 10.0), True, "CRRA"),
        r=Param(0.03, (-0.1, 0.2), False, "interest rate"),
        alpha=Param(0.36, (0.1, 0.9), True, "capital share"),
        delta=Param(0.08, (0.01, 0.3), True, "depreciation"),
        rho=Param(0.9, (0.0, 0.99), True, "persistence"),
        sigma_e=Param(0.1, (0.02, 0.3), True, "innovation sd"),
    )
    m.add_state(
        State("assets", bounds=(0.0, 60.0), grid_type="exp", n_points=40, grid_curvature=3.0)
    )
    m.add_state(State("income", shock=AR1Shock("rho", "sigma_e", 3, log=True)))
    m.add_control(Control("consumption"))
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=lambda c, p: crra(c, p.sigma),
        marginal_utility=lambda c, p: crra_marginal(c, p.sigma),
        inverse_marginal_utility=lambda x, p: crra_marginal_inverse(x, p.sigma),
        resources=lambda a, z, p: (1.0 + p.r) * a + (1.0 - p.alpha) * kd(p) ** p.alpha * z,
        marginal_return=lambda a, z, p: 1.0 + p.r,
    )
    return m.compile().with_params(**kw)


def kd(p):
    return ((p.r + p.delta) / p.alpha) ** (1.0 / (p.alpha - 1.0))


def excess_fn(mu, c, a_next, a_grid, z, p):
    return (mu * a_next).sum() - kd(p)


def equilibrium_numpy(**kw):
    from dynecon.solvers.equilibrium import solve_stationary_equilibrium

    base = build_ge(**kw)

    def excess(r, sol, mu):
        k = float((mu * sol.policy["assets_next"]).sum())
        d = float(((r + base.p.delta) / base.p.alpha) ** (1.0 / (base.p.alpha - 1.0)))
        return k - d, {}

    return solve_stationary_equilibrium(
        lambda r: base.with_params(r=r),
        "r",
        (0.001, 1.0 / base.p.beta - 1.0 - 1e-4),
        excess,
        tol=1e-13,
        solve_options={"tol": 1e-14, "max_iter": 100_000},
    )


def test_equilibrium_derivatives_match_finite_differences() -> None:
    from dynecon.estimation.sensitivity import equilibrium_moments

    names = ["beta", "sigma", "alpha", "delta", "rho", "sigma_e"]
    base = {"beta": 0.95, "sigma": 2.0, "alpha": 0.36, "delta": 0.08, "rho": 0.9, "sigma_e": 0.1}
    eq = equilibrium_numpy(**base)
    m, G, dr = equilibrium_moments(eq.model, names, "r", excess_fn, moment_fn)

    def numpy_eq(**kw):
        e = equilibrium_numpy(**kw)
        mu, s = e.distribution, e.household_solution
        a_next, c = s.policy["assets_next"], s.policy["consumption"]
        ma = (mu * a_next).sum()
        return e.price, np.array([ma, (mu * c).sum(), (mu * a_next**2).sum() - ma**2])

    r0, m0 = numpy_eq(**base)
    np.testing.assert_allclose(m, m0, rtol=1e-8)
    for j, name in enumerate(names):
        h = 1e-5 * max(abs(base[name]), 0.1)
        rp, mp = numpy_eq(**dict(base, **{name: base[name] + h}))
        rm, mm = numpy_eq(**dict(base, **{name: base[name] - h}))
        np.testing.assert_allclose(dr[j], (rp - rm) / (2 * h), rtol=2e-4, atol=1e-7, err_msg=name)
        np.testing.assert_allclose(G[:, j], (mp - mm) / (2 * h), rtol=2e-4, atol=1e-6, err_msg=name)


def test_equilibrium_moments_rejects_non_equilibrium() -> None:
    from dynecon.core.errors import ConvergenceError
    from dynecon.estimation.sensitivity import equilibrium_moments

    with pytest.raises(ConvergenceError):
        equilibrium_moments(build_ge(), ["beta"], "r", excess_fn, moment_fn)


def test_builtin_aiyagari_equilibrium_derivatives() -> None:
    from dynecon.estimation.sensitivity import equilibrium_moments
    from dynecon.models import aiyagari

    names = ["beta", "sigma", "alpha", "delta", "rho", "sigma_e"]
    base = {"beta": 0.96, "sigma": 3.0, "alpha": 0.36, "delta": 0.08, "rho": 0.6, "sigma_e": 0.2}
    grid = {
        "n_a": 40,
        "n_z": 3,
        "a_max": 80.0,
    }  # fixed grid: both sides see the same discretisation

    def solved(**o):
        e = aiyagari.equilibrium(**grid, **o)
        mu, s = e.distribution, e.household_solution
        a_next, c = s.policy["assets_next"], s.policy["consumption"]
        ma = (mu * a_next).sum()
        return e, e.price, np.array([ma, (mu * c).sum(), (mu * a_next**2).sum() - ma**2])

    eq, r0, m0 = solved(**base)
    m, G, dr = equilibrium_moments(eq.model, names, "r", aiyagari.excess_demand, moment_fn)
    np.testing.assert_allclose(m, m0, rtol=1e-7)
    for j, name in enumerate(names):
        h = 1e-4 * max(abs(base[name]), 0.1)
        _, rp, mp = solved(**dict(base, **{name: base[name] + h}))
        _, rm, mm = solved(**dict(base, **{name: base[name] - h}))
        np.testing.assert_allclose(dr[j], (rp - rm) / (2 * h), rtol=1e-3, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(G[:, j], (mp - mm) / (2 * h), rtol=1e-3, atol=1e-5, err_msg=name)


def test_builtin_huggett_equilibrium_derivatives() -> None:
    import jax.numpy as jnp

    from dynecon.estimation.sensitivity import equilibrium_moments
    from dynecon.models import huggett

    names = ["beta", "sigma", "rho", "sigma_e"]
    base = {"beta": 0.96, "sigma": 2.0, "rho": 0.9, "sigma_e": 0.2}
    grid = {"n_a": 40, "n_z": 3, "a_max": 60.0}

    def solved(**o):
        e = huggett.equilibrium(**grid, **o)
        mu, s = e.distribution, e.household_solution
        a_next, c = s.policy["assets_next"], s.policy["consumption"]
        return e, e.price, np.array([(mu * c).sum(), (mu * a_next**2).sum()])

    eq, r0, m0 = solved(**base)
    fn = lambda mu, c, a_next, *_: jnp.stack([(mu * c).sum(), (mu * a_next**2).sum()])  # noqa: E731
    m, G, dr = equilibrium_moments(eq.model, names, "r", huggett.excess_demand, fn)
    np.testing.assert_allclose(m, m0, rtol=1e-6)
    for j, name in enumerate(names):
        h = 1e-4 * max(abs(base[name]), 0.1)
        _, rp, mp = solved(**dict(base, **{name: base[name] + h}))
        _, rm, mm = solved(**dict(base, **{name: base[name] - h}))
        np.testing.assert_allclose(dr[j], (rp - rm) / (2 * h), rtol=1e-3, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(G[:, j], (mp - mm) / (2 * h), rtol=1e-3, atol=1e-5, err_msg=name)


def test_matrix_free_path_matches_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    import jax.numpy as jnp

    from dynecon.estimation import sensitivity
    from dynecon.models import aiyagari

    model = aiyagari.build(n_a=40, n_z=3, a_max=80.0, r=0.03).compile()
    names = ["beta", "sigma_e"]
    fn = lambda mu, c, a_next, *_: jnp.stack([(mu * a_next).sum(), (mu * c**2).sum()])  # noqa: E731
    _, dense = sensitivity.stationary_moments(model, names, fn)
    monkeypatch.setattr(sensitivity, "DENSE_STATES", 0)
    _, free = sensitivity.stationary_moments(model, names, fn)
    np.testing.assert_allclose(free, dense, rtol=1e-6, atol=1e-8)
