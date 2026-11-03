"""Model builders shared by tests (the built-in library arrives in P10)."""

from __future__ import annotations

import numpy as np

from dynecon.model import AR1Shock, Control, Model, Param, State


def consumption_savings(
    beta: float = 0.96,
    sigma: float = 2.0,
    r: float = 0.03,
    rho: float = 0.9,
    sigma_e: float = 0.1,
    n_a: int = 100,
    n_z: int = 5,
    a_max: float = 20.0,
    a_min: float = 0.0,
    curvature: float = 3.0,
) -> Model:
    m = Model("consumption_savings")
    m.params(
        beta=Param(beta, bounds=(0.5, 0.9999)),
        sigma=Param(sigma, bounds=(0.2, 10.0)),
        r=Param(r, bounds=(-0.1, 0.2)),
        rho=Param(rho, bounds=(0.0, 0.99)),
        sigma_e=Param(sigma_e, bounds=(0.001, 1.0)),
    )
    m.add_state(
        State(
            "assets", bounds=(a_min, a_max), grid_type="exp", n_points=n_a, grid_curvature=curvature
        )
    )
    m.add_state(State("income", shock=AR1Shock("rho", "sigma_e", n_z, log=True)))
    m.add_control(Control("consumption"))
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=lambda c, p: np.log(c) if p.sigma == 1.0 else c ** (1 - p.sigma) / (1 - p.sigma),
        marginal_utility=lambda c, p: c ** (-p.sigma),
        inverse_marginal_utility=lambda x, p: x ** (-1 / p.sigma),
        resources=lambda a, z, p: (1 + p.r) * a + z,
        marginal_return=lambda a, z, p: 1 + p.r,
    )
    return m


def permanent_income(n_a: int = 101, a_max: float = 10.0) -> Model:
    """Deterministic income y=1, r=0.03, beta=1/(1+r), CRRA 2, a_min=0 (P5.9 / P6.5)."""
    r = 0.03
    m = Model("permanent_income")
    m.params(beta=Param(1 / (1 + r)), sigma=Param(2.0), r=Param(r), y=Param(1.0))
    m.add_state(State("assets", bounds=(0.0, a_max), n_points=n_a))
    m.add_control(Control("consumption"))
    m.set_problem(
        "saving",
        asset="assets",
        control="consumption",
        discount="beta",
        utility=lambda c, p: c ** (1 - p.sigma) / (1 - p.sigma),
        marginal_utility=lambda c, p: c ** (-p.sigma),
        inverse_marginal_utility=lambda x, p: x ** (-1 / p.sigma),
        resources=lambda a, z, p: (1 + p.r) * a + p.y,
        marginal_return=lambda a, z, p: 1 + p.r,
    )
    return m


ALPHA, BETA_BM = 0.36, 0.96


def brock_mirman(n_k: int = 200, n_z: int = 7, rho: float = 0.9, sigma_e: float = 0.02) -> Model:
    """Log utility, Cobb-Douglas, full depreciation: k' = alpha*beta*z*k^alpha."""
    kstar = (ALPHA * BETA_BM) ** (1 / (1 - ALPHA))
    m = Model("brock_mirman")
    m.params(alpha=Param(ALPHA), beta=Param(BETA_BM))
    m.add_state(State("k", bounds=(0.05 * kstar, 2.0 * kstar), n_points=n_k))
    m.add_state(State("z", shock=AR1Shock(rho, sigma_e, n_z, log=True)))
    m.add_control(Control("c"))
    m.set_problem(
        "saving",
        asset="k",
        shock="z",
        control="c",
        discount="beta",
        utility=lambda c, p: np.log(c),
        marginal_utility=lambda c, p: 1.0 / c,
        inverse_marginal_utility=lambda x, p: 1.0 / x,
        resources=lambda k, z, p: z * k**p.alpha,
        marginal_return=lambda k, z, p: p.alpha * z * k ** (p.alpha - 1.0),
    )
    return m


def mean_consumption_metric(sol) -> dict[str, float]:
    """Importable metric function for process-pool sweeps."""
    return {"mean_c": float(sol.policy["consumption"].mean())}


def failing_metric(sol) -> dict[str, float]:
    raise ValueError("boom")
