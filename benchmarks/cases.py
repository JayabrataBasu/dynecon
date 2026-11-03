"""The benchmark cases (inputs are deterministic)."""

from __future__ import annotations

from typing import Any

import numpy as np

import econkit as ek
from benchmarks.harness import Case
from econkit import models
from econkit.estimation import kalman_filter
from econkit.numerics.interpolation import linear_interp_1d
from econkit.simulation import iterate_distribution, simulate
from econkit.solvers.dp.vfi import build_payoff_tensor
from econkit.solvers.perturbation import irf


def _cs(n_a: int, n_z: int = 7) -> Any:
    return models.load("consumption_savings", n_a=n_a, n_z=n_z, a_max=50.0).compile()


def interp1d() -> Case:
    def setup() -> Any:
        rng = np.random.default_rng(0)
        xp = np.sort(rng.uniform(0, 10, 1000)) + np.arange(1000) * 1e-4
        return xp, np.sin(xp), rng.uniform(0, 10, 1_000_000)

    return Case(
        "interp1d",
        "linear_interp_1d, 10^6 queries on a 1000-node grid",
        setup,
        lambda s, b: linear_interp_1d(s[2], s[0], s[1], backend=b),
        ("numpy", "numba", "jax"),
    )


def bellman_cs() -> Case:
    from econkit.numerics._dispatch import resolve_backend

    def setup() -> Any:
        m = _cs(1000)
        U = build_payoff_tensor(m, 4.0)
        evt = np.random.default_rng(1).standard_normal((7, 1000)).cumsum(axis=1) * 0.01
        return U, np.ascontiguousarray(evt)

    def run(s: Any, backend: str) -> np.ndarray:
        _, kernel = resolve_backend("bellman_saving", backend)
        v, idx = kernel(s[0], s[1], True)
        return np.concatenate([v.ravel(), idx.ravel().astype(float)])

    return Case(
        "bellman_cs",
        "one Bellman maximisation, n_a=1000, n_z=7",
        setup,
        run,
        ("numpy", "numba", "jax"),
    )


def vfi_cs() -> Case:
    return Case(
        "vfi_cs",
        "full VFI solve (monotone + Howard), n_a=500, n_z=7",
        lambda: _cs(500),
        lambda m, b: ek.solve(m, method="vfi", backend=b, tol=1e-8).value,
        ("numpy", "numba"),
    )


def egm_cs() -> Case:
    return Case(
        "egm_cs",
        "full EGM solve, n_a=1000, n_z=7",
        lambda: _cs(1000),
        lambda m, b: ek.solve(m, method="egm", backend=b).policy["consumption"],
        ("numpy", "numba", "jax"),
        tolerance=1e-6,
    )


def egm_rbc() -> Case:
    return Case(
        "egm_rbc",
        "RBC (saving form) EGM solve, n_k=300, n_z=7",
        lambda: models.load("rbc", problem="saving", n_k=300, n_z=7).compile(),
        lambda m, b: ek.solve(m, method="egm", backend=b).policy["c"],
    )


def young_stationary() -> Case:
    def setup() -> Any:
        sol = ek.solve(_cs(1000))
        return sol, np.full(sol.policy["assets_next"].shape, 1.0 / sol.policy["assets_next"].size)

    return Case(
        "young_stationary",
        "200 histogram steps, n_a=1000, n_z=7",
        setup,
        lambda s, b: iterate_distribution(s[1], s[0], 200, backend=b),
        ("numpy", "numba", "jax"),
    )


def simulate_panel() -> Case:
    return Case(
        "simulate_panel",
        "panel simulation, 2x10^4 agents x 500 periods",
        lambda: ek.solve(_cs(200)),
        lambda s, b: simulate(s, n_periods=500, n_agents=20_000, rng=0, backend=b).paths["assets"][
            -1
        ],
        tolerance=1e-8,
    )


def aiyagari_ge() -> Case:
    from econkit.models import aiyagari

    return Case(
        "aiyagari_ge",
        "Aiyagari general equilibrium, n_a=100, n_z=5",
        lambda: None,
        lambda s, b: np.array(
            [aiyagari.equilibrium(sigma=2.0, rho=0.9, sigma_e=0.2, n_a=100, n_z=5).price]
        ),
        ("numpy",),
        tolerance=1e-8,
    )


def perturbation_rbc() -> Case:
    def run(m: Any, backend: str) -> np.ndarray:
        sol = ek.solve(m, backend=backend)
        return np.concatenate([sol.F.ravel(), sol.P.ravel(), irf(sol, 0, horizon=20)["y"]])

    return Case(
        "perturbation_rbc",
        "RBC linearise + Klein + IRF",
        lambda: models.load("rbc").compile(),
        run,
        ("numpy", "jax"),
        tolerance=1e-6,
    )


def kalman_loglik() -> Case:
    def setup() -> Any:
        rng = np.random.default_rng(2)
        return np.cumsum(rng.standard_normal(500)) * 0.3 + rng.standard_normal(500)

    return Case(
        "kalman_loglik",
        "local-level Kalman likelihood, T=500",
        setup,
        lambda y, b: np.array(
            [
                kalman_filter(
                    y, np.eye(1), np.eye(1), np.eye(1), 0.1 * np.eye(1), np.zeros(1), 10 * np.eye(1)
                ).loglik
            ]
        ),
        ("numpy",),
    )


def smm_ar1() -> Case:
    from scipy.signal import lfilter

    from econkit.estimation import smm
    from econkit.model import Model, Param

    def setup() -> Any:
        m = Model("ar1")
        m.params(
            rho=Param(0.5, bounds=(0.0, 0.999), free=True),
            sigma=Param(1.0, bounds=(0.01, 5.0), free=True),
            beta=Param(0.9),
        )
        m.set_problem(
            "finite_dp",
            reward=lambda p: np.ones((1, 1)),
            transition=lambda p: np.ones((1, 1, 1)),
            discount="beta",
        )
        data = lfilter([0.5], [1.0, -0.9], np.random.default_rng(3).standard_normal(5000))
        return m.compile(), np.array([data.var(), np.corrcoef(data[1:], data[:-1])[0, 1]])

    def moments(model: Any, seed: int) -> np.ndarray:
        y = lfilter(
            [model.p.sigma],
            [1.0, -model.p.rho],
            np.random.default_rng(seed).standard_normal(100_000),
        )
        return np.array([y.var(), np.corrcoef(y[1:], y[:-1])[0, 1]])

    def run(s: Any, backend: str) -> np.ndarray:
        res = smm(
            s[0], ["rho", "sigma"], s[1], moments, n_obs=5000, n_sim=20.0, rng=0, omega_reps=10
        )
        return res.estimates

    return Case("smm_ar1", "SMM of an AR(1), S=20", setup, run, ("numpy",), tolerance=1e-6)


ALL = [
    interp1d,
    bellman_cs,
    vfi_cs,
    egm_cs,
    egm_rbc,
    young_stationary,
    simulate_panel,
    aiyagari_ge,
    perturbation_rbc,
    kalman_loglik,
    smm_ar1,
]
