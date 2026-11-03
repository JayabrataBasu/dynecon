"""SMM: closed-form recovery, J-test, identification, penalties and an end-to-end run."""

import numpy as np
import pytest
from scipy.signal import lfilter

import econkit as ek
from econkit import models
from econkit.core.errors import IdentificationError, ValidationError
from econkit.estimation import (
    MomentSet,
    MomentTarget,
    identification_report,
    loss_profile,
    make_moment_fn,
    smm,
)
from econkit.model import Model, Param

N_OBS, S = 5000, 20


def dummy_model(**params: Param) -> "ek.model.CompiledModel":
    """A parameter container: the 'model' is pure simulation inside moment_fn."""
    m = Model("dummy")
    m.params(**params)
    m.params(beta=Param(0.9))
    m.set_problem(
        "finite_dp",
        reward=lambda p: np.ones((1, 1)),
        transition=lambda p: np.ones((1, 1, 1)),
        discount="beta",
    )
    return m.compile()


def ar1_path(rho: float, sigma: float, n: int, seed: int) -> np.ndarray:
    e = np.random.default_rng(seed).standard_normal(n)
    return np.asarray(lfilter([sigma], [1.0, -rho], e))


def ac(y: np.ndarray, k: int) -> float:
    return float(np.corrcoef(y[k:], y[:-k])[0, 1])


def moments_of(y: np.ndarray, n_moments: int) -> np.ndarray:
    return np.array([y.var(), ac(y, 1), ac(y, 2)][:n_moments])


def ar1_model() -> "ek.model.CompiledModel":
    return dummy_model(
        rho=Param(0.5, bounds=(0.0, 0.999), free=True),
        sigma=Param(1.0, bounds=(0.01, 5.0), free=True),
    )


@pytest.mark.parametrize("seed", range(5))
def test_recovers_ar1_parameters_within_four_standard_errors(seed: int) -> None:
    n_mom = 2
    data = ar1_path(0.9, 0.5, N_OBS, 100 + seed)

    def fn(model, s):
        return moments_of(ar1_path(model.p.rho, model.p.sigma, S * N_OBS, s), n_mom)

    res = smm(
        ar1_model(),
        ["rho", "sigma"],
        moments_of(data, n_mom),
        fn,
        n_obs=N_OBS,
        n_sim=S,
        weights="identity",
        rng=seed,
        omega_reps=30,
    )
    assert res.converged and res.df == 0 and res.j_stat is None
    assert abs(res.params["rho"] - 0.9) < 4 * res.std_errors[0]
    assert abs(res.params["sigma"] - 0.5) < 4 * res.std_errors[1]
    assert res.loss < 1e-10  # exactly identified: the model hits the targets
    assert res.diagnostics["jacobian_step_gap"] < 1e-2
    assert "SMM estimates" in res.summary() and res.to_dataframe().shape == (2, 3)
    ident = identification_report(res)
    assert ident.rank == 2 and not ident.weak_parameters


def test_overidentified_j_test_and_optimal_weighting() -> None:
    passes = 0
    for seed in range(5):
        data = ar1_path(0.9, 0.5, N_OBS, 200 + seed)

        def fn(model, s):
            return moments_of(ar1_path(model.p.rho, model.p.sigma, S * N_OBS, s), 3)

        res = smm(
            ar1_model(),
            ["rho", "sigma"],
            moments_of(data, 3),
            fn,
            n_obs=N_OBS,
            n_sim=S,
            weights="optimal",
            rng=seed,
            omega_reps=40,
        )
        assert res.df == 1 and res.j_pvalue is not None
        passes += res.j_pvalue > 0.01
        assert res.config["weights"] == "optimal"
    assert passes >= 4


def test_loss_profile_minimised_at_estimate() -> None:
    data = ar1_path(0.9, 0.5, N_OBS, 7)

    def fn(model, s):
        return moments_of(ar1_path(model.p.rho, model.p.sigma, 2 * N_OBS, s), 3)

    res = smm(
        ar1_model(),
        ["rho", "sigma"],
        moments_of(data, 3),
        fn,
        n_obs=N_OBS,
        n_sim=2.0,
        omega_reps=10,
    )
    grid = res.params["rho"] + np.array([-0.05, 0.0, 0.05])
    prof = loss_profile(res, "rho", grid)
    assert prof[1] <= prof.min() + 1e-12
    re = loss_profile(res, "rho", grid, reoptimize=True)
    assert np.all(re <= prof + 1e-9)
    with pytest.raises(ValidationError):
        loss_profile(res, "nope", grid)


def test_weak_identification_rank_deficiency_detected() -> None:
    model = dummy_model(
        a=Param(1.0, bounds=(0.1, 5.0), free=True), b=Param(2.0, bounds=(0.1, 5.0), free=True)
    )

    def fn(m, s):
        e = np.random.default_rng(s).standard_normal(2) * 0.001
        ab = m.p.a * m.p.b
        return np.array([ab, ab**2]) + e

    res = smm(model, ["a", "b"], [3.0, 9.0], fn, n_obs=1000, n_sim=1.0, omega_reps=10)
    assert abs(res.params["a"] * res.params["b"] - 3.0) < 1e-2
    with pytest.raises(IdentificationError, match="rank-deficient"):
        identification_report(res)


def test_failed_evaluations_are_penalised_and_counted() -> None:
    calls = {"n": 0}
    model = dummy_model(x=Param(1.0, bounds=(0.1, 10.0), free=True))

    def fn(m, s):
        calls["n"] += 1
        if m.p.x > 3.0:
            raise ValueError("solver failed")
        return np.array([m.p.x])

    res = smm(model, ["x"], [2.0], fn, n_obs=100, n_sim=1.0, omega_reps=6)
    assert abs(res.params["x"] - 2.0) < 1e-3
    assert res.diagnostics["failed_evals"] >= 0


def test_argument_validation() -> None:
    model = ar1_model()

    def fn(m, s):
        return np.array([0.0])

    with pytest.raises(ValidationError):
        smm(model, [], [0.0], fn, n_obs=10)
    with pytest.raises(ValidationError, match="at least as many moments"):
        smm(model, ["rho", "sigma"], [0.0], fn, n_obs=10)
    with pytest.raises(ValidationError):
        smm(model, ["rho"], [0.0], fn, n_obs=10, weights="bogus")
    with pytest.raises(ValidationError):
        smm(model, ["rho"], [0.0, 1.0], fn, n_obs=10, weights=np.eye(3))


@pytest.mark.slow
def test_end_to_end_recovers_discount_factor_on_consumption_savings() -> None:
    base = dict(n_a=60, n_z=3, a_max=40.0)
    truth = 0.94
    names = ["mean_assets", "std_consumption"]
    sim_opts = {
        "n_periods": 300,
        "n_agents": 300,
        "burn_in": 100,
        "store": ["assets", "consumption"],
    }
    true_sol = ek.solve(models.load("consumption_savings", beta=truth, **base).compile())
    data_sim = ek.simulate(true_sol, rng=999, **sim_opts)
    moments = MomentSet(
        tuple(
            MomentTarget(n, float(v))
            for n, v in zip(names, ek.estimation.model_moments(data_sim, names), strict=True)
        )
    )
    model = models.load("consumption_savings", beta=0.9, **base)
    model.params(beta=Param(0.9, bounds=(0.8, 0.99), free=True))
    res = smm(
        model.compile(),
        ["beta"],
        None,
        moments,
        n_obs=300 * 300,
        n_sim=1.0,
        solve_options={"tol": 1e-8},
        sim_options=sim_opts,
        rng=1,
        omega_reps=10,
    )
    assert abs(res.params["beta"] / truth - 1) < 0.02
    assert res.df == 1 and res.j_pvalue is not None


def test_make_moment_fn_pipeline_runs() -> None:
    mset = MomentSet((MomentTarget("mean_k", 1.0), MomentTarget("std_c", 0.1)))
    fn = make_moment_fn(mset, {"method": "egm"}, {"n_periods": 50, "n_agents": 5})
    out = fn(models.load("brock_mirman", n_k=40, n_z=3).compile(), 3)
    assert out.shape == (2,) and np.all(np.isfinite(out))
