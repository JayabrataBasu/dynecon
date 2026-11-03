"""Moment sets, target losses, parameter transforms and priors."""

import numpy as np
import pytest

import econkit as ek
from econkit import models
from econkit.core.errors import ValidationError
from econkit.estimation import MomentSet, MomentTarget, model_moments, quadratic_moment_loss
from econkit.estimation.priors import Beta, Gamma, InverseGamma, Normal, Uniform
from econkit.model import Param
from econkit.model.specs import ParameterSet


def test_quadratic_loss_and_errors() -> None:
    targets = [MomentTarget("a", 1.0, 2.0), MomentTarget("b", 0.5)]
    assert quadratic_moment_loss({"a": 1.2, "b": 0.4}, targets) == pytest.approx(2 * 0.04 + 0.01)
    with pytest.raises(ValidationError):
        quadratic_moment_loss({"a": 1.0}, targets)
    with pytest.raises(ValidationError):
        quadratic_moment_loss({}, [])


def test_moment_set_drops_missing_data_and_builds_weights() -> None:
    ms = MomentSet(
        (MomentTarget("mean_assets", 1.0, 2.0), MomentTarget("std_assets", float("nan")))
    )
    assert ms.names == ["mean_assets"] and ms.dropped == ("std_assets",)
    np.testing.assert_allclose(ms.weights, [[2.0]])
    with pytest.raises(ValidationError):
        MomentSet((MomentTarget("x", float("nan")),))


def test_model_moment_names_on_a_simulation() -> None:
    sol = ek.solve(models.load("consumption_savings", n_a=60, n_z=3).compile())
    sim = ek.simulate(sol, n_periods=400, n_agents=30, rng=0)
    names = [
        "mean_assets",
        "autocorr2_income",
        "gini_assets",
        "corr:assets:consumption",
        "slope:consumption:assets",
        "p90_assets",
        "var_income",
    ]
    out = model_moments(sim, names)
    assert out.shape == (7,) and np.all(np.isfinite(out))
    assert 0.0 < out[2] < 1.0 and out[3] > 0.5 and 0.0 < out[4] < 1.0
    ms = MomentSet(tuple(MomentTarget(n, v) for n, v in zip(names, out, strict=True)))
    np.testing.assert_allclose(ms.model_moments(sim), out)


def test_transforms_round_trip_and_jacobian() -> None:
    ps = ParameterSet(
        {
            "a": Param(0.3, bounds=(0.0, 1.0), free=True),
            "b": Param(2.0, bounds=(1.0, np.inf), free=True),
            "c": Param(-1.0, free=True),
            "d": Param(0.5, bounds=(0.0, 1.0), free=True, transform="none"),
        }
    )
    x = ps.vector()
    u = ps.to_unconstrained(x)
    np.testing.assert_allclose(ps.from_unconstrained(u), x, atol=1e-12)
    assert np.all(np.isfinite(ps.from_unconstrained([50.0, 50.0, 50.0, 50.0])))
    assert 0.0 < ps.from_unconstrained([-50.0, -50.0, 0.0, 0.0])[0] < 1.0
    # log |d theta / d u| equals the numerical derivative
    for name in ("a", "b", "c"):
        i = ps.names().index(name)
        h = 1e-6
        up, dn = u.copy(), u.copy()
        up[i] += h
        dn[i] -= h
        deriv = (ps.from_unconstrained(up)[i] - ps.from_unconstrained(dn)[i]) / (2 * h)
        single = ParameterSet({name: ps[name]})
        assert single.log_jacobian([u[i]]) == pytest.approx(np.log(abs(deriv)), abs=1e-5)
    with pytest.raises(ValidationError):
        ps.to_unconstrained([1.5, 2.0, 0.0, 0.5])
    with pytest.raises(ValidationError):
        ps.to_unconstrained([0.5, 0.5, 0.0, 0.5])


@pytest.mark.parametrize(
    "prior,lo,hi",
    [
        (Normal(1.0, 0.5), -2.0, 4.0),
        (Uniform(2.0, 3.0), 2.0, 3.0),
        (Beta(2.0, 5.0), 0.0, 1.0),
        (Gamma(3.0, 0.5), 0.0, 12.0),
        (InverseGamma(4.0, 2.0), 0.0, 60.0),
    ],
)
def test_priors_integrate_to_one_and_sample(prior, lo: float, hi: float) -> None:
    grid = np.linspace(lo + 1e-9, hi - 1e-9, 20_001)
    dens = np.exp([prior.logpdf(x) for x in grid])
    area = np.trapezoid(dens, grid)
    assert abs(area - 1.0) < 1e-3
    rng = np.random.default_rng(0)
    draws = np.array([prior.sample(rng) for _ in range(4000)])
    mean_theory = float(np.trapezoid(grid * dens, grid))
    se = draws.std(ddof=1) / np.sqrt(draws.size)
    assert abs(draws.mean() - mean_theory) < 4 * se
    if np.isfinite(prior.support[0]):
        assert prior.logpdf(prior.support[0] - 1.0) == float("-inf")


def test_prior_validation() -> None:
    for bad in (
        lambda: Normal(0, 0),
        lambda: Uniform(1, 0),
        lambda: Beta(0, 1),
        lambda: Gamma(1, -1),
        lambda: InverseGamma(-1, 1),
    ):
        with pytest.raises(ValidationError):
            bad()
