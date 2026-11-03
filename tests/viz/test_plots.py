"""Plot smoke tests on the Agg backend: assert structure, not pixels."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import dynecon as ek  # noqa: E402
from dynecon import models  # noqa: E402
from dynecon.core.errors import ValidationError  # noqa: E402
from dynecon.estimation import metropolis, smm  # noqa: E402
from dynecon.estimation.priors import Normal  # noqa: E402
from dynecon.model import Model, Param  # noqa: E402
from dynecon.models import huggett  # noqa: E402
from dynecon.simulation import marginal_distribution, simulate_irf, stationary_distribution_young  # noqa: E402
from dynecon.viz import (  # noqa: E402
    dynecon_style,
    plot_convergence,
    plot_distribution,
    plot_equilibrium,
    plot_euler_errors,
    plot_irf,
    plot_lorenz,
    plot_mcmc_trace,
    plot_moment_fit,
    plot_policy,
    plot_posterior,
    plot_simulation,
    plot_sweep,
    plot_value,
)

pytestmark = pytest.mark.plot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def vfi():
    return ek.solve(models.load("consumption_savings", n_a=40, n_z=3).compile(), method="vfi")


@pytest.fixture(scope="module")
def egm():
    return ek.solve(models.load("consumption_savings", n_a=40, n_z=3).compile(), method="egm")


def test_policy_value_euler_convergence(vfi, egm) -> None:
    rc_before = dict(mpl.rcParams)
    ax = plot_policy(egm, "consumption")
    assert len(ax.lines) == 3 and ax.get_xlabel() == "assets" and ax.get_ylabel() == "consumption"
    np.testing.assert_allclose(ax.lines[0].get_xdata(), egm.grids["assets"])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), egm.policy["consumption"][:, 1])
    assert len(plot_policy(egm, "consumption", state_slice=1).lines) == 1
    assert len(plot_value(vfi).lines) == 3
    assert plot_euler_errors(egm).get_ylabel().startswith("log10")
    ax = plot_convergence(egm)
    assert ax.get_yscale() == "log" and len(ax.lines) == 1
    assert dict(mpl.rcParams) == rc_before  # global rcParams untouched
    with pytest.raises(ValidationError):
        plot_value(egm)
    with pytest.raises(ValidationError):
        plot_policy(egm, "nope")
    with pytest.raises(ValidationError):
        plot_convergence(ek.solve(models.load("nk3").compile()))


def test_simulation_and_irf(egm) -> None:
    sim = ek.simulate(egm, n_periods=60, n_agents=3, rng=0)
    ax = plot_simulation(sim, ["assets", "consumption"], agent=1)
    assert len(ax.lines) == 2 and ax.lines[0].get_xdata().size == 60
    sol = ek.solve(models.load("brock_mirman", n_k=80, n_z=5).compile(), method="egm")
    irf = simulate_irf(sol, horizon=6, n_paths=200, burn_in=50, rng=0)
    fig = plot_irf(irf)
    assert len([a for a in fig.axes if a.get_visible()]) == 3
    assert all(a.get_title() for a in fig.axes if a.get_visible())
    with pytest.raises(ValidationError):
        plot_simulation(sim, ["shock_index"])


def test_distribution_and_lorenz(egm) -> None:
    mu = stationary_distribution_young(egm)
    mass = marginal_distribution(mu, 0)
    fig = plot_distribution(mass, egm.grids["assets"])
    assert len(fig.axes) == 2
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata()[-1], 1.0)
    ax = plot_lorenz(egm.grids["assets"], mass)
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert (x[0], y[0]) == (0.0, 0.0)
    assert x[-1] == pytest.approx(1.0) and y[-1] == pytest.approx(1.0)
    assert "Gini" in ax.get_legend().get_texts()[0].get_text()


def test_estimation_plots() -> None:
    m = Model("d")
    m.params(x=Param(1.0, bounds=(0.1, 5.0), free=True), beta=Param(0.9))
    m.set_problem(
        "finite_dp",
        reward=lambda p: np.ones((1, 1)),
        transition=lambda p: np.ones((1, 1, 1)),
        discount="beta",
    )

    def fn(model, seed):
        e = np.random.default_rng(seed).standard_normal(2) * 0.01
        return np.array([model.p.x, model.p.x**2]) + e

    res = smm(m.compile(), ["x"], [2.0, 4.0], fn, n_obs=500, n_sim=1.0, omega_reps=8)
    ax = plot_moment_fit(res, ["m", "m2"])
    assert len(ax.lines) >= 2 and [t.get_text() for t in ax.get_xticklabels()] == ["m", "m2"]
    chain = metropolis(
        lambda v: float(-0.5 * v @ v), [0.0], 400, n_chains=2, burn_in=100, rng=0, names=["theta"]
    )
    ax = plot_mcmc_trace(chain, "theta")
    assert len(ax.lines) == 2 and ax.get_ylabel() == "theta"
    ax = plot_posterior(chain, 0, prior=Normal(0.0, 1.0))
    assert len(ax.patches) > 10 and len(ax.lines) == 1
    with pytest.raises(ValidationError):
        plot_mcmc_trace(chain, "nope")


def test_equilibrium_plot() -> None:
    eq = huggett.equilibrium(n_a=40, n_z=3)
    ax = plot_equilibrium(eq)
    xs = ax.lines[0].get_xdata()
    assert np.all(np.diff(xs) >= 0) and ax.get_xlabel() == "r"
    assert any(abs(line.get_xdata()[0] - eq.price) < 1e-12 for line in ax.lines[2:])


def test_style_context_does_not_leak() -> None:
    before = mpl.rcParams["axes.grid"]
    with dynecon_style():
        assert mpl.rcParams["axes.grid"] is True
    assert mpl.rcParams["axes.grid"] == before


def test_sweep_plot() -> None:
    from builders import mean_consumption_metric

    from dynecon.experiment import ParameterSweep

    model = models.load("consumption_savings", n_a=40, n_z=3).compile()
    res = ParameterSweep(model, {"beta": [0.94, 0.92, 0.96]}, mean_consumption_metric).run()
    ax = plot_sweep(res, "beta", "mean_c")
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.92, 0.94, 0.96])
    assert ax.get_xlabel() == "beta" and ax.get_ylabel() == "mean_c"
    with pytest.raises(ValidationError):
        plot_sweep(res, "nope", "mean_c")
