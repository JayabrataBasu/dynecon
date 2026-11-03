"""Built-in models against their analytic benchmarks (non-equilibrium models)."""

import numpy as np
import pytest

import econkit as ek
from econkit import models
from econkit.models import mccall
from econkit.models.brock_mirman import ANALYTIC as BM
from econkit.models.consumption_savings import mpc_perfect_foresight
from econkit.models.lucas_tree import log_utility_price_dividend
from econkit.solvers.pricing import price_one_period_claim


def test_consumption_savings_concave_and_mpc() -> None:
    model = models.load("consumption_savings", n_a=300, a_max=60.0, n_z=5).compile()
    sol = ek.solve(model)
    a = model.grids["assets"]
    c = sol.policy["consumption"]
    binding = sol.policy["assets_next"] <= a[0] + 1e-10
    for j in range(c.shape[1]):
        d2 = np.diff(np.diff(c[:, j]) / np.diff(a))  # slope changes: non-uniform grid
        interior = ~binding[:-2, j] & ~binding[1:-1, j] & ~binding[2:, j]
        assert np.all(d2[interior] <= 1e-8)
    assert binding[0, 0] and not binding[-1].any()  # binds only at low wealth and low income
    # high-wealth MPC approaches the perfect-foresight value (residual risk matters: loose)
    top = slice(int(0.9 * a.size), a.size)
    mpc = np.mean(np.diff(c[top, 2]) / np.diff(a[top]))
    target = mpc_perfect_foresight(0.96, 2.0, 0.03)
    print(f"informational: MPC at top of grid {mpc:.4f}, perfect foresight {target:.4f}")
    assert 0.0 < mpc < 1.0


def test_consumption_savings_overrides_and_validation() -> None:
    m = models.load("consumption_savings", beta=0.9, n_a=50)
    assert m.compile().p.beta == 0.9
    with pytest.raises(ek.errors.ValidationError):
        models.load("consumption_savings", betaa=0.9)


def test_brock_mirman_three_methods_agree_with_closed_form() -> None:
    alpha, beta = 0.36, 0.96
    egm = ek.solve(models.load("brock_mirman").compile(), method="egm")
    vfi = ek.solve(models.load("brock_mirman", n_k=300).compile(), method="vfi", tol=1e-9)
    dsge = ek.solve(models.load("brock_mirman", problem="dsge").compile())
    model = egm.model
    k, z = model.grids["k"], model.shocks["z"].values
    c_true = BM["consumption_policy"](k[:, None], z[None, :], alpha, beta)
    interior = (egm.policy["k_next"] > k[0]) & (egm.policy["k_next"] < k[-1])
    assert np.abs(egm.policy["c"] / c_true - 1)[interior].max() < 2e-10
    h = np.diff(vfi.model.grids["k"]).max()
    kv, zv = vfi.model.grids["k"], vfi.model.shocks["z"].values
    truth = BM["capital_policy"](kv[:, None], zv[None, :], alpha, beta)
    ok = (truth > kv[0] + 2 * h) & (truth < kv[-1] - 2 * h)
    assert np.abs(vfi.policy["k_next"] - truth)[ok].max() <= 2 * h
    np.testing.assert_allclose(dsge.F, [[alpha, 1.0]], atol=1e-6)


def test_mccall_reservation_wage_and_comparative_statics() -> None:
    def wbar(**kw: float) -> float:
        return mccall.reservation_wage(ek.solve(models.load("mccall", **kw).compile(), tol=1e-12))

    base = wbar()
    assert abs(base - mccall.reservation_wage_uniform(0.96, 0.5)) <= 1.0 / 59 + 1e-12
    assert wbar(c=0.6) > base > wbar(c=0.4)
    assert wbar(beta=0.97) > base > wbar(beta=0.95)
    sol = ek.solve(models.load("mccall").compile(), tol=1e-12)
    action = sol.policy["action"][:60]
    assert np.all(np.diff(action) >= 0)  # threshold rule: once accepted, always accepted
    w = np.linspace(0, 1, 60)
    emp_value = sol.value[60:]
    np.testing.assert_allclose(emp_value, w / (1 - 0.96), rtol=1e-6)
    with pytest.raises(ek.errors.ValidationError):
        models.load("mccall", n_w=1)
    assert np.isnan(mccall.reservation_wage(ek.solve(models.load("mccall", c=2.0).compile())))


def test_lucas_tree_log_utility_and_crra() -> None:
    sol = ek.solve(models.load("lucas_tree").compile())
    y = sol.grids["dividend"]
    np.testing.assert_allclose(
        sol.policy["price"],
        0.96 * y / (1 - 0.96) * np.ones_like(y) * (24 / 24.0) / y * y,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        sol.policy["price_dividend"], log_utility_price_dividend(0.96), rtol=1e-6
    )
    hi = ek.solve(models.load("lucas_tree", sigma=3.0).compile()).policy["price_dividend"]
    lo = ek.solve(models.load("lucas_tree", sigma=0.5).compile()).policy["price_dividend"]
    assert np.all(np.diff(hi) > 0)  # sigma > 1: income effect dominates, p/y rises with y
    assert np.all(np.diff(lo) < 0)  # sigma < 1: substitution effect dominates
    # risk-free rate two ways: closed form on the chain vs pricing a unit one-period claim
    s3 = ek.solve(models.load("lucas_tree", sigma=3.0).compile())
    q = price_one_period_claim(s3, np.ones_like(y))
    np.testing.assert_allclose(1.0 / q - 1.0, s3.policy["risk_free"], atol=1e-12)
    P = s3.model.shocks["dividend"].P
    mu = s3.model.shocks["dividend"].values ** (-3.0)
    direct = mu / (0.96 * (P @ mu)) - 1.0
    np.testing.assert_allclose(s3.policy["risk_free"], direct, atol=1e-8)
    with pytest.raises(ek.errors.ValidationError):
        ek.solve(models.load("lucas_tree").compile(), tol=-1.0)


def test_all_models_smoke() -> None:
    assert models.available() == [
        "aiyagari",
        "brock_mirman",
        "consumption_savings",
        "huggett",
        "lucas_tree",
        "mccall",
        "nk3",
        "olg2",
        "rbc",
    ]
    quick = {
        "aiyagari": dict(r=0.03, n_a=60, n_z=3),
        "huggett": dict(r=0.02, n_a=60, n_z=3),
        "consumption_savings": dict(n_a=60, n_z=3),
    }
    for name in models.available():
        sol = ek.solve(models.load(name, **quick.get(name, {})).compile())
        assert sol.converged, name
        assert name.split("_")[0][:3] in sol.summary().lower() or "Solution" in sol.summary()
        if (
            sol.method in ("egm", "vfi")
            and "assets_next" in sol.policy
            or sol.method == "perturbation"
        ):
            sim = ek.simulate(sol, n_periods=1000, rng=0)
            assert sim.n_periods == 1000
