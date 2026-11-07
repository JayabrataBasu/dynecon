"""EGM: analytic benchmarks, cross-method consistency, Euler errors, wiring."""

import numpy as np
import pytest
from builders import ALPHA, BETA_BM, brock_mirman, consumption_savings, permanent_income

import dynecon as ek
from dynecon.core.errors import ModelSpecificationError, ValidationError
from dynecon.model import Control, Model, Param, State
from dynecon.solvers.diagnostics import euler_errors


def test_brock_mirman_consumption_matches_closed_form() -> None:
    model = brock_mirman(n_k=200, n_z=7).compile()
    sol = ek.solve(model, method="egm")
    assert sol.converged and sol.method == "egm" and sol.value is None
    k, z = model.grids["k"], model.shocks["z"].values
    truth = (1 - ALPHA * BETA_BM) * z[None, :] * k[:, None] ** ALPHA
    interior = (sol.policy["k_next"] > k[0]) & (sol.policy["k_next"] < k[-1])
    rel = np.abs(sol.policy["c"] / truth - 1)[interior].max()
    assert rel < 2e-10  # measured 6.8e-11: log utility makes c linear in cash on hand, so exact
    assert sol.diagnostics["euler_error_max"] < -2.0


def test_permanent_income_policy_is_identity() -> None:
    model = permanent_income().compile()
    sol = ek.solve(model, method="egm")
    a = model.grids["assets"]
    assert sol.converged
    assert np.max(np.abs(sol.policy["assets_next"][:, 0] - a)) < 1e-6


def test_vfi_egm_consistency_and_euler_errors() -> None:
    model = consumption_savings(n_a=300, n_z=7).compile()
    egm = ek.solve(model, method="egm")
    vfi = ek.solve(model, method="vfi", tol=1e-9)
    a = model.grids["assets"]
    h = np.diff(a).max()
    interior = (egm.policy["assets_next"] > a[0] + 1e-9) & (egm.policy["assets_next"] < 0.5 * a[-1])
    gap = np.abs(egm.policy["assets_next"] - vfi.policy["assets_next"])[interior].max()
    assert gap <= 2 * h
    assert egm.diagnostics["euler_error_max"] <= vfi.diagnostics["euler_error_max"] + 0.5
    assert euler_errors(egm).size > 0
    assert egm.diagnostics["n_binding"] > 0  # borrowing constraint binds somewhere


@pytest.mark.numba
def test_numpy_numba_agree() -> None:
    pytest.importorskip("numba")
    model = consumption_savings(n_a=80, n_z=3).compile()
    a = ek.solve(model, method="egm", backend="numpy")
    b = ek.solve(model, method="egm", backend="numba")
    assert a.backend == "numpy" and b.backend == "numba"
    np.testing.assert_allclose(a.policy["consumption"], b.policy["consumption"], rtol=0, atol=1e-12)


def test_auto_prefers_egm_and_falls_back_to_vfi() -> None:
    model = consumption_savings(n_a=40, n_z=3).compile()
    assert ek.solve(model).method == "egm"
    m = consumption_savings(n_a=40, n_z=3)
    p = m._problem
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=p.utility,
        resources=p.resources,
    )
    sol = ek.solve(m.compile())
    assert sol.method == "vfi" and sol.metadata["requested_method"] == "auto"
    with pytest.raises(ValidationError, match="marginal"):
        ek.solve(m.compile(), method="egm")
    with pytest.raises(ValidationError):
        euler_errors(sol)


def test_nonmonotone_resources_rejected() -> None:
    m = consumption_savings(n_a=30, n_z=3)
    p = m._problem
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=p.utility,
        marginal_utility=p.marginal_utility,
        inverse_marginal_utility=p.inverse_marginal_utility,
        marginal_return=p.marginal_return,
        resources=lambda a, z, q: z - (a - 5.0) ** 2,
    )
    with pytest.raises(ModelSpecificationError, match="increasing"):
        ek.solve(m.compile(), method="egm")


def test_egm_option_validation_and_finite_dp_rejected() -> None:
    model = consumption_savings(n_a=30, n_z=3).compile()
    for bad in ({"tol": -1.0}, {"max_iter": 0}, {"damping": 0.0}):
        with pytest.raises(ValidationError):
            ek.solve(model, method="egm", **bad)
    with pytest.raises(ValidationError, match="unknown option"):
        ek.solve(model, method="egm", howard_steps=3)
    d = Model("dp")
    d.params(beta=Param(0.9))
    d.set_problem(
        "finite_dp",
        reward=lambda p: np.ones((2, 2)),
        transition=lambda p: np.full((2, 2, 2), 0.5),
        discount="beta",
    )
    with pytest.raises(ValidationError, match="saving"):
        ek.solve(d.compile(), method="egm")
    assert State("x").name == "x" and Control("c").name == "c"


def test_nonconcave_endogenous_grid_rejected() -> None:
    m = consumption_savings(n_a=30, n_z=3)
    p = m._problem
    # a return that rises steeply with assets breaks concavity: the endogenous grid folds back
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=p.utility,
        resources=p.resources,
        marginal_utility=p.marginal_utility,
        inverse_marginal_utility=p.inverse_marginal_utility,
        marginal_return=lambda a, z, q: np.exp(3.0 * a),
    )
    with pytest.raises(ModelSpecificationError, match="monotone"):
        ek.solve(m.compile(), method="egm")


def test_unbounded_saving_warns() -> None:
    from dynecon.core.errors import GridWarning

    with pytest.warns(GridWarning, match="without bound"):
        ek.solve(consumption_savings(beta=0.99, r=0.1, n_a=40, n_z=3).compile(), method="egm")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", GridWarning)  # beta (1 + r) = 0.9888 < 1: no warning
        ek.solve(consumption_savings(beta=0.96, r=0.03, n_a=40, n_z=3).compile(), method="egm")
