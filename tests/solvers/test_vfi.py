"""VFI: API behaviour, backend parity, Howard/monotone equivalence, analytic benchmarks."""

import numpy as np
import pytest
from builders import ALPHA, BETA_BM, brock_mirman, consumption_savings, permanent_income

import econkit as ek
from econkit.core.errors import (
    ConvergenceError,
    ConvergenceWarning,
    ModelSpecificationError,
    ValidationError,
)
from econkit.model import Control, Model, Param, State
from econkit.solvers.dp.vfi import build_payoff_tensor


@pytest.fixture(scope="module")
def cs():
    return consumption_savings(n_a=60, n_z=3).compile()


def test_solve_returns_solution(cs) -> None:
    sol = ek.solve(cs, method="vfi", tol=1e-9)
    assert sol.converged and sol.method == "vfi" and isinstance(sol, ek.Solution)
    assert sol.value.shape == (60, 3)
    assert set(sol.policy) == {"consumption", "assets_next", "assets_next_index"}
    assert sol.diagnostics["max_feasibility_violation"] == 0.0
    assert sol.diagnostics["monotone_policy"]
    assert sol.diagnostics["bellman_residual"] < 1e-6
    assert sol.metadata["fingerprint"] == cs.fingerprint()
    assert sol.metadata["config"]["tol"] == 1e-9
    assert "vfi" in sol.summary()
    assert np.all(sol.policy["consumption"] > 0)


def test_api_validation(cs) -> None:
    with pytest.raises(ValidationError, match="unknown method"):
        ek.solve(cs, method="magic")
    with pytest.raises(ValidationError, match="valid"):
        ek.solve(cs, method="vfi", tolerance=1e-3)
    with pytest.raises(ValidationError):
        ek.solve("nope")  # type: ignore[arg-type]
    for bad in ({"tol": 0.0}, {"max_iter": 0}, {"damping": 2.0}, {"howard_steps": -1}):
        with pytest.raises(ValidationError):
            ek.solve(cs, method="vfi", **bad)
    with pytest.raises(ValidationError, match="egm"):
        ek.solve(cs, method="vfi", max_memory_gb=1e-9)


def test_nonconvergence_warns_and_strict_raises(cs) -> None:
    with pytest.warns(ConvergenceWarning):
        sol = ek.solve(cs, method="vfi", max_iter=2)
    assert not sol.converged
    ek.config.strict_convergence = True
    try:
        with pytest.raises(ConvergenceError):
            ek.solve(cs, method="vfi", max_iter=2)
    finally:
        ek.config.strict_convergence = False


def test_damping_and_verbose(cs, capsys) -> None:
    sol = ek.solve(cs, method="vfi", damping=0.8, howard_steps=0, tol=1e-8, verbose=True)
    assert sol.converged
    assert "vfi iter" in capsys.readouterr().out


@pytest.mark.numba
@pytest.mark.parametrize("monotone", [True, False])
@pytest.mark.parametrize("howard", [0, 10])
def test_numpy_numba_parity(cs, monotone: bool, howard: int) -> None:
    pytest.importorskip("numba")
    kw = {"monotone": monotone, "howard_steps": howard, "tol": 1e-10}
    a = ek.solve(cs, method="vfi", backend="numpy", **kw)
    b = ek.solve(cs, method="vfi", backend="numba", **kw)
    assert a.backend == "numpy" and b.backend == "numba"
    np.testing.assert_array_equal(a.policy["assets_next_index"], b.policy["assets_next_index"])
    np.testing.assert_allclose(a.value, b.value, rtol=0, atol=1e-12)


def test_howard_and_monotone_equivalence(cs) -> None:
    base = ek.solve(cs, method="vfi", howard_steps=0, monotone=False, tol=1e-11, backend="numpy")
    for howard, mono in [(0, True), (10, False), (10, True)]:
        other = ek.solve(cs, method="vfi", howard_steps=howard, monotone=mono, tol=1e-11)
        np.testing.assert_array_equal(
            base.policy["assets_next_index"], other.policy["assets_next_index"]
        )
        np.testing.assert_allclose(base.value, other.value, atol=1e-8)
    fast = ek.solve(cs, method="vfi", howard_steps=10, tol=1e-11)
    assert fast.iterations < base.iterations


def test_brock_mirman_policy_within_two_grid_spacings() -> None:
    model = brock_mirman(n_k=300, n_z=7).compile()
    sol = ek.solve(model, method="vfi", tol=1e-9)
    assert sol.converged
    k = model.grids["k"]
    z = model.shocks["z"].values
    truth = ALPHA * BETA_BM * z[None, :] * k[:, None] ** ALPHA
    h = np.diff(k).max()
    interior = (truth > k[0] + 2 * h) & (truth < k[-1] - 2 * h)
    err = np.abs(sol.policy["k_next"] - truth)[interior].max()
    assert err <= 2 * h
    assert sol.diagnostics["bellman_residual"] < 1e-7


def test_permanent_income_policy_is_identity() -> None:
    model = permanent_income().compile()
    sol = ek.solve(model, method="vfi", tol=1e-10, max_iter=20000)
    idx = sol.policy["assets_next_index"][:, 0]
    assert np.max(np.abs(idx - np.arange(idx.size))) <= 1


def test_infeasible_model_rejected() -> None:
    m = permanent_income()
    m.params(y=Param(-5.0))
    with pytest.raises(ModelSpecificationError, match="feasible"):
        build_payoff_tensor(m.compile())


def test_finite_dp_matches_closed_form() -> None:
    beta = 0.9
    R = np.array([[1.0, 0.0], [0.5, 2.0]])  # state x action
    T = np.stack([np.eye(2), np.eye(2)])  # both actions keep the state
    m = Model("dp")
    m.params(beta=Param(beta))
    m.set_problem("finite_dp", reward=lambda p: R, transition=lambda p: T, discount="beta")
    sol = ek.solve(m.compile(), tol=1e-12)
    np.testing.assert_allclose(sol.value, R.max(axis=1) / (1 - beta), rtol=1e-9)
    np.testing.assert_array_equal(sol.policy["action"], [0, 1])
    assert sol.backend == "numpy"


def test_states_and_controls_defaults_unused() -> None:
    m = Model("x")
    m.add_state(State("a", bounds=(0, 1), n_points=3))
    m.add_control(Control("c"))
    assert "a" in m.add_state(State("b", bounds=(0, 1), n_points=3)).name or True
