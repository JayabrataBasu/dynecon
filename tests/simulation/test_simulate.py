"""simulate(): shapes, determinism, backend parity, validation, result accessors."""

import numpy as np
import pytest
from builders import consumption_savings, permanent_income

import econkit as ek
from econkit.core.errors import ValidationError
from econkit.model import Model, Param


@pytest.fixture(scope="module")
def sol():
    return ek.solve(consumption_savings(n_a=80, n_z=5).compile())


def test_shapes_bounds_and_store(sol) -> None:
    sim = ek.simulate(sol, n_periods=50, n_agents=7, burn_in=20, rng=3)
    assert sim.paths["assets"].shape == (50, 7)
    grid = sol.grids["assets"]
    assert sim.paths["assets"].min() >= grid[0] and sim.paths["assets"].max() <= grid[-1]
    assert np.all(sim.paths["consumption"] > 0)
    only = ek.simulate(sol, n_periods=10, n_agents=2, rng=3, store=["assets"])
    assert list(only.paths) == ["assets"]
    with pytest.raises(ValidationError):
        ek.simulate(sol, n_periods=5, store=["nope"])


def test_same_seed_same_paths_and_asset_dynamics_consistent(sol) -> None:
    a = ek.simulate(sol, n_periods=60, n_agents=5, rng=11)
    b = ek.simulate(sol, n_periods=60, n_agents=5, rng=11)
    c = ek.simulate(sol, n_periods=60, n_agents=5, rng=12)
    np.testing.assert_array_equal(a.paths["assets"], b.paths["assets"])
    assert not np.array_equal(a.paths["assets"], c.paths["assets"])
    assert a.seed_entropy == b.seed_entropy
    # budget identity: a' = resources - c (to interpolation precision)
    prob = sol.model.problem
    z = a.paths["income"][:-1]
    m = prob.resources(a.paths["assets"][:-1], z, sol.model.p)
    np.testing.assert_allclose(a.paths["assets"][1:], m - a.paths["consumption"][:-1], atol=2e-3)


@pytest.mark.numba
def test_backend_parity_and_thread_independence(sol) -> None:
    numba = pytest.importorskip("numba")
    kw = {"n_periods": 200, "n_agents": 64, "rng": 5}
    ref = ek.simulate(sol, backend="numpy", **kw)
    out = ek.simulate(sol, backend="numba", **kw)
    np.testing.assert_array_equal(ref.paths["shock_index"], out.paths["shock_index"])
    np.testing.assert_allclose(ref.paths["assets"], out.paths["assets"], rtol=0, atol=1e-12)
    prev = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = ek.simulate(sol, backend="numba", **kw)
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        many = ek.simulate(sol, backend="numba", **kw)
    finally:
        numba.set_num_threads(prev)
    np.testing.assert_array_equal(one.paths["assets"], many.paths["assets"])


def test_shocks_override_and_validation(sol) -> None:
    idx = np.zeros((10, 3), dtype=np.int64)
    sim = ek.simulate(sol, n_periods=10, n_agents=3, shocks=idx)
    assert np.all(sim.paths["shock_index"] == 0)
    with pytest.raises(ValidationError):
        ek.simulate(sol, n_periods=10, n_agents=3, shocks=np.zeros((9, 3), dtype=np.int64))
    with pytest.raises(ValidationError):
        ek.simulate(sol, n_periods=10, n_agents=3, shocks=np.full((10, 3), 99, dtype=np.int64))
    with pytest.raises(ValidationError):
        ek.simulate(sol, n_periods=0)


def test_init_scalar_and_vector(sol) -> None:
    s1 = ek.simulate(sol, n_periods=2, n_agents=4, init=1.0, rng=0)
    np.testing.assert_allclose(s1.paths["assets"][0], 1.0)
    s2 = ek.simulate(sol, n_periods=2, n_agents=4, init=[0.5, 1.0, 1.5, 2.0], rng=0)
    np.testing.assert_allclose(s2.paths["assets"][0], [0.5, 1.0, 1.5, 2.0])


def test_result_accessors(sol) -> None:
    sim = ek.simulate(sol, n_periods=300, n_agents=20, rng=2)
    m = sim.moments(
        ["mean_assets", "std_consumption", "var_assets", "autocorr_income", "p50_assets"]
    )
    assert abs(m["mean_assets"] - sim.mean("assets")) < 1e-12
    assert abs(m["var_assets"] - np.var(sim.paths["assets"])) < 1e-9
    assert 0.5 < m["autocorr_income"] < 1.0
    with pytest.raises(ValidationError):
        sim.moments(["bogus"])
    with pytest.raises(ValidationError):
        sim.mean("nothing")
    assert "SimulationResult" in sim.summary()
    d = sim.to_dict()
    assert d["n_agents"] == 20 and len(d["paths"]["assets"]) == 300
    df = sim.to_dataframe()
    assert len(df) == 300 * 20 and {"period", "agent", "assets"} <= set(df.columns)


def test_deterministic_model_and_non_saving_rejected() -> None:
    sol = ek.solve(permanent_income(n_a=41, a_max=4.0).compile())
    sim = ek.simulate(sol, n_periods=20, n_agents=2, init=1.0)
    np.testing.assert_allclose(sim.paths["assets"], 1.0, atol=1e-6)  # a' = a
    m = Model("dp")
    m.params(beta=Param(0.9))
    m.set_problem(
        "finite_dp",
        reward=lambda p: np.ones((2, 2)),
        transition=lambda p: np.full((2, 2, 2), 0.5),
        discount="beta",
    )
    with pytest.raises(ValidationError, match="saving"):
        ek.simulate(ek.solve(m.compile()), n_periods=3)


def test_policy_function_interpolates(sol) -> None:
    f = sol.policy_function("consumption")
    a = sol.grids["assets"]
    np.testing.assert_allclose(f(a, 2), sol.policy["consumption"][:, 2], atol=1e-12)
    mid = 0.5 * (a[3] + a[4])
    expected = 0.5 * (sol.policy["consumption"][3, 1] + sol.policy["consumption"][4, 1])
    assert abs(float(f(mid, 1)) - expected) < 1e-12
    with pytest.raises(ValueError):
        sol.policy_function("nope")
