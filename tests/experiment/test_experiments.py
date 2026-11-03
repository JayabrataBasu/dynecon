"""Sweeps, elasticities, experiments, policy experiments, welfare and resampling."""

import json

import numpy as np
import pytest
from builders import failing_metric, mean_consumption_metric

import dynecon as ek
from dynecon import models
from dynecon.core.errors import DyneconError, ValidationError
from dynecon.experiment import (
    Experiment,
    ParameterSweep,
    PolicyExperiment,
    bootstrap_ci,
    cev,
    cev_from_values,
    elasticities,
    evaluate_policy_value,
    monte_carlo,
)
from dynecon.model import State
from dynecon.utils.serialization import decode, encode, numeric_leaves

BUILD = ("consumption_savings", {"n_a": 50, "n_z": 3, "a_max": 30.0})


# ---------------------------------------------------------------- sweeps


def test_sweep_order_dataframe_best_and_process_pool_equivalence() -> None:
    grid = {"beta": [0.92, 0.95], "r": [0.02, 0.03, 0.04]}
    serial = ParameterSweep(BUILD, grid, mean_consumption_metric, n_jobs=1).run()
    parallel = ParameterSweep(BUILD, grid, mean_consumption_metric, n_jobs=2).run()
    assert [p["beta"] for p in serial.points] == [0.92, 0.92, 0.92, 0.95, 0.95, 0.95]
    assert serial.status == ["ok"] * 6
    np.testing.assert_array_equal(serial.column("mean_c"), parallel.column("mean_c"))
    df = serial.to_dataframe()
    assert list(df.columns)[:3] == ["beta", "r", "mean_c"] and len(df) == 6
    best = serial.best("mean_c")
    assert best == serial.points[int(np.argmax(serial.column("mean_c")))]
    assert (
        serial.best("mean_c", maximize=False)
        == serial.points[int(np.argmin(serial.column("mean_c")))]
    )


def test_sweep_records_failures_and_fail_fast() -> None:
    model = models.load(*BUILD[:1], **BUILD[1]).compile()
    res = ParameterSweep(model, [{"beta": 0.95}, {"beta": 0.96}], failing_metric).run()
    assert res.status == ["failed", "failed"] and "boom" in res.messages[0]
    assert np.isnan(res.column("anything")).all()
    with pytest.raises(ValidationError):
        res.best("x")
    with pytest.raises(DyneconError, match="sweep failed"):
        ParameterSweep(model, [{"beta": 0.95}], failing_metric, fail_fast=True).run()
    bad = ParameterSweep(model, [{"beta": 0.2}], mean_consumption_metric).run()  # outside bounds
    assert bad.status == ["failed"]


def test_sweep_validation() -> None:
    model = models.load(BUILD[0], **BUILD[1]).compile()
    with pytest.raises(ValidationError, match="empty"):
        ParameterSweep(model, [], mean_consumption_metric).run()
    with pytest.raises(ValidationError, match="build spec"):
        ParameterSweep(model, {"beta": [0.9, 0.95]}, mean_consumption_metric, n_jobs=2).run()


@pytest.mark.jax
def test_batched_jax_sweep_matches_loop() -> None:
    pytest.importorskip("jax")
    model = models.load(BUILD[0], **BUILD[1]).compile()
    grid = {"beta": [0.92, 0.94, 0.96]}
    loop = ParameterSweep(model, grid, mean_consumption_metric, method="egm").run()
    batch = ParameterSweep(model, grid, mean_consumption_metric, method="egm", backend="jax").run()
    np.testing.assert_allclose(loop.column("mean_c"), batch.column("mean_c"), atol=1e-8)


def test_elasticity_of_steady_state_capital_wrt_beta() -> None:
    alpha = 0.36
    out = elasticities(
        ("brock_mirman", {"problem": "dsge"}),
        ["beta"],
        lambda sol: sol.steady_state[0],
        rel_change=0.01,
    )
    assert abs(out["beta"] - 1.0 / (1.0 - alpha)) < 1e-4
    with pytest.raises(ValidationError):
        elasticities(("brock_mirman", {"problem": "dsge"}), ["nope"], lambda s: 1.0)


# ---------------------------------------------------------------- persistence and experiments


def test_solution_dict_save_load_roundtrip(tmp_path) -> None:
    sol = ek.solve(models.load(BUILD[0], **BUILD[1]).compile(), method="vfi")
    d = sol.to_dict()
    json.dumps(d)  # JSON-safe
    sol.save(tmp_path / "s")
    back = ek.Solution.load(tmp_path / "s", model=sol.model)
    np.testing.assert_array_equal(back.value, sol.value)
    np.testing.assert_array_equal(back.policy["assets_next_index"], sol.policy["assets_next_index"])
    assert back.method == "vfi" and back.params == sol.params
    assert float(back.policy_function("consumption")(1.0, 0)) == float(
        sol.policy_function("consumption")(1.0, 0)
    )


def test_serialization_encodes_special_floats() -> None:
    enc = encode({"a": np.array([1.0, np.nan, np.inf, -np.inf]), "b": np.float64(2.5), "c": (1, 2)})
    json.dumps(enc)
    dec = decode(enc)
    assert np.isnan(dec["a"][1]) and dec["a"][2] == np.inf and dec["a"][3] == -np.inf
    assert numeric_leaves({"x": [1, {"y": 2.0}], "z": True, "s": "t"}) == {
        "x[0]": 1.0,
        "x[1].y": 2.0,
        "z": 1.0,
    }


def test_experiment_run_and_reproduce(tmp_path) -> None:
    exp = Experiment(
        "cs_base",
        BUILD,
        solve_options={"method": "egm"},
        simulate_options={"n_periods": 80, "n_agents": 20, "burn_in": 10},
        moments=["mean_assets", "std_consumption"],
        outdir=tmp_path,
    )
    res = exp.run(rng=123)
    files = {p.name for p in res.path.iterdir()}
    assert {
        "config.json",
        "provenance.json",
        "result.json",
        "log.txt",
        "solution.npz",
        "solution.json",
    } <= files
    prov = json.loads((res.path / "provenance.json").read_text())
    assert prov["model_fingerprint"] == res.solution.metadata["fingerprint"] and prov["numpy"]
    report = Experiment.reproduce(res.path)
    assert report.passed and report.n_compared > 5 and report.differences == []
    cfg_path = res.path / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["build"][1]["n_a"] = 60  # alter the model
    cfg_path.write_text(json.dumps(cfg))
    bad = Experiment.reproduce(res.path)
    assert not bad.passed and "model_fingerprint" in bad.differences
    assert any(d.startswith("solution.policy") for d in bad.differences)


def test_experiment_validation(tmp_path) -> None:
    with pytest.raises(ValidationError, match="moments need"):
        Experiment("x", BUILD, moments=["mean_assets"], outdir=tmp_path).run()
    with pytest.raises(ValidationError):
        Experiment("x", "no_such_model", outdir=tmp_path).run()
    with pytest.raises(ValidationError):
        Experiment("x", 42, outdir=tmp_path).run()  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Experiment("x", "nonexistent_module:f", outdir=tmp_path).run()


# ---------------------------------------------------------------- welfare and policy


@pytest.mark.parametrize("sigma", [1.0, 2.0, 0.5])
def test_cev_exact_scaling(sigma: float) -> None:
    beta, c = 0.96, 1.7

    def v(cons: float) -> float:
        u = np.log(cons) if sigma == 1.0 else cons ** (1 - sigma) / (1 - sigma)
        return float(u / (1 - beta))

    assert abs(float(cev_from_values(v(c), v(1.05 * c), sigma, beta)) - 0.05) < 1e-10
    assert float(cev_from_values(v(c), v(c), sigma, beta)) == pytest.approx(0.0, abs=1e-14)
    assert float(cev_from_values(v(c), v(1.2 * c), sigma, beta)) > 0
    with pytest.raises(ValidationError):
        cev_from_values(-1.0, 1.0, 2.0, beta)


def test_policy_value_reproduces_vfi_value() -> None:
    model = models.load(BUILD[0], **BUILD[1]).compile()
    vfi = ek.solve(model, method="vfi", tol=1e-11)
    np.testing.assert_allclose(evaluate_policy_value(vfi), vfi.value, atol=1e-6)
    egm = ek.solve(model, method="egm")
    assert evaluate_policy_value(egm).shape == egm.policy["consumption"].shape
    with pytest.raises(ValidationError):
        evaluate_policy_value(ek.solve(models.load("nk3").compile()))


def test_policy_experiment_dict_and_function_reforms() -> None:
    base = models.load(BUILD[0], **BUILD[1])
    higher_r = PolicyExperiment(base, {"r": 0.04}).run()
    assert higher_r.welfare["cev"] > 0  # budget set expands: no borrowing, higher return
    assert (
        higher_r.aggregates["reform"]["mean_assets"]
        != higher_r.aggregates["baseline"]["mean_assets"]
    )
    assert "cev_state" in higher_r.welfare and np.all(higher_r.welfare["cev_state"] > -1e-9)

    def allow_borrowing(m):
        m = m.copy("reform")
        s = m.get_state("assets")
        m.replace_state(
            State(
                "assets",
                bounds=(-2.0, s.bounds[1]),
                grid_type="exp",
                n_points=s.n_points,
                grid_curvature=s.grid_curvature,
            )
        )
        return m

    loosened = PolicyExperiment(base, allow_borrowing).run()
    assert loosened.welfare["cev"] > 0  # relaxing a constraint cannot hurt
    assert "cev_state" not in loosened.welfare  # grids differ: aggregate only
    assert (
        loosened.aggregates["reform"]["borrowing_constrained_share"]
        != loosened.aggregates["baseline"]["borrowing_constrained_share"]
    )
    zero = PolicyExperiment(base, {"r": 0.03}).run()
    assert abs(zero.welfare["cev"]) < 1e-12
    with pytest.raises(ValidationError):
        PolicyExperiment(base.compile(), allow_borrowing).run()
    with pytest.raises(ValidationError):
        PolicyExperiment(base, {"r": 0.04}, welfare="other").run()
    with pytest.raises(ValidationError):
        base.get_state("nope")


def test_cev_needs_crra_and_models() -> None:
    sol = ek.solve(models.load("brock_mirman", n_k=40, n_z=3).compile(), method="egm")
    with pytest.raises(ValidationError, match="sigma"):
        cev(sol, sol)
    sol_no_model = ek.Solution.load.__func__  # noqa: F841 - keep reference to silence unused
    with pytest.raises(ValidationError):
        cev(
            ek.solve(models.load(BUILD[0], **BUILD[1]).compile()).__class__(
                **{
                    **{f: None for f in ()},
                    "method": "x",
                    "converged": True,
                    "iterations": 0,
                    "residual": 0.0,
                    "elapsed": 0.0,
                    "backend": "numpy",
                    "model_name": None,
                    "params": {},
                    "grids": {},
                    "policy": {},
                    "value": None,
                }
            ),
            sol,
        )


# ---------------------------------------------------------------- Monte Carlo and bootstrap


def test_monte_carlo_reproducible_and_independent() -> None:
    a = monte_carlo(lambda g: g.standard_normal(), 50, rng=3)
    b = monte_carlo(lambda g: g.standard_normal(), 50, rng=3)
    np.testing.assert_array_equal(a, b)
    assert abs(np.corrcoef(a[:-1], a[1:])[0, 1]) < 0.4
    with pytest.raises(ValidationError):
        monte_carlo(lambda g: 0, 0)


def test_bootstrap_ci_coverage_and_block_effect() -> None:
    rng = np.random.default_rng(0)
    hits = 0
    reps = 400
    for _ in range(reps):
        x = rng.standard_normal(50)
        lo, hi = bootstrap_ci(x, np.mean, n_boot=400, rng=rng)
        hits += lo <= 0.0 <= hi
    assert 0.92 <= hits / reps <= 0.98
    # AR(1) with rho = 0.7: block bootstrap s.e. exceeds the iid one
    n = 1500
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + e[t]
    iid_lo, iid_hi = bootstrap_ci(x, np.mean, n_boot=600, rng=1)
    blk_lo, blk_hi = bootstrap_ci(x, np.mean, n_boot=600, block_len=40, rng=1)
    assert (blk_hi - blk_lo) > 1.5 * (iid_hi - iid_lo)
    with pytest.raises(ValidationError):
        bootstrap_ci(x, np.mean, alpha=1.5)
