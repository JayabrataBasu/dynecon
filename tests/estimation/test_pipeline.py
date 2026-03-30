"""Tests for the end-to-end estimation pipeline helper."""

from econkit.estimation import run_solve_simulate_moments_pipeline


def test_run_solve_simulate_moments_pipeline_returns_expected_keys() -> None:
    report = run_solve_simulate_moments_pipeline(periods=120, random_seed=7)

    assert "objective" in report
    assert "rmse" in report
    assert "n_moments" in report
    assert "converged" in report
    assert "iterations" in report
    assert report["iterations"] >= 1.0
