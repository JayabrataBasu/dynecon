"""Convergence diagnostics tests.

Economics rationale:
Residual paths determine whether an economic fixed-point solve is trustworthy,
so summaries and reports should be correct and easy to inspect.
"""

import math
import json

import pytest

from econkit.solvers import (
    build_convergence_artifact,
    convergence_artifact_to_json,
    format_convergence_report,
    summarize_residual_history,
)


def test_summarize_residual_history_basic_metrics() -> None:
    summary = summarize_residual_history([1.0, 0.1, 0.01])

    assert summary.count == 3
    assert abs(summary.first - 1.0) < 1e-12
    assert abs(summary.last - 0.01) < 1e-12
    assert summary.monotone_nonincreasing
    assert abs(summary.avg_log10_drop_per_step - 1.0) < 1e-12


def test_format_convergence_report_contains_key_fields() -> None:
    report = format_convergence_report((1.0, 0.5, 0.25))

    assert "Residual convergence report" in report
    assert "steps=3" in report
    assert "monotone_nonincreasing=True" in report


def test_summarize_residual_history_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        summarize_residual_history([])
    with pytest.raises(ValueError):
        summarize_residual_history([1.0, -0.1])


def test_log_drop_with_single_point_is_zero() -> None:
    summary = summarize_residual_history([math.pi])
    assert abs(summary.avg_log10_drop_per_step) < 1e-12


def test_build_convergence_artifact_is_json_serializable() -> None:
    artifact = build_convergence_artifact([1.0, 0.5, 0.25], metadata={"model": "toy"})

    assert artifact["metadata"]["model"] == "toy"
    assert artifact["summary"]["count"] == 3
    assert artifact["history"] == [1.0, 0.5, 0.25]


def test_convergence_artifact_to_json_round_trip() -> None:
    text = convergence_artifact_to_json([1.0, 0.1], metadata={"session": 9})
    loaded = json.loads(text)

    assert loaded["metadata"]["session"] == 9
    assert loaded["summary"]["last"] == 0.1
