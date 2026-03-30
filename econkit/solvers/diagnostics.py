"""Solver diagnostics utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

NormName = Literal["sup", "l2"]
SUPPORTED_CONVERGENCE_SCHEMA_VERSIONS = {"1.0", "1.1"}


@dataclass(frozen=True)
class ResidualHistorySummary:
    """Summary statistics for residual convergence history."""

    count: int
    first: float
    last: float
    min_value: float
    max_value: float
    monotone_nonincreasing: bool
    avg_log10_drop_per_step: float


@dataclass(frozen=True)
class PolicyMonotonicitySummary:
    """Monotonicity diagnostics for policy arrays along one axis."""

    axis: int
    tolerance: float
    violation_count: int
    min_diff: float
    is_monotone_non_decreasing: bool


def bellman_residual(
    value_new: npt.NDArray[np.float64],
    value_old: npt.NDArray[np.float64],
    norm: NormName = "sup",
) -> float:
    """Compute Bellman residual between successive value iterates.

    Economics note:
    The Bellman residual measures policy/value fixed-point error and is the
    standard stopping diagnostic for value-function iteration.
    """
    new_arr = np.asarray(value_new, dtype=np.float64)
    old_arr = np.asarray(value_old, dtype=np.float64)

    if new_arr.shape != old_arr.shape:
        raise ValueError("value_new and value_old must have the same shape")

    delta = new_arr - old_arr
    if norm == "sup":
        return float(np.max(np.abs(delta)))
    if norm == "l2":
        return float(np.sqrt(np.mean(delta * delta)))
    raise ValueError(f"Unsupported residual norm: {norm}")


def summarize_residual_history(history: list[float] | tuple[float, ...]) -> ResidualHistorySummary:
    """Return compact convergence diagnostics for a residual history."""
    if not history:
        raise ValueError("history must contain at least one residual value")

    arr = np.asarray(history, dtype=np.float64)
    if np.any(arr < 0.0):
        raise ValueError("residual history cannot contain negative values")

    count = int(arr.shape[0])
    first = float(arr[0])
    last = float(arr[-1])
    min_value = float(arr.min())
    max_value = float(arr.max())
    monotone_nonincreasing = bool(np.all(arr[1:] <= arr[:-1])) if count > 1 else True

    safe = np.maximum(arr, np.finfo(np.float64).tiny)
    if count > 1:
        avg_log10_drop_per_step = float((np.log10(safe[0]) - np.log10(safe[-1])) / (count - 1))
    else:
        avg_log10_drop_per_step = 0.0

    return ResidualHistorySummary(
        count=count,
        first=first,
        last=last,
        min_value=min_value,
        max_value=max_value,
        monotone_nonincreasing=monotone_nonincreasing,
        avg_log10_drop_per_step=avg_log10_drop_per_step,
    )


def format_convergence_report(history: list[float] | tuple[float, ...]) -> str:
    """Format a human-readable convergence report for logs and notebooks."""
    summary = summarize_residual_history(history)
    return (
        "Residual convergence report | "
        f"steps={summary.count}, "
        f"first={summary.first:.3e}, "
        f"last={summary.last:.3e}, "
        f"min={summary.min_value:.3e}, "
        f"max={summary.max_value:.3e}, "
        f"monotone_nonincreasing={summary.monotone_nonincreasing}, "
        f"avg_log10_drop_per_step={summary.avg_log10_drop_per_step:.3f}"
    )


def build_convergence_artifact(
    history: list[float] | tuple[float, ...],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable convergence artifact."""
    summary = summarize_residual_history(history)
    return {
        "history": [float(x) for x in history],
        "summary": asdict(summary),
        "metadata": {} if metadata is None else dict(metadata),
    }


def convergence_artifact_to_json(
    history: list[float] | tuple[float, ...],
    metadata: dict[str, Any] | None = None,
    indent: int = 2,
) -> str:
    """Serialize convergence artifact to JSON for persistence."""
    artifact = build_convergence_artifact(history, metadata=metadata)
    return json.dumps(artifact, indent=indent, sort_keys=True)


def summarize_policy_monotonicity(
    policy: npt.NDArray[np.float64],
    axis: int = 0,
    tolerance: float = 1e-12,
) -> PolicyMonotonicitySummary:
    """Check whether a policy is non-decreasing along a specified axis."""
    arr = np.asarray(policy, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError("policy must have at least one dimension")
    if not (0 <= axis < arr.ndim):
        raise ValueError("axis out of range for policy array")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    diffs = np.diff(arr, axis=axis)
    min_diff = float(np.min(diffs)) if diffs.size > 0 else 0.0
    violations = np.sum(diffs < -tolerance)
    violation_count = int(violations)

    return PolicyMonotonicitySummary(
        axis=axis,
        tolerance=tolerance,
        violation_count=violation_count,
        min_diff=min_diff,
        is_monotone_non_decreasing=violation_count == 0,
    )


def write_convergence_artifact(
    path: str | Path,
    history: list[float] | tuple[float, ...],
    metadata: dict[str, Any] | None = None,
    schema_version: str = "1.0",
) -> Path:
    """Write convergence artifact to disk with schema metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = build_convergence_artifact(history, metadata=metadata)
    artifact["schema"] = {
        "name": "econkit.convergence_artifact",
        "version": schema_version,
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def validate_convergence_schema_version(version: str) -> bool:
    """Return True if schema version is supported."""
    return version in SUPPORTED_CONVERGENCE_SCHEMA_VERSIONS


def validate_convergence_artifact_schema(
    artifact: dict[str, Any],
    allow_legacy_without_schema: bool = True,
) -> list[str]:
    """Validate convergence artifact shape and schema metadata.

    Returns a list of validation errors. Empty list means valid.
    """
    errors: list[str] = []

    has_history = "history" in artifact
    has_summary = "summary" in artifact
    has_metadata = "metadata" in artifact
    if not has_history:
        errors.append("missing field: history")
    if not has_summary:
        errors.append("missing field: summary")
    if not has_metadata:
        errors.append("missing field: metadata")

    if has_history and not isinstance(artifact["history"], list):
        errors.append("history must be a list")
    if has_summary and not isinstance(artifact["summary"], dict):
        errors.append("summary must be a dict")
    if has_metadata and not isinstance(artifact["metadata"], dict):
        errors.append("metadata must be a dict")

    schema = artifact.get("schema")
    if schema is None:
        if not allow_legacy_without_schema:
            errors.append("missing field: schema")
        return errors

    if not isinstance(schema, dict):
        errors.append("schema must be a dict")
        return errors

    name = schema.get("name")
    version = schema.get("version")
    if name != "econkit.convergence_artifact":
        errors.append("schema.name must equal 'econkit.convergence_artifact'")
    if not isinstance(version, str):
        errors.append("schema.version must be a string")
    elif not validate_convergence_schema_version(version):
        errors.append(f"unsupported schema version: {version}")

    return errors
