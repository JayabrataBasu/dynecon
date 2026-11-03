"""Benchmark harness: repeatable timings with compile time reported separately.

Rules (plan R19): JIT/compile time is the first call and never mixed into steady-state
timings; repeats use identical inputs; every backend's result is checked against the NumPy
reference; machine and library versions are recorded; the minimum of the repeats is the
headline number. Timing is never a CI gate.
"""

from __future__ import annotations

import hashlib
import os
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from dynecon.utils.provenance import collect_provenance

__all__ = ["Case", "machine_info", "run_case"]


@dataclass(frozen=True)
class Case:
    """A benchmark: ``setup()`` builds inputs once; ``run(state, backend)`` returns an array."""

    name: str
    description: str
    setup: Callable[[], Any]
    run: Callable[[Any, str], np.ndarray]
    backends: tuple[str, ...] = ("numpy", "numba")
    tolerance: float = 1e-6


def machine_info() -> dict[str, Any]:
    """CPU, cores, OS and library versions of the machine running the benchmarks."""
    info: dict[str, Any] = {
        "processor": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "platform": platform.platform(),
        "versions": {
            k: v
            for k, v in collect_provenance().items()
            if k in ("python", "numpy", "scipy", "numba", "jax", "dynecon")
        },
    }
    try:
        import psutil  # optional

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["ram_gb"] = None
    return info


def run_case(
    case: Case,
    backend: str,
    reference: np.ndarray | None = None,
    n_repeat: int = 5,
    warmup: int = 1,
    state: Any = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Time one case on one backend; returns ``(record, result_array)``.

    ``compile_seconds`` is the first call minus the steady-state median (floored at zero).
    If ``reference`` is given the result must agree within ``case.tolerance`` (else raises).
    """
    state = case.setup() if state is None else state
    t0 = time.perf_counter()
    result = np.asarray(case.run(state, backend))
    first = time.perf_counter() - t0
    for _ in range(max(warmup - 1, 0)):
        case.run(state, backend)
    times = []
    for _ in range(n_repeat):
        t = time.perf_counter()
        case.run(state, backend)
        times.append(time.perf_counter() - t)
    median = statistics.median(times)
    diff = None
    if reference is not None:
        diff = float(np.max(np.abs(result.astype(float) - reference.astype(float))))
        scale = max(1.0, float(np.max(np.abs(reference))))
        if diff > case.tolerance * scale:
            raise AssertionError(
                f"{case.name}[{backend}] disagrees with the NumPy reference by {diff:.3g}"
            )
    record = {
        "name": case.name,
        "backend": backend,
        "description": case.description,
        "compile_seconds": max(first - median, 0.0),
        "first_call_seconds": first,
        "min": min(times),
        "median": median,
        "max": max(times),
        "n_repeat": n_repeat,
        "max_abs_diff_vs_numpy": diff,
        "result_hash": hashlib.sha256(np.round(result.astype(float), 6).tobytes()).hexdigest()[:16],
    }
    return record, result
