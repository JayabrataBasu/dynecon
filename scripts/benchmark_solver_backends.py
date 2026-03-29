"""Micro-benchmark for bounded optimization backends."""

from __future__ import annotations

import statistics
import time

from econkit.numerics.optimize import brent_max, golden_section_max


def _objective(x: float) -> float:
    return -((x - 2.0) ** 2) + 5.0


def _time_call(fn, repeats: int = 10) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.mean(samples)


def run_benchmark() -> dict[str, float]:
    results: dict[str, float] = {}

    results["golden_numpy_s"] = _time_call(
        lambda: golden_section_max(_objective, 0.0, 4.0, backend="numpy")
    )
    results["brent_numpy_s"] = _time_call(lambda: brent_max(_objective, 0.0, 4.0, backend="numpy"))

    try:
        results["golden_numba_s"] = _time_call(
            lambda: golden_section_max(_objective, 0.0, 4.0, backend="numba")
        )
        results["brent_numba_s"] = _time_call(
            lambda: brent_max(_objective, 0.0, 4.0, backend="numba")
        )
    except Exception:
        results["golden_numba_s"] = float("nan")
        results["brent_numba_s"] = float("nan")

    return results


if __name__ == "__main__":
    out = run_benchmark()
    for key in sorted(out):
        print(f"{key}: {out[key]:.8f}")
