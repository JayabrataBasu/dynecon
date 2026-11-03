"""Every compiled/accelerated operation must have a parity test against the NumPy reference."""

from pathlib import Path

from dynecon.numerics._dispatch import ensure_jax
from dynecon.numerics._registry import list_registered

TESTS = Path(__file__).resolve().parents[1]

# operation -> (test file, test function name) proving agreement with the NumPy reference
PARITY_CASES = {
    "linear_interp_1d": ("numerics/test_interpolation.py", "test_backend_parity"),
    "linear_interp_columns": ("numerics/test_interpolation.py", "test_columns_match_rowwise_1d"),
    "bellman_saving": ("solvers/test_vfi.py", "test_numpy_numba_parity"),
    "distribution_lottery": ("simulation/test_distribution.py", "test_numba_lottery_parity"),
    "markov_paths": ("numerics/test_markov.py", "test_backend_parity_exact"),
    "simulate_saving": (
        "simulation/test_simulate.py",
        "test_backend_parity_and_thread_independence",
    ),
}


def test_every_registered_op_has_parity_case() -> None:
    ensure_jax()
    accelerated = {op for (op, backend) in list_registered() if backend != "numpy"}
    missing = accelerated - set(PARITY_CASES)
    assert not missing, f"operations without a parity test: {sorted(missing)}"


def test_named_parity_tests_exist() -> None:
    for op, (file, name) in PARITY_CASES.items():
        text = (TESTS / file).read_text(encoding="utf-8")
        assert f"def {name}" in text, f"{op}: {file}::{name} not found"


def test_jax_ops_have_jax_parity_coverage() -> None:
    jax_ops = {op for (op, backend) in list_registered() if backend == "jax"}
    text = (TESTS / "jax" / "test_jax_backend.py").read_text(encoding="utf-8")
    for needle in (
        "interpolation_parity",
        "markov_paths_exact_parity",
        "vfi_and_distribution_parity",
    ):
        assert needle in text
    assert jax_ops <= {
        "linear_interp_1d",
        "linear_interp_columns",
        "bellman_saving",
        "distribution_lottery",
        "markov_paths",
    }
