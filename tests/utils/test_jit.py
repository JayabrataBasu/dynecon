import importlib

import pytest

from econkit.utils import jit as jit_module


def _f(x: float) -> float:
    return x * 2.0 + 1.0


def test_disabled_returns_plain_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECONKIT_DISABLE_JIT", "1")
    assert jit_module.jit(_f) is _f
    assert jit_module.jit(parallel=True)(_f) is _f


def test_results_match_pure_python() -> None:
    pytest.importorskip("numba")
    compiled = jit_module.jit(_f)
    assert compiled(3.0) == _f(3.0)
    compiled_par = jit_module.jit(parallel=True, cache=False)(_f)
    assert compiled_par(3.0) == _f(3.0)


def test_numba_flag_consistent() -> None:
    assert jit_module.NUMBA_AVAILABLE == (importlib.util.find_spec("numba") is not None)
