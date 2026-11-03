"""Config and backend introspection."""

import pytest

import dynecon as ek
from dynecon import backends
from dynecon.config import Config
from dynecon.core.errors import BackendError, BackendUnavailableError, ValidationError
from dynecon.numerics._dispatch import resolve_backend


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValidationError):
        Config().set_backend("fortran")


def test_unavailable_backend_warns_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "is_available", lambda name: name == "numpy")
    cfg = Config()
    with pytest.warns(UserWarning):
        cfg.set_backend("numba")
    assert cfg.active_backend == "numpy"


def test_unavailable_backend_strict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "is_available", lambda name: name == "numpy")
    cfg = Config(strict_backend=True)
    with pytest.raises(BackendUnavailableError):
        cfg.set_backend("jax")


def test_available_backends_lists_numpy_first() -> None:
    assert backends.available_backends()[0] == "numpy"
    assert backends.is_available("numpy")
    assert not backends.is_available("nonsense")


def test_describe_lists_registered_operations() -> None:
    text = ek.backends.describe()
    assert "linear_interp_1d" in text
    assert "numpy" in text


def test_strict_mode_unregistered_operation_raises() -> None:
    original = ek.config.strict_backend
    ek.config.strict_backend = True
    try:
        with pytest.raises(BackendError):
            resolve_backend("linear_interp_1d", backend="rust")
        with pytest.raises(BackendError):
            resolve_backend("no_such_operation", backend="numpy")
    finally:
        ek.config.strict_backend = original


def test_non_strict_falls_back_to_numpy() -> None:
    name, _ = resolve_backend("linear_interp_1d", backend="rust")
    assert name == "numpy"
