"""Error hierarchy: callers can catch ValueError for validation problems."""

import pytest

from econkit.core import errors as e


def test_hierarchy() -> None:
    assert issubclass(e.ValidationError, ValueError)
    assert issubclass(e.ValidationError, e.EconKitError)
    assert issubclass(e.ModelSpecificationError, e.ValidationError)
    assert issubclass(e.BackendUnavailableError, e.BackendError)
    assert issubclass(e.ConvergenceError, e.EconKitError)
    assert issubclass(e.ConvergenceWarning, Warning)


def test_validation_error_caught_as_value_error() -> None:
    with pytest.raises(ValueError):
        raise e.ValidationError("x")
