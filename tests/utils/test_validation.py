import numpy as np
import pytest

from dynecon.core.errors import ValidationError
from dynecon.utils.provenance import collect_provenance
from dynecon.utils.validation import check_array, check_increasing, check_stochastic_matrix


def test_check_array_messages_name_argument() -> None:
    with pytest.raises(ValidationError, match="myarg"):
        check_array([1.0, np.nan], "myarg")
    with pytest.raises(ValidationError, match="myarg"):
        check_array([[1.0]], "myarg", ndim=1)
    with pytest.raises(ValidationError, match="myarg"):
        check_array([1.0, 2.0], "myarg", shape=(3,))
    with pytest.raises(ValidationError, match="myarg"):
        check_array(["a"], "myarg")
    assert check_array([1, 2], "ok", shape=(None,)).dtype == np.float64


def test_check_increasing() -> None:
    check_increasing(np.array([0.0, 1.0, 2.0]), "g")
    with pytest.raises(ValidationError, match="grid"):
        check_increasing(np.array([0.0, 1.0, 1.0]), "grid")
    check_increasing(np.array([0.0, 1.0, 1.0]), "grid", strict=False)
    with pytest.raises(ValidationError):
        check_increasing(np.array([1.0, 0.0]), "grid", strict=False)


def test_check_stochastic_matrix() -> None:
    check_stochastic_matrix(np.eye(2), "P")
    with pytest.raises(ValidationError, match="Pmat"):
        check_stochastic_matrix(np.array([[0.5, 0.4], [0.5, 0.5]]), "Pmat")
    with pytest.raises(ValidationError, match="Pmat"):
        check_stochastic_matrix(np.array([[1.5, -0.5], [0.5, 0.5]]), "Pmat")
    with pytest.raises(ValidationError, match="Pmat"):
        check_stochastic_matrix(np.ones((2, 3)) / 3, "Pmat")


def test_provenance_has_versions() -> None:
    p = collect_provenance()
    assert p["numpy"] and p["scipy"] and p["python"] and p["dynecon"]
    assert "timestamp_utc" in p
