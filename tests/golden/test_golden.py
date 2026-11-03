"""Golden-file regression tests: catch silent numerical drift across refactors and upgrades.

Update deliberately with ``pytest tests/golden --update-golden``; record why in the changelog.
"""

from pathlib import Path

import numpy as np
import pytest

import dynecon as ek
from dynecon import models

GOLDEN = Path(__file__).parent


def _check(request: pytest.FixtureRequest, name: str, arrays: dict[str, np.ndarray]) -> None:
    path = GOLDEN / f"{name}.npz"
    if request.config.getoption("--update-golden"):
        np.savez_compressed(path, **arrays)
        pytest.skip(f"updated {path.name}")
    assert path.exists(), f"missing golden file {path.name}; run with --update-golden"
    with np.load(path) as ref:
        assert set(ref.files) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_allclose(value, ref[key], rtol=1e-8, atol=1e-10, err_msg=key)


def test_consumption_savings_egm_and_vfi(request) -> None:
    model = models.load("consumption_savings", n_a=100, n_z=5).compile()
    egm = ek.solve(model, method="egm", backend="numpy")
    vfi = ek.solve(model, method="vfi", backend="numpy", tol=1e-10)
    _check(
        request,
        "consumption_savings",
        {
            "egm_consumption": egm.policy["consumption"],
            "vfi_assets_next": vfi.policy["assets_next"],
            "vfi_value": vfi.value,
        },
    )


def test_rbc_perturbation_matrices(request) -> None:
    sol = ek.solve(models.load("rbc").compile())
    _check(
        request,
        "rbc_perturbation",
        {"F": sol.F, "P": sol.P, "Sigma": sol.Sigma, "ss": sol.steady_state},
    )
