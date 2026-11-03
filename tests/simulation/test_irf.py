"""Generalised IRFs against the exact log-linear Brock-Mirman response."""

import numpy as np
import pytest
from builders import ALPHA, brock_mirman

import dynecon as ek
from dynecon.core.errors import ValidationError
from dynecon.simulation import simulate_irf


@pytest.mark.parametrize("rho", [0.0001, 0.9])
def test_brock_mirman_irf_matches_loglinear_theory(rho: float) -> None:
    model = brock_mirman(n_k=300, n_z=9, rho=rho, sigma_e=0.02).compile()
    sol = ek.solve(model, method="egm")
    z = model.shocks["z"].values
    step = np.log(z[1]) - np.log(z[0])
    horizon = 8
    irf = simulate_irf(
        sol, shock_steps=1, horizon=horizon, n_paths=6000, burn_in=300, log=True, rng=4
    )
    # k_hat[t+1] = alpha k_hat[t] + z_hat[t]; z_hat[t] = rho^t * step
    theory = np.zeros(horizon + 1)
    for t in range(horizon):
        theory[t + 1] = ALPHA * theory[t] + rho**t * step
    np.testing.assert_allclose(irf["k"] / step, theory / step, atol=2e-2)
    assert abs(irf["z"][0] / step - 1.0) < 1e-9


def test_irf_validation() -> None:
    sol = ek.solve(brock_mirman(n_k=40, n_z=5).compile(), method="egm")
    with pytest.raises(ValidationError):
        simulate_irf(sol, horizon=0)
    with pytest.raises(ValidationError):
        simulate_irf(sol, shock_steps=50, horizon=3, n_paths=20)
