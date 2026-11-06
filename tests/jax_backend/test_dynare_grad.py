"""Exact likelihood gradient of a linear Dynare model against finite differences."""

import numpy as np
import pytest

from dynecon.core.errors import DeterminacyError, ValidationError

pytestmark = pytest.mark.jax
pytest.importorskip("jax.numpy")
from dynecon.solvers.perturbation.dynare import parse_linear_model  # noqa: E402
from dynecon.solvers.perturbation.dynare_grad import loglik_and_grad  # noqa: E402

MOD = """
var x pi r u pio robs;
varexo e1 e2;
model(linear);
  pi = b*pi(+1) + k*x + u;
  x = x(+1) - s*(r - pi(+1));
  r = phi*pi + rr*r(-1) + e2;
  u = ru*u(-1) + e1;
  pio = pi + c1;
  robs = r + c2;
end;
"""
NAMES = ["b", "k", "s", "phi", "rr", "ru", "sd1", "sd2", "c1", "c2"]
THETA = np.array([0.99, 0.15, 1.0, 1.6, 0.5, 0.7, 0.4, 0.3, 0.5, 1.0])


def to_model(theta):
    p = dict(zip(NAMES, theta, strict=True))
    return {k: p[k] for k in ("b", "k", "s", "phi", "rr", "ru", "c1", "c2")}, {
        "e1": p["sd1"],
        "e2": p["sd2"],
    }


def _data(seed=0, t=40):
    rng = np.random.default_rng(seed)
    return np.column_stack([0.5 + rng.normal(0, 0.6, t), 1.0 + rng.normal(0, 0.5, t)])


def test_gradient_matches_finite_differences() -> None:
    model = parse_linear_model(MOD)
    obs = ["pio", "robs"]
    y = _data()
    val, g = loglik_and_grad(model, y, THETA, to_model, obs, presample=2)
    params, sd = to_model(THETA)
    assert abs(val - model.loglik(y, params, sd, obs, presample=2, init="dynare2")) < 1e-8
    for i in range(THETA.size):
        h = 1e-6 * max(1.0, abs(THETA[i]))
        up, dn = THETA.copy(), THETA.copy()
        up[i] += h
        dn[i] -= h
        f = lambda th: model.loglik(y, *to_model(th), obs, presample=2, init="dynare2")  # noqa: E731
        assert abs(g[i] - (f(up) - f(dn)) / (2 * h)) < 1e-4 * (1 + abs(g[i])), NAMES[i]


def test_bad_inputs_and_indeterminacy() -> None:
    model = parse_linear_model(MOD)
    obs, y = ["pio", "robs"], _data()
    with pytest.raises(ValidationError):
        loglik_and_grad(model, y[:, :1], THETA, to_model, obs)
    with pytest.raises(ValidationError):
        loglik_and_grad(model, y, THETA, to_model, ["nope", "robs"])
    bad = THETA.copy()
    bad[3] = 0.2  # Taylor rule too weak: indeterminate
    with pytest.raises(DeterminacyError):
        loglik_and_grad(model, y, bad, to_model, obs)
