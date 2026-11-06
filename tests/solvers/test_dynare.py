import numpy as np
import pytest
from scipy import stats

from dynecon.core.errors import DeterminacyError, ValidationError
from dynecon.solvers.perturbation.dynare import evaluate_assignments, parse_linear_model

MOD = """
// forward-looking price with an AR(1) driver, observed with a constant
var p z pobs;
varexo e;
parameters a rho c;
model(linear);
  p = a*p(+1) + z;
  z = rho*z(-1) + e;
  pobs = p + c;   /* level of the observable */
end;
"""


def test_solution_matches_the_closed_form_and_the_ar1_likelihood() -> None:
    m = parse_linear_model(MOD)
    assert m.variables == ("p", "z", "pobs") and m.shocks == ("e",) and len(m.equations) == 3
    a, rho, c, sigma = 0.5, 0.8, 2.0, 0.3
    params = {"a": a, "rho": rho, "c": c}
    Z, T, Q, d, _ = m.state_space(params, {"e": sigma}, ["pobs"])
    k = 1.0 / (1.0 - a * rho)  # p = k z
    np.testing.assert_allclose(d, [c])
    rng = np.random.default_rng(0)
    u = np.empty(60)
    u[0] = rng.normal(0, k * sigma / np.sqrt(1 - rho**2))
    for t in range(1, 60):
        u[t] = rho * u[t - 1] + k * sigma * rng.normal()
    y = (u + c)[:, None]
    expected = stats.norm.logpdf(u[0], 0, k * sigma / np.sqrt(1 - rho**2))
    expected += stats.norm.logpdf(u[1:], rho * u[:-1], k * sigma).sum()
    assert abs(m.loglik(y, params, {"e": sigma}, ["pobs"]) - expected) < 1e-9
    # presample: dropping the first period removes exactly its contribution
    first = stats.norm.logpdf(u[0], 0, k * sigma / np.sqrt(1 - rho**2))
    assert abs(m.loglik(y, params, {"e": sigma}, ["pobs"], presample=1) - (expected - first)) < 1e-9


def test_indeterminacy_and_bad_inputs() -> None:
    m = parse_linear_model(MOD)
    with pytest.raises(DeterminacyError):
        m.state_space({"a": 1.5, "rho": 0.8, "c": 0.0}, {"e": 1.0}, ["p"])  # sunspots
    ok = {"a": 0.5, "rho": 0.8, "c": 0.0}
    with pytest.raises(ValidationError):
        m.state_space(ok, {"e": 1.0}, ["nope"])
    with pytest.raises(ValidationError):
        m.state_space(ok, {}, ["p"])  # missing shock sd
    with pytest.raises(ValidationError):
        m.state_space({"a": 0.5}, {"e": 1.0}, ["p"])  # undefined parameter
    with pytest.raises(ValidationError):
        m.loglik(np.ones((5, 2)), ok, {"e": 1.0}, ["p"])
    with pytest.raises(ValidationError):
        parse_linear_model("var x; model(linear); x = x(-1); end;")  # no varexo
    with pytest.raises(ValidationError):
        parse_linear_model("var x y; varexo e; model(linear); x = x(-1)+e; end;")  # 1 eq, 2 vars
    with pytest.raises(ValidationError):
        parse_linear_model("var x; varexo e; model(linear); x = x*x; end;").state_space(
            {}, {"e": 1.0}, ["x"]
        )


def test_evaluate_assignments_in_order() -> None:
    out = evaluate_assignments("b = a^2 + 1; // c\nc = b/2; d = exp(0)", {"a": 3.0})
    assert out == {"b": 10.0, "c": 5.0, "d": 1.0}
    with pytest.raises(ValidationError):
        evaluate_assignments("x = missing + 1", {})
    with pytest.raises(ValidationError):
        evaluate_assignments("x = 1/0", {})
