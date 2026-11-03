"""Model DSL: compilation, validation, immutability and fingerprints."""

import numpy as np
import pytest

from dynecon.core.errors import ModelSpecificationError, ValidationError
from dynecon.model import AR1Shock, Control, IIDShock, MarkovShock, Model, Param, State
from dynecon.model.shocks import discretize_shock


def cs_model(beta: float = 0.96, n_a: int = 40, n_z: int = 5, name: str = "cs") -> Model:
    m = Model(name)
    m.params(
        beta=Param(beta, bounds=(0.8, 0.999), description="Discount factor"),
        sigma=Param(2.0, bounds=(0.5, 5.0), description="CRRA"),
        r=Param(0.03, bounds=(-0.05, 0.1)),
        rho=Param(0.9, bounds=(0.0, 0.99), free=True),
    )
    m.add_state(State("assets", bounds=(0.0, 20.0), grid_type="exp", n_points=n_a))
    m.add_state(State("income", shock=AR1Shock(rho="rho", sigma_e=0.1, n_states=n_z, log=True)))
    m.add_control(Control("consumption"))
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=lambda c, p: c ** (1 - p.sigma) / (1 - p.sigma),
        marginal_utility=lambda c, p: c ** (-p.sigma),
        inverse_marginal_utility=lambda x, p: x ** (-1 / p.sigma),
        resources=lambda a, z, p: (1 + p.r) * a + z,
        marginal_return=lambda a, z, p: 1 + p.r,
    )
    return m


def test_compile_resources_and_describe() -> None:
    c = cs_model().compile()
    assert c.name == "cs"
    a, z = c.grids["assets"], c.shocks["income"].values
    got = c.problem.resources(a[:, None], z[None, :], c.p)
    np.testing.assert_allclose(got, (1 + 0.03) * a[:, None] + z[None, :])
    assert c.problem.supports_egm
    assert "assets" in c.describe() and "fingerprint" in repr(c)


def test_with_params_is_copy_on_write_and_updates_shocks() -> None:
    c = cs_model().compile()
    c2 = c.with_params(beta=0.9, rho=0.5)
    assert c.p.beta == 0.96 and c2.p.beta == 0.9
    assert not np.allclose(c.shocks["income"].P, c2.shocks["income"].P)
    with pytest.raises(ValidationError):
        c.with_params(nope=1.0)
    with pytest.raises(ModelSpecificationError):
        c.with_params(beta=0.5)  # outside bounds
    with pytest.raises(AttributeError):
        c.p.beta = 0.1  # type: ignore[misc]


def test_fingerprint_sensitive_and_stable() -> None:
    f1 = cs_model().compile().fingerprint()
    assert f1 == cs_model().compile().fingerprint()
    assert f1 != cs_model(beta=0.96 + 1e-12).compile().fingerprint()
    assert f1 != cs_model(n_a=41).compile().fingerprint()


def test_multiple_problems_reported_together() -> None:
    m = Model("bad")
    m.params(beta=Param(1.2))
    m.add_state(State("assets"))
    m.add_control(Control("c"))
    m.set_problem(
        "saving",
        asset="assets",
        control="c",
        discount="beta",
        utility=lambda c, p: c,
        resources=lambda a, z, p: a,
    )
    with pytest.raises(ModelSpecificationError) as err:
        m.compile()
    msg = str(err.value)
    assert "assets" in msg and "n_points" in msg


def test_wrong_shape_callable_names_field() -> None:
    m = cs_model()
    prob = m._problem
    assert prob is not None
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=prob.utility,
        resources=lambda a, z, p: np.zeros(3),
    )
    with pytest.raises(ModelSpecificationError, match="resources"):
        m.compile()


def test_partial_marginal_fields_rejected_and_unknown_template() -> None:
    m = cs_model()
    p = m._problem
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=p.utility,
        resources=p.resources,
        marginal_utility=p.marginal_utility,
    )
    with pytest.raises(ModelSpecificationError, match="EGM"):
        m.compile()
    with pytest.raises(ModelSpecificationError):
        m.set_problem("nope")
    with pytest.raises(ModelSpecificationError):
        m.set_problem("saving", asset="x")


def test_bad_inverse_marginal_utility_detected() -> None:
    m = cs_model()
    p = m._problem
    m.set_problem(
        "saving",
        asset="assets",
        shock="income",
        control="consumption",
        discount="beta",
        utility=p.utility,
        resources=p.resources,
        marginal_utility=p.marginal_utility,
        inverse_marginal_utility=lambda x, q: x,
        marginal_return=p.marginal_return,
    )
    with pytest.raises(ModelSpecificationError, match="inverse_marginal_utility"):
        m.compile()


def test_no_problem_and_builder_helpers() -> None:
    m = Model("empty")
    with pytest.raises(ModelSpecificationError, match="no problem"):
        m.compile()
    m2 = cs_model()
    cp = m2.copy("other")
    assert cp.name == "other" and "assets" in repr(cp)
    with pytest.raises(ValidationError):
        m2.add_state(State("assets"))
    with pytest.raises(ValidationError):
        m2.add_control(Control("consumption"))
    with pytest.raises(TypeError):
        m2.params(x=1.0)  # type: ignore[arg-type]


def test_deterministic_saving_without_shock() -> None:
    m = Model("det")
    m.params(beta=Param(0.95), r=Param(0.02), y=Param(1.0))
    m.add_state(State("a", bounds=(0.0, 5.0), n_points=20))
    m.add_control(Control("c"))
    m.set_problem(
        "saving",
        asset="a",
        control="c",
        discount="beta",
        utility=lambda c, p: np.log(c),
        resources=lambda a, z, p: (1 + p.r) * a + p.y,
    )
    c = m.compile()
    assert c.shock_process(None).P.shape == (1, 1)
    with pytest.raises(ValidationError):
        c.shock_process("zzz")


def test_finite_dp_template_validation() -> None:
    def build(T: np.ndarray) -> Model:
        m = Model("dp")
        m.params(beta=Param(0.9))
        m.set_problem(
            "finite_dp", reward=lambda p: np.ones((2, 2)), transition=lambda p: T, discount="beta"
        )
        return m

    good = np.full((2, 2, 2), 0.5)
    assert build(good).compile().problem.discount == "beta"
    with pytest.raises(ModelSpecificationError, match="transition"):
        build(np.full((2, 2, 2), 0.4)).compile()


def test_parameter_set_vector_view() -> None:
    c = cs_model().compile()
    ps = c.parameters
    assert ps.names() == ["rho"] and ps.names(free=False)[0] == "beta"
    np.testing.assert_allclose(ps.vector(), [0.9])
    assert ps.bounds(["beta"]) == [(0.8, 0.999)]
    ps2 = ps.from_vector([0.5])
    assert ps2["rho"].value == 0.5 and ps["rho"].value == 0.9
    with pytest.raises(ValidationError):
        ps.from_vector([1.0, 2.0])
    with pytest.raises(ValidationError):
        ps.vector(["zzz"])
    assert len(ps) == 4 and "beta" in ps and list(ps)[0] == "beta"


def test_shock_discretisations() -> None:
    mk = discretize_shock(MarkovShock(np.array([0.0, 1.0]), np.array([[0.9, 0.1], [0.2, 0.8]])), {})
    assert mk.P.shape == (2, 2)
    with pytest.raises(ValidationError):
        discretize_shock(MarkovShock(np.array([0.0, 1.0]), np.eye(3)), {})
    nrm = discretize_shock(IIDShock("normal", {"mu": 1.0, "sigma": 0.5}, 7), {})
    assert abs(nrm.P[0] @ nrm.values - 1.0) < 1e-12
    ln = discretize_shock(IIDShock("lognormal", {"mu": 0.0, "sigma": 0.2}, 9), {})
    assert abs(ln.P[0] @ ln.values - np.exp(0.02)) < 1e-6
    un = discretize_shock(IIDShock("uniform", {"low": 2.0, "high": 4.0}, 4), {})
    assert abs(un.P[0] @ un.values - 3.0) < 1e-12
    with pytest.raises(ValidationError):
        discretize_shock(AR1Shock("missing", 0.1, 3), {})
    tz = discretize_shock(AR1Shock(0.9, 0.1, 5, method="tauchen"), {})
    assert tz.values.shape == (5,)
    with pytest.raises(ValidationError):
        discretize_shock(IIDShock("cauchy", {}, 3), {})  # type: ignore[arg-type]
