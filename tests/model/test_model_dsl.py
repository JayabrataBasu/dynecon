"""Model DSL and compilation tests.

Economics rationale:
A structured model definition is required to clearly separate economic intent
from numerical execution details.
"""

from econkit.model import AR1Shock, Control, Model, Param, State


def test_model_compile_minimal_vertical_slice() -> None:
    model = Model(name="consumption_savings")

    params = model.params(
        beta=Param(value=0.96, bounds=(0.8, 0.999), description="Discount factor"),
        sigma=Param(value=2.0, bounds=(0.5, 5.0), description="CRRA"),
    )
    assert "beta" in params

    a = model.add_state(State("assets", bounds=(0.0, 50.0), grid_type="log", n_points=200))
    z = model.add_state(State("income", shock=AR1Shock(rho=0.9, sigma_e=0.1, n_states=7)))
    c = model.add_control(Control("consumption", bounds=(1e-6, None)))

    @model.equation("budget")
    def budget(a_now: float, z_now: float, c_now: float, r: float = 0.04) -> float:
        return (1 + r) * a_now + z_now - c_now

    compiled = model.compile()

    assert compiled.name == "consumption_savings"
    assert a.name in compiled.states
    assert z.name in compiled.states
    assert c.name in compiled.controls
    assert "budget" in compiled.equations
    assert abs(compiled.call_equation("budget", 10.0, 1.0, 2.0) - 9.4) < 1e-12


def test_compile_requires_state_control_and_equation() -> None:
    model = Model(name="invalid")
    model.params(beta=Param(value=0.95))

    try:
        model.compile()
    except ValueError as exc:
        assert "equation" in str(exc)
    else:
        raise AssertionError("compile() must fail when required components are missing")
