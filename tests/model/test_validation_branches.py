"""Every validation problem is detected and named (error branches of model compilation)."""

import numpy as np
import pytest

from dynecon.core.errors import ModelSpecificationError
from dynecon.model import AR1Shock, Control, Model, Param, State


def saving(**overrides) -> Model:
    m = Model("v")
    m.params(beta=Param(0.95), sigma=Param(2.0))
    m.add_state(State("a", bounds=(0.0, 5.0), n_points=10))
    m.add_state(State("z", shock=AR1Shock(0.5, 0.1, 3)))
    m.add_control(Control("c"))
    fields = dict(
        asset="a",
        shock="z",
        control="c",
        discount="beta",
        utility=lambda c, p: np.log(c),
        resources=lambda a, z, p: a + z,
    )
    fields.update(overrides)
    m.set_problem("saving", **fields)
    return m


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"asset": "nope"}, "saving.asset"),
        ({"shock": "a"}, "saving.shock"),
        ({"control": "nope"}, "saving.control"),
        ({"discount": "nope"}, "saving.discount"),
        ({"resources": lambda a, z, p: np.full((a.shape[0], z.shape[1]), np.nan)}, "non-finite"),
        ({"resources": lambda a, z, p: 1 / 0}, "raised ZeroDivisionError"),
        ({"utility": lambda c, p: np.zeros(2)}, "utility: returned shape"),
        (
            {
                "utility": lambda c, p: c,
                "marginal_utility": lambda c, p: np.zeros(2),
                "inverse_marginal_utility": lambda x, p: x,
                "marginal_return": lambda a, z, p: 1.0,
            },
            "marginal_utility: returned shape",
        ),
        (
            {
                "utility": lambda c, p: c,
                "marginal_utility": lambda c, p: c,
                "inverse_marginal_utility": lambda x, p: np.zeros(2),
                "marginal_return": lambda a, z, p: 1.0,
            },
            "inverse_marginal_utility",
        ),
        (
            {
                "utility": lambda c, p: c,
                "marginal_utility": lambda c, p: c,
                "inverse_marginal_utility": lambda x, p: x,
                "marginal_return": lambda a, z, p: np.zeros(4),
            },
            "marginal_return: returned shape",
        ),
    ],
)
def test_saving_problems_are_reported(overrides, message: str) -> None:
    with pytest.raises(ModelSpecificationError, match=message):
        saving(**overrides).compile()


def test_discount_outside_unit_interval_and_params_out_of_bounds() -> None:
    m = saving()
    m.params(beta=Param(1.5), sigma=Param(5.0, bounds=(0.5, 2.0)))
    with pytest.raises(ModelSpecificationError) as err:
        m.compile()
    assert "must lie in (0, 1)" in str(err.value) and "outside bounds" in str(err.value)


def test_duplicate_and_bad_grid_and_shock_reported_together() -> None:
    m = Model("bad")
    m.params(beta=Param(0.9))
    m.add_state(State("a", bounds=(1.0, 0.0), n_points=5))  # hi <= lo
    m.add_state(State("z", shock=AR1Shock(2.0, 0.1, 3)))  # |rho| >= 1
    m.add_control(Control("c"))
    m._states["x"] = State("a", bounds=(0, 1), n_points=3)  # a duplicate name inside the spec
    m.set_problem(
        "saving",
        asset="a",
        shock="z",
        control="c",
        discount="beta",
        utility=lambda c, p: c,
        resources=lambda a, z, p: a + z,
    )
    with pytest.raises(ModelSpecificationError) as err:
        m.compile()
    text = str(err.value)
    assert "duplicate state name" in text and "grid" in text and "shock" in text


def finite_dp(reward, transition, discount="beta", beta=0.9) -> Model:
    m = Model("dp")
    m.params(beta=Param(beta))
    m.set_problem("finite_dp", reward=reward, transition=transition, discount=discount)
    return m


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(reward=lambda p: np.ones(3), transition=lambda p: np.full((2, 2, 2), 0.5)), "reward"),
        (
            dict(reward=lambda p: np.ones((2, 2)), transition=lambda p: np.full((2, 2), 0.5)),
            "transition",
        ),
        (
            dict(reward=lambda p: np.ones((3, 2)), transition=lambda p: np.full((2, 2, 2), 0.5)),
            "inconsistent",
        ),
        (
            dict(
                reward=lambda p: np.ones((2, 2)),
                transition=lambda p: np.full((2, 2, 2), 0.5),
                discount="x",
            ),
            "finite_dp.discount",
        ),
        (
            dict(
                reward=lambda p: np.ones((2, 2)),
                transition=lambda p: np.full((2, 2, 2), 0.5),
                beta=1.2,
            ),
            "must lie in",
        ),
    ],
)
def test_finite_dp_problems_are_reported(kwargs, message: str) -> None:
    with pytest.raises(ModelSpecificationError, match=message):
        finite_dp(**kwargs).compile()


def test_pricing_problems_are_reported() -> None:
    def pricing(shock="d", discount="beta", mu=lambda c, p: c**-1.0, beta=0.9) -> Model:
        m = Model("lt")
        m.params(beta=Param(beta))
        m.add_state(State("d", shock=AR1Shock(0.5, 0.1, 3, log=True)))
        m.set_problem("pricing", shock=shock, discount=discount, marginal_utility=mu)
        return m

    pricing().compile()
    for kw, msg in [
        ({"shock": "nope"}, "pricing.shock"),
        ({"discount": "nope"}, "pricing.discount"),
        ({"beta": 1.1}, "must lie in"),
        ({"mu": lambda c, p: -c}, "positive"),
    ]:
        with pytest.raises(ModelSpecificationError, match=msg):
            pricing(**kw).compile()
