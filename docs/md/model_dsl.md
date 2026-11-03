# The model DSL

A model is declared with a mutable `Model` builder, then `compile()`d into an immutable
`CompiledModel`: shocks are discretised, grids built, and every template callable is evaluated
once on a small input so shape and finiteness errors surface at compile time (all problems are
reported together in one `ModelSpecificationError`).

## Worked example: consumption-savings (about 25 lines)

```python
import numpy as np
import dynecon as ek
from dynecon import Model, Param, State, Control, AR1Shock

m = Model("consumption_savings")
m.params(
    beta=Param(0.96, bounds=(0.5, 0.9999)),
    sigma=Param(2.0),
    r=Param(0.03),
    rho=Param(0.9),
    sigma_e=Param(0.1),
)
m.add_state(
    State("assets", bounds=(0.0, 50.0), grid_type="exp", n_points=200)
)  # lower bound = borrowing limit
m.add_state(State("income", shock=AR1Shock("rho", "sigma_e", 7, log=True)))
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
model = m.compile()
sol = ek.solve(model)  # EGM, because the marginal fields are given
sim = ek.simulate(sol, n_periods=500, n_agents=100, rng=0)
```

Template callables are plain NumPy functions that broadcast over arrays. They run **outside**
compiled kernels (kernels receive arrays computed from them), so you never write Numba code.
`p` is a read-only view of parameter values (`p.beta`). `AR1Shock` fields may name parameters
(`"rho"`), which makes them estimable; `with_params(rho=0.5)` re-discretises.

## A growth model differs in two lines

Brock-Mirman uses the same template with `resources = z * k**alpha` and
`marginal_return = alpha * z * k**(alpha - 1)`.

## Templates

| Template | Contract | Solved by |
|---|---|---|
| `saving` | one continuous asset, discrete/discretised shocks, one control | VFI, EGM |
| `finite_dp` | `reward(p) -> [S, A]`, `transition(p) -> [A, S, S']` | VFI (tabular) |
| `dsge` | `E_t f(y_{t+1}, y_t; p) = 0`, predetermined states first | perturbation (Klein) |
| `pricing` | dividend shock, discount, marginal utility | fixed point on prices |

`dsge` equations receive `Vars` (name-addressable: `y.k`, `yp.c`) of levels, return one
residual per variable and must use `dynecon.models.utility.log/exp` (not NumPy's) if you want
JAX exact derivatives. `log_variables` makes responses percent deviations.

## Immutability and fingerprints

`compiled.with_params(beta=0.9)` returns a new model. `compiled.fingerprint()` is a SHA-256
over parameters, grids, shock arrays and the *source text* of the template callables; it is
stored on every `Solution`. (Source text of lambdas defined in a REPL falls back to the
qualified name.)
