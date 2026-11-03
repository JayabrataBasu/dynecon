# EconKit

A Python library for computational economics: specify, solve, simulate and estimate dynamic
economic models with a NumPy reference, Numba acceleration and a JAX backend for exact
derivatives.

```python
import econkit as ek
from econkit import models

model = models.load("consumption_savings").compile()
sol = ek.solve(model)  # endogenous grid method
sim = ek.simulate(sol, n_periods=500, n_agents=1000, rng=0)
print(sim.moments(["mean_assets", "std_consumption"]))
```

```{toctree}
:maxdepth: 2
:caption: Learn

user_guide/index
tutorials/index
```

```{toctree}
:maxdepth: 1
:caption: Models

models/index
```

```{toctree}
:maxdepth: 1
:caption: Guides

md/model_dsl
md/backends
md/simulation
md/equilibrium
md/perturbation
md/estimation
md/experiments
md/benchmarks
md/known_limits
md/API_POLICY
md/BACKEND_ROADMAP
```

```{toctree}
:maxdepth: 1
:caption: Theory

theory/numerics
theory/solvers
theory/distributions_and_welfare
theory/estimation
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
```
