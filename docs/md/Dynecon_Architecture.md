<!-- ============================================================
     Dynecon — Technical Architecture & Design Rationale
     Sections 1–19 | Revision 2 | Aligned with the v1.0 Execution Plan (rev 2)
     ============================================================ -->

# Dynecon

> **A Python Library for Computational Economics**  
> *Technical Architecture & Design Rationale*  
> Revision 2 — written as an interpretation of `V1_EXECUTION_PLAN.md` (revision 2)

## How to read this document

This document and the execution plan are two views of one project and are meant to be read together.

- **The execution plan** says exactly what version 1.0 contains, in which order it is built, and how each part is tested. It is the source of truth for scope. If the two documents ever disagree, the plan wins and this document is amended.
- **This document** explains the design: why the pieces are shaped as they are, how they fit together, what the compiled-layer strategy is, and where the project goes after 1.0.
- Features are marked **[1.0]** (in the plan) or **after 1.0** (roadmap, Section 16). Nothing in this document describes v1.0 behaviour that the plan does not build.
- Everything in v1.0 follows one engineering rule, stated as R21 in the plan and summarised in Section 3.1: *one concept, one implementation; nothing without a caller and a test; deferred features return with their first consumer.*

---
# Section 1 — Project Vision

## 1.1 Purpose

Dynecon is a Python library designed to be the definitive open-source computational platform for economic modeling, analysis, and research. The library bridges the gap between the pedagogical simplicity that students require and the computational rigor and extensibility that professional researchers demand. Dynecon provides a unified, expressive, and performant framework for specifying, solving, simulating, and estimating a wide variety of economic models.

The library is designed from first principles with two interlocking goals: (1) to function as a serious research instrument capable of supporting publication-quality computational economics, and (2) to serve as an accessible, well-documented learning platform through which students and newcomers can build intuition for the quantitative methods underlying modern economics.

## 1.2 User Personas

### Graduate Students and Educators

Students in economics, finance, and related fields require tools that let them focus on understanding models rather than wrestling with low-level numerical infrastructure. Dynecon provides clean, expressive APIs backed by sensible defaults so that a first-year PhD student can implement a canonical consumption-savings problem in under twenty lines of code while still accessing advanced configuration when needed.

### Academic Researchers

Researchers working on frontier problems in heterogeneous agent macroeconomics, dynamic discrete choice, and structural estimation need a library that is both numerically robust and architecturally extensible. Dynecon is designed so that new solvers, estimators, and model components can be added without modifying core library internals, following open-closed design principles.

### Policy Analysts and Applied Economists

Practitioners in government institutions, central banks, and think tanks need reproducible, auditable pipelines. Dynecon supports structured experiment management, parameter sweeps, and integration with standard data workflows (pandas, polars, xarray) to fit naturally into professional analytical environments.

## 1.3 Problem Domains

Dynecon is designed to address the following classes of economic problems. Version 1.0 delivers a defined subset (marked **[1.0]**); the rest is the roadmap in Section 16.

- **[1.0]** Dynamic programming problems: infinite-horizon, discrete-time, deterministic and stochastic, with continuous or finite choices. Finite-horizon and continuous-time (HJB) problems follow after 1.0.
- **[1.0]** Equilibrium computation: stationary recursive competitive equilibria in partial and general equilibrium (Aiyagari, Huggett). Transition dynamics and aggregate-risk equilibria follow after 1.0.
- **[1.0]** Heterogeneous agent models: Bewley-Huggett-Aiyagari economies with exact (histogram) stationary distributions. Krusell-Smith economies and related HA frameworks follow after 1.0.
- **[1.0]** Structural estimation: simulated method of moments, maximum likelihood (including Kalman-filter likelihood), and Bayesian inference by Metropolis-Hastings.
- **[1.0]** Counterfactual policy experiments with consumption-equivalent welfare measures. Optimal-policy design (optimal tax schedules, monetary policy rules) follows after 1.0.
- **[1.0]** Dynamic discrete choice in its finite-state form (the McCall job-search model). Richer discrete-choice models (education, retirement, labour supply) follow after 1.0.

## 1.4 Differentiation from Existing Tools

The existing ecosystem of computational economics tools is fragmented. Each existing library addresses some of these needs but none addresses all of them in a unified, consistent framework:

| Tool | Strength | Gap Dynecon Fills |
| --- | --- | --- |
| QuantEcon | Excellent pedagogical library; clean Python/Julia split | No unified model specification; limited estimation; no HA infrastructure |
| Econ-ARK | Strong HA lifecycle modeling; active community | Steep learning curve; limited DSGE/GE support; fragmented API |
| dolo | Model specification via YAML; perturbation methods | Limited to DSGE-style models; thin documentation; niche adoption |
| statsmodels | Excellent econometrics and time series | No dynamic programming; no structural economic models |
| Dynare | Gold standard for DSGE estimation | MATLAB-only (or MATLAB-dependent); closed ecosystem; not Pythonic |

Dynecon unifies dynamic programming, GE computation, HA simulation, and structural estimation under a single coherent API, written in Python with performance layers (Numba and JAX in version 1.0; C and Rust later, if benchmarks justify them) behind a backend dispatcher.

## 1.5 Long-Term Ecosystem Goals

The long-term vision for Dynecon is to become the scikit-learn of computational economics: a stable, well-tested, well-documented foundation upon which researchers and educators can build without reinventing numerical infrastructure. This means:

1. Establishing a model zoo of canonical economic models with pedagogical documentation.
1. Creating, after 1.0, a plugin/extension system that allows domain-specific packages (e.g., econ-kit-finance, econ-kit-labor) to build on the core. Version 1.0 already has a public model registry (`register(name, builder)`), which is the seed of that system.
1. Building an ecosystem of tutorials, textbook integrations, and course materials.
1. Enabling reproducible research with structured provenance tracking and export.
1. Providing, in the long term, interoperability with Julia via shared model specification formats.

# Section 2 — Landscape Analysis

## 2.1 QuantEcon

QuantEcon is the most widely adopted computational economics library in Python and Julia. It provides a rich collection of algorithms for dynamic programming (value function iteration, policy function iteration, Howard improvement), Markov chains, linear quadratic control, and time series analysis. Its Jupyter notebook-based lectures are widely used in graduate economics courses.

### Strengths

- High-quality, well-tested implementations of core algorithms.
- Parallel Python and Julia implementations enabling comparison.
- Integrated lecture series that makes it excellent for pedagogy.
- Active maintenance and good community support.

### Limitations

- No unified model specification system: models are defined procedurally, making it difficult to generalize across algorithms.
- No structural estimation infrastructure (SMM, MLE, Bayesian).
- No native support for heterogeneous agent economies.
- Limited visualization and experiment management tooling.
- APIs are somewhat inconsistent across modules due to incremental development.

## 2.2 Econ-ARK / HARK

Econ-ARK (the Heterogeneous Agents Resources & toolKit, embodied in the HARK package) is designed specifically for lifecycle models with heterogeneous agents. It provides a sophisticated framework for solving consumer optimization problems with income risk, mortality, and portfolio choice.

### Strengths

- Best-in-class infrastructure for lifecycle heterogeneous agent models.
- Careful numerical treatment of borrowing constraints and non-convexities.
- Active research community; used in published work at major institutions.

### Limitations

- Steep learning curve with a complex class hierarchy that can be opaque to new users.
- Limited support for production-side models, GE, and DSGE frameworks.
- Performance bottlenecks at scale due to Python loops.
- Documentation quality is uneven across components.

## 2.3 dolo

dolo is a Python library for specifying and solving decision models using a YAML-based domain-specific language. It supports local perturbation methods (first- and second-order approximations) and global methods (time iteration, value function iteration) primarily for rational expectations models.

### Strengths

- Model-as-data philosophy: models defined in YAML enable sharing and version control.
- Clean interface for perturbation methods.
- Supports symbolic differentiation of model equations.

### Limitations

- Primarily oriented toward representative-agent DSGE models.
- Limited estimation support; no SMM or Bayesian tools.
- Small community; limited tutorials and examples.

## 2.4 statsmodels

statsmodels is a mature Python library for statistical models and econometric methods, covering OLS, GLM, time series (ARIMA, VAR, state space), panel data, and non-parametric methods.

### Strengths

- Comprehensive coverage of classical econometric estimators.
- Excellent time series tooling with state space models and Kalman filtering.
- Strong integration with pandas.

### Limitations

- No dynamic programming or structural economic modeling.
- Structural estimation (SMM, SMLE) is not natively supported.

## 2.5 Dynare

Dynare is the industry standard for DSGE model estimation. It provides a complete workflow from model specification to Bayesian estimation, including impulse response analysis, variance decomposition, and optimal policy computation.

### Strengths

- Extremely mature and battle-tested for DSGE estimation.
- Handles complex multi-equation systems reliably.
- Well-documented model specification language.

### Limitations

- MATLAB-centric ecosystem creates licensing costs and portability barriers.
- Not Pythonic; no natural integration with data science workflows.
- No support for heterogeneous agent models.

## 2.6 Julia Ecosystem (Dolo.jl, QuantEcon.jl, Agents.jl)

Julia's ecosystem offers excellent computational economics tools. QuantEcon.jl mirrors the Python QuantEcon library with better performance. Dolo.jl provides model specification in Julia. Agents.jl supports agent-based modeling.

### Relevance to Dynecon

The Julia ecosystem demonstrates that high-performance computational economics is achievable. Dynecon should offer Python APIs that are competitive with Julia performance via JIT compilation backends, while remaining in the Python ecosystem that most economists already use. Long-term, Dynecon should support a shared model specification format with Dolo.jl.


# Section 3 — High-Level Architecture

## 3.1 Architectural Principles

Dynecon rests on four principles:

1. **Layered abstraction.** The user-facing API is separated from the numerical computation underneath it. The Python surface is permanent; the execution layer beneath it grows over time (Section 19).
2. **Backend portability.** Every performance-critical operation is registered with a backend dispatcher under an `(operation, backend)` key. NumPy is the reference and the universal fallback; Numba and JAX register faster implementations of the same operations. C and Rust layers are added after 1.0 and only where profiling justifies them.
3. **Function-and-result API.** Users call functions that take a model and keyword options and return an immutable result object: `solve(model, ...) -> Solution`, `simulate(solution, ...) -> SimulationResult`, `smm(model, ...) -> SMMResult`. Options are validated against a frozen internal configuration record that is stored on the result so every run can be reproduced. There are no estimator or solver classes with a mutable `fit()`; one way of doing a thing is a design rule, not an accident.
4. **Lean by construction.** One concept has one implementation. Nothing is written without a caller and a test, nothing old is kept "for compatibility" before 1.0, and nothing is added in case it is needed later. A capability that is deferred returns together with its first consumer. This rule is enforced by tooling (vulture, ruff, and two usage-checking scripts run by `make deadcode`) and is stated in full as rule R21 of the execution plan.

Protocols (structural interfaces) are used only where there are at least two implementations inside the v1.0 scope. There is exactly one: `Solver`, implemented by VFI, EGM, perturbation and the equilibrium engine. Everything else is a concrete type.

## 3.2 Module Map (v1.0)

| Module | Responsibility | Dependencies |
| --- | --- | --- |
| `dynecon.core` | Error hierarchy, `Solution` result type | NumPy |
| `dynecon.utils` | `jit` decorator with pure-Python fallback, RNG contract, input validation, provenance record | NumPy |
| `dynecon.config`, `dynecon.backends` | Validated backend selection, strict mode, availability and introspection | none |
| `dynecon.numerics` | Grids, 1-D linear interpolation, quadrature, AR(1) discretisation, Markov-chain tools, the fixed-point iteration driver, numerical Jacobians; the backend dispatcher | NumPy, SciPy; Numba and JAX optional |
| `dynecon.model` | Declarative model specification, three problem templates, validation, immutable compiled models, fingerprint | core, numerics |
| `dynecon.solvers` | `solve()`; VFI, EGM, stationary-equilibrium engine, first-order perturbation, diagnostics | numerics, model |
| `dynecon.simulation` | Time-series and panel simulation, Young histogram distributions, aggregates, inequality measures, impulse responses | numerics, solvers |
| `dynecon.estimation` | SMM, MLE, Kalman filter likelihood, Metropolis–Hastings, priors and transforms, identification diagnostics | model, simulation, solvers |
| `dynecon.models` | Nine built-in canonical models with a uniform interface | model, solvers |
| `dynecon.data` | Macro filters (HP, Baxter–King, Hamilton) and FRED download | NumPy, SciPy; pandas optional |
| `dynecon.viz` | Matplotlib plot functions | matplotlib (optional extra) |
| `dynecon.experiment` | Parameter sweeps, experiments with provenance, policy experiments and welfare, Monte Carlo and bootstrap, elasticities | all modules |

Runtime dependencies are NumPy and SciPy. Numba, JAX, pandas and matplotlib are optional extras, and `import dynecon` works without any of them. There is no SymPy dependency in v1.0 (Section 6.4).

## 3.3 Layered Architecture

**Layer 1 — Core.** The error hierarchy and the `Solution` result type, plus the small utilities every other module uses. The core layer may depend on NumPy but on nothing else.

**Layer 2 — Computational backend.** The `numerics` module implements the low-level operations: grids, interpolation, quadrature, discretisation, Markov tools, the fixed-point driver and Jacobians. Operations on hot paths (called inside an iteration loop or per agent) have a NumPy reference and a Numba implementation; JAX implementations are added for the operations that benefit from autodiff or vectorisation. Cold-path operations (setup, one-off linear algebra) are NumPy or SciPy only. This layer is the only place where backend-specific code lives.

**Layer 3 — Solvers and simulation.** Solvers and the simulation engine are built on the numerics layer. They operate on compiled models and `Solution` objects and know nothing about particular model implementations: a solver reads the ingredients it needs from the model's problem template.

**Layer 4 — User-facing API.** The top-level `dynecon` namespace re-exports a curated list of functions and types (`solve`, `simulate`, `smm`, the model builder, and so on). The exact list is the content of `API_POLICY.md` and is frozen at 1.0.

## 3.4 Module Communication

Modules depend on each other only in the direction shown in the module map, and only through public names. The data flow is linear:

Model specification → compilation → `solve()` → `Solution` → `simulate()` → `SimulationResult` → moments → estimation, visualisation and experiments.

Every object in the chain is immutable and carries the fingerprint of the model that produced it. `Solution` persistence (`save`/`load`, as `.npz` arrays plus JSON metadata) arrives with the `Experiment` module, its first consumer; HDF5 is not used in v1.0.

# Section 4 — Core Abstractions

## 4.1 Design Philosophy

The abstractions are chosen to be few and to be the ones solvers actually consume. Where earlier drafts of this document described a rich object zoo (Agent, Equation, Constraint, Simulation protocol), v1.0 keeps only what has a caller. Each removal is recorded in the execution plan with the reason, and each item returns when a feature needs it.

## 4.2 Fundamental Objects

### Model and CompiledModel

A `Model` is a mutable builder. `compile()` validates it, discretises its shock processes, evaluates the problem-template callables on a small test input, and returns an immutable `CompiledModel`. Changing a parameter produces a new object: `compiled.with_params(beta=0.9)` leaves the original untouched. A compiled model can describe itself (`describe()`) and produce a SHA-256 fingerprint over its parameters, grids, shock matrices and the source text of its template callables; the fingerprint is stored on every `Solution` and used by `Experiment` to detect changes.

| Attribute | Description |
| --- | --- |
| states | Named state variables with bounds and grid specification; the lower bound of the asset state is the borrowing limit |
| controls | Named choice variables |
| shocks | Discretised shock processes `(values, P)` |
| parameters | `ParameterSet` with values, bounds, free/fixed status; priors and transforms are attached by the estimation module |
| problem | One of the three problem templates (Section 6.3) |

### Problem templates

A template is the contract between a model and a family of solvers. It replaces the free-form "equation" objects of earlier drafts: nothing in the solvers could call an arbitrary equation, so templates make the needed ingredients explicit. The templates are `saving`, `finite_dp` and `dsge`.

### State, Control, Shock

`State` is a named variable with bounds, a grid type (uniform or curved) and optionally a shock process. `Control` names a choice variable and is the key under which the solver stores its policy. There are three shock types: `AR1Shock` (discretised by Rouwenhorst or Tauchen, optionally in logs), `MarkovShock` (user-supplied values and transition matrix) and `IIDShock` (normal, lognormal or uniform, discretised by Gauss–Hermite or Gauss–Legendre nodes). All three end up as a `(values, P)` pair on the compiled model.

### Parameter

`Param` is a named scalar with a value, bounds and a free/fixed flag. `ParameterSet` provides the vector view (`vector`, `from_vector`, `bounds`, `names`) that estimation uses. Priors and parameter transforms (`none`, `log`, `logit`) are added in the estimation module, where they are first used.

### Solution

`Solution` is the one result type of every solver. Its fields are `method`, `converged`, `iterations`, `residual`, `elapsed`, `backend`, `model_name`, `params`, `grids`, `policy`, `value`, `diagnostics` and `metadata` (software versions, model fingerprint, resolved configuration). `summary()` prints a human-readable report. The equilibrium engine returns an `EquilibriumSolution` subclass that adds price, aggregates and the stationary distribution. Policy functions are evaluated through `Solution.policy_function(name)`, an interpolating callable.

### SimulationResult

The output of `simulate()`: paths of states, controls and shocks as arrays of shape `(T, N)`, the seed entropy and the solution fingerprint. It offers `.moments(names)`, `.mean(name)`, `.summary()` and `.to_dataframe()` (long format, pandas optional).

### Experiment

An `Experiment` is a named run built from a built-in model name plus parameter overrides, or from an importable builder function. It writes its configuration, provenance (package versions, git commit, platform, model fingerprint), solution and results to a run directory, and `Experiment.reproduce(path)` re-runs it and compares the numbers.

### What is not in v1.0

There is no `Agent` class: the problem templates play that role for single-agent and heterogeneous-agent models alike. A multi-agent abstraction returns when a model with strategically interacting agents is added. There is no general `Constraint` object: a borrowing limit is the lower bound of the asset state.

# Section 5 — Numerical Computation Layer

## 5.1 Overview

The numerics module provides the mathematical infrastructure used by solvers and simulation. Operations are classified as hot path or cold path. A hot-path operation has a NumPy reference implementation and a Numba implementation registered in the dispatcher, and a parity test proves that they agree. A cold-path operation is written once in NumPy or SciPy. SciPy is used wherever it does the job at reference quality; Dynecon writes its own numerical code only where compilation or domain semantics require it.

## 5.2 Interpolation

**v1.0.** One-dimensional linear interpolation, NumPy and Numba, with `"linear"` extrapolation or clamping. This is all v1.0 needs: every v1.0 model has a single continuous endogenous state plus discrete or discretised shocks, so policies are interpolated along one axis.

**After 1.0.** Multilinear interpolation in two or three dimensions returns with the Krusell–Smith solver, which is its first consumer. Cubic splines and Chebyshev approximation return with projection methods. Smolyak sparse grids, Gaussian-process regression and neural-network approximation are longer-term options.

## 5.3 Integration, Discretisation and Markov Tools

- Gauss–Hermite quadrature (normal shocks) and Gauss–Legendre quadrature (uniform shocks).
- Tauchen and Rouwenhorst discretisation of AR(1) processes (Rouwenhorst is the default because it stays accurate for persistent processes).
- Markov-chain tools: exact stationary distribution and chain simulation, both validated against closed forms.

Gauss–Laguerre rules, quasi-random sequences and Tauchen–Hussey are not in v1.0 because nothing calls them.

## 5.4 Iteration, Root Finding and Optimisation

- **One fixed-point driver.** `numerics.fixed_point(operator, x0, *, tol, max_iter, norm, damping, callbacks)` is the only convergence loop in the library. VFI, EGM, the Lucas-tree price functional and the stationary-distribution iteration all call it. Non-convergence emits `ConvergenceWarning` and sets `converged=False`; in strict mode it raises `ConvergenceError`.
- **Root finding** uses SciPy: `scipy.optimize.brentq` for the one-dimensional equilibrium price, `scipy.optimize.root` for steady states.
- **Optimisation** uses `scipy.optimize.minimize` for SMM and MLE. There is a single optimiser path.
- **Jacobians** are central finite differences in NumPy; the JAX backend adds exact autodiff Jacobians cross-checked against the finite-difference ones.

Anderson acceleration, bespoke Newton solvers and Brent or golden-section maximisation are not in v1.0. Value function iteration is grid-based, so no continuous refinement is needed; those methods return if profiling or a new model asks for them.

## 5.5 Grid Construction

Uniform grids and curved grids built by an exponential map (`x = lo + (hi − lo)·((e^{c·u} − 1)/(e^{c} − 1))`, with `u` uniform on `[0,1]` and a curvature parameter `c`), which put more nodes near the borrowing constraint where policies bend most.

## 5.6 Backend Dispatch

Backend selection is validated. An unknown name raises `ValidationError`; a known backend that is not installed raises `BackendUnavailableError` in strict mode and otherwise issues one warning and falls back to NumPy.

```python
import dynecon as ek

ek.config.set_backend("numba")  # validated; falls back or raises per strict mode
ek.backends.available_backends()  # e.g. ['numpy', 'numba']
ek.backends.describe()  # operation x backend table plus library versions
sol = ek.solve(model, backend="numpy")  # per-call override
```

# Section 6 — Model Specification System

## 6.1 Design Goals

The model system must be expressive enough for the nine v1.0 models and any model of the same families, readable by an economist, and honest: a model is declared by supplying the ingredients a solver needs, not by writing arbitrary equations that nothing can use. It must also keep users away from compiler constraints. Template callables are ordinary NumPy functions that broadcast over arrays. They run outside the compiled kernels: the kernels receive arrays computed from them, so a user never writes Numba-compatible code.

## 6.2 Problem Templates

**`saving`** — one continuous endogenous asset, discrete or discretised shocks, one control. This single template covers consumption–savings, Brock–Mirman, RBC with inelastic labour, and the Aiyagari and Huggett households. Growth models and consumption–savings models are the same template; they differ only in the `resources` function (cash on hand as a function of end-of-period assets and the shock).

**`finite_dp`** — finite states and actions, given by a reward array `[state, action]` and a transition array `[action, state, next_state]`. Used by the McCall job-search model.

**`dsge`** — a system of expectational equations `E_t f(y_{t+1}, y_t, ε_{t+1}; θ) = 0` over predetermined and jump variables. Used by the first-order perturbation solver and by the New Keynesian model.

## 6.3 Example: Consumption–Savings Model

```python
import dynecon as ek
from dynecon.model import Model, Param, State, Control, AR1Shock

model = Model(name="consumption_savings")

model.params(
    beta=Param(0.96, bounds=(0.8, 0.999), description="Discount factor"),
    sigma=Param(2.0, bounds=(0.5, 5.0), description="CRRA coefficient"),
    r=Param(0.03, bounds=(-0.05, 0.1), description="Net interest rate"),
)

model.add_state(
    State("assets", bounds=(0.0, 50.0), grid_type="exp", n_points=200)
)  # lower bound = borrowing limit
model.add_state(State("income", shock=AR1Shock(rho=0.9, sigma_e=0.1, n_states=7, log=True)))
model.add_control(Control("consumption"))

model.set_problem(
    "saving",
    asset="assets",
    shock="income",
    control="consumption",
    utility=lambda c, p: c ** (1 - p.sigma) / (1 - p.sigma),
    marginal_utility=lambda c, p: c ** (-p.sigma),
    inverse_marginal_utility=lambda x, p: x ** (-1 / p.sigma),
    resources=lambda a, z, p: (1 + p.r) * a + z,
    marginal_return=lambda a, z, p: 1 + p.r,
    discount="beta",
)

compiled = model.compile()  # validates, discretises, returns an immutable CompiledModel
sol = ek.solve(compiled)  # method="auto": EGM when the marginal fields are given, else VFI
```

*Implementation note (updated).* The repository builder uses `add_state(State(...))` and `add_control(Control(...))`; the curved grid type is `"exp"`. The object names (`Model`, `Param`, `State`, `Control`, `AR1Shock`), `set_problem`, `compile` and `solve` are fixed by the execution plan. The two builder calls that attach states and controls are written here in the form of the current `Model` builder; execution-plan step P4.10 keeps the existing builder methods unchanged ("no fluent aliases"), so if their exact spelling differs from this sketch, the repository's builder is authoritative and this example is corrected.

The same template describes a growth model by changing two lines: for Brock–Mirman, `resources = z * a**p.alpha` and `marginal_return = p.alpha * z * a**(p.alpha - 1)`.

Compilation collects all validation problems (missing fields, invalid stochastic matrices, parameters outside their bounds, wrong output shapes from a callable) and raises them together as one `ModelSpecificationError` that names the offending template field.

## 6.4 What is deferred

YAML model files, a callable library, and `from_dict` constructors are not in v1.0: `Experiment` reproduces a run from a built-in model name plus parameter overrides, or from an importable builder, which needs no serialisation format. A SymPy symbolic front end is deferred as well. Numerical Jacobians and JAX autodiff cover v1.0 without a second compilation path to test. Both return when a feature needs them.

# Section 7 — Solver Framework

## 7.1 Architecture

There is one entry point:

```python
solve(model, method="auto", *, backend=None, **options) -> Solution
```

Solvers register in a `SOLVERS` table and implement the `Solver` protocol (`__call__(model, **options) -> Solution`). `method="auto"` picks EGM for a `saving` problem that supplies `marginal_utility`, `inverse_marginal_utility` and `marginal_return`, VFI for other `saving` problems, and VFI for every `finite_dp` problem. The resolved method, backend and options are recorded on the `Solution`. Options are checked against a frozen configuration record, so a misspelled option is an error rather than silently ignored.

## 7.2 Value Function Iteration

- A precomputed payoff tensor `U[i, j, k]` (assets `i`, shock `j`, next assets `k`), so the Bellman maximisation is an array operation rather than a Python loop.
- A vectorised NumPy implementation (the reference) and a Numba kernel (`bellman_saving`) that exploits monotonicity of the policy in assets.
- Howard policy-function acceleration (several policy-evaluation sweeps per maximisation).
- Iteration runs on the shared `fixed_point` driver, preserving relaxation.

Multigrid acceleration is deferred (Part E of the plan).

## 7.3 Endogenous Grid Method

EGM (Carroll 2006) solves every `saving` problem without root finding. Given next-period consumption, the Euler equation is inverted on a grid of end-of-period assets to recover consumption and the endogenous cash-on-hand grid; the result is interpolated back to the exogenous grid. The general form uses the model's `resources` and `marginal_return`, so it applies to consumption–savings, Brock–Mirman and RBC alike. A constrained segment (`c = m − a_min`) handles the borrowing limit. The only compiled operation is `linear_interp_1d`. Euler-equation errors are reported as a diagnostic on every EGM solution.

Time iteration (Coleman–Reffett) is not built in v1.0: EGM solves the same problems without root finding. Time iteration returns for models EGM cannot handle, such as endogenous labour supply. Discrete-choice EGM with an upper envelope is also post-1.0.

## 7.4 Perturbation

First-order perturbation for `dsge` problems, following Klein (2000): a steady-state finder (`scipy.optimize.root`), linearisation by numerical Jacobians (exact JAX autodiff once the JAX backend lands), generalised Schur decomposition (`scipy.linalg.ordqz`) with stable eigenvalues ordered first, a Blanchard–Kahn determinacy check (`DeterminacyError` for indeterminacy or no stable solution), impulse responses and theoretical moments through a discrete Lyapunov solution. Second-order perturbation is post-1.0.

## 7.5 Stationary Equilibrium

A generic engine for heterogeneous-agent stationary equilibria: given a model factory that builds the household problem at a price, solve the household problem (warm-started), compute the stationary distribution, evaluate excess demand, and find the clearing price with `scipy.optimize.brentq`. It powers the Aiyagari capital market and the Huggett bond market. It raises `ConvergenceError` with the endpoint excess demands when no sign change is bracketed. The Krusell–Smith algorithm and MIT-shock transition paths are post-1.0.

## 7.6 Projection Methods

Chebyshev collocation, Galerkin methods and finite elements are post-1.0 and return together with cubic splines and Chebyshev approximation in the numerics layer.

# Section 8 — Simulation Engine

## 8.1 Overview

The simulation engine takes a `Solution` and generates time series, panels, stationary distributions and impulse responses. Policies are evaluated by linear interpolation in the continuous state. For perturbation solutions, `simulate` runs a linear simulation instead.

## 8.2 Time Series and Panel Simulation

```python
sim = ek.simulate(solution, n_periods=1000, n_agents=500, burn_in=100, rng=42)
sim.mean("consumption")
sim.moments(["mean_c", "var_c", "autocorr_c"])
sim.to_dataframe()  # long format with `period` and `agent`; needs the pandas extra
```

All shock indices are drawn up front in NumPy from per-agent spawned random streams, then a deterministic Numba kernel evaluates the policy, with `prange` over agents. Results therefore do not depend on the thread count.

## 8.3 Distributions

### Young (2010) histogram method

The distribution is a mass `μ[i, j]` on the (asset, shock) grid. For each policy value `a'`, mass is split between the two neighbouring grid points with weights that preserve the policy mean exactly, then mixed across next-period shocks with the transition matrix. This is noise-free, and the stationary distribution is obtained by iterating on the shared `fixed_point` driver. The sparse transition matrix is the NumPy/SciPy reference; a Numba kernel applies the same lottery without building the matrix and is validated against it.

### Aggregates and inequality

Means, marginals, CDFs, quantiles, Lorenz curves and Gini coefficients, computed exactly from the histogram.

### Particle simulation

Panel simulation with many agents is the particle method. Chunked simulation and memory-mapped output are not in v1.0: `simulate` stores only the variables named in `store=[...]`, which keeps the largest benchmark case within a few hundred megabytes.

## 8.4 Impulse Responses

`simulate_irf` computes generalised impulse responses of nonlinear models by paired simulations (shocked minus unshocked with common random numbers). For linear (perturbation) models, impulse responses come directly from the solution matrices.

## 8.5 Statistical Helpers

Sample moments (mean, standard deviation, autocorrelation, cross-sectional percentiles, correlation, regression slope), block-bootstrap standard errors and a Newey–West variance helper, shared by estimation and the experiment module.

# Section 9 — Econometric / Estimation Tools

## 9.1 Simulated Method of Moments (SMM)

SMM is a function that returns an immutable result, in the same style as `solve`:

```python
result = ek.smm(
    model,
    free_params=["beta", "sigma"],
    data_moments=m_hat,
    moment_fn=moment_fn,
    solve_options={},
    sim_options={"n_periods": 5000},
    weights="optimal",
    n_sim=20,
    rng=0,
)
print(result.summary())
```

For each candidate parameter vector the model is recompiled, solved, and simulated with common random numbers (the same shock draws at every evaluation), and the quadratic loss between model and data moments is minimised with `scipy.optimize.minimize`. Estimation is two-step: an identity (or diagonal) weighting matrix first, then the optimal matrix from the first-step estimate. Standard errors use the Jacobian of the moments and include the simulation-noise correction `(1 + 1/S)`; the Hansen J-test is reported for over-identified systems. Inner solves that fail to converge are replaced by a penalty and counted in `result.diagnostics["failed_evals"]`.

**Calibration is exactly identified SMM.** With as many moments as free parameters the minimum loss is zero and the estimate hits the targets, so no separate calibration API exists.

## 9.2 Maximum Likelihood

- `mle(loglik, theta0, bounds, method="L-BFGS-B")` with a numerical Hessian and standard errors from the inverse information.
- A Kalman filter for linear Gaussian state-space models gives the likelihood of a first-order perturbation solution, which makes likelihood-based DSGE estimation possible without further machinery.
- Gradients come from JAX autodiff when the JAX backend is installed.
- CCP-based and simulated MLE are post-1.0.

## 9.3 Bayesian Estimation

- Priors: Normal, Uniform, Beta, Gamma and Inverse-Gamma, each with a log-density, defined in `dynecon.estimation.priors`.
- Parameter transforms (`none`, `log`, `logit`) map bounded parameters to the real line; SMM, MLE and MCMC all use this one convention.
- Adaptive random-walk Metropolis–Hastings (Haario et al. covariance adaptation) with several chains.
- Diagnostics: split R-hat, effective sample size, acceptance rate, a posterior summary table.
- Hamiltonian Monte Carlo via NumPyro, Bayes factors and WAIC are post-1.0.

## 9.4 Identification Diagnostics

`identification_report(smm_result)` reports the rank and condition number of the moment Jacobian and its singular values, flags weakly identified parameters, and raises `IdentificationError` when the Jacobian is rank-deficient. A loss-profile helper evaluates the objective along one parameter.

# Section 10 — Built-In Model Library

## 10.1 Canonical Models (v1.0)

Nine models ship in v1.0 behind a uniform interface: `dynecon.models.available()` lists them and `dynecon.models.load(name, **param_overrides)` returns an uncompiled, editable `Model`. Each model module defines its parameters with bounds, a `build` function, the citation for its calibration, and an `ANALYTIC` dictionary of closed-form results used by tests and documentation. The registry's `register(name, builder)` is the same function the built-ins use, so users add their own models the same way; there is no separate plugin mechanism.

| Model | Template | Category | Analytic benchmark used in tests |
| --- | --- | --- | --- |
| Consumption–savings (income fluctuation) | `saving` | Incomplete markets, partial equilibrium | Permanent-income solution (`a' = a` when `β(1+r) = 1`) |
| Brock–Mirman growth | `saving` | Representative agent | `k' = αβ z kᵅ`, `c = (1 − αβ) z kᵅ` |
| Real business cycle (inelastic labour) | `saving` | Representative-agent DSGE | Steady state `k/y = α/(1/β − 1 + δ)` |
| McCall job search | `finite_dp` | Dynamic discrete choice | Reservation wage from the closed-form quadratic for uniform offers |
| Aiyagari | `saving` + equilibrium engine | Heterogeneous-agent GE | `r* < 1/β − 1` |
| Huggett | `saving` + equilibrium engine | Heterogeneous-agent GE, bond market | `r* < 1/β − 1` |
| Two-period OLG (Diamond) | exact law of motion | Overlapping generations | `k* = [β(1−α)/((1+β)(1+n))]^{1/(1−α)}` |
| New Keynesian (three equation) | `dsge` | Monetary DSGE | Closed form with `φₓ = 0`; determinacy condition `κ(φ_π − 1) + (1 − β)φₓ > 0` |
| Lucas tree | fixed-point on a price functional | Asset pricing | Log utility: `p/y = β/(1 − β)` |

## 10.2 Post-1.0 Models

Mortensen–Pissarides search and matching, the multi-period Auerbach–Kotlikoff OLG model, the Merton portfolio problem (continuous time), a Krusell–Smith economy with aggregate risk, a two-asset HANK model, a dynamic discrete-choice labour-supply model and Eaton–Kortum trade. Each is added together with the solver or machinery it needs (Section 16).

## 10.3 Model Documentation Standard

Each built-in model has: a page with the derivation and calibration notes (with citations); a tutorial notebook that executes in the docs build and ends with an assertion on a key number, so a broken tutorial fails the build; a test against the analytic benchmark above where one exists; and an optional cross-check against QuantEcon, run in a separate CI job and never a dependency.

# Section 11 — Data Handling

## 11.1 Design Philosophy

Dynecon does not wrap data libraries. pandas already provides indexed series and panels, readers and writers, and table export, so v1.0 contains no `EconTimeSeries`, `EconPanel`, CSV or Parquet wrappers and no synthetic-data generators. Functions take array-likes, return arrays, and preserve the index when given a `pandas.Series`. The core library imports without pandas.

## 11.2 v1.0 Contents

- **Filters:** Hodrick–Prescott, Baxter–King and Hamilton filters for detrending.
- **FRED:** download and parsing of a series from the St. Louis Fed API into a `pandas.Series` (optional extra).
- **Export:** `SimulationResult.to_dataframe()` provides the long-format panel; LaTeX and Markdown tables come from pandas' own `to_latex` and `to_markdown`.

## 11.3 After 1.0

First-class Polars support (and with it wrapper classes, if they would add behaviour), panel transformations, and loaders for PSID, SIPP, BHPS, CEX, SCF, Penn World Tables and the World Bank. Each loader is added when a tutorial or model needs it.

# Section 12 — Visualization

## 12.1 Design Philosophy

Plots are plain functions built on matplotlib (an optional extra), publication-ready by default, and each is used by at least one tutorial or model page so none is dead. Importing `dynecon` never imports matplotlib.

## 12.2 The v1.0 Plot Set

`plot_policy`, `plot_value`, `plot_euler_errors`, `plot_convergence`, `plot_simulation`, `plot_irf`, `plot_distribution`, `plot_lorenz`, moment-fit, MCMC trace and posterior plots, sweep plots and equilibrium (excess-demand) plots. Comparative statics are drawn by passing several solutions to the same plot call.

## 12.3 After 1.0

Plotly backends, animated distribution evolution and the Jupyter dashboard with parameter sliders are post-1.0 (Section 16).

# Section 13 — Experiment and Research Workflow

## 13.1 Parameter Sweeps

```python
sweep = ek.ParameterSweep(
    model=compiled,
    grid={"beta": np.linspace(0.90, 0.99, 10), "sigma": [1.0, 2.0, 5.0]},
    function=lambda sol: {"mean_assets": ...},
    n_jobs=-1,
)  # process-based parallelism, no extra dependency
result = sweep.run()  # SweepResult: results in grid order; failed points are recorded, not fatal
table = result.to_dataframe()  # one row per grid point
```

Each point is `with_params` → `solve` → the user function. With `backend="jax"` and `method="egm"` the whole grid is solved in one vectorised batch (`solve_many`).

## 13.2 Reproducible Experiments

```python
exp = ek.Experiment(
    "aiyagari_baseline",
    build=("aiyagari", {"beta": 0.95}),  # built-in model name + overrides, or "module:function"
    solve_options={"tol": 1e-9},
)
result = exp.run(rng=123)  # writes runs/aiyagari_baseline/<timestamp>-<fingerprint8>/
report = ek.Experiment.reproduce(
    "runs/aiyagari_baseline/<run directory>"
)  # re-runs, compares numerically
```

A run directory holds `config.json`, `provenance.json` (package versions, git commit, platform, model fingerprint), `solution.npz`, `result.json` and a log. The reproducibility contract is explicit about its limits: the same machine, library versions, backend and thread count.

## 13.3 Policy Experiments and Welfare

`PolicyExperiment(baseline_model, reform)` compares a baseline and a counterfactual (a reform is a dict of parameter overrides or a function from model to model) and reports aggregates, distributions and welfare. Welfare is measured as consumption-equivalent variation λ: the proportional change in baseline consumption that makes the household indifferent between regimes. For EGM solutions, which carry no value function, the value is computed from the policy as `V = (I − βT)⁻¹ u` using the policy-induced transition matrix.

## 13.4 Monte Carlo, Bootstrap and Elasticities

`monte_carlo` runs independent, reproducible replications on spawned random streams; `bootstrap_ci` gives percentile intervals (moving-block when a block length is supplied); `elasticities` are central differences of a metric with respect to a parameter, built on `ParameterSweep`.

# Section 14 — Performance Design and the Language Reality

## 14.1 An Honest Assessment of Python's Performance Ceiling

Python's Global Interpreter Lock and interpreted execution mean pure-Python code is typically 10x to 1000x slower than equivalent compiled code, depending on the workload. The inner loops of value function iteration (a policy iterated over a grid with thousands of points, repeated until convergence) are exactly where Python is worst, so an idiomatic-Python VFI is too slow for serious research use. This is a consequence of how the language works, not a fixable bug.

A second, subtler issue is control over dependencies. Building on NumPy, SciPy, pandas or matplotlib means inheriting their design decisions. Abstraction layers hide rough edges but add their own complexity.

These realities do not make Python the wrong language. They mean Dynecon is architected as a two-language system from the beginning: Python for the surface API and orchestration, compiled code for the computation layer. This is a founding principle, not a future plan.

## 14.2 The Two-Language Doctrine

- The Python layer is where economists think: model specification, experiments, visualisation, data handling and all user-facing interaction. It is the permanent contract with users.
- The compiled layer is where machines work: Bellman maximisation, interpolation kernels, simulation and histogram updates. It is never directly visible to users and is replaceable without changing user code.

This is how NumPy (Python over C), PyTorch (Python over C++/CUDA) and other scientific Python libraries are built.

## 14.3 Why Python Remains the Right Surface Language

- Ecosystem: most economists already work in Python.
- ML and autodiff infrastructure (JAX, PyTorch) is Python-native.
- Data tools (pandas, polars, xarray) have no equivalents of comparable maturity elsewhere.
- Readability as a research value: a Bellman equation in Python is legible to any economist.
- Compiled layers give control over performance-critical paths without changing the surface.

## 14.4 Vectorisation as the NumPy Floor

Before any compiled backend, Dynecon vectorises: the Bellman maximisation is an array operation over a precomputed payoff tensor, and EGM is array arithmetic plus one interpolation. Pure-NumPy Dynecon is fast enough for teaching and small experiments with no optional dependencies.

## 14.5 The Numba JIT Layer (v1.0)

Numba is active in v1.0 for the hot paths: the Bellman maximisation, linear interpolation, the simulation step and the histogram update. `dynecon.utils.jit` wraps `numba.njit` with a pure-Python fallback when Numba is absent, caching, and optional `parallel=True`. The following sketch shows the shape of the VFI kernel (schematic, not the final code):

```python
@jit(parallel=True)
def bellman_saving(U, EV, V_new, policy):
    # U[i, j, k]: payoff at assets i, shock j, next assets k (-inf if infeasible)
    # EV[k, j]  : beta * E[V(a'_k, z') | z_j], computed outside the kernel in NumPy
    n_a, n_z, _ = U.shape
    for j in prange(n_z):
        k_lo = 0
        for i in range(n_a):  # monotone policy: search from the previous optimum
            best, arg = -np.inf, k_lo
            for k in range(k_lo, n_a):
                v = U[i, j, k] + EV[k, j]
                if v > best:
                    best, arg = v, k
            V_new[i, j] = best
            policy[i, j] = arg
            k_lo = arg
```

Because user callables are evaluated outside the kernel, the kernel only ever sees float64 arrays. Parallelism never changes results: random numbers are drawn in NumPy before the deterministic kernels run.

## 14.6 The JAX Functional Backend (v1.0)

JAX adds exact derivatives, hardware portability (CPU/GPU) and `vmap`. In v1.0 it provides real implementations of the hot operations (replacing an earlier pilot that merely wrapped baseline behaviour), 64-bit precision enabled on first use (documented, with no opt-out because every tolerance assumes float64), exact Jacobians for perturbation cross-checked against finite differences, gradients for SMM and MLE, and vectorised parameter sweeps.

## 14.7 Memory Management

`simulate` stores only the variables the caller asks for. Chunked simulation, memory-mapped arrays and HDF5 storage are not in v1.0 because the largest benchmark case (10⁵ agents × 500 periods) fits comfortably in memory; they return when a model needs more.

## 14.8 Measured, Not Claimed

Speed claims come only from a recorded benchmark run. A harness in the repository runs fixed cases on each backend, reports compile time separately from steady-state time, and verifies that backends return the same answer. Results are committed as JSON, and the README quotes only numbers from that record. The v1.0 goals are comparisons, not promises: Numba Bellman ≥ 20× vectorised NumPy; EGM ≥ 20× Numba VFI; Numba panel simulation ≥ 10× a NumPy loop; Numba histogram step ≥ 3× the sparse-matrix iteration. A missed target is explained with profiler evidence rather than edited away. Timing is never a required CI check.

# Section 15 — Software Engineering Structure

## 15.1 Repository Layout

```
Dynecon/
├── dynecon/
│   ├── __init__.py          # curated exports (API_POLICY.md)
│   ├── config.py  backends.py
│   ├── core/                # errors, Solution
│   ├── utils/               # jit, rng, validation, provenance
│   ├── numerics/            # grids, interpolation, integration, discretization, markov,
│   │   │                    # fixedpoint, jacobian
│   │   └── backends/        # *_numpy.py, *_numba.py, *_jax.py
│   ├── model/               # specs, shocks, base, compiled, problems, validation, fingerprint
│   ├── solvers/             # api, vfi, dp/, egm, equilibrium, perturbation/, diagnostics
│   ├── simulation/          # simulate, distribution, moments, result
│   ├── estimation/          # params, priors, moments, smm, mle, kalman, bayes, identification
│   ├── models/              # registry + nine built-in models
│   ├── data/                # filters, fred
│   ├── viz/                 # plot functions
│   └── experiment/          # sweep, experiment, policy, welfare, montecarlo
├── tests/                   # per-domain folders, plus api/, models/, golden/, engineering/
├── benchmarks/              # harness, cases, recorded results
├── docs/                    # Sphinx site, tutorials, theory, model pages, md/ and latex/
├── scripts/                 # repository checks, phase extractor, benchmark comparison
├── .github/workflows/       # ci, docs, nightly, release
└── pyproject.toml  Makefile  CHANGELOG.md  CITATION.cff  CONTRIBUTING.md  README.md  LICENSE
```

There are no `cext/` or `rsext/` directories in v1.0. They are created when the first compiled kernel of that kind is justified by a benchmark.

## 15.2 Testing Strategy

- **Analytic benchmarks first.** Wherever a model has a closed form (Brock–Mirman, permanent income, McCall, Diamond OLG, the NK solution with `φₓ = 0`, log-utility Lucas tree), tests compare the solver with it. The catalogue is an appendix of the execution plan.
- **Cross-backend parity.** Every hot-path operation has a parametrised test that checks each registered backend against the NumPy reference; tests skip a backend that is not installed, and CI runs with and without Numba.
- **Property-based tests** (Hypothesis) for interpolation (exact at nodes), Markov tools (rows sum to one), the distribution lottery (mass and mean preserved) and perturbation.
- **Golden-file regression tests** for solved policies, refreshed deliberately with `pytest --update-golden`.
- **Determinism tests:** results do not depend on thread count.
- **Coverage floors:** at least 90% for `core`, `numerics`, `model` and `solvers`, at least 80% elsewhere.
- **Dead-code gate:** `make deadcode` runs vulture, a ruff rule set (unused code, commented-out code, unused arguments, complexity) and two scripts that check that every public symbol has a caller and a test, and that any code tagged `# NEAR-FUTURE(P<n>.<m>)` has a live consumer step within the next review window.

## 15.3 CI/CD Pipeline

- GitHub Actions: Python 3.11 and 3.12 on Linux, macOS and Windows; 3.13 as a non-blocking job; a job without Numba; a JAX job; a docs build; a nightly benchmark run that is never a gate.
- Ruff (lint and format) and mypy (untyped definitions disallowed), run locally through pre-commit and in CI.
- Coverage measured with Numba installed.
- Release: build, clean-room install test, TestPyPI, then PyPI, with a human gate before each upload.
- Supply-chain hygiene before release: Dependabot, pinned CI actions, a non-gating `pip-audit` job and trusted publishing.

## 15.4 Versioning

Semantic versioning. Before 1.0 there are no external users, so the API changes freely and old paths are deleted in the same commit as their replacement; there are no compatibility shims. From 1.0, breaking changes require a major release after a deprecation window of at least two minor versions. The deprecation helper is written when the first deprecation happens. `API_POLICY.md` defines what is public.

# Section 16 — Development Roadmap

## 16.0 Philosophy: Continuous Layered Development

Dynecon is not designed as a product with a launch point and a later pivot. Each stage deepens the architecture rather than replacing it. The Python API established in version 1.0 is the API in later versions; what changes beneath it is the compiled execution engine. There is no moment at which Dynecon "becomes serious" by switching languages, because the full layered design (Section 19) is present from the first release.

Two documents divide the work. The **execution plan** (`V1_EXECUTION_PLAN.md`) is the source of truth for what v1.0 contains and in which order it is built: 18 phases, 150 active steps, about 40–50 Claude Code sessions, with human review gates after phases 6, 11 and 16. **This document** explains why, and where the project goes after 1.0. If they disagree, the plan wins and this document is amended.

## 16.1 Version 1.0 — Foundation, Solvers, Estimation, JAX

Contents (details in Sections 3–15):

1. Backend dispatcher with NumPy reference, Numba hot paths and a real JAX backend; validated backend selection.
2. Numerics: grids, 1-D linear interpolation, quadrature, discretisation, Markov tools, the fixed-point driver, Jacobians.
3. Model DSL with three problem templates; immutable compiled models with fingerprints.
4. Solvers: `solve()` with vectorised and Numba VFI (Howard, monotone), EGM, the stationary-equilibrium engine, first-order perturbation, Euler-error and residual diagnostics.
5. Simulation: panels and time series, Young histograms, aggregates, inequality, impulse responses.
6. Estimation: SMM, MLE, Kalman likelihood, Metropolis–Hastings, priors, identification diagnostics.
7. Nine built-in models, each tested against an analytic benchmark.
8. Filters and FRED; plots; experiments, welfare, Monte Carlo and bootstrap.
9. Benchmarks recorded in the repository; Sphinx documentation with nine executed tutorials; wheels; release to PyPI.

Compiled-layer status at 1.0: NumPy reference (all operations), Numba (hot paths), JAX (hot operations, autodiff, vectorisation). C and Rust are not present.

## 16.2 Version 1.1 — Heterogeneous-Agent Depth and Transitions

Krusell–Smith with Den Haan error diagnostics (and, with it, 2-D and 3-D multilinear interpolation); MIT-shock transition paths; second-order perturbation; time iteration for problems EGM cannot handle (endogenous labour supply); discrete-choice EGM with an upper envelope; Mortensen–Pissarides; the Auerbach–Kotlikoff OLG model once transitions exist.

## 16.3 Version 1.2 — Compiled C Layer (only if profiles justify it)

pybind11 kernels for the two or three hottest loops that remain slow under Numba, built against the stable ABI, with wheels from cibuildwheel for Linux, macOS and Windows. They register in the dispatcher as `(operation, "c")`, are tested against the NumPy reference like every other backend, and change no user-facing API.

## 16.4 Version 1.3 — Scale and Breadth

Higher-dimensional state spaces, projection methods with cubic splines and Chebyshev approximation, Smolyak and adaptive grids, a new two-asset (HANK-style) solver, continuous-time HJB models (Merton), multigrid VFI, HMC via NumPyro, CCP and simulated MLE, Bayes factors and WAIC, first-class Polars, the SymPy front end, YAML model files, HDF5 persistence, additional data loaders, plotly and dashboards, MPI-style sweeps.

## 16.5 Version 2.x — Rust, Ecosystem

Rust (PyO3) kernels where memory safety and concurrency matter, such as large particle simulations and MCMC; a plugin system, a model hub, WebAssembly demonstrations and domain extension packages. These come only after the layers above are stable, and again behind the dispatcher.

**Rules for every compiled layer after 1.0:** the NumPy reference stays; a kernel is added only when the benchmark shows that the current best backend misses a target by a clear margin; every new kernel ships with a parity test and a benchmark case; the public API does not change; the wheel matrix is documented before the first release.

**Reinstatement rule.** Anything cut from v1.0 returns only together with its first consumer, a test and a documentation page. Nothing is kept as dormant code in the meantime.

# Section 17 — Community and Governance

## 17.1 Contribution Model

Dynecon adopts a contributor model similar to NumPy and scikit-learn. All code changes go through pull request review; core module changes require approval from a maintainer (from two maintainers once the project has them; at present there is one). CONTRIBUTING.md specifies coding standards, test requirements, and the review process.

## 17.2 Documentation Strategy

- Getting Started: installation, five-minute tutorial, conceptual overview.
- Tutorials: step-by-step Jupyter notebooks for each canonical model.
- API Reference: complete, auto-generated reference for all public functions.
- Developer Guide: architecture overview, contribution guidelines, testing instructions.

## 17.3 Academic Adoption Strategy

- Submitting Dynecon to the Journal of Open Source Software (JOSS) for peer-reviewed publication.
- Partnering with graduate computational economics textbook authors.
- Providing ready-to-use course materials (problem sets, lecture notebooks) for PhD courses.
- Presenting at NBER, AEA, and SED to build awareness in the research community.

## 17.4 Funding and Sustainability

- NSF grant applications for scientific software development.
- Institutional partnerships with Federal Reserve Banks and central banks.
- NumFOCUS fiscal sponsorship for infrastructure and community management.
- Google Summer of Code participation for student contributor development.


# Section 18 — Long-Term Vision

## 18.1 Five-Year Horizon

Within five years, Dynecon should be the default computational infrastructure for a significant fraction of published quantitative macroeconomics. Key milestones: 10,000+ GitHub stars; cited in 200+ published papers; adopted in PhD courses at 50+ universities worldwide.

## 18.2 Machine Learning Integration (long term, not scheduled)

- Neural network function approximation: PyTorch/JAX neural networks as policy function approximators for high-dimensional models (Deep VFI, Deep Equilibrium Nets).
- Reinforcement learning: OpenAI Gym-compatible environments for economic models.
- Causal ML for structural estimation: integration with DoWhy and EconML.

## 18.3 Policy Analysis Platform (long term, not scheduled)

A natural long-term extension is a web-based policy analysis platform that allows non-technical users (policy staff, journalists, legislators) to run pre-configured counterfactual simulations with calibrated Dynecon models, adjust parameters via sliders, and export policy memos in structured formats.

## 18.4 Ten-Year Horizon

- A shared model specification standard adopted across Python and Julia ecosystems.
- A peer-reviewed model repository where economists share calibrated, reproducible implementations.
- A certified computational economics curriculum adopted by 200+ universities globally.
- Dynecon models influencing real policy decisions at major central banks and international organizations.


# Section 19 — The Compiled Core Strategy: Language Layers and Continuous Architecture

## 19.1 The Foundational Principle

This section addresses a question that underlies every long-lived scientific computing project: how do you build a Python library that grows into a high-performance system without discarding what came before, and without asking users to change their code? The answer is a layered compiled architecture, designed from the first commit rather than retrofitted.

Dynecon is not a Python prototype that will eventually be rewritten in Rust or Julia. It is a permanently Python-surfaced system whose execution engine grows more sophisticated over time. The Python API that a researcher uses in version 1.0 keeps working through 1.x. What changes is what happens beneath that API: faster, safer, more parallel compiled code takes over the computation layer incrementally, operation by operation, validated against the NumPy reference implementation at every step.

Version 1.0 ships three layers: the NumPy reference, Numba and JAX. C and Rust are added after 1.0, only where benchmarks show a need. The dispatcher already accepts any backend name, so adding them needs no stubs, placeholder directories or API change now.

## 19.2 The Backend Dispatcher: The Architectural Keystone

Every performance-critical operation in `dynecon.numerics` is registered with the dispatcher at import time under an `(operation, backend)` key:

```python
# Internal structure of the dispatcher registry (schematic)
_registry = {
    ("linear_interp_1d", "numpy"): _linear_interp_1d_numpy,  # the reference
    ("linear_interp_1d", "numba"): _linear_interp_1d_numba,  # registered when Numba imports
    ("linear_interp_1d", "jax"): _linear_interp_1d_jax,  # registered when JAX imports
}


def linear_interp_1d(x, xp, fp, backend=None):
    b = backend or dynecon.config.active_backend
    impl = _registry.get(("linear_interp_1d", b))
    if impl is None:
        if dynecon.config.strict_backend:
            raise BackendError(f"no '{b}' implementation of linear_interp_1d")
        impl = _registry[("linear_interp_1d", "numpy")]  # graceful fallback
    return impl(x, xp, fp)
```

Adding a compiled implementation of an operation requires only registering it. No calling code, solver code or user code changes. The resolution rule is simple: use the requested backend if it has an implementation; otherwise fall back to NumPy, or raise in strict mode. The backends actually used are recorded on every `Solution`.

## 19.3 The Reference Implementation Doctrine

Every hot-path operation begins as a clear NumPy reference. This is not a prototype to be discarded; it is a permanent artifact with three roles:

- **Correctness ground truth:** every compiled implementation is validated against it in CI on every pull request, forever.
- **Debugging fallback:** `dynecon.config.set_backend("numpy")` forces the reference path to isolate whether a bug is numerical or a defect in a compiled implementation.
- **Documentation:** for a contributor who has never written compiled code, the NumPy version explains exactly what the algorithm does.

```python
# Reference implementation: permanent, validates every compiled version
def _linear_interp_1d_numpy(x, xp, fp):
    return np.interp(x, xp, fp)


# Compiled version (scalar kernel, schematic): registered separately, never replaces the reference
@jit
def _interp_scalar_numba(x, xp, fp):
    i = np.searchsorted(xp, x) - 1
    i = max(0, min(i, len(xp) - 2))
    t = (x - xp[i]) / (xp[i + 1] - xp[i])
    return fp[i] * (1 - t) + fp[i + 1] * t
```

Cold-path operations (setup, one-off linear algebra, SciPy calls) have a single implementation and do not need a parallel version.

## 19.4 Layer 0: The Python Foundation (Permanent)

The Python foundation performs no heavy numerical computation. Its job is to specify what must be computed and to coordinate the systems that do it, without assuming anything about the backend that will run the numerical work.

- `dynecon.core` and `dynecon.utils`: errors, `Solution`, RNG contract, validation, provenance. NumPy is the only dependency.
- `dynecon.model`: the specification DSL and compilation to arrays. Its callables are evaluated by NumPy, never inside compiled kernels.
- `dynecon.experiment`: sweeps, experiments and provenance; orchestration only.
- These modules will not have compiled equivalents: their job is coordination, not computation.

## 19.5 Layer 1: Numba JIT (v1.0)

Numba is active from version 1.0. The compiled set was chosen by what the canonical models spend time on: (1) the Bellman maximisation over the asset grid, (2) the linear interpolation called by EGM and by simulation, (3) the simulation step over agents, and (4) the histogram (lottery) update. Everything else is array arithmetic that NumPy already executes in compiled code.

### Numba Boundary Rules

The boundary is enforced by construction rather than by a runtime checker:

- Only scalars and float64/int64 NumPy arrays cross the boundary.
- User callables never cross it. Template functions (utility, resources and so on) are evaluated on arrays outside the kernel, and the kernel receives the resulting arrays.
- Discrete states are integer indices.
- Random numbers are drawn in NumPy before the kernel runs, so results do not depend on thread count.
- The suite runs with Numba installed and with Numba absent; in the second case the `jit` decorator falls back to plain Python and Numba-specific tests are skipped.

## 19.6 Layer 2: JAX Functional Backend (v1.0)

JAX traces functional Python into XLA computation graphs. In v1.0 it replaces the earlier pilot wrappers with real implementations of the hot operations: `linear_interp_1d`, the Bellman maximisation (`bellman_saving`, vectorised `argmax` with `vmap` over shocks), the histogram update (`distribution_step`) and Markov simulation (`lax.scan`). Discretisation stays in NumPy because it is a cold path. JAX is not claimed to be faster: on CPU it is often slower than Numba for these kernels. It is there for exact derivatives, vectorisation across parameters and GPU use. Kernels are written functionally by hand (no side effects, no in-place mutation, `jax.lax` control flow); Dynecon does not attempt to transform arbitrary user code.

### Automatic Differentiation as a Solver Primitive

JAX makes three things exact that would otherwise be finite differences: the Jacobians used to linearise DSGE models, the gradients of SMM and likelihood objectives, and sensitivities of equilibrium quantities. These emerge from the JAX backend without changes to user model specifications. Finite-difference Jacobians stay as the reference, and the JAX Jacobians are checked against them.

### JAX and Numba Coexistence

Numba is better for tight CPU loops with branching; JAX is better for differentiable computation, GPU workloads and `vmap`. They coexist in the dispatcher: the user selects a backend per call or globally, and an operation without an implementation in the selected backend falls back to NumPy.

## 19.7 Layer 3: C Extension Modules (after 1.0, only if justified)

The criterion for a C extension is strict: the operation must be a demonstrated bottleneck in the recorded benchmarks, the Numba and JAX implementations must already exist and be correct, and the best existing backend must miss its target by a clear margin. pybind11 is the binding technology, with zero-copy array sharing through the buffer protocol.

### Candidates

The candidates are the operations the benchmarks show to be hottest, most likely the interpolation kernel, the Bellman maximisation and the histogram update. The list is decided by measurement, not in advance.

### ABI Stability

C extension modules use Python's stable ABI (`Py_LIMITED_API`) so one binary serves several Python versions; cibuildwheel builds Linux (x86_64, aarch64), macOS (Intel and Apple Silicon) and Windows wheels. The `cext/` directory is created with the first such kernel.

## 19.8 Layer 4: Rust Extension Modules (after 1.0, only if justified)

Rust is the preferred language for new compiled subsystems that need safe concurrency: its ownership model rules out data races at compile time, and Rayon makes parallel iteration over agents simple and safe. PyO3 provides the Python binding, explicit GIL release for compute-bound code, and clean translation of Rust errors into Python exceptions.

### Candidate Subsystems

Large-scale particle simulation (millions of agents with idiosyncratic shocks) and concurrent MCMC chains. As with C, the decision follows the benchmarks. The `rsext/` directory is created with the first such subsystem.

## 19.9 The Contributor Model Across Layers

The layered architecture lets different contributor communities work at their natural level:

| Layer | Contributor Profile |
| --- | --- |
| Python surface (core, model, experiment) | Economists and social scientists with Python proficiency. Where most academic contributions will land. |
| Numba layer | Applied scientists comfortable with numerical methods. Numba stays in Python syntax. |
| JAX layer | ML researchers and applied mathematicians; requires functional programming fluency. |
| C extension layer (when it exists) | Numerical computing specialists with C/C++ experience, maintained by a small core team. |
| Rust extension layer (when it exists) | Systems programmers; the most demanding layer, but the most self-documenting through the compiler. |

## 19.10 Cross-Backend Testing

Every hot-path operation has a parametrised test that validates each registered implementation against the NumPy reference. This test never goes away; a new backend adds one name to the parameter list.

```python
@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
def test_linear_interp_1d_cross_backend(backend):
    if not dynecon.backends.is_available(backend):
        pytest.skip(f"{backend} not installed")
    xp = np.linspace(0, 10, 100)
    fp = np.sin(xp)
    x = np.random.default_rng(0).uniform(0, 10, 500)
    ref = dynecon.numerics.linear_interp_1d(x, xp, fp, backend="numpy")
    out = dynecon.numerics.linear_interp_1d(x, xp, fp, backend=backend)
    np.testing.assert_allclose(out, ref, rtol=1e-10)
```

At the whole-solver level, `solve(method="egm")` is also checked to agree between backends within tolerance.

## 19.11 Performance Goals by Layer

An earlier draft of this document carried a table of speed-up figures (for example "45 s → 0.5 s"). They were aspirations, never measurements, so they have been removed. The only numbers that may appear in the README or documentation are those recorded by the benchmark harness on a stated machine. The v1.0 goals, to be compared with recorded results and never edited to fit them, are:

| Case | Goal |
| --- | --- |
| Bellman maximisation (1000 assets × 7 shocks) | Numba ≥ 20× vectorised NumPy |
| Full VFI solve (monotone search + Howard) | ≥ 50× the pure-Python loop of the baseline repository |
| EGM solve | ≥ 20× Numba VFI |
| Panel simulation (10⁵ agents × 500 periods) | Numba ≥ 10× a NumPy loop |
| Stationary distribution (Young) | Numba kernel ≥ 3× sparse-matrix iteration |
| `import dynecon` | under 0.5 s with no optional dependencies imported |

Targets for C and Rust kernels are set when, and if, the benchmarks call for them.

## 19.12 The API Stability Contract

The promise to users: code written against the documented public API of version 1.0 keeps running through 1.x. After 1.0, breaking changes occur only in a major release and only after a deprecation window of at least two minor versions. The compiled layers are an implementation detail: upgrading may make code faster, with no change to user code.

The public API is the list in `API_POLICY.md`: the names re-exported by the top-level `dynecon` namespace and the documented sub-namespaces. Everything under `dynecon.numerics.backends` and anything beginning with an underscore is private and may change without notice.

## 19.13 The Compiled Stack: Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PYTHON SURFACE API  ·  dynecon.*  (permanent contract)                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  BACKEND DISPATCHER  ·  dynecon.numerics  (routes every call, v1.0)          │
├──────────────────────────────────────────────────────────────────────────────┤
│  NUMBA JIT LAYER      ·  numerics/backends/*_numba.py        (v1.0)          │
├──────────────────────────────────────────────────────────────────────────────┤
│  JAX FUNCTIONAL LAYER ·  numerics/backends/*_jax.py          (v1.0)          │
├──────────────────────────────────────────────────────────────────────────────┤
│  C EXTENSION LAYER    ·  pybind11, created on first justified kernel (>1.0)  │
├──────────────────────────────────────────────────────────────────────────────┤
│  RUST SUBSYSTEM LAYER ·  PyO3, created on first justified subsystem (>1.0)   │
├──────────────────────────────────────────────────────────────────────────────┤
│  NUMPY REFERENCE LAYER (always present, never removed, correctness truth)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

Each layer is independently testable, independently deployable as an optional extra and independently maintainable. The NumPy reference is always the base. Each compiled layer is additive: installing Numba activates the Numba implementations of every registered hot-path operation, with no configuration needed.

The way to build a Python library that grows into a high-performance system without discarding anything is to get the dispatch boundary right in version 1.0, write every operation as a reference before compiling it, test every compiled implementation against its reference on every commit, and add compiled layers continuously: never as a replacement, always as an addition, and only with a measurement to justify them. The Python surface does not change; what runs beneath it grows stronger.

---

*End of Architecture Document — Dynecon Blueprint (Revision 2, aligned with the v1.0 Execution Plan, revision 2)*
