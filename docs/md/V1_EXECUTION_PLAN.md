# EconKit v1.0 — Execution Plan for Claude Code

| | |
|---|---|
| **Document** | `docs/md/V1_EXECUTION_PLAN.md` (drop this file in that path) |
| **Written** | 2026-11-03 |
| **Revision** | 2 (2026-11-03): lean-by-construction pass, decisions A–E applied; see §7 |
| **Repo** | https://github.com/JayabrataBasu/EconKit (branch `master`) |
| **Baseline** | HEAD = commit of 2026-03-30 ("Added end2end solve -> simulate -> moments pipeline helper…"), 15 commits, 13 logged sessions |
| **Goal** | Take EconKit from its current foundation to a published, documented, API-stable **v1.0.0** on PyPI |
| **Executor** | Claude Code, run locally inside the repo, one phase per session |
| **Reviewer** | Jayabrata Basu (sole maintainer) |
| **Companion docs** | `docs/md/EconKit_Architecture.md` (the architecture, written as an interpretation of this plan), `docs/md/CAPTAINS_LOG.md`, `docs/latex/econkit_math.tex` |
| **Precedence** | This plan is the source of truth for what v1.0 contains and when. The architecture explains why and where the project goes next; if they ever disagree, this plan wins and the architecture is amended. |

---

# Part A — Context, protocol and scope

## 0. How to use this document

This plan is written so an AI coding agent can finish it. Every phase has the same shape: a goal, exact files, API signatures, algorithm specifications, numerical acceptance tests with tolerances, and a Definition of Done expressed as shell commands. A phase is finished when its gates pass, not when the work "looks complete".

**To start a session**, run `make phase P=<n>` (added in P0.11). It writes `.phase/P<n>.md`, which contains Section 2 (the operating protocol) plus that one phase, so the agent reads about 5 % of this document instead of all of it. Then paste the prompt in Appendix A into Claude Code. Until P0.11 exists, read Section 2 and the named phase by hand.

**Four ideas drive the whole plan:**

1. **The architecture is the contract.** The Python surface API is permanent. Compiled layers (Numba now, JAX next, C/Rust after 1.0) live behind the backend dispatcher and are never visible to users. There is no "rewrite in a faster language" step anywhere in this plan; performance work *adds* implementations behind the dispatcher and validates them against the NumPy reference.
2. **Analytic benchmarks over hand-waving.** Where a model has a closed form (Brock–Mirman growth, permanent-income consumption, log-utility Lucas tree, Diamond OLG, Taylor-principle determinacy), the test suite checks the solver against it. This is how an agent can verify its own numerical work without a human reading output.
3. **Honest progress.** Progress is reported only as checked boxes in `docs/md/V1_PROGRESS.md`. Self-assessed percentages are banned (see R6). The Captain's Log previously reported "~100% of Phase 1" while large parts of Phase 1 were unbuilt; that must not recur.
4. **Lean by construction.** One concept has one implementation. Nothing is written without a caller, nothing old is kept "for compatibility" before 1.0, and nothing is added "in case we need it". Capabilities return to the library together with their first consumer. This is rule R21, and it is enforced by tooling (`make deadcode`), not by good intentions.

---

## 1. Baseline audit (verified 2026-11-03)

### 1.1 What exists today

| Area | State | Files |
|---|---|---|
| Backend dispatcher | Working registry, `strict_backend`, NumPy fallback | `econkit/numerics/_registry.py`, `_dispatch.py`, `econkit/config.py` |
| Numerics (NumPy reference) | `linear_interp_1d`, Tauchen, Rouwenhorst, Gauss–Hermite, Gauss–Laguerre, `expectation_discrete`, `golden_section_max`, `brent_max` | `econkit/numerics/*.py`, `backends/*_numpy.py` |
| Numerics (Numba) | Same operations, plus optional-import pattern | `backends/*_numba.py`, `numba_backend.py` |
| Numerics (JAX) | **Pilot only**: `jnp.interp` wrapper and Tauchen/Rouwenhorst wrappers that delegate to baseline behaviour | `backends/jax_backend.py`, `discretization_jax.py` |
| Model DSL | `Model` builder → immutable `CompiledModel`; `Param`, `State`, `Control`, `AR1Shock`, `EquationSpec` | `econkit/model/` |
| Solvers | Generic `value_function_iteration` (operator-driven, Howard hook, relaxation, hooks); canonical helpers for deterministic, stochastic-income and liquid/illiquid consumption-savings; Bellman residual and convergence diagnostics with schema-versioned JSON artifacts | `econkit/solvers/` |
| Simulation | `simulate_markov_asset_policy`, summary moments, diagnostics report | `econkit/simulation/policy.py` |
| Estimation | `MomentTarget`, `quadratic_moment_loss`, report stub, one demo pipeline | `econkit/estimation/` |
| Tooling | `Makefile` (`install test lint typecheck format test-structure ci benchmark-backends`), ruff, mypy (`disallow_untyped_defs`), test-layout guard, one micro-benchmark script | `Makefile`, `scripts/` |
| Docs | `docs/md/CAPTAINS_LOG.md`, `REVIEW_1.md`, `economics_explainability.md`; LaTeX math reference | `docs/` |

**Size:** about 2,560 lines of source, 990 lines of tests.
**Tests:** 56 total. With a plain `pip install -e .[dev]`: **50 pass, 6 skip** (all skips are Numba parity tests, because `numba` is not in the `dev` extra). With Numba installed the Captain's Log reports 56 passing. Nothing is red.

### 1.2 Corrections to existing claims

These are factual mismatches between the repo's own documents and the repo. Phase 0 fixes them.

| Claim | Reality |
|---|---|
| Captain's Log: "~100% of Phase 1" (Session 13) | Phase 1 in the architecture document also required: EGM solver, RBC / McCall built-ins, a real SMM estimator, matplotlib plotting, Hypothesis tests, tutorial notebooks, Numba-compiled VFI loop. None exist. Review-1 itself estimated the full product at 42 % after Session 10. |
| README: "CI, tests, and contribution standards" | `.github/` is in `.gitignore` ("User-requested exclusion"), so the CI workflow and PR template are **not on GitHub**. |
| README/Log: `docs/md/BACKEND_ROADMAP.md`, `docs/md/EconKit_Architecture.md` | Both are listed in `.gitignore`; neither is in the repository. (`REVIEW_1.md` is also ignored but was committed before the rule.) |
| README: "Optional JAX pilot registrations" | Accurate, but the JAX kernels wrap baseline behaviour; there is no JAX speed or autodiff yet. |
| Architecture doc, §19.11 benchmark table (e.g. "45 s → 0.5 s") | These are **aspirational targets**, never measured. They must not appear in the README or docs until replaced by measured numbers (Phase 15). |

### 1.3 Technical debt that shapes this plan

| ID | Debt | Resolved in |
|---|---|---|
| TD1 | Canonical solvers use pure-Python double loops over the grid (`for i … for j …` in `canonical.py`). They are not vectorized, not Numba-compiled, and not routed through the dispatcher. This is the largest performance gap in the project. | P5 (`canonical.py` is deleted, not wrapped) |
| TD2 | Canonical helpers hard-code state/parameter names (`"assets"`, `beta`, `sigma`, `r`, `y`, `a_min`) and a CRRA utility. The model's registered **equations are never called** by them. The DSL is decorative. | P4, P5 |
| TD3 | `Solution` protocol only has `metadata`; `Simulation` protocol requires `to_dataframe()` but `SimulationResult` has none. No common result artifact. | P1.8 (protocols removed), P5.1 (`Solution` class), P7 (`SimulationResult`) |
| TD4 | `Config.set_backend()` does not validate the backend name or availability. | P1 |
| TD5 | `econkit/core/types.py` (`StateSpec`, `Parameter`) is unused and duplicates `model/specs.py`. | P0.12 (deleted) |
| TD6 | `pyproject.toml` lists packages by hand; a new subpackage that is forgotten silently drops out of the wheel. `numba` is not in `dev`. SciPy is not a dependency. | P0 |
| TD7 | JAX backend is a pilot wrapper. | P12 |
| TD8 | The liquid/illiquid solver is a brute-force prototype (`O(n²)` over paired grids). | P5.5 (deleted; a two-asset solver returns post-1.0 with a proper algorithm) |
| TD9 | `run_solve_simulate_moments_pipeline` builds a model inline; it is a demo, not a workflow engine. | P11.10 (deleted; `Experiment` in P14 replaces it) |

---

## 2. Operating protocol for Claude Code

These rules apply to **every** phase. They are the difference between a plan an agent can finish and one it cannot. Step IDs such as `P5.3` are stable identifiers: retired IDs are never reused (see §7).

**R1 — Start of session.** Read, in order: (a) this Section 2, (b) the last two sections of `docs/md/CAPTAINS_LOG.md`, (c) the phase you were assigned, (d) `docs/md/V1_PROGRESS.md`. Run `make install && make ci` and record the test count and any skips. If `make ci` is red before you change anything, fix or report that first; do not build on red.

**R2 — One phase per session.** Work phases in numerical order. Do not start phase *N+1* until every checkbox of phase *N* is ticked and its Definition of Done commands pass. Phases marked **splittable** may span sessions; stop at a sub-step boundary, log it, and resume next time.

**R3 — Reference-first doctrine** (architecture §19.3). Every numerical kernel is written first as a clear NumPy reference with tests. A compiled implementation (Numba, later JAX) is added **afterwards**, registered in the dispatcher, and validated against the reference with a cross-backend test. The reference is never deleted or weakened.

**R4 — Hot path vs cold path.** *Hot-path* operations (called inside an iteration loop or per agent: interpolation, Bellman maximisation, simulation step, histogram update) require NumPy + Numba parity. *Cold-path* operations (setup, one-off linear algebra, diagnostics) may be NumPy-only. Each phase lists which is which.

**R5 — Gates.** A phase is done only when all of these pass:
```bash
make ci                 # test-structure guard, ruff, mypy, pytest, deadcode (R21)
make coverage           # added in P0; floors in Section 2 / R11
```
plus the phase's own acceptance tests. Never mark a box ticked on the strength of "it should work".

**R6 — Honest progress.** Progress is the checklist in `docs/md/V1_PROGRESS.md`, nothing else. Never write a percentage. Never write "complete" for a phase with an unticked box. Performance numbers in docs must come from a recorded benchmark run (P15 defines the format); never invent or round up a number.

**R7 — Explainability.** Every public function and class has a NumPy-style docstring that includes an `Economics note:` line (existing convention, see `docs/md/economics_explainability.md`). Every new numerical method adds its formulas to `docs/latex/econkit_math.tex` in the same session (that file is retired in P16.4, when its contents move to `docs/theory/`, so the mathematics has exactly one home at any time).

**R8 — Tests.** Place tests in the domain folder (`tests/<domain>/`); `scripts/check_test_layout.py` enforces layout. Rules: deterministic seeds; no network; no wall-clock assertions; tolerances stated with a one-line justification; prefer analytic benchmarks (Appendix E); statistical tests use bounds such as "within 4 standard errors", never exact equality on random output. Markers (registered in P0): `slow`, `numba`, `jax`, `network`, `plot`.

**R9 — Existing tests are requirements, not artefacts.** Never delete or weaken a test just to make a change pass. When behaviour is deliberately replaced (R21c), *port* the test: every assertion that still describes wanted behaviour moves to the new API in the same commit; assertions that only described deleted behaviour are removed and listed in the Captain's Log. Loosening any tolerance by more than 10× requires a stop-and-ask (R14).

**R10 — API discipline.** Before 1.0 there are no external users, so the API changes freely: change the callers and tests in the same commit and delete the old path (R21c). Every user-visible change still gets a line in `CHANGELOG.md` under "Changed/Removed". From 1.0 on, `docs/md/API_POLICY.md` governs: breaking changes only in a major release after a deprecation window of at least two minor versions. The deprecation helper is written when the first deprecation happens, not before. New public symbols go in the owning module's `__all__`; top-level `econkit/__init__.py` re-exports only the list in `API_POLICY.md`.

**R11 — Coverage floors.** `core`, `numerics`, `model`, `solvers`: ≥ 90 % line coverage. Everything else ≥ 80 %. Measured by `make coverage`, which runs with Numba installed.

**R12 — Determinism.** All stochastic functions accept `rng: numpy.random.Generator | int | None` (helper in P1). Results must not depend on thread count: draw random numbers in NumPy up front, then run deterministic kernels.

**R13 — Scope.** Do exactly the phase. If you find something worth doing that is out of scope, add one line to the "Parking lot" section at the end of `V1_PROGRESS.md` and move on. Never "improve" unrelated modules.

**R14 — Stop-and-ask triggers.** Stop, write the question in the Captain's Log under "Blockers", and end the session if you hit any of: a licensing question; a new *required* runtime dependency not listed in Appendix G; a breaking change to a stable (post-1.0) public API; loosening tests (R9) or dropping assertions that still describe wanted behaviour; a numerical result that contradicts an analytic benchmark and you cannot find the bug in two focused attempts; a phase that would take more than the stated session budget.

**R15 — Environment hygiene.** Use the `.econkit` virtualenv created by `make install`. CI must also pass with Numba absent (tests skip via `pytest.importorskip`) and with Numba present. Never commit `.econkit/`, caches, or notebooks with stale outputs.

**R16 — Commits.** Small, reviewable commits, subject `P<phase>.<step>: <summary>` (for example `P5.2: vectorized Bellman kernel`). One logical change per commit. Do not push tags or publish packages; those are human-gated steps in P17.

**R17 — Captain's Log entry** (append to `docs/md/CAPTAINS_LOG.md`, template in Appendix B): phase and step, files changed, checklist IDs ticked, gate output (test counts), deviations from this plan, blockers, next step. Keep it factual.

**R18 — Numerical hygiene.** `float64` everywhere unless a phase says otherwise. Validate shapes and finiteness at public boundaries with `econkit.utils.validation` (P1) and raise `EconKitError` subclasses with actionable messages. Warn (`ConvergenceWarning`) rather than silently return unconverged results.

**R19 — Performance claims.** A speedup is "achieved" only if the P15 benchmark harness recorded it on the same machine, same inputs, same tolerance. Phases that state speedup *targets* (for example P5) say what to do if the target is missed: record the real number, profile, document; never adjust the target to match.

**R20 — Docs are part of done.** Every phase ends with: docstrings, `econkit_math.tex` update if numerics changed, CHANGELOG line, checklist ticks, Captain's Log entry.


**R21 — Lean by construction: no dead code, no duplicate paths, no speculative generality.**

- **(a) One concept, one implementation path.** Before adding a function, class, option or module, search for one that already does the job and extend it. A second implementation of the same operation is allowed only as a compiled backend registered under the same operation name (R3) and covered by a parity test.
- **(b) Every public symbol has a caller and a test.** A caller is another library module, or a tutorial/model page that is executed in the docs build. `make deadcode` enforces this with `vulture` (confidence ≥ 80), `ruff` rules `F`, `ERA`, `ARG`, `C90`, and `scripts/check_usage.py`. Allow-list entries (`scripts/vulture_whitelist.py`, `scripts/usage_allowlist.txt`) need a one-line reason each.
- **(c) No compatibility shims before 1.0.** No deprecated aliases, "legacy" wrappers, `_old` functions or parallel old/new APIs. Git history is the archive. Replace the code, migrate callers and tests (R9) in the same commit.
- **(d) No speculative generality.** No parameter, config field, base class, protocol, registry, optional extra or "hook" without a present consumer. A protocol needs at least two implementers in the v1 scope. Do not write stubs for future backends: the dispatcher already accepts any backend name.
- **(e) Near-future exception.** Code that has no caller yet is allowed only if a concrete, non-retired step of this plan will call it within the current or the next review window (windows: P0–P6, P7–P11, P12–P16, P17). Mark it `# NEAR-FUTURE(P<n>.<m>): <what will call it>`. `scripts/check_near_future.py` fails on: a tag naming a step that does not exist or is retired, a tag whose step is already ticked in `V1_PROGRESS.md` (then the code should have a caller), or a tag more than one window ahead. Tags are re-audited at every review gate.
- **(f) Prefer NumPy and SciPy to our own code** when they do the job at reference quality (for example `scipy.optimize.brentq`, `scipy.optimize.root`, `scipy.sparse`, `scipy.linalg.solve_discrete_lyapunov`). Write our own only where compilation (Numba/JAX) or domain semantics require it.
- **(g) No pass-through wrappers.** Do not wrap pandas, SciPy or matplotlib in classes or functions that add no behaviour.
- **(h) Skipping is a valid outcome.** If a step in this plan would create something that violates (a)–(g), skip it, write "skipped under R21 because …" in the Captain's Log, and tell the maintainer. Capabilities that were deferred or removed return only together with their first consumer (Part E).
---

## 3. v1.0 scope

### 3.1 Release philosophy

v1.0 is **not a throwaway MVP**. It is the first release that carries the API-stability promise: code written against the documented public API keeps running through 1.x, and what runs underneath may get faster in every release. It is also deliberately **lean**: everything in it has a caller or a documented use (R21).

- The compiled-layer strategy is unchanged: NumPy reference → Numba → JAX → (after 1.0, only where profiling justifies it) C extensions via pybind11 and Rust via PyO3. The dispatcher already accepts any backend name, so those layers need no stubs now.
- v1.0 ships NumPy + Numba (hot paths) + a real JAX backend (P12). It does not ship C or Rust. Adding them later requires no user-facing change and, under SemVer, only a minor release.

### 3.2 In scope for v1.0

| Module | v1.0 deliverables |
|---|---|
| `core`, `utils`, `config` | Error hierarchy; `Solution` result type; RNG contract; JIT decorator with pure-Python fallback; validated backend selection and introspection; provenance record |
| `numerics` | Uniform and curved (exponential-map) grids; 1-D linear interpolation (NumPy + Numba); Markov-chain tools (exact stationary distribution, simulation); one fixed-point iteration driver used by every solver; numerical Jacobians; Gauss–Hermite and Gauss–Legendre quadrature; Tauchen and Rouwenhorst discretisation |
| `model` | Declarative DSL: parameters, states, shocks (AR(1), Markov, IID), controls; three problem templates (`saving`, `finite_dp`, `dsge`); validation; immutable compiled models with `with_params`; fingerprint |
| `solvers` | One entry point `solve()`; value function iteration (vectorised NumPy and Numba, Howard acceleration, monotone search); endogenous grid method for every `saving` problem (consumption–savings and growth models alike); stationary-equilibrium engine (Aiyagari, Huggett); first-order perturbation (Klein/QZ) with IRFs and theoretical moments; Euler-error and residual diagnostics |
| `simulation` | Panel and time-series simulation with interpolated policies; Young (2010) histogram distributions; aggregates, Lorenz/Gini; thread-independent determinism; nonlinear impulse responses |
| `estimation` | SMM (two-step weighting, standard errors, J-test), MLE with Hessian standard errors, Kalman-filter likelihood, adaptive Metropolis–Hastings with R-hat/ESS, priors and parameter transforms, identification diagnostics. Exactly identified SMM is the calibration tool; there is no separate calibration API |
| `models` | Nine built-in models: consumption–savings, Brock–Mirman, RBC, McCall, Aiyagari, Huggett, two-period OLG, three-equation New Keynesian, Lucas tree |
| `data` | Hodrick–Prescott, Baxter–King and Hamilton filters (array in, array out); FRED download and parsing |
| `viz` | Matplotlib plots: policy, value, IRF, simulation, distribution, Lorenz, Euler errors, convergence, moment fit, MCMC trace, posterior, sweep, equilibrium |
| `experiment` | `ParameterSweep`, `Experiment` with provenance and `reproduce()`, policy experiments with consumption-equivalent welfare, Monte Carlo and bootstrap, elasticities |
| Engineering | CI matrix, coverage and dead-code gates, benchmark harness with recorded results, Sphinx site, nine executed tutorials, wheels, TestPyPI → PyPI release |

### 3.3 Out of scope for v1.0 (post-1.0; see Part E)

Each item returns together with its first consumer (R21h).

- **Solvers and methods:** time iteration (EGM covers every v1 saving problem; time iteration returns with models that EGM cannot handle, such as endogenous labour supply), Krusell–Smith, HANK / two-asset solvers, MIT-shock transition paths, second-order perturbation, continuous-time HJB, discrete-choice EGM, multigrid VFI, projection methods, Smolyak grids, finite elements, Anderson acceleration, HMC / NumPyro.
- **Numerics:** multilinear interpolation in 2-D/3-D (returns with Krusell–Smith), cubic splines and Chebyshev approximation (return with projection methods), Brent/golden-section maximisation (VFI is grid-based; they return with continuous-choice refinement if profiling asks for it), QMC draws, Tauchen–Hussey.
- **Models:** Mortensen–Pissarides, Auerbach–Kotlikoff multi-period OLG, Merton (continuous time), Krusell–Smith economy, two-asset HANK, Eaton–Kortum, labour-supply models.
- **Data and I/O:** `EconTimeSeries`/`EconPanel` wrapper classes (pandas already does this; revisit with first-class Polars), CSV/Parquet reader wrappers, PSID/SIPP/BHPS/CEX/SCF/PWT/World Bank loaders, YAML model files, HDF5.
- **Estimation breadth:** CCP-based and simulated MLE, Bayes factors, WAIC.
- **Platform:** SymPy symbolic front end, first-class Polars, plotly and dashboards, MPI, C and Rust layers, WebAssembly, plugin system, model hub, policy web platform, the multi-agent `Agent` abstraction.

**Removed from the baseline under R21** (replaced, not wrapped; the baseline is preserved by git tag `v0.1.0-baseline`): `solvers/canonical.py` and the liquid/illiquid prototype, `simulate_markov_asset_policy`, `run_solve_simulate_moments_pipeline`, `VFISolution`, `core/types.py`, the `EquationSpec` / `Model.equation` mechanism, the delegating JAX wrappers for Tauchen/Rouwenhorst, `gauss_laguerre_exp`, `golden_section_max`, `brent_max`, and `scripts/benchmark_backends.py`. Each deletion is confirmed by tooling in the step that performs it, not assumed.

### 3.4 Version map

| Version | After phase | Meaning |
|---|---|---|
| 0.1.x | P0–P1 | Hygiene + core contract |
| 0.2.0 | P3 | Numerics complete |
| 0.3.0 | P6 | Model DSL + VFI/EGM solvers |
| 0.4.0 | P8 | Simulation + equilibrium |
| 0.5.0 | P10 | Perturbation + model library |
| 0.6.0 | P11 | Estimation |
| 0.7.0 | P12 | JAX backend |
| 0.8.0 | P14 | Data, viz, experiments |
| 0.9.0 | P16 | Performance + docs |
| 1.0.0rc1 → 1.0.0 | P17 | Release |

Only P17 publishes to PyPI. The 0.x tags are local git tags (or TestPyPI uploads, human-gated) so the project's history is traceable.

---

## 4. Decisions log

### 4.1 Resolved by the maintainer (2026-11-03)

| ID | Question | Decision |
|---|---|---|
| **A** | Scope of v1.0 | Keep the larger v1.0 of this plan (it contains enough to complete the project). The architecture document is rewritten as an interpretation of this plan. |
| **B** | Model library | Nine models as listed in §3.2. Brock–Mirman is added to the architecture. Mortensen–Pissarides, Auerbach–Kotlikoff and Merton are post-1.0. |
| **C** | Purity of `core/` | `core/` may depend on NumPy (not only the standard library). |
| **D** | Estimator API | **One** public shape: functions with keyword options (validated against a frozen config dataclass that is recorded in the result for reproducibility) returning an immutable result (`smm(...) -> SMMResult`, same style as `solve(...) -> Solution`). No `SMMEstimator` class and no second wrapper. Rationale: the rest of the API is function-and-result; a stateful estimator object would add a second way to do the same thing, and a mutable `fit()` is harder to test and to trace with JAX. |
| **E** | `Agent` abstraction | Not in v1.0. The model DSL's problem templates play that role. Reserved for multi-agent models (Part E). |
| **F** | Engineering principle | Minimise technical debt; no dead code unless it is about to be used (R21). |

### 4.2 Open decisions

These still need a human answer. Each has a default so work is never blocked, except where noted.

| ID | Decision | Default if unanswered | Why it matters |
|---|---|---|---|
| **D1** | **License: keep GPL-3.0 or move to a permissive license (BSD-3-Clause / MIT).** | Keep GPL-3.0; blocks **only P17**. Decide before P17, ideally now. | The stated goal is "scikit-learn for computational economics". NumPy, SciPy, scikit-learn and statsmodels are BSD-licensed. GPL-3.0 is copyleft: any downstream package or institutional tool that imports EconKit must itself be GPL-compatible, which discourages exactly the adoption the project wants. Relicensing is trivial while you are the sole copyright holder and hard after outside contributions arrive. |
| **D2** | Make **SciPy a required dependency.** | Yes. | Needed for `ordqz` (Klein), sparse matrices (Young distributions), `brentq`/`root`/`minimize` (equilibrium, steady state, SMM), Lyapunov solver, `scipy.stats`. Reference kernels stay NumPy. |
| **D3** | Track `.github/` in git (remove it from `.gitignore`). | Yes. | CI, release and docs workflows cannot run from GitHub otherwise. |
| **D4** | Track `docs/md/EconKit_Architecture.md` and `BACKEND_ROADMAP.md` (both currently ignored). | Track both; the architecture is the single maintained copy (the `.docx` version is retired). | Contributors and CI cannot see ignored files; two copies of one document drift. |
| **D5** | PyPI project name `econkit`. | Verify availability in P0 (`pip index versions econkit`, or the PyPI project page). If taken, choose an alternative and update `pyproject.toml`. | Could not be checked while writing this plan (network allowlist), so it is **unverified**. |
| **D6** | Supported Python: 3.10–3.12 required in CI; 3.13 as non-blocking job. | As stated. | Numba/JAX wheels lag new CPython versions. |
| **D7** | Defer SymPy symbolic compilation to post-1.0. | Defer. | Callable templates + numerical Jacobians (and JAX autodiff) cover v1; SymPy adds a heavy dependency and a second compilation path to test. |
| **D8** | Real author name/email and project URLs in `pyproject.toml`, `CITATION.cff`. | Use "Jayabrata Basu" and the GitHub URL. | `authors = "EconKit Contributors"` is a placeholder. |

---

## 5. End-state repository layout (target at v1.0.0)

A path appears in this tree only if a plan step creates it **and** something calls it (R21).

```
EconKit/
├── econkit/
│   ├── __init__.py              # curated top-level exports (API_POLICY.md)
│   ├── _version.py
│   ├── config.py                # backend, strict mode, threads
│   ├── backends.py              # is_available(), available_backends(), describe()
│   ├── core/                    # errors, Solution
│   ├── utils/                   # jit, rng, validation, provenance
│   ├── numerics/                # grids, interpolation, integration, discretization, markov,
│   │   │                        # fixedpoint (the one iteration driver), jacobian
│   │   └── backends/            # *_numpy.py, *_numba.py, *_jax.py
│   ├── model/                   # specs, shocks, base, compiled, problems, validation, fingerprint
│   ├── solvers/                 # api, vfi, dp/, egm, equilibrium, perturbation/, diagnostics
│   ├── simulation/              # simulate, distribution, moments, result
│   ├── estimation/              # params, priors, moments, smm, mle, kalman, bayes,
│   │                            # identification, results
│   ├── models/                  # registry, utility functions, 9 built-in models
│   ├── data/                    # filters, fred
│   ├── viz/                     # style + plot functions
│   └── experiment/              # sweep, experiment, policy, welfare, montecarlo
├── tests/                       # per-domain folders + api/, models/, golden/, engineering/
├── benchmarks/                  # harness, cases, results/*.json
├── docs/
│   ├── conf.py, index.md, user_guide/, api/, theory/, tutorials/ (executed), models/
│   ├── md/                      # plan, progress, captain's log, API policy, benchmarks, known limits
│   └── latex/econkit_math.tex
├── scripts/                     # check_test_layout, check_progress, check_usage,
│                                # check_near_future, phase, backend_matrix, dump_api, smoke_installed
├── .github/workflows/           # ci.yml, docs.yml, nightly.yml, release.yml
├── pyproject.toml  Makefile  CHANGELOG.md  CITATION.cff  CONTRIBUTING.md  README.md  LICENSE
```

---

## 6. Phase overview

| Phase | Title | Size | Sessions (est.) | Depends on | Active step IDs |
|---|---|---|---|---|---|
| **P0** | Baseline repair, hygiene and lean tooling | M | 1–2 | — | P0.1–P0.12 |
| **P1** | Core contract: errors, RNG, JIT, config | S | 1–2 | P0 | P1.1, P1.3–P1.5, P1.7–P1.9 |
| **P2** | Numerics I: grids and 1-D interpolation | S | 1–2 | P1 | P2.1, P2.2, P2.7, P2.9, P2.10 |
| **P3** | Numerics II: iteration driver, Jacobians, Markov, quadrature | M | 1–2 | P2 | P3.2–P3.4, P3.6, P3.8, P3.9 |
| **P4** | Model specification system | L | 2–3 | P1, P3 | P4.1–P4.3, P4.5, P4.6, P4.8–P4.10 |
| **P5** | Solver core: `solve()`, `Solution`, vectorized and Numba VFI | L, splittable | 3–4 | P2, P4 | P5.1–P5.10 |
| **P6** | EGM | M | 2 | P5 | P6.1, P6.2, P6.4–P6.8 |
| **P7** | Simulation engine and distributions | L, splittable | 3 | P6 | P7.1–P7.5, P7.7–P7.10 |
| **P8** | Equilibrium solvers | M | 2 | P7 | P8.1–P8.6 |
| **P9** | Perturbation (first order) | L | 3 | P3, P4 | P9.1–P9.9 |
| **P10** | Built-in model library | L, splittable | 4–5 | P6, P8, P9 | P10.0–P10.13 |
| **P11** | Estimation | L, splittable | 3–4 | P7, P9, P10 | P11.1–P11.7, P11.9, P11.10 |
| **P12** | JAX backend and autodiff | M | 2–3 | P5–P7, P9, P11 | P12.1–P12.5, P12.7, P12.8 |
| **P13** | Data and visualization | M, splittable | 2–3 | P7, P11 | P13.3, P13.5, P13.7–P13.9 |
| **P14** | Experiments and reproducibility | M | 2 | P10, P11 | P14.1–P14.6, P14.8 |
| **P15** | Performance engineering and benchmarks | M | 2 | P12 | P15.1–P15.7 |
| **P16** | Documentation and tutorials | L, splittable | 4 | P13, P14 | P16.1–P16.11 |
| **P17** | Release engineering and v1.0.0 | M | 3 | all | P17.1–P17.11 |

**150 active steps.** Estimated total: **about 40–50 Claude Code sessions** plus human review time. These are planning estimates, not commitments; if a phase needs more than 1.5× its budget, trigger R14. Gaps in the step numbers are retired IDs (§7).

### 6.1 Dependency sketch

```
P0 → P1 → P2 → P3 ┬→ P4 → P5 → P6 → P7 → P8 ┐
                   └→ P9 ───────────────────┤→ P10 → P11 → P12 → P13 → P14 → P15 → P16 → P17
```
P9 (perturbation) only needs P3 and P4, so it can be done any time after P4 if you want to switch gears.

### 6.2 Human review gates

| Gate | After | Reviewer checks |
|---|---|---|
| **Review-2** | P6 | Numerics and solver correctness against analytic benchmarks; API shape of `solve()` and `Solution`; measured P5 speedups; audit of every `NEAR-FUTURE` tag (R21e). Last cheap moment to change core names. |
| **Review-3** | P11 | Estimation recovery experiments; model-library coverage; whether v1 scope still fits; `NEAR-FUTURE` audit. |
| **Review-4** | P16 | Docs read-through; tutorial runtime; API surface snapshot; license decision D1 final; `NEAR-FUTURE` audit (there should be none left). |
| **Release review** | P17.5 | Release candidate installed on a clean machine. |

---

## 7. Revision history and retired step IDs

### 7.1 Revisions
- **Rev 1 (2026-11-03):** first complete plan, P0–P17, 160 steps.
- **Rev 2 (2026-11-03):** maintainer decisions A–F applied (§4.1). Added R21 (lean by construction) with enforcing tooling (P0.11, P0.12). Merged the growth and consumption–savings templates into one `saving` template and dropped time iteration (EGM covers both); `finite_dp` replaces the `search` template. Removed compatibility shims and deprecation tooling (nothing is released yet). Removed capabilities with no v1 consumer (list in §3.3) and moved things to the phase that first uses them (for example `Solution` persistence to P14.2, priors to P11). Added property-based tests (P2.10, P3.9, P7.10, P9.9), golden-file tests (P6.8), optional external cross-validation (P10.13), JOSS preparation and a known-limits page (P16.10–P16.11) and supply-chain hygiene (P17.11). The architecture document was rewritten as an interpretation of this plan.

### 7.2 Retired step IDs
Step IDs are stable identifiers; retired IDs are never reused. A retired step produces nothing in v1.

| ID | Disposition |
|---|---|
| `P1.2` | `Solution` is created in P5.1, where `solve()` first returns it |
| `P1.6` | the deprecation helper is written when the first deprecation happens, after 1.0; R10 |
| `P2.3`–`P2.6` | 2-D/3-D multilinear interpolation, cubic splines and Chebyshev approximation have no v1 consumer (one continuous state); they return with Krusell–Smith and projection methods, Part E |
| `P2.8` | the old micro-benchmark script is deleted in P15.1 when the harness replaces it |
| `P3.1` | scalar root finding, vector Newton and bracketed solvers are SciPy calls (R21f) |
| `P3.5` | Tauchen–Hussey, `discretize_iid_normal` and library moment helpers have no consumer |
| `P3.7` | QMC draws have no consumer |
| `P4.4` | no general `constraint` API; the borrowing limit is the lower bound of the asset `State`, which fixes the inconsistent `a_min` handling in the old `canonical.py` |
| `P4.7` | YAML model files, the callable library and `from_dict` have no consumer in v1; `Experiment` reproduces runs from a built-in model name plus parameter overrides, or from an importable builder |
| `P6.3` | time iteration: EGM solves the same Euler problems without root finding; returns for models EGM cannot handle, Part E |
| `P7.6` | chunking has no consumer; `simulate` stores only the variables named in `store=[...]`, default all, and the benchmark case needs 4·10⁸ bytes at most |
| `P11.8` | calibration is exactly identified SMM: with as many moments as parameters the minimum loss is 0 and the estimate hits the targets; the SMM tutorial shows it |
| `P12.6` | JAX Monte Carlo has no consumer; `monte_carlo` in P14.5 uses spawned NumPy streams |
| `P13.1` | an `EconTimeSeries` wrapper around `pandas.Series` adds no behaviour; revisit with first-class Polars, Part E |
| `P13.2` | `EconPanel`; `SimulationResult.to_dataframe()` (P7.1) already gives the long-format panel |
| `P13.4` | CSV/Parquet reader wrappers; use pandas |
| `P13.6` | synthetic dataset generators; tests build their data inline and tutorials simulate from the model |
| `P14.7` | a results-table wrapper would only call pandas' `to_latex`/`to_markdown`; see P11.3 |

---

# Part B — Phases P0 to P5: Foundation, numerics, models, first real solvers

Each phase uses this template: **Goal · Prerequisites · Files · Specification · Acceptance tests · Definition of Done · Session budget · Pitfalls.**

---

## P0 — Baseline repair, hygiene and lean tooling

**Goal.** Make the repository trustworthy and reproducible before any new feature work: honest docs, complete packaging, CI that actually exists on GitHub, coverage measurement, a progress tracker, and the tooling that keeps the code base lean (R21) — then remove whatever that tooling proves dead.

**Prerequisites.** None.

**Files touched.** `pyproject.toml`, `Makefile`, `.gitignore`, `.github/workflows/ci.yml`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/md/V1_PROGRESS.md`, `docs/md/V1_EXECUTION_PLAN.md`, `docs/md/CAPTAINS_LOG.md`, `docs/md/BACKEND_ROADMAP.md`, `scripts/{check_test_layout,check_progress,check_usage,check_near_future,phase}.py`, `scripts/vulture_whitelist.py`, `scripts/usage_allowlist.txt`.

### P0.1 Install this plan and the progress tracker
- Copy this document to `docs/md/V1_EXECUTION_PLAN.md`.
- Create `docs/md/V1_PROGRESS.md`: one checkbox per tick-box ID in this plan (`- [ ] P0.1 …`), grouped by phase, plus an empty "Parking lot" and "Blockers" section. Tick P0.1 itself last.
- Write `scripts/check_progress.py`: parses every heading of the form `### P<phase>.<step>` in `V1_EXECUTION_PLAN.md` and fails if any ID is missing from `V1_PROGRESS.md` (or if the progress file lists an ID the plan does not contain). Add it to `make test-structure`. This keeps the plan and the tracker from drifting apart.

### P0.2 Correct the record
- Edit `docs/md/CAPTAINS_LOG.md`: append a "Session 14 (P0)" entry that states the corrections in §1.2 of this plan. **Do not rewrite earlier sessions**; append only.
- Edit `README.md`: replace any statement that implies Phase 1 is complete with "Early development, foundations in place; see `docs/md/V1_PROGRESS.md`". Remove or qualify the sentence on CI until P0.5 lands.

### P0.3 Packaging
- Replace the hand-maintained `[tool.setuptools] packages = [...]` with automatic discovery:
  ```toml
  [tool.setuptools.packages.find]
  include = ["econkit*"]
  ```
- Declare `numpy>=1.25` and `scipy>=1.11` (D2) as required; add optional extras:
  ```toml
  [project.optional-dependencies]
  numba = ["numba>=0.59"]
  jax   = ["jax>=0.4.25", "jaxlib>=0.4.25"]
  data  = ["pandas>=2.0"]
  viz   = ["matplotlib>=3.7"]
  docs  = ["sphinx>=7", "myst-nb>=1.0", "furo", "sphinx-autodoc-typehints", "jupytext"]
  dev   = ["pytest>=8", "pytest-cov", "hypothesis>=6", "ruff>=0.6", "mypy>=1.10", "vulture>=2.11",
           "numba>=0.59", "pandas>=2.0", "matplotlib>=3.7"]
  all   = ["econkit[numba,jax,data,viz]"]
  ```
  The `dev` extra now includes `numba` so the 6 currently skipped tests run.
- Metadata: real `authors` (D8), `keywords`, `classifiers` (Python 3.10–3.12, Scientific/Engineering, Typing :: Typed), `[project.urls]` (Homepage, Source, Issues, Documentation).
- Version: single source in `econkit/_version.py`; `pyproject.toml` uses `dynamic = ["version"]` with `[tool.setuptools.dynamic] version = {attr = "econkit._version.__version__"}`; `econkit.__version__` re-exported.
- Check `python -m build && twine check dist/*` succeeds (build tools installed in the venv only).

### P0.4 pytest and coverage configuration
```toml
[tool.pytest.ini_options]
addopts = "-q --strict-markers"
testpaths = ["tests"]
markers = [
  "slow: long-running (deselect with -m 'not slow')",
  "numba: requires numba",
  "jax: requires jax",
  "network: requires internet (never run in CI by default)",
  "plot: needs matplotlib",
]
filterwarnings = ["error::DeprecationWarning:econkit.*"]

[tool.coverage.run]
source = ["econkit"]
branch = true
[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:", "raise NotImplementedError"]
```
Add Makefile targets: `coverage` (`pytest --cov --cov-report=term-missing --cov-report=xml`), `build` (sdist+wheel), `docs` (placeholder until P16), `bench` (placeholder until P15), `ci` unchanged but now depends on installing `.[dev]`.

### P0.5 Track and create CI (D3)
- Remove `.github/` from `.gitignore`. Keep a comment explaining why (the earlier exclusion was a user preference that D3 reverses).
- Create `.github/workflows/ci.yml`: matrix `os: [ubuntu-latest, macos-latest, windows-latest]` × `python: [3.10, 3.11, 3.12]`; steps: checkout, setup-python with pip cache, `pip install -e .[dev]`, `make ci`. Add a second job `no-numba` on ubuntu/3.11 that installs only `.[dev]` minus numba (`pip uninstall -y numba` after install) and runs `pytest -m "not numba"`. Add `python 3.13` as `continue-on-error: true`.
- Add `.github/pull_request_template.md` and `.github/ISSUE_TEMPLATE/{bug_report,feature_request}.md`.
- If you cannot confirm the workflow runs on GitHub (no network access), validate YAML locally with `python -c "import yaml,sys;yaml.safe_load(open(sys.argv[1]))" .github/workflows/ci.yml` and write "CI file created, first GitHub run unverified" in the log.

### P0.6 PyPI name check (D5)
- Run `pip index versions econkit`. If it prints versions, the name is taken: record in Blockers, set the working name unchanged for now, and stop-and-ask only at P17. If "no matching distribution" appears, record that the name appears free (still re-check at P17). If there is no network, record "unverified".

### P0.7 Backend roadmap document
- Regenerate `docs/md/BACKEND_ROADMAP.md` from architecture §19: the layer table, dispatch semantics, validation doctrine, and the exact status per operation (table of operation × backend with ✅/🟡/⬜ filled from `list_registered()`; write a small script `scripts/backend_matrix.py` that prints this table and run it to fill the document).
- Track this file in git (D4): remove it from `.gitignore`.

### P0.8 Housekeeping files
- `CHANGELOG.md` in Keep-a-Changelog format with an `[Unreleased]` section summarising what exists.
- `CONTRIBUTING.md`: dev setup (`make install`), gates, doctrine R3/R7/R8, commit style.
- `.pre-commit-config.yaml` (ruff, ruff-format, end-of-file-fixer, trailing-whitespace).
- Do **not** touch `LICENSE` (D1).

### P0.9 Quality baseline
- Run `make ci` and `make coverage`; paste the total and per-package coverage into the log. Do not chase coverage in P0.
- Record the 6 formerly skipped tests now pass (they need numba from the `dev` extra).

### P0.10 Pin the repo state
- Tag locally: `git tag v0.1.0-baseline` (local only; do not push). This tag is the archive for everything R21 deletes later, and the source of the pre-rewrite timings used in P5.10. Create it **before** P0.12 removes anything.

### P0.11 Phase extractor (context saving)
- `scripts/phase.py <n>` prints Section 2 (the operating protocol) plus the text of phase `P<n>` from `docs/md/V1_EXECUTION_PLAN.md`; `make phase P=5` writes it to `.phase/P5.md`. Add `.phase/` to `.gitignore`. Nothing generated is committed (a committed copy would be a second source of truth).
- Appendix A's prompt starts by running `make phase P=<n>`.
- Test (`tests/engineering/test_phase_script.py`): the output for `5` contains `## P5 — ` and `R21`, and does not contain `## P6 — `.

### P0.12 Lean tooling and baseline audit (R21)
- Add to `pyproject.toml`: `[tool.vulture] min_confidence = 80, paths = ["econkit"]`; ruff `select` extended with `F`, `ERA`, `ARG`, `C90` (`max-complexity = 12`); fix findings or add a per-line `noqa` with a reason.
- `scripts/check_usage.py`: for every name in `__all__` of each public module, require (a) a reference in at least one file under `tests/`, and (b) a reference in `econkit/` outside its defining module and `__init__`, or in an executed doc page. Exceptions go in `scripts/usage_allowlist.txt` with a one-line reason each (for example top-level entry points such as `solve`).
- `scripts/check_near_future.py`: implements the R21e rules against `docs/md/V1_EXECUTION_PLAN.md` (existing, non-retired step) and `docs/md/V1_PROGRESS.md` (step not yet ticked), and the review-window rule.
- `make deadcode` runs all three; `make ci` depends on it.
- **Baseline audit.** Run the tools on the tagged baseline. For each finding decide: *delete*, *keep because it has a caller*, or *tag NEAR-FUTURE with the consuming step*. Likely candidates, to be **confirmed by the tools, not assumed**: `StateSpec`/`Parameter` in `core/types.py` (delete); the delegating JAX wrappers for Tauchen/Rouwenhorst in `numerics/backends/discretization_jax.py` (delete: they add no JAX computation; P12 writes real ones; keep the JAX linear-interpolation wrapper if it computes with `jnp`); `gauss_laguerre_exp` (no planned consumer: delete); `golden_section_max` / `brent_max` (the plan's VFI is grid-based and has no caller for them: delete). Tests that exercised only deleted code are deleted and listed in the log (R9); every other test is untouched.
- Record the audit result (deleted / kept / tagged, with reasons) in the Captain's Log entry. The deletions of `canonical.py`, `simulate_markov_asset_policy`, `run_solve_simulate_moments_pipeline` and the `Model.equation` mechanism are **not** done here; they happen in the phases that provide their replacements (P5.5, P7.1, P11.10, P4.3), so the tests keep protecting behaviour until the new path exists.

**Acceptance tests.** `tests/engineering/test_phase_script.py` (new). `make deadcode` passes. Existing tests: all pass with `.[dev]` installed (0 skipped), except those removed with the dead code listed in the log.

**Definition of Done.**
```bash
pip install -e ".[dev]" && make ci && make coverage
python -c "import econkit; print(econkit.__version__)"
python -m build && twine check dist/*
git check-ignore -q .github || echo ".github tracked"
test -f docs/md/V1_EXECUTION_PLAN.md -a -f docs/md/V1_PROGRESS.md
make deadcode && make phase P=5 && grep -q R21 .phase/P5.md
```
**Session budget.** 1–2. **Pitfalls.** Do not rename packages. If `pip index` is unavailable, don't guess. Do not enable the tests of `numba` on Windows if Numba wheels fail there; mark the `windows` job `continue-on-error` and log it.

---

## P1 — Core contract: errors, RNG, JIT, config

**Goal.** Create only the shared substrate that later phases call: one error hierarchy, one RNG convention, one JIT decorator, validated backend selection with introspection, provenance, and the written API policy. Remove the core code that has no owner.

**Prerequisites.** P0.

**Files.** `econkit/core/errors.py`, `utils/{__init__,rng,validation,jit,provenance}.py`, `econkit/backends.py`, `config.py` (extend), `econkit/__init__.py`, `docs/md/API_POLICY.md`; tests in `tests/core/`, `tests/utils/`, `tests/api/`.

### P1.1 Error hierarchy (`core/errors.py`)
```python
class EconKitError(Exception): ...


class ValidationError(EconKitError, ValueError): ...


class ModelSpecificationError(ValidationError): ...


class BackendError(EconKitError): ...


class BackendUnavailableError(BackendError): ...


class ConvergenceError(EconKitError): ...  # raised only in strict mode


class ConvergenceWarning(UserWarning): ...
```
Each class has a raising site in a phase that exists. The classes whose first raising site is later are defined there: `DeterminacyError` (P9.4) and `IdentificationError` (P11.9). Migrate existing `ValueError` raises only where the type matters to callers (dispatcher, config, model validation); `ValidationError` subclasses `ValueError`, so existing `pytest.raises(ValueError)` tests keep passing.

### P1.2 *(retired — `Solution` is created in P5.1, where `solve()` first returns it)*

### P1.3 RNG contract (`utils/rng.py`)
```python
def as_generator(rng: np.random.Generator | int | None = None) -> np.random.Generator   # PCG64
def spawn(rng, n) -> list[np.random.Generator]      # SeedSequence.spawn; NEAR-FUTURE(P7.1)
```
Rule: any function that draws random numbers takes `rng` and calls `as_generator`. The old `random_seed` argument of `simulate_markov_asset_policy` disappears when that function is replaced in P7.1 (no alias).

### P1.4 JIT decorator with fallback (`utils/jit.py`)
```python
def jit(*args, parallel: bool = False, cache: bool = True, fastmath: bool = False, **kw):
    """Numba njit if available and enabled; otherwise returns the pure-Python function."""


prange = numba.prange if numba else range
NUMBA_AVAILABLE: bool
```
Honour `ECONKIT_DISABLE_JIT=1` (used by coverage and debugging). `fastmath` defaults to **False**; flag any kernel that wants it and justify in a comment. Refactor the existing `backends/*_numba.py` files to import from here, preserving their optional-import behaviour.

### P1.5 Validation helpers (`utils/validation.py`)
`check_array(x, name, ndim=None, shape=None, finite=True)`, `check_increasing(x, name, strict=True)`, `check_stochastic_matrix(P, name, tol=1e-10)`. Each raises `ValidationError` naming the argument and summarising the offending value. Replace the equivalent hand-written checks in the public numerics and solver entry points (replace, do not add checks that change behaviour). Further helpers are added by the phase that first needs them.

### P1.6 *(retired — the deprecation helper is written when the first deprecation happens, after 1.0; R10)*

### P1.7 Config and backend introspection
- `Config.set_backend(name)` validates: unknown name → `ValidationError`; known but not installed → `BackendUnavailableError` when `strict_backend=True`, otherwise a single `UserWarning` and fall back to NumPy. There is one way to choose a backend globally (`set_backend`) and one per call (`backend=` argument); no context manager.
- `econkit/backends.py`:
  ```python
  def is_available(name: str) -> bool
  def available_backends() -> list[str]
  def describe() -> str            # operation x backend table + versions of Python/numpy/scipy/numba/jax (from provenance)
  ```
  `describe()` is also what the issue template asks bug reporters to paste (P0.5).

### P1.8 Core cleanup and API policy
- Delete `StateSpec` and `Parameter` from `core/types.py` if P0.12 has not already (confirm with `grep -rn`).
- **Protocols (R21d).** Inspect `core/protocols.py`. Keep a protocol only if at least two implementers exist in the v1 scope; the `Solver` protocol qualifies (VFI, EGM, perturbation, equilibrium) and is created in P5.1 where the registry needs it. Delete the `Simulation`, `Solution` and `Estimator` protocols (each has at most one concrete implementation; the concrete classes are created in P5.1, P7.1, P11.3). Delete `core/types.py` and `core/protocols.py` if they end up empty. Port their tests (R9).
- Write `docs/md/API_POLICY.md`: what is public (names in `__all__` of documented modules), what is stable at 1.0 (all of it), the deprecation window (two minor versions), versioning (SemVer), and that compiled layers are implementation details exempt from the API promise.
- `econkit/__init__.py` re-exports exactly the names in `API_POLICY.md`: `Model, Param, State, Control, AR1Shock, config, backends, errors, __version__` now; `solve, Solution` (P5.1) and `simulate` (P7.1) are added with their phases. Create no `experimental` package: nothing in v1 needs one.

### P1.9 Provenance helper (`utils/provenance.py`)
`collect_provenance() -> dict`: econkit/numpy/scipy/numba/jax versions, Python, platform, git commit if available (guarded `subprocess`), UTC timestamp. Called by `backends.describe()` now and by `Solution.metadata` (P5.1) and `Experiment` (P14.2) later.

**Acceptance tests.**
- `tests/core/test_errors.py`: hierarchy relations (`issubclass(ValidationError, ValueError)`), warnings are `Warning` subclasses.
- `tests/utils/test_rng.py`: `as_generator(3)` reproducible; `spawn` streams are independent (pairwise correlation of 10⁵ draws below 0.01 in absolute value) and reproducible.
- `tests/utils/test_jit.py`: with `ECONKIT_DISABLE_JIT=1` a jitted function equals the Python function result; with numba present the compiled and pure-Python results are equal (assert on results only, never on timing).
- `tests/utils/test_validation.py`: messages contain the argument name.
- `tests/core/test_config.py`: unknown backend raises; unavailable backend (monkeypatch `NUMBA_AVAILABLE=False`) raises or warns per strict flag; `describe()` lists the registered operations.
- `tests/api/test_public_api.py`: `set(econkit.__all__) == EXPECTED` (a literal set in the test; changing the public API requires editing the test deliberately).

**Definition of Done.** `make ci`; `make coverage` with `core` + `utils` ≥ 90 %; `python -c "import econkit as ek; print(ek.backends.describe())"` prints a table; `grep -rn "StateSpec" econkit tests` returns nothing.
**Session budget.** 1–2. **Pitfalls.** Don't change numerics; this phase is plumbing. Don't add a helper "because it will probably be needed".

---

## P2 — Numerics I: grids and 1-D interpolation

**Goal.** The grid and interpolation toolbox every solver uses: NumPy reference plus Numba parity on hot paths. Only 1-D linear interpolation is needed in v1 (one continuous endogenous state plus discrete shocks); higher-dimensional interpolation returns with its first consumer (Krusell–Smith, Part E).

**Prerequisites.** P1.

**Files.** `econkit/numerics/grids.py`, `interpolation.py` (extend), `backends/interp_numpy.py`, `backends/interp_numba.py`; tests under `tests/numerics/`; math in `docs/latex/econkit_math.tex`.

### P2.1 Grid constructors (cold path, NumPy only)
```python
uniform_grid(lo, hi, n) -> ndarray
exp_grid(lo, hi, n, curvature) -> ndarray     # x = lo + (hi-lo) * (exp(k u) - 1)/(exp(k) - 1), u uniform on [0,1]; dense near lo
make_grid(kind, lo, hi, n, **kw)              # kind in {"uniform", "exp"}; called by State.grid_type (P4.1)
```
Properties checked: strictly increasing, endpoints exact (`g[0]==lo`, `g[-1]==hi`), length `n`, curvature → 0 recovers uniform (tolerance 1e-8). One curved grid is enough: `exp_grid` handles any lower bound, including negative borrowing limits.

### P2.2 1-D linear interpolation: harden and accelerate
Existing operation `("linear_interp_1d", backend)`. Required behaviour: strictly increasing nodes (validated); extrapolation modes `"clamp"` (default; used by simulation) and `"linear"` (used by EGM, P6.1); vectorised over query points of any shape; binary-search bracketing. Provide one Numba scalar kernel `interp_linear_scalar(x, xp, fp, mode)` that is inlined by the other compiled kernels (Bellman, simulation, histogram update) and by the array operation, so there is a single bracketing/weighting implementation. Cross-backend test: random nodes/values/queries, `rtol=0, atol=1e-13`.

### P2.3 – P2.6 *(retired)*
Multilinear interpolation (2-D/3-D), cubic splines, Chebyshev approximation and the policy-interpolation helper have no consumer in v1 (R21). They return with Krusell–Smith (multilinear) and projection methods (splines, Chebyshev) per Part E. EGM's monotonicity check lives in P6.2.

### P2.7 Registration and dispatch coverage
Every hot-path op has NumPy and Numba implementations registered. `tests/numerics/test_backend_parity.py` is **table-driven** from the registry: for every `(op, "numba")` registered, a parity case must exist; adding an op without a parity test fails a meta-test (`test_every_registered_op_has_parity_case`).

### P2.8 *(retired — the old micro-benchmark script is deleted in P15.1 when the harness replaces it)*

### P2.9 Math doc and docstrings
Add to `econkit_math.tex`: linear interpolation formula and the exponential grid map. All new public functions have an `Economics note:` line (for example, "Grid curvature concentrates points near the borrowing constraint where the policy function has its kink").

### P2.10 Property-based tests (Hypothesis)
Profile `ci`: `derandomize=True`, `max_examples=100`, `deadline=None`. Properties of `linear_interp_1d`: exact at the nodes; output lies between the neighbouring node values; monotone data gives monotone output; result does not depend on query order; `"linear"` extrapolation reproduces an affine function exactly. Properties of grids: strictly increasing and exact endpoints for random `lo < hi`, `n ≥ 2`, `curvature`.

**Acceptance (analytic).** Interpolating a concave CRRA value function `u(c)=c^{1-σ}/(1-σ)` on `exp_grid(…, 200)` vs `uniform_grid(…, 200)` gives lower max error near the lower bound for the exponential grid (assert error_exp < error_uniform on the sub-interval [lo, lo+0.1(hi−lo)]).
**Definition of Done.** `make ci`; `pytest -m numba tests/numerics -q`; coverage ≥ 90 % on `numerics`.
**Session budget.** 1–2. **Pitfalls.** `searchsorted` on grids with repeated values; extrapolation at exactly the upper node; Numba type unification with int vs float queries (always cast to `float64`).

---

## P3 — Numerics II: the iteration driver, Jacobians, Markov tools, quadrature

**Goal.** The remaining building blocks, each with a named consumer: the **one** fixed-point iteration driver that every solver uses, numerical Jacobians, Markov-chain tools, and the quadrature rules the model layer needs. Root finding, steady-state solving and optimisation use SciPy directly (R21f).

**Prerequisites.** P2.

**Files.** `numerics/fixedpoint.py`, `numerics/jacobian.py`, `numerics/markov.py`, `numerics/integration.py` (extend), backends and tests.

### P3.1 *(retired)*
Scalar root finding, vector Newton and the bracketed solvers are `scipy.optimize.brentq` / `scipy.optimize.root` called where needed (equilibrium outer loop P8.1, steady state P9.2). The vectorised bisection that time iteration would have used is gone with time iteration (Part E).

### P3.2 Numerical Jacobian
`numerical_jacobian(F, x, eps=1e-6, relative=True)`: central differences with step `eps·max(1,|x_i|)`. Consumers: linearisation (P9.3), SMM Jacobian (P11.3), cross-check of the autodiff Jacobian (P12.3). Test: Jacobian of a polynomial map matches the analytic Jacobian to 1e-8; two step sizes agree to 1e-6.

### P3.3 The fixed-point iteration driver
```python
def fixed_point(operator, x0, *, tol, max_iter, norm="sup", damping=1.0,
                callbacks=(), name="fixed_point") -> FixedPointResult  # x, converged, iterations, residual, history
```
**This is the only convergence loop in the library.** Consumers: VFI (P5.1 moves the existing `value_function_iteration` loop onto it, preserving its relaxation and hook behaviour), EGM (P6.1), the Lucas-tree price functional (P10.9). Non-convergence warns (`ConvergenceWarning`) and sets `converged=False`; strict mode raises `ConvergenceError`. Callbacks receive `(iteration, x, residual)` and may return `True` to stop. Tests: `T(x)=cos(x)` converges to 0.7390851332151607 (1e-10); damping slows but still converges on a contraction; callback stop; the existing VFI relaxation/hook tests are re-pointed at the driver (R9).

### P3.4 Markov-chain utilities
```python
stationary_distribution(P) -> ndarray              # GTH (Grassmann–Taksar–Heyman) algorithm: exact, no eigensolver
simulate_markov(P, n, rng, init=None) -> int array # vectorised inverse-CDF; Numba kernel shared with the simulation engine (P7.1)
```
Tests: 2-state chain with transition probabilities `p,q` has stationary `(q/(p+q), p/(p+q))` to 1e-14; Rouwenhorst chains: stationary mean ≈ 0 and variance ≈ `σ²/(1−ρ²)` (relative error < 1e-10 for n=7); Tauchen within 5 % for n=21, m=3; simulated long-run frequencies within 4 standard errors of the stationary distribution for T=2·10⁵ (block-mean s.e.). The AR(1) moment formulas are computed inline in the tests (they are test oracles, not library code).

### P3.5 *(retired — Tauchen–Hussey, `discretize_iid_normal` and library moment helpers have no consumer)*

### P3.6 Quadrature addition
Add Gauss–Legendre (consumer: uniform IID shocks, P4.1). Existing Gauss–Hermite stays (normal and lognormal IID shocks). Tests: GL integrates polynomials of degree ≤ 2n−1 exactly (1e-12); GH `E[X²]=σ²+μ²` exactly at n ≥ 2 and `E[exp(X)]=exp(μ+σ²/2)` for n=10 within 1e-10.

### P3.7 *(retired — QMC draws have no consumer)*

### P3.8 Remaining baseline numerics audit
Confirm every other registered operation has a caller or a NEAR-FUTURE tag (`tauchen`, `rouwenhorst`: P4.1; `gauss_hermite_normal`: P4.1; `expectation_discrete`: check, delete if VFI/EGM use plain matrix products) and a parity test including edge cases. Remove what the tools say is unused. Keep one strict-backend test that uses a genuinely unregistered operation so the strict-mode failure path stays covered.

### P3.9 Property-based tests (Hypothesis)
Markov: random row-stochastic matrices give a stationary distribution that is nonnegative, sums to 1 and satisfies `πP = π` to 1e-10; `simulate_markov` states are always in range. `fixed_point`: the returned `x` satisfies `‖T(x)−x‖ ≤ tol` for random contractions `T(x)=Ax+b` with `‖A‖<1`. Quadrature: weights are positive and sum to the integral of the weight function.

**Definition of Done.** `make ci`; Numba parity tests for every hot op; coverage ≥ 90 % `numerics`; math doc updated for the GTH algorithm and the driver's convergence criterion.
**Session budget.** 1–2. **Pitfalls.** The driver must not grow options "in case"; every option it has is used by one of its three consumers.

---

## P4 — Model specification system

**Goal.** Make the DSL real (resolves TD2): every solver obtains, from a `CompiledModel`, the numerical ingredients it needs (utility, resources, transition, grids, shocks) instead of hard-coded names. The old free-form equation mechanism is replaced, not kept alongside.

**Prerequisites.** P1, P3.

**Files.** `econkit/model/{specs,shocks,base,compiled,problems,validation,fingerprint}.py`, tests `tests/model/`.

### P4.1 Shocks and state spaces
Extend `specs.py`:
```python
@dataclass(frozen=True) class AR1Shock: rho; sigma_e; n_states; mean=0.0; log=False; method: Literal["rouwenhorst","tauchen"]="rouwenhorst"
@dataclass(frozen=True) class MarkovShock: values: ndarray; P: ndarray
@dataclass(frozen=True) class IIDShock: kind: Literal["normal","lognormal","uniform"]; params: dict; n_nodes: int
```
`log=True` means the state takes values `exp(z)` (productivity). `State.shock` accepts any of the three; `compile()` discretises each shock (Rouwenhorst/Tauchen for AR(1); Gauss–Hermite for normal/lognormal; Gauss–Legendre for uniform) and stores `(values, P)` on the `CompiledModel` (IID shocks get identical rows). `State.grid_type` is passed to `make_grid` (P2.1); add `grid_curvature`.

### P4.2 Parameter sets
`ParameterSet`: ordered mapping built from `Param` objects, with `.vector(free=True)`, `.from_vector(x)`, `.bounds()`, `.names(free)`, and `CompiledModel.with_params(**overrides)` returning a **new** compiled model (models are immutable; copy-on-write). Priors and parameter transforms are not here: they are added in P11 where estimation first uses them (`Param.prior`, `Param.transform`).

### P4.3 Problem templates (replaces `Model.equation`)
A model declares what kind of problem it is; each template is the contract between the model and a family of solvers. Three templates exist in v1; `dsge` is added by P9.1, its first consumer.

**`saving`** — one continuous endogenous asset, discrete or discretised shocks, one control. Covers consumption–savings, Brock–Mirman, RBC (inelastic labour), Aiyagari and Huggett households. Growth models and consumption–savings models are the same template; they differ only in `resources`.
```python
model.set_problem(
    "saving",
    asset="assets",  # name of the State holding end-of-period assets a
    shock="income",  # name of the State that carries the shock process (omit if deterministic)
    control="consumption",  # name of the Control (policy output name)
    utility=lambda c, p: ...,
    marginal_utility=lambda c, p: ...,  # required by EGM and Euler errors
    inverse_marginal_utility=lambda x, p: ...,  # required by EGM
    resources=lambda a, z, p: (
        ...
    ),  # cash-on-hand m(a, z) to split between consumption and next assets
    marginal_return=lambda a, z, p: ...,  # d m / d a; required by EGM and Euler errors
    discount="beta",
)  # name of the discount-factor parameter
```
Examples: consumption–savings `resources = (1+p.r)*a + z`, `marginal_return = 1+p.r`; Brock–Mirman `resources = z * a**p.alpha`, `marginal_return = p.alpha * z * a**(p.alpha-1)`; RBC `resources = z*a**alpha + (1-delta)*a`.

**`finite_dp`** — finite states and actions (search models):
```python
model.set_problem(
    "finite_dp",
    reward=lambda p: R,  # array [state, action]
    transition=lambda p: T,  # array [action, state, next_state], rows sum to 1
    discount="beta",
    state_labels=[...],
    action_labels=[...],
)
```

All callables are plain NumPy functions of arrays (they broadcast). They run **outside** the compiled kernels: the kernels receive arrays computed from them, so a user never has to write Numba-compatible code and no JIT-compatibility check is needed. `compile()` evaluates each callable on a small test input and raises `ModelSpecificationError` naming the template field if the output is non-finite or has the wrong shape.

Delete the old `EquationSpec`, `Model.equation(...)`, and `CompiledModel.call_equation` in this step (nothing in the solvers calls them; R21c). Port their tests to `set_problem` (R9).

### P4.4 *(retired — no general `constraint` API; the borrowing limit is the lower bound of the asset `State`, which fixes the inconsistent `a_min` handling in the old `canonical.py`)*

### P4.5 Validation
`validation.py`: every state has bounds and a grid; every shock matrix is a valid stochastic matrix (`check_stochastic_matrix`); parameter values within bounds; no duplicate names; the template fields required by the requested solver exist (for `method="egm"`: the three marginal-utility fields). Errors are collected and raised together as one `ModelSpecificationError`.

### P4.6 Introspection and fingerprint
`CompiledModel.describe() -> str` (also used by `__repr__`), `.fingerprint() -> str` (SHA-256 over canonical JSON of parameters, grids, shock matrices and the *source text* of the template callables via `inspect.getsource`, falling back to the qualified name). Consumers: `Solution.metadata` (P5.1) and `Experiment` (P14.2). Test: identical models give identical fingerprints; changing β by 1e-12 changes the fingerprint.

### P4.7 *(retired — YAML model files, the callable library and `from_dict` have no consumer in v1; `Experiment` reproduces runs from a built-in model name plus parameter overrides, or from an importable builder)*

### P4.8 Port existing model tests
The existing tests in `tests/model/` pass unmodified except those that exercised `Model.equation`, which are ported to `set_problem` (R9).

### P4.9 Docs
Docstrings with `Economics note:`; a `docs/md/model_dsl.md` page with a complete worked example (consumption–savings in about 25 lines) and the three-line difference for a growth model.

### P4.10 Builder hygiene
`Model.copy()` (consumer: reform functions in P14.3) and `__repr__`. No fluent aliases: the existing builder methods are the one way to build a model.

**Acceptance tests.** All of the above, plus: a consumption–savings model built from the DSL and compiled has `resources(a, z) == (1+r)a + z`; `with_params(beta=0.9)` leaves the original model unchanged (immutability); several simultaneous validation problems are reported in one raise; a `resources` callable returning the wrong shape raises `ModelSpecificationError` naming `resources`.
**Definition of Done.** `make ci`; `model` coverage ≥ 90 %; `docs/md/model_dsl.md` exists; `grep -rn "call_equation\|EquationSpec" econkit tests` returns nothing.
**Session budget.** 2–3. **Pitfalls.** Source-text fingerprinting fails for lambdas defined in a REPL; always fall back gracefully. Keep `CompiledModel` frozen.

---

## P5 — Solver core: `solve()`, `Solution`, vectorized and Numba VFI

**Goal.** Make value function iteration fast and model-driven, and introduce the single public entry point `econkit.solve(model, method=..., **options)` with its result type. Resolves TD1, TD2, TD3, TD8. The pure-Python baseline solvers are deleted, not wrapped.

**Prerequisites.** P2, P3, P4.

**Files.** `econkit/core/solution.py`, `solvers/api.py`, `solvers/dp/{__init__,bellman,kernels_numpy,kernels_numba,howard}.py`, `solvers/vfi.py` (rewritten onto the driver), `solvers/diagnostics.py`, tests `tests/solvers/`. Deleted: `solvers/canonical.py`.

### P5.1 `Solution`, the `Solver` protocol and `solve()`
```python
@dataclass
class Solution:
    method: str; converged: bool; iterations: int; residual: float; elapsed: float; backend: str
    model_name: str | None; params: dict[str, float]
    grids: dict[str, ndarray]; policy: dict[str, ndarray]; value: ndarray | None
    diagnostics: dict[str, Any]; metadata: dict[str, Any]   # versions, model fingerprint, resolved config
    def summary(self) -> str
```
`to_dict`/`save`/`load` arrive in P14.2 and `policy_function` in P7.1, where they are first called.

```python
class Solver(Protocol):                      # implemented by vfi, egm, perturbation, equilibrium
    def __call__(self, model: CompiledModel, **options) -> Solution: ...
SOLVERS: dict[str, Solver]                   # populated by each solver module
def solve(model: CompiledModel, method: str = "auto", *, backend: str | None = None, **options) -> Solution
```
- **One way to configure.** Options are keyword arguments. Each method owns a frozen config dataclass (`VFIConfig` exists; `EGMConfig` P6.1, `PerturbationConfig` P9.4) that is the single source of truth for option names and defaults; `solve` resolves `**options` into it in one function and records the resolved config in `Solution.metadata`. Unknown option names raise `ValidationError` listing the valid ones. Config classes are not part of the public API.
- `method="auto"`: VFI until P6.7 teaches it to prefer EGM. Unknown method → `ValidationError` listing available methods. Non-convergence warns `ConvergenceWarning` and sets `converged=False` (strict mode raises `ConvergenceError`).
- **One iteration loop.** Move the loop of the existing `value_function_iteration` onto `numerics.fixed_point` (P3.3), preserving its relaxation and hook behaviour, then delete `value_function_iteration`, `VFISolution` and `VFISolver`; `solve(model, method="vfi")` replaces them. Port their tests (R9). `Solution` is added to the top-level exports and `API_POLICY.md`.

### P5.2 Payoff tensor and NumPy Bellman reference (`saving` problems)
- Build once per solve: `U[i,j,k] = utility(resources(a_i, z_j) − a_k)` with `-inf` where consumption `≤ 0`, using the model's callables with NumPy broadcasting (this is the only place the callables are evaluated, so they need no Numba compatibility). Assert every state `(i, j)` has at least one feasible choice, else `ModelSpecificationError`.
- Memory guard: tensor size `n_a² · n_z · 8` bytes against `max_memory_gb` (default 2); if exceeded raise `ValidationError("grid too large for the VFI payoff tensor; use method='egm'")`. No chunking machinery.
- One iteration: `EV[k,j] = β Σ_j' P[j,j'] V[k,j']` (a matrix product), `V_new[i,j] = max_k (U[i,j,k] + EV[k,j])`, `policy_index = argmax_k`. This is the operator handed to `fixed_point`.
- There is **no continuous-choice refinement** in v1 (it would need user callables inside compiled loops). VFI policies are exact on the grid; their accuracy is O(grid spacing). EGM (P6) is the accurate method.
- `finite_dp` problems use the same driver with the tabular operator `V = max_a (R[:,a] + β T[a] @ V)` in NumPy (no Numba: state spaces are small).

### P5.3 Numba Bellman kernel (hot path)
Register `("bellman_saving", "numpy")` (the reference above) and `("bellman_saving", "numba")`: `prange` over the shock index `j`, loops over `i` and `k`, with the monotone-policy shortcut of P5.4. Inputs are arrays only (`U`, `EV`, previous argmax), so the kernel contains no user code.

### P5.4 Howard acceleration and monotone search
- **Howard:** after each maximisation do `howard_steps` (default 10) policy evaluations `V ← U[i,j,k*] + β Σ_j' P[j,j'] V[k*,j']` (a NumPy gather and matrix product; no maximisation).
- **Monotone search:** for the savings policy increasing in `a`, restrict `k` for `a_{i+1}` to `k ≥ k*_i`. Default on for `saving` problems (the template implies a monotone policy for concave utility). Parallelise over `j`, not `i`, when this is on.
- Test: all four combinations (plain, Howard, monotone, both) give identical policy indices (exact) and values within `tol`.

### P5.5 Delete the baseline solvers and port their tests (resolves TD1, TD2, TD8)
- Delete `solvers/canonical.py`: the deterministic and stochastic-income helpers, and the liquid/illiquid prototype. The prototype has no consumer in v1; a proper two-asset solver returns in Part E.
- Port the tests of the deterministic and stochastic-income helpers to models built with the DSL and `solve(...)`: the assertions about shapes, feasibility, monotonicity, convergence and policy values keep; assertions about the deleted result types and the prototype are removed and listed in the Captain's Log (R9). The deterministic case is `n_z = 1`.
- Existing helpers that `solve` still needs (Bellman residual, residual-history summaries, policy monotonicity check) move to `solvers/diagnostics.py` and are called by `solve`.

### P5.6 Diagnostics
`Solution.diagnostics` contains `bellman_residual`, `monotone_policy`, `max_feasibility_violation`, `n_binding` (share of states at the borrowing constraint) and `residual_history`. The existing schema-versioned convergence-artifact exporter: if `solve` calls it, keep it as the serialisation of `residual_history`; if nothing calls it, delete it and let `Solution.to_dict` (P14.2) carry the history. Decide with the tools.

### P5.7 Verbosity
`verbose=True` is a built-in callback for `fixed_point` printing one line every 10 iterations (iteration, sup-norm change, elapsed). No other progress mechanism.

### P5.8 Analytic validation: Brock–Mirman via VFI
Growth model (one-sector, log utility, Cobb–Douglas, full depreciation) expressed as a `saving` problem with `resources = z k^α`: true policy `k' = αβ z k^α`. Solve with VFI on a 500-point grid over `[0.05 k*, 2 k*]` (`k*` = deterministic steady state, `(αβ)^{1/(1−α)}`), `z` from Rouwenhorst (7 states, ρ=0.9, σ_ε=0.02). Assert the grid policy is within **two grid spacings** of `αβ z k^α` on interior states, and that the value function satisfies the Bellman equation to `tol`. Record the measured errors in the log; tighten the test to 3× the measurement but never loosen it.

### P5.9 Analytic validation: permanent income
Consumption–savings with CRRA utility, deterministic constant income `y` (`n_z = 1`) and `β(1+r) = 1`. The Euler equation gives constant consumption, so the policy is `a'(a) = a` and `c(a) = y + r a` as long as the borrowing limit does not bind. Use `a_min = 0`, `y = 1`, `r = 0.03`, `β = 1/1.03`, `σ = 2`. Assert `|k*(i) − i| ≤ 1` for every grid index (the optimum lies on the grid, so the index should equal `i` up to a tie).

### P5.10 Performance acceptance (measured, not asserted)
Benchmark the VFI solve at `(n_a, n_z) = (200, 7)` and `(1000, 7)` comparing: (a) the **old pure-Python solver, measured by checking out the tag `v0.1.0-baseline` into a temporary worktree** (no legacy copy is kept in the repo), (b) vectorised NumPy, (c) Numba exhaustive, (d) Numba monotone + Howard. **Targets** (R19): (d) ≥ 50× faster than (a) at (200,7); ≥ 5× faster than (b) at (1000,7). If missed, profile, fix obvious causes (allocation in the inner loop, non-contiguous arrays, parallel overhead), record the real numbers in `docs/md/benchmarks.md`, and state the shortfall. **Never assert speed in tests.**

**Acceptance tests.** `tests/solvers/test_api.py` (method registry, option validation, error messages, warnings), `test_dp_numpy_numba_parity.py` (policy indices exactly equal, values 1e-12), `test_brock_mirman_vfi.py`, `test_permanent_income_vfi.py`, `test_howard_monotone_equivalence.py`, and the ported canonical tests in `test_vfi_consumption_savings.py`.
**Definition of Done.** `make ci`; `pytest -m numba tests/solvers`; `solvers` coverage ≥ 90 %; `grep -rn "canonical\|VFISolution\|value_function_iteration" econkit tests` returns nothing.
**Session budget.** 3–4 (splittable at P5.3/P5.5). **Pitfalls.** `-inf` payoffs in `argmax` can pick infeasible choices when everything is infeasible — the feasibility assert in P5.2 prevents it. Numba and `inf` + `fastmath` do not mix. Keep `float64`.

---

# Part C — Phases P6 to P12: Solvers, simulation, equilibrium, perturbation, models, estimation, JAX

---

## P6 — EGM

**Goal.** Add the endogenous grid method (Carroll 2006) for every `saving` problem, with Euler-error diagnostics, and make `solve(method="auto")` use it. Time iteration is not built: EGM solves the same Euler-equation problems without root finding, and no v1 model needs time iteration (Part E). **Review-2 gate follows this phase.**

**Prerequisites.** P5.

**Files.** `solvers/egm.py`, `solvers/diagnostics.py` (Euler errors), tests `tests/solvers/`, `tests/golden/`.

### P6.1 EGM algorithm (infinite horizon, any `saving` problem with `marginal_utility`, `inverse_marginal_utility`, `marginal_return`)
Represent the policy as consumption `C[i,j]` on the asset grid `a_i` and shock states `z_j`; this array is the iterate handed to `fixed_point` (P3.3). The operator is:
1. For each shock state `j'`, build the interpolant `c_{j'}(m)` through the knots `(m_{i,j'}, C[i,j'])` with `m_{i,j'} = resources(a_i, z_{j'})`, using linear interpolation with **linear extrapolation** (`linear_interp_1d`, P2.2). `resources` must be increasing in `a` (validated).
2. On the exogenous grid of end-of-period assets `a'_k = a_k`, compute for every current state `j`:  
   `rhs[k,j] = β Σ_{j'} P[j,j'] · marginal_return(a'_k, z_{j'}) · u'( c_{j'}( resources(a'_k, z_{j'}) ) )`,  
   `c_end[k,j] = (u')^{-1}(rhs[k,j])`, and the endogenous cash-on-hand `m_end[k,j] = c_end[k,j] + a'_k`.
3. Borrowing constraint: for current cash-on-hand below `m_end[0,j]` consumption is `m − a_min`. Build the new interpolant for each `j` from the knots `(m_min_j, m_min_j − a_min)` (where `m_min_j = resources(a_min, z_j)`, included only if it lies below `m_end[0,j]`) followed by `(m_end[k,j], c_end[k,j])`.
4. New `C[i,j]` = new interpolant evaluated at `resources(a_i, z_j)`. Savings policy `a'(a_i, z_j) = resources(a_i,z_j) − C[i,j]`.
Convergence criterion: sup-norm change of `C` below `tol` (default 1e-10). `Solution.value` is `None` (welfare in P14.4 evaluates the policy). Cash-on-hand, not assets, is the interpolation variable; interpolating in assets silently reintroduces the root finding the method exists to avoid.
The only compiled operation is `linear_interp_1d` (P2.2); everything else is NumPy array arithmetic, so no separate EGM kernel is registered. Cross-backend test: the full `solve(method="egm")` result agrees between `numpy` and `numba` to 1e-12.

### P6.2 Monotone endogenous grid
Assert `m_end[:,j]` is strictly increasing in `k` for every `j`. If not, the problem is non-concave: raise `ModelSpecificationError("endogenous grid is not monotone; use method='vfi'")`. No fallback algorithm. Test: strictly increasing for the default consumption–savings calibration; a deliberately non-concave utility raises.

### P6.3 *(retired — time iteration; see the goal above)*

### P6.4 Euler-equation errors
`euler_errors(solution) -> ndarray` in `solvers/diagnostics.py`: the unit-free error `log10 | 1 − c_implied / c |` over interior states (excluding constrained points), where `c_implied = (u')^{-1}(β Σ_{j'} P[j,j'] R(a',z_{j'}) u'(c(a',z_{j'})))`, valid for solutions from **either** VFI or EGM. Stored in `Solution.diagnostics["euler_error_max"]` and `["euler_error_mean"]`.

### P6.5 Analytic validation: Brock–Mirman and permanent income via EGM
- **Brock–Mirman.** Known solution `k' = αβ z k^α`, `c = (1 − αβ) z k^α` for any process for `z`. The model of P5.8 solved by EGM on a 200-point grid (`n_z = 7`, ρ=0.9, σ_ε=0.02): max relative consumption error over interior states < 1e-3 (a placeholder: EGM's error comes from interpolation, O(Δ²); tighten to 3× the measurement per the P5.8 rule).
- **Permanent income.** The P5.9 setup: `max |a'(a) − a| < 1e-6` on the whole grid.

### P6.6 Cross-method consistency
For the stochastic-income consumption–savings model (n_a = 300, n_z = 7): VFI and EGM savings policies agree within **two grid spacings** on the interior; EGM's `euler_error_max` is at least as small as VFI's (assert `egm ≤ vfi + 0.5`, log10 units); `numpy` vs `numba` EGM identical to 1e-12.

### P6.7 Wire-up
`solve(model)` with `method="auto"` chooses EGM when the `saving` problem provides `marginal_utility`, `inverse_marginal_utility` and `marginal_return`, otherwise VFI; `finite_dp` problems always use VFI. `Solution.method` records what actually ran. `EGMConfig` (frozen dataclass: `tol`, `max_iter`, `damping`) is the single source of EGM option names.

### P6.8 Golden-file regression tests
Store small reference solutions (policy arrays only, `.npz`, under `tests/golden/`) for two canonical solves — EGM and VFI of the default consumption–savings model — and compare with `rtol=1e-8`. Updating a golden file requires `pytest --update-golden`, a CHANGELOG line and a reason in the log. (P9.9 adds the perturbation golden file.) Golden files catch silent numerical drift across refactors and dependency upgrades everywhere the analytic benchmarks don't look.

**Benchmark (record, don't assert).** EGM at (n_a, n_z) = (1000, 7) vs Numba VFI: expect ≥ 20× faster; record the real ratio.
**Definition of Done.** `make ci`; Brock–Mirman and permanent-income tests pass for both methods; Euler-error diagnostics recorded in the log; coverage floors.
**Session budget.** 2. **Pitfalls.** Extrapolation above the grid must be linear, not clamp, or high-wealth consumption is biased. `resources` must be increasing in `a` or the interpolation knots are not sorted.

> **Review-2 (human gate).** Stop here and publish a short summary: analytic-benchmark errors, measured speedups, `Solution` API shape, and the list of `NEAR-FUTURE` tags. Ask for approval of names and signatures before P7. If the human is unavailable, continue but record "Review-2 deferred" in the Blockers section.

---

## P7 — Simulation engine and distributions

**Goal.** Simulate solved models (time series and panels) and compute stationary distributions without simulation noise (Young 2010). Replaces the minimal `simulate_markov_asset_policy` (deleted here, not wrapped).

**Prerequisites.** P6.

**Files.** `simulation/{simulate,distribution,moments,result}.py`, tests `tests/simulation/`. Deleted: `simulation/policy.py`. Also adds `Solution.policy_function(name)` to `core/solution.py` (an interpolating callable; first caller is `simulate`).

### P7.1 `SimulationResult` and `simulate()`
```python
simulate(solution, *, n_periods, n_agents=1, burn_in=0, init=None,
         rng=None, shocks=None, backend=None) -> SimulationResult
```
`SimulationResult` has: `paths: dict[str, ndarray (T, N)]` (assets, consumption, shocks index and values, …), `rng_seed_entropy`, `solution_fingerprint`; `.summary()`, `.to_dict()`, `.to_dataframe()` (pandas, optional extra; long format with `period`, `agent`), `.moments(names)`, `.mean(name)`.
Algorithm: draw all shock indices **up front** in NumPy (using `simulate_markov` (P3.4) on a per-agent spawned stream so that results do not depend on thread count), then run a Numba kernel that evaluates the policy by linear interpolation (`linear_interp_1d`) in the continuous state and indexes the discrete shock; `prange` over agents. Agents with the same seed and the same `n_agents` are identical regardless of `parallel` setting.
Delete `simulate_markov_asset_policy` and `simulation/policy.py`; port their tests to `simulate()` (R9: summary moments and diagnostics assertions keep, assertions about the old function's signature go). `simulate` joins the top-level exports and `API_POLICY.md`. `utils.rng.spawn` (P1.3) is first called here, so its NEAR-FUTURE tag is removed. For `perturbation` solutions (P9) `simulate` dispatches to a linear simulation instead of policy interpolation.

### P7.2 Moments helpers (`simulation/moments.py`)
Means, std, autocorrelation at lag *k*, cross-sectional percentiles, correlation between two series, regression slope; panel versions pooled over agents and time. Standard errors via block bootstrap (`block_bootstrap_se(series, block_len, n_boot, rng)`), Newey–West helper (`newey_west_variance`). Tests: AR(1) simulated series (ρ=0.9, σ_ε=1, T=2·10⁵) has sample autocorrelation within 4 × block-bootstrap s.e. of ρ.

### P7.3 Young (2010) histogram method
Represent the distribution as a mass `μ[i, j]` on the `(a_i, z_j)` grid.  
Policy lottery: for policy value `a'(i,j)` find bracket `k` with `a_k ≤ a' < a_{k+1}`, weight `w = (a_{k+1} − a')/(a_{k+1} − a_k)`; mass moves to `k` with weight `w` and to `k+1` with weight `1−w`, then mixes over `j'` with `P[j, j']`.  
API:
```python
build_transition_matrix(solution) -> scipy.sparse.csr_matrix     # (n_a n_z) x (n_a n_z), row-stochastic on indices
stationary_distribution_young(solution, tol=1e-12, max_iter=10_000) -> ndarray (n_a, n_z)   # loop = numerics.fixed_point
iterate_distribution(mu, solution, n=1)
```
The sparse matrix (`build_transition_matrix`) is the **NumPy/SciPy reference**; the Numba histogram-update kernel (`("distribution_step", "numba")`, which applies the same lottery without building the matrix) is the hot path, with a parity test (R3). Iteration uses `numerics.fixed_point`. The unit-eigenvector cross-check with `scipy.sparse.linalg.eigs` lives in the tests only.
**Key analytic properties (tests).**  
(a) Mass conserved: `Σ μ = 1` to 1e-14 after each step.  
(b) *Lottery preserves the policy mean*: `Σ_k w_k a_k = a'` exactly, so the mean of next period's assets given `(i, j)` equals the policy value to 1e-13.  
(c) If the policy is `a'(a) = a` (the permanent-income example, P5.9) the asset marginal distribution is invariant (stationary distribution not unique; test that the iteration leaves any `μ` unchanged in the asset dimension and mixes only across `z`, converging to the shock's stationary distribution on the `z` marginal within 1e-10).  
(d) Marginal over `z` converges to the stationary distribution of `P` (P3.4) within 1e-10 for any policy.  
(e) Agreement with Monte Carlo: mean assets from a 10⁴-agent, T=2000 simulation lies within 4 s.e. of `Σ μ a`.

### P7.4 Aggregates and inequality
`aggregate(mu, grid, policy_or_values)` → means; `marginal_distribution`, `cdf`, `quantile(mu, q)`; `lorenz_curve(values, weights)` and `gini(values, weights)`. Tests: uniform on `[0,1]` gives Gini 1/3 (1e-3 with 10⁴ equal-weight points... use exact weighted formula and assert 1e-4 vs 1/3 for n=4000); equal values → Gini 0; lognormal with σ gives `2Φ(σ/√2) − 1` within 5e-3 for a 2000-point quantile grid; Lorenz curve is convex, starts at 0 and ends at 1.

### P7.5 Impulse responses from nonlinear models (policy-based)
`simulate_irf(solution, shock="z", size=..., horizon=40, n_paths=..., rng)`: generalised IRF via paired simulations (shocked minus unshocked with common random numbers). Test with Brock–Mirman, whose exact law of motion in log deviations is `k̂_{t+1} = α k̂_t + ẑ_t`: for an IID `z` (ρ = 0) a 1 % TFP shock gives the capital response `1, α, α², …`; for ρ > 0 the response is `k̂_{t+1} = α k̂_t + ρ^t` (so `1, α+ρ, …`). Compare within 1e-2 for large paths. (This reuses the P9 analytic log-linear result.)

### P7.6 *(retired — chunking has no consumer; `simulate` stores only the variables named in `store=[...]`, default all, and the benchmark case needs 4·10⁸ bytes at most)*

### P7.7 Determinism contract tests
Same seed → identical arrays across 1 and 4 Numba threads (the test calls `numba.set_num_threads` directly; no config API is added for it), and across `numpy`/`numba` backends within 1e-12 (shocks drawn by NumPy are identical; policy evaluation differs only at rounding).

### P7.8 Docs
`docs/md/simulation.md` explaining the two distribution methods and when to use which (histogram for stationary equilibrium objects, simulation for time series and non-stationary experiments).

### P7.9 Port check
Every existing simulation test case has been ported to the new API or listed as removed in the Captain's Log (R9).

### P7.10 Property-based tests (Hypothesis)
Young histogram: for random monotone policies on random grids, mass is conserved, nonnegative, and the next-period mean equals the policy value to 1e-12. `simulate`: all simulated assets lie within the grid bounds; the same seed gives the same paths.

**Definition of Done.** `make ci`; `pytest -m numba tests/simulation`; coverage ≥ 80 %; analytic tests (a)–(e) pass; `grep -rn "simulate_markov_asset_policy" econkit tests` returns nothing.
**Session budget.** 3 (splittable at P7.3). **Pitfalls.** Wealth mass outside grid (policy above top node): clamp and log `n_clamped`; assert zero for default calibrations. Always row-normalise sparse transition matrices before iterating, and check `abs(Σμ − 1)` each 100 iterations.

---

## P8 — Equilibrium solvers

**Goal.** General-equilibrium engine for stationary equilibria in heterogeneous-agent models: Aiyagari (capital market) and Huggett (bond market).

**Prerequisites.** P7.

**Files.** `solvers/equilibrium.py`, `models/_equilibrium_helpers.py`, tests `tests/solvers/test_equilibrium.py`, `tests/models/test_aiyagari.py`, `test_huggett.py` (models land in P10, but the engine is tested here with a minimal inline model).

### P8.1 Generic stationary-equilibrium engine
```python
solve_stationary_equilibrium(model_factory, price_name, bracket, excess_demand,
    tol=1e-8, warm_start=True) -> EquilibriumSolution
```
Loop: given price `p`, build the household model (`model_factory(p)`), solve it (`solve(...)`, warm-started from the previous price), compute the stationary distribution (P7.3), compute aggregate supply/demand, return excess demand. Outer root-finder: `scipy.optimize.brentq` over `bracket`; on failure (no sign change) raise `ConvergenceError` with the evaluated endpoint excess demands in the message. Cache household solutions by price (small dict) so the final solution is not re-solved.
`EquilibriumSolution` (subclass of `Solution`): price, aggregates, `excess_demand`, `household_solution`, `distribution`, `history` (price, excess demand).

### P8.2 Aiyagari implementation requirements (used by P10.5)
Household: log or CRRA utility, AR(1) log-income, borrowing limit `a_min = 0`. Firm: Cobb–Douglas, `r = α (K/L)^{α−1} − δ`, `w = (1−α)(K/L)^α`, `L` = mean of the labour process (normalised to 1 with `z = exp(ρ-chain)/E[...]`). Excess demand: `A(r) − K_d(r)`, `K_d(r) = L ((r+δ)/α)^{1/(α−1)}`. Bracket `r ∈ (−δ + 1e-4, 1/β − 1 − 1e-6)`.

### P8.3 Huggett implementation requirements
Bonds in zero net supply; excess demand `A(r) − 0`; borrowing limit `a_min = −φ` (default −2 × mean income); bracket `r ∈ (−0.1, 1/β − 1 − 1e-6)`.

### P8.4 Analytic/structural tests
1. **Precautionary-saving inequality.** For Aiyagari with CRRA σ = 2, α=0.36, δ=0.08, β=0.96, ρ=0.9, σ_ε=0.2 (income process std/persistence from Aiyagari's table): the equilibrium `r* < 1/β − 1 = 4.1667 %` strictly; `|A(r*) − K_d(r*)| / K_d < 1e-6`; capital rate `K/Y` consistent with `α/(r*+δ)` to 1e-8.  
2. **Comparative statics.** `r*` is monotonically *decreasing* in income risk σ_ε over `σ_ε ∈ {0.0 (use 1e-6), 0.1, 0.2}`; `r*(σ_ε→0)` is within 5·10⁻⁴ of `1/β − 1` from below (grid-resolution dependent; tolerance justified by the finite asset-grid upper bound).  
3. **Huggett.** `r* < 1/β − 1` for σ_ε>0; bond market clears: `|A(r*)| < 1e-6 × mean income`; a tighter borrowing limit (`φ` smaller) gives lower `r*` (more saving demand → lower rate).  
4. **Plumbing.** Engine solves a trivial equilibrium with a known root (excess demand `= p − 0.37`) to tolerance; no-sign-change bracket raises `ConvergenceError`.

### P8.5 Performance sanity
Warm-start reduces total inner iterations (assert `history.inner_iterations_total` with warm start < without, on the Aiyagari test).

### P8.6 Docs
`docs/md/equilibrium.md`: algorithm figure (text diagram), what each aggregate means, how to extend to a new market (factory + excess-demand callable).

**Definition of Done.** `make ci`; `slow`-marked full Aiyagari calibration test passes with `pytest -m slow tests/solvers/test_equilibrium.py`; runtime of the unmarked tests < 30 s on a laptop.
**Session budget.** 2. **Pitfalls.** The asset grid upper bound must be well above the stationary distribution's support at every candidate `r` near `1/β − 1`, or `A(r)` is biased downward and the bracket fails; use an upper bound scaling with `1/(1/β − 1 − r)` capped, and test that mass at the top node is < 1e-6.

---

## P9 — Perturbation (first-order, Klein/QZ)

**Goal.** Solve linear rational-expectations models: steady-state finder, numerical linearisation, Blanchard–Kahn determinacy check, policy matrices, impulse responses, theoretical moments. Standalone from the VFI stack; needs only P3 + P4.

**Prerequisites.** P3, P4. (Can run any time after P4.)

**Files.** `solvers/perturbation/{__init__,steady_state,linearize,klein,irf,moments}.py`, tests `tests/solvers/perturbation/`.

### P9.1 Problem specification (adds the `dsge` template to `set_problem`, P4.3)
Dynamic models as: *variables* `y_t = [x_t; u_t]` (`x` predetermined/states, `u` jump/controls), *shocks* `ε_t ~ N(0, I)`, *equations* `E_t f(y_{t+1}, y_t, ε_{t+1}; θ) = 0`. API on the model: `model.set_problem("dsge", variables=..., states=[...], shocks=[...], equations=lambda yp, y, e, p: ndarray, shock_loadings=...)`. Declare the exogenous states explicitly (for example `z_{t+1} = ρ z_t + σ ε_{t+1}` is one equation in `f` with `ε_{t+1}` entering the predetermined block).

### P9.2 Steady state
`steady_state(problem, guess=None, fixed=...)`: solve `f(y*, y*, 0) = 0` with `scipy.optimize.root` (hybrid method), residual tolerance 1e-12; analytic steady states may be provided via `ss_func`. Raise `ConvergenceError` listing the final residual vector.

### P9.3 Linearisation
`linearize(problem, y_ss) -> (A, B, Σ)`: first-order expansion `A E_t[ŷ_{t+1}] = B ŷ_t` in the Klein form with `A = ∂f/∂y_{t+1}`, `B = −∂f/∂y_t`; the shock loading matrix `Σ` maps shocks into the predetermined block. Jacobians by central finite differences (P3.2) in NumPy; **JAX autodiff variant added in P12** (cross-checked to 1e-8). Option for log-linearisation (`log_variables=[...]`: differentiate with respect to `log y` so reported responses are percent deviations).

### P9.4 Klein (2000) solution (defines `DeterminacyError` and `PerturbationConfig`)
Use the generalised Schur (QZ) decomposition `scipy.linalg.ordqz`, ordering stable generalized eigenvalues (|λ| < 1) first.  
- Count stable roots `n_s` and compare with `n_x` (predetermined variables).  
  - `n_s == n_x`: unique stable solution → policy `u_t = F x_t`, law of motion `x_{t+1} = P x_t + Σ ε_{t+1}`.  
  - `n_s > n_x`: indeterminacy (multiple stable solutions) → raise `DeterminacyError("indeterminate: n_stable=… > n_predetermined=…")`.  
  - `n_s < n_x`: no stable solution (explosive) → `DeterminacyError`.
- Singular `A` (static equations) must be handled: QZ works with singular `A`; infinite generalized eigenvalues count as unstable.
- **Sign-convention pitfall.** `scipy.linalg.ordqz(A, B)` returns `(alpha, beta)` with eigenvalues `alpha/beta` of the pencil `A − λB`, whereas Klein's stability condition is on `λ = b_ii/a_ii` for `B x = λ A x`. **Write the scalar test first**: `a=1, b=0.5` (i.e. `x_{t+1} = 0.5 x_t`) must be classified stable and `b=2` unstable; only then write the sort function.

### P9.5 Impulse responses and moments
`irf(solution, shock, size=1.0, horizon=40) -> dict[str, ndarray]` (iterate `x_{t+1} = P x_t` from `Σ e_shock`, map to `u = F x`). `theoretical_moments(solution)`: unconditional covariance `Γ₀` of `x` by solving the discrete Lyapunov equation `Γ₀ = P Γ₀ P' + ΣΣ'` (`scipy.linalg.solve_discrete_lyapunov`), autocovariances `Γ_k = P^k Γ₀`, variance decomposition, and `simulate_linear(solution, T, rng)` for sample paths.

### P9.6 Analytic validations
1. **Scalar BK classification** (as above) with exact eigenvalue counts.  
2. **Brock–Mirman, log-linearised.** Full depreciation, log utility, `z` AR(1). Exact law of motion: `ln k' = ln(αβ) + ln z + α ln k`. Therefore in log deviations: `k̂_{t+1} = α k̂_t + ẑ_t`, `ĉ_t = α k̂_t + ẑ_t` (since `c = (1−αβ) z k^α`). Assert policy matrices equal these coefficients to 1e-6 (finite-difference Jacobian error) / 1e-9 with analytic Jacobian. IRF of `k̂` to a unit `ẑ` shock: `0, 1, α+ρ, …` follow `k̂_{t+1} = α k̂_t + ẑ_t` exactly.  
3. **Three-equation New Keynesian model** (P10.8) with `φ_x = 0`, AR(1) policy shock `v` with persistence `ρ`: the unique solution is `π_t = a v_t`, `x_t = b v_t` with  
   `b = −1 / ( σ (1 − ρ) + κ (φ_π − ρ)/(1 − βρ) )`, `a = κ b / (1 − βρ)`  
   (derivation: substitute the guesses into the Phillips curve `π = β E π' + κ x` and the IS curve `x = E x' − (1/σ)(i − E π')`, `i = φ_π π + v`). Assert coefficients within 1e-8 (analytic Jacobian) or 1e-6 (numerical).  
4. **Determinacy region.** With `φ_x = 0`: determinate iff `φ_π > 1` (κ>0). Test grid `φ_π ∈ {0.5, 0.9, 1.1, 1.5, 3}` → `DeterminacyError` for the first two, success for the rest. General condition (Bullard–Mitra): `κ(φ_π − 1) + (1 − β) φ_x > 0`; test one combination with `φ_x > 0` and `φ_π < 1` that satisfies it (determinate).  
5. **Moments.** AR(1) shock: `Var(z) = σ²/(1−ρ²)`, `Corr(z_t, z_{t−1}) = ρ`, matched to 1e-10 by the Lyapunov solver; simulate 2·10⁵ periods and compare to theory within 4 s.e.

### P9.7 Interop
`perturbation` solutions are `Solution` subclasses (`method="perturbation"`) with `policy` containing the matrices and `grids` empty; `simulate()` (P7) dispatches on `method`: linear simulation for perturbation, interpolation for global.

### P9.8 Docs
`docs/md/perturbation.md` with the Klein form, how to write equations, the log-linearisation switch, and the determinacy message decoder.

### P9.9 Property-based and golden-file tests
- Hypothesis: for random stable linear systems with the right root count, the returned `F`, `P` satisfy the original equations (residual < 1e-10); the Lyapunov covariance satisfies `Γ₀ = PΓ₀P' + ΣΣ'`.
- Golden file (P6.8 mechanism): policy matrices of the default RBC model.

**Definition of Done.** `make ci`; all five analytic validations pass; `solvers.perturbation` coverage ≥ 90 %.
**Session budget.** 3. **Pitfalls.** Mixing up timing (`E_t y_{t+1}` vs `y_{t+1}` with realised shocks), variable ordering (predetermined first), finite-difference step too small for log-linearised variables (use relative step), and stochastic singularity if shocks < exogenous states.

---

## P10 — Built-in model library

**Goal.** Nine ready-to-run, documented, tested models in `econkit.models` with a uniform interface. Each is both a user-facing tool and an end-to-end integration test of everything built so far. Each model is its own session-sized step. **Review-3 gate is after P11.**

**Prerequisites.** P6, P8, P9.

**Files.** `models/__init__.py` (registry), `models/<name>.py`, `docs/models/<name>.md`, tests `tests/models/test_<name>.py`.

### P10.0 Uniform interface (do this first)
```python
econkit.models.available() -> list[str]
econkit.models.load("aiyagari", **param_overrides) -> Model     # uncompiled, editable; the one way to get a built-in model
```
Each model module defines `PARAMS` (with bounds and descriptions), `build(**overrides)`, `CALIBRATION_SOURCE` (citation text), and `ANALYTIC` (dict of closed-form results, if any) used by tests and docs. Utility functions shared by the models (`crra`, `crra_marginal`, `crra_marginal_inverse`, in `models/utility.py`) are the only shared helpers. Which solver runs is decided by `solve(method="auto")`; models carry no `recommended_solver`. Every model page in `docs/models/` states: economics, equations, calibration table, solution method, a 10-line usage example, an analytic or published benchmark and what the test checks.

### P10.1 `consumption_savings` (income-fluctuation problem)
Params: `beta=0.96, sigma=2.0, r=0.03, rho=0.9, sigma_e=0.1, a_min=0.0, n_a=200, n_z=7`. Solved by EGM. Tests: permanent-income limit (P5.9) and consumption concavity (policy `c(a)` concave in assets for `z` fixed: second differences ≤ 1e-8 away from the constraint kink); borrowing constraint binds only for `a ≤ threshold`; the marginal propensity to consume out of wealth at the top of the grid approaches the perfect-foresight value `MPC = 1 − β^{1/σ} (1+r)^{(1−σ)/σ}` (CRRA, infinite horizon, no income risk). Assert `c'(a)` over the top 10 % of the grid is within 10 % of this value (loose: grid truncation and residual income risk matter); if the check is unstable, record the numbers and demote it to an informational print rather than loosening further.

### P10.2 `brock_mirman` (stochastic growth, log utility, full depreciation)
Params: `alpha=0.36, beta=0.96, rho=0.9, sigma_e=0.02`. Declares both a `saving` problem (`resources = z k^α`; solved by EGM and VFI) and a `dsge` problem (solved by perturbation). `ANALYTIC`: policy functions. Tests: the three methods agree with the closed form (tolerances as P5.8/P6.5/P9.6).

### P10.3 `rbc` (real business cycle, inelastic labour)
Params: `alpha=0.33, beta=0.99, delta=0.025, sigma=1.0 (log), rho=0.95, sigma_e=0.007`. Declares a `dsge` problem (perturbation, the default) and a `saving` problem (`resources = z k^α + (1−δ)k`, solved by EGM). Tests: steady-state capital–output ratio `k/y = α/(1/β − 1 + δ)` ≈ 9.40 at default calibration (to 1e-10); investment share `i/y = δ α/(1/β − 1 + δ)`; with `delta=1.0` and log utility, perturbation coefficients equal Brock–Mirman's (to 1e-6); perturbation vs EGM policy at the steady state agree to 1e-3 relative; IRFs: output response to TFP shock peaks within 3 periods and decays monotonically thereafter.

### P10.4 `mccall` (job search)
Params: `beta=0.96, c=0.5 (unemployment benefit), wage ~ discrete grid of 60 points from Uniform[0,1] or lognormal`. Declares a `finite_dp` problem (states: unemployed holding offer `i`, and absorbing employed-at-`w_i`; actions: reject/accept), solved by the tabular VFI of P5.2. `ANALYTIC`: reservation wage `w̄` solves `w̄ = c + β/(1−β) · E[(w' − w̄)^+]`; for uniform offers on [0,1] this is `w̄ = c + β/(2(1−β)) (1 − w̄)²`, i.e. a quadratic: let `k = β/(2(1−β))`, `k w̄² − (2k + 1) w̄ + (k + c) = 0`, take the root in [0,1]: `w̄ = [(2k+1) − sqrt((2k+1)² − 4k(k+c))] / (2k)`. Tests: VFI reservation wage matches the closed form within the grid spacing; comparative statics (higher `c`, higher `β` → higher `w̄`, via signs of finite differences); accepting rule is a threshold (policy monotone).

### P10.5 `aiyagari` (P8 engine)
Defaults from the Aiyagari (1994) calibration `α=0.36, δ=0.08, β=0.96, σ=3 (CRRA), ρ=0.6, σ_e=0.2`; (use `ρ=0.9, σ=2` for the P8 tests and document both). Tests: those in P8.4; plus `slow` full-grid test: `n_a=500, n_z=7` converges in < 60 s.

### P10.6 `huggett` (bond economy, P8 engine)
Tests as P8.4(3); plus Gini of wealth > Gini of income (assert) under the default calibration.

### P10.7 `olg2` (two-period overlapping generations, log utility, Cobb–Douglas, full depreciation, population growth `n`)
Params (one period = about 30 years): `alpha=0.3, beta=0.96**30 (≈0.294), n=1.01**30 − 1 (≈0.348)`; the tests also use the generic values `beta=0.3, n=0.5`. `ANALYTIC`: young agents save `s = β/(1+β) · w`, wage `w = (1−α) k^α`, capital accumulation `(1+n) k_{t+1} = s`, so  
`k* = [ β (1−α) / ((1+β)(1+n)) ]^{1/(1−α)}`, `r* = α k*^{α−1}`, and the steady state is globally stable for `α ∈ (0,1)` (convergence from any `k_0 > 0`, test with three starting points). Solution by exact iteration of the law of motion (own small solver, no DP needed) plus the generic root-finder on the Euler equation as an independent check (agree to 1e-10).

### P10.8 `nk3` (three-equation New Keynesian)
Equations: `π_t = β E_t π_{t+1} + κ x_t`; `x_t = E_t x_{t+1} − (1/σ)(i_t − E_t π_{t+1})`; `i_t = φ_π π_t + φ_x x_t + v_t`; `v_t = ρ_v v_{t−1} + ε_t`. Defaults `β=0.99, κ=0.15, σ=1, φ_π=1.5, φ_x=0.125, ρ_v=0.5`. Tests: P9.6(3)–(4) (use `φ_x=0` for the closed form); IRF signs for a contractionary policy shock (`x<0, π<0` on impact — with `φ_x=0` implied by `b<0`, `a<0`).

### P10.9 `lucas_tree` (asset pricing, endowment economy)
Params: `beta=0.96, sigma=1.0 (log), dividend log-AR(1): rho=0.9, sigma_e=0.05`. Method: fixed point of the price functional, iterated with `numerics.fixed_point` (P3.3), `p(y) u'(y) = β E[u'(y')(p(y') + y')]`. `ANALYTIC` (log utility): price–dividend ratio `p(y)/y = β/(1−β)` for any dividend process (derivation: guess `p = p̂ y`; `p̂ = β(p̂+1)`). Tests: numerical price function equals `β y/(1−β)` to 1e-6 relative; with CRRA σ≠1 the price–dividend ratio varies with y (strictly decreasing for σ>1 when dividends persistent); risk-free rate `1/(β E[(y'/y)^{-σ}]) − 1` computed two ways agrees to 1e-8.

### P10.10 Cross-model integration test
`tests/models/test_all_models_smoke.py`: every registered model loads, compiles, solves with default calibration under a 20-second budget guard (`pytest.mark.slow` for >5 s ones), `Solution.summary()` works, and (for those with `simulate`) a 1000-period simulation runs.

### P10.11 Model registry docs
Auto-generate `docs/models/index.md` from the registry with a script `scripts/gen_model_docs.py` (run in `make docs`).

### P10.12 Extension point
The registry's `register(name, builder)` is the same function the nine built-ins use, so it is public and documented in `CONTRIBUTING.md`; there is no separate plugin mechanism (Part E).

### P10.13 External cross-validation (optional)
Tests marked `external` that use `pytest.importorskip("quantecon")`: compare Tauchen and Rouwenhorst grids and matrices, and stationary distributions, against QuantEcon where the conventions can be matched; add an Aiyagari equilibrium comparison only if the calibrations can be made identical. QuantEcon is **not** a dependency; the tests simply skip when it is absent, and the CI `external` job installs it. Document convention differences instead of loosening tolerances.

**Definition of Done.** All nine models pass their analytic tests; `make ci`; `models` coverage ≥ 80 %.
**Session budget.** 4–5 (one model per half-session). **Pitfalls.** Calibration drift: do not tune parameters until a published number matches; the tests check *identities and closed forms*, not reproductions of papers' tables. Cite sources only when you are sure of the citation (otherwise write "textbook" and leave a TODO for the maintainer to confirm).

---

## P11 — Estimation

**Goal.** Take models to data: simulated method of moments, maximum likelihood, Kalman-filter likelihood for linear state space models, Bayesian MCMC and identification diagnostics. Exactly identified SMM is the calibration tool (no separate API). Replaces the demo-only `run_solve_simulate_moments_pipeline` (TD9), which is deleted. **Review-3 gate follows.**

**Prerequisites.** P7, P9, P10.

**Files.** `estimation/{moments,priors,smm,mle,kalman,bayes,identification,results}.py` (keep the `moments.py` `MomentTarget` API), tests `tests/estimation/`.

### P11.1 Parameter transforms
Extend P4.2's `ParameterSet` instead of adding a second mapping class: `Param.transform` (`"none" | "log" | "logit"`) and `ParameterSet.to_unconstrained(x)`/`from_unconstrained(u)` apply the bounds through logistic/exponential transforms. Used by SMM, MLE and MCMC (one place, one convention).

### P11.2 Moment targets (extend existing)
Keep `MomentTarget` and `quadratic_moment_loss`. Add `MomentSet(targets, weights)` with `.model_moments(simulation_result)`; built-in moment functions (mean, std, autocorr(k), corr(x,y), quantile, regression slope, Gini). Handle missing data moments (drop with note).

### P11.3 SMM
```python
smm(model, free_params, data_moments, moment_fn, *, solve_options, sim_options,
    weights="identity"|"optimal"|ndarray, n_sim=..., method="Nelder-Mead", rng=...) -> SMMResult
```
Algorithm: for candidate θ → recompile model → solve → simulate with **common random numbers** (fixed shock draws across θ evaluations) → model moments `m(θ)` → loss `(m − m̂)' W (m − m̂)`. Two-step: step 1 `W = I` (or diagonal of inverse variances), step 2 `W = Ω̂^{-1}` with `Ω̂` from the first-step estimate (bootstrap or Newey–West on the data moments).  
Inference: Jacobian `G = ∂m/∂θ` by central differences with common random numbers; asymptotic variance  
`V = (1 + 1/S) (G' W G)^{-1} G' W Ω W G (G' W G)^{-1} / N`  
(`S` = simulation-to-data length ratio, `N` = sample size); with `W = Ω^{-1}` this reduces to `(1 + 1/S)(G'Ω^{-1}G)^{-1}/N`. Report std errors, t-stats, `J = N/(1 + 1/S) · g' Ω^{-1} g ~ χ²(q − p)` p-value. `SMMResult.summary()` prints a table and `.to_dataframe()` returns it (LaTeX and Markdown export are pandas' own `to_latex` / `to_markdown`; no table wrapper is written). The optimiser is `scipy.optimize.minimize` with the chosen `method`; there is no other optimiser path. MLE and Metropolis share the same transform and optimiser plumbing (P11.1).
Warnings: non-convergent inner solves are replaced by a large penalty and counted in `result.diagnostics["failed_evals"]`.

### P11.4 SMM recovery tests (with closed-form model)
Use an AR(1) process `y_t = ρ y_{t−1} + σ ε_t` as the "model" (pure simulation, no solver): moments `var(y)`, `autocorr(y,1)` ↔ `(ρ, σ)` via closed forms `σ² /(1−ρ²)` and `ρ`. Simulate "data" with `T=5000`, `ρ=0.9, σ=0.5`; SMM with `S = 20` recovers parameters: `|ρ̂ − 0.9| < 4 s.e.` and `|σ̂ − 0.5| < 4 s.e.` over 5 seeds (all 5 must pass), for the J-test, add a third moment `autocorr(y,2) = ρ²` so the system is over-identified (3 moments, 2 parameters, 1 degree of freedom): the J-statistic p-value must exceed 0.01 in at least 4 of 5 seeds. Mark `slow` if > 20 s. Also an end-to-end recovery test on `consumption_savings`: generate panel moments at known `β`, estimate `β` only (others fixed) from {mean assets, std consumption growth} and recover within 2 % (grid-limited), using EGM + Numba.

### P11.5 MLE
`mle(loglik, theta0, bounds, method="L-BFGS-B")` with a numerical Hessian (central differences); standard errors from `(−H)^{-1}`. Test: Gaussian AR(1) conditional MLE equals the OLS estimate to 1e-6 and standard error equals `sqrt((1−ρ²)/T)` within 5 %.

### P11.6 Kalman filter and linear-Gaussian likelihood
`kalman_filter(y, Z, H, T, Q, a0, P0) -> (loglik, filtered, predicted, ...)` and `loglik_state_space(solution, observables, ...)` that builds `(T, Q, Z, H)` from a perturbation `Solution` (P9) with measurement error. Tests: **local-level model** `y_t = μ_t + ε_t, μ_{t+1} = μ_t + η_t`: steady-state predicted variance `P` (in units of `σ_ε²`) satisfies `P = (q + sqrt(q² + 4q))/2 + ...` where `q = σ_η²/σ_ε²` — the steady-state *predicted* state variance in units of σ_ε² solves `P̄ = P̄ − P̄²/(P̄ + 1) + q`, i.e. `P̄² − q P̄ − q = 0`, so `P̄ = (q + sqrt(q² + 4q))/2`, and the gain is `K = P̄/(P̄ + 1)`; assert the filter's gain converges to this within 1e-8 after 200 periods. Log-likelihood equals a brute-force multivariate normal density (`scipy.stats.multivariate_normal` with the implied covariance) to 1e-8 for T=15. Missing observations (NaN) are handled by skipping updates (test: likelihood equals that of the series with those rows removed from `Z`).

### P11.7 Bayesian estimation (random-walk Metropolis–Hastings)
`metropolis(logpost, theta0, n_draws, n_chains=4, rng)` with adaptive covariance (Haario et al.) after a burn-in; diagnostics: R-hat (split), effective sample size, acceptance rate; posterior summary table. `bayes_estimate(model, priors, loglik, ...)` joins the priors with the P11.5/6 likelihood. Priors (`estimation/priors.py`): `Normal`, `Uniform`, `Beta`, `Gamma`, `InverseGamma`, each with `logpdf`, `sample(rng)` and `support`; `Param.prior` is added here. Test: each `logpdf` integrates to 1 numerically (trapezoid, 1e-4); `sample` means within 4 s.e. Tests: target = bivariate Gaussian with known mean and covariance: posterior mean within 4 MC s.e. (using ESS), covariance entries within 10 %, R-hat < 1.05 on 4 chains × 5000 draws; target = Beta(2,5) on [0,1] via logit transform: mean within 0.01 of 2/7.

### P11.8 *(retired — calibration is exactly identified SMM: with as many moments as parameters the minimum loss is 0 and the estimate hits the targets; the SMM tutorial shows it)*

### P11.9 Identification diagnostics
`identification_report(smm_result)`: Jacobian rank, condition number of `G'WG`, singular values; list weakly identified parameters (relative singular value < 1e-3) in the report and raise `IdentificationError` (defined here) when the Jacobian is rank-deficient. Profile `loss(θ_k)` helper (grid over one parameter, re-optimising others optional). Test: model where two parameters enter only via their product → rank deficiency detected.

### P11.10 Delete the demo pipeline
Delete `run_solve_simulate_moments_pipeline` and `estimation/report` helpers that only served it; port its test to an end-to-end test built from `solve → simulate → MomentSet` (R9). The end-to-end workflow with provenance is `Experiment` (P14.2).

**Definition of Done.** `make ci`; recovery tests pass on 5 seeds; coverage ≥ 80 %; `docs/md/estimation.md` with a worked SMM example and the inference formulas.
**Session budget.** 3–4 (splittable at P11.3, P11.6). **Pitfalls.** Common random numbers must be reused across θ or the objective is noisy and optimisers stall; bounds via transforms rather than clipping; the finite-difference step for `G` needs to exceed solver tolerance effects (use ≥ 1e-4 relative and verify with two step sizes agreeing to 1e-2).

> **Review-3 (human gate).** Summarise recovery results and the model library; confirm v1 scope (§3.2/3.3) still fits.

---

## P12 — JAX backend and automatic differentiation

**Goal.** Replace the pilot wrappers with real JAX implementations of hot operations, enable autodiff (exact Jacobians for perturbation, gradients for SMM/MLE), and `vmap`/`jit` for parameter sweeps. GPU use is allowed but not required.

**Prerequisites.** P5–P7, P9 (perturbation), P11 (for gradients).

**Files.** `numerics/backends/*_jax.py`, `solvers/dp/kernels_jax.py`, `simulation/kernels_jax.py`, `solvers/perturbation/jacobian_jax.py`, `econkit/_jax_setup.py`, tests `tests/**/test_*_jax.py` (marker `jax`).

### P12.1 Setup and policy
`_jax_setup.py` enables 64-bit (`jax.config.update("jax_enable_x64", True)`) on first import of the backend (document the global side effect; there is no opt-out, because every tolerance in this plan assumes `float64`). Register `"jax"` in `econkit.backends.available_backends()` only when importable. CPU is the CI target; no GPU-only tests.

### P12.2 Operations to implement (each with NumPy parity test at 1e-10 under x64)
`linear_interp_1d`, `bellman_saving` (vectorised `argmax`, `vmap` over shocks), `distribution_step` (histogram update with `segment_sum` / scatter-add), `markov_simulate` (`lax.scan`), `discretize` (not required in JAX; leave NumPy). Use `jax.jit` with static grid sizes; document recompilation behaviour.
Do not claim JAX is faster: it will often be slower on CPU for these kernels than Numba. The reason for the backend is autodiff, vectorisation across parameters, and GPU.

### P12.3 Autodiff in perturbation
`linearize(..., jacobian="jax")` uses `jax.jacfwd`. Requirement: the equations callable must be JAX-traceable; detect failures and raise a clear error naming the equation. Cross-check: JAX Jacobians vs finite-difference Jacobians agree to 1e-7 on the RBC model; and the Brock–Mirman coefficients equal the analytic values to 1e-12 (exact derivative, no truncation error). The test is an **improvement check**: with JAX, `P9.6(2)` tolerance tightens from 1e-6 to 1e-10.

### P12.4 Gradients for estimation
`mle(..., gradient="jax")` for Kalman log-likelihood (the linear-Gaussian likelihood is a smooth function of `θ` through the perturbation solution if both are traceable): gradient vs finite differences to 1e-6. SMM stays derivative-free (simulation noise and `argmax` make its moments non-smooth).

### P12.5 Vectorisation across parameters (consumer: `ParameterSweep`, P14.1)
`solve_many(model, param_grid, method="egm", backend="jax")`: `vmap` the EGM iteration over a batch of parameter vectors (fixed iteration count with convergence masking via `lax.while_loop`). Test: 8 `β` values solved in one call match 8 sequential NumPy solves to 1e-8.

### P12.6 *(retired — JAX Monte Carlo has no consumer; `monte_carlo` in P14.5 uses spawned NumPy streams)*

### P12.7 Dispatch semantics
Respect `strict_backend`: requesting `"jax"` without JAX installed raises `BackendUnavailableError`; non-strict falls back with a single warning. `backends.describe()` shows JAX device (cpu/gpu) and x64 status.

### P12.8 CI
Add a CI job installing `.[jax]` on Linux/py3.11 that runs `pytest -m jax`. macOS/Windows JAX jobs are optional (`continue-on-error`).

**Definition of Done.** `make ci` (without JAX → JAX tests skipped); `pytest -m jax -q` green with JAX installed; docs page `docs/md/backends.md` with the honest backend guide (when to use NumPy, Numba, JAX).
**Session budget.** 2–3. **Pitfalls.** Python control flow inside traced functions; `jnp.searchsorted` semantics differ at ties (cross-check edge cases); `-inf` and `argmax` gradients; accidental float32 when x64 is off; retracing on shape changes (cache jitted functions by static shapes).


---

# Part D — Phases P13 to P17: Data, experiments, performance, documentation, release

---

## P13 — Data and visualization

**Goal.** The two things economists need around a model and that pandas and matplotlib do not already provide: standard macro filters and a FRED download/parse function, plus a complete set of plots. No wrapper classes around pandas (R21g); the core library still imports without pandas or matplotlib.

**Prerequisites.** P7, P11.

**Files.** `data/{__init__,filters,fred}.py`, `viz/{__init__,style,policy,simulation,distribution,estimation,diagnostics}.py`, tests `tests/data/`, `tests/viz/`.

### P13.1 *(retired — an `EconTimeSeries` wrapper around `pandas.Series` adds no behaviour; revisit with first-class Polars, Part E)*
### P13.2 *(retired — `EconPanel`; `SimulationResult.to_dataframe()` (P7.1) already gives the long-format panel)*

### P13.3 Filters (cold path, NumPy/SciPy; array-like in, arrays out, index preserved when given a `pandas.Series`)
- `hp_filter(y, lamb=1600.0) -> (trend, cycle)`: solve `(I + λ K'K) τ = y` with `K` the second-difference operator using a `scipy.sparse` banded solve. 1600 is the conventional quarterly value; document 100 (annual) and 14400 (monthly) as conventions, not defaults. Tests: a linear trend gives cycle `= 0` to 1e-8; `trend + cycle == y` to 1e-12; for white noise the cycle variance is below the series variance.
- `baxter_king(y, low=6, high=32, K=12)`: band-pass with weights from the ideal filter adjusted to sum to zero. Tests: weights sum to 0 (1e-12) and are symmetric; output length `T − 2K`; a pure 16-period sinusoid passes with gain ≈ 1 (within 5 %); a constant and a linear trend are removed (|output| < 1e-8).
- `hamilton_filter(y, h=8, p=4) -> (trend, cycle)`: regress `y_{t+h}` on a constant and `y_t … y_{t−p+1}`; the residual is the cycle. Tests: for a deterministic polynomial of degree ≤ p−1 the residual is 0 to 1e-8; for a random walk the residual equals the h-step forecast error.
Consumer: tutorial 5 (P16.3) compares the three filters on a simulated RBC output series. Edge effects of HP/BK at the sample ends are real; document them.

### P13.4 *(retired — CSV/Parquet reader wrappers; use pandas)*

### P13.5 FRED download and parse
`fetch_fred(series_id, *, start=None, end=None, api_key=None, cache_dir=None) -> pandas.Series` using only the standard library (`urllib`), and `parse_fred_csv(text) -> pandas.Series`. `fetch_fred` caches to `cache_dir`; with no network it raises a clear `EconKitError`. **Tests never touch the network**: they parse a bundled sample string through `parse_fred_csv`. The real endpoint and column format must be confirmed manually against FRED's documentation at implementation time (I have not verified them); put a `network`-marked, default-skipped smoke test next to it. Consumer: tutorial 5 parses a small bundled FRED-format sample file (synthetic values, no licensed data) with `parse_fred_csv` and HP-filters it.

### P13.6 *(retired — synthetic dataset generators; tests build their data inline and tutorials simulate from the model)*

### P13.7 Visualization infrastructure
`viz/style.py`: one `econkit_style()` context manager (readable defaults, colour-blind-safe cycle, constrained layout). All plot functions take the object to plot plus optional `ax=None`, return the `Axes` (or `Figure` for multi-panel), never call `plt.show()`, and never change global rcParams outside the context manager. Calling a plot function without matplotlib raises an informative `ImportError`.

### P13.8 The plot set (each with a smoke test using the `Agg` backend, and each used by at least one tutorial or model page — `check_usage.py` enforces it)
`plot_policy(solution, var, state_slice)`, `plot_value(solution)`, `plot_euler_errors(solution)`, `plot_convergence(solution)` (residual history, log scale), `plot_simulation(result, vars)`, `plot_irf(irf_dict)` (grid of panels), `plot_distribution(mu, grid)` (marginal + CDF), `plot_lorenz(values, weights)` (45° line, Gini in the legend), `plot_moment_fit(smm_result)` (data vs model with s.e.), `plot_mcmc_trace(chain)`, `plot_posterior(chain, priors)`, `plot_sweep(sweep_result, x, y)`, `plot_equilibrium(eq)` (asset supply vs demand against `r`).
Tests assert **structure, not pixels**: number of lines/collections, axis labels non-empty, x-data equals the grid, y-data equals the policy values, `ax.get_yscale()` for log plots, Lorenz curve endpoints `(0,0)`/`(1,1)`. No image-diff tests. A plot with no tutorial or model-page use is removed (R21b).

### P13.9 Import hygiene test
`tests/api/test_lazy_imports.py`: in a subprocess, `import econkit` must not import `pandas`, `matplotlib`, `numba` or `jax` (inspect `sys.modules`). Fix any module that violates this with lazy imports.

**Definition of Done.** `make ci`; `pytest tests/data tests/viz`; coverage ≥ 80 %; import-hygiene test passes.
**Session budget.** 2–3 (splittable at P13.3 / P13.8). **Pitfalls.** pandas 2.x frequency aliases changed ("Q" vs "QE"); if `parse_fred_csv` builds a `PeriodIndex`, test both spellings.

---

## P14 — Experiments and reproducibility

**Goal.** Orchestrate parameter sweeps, counterfactual policy experiments with welfare measures, Monte Carlo and bootstrap, with provenance saved so results can be reproduced.

**Prerequisites.** P10, P11.

**Files.** `experiment/{__init__,sweep,experiment,policy,welfare,montecarlo}.py`; adds `Solution.to_dict/save/load` to `core/solution.py`, tests `tests/experiment/`.

### P14.1 `ParameterSweep`
```python
ParameterSweep(model, grid: dict[str, array-like] | list[dict], function: Callable[[Solution], dict],
               solve_options=..., n_jobs=1, backend=None).run() -> SweepResult
```
Cartesian product (or explicit list) of parameter values; for each point: `with_params` → `solve` → user function (for example aggregate moments). `n_jobs>1` uses `concurrent.futures.ProcessPoolExecutor` (no new dependency; the model builder must be importable - document it), results returned in grid order independent of completion order. Failed points are recorded (`status="failed"`, message) not fatal, unless `fail_fast=True`. When `backend="jax"` and `method="egm"`, the whole grid is solved in one batch by `solve_many` (P12.5, its consumer); otherwise points are solved in a loop or process pool. `SweepResult.to_dataframe()` and `.best(metric)`. Test: results identical for `n_jobs=1` and `n_jobs=2`.

### P14.2 `Solution` persistence, `Experiment` and the provenance store
First add `Solution.to_dict()` (JSON-safe: arrays as lists, NaN/inf encoded explicitly), `Solution.save(path)` (`.npz` for arrays + sidecar JSON) and `Solution.load(path)`; they are first called here.
```python
exp = Experiment(
    name,
    build=(
        "aiyagari",
        {"beta": 0.95},
    ),  # a built-in model name + overrides, or an importable "module:function"
    solve_options={...},
    simulate_options=None,
    moments=None,
    outdir="runs/",
)
result = exp.run(rng=123)  # always solve; simulate and compute moments only if configured
```
Writes `runs/<name>/<timestamp>-<fingerprint8>/` containing `config.json`, `provenance.json` (P1.9: versions, git commit, platform, fingerprint of model), `solution.npz`, `result.json`, `log.txt`. `Experiment.reproduce(path)` reloads the config, re-runs, and compares `result.json` numerically (relative tolerance 1e-10 for the same backend and thread count); returns a `ReproductionReport`. Models are never serialised as code: `build` names how to rebuild them. Test: run → reproduce passes; altering a parameter in `config.json` makes the report fail with the differing keys listed.

### P14.3 Counterfactual policy experiments
`PolicyExperiment(baseline_model, reform: dict | Callable[[Model], Model], welfare="cev")`. `compare(baseline_solution, reform_solution)` returns aggregates, distributions, and welfare. A reform is a dict of parameter overrides (for example a tax rate or the borrowing limit) or a function `Model -> Model` (using `Model.copy`, P4.10).

### P14.4 Welfare: consumption-equivalent variation (`welfare.py`)
Definition: `λ` such that expected lifetime utility at baseline consumption scaled by `(1+λ)` equals expected lifetime utility in the reform.  
- CRRA with `σ ≠ 1`: `V` scales by `(1+λ)^{1−σ}` so `λ = (V_reform / V_base)^{1/(1−σ)} − 1` (with `V_base` and `V_reform` of the same sign; for `σ>1` both negative: the ratio is positive).  
- Log utility: `V` shifts by `ln(1+λ)/(1−β)`, so `λ = exp( (1−β)(V_reform − V_base) ) − 1`.  
- Aggregate version averages over the stationary distribution (P7.3); transition costs are out of scope for v1 (stationary comparisons only; the docs say so).  
- EGM solutions carry no value function. `evaluate_policy_value(solution)` computes it from the policy: `V = (I − βT)^{-1} u`, with `T` the policy-induced sparse transition matrix from P7.3 (`build_transition_matrix`) and `u` the flow utility. Test: on a VFI solution it reproduces the VFI value to `tol`.  
**Tests (exact).** Scaling all consumption in the baseline by 1.05 reproduces `λ = 0.05` to 1e-10 in both the log and CRRA cases (construct `V_reform` analytically from a deterministic consumption stream with `β` and an infinite-horizon constant consumption `V = u(c)/(1−β)`); `λ = 0` when models are identical; sign: a reform that raises every state's value gives `λ > 0`.

### P14.5 Monte Carlo and bootstrap
`monte_carlo(fn, n_reps, rng)` using `spawn` streams so replications are independent and reproducible; `bootstrap_ci(data, stat, n_boot, alpha=0.05, block_len=None)` (percentile intervals; moving-block bootstrap when `block_len` is given). Tests: nominal 95 % percentile CI for the mean of N(0,1) samples with n=50 over 400 replications has empirical coverage in [0.92, 0.98] (deterministic seed; if the observed value is outside, investigate the bug first, do not widen the band); block bootstrap s.e. of an AR(1) mean is larger than the iid bootstrap s.e. when ρ=0.7.

### P14.6 Elasticities
`elasticities(model, params, metric, rel_change=0.01)` built on `ParameterSweep` (central differences; elasticity = d ln metric / d ln param). No separate one-at-a-time function: a sweep over `±rel_change` is exactly that. Test: for Brock–Mirman the elasticity of steady-state capital `k* = (αβ)^{1/(1−α)}` (z=1) with respect to `β` equals `1/(1−α)` (tolerance 1e-4).

### P14.7 *(retired — a results-table wrapper would only call pandas' `to_latex`/`to_markdown`; see P11.3)*

### P14.8 Docs
`docs/md/experiments.md`: reproducibility contract (what is and isn't guaranteed: same machine, same versions, same backend, same thread count), welfare definitions, bootstrap guidance.

**Definition of Done.** `make ci`; experiment tests; coverage ≥ 80 %.
**Session budget.** 2. **Pitfalls.** Never serialise lambdas; the `build` entry names how to rebuild a model and the fingerprint verifies it. JSON cannot hold NaN/inf by default; encode explicitly.

---

## P15 — Performance engineering and benchmarks

**Goal.** Replace aspiration with measurement: a repeatable benchmark harness, recorded results committed to the repo, profiling-driven fixes, and honest documentation of what is fast, what is not, and why.

**Prerequisites.** P12.

**Files.** `benchmarks/{__init__,harness,cases/*.py,run.py}`, `benchmarks/results/*.json`, `scripts/bench_compare.py`, `docs/md/benchmarks.md`, `.github/workflows/nightly.yml`.

### P15.1 Harness
`benchmarks.harness.run_case(case, backend, n_repeat=5, warmup=1)` returns `{name, backend, params, compile_seconds, min, median, max, peak_memory_mb(optional via tracemalloc for NumPy-only paths), result_hash}`. Rules: JIT compile time is measured separately (first call) and reported, never mixed into steady-state timings; repeated runs use identical inputs; `result_hash` guarantees that different backends solved the same problem to the same answer (tolerance check against the NumPy reference, fail loudly if not). Machine info captured (CPU model, cores, RAM, OS, Python/NumPy/Numba/JAX versions). This step also **deletes** `scripts/benchmark_backends.py` and the `benchmark-backends` Makefile target; `make bench` replaces them (R21a).

### P15.2 Benchmark cases (name → description)
`interp1d` (10⁶ queries), `bellman_cs` (n_a=1000, n_z=7), `vfi_cs` (full solve), `egm_cs` (full solve), `egm_rbc` (n_k=300, n_z=7), `young_stationary` (n_a=1000, n_z=7), `simulate_panel` (10⁵ agents × 500 periods), `aiyagari_ge` (full equilibrium), `perturbation_rbc` (linearize + Klein), `kalman_loglik` (T=500), `smm_ar1` (S=20). For each: backends `numpy`, `numba` (and `jax` where implemented).

### P15.3 Record and publish
`python -m benchmarks.run --all --out benchmarks/results/<YYYYMMDD>_<host>.json`; `scripts/gen_bench_doc.py` turns the newest JSON into `docs/md/benchmarks.md` (table: case × backend, min time, speedup vs NumPy, compile time) with the machine info header. **Only numbers in that file may be quoted in the README or docs** (R19).

### P15.4 Targets and honest reporting
Targets (to be compared against the *recorded* numbers; they are goals, not assertions):  
- `bellman_cs`: Numba ≥ 20× NumPy vectorised;  
- `vfi_cs` (monotone + Howard): ≥ 50× the legacy pure-Python loop (P5.10);  
- `egm_cs`: ≥ 20× Numba VFI;  
- `simulate_panel`: Numba ≥ 10× NumPy loop;  
- `young_stationary`: Numba kernel ≥ 3× sparse-matrix iteration.  
For each missed target add a short "Why" paragraph with profiler evidence. Do not edit the target to match the result; add a follow-up line to Parking lot if a real optimisation opportunity remains.

### P15.5 Profiling-driven fixes (time-boxed)
Use `cProfile` + `snakeviz`-style top-10 listings in the log for the three slowest cases. Typical wins to check, in this order: removing allocation inside inner loops; contiguous `float64` arrays; avoiding Python-object paths inside `njit`; `prange` placement and `NUMBA_NUM_THREADS`; replacing `np.searchsorted` in inner loops by a monotone pointer walk; `cache=True` effectiveness (confirm second process start has no compile cost, measured through the harness `compile_seconds`). Parallelism must not change results (determinism tests from P7.7 must still pass).

### P15.6 Regression detection
`scripts/bench_compare.py old.json new.json --threshold 1.30` prints cases that slowed by more than 30 %; the nightly workflow (scheduled, non-gating) runs a reduced suite and uploads the JSON as an artifact. Timing is **never** a required CI check.

### P15.7 Import-time and footprint budget
Measure `python -X importtime -c "import econkit"`; record the number. Target: `import econkit` < 0.5 s on the reference machine with no optional dependencies imported (P13.9 guards the imports; this records the time). Record installed wheel size.

**Definition of Done.** `python -m benchmarks.run --all` works; JSON committed; `docs/md/benchmarks.md` generated; every number in README comes from it.
**Session budget.** 2. **Pitfalls.** Laptop thermal throttling and noisy neighbours — take the `min` of repeats, note it; never compare numbers from different machines.

---

## P16 — Documentation and tutorials

**Goal.** A documentation site a new user can learn from without reading source code: install, concepts, nine executed tutorials, model pages, theory pages, API reference, contributor guide. **Review-4 gate follows.**

**Prerequisites.** P13, P14 (P15 for the benchmark page).

**Files.** `docs/conf.py`, `docs/index.md`, `docs/user_guide/*.md`, `docs/tutorials/*.md` (jupytext, executed), `docs/api/*.rst|md`, `docs/theory/*.md`, `docs/models/*.md`, `.readthedocs.yaml`, `.github/workflows/docs.yml`, `scripts/check_docstrings.py`, `README.md`.

### P16.1 Sphinx scaffold
Sphinx + MyST-NB + `furo` theme; `autosummary` for the API reference, `sphinx-autodoc-typehints`, `sphinx.ext.napoleon`, `sphinx.ext.mathjax`, `sphinx.ext.intersphinx` (NumPy, SciPy, Python). `make docs` builds to `docs/_build/html` with warnings as errors (`-W --keep-going` after a first clean pass). Notebook execution `nb_execution_mode = "force"`, `nb_execution_timeout = 120`.

### P16.2 User guide
Pages: installation (extras table), the mental model (model → solve → simulate → estimate), backends (what each is for; link to `docs/md/backends.md`), conventions (rng, errors, Solution), choosing a solver, performance tips, FAQ. Write for an economist who knows Python basics but not Numba.

### P16.3 Tutorials (nine, each executes in ≤ 60 s; each ends with an assertion cell that checks a key number so broken tutorials fail the docs build)
1. **Quickstart:** a consumption–savings model in 15 lines; solve, plot policy, simulate.
2. **Consumption–savings with EGM:** borrowing constraints, Euler errors, comparison with VFI.
3. **Stochastic growth three ways:** VFI, EGM, perturbation vs the closed form (Brock–Mirman).
4. **Aiyagari:** equilibrium interest rate, Lorenz curve, precautionary saving.
5. **RBC and New Keynesian by perturbation:** IRFs, determinacy region, business-cycle moments from HP-filtered simulated output (HP, Baxter–King and Hamilton compared), and a bundled FRED-format sample series parsed with `parse_fred_csv` (consumers of P13.3 and P13.5).
6. **McCall job search:** reservation wage, comparative statics, analytic check.
7. **Estimating a model with SMM:** moments, standard errors, J-test, identification diagnostics; shows that exactly identified SMM is calibration.
8. **Bayesian estimation of a state-space model** (Kalman + Metropolis): posterior plots, R-hat.
9. **Policy experiment and welfare:** a borrowing-limit reform in Huggett, CEV decomposition.
Tutorials are stored as jupytext Markdown (`.md`) so diffs are readable; `jupytext --to ipynb` produces notebooks for download. Never commit executed outputs.

### P16.4 Model pages and theory pages
Model pages are generated in P10.11; verify each renders. Port the contents of `docs/latex/econkit_math.tex` once into MyST pages under `docs/theory/` (interpolation, VFI/EGM, Young's method, Klein, SMM inference, Kalman, MCMC, CEV), then **delete the `.tex` file** and update R7's reference: from here on the theory pages are the only home of the mathematics. Hand-port; no conversion tooling is written.

### P16.5 API reference and docstring audit
`scripts/check_docstrings.py`: for every public symbol (`__all__` in each documented module) assert a docstring exists, has `Parameters`/`Returns` where applicable, and contains an `Economics note:` line for numerical/economic functions (a small allow-list for pure plumbing such as `config` setters). Add to `make ci`, next to `make deadcode` (whose `check_usage.py` also requires every public symbol to be used by the library, a test, or an executed doc page). Run `pytest --doctest-modules` on modules whose docstrings contain examples; make examples deterministic and fast (<1 s each).

### P16.6 Honest README
Rewrite `README.md`: what EconKit is; install; a 15-line example; feature list matching reality (use the P0 table style); backend table; benchmark table copied from `docs/md/benchmarks.md` with the machine noted; status badge; license; how to cite; contributing; "what is not in v1" (§3.3). No claims about unreleased features. No unmeasured speedups.

### P16.7 Contributor docs
`CONTRIBUTING.md` expanded: adding a model (P10.0 interface), adding a backend kernel (R3 doctrine + parity test), adding a solver (register in `SOLVERS`), style, tests, docs, release process. `docs/md/architecture_overview.md` — a 2-page condensation of the architecture document that lives in the repo (D4).

### P16.8 Build and publish docs
`.readthedocs.yaml` (Python 3.11, `pip install .[docs,all]`) **or** a GitHub Pages workflow `docs.yml` (build on push to `master`, deploy to `gh-pages`). Choose one (default: Read the Docs config file + GitHub workflow that only builds for PRs). Link check with `sphinx -b linkcheck` in a non-gating job.

### P16.9 Documentation review pass
Read each page start to finish, executing the code in a clean venv. Fix errors. Record the list of pages reviewed in the log. Verify no page uses names from outdated APIs (`grep` for removed/deprecated names).

### P16.10 JOSS paper preparation
`paper/paper.md` and `paper/paper.bib`: summary, statement of need, comparison with QuantEcon / dolo / HARK / Dynare (reuse Architecture §2), a short reproducible example, acknowledgements. Check the current Journal of Open Source Software submission requirements at the time (for example public development history and license); I have not verified them. Submission is a maintainer action after 1.0.

### P16.11 Known limits page
`docs/md/known_limits.md`: for each solver, the failure modes (non-concavity, grid too small, indeterminacy, weak identification), how EconKit detects each (the error or diagnostic that fires), and what the user should do.

> **Review-4 (human gate).** Maintainer reads the docs, confirms API names, and makes the final license decision (D1).

**Definition of Done.** `make docs` passes with `-W`; all tutorials execute; `scripts/check_docstrings.py` clean; README numbers match `docs/md/benchmarks.md`.
**Session budget.** 4 (splittable by page group). **Pitfalls.** Docs rot: make every tutorial self-checking. Don't paste plot images into the repo; build them.

---

## P17 — Release engineering and v1.0.0

**Goal.** Ship. Freeze the public API, finish licensing and metadata, verify installs on clean environments, publish to TestPyPI, then PyPI, tag, and write release notes. Several steps are **human-gated**; the agent prepares everything and stops at the gate.

**Prerequisites.** P1–P16 complete; Review-4 approved; D1, D5, D8 resolved.

**Files.** `econkit/_version.py`, `pyproject.toml`, `LICENSE`, `CITATION.cff`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `scripts/dump_api.py`, `tests/api/api_snapshot.json`, `scripts/smoke_installed.py`, `.github/workflows/release.yml`.

### P17.1 API freeze
`scripts/dump_api.py` writes a JSON snapshot of every public symbol (module path, name, signature via `inspect.signature`, kind) for all documented modules. Commit as `tests/api/api_snapshot.json`; `tests/api/test_api_snapshot.py` compares the live API with the snapshot and fails on any diff, with a message explaining that API changes after 1.0 need a CHANGELOG entry and a minor/major bump. Review the snapshot for accidental exposure (private names, test helpers); fix by `__all__` hygiene before freezing.

### P17.2 Licence and metadata (needs D1, D8)
Apply the decision. If staying GPL-3.0: keep `LICENSE`, set classifier "GNU General Public License v3 (GPLv3)". If switching to BSD-3-Clause or MIT: replace `LICENSE`, update `pyproject.toml` (`license` and classifier), update file headers if any, and note in the changelog. **Do not change the licence without the maintainer's explicit written decision (R14).** Add `CITATION.cff` (title, authors, version, repository URL, license), `SECURITY.md` (how to report), `CODE_OF_CONDUCT.md` (Contributor Covenant, adopted verbatim by the maintainer).

### P17.3 Final quality sweep
Run, record in the log, and fix findings:
```bash
make ci && make coverage && make docs
pytest -m "slow" -q                    # nightly/slow tests
python -m benchmarks.run --all         # sanity run (not part of gate)
ruff check . && mypy econkit           # strict
python -W error -c "import econkit"    # no import-time warnings
```
Confirm coverage floors (R11). Confirm test counts: all tests green with and without Numba (CI matrix), and with JAX in its own job.

### P17.4 Build and clean-room install
`python -m build` produces a pure-Python `py3-none-any` wheel and an sdist (EconKit has no compiled extension in v1.0; cibuildwheel is not needed until a C/Rust layer exists). `twine check dist/*`. Then, in a *fresh* virtualenv outside the repo directory: `pip install dist/econkit-*.whl`; `python scripts/smoke_installed.py` (imports, solves a small consumption–savings model with EGM, simulates, prints `ok` — no pytest, no repo checkout); repeat with `pip install "econkit[all]"` from the wheel. Inspect `unzip -l dist/*.whl` for stray files (tests, `.econkit`, `__pycache__`) and confirm `py.typed` is present. Run the whole test suite against the *installed* wheel (`pytest --pyargs econkit` is not supported unless tests ship; instead run `pytest tests/` with `PYTHONPATH` empty from a copy of the repo) to catch packaging omissions.

### P17.5 Release candidate (human-gated)
Set version `1.0.0rc1`; commit; tag locally `v1.0.0rc1`. **Stop and ask the maintainer to:** (1) create PyPI and TestPyPI accounts/projects (check name availability — D5 — before anything else), (2) configure **trusted publishing** (OIDC) for the repository and `release.yml`, (3) push the tag. The agent prepares `release.yml` (build → `twine check` → publish to TestPyPI on `rc` tags, to PyPI on final tags using environment protection) but does not publish and does not push tags.

### P17.6 Soak
After the rc is on TestPyPI: install from TestPyPI in a clean environment on Linux, macOS and Windows (maintainer or CI), run the tutorials end-to-end, collect issues. The agent fixes only release-blocking bugs (anything that makes a tutorial or smoke test fail, or a wrong number). Everything else goes to the post-1.0 list.

### P17.7 Final changelog and release notes
`CHANGELOG.md` `## [1.0.0] - <date>` with Added (grouped by module), Changed, Removed (the baseline items listed under §3.3), Known limitations (Section 3.3 items, JAX CPU performance caveat, Windows/Numba status). `docs/release_notes_1.0.md` with the story: what v1 is, what is stable, how compiled layers will arrive without breaking code (architecture §19.12 promise restated).

### P17.8 Final release (human-gated)
Version `1.0.0`; commit; tag `v1.0.0`; maintainer pushes the tag; `release.yml` publishes to PyPI after maintainer approval of the environment; GitHub Release created from the changelog section. The agent then verifies from a clean environment: `pip install econkit==1.0.0`, runs `scripts/smoke_installed.py`, and records the result.

### P17.9 Post-release hygiene
Branch `release/1.0.x` for patch releases; reopen `[Unreleased]` in the changelog; bump `_version.py` to `1.1.0.dev0`; create issues for every Parking-lot item and label them (`bug`, `performance`, `enhancement`, `good first issue`, `docs`); pin a "Roadmap to 1.1" issue copied from Part E. Optional: Zenodo DOI via the GitHub integration (maintainer), DOI added to `CITATION.cff` in a patch release.

### P17.10 Retrospective
Append a final Captain's Log entry: planned vs actual sessions per phase, which acceptance tests caught real bugs, which tolerances had to be tightened, final measured benchmark numbers, and the three biggest lessons. This is the record that makes the next major release plan better.

### P17.11 Supply-chain hygiene
Dependabot for pip and GitHub Actions; Actions pinned to a major version or commit SHA; a non-gating `pip-audit` job; trusted publishing (already in P17.5); `SECURITY.md` states the response time. Configuration files only; no new library code.

**Definition of Done for v1.0.0.** Everything in Appendix D is ticked; `pip install econkit==1.0.0` works in a clean environment; docs are live; the tag exists.
**Session budget.** 3 plus soak time. **Pitfalls.** Publishing is irreversible (PyPI does not let you reuse a version number). Never publish from a dirty tree; never publish without TestPyPI first; never `--skip-existing` blindly.

---

# Part E — After v1.0 (roadmap, not part of the v1 contract)

The order below follows the architecture's layering principle: add capability and add speed behind the same public API. **Reinstatement rule (R21h):** a capability deferred or removed from v1 returns only together with its first consumer, a test and a documentation page; none of it is kept as dormant code in the meantime.

| Release | Theme | Contents |
|---|---|---|
| 1.1 | Heterogeneous-agent depth and transitions | Krusell–Smith aggregate-uncertainty solver (Den Haan error diagnostics) and with it 2-D/3-D multilinear interpolation; transition paths after MIT shocks; second-order perturbation; time iteration for models EGM cannot handle (endogenous labour supply); discrete-choice EGM / upper envelope; Mortensen–Pissarides; Auerbach–Kotlikoff (needs the transition machinery); the `Agent` abstraction if multi-agent models arrive |
| 1.2 | Compiled C layer (justified by profiles only) | pybind11 (stable ABI / `Py_LIMITED_API`) kernels for the two or three hottest loops that remain slow under Numba, wheels via cibuildwheel for Linux/macOS/Windows; dispatcher registration `("op","c")`; same parity tests; zero API change |
| 1.3 | Scale and breadth | Higher-dimensional state spaces, projection methods with cubic splines and Chebyshev approximation, Smolyak/adaptive grids, a fresh two-asset (HANK-style) solver, continuous-time HJB models (Merton), multigrid VFI, Brent/golden-section refinement if profiling asks for it, HMC/NumPyro, CCP and simulated MLE, Bayes factors/WAIC, first-class Polars (and, with it, `EconTimeSeries`/`EconPanel` if they would add behaviour), SymPy symbolic front end (D7), YAML model files and HDF5 persistence, PSID/SIPP/BHPS/CEX/SCF/PWT/World Bank loaders, plotly and dashboards, MPI-style sweeps |
| 2.0 candidates | Rust/PyO3 layer where memory safety and concurrency matter (large simulations, MCMC), WebAssembly demos, plugin system, model hub | only after the above are stable; again behind the dispatcher |

Rules for every post-1.0 compiled layer: (1) the NumPy reference stays; (2) a kernel is added only when the P15 benchmark shows the current best backend misses a target by a clear margin; (3) every new kernel ships with a cross-backend parity test and a benchmark case; (4) no public-API change; (5) the wheel matrix and platform support are documented before the first release.

---

# Appendices

## Appendix A — Session starter prompt (paste into Claude Code)

```text
You are working on EconKit (repo root = current directory).
Execute phase <PHASE ID, e.g. P5> of docs/md/V1_EXECUTION_PLAN.md.

Before changing code:
1. Run `make phase P=<n>` and read `.phase/P<n>.md` (operating protocol R1–R21 plus this one phase).
2. Read the last two entries of docs/md/CAPTAINS_LOG.md and docs/md/V1_PROGRESS.md.
3. Run `make install && make ci`; report the test count and any skips.
4. Create a task list from the phase's sub-steps (one task per P<phase>.<step>).

While working:
- Work the sub-steps in order. Reference implementation first (R3), tests with the stated
  analytic benchmarks (R8), docstrings with "Economics note:" (R7).
- Commit per sub-step: "P<phase>.<step>: <summary>".
- If a stop-and-ask trigger (R14) fires, write it under Blockers in V1_PROGRESS.md and stop.
- Do only this phase (R13). Add out-of-scope ideas to the Parking lot.
- Lean rule (R21): reuse before adding; delete replaced code and port its tests in the same commit;
  no shims, no unused options; tag unavoidable not-yet-called code `# NEAR-FUTURE(P<n>.<m>)`.

Before finishing:
- Run the phase's Definition of Done commands (including `make deadcode`) and paste the output summary.
- Tick only the boxes whose acceptance tests pass in V1_PROGRESS.md.
- Append a Captain's Log entry using the template in Appendix B.
- Report: what was done, what failed or was deferred, measured numbers, next step.
Do not claim completion percentages. Do not push tags or publish packages.
```

## Appendix B — Captain's Log entry template

```markdown
## Session <N> — <date> — <Phase and steps, e.g. P5.1–P5.4>

**Scope:** one sentence.
**Files changed:** list (new / modified).
**Checklist ticked:** P5.1, P5.2, ...
**Gate output:** `make ci` -> <passed/failed counts, skips>; coverage -> <numbers per package>.
**Analytic checks:** <name: measured error vs tolerance>.
**Measured performance (if any):** <case, backend, seconds, machine>.
**Deleted / retired under R21:** code removed, tests ported or dropped (with reasons), NEAR-FUTURE tags added or removed (or "none").
**Deviations from plan:** what and why (or "none").
**Blockers / questions for maintainer:** (or "none").
**Next step:** the next sub-step or phase.
```

## Appendix C — Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Numba/JAX wheels unavailable for a new Python version | Medium | Medium | Support matrix 3.10–3.12, 3.13 non-blocking (D6); all kernels have a NumPy fallback; `ECONKIT_DISABLE_JIT` |
| 2 | Agent self-reports progress inflated again | Medium | High | R6: only ticked boxes backed by passing acceptance tests; review gates after P6, P11, P16 |
| 3 | Numerical tolerance set too loose to catch bugs | Medium | High | Analytic benchmarks (Appendix E); R9 forbids loosening >10× without asking; tighten to 3× measured error |
| 4 | Speedup targets missed | Medium | Low | R19: record real numbers and explain; do not alter targets or claims |
| 5 | Scope creep into HANK/Krusell–Smith before v1 | Medium | High | §3.3 fixes scope; R13 Parking lot |
| 6 | API churn after Review-2/3 | Medium | Medium | Early review gates; API policy; API snapshot at P17.1 |
| 7 | PyPI name `econkit` already taken | Unknown | Medium | D5: check in P0 and again at P17.5; fallback name chosen by maintainer |
| 8 | GPL-3.0 limits adoption | High | Medium | D1 raised early; decision needed before P17.2 |
| 9 | Windows/macOS CI failures from threading or Numba caching | Medium | Medium | `continue-on-error` for new platforms at first; cache dirs configurable; log and fix before rc |
| 10 | Test-suite runtime balloons | Medium | Medium | `slow` marker; unmarked tests total < 3 min (check in P17.3); nightly slow job |
| 11 | JAX x64 global side effect surprises users | Low | Medium | Documented (P12.1); no opt-out because every tolerance assumes `float64` |
| 12 | Random-number reproducibility across NumPy versions | Low | Medium | Pin generator (PCG64) in `as_generator`; tests compare statistics, not draws, across versions |
| 13 | CI `.github/` still ignored by git | Medium | High | P0.5 un-ignores it (D3) and adds a test in `scripts/` that fails if `.github/workflows/ci.yml` is ignored |
| 14 | Claude Code session loses context mid-phase | High | Low | Phases split into sub-steps with commits and log entries; the progress file is the source of truth; `make phase` keeps the per-session reading small |
| 15 | Over-pruning: something removed under R21 turns out to be needed | Medium | Low | Each removal is its own commit and the baseline is tagged; Part E's reinstatement rule brings a capability back with its consumer |
| 16 | Dead-code tools give false positives and the agent silences them with allow-lists | Medium | Medium | Every allow-list entry needs a one-line reason; allow-list size and `NEAR-FUTURE` tags are reviewed at each gate |

## Appendix D — Master release checklist (all must be true for v1.0.0)

**Correctness**
- [ ] All analytic benchmarks in Appendix E pass; measured errors recorded in the log.
- [ ] Cross-backend parity tests exist for every registered hot-path operation (meta-test passes).
- [ ] Determinism tests: same seed → identical results, any thread count.

**Engineering**
- [ ] `make ci`, `make coverage` (floors met), `make docs` (`-W`) green on Python 3.10, 3.11, 3.12 on Linux, macOS, Windows (Windows may carry documented caveats).
- [ ] CI passes with and without Numba; JAX job green.
- [ ] `import econkit` pulls in no optional dependency (P13.9).
- [ ] API snapshot frozen (P17.1); `__all__` lists reviewed.
- [ ] No `TODO`/`FIXME` left in `econkit/` without a linked issue.
- [ ] `make deadcode` passes with **no** `NEAR-FUTURE` tags left; both allow-lists reviewed; no compatibility shims or `legacy` code in `econkit/`.
- [ ] Every retired step ID is listed in §7 and no retired capability is present in the code.

**Content**
- [ ] Nine built-in models with docs pages; nine executed tutorials.
- [ ] Every public symbol documented; `Economics note:` present where required.
- [ ] README and docs contain only measured benchmark numbers.

**Governance**
- [ ] Licence decision applied (D1); `CITATION.cff`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `CHANGELOG.md` present.
- [ ] PyPI name confirmed (D5); trusted publishing configured by maintainer.
- [ ] TestPyPI install verified on a clean machine; final `pip install econkit==1.0.0` verified.

## Appendix E — Analytic benchmark catalogue

| Domain | Benchmark | Statement | Used in |
|---|---|---|---|
| Growth | Brock–Mirman | `u=ln c`, `y = z k^α`, `δ=1`: `k' = αβ z k^α`, `c = (1−αβ) z k^α`; holds for any process for `z` | P5.8, P6.5, P9.6, P10.2, P7.5 |
| Consumption | Permanent income | CRRA, `β(1+r)=1`, constant income, no binding limit: `c = y + r a`, `a' = a` | P5.9, P7.3(c) |
| Consumption | Perfect-foresight MPC | `MPC = 1 − β^{1/σ}(1+r)^{(1−σ)/σ}` | P10.1 |
| Growth | RBC steady state | `k/y = α/(1/β − 1 + δ)`, `i/y = δ k/y` | P10.3 |
| Search | McCall reservation wage | `w̄ = c + β/(1−β) · E[(w'−w̄)^+]`; uniform[0,1]: `k w̄² − (2k+1) w̄ + (k + c) = 0`, `k = β/(2(1−β))` | P10.4 |
| OLG | Diamond (log, CD, δ=1) | `k* = [β(1−α)/((1+β)(1+n))]^{1/(1−α)}` | P10.7 |
| NK | Closed-form solution (φ_x=0) | `x = b v`, `π = a v` with `b = −1/(σ(1−ρ) + κ(φ_π−ρ)/(1−βρ))`, `a = κ b/(1−βρ)` | P9.6, P10.8 |
| NK | Determinacy | `κ(φ_π − 1) + (1−β) φ_x > 0` | P9.6 |
| Asset pricing | Lucas tree, log utility | `p(y)/y = β/(1−β)` | P10.9 |
| GE | Precautionary saving | Aiyagari/Huggett: `r* < 1/β − 1`, decreasing in income risk | P8.4 |
| Markov | 2-state chain | stationary `(q/(p+q), p/(p+q))` | P3.4 |
| Markov | AR(1) moments | `Var = σ²/(1−ρ²)`, `Corr_1 = ρ` | P3.4, P9.6, P11.4 |
| Distribution | Young lottery | next-period mean = policy value; mass conserved | P7.3 |
| Inequality | Gini | uniform → 1/3; lognormal → `2Φ(σ/√2) − 1`; equal → 0 | P7.4 |
| Welfare | CEV scaling | `c → (1+λ)c` ⇒ CEV = λ | P14.4 |
| Filtering | Local-level Kalman | `P̄ = (q+√(q²+4q))/2`, `K = P̄/(P̄+1)` | P11.6 |
| Statistics | MLE for AR(1) | conditional MLE = OLS, `se(ρ̂) ≈ √((1−ρ²)/T)` | P11.5 |
| Elasticity | Brock–Mirman | `d ln k*/d ln β = 1/(1−α)` | P14.6 |
| Interpolation | Linear exactness | exact at nodes; `"linear"` extrapolation exact for affine functions | P2.2, P2.10 |
| Iteration | Cosine fixed point | `x = cos x` ⇒ `x* = 0.7390851332151607` | P3.3 |

## Appendix F — Command cheat sheet

```bash
make install            # create .econkit venv and install -e .[dev]
make ci                 # test-structure guard, ruff, mypy, deadcode, pytest
make deadcode           # vulture + check_usage + check_near_future (R21)
make phase P=5          # write .phase/P5.md: protocol + one phase (P0.11)
make coverage           # pytest with coverage report
make docs               # build Sphinx documentation (P16)
make bench              # run benchmark harness (P15)
pytest -m "not slow"    # fast test subset
pytest -m numba -q      # numba-only tests
pytest -m jax -q        # jax-only tests
pytest -m external -q   # optional cross-checks against QuantEcon (P10.13)
ECONKIT_DISABLE_JIT=1 pytest -q    # run kernels as pure Python (debugging, coverage)
python scripts/backend_matrix.py   # operation x backend table
python scripts/dump_api.py         # public API snapshot (P17)
```

## Appendix G — Dependency policy

| Dependency | Role | Status |
|---|---|---|
| numpy ≥ 1.25 | arrays, reference kernels | required |
| scipy ≥ 1.11 | `ordqz`, sparse, Lyapunov, `brentq`/`root`/`minimize`, `scipy.stats` | required (D2) |
| numba ≥ 0.59 | hot-path compilation | optional extra `numba`; in `dev` |
| jax, jaxlib ≥ 0.4.25 | autodiff, vmap, GPU | optional extra `jax` |
| pandas ≥ 2.0 | `to_dataframe()`, FRED series | optional extra `data` |
| matplotlib ≥ 3.7 | plots | optional extra `viz` |
| sphinx, myst-nb, furo, jupytext | docs | extra `docs` |
| pytest, pytest-cov, hypothesis, ruff, mypy, vulture | development | extra `dev` |
| quantecon | optional cross-checks (P10.13) | installed only by the CI `external` job; never a dependency |

**Not allowed without a stop-and-ask (R14):** any new required dependency; scikit-learn, statsmodels, SymPy (D7), Polars, Dask, or MPI as a core dependency; any dependency with a licence incompatible with the chosen project licence (D1).

---

*End of plan. This document is a living artefact: when reality disagrees with it, update the plan in a dedicated commit (`docs: amend V1 plan — <reason>`) and log the change rather than silently deviating.*
