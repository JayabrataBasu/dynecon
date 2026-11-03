# Dynecon: hand-off notes (read this first in a new chat)

Written at the end of the long build session so work can continue in a fresh conversation.
Tell the assistant: "Read `docs/md/HANDOFF.md`, then `docs/md/CAPTAINS_LOG.md` (last two sessions)
and `docs/md/V1_PROGRESS.md`."

## 1. Project in one paragraph

**Dynecon** (import name `dynecon`, PyPI name `dynecon`) is a Python library for computational
economics: a model DSL (`saving`, `finite_dp`, `dsge`, `pricing` templates), solvers (VFI, EGM,
Klein perturbation, Lucas pricing, stationary equilibrium), simulation and Young distributions,
estimation (SMM, MLE, Kalman, MCMC, identification), nine built-in models, a JAX backend,
filters/plots/experiments/welfare, benchmarks and docs. It was built by following
`docs/md/V1_EXECUTION_PLAN.md` (the source of truth for scope) and
`docs/md/Dynecon_Architecture.md` (the design rationale).

## 2. Identity and licence decisions (made by the maintainer)

- Name: **Dynecon**. Originally EconKit / `econkit`; renamed because `econkit` is already taken on
  PyPI by another project. The GitHub repo has been renamed to `JayabrataBasu/dynecon`.
- Author / copyright holder: **Jayabrata Basu**, jayabratabasu@gmail.com (sole author).
- Licence: **BSD-3-Clause** (changed from GPL-3.0; decision D1 closed, D8 closed).
- PyPI: `dynecon` is registered as a **pending trusted publisher** (owner `JayabrataBasu`, repo
  `dynecon`, workflow `release.yml`, environment `pypi`). Nothing has been uploaded.

## 3. State of the code (all committed on `master`, working tree was clean)

- Last commits: `80348c4` repo URLs -> `dynecon`; `120fc0b` rename; `1fb3ef2` BSD licence; earlier
  commits P0-P17 (see `git log`). Tag `v0.1.0-baseline` marks the pre-rewrite code (kept on purpose).
- Gates (run `python scripts/ci.py`, add `--coverage` for coverage, `--skip-tests` for static only):
  278 tests passed / 1 skipped, coverage 97%, ruff, mypy, vulture, usage / near-future / docstring
  checks all green. Sphinx builds with `-W`. Wheel + sdist build and pass `twine check`. Clean-room
  install in a fresh venv (no Numba, no JAX) passes `scripts/smoke_installed.py`.
- Progress: 139 of 150 plan steps ticked in `docs/md/V1_PROGRESS.md` (checklist only, no percentages).
- `import dynecon` takes about 0.3 s and loads no numba / pandas / matplotlib / jax (lazy JIT and
  lazy SciPy submodule imports; a test guards it).
- Benchmarks are recorded in `benchmarks/results/*.json`; `docs/md/benchmarks.md` is generated from
  them by `scripts/gen_bench_doc.py`. Only numbers from that file may be quoted.

## 4. Things that were NOT verified (be honest about these)

- **Only Python 3.14 on Windows 11 was run.** The CI matrix (Python 3.10-3.12; Linux, macOS) and every
  GitHub workflow (`ci.yml`, `docs.yml`, `nightly.yml`, `release.yml`, `dependabot.yml`) have **never run**.
  Declared minimums (`numpy>=1.25`, `scipy>=1.11`, `numba>=0.59`, `jax>=0.4.25`) are untested.
- The FRED download endpoint (`fredgraph.csv`) is unverified against the live service.
- GitHub Actions are pinned by tag, not commit SHA; trusted publishing is only "pending" on PyPI.

## 5. Honest results worth remembering

- EGM is more accurate than VFI but **not faster** than Numba VFI at the default 1e-10 tolerance
  (plan goal of 20x missed; explained in `docs/md/benchmarks.md`). Other goals met (Bellman 30x,
  panel simulation 20x, histogram step 4x, VFI vs NumPy 7.7x).
- One plan claim was wrong: for sigma > 1 the Lucas-tree price-dividend ratio *rises* with the dividend.
- Aggregate welfare (CEV) is evaluated on the **baseline** stationary distribution (the plan's
  "average over each regime's own distribution" gave a negative CEV for relaxing a borrowing limit).
- 15 deliberate deviations from the plan are listed in `docs/md/CAPTAINS_LOG.md` (Session 14).

## 6. Open items / next steps

1. Create the GitHub environment `pypi` (repo Settings -> Environments), push, and watch the CI matrix
   run; fix whatever fails on Python 3.10-3.12 / Linux / macOS (first real run).
2. Update local `origin`: `git remote set-url origin https://github.com/JayabrataBasu/dynecon.git`.
3. **Commit trailers:** every commit made in the build session ends with
   `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. To appear as sole author on GitHub,
   strip these from the history **before the first push** (rewrites history; safe only while unpushed).
4. Decide when to publish: publish a GitHub release (runs `release.yml`) after the human review gates in
   the plan (Review-2/3/4, release candidate, soak). Consider bumping the version from `0.1.0.dev0`
   to `1.0.0rc1` first (edit `dynecon/_version.py`, CHANGELOG, `CITATION.cff`).
5. Unticked plan steps: P10.13 (optional QuantEcon cross-check), P15.5 (profile the three slowest
   benchmarks, only EGM was profiled), P16.8 (host the docs), P16.9 (docs review), P17.5-P17.10 (release
   steps), P17.11 (pin actions by SHA, configure trusted publisher).
6. Parking lot is at the bottom of `docs/md/V1_PROGRESS.md` (EGM speed, FRED check, etc.).
7. Confirm HARK's preferred citation in `paper/paper.bib` (marked TODO) before any JOSS submission.

## 7. How to work in this repo

- Python env: the old venv `.econkit` has absolute paths baked in and **breaks when the folder is
  renamed**. Recreate: `python -m venv .venv`, activate, `pip install -e ".[dev]"` (add `,jax` for the JAX
  tests and `,docs` for Sphinx). Docs and the Makefile refer to `.venv`.
- `make` is not installed on the maintainer's Windows machine: use `python scripts/ci.py`.
- Windows launcher entries for Python 3.12 / 3.13 are broken; Python 3.14 works.
- Useful commands: `python scripts/ci.py --coverage`; `python -m benchmarks.run --all`;
  `python scripts/gen_bench_doc.py`; `python scripts/gen_model_docs.py`;
  `python scripts/run_tutorials.py`; `sphinx-build -b html docs docs/_build/html -W --keep-going`;
  `python scripts/dump_api.py` (regenerate `docs/md/API_SURFACE.txt` if the public API changes on purpose);
  `pytest tests/golden --update-golden` (only with a changelog line and reason).
- Pitfalls found: very long `python - <<EOF` shell commands often fail to parse (write files with the
  editor/Write tool instead); `pytest --cov=<subpackage>` breaks (numpy double import; use plain `--cov`);
  do not edit source while a coverage run is in progress.
- Rules of the project (from the plan): R3 NumPy reference first, compiled backends need parity tests;
  R21 lean by construction (no dead code, every public symbol has a caller and a test, or an entry with a
  reason in `scripts/usage_allowlist.txt`); R19 performance claims only from recorded benchmarks;
  tests prefer analytic benchmarks and fixed seeds.
- History is intentionally kept under the old name in `CAPTAINS_LOG.md`, `REVIEW_1.md` and the git tag
  `v0.1.0-baseline`; `scripts/legacy_vfi_timing.py` imports `econkit` inside an embedded snippet because it
  runs the tagged baseline.

## 7a. Renaming the project folder (and keeping the Claude conversation)

Claude Code stores conversations per project path under
`C:\Users\jayab\.claude\projects\<path with \ : . replaced by ->`, e.g. `C--Users-jayab-EconKit`
(this folder also holds the project memory). A renamed folder maps to a new, empty store, so
`claude --resume` will probably not show the old chat (not verified). To try to keep it: exit Claude,
rename the project folder, rename the matching folder under `.claude\projects\`, recreate `.venv`, start
Claude from the new folder. A new chat plus this file works equally well.

## 8. Layout cheat sheet

```
dynecon/            library (core, utils, numerics[+backends], model, models, solvers[dp, perturbation],
                    simulation, estimation, experiment, data, viz)
tests/              per-domain folders + golden/ (npz) + jax/ + engineering/ + api/
benchmarks/         harness, cases, run.py, results/*.json
scripts/            ci.py, check_*.py, gen_*.py, run_tutorials.py, dump_api.py, legacy_vfi_timing.py ...
docs/               conf.py, index.md, user_guide/, tutorials/ (9 executable), theory/, models/ (generated),
                    api/, md/ (plan, architecture, progress, log, guides, benchmarks, this file)
paper/              JOSS draft
.github/            workflows (ci, docs, nightly, release), dependabot, templates
```
