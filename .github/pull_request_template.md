## What and why

<!-- Economic interpretation of the change and the numerical method involved. -->

## Checklist

- [ ] `make ci` passes locally (layout, lint, mypy, tests, dead-code gate)
- [ ] New numerical kernels: NumPy reference first, compiled backend registered with a parity test
- [ ] Analytic benchmark or property test added where a closed form exists
- [ ] Public functions have docstrings with an `Economics note:` line
- [ ] `CHANGELOG.md` updated; no unmeasured performance claims
- [ ] No dead code, no compatibility shims (rule R21)
