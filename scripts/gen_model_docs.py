"""Generate ``docs/models/*.md`` and ``docs/models/index.md`` from the model registry."""

from __future__ import annotations

import importlib
from pathlib import Path

from dynecon import models

ROOT = Path(__file__).resolve().parents[1]

# Per-model prose that cannot be derived from code: equations, method, benchmark and test.
META: dict[str, dict[str, str]] = {
    "consumption_savings": {
        "equations": "$V(a,z)=\\max_{a'\\ge a_{min}}u(c)+\\beta E[V(a',z')]$, $c=(1+r)a+z-a'$; $\\ln z$ AR(1).",
        "method": "EGM (default); VFI for comparison.",
        "benchmark": "Permanent-income limit ($\\beta(1+r)=1$, no risk: $a'=a$); the infinite-horizon MPC "
        "$1-\\beta^{1/\\sigma}(1+r)^{(1-\\sigma)/\\sigma}$ (informational).",
        "test": "Policy identity under permanent income; concavity of $c(a)$; the constraint binds only at "
        "low wealth and income.",
        "example": "sol = ek.solve(models.load('consumption_savings').compile())",
    },
    "brock_mirman": {
        "equations": "$k'=\\alpha\\beta zk^\\alpha$, $c=(1-\\alpha\\beta)zk^\\alpha$ with log utility and full depreciation.",
        "method": "VFI, EGM (`saving` form) and perturbation (`problem='dsge'`).",
        "benchmark": "The closed-form policy above (exact for every $z$ process).",
        "test": "VFI within two grid spacings, EGM to 1e-10, perturbation coefficients $(\\alpha,1)$ to 1e-10.",
        "example": "sol = ek.solve(models.load('brock_mirman', problem='dsge').compile())",
    },
    "rbc": {
        "equations": "Euler $c^{-\\sigma}=\\beta E[c'^{-\\sigma}(\\alpha z'k'^{\\alpha-1}+1-\\delta)]$, $k'=(1-\\delta)k+i$, $y=zk^\\alpha$.",
        "method": "Perturbation (default); EGM via `problem='saving'`.",
        "benchmark": "Steady state $k/y=\\alpha/(1/\\beta-1+\\delta)$, $i/y=\\delta k/y$.",
        "test": "Steady-state identities, $\\delta=1$ reproduces Brock-Mirman, perturbation vs EGM near the steady state, IRF shape.",
        "example": "sol = ek.solve(models.load('rbc').compile()); irf = irf(sol, 'eps_z')",
    },
    "mccall": {
        "equations": "$\\bar w=c+\\frac{\\beta}{1-\\beta}E[(w'-\\bar w)^+]$.",
        "method": "Tabular VFI on a `finite_dp` problem.",
        "benchmark": "Uniform offers: $k\\bar w^2-(2k+1)\\bar w+(k+c)=0$, $k=\\beta/(2(1-\\beta))$.",
        "test": "Reservation wage within one grid spacing; comparative statics in $c$ and $\\beta$; threshold rule.",
        "example": "w = mccall.reservation_wage(ek.solve(models.load('mccall').compile()))",
    },
    "aiyagari": {
        "equations": "Household as consumption-savings with wage $w(r)$; firm $r=\\alpha(K/L)^{\\alpha-1}-\\delta$.",
        "method": "EGM + Young distribution + Brent on the capital market.",
        "benchmark": "$r^*<1/\\beta-1$ strictly; $K/Y=\\alpha/(r^*+\\delta)$.",
        "test": "Market clearing, $r^*$ decreasing in income risk, warm starts save iterations.",
        "example": "eq = aiyagari.equilibrium(sigma=2.0, rho=0.9, sigma_e=0.2)",
    },
    "huggett": {
        "equations": "Bonds in zero net supply; borrowing limit $-\\phi$.",
        "method": "EGM + Young distribution + Brent on the bond market.",
        "benchmark": "$r^*<1/\\beta-1$; a tighter limit lowers $r^*$.",
        "test": "Bond market clears; wealth Gini exceeds income Gini.",
        "example": "eq = huggett.equilibrium(phi=2.0)",
    },
    "olg2": {
        "equations": "$(1+n)k'=\\frac{\\beta}{1+\\beta}(1-\\alpha)k^\\alpha$.",
        "method": "Exact law of motion; perturbation for the slope; Euler root-finding as a check.",
        "benchmark": "$k^*=[\\beta(1-\\alpha)/((1+\\beta)(1+n))]^{1/(1-\\alpha)}$, globally stable.",
        "test": "Steady state three ways to 1e-10; convergence from three starting points.",
        "example": "sol = ek.solve(models.load('olg2').compile())",
    },
    "nk3": {
        "equations": "$\\pi=\\beta E\\pi'+\\kappa x$; $x=Ex'-(i-E\\pi')/\\sigma$; $i=\\phi_\\pi\\pi+\\phi_xx+v$.",
        "method": "Perturbation (Klein).",
        "benchmark": "With $\\phi_x=0$: $x=bv$, $b=-1/(\\sigma(1-\\rho)+\\kappa(\\phi_\\pi-\\rho)/(1-\\beta\\rho))$, $\\pi=\\kappa b v/(1-\\beta\\rho)$; "
        "determinate iff $\\kappa(\\phi_\\pi-1)+(1-\\beta)\\phi_x>0$.",
        "test": "Closed-form coefficients; determinacy grid; Bullard-Mitra case.",
        "example": "sol = ek.solve(models.load('nk3').compile())",
    },
    "lucas_tree": {
        "equations": "$p(y)u'(y)=\\beta E[u'(y')(p(y')+y')]$.",
        "method": "Fixed point of the price functional.",
        "benchmark": "Log utility: $p(y)/y=\\beta/(1-\\beta)$ for any dividend process.",
        "test": "Constant price-dividend ratio; sign of its slope for $\\sigma\\gtrless1$; risk-free rate two ways.",
        "example": "sol = ek.solve(models.load('lucas_tree', sigma=3.0).compile())",
    },
}


def main() -> int:
    out_dir = ROOT / "docs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = models.available()
    for name in names:
        mod = importlib.import_module(f"dynecon.models.{name}")
        meta = META[name]
        rows = "\n".join(
            f"| `{k}` | {p.value:g} | {p.bounds} | {p.description} |" for k, p in mod.PARAMS.items()
        )
        defaults = getattr(mod, "DEFAULTS", {})
        opts = ", ".join(f"`{k}={v!r}`" for k, v in defaults.items()) or "none"
        text = f"""# {name}

{(mod.__doc__ or "").strip().split(chr(10) + chr(10))[0]}

**Economics.** {(mod.__doc__ or "").split("Economics note:")[-1].strip()}

**Equations.** {meta["equations"]}

## Calibration

Source: {mod.CALIBRATION_SOURCE}

| parameter | default | bounds | description |
|---|---:|---|---|
{rows}

Structural options (not parameters): {opts}.

## Solution method

{meta["method"]}

## Usage

```python
import dynecon as ek
from dynecon import models
{("from dynecon.models import " + name) if name in ("aiyagari", "huggett", "mccall") else ""}
{meta["example"]}
```

## Benchmark and test

**Benchmark:** {meta["benchmark"]}

**What the test checks:** {meta["test"]} (see `tests/models/` and `tests/solvers/perturbation/`).
"""
        (out_dir / f"{name}.md").write_text(text, encoding="utf-8")
    index = "# Built-in models\n\n```{toctree}\n:maxdepth: 1\n\n" + "\n".join(names) + "\n```\n"
    (out_dir / "index.md").write_text(index, encoding="utf-8")
    print(f"wrote {len(names)} model pages to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
