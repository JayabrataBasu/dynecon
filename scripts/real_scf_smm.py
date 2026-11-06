"""Fit a one-asset savings model to SCF wealth moments with exact-gradient SMM (+ HMC).

Usage: python scripts/real_scf_smm.py MOMENTS.json   (made by real_scf_moments.py)
"""

from __future__ import annotations

import json
import sys
import time

import jax.numpy as jnp
import numpy as np

from dynecon import models
from dynecon.estimation.smm_exact import smm_exact


def moment_fn(mu, c, a_next, a_grid, z, p):
    """[wealth/income, Gini, bottom-50% share, top-10% share] of the stationary distribution."""
    m = mu.sum(axis=1)
    inc = (mu.sum(axis=0) * z).sum()
    w = m * a_grid
    pop = jnp.concatenate([jnp.zeros(1), jnp.cumsum(m)])
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(w) / w.sum()])
    gini = 1.0 - jnp.sum((cum[1:] + cum[:-1]) * m)
    return jnp.array(
        [
            w.sum() / inc,
            gini,
            jnp.interp(0.5, pop, cum),
            1.0 - jnp.interp(0.9, pop, cum),
            1.0 - jnp.interp(0.99, pop, cum),
        ]
    )


def main(path: str, patient: bool = False) -> None:
    s = json.load(open(path))
    m_hat, omega, n = np.array(s["moments"]), np.array(s["omega"]), s["n_households"]
    model = models.load(
        "consumption_savings",
        n_a=200,
        n_z=7,
        a_max=3000.0,
        curvature=4.0,
        r=0.02,
        sigma=2.0,
        beta=0.95,
        rho=0.95,
        sigma_e=0.2,
        beta_patient=0.975,
        patient_share=0.04 if patient else 0.0,
    ).compile()
    free = ["beta", "sigma_e", "rho"] + (["beta_patient", "patient_share"] if patient else [])
    t = time.time()
    res = smm_exact(
        model,
        free,
        m_hat,
        moment_fn,
        omega,
        n_obs=n,
        starts=[[0.92, 0.22, 0.99, 0.975, 0.03]] if patient else [[0.97, 0.3, 0.9]],
        patient=patient,
    )
    print(f"smm_exact {time.time() - t:.1f}s converged={res.converged} evals={res.diagnostics}")
    for k, e, se in zip(free, res.estimates, res.std_errors):
        print(f"  {k}: {e:.4f} (se {se:.4f})")
    print("data  ", np.round(m_hat, 3))
    print("model ", np.round(res.model_moments, 3))
    print(f"J = {res.j_stat}, df={res.df}, p={res.j_pvalue}")


if __name__ == "__main__":
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "patient")
