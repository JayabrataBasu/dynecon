"""Exact gradient of the Smets-Wouters log likelihood, checked against finite differences.

The method lives in ``dynecon.solvers.perturbation.dynare_grad``; this script wires it to the
replication package and prints the agreement and the speed-up.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
from real_sw_estimate import SW

from dynecon.solvers.perturbation.dynare_grad import loglik_and_grad as _grad

warnings.filterwarnings("ignore", category=RuntimeWarning)


def loglik_and_grad(sw: SW, x):
    return _grad(sw.model, sw.data, x, sw.params, sw.obs_names, presample=4)


def main(folder: str) -> None:
    sw = SW(folder)
    x = np.array(sw.mode)
    loglik_and_grad(sw, x)  # compile
    t = time.time()
    val, g = loglik_and_grad(sw, x)
    t_exact = time.time() - t
    print(f"loglik {val:.6f} (reader: {sw.loglik(x):.6f})   exact gradient {t_exact * 1000:.0f} ms")
    t = time.time()
    fd = np.empty_like(g)
    for i in range(len(x)):
        step = 1e-6 * max(1.0, abs(x[i]))
        up, dn = x.copy(), x.copy()
        up[i] += step
        dn[i] -= step
        fd[i] = (sw.loglik(up) - sw.loglik(dn)) / (2 * step)
    t_fd = time.time() - t
    err = np.abs(g - fd) / (1 + np.abs(fd))
    print(f"finite differences {t_fd * 1000:.0f} ms; max relative error {err.max():.2e}")
    print("speedup", round(t_fd / t_exact, 1))


if __name__ == "__main__":
    main(sys.argv[1])
