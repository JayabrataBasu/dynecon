"""Exact gradient of the Smets-Wouters log posterior (prototype), checked against finite differences.

Solution y_t = G y_{t-1} + H e_t of  C0 E y' + C1 y + C2 y_{-1} + D e = 0  satisfies
  C0 G G + C1 G + C2 = 0        and        (C0 G + C1) H + D = 0.
Differentiating those two equations gives linear (Sylvester-type) systems for dG and dH, so no
derivative of the QZ decomposition is needed. Derivatives of the coefficient matrices with respect
to the parameters are central differences of the parsed matrices; the likelihood derivative with
respect to (G, H, shock sd, observable means) comes from JAX.
"""

from __future__ import annotations

import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from real_sw_estimate import SW

from dynecon.estimation.kalman_jax import kalman_loglik_jax

warnings.filterwarnings("ignore", category=RuntimeWarning)


def system(sw: SW, x):
    """Coefficient matrices, constants and shock sds at parameter vector ``x``."""
    ns, sd = sw.params(x)
    C0, C1, C2, D, const = sw.model.matrices(ns)
    return C0, C1, C2, D, const, np.array([sd[e] for e in sw.model.shocks])


def solve_GH(C0, C1, C2, D, n, m):
    """Stable solution (G, H) via the Klein solver."""
    from dynecon.solvers.perturbation.klein import klein_solve

    N = 2 * n + m
    A, B = np.zeros((N, N)), np.zeros((N, N))
    A[:n, :n] = np.eye(n)
    B[:n, n + m :] = np.eye(n)
    A[n : n + m, n : n + m] = np.eye(m)
    A[n + m :, n + m :] = C0
    B[n + m :, n + m :] = -C1
    B[n + m :, :n] = -C2
    B[n + m :, n : n + m] = -D
    F, _, _ = klein_solve(A, B, n + m)
    return F[:, :n], F[:, n:]


def sylvester_batch(K, G, W):
    """Solve X + K X G = W for a stack of right-hand sides W (p, n, n) by Schur on G."""
    T, U = sla.schur(G, output="complex")
    Wt = W @ U
    X = np.zeros_like(Wt)
    n = G.shape[0]
    for j in range(n):
        rhs = Wt[:, :, j].copy()
        if j:
            rhs -= (K @ (X[:, :, :j] @ T[:j, j])[..., None])[..., 0]
        X[:, :, j] = np.linalg.solve(np.eye(n) + T[j, j] * K, rhs.T).T
    return (X @ U.conj().T).real


def loglik_and_grad(sw: SW, x, h=1e-6):
    model = sw.model
    n, m = len(model.variables), len(model.shocks)
    C0, C1, C2, D, const, sd = system(sw, x)
    G, H = solve_GH(C0, C1, C2, D, n, m)
    p = len(x)
    # derivatives of the coefficient matrices (central differences, relative step)
    dm = []
    for i in range(p):
        step = h * max(1.0, abs(x[i]))
        up, dn = np.array(x, float), np.array(x, float)
        up[i] += step
        dn[i] -= step
        a, b = system(sw, up), system(sw, dn)
        dm.append([(u - d) / (2 * step) for u, d in zip(a, b)])
    dC0, dC1, dC2, dD, dconst, dsd = (np.stack([d[k] for d in dm]) for k in range(6))
    M = C0 @ G + C1
    Minv = np.linalg.inv(M)
    S = dC0 @ G @ G + dC1 @ G + dC2  # (p, n, n)
    dG = sylvester_batch(Minv @ C0, G, -(Minv @ S))
    dH = -(Minv @ ((dC0 @ G + C0 @ dG + dC1) @ H + dD))
    # likelihood and its partials in (G, H, sd, d)
    rows = [model.variables.index(o) for o in sw.obs_names]
    lhs = [e.split("=")[0].strip() for e in model.equations]
    eq_of = [lhs.index(o) for o in sw.obs_names]
    lagged = (np.abs(C2).sum(axis=0) > 0).astype(float)
    if not hasattr(sw, "_vg"):
        y = jnp.asarray(sw.data)
        rows_j, eq_j = jnp.asarray(rows), jnp.asarray(eq_of)

        def ll(G_, H_, sd_, const_, lagged_):
            d_ = -const_[eq_j]
            F = jnp.concatenate([G_, H_], axis=1)  # y_t = F [y_{t-1}; e_t]
            Z = F[rows_j]
            Tm = jnp.zeros((n + m, n + m)).at[:n].set(F)
            R = jnp.zeros((n + m, m)).at[n:].set(jnp.diag(sd_))
            Q = R @ R.T
            P0 = jnp.diag(jnp.concatenate([10.0 * lagged_, sd_**2]))
            a0, Hm = jnp.zeros(n + m), jnp.zeros((7, 7))
            full = kalman_loglik_jax(y - d_, Z, Hm, Tm, Q, a0, P0)
            return full - kalman_loglik_jax((y - d_)[:4], Z, Hm, Tm, Q, a0, P0)

        sw._vg = jax.jit(jax.value_and_grad(ll, argnums=(0, 1, 2, 3)))
    val, (gG, gH, gsd, gc) = sw._vg(
        jnp.asarray(G), jnp.asarray(H), jnp.asarray(sd), jnp.asarray(const), jnp.asarray(lagged)
    )
    grad = (
        np.einsum("ij,pij->p", np.asarray(gG), dG)
        + np.einsum("ij,pij->p", np.asarray(gH), dH)
        + dsd @ np.asarray(gsd)
        + dconst @ np.asarray(gc)
    )
    return float(val), grad


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
    print(
        f"finite differences {t_fd * 1000:.0f} ms;  max relative error {err.max():.2e} at {err.argmax()}"
    )
    print("speedup", round(t_fd / t_exact, 1))


if __name__ == "__main__":
    main(sys.argv[1])
