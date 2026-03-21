"""NumPy reference bounded optimization kernels."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizeResult:
    x: float
    fun: float
    nit: int
    converged: bool
    method: str


def golden_section_max_numpy(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> OptimizeResult:
    """Maximize a scalar function on [a, b] via golden-section search.

    Economics note:
    This routine is used for one-dimensional policy choices where objective
    derivatives are unavailable or too expensive.
    """
    if not a < b:
        raise ValueError("Require a < b")

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0

    x1 = a + invphi2 * (b - a)
    x2 = a + invphi * (b - a)
    f1 = float(func(x1))
    f2 = float(func(x2))

    nit = 0
    for nit in range(1, max_iter + 1):
        if abs(b - a) <= tol:
            break
        if f1 < f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + invphi * (b - a)
            f2 = float(func(x2))
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + invphi2 * (b - a)
            f1 = float(func(x1))

    x_star = x1 if f1 >= f2 else x2
    f_star = f1 if f1 >= f2 else f2
    return OptimizeResult(
        x=float(x_star),
        fun=float(f_star),
        nit=nit,
        converged=(abs(b - a) <= tol),
        method="golden",
    )


def brent_max_numpy(
    func: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> OptimizeResult:
    """Brent-style bounded maximization on [a, b].

    Uses safeguarded parabolic interpolation with golden fallback.
    """
    if not a < b:
        raise ValueError("Require a < b")

    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0

    x = w = v = a + invphi2 * (b - a)
    fx = fw = fv = float(func(x))
    d = e = 0.0

    nit = 0
    for nit in range(1, max_iter + 1):
        mid = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-14
        tol2 = 2.0 * tol1

        if abs(x - mid) <= (tol2 - 0.5 * (b - a)):
            break

        p = q = r = 0.0
        accepted_parabolic = False
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            if q > 0.0 and abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if (mid - x) >= 0.0 else -tol1
                accepted_parabolic = True

        if not accepted_parabolic:
            e = (b - x) if x < mid else (a - x)
            d = invphi2 * e

        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0.0 else -tol1))
        fu = float(func(u))

        if fu >= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu >= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu >= fv or v == x or v == w:
                v, fv = u, fu
        e = d

    return OptimizeResult(
        x=float(x),
        fun=float(fx),
        nit=nit,
        converged=(abs(b - a) <= tol or nit < max_iter),
        method="brent",
    )
