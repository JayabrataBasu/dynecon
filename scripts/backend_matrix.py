"""Print the operation x backend implementation matrix from the dispatcher registry."""

from __future__ import annotations

import dynecon  # noqa: F401  (importing registers every operation)
from dynecon.numerics._registry import list_registered

BACKENDS = ["numpy", "numba", "jax"]


def main() -> None:
    reg = list_registered()
    ops = sorted({k[0] for k in reg})
    print("| operation | " + " | ".join(BACKENDS) + " |")
    print("|---|" + "---|" * len(BACKENDS))
    for op in ops:
        cells = ["yes" if (op, b) in reg else "-" for b in BACKENDS]
        print(f"| {op} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
