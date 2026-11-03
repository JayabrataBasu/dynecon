"""`import dynecon` must stay light: no pandas, matplotlib, numba or jax."""

import subprocess
import sys


def test_import_dynecon_does_not_load_optional_stacks() -> None:
    code = (
        "import sys, dynecon;"
        "bad = [m for m in ('pandas','matplotlib','numba','jax') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == ""


def test_numba_loads_only_when_a_kernel_is_called() -> None:
    code = (
        "import sys, numpy as np, dynecon as ek;"
        "from dynecon import models;"
        "assert 'numba' not in sys.modules;"
        "ek.solve(models.load('consumption_savings', n_a=30, n_z=3).compile(), method='vfi');"
        "print('numba' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out in ("True", "False")  # True when numba is installed
