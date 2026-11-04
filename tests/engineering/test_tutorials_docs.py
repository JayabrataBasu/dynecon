"""Tutorials execute (their final cells assert key numbers); generated docs are consistent."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = sorted((ROOT / "docs" / "tutorials").glob("[0-9][0-9]_*.md"))


def test_ten_tutorials_exist() -> None:
    assert len(TUTORIALS) == 10


@pytest.mark.slow
@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.name)
def test_tutorial_runs(path: Path) -> None:
    if "import jax" in path.read_text(encoding="utf-8"):
        pytest.importorskip("jax.numpy")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_tutorials.py"), path.name[:2]],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_docstring_audit_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_docstrings.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout


def test_model_docs_generate(tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_model_docs.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    index = (ROOT / "docs" / "models" / "index.md").read_text(encoding="utf-8")
    for name in ("aiyagari", "rbc", "lucas_tree"):
        assert name in index and (ROOT / "docs" / "models" / f"{name}.md").exists()
