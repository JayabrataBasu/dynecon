"""Pytest configuration: make tests/builders.py importable; golden-file update switch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite tests/golden/*.npz (needs a CHANGELOG line and a reason in the log)",
    )
