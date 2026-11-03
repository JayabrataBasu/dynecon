"""Sphinx configuration for the EconKit documentation."""

from __future__ import annotations

project = "EconKit"
author = "Jayabrata Basu"
copyright = "2026, Jayabrata Basu"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]
autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
nb_execution_mode = "force"
nb_execution_timeout = 120
nb_execution_raise_on_error = True
nb_custom_formats = {}

html_theme = "furo"
html_title = "EconKit"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

exclude_patterns = [
    "_build",
    "md/CAPTAINS_LOG.md",
    "md/REVIEW_1.md",
    "md/V1_EXECUTION_PLAN.md",
    "md/V1_PROGRESS.md",
    "md/EconKit_Architecture.md",
    "md/economics_explainability.md",
    "**/.ipynb_checkpoints",
]
