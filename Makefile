ENV_DIR := .econkit
PYTHON := $(ENV_DIR)/bin/python
PIP := $(ENV_DIR)/bin/pip
RUFF := $(ENV_DIR)/bin/ruff
MYPY := $(ENV_DIR)/bin/mypy
PYTEST := $(ENV_DIR)/bin/pytest

.PHONY: install test lint typecheck format test-structure ci

install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -e '.[dev]'

test:
	$(PYTEST)

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

typecheck:
	$(MYPY) econkit

test-structure:
	$(PYTHON) scripts/check_test_layout.py

ci: test-structure lint typecheck test
