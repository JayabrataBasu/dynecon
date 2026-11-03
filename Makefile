ENV_DIR := .econkit
ifneq ($(wildcard $(ENV_DIR)/Scripts),)
BIN := $(ENV_DIR)/Scripts
else
BIN := $(ENV_DIR)/bin
endif
PYTHON := $(BIN)/python
PIP := $(BIN)/pip
RUFF := $(BIN)/ruff
MYPY := $(BIN)/mypy
PYTEST := $(BIN)/pytest

.PHONY: install test lint typecheck format test-structure deadcode coverage build ci phase docs bench

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
	$(PYTHON) scripts/check_progress.py

deadcode:
	$(BIN)/vulture econkit scripts/vulture_whitelist.py
	$(PYTHON) scripts/check_usage.py
	$(PYTHON) scripts/check_near_future.py

coverage:
	$(PYTEST) --cov --cov-report=term-missing --cov-report=xml

build:
	$(PYTHON) -m build

phase:
	$(PYTHON) scripts/phase.py $(P) > .phase/P$(P).md

docs:
	@echo "docs build arrives in P16"

bench:
	@echo "benchmark harness arrives in P15"

ci: test-structure lint typecheck test deadcode
