# ==============================================================================
# Automated CCTV Accident Detection System (ACS)
# Production Automation Makefile
# ==============================================================================

SHELL := /bin/bash
PYTHON ?= ./venv/bin/python
PIP ?= ./venv/bin/pip
PYTEST ?= ./venv/bin/pytest
NPM ?= npm

.PHONY: help setup install data audit train eval tune promote test test-unit test-e2e clean

# Default Target
help:
	@echo "================================================================================"
	@echo " Automated CCTV Accident Detection System (ACS) - Makefile"
	@echo "================================================================================"
	@echo "  make setup       - Create Python venv and install backend & frontend dependencies"
	@echo "  make data        - Fetch supplemental datasets and build merged split dataset"
	@echo "  make audit       - Audit dataset integrity, bounding boxes, and leakage"
	@echo "  make train       - Train model using the 2-phase fine-tuning recipe (Exp A)"
	@echo "  make eval        - Evaluate baseline & candidate models on TEST split + holdout video"
	@echo "  make tune        - Run threshold, debounce, and overlap grid-search on tuning suite"
	@echo "  make promote     - Execute model promotion gate check against baseline"
	@echo "  make test        - Run full test suite (pytest unit tests + Playwright smoke test)"
	@echo "  make test-unit   - Run all pytest unit tests (backend, detector, event manager)"
	@echo "  make test-e2e    - Run frontend Playwright end-to-end smoke test"
	@echo "  make clean       - Remove cache directories, build artifacts, and temp files"
	@echo "================================================================================"

setup:
	@echo "Setting up Python virtual environment..."
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setting up frontend dependencies & Playwright..."
	cd frontend && $(NPM) install && npx playwright install chromium

install: setup

data:
	@echo "Running dataset acquisition and synthesis pipeline..."
	$(PYTHON) ml/scripts/fetch_supplemental_data.py
	$(PYTHON) ml/scripts/inject_hard_negatives.py
	$(PYTHON) ml/scripts/build_dataset.py
	$(PYTHON) ml/scripts/audit_dataset.py

audit:
	@echo "Auditing merged dataset..."
	$(PYTHON) ml/scripts/audit_dataset.py

train:
	@echo "Running two-phase fine-tuning training recipe..."
	$(PYTHON) ml/scripts/train.py --experiment exp_a

eval:
	@echo "Running model benchmark on test split and holdout video..."
	$(PYTHON) ml/scripts/evaluate_all.py --split test --video acc_ex.mp4

tune:
	@echo "Running threshold and event manager grid-search tuning..."
	$(PYTHON) ml/scripts/tune_thresholds.py --target-fa-per-hour 0.25

promote:
	@echo "Running automated promotion gate pipeline..."
	$(PYTHON) ml/scripts/promote.py

test: test-unit test-e2e

test-unit:
	@echo "Running pytest backend unit test suite..."
	$(PYTEST) tests/
	@echo "Running Vitest frontend unit test suite..."
	cd frontend && $(NPM) test

test-e2e:
	@echo "Running Playwright frontend smoke test..."
	cd frontend && $(NPM) run test:e2e

clean:
	@echo "Cleaning temporary files and build artifacts..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache frontend/test-results frontend/playwright-report
	rm -rf temp/*
