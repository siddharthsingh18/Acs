# ==============================================================================
# Automated CCTV Accident Detection System (ACS)
# Research Prototype Automation Makefile
# ==============================================================================

SHELL := /bin/bash
PYTHON ?= ./venv/bin/python
PIP ?= ./venv/bin/pip
PYTEST ?= ./venv/bin/pytest
NPM ?= npm
DUPLICATE_ENCODER ?= ml/models/clip_dinov2_encoder.pt

.PHONY: help setup install data audit train eval eval-verification tune promote test test-unit test-e2e clean

# Default Target
help:
	@echo "================================================================================"
	@echo " Automated CCTV Accident Detection System (ACS) - Makefile"
	@echo "================================================================================"
	@echo "  make setup       - Create Python venv and install backend & frontend dependencies"
	@echo "  make data        - Fetch supplemental datasets and build merged split dataset"
	@echo "  make audit       - Audit dataset integrity, bounding boxes, and leakage"
	@echo "  make train       - Train model using the 2-phase fine-tuning recipe (Exp A)"
	@echo "  make eval        - Generate provenance-stamped same-run evaluation reports"
	@echo "  make eval-verification - Evaluate incident-level verifier (YOLO vs Verified)"
	@echo "  make tune        - Run the shared evaluation pipeline (blocked without tuning clips)"
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
	@echo "Fetching only license-cleared, checksum-pinned datasets and rebuilding derived data..."
	$(PYTHON) ml/scripts/fetch_data.py
	$(PYTHON) ml/scripts/rebuild_dataset.py --embedding-model "$(DUPLICATE_ENCODER)"
	$(PYTHON) ml/scripts/review_labels.py

audit:
	@echo "Auditing merged dataset..."
	$(PYTHON) ml/scripts/audit_dataset.py

train:
	@echo "Running two-phase fine-tuning training recipe..."
	$(PYTHON) ml/scripts/train.py --experiment exp_a

eval:
	@echo "Generating same-run reports; this fails closed when data/evaluation evidence is missing..."
	$(PYTHON) -m ml.eval.run_all --model best.pt --data-manifest ml/eval/data_manifest.yaml

eval-verification:
	@echo "Running incident-level verification stage evaluation..."
	$(PYTHON) ml/scripts/eval_verification.py

tune:
	@echo "Threshold tuning is owned by the shared evaluation run; no frozen tuning clips are available yet..."
	$(PYTHON) -m ml.eval.run_all --model best.pt --data-manifest ml/eval/data_manifest.yaml

promote:
	@echo "Running automated promotion gate pipeline..."
	$(PYTHON) ml/scripts/promote.py

export-feedback:
	@echo "Exporting operator review feedback to YOLO staging dataset..."
	$(PYTHON) ml/scripts/export_feedback.py

retrain:
	@echo "Executing monthly retraining workflow with staging merge and frozen test protection..."
	$(PYTHON) ml/scripts/retrain_workflow.py

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
