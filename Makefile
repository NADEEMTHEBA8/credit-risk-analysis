# ============================================================================
# Credit Risk Pipeline — Makefile
# ============================================================================

.PHONY: help setup run test lint clean docker-up docker-down \
        sql-init sql-load sql-run sql-reset

PYTHON := python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

DB_NAME := credit_risk
DB_USER := postgres
DB_HOST := localhost

# ---------------------------------------------------------------------------
# HELP
# ---------------------------------------------------------------------------

help: ## Show available commands
	@echo "Credit Risk Pipeline — available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; \
		{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# PYTHON ENVIRONMENT
# ---------------------------------------------------------------------------

setup: $(VENV)/bin/activate ## Create virtual environment

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

run: ## Run full ML pipeline
	$(PY) -m src.main

test: ## Run pytest suite
	$(VENV)/bin/pytest tests/ -v

lint: ## Run ruff lint checks
	$(VENV)/bin/ruff check src tests

# ---------------------------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------------------------

clean: ## Remove generated caches
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete

# ---------------------------------------------------------------------------
# DOCKER
# ---------------------------------------------------------------------------

docker-up: ## Start PostgreSQL container
	docker compose up -d postgres

docker-down: ## Stop containers
	docker compose down -v

# ---------------------------------------------------------------------------
# SQL WORKFLOW
# ---------------------------------------------------------------------------

sql-init: ## Create SQL schema only
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) \
	-c "$$(sed -n '/^DROP TABLE/,/^);/p' sql/analysis.sql)"

sql-load: ## Load processed analytical CSV
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) \
	-c "\COPY credit_data FROM 'data/processed/credit_data_sql.csv' WITH (FORMAT csv, HEADER true);"

sql-run: ## Run SQL analytics sections only
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) \
	-c "$$(sed -n '/^-- 3. DATA QUALITY/,$$p' sql/analysis.sql)"

sql-reset: sql-init sql-load sql-run ## Full SQL workflow

# ============================================================================
# END
# ============================================================================