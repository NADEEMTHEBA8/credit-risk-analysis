# ============================================================================
# Credit Risk Pipeline — Makefile
# ============================================================================
# Common dev commands. Run `make help` to see all targets.
# ============================================================================

.PHONY: help setup run test lint clean docker-up docker-down sql-load sql-run

PYTHON := python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

help:  ## Show this help message
	@echo "Credit Risk Pipeline — available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/bin/activate  ## Create venv and install dependencies

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

run:  ## Run the full pipeline end-to-end
	$(PY) -m src.pipeline

test:  ## Run pytest test suite
	$(VENV)/bin/pytest tests/ -v

lint:  ## Run ruff linter (if installed)
	$(VENV)/bin/ruff check src/ tests/ dags/ 2>/dev/null || echo "ruff not installed - skipping"

clean:  ## Remove generated files and caches
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete

docker-up:  ## Start Postgres container
	docker compose up -d postgres

docker-down:  ## Stop containers
	docker compose down

sql-load:  ## Load processed CSV into Postgres
	psql -h localhost -U postgres -d credit_risk \
		-c "\COPY application FROM 'data/processed/credit_data_sql.csv' WITH (FORMAT csv, HEADER true);"

sql-run:  ## Run the full SQL analysis script
	psql -h localhost -U postgres -d credit_risk -f sql/analysis.sql
