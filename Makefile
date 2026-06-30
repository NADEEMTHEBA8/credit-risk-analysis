# ============================================================================
# Credit Risk Pipeline — Makefile
# ============================================================================

.PHONY: help setup extract transform train load test test-all lint clean docker-up docker-down

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
	$(PIP) install dbt-postgres

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

# ---------------------------------------------------------------------------
# PIPELINE (DAG SIMULATION)
# ---------------------------------------------------------------------------

extract: ## Run data extraction (load raw CSVs)
	$(PY) -m src.main --step extract

transform: ## Run aggregations, merging, and feature engineering
	$(PY) -m src.main --step transform

train: ## Train machine learning models
	$(PY) -m src.main --step train

# ---------------------------------------------------------------------------
# TESTING & LINTING
# ---------------------------------------------------------------------------

test: ## Run Python pytest suite
	$(VENV)/bin/pytest tests/ -v

test-all: test ## Run pytest suite and dbt tests
	cd dbt && ../$(VENV)/bin/dbt test

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
# DOCKER & DATABASE
# ---------------------------------------------------------------------------

docker-up: ## Start PostgreSQL container
	docker compose up -d postgres

docker-down: ## Stop containers
	docker compose down -v

load: ## Load processed CSV into PostgreSQL for dbt
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -c "DROP TABLE IF EXISTS credit_data CASCADE; CREATE TABLE credit_data (amt_income_total NUMERIC, amt_credit NUMERIC, amt_goods_price NUMERIC, days_birth INTEGER, days_employed NUMERIC, income_credit_ratio NUMERIC, employment_age_ratio NUMERIC, annuity_income_ratio NUMERIC, bur_total_debt NUMERIC, bur_num_credits NUMERIC, bur_max_overdue NUMERIC, prev_num_applications NUMERIC, prev_approval_rate NUMERIC, inst_late_rate NUMERIC, inst_days_late_mean NUMERIC, cc_utilisation NUMERIC, cc_dpd_max NUMERIC, pos_sk_dpd_max NUMERIC, pos_completion_rate NUMERIC, target NUMERIC);"
	psql -h $(DB_HOST) -U $(DB_USER) -d $(DB_NAME) -c "\COPY credit_data FROM 'data/processed/credit_data_sql.csv' WITH (FORMAT csv, HEADER true);"

# ============================================================================
# END
# ============================================================================