"""
Export — write final outputs and optionally log to MLflow.
"""

from __future__ import annotations

import logging

import pandas as pd

try:
    from src.utils import (DECISION_BETA, MISSING_DROP_PCT, PROCESSED,
                           RANDOM_STATE, TEST_SIZE)
except ModuleNotFoundError:
    from utils import (DECISION_BETA, MISSING_DROP_PCT, PROCESSED,
                        RANDOM_STATE, TEST_SIZE)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

log = logging.getLogger(__name__)


# Columns exported for SQL analysis layer.
# Matches the analysis.sql schema. Keep in sync if you add SQL features.
SQL_COLUMNS = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'income_credit_ratio', 'employment_age_ratio', 'annuity_income_ratio',
    'bur_total_debt', 'bur_num_credits', 'bur_max_overdue',
    'prev_num_applications', 'prev_approval_rate',
    'inst_late_rate', 'inst_days_late_mean',
    'cc_utilisation', 'cc_dpd_max',
    'pos_sk_dpd_max', 'pos_completion_rate', 'TARGET',
]


def export_csvs(train_df: pd.DataFrame) -> None:
    """Write the SQL-facing slim CSV and the full enriched CSV."""
    sql_out = [c for c in SQL_COLUMNS if c in train_df.columns]
    train_df[sql_out].to_csv(f'{PROCESSED}/credit_data_sql.csv', index=False)
    train_df.to_csv(f'{PROCESSED}/final_enriched_train.csv', index=False)

    log.info(f"  Saved: credit_data_sql.csv ({len(sql_out)} columns, "
             f"{len(train_df):,} rows)")
    log.info(f"  Saved: final_enriched_train.csv ({train_df.shape[1]} columns)")


def log_to_mlflow(results: dict, opt_thr: float) -> None:
    """Log each model's metrics to MLflow if available. No-op otherwise."""
    if not MLFLOW_AVAILABLE:
        return
    mlflow.set_experiment("home_credit_credit_scoring")
    for name, res in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                'model'             : name,
                'test_size'         : TEST_SIZE,
                'random_state'      : RANDOM_STATE,
                'missing_drop_pct'  : MISSING_DROP_PCT,
                'decision_beta'     : DECISION_BETA,
                'decision_threshold': opt_thr,
            })
            mlflow.log_metrics({k: v for k, v in res.items() if k != 'y_proba'})
    log.info("  MLflow logged — view with: mlflow ui")
