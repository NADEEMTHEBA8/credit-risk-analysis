"""
Export — write final outputs, persist the fitted model, and optionally log
to MLflow.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import joblib
import pandas as pd

from src.utils import (DECISION_BETA, MISSING_DROP_PCT, PROCESSED,
                       RANDOM_STATE, TEST_SIZE)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

log = logging.getLogger(__name__)

# Directory where serialised model artefacts are written.
MODELS_DIR = 'models'

# Columns exported for the SQL analysis layer.
# Matches the analysis.sql schema — if you add a feature here, add the
# column to the CREATE TABLE statement in sql/analysis.sql as well.
# A missing column raises KeyError immediately rather than silently
# writing a short CSV that breaks downstream SQL imports.
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
    """Write the SQL-facing slim CSV and the full enriched CSV.

    Raises:
        KeyError: If a column in SQL_COLUMNS is absent from train_df. This
            catches upstream renames or dropped features before they silently
            produce a shorter CSV that breaks the SQL import.
    """
    missing_cols = [c for c in SQL_COLUMNS if c not in train_df.columns]
    if missing_cols:
        raise KeyError(
            f"SQL export schema mismatch — columns absent from feature matrix: "
            f"{missing_cols}. Update SQL_COLUMNS in export.py or fix upstream "
            f"feature engineering."
        )
    train_df[SQL_COLUMNS].to_csv(f'{PROCESSED}/credit_data_sql.csv', index=False)
    train_df.to_csv(f'{PROCESSED}/final_enriched_train.csv', index=False)

    log.info(f"  Saved: credit_data_sql.csv ({len(SQL_COLUMNS)} columns, "
             f"{len(train_df):,} rows)")
    log.info(f"  Saved: final_enriched_train.csv ({train_df.shape[1]} columns)")


def save_model(model, model_name: str) -> str:
    """Serialise a fitted sklearn Pipeline to disk with a timestamp.

    Args:
        model:      Fitted sklearn Pipeline object.
        model_name: Human-readable name used in the filename (e.g. 'XGBoost').

    Returns:
        Path to the written artefact.

    Example output path:
        models/XGBoost_20260627_235959.joblib
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_name = model_name.replace(' ', '_')
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    artefact_path = os.path.join(MODELS_DIR, f'{safe_name}_{timestamp}.joblib')
    joblib.dump(model, artefact_path)
    log.info(f"  Model saved: {artefact_path}")
    return artefact_path


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
