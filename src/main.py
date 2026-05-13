"""
Home Credit Default Risk — pipeline orchestrator.

This is the new entry point. Each stage is a single function call into
its dedicated module. To run:

    python -m src.main      (preferred)
    python -m src.pipeline  (backward-compatible shim)

Top features from actual EDA:
  EXT_SOURCE_3, EXT_SOURCE_2, EXT_SOURCE_1 dominate — external bureau scores.
  Among behavioural features: age, cc_utilisation, inst_late_rate are strongest.
  income_credit_ratio ranks low despite being the expected primary signal.

Experiments tried and rejected (see commit history for details):
  - GradientBoostingClassifier — recall 0.0449, 5x slower than LightGBM.
  - INNER JOIN on secondary tables — dropped 37K first-time borrowers,
    recall fell 0.69 → 0.61. Switched to LEFT JOIN + median imputation.
  - income_credit_ratio as primary SQL segment — corr 0.0018 (17th of 19).
    SQL rebuilt around age + employment_age_ratio instead.
  - F1-based threshold — wrong cost model for credit decisions. F-beta β=2.5.
  - SMOTE oversampling — improved train AUC, degraded val by 0.008. Discarded.
"""

from __future__ import annotations

import gc
import logging
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

try:
    from src.utils import (FIGURES, PROCESSED, RAW_DIR,
                           load_csv, validate_inputs)
    from src.aggregate import bureau as agg_bureau
    from src.aggregate import previous as agg_previous
    from src.aggregate import pos_cash as agg_pos
    from src.aggregate import credit_card as agg_cc
    from src.aggregate import installments as agg_inst
    from src.features    import engineer as engineer_features
    from src.preprocess  import encode, impute_and_cap, split
    from src.eda         import run as run_eda
    from src.train       import (cross_validate, evaluate, plot_confusion,
                                  plot_curves, plot_importance, plot_lift,
                                  train_models)
    from src.threshold   import select_threshold
    from src.export      import export_csvs, log_to_mlflow
except ModuleNotFoundError:
    from utils import (FIGURES, PROCESSED, RAW_DIR, load_csv, validate_inputs)
    from aggregate import bureau as agg_bureau
    from aggregate import previous as agg_previous
    from aggregate import pos_cash as agg_pos
    from aggregate import credit_card as agg_cc
    from aggregate import installments as agg_inst
    from features    import engineer as engineer_features
    from preprocess  import encode, impute_and_cap, split
    from eda         import run as run_eda
    from train       import (cross_validate, evaluate, plot_confusion,
                              plot_curves, plot_importance, plot_lift,
                              train_models)
    from threshold   import select_threshold
    from export      import export_csvs, log_to_mlflow

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def run() -> None:
    """Execute the full pipeline end-to-end."""
    for d in [RAW_DIR, PROCESSED, FIGURES]:
        os.makedirs(d, exist_ok=True)

    log.info("=" * 65)
    log.info("  HOME CREDIT DEFAULT RISK — MULTI-TABLE PIPELINE")
    log.info("=" * 65)

    # ── 1. INGEST ────────────────────────────────────────────────────────
    log.info("\n[1/8] Loading application tables...")
    validate_inputs()
    train_raw = load_csv('application_train.csv')
    test_raw  = load_csv('application_test.csv')
    train_raw['SET']   = 'train'
    test_raw['SET']    = 'test'
    test_raw['TARGET'] = np.nan
    app = pd.concat([train_raw, test_raw], ignore_index=True, sort=False)
    app_columns = list(app.columns)

    default_rate = app[app['SET'] == 'train']['TARGET'].mean() * 100
    log.info(f"  Train: {(app['SET']=='train').sum():,}  |  "
             f"Test: {(app['SET']=='test').sum():,}  |  "
             f"Default rate: {default_rate:.2f}%")

    # ── 2. AGGREGATE ─────────────────────────────────────────────────────
    log.info("\n[2/8] Aggregating secondary tables...")
    bureau_agg = agg_bureau.run()
    prev_agg   = agg_previous.run()
    pos_agg    = agg_pos.run()
    cc_agg     = agg_cc.run()
    inst_agg   = agg_inst.run()

    # ── 3. MERGE ─────────────────────────────────────────────────────────
    log.info("\n[3/8] Merging all tables...")
    df = app.copy()
    for agg_df, name in [
        (bureau_agg,  'bureau'),
        (prev_agg,    'previous_application'),
        (pos_agg,     'POS_cash'),
        (cc_agg,      'credit_card'),
        (inst_agg,    'installments'),
    ]:
        df = df.merge(agg_df, on='SK_ID_CURR', how='left')
        log.info(f"  After merging {name:<25}: {df.shape}")
    del bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg
    gc.collect()
    log.info(f"  Final merged shape: {df.shape}")

    # ── 4. FEATURE ENGINEERING ───────────────────────────────────────────
    log.info("\n[4/8] Engineering features...")
    df = engineer_features(df, app_columns)

    # ── 5. ENCODE ────────────────────────────────────────────────────────
    log.info("\n[5/8] Encoding...")
    df, SET_col, df_ids = encode(df)

    # ── 6. IMPUTE + CAP + SPLIT ──────────────────────────────────────────
    log.info("\n[6/8] Imputing and capping...")
    df = impute_and_cap(df)

    log.info("\n[7/8] Splitting and training...")
    train_df, test_df, X_train, X_val, y_train, y_val, cw_dict = split(df, SET_col, df_ids)

    # ── EDA ──────────────────────────────────────────────────────────────
    run_eda(train_df)

    # ── 7. TRAIN + EVALUATE ──────────────────────────────────────────────
    models  = train_models(X_train, y_train)
    log.info("\n  Evaluating models...")
    results = evaluate(models, X_val, y_val)

    best_name  = max(results, key=lambda k: results[k]['AUC_ROC'])
    best_model = models[best_name]
    log.info(f"\n  Best model: {best_name} (AUC={results[best_name]['AUC_ROC']})")

    plot_curves(results, y_val)
    plot_importance(best_model, best_name, X_train.columns.tolist())
    y_proba_best = plot_confusion(best_model, best_name, X_val, y_val)
    lift = plot_lift(X_val, y_val, y_proba_best)

    # ── THRESHOLD + CV ───────────────────────────────────────────────────
    opt_thr = select_threshold(y_val, y_proba_best)
    cross_validate(best_model, best_name, X_train, y_train)

    # ── MLflow ───────────────────────────────────────────────────────────
    log_to_mlflow(results, opt_thr)

    # ── 8. EXPORT ────────────────────────────────────────────────────────
    log.info("\n[8/8] Exporting...")
    export_csvs(train_df)

    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Best model        : {best_name}")
    log.info(f"  AUC-ROC           : {results[best_name]['AUC_ROC']}")
    log.info(f"  Decision threshold: {opt_thr} (F-beta β=2.5)")
    log.info(f"  Business lift     : {lift:.1f}% of defaults in top 20% "
             f"({lift / 20:.1f}x random)")
    log.info("=" * 65)


if __name__ == "__main__":
    run()
