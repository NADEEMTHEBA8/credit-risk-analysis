"""
Pipeline orchestrator for the Home Credit default-risk project.

Runs all stages end to end. Entry point: python -m src.main

Stage checkpointing
-------------------
Aggregated secondary tables are cached as Parquet files in data/processed/.
On a restart after a mid-pipeline failure, the pipeline skips already-
completed aggregation stages and picks up where it left off.

Pass --fresh on the command line to bypass the cache and re-run everything
from scratch:

    python -m src.main --fresh

To run specific steps (simulating a DAG):
    python -m src.main --step extract
    python -m src.main --step transform
    python -m src.main --step train
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

from src.utils import FIGURES, PROCESSED, RAW_DIR, load_csv, validate_inputs
from src.aggregate import bureau as agg_bureau
from src.aggregate import previous as agg_previous
from src.aggregate import pos_cash as agg_pos
from src.aggregate import credit_card as agg_cc
from src.aggregate import installments as agg_inst
from src.features import engineer as engineer_features
from src.preprocess import encode, impute_and_cap, split
from src.eda import run as run_eda
from src.train import (cross_validate, evaluate, plot_confusion, plot_curves,
                       plot_importance, plot_lift, train_models)
from src.threshold import select_threshold
from src.export import export_csvs, log_to_mlflow, save_model

# Suppress only the specific known-harmless warnings; leave everything else
# visible so genuine deprecation signals are not swallowed in production logs.
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Aggregate cache paths ────────────────────────────────────────────────────
# Each secondary table is written here after aggregation and loaded on restart.
_CACHE = {
    'bureau'    : os.path.join(PROCESSED, 'agg_bureau.parquet'),
    'previous'  : os.path.join(PROCESSED, 'agg_previous.parquet'),
    'pos'       : os.path.join(PROCESSED, 'agg_pos.parquet'),
    'cc'        : os.path.join(PROCESSED, 'agg_cc.parquet'),
    'inst'      : os.path.join(PROCESSED, 'agg_inst.parquet'),
    'app_raw'   : os.path.join(PROCESSED, 'app_raw.parquet'),
    'transformed': os.path.join(PROCESSED, 'app_transformed.parquet'),
}


def _load_or_run(key: str, run_fn, fresh: bool) -> pd.DataFrame:
    """Return cached aggregate Parquet if it exists; otherwise run run_fn.

    Args:
        key:    Cache key matching a path in _CACHE.
        run_fn: Callable that returns the aggregate DataFrame.
        fresh:  If True, ignore the cache and always re-run.

    Returns:
        The aggregate DataFrame keyed on SK_ID_CURR.
    """
    cache_path = _CACHE[key]
    if not fresh and os.path.exists(cache_path):
        log.info(f"  [{key}] Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    df = run_fn()
    df.to_parquet(cache_path, index=False)
    log.info(f"  [{key}] Cached to: {cache_path}")
    return df


def step_extract() -> None:
    """Extract raw data into a consolidated application table."""
    t0 = time.perf_counter()
    log.info("\n[1/3] EXTRACT: Loading application tables...")
    validate_inputs()
    train_raw = load_csv('application_train.csv')
    test_raw  = load_csv('application_test.csv')
    train_raw['SET']   = 'train'
    test_raw['SET']    = 'test'
    test_raw['TARGET'] = np.nan
    app = pd.concat([train_raw, test_raw], ignore_index=True, sort=False)
    
    default_rate = app[app['SET'] == 'train']['TARGET'].mean() * 100
    log.info(f"  Train: {(app['SET']=='train').sum():,}  |  "
             f"Test: {(app['SET']=='test').sum():,}  |  "
             f"Default rate: {default_rate:.2f}%")
    
    app.to_parquet(_CACHE['app_raw'], index=False)
    log.info(f"  Saved raw extract to {_CACHE['app_raw']}")
    log.info(f"  Done in {time.perf_counter() - t0:.1f}s")


def step_transform(fresh: bool) -> None:
    """Transform extracted data: aggregations, merging, and feature engineering."""
    t0 = time.perf_counter()
    log.info("\n[2/3] TRANSFORM: Aggregating and engineering features...")
    
    if not os.path.exists(_CACHE['app_raw']):
        raise FileNotFoundError(f"Missing {_CACHE['app_raw']}. Run --step extract first.")
    
    app = pd.read_parquet(_CACHE['app_raw'])
    app_columns = list(app.columns)
    n_app_rows  = len(app)

    bureau_agg = _load_or_run('bureau',   agg_bureau.run,   fresh)
    prev_agg   = _load_or_run('previous', agg_previous.run, fresh)
    pos_agg    = _load_or_run('pos',      agg_pos.run,      fresh)
    cc_agg     = _load_or_run('cc',       agg_cc.run,       fresh)
    inst_agg   = _load_or_run('inst',     agg_inst.run,     fresh)

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

    if len(df) != n_app_rows:
        raise RuntimeError(
            f"Merge exploded rows: expected {n_app_rows:,}, got {len(df):,}. "
            "A secondary table has duplicate SK_ID_CURR values — inspect the "
            "aggregate outputs in data/processed/agg_*.parquet."
        )

    df = engineer_features(df, app_columns)
    df, SET_col, df_ids = encode(df)
    df = impute_and_cap(df)
    
    df['SET'] = SET_col.values
    df['SK_ID_CURR'] = df_ids.values
    
    df.to_parquet(_CACHE['transformed'], index=False)
    log.info(f"  Saved transformed features to {_CACHE['transformed']}")
    
    # Export for SQL Analytics Layer
    log.info("\nExporting SQL analytics dataset...")
    train_df, _, _, _, _, _, _ = split(df, SET_col, df_ids)
    export_csvs(train_df)
    log.info(f"  Done in {time.perf_counter() - t0:.1f}s")


def step_train() -> None:
    """Train models on the transformed data."""
    t0 = time.perf_counter()
    log.info("\n[3/3] TRAIN: Splitting and training...")
    
    if not os.path.exists(_CACHE['transformed']):
        raise FileNotFoundError(f"Missing {_CACHE['transformed']}. Run --step transform first.")
        
    df = pd.read_parquet(_CACHE['transformed'])
    SET_col = df['SET'].copy() if 'SET' in df.columns else None
    
    # Needs to recreate df_ids since it was encoded and we didn't save it separately
    # The split function handles it based on SET
    # Wait, encode modifies df inline and returns SET_col, df_ids. Let's just pass SET.
    # We can reconstruct df_ids by keeping SK_ID_CURR
    if 'SK_ID_CURR' in df.columns:
        df_ids = df['SK_ID_CURR'].copy()
    else:
        df_ids = pd.Series(index=df.index, dtype=int)
        
    train_df, test_df, X_train, X_val, y_train, y_val, cw_dict = split(df, SET_col, df_ids)
    run_eda(train_df)

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

    opt_thr = select_threshold(y_val, y_proba_best)
    cross_validate(best_model, best_name, X_train, y_train)
    log_to_mlflow(results, opt_thr)
    save_model(best_model, best_name)

    log.info("\nPipeline complete.")
    log.info(f"  Best model        : {best_name}")
    log.info(f"  AUC-ROC           : {results[best_name]['AUC_ROC']}")
    log.info(f"  Decision threshold: {opt_thr} (F-beta beta=2.5)")
    log.info(f"  Business lift     : {lift:.1f}% of defaults in top 20% "
             f"({lift / 20:.1f}x random)")
    log.info(f"  Done in {time.perf_counter() - t0:.1f}s")


def run(fresh: bool = False, step: str = 'all') -> None:
    """Execute the pipeline stages.

    Args:
        fresh: When True, ignore Parquet caches and re-run all aggregations.
        step: Specific step to run ('extract', 'transform', 'train', 'all').
    """
    for d in [RAW_DIR, PROCESSED, FIGURES]:
        os.makedirs(d, exist_ok=True)

    log.info("Home Credit default-risk pipeline")
    if fresh:
        log.info("  --fresh: all aggregate caches will be regenerated")

    if step in ['all', 'extract']:
        step_extract()
    if step in ['all', 'transform']:
        step_transform(fresh)
    if step in ['all', 'train']:
        step_train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Home Credit default-risk pipeline"
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help="Ignore Parquet caches and re-run all aggregation stages.",
    )
    parser.add_argument(
        '--step',
        choices=['all', 'extract', 'transform', 'train'],
        default='all',
        help="Specific pipeline step to run (default: all)",
    )
    args = parser.parse_args()
    run(fresh=args.fresh, step=args.step)
