"""
Utility functions, configuration constants, and segmentation bin definitions
for the credit risk pipeline.

All pd.cut bin edges and labels for age, employment, and income segmentation
live here so that features.py and eda.py share a single source of truth.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd


# config
RAW_DIR      = 'data/raw'
PROCESSED    = 'data/processed'
FIGURES      = 'figures'
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# drop columns with more than this % missing
MISSING_DROP_PCT = 80

# Threshold selection uses F-beta rather than F1: a missed default costs
# the bank roughly 8x more than a false positive, so recall is weighted
# higher. beta=2.5 reflects that asymmetry.
DECISION_BETA = 2.5

# ── Segmentation bin definitions ──────────────────────────────────────────
# Single source of truth used by both features.py and eda.py.
# Change bin edges here; both modules pick them up automatically.

AGE_BINS   = [0, 30, 40, 50, 60, float('inf')]
AGE_LABELS = ['18-29', '30-39', '40-49', '50-59', '60+']

EMP_BINS   = [-0.001, 0.1, 0.3, 0.6, float('inf')]
EMP_LABELS = ['Unstable', 'Short-term', 'Moderate', 'Stable']

INC_BINS   = [0, 100_000, 200_000, float('inf')]
INC_LABELS = ['Low', 'Medium', 'High']

RISK_BINS   = [-0.001, 0.3, 0.6, float('inf')]
RISK_LABELS = ['High Risk', 'Medium Risk', 'Low Risk']

# ── Domain-appropriate colour palette ─────────────────────────────────────
PALETTE = {
    'risk'    : '#C1292E',  # brick red — high risk
    'safe'    : '#2D6A4F',  # forest green — low risk
    'neutral' : '#4A4E69',  # slate — neutral
    'accent'  : '#22577A',  # deep teal — highlights
    'warm'    : '#9A8C98',  # warm grey — secondary
}

log = logging.getLogger(__name__)


def load_csv(filename: str, usecols: list = None) -> pd.DataFrame:
    """Load a CSV from RAW_DIR and log its shape."""
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path, usecols=usecols)
    log.info(f"  Loaded {filename:<45} shape: {df.shape}")
    return df


def missing_profile(df: pd.DataFrame, label: str = '') -> pd.DataFrame:
    """Missing-value count and percentage per column."""
    m = df.isnull().sum()
    pct = (m / len(df) * 100).round(2)
    res = (
        pd.DataFrame({
            'missing_count': m,
            'missing_pct': pct,
        })
        .query('missing_count > 0')
        .sort_values('missing_pct', ascending=False)
    )
    if label:
        log.info(f"  [{label}] columns with missing values: {len(res)}")
    return res


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to smaller dtypes."""
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def safe_divide(a, b, fill: float = 0):
    """Element-wise a / b, returning fill wherever b == 0."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a_arr, b_arr).shape, fill, dtype=float)
    np.divide(a_arr, b_arr, out=out, where=(b_arr != 0))
    if out.ndim == 0:
        return out.item()
    return out


def validate_inputs() -> None:
    """Check that required raw CSVs exist and contain a plausible row count.

    Thresholds:
      - < 50 % of expected rows → hard error (file is clearly truncated / wrong)
      - < 90 % of expected rows → warning  (partial load, results may differ)

    Row counts are approximated by counting newlines, which is fast but treats
    multi-line quoted fields as extra rows.  The 50 % / 90 % bands are wide
    enough to absorb that imprecision safely.
    """
    required = {
        'application_train.csv'    : 307_511,
        'bureau.csv'               : 1_716_428,
        'bureau_balance.csv'       : 27_299_925,
        'previous_application.csv' : 1_670_214,
        'POS_CASH_balance.csv'     : 10_001_358,
        'credit_card_balance.csv'  :  3_840_312,
        'installments_payments.csv': 13_605_401,
    }
    for fname, expected_rows in required.items():
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Download from: "
                f"https://www.kaggle.com/c/home-credit-default-risk/data"
            )
        with open(path, 'rb') as fh:
            approx_rows = sum(1 for _ in fh)
        if approx_rows < expected_rows * 0.50:
            raise ValueError(
                f"{fname}: only ~{approx_rows:,} lines (expected ~{expected_rows:,}). "
                f"File appears truncated or is the wrong version."
            )
        if approx_rows < expected_rows * 0.90:
            log.warning(
                f"  {fname}: ~{approx_rows:,} lines — "
                f"below 90 %% of expected {expected_rows:,}. "
                f"Results may differ from published benchmarks."
            )
    log.info("  Input validation passed.")
