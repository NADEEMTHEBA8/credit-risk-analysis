"""
Utility functions and configuration for the credit risk pipeline.

All pure-function helpers that have no dependencies on pipeline state.
These can be imported from anywhere — pipeline modules, tests, notebooks.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
RAW_DIR      = 'data/raw'
PROCESSED    = 'data/processed'
FIGURES      = 'figures'
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# Threshold for dropping columns with too much missing data.
# Tested 60/70/80/90% — 80% gave best AUC vs memory tradeoff.
MISSING_DROP_PCT = 80

# Threshold selection uses F-beta instead of F1.
# A missed default (false negative) costs roughly 8x more than
# a false positive at a bank. beta=2.5 reflects this asymmetry.
DECISION_BETA = 2.5

# Domain-appropriate colour palette.
PALETTE = {
    'risk'    : '#C1292E',  # brick red — high risk
    'safe'    : '#2D6A4F',  # forest green — low risk
    'neutral' : '#4A4E69',  # slate — neutral
    'accent'  : '#22577A',  # deep teal — highlights
    'warm'    : '#9A8C98',  # warm grey — secondary
}


# ─── LOGGER ───────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)


# ─── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def load_csv(filename: str, usecols: list = None) -> pd.DataFrame:
    """Load a CSV from RAW_DIR and log its shape.

    Args:
        filename: CSV filename relative to RAW_DIR.
        usecols:  Optional column subset to load.

    Returns:
        Loaded DataFrame.
    """
    path = os.path.join(RAW_DIR, filename)
    df   = pd.read_csv(path, usecols=usecols)
    log.info(f"  Loaded {filename:<45} shape: {df.shape}")
    return df


def missing_profile(df: pd.DataFrame, label: str = '') -> pd.DataFrame:
    """Return a DataFrame sorted by missing percentage, descending.

    Args:
        df:    Input DataFrame.
        label: Optional label for log output.

    Returns:
        DataFrame with missing_count and missing_pct columns,
        filtered to columns with at least one missing value.
    """
    m   = df.isnull().sum()
    pct = (m / len(df) * 100).round(2)
    res = (pd.DataFrame({'missing_count': m, 'missing_pct': pct})
           .query('missing_count > 0')
           .sort_values('missing_pct', ascending=False))
    if label:
        log.info(f"  [{label}] columns with missing values: {len(res)}")
    return res


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to smallest fitting type.

    Cuts RAM by ~40% on this dataset. Without this, loading
    bureau_balance (27M rows) caused OOM on a 16GB machine.

    Args:
        df: DataFrame to optimise in-place.

    Returns:
        Same DataFrame with downcasted dtypes.
    """
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df


def safe_divide(a, b, fill: float = 0):
    """Element-wise a/b, returning fill wherever b == 0.

    Works for scalars, numpy arrays, and pandas Series. Avoids the
    np.where eager-evaluation trap where ``a / b`` would raise before
    np.where could pick the fill value.

    Args:
        a:    Numerator.
        b:    Denominator.
        fill: Value used when denominator is zero.

    Returns:
        Same shape as inputs, with zero-safe division results.
    """
    # numpy.divide with where= mask avoids evaluating a/b at the zero
    # positions at all. out= gives the fill value its starting point.
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a_arr, b_arr).shape, fill, dtype=float)
    np.divide(a_arr, b_arr, out=out, where=(b_arr != 0))
    # If both inputs were scalars, return a scalar
    if out.ndim == 0:
        return out.item()
    return out


def validate_inputs() -> None:
    """Check that all required raw CSVs exist and are plausible.

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If a required file looks suspiciously small.
    """
    required = {
        'application_train.csv'    : 300_000,
        'bureau.csv'               : 1_000_000,
        'bureau_balance.csv'       : 20_000_000,
        'previous_application.csv' : 1_000_000,
        'POS_CASH_balance.csv'     : 5_000_000,
        'credit_card_balance.csv'  : 2_000_000,
        'installments_payments.csv': 10_000_000,
    }
    for fname, min_rows in required.items():
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Download from: "
                f"https://www.kaggle.com/c/home-credit-default-risk/data"
            )
        with open(path, 'rb') as f:
            approx_rows = sum(1 for _ in f)
        if approx_rows < min_rows // 100:
            raise ValueError(
                f"{fname}: only ~{approx_rows} lines. File may be incomplete."
            )
    log.info("  Input validation passed.")
