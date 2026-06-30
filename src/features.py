"""
Feature engineering and segmentation.

Operates on the merged application + aggregates DataFrame:
  - Ratio features (income_credit_ratio, employment_age_ratio, etc.)
  - Cross-table interaction features
  - Segmentation labels (for SQL and EDA, dropped before ML)

Key finding noted in pipeline header:
  income_credit_ratio has correlation 0.0018 with TARGET (17th of 19).
  Age (DAYS_BIRTH) is 44x stronger. Kept for completeness and SQL alignment.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from src.utils import load_csv, reduce_memory, safe_divide, AGE_BINS, AGE_LABELS, EMP_BINS, EMP_LABELS, INC_BINS, INC_LABELS, RISK_BINS, RISK_LABELS

log = logging.getLogger(__name__)


def engineer(df: pd.DataFrame, app_columns: list[str]) -> pd.DataFrame:
    """Add ratio, interaction, and segmentation features in place.

    Args:
        df:          Merged DataFrame (application + 5 aggregates).
        app_columns: Original application column names. Used to identify
                     DAYS_* columns that came from the application table
                     and should be made positive.

    Returns:
        Same DataFrame with engineered features added.
    """
    # Replace DAYS_EMPLOYED sentinel BEFORE abs() so it doesn't become a
    # large positive number that distorts employment_age_ratio.
    df['DAYS_EMPLOYED']      = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['DAYS_EMPLOYED_FLAG'] = df['DAYS_EMPLOYED'].isnull().astype(np.int8)

    # Convert all DAYS_ columns from the application table to positive.
    # Aggregated DAYS_ columns from secondary tables are left alone — they
    # already have meaningful sign conventions in their aggregations.
    days_app = [c for c in df.columns if c.startswith('DAYS_') and c in app_columns]
    for c in days_app:
        if c in df.columns:
            df[c] = df[c].abs()

    # Ratio features. income_credit_ratio is famously weak here — kept anyway
    # because the SQL layer references it.
    df['income_credit_ratio']  = safe_divide(df['AMT_INCOME_TOTAL'], df['AMT_CREDIT'])
    df['annuity_income_ratio'] = safe_divide(df['AMT_ANNUITY'],      df['AMT_INCOME_TOTAL'])
    df['credit_goods_ratio']   = safe_divide(df['AMT_CREDIT'],       df['AMT_GOODS_PRICE'])
    df['employment_age_ratio'] = safe_divide(df['DAYS_EMPLOYED'],    df['DAYS_BIRTH'])
    df['age_years']            = df['DAYS_BIRTH'].abs() / 365
    df['employed_years']       = df['DAYS_EMPLOYED'].abs() / 365
    df['income_per_person']    = safe_divide(df['AMT_INCOME_TOTAL'],
                                              df['CNT_FAM_MEMBERS'].fillna(1))

    # Cross-table interactions
    df['bur_debt_income_ratio'] = safe_divide(
        df.get('bur_total_debt', pd.Series(0, index=df.index)),
        df['AMT_INCOME_TOTAL'])
    df['inst_late_per_credit']  = safe_divide(
        df.get('inst_num_late', pd.Series(0, index=df.index)),
        df.get('prev_num_applications', pd.Series(1, index=df.index)))
    df['cc_util_income_ratio']  = safe_divide(
        df.get('cc_utilisation', pd.Series(0, index=df.index)),
        df['AMT_INCOME_TOTAL'])

    # Segmentation labels — dropped before ML, used only for SQL and EDA.
    # Bin edges and labels are defined in utils.py so eda.py shares them.
    df['income_group']     = pd.cut(df['AMT_INCOME_TOTAL'],
        bins=INC_BINS, labels=INC_LABELS, right=False)
    df['loan_size']        = pd.cut(df['AMT_CREDIT'],
        bins=[0, 100_000, 500_000, float('inf')],
        labels=['Small', 'Medium', 'Large'], right=False)
    df['age_group']        = pd.cut(df['age_years'],
        bins=AGE_BINS, labels=AGE_LABELS)
    df['risk_level']       = pd.cut(df['income_credit_ratio'],
        bins=RISK_BINS, labels=RISK_LABELS)
    df['employment_group'] = pd.cut(df['employment_age_ratio'],
        bins=EMP_BINS, labels=EMP_LABELS)

    log.info(f"  Features after engineering: {df.shape[1]}")
    return df
