"""
Bureau aggregation — two-level groupby.

bureau.csv is at the credit level (one row per past external credit).
bureau_balance.csv is at the month level (one row per month per credit).

Pipeline:
  1. Aggregate bureau_balance up to one row per SK_ID_BUREAU.
  2. Merge that summary back into bureau.
  3. Aggregate the enriched bureau up to one row per SK_ID_CURR.

Returns 19 features per customer.
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd

try:
    from src.utils import load_csv, reduce_memory
except ModuleNotFoundError:
    from utils import load_csv, reduce_memory

log = logging.getLogger(__name__)


def run() -> pd.DataFrame:
    """Build per-customer bureau features.

    Returns:
        DataFrame keyed on SK_ID_CURR with 19 feature columns.
    """
    bureau = load_csv('bureau.csv')
    bureau = reduce_memory(bureau)

    bb = load_csv('bureau_balance.csv')
    bb = reduce_memory(bb)

    # STATUS 1-5 = DPD bands (1=1-30 days late, etc.). C=closed, X=unknown.
    bb['STATUS_BAD'] = bb['STATUS'].isin(['1', '2', '3', '4', '5']).astype(np.int8)
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(
        bb_months_total = ('MONTHS_BALANCE', 'count'),
        bb_months_bad   = ('STATUS_BAD',     'sum'),
        bb_max_dpd_band = ('STATUS_BAD',     'max'),
    ).reset_index()
    del bb
    gc.collect()

    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    del bb_agg
    gc.collect()

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        bur_num_credits       = ('SK_ID_BUREAU',          'count'),
        bur_num_active        = ('CREDIT_ACTIVE',          lambda x: (x == 'Active').sum()),
        bur_num_closed        = ('CREDIT_ACTIVE',          lambda x: (x == 'Closed').sum()),
        bur_total_credit      = ('AMT_CREDIT_SUM',         'sum'),
        bur_avg_credit        = ('AMT_CREDIT_SUM',         'mean'),
        bur_total_debt        = ('AMT_CREDIT_SUM_DEBT',    'sum'),
        bur_max_overdue       = ('AMT_CREDIT_SUM_OVERDUE', 'max'),
        bur_avg_overdue       = ('AMT_CREDIT_SUM_OVERDUE', 'mean'),
        bur_total_overdue     = ('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        bur_days_credit_mean  = ('DAYS_CREDIT',            'mean'),
        bur_days_credit_min   = ('DAYS_CREDIT',            'min'),
        bur_days_enddate_mean = ('DAYS_CREDIT_ENDDATE',    'mean'),
        bur_days_enddate_max  = ('DAYS_CREDIT_ENDDATE',    'max'),
        bur_prolong_sum       = ('CNT_CREDIT_PROLONG',     'sum'),
        bur_num_bad_months    = ('bb_months_bad',          'sum'),
        bur_avg_bad_months    = ('bb_months_bad',          'mean'),
    ).reset_index()

    bur_cr = bureau_agg['bur_total_credit'].replace(0, np.nan)
    bureau_agg['bur_debt_credit_ratio']    = bureau_agg['bur_total_debt']    / bur_cr
    bureau_agg['bur_overdue_credit_ratio'] = bureau_agg['bur_total_overdue'] / bur_cr
    bureau_agg['bur_active_ratio']         = (bureau_agg['bur_num_active']
                                              / bureau_agg['bur_num_credits'].replace(0, np.nan))

    bureau_agg = reduce_memory(bureau_agg)
    log.info(f"  Bureau: {bureau_agg.shape[1] - 1} features")

    del bureau
    gc.collect()

    return bureau_agg
