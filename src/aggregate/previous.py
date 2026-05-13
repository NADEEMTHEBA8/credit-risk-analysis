"""
Previous application aggregation.

previous_application.csv contains prior loan applications at Home Credit
(internal, not external bureau). Aggregated to one row per customer.

Returns 18 features.
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
    """Build per-customer previous-application features.

    Returns:
        DataFrame keyed on SK_ID_CURR.
    """
    prev = load_csv('previous_application.csv')
    prev = reduce_memory(prev)

    # 365243 is the "no value" sentinel for DAYS_ columns. Replace with NaN
    # so it doesn't dominate the mean/min/max aggregations.
    for c in [col for col in prev.columns if col.startswith('DAYS_')]:
        prev[c] = prev[c].replace(365243, np.nan)

    prev_agg = prev.groupby('SK_ID_CURR').agg(
        prev_num_applications      = ('SK_ID_PREV',             'count'),
        prev_num_approved          = ('NAME_CONTRACT_STATUS',    lambda x: (x == 'Approved').sum()),
        prev_num_refused           = ('NAME_CONTRACT_STATUS',    lambda x: (x == 'Refused').sum()),
        prev_num_cancelled         = ('NAME_CONTRACT_STATUS',    lambda x: (x == 'Canceled').sum()),
        prev_amt_credit_sum        = ('AMT_CREDIT',              'sum'),
        prev_amt_credit_mean       = ('AMT_CREDIT',              'mean'),
        prev_amt_annuity_mean      = ('AMT_ANNUITY',             'mean'),
        prev_amt_down_mean         = ('AMT_DOWN_PAYMENT',        'mean'),
        prev_days_decision_mean    = ('DAYS_DECISION',           'mean'),
        prev_days_decision_min     = ('DAYS_DECISION',           'min'),
        prev_hour_appr_mean        = ('HOUR_APPR_PROCESS_START', 'mean'),
        prev_rate_down_mean        = ('RATE_DOWN_PAYMENT',       'mean'),
        prev_days_first_due_mean   = ('DAYS_FIRST_DUE',          'mean'),
        prev_days_last_due_mean    = ('DAYS_LAST_DUE',           'mean'),
        prev_days_termination_mean = ('DAYS_TERMINATION',        'mean'),
        prev_cnt_payment_mean      = ('CNT_PAYMENT',             'mean'),
    ).reset_index()

    prev_agg['prev_approval_rate'] = (prev_agg['prev_num_approved']
                                       / prev_agg['prev_num_applications'].replace(0, np.nan))
    prev_agg['prev_refusal_rate']  = (prev_agg['prev_num_refused']
                                       / prev_agg['prev_num_applications'].replace(0, np.nan))

    prev_agg = reduce_memory(prev_agg)
    log.info(f"  Previous apps: {prev_agg.shape[1] - 1} features")

    del prev
    gc.collect()

    return prev_agg
