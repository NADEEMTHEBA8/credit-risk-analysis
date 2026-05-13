"""
Credit card balance aggregation.

credit_card_balance.csv contains monthly snapshots of credit card accounts.

Note: cc_utilisation = mean_balance / mean_limit can EXCEED 1.0 when a
customer is over their credit limit. In EDA, util > 1.0 had a 25.91%
default rate — the single highest default rate of any feature segment.

Returns 16 features.
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
    """Build per-customer credit card features.

    Returns:
        DataFrame keyed on SK_ID_CURR.
    """
    cc = load_csv('credit_card_balance.csv')
    cc = reduce_memory(cc)

    cc_agg = cc.groupby('SK_ID_CURR').agg(
        cc_num_records          = ('SK_ID_PREV',              'count'),
        cc_months_balance_mean  = ('MONTHS_BALANCE',          'mean'),
        cc_balance_mean         = ('AMT_BALANCE',             'mean'),
        cc_balance_max          = ('AMT_BALANCE',             'max'),
        cc_credit_limit_mean    = ('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
        cc_drawings_mean        = ('AMT_DRAWINGS_CURRENT',    'mean'),
        cc_drawings_total       = ('AMT_DRAWINGS_CURRENT',    'sum'),
        cc_payment_current_mean = ('AMT_PAYMENT_CURRENT',     'mean'),
        cc_payment_total_mean   = ('AMT_TOTAL_RECEIVABLE',    'mean'),
        cc_dpd_mean             = ('SK_DPD',                  'mean'),
        cc_dpd_max              = ('SK_DPD',                  'max'),
        cc_dpd_def_mean         = ('SK_DPD_DEF',              'mean'),
        cc_cnt_drawings_mean    = ('CNT_DRAWINGS_CURRENT',    'mean'),
    ).reset_index()

    # Mean balance / mean limit captures typical behaviour, not peak stress.
    cc_agg['cc_utilisation'] = (cc_agg['cc_balance_mean']
                                 / cc_agg['cc_credit_limit_mean'].replace(0, np.nan))
    cc_agg['cc_dpd_flag']    = (cc_agg['cc_dpd_max'] > 0).astype(np.int8)
    cc_agg['cc_over_limit']  = (cc_agg['cc_utilisation'] > 1.0).astype(np.int8)

    cc_agg = reduce_memory(cc_agg)
    log.info(f"  Credit card: {cc_agg.shape[1] - 1} features")

    del cc
    gc.collect()

    return cc_agg
