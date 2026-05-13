"""
POS Cash balance aggregation.

POS_CASH_balance.csv contains monthly snapshots of POS and cash loan accounts.
Aggregated to one row per customer.

Returns 13 features.
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
    """Build per-customer POS cash features.

    Returns:
        DataFrame keyed on SK_ID_CURR.
    """
    pos = load_csv('POS_CASH_balance.csv')
    pos = reduce_memory(pos)

    pos_agg = pos.groupby('SK_ID_CURR').agg(
        pos_num_records           = ('SK_ID_PREV',           'count'),
        pos_months_balance_mean   = ('MONTHS_BALANCE',       'mean'),
        pos_months_balance_max    = ('MONTHS_BALANCE',       'max'),
        pos_cnt_instalment_mean   = ('CNT_INSTALMENT',       'mean'),
        pos_cnt_instalment_future = ('CNT_INSTALMENT_FUTURE','mean'),
        pos_sk_dpd_mean           = ('SK_DPD',               'mean'),
        pos_sk_dpd_max            = ('SK_DPD',               'max'),
        pos_sk_dpd_def_mean       = ('SK_DPD_DEF',           'mean'),
        pos_sk_dpd_def_max        = ('SK_DPD_DEF',           'max'),
        pos_num_completed         = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').sum()),
        pos_num_active            = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Active').sum()),
    ).reset_index()

    pos_agg['pos_dpd_flag']        = (pos_agg['pos_sk_dpd_max'] > 0).astype(np.int8)
    pos_agg['pos_completion_rate'] = (pos_agg['pos_num_completed']
                                       / pos_agg['pos_num_records'].replace(0, np.nan))

    pos_agg = reduce_memory(pos_agg)
    log.info(f"  POS Cash: {pos_agg.shape[1] - 1} features")

    del pos
    gc.collect()

    return pos_agg
