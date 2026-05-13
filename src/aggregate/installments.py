"""
Installment payments aggregation.

installments_payments.csv tracks actual payment behaviour on previous
Home Credit loans — scheduled vs actual payment dates and amounts.

inst_late_rate is the 3rd strongest predictor of default after DAYS_BIRTH
and cc_utilisation.

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
    """Build per-customer installment payment features.

    Returns:
        DataFrame keyed on SK_ID_CURR.
    """
    inst = load_csv('installments_payments.csv')
    inst = reduce_memory(inst)

    # DAYS_LATE > 0 = paid late.  < 0 = paid early.
    inst['DAYS_LATE']       = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
    inst['PAYMENT_RATIO']   = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT'].replace(0, np.nan)
    inst['PAYMENT_DIFF']    = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['PAID_LATE_FLAG']  = (inst['DAYS_LATE'] > 0).astype(np.int8)
    inst['PAID_EARLY_FLAG'] = (inst['DAYS_LATE'] < 0).astype(np.int8)

    inst_agg = inst.groupby('SK_ID_CURR').agg(
        inst_num_records        = ('SK_ID_PREV',     'count'),
        inst_days_late_mean     = ('DAYS_LATE',       'mean'),
        inst_days_late_max      = ('DAYS_LATE',       'max'),
        inst_days_late_sum      = ('DAYS_LATE',       'sum'),
        inst_payment_ratio_mean = ('PAYMENT_RATIO',   'mean'),
        inst_payment_ratio_min  = ('PAYMENT_RATIO',   'min'),
        inst_payment_diff_mean  = ('PAYMENT_DIFF',    'mean'),
        inst_payment_diff_max   = ('PAYMENT_DIFF',    'max'),
        inst_num_late           = ('PAID_LATE_FLAG',  'sum'),
        inst_num_early          = ('PAID_EARLY_FLAG', 'sum'),
        inst_amt_payment_sum    = ('AMT_PAYMENT',     'sum'),
        inst_amt_instalment_sum = ('AMT_INSTALMENT',  'sum'),
    ).reset_index()

    inst_agg['inst_late_rate'] = (inst_agg['inst_num_late']
                                   / inst_agg['inst_num_records'].replace(0, np.nan))

    inst_agg = reduce_memory(inst_agg)
    log.info(f"  Installments: {inst_agg.shape[1] - 1} features")

    del inst
    gc.collect()

    return inst_agg
