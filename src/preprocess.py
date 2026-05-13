"""
Preprocessing — encoding, imputation, capping, train/val split.

Operates on the merged + engineered DataFrame and produces:
  - train_df, test_df  (pre-Kaggle-split frames with SK_ID_CURR preserved)
  - X_train, X_val, y_train, y_val  (model-ready arrays)
  - class_weights for cost-sensitive training
"""

from __future__ import annotations

import gc
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.utils           import class_weight

try:
    from src.utils import MISSING_DROP_PCT, RANDOM_STATE, TEST_SIZE, missing_profile, reduce_memory
except ModuleNotFoundError:
    from utils import MISSING_DROP_PCT, RANDOM_STATE, TEST_SIZE, missing_profile, reduce_memory

log = logging.getLogger(__name__)


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Drop ID + segmentation columns, label-encode binary, one-hot the rest.

    Args:
        df: Merged feature DataFrame with SK_ID_CURR and SET columns.

    Returns:
        Tuple of (encoded_df, set_column, sk_id_curr_column).
        SET and SK_ID_CURR are returned separately so they can be re-attached
        for the train/test split.
    """
    drop_cols = ['SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'SET',
                 'income_group', 'loan_size', 'age_group', 'risk_level', 'employment_group']

    SET_col = df['SET'].copy()
    df_ids  = df['SK_ID_CURR'].copy()
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    bin_cats = [c for c in df.select_dtypes('object').columns if df[c].nunique() <= 2]
    le = LabelEncoder()
    for col in bin_cats:
        df[col] = le.fit_transform(df[col].astype(str))

    cat_remain = df.select_dtypes('object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_remain, drop_first=True, dtype=np.int8)
    log.info(f"  Shape after encoding: {df.shape}")
    missing_profile(df, label='after encoding')

    return df, SET_col, df_ids


def impute_and_cap(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-missing columns, median-impute the rest, cap money outliers.

    Args:
        df: Encoded DataFrame.

    Returns:
        Cleaned DataFrame ready for splitting.
    """
    mp        = missing_profile(df)
    high_miss = mp[mp['missing_pct'] > MISSING_DROP_PCT].index.tolist()
    df.drop(columns=high_miss, inplace=True)
    log.info(f"  Dropped {len(high_miss)} columns with >{MISSING_DROP_PCT}% missing")

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'TARGET']
    medians  = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    assert df[num_cols].isnull().sum().sum() == 0, "Imputation left NaN values"

    # Cap monetary columns at 1st and 99th percentiles to prevent extreme
    # outliers from dominating tree splits.
    for col in [c for c in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                              'AMT_GOODS_PRICE'] if c in df.columns]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    df = reduce_memory(df)
    log.info(f"  Final feature matrix: {df.shape}")
    return df


def split(df: pd.DataFrame, SET_col: pd.Series, df_ids: pd.Series):
    """Re-attach SET + SK_ID_CURR, slice train/test, then train/val.

    Args:
        df:      Encoded + imputed DataFrame (no SET/SK_ID_CURR columns).
        SET_col: 'train'/'test' label series from the merged DataFrame.
        df_ids:  SK_ID_CURR series from the merged DataFrame.

    Returns:
        train_df, test_df, X_train, X_val, y_train, y_val, class_weights_dict
    """
    df['SK_ID_CURR'] = df_ids.values
    df['SET']        = SET_col.values

    train_df = df[df['SET'] == 'train'].drop(columns=['SET', 'SK_ID_CURR'])
    test_df  = df[df['SET'] == 'test'].drop(columns=['SET', 'SK_ID_CURR', 'TARGET'])
    del df
    gc.collect()

    X = train_df.drop(columns=['TARGET'])
    y = train_df['TARGET'].astype(int)

    # stratify=y preserves the 8.07% default rate in both splits.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    log.info(f"  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}")

    cw      = class_weight.compute_class_weight(
        'balanced', classes=np.array([0, 1]), y=y_train)
    cw_dict = {0: cw[0], 1: cw[1]}
    log.info(f"  Class weights: {cw_dict}")

    return train_df, test_df, X_train, X_val, y_train, y_val, cw_dict
