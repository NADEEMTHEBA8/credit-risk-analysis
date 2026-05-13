"""
Threshold selection — F-beta with beta=2.5.

F1 (beta=1) treats false negatives and false positives equally.
For credit decisions, a missed default costs roughly 8x more than a
false alarm. beta=2.5 weights recall ~6x more than precision, which
matches that economic reality.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, fbeta_score

try:
    from src.utils import DECISION_BETA, FIGURES, PALETTE
except ModuleNotFoundError:
    from utils import DECISION_BETA, FIGURES, PALETTE

log = logging.getLogger(__name__)


def select_threshold(y_val: pd.Series, y_proba: np.ndarray) -> float:
    """Sweep thresholds 0.05–0.90, return one that maximises F-beta.

    Args:
        y_val:   True validation labels.
        y_proba: Predicted probabilities from best model.

    Returns:
        Optimal threshold as a float.
    """
    log.info("  Selecting threshold (F-beta beta=2.5)...")

    thr_rows = []
    for t in np.arange(0.05, 0.91, 0.05):
        yp_t = (y_proba >= t).astype(int)
        fb   = fbeta_score(y_val, yp_t, beta=DECISION_BETA, zero_division=0)
        rep  = classification_report(y_val, yp_t, output_dict=True, zero_division=0)
        thr_rows.append({
            'threshold': round(t, 2), 'fbeta': round(fb, 4),
            'precision': round(rep['1']['precision'], 4),
            'recall'   : round(rep['1']['recall'],    4),
            'f1'       : round(rep['1']['f1-score'],  4),
        })

    thr_df  = pd.DataFrame(thr_rows)
    opt_idx = thr_df['fbeta'].idxmax()
    opt_thr = thr_df.loc[opt_idx, 'threshold']
    log.info(f"  Optimal threshold: {opt_thr} "
             f"(recall={thr_df.loc[opt_idx, 'recall']:.3f}, "
             f"precision={thr_df.loc[opt_idx, 'precision']:.3f})")

    _plot(thr_df, opt_thr)
    return opt_thr


def _plot(thr_df: pd.DataFrame, opt_thr: float) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(thr_df.threshold, thr_df.precision, color=PALETTE['safe'],
             marker='o', label='Precision')
    plt.plot(thr_df.threshold, thr_df.recall, color=PALETTE['risk'],
             marker='o', label='Recall')
    plt.plot(thr_df.threshold, thr_df.f1, color=PALETTE['neutral'],
             marker='o', label='F1')
    plt.plot(thr_df.threshold, thr_df.fbeta, color=PALETTE['accent'],
             marker='o', lw=2.5, label=f'F-beta (β={DECISION_BETA})')
    plt.axvline(opt_thr, color='gray', linestyle='--', label=f'Chosen = {opt_thr}')
    plt.title(f'Threshold Selection — Default Class\n'
              f'F-beta (β={DECISION_BETA}) reflects ~8x cost asymmetry of missed defaults',
              fontweight='bold')
    plt.xlabel('Decision Threshold')
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/09_threshold_tuning.png', dpi=150, bbox_inches='tight')
    plt.close()
