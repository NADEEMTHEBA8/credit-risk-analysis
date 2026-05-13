"""
Exploratory data analysis — saves 4 PNG figures to figures/ folder.

This is exploration code by nature — one-off, plot-heavy, and not really
reusable. Kept in its own module so pipeline.py and main.py don't get
cluttered with matplotlib boilerplate.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from src.utils import FIGURES, PALETTE
except ModuleNotFoundError:
    from utils import FIGURES, PALETTE

log = logging.getLogger(__name__)


def run(train_df: pd.DataFrame) -> None:
    """Generate the 4 EDA figures from the training feature matrix."""
    eda = train_df.copy()
    eda['age_group']        = pd.cut(eda['age_years'],
        bins=[0, 30, 40, 50, 60, float('inf')],
        labels=['18-29', '30-39', '40-49', '50-59', '60+'])
    eda['employment_group'] = pd.cut(eda['employment_age_ratio'],
        bins=[-0.001, 0.1, 0.3, 0.6, float('inf')],
        labels=['Unstable', 'Short-term', 'Moderate', 'Stable'])
    eda['income_group']     = pd.cut(eda['AMT_INCOME_TOTAL'],
        bins=[0, 100_000, 200_000, float('inf')],
        labels=['Low', 'Medium', 'High'], right=False)

    _plot_top_predictors(eda)
    _plot_age_employment_heatmap(eda)
    _plot_behavioural_distributions(eda)
    _plot_correlations(eda)

    log.info("  EDA figures saved")


def _plot_top_predictors(eda: pd.DataFrame) -> None:
    """Figure 01 — default rate by age / employment / income."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    age_dr = eda.groupby('age_group', observed=True)['TARGET'].mean() * 100
    colors_age = [PALETTE['risk'] if v > 9 else PALETTE['neutral'] if v > 7
                  else PALETTE['safe'] for v in age_dr.values]
    age_dr.plot(kind='bar', ax=axes[0], color=colors_age, edgecolor='white')
    axes[0].set_title('Default Rate by Age Group\n'
                      '(Strongest behavioural predictor — corr 0.078)',
                      fontweight='bold', fontsize=10)
    axes[0].set_ylabel('Default Rate (%)')
    axes[0].tick_params(axis='x', rotation=0)
    for i, v in enumerate(age_dr.values):
        axes[0].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=8)

    emp_dr = eda.groupby('employment_group', observed=True)['TARGET'].mean() * 100
    colors_emp = [PALETTE['risk'] if v > 9 else PALETTE['neutral'] if v > 6
                  else PALETTE['safe'] for v in emp_dr.values]
    emp_dr.plot(kind='bar', ax=axes[1], color=colors_emp, edgecolor='white')
    axes[1].set_title('Default Rate by Employment Stability\n'
                      '(2nd strongest behavioural — corr 0.058)',
                      fontweight='bold', fontsize=10)
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].tick_params(axis='x', rotation=0)
    for i, v in enumerate(emp_dr.values):
        axes[1].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=8)

    inc_dr = eda.groupby('income_group', observed=True)['TARGET'].mean() * 100
    inc_dr.plot(kind='bar', ax=axes[2], color=PALETTE['neutral'], edgecolor='white')
    axes[2].set_title('Default Rate by Income Group\n(9th strongest — corr 0.023)',
                      fontweight='bold', fontsize=10)
    axes[2].set_ylabel('Default Rate (%)')
    axes[2].tick_params(axis='x', rotation=0)
    for i, v in enumerate(inc_dr.values):
        axes[2].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=8)

    plt.suptitle('Key Insight: Age and Employment Stability predict default better than Income',
                 fontweight='bold', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/01_top_predictors.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_age_employment_heatmap(eda: pd.DataFrame) -> None:
    """Figure 02 — interaction of two strongest behavioural predictors."""
    pivot = (eda.groupby(['age_group', 'employment_group'], observed=True)['TARGET']
             .mean().unstack() * 100)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.5, cbar_kws={'label': 'Default Rate (%)'})
    plt.title('Default Rate — Age × Employment Stability\n'
              'Young + Unstable = 12.47% vs Old + Stable = 3.05%',
              fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/02_age_employment_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_behavioural_distributions(eda: pd.DataFrame) -> None:
    """Figure 03 — distribution overlays for top behavioural features.

    dropna() and direct ax.hist() bypass the pandas/matplotlib bin-count
    mismatch bug on matplotlib 3.10 + numpy 1.26.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    features = [
        ('inst_late_rate',     'Installment Late Rate (3rd strongest, corr=0.070)'),
        ('cc_utilisation',     'CC Utilisation (2nd strongest, corr=0.075)'),
        ('prev_approval_rate', 'Previous Approval Rate (5th strongest, corr=0.063)'),
        ('bur_max_overdue',    'Bureau Max Overdue (13th, corr=0.009)'),
        ('pos_sk_dpd_mean',    'POS Avg DPD'),
        ('inst_days_late_mean','Avg Days Late per Payment'),
    ]
    for ax, (feat, label) in zip(axes.flatten(), features):
        if feat in eda.columns:
            try:
                v0 = eda.loc[eda['TARGET'] == 0, feat].dropna()
                v1 = eda.loc[eda['TARGET'] == 1, feat].dropna()
                ax.hist(v0, bins=40, alpha=0.6, color=PALETTE['safe'],
                        label='Non-Default', density=True)
                ax.hist(v1, bins=40, alpha=0.6, color=PALETTE['risk'],
                        label='Default', density=True)
                ax.set_title(label, fontweight='bold', fontsize=9)
                ax.legend(fontsize=7)
            except Exception as e:
                log.warning(f"  Histogram skipped for {feat}: {e}")
                ax.set_visible(False)
    plt.suptitle('Behavioural Feature Distributions by Default Status',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/03_behavioral_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_correlations(eda: pd.DataFrame) -> None:
    """Figure 04 — top 20 features by |correlation| with TARGET."""
    corr_cols = [c for c in eda.select_dtypes(include=[np.number]).columns
                 if c != 'TARGET']
    top_corr = (eda[corr_cols + ['TARGET']].corr()['TARGET']
                .drop('TARGET').abs().nlargest(20).sort_values())
    plt.figure(figsize=(10, 7))
    top_corr.plot(kind='barh', color=PALETTE['accent'], edgecolor='white')
    plt.title('Top 20 Features by Correlation with Default', fontweight='bold')
    plt.xlabel('|Pearson Correlation with TARGET|')
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/04_feature_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()
