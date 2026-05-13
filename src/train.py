"""
Model training and evaluation.

Trains 4 models with sensible defaults:
  - Logistic Regression  — interpretable baseline
  - Random Forest        — non-linear, robust to outliers
  - XGBoost              — strong on tabular
  - LightGBM             — close to XGBoost, faster

GradientBoostingClassifier was tested and removed — see notes at the bottom
of main.py for the experiments-rejected log.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier
from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    ConfusionMatrixDisplay, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

try:
    from src.utils import CV_FOLDS, FIGURES, PALETTE, PROCESSED, RANDOM_STATE
except ModuleNotFoundError:
    from utils import CV_FOLDS, FIGURES, PALETTE, PROCESSED, RANDOM_STATE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

log = logging.getLogger(__name__)


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train all available models. Returns dict of name -> fitted sklearn Pipeline."""
    models = {}

    log.info("  Training Logistic Regression...")
    lr_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(class_weight='balanced', max_iter=500,
                                       solver='saga', random_state=RANDOM_STATE, n_jobs=-1))
    ])
    lr_pipe.fit(X_train, y_train)
    models['Logistic Regression'] = lr_pipe

    log.info("  Training Random Forest...")
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model',   RandomForestClassifier(n_estimators=300, max_depth=12,
                                            class_weight='balanced', n_jobs=-1,
                                            random_state=RANDOM_STATE))
    ])
    rf_pipe.fit(X_train, y_train)
    models['Random Forest'] = rf_pipe

    if XGBOOST_AVAILABLE:
        log.info("  Training XGBoost...")
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model',   XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       scale_pos_weight=spw, eval_metric='auc',
                                       random_state=RANDOM_STATE, n_jobs=-1,
                                       use_label_encoder=False))
        ])
        xgb_pipe.fit(X_train, y_train)
        models['XGBoost'] = xgb_pipe

    if LGBM_AVAILABLE:
        log.info("  Training LightGBM...")
        lgbm_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model',   LGBMClassifier(n_estimators=600, max_depth=7, learning_rate=0.03,
                                        num_leaves=63, class_weight='balanced',
                                        n_jobs=-1, verbose=-1, random_state=RANDOM_STATE))
        ])
        lgbm_pipe.fit(X_train, y_train)
        models['LightGBM'] = lgbm_pipe

    log.info(f"  Trained: {list(models.keys())}")
    return models


def evaluate(models: dict, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Score every model on validation. Returns dict of name -> metrics dict."""
    results = {}
    for name, pipe in models.items():
        yp  = pipe.predict_proba(X_val)[:, 1]
        ypb = pipe.predict(X_val)
        rep = classification_report(y_val, ypb, output_dict=True, zero_division=0)
        results[name] = {
            'AUC_ROC'          : round(roc_auc_score(y_val, yp),            4),
            'Avg_Precision'    : round(average_precision_score(y_val, yp),  4),
            'Recall_default'   : round(rep['1']['recall'],                   4),
            'Precision_default': round(rep['1']['precision'],                4),
            'F1_default'       : round(rep['1']['f1-score'],                 4),
            'y_proba'          : yp,
        }
        log.info(f"  {name:<22} AUC={results[name]['AUC_ROC']:.4f}  "
                 f"Recall={results[name]['Recall_default']:.4f}")

    # Save metrics summary CSV for SQL/dashboard use
    (pd.DataFrame({k: {m: v for m, v in v.items() if m != 'y_proba'}
                   for k, v in results.items()}).T
     .to_csv(f'{PROCESSED}/model_results.csv'))
    return results


def plot_curves(results: dict, y_val: pd.Series) -> None:
    """ROC + PR curves saved as figure 05."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random (0.50)')
    for (nm, res), col in zip(results.items(), list(PALETTE.values())):
        fpr, tpr, _ = roc_curve(y_val, res['y_proba'])
        axes[0].plot(fpr, tpr, lw=2, color=col,
                     label=f"{nm} ({res['AUC_ROC']:.4f})")
    axes[0].set(title='ROC Curves',
                xlabel='False Positive Rate', ylabel='True Positive Rate')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for (nm, res), col in zip(results.items(), list(PALETTE.values())):
        p, r, _ = precision_recall_curve(y_val, res['y_proba'])
        axes[1].plot(r, p, lw=2, color=col,
                     label=f"{nm} (AP={res['Avg_Precision']:.4f})")
    axes[1].axhline(y_val.mean(), color='k', linestyle='--', lw=1, label='Baseline')
    axes[1].set(title='Precision-Recall Curves', xlabel='Recall', ylabel='Precision')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/05_roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_importance(best_model, best_name: str, feat_names: list[str]) -> None:
    """Feature importance plot saved as figure 06."""
    try:
        step = best_model.named_steps['model']
        if hasattr(step, 'feature_importances_'):
            fi = pd.Series(step.feature_importances_,
                           index=feat_names).nlargest(30).sort_values()
        elif hasattr(step, 'coef_'):
            fi = pd.Series(np.abs(step.coef_[0]),
                           index=feat_names).nlargest(30).sort_values()
        else:
            log.warning("  Best model has no feature_importances_ or coef_")
            return
        plt.figure(figsize=(10, 9))
        fi.plot(kind='barh', color=PALETTE['accent'], edgecolor='white')
        plt.title(f'Top 30 Feature Importances — {best_name}', fontweight='bold')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'{FIGURES}/06_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"\n  Top 10 features:\n{fi.tail(10).iloc[::-1].to_string()}")
    except Exception as e:
        log.warning(f"  Feature importance skipped: {e}")


def plot_confusion(best_model, best_name: str,
                    X_val: pd.DataFrame, y_val: pd.Series) -> np.ndarray:
    """Confusion matrix saved as figure 07. Returns predicted probabilities."""
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred),
        display_labels=['Non-Default', 'Default']).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/07_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    return y_proba


def plot_lift(X_val: pd.DataFrame, y_val: pd.Series, y_proba: np.ndarray) -> float:
    """Cumulative gain / lift chart (figure 08). Returns top-20 lift percentage."""
    val_df = X_val.copy()
    val_df['TARGET']     = y_val.values
    val_df['risk_score'] = y_proba
    sdf = val_df.sort_values('risk_score', ascending=False).reset_index(drop=True)
    cum = sdf['TARGET'].cumsum() / y_val.sum() * 100
    pop = np.arange(1, len(sdf) + 1) / len(sdf) * 100

    plt.figure(figsize=(9, 6))
    plt.plot(pop, cum, lw=2.5, color=PALETTE['accent'], label='Model Lift Curve')
    plt.plot([0, 100], [0, 100], 'k--', lw=1.5, label='Random Baseline')
    plt.fill_between(pop, cum, pop, alpha=0.1, color=PALETTE['accent'])
    plt.xlabel('% Customers Targeted (by risk score)')
    plt.ylabel('% Defaults Captured')
    plt.title('Cumulative Gain / Lift Chart', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/08_lift_chart.png', dpi=150, bbox_inches='tight')
    plt.close()

    top20 = val_df.nlargest(int(len(val_df) * 0.2), 'risk_score')
    lift  = top20['TARGET'].sum() / y_val.sum() * 100
    log.info(f"\n  Business lift: top 20% captures {lift:.1f}% of defaults "
             f"({lift / 20:.1f}x)")
    return lift


def cross_validate(best_model, best_name: str,
                    X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """5-fold CV on a 50K subsample to confirm stability."""
    log.info(f"\n  Cross-validation ({CV_FOLDS}-fold, {best_name})...")
    sample_n = min(50_000, len(X_train))
    X_cv = X_train.sample(sample_n, random_state=RANDOM_STATE)
    y_cv = y_train.loc[X_cv.index]
    scores = cross_val_score(best_model, X_cv, y_cv, cv=CV_FOLDS,
                              scoring='roc_auc', n_jobs=-1)
    log.info(f"  CV-AUC: {scores.mean():.4f} ± {scores.std():.4f}")
