"""
Home Credit Default Risk — Multi-Table Credit Scoring Pipeline
==============================================================
Dataset  : Home Credit Default Risk (Kaggle, 2018)
Tables   : application_train/test · bureau · bureau_balance
           previous_application · POS_CASH_balance
           credit_card_balance · installments_payments

Top features (from actual EDA):
  EXT_SOURCE_3, EXT_SOURCE_2, EXT_SOURCE_1 dominate — external bureau scores.
  Among behavioural features: age, cc_utilisation, inst_late_rate are strongest.
  income_credit_ratio ranks low despite being the expected primary signal.

Pipeline stages:
  Load → Aggregate 6 secondary tables → Merge → Feature engineering
  → Encode → Impute → Train/Val split → EDA → Model training
  → Evaluation → Threshold selection → Export
"""

import gc
import logging
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    ConfusionMatrixDisplay, average_precision_score, classification_report,
    confusion_matrix, fbeta_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.utils import class_weight

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

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ─── IMPORTS FROM SUB-MODULES ─────────────────────────────────────────────────
try:
    from src.utils import (  # noqa: E402
        CV_FOLDS, DECISION_BETA, FIGURES, MISSING_DROP_PCT, PALETTE,
        PROCESSED, RANDOM_STATE, RAW_DIR, TEST_SIZE,
        load_csv, missing_profile, reduce_memory, safe_divide, validate_inputs,
    )
    from src.aggregate import bureau as agg_bureau
    from src.aggregate import previous as agg_previous
    from src.aggregate import pos_cash as agg_pos
    from src.aggregate import credit_card as agg_cc
    from src.aggregate import installments as agg_inst
    from src.features import engineer as engineer_features
except ModuleNotFoundError:
    from utils import (  # noqa: E402
        CV_FOLDS, DECISION_BETA, FIGURES, MISSING_DROP_PCT, PALETTE,
        PROCESSED, RANDOM_STATE, RAW_DIR, TEST_SIZE,
        load_csv, missing_profile, reduce_memory, safe_divide, validate_inputs,
    )
    from aggregate import bureau as agg_bureau
    from aggregate import previous as agg_previous
    from aggregate import pos_cash as agg_pos
    from aggregate import credit_card as agg_cc
    from aggregate import installments as agg_inst
    from features import engineer as engineer_features

for d in [RAW_DIR, PROCESSED, FIGURES]:
    os.makedirs(d, exist_ok=True)

log.info("=" * 65)
log.info("  HOME CREDIT DEFAULT RISK — MULTI-TABLE PIPELINE")
log.info("=" * 65)


# ── DATA INGESTION ────────────────────────────────────────────────────────────
log.info("\n[1/8] Loading application tables...")
validate_inputs()

train_raw = load_csv('application_train.csv')
test_raw  = load_csv('application_test.csv')

train_raw['SET']   = 'train'
test_raw['SET']    = 'test'
test_raw['TARGET'] = np.nan
app = pd.concat([train_raw, test_raw], ignore_index=True, sort=False)
app_columns = list(app.columns)

default_rate = app[app['SET'] == 'train']['TARGET'].mean() * 100
log.info(f"  Train: {(app['SET']=='train').sum():,}  |  "
         f"Test: {(app['SET']=='test').sum():,}  |  "
         f"Default rate: {default_rate:.2f}%")


# ── SECONDARY TABLE AGGREGATION ───────────────────────────────────────────────
log.info("\n[2/8] Aggregating secondary tables...")
bureau_agg = agg_bureau.run()
prev_agg   = agg_previous.run()
pos_agg    = agg_pos.run()
cc_agg     = agg_cc.run()
inst_agg   = agg_inst.run()


# ── TABLE MERGE ───────────────────────────────────────────────────────────────
log.info("\n[3/8] Merging all tables...")
df = app.copy()
for agg_df, name in [
    (bureau_agg,  'bureau'),
    (prev_agg,    'previous_application'),
    (pos_agg,     'POS_cash'),
    (cc_agg,      'credit_card'),
    (inst_agg,    'installments'),
]:
    df = df.merge(agg_df, on='SK_ID_CURR', how='left')
    log.info(f"  After merging {name:<25}: {df.shape}")

del bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg
gc.collect()
log.info(f"  Final merged shape: {df.shape}")


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
log.info("\n[4/8] Engineering features...")
df = engineer_features(df, app_columns)


# ── ENCODING ──────────────────────────────────────────────────────────────────
log.info("\n[5/8] Encoding...")

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


# ── IMPUTATION & CAPPING ──────────────────────────────────────────────────────
log.info("\n[6/8] Imputing and capping...")

mp        = missing_profile(df)
high_miss = mp[mp['missing_pct'] > MISSING_DROP_PCT].index.tolist()
df.drop(columns=high_miss, inplace=True)
log.info(f"  Dropped {len(high_miss)} columns with >{MISSING_DROP_PCT}% missing")

num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'TARGET']
medians  = df[num_cols].median()
df[num_cols] = df[num_cols].fillna(medians)
assert df[num_cols].isnull().sum().sum() == 0, "Imputation left NaN values"

for col in [c for c in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                          'AMT_GOODS_PRICE'] if c in df.columns]:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lo, hi)

df = reduce_memory(df)
log.info(f"  Final feature matrix: {df.shape}")


# ── TRAIN / VAL SPLIT ─────────────────────────────────────────────────────────
log.info("\n[7/8] Splitting and training...")

df['SK_ID_CURR'] = df_ids.values
df['SET']        = SET_col.values

train_df = df[df['SET'] == 'train'].drop(columns=['SET', 'SK_ID_CURR'])
test_df  = df[df['SET'] == 'test'].drop(columns=['SET', 'SK_ID_CURR', 'TARGET'])
del df; gc.collect()

X = train_df.drop(columns=['TARGET'])
y = train_df['TARGET'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
log.info(f"  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}")

cw      = class_weight.compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
cw_dict = {0: cw[0], 1: cw[1]}
log.info(f"  Class weights: {cw_dict}")


# ── EDA ───────────────────────────────────────────────────────────────────────
eda = train_df.copy()
eda['age_group']        = pd.cut(eda['age_years'],
    bins=[0,30,40,50,60,float('inf')], labels=['18-29','30-39','40-49','50-59','60+'])
eda['employment_group'] = pd.cut(eda['employment_age_ratio'],
    bins=[-0.001,0.1,0.3,0.6,float('inf')], labels=['Unstable','Short-term','Moderate','Stable'])
eda['income_group']     = pd.cut(eda['AMT_INCOME_TOTAL'],
    bins=[0,100_000,200_000,float('inf')], labels=['Low','Medium','High'], right=False)

# Plot 1: Top 3 predictors vs default rate
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

age_dr = eda.groupby('age_group', observed=True)['TARGET'].mean() * 100
colors_age = [PALETTE['risk'] if v > 9 else PALETTE['neutral'] if v > 7
              else PALETTE['safe'] for v in age_dr.values]
age_dr.plot(kind='bar', ax=axes[0], color=colors_age, edgecolor='white')
axes[0].set_title('Default Rate by Age Group\n(Strongest behavioural predictor — corr 0.078)',
                  fontweight='bold', fontsize=10)
axes[0].set_ylabel('Default Rate (%)')
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(age_dr.values):
    axes[0].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=8)

emp_dr = eda.groupby('employment_group', observed=True)['TARGET'].mean() * 100
colors_emp = [PALETTE['risk'] if v > 9 else PALETTE['neutral'] if v > 6
              else PALETTE['safe'] for v in emp_dr.values]
emp_dr.plot(kind='bar', ax=axes[1], color=colors_emp, edgecolor='white')
axes[1].set_title('Default Rate by Employment Stability\n(2nd strongest behavioural — corr 0.058)',
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

# Plot 2: Risk heatmap
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

# Plot 3: Behavioural features (with NaN handling for matplotlib quirk)
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
            v0 = eda.loc[eda['TARGET']==0, feat].dropna()
            v1 = eda.loc[eda['TARGET']==1, feat].dropna()
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

# Plot 4: Feature correlation ranking
corr_cols = [c for c in eda.select_dtypes(include=[np.number]).columns if c != 'TARGET']
top_corr = (eda[corr_cols + ['TARGET']].corr()['TARGET']
            .drop('TARGET').abs().nlargest(20).sort_values())
plt.figure(figsize=(10, 7))
top_corr.plot(kind='barh', color=PALETTE['accent'], edgecolor='white')
plt.title('Top 20 Features by Correlation with Default',
          fontweight='bold')
plt.xlabel('|Pearson Correlation with TARGET|')
plt.tight_layout()
plt.savefig(f'{FIGURES}/04_feature_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

log.info("  EDA figures saved")
del eda; gc.collect()


# ── MODEL TRAINING ────────────────────────────────────────────────────────────
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


# ── EVALUATION ────────────────────────────────────────────────────────────────
log.info("\n  Evaluating models...")

results = {}
for name, pipe in models.items():
    yp  = pipe.predict_proba(X_val)[:, 1]
    ypb = pipe.predict(X_val)
    rep = classification_report(y_val, ypb, output_dict=True, zero_division=0)
    results[name] = {
        'AUC_ROC'          : round(roc_auc_score(y_val, yp),           4),
        'Avg_Precision'    : round(average_precision_score(y_val, yp), 4),
        'Recall_default'   : round(rep['1']['recall'],                  4),
        'Precision_default': round(rep['1']['precision'],               4),
        'F1_default'       : round(rep['1']['f1-score'],                4),
        'y_proba'          : yp
    }
    log.info(f"  {name:<22} AUC={results[name]['AUC_ROC']:.4f}  "
             f"Recall={results[name]['Recall_default']:.4f}")

best_name  = max(results, key=lambda k: results[k]['AUC_ROC'])
best_model = models[best_name]
log.info(f"\n  Best model: {best_name} (AUC={results[best_name]['AUC_ROC']})")

(pd.DataFrame({k: {m: v for m, v in v.items() if m != 'y_proba'}
               for k, v in results.items()}).T
 .to_csv(f'{PROCESSED}/model_results.csv'))

# ROC + PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot([0,1],[0,1],'k--', lw=1, label='Random (0.50)')
for (nm, res), col in zip(results.items(), list(PALETTE.values())):
    fpr, tpr, _ = roc_curve(y_val, res['y_proba'])
    axes[0].plot(fpr, tpr, lw=2, color=col, label=f"{nm} ({res['AUC_ROC']:.4f})")
axes[0].set(title='ROC Curves', xlabel='False Positive Rate', ylabel='True Positive Rate')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

for (nm, res), col in zip(results.items(), list(PALETTE.values())):
    p, r, _ = precision_recall_curve(y_val, res['y_proba'])
    axes[1].plot(r, p, lw=2, color=col, label=f"{nm} (AP={res['Avg_Precision']:.4f})")
axes[1].axhline(y_val.mean(), color='k', linestyle='--', lw=1, label='Baseline')
axes[1].set(title='Precision-Recall Curves', xlabel='Recall', ylabel='Precision')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES}/05_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance
try:
    step      = best_model.named_steps['model']
    feat_names = X_train.columns.tolist()
    if hasattr(step, 'feature_importances_'):
        fi = pd.Series(step.feature_importances_, index=feat_names).nlargest(30).sort_values()
    elif hasattr(step, 'coef_'):
        fi = pd.Series(np.abs(step.coef_[0]), index=feat_names).nlargest(30).sort_values()
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

# Confusion matrix
y_pred_best  = best_model.predict(X_val)
y_proba_best = results[best_name]['y_proba']
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred_best),
    display_labels=['Non-Default', 'Default']).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES}/07_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# Lift chart
val_df = X_val.copy()
val_df['TARGET']     = y_val.values
val_df['risk_score'] = y_proba_best
sdf = val_df.sort_values('risk_score', ascending=False).reset_index(drop=True)
cum = sdf['TARGET'].cumsum() / y_val.sum() * 100
pop = np.arange(1, len(sdf)+1) / len(sdf) * 100
plt.figure(figsize=(9, 6))
plt.plot(pop, cum, lw=2.5, color=PALETTE['accent'], label='Model Lift Curve')
plt.plot([0,100],[0,100],'k--', lw=1.5, label='Random Baseline')
plt.fill_between(pop, cum, pop, alpha=0.1, color=PALETTE['accent'])
plt.xlabel('% Customers Targeted (by risk score)')
plt.ylabel('% Defaults Captured')
plt.title('Cumulative Gain / Lift Chart', fontweight='bold')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES}/08_lift_chart.png', dpi=150, bbox_inches='tight')
plt.close()

top20 = val_df.nlargest(int(len(val_df)*0.2), 'risk_score')
lift  = top20['TARGET'].sum() / y_val.sum() * 100
log.info(f"\n  Business lift: top 20% captures {lift:.1f}% of defaults ({lift/20:.1f}x)")


# ── THRESHOLD SELECTION ───────────────────────────────────────────────────────
log.info("  Selecting threshold (F-beta beta=2.5)...")

thr_rows = []
for t in np.arange(0.05, 0.91, 0.05):
    yp_t = (y_proba_best >= t).astype(int)
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
         f"(recall={thr_df.loc[opt_idx,'recall']:.3f}, "
         f"precision={thr_df.loc[opt_idx,'precision']:.3f})")

plt.figure(figsize=(9, 5))
plt.plot(thr_df.threshold, thr_df.precision, color=PALETTE['safe'],   marker='o', label='Precision')
plt.plot(thr_df.threshold, thr_df.recall,    color=PALETTE['risk'],   marker='o', label='Recall')
plt.plot(thr_df.threshold, thr_df.f1,        color=PALETTE['neutral'],marker='o', label='F1')
plt.plot(thr_df.threshold, thr_df.fbeta, color=PALETTE['accent'],
         marker='o', lw=2.5, label=f'F-beta (β={DECISION_BETA})')
plt.axvline(opt_thr, color='gray', linestyle='--', label=f'Chosen = {opt_thr}')
plt.title(f'Threshold Selection — Default Class\n'
          f'F-beta (β={DECISION_BETA}) reflects ~8x cost asymmetry of missed defaults',
          fontweight='bold')
plt.xlabel('Decision Threshold')
plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGURES}/09_threshold_tuning.png', dpi=150, bbox_inches='tight')
plt.close()

# Cross-validation (best model only)
log.info(f"\n  Cross-validation ({CV_FOLDS}-fold, {best_name})...")
sample_n = min(50_000, len(X_train))
X_cv = X_train.sample(sample_n, random_state=RANDOM_STATE)
y_cv = y_train.loc[X_cv.index]
scores = cross_val_score(best_model, X_cv, y_cv, cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1)
log.info(f"  CV-AUC: {scores.mean():.4f} ± {scores.std():.4f}")


# ── MLFLOW ────────────────────────────────────────────────────────────────────
if MLFLOW_AVAILABLE:
    mlflow.set_experiment("home_credit_credit_scoring")
    for name, res in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({'model': name, 'test_size': TEST_SIZE,
                               'random_state': RANDOM_STATE,
                               'missing_drop_pct': MISSING_DROP_PCT,
                               'decision_beta': DECISION_BETA,
                               'decision_threshold': opt_thr})
            mlflow.log_metrics({k: v for k, v in res.items() if k != 'y_proba'})
    log.info("  MLflow logged — view with: mlflow ui")


# ── EXPORT ────────────────────────────────────────────────────────────────────
log.info("\n[8/8] Exporting...")

sql_cols = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
    'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'income_credit_ratio', 'employment_age_ratio', 'annuity_income_ratio',
    'bur_total_debt', 'bur_num_credits', 'bur_max_overdue',
    'prev_num_applications', 'prev_approval_rate',
    'inst_late_rate', 'inst_days_late_mean',
    'cc_utilisation', 'cc_dpd_max',
    'pos_sk_dpd_max', 'pos_completion_rate', 'TARGET'
]
sql_out = [c for c in sql_cols if c in train_df.columns]
train_df[sql_out].to_csv(f'{PROCESSED}/credit_data_sql.csv', index=False)
train_df.to_csv(f'{PROCESSED}/final_enriched_train.csv', index=False)

log.info(f"  Saved: credit_data_sql.csv ({len(sql_out)} columns, {len(train_df):,} rows)")
log.info(f"  Saved: final_enriched_train.csv ({train_df.shape[1]} columns)")

log.info("\n" + "=" * 65)
log.info("  PIPELINE COMPLETE")
log.info(f"  Best model        : {best_name}")
log.info(f"  AUC-ROC           : {results[best_name]['AUC_ROC']}")
log.info(f"  Decision threshold: {opt_thr} (F-beta β={DECISION_BETA})")
log.info(f"  Business lift     : {lift:.1f}% of defaults in top 20% ({lift/20:.1f}x random)")
log.info("=" * 65)