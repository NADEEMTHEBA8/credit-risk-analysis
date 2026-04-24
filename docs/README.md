# Home Credit Default Risk — End-to-End Data Project

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-0.7839_AUC-orange?style=flat)](https://lightgbm.readthedocs.io/)

## Project Overview

307,511 loan applicants. 8% will default. Identifying which ones before approving their loan saves millions — but rejecting good customers costs business.

This project builds a complete credit risk scoring system that joins **8 relational tables** (60M+ rows), engineers 300+ features from customer behaviour history, trains and compares 4 ML models, and surfaces insights through SQL analysis in PostgreSQL.

---

## Key Results

| Metric | Value |
|---|---|
| Best model | LightGBM |
| AUC-ROC | **0.7839** |
| Average Precision | 0.2814 |
| Business lift | **2.8× over random** |
| Top 20% risk score captures | **56.4% of all defaults** |

> Reviewing the top 20% of risk-scored applicants catches 56% of all defaults. For a team of 10 credit officers, the model delivers the efficiency of 28.

---

## Surprising Finding

I expected income and external credit scores to be the strongest predictors. The data said otherwise.

| Feature | Correlation | Default Rate Range |
|---|---|---|
| Age (`DAYS_BIRTH`) | 0.078 | 4.9% (60+) → 11.5% (18-29) |
| CC Utilisation | 0.075 | 8.3% normal → **25.9% over-limit** |
| Installment Late Rate | 0.070 | 6.8% → 16.4% |
| Employment Stability | 0.058 | 3.1% → 10.5% |
| `income_credit_ratio` | **0.002** | 6.6% → 8.8% — nearly flat |

The SQL analysis and composite risk score are built around age and employment — not income.

---

## Architecture

```
8 Raw CSV Files (60M+ rows)
        │
        ├── application_train/test  ─────────────────────────────────┐
        ├── bureau + bureau_balance → aggregate per customer ────────┤
        ├── previous_application   → aggregate per customer ────────┤  307,511-row
        ├── POS_CASH_balance       → aggregate per customer ────────┤  Feature Matrix
        ├── credit_card_balance    → aggregate per customer ────────┤  (300+ features)
        └── installments_payments  → aggregate per customer ────────┘
                                                                     │
                        ┌────────────────────────────────────────────┤
                        │                                            │
                   SQL Analysis                               ML Models
                   (PostgreSQL)                       LR · RF · XGB · LightGBM
                   22 queries                         AUC 0.7839 · 2.8× lift
```

---

## Repository Structure

```
credit-risk-analysis/
│
├── src/
│   ├── credit_risk_pipeline_final.py   # Full ML pipeline (8 stages)
│   └── dashboard_export.py             # Generates analysis-ready CSV
│
├── sql/
│   └── credit_risk_analysis_final.sql  # 22 SQL sections (PostgreSQL)
│
├── data/
│   ├── raw/                            # Kaggle CSVs go here (git-ignored)
│   └── processed/                      # Pipeline outputs (git-ignored)
│
├── figures/                            # Generated plots (git-ignored)
│
├── docs/
│   └── setup_guide.md                  # Step-by-step setup instructions
│
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR-USERNAME/credit-risk-analysis.git
cd credit-risk-analysis

# 2. Install
pip install -r requirements.txt

# 3. Download data from Kaggle → place all CSVs in data/raw/
# https://www.kaggle.com/competitions/home-credit-default-risk/data

# 4. Run pipeline
python src/credit_risk_pipeline_final.py

# 5. Open pgAdmin → run sql/credit_risk_analysis_final.sql
```

---

## Model Comparison

| Model | AUC-ROC | Avg Precision | Recall (Default) |
|---|---|---|---|
| **LightGBM** | **0.7839** | **0.2814** | **0.6689** |
| XGBoost | 0.7830 | 0.2802 | 0.6437 |
| Logistic Regression | 0.7721 | 0.2570 | 0.6961 |
| Random Forest | 0.7513 | 0.2305 | 0.4872 |

**Why LightGBM?** 3× faster than XGBoost, only 0.0009 AUC difference. Also handles high-cardinality categoricals natively.

**Why not accuracy?** 92% of customers did not default. A model predicting "never default" achieves 92% accuracy while catching zero defaults. AUC-ROC measures ranking ability regardless of class distribution.

**Threshold selection:** F-beta (β=2.5) instead of F1. A missed default costs ~8× more than a false positive at a bank. F1 treats both errors equally — wrong for credit decisions.

---

## Key Technical Decisions

| Decision | Why |
|---|---|
| LEFT JOIN for secondary tables | INNER JOIN dropped 37,000 first-time borrowers — recall fell 0.69 → 0.61 |
| `reduce_memory()` downcast | Cut RAM by ~40% — bureau_balance (27M rows) caused OOM without it |
| Two-level bureau aggregation | bureau_balance → loan level → customer level. Cannot skip the first level |
| F-beta threshold (β=2.5) | Missed default costs ~8× more than false positive at a bank |
| GradientBoosting removed | Recall=0.045 in actual run, 5× slower than LightGBM |

---

## Known Data Quirks

- `bur_total_debt` can be negative — customer overpaid a loan (valid, not an error)
- `cc_utilisation` can exceed 1.0 — customer is over credit limit (these 1,042 customers have 25.9% default rate)
- `inst_days_late_mean` can be negative — customer paid before the due date

---

## SQL Analysis Covers

22 sections including: portfolio KPIs · age group analysis · employment stability · age × employment heatmap · payment behaviour segmentation · credit card utilisation · bureau history · 3-way risk matrix · composite risk score (3-CTE pattern) · NTILE / RANK / PARTITION BY / PERCENT_RANK window functions · PERCENTILE_CONT for median · stored function · reusable view · EXPLAIN ANALYZE · data quality checks

---

## Technologies

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Experiment Tracking | MLflow |
| Database | PostgreSQL 14+ |

---

## Dataset

Kaggle — Home Credit Default Risk
https://www.kaggle.com/competitions/home-credit-default-risk

307,511 applicants · 8 relational tables · Binary classification
