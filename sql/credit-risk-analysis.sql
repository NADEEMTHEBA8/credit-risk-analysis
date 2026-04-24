-- ================================================================
-- Home Credit Default Risk — SQL Analysis
-- Database  : PostgreSQL 14+
-- Source    : credit_data_sql.csv (307,511 rows, 20 columns)
-- ================================================================
-- Built after analysing the actual output data from the Python pipeline.
--
-- KEY FINDING that shaped this file:
--   income_credit_ratio is the 17th strongest predictor of default
--   (Pearson correlation = 0.0018). The earlier SQL version used it
--   as the primary segmentation basis — that was wrong.
--
--   Actual top predictors from the data:
--     1. DAYS_BIRTH (age)          corr = 0.078  range: 4.9% to 11.5%
--     2. cc_utilisation            corr = 0.075  over-limit = 25.9%
--     3. inst_late_rate            corr = 0.070  range: 6.8% to 16.4%
--     4. DAYS_EMPLOYED             corr = 0.064
--     5. prev_approval_rate        corr = 0.063  refused = 11.1%
--     6. employment_age_ratio      corr = 0.058  range: 3.1% to 10.5%
--    17. income_credit_ratio       corr = 0.002  range: 6.6% to 8.8%
--
--   The SQL analysis is structured around what the data actually shows,
--   not what we assumed before running the pipeline.
--
-- KNOWN DATA QUIRKS (from analysis):
--   bur_total_debt can be negative (customer overpaid a loan)
--   cc_utilisation can exceed 1.0 (customer is over credit limit)
--   inst_days_late_mean can be negative (customer paid early)
--
-- RUN ORDER:
--   (1) Schema setup block   — creates table and indexes
--   (2) Import CSV            — right-click table → Import/Export
--   (3) Segmentation block   — adds and fills 5 label columns
--   (4) Analysis queries     — run one block at a time, press F5
-- ================================================================


-- ════════════════════════════════════════════════════════════════
-- SCHEMA SETUP
-- ════════════════════════════════════════════════════════════════

DROP TABLE IF EXISTS application CASCADE;

CREATE TABLE application (

    -- Application-level fields
    amt_income_total       NUMERIC,   -- Annual income
    amt_credit             NUMERIC,   -- Requested loan amount
    amt_goods_price        NUMERIC,   -- Goods price (NULL for cash loans)
    days_birth             NUMERIC,   -- Age in days (positive — converted in pipeline)
    days_employed          NUMERIC,   -- Employment days (positive; NULL = unemployed)

    -- Engineered ratios from the Python pipeline
    -- Note: income_credit_ratio has very low correlation (0.002) with default.
    -- It is kept here for completeness but is not the primary analysis signal.
    income_credit_ratio    NUMERIC,   -- income / credit — expected to be strong, was not
    employment_age_ratio   NUMERIC,   -- employed_days / birth_days — 6th strongest (0.058)
    annuity_income_ratio   NUMERIC,   -- monthly payment / income

    -- Bureau signals (from bureau.csv + bureau_balance.csv)
    -- bur_total_debt can be negative due to overpayments — this is not an error.
    bur_total_debt         NUMERIC,   -- total outstanding debt at other banks (can be negative)
    bur_num_credits        NUMERIC,   -- count of past bureau credits
    bur_max_overdue        NUMERIC,   -- largest overdue amount; even one matters (16.2% default)

    -- Previous application signals
    prev_num_applications  NUMERIC,   -- total past Home Credit applications
    prev_approval_rate     NUMERIC,   -- fraction approved; 0-0.5 = 11.1% default rate

    -- Installment payment behaviour — 3rd strongest signal (corr=0.070)
    inst_late_rate         NUMERIC,   -- fraction of payments made late (0 to 0.75 in data)
    inst_days_late_mean    NUMERIC,   -- avg days late per payment (negative = paid early)

    -- Credit card signals — 2nd strongest (corr=0.075)
    -- cc_utilisation can exceed 1.0 when customer is over credit limit.
    -- Over-limit customers have 25.9% default rate vs 8.3% for normal users.
    cc_utilisation         NUMERIC,   -- balance / limit; can exceed 1.0
    cc_dpd_max             NUMERIC,   -- worst DPD event on credit card

    -- POS cash loan signals
    pos_sk_dpd_max         NUMERIC,   -- worst DPD on a POS/cash loan
    pos_completion_rate    NUMERIC,   -- fraction of POS loans fully completed

    target                 NUMERIC    -- 1 = defaulted, 0 = repaid
);

-- Index on strongest predictors and common filter columns
CREATE INDEX idx_days_birth     ON application(days_birth);        -- strongest predictor
CREATE INDEX idx_cc_util        ON application(cc_utilisation);    -- 2nd strongest
CREATE INDEX idx_inst_late      ON application(inst_late_rate);    -- 3rd strongest
CREATE INDEX idx_emp_ratio      ON application(employment_age_ratio);
CREATE INDEX idx_prev_approval  ON application(prev_approval_rate);
CREATE INDEX idx_target         ON application(target);
CREATE INDEX idx_income         ON application(amt_income_total);
CREATE INDEX idx_credit         ON application(amt_credit);

-- After running this block, import credit_data_sql.csv:
--   Right-click "application" in left panel → Import/Export Data
--   Format: csv | Header: ON | Delimiter: ,
--   Expected: "Successfully completed. 307511 rows"


-- ════════════════════════════════════════════════════════════════
-- SEGMENTATION SETUP
-- ════════════════════════════════════════════════════════════════

ALTER TABLE application
    ADD COLUMN IF NOT EXISTS income_group      VARCHAR(20),
    ADD COLUMN IF NOT EXISTS loan_size         VARCHAR(20),
    ADD COLUMN IF NOT EXISTS employment_group  VARCHAR(20),
    ADD COLUMN IF NOT EXISTS age_group         VARCHAR(10),
    ADD COLUMN IF NOT EXISTS risk_level        VARCHAR(20);

-- Single UPDATE pass — one table scan fills all 5 columns.
-- Thresholds match the Python pipeline pd.cut bins exactly so
-- SQL and Python segment assignments are always identical.
UPDATE application SET

    income_group = CASE
        WHEN amt_income_total < 100000  THEN 'Low'
        WHEN amt_income_total < 200000  THEN 'Medium'
        ELSE                                 'High'
    END,

    loan_size = CASE
        WHEN amt_credit < 100000  THEN 'Small'
        WHEN amt_credit < 500000  THEN 'Medium'
        ELSE                           'Large'
    END,

    -- employment_age_ratio = days_employed / days_birth
    -- Unstable (<0.1): 10.5% default rate
    -- Stable   (>0.6): 3.1% default rate — 3.5x difference
    employment_group = CASE
        WHEN employment_age_ratio IS NULL
          OR employment_age_ratio < 0.1  THEN 'Unstable'
        WHEN employment_age_ratio < 0.3  THEN 'Short-term'
        WHEN employment_age_ratio < 0.6  THEN 'Moderate'
        ELSE                                  'Stable'
    END,

    -- Age group. Days_birth / 365 = age in years.
    -- 18-29: 11.5% default rate vs 60+: 4.9% — strongest single predictor.
    age_group = CASE
        WHEN (days_birth / 365) < 30  THEN '18-29'
        WHEN (days_birth / 365) < 40  THEN '30-39'
        WHEN (days_birth / 365) < 50  THEN '40-49'
        WHEN (days_birth / 365) < 60  THEN '50-59'
        ELSE                               '60+'
    END,

    -- income_credit_ratio segments kept for compatibility with pipeline naming.
    -- Note: actual default rates across these bands are nearly flat (6.6% to 8.8%).
    -- Do not use risk_level alone as a risk signal — use age/employment instead.
    risk_level = CASE
        WHEN income_credit_ratio < 0.3  THEN 'High Risk'
        WHEN income_credit_ratio < 0.6  THEN 'Medium Risk'
        ELSE                                 'Low Risk'
    END;

CREATE INDEX IF NOT EXISTS idx_income_group ON application(income_group);
CREATE INDEX IF NOT EXISTS idx_age_group    ON application(age_group);
CREATE INDEX IF NOT EXISTS idx_emp_group    ON application(employment_group);
CREATE INDEX IF NOT EXISTS idx_risk_level   ON application(risk_level);

-- Quick verify after UPDATE
SELECT age_group, employment_group,
       COUNT(*)                                   AS customers,
       ROUND(AVG(target) * 100::NUMERIC, 2)       AS default_rate_pct
FROM application
GROUP BY age_group, employment_group
ORDER BY default_rate_pct DESC
LIMIT 5;
-- Expected top rows: 18-29 / Unstable ~12.5%, 30-39 / Unstable ~11.9%


-- ════════════════════════════════════════════════════════════════
-- DATA VERIFICATION
-- ════════════════════════════════════════════════════════════════

SELECT
    COUNT(*)                                      AS total_rows,        -- expect 307511
    SUM(target)                                   AS total_defaults,    -- expect ~24825
    ROUND(AVG(target) * 100::NUMERIC, 2)          AS default_rate_pct, -- expect 8.07
    ROUND(AVG(bur_total_debt)::NUMERIC, 0)        AS avg_bureau_debt,  -- note: can be negative
    COUNT(*) FILTER (WHERE bur_total_debt < 0)    AS negative_debt_rows, -- ~1296 expected
    COUNT(*) FILTER (WHERE cc_utilisation > 1.0)  AS over_limit_rows     -- ~1042 expected
FROM application;


-- ════════════════════════════════════════════════════════════════
-- PORTFOLIO KPI OVERVIEW
-- ════════════════════════════════════════════════════════════════

SELECT
    COUNT(*)                                           AS total_customers,
    SUM(target)                                        AS total_defaults,
    COUNT(*) - SUM(target)                             AS total_non_defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)               AS default_rate_pct,
    ROUND(AVG(amt_income_total)::NUMERIC, 0)           AS avg_income,
    ROUND(AVG(amt_credit)::NUMERIC, 0)                 AS avg_loan,
    ROUND((SUM(amt_credit) / 1e9)::NUMERIC, 2)         AS total_portfolio_billion,
    ROUND(AVG(employment_age_ratio)::NUMERIC, 4)       AS avg_employment_stability,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)             AS avg_installment_late_rate,
    ROUND(AVG(cc_utilisation)::NUMERIC, 4)             AS avg_cc_utilisation
FROM application;


-- ════════════════════════════════════════════════════════════════
-- PRIMARY ANALYSIS: AGE GROUP
-- Strongest predictor — correlation 0.078 with default.
-- Default rate drops from 11.5% (18-29) to 4.9% (60+).
-- ════════════════════════════════════════════════════════════════

SELECT
    age_group,
    COUNT(*)                                       AS customers,
    SUM(target)                                    AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)           AS default_rate_pct,
    ROUND(AVG(amt_income_total)::NUMERIC, 0)       AS avg_income,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)         AS avg_late_rate,
    ROUND(AVG(employment_age_ratio)::NUMERIC, 4)   AS avg_employment_stability,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()
    ::NUMERIC, 2)                                  AS pct_of_portfolio
FROM application
GROUP BY age_group
ORDER BY default_rate_pct DESC;


-- ════════════════════════════════════════════════════════════════
-- PRIMARY ANALYSIS: EMPLOYMENT STABILITY
-- 6th strongest — but shows 3.5x range (3.1% to 10.5%).
-- Combined with age, this gives the best segmentation in the dataset.
-- ════════════════════════════════════════════════════════════════

SELECT
    employment_group,
    COUNT(*)                                       AS customers,
    SUM(target)                                    AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)           AS default_rate_pct,
    ROUND(AVG(amt_income_total)::NUMERIC, 0)       AS avg_income,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)         AS avg_late_rate,
    ROUND(AVG(employment_age_ratio)::NUMERIC, 4)   AS avg_emp_ratio_mean
FROM application
GROUP BY employment_group
ORDER BY default_rate_pct DESC;


-- ════════════════════════════════════════════════════════════════
-- KEY FINDING: AGE × EMPLOYMENT COMBINED
-- These two features together produce the widest default rate range
-- in the entire dataset: 3.05% (Stable + 60+) to 12.47% (Unstable + 18-29).
-- ════════════════════════════════════════════════════════════════

SELECT
    age_group,
    employment_group,
    COUNT(*)                                       AS customers,
    SUM(target)                                    AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)           AS default_rate_pct,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)         AS avg_late_rate
FROM application
GROUP BY age_group, employment_group
HAVING COUNT(*) > 500       -- minimum sample for statistical reliability
ORDER BY default_rate_pct DESC;


-- ════════════════════════════════════════════════════════════════
-- PAYMENT BEHAVIOUR ANALYSIS
-- inst_late_rate: 3rd strongest predictor (corr = 0.070).
-- "Usually Late" (>50%) has 16.4% default rate — 2.4x portfolio average.
-- ════════════════════════════════════════════════════════════════

WITH pay_segments AS (
    SELECT target, amt_income_total, employment_age_ratio,
        CASE
            WHEN inst_late_rate IS NULL  THEN '1. No Payment History'
            WHEN inst_late_rate = 0      THEN '2. Always On Time'
            WHEN inst_late_rate < 0.10   THEN '3. Rarely Late    (<10%)'
            WHEN inst_late_rate < 0.30   THEN '4. Sometimes Late (10-30%)'
            WHEN inst_late_rate < 0.50   THEN '5. Often Late     (30-50%)'
            ELSE                              '6. Usually Late   (>50%)'
        END AS payment_segment
    FROM application
)
SELECT
    payment_segment,
    COUNT(*)                                          AS customers,
    SUM(target)                                       AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)              AS default_rate_pct,
    ROUND(AVG(amt_income_total)::NUMERIC, 0)          AS avg_income,
    ROUND(AVG(employment_age_ratio)::NUMERIC, 4)      AS avg_employment_stability
FROM pay_segments
GROUP BY payment_segment
ORDER BY payment_segment;
-- Expected: monotonic increase from "Always On Time" (6.8%) to "Usually Late" (16.4%)


-- ════════════════════════════════════════════════════════════════
-- CREDIT CARD UTILISATION ANALYSIS
-- 2nd strongest predictor (corr = 0.075).
-- Customers OVER their credit limit (util > 1.0) have 25.9% default rate.
-- This is the single highest default rate of any segment in this analysis.
-- ════════════════════════════════════════════════════════════════

WITH cc_segments AS (
    SELECT target,
        CASE
            WHEN cc_utilisation IS NULL    THEN '1. No Credit Card'
            WHEN cc_utilisation = 0        THEN '2. Unused Card (0%)'
            WHEN cc_utilisation < 0.30     THEN '3. Low Use      (1-30%)'
            WHEN cc_utilisation < 0.60     THEN '4. Moderate     (30-60%)'
            WHEN cc_utilisation < 0.90     THEN '5. High         (60-90%)'
            WHEN cc_utilisation <= 1.0     THEN '6. Near-Limit   (90-100%)'
            ELSE                                '7. OVER LIMIT   (>100%)'  -- 25.9% default rate
        END AS cc_segment
    FROM application
)
SELECT
    cc_segment,
    COUNT(*)                                   AS customers,
    SUM(target)                                AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)       AS default_rate_pct
FROM cc_segments
GROUP BY cc_segment
ORDER BY cc_segment;


-- ════════════════════════════════════════════════════════════════
-- PREVIOUS APPLICATION HISTORY
-- 5th strongest predictor (corr = 0.063).
-- "Mostly Refused" bucket (0-50% approval) has 11.1% default rate.
-- Note: "Always Refused" (approval_rate=0) only has 8.7% — the
-- bucket with 0-50% approval is actually riskier than 0%.
-- ════════════════════════════════════════════════════════════════

WITH prev_segments AS (
    SELECT target,
        CASE
            WHEN prev_num_applications IS NULL
              OR prev_num_applications = 0    THEN '1. First-Time Applicant'
            WHEN prev_approval_rate >= 0.9    THEN '2. Mostly Approved (>90%)'
            WHEN prev_approval_rate >= 0.5    THEN '3. Mixed (50-90%)'
            WHEN prev_approval_rate > 0       THEN '4. Mostly Refused (1-50%)'  -- 11.1% rate
            ELSE                                   '5. Always Refused (0%)'     -- 8.7% rate
        END AS prev_segment
    FROM application
)
SELECT
    prev_segment,
    COUNT(*)                                   AS customers,
    SUM(target)                                AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)       AS default_rate_pct
FROM prev_segments
GROUP BY prev_segment
ORDER BY prev_segment;


-- ════════════════════════════════════════════════════════════════
-- BUREAU HISTORY IMPACT
-- bur_max_overdue > 0: 16.2% default rate (vs 8.0% clean)
-- Note: bur_total_debt can be negative (customer overpaid).
--       GREATEST(bur_total_debt, 0) used where debt amount is needed.
-- ════════════════════════════════════════════════════════════════

WITH bureau_segments AS (
    SELECT target, amt_income_total,
        CASE
            WHEN bur_num_credits IS NULL
              OR bur_num_credits = 0    THEN '1. No Bureau History'
            WHEN bur_max_overdue > 0    THEN '2. Has Overdue at Other Banks'  -- 16.2%
            ELSE                             '3. Clean Bureau Record'
        END AS bureau_segment
    FROM application
)
SELECT
    bureau_segment,
    COUNT(*)                                       AS customers,
    SUM(target)                                    AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)           AS default_rate_pct,
    ROUND(AVG(amt_income_total)::NUMERIC, 0)       AS avg_income
FROM bureau_segments
GROUP BY bureau_segment
ORDER BY default_rate_pct DESC;


-- ════════════════════════════════════════════════════════════════
-- INCOME GROUP (WEAK SIGNAL — SHOWN FOR COMPLETENESS)
-- Default rates: Low=8.2%, Medium=8.6%, High=7.1%.
-- Range is only 1.5 percentage points.
-- Income alone is a poor predictor of default.
-- ════════════════════════════════════════════════════════════════

WITH income_summary AS (
    SELECT
        income_group,
        COUNT(*)                                      AS customers,
        SUM(target)                                   AS defaults,
        ROUND(AVG(target) * 100::NUMERIC, 2)          AS default_rate_pct,
        ROUND(AVG(amt_income_total)::NUMERIC, 0)      AS avg_income,
        ROUND(AVG(inst_late_rate)::NUMERIC, 4)        AS avg_late_rate,
        ROUND(AVG(employment_age_ratio)::NUMERIC, 4)  AS avg_employment_stability
    FROM application
    GROUP BY income_group
)
SELECT *,
    ROUND(customers * 100.0 / SUM(customers) OVER ()::NUMERIC, 2) AS pct_of_portfolio
FROM income_summary
ORDER BY default_rate_pct DESC;
-- Expected: very flat rates (~7-9%) across all groups.
-- Compare to age groups (4.9% to 11.5%) to see why income is a weaker signal.


-- ════════════════════════════════════════════════════════════════
-- NTILE RISK BUCKETS — BY AGE (STRONGEST PREDICTOR)
-- Using DAYS_BIRTH instead of income_credit_ratio because age
-- has 44x higher correlation with default.
-- Higher bucket number = older = lower risk.
-- ════════════════════════════════════════════════════════════════

WITH age_buckets AS (
    SELECT
        target,
        days_birth,
        inst_late_rate,
        cc_utilisation,
        employment_age_ratio,
        NTILE(5) OVER (ORDER BY days_birth ASC) AS age_bucket
        -- ASC = younger first = highest risk in bucket 1
    FROM application
    WHERE days_birth IS NOT NULL
)
SELECT
    age_bucket,
    CASE age_bucket
        WHEN 1 THEN 'Youngest 20%   — highest risk'
        WHEN 2 THEN 'Young-mid 20%'
        WHEN 3 THEN 'Middle 20%'
        WHEN 4 THEN 'Older-mid 20%'
        WHEN 5 THEN 'Oldest 20%     — lowest risk'
    END                                               AS age_label,
    ROUND(MIN(days_birth / 365.0)::NUMERIC, 1)        AS min_age_years,
    ROUND(MAX(days_birth / 365.0)::NUMERIC, 1)        AS max_age_years,
    COUNT(*)                                          AS customers,
    SUM(target)                                       AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)              AS default_rate_pct,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)            AS avg_late_rate,
    ROUND(AVG(cc_utilisation)::NUMERIC, 4)            AS avg_cc_util
FROM age_buckets
GROUP BY age_bucket
ORDER BY age_bucket;


-- ════════════════════════════════════════════════════════════════
-- DEFAULTER vs NON-DEFAULTER SIDE-BY-SIDE PROFILE
-- PERCENTILE_CONT(0.5) = exact median (robust to skewed income data).
-- ════════════════════════════════════════════════════════════════

SELECT
    CASE WHEN target = 1 THEN 'Defaulted' ELSE 'Non-Default' END  AS customer_type,
    COUNT(*)                                                       AS customers,
    ROUND(AVG(days_birth / 365.0)::NUMERIC, 1)                    AS avg_age_years,
    ROUND(AVG(employment_age_ratio)::NUMERIC, 4)                  AS avg_employment_stability,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)                        AS avg_late_rate,
    ROUND(AVG(cc_utilisation)::NUMERIC, 4)                        AS avg_cc_util,
    ROUND(AVG(amt_income_total)::NUMERIC, 0)                      AS avg_income,
    ROUND(AVG(amt_credit)::NUMERIC, 0)                            AS avg_loan,
    ROUND(AVG(prev_approval_rate)::NUMERIC, 4)                    AS avg_prev_approval_rate,
    ROUND(AVG(bur_max_overdue)::NUMERIC, 2)                       AS avg_bureau_overdue,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY days_birth / 365.0) ::NUMERIC
    , 1)                                                           AS median_age_years,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amt_income_total) ::NUMERIC
    , 0)                                                           AS median_income
FROM application
GROUP BY target
ORDER BY target;


-- ════════════════════════════════════════════════════════════════
-- 3-WAY RISK MATRIX: AGE × EMPLOYMENT × PAYMENT BEHAVIOUR
-- Using the three strongest independent signals.
-- HAVING COUNT(*) > 500 removes unreliable small segments.
-- ════════════════════════════════════════════════════════════════

WITH pay_flag AS (
    SELECT *,
        CASE
            WHEN inst_late_rate IS NULL OR inst_late_rate = 0 THEN 'Good Payer'
            WHEN inst_late_rate < 0.20                        THEN 'Occasional Late'
            ELSE                                                   'Frequent Late'
        END AS payment_behaviour
    FROM application
)
SELECT
    age_group,
    employment_group,
    payment_behaviour,
    COUNT(*)                                     AS customers,
    SUM(target)                                  AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)         AS default_rate_pct
FROM pay_flag
GROUP BY age_group, employment_group, payment_behaviour
HAVING COUNT(*) > 500
ORDER BY default_rate_pct DESC
LIMIT 15;


-- ════════════════════════════════════════════════════════════════
-- CUSTOMER RANKING — PARTITION BY + MULTIPLE WINDOW FUNCTIONS
-- RANK() OVER (ORDER BY ...)              — global rank
-- RANK() OVER (PARTITION BY ... ORDER BY) — rank within group
-- PERCENT_RANK() * 100                    — normalised score 0-100
--
-- Ranking by days_birth ASC (youngest = riskiest = rank 1)
-- ════════════════════════════════════════════════════════════════

SELECT
    ROUND(days_birth / 365.0, 1)                     AS age_years,
    ROUND(employment_age_ratio, 4)                   AS emp_stability,
    ROUND(inst_late_rate, 4)                         AS late_rate,
    ROUND(cc_utilisation, 4)                         AS cc_util,
    target,
    age_group,
    employment_group,
    -- Youngest customer globally = rank 1
    RANK() OVER (ORDER BY days_birth ASC)            AS global_age_rank,
    -- Youngest within each employment group
    RANK() OVER (
        PARTITION BY employment_group
        ORDER BY days_birth ASC
    )                                                AS age_rank_in_emp_group,
    -- Percentile: 0 = oldest (safest), 100 = youngest (riskiest)
    ROUND(
        (PERCENT_RANK() OVER (ORDER BY days_birth ASC) * 100)::NUMERIC
    , 2)                                             AS youth_risk_percentile
FROM application
ORDER BY days_birth ASC
LIMIT 20;


-- ════════════════════════════════════════════════════════════════
-- DECILE ANALYSIS — BY EMPLOYMENT STABILITY
-- 10 equal groups; decile 1 = most unstable = highest risk.
-- ════════════════════════════════════════════════════════════════

WITH deciles AS (
    SELECT
        target,
        employment_age_ratio,
        inst_late_rate,
        cc_utilisation,
        NTILE(10) OVER (ORDER BY employment_age_ratio ASC) AS decile
    FROM application
    WHERE employment_age_ratio IS NOT NULL
)
SELECT
    decile,
    ROUND(MIN(employment_age_ratio)::NUMERIC, 4)   AS min_emp_ratio,
    ROUND(MAX(employment_age_ratio)::NUMERIC, 4)   AS max_emp_ratio,
    COUNT(*)                                        AS customers,
    SUM(target)                                     AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)            AS default_rate_pct,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)          AS avg_late_rate
FROM deciles
GROUP BY decile
ORDER BY decile;


-- ════════════════════════════════════════════════════════════════
-- SCALAR SUBQUERY — CUSTOMERS BELOW AVERAGE AGE
-- Younger than average = higher risk group.
-- Inner SELECT runs once, outer query compares each row to it.
-- ════════════════════════════════════════════════════════════════

SELECT
    ROUND(days_birth / 365.0, 1)              AS age_years,
    ROUND(inst_late_rate, 4)                  AS late_rate,
    ROUND(cc_utilisation, 4)                  AS cc_util,
    ROUND(employment_age_ratio, 4)            AS emp_stability,
    target,
    age_group,
    employment_group
FROM application
WHERE days_birth < (SELECT AVG(days_birth) FROM application)  -- younger than average
ORDER BY days_birth ASC
LIMIT 20;


-- ════════════════════════════════════════════════════════════════
-- COMPOSITE RISK SCORE
-- Three CTEs required because window functions cannot appear
-- in GROUP BY directly (computed in final_bands CTE, then grouped).
--
-- Weights updated based on actual correlation analysis:
--   Old (wrong): income_credit_ratio 50%, late_rate 30%, cc_util 20%
--   New (correct): age 30%, late_rate 30%, cc_util 25%, employment 15%
--   income_credit_ratio excluded — correlation too weak to include.
-- ════════════════════════════════════════════════════════════════

WITH signals AS (
    SELECT
        target,
        days_birth,
        inst_late_rate,
        cc_utilisation,
        employment_age_ratio,
        -- NTILE quintile per signal: 5 = worst, 1 = best
        -- Age: younger (lower days_birth) = higher risk = higher quintile
        NTILE(5) OVER (ORDER BY days_birth ASC)                            AS q_age,
        NTILE(5) OVER (ORDER BY COALESCE(inst_late_rate,  0) DESC)         AS q_late_rate,
        NTILE(5) OVER (ORDER BY COALESCE(cc_utilisation,  0) DESC)         AS q_cc_util,
        NTILE(5) OVER (ORDER BY COALESCE(employment_age_ratio, 0) ASC)     AS q_employment
    FROM application
    WHERE days_birth IS NOT NULL
),
composite AS (
    SELECT *,
        ROUND(
            (q_age        * 0.30) +   -- age: strongest predictor, corr=0.078
            (q_late_rate  * 0.30) +   -- payment behaviour, corr=0.070
            (q_cc_util    * 0.25) +   -- cc utilisation, corr=0.075
            (q_employment * 0.15)     -- employment stability, corr=0.058
        , 2) AS composite_score
    FROM signals
),
final_bands AS (
    -- Must be a separate CTE so NTILE result can be used in GROUP BY
    SELECT *,
        NTILE(5) OVER (ORDER BY composite_score DESC) AS final_risk_band
    FROM composite
)
SELECT
    final_risk_band,
    COUNT(*)                                          AS customers,
    SUM(target)                                       AS defaults,
    ROUND(AVG(target)::NUMERIC * 100, 2)              AS default_rate_pct,
    ROUND(AVG(composite_score)::NUMERIC, 2)           AS avg_composite_score,
    ROUND(AVG(days_birth / 365.0)::NUMERIC, 1)        AS avg_age_years,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)            AS avg_late_rate,
    ROUND(AVG(cc_utilisation)::NUMERIC, 4)            AS avg_cc_util
FROM final_bands
GROUP BY final_risk_band
ORDER BY final_risk_band;


-- ════════════════════════════════════════════════════════════════
-- STORED FUNCTION — FLEXIBLE SEGMENT LOOKUP
-- NULL parameter = no filter on that dimension.
-- Named parameter syntax (=>) lets callers skip optional params.
-- ════════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION get_risk_profile(
    p_age_group        VARCHAR DEFAULT NULL,
    p_employment_group VARCHAR DEFAULT NULL,
    p_income_group     VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    seg_age_group        VARCHAR,
    seg_employment_group VARCHAR,
    seg_income_group     VARCHAR,
    total_customers      BIGINT,
    total_defaults       BIGINT,
    default_rate_pct     NUMERIC,
    avg_age_years        NUMERIC,
    avg_late_rate        NUMERIC,
    avg_cc_util          NUMERIC,
    avg_employment       NUMERIC
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        a.age_group::VARCHAR,
        a.employment_group::VARCHAR,
        a.income_group::VARCHAR,
        COUNT(*)::BIGINT                                    AS total_customers,
        SUM(a.target)::BIGINT                               AS total_defaults,
        ROUND(AVG(a.target) * 100::NUMERIC, 2)             AS default_rate_pct,
        ROUND(AVG(a.days_birth / 365.0)::NUMERIC, 1)       AS avg_age_years,
        ROUND(AVG(a.inst_late_rate)::NUMERIC, 4)           AS avg_late_rate,
        ROUND(AVG(a.cc_utilisation)::NUMERIC, 4)           AS avg_cc_util,
        ROUND(AVG(a.employment_age_ratio)::NUMERIC, 4)     AS avg_employment
    FROM application a
    WHERE
        (p_age_group        IS NULL OR a.age_group        = p_age_group)
        AND (p_employment_group IS NULL OR a.employment_group = p_employment_group)
        AND (p_income_group     IS NULL OR a.income_group     = p_income_group)
    GROUP BY a.age_group, a.employment_group, a.income_group
    ORDER BY default_rate_pct DESC;
END;
$$;

-- Usage examples
SELECT * FROM get_risk_profile();                                              -- all segments
SELECT * FROM get_risk_profile(p_age_group => '18-29');                        -- youngest only
SELECT * FROM get_risk_profile(p_age_group => '18-29',
                               p_employment_group => 'Unstable');              -- highest risk combo
SELECT * FROM get_risk_profile(p_employment_group => 'Stable');                -- most stable


-- ════════════════════════════════════════════════════════════════
-- REUSABLE VIEW FOR DASHBOARDS
-- ════════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW vw_risk_summary AS
SELECT
    age_group,
    employment_group,
    income_group,
    COUNT(*)                                       AS customers,
    SUM(target)                                    AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)           AS default_rate_pct,
    ROUND((SUM(amt_credit) / 1e6)::NUMERIC, 2)     AS total_credit_m,
    ROUND(AVG(employment_age_ratio)::NUMERIC, 4)   AS avg_employment_stability,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)         AS avg_late_rate,
    ROUND(AVG(cc_utilisation)::NUMERIC, 4)         AS avg_cc_utilisation,
    ROUND(AVG(bur_max_overdue)::NUMERIC, 2)        AS avg_bureau_overdue
FROM application
GROUP BY age_group, employment_group, income_group;

SELECT * FROM vw_risk_summary ORDER BY default_rate_pct DESC;


-- ════════════════════════════════════════════════════════════════
-- EXPLAIN ANALYZE — QUERY PERFORMANCE
-- ════════════════════════════════════════════════════════════════

EXPLAIN ANALYZE
SELECT
    age_group,
    employment_group,
    ROUND(AVG(target) * 100::NUMERIC, 2)           AS default_rate_pct,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)         AS avg_late_rate
FROM application
WHERE employment_age_ratio < 0.1    -- Unstable segment
GROUP BY age_group, employment_group;


-- ════════════════════════════════════════════════════════════════
-- DATA QUALITY CHECKS
-- ════════════════════════════════════════════════════════════════

SELECT
    COUNT(*) FILTER (WHERE amt_income_total       IS NULL) AS null_income,
    COUNT(*) FILTER (WHERE amt_credit             IS NULL) AS null_credit,
    COUNT(*) FILTER (WHERE days_birth             IS NULL) AS null_age,
    COUNT(*) FILTER (WHERE target                 IS NULL) AS null_target,
    COUNT(*) FILTER (WHERE employment_age_ratio   IS NULL) AS null_emp_ratio,
    COUNT(*) FILTER (WHERE inst_late_rate         IS NULL) AS null_late_rate,
    COUNT(*) FILTER (WHERE cc_utilisation         IS NULL) AS null_cc_util,
    COUNT(*) FILTER (WHERE bur_total_debt         IS NULL) AS null_bureau_debt,
    COUNT(*) FILTER (WHERE age_group              IS NULL) AS null_age_group,
    COUNT(*) FILTER (WHERE employment_group       IS NULL) AS null_emp_group
FROM application;

-- Known data quirks — these are NOT errors
SELECT
    COUNT(*) FILTER (WHERE bur_total_debt < 0)       AS negative_bureau_debt,  -- ~1296 (overpayments)
    COUNT(*) FILTER (WHERE cc_utilisation > 1.0)     AS over_credit_limit,     -- ~1042 (25.9% default rate)
    COUNT(*) FILTER (WHERE inst_days_late_mean < 0)  AS paid_early_customers,  -- negative = paid early
    MIN(inst_days_late_mean)                          AS earliest_payment_days  -- most early payers
FROM application;

-- Value range sanity check
SELECT
    MIN(target)                AS min_target,   -- must be 0
    MAX(target)                AS max_target,   -- must be 1
    MIN(days_birth / 365.0)    AS youngest_yrs, -- expect ~21
    MAX(days_birth / 365.0)    AS oldest_yrs,   -- expect ~69
    MIN(inst_late_rate)        AS min_late,     -- must be 0
    MAX(inst_late_rate)        AS max_late,     -- expect <= 1
    MIN(cc_utilisation)        AS min_cc_util,  -- must be >= 0
    MAX(cc_utilisation)        AS max_cc_util   -- can exceed 1.0
FROM application;
