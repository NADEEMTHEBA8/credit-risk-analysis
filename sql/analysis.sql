-- ================================================================
-- Home Credit Default Risk — SQL Analysis
-- Database  : PostgreSQL 14+
-- Source    : credit_data_sql.csv (one row per customer)
-- ================================================================
-- Customer-level risk analysis on the output of the Python pipeline.
-- The analysis is organised around age, employment stability, and
-- payment behaviour, which the pipeline's feature analysis found to be
-- the most useful signals. Income-to-credit ratio looked promising but
-- turned out weak, so it is not used as a primary segmentation basis.
--
-- KNOWN DATA QUIRKS (not errors):
--   bur_total_debt can be negative      — customer overpaid a loan
--   cc_utilisation can exceed 1.0       — customer is over credit limit
--   inst_days_late_mean can be negative — customer paid early
--
-- RUN ORDER:
--   (1) Schema setup block   — creates the table
--   (2) Import CSV            — right-click table -> Import/Export
--   (3) Segmentation block   — adds and fills the label columns
--   (4) Analysis queries     — run one block at a time
-- ================================================================


-- ================================================================
-- SCHEMA SETUP
-- ================================================================

DROP TABLE IF EXISTS application CASCADE;

CREATE TABLE application (

    -- Application-level fields
    amt_income_total       NUMERIC,   -- Annual income
    amt_credit             NUMERIC,   -- Requested loan amount
    amt_goods_price        NUMERIC,   -- Goods price (NULL for cash loans)
    days_birth             NUMERIC,   -- Age in days (positive — converted in pipeline)
    days_employed          NUMERIC,   -- Employment days (positive; NULL = unemployed)

    -- Engineered ratios from the Python pipeline
    income_credit_ratio    NUMERIC,   -- income / credit — weak signal, kept for completeness
    employment_age_ratio   NUMERIC,   -- employed_days / birth_days
    annuity_income_ratio   NUMERIC,   -- monthly payment / income

    -- Bureau signals (from bureau.csv + bureau_balance.csv)
    bur_total_debt         NUMERIC,   -- total outstanding debt at other banks (can be negative)
    bur_num_credits        NUMERIC,   -- count of past bureau credits
    bur_max_overdue        NUMERIC,   -- largest overdue amount

    -- Previous application signals
    prev_num_applications  NUMERIC,   -- total past Home Credit applications
    prev_approval_rate     NUMERIC,   -- fraction approved

    -- Installment payment behaviour
    inst_late_rate         NUMERIC,   -- fraction of payments made late
    inst_days_late_mean    NUMERIC,   -- avg days late per payment (negative = paid early)

    -- Credit card signals
    cc_utilisation         NUMERIC,   -- balance / limit; can exceed 1.0
    cc_dpd_max             NUMERIC,   -- worst DPD event on credit card

    -- POS cash loan signals
    pos_sk_dpd_max         NUMERIC,   -- worst DPD on a POS/cash loan
    pos_completion_rate    NUMERIC,   -- fraction of POS loans fully completed

    target                 NUMERIC    -- 1 = defaulted, 0 = repaid
);

-- After running this block, import credit_data_sql.csv:
--   Right-click "application" in left panel -> Import/Export Data
--   Format: csv | Header: ON | Delimiter: ,


-- ================================================================
-- SEGMENTATION SETUP
-- ================================================================

ALTER TABLE application
    ADD COLUMN IF NOT EXISTS income_group      VARCHAR(20),
    ADD COLUMN IF NOT EXISTS loan_size         VARCHAR(20),
    ADD COLUMN IF NOT EXISTS employment_group  VARCHAR(20),
    ADD COLUMN IF NOT EXISTS age_group         VARCHAR(10),
    ADD COLUMN IF NOT EXISTS risk_level        VARCHAR(20);

-- Single UPDATE pass fills all 5 columns.
-- Thresholds match the Python pipeline pd.cut bins exactly so
-- SQL and Python segment assignments stay identical.
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

    -- employment_age_ratio = days_employed / days_birth.
    -- More stable employment generally means lower default risk.
    employment_group = CASE
        WHEN employment_age_ratio IS NULL
          OR employment_age_ratio < 0.1  THEN 'Unstable'
        WHEN employment_age_ratio < 0.3  THEN 'Short-term'
        WHEN employment_age_ratio < 0.6  THEN 'Moderate'
        ELSE                                  'Stable'
    END,

    -- Age group. days_birth / 365 = age in years.
    -- Younger customers tend to default more often.
    age_group = CASE
        WHEN (days_birth / 365) < 30  THEN '18-29'
        WHEN (days_birth / 365) < 40  THEN '30-39'
        WHEN (days_birth / 365) < 50  THEN '40-49'
        WHEN (days_birth / 365) < 60  THEN '50-59'
        ELSE                               '60+'
    END,

    -- risk_level kept for compatibility with pipeline naming.
    -- Default rates across these bands are nearly flat — do not use
    -- risk_level alone as a risk signal; prefer age/employment.
    risk_level = CASE
        WHEN income_credit_ratio < 0.3  THEN 'High Risk'
        WHEN income_credit_ratio < 0.6  THEN 'Medium Risk'
        ELSE                                 'Low Risk'
    END;

-- Quick verify after UPDATE
SELECT age_group, employment_group,
       COUNT(*)                                   AS customers,
       ROUND(AVG(target) * 100::NUMERIC, 2)       AS default_rate_pct
FROM application
GROUP BY age_group, employment_group
ORDER BY default_rate_pct DESC
LIMIT 5;


-- ================================================================
-- DATA VERIFICATION
-- ================================================================

SELECT
    COUNT(*)                                      AS total_rows,
    SUM(target)                                   AS total_defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)          AS default_rate_pct,
    ROUND(AVG(bur_total_debt)::NUMERIC, 0)        AS avg_bureau_debt,
    COUNT(*) FILTER (WHERE bur_total_debt < 0)    AS negative_debt_rows,
    COUNT(*) FILTER (WHERE cc_utilisation > 1.0)  AS over_limit_rows
FROM application;


-- ================================================================
-- PORTFOLIO KPI OVERVIEW
-- ================================================================

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


-- ================================================================
-- PRIMARY ANALYSIS: AGE GROUP
-- Age is one of the clearest signals in the data — default rates
-- fall steadily as customers get older.
-- ================================================================

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


-- ================================================================
-- PRIMARY ANALYSIS: EMPLOYMENT STABILITY
-- Employment stability separates the portfolio well, and combined
-- with age it gives the most useful segmentation in the dataset.
-- ================================================================

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


-- ================================================================
-- AGE x EMPLOYMENT COMBINED
-- These two together produce the widest spread in default rate
-- of any segmentation tried here.
-- ================================================================

SELECT
    age_group,
    employment_group,
    COUNT(*)                                       AS customers,
    SUM(target)                                    AS defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2)           AS default_rate_pct,
    ROUND(AVG(inst_late_rate)::NUMERIC, 4)         AS avg_late_rate
FROM application
GROUP BY age_group, employment_group
HAVING COUNT(*) > 500       -- minimum sample for a reliable rate
ORDER BY default_rate_pct DESC;


-- ================================================================
-- PAYMENT BEHAVIOUR ANALYSIS
-- Installment late rate is one of the stronger behavioural signals —
-- customers who are usually late default well above portfolio average.
-- ================================================================

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
-- Default rate should rise steadily from "Always On Time" to "Usually Late".


-- ================================================================
-- CREDIT CARD UTILISATION ANALYSIS
-- Utilisation tracks default risk closely. Customers over their
-- credit limit are the highest-risk segment in this analysis.
-- ================================================================

WITH cc_segments AS (
    SELECT target,
        CASE
            WHEN cc_utilisation IS NULL    THEN '1. No Credit Card'
            WHEN cc_utilisation = 0        THEN '2. Unused Card (0%)'
            WHEN cc_utilisation < 0.30     THEN '3. Low Use      (1-30%)'
            WHEN cc_utilisation < 0.60     THEN '4. Moderate     (30-60%)'
            WHEN cc_utilisation < 0.90     THEN '5. High         (60-90%)'
            WHEN cc_utilisation <= 1.0     THEN '6. Near-Limit   (90-100%)'
            ELSE                                '7. OVER LIMIT   (>100%)'
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


-- ================================================================
-- PREVIOUS APPLICATION HISTORY
-- Approval history on past applications carries some signal.
-- Worth noting: the "Mostly Refused" bucket comes out riskier than
-- "Always Refused" — the mixed-history group is not the safest.
-- ================================================================

WITH prev_segments AS (
    SELECT target,
        CASE
            WHEN prev_num_applications IS NULL
              OR prev_num_applications = 0    THEN '1. First-Time Applicant'
            WHEN prev_approval_rate >= 0.9    THEN '2. Mostly Approved (>90%)'
            WHEN prev_approval_rate >= 0.5    THEN '3. Mixed (50-90%)'
            WHEN prev_approval_rate > 0       THEN '4. Mostly Refused (1-50%)'
            ELSE                                   '5. Always Refused (0%)'
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


-- ================================================================
-- BUREAU HISTORY IMPACT
-- Any recorded overdue at another bank is a clear risk marker.
-- Note: bur_total_debt can be negative (overpayment), so
-- GREATEST(bur_total_debt, 0) is used where a debt amount is needed.
-- ================================================================

WITH bureau_segments AS (
    SELECT target, amt_income_total,
        CASE
            WHEN bur_num_credits IS NULL
              OR bur_num_credits = 0    THEN '1. No Bureau History'
            WHEN bur_max_overdue > 0    THEN '2. Has Overdue at Other Banks'
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


-- ================================================================
-- INCOME GROUP (WEAK SIGNAL — SHOWN FOR COMPLETENESS)
-- Default rates are nearly flat across income groups. Income alone
-- is a poor predictor — included mainly as a contrast to age.
-- ================================================================

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
-- Compare the spread here against the age groups to see why income
-- is treated as a weaker signal.


-- ================================================================
-- NTILE RISK BUCKETS — BY AGE
-- Five equal buckets ordered youngest to oldest.
-- Bucket 1 = youngest = highest risk.
-- ================================================================

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


-- ================================================================
-- DEFAULTER vs NON-DEFAULTER SIDE-BY-SIDE PROFILE
-- Median via PERCENTILE_CONT alongside the averages, since income
-- is skewed and the mean alone can mislead.
-- ================================================================

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


-- ================================================================
-- 3-WAY RISK MATRIX: AGE x EMPLOYMENT x PAYMENT BEHAVIOUR
-- Crossing the three signals that segment the portfolio best.
-- HAVING COUNT(*) > 500 drops segments too small to read into.
-- ================================================================

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


-- ================================================================
-- CUSTOMER RANKING — WINDOW FUNCTIONS
-- Global rank, rank within employment group, and a 0-100 percentile.
-- Ranked by days_birth ASC, so youngest customers rank first.
-- ================================================================

SELECT
    ROUND(days_birth / 365.0, 1)                     AS age_years,
    ROUND(employment_age_ratio, 4)                   AS emp_stability,
    ROUND(inst_late_rate, 4)                         AS late_rate,
    ROUND(cc_utilisation, 4)                         AS cc_util,
    target,
    age_group,
    employment_group,
    RANK() OVER (ORDER BY days_birth ASC)            AS global_age_rank,
    RANK() OVER (
        PARTITION BY employment_group
        ORDER BY days_birth ASC
    )                                                AS age_rank_in_emp_group,
    -- 0 = oldest (safest), 100 = youngest (riskiest)
    ROUND(
        (PERCENT_RANK() OVER (ORDER BY days_birth ASC) * 100)::NUMERIC
    , 2)                                             AS youth_risk_percentile
FROM application
ORDER BY days_birth ASC
LIMIT 20;


-- ================================================================
-- DECILE ANALYSIS — BY EMPLOYMENT STABILITY
-- 10 equal groups; decile 1 = most unstable = highest risk.
-- ================================================================

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


-- ================================================================
-- CUSTOMERS BELOW AVERAGE AGE
-- Younger-than-average customers, as a higher-risk slice to inspect.
-- ================================================================

SELECT
    ROUND(days_birth / 365.0, 1)              AS age_years,
    ROUND(inst_late_rate, 4)                  AS late_rate,
    ROUND(cc_utilisation, 4)                  AS cc_util,
    ROUND(employment_age_ratio, 4)            AS emp_stability,
    target,
    age_group,
    employment_group
FROM application
WHERE days_birth < (SELECT AVG(days_birth) FROM application)
ORDER BY days_birth ASC
LIMIT 20;


-- ================================================================
-- COMPOSITE RISK SCORE
-- Built in three CTEs: per-signal quintiles, then a weighted score,
-- then a final risk band. The score has to be computed in its own
-- CTE before NTILE can band it, since a window result cannot be
-- used directly in GROUP BY.
--
-- Weights roughly reflect how useful each signal looked in the
-- pipeline's feature analysis. income_credit_ratio is left out —
-- it was too weak to contribute.
-- ================================================================

WITH signals AS (
    SELECT
        target,
        days_birth,
        inst_late_rate,
        cc_utilisation,
        employment_age_ratio,
        -- NTILE quintile per signal: 5 = worst, 1 = best
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
            (q_age        * 0.30) +   -- age
            (q_late_rate  * 0.30) +   -- payment behaviour
            (q_cc_util    * 0.25) +   -- cc utilisation
            (q_employment * 0.15)     -- employment stability
        , 2) AS composite_score
    FROM signals
),
final_bands AS (
    -- Separate CTE so the NTILE result can be used in GROUP BY
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


-- ================================================================
-- STORED FUNCTION — FLEXIBLE SEGMENT LOOKUP
-- Pass NULL for any dimension to leave it unfiltered.
-- ================================================================

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
SELECT * FROM get_risk_profile();
SELECT * FROM get_risk_profile(p_age_group => '18-29');
SELECT * FROM get_risk_profile(p_age_group => '18-29',
                               p_employment_group => 'Unstable');
SELECT * FROM get_risk_profile(p_employment_group => 'Stable');


-- ================================================================
-- REUSABLE VIEW FOR DASHBOARDS
-- ================================================================

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


-- ================================================================
-- DATA QUALITY CHECKS
-- ================================================================

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

-- Known data quirks — these are expected, not errors
SELECT
    COUNT(*) FILTER (WHERE bur_total_debt < 0)       AS negative_bureau_debt,   -- overpayments
    COUNT(*) FILTER (WHERE cc_utilisation > 1.0)     AS over_credit_limit,      -- over credit limit
    COUNT(*) FILTER (WHERE inst_days_late_mean < 0)  AS paid_early_customers,   -- negative = paid early
    MIN(inst_days_late_mean)                          AS earliest_payment_days
FROM application;

-- Value range sanity check
SELECT
    MIN(target)                AS min_target,   -- expect 0
    MAX(target)                AS max_target,   -- expect 1
    MIN(days_birth / 365.0)    AS youngest_yrs,
    MAX(days_birth / 365.0)    AS oldest_yrs,
    MIN(inst_late_rate)        AS min_late,     -- expect 0
    MAX(inst_late_rate)        AS max_late,
    MIN(cc_utilisation)        AS min_cc_util,  -- expect >= 0
    MAX(cc_utilisation)        AS max_cc_util
FROM application;