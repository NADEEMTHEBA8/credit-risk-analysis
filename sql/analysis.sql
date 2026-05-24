-- ============================================================================
-- Credit Risk — PostgreSQL Analysis Layer
-- ============================================================================
-- Exploratory analysis on the processed export from the Python pipeline.
-- Source: data/processed/credit_data_sql.csv (one row per customer, 20 columns).
--
-- This is an analyst exploration layer, not a transformation pipeline. The
-- table mirrors the CSV exactly; risk groups are computed inline per query
-- with CASE rather than stored, so nothing here mutates the imported data.
--
-- RUN ORDER:
--   (1) Schema block below — creates the table
--   (2) Import the CSV     — \copy command or pgAdmin import
--   (3) Run any analysis block independently
--
-- Note: running this whole file re-creates the table empty. To see results,
-- run the schema block once, load the CSV, then run the analysis blocks.
--
-- Data notes (from the pipeline, not errors):
--   DAYS_BIRTH / DAYS_EMPLOYED are positive day counts (age = DAYS_BIRTH/365).
--   bur_total_debt can be negative — a customer overpaid a loan.
--   cc_utilisation can exceed 1.0 — a customer over their credit limit.
--   inst_days_late_mean can be negative — a customer paid early.
-- ============================================================================


-- ============================================================================
-- 1. TABLE CREATION
-- ============================================================================

DROP TABLE IF EXISTS credit_data;

CREATE TABLE credit_data (
    amt_income_total       NUMERIC,
    amt_credit             NUMERIC,
    amt_goods_price        NUMERIC,
    days_birth             INTEGER,
    days_employed          NUMERIC,
    income_credit_ratio    NUMERIC,
    employment_age_ratio   NUMERIC,
    annuity_income_ratio   NUMERIC,
    bur_total_debt         NUMERIC,
    bur_num_credits        NUMERIC,
    bur_max_overdue        NUMERIC,
    prev_num_applications  NUMERIC,
    prev_approval_rate     NUMERIC,
    inst_late_rate         NUMERIC,
    inst_days_late_mean    NUMERIC,
    cc_utilisation         NUMERIC,
    cc_dpd_max             NUMERIC,
    pos_sk_dpd_max         NUMERIC,
    pos_completion_rate    NUMERIC,
    target                 NUMERIC
);


-- ============================================================================
-- 2. CSV IMPORT
-- ============================================================================
-- From psql, with the file path adjusted to your machine:
--
--   \copy credit_data FROM 'data/processed/credit_data_sql.csv' WITH (FORMAT csv, HEADER true)
--
-- Or use the pgAdmin Import/Export dialog on the credit_data table
-- (Format: csv, Header: on, Delimiter: comma).


SELECT '==================================================' AS section,
       'Data quality checks' AS analysis;

-- ============================================================================
-- 3. DATA QUALITY CHECKS
-- ============================================================================

-- Row count and how many rows carry a label.
SELECT
    COUNT(*)                                  AS total_rows,
    COUNT(target)                             AS labelled_rows,
    COUNT(*) - COUNT(target)                  AS unlabelled_rows
FROM credit_data;

-- NULL counts on the columns the analysis depends on most.
SELECT
    COUNT(*) FILTER (WHERE amt_income_total IS NULL)  AS null_income,
    COUNT(*) FILTER (WHERE amt_credit       IS NULL)  AS null_credit,
    COUNT(*) FILTER (WHERE days_birth       IS NULL)  AS null_days_birth,
    COUNT(*) FILTER (WHERE inst_late_rate   IS NULL)  AS null_inst_late_rate,
    COUNT(*) FILTER (WHERE cc_utilisation   IS NULL)  AS null_cc_utilisation
FROM credit_data;

-- TARGET should only ever be 0 or 1 where present.
SELECT target, COUNT(*) AS rows
FROM credit_data
GROUP BY target
ORDER BY target;

-- Sanity ranges. days_birth/365 should land in a believable adult range;
-- the known quirks (negative debt, utilisation > 1) are expected, not bugs.
SELECT
    ROUND(MIN(days_birth) / 365.0, 1)              AS youngest_age,
    ROUND(MAX(days_birth) / 365.0, 1)              AS oldest_age,
    COUNT(*) FILTER (WHERE bur_total_debt < 0)     AS negative_debt_rows,
    COUNT(*) FILTER (WHERE cc_utilisation > 1.0)   AS over_limit_rows,
    COUNT(*) FILTER (WHERE inst_days_late_mean < 0) AS paid_early_rows
FROM credit_data;


SELECT '==================================================' AS section,
       'Portfolio overview' AS analysis;

-- ============================================================================
-- 4. PORTFOLIO OVERVIEW
-- ============================================================================

SELECT
    COUNT(*)                                       AS customers,
    SUM(target)                                    AS defaults,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct,
    ROUND(AVG(amt_income_total)::numeric, 0)       AS avg_income,
    ROUND(AVG(amt_credit)::numeric, 0)             AS avg_credit,
    ROUND(AVG(inst_late_rate)::numeric, 4)         AS avg_late_rate,
    ROUND(AVG(cc_utilisation)::numeric, 4)         AS avg_cc_utilisation
FROM credit_data
WHERE target IS NOT NULL;


SELECT '==================================================' AS section,
       'Risk segmentation: age, employment, income' AS analysis;

-- ============================================================================
-- 5. RISK SEGMENTATION  (inline CASE — no stored group columns)
-- ============================================================================

-- Default rate by age band.
WITH age_banded AS (
    SELECT
        target,
        CASE
            WHEN days_birth / 365.0 < 30 THEN '18-29'
            WHEN days_birth / 365.0 < 40 THEN '30-39'
            WHEN days_birth / 365.0 < 50 THEN '40-49'
            WHEN days_birth / 365.0 < 60 THEN '50-59'
            ELSE                              '60+'
        END AS age_band
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    age_band,
    COUNT(*)                                       AS customers,
    SUM(target)                                    AS defaults,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_portfolio
FROM age_banded
GROUP BY age_band
ORDER BY age_band;

-- Default rate by employment stability (employed days / age days).
WITH emp_banded AS (
    SELECT
        target,
        CASE
            WHEN employment_age_ratio IS NULL
              OR employment_age_ratio < 0.10 THEN 'Unstable'
            WHEN employment_age_ratio < 0.30 THEN 'Short-term'
            WHEN employment_age_ratio < 0.60 THEN 'Moderate'
            ELSE                                  'Stable'
        END AS employment_band
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    employment_band,
    COUNT(*)                                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct
FROM emp_banded
GROUP BY employment_band
ORDER BY default_rate_pct DESC;

-- Default rate by income band.
WITH income_banded AS (
    SELECT
        target,
        amt_income_total,
        CASE
            WHEN amt_income_total < 100000 THEN 'Low'
            WHEN amt_income_total < 200000 THEN 'Medium'
            ELSE                                'High'
        END AS income_band
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    income_band,
    COUNT(*)                                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct,
    ROUND(AVG(amt_income_total)::numeric, 0)       AS avg_income
FROM income_banded
GROUP BY income_band
ORDER BY default_rate_pct DESC;


SELECT '==================================================' AS section,
       'Age x employment cohort matrix' AS analysis;

-- ============================================================================
-- 6. AGE x EMPLOYMENT COHORT MATRIX
-- ============================================================================
-- Crossing two bands gives the widest spread in default rate. Cohorts with
-- too few rows are dropped so the rate is worth reading.

WITH banded AS (
    SELECT
        target,
        CASE
            WHEN days_birth / 365.0 < 30 THEN '18-29'
            WHEN days_birth / 365.0 < 40 THEN '30-39'
            WHEN days_birth / 365.0 < 50 THEN '40-49'
            WHEN days_birth / 365.0 < 60 THEN '50-59'
            ELSE                              '60+'
        END AS age_band,
        CASE
            WHEN employment_age_ratio IS NULL
              OR employment_age_ratio < 0.10 THEN 'Unstable'
            WHEN employment_age_ratio < 0.30 THEN 'Short-term'
            WHEN employment_age_ratio < 0.60 THEN 'Moderate'
            ELSE                                  'Stable'
        END AS employment_band
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    age_band,
    employment_band,
    COUNT(*)                                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct
FROM banded
GROUP BY age_band, employment_band
HAVING COUNT(*) >= 500
ORDER BY default_rate_pct DESC;


SELECT '==================================================' AS section,
       'Behavioural analysis: payment, card, bureau, prior apps' AS analysis;

-- ============================================================================
-- 7. BEHAVIOURAL ANALYSIS
-- ============================================================================

-- Installment payment behaviour. Late-rate buckets, ordered worst to best.
WITH payment_segments AS (
    SELECT
        target,
        CASE
            WHEN inst_late_rate IS NULL THEN '0. No payment history'
            WHEN inst_late_rate = 0     THEN '1. Always on time'
            WHEN inst_late_rate < 0.10  THEN '2. Rarely late'
            WHEN inst_late_rate < 0.30  THEN '3. Sometimes late'
            WHEN inst_late_rate < 0.50  THEN '4. Often late'
            ELSE                             '5. Usually late'
        END AS payment_segment
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    payment_segment,
    COUNT(*)                                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct
FROM payment_segments
GROUP BY payment_segment
ORDER BY payment_segment;

-- Credit card utilisation, including the over-limit group (utilisation > 1.0).
WITH cc_segments AS (
    SELECT
        target,
        CASE
            WHEN cc_utilisation IS NULL  THEN '0. No credit card'
            WHEN cc_utilisation = 0      THEN '1. Unused'
            WHEN cc_utilisation < 0.30   THEN '2. Low use'
            WHEN cc_utilisation < 0.60   THEN '3. Moderate'
            WHEN cc_utilisation < 0.90   THEN '4. High'
            WHEN cc_utilisation <= 1.0   THEN '5. Near limit'
            ELSE                              '6. Over limit'
        END AS cc_segment
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    cc_segment,
    COUNT(*)                                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct
FROM cc_segments
GROUP BY cc_segment
ORDER BY cc_segment;

-- Bureau history: any recorded overdue at another lender vs a clean record.
WITH bureau_segments AS (
    SELECT
        target,
        bur_num_credits,
        CASE
            WHEN bur_num_credits IS NULL
              OR bur_num_credits = 0  THEN '0. No bureau history'
            WHEN bur_max_overdue > 0  THEN '1. Has overdue elsewhere'
            ELSE                           '2. Clean bureau record'
        END AS bureau_segment
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    bureau_segment,
    COUNT(*)                                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct,
    ROUND(AVG(bur_num_credits)::numeric, 2)        AS avg_bureau_credits
FROM bureau_segments
GROUP BY bureau_segment
ORDER BY default_rate_pct DESC;

-- Previous-application history by approval rate.
WITH prev_segments AS (
    SELECT
        target,
        CASE
            WHEN prev_num_applications IS NULL
              OR prev_num_applications = 0 THEN '0. First-time applicant'
            WHEN prev_approval_rate >= 0.90 THEN '1. Mostly approved'
            WHEN prev_approval_rate >= 0.50 THEN '2. Mixed'
            WHEN prev_approval_rate >  0    THEN '3. Mostly refused'
            ELSE                                 '4. Always refused'
        END AS prev_segment
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    prev_segment,
    COUNT(*)                                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct
FROM prev_segments
GROUP BY prev_segment
ORDER BY prev_segment;


SELECT '==================================================' AS section,
       'Window-function analysis: age and income quantiles' AS analysis;

-- ============================================================================
-- 8. WINDOW-FUNCTION ANALYSIS
-- ============================================================================

-- Default rate by age quintile (NTILE over days_birth). Quintile 1 = youngest.
WITH age_quintiles AS (
    SELECT
        target,
        days_birth,
        inst_late_rate,
        cc_utilisation,
        NTILE(5) OVER (ORDER BY days_birth) AS age_quintile
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    age_quintile,
    ROUND(MIN(days_birth) / 365.0, 1)              AS min_age,
    ROUND(MAX(days_birth) / 365.0, 1)              AS max_age,
    COUNT(*)                                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)           AS default_rate_pct,
    ROUND(AVG(inst_late_rate)::numeric, 4)         AS avg_late_rate,
    ROUND(AVG(cc_utilisation)::numeric, 4)         AS avg_cc_utilisation
FROM age_quintiles
GROUP BY age_quintile
ORDER BY age_quintile;

-- Default rate by income decile, with each decile compared to the
-- portfolio-wide rate using a window AVG over the whole table.
WITH income_deciles AS (
    SELECT
        target,
        amt_income_total,
        NTILE(10) OVER (ORDER BY amt_income_total) AS income_decile
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    income_decile,
    COUNT(*)                                       AS customers,
    ROUND(MIN(amt_income_total)::numeric, 0)       AS min_income,
    ROUND(MAX(amt_income_total)::numeric, 0)       AS max_income,
    ROUND(AVG(target)::numeric * 100, 2)           AS decile_default_rate,
    ROUND(AVG(AVG(target)) OVER ()::numeric * 100, 2) AS portfolio_default_rate,
    ROUND((AVG(target) - AVG(AVG(target)) OVER ())::numeric * 100, 2) AS rate_vs_portfolio
FROM income_deciles
GROUP BY income_decile
ORDER BY income_decile;


SELECT '==================================================' AS section,
       'Ranking and cohort comparison' AS analysis;

-- ============================================================================
-- 9. RANKING / COHORT COMPARISON
-- ============================================================================

-- Defaulter vs non-defaulter side by side. PERCENTILE_CONT gives the median,
-- which is more robust than the mean on skewed money columns.
SELECT
    CASE WHEN target = 1 THEN 'Defaulted' ELSE 'Repaid' END  AS outcome,
    COUNT(*)                                       AS customers,
    ROUND(AVG(days_birth) / 365.0, 1)              AS avg_age,
    ROUND(AVG(amt_income_total)::numeric, 0)       AS avg_income,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amt_income_total)::numeric, 0)
                                                   AS median_income,
    ROUND(AVG(amt_credit)::numeric, 0)             AS avg_credit,
    ROUND(AVG(inst_late_rate)::numeric, 4)         AS avg_late_rate,
    ROUND(AVG(cc_utilisation)::numeric, 4)         AS avg_cc_utilisation,
    ROUND(AVG(prev_approval_rate)::numeric, 4)     AS avg_prev_approval
FROM credit_data
WHERE target IS NOT NULL
GROUP BY target
ORDER BY target DESC;

-- Highest-risk age-employment cohorts, ranked. A composite ordering key
-- (default rate, then size) breaks ties toward the larger cohort.
WITH banded AS (
    SELECT
        target,
        CASE
            WHEN days_birth / 365.0 < 30 THEN '18-29'
            WHEN days_birth / 365.0 < 40 THEN '30-39'
            WHEN days_birth / 365.0 < 50 THEN '40-49'
            WHEN days_birth / 365.0 < 60 THEN '50-59'
            ELSE                              '60+'
        END AS age_band,
        CASE
            WHEN employment_age_ratio IS NULL
              OR employment_age_ratio < 0.10 THEN 'Unstable'
            WHEN employment_age_ratio < 0.30 THEN 'Short-term'
            WHEN employment_age_ratio < 0.60 THEN 'Moderate'
            ELSE                                  'Stable'
        END AS employment_band
    FROM credit_data
    WHERE target IS NOT NULL
),
cohorts AS (
    SELECT
        age_band,
        employment_band,
        COUNT(*)                       AS customers,
        AVG(target)::numeric * 100     AS default_rate_pct
    FROM banded
    GROUP BY age_band, employment_band
    HAVING COUNT(*) >= 500
)
SELECT
    RANK() OVER (ORDER BY default_rate_pct DESC)   AS risk_rank,
    age_band,
    employment_band,
    customers,
    ROUND(default_rate_pct, 2)                     AS default_rate_pct
FROM cohorts
ORDER BY risk_rank;


SELECT '==================================================' AS section,
       'Helper view: cohort default-rate summary' AS analysis;

-- ============================================================================
-- 10. HELPER VIEW
-- ============================================================================
-- One reusable view: the age x employment cohort summary, which several
-- of the queries above lean on. Kept as a view because it is genuinely
-- re-run often; everything else stays as ad-hoc queries.

CREATE OR REPLACE VIEW vw_cohort_default_rates AS
WITH banded AS (
    SELECT
        target,
        CASE
            WHEN days_birth / 365.0 < 30 THEN '18-29'
            WHEN days_birth / 365.0 < 40 THEN '30-39'
            WHEN days_birth / 365.0 < 50 THEN '40-49'
            WHEN days_birth / 365.0 < 60 THEN '50-59'
            ELSE                              '60+'
        END AS age_band,
        CASE
            WHEN employment_age_ratio IS NULL
              OR employment_age_ratio < 0.10 THEN 'Unstable'
            WHEN employment_age_ratio < 0.30 THEN 'Short-term'
            WHEN employment_age_ratio < 0.60 THEN 'Moderate'
            ELSE                                  'Stable'
        END AS employment_band
    FROM credit_data
    WHERE target IS NOT NULL
)
SELECT
    age_band,
    employment_band,
    COUNT(*)                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)    AS default_rate_pct
FROM banded
GROUP BY age_band, employment_band;

-- Example use:
--   SELECT * FROM vw_cohort_default_rates ORDER BY default_rate_pct DESC;