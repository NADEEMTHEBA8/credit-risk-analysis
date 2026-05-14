-- ============================================================================
-- SQL Data Quality Checks
-- ============================================================================
-- Run after pipeline output is loaded into Postgres. Each check uses
-- ON_ERROR_STOP so a failure produces a non-zero exit code, which Airflow
-- treats as a task failure and halts the DAG.
--
-- Invoked from:
--   dags/credit_risk_dag.py  (dq_checks task)
--   Makefile                 (sql-test target)
-- ============================================================================



-- ─── CHECK 1: Row count is exactly what we expect ───────────────────────────
DO $$
DECLARE
    actual_rows BIGINT;
BEGIN
    SELECT COUNT(*) INTO actual_rows FROM application;
    IF actual_rows != 307511 THEN
        RAISE EXCEPTION 'DQ FAIL: expected 307,511 rows, got %', actual_rows;
    END IF;
    RAISE NOTICE 'DQ PASS: row count = %', actual_rows;
END $$;


-- ─── CHECK 2: TARGET is strictly binary ─────────────────────────────────────
DO $$
DECLARE
    bad_targets BIGINT;
BEGIN
    SELECT COUNT(*) INTO bad_targets
    FROM application
    WHERE target IS NOT NULL AND target NOT IN (0, 1);

    IF bad_targets > 0 THEN
        RAISE EXCEPTION 'DQ FAIL: % rows have invalid TARGET values', bad_targets;
    END IF;
    RAISE NOTICE 'DQ PASS: all TARGET values in {0, 1}';
END $$;


-- ─── CHECK 3: Default rate stays in the expected band (7-9%) ────────────────
-- Home Credit default rate is 8.07%. Drift outside +/- 1% suggests data issues.
DO $$
DECLARE
    default_rate NUMERIC;
BEGIN
    SELECT ROUND(AVG(target) * 100::NUMERIC, 2) INTO default_rate
    FROM application WHERE target IS NOT NULL;

    IF default_rate < 7.0 OR default_rate > 9.0 THEN
        RAISE EXCEPTION 'DQ FAIL: default rate % outside 7-9%% band', default_rate;
    END IF;
    RAISE NOTICE 'DQ PASS: default rate = % percent', default_rate;
END $$;


-- ─── CHECK 4: Critical columns have no NULLs ────────────────────────────────
DO $$
DECLARE
    nulls BIGINT;
BEGIN
    SELECT COUNT(*) INTO nulls
    FROM application
    WHERE amt_income_total IS NULL
       OR amt_credit IS NULL
       OR days_birth IS NULL;

    IF nulls > 0 THEN
        RAISE EXCEPTION 'DQ FAIL: % rows have NULL in critical columns', nulls;
    END IF;
    RAISE NOTICE 'DQ PASS: no NULLs in critical columns';
END $$;


-- ─── CHECK 5: Ages are plausible (18-100 years) ─────────────────────────────
DO $$
DECLARE
    bad_ages BIGINT;
BEGIN
    SELECT COUNT(*) INTO bad_ages
    FROM application
    WHERE (days_birth / 365.0) < 18 OR (days_birth / 365.0) > 100;

    IF bad_ages > 0 THEN
        RAISE EXCEPTION 'DQ FAIL: % rows have implausible age', bad_ages;
    END IF;
    RAISE NOTICE 'DQ PASS: all ages in [18, 100]';
END $$;


-- ─── CHECK 6: Segmentation labels are fully populated ───────────────────────
DO $$
DECLARE
    nulls BIGINT;
BEGIN
    SELECT COUNT(*) INTO nulls
    FROM application
    WHERE age_group IS NULL OR employment_group IS NULL;

    IF nulls > 0 THEN
        RAISE EXCEPTION 'DQ FAIL: % rows missing segmentation labels', nulls;
    END IF;
    RAISE NOTICE 'DQ PASS: all rows segmented';
END $$;


-- ─── CHECK 7: age_group values match the expected vocabulary ────────────────
DO $$
DECLARE
    bad_groups BIGINT;
BEGIN
    SELECT COUNT(*) INTO bad_groups
    FROM application
    WHERE age_group NOT IN ('18-29', '30-39', '40-49', '50-59', '60+');

    IF bad_groups > 0 THEN
        RAISE EXCEPTION 'DQ FAIL: % rows have unexpected age_group value', bad_groups;
    END IF;
    RAISE NOTICE 'DQ PASS: all age_group values valid';
END $$;


-- ─── CHECK 8: Known data quirks stay within expected counts ─────────────────
-- These aren't errors but their counts should be stable across reloads.
-- A 10x change indicates an upstream data issue worth investigating.
DO $$
DECLARE
    neg_debt BIGINT;
    over_limit BIGINT;
BEGIN
    SELECT COUNT(*) FILTER (WHERE bur_total_debt < 0),
           COUNT(*) FILTER (WHERE cc_utilisation > 1.0)
    INTO neg_debt, over_limit
    FROM application;

    -- Expected: ~1296 negative debt, ~1042 over-limit
    IF neg_debt > 5000 THEN
        RAISE WARNING 'DQ WARN: unusually high negative debt count: %', neg_debt;
    END IF;
    IF over_limit > 5000 THEN
        RAISE WARNING 'DQ WARN: unusually high over-limit count: %', over_limit;
    END IF;
    RAISE NOTICE 'DQ PASS: data quirks within expected range (% neg_debt, % over_limit)',
                 neg_debt, over_limit;
END $$;


-- ─── SUMMARY ────────────────────────────────────────────────────────────────
SELECT
    'application' AS table_name,
    COUNT(*) AS total_rows,
    SUM(target)::BIGINT AS total_defaults,
    ROUND(AVG(target) * 100::NUMERIC, 2) AS default_rate_pct,
    COUNT(DISTINCT age_group) AS age_buckets,
    COUNT(DISTINCT employment_group) AS employment_buckets,
    NOW() AS checked_at
FROM application;
