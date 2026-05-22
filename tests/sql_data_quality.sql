-- ============================================================================
-- SQL Data Quality Checks
-- ============================================================================
-- Run after the processed CSV is loaded into Postgres. Each check raises an
-- exception on failure, which (with ON_ERROR_STOP) gives a non-zero exit code.
--
-- Run manually with:
--   psql -h localhost -U postgres -d credit_risk -f tests/sql_data_quality.sql
-- ============================================================================


-- ─── CHECK 1: Row count matches the expected dataset size ───────────────────
DO $$
DECLARE
    actual_rows BIGINT;
BEGIN
    SELECT COUNT(*) INTO actual_rows FROM credit_data;
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
    FROM credit_data
    WHERE target IS NOT NULL AND target NOT IN (0, 1);

    IF bad_targets > 0 THEN
        RAISE EXCEPTION 'DQ FAIL: % rows have invalid TARGET values', bad_targets;
    END IF;
    RAISE NOTICE 'DQ PASS: all TARGET values in {0, 1}';
END $$;


-- ─── CHECK 3: Default rate stays in the expected band (7-9%) ────────────────
-- A default rate drifting outside this band suggests a data load problem.
DO $$
DECLARE
    default_rate NUMERIC;
BEGIN
    SELECT ROUND(AVG(target)::numeric * 100, 2) INTO default_rate
    FROM credit_data WHERE target IS NOT NULL;

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
    FROM credit_data
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
    FROM credit_data
    WHERE (days_birth / 365.0) < 18 OR (days_birth / 365.0) > 100;

    IF bad_ages > 0 THEN
        RAISE EXCEPTION 'DQ FAIL: % rows have implausible age', bad_ages;
    END IF;
    RAISE NOTICE 'DQ PASS: all ages in [18, 100]';
END $$;


-- ─── CHECK 6: Known data quirks stay within a sane range ────────────────────
-- Negative bureau debt and over-limit utilisation are expected (see
-- analysis.sql), but a large jump in either would point to an upstream
-- data problem worth investigating.
DO $$
DECLARE
    neg_debt BIGINT;
    over_limit BIGINT;
BEGIN
    SELECT COUNT(*) FILTER (WHERE bur_total_debt < 0),
           COUNT(*) FILTER (WHERE cc_utilisation > 1.0)
    INTO neg_debt, over_limit
    FROM credit_data;

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
    'credit_data' AS table_name,
    COUNT(*) AS total_rows,
    SUM(target)::BIGINT AS total_defaults,
    ROUND(AVG(target)::numeric * 100, 2) AS default_rate_pct
FROM credit_data;