-- Check 1 & 7: Row count matches expected 307511 exactly
SELECT
    COUNT(*) as total_rows
FROM {{ ref('stg_credit_data') }}
HAVING COUNT(*) != 307511
