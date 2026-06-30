-- Check 6: Known data quirks stay within a sane range
{{ config(severity = 'warn') }}

WITH quirks AS (
    SELECT 
        COUNT(*) FILTER (WHERE bur_total_debt < 0) as neg_debt,
        COUNT(*) FILTER (WHERE cc_utilisation > 1.0) as over_limit
    FROM {{ ref('stg_credit_data') }}
)
SELECT * 
FROM quirks
WHERE neg_debt > 5000 OR over_limit > 5000
