-- Check 3: Default rate stays in the expected band (7-9%)
SELECT
    ROUND(AVG(target)::numeric * 100, 2) as default_rate
FROM {{ ref('stg_credit_data') }}
WHERE target IS NOT NULL
HAVING ROUND(AVG(target)::numeric * 100, 2) < 7.0
    OR ROUND(AVG(target)::numeric * 100, 2) > 9.0
