-- Check 5: Ages are plausible (18-100 years)
SELECT
    days_birth
FROM {{ ref('stg_credit_data') }}
WHERE (days_birth / 365.0) < 18 
   OR (days_birth / 365.0) > 100
