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
    FROM {{ ref('stg_credit_data') }}
    WHERE target IS NOT NULL
)
SELECT
    age_band,
    employment_band,
    COUNT(*)                       AS customers,
    ROUND(AVG(target)::numeric * 100, 2)    AS default_rate_pct
FROM banded
GROUP BY age_band, employment_band
