WITH raw_data AS (
    SELECT * FROM {{ source('credit_risk', 'credit_data') }}
)
SELECT
    amt_income_total,
    amt_credit,
    amt_goods_price,
    days_birth,
    days_employed,
    income_credit_ratio,
    employment_age_ratio,
    annuity_income_ratio,
    bur_total_debt,
    bur_num_credits,
    bur_max_overdue,
    prev_num_applications,
    prev_approval_rate,
    inst_late_rate,
    inst_days_late_mean,
    cc_utilisation,
    cc_dpd_max,
    pos_sk_dpd_max,
    pos_completion_rate,
    target
FROM raw_data
