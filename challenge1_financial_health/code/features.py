"""Tạo biến cho Financial Health Score và Customer Engagement Score.

Mọi biến chỉ dùng dữ liệu xảy ra TRƯỚC mốc `cutoff`, để có thể kiểm chứng điểm số bằng kết quả SAU mốc.
Hai bộ biến không dùng chung cột nào:
  - Health: lịch sử trả nợ, khả năng trả, mức nợ, độ tin cậy tín dụng, độ ổn định (CONTRACT, CUSTOMER, MAX_OFFER).
  - Engagement: recency, frequency, tương tác, lead, sản phẩm (PAST_BEHAVIOR, LEADS, CALL, app).
  Số lần xin vay thuộc Health (áp lực tìm vốn), không dùng trong Engagement.
"""
import os
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parent.parent
# đặt biến môi trường BI10_CLEAN để chạy trên bộ dữ liệu clean khác (ví dụ eda/clean_team)
CLEAN = pathlib.Path(os.environ.get("BI10_CLEAN", ROOT / "eda" / "clean_team"))
DATA_END = "2025-06-05"          # ngày sau bản ghi cuối của CONTRACT / LEADS / PAST_BEHAVIOR
INSTALLMENT = "('CD','TW','CL','CW')"
# Sản phẩm có lịch sử trả quan sát được: trả góp + BNPL + thẻ (BNPL/CC không có số kỳ nhưng có đủ cờ FIRST..FOURTH_DUE).
# v1.0 chỉ dùng trả góp nên khách chỉ dùng BNPL/thẻ bị xếp nhầm vào nhóm thiếu lịch sử.
HISTORY_MIN_INSTALLMENTS = 3    # có bằng chứng trả nợ = có ít nhất 1 khoản vay (mọi sản phẩm) đã qua ≥ 3 kỳ trả
REPAY_OBSERVED = "('CD','TW','CL','CW','BNPL','CC')"
MATURE_DAYS = 125                # đủ để 4 kỳ trả góp hàng tháng đến hạn


def connect():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("SET enable_progress_bar = false")
    con.execute("SET memory_limit = '3GB'")   # máy 16GB: chừa RAM cho pandas/LightGBM
    for t in ["customer", "contract", "leads", "past", "call"]:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM '{(CLEAN / f'{t}_clean.parquet').as_posix()}'")
    return con


# ---------------------------------------------------------------------------
# Financial Health
# ---------------------------------------------------------------------------
def health_features(con, cutoff):
    """Biến sức khỏe tài chính tại mốc `cutoff` cho mọi khách."""
    c = f"TIMESTAMP '{cutoff}'"
    df = con.execute(f"""
    WITH k AS (
      SELECT *,
        -- cờ DUE và trạng thái là trạng thái cuối cùng của dữ liệu, nên chỉ dùng cho HĐ mà 4 kỳ đầu đã đến hạn trước mốc
        is_signed AND product IN {REPAY_OBSERVED} AND t_sign_filled < {c} - INTERVAL {MATURE_DAYS} DAY AS obs,
        is_signed AND t_sign_filled < {c} AND (
            (NOT is_revolving AND t_sign_filled + to_days(CAST(coalesce(tenor, 0) * 30 AS INT)) > {c})
            OR (is_revolving AND status = 'Active')) AS active_at_cut
      FROM contract WHERE t_apply < {c}),
    agg AS (
      SELECT skp_client,
        count(*)                                                     AS n_apps,
        count(*) FILTER (WHERE is_signed)                            AS n_signed,
        avg(is_rejected::INT)                                        AS refused_ratio,
        count(*) FILTER (WHERE is_rejected AND t_apply >= {c} - INTERVAL 365 DAY) AS n_refused_12m,
        count(*) FILTER (WHERE t_apply >= {c} - INTERVAL 180 DAY)   AS n_apply_6m,
        count(*) FILTER (WHERE t_apply >= {c} - INTERVAL 365 DAY)   AS n_apply_12m,
        count(*) FILTER (WHERE obs)                                  AS n_observed,
        avg(any_due::INT) FILTER (WHERE obs)                         AS pct_overdue,
        sum(d1) FILTER (WHERE obs)                                   AS n_fpd,
        sum(n_due_flags) FILTER (WHERE obs)                          AS n_due_total,
        -- chi tiết trả nợ trên các kỳ đã đến hạn (mỗi HĐ quan sát được 4 kỳ đầu)
        4 * count(*) FILTER (WHERE obs)                              AS n_installments_due,
        4 * count(*) FILTER (WHERE obs) - coalesce(sum(n_due_flags) FILTER (WHERE obs), 0) AS n_on_time,
        1 - sum(n_due_flags) FILTER (WHERE obs) / (4.0 * count(*) FILTER (WHERE obs)) AS on_time_rate,
        count(*) FILTER (WHERE obs AND any_due)                      AS n_contracts_late,
        count(*) FILTER (WHERE obs AND (d1*d2 + d2*d3 + d3*d4) > 0)  AS n_consec2_late,
        count(*) FILTER (WHERE obs AND (d1*d2*d3 + d2*d3*d4) > 0)    AS n_consec3_late,
        max(CASE WHEN d1*d2*d3*d4 = 1 THEN 4 WHEN d1*d2*d3 + d2*d3*d4 > 0 THEN 3
                 WHEN d1*d2 + d2*d3 + d3*d4 > 0 THEN 2 WHEN n_due_flags > 0 THEN 1 ELSE 0 END) FILTER (WHERE obs) AS max_late_streak,
        -- trễ một kỳ rồi các kỳ sau trả đúng hạn (tự khắc phục)
        count(*) FILTER (WHERE obs AND ((d1 = 1 AND d2 + d3 + d4 = 0) OR (d2 = 1 AND d3 + d4 = 0) OR (d3 = 1 AND d4 = 0))) AS n_cured,
        count(*) FILTER (WHERE obs AND any_due AND t_sign_filled >= {c} - INTERVAL 545 DAY) AS n_late_recent,
        max(n_due_flags) FILTER (WHERE obs)                          AS max_due_flags,
        count(*) FILTER (WHERE obs AND n_due_flags >= 2)             AS n_multi_due,
        date_diff('day', max(t_sign_filled) FILTER (WHERE obs AND any_due), {c}) AS days_since_overdue,
        count(*) FILTER (WHERE obs AND NOT any_due AND status IN ('Finished','Paid off')) AS n_clean_finished,
        max(is_writeoff::INT) FILTER (WHERE is_signed AND t_sign_filled < {c} - INTERVAL 365 DAY) AS ever_writeoff,
        count(*) FILTER (WHERE active_at_cut)                        AS n_active,
        count(DISTINCT product) FILTER (WHERE active_at_cut)         AS n_active_products,
        sum(annuity) FILTER (WHERE active_at_cut)                    AS active_annuity,
        date_diff('day', min(t_sign_filled) FILTER (WHERE is_signed), {c}) / 365.25 AS history_years,
        avg(flag_ins) FILTER (WHERE is_signed)                       AS pct_insured,
        avg(init_pay / (amt_credit + init_pay)) FILTER (WHERE is_signed AND init_pay > 0 AND amt_credit > 0) AS down_payment_ratio,
        arg_max(inc_main, (t_apply, case_id)) FILTER (WHERE inc_main IS NOT NULL)        AS income,
        arg_max(annuity_income_ratio, (t_apply, case_id)) FILTER (WHERE annuity_income_ratio IS NOT NULL) AS last_annuity_income,
        -- tỷ lệ so với thu nhập (dataset không có sao kê chi tiêu; AMT_BILLING và giá trị món hàng là đại diện gần nhất)
        arg_max(amt_billing / inc_main, (t_apply, case_id)) FILTER (WHERE is_signed AND amt_billing > 0 AND inc_main > 0) AS billing_income,
        avg(amt_billing / inc_main) FILTER (WHERE is_signed AND amt_billing > 0 AND inc_main > 0)        AS avg_billing_income,
        arg_max((amt_credit + init_pay) / inc_main, (t_apply, case_id))
            FILTER (WHERE is_signed AND product IN ('CD','TW') AND amt_credit > 0 AND inc_main > 0)       AS purchase_income,
        arg_max(init_pay / inc_main, (t_apply, case_id))
            FILTER (WHERE is_signed AND product IN ('CD','TW') AND init_pay >= 0 AND inc_main > 0)       AS init_pay_income,
        max(annuity_income_ratio) FILTER (WHERE is_signed)                                              AS max_annuity_income,
        arg_max(amt_credit / inc_main, (t_apply, case_id)) FILTER (WHERE is_signed AND amt_credit > 0 AND inc_main > 0) AS credit_income,
        sum(amt_credit) FILTER (WHERE is_signed AND t_sign_filled >= {c} - INTERVAL 365 DAY)            AS credit_12m,
        -- (mentor) đặc điểm hợp đồng: kỳ hạn, chi phí vay của HĐ trả góp gần nhất
        avg(tenor) FILTER (WHERE is_signed AND NOT is_revolving)                                        AS avg_tenor,
        max(tenor) FILTER (WHERE is_signed AND NOT is_revolving)                                        AS max_tenor,
        arg_max(tenor, (t_sign_filled, case_id)) FILTER (WHERE is_signed AND NOT is_revolving AND tenor > 0)        AS last_tenor,
        arg_max(annuity, (t_sign_filled, case_id)) FILTER (WHERE is_signed AND NOT is_revolving AND annuity > 0 AND amt_credit > 0 AND tenor > 0) AS last_annuity,
        arg_max(amt_credit, (t_sign_filled, case_id)) FILTER (WHERE is_signed AND NOT is_revolving AND annuity > 0 AND amt_credit > 0 AND tenor > 0) AS last_credit,
        arg_max(tenor, (t_sign_filled, case_id)) FILTER (WHERE is_signed AND NOT is_revolving AND annuity > 0 AND amt_credit > 0 AND tenor > 0) AS last_rate_tenor,
        -- (mentor) dư nợ ước tính = góp × số kỳ còn lại của HĐ đang chạy (dữ liệu không có dư nợ thật)
        sum(annuity * greatest(tenor - date_diff('month', t_sign_filled, {c}), 0))
            FILTER (WHERE active_at_cut AND NOT is_revolving AND annuity > 0)                           AS est_outstanding,
        -- (mentor) tỷ lệ dùng hạn mức thẻ: hóa đơn đầu tiên / hạn mức (chỉ thẻ tín dụng)
        arg_max(amt_billing / amt_credit, (t_sign_filled, case_id))
            FILTER (WHERE is_signed AND product = 'CC' AND amt_billing > 0 AND amt_credit > 0)          AS card_utilization,
        arg_max(income_type, (t_apply, case_id)) FILTER (WHERE income_type IS NOT NULL)  AS income_type,
        arg_max(profession, (t_apply, case_id)) FILTER (WHERE profession IS NOT NULL)    AS profession
      FROM k GROUP BY 1),
    -- (mentor) xu hướng trả nợ theo cửa sổ 3/6/12 tháng: ước ngày đến hạn kỳ k = ngày ký + k tháng,
    -- chỉ tính các kỳ đã đến hạn trước mốc ít nhất 15 ngày. Dữ liệu không có số ngày trễ (DPD) nên dùng cờ trễ của kỳ.
    inst AS (
      SELECT skp_client, t_sign_filled + to_months(k) AS due, flag
      FROM k, (VALUES (1), (2), (3), (4)) v(k),
           LATERAL (SELECT CASE v.k WHEN 1 THEN d1 WHEN 2 THEN d2 WHEN 3 THEN d3 ELSE d4 END AS flag)
      WHERE is_signed AND product IN {REPAY_OBSERVED} AND coalesce(tenor, 4) >= v.k   -- BNPL/CC: không có số kỳ, dùng 4 kỳ đầu
        AND t_sign_filled + to_months(v.k) < {c} - INTERVAL 15 DAY),
    trend AS (
      SELECT skp_client,
        count(*) FILTER (WHERE due >= {c} - INTERVAL 3 MONTH)                 AS n_inst_due_3m,
        sum(flag) FILTER (WHERE due >= {c} - INTERVAL 3 MONTH)                AS n_inst_late_3m,
        count(*) FILTER (WHERE due >= {c} - INTERVAL 6 MONTH)                 AS n_inst_due_6m,
        sum(flag) FILTER (WHERE due >= {c} - INTERVAL 6 MONTH)                AS n_inst_late_6m,
        count(*) FILTER (WHERE due >= {c} - INTERVAL 12 MONTH)                AS n_inst_due_12m,
        sum(flag) FILTER (WHERE due >= {c} - INTERVAL 12 MONTH)               AS n_inst_late_12m,
        avg(flag) FILTER (WHERE due >= {c} - INTERVAL 6 MONTH)                AS late_rate_6m,
        avg(flag) FILTER (WHERE due < {c} - INTERVAL 6 MONTH AND due >= {c} - INTERVAL 24 MONTH) AS late_rate_6_24m
      FROM inst GROUP BY 1),
    offer AS (
      SELECT skp_client, arg_max(max_offer, (t_created, max_offer)) AS max_offer   -- trùng thời điểm: lấy hạn mức cao nhất
      FROM leads WHERE t_created < {c} AND max_offer IS NOT NULL GROUP BY 1)
    SELECT cu.skp_client,
      date_diff('year', cu.date_birth, {c}) AS age,
      cu.education, cu.cnt_children,
      a.* EXCLUDE (skp_client),
      a.active_annuity / NULLIF(a.income, 0) AS active_annuity_income,
      a.credit_12m / NULLIF(a.income * 12, 0) AS credit_12m_income,
      a.est_outstanding / NULLIF(a.income, 0) AS est_outstanding_income,
      a.last_annuity * a.last_rate_tenor / NULLIF(a.last_credit, 0) - 1 AS last_loan_cost_ratio,
      tr.n_inst_due_3m, tr.n_inst_late_3m, tr.n_inst_due_6m, tr.n_inst_late_6m, tr.n_inst_due_12m, tr.n_inst_late_12m,
      tr.late_rate_6m, tr.late_rate_6_24m,
      tr.late_rate_6m - tr.late_rate_6_24m AS late_trend,
      CASE WHEN coalesce(a.n_observed, 0) = 0 THEN '1. Chưa có lịch sử quan sát được'
           WHEN a.days_since_overdue IS NULL THEN '2. Không trễ hạn lần nào'
           WHEN a.days_since_overdue <= 365 THEN '5. Trễ trong 1 năm qua'
           WHEN a.days_since_overdue <= 1095 THEN '4. Trễ cách đây 1–3 năm'
           ELSE '3. Trễ cách đây trên 3 năm' END AS overdue_recency,
      o.max_offer,
      o.max_offer / NULLIF(a.income, 0) AS offer_to_income
    FROM customer cu
    LEFT JOIN agg a USING (skp_client)
    LEFT JOIN offer o USING (skp_client)
    LEFT JOIN trend tr USING (skp_client)
    -- giữ cả khách có ngày sinh không hợp lệ: điểm sức khỏe không dùng tuổi
    """).df()
    df["implied_monthly_rate"] = implied_rate(df.last_credit, df.last_annuity, df.last_rate_tenor)
    return df


def implied_rate(principal, annuity, n, iters=60):
    """(mentor) lãi suất tháng suy ngược từ công thức niên kim: annuity = P·r / (1 − (1+r)^−n). Giải bằng chia đôi.
    Dữ liệu không có lãi suất; số này có thể gồm cả phí và bảo hiểm trả góp."""
    import numpy as np
    P, A, N = (np.asarray(x, dtype=float) for x in (principal, annuity, n))
    ok = (P > 0) & (A > 0) & (N > 0) & (A * N > P)
    lo, hi = np.full(P.shape, 1e-9), np.full(P.shape, 1.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        pay = P * mid / (1 - (1 + mid) ** (-np.where(ok, N, 1)))
        hi = np.where(pay > A, mid, hi)
        lo = np.where(pay > A, lo, mid)
    r = (lo + hi) / 2
    return np.where(ok, r, np.where((P > 0) & (A * N <= P) & (A > 0), 0.0, np.nan))


def health_label(con, cutoff, months=2):
    """Nhãn xấu: khách có HĐ trả góp ký trong [cutoff, cutoff + months) bị trễ ở 4 kỳ đầu hoặc bị xóa nợ.
    Chỉ dùng HĐ đã đủ độ chín (ký trước 01/02/2025)."""
    c = f"TIMESTAMP '{cutoff}'"
    return con.execute(f"""
      SELECT skp_client, max((any_due OR is_writeoff)::INT) AS bad, count(*) AS n_label_contracts
      FROM contract
      WHERE is_signed AND product IN {INSTALLMENT}
        AND t_sign_filled >= {c} AND t_sign_filled < {c} + INTERVAL {months} MONTH
        AND t_sign_filled < TIMESTAMP '2025-02-01'
      GROUP BY 1""").df()


# ---------------------------------------------------------------------------
# Financial Health – biến bổ sung theo 7 nhóm gợi ý (thu nhập, thu–chi, nợ, lịch sử, dự phòng, áp lực vay, bảo vệ)
# ---------------------------------------------------------------------------
def extra_features(con, cutoff):
    c = f"TIMESTAMP '{cutoff}'"
    return con.execute(f"""
    WITH k AS (
      SELECT *,
        is_signed AND product IN {REPAY_OBSERVED} AND t_sign_filled < {c} - INTERVAL {MATURE_DAYS} DAY AS obs,
        is_signed AND t_sign_filled < {c} AND (
            (NOT is_revolving AND t_sign_filled + to_days(CAST(coalesce(tenor, 0) * 30 AS INT)) > {c})
            OR (is_revolving AND status = 'Active')) AS active_at_cut,
        is_signed AND t_sign_filled < {c} AS signed_before
      FROM contract WHERE t_apply < {c}),
    -- 1. Thu nhập và độ ổn định (thu nhập khai báo qua các lần nộp hồ sơ trong 3 năm)
    inc AS (
      SELECT skp_client,
        stddev_samp(inc_main) / NULLIF(avg(inc_main), 0)                     AS inc_cv,
        count(DISTINCT inc_main)                                             AS n_income_values,
        arg_max(inc_main, (t_apply, case_id)) / NULLIF(arg_min(inc_main, (t_apply, case_id)), 0) AS inc_growth,
        arg_max(inc_main, (t_apply, case_id)) / NULLIF(max(inc_main), 0)     AS inc_vs_peak,
        count(DISTINCT profession)                                           AS n_professions,
        count(DISTINCT income_type)                                          AS n_income_types,
        date_diff('day', min(t_apply), {c}) / 365.25                         AS income_years_declared
      FROM k WHERE inc_main > 0 AND t_apply >= {c} - INTERVAL 1095 DAY GROUP BY 1),
    ag AS (
      SELECT skp_client,
        -- 3. Gánh nặng nợ
        count(*) FILTER (WHERE active_at_cut AND is_revolving)               AS n_active_revolving,
        count(DISTINCT product) FILTER (WHERE active_at_cut)                 AS n_active_products2,
        sum(amt_credit) FILTER (WHERE active_at_cut AND amt_credit > 0)      AS active_credit,
        -- 4. Lịch sử thanh toán (bổ sung)
        avg(d4) FILTER (WHERE obs)                                           AS late4_rate,
        avg(d1) FILTER (WHERE obs)                                           AS fpd_rate,
        -- 5. Dự phòng tài chính (đại diện: khả năng trả trước bằng tiền mặt)
        avg(init_pay) FILTER (WHERE signed_before AND init_pay > 0)          AS avg_init_pay,
        max(init_pay) FILTER (WHERE signed_before AND t_sign_filled >= {c} - INTERVAL 730 DAY) AS max_init_pay_24m,
        avg((init_pay > 0)::INT) FILTER (WHERE signed_before AND product IN ('CD','TW')) AS pct_with_downpay,
        -- 6. Áp lực vay nợ
        date_diff('day', max(t_apply), {c})                                  AS days_since_last_apply,
        count(*) FILTER (WHERE t_apply >= {c} - INTERVAL 30 DAY)             AS n_apply_1m,
        count(*) FILTER (WHERE t_apply >= {c} - INTERVAL 90 DAY)             AS n_apply_3m,
        count(DISTINCT product) FILTER (WHERE t_apply >= {c} - INTERVAL 180 DAY) AS n_products_applied_6m,
        count(*) FILTER (WHERE is_rejected AND t_apply >= {c} - INTERVAL 90 DAY) AS n_refused_3m,
        count(*) FILTER (WHERE is_cancelled AND t_apply >= {c} - INTERVAL 365 DAY) AS n_cancelled_12m,
        arg_max(is_rejected::INT, (t_apply, case_id))                        AS last_app_rejected,
        count(*) FILTER (WHERE signed_before AND t_sign_filled >= {c} - INTERVAL 365 DAY) AS n_signed_12m,
        sum(amt_credit) FILTER (WHERE signed_before AND t_sign_filled >= {c} - INTERVAL 365 DAY)  AS credit_last12,
        sum(amt_credit) FILTER (WHERE signed_before AND t_sign_filled < {c} - INTERVAL 365 DAY
                                      AND t_sign_filled >= {c} - INTERVAL 730 DAY)               AS credit_prev12,
        date_diff('day', max(t_sign_filled) FILTER (WHERE signed_before), {c}) AS days_since_last_signed,
        -- 7. Kế hoạch và bảo vệ
        avg(flag_ins) FILTER (WHERE signed_before)                           AS pct_insured2,
        arg_max(flag_ins, (t_sign_filled, case_id)) FILTER (WHERE signed_before) AS last_insured
      FROM k GROUP BY 1),
    last3 AS (   -- tỷ lệ HĐ trễ trong 3 HĐ quan sát được gần nhất
      SELECT skp_client, avg(any_due::INT) AS late_rate_last3
      FROM (SELECT skp_client, any_due, row_number() OVER (PARTITION BY skp_client ORDER BY t_sign_filled DESC, case_id DESC) rn
            FROM k WHERE obs) WHERE rn <= 3 GROUP BY 1),
    -- 6. Hạn mức công ty đề xuất thay đổi thế nào (bị hạ hạn mức = rủi ro tăng)
    off AS (
      SELECT skp_client,
        arg_max(max_offer, (t_created, max_offer)) / NULLIF(arg_max(max_offer, (t_created, max_offer))
            FILTER (WHERE t_created < {c} - INTERVAL 180 DAY), 0)            AS offer_change_6m,
        count(*) FILTER (WHERE t_created >= {c} - INTERVAL 365 DAY)          AS n_offers_12m,
        max(max_offer) FILTER (WHERE t_created >= {c} - INTERVAL 365 DAY)    AS max_offer_12m
      FROM leads WHERE t_created < {c} AND max_offer IS NOT NULL GROUP BY 1),
    -- 6. Số lần đi chấm điểm hồ sơ (PAST_BEHAVIOR) — trùng nguồn với Engagement, chỉ để thử
    sc AS (
      SELECT skp_client,
        count(*) FILTER (WHERE t_event >= {c} - INTERVAL 90 DAY)             AS n_scoring_3m,
        count(*) FILTER (WHERE t_event >= {c} - INTERVAL 180 DAY)            AS n_scoring_6m,
        count(DISTINCT channel) FILTER (WHERE t_event >= {c} - INTERVAL 180 DAY) AS n_scoring_channels_6m,
        count(DISTINCT product) FILTER (WHERE t_event >= {c} - INTERVAL 180 DAY) AS n_scoring_products_6m
      FROM past WHERE is_scoring AND t_event < {c} AND t_event >= {c} - INTERVAL 365 DAY GROUP BY 1)
    SELECT skp_client, inc.* EXCLUDE (skp_client), ag.* EXCLUDE (skp_client), last3.late_rate_last3,
           off.* EXCLUDE (skp_client), sc.* EXCLUDE (skp_client)
    FROM ag FULL JOIN inc USING (skp_client) LEFT JOIN last3 USING (skp_client)
    FULL JOIN off USING (skp_client) FULL JOIN sc USING (skp_client)
    """).df()


def health_features_full(con, cutoff, with_label=True):
    """Biến Health gốc + biến bổ sung theo 7 nhóm (extra_features)."""
    import numpy as np
    d = health_features(con, cutoff)
    if with_label:
        d = d.merge(health_label(con, cutoff), on="skp_client")
    d = d.merge(extra_features(con, cutoff), on="skp_client", how="left")
    d["cutoff"] = cutoff
    # 2. Cân đối thu–chi: thu nhập còn lại sau khi trả các khoản góp đang chạy (không có dữ liệu chi tiêu)
    d["residual_income"] = d.income - d.active_annuity.fillna(0)
    d["active_credit_income"] = d.active_credit / (d.income * 12).replace(0, np.nan)
    d["credit_growth"] = d.credit_last12 / d.credit_prev12.replace(0, np.nan)
    for col in ["n_scoring_3m", "n_scoring_6m", "n_scoring_channels_6m", "n_scoring_products_6m"]:
        d[col] = d[col].fillna(0)
    return d


# ---------------------------------------------------------------------------
# Engagement
# ---------------------------------------------------------------------------
def engagement_features(con, cutoff):
    """Biến gắn kết tại mốc `cutoff`, cửa sổ nhìn lại 12 tháng."""
    c = f"TIMESTAMP '{cutoff}'"
    lb = f"{c} - INTERVAL 365 DAY"
    return con.execute(f"""
    WITH ev AS (   -- mọi lần khách chủ động tương tác
      SELECT skp_client, t_event t, 'scoring' kind, product FROM past WHERE t_event < {c} AND t_event >= {lb}
      UNION ALL SELECT skp_client, t_created, 'self_lead', product FROM leads
        WHERE is_self_initiated AND t_created < {c} AND t_created >= {lb}
      UNION ALL SELECT skp_client, t_start, 'inbound', product FROM call
        WHERE is_inbound AND t_start < {c} AND t_start >= {lb}
),
    e AS (
      SELECT skp_client,
        date_diff('day', max(t), {c})                                   AS days_since_last,
        count(*) FILTER (WHERE t >= {c} - INTERVAL 90 DAY)              AS n_events_90d,
        count(*)                                                        AS n_events_365d,
        count(DISTINCT date_trunc('month', t))                          AS active_months,
        count(*) FILTER (WHERE kind = 'scoring')                        AS n_scoring_365d,
        count(DISTINCT product) FILTER (WHERE product <> 'UNKNOWN')     AS n_products_touched
      FROM ev GROUP BY 1),
    p AS (
      SELECT skp_client, count(*) FILTER (WHERE is_self_service) AS n_self_service,
             count(DISTINCT channel) AS n_channels
      FROM past WHERE t_event < {c} AND t_event >= {lb} GROUP BY 1),
    l AS (
      SELECT skp_client, count(*) n_leads, sum(is_self_initiated::INT) n_self_leads,
             avg(is_deal::INT) deal_rate, avg((reason = 'Not Interest')::INT) not_interest_rate
      FROM leads WHERE t_created < {c} AND t_created >= {lb} GROUP BY 1),
    cl AS (
      SELECT skp_client, count(*) n_calls, sum(is_connected::INT) n_connected,
             avg(is_connected::INT) connect_rate, sum(is_inbound::INT) n_inbound,
             avg((response = 'Refuse')::INT) FILTER (WHERE response IS NOT NULL AND NOT is_own_code) refuse_rate
      FROM call WHERE t_start < {c} AND t_start >= {lb} AND in_data_window GROUP BY 1)
    SELECT cu.skp_client, cu.avg_session,
      e.* EXCLUDE (skp_client), p.* EXCLUDE (skp_client), l.* EXCLUDE (skp_client), cl.* EXCLUDE (skp_client)
    FROM customer cu
    LEFT JOIN e USING (skp_client) LEFT JOIN p USING (skp_client)
    LEFT JOIN l USING (skp_client) LEFT JOIN cl USING (skp_client)
    WHERE NOT cu.dq_invalid_age""").df()


def engagement_outcome(con, cutoff, days=60):
    """Kết quả kinh doanh sau mốc: khách nộp / ký hồ sơ mới trong `days` ngày, và giá trị ký."""
    c = f"TIMESTAMP '{cutoff}'"
    return con.execute(f"""
      SELECT skp_client, 1 AS applied, max(is_signed::INT) AS signed,
             sum(coalesce(amt_credit, amt_billing, 0)) FILTER (WHERE is_signed) AS signed_value
      FROM contract WHERE t_apply >= {c} AND t_apply < {c} + INTERVAL {days} DAY GROUP BY 1""").df()
