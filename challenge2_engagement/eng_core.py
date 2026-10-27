# %% [code] setup-views
# Typed views on top of the team's Clean_data parquet. Every column is cast defensively, so the
# notebook works whether the parquet came out of Excel as typed columns or as text.
def create_views(con, pq_dir):
    p = lambda name: (pq_dir / f"{name}.parquet").as_posix()
    con.execute("""
        CREATE OR REPLACE MACRO num(x) AS TRY_CAST(CAST(x AS VARCHAR) AS DOUBLE);
        CREATE OR REPLACE MACRO flag(x) AS coalesce(num(x), 0) = 1;
        CREATE OR REPLACE MACRO ts(x) AS coalesce(
            TRY_CAST(CAST(x AS VARCHAR) AS TIMESTAMP),
            TIMESTAMP '1899-12-30' + to_seconds(CAST(num(x) * 86400 AS BIGINT)));   -- Excel serial date fallback
        CREATE OR REPLACE MACRO txt(x) AS NULLIF(NULLIF(trim(CAST(x AS VARCHAR)), ''), 'XNA');
    """)
    con.execute(f"""CREATE OR REPLACE VIEW customer AS
        SELECT CAST(num(SKP_CLIENT) AS BIGINT) AS client,
               coalesce(num(AVG_SESSION_PER_WEEK_2025), 0) AS app_sessions,
               flag(FLAG_GMA_USER) AS app_user
        FROM '{p("CUSTOMER")}'""")
    con.execute(f"""CREATE OR REPLACE VIEW contract AS
        SELECT CAST(num(SKP_CLIENT) AS BIGINT) AS client,
               coalesce(txt(PRODUCT_GROUP), 'UNKNOWN') AS product,
               txt(STATUS_GROUP) AS status_group,
               flag(IS_SIGNED) AS is_signed, flag(IS_REVOLVING) AS is_revolving,
               ts(APPLY_CONTRACT_TIME) AS t_apply,
               coalesce(ts(SIGN_CONTRACT_TIME), ts(APPLY_CONTRACT_TIME)) AS t_sign,
               num(PAYMENT_NUM) AS tenor,
               CASE WHEN num(AMT_CREDIT) > 0 THEN num(AMT_CREDIT) ELSE num(AMT_BILLING) END AS amount
        FROM '{p("CONTRACT")}'""")
    con.execute(f"""CREATE OR REPLACE VIEW leads AS
        SELECT CAST(num(SKP_CLIENT) AS BIGINT) AS client,
               ts(DTIME_CREATED) AS t_created,
               CASE WHEN year(ts(DATE_ACCEPTED)) < 2900 THEN ts(DATE_ACCEPTED) END AS t_accepted,  -- 3000-01-01 = sentinel
               CASE WHEN year(ts(DATE_ACCEPTED)) < 2900 THEN num(ACCEPT_LAG_HOURS) END AS accept_lag_h,
               txt(LEAD_OUTCOME) AS outcome, flag(IS_DEAL) AS is_deal,
               txt(LEAD_SOURCE) AS source, txt(LEAD_SOURCE) IN ('SOB', 'GMA', 'LANDING PAGE') AS is_self,
               txt(PRODUCT_GROUP) AS product
        FROM '{p("LEADS")}'""")
    con.execute(f"""CREATE OR REPLACE VIEW behavior AS
        SELECT CAST(num(SKP_CLIENT) AS BIGINT) AS client, ts(DTIME_EVENT) AS t_event,
               txt(CHANNEL) AS channel, txt(EVENT_TYPE) AS event_type, txt(PRODUCT_GROUP) AS product
        FROM '{p("PAST_BEHAVIOR")}'""")
    con.execute(f"""CREATE OR REPLACE VIEW call AS
        SELECT CAST(num(SKP_CLIENT) AS BIGINT) AS client, ts(TIME_OF_CALL_START) AS t_call,
               txt(CALL_TYPE) = 'Inbound' AS is_inbound, flag(IS_CONNECTED) AS is_connected,
               flag(IS_COLLECTION_CALL) AS is_collection, txt(RESPONSE_GROUP) AS response
        FROM '{p("CALL")}'
        WHERE NOT flag(FLAG_DUP_CALL_ID)""")


# %% [code] features
CALL_WINDOW_DAYS = 30   # CALL starts 2025-01-01 -> same short window at every as-of date keeps call features comparable

def build_features(con, as_of):
    """All candidate engagement variables measured strictly BEFORE `as_of` (no look-ahead)."""
    a = f"TIMESTAMP '{as_of}'"
    d12, d24 = f"{a} - INTERVAL 365 DAY", f"{a} - INTERVAL 730 DAY"
    d3, d1 = f"{a} - INTERVAL 90 DAY", f"{a} - INTERVAL 30 DAY"
    dc = f"{a} - INTERVAL {CALL_WINDOW_DAYS} DAY"
    days = lambda t: f"date_diff('day', {t}, {a})"
    sql = f"""
    WITH
    beh AS (
        SELECT client,
               {days("max(t_event) FILTER (WHERE event_type = 'SCORING')")} AS rec_scoring_days,
               count(*) FILTER (WHERE event_type = 'SCORING' AND t_event >= {d12}) AS n_scoring_12m,
               count(DISTINCT product) FILTER (WHERE event_type = 'SCORING' AND t_event >= {d12}) AS n_products_scored_12m,
               count(DISTINCT channel) FILTER (WHERE t_event >= {d12}) AS n_channels_12m,
               avg(CASE WHEN channel IN ('APP', 'WEB') THEN 1.0 ELSE 0 END) FILTER (WHERE t_event >= {d12}) AS self_service_share_12m,
               max(t_event) FILTER (WHERE event_type = 'SCORING') AS last_scoring
        FROM behavior WHERE t_event < {a} GROUP BY 1),
    lds AS (
        SELECT client,
               count(*) FILTER (WHERE t_created >= {d12}) AS n_leads_12m,
               count(*) FILTER (WHERE t_created >= {d12} AND t_accepted IS NOT NULL AND t_accepted < {a}) AS n_leads_accepted_12m,
               count(*) FILTER (WHERE t_created >= {d12} AND outcome = 'NOT_INTEREST') AS n_leads_not_interest_12m,
               count(*) FILTER (WHERE t_created >= {d12} AND is_deal) AS n_lead_deals_12m,
               count(*) FILTER (WHERE t_created >= {d12} AND is_self) AS n_self_leads_12m,
               count(*) FILTER (WHERE t_created >= {d12} AND NOT is_self) AS n_company_leads_12m,
               median(accept_lag_h) FILTER (WHERE t_created >= {d12} AND t_accepted < {a}) AS median_accept_lag_h_12m,
               {days('max(t_created)')} AS rec_lead_days,
               max(t_created) FILTER (WHERE is_self) AS last_self_lead
        FROM leads WHERE t_created < {a} GROUP BY 1),
    cl AS (
        SELECT client,
               count(*) FILTER (WHERE NOT is_inbound) AS n_sales_calls_30d,
               avg(is_connected::INT) FILTER (WHERE NOT is_inbound) AS call_answer_rate_30d,
               count(*) FILTER (WHERE is_inbound) AS n_inbound_calls_30d,
               count(*) FILTER (WHERE is_connected AND response = 'REFUSE') AS n_call_refuse_30d,
               count(*) FILTER (WHERE response = 'TERMINATION_REQUEST') AS n_termination_req_30d
        FROM call WHERE NOT is_collection AND t_call >= {dc} AND t_call < {a} GROUP BY 1),
    ct AS (   -- contracts known at as_of (applied before it)
        SELECT * REPLACE (is_signed AND t_sign < {a} AS is_signed),   -- a loan counts as booked only once signed before as_of
               t_sign + to_months(CAST(coalesce(tenor, 0) AS INT)) AS t_end
        FROM contract WHERE t_apply < {a}),
    apps AS (
        SELECT client,
               {days('max(t_sign) FILTER (WHERE is_signed)')} AS rec_booking_days,
               count(*) FILTER (WHERE is_signed AND t_apply >= {d24}) AS n_signed_24m,
               count(DISTINCT product) FILTER (WHERE is_signed AND product <> 'UNKNOWN') AS n_products_held,
               max(CASE WHEN is_signed AND is_revolving THEN 1 ELSE 0 END) AS has_revolving,
               count(*) FILTER (WHERE is_signed AND (is_revolving OR t_end >= {a})) AS n_running,
               count(*) FILTER (WHERE t_apply >= {d24}) AS n_apps_24m,
               avg(CASE WHEN status_group = 'CANCELLED' THEN 1.0 ELSE 0 END) FILTER (WHERE t_apply >= {d24}) AS cancel_rate_24m,
               avg(CASE WHEN status_group = 'REJECTED' THEN 1.0 ELSE 0 END) FILTER (WHERE t_apply >= {d24}) AS reject_rate_24m
        FROM ct GROUP BY 1),
    gaps AS (   -- median days between consecutive applications (application rhythm)
        SELECT client, median(gap) AS median_app_gap_days
        FROM (SELECT client, date_diff('day', lag(t_apply) OVER (PARTITION BY client ORDER BY t_apply), t_apply) AS gap FROM ct)
        WHERE gap IS NOT NULL GROUP BY 1),
    rep AS (   -- repeat rate: share of loans that ended before as_of and were followed by a new application
        SELECT e.client, avg(CASE WHEN EXISTS (SELECT 1 FROM ct n WHERE n.client = e.client AND n.t_apply > e.t_end) THEN 1.0 ELSE 0 END) AS repeat_rate
        FROM ct e WHERE e.is_signed AND NOT e.is_revolving AND e.tenor > 0 AND e.t_end < {a} GROUP BY 1),
    acts AS (     -- customer-initiated actions only: scoring, self-created leads, applications (not company steps)
        SELECT client, t_event AS t FROM behavior WHERE t_event >= {d12} AND t_event < {a}
        UNION ALL SELECT client, t_created FROM leads WHERE is_self AND t_created >= {d12} AND t_created < {a}
        UNION ALL SELECT client, t_apply FROM contract WHERE t_apply >= {d12} AND t_apply < {a}),
    months AS (
        SELECT client, count(DISTINCT date_trunc('month', t)) AS active_months_12m,
               avg(CASE WHEN t >= {d3} THEN 1.0 ELSE 0 END) AS recent_activity_share
        FROM acts GROUP BY 1),
    fun AS (      -- personal funnel: share of scoring events followed by an application within 30 days
        SELECT s.client, avg(CASE WHEN p.t_apply <= s.t_event + INTERVAL 30 DAY THEN 1.0 ELSE 0 END) AS scoring_to_app_rate_12m
        FROM (SELECT client, t_event FROM behavior WHERE event_type = 'SCORING' AND t_event >= {d12} AND t_event < {d1}) s
        ASOF LEFT JOIN (SELECT client, t_apply FROM contract WHERE t_apply < {a}) p
            ON s.client = p.client AND p.t_apply >= s.t_event
        GROUP BY 1)
    SELECT c.client, c.app_sessions, c.app_user,
           {days('greatest(b.last_scoring, l.last_self_lead)')} AS rec_touch_days,
           b.rec_scoring_days, a.rec_booking_days, l.rec_lead_days,
           coalesce(m.active_months_12m, 0) AS active_months_12m, m.recent_activity_share,
           coalesce(b.n_scoring_12m, 0) AS n_scoring_12m,
           coalesce(a.n_signed_24m, 0) AS n_signed_24m,
           coalesce(k.n_sales_calls_30d, 0) AS n_sales_calls_30d, k.call_answer_rate_30d,
           coalesce(k.n_inbound_calls_30d, 0) AS n_inbound_calls_30d,
           coalesce(k.n_call_refuse_30d, 0) AS n_call_refuse_30d,
           coalesce(k.n_termination_req_30d, 0) AS n_termination_req_30d,
           coalesce(b.n_channels_12m, 0) AS n_channels_12m, b.self_service_share_12m,
           coalesce(l.n_leads_12m, 0) AS n_leads_12m,
           l.n_leads_accepted_12m / nullif(l.n_leads_12m, 0) AS lead_accept_rate_12m,
           l.n_leads_not_interest_12m / nullif(l.n_leads_12m, 0) AS lead_not_interest_rate_12m,
           coalesce(l.n_lead_deals_12m, 0) AS n_lead_deals_12m,
           coalesce(l.n_self_leads_12m, 0) AS n_self_leads_12m,
           coalesce(l.n_company_leads_12m, 0) AS n_company_leads_12m,
           l.median_accept_lag_h_12m, f.scoring_to_app_rate_12m,
           coalesce(b.n_products_scored_12m, 0) AS n_products_scored_12m,
           coalesce(a.n_products_held, 0) AS n_products_held,
           coalesce(a.has_revolving, 0) AS has_revolving,
           coalesce(a.n_running, 0) AS n_running,
           r.repeat_rate,
           coalesce(a.n_apps_24m, 0) AS n_apps_24m, a.cancel_rate_24m, a.reject_rate_24m,
           g.median_app_gap_days
    FROM customer c
    LEFT JOIN beh b USING (client) LEFT JOIN lds l USING (client) LEFT JOIN cl k USING (client)
    LEFT JOIN apps a USING (client) LEFT JOIN gaps g USING (client) LEFT JOIN rep r USING (client)
    LEFT JOIN months m USING (client) LEFT JOIN fun f USING (client)
    ORDER BY c.client"""
    df = con.execute(sql).df()
    for col in df.columns:
        if col != "client":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def build_outcomes(con, as_of, end):
    """Business outcomes in the window [as_of, end): conversion = new signed loan applied in the window or a lead deal."""
    a, e = f"TIMESTAMP '{as_of}'", f"TIMESTAMP '{end}'"
    return con.execute(f"""
    WITH k AS (SELECT client, count(*) AS new_apps, count(*) FILTER (WHERE is_signed) AS new_booked,
                      sum(amount) FILTER (WHERE is_signed) AS new_booked_amount
               FROM contract WHERE t_apply >= {a} AND t_apply < {e} GROUP BY 1),
         d AS (SELECT client, count(*) AS new_lead_deals FROM leads WHERE is_deal AND t_created >= {a} AND t_created < {e} GROUP BY 1)
    SELECT c.client,
           CASE WHEN coalesce(k.new_booked, 0) + coalesce(d.new_lead_deals, 0) > 0 THEN 1 ELSE 0 END AS converted,
           CASE WHEN coalesce(k.new_apps, 0) > 0 THEN 1 ELSE 0 END AS applied,
           coalesce(k.new_booked_amount, 0) AS new_booked_amount
    FROM customer c LEFT JOIN k USING (client) LEFT JOIN d USING (client) ORDER BY c.client""").df()
