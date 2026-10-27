"""Chuyển bộ dữ liệu clean dạng Excel (nhiều sheet / bảng) sang parquet, rồi ánh xạ về đúng tên cột mà
features.py dùng. Kết quả: eda/clean_team/<bang>_clean.parquet

Chạy: python -u solution/load_team_clean.py "<thư mục chứa *_clean.xlsx>"
"""
import pathlib
import sys
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from python_calamine import CalamineWorkbook

SRC = pathlib.Path(sys.argv[1])
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "eda" / "clean_team" / "raw"
OUT = ROOT / "eda" / "clean_team"
RAW.mkdir(parents=True, exist_ok=True)
TABLES = ["CUSTOMER", "CONTRACT", "LEADS", "PAST_BEHAVIOR", "CALL"]


def to_text(v):
    if v is None or v == "":
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


# 1. Excel -> parquet (mọi cột dạng chuỗi, ép kiểu sau bằng DuckDB)
for t in TABLES:
    if (RAW / t).exists() and any((RAW / t).glob("*.parquet")):
        print(f"{t}: đã có parquet, bỏ qua", flush=True)
        continue
    (RAW / t).mkdir(exist_ok=True)
    t0 = time.time()
    wb = CalamineWorkbook.from_path(str(SRC / f"{t}_clean.xlsx"))
    total = 0
    for i, name in enumerate(wb.sheet_names):
        rows = wb.get_sheet_by_name(name).iter_rows()
        header = [str(h) for h in next(rows)]
        cols = [[] for _ in header]
        for r in rows:
            for j in range(len(header)):
                cols[j].append(to_text(r[j]) if j < len(r) else None)
        tbl = pa.table({h: pa.array(c, type=pa.string()) for h, c in zip(header, cols)})
        pq.write_table(tbl, RAW / t / f"part{i}.parquet")
        total += tbl.num_rows
        del cols, tbl
    print(f"{t}: {total:,} dòng, {len(wb.sheet_names)} sheet ({time.time() - t0:.0f}s)", flush=True)

# 2. Ánh xạ sang lược đồ mà pipeline dùng
con = duckdb.connect()
for t in TABLES:
    con.execute(f"CREATE VIEW r_{t} AS SELECT * FROM '{(RAW / t).as_posix()}/*.parquet'")
ts = lambda c: f"TRY_CAST(NULLIF({c}, '') AS TIMESTAMP)"
num = lambda c: f"TRY_CAST(NULLIF({c}, '') AS DOUBLE)"
flag = lambda c: f"coalesce(TRY_CAST(TRY_CAST({c} AS DOUBLE) AS INT), 0) = 1"
nx = lambda c: f"NULLIF(NULLIF({c}, 'XNA'), '')"
SIGNED = "('Active','Finished','Paid off','Signed','Written off')"

con.execute(f"""COPY (
  SELECT TRY_CAST(TRY_CAST(SKP_CLIENT AS DOUBLE) AS BIGINT) skp_client, {nx('NAME_GENDER')} gender,
    {nx('NAME_EDUCATION_TYPE')} education, {ts('DATE_BIRTH')} date_birth, {num('AGE')} age,
    {flag('FLAG_INVALID_AGE')} dq_invalid_age, {num('CNT_CHILDREN')} cnt_children, ADDRESS address,
    PROVINCE province, REGION region, {num('AVG_SESSION_PER_WEEK_2025')} avg_session, {flag('FLAG_GMA_USER')} is_app_user
  FROM r_CUSTOMER) TO '{(OUT / 'customer_clean.parquet').as_posix()}' (FORMAT parquet)""")

con.execute(f"""COPY (
  WITH c0 AS (SELECT *, {ts('APPLY_CONTRACT_TIME')} t_apply, {ts('APPROVE_CONTRACT_TIME')} t_approve,
                    {ts('SIGN_CONTRACT_TIME')} t_sign, {num('ANNUITY')} ann, {num('AMT_INCOME_MAIN')} inc,
                    {num('AMT_CREDIT')} cr, {num('PAYMENT_NUM')} ten,
                    coalesce(TRY_CAST(TRY_CAST(FIRST_DUE AS DOUBLE) AS INT),0) d1, coalesce(TRY_CAST(TRY_CAST(SECOND_DUE AS DOUBLE) AS INT),0) d2,
                    coalesce(TRY_CAST(TRY_CAST(THIRD_DUE AS DOUBLE) AS INT),0) d3, coalesce(TRY_CAST(TRY_CAST(FOURTH_DUE AS DOUBLE) AS INT),0) d4
             FROM r_CONTRACT),
  c2 AS (SELECT *, {num('INIT_PAY')} ip, coalesce(NULLIF(PRODUCT_GROUP, ''), NULLIF(PRODUCT, '')) sp FROM c0),
  c AS (SELECT *, CASE
          WHEN sp = 'OTHER' OR PRODUCT = 'IN'                                  THEN 'Sản phẩm không xác định (IN)'
          WHEN sp = 'CC' AND cr IS NOT NULL AND cr < 1e6                       THEN 'Hạn mức thẻ < 1 triệu (gồm giá trị 1 đồng)'
          WHEN sp IN ('CD','TW') AND cr > 0 AND ip > 0 AND cr / (cr + ip) < 0.05 THEN 'Khoản vay < 5% giá trị món hàng'
          WHEN ann > 0 AND ten > 0 AND cr > 0 AND ann * ten < 0.99 * cr        THEN 'Tổng trả < gốc'
          WHEN cr IS NULL AND sp NOT IN ('BNPL') AND sp IS NOT NULL AND NAME_CREDIT_STATUS IN {SIGNED} THEN 'Thiếu số tiền ở HĐ đã ký'
        END AS dq_credit FROM c2)
  SELECT TRY_CAST(TRY_CAST(SKP_CREDIT_CASE AS DOUBLE) AS BIGINT) case_id, TRY_CAST(TRY_CAST(SKP_CLIENT AS DOUBLE) AS BIGINT) skp_client,
    coalesce(NULLIF(PRODUCT_GROUP, ''), NULLIF(PRODUCT, ''), 'UNKNOWN') product, NAME_CREDIT_STATUS status,
    {flag('IS_SIGNED')} is_signed, NAME_CREDIT_STATUS = 'Rejected' is_rejected, NAME_CREDIT_STATUS = 'Cancelled' is_cancelled,
    NAME_CREDIT_STATUS IN ('Written off', 'Sold') is_writeoff, {flag('IS_REVOLVING')} is_revolving,
    t_apply, t_approve, t_sign, coalesce(t_sign, CASE WHEN {flag('IS_SIGNED')} THEN t_approve END) t_sign_filled,
    {ts('TRANSAC_DATE')} t_transac,
    -- AMT_CREDIT không hợp lệ -> NULL (HĐ vẫn giữ lại cho các biến lịch sử trả nợ)
    CASE WHEN cr > 0 AND dq_credit IS NULL THEN cr END amt_credit, dq_credit AS dq_amt_credit,
    -- đáng ngờ: giữ giá trị, chỉ gắn cờ
    CASE WHEN ann * ten > 3 * cr AND cr > 0 THEN 'Tổng trả ≥ 3 lần gốc'
         WHEN cr / NULLIF(inc, 0) >= 50 THEN 'Khoản vay ≥ 50 lần thu nhập' END AS dq_amt_credit_suspect,
    CASE WHEN ten > 0 THEN ten END tenor, {num('INIT_PAY')} init_pay,
    CASE WHEN ann > 0 THEN ann END annuity, {num('AMT_BILLING')} amt_billing,
    coalesce(TRY_CAST(TRY_CAST(FLAG_INS AS DOUBLE) AS INT), 0) flag_ins,
    d1, d2, d3, d4, d1 + d2 + d3 + d4 n_due_flags, d1 + d2 + d3 + d4 > 0 any_due,
    CASE WHEN inc > 0 THEN inc END inc_main, {nx('NAME_EDUCATION_TYPE')} education,
    {nx('NAME_INCOME_TYPE')} income_type, {nx('CODE_PROFESSION')} profession, {num('CNT_CHILDREN')} cnt_children,
    CASE WHEN ann > 0 AND inc > 0 THEN ann / inc END annuity_income_ratio,
    {flag('FLAG_TIME_INCONSISTENT')} dq_time_inconsistent
  FROM c) TO '{(OUT / 'contract_clean.parquet').as_posix()}' (FORMAT parquet)""")

con.execute(f"""COPY (
  SELECT TRY_CAST(TRY_CAST(SKP_CLIENT AS DOUBLE) AS BIGINT) skp_client, {ts('DTIME_CREATED')} t_created,
    CASE WHEN year({ts('DATE_ACCEPTED')}) < 2900 THEN {ts('DATE_ACCEPTED')} END t_accepted,
    coalesce(year({ts('DATE_ACCEPTED')}) < 2900, false) is_accepted, {flag('FLAG_ASSIGNED')} is_assigned,
    DESC_FIN_REASON reason, {flag('IS_DEAL')} is_deal, CODE_PRODUCT_TYPE product_code,
    coalesce(NULLIF(PRODUCT_GROUP, ''), 'UNKNOWN') product, LEAD_SOURCE lead_source,
    LEAD_SOURCE IN ('SOB', 'GMA', 'LANDING PAGE') is_self_initiated,
    CASE WHEN {num('MAX_OFFER')} >= 100000 THEN {num('MAX_OFFER')} END max_offer, {num('MAX_OFFER')} < 100000 dq_offer_suspect
  FROM r_LEADS) TO '{(OUT / 'leads_clean.parquet').as_posix()}' (FORMAT parquet)""")

con.execute(f"""COPY (
  SELECT TRY_CAST(TRY_CAST(SKP_CLIENT AS DOUBLE) AS BIGINT) skp_client, {ts('DTIME_EVENT')} t_event, ACTION action_name,
    CHANNEL channel, EVENT_TYPE = 'SCORING' is_scoring, ACTION IN ('APP SCORING', 'WEB SCORING') is_self_service,
    PRODUCT_CODE product_code, coalesce(NULLIF(PRODUCT_GROUP, ''), 'UNKNOWN') product
  FROM r_PAST_BEHAVIOR) TO '{(OUT / 'past_clean.parquet').as_posix()}' (FORMAT parquet)""")

con.execute(f"""COPY (
  SELECT TRY_CAST(TRY_CAST(SKP_CLIENT AS DOUBLE) AS BIGINT) skp_client, CALL_ID call_id, {ts('TIME_OF_CALL_START')} t_start,
    {ts('TIME_OF_CALL_END')} t_end, {num('DURATIONS')} duration, CALL_TYPE call_type, CALL_TYPE = 'Inbound' is_inbound,
    CALL_SOURCE call_source, OFF_BY off_by, OFFER_TYPE offer_type, {flag('IS_CONNECTED')} is_connected,
    NULLIF(NULLIF(RESPONSE_RESULT_2, 'none'), '') response, {flag('IS_COLLECTION_CALL')} is_own_code,
    CALL_RESULT_SYSTEM result_sys, PRODUCT_TYPE product_code, coalesce(NULLIF(PRODUCT_GROUP, ''), 'UNKNOWN') product,
    NOT {flag('IS_POST_SNAPSHOT')} in_data_window
  FROM r_CALL) TO '{(OUT / 'call_clean.parquet').as_posix()}' (FORMAT parquet)""")

for t in ["customer", "contract", "leads", "past", "call"]:
    n = con.execute(f"SELECT count(*) FROM '{(OUT / f'{t}_clean.parquet').as_posix()}'").fetchone()[0]
    print(f"{t}_clean.parquet: {n:,} dòng", flush=True)
print(con.execute(f"""SELECT coalesce(dq_amt_credit, '(hợp lệ)') ly_do, count(*) n FROM '{(OUT / 'contract_clean.parquet').as_posix()}'
                      GROUP BY 1 ORDER BY n DESC""").df().to_string(index=False))
print(con.execute(f"""SELECT dq_amt_credit_suspect, count(*) n FROM '{(OUT / 'contract_clean.parquet').as_posix()}'
                      WHERE dq_amt_credit_suspect IS NOT NULL GROUP BY 1""").df().to_string(index=False))
