"""Financial Health Score – pipeline 10 bước (Gradient Boosting LightGBM + giải thích SHAP).

Câu hỏi của điểm: khách có quản lý được tài chính hiện tại và chống chịu được biến cố không?

 1. config               cấu hình: ngày chấm, cửa sổ quan sát, nhãn, biến, tham số, ngưỡng hạng, phiên bản
 2. load_and_check       đọc bảng clean; kiểm tra khóa, trùng, kiểu, thiếu, quan hệ giữa các bảng
 3. build_snapshot       chỉ giữ dữ liệu có trước ngày chấm; mỗi khách một dòng tại một thời điểm
 4. feature_engineering  tính chỉ tiêu theo 7 nhóm (thu nhập, thu–chi, nợ, lịch sử, dự phòng, áp lực vay, bảo vệ)
 5. validate_features    mẫu số 0 / vô cực, giá trị bất hợp lý, biến trùng thông tin, độ đầy đủ lịch sử
 6. score_features       điểm từng chỉ tiêu 0–100 (từ SHAP), quy tắc xử lý thiếu dữ liệu
 7. aggregate_scores     điểm từng nhóm 0–100 + Financial Health Score; khách không đủ dữ liệu
 8. explain_scores       3 yếu tố kéo điểm xuống, 3 điểm mạnh, mức độ đầy đủ dữ liệu
 9. validate_scorecard   kiểm tra logic, ổn định, độ nhạy ngưỡng, bỏ từng nhóm biến, kiểm tra dữ liệu độc lập (CALL)
10. export_and_monitor   xuất điểm có phiên bản; theo dõi phân bố điểm, thiếu dữ liệu, trôi dạt biến

Chạy:  python -u solution/health_pipeline.py            (dùng lại biến đã tính nếu có)
       python -u solution/health_pipeline.py --rebuild  (tính lại biến)
Kết quả: solution/out_team/pipeline/<phiên bản>/
"""
import datetime
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("BI10_CLEAN", str(HERE.parent / "eda" / "clean_team"))

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import features
from features import connect, health_features_full
from scorecard import ks, psi

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)


# =====================================================================
# 1. CONFIG
# =====================================================================
def config():
    # biến: (nhóm, mô tả, ràng buộc chiều: +1 càng lớn càng rủi ro, -1 càng lớn càng an toàn, 0 để mô hình tự học)
    V = {
        "income": ("1. Thu nhập & ổn định", "Thu nhập chính khai báo", 0),
        "inc_cv": ("1. Thu nhập & ổn định", "Độ biến động thu nhập khai báo (3 năm)", 0),
        "n_income_values": ("1. Thu nhập & ổn định", "Số mức thu nhập khác nhau đã khai", 0),
        "inc_growth": ("1. Thu nhập & ổn định", "Thu nhập mới nhất / cũ nhất (3 năm)", 0),
        "inc_vs_peak": ("1. Thu nhập & ổn định", "Thu nhập mới nhất / cao nhất", 0),
        "n_professions": ("1. Thu nhập & ổn định", "Số nghề nghiệp đã khai", 0),
        "n_income_types": ("1. Thu nhập & ổn định", "Số loại nguồn thu nhập đã khai", 0),
        "income_years_declared": ("1. Thu nhập & ổn định", "Số năm có khai báo thu nhập", 0),
        "residual_income": ("2. Cân đối thu–chi", "Thu nhập còn lại sau tiền góp hàng tháng", 0),
        "n_active": ("3. Gánh nặng nợ", "Số HĐ đang hoạt động", 0),
        "n_active_revolving": ("3. Gánh nặng nợ", "Số HĐ quay vòng (thẻ/hạn mức) đang mở", 0),
        "n_active_products2": ("3. Gánh nặng nợ", "Số loại sản phẩm đang vay", 0),
        "active_annuity_income": ("3. Gánh nặng nợ", "Tiền góp hàng tháng / thu nhập", 0),
        "active_credit_income": ("3. Gánh nặng nợ", "Tổng khoản vay đang chạy / thu nhập năm", 0),
        "overdue_recency_code": ("4. Lịch sử thanh toán", "Lần trễ hạn gần nhất", 0),
        "n_due_total": ("4. Lịch sử thanh toán", "Tổng số kỳ trễ hạn", 1),
        "on_time_rate": ("4. Lịch sử thanh toán", "Tỷ lệ kỳ trả đúng hạn", -1),
        "n_on_time": ("4. Lịch sử thanh toán", "Số kỳ trả đúng hạn", 0),
        "max_late_streak": ("4. Lịch sử thanh toán", "Chuỗi kỳ trễ liên tiếp dài nhất", 1),
        "n_inst_late_12m": ("4. Lịch sử thanh toán", "Số kỳ trễ trong 12 tháng", 1),
        "n_inst_late_6m": ("4. Lịch sử thanh toán", "Số kỳ trễ trong 6 tháng", 1),
        "n_clean_finished": ("4. Lịch sử thanh toán", "Số HĐ tất toán không trễ kỳ nào", 0),
        "n_fpd": ("4. Lịch sử thanh toán", "Số HĐ trễ ngay kỳ đầu", 1),
        "n_consec2_late": ("4. Lịch sử thanh toán", "Số HĐ trễ 2 kỳ liên tiếp", 1),
        "late4_rate": ("4. Lịch sử thanh toán", "Tỷ lệ HĐ trễ ở kỳ thứ 4", 1),
        "fpd_rate": ("4. Lịch sử thanh toán", "Tỷ lệ HĐ trễ kỳ đầu", 1),
        "late_rate_last3": ("4. Lịch sử thanh toán", "Tỷ lệ trễ trong 3 HĐ gần nhất", 1),
        "ever_writeoff": ("4. Lịch sử thanh toán", "Từng bị xóa nợ", 1),
        "n_multi_due": ("4. Lịch sử thanh toán", "Số HĐ trễ từ 2 kỳ trở lên", 1),
        "days_since_overdue": ("4. Lịch sử thanh toán", "Số ngày kể từ HĐ trễ gần nhất", -1),
        "down_payment_ratio": ("5. Dự phòng tài chính", "Tỷ lệ trả trước (CD/TW)", 0),
        "init_pay_income": ("5. Dự phòng tài chính", "Tiền trả trước / thu nhập", 0),
        "avg_init_pay": ("5. Dự phòng tài chính", "Tiền trả trước trung bình", 0),
        "max_init_pay_24m": ("5. Dự phòng tài chính", "Tiền trả trước lớn nhất 24 tháng", 0),
        "pct_with_downpay": ("5. Dự phòng tài chính", "Tỷ lệ HĐ có trả trước", 0),
        "max_offer": ("6. Áp lực vay & độ tin cậy", "Hạn mức công ty sẵn sàng cấp (gần nhất)", -1),
        "max_offer_12m": ("6. Áp lực vay & độ tin cậy", "Hạn mức đề xuất cao nhất 12 tháng", 0),
        "offer_change_6m": ("6. Áp lực vay & độ tin cậy", "Hạn mức hiện tại / 6 tháng trước", 0),
        "n_offers_12m": ("6. Áp lực vay & độ tin cậy", "Số lần được đề xuất hạn mức 12 tháng", 0),
        "refused_ratio": ("6. Áp lực vay & độ tin cậy", "Tỷ lệ hồ sơ bị từ chối", 1),
        "n_refused_12m": ("6. Áp lực vay & độ tin cậy", "Số lần bị từ chối 12 tháng", 0),
        "n_refused_3m": ("6. Áp lực vay & độ tin cậy", "Số lần bị từ chối 3 tháng", 0),
        "last_app_rejected": ("6. Áp lực vay & độ tin cậy", "Hồ sơ gần nhất bị từ chối", 1),
        "n_apply_1m": ("6. Áp lực vay & độ tin cậy", "Số lần xin vay 1 tháng", 0),
        "n_apply_3m": ("6. Áp lực vay & độ tin cậy", "Số lần xin vay 3 tháng", 0),
        "n_apply_6m": ("6. Áp lực vay & độ tin cậy", "Số lần xin vay 6 tháng", 0),
        "n_products_applied_6m": ("6. Áp lực vay & độ tin cậy", "Số loại sản phẩm xin vay 6 tháng", 0),
        "n_cancelled_12m": ("6. Áp lực vay & độ tin cậy", "Số hồ sơ tự hủy 12 tháng", 0),
        "days_since_last_apply": ("6. Áp lực vay & độ tin cậy", "Số ngày từ lần xin vay gần nhất", 0),
        "n_signed": ("6. Áp lực vay & độ tin cậy", "Số HĐ đã ký", 0),
        "n_signed_12m": ("6. Áp lực vay & độ tin cậy", "Số HĐ ký trong 12 tháng", 0),
        "credit_growth": ("6. Áp lực vay & độ tin cậy", "Khoản vay 12 tháng gần / 12 tháng trước", 0),
        "days_since_last_signed": ("6. Áp lực vay & độ tin cậy", "Số ngày từ HĐ ký gần nhất", 0),
        "history_years": ("6. Áp lực vay & độ tin cậy", "Số năm có lịch sử vay", 0),
        "pct_insured2": ("7. Kế hoạch & bảo vệ", "Tỷ lệ HĐ có mua bảo hiểm khoản vay", 0),
        "last_insured": ("7. Kế hoạch & bảo vệ", "HĐ gần nhất có mua bảo hiểm", 0),
    }
    cfg = dict(
        version="FH-GB-v1.1",
        score_date=features.DATA_END,                 # ngày chấm điểm (ngày sau bản ghi cuối của CONTRACT/LEADS/PAST)
        dev_cutoffs=["2024-02-01", "2024-04-01", "2024-06-01", "2024-08-01", "2024-10-01"],
        valid_cutoff="2024-10-01",                    # mốc dùng để chọn số cây và biến (không dùng tập kiểm chứng)
        oot_cutoff="2024-12-01",                      # kiểm chứng ngoài thời gian
        mature_days=features.MATURE_DAYS,             # HĐ phải ký trước mốc ≥ 125 ngày thì 4 kỳ đầu mới đến hạn
        label="Xấu = có HĐ trả góp ký trong 2 tháng sau mốc bị trễ ở 4 kỳ đầu hoặc bị xóa nợ",
        windows_days=dict(short=90, medium=180, year=365, income=1095),
        variables=V,
        min_shap_share=0.005,                         # bỏ biến đóng góp < 0.5% tổng |SHAP|
        lgb_params=dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=300,
                        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                        verbose=-1, seed=0, num_threads=8, monotone_constraints_method="advanced"),
        pdo=15, base_score=50,                        # 50 điểm = odds tốt/xấu trung bình tập huấn luyện; +15 điểm = odds gấp đôi
        bands=[(0, 40, "E"), (40, 50, "D"), (50, 58, "C"), (58, 65, "B"), (65, 101, "A")],
        band_name={"A": "A · Rất khỏe", "B": "B · Khỏe", "C": "C · Trung bình", "D": "D · Yếu",
                   "E": "E · Rủi ro cao", "N": "N · Chưa đủ lịch sử trả nợ"},
        band_action={"A": "Ưu tiên tăng hạn mức, mời sản phẩm giá trị cao (CL, thẻ), duyệt nhanh",
                     "B": "Duy trì, bán thêm có chọn lọc với hạn mức hiện tại",
                     "C": "Cho vay có kiểm soát: sản phẩm có trả trước (CD, TW), hạn mức vừa phải",
                     "D": "Hạn chế cấp mới, chỉ khoản nhỏ có trả trước; theo dõi sát",
                     "E": "Không cấp tín dụng mới; ưu tiên nhắc nợ, tư vấn tài chính",
                     "N": "Hạn mức nhỏ (BNPL/CD), tăng dần hạn mức sau mỗi kỳ trả đúng hạn; chưa xếp hạng khỏe khi chưa có bằng chứng trả nợ"},
        explain_min_shap=0.01,                        # yếu tố chỉ được nêu khi |SHAP| > 0.01 log-odds
    )
    cfg["out"] = HERE / "out_team" / "pipeline" / cfg["version"]
    (cfg["out"] / "fig").mkdir(parents=True, exist_ok=True)
    (cfg["out"] / "cache").mkdir(exist_ok=True)
    return cfg


# =====================================================================
# 2. LOAD AND CHECK
# =====================================================================
KEYS = {"customer": ["skp_client"], "contract": ["case_id"], "leads": None, "past": None, "call": None}
TIME_COL = {"contract": "t_apply", "leads": "t_created", "past": "t_event", "call": "t_start"}


def load_and_check(cfg):
    con = connect()
    rows, fatal = [], []
    for t, key in KEYS.items():
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        cols = con.execute(f"DESCRIBE {t}").df()
        r = dict(bang=t, so_dong=n, so_cot=len(cols))
        if key:
            k = ", ".join(key)
            r["trung_khoa"] = n - con.execute(f"SELECT count(*) FROM (SELECT DISTINCT {k} FROM {t})").fetchone()[0]
            r["khoa_null"] = con.execute(f"SELECT count(*) FROM {t} WHERE {' OR '.join(c + ' IS NULL' for c in key)}").fetchone()[0]
            if r["trung_khoa"] or r["khoa_null"]:
                fatal.append(f"{t}: khóa {k} trùng {r['trung_khoa']} / null {r['khoa_null']}")
        r["dong_trung_hoan_toan"] = n - con.execute(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {t})").fetchone()[0]
        if t != "customer":
            r["khach_khong_co_trong_CUSTOMER"] = con.execute(
                f"SELECT count(DISTINCT skp_client) FROM {t} WHERE skp_client NOT IN (SELECT skp_client FROM customer)").fetchone()[0]
        if t in TIME_COL:
            lo, hi, nn = con.execute(f"SELECT min({TIME_COL[t]}), max({TIME_COL[t]}), count(*) FILTER (WHERE {TIME_COL[t]} IS NULL) FROM {t}").fetchone()
            r.update(tu_ngay=str(lo), den_ngay=str(hi), ngay_null=nn)
        rows.append(r)
    checks = pd.DataFrame(rows)
    # kiểu dữ liệu và tỷ lệ thiếu của các cột mà biến Health dùng
    need = {"contract": ["skp_client", "product", "status", "is_signed", "is_rejected", "t_apply", "t_sign_filled", "amt_credit",
                         "tenor", "init_pay", "annuity", "amt_billing", "flag_ins", "d1", "d2", "d3", "d4", "inc_main"],
            "leads": ["skp_client", "t_created", "max_offer"], "customer": ["skp_client"]}
    miss = []
    for t, cols in need.items():
        schema = dict(con.execute(f"DESCRIBE {t}").df()[["column_name", "column_type"]].values)
        for c in cols:
            if c not in schema:
                fatal.append(f"{t}.{c} không tồn tại"); continue
            pct = con.execute(f"SELECT avg(({c} IS NULL)::INT) FROM {t}").fetchone()[0]
            miss.append(dict(bang=t, cot=c, kieu=schema[c], ty_le_thieu=pct))
    miss = pd.DataFrame(miss)
    # logic thời gian và giá trị
    logic = con.execute("""SELECT
        count(*) FILTER (WHERE is_signed AND t_sign_filled < t_apply)      AS ky_truoc_ngay_nop,
        count(*) FILTER (WHERE amt_credit < 0 OR annuity < 0 OR init_pay < 0) AS so_tien_am,
        count(*) FILTER (WHERE d1 NOT IN (0,1) OR d2 NOT IN (0,1) OR d3 NOT IN (0,1) OR d4 NOT IN (0,1)) AS co_due_khong_hop_le,
        count(*) FILTER (WHERE is_signed AND is_rejected)                  AS vua_ky_vua_tu_choi
      FROM contract""").df().T.rename(columns={0: "so_dong"})
    checks.to_csv(cfg["out"] / "02_table_checks.csv", index=False, encoding="utf-8-sig")
    miss.to_csv(cfg["out"] / "02_column_missing.csv", index=False, encoding="utf-8-sig")
    logic.to_csv(cfg["out"] / "02_logic_checks.csv", encoding="utf-8-sig")
    log("Bước 2 – kiểm tra bảng:\n" + checks.to_string(index=False) + "\n" + logic.to_string())
    if fatal:
        raise ValueError("Dữ liệu không đạt yêu cầu đầu vào: " + "; ".join(fatal))
    return con


# =====================================================================
# 3. BUILD SNAPSHOT
# =====================================================================
def build_snapshot(con, cutoff):
    """Tạo view chỉ chứa dữ liệu trước `cutoff` (contract_s, leads_s, past_s) và danh sách khách với mức độ có dữ liệu.
    Các hàm tạo biến cũng lọc theo cutoff; bước này kiểm tra lại rằng không có bản ghi nào sau mốc lọt vào."""
    c = f"TIMESTAMP '{cutoff}'"
    con.execute(f"CREATE OR REPLACE VIEW contract_s AS SELECT * FROM contract WHERE t_apply < {c}")
    con.execute(f"CREATE OR REPLACE VIEW leads_s AS SELECT * FROM leads WHERE t_created < {c}")
    snap = con.execute(f"""
      SELECT cu.skp_client,
        coalesce(k.n_apps, 0) AS snap_n_apps, coalesce(k.n_signed_any, 0) AS snap_n_signed,
        coalesce(k.n_mature, 0) AS snap_n_mature, coalesce(k.n_hist, 0) AS snap_n_hist, coalesce(l.n_offer, 0) AS snap_n_offer, k.last_apply
      FROM customer cu
      LEFT JOIN (SELECT skp_client, count(*) n_apps, count(*) FILTER (WHERE is_signed AND t_sign_filled < {c}) n_signed_any,
                        count(*) FILTER (WHERE is_signed AND product IN {features.REPAY_OBSERVED}
                                           AND t_sign_filled < {c} - INTERVAL {features.MATURE_DAYS} DAY) n_mature,
                        -- bằng chứng trả nợ: khoản vay bất kỳ (trả góp, BNPL, thẻ) có kỳ trả thứ 3 đến hạn trước mốc ≥ 15 ngày
                        count(*) FILTER (WHERE is_signed AND product IN {features.REPAY_OBSERVED}
                                           AND t_sign_filled + to_months({features.HISTORY_MIN_INSTALLMENTS}) < {c} - INTERVAL 15 DAY) n_hist,
                        max(t_apply) last_apply
                 FROM contract_s GROUP BY 1) k USING (skp_client)
      LEFT JOIN (SELECT skp_client, count(*) n_offer FROM leads_s WHERE max_offer IS NOT NULL GROUP BY 1) l USING (skp_client)
    """).df()
    leak = snap.last_apply.dropna().max()
    assert pd.isna(leak) or leak < pd.Timestamp(cutoff), f"Dữ liệu sau mốc {cutoff} lọt vào snapshot"
    # mức độ đầy đủ dữ liệu
    snap["data_completeness"] = np.select(
        [snap.snap_n_hist >= 2, snap.snap_n_hist == 1],
        ["Đầy đủ (≥ 2 HĐ đã qua ≥ 3 kỳ trả)", "Một phần (1 HĐ đã qua ≥ 3 kỳ trả)"], "Chưa đủ lịch sử trả nợ (chưa có HĐ nào qua ≥ 3 kỳ trả)")
    return snap.drop(columns="last_apply")


# =====================================================================
# 4. FEATURE ENGINEERING
# =====================================================================
def feature_engineering(con, cfg, cutoff, with_label=True, rebuild=False):
    path = cfg["out"] / "cache" / f"features_{cutoff}{'' if with_label else '_nolabel'}.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    snap = build_snapshot(con, cutoff)
    d = health_features_full(con, cutoff, with_label=with_label)
    # '1. Chưa có lịch sử' → 0, '2. Không trễ' → 1, '3. > 3 năm' → 2, '4. 1–3 năm' → 3, '5. < 1 năm' → 4
    d["overdue_recency_code"] = d.overdue_recency.astype(str).str[0].astype(int) - 1
    d = d.merge(snap, on="skp_client", how="left")
    d.to_parquet(path)
    return d


# =====================================================================
# 5. VALIDATE FEATURES
# =====================================================================
RATE_VARS = ["on_time_rate", "refused_ratio", "late4_rate", "fpd_rate", "late_rate_last3", "pct_with_downpay",
             "pct_insured2", "down_payment_ratio", "inc_vs_peak"]
NONNEG = ["income", "n_due_total", "n_on_time", "max_late_streak", "n_inst_late_12m", "n_inst_late_6m", "n_clean_finished",
          "n_active", "max_offer", "n_signed", "n_apply_6m", "avg_init_pay", "history_years", "days_since_overdue"]


def validate_features(df, cfg, name, report=True):
    V = list(cfg["variables"])
    d = df.copy()
    rows = []
    for v in V:
        x = pd.to_numeric(d[v], errors="coerce").astype(float)
        n_inf = int(np.isinf(x).sum())
        bad = pd.Series(False, index=x.index)
        if v in RATE_VARS:
            bad |= (x < 0) | (x > 1.0001)
        if v in NONNEG:
            bad |= x < 0
        if v == "income":
            bad |= x == 0
        x = x.where(~np.isinf(x) & ~bad)          # vô cực (mẫu số 0) và giá trị bất hợp lý → thiếu
        d[v] = x
        rows.append(dict(bien=v, nhom=cfg["variables"][v][0], co_du_lieu=x.notna().mean(), vo_cuc=n_inf,
                         bat_hop_ly=int(bad.sum()), min=x.min(), p50=x.median(), p99=x.quantile(.99), max=x.max()))
    q = pd.DataFrame(rows)
    # biến trùng thông tin: báo cáo cặp |Spearman| > 0.95; chỉ loại khi cả Spearman và Pearson > 0.99.
    # (biến đếm trễ hạn có ~97% giá trị 0 nên Spearman gần 1 dù thông tin khác nhau)
    corr = d[V].rank().corr().abs()
    pear = d[V].corr().abs()
    dup = [(a, b, corr.loc[a, b], pear.loc[a, b]) for i, a in enumerate(V) for b in V[i + 1:] if corr.loc[a, b] > 0.95]
    dup = pd.DataFrame(dup, columns=["bien_1", "bien_2", "spearman", "pearson"])
    dup["loai_bien_2"] = (dup.spearman > .99) & (dup.pearson > .99)
    if report:
        q.to_csv(cfg["out"] / f"05_feature_quality_{name}.csv", index=False, encoding="utf-8-sig")
        dup.to_csv(cfg["out"] / "05_redundant_pairs.csv", index=False, encoding="utf-8-sig")
        log(f"Bước 5 – {name}: {int(q.vo_cuc.sum())} giá trị vô cực, {int(q.bat_hop_ly.sum())} giá trị bất hợp lý (đã đặt = thiếu); "
            f"{len(dup)} cặp biến trùng thông tin (|ρ| > 0.95)")
        if len(dup):
            log("\n" + dup.round(3).to_string(index=False))
        log("Độ đầy đủ lịch sử:\n" + d.data_completeness.value_counts(normalize=True).round(3).to_string())
    return d, dup


# =====================================================================
# Mô hình (dùng trong bước 6–7)
# =====================================================================
def _train(cfg, feats, d_tr, d_va=None, rounds=None, mono=True):
    p = dict(cfg["lgb_params"], monotone_constraints=[cfg["variables"][f][2] if mono else 0 for f in feats])
    ds = lgb.Dataset(d_tr[feats].astype(float), d_tr.bad)
    if d_va is not None:
        dv = lgb.Dataset(d_va[feats].astype(float), d_va.bad, reference=ds)
        return lgb.train(p, ds, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(p, ds, rounds)


def fit_model(cfg, dev, dup):
    tr, va = dev[dev.cutoff != cfg["valid_cutoff"]], dev[dev.cutoff == cfg["valid_cutoff"]]
    # với mỗi cặp trùng thông tin, giữ biến xuất hiện trước trong cấu hình
    V = [v for v in cfg["variables"] if v not in set(dup.bien_2[dup.loai_bien_2])]
    m0 = _train(cfg, V, tr, va)
    c = m0.predict(va[V].astype(float), pred_contrib=True)[:, :-1]
    share = pd.Series(np.abs(c).mean(0), index=V); share /= share.sum()
    feats = [f for f in share.sort_values(ascending=False).index if share[f] >= cfg["min_shap_share"]]
    m1 = _train(cfg, feats, tr, va)
    rounds = int(m1.best_iteration * 1.1)
    auc_va = roc_auc_score(va.bad, m1.predict(va[feats].astype(float)))
    log(f"Chọn biến: {len(V)} → {len(feats)} biến (SHAP ≥ {cfg['min_shap_share']:.1%}); AUC mốc {cfg['valid_cutoff']} {auc_va:.4f}; {rounds} cây")
    model = _train(cfg, feats, dev, rounds=rounds)
    ref = np.log((1 - dev.bad.mean()) / dev.bad.mean())
    return dict(model=model, feats=feats, rounds=rounds, ref=ref, factor=cfg["pdo"] / np.log(2), auc_valid=auc_va)


def shap_matrix(M, d, chunk=200_000):
    X = d[M["feats"]].astype(float)
    out = [M["model"].predict(X.iloc[s:s + chunk], pred_contrib=True) for s in range(0, len(X), chunk)]
    c = np.vstack(out)
    return pd.DataFrame(c[:, :-1], columns=M["feats"], index=d.index), c[:, -1]


# =====================================================================
# 6. SCORE FEATURES
# =====================================================================
def score_features(M, cfg, d, ref_range=None):
    """Điểm chỉ tiêu 0–100 = vị trí đóng góp SHAP của chỉ tiêu đó giữa mức tệ nhất (p1) và tốt nhất (p99) trên tập huấn luyện.
    100 = chỉ tiêu này đang giúp khách nhiều nhất, 0 = đang kéo điểm xuống nhiều nhất.
    Thiếu dữ liệu: LightGBM học một nhánh riêng cho giá trị thiếu, nên khách thiếu chỉ tiêu nhận đúng mức rủi ro quan sát được
    của nhóm khách thiếu; điểm chỉ tiêu vẫn được tính và cột `miss_<biến>` đánh dấu để báo cáo."""
    S, base = shap_matrix(M, d)
    if ref_range is None:
        ref_range = {f: (S[f].quantile(.01), S[f].quantile(.99)) for f in M["feats"]}
    pts = pd.DataFrame(index=d.index)
    for f in M["feats"]:
        lo, hi = ref_range[f]
        pts[f] = (100 * (hi - S[f]) / (hi - lo)).clip(0, 100) if hi > lo else 50.0
    miss = d[M["feats"]].isna().add_prefix("miss_")
    return S, base, pts, miss, ref_range


# =====================================================================
# 7. AGGREGATE SCORES
# =====================================================================
def band_of(cfg, score, completeness):
    b = pd.cut(score, [x[0] for x in cfg["bands"]] + [101], labels=[x[2] for x in cfg["bands"]], right=False).astype(str)
    thin = ~completeness.str.startswith(("Đầy đủ", "Một phần"))   # chưa có khoản vay nào qua ≥ 3 kỳ trả
    return b.where(~thin, "N")


def aggregate_scores(M, cfg, d, S, base, group_range=None):
    """Financial Health Score = base_score + PDO/ln2 × (log-odds tốt − log-odds tốt trung bình huấn luyện).
    Vì SHAP cộng tính trên log-odds, điểm tổng = điểm gốc − Σ PDO/ln2 × SHAP_biến → mỗi nhóm đóng góp một số điểm cộng/trừ rõ ràng.
    Điểm nhóm 0–100: vị trí tổng SHAP của nhóm giữa p1 và p99 trên tập huấn luyện.
    Khách chưa có khoản vay nào qua ≥ 3 kỳ trả: vẫn có điểm nhưng xếp hạng N (chưa đủ lịch sử trả nợ)."""
    G = pd.Series({f: cfg["variables"][f][0] for f in M["feats"]})
    logit_bad = base + S.sum(1).values
    score = np.clip(cfg["base_score"] + M["factor"] * (-logit_bad - M["ref"]), 0, 100)
    gs = pd.DataFrame({g: S[list(fs)].sum(1) for g, fs in G.groupby(G).groups.items()})   # tổng SHAP theo nhóm
    if group_range is None:
        group_range = {g: (gs[g].quantile(.01), gs[g].quantile(.99)) for g in gs}
    out = pd.DataFrame({"skp_client": d.skp_client.values, "financial_health_score": score.round(1)}, index=d.index)
    for g in sorted(gs):
        lo, hi = group_range[g]
        out[f"diem_nhom_{g[:1]}"] = (100 * (hi - gs[g]) / (hi - lo)).clip(0, 100).round(1)
        out[f"diem_cong_tru_nhom_{g[:1]}"] = (-M["factor"] * gs[g]).round(2)
    out["xac_suat_xau"] = 1 / (1 + np.exp(-logit_bad))
    out["health_band"] = band_of(cfg, out.financial_health_score, d.data_completeness)
    out["band_name"] = out.health_band.map(cfg["band_name"])
    out["data_completeness"] = d.data_completeness.values
    return out, group_range


# =====================================================================
# 8. EXPLAIN SCORES
# =====================================================================
def explain_scores(M, cfg, d, S, miss):
    desc = np.array([cfg["variables"][f][1] for f in M["feats"]])
    c = S.values
    order = np.argsort(-c, axis=1)
    rr = np.arange(len(c))
    ex = pd.DataFrame(index=d.index)
    thr = cfg["explain_min_shap"]
    for k in range(3):
        r, g = order[:, k], order[:, -1 - k]
        ex[f"ly_do_tru_diem_{k + 1}"] = pd.Categorical(np.where(c[rr, r] > thr, desc[r], ""))
        ex[f"diem_tru_{k + 1}"] = np.where(c[rr, r] > thr, -M["factor"] * c[rr, r], 0).round(1)
        ex[f"diem_manh_{k + 1}"] = pd.Categorical(np.where(c[rr, g] < -thr, desc[g], ""))
        ex[f"diem_cong_{k + 1}"] = np.where(c[rr, g] < -thr, -M["factor"] * c[rr, g], 0).round(1)
    ex["ty_le_chi_tieu_co_du_lieu"] = (1 - miss.mean(1)).round(2)
    return ex


# =====================================================================
# 9. VALIDATE SCORECARD
# =====================================================================
def capture(y, score, top=.1):
    k = int(len(y) * top)
    return np.asarray(y)[np.argsort(np.asarray(score))[:k]].sum() / np.asarray(y).sum()


def _perf(y, s):
    a = roc_auc_score(y, -s)
    return dict(AUC=a, Gini=2 * a - 1, KS=ks(y, s), Bat_xau_top10=capture(y, s))


def validate_scorecard(M, cfg, dev, oot, sc_dev, sc_oot, S_oot, con, final_scores, final):
    out = cfg["out"]
    rep = {}
    # 9a. hiệu quả: huấn luyện, từng mốc, kiểm chứng ngoài thời gian
    perf = [dict(Tap="Huấn luyện (5 mốc)", **_perf(dev.bad, sc_dev.financial_health_score))]
    for c in cfg["dev_cutoffs"]:
        m = dev.cutoff == c
        perf.append(dict(Tap=f"  mốc {c}", **_perf(dev.bad[m], sc_dev.financial_health_score[m])))
    perf.append(dict(Tap=f"Kiểm chứng {cfg['oot_cutoff']}", **_perf(oot.bad, sc_oot.financial_health_score)))
    perf = pd.DataFrame(perf)
    perf["PSI_so_voi_huan_luyen"] = np.nan
    perf.loc[perf.index[-1], "PSI_so_voi_huan_luyen"] = psi(sc_dev.financial_health_score.values, sc_oot.financial_health_score.values)
    perf.to_csv(out / "09a_performance.csv", index=False, encoding="utf-8-sig")
    rep["performance"] = perf
    # 9b. hạng: tỷ lệ xấu phải tăng dần A → E
    bt = sc_oot.assign(bad=oot.bad.values).groupby("health_band").agg(so_khach=("bad", "size"), ty_le_xau=("bad", "mean"),
                                                                         diem_tb=("financial_health_score", "mean"))
    bt["ty_trong"] = bt.so_khach / bt.so_khach.sum()
    order = [b for b in "ABCDE" if b in bt.index]
    mono = bool(np.all(np.diff(bt.loc[order, "ty_le_xau"].values) > 0))
    bt.to_csv(out / "09b_bands_oot.csv", encoding="utf-8-sig")
    rep["bands"], rep["band_monotonic"] = bt, mono
    # 9c. logic: chiều tác động của từng biến (tương quan giá trị – SHAP) so với kỳ vọng nghiệp vụ
    X = oot[M["feats"]].astype(float)
    lg = []
    for f in M["feats"]:
        ok = X[f].notna()
        r = np.corrcoef(X.loc[ok, f].rank(), S_oot.loc[ok, f])[0, 1] if ok.sum() > 100 else np.nan
        exp = cfg["variables"][f][2]
        lg.append(dict(bien=f, mo_ta=cfg["variables"][f][1], nhom=cfg["variables"][f][0], rang_buoc=exp,
                       tuong_quan_gia_tri_vs_rui_ro=r,
                       dung_chieu=("—" if exp == 0 else ("Đạt" if np.sign(r) == exp else "KHÔNG ĐẠT"))))
    lg = pd.DataFrame(lg)
    lg.to_csv(out / "09c_logic_direction.csv", index=False, encoding="utf-8-sig")
    rep["logic"] = lg
    # 9d. bỏ từng nhóm biến: huấn luyện lại không có nhóm đó, đo AUC kiểm chứng
    G = pd.Series({f: cfg["variables"][f][0] for f in M["feats"]})
    base_auc = roc_auc_score(oot.bad, -sc_oot.financial_health_score)
    drop = []
    for g in sorted(G.unique()):
        fs = [f for f in M["feats"] if G[f] != g]
        m = _train(cfg, fs, dev, rounds=M["rounds"])
        a = roc_auc_score(oot.bad, m.predict(oot[fs].astype(float)))
        drop.append(dict(bo_nhom=g, so_bien_bo=int((G == g).sum()), AUC_con_lai=a, AUC_giam=base_auc - a))
    drop = pd.DataFrame(drop).sort_values("AUC_giam", ascending=False)
    drop.to_csv(out / "09d_drop_group.csv", index=False, encoding="utf-8-sig")
    rep["drop_group"] = drop
    # 9e. độ nhạy: dời toàn bộ ngưỡng hạng ±2 điểm; đổi PDO 10/20 (không đổi thứ hạng, chỉ đổi độ giãn)
    sens = []
    for shift in (-2, 2):
        c2 = dict(cfg, bands=[(max(lo + shift, 0) if lo else 0, hi + shift if hi < 101 else 101, b) for lo, hi, b in cfg["bands"]])
        b2 = band_of(c2, sc_oot.financial_health_score, sc_oot.data_completeness)
        t = pd.Series(oot.bad.values).groupby(b2.values).mean()
        sens.append(dict(kich_ban=f"Ngưỡng hạng {shift:+d} điểm", ty_le_khach_doi_hang=(b2 != sc_oot.health_band).mean(),
                         ty_le_xau_A=t.get("A"), ty_le_xau_E=t.get("E"),
                         van_tang_dan=bool(np.all(np.diff([t.get(b, np.nan) for b in "ABCDE"]) > 0))))
    for pdo in (10, 20):
        s = np.clip(cfg["base_score"] + (pdo / cfg["pdo"]) * (sc_oot.financial_health_score - cfg["base_score"]), 0, 100)
        b2 = band_of(cfg, s, sc_oot.data_completeness)
        sens.append(dict(kich_ban=f"PDO = {pdo}", ty_le_khach_doi_hang=(b2 != sc_oot.health_band).mean(),
                         ty_le_xau_A=oot.bad[b2.values == "A"].mean(), ty_le_xau_E=oot.bad[b2.values == "E"].mean(),
                         van_tang_dan=bool(np.all(np.diff([oot.bad[b2.values == b].mean() for b in "ABCDE"]) > 0))))
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "09e_sensitivity.csv", index=False, encoding="utf-8-sig")
    rep["sensitivity"] = sens
    # 9f. dữ liệu độc lập: CALL (01/2025–08/2026) là nguồn duy nhất có thông tin SAU ngày chấm. Mã phản hồi OWN* trông như
    # nhắc nợ (PTP, Agreement, Voluntary payment). Trước khi dùng làm thước đo, kiểm tra nó có gắn với rủi ro thật không:
    # nếu là cuộc gọi nhắc nợ, khách từng trễ hạn phải bị gọi nhiều hơn rõ rệt.
    own = con.execute(f"""SELECT skp_client, 1 AS bi_goi_own,
                                 max((response IN ('OWNPTPConfirmed','OWNAgreement','OWNVoluntaryPayment',
                                                   'OWNVoluntaryPaymentBA','OWNVoluntaryPaymentST'))::INT) AS noi_dung_tra_no
                          FROM call WHERE is_own_code AND t_start >= TIMESTAMP '{cfg['score_date']}'
                            AND t_start < TIMESTAMP '{cfg['score_date']}' + INTERVAL 180 DAY GROUP BY 1""").df()
    offer = con.execute("SELECT offer_type, count(*) n FROM call WHERE is_own_code GROUP BY 1 ORDER BY 2 DESC").df()
    offer["ty_trong"] = offer.n / offer.n.sum()
    f = final_scores[["skp_client", "health_band", "financial_health_score"]].merge(
        final[["skp_client", "n_active", "n_inst_late_12m"]], on="skp_client").merge(own, on="skp_client", how="left")
    f = f[f.n_active.fillna(0) > 0].fillna({"bi_goi_own": 0, "noi_dung_tra_no": 0})    # chỉ khách đang có khoản vay
    f["tre_12_thang"] = np.select([f.n_inst_late_12m.isna(), f.n_inst_late_12m == 0, f.n_inst_late_12m == 1],
                                  ["Chưa có kỳ đến hạn", "0 kỳ trễ", "1 kỳ trễ"], "≥ 2 kỳ trễ")
    ind = f.groupby("tre_12_thang").agg(so_khach=("bi_goi_own", "size"), ty_le_bi_goi_OWN=("bi_goi_own", "mean"),
                                        ty_le_noi_dung_tra_no=("noi_dung_tra_no", "mean"))
    auc_late = roc_auc_score(f.bi_goi_own, f.n_inst_late_12m.fillna(0))
    a_ind = roc_auc_score(f.bi_goi_own, -f.financial_health_score)
    usable = auc_late > .55
    ind.to_csv(out / "09f_independent_call_check.csv", encoding="utf-8-sig")
    rep.update(independent=ind, independent_auc=a_ind, independent_auc_late=auc_late, independent_usable=usable,
               own_offer=offer.head(6))
    rep["base_auc"] = base_auc
    log("Bước 9 – hiệu quả:\n" + perf.round(4).to_string(index=False))
    log(f"Hạng (kiểm chứng), tỷ lệ xấu tăng dần A→E: {mono}\n" + bt.round(4).to_string())
    log("Logic chiều tác động – biến không đạt: " + str(lg[lg.dung_chieu == "KHÔNG ĐẠT"].bien.tolist()))
    log("Bỏ từng nhóm:\n" + drop.round(4).to_string(index=False))
    log("Độ nhạy:\n" + sens.round(4).to_string(index=False))
    log(f"Dữ liệu độc lập (CALL mã OWN*, khách đang vay): AUC theo lịch sử trễ {auc_late:.3f}, theo điểm {a_ind:.3f} → "
        f"{'dùng được' if usable else 'KHÔNG dùng được làm thước đo rủi ro'}\n" + ind.round(4).to_string())
    return rep


# =====================================================================
# 10. EXPORT AND MONITOR
# =====================================================================
def export_and_monitor(M, cfg, final, scores, expl, pts, dev, sc_dev, rep):
    out = cfg["out"]
    res = pd.concat([scores, expl], axis=1)
    res.insert(1, "score_date", cfg["score_date"]); res.insert(2, "score_version", cfg["version"])
    res["hanh_dong_goi_y"] = res.health_band.map(cfg["band_action"])
    res.to_csv(out / "10_customer_financial_health.csv", index=False, encoding="utf-8-sig")
    res.to_parquet(out / "10_customer_financial_health.parquet")
    pts.assign(skp_client=final.skp_client.values).to_parquet(out / "10_indicator_scores.parquet")
    M["model"].save_model(str(out / "model_lgb.txt"))
    meta = {k: v for k, v in cfg.items() if k not in ("out",)}
    meta.update(features_used=M["feats"], n_trees=M["rounds"], ref_logodds=M["ref"], created=datetime.datetime.now().isoformat(),
                data_source=os.environ["BI10_CLEAN"])
    (out / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # theo dõi: phân bố điểm, thiếu dữ liệu, trôi dạt từng biến so với tập huấn luyện
    mon = []
    for f in M["feats"]:
        a, b = dev[f].astype(float), final[f].astype(float)
        both = pd.concat([a, b]).dropna()
        mon.append(dict(bien=f, mo_ta=cfg["variables"][f][1], thieu_huan_luyen=a.isna().mean(), thieu_hien_tai=b.isna().mean(),
                        PSI=psi(a.fillna(both.min() - 1).values, b.fillna(both.min() - 1).values) if len(both) else np.nan))
    mon = pd.DataFrame(mon).sort_values("PSI", ascending=False)
    mon["canh_bao"] = np.select([mon.PSI > .25, mon.PSI > .1], ["Trôi dạt mạnh", "Cần theo dõi"], "Ổn định")
    mon.to_csv(out / "10_feature_drift.csv", index=False, encoding="utf-8-sig")
    dist = res.health_band.value_counts().reindex(list(cfg["band_name"])).rename("so_khach").to_frame()
    dist["ty_trong"] = dist.so_khach / len(res)
    dist.to_csv(out / "10_band_distribution.csv", encoding="utf-8-sig")
    score_psi = psi(sc_dev.financial_health_score.values, scores.financial_health_score.values)
    hist = pd.DataFrame([dict(score_date=cfg["score_date"], version=cfg["version"], so_khach=len(res),
                              diem_tb=res.financial_health_score.mean(), PSI_diem=score_psi,
                              **{f"ty_trong_{b}": dist.ty_trong.get(b, 0) for b in cfg["band_name"]},
                              so_bien_troi_dat=int((mon.PSI > .25).sum()), AUC_kiem_chung=rep["base_auc"])])
    hp = HERE / "out_team" / "pipeline" / "monitoring_history.csv"
    if hp.exists():
        old = pd.read_csv(hp)
        hist = pd.concat([old[~((old.score_date == cfg["score_date"]) & (old.version == cfg["version"]))], hist])
    hist.to_csv(hp, index=False, encoding="utf-8-sig")
    log(f"Bước 10 – đã xuất {len(res):,} khách; PSI điểm so với huấn luyện {score_psi:.3f}\n" + dist.round(3).to_string()
        + "\nBiến trôi dạt nhiều nhất:\n" + mon.head(8).round(3).to_string(index=False))
    return res, mon


def plots(M, cfg, oot, S_oot, sc_oot):
    fig_dir = cfg["out"] / "fig"
    desc = {f: cfg["variables"][f][1] for f in M["feats"]}
    imp = S_oot.abs().mean().sort_values(ascending=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    top = imp.index[:20][::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh([desc[f] for f in top], imp[top], color="#3b6fb6")
    ax.set_xlabel("Trung bình |SHAP| (log-odds xác suất xấu)"); ax.set_title("Biến ảnh hưởng nhiều nhất")
    fig.tight_layout(); fig.savefig(fig_dir / "shap_importance.png", dpi=160); plt.close(fig)
    G = pd.Series({f: cfg["variables"][f][0] for f in M["feats"]})
    gi = imp.groupby(G).sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(gi.index, gi / gi.sum() * 100, color="#5b8f3a"); ax.set_xlabel("% tổng |SHAP|")
    ax.set_title("Trọng số thực tế của từng nhóm chỉ tiêu"); fig.tight_layout(); fig.savefig(fig_dir / "shap_by_group.png", dpi=160); plt.close(fig)
    X = oot[M["feats"]].astype(float)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for i, f in enumerate(top):
        idx = rng.choice(len(X), min(4000, len(X)), replace=False)
        col = X[f].rank(pct=True).fillna(-1).values[idx]; s = S_oot[f].values[idx]; m = col < 0
        ax.scatter(s[~m], i + rng.normal(0, .12, (~m).sum()), c=col[~m], cmap="coolwarm", s=3, alpha=.6, vmin=0, vmax=1)
        ax.scatter(s[m], i + rng.normal(0, .12, m.sum()), c="#9a9a9a", s=3, alpha=.4)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([desc[f] for f in top]); ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("SHAP (> 0: tăng xác suất xấu → giảm điểm)")
    ax.set_title("Mỗi chấm là một khách · xanh = giá trị thấp, đỏ = cao, xám = thiếu")
    fig.tight_layout(); fig.savefig(fig_dir / "shap_beeswarm.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, f in zip(axes.ravel(), imp.index[:6]):
        ok = X[f].notna()
        ax.scatter(X.loc[ok, f].clip(upper=X.loc[ok, f].quantile(.99)), S_oot.loc[ok, f], s=2, alpha=.25, color="#3b6fb6")
        ax.axhline(0, color="k", lw=.6); ax.set_title(desc[f], fontsize=9); ax.set_ylabel("SHAP")
    fig.tight_layout(); fig.savefig(fig_dir / "shap_dependence.png", dpi=150); plt.close(fig)
    bt = sc_oot.assign(bad=oot.bad.values).groupby("health_band").bad.mean().reindex(list("ABCDEN"))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(bt.index, bt * 100, color=["#2e7d32", "#7cb342", "#fbc02d", "#f57c00", "#c62828", "#9e9e9e"])
    ax.set_ylabel("% khách xấu (kiểm chứng)"); ax.set_title("Tỷ lệ xấu theo hạng"); fig.tight_layout()
    fig.savefig(fig_dir / "bad_rate_by_band.png", dpi=160); plt.close(fig)


# =====================================================================
def main(rebuild=False):
    cfg = config()                                                          # 1
    con = load_and_check(cfg)                                               # 2
    parts = []
    for c in cfg["dev_cutoffs"] + [cfg["oot_cutoff"]]:                      # 3 + 4
        parts.append(feature_engineering(con, cfg, c, rebuild=rebuild)); log(f"Bước 3–4 – mốc {c}: {len(parts[-1]):,} khách")
    data = pd.concat(parts, ignore_index=True)
    data, dup = validate_features(data, cfg, "dev_oot")                     # 5
    dev = data[data.cutoff.isin(cfg["dev_cutoffs"])].reset_index(drop=True)
    oot = data[data.cutoff == cfg["oot_cutoff"]].reset_index(drop=True)
    log(f"Huấn luyện {len(dev):,} (xấu {dev.bad.mean():.2%}) · kiểm chứng {len(oot):,} (xấu {oot.bad.mean():.2%})")
    M = fit_model(cfg, dev, dup)
    S_dev, b_dev, _, _, rng_f = score_features(M, cfg, dev)                 # 6
    sc_dev, rng_g = aggregate_scores(M, cfg, dev, S_dev, b_dev)             # 7
    S_oot, b_oot, _, _, _ = score_features(M, cfg, oot, rng_f)
    sc_oot, _ = aggregate_scores(M, cfg, oot, S_oot, b_oot, rng_g)
    final = feature_engineering(con, cfg, cfg["score_date"], with_label=False, rebuild=rebuild)
    final, _ = validate_features(final, cfg, "score_date", report=True)
    S_f, b_f, pts_f, miss_f, _ = score_features(M, cfg, final, rng_f)
    sc_f, _ = aggregate_scores(M, cfg, final, S_f, b_f, rng_g)
    expl = explain_scores(M, cfg, final, S_f, miss_f)                       # 8
    rep = validate_scorecard(M, cfg, dev, oot, sc_dev, sc_oot, S_oot, con, sc_f, final)   # 9
    plots(M, cfg, oot, S_oot, sc_oot)
    export_and_monitor(M, cfg, final, sc_f, expl, pts_f, dev, sc_dev, rep)  # 10
    log(f"Xong. Kết quả: {cfg['out']}")


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv)
