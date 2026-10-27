"""EDA các yếu tố ảnh hưởng sức khỏe tài chính (chuyển từ notebook Colab Financial_Health_Score.ipynb, mục 3 và 5.3)
sang dữ liệu clean của nhóm. Dùng làm căn cứ (reasoning) cho việc chọn chỉ tiêu ở Câu 1 và Câu 4.

Đơn vị phân tích: từng HĐ ĐÃ KÝ và đủ 4 kỳ thanh toán (ký trước 01/02/2025), trên toàn bộ lịch sử.
Kết quả: HĐ có trễ ở bất kỳ kỳ nào trong 4 kỳ đầu (any_due) – cùng định nghĩa "xấu" với nhãn huấn luyện.
"""
import numpy as np
import pandas as pd

MATURE_END = "2025-02-01"


def _buckets(con, expr, cases, where="TRUE", table="ct_eda", order=None):
    """Tỷ lệ trễ theo nhóm: cases = list (điều kiện SQL, nhãn); trả về DataFrame nhom, ty_le_tre, so_hd."""
    case = "CASE " + " ".join(f"WHEN {c} THEN '{l}'" for c, l in cases) + " END"
    d = con.execute(f"""SELECT {case} AS nhom, avg(od) AS ty_le_tre, count(*) AS so_hd
                        FROM (SELECT *, {expr} AS v FROM {table} WHERE {where}) GROUP BY 1""").df()
    labels = [l for _, l in cases]
    return d.set_index("nhom").reindex(order or labels).dropna(how="all").reset_index()


def contract_eda(con):
    # bảng HĐ đã ký, đủ 4 kỳ, sắp theo thời gian trong từng khách + thông tin các HĐ TRƯỚC đó của chính khách
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ct_eda AS
    SELECT *,
      row_number() OVER w AS k,
      coalesce(sum(od) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS n_prior_od,
      max(CASE WHEN od = 1 THEN t END) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS last_od_before,
      lag(n_due_flags) OVER w AS prev_flags,
      lag(t) OVER w AS prev_t,
      min(t) OVER (PARTITION BY skp_client) AS first_t
    FROM (SELECT skp_client, case_id, product, is_revolving, t_sign_filled AS t, any_due::INT AS od, n_due_flags,
                 d1, d2, d3, d4, annuity, inc_main, amt_credit, init_pay, flag_ins, tenor, status
          FROM contract WHERE is_signed AND t_sign_filled < TIMESTAMP '{MATURE_END}')
    WINDOW w AS (PARTITION BY skp_client ORDER BY t, case_id)""")
    E = {}
    E["tong_quan"] = con.execute("SELECT count(*) so_hd, count(DISTINCT skp_client) so_kh, avg(od) ty_le_tre FROM ct_eda").df()
    E["san_pham"] = con.execute("""SELECT product AS nhom, avg(od) ty_le_tre, count(*) so_hd FROM ct_eda
                                   GROUP BY 1 HAVING count(*) > 1000 ORDER BY 2 DESC""").df()
    E["theo_ky"] = con.execute("SELECT avg(d1) ky_1, avg(d2) ky_2, avg(d3) ky_3, avg(d4) ky_4 FROM ct_eda").df()
    E["d1_va_d4"] = con.execute("SELECT avg((d4 = 1)::INT) FILTER (WHERE d1 = 1) AS ty_le FROM ct_eda").fetchone()[0]
    E["theo_nam"] = con.execute("""SELECT year(t) AS nhom, avg(od) ty_le_tre, count(*) so_hd FROM ct_eda
                                   WHERE t >= TIMESTAMP '2017-01-01' GROUP BY 1 ORDER BY 1""").df()
    E["trang_thai"] = con.execute("""SELECT status AS nhom, avg(od) ty_le_tre, count(*) so_hd FROM ct_eda
                                     GROUP BY 1 HAVING count(*) > 100 ORDER BY 2 DESC""").df()
    # 3.3 lịch sử trả nợ: HĐ thứ 2 trở đi
    E["so_hd_tre_truoc"] = _buckets(con, "n_prior_od", [("v = 0", "0"), ("v = 1", "1"), ("v = 2", "2"), ("v >= 3", "3+")], "k >= 2")
    E["lan_tre_gan_nhat"] = _buckets(con, "date_diff('day', last_od_before, t) / 30.44",
                                     [("v IS NULL", "Chưa từng"), ("v > 48", "> 4 năm"), ("v > 24", "2–4 năm"), ("v > 12", "1–2 năm"),
                                      ("v > 6", "6–12 tháng"), ("v >= 0", "≤ 6 tháng")], "k >= 2")
    E["muc_do_hd_truoc"] = _buckets(con, "prev_flags", [("v = 0", "0 kỳ"), ("v = 1", "1 kỳ"), ("v = 2", "2 kỳ"), ("v = 3", "3 kỳ"),
                                                         ("v = 4", "4 kỳ")], "k >= 2")
    # 3.4 độ bền
    E["so_hd_tot_truoc"] = _buckets(con, "k - 1", [("v = 0", "0 (HĐ đầu)"), ("v = 1", "1"), ("v = 2", "2"), ("v <= 5", "3–5"),
                                                    ("v > 5", "6+")], "n_prior_od = 0")
    E["tham_nien"] = _buckets(con, "date_diff('day', first_t, t) / 365.25",
                              [("v = 0", "HĐ đầu"), ("v < 1", "< 1 năm"), ("v < 3", "1–3"), ("v < 5", "3–5"), ("v < 8", "5–8"), ("v >= 8", "8+")])
    E["nhip_vay"] = _buckets(con, "date_diff('day', prev_t, t) / 30.44",
                             [("v IS NULL", "HĐ đầu"), ("v < 1", "< 1 tháng"), ("v < 3", "1–3"), ("v < 6", "3–6"), ("v < 12", "6–12"),
                              ("v < 24", "12–24"), ("v >= 24", "> 24 tháng")])
    # 3.5 gánh nặng & thu nhập
    E["pti"] = _buckets(con, "annuity / NULLIF(inc_main, 0)", [("v IS NULL", "Thiếu"), ("v <= .1", "≤ 10%"), ("v <= .2", "10–20%"),
                                                               ("v <= .35", "20–35%"), ("v <= .5", "35–50%"), ("v > .5", "> 50%")],
                        "NOT is_revolving", order=["≤ 10%", "10–20%", "20–35%", "35–50%", "> 50%", "Thiếu"])
    E["thu_nhap"] = _buckets(con, "inc_main", [("v IS NULL OR v <= 0", "Không khai"), ("v <= 5e6", "≤ 5tr"), ("v <= 8e6", "5–8tr"),
                                               ("v <= 12e6", "8–12tr"), ("v <= 20e6", "12–20tr"), ("v > 20e6", "> 20tr")],
                             order=["≤ 5tr", "5–8tr", "8–12tr", "12–20tr", "> 20tr", "Không khai"])
    E["so_tien_vay"] = _buckets(con, "amt_credit", [("v <= 3e6", "≤ 3tr"), ("v <= 6e6", "3–6tr"), ("v <= 10e6", "6–10tr"),
                                                    ("v <= 20e6", "10–20tr"), ("v > 20e6", "> 20tr")], "amt_credit > 0")
    # 3.6 đặc điểm khoản vay
    E["ky_han"] = _buckets(con, "tenor", [("v <= 4", "≤ 4"), ("v <= 6", "5–6"), ("v <= 9", "7–9"), ("v <= 12", "10–12"),
                                          ("v <= 24", "13–24"), ("v > 24", "> 24")], "NOT is_revolving AND tenor > 0")
    E["tra_truoc"] = _buckets(con, "init_pay / NULLIF(init_pay + amt_credit, 0)",
                              [("v = 0", "0%"), ("v <= .2", "≤ 20%"), ("v <= .4", "20–40%"), ("v <= .6", "40–60%"), ("v > .6", "> 60%")],
                              "product IN ('CD','TW') AND init_pay IS NOT NULL AND amt_credit > 0")
    E["bao_hiem"] = con.execute("""SELECT product AS san_pham, avg(od) FILTER (WHERE flag_ins = 0) khong_bao_hiem,
                                          avg(od) FILTER (WHERE flag_ins = 1) co_bao_hiem
                                   FROM ct_eda WHERE product IN ('CD','TW','CL','CW') GROUP BY 1 ORDER BY 1""").df()
    # 3.7 cấp khách hàng: bị từ chối, hạn mức được chào
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE cl_eda AS
    SELECT y.skp_client, y.od, r.rej_share, o.max_offer, cu.gender, cu.region,
           date_diff('year', cu.date_birth, TIMESTAMP '{MATURE_END}') AS age
    FROM (SELECT skp_client, max(od) od FROM ct_eda GROUP BY 1) y
    LEFT JOIN (SELECT skp_client, avg(is_rejected::INT) rej_share FROM contract
               WHERE is_signed OR is_rejected OR is_cancelled GROUP BY 1) r USING (skp_client)
    LEFT JOIN (SELECT skp_client, arg_max(max_offer, t_created) max_offer FROM leads
               WHERE max_offer IS NOT NULL GROUP BY 1) o USING (skp_client)
    LEFT JOIN customer cu USING (skp_client)""")
    E["kh_tong_quan"] = con.execute("SELECT count(*) so_kh, avg(od) ty_le_kh_tung_tre FROM cl_eda").df()
    E["ty_le_tu_choi"] = _buckets(con, "rej_share", [("v = 0", "0%"), ("v <= .25", "≤ 25%"), ("v <= .5", "25–50%"), ("v <= .75", "50–75%"),
                                                     ("v > .75", "> 75%")], table="cl_eda")
    E["max_offer"] = _buckets(con, "max_offer", [("v IS NULL", "Chưa được chào"), ("v <= 5e6", "≤ 5tr"), ("v <= 10e6", "5–10tr"),
                                                 ("v <= 20e6", "10–20tr"), ("v <= 50e6", "20–50tr"), ("v > 50e6", "> 50tr")],
                              table="cl_eda", order=["≤ 5tr", "5–10tr", "10–20tr", "20–50tr", "> 50tr", "Chưa được chào"])
    # 3.9 nhân khẩu học (chỉ để kiểm tra công bằng, không dùng làm biến)
    for col in ["gender", "region"]:
        E[f"nhan_khau_{col}"] = con.execute(f"""SELECT coalesce({col}, 'Thiếu') AS nhom, avg(od) ty_le_tre, count(*) so_hd
                                                FROM cl_eda GROUP BY 1 ORDER BY 1""").df()
    E["nhan_khau_age"] = _buckets(con, "age", [("v < 25", "18–24"), ("v < 35", "25–34"), ("v < 45", "35–44"), ("v < 55", "45–54"),
                                               ("v >= 55", "55+")], "age BETWEEN 18 AND 90", table="cl_eda")
    return E


def relative_risk(dev):
    """RR = tỷ lệ xấu của nhóm CÓ yếu tố ÷ nhóm KHÔNG có, trên đúng quần thể huấn luyện (biến tại mốc, nhãn sau mốc)."""
    d = dev
    flags = {
        "Trễ hạn trong 12 tháng gần nhất": d.n_inst_late_12m.fillna(0).gt(0),
        "Từng có kỳ trễ hạn": d.n_due_total.fillna(0).gt(0),
        "Trễ 2 kỳ liên tiếp trở lên": d.max_late_streak.fillna(0).ge(2),
        "Hồ sơ gần nhất bị từ chối": d.last_app_rejected.eq(1),
        "Tỷ lệ bị từ chối ≥ 50%": d.refused_ratio.ge(.5),
        "Chưa có HĐ nào đủ 4 kỳ (hồ sơ mỏng)": d.n_observed.fillna(0).eq(0),
        "≥ 3 hồ sơ trong 3 tháng": d.n_apply_3m.fillna(0).ge(3),
        "Tiền góp > 50% thu nhập": d.active_annuity_income.gt(.5),
        "Thu nhập > 12 triệu": d.income.gt(12e6),
        "Tỷ lệ trả trước ≥ 30%": d.down_payment_ratio.ge(.3),
        "Mua bảo hiểm khoản vay ≥ 50% HĐ": d.pct_insured2.ge(.5),
        "≥ 2 HĐ tất toán không trễ": d.n_clean_finished.fillna(0).ge(2),
        "Lịch sử vay ≥ 3 năm": d.history_years.ge(3),
        "Hạn mức được chào > 20 triệu": d.max_offer.gt(20e6),
    }
    rows = []
    for name, m in flags.items():
        m = m.fillna(False).astype(bool)
        if m.sum() < 300 or (~m).sum() < 300:
            continue
        r1, r0 = d.bad[m].mean(), d.bad[~m].mean()
        rows.append(dict(yeu_to=name, so_kh=int(m.sum()), ty_trong=m.mean(), xau_khi_co=r1, xau_khi_khong=r0, RR=r1 / r0))
    return pd.DataFrame(rows).set_index("yeu_to").sort_values("RR")
