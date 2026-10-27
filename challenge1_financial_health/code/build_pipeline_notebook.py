"""Sinh solution/Financial_Health_Pipeline.ipynb (pipeline 10 bước, chạy end-to-end, có kết quả và biểu đồ).
Chạy: python solution/build_pipeline_notebook.py   rồi thực thi notebook (xem cuối file)."""
import pathlib
import sys

import nbformat as nbf

OUT = pathlib.Path(__file__).resolve().parent / "Financial_Health_Pipeline.ipynb"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Customer Financial Health Score — pipeline end-to-end

**Câu hỏi của điểm:** *Khách có quản lý được tài chính hiện tại và chống chịu được biến cố không?*

**Phương pháp:** Gradient Boosting (LightGBM) có ràng buộc đơn điệu cho các biến có chiều nghiệp vụ rõ, giải thích bằng **SHAP** (TreeSHAP có sẵn trong LightGBM).
Vì SHAP cộng tính trên log-odds, điểm 0–100 tách được chính xác thành điểm cộng/trừ của từng chỉ tiêu và từng nhóm.

**Dữ liệu:** bộ clean của nhóm `eda/clean_team/Clean_data-20261018T004659Z-1-001` (đã chuyển sang parquet bằng `load_team_clean.py`).

**Thiết kế thời gian** — biến chỉ dùng dữ liệu *trước* mốc, nhãn lấy *sau* mốc:

| Tập | Mốc | Nhãn "xấu" |
|---|---|---|
| Huấn luyện | 01/02, 01/04, 01/06, 01/08/2024 | HĐ trả góp ký trong 2 tháng sau mốc bị trễ ở 4 kỳ đầu hoặc bị xóa nợ |
| Chọn số cây và biến | 01/10/2024 | như trên |
| Kiểm chứng ngoài thời gian | 01/12/2024 | như trên (HĐ ký 12/2024–01/2025) |
| **Chấm điểm thật** | **05/06/2025** | — |
| Kiểm tra dữ liệu độc lập | 06–12/2025 | Cuộc gọi mã OWN* trong CALL: kiểm tra xem có phản ánh rủi ro không (kết quả: không, xem 9f) |

**Cấu trúc notebook theo đề Challenge 1:**

| Câu hỏi của đề | Mục | Bước pipeline |
|---|---|---|
| 1. Phân tích các yếu tố phản ánh sức khỏe tài chính | Câu 1 | 1–5: cấu hình, kiểm tra dữ liệu, snapshot, **mô tả biến đầu vào và feature engineering**, kiểm tra biến, phân tích từng biến |
| 2. Xây dựng điểm 0–100 | Câu 2 | huấn luyện, 6 chấm từng chỉ tiêu, 7 tổng hợp điểm |
| 3. Phân nhóm khách hàng có ý nghĩa kinh doanh | Câu 3 | hạng A–E, N, chân dung và hành động |
| 4. Yếu tố tác động tích cực / tiêu cực | Câu 4 | SHAP toàn cục, 8 giải thích từng khách |
| 5. Bằng chứng phương pháp hợp lý | Câu 5 | 9 kiểm định, 10 xuất và theo dõi |

**Pipeline 10 bước** (mã nguồn: `solution/health_pipeline.py`, mỗi bước là một hàm):

| Bước | Hàm | Đầu ra |
|---|---|---|
| 1. Cấu hình | `config` | ngày chấm, cửa sổ, biến, tham số, ngưỡng, phiên bản |
| 2. Đọc và kiểm tra dữ liệu | `load_and_check` | khóa, trùng, kiểu, thiếu, quan hệ bảng |
| 3. Snapshot khách hàng | `build_snapshot` | mỗi khách một dòng tại ngày chấm, mức độ đầy đủ dữ liệu |
| 4. Tạo biến | `feature_engineering` | 57 chỉ tiêu theo 7 nhóm |
| 5. Kiểm tra biến | `validate_features` | mẫu số 0, giá trị bất hợp lý, biến trùng, độ đầy đủ |
| 6. Chấm từng chỉ tiêu | `score_features` | điểm chỉ tiêu 0–100 |
| 7. Tổng hợp điểm | `aggregate_scores` | điểm nhóm + Financial Health Score + hạng |
| 8. Giải thích | `explain_scores` | 3 lý do trừ điểm, 3 điểm mạnh |
| 9. Kiểm định | `validate_scorecard` | hiệu quả, logic, độ nhạy, bỏ từng nhóm, kiểm tra dữ liệu độc lập |
| 10. Xuất và theo dõi | `export_and_monitor` | file điểm có phiên bản, trôi dạt biến, lịch sử theo dõi |
""")

md("## 0. Thiết lập")
code(r'''
import os, sys, pathlib, warnings
HERE = pathlib.Path.cwd()
if HERE.name != "solution":
    HERE = HERE / "solution"
os.chdir(HERE); sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import health_pipeline as hp
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
%matplotlib inline
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_columns", 40, "display.width", 200, "display.float_format", lambda x: f"{x:,.4f}")
BAND_COLORS = {"A": "#2e7d32", "B": "#7cb342", "C": "#fbc02d", "D": "#f57c00", "E": "#c62828", "N": "#9e9e9e"}
''')

# ---------------------------------------------------------------- 1
md("""
# Câu 1 · Các yếu tố phản ánh sức khỏe tài chính
Phần này đi từ dữ liệu gốc → kiểm tra chất lượng → **biến đầu vào và cách tạo chỉ tiêu** → kiểm tra chỉ tiêu → **phân tích từng chỉ tiêu có thật sự tách được khách tốt/xấu không**.
""")
md("""
## Bước 1 · Cấu hình (`config`)
Mọi tham số nằm ở một chỗ: ngày chấm, các mốc, định nghĩa nhãn, danh sách biến (nhóm, mô tả, chiều ràng buộc), tham số mô hình, thang điểm, ngưỡng hạng, phiên bản.
Chiều ràng buộc: **+1** biến càng lớn rủi ro không được giảm; **−1** biến càng lớn rủi ro không được tăng; **0** để mô hình tự học (quan hệ có thể không đơn điệu).
""")
code(r'''
cfg = hp.config()
display(pd.Series({k: cfg[k] for k in ["version", "score_date", "dev_cutoffs", "valid_cutoff", "oot_cutoff", "mature_days",
                                       "label", "pdo", "base_score", "min_shap_share"]}, name="giá trị").to_frame())
VARS = pd.DataFrame([(k, *v) for k, v in cfg["variables"].items()], columns=["bien", "nhom", "mo_ta", "rang_buoc"])
display(VARS.groupby("nhom").agg(so_bien=("bien", "size"), vi_du=("mo_ta", lambda x: "; ".join(x[:3]))))
''')
md("""
**Đối chiếu 7 nhóm gợi ý với dữ liệu có sẵn**

| Nhóm gợi ý | Có trong dữ liệu | Không có (giới hạn) |
|---|---|---|
| 1. Thu nhập & độ ổn định | thu nhập khai báo, độ biến động, số lần đổi nghề/nguồn thu nhập | thu nhập ròng thực, thời gian làm việc |
| 2. Cân đối thu–chi | thu nhập còn lại sau tiền góp | chi tiêu, chi phí thiết yếu |
| 3. Gánh nặng nợ | số HĐ đang chạy, HĐ quay vòng, tiền góp/thu nhập, khoản vay/thu nhập | dư nợ thật, nợ ở tổ chức khác |
| 4. Lịch sử thanh toán | cờ trễ 4 kỳ đầu, chuỗi trễ, trễ 6/12 tháng, xóa nợ, thời gian từ lần trễ | số ngày quá hạn (DPD), dư nợ quá hạn |
| 5. Dự phòng tài chính | đại diện: tiền trả trước bằng tiền mặt | tiết kiệm, quỹ dự phòng |
| 6. Áp lực vay nợ | số lần xin vay/bị từ chối/tự hủy, hạn mức đề xuất và thay đổi hạn mức | số tổ chức đang vay, cơ cấu nợ |
| 7. Kế hoạch & bảo vệ | mua bảo hiểm khoản vay | tiết kiệm dài hạn, bảo hiểm khác |

Không dùng biến nhân khẩu học (tuổi, giới tính, vùng). Không dùng PAST_BEHAVIOR vì bảng này đã thuộc Engagement Score (hai điểm không dùng chung biến).
CALL chỉ có từ 01/2025 (sau mọi mốc huấn luyện) nên không dùng để huấn luyện; bước 9f kiểm tra xem có dùng được làm dữ liệu độc lập không.
""")

md("""
### 1.1 Biến đầu vào (cột gốc của dữ liệu)
Cột gốc theo từ điển dữ liệu của ban tổ chức, tên cột sau khi làm sạch, và cách dùng trong điểm Health.
""")
code(r'''
import importlib, feature_doc; importlib.reload(feature_doc)
raw = pd.DataFrame(feature_doc.RAW_INPUTS, columns=["Bảng", "Cột gốc", "Cột sau clean", "Ý nghĩa", "Cách dùng trong điểm Health"])
display(raw.style.hide(axis="index").set_properties(**{"text-align": "left"}))
''')
md("""
### 1.2 Feature engineering — 57 chỉ tiêu theo 7 nhóm
**Nguyên tắc chung**
- **Cấp khách hàng tại một thời điểm:** mọi chỉ tiêu được gộp từ nhiều hồ sơ/HĐ về một dòng cho mỗi khách, *chỉ dùng dữ liệu trước ngày chấm*.
- **Độ chín của lịch sử trả nợ:** cờ FIRST..FOURTH_DUE là trạng thái cuối cùng của dữ liệu, nên chỉ dùng cho HĐ trả góp ký trước ngày chấm ít nhất 125 ngày (4 kỳ đã đến hạn) → tránh rò rỉ thông tin tương lai.
- **HĐ đang chạy:** HĐ trả góp đã ký và chưa hết kỳ hạn (ngày ký + số kỳ × 30 ngày > ngày chấm), hoặc HĐ quay vòng đang Active.
- **Cửa sổ thời gian:** 1, 3, 6, 12, 24 tháng và 3 năm, để đo cả *mức độ* và *xu hướng gần đây*.
- **Tỷ lệ:** mẫu số bằng 0 → để trống (xử lý ở bước 5), không thay bằng 0.
- **Chiều ràng buộc:** +1 = càng lớn càng rủi ro, −1 = càng lớn càng an toàn, 0 = để mô hình học (quan hệ có thể hình chữ U).
""")
code(r'''
fe = pd.DataFrame([(cfg["variables"][v][0], v, cfg["variables"][v][1], *feature_doc.FEATURE_DOC[v], cfg["variables"][v][2])
                   for v in cfg["variables"]],
                  columns=["Nhóm", "Mã biến", "Chỉ tiêu", "Cột nguồn", "Công thức", "Cửa sổ", "Lý do nghiệp vụ", "Ràng buộc"])
fe.to_csv(cfg["out"] / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
for g, t in fe.groupby("Nhóm"):
    display(Markdown(f"**{g}** — {len(t)} chỉ tiêu"))
    display(t.drop(columns="Nhóm").style.hide(axis="index").set_properties(**{"text-align": "left"}))
''')

# ---------------------------------------------------------------- 2
md("## Bước 2 · Đọc và kiểm tra dữ liệu (`load_and_check`)\nPipeline dừng nếu khóa bị trùng/null hoặc thiếu cột bắt buộc.")
code(r'''
con = hp.load_and_check(cfg)
display(Markdown("**Kiểm tra bảng**"), pd.read_csv(cfg["out"] / "02_table_checks.csv"))
display(Markdown("**Kiểm tra logic CONTRACT**"), pd.read_csv(cfg["out"] / "02_logic_checks.csv", index_col=0))
miss = pd.read_csv(cfg["out"] / "02_column_missing.csv")
fig, ax = plt.subplots(figsize=(8, 4))
m = miss.sort_values("ty_le_thieu")
ax.barh(m.bang + "." + m.cot, m.ty_le_thieu * 100, color="#3b6fb6"); ax.set_xlabel("% thiếu")
ax.set_title("Tỷ lệ thiếu của các cột dùng cho điểm Health"); plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- 3
md("## Bước 3 · Snapshot khách hàng (`build_snapshot`)\nChỉ giữ dữ liệu có trước ngày chấm; mỗi khách một dòng. Hàm tự kiểm tra không có bản ghi nào sau mốc lọt vào, và gắn **mức độ đầy đủ dữ liệu** cho từng khách.")
code(r'''
snap = hp.build_snapshot(con, cfg["score_date"])
comp = snap.data_completeness.value_counts()
display(comp.to_frame("so_khach").assign(ty_trong=lambda d: d.so_khach / d.so_khach.sum()))
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(comp.index[::-1], comp.values[::-1] / 1e3, color="#5b8f3a"); ax.set_xlabel("nghìn khách")
ax.set_title(f"Mức độ đầy đủ dữ liệu tại ngày chấm {cfg['score_date']}"); plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- 4
md("## Bước 4 · Tạo biến (`feature_engineering`)\nTạo biến cho 5 mốc huấn luyện và mốc kiểm chứng (lần đầu khoảng 1 phút/mốc, sau đó đọc từ cache).")
code(r'''
parts = [hp.feature_engineering(con, cfg, c) for c in cfg["dev_cutoffs"] + [cfg["oot_cutoff"]]]
data = pd.concat(parts, ignore_index=True)
by = data.groupby("cutoff").agg(so_khach=("bad", "size"), ty_le_xau=("bad", "mean"))
display(by)
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(by.index, by.ty_le_xau * 100, color=["#3b6fb6"] * 5 + ["#f57c00"])
for i, v in enumerate(by.ty_le_xau): ax.text(i, v * 100, f"{v:.2%}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("% khách xấu"); ax.set_title("Tỷ lệ xấu theo mốc (cam = kiểm chứng)"); plt.tight_layout(); plt.show()
display(data[["skp_client", "cutoff"] + list(cfg["variables"])[:12]].head())
''')

# ---------------------------------------------------------------- 5
md("## Bước 5 · Kiểm tra biến (`validate_features`)\nGiá trị vô cực (mẫu số bằng 0) và giá trị bất hợp lý (tỷ lệ ngoài 0–1, số âm, thu nhập bằng 0) được đặt là **thiếu**. Cặp biến có |Spearman| > 0.95 được báo cáo; chỉ loại biến khi cả Spearman và Pearson > 0.99 (các biến đếm trễ hạn có ~97% giá trị 0 nên Spearman gần 1 dù mang thông tin khác nhau).")
code(r'''
data, dup = hp.validate_features(data, cfg, "dev_oot")
q = pd.read_csv(cfg["out"] / "05_feature_quality_dev_oot.csv")
display(q[(q.vo_cuc > 0) | (q.bat_hop_ly > 0)][["bien", "nhom", "co_du_lieu", "vo_cuc", "bat_hop_ly"]])
display(Markdown("**Cặp biến trùng thông tin**"), dup)
fig, ax = plt.subplots(figsize=(7, 3))
cov = q.groupby("nhom").co_du_lieu.mean().sort_values()
ax.barh(cov.index, cov * 100, color="#3b6fb6"); ax.set_xlabel("% khách có dữ liệu (trung bình các biến)")
ax.set_title("Độ đầy đủ theo nhóm chỉ tiêu"); plt.tight_layout(); plt.show()
dev = data[data.cutoff.isin(cfg["dev_cutoffs"])].reset_index(drop=True)
oot = data[data.cutoff == cfg["oot_cutoff"]].reset_index(drop=True)
import gc; del parts, data, snap; gc.collect()   # giải phóng bộ nhớ (máy 16GB)
''')
md("""
### 1.3 Phân tích từng chỉ tiêu: có tách được khách tốt / xấu không?
Trên tập huấn luyện: AUC đơn biến (0,5 = không phân biệt được), chiều thực tế (giá trị cao đi cùng rủi ro cao hay thấp), tỷ lệ xấu khi thiếu dữ liệu, và tỷ lệ xấu theo nhóm giá trị.
""")
code(r'''
from sklearn.metrics import roc_auc_score
rows = []
for v in cfg["variables"]:
    x = dev[v].astype(float)
    if x.notna().sum() < 1000 or x.nunique() < 2:
        continue
    a = roc_auc_score(dev.bad, x.fillna(x.median()))
    rows.append(dict(Nhóm=cfg["variables"][v][0], Chỉ_tiêu=cfg["variables"][v][1], Mã=v, Có_dữ_liệu=x.notna().mean(),
                     AUC_đơn_biến=max(a, 1 - a), Chiều_thực_tế="cao → rủi ro hơn" if a > .5 else "cao → an toàn hơn",
                     Xấu_khi_thiếu=dev.bad[x.isna()].mean() if x.isna().any() else np.nan))
uni = pd.DataFrame(rows).sort_values("AUC_đơn_biến", ascending=False)
display(uni.head(25).style.hide(axis="index").format({"Có_dữ_liệu": "{:.1%}", "AUC_đơn_biến": "{:.3f}", "Xấu_khi_thiếu": "{:.2%}"}))
gbest = uni.groupby("Nhóm").AUC_đơn_biến.max().sort_values()
fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(gbest.index, gbest, color="#3b6fb6"); ax.set_xlim(.5, max(.7, gbest.max() + .02))
ax.set_title("AUC đơn biến tốt nhất trong mỗi nhóm"); plt.tight_layout(); plt.show()
''')
code(r'''
key = ["max_offer", "on_time_rate", "refused_ratio", "down_payment_ratio", "n_inst_late_12m", "pct_insured2",
       "active_annuity_income", "inc_cv", "n_apply_3m"]
fig, axes = plt.subplots(3, 3, figsize=(14, 9))
for ax, v in zip(axes.ravel(), key):
    x = dev[v].astype(float)
    q = np.unique(np.nanquantile(x, [0, .2, .4, .6, .8, 1]))
    b = pd.cut(x, q, include_lowest=True, duplicates="drop") if len(q) > 2 else pd.cut(x, [-np.inf, 0.5, np.inf])
    lab = b.cat.categories.astype(str).tolist()
    t = dev.groupby(b, observed=False).bad.mean().tolist() + ([dev.bad[x.isna()].mean()] if x.isna().any() else [])
    lab += ["Thiếu"] if x.isna().any() else []
    ax.bar(range(len(t)), np.array(t) * 100, color=["#9e9e9e" if l == "Thiếu" else "#3b6fb6" for l in lab])
    ax.set_xticks(range(len(t))); ax.set_xticklabels([l if len(l) < 20 else l[:19] + "…" for l in lab], rotation=35, fontsize=7)
    ax.set_title(cfg["variables"][v][1], fontsize=9); ax.set_ylabel("% xấu")
plt.suptitle("Tỷ lệ xấu theo nhóm giá trị (phân vị; xám = thiếu dữ liệu) — tập huấn luyện"); plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- model
md("""
# Câu 2 · Customer Financial Health Score 0–100
**Vì sao Gradient Boosting + SHAP?** Trên cùng tập kiểm chứng, scorecard WoE truyền thống đạt AUC 0,661; Gradient Boosting cao hơn vì học được quan hệ phi tuyến và tương tác giữa các chỉ tiêu (ví dụ: bị từ chối nhiều chỉ đáng lo khi hạn mức thấp).
Để vẫn giải thích được như scorecard: (1) ràng buộc đơn điệu cho các biến có chiều nghiệp vụ rõ; (2) SHAP tách điểm thành điểm cộng/trừ của từng chỉ tiêu.
""")
md("""
## Huấn luyện mô hình
1. Huấn luyện trên 4 mốc đầu, dừng sớm theo AUC mốc 10/2024.
2. Bỏ biến đóng góp dưới 0,5% tổng |SHAP| (đo trên mốc 10/2024, không đụng tập kiểm chứng).
3. Fit lại trên cả 5 mốc với số cây đã chọn.
""")
code(r'''
M = hp.fit_model(cfg, dev, dup)
sel = VARS.set_index("bien").loc[M["feats"]].reset_index()
display(sel.groupby("nhom").bien.apply(lambda x: ", ".join(x)).to_frame("bien_duoc_giu"))
''')

# ---------------------------------------------------------------- 6
md("""
## Bước 6 · Chấm từng chỉ tiêu (`score_features`)
Điểm chỉ tiêu 0–100 = vị trí đóng góp SHAP của chỉ tiêu đó, nằm giữa mức tệ nhất (phân vị 1%) và tốt nhất (phân vị 99%) trên tập huấn luyện.
**100** = chỉ tiêu đang giúp khách nhiều nhất, **0** = đang kéo điểm xuống nhiều nhất.

**Quy tắc thiếu dữ liệu:** LightGBM học một nhánh riêng cho giá trị thiếu, nên khách thiếu một chỉ tiêu nhận đúng mức rủi ro quan sát được của nhóm khách cũng thiếu chỉ tiêu đó (thường là khách chưa có lịch sử, nên bị trừ điểm nhẹ). Cột `miss_<biến>` đánh dấu để báo cáo.
""")
code(r'''
S_dev, b_dev, pts_dev, miss_dev, rng_f = hp.score_features(M, cfg, dev)
S_oot, b_oot, pts_oot, miss_oot, _ = hp.score_features(M, cfg, oot, rng_f)
ex = pts_oot.sample(5, random_state=1).round(0)
ex.columns = [cfg["variables"][c][1] for c in ex.columns]
display(Markdown("**Điểm chỉ tiêu của 5 khách ngẫu nhiên (tập kiểm chứng)**"), ex.T.head(15))
del pts_dev, miss_dev, pts_oot, miss_oot; gc.collect()
''')

# ---------------------------------------------------------------- 7
md("""
## Bước 7 · Tổng hợp điểm (`aggregate_scores`)
**Financial Health Score = 50 + 15/ln2 × (log-odds tốt − log-odds tốt trung bình của tập huấn luyện)**, cắt về 0–100.
50 điểm = mức rủi ro trung bình; mỗi +15 điểm = odds tốt/xấu gấp đôi.
Vì SHAP cộng tính: **điểm = điểm gốc + Σ điểm cộng/trừ của từng nhóm** (cột `diem_cong_tru_nhom_*`). Điểm nhóm 0–100 tính như điểm chỉ tiêu nhưng trên tổng SHAP của nhóm.

**Không đủ dữ liệu:** khách chưa có khoản vay nào (trả góp, BNPL hoặc thẻ) qua **≥ 3 kỳ trả** vẫn có điểm nhưng được xếp hạng **N · Chưa đủ lịch sử trả nợ**, gồm cả khách chưa từng vay và khách có khoản vay chưa đến kỳ trả. Nhóm được định nghĩa theo **bằng chứng trả nợ**, không theo loại sản phẩm: thiếu bằng chứng thì không xếp hạng khỏe.
""")
code(r'''
sc_dev, rng_g = hp.aggregate_scores(M, cfg, dev, S_dev, b_dev)
sc_oot, _ = hp.aggregate_scores(M, cfg, oot, S_oot, b_oot, rng_g)
display(sc_oot.head())
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
bins = np.arange(0, 101, 2)
for d, lab in [(sc_dev, "Huấn luyện"), (sc_oot, "Kiểm chứng")]:
    axes[0].hist(d.financial_health_score, bins=bins, alpha=.55, density=True, label=lab)
for lo, hi, b in cfg["bands"]:
    axes[0].axvline(lo, color="k", lw=.5, ls=":")
axes[0].legend(); axes[0].set_title("Phân bố Financial Health Score"); axes[0].set_xlabel("điểm")
o = sc_oot.assign(bad=oot.bad.values, bin=pd.cut(sc_oot.financial_health_score, np.arange(0, 101, 5)))
r = o.groupby("bin", observed=True).bad.mean()
axes[1].plot([i.mid for i in r.index], r.values * 100, "o-", color="#c62828")
axes[1].set_title("Tỷ lệ xấu theo điểm (kiểm chứng)"); axes[1].set_xlabel("điểm"); axes[1].set_ylabel("% xấu")
plt.tight_layout(); plt.show()
''')

md("### Chấm điểm toàn bộ khách tại ngày chấm")
code(r'''
final = hp.feature_engineering(con, cfg, cfg["score_date"], with_label=False)
final, _ = hp.validate_features(final, cfg, "score_date", report=False)
S_f, b_f, pts_f, miss_f, _ = hp.score_features(M, cfg, final, rng_f)
sc_f, _ = hp.aggregate_scores(M, cfg, final, S_f, b_f, rng_g)
display(sc_f.financial_health_score.describe().to_frame("điểm").T)
fig, ax = plt.subplots(figsize=(8, 3.2))
for b, t in sc_f.groupby("health_band"):
    ax.hist(t.financial_health_score, bins=np.arange(0, 101, 1), color=BAND_COLORS[b], alpha=.8, label=cfg["band_name"][b])
ax.legend(fontsize=7); ax.set_xlabel("điểm"); ax.set_ylabel("số khách"); ax.set_title(f"Financial Health Score – {len(sc_f):,} khách tại {cfg['score_date']}")
plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- Câu 3
md("""
# Câu 3 · Phân nhóm khách hàng
Ngưỡng hạng được chọn sao cho tỷ lệ xấu tăng rõ rệt giữa các hạng và mỗi hạng đủ lớn để có chính sách riêng. Khách chưa có khoản vay nào qua ≥ 3 kỳ trả được tách riêng thành **N · Chưa đủ lịch sử trả nợ** (điểm dựa trên rất ít bằng chứng; chính sách: hạn mức nhỏ BNPL/CD, tăng dần sau mỗi kỳ trả đúng hạn).
""")
code(r'''
seg = pd.DataFrame([(b, cfg["band_name"][b], f"{lo}–{hi if hi < 101 else 100}", cfg["band_action"][b]) for lo, hi, b in cfg["bands"][::-1]]
                   + [("N", cfg["band_name"]["N"], "chưa có khoản vay nào qua ≥ 3 kỳ trả", cfg["band_action"]["N"])],
                   columns=["hạng", "tên", "khoảng điểm", "hành động gợi ý"]).set_index("hạng")
o = sc_oot.assign(bad=oot.bad.values)
seg["tỷ lệ xấu (kiểm chứng)"] = o.groupby("health_band").bad.mean()
seg["tỷ trọng (kiểm chứng)"] = o.health_band.value_counts(normalize=True)
seg["số khách hiện tại"] = sc_f.health_band.value_counts()
seg["tỷ trọng hiện tại"] = sc_f.health_band.value_counts(normalize=True)
display(seg.style.format({"tỷ lệ xấu (kiểm chứng)": "{:.2%}", "tỷ trọng (kiểm chứng)": "{:.1%}", "tỷ trọng hiện tại": "{:.1%}",
                          "số khách hiện tại": "{:,.0f}"}).set_properties(**{"text-align": "left"}))
''')
md("**Chân dung từng hạng** (trung vị chỉ tiêu, khách hiện tại) — giúp đặt tên và chính sách cho từng nhóm.")
code(r'''
prof_vars = ["income", "max_offer", "on_time_rate", "n_due_total", "refused_ratio", "n_apply_6m", "n_active",
             "active_annuity_income", "down_payment_ratio", "pct_insured2", "history_years"]
pf = final.assign(band=sc_f.health_band.values)
prof = pf.groupby("band")[prof_vars].median().reindex(list("ABCDEN"))
prof.insert(0, "late", pf.assign(late=pf.n_due_total.fillna(0) > 0).groupby("band").late.mean())
prof.columns = ["% có HĐ trễ"] + [cfg["variables"][v][1] for v in prof_vars]
display(prof.T.style.format("{:,.2f}"))
z = (prof - prof.mean()) / prof.std()
fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(z.T.values, cmap="RdYlGn_r", aspect="auto", vmin=-1.5, vmax=1.5)
ax.set_xticks(range(len(z.index))); ax.set_xticklabels([cfg["band_name"][b] for b in z.index], fontsize=8)
ax.set_yticks(range(len(z.columns))); ax.set_yticklabels(z.columns, fontsize=8)
ax.set_title("Chân dung hạng (chuẩn hóa theo hàng; đỏ = cao hơn các hạng khác)"); plt.colorbar(im, ax=ax, shrink=.7)
plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- SHAP global
md("""
# Câu 4 · Yếu tố tác động tích cực / tiêu cực đến sức khỏe tài chính
### Giải thích toàn cục bằng SHAP
- **Mức quan trọng:** trung bình |SHAP| của từng biến.
- **Trọng số thực tế của nhóm:** tỷ trọng |SHAP| cộng theo nhóm (thay cho trọng số gán tay).
- **Beeswarm:** mỗi chấm là một khách; trục x > 0 là tăng xác suất xấu (giảm điểm); màu đỏ là giá trị biến cao.
- **Dependence:** cách điểm thay đổi theo giá trị của biến.
""")
code(r'''
desc = {f: cfg["variables"][f][1] for f in M["feats"]}
G = pd.Series({f: cfg["variables"][f][0] for f in M["feats"]})
imp = S_oot.abs().mean().sort_values(ascending=False)
gi = imp.groupby(G).sum().sort_values()
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1]})
top = imp.index[:20][::-1]
axes[0].barh([desc[f] for f in top], imp[top], color=[plt.cm.tab10(sorted(G.unique()).index(G[f])) for f in top])
axes[0].set_title("20 biến ảnh hưởng nhiều nhất (màu = nhóm)"); axes[0].set_xlabel("trung bình |SHAP|")
axes[1].barh(gi.index, gi / gi.sum() * 100, color=[plt.cm.tab10(sorted(G.unique()).index(g)) for g in gi.index])
for i, v in enumerate(gi / gi.sum() * 100): axes[1].text(v, i, f" {v:.1f}%", va="center")
axes[1].set_title("Trọng số thực tế của từng nhóm"); axes[1].set_xlabel("% tổng |SHAP|")
plt.tight_layout(); plt.show()
''')
code(r'''
X = oot[M["feats"]].astype(float)
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(9, 7.5))
for i, f in enumerate(top):
    idx = rng.choice(len(X), 4000, replace=False)
    col = X[f].rank(pct=True).fillna(-1).values[idx]; s = S_oot[f].values[idx]; m = col < 0
    sc = ax.scatter(s[~m], i + rng.normal(0, .12, (~m).sum()), c=col[~m], cmap="coolwarm", s=3, alpha=.6, vmin=0, vmax=1)
    ax.scatter(s[m], i + rng.normal(0, .12, m.sum()), c="#9a9a9a", s=3, alpha=.4)
ax.set_yticks(range(len(top))); ax.set_yticklabels([desc[f] for f in top]); ax.axvline(0, color="k", lw=.6)
plt.colorbar(sc, ax=ax, label="phân vị giá trị biến", shrink=.6)
ax.set_xlabel("SHAP (> 0: tăng xác suất xấu → giảm điểm)"); ax.set_title("SHAP beeswarm · xám = thiếu dữ liệu")
plt.tight_layout(); plt.show()
''')
code(r'''
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axes.ravel(), imp.index[:6]):
    ok = X[f].notna()
    ax.scatter(X.loc[ok, f].clip(upper=X.loc[ok, f].quantile(.99)), -M["factor"] * S_oot.loc[ok, f], s=2, alpha=.2, color="#3b6fb6")
    ax.axhline(0, color="k", lw=.6); ax.set_title(desc[f], fontsize=9); ax.set_ylabel("điểm cộng/trừ")
plt.suptitle("Dependence: điểm cộng/trừ theo giá trị biến (6 biến đứng đầu)"); plt.tight_layout(); plt.show()
''')
md("**Tổng hợp yếu tố tích cực / tiêu cực:** chiều = tương quan hạng giữa giá trị chỉ tiêu và điểm cộng/trừ; mức tác động = trung bình |điểm cộng/trừ|.")
code(r'''
rows = []
for f in imp.index:
    ok = X[f].notna()
    r = np.corrcoef(X.loc[ok, f].rank(), -S_oot.loc[ok, f])[0, 1] if ok.sum() > 100 else np.nan
    pts = -M["factor"] * S_oot[f]
    rows.append(dict(Nhóm=G[f], Chỉ_tiêu=desc[f], Tác_động_TB_điểm=(M["factor"] * S_oot[f].abs()).mean(),
                     Chiều=("Tích cực: càng cao càng khỏe" if r > .1 else "Tiêu cực: càng cao càng yếu" if r < -.1 else "Phi tuyến / phụ thuộc ngữ cảnh"),
                     Điểm_cộng_tối_đa=pts.quantile(.99), Điểm_trừ_tối_đa=pts.quantile(.01)))
pn = pd.DataFrame(rows)
display(pn.head(20).style.hide(axis="index").format({"Tác_động_TB_điểm": "{:.2f}", "Điểm_cộng_tối_đa": "{:+.1f}", "Điểm_trừ_tối_đa": "{:+.1f}"}))
''')

# ---------------------------------------------------------------- 8
md("## Bước 8 · Giải thích từng khách (`explain_scores`)\nMỗi khách nhận 3 yếu tố kéo điểm xuống, 3 điểm mạnh (kèm số điểm cộng/trừ), và tỷ lệ chỉ tiêu có dữ liệu.")
code(r'''
expl = hp.explain_scores(M, cfg, final, S_f, miss_f)
view = pd.concat([sc_f[["skp_client", "financial_health_score", "band_name"]], expl], axis=1)
display(view.groupby(sc_f.health_band.values).sample(1, random_state=3)
        [["skp_client", "financial_health_score", "band_name", "ly_do_tru_diem_1", "diem_tru_1", "ly_do_tru_diem_2",
          "diem_manh_1", "diem_cong_1", "diem_manh_2", "ty_le_chi_tieu_co_du_lieu"]])
''')
code(r'''
# Waterfall: điểm của một khách hạng D được tạo thành như thế nào
i = sc_f.index[sc_f.health_band.eq("D")][0]
contrib = (-M["factor"] * S_f.loc[i]).sort_values(key=abs, ascending=False)
top_c = contrib.head(10); rest = contrib.iloc[10:].sum()
base_pts = cfg["base_score"] + M["factor"] * (-b_f[0] - M["ref"])
vals = [base_pts] + list(top_c.values) + [rest]
labels = ["Điểm gốc"] + [desc[f] for f in top_c.index] + ["Các biến còn lại"]
cum = np.cumsum([0] + vals[:-1])
fig, ax = plt.subplots(figsize=(9, 5))
for k, (v, c0) in enumerate(zip(vals, cum)):
    ax.barh(k, v, left=c0 if k else 0, color="#9e9e9e" if k == 0 else ("#2e7d32" if v > 0 else "#c62828"))
    ax.text((c0 if k else 0) + v, k, f" {v:+.1f}" if k else f" {v:.1f}", va="center", fontsize=8)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.axvline(sc_f.financial_health_score[i], color="k", ls="--", lw=.8)
ax.set_title(f"Khách {sc_f.skp_client[i]}: {sc_f.financial_health_score[i]:.1f} điểm ({sc_f.band_name[i]})")
ax.set_xlabel("điểm"); plt.tight_layout(); plt.show()
''')

# ---------------------------------------------------------------- 9
md("""
# Câu 5 · Bằng chứng phương pháp chấm điểm hợp lý
## Bước 9 · Kiểm định (`validate_scorecard`)
- **9a. Hiệu quả:** AUC, Gini, KS, tỷ lệ khách xấu nằm trong 10% điểm thấp nhất; từng mốc; PSI giữa tập huấn luyện và tập kiểm chứng.
- **9b. Hạng:** tỷ lệ xấu phải tăng dần từ A đến E.
- **9c. Logic:** chiều tác động của mỗi biến so với kỳ vọng nghiệp vụ.
- **9d. Bỏ từng nhóm biến:** huấn luyện lại mô hình khi không có nhóm đó, đo mức AUC giảm.
- **9e. Độ nhạy:** dời ngưỡng hạng ±2 điểm; đổi PDO.
- **9f. Dữ liệu độc lập:** CALL là nguồn duy nhất có dữ liệu sau ngày chấm. Kiểm tra xem cuộc gọi mã OWN* (trông như nhắc nợ) có phản ánh rủi ro thật không trước khi dùng làm thước đo.
""")
code(r'''
rep = hp.validate_scorecard(M, cfg, dev, oot, sc_dev, sc_oot, S_oot, con, sc_f, final)
display(Markdown("**9a. Hiệu quả**"), rep["performance"])
''')
code(r'''
from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for d, s, lab in [(dev, sc_dev, "Huấn luyện"), (oot, sc_oot, "Kiểm chứng")]:
    fpr, tpr, _ = roc_curve(d.bad, -s.financial_health_score)
    axes[0].plot(fpr, tpr, label=f"{lab} (AUC {np.trapz(tpr, fpr):.3f})")
axes[0].plot([0, 1], [0, 1], "k:", lw=.6); axes[0].legend(); axes[0].set_title("ROC"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
s = np.sort(sc_oot.financial_health_score.values); yb = oot.bad.values[np.argsort(sc_oot.financial_health_score.values)]
cb, cg = np.cumsum(yb) / yb.sum(), np.cumsum(1 - yb) / (1 - yb).sum()
axes[1].plot(s, cb, label="xấu", color="#c62828"); axes[1].plot(s, cg, label="tốt", color="#2e7d32")
k = np.argmax(cb - cg); axes[1].vlines(s[k], cg[k], cb[k], color="k", ls="--")
axes[1].set_title(f"KS = {(cb - cg)[k]:.3f} tại {s[k]:.0f} điểm (kiểm chứng)"); axes[1].legend(); axes[1].set_xlabel("điểm")
pf = rep["performance"]; pf = pf[pf.Tap.str.contains("mốc|Kiểm")]
axes[2].bar(pf.Tap.str.replace("  mốc ", ""), pf.AUC, color=["#3b6fb6"] * (len(pf) - 1) + ["#f57c00"])
axes[2].set_ylim(.6, .8); axes[2].set_title("AUC từng mốc (cam = kiểm chứng)"); axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
''')
code(r'''
bt = rep["bands"].reindex(list("ABCDEN"))
display(Markdown(f"**9b. Hạng (kiểm chứng)** — tỷ lệ xấu tăng dần A→E: **{rep['band_monotonic']}**"), bt)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].bar(bt.index, bt.ty_le_xau * 100, color=[BAND_COLORS[b] for b in bt.index])
for i, v in enumerate(bt.ty_le_xau): axes[0].text(i, v * 100, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
axes[0].set_title("Tỷ lệ xấu theo hạng (kiểm chứng)"); axes[0].set_ylabel("% xấu")
axes[1].bar(bt.index, bt.ty_trong * 100, color=[BAND_COLORS[b] for b in bt.index]); axes[1].set_title("Tỷ trọng khách theo hạng"); axes[1].set_ylabel("%")
plt.tight_layout(); plt.show()
''')
code(r'''
display(Markdown("**9c. Logic chiều tác động** (biến có ràng buộc phải 'Đạt')"),
        rep["logic"].sort_values(["rang_buoc", "nhom"], ascending=[False, True]))
''')
code(r'''
dg = rep["drop_group"].sort_values("AUC_giam")
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(dg.bo_nhom, dg.AUC_giam, color="#3b6fb6")
for i, v in enumerate(dg.AUC_giam): ax.text(v, i, f" {v:+.4f}", va="center", fontsize=8)
ax.set_title(f"9d. AUC giảm khi bỏ từng nhóm (AUC đầy đủ = {rep['base_auc']:.4f})"); ax.set_xlabel("mức giảm AUC kiểm chứng")
plt.tight_layout(); plt.show()
display(Markdown("**9e. Độ nhạy**"), rep["sensitivity"])
''')
code(r'''
ind = rep["independent"].reindex(["Chưa có kỳ đến hạn", "0 kỳ trễ", "1 kỳ trễ", "≥ 2 kỳ trễ"])
display(Markdown(f"""**9f. Kiểm tra CALL làm dữ liệu độc lập** (khách đang có khoản vay tại ngày chấm, cuộc gọi trong 6 tháng sau)

- AUC dự báo "bị gọi mã OWN*" bằng **số kỳ trễ 12 tháng** (lịch sử trễ thật): **{rep['independent_auc_late']:.3f}**
- AUC bằng Financial Health Score: {rep['independent_auc']:.3f}
- Kết luận: **{'dùng được' if rep['independent_usable'] else 'KHÔNG dùng được làm thước đo rủi ro'}**"""), ind)
display(Markdown("Loại ưu đãi (offer_type) của các cuộc gọi mã OWN*:"), rep["own_offer"])
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(ind.index, ind.ty_le_bi_goi_OWN * 100, color=["#9e9e9e", "#2e7d32", "#f57c00", "#c62828"])
for i, v in enumerate(ind.ty_le_bi_goi_OWN): ax.text(i, v * 100, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
ax.set_title("Tỷ lệ bị gọi mã OWN* theo lịch sử trễ hạn 12 tháng"); ax.set_ylabel("%")
plt.tight_layout(); plt.show()
''')
md("""
**Diễn giải 9f:** nếu OWN* là cuộc gọi nhắc nợ, khách từng trễ hạn phải bị gọi nhiều hơn rõ rệt. Dữ liệu cho thấy ngược lại: tỷ lệ bị gọi gần như như nhau (khách trễ ≥ 2 kỳ còn bị gọi ít hơn), và phần lớn cuộc gọi OWN* mang loại ưu đãi *PRODUCT OFFER*, tức cuộc gọi chào bán.
Vì vậy CALL **không** được dùng làm bằng chứng độc lập. Bằng chứng chính của mô hình là **kiểm chứng ngoài thời gian** (9a–9b): dữ liệu 12/2024–01/2025 hoàn toàn không tham gia huấn luyện hay chọn biến.
""")

# ---------------------------------------------------------------- 10
md("""
## Bước 10 · Xuất và theo dõi (`export_and_monitor`)
Xuất file điểm có ngày chấm và phiên bản, kèm hành động gợi ý theo hạng; lưu mô hình và cấu hình; đo trôi dạt từng biến (PSI, tỷ lệ thiếu) so với tập huấn luyện; ghi lịch sử theo dõi.

**Lưu ý khi đọc PSI tại ngày chấm:** tập huấn luyện chỉ gồm khách **vừa được duyệt một HĐ trả góp mới** sau mỗi mốc (chỉ họ mới có nhãn tốt/xấu), còn ngày chấm gồm **toàn bộ khách**, kể cả khách bị từ chối, khách lâu không vay. Vì vậy PSI cao ở đây phản ánh **khác biệt tập khách**, không phải mô hình bị lỗi thời. Ô bên dưới tách phân bố điểm theo loại khách để thấy phần nào do khác biệt tập khách.
""")
code(r'''
res, mon = hp.export_and_monitor(M, cfg, final, sc_f, expl, pts_f, dev, sc_dev, rep)
dist = res.health_band.value_counts().reindex(list("ABCDEN"))
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(dist.index, dist.values / 1e3, color=[BAND_COLORS[b] for b in dist.index])
for i, v in enumerate(dist.values): axes[0].text(i, v / 1e3, f"{v / dist.sum():.1%}", ha="center", va="bottom", fontsize=8)
axes[0].set_title(f"Phân bố hạng – toàn bộ {len(res):,} khách tại {cfg['score_date']}"); axes[0].set_ylabel("nghìn khách")
mm = mon.head(12)[::-1]
axes[1].barh(mm.mo_ta, mm.PSI, color=["#c62828" if p > .25 else "#fbc02d" if p > .1 else "#2e7d32" for p in mm.PSI])
axes[1].axvline(.1, color="k", ls=":", lw=.6); axes[1].axvline(.25, color="k", ls="--", lw=.6)
axes[1].set_title("Trôi dạt biến so với huấn luyện (PSI)"); plt.tight_layout(); plt.show()
from scorecard import psi
grp = {"Hồ sơ gần nhất bị từ chối": final.last_app_rejected.eq(1).values,
       "Ký HĐ trong 60 ngày gần nhất": final.days_since_last_signed.le(60).values,
       "Toàn bộ khách": np.ones(len(final), bool)}
cmp_ = pd.DataFrame({k: {"so_khach": m.sum(), "PSI_diem_vs_huan_luyen": psi(sc_dev.financial_health_score.values, res.financial_health_score.values[m]),
                         **{f"ty_trong_{b}": (res.health_band.values[m] == b).mean() for b in "AE"}} for k, m in grp.items()}).T
cmp_.loc["Tập kiểm chứng 12/2024"] = [len(sc_oot), psi(sc_dev.financial_health_score.values, sc_oot.financial_health_score.values),
                                     (sc_oot.health_band == "A").mean(), (sc_oot.health_band == "E").mean()]
display(Markdown("**Phân bố điểm theo loại khách**"), cmp_.style.format({"so_khach": "{:,.0f}", "PSI_diem_vs_huan_luyen": "{:.3f}",
                                                                          "ty_trong_A": "{:.1%}", "ty_trong_E": "{:.1%}"}))
display(Markdown("""**Diễn giải và giới hạn:** khách có hồ sơ gần nhất bị từ chối (khoảng 28% danh mục) là nguồn lớn nhất của hạng E; nhóm này gần như vắng mặt trong tập huấn luyện vì chỉ khách được duyệt HĐ mới có nhãn.
Tuy nhiên ngay cả khách vừa ký HĐ cũng có tỷ trọng E cao hơn tập kiểm chứng, nên phân bố điểm tháng 6/2025 khác cuối 2024 thật sự. Dữ liệu chưa có kết quả trả nợ sau 06/2025 để kiểm chứng tỷ lệ xấu thực của từng hạng tại ngày chấm.
**Khuyến nghị:** giữ thứ hạng (mô hình đã chứng minh xếp hạng đúng trên dữ liệu ngoài thời gian), nhưng hiệu chỉnh lại ngưỡng hạng khi có 4 kỳ trả nợ của các HĐ ký sau 06/2025; theo dõi PSI hàng tháng theo bảng lịch sử bên dưới."""))
display(Markdown("**Hành động gợi ý theo hạng**"),
        pd.DataFrame({"hang": list(cfg["band_name"].values()), "hanh_dong": list(cfg["band_action"].values())}))
display(Markdown("**Lịch sử theo dõi**"), pd.read_csv(HERE / "out_team" / "pipeline" / "monitoring_history.csv"))
display(Markdown(f"**File kết quả:** `{cfg['out'] / '10_customer_financial_health.csv'}`"), res.head())
''')

md("## Tổng kết")
code(r'''
p = rep["performance"].set_index("Tap")
oot_row = p.iloc[-1]
display(Markdown(f"""
| Chỉ tiêu | Giá trị |
|---|---|
| Số biến trong mô hình | {len(M['feats'])} (từ 57 biến ứng viên) |
| AUC / Gini kiểm chứng 12/2024 | {oot_row.AUC:.3f} / {oot_row.Gini:.3f} (scorecard cũ: 0.661 / 0.322) |
| KS kiểm chứng | {oot_row.KS:.3f} |
| Tỷ lệ khách xấu nằm trong 10% điểm thấp nhất | {oot_row.Bat_xau_top10:.1%} |
| PSI điểm (huấn luyện → kiểm chứng) | {oot_row.PSI_so_voi_huan_luyen:.3f} |
| Tỷ lệ xấu tăng dần A→E | {rep['band_monotonic']} |
| Biến sai chiều nghiệp vụ | {int((rep['logic'].dung_chieu == 'KHÔNG ĐẠT').sum())} |
| Dữ liệu độc lập (CALL mã OWN*) | không phản ánh rủi ro (AUC theo lịch sử trễ {rep['independent_auc_late']:.3f}) → không dùng làm bằng chứng |
| Nhóm quan trọng nhất (bỏ đi thì AUC giảm nhiều nhất) | {rep['drop_group'].iloc[0].bo_nhom} |
"""))
''')

import eda_cells   # phần EDA (từ notebook Colab) làm căn cứ chọn chỉ tiêu
cells = eda_cells.insert_eda(cells)
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                                "language_info": {"name": "python"}})
nbf.write(nb, OUT)
print("Đã tạo", OUT)

if "--run" in sys.argv:
    from nbclient import NotebookClient
    nb = nbf.read(OUT, as_version=4)
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbf.write(nb, OUT)
    print("Đã chạy xong và lưu kết quả vào", OUT)
