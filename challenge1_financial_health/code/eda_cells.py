"""Các cell EDA (chuyển từ notebook Colab, mục 3 và 5.3) chèn vào notebook Challenge 1.
EDA_CELLS: đặt trong Câu 1, trước mục Feature engineering. RR_CELLS: đặt đầu Câu 4."""

EDA_CELLS = [
("md", """
### 1.2 EDA – các yếu tố ảnh hưởng tới sức khỏe tài chính (căn cứ chọn chỉ tiêu)
Trước khi tạo chỉ tiêu, tìm hiểu **điều gì phân biệt khách trả nợ tốt với khách gặp khó khăn**, rồi lấy đó làm căn cứ thiết kế chỉ tiêu ở mục 1.3.

- **Đơn vị phân tích:** từng **HĐ đã ký và đủ 4 kỳ thanh toán** (ký trước 01/02/2025), trên toàn bộ lịch sử.
- **Biến kết quả:** HĐ có trễ hạn ở bất kỳ kỳ nào trong 4 kỳ đầu, cùng định nghĩa "xấu" với nhãn huấn luyện.
- Mọi quan hệ dưới đây là **tương quan, không phải nhân quả**.

| # | Câu hỏi | Nhóm chỉ tiêu liên quan |
|---|---|---|
| 3.1 | Trễ hạn phân bố thế nào theo sản phẩm, theo kỳ, theo năm? | Bức tranh chung |
| 3.2 | Trạng thái HĐ (Finished, Paid off, Written off…) có dùng được không? | Rò rỉ thông tin |
| 3.3 | Trễ hạn trong quá khứ có lặp lại không? Gần đây hay đã lâu có khác không? | 4. Lịch sử thanh toán |
| 3.4 | Thành tích tốt, thâm niên dài có bảo vệ khách không? Vay dồn dập thì sao? | 6. Độ tin cậy tín dụng |
| 3.5 | Gánh nặng trả nợ và thu nhập ảnh hưởng thế nào? | 1. Thu nhập, 2. Thu–chi, 3. Gánh nặng nợ |
| 3.6 | Kỳ hạn, trả trước, bảo hiểm có liên quan không? | 5. Dự phòng, 7. Bảo vệ |
| 3.7 | Bị từ chối, hạn mức được chào có phản ánh rủi ro không? | 6. Áp lực vay & độ tin cậy |
| 3.8 | Nhân khẩu học có khác biệt không, và vì sao không dùng? | Công bằng |
"""),
("code", r'''
import eda_reasoning
from features import connect as _connect
con_eda = _connect()
E = eda_reasoning.contract_eda(con_eda)

def rate_bar(ax, d, title, base=None, fmt="{:.1%}", rot=0):
    """Cột = tỷ lệ trễ của từng nhóm; số nhỏ dưới chân cột = số HĐ/KH."""
    d = d.dropna(subset=["ty_le_tre"])
    ax.bar(d.nhom.astype(str), d.ty_le_tre * 100, color="#3b6fb6", width=.65)
    for i, (v, n) in enumerate(zip(d.ty_le_tre, d.so_hd)):
        ax.text(i, v * 100, fmt.format(v), ha="center", va="bottom", fontsize=7.5)
        ax.text(i, 0, f"n={n / 1e3:,.0f}K" if n >= 1e3 else f"n={n:,.0f}", ha="center", va="bottom", fontsize=6.5, color="white")
    if base is not None:
        ax.axhline(base * 100, color="k", ls="--", lw=.8)
    ax.set_title(title, fontsize=9); ax.set_ylabel("% trễ hạn"); ax.tick_params(axis="x", labelsize=7.5, rotation=rot)

BASE = float(E["tong_quan"].ty_le_tre[0])
display(E["tong_quan"].rename(columns={"so_hd": "số HĐ đủ 4 kỳ", "so_kh": "số khách", "ty_le_tre": "tỷ lệ HĐ trễ hạn"})
        .style.format({"số HĐ đủ 4 kỳ": "{:,.0f}", "số khách": "{:,.0f}", "tỷ lệ HĐ trễ hạn": "{:.2%}"}).hide(axis="index"))
'''),
("md", "#### 3.1 Bức tranh trễ hạn"),
("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
rate_bar(axes[0], E["san_pham"], "Tỷ lệ HĐ trễ hạn theo sản phẩm", BASE, fmt="{:.2%}")
k = E["theo_ky"].T.reset_index(); k.columns = ["nhom", "ty_le_tre"]; k["so_hd"] = E["tong_quan"].so_hd[0]
rate_bar(axes[1], k, "Tỷ lệ HĐ có cờ trễ ở từng kỳ", fmt="{:.2%}")
y = E["theo_nam"]
axes[2].plot(y.nhom.astype(int), y.ty_le_tre * 100, "o-", color="#3b6fb6")
for x_, v in zip(y.nhom, y.ty_le_tre): axes[2].text(x_, v * 100, f"{v:.1%}", ha="center", va="bottom", fontsize=7.5)
axes[2].set_title("Tỷ lệ HĐ trễ hạn theo năm ký", fontsize=9); axes[2].set_ylabel("% trễ hạn")
plt.tight_layout(); plt.show()
print(f"Trong các HĐ trễ kỳ 1, {E['d1_va_d4']:.1%} vẫn trễ ở kỳ 4")
'''),
("md", """
> **Insight 3.1 – bức tranh trễ hạn** (2.759.040 HĐ đủ 4 kỳ của 812.709 khách, tỷ lệ trễ 2,02%)
> - **Rủi ro khác nhau theo sản phẩm:** xe máy TW **4,27%**, CW 2,77%, thẻ CC 2,26%, BNPL 2,20%, điện máy CD 1,99%, vay tiền mặt CL **1,36%**. CL thấp nhất, có lẽ vì chỉ được duyệt cho khách đã có lịch sử tốt.
> - **Trễ hạn tích lũy dần qua các kỳ:** tỷ lệ HĐ có cờ trễ tăng từ 0,49% (kỳ 1) → 0,76% → 1,05% → **1,29% (kỳ 4)**. Một nửa (50,6%) HĐ trễ kỳ 1 vẫn trễ ở kỳ 4.
>   → *Căn cứ:* không chỉ đếm "có / không trễ" mà cần đo **mức độ và độ kéo dài**: `max_late_streak` (chuỗi trễ liên tiếp), `late_rate_last3`, `n_inst_late_6m`, `n_inst_late_12m`.
> - **Theo năm ký:** năm 2021 tăng vọt lên **3,3%** (thời kỳ Covid); 2024–2025 tăng trở lại (2,2–2,5%). Rủi ro phụ thuộc chu kỳ kinh tế.
>   → *Căn cứ:* mô hình phải được kiểm chứng **theo thời gian** (huấn luyện 5 mốc 2024, kiểm chứng 12/2024), không chỉ chia ngẫu nhiên.
"""),
("md", "#### 3.2 Trạng thái hợp đồng và vấn đề rò rỉ thông tin"),
("code", r'''
st = E["trang_thai"]
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.barh(st.nhom[::-1], st.ty_le_tre[::-1] * 100,
        color=["#f57c00" if s in ("Paid off", "Written off", "Sold") else "#3b6fb6" for s in st.nhom[::-1]])
for i, (v, n) in enumerate(zip(st.ty_le_tre[::-1], st.so_hd[::-1])):
    ax.text(v * 100, i, f" {v:.1%}  (n={n:,.0f})", va="center", fontsize=8)
ax.set_xlim(0, st.ty_le_tre.max() * 125); ax.set_title("Tỷ lệ HĐ từng trễ hạn theo trạng thái HIỆN TẠI (2025)", fontsize=9)
plt.tight_layout(); plt.show()
'''),
("md", """
> **Insight 3.2 – trạng thái hợp đồng là *kết quả*, không phải *nguyên nhân***
> - **Paid off: 82,1% từng trễ hạn**, Written off 53,5%, Sold 48,6%; trong khi Finished chỉ 1,35% và Active 2,2%. "Paid off" là tất toán **sau khi đã gặp sự cố**, không phải dấu hiệu tốt.
> - Đây là trạng thái **tại 2025**. Một HĐ cũ chuyển sang Written off *sau* mốc huấn luyện (2024) sẽ làm lộ thông tin tương lai cho mô hình.
>
> → *Căn cứ:* **không dùng trạng thái HĐ hiện tại làm chỉ tiêu.** Hai chỉ tiêu ứng viên có dùng trạng thái (`n_clean_finished`, `ever_writeoff`) **không có trong 35 biến của mô hình cuối** (bị loại ở bước kiểm tra trùng thông tin và chọn biến theo SHAP). Lịch sử trả nợ chỉ dựa trên cờ trễ của các HĐ đã đủ 4 kỳ **trước mốc**.
"""),
("md", "#### 3.3 Lịch sử trả nợ: trễ hạn có lặp lại không?\nXét các HĐ **từ HĐ thứ 2 trở đi** của mỗi khách: HĐ hiện tại có trễ không, tùy theo những gì đã xảy ra ở các HĐ **trước đó**."),
("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
rate_bar(axes[0], E["so_hd_tre_truoc"], "Theo số HĐ đã trễ trước đó")
rate_bar(axes[1], E["lan_tre_gan_nhat"], "Theo thời gian từ lần trễ gần nhất")
rate_bar(axes[2], E["muc_do_hd_truoc"], "Theo số kỳ trễ của HĐ liền trước")
plt.tight_layout(); plt.show()
'''),
("md", """
> **Insight 3.3 – trễ hạn có tính lặp lại (yếu tố tiêu cực mạnh nhất)**
>
> | Số HĐ đã trễ trước đó | 0 | 1 | 2 | ≥ 3 |
> |---|---|---|---|---|
> | Tỷ lệ trễ HĐ hiện tại | 1,4% | **10,2%** | **18,6%** | **27,2%** |
>
> - Chỉ một lần trễ đã làm rủi ro tăng khoảng **7 lần**.
> - **Độ gần đây quyết định mức nguy hiểm:** lần trễ gần nhất ≤ 6 tháng → **36,9%**; 6–12 tháng 12,5%; 1–2 năm 9,0%; 2–4 năm 5,1%; > 4 năm 4,3%. Rủi ro "phai dần" nhưng vẫn cao hơn khách chưa từng trễ (1,4%).
> - **Mức độ:** HĐ liền trước trễ 1 kỳ → 13,0%; trễ cả 4 kỳ → **43,7%**.
>
> → *Căn cứ:* nhóm **4. Lịch sử thanh toán** cần đủ 3 khía cạnh:
> - **số lần / tỷ lệ:** `n_due_total`, `on_time_rate`, `n_on_time`;
> - **độ gần đây:** `days_since_overdue`, `n_inst_late_6m`, `n_inst_late_12m`, `late_rate_last3`;
> - **mức độ:** `max_late_streak`, `n_consec2_late`.
>
> Các quan hệ đều **đơn điệu** → áp ràng buộc đơn điệu trong LightGBM (`on_time_rate` −1, `n_inst_late_*` +1, `days_since_overdue` −1…).
"""),
("md", "#### 3.4 Độ bền: thành tích tốt và thâm niên có bảo vệ khách không?"),
("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
rate_bar(axes[0], E["so_hd_tot_truoc"], "Theo số HĐ trước đó trả tốt (khách chưa từng trễ)", BASE)
rate_bar(axes[1], E["tham_nien"], "Theo thâm niên tín dụng khi ký", BASE)
rate_bar(axes[2], E["nhip_vay"], "Theo khoảng cách từ HĐ trước (tháng)", BASE, rot=20)
plt.tight_layout(); plt.show()
'''),
("md", """
> **Insight 3.4 – độ bền và nhịp vay (yếu tố tích cực mạnh nhất)**
> - **Thành tích trả tốt bảo vệ rõ ràng** (khách chưa từng trễ): HĐ đầu tiên 3,0% → sau 1 HĐ tốt 1,9% → 2 HĐ 1,5% → 3–5 HĐ 1,1% → **≥ 6 HĐ 0,7%**.
> - **Thâm niên:** HĐ đầu 3,0%, giảm còn khoảng 1,3–1,5% khi khách có từ 3 năm quan hệ tín dụng.
> - **Nhịp vay hình chữ U:** vay **dồn dập** (< 1 tháng sau HĐ trước) 2,3%; quay lại sau 6–12 tháng thấp nhất 1,3%; **quá lâu mới quay lại** (> 24 tháng) tăng lên 1,8%.
>
> → *Căn cứ:*
> - độ bền: `n_on_time`, `n_signed`, `history_years`;
> - nhịp vay: `days_since_last_signed`, `days_since_last_apply` — **không ép chiều** vì quan hệ hình chữ U;
> - khách chưa có HĐ nào đủ 4 kỳ là nhóm rủi ro cao nhất và ít bằng chứng nhất → tách thành hạng **N · Chưa đủ lịch sử trả nợ**.
"""),
("md", "#### 3.5 Gánh nặng trả nợ và thu nhập\n`PTI` (payment-to-income) = tiền góp hàng tháng của HĐ ÷ thu nhập khai báo **khi nộp hồ sơ**."),
("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
rate_bar(axes[0], E["pti"], "Theo PTI của chính HĐ (trả góp)", BASE)
rate_bar(axes[1], E["thu_nhap"], "Theo thu nhập khai báo", BASE)
rate_bar(axes[2], E["so_tien_vay"], "Theo số tiền vay", BASE)
plt.tight_layout(); plt.show()
'''),
("md", """
> **Insight 3.5 – gánh nặng trả nợ và thu nhập**
> - **PTI:** đến 35% gần như không đổi (1,9–2,1%), nhưng **35–50% → 2,8%** và **> 50% → 4,2%** (gấp đôi). Gánh nặng nợ chỉ thành rủi ro khi **vượt ngưỡng** → quan hệ phi tuyến.
> - **Thu nhập khai báo:** tác động nhẹ, từ 2,5% (≤ 5 triệu) xuống 1,8% (> 20 triệu). Nhóm **không khai thu nhập** rủi ro gấp đôi (4,3%).
> - **Số tiền vay:** gần như không ảnh hưởng (1,9–2,2% ở mọi mức).
>
> → *Căn cứ:*
> - đo gánh nặng **cộng dồn** của các HĐ đang chạy: `active_annuity_income` (tiền góp/thu nhập), `active_credit_income`, `residual_income` (thu nhập còn lại), `n_active_revolving`;
> - **không ép chiều** (có ngưỡng); `income` giữ giá trị thiếu để mô hình học tín hiệu "không khai thu nhập";
> - độ ổn định thu nhập: `inc_cv`, `inc_vs_peak`, `inc_growth`, `n_income_types`, `income_years_declared`.
>
> **Giới hạn:** ở cấp khách, gánh nặng nợ cao lại đi cùng rủi ro *thấp* hơn (xem RR ở Câu 4) vì khách được duyệt vay nhiều là khách đã qua thẩm định — tương quan do chọn lọc. Đây là lý do nhóm 1–3 đóng góp ít khi bỏ khỏi mô hình (bước 9d).
"""),
("md", "#### 3.6 Đặc điểm khoản vay: kỳ hạn, trả trước, bảo hiểm"),
("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
rate_bar(axes[0], E["ky_han"], "Theo kỳ hạn (số kỳ, trả góp)", BASE)
rate_bar(axes[1], E["tra_truoc"], "Theo tỷ lệ trả trước (CD & TW)", BASE)
b = E["bao_hiem"]; x = np.arange(len(b)); w = .38
axes[2].bar(x - w / 2, b.khong_bao_hiem * 100, w, label="Không bảo hiểm", color="#3b6fb6")
axes[2].bar(x + w / 2, b.co_bao_hiem * 100, w, label="Có bảo hiểm", color="#f57c00")
axes[2].set_xticks(x); axes[2].set_xticklabels(b.san_pham); axes[2].legend(fontsize=8)
axes[2].set_title("Tỷ lệ trễ: có / không mua bảo hiểm (theo sản phẩm)", fontsize=9); axes[2].set_ylabel("% trễ hạn")
plt.tight_layout(); plt.show()
display(b.style.format({"khong_bao_hiem": "{:.2%}", "co_bao_hiem": "{:.2%}"}).hide(axis="index"))
'''),
("md", """
> **Insight 3.6 – đặc điểm khoản vay**
> - **Kỳ hạn:** không có xu hướng rõ (1,7–2,6%).
> - **Tỷ lệ trả trước (CD & TW):** trả trước càng nhiều càng an toàn: ≤ 20% → **2,8%**, > 60% → **1,2%**. Nhóm 0% thấp (1,5%), nhiều khả năng do chương trình trả góp 0% chỉ dành cho khách tốt.
>   → *Căn cứ:* nhóm **5. Dự phòng tài chính** (khả năng trả trước bằng tiền mặt là dấu hiệu có tích lũy): `down_payment_ratio`, `init_pay_income`, `avg_init_pay`, `max_init_pay_24m`. Nhóm này đóng góp 14,7% tổng |SHAP|, lớn thứ hai trong mô hình.
> - **Bảo hiểm khoản vay:** trong *cùng một sản phẩm*, HĐ có bảo hiểm lại trễ **nhiều hơn** (TW 4,5% so với 3,2%; CL 1,5% so với 1,0%; CW 3,2% so với 1,7%), có thể do **tự chọn lọc ngược** (khách tự biết mình rủi ro thì mua bảo hiểm). Nhưng ở cấp khách, khách mua bảo hiểm nhiều lại rủi ro thấp hơn (RR 0,72 ở Câu 4) vì bảo hiểm phổ biến ở sản phẩm an toàn.
>   → *Căn cứ:* `pct_insured2` (nhóm **7. Bảo vệ**) **không ép chiều**, để mô hình tự học quan hệ có điều kiện.
"""),
("md", "#### 3.7 Mức độ được duyệt và đánh giá của bên cho vay (cấp khách hàng)\nGộp về **cấp khách**: khách có HĐ đủ 4 kỳ được xem là \"có vấn đề\" nếu **ít nhất một** HĐ từng trễ."),
("code", r'''
BASE_C = float(E["kh_tong_quan"].ty_le_kh_tung_tre[0])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
rate_bar(axes[0], E["ty_le_tu_choi"], "% khách từng trễ – theo tỷ lệ hồ sơ bị từ chối", BASE_C)
rate_bar(axes[1], E["max_offer"], "% khách từng trễ – theo hạn mức được chào gần nhất", BASE_C, rot=15)
plt.tight_layout(); plt.show()
print(f"Khách có HĐ đủ 4 kỳ: {int(E['kh_tong_quan'].so_kh[0]):,} | tỷ lệ khách từng trễ: {BASE_C:.2%}")
'''),
("md", """
> **Insight 3.7 – được duyệt và đánh giá của bên cho vay** (812.709 khách, 6,19% từng trễ)
>
> | Tỷ lệ hồ sơ bị từ chối | 0% | ≤ 25% | 25–50% | 50–75% | > 75% |
> |---|---|---|---|---|---|
> | % khách từng trễ | 4,1% | 6,8% | 10,2% | 16,5% | **23,1%** |
>
> - **Bị từ chối là tín hiệu rủi ro rất mạnh**: bên cho vay từng từ chối thường có lý do.
> - **Hạn mức bên cho vay chào (MAX_OFFER)** phản ánh đánh giá năng lực của chính công ty: ≤ 5 triệu → 7,0%; > 50 triệu → **2,0%**; **chưa từng được chào → 10,6%**.
>
> → *Căn cứ:* nhóm **6. Áp lực vay & độ tin cậy**: `refused_ratio` (+1), `last_app_rejected` (+1), `n_refused_12m`, `max_offer` (−1, giữ thiếu = chưa được chào), `max_offer_12m`, `offer_change_6m`, `n_offers_12m`. Đây là nhóm quan trọng nhất của mô hình (57,7% tổng |SHAP|; bỏ nhóm này AUC giảm 0,020).
"""),
("md", "#### 3.8 Nhân khẩu học"),
("code", r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, (k, t) in zip(axes, [("nhan_khau_gender", "giới tính"), ("nhan_khau_age", "nhóm tuổi"), ("nhan_khau_region", "vùng miền")]):
    d = E[k][E[k].so_hd > 1000]
    rate_bar(ax, d, f"% khách từng trễ – theo {t}", BASE_C)
plt.tight_layout(); plt.show()
'''),
("md", """
> **Insight 3.8 – nhân khẩu học**
> - Có chênh lệch: **nam 7,2%** so với nữ 5,2%; tuổi 55+ là 7,2%, 25–34 là 5,5%; vùng miền gần như không khác (6,0–6,3%).
> - Đề bài yêu cầu *không dùng nhân khẩu học để xác định rủi ro khi không có đủ bằng chứng*. Các chênh lệch này có thể đến từ **hành vi** (sản phẩm vay, lịch sử) chứ không phải từ chính giới tính hay tuổi.
>
> → *Căn cứ:* **loại** giới tính, tuổi, vùng miền, học vấn, số con khỏi chỉ tiêu.
"""),
("md", """
#### 3.9 Từ insight đến chỉ tiêu

| # | Insight (bằng chứng) | Chỉ tiêu trong mô hình | Chiều / xử lý |
|---|---|---|---|
| 1 | Trễ lặp lại: 1 HĐ trễ trước đó → rủi ro ×7; ≥ 3 HĐ → 27% | `n_due_total`, `on_time_rate`, `n_on_time` | ↑ / ↓, ràng buộc đơn điệu |
| 2 | Trễ ≤ 6 tháng → 37%; > 4 năm → 4% | `days_since_overdue`, `n_inst_late_6m`, `n_inst_late_12m`, `late_rate_last3` | ↓ / ↑ |
| 3 | Mức độ: HĐ trước trễ 1 kỳ → 13%, 4 kỳ → 44%; một nửa HĐ trễ kỳ 1 vẫn trễ kỳ 4 | `max_late_streak`, `n_consec2_late` | ↑ |
| 4 | Trạng thái HĐ là kết quả, gây rò rỉ (Paid off 82% từng trễ) | **Không dùng trạng thái** trong mô hình cuối | – |
| 5 | Mỗi HĐ trả tốt giảm rủi ro (3,0% → 0,7%); thâm niên dài an toàn hơn | `n_on_time`, `n_signed`, `history_years` | ↓ |
| 6 | Nhịp vay hình chữ U | `days_since_last_signed`, `days_since_last_apply`, `n_apply_6m` | không ép chiều |
| 7 | Bị từ chối 0% → 4,1%; > 75% → 23,1% | `refused_ratio`, `last_app_rejected`, `n_refused_12m` | ↑ |
| 8 | PTI > 50% → rủi ro gấp đôi (có ngưỡng) | `active_annuity_income`, `active_credit_income`, `residual_income`, `n_active_revolving` | không ép chiều |
| 9 | Không khai thu nhập → 4,3%; thu nhập cao an toàn hơn nhẹ | `income` (giữ thiếu), `inc_cv`, `inc_vs_peak`, `inc_growth`, `n_income_types` | không ép chiều |
| 10 | MAX_OFFER > 50 triệu → 2,0%; chưa được chào → 10,6% | `max_offer`, `max_offer_12m`, `offer_change_6m`, `n_offers_12m` | ↓ |
| 11 | Trả trước càng nhiều càng an toàn (≤ 20% → 2,8%; > 60% → 1,2%) | `down_payment_ratio`, `init_pay_income`, `avg_init_pay`, `max_init_pay_24m` | không ép chiều (nhóm 0% đặc biệt) |
| 12 | Bảo hiểm: ngược chiều giữa cấp HĐ và cấp khách | `pct_insured2` | không ép chiều |
| 13 | HĐ đầu tiên (chưa có lịch sử) rủi ro 3,0%, cao nhất | hạng **N · Chưa đủ lịch sử trả nợ** | tách riêng |
| 14 | Nhân khẩu học có chênh lệch nhưng không được phép dùng | **Loại** | – |
| 15 | Rủi ro thay đổi theo năm (2021, 2024–2025) | kiểm chứng ngoài thời gian 12/2024 | – |

Các insight trên được **kiểm định lại trên đúng quần thể huấn luyện** ở mục 1.4 (AUC đơn biến, tỷ lệ xấu theo nhóm giá trị) và bằng relative risk ở Câu 4.
"""),
]

RR_CELLS = [
("md", """
### 4.1 Relative risk từ dữ liệu (độc lập với mô hình)
RR = tỷ lệ xấu của nhóm *có* yếu tố ÷ tỷ lệ xấu của nhóm *không có*, tính trên đúng quần thể huấn luyện (chỉ tiêu tại mốc, nhãn sau mốc). RR > 1 làm **xấu** sức khỏe tài chính, RR < 1 làm **tốt**. Kết quả này sau đó được đối chiếu với SHAP của mô hình (4.2).
"""),
("code", r'''
rr = eda_reasoning.relative_risk(dev)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(rr.index, np.log2(rr.RR), color=["#c62828" if v > 1 else "#2e7d32" for v in rr.RR], height=.65)
for i, v in enumerate(rr.RR):
    ax.text(np.log2(v), i, f"  ×{v:.2f}" if v > 1 else f"×{v:.2f}  ", va="center", ha="left" if v > 1 else "right", fontsize=8)
ax.axvline(0, color="k", lw=.8)
ticks = [.25, .5, 1, 2, 4, 8]
ax.set_xticks(np.log2(ticks)); ax.set_xticklabels([f"×{t:g}" for t in ticks])
lim = np.abs(np.log2(rr.RR)).max() + .9; ax.set_xlim(-lim, lim)
ax.set_title("Relative risk trễ hạn ở HĐ mới · đỏ = làm XẤU sức khỏe tài chính · xanh = làm TỐT", fontsize=9)
plt.tight_layout(); plt.show()
display(rr.style.format({"so_kh": "{:,.0f}", "ty_trong": "{:.1%}", "xau_khi_co": "{:.2%}", "xau_khi_khong": "{:.2%}", "RR": "{:.2f}"}))
'''),
("md", """
> **Nhận xét – yếu tố tích cực / tiêu cực theo relative risk**
> - **Làm XẤU (RR > 1):** trễ hạn trong 12 tháng gần nhất **×11,9**; trễ 2 kỳ liên tiếp ×6,7; từng có kỳ trễ ×4,3; ≥ 3 hồ sơ trong 3 tháng ×2,9; bị từ chối ≥ 50% hồ sơ ×2,5; hồ sơ gần nhất bị từ chối ×2,3; hồ sơ mỏng ×2,0.
> - **Làm TỐT (RR < 1):** hạn mức được chào > 20 triệu **×0,27**; ≥ 2 HĐ tất toán không trễ ×0,45; trả trước ≥ 30% ×0,54; lịch sử vay ≥ 3 năm ×0,56; mua bảo hiểm ≥ 50% HĐ ×0,72; thu nhập > 12 triệu ×0,76.
> - **Cách đọc:** tín hiệu tiêu cực mạnh nhất đều là **hành vi trả nợ đã xảy ra** và **bị bên cho vay từ chối**; tín hiệu tích cực mạnh nhất là **đánh giá tốt của bên cho vay** và **lịch sử tín dụng tốt đã được chứng minh**. "Tiền góp > 50% thu nhập" có RR 0,66 chủ yếu do **chọn lọc** (khách được duyệt vay nhiều là khách được đánh giá tốt), không nên diễn giải theo nghĩa nhân quả.
> - Các chiều này **trùng khớp** với chiều tác động trong SHAP bên dưới: hai phương pháp độc lập cho cùng kết luận.
"""),
]


def _mk(kind, src):
    import nbformat as nbf
    return nbf.v4.new_markdown_cell(src.strip("\n")) if kind == "md" else nbf.v4.new_code_cell(src.strip("\n"))


def insert_eda(cells, eda=None, rr=None):
    """Chèn EDA vào danh sách cell của notebook Challenge 1 (đã chạy hoặc chưa).
    eda / rr: danh sách cell nbformat (đã thực thi nếu có); mặc định tạo mới từ EDA_CELLS / RR_CELLS."""
    eda = eda or [_mk(k, s) for k, s in EDA_CELLS]
    rr = rr or [_mk(k, s) for k, s in RR_CELLS]
    find = lambda prefix: next(i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.lstrip().startswith(prefix))
    # đánh số lại mục Câu 1
    i_fe = find("### 1.2 Feature engineering")
    cells[i_fe].source = cells[i_fe].source.replace("### 1.2 Feature engineering", "### 1.3 Feature engineering", 1)
    i_uni = find("### 1.3 Phân tích từng chỉ tiêu")
    cells[i_uni].source = cells[i_uni].source.replace("### 1.3 Phân tích từng chỉ tiêu", "### 1.4 Phân tích từng chỉ tiêu", 1)
    cells[i_fe:i_fe] = eda
    # Câu 4: relative risk trước SHAP
    i4 = find("# Câu 4")
    cells[i4].source = cells[i4].source.replace(
        "# Câu 4 · Yếu tố tác động tích cực / tiêu cực đến sức khỏe tài chính\n### Giải thích toàn cục bằng SHAP",
        "### 4.2 Giải thích toàn cục bằng SHAP (mô hình)", 1)
    rr[0].source = "# Câu 4 · Yếu tố tác động tích cực / tiêu cực đến sức khỏe tài chính\n" + rr[0].source
    cells[i4:i4] = rr
    # bảng mục lục ở đầu notebook
    cells[0].source = cells[0].source.replace(
        "| 1–5: cấu hình, kiểm tra dữ liệu, snapshot, **mô tả biến đầu vào và feature engineering**, kiểm tra biến, phân tích từng biến |",
        "| 1–5: cấu hình, kiểm tra dữ liệu, **EDA các yếu tố ảnh hưởng → căn cứ chọn chỉ tiêu**, **mô tả biến đầu vào và feature engineering**, kiểm tra biến, phân tích từng biến |").replace(
        "| SHAP toàn cục, 8 giải thích từng khách |", "| relative risk từ dữ liệu, SHAP toàn cục, 8 giải thích từng khách |")
    return cells
