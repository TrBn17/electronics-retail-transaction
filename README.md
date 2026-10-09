# Dự báo lượng bán theo Cửa hàng × Nhóm hàng × Ngày

Dự báo **số lượng bán mỗi ngày** cho từng cặp cửa hàng × nhóm hàng của chuỗi bán lẻ điện tử Genbyte, so sánh 3 baseline và 5 mô hình học máy.

| Dữ liệu | Giá trị |
|---|---|
| Nguồn | Bảng kê bán hàng Genbyte, toàn công ty |
| Thời gian | 01/01/2025 → 31/03/2025 (90 ngày, có Tết Ất Tỵ 29/01) |
| Quy mô | 216.995 dòng giao dịch · 130 cửa hàng · 15 khu vực · 23 nhóm hàng · 3.557 vật tư |
| Bảng mô hình | 2.166 cặp cửa hàng × nhóm hàng × 90 ngày = **194.940 dòng** |
| Biến mục tiêu | `so_luong` — 70,8% ô bằng 0, phương sai/trung bình = 45,5 (rất thưa, quá phân tán) |

---

## Mục lục

- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cách chạy](#cách-chạy)
- [Phân công](#phân-công)
- [Chuẩn đầu ra mỗi mô hình](#chuẩn-đầu-ra-mỗi-mô-hình)
- [Nộp kết quả vào báo cáo](#nộp-kết-quả-vào-báo-cáo)
- [Checklist tiến độ](#checklist-tiến-độ)
- [Kết quả hiện có](#kết-quả-hiện-có) → **[bao_cao/BAO_CAO.md](bao_cao/BAO_CAO.md)**

---

## Cấu trúc thư mục

```text
data_eda/
├── data/                                # dữ liệu gốc — KHÔNG đưa lên Git
│   └── electronics-retail-transaction-q1-2025.xlsx
├── module3/                             # Module 3 — Dữ liệu, tiền xử lý, Random Forest
│   ├── module3_preprocess.py            # code dùng chung cho cả 5 module
│   ├── Module3_1_tien_xu_ly.ipynb       # làm sạch → bảng mô hình
│   ├── Module3_2_eda.ipynb              # mô tả nhóm hàng, cửa hàng, thời gian, độ phủ
│   └── Module3_3_random_forest.ipynb    # mô hình Random Forest
├── module5/                             # Module 5 — Tổng hợp, CatBoost
│   ├── Module5_catboost.ipynb           # mô hình CatBoost
│   └── tao_bao_cao.py                   # cập nhật bản báo cáo tổng hợp từ bao_cao/du_lieu/
├── bao_cao/                             # ★ bản báo cáo kết quả — CÓ đưa lên Git (chỉ số liệu đã tổng hợp)
│   ├── BAO_CAO.md                       # so sánh, giải thích, kết luận — đọc ở đây
│   ├── du_lieu/<mô hình>/               # kết quả đã lọc mỗi mô hình nộp (nop_ket_qua)
│   └── hinh/                            # biểu đồ sinh tự động
├── outputs/                             # kết quả đầy đủ khi chạy — KHÔNG đưa lên Git
│   ├── daily_store_category.csv         # ★ bảng mô hình dùng chung
│   └── rf/ · catboost/                  # kết quả từng mô hình
├── .env.example                         # mẫu cấu hình → copy thành .env
├── CLAUDE.md                            # quy ước dự án: dữ liệu, so sánh mô hình, sửa code chung
└── README.md
```

## Cách chạy

1. Copy `.env.example` thành `.env`. Nếu chưa có file dữ liệu trong `data/`, xin `DRIVE_ID` trong nhóm và điền vào `.env` để module tự tải từ Google Drive.
   > ⚠️ Repo public và dữ liệu có tên khách hàng/nhân viên: **không** commit `.env`, file dữ liệu hay `DRIVE_ID`.
2. Cài thư viện: `pip install pandas numpy openpyxl matplotlib scikit-learn joblib requests catboost`.
3. Chạy `module3/Module3_1_tien_xu_ly` trước (tạo bảng mô hình), sau đó các notebook còn lại theo thứ tự bất kỳ: `Module3_2_eda`, `Module3_3_random_forest`, `module5/Module5_catboost`.
4. Xong một mô hình: [nộp kết quả vào báo cáo](#nộp-kết-quả-vào-báo-cáo) → `python module5/tao_bao_cao.py`.

| Biến trong `.env` | Mặc định | Ý nghĩa |
|---|---|---|
| `DATA_FILE` | `data/electronics-retail-transaction-q1-2025.xlsx` | File dữ liệu gốc (tính từ gốc repo) |
| `DRIVE_ID` | — | ID file trên Google Drive để tự tải (chỉ để trong `.env`) |
| `OUT_DIR` | `outputs` | Thư mục ghi kết quả (tính từ gốc repo) |
| `KHU_VUC` | trống = toàn công ty | Chạy riêng một khu vực, vd `KVHN1` |

**Các module khác dùng lại tiền xử lý của Module 3:**

```python
import sys; sys.path.append("../module3")
from module3_preprocess import doc_bang_ngay, tao_dac_trung, chia_tap, danh_gia, bang_baseline, loi_theo_nhom

daily = doc_bang_ngay()              # đọc outputs/daily_store_category.csv (chưa có thì tự dựng)
d, FEATS = tao_dac_trung(daily)      # lag 1/7/14, rolling 7/14, lịch, mã cửa hàng/nhóm hàng/vùng
fit, val, tr, te = chia_tap(d)       # mốc chia chung: tối ưu trên fit/val, train lại trên tr, chấm trên te
rows = bang_baseline(te)             # Naive, Seasonal Naive, MA7
rows["Mô hình X"] = danh_gia(te["so_luong"], du_bao)   # MAE, RMSE, WAPE, R², Bias
```

**Google Colab:** upload `module3_preprocess.py` và `.env` cùng thư mục với notebook rồi Runtime → Run all; cuối mỗi notebook có ô nén và tải `outputs/` về.

---

## Phân công

| Module | Nội dung | Mô hình | Phụ trách |
|---|---|---|---|
| **1** · Bối cảnh và Baseline | I. Introduction · IV. Naive, Seasonal Naive, MA7 · V. Kết quả baseline | Ridge Regression | Nguyên |
| **2** · Tổng quan nghiên cứu | II. Literature Review · IV. Problem formulation, metrics · References | Poisson Regression | Phuong |
| **3** · Dữ liệu và phân tích cửa hàng | III. Data Analysis · IV. Tiền xử lý, lọc, tổng hợp · V. Kết quả theo nhóm hàng/cửa hàng | Random Forest | Phúc |
| **4** · Đặc trưng và thiết kế thí nghiệm | IV. Lag, rolling, lịch, leakage · Train–Validation–Test, tuning · V. Phân tích lỗi | XGBoost | Sơn |
| **5** · Tổng hợp và giá trị thực tế | V. So sánh 5 mô hình · Business implications, limitations · VI. Conclusion | CatBoost | Ngọc |

Mỗi module đều phải: train → tối ưu → lưu kết quả → viết phần tương ứng cho mô hình của mình, theo [Chuẩn đầu ra mỗi mô hình](#chuẩn-đầu-ra-mỗi-mô-hình).

---

## Chuẩn đầu ra mỗi mô hình

Mọi mô hình sinh **cùng bộ file và cùng khung mục kết quả**, để bản báo cáo tổng hợp tự gộp mà không phải viết code riêng cho từng mô hình. Mẫu: `module3/Module3_3_random_forest.ipynb` và `module5/Module5_catboost.ipynb`.

**Tên gọi** (cũng khai báo trong code ở `MO_HINH` của `module3_preprocess.py` — sửa thì sửa cả hai nơi)

| Module | Mô hình | Notebook | Thư mục | Tiền tố `<p>` | Cột dự báo | Tên dòng trong `<p>_metrics.csv` |
|---|---|---|---|---|---|---|
| 1 | Ridge | `module1/Module1_ridge.ipynb` | `outputs/ridge/` | `ridge` | `du_bao_ridge` | `Ridge` |
| 2 | Poisson | `module2/Module2_poisson.ipynb` | `outputs/poisson/` | `poisson` | `du_bao_poisson` | `Poisson` |
| 3 | Random Forest | `module3/Module3_3_random_forest.ipynb` | `outputs/rf/` | `rf` | `du_bao_rf` | `Random Forest` |
| 4 | XGBoost | `module4/Module4_xgboost.ipynb` | `outputs/xgb/` | `xgb` | `du_bao_xgb` | `XGBoost` |
| 5 | CatBoost | `module5/Module5_catboost.ipynb` | `outputs/catboost/` | `cb` | `du_bao_cb` | `CatBoost` |

**File bắt buộc** (CSV ghi `encoding="utf-8-sig"`)

| File | Nội dung |
|---|---|
| `<p>_tuning.csv` | Mỗi cấu hình đã thử 1 dòng, sắp theo MAE Validation: `id`, các tham số, `MAE, RMSE, WAPE, R2, Bias` (từ `danh_gia` trên Validation), `giây` |
| `<p>_best_params.json` | Cấu hình được chọn, đủ để dựng lại mô hình (boosting có early stopping: ghi cả số vòng) |
| `<p>_metrics.csv` | Kết quả Test. Dòng: `Naive (lag 1)`, `Seasonal Naive (lag 7)`, `MA7`, tên mô hình. Cột: `MAE, RMSE, WAPE, R2, Bias, WAPE so với MA7` |
| `<p>_predictions.csv` | Dự báo Test: `Mã cửa hàng, Nhóm hàng, Khu vực, Vùng, Ngày, so_luong, du_bao_<p>, lag_1, lag_7, tb_7` |
| `<p>_model.*` | Mô hình đã train lại trên 15/01–28/02 (`.joblib`, `.cbm`, `.json`…) |
| `<p>_feature_importance.csv` + `.png` | Chỉ số là `FEATS`, cột `importance` (Ridge/Poisson: trị tuyệt đối hệ số trên đặc trưng đã chuẩn hoá) |
| `<p>_error_by_category.csv`, `<p>_error_by_store.csv` | `loi_theo_nhom(te, "du_bao_<p>", "Nhóm hàng")` và `loi_theo_nhom(te, "du_bao_<p>", ["Mã cửa hàng", "Khu vực"])` |
| `<p>_thuc_te_vs_du_bao.png` | Tổng số lượng mỗi ngày tháng 3: thực tế, mô hình, MA7 |

**Các mục trong notebook** — đúng thứ tự, đúng tiêu đề:

1. Ô markdown đầu: đầu vào, đầu ra, cách chạy trên Colab
2. `## 1. Đặc trưng và chia tập` — `tao_dac_trung`, `chia_tap`, in `bang_baseline(val)` và `bang_baseline(te)`
3. `## 2. Tối ưu tham số trên Validation` — chọn theo MAE Validation → `<p>_tuning.csv`, `<p>_best_params.json`
4. `## 3. Train lại trên toàn bộ Train và chấm Test` — cắt dự báo âm về 0 → `<p>_metrics.csv`, mô hình
5. `## 4. Đặc trưng quan trọng và lỗi theo nhóm hàng / cửa hàng / ngày` — các file còn lại; in số nhóm hàng / cửa hàng có WAPE thấp hơn MA7
6. `## 5. Kết quả và thảo luận` — viết tay từ file đã lưu, theo khung dưới
7. `## Nộp kết quả vào báo cáo và tải về (Colab)` — `nop_ket_qua("<thư mục>", val)` rồi `tai_ve(...)` (xem [Nộp kết quả vào báo cáo](#nộp-kết-quả-vào-báo-cáo))

**Khung mục 5** (copy rồi điền số từ `outputs/<thư mục>/`):

```markdown
## 5. Kết quả và thảo luận

*Số liệu từ lần chạy ngày dd/mm/yyyy trên bảng mô hình <chưa lọc | bộ lọc đã chốt>.*

**Tối ưu tham số.** Cấu hình tốt nhất: `...` — MAE Validation x,xxx so với MA7 x,xxx.
<Tham số nào ảnh hưởng nhiều nhất. Bảng 3 cấu hình tốt nhất (MAE · RMSE · WAPE · Bias trên Validation) + dòng MA7 tham chiếu.>

**Kết quả trên Test (tháng 3, <số> ô):**

| Mô hình | MAE | RMSE | WAPE | R² | Bias |
|---|---:|---:|---:|---:|---:|
| Naive (lag 1) | | | | | |
| Seasonal Naive (lag 7) | | | | | |
| MA7 | | | | | |
| **<Tên mô hình>** | | | | | |

**Thảo luận.**
1. So với MA7 trên từng chỉ số — thắng hay thua, chênh bao nhiêu.
2. Tổng lượng — Bias; trung bình dự báo/ngày toàn hệ thống so với thực tế và MA7; tổng dự báo ở các ô bán ≥ 20 sản phẩm/ngày.
3. Theo nhóm — WAPE thấp hơn MA7 ở x/23 nhóm hàng, y/130 cửa hàng; nêu Điện thoại, Phụ kiện, Hcare.
4. Đặc trưng quan trọng nhất (kèm %).
5. So với các mô hình đã có kết quả; hàm ý cho nhóm.

**Giới hạn:** ...
```

Giữ đúng tiêu đề `## 5. Kết quả và thảo luận` và nhãn `**Thảo luận.**` — từ nhãn này đến hết ô được tự trích vào mục 10 của báo cáo tổng hợp.

**Quy ước số:** 3 chữ số thập phân, dấu phẩy thập phân, dấu chấm hàng nghìn (`0,757`, `2.393`); Bias là % có dấu (`+0,8%`, `−41,9%`); in đậm giá trị tốt nhất mỗi cột; mọi số lấy từ `outputs/`, không ước lượng. **R² giữ nguyên như `danh_gia` tính** — không cắt về 0, không bỏ cột (giải thích R² âm: mục 9 của [báo cáo](bao_cao/BAO_CAO.md)). Xong mô hình thì [nộp vào báo cáo](#nộp-kết-quả-vào-báo-cáo) và đánh dấu checklist.

---

## Nộp kết quả vào báo cáo

**[`bao_cao/BAO_CAO.md`](bao_cao/BAO_CAO.md) là bản báo cáo kết quả duy nhất của nhóm** — so sánh 3 baseline và 5 mô hình, giải thích, chọn mô hình, kết luận. Bảng, biểu đồ và nhận định trong đó được `module5/tao_bao_cao.py` sinh lại từ kết quả các module nộp; mỗi module nộp xong là báo cáo tự có thêm mô hình đó.

**Các bước khi xong một mô hình**

1. Notebook đúng [chuẩn đầu ra](#chuẩn-đầu-ra-mỗi-mô-hình), chạy hết, đã viết mục 5.
2. Ô cuối notebook gọi `nop_ket_qua("<thư mục>", val)` (vd `nop_ket_qua("xgb", val)`): lọc kết quả cần cho báo cáo từ `outputs/<thư mục>/` sang `bao_cao/du_lieu/<thư mục>/`.
   - **Chạy trên Colab:** hàm tự tải về `bao_cao_<thư mục>.zip` → giải nén vào `bao_cao/du_lieu/<thư mục>/` trong repo trên máy.
3. Ở gốc repo: `python module5/tao_bao_cao.py` → cập nhật bảng, biểu đồ, nhận định trong `bao_cao/BAO_CAO.md` (không cần dữ liệu gốc).
4. Mở `bao_cao/BAO_CAO.md` kiểm tra:
   - dòng **Kiểm tra so sánh công bằng** phải là ✔ — nếu ⚠️ (khác bảng mô hình, khác tập Test) thì chạy lại theo bảng chung, **không** commit kết quả lệch;
   - mục **Tóm tắt** và **Chọn mô hình**: nếu kết luận thay đổi, báo Module 5 cập nhật mục 11 (phần viết tay).
5. Commit notebook + `bao_cao/` (không commit `outputs/`).

**Kết quả nào được đưa vào báo cáo**

| Kết quả của mô hình | Đưa vào báo cáo | Mục |
|---|---|---|
| `<p>_metrics.csv` | ✔ nguyên bảng Test + số nhóm hàng / cửa hàng thắng MA7 | 1, 2, 3 |
| `<p>_tuning.csv`, `<p>_best_params.json` | Chỉ cấu hình chọn, MAE Validation so với MA7, số cấu hình đã thử | 4 |
| `<p>_predictions.csv` (≈ 6 MB) | Chỉ bản tổng hợp: tổng mỗi ngày (`<p>_theo_ngay.csv`) và theo quy mô ô (`<p>_theo_quy_mo.csv`) | 5 |
| `<p>_error_by_category.csv` | 8 nhóm hàng lớn nhất + gộp phần còn lại | 6 |
| `<p>_error_by_store.csv` | Gộp theo khu vực + số cửa hàng thắng MA7 | 2, 7 |
| `<p>_feature_importance.csv` | Gộp 3 nhóm đặc trưng + top 3 | 8 |
| Sai số dồn ở vài ô (`<p>_thong_tin.json`) | Giải thích R² âm | 9 |
| Mục 5 của notebook (từ **Thảo luận.**) | Trích nguyên văn | 10 |
| `<p>_model.*`, các `.png` riêng của mô hình | ✘ — báo cáo tự vẽ lại biểu đồ chung, cùng màu cho mỗi mô hình | — |

**Quy tắc:** không sửa tay nội dung giữa `<!-- TU_DONG:… -->` và `<!-- /TU_DONG -->` (bị ghi đè khi chạy lại); muốn đổi bảng/biểu đồ tự động thì sửa `module5/tao_bao_cao.py`. Phần viết tay (cách đọc chỉ số, ghi chú, mục 11 Kết luận) nằm ngoài các khối đó, ghi ngày viết, và số phải khớp với khối tự động.

---

## Checklist tiến độ

> ✅ xong · ⚠️ có nhưng cần sửa/bổ sung · ⬜ chưa làm

### Module 3 — Dữ liệu và phân tích cửa hàng · *đánh giá ngày 09/10/2026*

**III. Data Analysis**

- [x] ✅ Mô tả dataset — 216.995 dòng, 90 ngày, 130 cửa hàng, 23 nhóm hàng (`quality_report.csv`)
- [x] ✅ Chất lượng dữ liệu — tổng sau làm sạch **khớp dòng "Tổng cộng"** của file (SL 216.592; 902,95 tỷ VND); không thiếu số, không âm
- [x] ✅ Cửa hàng — `by_store.csv`; online HN007 lớn nhất (9,8% SL); không cửa hàng nào mở/đóng giữa kỳ
- [x] ✅ Thời gian — theo ngày, thứ (T7 cao nhất ≈ 2.736/ngày, T3 thấp nhất ≈ 2.113), Tết (29/01 chỉ 28 giao dịch)
- [x] ✅ Độ phủ — 29,4% ô có giao dịch (`coverage.csv`)
- [ ] ⚠️ Sản phẩm — mới tổng hợp theo nhóm hàng; thiếu top vật tư và theo hãng (211 hãng)
- [ ] ⚠️ Lượng bán — có thống kê, thiếu biểu đồ phân phối số lượng/ô
- [ ] ⚠️ Nhóm `KHAC001`, `THI001` chưa có tên tiếng Việt (module đã cảnh báo)
- [ ] ⚠️ 21.578 dòng trùng hoàn toàn đang giữ lại — cần xác minh trước khi viết vào báo cáo
- [ ] ⬜ Viết thành văn mục III

**IV. Methodology — tiền xử lý**

- [x] ✅ Làm sạch, gắn cờ (bán buôn, khu vực Khác, 0 đồng, trả lại) — module `module3_preprocess.py`
- [x] ✅ Tổng hợp Cửa hàng × Nhóm hàng × Ngày, lấp đủ lịch, tự kiểm tra tổng không đổi
- [x] ✅ Cấu hình qua `.env`; 3 notebook tách theo nội dung; các module khác import được
- [ ] ⚠️ **Lọc giao dịch chưa chốt** — biến mục tiêu đang gồm bán buôn (7,8% SL) và khu vực Khác/online (14,5% SL); ô lớn nhất 1.274 là 1.232 dòng bán buôn tại HN007. Đã có tham số `tao_bang_ngay(bo_ban_buon=…, bo_khu_vuc_khac=…, bo_tra_lai=…, bo_0_dong=…)` — **cần cả nhóm chốt**. Ảnh hưởng tới R² của baseline trên Test (Naive · Seasonal Naive · MA7): giữ tất cả −0,142 · −0,160 · 0,365; **bỏ bán buôn 0,496 · 0,529 · 0,700** (ô lớn nhất còn 207); bỏ khu vực Khác 0,326 · 0,315 · 0,495; bỏ cả hai 0,485 · 0,476 · 0,688
- [ ] ⚠️ Hàng "Trả lại" (462 dòng) đang cộng như hàng bán
- [ ] ⬜ Viết thành văn phần tiền xử lý

**V. Results — theo nhóm hàng và cửa hàng**

- [x] ✅ Lỗi Random Forest theo nhóm hàng và cửa hàng (`rf_error_by_category.csv`, `rf_error_by_store.csv`)
- [ ] ⚠️ Tổng hợp các mô hình theo nhóm hàng/khu vực — tự động trong [báo cáo](bao_cao/BAO_CAO.md) mục 6–7, đang có RF + CatBoost; chờ 3 mô hình còn lại

**Mô hình Random Forest**

- [x] ✅ Train, tối ưu 12 bộ tham số trên Validation, train lại, lưu mô hình + dự báo (`outputs/rf/`, chạy lại 09/10/2026, số khớp bảng dưới)
- [x] ✅ Kiểm tra rò rỉ dữ liệu (đặc trưng chỉ dùng quá khứ)
- [x] ✅ Notebook theo [chuẩn đầu ra](#chuẩn-đầu-ra-mỗi-mô-hình) (mục 1–5); viết phần kết quả và thảo luận (mục 5 trong notebook)
- [x] ✅ Đã nộp vào [báo cáo](bao_cao/BAO_CAO.md) (`bao_cao/du_lieu/rf/`)
- [ ] ⚠️ **Chưa thắng MA7** trên MAE/WAPE, thua cả trên Validation (MAE 0,868 so với 0,847). Thắng MA7 ở 4 nhóm lớn nhất (Điện thoại, Phụ kiện, Hcare, Tai nghe — 81% lượng bán) nhưng dự báo dư 15% ở 19 nhóm thưa. Cấu hình tốt nhất nằm ở biên không gian tìm kiếm (`min_samples_leaf=20`, `max_depth=8`) → thử cây đơn giản hơn; chạy lại sau khi chốt bộ lọc
- [ ] ⚠️ Đặc trưng và mốc chia đang do Module 3 tự đặt — cần thống nhất với Module 4

### Module 5 — Tổng hợp và giá trị thực tế · *đánh giá ngày 09/10/2026*

**Mô hình CatBoost** (`module5/Module5_catboost.ipynb`)

- [x] ✅ Train, tối ưu 12 cấu hình (4 hàm mất mát × 3 bộ tham số, early stopping trên Validation), train lại, lưu mô hình + dự báo (`outputs/catboost/`)
- [x] ✅ Dùng chung bảng, đặc trưng, mốc chia, chỉ số với Random Forest (`chia_tap`, `danh_gia`, `bang_baseline`)
- [x] ✅ Lỗi theo nhóm hàng/cửa hàng, đặc trưng quan trọng, biểu đồ thực tế vs dự báo
- [x] ✅ Viết phần kết quả và thảo luận (mục 5 trong notebook)
- [x] ✅ Đã nộp vào [báo cáo](bao_cao/BAO_CAO.md) (`bao_cao/du_lieu/catboost/`)
- [ ] ⚠️ **Thắng MA7 về MAE/WAPE nhưng dự báo thiếu 42% tổng lượng** (loss MAE ≈ trung vị ≈ 0 với dữ liệu thưa) — cần nhóm chốt tiêu chí chọn mô hình có xét Bias, rồi chạy lại
- [ ] ⚠️ Chạy lại sau khi chốt bộ lọc giao dịch

**Phần việc còn lại của Module 5**

- [x] ✅ Bản báo cáo tổng hợp tự động [bao_cao/BAO_CAO.md](bao_cao/BAO_CAO.md) (`module5/tao_bao_cao.py`): so sánh, chọn mô hình, Validation → Test, tổng lượng, nhóm hàng, khu vực, đặc trưng, R² âm, nhận xét từng module
- [ ] ⚠️ Kết luận (mục 11 báo cáo) mới là bản tạm với RF + CatBoost — chọn mô hình tốt nhất khi có đủ Ridge, Poisson, XGBoost
- [ ] ⬜ Business implications, limitations, recommendations
- [ ] ⬜ VI. Conclusion

### Module 1, 2

- [x] ✅ Baseline Naive, Seasonal Naive, MA7 — có sẵn trong code chung (`bang_baseline`)
- [ ] ⬜ Module 1 — I. Introduction · V. Kết quả baseline · Ridge Regression
- [ ] ⬜ Module 2 — II. Literature Review · IV. Problem formulation, metrics · References · Poisson Regression

### Module 4 — Đặc trưng và thiết kế thí nghiệm

- [ ] ⚠️ **Validation 15–28/02 không đại diện cho Test** — rơi vào giai đoạn phục hồi sau Tết: Ridge, Poisson, CatBoost đều thắng MA7 trên Validation nhưng Ridge/Poisson thua trên Test. Cân nhắc backtest nhiều mốc hoặc dời Validation
- [ ] ⚠️ `ngay_trong_thang` học hiệu ứng Tết tháng 1 (hệ số âm ở Ridge/Poisson, quan trọng 12,5% ở CatBoost) — cân nhắc bỏ hoặc thay bằng cờ giai đoạn Tết
- [ ] ⬜ Chốt đặc trưng, leakage, Train–Validation–Test, quy trình tuning · XGBoost · phân tích lỗi

---

## Kết quả hiện có

Toàn bộ kết quả, so sánh và kết luận nằm ở **[bao_cao/BAO_CAO.md](bao_cao/BAO_CAO.md)** — tự cập nhật khi mỗi module [nộp kết quả](#nộp-kết-quả-vào-báo-cáo); README không chép lại số để tránh lệch.

Tình trạng 09/10/2026: đã nộp Random Forest và CatBoost (bảng mô hình chưa lọc); chờ Ridge, Poisson, XGBoost. Hai việc cần cả nhóm chốt trước khi kết luận — **bộ lọc giao dịch** và **tiêu chí chọn mô hình có xét Bias** — xem mục 9 và 11 của báo cáo.
