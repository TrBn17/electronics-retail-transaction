# CLAUDE.md

Project conventions for the retail demand forecasting assignment. See `README.md` for task assignments, the progress checklist and current results.

## Project

Five-person group project: forecast daily `so_luong` (units sold) for each **store × product group** pair of the Genbyte electronics retail chain, using Q1/2025 data (01/01–31/03/2025, 90 days). Each module owns part of the report (sections I–VI) and one model: Ridge (Module 1), Poisson (Module 2), Random Forest (Module 3), XGBoost (Module 4), CatBoost (Module 5). The end product is a **fair comparison** of 3 baselines and 5 models.

- The report, notebook markdown and code comments are written in **Vietnamese**.
- Notebooks are usually run on **Google Colab**; code must work both locally and on Colab.

## Layout

- `module3/module3_preprocess.py` — **shared code** for all modules:
  - data: `doc_va_lam_sach`, `tao_bang_ngay`, `doc_bang_ngay`, `tao_dac_trung`;
  - evaluation: `chia_tap`, `danh_gia`, `bang_baseline`, `loi_theo_nhom`;
  - utilities: `tai_ve`, `nop_ket_qua` (submit a model's results to the report);
  - constants: `NHOM_HANG`, `KHU_VUC_KHAC`, `VUNG`, `TET`, `FEATS`, `KEY`, `VAL_START`, `TRAIN_END`, `TEST_START`, `SEED`, `MO_HINH` (model names/folders/prefixes), `BAO_CAO_DIR`.
- `module3/Module3_1_tien_xu_ly.ipynb` builds the modelling table; `Module3_2_eda.ipynb`, `Module3_3_random_forest.ipynb` and `module5/Module5_catboost.ipynb` read it.
- Each module keeps its notebooks in its own folder `moduleN/` and imports the shared code:
  ```python
  import sys; sys.path.append("../module3")
  from module3_preprocess import doc_bang_ngay, tao_dac_trung, chia_tap, danh_gia, bang_baseline, loi_theo_nhom
  ```
- `data/` (raw data) and `outputs/` (generated results) are **not committed**. Each model writes to its own folder under `outputs/`; names are fixed in the [Output contract](#output-contract-every-model).
- `bao_cao/` **is committed** and holds aggregated results only: `BAO_CAO.md` (the group's results report), `du_lieu/<folder>/` (filtered results per model), `hinh/` (generated charts). `module5/tao_bao_cao.py` rebuilds it — see [Results report](#results-report-bao_caobao_caomd).
- Configuration lives in `.env` (template: `.env.example`): `DATA_FILE`, `DRIVE_ID`, `OUT_DIR`, `KHU_VUC`. Relative paths resolve from the repo root. Paths and Drive IDs are never hard-coded in notebooks; new settings go into `.env.example`.

## Data

- The raw data contains customer and sales-staff names, and the repo is public. Never commit the data file, `.env`, the Drive ID, or notebook outputs that show raw transaction rows.
- Excel export layout: 2 empty header rows, column names on row 3 (`header=2`), plus a **"Tổng cộng"** (grand total) row. `doc_va_lam_sach` checks the cleaned totals against that row and fails on any mismatch — keep this check.
- `data/electronics-retail-transaction.xlsx` is a **different dataset** (2026, 40 days, 48 columns) and is not used here.
- Almost every row has `Số lượng = 1`; there are no negative quantities. Returns are rows with `Tình trạng = "Trả lại"` and still carry a positive quantity.
- All transactions are **kept and flagged** (`la_ban_buon`, `la_khu_vuc_khac`, `la_0_dong`, `la_tra_lai`). Which ones count toward the target is **not decided yet** (see the README checklist). Filtering goes through `tao_bang_ngay(bo_ban_buon=…, …)`, never ad-hoc in a single notebook.
- The "Khác" region groups non-shop codes in `KHU_VUC_KHAC` (online, company wholesale, repair, warehouse). HN007 and SG007 are online channels with very large wholesale orders (largest cell: 1,274 units).
- Tết (29/01/2025) collapses sales from 28/01 to 01/02: only 28 rows on 29/01 versus about 2,400 on a normal day.

## Modelling rules

- **Fair comparison:** every model uses the same table (`doc_bang_ngay()`), the same features (`tao_dac_trung()` → `FEATS`), the same split (`chia_tap`) and the same metric function (`danh_gia`). The split is defined once in the shared code (Module 4 may finalise it):
  - Tuning train 15/01–14/02 · Validation 15/02–28/02 · Test 01/03–31/03 (refit on 15/01–28/02 before scoring Test).
- Hyperparameters are chosen by **MAE on Validation**.
- Metrics: MAE, RMSE, WAPE, R² and Bias (total forecast / total actual − 1) on Test, always reported next to `bang_baseline(te)` (Naive lag 1, Seasonal Naive lag 7, MA7). Negative predictions are clipped to 0.
- Model-specific handling of the same columns is fine (e.g. CatBoost treats the three `_code` columns as categorical); adding new information that other models do not get is not.
- **No leakage:** features use past data only (shifted by at least 1 day); no tuning on Test; same-day `doanh_thu`, `so_dong` and `co_giao_dich` are never features.
- Saved files and the results section follow the [Output contract](#output-contract-every-model) so Module 5 can merge them.
- `random_state=42` everywhere.
- Results are reported as they are: if a model loses to a baseline, say so rather than picking the metric that flatters it.
- R² is reported exactly as `danh_gia` computes it (`r2_score` on all Test cells) — never clipped at 0, dropped, or replaced by an R² on another scale or level. R² < 0 means worse than predicting the Test mean. On the unfiltered table Naive and Seasonal Naive are negative because 10 cells (wholesale orders, mostly HN007) carry ~80% of their squared error; dropping wholesale (`bo_ban_buon=True`) makes all three baselines positive, but that is the pending group filtering decision, not a fix for one notebook.

## Output contract (every model)

Every model notebook produces the same files and the same results section, so the 5 models can be compared without per-model code. Reference implementations: `module3/Module3_3_random_forest.ipynb`, `module5/Module5_catboost.ipynb`. The README section **Chuẩn đầu ra mỗi mô hình** has the copyable template for section 5.

| Module | Model | Notebook | Folder | Prefix `<p>` | Prediction column | Row name in `<p>_metrics.csv` |
|---|---|---|---|---|---|---|
| 1 | Ridge | `module1/Module1_ridge.ipynb` | `outputs/ridge/` | `ridge` | `du_bao_ridge` | `Ridge` |
| 2 | Poisson | `module2/Module2_poisson.ipynb` | `outputs/poisson/` | `poisson` | `du_bao_poisson` | `Poisson` |
| 3 | Random Forest | `module3/Module3_3_random_forest.ipynb` | `outputs/rf/` | `rf` | `du_bao_rf` | `Random Forest` |
| 4 | XGBoost | `module4/Module4_xgboost.ipynb` | `outputs/xgb/` | `xgb` | `du_bao_xgb` | `XGBoost` |
| 5 | CatBoost | `module5/Module5_catboost.ipynb` | `outputs/catboost/` | `cb` | `du_bao_cb` | `CatBoost` |

This table is also `MO_HINH` in `module3_preprocess.py`; `nop_ket_qua` and the report read it, so change both together.

Files in the model folder:
- `<p>_tuning.csv` — one row per configuration, sorted by Validation MAE: `id`, hyperparameters, `MAE, RMSE, WAPE, R2, Bias` from `danh_gia` on Validation, `giây`.
- `<p>_best_params.json` — enough to rebuild the final model (boosting with early stopping: include the iteration count).
- `<p>_metrics.csv` — Test results. Index exactly `Naive (lag 1)`, `Seasonal Naive (lag 7)`, `MA7`, then the row name; columns `MAE, RMSE, WAPE, R2, Bias, WAPE so với MA7`. Build it with `rows = bang_baseline(te); rows[name] = danh_gia(te["so_luong"], te[pred_col])`.
- `<p>_predictions.csv` — Test rows with `Mã cửa hàng, Nhóm hàng, Khu vực, Vùng, Ngày, so_luong, du_bao_<p>, lag_1, lag_7, tb_7`.
- `<p>_model.*` — the model refitted on 15/01–28/02, in its native format.
- `<p>_feature_importance.csv` + `.png` — index `FEATS`, column `importance` (Ridge/Poisson: absolute coefficients on standardised features).
- `<p>_error_by_category.csv`, `<p>_error_by_store.csv` — `loi_theo_nhom(te, pred_col, "Nhóm hàng")` and `loi_theo_nhom(te, pred_col, ["Mã cửa hàng", "Khu vực"])`.
- `<p>_thuc_te_vs_du_bao.png` — daily March total: actual, model, MA7.

Notebook sections, with these exact headings: header markdown (inputs, outputs, Colab) · `## 1. Đặc trưng và chia tập` (prints `bang_baseline(val)` and `bang_baseline(te)`) · `## 2. Tối ưu tham số trên Validation` · `## 3. Train lại trên toàn bộ Train và chấm Test` · `## 4. Đặc trưng quan trọng và lỗi theo nhóm hàng / cửa hàng / ngày` (prints how many product groups / stores beat MA7 on WAPE) · `## 5. Kết quả và thảo luận` · `## Nộp kết quả vào báo cáo và tải về (Colab)` (last cell: `nop_ket_qua("<folder>", val)` then `tai_ve(...)`).

Section 5 is written in Vietnamese from the saved files, in this order:
1. Italic line: run date, which filter the modelling table used, number of Test cells.
2. **Tối ưu tham số** — best configuration and its Validation MAE next to MA7's Validation MAE; a short table of the top configurations with an MA7 reference row.
3. **Kết quả trên Test** — table with exactly the 3 baselines + the model, columns `MAE | RMSE | WAPE | R² | Bias`.
4. **Thảo luận** (keep the exact label `**Thảo luận.**` — the report quotes from it to the end of the cell), numbered, always covering: vs MA7 on each metric; total volume (Bias, mean daily system-wide forecast vs actual and MA7, forecast total on cells selling ≥ 20 units/day); wins vs MA7 on WAPE by product group (x/23) and store (y/130), naming Điện thoại, Phụ kiện, Hcare; top features with %; comparison with models that already have results.
5. **Giới hạn** (and **Hướng cải thiện** if any).

Number format in reports: 3 decimals, Vietnamese separators (`0,757`, `2.393`), Bias as a signed % (`+0,8%`, `−41,9%`), best value per column in bold, all numbers taken from `outputs/`. After a model is done, submit it to the report (next section) and tick the README checklist.

## Results report (`bao_cao/BAO_CAO.md`)

`bao_cao/BAO_CAO.md` is the group's single results report: it compares the 3 baselines and every submitted model, explains the differences and holds the conclusion. README **Kết quả hiện có** only links to it — never copy result numbers back into README.

**When a model notebook is finished** (yours or one you are asked to complete), do all of these:
1. The last cell calls `nop_ket_qua("<folder>", val)`. It copies only the report-worthy files from `outputs/<folder>/` to `bao_cao/du_lieu/<folder>/`: metrics, tuning, best params, feature importance, errors by product group and by store, plus files derived from the predictions (`<p>_theo_ngay.csv` daily totals, `<p>_theo_quy_mo.csv` by cell size, `<p>_thong_tin.json` run info, table filter, Validation baselines and squared-error concentration). Predictions, models and per-model PNGs stay out of the report. On Colab it downloads `bao_cao_<folder>.zip`, which is unzipped into `bao_cao/du_lieu/<folder>/` in the repo.
2. Run `python module5/tao_bao_cao.py` from the repo root. It needs no raw data and rewrites every `<!-- TU_DONG:… -->` block and `bao_cao/hinh/*.png`.
3. Read the regenerated report. The status line must say **✔ cùng bảng, cùng tập Test, cùng baseline**; if it shows ⚠️, the model was scored on a different table or split. Fix and rerun it instead of committing a comparison that is not fair.
4. If **Tóm tắt** or **Chọn mô hình** changed (e.g. a new model beats MA7 or becomes the pick), update the hand-written section 11 **Kết luận và khuyến nghị** (Module 5 owns it) with the date and numbers taken from the report, or flag it to the user.
5. Commit the notebook and `bao_cao/` together; never `outputs/`.

**Editing the report:**
- Never edit inside a `<!-- TU_DONG:name -->` … `<!-- /TU_DONG -->` block; it is overwritten on every run. To change generated tables, charts or wording, edit `module5/tao_bao_cao.py`; a new block needs both a marker pair in `BAO_CAO.md` and a key in `main()`.
- Hand-written text (metric guide, notes after sections 3, 8, 9, and section 11) lives outside the blocks, is dated, and its numbers must match the generated blocks. Rerun the script after any change to results and reread the hand-written parts for stale numbers.
- Charts keep one fixed colour per model (`MAU` in the script): actual sales in black, MA7 in grey. Number format follows the rules above.
- Results that disagree with a model's own notebook section 5 mean one of them is stale: rerun the notebook, resubmit, rebuild.

## Changing the shared module

- `module3_preprocess.py` affects **all five modules**. Changes to default behaviour (filtering, features, category mappings) need group agreement and a note in the README.
- After changing the module or the preprocessing notebook, rerun `Module3_1_tien_xu_ly`, confirm the `assert`s still pass, and compare the total `so_luong` in `outputs/daily_store_category.csv` with the previous run (currently 216,592 with no filtering).
- New product-group or region codes are added to `NHOM_HANG` / `KHU_VUC_KHAC`, not patched inside a notebook.

## Code style

- Match the existing code: compact pandas, Vietnamese identifiers without diacritics (`so_luong`, `doanh_thu`, `tao_bang_ngay`), short Vietnamese comments, `assert` for data invariants.
- CSVs are written with `encoding="utf-8-sig"` so they open correctly in Excel.
- Every notebook starts with a markdown cell stating inputs and outputs and ends with `tai_ve(...)` to download results on Colab (model notebooks call `nop_ket_qua(...)` just before it).

## README checklist

When an item is finished, update the **Checklist tiến độ** in `README.md` (✅ / ⚠️ / ⬜) with real numbers taken from `outputs/` or the report, not estimates. Model results themselves go into `bao_cao/BAO_CAO.md` via `nop_ket_qua` (see Results report).
