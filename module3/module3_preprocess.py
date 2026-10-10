"""Module 3 — tiền xử lý bảng kê bán hàng Genbyte, dùng chung cho cả 5 mô hình.

    import sys; sys.path.append("../module3")       # notebook ở thư mục module khác (module1/, module4/…)
    from module3_preprocess import doc_va_lam_sach, tao_bang_ngay, doc_bang_ngay, tao_dac_trung
    df, info = doc_va_lam_sach()                 # giao dịch đã làm sạch + cờ
    daily = tao_bang_ngay(df)                    # Cửa hàng × Nhóm hàng × Ngày, đủ lịch
    daily = doc_bang_ngay()                      # hoặc đọc lại bảng đã xuất ở notebook tiền xử lý
    d, FEATS = tao_dac_trung(daily)              # lag/rolling/lịch, chỉ dùng quá khứ
    fit, val, tr, te = chia_tap(d)               # mốc chia chung; chấm bằng danh_gia, so với bang_baseline(te)
    nop_ket_qua("rf", val)                       # cuối notebook mô hình: lọc kết quả vào bao_cao/du_lieu/rf/
"""
import os
import warnings

import numpy as np
import pandas as pd
import requests

# Gốc repo = thư mục chứa .env: cạnh module (Colab upload phẳng) hoặc thư mục cha (repo: module3/ nằm dưới gốc)
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next((p for p in (_HERE, os.path.dirname(_HERE)) if os.path.exists(os.path.join(p, ".env"))),
            os.path.dirname(_HERE))


def _doc_env(path=os.path.join(ROOT, ".env")):
    """Nạp KEY=VALUE từ .env vào os.environ (biến đã đặt sẵn trong môi trường được ưu tiên)."""
    if os.path.exists(path):
        for dong in open(path, encoding="utf-8"):
            dong = dong.strip()
            if dong and not dong.startswith("#") and "=" in dong:
                k, v = dong.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_doc_env()
# Đường dẫn tương đối trong .env tính từ gốc repo, không phụ thuộc thư mục đang chạy notebook
DATA_FILE = os.path.join(ROOT, os.getenv("DATA_FILE", "data/electronics-retail-transaction-q1-2025.xlsx"))
DRIVE_ID = os.getenv("DRIVE_ID")                 # ID file trên Google Drive, để tự tải khi chưa có DATA_FILE
OUT_DIR = os.path.join(ROOT, os.getenv("OUT_DIR", "outputs"))
KHU_VUC = os.getenv("KHU_VUC") or None           # trống = toàn công ty; hoặc vd KVHN1
TET = pd.Timestamp("2025-01-29")                 # Mùng 1 Tết Ất Tỵ
# Mốc chia Train–Validation–Test dùng chung cho mọi mô hình (Module 4 chốt lại nếu cần)
VAL_START, TRAIN_END, TEST_START = "2025-02-15", "2025-02-28", "2025-03-01"
SEED = 42

NUM = ["Số lượng", "Tiền đã CK/giảm giá", "Tổng đơn KH", "Hoa hồng BH"]
KEY = ["Mã cửa hàng", "Nhóm hàng"]
NHOM_HANG = {"DIENTHOAI": "Điện thoại", "LAPTOP": "Laptop", "MAYTINHBANG": "Máy tính bảng", "MANHINH": "Màn hình",
             "PHUKIEN": "Phụ kiện", "TAINGHE": "Tai nghe", "DONGHO": "Đồng hồ", "HCARE001": "Hcare (bảo hành)",
             "LINHKIENK": "Linh kiện", "DIENTU": "Điện tử (TV)", "LOA": "Loa", "SMARTHOME": "Smarthome",
             "CAMERA": "Camera/mạng", "DIENLANH": "Điện lạnh", "GIADUNG": "Gia dụng", "SIMTHE": "Sim thẻ",
             "KHUYENMAI": "Khuyến mãi", "LINHKIENMT": "Linh kiện máy tính", "MAYTINHPC": "Máy tính PC",
             "SCBH": "Sửa chữa BH", "PHANMEM001": "Phần mềm", "NMANG001": "Nhà mạng"}
# Mã KVKD không phải vùng shop bán lẻ (đã đối chiếu danh mục khu vực Genbyte 06/10/2026)
KHU_VUC_KHAC = {"KVOL": "Khác - Online", "KVHN0": "Khác - Công ty (bán buôn)", "KVHCM0": "Khác - Chi nhánh HCM",
                "KVKDSCDV": "Khác - Dịch vụ sửa chữa", "KVMTAY": "Khác - Kho Miền Tây"}
VUNG = {"HCM": "Miền HCM", "HN": "Miền Hà Nội", "MB": "Miền Bắc", "MN": "Miền Nam", "MTR": "Miền Trung"}
FEATS = ["lag_1", "lag_7", "lag_14", "tb_7", "tb_14", "std_7", "std_14", "thu", "cuoi_tuan", "ngay_trong_thang",
         "ngay_toi_tet", "Mã cửa hàng_code", "Nhóm hàng_code", "Vùng_code"]
# Chuẩn đầu ra: thư mục trong outputs/ → (tiền tố file, tên mô hình, module, notebook)
MO_HINH = {"ridge": ("ridge", "Ridge", 1, "module1/Module1_ridge.ipynb"),
           "poisson": ("poisson", "Poisson", 2, "module2/Module2_poisson.ipynb"),
           "rf": ("rf", "Random Forest", 3, "module3/Module3_3_random_forest.ipynb"),
           "xgb": ("xgb", "XGBoost", 4, "module4/Module4_xgboost.ipynb"),
           "lightgbm": ("lgbm", "LightGBM", 5, "module5/Module5_lightgbm.ipynb")}
BAO_CAO_DIR = os.path.join(ROOT, "bao_cao")      # bản báo cáo tổng hợp — CÓ commit (chỉ số liệu đã tổng hợp)


def tai_du_lieu(file=DATA_FILE):
    """Tải file từ Google Drive nếu chưa có trên máy."""
    if not os.path.exists(file):
        if not DRIVE_ID:
            raise FileNotFoundError(f"Không có {file}; đặt file vào đó hoặc khai báo DRIVE_ID trong .env để tự tải")
        os.makedirs(os.path.dirname(file) or ".", exist_ok=True)
        r = requests.get(f"https://docs.google.com/uc?export=download&id={DRIVE_ID}", timeout=300)
        r.raise_for_status()
        with open(file, "wb") as f:
            f.write(r.content)
        print("Đã tải", file)
    return file


def doc_va_lam_sach(file=DATA_FILE, khu_vuc=KHU_VUC):
    """Đọc file export Genbyte → (giao dịch đã làm sạch + cờ, dict số liệu cho báo cáo chất lượng).

    File export: 2 dòng tiêu đề trống, dòng 3 là tên cột, có dòng 'Tổng cộng' để đối chiếu.
    Giữ mọi giao dịch, chỉ gắn cờ: la_ban_buon, la_khu_vuc_khac, la_0_dong, la_tra_lai.
    khu_vuc: None = toàn công ty, hoặc một Mã KVKD (vd "KVHN1"); mặc định lấy KHU_VUC trong .env.
    """
    raw = pd.read_excel(tai_du_lieu(file), header=2, dtype={"Mã ca": str, "Mã cửa hàng": str, "Mã KVKD": str})
    la_tong = raw["Tên vật tư"].astype(str).str.strip().eq("Tổng cộng")
    tong_sl, tong_dt = raw.loc[la_tong, ["Số lượng", "Tổng đơn KH"]].iloc[0].astype(float)

    df = raw[raw["Ngày CT"].notna() & raw["Tên vật tư"].notna() & ~la_tong].copy()
    text = [c for c in df.columns if c not in NUM + ["Ngày CT"]]
    df[text] = df[text].apply(lambda s: s.astype("string").str.strip().replace({"": pd.NA}).fillna("Không rõ"))
    df[NUM] = df[NUM].apply(pd.to_numeric, errors="coerce")
    so_thieu_so = int(df[NUM].isna().sum().sum())
    df[NUM] = df[NUM].fillna(0)
    df["Ngày"] = pd.to_datetime(df["Ngày CT"]).dt.normalize()

    df["Nhóm hàng"] = df["Chủng loại"].map(NHOM_HANG).fillna(df["Chủng loại"])
    df["Khu vực"] = df["Mã KVKD"].map(KHU_VUC_KHAC).fillna(df["Mã KVKD"])
    vung = df["Mã KVKD"].str.extract(r"^KV(HCM|HN|MB|MN|MTR)", expand=False).map(VUNG)
    la_khac = df["Mã KVKD"].isin(list(KHU_VUC_KHAC))
    df["Vùng"] = vung.fillna("Khác").mask(la_khac, "Khác")

    ma_la = sorted(set(df["Chủng loại"]) - set(NHOM_HANG))
    if ma_la:
        warnings.warn(f"Chủng loại chưa có tên tiếng Việt, giữ nguyên mã: {ma_la}")
    kv_la = sorted(set(df.loc[vung.isna() & ~la_khac, "Mã KVKD"]))
    if kv_la:
        warnings.warn(f"Mã KVKD không thuộc vùng nào, đang xếp vào 'Khác': {kv_la}")

    flags = {"la_ban_buon": df["Loại hình bán"].eq("Bán buôn"), "la_khu_vuc_khac": df["Vùng"].eq("Khác"),
             "la_0_dong": df["Tổng đơn KH"].eq(0), "la_tra_lai": df["Tình trạng"].eq("Trả lại")}
    for k, v in flags.items():
        df[k] = v.astype(int)

    sl, dt = df["Số lượng"].sum(), df["Tổng đơn KH"].sum()
    assert round(sl) == round(tong_sl) and round(dt) == round(tong_dt), \
        f"Lệch dòng Tổng cộng: SL {sl:,.0f} vs {tong_sl:,.0f}, Tổng đơn KH {dt:,.0f} vs {tong_dt:,.0f}"

    info = {"so_dong_doc": len(raw), "so_dong_bo": len(raw) - len(df), "so_thieu_so": so_thieu_so,
            "so_dong_trung": int(raw.loc[df.index].duplicated().sum()), "tong_sl_file": tong_sl, "tong_dt_file": tong_dt}
    if khu_vuc:
        df = df[df["Mã KVKD"].eq(khu_vuc)]
    return df, info


def tao_bang_ngay(df, bo_ban_buon=False, bo_khu_vuc_khac=False, bo_tra_lai=False, bo_0_dong=False):
    """Tổng hợp Cửa hàng × Nhóm hàng × Ngày, lấp đủ lịch (ngày không bán = 0, co_giao_dich = 0).

    Các tham số bo_* lọc giao dịch theo cờ trước khi tổng hợp; mặc định giữ tất cả.
    """
    for co, bo in [("la_ban_buon", bo_ban_buon), ("la_khu_vuc_khac", bo_khu_vuc_khac),
                   ("la_tra_lai", bo_tra_lai), ("la_0_dong", bo_0_dong)]:
        if bo:
            df = df[df[co].eq(0)]
    agg = (df.groupby(KEY + ["Ngày"], as_index=False)
             .agg(so_luong=("Số lượng", "sum"), doanh_thu=("Tổng đơn KH", "sum"), so_dong=("Số lượng", "size"),
                  so_dong_tra_lai=("la_tra_lai", "sum"), so_dong_ban_buon=("la_ban_buon", "sum")))
    lich = pd.date_range(df["Ngày"].min(), df["Ngày"].max(), freq="D")
    cap = agg[KEY].drop_duplicates()
    daily = cap.merge(pd.DataFrame({"Ngày": lich}), how="cross").merge(agg, on=KEY + ["Ngày"], how="left")
    daily["co_giao_dich"] = daily["so_dong"].notna().astype(int)
    shop = df.groupby("Mã cửa hàng")[["Mã KVKD", "Khu vực", "Vùng"]].first()
    daily = daily.fillna(0).merge(shop, on="Mã cửa hàng", how="left").sort_values(KEY + ["Ngày"]).reset_index(drop=True)

    # tổng hợp không làm mất/đẻ thêm số lượng hay doanh thu
    assert round(daily["so_luong"].sum()) == round(df["Số lượng"].sum())
    assert round(daily["doanh_thu"].sum()) == round(df["Tổng đơn KH"].sum())
    assert len(daily) == len(cap) * len(lich) and not daily.duplicated(KEY + ["Ngày"]).any()
    return daily


def doc_bang_ngay(path=None):
    """Đọc bảng mô hình đã xuất (outputs/daily_store_category.csv); chưa có thì dựng lại từ file gốc và lưu."""
    path = path or f"{OUT_DIR}/daily_store_category.csv"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tao_bang_ngay(doc_va_lam_sach()[0]).to_csv(path, index=False, encoding="utf-8-sig")
    text = ["Mã cửa hàng", "Nhóm hàng", "Mã KVKD", "Khu vực", "Vùng"]
    return pd.read_csv(path, parse_dates=["Ngày"], dtype=dict.fromkeys(text, "string"), encoding="utf-8-sig")


def tai_ve(thu_muc, ten_zip):
    """Nén thư mục kết quả; trên Colab thì tải file zip về máy (file trên Colab mất khi tắt phiên)."""
    import shutil
    zip_path = shutil.make_archive(ten_zip, "zip", thu_muc)
    try:
        from google.colab import files
        files.download(zip_path)
    except ImportError:
        print("Đã tạo", zip_path)


def tao_dac_trung(daily):
    """Thêm lag/rolling (dời ≥ 1 ngày → không lộ tương lai), lịch, mã hoá cửa hàng/nhóm hàng/vùng.

    Trả về (bảng đầy đủ, danh sách cột đặc trưng). 14 ngày đầu mỗi chuỗi thiếu lag_14 → dropna(subset=FEATS).
    """
    d = daily.sort_values(KEY + ["Ngày"]).reset_index(drop=True)
    g = d.groupby(KEY, sort=False)["so_luong"]
    for lag in (1, 7, 14):
        d[f"lag_{lag}"] = g.shift(lag)
    pg = d.groupby(KEY, sort=False)["lag_1"]
    for w in (7, 14):
        d[f"tb_{w}"] = pg.transform(lambda s: s.rolling(w).mean())
        d[f"std_{w}"] = pg.transform(lambda s: s.rolling(w).std())
    d["thu"] = d["Ngày"].dt.dayofweek
    d["cuoi_tuan"] = (d["thu"] >= 5).astype(int)
    d["ngay_trong_thang"] = d["Ngày"].dt.day
    d["ngay_toi_tet"] = (d["Ngày"] - TET).dt.days
    for c in ["Mã cửa hàng", "Nhóm hàng", "Vùng"]:
        d[c + "_code"] = d[c].astype("category").cat.codes

    # kiểm tra rò rỉ: lag_1 ngày t = số lượng ngày t-1
    k = d.dropna(subset=["lag_1"]).sample(500, random_state=42)
    prev = d.set_index(KEY + ["Ngày"])["so_luong"]
    assert (k["lag_1"].values == prev.loc[list(zip(k["Mã cửa hàng"], k["Nhóm hàng"],
                                                   k["Ngày"] - pd.Timedelta(days=1)))].values).all()
    return d, FEATS


def chia_tap(d):
    """Chia theo thời gian, dùng chung cho mọi mô hình → (fit, val, tr, te).

    fit: train để tối ưu · val: chấm khi tối ưu · tr = fit + val: train lại trước khi chấm · te: Test (tháng 3).
    """
    data = d.dropna(subset=FEATS)            # 14 ngày đầu thiếu lag 14
    tr = data[data["Ngày"] <= TRAIN_END]
    fit, val = tr[tr["Ngày"] < VAL_START], tr[tr["Ngày"] >= VAL_START]
    te = data[data["Ngày"] >= TEST_START].copy()
    assert tr["Ngày"].max() < te["Ngày"].min()
    for n, x in [("Train (tối ưu)", fit), ("Validation", val), ("Train đầy đủ", tr), ("Test", te)]:
        print(f"{n:15} {x['Ngày'].min():%d/%m} → {x['Ngày'].max():%d/%m} | {len(x):,} dòng")
    return fit, val, tr, te


def danh_gia(y, p):
    """MAE, RMSE, WAPE, R², Bias — cùng một hàm cho baseline và mọi mô hình.

    Bias = tổng dự báo / tổng thực tế − 1 (âm = dự báo thiếu tổng lượng).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "WAPE": np.abs(y - p).sum() / max(y.sum(), 1e-9), "R2": r2_score(y, p),
            "Bias": p.sum() / max(y.sum(), 1e-9) - 1}


def bang_baseline(te):
    """Naive (lag 1), Seasonal Naive (lag 7), MA7 trên cùng tập → dict {tên: chỉ số}."""
    return {"Naive (lag 1)": danh_gia(te["so_luong"], te["lag_1"]),
            "Seasonal Naive (lag 7)": danh_gia(te["so_luong"], te["lag_7"]),
            "MA7": danh_gia(te["so_luong"], te["tb_7"])}


def nop_ket_qua(thu_muc, val):
    """Lọc kết quả đưa vào báo cáo: outputs/<thu_muc>/ → bao_cao/du_lieu/<thu_muc>/ (chỉ file nhỏ, đã tổng hợp).

    Gọi ở cuối notebook mô hình, sau khi đã lưu đủ file theo chuẩn đầu ra. val = tập Validation của chia_tap.
    Sau đó chạy `python module5/tao_bao_cao.py` để cập nhật bao_cao/BAO_CAO.md. Trên Colab: tải zip về.
    """
    import json, shutil
    p, ten = MO_HINH[thu_muc][:2]
    src, dich = f"{OUT_DIR}/{thu_muc}", f"{BAO_CAO_DIR}/du_lieu/{thu_muc}"
    os.makedirs(dich, exist_ok=True)
    for f in ["metrics.csv", "tuning.csv", "best_params.json", "feature_importance.csv",
              "error_by_category.csv", "error_by_store.csv"]:
        shutil.copy(f"{src}/{p}_{f}", f"{dich}/{p}_{f}")

    pr = pd.read_csv(f"{src}/{p}_predictions.csv", parse_dates=["Ngày"], encoding="utf-8-sig")
    cot = f"du_bao_{p}"
    assert cot in pr, f"{p}_predictions.csv thiếu cột {cot}"
    assert ten in pd.read_csv(f"{src}/{p}_metrics.csv", index_col=0, encoding="utf-8-sig").index, \
        f"{p}_metrics.csv thiếu dòng '{ten}'"
    # tổng theo ngày (biểu đồ) và theo quy mô ô (tổng lượng ở ô nhỏ/lớn)
    pr.groupby("Ngày")[["so_luong", cot, "tb_7"]].sum().rename(columns={cot: "du_bao"}) \
        .to_csv(f"{dich}/{p}_theo_ngay.csv", encoding="utf-8-sig")
    pr["quy_mo"] = pd.cut(pr["so_luong"], [-1, 0, 4, 19, np.inf], labels=["0", "1–4", "5–19", "≥ 20"])
    pr["ae"], pr["ae_ma7"] = (pr["so_luong"] - pr[cot]).abs(), (pr["so_luong"] - pr["tb_7"]).abs()
    pr.groupby("quy_mo", observed=False).agg(so_o=("so_luong", "size"), so_luong=("so_luong", "sum"),
                                             du_bao=(cot, "sum"), tb_7=("tb_7", "sum"), MAE=("ae", "mean"),
                                             MAE_MA7=("ae_ma7", "mean")) \
        .to_csv(f"{dich}/{p}_theo_quy_mo.csv", encoding="utf-8-sig")

    # tỷ trọng bình phương sai số do 10 ô lớn nhất gây ra — để giải thích R² âm
    def top10(c):
        se = (pr["so_luong"] - pr[c]) ** 2
        ch = se.groupby(pr["Mã cửa hàng"]).sum()
        return {"top10_o": se.nlargest(10).sum() / se.sum(), "cua_hang": ch.idxmax(), "ty_trong_cua_hang": ch.max() / se.sum()}
    daily = doc_bang_ngay()
    bo_loc = {"bán buôn": daily["so_dong_ban_buon"].sum() > 0, "khu vực Khác": daily["Vùng"].eq("Khác").any(),
              "trả lại": daily["so_dong_tra_lai"].sum() > 0}
    info = {"thu_muc": thu_muc, "ten": ten, "module": MO_HINH[thu_muc][2],
            "ngay_chay": pd.Timestamp.fromtimestamp(os.path.getmtime(f"{src}/{p}_metrics.csv")).strftime("%d/%m/%Y"),
            "bang_tong_so_luong": float(daily["so_luong"].sum()),
            "bang_giu": [k for k, v in bo_loc.items() if v], "bang_bo": [k for k, v in bo_loc.items() if not v],
            "test_so_o": len(pr), "test_tong": float(pr["so_luong"].sum()),
            "baseline_val": pd.DataFrame(bang_baseline(val)).T.to_dict("index"),
            "sai_so_top10": {c: top10(c) for c in ["lag_1", "lag_7", "tb_7", cot]}}
    with open(f"{dich}/{p}_thong_tin.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=str)
    print(f"Đã nộp {ten} → {dich}\nBước tiếp: python module5/tao_bao_cao.py → cập nhật bao_cao/BAO_CAO.md, rồi commit bao_cao/")
    try:                                 # Colab: tải zip về, giải nén vào bao_cao/du_lieu/<thu_muc>/ trong repo
        import google.colab  # noqa: F401
        tai_ve(dich, f"bao_cao_{thu_muc}")
    except ImportError:
        pass
    return dich


def loi_theo_nhom(te, cot_du_bao, gcols):
    """Lỗi theo nhóm (vd "Nhóm hàng" hoặc ["Mã cửa hàng", "Khu vực"]), kèm WAPE của MA7 để so sánh."""
    from sklearn.metrics import mean_absolute_error
    wape = lambda s, c: np.abs(s["so_luong"] - s[c]).sum() / max(s["so_luong"].sum(), 1e-9)
    x = te.groupby(gcols).apply(lambda s: pd.Series({
        "so_luong_thuc": s["so_luong"].sum(), "so_luong_du_bao": s[cot_du_bao].sum(),
        "MAE": mean_absolute_error(s["so_luong"], s[cot_du_bao]),
        "WAPE": wape(s, cot_du_bao), "WAPE_MA7": wape(s, "tb_7")}), include_groups=False)
    return x.sort_values("so_luong_thuc", ascending=False)
