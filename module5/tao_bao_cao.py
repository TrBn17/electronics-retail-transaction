"""Module 5 — cập nhật bản báo cáo tổng hợp bao_cao/BAO_CAO.md từ kết quả các module đã nộp.

    python module5/tao_bao_cao.py

Chỉ đọc bao_cao/du_lieu/<mô hình>/ (do nop_ket_qua ghi) và mục 5 của notebook mỗi mô hình — không cần dữ liệu gốc.
Ghi đè nội dung giữa <!-- TU_DONG:ten --> và <!-- /TU_DONG -->, vẽ lại bao_cao/hinh/*.png; phần viết tay giữ nguyên.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "module3"))
from module3_preprocess import MO_HINH, BAO_CAO_DIR, ROOT  # noqa: E402

DU_LIEU, HINH, FILE = f"{BAO_CAO_DIR}/du_lieu", f"{BAO_CAO_DIR}/hinh", f"{BAO_CAO_DIR}/BAO_CAO.md"
BASE = ["Naive (lag 1)", "Seasonal Naive (lag 7)", "MA7"]
COLS = ["MAE", "RMSE", "WAPE", "R2", "Bias"]
NGUONG_BIAS = 0.10                                   # đề xuất: chỉ chọn mô hình lệch tổng lượng ≤ 10%
# màu cố định theo mô hình (không đổi khi thêm/bớt mô hình); thực tế = mực đen, baseline = xám
MAU = {"ridge": "#2a78d6", "poisson": "#eb6834", "rf": "#1baf7a", "xgb": "#eda100", "catboost": "#e87ba4"}
MUC, XAM, LUOI, NEN = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
NHOM_DT = {"Bán gần đây (lag, trung bình, độ lệch chuẩn)": ["lag_1", "lag_7", "lag_14", "tb_7", "tb_14", "std_7", "std_14"],
           "Lịch (thứ, cuối tuần, ngày trong tháng, Tết)": ["thu", "cuoi_tuan", "ngay_trong_thang", "ngay_toi_tet"],
           "Mã cửa hàng / nhóm hàng / vùng": ["Mã cửa hàng_code", "Nhóm hàng_code", "Vùng_code"]}


# ---------- định dạng ----------
def so(x, n=3):
    """Số kiểu Việt: 0,757 · 2.393 · −0,142."""
    return f"{x:,.{n}f}".replace(",", " ").replace(".", ",").replace(" ", ".").replace("-", "−")


def pt(x, n=1):
    """Tỷ lệ có dấu: +0,8% · −41,9%."""
    return f"{x * 100:+.{n}f}%".replace(".", ",").replace("-", "−")


def bang(dau, dong, trai=1):
    """Bảng markdown; `trai` cột đầu căn trái, còn lại căn phải."""
    return "\n".join(["| " + " | ".join(dau) + " |",
                      "|" + "|".join(["---"] * trai + ["---:"] * (len(dau) - trai)) + "|"]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in dong])


def dam(gia_tri, tot):
    """In đậm ô tốt nhất: gia_tri = list số, tot = hàm chọn chỉ số tốt nhất."""
    i = tot(gia_tri)
    return lambda j, s: f"**{s}**" if j == i else s


# ---------- đọc dữ liệu đã nộp ----------
def doc():
    kq = {}
    for t, (p, ten, module, nb) in MO_HINH.items():
        d = f"{DU_LIEU}/{t}"
        if not os.path.exists(f"{d}/{p}_thong_tin.json"):
            continue
        r = lambda f, **k: pd.read_csv(f"{d}/{p}_{f}", encoding="utf-8-sig", **k)  # noqa: E731
        kq[t] = {"p": p, "ten": ten, "module": module, "notebook": nb,
                 "metrics": r("metrics.csv", index_col=0), "tuning": r("tuning.csv").sort_values("MAE"),
                 "params": json.load(open(f"{d}/{p}_best_params.json", encoding="utf-8")),
                 "imp": r("feature_importance.csv", index_col=0)["importance"].abs(),
                 "cat": r("error_by_category.csv", index_col=0), "store": r("error_by_store.csv"),
                 "ngay": r("theo_ngay.csv", index_col=0, parse_dates=True),
                 "quy_mo": r("theo_quy_mo.csv", dtype={"quy_mo": str}).set_index("quy_mo"),
                 "info": json.load(open(f"{d}/{p}_thong_tin.json", encoding="utf-8"))}
    return kq


def kiem_tra(kq):
    """So sánh công bằng: mọi mô hình cùng bảng, cùng tập Test, cùng baseline."""
    g = next(iter(kq.values()))
    khoa = lambda m: (m["info"]["bang_tong_so_luong"], m["info"]["test_so_o"], m["info"]["test_tong"])  # noqa: E731
    loi = [f"{m['ten']}: bảng/tập Test khác {g['ten']} (tổng bảng, số ô Test, tổng Test = {khoa(m)} so với {khoa(g)})"
           for m in kq.values() if khoa(m) != khoa(g)]
    loi += [f"{m['ten']}: chỉ số baseline khác {g['ten']}" for m in kq.values()
            if not np.allclose(m["metrics"].loc[BASE, COLS], g["metrics"].loc[BASE, COLS])]
    return loi


def wape_gop(df, cot_wape, thuc="so_luong_thuc"):
    """Gộp WAPE nhiều nhóm = tổng sai số tuyệt đối / tổng thực tế."""
    return (df[cot_wape] * np.maximum(df[thuc], 1e-9)).sum() / max(df[thuc].sum(), 1e-9)


# ---------- các khối ----------
def khoi_trang_thai(kq, loi):
    i = next(iter(kq.values()))["info"]
    tt = " · ".join(f"✅ {MO_HINH[t][1]}" if t in kq else f"⬜ {MO_HINH[t][1]}" for t in MO_HINH)
    giu = ", ".join(i["bang_giu"]) or "không"
    bo = ", ".join(i["bang_bo"]) or "không"
    cong_bang = "✔ cùng bảng, cùng tập Test, cùng baseline" if not loi else "⚠️ " + "; ".join(loi)
    return (f"> **Cập nhật tự động {pd.Timestamp.now():%d/%m/%Y %H:%M}** · Test 01/03–31/03/2025: "
            f"{so(i['test_so_o'], 0)} ô, tổng thực tế {so(i['test_tong'], 0)} sản phẩm · bảng mô hình tổng "
            f"{so(i['bang_tong_so_luong'], 0)} sản phẩm — giữ: {giu}; bỏ: {bo}\n>\n"
            f"> **Đã nộp {len(kq)}/{len(MO_HINH)}:** {tt}  \n> **Kiểm tra so sánh công bằng:** {cong_bang}")


def bang_chinh(kq):
    m0 = next(iter(kq.values()))["metrics"]
    rows = pd.concat([m0.loc[BASE, COLS]] + [m["metrics"].loc[[m["ten"]], COLS] for m in kq.values()])
    thang = {m["ten"]: f"{(m['cat'].WAPE < m['cat'].WAPE_MA7).sum()}/{len(m['cat'])} · "
                       f"{(m['store'].WAPE < m['store'].WAPE_MA7).sum()}/{len(m['store'])}" for m in kq.values()}
    return rows, thang


def khoi_tom_tat(kq, rows):
    ml = rows.drop(BASE)
    ma7 = rows.loc["MA7"]
    tot = lambda c, f=min: f(rows.index, key=lambda r: rows.loc[r, c])  # noqa: E731
    thang = [r for r in ml.index if ml.loc[r, "MAE"] < ma7["MAE"]]
    thua = [r for r in ml.index if r not in thang]
    lech = [r for r in rows.index if abs(rows.loc[r, "Bias"]) > NGUONG_BIAS]
    dat = rows[rows["Bias"].abs() <= NGUONG_BIAS]
    chon = dat["MAE"].idxmin()
    chua = [MO_HINH[t][1] for t in MO_HINH if t not in kq]
    nhat = lambda c, f=min: f"{tot(c, f)} {so(rows.loc[tot(c, f), c])}"  # noqa: E731
    out = [f"- **MAE thấp nhất:** {nhat('MAE')} (MA7 {so(ma7['MAE'])}) · **WAPE thấp nhất:** {nhat('WAPE')} "
           f"(MA7 {so(ma7['WAPE'])}).",
           f"- **RMSE thấp nhất:** {nhat('RMSE')} (MA7 {so(ma7['RMSE'])}) · **R² cao nhất:** {nhat('R2', max)} "
           f"(MA7 {so(ma7['R2'])}).",
           f"- **Thắng MA7 về MAE:** {', '.join(thang) or 'chưa có mô hình nào'}"
           + (f" · **thua MA7:** " + ", ".join(f"{r} (MAE {pt(ml.loc[r, 'MAE'] / ma7['MAE'] - 1)})" for r in thua)
              if thua else "") + ".",
           f"- **Lệch tổng lượng quá {NGUONG_BIAS:.0%} (|Bias|):** "
           + (", ".join(f"{r} ({pt(rows.loc[r, 'Bias'])})" for r in lech) or "không có") + ".",
           f"- **Theo tiêu chí đề xuất (MAE thấp nhất trong các mô hình có |Bias| ≤ {NGUONG_BIAS:.0%}):** **{chon}**"
           + (" — chưa mô hình học máy nào vượt MA7 khi xét cả tổng lượng." if chon in BASE else ".")]
    if chua:
        out.append(f"- **Chưa nộp:** {', '.join(chua)} — kết luận dưới đây là tạm thời.")
    return "\n".join(out)


def khoi_bang_test(kq, rows, thang):
    f = {c: dam(list(rows[c]), (lambda v: int(np.argmax(v))) if c == "R2" else
                (lambda v: int(np.argmin(np.abs(v)))) if c == "Bias" else (lambda v: int(np.argmin(v)))) for c in COLS}
    dong = []
    for j, (r, x) in enumerate(rows.iterrows()):
        canh = " ⚠️" if abs(x["Bias"]) > NGUONG_BIAS else ""
        dong.append([f"**{r}**" if r not in BASE else r]
                    + [f[c](j, so(x[c])) for c in COLS[:4]] + [f["Bias"](j, pt(x["Bias"])) + canh]
                    + ["—" if r == "MA7" else pt(x["WAPE"] / rows.loc["MA7", "WAPE"] - 1), thang.get(r, "—")])
    ve_mae_bias(kq, rows)
    return (bang(["Mô hình", "MAE", "RMSE", "WAPE", "R²", "Bias", "WAPE so với MA7", "WAPE < MA7 (nhóm hàng · cửa hàng)"], dong)
            + "\n\nIn đậm = tốt nhất mỗi cột · ⚠️ = lệch tổng lượng quá 10%.\n\n"
            "![MAE và Bias của từng mô hình](hinh/mae_bias.png)")


def khoi_chon(rows):
    ml = rows.drop(BASE)
    ma7 = rows.loc["MA7"]
    tieu_chi = [("MAE thấp nhất", "MAE", min), ("WAPE thấp nhất", "WAPE", min), ("RMSE thấp nhất", "RMSE", min),
                ("R² cao nhất", "R2", max), (r"\|Bias\| nhỏ nhất (đúng tổng lượng)", "Bias", None)]
    dong = []
    for ten, c, f in tieu_chi:
        key = (lambda d: d[c].abs().idxmin()) if f is None else (lambda d: d[c].idxmin() if f is min else d[c].idxmax())
        v = (lambda r: pt(rows.loc[r, c])) if c == "Bias" else (lambda r: so(rows.loc[r, c]))
        a, b = key(rows), key(ml)
        tot_hon = (abs(ml.loc[b, c]) < abs(ma7[c])) if f is None else (ml.loc[b, c] < ma7[c]) if f is min else (ml.loc[b, c] > ma7[c])
        dong.append([ten, f"{a} ({v(a)})", f"{b} ({v(b)})", "✔" if tot_hon else "✘"])
    dat = rows[rows["Bias"].abs() <= NGUONG_BIAS]
    ml_dat = dat.drop([r for r in BASE if r in dat.index])
    b = ml_dat["MAE"].idxmin() if len(ml_dat) else None
    dong.append([rf"**MAE thấp nhất trong các mô hình có \|Bias\| ≤ {NGUONG_BIAS:.0%} (đề xuất)**",
                 f"**{dat['MAE'].idxmin()}** ({so(dat['MAE'].min())})",
                 f"{b} ({so(ml_dat.loc[b, 'MAE'])})" if b else "không có",
                 ("✔" if ml_dat.loc[b, "MAE"] < ma7["MAE"] else "✘") if b else "✘"])
    return bang(["Tiêu chí", "Tốt nhất (gồm baseline)", "Mô hình học máy tốt nhất", "Hơn MA7?"], dong, trai=3)


def khoi_validation(kq, rows):
    dong, ghi_chu = [], []
    for t, m in kq.items():
        tv = m["tuning"].iloc[0]
        ma7v = m["info"]["baseline_val"]["MA7"]["MAE"]
        mt, ma7t = rows.loc[m["ten"], "MAE"], rows.loc["MA7", "MAE"]
        val_ok, test_ok = tv["MAE"] < ma7v, mt < ma7t
        cau_hinh = ", ".join(f"{k}={v}" for k, v in m["params"].items())
        dong.append([m["ten"], f"`{cau_hinh}`", len(m["tuning"]), so(tv["MAE"]), so(ma7v), so(mt), so(ma7t),
                     ("✔ thắng" if val_ok else "✘ thua") + " → " + ("✔ thắng" if test_ok else "✘ thua")])
        if val_ok != test_ok:
            ghi_chu.append(f"- ⚠️ {m['ten']} {'thắng' if val_ok else 'thua'} MA7 trên Validation nhưng "
                           f"{'thắng' if test_ok else 'thua'} trên Test — Validation chưa đại diện cho Test.")
    return (bang(["Mô hình", "Cấu hình chọn", "Số cấu hình thử", "MAE Val", "MA7 Val", "MAE Test", "MA7 Test",
                  "So với MA7: Val → Test"], dong, trai=2)
            + ("\n\n" + "\n".join(ghi_chu) if ghi_chu else ""))


def khoi_tong_luong(kq):
    m0 = next(iter(kq.values()))
    ten = [m["ten"] for m in kq.values()]
    ngay = m0["ngay"]
    dong = [["Trung bình / ngày toàn hệ thống", so(ngay["so_luong"].mean(), 0), so(ngay["tb_7"].mean(), 0)]
            + [so(m["ngay"]["du_bao"].mean(), 0) for m in kq.values()]]
    q0 = m0["quy_mo"]
    for b in q0.index:
        dong.append([f"Ô bán {b} ({so(q0.loc[b, 'so_o'], 0)} ô)", so(q0.loc[b, "so_luong"], 0),
                     so(q0.loc[b, "tb_7"], 0)] + [so(m["quy_mo"].loc[b, "du_bao"], 0) for m in kq.values()])
    mae = []
    for b in q0.index:
        v = [q0.loc[b, "MAE_MA7"]] + [m["quy_mo"].loc[b, "MAE"] for m in kq.values()]
        f = dam(v, lambda x: int(np.argmin(x)))
        mae.append([f"Ô bán {b}"] + [f(j, so(x)) for j, x in enumerate(v)])
    nhan_xet = []
    for m in kq.values():
        q = m["quy_mo"]
        hon = [b for b in q.index if q.loc[b, "MAE"] < q.loc[b, "MAE_MA7"]]
        kem = [b for b in q.index if b not in hon]
        nhan_xet.append(f"- **{m['ten']}** tốt hơn MA7 ở ô {', '.join(hon) or '—'}; kém hơn ở ô {', '.join(kem) or '—'}. "
                        f"Ô bán ≥ 20: dự báo {so(q.loc['≥ 20', 'du_bao'], 0)} so với thực tế "
                        f"{so(q.loc['≥ 20', 'so_luong'], 0)} (MA7 {so(q.loc['≥ 20', 'tb_7'], 0)}).")
    ve_theo_ngay(kq)
    return ("![Tổng số lượng mỗi ngày tháng 3](hinh/tong_theo_ngay.png)\n\n"
            "**Tổng lượng (sản phẩm), chia theo lượng bán thực tế của ô:**\n\n"
            + bang(["Tổng lượng", "Thực tế", "MA7"] + ten, dong)
            + "\n\n**MAE theo quy mô ô** (in đậm = tốt nhất mỗi dòng):\n\n"
            + bang(["Quy mô ô", "MA7"] + ten, mae) + "\n\n" + "\n".join(nhan_xet))


def khoi_nhom(kq, khoa, ten_cot, n_dau):
    """Bảng WAPE theo nhóm hàng (khoa=cat) hoặc khu vực (khoa=store) — mỗi mô hình một cột, cùng MA7."""
    if khoa == "cat":
        g = {t: m["cat"] for t, m in kq.items()}
    else:
        g = {t: m["store"].groupby("Khu vực").apply(lambda s: pd.Series({
            "so_luong_thuc": s.so_luong_thuc.sum(), "WAPE": wape_gop(s, "WAPE"), "WAPE_MA7": wape_gop(s, "WAPE_MA7")}),
            include_groups=False) for t, m in kq.items()}
    g0 = next(iter(g.values())).sort_values("so_luong_thuc", ascending=False)
    dau, con = g0.index[:n_dau], g0.index[n_dau:]
    dong = []
    nhom = [(x, [x]) for x in dau] + ([(f"{len(con)} {ten_cot.lower()} còn lại", list(con))] if len(con) else [])
    for nhan, idx in nhom:
        thuc = g0.loc[idx, "so_luong_thuc"].sum()
        v = [wape_gop(g0.loc[idx], "WAPE_MA7")] + [wape_gop(g[t].loc[idx], "WAPE") for t in kq]
        f = dam(v, lambda x: int(np.argmin(x)))
        dong.append([nhan, so(thuc, 0)] + [f(j, so(x)) for j, x in enumerate(v)])
    dem = ["**Thắng MA7**", "", ""] + [f"{(g[t].WAPE < g[t].WAPE_MA7).sum()}/{len(g[t])}" for t in kq]
    return bang([ten_cot, "Thực tế", "MA7"] + [m["ten"] for m in kq.values()], dong + [dem]) + \
        f"\n\nGiá trị là WAPE (thấp hơn = tốt hơn), in đậm = tốt nhất mỗi dòng · *Thắng MA7* = số {ten_cot.lower()} có WAPE thấp hơn MA7."


def khoi_nhom_hang(kq):
    out = khoi_nhom(kq, "cat", "Nhóm hàng", 8) + "\n"
    for m in kq.values():
        c = m["cat"]
        top = c.sort_values("so_luong_thuc", ascending=False).head(3)
        ty = c.loc[c.WAPE < c.WAPE_MA7, "so_luong_thuc"].sum() / c.so_luong_thuc.sum()
        out += (f"\n- **{m['ten']}** thắng MA7 ở {(c.WAPE < c.WAPE_MA7).sum()}/{len(c)} nhóm, chiếm {ty:.0%} lượng bán; "
                + ", ".join(f"{n} {so(r.WAPE)} vs {so(r.WAPE_MA7)} {'✔' if r.WAPE < r.WAPE_MA7 else '✘'}"
                            for n, r in top.iterrows()) + ".")
    return out


def khoi_dac_trung(kq):
    pct = pd.DataFrame({m["ten"]: m["imp"] / m["imp"].sum() for m in kq.values()}).fillna(0)
    dong = [[n] + [f"{pct.loc[[c for c in cs if c in pct.index], t].sum():.0%}" for t in pct.columns]
            for n, cs in NHOM_DT.items()]
    top = [["Top 3 đặc trưng"] + ["<br>".join(f"`{c}` {v:.1%}".replace(".", ",") for c, v in pct[t].nlargest(3).items())
                                   for t in pct.columns]]
    return (bang(["Nhóm đặc trưng"] + list(pct.columns), dong + top)
            + "\n\nTỷ trọng mức quan trọng (Ridge/Poisson: trị tuyệt đối hệ số trên đặc trưng chuẩn hoá), mỗi cột cộng lại 100%.")


def khoi_r2(kq, rows):
    am = rows[rows["R2"] < 0]
    if am.empty:
        return "Không có mô hình hay baseline nào có R² âm trên tập Test."
    cot = {"Naive (lag 1)": "lag_1", "Seasonal Naive (lag 7)": "lag_7", "MA7": "tb_7"}
    cot.update({m["ten"]: f"du_bao_{m['p']}" for m in kq.values()})
    m0 = next(iter(kq.values()))
    dong = []
    for r in am.index:
        src = m0 if r in BASE else next(m for m in kq.values() if m["ten"] == r)
        s = src["info"]["sai_so_top10"][cot[r]]
        dong.append([r, so(am.loc[r, "R2"]), f"{s['top10_o']:.0%}", f"{s['cua_hang']} ({s['ty_trong_cua_hang']:.0%})"])
    return ("R² = 1 − (tổng bình phương sai số) / (tổng bình phương độ lệch so với trung bình Test). **R² < 0 nghĩa là dự báo "
            "tệ hơn việc đoán mọi ô bằng trung bình tập Test** — không phải lỗi tính, và không cắt về 0.\n\n"
            + bang(["Có R² âm", "R²", "10 ô sai nhiều nhất chiếm", "Cửa hàng chiếm nhiều nhất"], dong))


def khoi_nhan_xet(kq):
    out = []
    for t, m in kq.items():
        nb = os.path.join(ROOT, m["notebook"])
        text = "*Notebook chưa có mục `## 5. Kết quả và thảo luận`.*"
        if os.path.exists(nb):
            for c in json.load(open(nb, encoding="utf-8"))["cells"]:
                s = "".join(c["source"])
                if c["cell_type"] == "markdown" and s.startswith("## 5. Kết quả và thảo luận"):
                    i = s.find("**Thảo luận.**")
                    text = (s[i:] if i >= 0 else s.split("\n", 1)[-1]).strip()
        out.append(f"<details>\n<summary><b>Module {m['module']} · {m['ten']}</b> — trích mục 5 của "
                   f"<code>{m['notebook']}</code> (chạy {m['info']['ngay_chay']})</summary>\n\n{text}\n\n</details>")
    return "\n\n".join(out)


def khoi_nguon(kq):
    dong = []
    for t, (p, ten, module, nb) in MO_HINH.items():
        if t in kq:
            i = kq[t]["info"]
            dong.append([ten, module, i["ngay_chay"], f"`bao_cao/du_lieu/{t}/`", f"`outputs/{t}/`", f"`{nb}`"])
        else:
            dong.append([ten, module, "chưa nộp", "—", f"`outputs/{t}/`", f"`{nb}`"])
    return bang(["Mô hình", "Module", "Ngày chạy", "Số liệu đã lọc", "Kết quả đầy đủ (không commit)", "Notebook"], dong, trai=6)


# ---------- biểu đồ ----------
def _truc(ax):
    ax.set_facecolor(NEN)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(color=LUOI, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: so(v, 0 if abs(v) >= 10 or v == 0 else 3)))


def ve_mae_bias(kq, rows):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=NEN)
    _truc(ax)
    ax.axvspan(-NGUONG_BIAS * 100, NGUONG_BIAS * 100, color="#e8f3e8", zorder=0, label=f"|Bias| ≤ {NGUONG_BIAS:.0%}")
    ten_mau = {m["ten"]: MAU[t] for t, m in kq.items()}
    for r, x in rows.iterrows():
        mau = ten_mau.get(r, XAM)
        ax.scatter(x["Bias"] * 100, x["MAE"], s=70, color=mau, edgecolor=NEN, linewidth=2, zorder=3)
        ax.annotate(r, (x["Bias"] * 100, x["MAE"]), xytext=(7, 4), textcoords="offset points", fontsize=9, color=MUC)
    ax.set_xlabel("Bias — lệch tổng lượng (%)", color="#52514e")
    ax.set_ylabel("MAE trên Test (thấp hơn = tốt hơn)", color="#52514e")
    ax.set_title("Độ chính xác từng ô (MAE) và độ đúng tổng lượng (Bias)", color=MUC, loc="left", fontsize=11)
    lo, hi = min(rows["Bias"].min() * 100, -15) - 5, max(rows["Bias"].max() * 100, 15) + 15
    ax.set_xlim(lo, hi)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{HINH}/mae_bias.png", dpi=150, facecolor=NEN)
    plt.close(fig)


def ve_theo_ngay(kq):
    m0 = next(iter(kq.values()))
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=NEN)
    _truc(ax)
    ax.plot(m0["ngay"].index, m0["ngay"]["so_luong"], color=MUC, lw=2.2, label="Thực tế", zorder=4)
    ax.plot(m0["ngay"].index, m0["ngay"]["tb_7"], color=XAM, lw=1.6, label="MA7", zorder=3)
    for t, m in kq.items():
        ax.plot(m["ngay"].index, m["ngay"]["du_bao"], color=MAU[t], lw=2, label=m["ten"], zorder=3)
    ax.set_ylim(0, None)
    ax.set_ylabel("Sản phẩm / ngày", color="#52514e")
    ax.set_title("Tổng số lượng mỗi ngày tháng 3/2025 — thực tế và dự báo", color=MUC, loc="left", fontsize=11)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=len(kq) + 2)
    fig.tight_layout()
    fig.savefig(f"{HINH}/tong_theo_ngay.png", dpi=150, facecolor=NEN)
    plt.close(fig)


# ---------- ghi vào BAO_CAO.md ----------
def ghi(khoi):
    s = open(FILE, encoding="utf-8").read()
    for ten, noi_dung in khoi.items():
        pat = re.compile(rf"(<!-- TU_DONG:{ten} -->).*?(<!-- /TU_DONG -->)", re.S)
        assert pat.search(s), f"BAO_CAO.md thiếu khối <!-- TU_DONG:{ten} --> … <!-- /TU_DONG -->"
        s = pat.sub(lambda m: f"{m.group(1)}\n{noi_dung.strip()}\n{m.group(2)}", s)
    open(FILE, "w", encoding="utf-8").write(s)


def main():
    kq = doc()
    if not kq:
        sys.exit(f"Chưa có mô hình nào trong {DU_LIEU} — chạy nop_ket_qua(...) ở cuối notebook mô hình trước.")
    os.makedirs(HINH, exist_ok=True)
    loi = kiem_tra(kq)
    rows, thang = bang_chinh(kq)
    ghi({"trang_thai": khoi_trang_thai(kq, loi), "tom_tat": khoi_tom_tat(kq, rows),
         "bang_test": khoi_bang_test(kq, rows, thang), "chon_mo_hinh": khoi_chon(rows),
         "validation": khoi_validation(kq, rows), "tong_luong": khoi_tong_luong(kq),
         "nhom_hang": khoi_nhom_hang(kq), "khu_vuc": khoi_nhom(kq, "store", "Khu vực", 20),
         "dac_trung": khoi_dac_trung(kq), "r2": khoi_r2(kq, rows), "nhan_xet": khoi_nhan_xet(kq),
         "nguon": khoi_nguon(kq)})
    print(f"Đã cập nhật {FILE} với {len(kq)} mô hình: {', '.join(m['ten'] for m in kq.values())}")
    for x in loi:
        print("⚠️", x)


if __name__ == "__main__":
    main()
