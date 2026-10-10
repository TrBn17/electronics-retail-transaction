"""Huấn luyện và xuất kết quả LightGBM cho Module 5 trên bảng chung của Module 3."""

import json
import os
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module3"))
from module3_preprocess import (  # noqa: E402
    OUT_DIR, SEED, bang_baseline, chia_tap, danh_gia, doc_bang_ngay,
    loi_theo_nhom, nop_ket_qua, tao_dac_trung,
)

OUT_LGBM = Path(OUT_DIR) / "lightgbm"
CAT = ["Mã cửa hàng_code", "Nhóm hàng_code", "Vùng_code"]
OBJECTIVES = [("regression", "l2"), ("regression_l1", "l1"),
              ("poisson", "poisson"), ("tweedie", "tweedie")]
SPACE = {"num_leaves": [15, 31, 63], "min_child_samples": [20, 50, 100],
         "learning_rate": [0.03, 0.05, 0.1], "reg_lambda": [1, 3, 10]}
N_PER_OBJECTIVE, MAX_ITER, STOPPING_ROUNDS = 3, 1500, 75


def load_data():
    """Giữ nguyên dữ liệu, đặc trưng, mốc chia và baseline của các mô hình khác."""
    OUT_LGBM.mkdir(parents=True, exist_ok=True)
    daily = doc_bang_ngay()
    d, feats = tao_dac_trung(daily)
    fit, val, tr, te = chia_tap(d)
    assert set(CAT).issubset(feats)
    assert all(d[c].min() >= 0 for c in CAT), "LightGBM coi mã âm là missing"
    base_val = pd.DataFrame(bang_baseline(val)).T
    rows = bang_baseline(te)
    print(f"Bảng mô hình: {len(daily):,} dòng; Validation MA7={base_val.loc['MA7', 'MAE']:.4f}")
    return daily, feats, fit, val, tr, te, base_val, rows


def tune(feats, fit, val):
    """Early stopping theo loss tương ứng, chọn cấu hình có MAE Validation thấp nhất."""
    sampled = list(ParameterSampler(SPACE, n_iter=N_PER_OBJECTIVE, random_state=SEED))
    params = [{"objective": objective, "metric": metric,
               **({"tweedie_variance_power": 1.5} if objective == "tweedie" else {}), **p}
              for objective, metric in OBJECTIVES for p in sampled]
    log = []
    for i, p in enumerate(params):
        started = time.time()
        model = lgb.LGBMRegressor(**p, n_estimators=MAX_ITER, random_state=SEED,
                                  n_jobs=4, verbosity=-1, importance_type="gain")
        model.fit(fit[feats], fit["so_luong"], eval_set=[(val[feats], val["so_luong"])],
                  categorical_feature=CAT,
                  callbacks=[lgb.early_stopping(STOPPING_ROUNDS, verbose=False)])
        pred = np.clip(model.predict(val[feats]), 0, None)
        scores = danh_gia(val["so_luong"], pred)
        iterations = int(model.best_iteration_ or model.n_iter_)
        log.append({"id": i, **p, "so_vong": iterations, **scores,
                    "giay": round(time.time() - started, 1)})
        print(f"{i + 1:2}/{len(params)} MAE={scores['MAE']:.4f} "
              f"Bias={scores['Bias']:+.2%} vòng={iterations:4} {p}", flush=True)
    tuning = pd.DataFrame(log).sort_values("MAE").reset_index(drop=True)
    tuning.to_csv(OUT_LGBM / "lgbm_tuning.csv", index=False, encoding="utf-8-sig")
    best = {**params[int(tuning.loc[0, "id"])],
            "n_estimators": int(tuning.loc[0, "so_vong"])}
    (OUT_LGBM / "lgbm_best_params.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Cấu hình chọn:", best)
    return tuning, best


def train_and_score(feats, tr, te, rows, best):
    """Train lại với fit + Validation, chấm đúng một lần trên Test tháng 3."""
    model = lgb.LGBMRegressor(**best, random_state=SEED, n_jobs=4,
                              verbosity=-1, importance_type="gain")
    model.fit(tr[feats], tr["so_luong"], categorical_feature=CAT)
    te = te.copy()
    te["du_bao_lgbm"] = np.clip(model.predict(te[feats]), 0, None)
    rows = {**rows, "LightGBM": danh_gia(te["so_luong"], te["du_bao_lgbm"])}
    result = pd.DataFrame(rows).T
    result["WAPE so với MA7"] = result["WAPE"] / result.loc["MA7", "WAPE"] - 1
    result.to_csv(OUT_LGBM / "lgbm_metrics.csv", encoding="utf-8-sig")
    joblib.dump(model, OUT_LGBM / "lgbm_model.joblib")
    print(result)
    return model, te, result


def analyze(model, feats, te):
    """Xuất importance, sai số theo nhóm và dự báo theo đúng chuẩn chung."""
    imp = pd.Series(model.feature_importances_, index=feats, name="importance").sort_values()
    imp.to_csv(OUT_LGBM / "lgbm_feature_importance.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color="#e87ba4")
    ax.set_title("LightGBM — mức quan trọng đặc trưng (gain)")
    fig.tight_layout()
    fig.savefig(OUT_LGBM / "lgbm_feature_importance.png", dpi=150)
    plt.close(fig)

    err_cat = loi_theo_nhom(te, "du_bao_lgbm", "Nhóm hàng")
    err_store = loi_theo_nhom(te, "du_bao_lgbm", ["Mã cửa hàng", "Khu vực"])
    err_cat.to_csv(OUT_LGBM / "lgbm_error_by_category.csv", encoding="utf-8-sig")
    err_store.to_csv(OUT_LGBM / "lgbm_error_by_store.csv", encoding="utf-8-sig")
    print(f"LightGBM thắng MA7 ở {(err_cat.WAPE < err_cat.WAPE_MA7).sum()}/{len(err_cat)} nhóm, "
          f"{(err_store.WAPE < err_store.WAPE_MA7).sum()}/{len(err_store)} cửa hàng")

    day = te.groupby("Ngày")[["so_luong", "du_bao_lgbm", "tb_7"]].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(day.index, day["so_luong"], color="#1f2a2a", lw=2, label="Thực tế")
    ax.plot(day.index, day["du_bao_lgbm"], color="#e87ba4", lw=2, label="LightGBM")
    ax.plot(day.index, day["tb_7"], color="#eb6834", lw=1.5, ls="--", label="MA7")
    ax.set_title("Tổng số lượng tháng 3 — thực tế và dự báo")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_LGBM / "lgbm_thuc_te_vs_du_bao.png", dpi=150)
    plt.close(fig)
    te[["Mã cửa hàng", "Nhóm hàng", "Khu vực", "Vùng", "Ngày", "so_luong",
        "du_bao_lgbm", "lag_1", "lag_7", "tb_7"]].to_csv(
            OUT_LGBM / "lgbm_predictions.csv", index=False, encoding="utf-8-sig")
    return imp, err_cat, err_store, day


def run():
    daily, feats, fit, val, tr, te, base_val, rows = load_data()
    tuning, best = tune(feats, fit, val)
    model, te, result = train_and_score(feats, tr, te, rows, best)
    imp, err_cat, err_store, day = analyze(model, feats, te)
    nop_ket_qua("lightgbm", val)
    return {"daily": daily, "feats": feats, "fit": fit, "val": val, "tr": tr,
            "te": te, "base_val": base_val, "tuning": tuning, "best": best,
            "model": model, "result": result, "imp": imp, "err_cat": err_cat,
            "err_store": err_store, "day": day}


if __name__ == "__main__":
    run()
