# -*- coding: utf-8 -*-
"""H2 Train-only OLS interaction ablation and fold comparison.

Uses exactly the four-scenario common sample from evaluate_rankic.py.
DART/6m/9m are provisional, NOT fully PIT verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.h2.evaluation.evaluate_rankic import (
    ROOT, PROCESSED, FOLDS, read_csv, norm_code,
    calc_ic,
)

from src.h2.features.transforms import monthly_standardize
from src.h2.models.estimators import model_instance

KEY = ["stock_id", "feature_month"]
ABLATIONS = {
    "size_only": ["log_size"],
    "no_interaction": ["policy", "debt", "log_size"],
    "full_interaction": ["policy", "debt", "policy_x_debt", "log_size"],
}


def path(root, p):
    return p if p.is_absolute() else root / p


def read_debt_file(filepath, value_col, output_col):
    x = read_csv(filepath)
    cols = set(KEY + [value_col, "prediction_date", "source_fiscal_date", "source_available_date"])
    if value_col == "proxy_masked":
        cols = set(KEY + [value_col, "prediction_date", "fiscal_date", "proxy_available_date"])
    missing = cols.difference(x.columns)
    if missing:
        raise ValueError(f"{filepath.name}: 누락 컬럼 {sorted(missing)}")
    x["stock_id"] = norm_code(x["stock_id"])
    x["feature_month"] = x["feature_month"].str.slice(0, 7)
    if x[KEY].isna().any().any() or x.duplicated(KEY).any():
        raise ValueError(f"{filepath.name}: 종목-월 키 누락/중복")
    raw = pd.to_numeric(x[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    prediction = pd.to_datetime(x["prediction_date"], errors="coerce")
    expected = pd.to_datetime(x["feature_month"] + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
    f_col = "fiscal_date" if value_col == "proxy_masked" else "source_fiscal_date"
    a_col = "proxy_available_date" if value_col == "proxy_masked" else "source_available_date"
    fiscal = pd.to_datetime(x[f_col], errors="coerce")
    available = pd.to_datetime(x[a_col], errors="coerce")
    valid = raw.notna()
    invalid = valid & (prediction.isna() | expected.isna() | prediction.ne(expected)
                       | fiscal.isna() | available.isna() | fiscal.gt(prediction)
                       | available.gt(prediction))
    if invalid.any():
        raise ValueError(f"{filepath.name}: 시점 위반 {int(invalid.sum())}건")
    x[output_col] = raw
    return x[KEY + [output_col]]


def prepare_panel(panel_file, lag_dir, hybrid_file):
    p = read_csv(panel_file)
    required = set(KEY + ["target_month", "return_t1", "market_cap",
                          "debt_ratio_raw", "final_rule_v2_sensitivity"])
    if required.difference(p.columns):
        raise ValueError(f"패널 컬럼 누락: {sorted(required.difference(p.columns))}")
    p["stock_id"] = norm_code(p["stock_id"])
    for c in ("feature_month", "target_month"):
        p[c] = p[c].str.slice(0, 7)
    if p[KEY].isna().any().any() or p.duplicated(KEY).any():
        raise ValueError("패널 종목-월 키 누락/중복")
    expected_target = (pd.PeriodIndex(p["feature_month"], freq="M") + 1).astype(str)
    if not (p["target_month"].to_numpy() == expected_target.to_numpy()).all():
        raise ValueError("특성월 → 목표월 시차가 일치하지 않습니다")
    p = p.loc[p["target_month"].between("2001-01", "2021-12")].copy()
    p["policy"] = pd.to_numeric(p["final_rule_v2_sensitivity"], errors="coerce")
    p["return_t1"] = pd.to_numeric(p["return_t1"], errors="coerce")
    market_cap = pd.to_numeric(p["market_cap"], errors="coerce")
    p["log_size"] = np.log(market_cap.where(market_cap > 0))
    p["debt_legacy"] = pd.to_numeric(p["debt_ratio_raw"], errors="coerce")
    for lag in ("6m", "9m"):
        file = lag_dir / f"h2_debt_lag_{lag}_fiscal_screened_2001_2021.csv"
        debt = read_debt_file(file, "proxy_masked", f"debt_{lag}")
        p = p.merge(debt, on=KEY, how="left", validate="one_to_one")
    debt = read_debt_file(hybrid_file, "debt_final_raw", "debt_hybrid")
    p = p.merge(debt, on=KEY, how="left", validate="one_to_one")
    for scenario in ("legacy", "6m", "9m", "hybrid"):
        p[f"z_{scenario}"] = monthly_standardize(p, f"debt_{scenario}")
    common = p[["policy", "log_size", "return_t1", "z_legacy", "z_6m", "z_9m", "z_hybrid"]].notna().all(axis=1)
    if "eligible" in p.columns:
        common &= p["eligible"].astype("string").str.lower().isin(["true", "1"])
    return p.loc[common].copy()


def build_ablation(p, min_stocks):
    records, coefficients = [], []
    for scenario in ("6m", "hybrid"):
        df = p.copy()
        df["debt"] = df[f"z_{scenario}"]
        df["policy_x_debt"] = df["policy"] * df["debt"]
        for fold, (start, end) in enumerate(FOLDS, 1):
            train = df.loc[df["target_month"] < start]
            val = df.loc[df["target_month"].between(start, end)]
            if len(train) < 100 or len(val) < 100:
                raise ValueError(f"표본 부족: {scenario} fold={fold}")
            for variant, features in ABLATIONS.items():
                model = model_instance("OLS")  # 원래 OLS와 동일한 파이프라인
                x_train = train[features].replace([np.inf, -np.inf], np.nan)
                x_val = val[features].replace([np.inf, -np.inf], np.nan)
                model.fit(x_train, train["return_t1"])
                pred = model.predict(x_val)
                for month, n, ic in calc_ic(val, pred, min_stocks):
                    records.append({"scenario": scenario, "fold": fold,
                                    "target_month": month, "variant": variant,
                                    "n_stocks": n, "rank_ic": ic})
                if variant == "full_interaction":
                    lr = model.named_steps["linearregression"]
                    row = {"scenario": scenario, "fold": fold, "n_train": len(train),
                           "n_val": len(val), "intercept_scaled": float(lr.intercept_)}
                    row.update({f"coef_{name}": float(coef)
                                for name, coef in zip(features, lr.coef_)})
                    coefficients.append(row)
            print(f"{scenario:7s} fold={fold} Train={len(train):,} Val={len(val):,}", flush=True)
    return pd.DataFrame(records), pd.DataFrame(coefficients)


def summarize(monthly):
    summary = (monthly.groupby(["scenario", "variant"], as_index=False)
               .agg(mean_ic=("rank_ic", "mean"), median_ic=("rank_ic", "median"),
                    positive_share=("rank_ic", lambda s: float((s > 0).mean())),
                    n_months=("target_month", "nunique"),
                    mean_stocks=("n_stocks", "mean")))
    w = monthly.pivot(index=["scenario", "fold", "target_month"],
                      columns="variant", values="rank_ic").reset_index()
    if w[["size_only", "no_interaction", "full_interaction"]].isna().any().any():
        raise ValueError("IC 평가 월이 일부 누락되어 비교할 수 없습니다")
    w["debt_added_delta"] = w["no_interaction"] - w["size_only"]
    w["interaction_delta"] = w["full_interaction"] - w["no_interaction"]
    per_fold = (w.groupby(["scenario", "fold"], as_index=False)
                .agg(size_only_ic=("size_only", "mean"),
                     no_interaction_ic=("no_interaction", "mean"),
                     full_interaction_ic=("full_interaction", "mean"),
                     debt_added_delta=("debt_added_delta", "mean"),
                     interaction_delta=("interaction_delta", "mean"),
                     interaction_improved_months=("interaction_delta", lambda s: int((s > 1e-12).sum())),
                     interaction_worse_months=("interaction_delta", lambda s: int((s < -1e-12).sum())),
                     interaction_unchanged_months=("interaction_delta", lambda s: int((s.abs() <= 1e-12).sum())),
                     n_months=("target_month", "size")))
    return summary, w, per_fold


def fold_compare(ablation_wide):
    w = ablation_wide.pivot(index=["fold", "target_month"],
                            columns="scenario", values="full_interaction").reset_index()
    w["hybrid_minus_6m"] = w["hybrid"] - w["6m"]
    report = (w.groupby("fold", as_index=False)
              .agg(ic_6m=("6m", "mean"), ic_hybrid=("hybrid", "mean"),
                   delta=("hybrid_minus_6m", "mean"),
                   improved_months=("hybrid_minus_6m", lambda s: int((s > 1e-12).sum())),
                   worse_months=("hybrid_minus_6m", lambda s: int((s < -1e-12).sum())),
                   unchanged_months=("hybrid_minus_6m", lambda s: int((s.abs() <= 1e-12).sum())),
                   n_months=("target_month", "size")))
    report["period"] = [f"{s}~{e}" for s, e in FOLDS]
    return report


def reference_check(out, existing_dir, monthly):
    ref_file = existing_dir / "monthly_rank_ic.csv"
    if not ref_file.is_file():
        return {"existing_result_found": False}
    ref = pd.read_csv(ref_file, dtype={"target_month": "string"})
    ref = ref.loc[(ref["sample"] == "common")
                  & (ref["model"] == "OLS_policy_debt_size")
                  & ref["scenario"].isin(["6m", "hybrid"]),
                  ["scenario", "target_month", "rank_ic"]]
    current = monthly.loc[monthly["variant"] == "full_interaction",
                          ["scenario", "target_month", "rank_ic"]]
    m = current.merge(ref, on=["scenario", "target_month"], how="outer",
                      suffixes=("_new", "_previous"), indicator=True)
    max_diff = float((m["rank_ic_new"] - m["rank_ic_previous"]).abs().max())
    result = {"existing_result_found": True,
              "months_matched": int((m["_merge"] == "both").sum()),
              "months_unmatched": int((m["_merge"] != "both").sum()),
              "max_abs_ic_reproduction_error": max_diff,
              "reproduced": bool(m["_merge"].eq("both").all() and max_diff <= 1e-8)}
    if not result["reproduced"]:
        print("주의: 기존 IC 결과와 다릅니다. 입력 파일·표본·코드를 확인하세요.", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--panel", type=Path, default=PROCESSED / "dataset.csv")
    parser.add_argument("--lag-dir", type=Path, default=PROCESSED / "debt")
    parser.add_argument("--hybrid", type=Path, default=PROCESSED / "debt_hybrid" / "h2_debt_hybrid_candidate_2001_2021.csv")
    parser.add_argument("--previous", type=Path, default=Path("results/rankic_hybrid_comparison"))
    parser.add_argument("--out", type=Path, default=Path("results/h2_interaction_check"))
    parser.add_argument("--min-stocks", type=int, default=20)
    a = parser.parse_args()
    root = a.project.resolve()
    out = path(root, a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = prepare_panel(path(root, a.panel), path(root, a.lag_dir), path(root, a.hybrid))
    monthly, coefficients = build_ablation(p, a.min_stocks)
    summary, wide, per_fold = summarize(monthly)
    fold = fold_compare(wide)
    reproduction = reference_check(out, path(root, a.previous), monthly)
    for name, df in {
        "interaction_ablation_monthly.csv": monthly,
        "interaction_ablation_summary.csv": summary,
        "interaction_ablation_fold.csv": per_fold,
        "fold_hybrid_vs_6m.csv": fold,
        "ols_full_coefficients.csv": coefficients,
    }.items():
        df.to_csv(out / name, index=False, encoding="utf-8-sig")
    diagnostics = {"train_only": True, "four_way_common_rows": int(len(p)),
                   "monthly_ic_rows": len(monthly), "reference": reproduction,
                   "fully_pit_verified": False,
                   "notes": "Train-only; same four-way common sample. Exploratory IC, not causal evidence."}
    (out / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[Fold별 Hybrid - 6m]\n", fold.to_string(index=False))
    print("\n[상호작용항 제거 실험 - Fold]\n", per_fold.to_string(index=False))
    print("\n[검증 상태]\n", json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("\n저장:", out)


if __name__ == "__main__":
    main()
