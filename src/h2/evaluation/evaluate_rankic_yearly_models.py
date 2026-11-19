# -*- coding: utf-8 -*-
"""H2: TRAIN-only annual Expanding Window, 6 models x 2 debt sources.

Place this file in src/h2/evaluation/evaluate_rankic_yearly_models.py
Run from project root:
    python -m src.h2.evaluation.evaluate_rankic_yearly_models

This script reuses the existing H2 data preparation and OLS/Ridge/LightGBM/
XGBoost configurations. It does NOT read 2022-2025 target returns.
The DART/DataGuide hybrid has NOT been fully point-in-time verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.h2.data_io import ROOT, PROCESSED, sha256
from src.h2.evaluation.evaluate_rankic import resolve
from src.h2.evaluation.evaluate_rankic_yearly import (
    FEATURE_COLS,
    load_hybrid,
    load_panel,
    load_six_month,
)
from src.h2.features.transforms import monthly_standardize
from src.h2.models.estimators import model_instance

ALL_MODELS = ("OLS", "Ridge", "ElasticNet", "RandomForest", "LightGBM", "XGBoost")
SCENARIOS = ("6m", "hybrid")
FEATURE_NAMES = ("policy", "debt", "policy_x_debt", "log_size")
SEED = 2026


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--project", type=Path, default=ROOT)

    p.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/final/dataset.csv"),
    )

    p.add_argument(
        "--lag-6m",
        type=Path,
        default=Path(
            "data/processed/final/debt/"
            "h2_debt_lag_6m_fiscal_screened_2001_2021.csv"
        ),
    )

    p.add_argument(
        "--hybrid",
        type=Path,
        default=Path(
            "data/processed/final/debt_hybrid/"
            "h2_debt_hybrid_candidate_2001_2021.csv"
        ),
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("results/rankic_yearly_all_models_2006_2021"),
    )
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    p.add_argument("--min-stocks", type=int, default=20)
    p.add_argument("--start-year", type=int, default=2006)
    p.add_argument("--end-year", type=int, default=2021)
    p.add_argument("--rf-trees", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument(
        "--previous-ols-summary", type=Path,
        default=Path("results/rankic_yearly_2006_2021/summary.json"),
    )
    return p.parse_args()


def make_estimator(name, a):
    if name in {"OLS", "Ridge", "LightGBM", "XGBoost"}:
        return model_instance(name)
    if name == "ElasticNet":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNet(alpha=1e-5, l1_ratio=0.5, max_iter=10000, random_state=SEED),
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=a.rf_trees,
            max_depth=8,
            min_samples_leaf=80,
            max_features=1.0,
            n_jobs=a.n_jobs,
            random_state=SEED,
        )
    raise KeyError(name)


def load_common_panel(a, paths):
    p = load_panel(paths["panel"])
    p = p.merge(load_six_month(paths["6m"]),
                on=["stock_id", "feature_month"], how="left", validate="one_to_one")
    p = p.merge(load_hybrid(paths["hybrid"]),
                on=["stock_id", "feature_month"], how="left", validate="one_to_one")

    # Identical preprocessing to the previous yearly OLS experiment.
    for scenario in SCENARIOS:
        p[f"z_{scenario}"] = monthly_standardize(p, f"debt_{scenario}")

    common = p[["return_t1", "policy", "log_size", "z_6m", "z_hybrid"]].notna().all(axis=1)
    if "eligible" in p.columns:
        common &= p["eligible"].astype("string").str.lower().isin(["true", "1"])
    d = p.loc[common].copy()
    if d.empty:
        raise ValueError("6개월 지연/Hybrid 공통 표본이 없습니다")
    if d["target_month"].lt("2001-01").any() or d["target_month"].gt("2021-12").any():
        raise ValueError("Train 이외의 target_month가 포함됨")
    return d


def monthly_rank_ic(val, preds, months, min_stocks):
    """Always record every month, including undefined IC (e.g. constant predictions).

    Undefined IC stays NaN; it is NOT silently dropped or labeled a real zero IC.
    A separate conservative 'zero-fill' summary counts undefined ranking as 0.
    """
    # Sort by source row order; only one prediction per validation row.
    v = val[["target_month", "stock_id", "return_t1"]].copy()
    v["prediction"] = np.asarray(preds, dtype=float)
    rows = []
    for month in months:
        g = v.loc[v["target_month"].eq(month)]
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction", "return_t1"])
        n = len(g)
        ic = float("nan")
        issue = ""
        if n < min_stocks:
            issue = "insufficient_stocks"
        elif g["prediction"].nunique() < 2:
            issue = "constant_prediction"
        elif g["return_t1"].nunique() < 2:
            issue = "constant_target"
        else:
            # Spearman corr == Pearson corr of average ranks, pandas handles ties.
            ic = float(g["prediction"].corr(g["return_t1"], method="spearman"))
            if not np.isfinite(ic):
                ic = float("nan")
                issue = "undefined_correlation"
        rows.append({"target_month": month, "n_stocks": n,
                     "rank_ic": ic, "issue": issue})
    return rows


def format_ic(value):
    return "NaN" if pd.isna(value) else f"{value:+.6f}"


def save_results(a, output, d, monthly, folds, paths, evaluation_years):
    m = pd.DataFrame(monthly)
    f = pd.DataFrame(folds)
    expected_rows = len(evaluation_years) * 12 * len(a.models) * len(SCENARIOS)
    if len(m) != expected_rows:
        raise AssertionError(f"월별 결과 행 수 오류: {len(m)} != {expected_rows}")
    if m.duplicated(["model", "scenario", "target_month"]).any():
        raise AssertionError("모델/시나리오/목표월 중복 결과")

    # All models are evaluated on the same stock-month rows in every fold.
    summary = (m.groupby(["model", "scenario"], as_index=False)
               .agg(n_months=("target_month", "size"),
                    n_valid_months=("rank_ic", "count"),
                    mean_valid_rank_ic=("rank_ic", "mean"),
                    median_valid_rank_ic=("rank_ic", "median"),
                    std_valid_rank_ic=("rank_ic", "std"),
                    positive_valid_months=("rank_ic", lambda s: int((s > 0).sum())),
                    negative_valid_months=("rank_ic", lambda s: int((s < 0).sum()))))
    # Explicit evaluation policy; not confused with observed IC.
    filled = (m.assign(rank_ic_zero_filled=m["rank_ic"].fillna(0.0))
              .groupby(["model", "scenario"], as_index=False)
              ["rank_ic_zero_filled"].mean()
              .rename(columns={"rank_ic_zero_filled": "mean_ic_undefined_as_zero"}))
    summary = summary.merge(filled, on=["model", "scenario"], validate="one_to_one")
    summary["n_undefined_months"] = summary["n_months"] - summary["n_valid_months"]
    summary["positive_share_all_months"] = (summary["positive_valid_months"] / summary["n_months"])
    summary = summary.sort_values(["mean_ic_undefined_as_zero", "model", "scenario"],
                                  ascending=[False, True, True]).reset_index(drop=True)

    yearly = (m.groupby(["model", "scenario", "validation_year"], as_index=False)
              .agg(mean_rank_ic=("rank_ic", "mean"),
                   n_valid_months=("rank_ic", "count"),
                   n_months=("rank_ic", "size")))

    # Same model, same month, paired hybrid minus 6m comparison.
    w = m.pivot(index=["model", "target_month"], columns="scenario", values="rank_ic").reset_index()
    w["hybrid_minus_6m"] = w["hybrid"] - w["6m"]
    tolerance = 1e-12
    w["status"] = np.select(
        [w["hybrid_minus_6m"].gt(tolerance),
         w["hybrid_minus_6m"].lt(-tolerance),
         w["hybrid_minus_6m"].isna()],
        ["hybrid_better", "six_month_better", "undefined"],
        default="identical",
    )
    paired = (w.groupby("model", as_index=False)
              .agg(n_valid_pairs=("hybrid_minus_6m", "count"),
                   mean_hybrid_minus_6m=("hybrid_minus_6m", "mean"),
                   better_months=("status", lambda s: int((s == "hybrid_better").sum())),
                   worse_months=("status", lambda s: int((s == "six_month_better").sum())),
                   identical_months=("status", lambda s: int((s == "identical").sum())),
                   undefined_months=("status", lambda s: int((s == "undefined").sum()))))

    # Model-vs-OLS on identical source scenario + month; positive means model improved.
    baseline = m.loc[m["model"].eq("OLS"), ["scenario", "target_month", "rank_ic"]].rename(
        columns={"rank_ic": "ols_rank_ic"})
    if len(baseline):
        comparisons = m.merge(baseline, on=["scenario", "target_month"], validate="many_to_one")
        comparisons["model_minus_ols"] = comparisons["rank_ic"] - comparisons["ols_rank_ic"]
        vs_ols = (comparisons.groupby(["model", "scenario"], as_index=False)
                  .agg(n_valid_pairs=("model_minus_ols", "count"),
                       mean_model_minus_ols=("model_minus_ols", "mean"),
                       better_months=("model_minus_ols", lambda s: int((s > 1e-12).sum())),
                       worse_months=("model_minus_ols", lambda s: int((s < -1e-12).sum()))))
    else:
        vs_ols = pd.DataFrame(columns=["model", "scenario", "n_valid_pairs", "mean_model_minus_ols",
                                       "better_months", "worse_months"])

    output.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "monthly_rank_ic.csv": m,
        "fold_rank_ic.csv": f,
        "yearly_rank_ic.csv": yearly,
        "model_summary.csv": summary,
        "paired_hybrid_vs_6m.csv": paired,
        "paired_monthly_ic.csv": w,
        "paired_models_vs_ols.csv": vs_ols,
    }.items():
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")

    fingerprints = {name: sha256(path) for name, path in paths.items()}
    info = {
        "evaluation": "TRAIN-only annual expanding window",
        "train_start": "2001-01",
        "validation_start": f"{evaluation_years[0]}-01",
        "validation_end": f"{evaluation_years[-1]}-12",
        "folds": len(evaluation_years),
        "validation_months_per_model_scenario": len(evaluation_years) * 12,
        "common_stock_month_rows": int(len(d)),
        "models": list(a.models),
        "scenarios": list(SCENARIOS),
        "features": list(FEATURE_COLS),
        "random_seed": SEED,
        "min_stocks": a.min_stocks,
        "input_sha256": fingerprints,
        "not_fully_pit_verified": True,
        "undefined_ic_policy": "Raw monthly rank_ic remains NaN. Separate summary mean_ic_undefined_as_zero treats undefined IC as zero, not as a measured IC.",
        "model_hyperparameters": {
            "OLS": "Reuse src.h2.models.estimators.model_instance('OLS')",
            "Ridge": "Reuse existing alpha=10.0 pipeline",
            "ElasticNet": "alpha=1e-5, l1_ratio=0.5, StandardScaler, seed=2026",
            "RandomForest": f"trees={a.rf_trees}, max_depth=8, min_samples_leaf=80, seed=2026",
            "LightGBM": "Reuse existing n_estimators=140, num_leaves=15 etc.",
            "XGBoost": "Reuse existing n_estimators=140, max_depth=5 etc.",
        },
        "notes": [
            "Models share the same stocks, months and four predictors within each scenario.",
            "All hyperparameters are pre-specified, not tuned on 2022-2025 Test returns.",
            "Rank IC is Spearman correlation across stocks in each target month.",
            "No 2022-2025 Test return labels are used.",
        ],
    }

    # Reproduce the previous 6m/Hybrid OLS experiment if its summary is present
    # and the inputs match exactly.
    previous = resolve(a.project.resolve(), a.previous_ols_summary)
    if "OLS" in a.models and evaluation_years[0] == 2006 and evaluation_years[-1] == 2021 and previous.is_file():
        old = json.loads(previous.read_text(encoding="utf-8-sig"))
        old_paths = old.get("input_sha256", {})
        identical_inputs = (
            old_paths.get("panel") == fingerprints["panel"]
            and old_paths.get("six_month") == fingerprints["6m"]
            and old_paths.get("hybrid") == fingerprints["hybrid"]
        )
        reproduction = {"previous_file": str(previous), "input_hashes_match": identical_inputs}
        if identical_inputs:
            errors = {}
            for scenario, old_key in [("6m", "mean_ic_6m"), ("hybrid", "mean_ic_hybrid")]:
                actual = float(summary.loc[
                    summary["model"].eq("OLS") & summary["scenario"].eq(scenario),
                    "mean_valid_rank_ic"].iloc[0])
                errors[scenario] = float(actual - old[old_key])
            reproduction["mean_ic_errors"] = errors
            reproduction["reproduced_within_1e-8"] = all(abs(x) < 1e-8 for x in errors.values())
            if not reproduction["reproduced_within_1e-8"]:
                print("WARNING: Previous OLS ICs were NOT reproduced. Check preprocessing.", flush=True)
        else:
            reproduction["note"] = "Inputs changed; previous OLS IC not directly comparable"
        info["previous_ols_reproduction"] = reproduction

    (output / "summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("\n[Model summary: 6m/Hybrid, IC zero-filled only if undefined]")
    cols = ["model", "scenario", "mean_ic_undefined_as_zero", "mean_valid_rank_ic",
            "n_valid_months", "n_undefined_months", "positive_share_all_months"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n[Hybrid - 6m]")
    print(paired.to_string(index=False, float_format=lambda v: f"{v:+.6f}"))
    print("\nResults saved to:", output)
    return summary


def main():
    a = parse_args()
    if not 2006 <= a.start_year <= a.end_year <= 2021:
        raise ValueError("검증 연도는 2006~2021 범위여야 합니다")
    if len(set(a.models)) != len(a.models):
        raise ValueError("--models에 중복된 모델이 있습니다")
    if a.min_stocks < 3 or a.rf_trees < 1:
        raise ValueError("min-stocks >=3, rf-trees >=1이어야 합니다")
    root = a.project.resolve()
    paths = {
        "panel": resolve(root, a.panel),
        "6m": resolve(root, a.lag_6m),
        "hybrid": resolve(root, a.hybrid),
    }
    missing = [str(x) for x in paths.values() if not x.is_file()]
    if missing:
        raise FileNotFoundError("입력 CSV 파일 없음:\n" + "\n".join(missing))

    d = load_common_panel(a, paths)
    years = list(range(a.start_year, a.end_year + 1))
    print(f"Training data common observations: {len(d):,}")
    print(f"Validation years: {years[0]}-{years[-1]}, {len(years)} annual folds")
    print("Algorithms:", ", ".join(a.models))
    print("Features:", ", ".join(FEATURE_COLS))
    print("Comparison: 6m vs hybrid, same stock-month sample")

    monthly, folds = [], []
    for fold, year in enumerate(years, 1):
        start, end = f"{year}-01", f"{year}-12"
        expected_months = [f"{year}-{mo:02d}" for mo in range(1, 13)]
        train_mask = d["target_month"].lt(start)
        val_mask = d["target_month"].between(start, end)
        if not train_mask.any() or not val_mask.any():
            raise ValueError(f"{year}: 학습 또는 검증 자료가 없습니다")
        if len(d.loc[train_mask]) < 100:
            raise ValueError(f"{year}: 최소 학습 표본 100행 미만")

        for scenario in SCENARIOS:
            # One fold/scenario X matrix shared by all algorithms.
            cols = pd.DataFrame({
                "policy": d["policy"],
                "debt": d[f"z_{scenario}"],
                "policy_x_debt": d["policy"] * d[f"z_{scenario}"],
                "log_size": d["log_size"],
            }, index=d.index)
            if tuple(FEATURE_COLS) != FEATURE_NAMES:
                raise ValueError("기존 FEATURES 스키마가 변경되었습니다. 코드를 확인하세요.")
            x_tr, x_va = cols.loc[train_mask, list(FEATURE_COLS)], cols.loc[val_mask, list(FEATURE_COLS)]
            y_tr = d.loc[train_mask, "return_t1"]
            val_info = d.loc[val_mask, ["target_month", "stock_id", "return_t1"]]
            if not np.isfinite(x_tr.to_numpy(dtype=float)).all() or not np.isfinite(x_va.to_numpy(dtype=float)).all():
                raise ValueError(f"{year} {scenario}: 모델 입력의 NaN/Inf 존재")
            if not np.isfinite(y_tr.to_numpy(dtype=float)).all():
                raise ValueError(f"{year} {scenario}: 학습 목표수익률의 NaN/Inf 존재")

            for name in a.models:
                reg = make_estimator(name, a)
                reg.fit(x_tr, y_tr)
                preds = reg.predict(x_va)
                scores = monthly_rank_ic(val_info, preds, expected_months, a.min_stocks)
                for x in scores:
                    monthly.append({"fold": fold, "validation_year": year,
                                    "train_start": "2001-01", "train_end": f"{year-1}-12",
                                    "scenario": scenario, "model": name, **x})
                scores_nonmissing = [x["rank_ic"] for x in scores if np.isfinite(x["rank_ic"])]
                ic_mean = float(np.mean(scores_nonmissing)) if scores_nonmissing else float("nan")
                folds.append({
                    "fold": fold, "train_start": "2001-01", "train_end": f"{year-1}-12",
                    "validation_start": start, "validation_end": end,
                    "scenario": scenario, "model": name, "n_train": len(x_tr),
                    "n_val": len(x_va), "n_months": 12,
                    "n_valid_ic_months": len(scores_nonmissing),
                    "mean_valid_ic": ic_mean,
                })
                print(f"Fold {fold:02d}/{len(years):02d} | train 2001-{year-1} | val {year} | "
                      f"{scenario:6s} | {name:13s} | IC {format_ic(ic_mean)} | "
                      f"{len(scores_nonmissing)}/12 months", flush=True)

    out = resolve(root, a.out)
    save_results(a, out, d, monthly, folds, paths, years)


if __name__ == "__main__":
    main()
