# -*- coding: utf-8 -*-
"""H2 annual expanding-window interaction ablation, six models x two debt sources.

Run from finance_v2 project ROOT:
    python -m src.h2.evaluation.evaluate_h2_interaction_yearly_models

Add this file under src/h2/evaluation/. Existing source scripts and prior
results are not changed. All target months are restricted to 2001--2021.
Annual folds: 2001--2005 -> 2006, ..., 2001--2020 -> 2021.

Comparison, on exactly the same stock-months:
    no_interaction : policy, debt, log_size
    full_interaction: policy, debt, policy_x_debt, log_size

Each scenario (6m / hybrid) uses the same raw inputs and fixed estimator
settings as evaluate_rankic_yearly_models.py. The monthly IC is cross-sectional
Spearman correlation for the target month. No causal effect is implied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.h2.data_io import ROOT, PROCESSED, sha256
from src.h2.evaluation.evaluate_rankic import resolve
from src.h2.evaluation.evaluate_rankic_yearly_models import (
    ALL_MODELS,
    SCENARIOS,
    SEED,
    load_common_panel,
    make_estimator,
    monthly_rank_ic,
)

VARIANTS = {
    "no_interaction": ("policy", "debt", "log_size"),
    "full_interaction": ("policy", "debt", "policy_x_debt", "log_size"),
}
EPS = 1e-12


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=ROOT)
    p.add_argument("--panel", type=Path, default=PROCESSED / "dataset.csv")
    p.add_argument("--lag-6m", type=Path,
                   default=PROCESSED / "debt" /
                   "h2_debt_lag_6m_fiscal_screened_2001_2021.csv")
    p.add_argument("--hybrid", type=Path,
                   default=PROCESSED / "debt_hybrid" /
                   "h2_debt_hybrid_candidate_2001_2021.csv")
    p.add_argument("--out", type=Path,
                   default=Path("results/h2_interaction_ablation_yearly_2006_2021"))
    p.add_argument("--models", nargs="+", choices=ALL_MODELS,
                   default=list(ALL_MODELS))
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                   default=list(SCENARIOS))
    p.add_argument("--start-year", type=int, default=2006)
    p.add_argument("--end-year", type=int, default=2021)
    p.add_argument("--rf-trees", type=int, default=120)
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--min-stocks", type=int, default=20)
    p.add_argument("--reference-monthly", type=Path,
                   default=Path("results/rankic_yearly_all_models_2006_2021/monthly_rank_ic.csv"))
    p.add_argument("--no-plots", action="store_true",
                   help="Skip charts; CSV/JSON always saved")
    return p.parse_args()


def prepare_inputs(d: pd.DataFrame, scenario: str):
    debt = d[f"z_{scenario}"]
    return pd.DataFrame({
        "policy": d["policy"],
        "debt": debt,
        "policy_x_debt": d["policy"] * debt,
        "log_size": d["log_size"],
    }, index=d.index)


def paired_results(monthly: pd.DataFrame):
    # One row per model/scenario/month, with ICs for both variants.
    w = (monthly.pivot(index=["model", "scenario", "fold", "validation_year", "target_month"],
                       columns="variant", values="rank_ic")
         .reset_index())
    w.columns.name = None
    w["delta_ic"] = w["full_interaction"] - w["no_interaction"]
    w["status"] = np.select([
        w["delta_ic"].isna(),
        w["delta_ic"] > EPS,
        w["delta_ic"] < -EPS,
    ], ["undefined", "improved", "worsened"], default="unchanged")
    return w


def aggregate(paired: pd.DataFrame, keys: list[str]):
    grouped = paired.groupby(keys, as_index=False)
    out = grouped.agg(
        mean_ic_without=("no_interaction", "mean"),
        mean_ic_with=("full_interaction", "mean"),
        mean_delta_ic=("delta_ic", "mean"),
        median_delta_ic=("delta_ic", "median"),
        std_delta_ic=("delta_ic", "std"),
        n_months=("target_month", "size"),
        n_valid_pairs=("delta_ic", "count"),
        improved_months=("status", lambda s: int(s.eq("improved").sum())),
        worsened_months=("status", lambda s: int(s.eq("worsened").sum())),
        unchanged_months=("status", lambda s: int(s.eq("unchanged").sum())),
        undefined_months=("status", lambda s: int(s.eq("undefined").sum())),
    )
    # A separate, explicitly labeled conservative summary for undefined ICs.
    out["mean_delta_undefined_as_zero"] = (
        paired.assign(delta_fill=paired["delta_ic"].fillna(0.0))
        .groupby(keys, sort=False)["delta_fill"]
        .mean().reindex(pd.MultiIndex.from_frame(out[keys]) if len(keys) > 1
                      else out[keys[0]].values).to_numpy()
    )
    return out


def check_reference(root: Path, reference: Path, monthly: pd.DataFrame):
    """Cross-check that the *with-interaction* arm reproduces previous results."""
    ref_path = resolve(root, reference)
    if not ref_path.is_file():
        return {"reference_found": False, "reference_path": str(ref_path)}

    prev = pd.read_csv(ref_path, encoding="utf-8-sig", low_memory=False,
                       dtype={"target_month": "string"})
    expected = {"model", "scenario", "target_month", "rank_ic"}
    if not expected.issubset(prev.columns):
        return {"reference_found": True, "error": "Reference columns unavailable"}
    current = monthly.loc[monthly["variant"].eq("full_interaction"),
                          ["model", "scenario", "target_month", "rank_ic"]]
    prev = prev[prev["model"].isin(current["model"].unique()) &
                prev["scenario"].isin(current["scenario"].unique()) &
                prev["target_month"].isin(current["target_month"].unique())]
    join = current.merge(prev, on=["model", "scenario", "target_month"],
                         how="outer", validate="one_to_one", indicator=True,
                         suffixes=("_current", "_reference"))
    diff = (join["rank_ic_current"] - join["rank_ic_reference"]).abs()
    valid = join["_merge"].eq("both") & (
        (diff <= 1e-8) |
        (join["rank_ic_current"].isna() & join["rank_ic_reference"].isna()))
    result = {
        "reference_found": True,
        "reference_path": str(ref_path),
        "matched_rows": int(join["_merge"].eq("both").sum()),
        "unmatched_rows": int(join["_merge"].ne("both").sum()),
        "max_abs_ic_difference": float(diff.max()) if diff.notna().any() else None,
        "full_interaction_reproduced": bool(valid.all()),
    }
    return result


def make_figures(out: Path, paired: pd.DataFrame, yearly: pd.DataFrame):
    """Generate presentation-useful 16-fold charts using observed monthly pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotdir = out / "figures"
    plotdir.mkdir(parents=True, exist_ok=True)
    for scenario in sorted(paired["scenario"].unique()):
        s = paired[paired["scenario"].eq(scenario)]
        c = (s.groupby("model", as_index=False)
             .agg(improved=("status", lambda x: int(x.eq("improved").sum())),
                  worsened=("status", lambda x: int(x.eq("worsened").sum())),
                  unchanged=("status", lambda x: int(x.eq("unchanged").sum())),
                  undefined=("status", lambda x: int(x.eq("undefined").sum()))))
        c = c.sort_values("model")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(c))
        bottom = np.zeros(len(c), dtype=float)
        for name, col in (("Improved", "improved"), ("Worsened", "worsened"),
                          ("Unchanged", "unchanged"), ("Undefined", "undefined")):
            values = c[col].to_numpy()
            ax.bar(x, values, bottom=bottom, label=name)
            for k, value in enumerate(values):
                if value > 0:
                    ax.text(k, bottom[k] + value / 2, str(value),
                            ha="center", va="center", fontsize=9)
            bottom += values
        ax.set_xticks(x, c["model"], rotation=15)
        ax.set_ylabel("Months")
        ax.set_title(f"Interaction ablation: monthly Rank IC wins/losses ({scenario})")
        ax.set_ylim(0, max(200, float(bottom.max() * 1.10)))
        ax.legend(loc="upper right", ncol=2)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plotdir / f"monthly_win_loss_{scenario}.png", dpi=180)
        plt.close(fig)

        annual = yearly[yearly["scenario"].eq(scenario)]
        pivot = annual.pivot(index="validation_year", columns="model",
                             values="mean_delta_ic").sort_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        for model in pivot.columns:
            ax.plot(pivot.index.astype(str), pivot[model], marker=".", label=model)
        ax.axhline(0, linewidth=1, color="gray")
        ax.set_title(f"Yearly mean IC effect of Policy x Debt ({scenario})")
        ax.set_xlabel("Validation year (annual expanding window)")
        ax.set_ylabel("Delta Rank IC: With - Without")
        ax.legend(ncol=3)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plotdir / f"yearly_delta_ic_{scenario}.png", dpi=180)
        plt.close(fig)

        # Individual month-by-month effects; these are NOT the 3-fold results.
        for model in sorted(s["model"].unique()):
            q = (s[s["model"].eq(model)]
                 .sort_values("target_month")
                 .reset_index(drop=True))
            fig, ax = plt.subplots(figsize=(14, 4.5))
            ax.plot(np.arange(len(q)), q["delta_ic"], linewidth=1.2)
            ax.axhline(0, color="gray", linewidth=1)
            idx = list(range(0, len(q), 12))
            ax.set_xticks(idx, q.loc[idx, "validation_year"].astype(str))
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Validation year (2006--2021 annual expanding window)")
            ax.set_ylabel("Monthly delta IC (with - without)")
            ax.set_title(f"Monthly Policy x Debt ablation: {model} / {scenario}")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(plotdir / f"monthly_delta_ic_{scenario}_{model}.png", dpi=180)
            plt.close(fig)


def main():
    a = arguments()
    if not 2006 <= a.start_year <= a.end_year <= 2021:
        raise ValueError("Years must lie within 2006--2021")
    if a.min_stocks < 3 or a.rf_trees < 1 or a.n_jobs < 1:
        raise ValueError("Check --min-stocks, --rf-trees, --n-jobs")
    if len(set(a.models)) != len(a.models) or len(set(a.scenarios)) != len(a.scenarios):
        raise ValueError("Duplicate models or scenarios")

    root = a.project.resolve()
    paths = {
        "panel": resolve(root, a.panel),
        "6m": resolve(root, a.lag_6m),
        "hybrid": resolve(root, a.hybrid),
    }
    for label, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"{label} input missing: {p}")

    d = load_common_panel(a, paths)
    years = list(range(a.start_year, a.end_year + 1))
    print(f"Same-stock-month common sample: {len(d):,} rows")
    print(f"Annual Expanding Window: {years[0]}--{years[-1]} ({len(years)} folds)")
    print("Models:", ", ".join(a.models), "; scenarios:", ", ".join(a.scenarios))

    monthly = []
    for fold, year in enumerate(years, 1):
        start = f"{year}-01"
        train_mask = d["target_month"].lt(start)
        valid_mask = d["target_month"].between(start, f"{year}-12")
        if not train_mask.any() or not valid_mask.any():
            raise ValueError(f"{year}: empty train/validation fold")
        if train_mask.sum() < 100:
            raise ValueError(f"{year}: fewer than 100 training rows")
        target_months = [f"{year}-{m:02d}" for m in range(1, 13)]
        val_info = d.loc[valid_mask, ["target_month", "stock_id", "return_t1"]]
        y_tr = d.loc[train_mask, "return_t1"]
        if not np.isfinite(y_tr.to_numpy(dtype=float)).all():
            raise ValueError(f"{year}: nonfinite training target")

        for scenario in a.scenarios:
            X = prepare_inputs(d, scenario)
            for variant, cols in VARIANTS.items():
                x_tr = X.loc[train_mask, list(cols)]
                x_va = X.loc[valid_mask, list(cols)]
                if not (np.isfinite(x_tr.to_numpy(dtype=float)).all() and
                        np.isfinite(x_va.to_numpy(dtype=float)).all()):
                    raise ValueError(f"{year} {scenario}: nonfinite predictors")
                for name in a.models:
                    model = make_estimator(name, a)
                    model.fit(x_tr, y_tr)
                    pred = model.predict(x_va)
                    scores = monthly_rank_ic(val_info, pred, target_months, a.min_stocks)
                    for r in scores:
                        monthly.append({
                            "fold": fold,
                            "validation_year": year,
                            "train_start": "2001-01",
                            "train_end": f"{year-1}-12",
                            "scenario": scenario,
                            "model": name,
                            "variant": variant,
                            **r,
                        })
                    nvalid = sum(np.isfinite(r["rank_ic"]) for r in scores)
                    print(f"Fold {fold:02d}/{len(years):02d} | {year} | {scenario:6s} | "
                          f"{name:13s} | {variant:16s} | {nvalid}/12 months", flush=True)

    m = pd.DataFrame(monthly)
    expected = len(years) * 12 * len(a.scenarios) * len(a.models) * len(VARIANTS)
    if len(m) != expected:
        raise AssertionError(f"Monthly record count mismatch: {len(m)} != {expected}")
    if m.duplicated(["model", "scenario", "variant", "target_month"]).any():
        raise AssertionError("Duplicate IC row")

    pair = paired_results(m)
    expected_pairs = len(years) * 12 * len(a.scenarios) * len(a.models)
    if len(pair) != expected_pairs:
        raise AssertionError("Missing comparison rows")
    summary = aggregate(pair, ["model", "scenario"])
    yearly = aggregate(pair, ["model", "scenario", "validation_year"])
    summary = summary.sort_values("mean_delta_ic", ascending=False).reset_index(drop=True)

    out = resolve(root, a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {
        "ablation_monthly_rank_ic.csv": m,
        "paired_monthly_interaction.csv": pair,
        "ablation_model_summary.csv": summary,
        "ablation_yearly_summary.csv": yearly,
    }.items():
        df.to_csv(out / name, index=False, encoding="utf-8-sig")

    reference = check_reference(root, a.reference_monthly, m)
    config = {
        "evaluation": "Annual Expanding Window, Train only",
        "training_start": "2001-01",
        "validation_start": f"{a.start_year}-01",
        "validation_end": f"{a.end_year}-12",
        "folds": len(years),
        "validation_months_per_model_scenario": 12 * len(years),
        "common_stock_month_rows": len(d),
        "models": list(a.models),
        "scenarios": list(a.scenarios),
        "variants": {k: list(v) for k, v in VARIANTS.items()},
        "seed": SEED,
        "rf_trees": a.rf_trees,
        "min_stocks": a.min_stocks,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "previous_full_interaction_reproduction": reference,
        "not_fully_pit_verified": True,
        "notes": [
            "Same model/scenario/month/stock sample for ablation arms.",
            "Delta IC is predictive comparison, not a causal interaction effect.",
            "Rank IC undefined on constant predictions; missing values are not silently 0.",
            "2022--2025 Test target returns are not read.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    if not a.no_plots:
        make_figures(out, pair, yearly)

    print("\n[Interaction ablation summary: with - without]")
    print(summary[["model", "scenario", "mean_ic_without", "mean_ic_with",
                   "mean_delta_ic", "improved_months", "worsened_months",
                   "unchanged_months", "undefined_months"]].to_string(
                       index=False, float_format=lambda v: f"{v:+.6f}"))
    print("\nReference reproduction:", reference)
    print("Saved:", out)


if __name__ == "__main__":
    main()
