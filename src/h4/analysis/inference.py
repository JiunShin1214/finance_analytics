"""H4 EPU proxy inference; no common modules are modified."""

from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import zipfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from ..features.epu import (
    load_epu_index,
    align_epu_to_base_month,
    calculate_h4_epu_features,
)


def holm(pvalues):
    """Supply the entire prespecified hypothesis family, not selected p-values."""
    names = list(pvalues)
    values = np.array([pvalues[n] for n in names], dtype=float)
    if (
        not len(values)
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("p-values must be finite and in [0,1]")
    return dict(zip(names, multipletests(values, method="holm")[1].tolist()))


def prepare(root, price_floor=1000):
    mom = pd.read_parquet(root / "data/processed/mom.parquet")
    panel = pd.read_parquet(root / "data/interim/panel.parquet")
    panel["month"] = pd.PeriodIndex(panel["month"], freq="M").astype(str)
    mom["base_month"] = pd.PeriodIndex(mom["base_month"], freq="M").astype(str)
    if panel.duplicated(["stock_id", "month"]).any():
        raise ValueError("duplicate panel keys")
    # Restrict targets before joining; do not evaluate sealed Test data.
    mom = mom[mom.base_month.between("2000-12", "2021-11")].copy()
    keys = mom.loc[
        mom.eligible.fillna(False) & mom.mom_available.fillna(False),
        ["stock_id", "base_month"],
    ]
    epu = align_epu_to_base_month(
        load_epu_index(root / "data/raw/Korea_Overall_EPU.xlsx")
    )
    data = calculate_h4_epu_features(mom, keys, epu)
    base = panel.rename(columns={"month": "base_month"})
    data = data.merge(
        base[["stock_id", "base_month", "market_cap", "price"]],
        on=["stock_id", "base_month"],
        validate="one_to_one",
    )
    data["target_month"] = (pd.PeriodIndex(data.base_month, freq="M") + 1).astype(str)
    target = panel[panel.month.between("2001-01", "2021-12")].rename(
        columns={"month": "target_month", "return": "y"}
    )
    data = data.merge(
        target[["stock_id", "target_month", "y"]],
        on=["stock_id", "target_month"],
        how="left",
        validate="one_to_one",
    )
    counts = {
        "eligible_momentum_rows": len(data),
        "missing_y": int(data.y.isna().sum()),
        "missing_epu": int(data.epu_raw.isna().sum()),
    }
    data["log_size"] = np.log(data.market_cap.where(data.market_cap > 0))
    data = (
        data[data.price.ge(price_floor)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["mom", "epu_raw", "y", "log_size"])
        .copy()
    )
    months = data[["base_month", "epu_raw"]].drop_duplicates()
    mean, std = months.epu_raw.mean(), months.epu_raw.std(ddof=1)
    if not np.isfinite(std) or std <= 0:
        raise ValueError("constant EPU")
    data["epu_z"] = (data.epu_raw - mean) / std
    counts.update(
        final_rows=len(data),
        months=data.base_month.nunique(),
        epu_mean=mean,
        epu_std=std,
    )
    return data, counts


def inference(data, winsor=True):
    data = data.copy()
    if winsor:
        for col in ["mom", "y"]:
            g = data.groupby("base_month")[col]
            data[col] = data[col].clip(
                g.transform(lambda x: x.quantile(0.01)),
                g.transform(lambda x: x.quantile(0.99)),
            )
    data["interaction"] = data.mom * data.epu_z
    cols = ["y", "mom", "interaction", "log_size"]
    dm = data[cols] - data.groupby("base_month")[cols].transform("mean")
    if np.linalg.matrix_rank(dm[cols[1:]]) < 3:
        raise ValueError("rank deficient panel design")
    model = sm.OLS(dm.y, dm[cols[1:]]).fit(
        cov_type="cluster", cov_kwds={"groups": data.base_month}
    )
    rows = []
    for month, g in data.groupby("base_month", sort=True):
        if len(g) < 30 or g.mom.nunique() < 2:
            raise ValueError(
                f"insufficient monthly sample: {month}; HAC requires consecutive months"
            )
        fit = sm.OLS(g.y, sm.add_constant(g[["mom"]], has_constant="add")).fit()
        rows.append(
            dict(
                base_month=month,
                mom_slope=fit.params["mom"],
                epu_z=g.epu_z.iloc[0],
                n_stocks=len(g),
            )
        )
    monthly = pd.DataFrame(rows)
    expected = pd.period_range(
        monthly.base_month.min(), monthly.base_month.max(), freq="M"
    ).astype(str)
    if monthly.base_month.tolist() != expected.tolist():
        raise ValueError("calendar gaps invalidate positional HAC lags")
    lags = int(4 * (len(monthly) / 100) ** (2 / 9))
    second = sm.OLS(
        monthly.mom_slope, sm.add_constant(monthly[["epu_z"]], has_constant="add")
    ).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    results = []
    for method, fit, term in [
        ("panel_month_FE_cluster_month", model, "interaction"),
        ("two_stage_HAC", second, "epu_z"),
    ]:
        results.append(
            dict(
                method=method,
                coefficient=float(fit.params[term]),
                standard_error=float(fit.bse[term]),
                statistic=float(fit.tvalues[term]),
                p_two_sided=float(fit.pvalues[term]),
                p_negative_one_sided=float(norm.cdf(fit.tvalues[term])),
                n=int(fit.nobs),
                hac_lags=lags if fit is second else None,
            )
        )
    return pd.DataFrame(results), monthly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--holm-family",
        type=Path,
        help="JSON of ALL prespecified hypothesis p-values, including H4",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = root / "outputs/model_results/h4" / (stamp + "_h4_epu_ols-inference")
    out.mkdir(parents=True, exist_ok=False)
    data, counts = prepare(root)
    tables = {}
    for label, winsor in [("winsor_1_99", True), ("raw_sensitivity", False)]:
        tables[label + "_tests.csv"], tables[label + "_monthly_slopes.csv"] = inference(
            data, winsor
        )
    hashes = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest()
        for p in [
            "data/processed/mom.parquet",
            "data/interim/panel.parquet",
            "data/raw/Korea_Overall_EPU.xlsx",
            "src/h4/features/epu.py",
            "src/h4/analysis/inference.py",
        ]
    }
    run = dict(
        author="h4",
        experiment="h4_epu",
        status="INFERENCE_ONLY_NOT_TEAM_MODEL_COMPARABLE",
        created_local=stamp,
        timezone="Asia/Seoul",
        counts=counts,
        hashes=hashes,
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        versions={
            p: importlib.metadata.version(p)
            for p in ["pandas", "numpy", "scipy", "statsmodels", "pyarrow"]
        },
        conditions=dict(
            target_period=["2001-01", "2021-12"],
            price_floor=1000,
            epu_lag_months=1,
            epu_definition="Cho-Kim (2023) Korean news-based EPU; user-supplied H4 specification",
            epu_source="https://policyuncertainty.com/korea_chokim.html",
            epu_availability=(
                "specified one-month lag; historical release dates and vintages "
                "unverified"
            ),
            epu_standardization="equal-weight observed months, ddof=1; descriptive inference only",
            panel=(
                "month fixed effects; EPU main effect absorbed; month-clustered "
                "normal asymptotic inference"
            ),
            two_stage=(
                "monthly y~const+mom; then slope~const+epu_z, HAC normal "
                "asymptotic one-sided gamma<0"
            ),
            primary="winsorized monthly mom and y at 1%,99%; not a forecast evaluation",
            sensitivity="unwinsorized",
            holm="pending full prespecified hypothesis family",
            rank_ic="not run: local common model_test and required common features absent",
            test_access=False,
        ),
    )
    if args.holm_family:
        family = json.loads(args.holm_family.read_text(encoding="utf-8"))
        expected = float(tables["winsor_1_99_tests.csv"].iloc[1].p_negative_one_sided)
        if "H4" not in family or not np.isclose(
            family["H4"], expected, atol=1e-12, rtol=0
        ):
            raise ValueError("family H4 must match primary two-stage p-value")
        run["holm_adjusted"] = holm(family)
    with zipfile.ZipFile(out / "result.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "run.json", json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False)
        )
        for name, table in tables.items():
            z.writestr(name, table.to_csv(index=False))
        z.writestr(
            "README.txt",
            (
                "Exploratory H4 EPU proxy inference, not official model_test "
                "output. No predictions or source data. Historical EPU release "
                "dates unverified; one-month lag is an assumption. Full family "
                "Holm pending. Common Rank IC comparison pending current team "
                "modules/data.\n"
            ),
        )
    print(out / "result.zip")
    print(tables["winsor_1_99_tests.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
