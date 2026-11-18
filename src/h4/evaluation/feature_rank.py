"""H4 feature-only Rank IC, following H3's raw/winsor paired analysis."""

from pathlib import Path
from datetime import datetime
import hashlib, json, zipfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ..features.epu import (
    load_epu_index,
    align_epu_to_base_month,
    calculate_h4_epu_features,
)
from ..evaluation.rank_snapshot import (
    rank_ic_for_month,
    join_training_targets,
    newey_west_mean_test,
)


def prepare_data(root):
    """Build paired raw/winsor scores before joining next-month returns."""
    mom = pd.read_parquet(root / "data/processed/mom.parquet")
    panel = pd.read_parquet(root / "data/interim/panel.parquet")
    mom["base_month"] = pd.PeriodIndex(mom.base_month, freq="M").astype(str)
    panel["month"] = pd.PeriodIndex(panel.month, freq="M").astype(str)
    mom = mom[mom.base_month.between("2000-12", "2021-11")].copy()
    keys = mom.loc[
        mom.eligible.fillna(False) & mom.mom_available.fillna(False),
        ["stock_id", "base_month"],
    ]
    epu = align_epu_to_base_month(
        load_epu_index(root / "data/raw/Korea_Overall_EPU.xlsx")
    )
    f = calculate_h4_epu_features(mom, keys, epu)
    f["score"] = f.h4_interaction.where(f.h4_available)
    if np.isinf(f.score).any():
        raise ValueError("infinite feature")
    # Bounds estimated on all available eligible features BEFORE y is attached.
    g = f.groupby("base_month").score
    f["q01"] = g.transform(lambda s: s.quantile(0.01, interpolation="linear"))
    f["q99"] = g.transform(lambda s: s.quantile(0.99, interpolation="linear"))
    f["winsor_score"] = f.score.clip(lower=f.q01, upper=f.q99)
    returns = panel[panel.month.between("2000-01", "2021-12")][
        ["stock_id", "month", "return"]
    ].copy()
    returns.attrs["units"] = {"return": "decimal"}
    d = join_training_targets(f, returns)
    return d


def evaluate_months(data):
    """Compute both feature ICs on the same stock-month pairs."""
    rows = []
    coverage = []
    error = []
    for month, part in data.groupby("base_month", sort=True):
        paired = np.isfinite(part.score) & np.isfinite(part.winsor_score)
        ns = []
        for label, col in [("h4_raw", "score"), ("h4_winsor", "winsor_score")]:
            x = part[["stock_id", "target_month", "y"]].copy()
            x["score"] = part[col].where(paired)
            x = x.assign(model="feature_score", candidate=label, fold="development")
            row = rank_ic_for_month(x, score_column="score").iloc[0].to_dict()
            rows.append(row)
            ns.append(row["n"])
            valid = x[["score", "y"]].dropna()
            if not row["reason"]:
                error.append(
                    abs(float(spearmanr(valid.score, valid.y).statistic) - row["ic"])
                )
        assert ns[0] == ns[1]
        coverage.append(
            dict(
                base_month=month,
                target_month=part.target_month.iloc[0],
                eligible_mom_rows=len(part),
                feature_available=int(paired.sum()),
                valid_pairs=ns[0],
                missing_target=int(part.y.isna().sum()),
                clipped=int((paired & part.score.ne(part.winsor_score)).sum()),
                q01=part.q01.iloc[0],
                q99=part.q99.iloc[0],
            )
        )
    return rows, coverage, error


def summarize_results(rows, coverage, error):
    """Complete the calendar and summarize only shared valid months."""
    monthly = pd.DataFrame(rows)
    expected = pd.period_range("2001-01", "2021-12", freq="M").astype(str).tolist()
    missing = set(expected) - set(monthly.target_month)
    for month in sorted(missing):
        for candidate in ["h4_raw", "h4_winsor"]:
            rows.append(
                dict(
                    model="feature_score",
                    candidate=candidate,
                    fold="development",
                    target_month=month,
                    n=0,
                    ic=np.nan,
                    reason="no_eligible_available_momentum_rows",
                )
            )
    monthly = (
        pd.DataFrame(rows)
        .sort_values(["target_month", "candidate"])
        .reset_index(drop=True)
    )
    wide = monthly.pivot(
        index="target_month", columns="candidate", values="ic"
    ).sort_index()
    assert wide.index.tolist() == expected and len(wide) == 252
    common = wide.dropna()
    assert (
        common.index.tolist()
        == pd.period_range(common.index.min(), common.index.max(), freq="M")
        .astype(str)
        .tolist()
    )
    assert max(error) < 1e-12
    stats = []
    for c in wide.columns:
        h = newey_west_mean_test(common[c].to_numpy())
        stats.append(
            dict(
                candidate=c,
                mean_rank_ic=float(common[c].mean()),
                months=len(common),
                positive_month_fraction=float(common[c].gt(0).mean()),
                **h
            )
        )
    delta = common.h4_winsor - common.h4_raw
    summary = pd.DataFrame(stats)
    yearly = (
        monthly.assign(year=monthly.target_month.str[:4])
        .groupby(["candidate", "year"])
        .agg(
            mean_rank_ic=("ic", "mean"),
            valid_months=("ic", "count"),
            stock_months=("n", "sum"),
        )
        .reset_index()
    )
    tables = {
        "monthly_ic.csv": monthly,
        "summary.csv": summary,
        "yearly_ic.csv": yearly,
        "coverage.csv": pd.DataFrame(coverage),
        "paired_ic.csv": wide.reset_index(),
    }
    checks = dict(
        calendar_months=252,
        valid_common_months=len(common),
        unavailable_months=sorted(set(expected) - set(common.index)),
        paired_samples_equal=True,
        independent_scipy_max_error=max(error),
    )
    return tables, checks, newey_west_mean_test(delta.to_numpy())


def build_run_metadata(root, checks, delta_hac):
    """Record definitions and source hashes for the exported analysis."""
    files = [
        "data/processed/mom.parquet",
        "data/interim/panel.parquet",
        "data/raw/Korea_Overall_EPU.xlsx",
        "src/h4/evaluation/feature_rank.py",
        "src/h4/evaluation/rank_snapshot.py",
        "src/h4/features/epu.py",
    ]
    run = dict(
        author="h4",
        status="FEATURE_ANALYSIS_NOT_OFFICIAL_MODEL_TEST",
        period="2001-01..2021-12",
        method="H3-style paired direct feature Spearman; no model fitting or expanding window",
        feature="raw EPU(t-1)*team-provided raw MOM(t)",
        winsor=(
            "monthly 1%,99% linear interpolation; feature-only; bounds before "
            "y join; no target clipping"
        ),
        universe=(
            "base eligible & mom_available & finite feature/target; same rows "
            "both variants; no extra price filter"
        ),
        epu_definition="Cho-Kim (2023) Korean news-based EPU; user-supplied H4 specification",
        epu_source="https://policyuncertainty.com/korea_chokim.html",
        epu_availability=(
            "specified one-month lag; historical release dates and vintages "
            "unverified"
        ),
        common_functions=(
            "Exact upstream function snapshots under src/h4; existing common "
            "modules unchanged"
        ),
        upstream_rank_blob="66669c81302b06f3392ab1606c93bd9afee5d0a5",
        checks=checks,
        delta_hac=delta_hac,
        holm="not applied: exploratory comparison; not full project hypothesis family",
        test_evaluated=False,
        hashes={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files},
    )
    return run


def save_results(root, tables, metadata):
    """Write aggregate tables only; never package source data or predictions."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output = root / "outputs/model_results/h4" / (stamp + "_h4_feature_rank_raw-winsor")
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "result.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "run.json",
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        )
        for name, table in tables.items():
            bundle.writestr(name, table.to_csv(index=False))
        bundle.writestr(
            "README.txt",
            "H3-style direct H4 feature analysis. Raw and winsorized feature on identical stock-month pairs. "
            "Not official model_test PASSED output or OLS forecast results. Monthly actual returns unchanged. "
            "No extra penny-stock exclusion. No source data or per-stock predictions.\n",
        )
    tables["summary.csv"].to_csv(output / "summary.csv", index=False)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("invalid_result_archive")
    return archive


def main(root=Path.cwd()):
    root = Path(root).resolve()
    data = prepare_data(root)
    rows, coverage, errors = evaluate_months(data)
    tables, checks, delta_hac = summarize_results(rows, coverage, errors)
    archive = save_results(root, tables, build_run_metadata(root, checks, delta_hac))
    print(archive)
    columns = [
        "candidate",
        "mean_rank_ic",
        "months",
        "positive_month_fraction",
        "p_two_sided",
    ]
    print(tables["summary.csv"][columns].to_string(index=False))
    monthly, coverage = tables["monthly_ic.csv"], tables["coverage.csv"]
    print(
        "pairs",
        int(monthly.loc[monthly.candidate.eq("h4_raw"), "n"].sum()),
        "clipped",
        int(coverage.clipped.sum()),
        "checks passed",
        checks["independent_scipy_max_error"],
    )


if __name__ == "__main__":
    main()
