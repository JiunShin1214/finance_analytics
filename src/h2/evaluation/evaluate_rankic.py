
# -*- coding: utf-8 -*-
"""H2 TRAIN-only Rank IC: legacy, 6m, 9m, hybrid. NOT fully PIT verified."""

from __future__ import annotations

from src.h2.data_io import ROOT, PROCESSED
from src.h2.features.transforms import FEATURES, monthly_standardize
from src.h2.models.estimators import model_instance

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr


FOLDS = [
    ("2013-01", "2015-12"),
    ("2016-01", "2018-12"),
    ("2019-01", "2021-12"),
]

ALGORITHMS = [
    "OLS",
    "Ridge",
    "LightGBM",
    "XGBoost",
]


LAGS = {
    "legacy": None,
    "6m": "h2_debt_lag_6m_fiscal_screened_2001_2021.csv",
    "9m": "h2_debt_lag_9m_fiscal_screened_2001_2021.csv",
    "hybrid": None,
}


def args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--project",
        type=Path,
        default=ROOT,
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/rankic_hybrid_comparison"),
    )

    p.add_argument(
        "--panel",
        type=Path,
        default=PROCESSED / "dataset.csv",
    )

    p.add_argument(
        "--lag-dir",
        type=Path,
        default=PROCESSED / "debt",
    )

    p.add_argument(
        "--hybrid",
        type=Path,
        default=(
            PROCESSED
            / "debt_hybrid"
            / "h2_debt_hybrid_candidate_2001_2021.csv"
        ),
    )

    p.add_argument(
        "--min-stocks",
        type=int,
        default=20,
    )

    return p.parse_args()


def resolve(root, path):
    return path if path.is_absolute() else root / path


def norm_code(s):
    value = s.astype("string").str.strip()
    value = value.str.replace(
        r"^A(?=\d{6}$)", "", regex=True
    )
    value = value.str.replace(
        r"\.0$", "", regex=True
    )

    if not value.str.fullmatch(
        r"\d{1,6}", na=False
    ).all():
        raise ValueError(
            "잘못된 종목코드가 존재합니다."
        )

    return value.str.zfill(6)


def read_csv(path):
    return pd.read_csv(
        path,
        dtype={
            "stock_id": "string",
            "feature_month": "string",
            "target_month": "string",
        },
        low_memory=False,
    )


def hashfile(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for b in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(b)

    return h.hexdigest()






def calc_ic(df, pred, min_stocks):
    d = df[
        ["target_month", "return_t1", "stock_id"]
    ].copy()

    d["prediction"] = np.asarray(
        pred, dtype=float
    )

    rows = []

    for month, g in d.groupby(
        "target_month", sort=True
    ):
        g = g.dropna(
            subset=["return_t1", "prediction"]
        )

        if len(g) < min_stocks:
            continue

        if (
            g["prediction"].nunique() < 2
            or g["return_t1"].nunique() < 2
        ):
            continue

        ic = spearmanr(
            g["prediction"].to_numpy(),
            g["return_t1"].to_numpy(),
        ).statistic

        if np.isfinite(ic):
            rows.append(
                (month, len(g), float(ic))
            )

    return rows


def main():
    a = args()

    root = a.project.resolve()
    out = resolve(root, a.out)

    out.mkdir(
        parents=True, exist_ok=True
    )

    panel_path = resolve(root, a.panel)
    screened = resolve(root, a.lag_dir)

    paths = {
        "panel": panel_path,
        **{
            name: screened / file
            for name, file in LAGS.items()
            if file
        },
        "hybrid": resolve(root, a.hybrid),
    }

    missing = [
        str(path)
        for path in paths.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "필요한 파일이 없습니다:\n"
            + "\n".join(missing)
        )

    # -----------------------------------
    # 1. 기존 H2 패널
    # -----------------------------------
    p = read_csv(panel_path)

    columns = {
        "stock_id",
        "feature_month",
        "target_month",
        "return_t1",
        "market_cap",
        "debt_ratio_raw",
        "final_rule_v2_sensitivity",
    }

    missing_columns = columns - set(p.columns)

    if missing_columns:
        raise ValueError(
            f"H2 패널 컬럼 부족: {missing_columns}"
        )

    p["stock_id"] = norm_code(p["stock_id"])

    p["feature_month"] = (
        p["feature_month"].str.slice(0, 7)
    )

    p["target_month"] = (
        p["target_month"].str.slice(0, 7)
    )

    if (
        p[
            ["stock_id", "feature_month", "target_month"]
        ].isna().any().any()
        or p[
            ["stock_id", "feature_month"]
        ].duplicated().any()
    ):
        raise ValueError(
            "기존 H2 패널 종목×특성월 누락 또는 중복"
        )

    expected_target = (
        pd.PeriodIndex(
            p["feature_month"], freq="M"
        ) + 1
    ).astype(str)

    if not (
        p["target_month"].to_numpy()
        == expected_target.to_numpy()
    ).all():
        raise ValueError(
            "특성월과 다음 달 타깃의 시차가 맞지 않습니다."
        )

    # Train 타깃: 2021년 12월까지
    # 2022~2025 Test 정답은 사용하지 않음
    p = p.loc[
        p["target_month"].between(
            "2001-01", "2021-12"
        )
    ].copy()

    p["return_t1"] = pd.to_numeric(
        p["return_t1"], errors="coerce"
    )

    p["policy"] = pd.to_numeric(
        p["final_rule_v2_sensitivity"],
        errors="coerce",
    )

    p["market_cap"] = pd.to_numeric(
        p["market_cap"], errors="coerce"
    )

    p["log_size"] = np.log(
        p["market_cap"].where(
            p["market_cap"] > 0
        )
    )

    p["debt_legacy"] = pd.to_numeric(
        p["debt_ratio_raw"],
        errors="coerce",
    )

    # -----------------------------------
    # 2. DataGuide 6개월 / 9개월
    # -----------------------------------
    for label in ("6m", "9m"):
        lag = read_csv(paths[label])

        need = {
            "stock_id",
            "feature_month",
            "proxy_masked",
            "prediction_date",
            "proxy_available_date",
            "fiscal_date",
        }

        if need - set(lag.columns):
            raise ValueError(
                f"{label} 시차 자료 컬럼 부족: "
                f"{need - set(lag.columns)}"
            )

        lag["stock_id"] = norm_code(
            lag["stock_id"]
        )

        lag["feature_month"] = (
            lag["feature_month"].str.slice(0, 7)
        )

        keys = [
            "stock_id",
            "feature_month",
        ]

        if lag[keys].duplicated().any():
            raise ValueError(
                f"{label} 중복 종목월"
            )

        prediction = pd.to_datetime(
            lag["prediction_date"],
            errors="coerce",
        )

        available = pd.to_datetime(
            lag["proxy_available_date"],
            errors="coerce",
        )

        fiscal = pd.to_datetime(
            lag["fiscal_date"],
            errors="coerce",
        )

        expected = (
            pd.to_datetime(
                lag["feature_month"] + "-01",
                errors="coerce",
            )
            + pd.offsets.MonthEnd(0)
        )

        valid = pd.to_numeric(
            lag["proxy_masked"],
            errors="coerce",
        ).notna()

        bad = valid & (
            prediction.isna()
            | available.isna()
            | fiscal.isna()
            | expected.isna()
            | prediction.ne(expected)
            | available.gt(prediction)
            | fiscal.gt(prediction)
        )

        if bad.any():
            raise ValueError(
                f"{label} 미래 시점 연결 "
                f"{int(bad.sum())}건"
            )

        lag = lag[
            keys + ["proxy_masked"]
        ].rename(
            columns={
                "proxy_masked": f"debt_{label}"
            }
        )

        p = p.merge(
            lag,
            on=keys,
            how="left",
            validate="one_to_one",
        )

    # -----------------------------------
    # 3. DART + DataGuide 하이브리드
    # -----------------------------------
    # 주의: 완전히 PIT 검증된 데이터는 아님
    hyb = read_csv(paths["hybrid"])

    need = {
        "stock_id",
        "feature_month",
        "debt_final_raw",
        "prediction_date",
        "source_fiscal_date",
        "source_available_date",
        "source_type",
    }

    if need - set(hyb.columns):
        raise ValueError(
            "하이브리드 컬럼 부족: "
            f"{need - set(hyb.columns)}"
        )

    hyb["stock_id"] = norm_code(
        hyb["stock_id"]
    )

    hyb["feature_month"] = (
        hyb["feature_month"].str.slice(0, 7)
    )

    keys = [
        "stock_id",
        "feature_month",
    ]

    if (
        hyb[keys].isna().any().any()
        or hyb[keys].duplicated().any()
    ):
        raise ValueError(
            "hybrid 중복 또는 누락 종목월"
        )

    hyb["debt_final_raw"] = pd.to_numeric(
        hyb["debt_final_raw"],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    prediction = pd.to_datetime(
        hyb["prediction_date"],
        errors="coerce",
    )

    expected = (
        pd.to_datetime(
            hyb["feature_month"] + "-01",
            errors="coerce",
        )
        + pd.offsets.MonthEnd(0)
    )

    fiscal = pd.to_datetime(
        hyb["source_fiscal_date"],
        errors="coerce",
    )

    available = pd.to_datetime(
        hyb["source_available_date"],
        errors="coerce",
    )

    valid = hyb["debt_final_raw"].notna()

    # 공시 가능일 및 회계기간 검증
    invalid = valid & (
        prediction.isna()
        | expected.isna()
        | prediction.ne(expected)
        | fiscal.isna()
        | available.isna()
        | fiscal.gt(prediction)
        | available.gt(prediction)
    )

    if invalid.any():
        raise ValueError(
            "hybrid 이용시점/회계기간 위반 "
            f"{int(invalid.sum())}건"
        )

    valid_sources = [
        "DART_FIRST_FILED_CANDIDATE",
        "DATAGUIDE_6M_PROXY",
    ]

    if (
        valid
        & ~hyb["source_type"].isin(
            valid_sources
        )
    ).any():
        raise ValueError(
            "hybrid 출처 표기 누락 또는 미지원"
        )

    hybrid_provenance = (
        hyb.loc[valid, "source_type"]
        .value_counts()
        .to_dict()
    )

    hyb = hyb[
        keys + ["debt_final_raw"]
    ].rename(
        columns={
            "debt_final_raw": "debt_hybrid"
        }
    )

    p = p.merge(
        hyb,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------
    # 4. 시나리오별 월별 표준화
    # -----------------------------------
    for scenario in LAGS:
        p[f"z_{scenario}"] = (
            monthly_standardize(
                p,
                f"debt_{scenario}",
            )
        )

    # -----------------------------------
    # 5. 공통 표본 설정
    # -----------------------------------
    base_valid = p[
        ["policy", "log_size", "return_t1"]
    ].notna().all(axis=1)

    prior_common = (
        base_valid
        & p[
            ["z_legacy", "z_6m", "z_9m"]
        ].notna().all(axis=1)
    )

    common = (
        prior_common
        & p["z_hybrid"].notna()
    )

    if "eligible" in p.columns:
        eligible = (
            p["eligible"]
            .astype("string")
            .str.lower()
            .isin(["true", "1"])
        )

        common &= eligible
        prior_common &= eligible
        base_valid &= eligible

    p["common_sample"] = common

    print(
        f"전체 Train 행: {len(p):,} / "
        "기존 3시나리오 공통표본: "
        f"{int(prior_common.sum()):,} / "
        "4시나리오 공통표본: "
        f"{int(common.sum()):,}",
        flush=True,
    )

    # -----------------------------------
    # 6. Train Rank IC 검증
    # -----------------------------------
    monthly = []
    fold_log = []

    for sample_mode in (
        "common",
        "available",
    ):
        for scenario in LAGS:
            mask = (
                common
                if sample_mode == "common"
                else (
                    base_valid
                    & p[f"z_{scenario}"].notna()
                )
            )

            df = p.loc[mask].copy()

            df["debt"] = (
                df[f"z_{scenario}"]
            )

            df["policy_x_debt"] = (
                df["policy"] * df["debt"]
            )

            df = df.loc[
                df["return_t1"].notna()
                & df["policy"].notna()
                & df["log_size"].notna()
            ].copy()

            for fold, (start, end) in enumerate(
                FOLDS, 1
            ):
                tr = df.loc[
                    df["target_month"] < start
                ]

                va = df.loc[
                    df["target_month"].between(
                        start, end
                    )
                ]

                if len(tr) < 100 or len(va) < 100:
                    raise ValueError(
                        "표본 부족 "
                        f"{sample_mode}/"
                        f"{scenario}/fold={fold}"
                    )

                for alg in ALGORITHMS:
                    for feature_set, features in (
                        FEATURES.items()
                    ):
                        model = model_instance(alg)

                        xtr = tr[features].replace(
                            [np.inf, -np.inf],
                            np.nan,
                        )

                        xva = va[features].replace(
                            [np.inf, -np.inf],
                            np.nan,
                        )

                        model.fit(
                            xtr,
                            tr["return_t1"],
                        )

                        prediction = model.predict(
                            xva
                        )

                        results = calc_ic(
                            va,
                            prediction,
                            a.min_stocks,
                        )

                        model_name = (
                            f"{alg}_{feature_set}"
                        )

                        for mon, n, ic in results:
                            monthly.append({
                                "sample": sample_mode,
                                "scenario": scenario,
                                "target_month": mon,
                                "fold": fold,
                                "model": model_name,
                                "algorithm": alg,
                                "feature_set": feature_set,
                                "n_stocks": n,
                                "rank_ic": ic,
                            })

                        mean_ic = (
                            np.mean(
                                [r[2] for r in results]
                            )
                            if results
                            else np.nan
                        )

                        fold_log.append({
                            "sample": sample_mode,
                            "scenario": scenario,
                            "fold": fold,
                            "validation_start": start,
                            "validation_end": end,
                            "model": model_name,
                            "n_train": len(tr),
                            "n_val": len(va),
                            "n_months": len(results),
                            "mean_ic": mean_ic,
                        })

                        print(
                            f"{sample_mode:9s} "
                            f"{scenario:6s} "
                            f"fold{fold} "
                            f"{model_name:32s} "
                            f"IC={mean_ic:+.5f}",
                            flush=True,
                        )

    if not monthly:
        raise ValueError(
            "Rank IC가 계산되지 않았습니다. "
            "타깃과 검증 표본을 확인하세요."
        )

    # -----------------------------------
    # 7. 월별 및 Fold별 결과
    # -----------------------------------
    m = pd.DataFrame(monthly)
    f = pd.DataFrame(fold_log)

    m.to_csv(
        out / "monthly_rank_ic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    f.to_csv(
        out / "fold_rank_ic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        m.groupby([
            "sample",
            "scenario",
            "algorithm",
            "feature_set",
            "model",
        ])
        .agg(
            mean_ic=("rank_ic", "mean"),
            median_ic=("rank_ic", "median"),
            std_ic=("rank_ic", "std"),
            positive_month_share=(
                "rank_ic",
                lambda x: (x > 0).mean(),
            ),
            n_months=(
                "target_month", "nunique"
            ),
            mean_n_stocks=(
                "n_stocks", "mean"
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        out / "rank_ic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------
    # 8. 시나리오 간 Rank IC 변화
    # -----------------------------------
    pairs = []

    comparisons = [
        ("legacy", "6m"),
        ("legacy", "9m"),
        ("legacy", "hybrid"),
        ("6m", "hybrid"),
    ]

    for (sample, model), g in m.groupby(
        ["sample", "model"]
    ):
        wide = g.pivot(
            index="target_month",
            columns="scenario",
            values="rank_ic",
        )

        for baseline, candidate in comparisons:
            pair = wide[
                [baseline, candidate]
            ].dropna()

            if len(pair):
                delta = (
                    pair[candidate]
                    - pair[baseline]
                )

                pairs.append({
                    "sample": sample,
                    "model": model,
                    "comparison": (
                        f"{candidate}_minus_{baseline}"
                    ),
                    "n_paired_months": len(pair),
                    "baseline_mean_ic": (
                        pair[baseline].mean()
                    ),
                    "candidate_mean_ic": (
                        pair[candidate].mean()
                    ),
                    "delta_mean_ic": delta.mean(),
                    "improved_month_share": (
                        (delta > 0).mean()
                    ),
                })

    pd.DataFrame(pairs).to_csv(
        out / "paired_ic_changes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------
    # 9. 표본 커버리지
    # -----------------------------------
    cov = []

    for scenario in LAGS:
        cov.append({
            "scenario": scenario,
            "raw_nonmissing": int(
                p[f"debt_{scenario}"].notna().sum()
            ),
            "z_nonmissing": int(
                p[f"z_{scenario}"].notna().sum()
            ),
            "train_model_ready": int(
                (
                    base_valid
                    & p[f"z_{scenario}"].notna()
                ).sum()
            ),
            "common_sample": int(
                common.sum()
            ),
        })

    pd.DataFrame(cov).to_csv(
        out / "sample_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------
    # 10. 실행 설정 및 재현성 기록
    # -----------------------------------
    config = {
        "train_only": True,
        "folds": FOLDS,
        "inputs_sha256": {
            key: hashfile(path)
            for key, path in paths.items()
        },
        "policy": "final_rule_v2_sensitivity",
        "target": "return_t1",
        "models": ALGORITHMS,
        "feature_sets": FEATURES,
        "scenarios": list(LAGS),
        "old_three_way_common_rows": int(
            prior_common.sum()
        ),
        "four_way_common_rows": int(
            common.sum()
        ),
        "hybrid_source_counts_in_input": (
            hybrid_provenance
        ),
        "note": (
            "PROVISIONAL: 6m/9m and DART-DG hybrid "
            "are NOT fully PIT verified. "
            "Common sample is comparable across "
            "scenarios; available sample can differ. "
            "No Test targets used for model selection."
        ),
    }

    (
        out / "run_config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== 공통 표본 8개 모델 ==="
    )

    print(
        summary.loc[
            summary["sample"].eq("common"),
            [
                "scenario",
                "model",
                "mean_ic",
                "n_months",
            ],
        ].to_string(index=False)
    )

    print(
        "\n완료:", out, flush=True
    )


if __name__ == "__main__":
    main()
