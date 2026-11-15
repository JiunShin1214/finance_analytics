
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.h2.data_io import ROOT
BASE = ROOT / "data/processed/final"

PROXY_FILE = (
    BASE / "debt"
    / "h2_debt_lag_6m_fiscal_screened_2001_2021.csv"
)
DART_FILE = (
    BASE / "debt_disclosure"
    / "h2_debt_disclosure_asof_candidate.csv"
)
OUT = BASE / "debt_comparison"


def normalize_id(series):
    s = series.astype("string").str.strip()
    s = s.str.replace(r"^A(?=\d{6}$)", "", regex=True)
    s = s.str.replace(r"\.0$", "", regex=True)

    if not s.str.fullmatch(r"\d{1,6}", na=False).all():
        raise ValueError("잘못된 종목코드가 있습니다.")

    return s.str.zfill(6)


def read_panel(path, required):
    if not path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "stock_id": "string",
            "feature_month": "string",
        },
        low_memory=False,
    )

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} 누락 컬럼: {sorted(missing)}"
        )

    df["stock_id"] = normalize_id(df["stock_id"])
    df["feature_month"] = (
        df["feature_month"].str.slice(0, 7)
    )

    keys = ["stock_id", "feature_month"]

    if (
        df[keys].isna().any().any()
        or df.duplicated(keys).any()
    ):
        raise ValueError(f"{path.name}: 종목·월 키 오류")

    return df


def main():
    proxy = read_panel(
        PROXY_FILE,
        [
            "stock_id",
            "feature_month",
            "proxy_masked",
            "prediction_date",
            "proxy_available_date",
            "fiscal_date",
        ],
    )

    dart = read_panel(
        DART_FILE,
        [
            "stock_id",
            "feature_month",
            "debt_disclosure_raw",
            "prediction_date",
            "source_disclosure_date",
            "source_fiscal_date",
            "asof_available_date",
        ],
    )

    proxy = proxy[
        proxy.feature_month.between("2001-01", "2021-12")
    ].copy()

    dart = dart[
        dart.feature_month.between("2001-01", "2021-12")
    ].copy()

    proxy["proxy_value"] = pd.to_numeric(
        proxy["proxy_masked"], errors="coerce"
    )

    dart["dart_value"] = pd.to_numeric(
        dart["debt_disclosure_raw"], errors="coerce"
    )

    proxy = proxy.rename(
        columns={
            "prediction_date": "prediction_date_proxy",
            "fiscal_date": "proxy_fiscal_date",
        }
    )

    dart = dart.rename(
        columns={
            "prediction_date": "prediction_date_dart",
            "source_fiscal_date": "dart_fiscal_date",
        }
    )

    proxy_cols = [
        "stock_id",
        "feature_month",
        "prediction_date_proxy",
        "proxy_value",
        "proxy_fiscal_date",
        "proxy_available_date",
    ]

    dart_cols = [
        "stock_id",
        "feature_month",
        "prediction_date_dart",
        "dart_value",
        "dart_fiscal_date",
        "source_disclosure_date",
        "asof_available_date",
    ]

    for col in [
        "source_rcept_no",
        "source_quality",
        "timing_status",
    ]:
        if col in dart.columns:
            dart_cols.append(col)

    df = proxy[proxy_cols].merge(
        dart[dart_cols],
        on=["stock_id", "feature_month"],
        how="left",
        validate="one_to_one",
    )

    df["prediction_date"] = pd.to_datetime(
        df["prediction_date_proxy"], errors="coerce"
    )

    dart_prediction = pd.to_datetime(
        df["prediction_date_dart"], errors="coerce"
    )

    if (
        dart_prediction.notna()
        & dart_prediction.ne(df["prediction_date"])
    ).any():
        raise ValueError(
            "DataGuide와 DART의 예측 기준일 불일치"
        )

    date_columns = [
        "proxy_fiscal_date",
        "proxy_available_date",
        "dart_fiscal_date",
        "source_disclosure_date",
        "asof_available_date",
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col], errors="coerce"
        )

    # 6개월 지연 자료의 시점 조건
    proxy_ok = (
        df.proxy_value.notna()
        & np.isfinite(df.proxy_value)
        & df.proxy_fiscal_date.notna()
        & df.proxy_available_date.notna()
        & df.proxy_fiscal_date.le(df.prediction_date)
        & df.proxy_available_date.le(df.prediction_date)
    )

    # DART 후보의 시점 조건
    dart_ok = (
        df.dart_value.notna()
        & np.isfinite(df.dart_value)
        & df.dart_fiscal_date.notna()
        & df.source_disclosure_date.notna()
        & df.asof_available_date.notna()
        & df.dart_fiscal_date.le(df.prediction_date)
        & df.source_disclosure_date.lt(df.prediction_date)
        & df.asof_available_date.le(df.prediction_date)
    )

    if (df.proxy_value.notna() & ~proxy_ok).any():
        raise ValueError("6개월 자료에서 시점 오류 발견")

    if (df.dart_value.notna() & ~dart_ok).any():
        raise ValueError("DART 자료에서 시점 오류 발견")

    # 공통 관측치 비교
    df["both_available"] = proxy_ok & dart_ok

    df["difference_pp"] = (
        df.dart_value - df.proxy_value
    ).where(df.both_available)

    df["abs_difference_pp"] = df.difference_pp.abs()

    df["same_fiscal_date"] = (
        df.both_available
        & df.dart_fiscal_date.eq(df.proxy_fiscal_date)
    )

    df["dart_newer_fiscal"] = (
        df.both_available
        & df.dart_fiscal_date.gt(df.proxy_fiscal_date)
    )

    df["dart_older_fiscal"] = (
        df.both_available
        & df.dart_fiscal_date.lt(df.proxy_fiscal_date)
    )

    # DART가 없으면 6개월 근사값 사용.
    # DART의 회계기간이 더 오래됐다면 덮어쓰지 않음.
    take_dart = dart_ok & (
        ~proxy_ok
        | df.dart_fiscal_date.ge(df.proxy_fiscal_date)
    )

    df["hybrid_candidate_raw"] = (
        df.proxy_value.where(~take_dart, df.dart_value)
    )

    df["hybrid_source"] = np.select(
        [take_dart, proxy_ok],
        [
            "DART_CANDIDATE_NOT_VERIFIED_PIT",
            "DG_6M_PROXY",
        ],
        default="MISSING",
    )

    df["hybrid_fiscal_date"] = (
        df.proxy_fiscal_date.where(
            ~take_dart, df.dart_fiscal_date
        )
    )

    df["hybrid_available_date"] = (
        df.proxy_available_date.where(
            ~take_dart, df.asof_available_date
        )
    )

    # 월별 횡단면 이상치 처리와 재표준화
    df["hybrid_winsor"] = np.nan
    df["hybrid_z"] = np.nan

    for month, group in df.groupby(
        "feature_month", sort=False
    ):
        values = group.hybrid_candidate_raw.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if len(values) < 2:
            continue

        lower, upper = values.quantile([0.01, 0.99])

        clipped = group.hybrid_candidate_raw.clip(
            lower, upper
        )

        df.loc[group.index, "hybrid_winsor"] = clipped

        std = clipped.std(ddof=0)

        if pd.notna(std) and std > 0:
            df.loc[group.index, "hybrid_z"] = (
                (clipped - clipped.mean()) / std
            )

    # 최종 시점 검사
    invalid = (
        df.hybrid_candidate_raw.notna()
        & (
            df.hybrid_available_date.isna()
            | df.hybrid_available_date.gt(
                df.prediction_date
            )
            | df.hybrid_fiscal_date.gt(
                df.prediction_date
            )
        )
    )

    if invalid.any():
        raise AssertionError(
            "하이브리드 후보에 미래 시점 오류 발생"
        )

    # 월별 커버리지
    coverage = (
        df.groupby("feature_month", as_index=False)
        .agg(
            rows=("stock_id", "size"),
            proxy_rows=("proxy_value", "count"),
            dart_rows=("dart_value", "count"),
            both_rows=("both_available", "sum"),
            hybrid_rows=("hybrid_candidate_raw", "count"),
        )
    )

    for name in ["proxy", "dart", "hybrid"]:
        coverage[f"{name}_coverage"] = (
            coverage[f"{name}_rows"] / coverage["rows"]
        )

    same = df.loc[
        df.same_fiscal_date, "abs_difference_pp"
    ]

    summary = {
        "rows": len(df),
        "proxy_rows": int(proxy_ok.sum()),
        "dart_rows": int(dart_ok.sum()),
        "both_rows": int(df.both_available.sum()),
        "dart_only_rows": int(
            (dart_ok & ~proxy_ok).sum()
        ),
        "proxy_only_rows": int(
            (proxy_ok & ~dart_ok).sum()
        ),
        "same_fiscal_date_rows": int(
            df.same_fiscal_date.sum()
        ),
        "dart_newer_fiscal_rows": int(
            df.dart_newer_fiscal.sum()
        ),
        "dart_older_fiscal_rows": int(
            df.dart_older_fiscal.sum()
        ),
        "same_fiscal_date_abs_diff_median_pp": (
            float(same.median()) if len(same) else None
        ),
        "hybrid_rows": int(
            df.hybrid_candidate_raw.notna().sum()
        ),
        "hybrid_dart_rows": int(take_dart.sum()),
        "future_joins": int(invalid.sum()),
        "fully_pit_verified": False,
        "final_training_ready": False,
        "note": (
            "DART 최초 공시 및 정정 이력 미검증. "
            "하이브리드 결과는 최종 학습용이 아닌 "
            "비교 실험 후보임."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)

    df.sort_values(
        ["feature_month", "stock_id"]
    ).to_csv(
        OUT / "debt_monthly_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coverage.to_csv(
        OUT / "coverage_by_month.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (OUT / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("결과 저장:", OUT)


if __name__ == "__main__":
    main()
