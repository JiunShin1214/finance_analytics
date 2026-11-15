
"""Train-only DART + DataGuide hybrid debt dataset.

DART first-filed candidates are used only before an unresolved
subsequent filing. Otherwise, use the DataGuide 6-month proxy.
This is NOT a fully verified historical PIT dataset.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from src.h2.data_io import ROOT

DEFAULT_PROXY = (
    ROOT / "data/processed/final/debt"
    / "h2_debt_lag_6m_fiscal_screened_2001_2021.csv"
)
DEFAULT_AUDIT = (
    ROOT / "results/validation/dart_versions"
    / "dart_event_version_audit.csv"
)
DEFAULT_OUTPUT = ROOT / "data/processed/final/debt_hybrid"

KEY = ["stock_id", "feature_month"]


def stocks(s):
    s = s.astype("string").str.strip()
    s = s.str.replace(r"^A(?=\d{6}$)", "", regex=True)
    s = s.str.replace(r"\.0$", "", regex=True)

    if not s.str.fullmatch(r"\d{1,6}", na=False).all():
        raise ValueError("유효하지 않은 종목코드가 있습니다.")

    return s.str.zfill(6)


def yes(s):
    return s.astype("string").str.lower().isin(
        ["true", "1", "yes"]
    )


def read_proxy(path):
    x = pd.read_csv(
        path,
        dtype={
            "stock_id": "string",
            "feature_month": "string",
        },
        low_memory=False,
    )

    required = set(
        KEY + [
            "proxy_masked",
            "prediction_date",
            "fiscal_date",
            "proxy_available_date",
        ]
    )

    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"6개월 패널 컬럼 누락: {missing}")

    x = x[
        x.feature_month.between("2001-01", "2021-12")
    ].copy()

    x["stock_id"] = stocks(x["stock_id"])

    if x.empty or x.duplicated(KEY).any():
        raise ValueError("6개월 패널의 종목·월 중복 또는 빈 데이터")

    expected = (
        pd.to_datetime(x.feature_month + "-01")
        + pd.offsets.MonthEnd(0)
    )

    for c in [
        "prediction_date",
        "fiscal_date",
        "proxy_available_date",
    ]:
        x[c] = pd.to_datetime(x[c], errors="coerce")

    if (
        x.prediction_date.isna().any()
        or not x.prediction_date.eq(expected).all()
    ):
        raise ValueError("예측월과 월말 날짜가 일치하지 않습니다.")

    x["proxy_masked"] = pd.to_numeric(
        x.proxy_masked, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    valid = x.proxy_masked.notna()

    invalid = valid & (
        x.fiscal_date.isna()
        | x.proxy_available_date.isna()
        | x.fiscal_date.gt(x.prediction_date)
        | x.proxy_available_date.gt(x.prediction_date)
    )

    if invalid.any():
        raise ValueError("DataGuide 입력에 시점 오류가 있습니다.")

    return x.rename(
        columns={
            "proxy_masked": "dg_ratio",
            "fiscal_date": "dg_fiscal_date",
            "proxy_available_date": "dg_available_date",
        }
    )


def read_dart(path):
    e = pd.read_csv(
        path,
        dtype={
            "stock_code": "string",
            "rcept_no": "string",
        },
        low_memory=False,
    )

    required = {
        "stock_code",
        "rcept_no",
        "fiscal_date",
        "disclosure_date",
        "next_filing_date",
        "debt_ratio_pct",
        "source_status",
        "version_audit_status",
        "initial_candidate",
        "match_rounded2",
        "date_matches_history",
        "title_year_matches",
        "title_is_dec",
        "original_fiscal_before_filing",
    }

    missing = required - set(e.columns)
    if missing:
        raise ValueError(f"DART 감사 컬럼 누락: {missing}")

    e["stock_id"] = stocks(e["stock_code"])

    for c in [
        "fiscal_date",
        "disclosure_date",
        "next_filing_date",
    ]:
        e[c] = pd.to_datetime(e[c], errors="coerce")

    e["debt_ratio_pct"] = pd.to_numeric(
        e.debt_ratio_pct, errors="coerce"
    )

    # 최초 제출 메타데이터와 원문 추출 조건 확인
    eligible = (
        yes(e.initial_candidate)
        & e.version_audit_status.eq(
            "FIRST_FILING_METADATA_MATCH_CANDIDATE"
        )
        & e.source_status.eq(
            "STRICT_ORIGINAL_CFS_CURRENT_CANDIDATE"
        )
        & yes(e.match_rounded2)
        & yes(e.date_matches_history)
        & yes(e.title_year_matches)
        & yes(e.title_is_dec)
        & yes(e.original_fiscal_before_filing)
        & e.fiscal_date.notna()
        & e.disclosure_date.notna()
        & e.fiscal_date.lt(e.disclosure_date)
        & e.debt_ratio_pct.notna()
        & np.isfinite(e.debt_ratio_pct)
        & e.debt_ratio_pct.gt(0)
    )

    e = e.loc[eligible].copy()

    if e.duplicated(["stock_id", "fiscal_date"]).any():
        raise ValueError("동일 결산기간의 최초 공시 중복")

    invalid_next = (
        e.next_filing_date.notna()
        & e.next_filing_date.lt(e.disclosure_date)
    )

    if invalid_next.any():
        raise ValueError("후속 공시 날짜 오류")

    # 게시 시간 불명확: 공시 다음 날부터 사용
    e["dart_available_date"] = (
        e.disclosure_date + pd.Timedelta(days=1)
    )

    e = e.rename(
        columns={
            "debt_ratio_pct": "dart_ratio",
            "fiscal_date": "dart_fiscal_date",
            "disclosure_date": "dart_disclosure_date",
            "next_filing_date": "dart_next_filing_date",
            "rcept_no": "dart_rcept_no",
        }
    )

    return e[
        [
            "stock_id",
            "dart_ratio",
            "dart_fiscal_date",
            "dart_disclosure_date",
            "dart_available_date",
            "dart_next_filing_date",
            "dart_rcept_no",
        ]
    ]


def build(proxy, dart):
    panel = proxy.copy()

    # 기업별 DART 공시 후보와 월별 패널 결합
    expanded = panel[
        KEY + ["prediction_date"]
    ].merge(
        dart,
        on="stock_id",
        how="left",
    )

    cutoff = expanded.prediction_date

    # 공시일 이전에는 사용할 수 없음.
    # 후속 공시가 발생하면 최초 버전은 보류.
    valid = (
        expanded.dart_ratio.notna()
        & expanded.dart_available_date.le(cutoff)
        & expanded.dart_fiscal_date.le(cutoff)
        & (
            expanded.dart_next_filing_date.isna()
            | expanded.dart_next_filing_date.gt(cutoff)
        )
    )

    # 가장 최근 공시가 아니라
    # 가장 최신 결산기간의 유효한 공시를 선택
    eligible = expanded.loc[valid].sort_values(
        KEY + [
            "dart_fiscal_date",
            "dart_disclosure_date",
            "dart_rcept_no",
        ],
        ascending=[True, True, False, False, False],
    )

    selected = eligible.drop_duplicates(KEY)[
        KEY + [
            "dart_ratio",
            "dart_fiscal_date",
            "dart_disclosure_date",
            "dart_available_date",
            "dart_next_filing_date",
            "dart_rcept_no",
        ]
    ]

    # 후속 공시 후 수치 미검증 구간 표시
    blocked = (
        expanded.dart_ratio.notna()
        & expanded.dart_available_date.le(cutoff)
        & expanded.dart_next_filing_date.notna()
        & expanded.dart_next_filing_date.le(cutoff)
    )

    blocked_keys = expanded.loc[
        blocked, KEY
    ].drop_duplicates()

    blocked_keys["has_unresolved_followup"] = True

    x = panel.merge(
        selected,
        on=KEY,
        how="left",
        validate="1:1",
    )

    x = x.merge(
        blocked_keys,
        on=KEY,
        how="left",
        validate="1:1",
    )

    x["has_unresolved_followup"] = (
        x.has_unresolved_followup.eq(True)
    )

    # DART가 DataGuide보다 최신 또는 동일 결산기간일 때 우선
    take_dart = (
        x.dart_ratio.notna()
        & (
            x.dg_ratio.isna()
            | x.dart_fiscal_date.ge(x.dg_fiscal_date)
        )
    )

    x["debt_final_raw"] = x.dg_ratio.where(
        ~take_dart, x.dart_ratio
    )

    x["source_type"] = np.select(
        [take_dart, x.dg_ratio.notna()],
        [
            "DART_FIRST_FILED_CANDIDATE",
            "DATAGUIDE_6M_PROXY",
        ],
        default="MISSING",
    )

    x["source_fiscal_date"] = x.dg_fiscal_date.where(
        ~take_dart, x.dart_fiscal_date
    )

    x["source_available_date"] = (
        x.dg_available_date.where(
            ~take_dart, x.dart_available_date
        )
    )

    x["source_rcept_no"] = (
        x.dart_rcept_no.where(take_dart)
    )

    x["source_confidence"] = np.select(
        [
            take_dart,
            x.dg_ratio.notna() & x.has_unresolved_followup,
            x.dg_ratio.notna(),
        ],
        [
            "DART_FIRST_VERSION_NOT_FULL_PIT",
            "DG_6M_PROXY_UNRESOLVED_REVISION",
            "DG_6M_PROXY_VINTAGE_UNVERIFIED",
        ],
        default="NO_DATA",
    )

    # 날짜 조건 검사
    active = x.debt_final_raw.notna()

    invalid = active & (
        x.source_fiscal_date.isna()
        | x.source_available_date.isna()
        | x.source_fiscal_date.gt(x.prediction_date)
        | x.source_available_date.gt(x.prediction_date)
    )

    if invalid.any():
        raise AssertionError(
            f"미래정보 시점 조건 위반: {int(invalid.sum())}건"
        )

    # 최종 부채비율 기준 월별 재표준화
    def standardize(group):
        values = group.debt_final_raw.replace(
            [np.inf, -np.inf], np.nan
        )

        if values.notna().sum() < 2:
            group["debt_final_winsor"] = np.nan
            group["debt_final_z"] = np.nan
            return group

        low, high = values.quantile([0.01, 0.99])
        clipped = values.clip(low, high)

        group["debt_final_winsor"] = clipped

        sd = clipped.std(ddof=0)

        group["debt_final_z"] = (
            (clipped - clipped.mean()) / sd
            if sd > 0 else np.nan
        )

        return group

    x = pd.concat(
        [
            standardize(group.copy())
            for _, group in x.groupby(
                "feature_month", sort=False
            )
        ],
        ignore_index=True,
    ).sort_values(KEY)

    return (
        x,
        int(blocked_keys.shape[0]),
        int(take_dart.sum()),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--proxy",
        type=Path,
        default=DEFAULT_PROXY,
    )

    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    for path in [args.proxy, args.audit]:
        if not path.is_file():
            raise FileNotFoundError(path)

    proxy = read_proxy(args.proxy)
    dart = read_dart(args.audit)

    result, blocked, dart_rows = build(proxy, dart)

    args.output.mkdir(parents=True, exist_ok=True)

    output_file = (
        args.output
        / "h2_debt_hybrid_candidate_2001_2021.csv"
    )

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    coverage = result.groupby(
        "feature_month", as_index=False
    ).agg(
        rows=("stock_id", "size"),
        hybrid_rows=("debt_final_raw", "count"),
        dart_rows=(
            "source_type",
            lambda s: int(
                (s == "DART_FIRST_FILED_CANDIDATE").sum()
            ),
        ),
        revision_risk_rows=(
            "has_unresolved_followup", "sum"
        ),
    )

    coverage["coverage"] = (
        coverage.hybrid_rows / coverage.rows
    )

    coverage.to_csv(
        args.output / "coverage_by_month.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "total_rows": len(result),
        "dg_proxy_rows": int(
            result.dg_ratio.notna().sum()
        ),
        "eligible_dart_events": len(dart),
        "dart_selected_rows": dart_rows,
        "unresolved_followup_stock_months": blocked,
        "hybrid_rows": int(
            result.debt_final_raw.notna().sum()
        ),
        "hybrid_coverage": float(
            result.debt_final_raw.notna().mean()
        ),
        "date_violations": 0,
        "historical_pit_verified": False,
        "final_model_approved": False,
    }

    (args.output / "summary.json").write_text(
        json.dumps(
            summary, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        summary, ensure_ascii=False, indent=2
    ))
    print("저장 완료:", output_file)


if __name__ == "__main__":
    main()
