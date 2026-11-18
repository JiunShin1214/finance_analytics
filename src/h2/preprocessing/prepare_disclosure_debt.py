# src/prepare_disclosure_debt.py

"""
DART 실제 공시일 기반 부채비율 후보 패널 생성.

주의:
- 최종적으로 검증된 PIT 데이터가 아님
- 2018~2021년 사업보고서 원문 후보 중심
- 원본 패널과 기존 6개월 지연 데이터는 변경하지 않음
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.h2.data_io import ROOT, PROCESSED, write_csv, write_json


ANNUAL_TITLE = re.compile(
    r"사업보고서\s*\(\s*(\d{4})\s*\.\s*(\d{1,2})\s*\)"
)

CORRECTION = re.compile(r"정정|첨부추가|철회")


def normalize_codes(series):
    """종목코드를 6자리 문자열로 통일한다."""

    result = series.astype("string").str.strip()

    result = result.str.replace(
        r"^A(?=\d{6}$)", "", regex=True
    )

    result = result.str.replace(
        r"\.0$", "", regex=True
    )

    valid = result.str.fullmatch(r"\d{1,6}", na=False)

    if not valid.all():
        raise ValueError(
            f"잘못된 종목코드: {result[~valid].head().tolist()}"
        )

    return result.str.zfill(6).astype(str)


def prepare_events(
    events_path: Path,
    history_path: Path,
    require_dg_match: bool = True,
):
    """
    DART 원문에서 추출한 부채비율을 공시 메타데이터와 대조한다.

    공시 제목, 회계연도, 정정 여부 및 DataGuide 일치 여부를
    확인하고 승인 가능한 후보 이벤트를 선택한다.
    """

    events = pd.read_csv(
        events_path,
        dtype={"stock_code": str, "rcept_no": str},
        low_memory=False,
    )

    required = {
        "stock_code",
        "rcept_no",
        "fiscal_year",
        "disclosure_date",
        "debt_ratio_pct",
    }

    missing = required - set(events.columns)

    if missing:
        raise ValueError(
            f"이벤트 파일 필수 컬럼 누락: {sorted(missing)}"
        )

    events["stock_code"] = normalize_codes(
        events["stock_code"]
    )

    events["rcept_no"] = (
        events["rcept_no"].astype("string").str.strip()
    )

    events["fiscal_year"] = pd.to_numeric(
        events["fiscal_year"], errors="coerce"
    )

    events["disclosure_date"] = pd.to_datetime(
        events["disclosure_date"], errors="coerce"
    )

    events["debt_ratio_pct"] = pd.to_numeric(
        events["debt_ratio_pct"], errors="coerce"
    )

    events["reason"] = ""

    # DataGuide 수치 일치 여부는 추출 품질 검사에만 사용한다.
    if require_dg_match:
        if "match_rounded2" not in events.columns:
            raise ValueError(
                "match_rounded2 컬럼이 없습니다. "
                "DataGuide 대조 결과를 입력하세요."
            )

        matched = (
            events["match_rounded2"]
            .astype("string")
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        events.loc[
            ~matched, "reason"
        ] = "DG_MISMATCH_OR_MISSING"

    # DART 공시 목록과 접수번호 연결
    history = pd.read_csv(
        history_path,
        dtype={"stock_code": str, "rcept_no": str},
        low_memory=False,
    )

    history_required = {
        "stock_code", "rcept_no", "report_nm"
    }

    if history_required - set(history.columns):
        raise ValueError(
            "DART 공시 메타데이터에 "
            "stock_code, rcept_no, report_nm이 필요합니다."
        )

    history["stock_code"] = normalize_codes(
        history["stock_code"]
    )

    history["rcept_no"] = (
        history["rcept_no"].astype("string").str.strip()
    )

    if history.duplicated(
        ["stock_code", "rcept_no"]
    ).any():
        raise ValueError(
            "DART 메타데이터의 종목·접수번호가 중복됩니다."
        )

    history = history[
        ["stock_code", "rcept_no", "report_nm"]
    ].rename(
        columns={"report_nm": "_dart_report_nm"}
    )

    events = events.merge(
        history,
        on=["stock_code", "rcept_no"],
        how="left",
        validate="m:1",
    )

    events["report_nm"] = events["_dart_report_nm"]

    events["title_year"] = np.nan
    events["title_month"] = np.nan
    events["amendment_in_title"] = False

    for idx, name in events["report_nm"].dropna().items():

        name = str(name)
        match = ANNUAL_TITLE.search(name)

        events.at[idx, "amendment_in_title"] = bool(
            CORRECTION.search(name)
        )

        if match:
            year, month = map(int, match.groups())

            events.at[idx, "title_year"] = year
            events.at[idx, "title_month"] = month

    # 사업보고서 제목 확인
    missing_title = events["title_year"].isna()

    events.loc[
        events["reason"].eq("") & missing_title,
        "reason",
    ] = "NO_VERIFIABLE_ANNUAL_TITLE"

    # 비12월 결산은 실제 재무제표 기준일 검증 전까지 보류
    non_dec = (
        events["title_month"].notna()
        & events["title_month"].ne(12)
    )

    events.loc[
        events["reason"].eq("") & non_dec,
        "reason",
    ] = "NON_DEC_REQUIRES_STATEMENT_DATE"

    # 회계연도 불일치
    wrong_year = (
        events["title_year"].notna()
        & events["title_year"].ne(events["fiscal_year"])
    )

    events.loc[
        events["reason"].eq("") & wrong_year,
        "reason",
    ] = "FISCAL_YEAR_MISMATCH"

    # 정정·첨부추가 원문은 버전 확인 전까지 보류
    events.loc[
        events["reason"].eq("")
        & events["amendment_in_title"],
        "reason",
    ] = "AMENDMENT_REQUIRES_REVIEW"

    invalid = (
        events["disclosure_date"].isna()
        | events["fiscal_year"].isna()
        | events["debt_ratio_pct"].isna()
        | events["debt_ratio_pct"].le(0)
    )

    events.loc[
        events["reason"].eq("") & invalid,
        "reason",
    ] = "INVALID_DATE_OR_RATIO"

    # 12월 결산 후보에 한해 12월 31일 기준일 생성
    fiscal_year = (
        events["fiscal_year"]
        .astype("Int64")
        .astype("string")
    )

    events["fiscal_date_candidate"] = pd.to_datetime(
        fiscal_year + "-12-31",
        errors="coerce",
    )

    invalid_timing = (
        events["disclosure_date"]
        <= events["fiscal_date_candidate"]
    )

    events.loc[
        events["reason"].eq("") & invalid_timing,
        "reason",
    ] = "DISCLOSED_BEFORE_FISCAL_END"

    # DART 접수번호 형식 확인
    invalid_receipt = ~events[
        "rcept_no"
    ].str.fullmatch(r"\d{14}", na=False)

    events.loc[
        events["reason"].eq("") & invalid_receipt,
        "reason",
    ] = "INVALID_RECEIPT_NUMBER"

    # 동일 기업·회계연도의 복수 후보는 자동 선택하지 않음
    selected = events.loc[
        events["reason"].eq("")
    ].copy()

    duplicate_year = selected.duplicated(
        ["stock_code", "fiscal_year"],
        keep=False,
    )

    events.loc[
        selected.index[duplicate_year],
        "reason",
    ] = "MULTIPLE_VERSIONS_REQUIRE_REVIEW"

    selected = events.loc[
        events["reason"].eq("")
    ].copy()

    selected["fiscal_date"] = (
        selected["fiscal_date_candidate"]
    )

    selected["available_date"] = (
        selected["disclosure_date"]
    )

    selected["source_quality"] = (
        "DART_ANNUAL_CANDIDATE_NOT_FULL_PIT"
    )

    return selected, events


def build_monthly_panel(
    panel_path: Path,
    events_path: Path,
    history_path: Path,
    output: Path,
    start: str,
    end: str,
):
    """월말에 실제로 공개된 최신 DART 후보값을 연결한다."""

    accepted, audit = prepare_events(
        events_path,
        history_path,
    )

    panel = pd.read_csv(
        panel_path,
        dtype={
            "stock_id": str,
            "feature_month": str,
            "target_month": str,
        },
        low_memory=False,
    )

    panel = panel.loc[
        panel["feature_month"].between(start, end)
    ].copy()

    if panel.empty:
        raise ValueError("요청한 기간의 H2 패널이 없습니다.")

    panel["stock_id"] = normalize_codes(
        panel["stock_id"]
    )

    if panel.duplicated(
        ["stock_id", "feature_month"]
    ).any():
        raise ValueError("종목·월 중복이 있습니다.")

    if "target_month" not in panel.columns:
        panel["target_month"] = (
            pd.PeriodIndex(
                panel["feature_month"], freq="M"
            ) + 1
        ).astype(str)

    panel["prediction_date"] = (
        pd.to_datetime(
            panel["feature_month"] + "-01"
        ) + pd.offsets.MonthEnd(0)
    )

    # 모델 결합에 필요한 키만 사용
    panel = panel[
        [
            "stock_id",
            "feature_month",
            "target_month",
            "prediction_date",
        ]
    ].copy()

    events = accepted.rename(
        columns={
            "stock_code": "stock_id",
            "debt_ratio_pct": "debt_disclosure_raw",
            "fiscal_year": "source_fiscal_year",
            "fiscal_date": "source_fiscal_date",
            "disclosure_date": "source_disclosure_date",
            "rcept_no": "source_rcept_no",
        }
    )

    events = events[
        [
            "stock_id",
            "source_disclosure_date",
            "debt_disclosure_raw",
            "source_fiscal_year",
            "source_fiscal_date",
            "source_rcept_no",
            "source_quality",
        ]
    ].copy()

    # 공시 당일 이용 가능 시각이 확인되지 않았으므로
    # 공시 다음 날부터 사용 가능한 것으로 처리
    events["asof_available_date"] = (
        events["source_disclosure_date"]
        + pd.Timedelta(days=1)
    )

    events = events.sort_values(
        [
            "asof_available_date",
            "stock_id",
            "source_rcept_no",
        ]
    )

    # 동일 일자에 복수 후보가 있다면 검토 없이 임의 선택 금지
    duplicate_date = events.duplicated(
        ["stock_id", "asof_available_date"],
        keep=False,
    )

    events = events.loc[~duplicate_date].copy()

    left = panel.sort_values(
        ["prediction_date", "stock_id"]
    ).reset_index(drop=True)

    if events.empty:
        result = left.copy()

        result["debt_disclosure_raw"] = np.nan
        result["source_fiscal_year"] = np.nan
        result["source_fiscal_date"] = pd.NaT
        result["source_disclosure_date"] = pd.NaT
        result["source_rcept_no"] = pd.NA
        result["source_quality"] = pd.NA
        result["asof_available_date"] = pd.NaT

    else:
        right = events.sort_values(
            ["asof_available_date", "stock_id"]
        ).reset_index(drop=True)

        result = pd.merge_asof(
            left,
            right,
            left_on="prediction_date",
            right_on="asof_available_date",
            by="stock_id",
            direction="backward",
        )

    result["candidate_available"] = (
        result["debt_disclosure_raw"].notna()
    )

    result["future_disclosure_violation"] = (
        result["candidate_available"]
        & (
            result["source_disclosure_date"]
            >= result["prediction_date"]
        )
    )

    result["future_fiscal_violation"] = (
        result["candidate_available"]
        & (
            result["source_fiscal_date"]
            > result["prediction_date"]
        )
    )

    if (
        result["future_disclosure_violation"].any()
        or result["future_fiscal_violation"].any()
    ):
        raise AssertionError(
            "미래 공시정보가 결합되었습니다."
        )

    # 월별 횡단면 이상치 처리 및 표준화
    result["debt_disclosure_winsor"] = np.nan
    result["debt_disclosure_z"] = np.nan

    for month, group in result.groupby("feature_month"):

        values = group["debt_disclosure_raw"].replace(
            [np.inf, -np.inf], np.nan
        )

        available = values.dropna()

        if len(available) < 2:
            continue

        lower, upper = available.quantile(
            [0.01, 0.99]
        )

        clipped = values.clip(lower, upper)

        result.loc[
            group.index, "debt_disclosure_winsor"
        ] = clipped

        std = clipped.std(ddof=0)

        if pd.notna(std) and std > 0:
            result.loc[
                group.index, "debt_disclosure_z"
            ] = (
                (clipped - clipped.mean()) / std
            )

    result["timing_status"] = np.where(
        result["candidate_available"],
        "DART_CANDIDATE_NOT_FULL_PIT",
        "NO_DART_CANDIDATE",
    )

    output.mkdir(parents=True, exist_ok=True)

    write_csv(
        audit,
        output / "filing_event_audit.csv",
    )

    write_csv(
        accepted,
        output / "accepted_filing_events.csv",
    )

    write_csv(
        result,
        output / "h2_debt_disclosure_asof_candidate.csv",
    )

    coverage = (
        result.groupby("feature_month")
        .agg(
            rows=("stock_id", "size"),
            candidate_rows=("candidate_available", "sum"),
        )
        .reset_index()
    )

    coverage["coverage"] = (
        coverage["candidate_rows"] / coverage["rows"]
    )

    write_csv(
        coverage,
        output / "coverage_by_month.csv",
    )

    summary = {
        "input_events": int(len(audit)),
        "accepted_events": int(len(accepted)),
        "rejected_events": int(
            (audit["reason"] != "").sum()
        ),
        "panel_rows": int(len(result)),
        "candidate_rows": int(
            result["candidate_available"].sum()
        ),
        "coverage": round(
            float(result["candidate_available"].mean()),
            6,
        ),
        "future_disclosure_joins": int(
            result["future_disclosure_violation"].sum()
        ),
        "future_fiscal_joins": int(
            result["future_fiscal_violation"].sum()
        ),
        "fully_pit_verified": False,
        "final_train_ready": False,
        "warning": (
            "DART 원문 후보 데이터. 최초 제출 및 정정 이력, "
            "당시 재무수치 버전이 전수 검증되지 않았음."
        ),
    }

    write_json(
        summary,
        output / "summary.json",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--panel",
        type=Path,
        default=PROCESSED / "dataset.csv",
    )

    parser.add_argument(
        "--events",
        type=Path,
        default=(
            PROCESSED
            / "disclosures/reconciliation"
            / "dataguide_dart_event_reconciliation.csv"
        ),
    )

    parser.add_argument(
        "--history",
        type=Path,
        default=(
            PROCESSED
            / "disclosures/dart_disclosure_history.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED / "debt_disclosure",
    )

    parser.add_argument(
        "--start",
        default="2001-01",
    )

    parser.add_argument(
        "--end",
        default="2021-12",
    )

    args = parser.parse_args()

    build_monthly_panel(
        panel_path=args.panel,
        events_path=args.events,
        history_path=args.history,
        output=args.output,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()