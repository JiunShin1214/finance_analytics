
"""
DART 최초 사업보고서와 후속 공시 이력 감사.

- 기존 원본 데이터 수정 없음
- DART API 재호출 없음
- 학습용 부채비율 자동 교체 없음
- 접수번호, 공시일, 정정 이력, 월별 사용 위험 검사
"""

from pathlib import Path
import argparse
import json
import re
import pandas as pd

from src.h2.data_io import ROOT
BASE = ROOT / "data" / "processed" / "final"

TITLE = re.compile(
    r"사업보고서\s*\(\s*(\d{4})\s*\.\s*(\d{1,2})\s*\)"
)
CHANGED = re.compile(r"정정|첨부추가|철회")


def code(s):
    v = s.astype("string").str.strip()
    v = v.str.replace(r"^A(?=\d{6}$)", "", regex=True)
    v = v.str.replace(r"\.0$", "", regex=True).str.zfill(6)

    if not v.str.fullmatch(r"\d{6}", na=False).all():
        raise ValueError("유효하지 않은 종목코드가 있습니다.")

    return v


def find_file(user_arg, choices):
    if user_arg:
        p = Path(user_arg).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    for p in choices:
        if p.is_file():
            return p

    raise FileNotFoundError(
        "입력 파일을 찾지 못했습니다. "
        "--history 또는 --events 경로를 지정하세요.\n"
        + "\n".join(map(str, choices))
    )


def read_history(path):
    h = pd.read_csv(
        path,
        dtype={"stock_code": "string", "rcept_no": "string"},
        low_memory=False,
    )

    for c in (
        "stock_code",
        "rcept_no",
        "report_nm",
        "filing_date",
    ):
        if c not in h.columns:
            raise ValueError(f"공시 목록에 {c} 열이 없습니다: {path}")

    h["stock_code"] = code(h["stock_code"])
    h["rcept_no"] = h["rcept_no"].str.strip()

    if not h["rcept_no"].str.fullmatch(
        r"\d{14}", na=False
    ).all():
        raise ValueError("DART 접수번호 형식 오류")

    h["filing_date"] = pd.to_datetime(
        h["filing_date"], errors="coerce"
    )

    if h["filing_date"].isna().any():
        raise ValueError("공시일을 파싱할 수 없는 행이 있습니다")

    keys = ["stock_code", "rcept_no"]

    if h.duplicated(keys).any():
        conflict = h.groupby(keys)[
            ["report_nm", "filing_date"]
        ].nunique(dropna=False)

        if (conflict > 1).any().any():
            raise ValueError(
                "동일 접수번호의 공시명 또는 공시일이 다릅니다"
            )

        h = h.drop_duplicates(keys).copy()

    parts = h["report_nm"].astype("string").str.extract(
        TITLE.pattern
    )

    h["period_year"] = pd.to_numeric(
        parts[0], errors="coerce"
    )
    h["period_month"] = pd.to_numeric(
        parts[1], errors="coerce"
    )

    h = h.loc[
        h["period_year"].notna()
        & h["period_month"].between(1, 12)
    ].copy()

    h["period_year"] = h["period_year"].astype(int)
    h["period_month"] = h["period_month"].astype(int)

    h["has_change_marker"] = h[
        "report_nm"
    ].str.contains(CHANGED, na=False)

    return h.sort_values(
        [
            "stock_code",
            "period_year",
            "period_month",
            "filing_date",
            "rcept_no",
        ]
    ).reset_index(drop=True)


def read_events(path):
    e = pd.read_csv(
        path,
        dtype={"stock_code": "string", "rcept_no": "string"},
        low_memory=False,
    )

    required = [
        "stock_code",
        "rcept_no",
        "fiscal_year",
        "disclosure_date",
        "debt_ratio_pct",
    ]

    for c in required:
        if c not in e.columns:
            raise ValueError(f"후보 공시 파일에 {c} 열이 없습니다")

    e["stock_code"] = code(e["stock_code"])
    e["rcept_no"] = e["rcept_no"].str.strip()

    if e.duplicated(["stock_code", "rcept_no"]).any():
        raise ValueError("후보 공시에 종목·접수번호 중복")

    e["fiscal_year"] = pd.to_numeric(
        e["fiscal_year"], errors="coerce"
    )
    e["disclosure_date"] = pd.to_datetime(
        e["disclosure_date"], errors="coerce"
    )
    e["debt_ratio_pct"] = pd.to_numeric(
        e["debt_ratio_pct"], errors="coerce"
    )

    return e


def audit(h, e):
    keys = [
        "stock_code",
        "period_year",
        "period_month",
    ]

    # 같은 기업·결산기간의 가장 이른 공시
    first = h.drop_duplicates(keys).copy()

    first = first.rename(
        columns={
            "rcept_no": "first_rcept_no",
            "filing_date": "first_filing_date",
            "report_nm": "first_report_nm",
            "has_change_marker": "first_has_change_marker",
        }
    )

    first = first[
        keys
        + [
            "first_rcept_no",
            "first_filing_date",
            "first_report_nm",
            "first_has_change_marker",
        ]
    ]

    dates = h.groupby(keys, as_index=False).agg(
        history_count=("rcept_no", "size"),
        change_marker_count=("has_change_marker", "sum"),
    )

    h = h.merge(
        first,
        on=keys,
        how="left",
        validate="m:1",
    )

    # 정정 문구가 없더라도 후속 제출이면 검토 대상
    later = h.loc[
        h["rcept_no"].ne(h["first_rcept_no"])
    ].copy()

    later = later.sort_values(
        keys + ["filing_date", "rcept_no"]
    )

    next_filing = later.drop_duplicates(keys).rename(
        columns={
            "filing_date": "next_filing_date",
            "rcept_no": "next_rcept_no",
            "report_nm": "next_report_nm",
            "has_change_marker": "next_has_change_marker",
        }
    )

    next_filing = next_filing[
        keys
        + [
            "next_filing_date",
            "next_rcept_no",
            "next_report_nm",
            "next_has_change_marker",
        ]
    ]

    first = first.merge(
        dates, on=keys, validate="1:1"
    ).merge(
        next_filing,
        on=keys,
        how="left",
        validate="1:1",
    )

    # 동일 날짜에 최초·정정이 존재하면 순서 불확실
    same_date = (
        h.groupby(keys + ["filing_date"])
        .size()
        .rename("filings_same_day")
        .reset_index()
    )

    initial_same_day = same_date.merge(
        first[keys + ["first_filing_date"]],
        left_on=keys + ["filing_date"],
        right_on=keys + ["first_filing_date"],
        how="inner",
    )

    first = first.merge(
        initial_same_day[
            keys + ["filings_same_day"]
        ],
        on=keys,
        validate="1:1",
    )

    first["candidate_original_date_ambiguous"] = (
        first["filings_same_day"].gt(1)
    )

    # 기존 DART 수치 후보와 실제 공시 목록 연결
    meta_cols = [
        "stock_code",
        "rcept_no",
        "period_year",
        "period_month",
        "filing_date",
        "report_nm",
        "has_change_marker",
    ]

    e = e.merge(
        h[meta_cols].rename(
            columns={"report_nm": "hist_report_nm"}
        ),
        on=["stock_code", "rcept_no"],
        how="left",
        validate="1:1",
        indicator="_history_merge",
    )

    e = e.merge(
        first.drop(columns=["first_report_nm"]),
        on=keys,
        how="left",
        validate="m:1",
    )

    e["is_first_receipt"] = e["rcept_no"].eq(
        e["first_rcept_no"]
    )

    e["date_matches_history"] = e[
        "disclosure_date"
    ].eq(e["filing_date"])

    e["title_year_matches"] = e[
        "fiscal_year"
    ].eq(e["period_year"])

    e["title_is_dec"] = e["period_month"].eq(12)

    e["has_positive_ratio"] = e[
        "debt_ratio_pct"
    ].gt(0)

    e["strict_extraction"] = (
        e["source_status"].eq(
            "STRICT_ORIGINAL_CFS_CURRENT_CANDIDATE"
        )
        if "source_status" in e.columns
        else pd.Series(False, index=e.index)
    )

    fiscal = pd.to_datetime(
        e.get(
            "fiscal_date",
            pd.Series(pd.NaT, index=e.index),
        ),
        errors="coerce",
    )

    e["original_fiscal_before_filing"] = (
        fiscal < e["disclosure_date"]
    )

    # 불확실한 후보는 자동 승인하지 않음
    conditions = [
        (
            e["_history_merge"] != "both",
            "NO_METADATA_MATCH",
        ),
        (
            ~e["title_is_dec"].fillna(False),
            "NOT_DECEMBER_ANNUAL",
        ),
        (
            ~e["title_year_matches"].fillna(False),
            "FISCAL_YEAR_MISMATCH",
        ),
        (
            ~e["date_matches_history"].fillna(False),
            "DISCLOSURE_DATE_MISMATCH",
        ),
        (
            ~e["is_first_receipt"].fillna(False),
            "NOT_EARLIEST_FILING",
        ),
        (
            e["first_has_change_marker"].fillna(True),
            "EARLIEST_FILING_IS_CHANGE",
        ),
        (
            e["candidate_original_date_ambiguous"].fillna(True),
            "MULTIPLE_FILINGS_SAME_FIRST_DATE",
        ),
        (
            ~e["has_positive_ratio"].fillna(False),
            "INVALID_RATIO",
        ),
        (
            ~e["strict_extraction"].fillna(False),
            "SOURCE_EXTRACTION_NOT_STRICT",
        ),
        (
            ~e["original_fiscal_before_filing"].fillna(False),
            "FISCAL_DATE_UNVERIFIED",
        ),
    ]

    e["version_audit_status"] = (
        "FIRST_FILING_METADATA_MATCH_CANDIDATE"
    )

    for bad, name in conditions:
        e.loc[
            (
                e["version_audit_status"]
                == "FIRST_FILING_METADATA_MATCH_CANDIDATE"
            )
            & bad,
            "version_audit_status",
        ] = name

    e["initial_candidate"] = e[
        "version_audit_status"
    ].eq("FIRST_FILING_METADATA_MATCH_CANDIDATE")

    # 후속 공시일부터는 최초값을 계속 사용해도
    # 되는지 확인이 필요함
    e["valid_until_exclusive"] = e[
        "next_filing_date"
    ].where(e["initial_candidate"])

    e["later_filing_exists"] = e[
        "next_filing_date"
    ].notna()

    return e, first, later


def monthly_audit(events, input_path, output):
    if not input_path or not input_path.is_file():
        return {"monthly_checked": False}

    d = pd.read_csv(
        input_path,
        dtype={
            "stock_id": "string",
            "source_rcept_no": "string",
        },
        low_memory=False,
    )

    required = [
        "stock_id",
        "source_rcept_no",
        "prediction_date",
    ]

    for c in required:
        if c not in d.columns:
            raise ValueError(
                f"월별 후보 파일에 {c} 열이 없습니다"
            )

    use = d.loc[
        d["source_rcept_no"].notna()
    ].copy()

    use["stock_code"] = code(use["stock_id"])
    use["rcept_no"] = use[
        "source_rcept_no"
    ].str.strip()

    use["prediction_date"] = pd.to_datetime(
        use["prediction_date"], errors="coerce"
    )

    r = use.merge(
        events[
            [
                "stock_code",
                "rcept_no",
                "initial_candidate",
                "next_filing_date",
                "disclosure_date",
                "version_audit_status",
            ]
        ],
        on=["stock_code", "rcept_no"],
        how="left",
        validate="m:1",
    )

    r["after_subsequent_filing"] = (
        r["next_filing_date"].notna()
        & r["prediction_date"].ge(r["next_filing_date"])
    )

    r["missing_or_rejected_original"] = ~r[
        "initial_candidate"
    ].fillna(False)

    r["requires_review"] = (
        r["after_subsequent_filing"]
        | r["missing_or_rejected_original"]
    )

    r.loc[
        r["requires_review"]
    ].to_csv(
        output / "monthly_rows_requiring_review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "monthly_checked": True,
        "monthly_candidate_rows": int(len(r)),
        "monthly_after_subsequent_filing": int(
            r["after_subsequent_filing"].sum()
        ),
        "monthly_unverified_original": int(
            r["missing_or_rejected_original"].sum()
        ),
        "monthly_review_rows": int(
            r["requires_review"].sum()
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--history", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--monthly", type=Path)

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/validation/dart_versions",
    )

    args = parser.parse_args()

    history = find_file(
        args.history,
        [
            BASE
            / "disclosures"
            / "dart_disclosure_history.csv"
        ],
    )

    events = find_file(
        args.events,
        [
            BASE
            / "debt_disclosure"
            / "accepted_filing_events.csv",
            BASE
            / "disclosures/reconciliation"
            / "dataguide_dart_event_reconciliation.csv",
        ],
    )

    monthly_path = args.monthly or (
        BASE
        / "debt_disclosure"
        / "h2_debt_disclosure_asof_candidate.csv"
    )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    hist = read_history(history)

    e, groups, later = audit(
        hist,
        read_events(events),
    )

    groups.to_csv(
        out / "annual_filing_version_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    e.to_csv(
        out / "dart_event_version_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    keys = [
        "stock_code",
        "period_year",
        "period_month",
    ]

    affected = later.merge(
        e.loc[
            e["initial_candidate"],
            keys + ["rcept_no"],
        ].rename(
            columns={
                "rcept_no": "accepted_initial_rcept_no"
            }
        ),
        on=keys,
        how="inner",
    )

    affected.to_csv(
        out / "subsequent_filings_review_queue.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "history_file": str(history),
        "candidate_file": str(events),
        "annual_period_groups": len(groups),
        "candidate_events": len(e),
        "history_first_date": str(
            hist["filing_date"].min().date()
        ),
        "history_last_date": str(
            hist["filing_date"].max().date()
        ),
        "initial_metadata_matched": int(
            e["initial_candidate"].sum()
        ),
        "initial_with_later_filing": int(
            (
                e["initial_candidate"]
                & e["later_filing_exists"]
            ).sum()
        ),
        "followup_filings_to_review": len(affected),
        "status_counts": (
            e["version_audit_status"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "fully_pit_verified": False,
        "warning": (
            "최초 접수 여부와 후속 공시 이력을 확인했지만, "
            "정정 재무수치와 당시 원문 버전의 전수 검증은 "
            "별도로 필요합니다."
        ),
    }

    summary.update(
        monthly_audit(e, monthly_path, out)
    )

    (out / "summary.json").write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print("결과 저장:", out)


if __name__ == "__main__":
    main()
