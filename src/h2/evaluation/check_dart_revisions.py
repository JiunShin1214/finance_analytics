
from pathlib import Path
import json
import pandas as pd

from src.h2.evaluation.validate_disclosures import parse_source

from src.h2.data_io import ROOT

AUDIT = ROOT / "results/validation/dart_versions"
REVIEW = AUDIT / "subsequent_filings_review_queue.csv"

HISTORY = (
    ROOT / "data/processed/final/disclosures"
    / "dart_disclosure_history.csv"
)

INITIAL = (
    ROOT / "data/processed/final/debt_disclosure"
    / "accepted_filing_events.csv"
)

QUEUE = AUDIT / "followup_download_queue.csv"
ORIGINALS = AUDIT / "followup_originals/originals"


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return pd.read_csv(
        path,
        dtype={
            "stock_code": str,
            "corp_code": str,
            "rcept_no": str,
            "first_rcept_no": str,
        },
        low_memory=False,
    )


def main(audit_dir=None, history_path=None, initial_path=None, originals_dir=None):
    AUDIT = Path(audit_dir) if audit_dir is not None else ROOT / "results/validation/dart_versions"
    REVIEW = AUDIT / "subsequent_filings_review_queue.csv"
    HISTORY = Path(history_path) if history_path is not None else ROOT / "data/processed/final/disclosures/dart_disclosure_history.csv"
    INITIAL = Path(initial_path) if initial_path is not None else ROOT / "data/processed/final/debt_disclosure/accepted_filing_events.csv"
    QUEUE = AUDIT / "followup_download_queue.csv"
    ORIGINALS = Path(originals_dir) if originals_dir is not None else AUDIT / "followup_originals/originals"
    AUDIT.mkdir(parents=True, exist_ok=True)

    review = load(REVIEW)
    history = load(HISTORY)
    initial = load(INITIAL)

    required_review = {
        "stock_code", "rcept_no", "first_rcept_no",
        "period_year", "filing_date"
    }

    required_history = {
        "stock_code", "rcept_no", "corp_code"
    }

    if not required_review.issubset(review.columns):
        raise ValueError("후속 공시 검토 목록의 컬럼 누락")

    if not required_history.issubset(history.columns):
        raise ValueError("DART 이력의 컬럼 누락")

    for df in (review, history, initial):
        df["stock_code"] = (
            df["stock_code"].astype("string").str.zfill(6)
        )
        df["rcept_no"] = df["rcept_no"].astype("string")

    review["first_rcept_no"] = (
        review["first_rcept_no"].astype("string")
    )

    meta = history[
        ["stock_code", "rcept_no", "corp_code"]
    ].drop_duplicates()

    if meta.duplicated(["stock_code", "rcept_no"]).any():
        raise ValueError("공시 메타데이터 중복")

    queue = review.merge(
        meta,
        on=["stock_code", "rcept_no"],
        how="left",
        validate="m:1",
        suffixes=("", "_history"),
    )

    if "corp_code_history" in queue.columns:
        if "corp_code" in queue.columns:
            queue["corp_code"] = queue[
                "corp_code_history"
            ].combine_first(queue["corp_code"])
        else:
            queue["corp_code"] = queue["corp_code_history"]

    queue["fiscal_year_label"] = pd.to_numeric(
        queue["period_year"], errors="raise"
    ).astype(int)

    if queue["corp_code"].isna().any():
        raise ValueError("기업고유번호 매핑 누락")

    queue["corp_code"] = (
        queue["corp_code"].astype("string").str.zfill(8)
    )

    queue["filing_date"] = pd.to_datetime(
        queue["filing_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    if not queue["rcept_no"].str.fullmatch(
        r"\d{14}", na=False
    ).all():
        raise ValueError("접수번호 형식 오류")

    if queue.duplicated("rcept_no").any():
        raise ValueError("후속 공시 접수번호 중복")

    download_columns = [
        "stock_code", "corp_code",
        "fiscal_year_label", "filing_date", "rcept_no"
    ]

    queue[download_columns].to_csv(
        QUEUE, index=False, encoding="utf-8-sig"
    )

    original_map = initial.set_index(
        ["stock_code", "rcept_no"]
    )["debt_ratio_pct"]

    results = []

    for _, row in queue.iterrows():
        receipt = str(row["rcept_no"])
        path = ORIGINALS / f"{receipt}.zip"

        item = {
            "stock_code": row["stock_code"],
            "rcept_no": receipt,
            "first_rcept_no": row["first_rcept_no"],
            "filing_date": row["filing_date"],
            "fiscal_year": row["fiscal_year_label"],
            "status": "PENDING_DOWNLOAD",
            "original_ratio": None,
            "followup_ratio": None,
            "difference_pp": None,
        }

        key = (row["stock_code"], row["first_rcept_no"])

        if key in original_map.index:
            first_value = original_map.loc[key]
            item["original_ratio"] = float(first_value)

        if path.is_file():
            try:
                parsed = parse_source(
                    path, int(row["fiscal_year_label"])
                )

                if parsed.get("status") != (
                    "STRICT_ORIGINAL_CFS_CURRENT_CANDIDATE"
                ):
                    item["status"] = (
                        "EXTRACTION_REVIEW:"
                        + str(parsed.get("status"))
                    )
                elif item["original_ratio"] is None:
                    item["status"] = "INITIAL_RATIO_MISSING"
                else:
                    new_ratio = float(
                        parsed["debt_ratio_pct"]
                    )

                    diff = new_ratio - item["original_ratio"]

                    item["followup_ratio"] = new_ratio
                    item["difference_pp"] = diff

                    if abs(diff) <= 0.005:
                        item["status"] = "MATCH_FIRST"
                    else:
                        item["status"] = "CHANGED_FROM_FIRST"

            except Exception as exc:
                item["status"] = (
                    "PARSE_ERROR:" + type(exc).__name__
                )

        results.append(item)

    result = pd.DataFrame(results)

    result.to_csv(
        AUDIT / "followup_financial_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "followup_queue_rows": len(queue),
        "downloaded_files": int(
            (result["status"] != "PENDING_DOWNLOAD").sum()
        ),
        "pending_download": int(
            (result["status"] == "PENDING_DOWNLOAD").sum()
        ),
        "same_as_first": int(
            (result["status"] == "MATCH_FIRST").sum()
        ),
        "changed_from_first": int(
            (result["status"] == "CHANGED_FROM_FIRST").sum()
        ),
        "requires_source_review": int(
            result["status"].str.startswith(
                ("EXTRACTION_REVIEW", "PARSE_ERROR",
                 "INITIAL_RATIO_MISSING")
            ).sum()
        ),
        "fully_pit_verified": False,
        "note": (
            "최초 제출값과 후속 제출값의 비교 결과이며, "
            "정정 원문 추출 및 버전 전환의 최종 인증은 아님."
        ),
    }

    (AUDIT / "followup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
