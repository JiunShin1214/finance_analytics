"""DART 공시 목록 수집 결과 -> 공통 입력 data/interim/dart_filings.parquet에 새 공시를 덧붙인다(2026-11-26).

2026-11-08에는 일회용 스크립트(outputs/run_id/data_공통입력정리/20261108/prepare_data.py)가 시장 전체 목록
(market_19990101_19991231, market_20000101_20211231)을 이 표로 바꿨다. 새 기간(2022~)을 받으면 같은 변환을 이 명령으로 한다.

흐름
  1. 수집: python -m src.preprocessing.dart_collect list --market-wide --start-date 20220101 --end-date 20251231
     -> data/raw/opendart/market_list/market_20220101_20251231/filings.parquet (원본 응답은 data/raw/opendart/list_only/)
  2. 이 명령: python -m src.preprocessing.dart_filings            (market_list 아래 폴더 전부, 검사·건수만)
             python -m src.preprocessing.dart_filings --execute  (쓰기)
     이미 반영된 폴더는 추가 0건이라 같은 명령을 여러 번 돌려도 결과가 같다.
  3. 이후: equity_dating / equity_asof / features.base가 이 표의 접수일을 쓴다.

변환 규칙(prepare_data.py와 같음): rcept_no->receipt_id, rcept_dt->published_at, report_nm->report_name,
listed_in_public_index=True, revision=dart_extract._revision(보고서명 앞 [..]), disclosure_format=없음(원문 추출 안 함).
기존 표의 행과 순서는 바꾸지 않고 새 공시만 뒤에 붙인다(같은 순위 금액 투표가 행 순서를 쓰기 때문).
이미 있는 접수번호는 회사·접수일이 같아야 하고(다르면 멈춤), 원문 추출로만 알던 공시(listed_in_public_index=False)가
공개 목록에 나오면 True로만 바꾼다.

멈추는 경우: 수집이 끝나지 않음(progress.json status != complete), 수집 기간 밖 접수일, 같은 접수번호의 회사·접수일 충돌,
결과 접수번호 중복. 쓰기 전 기존 표를 data/interim/_backups/dart_filings_<시각>.parquet로 복사하고 같은 이름의 .json에 기록한다.
outputs/는 읽지도 쓰지도 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.contracts import DART_FILING_INPUT_COLUMNS
from src.preprocessing.dart_collect import MARKET_RUN
from src.preprocessing.dart_extract import DART_FILINGS, _revision

BACKUPS = Path("data/interim/_backups")
RENAME = {"rcept_no": "receipt_id", "rcept_dt": "published_at", "report_nm": "report_name"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_market_run(run: Path) -> tuple[pd.DataFrame, dict]:
    """시장 전체 목록 수집 폴더 하나 -> 공통 입력 열. 수집이 끝난 폴더만 받는다."""
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "complete":
        raise ValueError(f"dart_list_run_not_complete:{run.as_posix()}:{progress.get('status')}:{progress.get('stop_reason')}")
    criteria = progress.get("criteria") or {}
    start, end = criteria.get("receipt_start"), criteria.get("receipt_end")
    if not (start and end):
        name = run.name.split("_")          # market_<start>_<end>
        start, end = name[-2], name[-1]
    raw = pd.read_parquet(run / "filings.parquet")
    if raw.empty:
        raise ValueError(f"dart_list_run_empty:{run.as_posix()}")
    if not raw.rcept_dt.astype(str).between(start, end).all():
        raise ValueError(f"dart_list_receipt_outside_window:{run.as_posix()}")
    frame = raw.rename(columns=RENAME).copy()
    frame["listed_in_public_index"] = True
    frame["revision"] = frame.report_name.map(_revision)
    frame["disclosure_format"] = pd.NA
    frame = frame.reindex(columns=list(DART_FILING_INPUT_COLUMNS))
    if frame.receipt_id.duplicated().any():         # 같은 접수번호가 두 창에 걸치면 첫 행만(수집기는 (회사, 접수번호)로 모음)
        frame = frame.drop_duplicates("receipt_id", keep="first")
    info = {"run": run.as_posix(), "window": [start, end], "rows": int(len(raw)), "receipts": int(len(frame)),
            "filings_sha256": _sha(run / "filings.parquet"), "completed_windows": len(progress.get("completed_windows", []))}
    return frame, info


def merge(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """기존 표 뒤에 새 공시를 붙인다. 기존 행·순서는 그대로(공개 표시만 False->True 가능)."""
    out = existing.copy()
    idx = pd.Index(out.receipt_id)
    if not idx.is_unique:
        raise ValueError("existing_dart_filings_duplicate_receipts")
    known = new.receipt_id.isin(idx)
    both = new[known].set_index("receipt_id")
    old = out.set_index("receipt_id").loc[both.index]
    for col in ("corp_code", "published_at"):
        if not (old[col].astype(str).to_numpy() == both[col].astype(str).to_numpy()).all():
            raise ValueError(f"dart_filing_identity_conflict:{col}")
    promote = old.index[~old.listed_in_public_index.astype(bool).to_numpy()]
    # 원문 추출로만 알던 공시가 공개 목록에 나오면 공개 목록의 값(회사명·종목코드·보고서 이름·종류·기간·출처)으로 채운다.
    # 11-08 prepare_data와 같음(공개 목록 행이 먼저, 추출 정보는 disclosure_format만). 접수번호·회사·접수일은 위에서 같음을 확인.
    fill = ["corp_name", "stock_code", "report_name", "report_kind", "report_period_label", "revision", "source_path", "source_sha256"]
    rows = out.receipt_id.isin(promote)
    src = new.set_index("receipt_id").loc[out.loc[rows, "receipt_id"], fill]
    out.loc[rows, fill] = src.to_numpy()
    out.loc[rows, "listed_in_public_index"] = True
    added = new[~known]
    out = pd.concat([out, added[list(DART_FILING_INPUT_COLUMNS)]], ignore_index=True)
    for col in DART_FILING_INPUT_COLUMNS:
        if col != "listed_in_public_index":
            out[col] = out[col].astype("string")
    out["listed_in_public_index"] = out.listed_in_public_index.astype(bool)
    if out.receipt_id.duplicated().any():
        raise ValueError("merged_dart_filings_duplicate_receipts")
    summary = {"existing_rows": int(len(existing)), "new_receipts": int(len(new)), "already_present": int(known.sum()),
               "promoted_to_public": int(len(promote)), "added": int(len(added)), "result_rows": int(len(out)),
               "added_published_range": [str(added.published_at.min()), str(added.published_at.max())] if len(added) else None,
               "added_report_kinds": added.report_kind.fillna("none").value_counts().to_dict() if len(added) else {}}
    return out, summary


def market_runs(root: Path) -> list[Path]:
    """data/raw/opendart/market_list/market_<시작>_<끝> 폴더 전부(이름 순)."""
    runs = sorted(p.relative_to(root) for p in (root / MARKET_RUN).glob("market_*") if p.is_dir())
    if not runs:
        raise ValueError(f"no_market_list_runs:{MARKET_RUN.as_posix()}")
    return runs


def run(root: Path, runs: list[Path] | None = None, *, execute: bool = False) -> dict:
    runs = list(runs) if runs else market_runs(root)
    target = root / DART_FILINGS
    existing = pd.read_parquet(target)
    parts, infos = [], []
    for r in runs:
        frame, info = read_market_run(root / r)
        parts.append(frame)
        infos.append(info)
    new = pd.concat(parts, ignore_index=True).drop_duplicates("receipt_id", keep="first")
    merged, summary = merge(existing, new)
    result = {"inputs": infos, "summary": summary, "execute": execute, "target": DART_FILINGS.as_posix(),
              "target_sha256_before": _sha(target)}
    if not execute:
        return result
    if summary["added"] == 0 and summary["promoted_to_public"] == 0:
        result["unchanged"] = True                  # 이미 반영됨: 쓰지 않는다
        return result
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = root / BACKUPS / f"dart_filings_{stamp}.parquet"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target, backup)
    tmp = target.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    pd.testing.assert_frame_equal(merged, pd.read_parquet(tmp), check_exact=True)
    tmp.replace(target)
    result.update(target_sha256_after=_sha(target), backup=(BACKUPS / backup.name).as_posix(), command=" ".join(sys.argv),
                  finished=datetime.now().isoformat(timespec="seconds"))
    backup.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, action="append",
                        help="수집 폴더 하나만 쓸 때(기본: data/raw/opendart/market_list/market_* 전부). 여러 번 줄 수 있음")
    parser.add_argument("--execute", action="store_true", help="없으면 검사와 건수만 출력하고 쓰지 않음")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.run, execute=args.execute), ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
