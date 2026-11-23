"""Build an OpenDART company_map.csv using the historical H2 stock universe.

Run from finance_analytics project root:
  python -m src.h2.preprocessing.build_company_map --check-only
  python -m src.h2.preprocessing.build_company_map --execute
No existing data are overwritten. Requires DART_API_KEY only for --execute.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import pandas as pd

CORP_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
STOCK_COLUMNS = ("stock_code", "stock_id", "ticker", "code")


def normalize_stock(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.startswith("A"):
        s = s[1:]
    if re.fullmatch(r"\d{1,6}(?:\.0)?", s):
        return s.removesuffix(".0").zfill(6)
    return None


def read_universe(path: Path) -> tuple[set[str], str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"H2 Hybrid 입력파일 없음: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    col = next((c for c in STOCK_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(f"종목 컬럼을 찾지 못함. 기대 컬럼={STOCK_COLUMNS}; 실제={list(df.columns)}")
    codes = {code for code in (normalize_stock(v) for v in df[col]) if code}
    if not codes:
        raise ValueError(f"유효 종목코드가 없음: {col}")
    return codes, col, len(df)


def download_corp_records(key: str, timeout: int) -> list[dict[str, str]]:
    query = urlencode({"crtfc_key": key})
    req = Request(f"{CORP_URL}?{query}", headers={"User-Agent": "finance-analytics/1.0"})
    with urlopen(req, timeout=timeout) as res:
        body = res.read()
    if not zipfile.is_zipfile(io.BytesIO(body)):
        # DART responds with XML error details on authentication failures
        try:
            root = ET.fromstring(body)
            raise RuntimeError(f"OpenDART: {root.findtext('status')} - {root.findtext('message')}")
        except ET.ParseError as exc:
            raise RuntimeError("OpenDART did not return a ZIP file") from exc
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xml")]
        if not names:
            raise ValueError("OpenDART ZIP에 XML 파일이 없음")
        with z.open(names[0]) as f:
            root = ET.parse(f).getroot()
    records = []
    for item in root.findall(".//list"):
        stock = normalize_stock(item.findtext("stock_code", ""))
        corp = (item.findtext("corp_code") or "").strip()
        if stock and re.fullmatch(r"\d{8}", corp):
            records.append({"stock_code": stock, "corp_code": corp,
                            "corp_name": (item.findtext("corp_name") or "").strip()})
    if not records:
        raise ValueError("OpenDART XML에서 종목-법인 매핑을 찾지 못함")
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="H2 Hybrid 종목을 OpenDART corp_code와 매핑")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--hybrid", type=Path, default=Path("data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv"))
    p.add_argument("--out", type=Path, default=Path("data/raw/h2/company_map.csv"))
    p.add_argument("--timeout", type=int, default=45)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    source = args.hybrid if args.hybrid.is_absolute() else root / args.hybrid
    dest = args.out if args.out.is_absolute() else root / args.out
    missing_path = dest.with_name("company_map_unmatched.csv")
    ambiguous_path = dest.with_name("company_map_ambiguous.csv")
    audit_path = dest.with_name("company_map_audit.json")
    universe, stock_col, rows = read_universe(source)
    print(f"Hybrid 행={rows:,}, 유효 고유 종목={len(universe):,}, 종목컬럼={stock_col}")
    print(f"출력 예정={dest}")
    if args.check_only:
        print("[점검 완료] 네트워크 접속이나 파일 저장을 수행하지 않았습니다")
        return
    existing = [p for p in (dest, missing_path, ambiguous_path, audit_path) if p.exists()]
    if existing:
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {[str(p) for p in existing]}")
    key = os.environ.get("DART_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DART_API_KEY 환경변수가 없음. 키를 코드·Git·채팅에 기록하지 마세요")
    records = download_corp_records(key, args.timeout)
    by_stock: dict[str, list[dict[str, str]]] = {}
    for rec in records:
        by_stock.setdefault(rec["stock_code"], []).append(rec)
    matched, missing, ambiguous = [], [], []
    for stock in sorted(universe):
        matches = list({r["corp_code"]: r for r in by_stock.get(stock, [])}.values())
        if len(matches) == 1:
            matched.append(matches[0])
        elif len(matches) == 0:
            missing.append({"stock_code": stock, "reason": "no_dart_match"})
        else:
            for m in matches:
                ambiguous.append({**m, "reason": "multiple_corp_codes_for_stock"})
    # Do not silently choose from ambiguous matches.
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_hybrid": str(source), "stock_column": stock_col,
        "hybrid_rows": rows, "universe_stocks": len(universe),
        "matched_stocks": len(matched), "unmatched_stocks": len(missing),
        "ambiguous_stocks": len({r["stock_code"] for r in ambiguous}),
        "coverage": len(matched) / len(universe),
        "warning": "OpenDART's latest corpCode listing is not a historical point-in-time identity map. Review unmatched and ambiguous old/delisted codes.",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matched, columns=["stock_code", "corp_code", "corp_name"]).to_csv(dest, index=False, encoding="utf-8-sig")
    pd.DataFrame(missing, columns=["stock_code", "reason"]).to_csv(missing_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(ambiguous, columns=["stock_code", "corp_code", "corp_name", "reason"]).to_csv(ambiguous_path, index=False, encoding="utf-8-sig")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("[완료] 매핑, 미매칭, 중복, 점검 JSON 저장")


if __name__ == "__main__":
    main()
