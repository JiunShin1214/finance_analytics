"""Stage 2, DART value extraction: original ZIP -> equity facts and one decision per filing (extractor v11),
plus the DART-only as-of rules (overrides, unit corrections, join) reused by the B/M as-of.
Merged 2026-11-04 from dart_asof.py and dart_equity.py; CLI = the former dart_equity CLI.
"""
from __future__ import annotations

from decimal import Decimal
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Mapping
from src.contracts import Config
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
import hashlib
import io
import os
import re
import shutil
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from collections import Counter, defaultdict
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import uuid
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import warnings


if TYPE_CHECKING:
    from pandas import DataFrame


def plan_dart_collection(requests_table: DataFrame, *, config: Config) -> DataFrame:
    """기업번호·필요 기준월 요청표를 받아 공시 수집 목록과 출처·상태를 구성한다.

    DART-003의 승인된 2015~2021 사업연도·네 보고서·CFS/OFS 요청 단위를 계획한다.
    실제 네트워크/원본 보존은 DartCollector가 수행한다. 월별 자기자본 선택은 하지 않는다.
    """
    import pandas as pd
    corps = requests_table["corp_code"].dropna().drop_duplicates().tolist()
    if any(not isinstance(c, str) or len(c) != 8 or not c.isdigit() for c in corps):
        raise ValueError("corp_code must be an eight-digit string")
    return pd.DataFrame([
        {"corp_code": c, "bsns_year": str(y), "reprt_code": r, "fs_div": fs}
        for c in corps for y in range(2015, 2022)
        for r in ("11011", "11012", "11013", "11014") for fs in ("CFS", "OFS")
    ])


MARCAP_REL = Path("data/raw/marcap")


EQUITY_VALIDITY_MONTHS = 18  # P3: statement date + 18 months, through that month's end




def available_month(day: str, last_trading_day: Mapping[str, str]) -> str:
    """P4: a receipt on or after the month's last trading day counts from the next month."""
    month = f"{day[:4]}-{day[4:6]}"
    if month not in last_trading_day:
        if month > max(last_trading_day):
            return month  # beyond the calendar, hence after every base month it serves; never used
        raise ValueError("trading_calendar_missing_month:" + month)
    if day < last_trading_day[month]:
        return month
    return f"{int(day[:4]) + (day[4:6] == '12'):04d}-{int(day[4:6]) % 12 + 1:02d}"


def _expiry(period_end: str) -> str:
    import pandas as pd
    end = pd.Timestamp(period_end) + pd.DateOffset(months=EQUITY_VALIDITY_MONTHS) + pd.offsets.MonthEnd(0)
    return end.strftime("%Y%m%d")


def _given(value) -> bool:
    import pandas as pd
    return value is not None and not pd.isna(value) and str(value) != ""


def _versions(filings: DataFrame, last_trading_day: Mapping[str, str]) -> dict[str, list[dict]]:
    """Per corporation, every filing version with its P4 month and why its value can or cannot be used."""
    versions = {}
    for row in filings.to_dict("records"):
        fiscal = row.get("fiscal_period")
        fiscal = None if fiscal is None or fiscal != fiscal else str(fiscal)[:10]
        equity = row.get("equity")
        equity = None if equity is None or equity != equity else str(equity)
        if equity is None:
            skip = "no_value:" + str(row.get("reason"))
        elif fiscal is None:
            skip = "fiscal_date_unverified"
        elif str(row.get("unit_suspect")) == "True":
            skip = "unit_suspect"  # P7: held for review; never corrected here
        elif str(row.get("owners_identity_ok")) == "False" and not _given(row.get("owners_resolution")):
            skip = "owners_identity_failed"  # owners + NCI != total in its own table: held for review like P7
        else:
            skip = None
        versions.setdefault(row["corp_code"], []).append({
            **row, "equity": equity, "fiscal_period_end": fiscal, "skip": skip,
            "availability_month": available_month(row["published_at"], last_trading_day),
            "rank": 0 if row["revision"] == "최초표시" else 1})
    for items in versions.values():
        # Newest period first; inside a period the latest day, then corrections after their original.
        items.sort(key=lambda v: (v["report_period_label"], v["published_at"], v["rank"], v["receipt_id"]), reverse=True)
    return versions


def _select(available: list[dict]) -> dict:
    """P1/P2/P5/P10 for one corporation and month. `available` is already newest-first."""
    skipped = []
    labels = list(dict.fromkeys(v["report_period_label"] for v in available))
    for p_index, label in enumerate(labels):
        groups = {}
        for v in available:
            if v["report_period_label"] == label:
                groups.setdefault((v["published_at"], v["rank"]), []).append(v)
        for g_index, group in enumerate(groups.values()):
            usable = [v for v in group if v["skip"] is None]
            if not usable:
                skipped += [(v["receipt_id"], v["skip"]) for v in group]
                continue
            # Same day, same lineage step: the receipt number does not prove which came first.
            unverified = len(group) > 1
            if unverified and len({v["equity"] for v in usable}) > 1:
                return {"asof_status": "same_day_value_conflict", "skipped": skipped + [(v["receipt_id"], "same_day_value_conflict") for v in group],
                        "same_day_order_unverified": True}
            chosen = max(usable, key=lambda v: v["receipt_id"])
            fallback = "none" if p_index == 0 and g_index == 0 else "older_version_same_period" if p_index == 0 else "older_period"
            return {"asof_status": "selected", "chosen": chosen, "fallback": fallback, "skipped": skipped,
                    "same_day_order_unverified": unverified}
    return {"asof_status": "no_usable_value", "skipped": skipped, "same_day_order_unverified": False}


def join_equity_asof(mapped_panel: DataFrame, filings: DataFrame, *, config: Config | None = None,
                     last_trading_day: Mapping[str, str], history_complete_from: str | None = None) -> DataFrame:
    """기업 연결 패널(stock_id, base_month, corp_code, mapping_status)과 공시 버전표 -> ASOF_COLUMNS 표.

    P4: 기준월 마지막 KRX 거래일~말일 접수는 다음 기준월부터. P1: 최신 버전 값이 없으면 같은 기간
    이전 버전. P2: 기간 전체에 값이 없으면 이전 기간. P5: 단계는 공시별 선택을 그대로 쓰며 최신 기간이
    우선. P10: 원본→정정 계보는 인정하고, 같은 날 같은 단계 버전의 금액이 다르면 보류(이월로 숨기지 않음).
    P3: 재무상태표 기준일 + 18개월의 월말이 기준월 마지막 거래일보다 앞서면 B/M 불가(값은 보존).
    기준일 미확인·단위 의심 값은 쓰지 않고 이월 대상으로만 둔다. 정정은 공개 이후만 적용한다.
    자기자본 원천은 정기보고서뿐이다. 첫 정기보고서 공개 전(상장 직후 등)은 before_first_regular_report,
    수집 범위에 그 회사의 정기보고서가 전혀 없으면 no_regular_report_in_window로 남긴다.
    """
    import pandas as pd
    from src.contracts import ASOF_COLUMNS
    if mapped_panel.duplicated(["stock_id", "base_month"]).any():
        raise ValueError("duplicate_stock_month")
    versions = _versions(filings, last_trading_day)
    rows, previous_scope = [], {}
    for item in mapped_panel.sort_values(["stock_id", "base_month"]).to_dict("records"):
        month, corp = item["base_month"], item.get("corp_code")
        row = {k: None for k in ASOF_COLUMNS}
        row.update(stock_id=item["stock_id"], base_month=month, corp_code=corp, mapping_status=item["mapping_status"],
                   mapping_basis=item.get("mapping_basis"), same_day_order_unverified=False, scope_changed=False,
                   history_incomplete=bool(history_complete_from and month < history_complete_from))
        if item["mapping_status"] != "current_exact" or corp is None or corp != corp:
            row["asof_status"] = "unmapped"
        else:
            available = [v for v in versions.get(corp, []) if v["availability_month"] <= month]
            row["candidates_n"] = len(available)
            # Regular reports only: before the first one (e.g. right after listing) there is no value to use.
            empty = "before_first_regular_report" if versions.get(corp) else "no_regular_report_in_window"
            result = _select(available) if available else {"asof_status": empty, "skipped": [], "same_day_order_unverified": False}
            row["asof_status"] = result["asof_status"]
            row["same_day_order_unverified"] = result["same_day_order_unverified"]
            row["skipped_receipt_ids"] = ";".join(r for r, _ in result["skipped"]) or None
            row["skipped_reasons"] = ";".join(s for _, s in result["skipped"]) or None
            chosen = result.get("chosen")
            if chosen:
                row.update({k: chosen.get(k) for k in ("equity", "equity_tier", "statement_type", "account_name", "source_kind",
                                                       "section", "column_basis", "receipt_id", "published_at", "availability_month",
                                                       "report_kind", "report_period_label", "fiscal_period_end",
                                                       "fiscal_period_basis", "revision",
                                                       "report_name", "raw_path", "sha256", "identity_ok", "xbrl_document_agree",
                                                       "owners_resolution", "override_id")})
                row["fallback"] = result["fallback"]
                end = pd.Timestamp(chosen["fiscal_period_end"])
                row["equity_age_months"] = (int(month[:4]) - end.year) * 12 + int(month[5:]) - end.month
                row["expiry_date"] = _expiry(chosen["fiscal_period_end"])
                scope = chosen["statement_type"]
                row["scope_changed"] = item["stock_id"] in previous_scope and previous_scope[item["stock_id"]] != scope
                previous_scope[item["stock_id"]] = scope
        expired = row["expiry_date"] is not None and last_trading_day[month] > row["expiry_date"]
        positive = row["equity"] is not None and pd.to_numeric(row["equity"]) > 0
        reasons = [r for r, hit in (("unmapped", row["asof_status"] == "unmapped"),
                                    ("no_regular_report_in_window", row["asof_status"] == "no_regular_report_in_window"),
                                    ("before_first_regular_report", row["asof_status"] == "before_first_regular_report"),
                                    ("no_usable_value", row["asof_status"] == "no_usable_value"),
                                    ("same_day_value_conflict", row["asof_status"] == "same_day_value_conflict"),
                                    ("history_incomplete", row["history_incomplete"]),
                                    ("expired_18m", expired),
                                    ("nonpositive_equity", row["asof_status"] == "selected" and not positive)) if hit]
        row["usable"] = not reasons
        row["exclusion_reason"] = ";".join(reasons) or None
        rows.append(row)
    table = pd.DataFrame(rows, columns=ASOF_COLUMNS)
    if any(Decimal(s) % 1 for s in table.equity.dropna()):
        raise ValueError("fractional_krw_equity")
    table["equity"] = pd.array([pd.NA if pd.isna(s) else int(Decimal(s)) for s in table.equity], dtype="Int64")
    for column in ("candidates_n", "equity_age_months"):
        table[column] = table[column].astype("Int64")
    for column in ("same_day_order_unverified", "scope_changed", "history_incomplete", "usable"):
        table[column] = table[column].astype("boolean")
    return table


OVERRIDES = Path("config/equity_overrides.json")


OVERRIDE_FIELDS = {"equity", "equity_tier", "statement_type", "account_name", "section", "fiscal_period",
                   "fiscal_period_basis", "owners_resolution"}


def apply_overrides(root: Path, filings: DataFrame, path: Path = OVERRIDES) -> tuple[DataFrame, dict]:
    """Approved manual decisions onto filing-version rows, before the as-of join.

    Each entry names the extracted values it replaces; if the extractor now says something else, or the
    evidence file's hash changed, the run stops so the decision is reviewed again rather than misapplied.
    Applied after extraction, so editing it does not change any extracted decision.
    """
    import pandas as pd
    from collections import Counter
    entries = json.loads((root / path).read_text(encoding="utf-8")) if (root / path).exists() else []
    if len({e["id"] for e in entries}) != len(entries) or len({e["receipt_id"] for e in entries}) != len(entries):
        raise ValueError("override_duplicate_id_or_receipt")
    filings = filings.assign(override_id=pd.Series(pd.NA, index=filings.index, dtype="string"))
    if "owners_resolution" not in filings:
        filings["owners_resolution"] = pd.Series(pd.NA, index=filings.index, dtype="string")
    counts = Counter(e["status"] for e in entries)
    for entry in entries:
        if entry["status"] != "approved":
            continue
        if set(entry["set"]) - OVERRIDE_FIELDS:
            raise ValueError(f"override_field_not_allowed:{entry['id']}")
        hit = filings.index[filings.receipt_id == entry["receipt_id"]]
        if len(hit) != 1:
            raise ValueError(f"override_receipt_not_found:{entry['id']}")
        row = hit[0]
        for field, expected in entry["expected_before"].items():
            current = filings.at[row, field]
            if (None if pd.isna(current) else str(current)) != expected:
                raise ValueError(f"override_stale:{entry['id']}:{field}")
        evidence = entry["evidence"]
        if hashlib.sha256((root / evidence["raw_path"]).read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError(f"override_evidence_hash_mismatch:{entry['id']}")
        for field, value in entry["set"].items():
            filings.at[row, field] = value
        filings.at[row, "override_id"] = entry["id"]
        counts["applied"] += 1
    return filings, dict(counts)


UNIT_SHIFTS = (3, 6)  # 천원 and 백만원 written with the wrong unit label


def apply_unit_corrections(filings: DataFrame, facts: DataFrame) -> tuple[DataFrame, list[dict]]:
    """P7 automatic case: a unit-suspect value replaced only by evidence inside the same filing.

    A filer's unit slip moves the body table and the XBRL together, but the attached audit report often
    keeps the right unit. Replace only when that filing has a balanced full statement of the same concept
    and scope whose amount is exactly 10^3 or 10^6 times off, every such amount agrees, its statement date
    does not contradict, and the firm's history no longer flags the replaced value. Anything else stays held.
    Manual overrides win: a row they already set is left alone. Applied after extraction, so it does not change
    any extracted decision.
    """
    import pandas as pd
    from src.preprocessing.dart_extract import unit_suspects
    filings = filings.copy()
    if "override_id" not in filings:
        filings["override_id"] = pd.Series(pd.NA, index=filings.index, dtype="string")
    concept_of = {"T1_consolidated_owners": "owners", "T2_consolidated_total": "total", "T3_separate_total": "total"}
    applied = []
    suspects = filings[(filings.unit_suspect == "True") & filings.equity.notna() & filings.override_id.isna()]
    for row, item in suspects.iterrows():
        concept = concept_of.get(item.equity_tier)
        if concept is None:
            continue
        value = Decimal(item.equity)
        pool = facts[(facts.receipt_id == item.receipt_id) & (facts.concept == concept)
                     & (facts.statement_type == item.statement_type) & (facts.section == "full")
                     & (facts.source_kind == "document") & (facts.identity_ok.astype(str) == "True")]
        found = {}
        for fact in pool.itertuples():
            amount = Decimal(fact.equity)
            shift = next((k for k in UNIT_SHIFTS for s in (k, -k) if amount * Decimal(10) ** s == value), None)
            if shift is not None and amount != value:
                found.setdefault(amount, []).append(fact)
        if len(found) != 1:
            continue
        amount, evidence = next(iter(found.items()))
        if any(_given(f.fiscal_period) and _given(item.fiscal_period) and str(f.fiscal_period)[:10] != str(item.fiscal_period)[:10]
               for f in evidence):
            continue
        trial = filings[filings.corp_code == item.corp_code].copy()
        trial.at[row, "equity"] = format(amount, "f")
        if unit_suspects(trial).get(row, True):
            continue
        fact = evidence[0]
        shift = round(math.log10(abs(value / amount)))
        filings.loc[row, ["equity", "unit_suspect", "source_kind", "section", "column_basis", "account_name", "raw_path", "sha256"]] = [
            format(amount, "f"), "False", "document", "full", fact.column_basis, fact.account_name, fact.raw_path, fact.sha256]
        filings.at[row, "override_id"] = f"AUTO_UNIT:x10^{-shift}:{fact.member}:{fact.location}"
        applied.append({"receipt_id": item.receipt_id, "before": str(item.equity), "after": format(amount, "f"),
                        "evidence": f"{fact.member}:{fact.location}", "unit": fact.column_basis.rsplit("|", 1)[-1]})
    return filings, applied






KST = timezone(timedelta(hours=9))


RUN_REL = Path("outputs/collection/DART003")


RAW_REL = Path("data/raw/opendart")


PROCESSED_REL = Path("data/processed/opendart")


CONCEPTS = {"EquityAttributableToOwnersOfParent", "Equity"}


NS = {"x": "http://www.xbrl.org/2003/instance", "d": "http://xbrl.org/2006/xbrldi"}


FACT_COLUMNS = (
    "fact_id", "corp_code", "receipt_id", "bsns_year", "reprt_code",
    "fiscal_period", "statement_type", "account", "account_name", "equity",
    "amount_raw", "unit", "decimals", "context_ref", "dimensions", "source_kind",
    "raw_path", "sha256", "member", "location", "extraction_status",
    "crosscheck_status", "published_at", "availability_month", "revision",
)


VERSION_COLUMNS = (
    "corp_code", "receipt_id", "published_at", "availability_month", "report_name",
    "report_kind", "fiscal_period_label", "bsns_year", "reprt_code", "filing_group",
    "revision", "candidate_initial_receipt", "initial_receipt", "previous_receipt",
    "version_relation", "index_path", "index_sha256", "xbrl_path", "document_path",
    "original_status", "extraction_status", "crosscheck_status", "reason",
    "document_parse_status", "document_parse_issues",
)


MAPPING_COLUMNS = (
    "stock_id", "corp_code", "corp_name", "registry_modify_date", "mapping_status",
    "historical_identity_status", "valid_from", "valid_to", "evidence_path", "sha256",
)


def _verified_predecessors(versions: pd.DataFrame) -> dict:
    """A receipt number is not a verified ordering rule for same-day corrections."""
    result = {}
    for _, group in versions[versions.initial_receipt.notna()].groupby("filing_group"):
        for idx, row in group.iterrows():
            same_day = group[group.published_at == row.published_at]
            earlier = group[group.published_at < row.published_at]
            if len(same_day) > 1:
                result[idx] = (None, "same_day_version_order_unverified")
            elif len(earlier):
                previous = earlier[earlier.published_at == earlier.published_at.max()]
                result[idx] = ((previous.iloc[0].receipt_id, None) if len(previous) == 1 else
                               (None, "preceding_day_order_unverified"))
            else:
                result[idx] = (None, None)
    return result


class CollectionStop(RuntimeError):
    """A recorded technical condition; never contains a request URL or a key."""


def _now() -> str:
    return datetime.now(KST).isoformat()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _request_key(endpoint: str, params: dict) -> str:
    return json.dumps([endpoint, params], sort_keys=True, separators=(",", ":"))


def _month_available(day: str | None) -> str | None:
    if not day:
        return None
    d = datetime.strptime(day, "%Y%m%d").date()
    if d.day == calendar.monthrange(d.year, d.month)[1]:
        return f"{d.year + int(d.month == 12):04d}-{d.month % 12 + 1:02d}"
    return f"{d.year:04d}-{d.month:02d}"


def _amount(text: str | None, scale: int = 1) -> str | None:
    if text is None or text.strip() in ("", "-", "NA"):
        return None
    s = text.strip().replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    if s.startswith(("△", "▲")):
        s = "-" + s[1:]
    try:
        d = Decimal(s) * scale
    except InvalidOperation:
        return None
    return format(d, "f") if d.is_finite() else None


def _report_group(name: str) -> tuple[str, str, str] | None:
    match = re.search(r"(사업|반기|분기)보고서\s*\((\d{4})\.(\d{2})\)", name)
    if not match:
        return None
    kind, y, m = match.groups()
    if not 1 <= int(m) <= 12:
        return None
    return kind, f"{y}-{m}", f"{kind}:{y}-{m}"


def _decode_document(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(encoding, errors="strict"), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("document_encoding_unresolved")


def _header_grid(table) -> list[list]:
    """Expand header spans into physical columns; retain the original cell objects."""
    head = table.find("thead", recursive=False)
    rows = head.find_all("tr") if head else []
    occupied, grid = {}, []
    for ri, row in enumerate(rows):
        col = 0
        for cell in row.find_all(["th", "td"], recursive=False):
            while (ri, col) in occupied:
                col += 1
            width, height = int(cell.get("colspan", 1)), int(cell.get("rowspan", 1))
            if not 1 <= width <= 100 or not 1 <= height <= 100:
                raise ValueError("header_span_out_of_bounds")
            for dr in range(height):
                for dc in range(width):
                    occupied[ri + dr, col + dc] = cell
            col += width
        last = max((c for rr, c in occupied if rr == ri), default=-1)
        grid.append([occupied.get((ri, c)) for c in range(last + 1)])
    return grid


def _current_columns(table, context: str, period: str) -> tuple[list[int], str | None]:
    """Use dated/term headers, never the first nonempty value across fiscal years."""
    dates = re.findall(r"(\d{4})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})", context)
    ends = {f"{int(y):04d}-{int(m):02d}-{int(d):02d}" for y, m, d in dates
            if f"{int(y):04d}-{int(m):02d}" == period}
    if len(ends) != 1:
        return [], None
    end = next(iter(ends))
    # A term label has to be linked to a date in the nearby statement heading.
    term_dates = {}
    for term, y, mo, day in re.findall(
            r"제\s*(\d+)\s*(?:\([^)]*\))?\s*기.{0,60}?"
            r"(\d{4})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})", context):
        term_dates[term] = f"{int(y):04d}-{int(mo):02d}-{int(day):02d}"
    blocks = []
    for row in _header_grid(table):
        for col, cell in enumerate(row):
            if cell is None:
                continue
            text = re.sub(r"\s+", "", cell.get_text(" ", strip=True))
            hd = re.findall(r"(\d{4})[./년-](\d{1,2})[./월-](\d{1,2})", text)
            term = re.search(r"제(\d+)(?:\([^)]*\))?기", text)
            matches = (any(f"{int(y):04d}-{int(m):02d}-{int(d):02d}" == end for y, m, d in hd)
                       or bool(term and term_dates.get(term[1]) == end)
                       or ("당기" in text and "전" not in text and bool(dates)
                           and f"{int(dates[0][0]):04d}-{int(dates[0][1]):02d}-{int(dates[0][2]):02d}" == end))
            if matches and not any(x[0] is cell for x in blocks):
                blocks.append((cell, [i for i, c in enumerate(row) if c is cell]))
    if not blocks:
        return [], end
    # Multiple header levels may describe the same block; inconsistent blocks are ambiguous.
    common = set(blocks[0][1])
    for _, columns in blocks[1:]:
        common &= set(columns)
    return sorted(common), end


def _document_evidence(body: bytes, corp: str, period: str) -> dict:
    """Decode without editing raw bytes; do not confuse errata tables with the BS."""
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    import warnings
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = [n for n in archive.namelist()
                 if n.endswith(".xml") and not re.search(r"_\d{5}\.xml$", n)]
        if len(names) != 1:
            raise ValueError("document_main_member_ambiguous")
        raw = archive.read(names[0])
    text, encoding = _decode_document(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(text, "html.parser")
    company = soup.find("company-name")
    if company is None or company.get("aregcik") != corp:
        raise ValueError("document_corporation_unverified")
    flat = soup.get_text(" ", strip=True)
    original_day = None
    match = re.search(r"최초\s*제출일\s*[:：]?\s*(\d{4})\s*[년./-]\s*(\d{1,2})\s*[월./-]\s*(\d{1,2})", flat)
    if match:
        original_day = "".join((match[1], match[2].zfill(2), match[3].zfill(2)))
    candidates, parse_issues = [], []
    account_names = {
        "지배기업소유주지분": "owners",
        "지배기업의소유주에게귀속되는자본": "owners",
        "지배기업소유주에게귀속되는자본": "owners",
        "자본총계": "total",
    }
    for index, table in enumerate(soup.find_all("table")):
        nearby = []
        for node in table.previous_siblings:
            if not getattr(node, "get_text", None):
                continue
            text = node.get_text(" ", strip=True)
            if not text:
                continue
            nearby.append(text)
            if "재무상태표" in re.sub(r"\s+", "", text) or len(nearby) >= 6:
                break
        context = " ".join(reversed(nearby))[-3000:]
        normalized = re.sub(r"\s+", "", context)
        if "재무상태표" not in normalized:
            continue
        columns, end = _current_columns(table, context, period)
        unit_text = normalized + re.sub(r"\s+", "", table.find("thead").get_text() if table.find("thead") else "")
        if "백만원" in unit_text:
            scale, unit = 1000000, "KRW"
        elif "천원" in unit_text:
            scale, unit = 1000, "KRW"
        elif re.search(r"단위[:：]원", unit_text):
            scale, unit = 1, "KRW"
        else:
            continue
        fs = "CFS" if "연결재무상태표" in normalized else "OFS"
        for row_index, row in enumerate(table.find_all("tr")):
            cells = row.find_all(["td", "th"], recursive=False)
            if len(cells) < 2:
                continue
            label = cells[0].get_text(" ", strip=True)
            typ = account_names.get(re.sub(r"[\s\[\]]", "", label))
            if not typ or (typ == "owners" and fs != "CFS"):
                continue
            if not columns or end is None:
                parse_issues.append(f"TABLE[{index}]/TR[{row_index}]:current_period_column_unresolved")
                continue
            expanded = []
            for ci, cell in enumerate(cells):
                expanded.extend([(ci, cell)] * int(cell.get("colspan", 1)))
            # A monetary subtotal can occupy either subcolumn of a dated colspan=2.
            # Only inspect that header block; a blank current block never falls into prior year.
            selected = {ci: cell for c in columns if c < len(expanded) for ci, cell in [expanded[c]]}
            nonempty = [(ci, cell.get_text(" ", strip=True)) for ci, cell in selected.items()
                        if cell.get_text(" ", strip=True) not in ("", "\\", "₩", "KRW")]
            if len(nonempty) != 1:
                parse_issues.append(f"TABLE[{index}]/TR[{row_index}]:blank_or_ambiguous_current_amount")
                continue
            ci, original = nonempty[0]
            value = _amount(original, scale)
            if value is not None:
                candidates.append({"statement_type": fs, "account": None,
                    "account_name": label, "equity": value, "amount_raw": original,
                    "unit": unit, "fiscal_period": end, "decimals": str(-(len(str(scale)) - 1)),
                    "context_ref": None, "dimensions": None, "member": names[0],
                    "location": f"TABLE[{index}]/TR[{row_index}]/TD[{ci}]; {context}",
                    "extraction_status": "label_only_concept_unverified",
                    "source_kind": "document"})
    cover = flat.replace("&cr;", " ").replace("&cr", " ")
    quarter = re.search(r"분\s*기\s*보\s*고\s*서\s*\(\s*제\s*\d+\s*기\s*([13])\s*분기", cover)
    report_code = {"1": "11013", "3": "11014"}.get(quarter[1]) if quarter else None
    cover_period = re.search(r"사업연도\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*부터\s*"
                             r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*까지", cover)
    period_evidence = None
    if cover_period:
        sy, sm, sd, ey, em, ed = map(int, cover_period.groups())
        period_evidence = f"{sy:04d}-{sm:02d}-{sd:02d}/{ey:04d}-{em:02d}-{ed:02d}"
        months = (ey - sy) * 12 + em - sm + 1
        if (report_code is None and "분기보고서" in flat[:100]
                and sd == 1 and ed == calendar.monthrange(ey, em)[1]
                and f"{ey:04d}-{em:02d}" == period):
            report_code = {3: "11013", 9: "11014"}.get(months)
    return {"encoding": encoding, "initial_date": original_day,
            "reprt_code_from_cover": report_code,
            "cover_period_evidence": period_evidence,
            "facts": candidates, "member": names[0], "parse_issues": parse_issues}


def _xbrl_facts(body: bytes, corp: str, period: str) -> tuple[list[dict], dict]:
    """Select by concept, entity, actual instant, unit and entire dimensional scope."""
    facts, audit = [], {"parsed_members": 0, "candidate_concepts": 0,
                        "rejected_entity": 0, "rejected_dimensions": 0,
                        "rejected_unit": 0, "missing_concepts": []}
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".xbrl")]
        if not members:
            raise ValueError("xbrl_instance_missing")
        for member in members:
            raw = archive.read(member)
            tree = ET.fromstring(raw)
            audit["parsed_members"] += 1
            contexts = {x.get("id"): x for x in tree.findall("x:context", NS)}
            units = {x.get("id"): x for x in tree.findall("x:unit", NS)}
            for node in tree:
                local = node.tag.split("}")[-1]
                if local not in CONCEPTS or "ifrs" not in node.tag.split("}")[0]:
                    continue
                audit["candidate_concepts"] += 1
                ctx = contexts.get(node.get("contextRef"))
                if ctx is None:
                    continue
                instant = ctx.findtext("x:period/x:instant", namespaces=NS)
                if not instant or instant[:7] != period:
                    continue
                entity = ctx.findtext("x:entity/x:identifier", namespaces=NS)
                if entity != corp:
                    audit["rejected_entity"] += 1
                    continue
                dims = [(d.get("dimension"), d.text)
                        for d in ctx.findall(".//d:explicitMember", NS)]
                typed = ctx.findall(".//d:typedMember", NS)
                fs_dims = [(axis, value) for axis, value in dims
                           if axis.split(":")[-1] == "ConsolidatedAndSeparateFinancialStatementsAxis"]
                if typed or len(dims) != 1 or len(fs_dims) != 1:
                    audit["rejected_dimensions"] += 1
                    continue
                fs_member = (fs_dims[0][1] or "").split(":")[-1]
                fs = {"ConsolidatedMember": "CFS", "SeparateMember": "OFS"}.get(fs_member)
                if fs is None:
                    audit["rejected_dimensions"] += 1
                    continue
                unit_node = units.get(node.get("unitRef"))
                measures = [] if unit_node is None else unit_node.findall("x:measure", NS)
                if len(measures) != 1 or not (measures[0].text or "").startswith("iso4217:"):
                    audit["rejected_unit"] += 1
                    continue
                raw_amount = node.text
                value = _amount(raw_amount)
                namespace = node.tag.split("}")[0][1:]
                prefix = "ifrs-full" if "ifrs-full" in namespace else "ifrs"
                facts.append({"statement_type": fs, "account": prefix + "_" + local,
                    "account_name": local, "equity": value, "amount_raw": raw_amount,
                    "unit": measures[0].text.split(":")[-1], "decimals": node.get("decimals"),
                    "context_ref": node.get("contextRef"),
                    "dimensions": json.dumps(dims, ensure_ascii=False),
                    "fiscal_period": instant, "member": member,
                    "location": f"concept={node.tag};contextRef={node.get('contextRef')}",
                    "extraction_status": "extracted" if value is not None else "numeric_parse_or_blank",
                    "source_kind": "xbrl"})
    seen = set()
    for f in facts:
        key = (f["statement_type"], f["account"], f["context_ref"], f["fiscal_period"], f["unit"])
        if key in seen:
            raise ValueError("duplicate_xbrl_fact_key")
        seen.add(key)
    return facts, audit


def plan_financial_requests(corp: str, index: list[dict], *, index_complete: bool) -> tuple[pd.DataFrame, dict]:
    """Only gate a fully paginated empty index; ambiguous fiscal calendars retain 56.

    No inference from listing date, later survival, a missing CFS response, or year-end
    arithmetic. All initial/intermediate correction receipts are collected separately.
    """
    fixed = plan_dart_collection(pd.DataFrame({"corp_code": [corp]}), config={})
    skip = index_complete and not index
    return fixed.iloc[:0].copy() if skip else fixed, {
        "fixed_requests": len(fixed), "planned_requests": 0 if skip else len(fixed),
        "reason": "complete_index_empty" if skip else "fiscal_calendar_safe_fallback",
        "absence_scope": "index_query_only" if skip else None,
    }


def _parse_preserved_document(job: tuple) -> tuple:
    path, corp, period = job
    try:
        return job, _document_evidence(Path(path).read_bytes(), corp, period), None
    except (ValueError, zipfile.BadZipFile) as error:
        message = str(error)
        detail = ":" + message if re.fullmatch(r"[a-z_]+", message) else ""
        return job, None, type(error).__name__ + detail


class DartCollector:
    def __init__(self, root: Path, *, snapshot: str = "20261031", workers: int = 2,
                 offline: bool = False, max_calls: int | None = None):
        self.root = root.resolve()
        self.snapshot = snapshot
        datetime.strptime(snapshot, "%Y%m%d")
        self.run = self.root / RUN_REL
        self.raw = self.root / RAW_REL
        self.processed = self.root / PROCESSED_REL
        self.run.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run / "manifest.jsonl"
        self.progress_path = self.run / "progress.json"
        self.lock = threading.RLock()
        self.rate_lock = threading.Lock()
        self.last_request = 0.0
        self.workers = workers
        self.stopped = threading.Event()
        self.network_calls = 0
        self.offline, self.max_calls = offline, max_calls
        self.cache_hits = 0
        self.cache = {}
        self.raw_records = []
        self.transport_failures = 0
        self.key = None
        env = self.root / ".env"
        for line in (env.read_text(encoding="utf-8-sig").splitlines() if env.is_file() else []):
            s = line.strip()
            if s.startswith("OPENDART_API_KEY="):
                self.key = s.split("=", 1)[1].strip().strip("\"'")
        if not self.key:
            self.key = os.environ.get("OPENDART_API_KEY")
        if not self.key and not offline:
            raise CollectionStop("OPENDART_API_KEY_missing")
        if self.key:
            os.environ["OPENDART_API_KEY"] = self.key
        for line in (self.log_path.read_text(encoding="utf-8").splitlines() if self.log_path.is_file() else []):
            r = json.loads(line)
            if r.get("event") == "raw":
                path = self.root / r["path"]
                if not path.is_file() or _sha(path.read_bytes()) != r["sha256"]:
                    raise CollectionStop("cached_raw_hash_mismatch")
                self.raw_records.append(r)
                if r["http"] == 200 and r["status"] in ("000", "ZIP_BODY", "013", "014"):
                    self.cache[r["key"]] = r
        if self.progress_path.exists():
            self.progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if self.progress["snapshot"] != snapshot:
                raise CollectionStop("snapshot_change_requires_separate_run")
        else:
            self.progress = {"snapshot": snapshot, "completed_corps": [], "batches": [],
                             "status": "prepared", "stop_reason": None}
        self.mapping = self._load_table("company_mapping", MAPPING_COLUMNS)
        self.versions = self._load_table("filing_versions", VERSION_COLUMNS)
        self.facts = self._load_table("equity_facts", FACT_COLUMNS)

    def _load_table(self, name: str, columns: tuple) -> pd.DataFrame:
        path = self.processed / f"{name}.parquet"
        return (pd.read_parquet(path).reindex(columns=columns).astype("string")
                if path.exists() else pd.DataFrame(columns=columns).astype("string"))

    def log(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        if self.key in line:
            raise CollectionStop("secret_in_log_rejected")
        with self.lock, self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()

    def checkpoint(self) -> None:
        for name, table, columns in (
            ("company_mapping", self.mapping, MAPPING_COLUMNS),
            ("filing_versions", self.versions, VERSION_COLUMNS),
            ("equity_facts", self.facts, FACT_COLUMNS),
        ):
            table = table.reindex(columns=columns).copy()
            for column in columns:
                table[column] = table[column].astype("string")
            path = self.processed / f"{name}.parquet"
            temp = path.with_suffix(".parquet.tmp")
            table.to_parquet(temp, index=False)
            temp.replace(path)
        self.progress.update(updated_at=_now(), network_calls_this_process=self.network_calls,
                             cache_hits_this_process=self.cache_hits)
        temp = self.progress_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.progress, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.progress_path)

    def get(self, endpoint: str, params: dict) -> tuple[dict, bytes, dict | None]:
        key = _request_key(endpoint, params)
        cached = self.cache.get(key)
        if cached:
            with self.lock:
                self.cache_hits += 1
            body = (self.root / cached["path"]).read_bytes()
            return cached, body, json.loads(body) if endpoint.endswith(".json") else None
        if self.offline:
            raise CollectionStop("offline_cache_miss:" + endpoint)
        if self.stopped.is_set():
            raise CollectionStop("another_request_stopped_collection")
        if shutil.disk_usage(self.root).free < 2 * 1024**3:
            self.stopped.set()
            raise CollectionStop("free_disk_below_2GiB")
        response = None
        for attempt in range(2):
            with self.rate_lock:
                wait = max(0, 0.16 - (time.monotonic() - self.last_request))
                if wait:
                    time.sleep(wait)
                self.last_request = time.monotonic()
            if self.stopped.is_set():
                raise CollectionStop("collection_stopped")
            try:
                with self.lock:
                    if self.max_calls is not None and self.network_calls >= self.max_calls:
                        self.stopped.set()
                        raise CollectionStop("request_budget_reached")
                    self.network_calls += 1
                self.log({"event": "request_started", "endpoint": endpoint, "params": params,
                          "attempt": attempt + 1, "at": _now()})
                response = requests.get("https://opendart.fss.or.kr/api/" + endpoint,
                    params={"crtfc_key": self.key, **params},
                    timeout=(15, 60), allow_redirects=False)
                break
            except (requests.Timeout, requests.ConnectionError) as e:
                self.log({"event": "transport_error", "endpoint": endpoint, "params": params,
                          "error_type": type(e).__name__, "attempt": attempt + 1, "at": _now()})
                if attempt == 0:
                    time.sleep(2)
        if response is None:
            self.stopped.set()
            raise CollectionStop("transport_failed_twice")
        body = response.content
        secret = self.key.encode()
        if secret in body:
            self.stopped.set()
            raise CollectionStop("secret_echo_response_rejected")
        zipped = zipfile.is_zipfile(io.BytesIO(body))
        if zipped:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                if sum(x.file_size for x in archive.infolist()) > 256 * 1024**2:
                    raise CollectionStop("archive_uncompressed_size_over_256MiB")
                if any(secret in archive.read(n) for n in archive.namelist()):
                    self.stopped.set()
                    raise CollectionStop("secret_echo_zip_rejected")
            status, data, suffix = "ZIP_BODY", None, ".zip"
        else:
            try:
                data = response.json() if endpoint.endswith(".json") else {
                    x.tag: x.text for x in ET.fromstring(body)}
            except (ValueError, ET.ParseError):
                data = {"status": "UNPARSEABLE"}
            status = str(data.get("status", "STATUS_MISSING"))
            suffix = ".json" if endpoint.endswith(".json") else ".xml"
        kind = ("registry" if endpoint == "corpCode.xml" else
                "filing_index" if endpoint == "list.json" else
                "financial_api" if endpoint == "fnlttSinglAcntAll.json" else "filing_originals")
        folder = self.raw / kind
        folder.mkdir(exist_ok=True)
        label = "_".join(str(params[k]) for k in
                         ("corp_code", "bsns_year", "reprt_code", "fs_div", "rcept_no", "page_no")
                         if k in params)
        stamp = datetime.now(KST).strftime("%Y%m%dT%H%M%S%f")
        name = f"{endpoint.split('.')[0]}_{label}_{_sha(key.encode())[:12]}_{status}_{stamp}{suffix}"
        path = folder / name
        with path.open("xb") as stream:
            stream.write(body)
        record = {"event": "raw", "source": "FSS OpenDART", "endpoint": endpoint,
                  "params": params, "key": key, "path": path.relative_to(self.root).as_posix(),
                  "sha256": _sha(body), "bytes": len(body), "http": response.status_code,
                  "status": status, "collected_at": _now(), "origin": "DART003"}
        self.log(record)
        with self.lock:
            self.raw_records.append(record)
            if response.status_code == 200 and status in ("000", "ZIP_BODY", "013", "014"):
                self.cache[key] = record
        if response.status_code != 200 or status not in ("000", "ZIP_BODY", "013", "014"):
            self.stopped.set()
            raise CollectionStop(f"http_{response.status_code}_api_{status}")
        return record, body, data

    def prepare_mapping(self) -> dict:
        import yaml
        from src.data_io import load_panel_contract
        config = yaml.safe_load((self.root / "config/experiment.yaml").read_text(encoding="utf-8"))
        panel = load_panel_contract(self.root, column_mapping=config["COLUMN_MAPPING"]["panel"],
                                    config=config)
        assert panel.eligible.isna().all()
        codes = sorted(panel.stock_id.unique())
        registry, body, _ = self.get("corpCode.xml", {})
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            all_records = []
            for member in archive.namelist():
                all_records.extend({n.tag: (n.text or "").strip() for n in node}
                                   for node in ET.fromstring(archive.read(member)).findall("list"))
        mapping = []
        for stock in codes:
            matches = [r for r in all_records if r.get("stock_code") == stock]
            if not matches:
                matches = [{}]
            for r in matches:
                mapping.append({"stock_id": stock, "corp_code": r.get("corp_code"),
                    "corp_name": r.get("corp_name"), "registry_modify_date": r.get("modify_date"),
                    "mapping_status": "not_in_current_registry" if not r else
                                      "ambiguous_current_mapping" if len(matches) > 1 else "current_exact",
                    "historical_identity_status": "unverified",
                    "valid_from": None, "valid_to": None,
                    "evidence_path": registry["path"], "sha256": registry["sha256"]})
        self.mapping = pd.DataFrame(mapping, columns=MAPPING_COLUMNS)
        matched = self.mapping[self.mapping.mapping_status == "current_exact"]
        stats = {"panel_codes": len(codes), "matched_codes": matched.stock_id.nunique(),
                 "unmatched_codes": self.mapping.loc[self.mapping.corp_code.isna(), "stock_id"].nunique(),
                 "unique_corps": matched.corp_code.nunique(),
                 "ambiguous_codes": self.mapping.loc[self.mapping.mapping_status == "ambiguous_current_mapping",
                                                      "stock_id"].nunique(),
                 "historical_unverified_codes": len(codes),
                 "multi_stock_corps": int((matched.groupby("corp_code").stock_id.nunique() > 1).sum())}
        self.progress["mapping"] = {k: int(v) for k, v in stats.items()}
        self.checkpoint()
        return self.progress["mapping"]

    def _add_facts(self, facts: list, version: dict, raw: dict) -> list:
        result = []
        for f in facts:
            r = {**f, "corp_code": version["corp_code"], "receipt_id": version["receipt_id"],
                 "bsns_year": version.get("bsns_year"), "reprt_code": version.get("reprt_code"),
                 "published_at": version.get("published_at"),
                 "availability_month": version.get("availability_month"),
                 "revision": version.get("revision"), "raw_path": raw["path"],
                 "sha256": raw["sha256"], "crosscheck_status": "not_checked"}
            r["fact_id"] = _sha(json.dumps([r.get(k) for k in
                ("corp_code", "receipt_id", "source_kind", "raw_path", "statement_type",
                 "account", "context_ref", "fiscal_period", "location", "unit")],
                ensure_ascii=False, default=str).encode())
            result.append(r)
        return result

    def collect_corp(self, corp: str) -> tuple[list, list, dict]:
        index, index_sources = [], {}
        page, total_page = 1, 1
        while page <= total_page:
            rec, _, data = self.get("list.json", {
                "corp_code": corp, "bgn_de": "20150101", "end_de": self.snapshot,
                "last_reprt_at": "N", "pblntf_ty": "A", "sort": "date",
                "sort_mth": "asc", "page_no": str(page), "page_count": "100"})
            if data["status"] == "013":
                break
            if data["status"] != "000":
                raise CollectionStop("filing_index_status_" + str(data["status"]))
            total_page = int(data["total_page"])
            rows = data.get("list", [])
            if any(r.get("corp_code") != corp for r in rows):
                raise CollectionStop("filing_index_corporation_mismatch")
            if len(rows) != min(100, max(0, int(data["total_count"]) - (page - 1) * 100)):
                raise CollectionStop("filing_index_page_count_mismatch")
            index.extend(rows)
            index_sources.update({r["rcept_no"]: rec for r in rows})
            page += 1
        if len({r["rcept_no"] for r in index}) != len(index):
            raise CollectionStop("duplicate_index_receipt")
        plan, request_plan = plan_financial_requests(corp, index, index_complete=True)
        self.log({"event": "financial_plan", "corp_code": corp, "at": _now(), **request_plan})
        api_results = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.get, "fnlttSinglAcntAll.json", row): row
                       for row in plan.to_dict("records")}
            for future in as_completed(futures):
                api_results.append((futures[future], future.result()))
        by_receipt = {r["rcept_no"]: r for r in index}
        anchors, api_rows = {}, []
        for params, (raw, _, data) in api_results:
            if data["status"] in ("013", "014"):
                continue
            for row in data.get("list", []):
                receipt = row.get("rcept_no")
                if (row.get("corp_code") != corp or row.get("bsns_year") != params["bsns_year"]
                        or row.get("reprt_code") != params["reprt_code"]):
                    raise CollectionStop("financial_api_identity_year_or_report_mismatch")
                source = by_receipt.get(receipt)
                group = _report_group(source["report_nm"]) if source else None
                if group:
                    anchor = (params["bsns_year"], params["reprt_code"])
                    if group[2] in anchors and anchors[group[2]] != anchor:
                        raise CollectionStop("conflicting_fiscal_year_report_anchor")
                    anchors[group[2]] = anchor
                if row.get("sj_div") == "BS" and row.get("account_id") in (
                        "ifrs_EquityAttributableToOwnersOfParent", "ifrs-full_EquityAttributableToOwnersOfParent",
                        "ifrs_Equity", "ifrs-full_Equity"):
                    api_rows.append((params, raw, row))
        versions = []
        for row in index:
            group = _report_group(row["report_nm"])
            if group is None:
                continue
            anchor = anchors.get(group[2])
            if anchor is None and not "2015" <= group[1][:4] <= "2021":
                continue
            raw = index_sources[row["rcept_no"]]
            code = anchor[1] if anchor else {"사업": "11011", "반기": "11012"}.get(group[0])
            revision = "correction" if "정정" in row["report_nm"] or "첨부추가" in row["report_nm"] else "initial"
            versions.append({"corp_code": corp, "receipt_id": row["rcept_no"],
                "published_at": row["rcept_dt"], "availability_month": _month_available(row["rcept_dt"]),
                "report_name": row["report_nm"], "report_kind": group[0], "fiscal_period_label": group[1],
                "bsns_year": anchor[0] if anchor else None, "reprt_code": code,
                "filing_group": corp + ":" + group[2], "revision": revision,
                "version_relation": "unverified", "index_path": raw["path"],
                "index_sha256": raw["sha256"], "original_status": "pending",
                "extraction_status": "pending", "crosscheck_status": "not_checked", "reason": None})
        groups = {}
        for v in versions:
            groups.setdefault(v["filing_group"], []).append(v)
        for values in groups.values():
            values.sort(key=lambda v: (v["published_at"], v["receipt_id"]))
            initial = [v for v in values if v["revision"] == "initial"]
            for v in values:
                if len(initial) == 1:
                    v["candidate_initial_receipt"] = initial[0]["receipt_id"]
                    if v["revision"] == "initial":
                        v.update(initial_receipt=v["receipt_id"], version_relation="initial_from_index")
                else:
                    v["reason"] = "missing_or_multiple_initial_versions"
        version_map = {v["receipt_id"]: v for v in versions}
        facts = []
        for params, raw, row in api_rows:
            v = version_map.get(row["rcept_no"])
            if v is None:
                # Do not invent a receipt date from a receipt-number prefix.
                v = {"corp_code": corp, "receipt_id": row["rcept_no"], "bsns_year": params["bsns_year"],
                     "reprt_code": params["reprt_code"], "published_at": None, "availability_month": None,
                     "revision": None}
            group = _report_group(by_receipt.get(row["rcept_no"], {}).get("report_nm", ""))
            f = {"statement_type": params["fs_div"], "account": row["account_id"],
                 "account_name": row["account_nm"], "equity": _amount(row.get("thstrm_amount")),
                 "amount_raw": row.get("thstrm_amount"), "unit": row.get("currency"),
                 "fiscal_period": None, "context_ref": None, "decimals": None, "dimensions": None,
                 "source_kind": "financial_api", "member": None,
                 "location": "BS/ord=" + row.get("ord", "") + ";thstrm_nm=" + row.get("thstrm_nm", ""),
                 "extraction_status": "current_response_only"}
            facts.extend(self._add_facts([f], v, raw))
        audits = []
        for v in versions:
            source_facts, audit = [], {}
            xraw = None
            preloaded_doc = None
            if not v["reprt_code"]:
                draw, body, _ = self.get("document.xml", {"rcept_no": v["receipt_id"]})
                v["document_path"] = draw["path"]
                if draw["status"] == "ZIP_BODY":
                    try:
                        preloaded_doc = _document_evidence(body, corp, v["fiscal_period_label"])
                        v["reprt_code"] = preloaded_doc["reprt_code_from_cover"]
                        if v["reprt_code"]:
                            v["reason"] = "reprt_from_cover_fiscal_year_unanchored:" + str(preloaded_doc["cover_period_evidence"])
                    except (ValueError, zipfile.BadZipFile):
                        v["reason"] = "document_cover_report_code_unresolved"
            if v["reprt_code"]:
                xraw, body, _ = self.get("fnlttXbrl.xml", {"rcept_no": v["receipt_id"],
                                                         "reprt_code": v["reprt_code"]})
                v["xbrl_path"] = xraw["path"]
                if xraw["status"] == "ZIP_BODY":
                    v["original_status"] = "xbrl_acquired"
                    try:
                        source_facts, audit = _xbrl_facts(body, corp, v["fiscal_period_label"])
                        facts.extend(self._add_facts(source_facts, v, xraw))
                    except (ValueError, ET.ParseError, zipfile.BadZipFile) as e:
                        v["reason"] = "xbrl_parse:" + type(e).__name__
                else:
                    v["reason"] = "xbrl_status_" + xraw["status"]
            else:
                v["reason"] = "reprt_code_unresolved_noncalendar_safe"
            needs_document = v["revision"] == "correction" or not source_facts
            if needs_document:
                draw, body, _ = self.get("document.xml", {"rcept_no": v["receipt_id"]})
                v["document_path"] = draw["path"]
                if draw["status"] == "ZIP_BODY":
                    v["original_status"] = ("xbrl_and_document_acquired"
                        if xraw is not None and xraw["status"] == "ZIP_BODY" else "document_acquired")
                    try:
                        doc = preloaded_doc or _document_evidence(body, corp, v["fiscal_period_label"])
                        v["document_parse_status"] = ("amount_extracted_label_only" if doc["facts"]
                                                       else "no_supported_amount_unresolved")
                        v["document_parse_issues"] = json.dumps(doc["parse_issues"])
                        candidate = version_map.get(v.get("candidate_initial_receipt"))
                        if v["revision"] == "correction" and candidate is not None:
                            if doc["initial_date"] == candidate["published_at"]:
                                v.update(initial_receipt=candidate["receipt_id"],
                                         version_relation="document_initial_date_corp_period")
                            else:
                                v["reason"] = "correction_initial_date_not_verified"
                        if source_facts:
                            matched, mismatches = 0, 0
                            for f in source_facts:
                                label_type = "owners" if f["account"].endswith("OwnersOfParent") else "total"
                                candidates = [d for d in doc["facts"]
                                    if d["statement_type"] == f["statement_type"]
                                    and d["unit"] == f["unit"] and d["fiscal_period"] == f["fiscal_period"]
                                    and (("지배" in d["account_name"]) == (label_type == "owners"))]
                                if candidates:
                                    if any(d["equity"] == f["equity"] for d in candidates):
                                        matched += 1
                                    else:
                                        mismatches += 1
                            v["crosscheck_status"] = ("mismatch" if mismatches else
                                "document_amount_match" if matched else "document_comparison_unavailable")
                        # Keep exact source provenance even when XBRL was also available.
                        # Body labels are diagnostic, never promoted to standard concepts.
                        facts.extend(self._add_facts(doc["facts"], v, draw))
                    except (ValueError, zipfile.BadZipFile) as e:
                        v["reason"] = "document_parse:" + type(e).__name__
                        v["document_parse_status"] = "parse_failed:" + type(e).__name__
                else:
                    v["reason"] = "document_status_" + draw["status"]
            v["extraction_status"] = "xbrl_extracted" if source_facts else "no_verified_xbrl_fact"
            audits.append({"receipt_id": v["receipt_id"], **audit})
            for f in facts:
                if f["receipt_id"] == v["receipt_id"]:
                    f["crosscheck_status"] = ("version_level_document_match_only"
                        if v["crosscheck_status"] == "document_amount_match" else "not_checked")
            # Display rounding and label ambiguity are unresolved comparison states,
            # not permission to replace either original amount.
        graph = pd.DataFrame(versions, columns=VERSION_COLUMNS)
        for idx, (previous, reason) in _verified_predecessors(graph).items():
            versions[idx]["previous_receipt"] = previous
            if reason:
                versions[idx]["reason"] = reason
        stats = {"corp_code": corp, "index_rows": len(index), "index_pages": total_page,
                 "financial_requests": len(api_results),
                 "request_plan": request_plan,
                 "financial_statuses": pd.Series([r[1][0]["status"] for r in api_results]).value_counts().to_dict(),
                 "versions": len(versions), "facts": len(facts),
                 "xbrl_extracted": sum(v["extraction_status"] == "xbrl_extracted" for v in versions),
                 "version_links_verified": sum(bool(v.get("initial_receipt")) for v in versions),
                 "parse_audits": audits}
        return versions, facts, stats

    def repair_preserved_documents(self, *, quarter_headers_only: bool = False) -> dict:
        """Regenerate body observations from cached originals only; never call get()."""
        if self.progress["status"] == "running":
            raise CollectionStop("repair_requires_checkpoint_stop")
        before = self.facts.copy()
        raw_by_receipt = {r["params"]["rcept_no"]: r for r in self.raw_records
                          if r["endpoint"] == "document.xml" and r["status"] == "ZIP_BODY"}
        jobs, version_for_job = [], {}
        for idx, row in self.versions.iterrows():
            raw = raw_by_receipt.get(row.receipt_id)
            if raw:
                if quarter_headers_only:
                    issue = row.get("document_parse_issues")
                    if pd.isna(issue) or "current_period_column_unresolved" not in issue:
                        continue
                    with zipfile.ZipFile(self.root / raw["path"]) as archive:
                        main = [n for n in archive.namelist() if n.endswith(".xml")
                                and not re.search(r"_\d{5}\.xml$", n)]
                        if len(main) != 1:
                            continue
                        text, _ = _decode_document(archive.read(main[0]))
                    if not re.search(r"제\s*\d+\s*(?:\([^)]*\))?\s*기\s*[13]\s*분기", text):
                        continue
                job = (str(self.root / raw["path"]), row.corp_code, row.fiscal_period_label)
                jobs.append(job)
                version_for_job[job] = idx
        body_facts, issues, changed_versions = [], [], []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            for count, (job, doc, error) in enumerate(pool.map(_parse_preserved_document, jobs), 1):
                idx = version_for_job[job]
                version = self.versions.loc[idx].where(self.versions.loc[idx].notna(), None).to_dict()
                raw = raw_by_receipt[version["receipt_id"]]
                self.versions.at[idx, "document_path"] = raw["path"]
                if error:
                    self.versions.at[idx, "document_parse_status"] = "parse_failed:" + error
                    issues.append({"receipt_id": version["receipt_id"], "error": error})
                else:
                    self.versions.at[idx, "document_parse_status"] = (
                        "amount_extracted_label_only" if doc["facts"] else "no_supported_amount_unresolved")
                    self.versions.at[idx, "document_parse_issues"] = json.dumps(doc["parse_issues"])
                    body_facts.extend(self._add_facts(doc["facts"], version, raw))
                    candidate = self.versions[self.versions.receipt_id.eq(version.get("candidate_initial_receipt"))]
                    if version["revision"] == "correction" and len(candidate) == 1:
                        if doc["initial_date"] == candidate.iloc[0].published_at:
                            self.versions.at[idx, "initial_receipt"] = candidate.iloc[0].receipt_id
                            self.versions.at[idx, "version_relation"] = "document_initial_date_corp_period"
                    if doc["facts"]:
                        changed_versions.append(version["receipt_id"])
                if count % 25 == 0 or count == len(jobs):
                    print(json.dumps({"repair_documents": count, "total": len(jobs),
                                      "body_facts": len(body_facts)}, ensure_ascii=True), flush=True)
        reparsed_receipts = {self.versions.loc[idx, "receipt_id"] for idx in version_for_job.values()}
        original_nonbody = before[before.source_kind.ne("document") |
                                  ~before.receipt_id.isin(reparsed_receipts)].copy()
        self.facts = pd.concat([original_nonbody, pd.DataFrame(body_facts, columns=FACT_COLUMNS)], ignore_index=True)
        def signature(frame):
            return set(map(tuple, frame[["corp_code", "receipt_id", "statement_type", "equity", "unit", "fiscal_period"]]
                           .where(frame.notna(), None).to_numpy()))
        old_body = before[before.source_kind.eq("document")]
        new_body = self.facts[self.facts.source_kind.eq("document")]
        report = {"before_rows": len(before), "after_rows": len(self.facts),
                  "before_document_rows": len(old_body), "after_document_rows": len(new_body),
                  "new_body_amount_signatures": len(signature(new_body) - signature(old_body)),
                  "removed_body_amount_signatures": len(signature(old_body) - signature(new_body)),
                  "documents_examined": len(jobs), "documents_with_supported_amount": len(changed_versions),
                  "parse_failures": issues, "network_calls": self.network_calls,
                  "original_API_XBRL_rows_preserved": int(before.source_kind.ne("document").sum()),
                  "quarter_headers_only": quarter_headers_only}
        self.progress.setdefault("repair_reports", []).append(report)
        self.progress["repair_report"] = report
        self.log({"event": "preserved_document_repair", "at": _now(), "report": report})
        self.validate()
        self.checkpoint()
        return report

    def corporation_states(self) -> dict:
        """Procedure, source acquisition and research PIT are separate axes."""
        states = {}
        for corp in self.progress["completed_corps"]:
            v = self.versions[self.versions.corp_code.eq(corp)]
            f = self.facts[self.facts.corp_code.eq(corp)]
            originals = f[f.source_kind.isin(["xbrl", "document"]) & f.equity.notna()]
            numeric_receipts = set(originals.receipt_id)
            raw = [r for r in self.raw_records if r["params"].get("corp_code") == corp or
                   r["params"].get("rcept_no") in set(v.receipt_id)]
            counts = pd.Series([r["status"] for r in raw], dtype="string").value_counts().to_dict()
            source_bad = any(r["status"] not in ("000", "ZIP_BODY", "013", "014") for r in raw)
            index_absent = any(r["endpoint"] == "list.json" and r["status"] == "013"
                               and r["params"].get("bgn_de") == "20150101" for r in raw)
            state = ("no_index_reports_in_query_scope" if v.empty and f.empty and index_absent else
                     "no_target_period_versions_identified" if v.empty and f.empty else
                     "original_amounts_all_indexed_versions" if len(v) and set(v.receipt_id) <= numeric_receipts else
                     "partial_original_amounts" if len(originals) else
                     "api_amounts_original_unverified" if f.equity.notna().any() else
                     "reports_exist_amount_extraction_unresolved")
            eligible = v[v.initial_receipt.notna() & v.previous_receipt.notna().where(v.revision.eq("correction"), True)
                         & v.bsns_year.notna() & v.reprt_code.notna() & v.published_at.notna()]
            ready = eligible[eligible.receipt_id.isin(originals.loc[
                originals.source_kind.eq("xbrl") & originals.statement_type.eq("CFS") &
                originals.account.str.endswith("EquityAttributableToOwnersOfParent", na=False), "receipt_id"])]
            states[corp] = {"procedure": "error_recorded" if source_bad else "complete",
                "acquisition": state, "pit": "not_verified_historical_identity_and_full_version_coverage",
                "versions": len(v), "original_numeric_versions": len(numeric_receipts),
                "version_linked": int(v.initial_receipt.notna().sum()),
                "source_PIT_candidate_versions": len(ready), "numeric_rows": int(f.equity.notna().sum()),
                "api_status_counts": counts,
                "missing_source_014": counts.get("014", 0), "request_absent_013": counts.get("013", 0),
                "body_unresolved_versions": int(v.document_parse_status.eq("no_supported_amount_unresolved").sum())}
        return states

    def audit_request_plans(self, corps: list[str]) -> dict:
        """Counterfactual against preserved fixed-plan responses; zero network access."""
        reports = []
        for corp in corps:
            indexes = [r for r in self.raw_records if r["endpoint"] == "list.json"
                       and r["params"].get("corp_code") == corp
                       and r["params"].get("bgn_de") == "20150101"
                       and r["params"].get("end_de") == self.snapshot]
            rows = []
            complete = False
            if indexes:
                pages = {str(r["params"]["page_no"]): r for r in indexes}
                first = json.loads((self.root / pages["1"]["path"]).read_bytes())
                if first["status"] == "013":
                    complete = True
                elif first["status"] == "000":
                    complete = set(map(str, range(1, int(first["total_page"]) + 1))) <= set(pages)
                    for page in pages.values():
                        rows.extend(json.loads((self.root / page["path"]).read_bytes()).get("list", []))
                    complete &= len(rows) == int(first["total_count"])
            plan, report = plan_financial_requests(corp, rows, index_complete=complete)
            allowed = {_request_key("fnlttSinglAcntAll.json", x) for x in plan.to_dict("records")}
            fixed = plan_dart_collection(pd.DataFrame({"corp_code": [corp]}), config={})
            fixed_keys = {_request_key("fnlttSinglAcntAll.json", x) for x in fixed.to_dict("records")}
            records = {k: self.cache[k] for k in fixed_keys if k in self.cache}
            excluded = [r for k, r in records.items() if k not in allowed]
            missed = [r["params"] for r in excluded if r["status"] == "000"]
            if missed:
                raise CollectionStop("request_plan_excludes_successful_response")
            reports.append({"corp_code": corp, **report, "fixed_cache_coverage": len(records),
                            "excluded_successes": len(missed), "excluded_observed": len(excluded)})
        return {"corporations": len(reports), "fixed_requests": sum(r["fixed_requests"] for r in reports),
                "planned_requests": sum(r["planned_requests"] for r in reports),
                "excluded_successes": 0, "per_corp": reports}

    def validate(self) -> dict:
        if self.mapping.stock_id.nunique() != 1086:
            raise CollectionStop("mapping_stock_count_changed")
        if self.versions.duplicated(["corp_code", "receipt_id"]).any():
            raise CollectionStop("duplicate_filing_key")
        if self.facts.fact_id.duplicated().any():
            raise CollectionStop("duplicate_fact_key")
        orphan = set(self.facts.receipt_id.dropna()) - set(self.versions.receipt_id.dropna())
        # API-only orphan observations are retained with explicitly unknown publication dates.
        unknown = self.facts[self.facts.receipt_id.isin(orphan)]
        if unknown.published_at.notna().any():
            raise CollectionStop("invented_publication_date")
        regression = self.facts[(self.facts.corp_code == "00111704") &
            (self.facts.source_kind == "xbrl") & (self.facts.statement_type == "CFS") &
            (self.facts.account.str.endswith("EquityAttributableToOwnersOfParent", na=False)) &
            (self.facts.fiscal_period == "2015-12-31")]
        for receipt, expected in (("20160330004251", "746668673663"), ("20170511004419", "929062351201")):
            rows = regression[regression.receipt_id == receipt]
            if len(rows) and set(rows.equity) != {expected}:
                raise CollectionStop("DART002_regression_amount_failed")
        return {"mapping_rows": len(self.mapping), "version_rows": len(self.versions),
                "fact_rows": len(self.facts), "orphan_api_receipts": len(orphan),
                "DART002_regression_receipts": sorted(regression.receipt_id.unique().tolist())}

    def finalize_quality(self) -> dict:
        """Audit acquired artifacts without any network call or monthly equity selection."""
        if self.progress["status"] == "running":
            raise CollectionStop("quality_finalize_requires_stopped_collector")
        for idx, (previous, reason) in _verified_predecessors(self.versions).items():
            self.versions.at[idx, "previous_receipt"] = previous
            if reason:
                self.versions.at[idx, "reason"] = reason
        by_path = {r["path"]: r for r in self.raw_records}
        for idx, row in self.versions.iterrows():
            x = by_path.get(row.xbrl_path, {}) if isinstance(row.xbrl_path, str) else {}
            d = by_path.get(row.document_path, {}) if isinstance(row.document_path, str) else {}
            flags = (x.get("status") == "ZIP_BODY", d.get("status") == "ZIP_BODY")
            self.versions.at[idx, "original_status"] = {
                (True, True): "xbrl_and_document_acquired", (True, False): "xbrl_acquired",
                (False, True): "document_acquired", (False, False): "not_acquired",
            }[flags]
        self.facts["crosscheck_status"] = "not_individually_checked"
        keys = ["corp_code", "receipt_id", "statement_type", "account", "unit"]
        api = self.facts[(self.facts.source_kind == "financial_api") & self.facts.equity.notna()].copy()
        xml = self.facts[(self.facts.source_kind == "xbrl") & self.facts.equity.notna()].copy()
        api["_row"] = api.index
        xml["_row"] = xml.index
        unique_xml = xml.drop_duplicates(keys, keep=False)
        paired = api.merge(unique_xml, on=keys, suffixes=("_api", "_xbrl"))
        mismatch, matched = [], []
        for r in paired.to_dict("records"):
            same = Decimal(r["equity_api"]) == Decimal(r["equity_xbrl"])
            status = "api_xbrl_exact_match" if same else "api_xbrl_amount_mismatch"
            for i in (r["_row_api"], r["_row_xbrl"]):
                self.facts.at[i, "crosscheck_status"] = status
            (matched if same else mismatch).append({k: r[k] for k in keys})
        # Label-only body comparisons are diagnostic, separate from standard-concept pairs.
        body_matches = body_rounding = 0
        for idx, row in self.facts[self.facts.source_kind.eq("document")].iterrows():
            is_owner = "지배" in re.sub(r"\s+", "", row.account_name)
            candidates = xml[xml.corp_code.eq(row.corp_code) & xml.receipt_id.eq(row.receipt_id)
                & xml.statement_type.eq(row.statement_type) & xml.unit.eq(row.unit)
                & xml.fiscal_period.eq(row.fiscal_period)
                & xml.account.str.endswith("EquityAttributableToOwnersOfParent" if is_owner else "_Equity", na=False)]
            amounts = set(candidates.equity)
            if len(amounts) != 1 or pd.isna(row.equity):
                continue
            delta = abs(Decimal(row.equity) - Decimal(next(iter(amounts))))
            if delta == 0:
                body_matches += 1
                self.facts.at[idx, "crosscheck_status"] = "document_label_amount_match_diagnostic"
            elif pd.notna(row.decimals) and delta <= Decimal(10) ** (-int(row.decimals)) / 2:
                body_rounding += 1
                self.facts.at[idx, "crosscheck_status"] = "document_display_rounding_diagnostic"
        # Check API identities against the preserved request, not a synthesized table.
        index_identity_errors, financial_identity_errors = [], []
        for raw in self.raw_records:
            if raw["status"] != "000" or raw["endpoint"] not in ("list.json", "fnlttSinglAcntAll.json"):
                continue
            rows = json.loads((self.root / raw["path"]).read_bytes()).get("list", [])
            p = raw["params"]
            for row in rows:
                if row.get("corp_code") != p["corp_code"]:
                    index_identity_errors.append(raw["path"])
                if raw["endpoint"] == "fnlttSinglAcntAll.json" and (
                        row.get("bsns_year") != p["bsns_year"] or row.get("reprt_code") != p["reprt_code"]):
                    financial_identity_errors.append(raw["path"])
        if index_identity_errors or financial_identity_errors:
            raise CollectionStop("acquired_api_request_identity_audit_failed")
        target_corps = set(self.mapping.loc[self.mapping.mapping_status == "current_exact", "corp_code"])
        remaining = sorted(target_corps - set(self.progress["completed_corps"]))
        unique_requests = {r["key"]: r for r in self.raw_records}
        financial = [r for r in unique_requests.values() if r["endpoint"] == "fnlttSinglAcntAll.json"
                     and r["params"].get("corp_code") in target_corps]
        years = {}
        for year in map(str, range(2015, 2022)):
            requests_in_year = [r for r in financial if r["params"]["bsns_year"] == year]
            years[year] = {"expected_requests": len(target_corps) * 8,
                "observed_requests": len(requests_in_year),
                "api_status_counts": pd.Series([r["status"] for r in requests_in_year]).value_counts().to_dict(),
                "CFS_observed": sum(r["params"]["fs_div"] == "CFS" for r in requests_in_year),
                "OFS_observed": sum(r["params"]["fs_div"] == "OFS" for r in requests_in_year),
                "version_rows_api_year_known": int(self.versions.bsns_year.eq(year).sum())}
        report = {**self.validate(), "remaining_corps": len(remaining),
            "financial_request_target": len(target_corps) * 56,
            "financial_requests_observed": len(financial),
            "api_xbrl_pairs_compared": len(paired), "exact_pairs": len(matched),
            "amount_mismatch_pairs": mismatch, "ambiguous_xbrl_keys": int(xml.duplicated(keys, keep=False).sum()),
            "numeric_rows": int(self.facts.equity.notna().sum()),
            "strict_pair_participating_rows": int(self.facts.crosscheck_status.eq("api_xbrl_exact_match").sum()),
            "body_label_matches_diagnostic": body_matches, "body_rounding_diagnostic": body_rounding,
            "version_extraction": self.versions.extraction_status.value_counts().to_dict(),
            "version_relations": self.versions.version_relation.value_counts().to_dict(),
            "original_acquisition": self.versions.original_status.value_counts().to_dict(),
            "fact_source_counts": self.facts.source_kind.value_counts().to_dict(),
            "unanchored_business_year_versions": int(self.versions.bsns_year.isna().sum()),
            "unresolved_report_code_versions": int(self.versions.reprt_code.isna().sum()),
            "yearly": years, "last_collection_network_calls": self.progress.get("network_calls_this_process"),
            "code_sha256": _sha(Path(__file__).read_bytes())}
        self.progress["remaining_corp_codes"] = remaining
        self.progress["corporation_states"] = self.corporation_states()
        report["corporation_state_counts"] = {
            axis: pd.Series([s[axis] for s in self.progress["corporation_states"].values()]).value_counts().to_dict()
            for axis in ("procedure", "acquisition", "pit")}
        self.progress["quality_report"] = report
        self.log({"event": "quality_finalized", "at": _now(), "report": report})
        self.checkpoint()
        return report

    def run_collection(self, *, batch_size: int = 5, first_batch: bool = False,
                       prepare_only: bool = False, max_new_corps: int | None = 10,
                       audit_fixed: bool = True) -> dict:
        if self.mapping.empty:
            self.prepare_mapping()
        if prepare_only:
            return {"status": "prepared", **self.progress["mapping"]}
        ordered = self.mapping[self.mapping.mapping_status == "current_exact"].drop_duplicates("corp_code")
        priority = {"042660": 0, "005930": 1, "000660": 2, "001720": 3}
        ordered = ordered.assign(_priority=ordered.stock_id.map(priority).fillna(10)).sort_values(
            ["_priority", "stock_id"])
        corps = [c for c in ordered.corp_code if c not in self.progress["completed_corps"]]
        if max_new_corps is not None:
            corps = corps[:max_new_corps]
        self.progress.update(status="running", stop_reason=None)
        self.checkpoint()
        try:
            for offset in range(0, len(corps), batch_size):
                batch = corps[offset:offset + batch_size]
                stats = []
                for corp in batch:
                    if self.progress_path.with_name("STOP").exists():
                        raise CollectionStop("user_stop_at_corporation_checkpoint")
                    started = time.monotonic()
                    self.log({"event": "corporation_started", "corp_code": corp, "at": _now()})
                    versions, facts, result = self.collect_corp(corp)
                    if audit_fixed:
                        # Pilot only: query omitted requests to verify the counterfactual.
                        # These audit calls are reported separately, not counted as savings.
                        fixed = plan_dart_collection(pd.DataFrame({"corp_code": [corp]}), config={})
                        for params in fixed.to_dict("records"):
                            self.get("fnlttSinglAcntAll.json", params)
                        result["fixed_plan_audit"] = self.audit_request_plans([corp])
                    result["elapsed_seconds"] = time.monotonic() - started
                    self.versions = pd.concat([self.versions,
                        pd.DataFrame(versions, columns=VERSION_COLUMNS)], ignore_index=True)
                    self.facts = pd.concat([self.facts,
                        pd.DataFrame(facts, columns=FACT_COLUMNS)], ignore_index=True)
                    self.validate()
                    self.progress["completed_corps"].append(corp)
                    self.progress.setdefault("corporation_timings", {})[corp] = result["elapsed_seconds"]
                    self.log({"event": "corporation_complete", "at": _now(), "report": result})
                    stats.append(result)
                    self.checkpoint()
                    print(json.dumps({"completed_corps": len(self.progress["completed_corps"]),
                        "target_corps": self.progress["mapping"]["unique_corps"],
                        "network_calls": self.network_calls, "cache_hits": self.cache_hits,
                        "last_corp": corp, "versions": len(self.versions), "facts": len(self.facts)},
                        ensure_ascii=True), flush=True)
                record = {"event": "batch", "at": _now(), "corps": batch, "results": stats,
                          "qa": self.validate()}
                self.log(record)
                self.progress["batches"].append({"corps": batch, "at": record["at"]})
                self.checkpoint()
                if first_batch:
                    self.progress["status"] = "first_batch_complete"
                    break
            else:
                self.progress["status"] = "pilot_complete" if max_new_corps is not None else "collection_attempts_complete"
        except CollectionStop as e:
            self.progress.update(status="technical_stop", stop_reason=str(e))
            self.log({"event": "technical_stop", "reason": str(e), "at": _now()})
        except Exception as e:
            # Preserve raw evidence and checkpoints; redact exception text that may contain a URL.
            self.progress.update(status="technical_stop", stop_reason="unhandled_" + type(e).__name__)
            self.log({"event": "technical_stop", "reason": self.progress["stop_reason"], "at": _now()})
        finally:
            self.checkpoint()
        return {**self.validate(), "status": self.progress["status"],
                "stop_reason": self.progress["stop_reason"],
                "completed_corps": len(self.progress["completed_corps"]),
                "batch_remaining_corps": len(corps) - sum(c in self.progress["completed_corps"] for c in corps),
                "remaining_corps": len(set(ordered.corp_code) - set(self.progress["completed_corps"]))}


MAPPING = Path('data/processed/opendart/company_mapping.parquet')


def read_events(path):
    if not path.exists():
        return []
    # A torn append is an integrity error, not permission to reset or silently truncate history.
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def atomic_json(path, value):
    """Never delete the previous checkpoint; retain the new temporary snapshot on failure."""
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(4):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.1 * (attempt + 1))


@contextmanager
def run_lock(folder):
    lock = folder / 'RUNNING'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


OUTPUT = Path("outputs/run_id/DART_EQUITY_EXTRACT")


# 원문 재추출의 입력은 data/raw/opendart에 보존된 원문뿐이다(2026-11-08). 수집 운영 기록(manifest)은 읽지 않는다.


FACT_FIELDS = ("corp_code", "receipt_id", "concept", "statement_type", "equity", "amount_raw", "unit",
               "fiscal_period", "fiscal_period_basis", "source_kind", "section", "column_basis", "account_name", "member",
               "location", "identity_ok", "raw_path", "sha256", "components_match")


FILING_FIELDS = ("corp_code", "receipt_id", "published_at", "availability_month", "report_kind",
                 "report_period_label", "revision", "report_name", "fiscal_period", "fiscal_period_basis", "equity", "equity_tier",
                 "statement_type", "account_name", "unit", "source_kind", "section", "column_basis",
                 "identity_ok", "owners_identity_ok", "owners_resolution", "xbrl_document_agree", "nci_observed", "unit_suspect", "reason",
                 "raw_path", "sha256")


DERIVED_TIER = "T1D_consolidated_total_less_nci"


TIERS = (("T1_consolidated_owners", "CFS", "owners"), (DERIVED_TIER, "CFS", "total_less_nci"),
         ("T2_consolidated_total", "CFS", "total"), ("T3_separate_total", "OFS", "total"))


XBRL_OWNERS_TOLERANCE = Decimal("0.05")  # a smaller gap can be a real extra equity line, not a bad tag


UNIT_STEPS = (3, 6)  # 천원 and 백만원 slips, as log10 distance


UNIT_MARGIN = 0.3  # about a factor of two either way around the step


SCALE = {"억원": 100000000, "백만원": 1000000, "천원": 1000, "원": 1}


# '요약별도재무제표' heads a summary table too; without it an earlier prose mention became the title.
# A bare '재무제표' match is dropped in _statement_title.
TITLE = re.compile(r"(요약)?(연결|별도|개별)?(요약)?(재무상태표|대차대조표|재무정보|재무제표)")


SECTION_TITLE = re.compile(r"<TITLE\b[^>]*>(.*?)</TITLE>", re.I | re.S)


# 'VII. 주주에 관한 사항' carries the largest shareholder's financials (e.g. a parent holding company).
OTHER_ENTITY_SECTION = re.compile(r"주주에관한사항|계열회사|관계회사")  # affiliates' results tables too


# '[신탁계정]' heads a trust company's statement of clients' assets; '변경전' an appended pre-change restatement.
FOREIGN_SUBJECT = re.compile(r"종속|관계기업|관계회사|공동기업|공동지배|피투자|지분법|피합병|합병|분할|사업결합|인수|정정전|정정후|신탁계정|변경전")


REFERENCE_ONLY = re.compile(r"^\W{0,3}(?:정정전|변경전)")  # '대차대조표 - 정정전': the superseded version


DATE = re.compile(r"(\d{4})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})")


TERM = re.compile(r"제(\d+)(?:\([^)]*\))?기")


UNIT = re.compile(r"단위[^가-힣A-Za-z0-9]{0,6}(억원|백만원|천원|원|USD|달러|US\$|엔|JPY|위안|CNY|EUR|유로)")


MARKUP = re.compile(r"<[^>]+>")


SPACE = re.compile(r"\s+")


NUMBERING = re.compile(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]+|\d+|[가-하])[.)]+")


INCOME_START = re.compile(r"^\[?(?:매출액|매출수익|영업수익|영업이익|영업손익|영업손실|법인세|당기순|분기순|반기순|계속영업)")


# Summary tables carry several periods with no statement date; only an explicit date is safe there.
SUMMARY_BASES = ("header_date", "header_term_dated")


# R3 (user decision 2026-11-03): up to this label, a column found by the cover's term number also counts in a
# summary, and the cover's '제N기 (start ~ end)' end date dates any table found that way. Later labels unchanged.
COVER_TERM_LAST_PERIOD = "2014-12"


COVER_PERIOD = r"(?:\([^)]*\))?기.{0,20}?(\d{4})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})\s*일?\s*(?:부터|~|-|～|∼)\s*(\d{4})\s*[./년-]\s*(\d{1,2})\s*[./월-]\s*(\d{1,2})"


COVER = re.compile(r"<COVER\b.*?</COVER>", re.I | re.S)


def cover_term_period(text: str, period: str) -> tuple[str | None, str | None]:
    """(term, end) from the cover's own '제N기 … start ~ end' whose end month is the label's month.

    A correction filing opens with a notice that names other terms ('제25기 재무제표의 내용 중…'), and a
    long cover table pushes the period past the first few thousand characters, so read the whole cover.
    """
    block = COVER.search(text)
    flat = _flat(MARKUP.sub(" ", block[0] if block else text[:20000]))
    for match in re.finditer(r"제(\d+)" + COVER_PERIOD, flat):
        end = f"{int(match[5]):04d}-{int(match[6]):02d}-{int(match[7]):02d}"
        if end[:7] == period:
            return match[1], end
    return None, None


def _bracketed(raw: str) -> bool:
    flat = raw.replace(" ", "")
    return flat.startswith("(") and flat.endswith(")") and not flat.startswith("(-)") and "-" not in flat and "△" not in flat


def _real_date(end: str | None, basis: str | None) -> tuple[str | None, str | None]:
    """A written day the month does not have ('2003.06.31') is that month's end; the month itself is stated."""
    if end is None:
        return end, basis
    year, month, day = (int(p) for p in end.split("-"))
    if not 1 <= month <= 12:
        return None, None
    last = calendar.monthrange(year, month)[1]
    return (end, basis) if 1 <= day <= last else (f"{year:04d}-{month:02d}-{last:02d}", "invalid_day_month_end")


def cover_period_end(cover: str, term: str | None, period: str) -> str | None:
    """End date of the cover's own term, only when its month is the report label's month."""
    match = re.search("제" + term + COVER_PERIOD, cover) if term else None
    if not match:
        return None
    end = f"{int(match[4]):04d}-{int(match[5]):02d}-{int(match[6]):02d}"
    return end if end[:7] == period else None


def _flat(text: str) -> str:
    return SPACE.sub("", text.replace("&cr;", " ").replace("&nbsp;", " ").replace("　", " "))


def concept_of(label: str) -> str | None:
    """Row label -> concept. Nothing substitutes for another concept here."""
    # 2001-era labels append English: 자 본 총 계 (Total Stockholders' Equity)
    name = re.sub(r"[\s\[\]<>()　]|&nbsp;|\(?[A-Za-z][A-Za-z'’&.,:/\- ]*\)?", "", label)
    name = NUMBERING.sub("", name)
    if re.search(r"매각예정", name) and re.search(r"자본$|포괄손익", name) and not re.search(r"총계|합계", name):
        return "held_for_sale_equity"  # its own equity line beside owners and NCI; used in the owners identity only
    if not name or len(name) > 40 or re.search(r"이익|손익|손실|포괄", name):
        return None  # 지배회사지분순이익 is an income line
    if re.search(r"비지배|소수주주|외부주주|소수지분", name):
        return None if re.search(r"자산|부채", name) else "nci"  # 비지배지분부채 is a liability line
    if re.search(r"^(?:지배기업|지배회사|지배주주|지배지분|모회사|모기업|소유주)", name) and re.search(r"소유주|지분|귀속|주주", name):
        # 소유주분배예정자산집단 is an asset line held for distribution, not equity
        return None if re.search(r"자산|부채|분배", name) else "owners"
    if re.search(r"부채|자산", name) and re.search(r"자본", name):
        return None  # 부채와자본총계 is total assets by another name
    if re.fullmatch(r"자본(?:총계|합계|계|의총계|총액)|총자본|기말자본", name):
        return "total"
    if re.fullmatch(r"자산(?:총계|합계|계)|총자산", name):
        return "assets"
    if re.fullmatch(r"부채(?:총계|합계|계)|총부채", name):
        return "liabilities"
    return None


def _grid(table) -> list[list[str]]:
    """Expand spans over every row: DART headers often live in TBODY, values in TE."""
    occupied, grid = {}, []
    for ri, row in enumerate(table.find_all("tr")):
        col = 0
        for cell in row.find_all(["td", "th", "te", "tu"], recursive=False):
            while (ri, col) in occupied:
                col += 1
            width = min(max(int(cell.get("colspan", 1) or 1), 1), 100)
            height = min(max(int(cell.get("rowspan", 1) or 1), 1), 100)
            text = cell.get_text(" ", strip=True).replace("&cr;", " ")
            for dr in range(height):
                for dc in range(width):
                    occupied[ri + dr, col + dc] = text
            col += width
        last = max((c for r, c in occupied if r == ri), default=-1)
        grid.append([occupied.get((ri, c), "") for c in range(last + 1)])
    return grid


def _explode_multiline(table, grid: list[list[str]]) -> tuple[list[list[str]], bool]:
    """2001-era statements may hold every account in one <br>-separated row.

    Split only when label and amount line counts agree; the caller then demands the
    balance-sheet identity, so a misaligned split cannot produce a value.
    """
    for ri, row in enumerate(table.find_all("tr")):
        cells = row.find_all(["td", "th", "te", "tu"], recursive=False)
        if len(cells) < 2:
            continue
        lines = [[part.strip() for part in cell.get_text("\n").split("\n")] for cell in cells]
        if len(lines[0]) < 8 or not any(len(column) == len(lines[0]) for column in lines[1:]):
            continue
        rows = [[column[i] if len(column) == len(lines[0]) else "" for column in lines] for i in range(len(lines[0]))]
        return grid[:ri] + rows + grid[ri + 1:], True
    return grid, False


def _numeric(text: str) -> bool:
    return _amount(text.replace("(-)", "-").replace(" ", "")) is not None


def _value(text: str, scale: int) -> str | None:
    return _amount(text.replace("(-)", "-").replace(" ", ""), scale)


def _header_end(headers: list[str], columns: list[int], period: str) -> tuple[str | None, str | None]:
    """Statement date written in the current column's own header: a full date, or an explicit 'M월말'."""
    for col in columns:
        for y, m, d in DATE.findall(_flat(headers[col])):
            if f"{int(y):04d}-{int(m):02d}" == period:
                return f"{int(y):04d}-{int(m):02d}-{int(d):02d}", "header_full_date"
    for col in columns:
        for y, m in re.findall(r"(\d{4})[./년-](\d{1,2})월?말", _flat(headers[col])):
            if f"{int(y):04d}-{int(m):02d}" == period:
                last = calendar.monthrange(int(y), int(m))[1]
                return f"{int(y):04d}-{int(m):02d}-{last:02d}", "header_month_end_word"
    return None, None


def resolve_current_columns(headers: list[str], context: str, period: str, cover_term: str | None) -> tuple[list[int], str | None, str | None, str | None]:
    """Return (columns, basis, period end, end basis). A blank or prior column is never a fallback.

    The period end comes from a dated statement heading, else from the current column's header.
    A report label's month is never turned into a statement date here.
    """
    dates = [(m.start(), f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}") for m in DATE.finditer(context)]
    ends = [d for _, d in dates if d[:7] == period]
    if dates and not ends:
        return [], "statement_date_is_other_period", None, None
    end = ends[0] if ends else None
    end_basis = "context_date" if end else None
    first_is_current = bool(dates) and dates[0][1][:7] == period
    terms = {}
    for term, year, month, _ in re.findall(r"제(\d+)(?:\([^)]*\))?기.{0,30}?(\d{4})[./년-](\d{1,2})[./월-](\d{1,2})", context):
        terms.setdefault(f"{int(year):04d}-{int(month):02d}", term)
    term = terms.get(period) or cover_term
    by_date, by_term, by_word, numbered = [], [], [], {}
    for col, raw in enumerate(headers):
        text = _flat(raw)
        if not text or "주석" in text or col == 0:
            continue
        found = [f"{int(y):04d}-{int(m):02d}" for y, m, _ in DATE.findall(text)]
        found += [f"{int(y):04d}-{int(m):02d}" for y, m in re.findall(r"(\d{4})[./년-](\d{1,2})월?말?$", text)]
        if period in found:
            by_date.append(col)
        match = TERM.search(text)
        if match:
            numbered.setdefault(match[1], []).append(col)
            if term and match[1] == term:
                by_term.append(col)
        if re.search(r"당(?:기|반기|분기|\))", text) and "전" not in text.replace("당", "", 1)[:3]:
            by_word.append(col)
    if by_date:
        if end is None:
            end, end_basis = _header_end(headers, by_date, period)
        return by_date, "header_date", end, end_basis
    if by_term:
        return by_term, "header_term_dated" if terms.get(period) else "header_term_cover", end, end_basis
    if by_word and (first_is_current or not dates):
        return by_word, "header_current_word", end, end_basis
    if numbered and (first_is_current or not dates):
        latest = max(numbered, key=int)
        return numbered[latest], "header_latest_term", end, end_basis
    return [], "current_column_unresolved", end, end_basis


def _statement_title(context: str, section: str = "") -> dict | None:
    """Nearest statement title above a table, or None when the table is not that statement.

    A title without 연결/별도 takes its scope from the enclosing DART section ('2. 연결재무제표' or
    '4. 재무제표') before the '…와 그 종속기업' wording, which filers also leave on separate statements.
    """
    flat = _flat(context)
    # A bare '재무제표' is usually prose ('연결재무제표 주석 참조'); only '요약…재무제표' heads a table.
    matches = [m for m in TITLE.finditer(flat) if m[4] != "재무제표" or m[1] or m[3]]
    if not matches:
        return _section_statement_title(flat, section)
    title = matches[-1]
    after = flat[title.end():]
    if FOREIGN_SUBJECT.search(flat[max(0, title.start() - 30):title.start()]) or REFERENCE_ONLY.match(after):
        return None
    if len(after) > 500:
        return _section_statement_title(flat, section)
    summary = bool(title[1] or title[3]) or title[4] == "재무정보"
    if title[2] == "연결":
        scope, basis = "CFS", "title"
    elif title[2] in {"별도", "개별"}:
        scope, basis = "OFS", "title"
    elif re.fullmatch(r"\d+\.연결재무제표", section):
        scope, basis = "CFS", "section"
    elif re.fullmatch(r"\d+\.재무제표", section):
        scope, basis = "OFS", "section"
    elif re.search(r"연결기준|연결실체|그종속|및종속|\(연결\)", after):
        scope, basis = "CFS", "title_subject"
    else:
        scope, basis = "OFS", "unlabeled_title"
    return {"scope": scope, "scope_basis": basis, "section": "summary" if summary else "full", "after": after}


def _section_statement_title(flat: str, section: str) -> dict | None:
    """The DART section itself named as the statement ('대차대조표'), when no title sits right above the table.

    2000-era filings put the statement name only in <TITLE> and a small date/unit table in between, so the
    nearby text has no title. The text before the table still supplies dates and the unit.
    """
    matches = [m for m in TITLE.finditer(section) if m[4] != "재무제표" or m[1] or m[3]]
    if not matches or FOREIGN_SUBJECT.search(section):
        return None
    title = matches[-1]
    summary = bool(title[1] or title[3]) or title[4] == "재무정보"
    after = flat[-500:]
    if title[2] == "연결":
        scope, basis = "CFS", "section_title"
    elif title[2] in {"별도", "개별"}:
        scope, basis = "OFS", "section_title"
    elif re.search(r"연결기준|연결실체|그종속|및종속|\(연결\)", after):
        scope, basis = "CFS", "title_subject"
    else:
        scope, basis = "OFS", "unlabeled_section_title"
    return {"scope": scope, "scope_basis": basis, "section": "summary" if summary else "full", "after": after}


def _unit(after: str, table_text: str) -> tuple[int | None, str | None]:
    found = UNIT.findall(after) or UNIT.findall(_flat(table_text[:600]))
    if not found:
        return None, None
    return SCALE.get(found[-1]), found[-1]


def is_statement_shape(chunk: str) -> bool:
    stripped = _flat(MARKUP.sub(" ", chunk))
    return bool(re.search(r"자본총계|자본합계|총자본|자본계|소유주", stripped)
                and re.search(r"자산총계|자산합계|총자산|자산계|부채총계|부채합계|부채계", stripped))


def document_facts(text: str, period: str, *, member: str = "") -> tuple[list[dict], Counter]:
    """All balance-sheet concepts of the current column, with the table shape as a gate."""
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
    facts, audit = [], Counter()
    head = _flat(MARKUP.sub(" ", text[:6000]))
    cover = TERM.search(head)
    cover_term = cover[1] if cover else None
    by_cover = period <= COVER_TERM_LAST_PERIOD
    cover_end = None
    if by_cover:
        own_term, cover_end = cover_term_period(text, period)
        cover_term = own_term or cover_term
        cover_end = cover_end or cover_period_end(head, cover_term, period)
    sections = [(m.start(), _flat(MARKUP.sub(" ", m[1]))) for m in SECTION_TITLE.finditer(text)]
    starts = [s for s, _ in sections]
    position = 0
    for number, match in enumerate(re.finditer(r"<table\b.*?</table>", text, re.I | re.S)):
        chunk = match[0]
        if not is_statement_shape(chunk):
            position = match.end()
            continue
        audit["shape_candidate_tables"] += 1
        at = bisect.bisect_left(starts, match.start()) - 1
        section = sections[at][1] if at >= 0 else ""
        if OTHER_ENTITY_SECTION.search(section):
            audit["other_entity_section"] += 1
            position = match.end()
            continue
        context = MARKUP.sub(" ", text[max(position - 4000, 0):match.start()])
        position = match.end()
        title = _statement_title(context, section)
        if title is None:
            audit["no_statement_title"] += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            table = BeautifulSoup(chunk, "html.parser").find("table")
        grid, exploded = _explode_multiline(table, _grid(table))
        first = next((i for i, row in enumerate(grid) if any(_numeric(c) for c in row[1:])), None)
        if first is None:
            continue
        width = max(len(row) for row in grid)
        heads = [" ".join(dict.fromkeys(row[c] for row in grid[:first] if c < len(row))) for c in range(width)]
        columns, basis, end, end_basis = resolve_current_columns(heads, title["after"], period, cover_term)
        if end is None and basis == "header_term_cover" and cover_end:
            end, end_basis = cover_end, "cover_term_period"
        end, end_basis = _real_date(end, end_basis)
        if not columns:
            audit[basis or "current_column_unresolved"] += 1
            continue
        latest = TERM.search(_flat(heads[columns[0]])) if basis == "header_latest_term" else None
        if by_cover and latest and cover_term and int(latest[1]) < int(cover_term):
            audit["latest_term_is_prior_period"] += 1  # a comparative table of an earlier year, not this report's
            continue
        scale, unit = _unit(title["after"], chunk)
        if scale is None:
            audit["unit_unresolved" if unit is None else "unit_not_krw"] += 1
            continue
        if title["section"] == "summary" and basis not in SUMMARY_BASES and not (by_cover and basis == "header_term_cover"):
            audit["summary_column_not_dated"] += 1
            continue
        found = {}
        for ri, row in enumerate(grid[first:], first):
            label = next((c for c in row[:2] if c.strip() and not _numeric(c)), "")
            if title["section"] == "summary" and INCOME_START.search(_flat(label)):
                break  # summaries continue with income lines that reuse equity-like labels
            concept = concept_of(label)
            if concept is None or concept in found:
                continue
            values = {row[c].strip() for c in columns if c < len(row) and row[c].strip() not in ("", "\\", "₩", "KRW", "-")}
            if len(values) != 1:
                audit["blank_current_amount" if not values else "ambiguous_current_amount"] += 1
                continue
            raw = next(iter(values))
            amount = _value(raw, scale)
            if amount is not None:
                found[concept] = {"label": label.strip(), "raw": raw, "amount": amount, "row": ri}
        if not {"total", "owners"} & set(found):
            continue
        if not {"assets", "liabilities"} <= set(found):
            audit["no_balance_sheet_shape"] += 1
            continue
        bracketed = Decimal(found["assets"]["amount"]) < 0 and _bracketed(found["assets"]["raw"])
        if bracketed:
            # Total assets cannot be negative: this table brackets its totals, so a bracket is not a minus sign here.
            for item in found.values():
                if _bracketed(item["raw"]):
                    item["amount"] = format(abs(Decimal(item["amount"])), "f")
            audit["brackets_as_subtotals"] += 1
        identity = None
        if "total" in found:
            gap = abs(Decimal(found["assets"]["amount"]) - Decimal(found["liabilities"]["amount"]) - Decimal(found["total"]["amount"]))
            identity = gap <= 2 * scale
            audit["identity_failed"] += not identity
        if exploded and identity is not True:
            audit["multiline_identity_unconfirmed"] += 1
            continue
        components = _components_match(grid, columns, found, scale) if {"owners", "total"} <= set(found) else None
        for concept in ("owners", "nci", "held_for_sale_equity", "total"):
            if concept in found:
                item = found[concept]
                facts.append({"concept": concept, "statement_type": title["scope"], "equity": item["amount"],
                              "amount_raw": item["raw"], "unit": "KRW", "fiscal_period": end,
                              "fiscal_period_basis": end_basis, "source_kind": "document",
                              "section": title["section"], "column_basis": f"{basis}|{title['scope_basis']}|{unit}",
                              "account_name": item["label"], "member": member,
                              "location": f"TABLE[{number}]/ROW[{item['row']}]", "identity_ok": identity,
                              **({"brackets_as_subtotals": True} if bracketed else {}),
                              **({"components_match": components} if concept == "owners" else {})})
    return facts, audit


def _components_match(grid: list[list[str]], columns: list[int], found: dict, scale: int) -> bool | None:
    """Do the rows under the owners subtotal add up to total less NCI? None when it cannot be told.

    Flat statements list each component once; two-column ones put details left and subtotals right, so
    either every row or only the owners row's own column must add up. Nested subtotals fit neither and
    stay undecided, which leaves a broken owners row to manual review.
    """
    owners, total = found["owners"], found["total"]
    nci = found.get("nci")
    stop = nci["row"] if nci and owners["row"] < nci["row"] < total["row"] else total["row"]
    if stop - owners["row"] < 2:
        return None
    column = next((c for c in columns if c < len(grid[owners["row"]]) and grid[owners["row"]][c].strip() == owners["raw"]), None)
    every, same = Decimal(0), Decimal(0)
    for row in grid[owners["row"] + 1:stop]:
        cells = [(c, _value(row[c].strip(), scale)) for c in columns
                 if c < len(row) and row[c].strip() not in ("", "\\", "₩", "KRW", "-")]
        cells = [(c, v) for c, v in cells if v is not None]
        if len({v for _, v in cells}) > 1:
            return None
        if cells:
            every += Decimal(cells[0][1])
            same += Decimal(cells[0][1]) if cells[0][0] == column else 0
    target = Decimal(total["amount"]) - (Decimal(nci["amount"]) if nci else 0)
    return any(abs(s - target) <= 2 * scale for s in (every, same))


def xbrl_facts(body: bytes, corp: str, period: str) -> list[dict]:
    facts, _ = _xbrl_facts(body, corp, period)
    return [{"concept": "owners" if f["account_name"] == "EquityAttributableToOwnersOfParent" else "total",
             "statement_type": f["statement_type"], "equity": f["equity"], "amount_raw": f["amount_raw"],
             "unit": f["unit"], "fiscal_period": f["fiscal_period"], "fiscal_period_basis": "xbrl_instant",
             "source_kind": "xbrl", "section": "xbrl",
             "column_basis": "instant_context", "account_name": f["account"], "member": f["member"],
             "location": f["location"], "identity_ok": None} for f in facts if f["equity"] is not None]


def _rank(fact: dict) -> tuple:
    return (fact["source_kind"] == "xbrl", fact["section"] == "full", fact.get("identity_ok") is True)


def _total_less_nci(facts: list[dict]) -> dict | None:
    """Owners by subtraction, only inside one balanced statement that has no owners row."""
    tables = defaultdict(dict)
    for fact in facts:
        if (fact["source_kind"] == "document" and fact["statement_type"] == "CFS" and fact["section"] == "full"
                and fact.get("identity_ok") is True and fact["unit"] == "KRW"):
            tables[fact.get("member"), fact.get("location", "").split("/")[0]][fact["concept"]] = fact
    found = [{**t["total"], "concept": "owners", "account_name": "자본총계-비지배지분",
              "equity": format(Decimal(t["total"]["equity"]) - Decimal(t["nci"]["equity"]), "f")}
             for t in tables.values() if "total" in t and "nci" in t and "owners" not in t]
    votes = Counter(f["equity"] for f in found).most_common()
    if not votes or (len(votes) > 1 and votes[0][1] == votes[1][1]):
        return None
    return next(f for f in found if f["equity"] == votes[0][0])


def _dated_when_possible(facts: list[dict]) -> list[dict]:
    """Undated facts leave the pool when the filing has a dated full statement (XBRL or a balanced body table).

    An undated value can never serve a month, yet it outranked or outvoted dated ones: a summary T1 over a
    dated T1D of the same amount, and a thrice-repeated prior-year adjustment table over the real statement.
    """
    if any(f.get("fiscal_period") and (f["source_kind"] == "xbrl" or (f["section"] == "full" and f.get("identity_ok") is True))
           for f in facts):
        return [f for f in facts if f.get("fiscal_period")]
    return facts


def _owners_identity(facts: list[dict], chosen: dict) -> bool | None:
    """Chosen body owners vs owners + NCI = total in its own table; None when there is nothing to check.

    A break is a misread column or a filer's wrong subtotal. Dropping the row and taking the next candidate
    was tried and picked a mis-scaled XBRL in one filing, so a break only marks the value for review.
    An XBRL choice is checked through the body owners rows of the same amount: filers tag their own subtotal.
    """
    checked = [_owners_balance(facts, f) for f in _body_owner_rows(facts, chosen) if _table_has_total(facts, f)]
    return (True in checked) if checked else None


def _body_owner_rows(facts: list[dict], chosen: dict) -> list[dict]:
    if chosen["concept"] != "owners" or chosen["account_name"] == "자본총계-비지배지분":
        return []
    if chosen["source_kind"] == "document":
        return [chosen]
    return [f for f in facts if f["concept"] == "owners" and f["source_kind"] == "document" and f["statement_type"] == "CFS"
            and abs(Decimal(f["equity"]) - Decimal(chosen["equity"])) <= _scale_of(f)]


def _owners_from_components(facts: list[dict], chosen: dict) -> dict | None:
    """Total less NCI of the broken row's own table, only when that table balances and its component rows
    add up to it. Anything less certain stays a flagged value for manual entry."""
    found = []
    for row in _body_owner_rows(facts, chosen):
        if row.get("components_match") is not True or row.get("identity_ok") is not True:
            continue
        key = (row.get("member"), row.get("location", "").split("/")[0])
        table = {f["concept"]: f for f in facts
                 if f["source_kind"] == "document" and (f.get("member"), f.get("location", "").split("/")[0]) == key}
        nci = Decimal(table["nci"]["equity"]) if "nci" in table else Decimal(0)
        found.append({**table["total"], "concept": "owners", "account_name": "자본총계-비지배지분",
                      "equity": format(Decimal(table["total"]["equity"]) - nci, "f")})
    return found[0] if found and len({f["equity"] for f in found}) == 1 else None


def _table_has_total(facts: list[dict], fact: dict) -> bool:
    key = (fact.get("member"), fact.get("location", "").split("/")[0])
    return any(f["concept"] == "total" and f["source_kind"] == "document"
               and (f.get("member"), f.get("location", "").split("/")[0]) == key for f in facts)


def _owners_balance(facts: list[dict], owners: dict) -> bool:
    """owners + NCI = total within the owners row's own table; no NCI row means owners must equal total."""
    if owners["source_kind"] != "document":
        return False
    key = (owners.get("member"), owners.get("location", "").split("/")[0])
    table = {f["concept"]: f for f in facts
             if f["source_kind"] == "document" and (f.get("member"), f.get("location", "").split("/")[0]) == key}
    if "total" not in table:
        return False
    nci = Decimal(table["nci"]["equity"]) if "nci" in table else Decimal(0)
    # A held-for-sale equity line is either a sibling of owners and NCI or a component inside owners;
    # the grid has no indentation to tell which, so either sum may balance.
    held = [Decimal(0)] + ([Decimal(table["held_for_sale_equity"]["equity"])] if "held_for_sale_equity" in table else [])
    return any(abs(Decimal(owners["equity"]) + nci + h - Decimal(table["total"]["equity"])) <= 2 * _scale_of(owners) for h in held)


def _decision(chosen: dict, tier: str, agree: bool | None) -> dict:
    return {**{k: chosen[k] for k in ("equity", "statement_type", "account_name", "unit", "source_kind",
                                      "section", "column_basis", "identity_ok", "fiscal_period")},
            "fiscal_period_basis": chosen.get("fiscal_period_basis"),
            "raw_path": chosen.get("raw_path"), "sha256": chosen.get("sha256"),
            "equity_tier": tier, "xbrl_document_agree": agree, "reason": None}


def _cover_summary(fact: dict) -> bool:
    return fact["section"] == "summary" and fact["column_basis"].startswith("header_term_cover")


def select_filing(facts: list[dict]) -> dict:
    """One filing's tier. A tie in repeated amounts stays unresolved, never picked.

    The same statement appears in the report body and its audit attachment, so among
    equal-rank amounts one repeated value beats one-off note tables of other entities.
    A summary column found only by the cover's term (R3) is a last resort: it is used when the
    filing gives no value otherwise, and never outranks a statement by tier.
    """
    if any(_cover_summary(f) for f in facts):
        first = select_filing([f for f in facts if not _cover_summary(f)])
        if first["equity"] is not None:
            return first
    facts = _dated_when_possible(facts)
    derived = _total_less_nci(facts)
    for tier, scope, concept in TIERS:
        if concept == "total_less_nci":
            if derived:
                return _decision(derived, tier, None)
            continue
        pool = [f for f in facts if f["statement_type"] == scope and f["concept"] == concept and f["unit"] == "KRW"
                and f.get("identity_ok") is not False]
        if not pool:
            continue
        best = max(_rank(f) for f in pool)
        top = [f for f in pool if _rank(f) == best]
        amounts = [Decimal(f["equity"]) for f in top]
        if max(amounts) - min(amounts) < max(_scale_of(f) for f in top):
            # One statement shown in won and in rounded millions is one amount: keep the finer figure.
            finest = min(_scale_of(f) for f in top)
            top = [f for f in top if _scale_of(f) == finest]
        if concept == "owners" and len({Decimal(f["equity"]) for f in top}) > 1:
            # A misread column or a typo breaks owners + NCI = total in its own table; another
            # company's table balances too, so two balanced amounts stay a conflict.
            balanced = [f for f in top if _owners_balance(facts, f)]
            if len({Decimal(f["equity"]) for f in balanced}) == 1:
                top = balanced
        votes = Counter(Decimal(f["equity"]) for f in top).most_common()
        if len(votes) > 1 and votes[0][1] == votes[1][1]:
            return {"equity": None, "equity_tier": None, "reason": "same_rank_amounts_conflict:" + tier}
        chosen = next(f for f in top if Decimal(f["equity"]) == votes[0][0])
        others = [f for f in pool if f["source_kind"] != chosen["source_kind"]]
        agree = None
        if chosen["source_kind"] == "xbrl" and others:
            tolerance = max(Decimal(_scale_of(f)) for f in others)
            agree = any(abs(Decimal(chosen["equity"]) - Decimal(f["equity"])) <= tolerance for f in others)
            # Filer tagging errors exist (one instance is exactly 10^6 times its statement).
            balanced = [f for f in others if f["section"] == "full" and f.get("identity_ok") is True]
            if not agree and len({Decimal(f["equity"]) for f in balanced}) == 1:
                chosen = balanced[0]
        elif concept == "owners" and chosen["source_kind"] == "xbrl" and derived:
            # Some filers tag paid-in capital as owners' equity; the balanced statement outranks it.
            value = Decimal(derived["equity"])
            if abs(Decimal(chosen["equity"]) - value) > abs(value) * XBRL_OWNERS_TOLERANCE:
                return _decision(derived, DERIVED_TIER, False)
        identity = _owners_identity(facts, chosen)
        fixed = _owners_from_components(facts, chosen) if identity is False else None
        if fixed:
            return {**_decision(fixed, DERIVED_TIER, None), "owners_identity_ok": False,
                    "owners_resolution": "components_verified_total_less_nci"}
        return {**_decision(chosen, tier, agree), "owners_identity_ok": identity}
    return {"equity": None, "equity_tier": None, "reason": "target_concepts_not_observed"}


def _scale_of(fact: dict) -> int:
    return SCALE.get(fact["column_basis"].rsplit("|", 1)[-1], 1)


def extract_filing(job: dict) -> dict:
    """Worker: verifies source bytes, parses, never raises for one bad filing."""
    root = Path(job["root"])
    facts, audit, problems = [], Counter(), []
    for kind in ("xbrl", "document"):
        source = job.get(kind)
        if not source:
            continue
        try:
            body = (root / source["path"]).read_bytes()
        except OSError:
            problems.append(kind + ":source_missing")
            continue
        digest = hashlib.sha256(body).hexdigest()
        if digest != source["sha256"]:
            problems.append(kind + ":source_hash_mismatch")
            continue
        try:
            if kind == "xbrl":
                found = xbrl_facts(body, job["corp_code"], job["report_period_label"])
                facts.extend({**f, "raw_path": source["path"], "sha256": digest} for f in found)
                continue
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                parts = [(n, archive.read(n)) for n in sorted(archive.namelist()) if n.lower().endswith(".xml")]
        except Exception as error:
            problems.append(f"{kind}:parse_failed:{type(error).__name__}")
            continue
        for name, raw in parts:  # one unreadable attachment must not hide the other members
            try:
                found, seen = document_facts(_decode_document(raw)[0], job["report_period_label"], member=name)
            except Exception as error:
                problems.append(f"document:{name}:parse_failed:{type(error).__name__}")
                continue
            audit.update(seen)
            facts.extend({**f, "raw_path": source["path"], "sha256": digest} for f in found)
    decision = select_filing(facts)
    if decision["equity"] is None and not (job.get("xbrl") or job.get("document")):
        decision["reason"] = "no_verified_original"
    elif decision["equity"] is None and problems:
        decision["reason"] = ";".join(problems)[:300]
    return {"event": "extraction", "source_key": job["source_key"], "corp_code": job["corp_code"],
            "receipt_id": job["rcept_no"], "facts": facts, "decision": decision,
            "audit": dict(audit), "problems": problems, "extractor_version": EXTRACTOR_VERSION,
            "code_sha256": CODE_FINGERPRINT}


def load_jobs(root: Path, extra_runs: list[Path], source_runs=()) -> tuple[list[dict], Counter]:
    """Last endpoint state per run, later runs winning; only verified sources become inputs.

    수집 도구(dart_collect·kind_collect)는 자기 수집 폴더를 extra_runs로 넘긴다. 폴더를 넘기지 않으면
    data/raw/opendart의 보존 원문만 쓴다(원문 재추출 기본 경로, 2026-11-08).
    """
    filings, counts = {}, Counter()
    for run in [root / r for r in source_runs] + extra_runs:
        for event in read_events(run / "manifest.jsonl"):
            if event.get("event") != "endpoint_state":
                continue
            kind = "xbrl" if event["endpoint"].startswith("fnlttXbrl") else "document"
            filing = filings.setdefault(event["rcept_no"], {
                "root": str(root), "corp_code": event["corp_code"], "rcept_no": event["rcept_no"],
                "rcept_dt": event["rcept_dt"], "report_nm": event["report_nm"],
                "report_period_label": event["report_period_label"]})
            if filing["corp_code"] != event["corp_code"]:
                raise ValueError("receipt_linked_to_two_corporations")
            filing.setdefault("states", {})[kind] = event["state"]
            if event["state"] == "original_verified" and event.get("path"):
                filing[kind] = {"path": event["path"], "sha256": event["sha256"]}
    if not filings:
        metadata = filing_observations(root)
        sources = preserved_dart_sources(root, metadata)
        for event in sources:
            receipt = event['params']['rcept_no']
            m = metadata.loc[metadata.rcept_no.eq(receipt)].iloc[0]
            kind = 'xbrl' if event['endpoint'] == 'fnlttXbrl.xml' else 'document'
            job = filings.setdefault(receipt, {'root': str(root), 'corp_code': m.corp_code,
                'rcept_no': receipt, 'rcept_dt': m.rcept_dt, 'report_nm': m.report_nm,
                'report_period_label': m.report_period_label})
            candidate = {'path': event['path'], 'sha256': event['sha256']}
            if kind in job and job[kind]['sha256'] != candidate['sha256']:
                raise ValueError('multiple_original_payloads_without_collection_selection:' + receipt)
            job[kind] = candidate
        if not filings:
            raise FileNotFoundError('required_DART_originals_missing:data/raw/opendart')
    for filing in filings.values():
        counts["filings"] += 1
        counts["no_verified_original"] += not (filing.get("xbrl") or filing.get("document"))
        filing["source_key"] = source_key(filing["rcept_no"], {k: filing.get(k) for k in ("xbrl", "document")})
    return sorted(filings.values(), key=lambda f: f["rcept_no"]), counts


def source_key(receipt: str, sources: dict) -> str:
    """Resume key. The code fingerprint makes any parser edit re-extract, whatever the version string says."""
    return hashlib.sha256(json.dumps([receipt, sources, EXTRACTOR_VERSION, CODE_FINGERPRINT],
                                     sort_keys=True).encode()).hexdigest()


def _revision(name: str) -> str:
    match = re.match(r"\[([^\]]+)\]", name)
    return match[1] if match else "최초표시"


def unit_suspects(filings: pd.DataFrame) -> pd.Series:
    """Flag values about 10^3 or 10^6 from the firm's last unflagged earlier value.

    A filer's unit slip scales the body, the XBRL and the balance identity together, so only the
    firm's own history shows it. Earlier filings only, so the flag is usable point in time.
    The first filing can itself be the slip, so a run outnumbering its reference replaces it.
    This marks a filing for exclusion or review; no amount is corrected here.
    """
    flags = pd.Series(False, index=filings.index)
    ordered = filings.assign(value=pd.to_numeric(filings.equity, errors="coerce")).sort_values(["published_at", "receipt_id"])
    for _, group in ordered[ordered.value > 0].groupby("corp_code"):
        last, support, run = None, 0, 0
        for index, value in group.value.items():
            if last is not None and any(abs(abs(math.log10(value / last)) - step) <= UNIT_MARGIN for step in UNIT_STEPS):
                flags[index] = True
                run += 1
                if run > support:
                    last, support, run = value, run, 0
            else:
                last, support, run = value, support + 1, 0
    return flags


DART_FILINGS = Path("data/interim/dart_filings.parquet")
DART_FACTS = Path("data/interim/dart_equity_facts.parquet")


def filing_observations(root: Path) -> pd.DataFrame:
    """Filing identity from common data; recover market labels from original list responses."""
    metadata = pd.read_parquet(root / DART_FILINGS)
    frame = metadata.rename(columns={'receipt_id': 'rcept_no', 'published_at': 'rcept_dt',
                                     'report_name': 'report_nm'}).copy()
    if frame.rcept_no.duplicated().any():
        raise ValueError('duplicate_filing_observation')
    for col in ('corp_cls', 'rm'):
        if col not in frame: frame[col] = ''
    # These are source responses, not collection manifests. No missing label is inferred.
    for path, group in frame[frame.source_path.notna()].groupby('source_path', sort=False):
        source = (root / path).resolve()
        if not source.is_relative_to((root / RAW_REL).resolve()):
            raise ValueError('filing_source_outside_opendart')
        if not source.is_file(): continue
        body = source.read_bytes()
        expected = group.source_sha256.dropna().unique()
        if len(expected) != 1 or _sha(body) != expected[0]:
            raise ValueError('filing_source_hash_mismatch:' + path)
        rows = {r['rcept_no']: r for r in json.loads(body).get('list', [])}
        for i in group.index:
            row = rows.get(frame.at[i, 'rcept_no'])
            if row is None or row.get('corp_code') != frame.at[i, 'corp_code']:
                raise ValueError('filing_source_identity_mismatch')
            for col in ('corp_cls', 'rm'):
                frame.at[i, col] = row.get(col, '')
    return frame


def preserved_dart_sources(root: Path, metadata: pd.DataFrame | None = None) -> list[dict]:
    """Identify preserved ZIPs by factual provenance or receipt-bearing filenames.

    No progress, collection, or extraction journal is required. A known source whose
    bytes changed is an error. Missing originals remain missing, never restored from facts.
    """
    if metadata is None: metadata = filing_observations(root)
    identified = metadata.set_index('rcept_no')
    paths = {}
    facts_path = root / DART_FACTS
    if facts_path.is_file():
        facts = pd.read_parquet(facts_path, columns=['receipt_id', 'source_kind', 'raw_path', 'sha256'])
        for f in facts.dropna(subset=['raw_path']).drop_duplicates(['receipt_id', 'source_kind', 'raw_path', 'sha256']).itertuples():
            if f.receipt_id in identified.index:
                paths[f.raw_path] = (f.receipt_id, 'xbrl' if f.source_kind == 'xbrl' else 'document', f.sha256)
    # Only the declared original directories and their direct run children are inspected.
    for relative in ('filing_originals', 'original_collect', 'quarter_collect', 'history_collect'):
        parent = root / RAW_REL / relative
        if not parent.is_dir(): continue
        folders = [parent] + [p for p in parent.iterdir() if p.is_dir()]
        for folder in folders:
            for p in folder.glob('*.zip'):
                match = re.search(r'(\d{14})_(document|xbrl)|(?:document|fnlttXbrl)_(\d{14})_', p.name)
                if not match: continue
                receipt = match[1] or match[3]
                if receipt not in identified.index: continue
                kind = match[2] or ('xbrl' if p.name.startswith('fnlttXbrl') else 'document')
                paths.setdefault(p.relative_to(root).as_posix(), (receipt, kind, None))
    records = []
    for relative, (receipt, kind, expected) in sorted(paths.items()):
        path = (root / relative).resolve()
        if not path.is_relative_to((root / RAW_REL).resolve()):
            raise ValueError('original_source_outside_opendart')
        if not path.is_file(): continue
        body = path.read_bytes(); digest = _sha(body)
        if expected and digest != expected:
            raise ValueError('preserved_original_hash_mismatch:' + relative)
        if not zipfile.is_zipfile(io.BytesIO(body)):
            raise ValueError('preserved_original_not_zip:' + relative)
        records.append({'event': 'raw', 'endpoint': 'fnlttXbrl.xml' if kind == 'xbrl' else 'document.xml',
            'params': {'rcept_no': receipt}, 'corp_code': identified.at[receipt, 'corp_code'],
            'path': relative, 'sha256': digest, 'bytes': len(body), 'http': 200, 'status': 'ZIP_BODY'})
    return records


def load_common_dart(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read public filing metadata and extracted observations, without execution state."""
    from src.contracts import DART_FILING_INPUT_COLUMNS, DART_FACT_INPUT_COLUMNS
    metadata, facts = (pd.read_parquet(root / p) for p in (DART_FILINGS, DART_FACTS))
    for table, columns in ((metadata, DART_FILING_INPUT_COLUMNS), (facts, DART_FACT_INPUT_COLUMNS)):
        missing = set(columns) - set(table.columns)
        if missing:
            raise ValueError("dart_input_missing_columns:" + ",".join(sorted(missing)))
    if metadata.receipt_id.isna().any() or metadata.receipt_id.duplicated().any():
        raise ValueError("dart_duplicate_or_missing_receipt")
    if metadata[["corp_code", "published_at"]].isna().any().any():
        raise ValueError("dart_missing_filing_identity")
    if not set(facts.receipt_id).issubset(set(metadata.receipt_id)):
        raise ValueError("dart_fact_without_filing")
    corp = facts.receipt_id.map(metadata.set_index("receipt_id").corp_code)
    if facts.corp_code.ne(corp).any():
        raise ValueError("dart_fact_corp_conflict")
    return metadata, facts


def public_filings(metadata: pd.DataFrame) -> pd.DataFrame:
    """Original public-list population, including filings with no extracted value."""
    return metadata.loc[metadata.listed_in_public_index].rename(columns={
        "receipt_id": "rcept_no", "published_at": "rcept_dt", "report_name": "report_nm"}).copy()


def decisions_from_facts(metadata: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """Recompute tier selection and unit flags from observations using the existing rules.

    Preserve repeated facts: repetitions are votes in select_filing. Unit flags retain
    the two original disclosure-format populations; neither run folders nor journals
    are calculation inputs.
    """
    selected = metadata[metadata.disclosure_format.notna()]
    if not selected.disclosure_format.isin(["document_only", "structured_supported"]).all():
        raise ValueError("dart_unknown_disclosure_format")
    groups = {}
    for row in facts.astype(object).where(facts.notna(), None).to_dict("records"):
        for field in ("identity_ok", "components_match"):
            value = row.get(field)
            if value in ("True", "False"):
                row[field] = value == "True"
        groups.setdefault(row["receipt_id"], []).append(row)
    rows = []
    for m in selected.itertuples(index=False):
        observed = groups.get(m.receipt_id, [])
        decision = select_filing(observed)
        matching = next((f for f in observed if f["equity"] == decision.get("equity")), {})
        rows.append({**decision, "corp_code": m.corp_code, "receipt_id": m.receipt_id,
                     "published_at": m.published_at, "availability_month": _month_available(m.published_at),
                     "report_kind": m.report_kind, "report_period_label": m.report_period_label,
                     "revision": _revision(m.report_name), "report_name": m.report_name,
                     "nci_observed": any(f["concept"] == "nci" for f in observed),
                     "raw_path": decision.get("raw_path") or matching.get("raw_path"),
                     "sha256": decision.get("sha256") or matching.get("sha256")})
    result = pd.DataFrame(rows, columns=FILING_FIELDS).astype("string")
    formats = result.receipt_id.map(selected.set_index("receipt_id").disclosure_format)
    for form in formats.unique():
        indices = result.index[formats.eq(form)]
        result.loc[indices, "unit_suspect"] = unit_suspects(result.loc[indices]).astype("string")
    return result


def write_tables(folder: Path, jobs: list[dict]) -> dict:
    """Rebuild both tables from the append-only journal; one row per filing version."""
    latest = {e["source_key"]: e for e in read_events(folder / "extractions.jsonl")}
    fact_rows, filing_rows, proxy = [], [], Counter()
    for job in jobs:
        event = latest.get(job["source_key"])
        if event is None:
            continue
        group = _report_group(job["report_nm"])
        decision = event["decision"]
        _score_document_only(event["facts"], proxy)
        fact_rows.extend({**{k: f.get(k) for k in FACT_FIELDS}, "corp_code": job["corp_code"],
                          "receipt_id": job["rcept_no"]} for f in event["facts"])
        filing_rows.append({**{k: decision.get(k) for k in FILING_FIELDS}, "corp_code": job["corp_code"],
                            "receipt_id": job["rcept_no"], "published_at": job["rcept_dt"],
                            "availability_month": _month_available(job["rcept_dt"]),
                            "report_kind": group[0] if group else None,
                            "report_period_label": job["report_period_label"],
                            "revision": _revision(job["report_nm"]), "report_name": job["report_nm"],
                            "nci_observed": any(f["concept"] == "nci" for f in event["facts"]),
                            "raw_path": decision.get("raw_path") or next((f["raw_path"] for f in event["facts"] if f["equity"] == decision.get("equity")), None),
                            "sha256": decision.get("sha256") or next((f["sha256"] for f in event["facts"] if f["equity"] == decision.get("equity")), None)})
    facts = pd.DataFrame(fact_rows, columns=FACT_FIELDS).astype("string")
    filings = pd.DataFrame(filing_rows, columns=FILING_FIELDS).astype("string")
    filings["unit_suspect"] = unit_suspects(filings).astype("string")
    facts.to_parquet(folder / "equity_facts.parquet", index=False)
    filings.to_parquet(folder / "equity_filings.parquet", index=False)
    return {**quality(facts, filings), "document_only_proxy": dict(proxy)}


def _score_document_only(facts: list[dict], counts: Counter) -> None:
    """Pre-2015 filings have only a body: rehearse that on filings where XBRL can grade it."""
    truth = select_filing([f for f in facts if f["source_kind"] == "xbrl"])
    if truth["equity"] is None:
        return
    guess = select_filing([f for f in facts if f["source_kind"] == "document"])
    counts["graded_filings"] += 1
    if guess["equity"] is None:
        counts["document_found_nothing"] += 1
    elif guess["equity_tier"] != truth["equity_tier"]:
        counts["document_other_tier"] += 1
    elif abs(Decimal(guess["equity"]) - Decimal(truth["equity"])) <= _scale_of_decision(guess):
        counts["document_same_tier_same_amount"] += 1
    else:
        counts["document_same_tier_wrong_amount"] += 1


def _scale_of_decision(decision: dict) -> int:
    return SCALE.get(decision["column_basis"].rsplit("|", 1)[-1], 1)


def quality(facts: pd.DataFrame, filings: pd.DataFrame) -> dict:
    """Coverage by era, plus body-parser accuracy measured against XBRL on the same concept."""
    frame = filings.assign(year=filings.report_period_label.str[:4])
    report = {"filings": len(frame), "equity_found": int(frame.equity.notna().sum()),
              "tier_counts": frame.equity_tier.value_counts(dropna=False).rename(str).to_dict(),
              "reason_counts": frame.reason.value_counts().to_dict(),
              "tier_by_year": {str(y): g.equity_tier.fillna("none").value_counts().to_dict() for y, g in frame.groupby("year")},
              "nonpositive_equity": int(pd.to_numeric(frame.equity, errors="coerce").le(0).sum()),
              "unit_suspect": int(frame.unit_suspect.eq("True").sum()),
              "owners_identity_failed": int(frame.owners_identity_ok.eq("False").sum()),
              "owners_resolved_by_components": int(frame.owners_resolution.notna().sum())}
    keys = ["receipt_id", "statement_type", "concept"]
    x = facts[facts.source_kind.eq("xbrl")][keys + ["equity"]].drop_duplicates(keys)
    doc = facts[facts.source_kind.eq("document")]
    pairs = doc.merge(x, on=keys, suffixes=("", "_xbrl"))
    if len(pairs):
        pairs = pairs.assign(scale=pairs.column_basis.str.rsplit("|", n=1).str[-1].map(SCALE).astype("Float64"),
                             gap=(pd.to_numeric(pairs.equity) - pd.to_numeric(pairs.equity_xbrl)).abs())
        pairs["match"] = pairs.gap <= pairs.scale
        best = pairs.groupby(keys + ["section"], as_index=False).match.max()
        report["document_vs_xbrl"] = {f"{s}_{c}": {"pairs": len(g), "match_rate": round(float(g.match.mean()), 5)}
                                      for (s, c), g in best.groupby(["section", "concept"])}
        report["document_vs_xbrl_by_basis"] = {b: {"pairs": len(g), "match_rate": round(float(g.match.mean()), 5)}
                                               for b, g in pairs.groupby("column_basis")}
    report["note"] = "Filing-version values only. Not monthly as-of, not PIT verified; tier choice remains reversible."
    return report


EXTRACTOR_VERSION = "9"


def code_fingerprint() -> str:
    """This module's source (since 2026-11-04 it also holds the former dart_asof helpers and the DART-only as-of)."""
    import inspect
    parts = [Path(__file__).read_bytes()] + [inspect.getsource(f).encode() for f in (_amount, _decode_document, _xbrl_facts)]
    parts.append(repr((sorted(CONCEPTS), NS)).encode())
    return hashlib.sha256(b"\0".join(parts)).hexdigest()


# A resumed main_v8 kept 975 records made before a mid-run edit under the same version string.
CODE_FINGERPRINT = code_fingerprint()










def legacy_extract_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Resume this DART_EQUITY_EXTRACT folder")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, help="Evenly spaced subset; the same subset on resume")
    parser.add_argument("--execute", action="store_true", help="Write outputs; this mode never uses the network")
    parser.add_argument("--source-run", type=Path, action="append",
                        help="수집 도구 전용: 방금 수집한 폴더의 원문을 추출(dart_collect 밤 체인이 명시적으로 넘김)")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    # 기본(--reextract)은 data/raw/opendart의 보존 원문뿐이다(2026-11-08). 수집 운영 기록을 기본으로 찾지 않으므로
    # 누구의 컴퓨터에서나 같은 목록이 된다. --source-run은 수집 도구가 자기 폴더를 명시할 때만 쓴다.
    source_runs = tuple(args.source_run or ())
    jobs, counts = load_jobs(root, [], source_runs=source_runs)
    if args.limit:
        step = max(1, len(jobs) // args.limit)
        jobs = jobs[::step][:args.limit]
    folder = (root / args.run_dir).resolve() if args.run_dir else root / OUTPUT / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    print(json.dumps({"filings_available": counts["filings"], "selected": len(jobs), "no_verified_original": counts["no_verified_original"],
                      "workers": args.workers, "run_dir": str(folder),
                      "network_enabled": False}, ensure_ascii=False, indent=2))
    if not args.execute:
        print("DRY-RUN: no outputs written.")
        return
    folder.mkdir(parents=True, exist_ok=True)
    with run_lock(folder):
        run = folder / "collection.json"
        if not run.exists():
            with run.open("x", encoding="utf-8") as stream:
                json.dump({"extractor_version": EXTRACTOR_VERSION, "code_sha256": CODE_FINGERPRINT,
                           "source": ([str(p) for p in source_runs] if source_runs
                                      else "data/raw/opendart preserved originals"),
                           "limit": args.limit,
                           "tiers": [t[0] for t in TIERS]}, stream, ensure_ascii=False, indent=2)
        seen = {e["source_key"] for e in read_events(folder / "extractions.jsonl")}
        todo = [j for j in jobs if j["source_key"] not in seen]
        done, status = 0, "extraction_finished_not_PIT_verified"
        pool = ProcessPoolExecutor(max_workers=args.workers)
        try:
            pending = [pool.submit(extract_filing, job) for job in todo]
            with (folder / "extractions.jsonl").open("a", encoding="utf-8") as stream:
                for future in as_completed(pending):
                    if (folder / "STOP").exists():
                        status = "stop_requested"
                        break
                    stream.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                    stream.flush()
                    done += 1
                    if done % 500 == 0:
                        print(json.dumps({"done": done, "of": len(todo)}), flush=True)
        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        report = {"status": status, "recorded_this_invocation": done, "extractor_version": EXTRACTOR_VERSION,
                  "code_sha256": CODE_FINGERPRINT, **write_tables(folder, jobs)}
        atomic_json(folder / "progress.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    """Recompute filing choices from the common observations; raw re-extraction is explicit."""
    import sys
    if "--reextract" in sys.argv:
        sys.argv.remove('--reextract')
        legacy_extract_main()
        return
    if "--source-run" in sys.argv:  # 수집 도구(dart_collect 밤 체인) 전용
        legacy_extract_main()
        return
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--execute", action="store_true", help="Save recomputed choices under outputs/run_id")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    metadata, facts = load_common_dart(root)
    filings = decisions_from_facts(metadata, facts)
    report = {"filings": len(filings), "observations": len(facts),
              "selected_values": int(filings.equity.notna().sum()),
              "inputs_sha256": {p.as_posix(): _sha((root / p).read_bytes()) for p in (DART_FILINGS, DART_FACTS)},
              "code_sha256": CODE_FINGERPRINT, "network_enabled": False}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.execute:
        folder = root / OUTPUT / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        folder.mkdir(parents=True, exist_ok=False)
        filings.to_parquet(folder / "equity_filings.parquet", index=False)
        atomic_json(folder / "run.json", report)


def record_cli_execution(callback, name):
    """Record this invocation as output; calculation never requires this record."""
    import sys
    command=list(sys.argv)
    try:
        result=callback()
    except BaseException:
        raise
    if '--execute' in command:
        folder=Path.cwd()/'outputs/run_id'/name/datetime.now().strftime('%Y%m%dT%H%M%S%f')
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'run.json').write_text(json.dumps({'status':'EXECUTED','command':command,
            'code_sha256':CODE_FINGERPRINT},
            ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__ == "__main__":
    main()
