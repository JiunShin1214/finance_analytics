"""Stage 1, DART collection (kept as code; no further collection planned): filing lists, originals, night chain.
Merged 2026-11-04 from dart_list_only, dart_original_collect, dart_history_rolling and their helpers.
CLI: python -m src.preprocessing.dart_collect {list|originals|history|...} [former options].
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import zipfile
import pandas as pd
from datetime import date
import html
import io
import xml.etree.ElementTree as ET
from collections import Counter
import os
import time
import requests
import calendar
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
import threading
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from decimal import Decimal
import sqlite3
import subprocess
import sys

from src.preprocessing.dart_extract import (
    CODE_FINGERPRINT, CollectionStop, DartCollector, EXTRACTOR_VERSION, MAPPING, MARKUP, SECTION_TITLE, _decode_document, _document_evidence, _flat, _now, _report_group, _request_key, _sha, _xbrl_facts, atomic_json, document_facts, extract_filing, is_statement_shape, load_jobs, read_events, run_lock, select_filing, write_tables)
from src.preprocessing.dart_extract import OUTPUT as EXTRACT_OUTPUT
from src.preprocessing.dart_extract import DART_FILINGS, DART_FACTS, filing_observations, preserved_dart_sources


PROFILE = Path("outputs/collection/DART_LIST_ONLY/20150101_20221231/filings.parquet")


DESTINATION = Path("outputs/collection/DART_CACHE_AUDIT")


ENDPOINTS = {"xbrl": "fnlttXbrl.xml", "document": "document.xml"}


CATEGORIES = ("기존 원본 확보", "일부 원본만 확보", "원본 미확보", "확인 불가")


MISSING = {"no_manifest_record", "failed_response", "missing_file", "deleted_on_purpose"}


# Originals deleted on purpose in an approved cleanup (path, bytes, sha256). A recorded file that is gone and
# listed here with the same hash is requested again, into the same run folder; any other vanished file stays
# a source_integrity_failure.
DELETION_LEDGERS = (Path("outputs/collection/CLEANUP/20261104T070459/deleted_raw.csv"),)


def deletion_ledger(root: Path) -> dict[str, str]:
    deleted = {}
    for ledger in DELETION_LEDGERS:
        if (root / ledger).exists():
            with (root / ledger).open(encoding="utf-8", newline="") as stream:
                deleted.update((row["path"], row["sha256"]) for row in csv.DictReader(stream))
    return deleted


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Cache:
    deleted: dict = {}  # path -> sha256 of approved deletions; set from deletion_ledger() by every builder

    def __init__(self, root: Path, manifests: list[Path], known_only: list[Path] = ()):
        """known_only: later runs whose files count as accounted for on disk, but are never reused from here."""
        self.root = root.resolve()
        self.raw = (self.root / "data/raw/opendart").resolve()
        self.deleted = deletion_ledger(self.root)
        self.index = defaultdict(list)
        self.checked = {}
        self.manifest_issues = []
        known = set()
        for manifest in known_only:
            for event in read_events(manifest):
                if event.get("event") == "raw" and isinstance(event.get("path"), str):
                    known.add((self.root / event["path"]).resolve())
        for manifest in dict.fromkeys(manifests):
            if not manifest.is_file():
                continue
            for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.manifest_issues.append({"manifest": str(manifest), "line": line_number,
                                                 "reason": "manifest_json_decode_error"})
                    continue
                if record.get("event") != "raw":
                    continue
                rel = record.get("path")
                if isinstance(rel, str):
                    path = (self.root / rel).resolve()
                    if path.is_relative_to(self.raw):
                        known.add(path)
                if record.get("endpoint") not in ENDPOINTS.values():
                    continue  # a financial-API JSON never counts as a receipt original
                receipt = record.get("params", {}).get("rcept_no")
                if not isinstance(receipt, str) or not re.fullmatch(r"\d{14}", receipt):
                    self.manifest_issues.append({"manifest": str(manifest), "line": line_number,
                                                 "reason": "original_receipt_not_identifiable"})
                    continue
                self.index[receipt, record["endpoint"]].append(record)
        manifest_keys = set(self.index)
        if (self.root / DART_FILINGS).is_file():
            for record in preserved_dart_sources(self.root):
                receipt = record['params']['rcept_no']
                key = (receipt, record['endpoint'])
                known.add((self.root / record['path']).resolve())
                if key not in manifest_keys: self.index[key].append(record)
        # Only inventory the explicitly approved OpenDART directory, never data/raw globally.
        actual = set()
        if self.raw.exists():
            for path in self.raw.rglob("*"):
                resolved = path.resolve()
                if resolved.is_relative_to(self.raw) and resolved.is_file():
                    actual.add(resolved)
        untracked = sorted(actual - known)
        self.unmanifested_files = [str(p.relative_to(self.raw)) for p in untracked]
        non_original_folders = {"registry", "financial_api", "filing_index", "list_only"}
        self.unlinked_original_files = [str(p.relative_to(self.raw)) for p in untracked
            if p.relative_to(self.raw).parts[0] not in non_original_folders and p.name != ".gitkeep"]

    def inspect(self, record: dict) -> dict:
        cache_key = json.dumps(record, sort_keys=True)
        if cache_key in self.checked:
            return self.checked[cache_key]
        result = {"path": record.get("path"), "http": record.get("http"),
                  "response_status": record.get("status"), "file_exists": None,
                  "response_success": record.get("http") == 200 and record.get("status") in {"000", "ZIP_BODY"},
                  "hash_matches": None, "state": "unknown"}
        try:
            if not isinstance(record.get("path"), str):
                raise ValueError("manifest_path_missing")
            path = (self.root / record["path"]).resolve()
            if not path.is_relative_to(self.raw):
                raise ValueError("manifest_path_outside_opendart")
            try:
                size = path.stat().st_size
                result["file_exists"] = True
            except FileNotFoundError:
                result.update(file_exists=False, state="deleted_on_purpose" if self.is_deleted(record) else "missing_file")
                return self._remember(cache_key, result)
            if not record.get("sha256") or "bytes" not in record:
                raise ValueError("manifest_hash_or_size_missing")
            result["hash_matches"] = size == record["bytes"] and sha(path) == record["sha256"]
            if not result["hash_matches"]:
                raise ValueError("file_hash_or_size_mismatch")
            if not isinstance(record.get("http"), int) or not isinstance(record.get("status"), str):
                raise ValueError("response_status_not_documented")
            if not result["response_success"]:
                result["state"] = "failed_response"
                return self._remember(cache_key, result)
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                suffix = ".xbrl" if record["endpoint"] == "fnlttXbrl.xml" else ".xml"
                if not any(name.lower().endswith(suffix) for name in names):
                    raise ValueError("expected_xml_member_missing")
            result["state"] = "verified_original"
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            result["reason"] = str(error) if isinstance(error, ValueError) else type(error).__name__
        return self._remember(cache_key, result)

    def is_deleted(self, record: dict) -> bool:
        """Listed in an approved deletion ledger with the same hash, and no longer on disk."""
        path = record.get("path")
        return (isinstance(path, str) and self.deleted.get(path) == record.get("sha256") is not None
                and not (self.root / path).exists())

    def _remember(self, key: str, result: dict) -> dict:
        self.checked[key] = result
        return result

    def original(self, receipt: str, corp: str, endpoint: str) -> dict:
        records = self.index.get((receipt, endpoint), [])
        if not records:
            ambiguous = bool(self.unlinked_original_files or self.manifest_issues)
            return {"state": "unknown_unlinked_cache" if ambiguous else "no_manifest_record",
                    "reusable": False, "file_exists": False if not ambiguous else None,
                    "response_success": False, "uncertainty": ambiguous, "evidence": []}
        evidence = []
        valid_hashes = set()
        mismatch = False
        for record in records:
            if record.get("params", {}).get("corp_code", corp) != corp:
                mismatch = True
                evidence.append({"state": "unknown", "reason": "manifest_corporation_mismatch"})
                continue
            result = self.inspect(record)
            evidence.append(result)
            if result["state"] == "verified_original":
                valid_hashes.add(record["sha256"])
        if mismatch or len(valid_hashes) > 1:
            state = "unknown_corporation_or_multiple_success_payloads"
        elif len(valid_hashes) == 1:
            state = "verified_original"
        elif any(e["state"].startswith("unknown") for e in evidence):
            state = "unknown"
        elif any(e["state"] == "missing_file" for e in evidence):
            state = "missing_file"
        elif evidence[-1]["state"] == "deleted_on_purpose":
            # The latest recorded answer was deleted by the approved cleanup (older failures don't matter);
            # once re-requested, the newer response (absence, failure) is last and decides instead.
            state = "deleted_on_purpose"
        else:
            state = "failed_response"
        exists = (True if any(e.get("file_exists") is True for e in evidence) else
                  None if any(e.get("file_exists") is None for e in evidence) else False)
        return {"state": state, "reusable": state == "verified_original",
                "file_exists": exists,
                "response_success": any(e.get("response_success") for e in evidence),
                "uncertainty": state.startswith("unknown") or any(e["state"].startswith("unknown") for e in evidence),
                "evidence": evidence}


def audit_frame(filings: pd.DataFrame, cache: Cache, legacy_corps: set[str]) -> tuple[pd.DataFrame, dict]:
    required = {"corp_code", "rcept_no", "rcept_dt", "report_nm", "report_kind", "report_period_label"}
    if not required <= set(filings):
        raise ValueError("filing_list_schema_missing")
    frame = filings[filings.report_period_label.str[:4].between("2015", "2021", inclusive="both")].copy()
    if frame[list(required)].isna().any().any() or frame.duplicated(["corp_code", "rcept_no"]).any():
        raise ValueError("candidate_keys_missing_or_duplicated")
    for column, pattern in [("corp_code", r"\d{8}"), ("rcept_no", r"\d{14}"),
                            ("rcept_dt", r"\d{8}"), ("report_period_label", r"\d{4}-(?:0[1-9]|1[0-2])")]:
        if not frame[column].str.fullmatch(pattern).all():
            raise ValueError("candidate_identifier_or_date_format_invalid")
    pd.to_datetime(frame.rcept_dt, format="%Y%m%d", errors="raise")
    frame = frame.sort_values(["corp_code", "report_kind", "report_period_label", "rcept_dt", "rcept_no"], kind="stable")
    group_cols = ["corp_code", "report_kind", "report_period_label"]
    frame["group_id"] = frame[group_cols].agg("|".join, axis=1)
    frame["revision_tags"] = frame.report_nm.map(lambda name: json.dumps(re.findall(r"\[([^\]]+)\]", name), ensure_ascii=False))
    frame["initial_candidate"] = frame.report_nm.str.fullmatch(r"(?:사업|반기|분기)보고서\s*\(\d{4}\.\d{2}\)\s*")
    frame["attachment_addition"] = frame.report_nm.str.contains(r"\[\s*첨부추가\s*\]", regex=True)
    grouped = frame.groupby("group_id", sort=False)
    frame["version_order"] = grouped.cumcount() + 1
    initial_counts = grouped.initial_candidate.transform("sum")
    frame["missing_initial"] = initial_counts.eq(0)
    frame["multiple_initial_candidates"] = initial_counts.gt(1)
    frame["same_day_order_unverified"] = frame.duplicated(["group_id", "rcept_dt"], keep=False)
    frame["receipt_link_ambiguous"] = frame.groupby("rcept_no").corp_code.transform("nunique").gt(1)
    missing_groups = []
    for group_id, rows in frame[frame.missing_initial].groupby("group_id", sort=False):
        missing_groups.append({"group_id": group_id, "corp_code": rows.iloc[0].corp_code,
            "report_kind": rows.iloc[0].report_kind, "report_period_label": rows.iloc[0].report_period_label,
            "observed_receipts": rows.rcept_no.tolist()})
    audited = []
    for row in frame.to_dict("records"):
        states = {}
        for kind, endpoint in ENDPOINTS.items():
            result = cache.original(row["rcept_no"], row["corp_code"], endpoint)
            if row["receipt_link_ambiguous"]:
                result = {**result, "state": "unknown_receipt_corporation_link", "reusable": False, "uncertainty": True}
            states[kind] = result
            row[kind + "_state"] = result["state"]
            row[kind + "_file_exists"] = result["file_exists"]
            row[kind + "_response_success"] = result["response_success"]
            row[kind + "_reusable"] = result["reusable"]
            row[kind + "_evidence"] = json.dumps(result["evidence"], ensure_ascii=False)
        reusable = sum(state["reusable"] for state in states.values())
        uncertain = any(state["uncertainty"] for state in states.values())
        row["cache_category"] = CATEGORIES[0 if reusable == 2 else 1 if reusable == 1 else 3 if uncertain else 2]
        row["cache_uncertainty"] = uncertain
        # No strategy decision: minimum-one-original gaps and two-source gaps are separate.
        row["new_collection_required_minimum"] = row["cache_category"] == CATEGORIES[2]
        row["missing_xbrl"] = states["xbrl"]["state"] in MISSING
        row["missing_document"] = states["document"]["state"] in MISSING
        row["exception"] = bool(row["missing_initial"] or row["multiple_initial_candidates"] or
                                row["same_day_order_unverified"] or uncertain)
        row["in_legacy_93"] = row["corp_code"] in legacy_corps
        audited.append(row)
    result = pd.DataFrame(audited)
    if result.empty:
        raise ValueError("no_candidates")
    receipts = lambda mask: result.loc[mask, "rcept_no"].tolist()
    summary = {"candidate_count": len(result), "candidate_corporations": result.corp_code.nunique(),
        "category_counts": {label: int(result.cache_category.eq(label).sum()) for label in CATEGORIES},
        "receipt_lists": {label: receipts(result.cache_category.eq(label)) for label in CATEGORIES},
        "any_original_reusable_count": int((result.xbrl_reusable | result.document_reusable).sum()),
        "new_collection_required_count": int(result.new_collection_required_minimum.sum()),
        "new_collection_required_receipts": receipts(result.new_collection_required_minimum),
        "minimum_requirement_definition": "No reusable original and no unresolved cache uncertainty; at least one original needed, not an API-call forecast",
        "partial_originals_need_policy_decision": receipts(result.cache_category.eq(CATEGORIES[1])),
        "missing_xbrl_receipts": receipts(result.missing_xbrl),
        "missing_document_receipts": receipts(result.missing_document),
        "missing_original_endpoint_count": int(result.missing_xbrl.sum() + result.missing_document.sum()),
        "verification_required_receipts": receipts(result.cache_uncertainty),
        "missing_initial_group_count": len(missing_groups), "missing_initial_groups": missing_groups,
        "missing_initial_receipts": receipts(result.missing_initial),
        "exception_count": int(result.exception.sum()), "exception_receipts": receipts(result.exception),
        "exception_group_count": result.loc[result.exception, "group_id"].nunique(),
        "exception_breakdown": {"missing_initial_groups": len(missing_groups),
            "multiple_initial_groups": result.loc[result.multiple_initial_candidates, "group_id"].nunique(),
            "same_day_order_unverified_receipts": int(result.same_day_order_unverified.sum()),
            "uncertain_cache_receipts": int(result.cache_uncertainty.sum()),
            "ambiguous_receipt_corporations": int(result.receipt_link_ambiguous.sum())},
        "legacy_completed_corporations": len(legacy_corps),
        "legacy_count_matches_93": len(legacy_corps) == 93,
        "legacy_candidate_count": int(result.in_legacy_93.sum()),
        "legacy_reusable_candidate_count": int((result.in_legacy_93 & (result.xbrl_reusable | result.document_reusable)).sum()),
        "verified_original_files": len({e["path"] for e in cache.checked.values() if e["state"] == "verified_original"}),
        "record_state_counts": dict(Counter(e["state"] for e in cache.checked.values())),
        "manifest_issues": cache.manifest_issues, "unmanifested_raw_files": cache.unmanifested_files,
        "unlinked_original_files": cache.unlinked_original_files,
        "scope_note": "2015-2021 report_period_label candidates, not confirmed fiscal years. ZIP/member/hash checks do not prove monetary extraction or PIT.",
        "version_order_note": "Sort by receipt date then receipt ID only; same-day order unverified. No inferred initial receipt or final correction relationship."}
    return result, json.loads(json.dumps(summary, ensure_ascii=False, default=int))


def current_candidates(root):
    """The original structured-report population, identified in common observation data."""
    metadata = filing_observations(root)
    return metadata[metadata.disclosure_format.eq('structured_supported')].copy()


def _quarter_data(root):
    """Apply the existing cover rules to current source coverage, without old audit state."""
    cache = Cache(root, [])
    frame, _ = audit_frame(current_candidates(root), cache, set())
    rows = []
    for item in frame[frame.report_kind.eq('분기')].to_dict('records'):
        receipt, corp = item['rcept_no'], item['corp_code']
        found = {kind: cache.original(receipt, corp, endpoint) for kind, endpoint in ENDPOINTS.items()}
        if found['xbrl']['reusable']: continue
        evidence = {'code': None, 'period_start': None, 'period_end': None, 'issues': []}
        records = cache.index.get((receipt, ENDPOINTS['document']), [])
        if found['document']['reusable']:
            record = next(r for r in records if cache.inspect(r)['state'] == 'verified_original')
            evidence = quarter_cover_metadata((root / record['path']).read_bytes(), corp, receipt, item['report_nm'])
        ready = bool(evidence.get('code') in {'11013', '11014'} and
                     not found['document']['uncertainty'] and not found['xbrl']['uncertainty'])
        rows.append({**item, 'job': receipt + ':fnlttXbrl.xml', 'endpoint': 'fnlttXbrl.xml',
            'action': 'request_ready' if ready else 'unresolved', 'confirmed_reprt_code': evidence.get('code'),
            'request_params_if_ready': json.dumps({'rcept_no': receipt, 'reprt_code': evidence['code']} if ready else None),
            'metadata_evidence': json.dumps(evidence, ensure_ascii=False), 'source_listing_matched': True,
            'period_start': evidence.get('period_start'), 'period_end': evidence.get('period_end'),
            'cache_evidence': json.dumps(found, ensure_ascii=False), 'source_files': json.dumps(records, ensure_ascii=False),
            'audit_exception': item['exception']})
    columns = list(frame.columns) + [c for c in ('job', 'endpoint', 'action', 'confirmed_reprt_code',
        'request_params_if_ready', 'metadata_evidence', 'source_listing_matched', 'period_start', 'period_end',
        'cache_evidence', 'source_files', 'audit_exception') if c not in frame]
    result = pd.DataFrame(rows, columns=columns)
    summary = {'by_action': {a: int(result.action.eq(a).sum()) for a in ('offline_resolved', 'request_ready', 'unresolved')},
        'confirmed_reprt_codes': dict(Counter(result.loc[result.action.eq('request_ready'), 'confirmed_reprt_code'])),
        'inputs_sha256': {p.as_posix(): sha(root / p) for p in (DART_FILINGS, DART_FACTS)},
        'basis': 'current filing observations and preserved originals; historical attempts unavailable'}
    return result, summary, cache


def quarter_selection_from_data(root):
    frame, summary, cache = _quarter_data(root)
    selected = validate_quarter_selection(frame, summary,
        expected_ready=summary['by_action']['request_ready'], expected_total=len(frame))
    digest = hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()
    return selected, cache, {'mode': 'quarter-collect', 'candidate_count': len(selected), 'target_endpoints': len(selected),
        'audited_endpoints': len(frame), 'selection_csv': 'recomputed_from_common_data', 'selection_csv_sha256': digest,
        'inputs_sha256': summary['inputs_sha256'], 'raw_root': QUARTER_RAW.as_posix(),
        'policy': 'Existing cover-confirmed XBRL request rules; current source coverage'}


def main_cache_audit() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-candidates", type=int, default=24715)
    parser.add_argument("--expected-missing-initial-groups", type=int, default=98)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--console-limit", type=int, default=20, help="0 prints all receipt IDs; JSON always contains full lists")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.console_limit < 0:
        parser.error("console-limit must be nonnegative")
    manifests = [root / "outputs/collection/DART003/manifest.jsonl"]
    manifests.extend(sorted((root / "outputs/collection/DART_LIST_ONLY").glob("*/manifest.jsonl")))
    legacy_path = root / "outputs/collection/DART003/progress.json"
    inputs = [p for p in [root / PROFILE, legacy_path, *manifests] if p.is_file()]
    if not (root / PROFILE).is_file(): inputs += [root / DART_FILINGS, root / DART_FACTS]
    initial_hashes = {str(path.relative_to(root)): sha(path) for path in inputs}
    filings = pd.read_parquet(root / PROFILE) if (root / PROFILE).is_file() else current_candidates(root)
    actual = int(filings.report_period_label.str[:4].between("2015", "2021").sum())
    if actual != args.expected_candidates:
        parser.error(f"candidate count changed: expected {args.expected_candidates}, observed {actual}")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.is_file() else {'completed_corps': []}
    cache = Cache(root, manifests)
    result, summary = audit_frame(filings, cache, set(legacy["completed_corps"]))
    summary["expected_missing_initial_groups"] = args.expected_missing_initial_groups
    summary["missing_initial_count_matches_expected"] = summary["missing_initial_group_count"] == args.expected_missing_initial_groups
    if any(sha(root / rel) != digest for rel, digest in initial_hashes.items()):
        raise RuntimeError("audit_inputs_changed_during_read")
    summary["inputs"] = initial_hashes
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%dT%H%M%S%f")
    out = args.output_dir or DESTINATION / stamp
    out = (root / out).resolve()
    if not out.is_relative_to((root / DESTINATION).resolve()):
        parser.error("output-dir must be inside outputs/collection/DART_CACHE_AUDIT")
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier audit
    with (out / "candidates.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        result.to_csv(stream, index=False)
    with (out / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"후보 {summary['candidate_count']}건")
    for label in CATEGORIES:
        values = summary["receipt_lists"][label]
        shown = values if args.console_limit == 0 else values[:args.console_limit]
        print(f"{label}: {len(values)}건 | 접수번호: {', '.join(shown) or '없음'}")
    print(f"최소 신규 원본 필요: {summary['new_collection_required_count']}건; 부분 확보는 추가수집 정책 판단 대기")
    print(f"최초본 미조회: {summary['missing_initial_group_count']}그룹 (비교 기준 {args.expected_missing_initial_groups})")
    print(f"예외: {summary['exception_count']}접수 / {summary['exception_group_count']}그룹")
    print(f"기존 {summary['legacy_completed_corporations']}개사 후보 중 캐시 재사용: {summary['legacy_reusable_candidate_count']}건")
    print("모든 분류·수집공백·예외 접수번호 전체 목록: " + str(out / "summary.json"))
    print("버전 순서·파일 존재·성공 상태·검증 근거: " + str(out / "candidates.csv"))


DATE = r"(\d{4})\s*[년./-]\s*(\d{1,2})\s*[월./-]\s*(\d{1,2})\s*일?"


YEAR_TAGS = {"DocumentFiscalYearFocus"}


START_TAGS = {"DocumentPeriodStartDate", "CurrentFiscalYearStartDate", "FiscalYearStartDate"}


END_TAGS = {"DocumentPeriodEndDate", "CurrentFiscalYearEndDate", "FiscalYearEndDate"}


CURRENT = "PeriodCoveredbytheReportofCurrentFiscalYearorQuarterorHalfYearMember"


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", text))
                  .replace("&cr;", " ").replace("&cr", " ")).strip()


def _iso(groups) -> str:
    return date(*map(int, groups)).isoformat()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xbrl_metadata(text: str, corp: str, month: str | None) -> tuple:
    """Read period metadata, not monetary facts. PeriodAxis selects current values."""
    root = ET.fromstring(text)
    contexts, current_contexts = {}, {}
    for node in root:
        if _local(node.tag) != "context":
            continue
        identity = next((n.text for n in node.iter() if _local(n.tag) == "identifier"), None)
        if identity != corp:
            continue
        members = [(n.get("dimension", "").split(":")[-1], (n.text or "").split(":")[-1])
                   for n in node.iter() if _local(n.tag) == "explicitMember"]
        period_members = [value for axis, value in members if axis == "PeriodAxis"]
        if period_members and period_members != [CURRENT]:
            continue  # prior/prior-prior metadata may share the current context duration
        dates = {_local(n.tag): n.text for n in node.iter()
                 if _local(n.tag) in {"startDate", "endDate"}}
        if period_members == [CURRENT] or (month and (dates.get("endDate") or "")[:7] == month):
            contexts[node.get("id")] = (dates, period_members == [CURRENT])
            if period_members == [CURRENT]:
                current_contexts[node.get("id")] = dates
    values = defaultdict(lambda: defaultdict(set))
    years, ordinals, fallback, problems = set(), set(), set(), set()
    for node in root:
        local, ref, value = _local(node.tag), node.get("contextRef"), (node.text or "").strip()
        if ref not in contexts:
            continue
        if local in YEAR_TAGS and re.fullmatch(r"(?:19|20)\d{2}", value):
            years.add(value)  # never derive this field from CFY..., an end date, or ordinal
        elif local in START_TAGS | END_TAGS:
            try:
                parsed = date.fromisoformat(value).isoformat()
            except ValueError:
                problems.add("invalid_xbrl_metadata_date")
                continue
            values[ref]["start" if local in START_TAGS else "end"].add(parsed)
        elif local == "DocumentOrdinalYearNumberofFiscalYear" and value.isdigit():
            if contexts[ref][1]:
                ordinals.add(value)
    explicit = set()
    for ref, pair in values.items():
        if len(pair["start"]) == len(pair["end"]) == 1:
            interval = next(iter(pair["start"])), next(iter(pair["end"]))
            if interval[0] <= interval[1] and (not month or interval[1][:7] == month):
                (explicit if contexts[ref][1] else fallback).add(interval)
    # Only if explicit current metadata is absent, inspect current duration contexts.
    if not explicit:
        for dates, _ in contexts.values():
            if dates.get("startDate") and dates.get("endDate"):
                try:
                    interval = tuple(date.fromisoformat(dates[k]).isoformat()
                                     for k in ("startDate", "endDate"))
                except ValueError:
                    continue
                if interval[0] <= interval[1] and (not month or interval[1][:7] == month):
                    fallback.add(interval)
    periods = explicit or fallback
    if len(periods) > 1:
        problems.add("current_xbrl_period_ambiguous")
        periods = set()
    return years, periods, ordinals, problems


def infer_headers(title: str, documents: list[str], instances: list[str], corp: str) -> dict:
    """Independent numeric year and accounting-period fields; no API values enter."""
    years, ranges, ordinals, problems = set(), set(), set(), set()
    group = _report_group(title)
    month = group[1] if group else None
    for text in documents:
        company = re.search(r"<COMPANY-NAME\b[^>]*\bAREGCIK=[\"']([^\"']+)", text, re.I)
        if not company or company[1] != corp:
            problems.add("body_entity_unverified")
            continue
        flat = _plain(text)
        years.update(re.findall(r"(?:사업|회계)\s*연도\s*[:：]?\s*(\d{4})\s*년도", flat))
        periods = set()
        for match in re.finditer(r"사업\s*연도\s*[:：]?\s*" + DATE +
                                 r"\s*(?:부터|~|∼|–)\s*" + DATE + r"\s*(?:까지)?", flat):
            try:
                periods.add((_iso(match.groups()[:3]), _iso(match.groups()[3:])))
            except ValueError:
                problems.add("invalid_cover_date")
        # Audit attachments use TU/AUNIT rather than a main_year_audit-report business-year label.
        pending = None
        for match in re.finditer(r"<(?:TU|TE|TD)\b([^>]*)>", text, re.I):
            attrs = match[1]
            key = re.search(r'\bAUNIT=["\'](PERIODFROM|PERIODTO)["\']', attrs, re.I)
            value = re.search(r'\bAUNITVALUE=["\'](\d{8})["\']', attrs, re.I)
            if not key or not value:
                continue
            try:
                parsed = date.fromisoformat(value[1][:4] + "-" + value[1][4:6] +
                                            "-" + value[1][6:]).isoformat()
            except ValueError:
                problems.add("invalid_attachment_period")
                continue
            if key[1].upper() == "PERIODFROM":
                pending = parsed
            elif pending:
                periods.add((pending, parsed))
                pending = None
        for interval in periods:
            if interval[0] > interval[1]:
                problems.add("reversed_cover_period")
            elif not month or interval[1][:7] == month:
                ranges.add(interval)  # April starts and shortened fiscal periods are valid
    for text in instances:
        try:
            ys, periods, terms, issues = _xbrl_metadata(text, corp, month)
            years.update(ys)
            ranges.update(periods)
            ordinals.update(terms)
            problems.update(issues)
        except ET.ParseError:
            problems.add("xbrl_xml_parse_failed")
    years.update(re.findall(r"(?<!\d)(\d{4})\s*년도?\s*(?:사업|반기|분기)보고서", title))
    if len(years) > 1:
        problems.add("numeric_year_sources_conflict")
    if len(ranges) > 1:
        problems.add("accounting_period_sources_conflict")
    year = next(iter(years)) if len(years) == 1 else None
    period = next(iter(ranges)) if len(ranges) == 1 else None
    if not year:
        problems.add("numeric_fiscal_year_name_unconfirmed")
    return {"numeric_fiscal_year": year, "period_start": period[0] if period else None,
            "period_end": period[1] if period else None,
            "ordinal_fiscal_year": next(iter(ordinals)) if len(ordinals) == 1 else None,
            "year": year, "period": period, "source_absent": False,
            "problem": ",".join(sorted(problems)) or None}


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def truth(value):
    return str(value).lower() == 'true'


def within(root, path, parent):
    path = (root / path).resolve()
    if not path.is_relative_to((root / parent).resolve()):
        raise ValueError('path_outside_approved_scope')
    return path


def original_path(root, state):
    if not state['reusable']:
        return None
    return within(root, next(e['path'] for e in state['evidence']
                            if e['state'] == 'verified_original'), 'data/raw/opendart')


def key_from_environment(root):
    value = os.environ.get('OPENDART_API_KEY')
    if not value:
        # Parse only the authorized key, without displaying any .env content.
        for line in (root / '.env').read_text(encoding='utf-8-sig').splitlines():
            match = re.fullmatch(r'\s*(?:export\s+)?OPENDART_API_KEY\s*=\s*(.*?)\s*', line)
            if match:
                raw = match[1]
                value = raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'" else raw.split(' #', 1)[0].strip()
                break
    if not value or not re.fullmatch(r'[a-zA-Z0-9]+', value):
        raise ValueError('OPENDART_API_KEY_missing_or_invalid')
    return value


OUTPUT = Path('outputs/collection/DART_ORIGINAL_COLLECT')


RAW = Path('data/raw/opendart/original_collect')


AUDIT = Path('outputs/collection/DART_CACHE_AUDIT/20261101T213601596824')


QUARTER_OUTPUT = Path('outputs/collection/DART_QUARTER_COLLECT')


QUARTER_RAW = Path('data/raw/opendart/quarter_collect')


QUARTER_AUDIT = Path('outputs/collection/DART_QUARTER_AUDIT/20261102T072330827830')


QUARTER_SOURCE = OUTPUT / '20261101T220713895061'


HISTORY_OUTPUT = Path('outputs/collection/DART_HISTORY_COLLECT')


HISTORY_RAW = Path('data/raw/opendart/history_collect')


MARKET_LIST = Path('outputs/collection/DART_LIST_ONLY/market_20000101_20211231')


CORP_LIST = Path('outputs/collection/DART_LIST_ONLY/20150101_20221231')


KIND_REVIEW = Path('outputs/collection/KIND_ORIGINAL_COLLECT/20000101_20211231/scope_reviews/20261102T113345758926')


SAMPLE_SEED = 42


# Last states a --retry-failed run may request again. Absences (013/014), rejected responses and auth stay as recorded.
RETRYABLE = {'retry_exhausted', 'endpoint_failed', 'interrupted_request_unconfirmed', 'pending', 'limited'}


SCOPE = ('2015-2021 report-period-label candidates; receipt window 20150101-20221231. '
         'Later corrections are outside this input; neither full PIT history nor as-of selection is certified.')


def quarter_cover_metadata(body, corp, receipt, title):
    """Metadata only. ACODE, calendar quarter and monetary facts never select a code."""
    evidence = {'code': None, 'period_start': None, 'period_end': None,
                'members': [], 'issues': [], 'legacy_diagnostic': None}
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith('.xml')]
        main_originals = [n for n in names if not re.search(r'_\d{5}\.xml$', n, re.I)]
        evidence['members'] = names
        if len(main_originals) != 1:
            evidence['issues'].append('document_main_member_ambiguous')
            return evidence
        name = main_originals[0]
        if Path(name).stem != receipt:
            evidence['issues'].append('member_receipt_not_confirmed')
            return evidence
        if archive.getinfo(name).file_size > 128 * 1024**2:
            evidence['issues'].append('member_size_safety_limit')
            return evidence
        text, encoding = _decode_document(archive.read(name))
    evidence.update(member=name, encoding=encoding)
    company = re.search(r'<COMPANY-NAME\b[^>]*\bAREGCIK=["\']([^"\']+)["\'][^>]*>.*?</COMPANY-NAME>', text, re.I | re.S)
    if not company or company[1] != corp:
        evidence['issues'].append('document_corporation_unverified')
        return evidence
    header = re.search(r'<DOCUMENT-NAME\b([^>]*)>(.*?)</DOCUMENT-NAME>', text, re.I | re.S)
    evidence['header_report_name'] = _plain(header[2]) if header else None
    acode = re.search(r'\bACODE=["\']([^"\']+)', header[1], re.I) if header else None
    evidence['header_acode_not_used_for_quarter'] = acode[1] if acode else None
    covers = list(re.finditer(r'<COVER\b[^>]*>.*?</COVER>', text, re.I | re.S))
    if len(covers) != 1:
        evidence['issues'].append('report_cover_missing_or_ambiguous')
        return evidence
    cover = covers[0][0]
    flat = _plain(cover)
    evidence['cover_location'] = f'{name}:chars[{covers[0].start()}:{covers[0].end()}]/COVER'
    evidence['period_attribute_evidence'] = [
        {'source_location': f'{name}:chars[{covers[0].start() + m.start()}:{covers[0].start() + m.end()}]',
         'tag': m[0]} for m in re.finditer(r'<(?:TU|TE|TD)\b[^>]*\bAUNIT=["\'](?:PERIODFROM|PERIODTO)["\'][^>]*>[^<]*', cover, re.I)]
    evidence['corporation_confirmed'] = True
    # Reuse the independent audit's AUNIT/text date reader on the actual cover,
    # excluding previous-period financial tables and correction comparison tables.
    meta = infer_headers(title, [company[0] + cover], [], corp)
    evidence.update(period_start=meta['period_start'], period_end=meta['period_end'],
                    metadata_issues=meta['problem'])
    quarter = re.search(r'분\s*기\s*보\s*고\s*서\s*\(\s*제\s*\d+\s*기\s*([13])\s*분기', flat)
    old_dates = re.search(r'사업연도\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*부터\s*\d{4}년\s*\d{1,2}월\s*\d{1,2}일까지', flat)
    if not quarter and not old_dates:
        evidence['legacy_diagnostic'] = 'legacy_cover_patterns_do_not_match_observed_format'
    else:
        evidence['legacy_diagnostic'] = 'legacy_inner_failure_not_recorded_requires_separate_diagnosis'
    if not header or re.sub(r'\s+', '', _plain(header[2])) != '분기보고서':
        evidence['issues'].append('original_report_kind_not_quarter')
        return evidence
    start, end = meta['period_start'], meta['period_end']
    if not start or not end:
        evidence['issues'].append('current_cover_period_unconfirmed')
        return evidence
    span = (int(end[:4]) - int(start[:4])) * 12 + int(end[5:7]) - int(start[5:7]) + 1
    complete_months = start[8:] == '01' and int(end[8:]) == calendar.monthrange(int(end[:4]), int(end[5:7]))[1]
    duration_code = {3: '11013', 9: '11014'}.get(span) if complete_months else None
    explicit_code = {'1': '11013', '3': '11014'}.get(quarter[1]) if quarter else None
    if explicit_code and duration_code and explicit_code != duration_code:
        evidence['issues'].append('cover_ordinal_duration_conflict')
        return evidence
    evidence['code'] = explicit_code or duration_code
    evidence['code_basis'] = 'explicit_cover_quarter_and_period' if explicit_code else 'original_cover_business_year_duration'
    evidence['cover_month_span'] = span
    if not evidence['code']:
        evidence['issues'].append('quarter_code_not_confirmed_by_cover')
    return evidence


def audit_quarter_endpoints(root, folder, output, expected=940):
    """Read-only reconstruction. Never instantiate Collector, Session or read keys."""
    controls = [folder / name for name in ('collection.json', 'progress.json', 'manifest.jsonl')]
    if not any(p.is_file() for p in controls):
        frame, summary, _ = _quarter_data(root)
        output.mkdir(parents=True, exist_ok=False)
        frame.to_csv(output / 'candidates.csv', index=False, encoding='utf-8-sig')
        summary.update(source_run_dir=str(folder.relative_to(root)), endpoints=len(frame),
                       input_hashes=summary['inputs_sha256'], current_observations=True)
        atomic_json(output / 'summary.json', summary)
        return summary
    if (folder / 'RUNNING').exists():
        raise ValueError('collector_still_marked_running')
    before = {str(p.relative_to(root)): sha(p) for p in controls}
    run, progress = [json.loads(p.read_text(encoding='utf-8')) for p in controls[:2]]
    states = {}
    def events(path):
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    yield line_number, json.loads(line)
    for _, event in events(controls[2]):
        if event['event'] == 'endpoint_state':
            states[event['job']] = event
    counts = dict(Counter(e['state'] for e in states.values()))
    if counts != progress['endpoint_states'] or len(states) != progress['target_endpoints']:
        raise ValueError('progress_manifest_state_mismatch')
    pending = [e for e in states.values() if e['state'] == 'quarter_code_unconfirmed']
    if len(pending) != expected:
        raise ValueError('unconfirmed_endpoint_count_mismatch')
    receipts = {e['rcept_no'] for e in pending}
    audit = within(root, Path(run['audit_dir']), 'outputs/collection/DART_CACHE_AUDIT')
    candidate_path = audit / 'candidates.csv'
    if sha(candidate_path) != run['audit_sha256']:
        raise ValueError('original_candidate_input_changed')
    before[str(candidate_path.relative_to(root))] = sha(candidate_path)
    frame = pd.read_csv(candidate_path, dtype=str, keep_default_na=False)
    inputs = {(r['corp_code'], r['rcept_no']): r for r in frame.to_dict('records') if r['rcept_no'] in receipts}
    filing_path = root / 'outputs/collection/DART_LIST_ONLY/20150101_20221231/filings.parquet'
    before[str(filing_path.relative_to(root))] = sha(filing_path)
    fields = ['corp_code', 'rcept_no', 'rcept_dt', 'report_nm', 'report_kind', 'report_period_label']
    listings = {(r['corp_code'], r['rcept_no']): r for r in pd.read_parquet(filing_path, columns=fields).to_dict('records') if r['rcept_no'] in receipts}
    manifests = list(dict.fromkeys([root / p for p in run['cache_manifests']] + [controls[2]]))
    # Narrow reuse of Cache.inspect/original: don't inventory or rehash unrelated raw files.
    cache = Cache.__new__(Cache)
    cache.root, cache.raw, cache.checked = root, (root / 'data/raw/opendart').resolve(), {}
    cache.deleted = deletion_ledger(root)
    cache.index, cache.unlinked_original_files, cache.manifest_issues = defaultdict(list), [], []
    histories = defaultdict(list)
    for manifest in manifests:
        before[str(manifest.relative_to(root))] = sha(manifest)
        for line_number, event in events(manifest):
            receipt = event.get('params', {}).get('rcept_no') or event.get('rcept_no')
            endpoint = event.get('endpoint')
            if receipt not in receipts or endpoint not in ENDPOINTS.values():
                continue
            source = {'source_manifest': str(manifest.relative_to(root)), 'line': line_number}
            if event['event'] == 'raw':
                record = dict(event, **source)
                cache.index[receipt, endpoint].append(record)
            if event['event'] in {'raw', 'request_started', 'transport_error'}:
                histories[receipt, endpoint].append({**source, 'event': event['event'],
                    'at': event.get('at'), 'attempt_id': event.get('attempt_id'),
                    'params': {k: v for k, v in event.get('params', {}).items() if k in {'rcept_no', 'reprt_code', 'corp_code', 'bsns_year'}},
                    'http': event.get('http'), 'status': event.get('status'), 'path': event.get('path')})
    rows, source_hashes = [], {}
    for index, state in enumerate(pending):
        receipt, corp, endpoint = state['rcept_no'], state['corp_code'], state['endpoint']
        row = inputs.get((corp, receipt), {})
        listing = listings.get((corp, receipt), {})
        linked = bool(row and listing and all(row.get(k) == listing.get(k) for k in fields))
        found = {kind: cache.original(receipt, corp, value) for kind, value in ENDPOINTS.items()}
        metadata = {'code': None, 'period_start': None, 'period_end': None, 'issues': []}
        body_statuses, response_issues = defaultdict(list), []
        for kind, value in ENDPOINTS.items():
            for record in cache.index[receipt, value]:
                checked = cache.inspect(record)
                if record.get('corp_code', corp) != corp:
                    response_issues.append('manifest_corporation_mismatch:' + kind)
                if checked.get('hash_matches') and record.get('status') in {'013', '014'}:
                    try:
                        error = ET.fromstring((root / record['path']).read_bytes())
                        status = next((n.text for n in error.iter() if n.tag.rsplit('}', 1)[-1] == 'status'), None)
                        body_statuses[kind].append(status)
                        if status != record['status']:
                            response_issues.append('error_response_status_mismatch:' + kind)
                    except ET.ParseError:
                        response_issues.append('error_response_xml_unreadable:' + kind)
        document = found['document']
        if document['reusable']:
            path = original_path(root, document)
            try:
                metadata = quarter_cover_metadata(path.read_bytes(), corp, receipt, row.get('report_nm', ''))
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                metadata['issues'] = ['metadata_read_' + type(error).__name__]
        for kind, evidence in found.items():
            for item in evidence['evidence']:
                if item.get('hash_matches'):
                    source_hashes[item['path']] = next(r['sha256'] for r in cache.index[receipt, ENDPOINTS[kind]] if r.get('path') == item['path'])
        source_files = [{k: record.get(k) for k in ('endpoint', 'path', 'sha256', 'bytes', 'http', 'status', 'source_manifest', 'line')}
                        for ep in ENDPOINTS.values() for record in cache.index[receipt, ep]]
        code = metadata['code']
        exact_absence = []
        if code:
            for record in cache.index[receipt, endpoint]:
                if record.get('params') == {'rcept_no': receipt, 'reprt_code': code} and record.get('status') in {'013', '014'}:
                    checked = cache.inspect(record)
                    if checked['hash_matches'] is True and checked['http'] == 200:
                        exact_absence.append(record['status'])
        if not linked or truth(row.get('receipt_link_ambiguous', False)):
            action, cause = 'unresolved', 'filing_metadata_join_unconfirmed'
        elif response_issues:
            action, cause = 'unresolved', 'response_body_or_corporation_evidence_conflict'
        elif found['xbrl']['reusable']:
            action, cause = 'offline_resolved', 'same_receipt_xbrl_original_verified'
        elif exact_absence:
            action, cause = 'offline_resolved', 'exact_request_absence_already_verified'
        elif found['xbrl']['uncertainty'] or document['uncertainty']:
            action, cause = 'unresolved', 'cache_integrity_or_multiple_payloads_unconfirmed'
        elif code:
            action, cause = 'request_ready', 'document_cover_confirms_request_xbrl_not_acquired'
        else:
            action = 'unresolved'
            cause = ('document_absence_014_verified_no_request_code' if not document['reusable'] and '014' in body_statuses['document'] else
                     'document_original_unavailable' if not document['reusable'] else
                     metadata['issues'][0] if metadata['issues'] else 'quarter_code_unconfirmed_by_original')
        rows.append({**{k: row.get(k, state.get(k, '')) for k in fields},
            'endpoint': endpoint, 'job': state['job'], 'last_state': state['state'],
            'group_id': state.get('group_id'), 'version_order': state.get('version_order'),
            'missing_initial': state.get('missing_initial'), 'audit_exception': state.get('audit_exception'),
            'source_listing_matched': linked, 'cause': cause, 'action': action,
            'confirmed_reprt_code': code, 'period_start': metadata['period_start'], 'period_end': metadata['period_end'],
            'original_period_confirmed': bool(metadata['period_start'] and metadata['period_end']),
            'xbrl_cache_state': found['xbrl']['state'], 'document_cache_state': document['state'],
            'other_endpoint_final_state': states.get(receipt + ':document.xml', {}).get('state'),
            'xbrl_request_attempts_in_preserved_manifests': sum(e['event'] == 'request_started' for e in histories[receipt, endpoint]),
            'request_params_if_ready': {'rcept_no': receipt, 'reprt_code': code} if action == 'request_ready' else None,
            'metadata_evidence': metadata, 'cache_evidence': found,
            'source_files': source_files,
            'verified_error_body_statuses': dict(body_statuses), 'response_evidence_issues': response_issues,
            'request_response_history': {kind: histories[receipt, ep] for kind, ep in ENDPOINTS.items()},
            'historical_inner_parser_exception': 'not_logged',
            'equity_extraction_verified': False, 'PIT_verified': False})
        if (index + 1) % 100 == 0:
            print(f'Offline quarter audit: {index + 1}/{len(pending)}')
    if len(rows) != expected or len({r['job'] for r in rows}) != expected:
        raise ValueError('audit_total_or_job_uniqueness_failed')
    source_failures = [p for p, digest in source_hashes.items() if sha(root / p) != digest]
    control_failures = [p for p, digest in before.items() if sha(root / p) != digest]
    if source_failures or control_failures:
        raise ValueError('source_changed_during_read_only_audit')
    summary = {'created_at': now(), 'source_run_dir': str(folder.relative_to(root)),
        'endpoints': len(rows), 'unique_receipts': len({r['rcept_no'] for r in rows}),
        'by_endpoint': dict(Counter(r['endpoint'] for r in rows)),
        'by_action': {action: sum(r['action'] == action for r in rows) for action in ('offline_resolved', 'request_ready', 'unresolved')},
        'by_cause': dict(Counter(r['cause'] for r in rows)),
        'by_other_endpoint_state': dict(Counter(r['other_endpoint_final_state'] for r in rows)),
        'original_period_confirmed': sum(r['original_period_confirmed'] for r in rows),
        'quarter_code_confirmed': sum(bool(r['confirmed_reprt_code']) for r in rows),
        'confirmed_reprt_codes': dict(Counter(r['confirmed_reprt_code'] for r in rows if r['confirmed_reprt_code'])),
        'xbrl_attempts_for_unconfirmed_targets': sum(r['xbrl_request_attempts_in_preserved_manifests'] for r in rows),
        'unresolved_cover_month_spans': dict(Counter(r['metadata_evidence'].get('cover_month_span') for r in rows
                                                   if r['action'] == 'unresolved' and r['original_period_confirmed'])),
        'legacy_diagnostics': dict(Counter(r['metadata_evidence'].get('legacy_diagnostic') for r in rows)),
        'originals_rehashed': len(source_hashes), 'source_integrity_unchanged': True,
        'input_hashes': before, 'network_calls': 0, 'equity_extraction_verified': False, 'PIT_verified': False,
        'action_basis': 'Endpoint coverage: document metadata does not mean an XBRL ZIP is already acquired.',
        'limitations': ['Unmanifested files are not guessed or searched.', 'Historical swallowed parser exceptions were not logged.',
                        'Header ACODE is not a first/third quarter selector.', 'No calendar-month or report-title-only request inference.',
                        'request_ready is a verified request plan, not proof the endpoint will return a ZIP.']}
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'candidates.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    counts = summary['by_action']
    text = ['# DART 분기 코드 오프라인 감사', '',
        f'- 엔드포인트 {len(rows)}개 / 고유 접수번호 {summary["unique_receipts"]}개. 합계 검증 통과.',
        '- 집계 기준은 엔드포인트 원본 확보/정확한 부재 응답이다. 본문으로 메타데이터를 확인해도 XBRL 확보로 취급하지 않는다.', '',
        '| 조치 | 엔드포인트 수 |', '| --- | ---: |',
        f'| 추가 호출 없이 엔드포인트 해결 가능 | {counts.get("offline_resolved", 0)} |',
        f'| 원문에서 요청조건 확인, XBRL 수집 필요 | {counts.get("request_ready", 0)} |',
        f'| 원본/정보 부족·불확실 | {counts.get("unresolved", 0)} |', '',
        f'원문 회계기간 확인 {summary["original_period_confirmed"]}건, 요청 코드 확인 {summary["quarter_code_confirmed"]}건.',
        f'{len(rows)}개 대상의 기존 XBRL 요청 시도 합계 {summary["xbrl_attempts_for_unconfirmed_targets"]}회. 본문으로 해결 가능한 메타데이터와 XBRL 엔드포인트의 미확보를 구분한다.',
        '자기자본 추출/PIT 검증은 수행하지 않았다. 분기 코드의 확인과 XBRL 응답 성공도 별개다.', '',
        '## 원인별 집계', '']
    text += [f'- {cause}: {n}' for cause, n in summary['by_cause'].items()]
    text += ['', '## 재수집 제안', '',
        '기존 수집기를 전체 재실행하지 말고 request_ready의 request_params_if_ready만 검토해 별도 승인 후 선택적으로 요청한다.',
        '기존 report_code/collect 로직은 변경하지 않았다. 감사 CSV의 확정 요청조건을 사용하는 선택 실행 경로는 별도 승인 후 기존 요청 함수를 재사용해 연결해야 한다.',
        '미확정 건은 보고서명·달력월·DOCUMENT-NAME ACODE로 코드를 채우지 않는다. 동일한 확정014를 무조건 재시도하지 않는다.',
        'document로 연구 목적을 충족할 수 있는지와 XBRL까지 확보할지는 별도 선택이다. 기존 document는 재다운로드하지 않는다.', '',
        '## 근거·한계', '',
        '원본 경로·해시·ZIP 멤버·COVER 위치·기간·요청 및 응답은 candidates.csv에 기록했다. 제어 파일 해시는 summary.json에 기록했다.',
        '현재 표지 메타데이터 구조에 대한 진단과 과거에 실제 발생한 내부 파서 예외는 구분한다. 과거 내부 예외는 기록되지 않아 단정할 수 없다.',
        f'요청 미확정 표지의 회계기간 길이(개월): {summary["unresolved_cover_month_spans"]}. 비표준 길이만으로 결산기 변경/신설 등 원인을 확정하지 않는다.',
        *['- ' + item for item in summary['limitations']], '']
    (output / 'SUMMARY.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({'audit_output': str(output), **summary['by_action'], 'endpoints': len(rows)}, ensure_ascii=False))
    return summary


def validate_quarter_selection(frame, summary, expected_ready=510, expected_total=940):
    """Use the reviewed params verbatim; never infer a quarter in this mode."""
    required = {'job', 'action', 'endpoint', 'rcept_no', 'corp_code', 'confirmed_reprt_code',
                'request_params_if_ready', 'metadata_evidence', 'source_listing_matched',
                'period_start', 'period_end', 'cache_evidence', 'source_files', 'audit_exception',
                'report_kind', 'report_nm', 'report_period_label', 'rcept_dt', 'group_id', 'version_order', 'missing_initial'}
    if not required <= set(frame) or len(frame) != expected_total:
        raise ValueError('quarter_selection_schema_or_total_mismatch')
    if frame.job.duplicated().any() or frame.rcept_no.duplicated().any():
        raise ValueError('quarter_selection_duplicate_endpoint_or_receipt')
    counts = {action: int(frame.action.eq(action).sum()) for action in ('offline_resolved', 'request_ready', 'unresolved')}
    if counts != summary['by_action'] or sum(counts.values()) != expected_total:
        raise ValueError('quarter_selection_summary_mismatch')
    selected = []
    for index, row in frame.iterrows():
        if row['action'] != 'request_ready':
            continue
        row = row.to_dict()
        params, evidence = json.loads(row['request_params_if_ready']), json.loads(row['metadata_evidence'])
        if (row['endpoint'] != ENDPOINTS['xbrl'] or row['job'] != row['rcept_no'] + ':' + ENDPOINTS['xbrl'] or
            not re.fullmatch(r'\d{14}', row['rcept_no']) or not re.fullmatch(r'\d{8}', row['corp_code']) or
            not isinstance(params, dict) or set(params) != {'rcept_no', 'reprt_code'} or
            params['rcept_no'] != row['rcept_no'] or params['reprt_code'] not in {'11013', '11014'} or
            params['reprt_code'] != row['confirmed_reprt_code'] or params['reprt_code'] != evidence.get('code') or
            row['report_kind'] != '분기' or not truth(row['source_listing_matched']) or not evidence.get('corporation_confirmed') or
            not evidence.get('cover_location') or not row['period_start'] or not row['period_end'] or
            row['period_start'] != evidence.get('period_start') or row['period_end'] != evidence.get('period_end')):
            raise ValueError('quarter_selection_request_not_bound_to_audited_evidence')
        # Compatibility fields consumed by the unchanged resolve/set_state methods.
        row.update(approved_params=params, audit_csv_line=int(index) + 2,
                   receipt_link_ambiguous='False', exception=row['audit_exception'],
                   xbrl_reusable=str(json.loads(row['cache_evidence'])['xbrl']['reusable']))
        selected.append(row)
    if len(selected) != expected_ready or len({r['rcept_no'] for r in selected}) != expected_ready:
        raise ValueError('quarter_selection_ready_count_mismatch')
    if dict(Counter(r['approved_params']['reprt_code'] for r in selected)) != summary['confirmed_reprt_codes']:
        raise ValueError('quarter_selection_confirmed_code_counts_mismatch')
    return selected


def appended_to(path: Path, digest: str) -> bool:
    """True when the file is the pinned version, or that version with whole lines appended after it."""
    running = hashlib.sha256()
    if running.hexdigest() == digest:
        return True
    with path.open('rb') as stream:
        for line in stream:
            running.update(line)
            if running.copy().hexdigest() == digest:
                return True
    return False


def load_quarter_selection(root, audit, source, folder):
    csv_path, summary_path = audit / 'candidates.csv', audit / 'summary.json'
    if not csv_path.is_file() or not summary_path.is_file():
        return quarter_selection_from_data(root)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('current_observations'):
        return quarter_selection_from_data(root)
    if within(root, Path(summary['source_run_dir']), OUTPUT) != source:
        raise ValueError('quarter_selection_source_run_mismatch')
    if (source / 'RUNNING').exists():
        raise ValueError('original_collector_still_marked_running')
    controls = [source / name for name in ('collection.json', 'progress.json', 'manifest.jsonl')]
    current = {str(p.relative_to(root)): sha(p) for p in [csv_path, summary_path, *controls]}
    pins, changed = dict(current), set()
    for path in controls:
        recorded = next((digest for rel, digest in summary['input_hashes'].items()
                         if (root / rel).resolve() == path), None)
        pins[str(path.relative_to(root))] = recorded
        if recorded != current[str(path.relative_to(root))]:
            changed.add(path.name)
    # The source run may only have grown by appended manifest lines (e.g. re-requesting cleanup-deleted
    # originals); progress.json is derived from that manifest. Anything else invalidates the audit.
    if changed and not (changed <= {'progress.json', 'manifest.jsonl'} and 'manifest.jsonl' in changed
                        and appended_to(controls[2], pins[str(controls[2].relative_to(root))])):
        raise ValueError('quarter_selection_original_snapshot_changed')
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    selected = validate_quarter_selection(frame, summary)
    source_run = json.loads(controls[0].read_text(encoding='utf-8'))
    manifests = [root / rel for rel in source_run['cache_manifests']] + [controls[2]]
    manifests += sorted((root / QUARTER_OUTPUT).glob('*/manifest.jsonl'))
    cache = Cache.__new__(Cache)  # Reuse source verification without inventorying unrelated raw data.
    cache.root, cache.raw, cache.checked = root, (root / 'data/raw/opendart').resolve(), {}
    cache.deleted = deletion_ledger(root)
    cache.index, cache.unlinked_original_files, cache.manifest_issues = defaultdict(list), [], []
    receipts = {r['rcept_no'] for r in selected}
    allowed = {'DART003', 'DART_LIST_ONLY', 'DART_ORIGINAL_PILOT', 'DART_ORIGINAL_COLLECT', 'DART_QUARTER_COLLECT'}
    for path in dict.fromkeys(manifests):
        path = path.resolve()
        parent = root / 'outputs/collection'
        if (not path.is_relative_to(parent) or path.relative_to(parent).parts[0] not in allowed or
            path.name != 'manifest.jsonl'):
            raise ValueError('quarter_selection_manifest_outside_approved_scope')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                if (event.get('event') == 'raw' and event.get('endpoint') == ENDPOINTS['xbrl'] and
                    event.get('params', {}).get('rcept_no') in receipts):
                    cache.index[event['params']['rcept_no'], event['endpoint']].append(event)
    # Pin and verify the exact document sources used by the approved audit. No parsing.
    for row in selected:
        sources = [r for r in json.loads(row['source_files']) if r['endpoint'] == ENDPOINTS['document'] and
                   r['http'] == 200 and r['status'] in {'ZIP_BODY', '000'}]
        if not sources:
            raise ValueError('quarter_selection_document_evidence_missing')
        for record in sources:
            # deleted_on_purpose: the approved cleanup ledger holds this exact hash, so the evidence was this file.
            if cache.inspect(record)['state'] not in {'verified_original', 'deleted_on_purpose'}:
                raise ValueError('quarter_selection_document_evidence_changed')
    if any(sha(root / rel) != digest for rel, digest in current.items()):
        raise ValueError('quarter_selection_input_changed_while_loading')
    identity = {'mode': 'quarter-collect', 'candidate_count': len(selected), 'target_endpoints': len(selected),
                'selection_csv': str(csv_path.relative_to(root)), 'selection_csv_sha256': pins[str(csv_path.relative_to(root))],
                'selection_summary_sha256': pins[str(summary_path.relative_to(root))],
                'source_run_dir': str(source.relative_to(root)), 'source_snapshot_hashes': pins,
                'raw_root': QUARTER_RAW.as_posix(), 'policy': 'Audited request_ready XBRL only; params verbatim; no document requests, inference, equity or PIT.'}
    if (folder / 'collection.json').exists():
        previous = json.loads((folder / 'collection.json').read_text(encoding='utf-8'))
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError('immutable_quarter_selection_changed')
    return selected, cache, identity


def select_history(market, listed, mapping, kind_kospi, done, sample_per_stratum):
    """Regular filings still lacking originals, broadly matched to the provided universe.

    Any single market field can be wrong for a delisted or moved firm, so one basis suffices and
    all are recorded. This selects download targets; it does not certify historical eligibility.
    """
    frame = pd.concat([market, listed], ignore_index=True).astype('string')
    frame = frame[frame.report_kind.notna() & frame.rcept_no.str.fullmatch(r'\d{14}').fillna(False)]
    frame = frame[frame.report_period_label.between('1999-12', '2021-12') & (frame.rcept_dt <= '20211231')]
    if frame.groupby('rcept_no').corp_code.nunique().gt(1).any():
        raise ValueError('receipt_linked_to_two_corporations')
    frame = frame.drop_duplicates('rcept_no')
    frame = frame[~frame.rcept_no.isin(done)].copy()
    frame['basis_panel_code'] = frame.stock_code.fillna('').isin(set(mapping.stock_id))
    frame['basis_mapped_corp'] = frame.corp_code.isin(set(mapping.corp_code.dropna()))
    frame['basis_kospi_label'] = frame.corp_cls.eq('Y').fillna(False) | frame.rm.fillna('').str.contains('유')
    frame['basis_kind_kospi'] = frame.rcept_no.isin(kind_kospi)
    bases = ['basis_panel_code', 'basis_mapped_corp', 'basis_kospi_label', 'basis_kind_kospi']
    frame = frame[frame[bases].any(axis=1)].copy()
    frame['stratum'] = frame.report_period_label.str[:4] + '|' + frame.report_kind
    frame = frame.sort_values(['report_period_label', 'corp_code', 'rcept_dt', 'rcept_no'], kind='stable')
    # Stratified sample first: a low --max-calls is a coverage probe of this same run, not a separate pilot.
    # Drawn at random inside a stratum; period order would pick only the non-December year ends.
    shuffled = frame.sample(frac=1, random_state=SAMPLE_SEED)
    frame['sample_first'] = shuffled.groupby('stratum').cumcount() < sample_per_stratum
    return frame.sort_values('sample_first', ascending=False, kind='stable').reset_index(drop=True)


def load_history_selection(root, market_dir, folder, sample_per_stratum):
    if not (market_dir / 'filings.parquet').is_file():
        market = filing_observations(root)
        mapping = pd.read_parquet(root / MAPPING)
        # Current source coverage replaces historical 'already visited' state.
        modern = set(market.loc[market.disclosure_format.eq('structured_supported'), 'rcept_no'])
        frame = select_history(market, market.iloc[:0], mapping, set(), modern, sample_per_stratum)
        identity = {'mode': 'history-collect', 'candidate_count': len(frame), 'target_endpoints': len(frame),
                    'selection_csv': str((folder / 'selection.csv').relative_to(root)),
                    'inputs_sha256': {p.as_posix(): sha(root / p) for p in (DART_FILINGS, DART_FACTS)},
                    'raw_root': HISTORY_RAW.as_posix(), 'sample_per_stratum': sample_per_stratum,
                    'policy': 'document.xml only; original history selection rules; current source coverage'}
        if (folder / 'collection.json').is_file():
            previous = json.loads((folder / 'collection.json').read_text(encoding='utf-8'))
            if any(previous.get(k) != v for k, v in identity.items()):
                raise ValueError('immutable_history_selection_changed')
            frame = pd.read_csv(folder / 'selection.csv', dtype=str, keep_default_na=False)
        records = frame.to_dict('records')
        for row in records:
            row.update(group_id=f"{row['corp_code']}|{row['report_kind']}|{row['report_period_label']}",
                       version_order='', missing_initial='False', exception='False',
                       receipt_link_ambiguous='False', document_reusable='False')
        return frame, records, Cache(root, [folder / 'manifest.jsonl']), identity
    market = pd.read_parquet(market_dir / 'filings.parquet')
    if json.loads((market_dir / 'progress.json').read_text(encoding='utf-8'))['status'] != 'complete':
        raise ValueError('market_list_incomplete')
    listed = pd.read_parquet(root / CORP_LIST / 'filings.parquet')
    mapping = pd.read_parquet(root / MAPPING)
    review = pd.read_csv(root / KIND_REVIEW / 'scope_review.csv', dtype=str, keep_default_na=False,
                         usecols=['receipt_no', 'market_group'], encoding='utf-8-sig')
    done = {e['rcept_no'] for e in read_events(root / QUARTER_SOURCE / 'manifest.jsonl') if e['event'] == 'endpoint_state'}
    frame = select_history(market, listed, mapping, set(review.receipt_no[review.market_group == 'KOSPI']),
                           done, sample_per_stratum)
    path = folder / 'selection.csv'
    if path.exists():
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)  # frozen on first execute
    identity = {'mode': 'history-collect', 'candidate_count': len(frame), 'target_endpoints': len(frame),
                'selection_csv': str(path.relative_to(root)), 'market_list_dir': str(market_dir.relative_to(root)),
                'market_filings_sha256': sha(market_dir / 'filings.parquet'), 'raw_root': HISTORY_RAW.as_posix(),
                'policy': 'document.xml only; every version kept; 013/014 recorded; no inference, equity or PIT.'}
    if (folder / 'collection.json').exists():
        previous = json.loads((folder / 'collection.json').read_text(encoding='utf-8'))
        identity['selection_csv_sha256'] = sha(path)
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError('immutable_history_selection_changed')
    records = frame.to_dict('records')
    for row in records:
        row.update(group_id=f"{row['corp_code']}|{row['report_kind']}|{row['report_period_label']}", version_order='',
                   missing_initial='False', exception='False', receipt_link_ambiguous='False', document_reusable='False')
    manifests = [root / 'outputs/collection/DART003/manifest.jsonl']
    for name in ('DART_LIST_ONLY', 'DART_ORIGINAL_PILOT', 'DART_ORIGINAL_COLLECT', 'DART_QUARTER_COLLECT', 'DART_HISTORY_COLLECT'):
        manifests.extend(sorted((root / 'outputs/collection' / name).glob('*/manifest.jsonl')))
    cache = Cache.__new__(Cache)  # Same reuse rules; no global raw inventory for rows that were never collected.
    cache.root, cache.raw, cache.checked = root, (root / 'data/raw/opendart').resolve(), {}
    cache.deleted = deletion_ledger(root)
    cache.index, cache.unlinked_original_files, cache.manifest_issues = defaultdict(list), [], []
    receipts = {r['rcept_no'] for r in records}
    for manifest in dict.fromkeys(manifests):
        if manifest.exists():
            with manifest.open(encoding='utf-8') as stream:
                for line in stream:
                    event = json.loads(line)
                    if (event.get('event') == 'raw' and event.get('endpoint') == ENDPOINTS['document'] and
                        event.get('params', {}).get('rcept_no') in receipts):
                        cache.index[event['params']['rcept_no'], event['endpoint']].append(event)
    return frame, records, cache, identity


def open_history_run(root, folder, frame, identity):
    """Freeze the selection once; every later invocation must match it."""
    if not (folder / 'collection.json').exists():
        frame.to_csv(folder / 'selection.csv', index=False, encoding='utf-8', mode='x')
        identity['selection_csv_sha256'] = sha(folder / 'selection.csv')
        with (folder / 'collection.json').open('x', encoding='utf-8') as stream:
            json.dump({**identity, 'created_at': now()}, stream, ensure_ascii=False, indent=2)
        (folder / 'manifest.jsonl').touch(exist_ok=False)


@contextmanager
def keep_awake():
    """Windows: no idle sleep while a long run holds this; lid-close and manual sleep still apply."""
    import ctypes
    kernel = getattr(getattr(ctypes, 'windll', None), 'kernel32', None)
    if kernel:
        kernel.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    try:
        yield bool(kernel)
    finally:
        if kernel:
            kernel.SetThreadExecutionState(0x80000000)


def load_inputs(root, audit, expected):
    path = audit / 'candidates.csv'
    if not path.is_file():
        cache = Cache(root, [])
        frame, _ = audit_frame(current_candidates(root), cache, set())
        if expected is not None and len(frame) != expected:
            raise ValueError('audit_schema_or_candidate_count_changed')
        cache.manifest_inputs = {}
        digest = hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()
        return frame, cache, digest
    input_digest = sha(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'corp_code', 'rcept_no', 'rcept_dt', 'report_nm', 'report_kind', 'report_period_label',
                'group_id', 'version_order', 'missing_initial', 'exception',
                'receipt_link_ambiguous', 'xbrl_reusable', 'document_reusable', 'cache_category'}
    if not required <= set(frame) or len(frame) != expected:
        raise ValueError('audit_schema_or_candidate_count_changed')
    for column, pattern in [('corp_code', r'\d{8}'), ('rcept_no', r'\d{14}'),
                            ('report_period_label', r'20(?:1[5-9]|2[01])-(?:0[1-9]|1[0-2])')]:
        if not frame[column].str.fullmatch(pattern).all():
            raise ValueError('audit_identifier_or_period_invalid')
    if frame.duplicated(['corp_code', 'rcept_no']).any():
        raise ValueError('duplicate_candidate_key')
    dates = pd.to_datetime(frame.rcept_dt, format='%Y%m%d', errors='raise')
    if not dates.between('2015-01-01', '2022-12-31').all():
        raise ValueError('receipt_outside_approved_input_window')
    frame = frame.sort_values(['corp_code', 'report_kind', 'report_period_label', 'rcept_dt', 'rcept_no'], kind='stable')
    manifests = [root / 'outputs/collection/DART003/manifest.jsonl']
    manifests = [p for p in manifests if p.is_file()]
    for name in ('DART_LIST_ONLY', 'DART_ORIGINAL_PILOT', 'DART_ORIGINAL_COLLECT'):
        manifests.extend(sorted((root / 'outputs/collection' / name).glob('*/manifest.jsonl')))
    # Quarter/history runs came after this one: their files are accounted for, not reused here.
    later = [p for name in ('DART_QUARTER_COLLECT', 'DART_HISTORY_COLLECT')
             for p in sorted((root / 'outputs/collection' / name).glob('*/manifest.jsonl'))]
    cache = Cache(root, manifests, known_only=later)
    cache.manifest_inputs = {str(p.relative_to(root)): sha(p) for p in manifests}
    if sha(path) != input_digest:
        raise ValueError('audit_input_changed_while_reading')
    return frame, cache, input_digest


def plan_summary(frame, cache):
    """Metadata-only preview; live collection verifies cached bytes before reuse."""
    count = Counter()
    for row in frame.to_dict('records'):
        for kind, endpoint in ENDPOINTS.items():
            records = cache.index.get((row['rcept_no'], endpoint), [])
            if truth(row[kind + '_reusable']) or any(r.get('http') == 200 and r.get('status') in {'ZIP_BODY', '000'} for r in records):
                count['success_metadata_to_verify'] += 1
            elif any(r.get('http') == 200 and r.get('status') in {'013', '014'} for r in records):
                count['absence_metadata_to_verify_with_exact_params'] += 1
            else:
                count['endpoint_gaps_including_unresolved_quarter_codes'] += 1
    return {'candidates': len(frame), 'endpoint_targets': 2 * len(frame),
            'audit_categories': dict(Counter(frame.cache_category)),
            'missing_initial_groups': frame.loc[frame.missing_initial.map(truth), 'group_id'].nunique(),
            'exception_receipts': int(frame.exception.map(truth).sum()),
            'exception_groups': frame.loc[frame.exception.map(truth), 'group_id'].nunique(),
            'endpoint_metadata': dict(count), 'scope': SCOPE,
            'note': 'Preview is not a hash audit, exact call forecast, extraction result, or PIT verification.'}


class Collector:
    def __init__(self, root, folder, cache, max_calls, *, raw_parent=RAW):
        self.root, self.folder, self.cache = root, folder, cache
        self.raw_parent = raw_parent
        self.max_calls, self.calls = max_calls, 0  # per invocation, not a reset of prior history
        self.stop_reason = None  # latched for this invocation; only explicit execution can resume
        self.retry_failed = False  # set only by an explicit --retry-failed run
        if max_calls < 1:
            raise ValueError('max_calls_must_be_positive')
        self.invocation = uuid.uuid4().hex
        run = folder / 'collection.json'
        config = json.loads(run.read_text(encoding='utf-8')) if run.exists() else {}
        self.target_endpoints = config.get('target_endpoints', 2 * config.get('candidate_count', 0)) or None
        self.manifest = folder / 'manifest.jsonl'
        self.jobs, self.states = defaultdict(list), {}
        self.counts, self.totals = Counter(), Counter()
        self.raw_bytes = self.purged_bytes = 0
        self.purged = set()
        self.refetch = set()  # jobs re-requested this invocation because the approved cleanup deleted their original
        for event in read_events(self.manifest):
            self._index(event)
        self.session = requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
        self.last_call = 0.0

    def _index(self, event):
        job = event.get('job')
        if job:
            self.jobs[job].append(event)
        if event['event'] == 'endpoint_state':
            if job in self.states:
                self.counts[self.states[job]['state']] -= 1
            self.states[job] = event
            self.counts[event['state']] += 1
        if event['event'] == 'request_started':
            self.totals['attempts_lifetime'] += 1
        if event['event'] == 'raw':
            self.totals['originals_acquired' if event['original_success'] else 'unsuccessful_responses'] += 1
            self.totals['response_' + event['status']] += 1
            self.raw_bytes += event.get('bytes', 0)
            if self.cache.is_deleted(event):
                self.purged_bytes += event.get('bytes', 0)  # removed by the approved cleanup: not on disk
        if event['event'] == 'original_purged':
            self.purged_bytes += event['bytes']
            self.purged.add(event['rcept_no'])
        if event['event'] == 'transport_error':
            self.totals['transport_errors'] += 1

    def log(self, event):
        event = {'at': now(), 'invocation_id': self.invocation, **event}
        with self.manifest.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self._index(event)

    def checkpoint(self, status):
        report = {'status': status, 'updated_at': now(), 'invocation_id': self.invocation,
                  'stop_reason': self.stop_reason,
                  'calls_this_invocation': self.calls, 'max_calls_this_invocation': self.max_calls,
                  'endpoint_states': dict(+self.counts), 'totals': dict(self.totals),
                  'target_endpoints': self.target_endpoints,
                  'unvisited_endpoints': max(0, self.target_endpoints - len(self.states)) if self.target_endpoints else None,
                  'all_targets_verified_or_absent': bool(self.target_endpoints and
                      sum(self.counts[s] for s in ('original_verified', 'absence_013', 'absence_014')) == self.target_endpoints),
                  'state_scope': 'Last observed per endpoint; snapshot is derived from manifest, sources rechecked when visited.',
                  'scope': SCOPE, 'checkpoint_authority': 'append-only manifest; no success without verified source bytes'}
        if getattr(self, 'selection_context', None):
            responses = [e for events in self.jobs.values() for e in events if e['event'] == 'raw']
            report.update(mode=self.selection_context.get('mode', 'quarter-collect'), scope=self.selection_context.get('policy', SCOPE), selection_csv=self.selection_context['selection_csv'],
                          selection_csv_sha256=self.selection_context['selection_csv_sha256'],
                          http_status_counts=dict(Counter(str(e['http']) for e in responses)),
                          response_status_counts=dict(Counter(e['status'] for e in responses)),
                          raw_bytes_this_run=self.raw_bytes, raw_bytes_on_disk=self.raw_bytes - self.purged_bytes)
        atomic_json(self.folder / 'progress.json', report)
        return report

    def stop(self):
        return (self.folder / 'STOP').exists()

    def wait(self, seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if self.stop():
                return False
            time.sleep(min(0.2, max(0, until - time.monotonic())))
        return not self.stop()

    def set_state(self, row, kind, state, **extra):
        job = row['rcept_no'] + ':' + ENDPOINTS[kind]
        value = {'event': 'endpoint_state', 'job': job, 'endpoint': ENDPOINTS[kind],
                 'corp_code': row['corp_code'], 'rcept_no': row['rcept_no'], 'rcept_dt': row['rcept_dt'],
                 'report_nm': row['report_nm'], 'report_period_label': row['report_period_label'],
                 'group_id': row['group_id'], 'version_order': row['version_order'],
                 'missing_initial': truth(row['missing_initial']), 'audit_exception': truth(row['exception']),
                 'state': state, **extra}
        previous = self.states.get(job)
        if previous is None or any(previous.get(k) != v for k, v in value.items()):
            self.log(value)

    def resolve(self, row, kind, params=None):
        endpoint = ENDPOINTS[kind]
        if truth(row['receipt_link_ambiguous']):
            return 'identity_unconfirmed', None
        found = self.cache.original(row['rcept_no'], row['corp_code'], endpoint)
        if found['uncertainty']:
            return 'cache_unconfirmed', None
        if found['reusable']:
            return 'original_verified', original_path(self.root, found)
        if found['state'] == 'deleted_on_purpose':
            # Approved cleanup removed it: request it again with the same parameters, into this run folder.
            self.refetch.add(row['rcept_no'] + ':' + endpoint)
            return 'needed', None
        # Previously successful files that disappeared/corrupted are integrity incidents, not new gaps.
        # Files the approved cleanup deleted are not: after a re-request, DART's new answer is what counts.
        live = [r for r in self.cache.index.get((row['rcept_no'], endpoint), []) if not self.cache.is_deleted(r)]
        refetched = len(live) < len(self.cache.index.get((row['rcept_no'], endpoint), []))
        if ((truth(row[kind + '_reusable']) and not refetched) or
                any(r.get('http') == 200 and r.get('status') in {'ZIP_BODY', '000'} for r in live)):
            return 'source_integrity_failure', None
        if params is not None:
            for record in live:
                if (record.get('http') == 200 and record.get('status') in {'013', '014'}
                        and record.get('params') == params):
                    checked = self.cache.inspect(record)
                    if checked.get('hash_matches') is True:
                        return 'absence_' + record['status'], self.root / record['path']
                    return 'absence_evidence_unconfirmed', None
        return 'needed', None

    def report_code(self, row, document):
        fixed = {'사업': '11011', '반기': '11012'}.get(row['report_kind'])
        if fixed:
            return fixed, 'report_kind'
        if row['report_kind'] != '분기':
            return None, 'report_kind_unconfirmed'
        codes = {r.get('params', {}).get('reprt_code') for r in
                 self.cache.index.get((row['rcept_no'], ENDPOINTS['xbrl']), [])}
        codes.discard(None)
        if len(codes) == 1 and next(iter(codes)) in {'11013', '11014'}:
            return next(iter(codes)), 'existing_request_manifest'
        if document:
            try:
                evidence = _document_evidence(document.read_bytes(), row['corp_code'], row['report_period_label'])
                code = evidence['reprt_code_from_cover']
                if code in {'11013', '11014'}:
                    return code, 'document_cover'
            except Exception:
                pass
        return None, 'quarter_code_unconfirmed'

    def request(self, row, kind, params, secret):
        if self.stop_reason:
            return 'limited', None, 'stopped'
        endpoint = ENDPOINTS[kind]
        job = row['rcept_no'] + ':' + endpoint
        prior = self.jobs[job]
        # Attempts up to an original the approved cleanup deleted belong to the first collection, not this one:
        # a re-request (and its resumes after 020/429) starts counting retries afresh from there.
        cut = max((i for i, e in enumerate(prior) if e['event'] == 'raw' and self.cache.is_deleted(e)), default=-1)
        prior = prior[cut + 1:]
        if self.retry_failed:
            # The user asked to rerun recorded failures: only this invocation's attempts count.
            prior = [e for e in prior if e.get('invocation_id') == self.invocation]
        started = [e for e in prior if e['event'] == 'request_started']
        finished = {e['attempt_id'] for e in prior if e['event'] in {'raw', 'transport_error', 'rejected_response'}}
        if any(e['attempt_id'] not in finished for e in started):
            return 'interrupted_request_unconfirmed', None, None
        if any(e['event'] == 'rejected_response' for e in prior):
            return 'response_rejected', None, None
        # Nonretryable failures remain recorded; never retry them just because the command resumed.
        last_raw = next((e for e in reversed(prior) if e['event'] == 'raw'), None)
        if last_raw and last_raw['disposition'] == 'failure':
            return 'endpoint_failed', None, None
        retries = sum(e['event'] in {'transport_error', 'raw'} and e.get('disposition') == 'transient' for e in prior)
        while retries < 3:
            if self.stop():
                return 'pending', None, 'stop_requested'
            if self.calls >= self.max_calls:
                return 'pending', None, 'call_cap_reached'
            if not self.wait(max(0, 0.2 - (time.monotonic() - self.last_call))):
                return 'pending', None, 'stop_requested'
            attempt_id = self.invocation + '-' + str(self.calls + 1)
            requested_at = now()
            attempt_number = sum(e['event'] == 'request_started' for e in self.jobs[job]) + 1
            self.log({'event': 'request_started', 'job': job, 'attempt_id': attempt_id,
                      'endpoint': endpoint, 'params': params, 'rcept_no': row['rcept_no'],
                      'corp_code': row['corp_code'], 'attempt': attempt_number,
                      'retry_count': attempt_number - 1, 'transient_retry_count': retries, 'requested_at': requested_at})
            self.calls += 1  # journal reservation precedes HTTP, including failures and retries
            self.last_call = time.monotonic()
            try:
                response = self.session.get('https://opendart.fss.or.kr/api/' + endpoint,
                    params={'crtfc_key': secret, **params}, timeout=(15, 60), allow_redirects=False)
            except requests.RequestException as error:
                self.log({'event': 'transport_error', 'job': job, 'attempt_id': attempt_id,
                          'disposition': 'transient', 'error_type': type(error).__name__,
                          'endpoint': endpoint, 'params': params, 'retry_count': attempt_number - 1})
                retries += 1
                if retries < 3 and not self.wait(2 ** retries):
                    return 'pending', None, 'stop_requested'
                continue
            if response.status_code == 429:
                self.stop_reason = 'http_429'
                self.log({'event': 'collection_stopped', 'status': 'stopped', 'stop_reason': self.stop_reason,
                          'job': job, 'attempt_id': attempt_id, 'http': 429,
                          'api_status': None, 'calls_this_invocation': self.calls})
            body, status, success = response.content, 'UNPARSEABLE', False
            zipped = zipfile.is_zipfile(io.BytesIO(body))
            try:
                if secret.encode() in body:
                    raise ValueError('secret_echo_rejected')
                if zipped:
                    with zipfile.ZipFile(io.BytesIO(body)) as archive:
                        if sum(i.file_size for i in archive.infolist()) > 256 * 1024**2:
                            raise ValueError('archive_too_large')
                        if any(secret.encode() in archive.read(n) for n in archive.namelist()):
                            raise ValueError('secret_echo_rejected')
                        suffix = '.xbrl' if kind == 'xbrl' else '.xml'
                        success = response.status_code == 200 and any(n.lower().endswith(suffix) for n in archive.namelist())
                    status = 'ZIP_BODY'
                else:
                    try:
                        error_data = json.loads(body)
                        status = str(error_data.get('status', 'STATUS_MISSING')) if isinstance(error_data, dict) else 'STATUS_MISSING'
                    except (ValueError, UnicodeDecodeError):
                        try:
                            root = ET.fromstring(body)
                            status = next((node.text for node in root.iter()
                                           if node.tag.rsplit('}', 1)[-1] == 'status'), None) or 'STATUS_MISSING'
                        except ET.ParseError:
                            pass
            except (ValueError, RuntimeError, zipfile.BadZipFile) as error:
                self.log({'event': 'rejected_response', 'job': job, 'attempt_id': attempt_id,
                          'error_type': type(error).__name__, 'endpoint': endpoint,
                          'params': params, 'status': 'REJECTED', 'http': response.status_code})
                if self.stop_reason:
                    return 'limited', None, 'stopped'
                return 'response_rejected', None, 'unsafe_response'
            http = response.status_code
            disposition = ('success' if success else 'limited' if http == 429 or status == '020' else
                           'auth' if http in {401, 403} or status in {'010', '011', '012'} else
                           'absence' if http == 200 and status in {'013', '014'} else
                           'transient' if http == 408 or 500 <= http <= 599 or status == '800' else 'failure')
            if disposition == 'limited' and not self.stop_reason:
                self.stop_reason = 'http_429' if http == 429 else 'opendart_request_limit_020'
                self.log({'event': 'collection_stopped', 'status': 'stopped', 'stop_reason': self.stop_reason,
                          'job': job, 'attempt_id': attempt_id, 'http': http, 'api_status': status,
                          'calls_this_invocation': self.calls})
            raw = self.root / self.raw_parent / self.folder.name
            raw.mkdir(parents=True, exist_ok=True)
            path = raw / f'{row["rcept_no"]}_{kind}_{attempt_id}{".zip" if zipped else ".xml"}'
            with path.open('xb') as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            record = {'event': 'raw', 'job': job, 'attempt_id': attempt_id,
                      'endpoint': endpoint, 'params': params, 'rcept_no': row['rcept_no'],
                      'corp_code': row['corp_code'], 'attempt': attempt_number, 'retry_count': attempt_number - 1,
                      'source': 'FSS OpenDART', 'origin': 'DART_ORIGINAL_COLLECT',
                      'http': http, 'status': status, 'original_success': success, 'disposition': disposition,
                      'path': path.relative_to(self.root).as_posix(), 'sha256': hashlib.sha256(body).hexdigest(),
                      'bytes': len(body), 'collected_at': now()}
            record['requested_at'] = requested_at
            self.log(record)
            self.cache.index[row['rcept_no'], endpoint].append(record)
            if disposition == 'success':
                return 'original_verified', path, None
            if disposition == 'absence':
                return 'absence_' + status, path, None
            if disposition == 'limited':
                return 'limited', None, 'stopped'  # no retry, cooldown sleep, or next receipt
            if disposition == 'auth':
                return 'auth', None, 'auth'
            if disposition != 'transient':
                return 'endpoint_failed', None, None
            retries += 1
            if retries < 3 and not self.wait(2 ** retries):
                return 'pending', None, 'stop_requested'
        return 'retry_exhausted', None, None

    def collect(self, rows, secret):
        self.target_endpoints = 2 * len(rows)
        status = 'traversal_finished_not_PIT_verified'
        self.log({'event': 'invocation_started', 'mode': 'collect', 'max_calls': self.max_calls})
        cooldowns = [e for e in read_events(self.manifest) if e['event'] == 'cooldown']
        try:
            if self.stop_reason:
                status = 'stopped'
                return
            if cooldowns and datetime.fromisoformat(cooldowns[-1]['until']) > datetime.now(timezone.utc):
                status = 'cooldown_active'
                return
            for index, row in enumerate(rows):
                if self.stop():
                    status = 'stop_requested'
                    break
                document = None
                for kind in ('document', 'xbrl'):
                    params = {'rcept_no': row['rcept_no']}
                    # A cached success needs no report-code inference or request.
                    state, path = self.resolve(row, kind)
                    if state == 'needed':
                        if kind == 'xbrl':
                            code, basis = self.report_code(row, document)
                            if not code:
                                self.set_state(row, kind, basis)
                                continue
                            params['reprt_code'] = code
                        state, path = self.resolve(row, kind, params)
                    halt = None
                    if state == 'needed':
                        state, path, halt = self.request(row, kind, params, secret)
                    extra = {}
                    if path:
                        extra = {'path': path.relative_to(self.root).as_posix(), 'sha256': sha(path)}
                    self.set_state(row, kind, state, **extra)
                    if kind == 'document' and state == 'original_verified':
                        document = path
                    if halt:
                        status = halt
                        break
                if status != 'traversal_finished_not_PIT_verified':
                    break
                if index % 20 == 0:
                    self.checkpoint('collecting')
        except KeyboardInterrupt:
            status = 'stopped' if self.stop_reason else 'interrupted'
        except Exception as error:
            status = 'stopped' if self.stop_reason else 'error_' + type(error).__name__
            raise
        finally:
            self.session.close()
            print(json.dumps(self.checkpoint(status), ensure_ascii=False, indent=2))

    def collect_quarter(self, rows, secret, selection):
        """One explicit audited XBRL traversal; no document requests or code inference."""
        self.target_endpoints = len(rows)
        self.selection_context = selection
        status = 'selected_traversal_finished_not_PIT_verified'
        self.log({'event': 'invocation_started', 'mode': 'quarter-collect', 'max_calls': self.max_calls,
                  'selection_csv': selection['selection_csv'], 'selection_csv_sha256': selection['selection_csv_sha256']})
        try:
            self.checkpoint('quarter_collecting')
            for index, row in enumerate(rows):
                if self.stop():
                    status = 'stop_requested'
                    break
                job, params = row['rcept_no'] + ':' + ENDPOINTS['xbrl'], row['approved_params']
                binding = {'event': 'selection_binding', 'job': job, 'rcept_no': row['rcept_no'],
                           'endpoint': ENDPOINTS['xbrl'], 'params': params,
                           'selection_csv': selection['selection_csv'],
                           'selection_csv_sha256': selection['selection_csv_sha256'],
                           'selection_csv_line': row['audit_csv_line']}
                if not any(e.get('event') == 'selection_binding' for e in self.jobs[job]):
                    self.log(binding)
                state, path = self.resolve(row, 'xbrl', params)
                halt = None
                if state == 'needed':
                    state, path, halt = self.request(row, 'xbrl', params, secret)
                    if state == 'original_verified' or state.startswith('absence_'):
                        verified, verified_path = self.resolve(row, 'xbrl', params)
                        if verified != state:
                            state, path, halt = 'source_integrity_failure', None, 'source_integrity_failure'
                        else:
                            path = verified_path
                extra = {'params': params, 'selection_csv_line': row['audit_csv_line'],
                         'selection_csv_sha256': selection['selection_csv_sha256']}
                if path:
                    extra.update(path=path.relative_to(self.root).as_posix(), sha256=sha(path))
                self.set_state(row, 'xbrl', state, **extra)
                if halt:
                    status = halt
                    break
                if index % 20 == 0:
                    self.checkpoint('quarter_collecting')
        except KeyboardInterrupt:
            status = 'stopped' if self.stop_reason else 'interrupted'
        except Exception as error:
            status = 'stopped' if self.stop_reason else 'error_' + type(error).__name__
            raise
        finally:
            self.session.close()
            print(json.dumps(self.checkpoint(status), ensure_ascii=False, indent=2))

    def collect_history(self, rows, secret, selection, max_raw_bytes, total=None, retry_failed=False):
        """Bodies only: fnlttXbrl is documented from 2015, and 014 here proves nothing about content.

        The byte limit counts files on disk, so a rolling parse-and-delete cycle frees it again.
        retry_failed visits only receipts whose last state is a failure; 013/014 and rejected responses stay.
        """
        self.target_endpoints = total or len(rows)
        self.selection_context = selection
        status = 'history_traversal_finished_not_PIT_verified'
        if retry_failed:
            self.retry_failed = True
            rows = [r for r in rows if self.states.get(r['rcept_no'] + ':' + ENDPOINTS['document'], {}).get('state') in RETRYABLE]
            status = 'retry_failed_finished'
        self.log({'event': 'invocation_started', 'mode': 'history-collect', 'max_calls': self.max_calls,
                  'selection_csv': selection['selection_csv'], 'selection_csv_sha256': selection['selection_csv_sha256'],
                  **({'retry_failed': len(rows)} if retry_failed else {})})
        try:
            self.checkpoint('history_collecting')
            for index, row in enumerate(rows):
                if self.stop():
                    status = 'stop_requested'
                    break
                if self.raw_bytes - self.purged_bytes >= max_raw_bytes:
                    status = 'storage_budget_reached'
                    break
                if shutil.disk_usage(self.root).free < 3 * 1024**3:
                    status = 'free_disk_below_3GiB'
                    break
                if row['rcept_no'] in self.purged:
                    continue  # deleted on purpose after extraction; its state stays original_verified
                params = {'rcept_no': row['rcept_no']}
                state, path = self.resolve(row, 'document', params)
                halt = None
                if state == 'needed':
                    state, path, halt = self.request(row, 'document', params, secret)
                extra = {'path': path.relative_to(self.root).as_posix(), 'sha256': sha(path)} if path else {}
                self.set_state(row, 'document', state, **extra)
                if halt:
                    status = halt
                    break
                if index % 20 == 0:
                    self.checkpoint('history_collecting')
        except KeyboardInterrupt:
            status = 'stopped' if self.stop_reason else 'interrupted'
        except Exception as error:
            status = 'stopped' if self.stop_reason else 'error_' + type(error).__name__
            raise
        finally:
            self.session.close()
            self.last_status = status
            print(json.dumps(self.checkpoint(status), ensure_ascii=False, indent=2))


def main_originals():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=AUDIT)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--mode', choices=('collect', 'quarter-audit', 'quarter-collect', 'history-collect'), default='collect')
    parser.add_argument('--market-list-dir', type=Path, default=MARKET_LIST, help='Completed market-wide list for history-collect')
    parser.add_argument('--sample-per-stratum', type=int, default=20, help='history-collect: filings per year|kind visited first')
    parser.add_argument('--from-period', help='history-collect: only labels from this YYYY-MM, e.g. 2015-01 for originals rolling never touches')
    parser.add_argument('--to-period', help='history-collect: only labels up to this YYYY-MM, e.g. 2014-12 (2015+ rows outside the panel are not needed)')
    parser.add_argument('--max-raw-gb', type=float, default=15.0, help='history-collect: stop when this run has stored this many GB')
    parser.add_argument('--retry-failed', action='store_true',
                        help='history-collect: request again only receipts whose last state is a failure (not 013/014)')
    parser.add_argument('--quarter-audit-dir', type=Path, help='Quarter-audit output directory / quarter-collect input directory')
    parser.add_argument('--source-run-dir', type=Path, default=QUARTER_SOURCE, help='Read-only original run for quarter-collect')
    parser.add_argument('--max-calls', type=int, default=1000, help='HTTP attempt cap for this invocation, including failures/retries')
    parser.add_argument('--expected-candidates', type=int, default=24715)
    parser.add_argument('--execute', action='store_true', help='Perform selected mode; extract/quarter-audit remain network-free')
    args = parser.parse_args()
    if args.max_calls < 1:
        parser.error('max-calls must be positive')
    root = Path.cwd().resolve()
    if args.mode == 'history-collect':
        market = within(root, args.market_list_dir, 'outputs/collection/DART_LIST_ONLY')
        folder = within(root, args.run_dir or HISTORY_OUTPUT / market.name, HISTORY_OUTPUT)
        frame, rows, cache, identity = load_history_selection(root, market, folder, args.sample_per_stratum)
        if args.from_period:
            rows = [r for r in rows if r['report_period_label'] >= args.from_period]
        if args.to_period:
            rows = [r for r in rows if r['report_period_label'] <= args.to_period]
        print(json.dumps({'mode': args.mode, 'selected_endpoints': len(rows), 'run_dir': str(folder),
                          'raw_dir': str(root / HISTORY_RAW / folder.name),
                          'by_stratum': frame.groupby('stratum').size().to_dict(),
                          'by_basis': {c: int(frame[c].astype(str).eq('True').sum()) for c in frame if c.startswith('basis_')},
                          'max_calls_this_invocation': args.max_calls, 'max_raw_gb': args.max_raw_gb,
                          'retry_failed': args.retry_failed, 'network_enabled': args.execute}, ensure_ascii=False, indent=2))
        if not args.execute:
            print('DRY-RUN: no key read, network or output writes.')
            return
        secret = key_from_environment(root)
        folder.mkdir(parents=True, exist_ok=True)
        with run_lock(folder), keep_awake():
            open_history_run(root, folder, frame, identity)
            collector = Collector(root, folder, cache, args.max_calls, raw_parent=HISTORY_RAW)
            try:
                collector.collect_history(rows, secret, identity, int(args.max_raw_gb * 1e9),
                                          total=len(rows), retry_failed=args.retry_failed)
            except Exception as error:
                print('Stopped safely; error type:', type(error).__name__)
                raise SystemExit(1) from None
        return
    if args.mode == 'quarter-collect':
        audit = within(root, args.quarter_audit_dir or QUARTER_AUDIT, 'outputs/collection/DART_QUARTER_AUDIT')
        source = within(root, args.source_run_dir, OUTPUT)
        folder = within(root, args.run_dir or QUARTER_OUTPUT / audit.name, QUARTER_OUTPUT)
        if folder.parent != (root / QUARTER_OUTPUT).resolve():
            parser.error('quarter-collect run-dir must be a direct child of DART_QUARTER_COLLECT')
        rows, cache, identity = load_quarter_selection(root, audit, source, folder)
        audited = identity.get('audited_endpoints')
        if audited is None: audited = len(pd.read_csv(audit / 'candidates.csv', dtype=str, keep_default_na=False))
        print(json.dumps({'mode': args.mode, 'selected_endpoints': len(rows),
                          'unique_receipts': len({r['rcept_no'] for r in rows}), 'excluded_endpoints': audited - len(rows),
                          'run_dir': str(folder), 'raw_dir': str(root / QUARTER_RAW / folder.name),
                          'selection_csv_sha256': identity['selection_csv_sha256'],
                          'request_codes': dict(Counter(r['approved_params']['reprt_code'] for r in rows)),
                          'max_calls_this_invocation': args.max_calls, 'network_enabled': args.execute}, ensure_ascii=False, indent=2))
        if not args.execute:
            print(f'DRY-RUN: no Session, key read, network or output writes; only {len(rows)} audited XBRL targets selected.')
            return
        secret = key_from_environment(root)
        folder.mkdir(parents=True, exist_ok=True)
        with run_lock(folder):
            if not (folder / 'collection.json').exists():
                if any(p.name != 'RUNNING' for p in folder.iterdir()):
                    raise ValueError('nonempty_quarter_run_without_run_metadata')
                with (folder / 'collection.json').open('x', encoding='utf-8') as stream:
                    json.dump({**identity, 'created_at': now()}, stream, ensure_ascii=False, indent=2)
                (folder / 'manifest.jsonl').touch(exist_ok=False)
            elif not (folder / 'manifest.jsonl').exists():
                raise ValueError('quarter_manifest_missing_do_not_reset')
            collector = Collector(root, folder, cache, args.max_calls, raw_parent=QUARTER_RAW)
            try:
                collector.collect_quarter(rows, secret, identity)
            except Exception as error:
                print('Stopped safely; error type:', type(error).__name__)
                raise SystemExit(1) from None
        return
    if args.mode == 'quarter-audit':
        if not args.run_dir:
            parser.error('quarter-audit requires the completed --run-dir')
        folder = within(root, args.run_dir, OUTPUT)
        output = within(root, args.quarter_audit_dir or
                        Path('outputs/collection/DART_QUARTER_AUDIT') / datetime.fromisoformat(now()).strftime('%Y%m%dT%H%M%S%f'),
                        'outputs/collection/DART_QUARTER_AUDIT')
        if not args.execute:
            print(json.dumps({'mode': 'quarter-audit', 'network_enabled': False, 'run_dir': str(folder),
                              'output_dir': str(output), 'expected_endpoints': 940, 'write_enabled': False}, ensure_ascii=False))
            return
        audit_quarter_endpoints(root, folder, output)
        return
    audit = within(root, args.audit_dir, 'outputs/collection/DART_CACHE_AUDIT')
    frame, cache, digest = load_inputs(root, audit, args.expected_candidates)
    report = plan_summary(frame, cache)
    report.update(mode=args.mode, max_calls_this_invocation=args.max_calls,
                  network_enabled=args.execute and args.mode == 'collect')
    print(json.dumps(report, ensure_ascii=False, indent=2, default=int))
    # Default: continue the run already made from this same audit (same command = same folder); else a new one.
    same_audit = [p.parent for p in sorted((root / OUTPUT).glob('*/collection.json'))
                  if json.loads(p.read_text(encoding='utf-8')).get('audit_sha256') == digest]
    if args.run_dir or same_audit:
        folder = within(root, args.run_dir, OUTPUT) if args.run_dir else same_audit[-1]
        if (folder / 'collection.json').is_file():
            run = json.loads((folder / 'collection.json').read_text(encoding='utf-8'))
            if run['audit_sha256'] != digest or Path(run['audit_dir']).resolve() != audit:
                raise ValueError('immutable_audit_input_changed')
    else:
        folder = root / OUTPUT / datetime.now().strftime('%Y%m%dT%H%M%S%f')
    print(json.dumps({'run_dir': str(folder), 'continues_existing_run': (folder / 'collection.json').exists()}))
    if not args.execute:
        print('DRY-RUN: network=0, key not read, no files written; success/absence metadata needs live file verification.')
        return
    secret = key_from_environment(root) if args.mode == 'collect' else None
    if not (folder / 'collection.json').exists():
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'collection.json').open('x', encoding='utf-8') as stream:
            json.dump({'audit_dir': str(audit), 'audit_sha256': digest, 'created_at': now(),
                       'candidate_count': len(frame), 'cache_manifests': cache.manifest_inputs,
                       'policy': 'Both endpoints independently; document first; unknown quarter code is deferred.',
                       'scope': SCOPE}, stream, ensure_ascii=False, indent=2)
        frame[frame.exception.map(truth)].to_csv(folder / 'exceptions.csv', index=False, encoding='utf-8-sig', mode='x')
        (folder / 'manifest.jsonl').touch(exist_ok=False)
    print('RUN_DIR:', folder)
    with run_lock(folder):
        collector = Collector(root, folder, cache, args.max_calls)
        try:
            collector.collect(frame.to_dict('records'), secret)
        except Exception as error:
            # requests exception strings may embed authenticated URLs.
            print('Stopped safely; error type:', type(error).__name__)
            raise SystemExit(1) from None


RUN = Path("outputs/collection/DART_LIST_ONLY")


LIST_RAW = Path("data/raw/opendart/list_only")


BASE = {"last_reprt_at": "N", "pblntf_ty": "A", "sort": "date",
        "sort_mth": "asc", "page_count": "100"}


FIELDS = ("corp_code", "rcept_no", "rcept_dt", "report_nm", "source_path", "source_sha256",
          "report_kind", "report_period_label", "bsns_year_status")


def _day(value: str):
    if len(value) != 8 or not value.isdigit():
        raise ValueError("date_requires_YYYYMMDD")
    return datetime.strptime(value, "%Y%m%d").date()


def _next(value: str, count: int = 1) -> str:
    return (_day(value) + timedelta(days=count)).strftime("%Y%m%d")


def quarter_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Calendar quarters clipped to the range: list.json allows 3 months without corp_code."""
    first, last = _day(start), _day(end)
    if first > last:
        raise CollectionStop("invalid_list_search_dates")
    windows, year, quarter = [], first.year, (first.month - 1) // 3
    while (year, quarter) <= (last.year, (last.month - 1) // 3):
        lo = datetime(year, quarter * 3 + 1, 1).date()
        hi = (datetime(year + (quarter == 3), (quarter + 1) % 4 * 3 + 1, 1) - timedelta(days=1)).date()
        windows.append((max(lo, first).strftime("%Y%m%d"), min(hi, last).strftime("%Y%m%d")))
        year, quarter = year + (quarter == 3), (quarter + 1) % 4
    return windows


def _api_key(root: Path) -> str | None:
    if (root / ".env").is_file():
        for line in (root / ".env").read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("OPENDART_API_KEY="):
                return line.strip().split("=", 1)[1].strip().strip("\"'")
    return os.environ.get("OPENDART_API_KEY")


def _subtract(intervals: list[tuple[str, str]], cover: tuple[str, str]) -> list:
    result = []
    for start, end in intervals:
        lo, hi = max(start, cover[0]), min(end, cover[1])
        if lo > hi:
            result.append((start, end))
        else:
            if start < lo:
                result.append((start, _next(lo, -1)))
            if hi < end:
                result.append((_next(hi), end))
    return result


def validate_pages(pages: dict[int, tuple[dict, dict]], *, require_complete: bool) -> list | None:
    """Validate one exact query window; never mix pages from different windows."""
    if 1 not in pages:
        return None
    first = pages[1][1]
    if first.get("status") == "013":
        if len(pages) != 1 or first.get("list") or int(first.get("total_count", 0)):
            raise CollectionStop("list_absence_response_inconsistent")
        return []
    if first.get("status") != "000":
        raise CollectionStop("list_response_not_successful")
    count, total = int(first["total_count"]), int(first["total_page"])
    expected = max(1, (count + 99) // 100)
    if total != expected or any(p < 1 or p > total for p in pages):
        raise CollectionStop("list_page_total_inconsistent")
    rows, seen, last = [], set(), None
    for page, (_, data) in sorted(pages.items()):
        if (data.get("status") != "000" or int(data["total_count"]) != count
                or int(data["total_page"]) != total or int(data.get("page_no", page)) != page
                or int(data.get("page_count", 100)) != 100):
            raise CollectionStop("list_pagination_snapshot_changed")
        part = data.get("list", [])
        if len(part) != min(100, max(0, count - (page - 1) * 100)):
            raise CollectionStop("list_page_row_count_mismatch")
        for row in part:
            # A combined title (사업ㆍ반기ㆍ분기보고서) comes back once per report kind, as identical
            # adjacent rows that may straddle a page break; any other repeat means a shifted page.
            if row["rcept_no"] in seen:
                if row != last:
                    raise CollectionStop("duplicate_receipt_across_pages")
                continue
            seen.add(row["rcept_no"])
            rows.append(row)
            last = row
    complete = set(pages) == set(range(1, total + 1))
    if require_complete and not complete:
        raise CollectionStop("list_pages_incomplete")
    return rows if complete else None


class ListCollector(DartCollector):
    """Reuse get/log/rate/retry/source hashing, with isolated state and paths.

    Do not call the parent initializer/checkpoint/run_collection: they own the
    original DART-003 Parquets and corporation-completion state.
    """

    def __init__(self, root: Path, *, start: str = "20150101", end: str = "20261031",
                 max_calls: int = 4000, offline: bool = False):
        self.root = root.resolve()
        if _day(start) < _day("20150101") or _day(start) > _day(end):
            raise CollectionStop("invalid_list_search_dates")
        self.start, self.end = start, end
        self.run = self.root / RUN / f"{start}_{end}"
        self.raw = self.root / LIST_RAW
        self.log_path, self.progress_path = self.run / "manifest.jsonl", self.run / "progress.json"
        self.lock, self.rate_lock = threading.RLock(), threading.Lock()
        self.last_request = 0.0
        self.stopped = threading.Event()
        self.network_calls = self.cache_hits = 0
        self.offline, self.max_calls = offline, max_calls
        self.cache, self.raw_records = {}, []
        self.key = "OFFLINE_LIST_ONLY_TEST" if offline else None
        self.key = self.key or _api_key(self.root)
        if not self.key:
            raise CollectionStop("OPENDART_API_KEY_missing")
        mapping_path = self.root / "data/processed/opendart/company_mapping.parquet"
        mapping = pd.read_parquet(mapping_path)
        mapped = mapping[mapping.mapping_status.eq("current_exact")].sort_values("stock_id")
        self.corps = mapped.corp_code.dropna().drop_duplicates().tolist()
        if len(self.corps) != 999 or any(len(c) != 8 or not c.isdigit() for c in self.corps):
            raise CollectionStop("expected_existing_999_corporations")
        self.criteria = {"receipt_start": start, "receipt_end": end, **BASE,
                         "target_business_years": [2015, 2021],
                         "business_year_selection": "unconfirmed_list_candidates_not_filing_year",
                         "mapping_sha256": _sha(mapping_path.read_bytes())}
        # Read-only reuse of old and other-window list responses. No other raw endpoint is opened.
        logs = [self.root / "outputs/collection/DART003/manifest.jsonl"]
        logs.extend(sorted((self.root / RUN).glob("*/manifest.jsonl")))
        for path in dict.fromkeys(logs):
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("event") != "raw" or rec.get("endpoint") != "list.json":
                    continue
                source = (self.root / rec["path"]).resolve()
                if not source.is_relative_to((self.root / "data/raw/opendart").resolve()):
                    raise CollectionStop("list_cache_path_outside_opendart")
                if _sha(source.read_bytes()) != rec["sha256"]:
                    raise CollectionStop("list_cache_hash_mismatch")
                if rec.get("key") != _request_key("list.json", rec["params"]):
                    raise CollectionStop("list_cache_request_key_mismatch")
                if rec["http"] == 200 and rec["status"] in ("000", "013"):
                    self.cache[rec["key"]] = rec
        self.raw_records = list(self.cache.values())
        self.progress = {"criteria": self.criteria, "completed_corps": [], "status": "prepared"}
        self.filings = {}
        if self.progress_path.exists():
            self.progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if self.progress["criteria"] != self.criteria:
                raise CollectionStop("list_checkpoint_criteria_changed")
            for row in pd.read_parquet(self.run / "filings.parquet").to_dict("records"):
                self.filings[row["corp_code"], row["rcept_no"]] = row
        self.run.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)

    def get(self, endpoint: str, params: dict):
        if endpoint != "list.json" or any(params.get(k) != v for k, v in BASE.items()):
            raise CollectionStop("list_only_endpoint_or_search_contract_violation")
        if (self.run / "STOP").exists():
            raise CollectionStop("list_stop_requested")
        if not self.start <= params["bgn_de"] <= params["end_de"] <= self.end:
            raise CollectionStop("list_request_outside_receipt_window")
        key = _request_key(endpoint, params)
        if key not in self.cache:
            normalized = {k: str(v) for k, v in params.items()}
            for record in list(self.cache.values()):
                if {k: str(v) for k, v in record["params"].items()} == normalized:
                    self.cache[key] = record
                    break
        if key in self.cache:
            self.log({"event": "list_cache_reused", "source": self.cache[key]})
        return super().get(endpoint, params)

    def log(self, record: dict) -> None:
        if record.get("event") == "raw":
            record["origin"] = "DART_LIST_ONLY"
        super().log(record)

    def checkpoint(self) -> None:
        frame = pd.DataFrame(list(self.filings.values()))
        if frame.empty:
            frame = pd.DataFrame(columns=FIELDS)
        temp = self.run / "filings.parquet.tmp"
        frame.astype("string").to_parquet(temp, index=False)
        temp.replace(self.run / "filings.parquet")
        self.progress.update(updated_at=_now(), network_calls_this_process=self.network_calls,
                             cache_hits_this_process=self.cache_hits)
        temp = self.progress_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.progress, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.progress_path)

    def _cached_windows(self, corp: str) -> dict:
        windows = defaultdict(dict)
        for rec in self.cache.values():
            p = rec["params"]
            if p.get("corp_code") != corp:
                continue
            contract = {k: val for k, val in p.items() if k not in {
                "corp_code", "bgn_de", "end_de", "page_no"}}
            contract = {k: str(value) for k, value in contract.items()}
            if contract != BASE:
                continue  # an A001-only or last_reprt_at=Y cache cannot cover this query
            data = json.loads((self.root / rec["path"]).read_bytes())
            windows[p["bgn_de"], p["end_de"]][int(p["page_no"])] = rec, data
        return windows

    @staticmethod
    def _rows(pages: dict, lo: str, hi: str, corp: str) -> list:
        result = []
        for rec, data in pages.values():
            for row in data.get("list", []):
                if row.get("corp_code") != corp or not rec["params"]["bgn_de"] <= row["rcept_dt"] <= rec["params"]["end_de"]:
                    raise CollectionStop("list_receipt_identity_or_date_mismatch")
                _day(row["rcept_dt"])
                if lo <= row["rcept_dt"] <= hi:
                    result.append({**row, "source_path": rec["path"], "source_sha256": rec["sha256"]})
        return result

    def _interval(self, corp: str, lo: str, hi: str) -> list:
        pages, page, total = {}, 1, 1
        while page <= total:
            self.progress.update(current_corp=corp, pending_interval=[lo, hi], pending_page=page)
            self.checkpoint()
            params = {"corp_code": corp, "bgn_de": lo, "end_de": hi, **BASE, "page_no": str(page)}
            rec, _, data = self.get("list.json", params)
            pages[page] = rec, data
            validate_pages(pages, require_complete=False)
            if data["status"] == "013":
                break
            total = int(data["total_page"])
            page += 1
        if validate_pages(pages, require_complete=True) is None:
            raise CollectionStop("list_pages_incomplete")
        return self._rows(pages, lo, hi, corp)

    def collect_company(self, corp: str) -> list:
        remaining, result = [(self.start, self.end)], []
        windows = self._cached_windows(corp)
        partial = []
        for interval, pages in sorted(windows.items(), key=lambda pair: (pair[0][0], -int(pair[0][1]))):
            if validate_pages(pages, require_complete=False) is None:
                partial.append(interval)
                continue
            used = False
            for lo, hi in remaining:
                overlap = max(lo, interval[0]), min(hi, interval[1])
                if overlap[0] <= overlap[1]:
                    result.extend(self._rows(pages, *overlap, corp))
                    used = True
            if used:
                self.cache_hits += len(pages)
                for rec, _ in pages.values():
                    self.log({"event": "list_cache_reused", "source": rec})
            remaining = _subtract(remaining, interval)
        # Contained partial windows retain their original pagination; get() fetches missing pages only.
        work = []
        for interval in partial:
            if any(lo <= interval[0] <= interval[1] <= hi for lo, hi in remaining):
                work.append(interval)
                remaining = _subtract(remaining, interval)
        for lo, hi in sorted(work + remaining):
            result.extend(self._interval(corp, lo, hi))
        unique = {}
        for row in result:
            receipt = row["rcept_no"]
            previous = unique.get(receipt)
            if previous and any(previous[k] != row[k] for k in ("corp_code", "rcept_dt", "report_nm")):
                raise CollectionStop("overlapping_list_sources_conflict")
            unique.setdefault(receipt, row)
        return list(unique.values())

    def run_lists(self, max_corps: int | None) -> dict:
        remaining = [c for c in self.corps if c not in self.progress["completed_corps"]]
        selected = remaining if max_corps is None else remaining[:max_corps]
        self.progress.update(status="running", stop_reason=None)
        self.checkpoint()
        try:
            for corp in selected:
                if (self.run / "STOP").exists():
                    raise CollectionStop("list_stop_requested")
                rows = self.collect_company(corp)
                for row in rows:
                    group = _report_group(row["report_nm"])
                    if group:
                        self.filings[corp, row["rcept_no"]] = {**row, "report_kind": group[0],
                            "report_period_label": group[1], "bsns_year_status": "unconfirmed"}
                self.progress["completed_corps"].append(corp)
                self.progress.update(current_corp=None, pending_interval=None, pending_page=None)
                self.log({"event": "list_company_complete", "corp_code": corp, "rows": len(rows),
                          "criteria": self.criteria, "at": _now()})
                self.checkpoint()
            self.progress["status"] = "complete" if len(self.progress["completed_corps"]) == 999 else "bounded_complete"
        except CollectionStop as error:
            self.progress.update(status="stopped", stop_reason=str(error))
        except Exception as error:
            import traceback
            print(traceback.format_exc().replace(self.key, "[REDACTED]"), flush=True)
            self.progress.update(status="stopped", stop_reason="unhandled_" + type(error).__name__)
        finally:
            self.checkpoint()
        return {"status": self.progress["status"], "stop_reason": self.progress["stop_reason"],
                "completed_list_corps": len(self.progress["completed_corps"]),
                "remaining_list_corps": 999 - len(self.progress["completed_corps"]),
                "network_calls": self.network_calls, "cache_hits": self.cache_hits,
                "filings": len(self.filings), "checkpoint": str(self.progress_path)}


class MarketListCollector(ListCollector):
    """Every regular filing per quarter, without corp_code.

    The 999 corporation route cannot reach filers missing from today's registry match, and
    delisted firms are the survivorship risk. Rows are kept unfiltered; universe choice is
    a later, explicit step and a market field here is not a historical KOSPI certification.
    """

    def __init__(self, root: Path, *, start: str = "20000101", end: str = "20211231",
                 max_calls: int = 4000, offline: bool = False):
        self.root = root.resolve()
        self.start, self.end = start, end
        self.windows = quarter_windows(start, end)
        self.run = self.root / RUN / f"market_{start}_{end}"
        self.raw = self.root / LIST_RAW
        self.log_path, self.progress_path = self.run / "manifest.jsonl", self.run / "progress.json"
        self.lock, self.rate_lock = threading.RLock(), threading.Lock()
        self.last_request = 0.0
        self.stopped = threading.Event()
        self.network_calls = self.cache_hits = 0
        self.offline, self.max_calls = offline, max_calls
        self.cache, self.raw_records = {}, []
        self.key = "OFFLINE_LIST_ONLY_TEST" if offline else _api_key(self.root)
        if not self.key:
            raise CollectionStop("OPENDART_API_KEY_missing")
        self.criteria = {"receipt_start": start, "receipt_end": end, **BASE, "scope": "market_wide_no_corp_code"}
        self.progress = {"criteria": self.criteria, "completed_windows": [], "status": "prepared"}
        self.filings = {}
        self.run.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)
        if self.progress_path.exists():
            self.progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if self.progress["criteria"] != self.criteria:
                raise CollectionStop("list_checkpoint_criteria_changed")
            for row in pd.read_parquet(self.run / "filings.parquet").to_dict("records"):
                self.filings[row["corp_code"], row["rcept_no"]] = row
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("event") == "raw" and rec["http"] == 200 and rec["status"] in ("000", "013"):
                    if _sha((self.root / rec["path"]).read_bytes()) != rec["sha256"]:
                        raise CollectionStop("list_cache_hash_mismatch")
                    self.cache[rec["key"]] = rec

    def collect_window(self, lo: str, hi: str) -> list:
        pages, page, total = {}, 1, 1
        while page <= total:
            params = {"bgn_de": lo, "end_de": hi, **BASE, "page_no": str(page)}
            rec, _, data = self.get("list.json", params)
            pages[page] = rec, data
            validate_pages(pages, require_complete=False)
            if data["status"] == "013":
                break
            total = int(data["total_page"])
            page += 1
        if validate_pages(pages, require_complete=True) is None:
            raise CollectionStop("list_pages_incomplete")
        rows = []
        for rec, data in pages.values():
            for row in data.get("list", []):
                if not lo <= row["rcept_dt"] <= hi:
                    raise CollectionStop("list_receipt_identity_or_date_mismatch")
                group = _report_group(row["report_nm"])
                rows.append({**row, "source_path": rec["path"], "source_sha256": rec["sha256"],
                             "report_kind": group[0] if group else None,
                             "report_period_label": group[1] if group else None,
                             "bsns_year_status": "unconfirmed"})
        return rows

    def run_lists(self, max_corps: int | None = None) -> dict:
        done = {tuple(w) for w in self.progress["completed_windows"]}
        self.progress.update(status="running", stop_reason=None)
        self.checkpoint()
        try:
            for lo, hi in self.windows:
                if (lo, hi) in done:
                    continue
                if (self.run / "STOP").exists():
                    raise CollectionStop("list_stop_requested")
                rows = self.collect_window(lo, hi)
                for row in rows:
                    self.filings[row["corp_code"], row["rcept_no"]] = row
                self.progress["completed_windows"].append([lo, hi])
                self.log({"event": "list_window_complete", "window": [lo, hi], "rows": len(rows),
                          "repeated_rows": len(rows) - len({r["rcept_no"] for r in rows}), "at": _now()})
                self.checkpoint()
            self.progress["status"] = "complete"
        except CollectionStop as error:
            self.progress.update(status="stopped", stop_reason=str(error))
        except Exception as error:
            self.progress.update(status="stopped", stop_reason="unhandled_" + type(error).__name__)
        finally:
            self.checkpoint()
        return {"status": self.progress["status"], "stop_reason": self.progress["stop_reason"],
                "completed_windows": len(self.progress["completed_windows"]), "total_windows": len(self.windows),
                "network_calls": self.network_calls, "cache_hits": self.cache_hits,
                "filings": len(self.filings), "checkpoint": str(self.progress_path)}


def main_list() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--start-date", help="Receipt date, NOT fiscal year; default 20150101, market 20000101")
    parser.add_argument("--end-date", help="Default 20261031, market 20211231 (last development base month)")
    parser.add_argument("--market-wide", action="store_true", help="No corp_code: all filers by quarter window")
    parser.add_argument("--max-corps", type=int, default=5)
    parser.add_argument("--remaining", action="store_true", help="Continue remaining list-only corporations")
    parser.add_argument("--max-calls", type=int, default=4000)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.max_corps < 1 or args.max_calls < 1:
        parser.error("corporation/call limits must be positive")
    try:
        if args.market_wide:
            collector = MarketListCollector(args.root, start=args.start_date or "20000101",
                                            end=args.end_date or "20211231",
                                            max_calls=args.max_calls, offline=args.offline)
        else:
            collector = ListCollector(args.root, start=args.start_date or "20150101",
                                      end=args.end_date or "20261031",
                                      max_calls=args.max_calls, offline=args.offline)
        print("공시 목록 전용 checkpoint: " + str(collector.progress_path), flush=True)
        report = collector.run_lists(None if args.remaining else args.max_corps)
        print(json.dumps(report, ensure_ascii=True), flush=True)
        if report["status"] == "stopped":
            raise SystemExit(2)
    except (CollectionStop, OSError, ValueError) as error:
        # Never print exception text that could contain credentials or a request URL.
        failure = {"status": "stopped", "error_type": type(error).__name__}
        if isinstance(error, CollectionStop):
            failure["reason"] = str(error)  # controlled, credential-free stop codes
        print(json.dumps(failure))
        raise SystemExit(2)


EXTRACT_DIR = EXTRACT_OUTPUT / "history_rolling"


LAST_ROLLING_PERIOD = "2014-12"


def _map(function, items: list, workers: int) -> list:
    if workers <= 1 or len(items) < 2:
        return [function(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items, chunksize=8))


def _append(path: Path, events: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


SETTLED = {"original_verified", "absence_013", "absence_014"}


def _journal(path: Path) -> dict:
    """source_key -> what the report needs; the facts lists are dropped while reading."""
    keep = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    event = json.loads(line)
                    keep[event["source_key"]] = {"statement_tables": event["audit"].get("shape_candidate_tables", 0),
                                                 "problems": event["problems"]}
    return keep


def history_report(root: Path, folder: Path, extract: Path) -> dict:
    """Morning summary: collection states, failures to retry, disk, and the extraction of what arrived. Offline."""
    events = read_events(folder / "manifest.jsonl")
    last, invocations, purged = {}, {}, set()
    for event in events:
        kind, run = event["event"], event.get("invocation_id")
        if kind == "endpoint_state":
            last[event["rcept_no"]] = event["state"]
        elif kind == "invocation_started":
            invocations[run] = {"started_at": event["at"], "max_calls": event.get("max_calls"), "requests": 0,
                                "retry_failed": event.get("retry_failed"), "stop_reason": None}
        elif kind == "request_started" and run in invocations:
            invocations[run]["requests"] += 1
            invocations[run]["last_request_at"] = event["at"]
        elif kind == "collection_stopped" and run in invocations:
            invocations[run]["stop_reason"] = event["stop_reason"]
        elif kind == "original_purged":
            purged.add(event["rcept_no"])
    frame = pd.read_csv(folder / "selection.csv", dtype=str, keep_default_na=False)
    frame["state"] = frame.rcept_no.map(last).fillna("unvisited")
    frame["year"] = frame.report_period_label.str[:4]
    failed = frame[~frame.state.isin(SETTLED | {"unvisited"})]
    absent = frame[frame.state.str.startswith("absence")]
    raw_dir = root / HISTORY_RAW / folder.name
    progress = json.loads((folder / "progress.json").read_text(encoding="utf-8")) if (folder / "progress.json").exists() else {}
    collection = {
        "status": progress.get("status"), "stop_reason": progress.get("stop_reason"), "updated_at": progress.get("updated_at"),
        "running": (folder / "RUNNING").exists(), "stop_file": (folder / "STOP").exists(),
        "invocations": list(invocations.values())[-3:],
        "states": frame.state.value_counts().to_dict(),
        "by_year": {y: g.state.value_counts().to_dict() for y, g in frame.groupby("year")},
        "absence_by_receipt_month": absent.rcept_dt.str[:6].value_counts().head(12).to_dict(),
        "failed": failed[["rcept_no", "report_period_label", "state"]].to_dict("records"),
        "raw_bytes_on_disk": sum(p.stat().st_size for p in raw_dir.iterdir()) if raw_dir.exists() else 0,
        "purged": len(purged), "free_disk_bytes": shutil.disk_usage(root).free,
        "unvisited": int(frame.state.eq("unvisited").sum()), "targets": len(frame)}
    report = {"generated_at": now(), "run_dir": folder.relative_to(root).as_posix(), "collection": collection}
    if not (extract / "equity_filings.parquet").exists():
        return report
    done = json.loads((extract / "progress.json").read_text(encoding="utf-8")) if (extract / "progress.json").exists() else {}
    journal = _journal(extract / "extractions.jsonl")
    jobs = load_jobs(root, [folder], source_runs=())[0]
    verified = [j for j in jobs if j.get("document")]
    filings = pd.read_parquet(extract / "equity_filings.parquet")
    filings = filings[filings.receipt_id.isin({j["rcept_no"] for j in verified})].assign(year=lambda f: f.report_period_label.str[:4])
    keys = {j["rcept_no"]: j["source_key"] for j in verified}
    no_table = {r for r, k in keys.items() if journal.get(k, {}).get("statement_tables", 1) == 0}
    value = filings.equity.notna()
    report["extraction"] = {
        "extract_dir": extract.relative_to(root).as_posix(), "status": done.get("status"),
        "code_matches_current": done.get("code_sha256") == CODE_FINGERPRINT,
        "originals": len(verified), "not_extracted_with_current_code": sum(j["source_key"] not in journal for j in verified),
        "by_year": {y: {"originals": len(g), "value": int(value[g.index].sum()),
                        "rate": round(float(value[g.index].mean()), 3),
                        "tiers": g.equity_tier.fillna("none").value_counts().to_dict(),
                        "no_statement_table": int(g.receipt_id.isin(no_table).sum())} for y, g in filings.groupby("year")},
        "review": {"owners_identity_failed": int(filings.owners_identity_ok.eq("False").sum()),
                   "unit_suspect": int(filings.unit_suspect.eq("True").sum()),
                   "nonpositive": int(pd.to_numeric(filings.equity, errors="coerce").le(0).sum()),
                   "same_rank_conflict": int(filings.reason.fillna("").str.startswith("same_rank").sum()),
                   "statement_date_unconfirmed": int((value & filings.fiscal_period.isna()).sum()),
                   "no_value_with_statement_table": int((~value & ~filings.receipt_id.isin(no_table)).sum()),
                   "no_statement_table": len(no_table)},
        "reasons": filings.reason.value_counts().head(10).to_dict()}
    return report


def report_markdown(report: dict) -> str:
    c = report["collection"]
    gb = lambda n: f"{n / 1e9:.2f}GB"
    lines = [f"# 과거 구간 수집 보고 {report['generated_at'][:16]}", "",
             f"- 수집 상태: {c['status']} / 정지 사유: {c['stop_reason']} / 실행 중: {c['running']} / STOP 파일: {c['stop_file']}",
             f"- 대상 {c['targets']:,} 중 미방문 {c['unvisited']:,}; 상태 {c['states']}",
             f"- 원본 용량 {gb(c['raw_bytes_on_disk'])}, 삭제 {c['purged']:,}건, 디스크 여유 {gb(c['free_disk_bytes'])}"
             + (f" — **삭제 필요**(여유 {gb(PURGE_FREE_BYTES)} 미만: --purge 실행)" if c["free_disk_bytes"] < PURGE_FREE_BYTES else ""),
             f"- 재수집 대상(실패 상태) {len(c['failed'])}건" + (f": {[r['state'] for r in c['failed']][:5]}…" if c["failed"] else ""),
             f"- 014 접수월 상위: {c['absence_by_receipt_month']}", "", "## 최근 실행", "",
             "| 시작 | 요청 | 상한 | 재수집 | 정지 사유 |", "|---|---|---|---|---|"]
    lines += [f"| {i['started_at'][:16]} | {i['requests']:,} | {i['max_calls']} | {i['retry_failed'] or ''} | {i['stop_reason'] or ''} |"
              for i in c["invocations"]]
    lines += ["", "## 라벨 연도별 수집", "", "| 연도 | 원본 | 014 | 실패 | 미방문 |", "|---|---|---|---|---|"]
    for year, states in c["by_year"].items():
        failed = sum(v for k, v in states.items() if k not in SETTLED | {"unvisited"})
        lines.append(f"| {year} | {states.get('original_verified', 0):,} | {states.get('absence_014', 0):,} | {failed} | {states.get('unvisited', 0):,} |")
    e = report.get("extraction")
    if e:
        lines += ["", f"## 추출 ({e['extract_dir']})", "",
                  f"- 상태 {e['status']} / 현재 코드와 같은 지문: {e['code_matches_current']} / 원본 {e['originals']:,} 중 현재 코드로 미추출 {e['not_extracted_with_current_code']:,}",
                  f"- 검토 대상: {e['review']}", "",
                  "| 연도 | 원본 | 값 | 확보율 | 재무제표 표 없음 | 단계 |", "|---|---|---|---|---|---|"]
        lines += [f"| {y} | {v['originals']:,} | {v['value']:,} | {v['rate']:.1%} | {v['no_statement_table']} | {v['tiers']} |"
                  for y, v in e["by_year"].items()]
    return "\n".join(lines) + "\n"


def _step(root: Path, log: Path, name: str, args: list[str]) -> int:
    """One child process, its output shown and appended to the night log line by line."""
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n=== {now()} {name}: {' '.join(args)}\n")
        out.flush()
        process = subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-u", "-m", *args], cwd=root,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            out.write(line)
            out.flush()
        code = process.wait()
        out.write(f"=== {now()} {name} exit {code}\n")
    return code


def night(root: Path, folder: Path, extract: Path, *, max_calls: int, max_raw_gb: float, workers: int,
          retry_calls: int, not_before: datetime | None, to_period: str | None = None) -> dict:
    """Unattended: wait, rerun failures, collect, extract, report. No deletion. Each step runs even if one before failed.
    KIND is not part of the chain (2026-11-04: KIND on hold, its code lives in kind_collect)."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    (folder / "logs").mkdir(exist_ok=True)
    log = folder / "logs" / f"night_{stamp}.log"
    steps = {}
    run_dir, extract_dir = folder.relative_to(root).as_posix(), extract.relative_to(root).as_posix()
    with keep_awake() as held:
        with log.open("a", encoding="utf-8") as out:
            out.write(f"=== {now()} night start; keep_awake={held}; not_before={not_before}\n")
        while not_before and datetime.now() < not_before:
            if (folder / "STOP").exists():
                return {"status": "stop_requested_while_waiting", "log": log.relative_to(root).as_posix()}
            time.sleep(min(30, max(1, (not_before - datetime.now()).total_seconds())))
        started = now()
        collect = ["src.preprocessing.dart_collect", "originals", "--mode", "history-collect", "--run-dir", run_dir,
                   "--max-raw-gb", str(max_raw_gb), "--execute"] + (["--to-period", to_period] if to_period else [])
        if retry_calls:
            steps["retry_failed"] = _step(root, log, "retry_failed", collect + ["--retry-failed", "--max-calls", str(retry_calls)])
        # The cap covers the whole night: retries already spent part of it.
        used = sum(e["event"] == "request_started" and e["at"] >= started for e in read_events(folder / "manifest.jsonl"))
        if max_calls - used >= 1 and not (folder / "STOP").exists():
            steps["collect"] = _step(root, log, "collect", collect + ["--max-calls", str(max_calls - used)])
        steps["extract"] = _step(root, log, "extract", ["src.preprocessing.dart_extract", "--source-run", run_dir,
                                                        "--run-dir", extract_dir, "--workers", str(workers), "--execute"])
        steps["report"] = _step(root, log, "report", ["src.preprocessing.dart_collect", "history", "--report", "--run-dir", run_dir,
                                                      "--extract-dir", extract_dir, "--execute"])
    return {"status": "night_finished", "exit_codes": steps, "log": log.relative_to(root).as_posix()}


PURGE_FREE_BYTES = 8 * 10**9  # user decision: delete only when free space falls below this


def _change(old: dict, new: dict) -> str:
    """Kind of difference between two extractor versions' decisions for one filing."""
    key = lambda d: (d.get("equity"), d.get("equity_tier"), d.get("fiscal_period"))
    if key(old) == key(new):
        return "same"
    before, after = old.get("equity"), new.get("equity")
    if before is None and after is not None:
        return "recovered" + (":dated" if new.get("fiscal_period") else ":undated")
    if before is not None and after is None:
        return "lost:" + str(new.get("reason"))[:30]
    if before == after and old.get("equity_tier") == new.get("equity_tier"):
        return "date_changed:" + str(new.get("fiscal_period_basis"))
    if before.lstrip("-") == after.lstrip("-"):
        return "sign_flipped_to_" + ("negative" if after.startswith("-") else "positive")
    return "tier_changed" if old.get("equity_tier") != new.get("equity_tier") else "amount_changed"


def compare_extracts(old: Path, new: Path) -> dict:
    """Version-to-version change table: after a rule change only the changed filings need a look. Offline."""
    decisions = []
    for folder in (old, new):
        latest = {}
        for event in read_events(folder / "extractions.jsonl"):
            latest[event.get("receipt_id") or event.get("kind_receipt_no")] = event["decision"]
        decisions.append(latest)
    rows = []
    for receipt in sorted(set(decisions[0]) | set(decisions[1])):
        if receipt not in decisions[1] or receipt not in decisions[0]:
            rows.append({"receipt_id": receipt, "change": "only_in_" + ("old" if receipt in decisions[0] else "new")})
            continue
        before, after = decisions[0][receipt], decisions[1][receipt]
        rows.append({"receipt_id": receipt, "change": _change(before, after),
                     **{f"{side}_{k}": d.get(k) for side, d in (("old", before), ("new", after))
                        for k in ("equity", "equity_tier", "fiscal_period", "fiscal_period_basis", "reason")}})
    frame = pd.DataFrame(rows)
    stem = f"compare_{old.name}_to_{new.name}"
    frame[frame.change.ne("same")].to_csv(new / f"{stem}.csv", index=False, encoding="utf-8-sig")
    summary = {"old": old.name, "new": new.name, "filings": len(frame), "changes": frame.change.value_counts().to_dict(),
               "csv": f"{stem}.csv"}
    atomic_json(new / f"{stem}.json", summary)
    return summary


def main_history() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-list-dir", type=Path, default=MARKET_LIST)
    parser.add_argument("--run-dir", type=Path, help="DART_HISTORY_COLLECT folder; default is named after the market list")
    parser.add_argument("--max-calls", type=int, default=2000, help="HTTP attempts for this invocation")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--report", action="store_true", help="Offline morning report; --execute also saves it under <run-dir>/reports")
    parser.add_argument("--night", action="store_true",
                        help="Unattended chain without deletion: [--retry-calls] failures, history-collect, extract, report")
    parser.add_argument("--extract-dir", type=Path, help="DART_EQUITY_EXTRACT folder for --report/--night (e.g. .../history_v9)")
    parser.add_argument("--max-raw-gb", type=float, default=5.0, help="--night: history-collect storage stop")
    parser.add_argument("--retry-calls", type=int, default=0, help="--night: calls first spent on failed receipts (part of --max-calls)")
    parser.add_argument("--not-before", type=datetime.fromisoformat, help="--night: wait until this local time, e.g. 2026-11-04T00:05")
    parser.add_argument("--to-period", help="--night: collect labels only up to this YYYY-MM (2014-12: 2015+ rows are outside the panel)")
    parser.add_argument("--compare-extract", type=Path, help="Offline: change table from this older extraction folder to --extract-dir")
    parser.add_argument("--execute", action="store_true", help="--night: run the chain; --report: also save it; default is a dry-run")
    args = parser.parse_args()
    if args.max_calls < 1:
        parser.error("max-calls must be positive")
    if not (args.report or args.night or args.compare_extract):
        parser.error("choose --report, --night or --compare-extract (collection itself: originals --mode history-collect)")
    if not args.extract_dir:
        parser.error("--report, --night and --compare-extract need --extract-dir")
    root = Path.cwd().resolve()
    market = within(root, args.market_list_dir, "outputs/collection/DART_LIST_ONLY")
    folder = within(root, args.run_dir or HISTORY_OUTPUT / market.name, HISTORY_OUTPUT)
    extract = within(root, args.extract_dir, EXTRACT_OUTPUT) if args.extract_dir else root / EXTRACT_DIR
    if args.compare_extract:
        print(json.dumps(compare_extracts(within(root, args.compare_extract, EXTRACT_OUTPUT.parent), extract), ensure_ascii=False, indent=2))
        return
    if args.report:
        report = history_report(root, folder, extract)
        text = report_markdown(report)
        print(text)
        if args.execute:
            (folder / "reports").mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            atomic_json(folder / "reports" / f"{stamp}.json", report)
            (folder / "reports" / f"{stamp}.md").write_text(text, encoding="utf-8")
            pd.DataFrame(report["collection"]["failed"], columns=["rcept_no", "report_period_label", "state"]).to_csv(
                folder / "reports" / f"{stamp}_failed.csv", index=False, encoding="utf-8-sig")
        return
    if args.night:
        plan = {"mode": "night", "run_dir": str(folder), "extract_dir": str(extract), "max_calls": args.max_calls,
                "retry_calls": args.retry_calls, "max_raw_gb": args.max_raw_gb, "workers": args.workers,
                "not_before": str(args.not_before), "to_period": args.to_period, "deletion": False, "network_enabled": args.execute}
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        if not args.execute:
            print("DRY-RUN: no waiting, network, extraction or report.")
            return
        print(json.dumps(night(root, folder, extract, max_calls=args.max_calls, max_raw_gb=args.max_raw_gb, workers=args.workers,
                               retry_calls=args.retry_calls, not_before=args.not_before, to_period=args.to_period),
                         ensure_ascii=False, indent=2))
        return


SUBCOMMANDS = {"list": main_list, "cache-audit": main_cache_audit, "originals": main_originals, "history": main_history}


def main() -> None:
    """First argument picks the former CLI; the rest are its options."""
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        raise SystemExit('usage: python -m src.preprocessing.dart_collect {' + '|'.join(SUBCOMMANDS) + '} [options]')
    command = sys.argv.pop(1)
    SUBCOMMANDS[command]()


if __name__ == "__main__":
    from src.preprocessing.dart_extract import record_cli_execution
    record_cli_execution(main,"DART_COLLECTION")
