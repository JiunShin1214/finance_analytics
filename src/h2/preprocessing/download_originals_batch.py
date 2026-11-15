"""Download archived DART document.xml filings from a verified candidate queue.

Pure download+integrity audit. NO retrospective PIT approval or model fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API = 'https://opendart.fss.or.kr/api/document.xml'
from src.h2.data_io import ROOT
REQUIRED = {'stock_code', 'corp_code', 'fiscal_year_label', 'filing_date', 'rcept_no'}


def read_queue(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={'stock_code': str, 'corp_code': str, 'rcept_no': str})
    missing = REQUIRED - set(d.columns)
    if missing:
        raise ValueError(f'큐 파일의 필수 컬럼 누락: {sorted(missing)}')
    for c in ('stock_code', 'corp_code', 'rcept_no'):
        d[c] = d[c].astype(str).str.strip()
    d['stock_code'] = d['stock_code'].str.zfill(6)
    d['corp_code'] = d['corp_code'].str.zfill(8)
    if d['rcept_no'].isna().any() or not d.rcept_no.str.fullmatch(r'\d{14}').all():
        raise ValueError('공시접수번호가 14자리 숫자가 아닌 행이 있습니다.')
    if d.rcept_no.duplicated().any():
        raise ValueError('큐 파일에 접수번호 중복이 있습니다. 먼저 중복과 회계기간을 조사하세요.')
    d['filing_date'] = pd.to_datetime(d['filing_date'], errors='raise').dt.strftime('%Y-%m-%d')
    return d.sort_values(['fiscal_year_label', 'stock_code', 'rcept_no']).reset_index(drop=True)


def check_zip(data: bytes) -> tuple[bool, int, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            if not names or not any(n.lower().endswith(('.xml', '.htm', '.html')) for n in names):
                return False, len(names), 'ZIP_HAS_NO_DOCUMENT'
            for i in z.infolist():
                if i.file_size > 300_000_000:  # guard oversized compressed entries
                    return False, len(names), 'OVERSIZED_ENTRY'
            bad = z.testzip()
            if bad is not None:
                return False, len(names), f'CRC_ERROR:{bad}'
            return True, len(names), 'VALID_ARCHIVE_NOT_FINANCIALLY_VERIFIED'
    except Exception as e:
        return False, 0, f'NOT_ZIP_OR_CORRUPTED:{type(e).__name__}'


def download(session: requests.Session, key: str, rcept: str, attempts: int, pause: float) -> tuple[bytes | None, str, int]:
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(API, params={'crtfc_key': key, 'rcept_no': rcept}, timeout=(15, 90))
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt < attempts:
                    time.sleep(pause * (2 ** (attempt - 1)) + random.random() * 0.25)
                    continue
                return None, f'HTTP_{resp.status_code}', attempt
            resp.raise_for_status()
            if len(resp.content) > 100_000_000:
                return None, 'TOO_LARGE', attempt
            if not zipfile.is_zipfile(io.BytesIO(resp.content)):
                # XML API error; never print the query string (contains API key)
                try:
                    from xml.etree import ElementTree as ET
                    root = ET.fromstring(resp.content)
                    code = root.findtext('.//status') or root.findtext('.//code') or '?'
                    msg = root.findtext('.//message') or ''
                    if code in {'020', '021'}:  # API daily limit / error policies; stop batch
                        return None, f'API_LIMIT_{code}', attempt
                    return None, f'API_RESPONSE_{code}:{msg[:110]}', attempt
                except Exception:
                    return None, 'INVALID_NON_ZIP_RESPONSE', attempt
            return resp.content, 'DOWNLOADED', attempt
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt >= attempts:
                return None, f'NETWORK_{type(e).__name__}', attempt
            time.sleep(pause * (2 ** (attempt - 1)))
        except requests.HTTPError as e:
            return None, f'HTTP_{e.response.status_code if e.response is not None else "UNKNOWN"}', attempt
    return None, 'UNKNOWN', attempts


def main() -> None:
    ap = argparse.ArgumentParser(description='DART 2018–2021 사업보고서 원문 100건 단위 다운로드')
    ap.add_argument('--queue', type=Path, default=Path('results/h2_disclosure_audit/original_filing_download_queue_2018_2021.csv'))
    ap.add_argument('--output', type=Path, default=Path('results/h2_disclosure_audit/bulk_originals_2018_2021'))
    ap.add_argument('--batch-index', type=int, default=0, help='0부터 시작, batch-size 기준. --all 지정 시 무시')
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--all', action='store_true', help='전체 큐 실행. 요청 제한에 유의')
    ap.add_argument('--max-downloads', type=int, default=None, help='테스트 시 내려받을 최대 건수')
    ap.add_argument('--retry-failed', action='store_true', help='실패 레코드 재시도 (기본은 자동 재실행 시 재시도)')
    ap.add_argument('--pause', type=float, default=0.4)
    ap.add_argument('--attempts', type=int, default=3)
    args = ap.parse_args()
    if args.batch_size < 1 or args.batch_index < 0 or args.pause < 0:
        ap.error('batch-size >=1, batch-index>=0, pause>=0이어야 합니다.')
    queue = args.queue.expanduser()
    if not queue.is_file():
        fallback = ROOT / queue
        if fallback.is_file():
            queue = fallback
        else:
            raise FileNotFoundError(f'큐 CSV 없음: {queue.resolve()} (패키지의 CSV를 지정하거나 복사하세요)')
    out = args.output.expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    orig = out / 'originals'
    orig.mkdir(parents=True, exist_ok=True)
    key = os.getenv('DART_API_KEY', '')
    if not key:
        raise RuntimeError('DART_API_KEY 환경변수가 없습니다. 사용자 PC의 현재 PowerShell에 설정하세요.')
    d = read_queue(queue)
    if args.all:
        batch = d
    else:
        batch = d.iloc[args.batch_index * args.batch_size : (args.batch_index + 1) * args.batch_size]
    if batch.empty:
        print('선택한 배치에 대상이 없습니다. total=', len(d))
        return
    if args.max_downloads is not None:
        batch = batch.head(args.max_downloads)
    # Atomic per-batch manifest to avoid truncation of previous batches
    manifest_name = 'manifest_all.csv' if args.all else f'manifest_batch_{args.batch_index:03d}.csv'
    manifest_file = out / manifest_name
    previous = {}
    if manifest_file.exists():
        old = pd.read_csv(manifest_file, dtype={'rcept_no': str})
        previous = {str(x['rcept_no']): dict(x) for x in old.to_dict('records')}
    results = dict(previous)
    session = requests.Session()
    stop_on_limit = False
    for n, (_, r) in enumerate(batch.iterrows(), 1):
        rcept = str(r['rcept_no'])
        filename = orig / f'{rcept}.zip'
        state = 'UNKNOWN'; attempts = 0
        if filename.exists():
            existing = filename.read_bytes()
            ok, n_files, info = check_zip(existing)
            if ok:
                content = existing
                state = 'CACHED_VALID_ARCHIVE_NOT_FINANCIALLY_VERIFIED'
            else:
                filename.rename(filename.with_suffix('.corrupt'))
                content = None
        else:
            content = None
        if content is None:
            content, state, attempts = download(session, key, rcept, args.attempts, args.pause)
            if content is not None:
                ok, n_files, info = check_zip(content)
                if ok:
                    temp = filename.with_suffix('.tmp')
                    temp.write_bytes(content)
                    temp.replace(filename)
                    state = 'DOWNLOADED_VALID_ARCHIVE_NOT_FINANCIALLY_VERIFIED'
                else:
                    state = info
                    content = None
        if content is not None:
            checksum = hashlib.sha256(content).hexdigest()
            byte_size = len(content)
        else:
            checksum = ''; byte_size = 0; n_files = 0
        entry = {k: ('' if pd.isna(v) else v) for k, v in r.to_dict().items()}
        entry.update({'download_status': state, 'zip_sha256': checksum, 'zip_bytes': byte_size,
                      'zip_member_count': n_files, 'request_attempts': attempts,
                      'fiscal_period_verified_from_original': False,
                      'cfs_current_liability_equity_verified': False,
                      'historical_asof_verified': False,
                      'checked_utc': datetime.now(timezone.utc).isoformat()})
        results[rcept] = entry
        temp_manifest = manifest_file.with_suffix('.tmp')
        pd.DataFrame(list(results.values())).to_csv(temp_manifest, index=False, encoding='utf-8-sig')
        temp_manifest.replace(manifest_file)
        print(f'[{n}/{len(batch)}] {rcept} {state} ({byte_size} bytes)', flush=True)
        if state.startswith('API_LIMIT_'):
            print('API 호출 제한 감지: 배치 중단. 다음날 재실행하세요.', file=sys.stderr)
            stop_on_limit = True
            break
        if n < len(batch) and content is not None and attempts > 0:
            time.sleep(args.pause)
    all_manifests = sorted(out.glob('manifest_batch_*.csv'))
    if not args.all:
        manifest_frames = [pd.read_csv(p, dtype={'rcept_no': str}) for p in all_manifests]
        if manifest_frames:
            combined = pd.concat(manifest_frames, ignore_index=True)
            combined = combined.drop_duplicates('rcept_no', keep='last')
            combined.to_csv(out / 'manifest_combined.csv', index=False, encoding='utf-8-sig')
    else:
        combined = pd.read_csv(manifest_file, dtype={'rcept_no': str})
    summary = {
        'queue_rows': len(d), 'batch_rows_requested': len(batch),
        'manifest_records': len(combined) if 'combined' in locals() else len(results),
        'valid_archive_records': int(combined.download_status.str.contains('VALID_ARCHIVE', na=False).sum()) if 'combined' in locals() else 0,
        'api_limit_stop': stop_on_limit,
        'original_filing_CFS_period_verified': 0,
        'point_in_time_verified': 0,
        'note': 'ZIP integrity only; receipt no./date alone do not prove original as-filed CFS financials. No PIT model use.',
    }
    (out / 'download_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
