"""Read-only audit of historical DataGuide files and financial-account availability.

Run at repository root:
    python -m src.h2.preprocessing.audit_legacy_financial_sources
    python -m src.h2.preprocessing.audit_legacy_financial_sources --out outputs/h2_legacy_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOKENS = {
    'current_liabilities': ['유동부채', 'currentliabil', 'shorttermliabil'],
    'total_liabilities': ['부채총계', 'totalliabil', 'liabilities'],
    'operating_income': ['영업이익', 'operatingincome', 'operatingprofit'],
    'interest_expense': ['이자비용', 'interestexpense', 'interestpaid'],
    'finance_costs': ['금융비용', '금융원가', 'financecost'],
    'short_term_borrowings': ['단기차입금', 'shorttermborrow', 'shorttermdebt'],
}
SUFFIXES = {'.csv', '.xlsx', '.xlsm', '.parquet'}
PREFERRED = ('dataguide','financial','finance','account','재무','유동','부채','이자','영업','차입','1999','2009','2010','2021')
MAX_SAMPLE_ROWS = 18


def concepts(items):
    text = ' '.join(str(v) for v in items if v is not None).lower().replace('_','').replace(' ','')
    return sorted(k for k, patterns in TOKENS.items() if any(x.lower().replace('_','').replace(' ','') in text for x in patterns))


def audit_csv(path):
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
        rd = csv.reader(f)
        head = next(rd, [])
        examples = [r[:min(12, len(r))] for _, r in zip(range(10), rd)]
    sample_text = [str(v) for row in examples for v in row]
    return head[:120], concepts(head + sample_text), examples[:2], None


def audit_parquet(path):
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        head = list(pf.schema_arrow.names)
        hits = concepts(head)
        examples = []
        if not hits and any(k in ' '.join(head).lower() for k in ['account','항목','계정']):
            cols = [s for s in head if any(t in s.lower() for t in ['account','항목','계정'])][:2]
            if cols:
                batch = next(pf.iter_batches(batch_size=16, columns=cols), None)
                if batch is not None:
                    examples = batch.to_pydict()
                    hits = concepts(str(examples))
        return head[:120], hits, examples, pf.metadata.num_rows
    except ImportError:
        return [], [], 'pyarrow_not_installed', None


def audit_xlsx(path):
    try:
        from openpyxl import load_workbook
    except ImportError:
        return [], [], 'openpyxl_not_installed', None
    wb = load_workbook(path, read_only=True, data_only=True)
    head = []
    preview = []
    for sheet in wb.worksheets[:3]:
        sample = []
        for i, row in enumerate(sheet.iter_rows(values_only=True)):
            if i >= MAX_SAMPLE_ROWS:
                break
            vals = [str(x).strip() for x in row if x is not None]
            sample.extend(vals[:80])
        head.append(sheet.title)
        preview.extend(sample)
    wb.close()
    return head, concepts(preview), preview[:15], None


def audit_file(path):
    if path.suffix.lower() == '.csv': return audit_csv(path)
    if path.suffix.lower() == '.parquet': return audit_parquet(path)
    return audit_xlsx(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='outputs/h2_legacy_audit')
    ap.add_argument('--max-files', type=int, default=1000)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    data = root / 'data'
    if not data.is_dir(): raise SystemExit(f'No data folder: {data}')
    files = [p for p in data.rglob('*') if p.is_file() and p.suffix.lower() in SUFFIXES]
    files.sort(key=lambda p: (not any(s in str(p).lower() for s in PREFERRED), str(p)))
    files = files[:args.max_files]
    rows = []
    for i, path in enumerate(files, 1):
        try:
            columns, matches, example, nrows = audit_file(path)
            problem = ''
        except Exception as e:
            columns, matches, example, nrows, problem = [], [], [], None, f'{type(e).__name__}: {str(e)[:220]}'
        row = {
            'file': str(path.relative_to(root)), 'bytes': path.stat().st_size,
            'columns_or_sheets': columns, 'matched_concepts_in_headers_or_preview': matches,
            'preview': example, 'parquet_rows': nrows, 'error': problem,
        }
        rows.append(row)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'source_audit.json').open('w', encoding='utf-8') as f:
        json.dump({'created_utc': datetime.now(timezone.utc).isoformat(), 'total_files_found': len(files), 'audited':rows}, f, ensure_ascii=False, indent=2, default=str)
    with (out / 'source_audit_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['file','bytes','matches','columns_or_sheets','error'])
        for r in rows:
            wr.writerow([r['file'],r['bytes'],'|'.join(r['matched_concepts_in_headers_or_preview']), '|'.join(map(str, r['columns_or_sheets']))[:1000],r['error']])
    print(f'검사 파일: {len(rows)}개')
    print('※ 파일명/헤더/일부 미리보기 기반 후보 탐색이며, 2001~2014년 값의 존재를 확정하지 않음')
    print('\n[관련 계정 후보 파일]')
    matching = [r for r in rows if r['matched_concepts_in_headers_or_preview']]
    for r in matching[:40]:
        print(f"{r['file']}  => {', '.join(r['matched_concepts_in_headers_or_preview'])}")
    print(f'계정 후보 파일 {len(matching)}개 / {len(rows)}개')
    print('\n[기존 DataGuide 유력 파일]')
    for r in rows:
        if any(t in r['file'].lower() for t in ('external_data','dataguide','부채비율.xlsx')):
            print(r['file'], '=>', ','.join(r['matched_concepts_in_headers_or_preview']) or '헤더 일치 없음(원천 항목코드 추가 확인 필요)')
    print('\n요약:', out / 'source_audit_summary.csv')
    print('상세:', out / 'source_audit.json')


if __name__ == '__main__': main()
