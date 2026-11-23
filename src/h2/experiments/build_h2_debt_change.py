# -*- coding: utf-8 -*-
"""Build point-in-time-aware DebtChange from the existing H2 Hybrid monthly panel.

Run at repository root:
  python -m src.h2.experiments.build_h2_debt_change --check-only
  python -m src.h2.experiments.build_h2_debt_change --execute

This program does not modify input files. It refuses to overwrite output files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path('data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv')
DEFAULT_OUTPUT = Path('data/processed/final/debt_hybrid/h2_debt_change_2001_2021.csv')
KEY = ['stock_id', 'feature_month']
REQUIRED = KEY + ['debt_final_raw', 'source_fiscal_date', 'source_available_date', 'source_type']


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stock_code(s: pd.Series) -> pd.Series:
    s = s.astype('string').str.strip().str.replace(r'^A(?=\d{6}$)', '', regex=True)
    s = s.str.replace(r'\.0$', '', regex=True)
    if not s.str.fullmatch(r'\d{1,6}', na=False).all():
        raise ValueError('stock_id 형식 오류')
    return s.str.zfill(6)


def read_hybrid(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={'stock_id': 'string', 'feature_month': 'string'}, low_memory=False)
    missing = set(REQUIRED) - set(d.columns)
    if missing:
        raise ValueError(f'Hybrid 필수 컬럼 누락: {sorted(missing)}')
    d = d[REQUIRED].copy()
    d['stock_id'] = stock_code(d['stock_id'])
    if not d['feature_month'].str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', na=False).all():
        raise ValueError('feature_month는 YYYY-MM이어야 합니다.')
    if d[KEY].isna().any().any() or d.duplicated(KEY).any():
        raise ValueError('stock_id, feature_month 중복 또는 결측')
    d['debt_final_raw'] = pd.to_numeric(d['debt_final_raw'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for name in ['source_fiscal_date', 'source_available_date']:
        d[name] = pd.to_datetime(d[name], errors='coerce')
    d['prediction_date'] = pd.PeriodIndex(d['feature_month'], freq='M').to_timestamp('M')
    active = d['debt_final_raw'].notna()
    invalid = active & (d['source_fiscal_date'].isna() | d['source_available_date'].isna()
                        | d['source_fiscal_date'].gt(d['prediction_date'])
                        | d['source_available_date'].gt(d['prediction_date']))
    if invalid.any():
        example = d.loc[invalid, KEY + ['source_fiscal_date', 'source_available_date']].head(3)
        raise ValueError(f'미래/누락 공시 정보: {invalid.sum():,}행\n{example}')
    return d.sort_values(KEY).reset_index(drop=True)


def compute_debt_change(data: pd.DataFrame) -> pd.DataFrame:
    """Change versus last *distinct fiscal-period* report seen as-of each month.

    Carry the most recently computed report-to-report change during months without
    a new report. First report has no previous report -> NaN. A regression in the
    selected fiscal period is considered invalid (not a new observation). A source
    change within the same fiscal period is not counted as a new report; a changed
    value is flagged and not used to update the reference.
    """
    data = data.sort_values(KEY).reset_index(drop=True).copy()
    records = []
    for stock, group in data.groupby('stock_id', sort=False):
        prev_fiscal = None
        prev_value = None
        prev_delta = np.nan
        prev_report_month = None
        for r in group.itertuples(index=False):
            val, fiscal = r.debt_final_raw, r.source_fiscal_date
            flag = 'missing_source'
            delta = np.nan
            prev_used_fiscal, prev_used_value = pd.NaT, np.nan
            if pd.notna(val) and pd.notna(fiscal):
                if prev_fiscal is None:
                    flag = 'first_report'
                    prev_fiscal, prev_value = fiscal, float(val)
                    prev_report_month = r.feature_month
                elif fiscal < prev_fiscal:
                    flag = 'fiscal_regression'
                elif fiscal == prev_fiscal:
                    if not np.isclose(float(val), prev_value, atol=1e-9, rtol=1e-9):
                        flag = 'same_fiscal_value_changed'
                    else:
                        flag = 'carried_forward'
                        delta = prev_delta
                else:
                    flag = 'new_report'
                    prev_used_fiscal, prev_used_value = prev_fiscal, prev_value
                    delta = float(val) - prev_value
                    prev_fiscal, prev_value = fiscal, float(val)
                    prev_delta = delta
                    prev_report_month = r.feature_month
            records.append((stock, r.feature_month, delta, flag,
                            prev_used_fiscal, prev_used_value, prev_report_month))
    result = pd.DataFrame(records, columns=KEY + [
        'debt_change_raw', 'debt_change_status', 'previous_fiscal_date',
        'previous_debt_raw', 'latest_report_first_seen_month'])
    # Include information to audit report selection and timing. No post-2021 data needed.
    result = result.merge(data[KEY + ['source_fiscal_date', 'source_available_date',
                                      'source_type', 'debt_final_raw']], on=KEY,
                          how='left', validate='one_to_one')
    return result.sort_values(KEY).reset_index(drop=True)


def summarize(result: pd.DataFrame) -> dict:
    counts = result['debt_change_status'].value_counts(dropna=False)
    return {
        'rows': len(result),
        'stocks': int(result.stock_id.nunique()),
        'months': int(result.feature_month.nunique()),
        'debt_change_available_rows': int(result.debt_change_raw.notna().sum()),
        'debt_change_available_share': float(result.debt_change_raw.notna().mean()),
        'status_counts': {str(k): int(v) for k,v in counts.items()},
        'definition': 'Current available fiscal report ratio minus previous distinct available fiscal report ratio; held constant until next report',
        'unit': 'percentage points when debt_final_raw is percentage',
        'pit_limitation': 'Uses Hybrid source vintage; source is not guaranteed fully historical PIT',
    }


def main():
    p = argparse.ArgumentParser(description='Build H2 DebtChange from Hybrid debt report vintage')
    p.add_argument('--root', type=Path, default=Path('.'), help='Project root; run at finance_analytics root')
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Diagnose without writing (default)')
    mode.add_argument('--execute', action='store_true', help='Write a new CSV and audit JSON (never overwrite)')
    a = p.parse_args()
    root = a.root.resolve()
    input_path = a.input if a.input.is_absolute() else root / a.input
    out_path = a.out if a.out.is_absolute() else root / a.out
    if not input_path.is_file():
        raise FileNotFoundError(f'Hybrid input not found: {input_path}')
    if a.execute and (out_path.exists() or out_path.with_suffix('.json').exists()):
        raise FileExistsError(f'기존 파일 보호: {out_path} 또는 감사 JSON 파일이 존재합니다.')
    data = read_hybrid(input_path)
    result = compute_debt_change(data)
    info = summarize(result)
    info.update({'input_path': str(input_path), 'input_sha256': file_hash(input_path),
                 'created_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print('\n월별 커버리지 마지막 8개월:\n' +
          result.groupby('feature_month').debt_change_raw.apply(lambda x: round(x.notna().mean(), 3)).tail(8).to_string())
    if not a.execute:
        print('\n[점검 완료] 파일을 생성하지 않았습니다. 저장하려면 --execute를 사용하세요.')
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse a concurrent overwrite, too.
    with out_path.open('x', encoding='utf-8-sig', newline='') as f:
        result.to_csv(f, index=False)
    with out_path.with_suffix('.json').open('x', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f'\n[저장 완료] {out_path}')


if __name__ == '__main__':
    main()
