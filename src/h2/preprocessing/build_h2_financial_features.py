"""OpenDART 파일럿 진단 및 보고서별 H2 재무 변수 생성.

실행 (프로젝트 루트):
 python -m src.h2.preprocessing.build_h2_financial_features --check-only
 python -m src.h2.preprocessing.build_h2_financial_features --execute

보고서 단위 결과이며 월별 H2 feature panel에 자동 병합하지 않음.
OpenDART 현재 응답은 historical point-in-time(PIT) 보장 불가.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

DIRECT_INTEREST = {'dart_InterestExpenseFinanceExpense'}
FINANCE_COSTS = {'ifrs_FinanceCosts', 'ifrs-full_FinanceCosts'}
OP_INCOME = {'dart_OperatingIncomeLoss', 'ifrs_ProfitLossFromOperatingActivities', 'ifrs-full_ProfitLossFromOperatingActivities'}
CURRENT_LIAB = {'ifrs_CurrentLiabilities', 'ifrs-full_CurrentLiabilities'}
TOTAL_LIAB = {'ifrs_Liabilities', 'ifrs-full_Liabilities'}
FSTYPE = ('IS', 'CIS')


def money(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace(',', '').replace(' ', '')
    if not s or s in ('-', '–', '—', 'nan', 'None'): return np.nan
    if s.startswith('(') and s.endswith(')'): s = '-' + s[1:-1]
    try: return float(s)
    except ValueError: return np.nan


def amount(row, is_flow):
    # DART 현재 당기값: q1/annual = 보고기간 전체; h1/q3는 개별 분기 금액이 있을 때만 사용.
    # 두 흐름변수 모두 동일한 회계기간이 되도록 한다.
    if not is_flow: return money(row['thstrm_amount']), 'balance_sheet'
    period = row['report_type']
    if period in ('q1', 'annual'):
        return money(row['thstrm_amount']), 'q1' if period == 'q1' else 'annual'
    val = money(row['thstrm_add_amount'])
    if np.isfinite(val): return val, 'quarter_discrete'
    return np.nan, 'missing_discrete_quarter'


def choose(rows, ids, stmt, is_flow, allow_name=None):
    ss = rows[rows['sj_div'].isin(stmt)].copy()
    exact = ss[ss['account_id'].isin(ids)]
    if len(exact) == 0 and allow_name:
        exact = ss[ss['account_nm'].isin(allow_name)]
    # IS/CIS are alternate statements, never sum them; prefer IS if both exist.
    if len(exact) and set(stmt).intersection(FSTYPE):
        if (exact['sj_div'] == 'IS').any(): exact = exact[exact['sj_div'] == 'IS']
    amounts = [(float(v), basis, str(row['account_id'])) for _, row in exact.iterrows()
               for v, basis in [amount(row, is_flow)] if np.isfinite(v)]
    if not amounts: return np.nan, 'missing', None
    # Do not silently select multiple conflicting candidate amounts.
    unique = {round(v, 3) for v, _, _ in amounts}
    if len(unique) > 1: return np.nan, 'conflict', None
    v, basis, account = amounts[0]
    return v, basis, account


def build(df):
    required = {'stock_code','corp_code','bsns_year','report_type','rcept_no','fs_div',
                'sj_div','account_id','account_nm','thstrm_amount','thstrm_add_amount'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'필수 컬럼 누락: {sorted(missing)}')
    # A report is defined by corp, fiscal year, report type, filing and financial statement type.
    keys = ['stock_code','corp_code','bsns_year','report_type','rcept_no','fs_div']
    out=[]
    for key, part in df.groupby(keys, dropna=False, sort=True):
        rec = dict(zip(keys, key))
        cur, cs, _ = choose(part, CURRENT_LIAB, ['BS'], False, ['유동부채'])
        tot, ts, _ = choose(part, TOTAL_LIAB, ['BS'], False, ['부채총계'])
        op, os, oa = choose(part, OP_INCOME, FSTYPE, True, ['영업이익(손실)','영업이익'])
        direct, ds, da = choose(part, DIRECT_INTEREST, FSTYPE, True)
        costs, fs, fa = choose(part, FINANCE_COSTS, FSTYPE, True)
        cf = part[(part['sj_div']=='CF') & (part['account_id']=='dart_AdjustmentsForInterestExpenses')]
        rec.update({
            'current_liabilities':cur,'total_liabilities':tot,
            'short_term_debt_ratio':cur/tot if np.isfinite(cur) and np.isfinite(tot) and tot>0 and cur>=0 else np.nan,
            'operating_income':op,'operating_income_basis':os,
            'interest_expense_direct':direct,'interest_direct_basis':ds,'interest_direct_account':da,
            'finance_costs_proxy':costs,'finance_costs_basis':fs,'finance_costs_account':fa,
            'cf_interest_adjustment_present':not cf.empty,
            'interest_coverage_strict': op/direct if np.isfinite(op) and np.isfinite(direct) and direct>0 and os==ds else np.nan,
            'interest_coverage_finance_costs_proxy':op/costs if np.isfinite(op) and np.isfinite(costs) and costs>0 and os==fs else np.nan,
        })
        out.append(rec)
    if not out: raise ValueError('보고서 데이터가 없습니다.')
    frame = pd.DataFrame(out)
    frame['rcept_date'] = pd.to_datetime(frame['rcept_no'].astype(str).str.slice(0,8),format='%Y%m%d',errors='coerce').dt.strftime('%Y-%m-%d')
    return frame


def main():
    parser=argparse.ArgumentParser(description='H2 OpenDART 재무변수 생성: 보고서 단위, 보수적 직접 이자비용')
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--input', type=Path, default=Path('data/raw/h2/opendart_financials_pilot.csv'))
    parser.add_argument('--out', type=Path, default=Path('data/processed/final/h2/opendart_financial_features_pilot.csv'))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args=parser.parse_args()
    base=args.root.resolve()
    src=base/args.input
    dst=base/args.out
    if not src.is_file(): raise FileNotFoundError(f'입력 데이터 없음: {src}')
    df=pd.read_csv(src,dtype=str,encoding='utf-8-sig')
    frame=build(df)
    reports=len(frame)
    def summarize(column):
        return {'valid_reports':int(frame[column].notna().sum()), 'total_reports':reports,
                'coverage':round(float(frame[column].notna().mean()),6)}
    diagnostic = {
        'input_file':str(src), 'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'source_rows':len(df),'companies':int(frame.stock_code.nunique()),'report_count':reports,
        'report_count_by_stock':frame.groupby('stock_code').size().to_dict(),
        'short_term_debt_ratio':summarize('short_term_debt_ratio'),
        'strict_interest_coverage':summarize('interest_coverage_strict'),
        'finance_costs_proxy_coverage':summarize('interest_coverage_finance_costs_proxy'),
        'direct_interest_reports':int(frame.interest_expense_direct.notna().sum()),
        'direct_interest_conflicts':int((frame.interest_direct_basis=='conflict').sum()),
        'finance_costs_reports':int(frame.finance_costs_proxy.notna().sum()),
        'cf_adjustment_reports':int(frame.cf_interest_adjustment_present.sum()),
        'negative_or_zero_direct_interest':int((frame.interest_expense_direct.le(0)).fillna(False).sum()),
        'missing_discrete_operating_income':int((frame.operating_income_basis=='missing').sum()),
        'limitations':['Current API financials are not guaranteed historical PIT',
                       'rcept_no date is a filing identifier, not proof of first-filed vintage',
                       'strict coverage uses only IS/CIS direct interest expense',
                       'FinanceCosts proxy is distinct and never silently substituted',
                       'h1/q3 flow values without discrete-quarter amount remain missing',
                       'CFS-only collection excludes firms without consolidated statements',
                       'report-level output is NOT month-panel mapped'],
        'created_utc':datetime.now(timezone.utc).isoformat()
    }
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False))
    if not args.execute:
        print('[점검 완료] 출력 파일 생성하지 않음. --execute로 저장')
        return
    audit=dst.with_name(dst.stem+'_audit.json')
    if dst.exists() or audit.exists(): raise FileExistsError(f'파일 덮어쓰기 방지: {dst} / {audit}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(dst,index=False,encoding='utf-8-sig')
    audit.write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2),encoding='utf-8')
    print('[생성 완료]',dst,'\n[진단 JSON]',audit)

if __name__ == '__main__': main()
