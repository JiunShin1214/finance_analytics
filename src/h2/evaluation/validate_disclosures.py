"""DART metadata, fiscal-period screen and original CFS evidence."""

def metadata(output=None):
    output=Path(output) if output else PROCESSED/'disclosures'
    base=ROOT/'results/h2_disclosure_audit/all_stocks'
    # The cached per-company record ties an issuer's corp_code to stock_code.
    mapping={}
    for p in sorted((base/'per_stock').glob('*.json')):
        d=json.loads(p.read_text(encoding='utf-8'))
        if d.get('complete'): mapping.setdefault(d['corp_code'],set()).add(d['stock_code'])
    rows=[]
    for p in sorted((base/'api_cache').glob('list*.json')):
        d=json.loads(p.read_text(encoding='utf-8'))
        for item in d.get('list',[]):
            for stock in sorted(mapping.get(item.get('corp_code'),[])):
                parsed=parse_filing(item,stock,item['corp_code'])
                if parsed: rows.append(parsed)
    if not rows: raise FileNotFoundError('No cached raw DART list metadata')
    history=pd.DataFrame(rows).drop_duplicates(['stock_code','rcept_no']).sort_values(['stock_code','filing_date','rcept_no'])
    reference = history[
    history["report_type"].eq("사업보고서") & history["fiscal_month_label"].ne(12)].copy()
    reference['receipt_prefix_date']=pd.to_datetime(reference.rcept_no.str[:8],format='%Y%m%d').dt.strftime('%Y-%m-%d')
    write_csv(history,output/'dart_disclosure_history.csv')
    write_csv(reference,output/'nonstandard_period_filings_reference.csv')
    write_json({'history_rows':len(history),'reference_rows':len(reference),'live_api_requested':False,
                'source':'Raw DART list API cache; fiscal month from report title, not independently verified from statement'},output/'metadata_summary.json')
    return output/'nonstandard_period_filings_reference.csv'
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
from src.h2.data_io import ROOT, PROCESSED, VALIDATION, write_csv, write_json

import io, zipfile, hashlib
from lxml import html

REPORT_PATTERN = re.compile('(사업보고서|분기보고서|반기보고서)\\s*\\(\\s*(\\d{4})\\s*\\.\\s*(\\d{2})\\s*\\)')

def parse_filing(raw, stock, corp):
    name = str(raw.get('report_nm', ''))
    match = REPORT_PATTERN.search(name)
    if not match:
        return None
    report_type, yy, mm = match.groups()
    mm = int(mm)
    year = int(yy)
    if report_type == "사업보고서":
        period_status = (
            "DEC_ANNUAL_TITLE_CANDIDATE"
            if mm == 12
            else "NON_DEC_ANNUAL_TITLE_CANDIDATE"
        )
    else:
        period_status = "INTERIM_REPORT_TITLE_MONTH_NOT_FISCAL_YEAR"
    rcept_date = str(raw.get('rcept_dt', ''))
    if not re.fullmatch('\\d{8}', rcept_date):
        return None
    date = pd.to_datetime(rcept_date, format='%Y%m%d', errors='coerce')
    if pd.isna(date):
        return None
    rcept_no = str(raw.get('rcept_no', ''))
    return {'stock_code': stock, 'corp_code': corp, 'corp_name': raw.get('corp_name', ''), 'report_nm': name, 'report_type': report_type, 'fiscal_year_label': year, 'fiscal_month_label': mm, 'fiscal_period_label': f'{year:04d}-{mm:02d}', 'filing_date': date.strftime('%Y-%m-%d'), 'rcept_no': rcept_no,
            'is_amendment_name': bool(re.search(r"정정|첨부추가|철회", name)), 'period_status': period_status, 'rcept_url': 'https://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcept_no}

def normal(s):
    return re.sub('\\s+', '', s or '')

def text(e):
    return ' '.join(' '.join(e.itertext()).split())

def html_root(blob):
    for enc in ('cp949', 'utf-8'):
        try:
            return html.fromstring(blob.decode(enc).encode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            pass
    return html.fromstring(blob.decode('cp949', errors='replace').encode('utf-8'))

def nums(row):
    cells = row.xpath('./th|./td')
    values = []
    for c in cells[1:]:
        s = normal(text(c)).replace('△', '-').replace('−', '-')
        if re.fullmatch('\\(?-?\\d[\\d,]*\\)?(?:\\.0+)?', s):
            try:
                values.append(int(s.replace(',', '').replace('(', '-').replace(')', '').split('.')[0]))
            except ValueError:
                pass
    return values

def total(table, pattern):
    values = []
    for tr in table.xpath('.//tr'):
        cs = tr.xpath('./th|./td')
        if len(cs) < 2:
            continue
        label = normal(text(cs[0]))
        if re.fullmatch(pattern, label):
            n = nums(tr)
            if n:
                values.append((n[0], len(n), text(tr)[:200]))
    return values

def parse_source(path, expected_year):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if re.search('_00761\\.xml$', n, re.I)]
        if len(names) != 1:
            return {'status': 'REVIEW_NO_UNIQUE_CFS_AUDIT_XML', 'xml_count': len(names)}
        doc = html_root(z.read(names[0]))
        candidates = []
        for ix, t in enumerate(doc.xpath('//table')):
            la = total(t, '(?:\\(?[가-힣0-9]?\\)?|\\(?[A-Z]\\)?)?부채총계(?:\\(?[A-Z]\\)?)?')
            eq = total(t, '자본총계(?:\\(?[A-Z]\\)?)?')
            if len(la) != 1 or len(eq) != 1:
                continue
            l, nl, lrow = la[0]
            e, ne, erow = eq[0]
            if l <= 0 or e <= 0 or nl < 2 or (ne < 2):
                continue
            pre = ' '.join(' '.join(t.xpath('preceding::text()')[-100:]).split())[-650:]
            is_cfs = bool(re.search('연결재무상태표', normal(pre)))
            has_year = str(expected_year) in pre[-400:]
            if not (is_cfs and has_year):
                continue
            ratio = 100 * l / e
            if not 0.001 < ratio < 100000:
                continue
            candidates.append({'table_index': ix, 'liabilities': l, 'equity': e, 'debt_ratio_pct': ratio, 'section_context': pre[-250:], 'liability_row': lrow, 'equity_row': erow})
        if len(candidates) != 1:
            return {'status': 'REVIEW_AMBIGUOUS_CFS_TABLE', 'strict_candidates': len(candidates), 'cfs_xml': names[0]}
        return {'status': 'STRICT_ORIGINAL_CFS_CURRENT_CANDIDATE', 'strict_candidates': 1, 'cfs_xml': names[0], **candidates[0]}


def extract_originals(output=None):
    from types import SimpleNamespace
    args=SimpleNamespace(queue=ROOT/'results/h2_disclosure_audit/original_filing_download_queue_2018_2021.csv', originals=ROOT/'results/h2_disclosure_audit/bulk_originals_2018_2021/originals', output=Path(output) if output else PROCESSED/'disclosures/bulk', start_month='2018-01', end_month='2021-12')
    q = pd.read_csv(args.queue, dtype={'stock_code': str, 'rcept_no': str, 'corp_code': str})
    req = {'stock_code', 'rcept_no', 'fiscal_year_label', 'filing_date'}
    if req - set(q.columns):
        raise ValueError('missing queue columns: ' + str(req - set(q.columns)))
    q['stock_code'] = q.stock_code.str.zfill(6)
    q['filing_date'] = pd.to_datetime(q['filing_date'], errors='coerce')
    if q[['stock_code', 'rcept_no', 'filing_date']].isna().any().any():
        raise ValueError('missing key/date')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in q.iterrows():
        source = args.originals / (str(r.rcept_no) + '.zip')
        item = {'stock_code': r.stock_code, 'rcept_no': str(r.rcept_no), 'fiscal_year': int(r.fiscal_year_label), 'fiscal_date': f'{int(r.fiscal_year_label)}-12-31', 'disclosure_date': r.filing_date.strftime('%Y-%m-%d'), 'original_file': str(source), 'source_status': 'MISSING_ORIGINAL'}
        if source.is_file():
            try:
                item['original_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
                res = parse_source(source, int(r.fiscal_year_label))
                item.update(res)
                item['source_status'] = item.pop('status')
            except Exception as ex:
                item['source_status'] = 'ERROR_REVIEW'
                item['error'] = str(ex)[:300]
        out.append(item)
        if (i + 1) % 100 == 0:
            print(f'parsed {i + 1}/{len(q)}', flush=True)
    d = pd.DataFrame(out)
    d.to_csv(output / 'original_financial_extraction_all.csv', index=False, encoding='utf-8-sig')
    good = d[d.source_status == 'STRICT_ORIGINAL_CFS_CURRENT_CANDIDATE'].copy()
    good['disclosure_date'] = pd.to_datetime(good.disclosure_date)
    good['fiscal_date'] = pd.to_datetime(good.fiscal_date)
    good = good[good.disclosure_date >= good.fiscal_date].copy()
    good['available_date'] = good.disclosure_date
    dup = good.duplicated(['stock_code', 'fiscal_year'], keep=False)
    good.loc[dup, 'source_status'] = 'REVIEW_DUPLICATE_STOCK_YEAR'
    good = good[~dup].copy()
    events = good.sort_values(['stock_code', 'available_date'])
    events.to_csv(output / 'strict_pit_candidate_events.csv', index=False, encoding='utf-8-sig')
    months = pd.period_range(args.start_month, args.end_month, freq='M')
    stocks = sorted(q.stock_code.unique())
    panel = pd.MultiIndex.from_product([stocks, months.to_timestamp('M')], names=['stock_code', 'prediction_date']).to_frame(index=False)
    panel = panel.sort_values(['stock_code', 'prediction_date']).reset_index(drop=True)
    if events.empty:
        for k in ('debt_ratio_pct', 'source_disclosure_date', 'source_fiscal_date', 'source_rcept_no'):
            panel[k] = pd.NA
    else:
        right = events.rename(columns={'available_date': 'source_disclosure_date', 'fiscal_date': 'source_fiscal_date', 'rcept_no': 'source_rcept_no'})[['stock_code', 'source_disclosure_date', 'source_fiscal_date', 'source_rcept_no', 'debt_ratio_pct']]
        left = panel.sort_values(['prediction_date', 'stock_code']).reset_index(drop=True)
        right = right.sort_values(['source_disclosure_date', 'stock_code']).reset_index(drop=True)
        panel = pd.merge_asof(left, right, left_on='prediction_date', right_on='source_disclosure_date', by='stock_code', direction='backward', allow_exact_matches=True)
        panel = panel.sort_values(['stock_code', 'prediction_date']).reset_index(drop=True)
    panel['source_status'] = panel.source_rcept_no.notna().map({True: 'PRELIMINARY_ORIGINAL_PIT_CANDIDATE', False: 'NO_VERIFIED_ORIGINAL'})
    violation = (panel['source_disclosure_date'] > panel['prediction_date']).fillna(False)
    if violation.any():
        raise AssertionError('future disclosure joined')
    panel['target_month'] = (panel.prediction_date.dt.to_period('M') + 1).astype(str)
    panel['prediction_month'] = panel.prediction_date.dt.strftime('%Y-%m')
    panel.to_csv(output / 'monthly_pit_candidates.csv', index=False, encoding='utf-8-sig')
    summary = {'source_queue_rows': len(q), 'original_files_present': int((d.source_status != 'MISSING_ORIGINAL').sum()), 'strict_candidate_events': len(events), 'candidate_month_rows': int(panel.source_rcept_no.notna().sum()), 'total_month_rows': len(panel), 'future_disclosure_joins': int(violation.sum()), 'needs_manual_source_review': int((d.source_status != 'STRICT_ORIGINAL_CFS_CURRENT_CANDIDATE').sum()), 'fully_PIT_verified': False, 'coverage_limit': 'Queue covers annual 2018-2021 filings only. Missing years/issuers not filled.', 'warning': 'Do not use as final Train or test feature without DataGuide reconciliation, submission/amendment review and coverage audit.'}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def reconcile(output=None, extracted=None, panel=None):
    IN=Path(extracted) if extracted else PROCESSED/'disclosures/bulk'
    O=Path(output) if output else PROCESSED/'disclosures/reconciliation'
    O.mkdir(parents=True,exist_ok=True)
    PANEL=Path(panel) if panel else PROCESSED/'dataset.csv'
    DEBT=ROOT/'data/raw/부채비율.xlsx'
    read = lambda f: pd.read_csv(f, dtype={'stock_code': str, 'rcept_no': str, 'source_rcept_no': str, 'stock_id': str})
    for name in ['strict_pit_candidate_events.csv', 'original_financial_extraction_all.csv', 'monthly_pit_candidates.csv']:
        if not (IN / name).is_file():
            raise FileNotFoundError(f'DART 추출 파일 없음: {IN / name}\n--input-dir 옵션으로 지정하세요.')
    E = read(IN / 'strict_pit_candidate_events.csv')
    X = read(IN / 'original_financial_extraction_all.csv')
    M = read(IN / 'monthly_pit_candidates.csv')
    H = read(PANEL)
    for d, c in [(E, 'stock_code'), (X, 'stock_code'), (M, 'stock_code'), (H, 'stock_id')]:
        d[c] = d[c].astype(str).str.replace('\\.0$', '', regex=True).str.zfill(6)
    df = pd.read_excel(DEBT, header=None, engine='openpyxl')
    codes = df.iloc[8, 1:].astype(str).str.extract('(\\d{6})')[0]
    jan = df.iloc[14:, :]
    dates = pd.to_datetime(jan.iloc[:, 0], errors='coerce')
    jan = jan.loc[(dates.dt.month == 1) & dates.dt.year.between(2018, 2021)]
    r = []
    for i, row in jan.iterrows():
        y = pd.Timestamp(row.iloc[0]).year
        vals = pd.to_numeric(row.iloc[1:], errors='coerce')
        ok = vals.notna() & codes.notna()
        r.extend(zip(codes[ok].values, [y] * ok.sum(), vals[ok].values))
    DG = pd.DataFrame(r, columns=['stock_code', 'fiscal_year', 'dg_ratio_pct'])
    DG = DG.drop_duplicates(['stock_code', 'fiscal_year'])
    E['fiscal_year'] = pd.to_numeric(E['fiscal_year'], errors='coerce').astype('Int64')
    E['debt_ratio_pct'] = pd.to_numeric(E['debt_ratio_pct'], errors='coerce')
    E['disclosure_date'] = pd.to_datetime(E['disclosure_date'], errors='coerce')
    E['fiscal_date'] = pd.to_datetime(E['fiscal_date'], errors='coerce')
    comp = E.merge(DG, on=['stock_code', 'fiscal_year'], how='left', validate='m:1', indicator=True)
    comp['difference_pp'] = comp.debt_ratio_pct - comp.dg_ratio_pct
    comp['match_rounded2'] = comp.dg_ratio_pct.notna() & (comp.debt_ratio_pct.round(2) == comp.dg_ratio_pct.round(2))
    comp['abs_difference_pp'] = comp.difference_pp.abs()
    comp['value_status'] = np.select([comp.dg_ratio_pct.isna(), comp.match_rounded2, comp.abs_difference_pp.le(0.05), comp.abs_difference_pp.le(1)], ['NO_DG_VALUE', 'ROUND2_MATCH', 'NEAR_WITHIN_0_05PP', 'DIFF_WITHIN_1PP'], default='MISMATCH_GT_1PP')
    comp['disclosure_after_fiscal'] = comp.disclosure_date > comp.fiscal_date
    comp.to_csv(O / 'dataguide_dart_event_reconciliation.csv', index=False, encoding='utf-8-sig')
    M['prediction_month'] = M['prediction_month'].astype(str).str[:7]
    H['feature_month'] = H['feature_month'].astype(str).str[:7]
    M['source_disclosure_date'] = pd.to_datetime(M.source_disclosure_date, errors='coerce')
    M['prediction_date'] = pd.to_datetime(M.prediction_date, errors='coerce')
    M['source_fiscal_date'] = pd.to_datetime(M.source_fiscal_date, errors='coerce')
    M['source_rcept_no'] = M.source_rcept_no.astype('string')
    H['debt_source_month'] = pd.to_datetime(H.debt_source_month, errors='coerce')
    H['debt_ratio_raw'] = pd.to_numeric(H.debt_ratio_raw, errors='coerce')
    H['debt_ratio_z'] = pd.to_numeric(H.debt_ratio_z, errors='coerce')
    Hpart = H.loc[H.feature_month.between('2018-01', '2021-12'), ['stock_id', 'feature_month', 'debt_source_month', 'debt_ratio_raw', 'debt_ratio_z', 'eligible', 'final_rule_v2_strict', 'target_month']]
    print('Hpart shape', Hpart.shape, 'unique', Hpart[['stock_id', 'feature_month']].duplicated().sum(), 'M', M.shape, 'dup', M[['stock_code', 'prediction_month']].duplicated().sum())
    joined = Hpart.merge(M, left_on=['stock_id', 'feature_month'], right_on=['stock_code', 'prediction_month'], how='left', indicator=True, validate='m:1')
    joined['legacy_raw_available'] = joined.debt_ratio_raw.notna()
    joined['pit_candidate_available'] = joined.debt_ratio_pct.notna()
    joined['bad_legacy_vs_verified_fiscal_end'] = joined.pit_candidate_available & joined.legacy_raw_available & (joined.debt_source_month < joined.source_fiscal_date)
    joined['pit_asof_violation'] = joined.pit_candidate_available & (joined.source_disclosure_date > joined.prediction_date)
    joined['raw_difference_pp'] = joined.debt_ratio_raw - joined.debt_ratio_pct
    joined['raw_round2_match'] = joined.legacy_raw_available & joined.pit_candidate_available & (joined.debt_ratio_raw.round(2) == joined.debt_ratio_pct.round(2))
    joined['old_source_year'] = joined.debt_source_month.dt.year
    joined['source_year_equal_current'] = joined.old_source_year.eq(joined.source_fiscal_date.dt.year)
    L = Hpart.loc[Hpart.debt_ratio_raw.notna()].copy()
    L['source_year'] = L.debt_source_month.dt.year
    C = comp.loc[comp.match_rounded2, ['stock_code', 'fiscal_year', 'disclosure_date', 'debt_ratio_pct', 'rcept_no']].copy()
    legacy = L.merge(C, left_on=['stock_id', 'source_year'], right_on=['stock_code', 'fiscal_year'], how='inner')
    legacy['same_dg_year_original'] = legacy.debt_ratio_raw.round(2) == legacy.debt_ratio_pct.round(2)
    legacy['feature_cutoff'] = pd.to_datetime(legacy.feature_month + '-01') + pd.offsets.MonthEnd(0)
    legacy['before_fiscal_end'] = legacy.feature_cutoff < pd.to_datetime(legacy.fiscal_year.astype(str) + '-12-31')
    legacy['before_filing'] = legacy.feature_cutoff < legacy.disclosure_date
    legacy_hits = legacy.loc[legacy.same_dg_year_original].copy()
    legacy_hits.to_csv(O / 'legacy_source_year_verified_matches.csv', index=False, encoding='utf-8-sig')
    comp_2 = comp[['rcept_no', 'value_status', 'match_rounded2', 'fiscal_year', 'dg_ratio_pct']].copy()
    comp_2['rcept_no'] = comp_2.rcept_no.astype('string')
    joined = joined.merge(comp_2, left_on='source_rcept_no', right_on='rcept_no', how='left', validate='m:1')
    joined.to_csv(O / 'legacy_vs_candidate_monthly_2018_2021.csv', index=False, encoding='utf-8-sig')
    E_year = comp.groupby('fiscal_year').agg(events=('rcept_no', 'size'), match=('match_rounded2', 'sum'), with_dataguide=('dg_ratio_pct', 'count'), diff_median_pp=('abs_difference_pp', 'median')).reset_index()
    E_year.to_csv(O / 'event_reconciliation_by_year.csv', index=False, encoding='utf-8-sig')
    monthly = joined.groupby('feature_month').agg(legacy_rows=('legacy_raw_available', 'sum'), pit_rows=('pit_candidate_available', 'sum'), rows=('stock_id', 'size'), verified_dg_pit_rows=('match_rounded2', 'sum'), asof_violations=('pit_asof_violation', 'sum')).reset_index()
    monthly['pit_coverage'] = monthly.pit_rows / monthly.rows
    monthly.to_csv(O / 'pit_coverage_by_month.csv', index=False, encoding='utf-8-sig')
    summary = {'original_events': len(X), 'strict_candidate_events': len(E), 'candidate_month_rows': int(M.debt_ratio_pct.notna().sum()), 'candidate_total_month_rows': len(M), 'dataguide_2018_2021_january_stock_years': len(DG), 'events_dataguide_value_available': int(comp.dg_ratio_pct.notna().sum()), 'events_round2_match': int(comp.match_rounded2.sum()), 'events_round2_match_rate_among_comparable': float(comp.match_rounded2.sum() / max(1, comp.dg_ratio_pct.notna().sum())), 'event_diff_gt_1pp': int((comp.abs_difference_pp > 1).sum()), 'event_diff_0_05_to_1pp': int(comp.abs_difference_pp.between(0.05, 1, inclusive='right').sum()), 'h2_rows_2018_2021': len(Hpart), 'h2_rows_joined_with_candidate': int(joined.pit_candidate_available.sum()), 'h2_rows_without_candidate': int((~joined.pit_candidate_available).sum()), 'h2_rows_with_legacy_debt': int(joined.legacy_raw_available.sum()), 'h2_rows_both': int((joined.pit_candidate_available & joined.legacy_raw_available).sum()), 'h2_rows_raw_equal_pit': int(joined.raw_round2_match.sum()), 'h2_pit_asof_violations': int(joined.pit_asof_violation.sum()), 'h2_pit_rows_dg_match': int(joined.match_rounded2.fillna(False).sum()), 'legacy_source_year_matched_rows': len(legacy_hits), 'legacy_before_fiscal_end_rows_matched': int(legacy_hits.before_fiscal_end.sum()), 'legacy_before_filing_rows_matched': int(legacy_hits.before_filing.sum()), 'original_legacy_panel_unmodified': True, 'point_in_time_financial_values_fully_verified': False}
    (O / 'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('\nEVENT STATUSES', comp.value_status.value_counts().to_string())
    print('\nYEARS', E_year.to_string(index=False))
    print('\nCOVERAGE', monthly.pit_coverage.describe().to_string())
    print('\nCOMP MISMATCHES', comp.loc[comp.value_status == 'MISMATCH_GT_1PP', ['stock_code', 'fiscal_year', 'debt_ratio_pct', 'dg_ratio_pct', 'difference_pp']].head(20).to_string(index=False))


def build_candidate(output=None, panel=None):
    O=Path(output) if output else PROCESSED/'disclosures/reconciliation'
    PANEL=Path(panel) if panel else PROCESSED/'dataset.csv'
    E_PATH = O / 'dataguide_dart_event_reconciliation.csv'
    if not E_PATH.is_file():
        raise FileNotFoundError(f'먼저 analyze_pit_reconciliation.py를 실행하세요. 중간 파일 없음: {E_PATH}')
    E = pd.read_csv(E_PATH, dtype={'stock_code': str, 'rcept_no': str})
    E.stock_code = E.stock_code.str.zfill(6)
    H = pd.read_csv(PANEL, usecols=['stock_id', 'feature_month', 'target_month', 'debt_source_month', 'debt_ratio_raw', 'debt_ratio_z', 'eligible'], dtype={'stock_id': str})
    H.stock_id = H.stock_id.str.zfill(6)
    H = H.loc[H.feature_month.between('2018-01', '2021-12')].copy()
    H['prediction_date'] = pd.to_datetime(H.feature_month + '-01') + pd.offsets.MonthEnd(0)
    E = E.loc[E.match_rounded2 & E.debt_ratio_pct.notna() & E.disclosure_date.notna()].copy()
    E['available_date'] = pd.to_datetime(E.disclosure_date)
    E['fiscal_date'] = pd.to_datetime(E.fiscal_date)
    E = E.sort_values(['available_date', 'stock_code', 'rcept_no']).drop_duplicates(['stock_code', 'available_date'], keep='first')
    H1 = H.sort_values(['prediction_date', 'stock_id'])
    E1 = E.sort_values(['available_date', 'stock_code'])
    J = pd.merge_asof(H1, E1, left_on='prediction_date', right_on='available_date', left_by='stock_id', right_by='stock_code', direction='backward', allow_exact_matches=True, suffixes=('_legacy', '_verified'))
    J['candidate_available'] = J.debt_ratio_pct.notna()
    J['date_violation'] = J.candidate_available & (J.available_date > J.prediction_date)
    J['original_legacy_debt_ratio_pct'] = pd.to_numeric(J.debt_ratio_raw, errors='coerce')
    J['reconstructed_candidate_debt_pct'] = J.debt_ratio_pct
    J['original_legacy_vs_candidate_diff_pp'] = J.original_legacy_debt_ratio_pct - J.reconstructed_candidate_debt_pct
    J['candidate_evidence_level'] = 'ANNUAL_ORIGINAL_CFS_CANDIDATE_MATCHED_2026_DATAGUIDE_NOT_FULL_PIT'
    J.loc[~J.candidate_available, 'candidate_evidence_level'] = 'NO_RECONCILED_FILING'
    cols = ['stock_id', 'feature_month', 'target_month', 'prediction_date', 'original_legacy_debt_ratio_pct', 'debt_ratio_z', 'debt_source_month', 'reconstructed_candidate_debt_pct', 'fiscal_year', 'fiscal_date', 'available_date', 'rcept_no', 'original_sha256', 'candidate_available', 'candidate_evidence_level', 'original_legacy_vs_candidate_diff_pp', 'date_violation']
    J[cols].to_csv(O / 'reconciled_annual_pit_candidate_2018_2021.csv', index=False, encoding='utf-8-sig')
    C = J.groupby('feature_month').agg(h2_rows=('stock_id', 'size'), reconciled_candidate=('candidate_available', 'sum'), legacy_valid=('original_legacy_debt_ratio_pct', 'count'), asof_violations=('date_violation', 'sum')).reset_index()
    C['coverage'] = C.reconciled_candidate / C.h2_rows
    C.to_csv(O / 'reconciled_candidate_monthly_coverage.csv', index=False, encoding='utf-8-sig')
    S = {'rows': len(J), 'reconciled_candidate_rows': int(J.candidate_available.sum()), 'no_candidate_rows': int((~J.candidate_available).sum()), 'coverage': round(float(J.candidate_available.mean()), 6), 'future_disclosure_joins': int(J.date_violation.sum()), 'source_events_reconciled': len(E), 'fiscal_year_first_available': int(E.fiscal_year.min()), 'fiscal_year_last_available': int(E.fiscal_year.max()), 'safe_for_final_training': False, 'reason': 'Annual 2018-2021 candidate only; issuer and earlier-year gaps; amendment and preliminary publication timing not exhaustively verified'}
    (O / 'reconciled_candidate_summary.json').write_text(json.dumps(S, ensure_ascii=False, indent=2), encoding='utf-8')
    print(S)
    print(C.head(15).to_string(index=False))
