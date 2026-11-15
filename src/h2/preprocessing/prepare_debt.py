"""Annual disclosure-lag proxies. NOT historically verified PIT."""

def prepare(panel_path=None, reference_path=None, output=None):
    output = Path(output) if output else PROCESSED/'debt'
    output.mkdir(parents=True,exist_ok=True)
    panel_path = panel_path or PROCESSED/'dataset.csv'
    reference_path = reference_path or PROCESSED/'disclosures/nonstandard_period_filings_reference.csv'
    annual, meta = load_annual_excel(ROOT/'data/raw/부채비율.xlsx')
    h=pd.read_csv(panel_path,usecols=['stock_id','feature_month','target_month','debt_ratio_raw'],dtype={'stock_id':str,'feature_month':str,'target_month':str})
    h['stock_id']=normalize_id(h.stock_id)
    if h[['stock_id','feature_month']].duplicated().any(): raise ValueError('Duplicate stock-month')
    h=h[h.feature_month.between('2001-01','2021-12')].copy()
    h['legacy_debt_raw']=pd.to_numeric(h.debt_ratio_raw,errors='coerce'); h=h.drop(columns='debt_ratio_raw')
    annual=annual[annual.stock_id.isin(h.stock_id.unique())].copy()
    write_csv(annual,output/'annual_debt_source_year_assumed.csv')
    reference=pd.read_csv(reference_path,dtype={'stock_code':'string'})
    risk=reference[reference.report_type.eq('사업보고서') & pd.to_numeric(reference.fiscal_month_label,errors='coerce').ne(12) & pd.to_numeric(reference.fiscal_year_label,errors='coerce').between(2000,2025)]
    risky=set(normalize_id(risk.stock_code).dropna().astype(str))
    if not risky: raise ValueError('Missing non-December fiscal evidence; screening cannot be silently disabled')
    summaries=[]
    for lag in [4,6,9]:
        lag_panel(h,annual,lag,.01,output)
        # Preserve the previous CSV boundary and dtype conversions exactly.
        x=pd.read_csv(output/f'h2_debt_lag_{lag}m_2001_2021.csv',dtype={'stock_id':'string','feature_month':'string','target_month':'string'},low_memory=False)
        x['stock_id']=normalize_id(x.stock_id)
        x['nondec_annual_filing_flag']=x.stock_id.isin(risky)
        x['risk_class']=np.where(x.nondec_annual_filing_flag,'NONDEC_ANNUAL_FILING_SEEN_EXCLUDE_PROXY','NO_NONDEC_FOUND_IN_METADATA_NOT_VERIFIED_DEC')
        x['proxy_masked']=x.debt_proxy_raw.where(~x.nondec_annual_filing_flag)
        x['debt_proxy_z_masked']=x.debt_proxy_z.where(~x.nondec_annual_filing_flag)
        x['source_assurance']='DATA_GUIDE_ANNUAL_LAG_APPROX_NOT_HISTORICAL_PIT'
        x['mask_reason']=np.where(x.nondec_annual_filing_flag,'NONDEC_ANNUAL_FILING_HISTORY','')
        x['lag_months']=lag
        write_csv(x,output/f'h2_debt_lag_{lag}m_fiscal_screened_2001_2021.csv')
        valid=x.proxy_masked.notna()
        for col in ['proxy_available_date','fiscal_date']:
            dates=pd.to_datetime(x[col],errors='coerce')
            if (valid & (dates.isna() | dates.gt(pd.to_datetime(x.prediction_date)))).any():
                raise ValueError(f'Invalid/future {col} for lag {lag}')
        summaries.append({'lag':lag,'rows':len(x),'nonmissing':int(valid.sum()),'masked_rows':int((x.debt_proxy_raw.notna() & ~valid).sum())})
    write_json({'status':'PROXY_ONLY_NOT_EXACT_PIT','excel_layout':meta,'nondec_issuers':len(risky),'scenarios':summaries,
                'limitations':['same-year January annual value mapping is an assumption','DataGuide restatements unresolved','actual filing dates not verified for all years','issuer-wide historical metadata screen is retrospective']},output/'summary.json')
    return output
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
from src.h2.data_io import ROOT, PROCESSED, VALIDATION, write_csv, write_json

def load_annual_excel(path):
    raw = pd.read_excel(path, header=None, engine='openpyxl')
    if raw.shape[0] < 20 or raw.shape[1] < 3:
        raise ValueError(f'Unexpected DataGuide layout {raw.shape}')
    codes = raw.iloc[8, 1:].astype('string').str.extract('(\\d{6})', expand=False)
    code_valid = codes.notna().to_numpy()
    dates = pd.to_datetime(raw.iloc[14:, 0], errors='coerce')
    jan_idx = np.flatnonzero((dates.dt.month == 1).fillna(False).to_numpy())
    rows = []
    duplicate_headers = int(codes.dropna().duplicated().sum())
    for ix in jan_idx:
        year = int(dates.iloc[ix].year)
        vals = pd.to_numeric(raw.iloc[14 + ix, 1:], errors='coerce')
        ok = code_valid & vals.notna().to_numpy() & np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))
        if ok.any():
            block = pd.DataFrame({'stock_id': codes.iloc[np.flatnonzero(ok)].astype(str).to_numpy(), 'fiscal_year': year, 'debt_ratio_pct': vals.iloc[np.flatnonzero(ok)].to_numpy(dtype=float)})
            rows.append(block)
    if not rows:
        raise RuntimeError('No valid January debt values in workbook')
    annual = pd.concat(rows, ignore_index=True)
    conflicts = annual.groupby(['stock_id', 'fiscal_year']).debt_ratio_pct.nunique(dropna=True)
    if (conflicts > 1).any():
        raise ValueError(f'Conflicting duplicate DataGuide values: {(conflicts > 1).sum()} stock-years; abort')
    annual = annual.drop_duplicates(['stock_id', 'fiscal_year'])
    annual['fiscal_date'] = pd.to_datetime(annual.fiscal_year.astype(str) + '-12-31')
    annual['source_type'] = 'DG_JAN_ANNUAL_ASSUMED_DEC_FISCAL_YEAR'
    return (annual.sort_values(['stock_id', 'fiscal_date']).reset_index(drop=True), {'workbook_rows': len(raw), 'workbook_cols': len(raw.columns), 'duplicate_stock_headers': duplicate_headers, 'jan_years': len(jan_idx)})

def normalize_id(v):
    return v.astype('string').str.extract('(\\d{1,6})(?:\\.0)?$', expand=False).str.zfill(6)

def lag_panel(panel, annual, lag, q, output):
    ev = annual.copy()
    ev['proxy_available_date'] = ev.fiscal_date + pd.DateOffset(months=lag)
    ev['proxy_available_date'] = ev.proxy_available_date + pd.offsets.MonthEnd(0)
    ev = ev.sort_values(['proxy_available_date', 'stock_id', 'fiscal_year']).reset_index(drop=True)
    base = panel.copy()
    base['prediction_date'] = pd.to_datetime(base.feature_month + '-01') + pd.offsets.MonthEnd(0)
    base['stock_id'] = base['stock_id'].astype(str)
    ev['stock_id'] = ev['stock_id'].astype(str)
    merged = pd.merge_asof(base.sort_values(['prediction_date', 'stock_id']), ev[['stock_id', 'fiscal_year', 'fiscal_date', 'proxy_available_date', 'debt_ratio_pct', 'source_type']].sort_values(['proxy_available_date', 'stock_id']), left_on='prediction_date', right_on='proxy_available_date', by='stock_id', direction='backward', allow_exact_matches=True)
    merged['debt_proxy_raw'] = merged['debt_ratio_pct']
    merged['fiscal_end_leak'] = merged['debt_proxy_raw'].notna() & (merged['fiscal_date'] > merged['prediction_date'])
    merged['proxy_date_leak'] = merged['debt_proxy_raw'].notna() & (merged['proxy_available_date'] > merged['prediction_date'])
    merged['debt_proxy_winsor'] = np.nan
    merged['debt_proxy_z'] = np.nan
    stats = []
    for month, g in merged.groupby('feature_month', sort=False):
        values = g.debt_proxy_raw.replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 2:
            stats.append({'feature_month': month, 'n_debt': len(values), 'lower': np.nan, 'upper': np.nan, 'mean': np.nan, 'sd': np.nan})
            continue
        lo, hi = values.quantile([q, 1 - q]).tolist()
        clipped = values.clip(lo, hi)
        mean = clipped.mean()
        sd = clipped.std(ddof=0)
        merged.loc[g.index, 'debt_proxy_winsor'] = g.debt_proxy_raw.clip(lo, hi)
        if sd > 0:
            merged.loc[g.index, 'debt_proxy_z'] = (merged.loc[g.index, 'debt_proxy_winsor'] - mean) / sd
        stats.append({'feature_month': month, 'n_debt': len(values), 'lower': lo, 'upper': hi, 'mean': mean, 'sd': sd})
    nobs = merged.debt_proxy_raw.notna().sum()
    features = merged[['stock_id', 'feature_month', 'target_month', 'prediction_date', 'debt_proxy_raw', 'debt_proxy_winsor', 'debt_proxy_z', 'fiscal_year', 'fiscal_date', 'proxy_available_date', 'source_type', 'fiscal_end_leak', 'proxy_date_leak']].copy()
    features['timing_status'] = np.where(features.debt_proxy_raw.notna(), 'PROXY_ONLY_NOT_EXACT_PIT', 'MISSING')
    features.to_csv(output / f'h2_debt_lag_{lag}m_2001_2021.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(stats).sort_values('feature_month').to_csv(output / f'debt_lag_{lag}m_monthly_stats.csv', index=False, encoding='utf-8-sig')
    return (features, {'lag_months': lag, 'rows': len(features), 'nonmissing': int(nobs), 'coverage': float(nobs / len(features)), 'leaks_fiscal_end': int(merged.fiscal_end_leak.sum()), 'leaks_proxy_date': int(merged.proxy_date_leak.sum()), 'first_nonmissing_feature_month': str(merged.loc[merged.debt_proxy_raw.notna(), 'feature_month'].min()) if nobs else None, 'no_actual_filing_verification': True})
