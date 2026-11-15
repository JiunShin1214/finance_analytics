"""Rebuild the historical evaluation panel from the preserved eligible universe.

The legacy debt feature is retained only for the baseline comparison. Lag proxies
are generated separately; Test labels never enter model fitting or selection.
"""
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
from src.h2.data_io import ROOT, PROCESSED, write_csv, write_json, stock_ids, assert_unique, sha256

def csv_boundary(frame):
    """Keep historical pandas float parsing at merged former script boundaries.

    The old pipeline wrote/read base, target and debt CSVs. Keeping these in-memory
    boundaries preserves its numerical representation without extra intermediate files.
    """
    return pd.read_csv(StringIO(frame.to_csv(index=False)),dtype={'stock_id':str},low_memory=False)

def workbook_months(path, required_stocks, min_month, max_month):
    raw = pd.read_excel(path,header=None,engine='openpyxl')
    codes = raw.iloc[8,1:].astype('string').str.extract(r'(\d{6})',expand=False)
    first = ~codes.duplicated() & codes.isin(required_stocks)
    positions = np.flatnonzero(first.to_numpy())
    dates = pd.to_datetime(raw.iloc[14:,0],errors='coerce').dt.strftime('%Y-%m')
    rows=[]
    for i, month in enumerate(dates):
        if not isinstance(month,str) or not min_month<=month<=max_month:
            continue
        vals=pd.to_numeric(raw.iloc[14+i,1:].iloc[positions],errors='coerce')
        for code,value in zip(codes.iloc[positions],vals):
            if pd.notna(value): rows.append((code,month,float(value)))
    result=pd.DataFrame(rows,columns=['stock_id','source_month','value'])
    assert_unique(result,['stock_id','source_month'])
    return result

def build(policy_path=None, output=None):
    output = Path(output) if output else PROCESSED/'dataset.csv'
    parquet=ROOT/'data/panel/panel.parquet'
    p=pd.read_parquet(parquet)
    p['stock_id']=stock_ids(p.stock_id)
    month=p.month.astype('string').str.strip()
    compact=month.str.fullmatch(r'\d{6}',na=False)
    month.loc[compact]=month.loc[compact].str[:4]+'-'+month.loc[compact].str[4:6]
    p['month']=pd.to_datetime(month,format='mixed',errors='raise').dt.to_period('M').astype(str)
    p['eligible']=p.eligible.astype('string').str.lower().isin(['true','1','1.0','yes','y','t','eligible'])
    assert_unique(p,['stock_id','month'])
    target=p[['stock_id','month','return']].rename(columns={'month':'target_month','return':'return_t1_raw'})
    f=p[p.eligible & p.month.between('2001-01','2021-12')].copy()
    f=f.rename(columns={'month':'feature_month','return':'return_t'})
    f['target_month']=(pd.PeriodIndex(f.feature_month,freq='M')+1).astype(str)
    f=f.merge(target,on=['stock_id','target_month'],how='left',validate='m:1')
    f=f[['stock_id','feature_month','target_month','eligible','return_t','return_t1_raw','market_cap','price','shares_outstanding','company_name','isin']]
    f=csv_boundary(f)
    missing=f.return_t1_raw.isna()
    returns=workbook_months(ROOT/'data/raw/수익률.xlsx',set(f.stock_id),'2001-01','2022-01')
    lookup=returns.set_index(['stock_id','source_month']).value/100.0
    f['return_dataguide_fallback']=np.nan
    f.loc[missing,'return_dataguide_fallback']=[lookup.get((s,m),np.nan) for s,m in zip(f.loc[missing,'stock_id'],f.loc[missing,'target_month'])]
    f['return_t1']=f.return_t1_raw.fillna(f.return_dataguide_fallback)
    f['return_t1_source']=np.select([f.return_t1_raw.notna(),f.return_dataguide_fallback.notna()],['panel','dataguide_fallback'],default='missing')
    f=csv_boundary(f)
    legacy=workbook_months(ROOT/'data/raw/부채비율.xlsx',set(f.stock_id),'2000-01','2021-12')
    legacy=legacy.rename(columns={'source_month':'debt_source_month','value':'debt_ratio_raw'})
    legacy['feature_month']=(pd.PeriodIndex(legacy.debt_source_month,freq='M')+3).astype(str)
    legacy=legacy[legacy.feature_month.between('2001-01','2021-12')].sort_values(['stock_id','feature_month','debt_source_month'])
    f=f.merge(legacy,on=['stock_id','feature_month'],how='left',validate='1:1')
    f['negative_equity_flag']=(f.debt_ratio_raw<0).astype(int)
    f['debt_ratio_winsor']=np.nan; f['debt_ratio_z']=np.nan
    for month,idx in f.groupby('feature_month').groups.items():
        values=pd.to_numeric(f.loc[idx,'debt_ratio_raw'],errors='coerce')
        valid=values[values>0]
        if len(valid)<2: continue
        win=valid.clip(valid.quantile(.01),valid.quantile(.99)); sd=win.std(ddof=1)
        f.loc[win.index,'debt_ratio_winsor']=win
        if pd.notna(sd) and sd!=0: f.loc[win.index,'debt_ratio_z']=(win-win.mean())/sd
    f=csv_boundary(f)
    f['_month_key']=f.feature_month
    policy=pd.read_csv(policy_path or PROCESSED/'policy/monthly.csv')
    assert_unique(policy,['month'])
    f=f.merge(policy[['month','final_rule_v2_strict','final_rule_v2_sensitivity']].rename(columns={'month':'_month_key'}),on='_month_key',how='left',validate='m:1')
    write_csv(f,output)
    write_json({'rows':len(f),'input_parquet_sha256':sha256(parquet),'feature_min':f.feature_month.min(),'feature_max':f.feature_month.max(),
                'target_min':f.target_month.min(),'target_max':f.target_month.max(),
                'note':'Legacy baseline includes 2022-01 targets; evaluation filters all target months after 2021-12 before fitting. Warmup parquet starts 2000; policy starts 2001.'},output.parent/'dataset_provenance.json')
    return output
