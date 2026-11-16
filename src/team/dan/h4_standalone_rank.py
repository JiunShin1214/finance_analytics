"""Exploratory H4-only expanding OLS and direct feature Rank IC; not team model_test."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import zipfile
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from .h4_epu import load_epu_index, align_epu_to_base_month, calculate_h4_epu_features


def monthly_ic(data, prediction):
    rows=[]
    for month,g in data.assign(prediction=prediction).groupby('target_month',sort=True):
        valid=g[['prediction','y']].replace([np.inf,-np.inf],np.nan).dropna()
        reason=''
        if len(valid)<3: reason='fewer_than_3_stocks'
        elif valid.prediction.nunique()<2 or valid.y.nunique()<2: reason='constant_ranks'
        ic=float(spearmanr(valid.prediction,valid.y).statistic) if not reason else np.nan
        rows.append(dict(target_month=month,n=len(valid),ic=ic,reason=reason))
    return pd.DataFrame(rows)


def main(root=Path.cwd()):
    root=Path(root).resolve()
    mom=pd.read_parquet(root/'data/raw/mom.parquet')
    panel=pd.read_parquet(root/'data/raw/panel.parquet')
    mom['base_month']=pd.PeriodIndex(mom.base_month,freq='M').astype(str)
    panel['month']=pd.PeriodIndex(panel.month,freq='M').astype(str)
    if panel.duplicated(['stock_id','month']).any(): raise ValueError('duplicate panel keys')
    mom=mom[mom.base_month.between('2000-12','2021-11')].copy()
    keys=mom.loc[mom.eligible.fillna(False)&mom.mom_available.fillna(False),['stock_id','base_month']]
    epu=align_epu_to_base_month(load_epu_index(root/'data/raw/Korea_Overall_EPU.xlsx'))
    data=calculate_h4_epu_features(mom,keys,epu)
    data['target_month']=(pd.PeriodIndex(data.base_month,freq='M')+1).astype(str)
    target=panel[panel.month.between('2001-01','2021-12')].rename(columns={'month':'target_month','return':'y'})
    data=data.merge(target[['stock_id','target_month','y']],on=['stock_id','target_month'],how='left',validate='one_to_one')
    before=len(data)
    data=data.replace([np.inf,-np.inf],np.nan).dropna(subset=['mom','epu_raw','y']).copy()
    data['year']=data.target_month.str[:4].astype(int)
    predictions=[]; folds=[]
    for year in range(2006,2022):
        train=data[data.year.between(2001,year-1)]
        valid=data[data.year.eq(year)].copy()
        months=train[['base_month','epu_raw']].drop_duplicates()
        mean=months.epu_raw.mean(); std=months.epu_raw.std(ddof=1)
        if len(train)==0 or len(valid)==0 or not np.isfinite(std) or std<=0: raise ValueError('invalid fold')
        def design(frame):
            z=(frame.epu_raw.to_numpy()-mean)/std
            return np.column_stack([frame.mom.to_numpy(),z,frame.mom.to_numpy()*z])
        X=design(train)
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(X)),X]))!=4: raise ValueError('singular fold')
        fit=LinearRegression().fit(X,train.y)
        valid['yhat']=fit.predict(design(valid))
        assert train.target_month.max()<valid.target_month.min()
        folds.append(dict(valid_year=year,train_first=train.target_month.min(),train_last=train.target_month.max(),train_n=len(train),valid_n=len(valid),epu_train_mean=mean,epu_train_std=std,intercept=float(fit.intercept_),coef_mom=float(fit.coef_[0]),coef_epu_z=float(fit.coef_[1]),coef_interaction=float(fit.coef_[2])))
        predictions.append(valid)
    validation=pd.concat(predictions,ignore_index=True)
    assert not validation.duplicated(['stock_id','target_month']).any()
    tables={}
    for label,col in [('h4_only_ols','yhat'),('direct_epu_times_mom','h4_interaction'),('mom_rank_diagnostic','mom')]:
        tables[label]=monthly_ic(validation,validation[col].to_numpy())
    for table in tables.values():
        assert len(table)==192 and table.ic.notna().all()
    # Independent average-rank Pearson implementation checks Spearman output.
    independent=[]
    for _,g in validation.groupby('target_month',sort=True):
        independent.append(g.yhat.rank(method='average').corr(g.y.rank(method='average')))
    np.testing.assert_allclose(independent,tables['h4_only_ols'].ic.to_numpy(),atol=1e-12,rtol=0)
    summary=[]
    for name,t in tables.items():
        summary.append(dict(evaluation=name,mean_rank_ic=float(t.ic.mean()),monthly_ic_std=float(t.ic.std(ddof=1)),positive_month_fraction=float(t.ic.gt(0).mean()),months=int(t.ic.notna().sum()),stock_months=int(t.n.sum())))
    summary=pd.DataFrame(summary)
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S')
    out=root/'outputs/model_results/dan'/(stamp+'_h4_standalone_rank_ols')
    out.mkdir(parents=True,exist_ok=False)
    files=['data/raw/mom.parquet','data/raw/panel.parquet','data/raw/Korea_Overall_EPU.xlsx','src/team/dan/h4_epu.py','src/team/dan/h4_standalone_rank.py']
    run=dict(author='dan',status='EXPLORATORY_STANDALONE_NOT_TEAM_MODEL_TEST',target_validation='2006-01 to 2021-12',folds=16,training='2001 through validation_year-1; annual expanding; no tuning',model='OLS intercept + raw MOM + train-month-standardized lagged EPU + MOM*EPU_z',primary_metric='equal-weight mean monthly Spearman; average ties; n>=3; constant excluded not zero',filters='base-month eligible & mom_available, finite MOM/EPU/y; no price/size filter; no winsorization; no imputation',eligible_momentum_rows=before,finite_train_validation_rows=len(data),epu_lag_months=1,epu_release_dates='unverified; one-month lag is a proxy assumption',test_evaluated=False,team_common_functions_used=False,official_result=False,notes=['Raw positive monthly EPU*MOM ranks can equal MOM ranks; direct score is diagnostic only.','OLS EPU main effect and MOM main effect retained; no H1 comparison.','Target-month returns unavailable are excluded; coverage differs from official team scoring universe.','No claim of incremental value beyond H1 or confirmed H4 hypothesis.'],versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow']},sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
    with zipfile.ZipFile(out/'result.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('run.json',json.dumps(run,ensure_ascii=False,indent=2,allow_nan=False))
        z.writestr('summary.csv',summary.to_csv(index=False))
        z.writestr('folds.csv',pd.DataFrame(folds).to_csv(index=False))
        for name,t in tables.items():z.writestr(name+'_monthly_ic.csv',t.to_csv(index=False))
        z.writestr('README.txt','Exploratory standalone H4 only; not official model_test or H1 comparison. No per-stock predictions or source data included. Run: python -m src.team.dan.h4_standalone_rank\n')
    summary.to_csv(out/'summary.csv',index=False)
    print(out/'result.zip')
    print(summary.to_string(index=False))
    print('Independent rank check passed; 16 folds, all 192 monthly ICs valid.')

if __name__=='__main__':main()
