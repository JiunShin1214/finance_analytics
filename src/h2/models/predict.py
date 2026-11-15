"""Fixed Train-only model and label-free 2022–2025 prediction interface."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.h2.data_io import ROOT, PROCESSED, write_csv, write_json, stock_ids, assert_unique, sha256
from src.h2.models.estimators import model_instance
from src.h2.features.transforms import monthly_standardize, FEATURES

FEATURE_COLUMNS = ['stock_id','feature_month','market_cap','policy','debt_raw',
                   'policy_release_date','policy_date_source','fiscal_date','proxy_available_date']

def training_frame(panel_path=None, lag='6m'):
    p=pd.read_csv(panel_path or PROCESSED/'dataset.csv',dtype={'stock_id':'string'})
    # Filter targets BEFORE any training statistics, sample selection or fitting.
    p=p[p.target_month.between('2001-01','2021-12')].copy()
    p['stock_id']=stock_ids(p.stock_id)
    debt=pd.read_csv(PROCESSED/f'debt/h2_debt_lag_{lag}_fiscal_screened_2001_2021.csv',dtype={'stock_id':'string'})
    p=p.merge(debt[['stock_id','feature_month','proxy_masked']],on=['stock_id','feature_month'],how='left',validate='1:1')
    p['debt']=monthly_standardize(p,'proxy_masked')
    p['policy']=pd.to_numeric(p.final_rule_v2_sensitivity,errors='coerce')
    p['log_size']=np.log(pd.to_numeric(p.market_cap,errors='coerce').where(p.market_cap>0))
    p['policy_x_debt']=p.policy*p.debt
    ready=p[FEATURES['policy_debt_size']+['return_t1']].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)
    if 'eligible' in p: ready &= p.eligible.astype('string').str.lower().isin(['true','1'])
    return p.loc[ready].copy()

def prediction_features(path, lag='6m'):
    # Reject target-bearing input rather than silently accepting a Test label file.
    columns=list(pd.read_csv(path,nrows=0).columns)
    if set(columns)!=set(FEATURE_COLUMNS):
        raise ValueError(f'Prediction input must contain exactly these label-free columns: {FEATURE_COLUMNS}; received {columns}')
    p=pd.read_csv(path,dtype={'stock_id':'string','feature_month':'string'})
    p['stock_id']=stock_ids(p.stock_id)
    assert_unique(p,['stock_id','feature_month'])
    periods=pd.PeriodIndex(p.feature_month,freq='M')
    p['target_month']=(periods+1).astype(str)
    expected=set(pd.period_range('2022-01','2025-12',freq='M').astype(str))
    if set(p.target_month)!=expected:
        raise ValueError('Prediction features must cover exactly 48 target months, 2022-01 through 2025-12')
    cutoff=pd.Series(periods.to_timestamp('M'),index=p.index)
    for col in ['policy_release_date','fiscal_date','proxy_available_date']:
        d=pd.to_datetime(p[col],errors='coerce')
        if d.isna().any() or (d>cutoff).any(): raise ValueError(f'Missing or future {col}')
    if (pd.to_datetime(p.policy_release_date)>=cutoff).any():
        raise ValueError('Policy publication on month-end is only usable next month')
    if not p.policy_date_source.isin(['verified_document_header','legacy_same_day_inferred']).all():
        raise ValueError('Unknown policy release-date assurance')
    fiscal=pd.to_datetime(p.fiscal_date)
    if not ((fiscal.dt.month==12)&(fiscal.dt.day==31)).all():
        raise ValueError('Annual lag proxy requires screened December fiscal years')
    assumed=fiscal+pd.DateOffset(months=int(lag[:-1]))+pd.offsets.MonthEnd(0)
    if not pd.to_datetime(p.proxy_available_date).eq(assumed).all():
        raise ValueError('Availability dates do not match the configured annual lag')
    reference=pd.read_csv(PROCESSED/'disclosures/nonstandard_period_filings_reference.csv',dtype={'stock_code':'string'})
    risk=reference[reference.report_type.eq('사업보고서') & pd.to_numeric(reference.fiscal_month_label).ne(12) & pd.to_numeric(reference.fiscal_year_label).between(2000,2025)]
    if p.stock_id.isin(stock_ids(risk.stock_code)).any():
        raise ValueError('Prediction contains an issuer excluded by the fiscal-period screen')
    for c in ['market_cap','policy','debt_raw']: p[c]=pd.to_numeric(p[c],errors='coerce')
    if p.groupby('feature_month').policy.nunique().ne(1).any() or (~p.policy.between(-1,1)).any():
        raise ValueError('Policy must be one valid document score per month')
    p['debt']=monthly_standardize(p,'debt_raw')
    p['log_size']=np.log(p.market_cap.where(p.market_cap>0))
    p['policy_x_debt']=p.policy*p.debt
    if not np.isfinite(p[FEATURES['policy_debt_size']].to_numpy(dtype=float)).all():
        raise ValueError('Prediction features contain missing/nonfinite values or degenerate monthly debt')
    return p

def predict(features=None, model_name='OLS', lag='6m', output=None):
    output=Path(output) if output else ROOT/'results/predictions'
    output.mkdir(parents=True,exist_ok=True)
    train=training_frame(lag=lag)
    if train.empty: raise ValueError('Empty training sample')
    columns=FEATURES['policy_debt_size']
    model=model_instance(model_name)
    model.fit(train[columns],train.return_t1)
    joblib.dump(model,output/'model.joblib')
    status={'model':model_name,'lag':lag,'features':columns,'selection':'fixed specification; no Test-target tuning',
            'train_rows':len(train),'train_target_min':train.target_month.min(),'train_target_max':train.target_month.max(),
            'sample':'available scenario sample','train_input_sha256':sha256(PROCESSED/'dataset.csv'),
            'pit_status':'PROXY_ONLY_NOT_EXACT_PIT','test_labels_used':False}
    feature_path=Path(features) if features else ROOT/'data/raw/prediction_features.csv'
    if not feature_path.is_file():
        status.update(status='BLOCKED_MISSING_TEST_FEATURES',prediction_generated=False,
                      required_input=str(feature_path),required_columns=FEATURE_COLUMNS,
                      reason='No complete 2022–2025 policy/debt/eligibility feature table. Raw Project.zip Test returns are never opened.')
        write_json(status,output/'prediction_status.json')
        return status
    try:
        p=prediction_features(feature_path,lag)
    except (ValueError,KeyError) as exc:
        status.update(status='BLOCKED_INVALID_TEST_FEATURES',prediction_generated=False,reason=str(exc))
        write_json(status,output/'prediction_status.json')
        raise
    result=p[['stock_id','target_month']].copy()
    result['prediction']=model.predict(p[columns])
    if not np.isfinite(result.prediction).all(): raise ValueError('Nonfinite model prediction')
    result=result.sort_values(['target_month','stock_id']).reset_index(drop=True)
    write_csv(result,output/'prediction.csv')
    status.update(status='PREDICTIONS_GENERATED_PROXY_NOT_PIT',prediction_generated=True,prediction_rows=len(result),
                  features_sha256=sha256(feature_path),prediction_sha256=sha256(output/'prediction.csv'))
    write_json(status,output/'prediction_status.json')
    return status
