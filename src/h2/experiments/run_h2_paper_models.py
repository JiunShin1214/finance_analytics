# -*- coding: utf-8 -*-
"""H2 논문 기반 Feature 확장 및 RankIC 검증 (독립 실행 스크립트).

사용:
    cd <finance_analytics 프로젝트 루트>
    python -m src.h2.experiments.run_h2_paper_models --check-only
    python -m src.h2.experiments.run_h2_paper_models --years 2006-2007
    python -m src.h2.experiments.run_h2_paper_models

원본 data/ 및 기존 outputs/ 내용을 절대 덮어쓰지 않습니다.
평가: 2001~검증연도 직전 학습, 2006~2021년 연 단위 expanding window.
2022~2025 테스트 데이터는 읽지도 않습니다.

핵심 변수는 Arin et al.(2025)의 정책×부채 이질성 아이디어 기반의 예측용 변형입니다.
Local Projection 및 정책 Surprise 재현은 아닙니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG = 42
COLS = ['stock_id', 'feature_month']
H2_CSV = Path('data/h2/h2_final_rule_v2_panel_2001_2021.csv')
HYBRID_CSV = Path('data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv')
H1_CSV = Path('data/processed/h1_components.parquet')
H3_CSV = Path('data/processed/h3.parquet')
PANEL_CSV = Path('data/interim/panel.parquet')
RATE_CSV = Path('data/h2/h2_rate_change_panel_2001_2021.csv')
POLICY_CSV = Path('data/h2/h2_final_policy_rule_v2_monthly_score_2001_2021.csv')


def _stock(s):
    return (s.astype('string').str.strip().str.replace(r'^A(?=\d{6}$)', '', regex=True)
            .str.replace(r'\.0$', '', regex=True).str.zfill(6))


def _mon(s):
    return s.astype('string').str.slice(0, 7)


def _check_keys(d, label):
    if not set(COLS).issubset(d.columns):
        raise ValueError(f'{label}: missing key columns: {set(COLS)-set(d.columns)}')
    if d[COLS].isna().any().any() or d.duplicated(COLS).any():
        raise ValueError(f'{label}: null/duplicate stock_id-feature_month keys')


def _numeric(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _end(s):
    return pd.PeriodIndex(s.astype(str), freq='M').to_timestamp('M')


def _join(d, other, cols, label):
    f = other[COLS + list(cols)].copy()
    f['stock_id'] = _stock(f['stock_id'])
    f['feature_month'] = _mon(f['feature_month'])
    _check_keys(f, label)
    return d.merge(f, on=COLS, how='left', validate='one_to_one')


def _winsor_z(group):
    s = _numeric(group)
    if s.notna().sum() < 3:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.quantile([.01, .99])
    clipped = s.clip(lower=lo, upper=hi)
    std = clipped.std(ddof=0)
    return (clipped-clipped.mean())/std if np.isfinite(std) and std > 0 else pd.Series(np.nan, index=s.index)


def _fallback_returns(d, root):
    """H1/H3 기존 특징 파일이 없을 때만 사용하는 간이 계산용 수익률 패널.

    월별 전체 달력으로 재색인하므로 거래 정지/결측 달을 건너뛴 롤링이 발생하지 않음.
    """
    if (root/PANEL_CSV).exists():
        p = pd.read_parquet(root/PANEL_CSV, columns=['stock_id', 'month', 'return'])
        p = p.rename(columns={'month':'feature_month','return':'return_t'})
        source = str(PANEL_CSV)
    else:
        p = d[COLS + ['return_t']].copy()
        source = str(H2_CSV)
    p['stock_id'] = _stock(p['stock_id'])
    p['feature_month'] = _mon(p['feature_month'])
    p = p[p.feature_month.between('2000-01', '2021-11')]
    _check_keys(p, 'returns_panel')
    months = pd.period_range('2000-01', '2021-11', freq='M').astype(str)
    wide = p.pivot(index='feature_month', columns='stock_id', values='return_t').reindex(months)
    wide = wide.apply(pd.to_numeric, errors='coerce')
    v = wide.where((wide > -1) & np.isfinite(wide))  # 복리 수익률은 -100% 초과만 유효
    # t-11 ~ t-1: 직전 11개월; t 당월 수익률 제외
    mom = (1+v).shift(1).rolling(11, min_periods=11).apply(np.prod, raw=True)-1
    # 간이 6개월 변동성: t 당월까지 관측 가능한 수익률(기존 H3 EWMA와는 다른 값)
    vol = wide.rolling(6, min_periods=6).std(ddof=1)
    out = pd.DataFrame({
        'stock_id': np.tile(wide.columns.to_numpy(), len(months)),
        'feature_month': np.repeat(months.to_numpy(), len(wide.columns)),
        'mom_fallback': mom.to_numpy().ravel(),
        'sigma_fallback': vol.to_numpy().ravel(),
    })
    return out, source


def prepare(root: Path, debt_source='hybrid'):
    path = root/H2_CSV
    if not path.is_file():
        raise FileNotFoundError(f'기본 H2 패널이 없습니다: {path}')
    d = pd.read_csv(path, dtype={'stock_id':'string', 'feature_month':'string', 'target_month':'string'},low_memory=False)
    need = {'stock_id','feature_month','target_month','eligible','return_t','return_t1','market_cap','final_rule_v2_sensitivity'}
    if not need.issubset(d):
        raise ValueError(f'H2 패널 필수 열 부족: {sorted(need-set(d))}')
    d['stock_id'] = _stock(d.stock_id)
    d['feature_month'] = _mon(d.feature_month)
    d['target_month'] = _mon(d.target_month)
    _check_keys(d, 'h2_base')
    expected = (pd.PeriodIndex(d.feature_month, freq='M') + 1).astype(str)
    if not (expected == d.target_month.to_numpy()).all():
        raise ValueError('target_month != feature_month + 1 (시점 정렬 실패)')
    # 테스트 기간(2022~2025)에는 접근하지 않음: 목표월 기준 차단.
    d = d[d.target_month.between('2001-01','2021-12')].copy()
    d['eligible'] = d.eligible.astype('string').str.lower().isin(['true','1','yes'])
    d['y'] = _numeric(d.return_t1)
    d['size'] = np.log(_numeric(d.market_cap).where(_numeric(d.market_cap)>0))
    d['policy'] = _numeric(d.final_rule_v2_sensitivity)
    if d.policy.isna().any() or not d.policy.between(-1,1).all():
        raise ValueError('PolicyScore missing/out of range')

    # 프로젝트에서 정의한 발표일 < 해당 기준월 마지막 날 조건 확인.
    policy_path = root/POLICY_CSV
    if policy_path.exists():
        q = pd.read_csv(policy_path, dtype={'month':'string'},usecols=['month','sensitivity_source_release_date'])
        if q.month.duplicated().any(): raise ValueError('duplicate policy month')
        release = pd.to_datetime(q.sensitivity_source_release_date,errors='coerce')
        if release.isna().any() or (release >= _end(q.month)).any():
            raise ValueError('policy release at/after cutoff or missing')

    if debt_source == 'hybrid':
        hybrid_path = root/HYBRID_CSV
        if not hybrid_path.is_file():
            raise FileNotFoundError(f'Hybrid 원천 파일 없음: {hybrid_path}; 대안: --debt-source base')
        h = pd.read_csv(hybrid_path,dtype={'stock_id':'string','feature_month':'string'},low_memory=False)
        required={'stock_id','feature_month','debt_final_raw','source_available_date','source_fiscal_date'}
        if not required.issubset(h):
            raise ValueError('Hybrid 열 누락: '+str(sorted(required-set(h))))
        h['stock_id'] = _stock(h.stock_id)
        h['feature_month'] = _mon(h.feature_month)
        _check_keys(h,'hybrid_debt')
        active = _numeric(h.debt_final_raw).notna()
        for c in ('source_available_date','source_fiscal_date'):
            when = pd.to_datetime(h[c], errors='coerce')
            end = _end(h.feature_month)
            if (active & (when.isna()|(when>end))).any():
                raise ValueError(f'Hybrid 미래 정보 또는 공개일 누락: {c}')
        if 'fiscal_end_leak' in h and h.loc[active,'fiscal_end_leak'].astype('string').str.lower().isin(['true','1']).any():
            raise ValueError('Hybrid fiscal_end_leak=True')
        h['debt_raw'] = _numeric(h.debt_final_raw)
        d = _join(d,h,['debt_raw'],'hybrid_debt')
        masked = d.debt_raw.where(d.eligible)
        d['debt'] = masked.groupby(d.feature_month).transform(_winsor_z)
    else:
        for c in ('debt_ratio_raw','debt_ratio_z'):
            if c not in d: raise ValueError('base debt missing '+c)
        d['debt_raw'] = _numeric(d.debt_ratio_raw)
        d['debt'] = _numeric(d.debt_ratio_z)

    d['policy_x_debt'] = d.policy * d.debt
    # 동일 월의 기업 간 부채수준은 미래 수익률과 독립적으로 결정.
    ranks = d.debt_raw.where(d.eligible).groupby(d.feature_month).rank(pct=True, method='average')
    d['debt_quartile'] = np.floor((ranks-.000000001)*4).clip(0,3)

    # H1/H3 최종 산출 특징이 있으면 반드시 우선 사용.
    diagnostics={'base_rows':int(len(d)), 'base_stocks':int(d.stock_id.nunique()),
                 'base_months':int(d.feature_month.nunique()), 'debt_source':debt_source,
                 'source_paths':{'base':str(H2_CSV), 'debt':str(HYBRID_CSV if debt_source=='hybrid' else H2_CSV)},
                 'feature_sources':{}}
    if (root/H1_CSV).is_file():
        h1=pd.read_parquet(root/H1_CSV)
        needed={'stock_id','base_month'}
        if not needed.issubset(h1): raise ValueError('h1_components.parquet 키 누락')
        h1=h1.rename(columns={'base_month':'feature_month'})
        use=[x for x in ('bm','mom','size') if x in h1.columns]
        h1=h1[COLS+use]
        d = _join(d,h1, [x for x in use if x!='size'], 'h1_components')
        diagnostics['feature_sources']['bm_mom']='data/processed/h1_components.parquet'
    else:
        diagnostics['feature_sources']['bm_mom']='not found; fallback mom; no B/M'
    if (root/H3_CSV).is_file():
        h3=pd.read_parquet(root/H3_CSV)
        if not {'stock_id','base_month','sigma'}.issubset(h3):
            raise ValueError('h3.parquet required columns missing')
        h3=h3.rename(columns={'base_month':'feature_month'})
        d=_join(d,h3,['sigma'],'h3')
        diagnostics['feature_sources']['sigma']='data/processed/h3.parquet (EWMA)'
    else:
        diagnostics['feature_sources']['sigma']='not found; fallback vol6'

    if 'mom' not in d or 'sigma' not in d:
        fall, source = _fallback_returns(d, root)
        diagnostics['feature_sources']['fallback_returns']=source
        if 'mom' not in d:
            d=_join(d,fall,['mom_fallback'],'momentum_fallback')
            d['mom']=d.pop('mom_fallback')
        if 'sigma' not in d:
            d=_join(d,fall,['sigma_fallback'],'sigma_fallback')
            d['sigma']=d.pop('sigma_fallback')

    # 선택: 기존 금리변화 패널이 로컬에 있을 때만 사용.
    if (root/RATE_CSV).is_file():
        rt=pd.read_csv(root/RATE_CSV, dtype={'stock_id':'string','feature_month':'string'},low_memory=False)
        if {'stock_id','feature_month','rate_change_pp'}.issubset(rt):
            rt['stock_id']=_stock(rt.stock_id);rt['feature_month']=_mon(rt.feature_month)
            _check_keys(rt,'rate_change_panel')
            d=_join(d,rt,['rate_change_pp'],'rate_change')
            d['rate_change']=_numeric(d.pop('rate_change_pp'))
            d['rate_x_debt']=d.rate_change * d.debt
            diagnostics['feature_sources']['rate']='data/h2/h2_rate_change_panel_2001_2021.csv'

    # 이용 가능한 실제 열만 baseline에 넣음. B/M이 전혀 없으면 자동 생략.
    base=['size','mom','sigma']
    if 'bm' in d and _numeric(d.bm).notna().sum()>0:
        base.insert(1,'bm')
    h2=['debt','policy','policy_x_debt','debt_quartile']
    if 'rate_x_debt' in d:
        h2+=['rate_change','rate_x_debt']
    for c in base+h2:
        d[c]=_numeric(d[c])
    diagnostics['baseline_features']=base
    diagnostics['h2_features']=h2
    diagnostics['missing_all_rows']={c:round(float(d[c].isna().mean()),5) for c in base+h2+['y']}
    diagnostics['missing_eligible_rows']={c:round(float(d.loc[d.eligible,c].isna().mean()),5) for c in base+h2+['y']}
    diagnostics['eligible_rows']=int(d.eligible.sum())
    diagnostics['complete_train_candidates']=int((d.eligible & d[base+h2+['y']].notna().all(axis=1)).sum())
    diagnostics['source_available_until']='2021-12 target month only'
    return d, base, h2, diagnostics


def _build_model(name):
    if name=='Ridge':
        return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('reg',Ridge(alpha=10.0))])
    if name=='RandomForest':
        return Pipeline([('impute',SimpleImputer(strategy='median')),
                         ('reg',RandomForestRegressor(n_estimators=100,max_depth=8,
                             min_samples_leaf=30,n_jobs=-1,random_state=RNG))])
    if name=='XGBoost':
        from xgboost import XGBRegressor
        return Pipeline([('impute',SimpleImputer(strategy='median')),
                         ('reg',XGBRegressor(n_estimators=250,max_depth=3,learning_rate=.05,
                             subsample=.8,colsample_bytree=.8,objective='reg:squarederror',
                             tree_method='hist',n_jobs=-1,random_state=RNG))])
    if name=='LightGBM':
        from lightgbm import LGBMRegressor
        return Pipeline([('impute',SimpleImputer(strategy='median')),
                         ('reg',LGBMRegressor(n_estimators=250,num_leaves=15,max_depth=5,
                             learning_rate=.05,min_child_samples=60,n_jobs=-1,random_state=RNG,verbosity=-1))])
    raise ValueError(name)


def _ic(p, y):
    if len(p)<10 or np.nanstd(p)==0 or np.nanstd(y)==0:return np.nan
    return float(spearmanr(p,y).statistic)


def fit_evaluate(d,base,h2,start_year,end_year,models):
    # 학습시 모두 동일한 완전 관측 표본 / 검증시 목표수익률 존재하는 적격 종목 전체.
    shared=base+h2
    observed=d[shared+['y']].notna().all(axis=1) & d.eligible
    eligible_eval=d.eligible & d.y.notna()
    output=[]; folds=[]
    for yr in range(start_year,end_year+1):
        train=d.target_month.lt(f'{yr}-01') & observed
        valid=d.target_month.between(f'{yr}-01',f'{yr}-12') & eligible_eval
        ntrain=int(train.sum());nval=int(valid.sum())
        if ntrain<100 or nval<50: raise RuntimeError(f'{yr}: train/validation too small: {ntrain}/{nval}')
        folds.append({'year':yr,'train_rows':ntrain,'validation_rows':nval,
                      'validation_months':int(d.loc[valid,'target_month'].nunique())})
        for group,features in (('baseline',base),('H2_extended',shared)):
            Xtrain=d.loc[train,features].to_numpy(float)
            ytrain=d.loc[train,'y'].to_numpy(float)
            Xval=d.loc[valid,features].to_numpy(float)
            v=d.loc[valid,['stock_id','target_month','y']].copy()
            for modelname in models:
                model=_build_model(modelname)
                model.fit(Xtrain,ytrain)
                v['prediction']=model.predict(Xval)
                for m,part in v.groupby('target_month',sort=True):
                    output.append({'model':modelname,'feature_set':group,'year':yr,'target_month':m,
                                   'n':len(part),'ic':_ic(part.prediction.to_numpy(),part.y.to_numpy())})
            print(f'  {yr} {group}: {ntrain:,} train / {nval:,} eval, models={len(models)}',flush=True)
    results=pd.DataFrame(output)
    if results.empty:raise RuntimeError('no evaluated rows')
    summary=(results.groupby(['feature_set','model'],sort=True)
             .agg(months=('ic','count'),mean_rankic=('ic','mean'),std_monthly_ic=('ic','std'),
                  median_rankic=('ic','median'),positive_ic_share=('ic',lambda s:(s>0).mean()),
                  eval_rows=('n','sum')).reset_index())
    summary['icir_monthly']=summary.mean_rankic / summary.std_monthly_ic.replace(0,np.nan)
    # 두 그룹이 같은 검증 대상에 대해 비교되는지 먼저 검증 후 차이 계산.
    a=results.pivot(index=['model','target_month'],columns='feature_set',values='ic')
    if a[['baseline','H2_extended']].dropna().shape[0]==0:raise RuntimeError('no aligned monthly IC pairs')
    a['delta_ic_H2_minus_base']=a['H2_extended']-a['baseline']
    delta=a.groupby(level=0)['delta_ic_H2_minus_base'].agg(['mean','median','count']).rename(
        columns={'mean':'mean_delta_ic','median':'median_delta_ic','count':'n_comparable_months'}).reset_index()
    return results,summary,delta,pd.DataFrame(folds)


def _write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.',help='finance_analytics 프로젝트 루트')
    ap.add_argument('--out',default=None,help='출력 폴더. 기본: outputs/h2_paper_features/<UTC 타임스탬프>')
    ap.add_argument('--debt-source',choices=['hybrid','base'],default='hybrid')
    ap.add_argument('--years',default='2006-2021',help='검증기간. 예: 2006-2007')
    ap.add_argument('--models',default='Ridge,RandomForest,XGBoost,LightGBM')
    ap.add_argument('--check-only',action='store_true',help='데이터 검증/Feature 커버리지까지만')
    args=ap.parse_args()
    root=Path(args.root).resolve()
    year_parts=args.years.split('-')
    if len(year_parts)!=2:raise ValueError('--years는 2006-2021 형식')
    start,end=map(int,year_parts)
    if start<2006 or end>2021 or start>end: raise ValueError('검증기간은 2006~2021만 허용')
    models=[m.strip() for m in args.models.split(',') if m.strip()]
    if not models or any(m not in ('Ridge','RandomForest','XGBoost','LightGBM') for m in models):
        raise ValueError('unknown models')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=Path(args.out).resolve() if args.out else root/'outputs'/'h2_paper_features'/stamp
    if out.exists():
        raise FileExistsError(f'출력 폴더가 이미 있음 (덮어쓰기 방지): {out}')
    d,base,h2,checks=prepare(root,debt_source=args.debt_source)
    out.mkdir(parents=True)
    checks['models_requested']=models
    checks['validation_years']=args.years
    checks['check_only']=args.check_only
    _write_json(out/'data_checks.json',checks)
    print(f'[점검 완료] {checks["base_rows"]:,} rows, {checks["eligible_rows"]:,} eligible, {checks["complete_train_candidates"]:,} complete candidates')
    print(f'  baseline={base}\n  H2={h2}\n  output={out}',flush=True)
    if args.check_only:
        print('데이터 점검만 실행했습니다. 모델 결과는 생성되지 않았습니다.')
        return
    monthly,summary,delta,folds=fit_evaluate(d,base,h2,start,end,models)
    monthly.to_csv(out/'monthly_rankic.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(out/'model_summary.csv',index=False,encoding='utf-8-sig')
    delta.to_csv(out/'h2_incremental_rankic.csv',index=False,encoding='utf-8-sig')
    folds.to_csv(out/'fold_coverage.csv',index=False,encoding='utf-8-sig')
    print('\n=== MODEL SUMMARY ===\n',summary.to_string(index=False))
    print('\n=== H2 INCREMENTAL RANKIC ===\n',delta.to_string(index=False))
    print('\n완료:',out)

if __name__=='__main__':
    main()
