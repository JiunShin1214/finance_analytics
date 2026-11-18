# -*- coding: utf-8 -*-
"""H2 마지막 확장: 부채 4분위 × 통화정책의 추가 예측력 검증.

실행 (finance_analytics 프로젝트 루트):
 python -m src.h2.experiments.run_h2_debt_quartile --check-only
  python -m src.h2.experiments.run_h2_debt_quartile --years 2006-2007 --models Ridge,RandomForest
  python -m src.h2.experiments.run_h2_debt_quartile

필요: run_h2_paper_models.py 가 같은 폴더에 있을 것.
기존 H1/H3 결과, H2 원천 Hybrid 데이터를 읽기만 하며 변경하지 않음.
A: Size, B/M, Momentum, H3 EWMA (M0)
B: A + PolicyScore + 부채분위 더미(Q2,Q3,Q4; Q1 기준)
C: B + PolicyScore*각 분위 더미 (Q2,Q3,Q4)
D: A + Debt(z) + PolicyScore (기존 M3의 비교용 재현)
일관 비교를 위해 A~D는 동일 완전관측 학습 표본과 동일 평가 종목/월 사용.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .run_h2_paper_models import prepare, _build_model, _ic

MODELS=('Ridge','RandomForest','XGBoost','LightGBM')


def main():
    p=argparse.ArgumentParser(description='H2 부채 분위별 정책 반응 실험')
    p.add_argument('--root',default='.')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--years',default='2006-2021')
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--debt-source',default='hybrid',choices=['hybrid','base'])
    p.add_argument('--out',default=None)
    a=p.parse_args()
    root=Path(a.root).resolve()
    begin,end=map(int,a.years.split('-'))
    if not (2006<=begin<=end<=2021):raise ValueError('years 범위는 2006-2021')
    models=[x.strip() for x in a.models.split(',') if x.strip()]
    if not models or any(x not in MODELS for x in models):raise ValueError(f'모델: {MODELS}')
    d,base,h2,checks=prepare(root,debt_source=a.debt_source)
    if set(base)!={'size','bm','mom','sigma'} or len(base)!=4:
        raise ValueError(f'H1 B/M 또는 H3 원래 EWMA가 준비되지 않았습니다. 현재 기본 변수: {base}')
    if not {'policy','debt','debt_quartile'}.issubset(d.columns):
        raise ValueError('Hybrid Debt/Policy/Quartile 없음')
    q=pd.to_numeric(d['debt_quartile'],errors='coerce')
    if (q.dropna()%1!=0).any() or not q.dropna().isin([0,1,2,3]).all():
        raise ValueError('debt_quartile는 0,1,2,3이어야 합니다')
    # Q1을 기준범주로 하는 one-hot. 미관측 부채는 NaN으로 유지하여 공통 표본에서 제외.
    for k in (1,2,3):
        d[f'debt_q{k+1}']=(q==k).astype(float).where(q.notna())
        d[f'policy_x_q{k+1}']=d['policy']*d[f'debt_q{k+1}']
    quart=[f'debt_q{k}' for k in (2,3,4)]
    inter=[f'policy_x_q{k}' for k in (2,3,4)]
    groups=OrderedDict([
        ('A_baseline',list(base)),
        ('B_policy_quartile',list(base)+['policy']+quart),
        ('C_policy_quartile_interaction',list(base)+['policy']+quart+inter),
        ('D_continuous_debt_policy',list(base)+['debt','policy']),
    ])
    features=list(dict.fromkeys(f for cols in groups.values() for f in cols))
    # 학습집합은 A~D에 대해 동일. 결측은 검증 시 기존 모델 파이프라인의 학습중앙값으로 대체.
    shared=d['eligible'] & d[features+['y']].notna().all(axis=1)
    valid_mask=d['eligible'] & d['y'].notna()
    counts=(d.loc[d.eligible].groupby('debt_quartile').size().to_dict())
    stat={**checks, 'experiment':'H2 debt quartile policy interaction',
          'groups':groups,'models':models,'validation_years':a.years,
          'shared_complete_training_rows':int(shared.sum()),
          'validation_target_available_rows':int(valid_mask.sum()),
          'quartile_counts_eligible':{str(k):int(v) for k,v in counts.items()},
          'feature_missing_fraction_eligible':{f:float(d.loc[d.eligible,f].isna().mean()) for f in features},
          'debt_source':a.debt_source,
          'method_notes':['Q1 is reference; Q2-Q4 dummies + policy interactions',
                          'No unobserved policy shock; BOK policy text score is observational',
                          'A-D training and evaluation sample are paired',
                          'No target after 2021-12 used in training/validation',
                          'Test 2022-2025 untouched']}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(a.out).resolve() if a.out else root/'outputs'/'h2_debt_quartile'/stamp
    print('기준 변수:',base)
    print('모델:',models)
    print('Feature sets:',json.dumps(groups,ensure_ascii=False,indent=2))
    print('학습 공통 표본 후보:',int(shared.sum()))
    print('부채 4분위 분포:',counts)
    print('출력 예정:',out)
    if a.check_only: return
    if out.exists():raise FileExistsError(f'기존 결과 덮어쓰기 방지: {out}')
    monthly=[]; folds=[]
    for year in range(begin,end+1):
        tr=shared & d.target_month.lt(f'{year}-01')
        va=valid_mask & d.target_month.between(f'{year}-01',f'{year}-12')
        if tr.sum()<100 or va.sum()<50:raise RuntimeError(f'{year}: 학습 {tr.sum()} 검증 {va.sum()}')
        y_tr=d.loc[tr,'y'].to_numpy(float)
        ev=d.loc[va,['stock_id','target_month','y']].copy()
        folds.append({'year':year,'train_rows':int(tr.sum()),'eval_rows':int(va.sum()),
                      'months':int(ev.target_month.nunique())})
        for group,cols in groups.items():
            xtr=d.loc[tr,cols].to_numpy(float)
            xva=d.loc[va,cols].to_numpy(float)
            for model_name in models:
                model=_build_model(model_name)
                model.fit(xtr,y_tr)
                tmp=ev.copy(); tmp['pred']=model.predict(xva)
                for month,sub in tmp.groupby('target_month',sort=True):
                    monthly.append({'year':year,'model':model_name,'feature_set':group,
                                    'target_month':month,'n':len(sub),
                                    'ic':_ic(sub.pred.to_numpy(float),sub.y.to_numpy(float))})
            print(f'{year} {group} / 학습 {tr.sum():,} 검증 {va.sum():,}',flush=True)
    monthly=pd.DataFrame(monthly)
    summary=(monthly.groupby(['model','feature_set']).agg(
        months=('ic','count'),mean_rankic=('ic','mean'),std_ic=('ic','std'),
        median_rankic=('ic','median'),positive_ic_share=('ic',lambda s:float((s>0).mean())),
        eval_rows=('n','sum')).reset_index())
    summary['icir_monthly']=summary.mean_rankic/summary.std_ic.replace(0,np.nan)
    w=monthly.pivot(index=['model','target_month'],columns='feature_set',values='ic')
    pairs=[('B_policy_quartile','A_baseline','B_minus_A'),
           ('C_policy_quartile_interaction','B_policy_quartile','C_minus_B'),
           ('C_policy_quartile_interaction','A_baseline','C_minus_A'),
           ('B_policy_quartile','D_continuous_debt_policy','B_minus_D'),
           ('C_policy_quartile_interaction','D_continuous_debt_policy','C_minus_D')]
    diffs=[];summary_diffs=[]
    for added,ref,name in pairs:
        delta=(w[added]-w[ref]).dropna()
        for (m,month),v in delta.items():
            diffs.append({'comparison':name,'model':m,'target_month':month,'delta_ic':float(v)})
        for m,g in delta.groupby(level='model'):
            z=g.to_numpy(float)
            summary_diffs.append({'comparison':name,'model':m,'added_set':added,'reference_set':ref,
                                  'n_months':len(z),'mean_delta_ic':float(z.mean()),
                                  'median_delta_ic':float(np.median(z)),
                                  'positive_month_share':float((z>0).mean())})
    out.mkdir(parents=True)
    (out/'data_checks.json').write_text(json.dumps(stat,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    results={'feature_sets.csv':pd.DataFrame([{'feature_set':k,'features':','.join(v),'n_features':len(v)} for k,v in groups.items()]),
             'model_summary.csv':summary,'monthly_rankic.csv':monthly,
             'comparisons.csv':pd.DataFrame(summary_diffs),
             'monthly_paired_delta.csv':pd.DataFrame(diffs),
             'fold_coverage.csv':pd.DataFrame(folds)}
    for name,frame in results.items():frame.to_csv(out/name,index=False,encoding='utf-8-sig')
    print('\n=== 평균 RankIC ===\n',summary[['model','feature_set','mean_rankic']].to_string(index=False))
    print('\n=== 평균 ΔRankIC ===\n',results['comparisons.csv'][['comparison','model','mean_delta_ic']].to_string(index=False))
    print('\n완료:',out)

if __name__=='__main__':main()
