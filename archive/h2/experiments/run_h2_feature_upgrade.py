# -*- coding: utf-8 -*-
"""H2 추가변수 통합 실행: H1 B/M·Mom, H3 EWMA, (선택) 월별 금리변화.

수정 대상: 없음. 기존 소스·기존 산출물 덮어쓰지 않음.
새 파일인 recover_h2_missing_features.py, run_h2_paper_models.py,
run_h2_factorial_models.py와 함께 finance_analytics 루트에 둔다.

python -m src.h2.experiments.run_h2_feature_upgrade --check-only
python -m src.h2.experiments.run_h2_feature_upgrade --build-features --check-only
python -m src.h2.experiments.run_h2_feature_upgrade --years 2006-2007 --models Ridge,RandomForest
python -m src.h2.experiments.run_h2_feature_upgrade
python -m src.h2.experiments.run_h2_feature_upgrade --rate-file data/h2/base_rate_monthly.csv

월별 금리 파일: month,rate_change_pp (단위: %p; 예: -0.25), 선택: release_date.
금리변화가 없는 달도 0으로 명시하여 기준월 전 기간을 커버해야 한다.
수익률 관측 기간은 학습·검증 2021-12 타깃까지만 허용한다.
"""
from __future__ import annotations
import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from .recover_h2_missing_features import FILES, build_h1, build_h3, report
from .run_h2_paper_models import prepare
from .run_h2_factorial_models import calc_monthly_ic, design_sets, ALLOWED_MODELS

REQUIRED_BASE = ('size', 'bm', 'mom', 'sigma')
REQUIRED_H2 = ('debt', 'policy', 'policy_x_debt')
RATE_CANDIDATE = Path('data/h2/base_rate_monthly.csv')


def maybe_build(root: Path):
    """프로젝트의 순정 함수만 호출. 기존 결과 파일은 절대 수정하지 않음."""
    if not (root / FILES['panel']).is_file():
        raise FileNotFoundError(f'원본 월별 패널 없음: {root / FILES["panel"]}')
    for name, needs, creator in (
        ('h1_components', ('equity_asof',), build_h1),
        ('h3', ('eligibility',), build_h3),
    ):
        dest = root / FILES[name]
        if dest.exists():
            print(f'[유지] {dest.relative_to(root)} (기존 파일 수정 안 함)', flush=True)
            continue
        missing = [key for key in needs if not (root / FILES[key]).is_file()]
        if missing:
            print(f'[중단] {name} 생성에 필요한 기존 파일 없음: {[str(FILES[k]) for k in missing]}')
            continue
        print(f'[생성 시작] {name}: 기존 src/features 모듈 호출', flush=True)
        out = creator(root)
        if not {'stock_id', 'base_month'}.issubset(out):
            raise ValueError(f'{name}: 필수 키 누락')
        if out[['stock_id', 'base_month']].isna().any().any() or out.duplicated(['stock_id', 'base_month']).any():
            raise ValueError(f'{name}: 중복/누락 키')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(f'다른 작업이 생성한 파일 보호: {dest}')
        out.to_parquet(dest, index=False)
        print(f'[생성 완료] {dest.relative_to(root)}: {len(out):,}행', flush=True)


def extend_rate(data: pd.DataFrame, root: Path, rate_file: str | None):
    """정책금리: 기존 H2 종목-월 rate_change 사용, 또는 월별 외부 CSV를 정확히 결합."""
    details={'source':'unavailable', 'rate_groups_enabled':False}
    existing = 'rate_change' in data.columns and pd.to_numeric(data['rate_change'], errors='coerce').notna().any()
    if rate_file and existing:
        print('[안내] --rate-file 지정으로 기존 H2 rate_change 대신 제공 CSV를 사용합니다.')
    elif existing:
        r = pd.to_numeric(data['rate_change'], errors='coerce')
        checks = pd.DataFrame({'month':data.feature_month, 'rate':r}).groupby('month').rate
        if (checks.max()-checks.min()).fillna(0).abs().gt(1e-8).any():
            raise ValueError('기존 금리변화가 같은 월 종목 간 서로 다릅니다.')
        data['rate_change']=r
        data['rate_x_debt']=r*data.debt
        details={'source':'data/h2/h2_rate_change_panel_2001_2021.csv',
                 'rate_groups_enabled':True}
        return data, details
    path = Path(rate_file) if rate_file else RATE_CANDIDATE
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        return data, details
    raw=pd.read_csv(path, dtype={'month':'string'})
    if 'month' not in raw and 'feature_month' in raw:
        raw=raw.rename(columns={'feature_month':'month'})
    if not {'month','rate_change_pp'}.issubset(raw):
        raise ValueError(f'금리 월별 CSV에는 month, rate_change_pp가 필요합니다: {path}')
    raw['month']=raw['month'].astype('string').str[:7]
    if raw['month'].isna().any() or raw['month'].duplicated().any():
        raise ValueError('금리 월별 CSV: month 누락/중복')
    rates=pd.to_numeric(raw.rate_change_pp,errors='coerce')
    if rates.isna().any() or ~np.isfinite(rates.to_numpy()).all():
        raise ValueError('rate_change_pp에는 숫자(단위 %p)만 넣고, 동결 월은 0으로 명시해야 합니다.')
    raw['rate_change_pp']=rates
    if 'release_date' in raw:
        when=pd.to_datetime(raw.release_date,errors='coerce')
        end=pd.PeriodIndex(raw.month, freq='M').to_timestamp('M')
        if when.isna().any() or (when > end).any():
            raise ValueError('금리 발표일이 누락됐거나 해당 월말보다 늦습니다.')
    missing=sorted(set(data.feature_month.dropna())-set(raw.month))
    if missing:
        raise ValueError(f'금리 CSV가 전체 기준월을 커버하지 않습니다. 누락 월 {len(missing)}개 예: {missing[:12]}')
    rates=raw[['month','rate_change_pp']].rename(columns={'month':'feature_month',
                                                          'rate_change_pp':'_rate_new'})
    data=data.drop(columns=[c for c in ['rate_change','rate_x_debt'] if c in data],errors='ignore')
    data=data.merge(rates,on='feature_month',how='left',validate='many_to_one')
    data['rate_change']=data.pop('_rate_new')
    data['rate_x_debt']=data.rate_change*data.debt
    details={'source':str(path), 'rate_groups_enabled':True,
             'rate_months':int(raw.month.nunique()),'note':'rate_change_pp 단위 %p'}
    return data,details


def rate_groups(base_sets: OrderedDict, data: pd.DataFrame, enabled: bool):
    groups=OrderedDict(base_sets)
    if enabled:
        groups['M5_plus_rate']=groups['M4_interaction']+['rate_change']
        groups['M6_plus_rate_x_debt']=groups['M5_plus_rate']+['rate_x_debt']
    return groups


def extend_comparisons(monthly, summary, comparison, paired):
    name_pairs=[('M5_plus_rate','M4_interaction','Rate_given_H2'),
                ('M6_plus_rate_x_debt','M5_plus_rate','RateDebtInteraction')]
    if 'M5_plus_rate' not in set(monthly.feature_set):
        return comparison,paired
    wide=monthly.pivot(index=['model','target_month'],columns='feature_set',values='ic')
    rows=[];pairs=[]
    for added,reference,label in name_pairs:
        diff=(wide[added]-wide[reference]).dropna()
        for (model,month),v in diff.items():
            pairs.append({'comparison':label,'model':model,'target_month':month,'delta_ic':float(v)})
        for model,part in diff.groupby(level='model'):
            a=part.to_numpy(float)
            rows.append({'model':model,'comparison':label,'added_set':added,
                         'reference_set':reference,'n_months':len(a),
                         'mean_delta_ic':float(a.mean()),
                         'median_delta_ic':float(np.median(a)),
                         'positive_month_share':float((a>0).mean())})
    return pd.concat([comparison,pd.DataFrame(rows)],ignore_index=True), pd.concat([paired,pd.DataFrame(pairs)],ignore_index=True)


def main():
    parser=argparse.ArgumentParser(description='기존 H1/H3를 H2에 복구하고 정책/부채 효과 검증')
    parser.add_argument('--root',default='.')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--build-features',action='store_true',
                        help='누락된 h1_components/h3 파일만 생성; 기존 파일은 보존')
    parser.add_argument('--rate-file',default=None,help='month,rate_change_pp CSV (전 월 커버)')
    parser.add_argument('--require-rate',action='store_true',help='금리 데이터 없으면 실행 거부')
    parser.add_argument('--debt-source',default='hybrid',choices=['hybrid','base'])
    parser.add_argument('--years',default='2006-2021')
    parser.add_argument('--models',default=','.join(ALLOWED_MODELS))
    parser.add_argument('--out',default=None)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    if not (root/'src').is_dir():
        raise SystemExit('src/가 있는 finance_analytics 프로젝트 루트에서 실행하세요.')
    print('=== 입력 파일 점검 ===',flush=True)
    report(root)
    if args.build_features:
        maybe_build(root)
        report(root)
    missing=[x for x in ('h1_components','h3') if not (root/FILES[x]).is_file()]
    if missing:
        msg=('원래 H1/H3 결과 파일이 없음: '+', '.join(missing)
             +'\n먼저 --build-features --check-only 실행, asof/eligibility 파일 존재 확인.')
        if args.check_only:
            print('[검사 결과] '+msg)
            return
        raise FileNotFoundError(msg)
    d,base,h2,checks=prepare(root,debt_source=args.debt_source)
    if tuple(base)!=REQUIRED_BASE:
        raise ValueError(f'기존 H1 B/M 또는 H3 EWMA가 사용되지 않았습니다: {base}; 필요: {REQUIRED_BASE}')
    for c in REQUIRED_H2:
        if c not in d:raise ValueError(f'H2 Feature 누락: {c}')
    d,rate_info=extend_rate(d,root,args.rate_file)
    if args.require_rate and not rate_info['rate_groups_enabled']:
        raise FileNotFoundError('금리변화 데이터 없음. 월별 CSV를 --rate-file로 제공하세요.')
    if not rate_info['rate_groups_enabled']:
        print('[알림] RateChange 자료 없음: M5/M6는 실행하지 않고 M0~M4를 실행합니다.')
    groups=rate_groups(design_sets(base),d,rate_info['rate_groups_enabled'])
    features=list(dict.fromkeys(x for group in groups.values() for x in group))
    if not set(features).issubset(d):
        raise ValueError(f'모델 Feature 누락: {sorted(set(features)-set(d))}')
    val_years=args.years.split('-')
    if len(val_years)!=2:raise ValueError('--years YYYY-YYYY 형식')
    year_a,year_b=map(int,val_years)
    if year_a<2006 or year_b>2021 or year_a>year_b:
        raise ValueError('검증 기간은 2006~2021년 사이')
    models=[m.strip() for m in args.models.split(',') if m.strip()]
    if not models or any(m not in ALLOWED_MODELS for m in models):
        raise ValueError('허용 모델: '+str(ALLOWED_MODELS))
    required_train=d.eligible & d[features+['y']].notna().all(axis=1)
    stat={**checks, 'experiment_feature_sets':groups,'rate_data':rate_info,
          'complete_train_candidate_rows':int(required_train.sum()),
          'feature_missing_pct':{k:round(100*float(d.loc[d.eligible,k].isna().mean()),3) for k in features},
          'validation_years':args.years,'models_requested':models,
          'exact_h1_h3_required':True,
          'note':'금리 변화(%p)는 정책 Surprise가 아님. ROA/유동비율은 이번 코드에 없음.'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(args.out).resolve() if args.out else root/'outputs'/'h2_feature_upgrade'/stamp
    if out.exists():raise FileExistsError(f'기존 출력 보호: {out}')
    out.mkdir(parents=True)
    (out/'data_checks.json').write_text(json.dumps(stat,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    pd.DataFrame([{'feature_set':k,'variables':','.join(v),'n_features':len(v)} for k,v in groups.items()]).to_csv(out/'feature_sets.csv',index=False,encoding='utf-8-sig')
    print('입력 변수:',features,'\n모델:',models,'\n완전 관측 학습 후보:',int(required_train.sum()),'\n출력:',out,flush=True)
    if args.check_only:
        print('[검사 완료] --check-only 제거 시 학습·검증 실행')
        return
    monthly,summary,comparison,paired,folds=calc_monthly_ic(d,models,groups,year_a,year_b)
    comparison,paired=extend_comparisons(monthly,summary,comparison,paired)
    for name,obj in [('monthly_rankic.csv',monthly),('model_summary.csv',summary),
                     ('comparisons.csv',comparison),('monthly_paired_delta.csv',paired),
                     ('fold_coverage.csv',folds)]:
        obj.to_csv(out/name,index=False,encoding='utf-8-sig')
    print('=== 모델별 평균 RankIC ===')
    print(summary[['model','feature_set','mean_rankic','icir_monthly']].to_string(index=False))
    print('실행 완료:',out)

if __name__=='__main__':
    main()
