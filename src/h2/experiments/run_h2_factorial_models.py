# -*- coding: utf-8 -*-
"""H2 변수 추가 효과: 기본 / 부채 / 정책 / 부채+정책 / 명시적 상호작용 비교.

기존 run_h2_paper_models.py 옆에 저장 후 프로젝트 루트에서 실행.
사용 예시:
  python -m src.h2.experiments.run_h2_factorial_models --check-only
  python -m src.h2.experiments.run_h2_factorial_models --years 2006-2007 --models Ridge,RandomForest
  python -m src.h2.experiments.run_h2_factorial_models

전체 실험은 target_month가 2021-12 이하인 데이터만 사용합니다.
기본 원본 파일을 수정하지 않으며, 출력은 새 실행시각 폴더로 분리합니다.
"""

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .run_h2_paper_models import prepare, _build_model, _ic


ALLOWED_MODELS = ('Ridge', 'RandomForest', 'XGBoost', 'LightGBM')


def design_sets(base):
    """부채 4분위는 본 실험에서 제외하고 확장 실험에 사용."""
    return OrderedDict([
        ('M0_base', list(base)),
        ('M1_debt', list(base) + ['debt']),
        ('M2_policy', list(base) + ['policy']),
        ('M3_debt_policy', list(base) + ['debt', 'policy']),
        ('M4_interaction', list(base) + ['debt', 'policy', 'policy_x_debt']),
    ])


def calc_monthly_ic(data, models, groups, start_year, end_year):
    """모든 그룹에 똑같은 train/validation 기업-월을 적용."""
    features_all = list(dict.fromkeys(c for cs in groups.values() for c in cs))
    train_eligible = data.eligible & data[features_all + ['y']].notna().all(axis=1)
    eval_eligible = data.eligible & data.y.notna()
    rows = []
    folds = []

    for year in range(start_year, end_year + 1):
        train = train_eligible & data.target_month.lt(f'{year}-01')
        valid = eval_eligible & data.target_month.between(f'{year}-01', f'{year}-12')
        n_train, n_valid = int(train.sum()), int(valid.sum())
        if n_train < 100 or n_valid < 50:
            raise RuntimeError(f'{year}: 학습/검증 표본 부족: {n_train}/{n_valid}')
        val_base = data.loc[valid, ['stock_id', 'target_month', 'y']].copy()
        y_train = data.loc[train, 'y'].to_numpy(float)
        folds.append({'year': year, 'train_rows': n_train,
                      'eval_rows': n_valid,
                      'months': int(val_base.target_month.nunique())})
        for group_name, cols in groups.items():
            X_train = data.loc[train, cols].to_numpy(float)
            X_val = data.loc[valid, cols].to_numpy(float)
            for model_name in models:
                model = _build_model(model_name)
                model.fit(X_train, y_train)
                val = val_base.copy()
                val['pred'] = model.predict(X_val)
                for month, part in val.groupby('target_month', sort=True):
                    rows.append({'year': year, 'model': model_name,
                                 'feature_set': group_name, 'target_month': month,
                                 'n': len(part),
                                 'ic': _ic(part.pred.to_numpy(float), part.y.to_numpy(float))})
            print(f'{year} / {group_name}: {n_train:,} train / {n_valid:,} valid', flush=True)

    monthly = pd.DataFrame(rows)
    summary = (monthly.groupby(['model', 'feature_set'], sort=True)
               .agg(months=('ic', 'count'), mean_rankic=('ic', 'mean'),
                    std_ic=('ic', 'std'), median_rankic=('ic', 'median'),
                    positive_ic_share=('ic', lambda s: float((s > 0).mean())),
                    eval_rows=('n', 'sum')).reset_index())
    summary['icir_monthly'] = summary.mean_rankic / summary.std_ic.replace(0, np.nan)

    diffs = [
        ('M1_debt', 'M0_base', 'Debt_vs_Base'),
        ('M2_policy', 'M0_base', 'Policy_vs_Base'),
        ('M3_debt_policy', 'M0_base', 'DebtPolicy_vs_Base'),
        ('M3_debt_policy', 'M1_debt', 'Policy_given_Debt'),
        ('M3_debt_policy', 'M2_policy', 'Debt_given_Policy'),
        ('M4_interaction', 'M3_debt_policy', 'Explicit_Interaction_Effect'),
        ('M4_interaction', 'M0_base', 'FullH2_vs_Base'),
    ]
    wide = monthly.pivot(index=['model', 'target_month'], columns='feature_set', values='ic')
    comparisons = []
    paired_monthly = []
    for added, reference, contrast in diffs:
        delta = (wide[added] - wide[reference]).dropna()
        if delta.empty:
            raise RuntimeError(f'비교 표본 없음: {contrast}')
        for (model_name, month), value in delta.items():
            paired_monthly.append({'comparison': contrast, 'model': model_name,
                                   'target_month': month, 'delta_ic': float(value)})
        for model_name, part in delta.groupby(level='model'):
            vals = part.to_numpy(dtype=float)
            comparisons.append({'model': model_name,
                                'comparison': contrast,
                                'added_set': added,
                                'reference_set': reference,
                                'n_months': int(len(vals)),
                                'mean_delta_ic': float(np.mean(vals)),
                                'median_delta_ic': float(np.median(vals)),
                                'positive_month_share': float(np.mean(vals > 0))})

    return monthly, summary, pd.DataFrame(comparisons), pd.DataFrame(paired_monthly), pd.DataFrame(folds)


def main():
    p = argparse.ArgumentParser(description='H2 5개 Feature 구성 비교 실험')
    p.add_argument('--root', default='.', help='finance_analytics 프로젝트 루트')
    p.add_argument('--years', default='2006-2021', help='예: 2006-2007')
    p.add_argument('--models', default=','.join(ALLOWED_MODELS))
    p.add_argument('--debt-source', choices=['hybrid', 'base'], default='hybrid')
    p.add_argument('--out', default=None, help='새 출력 폴더')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    start, end = map(int, args.years.split('-'))
    if start < 2006 or end > 2021 or start > end:
        raise ValueError('검증기간은 2006~2021만 허용')
    models = [s.strip() for s in args.models.split(',') if s.strip()]
    if not models or any(m not in ALLOWED_MODELS for m in models):
        raise ValueError('허용 모델: ' + ','.join(ALLOWED_MODELS))

    root = Path(args.root).resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(args.out).resolve() if args.out else root / 'outputs' / 'h2_factorial' / stamp
    if out.exists():
        raise FileExistsError(f'기존 출력 보호: {out}')
    d, base, h2, checks = prepare(root, debt_source=args.debt_source)
    groups = design_sets(base)
    all_features = list(dict.fromkeys(f for cols in groups.values() for f in cols))
    if not set(all_features).issubset(d.columns):
        raise ValueError('누락 Feature: ' + str(set(all_features)-set(d.columns)))
    # 일관 비교용 같은 표본: 각 실험 공통으로 필수 변수들이 존재하는 표본 사용.
    common_train = d.eligible & d[all_features + ['y']].notna().all(axis=1)
    checks['factorial_sets'] = groups
    checks['common_train_candidate_rows'] = int(common_train.sum())
    checks['validation_years'] = args.years
    checks['requested_models'] = models
    checks['notes'] = ('정책 단독변수는 한 달 모든 기업에 동일하며, '
                       '명시적 상호작용의 효과는 M4-M3으로 검증. 부채 4분위는 제외.')
    out.mkdir(parents=True)
    (out / 'data_checks.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame([{'feature_set': name, 'features': ', '.join(cols)}
                  for name, cols in groups.items()]).to_csv(out/'feature_sets.csv', index=False, encoding='utf-8-sig')
    print('5개 실험:', list(groups.keys()))
    print('동일 학습 표본 후보:', int(common_train.sum()), '행')
    print('결과 폴더:', out)
    if args.check_only:
        return
    monthly, summary, comparison, paired_monthly, folds = calc_monthly_ic(
        d, models, groups, start, end)
    for fn, result in [
        ('monthly_rankic.csv', monthly),
        ('model_summary.csv', summary),
        ('comparisons.csv', comparison),
        ('monthly_paired_delta.csv', paired_monthly),
        ('fold_coverage.csv', folds),
    ]:
        result.to_csv(out/fn, index=False, encoding='utf-8-sig')
    print('\n모델별 성능\n', summary.to_string(index=False))
    print('\nH2 추가 성능\n', comparison.to_string(index=False))
    print('\n완료:', out)


if __name__ == '__main__':
    main()
