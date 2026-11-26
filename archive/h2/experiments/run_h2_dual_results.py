# -*- coding: utf-8 -*-
"""H2 8-feature standalone vs H1/H3 baseline vs 12-feature integrated.

Runs one shared 2006-2021 expanding-window validation sample, without 2022-25 data.
Requires the existing H2 integrated ablation, DebtChange ablation and RateChange modules.

python -m src.h2.experiments.run_h2_dual_results --check-only
python -m src.h2.experiments.run_h2_dual_results --years 2006-2006 --models Ridge
python -m src.h2.experiments.run_h2_dual_results
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .run_h2_integrated_ablation import prepare_all
from .run_h2_only_ablation import ALLOWED_MODELS, DEFAULT_DEBT_CHANGE
from .run_h2_ratechange_ablation import RATE_DEFAULT
from .run_h2_paper_models import _build_model, _ic

H2_FEATURES = [
    'debt', 'debt_change', 'policy', 'policy_change', 'rate_change',
    'policy_x_debt', 'policy_change_x_debt', 'rate_change_x_debt',
]
ARMS = ('H2_only_8', 'H1_H3_baseline', 'H1_H3_plus_H2_12')
PAIRS = [
    ('H1_H3_plus_H2_12', 'H1_H3_baseline', 'H2_increment_given_H1H3'),
    ('H1_H3_plus_H2_12', 'H2_only_8', 'H1H3_increment_given_H2'),
]


def feature_sets_from_baseline(baseline):
    baseline = list(dict.fromkeys(baseline))
    overlap = set(baseline) & set(H2_FEATURES)
    if overlap:
        raise ValueError(f'H1/H3 and H2 overlap: {sorted(overlap)}')
    if len(baseline) != 4 or not {'size', 'bm', 'mom', 'sigma'}.issubset(baseline):
        raise ValueError(f'Expected four H1/H3 features size,bm,mom,sigma: {baseline}')
    return OrderedDict([
        ('H2_only_8', list(H2_FEATURES)),
        ('H1_H3_baseline', baseline),
        ('H1_H3_plus_H2_12', baseline + list(H2_FEATURES)),
    ])


def run_validation(d, common, features, years, models):
    rows, folds = [], []
    for year in years:
        training = common & d['target_month'].lt(f'{year}-01')
        evaluation = common & d['target_month'].between(f'{year}-01', f'{year}-12')
        tr, va = int(training.sum()), int(evaluation.sum())
        months = int(d.loc[evaluation, 'target_month'].nunique())
        if tr < 100 or va < 50 or months != 12:
            raise RuntimeError(f'{year}: train={tr}, eval={va}, months={months}; expected 12 months')
        folds.append({'year': year, 'train_rows': tr, 'eval_rows': va, 'eval_months': months})
        target = d.loc[evaluation, ['stock_id', 'target_month', 'y']].copy()
        if target.duplicated(['stock_id', 'target_month']).any():
            raise ValueError(f'{year}: duplicated validation stock-month keys')
        y_train = d.loc[training, 'y'].to_numpy(dtype=float)
        for arm, cols in features.items():
            X_train = d.loc[training, cols].to_numpy(dtype=float)
            X_valid = d.loc[evaluation, cols].to_numpy(dtype=float)
            for model_name in models:
                model = _build_model(model_name)
                model.fit(X_train, y_train)
                predicted = model.predict(X_valid)
                tmp = target[['target_month', 'y']].copy()
                tmp['prediction'] = predicted
                for month, part in tmp.groupby('target_month', sort=True):
                    rows.append({'year': year, 'model': model_name, 'feature_set': arm,
                                 'target_month': month, 'n': int(len(part)),
                                 'ic': float(_ic(part.prediction.to_numpy(float),
                                                 part.y.to_numpy(float)))})
            print(f'{year} | {arm} | train={tr:,} valid={va:,}', flush=True)
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        raise RuntimeError('No validation results')
    # The number of stocks must match across arms for each model and month.
    check = monthly.pivot(index=['model', 'target_month'], columns='feature_set', values='n')
    if check.isna().any().any() or (check.nunique(axis=1) != 1).any():
        raise RuntimeError('Evaluation sample is not identical across arms')
    summary = (monthly.groupby(['model', 'feature_set'], sort=True)
               .agg(months=('ic', 'count'), mean_rankic=('ic', 'mean'),
                    std_ic=('ic', 'std'), median_rankic=('ic', 'median'),
                    positive_ic_share=('ic', lambda s: float((s > 0).mean())),
                    eval_rows=('n', 'sum')).reset_index())
    summary['icir_monthly'] = summary.mean_rankic / summary.std_ic.replace(0, np.nan)
    wide = monthly.pivot(index=['model', 'target_month'], columns='feature_set', values='ic')
    paired, comps = [], []
    for added, ref, label in PAIRS:
        delta = (wide[added] - wide[ref]).dropna()
        if delta.empty:
            raise RuntimeError(f'No paired monthly RankIC: {label}')
        for (model, month), value in delta.items():
            paired.append({'comparison': label, 'model': model,
                           'target_month': month, 'delta_ic': float(value)})
        for model, series in delta.groupby(level='model'):
            vals = series.to_numpy(float)
            comps.append({'model': model, 'comparison': label, 'added_set': added,
                          'reference_set': ref, 'n_months': len(vals),
                          'mean_delta_ic': float(vals.mean()),
                          'median_delta_ic': float(np.median(vals)),
                          'positive_month_share': float((vals > 0).mean())})
    return monthly, summary, pd.DataFrame(comps), pd.DataFrame(paired), pd.DataFrame(folds)


def main():
    ap = argparse.ArgumentParser(description='H2 8-only and H1/H3+H2 12-feature fair comparison')
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--years', default='2006-2021')
    ap.add_argument('--models', default=','.join(ALLOWED_MODELS))
    ap.add_argument('--debt-change', type=Path, default=DEFAULT_DEBT_CHANGE)
    ap.add_argument('--rate-file', type=Path, default=RATE_DEFAULT)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    try:
        start, stop = (int(x) for x in args.years.split('-'))
    except (ValueError, TypeError) as exc:
        raise ValueError('--years must be YYYY-YYYY') from exc
    if not (2006 <= start <= stop <= 2021):
        raise ValueError('Only 2006-2021 internal validation is allowed')
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    if not models or len(set(models)) != len(models) or any(m not in ALLOWED_MODELS for m in models):
        raise ValueError(f'Invalid/duplicate model: {models}; expected {ALLOWED_MODELS}')
    root = args.root.resolve()
    resolve = lambda p: p if p.is_absolute() else root / p
    d, _, common_full, audit = prepare_all(root, resolve(args.debt_change),
                                           resolve(args.rate_file), require_bm=True)
    features = feature_sets_from_baseline(audit['baseline_features'])
    # prepare_all already computes a common complete-case sample over ALL eight H2 and
    # four H1/H3 features; same universe for both separate deliverables.
    needed = list(dict.fromkeys(x for cols in features.values() for x in cols))
    common = d['eligible'] & d[needed + ['y']].notna().all(axis=1)
    if not common.equals(common_full):
        raise RuntimeError('Common sample differs from integrated existing code')
    preview = {'years': args.years, 'models': models,
               'feature_sets': features, 'common_complete_rows': int(common.sum()),
               'sample_policy': 'Same complete-case stock-months for all three arms',
               'test_2022_2025_used': False}
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    if not common.any():
        raise RuntimeError('No shared complete observations')
    if args.check_only:
        print('[check-only] no model trained or outputs written')
        return
    out = resolve(args.out) if args.out else root / 'outputs' / 'h2_dual_results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if out.exists():
        raise FileExistsError(f'Existing output is protected: {out}')
    monthly, summary, comps, paired, folds = run_validation(d, common, features, range(start, stop+1), models)
    out.mkdir(parents=True, exist_ok=False)
    audit.update({'years': args.years, 'models': models, 'three_arm_feature_sets': features,
                  'three_arm_comparisons': PAIRS, 'shared_complete_rows': int(common.sum()),
                  'test_2022_2025_used': False})
    (out / 'data_checks.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tables = {
        'feature_sets.csv': pd.DataFrame([{'feature_set': name, 'n_features': len(cols),
                                           'features': ','.join(cols)} for name, cols in features.items()]),
        'monthly_rankic.csv': monthly,
        'model_summary.csv': summary,
        'comparisons.csv': comps,
        'monthly_paired_delta.csv': paired,
        'fold_coverage.csv': folds,
    }
    for filename, frame in tables.items():
        frame.to_csv(out / filename, index=False, encoding='utf-8-sig')
    for arm, folder in [('H2_only_8', 'H2_only'), ('H1_H3_baseline', 'H1_H3_baseline'),
                        ('H1_H3_plus_H2_12', 'Integrated_12')]:
        dest = out / folder
        dest.mkdir()
        summary.loc[summary.feature_set.eq(arm)].to_csv(dest / 'model_summary.csv', index=False, encoding='utf-8-sig')
        monthly.loc[monthly.feature_set.eq(arm)].to_csv(dest / 'monthly_rankic.csv', index=False, encoding='utf-8-sig')
    print('\n=== 3-arm RankIC ===\n', summary[['model','feature_set','mean_rankic']].to_string(index=False))
    print('\n=== Paired improvements ===\n', comps[['model','comparison','mean_delta_ic']].to_string(index=False))
    print('\nOutput:', out)


if __name__ == '__main__':
    main()
