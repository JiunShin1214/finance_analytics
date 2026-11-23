# -*- coding: utf-8 -*-
"""H1/H3 baseline + staged H2 ablation (2006-2021 expanding-window).

Run from repository root:
  python -m src.h2.experiments.run_h2_integrated_ablation --check-only
  python -m src.h2.experiments.run_h2_integrated_ablation --years 2006-2006 --models Ridge
  python -m src.h2.experiments.run_h2_integrated_ablation

Requirements: existing run_h2_paper_models.py, run_h2_only_ablation.py
(DebtChange edition), and run_h2_ratechange_ablation.py.
All sets train/evaluate on exactly the same eligible complete-case stock-months.
The 2022-2025 holdout is never loaded by the preprocessing helper.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .run_h2_paper_models import prepare, _build_model, _ic
from .run_h2_only_ablation import ALLOWED_MODELS, DEFAULT_DEBT_CHANGE, add_debt_change, add_policy_change, _sha256
from .run_h2_ratechange_ablation import RATE_DEFAULT, add_rate_change

# H2 features added incrementally; baseline features are dynamically resolved from
# existing H1/H3 output and are identical across every group.
ADDITIONS = OrderedDict([
    ('M0_H1_H3', []),
    ('M1_debt_policy', ['debt', 'policy']),
    ('M2_policy_change', ['debt', 'policy', 'policy_change']),
    ('M3_debt_change', ['debt', 'policy', 'policy_change', 'debt_change']),
    ('M4_rate_change', ['debt', 'policy', 'policy_change', 'debt_change', 'rate_change']),
    ('M5_interactions', ['debt', 'policy', 'policy_change', 'debt_change', 'rate_change',
                         'policy_x_debt', 'policy_change_x_debt', 'rate_change_x_debt']),
])
PAIRS = [
    ('M1_debt_policy', 'M0_H1_H3', 'InitialH2_given_H1H3'),
    ('M2_policy_change', 'M1_debt_policy', 'PolicyChange_increment'),
    ('M3_debt_change', 'M2_policy_change', 'DebtChange_increment'),
    ('M4_rate_change', 'M3_debt_change', 'RateChange_increment'),
    ('M5_interactions', 'M4_rate_change', 'Interactions_increment'),
    ('M3_debt_change', 'M0_H1_H3', 'H2_without_rate_vs_base'),
    ('M4_rate_change', 'M0_H1_H3', 'H2_with_rate_vs_base'),
    ('M5_interactions', 'M0_H1_H3', 'All_H2_vs_base'),
]


def get_feature_sets(baseline):
    baseline = list(dict.fromkeys(baseline))
    required = {'size', 'mom', 'sigma'}
    if not required.issubset(baseline):
        raise ValueError(f'H1/H3 baseline lacks {sorted(required - set(baseline))}')
    return OrderedDict((name, baseline + extra) for name, extra in ADDITIONS.items())


def prepare_all(root: Path, debt_file: Path, rate_file: Path, require_bm: bool):
    d, baseline, _, audit = prepare(root, debt_source='hybrid')
    # Prevent silent fallback to approximate momentum/volatility in integrated comparison.
    for rel, label in [(Path('data/processed/h1_components.parquet'), 'H1'),
                       (Path('data/processed/h3.parquet'), 'H3')]:
        if not (root / rel).is_file():
            raise FileNotFoundError(f'{label} source missing: {root / rel}; integrated experiment cannot use fallback')
    if require_bm and 'bm' not in baseline:
        raise ValueError('Expected bm in H1 baseline, but bm is absent or all-missing')
    d = add_policy_change(d)
    d, debt_audit = add_debt_change(d, debt_file)
    # Older optional rate panel loaded by prepare() may contain `rate_change`.
    # Always use the explicitly supplied, audited BOK monthly file instead.
    d = d.drop(columns=['rate_change', 'rate_x_debt', 'rate_change_x_debt', 'regime'], errors='ignore')
    d, rate_audit = add_rate_change(d, rate_file)
    features = get_feature_sets(baseline)
    required_cols = list(dict.fromkeys(c for cols in features.values() for c in cols))
    for c in required_cols:
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    common = d['eligible'] & d[required_cols + ['y']].notna().all(axis=1)
    audit.update({
        'experiment': 'H1+H3 baseline with incremental H2 variables',
        'baseline_features': baseline,
        'feature_sets': features,
        'comparisons': PAIRS,
        'common_complete_case_rows': int(common.sum()),
        'common_sample_all_models': True,
        'debt_change_audit': debt_audit,
        'rate_change_audit': rate_audit,
        'limitations': [
            'H1 size, B/M, momentum and H3 volatility are features of other hypotheses',
            'All arms use one common complete-case train and evaluation sample',
            'This is an in-sample-history expanding-window validation, NOT 2022-25 test',
            'RateChange is observed rate movement, not an exogenous surprise',
            'DebtChange inherits Hybrid incomplete historical PIT limitation',
            'H1/H3 files may themselves have timing limitations; verify upstream',
            'Each year refits on all earlier target months; no test-month outcomes in training',
            'M5 tests all three interactions as a BLOCK, not their individual effects',
        ],
    })
    return d, features, common, audit


def evaluate(d, features, common, years, models):
    records, coverage = [], []
    for yr in years:
        train = common & d['target_month'].lt(f'{yr}-01')
        valid = common & d['target_month'].between(f'{yr}-01', f'{yr}-12')
        ntrain, nvalid = int(train.sum()), int(valid.sum())
        months = int(d.loc[valid, 'target_month'].nunique())
        if ntrain < 100 or nvalid < 50 or months != 12:
            raise RuntimeError(f'{yr}: train={ntrain}, eval={nvalid}, months={months}; expected 12 validation months')
        coverage.append({'year': yr, 'train_rows': ntrain, 'eval_rows': nvalid,
                         'eval_months': months, 'first_train_target_month': str(d.loc[train, 'target_month'].min()),
                         'last_train_target_month': str(d.loc[train, 'target_month'].max())})
        meta = d.loc[valid, ['target_month', 'y']]
        ytrain = d.loc[train, 'y'].to_numpy(dtype=float)
        for group, cols in features.items():
            Xtrain = d.loc[train, cols].to_numpy(dtype=float)
            Xvalid = d.loc[valid, cols].to_numpy(dtype=float)
            for model_name in models:
                predictor = _build_model(model_name)
                predictor.fit(Xtrain, ytrain)
                predicted = predictor.predict(Xvalid)
                tmp = meta.copy()
                tmp['predicted'] = predicted
                for month, part in tmp.groupby('target_month', sort=True):
                    records.append({'year': yr, 'model': model_name, 'feature_set': group,
                                    'target_month': month, 'n': len(part),
                                    'ic': _ic(part['predicted'].to_numpy(float), part['y'].to_numpy(float))})
            print(f'{yr} {group}: train={ntrain:,} eval={nvalid:,}', flush=True)
    monthly = pd.DataFrame(records)
    if monthly.empty:
        raise RuntimeError('No RankIC outputs')
    summary = (monthly.groupby(['model', 'feature_set'], sort=True)
               .agg(months=('ic','count'), mean_rankic=('ic','mean'),
                    std_ic=('ic','std'), median_rankic=('ic','median'),
                    positive_ic_share=('ic',lambda x: (x>0).mean()),
                    eval_rows=('n','sum')).reset_index())
    summary['icir_monthly'] = summary.mean_rankic / summary.std_ic.replace(0, np.nan)
    wide = monthly.pivot(index=['model','target_month'], columns='feature_set', values='ic')
    contrasts, paired = [], []
    for added, reference, label in PAIRS:
        delta = (wide[added] - wide[reference]).dropna()
        if delta.empty:
            raise RuntimeError(f'No paired RankIC for {label}')
        for (model, target_month), value in delta.items():
            paired.append({'comparison': label, 'model': model,
                           'target_month': target_month, 'delta_ic': float(value)})
        for model, block in delta.groupby(level='model'):
            arr = block.to_numpy(float)
            contrasts.append({'model': model, 'comparison': label,
                              'added_set': added, 'reference_set': reference,
                              'n_months': len(arr), 'mean_delta_ic': float(arr.mean()),
                              'median_delta_ic': float(np.median(arr)),
                              'positive_month_share': float((arr>0).mean())})
    return monthly, summary, pd.DataFrame(contrasts), pd.DataFrame(paired), pd.DataFrame(coverage)


def main():
    ap = argparse.ArgumentParser(description='H1/H3 + H2 expanding-window ablation')
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--years', default='2006-2021')
    ap.add_argument('--models', default=','.join(ALLOWED_MODELS))
    ap.add_argument('--debt-change', type=Path, default=DEFAULT_DEBT_CHANGE)
    ap.add_argument('--rate-file', type=Path, default=RATE_DEFAULT)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--allow-no-bm', action='store_true', help='Allow H1 baseline without B/M')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    try:
        start, stop = (int(x) for x in a.years.split('-'))
    except (ValueError, TypeError) as e:
        raise ValueError('--years must be like 2006-2021') from e
    if not 2006 <= start <= stop <= 2021:
        raise ValueError('Validation years must remain within 2006..2021')
    models = [m.strip() for m in a.models.split(',') if m.strip()]
    if not models or any(m not in ALLOWED_MODELS for m in models):
        raise ValueError(f'Unknown models; allowed: {ALLOWED_MODELS}')
    root = a.root.resolve()
    resolve = lambda p: p if p.is_absolute() else root / p
    d, feature_sets, common, audit = prepare_all(root, resolve(a.debt_change),
                                                  resolve(a.rate_file), not a.allow_no_bm)
    audit.update({'years': a.years, 'models': models})
    print(json.dumps({'years': a.years, 'models': models,
                      'baseline': audit['baseline_features'],
                      'all_feature_sets': feature_sets,
                      'common_complete_rows': audit['common_complete_case_rows'],
                      'debt_change_available': audit['debt_change_audit']['available_rows'],
                      'rate_available': audit['rate_change_audit']['rate_change_nonmissing_panel_rows'],
                      'h1_h3_as_model_features': True}, indent=2, ensure_ascii=False, default=str))
    if not common.any():
        raise RuntimeError('No common complete-case rows; investigate feature missingness')
    if a.check_only:
        print('[check-only] no model trained and no files written')
        return
    out = resolve(a.out) if a.out else root / 'outputs' / 'h2_integrated_ablation' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if out.exists():
        raise FileExistsError(f'Output path already exists: {out}')
    monthly, summary, comparisons, paired, folds = evaluate(d, feature_sets, common, range(start, stop+1), models)
    out.mkdir(parents=True, exist_ok=False)
    (out/'data_checks.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    frames = {
        'feature_sets.csv': pd.DataFrame([{'feature_set':k, 'n_features':len(v), 'features':','.join(v)} for k,v in feature_sets.items()]),
        'monthly_rankic.csv': monthly,
        'model_summary.csv': summary,
        'comparisons.csv': comparisons,
        'monthly_paired_delta.csv': paired,
        'fold_coverage.csv': folds,
    }
    for name, frame in frames.items():
        frame.to_csv(out/name, index=False, encoding='utf-8-sig')
    print('\n=== RankIC ===\n', summary[['model','feature_set','mean_rankic']].to_string(index=False))
    print('\n=== Incremental RankIC ===\n', comparisons[['model','comparison','mean_delta_ic']].to_string(index=False))
    print('\nOutput directory:', out)


if __name__ == '__main__':
    main()
