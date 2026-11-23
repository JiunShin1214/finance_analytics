# -*- coding: utf-8 -*-
"""H2-only RateChange ablation, 2006-2021 expanding window.

Run from repository root:
  python -m src.h2.experiments.run_h2_ratechange_ablation --check-only
  python -m src.h2.experiments.run_h2_ratechange_ablation --years 2006-2006 --models Ridge
  python -m src.h2.experiments.run_h2_ratechange_ablation

Requires the existing run_h2_only_ablation.py (DebtChange version) and
run_h2_paper_models.py. No H1/H3 feature is used as model input.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .run_h2_only_ablation import (
    ALLOWED_MODELS, DEFAULT_DEBT_CHANGE, _sha256, add_debt_change,
    add_policy_change,
)
from .run_h2_paper_models import prepare, _build_model, _ic

RATE_DEFAULT = Path('data/raw/h2/bok_rate_monthly_2001_2021.csv')

SETS = OrderedDict([
    ('R0_debt_change', ['debt', 'debt_change']),
    ('R1_add_rate', ['debt', 'debt_change', 'rate_change']),
    ('R2_policy_baseline', ['debt', 'debt_change', 'policy', 'policy_change']),
    ('R3_policy_add_rate', ['debt', 'debt_change', 'policy', 'policy_change', 'rate_change']),
    ('R4_rate_debt_interaction', ['debt', 'debt_change', 'policy', 'policy_change',
                                  'rate_change', 'rate_change_x_debt']),
    ('R5_policy_interaction', ['debt', 'debt_change', 'policy', 'policy_change',
                               'policy_change_x_debt']),
    ('R6_policy_interaction_add_rate', ['debt', 'debt_change', 'policy', 'policy_change',
                                        'policy_change_x_debt', 'rate_change']),
])

COMPARISONS = [
    ('R1_add_rate', 'R0_debt_change', 'Rate_given_DebtChange'),
    ('R3_policy_add_rate', 'R2_policy_baseline', 'Rate_given_Policy_DebtChange'),
    ('R4_rate_debt_interaction', 'R3_policy_add_rate', 'RateDebtInteraction_given_Rate'),
    ('R6_policy_interaction_add_rate', 'R5_policy_interaction', 'Rate_given_PolicyInteraction'),
    ('R3_policy_add_rate', 'R1_add_rate', 'Policy_given_RateDebtChange'),
]


def add_rate_change(data: pd.DataFrame, rate_path: Path) -> tuple[pd.DataFrame, dict]:
    if not rate_path.is_file():
        raise FileNotFoundError(f'RateChange CSV not found: {rate_path}')
    rate = pd.read_csv(rate_path, dtype={'feature_month': 'string'}, encoding='utf-8-sig')
    required = {'feature_month', 'rate_pct', 'rate_change_pp', 'regime'}
    if not required.issubset(rate.columns):
        raise ValueError(f'Missing rate columns: {sorted(required - set(rate.columns))}')
    rate['feature_month'] = rate['feature_month'].str.strip()
    if rate.feature_month.isna().any() or not rate.feature_month.str.fullmatch(
        r'\d{4}-(0[1-9]|1[0-2])', na=False
    ).all() or rate.feature_month.duplicated().any():
        raise ValueError('Rate CSV month key missing, invalid or duplicated')
    periods = pd.PeriodIndex(rate.feature_month.astype(str), freq='M')
    if len(periods) > 1 and (periods.sort_values()[1:].asi8 - periods.sort_values()[:-1].asi8 != 1).any():
        raise ValueError('Rate CSV has gaps in calendar months')
    rate['rate_change_pp'] = pd.to_numeric(rate['rate_change_pp'], errors='coerce')
    rate['rate_pct'] = pd.to_numeric(rate['rate_pct'], errors='coerce')
    if rate[['rate_pct','rate_change_pp']].isna().any().any():
        raise ValueError('Rate CSV contains invalid numerical values')
    # Check each change is the current month's rate minus preceding calendar month.
    by_month = rate.set_index('feature_month').sort_index()
    expected = by_month.rate_pct.diff()
    actual = by_month.rate_change_pp
    mismatches = ~np.isclose(expected.iloc[1:], actual.iloc[1:], rtol=0, atol=1e-7)
    if mismatches.any():
        raise ValueError(f'RateChange inconsistent with rate_pct difference: {int(mismatches.sum())} months')
    # The first CSV month has no preceding month in this file. Do not silently treat as 0.
    first_month = str(by_month.index[0])
    rate.loc[rate.feature_month.eq(first_month), 'rate_change_pp'] = np.nan
    rate = rate.rename(columns={'rate_change_pp': 'rate_change'})
    d = data.merge(rate[['feature_month', 'rate_change', 'regime']],
                   on='feature_month', how='left', validate='many_to_one')
    # Month t is observed by end of t; target_month must be t+1.
    expected_target = (pd.PeriodIndex(d.feature_month.astype(str), freq='M') + 1).astype(str)
    if not np.array_equal(expected_target, d.target_month.astype(str).to_numpy()):
        raise ValueError('RateChange time alignment failed: target month != feature month + 1')
    d['rate_change_x_debt'] = d.rate_change * d.debt
    audit = {
        'input_file': str(rate_path), 'sha256': _sha256(rate_path),
        'rate_months_in_file': int(len(rate)),
        'first_file_month_forced_missing': first_month,
        'rate_change_nonmissing_panel_rows': int(d.rate_change.notna().sum()),
        'unmatched_panel_rows': int(d.regime.isna().sum()),
        'regime_counts_months': {str(k): int(v) for k, v in
            rate.regime.value_counts().items()},
        'rate_units': 'percentage points',
        'information_timing': 'rate through the end of feature_month; predicts next target_month',
        'limitation': 'Observed rate change, not an exogenous monetary policy shock; pre/post 2008 policy-rate regime differs'
    }
    return d, audit


def run_experiment(d, years, model_names):
    needed = list(dict.fromkeys(x for cols in SETS.values() for x in cols))
    common = d.eligible & d[needed + ['y']].notna().all(axis=1)
    if not common.any():
        raise ValueError('No complete-case H2 observations')
    rows, fold_rows = [], []
    for year in years:
        train_mask = common & d.target_month.lt(f'{year}-01')
        valid_mask = common & d.target_month.between(f'{year}-01', f'{year}-12')
        n_train, n_valid = int(train_mask.sum()), int(valid_mask.sum())
        if n_train < 100 or n_valid < 50:
            raise RuntimeError(f'{year}: insufficient train/eval rows {n_train}/{n_valid}')
        fold_rows.append({'year': year, 'train_rows': n_train, 'eval_rows': n_valid,
                          'months': int(d.loc[valid_mask, 'target_month'].nunique())})
        target = d.loc[valid_mask, ['stock_id', 'target_month', 'y']].copy()
        y_train = d.loc[train_mask, 'y'].to_numpy(dtype=float)
        for set_name, cols in SETS.items():
            X_train = d.loc[train_mask, cols].to_numpy(dtype=float)
            X_valid = d.loc[valid_mask, cols].to_numpy(dtype=float)
            for model_name in model_names:
                model = _build_model(model_name)
                model.fit(X_train, y_train)
                pred = target.copy()
                pred['pred'] = model.predict(X_valid)
                for month, part in pred.groupby('target_month', sort=True):
                    rows.append({'year': year, 'model': model_name, 'feature_set': set_name,
                                 'target_month': month, 'n': len(part),
                                 'ic': _ic(part.pred.to_numpy(float), part.y.to_numpy(float))})
            print(f'{year} {set_name}: train={n_train:,}, eval={n_valid:,}', flush=True)
    monthly = pd.DataFrame(rows)
    summary = (monthly.groupby(['model','feature_set'], sort=True)
               .agg(months=('ic', 'count'), mean_rankic=('ic','mean'),
                    std_ic=('ic','std'), median_rankic=('ic','median'),
                    positive_ic_share=('ic', lambda s: float((s > 0).mean())),
                    eval_rows=('n', 'sum')).reset_index())
    summary['icir_monthly'] = summary.mean_rankic / summary.std_ic.replace(0, np.nan)
    wide = monthly.pivot(index=['model','target_month'], columns='feature_set', values='ic')
    comparisons, paired = [], []
    for added, base, label in COMPARISONS:
        delta = (wide[added] - wide[base]).dropna()
        if delta.empty:
            raise ValueError(f'No paired IC observations for {label}')
        for (name, month), value in delta.items():
            paired.append({'comparison': label, 'model': name,
                           'target_month': month, 'delta_ic': float(value)})
        for name, group in delta.groupby(level='model'):
            a = group.to_numpy(float)
            comparisons.append({'model': name, 'comparison': label,
                                'added_set': added, 'reference_set': base,
                                'n_months': len(a), 'mean_delta_ic': float(a.mean()),
                                'median_delta_ic': float(np.median(a)),
                                'positive_month_share': float((a > 0).mean())})
    return monthly, summary, pd.DataFrame(comparisons), pd.DataFrame(paired), pd.DataFrame(fold_rows), int(common.sum())


def main():
    parser = argparse.ArgumentParser(description='H2 RateChange incremental ablation, no additional financial data')
    parser.add_argument('--root', default='.')
    parser.add_argument('--years', default='2006-2021')
    parser.add_argument('--models', default=','.join(ALLOWED_MODELS))
    parser.add_argument('--debt-change', type=Path, default=DEFAULT_DEBT_CHANGE)
    parser.add_argument('--rate-file', type=Path, default=RATE_DEFAULT)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        begin, end = map(int, args.years.split('-'))
    except ValueError as e:
        raise ValueError('Use --years YYYY-YYYY') from e
    if not (2006 <= begin <= end <= 2021):
        raise ValueError('Validation years must be within 2006..2021')
    models = [s.strip() for s in args.models.split(',') if s.strip()]
    if not models or any(m not in ALLOWED_MODELS for m in models):
        raise ValueError(f'Allowed models: {ALLOWED_MODELS}')
    root = Path(args.root).resolve()
    d, _, _, prep_audit = prepare(root, debt_source='hybrid')
    d = add_policy_change(d)
    change_path = args.debt_change if args.debt_change.is_absolute() else root / args.debt_change
    d, debt_audit = add_debt_change(d, change_path)
    rate_path = args.rate_file if args.rate_file.is_absolute() else root / args.rate_file
    d, rate_audit = add_rate_change(d, rate_path)
    needed = list(dict.fromkeys(x for cols in SETS.values() for x in cols))
    common = d.eligible & d[needed + ['y']].notna().all(axis=1)
    audit = {
        **prep_audit,
        'experiment': 'H2-only RateChange ablation',
        'feature_sets': SETS,
        'comparisons': COMPARISONS,
        'debt_change_audit': debt_audit,
        'rate_change_audit': rate_audit,
        'models': models,
        'validation_years': args.years,
        'h1_h3_as_model_features': False,
        'shared_complete_case_train_and_eval_rows': int(common.sum()),
        'common_sample_for_every_model_and_feature_set': True,
        'limitations': [
            'RateChange reflects observed monthly rate movements, not policy surprises',
            'RateChange is constant across stocks within each month',
            'Call rate target / BOK base rate regime changed in March 2008',
            'Monthly rate level is end-of-feature-month information, not necessarily available earlier',
            'DebtChange inherits Hybrid first-vintage/PIT limitations',
            'Do not select models using 2022-2025 holdout',
        ]
    }
    print(json.dumps({'validation_years': args.years, 'models': models,
                      'shared_complete_rows': int(common.sum()),
                      'rate_file': str(rate_path),
                      'rate_audit': rate_audit,
                      'feature_sets': SETS}, indent=2, ensure_ascii=False, default=str))
    if args.check_only:
        print('[check-only] no models trained, no files written')
        return
    out = (args.out if args.out.is_absolute() else root / args.out) if args.out else (
        root / 'outputs' / 'h2_only_ratechange_ablation' /
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if out.exists():
        raise FileExistsError(f'Output already exists (no overwrite): {out}')
    result = run_experiment(d, range(begin, end + 1), models)
    monthly, summary, comparisons, paired, folds, _ = result
    out.mkdir(parents=True, exist_ok=False)
    (out / 'data_checks.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    for name, frame in {
        'feature_sets.csv': pd.DataFrame([{'feature_set': name, 'features': ','.join(cols)}
                                          for name, cols in SETS.items()]),
        'monthly_rankic.csv': monthly,
        'model_summary.csv': summary,
        'comparisons.csv': comparisons,
        'monthly_paired_delta.csv': paired,
        'fold_coverage.csv': folds,
    }.items():
        frame.to_csv(out / name, index=False, encoding='utf-8-sig')
    print('\nMean RankIC:\n', summary[['model', 'feature_set', 'mean_rankic']].to_string(index=False))
    print('\nPaired delta RankIC:\n', comparisons[['model', 'comparison', 'mean_delta_ic']].to_string(index=False))
    print('\nOutput:', out)


if __name__ == '__main__':
    main()
