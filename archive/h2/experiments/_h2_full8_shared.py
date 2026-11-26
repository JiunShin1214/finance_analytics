# -*- coding: utf-8 -*-
"""Shared read-only preparation and validation for separate H2 and integrated experiments.

Requires the existing run_h2_integrated_ablation.py and prior H2 helpers.
The H2-only sample requires only H2 columns; the integrated comparison uses
a 12-feature common sample for its two arms. They need not have equal coverage.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .run_h2_integrated_ablation import prepare_all
from .run_h2_only_ablation import ALLOWED_MODELS, DEFAULT_DEBT_CHANGE, add_debt_change, add_policy_change
from .run_h2_ratechange_ablation import RATE_DEFAULT, add_rate_change
from .run_h2_paper_models import prepare, _build_model, _ic

H2_COLUMNS = (
    'debt', 'debt_change', 'policy', 'policy_change', 'rate_change',
    'policy_x_debt', 'policy_change_x_debt', 'rate_change_x_debt',
)
BASELINE_COLUMNS = ('size', 'bm', 'mom', 'sigma')


def add_arguments(parser):
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--years', default='2006-2021')
    parser.add_argument('--models', default=','.join(ALLOWED_MODELS))
    parser.add_argument('--debt-change', type=Path, default=DEFAULT_DEBT_CHANGE)
    parser.add_argument('--rate-file', type=Path, default=RATE_DEFAULT)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check-only', action='store_true')


def validate_args(args):
    try:
        start, end = map(int, args.years.split('-'))
    except (TypeError, ValueError) as exc:
        raise ValueError('--years must be YYYY-YYYY') from exc
    if not (2006 <= start <= end <= 2021):
        raise ValueError('Only 2006-2021 internal validation is allowed')
    models = [x.strip() for x in args.models.split(',') if x.strip()]
    if not models or len(models) != len(set(models)) or any(x not in ALLOWED_MODELS for x in models):
        raise ValueError(f'Invalid model selection: {models}')
    return list(range(start, end + 1)), models


def prepare_shared(args, *, integrated: bool):
    root = args.root.resolve()
    def resolve(path):
        return path if path.is_absolute() else root / path
    if integrated:
        df, _, original_common, audit = prepare_all(
            root, resolve(args.debt_change), resolve(args.rate_file), require_bm=True
        )
        needed = list(BASELINE_COLUMNS + H2_COLUMNS)
    else:
        # Standalone H2 does NOT require H1/H3 files or values to be present.
        df, _, _, audit = prepare(root, debt_source='hybrid')
        df = add_policy_change(df)
        df, debt_audit = add_debt_change(df, resolve(args.debt_change))
        df = df.drop(columns=['rate_change', 'rate_x_debt',
                              'rate_change_x_debt', 'regime'], errors='ignore')
        df, rate_audit = add_rate_change(df, resolve(args.rate_file))
        audit.update({'debt_change_audit': debt_audit, 'rate_change_audit': rate_audit})
        needed = list(H2_COLUMNS)
    if not set(needed).issubset(df.columns):
        raise ValueError(f'Missing input columns: {sorted(set(needed)-set(df.columns))}')
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf,-np.inf], np.nan)
    sample = df['eligible'] & df[needed + ['y']].notna().all(axis=1)
    if integrated and not sample.equals(original_common):
        raise RuntimeError('Mismatch with integrated-ablation common sample')
    if not sample.any():
        raise RuntimeError('No complete-case observations')
    return root, df, sample, audit


def evaluate(df, sample, features, years, model_names):
    records, folds = [], []
    for year in years:
        training = sample & df.target_month.lt(f'{year}-01')
        validating = sample & df.target_month.between(f'{year}-01', f'{year}-12')
        n_tr, n_va = int(training.sum()), int(validating.sum())
        month_count = df.loc[validating, 'target_month'].nunique()
        if n_tr < 100 or n_va < 50 or month_count != 12:
            raise RuntimeError(f'{year}: train={n_tr}, valid={n_va}, months={month_count}')
        key = ['stock_id', 'target_month']
        if df.loc[validating, key].duplicated().any():
            raise RuntimeError(f'{year}: duplicate stock-month validation keys')
        folds.append({'year': year, 'train_rows': n_tr, 'eval_rows': n_va, 'eval_months': month_count})
        y_tr = df.loc[training, 'y'].to_numpy(dtype=float)
        truth = df.loc[validating, ['target_month', 'y']].copy()
        for group, columns in features.items():
            x_tr = df.loc[training, columns].to_numpy(dtype=float)
            x_va = df.loc[validating, columns].to_numpy(dtype=float)
            for model_name in model_names:
                model = _build_model(model_name)
                model.fit(x_tr, y_tr)
                pred = truth.copy()
                pred['pred'] = model.predict(x_va)
                for month, part in pred.groupby('target_month', sort=True):
                    records.append({
                        'year': year, 'model': model_name, 'feature_set': group,
                        'target_month': month, 'n': len(part),
                        'ic': float(_ic(part.pred.to_numpy(dtype=float),
                                        part.y.to_numpy(dtype=float)))
                    })
            print(f'{year} | {group} | train={n_tr:,} | eval={n_va:,}', flush=True)
    monthly = pd.DataFrame(records)
    if monthly.empty:
        raise RuntimeError('No RankIC results')
    if monthly.groupby(['model', 'feature_set']).target_month.nunique().ne(12 * len(years)).any():
        raise RuntimeError('Missing evaluation months')
    if len(features) > 1:
        counts = monthly.pivot(index=['model', 'target_month'], columns='feature_set', values='n')
        if counts.isna().any().any() or (counts.nunique(axis=1) != 1).any():
            raise RuntimeError('Evaluation samples differ')
    summary = (monthly.groupby(['model', 'feature_set'], sort=True)
               .agg(months=('ic','count'), mean_rankic=('ic','mean'),
                    std_ic=('ic','std'), median_rankic=('ic','median'),
                    positive_ic_share=('ic', lambda s: float((s > 0).mean())),
                    eval_rows=('n', 'sum')).reset_index())
    summary['icir_monthly'] = summary.mean_rankic / summary.std_ic.replace(0, np.nan)
    return monthly, summary, pd.DataFrame(folds)


def output_dir(root, args, category):
    path = (args.out if args.out.is_absolute() else root / args.out) if args.out else (
        root / 'outputs' / category / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if path.exists():
        raise FileExistsError(f'Existing results are protected: {path}')
    return path


def persist(path, audit, features, monthly, summary, folds, extra=None):
    path.mkdir(parents=True, exist_ok=False)
    audit = dict(audit)
    audit.update({'experiment_features': features,
                  'sample_policy': ('Both integrated arms share the 12-feature complete-case sample'
                                    if len(features) > 1 else 'H2-only 8-feature complete-case sample; may differ from integrated'),
                  'test_2022_2025_used': False})
    (path / 'data_checks.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    pd.DataFrame([{'feature_set': k, 'features': ','.join(v), 'n_features': len(v)}
                  for k,v in features.items()]).to_csv(path / 'feature_sets.csv', index=False, encoding='utf-8-sig')
    monthly.to_csv(path / 'monthly_rankic.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(path / 'model_summary.csv', index=False, encoding='utf-8-sig')
    folds.to_csv(path / 'fold_coverage.csv', index=False, encoding='utf-8-sig')
    if extra:
        for name, frame in extra.items():
            frame.to_csv(path / name, index=False, encoding='utf-8-sig')
