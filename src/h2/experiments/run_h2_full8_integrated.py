# -*- coding: utf-8 -*-
"""H1/H3 baseline against H1/H3 + all eight H2 features (2006-2021).

python -m src.h2.experiments.run_h2_full8_integrated --check-only
python -m src.h2.experiments.run_h2_full8_integrated
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from ._h2_full8_shared import (
    H2_COLUMNS, BASELINE_COLUMNS, add_arguments, validate_args, prepare_shared,
    evaluate, output_dir, persist,
)


def compare(monthly):
    wide = monthly.pivot(index=['model', 'target_month'], columns='feature_set', values='ic')
    delta = (wide['Integrated_12'] - wide['H1_H3_baseline']).dropna()
    if delta.empty:
        raise RuntimeError('No paired RankIC months')
    paired = pd.DataFrame([
        {'model': model, 'target_month': month, 'comparison': 'H2_increment_given_H1H3',
         'delta_ic': float(value)} for (model, month), value in delta.items()
    ])
    comp = []
    for model, block in delta.groupby(level='model'):
        x = block.to_numpy(dtype=float)
        comp.append({'model': model, 'comparison': 'H2_increment_given_H1H3',
                     'added_set': 'Integrated_12', 'reference_set': 'H1_H3_baseline',
                     'n_months': len(x), 'mean_delta_ic': float(x.mean()),
                     'median_delta_ic': float(np.median(x)),
                     'positive_month_share': float((x > 0).mean())})
    return pd.DataFrame(comp), paired


def main():
    parser = argparse.ArgumentParser(description='Integrated 12-feature comparison with H1/H3 baseline')
    add_arguments(parser)
    args = parser.parse_args()
    years, models = validate_args(args)
    root, df, sample, audit = prepare_shared(args, integrated=True)
    features = {
        'H1_H3_baseline': list(BASELINE_COLUMNS),
        'Integrated_12': list(BASELINE_COLUMNS + H2_COLUMNS),
    }
    audit.update({'run': 'H1H3_vs_Integrated12', 'validation_years': args.years,
                  'models': models, 'shared_complete_rows': int(sample.sum())})
    print(json.dumps({'features': features, 'shared_complete_rows': int(sample.sum()),
                      'years': args.years, 'models': models}, ensure_ascii=False, indent=2))
    if args.check_only:
        print('[check-only] No training or output files')
        return
    out = output_dir(root, args, 'h2_full8_integrated')
    monthly, summary, folds = evaluate(df, sample, features, years, models)
    comparisons, paired = compare(monthly)
    persist(out, audit, features, monthly, summary, folds,
            extra={'comparisons.csv': comparisons, 'monthly_paired_delta.csv': paired})
    print('\n', summary[['model','feature_set','mean_rankic']].to_string(index=False))
    print('\n', comparisons[['model','mean_delta_ic']].to_string(index=False))
    print('\nOutput:', out)


if __name__ == '__main__':
    main()
