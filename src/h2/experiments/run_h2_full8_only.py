# -*- coding: utf-8 -*-
"""H2-only 8-feature expanding-window validation (2006-2021).

python -m src.h2.experiments.run_h2_full8_only --check-only
python -m src.h2.experiments.run_h2_full8_only
"""
from __future__ import annotations
import argparse
import json
from ._h2_full8_shared import (
    H2_COLUMNS, add_arguments, validate_args, prepare_shared,
    evaluate, output_dir, persist,
)


def main():
    parser = argparse.ArgumentParser(description='H2-only eight features, no H1/H3 model inputs')
    add_arguments(parser)
    args = parser.parse_args()
    years, models = validate_args(args)
    root, df, sample, audit = prepare_shared(args, integrated=False)
    features = {'H2_only_8': list(H2_COLUMNS)}
    audit.update({'run': 'H2_only_8', 'validation_years': args.years,
                  'models': models, 'shared_complete_rows': int(sample.sum()),
                  'h1_h3_model_inputs': False})
    print(json.dumps({'features': features, 'shared_complete_rows': int(sample.sum()),
                      'years': args.years, 'models': models}, ensure_ascii=False, indent=2))
    if args.check_only:
        print('[check-only] No training or output files')
        return
    out = output_dir(root, args, 'h2_full8_only')
    monthly, summary, folds = evaluate(df, sample, features, years, models)
    persist(out, audit, features, monthly, summary, folds)
    print('\n', summary[['model','feature_set','mean_rankic']].to_string(index=False))
    print('\nOutput:', out)


if __name__ == '__main__':
    main()
