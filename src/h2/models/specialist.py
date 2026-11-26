"""Development-only H2 specialist adapter (annual expanding folds).

Inputs are already transformed by the existing model_test.prepare_inputs.
Reuse the audited historical fit function rather than changing its math.
No target labels in prediction inputs; no file writes or test-period fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .blend import KEY, _validate, _same_keys
from .regime import build_regime_gate, MIN_TRAIN_MONTHS, MIN_STOCKS

RAW_FEATURES = ('debt', 'policy', 'policy_change', 'policy_x_debt',
                'debt_change', 'rate_change', 'policy_change_x_debt', 'rate_change_x_debt')
FEATURES = tuple('x_' + x for x in RAW_FEATURES)
KINDS = ('lgb_rankreg', 'xgb_rankreg', 'xgb_pairwise')


def feature_specs(config):
    """Existing common input-transform contract, now on canonical H2 parquet."""
    return {n: {'path': config['PATHS']['h2'],
                'transform': 'input' if n == 'debt_change' else 'none'}
            for n in RAW_FEATURES}


def frozen_run():
    return dict(trees=500, depth=3, learning_rate=0.01, xgb_l2=0,
                pairwise_pairs={'lambdarank_pair_method': 'mean',
                                'lambdarank_num_pair_per_sample': 8, 'seed': 0})


def _legacy_fit(*args):
    from src.h2.experiments.evaluate_h2_training_period_and_features import fit_predict
    return fit_predict(*args)


def _numeric(frame, columns, name, *, allow_nan):
    data = frame.copy()
    for c in columns:
        if pd.api.types.is_bool_dtype(data[c]):
            raise ValueError(f'{name}: boolean numerical column {c}')
        data[c] = pd.to_numeric(data[c], errors='raise').astype(float)
    arr = data[list(columns)].to_numpy(float)
    if np.isinf(arr).any() or (not allow_nan and not np.isfinite(arr).all()):
        raise ValueError(f'{name}: invalid numerical input')
    return data


def annual_train_end(valid_year):
    """Historical shared plan: reserve Y-1 for inner validation; no refit.

    Outer year Y uses targets through December Y-2. No final-test years.
    """
    if isinstance(valid_year, (bool, np.bool_)) or not isinstance(valid_year, (int, np.integer)) or not 2003 <= valid_year <= 2021:
        raise ValueError('Development validation years only (2003-2021)')
    return f'{int(valid_year)-2}-12'


def predict_annual_fold(train, forecast, eligible_keys, rates, *, valid_year,
                        train_start='2001-01', kind='xgb_pairwise', threads=4, device='cpu'):
    """Return (active scores, readiness audit) for one development validation year.

    train: KEY + FEATURES + y + fit_member. Caller supplies training history
    only, through Y-2 December (Y-1 is reserved for inner validation),
    with original common-model fit_member eligibility / label exclusions.
    forecast: KEY + FEATURES, full ex-ante eligible universe for requested months.
    eligible_keys: independent KEY table for that same forecast universe.
    rates: feature_month + rate_change_pp, observed by each base-month cutoff.
    Nothing here certifies upstream PIT, or authorizes 2022-25 scoring.
    """
    if kind not in KINDS or device not in ('cpu', 'cuda') or (kind == 'lgb_rankreg' and device != 'cpu'):
        raise ValueError('Unsupported model/device')
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('Invalid thread count')
    train_end = annual_train_end(valid_year)
    if not isinstance(train_start, str) or len(train_start) != 7 or str(pd.Period(train_start, freq='M')) != train_start:
        raise ValueError('Invalid train_start')
    if not '2001-01' <= train_start <= train_end:
        raise ValueError('Invalid training window')
    _validate(train, KEY + list(FEATURES) + ['y', 'fit_member'], 'train')
    _validate(forecast, KEY + list(FEATURES), 'forecast')
    _validate(eligible_keys, KEY, 'eligible_keys')
    _same_keys(forecast, eligible_keys, 'forecast vs eligibility')
    if forecast.empty or not forecast.target_month.between(f'{valid_year}-01', f'{valid_year}-12').all():
        raise ValueError('Forecast outside validation year')
    if not train.target_month.between(train_start, train_end).all():
        raise ValueError('Training data outside historical fit window (Y-2; Y-1 reserved)')
    if not all(isinstance(x, (bool, np.bool_)) for x in train.fit_member):
        raise ValueError('fit_member must be boolean')
    train = _numeric(train, FEATURES, 'train', allow_nan=True)
    forecast = _numeric(forecast, FEATURES, 'forecast', allow_nan=True)
    months = sorted(set(train.target_month) | set(forecast.target_month))
    gate = build_regime_gate(rates, months)
    hikes = set(gate.loc[gate.hike_1m, 'target_month'])
    tr = train.loc[train.fit_member & train.target_month.isin(hikes)].copy()
    tr = _numeric(tr, ['y'], 'training labels', allow_nan=False).sort_values(['target_month','stock_id'])
    counts = tr.groupby('target_month').size()
    ready = bool(len(counts) >= MIN_TRAIN_MONTHS and counts.ge(MIN_STOCKS).all())
    active = sorted(set(forecast.target_month) & hikes) if ready else []
    va = forecast.loc[forecast.target_month.isin(active)].sort_values(['target_month','stock_id'])
    audit = dict(model=kind, valid_year=int(valid_year), train_start=train_start, train_end=train_end,
                 train_rows=len(tr), train_months=len(counts), min_month_stocks=int(counts.min()) if len(counts) else 0,
                 train_month_list=sorted(counts.index), ready=ready, active_target_months=active,
                 forecast_rows=len(va), trees=0, test_used=False)
    scores = va[KEY].copy().reset_index(drop=True)
    if active:
        # Same feature order, label rank/deciles, qid and model settings as audited run.
        pred, _, n = _legacy_fit(tr, va[list(FEATURES)], list(FEATURES), kind, frozen_run(), threads, device)
        pred = np.asarray(pred)
        if n != 500 or pred.shape != (len(va),) or not np.isfinite(pred).all():
            raise ValueError('Unexpected tree count or forecast scores')
        scores['h2_score'] = pred
        audit['trees'] = n
    else:
        scores['h2_score'] = pd.Series(dtype=float)
    return scores, audit
