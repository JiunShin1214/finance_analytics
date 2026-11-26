"""H2 monthly routing. Uses only the supplied base-month rate change and fit plan.

Rate data must have been observable at each base-month cutoff. This module
cannot certify source vintages, model fitting, or course-specific permissions.
No label access, file IO, future-rate lookup, or model fitting.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

RULE = 'hike_1m'
MIN_TRAIN_MONTHS = 5
MIN_STOCKS = 20
_MONTH = re.compile(r'\d{4}-(0[1-9]|1[0-2])\Z')


def _months(values, name):
    if isinstance(values, str):
        raise ValueError(f'{name}: expected a collection of months')
    v = list(values)
    if any(not isinstance(x, str) or not _MONTH.fullmatch(x) for x in v):
        raise ValueError(f'{name}: expected YYYY-MM strings')
    if len(v) != len(set(v)):
        raise ValueError(f'{name}: duplicate months')
    return v


def _schema(frame, columns, name):
    if frame.columns.duplicated().any() or set(frame.columns) != set(columns):
        raise ValueError(f'{name}: expected exactly {columns}')


def build_regime_gate(rates: pd.DataFrame, target_months) -> pd.DataFrame:
    """Map each target month T to rate_change_pp at T-1, never at T.

    rates: feature_month, rate_change_pp (unrounded observed month-end change).
    Missing rows/NaN stay unknown and do not activate H2. Inf is rejected.
    """
    _schema(rates, ['feature_month', 'rate_change_pp'], 'rates')
    _months(rates.feature_month, 'feature_month')
    targets = _months(target_months, 'target_months')
    rr = rates.copy()
    if any(isinstance(x, (bool, np.bool_)) for x in rr.rate_change_pp):
        raise ValueError('Boolean rate change is invalid')
    rr['rate_change_pp'] = pd.to_numeric(rr.rate_change_pp, errors='raise').astype(float)
    if np.isinf(rr.rate_change_pp).any():
        raise ValueError('Infinite rate change')
    base = (pd.PeriodIndex(targets, freq='M') - 1).astype(str)
    out = pd.DataFrame({'target_month': targets, 'base_month': base})
    out = out.merge(rr.rename(columns={'feature_month': 'base_month'}), on='base_month', how='left', validate='one_to_one', sort=False)
    v = out.rate_change_pp
    out['rate_status'] = np.select([v.gt(0), v.eq(0), v.lt(0)], ['hike', 'hold', 'cut'], default='unknown')
    out[RULE] = v.gt(0)
    return out


def route_months(rates: pd.DataFrame, target_months, fit_plan: pd.DataFrame) -> pd.DataFrame:
    """Apply the historical annual-fold readiness contract, then the hike gate.

    fit_plan: valid_year, train_end, train_months, min_month_stocks, ready.
    Use the original make_plan output for these columns (one row per year).
    A missing year or ready=False means no specialist. A ready=True claim with
    insufficient counts or a training cutoff in/after validation year is invalid.
    This is an annual-fold contract, not a rolling refit scheduler.
    """
    cols = ['valid_year', 'train_end', 'train_months', 'min_month_stocks', 'ready']
    _schema(fit_plan, cols, 'fit_plan')
    p = fit_plan.copy()
    for col in ['valid_year', 'train_months', 'min_month_stocks']:
        if any(isinstance(x, (bool, np.bool_)) for x in p[col]):
            raise ValueError(f'{col}: boolean is invalid')
        x = pd.to_numeric(p[col], errors='raise').astype(float)
        if not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
            raise ValueError(f'{col}: expected nonnegative integers')
        p[col] = x.astype(int)
    if p.valid_year.duplicated().any() or not p.valid_year.between(1, 9999).all():
        raise ValueError('Invalid or duplicate validation year')
    if not all(isinstance(x, str) and bool(_MONTH.fullmatch(x)) for x in p.train_end):
        raise ValueError('Invalid train_end')
    if not all(isinstance(x, (bool, np.bool_)) for x in p.ready):
        raise ValueError('ready must contain actual booleans')
    # Numeric month ordinals avoid comparing Arrow strings with an empty
    # Series.map result (whose inferred dtype varies by pandas version).
    # The explicit int64 dtype also covers a completely empty fit plan.
    train_end_month = np.asarray(
        [int(m[:4]) * 12 + int(m[5:]) - 1 for m in p.train_end],
        dtype=np.int64,
    )
    valid_start_month = p.valid_year.to_numpy(dtype=np.int64) * 12
    safe = (
        (train_end_month < valid_start_month)
        & (p.train_months.to_numpy(dtype=np.int64) >= MIN_TRAIN_MONTHS)
        & (p.min_month_stocks.to_numpy(dtype=np.int64) >= MIN_STOCKS)
    )
    if np.any(p.ready.to_numpy(dtype=bool) & ~safe):
        raise ValueError('Ready plan violates training chronology or minimum sample')
    out = build_regime_gate(rates, target_months)
    out['valid_year'] = out.target_month.str[:4].astype(int)
    out = out.merge(p, on='valid_year', how='left', validate='many_to_one', sort=False)
    out['model_ready'] = out.ready.eq(True).fillna(False).astype(bool)
    out['h2_active'] = out[RULE] & out.model_ready
    out['route_reason'] = np.select(
        [out.rate_status.eq('unknown').to_numpy(dtype=bool),
         (~out[RULE]).to_numpy(dtype=bool),
         (~out.model_ready).to_numpy(dtype=bool)],
        ['rate_unknown', 'not_hike', 'model_not_ready'], default='active',
    )
    return out.drop(columns='ready')


def active_target_months(routing: pd.DataFrame) -> list[str]:
    """Pass this result to blend_conditional, with full ex-ante eligible keys."""
    if not {'target_month', 'h2_active'}.issubset(routing.columns):
        raise ValueError('Missing routing columns')
    _months(routing.target_month, 'routing target_month')
    if not all(isinstance(x, (bool, np.bool_)) for x in routing.h2_active):
        raise ValueError('h2_active must contain actual booleans')
    return routing.loc[routing.h2_active, 'target_month'].tolist()
