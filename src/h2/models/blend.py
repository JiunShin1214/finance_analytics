"""H2 conditional rank blend on an explicitly fixed ex-ante universe.

No fitting, label access, feature building or regime determination here.
The caller must supply the complete eligible universe at each base month and
active target months determined by the observed base-month rate and readiness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ['stock_id', 'base_month', 'target_month']


def _validate(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if frame.columns.duplicated().any() or set(frame.columns) != set(columns):
        raise ValueError(f'{name}: expected exactly {columns}')
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError(f'{name}: missing or duplicate keys')
    if not all(frame[c].map(lambda v: isinstance(v, str)).all() for c in KEY):
        raise ValueError(f'{name}: keys must be strings; preserve stock_id leading zeros')
    for c in ('base_month', 'target_month'):
        if not frame[c].str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all():
            raise ValueError(f'{name}: invalid month in {c}')
    if not frame.empty and not ((pd.PeriodIndex(frame.base_month, freq='M') + 1).astype(str) == frame.target_month.to_numpy()).all():
        raise ValueError(f'{name}: target_month must follow base_month')
    if not frame.stock_id.str.strip().ne('').all():
        raise ValueError(f'{name}: empty stock_id')


def _same_keys(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    check = left[KEY].merge(right[KEY], on=KEY, how='outer', indicator=True, validate='one_to_one')
    if not check['_merge'].eq('both').all():
        raise ValueError(f'{name}: incomplete or extra forecast universe')


def _scores(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    out = frame.copy()
    out[col] = pd.to_numeric(out[col], errors='raise').astype(float)
    if not np.isfinite(out[col].to_numpy()).all():
        raise ValueError(f'{col}: nonfinite scores')
    return out


def blend_conditional(baseline: pd.DataFrame, specialist: pd.DataFrame,
                      eligible_keys: pd.DataFrame, active_target_months) -> pd.DataFrame:
    """Combine average monthly ranks 95:5 only for explicitly active months.

    baseline: KEY + without_h2 (all eligible rows, no labels)
    specialist: KEY + h2_score (all and only active eligible rows)
    eligible_keys: KEY (independent base-month eligibility; no realized labels)
    active_target_months: iterable of YYYY-MM strings (already gated and ready)
    Returns baseline order, raw scores, h2_active and final_score.
    Inactive months preserve the raw baseline score, as in exante_universe_v2.
    Active months use exactly (95*b + 5*e)/(100*(n+1)), with average ties.
    """
    _validate(eligible_keys, KEY, 'eligible_keys')
    _validate(baseline, KEY + ['without_h2'], 'baseline')
    _validate(specialist, KEY + ['h2_score'], 'specialist')
    if eligible_keys.empty:
        raise ValueError('Empty eligible universe')
    if isinstance(active_target_months, str):
        raise ValueError('active_target_months must be a collection of months')
    active = set(active_target_months)
    if not active.issubset(set(eligible_keys.target_month)):
        raise ValueError('Active month outside eligible universe')
    _same_keys(baseline, eligible_keys, 'baseline vs eligibility')
    _same_keys(specialist, eligible_keys[eligible_keys.target_month.isin(active)], 'specialist vs active eligibility')
    b = _scores(baseline, 'without_h2')
    e = _scores(specialist, 'h2_score')
    out = b.merge(e, on=KEY, how='left', sort=False, validate='one_to_one')
    out['h2_active'] = out.target_month.isin(active)
    out['final_score'] = out.without_h2.copy()
    on = out.h2_active
    if on.any():
        g = out.loc[on].groupby('target_month', sort=False)
        n = g.stock_id.transform('size')
        br = g.without_h2.rank(method='average')
        er = g.h2_score.rank(method='average')
        out.loc[on, 'final_score'] = (95 * br + 5 * er) / (100 * (n + 1))
    return out
