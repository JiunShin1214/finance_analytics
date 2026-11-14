"""In-memory H3; no files, network, targets or H1 dependencies.

returns: stock_id/month/return, attrs['units']['return']='decimal'. base_keys:
stock_id/base_month, optional nullable boolean eligible; absent means unknown.
base_keys must contain the complete monthly candidate universe, independent of y.

Optional listing_history: stock_id/month/is_listed/basis, covering every processed
month. False certifies a FULL calendar month actually unlisted across ALL markets,
not missing returns, trading suspension or ineligibility for KOSPI. basis records
source evidence. Same-security linkage across codes must be established upstream.
The fourth consecutive unlisted month after a listed month resets variance and
both observation buffers once per gap. Future relisting is never consulted.
Partial-month dates must be resolved by a later verified input adapter.

Output preserves requested rows. attrs['state_history'] holds a separate full
calendar DataFrame generated in the same pass; detach attrs before serialization.
Scores are withheld for a month when a row of unknown eligibility has a volatility
value (it could belong to the universe); unknown rows without one do not withhold
the month (2026-11-08: 42 never-observed codes). All results are labelled
calculation_only_not_PIT_certified: supplied flags do not certify source quality.
Without listing_history, diagnostic EWMA works but reset is not assessed.
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal
from numbers import Real
from statistics import variance
from typing import Sequence
import math

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.preprocessing.panel import (
    enforce_unique_stock_month, normalize_month, normalize_stock_id,
)

STATE_COLUMNS = (
    'stock_id', 'base_month', 'return', 'observed_row', 'initialized',
    'initialized_this_month', 'initialization_month', 'n_valid_since_reset',
    'v', 'action', 'recent_valid_count', 'reset', 'reset_reason', 'reset_basis',
    'is_listed', 'unlisted_months', 'segment',
)
VALUE_COLUMNS = (
    'v', 'sqrt_v', 'recent_valid_count', 'initialized', 'vol_available',
    'vol_exclusion_reason', 'h3', 'h3_available', 'h3_exclusion_reason',
    'standardization_n', 'eligibility_complete', 'listing_history_supplied',
)

# 2026-11-12 MODEL_GRID.h3_lambda_learning: λ만 후보 10개 중에서 고르고 틀(초기화 10개, 최근 12개월 중 10개,
# 4개월 공백 초기화)은 고정. span N -> 가중치 2/(N+1), λ = 1 - 가중치(pandas span 정의). 0.97은 현재 값 그대로.
H3_LAMBDA_SPANS = (12, 18, 24, 30, 36, 42, 48, 54, 60)
H3_CURRENT_DECAY = 0.97


def h3_lambda_candidates(spans: Sequence[int] = H3_LAMBDA_SPANS, current: float = H3_CURRENT_DECAY) -> DataFrame:
    """λ 후보 표(label, span, decay, weight). 승인된 후보 밖의 값은 거부한다.

    weight는 새 관측의 가중치(1-λ). span은 2/(N+1)을 직접 쓰고, 0.97은 십진수로 1-0.97=0.03을 만들어
    기존 계산(.97*v + .03*r²)과 비트 단위로 같게 한다.
    """
    if tuple(spans) != H3_LAMBDA_SPANS or current != H3_CURRENT_DECAY:
        raise ValueError('H3 λ candidates must match MODEL_GRID.h3_lambda_learning')
    rows = [(f'span{n}', float(n), 1 - 2 / (n + 1), 2 / (n + 1)) for n in spans]
    rows.append(('decay' + repr(current).replace('.', ''),2 / (1 - current) - 1, current, float(Decimal('1') - Decimal(repr(current)))))
    return DataFrame(rows, columns=['label', 'span', 'decay', 'weight'])


def _decay_weight(decay) -> float:
    """승인된 λ의 가중치. 후보 밖이면 거부."""
    if isinstance(decay, (bool, np.bool_)) or not isinstance(decay, Real):
        raise ValueError('changed decay')
    table = h3_lambda_candidates()
    match = table.loc[table['decay'].eq(float(decay)), 'weight']
    if len(match) != 1:
        raise ValueError('decay is not an approved H3 λ candidate')
    return float(match.iloc[0])


def _return_value(value) -> float:
    if value is None or value is pd.NA:
        return math.nan
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('return must be a numeric decimal or missing')
    value = float(value)
    if not math.isnan(value) and (not math.isfinite(value) or value < -1):
        raise ValueError('nonfinite return or simple return below -1')
    return value


def initialize_ewma_state(first_valid_returns: Sequence[float], *, init_n_valid: int = 10, ddof: int = 1) -> float:
    """Ten finite decimal returns -> sample variance. Caller tracks months/tau."""
    if init_n_valid != 10 or ddof != 1 or len(first_valid_returns) != 10:
        raise ValueError('initialization requires exactly 10 returns and ddof=1')
    values = [_return_value(v) for v in first_valid_returns]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('initialization returns must be finite')
    result = float(variance(values))
    if not math.isfinite(result):
        raise ValueError('nonfinite initial variance')
    return result


def update_ewma_state(previous_variance: float, current_return: float | None, *, previous_month: str, current_month: str, decay: float = 0.97) -> float:
    """Advance exactly one month; NaN holds v, real zero decays it."""
    normalize_month([previous_month, current_month])
    if pd.Period(current_month, freq='M') != pd.Period(previous_month, freq='M') + 1:
        raise ValueError('state updates must advance exactly one calendar month')
    weight = _decay_weight(decay)
    if (isinstance(previous_variance, (bool, np.bool_))
            or not isinstance(previous_variance, Real)
            or not math.isfinite(previous_variance) or previous_variance < 0):
        raise ValueError('invalid variance or changed decay')
    value = _return_value(current_return)
    if math.isnan(value):
        return float(previous_variance)
    try:
        result = float(decay) * previous_variance + weight * value ** 2
    except OverflowError as exc:
        raise ValueError('nonfinite updated variance') from exc
    if not math.isfinite(result):
        raise ValueError('nonfinite updated variance')
    return float(result)


def _keys(table, month, required):
    if not isinstance(table, DataFrame) or not required.issubset(table.columns):
        raise ValueError(f'missing columns: {sorted(required)}')
    enforce_unique_stock_month(table, month_column=month)
    normalize_stock_id(table['stock_id'].tolist())
    normalize_month(table[month].tolist())
    if any(not '2000-01' <= m <= '2021-12' for m in table[month]):
        raise PermissionError('H3 currently accepts development months 2000-2021 only')


def _inputs(returns, base_keys, listing_history):
    _keys(returns, 'month', {'stock_id', 'month', 'return'})
    _keys(base_keys, 'base_month', {'stock_id', 'base_month'})
    units = returns.attrs.get('units', {})
    if not isinstance(units, dict) or units.get('return') != 'decimal':
        raise ValueError("attrs['units']['return']='decimal' required")
    r = returns.loc[:, ['stock_id', 'month', 'return']].copy()
    r['return'] = [_return_value(v) for v in r['return']]
    b = base_keys.loc[:, ['stock_id', 'base_month']].copy()
    eligible = base_keys['eligible'] if 'eligible' in base_keys else [pd.NA] * len(b)
    if any(not pd.isna(x) and not isinstance(x, (bool, np.bool_)) for x in eligible):
        raise ValueError('eligible must be boolean or missing')
    b['eligible'] = pd.array(eligible, dtype='boolean')
    history = None
    if listing_history is not None:
        _keys(listing_history, 'month', {'stock_id', 'month', 'is_listed', 'basis'})
        if any(not isinstance(x, (bool, np.bool_)) for x in listing_history['is_listed']):
            raise ValueError('listing history requires explicit non-null booleans')
        if any(not isinstance(x, str) or not x.strip() for x in listing_history['basis']):
            raise ValueError('listing history requires evidence basis')
        history = listing_history.set_index(['stock_id', 'month'])
    return r, b, history


def _calendar(stock, requested, r, history):
    last = requested['base_month'].max()
    obs = r.loc[(r['stock_id'] == stock) & (r['month'] <= last)]
    starts = [requested['base_month'].min(), *obs['month'].tolist()]
    if history is not None and stock in history.index.get_level_values(0):
        starts.extend(m for m in history.loc[stock].index if m <= last)
    return pd.period_range(min(starts), last, freq='M').astype(str).tolist(), dict(zip(obs['month'], obs['return']))


def _state_history(r, b, history, decays=(H3_CURRENT_DECAY,)):
    """달력 상태 이력. decays가 여럿이면 같은 경로(초기화·보류·공백 초기화)에서 λ별 v를 함께 갱신한다.

    상태 표의 v는 decays[0] 기준. 반환: (상태 표, λ별 v 행렬[행=상태 표 행, 열=decays]).
    초기 분산·관측 여부·최근 개수는 λ와 무관하므로 한 번만 계산한다.
    """
    decay_arr = np.array([float(d) for d in decays])
    weight_arr = np.array([_decay_weight(d) for d in decays])
    records, grid = [], []
    for stock, requested in b.groupby('stock_id', sort=True):
        calendar, observations = _calendar(stock, requested, r, history)
        initial, recent = [], deque(maxlen=12)
        vs = np.full(len(decays), math.nan)
        v, tau, previous = math.nan, None, None
        seen, gap, segment, n_valid = False, 0, 0, 0
        gap_basis = []
        for month in calendar:
            value = observations.get(month, math.nan)
            listed, reset, reset_basis = pd.NA, False, ''
            if history is not None:
                if (stock, month) not in history.index:
                    raise ValueError(f'listing coverage missing: {stock}/{month}')
                entry = history.loc[(stock, month)]
                listed = bool(entry['is_listed'])
                if not listed and math.isfinite(value):
                    raise ValueError('return contradicts a certified whole unlisted month')
                if listed:
                    seen, gap, gap_basis = True, 0, []
                elif seen:
                    gap += 1
                    gap_basis.append(str(entry['basis']))
                    reset = gap == 4
                if reset:
                    v, tau, initial, n_valid = math.nan, None, [], 0
                    vs = np.full(len(decays), math.nan)
                    recent.clear()
                    segment += 1
                    reset_basis = ' | '.join(gap_basis)
            recent.append(math.isfinite(value))
            n_valid += int(math.isfinite(value))
            initialized_now = False
            if reset:
                action = 'reset'
            elif math.isnan(v):
                action = 'pending'
                if math.isfinite(value):
                    initial.append(value)
                    if len(initial) == 10:
                        v = initialize_ewma_state(initial)
                        vs = np.full(len(decays), v)
                        tau, initialized_now, action = month, True, 'initialize'
            else:
                if pd.Period(month, freq='M') != pd.Period(previous, freq='M') + 1:
                    raise ValueError('state updates must advance exactly one calendar month')
                if math.isnan(value):
                    action = 'hold'
                else:
                    vs = decay_arr * vs + weight_arr * (value ** 2)
                    if not np.isfinite(vs).all():
                        raise ValueError('nonfinite updated variance')
                    action = 'update'
                v = float(vs[0])
            records.append((stock, month, value, month in observations, math.isfinite(v),
                            initialized_now, tau, n_valid, v, action, sum(recent), reset,
                            'actual_unlisted_4_calendar_months' if reset else '', reset_basis,
                            listed, gap, segment))
            grid.append(vs.copy())
            previous = month
    result = DataFrame.from_records(records, columns=STATE_COLUMNS)
    for col in ('initialized', 'initialized_this_month', 'reset', 'observed_row'):
        result[col] = result[col].astype(bool)
    result['is_listed'] = pd.array(result['is_listed'], dtype='boolean')
    return result, np.array(grid, dtype=float).reshape(len(records), len(decays))


def _monthly_flags(features):
    """Row-aligned (candidate, unknown-with-volatility) masks.

    A month is withheld only when a security of unknown eligibility has a
    volatility value, i.e. could have belonged to the universe. Unknown rows
    without volatility cannot enter the universe and do not withhold the month.
    """
    eligible = features['eligible']
    vol = features['vol_available'].astype(bool).to_numpy()
    candidate = eligible.eq(True).fillna(False).astype(bool).to_numpy() & vol
    unknown = eligible.isna().to_numpy() & vol
    return candidate, unknown


def _score(features):
    """Monthly sample Z of sqrt_v (ddof=1); one array pass per base month."""
    result = features.copy()
    n_rows = len(result)
    h3 = np.full(n_rows, np.nan)
    available = np.zeros(n_rows, dtype=bool)
    n_std = np.zeros(n_rows, dtype=np.int64)
    complete_col = np.zeros(n_rows, dtype=bool)
    eligible = result['eligible']
    candidate, unknown = _monthly_flags(result)
    reason = np.select(
        [eligible.isna().to_numpy(), ~eligible.eq(True).fillna(False).astype(bool).to_numpy(),
         ~result['vol_available'].astype(bool).to_numpy()],
        ['eligibility_unknown', 'ineligible', result['vol_exclusion_reason'].astype(object).to_numpy()],
        '').astype(object)
    sqrt_v = result['sqrt_v']
    for _, pos in result.groupby('base_month', sort=False).indices.items():
        cand = pos[candidate[pos]]
        complete = not unknown[pos].any()
        n_std[pos] = len(cand)
        complete_col[pos] = complete
        if not complete:
            reason[pos[reason[pos] == '']] = 'monthly_universe_unverified'
            continue
        if not len(cand):
            continue
        sigma = sqrt_v.iloc[cand]
        sd = sigma.std(ddof=1) if len(cand) >= 2 else math.nan
        if not math.isfinite(sd) or sd <= 0:
            reason[cand] = 'too_few_stocks' if len(cand) < 2 else 'zero_or_nonfinite_sd'
            continue
        h3[cand] = ((sigma - sigma.mean()) / sd).to_numpy()
        available[cand] = True
    result['h3'] = h3
    result['h3_available'] = available
    result['h3_exclusion_reason'] = ''
    result['h3_exclusion_reason'] = pd.Series(reason, index=result.index).astype(result['h3_exclusion_reason'].dtype)
    result['standardization_n'] = n_std
    result['eligibility_complete'] = complete_col
    return result


def calculate_h3_ewma_feature(returns: DataFrame, base_keys: DataFrame, *, decay: float = 0.97, recent_window: int = 12, recent_min_valid: int = 10, listing_history: DataFrame | None = None) -> DataFrame:
    """Same-pass calendar states in attrs['state_history']; optional history keyword.

    See module contract for column schemas, identity, whole-month listing semantics
    and diagnostic/PIT distinction. decay는 h3_lambda_candidates()의 10개 중 하나(기본 0.97),
    12·10은 고정. validate_h3_contract는 0.97 결과만 검산한다.
    """
    if recent_window != 12 or recent_min_valid != 10:
        raise ValueError('H3 window parameters are fixed at 12, 10')
    _decay_weight(decay)
    r, b, history = _inputs(returns, base_keys, listing_history)
    states, _ = _state_history(r, b, history, decays=(decay,))
    features = b.merge(states.loc[:, ['stock_id', 'base_month', 'v', 'initialized',
                                     'recent_valid_count']],
                       on=['stock_id', 'base_month'], how='left', validate='one_to_one', sort=False)
    features['vol_available'] = features['initialized'] & features['recent_valid_count'].ge(10)
    features['sqrt_v'] = np.sqrt(features['v']).where(features['vol_available'])
    features['vol_exclusion_reason'] = np.where(
        ~features['initialized'], 'not_initialized',
        np.where(~features['vol_available'], 'recent_valid_below_10', ''))
    features = _score(features)
    features['listing_history_supplied'] = history is not None
    features.attrs.update(state_history=states, units={'return': 'decimal'},
                          result_scope='calculation_only_not_PIT_certified',
                          listing_history_supplied=history is not None,
                          reset_assessed=history is not None)
    return features


def compute_h3_from_panel(panel: DataFrame, eligibility: DataFrame, *, last_base_month: str = '2021-11'):
    """panel(stock_id, month, return, eligible)·eligibility -> (특징표, 상태 이력, 상장 이력, 수익률 입력).

    기준월 2000-01~last_base_month 전 행. 수익률은 제공 원값만, 4개월 공백 초기화의 상장 여부는 KOSPI만 본다
    (2026-11-08 사용자 결정, src.preprocessing.listing.kospi_listing_history). 목표 열이 있으면 거부한다.
    """
    from src.preprocessing.listing import kospi_listing_history
    if 'y' in panel:
        raise ValueError('h3_input_must_not_include_targets')
    returns = panel[['stock_id', 'month', 'return']].copy()
    returns.attrs['units'] = {'return': 'decimal'}
    keys = panel.loc[panel.month.le(last_base_month), ['stock_id', 'month', 'eligible']].rename(columns={'month': 'base_month'})
    history = kospi_listing_history(eligibility, panel)
    features = calculate_h3_ewma_feature(returns, keys, listing_history=history)
    states = features.attrs['state_history']
    features = features.copy()
    features.attrs = {}
    return features, states, history, returns


def compute_h3_sigma_grid(panel: DataFrame, eligibility: DataFrame, *, last_base_month: str = '2021-11',
                          candidates: DataFrame | None = None) -> DataFrame:
    """λ 후보 10개의 σ를 한 번의 상태 계산으로 만든다(model_test 입력).

    입력·달력·초기화·공백 초기화는 compute_h3_from_panel과 같다. 반환 열: stock_id, base_month, eligible,
    initialized, recent_valid_count, vol_available, sigma_<label>(후보 순서). vol_available은 λ와 무관하고,
    σ는 산출 가능할 때만 값이 있다. attrs['h3_lambda_candidates']에 후보 표.
    """
    from src.preprocessing.listing import kospi_listing_history
    if 'y' in panel:
        raise ValueError('h3_input_must_not_include_targets')
    candidates = h3_lambda_candidates() if candidates is None else candidates
    expected = h3_lambda_candidates()
    if not candidates[['label', 'decay', 'weight']].equals(expected[['label', 'decay', 'weight']]):
        raise ValueError('H3 λ candidates must match MODEL_GRID.h3_lambda_learning')
    returns = panel[['stock_id', 'month', 'return']].copy()
    returns.attrs['units'] = {'return': 'decimal'}
    keys = panel.loc[panel.month.le(last_base_month), ['stock_id', 'month', 'eligible']].rename(columns={'month': 'base_month'})
    history = kospi_listing_history(eligibility, panel)
    r, b, history = _inputs(returns, keys, history)
    states, grid = _state_history(r, b, history, decays=tuple(candidates['decay']))
    labels = [f'sigma_{x}' for x in candidates['label']]
    states = pd.concat([states[['stock_id', 'base_month', 'initialized', 'recent_valid_count']],
                        DataFrame(grid, columns=labels, index=states.index)], axis=1)
    out = b.merge(states, on=['stock_id', 'base_month'], how='left', validate='one_to_one', sort=False)
    out['vol_available'] = out['initialized'] & out['recent_valid_count'].ge(10)
    for col in labels:
        out[col] = np.sqrt(out[col]).where(out['vol_available'])
    out = out[['stock_id', 'base_month', 'eligible', 'initialized', 'recent_valid_count', 'vol_available', *labels]]
    out = out.sort_values(['stock_id', 'base_month']).reset_index(drop=True)
    out.attrs['h3_lambda_candidates'] = candidates.copy()
    return out


def h3_output_table(features: DataFrame, states: DataFrame) -> DataFrame:
    """공유 저장용 H3 표(contracts.H3_COLUMNS). sigma는 산출 가능할 때만, v는 EWMA 상태 그대로."""
    from src.contracts import H3_COLUMNS
    out = features.merge(states[['stock_id', 'base_month', 'initialization_month']], on=['stock_id', 'base_month'],
                         how='left', validate='one_to_one')
    out = out.rename(columns={'sqrt_v': 'sigma'})
    return out[list(H3_COLUMNS)].sort_values(['stock_id', 'base_month']).reset_index(drop=True)


def _close(actual, expected, name):
    if pd.isna(expected):
        if not pd.isna(actual):
            raise ValueError(f'{name}: expected missing')
    elif pd.isna(actual) or not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f'{name}: contract mismatch')


def validate_h3_contract(features: DataFrame, state_history: DataFrame, returns: DataFrame, *, listing_history: DataFrame | None = None) -> None:
    """Audit source alignment, state equations, events and sample Z independently.

    Does not call the calculation/initialize/update/score producers. Pass the same
    optional listing_history. Full state coverage includes non-requested months.
    """
    r, b, history = _inputs(returns, features, listing_history)
    _keys(state_history, 'base_month', set(STATE_COLUMNS))
    if not set(VALUE_COLUMNS).issubset(features.columns):
        raise ValueError('incomplete H3 feature schema')
    if set(state_history['stock_id']) != set(b['stock_id']):
        raise ValueError('state security coverage mismatch')
    for stock, group in state_history.groupby('stock_id'):
        group = group.sort_values('base_month')
        calendar, sources = _calendar(stock, b[b['stock_id'] == stock], r, history)
        if group['base_month'].tolist() != calendar:
            raise ValueError('state calendar coverage mismatch')
        values, window = [], []
        previous, tau = math.nan, None
        seen, gap, segment, evidence = False, 0, 0, []
        for row in group.to_dict('records'):
            key = (stock, row['base_month'])
            value = sources.get(row['base_month'], math.nan)
            _close(row['return'], value, 'source return')
            reset, listed = False, pd.NA
            if history is not None:
                if key not in history.index:
                    raise ValueError('missing listing coverage')
                entry = history.loc[key]
                listed = bool(entry['is_listed'])
                if not listed and math.isfinite(value):
                    raise ValueError('unlisted month has a return')
                if listed:
                    seen, gap, evidence = True, 0, []
                elif seen:
                    gap += 1
                    evidence.append(str(entry['basis']))
                    reset = gap == 4
            if reset:
                values, window, previous, tau = [], [], math.nan, None
                segment += 1
            window = (window + [math.isfinite(value)])[-12:]
            pending = math.isnan(previous)
            if math.isfinite(value):
                values.append(value)
            initialized_now = pending and len(values) == 10
            if initialized_now:
                mean = math.fsum(values) / 10
                expected = math.fsum((x - mean) ** 2 for x in values) / 9
                tau = row['base_month']
            elif not pending and math.isfinite(value):
                expected = .97 * previous + .03 * value * value
            else:
                expected = previous
            action = ('reset' if reset else 'initialize' if initialized_now else
                      'pending' if math.isnan(expected) else
                      'hold' if math.isnan(value) else 'update')
            _close(row['v'], expected, 'variance')
            tau_matches = pd.isna(row['initialization_month']) if tau is None else row['initialization_month'] == tau
            listed_matches = pd.isna(row['is_listed']) if pd.isna(listed) else row['is_listed'] == listed
            if (row['initialized'] != math.isfinite(expected)
                    or row['initialized_this_month'] != initialized_now
                    or row['recent_valid_count'] != sum(window)
                    or row['n_valid_since_reset'] != len(values)
                    or row['action'] != action or row['reset'] != reset
                    or row['unlisted_months'] != gap or row['segment'] != segment
                    or row['observed_row'] != (row['base_month'] in sources)
                    or not tau_matches or not listed_matches
                    or row['reset_reason'] != ('actual_unlisted_4_calendar_months' if reset else '')
                    or row['reset_basis'] != (' | '.join(evidence) if reset else '')):
                raise ValueError('state event/count/evidence mismatch')
            previous = expected
    audit = state_history.set_index(['stock_id', 'base_month'])
    for row in features.to_dict('records'):
        state = audit.loc[(row['stock_id'], row['base_month'])]
        _close(row['v'], state['v'], 'feature variance')
        valid = bool(state['initialized'] and state['recent_valid_count'] >= 10)
        _close(row['sqrt_v'], math.sqrt(state['v']) if valid else math.nan, 'sqrt_v')
        reason = '' if valid else 'not_initialized' if not state['initialized'] else 'recent_valid_below_10'
        if (row['vol_available'] != valid or row['initialized'] != state['initialized']
                or row['recent_valid_count'] != state['recent_valid_count']
                or row['vol_exclusion_reason'] != reason
                or row['listing_history_supplied'] != (history is not None)):
            raise ValueError('feature availability mismatch')
    for _, group in features.groupby('base_month'):
        complete = not (group['eligible'].isna() & group['vol_available'].astype(bool)).any()
        candidates = group[group['eligible'].fillna(False) & group['vol_available']]
        xs = candidates['sqrt_v'].tolist()
        mean = math.fsum(xs) / len(xs) if xs else math.nan
        sd = math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) >= 2 else math.nan
        for row in group.to_dict('records'):
            if pd.isna(row['eligible']):
                reason = 'eligibility_unknown'
            elif not row['eligible']:
                reason = 'ineligible'
            elif not row['vol_available']:
                reason = row['vol_exclusion_reason']
            elif not complete:
                reason = 'monthly_universe_unverified'
            elif len(xs) < 2:
                reason = 'too_few_stocks'
            elif not math.isfinite(sd) or sd <= 0:
                reason = 'zero_or_nonfinite_sd'
            else:
                reason = ''
            expected = (row['sqrt_v'] - mean) / sd if not reason else math.nan
            _close(row['h3'], expected, 'h3')
            if (row['h3_available'] != (not reason) or row['h3_exclusion_reason'] != reason
                    or row['standardization_n'] != len(xs) or row['eligibility_complete'] != complete):
                raise ValueError('standardization contract mismatch')
