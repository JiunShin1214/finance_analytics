"""v2 모델 비교(model_test): 공통 입력 준비 + linear_model(OLS·Ridge) 실행. 설계는 config MODEL_GRID와 노션 「11주차 v2 모델」.

입력은 data/interim(panel, eligibility)·data/processed(h1_components, h3)와 config뿐이다. outputs/의 이전 결과를
읽지 않고 이어서 돌리지 않는다(같은 세션 메모리로 표를 만든다). 저장은 --execute일 때만, 쓰기 전용.

흐름
1. prepare_inputs: λ 후보 10개의 σ(한 번에), 매달 순위 [−1, 1] 입력, 목표값 3종(그 달 학습 행 기준), 학습·예측 행.
   λ=0.97 σ가 h3.parquet와 정확히 같은지 검사.
2. plan_folds: 학습 2003년 시작, 바깥 검증 2006~2021(16개), 내부 검증 = 학습 기간 마지막 1년, 재학습 없음.
3. h1_index: 지수형 ① — 각 해 Y의 H1 지수는 2001~Y−2로 학습·Y−1로 조정값을 고른 ①(②와 같은 모델·목표값·손실)의
   예측. 매달 순위 [−1, 1]로 바꿔 ②에 넣는다.
4. run_linear: Fold × (모델·목표값·손실·입력 조합)마다 조정값(H3 λ × Ridge λ)을 내부 검증 평균 IC 최고 한 쌍으로 고르고
   (정확히 같으면 Ridge λ 큰 쪽 → H3 긴 기간 쪽), 내부 학습 모델 그대로 바깥 검증 연도를 예측.
5. 보고 표와 종료 조건 검사. 하나라도 실패하면 결과를 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.expanding_window import verify_no_test_access
from src.features.h3 import compute_h3_sigma_grid, h3_lambda_candidates
from src.models import boosting, linear

COMPONENTS = ('size', 'bm', 'mom', 'sigma')
H1_PARTS = ('size', 'bm', 'mom')
TARGETS = ('raw', 'winsor_1_99', 'monthly_zscore')
RANK_TARGETS = ('decile', 'percentile')                # 순위 모델 정답: 10분위 floor(10(r−1)/N), 백분위 r/(N+1) [2026-11-14 결정]
BOOSTING_SETS = ('ind:sigma', 'ind:size+bm+mom+sigma', 'idx:h1', 'idx:h1+sigma')   # 부스팅은 주 결과 입력만(노션 §5.1)
# 실행 단계(2026-11-14 결정): 개별형을 먼저, 지수형은 ①(H1 지수) → ②로 따로. [σ] 단독은 지수형 3조합의 H3 칸이라 ②에 둔다.
BOOSTING_PHASES = {'individual': ('ind:size+bm+mom+sigma',), 'index': ('ind:sigma', 'idx:h1', 'idx:h1+sigma')}
WINSOR_Q = (0.01, 0.99)
BASE_FIRST, BASE_LAST = '2000-12', '2021-11'          # 목표월 2001-01~2021-12
TRAIN_START, INDEX_TRAIN_START = 2003, 2001
EXPECTED_VALIDATION_PREDICTION_ROWS = 133_876
PERIODS = {'2006~2010': (2006, 2010), '2011~2021': (2011, 2021), '2018~2021': (2018, 2021)}
INNER_SCOPES = ('full', 'common')
EXPECTED_GRID = {
    'approved': True, 'hypotheses': ['H1', 'H3'], 'train_start_target_year': TRAIN_START,
    'first_validation_year': 2006, 'last_validation_year': 2021, 'inner_validation_years': 1,
    'min_inner_train_years': 5, 'min_inner_train_exceptions': [1, 2, 3],
    'tuning_selection': 'joint_grid_max_inner_ic', 'tie_rules': {'trees': 'fewer', 'ridge_lambda': 'larger', 'h3_span': 'longer'},
    'refit_after_selection': False, 'training_rows': 'common_h1_h3_target', 'prediction_rows': 'all_stocks_with_target',
    'target_returns_unchanged': True, 'components': list(COMPONENTS), 'input_transform': 'monthly_rank_pm1',
    'h3_fixed': {'initial_returns': 10, 'recent_window': 12, 'recent_min_valid': 10, 'delisting_gap_reset_months': 4},
    'index_transform': 'monthly_rank_pm1', 'missing_feature_fill': 0, 'individual_sets': 'all_nonempty_subsets',
    'index_sets': [['h1_index'], ['sigma'], ['h1_index', 'sigma']], 'index_stage1_model': 'same_as_stage2',
    'index_stage1_out_of_sample': 'prior_years_expanding', 'regression_targets': list(TARGETS),
    'target_transform_basis': 'monthly_training_rows',
}


# ---------- 설정 ----------

def check_grid(config) -> dict:
    grid = config.get('MODEL_GRID') or {}
    bad = [k for k, v in EXPECTED_GRID.items() if grid.get(k) != v]
    if bad:
        raise PermissionError('model_grid_not_approved:' + ','.join(bad))
    lam = grid['h3_lambda_learning']
    h3_lambda_candidates(lam['span_months'], lam['include_current_value'])
    linear.check_linear_grid(grid)
    boosting.check_boosting_grid(grid)
    if config.get('FINAL_TEST_ENABLED') is not False:
        raise PermissionError('final_test_must_stay_disabled')
    return grid


def input_sets() -> dict:
    """서로 다른 입력 조합 17개: 개별형 15개 + 지수형 [H1 지수], [H1 지수, σ]. 지수형 [σ]는 개별형 [σ]와 같아 한 번만."""
    sets = {}
    for k in range(1, 5):
        for combo in itertools.combinations(COMPONENTS, k):
            sets['ind:' + '+'.join(combo)] = list(combo)
    sets['idx:h1'] = ['h1_index']
    sets['idx:h1+sigma'] = ['h1_index', 'sigma']
    return sets


# ---------- 1) 입력 준비 ----------

def rank_pm1(values: pd.Series, member: np.ndarray, months: pd.Series) -> np.ndarray:
    """매달 member 행 안 평균 동순위 r → 2r/(N+1) − 1. member가 아니면 NaN."""
    v = values.where(member)
    r = v.groupby(months.to_numpy()).rank(method='average')
    n = member.astype(int)
    count = pd.Series(n).groupby(months.to_numpy()).transform('sum').to_numpy()
    out = 2 * r.to_numpy() / (count + 1) - 1
    return np.where(member, out, np.nan)


def monthly_target_transforms(y: pd.Series, train: np.ndarray, months: pd.Series) -> dict:
    """목표값 3종 + 순위 정답 2종. 경계·평균·표준편차·순위는 그 목표월 학습 행에서만 계산하고 학습 행에만 값을 둔다.

    순위 정답(2026-11-14 결정): 그 달 학습 행 안 평균 동순위 r(1~N)으로 10분위 = floor(10(r−1)/N) ∈ {0..9},
    백분위 = r/(N+1) ∈ (0, 1)(입력 변환 2r/(N+1)−1과 같은 식, GKX 부록 식 A.1).
    """
    m = months.to_numpy()
    yt = y.where(train)
    g = yt.groupby(m)
    lo, hi = g.transform(lambda s: s.quantile(WINSOR_Q[0])), g.transform(lambda s: s.quantile(WINSOR_Q[1]))
    mean, sd = g.transform('mean'), g.transform(lambda s: s.std(ddof=1))
    rank, count = g.rank(method='average'), g.transform('count')
    out = {'raw': yt.to_numpy(float),
           'winsor_1_99': yt.clip(lower=lo, upper=hi).to_numpy(float),
           'monthly_zscore': ((yt - mean) / sd).to_numpy(float),
           'decile': np.floor(10 * (rank - 1) / count).to_numpy(float),
           'percentile': (rank / (count + 1)).to_numpy(float)}
    for name, arr in out.items():
        if not np.isfinite(arr[train]).all() or np.isfinite(arr[~train]).any():
            raise ValueError('target_transform_invalid:' + name)
    return out


def build_model_data(comp: pd.DataFrame, sigma_grid: pd.DataFrame, returns: pd.DataFrame, labels: list[str], *, units) -> pd.DataFrame:
    """특징(기준월 2000-12~2021-11)·λ별 σ·개발 수익률 → 종목×목표월 모델 입력(메모리).

    순위 입력은 매달 적격·산출 종목 안에서, 목표값 3종은 그 목표월 학습 행 안에서 만든다(목표값은 입력을 다 만든 뒤 결합).
    train = 적격이고 H1 구성요소 3개·σ·목표값이 모두 있는 행(조합 간 공통), predict = 목표값(다음 달 수익률)이 있는 행.
    """
    from src.preprocessing.panel import join_training_targets
    keys = ['stock_id', 'base_month']
    c = comp[comp.base_month.between(BASE_FIRST, BASE_LAST)]
    d = c[keys + ['eligible', *H1_PARTS, *[p + '_available' for p in H1_PARTS]]].merge(
        sigma_grid[keys + ['eligible', 'vol_available', *['sigma_' + x for x in labels]]].rename(columns={'eligible': 'eligible_h3'}),
        on=keys, how='left', validate='one_to_one')
    if len(d) != len(c) or not d.eligible.equals(d.eligible_h3):
        raise ValueError('component_h3_rows_or_eligibility_differ')
    r = returns[['stock_id', 'month', 'return']].copy()
    r.attrs['units'] = units
    d = join_training_targets(d.drop(columns='eligible_h3'), r)
    d = d.sort_values(['target_month', 'stock_id']).reset_index(drop=True)
    if d.target_month.ge('2022-01').any():
        raise PermissionError('test_period_in_samples')
    elig = d.eligible.eq(True).fillna(False).to_numpy(bool)
    avail = {p: d[p + '_available'].eq(True).to_numpy(bool) & elig for p in H1_PARTS}
    avail['sigma'] = d.vol_available.eq(True).to_numpy(bool) & elig
    months = d.base_month
    data = d[['stock_id', 'base_month', 'target_month', 'eligible', 'y']].copy()
    data['year'] = d.target_month.str[:4].astype(int).to_numpy()
    for p in H1_PARTS:
        data['x_' + p] = rank_pm1(d[p], avail[p], months)
    for x in labels:
        data['x_sigma_' + x] = rank_pm1(d['sigma_' + x], avail['sigma'], months)
    data['h1_available'] = avail['size'] & avail['bm'] & avail['mom']
    data['train'] = data.h1_available.to_numpy() & avail['sigma'] & d.y.notna().to_numpy()
    data['predict'] = d.y.notna().to_numpy()
    for name, arr in monthly_target_transforms(d.y, data.train.to_numpy(), data.target_month).items():
        data['t_' + name] = arr
    x = data[[c for c in data if c.startswith('x_')]].to_numpy(float)
    if np.nanmax(np.abs(x)) >= 1:
        raise ValueError('rank_inputs_outside_open_interval')
    return data


def compare_sigma_to_h3(grid: pd.DataFrame, h3: pd.DataFrame, *, atol: float) -> dict:
    """다시 계산한 λ=0.97 σ와 h3.parquet 비교. 키·결측 위치는 정확히 같아야 하고, 값은 |차이| ≤ atol.

    허용오차(2026-11-14 결정): Linux(Colab)와 Windows는 같은 파일에서도 마지막 자리(상대 ~2.7e-16)가 달랐다.
    atol은 config REPRODUCIBILITY_PREDICTION_ATOL(1e-10). 실제 최대 차이와 다른 값 수를 기록한다.
    """
    ref = h3.sort_values(['stock_id', 'base_month']).reset_index(drop=True)
    g = grid.sort_values(['stock_id', 'base_month']).reset_index(drop=True)
    if not g[['stock_id', 'base_month']].equals(ref[['stock_id', 'base_month']]):
        raise ValueError('sigma_decay097_keys_differ_from_h3_parquet')
    a, e = g['sigma_decay097'].to_numpy(float), ref['sigma'].to_numpy(float)
    if not np.array_equal(np.isnan(a), np.isnan(e)):
        raise ValueError('sigma_decay097_missing_differs_from_h3_parquet')
    ok = ~np.isnan(a)
    diff = np.abs(a[ok] - e[ok])
    max_diff = float(diff.max()) if len(diff) else 0.0
    if max_diff > atol:
        raise ValueError(f'sigma_decay097_differs_from_h3_parquet:{max_diff:.3e}')
    return {'sigma_decay097_equals_h3_parquet': True, 'sigma_decay097_atol': atol,
            'sigma_decay097_max_abs_diff': max_diff, 'sigma_decay097_values_differ': int((diff > 0).sum())}


def prepare_inputs(root: Path, config, *, h3_grid: pd.DataFrame | None = None) -> dict:
    """공유 표 4개만 읽어 메모리 입력을 만든다. 반환 dict: data(종목×월 표), sigma_labels, checks, inputs(해시)."""
    check_grid(config)
    verify_no_test_access(stage='validate', declared_target_years=list(range(2001, 2022)), source_scope_verified=True)
    paths = config['PATHS']
    names = ('panel', 'eligibility', 'h1_components', 'h3')
    inputs = {Path(paths[n]).as_posix(): sha256(root / paths[n]) for n in names}
    panel = pd.read_parquet(root / paths['panel'], columns=['stock_id', 'month', 'return', 'eligible'])
    if not panel.month.between('2000-01', '2021-12').all():
        raise PermissionError('development_months_only')
    eligibility = pd.read_parquet(root / paths['eligibility'])
    comp = pd.read_parquet(root / paths['h1_components'])
    h3 = pd.read_parquet(root / paths['h3'], columns=['stock_id', 'base_month', 'sigma'])
    if h3_grid is None:
        h3_grid = compute_h3_sigma_grid(panel, eligibility)
    cand = h3_grid.attrs.get('h3_lambda_candidates', h3_lambda_candidates())
    labels = list(cand['label'])
    g = h3_grid.sort_values(['stock_id', 'base_month']).reset_index(drop=True)
    checks = compare_sigma_to_h3(g, h3, atol=float(config['REPRODUCIBILITY_PREDICTION_ATOL']))
    data = build_model_data(comp, g, panel[['stock_id', 'month', 'return']], labels, units=config['UNITS'])
    val = data[data.predict & data.year.between(2006, 2021)]
    checks.update(validation_prediction_rows=int(len(val)), validation_prediction_rows_all_eligible=bool(val.eligible.eq(True).all()),
                  training_rows_2003_2021=int(data.train[data.year >= TRAIN_START].sum()),
                  training_rows_2001_2021=int(data.train.sum()))
    if len(val) != EXPECTED_VALIDATION_PREDICTION_ROWS or not checks['validation_prediction_rows_all_eligible']:
        raise ValueError('validation_prediction_rows_changed')
    return {'data': data, 'sigma_labels': labels, 'lambda_candidates': cand, 'checks': checks, 'inputs': inputs}


# ---------- 2) Fold ----------

def plan_folds(grid) -> list[dict]:
    """바깥 검증 Y마다 내부 학습 2003~Y−2, 내부 검증 Y−1. 재학습이 없으므로 바깥 검증 예측도 내부 학습 모델로 한다."""
    folds = []
    for k, y in enumerate(range(grid['first_validation_year'], grid['last_validation_year'] + 1), start=1):
        inner_years = (y - 2) - TRAIN_START + 1
        exception = inner_years < grid['min_inner_train_years']
        if exception != (k in grid['min_inner_train_exceptions']):
            raise ValueError('fold_exception_list_mismatch')
        folds.append({'fold': k, 'valid_year': y, 'train_years': (TRAIN_START, y - 2), 'inner_valid_year': y - 1,
                      'inner_train_years': inner_years, 'min_years_exception': exception})
    if len(folds) != 16 or folds[-1]['valid_year'] >= 2022:
        raise PermissionError('folds_must_stay_inside_validation')
    return folds


# ---------- 월별 Rank IC(빠른 계산) ----------

def _average_ranks(codes: np.ndarray, s: np.ndarray) -> np.ndarray:
    """월 코드별 평균 동순위(1부터). 정렬 한 번으로 모든 달을 처리한다."""
    n = len(s)
    if not n:
        return np.empty(0)
    order = np.lexsort((s, codes))
    cs, ss = codes[order], s[order]
    new_month = np.r_[True, cs[1:] != cs[:-1]]
    month_start = np.maximum.accumulate(np.where(new_month, np.arange(n), 0))
    pos = np.arange(n) - month_start + 1.0
    new_tie = new_month | np.r_[True, ss[1:] != ss[:-1]]
    starts = np.flatnonzero(new_tie)
    ends = np.r_[starts[1:] - 1, n - 1]
    avg = (pos[starts] + pos[ends]) / 2
    ranks = np.empty(n)
    ranks[order] = avg[np.cumsum(new_tie) - 1]
    return ranks


def _ic_from_parts(codes, n_m, ds, dy):
    """월별 Σds·dy / √(Σds²·Σdy²). 순위 편차는 0.5 단위라 합이 정확히 계산된다(합 순서와 무관)."""
    k = len(n_m)
    sy = np.bincount(codes, ds * dy, minlength=k)
    ss = np.bincount(codes, ds * ds, minlength=k)
    yy = np.bincount(codes, dy * dy, minlength=k)
    denom = np.sqrt(ss * yy)
    with np.errstate(divide='ignore', invalid='ignore'):
        ic = np.where((n_m >= 3) & (denom > 0), sy / np.where(denom > 0, denom, 1.0), np.nan)
    reason = np.select([n_m < 3, ss <= 0, yy <= 0], ['too_few_pairs', 'constant_score', 'constant_target'], '')
    return ic, reason


def monthly_ic(months: np.ndarray, score: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """목표월별 Spearman(평균 동순위의 Pearson). 비유한 쌍 제외, 3쌍 미만·상수는 NaN과 이유.
    rank_ic.rank_ic_for_month와 같은 규칙이며 종료 조건에서 scipy와 독립 대조한다."""
    score, y = np.asarray(score, float), np.asarray(y, float)
    ok = np.isfinite(score) & np.isfinite(y)
    uniq, codes = np.unique(np.asarray(months)[ok], return_inverse=True)
    n_m = np.bincount(codes, minlength=len(uniq))
    half = (n_m[codes] + 1) / 2
    ds = _average_ranks(codes, score[ok]) - half
    dy = _average_ranks(codes, y[ok]) - half
    ic, reason = _ic_from_parts(codes, n_m, ds, dy)
    return pd.DataFrame({'n': n_m, 'ic': ic, 'reason': reason}, index=pd.Index(uniq, name='target_month'))


def mean_ic(table: pd.DataFrame) -> float:
    return float(table.ic.mean()) if table.ic.notna().any() else np.nan


class InnerScorer:
    """고정된 내부 검증 행의 월별 IC를 빠르게: 원수익률 순위는 한 번만 계산한다(결과는 monthly_ic와 같음)."""

    def __init__(self, months: np.ndarray, y: np.ndarray):
        self.months, self.y = np.asarray(months), np.asarray(y, float)
        if not np.isfinite(self.y).all():
            raise ValueError('inner_validation_target_missing')
        uniq, self.codes = np.unique(self.months, return_inverse=True)
        self.n_m = np.bincount(self.codes, minlength=len(uniq))
        self.half = (self.n_m[self.codes] + 1) / 2
        self.dy = _average_ranks(self.codes, self.y) - self.half

    def score(self, yhat: np.ndarray) -> tuple[float, int]:
        yhat = np.asarray(yhat, float)
        if not np.isfinite(yhat).all():
            t = monthly_ic(self.months, yhat, self.y)
            return mean_ic(t), int(t.ic.notna().sum())
        ic, _ = _ic_from_parts(self.codes, self.n_m, _average_ranks(self.codes, yhat) - self.half, self.dy)
        s = pd.Series(ic)
        return (float(s.mean()) if s.notna().any() else np.nan), int(s.notna().sum())


# ---------- 3·4) 선형 모델 적합·선택 ----------

def _candidates(model: str, sigma_labels: list[str] | None, decay_of: dict) -> list[dict]:
    """H3 λ 바깥, Ridge λ 안쪽(작은 것부터) 순서. 같은 H3 λ 안에서 앞 λ의 Huber 해를 다음 λ의 시작점으로 쓴다."""
    lams = linear.RIDGE_LAMBDAS if model == 'ridge' else (None,)
    labels = sigma_labels if sigma_labels else [None]
    return [{'ridge_lambda': lam, 'h3_label': lab, 'decay': decay_of.get(lab, np.nan)} for lab in labels for lam in lams]


def choose(results: list[dict]) -> tuple[dict, int]:
    """내부 검증 평균 IC 최고 한 쌍. 정확히 같으면 Ridge λ 큰 쪽 → H3 λ 큰 쪽(긴 기간). 반환: (선택, 최고값 동률 수)."""
    alive = [r for r in results if np.isfinite(r['inner_ic'])]
    if not alive:
        raise ValueError('all_candidates_failed')
    best = max(r['inner_ic'] for r in alive)
    tied = [r for r in alive if r['inner_ic'] == best]
    key = lambda r: (r['ridge_lambda'] if r['ridge_lambda'] is not None else -1.0,
                     r['decay'] if np.isfinite(r['decay']) else -1.0)
    return max(tied, key=key), len(tied)


def _design(data: pd.DataFrame, columns: list[str], rows: np.ndarray, *, fill: bool) -> np.ndarray:
    X = data[columns].to_numpy(float)[rows]
    if fill:
        X = np.where(np.isfinite(X), X, 0.0)
    elif not np.isfinite(X).all():
        raise ValueError('training_rows_with_missing_inputs')
    return X


def _columns(input_cols: list[str], label: str | None, index_col: str | None) -> list[str]:
    out = []
    for c in input_cols:
        if c == 'sigma':
            out.append('x_sigma_' + label)
        elif c == 'h1_index':
            out.append(index_col)
        else:
            out.append('x_' + c)
    return out


class FoldContext:
    """한 Fold(또는 지수형 ①의 한 해)의 고정 행. 입력 행렬·목표값·Huber② σ를 한 번만 만들어 모든 설정이 함께 쓴다.

    fit_rows: 학습 행(공통 표본), inner_rows: 내부 검증 평가 행, out_rows: 예측 행(채움 허용).
    """

    def __init__(self, data: pd.DataFrame, fit_rows: np.ndarray, inner_rows: np.ndarray, out_rows: np.ndarray | None = None):
        self.data, self.fit_rows, self.inner_rows, self.out_rows = data, fit_rows, inner_rows, out_rows
        months = data.target_month.to_numpy()
        self.m_fit = months[fit_rows]
        self.scorer = InnerScorer(months[inner_rows], data.y.to_numpy(float)[inner_rows])
        self.n_train = int(fit_rows.sum())
        self.rows_sha256 = hashlib.sha256(np.flatnonzero(fit_rows).tobytes()).hexdigest()
        self._X, self._y, self._sigma, self._q999 = {}, {}, {}, {}
        self._train_scorer = None

    @property
    def train_scorer(self) -> 'InnerScorer':
        """학습 행의 월별 IC(원수익률 기준). 부스팅의 트리별 학습 IC용."""
        if self._train_scorer is None:
            self._train_scorer = InnerScorer(self.m_fit, self.data.y.to_numpy(float)[self.fit_rows])
        return self._train_scorer

    def X(self, cols: list[str], which: str) -> np.ndarray:
        key = (tuple(cols), which)
        if key not in self._X:
            rows = {'fit': self.fit_rows, 'inner': self.inner_rows, 'out': self.out_rows}[which]
            self._X[key] = _design(self.data, cols, rows, fill=which != 'fit')
        return self._X[key]

    def y(self, target: str) -> np.ndarray:
        if target not in self._y:
            self._y[target] = self.data['t_' + target].to_numpy(float)[self.fit_rows]
        return self._y[target]

    def q999(self, target: str) -> float:
        if target not in self._q999:
            self._q999[target] = linear.huber_delta_q999(self.y(target), self.m_fit)
        return self._q999[target]

    def sigma2(self, cols: list[str], target: str) -> dict:
        """Huber② σ: 같은 학습 행·입력·목표값의 HuberRegressor(OLS). OLS·Ridge가 함께 쓴다."""
        key = (tuple(cols), target)
        if key not in self._sigma:
            self._sigma[key] = linear.huber_sigma_sklearn(self.X(cols, 'fit'), self.y(target))
        return self._sigma[key]


def select_and_fit(ctx: FoldContext, *, model, target, loss, input_cols, sigma_labels, decay_of, index_col=None):
    """한 Fold·한 설정의 조정값 선택. 모든 후보가 같은 학습 행(ctx.fit_rows)으로 학습하고 내부 검증은 원수익률로 평가.
    반환: (선택 결과 dict, 후보별 내부 IC 목록)."""
    has_sigma = 'sigma' in input_cols
    y_fit = ctx.y(target)
    results, warm = [], {}
    for cand in _candidates(model, sigma_labels if has_sigma else None, decay_of):
        cols = _columns(input_cols, cand['h3_label'], index_col)
        X = ctx.X(cols, 'fit')
        delta, sig = None, None
        if loss == 'huber_q999':
            delta = ctx.q999(target)
        elif loss == 'huber_sigma135':
            sig = ctx.sigma2(cols, target)
            delta = sig['delta']
        try:
            fitted = linear.fit_linear(X, y_fit, model=model, loss=loss, ridge_lambda=cand['ridge_lambda'], delta=delta,
                                       start=warm.get(cand['h3_label']) if loss != 'mse' else None)
        except ValueError as error:
            results.append({**cand, 'inner_ic': np.nan, 'inner_months': 0, 'failure': str(error)})
            continue
        if loss != 'mse':
            warm[cand['h3_label']] = (fitted['intercept'], fitted['coef'])
        ic, months = ctx.scorer.score(linear.predict_linear(fitted, ctx.X(cols, 'inner')))
        results.append({**cand, 'inner_ic': ic, 'inner_months': months, 'failure': '', 'fitted': fitted, 'columns': cols,
                        'huber_sigma': sig['sigma'] if sig else np.nan,
                        'huber_sigma_converged': sig['converged'] if sig else None})
    chosen, ties = choose(results)
    table = [{k: v for k, v in r.items() if k not in ('fitted', 'columns')} for r in results]
    return {**chosen, 'ties': ties, 'n_train': ctx.n_train, 'train_rows_sha256': ctx.rows_sha256}, table


def build_h1_index(data, *, model, target, loss, inner_scope, decay_of, years=range(TRAIN_START, 2022)):
    """지수형 ①. 해 Y의 지수 = 2001~Y−2 학습 행으로 맞추고 Y−1 내부 검증으로 조정값을 고른 ①의 예측(재학습 없음).

    지수는 H1 구성요소 3개가 모두 있는 적격 행에만 만들고, 매달 그 행 안 순위 [−1, 1]로 바꾼다. 반환: (순위 지수 배열, 선택 표).
    """
    raw_index = np.full(len(data), np.nan)
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    scope = data.predict.to_numpy(bool) if inner_scope == 'full' else train
    h1 = data.h1_available.to_numpy(bool)
    rows = []
    for Y in years:
        target_rows = h1 & (year == Y)
        ctx = FoldContext(data, train & (year >= INDEX_TRAIN_START) & (year <= Y - 2), scope & (year == Y - 1), target_rows)
        chosen, _ = select_and_fit(ctx, model=model, target=target, loss=loss, input_cols=list(H1_PARTS),
                                   sigma_labels=None, decay_of=decay_of)
        raw_index[target_rows] = linear.predict_linear(chosen['fitted'], _design(data, chosen['columns'], target_rows, fill=False))
        rows.append({'index_year': Y, 'model': model, 'target': target, 'loss': loss, 'stage1_train_years': f'{INDEX_TRAIN_START}~{Y-2}',
                     'stage1_inner_valid_year': Y - 1, 'ridge_lambda': chosen['ridge_lambda'], 'inner_ic': chosen['inner_ic'],
                     'ties': chosen['ties'], 'n_train': chosen['n_train'], 'intercept': chosen['fitted']['intercept'],
                     **{'w_' + p: w for p, w in zip(H1_PARTS, chosen['fitted']['coef'])}})
    member = np.isfinite(raw_index)
    ranked = rank_pm1(pd.Series(raw_index), member, data.base_month.reset_index(drop=True))
    return ranked, pd.DataFrame(rows)


def linear_settings():
    return [(m, t, l) for m in linear.LINEAR_MODELS for t in TARGETS for l in linear.LOSSES]


LINEAR_PARTS = [(m, l) for m in linear.LINEAR_MODELS for l in linear.LOSSES]   # 노트북 셀 단위: 모델 × 손실 6개


def part_settings(model: str, loss: str):
    if (model, loss) not in LINEAR_PARTS:
        raise ValueError('linear_part_unknown')
    return [(model, t, loss) for t in TARGETS]


_WORKER = {}


def _init_worker(payload):
    _WORKER.clear()
    _WORKER.update(payload)


def _progress(message: str) -> None:
    """진행 줄. 순차 실행은 바로 출력, 병렬 실행은 큐로 본 프로세스에 보낸다(작업 프로세스 출력은 노트북에 안 보임)."""
    if _WORKER.get('queue') is not None:
        _WORKER['queue'].put(message)
    elif _WORKER.get('log') is not None:
        _WORKER['log'](message)


def _run_fold(fold: dict) -> dict:
    """한 Fold의 모든 (입력 조합 × 설정). 병렬 실행 단위. 파일을 읽거나 쓰지 않는다."""
    p = _WORKER
    data = p['data']
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    predict = data.predict.to_numpy(bool)
    scope = predict if p['inner_scope'] == 'full' else train
    t_fold = time.perf_counter()
    lo, hi = fold['train_years']
    out_rows = predict & (year == fold['valid_year'])
    ctx = FoldContext(data, train & (year >= lo) & (year <= hi), scope & (year == fold['inner_valid_year']), out_rows)
    y_all = data.y.to_numpy(float)
    selections, inner, timings, preds = [], [], [], {}
    n_sets = len(p['sets'])
    for i_set, (set_name, cols) in enumerate(p['sets'].items(), start=1):
        for model, target, loss in p['settings']:
            t0 = time.perf_counter()
            chosen, table = select_and_fit(ctx, model=model, target=target, loss=loss, input_cols=cols,
                                           sigma_labels=p['labels'], decay_of=p['decay_of'],
                                           index_col=p['index_cols'].get((model, target, loss)))
            tag = {'family': 'linear', 'model': model, 'target': target, 'loss': loss, 'input_set': set_name, 'fold': fold['fold']}
            f = chosen['fitted']
            preds[config_id(tag)] = linear.predict_linear(f, ctx.X(chosen['columns'], 'out'))
            train_ic = monthly_ic(ctx.m_fit, linear.predict_linear(f, ctx.X(chosen['columns'], 'fit')), y_all[ctx.fit_rows])
            selections.append({**tag, 'valid_year': fold['valid_year'], 'inner_valid_year': fold['inner_valid_year'],
                               'inner_train_years': fold['inner_train_years'], 'min_years_exception': fold['min_years_exception'],
                               'h3_label': chosen['h3_label'], 'h3_decay': chosen['decay'],
                               'h3_span': float(2 / (1 - chosen['decay']) - 1) if np.isfinite(chosen['decay']) else np.nan,
                               'ridge_lambda': np.nan if chosen['ridge_lambda'] is None else chosen['ridge_lambda'],
                               'ties': chosen['ties'], 'inner_ic': chosen['inner_ic'],
                               'train_ic': mean_ic(train_ic), 'n_train': chosen['n_train'],
                               'train_rows_sha256': chosen['train_rows_sha256'],
                               'delta': np.nan if f['delta'] is None else f['delta'],
                               'huber_sigma': chosen.get('huber_sigma', np.nan),
                               'huber_sigma_converged': chosen.get('huber_sigma_converged'),
                               'outlier_share': f.get('outlier_share', np.nan), 'huber_iterations': f.get('iterations', np.nan),
                               'intercept': f['intercept'], **{'w_' + c: w for c, w in zip(cols, f['coef'])}})
            inner.append(pd.DataFrame(table).assign(**tag))
            timings.append({'part': 'fold', 'fold': fold['fold'], 'model': model, 'target': target, 'loss': loss,
                            'input_set': set_name, 'seconds': time.perf_counter() - t0})
        elapsed = time.perf_counter() - t_fold
        _progress(f"  [Fold {fold['fold']:>2}] 조합 {i_set:>2}/{n_sets} ({set_name}) 완료, 경과 {elapsed:.0f}초, "
                  f"남은 예상 {elapsed / i_set * (n_sets - i_set):.0f}초")
    return {'fold': fold, 'selections': selections, 'inner': inner, 'timings': timings, 'preds': preds,
            'seconds': time.perf_counter() - t_fold}


def run_linear(prepared: dict, config, *, inner_scope: str = 'common', folds: list[dict] | None = None,
               settings=None, sets: dict | None = None, workers: int = 1, log=print) -> dict:
    """linear_model 전체(또는 일부 설정)를 실행한다. 반환: val_keys, predictions(설정별 배열), selections, inner_ic, stage1, timings.

    workers>1이면 Fold를 프로세스로 나눠 계산한다(난수 없음, 결과는 Fold 순서로 합쳐 같다).
    """
    if inner_scope not in INNER_SCOPES:
        raise ValueError('inner_scope_unknown')
    grid = check_grid(config)
    data = prepared['data'].copy()
    labels = prepared['sigma_labels']
    decay_of = _decay_of(prepared)
    folds = plan_folds(grid) if folds is None else folds
    settings = linear_settings() if settings is None else list(settings)
    sets = input_sets() if sets is None else sets
    val_keys, offsets = validation_keys(data, folds)
    stage1, timings, index_cols = [], [], {}
    if any('h1_index' in cols for cols in sets.values()):
        t_all = time.perf_counter()
        for model, target, loss in settings:
            t0 = time.perf_counter()
            col = f'idx_{model}_{target}_{loss}'
            data[col], table = build_h1_index(data, model=model, target=target, loss=loss, inner_scope=inner_scope, decay_of=decay_of)
            index_cols[(model, target, loss)] = col
            stage1.append(table)
            timings.append({'part': 'stage1', 'fold': None, 'model': model, 'target': target, 'loss': loss,
                            'seconds': time.perf_counter() - t0})
        log(f'지수형 ① 완료: {len(settings)}개 설정, {time.perf_counter() - t_all:.1f}초')
    payload = {'data': data, 'labels': labels, 'decay_of': decay_of, 'settings': settings, 'sets': sets,
               'index_cols': index_cols, 'inner_scope': inner_scope}
    done = execute_folds(_run_fold, payload, folds, workers=workers, log=log)
    return {**assemble_folds(done, folds, val_keys, offsets, timings), 'stage1': _concat(stage1), 'inner_scope': inner_scope}


# ---------- 선형·부스팅 공통 실행 뼈대 ----------

def _decay_of(prepared: dict) -> dict:
    return dict(zip(prepared['lambda_candidates'].label, prepared['lambda_candidates'].decay))


def _concat(frames: list) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validation_keys(data: pd.DataFrame, folds: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    """검증 행(바깥 검증 연도들의 수익률 있는 전 종목)은 설정과 무관하게 같다 → 키는 한 번, 예측은 설정별 배열."""
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    predict = data.predict.to_numpy(bool)
    fold_rows = {f['fold']: np.flatnonzero(predict & (year == f['valid_year'])) for f in folds}
    val_pos = np.concatenate([fold_rows[f['fold']] for f in folds])
    offsets = np.cumsum([0] + [len(fold_rows[f['fold']]) for f in folds])
    val_keys = pd.DataFrame({'stock_id': data.stock_id.to_numpy()[val_pos], 'target_month': data.target_month.to_numpy()[val_pos],
                             'fold': np.repeat([f['fold'] for f in folds], np.diff(offsets)),
                             'y': data.y.to_numpy(float)[val_pos], 'common': train[val_pos], 'row': val_pos})
    return val_keys, offsets


def execute_folds(fold_fn, payload: dict, folds: list[dict], *, workers: int = 1, log=print) -> dict:
    """Fold마다 fold_fn을 실행한다. workers>1이면 프로세스로 나눈다(작업 프로세스의 진행 줄은 큐로 전달). 반환: {fold 번호: 결과}."""
    done = {}
    report = lambda r: log(f"Fold {r['fold']['fold']:>2} (검증 {r['fold']['valid_year']}) 완료: {r['seconds']:.1f}초")
    if workers > 1 and len(folds) > 1:
        import multiprocessing
        import queue as queue_mod
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        order = sorted(folds, key=lambda f: -f['valid_year'])          # 큰 Fold부터 넣어 마지막 대기를 줄임
        with multiprocessing.Manager() as manager:
            messages = manager.Queue()

            def drain():
                while True:
                    try:
                        log(messages.get_nowait())
                    except queue_mod.Empty:
                        return

            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                     initargs=({**payload, 'queue': messages},)) as pool:
                pending = {pool.submit(fold_fn, f) for f in order}
                while pending:
                    finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    drain()
                    for fut in finished:
                        r = fut.result()
                        done[r['fold']['fold']] = r
                        report(r)
            drain()
    else:
        _init_worker({**payload, 'log': log})
        try:
            for f in folds:
                r = fold_fn(f)
                done[f['fold']] = r
                report(r)
        finally:
            _WORKER.clear()
    return done


def assemble_folds(done: dict, folds: list[dict], val_keys: pd.DataFrame, offsets: np.ndarray, timings: list) -> dict:
    """Fold 결과를 Fold 순서로 합친다: 설정별 예측 배열, 선택 표, 내부 검증 표, 시간(부스팅은 곡선도)."""
    preds, selections, inner, curves = {}, [], [], []
    timings = list(timings)
    for k, f in enumerate(folds):
        r = done[f['fold']]
        for cid, yhat in r['preds'].items():
            preds.setdefault(cid, np.full(len(val_keys), np.nan))[offsets[k]:offsets[k + 1]] = yhat
        selections += r['selections']
        inner += r['inner']
        timings += r['timings']
        curves += r.get('curves', [])
    out = {'val_keys': val_keys, 'predictions': preds, 'selections': pd.DataFrame(selections),
           'inner_ic': pd.concat(inner, ignore_index=True), 'timings': pd.DataFrame(timings)}
    if curves:
        out['curves'] = curves
    return out


def merge_results(parts: list[dict]) -> dict:
    """셀마다 따로 계산한 결과(같은 세션 메모리)를 하나로 합친다. 검증 행·내부 검증 범위가 같아야 하고 설정이 겹치면 안 된다."""
    if not parts:
        raise ValueError('no_parts')
    first = parts[0]
    preds = {}
    for p in parts:
        if not p['val_keys'].equals(first['val_keys']) or p['inner_scope'] != first['inner_scope']:
            raise ValueError('parts_differ_in_rows_or_scope')
        overlap = set(preds) & set(p['predictions'])
        if overlap:
            raise ValueError('parts_overlap:' + sorted(overlap)[0])
        preds.update(p['predictions'])
    cat = lambda key: pd.concat([p[key] for p in parts if len(p[key])], ignore_index=True)
    out = {'val_keys': first['val_keys'], 'predictions': preds, 'selections': cat('selections'), 'inner_ic': cat('inner_ic'),
           'stage1': cat('stage1') if any(len(p['stage1']) for p in parts) else pd.DataFrame(),
           'timings': cat('timings'), 'inner_scope': first['inner_scope']}
    if any('curves' in p for p in parts):
        out['curves'] = [c for p in parts for c in p.get('curves', [])]
    if any('params' in p for p in parts):
        out['params'] = {k: v for p in parts for k, v in p.get('params', {}).items()}
    for key in ('kind', 'phase', 'device', 'followup'):     # 부스팅: 같은 종류·단계·장치·후속 여부끼리만 합친다
        values = {p.get(key) for p in parts}
        if len(values) > 1:
            raise ValueError(f'parts_differ_in_{key}')
        if None not in values:
            out[key] = values.pop()
    return out


# ---------- 부스팅(boosting_model) ----------

def boosting_labels(prepared: dict) -> list[str]:
    """부스팅의 H3 λ 후보 이름(span 12·36·60개월). 입력 준비에 셋 다 있어야 한다."""
    labels = [f'span{n}' for n in boosting.H3_SPANS]
    missing = [x for x in labels if x not in prepared['sigma_labels']]
    if missing:
        raise ValueError('boosting_h3_labels_missing:' + ','.join(missing))
    return labels


def boosting_input_sets() -> dict:
    """부스팅 입력 4개(주 결과): [σ], [Size, B/M, Mom, σ], [H1 지수], [H1 지수, σ]. 부분집합 비교군은 선형에서만."""
    sets = input_sets()
    return {k: sets[k] for k in BOOSTING_SETS}


def _boost_tag(setting, input_set: str | None = None) -> dict:
    kind, target, loss, depth, lr = setting
    tag = {'family': 'boosting', 'model': kind, 'target': target, 'loss': loss, 'hparams': boosting.hparam_tag(depth, lr)}
    if input_set is not None:
        tag['input_set'] = input_set
    return tag


def choose_boosting(fits: list[dict]) -> tuple[dict, int, int]:
    """(H3 λ × 트리 수) 모든 쌍 중 내부 검증 IC 최고 한 쌍. 정확히 같으면 트리 적은 쪽 → H3 긴 기간(λ 큰 쪽) [tie_rules].
    반환: (고른 학습, 고른 트리 수, 최고값 동률 수)."""
    alive = [f for f in fits if np.isfinite(f['curve']).any()]
    if not alive:
        raise ValueError('all_candidates_failed')
    best = max(float(np.nanmax(f['curve'])) for f in alive)
    cands, ties = [], 0
    for f in alive:
        hit = np.flatnonzero(f['curve'] == best)
        ties += len(hit)
        if len(hit):
            cands.append((int(hit[0]) + 1, f))
    trees, chosen = max(cands, key=lambda c: (-c[0], c[1]['decay'] if np.isfinite(c[1]['decay']) else -1.0))
    return chosen, trees, ties


def _curve_record(tag: dict, fitted: dict, *, part: str, fold=None, index_year=None, h3_label=None) -> dict:
    return {**tag, 'part': part, 'fold': fold, 'index_year': index_year, 'h3_label': h3_label,
            'best_trees': fitted['best_trees'], 'seconds': fitted['seconds'],
            'inner': fitted['inner_curve'].astype(np.float32), 'train': fitted['train_curve'].astype(np.float32)}


def select_and_fit_boosting(ctx: FoldContext, setting, *, input_cols, sigma_labels, decay_of, index_col=None,
                            device='cpu', n_threads=None, verbose_every=None, say=None, prefix=''):
    """한 Fold·한 부스팅 설정. σ가 있으면 H3 λ 후보마다 1000트리를 학습하고, (λ × 트리 수) 중 내부 검증 IC 최고 한 쌍을 고른다.
    모든 학습이 같은 학습 행(ctx.fit_rows)을 쓰고 내부 검증은 원수익률로 평가한다. Huber② σ는 같은 행·입력·목표값의
    OLS HuberRegressor(ctx.sigma2, 선형과 같은 값). 반환: (선택 dict, 후보별 표, 학습 목록)."""
    kind, target, loss, depth, lr = setting
    labels = sigma_labels if 'sigma' in input_cols else [None]
    y_fit = ctx.y(target)
    m_inner = ctx.data.target_month.to_numpy()[ctx.inner_rows]
    fits, table = [], []
    for lab in labels:
        cols = _columns(input_cols, lab, index_col)
        sig = None
        if loss == 'huber_q999':
            delta = ctx.q999(target)
        elif loss == 'huber_sigma135':
            sig = ctx.sigma2(cols, target)
            delta = sig['delta']
        else:
            delta = None
        decay = decay_of.get(lab, np.nan)
        try:
            f = boosting.fit_boosting(kind, ctx.X(cols, 'fit'), y_fit, ctx.m_fit, target=target, loss=loss, depth=depth,
                                      learning_rate=lr, delta=delta, inner_X=ctx.X(cols, 'inner'), inner_months=m_inner,
                                      inner_score=lambda p: ctx.scorer.score(p)[0],
                                      train_score=lambda p: ctx.train_scorer.score(p)[0],
                                      device=device, n_threads=n_threads, verbose_every=verbose_every, log=say)
        except ValueError as error:
            table.append({'h3_label': lab, 'decay': decay, 'best_trees': np.nan, 'inner_ic': np.nan, 'inner_ic_tree1000': np.nan,
                          'failure': str(error), 'seconds': np.nan, 'huber_sigma': sig['sigma'] if sig else np.nan})
            continue
        fits.append({'h3_label': lab, 'decay': decay, 'fitted': f, 'columns': cols, 'curve': f['inner_curve'],
                     'huber_sigma': sig['sigma'] if sig else np.nan, 'huber_sigma_converged': sig['converged'] if sig else None})
        table.append({'h3_label': lab, 'decay': decay, 'best_trees': f['best_trees'], 'inner_ic': f['best_inner_ic'],
                      'inner_ic_tree1000': float(f['inner_curve'][-1]), 'failure': '', 'seconds': f['seconds'],
                      'huber_sigma': sig['sigma'] if sig else np.nan})
        if say:
            k = -(-f['best_trees'] // f['train_every'])                   # 고른 트리 수 이상 가장 가까운 기록 지점
            say(f"    {prefix}{' ' + lab if lab else ''}: 트리 {f['best_trees']:>4}/{boosting.MAX_TREES}, "
                f"학습 IC {f['train_curve'][k - 1]:.4f}(트리 {k * f['train_every']}), 내부 검증 IC {f['best_inner_ic']:.4f}, "
                f"{f['seconds']:.1f}초")
    chosen, trees, ties = choose_boosting(fits)
    return ({**chosen, 'trees': trees, 'inner_ic': float(chosen['curve'][trees - 1]), 'ties': ties,
             'n_train': ctx.n_train, 'train_rows_sha256': ctx.rows_sha256}, table, fits)


def build_h1_index_boosting(data, setting, *, inner_scope, decay_of, device='cpu', n_threads=None, log=None,
                            years=range(TRAIN_START, 2022)):
    """지수형 ①(부스팅). build_h1_index와 같은 행·연도 규칙에 ②와 같은 종류·목표값·손실·깊이·학습률, 트리 수는 Y−1 내부 검증.
    반환: (순위 지수 배열, 선택 표, 곡선 목록)."""
    raw_index = np.full(len(data), np.nan)
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    scope = data.predict.to_numpy(bool) if inner_scope == 'full' else train
    h1 = data.h1_available.to_numpy(bool)
    tag = _boost_tag(setting)
    rows, curves = [], []
    for Y in years:
        target_rows = h1 & (year == Y)
        ctx = FoldContext(data, train & (year >= INDEX_TRAIN_START) & (year <= Y - 2), scope & (year == Y - 1), target_rows)
        chosen, _, fits = select_and_fit_boosting(ctx, setting, input_cols=list(H1_PARTS), sigma_labels=None, decay_of=decay_of,
                                                  device=device, n_threads=n_threads, say=log, prefix=f'[① {Y}] ')
        f = chosen['fitted']
        raw_index[target_rows] = boosting.predict_boosting(f, _design(data, chosen['columns'], target_rows, fill=False), chosen['trees'])
        gain = boosting.feature_gain(f, len(H1_PARTS), chosen['trees'])
        share = gain / gain.sum() if gain.sum() > 0 else gain
        rows.append({'index_year': Y, **tag, 'stage1_train_years': f'{INDEX_TRAIN_START}~{Y-2}', 'stage1_inner_valid_year': Y - 1,
                     'trees': chosen['trees'], 'inner_ic': chosen['inner_ic'], 'ties': chosen['ties'], 'n_train': chosen['n_train'],
                     'delta': np.nan if f['delta'] is None else f['delta'], 'huber_sigma': chosen['huber_sigma'],
                     **{'gain_' + p: s for p, s in zip(H1_PARTS, share)}})
        curves += [_curve_record(tag, x['fitted'], part='stage1', index_year=Y) for x in fits]
    member = np.isfinite(raw_index)
    ranked = rank_pm1(pd.Series(raw_index), member, data.base_month.reset_index(drop=True))
    return ranked, pd.DataFrame(rows), curves


def _run_fold_boosting(fold: dict) -> dict:
    """한 Fold의 모든 (입력 조합 × 부스팅 설정). 파일을 읽거나 쓰지 않는다."""
    p = _WORKER
    data = p['data']
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    predict = data.predict.to_numpy(bool)
    scope = predict if p['inner_scope'] == 'full' else train
    t_fold = time.perf_counter()
    lo, hi = fold['train_years']
    ctx = FoldContext(data, train & (year >= lo) & (year <= hi), scope & (year == fold['inner_valid_year']),
                      predict & (year == fold['valid_year']))
    y_all = data.y.to_numpy(float)
    selections, inner, timings, preds, curves, params = [], [], [], {}, [], {}
    n_jobs, i_job = len(p['sets']) * len(p['settings']), 0
    for set_name, cols in p['sets'].items():
        for setting in p['settings']:
            i_job += 1
            t0 = time.perf_counter()
            tag = {**_boost_tag(setting, set_name), 'fold': fold['fold']}
            prefix = f"[Fold {fold['fold']:>2}] {setting[0]} {setting[1]} {setting[2]} {tag['hparams']} {set_name}"
            chosen, table, fits = select_and_fit_boosting(
                ctx, setting, input_cols=cols, sigma_labels=p['labels'], decay_of=p['decay_of'],
                index_col=p['index_cols'].get(setting), device=p['device'], n_threads=p['n_threads'],
                say=_progress, prefix=prefix)
            f, trees = chosen['fitted'], chosen['trees']
            preds[config_id(tag)] = boosting.predict_boosting(f, ctx.X(chosen['columns'], 'out'), trees)
            train_ic = monthly_ic(ctx.m_fit, boosting.predict_boosting(f, ctx.X(chosen['columns'], 'fit'), trees), y_all[ctx.fit_rows])
            gain = boosting.feature_gain(f, len(cols), trees)
            share = gain / gain.sum() if gain.sum() > 0 else gain
            selections.append({**tag, 'valid_year': fold['valid_year'], 'inner_valid_year': fold['inner_valid_year'],
                               'inner_train_years': fold['inner_train_years'], 'min_years_exception': fold['min_years_exception'],
                               'h3_label': chosen['h3_label'], 'h3_decay': chosen['decay'],
                               'h3_span': float(2 / (1 - chosen['decay']) - 1) if np.isfinite(chosen['decay']) else np.nan,
                               'trees': trees, 'ties': chosen['ties'], 'inner_ic': chosen['inner_ic'],
                               'train_ic': mean_ic(train_ic), 'n_train': chosen['n_train'],
                               'train_rows_sha256': chosen['train_rows_sha256'],
                               'delta': np.nan if f['delta'] is None else f['delta'], 'huber_sigma': chosen['huber_sigma'],
                               'huber_sigma_converged': chosen['huber_sigma_converged'],
                               **{'gain_' + c: s for c, s in zip(cols, share)}})
            inner.append(pd.DataFrame(table).assign(**tag))
            curves += [_curve_record(tag, x['fitted'], part='fold', fold=fold['fold'], h3_label=x['h3_label']) for x in fits]
            params.setdefault('|'.join(map(str, setting)), f['params'])
            seconds = time.perf_counter() - t0
            timings.append({'part': 'fold', 'fold': fold['fold'], 'model': setting[0], 'target': setting[1], 'loss': setting[2],
                            'hparams': tag['hparams'], 'input_set': set_name, 'seconds': seconds})
            elapsed = time.perf_counter() - t_fold
            _progress(f"  {prefix} 선택: {chosen['h3_label'] or '-'} 트리 {trees}, 내부 IC {chosen['inner_ic']:.4f}, "
                      f"학습 IC {mean_ic(train_ic):.4f}, {seconds:.0f}초 ({i_job}/{n_jobs}, 남은 예상 {elapsed / i_job * (n_jobs - i_job):.0f}초)")
    return {'fold': fold, 'selections': selections, 'inner': inner, 'timings': timings, 'preds': preds, 'curves': curves,
            'params': params, 'seconds': time.perf_counter() - t_fold}


def _kind_settings(kind: str, settings, followup: bool = False) -> list[tuple]:
    if kind not in boosting.KINDS:
        raise ValueError('boosting_kind_unknown')
    allowed = boosting.settings(kind, followup=followup)
    if not allowed:
        raise ValueError('no_followup_settings_for_kind')
    settings = allowed if settings is None else [tuple(s) for s in settings]
    if any(s[0] != kind for s in settings) or not set(settings) <= set(allowed):
        raise ValueError('settings_kind_mismatch')
    return settings


def _followup_suffix(kind: str) -> str:
    """후속 튜닝 결과 폴더 꼬리표: 깊이 4면 '_d4'."""
    return '_d' + ''.join(str(d) for d in boosting.FOLLOWUP_DEPTHS[kind])


def run_boosting_stage1(prepared: dict, config, kind: str, *, settings=None, inner_scope: str = 'common', device: str | None = None,
                        n_threads: int | None = None, followup: bool = False, log=print) -> dict:
    """지수형 ①: 설정마다 2003~2021년 H1 지수(순위 [−1, 1])를 만든다. 결과는 같은 세션 메모리로 run_boosting(phase='index')에
    넘긴다(파일로 저장했다가 다시 읽지 않음). 반환: kind, settings, inner_scope, n_rows, index(설정별 배열), stage1 표, curves, timings."""
    if inner_scope not in INNER_SCOPES:
        raise ValueError('inner_scope_unknown')
    check_grid(config)
    data = prepared['data']
    settings = _kind_settings(kind, settings, followup)
    device = device or boosting.default_device(kind)
    decay_of = _decay_of(prepared)
    index, tables, curves, timings = {}, [], [], []
    for setting in settings:
        t0 = time.perf_counter()
        index[setting], table, c = build_h1_index_boosting(data, setting, inner_scope=inner_scope, decay_of=decay_of,
                                                           device=device, n_threads=n_threads)
        tables.append(table)
        curves += c
        seconds = time.perf_counter() - t0
        timings.append({'part': 'stage1', 'fold': None, 'model': kind, 'target': setting[1], 'loss': setting[2],
                        'hparams': boosting.hparam_tag(*setting[3:]), 'seconds': seconds})
        log(f"지수형 ① {setting[1]} {setting[2]} {boosting.hparam_tag(*setting[3:])}: {seconds:.0f}초, "
            f"고른 트리 {table.trees.min()}~{table.trees.max()}")
    return {'kind': kind, 'settings': settings, 'inner_scope': inner_scope, 'device': device, 'followup': followup,
            'n_rows': len(data), 'index': index,
            'stage1': _concat(tables), 'curves': curves, 'timings': timings}


def run_boosting(prepared: dict, config, kind: str, *, phase: str, stage1: dict | None = None, settings=None,
                 sets: dict | None = None, folds: list[dict] | None = None, inner_scope: str = 'common', device: str | None = None,
                 n_threads: int | None = None, workers: int = 1, followup: bool = False, log=print) -> dict:
    """한 부스팅 종류·한 단계(전체 또는 일부 설정)를 실행한다. 반환 형식은 run_linear와 같고 curves·params가 더 있다.

    phase='individual': [Size, B/M, Mom, σ]. phase='index': [σ]·[H1 지수]·[H1 지수, σ], 같은 세션의 run_boosting_stage1 결과가
    필요하다(같은 종류·내부 검증 범위·자료, 설정 포함). followup=True면 후속 튜닝 설정(boosting.FOLLOWUP_DEPTHS)만 돌리고
    따로 저장한다. workers>1이면 Fold를 프로세스로 나눈다(n_threads를 함께 줄일 것). 난수 없음.
    """
    if inner_scope not in INNER_SCOPES:
        raise ValueError('inner_scope_unknown')
    if phase not in BOOSTING_PHASES:
        raise ValueError('boosting_phase_unknown')
    grid = check_grid(config)
    data = prepared['data'].copy()
    decay_of = _decay_of(prepared)
    folds = plan_folds(grid) if folds is None else folds
    settings = _kind_settings(kind, settings, followup)
    device = device or boosting.default_device(kind)       # 기본은 config 장치. 다르게 주면 run.json에 표시
    allowed = BOOSTING_PHASES[phase]
    sets = {k: v for k, v in boosting_input_sets().items() if k in allowed} if sets is None else sets
    if not sets or not set(sets) <= set(allowed):
        raise ValueError('boosting_input_set_not_in_phase')
    val_keys, offsets = validation_keys(data, folds)
    index_cols, stage1_rows, curves, timings = {}, pd.DataFrame(), [], []
    if any('h1_index' in cols for cols in sets.values()):
        if (stage1 is None or stage1['kind'] != kind or stage1['inner_scope'] != inner_scope or stage1['device'] != device
                or stage1['n_rows'] != len(data) or stage1.get('followup', False) != followup
                or not set(settings) <= set(stage1['settings'])):
            raise ValueError('stage1_result_required_for_same_kind_scope_device_settings')
        for setting in settings:
            col = 'idx_' + '_'.join(map(str, setting))
            data[col] = stage1['index'][setting]
            index_cols[setting] = col
        hp = {boosting.hparam_tag(*s[3:]) + '|' + s[1] + '|' + s[2] for s in settings}
        mine = lambda r: f"{r['hparams']}|{r['target']}|{r['loss']}" in hp
        st = stage1['stage1']
        stage1_rows = st[[mine(r) for r in st.to_dict('records')]].reset_index(drop=True)
        curves = [c for c in stage1['curves'] if mine(c)]
        timings = [t for t in stage1['timings'] if mine(t)]
    payload = {'data': data, 'labels': boosting_labels(prepared), 'decay_of': decay_of, 'settings': settings, 'sets': sets,
               'index_cols': index_cols, 'inner_scope': inner_scope, 'device': device, 'n_threads': n_threads}
    done = execute_folds(_run_fold_boosting, payload, folds, workers=workers, log=log)
    out = assemble_folds(done, folds, val_keys, offsets, timings)
    out['curves'] = curves + out.get('curves', [])
    out['params'] = {k: v for f in folds for k, v in done[f['fold']]['params'].items()}
    return {**out, 'stage1': stage1_rows, 'inner_scope': inner_scope, 'kind': kind, 'phase': phase, 'device': device,
            'followup': followup}


def curves_frame(curves: list[dict]) -> pd.DataFrame:
    """학습별 곡선 → 넓은 표: 키 열 + inner_0001~inner_1000(트리별 내부 검증 IC) + train_0010~train_1000(10트리마다 학습 IC)."""
    if not curves:
        return pd.DataFrame()
    keys = pd.DataFrame([{k: v for k, v in c.items() if k not in ('inner', 'train')} for c in curves])
    inner = np.vstack([c['inner'] for c in curves])
    train = np.vstack([c['train'] for c in curves])
    every = boosting.MAX_TREES // train.shape[1]
    wide = pd.DataFrame(np.hstack([inner, train]), columns=[f'inner_{i:04d}' for i in range(1, inner.shape[1] + 1)]
                        + [f'train_{every * i:04d}' for i in range(1, train.shape[1] + 1)])
    return pd.concat([keys, wide], axis=1)


def boosting_part_summary(result: dict) -> pd.DataFrame:
    """셀 하나의 진행 확인용 요약: 입력 × 목표값 × 손실 × 깊이·학습률별 바깥 검증 평균 IC와 Fold 평균 학습 IC·고른 트리 수."""
    ic = outer_monthly_ic(result['val_keys'], result['predictions'])
    keys = ['input_set', 'target', 'loss', 'hparams']
    valid = ic.groupby(keys + ['scope']).ic.mean().unstack('scope')
    sel = result['selections'].groupby(keys)
    out = pd.DataFrame({'valid_ic_full': valid['full'], 'valid_ic_common': valid['common'],
                        'train_ic': sel.train_ic.mean(), 'trees_median': sel.trees.median()})
    out['train_minus_valid'] = out.train_ic - out.valid_ic_full
    return out.reset_index()


def expected_boosting_ids(kind: str, sets=BOOSTING_SETS, followup: bool = False) -> set[str]:
    return {config_id(_boost_tag(s, name)) for s in boosting.settings(kind, followup=followup) for name in sets}


def finalize_boosting(root: Path, config, prepared: dict, result: dict, *, started: datetime, out_root: Path | None = None,
                      require_complete: bool = True, gpu_name: str | None = None, log=print) -> dict:
    """부스팅 한 종류·한 단계의 마무리: 종료 조건 → 표·예측·곡선(parquet) 저장. 폴더 <시각>_<종류>_<단계>(후속 튜닝은 끝에 _d4).
    지수형 단계는 ① 표·곡선도 함께 저장한다. 선형 Huber② σ 계산 코드도 해시에 포함."""
    kind, phase = result['kind'], result['phase']
    followup = result.get('followup', False)
    stage1 = result['stage1']
    suffix = f'{kind}_{phase}' + (_followup_suffix(kind) if followup else '')
    return finalize_run(root, config, prepared, result, family='boosting', folder_suffix=suffix,
                        expected_ids=expected_boosting_ids(kind, BOOSTING_PHASES[phase], followup),
                        code=LINEAR_CODE[:1] + ['src/models/boosting.py'] + LINEAR_CODE[1:],
                        device={'boosting': result['device'], 'config_device': boosting.default_device(kind),
                                'device_matches_config': result['device'] == boosting.default_device(kind), 'gpu': gpu_name}, part_keys=['part', 'target', 'loss', 'hparams'],
                        started=started, out_root=out_root, require_complete=require_complete,
                        extra_versions=boosting.library_versions(),
                        extra_tables={'curves': (curves_frame(result['curves']), 'parquet')},
                        extra_record={'kind': kind, 'phase': phase, 'followup': followup, 'library_params_by_setting': result.get('params', {}),
                                      'stage1_rows': int(len(stage1))}, log=log)


def measure_boosting_fit(prepared: dict, config, setting, *, fold_no: int = 16, input_set: str = 'ind:size+bm+mom+sigma',
                         h3_label: str = 'span36', device: str | None = None, n_threads: int | None = None,
                         inner_scope: str = 'common', verbose_every: int = 100, log=print) -> dict:
    """시간 측정용 학습 1회(1000트리, 100트리마다 자세한 줄). 지수형 입력은 쓰지 않는다(①이 필요해서).
    반환: 시간, 고른 트리 수, 내부 검증 IC, 학습 행 수, 내부 검증 예측(장치 간 비교용)."""
    grid = check_grid(config)
    fold = next(f for f in plan_folds(grid) if f['fold'] == fold_no)
    cols_in = input_sets()[input_set]
    if 'h1_index' in cols_in:
        raise ValueError('measure_without_index_inputs')
    data = prepared['data']
    year, train = data.year.to_numpy(), data.train.to_numpy(bool)
    scope = data.predict.to_numpy(bool) if inner_scope == 'full' else train
    lo, hi = fold['train_years']
    ctx = FoldContext(data, train & (year >= lo) & (year <= hi), scope & (year == fold['inner_valid_year']))
    kind, target, loss, depth, lr = setting
    device = device or boosting.default_device(kind)
    cols = _columns(cols_in, h3_label, None)
    delta = ctx.q999(target) if loss == 'huber_q999' else ctx.sigma2(cols, target)['delta'] if loss == 'huber_sigma135' else None
    log(f"[측정] {kind} {target} {loss} {boosting.hparam_tag(depth, lr)} {input_set} {h3_label}, Fold {fold_no}, "
        f"학습 {ctx.n_train:,}행, 장치 {device}")
    f = boosting.fit_boosting(kind, ctx.X(cols, 'fit'), ctx.y(target), ctx.m_fit, target=target, loss=loss, depth=depth,
                              learning_rate=lr, delta=delta, inner_X=ctx.X(cols, 'inner'),
                              inner_months=data.target_month.to_numpy()[ctx.inner_rows],
                              inner_score=lambda p: ctx.scorer.score(p)[0], train_score=lambda p: ctx.train_scorer.score(p)[0],
                              device=device, n_threads=n_threads, verbose_every=verbose_every, log=log)
    log(f"[측정] 끝: {f['seconds']:.1f}초, 트리 {f['best_trees']}, 내부 검증 IC {f['best_inner_ic']:.4f}")
    return {'seconds': f['seconds'], 'best_trees': f['best_trees'], 'best_inner_ic': f['best_inner_ic'], 'n_train': ctx.n_train,
            'fold': fold_no, 'device': device, 'inner_curve': f['inner_curve'],
            'inner_pred': boosting.predict_boosting(f, ctx.X(cols, 'inner'), boosting.MAX_TREES)}


def estimate_boosting_minutes(prepared: dict, kind: str, seconds_by_fold: dict, *, sets=BOOSTING_SETS, n_settings=None) -> dict:
    """한 종류 전체 예상 시간. seconds_by_fold = {Fold 번호: 학습 1회 초}(Fold 1·16 실측)를 학습 행 수로 보간한다.
    학습 횟수: Fold·설정마다 입력 조합별(σ 있으면 λ 10개, 없으면 1개), 지수형 ①은 설정마다 2003~2021년 19회."""
    data = prepared['data']
    train = data.train.to_numpy(bool)
    year = data.year.to_numpy()
    folds = plan_folds({'first_validation_year': 2006, 'last_validation_year': 2021, 'min_inner_train_years': 5,
                        'min_inner_train_exceptions': [1, 2, 3]})
    (f_a, s_a), (f_b, s_b) = sorted(seconds_by_fold.items())
    n_of = lambda lo, hi: int((train & (year >= lo) & (year <= hi)).sum())
    n_a, n_b = (n_of(*next(f for f in folds if f['fold'] == k)['train_years']) for k in (f_a, f_b))
    per_fit = lambda n: s_a + (n - n_a) / (n_b - n_a) * (s_b - s_a)
    all_sets = input_sets()
    fits_per_fold = sum(len(boosting.H3_SPANS) if 'sigma' in all_sets[s] else 1 for s in sets)
    n_settings = len(boosting.settings(kind)) if n_settings is None else n_settings
    folds_sec = sum(per_fit(n_of(*f['train_years'])) * fits_per_fold for f in folds) * n_settings
    stage1_sec = (sum(per_fit(n_of(INDEX_TRAIN_START, Y - 2)) for Y in range(TRAIN_START, 2022)) * n_settings
                  if any('h1_index' in all_sets[s] for s in sets) else 0.0)
    return {'kind': kind, 'settings': n_settings, 'fits_per_fold_setting': fits_per_fold,
            'folds_hours': folds_sec / 3600, 'stage1_hours': stage1_sec / 3600, 'total_hours': (folds_sec + stage1_sec) / 3600}


# ---------- 5) 평가·보고·종료 조건 ----------

CONFIG_KEYS = ['family', 'model', 'target', 'loss', 'input_set']


def config_id(tag: dict) -> str:
    """선형: family|model|target|loss|input_set. 부스팅은 끝에 |d<깊이>_lr<학습률>을 붙인다(선형 ID는 그대로)."""
    base = '|'.join(str(tag[k]) for k in CONFIG_KEYS)
    hp = tag.get('hparams')
    return base + '|' + hp if isinstance(hp, str) and hp else base


def split_config_id(cid: str) -> dict:
    return dict(zip(CONFIG_KEYS + ['hparams'], cid.split('|')))


def _selection_ids(sel: pd.DataFrame) -> list[str]:
    cols = CONFIG_KEYS + (['hparams'] if 'hparams' in sel else [])
    return [config_id(r) for r in sel[cols].to_dict('records')]


def _scopes(val_keys: pd.DataFrame):
    full = np.ones(len(val_keys), dtype=bool)
    return (('full', full), ('common', val_keys.common.to_numpy(bool)))


def outer_monthly_ic(val_keys: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """설정 × 범위(전체 종목·공통 표본) × 목표월 IC. 평가는 원수익률."""
    rows = []
    months, y = val_keys.target_month.to_numpy(), val_keys.y.to_numpy(float)
    for cid, yhat in predictions.items():
        for scope, mask in _scopes(val_keys):
            t = monthly_ic(months[mask], yhat[mask], y[mask]).reset_index()
            rows.append(t.assign(config=cid, scope=scope))
    ic = pd.concat(rows, ignore_index=True)
    parts = pd.DataFrame([split_config_id(c) for c in ic.config], index=ic.index)
    return pd.concat([parts, ic], axis=1)


def scipy_crosscheck(val_keys: pd.DataFrame, predictions: dict, ic: pd.DataFrame) -> float:
    """scipy.stats.spearmanr로 모든 (설정, 범위, 월) IC를 독립 재계산한 최대 절대 차이."""
    from scipy.stats import spearmanr
    ref = ic.set_index(['config', 'scope', 'target_month']).ic
    y = val_keys.y.to_numpy(float)
    groups = {scope: pd.Series(np.flatnonzero(mask)).groupby(val_keys.target_month.to_numpy()[mask]).apply(np.asarray)
              for scope, mask in _scopes(val_keys)}
    err = 0.0
    for cid, yhat in predictions.items():
        for scope, by_month in groups.items():
            for month, pos in by_month.items():
                ok = np.isfinite(yhat[pos]) & np.isfinite(y[pos])
                with np.errstate(all='ignore'), warnings.catch_warnings():
                    warnings.simplefilter('ignore')               # 상수 입력은 아래에서 결측끼리 비교
                    value = float(spearmanr(yhat[pos][ok], y[pos][ok]).statistic)
                mine = float(ref.loc[(cid, scope, month)])
                if np.isnan(value) and np.isnan(mine):
                    continue                                       # 둘 다 결측(상수 예측 등)이면 일치
                if np.isnan(value) or np.isnan(mine):
                    return float('inf')                            # 한쪽만 결측이면 불일치
                err = max(err, abs(value - mine))
    return err


def metrics(s: pd.Series) -> dict:
    s = s.astype(float)
    sd = float(s.std(ddof=1))
    return {'months': int(s.notna().sum()), 'mean_ic': float(s.mean()), 'ic_sd': sd,
            'icir': float(s.mean() / sd) if sd > 0 else np.nan, 'ic_positive_share': float(s.gt(0).mean())}


def r2_pct(y: np.ndarray, yhat: np.ndarray) -> float:
    """GKX 식 19: 1 − Σ(r − r̂)²/Σr²(평균 차감 없음), 수익률 단위 예측(원값·윈저 목표)만."""
    return float(100 * (1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)))


def report_tables(result: dict, ic: pd.DataFrame) -> dict:
    sel, preds, val_keys = result['selections'], result['predictions'], result['val_keys']
    sel = sel.assign(config=_selection_ids(sel))
    train_by_config = sel.groupby('config').train_ic.mean()
    masks = dict(_scopes(val_keys))
    y = val_keys.y.to_numpy(float)
    perf = []
    for (cid, scope), part in ic.groupby(['config', 'scope'], sort=True):
        s = part.set_index('target_month').ic
        cfg = split_config_id(cid)
        m = masks[scope]
        row = {**cfg, 'scope': scope, **metrics(s), 'train_ic_fold_mean': float(train_by_config.loc[cid]),
               'r2_pct': r2_pct(y[m], preds[cid][m]) if cfg['target'] in ('raw', 'winsor_1_99') else np.nan}
        row['train_minus_valid'] = row['train_ic_fold_mean'] - row['mean_ic']
        year = s.index.str[:4].astype(int)
        row.update({f'mean_ic_{k}': float(s[(year >= a) & (year <= b)].mean()) for k, (a, b) in PERIODS.items()})
        perf.append(row)
    fold_ic = ic[ic.scope.eq('full')].assign(valid_year=lambda f: f.target_month.str[:4].astype(int))
    fold_ic = fold_ic.groupby(['config', 'valid_year']).ic.mean().rename('valid_ic').reset_index()
    fold_table = sel.merge(fold_ic, on=['config', 'valid_year'], how='left', validate='one_to_one')
    fold_table['train_minus_valid'] = fold_table.train_ic - fold_table.valid_ic
    tuning = sel.groupby(['model', 'input_set']).agg(
        fold_settings=('fold', 'size'), ties_over_1=('ties', lambda s: int((s > 1).sum()))).reset_index()
    return {'performance_summary': pd.DataFrame(perf), 'fold_table': fold_table, 'tuning_ties': tuning,
            'monthly_ic': ic, 'inner_validation_ic': result['inner_ic'], 'stage1_h1_index': result['stage1'],
            'timings': result['timings']}


def part_summary(result: dict) -> pd.DataFrame:
    """셀(모델 × 손실) 하나의 요약: 입력 조합 × 목표값별 바깥 검증 평균 IC(전체 종목·공통 표본)와 Fold 평균 학습 IC.
    선택용이 아니라 진행 확인용(최종 표는 finalize_linear)."""
    ic = outer_monthly_ic(result['val_keys'], result['predictions'])
    valid = ic.groupby(['input_set', 'target', 'scope']).ic.mean().unstack('scope')
    train = result['selections'].groupby(['input_set', 'target']).train_ic.mean()
    out = pd.DataFrame({'valid_ic_full': valid['full'], 'valid_ic_common': valid['common'], 'train_ic': train})
    out['train_minus_valid'] = out.train_ic - out.valid_ic_full
    order = list(input_sets())
    return (out.reset_index().assign(_o=lambda f: f.input_set.map(order.index))
            .sort_values(['_o', 'target']).drop(columns='_o').reset_index(drop=True))


def prediction_frame(result: dict) -> pd.DataFrame:
    """저장용 넓은 표: 검증 행 키 + 설정별 예측 열."""
    keys = result['val_keys'].drop(columns='row')
    return pd.concat([keys, pd.DataFrame(result['predictions'], index=keys.index)], axis=1)


def end_checks(prepared: dict, result: dict, ic: pd.DataFrame, *, full_run: bool) -> dict:
    """종료 조건(노션 §11). 하나라도 False면 결과를 쓰지 않는다."""
    data, preds, sel, val_keys = prepared['data'], result['predictions'], result['selections'], result['val_keys']
    years = sorted(sel.valid_year.unique())
    expected = data[data.predict & data.year.isin(years)]
    same_rows = (len(expected) == len(val_keys)
                 and expected[['stock_id', 'target_month']].sort_values(['target_month', 'stock_id']).reset_index(drop=True)
                 .equals(val_keys[['stock_id', 'target_month']].sort_values(['target_month', 'stock_id']).reset_index(drop=True)))
    all_predicted = all(len(v) == len(val_keys) and np.isfinite(v).all() for v in preds.values())
    months_ok = ic.groupby(['config', 'scope']).ic.apply(lambda s: int(s.notna().sum()))
    checks = {
        'sigma_decay097_equals_h3_parquet': bool(prepared['checks'].get('sigma_decay097_equals_h3_parquet')),
        'no_test_period': bool(val_keys.target_month.lt('2022-01').all() and data.target_month.lt('2022-01').all()),
        'same_training_rows_across_configs': bool(sel.groupby('fold').train_rows_sha256.nunique().eq(1).all()),
        'predictions_equal_all_stocks_with_returns': bool(same_rows and all_predicted),
        'scipy_independent_max_error': scipy_crosscheck(val_keys, preds, ic),
    }
    checks['scipy_independent_agrees'] = checks['scipy_independent_max_error'] <= 1e-12
    if full_run:
        checks['validation_prediction_rows'] = len(val_keys) == EXPECTED_VALIDATION_PREDICTION_ROWS
        checks['all_192_months_ic'] = bool(months_ok.eq(192).all())
    return checks


# ---------- 실행 ----------

def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git(root: Path, *args) -> str | None:
    try:
        return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, check=True, timeout=20).stdout.strip()
    except Exception:
        return None


def _makespan(seconds: list[float], workers: int) -> float:
    """큰 작업부터 가장 한가한 작업자에게 배정했을 때 끝나는 시각(병렬 예상 시간)."""
    load = [0.0] * max(1, workers)
    for s in sorted(seconds, reverse=True):
        i = load.index(min(load))
        load[i] += s
    return max(load)


def estimate_part_seconds(prepared, config, *, model: str, loss: str, inner_scope='common', workers: int = 1, log=print) -> dict:
    """한 셀(모델 × 손실)의 예상 시간. Fold 1·16에서 σ 있는 조합·없는 조합 하나씩 실측해 학습 행 수로 보간하고,
    지수형 ①은 목표값 하나를 실측해 ×3. 병렬이면 Fold 시간을 작업자에 배정해 계산한다."""
    grid = check_grid(config)
    folds = plan_folds(grid)
    sets = input_sets()
    settings = part_settings(model, loss)
    sample = {'ind:size+bm+mom+sigma': sets['ind:size+bm+mom+sigma'], 'ind:size+bm+mom': sets['ind:size+bm+mom']}
    measured = {}
    for fold in (folds[0], folds[-1]):
        r = run_linear(prepared, config, inner_scope=inner_scope, folds=[fold], settings=settings, sets=sample, log=lambda *_: None)
        t = r['timings']
        measured[fold['fold']] = {name: float(t[t.input_set.eq(name)].seconds.sum()) for name in sample}
    data = prepared['data']
    n = {f['fold']: int((data.train & data.year.between(*f['train_years'])).sum()) for f in folds}
    n_sigma = sum('sigma' in c for c in sets.values())
    n_plain = len(sets) - n_sigma
    per_fold = []
    for f in folds:
        w = (n[f['fold']] - n[1]) / (n[16] - n[1])
        per = {k: measured[1][k] + w * (measured[16][k] - measured[1][k]) for k in sample}
        per_fold.append(n_sigma * per['ind:size+bm+mom+sigma'] + n_plain * per['ind:size+bm+mom'])
    decay_of = dict(zip(prepared['lambda_candidates'].label, prepared['lambda_candidates'].decay))
    t0 = time.perf_counter()
    build_h1_index(data, model=model, target='raw', loss=loss, inner_scope=inner_scope, decay_of=decay_of)
    stage1 = (time.perf_counter() - t0) * len(settings)
    folds_wall = _makespan(per_fold, workers)
    est = {'model': model, 'loss': loss, 'workers': workers, 'folds_seconds_serial': sum(per_fold),
           'folds_seconds_wall': folds_wall, 'stage1_seconds': stage1, 'total_minutes': (folds_wall + stage1) / 60,
           'measured_fold1': measured[1], 'measured_fold16': measured[16]}
    log(f"[{model} × {loss}] 예상 약 {est['total_minutes']:.1f}분 (Fold {folds_wall/60:.1f}분, 작업자 {workers}개 + 지수형 ① {stage1/60:.1f}분)")
    return est


def expected_config_ids() -> set[str]:
    return {config_id({'family': 'linear', 'model': m, 'target': t, 'loss': l, 'input_set': s})
            for m, t, l in linear_settings() for s in input_sets()}


LINEAR_CODE = ['src/evaluation/model_test.py', 'src/models/linear.py', 'src/features/h3.py', 'src/evaluation/expanding_window.py',
               'src/preprocessing/panel.py', 'config/experiment.yaml']


def finalize_linear(root: Path, config, prepared: dict, result: dict, *, started: datetime, out_root: Path | None = None,
                    require_complete: bool = True, log=print) -> dict:
    """합친 결과로 종료 조건 검사 → 보고 표 → 저장(쓰기 전용 새 폴더). 검사 실패면 표·예측을 저장하지 않고 run.json에 실패를 남긴다."""
    return finalize_run(root, config, prepared, result, family='linear', folder_suffix='linear',
                        expected_ids=expected_config_ids(), code=LINEAR_CODE, device={'linear': 'cpu'},
                        part_keys=['model', 'loss'], started=started, out_root=out_root,
                        require_complete=require_complete, log=log)


def finalize_run(root: Path, config, prepared: dict, result: dict, *, family: str, folder_suffix: str, expected_ids: set,
                 code: list[str], device: dict, part_keys: list[str], started: datetime, out_root: Path | None = None,
                 require_complete: bool = True, extra_versions: dict | None = None, extra_tables: dict | None = None,
                 extra_record: dict | None = None, log=print) -> dict:
    """선형·부스팅 공통 마무리: 종료 조건 → 보고 표 → 저장(폴더 <시각>_<folder_suffix>, 쓰기 전용).
    extra_tables: {이름: (표, 'csv'|'parquet')} — 통과했을 때만 함께 저장."""
    t0 = time.perf_counter()
    missing = expected_ids - set(result['predictions'])
    if require_complete and missing:
        raise ValueError(f'{family}_parts_missing:{len(missing)}')
    ic = outer_monthly_ic(result['val_keys'], result['predictions'])
    checks = end_checks(prepared, result, ic, full_run=not missing)
    tables = report_tables(result, ic)
    passed = all(v for k, v in checks.items() if k != 'scipy_independent_max_error')
    log(f"종료 조건 {'통과' if passed else '실패'} ({time.perf_counter() - t0:.0f}초): "
        + ', '.join(f'{k}={v}' for k, v in checks.items()))
    out_root = Path(out_root) if out_root is not None else root / config['PATHS']['run_records'] / 'model_test'
    folder = out_root / (started.strftime('%Y%m%dT%H%M%S') + '_' + folder_suffix)
    folder.mkdir(parents=True, exist_ok=False)
    if passed:
        for name, table in tables.items():
            table.to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
        prediction_frame(result).to_parquet(folder / 'validation_predictions.parquet', index=False)
        for name, (table, fmt) in (extra_tables or {}).items():
            if fmt == 'parquet':
                table.to_parquet(folder / f'{name}.parquet', index=False)
            else:
                table.to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
    import os
    import scipy
    import sklearn
    record = {'stage': 'model_test', 'family': family, 'status': 'PASSED' if passed else 'FAILED_END_CHECKS',
              'started': started.isoformat(), 'finished': datetime.now().isoformat(), 'command': ' '.join(sys.argv),
              'inner_scope': result['inner_scope'], 'checks': checks, 'input_checks': prepared['checks'],
              'input_sha256': prepared['inputs'], 'configs': len(result['predictions']), 'missing_configs': sorted(missing),
              'commit': _git(root, 'rev-parse', 'HEAD'), 'git_status_porcelain': _git(root, 'status', '--porcelain'),
              'model_grid': config['MODEL_GRID'], 'folds_plan': plan_folds(config['MODEL_GRID']),
              'compute_seconds_by_part': {'|'.join(map(str, k)): round(float(s), 1) for k, s in
                                          result['timings'].groupby(part_keys).seconds.sum().items()},
              'code_sha256': {p: sha256(root / p) for p in code if (root / p).is_file()},
              'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                           'scikit-learn': sklearn.__version__, 'scipy': scipy.__version__, **(extra_versions or {})},
              'device': {'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
                         **device},
              **(extra_record or {}),
              'outputs_sha256': {p.name: sha256(p) for p in sorted(folder.iterdir())}}
    for name, value in prepared['inputs'].items():
        if sha256(root / name) != value:
            record['status'] = 'FAILED_SOURCE_CHANGED'
    (folder / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    log(f"저장: {folder} ({record['status']})")
    return {'status': record['status'], 'run_dir': str(folder), 'checks': checks, 'tables': tables}


def run(root: Path, config, *, execute: bool = False, inner_scope: str = 'common', out_root: Path | None = None,
        workers: int = 1, prepared: dict | None = None, log=print) -> dict:
    """execute=False: 입력 준비·검사와 셀(모델 × 손실)별 예상 시간만. execute=True: 6개 셀을 차례로 계산 → 합침 → 종료 조건 → 저장."""
    started = datetime.now()
    t0 = time.perf_counter()
    prepared = prepare_inputs(root, config) if prepared is None else prepared
    log(f"입력 준비 {time.perf_counter() - t0:.0f}초: 검증 예측 행 {prepared['checks']['validation_prediction_rows']:,}, "
        f"학습 행(2003~) {prepared['checks']['training_rows_2003_2021']:,}, λ=0.97 σ 일치")
    summary = {'stage': 'model_test', 'family': 'linear', 'execute': execute, 'inner_scope': inner_scope, 'workers': workers,
               'input_checks': prepared['checks'], 'input_sha256': prepared['inputs'],
               'settings': len(linear_settings()), 'input_sets': len(input_sets()), 'folds': 16}
    if not execute:
        ests = [estimate_part_seconds(prepared, config, model=m, loss=l, inner_scope=inner_scope, workers=workers, log=log)
                for m, l in LINEAR_PARTS]
        summary['estimate_minutes'] = {f"{e['model']}|{e['loss']}": round(e['total_minutes'], 1) for e in ests}
        summary['estimate_total_minutes'] = round(sum(e['total_minutes'] for e in ests), 1)
        log(f"전체 예상 약 {summary['estimate_total_minutes']:.0f}분 + 마무리(종료 조건·저장) 약 2분")
        return summary
    parts = []
    for m, l in LINEAR_PARTS:
        t_part = time.perf_counter()
        parts.append(run_linear(prepared, config, inner_scope=inner_scope, settings=part_settings(m, l), workers=workers, log=log))
        log(f'[{m} × {l}] 완료 {time.perf_counter() - t_part:.0f}초')
    out = finalize_linear(root, config, prepared, merge_results(parts), started=started, out_root=out_root, log=log)
    return {**summary, **out, 'seconds': time.perf_counter() - t0}


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description='v2 model_test: linear_model(OLS·Ridge)')
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--inner-scope', choices=INNER_SCOPES, default='common')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--out-root', default=None)
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / args.config).read_text(encoding='utf-8'))
    out = run(root, config, execute=args.execute, inner_scope=args.inner_scope, workers=args.workers,
              out_root=Path(args.out_root) if args.out_root else None)
    out.pop('tables', None)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
