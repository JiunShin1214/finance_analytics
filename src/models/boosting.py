"""v2 부스팅 모델(MODEL_GRID): XGBoost·LightGBM 7종. 종류마다 떼어 쓸 수 있게 설정을 KINDS 한 곳에 모은다.

종류(kind)            라이브러리  목적함수                                       정답
xgb_reg               XGBoost    reg:squarederror / reg:pseudohubererror(δ)      수익률 목표 3종
lgb_reg               LightGBM   regression / huber(δ)                           〃
xgb_lambdarank        XGBoost    rank:ndcg(전체 쌍, 이득 2^등급−1)               매달 10분위 0~9
lgb_lambdarank        LightGBM   lambdarank(truncation 1000, 이득 2^등급−1)      〃
xgb_pairwise          XGBoost    rank:pairwise(전체 쌍)                          〃
xgb_rankreg           XGBoost    reg:squarederror                                매달 백분위 r/(N+1)
lgb_rankreg           LightGBM   regression                                      〃

공통 설정(MODEL_GRID): 깊이 {1,2,3} × 학습률 {0.01, 0.1}, 트리 1000개 모두 학습, 최소 헤시안 0.001, 잎 최소 행 1(LGB),
L1·L2 0, 행·특징 추출 없음(난수 없음), max_bin 256, LGB num_leaves = 2^깊이.
트리 수는 1000개를 다 학습한 뒤 내부 검증 IC가 가장 높은 지점(같으면 적은 쪽)을 고른다(인내값 없음).

학습 중 기록: 트리마다 내부 검증 IC, train_every(기본 10)트리마다 학습 IC. 두 라이브러리 모두 평가 함수에 그 시점까지의
누적 예측을 넘겨주므로 트리마다 다시 예측하지 않는다(predict(iteration_range/num_iteration)와 같은 값, 테스트로 확인).
IC 계산 함수는 호출하는 쪽이 넘긴다(이 모듈은 evaluation을 불러오지 않는다). 파일을 쓰지 않는다.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Callable, Mapping

import numpy as np

LIBRARIES = ('xgboost', 'lightgbm')
DEPTHS = (1, 2, 3)
LEARNING_RATES = (0.01, 0.1)
MAX_TREES = 1000
MIN_HESSIAN = 0.001
MIN_DATA_IN_LEAF = 1
MAX_BIN = 256
RANK_PAIRS = 1000            # 한 달 종목 수 이상이면 전체 쌍(XGB topk의 k, LGB truncation level). 넘으면 오류
N_GRADES = 10                # 10분위 0~9
REGRESSION_TARGETS = ('raw', 'winsor_1_99', 'monthly_zscore')
REGRESSION_LOSSES = ('mse', 'huber_q999', 'huber_sigma135')

KINDS = {
    'xgb_reg':        {'library': 'xgboost',  'task': 'regression', 'targets': REGRESSION_TARGETS, 'losses': REGRESSION_LOSSES},
    'lgb_reg':        {'library': 'lightgbm', 'task': 'regression', 'targets': REGRESSION_TARGETS, 'losses': REGRESSION_LOSSES},
    'xgb_lambdarank': {'library': 'xgboost',  'task': 'rank', 'targets': ('decile',), 'losses': ('lambdarank',)},
    'lgb_lambdarank': {'library': 'lightgbm', 'task': 'rank', 'targets': ('decile',), 'losses': ('lambdarank',)},
    'xgb_pairwise':   {'library': 'xgboost',  'task': 'rank', 'targets': ('decile',), 'losses': ('pairwise',)},
    'xgb_rankreg':    {'library': 'xgboost',  'task': 'regression', 'targets': ('percentile',), 'losses': ('mse',)},
    'lgb_rankreg':    {'library': 'lightgbm', 'task': 'regression', 'targets': ('percentile',), 'losses': ('mse',)},
}


def check_boosting_grid(grid: Mapping) -> None:
    """config MODEL_GRID의 부스팅 부분이 이 모듈의 고정값과 같은지 확인한다."""
    expected = {
        'rank_models': ['lambdarank', 'pairwise', 'rank_regression'], 'rank_label': 'monthly_decile_0_9', 'rank_pairs': 'all',
        'lambdarank_gain': 'exp2_minus_1', 'rank_regression_target': 'monthly_percentile', 'libraries': list(LIBRARIES),
        'depths': list(DEPTHS), 'learning_rates': list(LEARNING_RATES), 'hyperparameter_reporting': 'all_candidates',
        'max_trees': MAX_TREES, 'tree_count_selection': 'best_inner_validation_ic_over_all_trees',
        'lightgbm_num_leaves': 'two_power_depth', 'min_hessian_in_leaf': MIN_HESSIAN, 'min_data_in_leaf': MIN_DATA_IN_LEAF,
        'l2': 0, 'l1': 0, 'subsample': 1.0, 'colsample': 1.0, 'max_bin': MAX_BIN,
        'regression_losses': list(REGRESSION_LOSSES), 'regression_targets': list(REGRESSION_TARGETS),
    }
    bad = [k for k, v in expected.items() if grid.get(k) != v]
    if bad:
        raise ValueError('boosting_grid_not_approved:' + ','.join(bad))


def settings(kind: str) -> list[tuple]:
    """한 종류의 설정 목록 (kind, target, loss, depth, learning_rate). 회귀 54개, 순위 6개."""
    spec = KINDS[kind]
    return [(kind, t, l, d, lr) for t in spec['targets'] for l in spec['losses'] for d in DEPTHS for lr in LEARNING_RATES]


def hparam_tag(depth: int, learning_rate: float) -> str:
    return f'd{depth}_lr{learning_rate:g}'


# ---------- 라이브러리 설정 ----------

def _check_setting(kind, target, loss, depth, learning_rate, delta):
    if kind not in KINDS:
        raise ValueError('boosting_kind_unknown')
    spec = KINDS[kind]
    if target not in spec['targets'] or loss not in spec['losses']:
        raise ValueError('boosting_target_or_loss_not_approved')
    if depth not in DEPTHS or learning_rate not in LEARNING_RATES:
        raise ValueError('boosting_hyperparameter_not_approved')
    if loss.startswith('huber'):
        if delta is None or not (math.isfinite(delta) and delta > 0):
            raise ValueError('huber_delta_required')
    elif delta is not None:
        raise ValueError('non_huber_has_no_delta')


def xgb_params(kind, loss, depth, learning_rate, delta=None, *, device='cpu', n_threads=None) -> dict:
    p = {'tree_method': 'hist', 'device': device, 'max_depth': depth, 'eta': learning_rate,
         'min_child_weight': MIN_HESSIAN, 'lambda': 0.0, 'alpha': 0.0, 'gamma': 0.0,
         'subsample': 1.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 1.0, 'colsample_bynode': 1.0,
         'max_bin': MAX_BIN, 'disable_default_eval_metric': 1, 'seed': 0, 'verbosity': 0,
         'nthread': n_threads or os.cpu_count()}
    if loss == 'lambdarank':
        p.update(objective='rank:ndcg', ndcg_exp_gain=True)
    elif loss == 'pairwise':
        p.update(objective='rank:pairwise')
    elif loss == 'mse':
        p.update(objective='reg:squarederror')
    else:
        p.update(objective='reg:pseudohubererror', huber_slope=float(delta))
    if KINDS[kind]['task'] == 'rank':
        p.update(lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=RANK_PAIRS)
    return p


def lgb_params(kind, loss, depth, learning_rate, delta=None, *, device='cpu', n_threads=None) -> dict:
    p = {'boosting': 'gbdt', 'device_type': device, 'max_depth': depth, 'num_leaves': 2 ** depth,
         'learning_rate': learning_rate, 'min_data_in_leaf': MIN_DATA_IN_LEAF, 'min_sum_hessian_in_leaf': MIN_HESSIAN,
         'lambda_l1': 0.0, 'lambda_l2': 0.0, 'min_gain_to_split': 0.0,
         'bagging_fraction': 1.0, 'bagging_freq': 0, 'feature_fraction': 1.0, 'feature_fraction_bynode': 1.0,
         'max_bin': MAX_BIN, 'metric': 'None', 'deterministic': True, 'force_row_wise': True, 'seed': 0,
         'verbose': -1, 'num_threads': n_threads or os.cpu_count(), 'feature_pre_filter': False}
    if loss == 'lambdarank':
        p.update(objective='lambdarank', lambdarank_truncation_level=RANK_PAIRS,
                 label_gain=[float(2 ** i - 1) for i in range(N_GRADES)])
    elif loss == 'mse':
        p.update(objective='regression')
    else:
        p.update(objective='huber', alpha=float(delta))
    return p


def _groups(months) -> np.ndarray:
    """정렬된 월 배열 → 달마다 행 수(순위 학습의 그룹). 달이 이어져 있지 않거나 한 달이 RANK_PAIRS를 넘으면 오류."""
    months = np.asarray(months)
    change = np.r_[True, months[1:] != months[:-1]]
    starts = np.flatnonzero(change)
    if len(set(months[starts].tolist())) != len(starts):
        raise ValueError('rank_groups_not_contiguous')
    sizes = np.diff(np.r_[starts, len(months)])
    if sizes.max() > RANK_PAIRS:
        raise ValueError('rank_group_exceeds_all_pairs_limit')
    return sizes


def _check_label(kind, y):
    if KINDS[kind]['task'] == 'rank':
        if not (np.isin(y, np.arange(N_GRADES))).all():
            raise ValueError('rank_label_must_be_grade_0_9')
        return y.astype(int)
    return y


# ---------- 학습 ----------

def fit_boosting(kind: str, X, y, months, *, target: str, loss: str, depth: int, learning_rate: float,
                 delta: float | None = None, inner_X=None, inner_months=None, inner_score: Callable,
                 train_score: Callable, train_every: int = 10, device: str = 'cpu', n_threads: int | None = None,
                 verbose_every: int | None = None, log: Callable | None = None) -> dict:
    """학습 행(X, y)으로 1000트리를 학습하고 트리별 곡선과 고른 트리 수를 돌려준다.

    inner_score(예측) → 내부 검증 평균 IC, train_score(예측) → 학습 평균 IC. months는 학습 행의 목표월(정렬, 순위 그룹용).
    반환: booster, library, kind, inner_curve(길이 1000), train_curve(길이 1000/train_every, 트리 train_every·2·train_every…),
          best_trees, best_inner_ic, seconds, params(실제 설정 전체), device.
    """
    _check_setting(kind, target, loss, depth, learning_rate, delta)
    X = np.asarray(X, dtype=float)
    y = _check_label(kind, np.asarray(y, dtype=float).reshape(-1))
    if X.ndim != 2 or len(X) != len(y) or not len(y) or not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError('boosting_input_invalid')
    inner_X = np.asarray(inner_X, dtype=float)
    if inner_X.ndim != 2 or inner_X.shape[1] != X.shape[1] or not np.isfinite(inner_X).all():
        raise ValueError('boosting_inner_input_invalid')
    if MAX_TREES % train_every:
        raise ValueError('train_every_must_divide_max_trees')
    rank = KINDS[kind]['task'] == 'rank'
    groups = _groups(months) if rank else None
    inner_curve = np.full(MAX_TREES, np.nan)
    train_curve = np.full(MAX_TREES // train_every, np.nan)
    counter = {'inner': 0, 'train': 0}
    t0 = time.perf_counter()

    def record(pred, which):
        counter[which] += 1
        k = counter[which]
        if which == 'inner':
            inner_curve[k - 1] = inner_score(np.asarray(pred, float))
            if verbose_every and log and k % verbose_every == 0:
                j = int(np.nanargmax(inner_curve[:k]))
                log(f'      트리 {k:>4}/{MAX_TREES}: 내부 검증 IC {inner_curve[k - 1]:.4f} (지금까지 최고 {inner_curve[j]:.4f} @ {j + 1}), '
                    f'학습 IC {train_curve[k // train_every - 1] if k >= train_every else np.nan:.4f}, 경과 {time.perf_counter() - t0:.1f}초')
        elif k % train_every == 0:
            train_curve[k // train_every - 1] = train_score(np.asarray(pred, float))

    lib = KINDS[kind]['library']
    if lib == 'xgboost':
        import xgboost as xgb
        params = xgb_params(kind, loss, depth, learning_rate, delta, device=device, n_threads=n_threads)
        qid = np.repeat(np.arange(len(groups)), groups) if rank else None
        dtrain = xgb.DMatrix(X, label=y, qid=qid)
        inner_qid = np.repeat(np.arange(len(_groups(inner_months))), _groups(inner_months)) if rank else None
        dinner = xgb.DMatrix(inner_X, qid=inner_qid)

        def metric(pred, dm):
            record(pred, 'train' if dm is dtrain else 'inner')
            return 'ic', 0.0

        # train을 먼저 평가해야 inner의 자세한 줄에 같은 트리의 학습 IC가 찍힌다.
        booster = xgb.train(params, dtrain, MAX_TREES, evals=[(dtrain, 'train'), (dinner, 'inner')],
                            custom_metric=metric, verbose_eval=False)
        used = json.loads(booster.save_config())
    else:
        import lightgbm as lgb
        params = lgb_params(kind, loss, depth, learning_rate, delta, device=device, n_threads=n_threads)
        ds_params = {'max_bin': MAX_BIN, 'verbose': -1, 'feature_pre_filter': False, 'min_data_in_leaf': MIN_DATA_IN_LEAF}
        dtrain = lgb.Dataset(X, label=y, group=groups, params=ds_params, free_raw_data=False)
        dinner = lgb.Dataset(inner_X, label=np.zeros(len(inner_X)), group=_groups(inner_months) if rank else None,
                             reference=dtrain, params=ds_params, free_raw_data=False)

        def metric(pred, data):
            record(pred, 'train' if data is dtrain else 'inner')
            return 'ic', 0.0, True

        booster = lgb.train(params, dtrain, MAX_TREES, valid_sets=[dtrain, dinner], valid_names=['train', 'inner'], feval=metric)
        used = {**params, 'library_dump_parameters': booster.dump_model(num_iteration=1).get('parameters')}
    if counter['inner'] != MAX_TREES or counter['train'] != MAX_TREES:
        raise ValueError('boosting_curve_incomplete')
    if not np.isfinite(inner_curve).any():
        raise ValueError('boosting_inner_ic_all_missing')
    best = int(np.nanargmax(inner_curve))          # 같은 최고값이면 처음(트리 적은 쪽)
    return {'booster': booster, 'library': lib, 'kind': kind, 'target': target, 'loss': loss, 'depth': depth,
            'learning_rate': learning_rate, 'delta': delta, 'inner_curve': inner_curve, 'train_curve': train_curve,
            'train_every': train_every, 'best_trees': best + 1, 'best_inner_ic': float(inner_curve[best]),
            'n_train': len(y), 'seconds': time.perf_counter() - t0, 'params': used, 'device': device}


def predict_boosting(fitted: Mapping, X, n_trees: int | None = None) -> np.ndarray:
    """첫 n_trees(기본: 고른 트리 수)개 트리로 예측."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('boosting_predict_input')
    n = int(fitted['best_trees'] if n_trees is None else n_trees)
    if not 1 <= n <= MAX_TREES:
        raise ValueError('boosting_tree_count')
    if fitted['library'] == 'xgboost':
        import xgboost as xgb
        return fitted['booster'].predict(xgb.DMatrix(X), iteration_range=(0, n))
    return fitted['booster'].predict(X, num_iteration=n)


def feature_gain(fitted: Mapping, n_features: int, n_trees: int | None = None) -> np.ndarray:
    """첫 n_trees개 트리의 특징별 총 이득(gain). 쓰이지 않은 특징은 0."""
    n = int(fitted['best_trees'] if n_trees is None else n_trees)
    if fitted['library'] == 'xgboost':
        score = fitted['booster'][:n].get_score(importance_type='total_gain')
        return np.array([float(score.get(f'f{i}', 0.0)) for i in range(n_features)])
    return np.asarray(fitted['booster'].feature_importance(importance_type='gain', iteration=n), float)


def library_versions() -> dict:
    import lightgbm
    import xgboost
    return {'xgboost': xgboost.__version__, 'lightgbm': lightgbm.__version__}
