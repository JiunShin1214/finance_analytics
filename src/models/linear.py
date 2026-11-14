"""v2 선형 모델(MODEL_GRID): OLS·Ridge × MSE·Huber①·Huber②.

목적함수(학습 행 n개, 절편 c는 벌점 없음):
    MSE   : (1/n)·Σ (y − c − Xw)²           + λ‖w‖²     (Ridge는 sklearn Ridge(alpha = λ·n)와 같은 식, OLS는 λ=0)
    Huber : (1/n)·Σ ρ_δ(y − c − Xw)         + λ‖w‖²
            ρ_δ(r) = r²  (|r| ≤ δ),  2δ|r| − δ²  (|r| > δ)   — δ→∞이면 MSE와 같은 식이 되도록 r² 척도로 둔다.
δ 정의(MODEL_GRID):
    Huber① huber_q999     δ = |변환된 학습 목표값 − 그 달 학습 행 평균|의 99.9% 분위수(학습 행 전체) [GKX Table A.5 해석]
    Huber② huber_sigma135 δ = 1.35 × σ, σ는 같은 학습 행·입력·목표값에 scikit-learn HuberRegressor(epsilon=1.35,
                          alpha=0)를 맞춘 척도 추정값 [LWZ Table A.2, scikit-learn 기본값].
    HuberRegressor의 목적함수 Σ σ·H_ε(r/σ)는 σ를 고정하면 (1/σ)·Σ ρ_{εσ}(r)이라, OLS Huber②의 계수는 위 직접 구현에
    δ = 1.35σ를 넣은 것과 같다(check_huber_regressor_agreement). Ridge Huber②도 같은 δ에 같은 L2 벌점을 건다.
입력은 이미 매달 순위 [−1, 1]이라 추가 표준화를 하지 않는다(ridge_input_standardize: false).
난수 없음. 결과는 계수·절편과 진단값(dict)만 돌려주고 파일을 쓰지 않는다.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

LINEAR_MODELS = ('ols', 'ridge')
LOSSES = ('mse', 'huber_q999', 'huber_sigma135')
RIDGE_LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
HUBER_QUANTILE = 0.999
HUBER_EPSILON = 1.35


def check_linear_grid(grid: Mapping) -> None:
    """config MODEL_GRID의 선형 부분이 이 모듈의 고정값과 같은지 확인한다."""
    if (tuple(grid.get('linear_models', ())) != LINEAR_MODELS
            or tuple(grid.get('regression_losses', ())) != LOSSES
            or tuple(float(x) for x in grid.get('ridge_lambdas', ())) != RIDGE_LAMBDAS
            or float(grid.get('huber_q999_quantile', -1)) != HUBER_QUANTILE
            or float(grid.get('huber_sigma_epsilon', -1)) != HUBER_EPSILON
            or grid.get('ridge_input_standardize') is not False):
        raise ValueError('linear_model_grid_not_approved')


def _arrays(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or not len(y):
        raise ValueError('linear_input_shape')
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError('linear_input_nonfinite')
    if len(y) <= X.shape[1] + 1:
        raise ValueError('linear_too_few_rows')
    return X, y


def huber_delta_q999(y, months) -> float:
    """Huber① δ: |y − 그 달 평균(같은 학습 행)|의 99.9% 분위수(선형 보간)."""
    y = np.asarray(y, dtype=float)
    months = np.asarray(months)
    if len(y) != len(months) or not np.isfinite(y).all():
        raise ValueError('huber_delta_input')
    _, inverse = np.unique(months, return_inverse=True)
    means = np.bincount(inverse, weights=y) / np.bincount(inverse)
    delta = float(np.quantile(np.abs(y - means[inverse]), HUBER_QUANTILE))
    if not (math.isfinite(delta) and delta > 0):
        raise ValueError('huber_delta_nonpositive')
    return delta


def huber_sigma_sklearn(X, y) -> dict:
    """Huber② σ: scikit-learn HuberRegressor(epsilon=1.35, alpha=0, 절편 있음)의 scale_.

    max_iter는 수렴용으로 1000(라이브러리 기본 100은 15만 행에서 수렴 경고가 날 수 있음). 수렴 여부를 함께 돌려준다.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import HuberRegressor
    X, y = _arrays(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model = HuberRegressor(epsilon=HUBER_EPSILON, alpha=0.0, fit_intercept=True, max_iter=1000).fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    sigma = float(model.scale_)
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError('huber_sigma_nonpositive')
    return {'sigma': sigma, 'delta': HUBER_EPSILON * sigma, 'coef': np.asarray(model.coef_, float),
            'intercept': float(model.intercept_), 'converged': converged, 'n_iter': int(model.n_iter_)}


def _fit_mse(X, y, lam):
    n, p = X.shape
    if lam == 0:
        A = np.column_stack([np.ones(n), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return float(beta[0]), beta[1:], {}
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=lam * n, fit_intercept=True).fit(X, y)
    return float(model.intercept_), np.asarray(model.coef_, float), {'alpha': lam * n}


def _rho_mean(r, delta):
    a = np.abs(r)
    return float(np.where(a <= delta, r * r, 2 * delta * a - delta * delta).mean())


def _fit_huber(X, y, lam, delta, *, start, max_iter=100, tol=1e-12):
    """뉴턴법(활성 집합): 기울기 0이 되는 지점. ρ_δ는 볼록·한 번 미분 가능이라 단계 반감으로 감소를 보장한다.

    종료: (가) 기울기가 상대 tol 이하, 또는 (나) 온전한 뉴턴 걸음 뒤에도 이상치 집합이 같을 때. (나)는 그 집합 위에서
    목적함수가 정확히 이차식이라 걸음의 도착점이 정확한 최적점이기 때문이다. 수렴하지 않으면 오류(결과를 쓰지 않음).
    """
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    beta = np.concatenate([[start[0]], np.asarray(start[1], float)])
    penalty = np.full(p + 1, lam)
    penalty[0] = 0.0
    r = y - A @ beta
    obj = _rho_mean(r, delta) + lam * float(beta[1:] @ beta[1:])
    scale = max(1.0, float(np.mean(y * y)))
    for it in range(1, max_iter + 1):
        inside = np.abs(r) <= delta
        psi = np.where(inside, r, delta * np.sign(r))           # -dρ/dr / 2
        grad = -2 * (A.T @ psi) / n + 2 * penalty * beta
        if float(np.max(np.abs(grad))) <= tol * scale:
            return float(beta[0]), beta[1:], {'iterations': it - 1, 'outlier_share': float(1 - inside.mean())}
        hess = 2 * (A.T @ (A * inside[:, None])) / n + np.diag(2 * penalty)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError as exc:
            raise ValueError('huber_singular_hessian') from exc
        t = 1.0
        while True:
            cand = beta - t * step
            r_new = y - A @ cand
            new = _rho_mean(r_new, delta) + lam * float(cand[1:] @ cand[1:])
            if new <= obj + 1e-15 * abs(obj) or t < 1e-10:
                break
            t *= 0.5
        same_set = np.array_equal(np.abs(r_new) <= delta, inside)
        beta, obj, r = cand, new, r_new
        if same_set and t == 1.0:
            return float(beta[0]), beta[1:], {'iterations': it, 'outlier_share': float(1 - inside.mean())}
    raise ValueError('huber_not_converged')


def fit_linear(X, y, *, model: str, loss: str, ridge_lambda: float | None = None, delta: float | None = None,
               start: tuple | None = None) -> dict:
    """학습 행(X, y)만으로 선형 모델 하나를 맞춘다. 반환: intercept, coef, model, loss, lambda, delta, n_train, 진단값.

    Huber 손실은 delta를 호출부가 준다(①은 huber_delta_q999, ②는 huber_sigma_sklearn(...)['delta']).
    start=(절편, 계수)를 주면 Huber 반복을 그 점에서 시작한다(속도용, 볼록 문제라 같은 최적점). 없으면 같은 λ의 MSE 해에서 시작.
    """
    X, y = _arrays(X, y)
    if model not in LINEAR_MODELS or loss not in LOSSES:
        raise ValueError('linear_model_or_loss_unknown')
    if model == 'ols':
        if ridge_lambda not in (None, 0):
            raise ValueError('ols_has_no_lambda')
        lam = 0.0
    else:
        if ridge_lambda is None or float(ridge_lambda) not in RIDGE_LAMBDAS:
            raise ValueError('ridge_lambda_not_approved')
        lam = float(ridge_lambda)
    if loss != 'mse':
        if delta is None or not (math.isfinite(delta) and delta > 0):
            raise ValueError('huber_delta_required')
        if start is None:
            c, w, _ = _fit_mse(X, y, lam)
            start = (c, w)
        c, w, info = _fit_huber(X, y, lam, float(delta), start=start)
    else:
        if delta is not None:
            raise ValueError('mse_has_no_delta')
        c, w, info = _fit_mse(X, y, lam)
    if not (math.isfinite(c) and np.isfinite(w).all()):
        raise ValueError('linear_fit_nonfinite')
    return {'model': model, 'loss': loss, 'lambda': lam if model == 'ridge' else None, 'delta': delta,
            'intercept': c, 'coef': np.asarray(w, float), 'n_train': len(y), **info}


def predict_linear(fitted: Mapping, X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(fitted['coef']) or not np.isfinite(X).all():
        raise ValueError('linear_predict_input')
    return fitted['intercept'] + X @ fitted['coef']


def check_huber_regressor_agreement(X, y, *, atol: float = 1e-6) -> float:
    """OLS Huber② 직접 구현 계수가 HuberRegressor 계수와 같은지(최대 절대 차이). 다르면 오류."""
    ref = huber_sigma_sklearn(X, y)
    mine = fit_linear(X, y, model='ols', loss='huber_sigma135', delta=ref['delta'])
    err = float(max(abs(mine['intercept'] - ref['intercept']), np.max(np.abs(mine['coef'] - ref['coef']))))
    if err > atol:
        raise ValueError(f'huber_regressor_disagrees:{err}')
    return err
