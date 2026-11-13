# Ridge 회귀의 후보 구성, Fold별 학습·예측, 내부 검증 λ 선택.
# 벌점은 glmnet 방식 λ로 정의하고 sklearn alpha = λ × 학습 행 수로 바꾼다.
# 입력은 Fold 학습 행 기준으로 표준화한 뒤 벌점을 건다(glmnet standardize=TRUE와 같음). 계수는 원래 크기로도 보고한다.
# 산출물: Ridge 모형(계수·절편·λ), yhat 표, 내부 검증 표(메모리).

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence
from src.contracts import Config
from src.models.interface import Candidate, FoldSpec, apply_fold_transformer, fit_fold_transformer, format_prediction_frame, to_original_scale
if TYPE_CHECKING:
    from pandas import DataFrame

LAMBDAS = (0.001, 0.01, 0.1, 1, 10)


def enumerate_ridge_candidates(*, config: Config) -> Sequence[Candidate]:
    """config RIDGE -> λ별 후보. λ 목록·절편·스케일 규칙이 승인값과 다르면 오류."""
    r = config['RIDGE']
    if (tuple(float(x) for x in r.get('lambdas', ())) != LAMBDAS or r.get('fit_intercept') is not True
            or r.get('alpha_scaling') != 'lambda_times_train_rows' or r.get('tie_rule') != 'larger_lambda'
            or r.get('standardize') != 'train_rows_mean_sd_ddof0'):
        raise ValueError('ridge_config_not_approved')
    return [Candidate(candidate=f'ridge_lambda_{lam:g}', model='ridge',
                      parameters={'lambda': float(lam), 'fit_intercept': True}) for lam in LAMBDAS]


def fit_ridge_fold(train_X: DataFrame, train_y, train_keys: DataFrame, *, fold: FoldSpec, candidate: Candidate, transformer: Any) -> Any:
    """학습행만으로 Ridge 적합. alpha = λ × 학습 행 수(같은 λ면 행 수와 무관하게 같은 세기)."""
    import numpy as np
    from sklearn.linear_model import Ridge
    lam = candidate['parameters'].get('lambda')
    if candidate.get('model') != 'ridge' or lam is None or lam <= 0 or candidate['parameters'].get('fit_intercept') is not True:
        raise ValueError('ridge_candidate_invalid')
    if not train_keys.target_month.between(fold['train_start'], fold['train_end']).all():
        raise ValueError('train_rows_outside_fold')
    X = apply_fold_transformer(transformer, train_X)
    y = np.asarray(train_y, dtype=float)
    if not np.isfinite(y).all() or len(y) <= X.shape[1] + 1:
        raise ValueError('ridge_training_target_invalid')
    alpha = float(lam) * len(y)
    fitted = Ridge(alpha=alpha, fit_intercept=True).fit(X.to_numpy(float), y)
    coef = np.asarray(fitted.coef_, dtype=float)
    coef_original, intercept_original = to_original_scale(coef, fitted.intercept_, transformer)
    return {'model': 'ridge', 'candidate': candidate['candidate'], 'fold': fold['fold'], 'columns': list(X.columns),
            'lambda': float(lam), 'alpha': alpha, 'intercept': float(fitted.intercept_), 'coef': coef,
            'coef_original': coef_original, 'intercept_original': intercept_original,
            'n_train': len(y), 'transformer': transformer}


def predict_ridge_fold(model: Any, valid_X: DataFrame, valid_keys: DataFrame, *, fold: FoldSpec, candidate: Candidate, transformer: Any) -> DataFrame:
    """학습된 계수로만 예측. 열·Fold·후보가 학습과 다르면 오류."""
    if model.get('fold') != fold['fold'] or model.get('candidate') != candidate['candidate']:
        raise ValueError('model_fold_or_candidate_mismatch')
    if not valid_keys.target_month.between(fold['valid_start'], fold['valid_end']).all():
        raise ValueError('valid_rows_outside_fold')
    X = apply_fold_transformer(transformer, valid_X)
    if list(X.columns) != model['columns']:
        raise ValueError('prediction_columns_differ')
    yhat = model['intercept'] + X.to_numpy(float) @ model['coef']
    return format_prediction_frame(yhat, valid_keys, model='ridge', candidate=candidate['candidate'], fold=fold['fold'])


def choose_ridge_candidate(monthly_ic: DataFrame, candidates: Sequence[Candidate], failures: DataFrame) -> Candidate:
    """후보별 월별 IC -> 공통 유효월 동일가중 평균 IC 최대 후보. 정확히 같으면 큰 λ.

    실패 후보는 제외한다. 임의 허용오차로 동률을 만들지 않는다.
    오류: 공통월 없음·모든 후보 실패·중복 IC 키.
    """
    import numpy as np
    failed = set(failures.candidate) if len(failures) else set()
    alive = [c for c in candidates if c['candidate'] not in failed]
    if not alive:
        raise ValueError('all_ridge_candidates_failed')
    f = monthly_ic[monthly_ic.candidate.isin([c['candidate'] for c in alive])]
    if f.duplicated(['candidate', 'target_month']).any():
        raise ValueError('duplicate_ic_keys')
    wide = f.pivot(index='target_month', columns='candidate', values='ic')
    wide = wide.reindex(columns=[c['candidate'] for c in alive]).dropna()
    if wide.empty:
        raise ValueError('no_common_valid_months')
    means = wide.mean()
    best = means.max()
    tied = [c for c in alive if means[c['candidate']] == best and np.isfinite(best)]
    return max(tied, key=lambda c: c['parameters']['lambda'])


def select_ridge_lambda(train_X: DataFrame, train_y, train_keys: DataFrame, *, fold: FoldSpec, candidates: Sequence[Candidate],
                        transformer_spec: Any = None, eval_y=None, inner_years: int = 1):
    """Fold 학습 구간 안에서만 λ 선택: 마지막 `inner_years`년을 내부 검증으로 떼어 평균 Rank IC 최대.

    내부 학습은 fit 목표값(train_y), 내부 평가는 eval_y(기본 train_y, 제한 판에서는 원수익률)를 쓴다.
    입력 변환기(표준화)는 내부 학습 행으로만 다시 맞춘다. 바깥 검증 구간의 행·목표값은 받지 않는다.
    반환: (선택 후보, 내부 검증 월별 IC 표).
    """
    import numpy as np
    import pandas as pd
    from src.evaluation.expanding_window import inner_validation_split
    from src.evaluation.rank_ic import rank_ic_for_month
    if not train_keys.target_month.between(fold['train_start'], fold['train_end']).all():
        raise ValueError('train_rows_outside_fold')
    inner = inner_validation_split(fold, years=inner_years)
    eval_y = train_y if eval_y is None else eval_y
    fit_mask = train_keys.target_month.le(inner['train_end']).to_numpy()
    val_mask = ~fit_mask
    if not fit_mask.any() or not val_mask.any():
        raise ValueError('inner_split_empty')
    y_fit = np.asarray(train_y, dtype=float)[fit_mask]
    y_eval = np.asarray(eval_y, dtype=float)[val_mask]
    keys_fit, keys_val = train_keys[fit_mask], train_keys[val_mask]
    transformer = fit_fold_transformer(train_X[fit_mask], transformer_spec=transformer_spec)
    rows, failures = [], []
    for cand in candidates:
        try:
            model = fit_ridge_fold(train_X[fit_mask], y_fit, keys_fit, fold=inner, candidate=cand, transformer=transformer)
            pred = predict_ridge_fold(model, train_X[val_mask], keys_val, fold=inner, candidate=cand, transformer=transformer)
        except ValueError as error:
            failures.append({'candidate': cand['candidate'], 'reason': str(error)})
            continue
        pred['y'] = y_eval
        for _, part in pred.groupby('target_month', sort=True):
            row = rank_ic_for_month(part, score_column='yhat').iloc[0].to_dict()
            rows.append({**row, 'lambda': cand['parameters']['lambda'], 'inner_fold': inner['fold']})
    monthly = pd.DataFrame(rows)
    chosen = choose_ridge_candidate(monthly, candidates, pd.DataFrame(failures, columns=['candidate', 'reason']))
    return chosen, monthly
