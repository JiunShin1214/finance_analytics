# 절편을 포함한 OLS 기준모형의 설정·학습·예측.
# 공통 표본과 검증 구간을 사용하고 계수·입력 이상 여부를 검사한다.
# 산출물: OLS 모형(계수·절편)과 종목·목표월별 yhat 표(메모리).

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from src.models.interface import Candidate, FoldSpec, apply_fold_transformer, format_prediction_frame, _check_X
if TYPE_CHECKING:
    from pandas import DataFrame


def plan_ols_config(*, fit_intercept: bool = True) -> Candidate:
    """절편 포함 OLS 후보. 절편을 끄면 오류."""
    if fit_intercept is not True:
        raise ValueError('ols_requires_intercept')
    return Candidate(candidate='ols', model='ols', parameters={'fit_intercept': True})


def fit_ols_fold(train_X: DataFrame, train_y, train_keys: DataFrame, *, fold: FoldSpec, candidate: Candidate, transformer: Any) -> Any:
    """학습행만으로 절편 포함 최소제곱 적합. 반환: 계수·절편·열·진단 사전.

    검증 y나 학습 구간 밖 목표월은 받지 않는다(호출부 validate_fold_inputs).
    """
    import numpy as np
    if candidate.get('model') != 'ols' or candidate['parameters'].get('fit_intercept') is not True:
        raise ValueError('ols_candidate_invalid')
    if not train_keys.target_month.between(fold['train_start'], fold['train_end']).all():
        raise ValueError('train_rows_outside_fold')
    X = apply_fold_transformer(transformer, train_X)
    y = np.asarray(train_y, dtype=float)
    if not np.isfinite(y).all() or len(y) <= X.shape[1] + 1:
        raise ValueError('ols_training_target_invalid')
    design = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError('ols_rank_deficient')
    return {'model': 'ols', 'candidate': candidate['candidate'], 'fold': fold['fold'], 'columns': list(X.columns),
            'intercept': float(beta[0]), 'coef': beta[1:].astype(float), 'n_train': len(y), 'transformer': transformer}


def predict_ols_fold(model: Any, valid_X: DataFrame, valid_keys: DataFrame, *, fold: FoldSpec, candidate: Candidate, transformer: Any) -> DataFrame:
    """학습된 계수로만 예측. 열·Fold·후보가 학습과 다르면 오류."""
    if model.get('fold') != fold['fold'] or model.get('candidate') != candidate['candidate']:
        raise ValueError('model_fold_or_candidate_mismatch')
    if not valid_keys.target_month.between(fold['valid_start'], fold['valid_end']).all():
        raise ValueError('valid_rows_outside_fold')
    X = apply_fold_transformer(transformer, valid_X)
    if list(X.columns) != model['columns']:
        raise ValueError('prediction_columns_differ')
    yhat = model['intercept'] + X.to_numpy(float) @ model['coef']
    return format_prediction_frame(yhat, valid_keys, model='ols', candidate=candidate['candidate'], fold=fold['fold'])


def validate_ols_contract(candidate: Candidate, train_keys: DataFrame, valid_keys: DataFrame, reference_keys: DataFrame) -> None:
    """절편 포함 후보이고, 학습·검증 키가 기준 키(같은 Fold의 다른 모형)와 같은지 검사한다."""
    if candidate['parameters'].get('fit_intercept') is not True:
        raise ValueError('ols_requires_intercept')
    for role, keys in (('train', train_keys), ('valid', valid_keys)):
        ref = reference_keys[reference_keys.role.eq(role)][['stock_id', 'target_month']].reset_index(drop=True)
        if not keys[['stock_id', 'target_month']].reset_index(drop=True).equals(ref):
            raise ValueError(f'{role}_keys_differ_from_reference')
