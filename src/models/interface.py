# 모형들이 같은 입력·표본·검증 구간을 사용하도록 공통 인터페이스를 정의한다.
# 후보·Fold 형식, 입력·키 검사, 학습행 변환기 및 예측표 구성 함수를 제공한다.
# 산출물: 후보·Fold 명세와 공통 형식 예측표(메모리).

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypedDict
if TYPE_CHECKING:
    from pandas import DataFrame

KEY = ['stock_id', 'target_month']


class FoldSpec(TypedDict):
    """목표월 YYYY-MM 기준 학습·검증 구간의 양끝 경계."""

    fold: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str


class Candidate(TypedDict):
    """후보 식별자, 모형명 및 하이퍼파라미터."""

    candidate: str
    model: str
    parameters: Mapping[str, Any]


def _check_X(X: DataFrame, name: str) -> None:
    import numpy as np
    if not len(X.columns) or X.columns.duplicated().any():
        raise ValueError(f'{name}_columns_invalid')
    values = X.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError(f'{name}_nonfinite_or_missing')


def validate_fold_inputs(train_X: DataFrame, train_y, valid_X: DataFrame, train_keys: DataFrame, valid_keys: DataFrame, *, fold: FoldSpec) -> None:
    """특징/목표/행키/목표월 Fold 계약을 검사한다.

    X·y·키 행 수와 순서 일치, 학습·검증 특징열 동일, 결측·비유한값 금지,
    Fold 경계·시간순서(validate_fold_chronology). 검증 y는 받지 않는다.
    """
    import numpy as np
    from src.evaluation.expanding_window import validate_fold_chronology
    if not (len(train_X) == len(train_y) == len(train_keys)) or len(valid_X) != len(valid_keys):
        raise ValueError('row_count_mismatch')
    if list(train_X.columns) != list(valid_X.columns):
        raise ValueError('feature_columns_differ')
    for frame, keys in ((train_X, train_keys), (valid_X, valid_keys)):
        if not frame.index.equals(keys.index):
            raise ValueError('row_order_mismatch')
    _check_X(train_X, 'train_X'); _check_X(valid_X, 'valid_X')
    if not np.isfinite(np.asarray(train_y, dtype=float)).all():
        raise ValueError('train_y_nonfinite_or_missing')
    validate_fold_chronology(train_keys, valid_keys, fold=fold)


def fit_fold_transformer(train_X: DataFrame, *, transformer_spec: Mapping[str, Any] | None) -> Any:
    """학습 X로 입력 변환기를 만든다. 항등(identity) 또는 학습 행 표준화(standardize).

    standardize: 그 Fold 학습 행만으로 열별 평균·표준편차(ddof=0, glmnet 기본 표준화와 같은 1/n)를 구한다.
    벌점 모형(Ridge)이 입력 크기에 따라 계수를 다르게 깎지 않게 하기 위함이다(ESL 3.4.1, glmnet standardize=TRUE).
    검증·Test 행이나 전체 기간 통계는 쓰지 않는다. 다른 명세는 오류.
    """
    _check_X(train_X, 'train_X')
    columns = list(train_X.columns)
    if transformer_spec in (None, {}, {'kind': 'identity'}):
        return {'kind': 'identity', 'columns': columns}
    if transformer_spec == {'kind': 'standardize'}:
        mean = train_X.mean().to_numpy(float)
        sd = train_X.std(ddof=0).to_numpy(float)
        if not (sd > 0).all():
            raise ValueError('zero_variance_training_feature')
        return {'kind': 'standardize', 'columns': columns, 'mean': mean, 'sd': sd}
    raise ValueError('unsupported_transformer_spec')


def apply_fold_transformer(transformer: Any, X: DataFrame) -> DataFrame:
    """학습에서 만든 변환기를 X에 적용한다(재적합 없음). 열 구조·행 순서·인덱스를 보존한다."""
    if not isinstance(transformer, Mapping) or transformer.get('kind') not in ('identity', 'standardize'):
        raise ValueError('transformer_not_fitted')
    if list(X.columns) != transformer['columns']:
        raise ValueError('transformer_columns_differ')
    _check_X(X, 'X')
    if transformer['kind'] == 'identity':
        return X
    return (X - transformer['mean']) / transformer['sd']


def to_original_scale(coef, intercept, transformer):
    """표준화된 입력에서 구한 계수·절편을 원래 입력 크기로 되돌린다(glmnet처럼 원래 크기로 보고)."""
    import numpy as np
    coef = np.asarray(coef, dtype=float)
    if transformer['kind'] == 'identity':
        return coef, float(intercept)
    original = coef / transformer['sd']
    return original, float(intercept - (original * transformer['mean']).sum())


def format_prediction_frame(yhat, valid_keys: DataFrame, *, model: str, candidate: str, fold: str) -> DataFrame:
    """행순서가 일치하는 yhat/검증키 -> (model,candidate,fold,stock_id,target_month,yhat) 표.

    실제 y는 평가 단계에서 키로 1:1 결합한다. 예측값 비유한·키 중복·행 수 불일치는 오류.
    """
    import numpy as np
    import pandas as pd
    values = np.asarray(yhat, dtype=float).reshape(-1)
    if len(values) != len(valid_keys):
        raise ValueError('prediction_row_count_mismatch')
    if not np.isfinite(values).all():
        raise ValueError('nonfinite_prediction')
    out = pd.DataFrame({'model': model, 'candidate': candidate, 'fold': fold,
                        'stock_id': valid_keys.stock_id.to_numpy(), 'target_month': valid_keys.target_month.to_numpy(),
                        'yhat': values})
    if out.duplicated(['model', 'candidate', 'fold', *KEY]).any():
        raise ValueError('duplicate_prediction_keys')
    return out
