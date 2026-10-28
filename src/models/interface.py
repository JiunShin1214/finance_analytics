# 세 모형이 같은 입력·표본·검증 구간을 사용하도록 공통 인터페이스를 정의한다.
# 후보·Fold 형식, 학습행 전처리 및 예측표 구성 함수를 제공한다.
# 예상 산출물: 후보·Fold 명세, 입력 변환기, 변환 특징표와 공통 형식 예측표.

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, TypedDict
if TYPE_CHECKING:
    from pandas import DataFrame


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


def validate_fold_inputs(train_X: DataFrame, train_y: DataFrame, valid_X: DataFrame, train_keys: DataFrame, valid_keys: DataFrame, *, fold: FoldSpec) -> None:
    """특징/목표/행키/목표월 Fold 계약 -> 검증한다.

    TODO: X/y 키·순서 일치, 공통 complete-case, 특징열 동일, 비유한값 검사.
    학습<검증 목표월, warmup 학습 제외, Test 접근 금지, y 검증값 fit 제외.
    오류: 중복키/결측/다른 sample_id/빈 학습표/누수 -> 차단.
    """
    raise NotImplementedError


def fit_fold_transformer(train_X: DataFrame, *, transformer_spec: Mapping[str, Any]) -> Any:
    """학습 X와 변환 명세를 받아 학습행에 적합한 입력 변환기를 반환한다.

    TODO: 명세에 추가 변환이 있는 경우 해당 Fold 학습행만으로 적합한다.
    특징의 월별 Z-score와 구분하고 검증·Test 및 전체기간 통계를 사용하지 않는다.
    미확정 명세, 결측 보충 요청 또는 학습 범위 밖 적합은 오류로 처리한다.
    """
    raise NotImplementedError


def apply_fold_transformer(transformer: Any, X: DataFrame) -> DataFrame:
    """학습 Fold에서 적합한 변환기를 X에 적용한 표를 반환한다.

    TODO: transform만 허용하며 검증 X에 재적합하지 않는다.
    변환 전후 행 수·행 순서·인덱스를 유지하고,
    별도로 관리하는 종목·목표월 키와의 대응을 보존한다.
    특징열·유한값 검사 및 출력 열 구조를 확인한다.
    오류: 미적합 변환기·학습과 다른 입력열·행 손실·순서 불일치 -> 중단.
    """
    raise NotImplementedError


def format_prediction_frame(yhat: DataFrame, valid_keys: DataFrame, *, model: str, candidate: str, fold: str) -> DataFrame:
    """행순서가 일치하는 yhat/검증키/모형·후보·Fold ID -> 예측키+yhat 표를 반환한다.

    TODO: (model,candidate,fold,stock_id,target_month) 유일, 표본 수 일치.
    실제 y는 평가 시 별도 1:1 결합; fit/predict에서 사용 금지.
    오류: 누락/중복/순서 불일치/비유한 예측 -> 기록 후 평가 차단.
    """
    raise NotImplementedError
