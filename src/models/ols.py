# 절편을 포함한 OLS 기준모형의 설정·학습·예측 함수 골격을 정의한다.
# 공통 표본과 검증 구간을 사용하고 계수·입력 이상 여부를 검사한다.
# 예상 산출물: OLS 모형·진단과 종목·목표월별 yhat 반환표(predictions.csv 결합용).

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from src.models.interface import Candidate, FoldSpec
if TYPE_CHECKING:
    from pandas import DataFrame


def plan_ols_config(*, fit_intercept: bool = True) -> Candidate:
    """절편 설정 -> OLS 후보 ID/매개변수 설정을 반환한다.

    TODO: 후보 ID와 fit_intercept=True 설정을 구성한다.
    절편을 비활성화하면 오류로 처리한다.
    """
    raise NotImplementedError


def fit_ols_fold(train_X: DataFrame, train_y: DataFrame, train_keys: DataFrame, *, fold: FoldSpec, candidate: Candidate, transformer: Any) -> Any:
    """공통 학습 X/y/키와 Fold/후보/학습 변환기 -> OLS 모형과 진단을 반환한다.

    TODO: 키/열/목표월·공통 표본 검증, 학습행만 적합, 절편 포함.
    계수·절편·입력 이상·적합상태 기록. 검증 y/미래 수익률 사용 금지.
    오류: 비유한/결측/누수·빈 학습표·실패 -> 사유 기록 후 중단.
    """
    raise NotImplementedError


def predict_ols_fold(model: Any, valid_X: DataFrame, valid_keys: DataFrame, *, fold: FoldSpec, candidate: Candidate, transformer: Any) -> DataFrame:
    """적합 모형/검증 X·키/메타데이터 -> 공통 예측키+yhat 표를 반환한다.

    TODO: 학습된 변환기만 적용, 열·순서·표본·예측키 유일성 확인.
    오류: 미적합 모형, 비유한 예측, 다른 키/Fold -> 사유 기록.
    """
    raise NotImplementedError


def validate_ols_contract(candidate: Candidate, train_keys: DataFrame, valid_keys: DataFrame, reference_keys: DataFrame) -> None:
    """OLS 후보/학습·검증키/공통 기준키(role,fold 포함) -> 검사한다.

    TODO: 절편 포함, H1/H3 입력, Ridge/MLP와 동일 표본·월·Fold 확인.
    누수/다른 표본/중복키를 차단하고 공통 IC 평가 규격 확인.
    """
    raise NotImplementedError
