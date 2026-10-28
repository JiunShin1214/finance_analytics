# 종목코드와 달력월을 정규화하고 중복·결측을 구분하는 모듈이다.
# 기준월 특징을 다음 달 목표수익률과 연결하는 함수 골격을 정의한다.
# 예상 산출물: 정규화 패널과 목표월·목표수익률 결합표(panel.parquet·samples.parquet 구성용).

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    from pandas import DataFrame


def normalize_stock_id(values: Sequence[str]) -> Sequence[str]:
    """종목코드 문자열 열 -> 앞자리 0 보존 6자리 문자열 열을 반환한다.

    TODO: 형식/길이 검사. 숫자형으로 이미 소실된 식별자를 임의 복구하지 않음.
    오류: DART 8자리 번호 혼용, 비숫자 문자, 코드 재사용의 무근거 치환.
    """
    raise NotImplementedError


def normalize_month(values: Sequence[str]) -> Sequence[str]:
    """월 문자열 열 -> 달력상 유효 YYYY-MM 열을 반환한다.

    TODO: 명시된 날짜 형식만 변환, 시간대는 Asia/Seoul. 불명/불가능 월 거부.
    월별 시계열 공백을 행 순서로 압축하지 않는다.
    """
    raise NotImplementedError


def distinguish_zero_vs_missing(table: DataFrame, *, status_columns: Sequence[str]) -> DataFrame:
    """입력 표/상태열명 -> 원값 및 결측 사유가 보존된 복사 표를 반환한다.

    TODO: 실제 0, 자료 부재, 수집/추출/연결 실패, 미상장 기간 분리.
    오류: 상태 누락·모순은 기록/격리. 결측을 0/전체기간 평균으로 채우지 않음.
    """
    raise NotImplementedError


def enforce_unique_stock_month(table: DataFrame, *, month_column: str) -> None:
    """표/관측월 또는 기준월 열명 -> 유일성 확인한다.

    TODO: stock_id와 month_column의 존재·널 여부를 확인하고,
    (stock_id, month_column) 조합의 유일성을 검사한다.
    월 열은 관측월·기준월·목표월 중 호출자가 지정한 열을 사용한다.
    중복 행을 임의 삭제하지 않고 중복 키·건수·영향 범위를 기록한다.
    필수 열 누락, 널 키 또는 중복 키가 있으면 후속 처리를 중단한다.
    """
    raise NotImplementedError


def attach_target_month(features: DataFrame) -> DataFrame:
    """stock_id/base_month 특징표 -> target_month=t+1 달력월 추가 표를 반환한다.

    TODO: 2000-12 -> 2001-01 및 연도 경계 검사. 행 shift 금지.
    미래 y를 읽거나 특징 유효 종목을 목표 관측 여부로 걸러내지 않는다.
    오류: 비정상 월·중복 기준월 키.
    """
    raise NotImplementedError


def join_training_targets(features: DataFrame, returns: DataFrame) -> DataFrame:
    """특징표와 허용된 Train 수익률표 -> 목표 y/별도 제외사유 표를 반환한다.

    TODO: (stock_id,target_month)=(stock_id,관측월) 1:1 연결.
    특징 산출과 월별 Z-score 계산이 끝난 후 특징표 기준으로 y를 결합한다.
    y가 없는 특징행도 보존하고 목표 결측 사유와 특징 결측 사유를 분리한다.
    제외 여부는 이후 공통 유효 표본 구성 단계에서 결정한다.
    오류: Test/미확인 범위, 중복 목표 키, t와 t+1 불일치 -> 중단.
    """
    raise NotImplementedError
