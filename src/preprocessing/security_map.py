# 종목코드와 DART 기업번호를 당시 유효한 대응표로 연결하는 모듈이다.
# 코드 변경·재사용 및 유효기간 중복을 확인하고 모호한 연결을 구분한다.
# 예상 산출물: 기업번호·연결근거가 포함된 종목-월 대응표와 모호한 연결 목록.

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pandas import DataFrame


def plan_security_mapping(panel: DataFrame, security_map: DataFrame) -> DataFrame:
    """종목-월 표와 SECURITY_MAP_COLUMNS 표 -> 유효 기업번호/연결근거 표를 반환한다.

    TODO: 당시 유효기간으로 연결, 변경/재사용/합병/분할/상폐 이력 대조.
    기업 대응을 과거로 소급하지 않음.
    개별 미연결·근거 불명 행은 사유를 기록하고 격리한다.
    오류: 동일 종목·시점의 중복 대응, 유효기간 역전·충돌 등
    대응표 무결성 위반은 후속 결합 차단.
    """
    raise NotImplementedError


def verify_mapping_period(security_map: DataFrame) -> None:
    """대응표 -> 기간·문자열 키·연결근거 검사한다.

    TODO: 유효기간 중첩/시작>종료 차단. 종료일 미상 및 경계 포함 정책은
    실제 원천 명세를 확인 후 확정한다. 임의 무기한 연장 금지.
    """
    raise NotImplementedError


def flag_ambiguous_mapping(mapped_panel: DataFrame) -> DataFrame:
    """종목-기업 연결 후보표 -> 모호한 행/사유/영향 범위 표를 반환한다.

    TODO: 1:N 대응, 코드 재사용, 근거 미확인 행 표시. 첫 후보 자동 채택 금지.
    오류: 키 부재·근거 열 부재는 단계 중단. 유효 표와 격리 표 구분.
    """
    raise NotImplementedError
