# 목표월을 기준으로 학습·검증 구간을 나누는 연별 확장창을 정의한다.
# Fold 시간순서, 외부자료의 학습 이력 및 Test 접근 범위를 검사한다.
# 예상 산출물: 기간 구분이 추가된 표본표와 연도별 학습·검증 Fold 목록.

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from src.contracts import Config
from src.models.interface import FoldSpec
if TYPE_CHECKING:
    from pandas import DataFrame


def assign_target_month_split(samples: DataFrame, *, config: Config) -> DataFrame:
    """표의 target_month와 기간 설정을 받아 warmup/train/test 구분을 추가한다.

    TODO: 2000년 준비, 2001~2021년 학습, 2022~2025년 최종평가로 구분한다.
    준비기간은 시차와 EWMA 초기화에만 사용하고 학습·평가에서 제외한다.
    범위 밖 월과 잘못된 월 형식은 오류로 처리한다.
    """
    raise NotImplementedError


def plan_annual_expanding_folds(*, config: Config, available_target_months: Sequence[str] | None = None) -> Sequence[FoldSpec]:
    """기간설정/선택적 외부자료 가용 목표월 -> 연별 Fold 목록을 반환한다.

    TODO: 2001-01~2005-12 -> 2006-01~12 부터
    2001-01~2020-12 -> 2021-01~12 까지 16개 Fold.
    외부자료 하위기간은 최소 5년 학습 후 다음 해 검증, 이력 부족이면 보류.
    오류: 비연속/불충분 가용범위의 처리 근거 누락, Test 포함, 역순 경계.
    """
    raise NotImplementedError


def validate_fold_chronology(train_keys: DataFrame, valid_keys: DataFrame, *, fold: FoldSpec) -> None:
    """학습/검증 목표월키와 Fold -> 시간순서 검사한다.

    TODO: max(train target)<min(valid target), Fold 경계·키 무교집합 확인.
    warmup fit 제외, 동일 종목·월 공통표본 유지, 2021 검증 이후 확장 금지.
    오류: 기준월 분할/검증월 fit/겹친 키/Test 혼입 -> 후속 적합 차단.
    """
    raise NotImplementedError


def verify_no_test_access(*, stage: str, declared_target_years: Sequence[int], source_scope_verified: bool) -> None:
    """단계명, 선언된 목표연도, 파일 범위 확인 여부로 자료 접근을 검사한다.

    TODO: 파일을 열기 전에 final-test, 2022~2025년 및 범위 미확인 자료를 차단한다.
    읽은 뒤 기간을 필터링하는 방식으로 접근 검사를 대체하지 않는다.
    현재는 미구현 단계이므로 모든 호출을 차단한다.
    추후 구현 시 범위가 검증된 Train 자료만 허용하고,
    Test 자료와 범위 미확인 자료는 접근 전에 차단한다.
    """
    raise PermissionError("Test 데이터 접근이 차단되었습니다.")
