# H2의 정책기조 점수, 공개시점 정렬, 기업규모 표준화와 결합 변수를 검사한다.
# 무발표·수집실패 구분과 Train/Test 정보 차단을 확인한다.

import unittest


@unittest.skip("H2 계산 함수 구현 대기")
def test_h2_contract() -> None:
    """정책문서·월간패널·H2 특징표로 계산식과 시점 계약을 검증한다.

    TODO: 정책점수 범위·분모0/유효0, 실제 공개일 우선·proxy 표시,
    동일월 복수문서 최신 선택, 무발표 forward-fill과 수집/추출 실패 결측을 확인한다.
    직전 월말 양수 시가총액 -> ln -> 월별 Z(ddof=1) -> 부호반전으로
    sensitivity를 만들고 h2_interaction=policy_score*sensitivity를 검사한다.
    월별 표준화 집합과 정책사전·문맥규칙이 target y 또는 Test 정보에 의존하지 않는지 확인한다.
    """
    raise NotImplementedError
