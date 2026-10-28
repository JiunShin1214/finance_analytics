# H1의 모멘텀 기간, 재무값 유효성 및 월별 표준화 조건을 검사한다.
# 입력 시점과 공통 종목집합이 목표수익률 관측 여부에 의존하는지 확인한다.

import unittest


@unittest.skip("H1 계산 함수 구현 대기")
def test_h1_contract() -> None:
    """H1 구성값과 계산 이력으로 수식·기간·표준화 계약을 검증한다.

    TODO: t-11~t-1 정확히 11달, 한 달 누락·중복, t월 제외를 검사한다.
    양수 MC/BV, PIT·정정, 공통 유효집합, ddof1, n<2·0분산 미산출을 확인한다.
    -Size+BM+Mom 부호와 /3, 2001-01 목표의 달력 경계 및 y 독립성을 확인한다.
    """
    raise NotImplementedError
