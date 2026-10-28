# H3의 EWMA 초기화·상태 갱신과 최근 관측 가용성 조건을 검사한다.
# 결측 상태 보존, 달력월 정렬 및 월별 변동성 표준화를 확인한다.

import unittest


@unittest.skip("H3 계산 함수 구현 대기")
def test_h3_contract() -> None:
    """수익률과 상태 이력으로 초기화·재귀 갱신·특징 공개 조건을 검증한다.

    TODO: 첫 유효 10개 ddof1 초기화, tau 이중갱신 없음, 결측월 상태 보존.
    최근 t-11~t의 10/12, 조건 미달 시 상태 유지, sqrt 후 월별 Z를 확인한다.
    12개월·연도·Fold 재초기화 방지 및 .04/.10 -> .0391 산식 사례를 검사한다.
    """
    raise NotImplementedError
