# 종목·목표월 예측 연결과 월별 Spearman Rank IC 계산 조건을 검사한다.
# 계산 불가월의 사유와 공통 유효월의 동일가중 집계를 확인한다.

import unittest


@unittest.skip("Rank IC 계산 함수 구현 대기")
def test_rank_ic_contract() -> None:
    """예측·실제 표와 월별 평가표로 키·순위·집계 계약을 검증한다.

    TODO: 후보/Fold별 실제값 1:1 연결, 동순위 평균, 같은 순위1·반대-1을 확인한다.
    n<3·상수 순위는 결측/사유로 남기고 유효한 IC=0과 구분한다.
    후보별 공통 유효월과 월별 n·평가월수를 확인하고 종목수 가중을 배제한다.
    공통월 없음·중복키·비유한 쌍의 처리와 실패 후보 제외를 검사한다.
    """
    raise NotImplementedError
