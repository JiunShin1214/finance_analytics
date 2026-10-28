# OLS·Ridge·MLP의 후보 설정과 공통 표본·전처리 계약을 검사한다.
# 후보 동률 처리와 확장 비교에서 기준모형의 표본 일치를 확인한다.

import unittest


@unittest.skip("모형 학습·예측 함수 구현 대기")
def test_models_contract() -> None:
    """후보·학습키·검증키·진단표로 동일 조건의 모형 비교를 검증한다.

    TODO: OLS 절편, Ridge 5개 alpha·큰 alpha 동률, MLP 4개 고정 후보를 확인한다.
    MLP는 적은 총 노드 -> 큰 alpha 동률 규칙과 수렴 실패 제외를 확인한다.
    학습행만 변환 fit, 예측키 유일성, 공통 유효월 동일가중 평균을 검사한다.
    활성 H2/H4는 주효과·상호작용을 보존하고 기준모형도 같은 표본으로 재학습한다.
    """
    raise NotImplementedError
