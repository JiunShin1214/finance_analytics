# 목표월 기준 기간 구분과 연별 확장창의 시간순서를 검사한다.
# 파일 읽기 전에 Test·범위 미확인 자료를 차단하는지 확인한다.

import unittest


@unittest.skip("기간 분할·입출력 검사 함수 구현 대기")
def test_period_guard() -> None:
    """기간 메타데이터와 모의 입출력으로 시간 분리 및 접근 차단을 검증한다.

    TODO: 2000 준비기간 fit/IC 제외, 2000-12 기준의 2001-01 목표 Train 구분.
    2001~2005 -> 2006부터 2001~2020 -> 2021까지 16개 Fold를 검사한다.
    외부자료 최소 5년 학습, 부족 시 보류, Test와 범위 불명 파일의 읽기 전 차단.
    실제 Test 파일 대신 모의 입출력을 사용하며 final-test 접근 차단을 확인한다.
    """
    raise NotImplementedError
