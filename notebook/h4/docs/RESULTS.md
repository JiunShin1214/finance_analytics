# 결과 구분

## Ridge OOS 비교

`outputs/model_results/h4/20261118T134947_h4_paired_oos_ridge/result.zip`

2006~2021년 192개월. 원수익률 목표·전체 표본의 평균 IC는
기준모형 0.082221, EPU 추가모형 0.081415로 차이는 -0.000806이다.
ICIR은 0.770300과 0.766166이다. 월별 비교와 목표값별 결과는 ZIP의 집계 CSV에 있다.

## 직접 특징 Rank IC

`outputs/model_results/h4/20261118T135005_h4_feature_rank_raw-winsor/result.zip`

유효 251개월, 167,089 종목-월. 원값 평균 IC 0.025527, 특징 윈저 처리 IC 0.025531.
실제 수익률은 원값을 유지한다. 2001-01은 유효 MOM 표본이 없어 미산출이다.
양수 월별 EPU를 곱한 순위는 MOM 순위와 같을 수 있으므로 EPU의 추가 예측력을 입증하지 않는다.

## 탐색 분석

`src.h4.analysis.inference`는 월 고정효과 OLS·HAC 분석,
`src.h4.analysis.standalone_rank`는 이전 H4 단독 OLS 예측 실험이다.
가격 필터·표본·학습 방식이 달라 Ridge 비교와 숫자를 직접 비교하지 않는다.
분석 조건은 [추론 분석](INFERENCE_DETAILS.md)을 참조한다.
