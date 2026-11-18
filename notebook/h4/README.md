# H4 작업 안내

현재 실행·입력·설치 안내는 [코드 README](../../src/h4/README.md)를 기준으로 한다.
2026-11-18에 H1+H3 기준 대비 H4 추가 Ridge의 개발기간 OOS 비교를 완료했다.
이전 문서의 “공통 입력 없음 / 모델 결합 미실행”은 2026-11-16 당시 기록이다.

- 모델 입력: `data/processed/h4_epu.parquet` (별도 전달, Git 제외)
- 입력 생성: `python -m src.h4.preprocessing.export`
- 실제 Ridge: `python -m src.h4.evaluation.ridge`
- 직접 특징 IC: `python -m src.h4.evaluation.feature_rank`
- 대표 Ridge 결과: `outputs/model_results/h4/20261118T134947_h4_paired_oos_ridge/result.zip`

기존 노트북과 `docs/`는 직접 특징 분석·과거 탐색 분석의 기록이다.
그 안의 과거 입력 경로보다 최신 코드 README의 경로를 우선한다.
최종 테스트(2022~2025)는 사용하지 않았다. 개별 예측·데이터·백업은 공유 커밋에서 제외한다.
