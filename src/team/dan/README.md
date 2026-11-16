# Dan H4 코드

현재 진입점: `python -m src.team.dan.h4_feature_rank` (저장소 루트에서).

| 파일 | 역할 | 상태 |
|---|---|---|
| h4_feature_rank.py | H4 특징 원값·윈저라이징 월별 직접 Rank IC | 현재 결과, 실행·독립 대조 완료 |
| h4_epu.py | 팀 MOM 재사용, EPU 월 정렬·상호작용 생성 | 공통 개인 유틸리티 |
| upstream_rank_snapshot.py | 팀 main의 선택 함수 원문 사본 | 출처 blob 해시 보존, 공용 파일 수정 없음 |
| h4_analysis.py | 월 FE 패널 OLS·월 군집 SE·2단계 HAC | 별도 탐색 가설검정 |
| test_h4.py | 월 정렬·결측·키 중복·Holm·음의 상호작용 검증 | 실행 완료 |
| h4_standalone_rank.py | 2006~2021 확장창 OLS 예측 IC | 이전 실험, 팀 모델과 직접 비교 불가 |
| h4_rank_comparison.py | 최신 model_test 연결 | 미실행 초안, H1 전용 비교 구현 아님 |

실험 조건·추가 준비 항목은 `notebook/dan/README.md`와 `docs/`를 확인한다.
기존 코드 경로를 유지해 이전 ZIP의 코드 해시 및 실행 명령을 보존한다.

## EPU 정의·모델 결합 검토 업데이트

사용자 제공 새 H4 정의: Cho–Kim(2023) 뉴스 기반 한국 EPU(t-1) × 원 11개월 MOM. 기사 직접 수집은 하지 않는다. 로컬 엑셀의 출처 메모도 확인했다.

`h4_export.py`는 모델용 세 변수를 보존한 `data/processed/dan_h4_epu.parquet`를 생성한다(Git 제외). `h4_rank_comparison.py --prepare-only`는 최신 공통 API에 상대경로·transform=none으로 입력 준비만 한다. 모형 적합·공식 결과 저장은 미완료다.

상세 검토: `notebook/dan/docs/MODEL_REVIEW.md`. 개인 분석 코드는 선택 커밋 대상이지만 바로 모델 결합 검증 완료로 간주하지 않는다.
