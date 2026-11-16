# 커밋·모델 결합 검토 (2026-11-16)

## 결론

개인 H4 특징 생성·직접 Rank IC·탐색 가설검정 코드로 한정하면 검수 후 선택 커밋할 수 있다. 공식 H1+H4 모델 결합·최종 성능 검증 완료 상태로 제출하면 안 된다. 실제 결합 모델은 로컬 공통 모듈·데이터 부족으로 실행하지 못했다. 이번에 커밋하거나 push하지 않았다.

## 반영한 H4 정의

사용자가 제시한 새 명세를 개인 작업의 기준으로 적용한다. 한국 EPU는 Cho–Kim(2023), 기사를 직접 수집하지 않고 기존 뉴스 기반 지수를 사용한다. 기준월 t에는 t-1 EPU를 연결하며, 팀의 원 11개월 MOM과 상호작용을 함께 보존한다.
공용 README/features는 이번에 변경하지 않았다. 팀 main 반영 여부와 개인 분석 명세 적용을 혼동하지 않는다.

## 출처 확인

로컬 엑셀의 Korean New EPU Index 시트 메모에 Cho and Kim (2023), Macroeconomic Effects of Uncertainty Shocks: Evidence from Korea, Journal of Asian Economics 84, 101571과 13개 신문 기반이라는 설명이 있다. 따라서 로컬 파일의 지수 계열은 요청한 출처와 일치하는 것으로 확인했다. EPU 자료는 사용자가 검증 완료했다고 확인했다(2026-11-16). 별도 독립 공표일 검증을 이 코드가 수행했다는 뜻은 아니다.

출처: https://policyuncertainty.com/korea_chokim.html
저자 페이지: https://dooyeoncho.github.io/pages/new_korean_epu_index.html
출처·전월 연결 규칙을 기재한다. EPU 검증 완료는 사용자 확인에 근거한다.

## 모델 입력 파일: 생성·검증 완료

`data/processed/dan_h4_epu.parquet` (Git 제외)

- stock_id, base_month, 원 MOM, epu_month, epu_raw, h4_interaction, 가용성·제외사유.
- 적격 종목·2000-12~2021-11 기준월. y를 읽거나 결합하지 않고 생성.
- mom_available=False인 MOM 값은 사용하지 않는다. 구성값 결측을 0으로 채우지 않는다.
- 174,952행, H4 가용 167,576행. 중복·무한값·전월 정렬 검사.
- 모델별 적합 전에 학습행만으로 필요한 스케일링을 적합한다. 전체기간 EPU z-score를 예측 전처리에 쓰지 않는다.

실행: `python -m src.team.dan.h4_export`
입력 준비: `python -m src.team.dan.h4_rank_comparison --prepare-only`
두 번째 명령은 최신 팀 코드·입력이 있는 환경에서만 실행 가능하다. 실제 예측·공식 결과 저장을 하지 않는다.

## 중요한 결합 조건

1. **순위 변환 예외:** 최신 main prepare_inputs는 {path, column, transform}을 지원한다. 세 H4 열에 transform='none'과 data/ 상대경로를 명시했다. 기본 rank를 사용하면 월별 EPU가 횡단면 상수로 바뀌고, 양수 EPU×MOM은 MOM과 같은 순위가 되어 H4 정보가 소실된다.
2. **주효과 보존:** 원 MOM·EPU·원 MOM×EPU를 함께 고려한다. 기존 모델에 MOM이 있으면 원값/순위/표준화 여부를 확인해 중복 열과 설계행렬 선형종속을 검사한다. 세 변수를 무조건 추가해서 완료로 간주하지 않는다.
3. **공통 비교:** H1 기준과 확장모형을 같은 H4 관측 학습행·평가 표본·Fold·목표 변환에서 비교한다. H1 구성요소 개별형인지 H1 지수형인지 먼저 고정한다. 직접 특징 IC는 추가 예측력 평가를 대신하지 않는다.
4. **모형 검증:** 단순 중복, 상수 열, rank deficiency, 결측 예측 처리, 스케일링, fold별 미래정보 차단을 결합 환경에서 검사한 뒤 공식 finalize/save_result를 실행한다.

## 현재 부족한 공식 실행 입력

2026-11-16 main의 prepare_inputs를 재확인했다. 현재는 panel·eligibility·h1_components를 직접 읽고 H3 grid를 계산한다. 이전 버전과 달리 h3.parquet를 직접 읽는 조건은 없다.

- data/interim/panel.parquet: 없음
- data/interim/eligibility.parquet: 없음
- data/processed/h1_components.parquet: 없음
- 최신 src.evaluation.model_test 및 config MODEL_GRID: 로컬 없음

현재 직접 특징 IC에는 이 파일들이 필요하지 않다. main은 바뀔 수 있으므로 공식 비교 실행 시 코드 버전을 고정한다.

## 커밋 범위

개인 src/team/dan, notebook/dan 중 검토한 파일과 공유할 실행 ZIP만 선택한다. data/, outputs/run_id/, 개별 CSV, 수정된 공용 src/features/h1.py/h4.py는 제외한다. 연결 코드는 입력 준비 도구라는 상태를 유지한다.
2026-11-16 개인 파일과 대표 ZIP 2개를 명시 경로로 stage 목록에 반영했다. 공용 수정·데이터·백업·이전 중복 ZIP은 제외했다.
기존 ZIP은 실행 당시 기록 그대로 보존했으므로 현재 수정된 코드 해시와 다를 수 있다. 업데이트된 코드로 새 직접 IC 결과를 생성해 별도로 보관한다.

## 최종 확인

단위 테스트 5개와 문법 검사를 통과했다. 새 직접 IC 실행(20261116T153929)은 이전 251개월 IC와 동일하다. 모델용 입력 생성·키·곱·목표 독립성을 확인했다. data/와 검수 기록은 Git 제외다. 현재 공용 h4.py는 초기 백업보다 EPU 추가 구현이 빠진 상태로 확인되었으며 이번 작업에서 수정·복구하지 않았다. 초기 EPU 구현 사본은 백업과 개인 h4_epu.py에 보존되어 있다. 기존 공용 h4.py 끝의 빈 줄 경고는 이번 개인 변경 범위 밖이다.
