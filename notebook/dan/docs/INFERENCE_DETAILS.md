> 이전 단계 설명을 보존한 문서입니다. 현재 파일 위치·공유 상태는 상위 README.md가 기준입니다. 아래의 공식 비교 설명은 미실행 초안이며 실행 완료를 뜻하지 않습니다.

# Dan H4: EPU proxy experiment

기존 `src/features/h1.py`, `src/features/h4.py`와 루트 임시 파일은 수정하지 않았다.
원본 사본, Git diff, 상태 기록은 `outputs/run_id/dan_preorganization_20261116/`에 보관했다(Git 제외).

## 실행

저장소 루트에서 기존 가상환경을 사용한다.

```powershell
.\.venv\Scripts\python.exe -m unittest src.team.dan.test_h4 -v
.\.venv\Scripts\python.exe -m src.team.dan.h4_analysis
```

입력: `data/raw/mom.parquet`, `panel.parquet`, `Korea_Overall_EPU.xlsx`.
MOM은 팀 제공 값을 재사용하며 H1 개인 구현을 import하지 않는다.
입력·코드 SHA256, 환경 버전, 표본 수, 실험 조건은 result.zip/run.json에 기록한다.
목표월 2001-01~2021-12만 분석하며 Test 성과를 계산하지 않는다.

## 해석 조건

- H4는 사용자 제공 새 정의에 따라 Cho–Kim(2023)의 한국 EPU 지수를 사용한다. 기사를 직접 수집하거나 기사 비율을 계산하지 않는다.
- 전월 EPU를 기준월에 사용한다. EPU 검증 완료는 사용자 확인(2026-11-16)에 근거한다.
- 기존 분석대로 가격 1,000원 이상, 양수 시가총액, 유효 MOM/EPU/목표값을 사용한다.
- 주분석은 월별 MOM·목표수익률 1/99% 윈저라이징, 민감도 분석은 원값이다.
- EPU는 분석 표본 월에 동일가중 표준화(ddof=1). 전체 Train 기반 기술적 추론용이며 예측 검증 전처리로 재사용하지 않는다.
- 패널 OLS: 월 고정효과(월별 평균 제거), MOM·EPU×MOM·log_size, 월 군집 표준오차. EPU 주효과는 월 고정효과에 흡수된다.
- 2단계: 월별 y~상수+MOM의 기울기를 EPU z-score에 회귀한다. 1단계에는 규모 통제를 넣지 않으므로 패널 OLS와 다른 추정대상이다. 최소 월별 30종목과 연속 월을 요구한다.
- HAC lag=floor(4*(T/100)^(2/9)); 단측 p=정규분포 CDF(통계량), 대립가설 gamma<0.
- 패널·2단계·민감도 중 유리한 결과를 골라 확증 결과로 보고하지 않는다. 기존 결과를 본 후 정리한 탐색적 분석이다.
- Holm은 `--holm-family family.json`으로 전체 사전 지정 가설군의 p값을 입력할 때만 계산한다. H4는 주분석 2단계 단측 p와 일치해야 한다. 다른 가설 p값 미확보 상태에서는 보정 완료로 표시하지 않는다.

## 공식 Rank IC 비교

```powershell
.\.venv\Scripts\python.exe -m src.team.dan.h4_rank_comparison
```

현재 로컬 HEAD d7882aa에는 `src.evaluation.model_test` 및 최신 공통 입력이 없어 실행 불가다. 공용 파일을 임의 교체하거나 main을 병합하지 않았다.
관리자가 제공한 최신 공통 코드·config와 `data/interim/`, `data/processed/` 입력을 확보한 뒤 실행해야 한다.
코드는 공식 예제의 prepare_inputs(extra_features), plan_folds, fold_rows, validation_rows, save_result를 사용한다. 기준·확장모형을 같은 prepared 표본에서 비교한다.
공식 예제의 Fold 정책을 그대로 따르며, README 본문의 옛 Fold 설명과 차이가 있으므로 결과에 적용 공통 코드 버전을 확인한다.
양수 월별 EPU×MOM은 월별 횡단면 순위가 MOM과 같아져 공식 순위 변환에서 중복 특징이 될 수 있다. 이 비교는 중복 진단이며 원값 상호작용의 검정과 동일하지 않다.
생성된 `data/processed/dan_h4_epu.parquet`도 Git 제외다.

## 공유와 커밋 준비

`outputs/model_results/dan/<시각>_h4_epu_ols-inference/result.zip`에는 run.json·집계 검정표·월별 기울기만 있으며 원자료·종목별 예측은 없다.
이 묶음은 `INFERENCE_ONLY_NOT_TEAM_MODEL_COMPARABLE`: 공식 model_test의 PASSED 결과가 아니다. 추론 분석에는 학습 곡선이 적용되지 않는다.
공식 비교 묶음은 공통 save_result가 별도로 만든다.

현재 개인 브랜치 feature/epu-momentum을 유지한다. 개인 코드·노트북만 선택적으로 stage하고 result.zip만 강제 추가한다(로컬의 오래된 .gitignore가 모든 outputs를 제외하기 때문).
공용 파일의 기존 수정 및 루트 임시 파일, data/는 커밋에서 제외한다.
커밋 메시지 제안: `Add isolated Dan H4 EPU inference and official comparison adapter`.
커밋·push는 실행하지 않았다.
