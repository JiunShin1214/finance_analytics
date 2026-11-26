# H2 — 통화정책과 기업 부채 특성

> 개발 검증 및 재현성 정리 기준: 2026-11-26. 최종 평가(2022~2025) 및 완전한 historical PIT 인증은 완료되지 않았다.

## 1. 가설과 개발 결론

통화정책 변화에 대한 기업별 주식수익률 반응은 부채 특성에 따라 차이가 있을 수 있다는 가설을 검토한다. 정책 기조·부채 수준 중심의 초기 구성에서 정책 및 부채 변화, 실제 금리 변화와 상호작용항을 추가하였다. 목표는 다음 달 조정수익률의 횡단면 순위 예측이다. 인과효과 식별을 주장하지 않는다.

H2 단독 8개 변수 모델의 이전 비교 결과(2006~2021년, 192개월)는 RF 0.00841, LGB 0.00561, XGB 0.00520, Ridge 0.00323의 평균 RankIC였다. RF의 HAC(6) 양측 p-value는 약 0.0627이었다. 독립적인 예측력은 제한적이었다. 따라서 H2를 전체 기간에 직접 추가하는 대신 금리 인상 직후 목표월에만 H2 전문 모델을 약하게 결합하는 후보를 검증하였다.

## 2. 데이터 및 피처 정의

- 재무: DART 공시일 우선 + DataGuide 6개월 지연 보완 Hybrid 부채 자료. 공시일/결산일 조건을 검사하지만 원천의 역사적 vintage가 완전히 검증된 것은 아니다.
- 정책: 한국은행 통화정책방향 문서 기반 `final_rule_v2_sensitivity` (완화 +1, 긴축 -1). 기존 발표일 < 기준월 말일 규칙을 유지한다.
- 금리: 한국은행 정책금리 변경 이력 고정 스냅샷 (`build_bok_rate_history.make()`); 2008년 2월까지 콜금리 목표. 단위는 %p이며 외생적인 정책 충격의 추정치는 아니다.
- 공통 패널: 기준월 적격 종목을 대상으로 피처를 생성한다. 피처 생성 및 순위 산출 시 미래 수익률의 존재 여부로 종목을 고르지 않는다.

| 피처 | 정의 | 모델 입력 변환 |
|---|---|---|
| `debt` | 기준월 적격 종목의 부채비율 1/99% 윈저 후 z(ddof=0) | 추가 변환 없음 |
| `policy` | 기준월 통화정책 기조 | 추가 변환 없음 |
| `policy_change` | 직전 달력월 대비 기조 변화 | 추가 변환 없음 |
| `policy_x_debt` | policy × debt | 추가 변환 없음 |
| `debt_change` | 최신 이용 가능 보고서 부채비율 − 직전 다른 결산기 보고서 부채비율; 같은 보고서 기간에는 마지막 변화량 유지(비정상 상태는 결측) | 월별 평균 순위 `2r/(n+1)-1` |
| `rate_change` | 기준월 말 정책금리 − 직전월 말 정책금리 | 추가 변환 없음 |
| `policy_change_x_debt` | policy_change × debt | 추가 변환 없음 |
| `rate_change_x_debt` | rate_change × debt | 추가 변환 없음 |

`h2.parquet`에는 순위 변환 전 `debt_change` 원값을 저장한다. 첫 기준월(2001-01)의 정책 변화 및 피처용 금리 변화는 결측으로 유지한다. 게이트는 별도 월말 금리 이력에서 계산한다. 모델 입력 단계에서 기존 공통 변환 및 결측 0 대체 규칙을 재사용한다. 목표 수익률은 대체하지 않는다.

### 공식 피처 파일

`data/processed/h2.parquet` (별도의 specialist parquet 없음)

- 기준월 2001-01~2021-11, 대응 목표월 2001-02~2021-12
- 174,260행, 관리 열 4개(`stock_id`, `base_month`, `eligible`, `debt_raw`) + 모델 피처 8개, 키 중복 0
- `debt` 관측 158,046행(90.70%); `debt_change` 관측 146,531행(84.09%)
- 2026-11-26 확인 SHA-256: `f16815ef17cd38d7e050fbc8cfccdf98841a463d3630c2c8e971216d7680de62`
- 과거 4개 변수 및 동결된 8개 변수 Archive와 키/결측/피처 값 비교 완료(최대 절대 차이 0).

### 생성 및 교체

프로젝트 루트의 기존 `.venv`에서 모듈 방식으로 실행한다. 기존 원천·공통 패널·config가 필요하다.

```powershell
python -m src.features.h2 --hybrid-csv data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv
```

기본 동작은 읽기 전용 검사이다. 새 표와 기존 표가 동일하면 `EXISTS_SAME`이 출력된다. 실제 생성/교체는 명시적으로 `--execute` (기존과 다르면 `--execute --replace`)를 지정한다. 교체 검증 시 기존 4개 열과 8개 Archive를 비교하며, 불일치하면 중단한다. 교체 시 기존 parquet과 manifest를 백업한다. 이미 8개 피처로 교체한 환경에서는 재교체할 필요가 없다.

주요 필요 자료:

| 경로 | 용도 |
|---|---|
| `data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv` | Hybrid 부채 입력 |
| `data/raw/h2/monthly.csv` | 월별 정책 점수 입력 |
| `data/interim/panel.parquet` | 공통 패널 |
| `data/interim/eligibility.parquet` | 독립 적격 종목 검증 |
| `data/processed/archive/h2_8features.parquet` | 동결된 기존 8개 피처 비교 기준 |
| `config/experiment.yaml` | 공통 피처 경로 및 모델 설정 |
| `config/h2.yaml` | H2 동결 설정 및 재현성 해시 |

Archive는 피처 계산 원천이 아니라 검증 기준이다. Archive SHA-256: `8a504b7d02f5e961fe07302cb33ae30db17e67fefd92273311cc08019a598df1`. 이전 4개 변수 파일의 SHA-256: `ff77286361aa5df8e44e2ee5b3038d953eb84df7551b52cb20962902cb5ac580`. 기존 파일 및 manifest 백업과 Archive 원본 바이트를 삭제하지 않는다. 이전 감사는 옛 4개 변수 해시를 요구할 수 있으므로 별도의 과거 작업 사본에서 당시 입력/코드를 복원하여 재현한다. 정식 8개 파일로 옛 해시를 우회하거나 감사 기록을 덮어쓰지 않는다.

## 3. 조건부 전문 모델 설계

기준 모델은 Base + H1 + H3 + H4(30개 피처)이다. H2는 8개 피처만 사용하는 별도 모델이다. 기본 후보는 XGBoost Pairwise이며 전체 평가의 통계적 우월성이 입증된 것은 아니다.

| 설정 | 값 |
|---|---|
| 개발 외부 검증 | 2006~2021, 192개월 |
| 연도별 Expanding Window | 목표연도 Y의 학습은 2001년~Y-2년 12월; Y-1은 공통 모델의 내부 검증용으로 보존, 재학습 없음 |
| H2 학습 행 | 기존 `fit_member` 조건을 충족하고 기준월 금리가 인상된 표본 |
| 준비 조건 | 학습 대상 월 5개 이상, 모든 학습 대상 월 종목 20개 이상 |
| 게이트 | `hike_1m`: 기준월 금리 변화 > 0, 목표월은 기준월+1 |
| 모델 | `xgb_pairwise` (비교 지원: `xgb_rankreg`, `lgb_rankreg`) |
| 고정 설정 | 500 trees, depth 3, learning rate 0.01, XGB L2 0 |
| Pairwise | 목표월 단위 qid, 평균 순위 기반 0~9 라벨; pair method mean, 8 pairs/sample, seed 0 |
| 재현 환경 설정 | CPU, threads 4 (동결 참조와 일치해야 함) |
| 결합 | 활성화 월 Baseline 95%, H2 5% |

학습 순위 라벨은 월별 실제 수익률의 평균 순위 r과 종목 수 n으로부터 `floor(10*(r-1)/n)`으로 만든다. 실제 수익률은 학습용이며 예측 입력에 포함하지 않는다.

활성화 월은 기준월 말에 이미 관측된 금리의 양의 변화 및 학습 준비 여부로 결정한다. 인상 사이클 전체가 아니라 실제 금리 인상월의 다음 목표월만 대상으로 한다. 금리 정보 또는 준비 조건이 부족하면 기준 모델을 유지한다. 전체 사전 적격 종목(예측 유니버스)에서 두 점수의 평균 순위를 r_B, r_H, 종목 수 n이라 할 때 최종 점수는 `(95*r_B+5*r_H)/(100*(n+1))`이다. 비활성화 월은 Baseline 원점수를 그대로 보존한다.

이 규칙은 기준월 말 정보가 허용되는 순차 예측을 전제로 한다. 2021년 말에 이후 48개월을 한꺼번에 예측해야 하는 과제라면 2022~2025년 실현 금리와 재무정보를 사용할 수 없다. 현재 모듈은 개발기간으로 제한하며 `enabled_for_submission: false`를 유지한다.

## 4. 기존 실험 결과

동결된 ex-ante 유니버스 검증과 후속 PIT 민감도 검사(전체 종목, 192개월, 활성화 10개월)의 결과이다.

| 모델 | H2 제외 RankIC | H2 조건부 결합 RankIC | 차이 | HAC(6) 양측 p |
|---|---:|---:|---:|---:|
| LGB RankReg | 0.116862 | 0.116905 | +0.000044 | 0.1667 |
| XGB RankReg | 0.116581 | 0.116624 | +0.000043 | 0.1524 |
| XGB Pairwise | 0.119460 | 0.119484 | +0.000024 | 0.4468 |

XGB Pairwise 활성화 10개월의 평균 RankIC는 0.177523 → 0.177991이다. 전체 기간 개선은 매우 작으며 전체 종목 기준 5% 유의수준에서 유의하지 않다. 이 후보의 재현성은 확인했으나 일반화 성능이나 최종 포함을 확정하지 않는다.

후속 PIT 민감도 검사(XGB Pairwise, 전체 종목, 192개월): DataGuide 9개월 기준 사후 마스크 +0.000053, 12개월 기준 -0.000016, 미해결 후속공시 마스크 +0.000009 (각각 H2 제외 대비 RankIC 차이). 모두 전체 종목 기준 5% 유의수준에서 유의하지 않았다. 이는 이미 변환된 입력에 대한 마스킹 검사이며 원천 vintage의 완전한 재구축 또는 PIT 인증이 아니다.

## 5. 코드 구성

```text
config/h2.yaml                           # 동결 후보 설정
src/features/h2.py                       # 공식 8개 피처 생성
src/h2/
  preprocessing/build_bok_rate_history.py
  experiments/build_h2_debt_change.py    # 기존 보고서 변화량 생성 함수
  models/regime.py                       # 금리/학습 준비 게이트
  models/specialist.py                   # 학습 조건 및 기존 학습 함수 어댑터
  models/blend.py                        # 조건부 순위 결합
  evaluation/reproduce_specialist.py    # 기존 예측과 통합 재현 검사
  experiments/                           # 기존 검증 및 함수 의존성 보존
archive/h2/experiments/                  # 종료된 실험 14개 보관
archive/h2/maintenance/                  # 일회성 정리 스크립트 보관
test_h2_blend.py
test_h2_regime.py
test_h2_specialist.py
```

전문 모델은 독립적인 새 학습 알고리즘이 아니라 `evaluate_h2_training_period_and_features.fit_predict()`를 재사용하는 어댑터이다. 기존 `prepare_inputs`, conditional/exante/independent 실험 등에서 사용하는 함수와 검증 해시를 유지해야 한다. `src/h2/experiments` 전체를 이동하면 재현성이 깨질 수 있다. Archive 실험을 재현할 경우 해당 진입점을 원래 경로로 복원한 별도 작업 사본을 사용한다.

공통 `experiment.yaml`의 `PATHS.h2`, `H2.definition`, `BASE_FEATURES.model_input.main_run`은 기존 공통 모델과 공유한다. `H2.model_test.features`의 2개 입력 및 `base_model_runs.feature_specs`의 부스팅 4개 입력은 과거 직접 통합 비교용이며, 전문 모델 8개 설정으로 무조건 변경하지 않는다. `H2.enabled`와 `h2_specialist.enabled_for_submission`은 용도가 다르다. 후자는 false 유지.

활성 실험 코드 13개를 유지하고 종료된 실험 14개를 Archive에 보관하였다. 이동 후 단위 테스트 26개와 통합 `--check-only` 검사를 통과하였다. Archive의 스크립트는 현재 경로에서 바로 실행되는 것을 보장하지 않는다.

## 6. 재현 방법

PowerShell, 프로젝트 루트에서 기존 환경 사용. 재현 검사 전에 필요한 입력 파일 및 동결 참조가 준비되어야 한다. 기준 경로 `outputs/model_results/h2/exante_universe_v2/`의 `run_audit.json`, `fit_audit.csv`, `exante_active_predictions.csv`와 `outputs/model_results/h2/overfit_diagnostic_v1/run_audit.json`이 필요하다. `config/h2.yaml`의 해시를 임의 변경하지 않는다. 이 자료들과 공통 모델 코드/피처 입력 및 라이브러리 버전이 갖춰지지 않은 새 clone은 그대로 실행되지 않을 수 있다. 과거 실행 결과는 다시 생성한 파일로 몰래 대체하지 않는다.

```powershell
python -m compileall -q src/h2 src/features/h2.py
python -m unittest -v test_h2_specialist test_h2_regime test_h2_blend
python -m src.h2.evaluation.reproduce_specialist --check-only
```

재현성 사전 검사는 기본적으로 XGB Pairwise를 대상으로 한다. 정상 시 `CHECK-ONLY passed: fingerprints, routing, fit scope, universe, frozen blend; no fitting.`이 출력된다.

```powershell
python -m src.h2.evaluation.reproduce_specialist --execute
```

이미 기본 결과 폴더가 있으면 덮어쓰지 않고 새 폴더를 지정한다.

```powershell
python -m src.h2.evaluation.reproduce_specialist --execute --out outputs/model_results/h2/modular_reproduction_v2
```

### 2026-11-26 실제 로컬 통합 재현 기록

| 외부 검증연도 | 학습 종료 | 금리 인상 학습월 | 학습 행 | 예측 행 | H2 최대 절대 차이 | 결합 최대 절대 차이 |
|---|---|---:|---:|---:|---:|---:|
| 2008 | 2006-12 | 6 | 3,880 | 684 | 0 | 0 |
| 2010 | 2008-12 | 9 | 5,875 | 1,373 | 0 | 0 |
| 2011 | 2009-12 | 9 | 5,875 | 2,091 | 0 | 0 |
| 2017 | 2015-12 | 14 | 9,322 | 719 | 0 | 0 |
| 2018 | 2016-12 | 14 | 9,322 | 734 | 0 | 0 |
| 2021 | 2019-12 | 16 | 10,763 | 1,520 | 0 | 0 |

활성화 목표월: 2008-09, 2010-08, 2010-12, 2011-02, 2011-04, 2011-07, 2017-12, 2018-12, 2021-09, 2021-12. 합계 7,121행, 6개 Fold, XGB Pairwise 재학습 및 저장된 Baseline을 이용한 결합 결과 일치. 기존 점수의 native float32 비트 비교 및 결합 점수 비교를 통과했다. 출력 경로: `outputs/model_results/h2/modular_reproduction_v1/` (`reproduction.csv`, `active_predictions.csv`, `routing.csv`, `run_audit.json`).

이 검사는 Baseline을 재학습하지 않았으며 다른 두 비교 모델의 신규 모듈 재학습까지 확인한 것은 아니다. 과거 성능 결과의 재현이지 신규 독립 평가가 아니다.

## 7. 제출 전 남은 조건

- 2022~2025 데이터의 이용 가능 시점 및 과제의 예측 방식(순차/고정 시점) 확인.
- 원천 데이터의 라이선스, 공시 vintage 및 적격 종목 정합성 점검; 완전한 PIT 인증은 미완료.
- 공통 최종 예측 파이프라인 연결 및 팀원과 H2 채택 여부 합의. 2022~2025 실제 수익률로 후보를 선택하거나 튜닝하지 않는다.
- 공통 입력 및 동결 참조 결과의 전달/복원 방법과 환경 버전 기록. 코드를 공유한다고 필요한 데이터까지 자동 공유되는 것은 아니다.

현 상태는 개발기간 H2 전문 모델 및 결합 모듈의 재현 성공이다. 최종 `prediction.csv` 제출을 승인하거나 미래 성능을 보장하는 결과는 아니다.
