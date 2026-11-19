# H2 — 통화정책과 기업 부채비율을 활용한 주식수익률 예측

본 문서는 `finance_analytics` 프로젝트의 **H2(통화정책 × 기업 부채비율)** 분석을 설명한다. H2의 데이터 준비·Hybrid 부채 특징 생성·기존 평가 및 추가 모델 실험을 함께 정리했으며, 코드는 제공된 `src(8).zip`의 `src/features/h2.py`, `src/h2/`를 기준으로 확인했다. **실험 결과 표는 별도로 제공된 `model_summary(3).csv`, `model_summary(4).csv`를 기준으로 작성했다.**

> **핵심 질문:** 한국은행 통화정책 기조와 기업의 부채비율, 그리고 이들의 상호작용은 일반적인 기업 특성만 사용하는 모델보다 **다음 달 종목별 주식수익률 순위**를 더 잘 예측하는가?
>
> 본 프로젝트는 **미래 수익률의 횡단면 예측**을 다루며, 통화정책 충격의 인과효과를 추정하는 연구는 아니다.

## 1. 연구 배경 및 가설

통화정책 변화는 기업의 자금조달 비용, 할인율, 투자 여력 등에 영향을 미칠 수 있다. 특히 부채 수준이 다른 기업은 같은 통화정책 변화에도 주가 반응이 다를 수 있다. 이러한 기업별 이질성이 월별 주식수익률 예측에 유용한지 확인한다.

- **H2 연구 가설:** 통화정책 기조의 수익률 예측 효과는 기업의 부채비율에 따라 달라진다.
- **검증 대상:** 기준월에 알려진 기업 특성과 통화정책 정보를 사용한 **다음 달 수익률 순위 예측**.
- **핵심 비교:** 기본 특성 모델 대비 `debt`, `policy`, `policy × debt` 및 **부채 분위별 상호작용**의 추가 기여.

## 2. 참고 논문과 본 프로젝트의 적용

### 2.1 Arin et al. (2025) — 통화정책 충격에 대한 기업별 주가 반응

**서지정보**

> Arin, K. P., Kaplan, S., Polyzos, E., & Spagnolo, N. (2025). *Stock market responses to monetary policy shocks: Firm-level evidence*. **Journal of Macroeconomics, 83**, 103646. https://doi.org/10.1016/j.jmacro.2024.103646

**논문에서는 무엇을 했는가?**

- 미국 기업 단위 자료를 활용하여 **예상하지 못한 통화정책 충격 및 비전통적 통화정책 충격**에 대한 주가 반응을 분석했다.
- 기업별 반응의 차이를 설명하는 특성으로 **부채/레버리지, 기업규모, 기업연령** 등을 살펴보았다.
- 기업의 **부채 규모**가 부채 구성보다 중요했으며, 부채 수준에 따른 이질적 반응, 특히 **세 번째 부채 분위에서 나타나는 결과**를 보고했다.
- 주요 관심사는 정책 충격 이후 나타나는 주가의 반응과 기업 간 차이이지, 우리 프로젝트처럼 월별 RankIC를 최대화하는 모델 선정이 아니다.

**우리 프로젝트에서는 어떻게 적용했는가?**

1. **정책 정보:** 미국 통화정책의 외생적 충격 대신, 한국은행 통화정책방향 문서로부터 구축한 `PolicyScore` (`final_rule_v2_sensitivity`)를 사용했다.
2. **부채 수준:** DART 공시일 우선·DataGuide 시차 보완 방식의 **Hybrid 부채비율**을 사용했다.
3. **연속형 결합:** `policy × debt` 상호작용을 특징으로 생성하고, 해당 변수를 넣기 전후의 예측 성능을 비교했다.
4. **비선형 확장:** 부채를 기준월별 4분위로 구분하고, Q1을 기준으로 Q2~Q4 더미와 `policy × 분위 더미`를 추가했다. 논문의 부채 수준별 분석에서 **아이디어를 가져온 구현**이다.
5. **검증 목적의 전환:** 정책 충격의 주가 반응을 추정하는 대신, 위 정보를 추가했을 때 **월별 종목 간 Spearman RankIC가 개선되는지** 평가했다.

**중요한 차이:** `PolicyScore`는 중앙은행 문서의 기조를 점수화한 값이므로, 논문의 *예상하지 못한 정책 충격(unanticipated monetary policy shock)*과 동일하지 않다. 따라서 논문에서 보고한 기업별 반응을 본 데이터에 그대로 재현했다고 주장하지 않는다. 부채 분위 실험 역시 논문 결과를 동기 삼아 만든 **예측 Feature Engineering**으로 해석한다.

### 2.2 Gu, Kelly & Xiu (2020) — 기업 특성과 머신러닝 수익률 예측

**서지정보**

> Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. **The Review of Financial Studies, 33**(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009

**논문에서는 무엇을 했는가?**

- 다양한 기업 특성을 사용해 주식수익률의 **횡단면 예측**을 수행하고, 전통적 통계모형과 여러 머신러닝 방법을 비교했다.
- 기본적인 기업 특성의 예로 **Size, Book-to-Market, Momentum**을 다루고, 모멘텀·유동성·변동성 관련 특징의 예측 중요성을 논의했다.
- 여러 특징과 비선형 관계를 학습할 수 있는 방법들의 예측 성능을 실증적으로 비교했다.

**우리 프로젝트에서는 어떻게 적용했는가?**

1. 정책·부채 변수의 성능을 평가하기 전에 `Size`, `B/M`, `Momentum`, `EWMA Volatility`로 **기본 모델(Baseline)**을 구성했다.
2. 기존 H1에서 계산한 `bm`, `mom`과 H3에서 계산한 `sigma`를 결합하여 기본 특성을 임의로 대체하지 않았다.
3. Ridge, Random Forest, XGBoost, LightGBM의 **4가지 예측모델**을 동일한 검증 설정에서 비교했다.
4. 정책·부채 변수의 포함/제외에 따른 **ΔRankIC**를 비교해 이 변수들의 *추가 예측력*을 판단했다.

**중요한 차이:** Gu et al.의 방대한 특징 집합이나 전체 학습·검증 절차를 정확하게 복제한 것은 아니다. 특히 우리의 주요 지표는 월별 **Spearman RankIC**이며, 논문과 동일한 데이터·목표변수·모델 사양을 사용했다고 주장하지 않는다.

### 2.3 논문과 H2 실험 설계의 연결 요약

| 구분 | 참고 논문의 접근 | H2에서 실제 사용한 접근 |
| --- | --- | --- |
| 분석 대상 | Arin: 미국 기업별 정책 충격에 대한 주가 반응 | 한국 주식의 월별 다음 달 수익률 순위 |
| 통화정책 | Arin: 예상하지 못한/비전통적 통화정책 충격 | BOK 문서 기반 `PolicyScore` |
| 부채 정보 | Arin: 기업 부채 수준별 반응 차이 | Hybrid 부채비율, 월별 Z-score, 부채 4분위 |
| 상호작용 | 정책 효과의 기업별 이질성 분석 | `policy × debt`, `policy × 부채 분위 더미` |
| 기본 특징 | Gu et al.: 기업 특성과 다양한 예측변수 | Size, B/M, Momentum, EWMA Volatility |
| 예측 방법 | Gu et al.: 여러 ML 기법 비교 | Ridge, Random Forest, XGBoost, LightGBM |
| 평가 목적 | 논문마다 정책 반응/수익률 예측 등 상이 | 월별 RankIC와 추가 특징의 ΔRankIC |

## 3. H2 데이터 및 변수

### 3.1 통화정책 (`policy`)

한국은행 통화정책방향 문서를 수집·정제하고 Rule v2 Sensitivity 방식으로 정책 기조를 점수화한 자료를 사용한다. 추가 실험에서 사용하는 필드명은 `final_rule_v2_sensitivity`이고 모델 입력에서는 `policy`로 변환한다. 동일한 기준월에 속하는 종목은 동일한 정책 점수를 공유한다.

### 3.2 Hybrid 부채비율 (`debt`)

부채 원천은 **DART 실제 공시일 기준의 이용 가능 정보**를 우선 사용하고, 사용 가능한 자료가 부족한 구간을 **DataGuide 6개월 지연 자료**로 보완하는 Hybrid 방식이다. 기본 입력 예시는 다음과 같다.

```text
data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv
```

`src/features/h2.py`와 추가 실험의 기본 부채 원천은 `hybrid`이다. 추가 실험의 `prepare()`에서는 적격 종목의 부채 원값을 **기준월별 1%·99% 윈저라이징 후 표준화**해 `debt`를 만든다. `policy_x_debt = policy * debt`이다.

> `src/features/h2.py`가 만든 `data/processed/h2.parquet`와 추가 실험에서 직접 읽는 Hybrid 원천 CSV는 **서로 다른 산출물**이다. 최종 모델이 실제로 어느 파일을 사용하는지 반드시 확인한다. 역사적 정정공시까지 완전히 복원된 PIT 데이터인지에 대한 제한도 별도로 기록한다.

### 3.3 기본 예측변수

| 변수 | 코드 필드 | 설명 | 입력 출처 |
| --- | --- | --- | --- |
| 기업규모 | `size` | 시가총액 로그 | 기본 패널 |
| 장부가치/시장가치 | `bm` | B/M | `h1_components.parquet` |
| 모멘텀 | `mom` | 과거 수익률 기반 모멘텀 | `h1_components.parquet` |
| 변동성 | `sigma` | 기존 H3의 EWMA 변동성 | `h3.parquet` |
| 부채비율 | `debt` | Hybrid 원천의 월별 표준화 값 | Hybrid 부채 CSV |
| 정책 기조 | `policy` | BOK 정책문서 점수 | H2 월별 패널 |
| 연속형 상호작용 | `policy_x_debt` | `policy × debt` | 계산 변수 |
| 부채 분위 | `debt_quartile` | 기준월별 부채 순위 4분위(Q1~Q4) | 계산 변수 |

`run_h2_paper_models.py`에는 입력 데이터가 없을 때 간이 모멘텀·변동성을 만드는 경로가 있다. 다만 **최종 확장 실험** `run_h2_feature_upgrade.py`는 원래 H1·H3 특징이 존재해야 하며, `run_h2_debt_quartile.py`도 기본 특징이 정확히 4개인지 검증한다.

## 4. 코드 구성

```text
src/
├── features/
│   └── h2.py                             # H2 공통 특징표 생성 (Hybrid 기본)
└── h2/
    ├── data_io.py                        # 경로·파일 입출력 및 검증
    ├── pipeline.py                       # 기존 H2 재현 파이프라인
    ├── preprocessing/                    # BOK / DART / Hybrid 데이터 준비
    ├── features/                         # 정책 자료 준비·점수화·변환
    ├── models/                           # 기존 H2 모델 함수
    ├── evaluation/                       # RankIC, 상호작용, 공시 점검
    └── experiments/
        ├── recover_h2_missing_features.py
        ├── run_h2_paper_models.py
        ├── run_h2_factorial_models.py
        ├── run_h2_feature_upgrade.py
        └── run_h2_debt_quartile.py
```

기존 `src/h2/evaluation/`은 이전 분석·평가의 재현을 담당하고, `src/h2/experiments/`는 새로운 변수 구성에 따른 **예측력 비교 실험**을 담당한다.

## 5. 실험 설계

### 5.1 M0~M4: 일반 Feature를 통제한 H2 변수 포함/제외

| 실험 | 특징 구성 | 검증 목적 |
| --- | --- | --- |
| M0 | Size + B/M + Momentum + EWMA | 기본 예측 성능 |
| M1 | M0 + Debt | 부채의 추가 기여 |
| M2 | M0 + Policy | 정책의 추가 기여 |
| M3 | M0 + Debt + Policy | 두 변수의 결합 효과 |
| M4 | M3 + Policy × Debt | 명시적 상호작용의 추가 기여 |

이 변수 구성은 **H1·H3 원래 특징을 확보한 `run_h2_feature_upgrade.py` 기준**이다. 간이 입력을 허용하는 `run_h2_factorial_models.py`에서는 사용 가능한 기본 특징이 다를 수 있다.

### 5.2 A~D: 부채 4분위·정책 상호작용

| 실험 | 특징 구성 | 검증 목적 |
| --- | --- | --- |
| A | Size + B/M + Momentum + EWMA | 기준 모델 |
| B | A + Policy + 부채 분위 더미(Q2~Q4) | 부채 수준의 비선형 정보 |
| C | B + Policy × 부채 분위 더미(Q2~Q4) | 분위별 정책 차이의 추가 정보 |
| D | A + 연속형 Debt + Policy | 기존 M3와 비교 |

부채 분위는 **각 기준월에 적격 종목의 부채 원값을 기준으로 순위를 산출**해 나누며, `Q1`을 기준범주로 사용한다. `Q2`~`Q4`에 대한 3개 더미와 상호작용 변수를 만든다. B와 C가 A보다 나은지, C가 B보다 나은지, C가 D보다 나은지 비교한다.

> Arin et al.(2025)의 부채 분위별 주가 반응에 착안했지만, 본 실험은 그 논문의 충격반응 함수를 재추정하는 과정이 아니다.

### 5.3 학습 및 평가

- **검증 기간:** 2006~2021년, 연도별 Expanding Window.
- **학습:** 검증연도 이전에 이용 가능한 학습 데이터를 사용하며, 미래 목표월은 학습에 포함하지 않는다.
- **목표값:** 다음 달 종목별 수익률(`return_t1`).
- **모델:** Ridge, Random Forest, XGBoost, LightGBM.
- **평가지표:** 월별 Spearman RankIC, 평균 RankIC, 월별 ICIR 및 모델 간 ΔRankIC.
- **최종 테스트:** 2022~2025년은 이 실험의 Feature 선택과 하이퍼파라미터 결정에 사용하지 않는다.

모델별 RankIC가 양수라는 사실만으로 개별 H2 변수가 효과적이라고 해석하지 않는다. **M0 또는 A를 기준으로 H2를 추가했을 때 성능이 개선되는지**가 핵심이다.

## 6. 실행 방법

모든 명령어는 **프로젝트 최상위 폴더(`finance_analytics/`)**에서 실행한다. `src/h2/experiments/`의 모듈은 패키지 상대 import를 사용하므로 `python -m` 방식으로 실행한다.

### 6.1 Hybrid H2 특징표 점검

```powershell
# 기존 h2.parquet와 새 Hybrid 계산값 비교 — 기본적으로 덮어쓰지 않음
python -m src.features.h2

# 파일이 없거나 기존 결과와 같은 경우에만 생성/유지하는 실행 옵션
python -m src.features.h2 --execute
```

기존 결과가 다른 경우 보호 로직이 중단시킨다. `--execute --replace`는 백업 후 교체 작업을 수행하므로, **공용 파일을 교체하기로 명시적으로 결정한 경우에만** 사용한다.

### 6.2 H1·H3 특징 확인

```powershell
python -m src.h2.experiments.recover_h2_missing_features
```

필요한 원천 자료가 준비됐고 특징 파일이 없는 경우에만 `--execute`로 생성한다.

### 6.3 기본 포함/제외 실험

```powershell
python -m src.h2.experiments.run_h2_factorial_models --check-only
python -m src.h2.experiments.run_h2_factorial_models --years 2006-2007 --models Ridge,RandomForest
python -m src.h2.experiments.run_h2_factorial_models
```

### 6.4 H1·H3 원래 특징을 사용한 최종 확장 실험

```powershell
python -m src.h2.experiments.run_h2_feature_upgrade --check-only
python -m src.h2.experiments.run_h2_feature_upgrade
```

`RateChange` 월별 CSV가 확보되면 `--rate-file`로 전달해 M5·M6를 추가할 수 있다. 현재 코드에는 ROA·유동비율이 별도 특징으로 포함되어 있지 않다.

### 6.5 부채 4분위 실험

```powershell
python -m src.h2.experiments.run_h2_debt_quartile --check-only
python -m src.h2.experiments.run_h2_debt_quartile
```

### 6.6 기존 H2 평가 코드

```powershell
python -m src.h2.evaluation.evaluate_rankic_yearly_models --help
python -m src.h2.evaluation.evaluate_h2_interaction_yearly_models --help
```

`--help`는 사용 가능한 옵션만 출력하며 모델 학습을 수행하지 않는다. 다른 프로젝트에 `--project`를 지정하는 경우 기본 입력경로가 그 프로젝트를 기준으로 해석되도록 코드가 수정됐는지 확인한다.

## 7. 결과 저장 위치

| 실행 모듈 | 기본 결과 경로 |
| --- | --- |
| `run_h2_paper_models.py` | `outputs/h2_paper_features/<실행시각>/` |
| `run_h2_factorial_models.py` | `outputs/h2_factorial/<실행시각>/` |
| `run_h2_feature_upgrade.py` | `outputs/h2_feature_upgrade/<실행시각>/` |
| `run_h2_debt_quartile.py` | `outputs/h2_debt_quartile/<실행시각>/` |

실험에 따라 `model_summary.csv`, `monthly_rankic.csv`, `comparisons.csv`, `monthly_paired_delta.csv`, `feature_sets.csv`, `data_checks.json` 등이 생성된다. **현재 제공된 코드에는 `--share` 옵션이 없으므로** 팀 공유용 경로로 자동 복사된다고 가정하지 않는다.

## 8. 실험 결과 요약

아래 수치는 제공된 결과 CSV의 **2006~2021년, 192개월 내부 검증**에 해당하며 2022~2025년 최종 테스트 결과는 아니다.

### 8.1 일반 특징 + H2 연속형 변수(M0~M4)

| 모델 | M0 기본 | M1 부채 | M2 정책 | M3 결합 | M4 상호작용 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.06463 | 0.06220 | 0.06469 | 0.06239 | 0.06188 |
| Random Forest | **0.06628** | 0.06335 | 0.06440 | 0.06101 | 0.06027 |
| XGBoost | 0.06346 | 0.06001 | 0.06291 | 0.06131 | 0.05816 |
| LightGBM | 0.06032 | 0.05619 | 0.05742 | 0.05417 | 0.05198 |

최고 평균 RankIC는 **Random Forest의 M0(0.06628)**였다. Ridge에서 정책 점수만 추가한 M2가 M0보다 약간 높았으나, 부채와 정책을 함께 추가한 M3 및 상호작용항을 포함한 M4는 모든 모델에서 M0보다 낮았다.

### 8.2 부채 4분위 확장(A~D)

| 모델 | A 기본 | B 부채 분위 | C 분위 상호작용 | D 연속형 부채 |
| --- | ---: | ---: | ---: | ---: |
| Ridge | **0.06463** | 0.05729 | 0.05625 | 0.06239 |
| Random Forest | **0.06628** | 0.06275 | 0.06185 | 0.06101 |
| XGBoost | **0.06346** | 0.06235 | 0.06047 | 0.06131 |
| LightGBM | **0.06032** | 0.05503 | 0.05559 | 0.05417 |

부채 4분위 및 분위별 상호작용을 넣어도 **모든 모델에서 A가 가장 높은 성능**을 기록했다. 부채 분위 방식은 일부 모델에서 연속형 부채 방식보다 우수했지만, 기본 모델을 넘어서는 추가 예측력은 나타나지 않았다.

### 8.3 결론

- 통화정책·부채비율 및 상호작용의 **추가 예측력은 이번 내부 검증에서 일관되게 지지되지 않았다.**
- 이는 정책과 부채 사이에 경제적 관계가 없다는 뜻은 아니다. **충격반응 분석과 미래 수익률 순위 예측은 연구 질문이 다르다.**
- 최종 통합 모델에는 H2 변수를 무조건 포함하기보다, 동일한 학습·검증 구간에서 **H2 포함/제외 모델을 비교한 뒤 결정**한다.
- 월별 RankIC 차이의 유의성은 `monthly_paired_delta.csv`를 이용해 별도로 검정할 수 있다. 본 실험 코드가 HAC p-value를 기본 결과로 자동 산출한다고 가정하지 않는다.

## 9. 재현성 및 주의사항

1. **모듈 실행:** 항상 프로젝트 루트에서 `python -m src.h2.experiments.<모듈명>`을 사용한다. 파일 상단에 남아 있는 예전 루트 실행 예시는 사용하지 않는다.
2. **동일 표본 비교:** Feature 결측·적격종목 조건이 달라지면 RankIC 비교도 달라질 수 있다. `data_checks.json`과 월별 평가 종목 수를 함께 보관한다.
3. **미래 정보 차단:** 정책 발표일, DART 공시일, 회계 기준일 및 목표월의 정렬을 확인한다.
4. **Hybrid 버전 고정:** `data/processed/h2.parquet`를 교체할 경우 기존 산출물 백업과 입력 파일 해시를 기록한다. Hybrid 원천 CSV를 직접 읽는 실험 결과와 공통 특징표의 결과를 혼동하지 않는다.
5. **최종 검증 구간 분리:** 2006~2021년 내부 검증으로 변수 구성을 정하고, 2022~2025년 최종 평가 구간을 특징 선택에 사용하지 않는다.
6. **실행 결과 보존:** 실험별 출력 폴더와 실행 옵션을 함께 보관하고, 기존 결과를 덮어쓰지 않는다.
7. **논문과 구현 구분:** 논문의 통화정책 *충격*을 BOK 텍스트 *기조 점수*와 동일시하지 않으며, 논문의 결과를 그대로 재현한 것으로 서술하지 않는다.

## 10. 참고문헌

1. Arin, K. P., Kaplan, S., Polyzos, E., & Spagnolo, N. (2025). *Stock market responses to monetary policy shocks: Firm-level evidence*. **Journal of Macroeconomics, 83**, 103646. https://doi.org/10.1016/j.jmacro.2024.103646
2. Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. **The Review of Financial Studies, 33**(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009

---

*문서 범위: `src(8).zip`의 H2 코드 및 이번 대화에서 제공된 두 확장 실험 결과 CSV. 원문 논문의 연구 질문과 프로젝트 구현의 차이를 명시했다.*
