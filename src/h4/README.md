# H4 EPU 입력과 Ridge 비교

2026-11-18 기준. 저장소 루트에서 실행한다. 공용 features/config는 수정하지 않는다.
최종 테스트(2022~2025)는 열거나 사용하지 않는다.

## 폴더 구조

```text
src/h4/
  features/       EPU 정렬·상호작용 특징
  preprocessing/  모델 입력 parquet 생성
  evaluation/     직접 특징 IC·Ridge 비교·입력 준비
  analysis/       기존 탐색 회귀·단독 OLS 분석
  tests/          특징·미래정보 차단·표준화 검증
  README.md
  requirements.txt
notebook/h4/      기존 노트북과 과거 분석 기록
```

모델 적합은 팀 공통 `src/models`와 `src/evaluation/model_test.py`를 재사용한다.
같은 역할의 모델 구현을 H4 안에 복제하지 않는다.

## 설치와 입력

```powershell
python -m pip install -r src/h4/requirements.txt
```

공통 코드와 `config/experiment.yaml`이 있는 최신 저장소가 필요하다.
필요한 입력은 다음과 같다. 데이터는 Git에 포함하지 않고 별도로 전달한다.

- `data/interim/panel.parquet`: 2000~2021년 개발 패널
- `data/interim/eligibility.parquet`: 공통 적격성
- `data/processed/h1_components.parquet`: 공통 H1 구성요소
- `data/processed/mom.parquet`: 팀의 원 MOM, H4 생성에 사용
- `data/raw/Korea_Overall_EPU.xlsx`: `Korean New EPU Index` 시트

## H4 입력 파일 전달

팀원에게 전달할 파일명은 **`data/processed/h4_epu.parquet`**다.
이전에 전달한 `dan_h4_epu.parquet`와 2026-11-18 이름 변경 시점의 내용은 동일하다.
기존 이름의 로컬 사본은 이전 실행과 중단된 작업을 위해 보존한다.
코드는 앞으로 `h4_epu.parquet`를 사용한다.

열: `stock_id`, `base_month`, `mom`, `epu_month`, `epu_raw`,
`h4_interaction`, `h4_available`, `h4_exclusion_reason`.
기준월 2000-12~2021-11, 목표수익률 없음. 기존 생성본은 174,952행이다.
기준월 t에는 EPU(t-1)를 연결하고 상호작용은 원 MOM × EPU로 만든다.

파일이 이미 있으면 생성 명령은 생략한다. 재생성 명령은 기존 H4 파일을 덮어쓴다.

```powershell
python -m src.h4.preprocessing.export
```

기존 MOM이 있는 기준모형에는 `epu_raw`와 `h4_interaction`만 추가한다.
두 열은 월별 순위로 바꾸지 않는다. 원 MOM을 기존 MOM 옆에 무조건 중복 추가하지 않는다.
`h4_mom`은 별도 모형 구성을 위한 선택 입력이며 이번 Ridge에는 사용하지 않는다.

## Ridge 실행과 결과 확인

```powershell
python -m src.h4.evaluation.ridge --prepare-only
python -m src.h4.evaluation.ridge
```

`python -m src.h4.evaluation.prepare --prepare-only`도 동일한
Ridge 입력 준비를 호출한다. 입력 파일을 자동 재생성하지 않는다.

- 기준: Size + BM + MOM + sigma; 확장: 같은 변수 + EPU + 원 MOM×EPU.
- 학습 시작 2001년, 바깥 검증 2006~2021년(192개월).
- 각 검증연도 Y: 2001~Y-2 학습, Y-1 내부 검증, Y 평가. 설정 선택 후 재학습 없음.
- 동일한 H4 관측 학습행과 평가 행을 두 모형에 사용한다.
- H1/H3는 공통 월별 순위 입력. 모든 설계 열의 평균·표준편차는 내부 학습행에서만 계산한다.
- 예측 시 결측은 표준화 후 0(학습 평균)으로 채운다. EPU를 순위 변환하지 않는다.
- Ridge 규제 강도와 H3 후보는 공통 실행기의 내부 IC 기준으로 각각 선택한다.
- 목표값: 원수익률(주 분석), 1·99% 윈저 수익률, 월별 표준화 수익률(보조).
- `full`: 수익률이 있는 전체 평가 행. `common`: 모든 특징이 관측된 공통 행.
- Rank IC는 매월 예측·실제 원수익률의 Spearman 상관. ICIR은 월평균/표본표준편차이며 비연율화.

매 실행은 `outputs/model_results/h4/<실행시각>_h4_paired_oos_ridge/`를 새로 만든다.
`paired_summary.csv`, `paired_monthly_ic.csv`, `paired_annual_ic.csv`에 비교 결과를 저장한다.
`fold_table.csv`에는 선택한 설정, `train_only_scales.csv`에는 학습 전용 표준화값을 저장한다.
개별 종목 예측값 parquet는 로컬에만 두고, 공유는 **result.zip만** 한다.

대표 검증 실행: `20261118T134947_h4_paired_oos_ridge/result.zip`.
원수익률·전체 표본: 평균 IC 0.082221 → 0.081415, ICIR 0.770300 → 0.766166.
H4 추가로 소폭 하락했으며, 이를 다른 모델이나 최종 테스트 결과로 일반화하지 않는다.
저장 당시 예측값의 월별 IC 2,304개를 SciPy로 독립 검산했다.
과거 실행 ZIP/해시는 당시 기록이며 이후 코드 정리본과 코드 해시가 다를 수 있다.

## 직접 특징 분석과 테스트

```powershell
python -m src.h4.evaluation.feature_rank
python -m unittest src.h4.tests.test_features src.h4.tests.test_ridge -v
```

직접 특징 분석은 MOM·개발 패널·EPU를 사용하고, 학습 없이 특징 순위와 실제 수익률 순위를 비교한다.
이는 Ridge 예측 IC나 EPU의 추가 예측력 검증과 다르다.
`analysis/inference.py`는 별도의 패널 OLS·HAC 탐색 분석이고, `analysis/standalone_rank.py`는 이전 단독 OLS 실험이다.

EPU 과거 공표 시점·개정 이력은 별도 검증 대상이다. 전월 연결만으로 완전한 실시간 빈티지 검증을 뜻하지 않는다.
이번 공유 범위에 미완료 LightGBM/XGBoost 코드는 포함하지 않는다.

기존 개인 경로의 구현을 `src/h4/`로 옮겼다. 실행 시 위 새 명령을 사용한다. 호환용 복제본은 두지 않는다.
