# H4 탐색 회귀 분석

```powershell
python -m src.h4.analysis.inference
```

입력: `data/processed/mom.parquet`, `data/interim/panel.parquet`,
`data/raw/Korea_Overall_EPU.xlsx`. 목표월 2001~2021년만 분석한다.

## 표본과 변환

- 전월 Cho–Kim 한국 EPU와 팀 원 MOM을 결합한다.
- 가격 1,000원 이상, 양수 시가총액과 유효 MOM·EPU·수익률을 요구한다.
- 주분석은 MOM·목표수익률을 매달 1·99%에서 제한하고, 원값 민감도 분석을 병행한다.
- EPU는 분석 월에 동일가중 표준화한다. 이 전체기간 변환은 기술적 추론용이며 예측 전처리에 쓰지 않는다.

## 추정과 해석

패널 OLS는 월 고정효과, MOM·EPU×MOM·log_size와 월 군집 표준오차를 사용한다.
EPU 주효과는 월 고정효과에 흡수된다.

2단계 분석은 월별 수익률~상수+MOM 기울기를 EPU z-score에 회귀한다.
최소 월별 30종목과 연속 월을 요구한다. 규모 통제가 없어 패널 OLS와 추정대상이 다르다.
HAC lag는 floor(4*(T/100)^(2/9)), 음의 상호작용 단측 p값은 정규분포 근사다.

Holm 보정은 `--holm-family family.json`으로 전체 가설군의 p값이 주어질 때만 수행한다.
일부 분석의 유리한 결과를 골라 확증 결과로 보고하지 않는다.
과거 EPU 공표·개정 이력 검증과 개발기간 OOS 평가도 별도 과제다.
