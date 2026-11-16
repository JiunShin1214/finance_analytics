# Dan · H4 작업 안내

**지금 볼 결과: H4 특징 단독 Rank IC — 원값 / 월별 1·99% 윈저라이징 비교.**

## 폴더 지도

```text
src/team/dan/                    개인 분석 코드 (공용 모듈과 분리)
  README.md                     코드별 역할
  h4_epu.py                     팀 MOM + 전월 EPU 결합
  h4_feature_rank.py            현재 단독 특징 Rank IC 실행
  upstream_rank_snapshot.py     출처 해시가 있는 팀 계산 함수 사본
  h4_analysis.py                별도 패널 OLS·2단계 HAC 검정
  test_h4.py                    H4·Holm·합성 회귀 검증
  h4_standalone_rank.py          이전 OLS 예측 실험 (현재 주결과 아님)
  h4_rank_comparison.py          공통 실행기 입력 준비 (결합 모형 미검증)
notebook/dan/
  README.md                     이 안내
  h4_feature_rank.ipynb          현재 분석 실행·결과 확인
  h4_epu.ipynb                   별도 OLS·HAC 실행
  docs/
    RESULTS.md                  세 실험의 결과와 조건 구분
    CHECKLIST.md                준비 완료 / 추가 확인할 항목
    INFERENCE_DETAILS.md        이전 OLS·HAC 상세 설명 보존
outputs/model_results/dan/       실행별 결과 ZIP
outputs/run_id/                  개인 백업·점검 기록 (Git 제외)
data/raw/                       입력 자료 (Git 제외)
```

## 실행

저장소 루트에서 아래 명령을 실행한다. 실행할 때마다 새로운 결과 폴더를 만든다.

```powershell
.\.venv\Scripts\python.exe -m src.team.dan.h4_feature_rank
```

입력은 `data/raw/mom.parquet`, `data/raw/panel.parquet`, `data/raw/Korea_Overall_EPU.xlsx`다.
MOM을 개인 코드로 재계산하지 않는다. 다음 달 목표는 달력월 키로 연결한다.

## 최신 결과

`outputs/model_results/dan/20261116T153929_h4_feature_rank_raw-winsor/result.zip`

- 평가: 목표월 2001~2021년, 252개월 중 251개월 산출, 167,089 종목-월.
- 원값 평균 IC 0.025527, 특징 윈저라이징 평균 IC 0.025531.
- 2001-01은 유효 MOM 표본이 없어 미산출. 0으로 대체하지 않는다.
- 윈저 경계는 목표수익률 결합 전에 적격·가용 특징 집합에서 계산한다.
- 실제 수익률은 원값 유지. 동전주 추가 제외 없음. OLS 학습·Expanding Window 없음.
- H3의 직접 특징 분석을 따른 탐색 결과이며 공식 model_test PASSED 결과가 아니다.

**해석 주의:** 양수 월별 EPU를 곱해도 종목 간 MOM 순위는 유지된다. 이 IC는 EPU의 추가 효과를 입증하지 않는다. H4의 관계 약화 가설은 별도 상호작용·2단계 검정과 구분해서 보고한다.

## 보존·공유 상태

루트의 `check_*.py`, `test_h4.py`, 옛 CSV·TXT는 `outputs/run_id/dan_cleanup_20261116/originals/root__*`에 보존 이동했다.
기존 공용 `src/features/h1.py`, `h4.py` 수정은 그대로 보존했다. 되돌리거나 커밋하지 않았다.
기존 분석 코드 경로와 세 ZIP은 재현성과 이전 참조 링크를 위해 유지했다.
코드·노트북은 개인 폴더만, 결과는 검수한 ZIP만 선택해 공유한다. data/·백업·개별 CSV는 공유 커밋에서 제외한다.
최종 검수 후 개인 코드·문서와 대표 결과 ZIP 2개를 명시 경로로 커밋 준비 목록에 반영했다. 공용 수정·data/·백업·이전 중복 실험 ZIP은 제외했다. 커밋·push하지 않았다.

[결과 비교](docs/RESULTS.md) · [추가 확인 항목](docs/CHECKLIST.md)

## EPU 정의·모델 결합 검토 업데이트

사용자 제공 새 H4 정의: Cho–Kim(2023) 뉴스 기반 한국 EPU(t-1) × 원 11개월 MOM. 기사 직접 수집은 하지 않는다. 로컬 엑셀의 출처 메모도 확인했다.

`h4_export.py`는 모델용 세 변수를 보존한 `data/processed/dan_h4_epu.parquet`를 생성한다(Git 제외). `h4_rank_comparison.py --prepare-only`는 최신 공통 API에 상대경로·transform=none으로 입력 준비만 한다. 모형 적합·공식 결과 저장은 미완료다.

상세 검토: `notebook/dan/docs/MODEL_REVIEW.md`. 개인 분석 코드는 선택 커밋 대상이지만 바로 모델 결합 검증 완료로 간주하지 않는다.

[한 장 결과 요약](docs/SUMMARY.md) · [커밋·모델 결합 검토](docs/MODEL_REVIEW.md)
