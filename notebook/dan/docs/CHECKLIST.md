# 추가로 챙길 것

## 현재 H4 단독 평가: 추가 데이터 불필요

- [x] 팀 mom.parquet·panel.parquet·EPU 엑셀 존재, 원본은 Git 제외.
- [x] 키 중복 검사, 공통 표본, 목표 결합 전 특징 윈저 경계, 독립 SciPy 대조 및 ZIP 무결성 검사 완료.
- [x] 원값·윈저라이징 결과 둘 다 보존, 미산출 2001-01 기록.
- [x] 코드·실험 조건·결과 위치 안내 정리.

## 보고·공유 전에 확인할 것

1. **EPU 출처와 실제 공표일:** 출처 URL·로컬 저자 표기를 확인했고, 사용자가 EPU 검증 완료를 확인했다. 현 입력과 전월 연결 규칙을 유지한다.
2. **H4 정의 변경 근거:** 사용자 제공 새 H4 정의(Cho–Kim 한국 EPU·전월 연결)를 개인 코드에 반영했다. 공용 README/features의 반영은 관리자 영역이며 이번에 변경하지 않았다.
3. **해석:** 원 EPU가 양수라 직접 EPU×MOM IC는 MOM의 순위 효과와 분리되지 않는다. 이 IC만으로 H4의 추가 효과나 가설 지지를 주장하지 않는다.
4. **MOM 입력의 생성 이력:** 제공자의 코드 버전·산식(정확한 11개 달력월)·eligible 기준·가격 조정 여부를 확인한다. 파일을 재사용했지만 원천 산출 전 과정을 이번에 재검증한 것은 아니다. 동전주 제외가 필요하면 기준월 가격으로 별도 민감도 분석한다.
5. **Holm:** 팀이 확정한 전체 가설군·검정 방향·p값을 받아야 한다. 현재 미보정 결과를 보정 완료로 쓰지 않는다.
6. **재현 환경:** 개인 환경에 statsmodels/openpyxl이 있지만 기존 공용 requirements에는 빠져 있다. 공용 파일은 수정하지 않았다. 아래 개인 버전 기록을 환경 준비에 사용한다.
7. **Git 검수:** 공용 h1.py/h4.py의 기존 수정은 계속 남아 있다. 최종 커밋에서는 제외하고, 기존 stage 목록도 갱신 검토해야 한다. 데이터·백업·개별 CSV는 제외하고 필요한 ZIP만 선택한다. 아직 커밋·push 안 함.

## 나중에 H1 대 H1+H4 공식 모델 비교를 할 때만 필요

- data/interim/panel.parquet
- data/interim/eligibility.parquet
- data/processed/h1_components.parquet
- data/processed/h3.parquet (구버전 공통 실행기는 필요; 2026-11-16 확인한 최신 main prepare_inputs는 직접 읽지 않고 H3 grid를 계산)
- GitHub main의 최신 공통 코드·설정 (로컬 model_test와 MODEL_GRID 없음)

위 입력은 현재 직접 특징 Rank IC에는 요구하지 않는다. 연결 초안은 미실행 상태이며 코드 수정·검증 후에만 공식 비교에 사용한다.

## 이번에 확인한 개인 환경

Python 3.12.10

- numpy==2.5.4
- pandas==3.1.0
- scipy==1.18.1
- scikit-learn==1.9.1
- statsmodels==0.15.0
- openpyxl==3.1.5
- pyarrow==26.0.1

최신 공통 API 확인: 현재 prepare_inputs의 직접 필수 파일은 panel·eligibility·h1_components 세 개다. 앞 목록의 h3.parquet는 구버전 조건이며 현재 필수로 요청하지 않는다. 상세한 결합·커밋 판정은 MODEL_REVIEW.md 참조.
