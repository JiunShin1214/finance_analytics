# H4 모델 구성

입력은 `data/processed/h4_epu.parquet`다. 생성 명령:

```powershell
python -m src.h4.preprocessing.export
```

기준월 t에 전월 EPU를 연결하고 원 MOM과 곱한다.
파일에는 종목·기준월 키, 원 MOM, EPU, 상호작용과 가용 여부를 저장하며 목표수익률은 넣지 않는다.

Ridge 기준모형은 Size·BM·MOM·sigma, 확장모형은 EPU·상호작용을 추가한다.
월별 EPU는 횡단면 상수이므로 EPU·상호작용을 월별 순위로 바꾸지 않는다.
기존 MOM 주효과를 사용하고 원 MOM을 중복 열로 추가하지 않는다.

두 모형에 동일한 학습 행·평가 행·Fold·목표값 변환을 적용한다.
입력 표준화는 내부 학습행에서만 계산하고, 설계행렬의 완전 선형종속을 검사한다.
학습은 2001년 시작, 외부 검증은 2006~2021년이며 최종 테스트는 차단한다.

실행과 결과 파일 설명은 [README](../../../src/h4/README.md)를 참조한다.
