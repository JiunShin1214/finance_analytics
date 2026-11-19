# H4 실행 확인

- `src/h4/requirements.txt`의 패키지를 설치한다.
- README의 개발 패널·적격성·H1·MOM·EPU 입력을 준비한다.
- `data/processed/h4_epu.parquet`가 없으면 `python -m src.h4.preprocessing.export`로 생성한다.
- `python -m src.h4.evaluation.ridge --prepare-only`로 입력을 확인한다.
- `python -m unittest discover -s src/h4/tests -t . -v`로 검사한다.
- `python -m src.h4.evaluation.ridge`로 Ridge를 실행한다.
- `full`과 `common` 표본, 월별 IC 차이, 선택된 설정을 구분해서 확인한다.
- 2022~2025년 최종 테스트는 열거나 사용하지 않는다.
- 공유 결과는 `result.zip`이며 데이터·개별 예측·캐시는 포함하지 않는다.
