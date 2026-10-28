# 원본 패널을 읽고 열 이름을 공통 형식으로 맞추며 산출물을 저장하는 모듈이다.
# 입출력 스키마와 파일 접근 범위를 확인하는 함수 골격을 정의한다.
# 예상 산출물: 반환용 표준 입력표와 출력 계약에 맞춘 CSV·Parquet 파일.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping
from src.contracts import Config

if TYPE_CHECKING:
    from pandas import DataFrame


def load_panel_contract(path: Path, *, column_mapping: Mapping[str, str | None], config: Config) -> DataFrame:
    """패널 경로와 열 대응 설정을 받아 표준 PANEL_COLUMNS 표를 반환한다.

    TODO: 읽기 전에 파일의 선언된 기간을 확인하고 Test 포함·범위 불명 파일을 차단한다.
    종목코드 문자열, 달력월, 소수 수익률, 원 단위 시총과 적격 여부를 검사한다.
    열 대응·단위가 불명확하거나 필수 열이 없으면 오류를 발생시킨다.
    """
    raise NotImplementedError


def normalize_columns_contract(table: DataFrame, *, column_mapping: Mapping[str, str | None]) -> DataFrame:
    """열 대응표에 따라 입력 열을 표준 필드명으로 바꾼 복사본을 반환한다.

    0과 결측 보존, 단위 변환 근거를 별도 기록. 자동 % 추정 금지.
    """
    raise NotImplementedError


def validate_input_schema(table: DataFrame, *, contract_name: str, config: Config) -> None:
    """표와 panel/security_map/dart/documents 계약명 -> 검증한다.

    TODO: contracts.py의 필수 열/자료형/유일키·단위·시점 확인.
    계약명 불명, 코드 앞자리 소실, 중복/누락, 불명 단위는 오류 처리.
    documents는 H2/H4 전달용 메타데이터 계약이며 문서 메타데이터를 검사한다.
    정책 추출문은 POLICY_TEXT_COLUMNS, 뉴스 기사는 NEWS_ARTICLE_COLUMNS를 적용한다.
    H2/H4 특징은 H2_FEATURE_COLUMNS/H4_FEATURE_COLUMNS를 검사하고,
    통합 특징·표본은 활성 확장에 해당하는 EXTENDED_* 열을 추가 검사한다.
    """
    raise NotImplementedError


def write_output_contract(table: DataFrame, path: Path, *, contract_name: str, config: Config) -> None:
    """표와 출력 경로를 받아 계약에 맞는 CSV 또는 Parquet 파일로 저장한다.

    TODO: 필수 열, 유일키, 품질 검사 결과를 확인하고 저장한다.
    CSV는 UTF-8을 사용한다. 원본 및 기존 산출물의 충돌은 오류로 처리한다.
    """
    raise NotImplementedError
