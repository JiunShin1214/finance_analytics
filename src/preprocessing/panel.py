# 종목코드와 달력월을 정규화하고 중복·결측을 구분하는 모듈이다.
# 기준월 특징을 다음 달 목표수익률과 연결하는 함수 골격을 정의한다.
# 예상 산출물: 정규화 패널과 목표월·목표수익률 결합표(panel.parquet·samples.parquet 구성용).

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import re
if TYPE_CHECKING:
    from pandas import DataFrame


def normalize_stock_id(values: Sequence[str]) -> Sequence[str]:
    """종목코드 문자열 열 -> 앞자리 0 보존 6자리 문자열 열을 반환한다.

    문자 포함 코드를 보존하며 숫자형으로 소실된 식별자를 복구하지 않는다.
    표시용 이름이나 DART 8자리 번호를 종목코드로 대체하지 않는다.
    """
    result = list(values)
    if any(not isinstance(v, str) or len(v) != 6 or v.strip() != v
           or not v.isalnum() for v in result):
        raise ValueError("stock_id must be a six-character alphanumeric string")
    return result


def normalize_month(values: Sequence[str]) -> Sequence[str]:
    """월 문자열 열 -> 달력상 유효 YYYY-MM 열을 반환한다.

    YYYY-MM 문자열을 그대로 보존한다. 월 누락/순서 검사는 입력 계약에서 한다.
    """
    result = list(values)
    if any(not isinstance(v, str)
           or re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", v) is None
           or v.startswith("0000-") for v in result):
        raise ValueError("month must be a valid YYYY-MM calendar month")
    return result


def distinguish_zero_vs_missing(table: DataFrame, *, status_columns: Sequence[str]) -> DataFrame:
    """입력 표/상태열명 -> 원값 및 결측 사유가 보존된 복사 표를 반환한다.

    TODO: 실제 0, 자료 부재, 수집/추출/연결 실패, 미상장 기간 분리.
    오류: 상태 누락·모순은 기록/격리. 결측을 0/전체기간 평균으로 채우지 않음.
    """
    raise NotImplementedError


def enforce_unique_stock_month(table: DataFrame, *, month_column: str) -> None:
    """표/관측월 또는 기준월 열명 -> 유일성 확인한다.

    관측월/기준월/목표월은 호출자가 지정한다. 중복을 자동 삭제하지 않는다.
    """
    keys = ["stock_id", month_column]
    if not table.columns.is_unique or not set(keys).issubset(table.columns):
        raise ValueError("missing or duplicated stock/month columns")
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError("null or duplicate stock/month keys")


def attach_target_month(features: DataFrame) -> DataFrame:
    """다음 달력월을 붙인다. 개발 목표월 밖의 행은 자동 삭제하지 않고 차단한다."""
    import pandas as pd
    enforce_unique_stock_month(features, month_column="base_month")
    normalize_stock_id(features.stock_id.tolist())
    normalize_month(features.base_month.tolist())
    target = (pd.PeriodIndex(features.base_month, freq="M") + 1).astype(str)
    if not pd.Series(target).between("2001-01", "2021-12").all():
        raise PermissionError("training_target_month_only: 2001-01..2021-12")
    if "target_month" in features and not (features.target_month.to_numpy() == target).all():
        raise ValueError("target_must_be_next_calendar_month")
    out = features.copy()
    out["target_month"] = target
    return out


def join_training_targets(features: DataFrame, returns: DataFrame) -> DataFrame:
    """완성된 특징에 제공 개발 수익률을 키로 왼쪽 결합. 결측 목표 행도 보존한다."""
    import numpy as np
    import pandas as pd
    out = attach_target_month(features)
    enforce_unique_stock_month(returns, month_column="month")
    normalize_stock_id(returns.stock_id.tolist())
    normalize_month(returns.month.tolist())
    if not returns.month.between("2000-01", "2021-12").all():
        raise PermissionError("development_returns_only")
    if returns.attrs.get("units", {}).get("return") != "decimal":
        raise ValueError("return_unit_must_be_decimal")
    if "y" in out:
        raise ValueError("target_already_attached")
    if not pd.api.types.is_numeric_dtype(returns["return"]):
        raise ValueError("return_must_be_numeric")
    y = returns[["stock_id", "month", "return"]].rename(columns={"month":"target_month", "return":"y"})
    out = out.merge(y, on=["stock_id", "target_month"], how="left", validate="one_to_one")
    if (out.y.notna() & (~np.isfinite(out.y) | out.y.lt(-1))).any():
        raise ValueError("invalid_target_return")
    out["target_exclusion_reason"] = np.where(out.y.isna(), "target_missing", "")
    return out
