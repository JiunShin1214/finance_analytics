# 교수 제공 개발용 wide CSV를 메모리 관측 패널로 읽는다. 저장/목표값 결합은 하지 않는다.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.contracts import (
    Config, DEVELOPMENT_MONTHS, NUMERIC_PANEL_UNITS, WIDE_PANEL_COLUMNS,
    WIDE_VARIABLE_COLUMNS,
)
from src.preprocessing.panel import (
    enforce_unique_stock_month, normalize_month, normalize_stock_id,
)


def _approved_sources(root: Path, config: Config, periods: Sequence[str]) -> list[tuple[str, str, Path]]:
    """전체 요청을 preflight한다. 내용 open 없이 기간·명시 경로·링크 우회를 검사한다."""
    if (not periods or len(set(periods)) != len(periods)
            or any(p not in DEVELOPMENT_MONTHS for p in periods)):
        raise PermissionError("only warmup_2000 and train_2001_2021 are allowed")
    manifest = config.get("PATHS", {}).get("wide_csv", {})
    if not isinstance(manifest, Mapping) or set(manifest) != set(DEVELOPMENT_MONTHS):
        raise PermissionError("unverified development input manifest")
    root = root.resolve()
    if not root.is_dir():
        raise PermissionError("path must be the project root, not an arbitrary CSV")
    sources = []
    for period, files in manifest.items():
        if not isinstance(files, Mapping) or set(files) != set(WIDE_VARIABLE_COLUMNS):
            raise PermissionError("unverified variable manifest")
        for variable in WIDE_VARIABLE_COLUMNS:
            declared = files[variable]
            expected = Path("data") / "raw" / f"{period}__{variable}.csv"
            if not isinstance(declared, str) or Path(declared) != expected:
                raise PermissionError(f"unapproved source for {period}/{variable}")
            candidate = root / expected
            # symlink/junction을 통한 다른 경로의 Test/불명 파일 접근도 거부한다.
            if candidate.resolve() != candidate:
                raise PermissionError("redirected input path is not allowed")
            if period in periods:
                sources.append((period, variable, candidate))
    return sources


def _read_wide_csv(source: Path, period: str, variable: str) -> tuple[DataFrame, list[str], dict]:
    # 호출자는 전체 manifest preflight가 끝난 load_panel_contract뿐이다.
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if (len(header) < 2 or header[0] != "month"
                or len(set(header)) != len(header)):
            raise ValueError(f"{source.name}: missing/duplicate month or stock columns")
        stocks = list(normalize_stock_id(header[1:]))
        rows = []
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"{source.name}: wrong field count at line {line}")
            rows.append(row)
    months = list(normalize_month([row[0] for row in rows]))
    if len(set(months)) != len(months):
        raise ValueError(f"{source.name}: duplicate months")
    start, end = DEVELOPMENT_MONTHS[period]
    expected = pd.period_range(start, end, freq="M").astype(str).tolist()
    if months != expected:
        raise ValueError(f"{source.name}: missing, unordered or out-of-scope months")
    raw = pd.DataFrame([row[1:] for row in rows], columns=stocks, index=months, dtype=object)
    blank = raw.eq("")
    field = WIDE_VARIABLE_COLUMNS[variable]
    if field in NUMERIC_PANEL_UNITS:
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        invalid = ~blank & numeric.isna()
        if invalid.to_numpy().any():
            i, j = np.argwhere(invalid.to_numpy())[0]
            raise ValueError(
                f"{source.name}: numeric_parse_failure month={months[i]} "
                f"stock_id={stocks[j]} token={raw.iat[i, j]!r}; not an empty cell"
            )
        values = numeric.to_numpy(dtype=float)
        if (~np.isfinite(values) & ~blank.to_numpy()).any():
            raise ValueError(f"{source.name}: nonfinite numeric value")
        table = pd.DataFrame(values, index=months, columns=stocks)
    else:
        # keep_default_na와 같은 암묵적 NA 사전을 쓰지 않는다. 'NA' 등 원문 보존.
        table = raw.mask(blank, np.nan)
    audit = {
        "source": source.name, "period": period, "variable": variable,
        "rows": len(months), "columns": len(header),
        "empty_cells": int(blank.to_numpy().sum()),
        "zero_cells": int(table.eq(0).to_numpy().sum()) if field in NUMERIC_PANEL_UNITS else None,
        "unit": NUMERIC_PANEL_UNITS.get(field, "text"),
    }
    return table, stocks, audit


def load_panel_contract(
    path: Path, *, column_mapping: Mapping[str, str | None], config: Config,
    periods: Sequence[str] = tuple(DEVELOPMENT_MONTHS),
) -> DataFrame:
    """프로젝트 루트 + 명시 wide manifest -> stock_id/month 관측 패널(메모리).

    path는 CSV 한 개가 아닌 프로젝트 루트다. periods는 개발기간만 허용한다.
    모든 요청 경로를 첫 파일 open 전에 검사한다. 목표/특징/표본은 만들지 않는다.
    eligible는 원천 부재로 미확인 결측이며 추론/학습 적격성을 인증하지 않는다.
    """
    sources = _approved_sources(Path(path), config, periods)
    expected_mapping = {field: variable for variable, field in WIDE_VARIABLE_COLUMNS.items()}
    expected_mapping.update(stock_id="stock_id", month="month", eligible=None)
    if dict(column_mapping) != expected_mapping:
        raise ValueError("panel column mapping does not match the declared wide variables")
    for field, unit in NUMERIC_PANEL_UNITS.items():
        if config.get("UNITS", {}).get(field) != unit:
            raise ValueError(f"unknown unit for {field}: expected {unit}")
    panels, audits = [], []
    reference_stocks = None
    for period in periods:
        data = {}
        months = None
        stocks = None
        for declared_period, variable, source in sources:
            if declared_period != period:
                continue
            wide, stocks, audit = _read_wide_csv(source, period, variable)
            if reference_stocks is None:
                reference_stocks = stocks
            if stocks != reference_stocks:
                raise ValueError(f"{source.name}: stock columns/order mismatch")
            months = wide.index.tolist()
            data[variable] = wide.to_numpy().reshape(-1)
            audits.append(audit)
        observed = pd.DataFrame(data)
        observed["stock_id"] = np.tile(stocks, len(months))
        observed["month"] = np.repeat(months, len(stocks))
        observed = normalize_columns_contract(observed, column_mapping=column_mapping)
        observed["eligible"] = pd.Series(pd.NA, index=observed.index, dtype="boolean")
        panels.append(observed.loc[:, WIDE_PANEL_COLUMNS])
    result = pd.concat(panels, ignore_index=True)
    validate_input_schema(result, contract_name="panel", config=config)
    result.attrs.update(input_audit=audits, units=dict(NUMERIC_PANEL_UNITS),
                        stock_columns=tuple(reference_stocks), eligibility_status="unverified")
    return result


def normalize_columns_contract(table: DataFrame, *, column_mapping: Mapping[str, str | None]) -> DataFrame:
    """열 대응표에 따라 입력 열을 표준 필드명으로 바꾼 복사본을 반환한다.

    값·자료형·행을 바꾸지 않는다. null 매핑은 생성할 수 없는 원천을 뜻한다.
    """
    mapping = {source: target for target, source in column_mapping.items() if source is not None}
    if len(mapping) != sum(v is not None for v in column_mapping.values()):
        raise ValueError("ambiguous column mapping")
    if not table.columns.is_unique or not set(mapping).issubset(table.columns):
        raise ValueError("missing or duplicated source columns")
    result = table.rename(columns=mapping).copy()
    if not result.columns.is_unique:
        raise ValueError("column mapping creates duplicate columns")
    return result


def validate_input_schema(table: DataFrame, *, contract_name: str, config: Config) -> None:
    """현재는 개발용 관측 panel만 검증. 다른 입력/PIT 계약은 미구현이다."""
    if contract_name != "panel":
        raise NotImplementedError("only the development panel schema is implemented")
    if not table.columns.is_unique or not set(WIDE_PANEL_COLUMNS).issubset(table.columns):
        raise ValueError("missing or duplicated panel columns")
    enforce_unique_stock_month(table, month_column="month")
    normalize_stock_id(table["stock_id"].tolist())
    months = normalize_month(table["month"].tolist())
    if any(not "2000-01" <= month <= "2021-12" for month in months):
        raise ValueError("panel contains non-development months")
    for field, unit in NUMERIC_PANEL_UNITS.items():
        if config.get("UNITS", {}).get(field) != unit:
            raise ValueError(f"unknown unit for {field}")
        if (not pd.api.types.is_numeric_dtype(table[field])
                or pd.api.types.is_bool_dtype(table[field])):
            raise ValueError(f"{field}: numeric dtype required")
        if not np.isfinite(table[field].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{field}: nonfinite numeric value")
    for field in ("company_name", "isin"):
        if any(not isinstance(v, str) for v in table[field].dropna()):
            raise ValueError(f"{field}: text values required")
    if str(table["eligible"].dtype) != "boolean" or not table["eligible"].isna().all():
        raise ValueError("raw wide files do not certify historical eligibility")


def write_output_contract(table: DataFrame, path: Path, *, contract_name: str, config: Config) -> None:
    """표와 출력 경로를 받아 계약에 맞는 CSV 또는 Parquet 파일로 저장한다.

    TODO: 필수 열, 유일키, 품질 검사 결과를 확인하고 저장한다.
    CSV는 UTF-8을 사용한다. 원본 및 기존 산출물의 충돌은 오류로 처리한다.
    """
    raise NotImplementedError
