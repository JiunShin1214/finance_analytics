"""한국 EPU(Cho–Kim, 2023)와 팀 11개월 MOM으로 H4를 구성한다.

기준월 t에는 EPU(t-1)를 연결한다. 과거 실제 공표일/빈티지는 별도 검증 대상이다.
기사를 직접 수집하거나 기사 비율을 계산하지 않는다.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def load_epu_index(file_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(
        file_path,
        sheet_name="Korean New EPU Index",
        usecols="A:B",
        engine="openpyxl",
    )

    df = df.rename(
        columns={
            "Date": "epu_month",
            "News-based EPU Index": "epu_raw",
        }
    )

    df["epu_month"] = (
        pd.to_datetime(df["epu_month"], errors="raise")
        .dt.to_period("M")
    )

    df["epu_raw"] = pd.to_numeric(
        df["epu_raw"], errors="raise"
    )

    if df["epu_month"].duplicated().any():
        raise ValueError("EPU 월이 중복되어 있습니다.")

    if not np.isfinite(df["epu_raw"]).all():
        raise ValueError("EPU 값에 결측이 있습니다.")

    return df[["epu_month", "epu_raw"]].sort_values("epu_month")


def align_epu_to_base_month(epu: pd.DataFrame) -> pd.DataFrame:
    result = epu.copy()
    result["epu_month"] = pd.PeriodIndex(result["epu_month"], freq="M")
    if result["epu_month"].duplicated().any() or result["epu_month"].isna().any():
        raise ValueError("EPU 월은 결측 없이 유일해야 합니다.")

    result["base_month"] = (
        result["epu_month"] + 1
    ).astype(str)

    result["epu_month"] = result["epu_month"].astype(str)

    return result[
        ["base_month", "epu_month", "epu_raw"]
    ]


def calculate_h4_epu_features(
    momentum: pd.DataFrame,
    base_keys: pd.DataFrame,
    monthly_epu: pd.DataFrame,
) -> pd.DataFrame:
    keys = base_keys[["stock_id", "base_month"]].copy()
    mom = momentum[["stock_id", "base_month", "mom"]].copy()

    keys["base_month"] = keys["base_month"].astype(str)
    mom["base_month"] = mom["base_month"].astype(str)
    monthly_epu = monthly_epu.copy()
    monthly_epu["base_month"] = monthly_epu["base_month"].astype(str)
    for table in (keys, mom, monthly_epu):
        if table["base_month"].isna().any() or not table["base_month"].str.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])").all():
            raise ValueError("기준월은 유효한 YYYY-MM이어야 합니다.")
    if keys.isna().any().any() or mom[["stock_id", "base_month"]].isna().any().any():
        raise ValueError("종목-월 키에 결측이 있습니다.")
    if monthly_epu["base_month"].duplicated().any():
        raise ValueError("기준월별 EPU가 중복되었습니다.")
    source = pd.PeriodIndex(monthly_epu["epu_month"], freq="M")
    if source.isna().any() or not ((source + 1).astype(str) == monthly_epu["base_month"].to_numpy()).all():
        raise ValueError("기준월에는 정확히 전월 EPU를 연결해야 합니다.")
    for table, col in ((mom, "mom"), (monthly_epu, "epu_raw")):
        table[col] = pd.to_numeric(table[col], errors="raise")
        if np.isinf(table[col].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("MOM/EPU에 무한값이 있습니다.")

    if keys.duplicated(["stock_id", "base_month"]).any():
        raise ValueError("기준 종목-월 키가 중복되었습니다.")

    if mom.duplicated(["stock_id", "base_month"]).any():
        raise ValueError("MOM 종목-월 키가 중복되었습니다.")

    result = keys.merge(
        mom,
        on=["stock_id", "base_month"],
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        monthly_epu,
        on="base_month",
        how="left",
        validate="many_to_one",
    )

    result["h4_interaction"] = (
        result["epu_raw"] * result["mom"]
    )
    if np.isinf(result["h4_interaction"].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("상호작용 계산이 유한 범위를 벗어났습니다.")

    result["h4_available"] = (
        result["epu_raw"].notna()
        & result["mom"].notna()
    )

    result["h4_exclusion_reason"] = None
    result.loc[
        result["epu_raw"].isna(),
        "h4_exclusion_reason",
    ] = "missing_epu"

    result.loc[
        result["mom"].isna() & result["epu_raw"].notna(),
        "h4_exclusion_reason",
    ] = "missing_momentum"

    return result
