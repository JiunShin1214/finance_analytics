"""Exact selected upstream functions; original common modules unchanged.
Source: JiunShin1214/finance_analytics main
rank_ic blob 66669c81302b06f3392ab1606c93bd9afee5d0a5; panel blob 16c1bbaf05d8e9f56e3abbb3337c6a439dfa1174
"""

from __future__ import annotations

from typing import Sequence

import re

import numpy as np

import pandas as pd

from pandas import DataFrame

GROUP = ["model", "candidate", "fold", "target_month"]

IC_COLUMNS = [*GROUP, "n", "ic", "reason"]


def normalize_stock_id(values: Sequence[str]) -> Sequence[str]:
    """종목코드 문자열 열 -> 앞자리 0 보존 6자리 문자열 열을 반환한다.

    문자 포함 코드를 보존하며 숫자형으로 소실된 식별자를 복구하지 않는다.
    표시용 이름이나 DART 8자리 번호를 종목코드로 대체하지 않는다.
    """
    result = list(values)
    if any(
        not isinstance(v, str) or len(v) != 6 or v.strip() != v or not v.isalnum()
        for v in result
    ):
        raise ValueError("stock_id must be a six-character alphanumeric string")
    return result


def normalize_month(values: Sequence[str]) -> Sequence[str]:
    """월 문자열 열 -> 달력상 유효 YYYY-MM 열을 반환한다.

    YYYY-MM 문자열을 그대로 보존한다. 월 누락/순서 검사는 입력 계약에서 한다.
    """
    result = list(values)
    if any(
        not isinstance(v, str)
        or re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", v) is None
        or v.startswith("0000-")
        for v in result
    ):
        raise ValueError("month must be a valid YYYY-MM calendar month")
    return result


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
    if (
        "target_month" in features
        and not (features.target_month.to_numpy() == target).all()
    ):
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
    y = returns[["stock_id", "month", "return"]].rename(
        columns={"month": "target_month", "return": "y"}
    )
    out = out.merge(
        y, on=["stock_id", "target_month"], how="left", validate="one_to_one"
    )
    if (out.y.notna() & (~np.isfinite(out.y) | out.y.lt(-1))).any():
        raise ValueError("invalid_target_return")
    out["target_exclusion_reason"] = np.where(out.y.isna(), "target_missing", "")
    return out


def newey_west_mean_test(values) -> dict:
    """동일 간격·결측 없는 시계열 평균의 Bartlett HAC 양측 t 검정.

    L=floor(4*(T/100)^(2/9)), gamma_l=sum(u[t]*u[t-l])/T.
    Var(mean)=(gamma_0+2*sum((1-l/(L+1))*gamma_l))/T.
    추가 T/(T-k) 소표본 배율 없음. p·CI는 모두 t(T-1).
    결측 제거·시간 정렬은 하지 않고 호출부에서 검증한 순서를 사용한다.
    """
    from scipy.stats import t as student_t

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("hac_requires_finite_ordered_series_at_least_two")
    n = len(x)
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    if lag >= n:
        raise ValueError("hac_lag_exceeds_series")
    mean = float(x.mean())
    u = x - mean
    sd = float(x.std(ddof=1))
    meat = float(u @ u)
    for ell in range(1, lag + 1):
        meat += 2 * (1 - ell / (lag + 1)) * float(u[ell:] @ u[:-ell])
    if meat < -1e-12 * max(1.0, float(u @ u)):
        raise ValueError("negative_hac_variance")
    se = float(np.sqrt(max(0.0, meat)) / n)
    statistic = mean / se if se > 0 else np.nan
    p = float(2 * student_t.sf(abs(statistic), df=n - 1)) if se > 0 else np.nan
    critical = float(student_t.ppf(0.975, df=n - 1))
    low = mean - critical * se if se > 0 else np.nan
    high = mean + critical * se if se > 0 else np.nan
    usual_se = sd / np.sqrt(n)
    usual_t = mean / usual_se if usual_se > 0 else np.nan
    usual_p = (
        float(2 * student_t.sf(abs(usual_t), df=n - 1)) if usual_se > 0 else np.nan
    )
    if se > 0 and bool(p < 0.05) != bool(low > 0 or high < 0):
        raise ValueError("hac_p_ci_disagree")
    return {
        "valid_months": n,
        "mean": mean,
        "sample_sd": sd,
        "hac_se": se,
        "statistic": statistic,
        "p_two_sided": p,
        "ci95_low": low,
        "ci95_high": high,
        "lag": lag,
        "df": n - 1,
        "reference_distribution": "t",
        "kernel": "Bartlett",
        "small_sample_correction": False,
        "significant_raw_05": bool(p < 0.05) if np.isfinite(p) else False,
        "usual_se": usual_se,
        "usual_t": usual_t,
        "usual_p_two_sided": usual_p,
        "reason": "" if se > 0 else "zero_hac_standard_error",
    }


def rank_ic_for_month(
    month_predictions: DataFrame,
    *,
    min_valid_stocks: int = 3,
    tie_method: str = "average",
    score_column: str = "yhat",
) -> DataFrame:
    """단일 그룹·목표월 -> IC_COLUMNS 한 행. score_column으로 특징도 평가한다.

    NaN 쌍은 제외한다. 무한값은 입력 오류로 그 월을 미산출하고 이유를 남긴다.
    최소 3쌍, 양쪽 비상수. 평균 동순위의 Pearson 상관으로 Spearman을 계산한다.
    """
    if min_valid_stocks != 3 or tie_method != "average":
        raise ValueError("rank_ic_rule_change")
    f = month_predictions
    if not len(f) or not set([*GROUP, "stock_id", score_column, "y"]).issubset(f):
        raise ValueError("rank_ic_columns_or_rows_missing")
    if f[GROUP].isna().any().any() or len(f[GROUP].drop_duplicates()) != 1:
        raise ValueError("mixed_or_null_ic_group")
    enforce_unique_stock_month(f, month_column="target_month")
    normalize_stock_id(f.stock_id.tolist())
    normalize_month(f.target_month.tolist())
    if not f.target_month.between("2001-01", "2021-12").all():
        raise PermissionError("training_target_month_only")
    if not all(pd.api.types.is_numeric_dtype(f[c]) for c in (score_column, "y")):
        raise ValueError("rank_ic_requires_numeric")
    v = f[[score_column, "y"]].to_numpy(dtype=float, na_value=np.nan)
    paired = v[np.isfinite(v).all(axis=1)]
    n = len(paired)
    if np.isinf(v).any():
        reason = "nonfinite_input"
    elif n < 3:
        reason = "too_few_pairs"
    elif np.unique(paired[:, 0]).size < 2:
        reason = "constant_score"
    elif np.unique(paired[:, 1]).size < 2:
        reason = "constant_target"
    else:
        reason = ""
    ic = np.nan
    if not reason:
        ranks = pd.DataFrame(paired).rank(method="average").to_numpy()
        ic = float(np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1])
    return pd.DataFrame(
        [{**f.iloc[0][GROUP].to_dict(), "n": n, "ic": ic, "reason": reason}],
        columns=IC_COLUMNS,
    )
