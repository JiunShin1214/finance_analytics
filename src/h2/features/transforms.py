"""Shared model feature specification and unchanged monthly normalization."""
import numpy as np
import pandas as pd

FEATURES = {
    "policy_debt": [
        "policy",
        "debt",
        "policy_x_debt",
    ],
    "policy_debt_size": [
        "policy",
        "debt",
        "policy_x_debt",
        "log_size",
    ],
}

def monthly_standardize(df, raw_col):
    """
    특성월별 1%/99% Winsorization 후 Z-score.
    기존 평가 코드와 같은 정규화 방식.
    """
    values = pd.to_numeric(
        df[raw_col], errors="coerce"
    ).replace(
        [np.inf, -np.inf], np.nan
    )

    def f(s):
        if s.notna().sum() < 3:
            return pd.Series(np.nan, index=s.index)

        lo, hi = s.quantile([0.01, 0.99])

        v = s.clip(
            lower=lo,
            upper=hi,
        )

        std = v.std(ddof=0)

        if np.isfinite(std) and std > 0:
            return (v - v.mean()) / std

        return pd.Series(np.nan, index=s.index)

    return values.groupby(
        df["feature_month"]
    ).transform(f)
