# -*- coding: utf-8 -*-
"""H2: annual expanding-window TRAIN-only Rank IC comparison (2006-2021).

Compare OLS with DataGuide 6m debt and DART/DataGuide hybrid debt on the
same stock-month observations. Historical point-in-time values are NOT
fully verified. This does not access 2022-2025 targets.

Run from the project root:
    python -m src.h2.evaluation.evaluate_rankic_yearly
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.h2.data_io import ROOT, PROCESSED, sha256
from src.h2.evaluation.evaluate_rankic import calc_ic, norm_code, read_csv, resolve
from src.h2.features.transforms import FEATURES, monthly_standardize
from src.h2.models.estimators import model_instance

FOLDS = [(f"{y}-01", f"{y}-12") for y in range(2006, 2022)]
KEYS = ["stock_id", "feature_month"]
FEATURE_COLS = FEATURES["policy_debt_size"]
SOURCES = {"DART_FIRST_FILED_CANDIDATE", "DATAGUIDE_6M_PROXY"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=ROOT)
    p.add_argument("--panel", type=Path, default=PROCESSED / "dataset.csv")
    p.add_argument(
        "--lag-6m", type=Path,
        default=PROCESSED / "debt/h2_debt_lag_6m_fiscal_screened_2001_2021.csv",
    )
    p.add_argument(
        "--hybrid", type=Path,
        default=PROCESSED / "debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv",
    )
    p.add_argument("--out", type=Path, default=Path("results/rankic_yearly_2006_2021"))
    p.add_argument("--min-stocks", type=int, default=20)
    return p.parse_args()


def check_keys(df: pd.DataFrame, label: str):
    df["stock_id"] = norm_code(df["stock_id"])
    df["feature_month"] = df["feature_month"].astype("string").str[:7]
    if df[KEYS].isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError(f"{label}: 누락 또는 중복된 종목-특성월 키")
    if not df["feature_month"].str.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", na=False).all():
        raise ValueError(f"{label}: 잘못된 특성월 형식")


def assert_timing(df: pd.DataFrame, label: str, value_col: str,
                  fiscal_col: str, available_col: str):
    values = pd.to_numeric(df[value_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    pred = pd.to_datetime(df["prediction_date"], errors="coerce")
    expected = pd.to_datetime(df["feature_month"] + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
    fiscal = pd.to_datetime(df[fiscal_col], errors="coerce")
    available = pd.to_datetime(df[available_col], errors="coerce")
    good_value = values.notna()
    wrong = good_value & (
        pred.isna() | expected.isna() | pred.ne(expected)
        | fiscal.isna() | available.isna()
        | fiscal.gt(pred) | available.gt(pred)
    )
    if wrong.any():
        raise ValueError(f"{label}: 사용가능일/회계기준일 위반 {int(wrong.sum())}행")
    return values


def load_panel(path: Path):
    p = read_csv(path)
    required = {
        "stock_id", "feature_month", "target_month", "return_t1",
        "market_cap", "final_rule_v2_sensitivity",
    }
    if required - set(p.columns):
        raise ValueError(f"H2 패널 필수 컬럼 누락: {sorted(required - set(p.columns))}")
    check_keys(p, "H2 패널")
    p["target_month"] = p["target_month"].astype("string").str[:7]
    if p["target_month"].isna().any():
        raise ValueError("목표월 누락")
    expected = (pd.PeriodIndex(p["feature_month"], freq="M") + 1).astype(str)
    if not np.array_equal(p["target_month"].to_numpy(), expected.to_numpy()):
        raise ValueError("feature_month와 target_month가 한 달 차이가 아닙니다")

    # 미래 Test 목표수익률 사용 금지
    p = p.loc[p["target_month"].between("2001-01", "2021-12")].copy()
    if p.empty:
        raise ValueError("2001-2021년 Train 목표월 자료가 없습니다")
    p["return_t1"] = pd.to_numeric(p["return_t1"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    p["policy"] = pd.to_numeric(p["final_rule_v2_sensitivity"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    cap = pd.to_numeric(p["market_cap"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    p["log_size"] = np.log(cap.where(cap > 0))
    return p


def load_six_month(path: Path):
    d = read_csv(path)
    required = set(KEYS + ["proxy_masked", "prediction_date", "fiscal_date", "proxy_available_date"])
    if required - set(d.columns):
        raise ValueError(f"6m 컬럼 누락: {sorted(required - set(d.columns))}")
    check_keys(d, "6m")
    d["debt_6m"] = assert_timing(
        d, "6m", "proxy_masked", "fiscal_date", "proxy_available_date"
    )
    return d[KEYS + ["debt_6m"]]


def load_hybrid(path: Path):
    d = read_csv(path)
    required = set(KEYS + [
        "debt_final_raw", "prediction_date", "source_fiscal_date",
        "source_available_date", "source_type",
    ])
    if required - set(d.columns):
        raise ValueError(f"hybrid 컬럼 누락: {sorted(required - set(d.columns))}")
    check_keys(d, "hybrid")
    d["debt_hybrid"] = assert_timing(
        d, "hybrid", "debt_final_raw", "source_fiscal_date", "source_available_date"
    )
    wrong_source = d["debt_hybrid"].notna() & ~d["source_type"].isin(SOURCES)
    if wrong_source.any():
        raise ValueError(f"hybrid 출처 값 오류: {int(wrong_source.sum())}행")
    return d[KEYS + ["debt_hybrid", "source_type"]]


def main():
    a = arguments()
    root = a.project.resolve()
    paths = {
        "panel": resolve(root, a.panel),
        "six_month": resolve(root, a.lag_6m),
        "hybrid": resolve(root, a.hybrid),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} 파일 없음: {path}")

    p = load_panel(paths["panel"])
    p = p.merge(load_six_month(paths["six_month"]), on=KEYS, how="left", validate="one_to_one")
    p = p.merge(load_hybrid(paths["hybrid"]), on=KEYS, how="left", validate="one_to_one")

    # 기존 평가와 같은 특성월 기준 Winsor(1/99%) + Z-score
    for label in ("6m", "hybrid"):
        p[f"z_{label}"] = monthly_standardize(p, f"debt_{label}")

    common = p[["return_t1", "policy", "log_size", "z_6m", "z_hybrid"]].notna().all(axis=1)
    if "eligible" in p.columns:
        common &= p["eligible"].astype("string").str.lower().isin(["true", "1"])
    d = p.loc[common].copy()
    if d.empty:
        raise ValueError("공통 표본이 없습니다. 입력 데이터와 유효 월을 확인하세요")

    print(f"전체 Train 패널: {len(p):,}행")
    print(f"6m/Hybrid 두 시나리오 공통 표본: {len(d):,}행")
    print("검증: 2006-2021년 16-Fold / 연도별 12개월")
    print("알고리즘: OLS / 변수: Policy, Debt, Policy×Debt, Log(Size)")

    monthly, folds = [], []
    for fold, (start, end) in enumerate(FOLDS, 1):
        train_year_end = int(start[:4]) - 1
        for scenario in ("6m", "hybrid"):
            x = d.copy()
            x["debt"] = x[f"z_{scenario}"]
            x["policy_x_debt"] = x["policy"] * x["debt"]
            tr = x.loc[x["target_month"] < start]
            va = x.loc[x["target_month"].between(start, end)]
            if len(tr) < 100 or len(va) < a.min_stocks:
                raise ValueError(f"표본 부족: fold={fold} {scenario} train={len(tr)}, val={len(va)}")

            model = model_instance("OLS")
            model.fit(tr[FEATURE_COLS], tr["return_t1"])
            predictions = model.predict(va[FEATURE_COLS])
            scores = calc_ic(va, predictions, a.min_stocks)
            if len(scores) != 12:
                actual = [row[0] for row in scores]
                raise ValueError(
                    f"{scenario} fold {fold}: 12개월 중 {len(scores)}개월 IC만 계산됨: {actual}"
                )

            for month, n_stocks, ic in scores:
                monthly.append({
                    "fold": fold, "scenario": scenario,
                    "target_month": month, "n_stocks": n_stocks, "rank_ic": ic,
                })
            mean_ic = float(np.mean([row[2] for row in scores]))
            folds.append({
                "fold": fold,
                "train_start": "2001-01",
                "train_end": f"{train_year_end}-12",
                "validation_start": start,
                "validation_end": end,
                "scenario": scenario,
                "n_train": int(len(tr)), "n_val": int(len(va)),
                "n_months": len(scores), "mean_ic": mean_ic,
            })
            print(f"Fold {fold:02d} | Train 2001-{train_year_end} | Val {start[:4]} | "
                  f"{scenario:6s} | IC {mean_ic:+.6f}", flush=True)

    m = pd.DataFrame(monthly)
    f = pd.DataFrame(folds)
    wide = m.pivot(index="target_month", columns="scenario", values="rank_ic").sort_index()
    if len(wide) != 192 or wide.isna().any().any():
        raise AssertionError("192개월의 양쪽 Rank IC가 모두 존재해야 합니다")

    delta = wide["hybrid"] - wide["6m"]
    tol = 1e-12
    paired = wide.reset_index()
    paired["hybrid_minus_6m"] = delta.to_numpy()
    paired["status"] = np.select(
        [delta.to_numpy() > tol, delta.to_numpy() < -tol],
        ["improved", "worse"], default="unchanged",
    )
    summary = {
        "evaluation": "TRAIN-only, annual expanding window",
        "train_start": "2001-01",
        "validation_start": "2006-01",
        "validation_end": "2021-12",
        "folds": 16,
        "paired_months": len(wide),
        "common_stock_month_rows": int(len(d)),
        "mean_ic_6m": float(wide["6m"].mean()),
        "mean_ic_hybrid": float(wide["hybrid"].mean()),
        "hybrid_minus_6m": float(delta.mean()),
        "improved_months": int((delta > tol).sum()),
        "worse_months": int((delta < -tol).sum()),
        "unchanged_months": int((delta.abs() <= tol).sum()),
        "not_fully_pit_verified": True,
        "notes": (
            "동일 6m/Hybrid 공통 표본만 비교. 기존 3-Fold 4시나리오 "
            "공통 표본 154215행과 표본/검증월이 다를 수 있어 IC 직접 비교 불가. "
            "2022-2025 Test 목표수익률 사용하지 않음."
        ),
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
    }

    out = resolve(root, a.out)
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "monthly_rank_ic.csv", index=False, encoding="utf-8-sig")
    f.to_csv(out / "fold_rank_ic.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "paired_monthly_ic.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[최종 192개월 평균 Rank IC]")
    print(f"6개월 지연: {summary['mean_ic_6m']:+.6f}")
    print(f"Hybrid    : {summary['mean_ic_hybrid']:+.6f}")
    print(f"Hybrid-6m : {summary['hybrid_minus_6m']:+.6f}")
    print(f"개선/악화/동일: {summary['improved_months']}/"
          f"{summary['worse_months']}/{summary['unchanged_months']}개월")
    print("결과 저장:", out)


if __name__ == "__main__":
    main()
