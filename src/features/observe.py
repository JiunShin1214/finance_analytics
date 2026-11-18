"""데이터 관측: H1·H3 특징 가용성(2026-11-04 확정 항목, 2026-11-05 목표월 연도·공유 표 기준으로 갱신).

- 입력은 공유 interim 표(panel·eligibility, `python -m src.pipeline --stage ingest`)와 자기자본 as-of다. 적격성을 다시 계산하지 않는다.
- 대상 기준월은 2000-12~2021-11(목표월 2001-01~2021-12), 연도는 목표월(차월 수익률) 연도로 묶는다(노션 10주차).
- 분모는 그 기준월의 적격 종목. 수익률은 제공 CSV만 쓴다(코스닥→KOSPI 이전 종목은 별개 기업, 교수님 지침).
- 월 유효 판정을 둘로 나눈다.
  H1 산출 가능 월: 공통 유효 종목 2개 이상 + 규모·B/M·모멘텀 각각 표본표준편차 > 0.
  IC 계산 가능 월: 공통 유효 종목 중 차월 수익률이 있는 쌍 3개 이상 + 차월 수익률이 상수가 아님.
  H1 점수 자체의 변동 확인은 H1 점수 계산(H1-F) 뒤로 미룬다. 특징 원값은 표준편차 확인에만 쓰고 저장하지 않는다.
- 특징과 차월 수익률의 관계(상관·IC)는 계산하지 않는다. 차월 수익률은 있음/없음과 상수 여부만 본다.

실행: .\\.venv\\Scripts\\python.exe -B -X utf8 -m src.features.observe
결과: outputs/데이터관측/{H1,H3}/<시각>/ (덮어쓰지 않음)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OUT = Path("outputs/데이터관측")
BASE_FIRST, BASE_LAST = "2000-12", "2021-11"  # 목표월 2001-01~2021-12
MONTHS = pd.period_range("2000-01", "2021-12", freq="M").astype(str).tolist()
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # dataviz 기준 팔레트 1~4번(고정 순서)
INK, MUTED = "#0b0b0b", "#52514e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_month(base: pd.Series) -> pd.Series:
    """기준월 'YYYY-MM' -> 정확히 다음 달력월(행 shift가 아님)."""
    return (pd.PeriodIndex(base, freq="M") + 1).astype(str).to_series(index=base.index)


def month_validity(frame: pd.DataFrame, member: str, values: list[str], *, n_min: int) -> pd.DataFrame:
    """기준월별 member 종목 수와 values 각각의 표본표준편차(ddof=1)로 유효 여부를 판정한다."""
    rows = []
    for month, part in frame.groupby("base_month", sort=True):
        sel = part[part[member]]
        sds = {v: float(sel[v].std(ddof=1)) if len(sel) >= 2 else np.nan for v in values}
        n = len(sel)
        if n < n_min:
            reason = "too_few_stocks"
        elif any(not np.isfinite(s) or s <= 0 for s in sds.values()):
            reason = "zero_or_nonfinite_sd"
        else:
            reason = ""
        rows.append({"base_month": month, "n": n, **{f"sd_{v}": s for v, s in sds.items()}, "valid": reason == "",
                     "reason": reason})
    return pd.DataFrame(rows)


def ic_validity(frame: pd.DataFrame, member: str) -> pd.DataFrame:
    """기준월별 member 종목 중 차월 수익률이 있는 쌍 수와 IC 계산 가능 여부(쌍 3개 이상, 수익률 비상수)."""
    rows = []
    for month, part in frame.groupby("base_month", sort=True):
        y = part.loc[part[member], "y"].dropna()
        reason = "too_few_pairs" if len(y) < 3 else ("constant_target" if y.nunique() < 2 else "")
        rows.append({"base_month": month, "pairs": len(y), "ic_valid": reason == "", "ic_reason": reason})
    return pd.DataFrame(rows)


# ---------- 입력 ----------

def load_inputs(root: Path, config: dict | None = None) -> dict:
    if config is None:
        config = yaml.safe_load((root / "config/experiment.yaml").read_text(encoding="utf-8"))
    paths = {k: Path(config["PATHS"][k]) for k in ("panel", "eligibility", "asof")}
    panel = pd.read_parquet(root / paths["panel"])
    elig = pd.read_parquet(root / paths["eligibility"])
    asof = pd.read_parquet(root / paths["asof"], columns=["stock_code", "base_month", "equity", "usable", "reason"]).rename(
        columns={"stock_code": "stock_id"})
    if len(panel) != len(elig) or panel.duplicated(["stock_id", "month"]).any():
        raise ValueError("interim_tables_inconsistent")
    return dict(panel=panel, elig=elig, asof=asof, paths=paths,
                hashes={p.as_posix(): sha256(root / p) for p in paths.values()})


def base_frame(x: dict) -> pd.DataFrame:
    """기준월 2000-12~2021-11 종목×월: 목표월·연도, 적격성, 시가총액, 차월 수익률(y)."""
    p = x["panel"]
    f = p[["stock_id", "month", "market_cap", "eligible"]].rename(columns={"month": "base_month"})
    f = f[f.base_month.between(BASE_FIRST, BASE_LAST)].reset_index(drop=True)
    f["target_month"] = target_month(f.base_month)
    f["year"] = f.target_month.str[:4]
    y = p[["stock_id", "month", "return"]].rename(columns={"month": "target_month", "return": "y"})
    f = f.merge(y, on=["stock_id", "target_month"], how="left", validate="one_to_one")
    f["eligible"] = f.eligible.fillna(False).astype(bool)
    return f


def momentum_status(x: dict) -> pd.DataFrame:
    """기준월별 t-11~t-1 유효 달 수, 모멘텀 원값(표준편차 확인용), 미산출 사유."""
    p, e = x["panel"], x["elig"]
    wide = p.pivot(index="month", columns="stock_id", values="return").reindex(MONTHS)
    valid = wide.notna()
    count = valid.astype(int).rolling(11, min_periods=11).sum().shift(1)
    logsum = np.log1p(wide.where(valid, 0.0)).rolling(11, min_periods=11).sum().shift(1)
    before = (e.eligibility_basis.str.endswith(":before_listing")
              ).to_frame("b").assign(stock_id=e.stock_id, month=e.month).pivot(index="month", columns="stock_id", values="b")
    pre = (before.reindex(MONTHS).fillna(False).astype(bool) & ~valid).astype(int).rolling(11, min_periods=11).sum().shift(1)
    long = pd.DataFrame({"valid_in_window": count.stack(future_stack=True), "logsum": logsum.stack(future_stack=True),
                         "pre_listing_gaps": pre.reindex(columns=count.columns).stack(future_stack=True)})
    long = long.reset_index().rename(columns={"month": "base_month"})
    v, missing = long.valid_in_window, 11 - long.valid_in_window
    long["mom_available"] = v.eq(11)
    long["mom"] = np.where(long.mom_available, np.expm1(long.logsum), np.nan)
    long["mom_reason"] = np.select(
        [v.eq(11), v.isna(), long.pre_listing_gaps.gt(0), missing.eq(1), missing.between(2, 5)],
        ["", "window_before_2000", "window_includes_pre_listing", "missing_1", "missing_2_5"], "missing_6_plus")
    return long[["stock_id", "base_month", "valid_in_window", "mom", "mom_available", "mom_reason"]]


def h1_status(f: pd.DataFrame, x: dict, mom: pd.DataFrame) -> pd.DataFrame:
    g = f.merge(x["asof"], on=["stock_id", "base_month"], how="left", validate="one_to_one")
    g = g.merge(mom, on=["stock_id", "base_month"], how="left", validate="one_to_one")
    g["size_available"] = g.market_cap.gt(0)
    g["size"] = np.where(g.size_available, np.log(g.market_cap.where(g.size_available)), np.nan)
    g["size_reason"] = np.where(g.size_available, "", np.where(g.market_cap.isna(), "market_cap_missing", "market_cap_nonpositive"))
    usable = g.usable.fillna(False).astype(bool)
    g["bm_available"] = g.size_available & usable & g.equity.gt(0)
    g["bm"] = np.where(g.bm_available, g.equity / g.market_cap, np.nan)
    g["bm_reason"] = np.where(g.bm_available, "", np.where(~usable, g.reason.fillna("no_asof_row"),
                                                         np.where(~g.size_available, "market_cap_missing", "other")))
    g["joint_available"] = g.size_available & g.bm_available & g.mom_available
    lost = (~g.size_available).astype(int) + (~g.bm_available).astype(int) + (~g.mom_available).astype(int)
    g["joint_reason"] = np.where(g.joint_available, "", np.where(
        lost.eq(1), np.where(~g.size_available, "only_size", np.where(~g.bm_available, "only_bm", "only_mom")), "two_or_more"))
    g["joint_member"] = g.eligible & g.joint_available
    return g


def h3_status(f: pd.DataFrame, x: dict) -> pd.DataFrame:
    """H3 엔진의 누적 계산(_inputs·_state_history)과 엔진과 같은 산출 판정식. 표준화 단계(_score)는 전체 자료에서
    수십 분이 걸려 쓰지 않는다(2026-11-04 확인: 100종목 전 칸 판정 일치)."""
    from src.features.h3 import _inputs, _state_history
    p = x["panel"]
    stocks = sorted(p.stock_id.unique())
    keys = pd.DataFrame({"stock_id": np.repeat(stocks, len(MONTHS)), "base_month": np.tile(MONTHS, len(stocks))})
    r = p.loc[p["return"].notna(), ["stock_id", "month", "return"]].copy()
    r.attrs["units"] = {"return": "decimal"}
    ri, b, history = _inputs(r, keys, None)
    states = _state_history(ri, b, history)
    s = states[["stock_id", "base_month", "initialized", "recent_valid_count", "v"]]
    t = f.merge(s, on=["stock_id", "base_month"], how="left", validate="one_to_one")
    t["h3_available"] = t.initialized.fillna(False).astype(bool) & t.recent_valid_count.ge(10)
    t["sigma"] = np.where(t.h3_available, np.sqrt(t.v), np.nan)
    t["h3_reason"] = np.where(t.h3_available, "", np.where(~t.initialized.fillna(False).astype(bool), "not_initialized",
                                                         "recent_valid_below_10"))
    t["recent_valid"] = np.select([t.recent_valid_count.eq(12), t.recent_valid_count.eq(11), t.recent_valid_count.eq(10)],
                                  ["12", "11", "10"], "below_10")
    t["h3_member"] = t.eligible & t.h3_available
    return t


# ---------- 집계 ----------

def rates_by_year(g: pd.DataFrame, items: dict[str, str]) -> pd.DataFrame:
    e = g[g.eligible]
    rows = []
    for year, part in [("2001~2021", e), *e.groupby("year")]:
        for col, label in items.items():
            rows.append({"year": year, "item": label, "eligible_cells": len(part), "available": int(part[col].sum()),
                         "rate": float(part[col].mean()) if len(part) else np.nan})
    return pd.DataFrame(rows)


def reasons_by_year(g: pd.DataFrame, items: dict[str, str]) -> pd.DataFrame:
    e = g[g.eligible]
    out = []
    for col, label in items.items():
        miss = e[e[col].ne("")]
        t = miss.groupby(["year", col]).size().rename("cells").reset_index()
        tot = miss.groupby(col).size().rename("cells").reset_index().assign(year="2001~2021")
        out.append(pd.concat([tot, t]).rename(columns={col: "reason"}).assign(item=label))
    return pd.concat(out, ignore_index=True)[["item", "year", "reason", "cells"]]


def eligibility_by_year(f: pd.DataFrame) -> pd.DataFrame:
    def one(part):
        mc = part.market_cap.gt(0)
        return pd.Series({"cells": len(part), "eligible": int(part.eligible.sum()),
                          "market_cap_cells": int(mc.sum()), "market_cap_ineligible": int((mc & ~part.eligible).sum())})
    return pd.concat([one(f).to_frame("2001~2021").T, f.groupby("year").apply(one, include_groups=False)]
                     ).rename_axis("year").reset_index()


def monthly_table(g: pd.DataFrame, counts: dict[str, str], validity: pd.DataFrame, ic: pd.DataFrame) -> pd.DataFrame:
    e = g[g.eligible]
    m = e.groupby("base_month").agg(**{"eligible": ("stock_id", "size")}, **{k: (v, "sum") for k, v in counts.items()})
    m = m.reindex(sorted(g.base_month.unique()), fill_value=0).rename_axis("base_month").reset_index()
    m.insert(1, "target_month", target_month(m.base_month))
    v = validity.rename(columns={"valid": "score_month_valid", "reason": "score_month_reason", "n": "score_n"})
    return m.merge(v, on="base_month").merge(ic, on="base_month")


def coverage_by_year(monthly: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """노션 10주차 3.1 커버리지 표 형식(목표월 연도). 월평균은 그해 목표월들의 평균 종목 수."""
    m = monthly.assign(year=monthly.target_month.str[:4])
    agg = {"산출 가능 월 수": ("score_month_valid", "sum"), "IC 계산 가능 월 수": ("ic_valid", "sum"),
           "월 수": ("base_month", "size"), "월평균 적격 종목": ("eligible", "mean")}
    agg.update({f"{label} 월평균": (col, "mean") for col, label in columns.items()})
    agg["IC 계산 쌍 월평균"] = ("pairs", "mean")
    out = m.groupby("year").agg(**agg).reset_index()
    total = m.assign(year="2001~2021").groupby("year").agg(**agg).reset_index()
    return pd.concat([out, total], ignore_index=True).round(1).rename(columns={"year": "연도(목표월)"})


def min_max_months(monthly: pd.DataFrame, series: dict[str, str]) -> pd.DataFrame:
    part = monthly[monthly.target_month.ge("2001-02")]  # 목표월 2001-01(기준월 2000-12)은 H1이 0이라 따로 표시
    rows = []
    for col, label in series.items():
        lo, hi = part.loc[part[col].idxmin()], part.loc[part[col].idxmax()]
        rows.append({"series": label, "min_target_month": lo.target_month, "min_n": int(lo[col]),
                     "max_target_month": hi.target_month, "max_n": int(hi[col]), "mean_n": round(float(part[col].mean()), 1),
                     "zero_target_months": ", ".join(monthly.loc[monthly[col].eq(0), "target_month"])})
    return pd.DataFrame(rows)


def baselines(x: dict, mom: pd.DataFrame) -> pd.DataFrame:
    """이전 기록의 기준점을 그 기록과 같은 정의(기준월, 적격성 없음)로 다시 센다."""
    p, e = x["panel"], x["elig"]
    mc = p[p.market_cap.gt(0)]
    mc_base = mc[mc.month.le(BASE_LAST)]
    a = x["asof"].merge(mc_base[["stock_id", "month"]].rename(columns={"month": "base_month"}), on=["stock_id", "base_month"])
    m0 = mom.merge(mc[["stock_id", "month"]].rename(columns={"month": "base_month"}), on=["stock_id", "base_month"])
    basis = e.drop_duplicates("stock_id").set_index("stock_id").eligibility_basis.str.split(":").str[0]
    mce = mc_base.merge(e[["stock_id", "month", "eligible"]], on=["stock_id", "month"], suffixes=("", "_e"))
    rows = [
        ("시가총액 있는 칸(기준월 2000-01~2021-11)", 181439, len(mc_base), "as-of 기록"),
        ("그중 자기자본 값 있음 비율(%)", 99.4, round(100 * a.equity.notna().mean(), 1), "as-of 기록"),
        ("그중 B/M 사용 가능 비율(%)", 97.8, round(100 * a.usable.mean(), 1), "as-of 기록"),
        ("모멘텀 필요 칸(시가총액 있음, 기준월 2000-01~2021-12)", 182192, len(m0), "H1.md 측정"),
        ("그중 모멘텀 결측", 13985, int((~m0.mom_available).sum()), "H1.md 측정"),
        ("기록 없고 KOSPI 관측 없는 종목(판정 불가)", 42, int(basis.eq("no_event_not_kospi_observed").sum()), "적격성 규칙"),
        ("기록 없고 KOSPI 관측 있는 종목(상장 유지)", 450, int(basis.eq("no_event_kospi_observed").sum()), "적격성 규칙"),
        ("시가총액 있는 칸 중 비적격(기준월 2000-01~2021-11)", 336, int(mce.eligible.eq(False).sum()),
         "2026-11-04 규칙. 2026-11-05 이전 상장 기간 규칙으로 줄어드는 것이 정상"),
    ]
    out = pd.DataFrame(rows, columns=["item", "baseline", "this_run", "baseline_source"])
    out["judgement"] = np.where(out.baseline == out.this_run, "같음", "다름")
    return out[["item", "baseline", "this_run", "judgement", "baseline_source"]]


# ---------- 그림 ----------

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "lines.linewidth": 2, "figure.dpi": 130})
    return plt


def line_chart(plt, path: Path, x, series: dict[str, list], title: str, ylabel: str, percent: bool, xlabel: str):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for (label, ys), color in zip(series.items(), COLORS):
        ax.plot(x, ys, label=label, color=color)
    ax.set_title(title, loc="left", color=INK)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if percent:
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    step = max(1, len(x) // 12)
    ax.set_xticks(range(0, len(x), step))
    ax.set_xticklabels([x[i] for i in range(0, len(x), step)])
    if len(series) > 1:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write(folder: Path, tables: dict[str, pd.DataFrame], run: dict, readme: str):
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
    (folder / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (folder / "README.md").write_text(readme, encoding="utf-8")


def main(root: Path = Path("."), config: dict | None = None) -> dict:
    started = datetime.now()
    stamp = started.strftime("%Y%m%dT%H%M%S")
    x = load_inputs(root, config)
    run_common = {"started": started.isoformat(timespec="seconds"), "command": " ".join(sys.argv),
                  "base_months": [BASE_FIRST, BASE_LAST], "year_axis": "target_month_year",
                  "returns": "provided CSV only (KOSDAQ->KOSPI transfers treated as separate companies)",
                  "input_sha256": x["hashes"],
                  "code_sha256": {p: sha256(root / p) for p in ("src/features/observe.py", "src/features/h3.py")}}
    plt = setup_matplotlib()
    f = base_frame(x)
    elig_year = eligibility_by_year(f)

    # H1
    mom = momentum_status(x)
    g = h1_status(f, x, mom)
    h1_items = {"size_available": "규모", "bm_available": "자기자본·B/M", "mom_available": "모멘텀",
                "joint_available": "세 값 공동 유효"}
    avail = rates_by_year(g, h1_items)
    reasons = reasons_by_year(g, {"size_reason": "규모", "bm_reason": "자기자본·B/M", "mom_reason": "모멘텀",
                                  "joint_reason": "세 값 공동 유효"})
    valid1 = month_validity(g, "joint_member", ["size", "bm", "mom"], n_min=2)
    ic1 = ic_validity(g, "joint_member")
    e = g.assign(size_c=g.eligible & g.size_available, bm_c=g.eligible & g.bm_available, mom_c=g.eligible & g.mom_available)
    m1 = monthly_table(e, {"size": "size_c", "bm": "bm_c", "mom": "mom_c", "joint": "joint_member"}, valid1, ic1)
    cov1 = coverage_by_year(m1, {"size": "−Size", "bm": "B/M", "mom": "Momentum", "joint": "H1 공통"})
    mm1 = min_max_months(m1, {"eligible": "적격 종목", "joint": "세 값 공동 유효", "pairs": "IC 계산 쌍"})
    base1 = baselines(x, mom)
    d1 = root / OUT / "H1" / stamp
    write(d1, {"coverage_by_target_year": cov1, "availability_by_year": avail, "reasons_by_year": reasons,
               "eligibility_by_year": elig_year, "monthly": m1, "monthly_min_max": mm1, "baseline_check": base1},
          {**run_common, "h1_score_months": int(valid1.valid.sum()), "ic_months": int(ic1.ic_valid.sum()),
           "baseline_all_same": bool((base1.judgement == "같음").all())}, README_H1)
    years = [y for y in avail.year.unique() if y != "2001~2021"]
    line_chart(plt, d1 / "1_연도별_산출률.png", years,
               {lab: avail[(avail.item == lab) & (avail.year != "2001~2021")].rate.tolist() for lab in h1_items.values()},
               "H1 구성값 연도별 산출률 (분모: 기준월 적격 종목)", "산출률", True, "목표월 연도")
    line_chart(plt, d1 / "2_월별_종목수.png", m1.target_month.tolist(),
               {"적격 종목": m1.eligible.tolist(), "세 값 공동 유효": m1.joint.tolist(), "IC 계산 쌍": m1.pairs.tolist()},
               "월별 적격 종목 수, 세 값 공동 유효 종목 수, 차월 수익률까지 있는 쌍 수", "종목 수", False, "목표월")

    # H3
    t = h3_status(f, x).merge(g[["stock_id", "base_month", "joint_member", "joint_available"]],
                              on=["stock_id", "base_month"], validate="one_to_one")
    t["both_member"] = t.h3_member & t.joint_member
    t["both_available"] = t.h3_available & t.joint_available
    avail3 = rates_by_year(t, {"h3_available": "H3", "both_available": "H1·H3 동시 산출"})
    reasons3 = reasons_by_year(t, {"h3_reason": "H3"})
    e3 = t[t.eligible]
    recent3 = pd.concat([e3.groupby("recent_valid").size().rename("cells").reset_index().assign(year="2001~2021"),
                         e3.groupby(["year", "recent_valid"]).size().rename("cells").reset_index()])[["year", "recent_valid", "cells"]]
    valid3 = month_validity(t, "h3_member", ["sigma"], n_min=2)
    ic3 = ic_validity(t, "h3_member")
    m3 = monthly_table(t, {"h3": "h3_member", "both": "both_member"}, valid3, ic3)
    cov3 = coverage_by_year(m3, {"h3": "H3", "both": "H1·H3 동시"})
    mm3 = min_max_months(m3, {"eligible": "적격 종목", "h3": "H3 산출", "both": "H1·H3 동시 산출", "pairs": "IC 계산 쌍"})
    d3 = root / OUT / "H3" / stamp
    write(d3, {"coverage_by_target_year": cov3, "availability_by_year": avail3, "reasons_by_year": reasons3,
               "recent_valid_by_year": recent3, "eligibility_by_year": elig_year, "monthly": m3, "monthly_min_max": mm3},
          {**run_common, "h3_score_months": int(valid3.valid.sum()), "ic_months": int(ic3.ic_valid.sum())}, README_H3)
    years3 = [y for y in avail3.year.unique() if y != "2001~2021"]
    line_chart(plt, d3 / "1_연도별_산출률.png", years3,
               {lab: avail3[(avail3.item == lab) & (avail3.year != "2001~2021")].rate.tolist()
                for lab in ("H3", "H1·H3 동시 산출")},
               "H3 연도별 산출률과 H1·H3 동시 산출률 (분모: 기준월 적격 종목)", "산출률", True, "목표월 연도")
    line_chart(plt, d3 / "2_월별_종목수.png", m3.target_month.tolist(),
               {"적격 종목": m3.eligible.tolist(), "H3 산출": m3.h3.tolist(), "H1·H3 동시 산출": m3.both.tolist(),
                "IC 계산 쌍(H3)": m3.pairs.tolist()},
               "월별 적격 종목 수, H3 산출 종목 수, H1·H3 동시 산출 종목 수, H3 IC 계산 쌍 수", "종목 수", False, "목표월")
    return {"h1": d1, "h3": d3, "seconds": (datetime.now() - started).total_seconds()}


README_H1 = """# H1 가용성 관측(목표월 연도)

입력: `data/interim/panel.parquet`·`eligibility.parquet`(공유 표), `data/interim/asof.parquet`. 수익률은 제공 CSV만.
대상: 기준월 2000-12~2021-11(목표월 2001-01~2021-12). 연도는 목표월(차월 수익률) 연도. 분모는 기준월 적격 종목.

- `coverage_by_target_year.csv`: 노션 10주차 3.1 커버리지 표. 산출 가능 월 수(공통 유효 2개 이상 + 규모·B/M·모멘텀 표준편차 > 0),
  IC 계산 가능 월 수(공통 유효 중 차월 수익률 있는 쌍 3개 이상 + 차월 수익률 비상수. H1 점수 변동 확인은 H1 계산 후),
  월평균 적격 종목, 특징별·공통 월평균 종목 수, IC 계산 쌍 월평균.
- `availability_by_year.csv`: 특징별 산출 칸·비율. `reasons_by_year.csv`: 미산출 사유(규모 market_cap_missing/nonpositive,
  B/M은 as-of 사유 before_first_public_value·expired_18m·no_dataguide_value·nonpositive_equity 또는 market_cap_missing,
  모멘텀 window_includes_pre_listing(창 안 결측이 KOSPI 상장 전 달)·missing_1·missing_2_5·missing_6_plus).
- `eligibility_by_year.csv`: 적격 칸, 시가총액 있는 칸 중 비적격. `monthly.csv`: 기준월·목표월별 종목 수와 월 유효 판정.
- `monthly_min_max.csv`: 목표월 2001-02 이후 최소·최대 달, 0인 목표월. `baseline_check.csv`: 이전 기록과 같은 정의로 다시 센 대조.
- 그림 1 목표월 연도별 산출률, 그림 2 목표월별 종목 수.
특징과 차월 수익률의 관계(상관·IC)는 계산하지 않았다. 특징 원값은 표준편차 확인에만 쓰고 저장하지 않았다.
"""

README_H3 = """# H3 가용성 관측(목표월 연도)

입력·대상·연도 기준은 H1과 같다. 산출 = 초기화(최초 유효 수익률 10개) + 최근 12개월(t 포함) 유효 10개 이상.
4개월 공백 초기화 규칙은 적용하지 않았다.

- `coverage_by_target_year.csv`: 산출 가능 월 수(H3 종목 2개 이상 + sigma 표준편차 > 0), IC 계산 가능 월 수(H3 종목 중 차월 수익률 쌍 3개 이상
  + 비상수), 월평균 적격·H3·H1·H3 동시 종목 수, IC 계산 쌍 월평균.
- `availability_by_year.csv`(H3, H1·H3 동시), `reasons_by_year.csv`(not_initialized, recent_valid_below_10), `recent_valid_by_year.csv`,
  `eligibility_by_year.csv`, `monthly.csv`, `monthly_min_max.csv`, 그림 1·2.
"""


if __name__ == "__main__":
    result = main()
    print(json.dumps({k: str(v) for k, v in result.items()}, ensure_ascii=False))
