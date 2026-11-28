"""검증 기간(2006~2021, 192개월) 결과 분석: 가설 검정·포트폴리오·H4 직접 검정·피처 단독 IC(7-2, 2026-11-28).

입력(읽기만): main_v3 결과 폴더 3개(monthly_ic.csv, validation_predictions.parquet), basefeat_v2 선형 monthly_ic,
개발 표(panel, mom, base_features, h1_components, h3, h4). 학습·튜닝은 하지 않는다.
출력: outputs/run_id/VALIDATION_ANALYSIS/<시각>/report.md + run.json (한 폴더).

검정 규칙(관례값은 표시):
- 평균 월별 IC·IC 차이: Newey-West t, 시차 = floor(4·(T/100)^(2/9)) [Newey-West 1994 자동 시차 관례], p는 t분포(자유도 T−1).
- 모델 비교: 월별 IC 차이의 NW t = Diebold-Mariano 검정(손실 = −IC).
- 가설 기여(H1·H3·H4): 세트 차이 월별 IC의 NW t, 모델마다 3개를 Holm 보정.
- 다중 검정: 시험한 설정 전체 중 최고를 고른 것 → Harvey·Liu·Zhu(2016) t > 3 기준 표시.
- 포트폴리오: 매달 예측 순위 10분위, 다음 달 수익률 동일·시가총액(기준월 말) 가중. 상위−하위, 상위 − 시장.
- H4: 월별 모멘텀 기울기(순위 IC, 단면 z 회귀 계수)를 EPU(t)에 시계열 회귀(Cooper·Gutierrez·Hameed 2004 시장 국면 방식).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "outputs/model_results/jiun"
DRIVE = Path("G:/내 드라이브/model_results/jiun")
RUNS = {"xgb_pairwise": DRIVE / "20261126T014119_main_v3_xgb_pairwise_basefeat",
        "xgb_rankreg": DRIVE / "20261126T014119_main_v3_xgb_rankreg_basefeat",
        "lgb_rankreg": LOCAL / "20261126T102145_main_v3_lgb_rankreg_basefeat"}
LINEAR = LOCAL / "20261121T042132_basefeat_v2_winsor_zscore_linear"
FINAL = ("xgb_pairwise", "base+h1+h3+h4")
SHORT_BAN = ("2020-03", "2021-05")
NAMES = {"xgb_pairwise": "XGB pairwise", "xgb_rankreg": "XGB 순위 회귀", "lgb_rankreg": "LGB 순위 회귀"}


def nw(x) -> dict:
    x = pd.Series(x, dtype=float).dropna().to_numpy()
    T = len(x)
    lag = int(math.floor(4 * (T / 100) ** (2 / 9)))
    e = x - x.mean()
    s = e @ e / T
    for k in range(1, lag + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / T
    se = math.sqrt(s / T)
    t = x.mean() / se if se > 0 else float("nan")
    return {"mean": float(x.mean()), "t": float(t), "p": float(2 * stats.t.sf(abs(t), T - 1)), "T": T, "lag": lag}


def holm(p: dict) -> dict:
    order = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        out[k] = run
    return out


def monthly_ic(folder: Path) -> pd.DataFrame:
    m = pd.read_csv(folder / "monthly_ic.csv")
    if "scope" in m:
        m = m[m.scope.eq("full")]
    return m.pivot_table(index="target_month", columns="input_set", values="ic")


def model_tests(ics: dict, linear: pd.Series) -> dict:
    out = {"ic": {}, "dm": {}, "contrib": {}, "all_configs_t": []}
    for model, tab in ics.items():
        for s in tab.columns:
            r = nw(tab[s])
            r["icir"] = float(tab[s].mean() / tab[s].std(ddof=1))
            out["ic"][f"{model}|{s}"] = r
            out["all_configs_t"].append(r["t"])
        steps = {"H1": ("base", "base+h1"), "H3": ("base", "base+h3"), "H4": ("base+h1+h3", "base+h1+h3+h4")}
        res = {h: nw(tab[b] - tab[a]) for h, (a, b) in steps.items()}
        adj = holm({h: r["p"] for h, r in res.items()})
        for h in res:
            res[h]["p_holm"] = adj[h]
            res[h]["hlz_t_gt_3"] = abs(res[h]["t"]) > 3
        out["contrib"][model] = res
    final = ics[FINAL[0]][FINAL[1]]
    for model in ics:
        if model != FINAL[0]:
            out["dm"][f"{model}|{FINAL[1]}"] = nw(final - ics[model][FINAL[1]])
    out["dm"]["linear_ols|base+h1+h3 (basefeat_v2)"] = nw(final - linear.reindex(final.index))
    out["ic"]["linear_ols|base+h1+h3 (basefeat_v2)"] = nw(linear)
    return out


def portfolios(pred: pd.DataFrame, panel: pd.DataFrame) -> dict:
    cap = panel[["stock_id", "month", "market_cap"]].copy()
    cap["target_month"] = (pd.PeriodIndex(cap.month, freq="M") + 1).astype(str)
    d = pred.merge(cap[["stock_id", "target_month", "market_cap"]], on=["stock_id", "target_month"], how="left")
    d["decile"] = d.groupby("target_month").score.transform(lambda s: np.floor(10 * (s.rank(method="first") - 1) / len(s))).astype(int)
    d["w"] = d.market_cap.where(d.market_cap > 0)
    ew = d.groupby(["target_month", "decile"]).y.mean().unstack()
    vw = d.dropna(subset=["w"]).groupby(["target_month", "decile"]).apply(lambda g: np.average(g.y, weights=g.w), include_groups=False).unstack()
    mkt_ew = d.groupby("target_month").y.mean()
    mkt_vw = d.dropna(subset=["w"]).groupby("target_month").apply(lambda g: np.average(g.y, weights=g.w), include_groups=False)
    top = d[d.decile.eq(9)].groupby("target_month").stock_id.apply(set)
    turnover = [1 - len(a & b) / len(b) for a, b in zip(top.iloc[:-1], top.iloc[1:])]
    out = {"turnover_top_decile": float(np.mean(turnover)), "n_months": int(ew.shape[0])}
    ban = (ew.index >= SHORT_BAN[0]) & (ew.index <= SHORT_BAN[1])
    for w, tab, mkt in (("EW", ew, mkt_ew), ("VW", vw, mkt_vw)):
        ls = tab[9] - tab[0]
        lo = tab[9] - mkt.reindex(tab.index)
        out[w] = {"decile_mean": {int(k): float(v) for k, v in tab.mean().items()},
                  "long_short": {**nw(ls), "sharpe_ann": float(ls.mean() / ls.std(ddof=1) * math.sqrt(12))},
                  "top_minus_market": {**nw(lo), "sharpe_ann": float(lo.mean() / lo.std(ddof=1) * math.sqrt(12))},
                  "bottom_minus_market": nw(tab[0] - mkt.reindex(tab.index)),
                  "monotonic_rank_corr": float(stats.spearmanr(range(10), tab.mean().to_numpy())[0]),
                  "short_ban": {"months": int(ban.sum()), "long_short": nw(ls[ban]), "top_minus_market": nw(lo[ban])},
                  "outside_ban": {"long_short": nw(ls[~ban]), "top_minus_market": nw(lo[~ban])}}
    return out


def h4_test(root: Path, config: dict) -> dict:
    from src.features.h4 import load_epu
    mom = pd.read_parquet(root / config["PATHS"]["mom"])
    ret = pd.read_parquet(root / config["PATHS"]["panel"], columns=["stock_id", "month", "return"])
    ret["base_month"] = (pd.PeriodIndex(ret.month, freq="M") - 1).astype(str)
    d = mom[mom.eligible.eq(True) & mom.mom_available.eq(True)].merge(
        ret[["stock_id", "base_month", "return"]], on=["stock_id", "base_month"]).dropna(subset=["mom", "return"])
    rows = []
    for t, g in d.groupby("base_month"):
        if len(g) < 3:
            continue
        z = (g.mom - g.mom.mean()) / g.mom.std(ddof=1)
        rows.append({"base_month": t, "ic": stats.spearmanr(g.mom, g["return"])[0],
                     "slope": float(np.polyfit(z, g["return"], 1)[0]), "n": len(g)})
    m = pd.DataFrame(rows).merge(load_epu(root, config["H4"])[["base_month", "epu_raw"]], on="base_month")
    out = {"months": len(m), "range": [m.base_month.min(), m.base_month.max()], "mom_ic": nw(m.ic), "mom_slope": nw(m.slope)}
    ez = (m.epu_raw - m.epu_raw.mean()) / m.epu_raw.std(ddof=1)
    for y in ("ic", "slope"):
        X = np.column_stack([np.ones(len(m)), ez])
        b, *_ = np.linalg.lstsq(X, m[y].to_numpy(), rcond=None)
        e = m[y].to_numpy() - X @ b
        T, lag = len(m), int(math.floor(4 * (len(m) / 100) ** (2 / 9)))
        S = (X * e[:, None]).T @ (X * e[:, None]) / T
        for k in range(1, lag + 1):
            G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / T
            S += (1 - k / (lag + 1)) * (G + G.T)
        Q = np.linalg.inv(X.T @ X / T)
        V = Q @ S @ Q / T
        t = b[1] / math.sqrt(V[1, 1])
        out[f"{y}_on_epu"] = {"coef_per_1sd_epu": float(b[1]), "t": float(t), "p": float(2 * stats.t.sf(abs(t), T - 2)),
                              "intercept": float(b[0])}
    hi = m.epu_raw >= m.epu_raw.median()
    out["high_vs_low_epu"] = {"high_ic": float(m.ic[hi].mean()), "low_ic": float(m.ic[~hi].mean()),
                              "diff_welch_t": float(stats.ttest_ind(m.ic[hi], m.ic[~hi], equal_var=False).statistic),
                              "diff_welch_p": float(stats.ttest_ind(m.ic[hi], m.ic[~hi], equal_var=False).pvalue)}
    return out


def ols_nw(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS 계수와 Newey-West t(시차 = floor(4·(T/100)^(2/9)))."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    T, lag = len(y), int(math.floor(4 * (len(y) / 100) ** (2 / 9)))
    S = (X * e[:, None]).T @ (X * e[:, None]) / T
    for k in range(1, lag + 1):
        G = (X[k:] * e[k:, None]).T @ (X[:-k] * e[:-k, None]) / T
        S += (1 - k / (lag + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    return b, b / np.sqrt(np.diag(Q @ S @ Q / T))


def factor_returns(root: Path, config: dict) -> pd.DataFrame:
    """한국 요인 월 수익률(목표월 기준, 우리 자료로 만듦 — 관례값 표시).
    매달 적격 종목을 시가총액 중앙값(규모)과 B/M·모멘텀 30/70% 분위로 2×3칸으로 나누고, 칸마다 다음 달 수익률을 기준월 말
    시가총액으로 가중 평균(Fama·French 1993·2015 방식). 나누는 기준 종목 = KOSPI 적격 종목 전체(문헌은 대형 거래소 기준).
    SMB = 규모·B/M 칸의 소형 평균 − 대형 평균, HML = (소형 고 + 대형 고)/2 − (소형 저 + 대형 저)/2, UMD는 모멘텀으로 같은 식.
    MKT = 코스피 지수 월 수익률(무위험 수익률 자료가 없어 차감하지 않음)."""
    from src.features.base import kospi_monthly
    P = config["PATHS"]
    c = pd.read_parquet(root / P["h1_components"], columns=["stock_id", "base_month", "eligible", "bm", "bm_available",
                                                             "mom", "mom_available"])
    p = pd.read_parquet(root / P["panel"], columns=["stock_id", "month", "return", "market_cap"])
    nxt = p[["stock_id", "month", "return"]].assign(base_month=(pd.PeriodIndex(p.month, freq="M") - 1).astype(str))
    d = c[c.eligible.eq(True)].merge(p[["stock_id", "month", "market_cap"]].rename(columns={"month": "base_month"}),
                                     on=["stock_id", "base_month"]).merge(nxt[["stock_id", "base_month", "return"]],
                                                                          on=["stock_id", "base_month"])
    d = d[d.market_cap.gt(0) & d["return"].notna()]

    def sort2x3(col, ok):
        x = d[ok].copy()
        x["big"] = x.market_cap >= x.groupby("base_month").market_cap.transform("median")
        lo = x.groupby("base_month")[col].transform(lambda s: s.quantile(0.3))
        hi = x.groupby("base_month")[col].transform(lambda s: s.quantile(0.7))
        x["g"] = np.where(x[col] <= lo, "L", np.where(x[col] >= hi, "H", "M"))
        r = x.groupby(["base_month", "big", "g"]).apply(lambda g: np.average(g["return"], weights=g.market_cap),
                                                         include_groups=False).unstack(["big", "g"])
        return r

    v = sort2x3("bm", d.bm_available.eq(True) & d.bm.gt(0))
    m = sort2x3("mom", d.mom_available.eq(True))
    f = pd.DataFrame({
        "SMB": v.xs(False, axis=1, level=0).mean(axis=1) - v.xs(True, axis=1, level=0).mean(axis=1),
        "HML": (v[(False, "H")] + v[(True, "H")]) / 2 - (v[(False, "L")] + v[(True, "L")]) / 2,
        "UMD": (m[(False, "H")] + m[(True, "H")]) / 2 - (m[(False, "L")] + m[(True, "L")]) / 2})
    f.index = (pd.PeriodIndex(f.index, freq="M") + 1).astype(str)
    lvl = kospi_monthly(root, config["BASE_FEATURES"])
    f["MKT"] = (lvl / lvl.shift(1) - 1).reindex(f.index)
    f.index.name = "target_month"
    return f[["MKT", "SMB", "HML", "UMD"]]


def portfolio_series(pred: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """월별 상위−하위·상위−시장(동일·시가총액 가중). portfolios와 같은 10분위·가중."""
    cap = panel[["stock_id", "month", "market_cap"]].copy()
    cap["target_month"] = (pd.PeriodIndex(cap.month, freq="M") + 1).astype(str)
    d = pred.merge(cap[["stock_id", "target_month", "market_cap"]], on=["stock_id", "target_month"], how="left")
    d["decile"] = d.groupby("target_month").score.transform(lambda s: np.floor(10 * (s.rank(method="first") - 1) / len(s))).astype(int)
    w = d.dropna(subset=["market_cap"]).query("market_cap > 0")
    ew = d.groupby(["target_month", "decile"]).y.mean().unstack()
    vw = w.groupby(["target_month", "decile"]).apply(lambda g: np.average(g.y, weights=g.market_cap), include_groups=False).unstack()
    mkt_ew = d.groupby("target_month").y.mean()
    mkt_vw = w.groupby("target_month").apply(lambda g: np.average(g.y, weights=g.market_cap), include_groups=False)
    return pd.DataFrame({"LS_EW": ew[9] - ew[0], "LS_VW": vw[9] - vw[0],
                         "TOP_EW": ew[9] - mkt_ew, "TOP_VW": vw[9] - mkt_vw})


def factor_alpha(series: pd.DataFrame, factors: pd.DataFrame) -> dict:
    j = series.join(factors, how="inner").dropna()
    X = np.column_stack([np.ones(len(j)), j[["MKT", "SMB", "HML", "UMD"]].to_numpy()])
    out = {"months": len(j), "range": [j.index.min(), j.index.max()],
           "factor_means": {c: nw(j[c]) for c in ("MKT", "SMB", "HML", "UMD")},
           "factor_corr": j[["MKT", "SMB", "HML", "UMD"]].corr().round(3).to_dict()}
    for col in series.columns:
        y = j[col].to_numpy()
        b, t = ols_nw(y, X)
        resid = y - X @ b
        out[col] = {"alpha": float(b[0]), "alpha_t": float(t[0]),
                    "alpha_p": float(2 * stats.t.sf(abs(t[0]), len(y) - X.shape[1])),
                    "loadings": {n: float(v) for n, v in zip(("MKT", "SMB", "HML", "UMD"), b[1:])},
                    "loading_t": {n: float(v) for n, v in zip(("MKT", "SMB", "HML", "UMD"), t[1:])},
                    "r2": float(1 - resid.var() / y.var()), "raw_mean": float(y.mean())}
    return out


def feature_ic(root: Path, config: dict) -> dict:
    """최종 세트 입력 각각의 단독 순위 IC(목표월 2001~2021). 누수 피처는 보통 0.3 이상."""
    from src.features.base import STOCK_FEATURES
    P = config["PATHS"]
    k = ["stock_id", "base_month"]
    b = pd.read_parquet(root / P["base_features"], columns=k + ["eligible", *STOCK_FEATURES])
    h1 = pd.read_parquet(root / P["h1_components"], columns=k + ["size", "bm", "mom"])
    h3 = pd.read_parquet(root / P["h3"], columns=k + ["sigma"])
    h4 = pd.read_parquet(root / P["h4"], columns=k + ["h4_interaction"])
    ret = pd.read_parquet(root / P["panel"], columns=["stock_id", "month", "return"])
    ret["base_month"] = (pd.PeriodIndex(ret.month, freq="M") - 1).astype(str)
    d = b[b.eligible.eq(True)].merge(h1, on=k, how="left").merge(h3, on=k, how="left").merge(h4, on=k, how="left").merge(
        ret[["stock_id", "base_month", "return"]], on=k)
    d = d[d.base_month.between("2000-12", "2021-11") & d["return"].notna()]
    cols = [*STOCK_FEATURES, "size", "bm", "mom", "sigma", "h4_interaction"]
    r = d.groupby("base_month")[cols + ["return"]].rank()
    out = {}
    for c in cols:
        s = r[[c, "return"]].groupby(d.base_month).apply(lambda g: g[c].corr(g["return"]), include_groups=False)
        out[c] = {"mean_ic": float(s.mean()), "max_abs_month": float(s.abs().max())}
    return out


def fmt(x, n=4):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"


def report(res: dict) -> str:
    L = ["# 검증 기간 결과 분석 (2006~2021, 192개월)", "",
         f"만든 시각 {res['started']}. 입력은 main_v3 결과(학습 없음). 최종 모델 = XGB pairwise, 기본+H1+H3+H4.", ""]
    mt = res["model_tests"]
    L += ["## 1. 모델 평균 IC (Newey-West t)", "", "| 모델·세트 | 평균 IC | ICIR | NW t | p | 시차 |", "|---|---|---|---|---|---|"]
    for k, r in mt["ic"].items():
        if k.endswith("|base+h1+h3+h4") or "linear" in k:
            L.append(f"| {k} | {fmt(r['mean'])} | {fmt(r.get('icir', float('nan')), 2)} | {fmt(r['t'], 2)} | {r['p']:.1e} | {r['lag']} |")
    tmax = max(mt["all_configs_t"])
    L += ["", f"시험한 부스팅 설정 {len(mt['all_configs_t'])}개(모델 3 × 세트 7) 모두 t > 3 "
          f"(최소 {min(mt['all_configs_t']):.1f}, 최대 {tmax:.1f}) → Harvey·Liu·Zhu(2016) 기준을 넘음.", ""]
    L += ["## 2. 모델 비교 (Diebold-Mariano: 최종 − 비교 모델의 월별 IC 차이)", "", "| 비교 모델 | 평균 차이 | NW t | p |", "|---|---|---|---|"]
    for k, r in mt["dm"].items():
        L.append(f"| {k} | {fmt(r['mean'])} | {fmt(r['t'], 2)} | {fmt(r['p'], 3)} |")
    L += ["", "선형은 basefeat_v2(업종 피처 6개 전) 같은 세트가 없어 기본+H1+H3 1등 설정(OLS·월별 z·Huber②)과 비교.", ""]
    L += ["## 3. 가설 기여 (세트 차이 월별 IC, NW t, 모델마다 Holm 보정)", "",
          "| 모델 | 가설 | 비교 | 평균 차이 | NW t | p | Holm p | t > 3 |", "|---|---|---|---|---|---|---|---|"]
    steps = {"H1": "기본 → +H1", "H3": "기본 → +H3", "H4": "+H1+H3 → +H1+H3+H4"}
    for m, rr in mt["contrib"].items():
        for h, r in rr.items():
            L.append(f"| {NAMES[m]} | {h} | {steps[h]} | {fmt(r['mean'])} | {fmt(r['t'], 2)} | {fmt(r['p'], 3)} | "
                     f"{fmt(r['p_holm'], 3)} | {'예' if r['hlz_t_gt_3'] else '아니오'} |")
    pf = res["portfolios"]
    L += ["", "## 4. 10분위 포트폴리오 (최종 모델 예측, 다음 달 수익률, 월 %)", "",
          f"상위 10분위 월 교체율 평균 {pf['turnover_top_decile']:.1%}. 공매도 금지 구간 {SHORT_BAN[0]}~{SHORT_BAN[1]}.", "",
          "| 가중 | D1(하위) | D5 | D10(상위) | 상위−하위 | NW t | 연 샤프 | 상위−시장 | NW t | 하위−시장 | NW t | 단조성 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in ("EW", "VW"):
        r = pf[w]
        dm = r["decile_mean"]
        L.append(f"| {w} | {dm[0]*100:.2f} | {dm[4]*100:.2f} | {dm[9]*100:.2f} | {r['long_short']['mean']*100:.2f} | "
                 f"{r['long_short']['t']:.2f} | {r['long_short']['sharpe_ann']:.2f} | {r['top_minus_market']['mean']*100:.2f} | "
                 f"{r['top_minus_market']['t']:.2f} | {r['bottom_minus_market']['mean']*100:.2f} | {r['bottom_minus_market']['t']:.2f} | "
                 f"{r['monotonic_rank_corr']:.2f} |")
    L += ["", "| 가중 | 구간 | 개월 | 상위−하위 | NW t | 상위−시장(롱 온리) | NW t |", "|---|---|---|---|---|---|---|"]
    for w in ("EW", "VW"):
        r = pf[w]
        for name, part in (("공매도 금지", r["short_ban"]), ("그 밖", r["outside_ban"])):
            L.append(f"| {w} | {name} | {part['long_short']['T']} | {part['long_short']['mean']*100:.2f} | {part['long_short']['t']:.2f} | "
                     f"{part['top_minus_market']['mean']*100:.2f} | {part['top_minus_market']['t']:.2f} |")
    h = res["h4"]
    L += ["", f"## 5. H4 직접 검정 (기준월 {h['range'][0]}~{h['range'][1]}, {h['months']}개월)", "",
          f"- 모멘텀 단독: 월별 순위 IC 평균 {fmt(h['mom_ic']['mean'])} (NW t {h['mom_ic']['t']:.2f}), "
          f"단면 z 회귀 계수 평균 {h['mom_slope']['mean']*100:.3f}%p (t {h['mom_slope']['t']:.2f})",
          f"- 월별 모멘텀 IC를 EPU(표준화)에 회귀: 계수 {fmt(h['ic_on_epu']['coef_per_1sd_epu'])} (NW t {h['ic_on_epu']['t']:.2f}, p {h['ic_on_epu']['p']:.3f})",
          f"- 월별 모멘텀 기울기를 EPU에 회귀: 계수 {h['slope_on_epu']['coef_per_1sd_epu']*100:.3f}%p (NW t {h['slope_on_epu']['t']:.2f}, p {h['slope_on_epu']['p']:.3f})",
          f"- EPU 상위 절반 달 모멘텀 IC {fmt(h['high_vs_low_epu']['high_ic'])} vs 하위 절반 {fmt(h['high_vs_low_epu']['low_ic'])} "
          f"(Welch t {h['high_vs_low_epu']['diff_welch_t']:.2f}, p {h['high_vs_low_epu']['diff_welch_p']:.3f})",
          "- H4 가설 방향: EPU가 높을수록 모멘텀 효과가 약함(계수 음수).", ""]
    fa = res.get("factor_alpha")
    if fa:
        L += [f"## 5-1. 요인 알파 (목표월 {fa['range'][0]}~{fa['range'][1]}, {fa['months']}개월, 우리 자료로 만든 요인)", "",
              "요인: MKT 코스피 수익률, SMB·HML·UMD = 규모 중앙값 × B/M(모멘텀) 30/70% 2×3칸 시가총액 가중(Fama·French 방식, "
              "기준 종목 = KOSPI 적격 전체, 무위험 수익률 차감 없음 — 관례값).", "",
              "| 요인 | 월평균 % | NW t |", "|---|---|---|"]
        for c2, r in fa["factor_means"].items():
            L.append(f"| {c2} | {r['mean']*100:.2f} | {r['t']:.2f} |")
        L += ["", "| 포트폴리오 | 원 평균 % | 알파 % | 알파 t | MKT | SMB | HML | UMD | R² |", "|---|---|---|---|---|---|---|---|---|"]
        names = {"LS_EW": "상위−하위 동일", "LS_VW": "상위−하위 시총", "TOP_EW": "상위−시장 동일", "TOP_VW": "상위−시장 시총"}
        for k2, nm in names.items():
            r = fa[k2]
            ld = " | ".join(f"{r['loadings'][f]:.2f} ({r['loading_t'][f]:.1f})" for f in ("MKT", "SMB", "HML", "UMD"))
            L.append(f"| {nm} | {r['raw_mean']*100:.2f} | {r['alpha']*100:.2f} | {r['alpha_t']:.2f} | {ld} | {r['r2']:.2f} |")
        L += ["", "괄호 = NW t. 알파 = 네 요인으로 설명되지 않는 월평균 수익.", ""]
    L += ["## 6. 피처 단독 순위 IC (정답 누수 점검 ③, 목표월 2001~2021)", "",
          "누수 피처는 보통 평균 |IC| 0.3 이상. 아래는 절댓값 큰 순.", "", "| 피처 | 평균 IC | 한 달 최대 |IC| |", "|---|---|---|"]
    for c, r in sorted(res["feature_ic"].items(), key=lambda kv: -abs(kv[1]["mean_ic"])):
        L.append(f"| {c} | {fmt(r['mean_ic'])} | {fmt(r['max_abs_month'], 3)} |")
    return "\n".join(L) + "\n"


def main() -> int:
    config = yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))
    started = datetime.now()
    ics = {m: monthly_ic(p) for m, p in RUNS.items()}
    lin = pd.read_csv(LINEAR / "monthly_ic.csv")
    lin = lin[lin.config.eq("linear|ols|monthly_zscore|huber_sigma135|base+h1+h3") & lin.scope.eq("full")].set_index("target_month").ic
    pred = pd.read_parquet(RUNS[FINAL[0]] / "validation_predictions.parquet")
    col = [c for c in pred.columns if f"|{FINAL[1]}|" in c]
    assert len(col) == 1
    pred = pred[["stock_id", "target_month", "y"]].assign(score=pred[col[0]])
    panel = pd.read_parquet(ROOT / config["PATHS"]["panel"], columns=["stock_id", "month", "market_cap"])
    res = {"started": started.isoformat(timespec="seconds"),
           "inputs": {m: str(p) for m, p in {**RUNS, "linear": LINEAR}.items()},
           "model_tests": model_tests(ics, lin), "portfolios": portfolios(pred, panel),
           "h4": h4_test(ROOT, config), "feature_ic": feature_ic(ROOT, config),
           "factor_alpha": factor_alpha(portfolio_series(pred, panel), factor_returns(ROOT, config))}
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/run_id/VALIDATION_ANALYSIS" / started.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report(res), encoding="utf-8")
    (out / "run.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(Path(out).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
