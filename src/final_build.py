"""최종 표 실행 파일(7-4 ⑦): 자료 → data/final/ 표 10개를 한 명령으로 만든다(2026-11-27).

순서: ingest(패널·적격성·거래일) → 자기자본 시점 붙이기(equity_dating, 한 번만) → 자기자본 as-of → H1·H3 → 기본 피처 → H4.
기간 끝은 교수님 CSV 마지막 달(src.periods final). 학습용 표(data/interim·processed)는 읽기만 한다.

1) 모든 표를 data/final/_staging/에 먼저 만든다.
2) 과거 값 자동 확인: 새 표의 겹치는 기간을 (a) 학습용 표(PATHS), (b) 이전 최종 표(data/final/, 있으면)와 비교한다.
   값 열이 다르면 멈춘다(어느 표·열·몇 행·예시). 허용은 config DATA_PERIOD.known_history_changes에 적힌 것만(학습용 표 비교에만 적용).
   설명 열(이유·근거·출처·공시 번호 등)은 달라도 되며 건수와 예시를 기록한다.
3) --execute이고 모두 통과하면: 이전 data/final/*.parquet을 data/interim/_backups/final_<시각>/에 복사한 뒤 새 표로 바꾸고
   outputs/final/run.json에 기록한다. --execute가 없으면 _staging에 만든 표와 비교 결과만 남긴다.
예측(⑩)은 모델 고정 뒤에 붙인다.

실행: python -m src.final_build              (만들고 비교만, data/final은 그대로. 약 40분)
      python -m src.final_build --execute    (통과하면 data/final 교체)
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import periods

TABLES = ("panel", "eligibility", "trading_days", "asof", "h1_components", "mom", "h1", "h3", "base_features", "h4")
STAGING = "data/final/_staging/"
RUN = Path("outputs/final/run.json")
BACKUPS = Path("data/interim/_backups")
# 설명 열(달라도 통과, 건수만 기록): 출처·이유·판정 근거만(7-4 ⑦ 결정, 2026-11-28 좁힘).
# 상장일·폐지일은 적격 판정 근거 — 새 KRX 목록이 들어오면 과거 행에도 채워진다(적격 여부 자체는 값으로 비교).
# 공시 번호·결산월·사용 시작월·만료일·재무제표 등급 등은 값으로 본다(다르면 멈춤, 허용은 known_history_changes만).
DESCRIPTION = re.compile(r"reason|basis|^source$|^listing_date$|^delisting_date$")


def keys_of(table: pd.DataFrame) -> list[str]:
    stock = "stock_code" if "stock_code" in table else "stock_id" if "stock_id" in table else None
    month = "base_month" if "base_month" in table else "month"
    return [k for k in (stock, month) if k]


def month_col(table: pd.DataFrame) -> str:
    return "base_month" if "base_month" in table else "month"


def differs(a: pd.Series, b: pd.Series) -> np.ndarray:
    """정확히 비교(같은 결측은 같음). 숫자는 값, 그 밖은 문자열 표현."""
    na, nb = a.isna().to_numpy(), b.isna().to_numpy()
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
        same = (a.to_numpy(float) == b.to_numpy(float))
    else:
        same = (a.astype(object).astype(str).to_numpy() == b.astype(object).astype(str).to_numpy())
    return ~((na & nb) | (~na & ~nb & same))


def allowed_mask(name: str, col: str, old: pd.DataFrame, new: pd.DataFrame, rules: list[dict],
                 expected: pd.DataFrame | None = None) -> np.ndarray:
    """허용 목록에 맞는 행. 규칙에 verify가 있으면 new 값이 expected(같은 키·순서로 다시 계산한 값)와 같을 때만 허용."""
    ok = np.zeros(len(old), bool)
    for r in rules:
        cols = r["tables"].get(name)
        if cols is None or (cols != "all" and col not in cols):
            continue
        m = np.ones(len(old), bool)
        if "base_month" in r:
            m &= old[month_col(old)].astype(str).eq(r["base_month"]).to_numpy()
        if "stocks" in r:
            m &= old[keys_of(old)[0]].astype(str).isin(r["stocks"]).to_numpy()
        if r.get("only") == "missing_to_false":
            m &= old[col].isna().to_numpy() & new[col].astype(object).eq(False).to_numpy()
        if r.get("verify"):
            if expected is None:
                m &= False
            else:
                m &= ~differs(new[col].reset_index(drop=True), expected[col].reset_index(drop=True))
        ok |= m
    return ok


def compare(name: str, old: pd.DataFrame, new: pd.DataFrame, rules: list[dict], expected: pd.DataFrame | None = None) -> dict:
    """old의 기간 안에서 new와 비교. 값 열 위반(허용 밖)과 설명 열 변화를 센다."""
    k, m = keys_of(old), month_col(old)
    new = new[new[m].astype(str) <= str(old[m].max())]
    if list(old.columns) != list(new.columns):
        return {"columns_differ": [list(old.columns), list(new.columns)], "violations": {"<columns>": 1}}
    old = old.sort_values(k).reset_index(drop=True)
    new = new.sort_values(k).reset_index(drop=True)
    if len(old) != len(new) or not (old[k].astype(str).to_numpy() == new[k].astype(str).to_numpy()).all():
        return {"keys_differ": [len(old), len(new)], "violations": {"<keys>": abs(len(old) - len(new)) or 1}}
    out = {"rows": len(old), "violations": {}, "allowed": {}, "description_changes": {}, "examples": {}}
    for col in old.columns:
        if col in k:
            continue
        d = differs(old[col], new[col])
        if not d.any():
            continue
        if DESCRIPTION.search(col):
            out["description_changes"][col] = int(d.sum())
            continue
        ok = allowed_mask(name, col, old, new, rules, expected) & d
        bad = d & ~ok
        if ok.any():
            out["allowed"][col] = int(ok.sum())
        if bad.any():
            out["violations"][col] = int(bad.sum())
            ex = old.loc[bad, k].head(3).astype(str).agg(":".join, axis=1).tolist()
            out["examples"][col] = [f"{e} {old.loc[i, col]!r}->{new.loc[i, col]!r}"
                                    for e, i in zip(ex, old.index[bad][:3])]
    return out


def staged_config(config: dict) -> dict:
    c = copy.deepcopy(config)
    c["DATA_PERIOD"]["final_dir"] = STAGING
    return c


def build_all(root: Path, config: dict, log=print) -> dict:
    """_staging에 표 10개를 만든다. 반환: 단계별 기록."""
    from src.pipeline import plan_features, plan_ingest
    from src.preprocessing import equity_asof
    from src.preprocessing.equity_dating import build as build_dated_equity
    from src.features.base import ensure_base, select_annual
    from src.features.h4 import ensure_h4
    c = staged_config(config)
    staging = root / STAGING
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    steps = {}

    def step(name, fn):
        t0 = time.perf_counter()
        log(f"[{datetime.now():%H:%M:%S}] {name} 시작")
        result = fn()
        steps[name] = {"seconds": round(time.perf_counter() - t0, 1)}
        log(f"[{datetime.now():%H:%M:%S}] {name} 끝 ({steps[name]['seconds']:.0f}초)")
        return result

    r = step("ingest", lambda: plan_ingest(c, root=root, execute=True, final=True))
    steps["ingest"].update(checks=r["checks"], months=r["summary"]["months"])
    dated = step("equity_dating", lambda: build_dated_equity(root, include_test=True))
    table, meta = step("asof", lambda: equity_asof.build(root, final=True, dated=dated))
    table.to_parquet(root / periods.table_path(c, "asof", final=True), index=False)
    steps["asof"].update(checks=meta["checks"], rows=meta["rows"], months=meta["months"])
    r = step("features_h1_h3", lambda: plan_features(c, root=root, execute=True, final=True))
    steps["features_h1_h3"].update(checks=r["checks"], summary=r["summary"])
    statements = select_annual(dated[0])
    del dated
    r = step("base_features", lambda: ensure_base(root, c, execute=True, statements=statements, include_test=True, log=log))
    steps["base_features"].update(rows=r["rows"], base_months=r["base_months"], inputs_sha256=r["inputs_sha256"])
    r = step("h4", lambda: ensure_h4(root, c, final=True, execute=True))
    steps["h4"].update(rows=r["rows"], base_months=r["base_months"])
    return steps


def check_history(root: Path, config: dict) -> dict:
    """_staging 표를 학습용 표(예외 허용)·이전 최종 표(예외 없음)와 비교."""
    c = staged_config(config)
    rules = config["DATA_PERIOD"].get("known_history_changes") or []
    out = {}
    expected = {}
    if any(r.get("verify") == "h1_from_patched_components" for r in rules):
        expected["h1"] = h1_expected(root, config, c, rules)
    for name in TABLES:
        new = pd.read_parquet(root / periods.table_path(c, name, final=True))
        res = {"vs_development": compare(name, pd.read_parquet(root / periods.table_path(config, name)), new, rules,
                                         expected.get(name))}
        previous = root / periods.table_path(config, name, final=True)
        if previous.is_file():
            res["vs_previous_final"] = compare(name, pd.read_parquet(previous), new, [])
        out[name] = res
    return out


def h1_expected(root: Path, config: dict, staged: dict, rules: list[dict]) -> pd.DataFrame:
    """학습용 H1 구성요소에서 허용 목록의 행(자기자본 예외 종목·월)과 적격성만 새 값으로 바꿔 H1 점수를 다시 계산한다.
    새 H1 점수가 이것과 같으면, 바뀐 이유가 허용된 입력 변화뿐임이 확인된다(같은 달 표준화로 퍼진 변화)."""
    from src.features.h1 import calculate_h1_score
    old = pd.read_parquet(root / periods.table_path(config, "h1_components"))
    new = pd.read_parquet(root / periods.table_path(staged, "h1_components", final=True))
    k = ["stock_id", "base_month"]
    new = new[new.base_month <= old.base_month.max()].set_index(k).reindex(pd.MultiIndex.from_frame(old[k]))
    patched = old.set_index(k).copy()
    patched["eligible"] = new["eligible"].to_numpy()
    for r in rules:
        cols = r["tables"].get("h1_components")
        if not cols or "stocks" not in r:
            continue
        rows = patched.index.get_level_values(0).isin(r["stocks"]) & (patched.index.get_level_values(1) == r["base_month"])
        for col in (cols if cols != "all" else patched.columns):
            for c in (col, f"{col}_available", f"{col}_exclusion_reason"):
                if c in patched:
                    patched.loc[rows, c] = new.loc[rows, c].to_numpy()
    patched = patched.reset_index()
    return calculate_h1_score(patched)


def violations(history: dict) -> list[str]:
    return [f"{t}/{side}/{col}:{n}" for t, r in history.items() for side, v in r.items()
            for col, n in v.get("violations", {}).items()]


def replace_final(root: Path, config: dict, stamp: str) -> list[str]:
    final_dir = root / config["DATA_PERIOD"]["final_dir"]
    backup = root / BACKUPS / f"final_{stamp}"
    old = sorted(final_dir.glob("*.parquet"))
    if old:
        backup.mkdir(parents=True)
        for f in old:
            shutil.copy2(f, backup / f.name)
    for f in sorted((root / STAGING).glob("*.parquet")):
        shutil.move(str(f), final_dir / f.name)
    shutil.rmtree(root / STAGING)
    return [p.name for p in old]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--execute", action="store_true", help="과거 값 확인을 통과하면 data/final 교체")
    ap.add_argument("--compare-only", action="store_true", help="이미 만든 _staging 표로 비교만(개발·점검용)")
    a = ap.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / a.config).read_text(encoding="utf-8"))
    started = datetime.now()
    stamp = started.strftime("%Y%m%dT%H%M%S")
    period = periods.resolve(config, root, final=True)
    record = {"started": started.isoformat(timespec="seconds"), "command": " ".join(sys.argv), "execute": a.execute,
              "data_end": period.end, "last_base_month": period.last_base_month}
    if not a.compare_only:
        record["steps"] = build_all(root, config)
    record["history"] = check_history(root, config)
    bad = violations(record["history"])
    record["status"] = "HISTORY_CHANGED" if bad else ("WRITTEN" if a.execute else "STAGED_NOT_WRITTEN")
    if bad:
        print("과거 값이 바뀌어 멈춤:", *bad, sep="\n  ")
    elif a.execute:
        record["replaced"] = replace_final(root, config, stamp)
        record["backup"] = (BACKUPS / f"final_{stamp}").as_posix() if record["replaced"] else None
    (root / RUN).parent.mkdir(parents=True, exist_ok=True)
    (root / RUN).write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    summary = {t: {s: {k: v for k, v in r.items() if k in ("violations", "allowed", "description_changes")}
                   for s, r in res.items()} for t, res in record["history"].items()}
    print(json.dumps({"status": record["status"], "data_end": period.end, "history": summary}, ensure_ascii=False, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
