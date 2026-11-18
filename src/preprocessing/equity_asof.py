"""Stock x base-month equity as-of for B/M: the value public by each month's end (C1, 2026-11-04).

Input: DataGuide workbooks and the two common DART observation tables.
Principle: the value used in a month is the number that was public by that month's end.
- Versions (equity_dating.correction_versions) give, per company period, which value counts from which month (P4,
  corrections from their receipt, P1). A version whose value comes from DART must pass the DART rules the DART-only
  as-of applied: approved manual overrides and same-filing unit corrections first, then no unconfirmed statement date,
  no unit suspect, no failed owners identity. A failing DART version falls back to the DataGuide value from the first
  receipt (marked possibly_corrected) or, for a later correction, is dropped.
- Month t: among versions usable by t, the newest balance sheet month wins (P5, P2 carries the previous period);
  inside it the latest version. P3: the value expires after the month that holds statement date + 18 months
  (compared with that base month's last trading day).
- No DataGuide value for the month: the DART-only as-of (dart_asof.join_equity_asof, same calendar and rules) fills
  the month when the stock has a market cap that month (user decision 2026-11-04), source dart_only.
- Equity <= 0 stays in the table and is not usable for B/M (nonpositive_equity).
Every row carries a value or a reason. Checks stop the run on: a value used before its usable month, a used value past
its expiry, a duplicate stock-month, a row with neither value nor reason.
No network; DART originals are not read here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from src.preprocessing.equity_dating import (
    MARCAP, OVERRIDES, PANEL_NAMES, last_trading_days, load_overrides, build as build_dated_equity)
from src.preprocessing.dataguide_extract import (
    sha256, statement_month)
from src.preprocessing.dart_extract import (
    apply_overrides, apply_unit_corrections, join_equity_asof,
    DART_FILINGS, DART_FACTS, load_common_dart, public_filings, decisions_from_facts)


OUTPUT = Path("outputs/run_id/EQUITY_ASOF")


INTERIM = Path("data/interim/asof.parquet")


MONTHS = ("2000-01", "2021-11")


MARKET_CAP = (Path("data/raw/warmup_2000__raw_month_end_market_cap.csv"), Path("data/raw/train_2001_2021__raw_month_end_market_cap.csv"))


DART_SOURCES = {"dart_first_before_correction", "dart_first_differs", "dart_correction"}


VALIDITY_MONTHS = 18


COLUMNS = ["stock_code", "base_month", "equity", "tier", "statement_month", "statement_date", "source", "rcept_no", "rcept_dt",
           "available_month", "expiry_date", "equity_age_months", "possibly_corrected", "market_cap_present", "usable", "reason"]


def dart_decisions(root: Path) -> tuple[pd.DataFrame, dict]:
    """DART filing decisions after the approved overrides and the same-filing unit corrections, with a skip reason."""
    from src.preprocessing.dart_extract import apply_overrides, apply_unit_corrections
    metadata, facts = load_common_dart(root)
    filings = decisions_from_facts(metadata, facts)
    filings = filings.drop_duplicates("receipt_id").reset_index(drop=True)
    filings, counts = apply_overrides(root, filings)
    filings, corrected = apply_unit_corrections(filings, facts)
    given = lambda v: v is not None and not pd.isna(v) and str(v) != ""
    skip = []
    for r in filings.itertuples():
        skip.append("no_value" if not given(r.equity) else "fiscal_date_unverified" if not given(r.fiscal_period)
                    else "unit_suspect" if str(r.unit_suspect) == "True"
                    else "owners_identity_failed" if str(r.owners_identity_ok) == "False" and not given(r.owners_resolution)
                    else None)
    filings["skip"] = skip
    return filings, {"overrides": counts, "unit_corrections": len(corrected)}


def checked_versions(versions: pd.DataFrame, decisions: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """DART-sourced versions pass the DART rules or fall back (see module doc)."""
    by_receipt = decisions.set_index("receipt_id")
    dated = panel[panel.rcept_no.notna()]  # unreliable fiscal-month rows share a month but carry no receipt
    first_value = dated.set_index(["stock_code", "month"])[["equity", "tier", "rcept_no", "rcept_dt", "available_month"]]
    rows, counts = [], {"dart_versions": 0, "dart_value_replaced_by_override": 0, "fell_back_to_dataguide": 0, "correction_dropped": 0}
    for (code, month), g in versions.sort_values("seq").groupby(["stock_code", "month"], sort=False):
        fell_back = False
        for v in g.to_dict("records"):
            if v["source"] in DART_SOURCES:
                counts["dart_versions"] += 1
                d = by_receipt.loc[v["rcept_no"]] if v["rcept_no"] in by_receipt.index else None
                if d is not None and pd.isna(d["skip"]):  # stored as NA, not None, in string columns
                    if pd.notna(d["override_id"]) and float(d["equity"]) != v["equity"]:
                        counts["dart_value_replaced_by_override"] += 1
                    v.update(equity=float(d["equity"]), tier=d["equity_tier"])
                elif v["source"] == "dart_correction":
                    counts["correction_dropped"] += 1
                    continue
                else:
                    counts["fell_back_to_dataguide"] += 1
                    base = first_value.loc[(code, month)]
                    v.update(equity=float(base["equity"]), tier=base["tier"], rcept_no=base["rcept_no"], rcept_dt=base["rcept_dt"],
                             available_month=base["available_month"], source="dataguide_possibly_corrected")
                    fell_back = True
            elif fell_back and v["source"] == "dataguide_from_correction":
                continue  # the same DataGuide value is already in use from the first receipt
            rows.append(v)
    out = pd.DataFrame(rows)
    out["seq"] = out.groupby(["stock_code", "month"]).cumcount()
    return out, counts


def expiry_date(statement_month: str) -> pd.Timestamp:
    """P3: statement date + 18 months, through the end of that month."""
    return (pd.Period(statement_month, "M") + VALIDITY_MONTHS).to_timestamp(how="end").normalize()


def build_asof(versions: pd.DataFrame, stocks: list[str], months: list[str], last_days: pd.Series) -> pd.DataFrame:
    """DataGuide-based as-of; rows without a value carry the reason."""
    by_stock = {c: g.sort_values(["available_month", "month", "seq"]).to_dict("records") for c, g in versions.groupby("stock_code")}
    last = {str(k): v for k, v in last_days.items()}
    rows = []
    for code in stocks:
        items = by_stock.get(code)
        latest, i = {}, 0  # balance sheet month -> its newest version usable so far
        for t in months:
            row = dict(stock_code=code, base_month=t)
            if items is None:
                rows.append({**row, "reason": "no_dataguide_value"})
                continue
            while i < len(items) and items[i]["available_month"] <= t:
                cur = latest.get(items[i]["month"])
                if cur is None or items[i]["seq"] > cur["seq"]:
                    latest[items[i]["month"]] = items[i]
                i += 1
            if not latest:
                rows.append({**row, "reason": "before_first_public_value"})
                continue
            period = max(latest)
            v = pd.Series(latest[period])
            expiry = expiry_date(period)
            end = pd.Period(period, "M").to_timestamp(how="end").normalize()
            row.update(equity=v.equity, tier=v.tier, statement_month=period, statement_date=end.strftime("%Y-%m-%d"),
                       source=v.source, rcept_no=v.rcept_no, rcept_dt=v.rcept_dt, available_month=v.available_month,
                       expiry_date=expiry.strftime("%Y-%m-%d"),
                       equity_age_months=(pd.Period(t, "M") - pd.Period(period, "M")).n,
                       possibly_corrected=v.source == "dataguide_possibly_corrected")
            if last.get(t, pd.Period(t, "M").to_timestamp(how="end")) > expiry:
                row = {k: row[k] for k in ("stock_code", "base_month")} | {"reason": "expired_18m", "statement_month": period}
            rows.append(row)
    return pd.DataFrame(rows)


def dart_fill(root: Path, decisions: pd.DataFrame, need: pd.DataFrame, corp_of: dict[str, str], last_days: pd.Series) -> pd.DataFrame:
    """DART-only as-of for the stock-months without a DataGuide value (same rules as the earlier DART as-of)."""
    from src.preprocessing.dart_extract import join_equity_asof
    panel = need[["stock_code", "base_month"]].rename(columns={"stock_code": "stock_id"})
    panel["corp_code"] = panel.stock_id.map(corp_of)
    panel["mapping_status"] = panel.corp_code.notna().map({True: "current_exact", False: "unmapped"})
    calendar = {str(k): v.strftime("%Y%m%d") for k, v in last_days.items()}  # the same P4 calendar as the DataGuide side
    usable = decisions.drop(columns="skip")
    table = join_equity_asof(panel, usable[usable.corp_code.isin(set(corp_of.values()))], last_trading_day=calendar)
    return table[table.usable.fillna(False)]


def build(root: Path) -> tuple[pd.DataFrame, dict]:
    panel, versions, _, dating_meta = build_dated_equity(root)
    panel = panel[panel.panel]
    decisions, dart_counts = dart_decisions(root)
    versions, version_counts = checked_versions(versions, decisions, panel)
    stocks = []
    for p in PANEL_NAMES:
        with (root / p).open(encoding="utf-8-sig", newline="") as stream:
            stocks += [c for c in next(csv.reader(stream))[1:] if c not in stocks]
    months = pd.period_range(*MONTHS, freq="M").astype(str).tolist()
    dates = pd.concat(pd.read_parquet(p, columns=["Date"]).Date for p in [root / MARCAP / f"marcap-{year}.parquet" for year in range(2000,2022)])
    last_days = last_trading_days(dates)
    table = build_asof(versions, stocks, months, last_days)
    caps = []
    for p in MARKET_CAP:
        wide = pd.read_csv(root / p, dtype=str, encoding="utf-8-sig")
        wide = wide.set_index(wide.columns[0]).apply(pd.to_numeric, errors="coerce")
        wide.index = pd.to_datetime(wide.index).to_period("M").astype(str)
        caps.append(wide)
    cap = pd.concat(caps).stack()
    present = set(cap[cap > 0].index)
    table["market_cap_present"] = [(t, c) in present for c, t in zip(table.stock_code, table.base_month)]
    # DART-only fill (user decision 2026-11-04): only where DataGuide has no value and the stock has a market cap.
    metadata, _ = load_common_dart(root)
    filings = public_filings(metadata)
    corp_of = filings[filings.stock_code.ne("")].drop_duplicates("stock_code").set_index("stock_code").corp_code.to_dict()
    corp_of.update({e["stock_code"]: e["set"]["corp_code"] for e in load_overrides(root / OVERRIDES) if e["type"] == "corp_map"})
    need = table[table.equity.isna() & table.market_cap_present]
    filled = dart_fill(root, decisions, need, corp_of, last_days)
    fill = filled.set_index(["stock_id", "base_month"])
    hit = [(c, t) in fill.index for c, t in zip(table.stock_code, table.base_month)]
    for i in table.index[hit]:
        f = fill.loc[(table.at[i, "stock_code"], table.at[i, "base_month"])]
        statement = str(f.fiscal_period_end)[:7]
        table.loc[i, ["equity", "tier", "statement_month", "statement_date", "source", "rcept_no", "rcept_dt", "available_month",
                      "expiry_date", "equity_age_months", "possibly_corrected", "reason"]] = [
            float(f.equity), f.equity_tier, statement, str(f.fiscal_period_end)[:10], "dart_only", f.receipt_id, f.published_at,
            f.availability_month, f"{f.expiry_date[:4]}-{f.expiry_date[4:6]}-{f.expiry_date[6:]}", int(f.equity_age_months), False, None]
    table["usable"] = table.equity.notna() & table.equity.gt(0)
    table.loc[table.equity.notna() & ~table.usable, "reason"] = "nonpositive_equity"
    table = table.reindex(columns=COLUMNS)
    checks = run_checks(table, last_days)
    meta = summarize(root, table, versions, version_counts, dart_counts, checks)
    meta["dating"] = dating_meta
    return table, meta


def run_checks(table: pd.DataFrame, last_days: pd.Series) -> dict:
    used = table[table.equity.notna()]
    last = {str(k): v for k, v in last_days.items()}
    future = int((used.available_month > used.base_month).sum())
    expired = int(sum(last.get(t, pd.Timestamp.max) > pd.Timestamp(e) for t, e in zip(used.base_month, used.expiry_date)))
    statement_after = int((used.statement_month > used.base_month).sum())
    duplicate = int(table.duplicated(["stock_code", "base_month"]).sum())
    empty = int((table.equity.isna() & table.reason.isna()).sum())
    checks = {"value_before_usable_month": future, "value_past_expiry": expired, "statement_after_base_month": statement_after,
              "duplicate_stock_month": duplicate, "row_without_value_or_reason": empty}
    if any(checks.values()):
        raise ValueError("asof_checks_failed:" + json.dumps(checks))
    return checks


def summarize(root: Path, table: pd.DataFrame, versions: pd.DataFrame, version_counts: dict, dart_counts: dict,
              checks: dict) -> dict:
    cap = table[table.market_cap_present]
    year = table.base_month.str[:4]
    by_year = pd.DataFrame({"rows": table.groupby(year).size(), "usable": table.groupby(year).usable.mean().round(4),
                            "cap_rows": cap.groupby(cap.base_month.str[:4]).size(),
                            "cap_usable": cap.groupby(cap.base_month.str[:4]).usable.mean().round(4)})
    inputs = {p.as_posix(): sha256(root / p) for p in (DART_FILINGS, DART_FACTS)}
    for path in (Path("config/equity_overrides.json"), OVERRIDES):
        inputs[path.as_posix()] = sha256(root / path)
    return {"inputs_sha256": inputs, "months": list(MONTHS), "rows": len(table),
            "stocks": int(table.stock_code.nunique()), "usable": int(table.usable.sum()),
            "usable_share": round(float(table.usable.mean()), 4),
            "market_cap_rows": len(cap), "market_cap_usable": int(cap.usable.sum()),
            "market_cap_usable_share": round(float(cap.usable.mean()), 4),
            "source": table.source.value_counts().to_dict(), "tier": table.tier.value_counts().to_dict(),
            "reason": table.reason.value_counts().to_dict(),
            "reason_with_market_cap": cap.reason.value_counts().to_dict(),
            "possibly_corrected_rows": int(table.possibly_corrected.fillna(False).astype(bool).sum()),
            "version_checks": version_counts, "dart": dart_counts, "checks": checks,
            "by_year": by_year.reset_index(names="year").to_dict("records"),
            "content_hash": str(int(pd.util.hash_pandas_object(table, index=False).sum())), "rules": __doc__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Write outputs/run_id/EQUITY_ASOF/<stamp>/ and data/interim/asof.parquet")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    table, meta = build(root)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("rules", "by_year", "inputs_sha256")}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(meta["by_year"]).to_string(index=False))
    if not args.execute:
        print("DRY-RUN: nothing written.")
        return
    out = root / OUTPUT / datetime.now().strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True)
    table.to_parquet(out / "asof.parquet", index=False)
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / INTERIM).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(root / INTERIM, index=False)
    print("saved:", out.relative_to(root).as_posix(), "and", INTERIM.as_posix())


if __name__ == "__main__":
    main()
