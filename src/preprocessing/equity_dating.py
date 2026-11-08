"""Stage 5, public dates and checks for the DataGuide values (former dataguide_equity.py, minus the workbook reading).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing.dataguide_extract import (
    NO_VALUE, RAW, SOURCES, TIERS, combine, data_sheet, read_sheet, sha256)
from src.preprocessing.dart_extract import (
    _decode_document, DART_FILINGS, DART_FACTS, load_common_dart, public_filings, decisions_from_facts)


OUTPUT = Path("outputs/run_id/DATAGUIDE_EQUITY")


PANEL_NAMES = (Path("data/raw/warmup_2000__company_name.csv"), Path("data/raw/train_2001_2021__company_name.csv"))


MARCAP = Path("data/raw/marcap")


OVERRIDES = Path("config/dataguide_overrides.json")


# Normal filing window after a period end: the 99.9th percentile of first receipts matched by stock code in
# DATAGUIDE_EQUITY/20261104T011258 (분기 35,114 / 반기 18,368 / 사업 17,742 receipts; medians 45 / 45 / 90 days).
FILING_WINDOW_DAYS = {"분기": 63, "반기": 63, "사업": 123}


def normalize_name(name: str) -> str:
    return re.sub(r"\(주\)|㈜|주식회사|\s", "", name or "")


PRINTING_UNITS = (1e6, 1e5, 1e4, 1e3, 100.0, 10.0)


def printing_unit(x: float) -> float:
    """The unit the amount was rounded to, read from its trailing zeros (1 won up to 1 million won).

    DART tables print in won, thousands or millions; DataGuide K-IFRS values are often rounded to 10 won. An amount
    that happens to end in zeros gets a coarser unit than printed, which only widens the allowance by that unit.
    """
    return next((u for u in PRINTING_UNITS if x % u == 0), 1.0)


def printing_units(values: np.ndarray) -> np.ndarray:
    units = np.ones(len(values))
    for u in reversed(PRINTING_UNITS):  # coarsest match wins
        units = np.where(values % u == 0, u, units)
    return units


def same_amount(a: float, b: float) -> bool:
    """Equal once both are read in the coarser printing unit: the gap is at most one such unit (one side rounded, the
    other cut, can land one unit apart).

    DART tables print in won, thousands or millions and DataGuide mostly in thousands, so 1,718,662 thousand won and
    1,718 million won are the same amount, while two amounts printed in won must match to the won.
    """
    return abs(a - b) <= max(printing_unit(a), printing_unit(b))


def period_end(month: str) -> pd.Timestamp:
    return pd.Period(month, "M").to_timestamp(how="end").normalize()


def load_overrides(path: Path, approved_only: bool = True) -> list[dict]:
    """Entries of config/dataguide_overrides.json (each carries its evidence and who decided)."""
    if not path.exists():
        return []
    return [e for e in json.loads(path.read_text(encoding="utf-8")) if e.get("status") == "approved" or not approved_only]


def _rename_key(name: str) -> str:
    """2002 renaming of 상호신용금고 to 상호저축은행: both read as one name."""
    return normalize_name(name).replace("상호", "").replace("신용금고", "저축은행")


def override_candidates(panel: pd.DataFrame, filings: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """How the entries of config/dataguide_overrides.json are found. The decision on each stays with the user.

    corp_link, for a stock whose DART rows carry no stock code:
      amount - one DART company without a stock code of its own holds DataGuide amounts of this stock at the same
               balance sheet month in at least two periods, and no other such company holds any;
      rename - read with 상호신용금고 = 상호저축은행 (the 2002 renaming), the name names exactly one DART company.
    statement_month, for a value without a receipt or with a month override: its amount is in an original of its
      own company at exactly one other balance sheet month (DataGuide's fiscal month is wrong); free when no other
      dated value of the stock holds that month, else duplicate.
    facts: corp_code, receipt_id, v (amount), fpm (balance sheet month of the fact).
    """
    listed = set(filings.loc[filings.stock_code.ne(""), "corp_code"])
    coded = set(filings.stock_code)
    corp_of_code = filings[filings.stock_code.ne("")].drop_duplicates("stock_code").set_index("stock_code").corp_code
    names = filings.assign(k=filings.corp_name.map(_rename_key)).groupby("k").corp_code.agg(lambda s: sorted(set(s)))
    f = facts.dropna(subset=["v", "fpm"]).sort_values("v")
    values = f.v.to_numpy()

    def holders(v):
        lo, hi = values.searchsorted(v - 1e6), values.searchsorted(v + 1e6, side="right")
        return f.iloc[lo:hi][[same_amount(v, a) for a in values[lo:hi]]]

    def dg_values(r):
        return [x for x in (r.equity, r.sep, r.cons, r.owners) if pd.notna(x) and abs(x) >= 1e8]

    rows = []
    for code, g in panel[~panel.stock_code.isin(coded)].groupby("stock_code"):
        months = {}
        for r in g.itertuples():
            for v in dg_values(r):
                for h in holders(v).itertuples():
                    if h.fpm == r.month_dataguide and h.corp_code not in listed:
                        months.setdefault(h.corp_code, set()).add(r.month_dataguide)
        strong = [c for c, m in months.items() if len(m) >= 2]
        if len(strong) == 1 and len(months) == 1:
            rows.append(dict(type="corp_link", stock_code=code, corp_code=strong[0], basis="amount",
                             evidence=",".join(sorted(months[strong[0]]))))
        elif len(names.get(_rename_key(g.name.iloc[0]), [])) == 1 and set(months) <= set(names[_rename_key(g.name.iloc[0])]):
            rows.append(dict(type="corp_link", stock_code=code, corp_code=names[_rename_key(g.name.iloc[0])][0], basis="rename",
                             evidence=g.name.iloc[0]))
    links = {r["stock_code"]: r["corp_code"] for r in rows}
    # Per company arrays: amount, its printing unit and balance sheet month (same rule as same_amount, vectorized).
    by_corp = {c: (g.v.to_numpy(), printing_units(g.v.to_numpy()), g.fpm.to_numpy()) for c, g in f.groupby("corp_code")}
    dated = set(zip(panel.loc[panel.rcept_no.notna(), "stock_code"], panel.loc[panel.rcept_no.notna(), "month"]))
    look = panel[panel.rcept_no.isna() | panel.month_note.notna()]
    for r in look.itertuples():
        corp = corp_of_code.get(r.stock_code) or links.get(r.stock_code)
        if corp not in by_corp:
            continue
        amounts, units, months = by_corp[corp]
        hit = np.zeros(len(amounts), dtype=bool)
        for v in dg_values(r):
            hit |= np.abs(amounts - v) <= np.maximum(units, printing_unit(v))
        other = set(months[hit & (months != r.month_dataguide)])
        if len(other) == 1:
            target = next(iter(other))
            free = (r.stock_code, target) not in dated or r.month == target
            rows.append(dict(type="statement_month", stock_code=r.stock_code, fy=str(r.fy), quarter=r.quarter,
                             month_dataguide=r.month_dataguide, corp_code=corp, basis="free" if free else "duplicate",
                             evidence=target))
    return pd.DataFrame(rows, columns=["type", "stock_code", "fy", "quarter", "month_dataguide", "corp_code", "basis", "evidence"])


def verify_overrides(entries: list[dict], candidates: pd.DataFrame) -> list[str]:
    """Every approved entry must meet its stated criterion again; a rejected link that now meets one is reported."""
    links = {(r.stock_code, r.corp_code, r.basis) for r in candidates[candidates.type.eq("corp_link")].itertuples()}
    months = {(r.stock_code, r.fy, r.quarter): (r.basis, r.evidence) for r in candidates[candidates.type.eq("statement_month")].itertuples()}
    problems = []
    for e in entries:
        approved = e.get("status") == "approved"
        if e["type"] == "corp_map":
            met = (e["stock_code"], e["set"]["corp_code"], e["basis"]) in links
            if approved != met:
                problems.append(f"{e['id']}:{'criterion_not_met' if approved else 'rejected_but_criterion_met'}")
        elif approved:
            got = months.get((e["stock_code"], str(e["fy"]), e["quarter"]))
            if e["type"] == "statement_month" and got != ("free", e["set"]["month"]):
                problems.append(f"{e['id']}:criterion_not_met")
            if e["type"] == "statement_month_unreliable" and got is not None and got[0] == "free":
                problems.append(f"{e['id']}:correctable")
    return problems


def apply_month_overrides(panel: pd.DataFrame, overrides: list[dict]) -> pd.DataFrame:
    """B5 ④ (user decision 2026-11-03): DataGuide's fiscal month is wrong for a few company-years.

    A value moves only to the balance sheet date of a DART original of the same company holding the same amount
    (statement_month); the other values of those company-years are marked and not used (statement_month_unreliable).
    """
    panel = panel.assign(month_dataguide=panel.month, month_note=None)
    for e in overrides:
        if e["type"] not in ("statement_month", "statement_month_unreliable"):
            continue
        hit = panel.stock_code.eq(e["stock_code"]) & panel.fy.astype(str).eq(str(e["fy"])) & panel.quarter.eq(e["quarter"])
        if hit.sum() != 1 or panel.loc[hit, "month"].iloc[0] != e["month_dataguide"]:
            raise ValueError(f"dataguide_override_stale:{e['id']}")
        if e["type"] == "statement_month":
            panel.loc[hit, ["month", "month_note"]] = [e["set"]["month"], "corrected:" + e["id"]]
        else:
            panel.loc[hit, "month_note"] = "unreliable:" + e["id"]
    usable = panel[~panel.month_note.fillna("").str.startswith("unreliable")]
    if usable.duplicated(["code", "month"]).any():
        raise ValueError("dataguide_month_override_collides")
    return panel


GROUP = ["corp_code", "report_period_label", "report_kind", "label_ok"]


def label_checked(filings: pd.DataFrame) -> pd.DataFrame:
    """label_ok: the filing was received after the period in its title ended. One received earlier cannot be that
    period's report (e.g. KT '사업보고서 (2001.12)' received 2001-03-31 is the 2000 report), so it never joins that
    period's group; it can only serve the earlier period through label_after_receipt."""
    if "label_ok" in filings:
        return filings
    label = filings.report_period_label.astype("string").fillna("")
    ends = pd.Series([period_end(x) if re.fullmatch(r"\d{4}-\d{2}", x) else pd.NaT for x in label], index=filings.index)
    return filings.assign(label_ok=ends.isna() | (pd.to_datetime(filings.rcept_dt) > ends))


def attach_receipts(panel: pd.DataFrame, filings: pd.DataFrame, amounts: dict[str, list[float]] | None = None,
                    corp_map: dict[str, str] | None = None) -> pd.DataFrame:
    """First DART receipt of the same company, period label and report kind.

    Stock code first. DART drops the stock code of delisted firms, so a normalized name is used only when it
    names exactly one DART company and one DataGuide company (user decision 2026-11-03).
    Fallbacks inside one DART company (B5 decisions 2026-11-03):
    - corp_map: approved links for delisted firms whose name is shared (config/dataguide_overrides.json).
    - label_after_receipt: an early report whose title period is later than its own receipt (impossible) is the
      report of the DataGuide period when it has the same kind and was received within the normal filing window
      (FILING_WINDOW_DAYS) after that period.
    - same_date_other_kind: a fiscal-year change makes one balance sheet date a 분기 in one source and a 반기 in the
      other; accepted only within that report's filing window and when the DART original holds the same amount
      (amounts: receipt -> extracted values).
    A value left without a receipt gets no_receipt_reason.
    """
    amounts, corp_map = amounts or {}, corp_map or {}
    group = GROUP
    filings = label_checked(filings).assign(n=filings.corp_name.map(normalize_name)).sort_values(["rcept_dt", "rcept_no"])
    first = filings.drop_duplicates(group).copy()
    # Every version received on the first day was public that day (e.g. an original and its attachment addition).
    first_day = filings.merge(first[group + ["rcept_dt"]], on=group + ["rcept_dt"])
    first["same_day_receipts"] = first.set_index(group).index.map(first_day.groupby(group).rcept_no.agg(";".join))
    first["rd"] = pd.to_datetime(first.rcept_dt)
    first["rm"] = first.rd.dt.to_period("M")
    first["window"] = pd.to_timedelta(first.report_kind.map(FILING_WINDOW_DAYS).fillna(max(FILING_WINDOW_DAYS.values())), unit="D")
    label = first.report_period_label.astype("string").fillna("")
    first["lm"] = pd.PeriodIndex([x if re.fullmatch(r"\d{4}-\d{2}", x) else None for x in label], freq="M")
    list_end = first.rd.max()
    cols = ["rcept_dt", "rcept_no", "corp_code", "same_day_receipts"]
    ok = first[first.label_ok]  # a filing received before its title period ended cannot be that period's report
    by_code = ok[ok.stock_code.ne("")].drop_duplicates(["stock_code", "report_period_label", "report_kind"]).set_index(
        ["stock_code", "report_period_label", "report_kind"])[cols]
    corp_of_code = first[first.stock_code.ne("")].drop_duplicates("stock_code").set_index("stock_code").corp_code
    one_dart = filings.groupby("n").corp_code.nunique()
    panel = panel.assign(n=panel.name.map(normalize_name))
    one_dg = panel.groupby("n").code.nunique()
    nameless = ok[ok.stock_code.eq("") & ok.n.map(one_dart).eq(1)]
    corp_of_name = first[first.stock_code.eq("") & first.n.map(one_dart).eq(1)].drop_duplicates("n").set_index("n").corp_code
    nameless = nameless.set_index(["n", "report_period_label", "report_kind"])[cols]
    by_corp = dict(tuple(first.groupby("corp_code")))
    dg_values = [panel[c] if c in panel else pd.Series(float("nan"), index=panel.index) for c in ("sep", "cons", "owners")]

    def fallback(corp, m, k, values):
        fc = by_corp.get(corp)
        if fc is None:
            return None, "no_dart_company"
        p, end = pd.Period(m, "M"), period_end(m)
        due = end + pd.Timedelta(days=FILING_WINDOW_DAYS[k])
        hit = fc[fc.label_ok & (fc.lm == p) & (fc.report_kind == k)]
        if len(hit):
            return hit, "same_company"
        hit = fc[~fc.label_ok & (fc.report_kind == k) & (fc.rd > end) & (fc.rd <= due)]
        if len(hit):
            return hit, "label_after_receipt"
        other = fc[fc.label_ok & (fc.lm == p) & (fc.report_kind != k) & (fc.rd > end) & (fc.rd <= end + fc.window)]
        hit = other[[any(same_amount(v, a) for r in s.split(";") for a in amounts.get(r, []) for v in values) for s in other.same_day_receipts]]
        if len(hit):
            return hit, "same_date_other_kind"
        if len(other):
            # e.g. 2013 fiscal-year changers: DataGuide repeats the September value as the June value.
            return None, "same_date_other_kind_amount_differs"
        if due > list_end:
            return None, "receipt_after_dart_list_end"
        if fc.rd.min() > due:
            return None, "dart_filings_start_later"
        if fc.rd.max() <= due:
            return None, "dart_filings_end_before"
        return None, "no_regular_report_for_period"

    found = []
    for i, (c, n, m, k) in enumerate(zip(panel.stock_code, panel.n, panel.month, panel.kind)):
        if (c, m, k) in by_code.index:
            found.append((*by_code.loc[(c, m, k)], "stock_code", None))
            continue
        if one_dg.get(n, 0) == 1 and (n, m, k) in nameless.index:
            found.append((*nameless.loc[(n, m, k)], "unique_name", None))
            continue
        corp = corp_of_code.get(c) or (corp_of_name.get(n) if one_dg.get(n, 0) == 1 else None) or corp_map.get(c)
        if corp is None:
            found.append((None,) * 5 + ("dart_name_shared" if one_dart.get(n, 0) > 1 or one_dg.get(n, 0) > 1 else "no_dart_company",))
            continue
        values = [float(s.iat[i]) for s in dg_values if pd.notna(s.iat[i])]
        hit, basis = fallback(corp, m, k, values)
        if hit is None:
            found.append((None,) * 5 + (basis,))
        else:
            if basis == "same_company":
                basis = "corp_map" if c not in corp_of_code and corp_map.get(c) == corp else "same_company_unlisted_row"
            row = hit.sort_values(["rcept_dt", "rcept_no"]).iloc[0]
            found.append((row.rcept_dt, row.rcept_no, row.corp_code, row.same_day_receipts, basis, None))
    panel[cols + ["receipt_basis", "no_receipt_reason"]] = pd.DataFrame(found, index=panel.index)
    return panel.drop(columns="n")


def apply_kgaap_consolidated_rule(panel: pd.DataFrame, receipts_with_cfs: set[str]) -> pd.DataFrame:
    panel = panel.assign(tier_dataguide=panel.tier, equity_dataguide=panel.equity, kgaap_consolidated_in_report=None)
    consolidated = panel["std"].eq("KGAAP") & panel.tier.ne(TIERS[3])
    proven = panel.same_day_receipts.fillna("").map(lambda s: any(r in receipts_with_cfs for r in s.split(";") if r))
    panel.loc[consolidated, "kgaap_consolidated_in_report"] = proven[consolidated].map({True: "yes", False: "no"})
    demote = consolidated & ~proven
    has_sep = panel.sep.notna()
    panel.loc[demote & has_sep, "tier"] = TIERS[3]
    panel.loc[demote & has_sep, "equity"] = panel.loc[demote & has_sep, "sep"]
    panel.loc[demote & ~has_sep, "tier"] = NO_VALUE
    panel.loc[demote & ~has_sep, "equity"] = float("nan")
    return panel


def identity_check(panel: pd.DataFrame) -> pd.Series:
    """B2: assets - liabilities = the statement's equity total (separate for T3, consolidated otherwise). Never edits a value."""
    out = []
    for r in panel.itertuples():
        side = "sep" if r.tier == TIERS[3] else "cons"
        a, l, e = getattr(r, f"{side}_assets"), getattr(r, f"{side}_liabilities"), getattr(r, side)
        # Three amounts each rounded or cut to their printing unit: the gap stays under two of the coarsest units
        # (the same allowance the DART extractor gives its own tables).
        out.append(None if not isinstance(r.tier, str) or r.tier not in TIERS else
                   "no_data" if pd.isna(a) or pd.isna(l) or pd.isna(e) else
                   "ok" if abs(a - l - e) < 2 * max(printing_unit(a), printing_unit(l), printing_unit(e)) else "fail")
    return pd.Series(out, index=panel.index, dtype=object)


IDENTITY_EXCLUDE_RATIO = 0.01


def identity_gap_ratio(panel: pd.DataFrame) -> pd.Series:
    """|assets - liabilities - equity| / |equity| from the DataGuide statement the used value comes from."""
    out = []
    for r in panel.itertuples():
        side = "sep" if r.tier == TIERS[3] else "cons"
        a, l, e = getattr(r, f"{side}_assets"), getattr(r, f"{side}_liabilities"), getattr(r, side)
        out.append(abs(a - l - e) / abs(e) if pd.notna(a) and pd.notna(l) and pd.notna(e) and e else None)
    return pd.Series(out, index=panel.index, dtype=float)


def _original_text(root: Path, raw_path: str) -> str:
    from src.preprocessing.dart_extract import _decode_document
    with zipfile.ZipFile(root / raw_path) as archive:
        return " ".join(_decode_document(archive.read(n))[0] for n in archive.namelist() if n.lower().endswith(".xml"))


def _printed(value: float, text: str) -> bool:
    """The amount printed in won, thousands or millions, as a whole number (not part of a longer one)."""
    for unit in (1, 1000, 1_000_000):
        n = round(abs(value) / unit)
        if n >= 10000 and re.search(r"(?<![\d,.])" + re.escape(f"{n:,}") + r"(?![\d,])", text):
            return True
    return False


def resolve_differences(panel: pd.DataFrame, root: Path, raw_paths: dict[str, str]) -> pd.Series:
    """B4 (user decisions 2026-11-04) for values whose DART first-day decision differs for no known reason (unexplained).

    DART-side reading slips keep the DataGuide value: the DART amount equals the DataGuide value of another period of
    the same company (a comparative column), or equals the DataGuide value once multiplied or divided by 10^3 or 10^6
    (a unit read wrongly). Otherwise both sides are different statements; DataGuide often holds a later restated
    figure, so the DART first-day value (what was public then) is used, unless the DataGuide amount is printed in that
    original too. Rounding needs no rule here: same_amount already compares in the coarser printing unit.
    """
    by_code = panel.groupby("stock_code")
    out = pd.Series(None, index=panel.index, dtype=object)
    for i, r in panel[panel.dart_compare.eq("unexplained")].iterrows():
        other = by_code.get_group(r.stock_code)
        if any(same_amount(v, r.dart_equity) for v in other.loc[other.month.ne(r.month), "equity"]):
            out[i] = "keep_dataguide:dart_read_other_period"
        elif any(same_amount(r.equity, r.dart_equity * k) for k in (1e3, 1e6, 1e-3, 1e-6)):
            out[i] = "keep_dataguide:dart_unit_slip"
        elif any(_printed(r.equity, _original_text(root, raw_paths[x])) for x in r.same_day_receipts.split(";") if raw_paths.get(x)):
            out[i] = "keep_dataguide:printed_in_original"
        else:
            out[i] = "use_dart_first"
    return out


def compare_with_dart(panel: pd.DataFrame, filings: pd.DataFrame, decisions: dict[str, tuple[float, str]],
                      amounts: dict[str, list[float]]) -> pd.DataFrame:
    """B2: the used value against the DART originals of its own filing group. Review only, values are not changed.

    same: a DART decision received on the first day has the amount. tier_only: the amount (or another DataGuide tier)
    is in those originals under another tier. sign: same size, opposite sign. later_filing_only: only a later version
    (e.g. a correction) of the group has it. unexplained: DART has a value that matches none of these.
    """
    group = GROUP
    order = label_checked(filings).sort_values(["rcept_dt", "rcept_no"])
    members = order.groupby(group).rcept_no.agg(list)
    group_of = order.set_index("rcept_no")[group]
    rows = []
    for r in panel.itertuples():
        if not isinstance(r.rcept_no, str) or pd.isna(r.equity):
            rows.append((None, None, None))
            continue
        first = r.same_day_receipts.split(";")
        dec = [decisions[x] for x in first if x in decisions]
        later = [x for x in members.get(tuple(group_of.loc[r.rcept_no]), []) if x not in first]
        mine = [v for v in (r.equity, r.sep, r.cons, r.owners, r.cons - r.nci if pd.notna(r.cons) and pd.notna(r.nci) else None)
                if v is not None and pd.notna(v)]
        if any(same_amount(r.equity, d[0]) for d in dec):
            cls = "same"
        elif any(same_amount(v, a) for v in mine for x in first for a in amounts.get(x, []) + [d[0] for d in dec]):
            cls = "tier_only"
        elif any(same_amount(-r.equity, d[0]) for d in dec):
            cls = "sign"
        elif any(same_amount(r.equity, a) for x in later for a in amounts.get(x, []) + ([decisions[x][0]] if x in decisions else [])):
            cls = "later_filing_only"
        elif dec:
            cls = "unexplained"
        else:
            cls = "no_dart_value"
        rows.append((dec[0][0] if dec else None, dec[0][1] if dec else None, cls))
    return pd.DataFrame(rows, index=panel.index, columns=["dart_equity", "dart_tier", "dart_compare"])


def correction_versions(panel: pd.DataFrame, filings: pd.DataFrame, decisions: dict[str, tuple[float, str]],
                        amounts: dict[str, list[float]], last_days: pd.Series) -> pd.DataFrame:
    """B3 (user decision 2026-11-04): corrections are never back-dated (P1) and replace a value from their receipt (P10).

    Each dated value becomes one or more versions (value, receipt, usable month):
    - DataGuide holds the corrected value (dart_compare later_filing_only): until the correcting receipt the first-day
      DART value is used (its own tier); from the correcting receipt the DataGuide value. When the first-day original
      gave no value, the DataGuide value is used from the first receipt and marked
      dataguide_possibly_corrected (user decision 2026-11-04: a known limit, an exception to P1).
    - After the anchor, every later DART version of the same filing group whose decision has the DataGuide tier and a
      different amount starts a new version (dart_correction). Groups whose DART value differs for other reasons
      (tier_only, unexplained) are left to B4.
    """
    group = GROUP
    order = label_checked(filings).sort_values(["rcept_dt", "rcept_no"])
    members = order.groupby(group).rcept_no.agg(list)
    info = order.set_index("rcept_no")
    rows = []
    usable = panel.rcept_no.notna() & panel.equity.notna()
    if "exclude_reason" in panel:
        usable &= panel.exclude_reason.isna()
    for r in panel[usable].itertuples():
        first = r.same_day_receipts.split(";")
        later = [x for x in members.get(tuple(info.loc[r.rcept_no, group]), []) if x not in first]
        base = dict(stock_code=r.stock_code, month=r.month)
        versions, start = [], 0
        if getattr(r, "b4_action", None) == "use_dart_first":
            versions.append((r.dart_equity, r.dart_tier, r.rcept_no, r.rcept_dt, "dart_first_differs"))
        elif r.dart_compare == "later_filing_only":
            fix = next(i for i, x in enumerate(later) if any(same_amount(r.equity, a) for a in
                                                             amounts.get(x, []) + ([decisions[x][0]] if x in decisions else [])))
            dec = next((decisions[x] for x in first if x in decisions), None)
            if dec:
                versions.append((dec[0], dec[1], r.rcept_no, r.rcept_dt, "dart_first_before_correction"))
                versions.append((r.equity, r.tier, later[fix], info.loc[later[fix], "rcept_dt"], "dataguide_from_correction"))
            else:
                versions.append((r.equity, r.tier, r.rcept_no, r.rcept_dt, "dataguide_possibly_corrected"))
            start = fix + 1
        else:
            versions.append((r.equity, r.tier, r.rcept_no, r.rcept_dt, "dataguide"))
        if r.dart_compare in ("same", "later_filing_only"):
            current = r.equity
            for x in later[start:]:
                if x in decisions and decisions[x][1] == r.tier and not same_amount(current, decisions[x][0]):
                    current = decisions[x][0]
                    versions.append((current, r.tier, x, info.loc[x, "rcept_dt"], "dart_correction"))
        for seq, (value, tier, receipt, day, source) in enumerate(versions):
            rows.append({**base, "seq": seq, "tier": tier, "equity": value, "source": source, "rcept_no": receipt, "rcept_dt": day,
                         "available_month": usable_month(day, last_days)})
    return pd.DataFrame(rows, columns=["stock_code", "month", "seq", "tier", "equity", "source", "rcept_no", "rcept_dt", "available_month"])


def last_trading_days(dates: pd.Series) -> pd.Series:
    days = pd.Series(pd.to_datetime(dates).drop_duplicates().sort_values())
    return days.groupby(days.dt.to_period("M")).max()


def usable_month(receipt: str | None, last_days: pd.Series) -> str | None:
    """P4: a receipt on or after the month's last trading day is used from the next base month."""
    if not isinstance(receipt, str) or not receipt:
        return None
    day = pd.Timestamp(receipt)
    month = day.to_period("M")
    late = month in last_days.index and day >= last_days[month]
    return (month + int(late)).strftime("%Y-%m")


def build(root: Path, raw: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frames, inputs = {}, {}
    for role, (name, standard, items) in SOURCES.items():
        path = root / raw / name
        frames[role], info = read_sheet(path, items)
        inputs[role] = {"file": (raw / name).as_posix(), "sheet": data_sheet(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                        "standard": standard, "items": items, **info}
    overrides = load_overrides(root / OVERRIDES)
    panel = combine(frames)
    without_fiscal_month = panel.attrs.get("rows_without_fiscal_month")
    panel = apply_month_overrides(panel, overrides)
    metadata, facts = load_common_dart(root)
    filings = public_filings(metadata)
    filings = label_checked(filings.drop_duplicates("rcept_no"))
    valued = facts[facts.equity.notna() & facts.equity.ne("")]
    valued = valued.assign(v=valued.equity.astype(float), fpm=valued.fiscal_period.str[:7])
    amounts = valued.groupby("receipt_id").v.agg(list).to_dict()
    corp_map = {e["stock_code"]: e["set"]["corp_code"] for e in overrides if e["type"] == "corp_map"}
    panel = attach_receipts(panel, filings, amounts, corp_map)
    # Reproducibility: every approved override must meet the written criterion again on today's inputs.
    candidates = override_candidates(panel, filings, valued[["corp_code", "receipt_id", "v", "fpm"]])
    problems = verify_overrides(load_overrides(root / OVERRIDES, approved_only=False), candidates)
    if any(not p.endswith("rejected_but_criterion_met") for p in problems):
        raise ValueError("dataguide_override_criteria:" + ";".join(problems))
    unreliable = panel.month_note.fillna("").str.startswith("unreliable")
    panel.loc[unreliable, ["rcept_dt", "rcept_no", "corp_code", "same_day_receipts", "receipt_basis"]] = None
    panel.loc[unreliable, "no_receipt_reason"] = "dataguide_fiscal_month_wrong"
    history = facts[facts.receipt_id.isin(metadata.loc[metadata.disclosure_format.eq("document_only"), "receipt_id"])]
    with_cfs = set(history[history.statement_type.eq("CFS") & history.concept.isin(["total", "owners"])].receipt_id)
    panel = apply_kgaap_consolidated_rule(panel, with_cfs)
    recomputed = decisions_from_facts(metadata, facts)
    decided = recomputed[["receipt_id", "equity", "equity_tier"]]
    decided = decided[decided.equity.notna() & decided.equity.ne("")].drop_duplicates("receipt_id")
    decisions = {k: (float(v), t) for k, v, t in zip(decided.receipt_id, decided.equity, decided.equity_tier)}
    panel = panel.join(compare_with_dart(panel, filings, decisions, amounts))
    panel["identity_check"] = identity_check(panel)
    raw_paths = recomputed[["receipt_id", "raw_path"]].dropna().drop_duplicates("receipt_id")
    panel["b4_action"] = resolve_differences(panel, root, dict(zip(raw_paths.receipt_id, raw_paths.raw_path)))
    # B4 (user decision 2026-11-04): a DataGuide value whose own statement misses assets - liabilities = equity by 1%
    # of equity or more, without a DART original confirming it, is not used. Smaller misses are rounding or printing
    # in the source (observed: at most 0.003%, while the failures that matter are above 600%) and are only marked.
    panel["identity_gap_ratio"] = identity_gap_ratio(panel)
    unconfirmed = (panel.identity_check.eq("fail") & panel.identity_gap_ratio.ge(IDENTITY_EXCLUDE_RATIO)
                   & ~panel.dart_compare.isin(["same", "tier_only"]) & panel.rcept_no.notna())
    panel["exclude_reason"] = None
    panel.loc[unconfirmed, "exclude_reason"] = "identity_failed_unconfirmed"
    dates = pd.concat(pd.read_parquet(p, columns=["Date"]).Date for p in [root / MARCAP / f"marcap-{year}.parquet" for year in range(2000,2022)])
    last_days = last_trading_days(dates)
    panel["available_month"] = [usable_month(r, last_days) for r in panel.rcept_dt]
    codes = set()
    for p in PANEL_NAMES:
        with (root / p).open(encoding="utf-8-sig", newline="") as stream:
            codes |= set(next(csv.reader(stream))[1:])
    panel["panel"] = panel.stock_code.isin(codes)
    p = panel[panel.panel]
    versions = correction_versions(p, filings, decisions, amounts, last_days)
    late = versions[versions.source.ne("dataguide")]
    summary = {"rows": len(panel), "panel_rows": len(p), "panel_companies": int(p.stock_code.nunique()),
               "rows_without_fiscal_month": without_fiscal_month,
               "panel_tiers": p.tier.value_counts().to_dict(), "panel_with_receipt": int(p.rcept_dt.notna().sum()),
               "panel_receipt_basis": p.receipt_basis.value_counts().to_dict(),
               "panel_no_receipt_reason": p.no_receipt_reason.value_counts().to_dict(),
               "panel_month_overrides": p.month_note.fillna("").str.split(":").str[0].replace("", "none").value_counts().to_dict(),
               "panel_dart_compare": p.dart_compare.value_counts().to_dict(),
               "panel_identity_check": p.identity_check.value_counts().to_dict(),
               "panel_b4_action": p.b4_action.value_counts().to_dict(),
               "panel_exclude_reason": p.exclude_reason.value_counts().to_dict(),
               "override_candidates": (candidates.type + ":" + candidates.basis).value_counts().to_dict(),
               "override_problems": problems,
               "versions": len(versions), "version_sources": versions.source.value_counts().to_dict(),
               "periods_with_correction_versions": int(late.drop_duplicates(["stock_code", "month"]).shape[0]),
               "kgaap_consolidated_in_report": p.kgaap_consolidated_in_report.value_counts().to_dict()}
    meta = {"inputs": inputs, "dart_inputs": {p.as_posix(): sha256(root / p) for p in (DART_FILINGS, DART_FACTS)},
            "calculation_code_sha256": sha256(Path(__file__)),
            "overrides": {"file": OVERRIDES.as_posix(), "sha256": sha256(root / OVERRIDES) if (root / OVERRIDES).exists() else None,
                          "approved": len(overrides)},
            "rules": "\n".join(f.__doc__ or "" for f in (attach_receipts, apply_month_overrides, compare_with_dart, identity_check, resolve_differences, override_candidates,
                                                          correction_versions)) + "\n" + (__doc__ or ""), "summary": summary}
    return panel, versions, candidates, meta


COLUMNS = ["stock_code", "name", "month", "month_dataguide", "month_note", "fy", "quarter", "fm", "std", "kind", "tier", "equity",
           "owners", "nci", "cons", "sep", "tier_dataguide", "equity_dataguide", "kgaap_consolidated_in_report", "rcept_dt", "rcept_no",
           "same_day_receipts", "corp_code", "receipt_basis", "no_receipt_reason", "available_month", "sep_assets", "sep_liabilities",
           "cons_assets", "cons_liabilities", "identity_check", "identity_gap_ratio", "dart_equity", "dart_tier", "dart_compare", "b4_action", "exclude_reason", "panel"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Write outputs/run_id/DATAGUIDE_EQUITY/<stamp>/; default prints the summary only")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    panel, versions, candidates, meta = build(root)
    print(json.dumps(meta["summary"], ensure_ascii=False, indent=2))
    if not args.execute:
        print("DRY-RUN: nothing written.")
        return
    out = root / OUTPUT / datetime.now().strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True)
    panel[COLUMNS].to_parquet(out / "equity_panel.parquet", index=False)
    panel[COLUMNS].to_csv(out / "equity_panel.csv", index=False, encoding="utf-8-sig")
    versions.to_parquet(out / "equity_versions.parquet", index=False)  # panel stocks only; one row per (period, version)
    candidates.to_csv(out / "override_candidates.csv", index=False, encoding="utf-8-sig")  # how overrides are found
    (out / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
