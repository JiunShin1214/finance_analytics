"""Stage 4, DataGuide extraction: the four DataGuide workbooks -> one row per company x statement month.
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


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


RAW = Path("data/raw/external_data")


LAST_GAAP_MONTH = "2010-12"


# role -> (file, standard, {DataGuide item code: concept}). Item codes, not display names, pick the columns.
SOURCES = {
    "kgaap_separate": ("1999_2010년_개별.xlsx", "KGAAP", {"1000910000": "assets", "1000920000": "liabilities", "1000930000": "sep"}),
    "kgaap_consolidated": ("1999_2010_연결.xlsx", "KGAAP", {"4000910000": "assets", "4000920000": "liabilities", "4000930000": "cons",
                                                         "4000930001": "owners", "4000930002": "nci"}),
    "kgaap_fiscal_month": ("1999_2010_개별_결산월.xlsx", None, {"CP10002200": "fm"}),
    "kifrs_consolidated": ("2009_2021_연결.xlsx", "KIFRS", {"6000901001": "assets", "6000902001": "liabilities", "6000903001": "cons",
                                                          "6000903003": "owners", "6000903004": "nci", "CP10002200": "fm"}),
    "kifrs_separate": ("2009_2021_개별.xlsx", "KIFRS", {"3000901001": "assets", "3000902001": "liabilities", "3000903001": "sep"}),
}


BACK = {"1st-Quarter": 9, "Semi-Annual": 6, "3rd-Quarter": 3, "결산": 0}


KIND = {"결산": "사업", "Semi-Annual": "반기", "1st-Quarter": "분기", "3rd-Quarter": "분기"}


TIERS = ("T1_consolidated_owners", "T1D_consolidated_total_less_nci", "T2_consolidated_total", "T3_separate_total")


NO_VALUE = "no_value_consolidated_not_in_report"


def _column(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref)[0]:
        n = n * 26 + ord(ch) - 64
    return n - 1


def data_sheet(path: Path) -> str:
    """The largest worksheet: a refreshed workbook can keep an older small result on another sheet."""
    with zipfile.ZipFile(path) as z:
        sheets = [i for i in z.infolist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", i.filename)]
        return max(sheets, key=lambda i: i.file_size).filename


def iter_rows(path: Path, sheet: str | None = None):
    """(row number, {column index: text}) streamed from one worksheet; shared and inline strings resolved."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si"):
                shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
        with z.open(sheet or data_sheet(path)) as stream:
            for _, el in ET.iterparse(stream):
                if el.tag != NS + "row":
                    continue
                cells = {}
                for c in el.iter(NS + "c"):
                    v = c.find(NS + "v")
                    if v is not None:
                        cells[_column(c.get("r"))] = shared[int(v.text)] if c.get("t") == "s" else v.text
                    elif c.get("t") == "inlineStr":
                        cells[_column(c.get("r"))] = "".join(t.text or "" for t in c.iter(NS + "t"))
                yield int(el.get("r")), cells
                el.clear()


def read_sheet(path: Path, items: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """회계기준 output -> code/name/fy/quarter + one column per wanted item.

    The header row holds '코드' and '분기'; item codes sit two rows above it. An item listed once per period
    repeats the same value in many columns; one column is kept only after checking the copies agree.
    """
    previous, codes, base, cols, rows, differ = {}, {}, None, None, [], 0
    for _, cells in iter_rows(path):
        if base is None:
            values = set(cells.values())
            if {"코드", "분기"} <= values:
                base = next(i for i, v in cells.items() if v == "코드")
                cols = {}
                for i, code in codes.items():
                    if i >= base + 5 and code in items:
                        cols.setdefault(items[code], []).append(i)
                missing = set(items.values()) - set(cols)
                if missing:
                    raise ValueError(f"dataguide_items_missing:{path.name}:{sorted(missing)}")
            else:
                codes, previous = previous, cells
            continue
        code = str(cells.get(base, ""))
        if not code.startswith(("A", "Q")):
            continue
        row = {"code": code, "name": cells.get(base + 1, ""), "fy": cells.get(base + 3, ""), "quarter": cells.get(base + 4, "")}
        for concept, idx in cols.items():
            seen = {cells[i] for i in idx if cells.get(i) not in (None, "")}
            differ += len(seen) > 1
            row[concept] = next(iter(seen)) if len(seen) == 1 else None
        rows.append(row)
    if base is None:
        raise ValueError(f"dataguide_header_not_found:{path.name}")
    if differ:
        raise ValueError(f"dataguide_repeated_columns_disagree:{path.name}:{differ}")
    frame = pd.DataFrame(rows)
    for concept in cols:
        if concept != "fm":
            frame[concept] = pd.to_numeric(frame[concept], errors="coerce")
    return frame, {"rows": len(frame), "columns_per_item": {k: len(v) for k, v in cols.items()}}


def statement_month(fy: str, fiscal_month: str, quarter: str) -> str:
    month = int(float(fiscal_month))
    end = pd.Period(f"{int(float(fy)) + (month <= 3)}-{month:02d}", "M")
    return (end - BACK[quarter]).strftime("%Y-%m")


def choose_tier(row) -> tuple[str | None, float | None]:
    if pd.notna(row.owners):
        return TIERS[0], row.owners
    if pd.notna(row.cons) and pd.notna(row.nci):
        return TIERS[1], row.cons - row.nci
    if pd.notna(row.cons):
        return TIERS[2], row.cons
    if pd.notna(row.sep):
        return TIERS[3], row.sep
    return None, None


def combine(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Four pulls + fiscal months -> one row per company x statement month with the chosen standard and tier."""
    key = ["code", "name", "fy", "quarter"]
    parts = []
    for role, (_, standard, _) in SOURCES.items():
        if standard:
            side = "cons" if "consolidated" in role else "sep"  # assets and liabilities kept per statement for the B2 identity
            f = frames[role].drop(columns=["fm"], errors="ignore").rename(
                columns={"assets": f"{side}_assets", "liabilities": f"{side}_liabilities"})
            parts.append(f.assign(std=standard))
    wide = pd.concat(parts).groupby(key + ["std"], as_index=False).first()
    for col in ("sep", "cons", "owners", "nci", "sep_assets", "sep_liabilities", "cons_assets", "cons_liabilities"):
        if col not in wide:
            wide[col] = float("nan")
    wide = wide[wide[["sep", "cons", "owners", "nci"]].notna().any(axis=1)]
    fm = pd.concat([frames["kgaap_fiscal_month"][["code", "fy", "quarter", "fm"]],
                    frames["kifrs_consolidated"][["code", "fy", "quarter", "fm"]]])
    fm = fm[fm.fm.notna() & fm.fm.ne("")].drop_duplicates(["code", "fy", "quarter"])
    wide = wide.merge(fm, on=["code", "fy", "quarter"], how="left")
    dropped = int(wide.fm.isna().sum())
    wide = wide[wide.fm.notna()].copy()
    wide["month"] = [statement_month(y, m, q) for y, m, q in zip(wide.fy, wide.fm, wide.quarter)]
    has_gaap = set(zip(wide.code[wide["std"].eq("KGAAP")], wide.month[wide["std"].eq("KGAAP")]))
    early = wide.month <= LAST_GAAP_MONTH
    gaap_here = pd.Series([(c, m) in has_gaap for c, m in zip(wide.code, wide.month)], index=wide.index)
    pick = wide[(early & (wide["std"].eq("KGAAP") | ~gaap_here)) | (~early & wide["std"].eq("KIFRS"))].copy()
    pick[["tier", "equity"]] = pd.DataFrame([choose_tier(r) for r in pick.itertuples()], index=pick.index, columns=["tier", "equity"])
    pick = pick[pick.equity.notna()]
    # A fiscal-month change year can map two fiscal years onto one month: the later fiscal year is the newer statement.
    pick = pick.sort_values(["code", "month", "fy"]).drop_duplicates(["code", "month"], keep="last")
    pick["stock_code"] = pick.code.str[1:]
    pick["kind"] = pick.quarter.map(KIND)
    pick.attrs["rows_without_fiscal_month"] = dropped
    return pick.reset_index(drop=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
