"""KRX KOSPI 상장·상장폐지 목록 -> 종목×월 적격성(2026-11-04·05 사용자 결정).

적격성: 그 달 마지막 거래일에 KOSPI 주권(국내·외국)의 보통주로 상장되어 있으면(상장일 <= 그날 < 폐지일) 적격.
- 폐지 목록에 있으면 그 행의 상장일·폐지일. 상장 목록에도 있으면 두 상장일이 같아야 한다.
- 상장 목록에만 있으면 상장일부터 계속 상장(그 행의 폐지일이 2021년 이후면 기간 밖).
- 두 목록에 없고 marcap에서 KOSPI로 관측되면 2000년 전부터 2021년 말까지 상장 유지로 본다.
- 두 목록에 없고 KOSPI 관측도 없으면 판정 불가(NA).
비결측 수익률이나 제공 1,086종목이라는 사실로 적격성을 추정하지 않는다.

"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LISTING_DIR = Path("data/raw/상장")
DELISTING_DIR = Path("data/raw/상장폐지")
SECURITY_KINDS = ("주권", "외국주권")
PERIOD_END = pd.Timestamp("2021-12-31")


def _day(value) -> pd.Timestamp | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return pd.Timestamp(value.strip().replace("/", "-"))


def read_events(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """상장·상장폐지 CSV 전부(cp949) -> KOSPI 주권·외국주권 보통주 행만 남긴 두 표와 파일 목록."""
    def load(folder: Path) -> pd.DataFrame:
        files = sorted((root / folder).glob("*.csv"))
        if not files:
            raise FileNotFoundError(folder.as_posix())
        return pd.concat([pd.read_csv(f, encoding="cp949", dtype=str).assign(source_file=(folder / f.name).as_posix())
                          for f in files], ignore_index=True)

    listing, delisting = load(LISTING_DIR), load(DELISTING_DIR)
    info = {"listing_rows": len(listing), "delisting_rows": len(delisting)}

    def keep(t: pd.DataFrame) -> pd.DataFrame:
        return t[t["시장구분"].eq("KOSPI") & t["증권구분"].isin(SECURITY_KINDS) & t["주식종류"].eq("보통주")].copy()

    listing, delisting = keep(listing), keep(delisting)
    info.update(listing_common_rows=len(listing), delisting_common_rows=len(delisting))
    return listing, delisting, info


def listing_intervals(listing: pd.DataFrame, delisting: pd.DataFrame, stocks, kospi_observed: set[str]) -> pd.DataFrame:
    """종목별 상장 구간(start/end; None은 기간 밖), 판정 근거, KOSPI 이전상장일."""
    for name, table in (("listing", listing), ("delisting", delisting)):
        dup = table["종목코드"][table["종목코드"].duplicated()]
        if len(dup):
            raise ValueError(f"{name}_multiple_rows:{sorted(set(dup))[:5]}")
    lst, dl = listing.set_index("종목코드"), delisting.set_index("종목코드")
    rows = []
    for stock in stocks:
        in_l, in_d = stock in lst.index, stock in dl.index
        transfer = _day(lst.at[stock, "상장일"]) if in_l and lst.at[stock, "상장유형"] == "이전상장" else None
        known, start, end = True, None, None
        if in_d:
            start, end = _day(dl.at[stock, "상장일"]), _day(dl.at[stock, "폐지일"])
            if in_l and _day(lst.at[stock, "상장일"]) != start:
                raise ValueError(f"listing_date_conflict:{stock}")
            basis = "listing_and_delisting_lists" if in_l else "delisting_list"
        elif in_l:
            start, later = _day(lst.at[stock, "상장일"]), _day(lst.at[stock, "상장폐지일"])
            if later is not None and later <= PERIOD_END:
                raise ValueError(f"delisting_missing_from_delisting_list:{stock}")
            basis = "listing_list"
        elif stock in kospi_observed:
            basis = "no_event_kospi_observed"
        else:
            known, basis = False, "no_event_not_kospi_observed"
        rows.append({"stock_id": stock, "known": known, "start": start, "end": end, "basis": basis, "transfer_date": transfer})
    return pd.DataFrame(rows)


def monthly_eligibility(intervals: pd.DataFrame, months, last_trading_day: dict[str, pd.Timestamp],
                        kospi_month_end: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    """종목×월 eligible(nullable boolean)과 근거. 마지막 거래일이 없는 달은 달력 말일을 쓴다.

    kospi_month_end: 그 달 마지막 거래일에 marcap에서 KOSPI로 거래된 (종목, 월). 주어지면, 목록의 상장일 전이라도
    그 달 KOSPI 거래 기록이 있는 칸은 같은 코드의 이전 상장 기간으로 보고 적격으로 둔다(2026-11-05 사용자 결정,
    예: 000080 목록상 2009-10 상장이나 2001년 '진로'로 KOSPI 거래). 근거는 '…:prior_listing_kospi_observed'.
    """
    months = list(months)
    ltd = np.array([last_trading_day.get(m, pd.Period(m, "M").end_time.normalize()) for m in months], dtype="datetime64[ns]")
    n_m = len(months)
    iv = intervals.reset_index(drop=True)
    far_past, far_future = np.datetime64("1900-01-01"), np.datetime64("2100-01-01")
    start = np.array([far_past if pd.isna(s) else np.datetime64(s) for s in iv.start], dtype="datetime64[ns]")
    end = np.array([far_future if pd.isna(e) else np.datetime64(e) for e in iv.end], dtype="datetime64[ns]")
    day = np.tile(ltd, len(iv))
    s, e = np.repeat(start, n_m), np.repeat(end, n_m)
    listed = (s <= day) & (day < e)
    known = np.repeat(iv.known.to_numpy(bool), n_m)
    basis = np.repeat(iv.basis.to_numpy(object), n_m)
    state = np.where(~known, "", np.where(day < s, "before_listing", np.where(day >= e, "on_or_after_delisting", "listed")))
    stocks = np.repeat(iv.stock_id.to_numpy(object), n_m)
    month_col = np.tile(np.array(months, dtype=object), len(iv))
    if kospi_month_end:
        prior = (state == "before_listing") & np.array([(st, m) in kospi_month_end for st, m in zip(stocks, month_col)])
        listed = listed | prior
        state = np.where(prior, "prior_listing_kospi_observed", state)
    out = pd.DataFrame({"stock_id": stocks, "base_month": month_col,
                        "eligible": pd.array(np.where(known, listed, False), dtype="boolean"),
                        "eligibility_basis": [b if not st else f"{b}:{st}" for b, st in zip(basis, state)],
                        "listing_date": np.repeat(np.array([None if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")
                                                            for x in iv.start], dtype=object), n_m),
                        "delisting_date": np.repeat(np.array([None if pd.isna(x) else pd.Timestamp(x).strftime("%Y-%m-%d")
                                                              for x in iv.end], dtype=object), n_m)})
    out.loc[~known, "eligible"] = pd.NA
    return out


def kospi_listing_history(eligibility: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """H3 4개월 공백 초기화용 종목×월 KOSPI 상장 여부(2026-11-08 사용자 결정: KOSPI만 본다).

    eligibility: stock_id, month, eligible, eligibility_basis. returns: stock_id, month, return.
    is_listed=True: 월말 적격이거나, 월말엔 비적격이어도 그 달 제공 수익률이 있음(상장폐지 달 등 그 달 일부 상장).
    is_listed=False: 월말 비적격(또는 판정 불가)이고 그 달 수익률도 없음. 코스닥 등 다른 시장 상장은 보지 않는다
    (코스닥→KOSPI 이전은 별개 기업, 교수님 지침). basis에 근거를 남긴다.
    """
    keys = ["stock_id", "month"]
    t = eligibility[[*keys, "eligible", "eligibility_basis"]].merge(
        returns[[*keys, "return"]], on=keys, how="left", validate="one_to_one")
    month_end = t.eligible.eq(True).fillna(False).astype(bool)
    has_return = t["return"].notna()
    basis = np.where(month_end, "kospi_month_end_listed",
                     np.where(has_return, "kospi_partial_month_return_present", "kospi_not_listed_no_return"))
    return pd.DataFrame({"stock_id": t.stock_id, "month": t.month, "is_listed": (month_end | has_return).to_numpy(bool),
                         "basis": [f"{b}:{e}" for b, e in zip(basis, t.eligibility_basis.astype(str))]})


def kospi_month_end_observations(marcap: pd.DataFrame, last_trading_day: dict[str, pd.Timestamp]) -> set[tuple[str, str]]:
    """marcap 일별 행(Code, Market, Date) -> 그 달 마지막 거래일에 KOSPI로 거래된 (종목코드, 월) 집합."""
    days = pd.to_datetime(marcap.Date)
    month = days.dt.to_period("M").astype(str)
    last = month.map(last_trading_day)
    hit = marcap.Market.eq("KOSPI").to_numpy() & (days.to_numpy() == pd.to_datetime(last).to_numpy())
    return set(zip(marcap.Code.astype(str)[hit], month[hit]))
