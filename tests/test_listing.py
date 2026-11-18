# 적격성 규칙(KRX 상장·폐지 목록). 작은 합성 자료만 쓴다.

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing.listing import listing_intervals, monthly_eligibility, read_events

COLS_L = ["종목코드", "종목명", "시장구분", "증권구분", "주식종류", "상장일", "상장폐지일", "상장유형"]
COLS_D = ["종목코드", "종목명", "시장구분", "증권구분", "주식종류", "상장일", "폐지일"]


def frame(cols, rows):
    return pd.DataFrame(rows, columns=cols, dtype=object)


class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.listing = frame(COLS_L, [
            ["000010", "새상장", "KOSPI", "주권", "보통주", "2001/02/15", None, "신규상장"],
            ["000020", "이전", "KOSPI", "주권", "보통주", "2001/03/10", None, "이전상장"],
            ["000030", "상장후폐지", "KOSPI", "주권", "보통주", "2001/01/05", "2001/03/30", "신규상장"],
        ])
        self.delisting = frame(COLS_D, [
            ["000030", "상장후폐지", "KOSPI", "주권", "보통주", "2001/01/05", "2001/03/30"],
            ["000040", "옛상장", "KOSPI", "주권", "보통주", "1990/01/01", "2001/02/27"],
        ])
        self.stocks = ["000010", "000020", "000030", "000040", "000050", "000060"]
        # 2001-03 마지막 거래일은 3월 29일(가정), 2001-02는 2월 27일
        self.ltd = {"2001-01": pd.Timestamp("2001-01-31"), "2001-02": pd.Timestamp("2001-02-27"),
                    "2001-03": pd.Timestamp("2001-03-29")}

    def table(self):
        iv = listing_intervals(self.listing, self.delisting, self.stocks, kospi_observed={"000050"})
        e = monthly_eligibility(iv, ["2001-01", "2001-02", "2001-03"], self.ltd)
        return iv, e.set_index(["stock_id", "base_month"]).eligible

    def test_listed_on_last_trading_day_rule(self):
        _, e = self.table()
        self.assertFalse(e[("000010", "2001-01")])
        self.assertTrue(e[("000010", "2001-02")])  # 2/15 상장, 마지막 거래일 2/27
        self.assertFalse(e[("000040", "2001-02")])  # 폐지일 2/27 = 마지막 거래일 -> 그날 이미 상장 아님
        self.assertTrue(e[("000040", "2001-01")])
        # 폐지일 3/30이 마지막 거래일 3/29 뒤 -> 3월은 적격
        self.assertTrue(e[("000030", "2001-03")])

    def test_no_event_stocks(self):
        iv, e = self.table()
        self.assertTrue(e[("000050", "2001-01")])  # 기록 없음 + KOSPI 관측 -> 계속 상장
        self.assertTrue(pd.isna(e[("000060", "2001-01")]))  # 기록 없음 + 관측 없음 -> 판정 불가
        self.assertEqual(iv.set_index("stock_id").basis["000060"], "no_event_not_kospi_observed")

    def test_prior_listing_period_only_with_kospi_month_end_record(self):
        iv = listing_intervals(self.listing, self.delisting, self.stocks, kospi_observed={"000050"})
        months = ["2001-01", "2001-02", "2001-03"]
        # 000010은 목록상 2001-02-15 상장. 2001-01 월말에 KOSPI 거래 기록이 있으면 이전 상장 기간으로 적격.
        e = monthly_eligibility(iv, months, self.ltd, {("000010", "2001-01"), ("000040", "2001-03")})
        e = e.set_index(["stock_id", "base_month"])
        self.assertTrue(e.eligible[("000010", "2001-01")])
        self.assertTrue(e.eligibility_basis[("000010", "2001-01")].endswith(":prior_listing_kospi_observed"))
        # 폐지 뒤 거래 기록은 이 규칙의 대상이 아님(상장 전 칸만)
        self.assertFalse(e.eligible[("000040", "2001-03")])
        self.assertEqual(e.listing_date[("000010", "2001-01")], "2001-02-15")
        without = monthly_eligibility(iv, months, self.ltd).set_index(["stock_id", "base_month"])
        self.assertFalse(without.eligible[("000010", "2001-01")])

    def test_conflicting_listing_dates_stop(self):
        self.delisting.loc[0, "상장일"] = "2001/01/06"
        with self.assertRaises(ValueError):
            listing_intervals(self.listing, self.delisting, self.stocks, set())

    def test_delisting_inside_period_must_be_in_delisting_list(self):
        self.listing.loc[0, "상장폐지일"] = "2015/05/05"
        with self.assertRaises(ValueError):
            listing_intervals(self.listing, self.delisting, self.stocks, set())

    def test_read_events_keeps_common_shares_incl_foreign(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data/raw/상장").mkdir(parents=True)
            (root / "data/raw/상장폐지").mkdir(parents=True)
            l = frame(COLS_L, [["000010", "a", "KOSPI", "주권", "보통주", "2001/01/01", None, "신규상장"],
                               ["950210", "b", "KOSPI", "외국주권", "보통주", "2021/02/05", None, "신규상장"],
                               ["000015", "c", "KOSPI", "주권", "구형우선주", "2001/01/01", None, "신규상장"]])
            dl = frame(COLS_D, [["000099", "d", "KOSPI", "수익증권", "보통주", "2001/01/01", "2002/01/01"]])
            l.to_csv(root / "data/raw/상장/a.csv", index=False, encoding="cp949")
            dl.to_csv(root / "data/raw/상장폐지/a.csv", index=False, encoding="cp949")
            listing, delisting, _ = read_events(root)
        self.assertEqual(sorted(listing["종목코드"]), ["000010", "950210"])
        self.assertTrue(delisting.empty)


if __name__ == "__main__":
    unittest.main()
