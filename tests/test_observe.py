# 데이터 관측의 목표월 연도 변환과 월 유효 판정(H1 산출 가능 월, IC 계산 가능 월). 작은 합성 자료만 쓴다.

import unittest

import numpy as np
import pandas as pd

from src.features.observe import ic_validity, month_validity, target_month


class ObserveTests(unittest.TestCase):
    def test_target_month_is_next_calendar_month(self):
        out = target_month(pd.Series(["2000-12", "2021-11", "2005-06"]))
        self.assertEqual(out.tolist(), ["2001-01", "2021-12", "2005-07"])

    def frame(self):
        return pd.DataFrame({
            "base_month": ["m1"] * 3 + ["m2"] * 3 + ["m3"] * 1,
            "member": [True, True, False, True, True, True, True],
            "a": [1.0, 2.0, 9.0, 5.0, 5.0, 5.0, 1.0],
            "y": [0.1, np.nan, 0.3, 0.1, 0.2, 0.3, 0.1],
        })

    def test_score_month_needs_two_stocks_and_positive_sd(self):
        v = month_validity(self.frame(), "member", ["a"], n_min=2).set_index("base_month")
        self.assertTrue(v.valid["m1"])
        self.assertEqual(v.reason["m2"], "zero_or_nonfinite_sd")  # 세 값이 모두 같음
        self.assertEqual(v.reason["m3"], "too_few_stocks")

    def test_ic_month_needs_three_pairs_with_target_and_variation(self):
        ic = ic_validity(self.frame(), "member").set_index("base_month")
        self.assertEqual(ic.pairs["m1"], 1)  # 회원 2개 중 y 있는 쌍 1개, 비회원은 세지 않음
        self.assertEqual(ic.ic_reason["m1"], "too_few_pairs")
        self.assertTrue(ic.ic_valid["m2"])
        same = self.frame().assign(y=0.1)
        self.assertEqual(ic_validity(same, "member").set_index("base_month").ic_reason["m2"], "constant_target")


if __name__ == "__main__":
    unittest.main()
