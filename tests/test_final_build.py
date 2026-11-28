"""src/final_build.compare: 과거 값 자동 확인이 허용 목록 밖의 값 변화를 잡는지(7-4 ⑦)."""
import unittest

import numpy as np
import pandas as pd

from src.final_build import compare, violations

RULES = [
    {"name": "elig", "only": "missing_to_false", "tables": {"t": ["eligible"]}},
    {"name": "q", "base_month": "2021-11", "stocks": ["000020"], "tables": {"t": ["bm"]}},
]


def table():
    return pd.DataFrame({"stock_id": ["000010", "000020", "000010", "000020"],
                         "base_month": ["2021-10", "2021-10", "2021-11", "2021-11"],
                         "eligible": pd.array([True, pd.NA, True, True], dtype="boolean"),
                         "bm": [0.5, 0.6, 0.7, 0.8], "bm_exclusion_reason": ["", "", "", ""]})


class CompareTests(unittest.TestCase):
    def test_identical_and_longer_new_table_pass(self):
        old = table()
        new = pd.concat([old, old.assign(base_month="2021-12")], ignore_index=True)   # 새 기간은 비교 밖
        r = compare("t", old, new, RULES)
        self.assertEqual((r["violations"], r["allowed"]), ({}, {}))

    def test_listed_exceptions_allowed(self):
        new = table()
        new.loc[1, "eligible"] = False                 # 비어 있던 칸 → False
        new.loc[3, "bm"] = 0.81                        # 2021-11·000020만 허용
        new.loc[0, "bm_exclusion_reason"] = "x"        # 설명 열
        r = compare("t", table(), new, RULES)
        self.assertEqual(r["violations"], {})
        self.assertEqual(r["allowed"], {"eligible": 1, "bm": 1})
        self.assertEqual(r["description_changes"], {"bm_exclusion_reason": 1})

    def test_value_changes_outside_list_stop(self):
        new = table()
        new.loc[0, "eligible"] = False                 # True → False 뒤집힘은 허용 안 함
        new.loc[2, "bm"] = 0.71                        # 다른 종목의 2021-11
        new.loc[1, "bm"] = np.nan                      # 값이 사라짐
        r = compare("t", table(), new, RULES)
        self.assertEqual(r["violations"], {"eligible": 1, "bm": 2})
        self.assertEqual(len(violations({"t": {"vs_development": r}})), 2)

    def test_missing_rows_stop(self):
        r = compare("t", table(), table().iloc[:3], RULES)
        self.assertIn("<keys>", r["violations"])


if __name__ == "__main__":
    unittest.main()
