"""Feature Rank IC preprocessing and paired-sample regression checks."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ..evaluation import feature_rank


class FeatureRankTests(unittest.TestCase):
    def prepare_fixture(self, last_return):
        stocks = [str(i).zfill(6) for i in range(6)]
        mom = pd.DataFrame(
            {
                "stock_id": stocks,
                "base_month": "2001-02",
                "eligible": True,
                "mom_available": True,
                "mom": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0],
            }
        )
        panel = pd.DataFrame(
            {
                "stock_id": stocks,
                "month": "2001-03",
                "return": [0.01, 0.02, -0.01, 0.03, 0.04, last_return],
            }
        )
        epu = pd.DataFrame({"epu_month": ["2001-01"], "epu_raw": [2.0]})
        with patch.object(
            feature_rank.pd, "read_parquet", side_effect=[mom, panel]
        ), patch.object(feature_rank, "load_epu_index", return_value=epu):
            return feature_rank.prepare_data(Path.cwd())

    def test_winsor_bounds_do_not_depend_on_target_availability(self):
        missing = self.prepare_fixture(np.nan)
        observed = self.prepare_fixture(0.9)
        np.testing.assert_array_equal(missing.q99, observed.q99)
        expected = np.quantile([2.0, 4.0, 6.0, 8.0, 10.0, 2000.0], 0.99)
        np.testing.assert_allclose(missing.q99, expected)

    def test_both_candidates_use_the_same_observed_pairs(self):
        data = self.prepare_fixture(np.nan)
        rows, coverage, errors = feature_rank.evaluate_months(data)
        self.assertEqual([row["n"] for row in rows], [5, 5])
        self.assertEqual(coverage[0]["missing_target"], 1)
        self.assertLess(max(errors), 1e-12)


if __name__ == "__main__":
    unittest.main()
