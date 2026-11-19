import unittest
import numpy as np
import pandas as pd
from ..analysis.inference import holm, inference
from ..features.epu import align_epu_to_base_month, calculate_h4_epu_features


class H4Tests(unittest.TestCase):
    def test_calendar_lag_and_missing(self):
        e = align_epu_to_base_month(
            pd.DataFrame(
                {"epu_month": pd.PeriodIndex(["2000-12"], freq="M"), "epu_raw": [2.0]}
            )
        )
        self.assertEqual(e.base_month.iloc[0], "2001-01")
        keys = pd.DataFrame(
            {"stock_id": ["a", "b"], "base_month": ["2001-01", "2001-01"]}
        )
        mom = keys.assign(mom=[0.2, np.nan])
        got = calculate_h4_epu_features(mom, keys, e)
        self.assertAlmostEqual(got.h4_interaction.iloc[0], 0.4)
        self.assertFalse(got.h4_available.iloc[1])
        with self.assertRaises(ValueError):
            calculate_h4_epu_features(pd.concat([mom, mom]), keys, e)

    def test_rejects_future_epu_and_infinite_inputs(self):
        e = align_epu_to_base_month(
            pd.DataFrame(
                {"epu_month": pd.PeriodIndex(["2000-12"], freq="M"), "epu_raw": [2.0]}
            )
        )
        keys = pd.DataFrame({"stock_id": ["000001"], "base_month": ["2001-01"]})
        mom = keys.assign(mom=[0.2])
        wrong = e.copy()
        wrong["epu_month"] = "2001-01"
        with self.assertRaises(ValueError):
            calculate_h4_epu_features(mom, keys, wrong)
        with self.assertRaises(ValueError):
            calculate_h4_epu_features(mom.assign(mom=np.inf), keys, e)
        with self.assertRaises(ValueError):
            calculate_h4_epu_features(mom, keys, pd.concat([e, e]))

    def test_export_preserves_main_effects_without_targets(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from ..preprocessing.export import export, extra_feature_specs, OUTPUT

        mom = pd.DataFrame(
            {
                "stock_id": ["000001", "000002"],
                "base_month": ["2001-01"] * 2,
                "eligible": [True, True],
                "mom": [0.2, 0.9],
                "mom_available": [True, False],
            }
        )
        epu = pd.DataFrame(
            {"epu_month": pd.PeriodIndex(["2000-12"], freq="M"), "epu_raw": [2.0]}
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder)
            (root / "data/raw").mkdir(parents=True)
            (root / "data/processed").mkdir(parents=True)
            mom.to_parquet(root / "data/processed/mom.parquet", index=False)
            (root / "data/raw/Korea_Overall_EPU.xlsx").write_bytes(b"fixture")
            with patch("src.h4.preprocessing.export.load_epu_index", return_value=epu):
                manifest = export(root)
            f = pd.read_parquet(root / OUTPUT)
            self.assertNotIn("y", f)
            self.assertEqual(len(f), 2)
            self.assertAlmostEqual(f.h4_interaction.iloc[0], 0.4)
            self.assertFalse(f.h4_available.iloc[1])
            self.assertTrue(pd.isna(f.mom.iloc[1]))
            self.assertEqual(manifest["available_rows"], 1)
            for spec in extra_feature_specs().values():
                self.assertFalse(Path(spec["path"]).is_absolute())
                self.assertEqual(spec["transform"], "none")

    def test_holm(self):
        got = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        np.testing.assert_allclose(list(got.values()), [0.03, 0.06, 0.06])
        with self.assertRaises(ValueError):
            holm({"a": np.nan})

    def test_inference_recovers_negative_interaction(self):
        rng = np.random.default_rng(42)
        rows = []
        for i, month in enumerate(
            pd.period_range("2001-01", periods=36, freq="M").astype(str)
        ):
            e = np.sin(i)
            for j in range(40):
                m, s = rng.normal(size=2)
                rows.append(
                    dict(
                        base_month=month,
                        mom=m,
                        epu_z=e,
                        log_size=s,
                        y=0.1 * m - 0.2 * e * m + 0.02 * s + rng.normal(scale=0.01),
                    )
                )
        tests, monthly = inference(pd.DataFrame(rows), winsor=False)
        self.assertTrue((tests.coefficient < 0).all())
        self.assertTrue((tests.p_negative_one_sided < 0.001).all())
        bad = pd.DataFrame(rows)
        bad = bad[bad.base_month != "2001-05"]
        with self.assertRaises(ValueError):
            inference(bad, False)


if __name__ == "__main__":
    unittest.main()
