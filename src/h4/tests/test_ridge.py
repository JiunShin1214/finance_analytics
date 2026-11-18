"""Regression checks for development-only paired Ridge evaluation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from ..evaluation import ridge


class RidgeComparisonTests(unittest.TestCase):
    def test_train_only_scaling_ignores_future_values(self):
        rng = np.random.default_rng(42)
        columns = ['x_size', 'x_bm', 'x_mom', 'x_sigma_demo', 'x_h4_epu', 'x_h4_interaction']
        data = pd.DataFrame(rng.normal(size=(50, 6)), columns=columns)
        data['year'] = np.repeat([2001, 2004, 2005, 2006, 2007], 10)
        data['target_month'] = data.year.astype(str) + '-01'
        data['train'] = True
        data['predict'] = True
        prepared = {'data': data, 'sigma_labels': ['demo']}
        fold = {'fold': 1, 'train_years': (2001, 2004), 'inner_valid_year': 2005, 'valid_year': 2006}
        first, scales, _ = ridge.scale_fold(prepared, fold)
        changed = data.copy()
        changed.loc[changed.year >= 2005, columns] += 100000
        _, second_scales, _ = ridge.scale_fold({'data': changed, 'sigma_labels': ['demo']}, fold)
        pd.testing.assert_frame_equal(scales, second_scales)
        fit = first['data'].loc[data.year <= 2004, columns]
        np.testing.assert_allclose(fit.mean(), 0, atol=1e-12)
        np.testing.assert_allclose(fit.std(ddof=0), 1)

    def test_duplicate_design_column_is_rejected(self):
        rng = np.random.default_rng(7)
        columns = ['x_size', 'x_bm', 'x_mom', 'x_sigma_demo', 'x_h4_epu', 'x_h4_interaction']
        data = pd.DataFrame(rng.normal(size=(40, 6)), columns=columns)
        data['x_h4_interaction'] = data.x_mom
        data['year'] = 2001
        data['target_month'] = '2001-01'
        data['train'] = True
        data['predict'] = True
        fold = {'fold': 1, 'train_years': (2001, 2004), 'inner_valid_year': 2005, 'valid_year': 2006}
        with self.assertRaisesRegex(ValueError, 'perfect_multicollinearity'):
            ridge.scale_fold({'data': data, 'sigma_labels': ['demo']}, fold)

    def test_final_test_year_rejected_before_data_read(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / 'config').mkdir()
            config = {'FINAL_TEST_ENABLED': False, 'MODEL_GRID': {'last_validation_year': 2022}}
            (root / 'config/experiment.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
            with patch.object(ridge.mt, 'prepare_inputs') as load:
                with self.assertRaises(PermissionError):
                    ridge.prepare(root)
                load.assert_not_called()

    def test_final_test_path_rejected_before_data_read(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / 'config').mkdir()
            config = {'FINAL_TEST_ENABLED': False, 'MODEL_GRID': {'last_validation_year': 2021},
                      'PATHS': {'panel': 'data/final_test.parquet', 'eligibility': 'data/eligibility.parquet',
                                'h1_components': 'data/h1.parquet'}}
            (root / 'config/experiment.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
            with patch.object(ridge.mt, 'prepare_inputs') as load:
                with self.assertRaises(PermissionError):
                    ridge.prepare(root)
                load.assert_not_called()

    def test_pairing_and_delta_use_identical_months(self):
        rows = []
        months = pd.period_range('2006-01', '2021-12', freq='M').astype(str)
        for name, extra in [(ridge.BASE, 0), (ridge.EXT, .01)]:
            for i, month in enumerate(months):
                rows.append(dict(input_set=name, target='raw', loss='mse', scope='full',
                                 target_month=month, ic=.05 + .001 * i + extra, n=100))
        summary, paired, annual = ridge.comparison_tables(pd.DataFrame(rows))
        self.assertEqual(len(paired), 192)
        self.assertEqual(len(annual), 16)
        self.assertAlmostEqual(summary.delta_mean_ic.iloc[0], .01)
        bad = pd.DataFrame(rows)
        bad.loc[bad.input_set.eq(ridge.EXT), 'n'] = 99
        with self.assertRaises(AssertionError):
            ridge.comparison_tables(bad)


if __name__ == '__main__':
    unittest.main()
