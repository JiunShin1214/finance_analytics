"""H3 λ 후보(MODEL_GRID.h3_lambda_learning) 검사.

- 합성 자료: 후보 10개, 후보 밖 λ 거부, λ별 갱신식, 일괄 계산 = λ 하나씩 계산.
- 실제 자료(data/interim/panel·eligibility, data/processed/h3.parquet 세 파일만 지정해 읽음):
  λ=0.97 σ가 h3.parquet의 sigma와 정확히 같고, λ를 바꿔도 산출 여부(vol_available)가 같다.
"""
from pathlib import Path
import math
import unittest

import numpy as np
import pandas as pd

from src.features.h3 import (
    H3_CURRENT_DECAY, H3_LAMBDA_SPANS, calculate_h3_ewma_feature, compute_h3_sigma_grid,
    h3_lambda_candidates, update_ewma_state,
)
from src.pipeline import load_config

ROOT = Path(__file__).resolve().parents[1]


def _synthetic():
    rng = np.random.default_rng(0)
    months = pd.period_range('2000-01', '2002-12', freq='M').astype(str)
    rows = []
    for stock in ('000010', '000020'):
        for i, m in enumerate(months):
            value = np.nan if (stock == '000020' and i in (3, 15, 16)) else float(rng.normal(0, .08))
            rows.append((stock, m, value))
    r = pd.DataFrame(rows, columns=['stock_id', 'month', 'return'])
    r.attrs['units'] = {'return': 'decimal'}
    b = r[['stock_id', 'month']].rename(columns={'month': 'base_month'})
    b['eligible'] = True
    return r, b


class LambdaCandidateTests(unittest.TestCase):
    def test_candidates_match_config(self):
        grid = load_config(ROOT / 'config/experiment.yaml')['MODEL_GRID']['h3_lambda_learning']
        self.assertEqual(tuple(grid['span_months']), H3_LAMBDA_SPANS)
        self.assertEqual(grid['include_current_value'], H3_CURRENT_DECAY)
        table = h3_lambda_candidates(grid['span_months'], grid['include_current_value'])
        self.assertEqual(len(table), 10)
        self.assertEqual(table['label'].iloc[-1], 'decay097')
        self.assertEqual(table['decay'].iloc[-1], .97)
        self.assertEqual(table['weight'].iloc[-1], .03)
        for n, d, w in zip(H3_LAMBDA_SPANS, table['decay'], table['weight']):
            self.assertEqual(w, 2 / (n + 1))
            self.assertEqual(d, 1 - 2 / (n + 1))

    def test_unapproved_lambda_rejected(self):
        r, b = _synthetic()
        for decay in (.95, .9, 1.0, True):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                calculate_h3_ewma_feature(r, b, decay=decay)
        with self.assertRaises(ValueError):
            h3_lambda_candidates((12, 24), .97)

    def test_update_formula_per_lambda(self):
        for d, w in h3_lambda_candidates()[['decay', 'weight']].itertuples(index=False):
            got = update_ewma_state(.01, .1, previous_month='2000-01', current_month='2000-02', decay=d)
            self.assertEqual(got, d * .01 + w * .1 ** 2)
            self.assertEqual(update_ewma_state(.01, None, previous_month='2000-01', current_month='2000-02', decay=d), .01)

    def test_one_pass_equals_single_lambda_runs(self):
        r, b = _synthetic()
        from src.features.h3 import _inputs, _state_history
        table = h3_lambda_candidates()
        rr, bb, _ = _inputs(r, b, None)
        states, grid = _state_history(rr, bb, None, decays=tuple(table['decay']))
        for j, d in enumerate(table['decay']):
            single = calculate_h3_ewma_feature(r, b, decay=d)
            merged = bb.merge(states[['stock_id', 'base_month']].assign(v=grid[:, j]), on=['stock_id', 'base_month'])
            np.testing.assert_array_equal(merged['v'].to_numpy(), single['v'].to_numpy())
        self.assertTrue(np.array_equal(np.isnan(grid).all(axis=1), np.isnan(grid).any(axis=1)))


class RealDataLambdaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = load_config(ROOT / 'config/experiment.yaml')['PATHS']
        files = {k: ROOT / paths[k] for k in ('panel', 'eligibility', 'h3')}
        missing = [str(p) for p in files.values() if not p.is_file()]
        if missing:
            raise unittest.SkipTest(f'공유 표 없음: {missing}')
        panel = pd.read_parquet(files['panel'])
        eligibility = pd.read_parquet(files['eligibility'])
        cls.h3 = pd.read_parquet(files['h3']).sort_values(['stock_id', 'base_month']).reset_index(drop=True)
        cls.grid = compute_h3_sigma_grid(panel, eligibility)

    def test_decay097_sigma_exactly_equals_h3_parquet(self):
        g, h = self.grid, self.h3
        self.assertTrue(g[['stock_id', 'base_month']].equals(h[['stock_id', 'base_month']]))
        a, e = g['sigma_decay097'].to_numpy(), h['sigma'].to_numpy(dtype=float)
        self.assertTrue(np.array_equal(np.isnan(a), np.isnan(e)))
        self.assertTrue(np.array_equal(a[~np.isnan(a)], e[~np.isnan(e)]))   # 정확히 같음(허용 오차 없음)

    def test_availability_independent_of_lambda(self):
        g = self.grid
        expected = g['vol_available'].to_numpy()
        self.assertTrue(np.array_equal(expected, self.h3['sigma'].notna().to_numpy()))
        for col in [c for c in g if c.startswith('sigma_')]:
            with self.subTest(col=col):
                self.assertTrue(np.array_equal(g[col].notna().to_numpy(), expected))
                self.assertTrue((g.loc[expected, col] >= 0).all())


if __name__ == '__main__':
    unittest.main()
