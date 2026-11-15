"""OLS·Ridge 비교군 실행 모듈(묶음 2) 계약. 합성 자료만 사용한다."""
import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation import linear_reg_test as lr
from src.features.h1 import calculate_h1_score
from src.models import ridge
from src.models.interface import FoldSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config/experiment.yaml').read_text(encoding='utf-8'))
BASE = pd.period_range('2000-12', '2004-11', freq='M').astype(str)
FOLD = FoldSpec(fold='2004', train_start='2001-01', train_end='2003-12', valid_start='2004-01', valid_end='2004-12')


def synthetic(seed=0, stocks=40):
    rng = np.random.default_rng(seed)
    rows = []
    for month in BASE:
        for i in range(stocks):
            v = rng.normal(size=3)
            if month == '2000-12':
                v[2] = np.nan  # 첫 기준월은 모멘텀 없음 -> H1 없음
            if i == 0:
                v[1] = np.nan  # 결측 종목: B/M 없음
            if i == 1:
                v[0] *= 30  # 극단값
            rows.append({'stock_id': f'{i:06d}', 'base_month': month, 'eligible': True, 'size': v[0], 'bm': v[1], 'mom': v[2]})
    c = pd.DataFrame(rows)
    c['eligible'] = c.eligible.astype('boolean')
    for r in ('size', 'bm', 'mom'):
        c[r + '_available'] = np.isfinite(c[r])
    h1 = calculate_h1_score(c)
    sigma = np.abs(rng.normal(0.1, 0.03, size=len(c)))
    sigma[c.stock_id.eq('000002').to_numpy()] = np.nan  # H3 결측 종목
    h3 = c[['stock_id', 'base_month']].assign(sigma=sigma, h3_available=np.isfinite(sigma))
    st = h3[h3.h3_available].groupby('base_month').sigma.agg(['mean', 'std'])
    h3['h3'] = ((h3.sigma - h3.base_month.map(st['mean'])) / h3.base_month.map(st['std'])).where(h3.h3_available)
    months = pd.period_range('2001-01', '2004-12', freq='M').astype(str)
    panel = pd.DataFrame([{'stock_id': f'{i:06d}', 'month': m, 'return': rng.normal(0, .08)} for m in months for i in range(stocks)])
    panel.loc[panel.stock_id.eq('000003') & panel.month.eq('2002-05'), 'return'] = 5.0  # 극단 수익률
    panel.attrs['units'] = {'return': 'decimal'}
    return c, h1, h3, panel


class InputTests(unittest.TestCase):
    def setUp(self):
        self.c, self.h1, self.h3, self.panel = synthetic()

    def test_raw_inputs_match_stored_and_fill_missing_with_zero(self):
        x, checks = lr.build_inputs(self.c, self.h1, self.h3, variant='raw')
        self.assertLess(checks['stored_z_max_abs_error'], 1e-12)
        miss = x.stock_id.eq('000000') & x.base_month.ne('2000-12')
        self.assertTrue(x.loc[miss, 'z_bm'].eq(0).all())
        self.assertFalse(x.loc[miss, 'member'].any())
        # 결측 종목의 있는 값은 그 달 공통 집합 기준 Z
        m = x.base_month.eq('2003-06')
        common = self.c[self.c.base_month.eq('2003-06') & self.h1.h1_available.to_numpy()]
        expected = (self.c.loc[self.c.base_month.eq('2003-06') & self.c.stock_id.eq('000000'), 'size'].iloc[0] - common['size'].mean()) / common['size'].std()
        self.assertAlmostEqual(x.loc[m & x.stock_id.eq('000000'), 'z_size'].iloc[0], expected)
        self.assertTrue(x.loc[x.stock_id.eq('000002'), 'h3'].eq(0).all())
        self.assertTrue(np.isfinite(x[['z_size', 'z_bm', 'z_mom', 'h1', 'h3']].to_numpy()).all())

    def test_winsor_inputs_match_h1_winsorize_and_bound_extremes(self):
        raw, _ = lr.build_inputs(self.c, self.h1, self.h3, variant='raw')
        w, checks = lr.build_inputs(self.c, self.h1, self.h3, variant='winsor')
        self.assertLess(checks['h1_winsorize_max_abs_error'], 1e-9)
        ext = raw.stock_id.eq('000001') & raw.member
        self.assertLess(w.loc[ext, 'z_size'].abs().max(), raw.loc[ext, 'z_size'].abs().max())

    def test_targets_attached_after_inputs_and_winsor_only_train_target(self):
        x, _ = lr.build_inputs(self.c, self.h1, self.h3, variant='winsor')
        d = lr.attach_targets(x, self.panel, variant='winsor')
        self.assertTrue((pd.PeriodIndex(d.base_month, freq='M') + 1).astype(str).equals(pd.Index(d.target_month)))
        r = d[d.stock_id.eq('000003') & d.target_month.eq('2002-05')].iloc[0]
        self.assertEqual(r.y, 5.0)  # 평가 목표값은 원값
        self.assertLess(r.y_fit, 5.0)  # 학습 목표값만 제한
        self.assertTrue(d.loc[~d.train, 'y_fit'].isna().all())
        raw = lr.attach_targets(lr.build_inputs(self.c, self.h1, self.h3, variant='raw')[0], self.panel, variant='raw')
        self.assertTrue(raw.loc[raw.train, 'y_fit'].equals(raw.loc[raw.train, 'y']))


class FoldRunTests(unittest.TestCase):
    def setUp(self):
        c, h1, h3, panel = synthetic()
        self.panel = panel
        self.inputs = lr.build_inputs(c, h1, h3, variant='raw')[0]
        self.data = lr.attach_targets(self.inputs, panel, variant='raw')
        self.cands = ridge.enumerate_ridge_candidates(config=CONFIG)

    def test_predicts_every_stock_with_return_in_validation_year(self):
        preds, coefs, inner = lr.run_folds(self.data, [FOLD], self.cands, variant='raw', input_set='main')
        expected = self.data[self.data.predict & self.data.target_month.between('2004-01', '2004-12')]
        for model in ('ols', 'ridge'):
            self.assertEqual(len(preds[preds.model.eq(model)]), len(expected))
        self.assertTrue(preds.filled.any())  # 결측 채운 종목도 예측
        self.assertEqual(set(coefs.model), {'ols', 'ridge'})
        self.assertEqual(inner.chosen.sum(), 1)

    def test_validation_year_targets_do_not_change_fit(self):
        a, ca, ia = lr.run_folds(self.data, [FOLD], self.cands, variant='raw', input_set='main')
        shocked = self.panel.copy()
        shocked.attrs['units'] = {'return': 'decimal'}
        shocked.loc[shocked.month.between('2004-01', '2004-12'), 'return'] *= -3
        d2 = lr.attach_targets(self.inputs, shocked, variant='raw')
        b, cb, ib = lr.run_folds(d2, [FOLD], self.cands, variant='raw', input_set='main')
        pd.testing.assert_frame_equal(ca, cb)
        pd.testing.assert_frame_equal(ia, ib)
        np.testing.assert_allclose(a.yhat, b.yhat)

    def test_reference_inputs_use_same_rows(self):
        m, _, _ = lr.run_folds(self.data, [FOLD], self.cands, variant='raw', input_set='main')
        r, _, _ = lr.run_folds(self.data, [FOLD], self.cands, variant='raw', input_set='reference')
        key = ['model', 'stock_id', 'target_month']
        self.assertTrue(m[key].reset_index(drop=True).equals(r[key].reset_index(drop=True)))

    def test_monthly_ic_full_and_common(self):
        p, _, _ = lr.run_folds(self.data, [FOLD], self.cands, variant='raw', input_set='main')
        ic, err = lr.monthly_ic(p)
        self.assertLess(err, 1e-12)
        self.assertEqual(set(ic.scope), {'full', 'common'})
        full = ic[ic.scope.eq('full')].n.sum()
        common = ic[ic.scope.eq('common')].n.sum()
        self.assertGreater(full, common)


class SettingsTests(unittest.TestCase):
    def test_config_passes_and_changes_fail(self):
        lr.check_settings(CONFIG)
        for key, value in (('hypothesis_tests', True), ('variants', ['raw']), ('selected_candidate', 'ols'),
                           ('input_sets', {'main': ['z_size', 'z_bm', 'z_mom', 'h3', 'h1'], 'reference': ['h1', 'h3']})):
            bad = copy.deepcopy(CONFIG)
            bad['LINEAR_REG_TEST'][key] = value
            with self.assertRaises(PermissionError):
                lr.check_settings(bad)

    def test_stability_metrics(self):
        s = pd.Series(np.r_[np.full(60, .05), np.full(132, -.01)], index=pd.period_range('2006-01', '2021-12', freq='M').astype(str))
        row = lr.stability(s)
        self.assertAlmostEqual(row['mean_2006~2010'], .05)
        self.assertAlmostEqual(row['mean_2011~2021'], -.01)
        self.assertEqual(row['positive_years'], 5)


if __name__ == '__main__':
    unittest.main()
