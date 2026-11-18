"""H3 분석 보조 계약: KOSPI 상장 이력, 판정 불가 행의 월 보류 범위, 극단값 제한, 승인 차단. 합성 자료만 쓴다."""
import math
import unittest

import numpy as np
import pandas as pd

from src.evaluation.h3_analysis import winsorize_h3
from src.features.h3 import calculate_h3_ewma_feature, validate_h3_contract
from src.pipeline import plan_h3_hypothesis, plan_h3_test
from src.preprocessing.listing import kospi_listing_history


def frame(stock, values, eligible):
    months = pd.period_range('2000-01', periods=len(values), freq='M').astype(str).tolist()
    r = pd.DataFrame({'stock_id': stock, 'month': months, 'return': values})
    k = pd.DataFrame({'stock_id': stock, 'base_month': months,
                      'eligible': pd.array(eligible if isinstance(eligible, list) else [eligible] * len(values), dtype='boolean')})
    return r, k


class KospiListingHistoryTests(unittest.TestCase):
    def test_partial_month_return_counts_as_listed(self):
        e = pd.DataFrame({'stock_id': ['000001'] * 4, 'month': ['2000-01', '2000-02', '2000-03', '2000-04'],
                          'eligible': pd.array([True, False, False, pd.NA], dtype='boolean'),
                          'eligibility_basis': ['x:listed', 'x:on_or_after_delisting', 'x:on_or_after_delisting', 'none']})
        r = pd.DataFrame({'stock_id': ['000001'] * 4, 'month': e.month, 'return': [.1, -.2, np.nan, np.nan]})
        h = kospi_listing_history(e, r)
        self.assertEqual(h.is_listed.tolist(), [True, True, False, False])
        self.assertTrue(h.basis[1].startswith('kospi_partial_month_return_present'))
        self.assertTrue(h.basis[2].startswith('kospi_not_listed_no_return'))

    def test_gap_reset_through_engine(self):
        values = [-.1, .1] * 5 + [np.nan] * 4 + [.02] * 10
        elig = [True] * 10 + [False] * 4 + [True] * 10
        r, k = frame('000001', values, elig)
        e = k.rename(columns={'base_month': 'month'}).assign(eligibility_basis='synthetic')
        h = kospi_listing_history(e, r)
        r.attrs['units'] = {'return': 'decimal'}
        f = calculate_h3_ewma_feature(r, k, listing_history=h)
        s = f.attrs['state_history']
        validate_h3_contract(f, s, r, listing_history=h)
        self.assertEqual(int(s.reset.sum()), 1)
        self.assertEqual(s.loc[s.reset, 'base_month'].iloc[0], '2001-02')
        self.assertEqual(int(s.initialized_this_month.sum()), 2)


class UnknownEligibilityScopeTests(unittest.TestCase):
    def test_unknown_row_without_volatility_does_not_withhold_month(self):
        ra, ka = frame('000001', [0.] * 9 + [.1], True)
        rb, kb = frame('000002', [0.] * 9 + [.2], True)
        rc, kc = frame('000003', [np.nan] * 10, pd.NA)  # 개발 기간 자료가 전혀 없는 판정 불가 종목
        r = pd.concat([ra, rb, rc], ignore_index=True)
        r.attrs['units'] = {'return': 'decimal'}
        k = pd.concat([ka, kb, kc], ignore_index=True)
        f = calculate_h3_ewma_feature(r, k)
        validate_h3_contract(f, f.attrs['state_history'], r)
        last = f[f.base_month.eq('2000-10')]
        np.testing.assert_allclose(last.h3.iloc[:2], [-1 / math.sqrt(2), 1 / math.sqrt(2)], atol=1e-14)
        self.assertEqual(last.h3_exclusion_reason.iloc[2], 'eligibility_unknown')
        self.assertTrue(last.eligibility_complete.all())


class WinsorTests(unittest.TestCase):
    def test_clip_bounds_restandardize_and_ignore_nonmembers(self):
        rng = np.random.default_rng(1)
        n = 200
        sigma = np.exp(rng.normal(-2.5, .4, n))
        sigma[0] = 5.0
        f = pd.DataFrame({'stock_id': [f'{i:06d}' for i in range(n + 1)], 'base_month': '2001-01',
                          'eligible': pd.array([True] * n + [False], dtype='boolean'),
                          'sqrt_v': [*sigma, 9.9], 'h3_available': [True] * n + [False]})
        f['h3'] = np.where(f.h3_available, (f.sqrt_v - sigma.mean()) / sigma.std(ddof=1), np.nan)
        w = winsorize_h3(f)
        lo, hi = pd.Series(sigma).quantile([.01, .99], interpolation='linear')
        m = w[w.h3_available]
        self.assertAlmostEqual(m.sigma_winsor.max(), hi)
        self.assertAlmostEqual(m.sigma_winsor.min(), lo)
        self.assertAlmostEqual(m.h3_winsor.mean(), 0, places=12)
        self.assertAlmostEqual(m.h3_winsor.std(ddof=1), 1, places=12)
        self.assertTrue(np.isnan(w.h3_winsor.iloc[-1]))
        self.assertEqual(int(w.clipped.sum()), 4)
        self.assertGreaterEqual(m.h3.corr(m.h3_winsor, method='spearman'), .99)
        with self.assertRaises(ValueError):
            winsorize_h3(f.assign(y=0.))


class ApprovalTests(unittest.TestCase):
    def test_stages_require_approval(self):
        with self.assertRaises(PermissionError):
            plan_h3_test({'H3_TEST': {'approved': False}})
        with self.assertRaises(PermissionError):
            plan_h3_hypothesis({'H3_HYPOTHESIS': {'approved': False}})


if __name__ == '__main__':
    unittest.main()
