"""Small synthetic-only H3 contracts. No real CSV, Parquet, API or RUN access."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.h3 import (
    calculate_h3_ewma_feature, initialize_ewma_state,
    update_ewma_state, validate_h3_contract,
)


def fixture(values, stock='000001', eligible=True, start='2000-01'):
    months = pd.period_range(start, periods=len(values), freq='M').astype(str).tolist()
    returns = pd.DataFrame({'stock_id': stock, 'month': months, 'return': values})
    returns.attrs['units'] = {'return': 'decimal'}
    keys = pd.DataFrame({'stock_id': stock, 'base_month': months})
    keys['eligible'] = pd.array([eligible] * len(values), dtype='boolean')
    return returns, keys


def history(returns, flags=None, markets=None):
    h = returns[['stock_id', 'month']].copy()
    h['is_listed'] = True if flags is None else flags
    h['basis'] = ['synthetic listing evidence ' + month for month in h['month']]
    if markets is not None:
        h['market'] = markets
    return h


def plain(frame):
    copy = frame.copy(deep=True)
    copy.attrs = {}
    return copy


class H3ContractTests(unittest.TestCase):
    def checked(self, returns, keys, listing=None):
        result = calculate_h3_ewma_feature(returns, keys, listing_history=listing)
        validate_h3_contract(result, result.attrs['state_history'], returns,
                             listing_history=listing)
        return result, result.attrs['state_history']

    def test_01_nine_returns_stay_pending(self):
        r, b = fixture([.1] * 9)
        f, s = self.checked(r, b)
        self.assertFalse(s['initialized'].any())
        self.assertTrue(s['v'].isna().all())
        self.assertTrue(f['sqrt_v'].isna().all())
        self.assertEqual(s.iloc[-1]['n_valid_since_reset'], 9)
        with self.assertRaises(ValueError):
            initialize_ewma_state([.1] * 9)

    def test_02_ten_returns_sample_variance_nonconsecutive(self):
        # Five -.1 and five +.1: sum squares .1 / nine degrees of freedom = 1/90.
        self.assertAlmostEqual(initialize_ewma_state([-.1, .1] * 5), 1 / 90)
        r, b = fixture([-.1, .1] * 4 + [np.nan, -.1, .1])
        f, s = self.checked(r, b)
        self.assertFalse(s.iloc[-2]['initialized'])
        self.assertTrue(s.iloc[-1]['initialized_this_month'])
        self.assertAlmostEqual(f.iloc[-1]['v'], 1 / 90)
        self.assertAlmostEqual(f.iloc[-1]['sqrt_v'], math.sqrt(1 / 90))

    def test_03_initialization_month_not_updated_twice(self):
        r, b = fixture([0.] * 9 + [.1])
        f, s = self.checked(r, b)
        # Mean .01; squared deviations .009; sample variance .001.
        self.assertAlmostEqual(f.iloc[-1]['v'], .001)
        self.assertEqual(s.iloc[-1]['action'], 'initialize')
        self.assertEqual(s.iloc[-1]['initialization_month'], '2000-10')
        self.assertEqual(s['initialized_this_month'].sum(), 1)

    def test_04_recursive_equation(self):
        self.assertAlmostEqual(update_ewma_state(.04, .10,
                               previous_month='2000-12', current_month='2001-01'), .0391)
        r, b = fixture([-.1, .1] * 5 + [.2])
        f, _ = self.checked(r, b)
        self.assertAlmostEqual(f.iloc[-1]['v'], 97 / 9000 + .0012)
        self.assertAlmostEqual(update_ewma_state(.04, -.10,
                               previous_month='2000-12', current_month='2001-01'), .0391)

    def test_05_zero_and_nan_differ(self):
        r, b = fixture([-.1, .1] * 5 + [0., np.nan])
        f, s = self.checked(r, b)
        self.assertAlmostEqual(f.iloc[10]['v'], 97 / 9000)
        self.assertAlmostEqual(f.iloc[11]['v'], 97 / 9000)
        self.assertEqual(s.iloc[10]['action'], 'update')
        self.assertEqual(s.iloc[11]['action'], 'hold')
        self.assertEqual(f.iloc[-1]['recent_valid_count'], 11)

    def test_06_consecutive_missing_preserves_state_and_recovers(self):
        r, b = fixture([-.1, .1] * 5 + [np.nan] * 6 + [0.])
        f, s = self.checked(r, b)
        np.testing.assert_allclose(f.loc[10:15, 'v'], [1 / 90] * 6)
        self.assertFalse(f.iloc[15]['vol_available'])
        self.assertAlmostEqual(f.iloc[16]['v'], 97 / 9000)
        self.assertEqual(s['initialized_this_month'].sum(), 1)

    def test_07_missing_calendar_row_year_boundary_and_unsorted(self):
        r, b = fixture([-.1, .1] * 5 + [0., .1, np.nan, .2])
        # 2000-11 row absent. It must remain a missing calendar month, not be compressed.
        r = r.drop(index=10).sample(frac=1, random_state=7)
        b = b.sample(frac=1, random_state=8)
        before_r, before_b = r.copy(deep=True), b.copy(deep=True)
        f, s = self.checked(r, b)
        self.assertEqual(f['base_month'].tolist(), b['base_month'].tolist())
        nov = s[s['base_month'] == '2000-11'].iloc[0]
        self.assertFalse(nov['observed_row'])
        self.assertEqual(nov['action'], 'hold')
        self.assertAlmostEqual(nov['v'], 1 / 90)
        dec = s[s['base_month'] == '2000-12'].iloc[0]
        jan = s[s['base_month'] == '2001-01'].iloc[0]
        self.assertEqual(jan['action'], 'hold')
        self.assertAlmostEqual(dec['v'], 97 / 9000 + .0003)
        self.assertAlmostEqual(jan['v'], dec['v'])
        assert_frame_equal(r, before_r)
        assert_frame_equal(b, before_b)

    def test_08_exact_twelve_calendar_month_boundaries(self):
        for count in (9, 10, 11, 12):
            with self.subTest(valid=count):
                r, b = fixture([.1] * 12 + [np.nan] * (12 - count) + [.1] * count)
                f, _ = self.checked(r, b)
                last = f.iloc[-1]
                self.assertEqual(last['recent_valid_count'], count)
                self.assertEqual(last['vol_available'], count >= 10)
                self.assertEqual(pd.notna(last['sqrt_v']), count >= 10)
                self.assertTrue(last['initialized'])

    def test_09_kosdaq_preentry_returns_are_used(self):
        r, b = fixture([-.1, .1] * 5 + [0.])
        b.loc[:9, 'eligible'] = False
        h = history(r, markets=['KOSDAQ'] * 10 + ['KOSPI'])
        f, s = self.checked(r, b, h)
        self.assertAlmostEqual(f.iloc[-1]['v'], 97 / 9000)
        self.assertEqual(f.iloc[-1]['recent_valid_count'], 11)
        self.assertEqual(s.iloc[9]['initialization_month'], '2000-10')
        self.assertFalse(s['reset'].any())
        self.assertTrue(f.loc[:9, 'h3'].isna().all())

    def test_10_market_transfer_and_reentry_carries_state(self):
        r, b = fixture([-.1, .1] * 5 + [0., .2])
        b.loc[4:9, 'eligible'] = False
        h = history(r, markets=['KOSPI'] * 4 + ['KOSDAQ'] * 6 + ['KOSPI'] * 2)
        f, s = self.checked(r, b, h)
        self.assertFalse(s['reset'].any())
        self.assertAlmostEqual(f.iloc[-1]['v'], .97 * (97 / 9000) + .0012)
        self.assertEqual(s['initialized_this_month'].sum(), 1)

    def test_11_actual_unlisted_gap_three_four_five(self):
        for gap in (3, 4, 5):
            with self.subTest(gap=gap):
                r, b = fixture([-.1, .1] * 5 + [np.nan] * gap + [.02] * 10)
                h = history(r, [True] * 10 + [False] * gap + [True] * 10)
                f, s = self.checked(r, b, h)
                self.assertEqual(s['reset'].sum(), int(gap >= 4))
                if gap == 3:
                    self.assertAlmostEqual(s.iloc[13]['v'], 97 / 9000 + .000012)
                    self.assertEqual(s['initialized_this_month'].sum(), 1)
                else:
                    self.assertTrue(s.iloc[13]['reset'])
                    self.assertTrue(pd.isna(s.iloc[13]['v']))
                    self.assertEqual(s.iloc[13]['n_valid_since_reset'], 0)
                    self.assertEqual(s.iloc[13]['recent_valid_count'], 0)
                    self.assertIn('2001-02', s.iloc[13]['reset_basis'])
                    self.assertTrue(pd.isna(s.iloc[-2]['v']))
                    self.assertEqual(s.iloc[-1]['action'], 'initialize')
                    self.assertAlmostEqual(f.iloc[-1]['v'], 0.)
                    self.assertEqual(s['initialized_this_month'].sum(), 2)
                self.assertTrue(f.attrs['reset_assessed'])

    def test_12_four_nan_without_listing_evidence_never_resets(self):
        r, b = fixture([-.1, .1] * 5 + [np.nan] * 4 + [0.])
        f, s = self.checked(r, b)
        self.assertFalse(s['reset'].any())
        self.assertAlmostEqual(f.iloc[-1]['v'], 97 / 9000)
        self.assertFalse(f.attrs['reset_assessed'])
        self.assertEqual(f.attrs['result_scope'], 'calculation_only_not_PIT_certified')
        _, listed_states = self.checked(r, b, history(r))
        self.assertFalse(listed_states['reset'].any())

    def test_13_ineligible_does_not_reset_or_modify_returns(self):
        r, b = fixture([-.1, .1] * 5 + [.1] * 6, eligible=False)
        before = r.copy(deep=True)
        f, s = self.checked(r, b)
        self.assertFalse(s['reset'].any())
        self.assertEqual(s.iloc[-1]['n_valid_since_reset'], 16)
        self.assertTrue(f.iloc[-1]['vol_available'])
        self.assertTrue(f['h3'].isna().all())
        self.assertTrue(f['h3_exclusion_reason'].eq('ineligible').all())
        assert_frame_equal(before, r)

    def test_14_future_changes_do_not_change_past(self):
        r, b = fixture([-.1, .1] * 10)
        f, s = self.checked(r, b)
        changed = r.copy(deep=True)
        changed.loc[15:, 'return'] = .9
        ff, ss = self.checked(changed, b)
        assert_frame_equal(plain(f.iloc[:15]), plain(ff.iloc[:15]))
        assert_frame_equal(s.iloc[:15], ss.iloc[:15])
        prefix, ps = self.checked(r.iloc[:15].copy(), b.iloc[:15].copy())
        assert_frame_equal(plain(f.iloc[:15]), plain(prefix))
        assert_frame_equal(s.iloc[:15], ps)
        # A future relisting cannot retroactively trigger an earlier reset.
        missing, keys = fixture([-.1, .1] * 5 + [np.nan] * 4 + [.2])
        hist = history(missing, [True] * 10 + [False] * 4 + [True])
        _, full = self.checked(missing, keys, hist)
        _, past = self.checked(missing.iloc[:13].copy(), keys.iloc[:13].copy(), hist.iloc[:13].copy())
        assert_frame_equal(full.iloc[:13], past)

    def test_15_sample_zscore_small_and_constant_universe(self):
        # Two initial variances .001 and .004 -> sigmas s and 2s;
        # sample standard deviation s/sqrt(2), scores +/-1/sqrt(2).
        ra, ba = fixture([0.] * 9 + [.1], stock='000001')
        rb, bb = fixture([0.] * 9 + [.2], stock='000002')
        r = pd.concat([ra, rb], ignore_index=True)
        r.attrs['units'] = {'return': 'decimal'}
        b = pd.concat([ba, bb], ignore_index=True)
        f, _ = self.checked(r, b)
        last = f[f['base_month'] == '2000-10']
        np.testing.assert_allclose(last['h3'], [-1 / math.sqrt(2), 1 / math.sqrt(2)], atol=1e-14)
        self.assertTrue(last['h3_available'].all())
        single, _ = self.checked(ra, ba)
        self.assertEqual(single.iloc[-1]['h3_exclusion_reason'], 'too_few_stocks')
        r.loc[r['stock_id'] == '000002', 'return'] = ra['return'].to_numpy()
        equal, _ = self.checked(r, b)
        self.assertTrue(equal[equal['base_month'] == '2000-10']['h3_exclusion_reason'].eq('zero_or_nonfinite_sd').all())

    def test_16_input_guards(self):
        r, b = fixture([.1] * 12)
        bads = []
        for value in (np.inf, -np.inf, '0.1', 'bad', True, -1.01):
            bad = r.copy(deep=True)
            bad['return'] = bad['return'].astype(object)
            bad.at[0, 'return'] = value
            bads.append(bad)
        bad = r.copy(deep=True)
        bad.at[0, 'month'] = '2000-13'
        bads.append(bad)
        bads.append(pd.concat([r, r.iloc[:1]], ignore_index=True))
        bad = r.copy(deep=True)
        bad.at[0, 'stock_id'] = '1'
        bads.append(bad)
        for case, bad in enumerate(bads):
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    calculate_h3_ewma_feature(bad, b)
        for units in ({}, {'return': 'percent'}):
            bad = r.copy(deep=True)
            bad.attrs['units'] = units
            with self.assertRaises(ValueError):
                calculate_h3_ewma_feature(bad, b)
        for kwargs in ({'decay': .95}, {'recent_window': 11}, {'recent_min_valid': 9}):
            with self.assertRaises(ValueError):
                calculate_h3_ewma_feature(r, b, **kwargs)
        for previous, current in (('2000-02', '2000-02'), ('2000-02', '2000-01'), ('2000-02', '2000-04')):
            with self.assertRaises(ValueError):
                update_ewma_state(.1, .1, previous_month=previous, current_month=current)
        for value in (-.1, np.inf):
            with self.assertRaises(ValueError):
                update_ewma_state(value, .1, previous_month='2000-01', current_month='2000-02')
        with self.assertRaises(ValueError):
            initialize_ewma_state([.1] * 10, ddof=0)
        with self.assertRaises(ValueError):
            initialize_ewma_state([.1] * 11)
        with self.assertRaises(ValueError):
            calculate_h3_ewma_feature(r, pd.concat([b, b.iloc[:1]], ignore_index=True))
        edge = r.copy(deep=True)
        edge.at[0, 'return'] = -1.
        self.checked(edge, b)  # Exactly -100% is valid, not below -1.

    def test_unknown_eligibility_stays_unknown_and_withholds_monthly_score(self):
        ra, ba = fixture([0.] * 9 + [.1], stock='000001')
        rb, bb = fixture([0.] * 9 + [.2], stock='000002')
        rc, bc = fixture([0.] * 9 + [.3], stock='000003', eligible=pd.NA)
        r = pd.concat([ra, rb, rc], ignore_index=True)
        r.attrs['units'] = {'return': 'decimal'}
        b = pd.concat([ba, bb, bc], ignore_index=True)
        f, _ = self.checked(r, b)
        last = f[f['base_month'] == '2000-10']
        self.assertTrue(last['sqrt_v'].notna().all())
        self.assertTrue(last['h3'].isna().all())
        self.assertTrue(pd.isna(last.iloc[-1]['eligible']))
        self.assertEqual(last.iloc[0]['h3_exclusion_reason'], 'monthly_universe_unverified')
        absent, _ = self.checked(ra, ba.drop(columns='eligible'))
        self.assertTrue(absent['eligible'].isna().all())

    def test_listing_coverage_and_evidence_are_checked(self):
        r, b = fixture([.1] * 12)
        h = history(r)
        for bad in (h.iloc[1:].copy(), pd.concat([h, h.iloc[:1]], ignore_index=True)):
            with self.assertRaises(ValueError):
                calculate_h3_ewma_feature(r, b, listing_history=bad)
        for column, value in (('is_listed', False), ('basis', ''), ('is_listed', None)):
            bad = h.copy(deep=True)
            bad[column] = bad[column].astype(object)
            bad.at[0, column] = value
            with self.assertRaises(ValueError):
                calculate_h3_ewma_feature(r, b, listing_history=bad)

    def test_validator_detects_corrupt_state_and_features(self):
        r, b = fixture([-.1, .1] * 5 + [0., np.nan])
        f, s = self.checked(r, b)
        for col, value in (('v', .5), ('action', 'initialize'), ('recent_valid_count', 1),
                           ('reset', True), ('n_valid_since_reset', 1), ('reset_basis', 'invented')):
            bad = s.copy(deep=True)
            bad.at[11, col] = value
            with self.subTest(state=col), self.assertRaises(ValueError):
                validate_h3_contract(f, bad, r)
        for col, value in (('sqrt_v', .5), ('vol_available', False), ('h3', 9.),
                           ('h3_available', True), ('standardization_n', 20)):
            bad = plain(f)
            bad.at[11, col] = value
            with self.subTest(feature=col), self.assertRaises(ValueError):
                validate_h3_contract(bad, s, r)
        with self.assertRaises(ValueError):
            validate_h3_contract(f, s.drop(index=3), r)

    def test_no_annual_reset_no_io_and_target_independence(self):
        r, b = fixture([-.1, .1] * 18)
        b['y'] = np.nan
        b['h1'] = np.nan
        with patch('builtins.open', side_effect=AssertionError('file I/O forbidden')), \
             patch('socket.socket.connect', side_effect=AssertionError('network forbidden')):
            f, s = self.checked(r, b)
        self.assertEqual(s['initialized_this_month'].sum(), 1)
        self.assertFalse(s['reset'].any())
        b['y'], b['h1'] = 100., -100.
        other, _ = self.checked(r, b)
        assert_frame_equal(plain(f), plain(other))
        self.assertEqual(set(s.columns), {
            'stock_id', 'base_month', 'return', 'observed_row', 'initialized',
            'initialized_this_month', 'initialization_month', 'n_valid_since_reset',
            'v', 'action', 'recent_valid_count', 'reset', 'reset_reason', 'reset_basis',
            'is_listed', 'unlisted_months', 'segment'})


if __name__ == '__main__':
    unittest.main()
