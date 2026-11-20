"""기본 피처(src/features/base.py) 규칙 검사: 작은 가짜 자료만 쓴다(실제 data/ 파일을 읽지 않음)."""
import unittest

import numpy as np
import pandas as pd

from src.features import base


def _daily(dates, r, volume=None, stock='000001', close=None, high=None, amount=None):
    n = len(dates)
    close = close if close is not None else 100 * np.cumprod(1 + np.asarray(r))
    return pd.DataFrame({'stock_id': stock, 'date': pd.to_datetime(dates), 'r': r,
                         'Close': close, 'High': high if high is not None else close,
                         'Volume': volume if volume is not None else [10] * n,
                         'Amount': amount if amount is not None else [1000.0] * n}).assign(
        month=lambda x: x.date.dt.strftime('%Y-%m'))


class GuardTests(unittest.TestCase):
    def test_test_period_paths_blocked(self):
        for p in ('data/raw/external_data/test_sealed/x.xlsx', 'data/raw/test_2022_2025__adjusted_monthly_return.csv'):
            with self.assertRaises(PermissionError):
                base.guard_development(p)
        base.guard_development('data/raw/external_data/2009_2021_연결_feature.xlsx')


class WindowTests(unittest.TestCase):
    def test_long_term_reversal_window_and_missing(self):
        months = pd.period_range('2000-01', '2003-12', freq='M').astype(str)
        r = pd.Series(0.01, index=pd.MultiIndex.from_product([['000001'], months], names=['stock_id', 'month']))
        w = base._wide(r, months)
        ltr = np.expm1(np.log1p(w).rolling(24, min_periods=24).sum().shift(12))
        # t = 2002-12: t-35 = 2000-01 … t-12 = 2001-12, 24개월
        self.assertAlmostEqual(ltr.loc['2002-12', '000001'], 1.01 ** 24 - 1)
        self.assertTrue(np.isnan(ltr.loc['2002-11', '000001']))   # 창이 2000-01 이전으로 나감
        w.loc['2000-02', '000001'] = np.nan
        ltr2 = np.expm1(np.log1p(w).rolling(24, min_periods=24).sum().shift(12))
        self.assertTrue(np.isnan(ltr2.loc['2002-12', '000001']))  # 창(2000-01~2001-12) 안 하나라도 결측이면 미산출
        self.assertTrue(np.isnan(ltr2.loc['2003-01', '000001']))  # 창 2000-02~2002-01
        self.assertFalse(np.isnan(ltr2.loc['2003-02', '000001']))  # 창 2000-03~2002-02: 2000-02를 벗어남

    def test_wide_long_roundtrip(self):
        idx = pd.MultiIndex.from_tuples([('a', '2000-01'), ('b', '2000-02')], names=['stock_id', 'month'])
        s = pd.Series([1.0, 2.0], index=idx)
        back = base._long(base._wide(s, ['2000-01', '2000-02']), 'x').dropna()
        self.assertEqual(back.to_dict(), s.to_dict())


class DailyTests(unittest.TestCase):
    def test_market_return_uses_previous_day_cap(self):
        k = pd.DataFrame({'stock_id': ['a', 'a', 'b', 'b'], 'date': pd.to_datetime(['2000-01-03', '2000-01-04'] * 2),
                          'r': [0.0, 0.10, 0.0, -0.10], 'Marcap': [300.0, 330.0, 100.0, 90.0]})
        m = base.market_daily_return(k)
        self.assertAlmostEqual(m.loc['2000-01-04'], (300 * 0.10 + 100 * -0.10) / 400)
        self.assertNotIn(pd.Timestamp('2000-01-03'), m.index)        # 전일 시가총액이 없는 첫날은 제외

    def test_halt_day_excluded_and_ivol_min_days(self):
        dates = pd.bdate_range('2000-01-03', periods=20)
        r = np.linspace(-0.02, 0.03, 20)
        vol = [10] * 20
        vol[5] = 0                                                   # 거래정지일
        r[5] = 0.5                                                   # 거래정지일 값은 MAX에 들어가면 안 됨
        d = _daily(dates, r, volume=vol)
        mkt = pd.Series(np.linspace(-0.01, 0.01, 20), index=dates)
        out = base.daily_monthly(d, mkt, ivol_min_days=15).set_index('month')
        row = out.loc['2000-01']
        self.assertEqual(row.trade_days, 19)
        self.assertAlmostEqual(row.max_ret, 0.03)
        self.assertAlmostEqual(row.r_px, np.prod(1 + r) - 1)          # 주가수익률은 모든 날 누적
        self.assertFalse(np.isnan(row.ivol))
        out2 = base.daily_monthly(d, mkt, ivol_min_days=25).set_index('month')
        self.assertTrue(np.isnan(out2.loc['2000-01'].ivol))          # 거래일 부족 → 미산출

    def test_split_adjusted_high(self):
        dates = pd.bdate_range('2000-01-03', periods=3)
        # 분할: 종가 100 → 50인데 일수익률은 0(분할 보정) → 수정 고가는 분할 전과 같은 단위
        d = _daily(dates, [0.0, 0.0, 0.0], close=[100.0, 50.0, 50.0], high=[100.0, 50.0, 50.0])
        out = base.daily_monthly(d, pd.Series(0.0, index=dates), 15).set_index('month')
        self.assertAlmostEqual(out.loc['2000-01'].hi_m / out.loc['2000-01'].last_idx, 1.0)


class BetaTests(unittest.TestCase):
    def test_beta_two_and_min_weeks(self):
        dates = pd.bdate_range('2000-01-03', '2002-12-31')
        rng = np.random.default_rng(0)
        m = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
        # 주별 수익률이 시장의 정확히 2배가 되도록 주마다 하루만 움직임
        week = dates.to_period('W-FRI')
        first = ~pd.Series(week).duplicated().to_numpy()
        m = m.where(first, 0.0)
        d = _daily(dates, (2 * m).to_numpy())
        months = ['2000-06', '2001-06', '2002-12']
        b = base.weekly_beta(d, m, months).dropna()
        self.assertNotIn(('000001', '2000-06'), b.index)             # 52주 미만
        self.assertAlmostEqual(b.loc[('000001', '2002-12')], 2.0, places=6)


class AccountingTests(unittest.TestCase):
    def test_asof_availability_expiry_and_previous_assets(self):
        st = pd.DataFrame({'code': 'A000001', 'stock_code': '000001', 'fy': ['2009', '2010'], 'month': ['2009-12', '2010-12'],
                           'std': ['KGAAP', 'KGAAP'], 'tier': 'T3_separate_total', 'side': 'sep',
                           'available_month': ['2010-03', '2011-04'], 'rcept_dt': ['20100330', '20110401'],
                           'cons_assets': [np.nan, np.nan], 'sep_assets': [100.0, 200.0]})
        fl = pd.DataFrame({'code': 'A000001', 'fy': ['2009', '2010'], 'ni_owner': [10.0, 30.0], 'std': 'KGAAP', 'side': 'sep'})
        months = pd.period_range('2010-01', '2012-12', freq='M').astype(str).tolist()
        last = pd.Series(pd.to_datetime([pd.Period(m, 'M').to_timestamp(how='end') for m in months]), index=months)
        a = base.accounting_asof(st, fl, months, last).set_index('base_month')
        self.assertNotIn('2010-02', a.index)                              # 공개 전
        self.assertEqual(a.loc['2010-03'].ni_owner, 10.0)
        self.assertTrue(np.isnan(a.loc['2010-03'].assets_prev))          # 2008년 말 자산 없음
        self.assertEqual(a.loc['2011-03'].statement_month, '2009-12')     # 2010 사업보고서는 4월 공개
        self.assertEqual(a.loc['2011-04'].ni_owner, 30.0)
        self.assertEqual(a.loc['2011-04'].assets_prev, 100.0)            # 전년 말 총자산(같은 개별)
        self.assertEqual(a.loc['2012-06'].statement_month, '2010-12')
        self.assertEqual(a.loc['2012-07'].acc_reason, 'expired_18m')      # 2010-12 + 18개월 = 2012-06 말 만료


if __name__ == '__main__':
    unittest.main()
