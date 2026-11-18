# 6단계 as-of(equity_asof): 종목×월 선택, 만료, DART 거르기, 내장 검사.

import unittest


class EquityAsofTests(unittest.TestCase):
    """C1: stock x month equity as-of from the version table. Synthetic frames only."""

    @staticmethod
    def days():
        import pandas as pd
        from src.preprocessing.equity_dating import last_trading_days
        return last_trading_days(pd.Series(pd.date_range('2004-01-01', '2006-12-31', freq='B')))

    def test_newest_period_then_latest_version_and_18_month_expiry(self):
        import pandas as pd
        from src.preprocessing.equity_asof import build_asof
        t1, t3 = 'T1_consolidated_owners', 'T3_separate_total'
        versions = pd.DataFrame([
            ('000001', '2004-12', 0, t3, 100.0, 'dataguide', 'a', '20050331', '2005-04'),
            ('000001', '2004-12', 1, t3, 110.0, 'dart_correction', 'b', '20050610', '2005-06'),
            ('000001', '2005-03', 0, t1, 120.0, 'dataguide', 'c', '20050516', '2005-05')],
            columns=['stock_code', 'month', 'seq', 'tier', 'equity', 'source', 'rcept_no', 'rcept_dt', 'available_month'])
        months = ['2005-03', '2005-04', '2005-05', '2005-06', '2006-09', '2006-10']
        out = build_asof(versions, ['000001', '000002'], months, self.days()).set_index(['stock_code', 'base_month'])
        a = out.loc['000001']
        self.assertEqual(a.reason['2005-03'], 'before_first_public_value')
        self.assertEqual(a.equity['2005-04'], 100.0)
        self.assertEqual(a.equity['2005-05'], 120.0)   # the newer period wins, whatever its tier (P5)
        self.assertEqual(a.equity['2005-06'], 120.0)   # the 2004-12 correction does not replace a newer period
        self.assertEqual(a.equity['2006-09'], 120.0)   # 2005-03 + 18 months -> through 2006-09-30
        self.assertEqual(a.reason['2006-10'], 'expired_18m')
        self.assertEqual(out.loc['000002'].reason.unique().tolist(), ['no_dataguide_value'])

    def test_failing_dart_versions_fall_back_and_checks_stop_the_run(self):
        import pandas as pd
        from src.preprocessing.equity_asof import checked_versions, run_checks
        t1 = 'T1_consolidated_owners'
        panel = pd.DataFrame([('000001', '2019-06', 3011.0, t1, 'r0', '20190814', '2019-08')],
                             columns=['stock_code', 'month', 'equity', 'tier', 'rcept_no', 'rcept_dt', 'available_month'])
        versions = pd.DataFrame([
            ('000001', '2019-06', 0, t1, 3019.0, 'dart_first_before_correction', 'r0', '20190814', '2019-08'),
            ('000001', '2019-06', 1, t1, 3011.0, 'dataguide_from_correction', 'r1', '20191002', '2019-10')],
            columns=['stock_code', 'month', 'seq', 'tier', 'equity', 'source', 'rcept_no', 'rcept_dt', 'available_month'])
        decisions = pd.DataFrame([('r0', '3019', t1, 'unit_suspect', None)],
                                 columns=['receipt_id', 'equity', 'equity_tier', 'skip', 'override_id'])
        out, counts = checked_versions(versions, decisions, panel)
        self.assertEqual(list(zip(out.source, out.equity, out.available_month)),
                         [('dataguide_possibly_corrected', 3011.0, '2019-08')])
        self.assertEqual(counts['fell_back_to_dataguide'], 1)
        ok = decisions.assign(skip=pd.array([pd.NA], dtype='string'))  # how a passing decision is stored
        out, counts = checked_versions(versions, ok, panel)
        self.assertEqual(list(zip(out.source, out.equity)), [('dart_first_before_correction', 3019.0),
                                                             ('dataguide_from_correction', 3011.0)])
        bad = pd.DataFrame([dict(stock_code='000001', base_month='2019-07', equity=1.0, available_month='2019-08',
                                 statement_month='2019-06', expiry_date='2020-12-31', reason=None)])
        with self.assertRaisesRegex(ValueError, 'value_before_usable_month'):
            run_checks(bad, self.days())
