# 5단계 공개일 붙이기·검증(equity_dating): 접수일 대응, 보정 파일, DART 대조, 차이·정정 처리.

import unittest


class DataGuideEquityTests(unittest.TestCase):
    """DataGuide equity rows dated by DART receipts; tiny synthetic workbooks, no network."""

    @staticmethod
    def workbook(path, rows):
        """Minimal xlsx: one worksheet of inline strings and numbers, enough for the streaming reader."""
        import zipfile
        from xml.sax.saxutils import escape

        def col(i):
            name = ''
            i += 1
            while i:
                i, r = divmod(i - 1, 26)
                name = chr(65 + r) + name
            return name
        body = []
        for r, cells in enumerate(rows, 1):
            out = []
            for i, v in enumerate(cells):
                if v is None:
                    continue
                ref = f'{col(i)}{r}'
                if isinstance(v, (int, float)):
                    out.append(f'<c r="{ref}"><v>{v}</v></c>')
                else:
                    out.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(v)}</t></is></c>')
            body.append(f'<row r="{r}">{"".join(out)}</row>')
        xml = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
               + ''.join(body) + '</sheetData></worksheet>')
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('xl/worksheets/sheet1.xml', xml)

    def frames(self, **extra):
        import pandas as pd
        base = {'code': 'A000001', 'name': '가나', 'fy': '2005', 'quarter': '결산'}
        f = lambda **v: pd.DataFrame([{**base, **v}])
        return {'kgaap_separate': f(sep=100.0), 'kgaap_consolidated': f(cons=150.0, nci=20.0, owners=None),
                'kgaap_fiscal_month': f(fm='12'), 'kifrs_consolidated': f(cons=999.0, owners=900.0, nci=None, fm='12'),
                'kifrs_separate': f(sep=888.0), **extra}

    def test_consolidated_kgaap_value_needs_consolidated_statements_in_the_dated_report(self):
        import pandas as pd
        from src.preprocessing.equity_dating import apply_kgaap_consolidated_rule
        panel = pd.DataFrame({'std': ['KGAAP', 'KGAAP', 'KIFRS'], 'tier': ['T2_consolidated_total'] * 3, 'equity': [150.0] * 3,
                              'sep': [100.0, None, 90.0], 'same_day_receipts': ['r1;r2', 'r3', 'r4']})
        out = apply_kgaap_consolidated_rule(panel, {'r2'})
        self.assertEqual(out.tier.tolist(), ['T2_consolidated_total', 'no_value_consolidated_not_in_report', 'T2_consolidated_total'])
        self.assertEqual(out.kgaap_consolidated_in_report.tolist(), ['yes', 'no', None])

    def test_receipts_by_code_then_only_unambiguous_names(self):
        import pandas as pd
        from src.preprocessing.equity_dating import attach_receipts
        filings = pd.DataFrame([
            ('c1', '가나', '000001', 'r2', '20040331', '사업', '2003-12'),
            ('c1', '가나', '000001', 'r1', '20040330', '사업', '2003-12'),
            ('c1', '가나', '000001', 'r0', '20040330', '사업', '2003-12'),
            ('c2', '(주)다라', '', 'r5', '20040329', '사업', '2003-12'),
            ('c3', '마바', '', 'r6', '20040329', '사업', '2003-12'),
            ('c4', '마바', '', 'r7', '20040329', '사업', '2003-12')],
            columns=['corp_code', 'corp_name', 'stock_code', 'rcept_no', 'rcept_dt', 'report_kind', 'report_period_label'])
        panel = pd.DataFrame({'code': ['A000001', 'A000002', 'A000003'], 'stock_code': ['000001', '000002', '000003'],
                              'name': ['가나', '다라', '마바'], 'month': ['2003-12'] * 3, 'kind': ['사업'] * 3})
        out = attach_receipts(panel, filings)
        self.assertEqual(out.rcept_dt.fillna('none').tolist(), ['20040330', '20040329', 'none'])
        self.assertEqual(out.same_day_receipts[0], 'r0;r1')
        self.assertEqual(out.receipt_basis.fillna('none').tolist(), ['stock_code', 'unique_name', 'none'])  # the shared name is refused
        self.assertEqual(out.no_receipt_reason.fillna('').tolist(), ['', '', 'dart_name_shared'])

    def test_receipt_fallbacks_stay_inside_one_dart_company(self):
        import pandas as pd
        from src.preprocessing.equity_dating import attach_receipts
        columns = ['corp_code', 'corp_name', 'stock_code', 'rcept_no', 'rcept_dt', 'report_kind', 'report_period_label']
        filings = pd.DataFrame([
            ('c1', '가나', '000001', 'a1', '20000515', '분기', '2000-12'),   # title period after its own receipt
            ('c1', '가나', '000001', 'a2', '20010331', '사업', '2000-12'),
            ('c2', '다라', '000002', 'b1', '20130814', '분기', '2013-06'),   # fiscal-year change: 분기 here, 반기 in DataGuide
            ('c2', '다라', '000002', 'b2', '20131114', '반기', '2013-09'),
            ('c2', '다라', '000002', 'b3', '20140331', '사업', '2013-12'),
            ('c3', '은행', '', 'd1', '20000513', '분기', '2000-03'),
            ('c4', '은행', '', 'd2', '20000513', '분기', '2000-03'),
            ('c5', '마바', '000005', 'e1', '20100515', '분기', '2010-03'),
            ('c9', '끝', '000009', 'z1', '20220110', '분기', '2021-09')], columns=columns)
        panel = pd.DataFrame({'code': ['A000001', 'A000002', 'A000002', 'A000003', 'A000004', 'A000005', 'A000005'],
                              'stock_code': ['000001', '000002', '000002', '000003', '000004', '000005', '000005'],
                              'name': ['가나', '다라', '다라', '은행', '은행', '마바', '마바'],
                              'month': ['2000-03', '2013-06', '2013-09', '2000-03', '2000-03', '2009-12', '2021-12'],
                              'kind': ['분기', '반기', '분기', '분기', '분기', '사업', '사업'],
                              'sep': [1.0, 5_000_000.0, 7_000_000.0, 3.0, 3.0, 4.0, 4.0]})
        out = attach_receipts(panel, filings, amounts={'b1': [5_000_400.0], 'b2': [9_000_000.0]}, corp_map={'000003': 'c3'})
        self.assertEqual(out.rcept_no.fillna('').tolist(), ['a1', 'b1', '', 'd1', '', '', ''])
        self.assertEqual(out.receipt_basis.fillna('').tolist(), ['label_after_receipt', 'same_date_other_kind', '', 'corp_map', '', '', ''])
        self.assertEqual(out.no_receipt_reason.fillna('').tolist()[2:],
                         ['same_date_other_kind_amount_differs', '', 'dart_name_shared', 'dart_filings_start_later',
                          'receipt_after_dart_list_end'])

    def test_a_filing_received_before_its_title_period_ended_is_not_that_periods_report(self):
        import pandas as pd
        from src.preprocessing.equity_dating import attach_receipts
        filings = pd.DataFrame([('c1', 'KT', '030200', 'k0', '20010331', '사업', '2001-12'),   # really the 2000 report
                                ('c1', 'KT', '030200', 'k1', '20020329', '사업', '2001-12'),
                                ('c9', '끝', '000009', 'z1', '20220110', '분기', '2021-09')],
                               columns=['corp_code', 'corp_name', 'stock_code', 'rcept_no', 'rcept_dt', 'report_kind', 'report_period_label'])
        panel = pd.DataFrame({'code': ['A030200'] * 2, 'stock_code': ['030200'] * 2, 'name': ['KT'] * 2,
                              'month': ['2000-12', '2001-12'], 'kind': ['사업'] * 2, 'sep': [1.0, 2.0]})
        out = attach_receipts(panel, filings)
        self.assertEqual(list(zip(out.rcept_no, out.receipt_basis)), [('k0', 'label_after_receipt'), ('k1', 'stock_code')])

    def test_dart_comparison_classes_and_balance_identity(self):
        import pandas as pd
        from src.preprocessing.equity_dating import compare_with_dart, identity_check
        filings = pd.DataFrame([('c1', f'r{i}', d, '사업', '2005-12') for i, d in enumerate(['20060331', '20060331', '20060601'])],
                               columns=['corp_code', 'rcept_no', 'rcept_dt', 'report_kind', 'report_period_label'])
        base = dict(rcept_no='r0', same_day_receipts='r0;r1', sep=None, cons=None, owners=None, nci=None)
        panel = pd.DataFrame([{**base, 'equity': 100e6}, {**base, 'equity': 300e6, 'sep': 300e6, 'cons': 250e6},
                              {**base, 'equity': -100e6}, {**base, 'equity': 777e6}, {**base, 'equity': 555e6},
                              {**base, 'rcept_no': None, 'equity': 1.0}])
        out = compare_with_dart(panel, filings, {'r1': (100e6, 'T3'), 'r2': (777e6, 'T3')}, {'r0': [250e6]})
        self.assertEqual(out.dart_compare.fillna('none').tolist(), ['same', 'tier_only', 'sign', 'later_filing_only', 'unexplained', 'none'])
        rows = pd.DataFrame({'tier': ['T3_separate_total', 'T1_consolidated_owners', 'T3_separate_total', 'T3_separate_total'],
                             'sep_assets': [500.0, None, 9e9, 9e9], 'sep_liabilities': [400.0, None, 1e9, 3.432e9],
                             'sep': [100.0, None, 7e9, 5.567e9],  # last: one million apart, the rounding of old data
                             'cons_assets': [None, 900.0, None, None], 'cons_liabilities': [None, 300.0, None, None],
                             'cons': [None, 600.0, None, None]})
        self.assertEqual(identity_check(rows).tolist(), ['ok', 'ok', 'fail', 'ok'])
        from src.preprocessing.equity_dating import identity_gap_ratio
        self.assertAlmostEqual(identity_gap_ratio(rows)[2], 1e9 / 7e9)  # |9e9 - 1e9 - 7e9| / 7e9

    def test_unexplained_differences_keep_dataguide_only_for_dart_reading_slips(self):
        import pandas as pd
        from pathlib import Path
        from src.preprocessing.equity_dating import resolve_differences
        rows = [('2005-12', 100e9, 'unexplained', 90e9),        # DART read the 2004 column
                ('2004-12', 90e9, 'same', 90e9),
                ('2006-12', 5e9, 'unexplained', 5e12),          # unit
                ('2007-12', 639_900_200e3, 'unexplained', 639_529e6),  # a real gap of 371 million, not rounding
                ('2008-12', -80e9, 'unexplained', 503e9),       # restated later: what was public then is DART
                ('2009-12', 7e9, 'same', 7e9)]
        panel = pd.DataFrame([dict(stock_code='000001', month=m, equity=e, dart_compare=c, dart_equity=d, same_day_receipts='r')
                              for m, e, c, d in rows])
        out = resolve_differences(panel, Path('.'), {})
        self.assertEqual(out.fillna('').tolist(), ['keep_dataguide:dart_read_other_period', '', 'keep_dataguide:dart_unit_slip',
                                                   'use_dart_first', 'use_dart_first', ''])

    def test_same_amount_means_equal_in_the_coarser_printing_unit(self):
        from src.preprocessing.equity_dating import same_amount
        self.assertTrue(same_amount(1_718_662_000, 1_718_000_000))     # thousands vs millions, cut
        self.assertTrue(same_amount(191_393_232_000, 191_393_000_000))
        self.assertFalse(same_amount(7_000_000, 9_000_000))
        self.assertTrue(same_amount(16_859_033_000, 16_859_032_000))  # rounded vs cut
        self.assertTrue(same_amount(229_623_948_230, 229_623_948_228))   # DataGuide rounds K-IFRS values to 10 won
        self.assertTrue(same_amount(1_312_298_403_465, 1_312_298_403_000))   # won vs thousands
        self.assertFalse(same_amount(1_312_298_403_465, 1_312_298_402_465))  # both in won: must match to the won

    def test_override_criteria_are_rechecked(self):
        import pandas as pd
        from src.preprocessing.equity_dating import override_candidates, verify_overrides
        filings = pd.DataFrame([('c1', '광주은행', '', 'r1', '20000513', '분기', '2000-03'),
                                ('c2', '광주은행', '', 'r2', '20150513', '분기', '2015-03'),
                                ('c3', '해동상호저축은행', '', 'r3', '20000928', '사업', '2000-06'),
                                ('c4', '가나', '000004', 'r4', '20020401', '사업', '2001-12')],
                               columns=['corp_code', 'corp_name', 'stock_code', 'rcept_no', 'rcept_dt', 'report_kind', 'report_period_label'])
        base = dict(sep=None, cons=None, owners=None, rcept_no=None, month_note=None, quarter='결산')
        panel = pd.DataFrame([{**base, 'stock_code': '005770', 'name': '(주)광주은행', 'month': '2000-03', 'month_dataguide': '2000-03', 'fy': '2000', 'equity': 3e11},
                              {**base, 'stock_code': '005770', 'name': '(주)광주은행', 'month': '2000-06', 'month_dataguide': '2000-06', 'fy': '2000', 'equity': 4e11},
                              {**base, 'stock_code': '007420', 'name': '해동신용금고', 'month': '2000-06', 'month_dataguide': '2000-06', 'fy': '2000', 'equity': 5e9},
                              {**base, 'stock_code': '000004', 'name': '가나', 'month': '2002-03', 'month_dataguide': '2002-03', 'fy': '2001', 'equity': 9e9}])
        facts = pd.DataFrame([('c1', 'r1', 3e11, '2000-03'), ('c1', 'r1', 4e11, '2000-06'), ('c4', 'r4', 9e9, '2001-12')],
                             columns=['corp_code', 'receipt_id', 'v', 'fpm'])
        cand = override_candidates(panel, filings, facts)
        self.assertEqual(sorted(zip(cand.type, cand.stock_code, cand.basis, cand.evidence)),
                         [('corp_link', '005770', 'amount', '2000-03,2000-06'), ('corp_link', '007420', 'rename', '해동신용금고'),
                          ('statement_month', '000004', 'free', '2001-12')])
        entries = [{'id': 'A', 'type': 'corp_map', 'status': 'approved', 'stock_code': '005770', 'set': {'corp_code': 'c1'}, 'basis': 'amount'},
                   {'id': 'B', 'type': 'corp_map', 'status': 'approved', 'stock_code': '005770', 'set': {'corp_code': 'c2'}, 'basis': 'amount'},
                   {'id': 'C', 'type': 'statement_month', 'status': 'approved', 'stock_code': '000004', 'fy': '2001', 'quarter': '결산',
                    'set': {'month': '2001-12'}}]
        self.assertEqual(verify_overrides(entries, cand), ['B:criterion_not_met'])

    def test_corrections_are_never_back_dated(self):
        import pandas as pd
        from src.preprocessing.equity_dating import correction_versions, last_trading_days
        filings = pd.DataFrame([('c1', 'r0', '20190814', '반기', '2019-06'), ('c1', 'r1', '20191002', '반기', '2019-06'),
                                ('c2', 's0', '20140331', '사업', '2013-12'), ('c2', 's1', '20140404', '사업', '2013-12'),
                                ('c2', 's2', '20140501', '사업', '2013-12'),
                                ('c3', 'u0', '20190814', '반기', '2019-06'), ('c3', 'u1', '20191002', '반기', '2019-06')],
                               columns=['corp_code', 'rcept_no', 'rcept_dt', 'report_kind', 'report_period_label'])
        t1 = 'T1_consolidated_owners'
        panel = pd.DataFrame([dict(stock_code='000001', month='2019-06', tier=t1, equity=3011e9, rcept_no='r0', rcept_dt='20190814',
                                   same_day_receipts='r0', dart_compare='later_filing_only'),
                              dict(stock_code='000002', month='2013-12', tier=t1, equity=1044e8, rcept_no='s0', rcept_dt='20140331',
                                   same_day_receipts='s0', dart_compare='same'),
                              dict(stock_code='000003', month='2019-06', tier=t1, equity=5e9, rcept_no='u0', rcept_dt='20190814',
                                   same_day_receipts='u0', dart_compare='later_filing_only')])
        decisions = {'r0': (3019e9, t1), 'r1': (3011e9, t1), 's0': (1044e8, t1), 's1': (1078e8, t1), 's2': (1078e8, 'T3_separate_total'),
                     'u1': (5e9, t1)}  # u0: the first original gave no value
        days = last_trading_days(pd.Series(pd.to_datetime(['2014-03-31', '2014-04-30', '2019-08-30', '2019-10-31'])))
        out = correction_versions(panel, filings, decisions, {}, days)
        self.assertEqual(list(zip(out.stock_code, out.equity, out.source, out.available_month)), [
            ('000001', 3019e9, 'dart_first_before_correction', '2019-08'), ('000001', 3011e9, 'dataguide_from_correction', '2019-10'),
            ('000002', 1044e8, 'dataguide', '2014-04'), ('000002', 1078e8, 'dart_correction', '2014-04'),  # s2: other tier ignored
            ('000003', 5e9, 'dataguide_possibly_corrected', '2019-08')])

    def test_month_override_moves_only_listed_rows_and_refuses_stale_entries(self):
        import pandas as pd
        from src.preprocessing.equity_dating import apply_month_overrides
        panel = pd.DataFrame({'code': ['A000001'] * 2, 'stock_code': ['000001'] * 2, 'fy': ['2001', '2001'],
                              'quarter': ['결산', '3rd-Quarter'], 'month': ['2002-03', '2001-12']})
        entries = [{'id': 'X1', 'type': 'statement_month', 'stock_code': '000001', 'fy': '2001', 'quarter': '결산',
                    'month_dataguide': '2002-03', 'set': {'month': '2001-12'}},
                   {'id': 'X2', 'type': 'statement_month_unreliable', 'stock_code': '000001', 'fy': '2001', 'quarter': '3rd-Quarter',
                    'month_dataguide': '2001-12'}]
        out = apply_month_overrides(panel, entries)
        self.assertEqual(out.month.tolist(), ['2001-12', '2001-12'])  # same month twice is allowed only because one is unusable
        self.assertEqual(out.month_note.tolist(), ['corrected:X1', 'unreliable:X2'])
        with self.assertRaisesRegex(ValueError, 'collides'):
            apply_month_overrides(panel, entries[:1])
        with self.assertRaisesRegex(ValueError, 'stale'):
            apply_month_overrides(panel, [{**entries[0], 'month_dataguide': '2002-06'}])

    def test_month_end_receipt_counts_from_the_next_month(self):
        import pandas as pd
        from src.preprocessing.equity_dating import last_trading_days, usable_month
        days = last_trading_days(pd.Series(pd.to_datetime(['2004-03-29', '2004-03-30', '2004-04-30'])))
        self.assertEqual([usable_month(d, days) for d in ('20040329', '20040330', '20040331', None)],
                         ['2004-03', '2004-04', '2004-04', None])
