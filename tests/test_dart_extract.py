# 2단계 DART 값 추출(dart_extract): 원문 → 자기자본 사실·공시별 결정, DART 전용 as-of 규칙.

import unittest


class DartEquityAsofTests(unittest.TestCase):
    """Stock x base-month as-of rules on synthetic filings; no files or network."""

    LAST = {f"{y}-{m:02d}": f"{y}{m:02d}28" for y in range(2019, 2023) for m in range(1, 13)}
    LAST.update({"2020-03": "20200331", "2021-03": "20210331", "2021-04": "20210430"})

    @staticmethod
    def filing(receipt, label, published, equity, *, revision='최초표시', tier='T1_consolidated_owners', scope='CFS',
               fiscal='auto', unit='False', reason=None, corp='C1'):
        import pandas as pd
        end = (pd.Timestamp(label + '-01') + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d') if fiscal == 'auto' else fiscal
        return {'corp_code': corp, 'receipt_id': receipt, 'published_at': published, 'availability_month': None,
                'report_kind': '사업', 'report_period_label': label, 'revision': revision, 'report_name': 'x',
                'fiscal_period': end, 'equity': equity, 'equity_tier': tier if equity else None, 'statement_type': scope,
                'account_name': 'a', 'unit': 'KRW', 'source_kind': 'xbrl', 'section': 'xbrl', 'column_basis': 'c',
                'identity_ok': None, 'xbrl_document_agree': None, 'nci_observed': False, 'unit_suspect': unit,
                'reason': reason, 'raw_path': 'p/' + receipt, 'sha256': 'h' + receipt}

    def join(self, filings, months, *, stocks=(('A', 'C1', 'current_exact'),), complete_from=None):
        import pandas as pd
        from src.preprocessing.dart_extract import join_equity_asof
        panel = pd.DataFrame([{'stock_id': s, 'corp_code': c, 'mapping_status': st, 'base_month': m}
                              for s, c, st in stocks for m in months])
        frame = pd.DataFrame(filings).astype('string')
        return join_equity_asof(panel, frame, last_trading_day=self.LAST, history_complete_from=complete_from).set_index('base_month')

    def unit_case(self, attached, *, section='full', override=None):
        """Two normal years, then a filing whose body and XBRL both say 10^6 too much; `attached` is the audit report's amount."""
        import pandas as pd
        rows = [self.filing('R1', '2019-12', '20200330', '100000000000', tier='T3_separate_total', scope='OFS'),
                self.filing('R2', '2020-12', '20210330', '110000000000', tier='T3_separate_total', scope='OFS'),
                self.filing('R3', '2021-12', '20220330', '120000000000000000', tier='T3_separate_total', scope='OFS', unit='True')]
        filings = pd.DataFrame(rows).astype('string')
        filings['override_id'] = pd.Series([pd.NA, pd.NA, override], dtype='string')
        facts = pd.DataFrame([{'receipt_id': 'R3', 'concept': 'total', 'statement_type': 'OFS', 'section': section,
                               'source_kind': 'document', 'identity_ok': 'True', 'equity': attached, 'fiscal_period': '2021-12-31',
                               'member': 'R3_00760.xml', 'location': 'TABLE[1]/ROW[9]', 'column_basis': 'header_date|title|원',
                               'account_name': '자본총계', 'raw_path': 'p/R3', 'sha256': 'hR3'}]).astype('string')
        from src.preprocessing.dart_extract import apply_unit_corrections
        return apply_unit_corrections(filings, facts)

    def test_unit_slip_is_corrected_only_from_the_same_filings_full_statement(self):
        out, applied = self.unit_case('120000000000')
        row = out.set_index('receipt_id').loc['R3']
        self.assertEqual((row.equity, row.unit_suspect, row.source_kind), ('120000000000', 'False', 'document'))
        self.assertEqual(row.override_id, 'AUTO_UNIT:x10^-6:R3_00760.xml:TABLE[1]/ROW[9]')
        self.assertEqual(len(applied), 1)
        self.assertEqual(out.set_index('receipt_id').loc[['R1', 'R2']].equity.tolist(), ['100000000000', '110000000000'])

    def test_unit_slip_stays_held_without_exact_same_filing_evidence(self):
        for attached, section, override in (('120000000001', 'full', None),  # not exactly 10^6 apart
                                            ('120000000000', 'summary', None),  # a summary table is not enough
                                            ('120000000000', 'full', 'C-01')):  # a manual decision already covers it
            out, applied = self.unit_case(attached, section=section, override=override)
            self.assertEqual(applied, [])
            self.assertEqual(out.set_index('receipt_id').loc['R3'].unit_suspect, 'True')

    def test_p4_last_trading_day_and_after_count_from_next_month(self):
        from src.preprocessing.dart_extract import available_month
        self.assertEqual(available_month('20200330', self.LAST), '2020-03')
        self.assertEqual(available_month('20200331', self.LAST), '2020-04')
        self.assertEqual(available_month('20200429', self.LAST), '2020-05')  # after the 28th, a non-trading day here
        self.assertEqual(available_month('20201228', self.LAST), '2021-01')
        self.assertEqual(available_month('20230105', self.LAST), '2023-01')  # past the calendar: never in a served month
        with self.assertRaises(ValueError):
            available_month('20180105', self.LAST)
        table = self.join([self.filing('1', '2019-12', '20200331', '100')], ['2020-03', '2020-04'])
        self.assertEqual(table.loc['2020-03', 'asof_status'], 'before_first_regular_report')
        self.assertEqual(table.loc['2020-04', 'equity'], 100)
        nothing = self.join([self.filing('1', '2019-12', '20200331', '100', corp='C9')], ['2020-04'])
        self.assertEqual(nothing.loc['2020-04', 'asof_status'], 'no_regular_report_in_window')

    def test_p1_keeps_the_earlier_version_and_records_the_skipped_correction(self):
        table = self.join([self.filing('1', '2019-12', '20200310', '100'),
                           self.filing('2', '2019-12', '20200415', None, revision='기재정정', reason='no_verified_original')],
                          ['2020-03', '2020-04'])
        self.assertEqual((table.loc['2020-03', 'equity'], table.loc['2020-03', 'fallback']), (100, 'none'))
        row = table.loc['2020-04']
        self.assertEqual((row.equity, row.receipt_id, row.fallback), (100, '1', 'older_version_same_period'))
        self.assertEqual((row.skipped_receipt_ids, row.skipped_reasons), ('2', 'no_value:no_verified_original'))

    def test_newest_period_t3_beats_older_t1_and_p2_skips_an_undated_statement(self):
        table = self.join([self.filing('1', '2019-12', '20200310', '100'),
                           self.filing('3', '2020-03', '20200515', '50', tier='T3_separate_total', scope='OFS'),
                           self.filing('4', '2020-06', '20200814', '60', fiscal=None)],
                          ['2020-03', '2020-05', '2020-08'])
        self.assertEqual((table.loc['2020-05', 'equity_tier'], table.loc['2020-05', 'scope_changed']), ('T3_separate_total', True))
        row = table.loc['2020-08']
        self.assertEqual((row.receipt_id, row.fallback, row.skipped_reasons), ('3', 'older_period', 'fiscal_date_unverified'))

    def test_p10_lineage_conflict_and_equal_values(self):
        import pandas as pd
        base = self.filing('1', '2019-12', '20200310', '100')
        corrected = self.filing('2', '2019-12', '20200310', '90', revision='기재정정')
        self.assertEqual(self.join([base, corrected], ['2020-03']).loc['2020-03', 'equity'], 90)
        other = self.filing('3', '2019-12', '20200310', '80', revision='기재정정')
        held = self.join([base, corrected, other], ['2020-03']).loc['2020-03']
        self.assertEqual((held.asof_status, held.usable, held.same_day_order_unverified), ('same_day_value_conflict', False, True))
        self.assertTrue(pd.isna(held.equity))
        same = self.join([base, corrected, {**other, 'equity': '90'}], ['2020-03']).loc['2020-03']
        self.assertEqual((same.asof_status, same.equity, same.same_day_order_unverified), ('selected', 90, True))

    def test_p3_expires_after_the_month_end_eighteen_months_later(self):
        table = self.join([self.filing('1', '2019-09', '20191114', '10')], ['2021-03', '2021-04'])
        self.assertEqual((table.loc['2021-03', 'expiry_date'], table.loc['2021-03', 'usable']), ('20210331', True))
        row = table.loc['2021-04']
        self.assertEqual((row.usable, row.exclusion_reason, row.equity), (False, 'expired_18m', 10))

    def test_unmapped_nonpositive_history_and_unit_suspect(self):
        filings = [self.filing('1', '2019-12', '20200310', '-5'),
                   self.filing('2', '2020-03', '20200515', '7000', unit='True')]
        table = self.join(filings, ['2020-03', '2020-05'], stocks=(('A', 'C1', 'current_exact'), ('B', None, 'not_in_current_registry')),
                          complete_from='2020-04')
        a = table[table.stock_id == 'A']
        self.assertEqual(a.loc['2020-03', 'exclusion_reason'], 'history_incomplete;nonpositive_equity')
        self.assertEqual((a.loc['2020-05', 'equity'], a.loc['2020-05', 'skipped_reasons']), (-5, 'unit_suspect'))
        self.assertEqual(set(table[table.stock_id == 'B'].asof_status), {'unmapped'})


class DartCollectionTests(unittest.TestCase):
    """Source/PIT risks only; the H1 calculation stub above remains untouched."""

    def test_blank_subcolumn_preserved_bank_and_insurer(self):
        import json
        from pathlib import Path
        from src.preprocessing.dart_extract import _document_evidence
        root = Path(__file__).resolve().parents[1]
        for receipt, corp, cfs, ofs in [
            ('20160330004014', '00149293', '20839353000000', '20468080000000'),
            ('20160330003649', '00139214', '10085250713687', '9970587283574'),
        ]:
            originals = list((root / 'data/raw/opendart/filing_originals').glob(f'document_{receipt}_*_ZIP_BODY_*.zip'))
            if not originals:
                self.skipTest('real DART-003 sample deleted in the 2026-11-04 cleanup')
            doc = _document_evidence(originals[0].read_bytes(), corp, '2015-12')
            totals = [f for f in doc['facts'] if ''.join(f['account_name'].split()) == '자본총계']
            self.assertEqual({f['equity'] for f in totals if f['statement_type'] == 'CFS'}, {cfs})
            self.assertEqual({f['equity'] for f in totals if f['statement_type'] == 'OFS'}, {ofs})
            self.assertTrue(all(f['account'] is None and f['unit'] == 'KRW' for f in totals))
            self.assertTrue(all(f['fiscal_period'] == '2015-12-31' for f in totals))

    def test_dated_header_block_never_uses_previous_year_for_blank(self):
        import io
        import zipfile
        from src.preprocessing.dart_extract import _document_evidence
        for current, expected in [('', None), ('0', '0'), ('(1,234)', '-1234000'), ('△12', '-12000')]:
            text = '''<DOCUMENT><COMPANY-NAME AREGCIK="00111704">test</COMPANY-NAME>
            <P>연결재무상태표 제2기 기말 2021년 12월 31일 현재 제1기 기말 2020년 12월 31일 현재 (단위: 천원)</P>
            <TABLE><THEAD><TR><TH>과목</TH><TH COLSPAN="2">제2기 기말</TH><TH>제1기 기말</TH></TR></THEAD>
            <TBODY><TR><TD>자본총계</TD><TD></TD><TD>VALUE</TD><TD>999999</TD></TR></TBODY></TABLE></DOCUMENT>'''.replace('VALUE', current)
            out = io.BytesIO()
            with zipfile.ZipFile(out, 'w') as z:
                z.writestr('report.xml', text.encode('utf-8'))
            facts = _document_evidence(out.getvalue(), '00111704', '2021-12')['facts']
            self.assertEqual([f['equity'] for f in facts], [] if expected is None else [expected])
            self.assertTrue(all(f['statement_type'] == 'CFS' and f['account'] is None for f in facts))

    def test_request_plan_keeps_all_ten_successes_across_year_end_changes(self):
        import json
        from pathlib import Path
        from src.preprocessing.dart_extract import plan_financial_requests, _request_key
        root = Path(__file__).resolve().parents[1]
        cases = [('00144818', '2017', '11012'), ('00144818', '2017', '11013'),
                 ('00145686', '2016', '11012'), ('00159573', '2020', '11012'),
                 ('00159573', '2020', '11013')]
        checked = 0
        for corp, year, report in cases:
            # Ambiguous fiscal calendars retain every request; successful requests
            # must survive even when report labels do not match calendar quarters.
            index = [{'corp_code': corp, 'rcept_no': '20170901000001',
                      'report_nm': '반기보고서 (2017.06)', 'rcept_dt': '20170901'}]
            plan, _ = plan_financial_requests(corp, index, index_complete=True)
            planned = {_request_key('fnlttSinglAcntAll.json', p) for p in plan.to_dict('records')}
            for fs in ['CFS', 'OFS']:
                params = {'corp_code': corp, 'bsns_year': year, 'reprt_code': report, 'fs_div': fs}
                key = _request_key('fnlttSinglAcntAll.json', params)
                self.assertIn(key, planned)
                checked += 1
        self.assertEqual(checked, 10)

    def test_empty_incomplete_index_never_gates_and_absence_is_scoped(self):
        from src.preprocessing.dart_extract import plan_financial_requests
        plan, report = plan_financial_requests('00111704', [], index_complete=False)
        self.assertEqual(len(plan), 56)
        plan, report = plan_financial_requests('00111704', [], index_complete=True)
        self.assertEqual(len(plan), 0)
        self.assertEqual(report['absence_scope'], 'index_query_only')

    def test_preserved_request_cache_and_offline_miss_never_call_network(self):
        import json
        import threading
        from pathlib import Path
        from unittest.mock import patch
        from src.preprocessing.dart_extract import DartCollector, CollectionStop
        import tempfile
        from src.preprocessing.dart_extract import _sha, _request_key
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        body = b'{"status":"000","list":[]}'
        (root / 'response.json').write_bytes(body)
        params = {'corp_code': '00111704', 'bsns_year': '2021', 'reprt_code': '11011', 'fs_div': 'CFS'}
        raw = {'endpoint': 'fnlttSinglAcntAll.json', 'params': params, 'path': 'response.json',
               'key': _request_key('fnlttSinglAcntAll.json', params), 'sha256': _sha(body)}
        collector = DartCollector.__new__(DartCollector)
        collector.root, collector.cache = root, {raw['key']: raw}
        collector.lock, collector.cache_hits, collector.offline = threading.Lock(), 0, True
        with patch('src.preprocessing.dart_extract.requests.get') as network:
            for _ in range(2):
                record, body, data = collector.get(raw['endpoint'], raw['params'])
                self.assertEqual(record['sha256'], raw['sha256'])
                self.assertEqual(data['status'], '000')
            with self.assertRaises(CollectionStop):
                collector.get('document.xml', {'rcept_no': 'never_requested'})
            network.assert_not_called()
        self.assertEqual(collector.cache_hits, 2)

    def test_absence_source_failure_and_execution_error_are_separate(self):
        import pandas as pd
        from src.preprocessing.dart_extract import DartCollector, VERSION_COLUMNS, FACT_COLUMNS
        collector = DartCollector.__new__(DartCollector)
        collector.progress = {'completed_corps': ['00000001', '00000002', '00000003']}
        collector.versions = pd.DataFrame([{'corp_code': '00000002', 'receipt_id': 'r2',
            'revision': 'initial', 'published_at': '20210301'}]).reindex(columns=VERSION_COLUMNS).astype('string')
        collector.facts = pd.DataFrame(columns=FACT_COLUMNS).astype('string')
        collector.raw_records = [
            {'endpoint': 'list.json', 'params': {'corp_code': '00000001', 'bgn_de': '20150101'}, 'status': '013'},
            {'endpoint': 'fnlttSinglAcntAll.json', 'params': {'corp_code': '00000002'}, 'status': '013'},
            {'endpoint': 'fnlttXbrl.xml', 'params': {'rcept_no': 'r2'}, 'status': '014'},
            {'endpoint': 'list.json', 'params': {'corp_code': '00000003'}, 'status': '020'},
        ]
        states = collector.corporation_states()
        self.assertEqual(states['00000001']['procedure'], 'complete')
        self.assertEqual(states['00000001']['acquisition'], 'no_index_reports_in_query_scope')
        self.assertEqual(states['00000002']['procedure'], 'complete')
        self.assertEqual(states['00000002']['acquisition'], 'reports_exist_amount_extraction_unresolved')
        self.assertEqual(states['00000002']['missing_source_014'], 1)
        self.assertEqual(states['00000003']['procedure'], 'error_recorded')
        self.assertTrue(all(s['pit'].startswith('not_verified') for s in states.values()))

    def test_bounded_batch_reports_global_remaining_separately(self):
        import contextlib
        import io
        import pandas as pd
        from pathlib import Path
        from src.preprocessing.dart_extract import DartCollector, VERSION_COLUMNS, FACT_COLUMNS
        c = DartCollector.__new__(DartCollector)
        c.mapping = pd.DataFrame({'corp_code': ['00000001', '00000002', '00000003'],
            'stock_id': ['111111', '222222', '333333'], 'mapping_status': ['current_exact'] * 3})
        c.progress = {'completed_corps': ['00000001'], 'mapping': {'unique_corps': 3}, 'batches': []}
        c.versions = pd.DataFrame(columns=VERSION_COLUMNS)
        c.facts = pd.DataFrame(columns=FACT_COLUMNS)
        c.progress_path = Path(__file__).parent / '_nonexistent_no_writes' / 'progress.json'
        c.network_calls, c.cache_hits = 0, 0
        c.checkpoint = lambda: None
        c.log = lambda record: None
        c.validate = lambda: {}
        c.collect_corp = lambda corp: ([], [], {})
        with contextlib.redirect_stdout(io.StringIO()):
            result = c.run_collection(batch_size=1, max_new_corps=1, audit_fixed=False)
        self.assertEqual(result['completed_corps'], 2)
        self.assertEqual(result['batch_remaining_corps'], 0)
        self.assertEqual(result['remaining_corps'], 1)
        self.assertEqual(result['status'], 'pilot_complete')

    def test_fixed_collection_scope(self):
        import pandas as pd
        from src.preprocessing.dart_extract import plan_dart_collection
        plan = plan_dart_collection(pd.DataFrame({'corp_code': ['00111704', '00111704']}), config={})
        self.assertEqual(len(plan), 56)
        self.assertEqual(set(plan.bsns_year), {str(y) for y in range(2015, 2022)})
        self.assertEqual(set(plan.fs_div), {'CFS', 'OFS'})
        with self.assertRaises(ValueError):
            plan_dart_collection(pd.DataFrame({'corp_code': [111704]}), config={})

    def test_month_end_and_numeric_semantics(self):
        from src.preprocessing.dart_extract import _month_available, _amount
        self.assertEqual(_month_available('20210331'), '2021-04')
        self.assertEqual(_month_available('20210330'), '2021-03')
        self.assertEqual(_month_available('20211231'), '2022-01')
        self.assertIsNone(_month_available(None))
        self.assertEqual(_amount('0'), '0')
        self.assertIsNone(_amount(''))
        self.assertIsNone(_amount('not numeric'))
        self.assertEqual(_amount('(1,234)'), '-1234')
        self.assertEqual(_amount('9007199254740993'), '9007199254740993')

    def test_no_calendar_quarter_guess(self):
        from src.preprocessing.dart_extract import _report_group
        self.assertEqual(_report_group('[기재정정]분기보고서 (2020.06)'),
                         ('분기', '2020-06', '분기:2020-06'))
        self.assertIsNone(_report_group('분기보고서 (2020.13)'))

    def test_same_day_versions_do_not_invent_a_predecessor(self):
        import pandas as pd
        from src.preprocessing.dart_extract import _verified_predecessors
        versions = pd.DataFrame([
            {'receipt_id': 'initial', 'published_at': '20160330'},
            {'receipt_id': 'correction_a', 'published_at': '20160516'},
            {'receipt_id': 'correction_b', 'published_at': '20160516'},
            {'receipt_id': 'later', 'published_at': '20161019'},
        ]).assign(initial_receipt='initial', filing_group='same_period')
        links = _verified_predecessors(versions)
        self.assertEqual(links[0], (None, None))
        self.assertEqual(links[1], (None, 'same_day_version_order_unverified'))
        self.assertEqual(links[2], (None, 'same_day_version_order_unverified'))
        self.assertEqual(links[3], (None, 'preceding_day_order_unverified'))

    def test_xbrl_dimension_entity_and_unit_guards(self):
        import io
        import zipfile
        from src.preprocessing.dart_extract import _xbrl_facts
        template = '''<x:xbrl xmlns:x="http://www.xbrl.org/2003/instance"
          xmlns:d="http://xbrl.org/2006/xbrldi" xmlns:i="http://example.org/ifrs-full">
          <x:context id="c"><x:entity><x:identifier scheme="DART">{corp}</x:identifier></x:entity>
          <x:period><x:instant>2021-09-30</x:instant></x:period><x:scenario>
          <d:explicitMember dimension="i:ConsolidatedAndSeparateFinancialStatementsAxis">i:ConsolidatedMember</d:explicitMember>
          {extra}</x:scenario></x:context>
          <x:unit id="u"><x:measure>{unit}</x:measure></x:unit>
          <i:EquityAttributableToOwnersOfParent contextRef="c" unitRef="u" decimals="0">9007199254740993</i:EquityAttributableToOwnersOfParent>
          </x:xbrl>'''
        def parse(corp='00111704', extra='', unit='iso4217:KRW'):
            out = io.BytesIO()
            with zipfile.ZipFile(out, 'w') as z:
                z.writestr('report.xbrl', template.format(corp=corp, extra=extra, unit=unit))
            return _xbrl_facts(out.getvalue(), '00111704', '2021-09')[0]
        facts = parse()
        self.assertEqual(facts[0]['equity'], '9007199254740993')
        self.assertEqual(facts[0]['statement_type'], 'CFS')
        self.assertEqual(parse(corp='00999999'), [])
        self.assertEqual(parse(unit='x:shares'), [])
        self.assertEqual(parse(extra='<d:explicitMember dimension="i:ComponentsOfEquityAxis">i:IssuedCapitalMember</d:explicitMember>'), [])

    def test_preserved_dart002_versions(self):
        from pathlib import Path
        from src.preprocessing.dart_extract import _xbrl_facts, _document_evidence
        folder = Path(__file__).resolve().parents[1] / 'data/raw/opendart/filing_originals'
        for receipt, amount in [('20160330004251', '746668673663'), ('20170511004419', '929062351201')]:
            path = next(folder.glob('OpenDART_DART002_fnlttXbrl_*_' + receipt + '_*'), None)
            if path is None:
                self.skipTest('preserved DART-002 source sample not available')
            facts, _ = _xbrl_facts(path.read_bytes(), '00111704', '2015-12')
            selected = [f for f in facts if f['statement_type'] == 'CFS'
                        and f['account'].endswith('EquityAttributableToOwnersOfParent')]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]['equity'], amount)
        path = next(folder.glob('OpenDART_DART002_document_*_20170511004419_*'), None)
        if path is None:
            self.skipTest('preserved DART-002 document sample not available')
        evidence = _document_evidence(path.read_bytes(), '00111704', '2015-12')
        self.assertEqual(evidence['initial_date'], '20160330')
        self.assertTrue(any(f['equity'] == '929062351201' for f in evidence['facts']))
        with self.assertRaises(ValueError):
            _document_evidence(path.read_bytes(), '00999999', '2015-12')

    def test_quarter_from_actual_cover_not_header_acode_or_month(self):
        from pathlib import Path
        from src.preprocessing.dart_extract import _document_evidence
        folder = Path(__file__).resolve().parents[1] / 'data/raw/opendart/filing_originals'
        # This actual Q3 document's generic DOCUMENT-NAME ACODE is 11013.
        path = next(folder.glob('document_20151116001514_*'), None)
        if path is None:
            self.skipTest('DART-003 pilot quarter document not available')
        evidence = _document_evidence(path.read_bytes(), '00111704', '2015-09')
        self.assertEqual(evidence['reprt_code_from_cover'], '11014')
        # A June quarter for a March year-end is explicitly Q1 on its own cover.
        path = next(folder.glob('document_20150813000883_*'), None)
        if path is None:
            self.skipTest('DART-003 non-December pilot document not available')
        evidence = _document_evidence(path.read_bytes(), '00136721', '2015-06')
        self.assertEqual(evidence['reprt_code_from_cover'], '11013')
        # A term such as "62기 1분기" contains a digit before the fiscal date.
        self.assertTrue(any(f['statement_type'] == 'CFS' and f['equity'] == '1053797469385'
                            and f['fiscal_period'] == '2015-06-30' for f in evidence['facts']))
        self.assertTrue(any(f['statement_type'] == 'OFS' and f['equity'] == '993009484033'
                            for f in evidence['facts']))


class DartEquityExtractTests(unittest.TestCase):
    """Synthetic statements only; each rule below comes from a real failure in stored originals."""

    STATEMENT = ('<P>연 결 재 무 상 태 표 제 54 기 2015.09.30 현재 제 53 기 2014.09.30 현재 (단위 : 백만원)</P>'
                 '<TABLE><THEAD><TR><TH>과 목</TH><TH>주 석</TH><TH COLSPAN="2">제 54 (당) 기</TH>'
                 '<TH COLSPAN="2">제 53 (전) 기</TH></TR></THEAD><TBODY>{rows}</TBODY></TABLE>')

    @staticmethod
    def row(label, current, prior, cell='TD'):
        return (f'<TR><TD>{label}</TD><TD></TD><TD></TD><{cell}>{current}</{cell}>'
                f'<TD></TD><{cell}>{prior}</{cell}></TR>')

    def body(self, equity='60', **kwargs):
        rows = [self.row('자산총계', '100', '90', **kwargs), self.row('부채총계', '40', '40', **kwargs),
                self.row('지배기업 소유주지분', '55', '45', **kwargs), self.row('비지배지분', '5', '5', **kwargs),
                self.row('자 본 총 계', equity, '50', **kwargs), self.row('부채와자본총계', '100', '90', **kwargs)]
        return self.STATEMENT.format(rows=''.join(rows))

    @staticmethod
    def fact(concept, scope, amount, section='full', kind='document', identity=True, basis='header_term_dated|title|원'):
        return {'concept': concept, 'statement_type': scope, 'equity': amount, 'unit': 'KRW', 'source_kind': kind,
                'section': section, 'identity_ok': identity, 'column_basis': basis, 'account_name': concept,
                'fiscal_period': '2015-09-30'}

    def test_label_concepts_never_substitute_or_admit_income_lines(self):
        from src.preprocessing.dart_extract import concept_of
        cases = {'지배기업 소유주지분': 'owners', '[지배기업의 소유주에게 귀속되는 자본]': 'owners',
                 '지배회사지분': 'owners', '비지배지분': 'nci', '외부주주지분': 'nci', '소수주주지분': 'nci',
                 '자 본 총 계': 'total', "자 본 총 계 (Total Stockholders' Equity)": 'total', 'Ⅲ. 자본총계': 'total',
                 '자산총계': 'assets', '부채총계': 'liabilities',
                 '부채와자본총계': None, '부채및자본총계': None, '지배회사지분순이익': None,
                 '지배기업소유주지분순이익(손실)': None, '자본금': None, '이연법인세자산': None,
                 '지배지분': 'owners', '소유주분배예정자산집단': None, 'Ⅱ. 소유주분배예정자산': None,
                 '소유주에 대한 분배예정으로 분류된 비유동자산이나 처분자산집단': None, '소유주분배예정부채': None,
                 '비지배지분부채': None,
                 '매각예정자산처분집단관련 자본': 'held_for_sale_equity', '매각예정분류 자본': 'held_for_sale_equity',
                 '매각예정자산ㆍ부채 관련 기타포괄손익누계액': 'held_for_sale_equity', '매각예정비유동자산': None}
        for label, expected in cases.items():
            self.assertEqual(concept_of(label), expected, label)

    def test_asset_row_named_for_owners_does_not_hide_the_real_owners_row(self):
        from src.preprocessing.dart_extract import document_facts, select_filing
        rows = self.body().split('</TR>')
        spot = next(i for i, row in enumerate(rows) if '자산총계' in row.replace(' ', ''))
        rows[spot - 1] += '</TR>' + '<TR><TD>소유주분배예정자산집단</TD><TD></TD><TD>0</TD><TD>0'
        facts, _ = document_facts('</TR>'.join(rows), '2015-09')
        self.assertEqual(select_filing(facts)['equity'], '55000000')

    @staticmethod
    def table(number, member='a.xml'):
        return {'member': member, 'location': f'TABLE[{number}]/ROW[1]'}

    def test_total_less_nci_sits_between_owners_and_total_and_needs_one_statement(self):
        from src.preprocessing.dart_extract import select_filing
        total, nci = {**self.fact('total', 'CFS', '100'), **self.table(1)}, {**self.fact('nci', 'CFS', '30'), **self.table(1)}
        chosen = select_filing([total, nci])
        self.assertEqual((chosen['equity'], chosen['equity_tier']), ('70', 'T1D_consolidated_total_less_nci'))
        written = {**self.fact('owners', 'CFS', '69'), **self.table(1)}
        self.assertEqual(select_filing([total, nci, written])['equity_tier'], 'T1_consolidated_owners')
        split = select_filing([total, {**nci, **self.table(2)}])
        self.assertEqual(split['equity_tier'], 'T2_consolidated_total')
        unbalanced = select_filing([{**total, 'identity_ok': None}, nci])
        self.assertEqual(unbalanced['equity_tier'], 'T2_consolidated_total')
        summary = select_filing([{**total, 'section': 'summary'}, {**nci, 'section': 'summary'}])
        self.assertEqual(summary['equity_tier'], 'T2_consolidated_total')

    def test_xbrl_owners_far_from_the_statement_lose_to_total_less_nci(self):
        from src.preprocessing.dart_extract import select_filing
        total, nci = {**self.fact('total', 'CFS', '1000'), **self.table(1)}, {**self.fact('nci', 'CFS', '100'), **self.table(1)}
        bad = self.fact('owners', 'CFS', '30', kind='xbrl', section='xbrl', identity=None, basis='instant_context')
        chosen = select_filing([bad, total, nci])
        self.assertEqual((chosen['equity'], chosen['equity_tier'], chosen['xbrl_document_agree']),
                         ('900', 'T1D_consolidated_total_less_nci', False))
        close = {**bad, 'equity': '880'}
        self.assertEqual(select_filing([close, total, nci])['equity'], '880')

    def test_unit_suspects_use_only_earlier_filings_and_catch_a_lasting_slip(self):
        import pandas as pd
        from src.preprocessing.dart_extract import unit_suspects
        filing = lambda n, equity, scope='CFS', corp='a': {'corp_code': corp, 'statement_type': scope, 'equity': equity,
                                                           'published_at': f'2020{n:04d}', 'receipt_id': str(n)}
        frame = pd.DataFrame([filing(1, '100'), filing(2, '100000100'), filing(3, '110'),  # one spike, then back
                              filing(4, '90000'), filing(5, '88000'), filing(6, '120'),  # lasting 1000x run
                              filing(7, '5000'), filing(8, None), filing(9, '-4'),  # ordinary move, blank, negative
                              filing(10, '100000', 'OFS'), filing(11, '120', corp='b')]).astype('string')
        self.assertEqual(unit_suspects(frame).tolist(),
                         [False, True, False, True, True, False, False, False, False, False, False])
        first_is_the_slip = pd.DataFrame([filing(n, '5000000000' if n > 1 else '5000') for n in range(1, 7)]).astype('string')
        self.assertEqual(unit_suspects(first_is_the_slip).tolist(), [False, True, True, False, False, False])

    def test_current_column_scale_and_identity_with_te_cells_and_note_column(self):
        from src.preprocessing.dart_extract import document_facts
        for cell in ('TD', 'TE'):
            facts, _ = document_facts(self.body(cell=cell), '2015-09')
            found = {f['concept']: f for f in facts}
            self.assertEqual(found['owners']['equity'], '55000000')
            self.assertEqual(found['total']['equity'], '60000000')
            self.assertEqual(found['total']['statement_type'], 'CFS')
            self.assertEqual(found['total']['fiscal_period'], '2015-09-30')
            self.assertIs(found['total']['identity_ok'], True)

    def test_blank_current_amount_never_falls_back_to_prior_year(self):
        from src.preprocessing.dart_extract import document_facts
        facts, audit = document_facts(self.body(equity=''), '2015-09')
        self.assertNotIn('total', {f['concept'] for f in facts})
        self.assertEqual(audit['blank_current_amount'], 1)

    def test_other_period_statement_is_not_read_as_current(self):
        from src.preprocessing.dart_extract import document_facts
        facts, audit = document_facts(self.body(), '2016-09')
        self.assertEqual(facts, [])
        self.assertEqual(audit['statement_date_is_other_period'], 1)

    def test_blank_or_foreign_unit_is_not_guessed(self):
        from src.preprocessing.dart_extract import document_facts
        for unit, reason in (('(단위 : )', 'unit_unresolved'), ('(단위 : USD)', 'unit_not_krw')):
            facts, audit = document_facts(self.body().replace('(단위 : 백만원)', unit), '2015-09')
            self.assertEqual(facts, [])
            self.assertEqual(audit[reason], 1)

    def test_undated_summary_and_subsidiary_statement_rejected(self):
        from src.preprocessing.dart_extract import document_facts
        undated = ('<P>요약 연결재무정보 (단위 : 백만원)</P><TABLE><TR><TD>구 분</TD><TD>제54기</TD><TD>제53기</TD></TR>'
                   '<TR><TD>자산총계</TD><TD>100</TD><TD>90</TD></TR><TR><TD>부채총계</TD><TD>40</TD><TD>40</TD></TR>'
                   '<TR><TD>자본총계</TD><TD>60</TD><TD>50</TD></TR></TABLE>')
        facts, audit = document_facts(undated, '2015-09')
        self.assertEqual(facts, [])
        self.assertEqual(audit['summary_column_not_dated'], 1)
        facts, _ = document_facts(self.body().replace('연 결 재 무 상 태 표', '주요 종속기업의 재무상태표'), '2015-09')
        self.assertEqual(facts, [])

    SUMMARY = ('<P>요약 연결재무정보 (단위 : 백만원)</P><TABLE><TR><TD>구 분</TD><TD>제54기</TD><TD>제53기</TD></TR>'
               '<TR><TD>자산총계</TD><TD>100</TD><TD>90</TD></TR><TR><TD>부채총계</TD><TD>40</TD><TD>40</TD></TR>'
               '<TR><TD>자본총계</TD><TD>60</TD><TD>50</TD></TR></TABLE>')

    def test_cover_term_summary_counts_up_to_2014_dated_by_the_cover_period(self):
        from src.preprocessing.dart_extract import document_facts
        cover = '<P>사업보고서 제 54 기 (2003.01.01 ~ 2003.12.31)</P>'
        facts, _ = document_facts(cover + self.SUMMARY, '2003-12')
        total = next(f for f in facts if f['concept'] == 'total')
        self.assertEqual((total['equity'], total['section'], total['fiscal_period'], total['fiscal_period_basis']),
                         ('60000000', 'summary', '2003-12-31', 'cover_term_period'))
        self.assertTrue(total['column_basis'].startswith('header_term_cover|'))
        undated, _ = document_facts('<P>사업보고서 제 54 기</P>' + self.SUMMARY, '2003-12')
        self.assertEqual({f['fiscal_period'] for f in undated}, {None})  # no cover period: value kept, date unconfirmed
        other, _ = document_facts('<P>제 54 기 (2003.01.01 ~ 2003.06.30)</P>' + self.SUMMARY, '2003-12')
        self.assertEqual({f['fiscal_period'] for f in other}, {None})  # cover end in another month is not this statement's date
        later, audit = document_facts(cover.replace('2003', '2015') + self.SUMMARY, '2015-12')
        self.assertEqual((later, audit['summary_column_not_dated']), ([], 1))  # R3 stops at 2014 labels

    def test_cover_term_summary_is_used_only_when_nothing_else_gives_a_value(self):
        from src.preprocessing.dart_extract import select_filing
        summary = {**self.fact('total', 'CFS', '200', section='summary', basis='header_term_cover|title|원'), 'fiscal_period': '2003-12-31'}
        full = {**self.fact('total', 'OFS', '100'), 'fiscal_period': '2003-12-31'}
        self.assertEqual((select_filing([summary, full])['equity'], select_filing([summary, full])['equity_tier']), ('100', 'T3_separate_total'))
        self.assertEqual(select_filing([summary])['equity_tier'], 'T2_consolidated_total')

    def test_income_rows_after_summary_balance_do_not_become_owners(self):
        from src.preprocessing.dart_extract import document_facts
        text = ('<P>요약연결재무상태표 제 54 기 2015.09.30 현재 (단위 : 원)</P><TABLE><TR><TD>구 분</TD><TD>2015.09.30</TD></TR>'
                '<TR><TD>자산총계</TD><TD>100</TD></TR><TR><TD>부채총계</TD><TD>40</TD></TR><TR><TD>자본총계</TD><TD>60</TD></TR>'
                '<TR><TD>당기순이익</TD><TD>7</TD></TR><TR><TD>지배기업소유주지분</TD><TD>6</TD></TR></TABLE>')
        facts, _ = document_facts(text, '2015-09')
        self.assertEqual({f['concept'] for f in facts}, {'total'})

    def test_multiline_row_needs_the_balance_identity(self):
        from src.preprocessing.dart_extract import document_facts
        accounts = ['유동자산', '당좌자산', '재고자산', '고정자산', '자산총계', '유동부채', '고정부채', '부채총계', '자본금', '자본총계']
        amounts = ['50', '30', '20', '50', '100', '20', '20', '40', '10', '{equity}']
        text = ('<P>대차대조표 제 3 기 2001.03.31 현재 (단위 : 원)</P><TABLE><TR><TH>과 목</TH><TH>제 3 기</TH></TR>'
                '<TR><TD>' + '<BR>'.join(accounts) + '</TD><TD>' + '<BR>'.join(amounts) + '</TD></TR></TABLE>')
        facts, _ = document_facts(text.format(equity='60'), '2001-03')
        self.assertEqual([f['equity'] for f in facts], ['60'])
        facts, audit = document_facts(text.format(equity='70'), '2001-03')
        self.assertEqual(facts, [])
        self.assertEqual(audit['multiline_identity_unconfirmed'], 1)

    def test_tier_order_is_owners_then_consolidated_then_separate_total(self):
        from src.preprocessing.dart_extract import select_filing
        owners, total = self.fact('owners', 'CFS', '5'), self.fact('total', 'CFS', '6')
        separate = self.fact('total', 'OFS', '7')
        self.assertEqual(select_filing([separate, total, owners])['equity_tier'], 'T1_consolidated_owners')
        self.assertEqual(select_filing([separate, total])['equity_tier'], 'T2_consolidated_total')
        self.assertEqual(select_filing([separate])['equity_tier'], 'T3_separate_total')
        self.assertEqual(select_filing([])['reason'], 'target_concepts_not_observed')

    def test_failed_identity_excluded_repeat_wins_tie_stays_unresolved(self):
        from src.preprocessing.dart_extract import select_filing
        bad = self.fact('total', 'OFS', '9', identity=False)
        self.assertIsNone(select_filing([bad])['equity'])
        repeated = [self.fact('total', 'OFS', '7'), self.fact('total', 'OFS', '7'), self.fact('total', 'OFS', '1')]
        self.assertEqual(select_filing(repeated)['equity'], '7')
        tie = select_filing([self.fact('total', 'OFS', '7'), self.fact('total', 'OFS', '1')])
        self.assertIsNone(tie['equity'])
        self.assertTrue(tie['reason'].startswith('same_rank_amounts_conflict'))

    def test_balanced_statement_overrides_mis_scaled_xbrl_and_is_flagged(self):
        from src.preprocessing.dart_extract import select_filing
        xbrl = self.fact('total', 'OFS', '160000000', kind='xbrl', section='xbrl', identity=None, basis='instant_context')
        chosen = select_filing([xbrl, self.fact('total', 'OFS', '160')])
        self.assertEqual((chosen['equity'], chosen['source_kind'], chosen['xbrl_document_agree']), ('160', 'document', False))
        agreed = select_filing([{**xbrl, 'equity': '160'}, self.fact('total', 'OFS', '160')])
        self.assertEqual((agreed['source_kind'], agreed['xbrl_document_agree']), ('xbrl', True))

    def test_statement_date_from_the_current_column_header_only(self):
        from src.preprocessing.dart_extract import resolve_current_columns
        for head, end, basis in (('제54기 3분기 2015년 9월말', '2015-09-30', 'header_month_end_word'),
                                 ('제54기 3분기 (2015.09.30 현재)', '2015-09-30', 'header_full_date'),
                                 ('2015.09', None, None)):
            columns, column_basis, found, found_basis = resolve_current_columns(['구분', head, '제53기 2014년 12월말'], '', '2015-09', None)
            self.assertEqual((columns, column_basis, found, found_basis), ([1], 'header_date', end, basis), head)
        dated = resolve_current_columns(['구분', '제54기 3분기 2015년 9월말'], '2015.09.30 현재', '2015-09', None)
        self.assertEqual(dated[2:], ('2015-09-30', 'context_date'))

    def test_tables_in_the_shareholder_section_are_not_the_filers_statement(self):
        from src.preprocessing.dart_extract import document_facts
        facts, audit = document_facts('<TITLE ATOC="Y">VII. 주주에 관한 사항</TITLE>' + self.body(), '2015-09')
        self.assertEqual((facts, audit['other_entity_section']), ([], 1))
        facts, _ = document_facts('<TITLE ATOC="Y">III. 재무에 관한 사항</TITLE>' + self.body(), '2015-09')
        self.assertTrue(facts)

    def test_2000_era_statement_named_only_by_its_section(self):
        from src.preprocessing.dart_extract import document_facts
        body = self.body().replace('연 결 재 무 상 태 표', '')
        # The section's own title is far above: an income table and notes come first in these filings.
        header = '<P>' + '반기순손실 1,128,864,079 ' * 30 + '</P><TABLE><TR><TD>제 54 기 2015.09.30 현재 (단위 : 백만원)</TD></TR></TABLE>'
        facts, _ = document_facts('<TITLE ATOC="Y">대차대조표</TITLE>' + header + body, '2015-09')
        total = {f['concept']: f for f in facts}['total']
        self.assertEqual((total['equity'], total['statement_type'], total['fiscal_period']), ('60000000', 'OFS', '2015-09-30'))
        self.assertIn('unlabeled_section_title', total['column_basis'])
        facts, _ = document_facts('<TITLE ATOC="Y">합병전후의 대차대조표</TITLE>' + header + body, '2015-09')
        self.assertEqual(facts, [])
        facts, audit = document_facts('<TITLE ATOC="Y">나. 관계회사등의 결산실적</TITLE>' + self.body(), '2015-09')
        self.assertEqual((facts, audit['other_entity_section']), ([], 1))

    def test_section_title_sets_scope_before_subsidiary_wording(self):
        from src.preprocessing.dart_extract import _statement_title
        context = '재무상태표 제 54 기 2015.09.30 현재 주식회사가상과 그 종속기업 (단위 : 원)'
        self.assertEqual(_statement_title(context, '4.재무제표')['scope'], 'OFS')
        self.assertEqual(_statement_title(context, '2.연결재무제표')['scope'], 'CFS')
        self.assertEqual(_statement_title(context, 'III.재무에관한사항')['scope_basis'], 'title_subject')
        self.assertEqual(_statement_title('연결재무상태표 (단위 : 원)', '4.재무제표')['scope'], 'CFS')

    def test_owners_row_that_breaks_its_own_table_loses_a_conflict(self):
        from src.preprocessing.dart_extract import select_filing
        def table(n, owners, nci, total, member='a.xml'):
            loc = {'member': member, 'location': f'TABLE[{n}]/ROW[1]'}
            return [{**self.fact('owners', 'CFS', owners), **loc}, {**self.fact('nci', 'CFS', nci), **loc},
                    {**self.fact('total', 'CFS', total), **loc}]
        typo = table(1, '22098393922', '-6476480217', '15621913775') + table(2, '22098393992', '-6476480217', '15621913775', 'b.xml')
        self.assertEqual(select_filing(typo)['equity'], '22098393992')
        other_company = table(1, '700', '0', '700') + table(2, '900', '100', '1000', 'b.xml')
        self.assertTrue(select_filing(other_company)['reason'].startswith('same_rank_amounts_conflict'))

    def test_rounded_copy_of_one_amount_is_not_a_conflict(self):
        from src.preprocessing.dart_extract import select_filing
        exact = self.fact('owners', 'CFS', '1029099107805')
        rounded = self.fact('owners', 'CFS', '1029099000000', basis='header_term_dated|title|백만원')
        self.assertEqual(select_filing([exact, rounded])['equity'], '1029099107805')
        other = self.fact('owners', 'CFS', '1031000000000', basis='header_term_dated|title|백만원')
        self.assertTrue(select_filing([exact, other])['reason'].startswith('same_rank_amounts_conflict'))

    def test_undated_facts_yield_to_a_dated_full_statement(self):
        from src.preprocessing.dart_extract import select_filing
        undated = lambda f: {**f, 'fiscal_period': None}
        # A summary owners row with no date outranked the dated statement's total less NCI of the same amount.
        total, nci = {**self.fact('total', 'CFS', '100'), **self.table(1)}, {**self.fact('nci', 'CFS', '30'), **self.table(1)}
        summary = {**undated(self.fact('owners', 'CFS', '70', section='summary')), **self.table(2)}
        chosen = select_filing([summary, total, nci])
        self.assertEqual((chosen['equity'], chosen['equity_tier'], chosen['fiscal_period']),
                         ('70', 'T1D_consolidated_total_less_nci', '2015-09-30'))
        # A prior-year adjustment table repeated three times outvoted the one real statement.
        stale = [{**undated(self.fact('owners', 'CFS', '514')), **self.table(n)} for n in (3, 4, 5)]
        real = {**self.fact('owners', 'CFS', '593'), **self.table(6)}
        self.assertEqual(select_filing(stale + [real])['equity'], '593')
        # With nothing dated the undated value stays, still without a date, for review.
        alone = select_filing([undated(self.fact('total', 'OFS', '7'))])
        self.assertEqual((alone['equity'], alone['fiscal_period']), ('7', None))
        # A dated summary alone is not a full statement and does not drop the undated full table.
        mixed = select_filing([undated(self.fact('total', 'OFS', '7')), self.fact('total', 'OFS', '8', section='summary')])
        self.assertEqual(mixed['equity'], '7')

    def test_owners_row_breaking_its_table_is_kept_but_marked_and_held(self):
        import pandas as pd
        from src.preprocessing.dart_extract import _versions
        from src.preprocessing.dart_extract import select_filing
        loc = self.table(1)
        broken = [{**self.fact('owners', 'CFS', '128'), **loc}, {**self.fact('nci', 'CFS', '9'), **loc},
                  {**self.fact('total', 'CFS', '157'), **loc}]
        chosen = select_filing(broken)
        self.assertEqual((chosen['equity'], chosen['owners_identity_ok']), ('128', False))
        balanced = [{**f, 'equity': '148'} if f['concept'] == 'owners' else f for f in broken]
        self.assertIs(select_filing(balanced)['owners_identity_ok'], True)
        # A held-for-sale equity line sits beside owners and NCI; with it the filer's own subtotal balances.
        held = {**self.fact('held_for_sale_equity', 'CFS', '20'), **loc}
        self.assertIs(select_filing(broken + [held])['owners_identity_ok'], True)
        self.assertEqual(select_filing(broken + [held])['equity'], '128')
        self.assertIsNone(select_filing([self.fact('owners', 'CFS', '5')])['owners_identity_ok'])
        # The filer tagged its own wrong subtotal in XBRL too: the XBRL choice inherits the body row's break.
        tagged = self.fact('owners', 'CFS', '128', kind='xbrl', section='xbrl', identity=None, basis='instant_context')
        self.assertEqual(select_filing(broken + [tagged])['source_kind'], 'xbrl')
        self.assertIs(select_filing(broken + [tagged])['owners_identity_ok'], False)
        row = {'corp_code': 'a', 'receipt_id': '1', 'published_at': '20150515', 'report_period_label': '2015-03',
               'revision': '최초표시', 'fiscal_period': '2015-03-31', 'equity': '128', 'unit_suspect': 'False',
               'owners_identity_ok': 'False', 'reason': None}
        self.assertEqual(_versions(pd.DataFrame([row]), {'2015-05': '20150529'})['a'][0]['skip'], 'owners_identity_failed')
        resolved = {**row, 'owners_resolution': 'components_verified_total_less_nci'}
        self.assertIsNone(_versions(pd.DataFrame([resolved]), {'2015-05': '20150529'})['a'][0]['skip'])

    def test_broken_owners_subtotal_replaced_only_when_its_components_add_up(self):
        from src.preprocessing.dart_extract import document_facts, select_filing
        def statement(owners_row, components):
            rows = [self.row('자산총계', '100', '90'), self.row('부채총계', '40', '40'), self.row('지배기업 소유주지분', owners_row, '45')]
            rows += [self.row(label, amount, '1') for label, amount in components]
            rows += [self.row('비지배지분', '5', '5'), self.row('자 본 총 계', '60', '50')]
            return self.STATEMENT.format(rows=''.join(rows))
        # The filer's subtotal 50 is wrong; capital 20 + retained 35 = 55 = total 60 - NCI 5.
        facts, _ = document_facts(statement('50', [('자본금', '20'), ('이익잉여금', '35')]), '2015-09')
        chosen = select_filing(facts)
        self.assertEqual((chosen['equity'], chosen['equity_tier'], chosen['owners_resolution']),
                         ('55000000', 'T1D_consolidated_total_less_nci', 'components_verified_total_less_nci'))
        # Nested subtotals (capital 20 = common 15 + preferred 5) do not add up flat: left flagged for manual entry.
        facts, _ = document_facts(statement('50', [('자본금', '20'), ('보통주자본금', '15'), ('우선주자본금', '5'),
                                                   ('이익잉여금', '35')]), '2015-09')
        nested = select_filing(facts)
        self.assertEqual((nested['equity'], nested['owners_identity_ok']), ('50000000', False))
        self.assertNotIn('owners_resolution', nested)

    def test_overrides_apply_only_approved_unchanged_and_hash_verified(self):
        import hashlib, json, tempfile
        from pathlib import Path
        import pandas as pd
        from src.preprocessing.dart_extract import apply_overrides
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.zip').write_bytes(b'original')
            digest = hashlib.sha256(b'original').hexdigest()
            filings = pd.DataFrame([{'receipt_id': '1', 'equity': '50', 'equity_tier': 'T1_consolidated_owners', 'fiscal_period': None},
                                    {'receipt_id': '2', 'equity': '7', 'equity_tier': 'T3_separate_total', 'fiscal_period': None}]).astype('string')
            entry = lambda i, receipt, before, status='approved', sha=digest: {
                'id': i, 'receipt_id': receipt, 'status': status, 'expected_before': before,
                'set': {'equity': '55', 'owners_resolution': 'manual_override'}, 'evidence': {'raw_path': 'a.zip', 'sha256': sha}}
            path = root / 'o.json'
            path.write_text(json.dumps([entry('EQ-1', '1', {'equity': '50'}), entry('EQ-2', '2', {'equity': '7'}, 'proposed')]), encoding='utf-8')
            out, counts = apply_overrides(root, filings, Path('o.json'))
            self.assertEqual((out.equity.tolist(), out.override_id.tolist()[0], counts['applied']), (['55', '7'], 'EQ-1', 1))
            self.assertTrue(pd.isna(out.override_id.tolist()[1]))
            for bad, error in ((entry('EQ-1', '1', {'equity': '49'}), 'override_stale'),
                               (entry('EQ-1', '1', {'equity': '50'}, sha='0' * 64), 'override_evidence_hash_mismatch')):
                path.write_text(json.dumps([bad]), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, error):
                    apply_overrides(root, filings, Path('o.json'))

    def test_resume_key_changes_with_the_code_not_only_the_version(self):
        from unittest import mock
        from src.preprocessing import dart_extract as dart_equity
        sources = {'document': {'path': 'a.zip', 'sha256': 'x'}, 'xbrl': None}
        before = dart_equity.source_key('20150101000001', sources)
        self.assertEqual(before, dart_equity.source_key('20150101000001', sources))
        with mock.patch.object(dart_equity, 'CODE_FINGERPRINT', '0' * 64):
            self.assertNotEqual(before, dart_equity.source_key('20150101000001', sources))
        self.assertEqual(len(dart_equity.code_fingerprint()), 64)

    def test_source_hash_mismatch_never_yields_a_value(self):
        import hashlib
        import io
        import tempfile
        import zipfile
        from pathlib import Path
        from src.preprocessing.dart_extract import extract_filing
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as archive:
            archive.writestr('20151116000001.xml', self.body().encode('utf-8'))
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'a.zip').write_bytes(out.getvalue())
            job = {'root': temp, 'source_key': 'k', 'corp_code': '00111704', 'rcept_no': '20151116000001',
                   'report_period_label': '2015-09',
                   'document': {'path': 'a.zip', 'sha256': hashlib.sha256(out.getvalue()).hexdigest()}}
            self.assertEqual(extract_filing(job)['decision']['equity'], '55000000')
            job['document']['sha256'] = '0' * 64
            result = extract_filing(job)
            self.assertIsNone(result['decision']['equity'])
            self.assertEqual(result['problems'], ['document:source_hash_mismatch'])


class DartCommonInputTests(unittest.TestCase):
    """Observation inputs must support recomputation without a log or cached choice."""

    def inputs(self):
        import pandas as pd
        from src.contracts import DART_FILING_INPUT_COLUMNS, DART_FACT_INPUT_COLUMNS
        metadata = pd.DataFrame([{'corp_code': 'C1', 'receipt_id': 'R1', 'published_at': '20151113',
            'report_name': '분기보고서 (2015.09)', 'report_kind': '분기', 'report_period_label': '2015-09',
            'listed_in_public_index': True, 'disclosure_format': 'document_only'}], columns=DART_FILING_INPUT_COLUMNS)
        rows, _ = __import__('src.preprocessing.dart_extract', fromlist=['document_facts']).document_facts(
            DartEquityExtractTests().body(), '2015-09')
        facts = pd.DataFrame([{**r, 'corp_code': 'C1', 'receipt_id': 'R1'} for r in rows],
                             columns=DART_FACT_INPUT_COLUMNS).astype('string')
        return metadata, facts

    def test_two_observation_tables_recompute_choices_without_state(self):
        import tempfile
        from pathlib import Path
        from src.preprocessing.dart_extract import DART_FILINGS, DART_FACTS, load_common_dart, decisions_from_facts
        metadata, facts = self.inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / DART_FILINGS).parent.mkdir(parents=True)
            metadata.to_parquet(root / DART_FILINGS, index=False)
            facts.to_parquet(root / DART_FACTS, index=False)
            m, f = load_common_dart(root)
            chosen = decisions_from_facts(m, f)
            self.assertEqual(chosen.iloc[0].equity, '55000000')
            self.assertEqual(chosen.iloc[0].equity_tier, 'T1_consolidated_owners')
            self.assertFalse((root / 'outputs').exists())

    def test_duplicate_receipts_and_conflicting_corporation_are_rejected(self):
        import tempfile
        from pathlib import Path
        import pandas as pd
        from src.preprocessing.dart_extract import DART_FILINGS, DART_FACTS, load_common_dart
        metadata, facts = self.inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / DART_FILINGS).parent.mkdir(parents=True)
            facts.to_parquet(root / DART_FACTS, index=False)
            pd.concat([metadata, metadata]).to_parquet(root / DART_FILINGS, index=False)
            with self.assertRaisesRegex(ValueError, 'duplicate_or_missing_receipt'):
                load_common_dart(root)
            metadata.to_parquet(root / DART_FILINGS, index=False)
            facts['corp_code'] = 'C2'
            facts.to_parquet(root / DART_FACTS, index=False)
            with self.assertRaisesRegex(ValueError, 'corp_conflict'):
                load_common_dart(root)

    def test_components_observation_survives_storage_and_changes_broken_subtotal(self):
        from src.preprocessing.dart_extract import decisions_from_facts
        metadata, facts = self.inputs()
        owners = facts.concept.eq('owners')
        facts.loc[owners, 'equity'] = '51000000'
        facts.loc[owners, 'components_match'] = 'True'
        result = decisions_from_facts(metadata, facts).iloc[0]
        self.assertEqual(result.equity, '55000000')
        self.assertEqual(result.owners_resolution, 'components_verified_total_less_nci')


class DartEquityV11Tests(unittest.TestCase):
    """v11 rules, each from originals seen in history_v10 (2026-11-03 review)."""

    @staticmethod
    def table(rows, heads=('제 26 기', '제 25 기')):
        cells = ''.join(f'<TH>{h}</TH>' for h in heads)
        body = ''.join(f'<TR><TD>{label}</TD>' + ''.join(f'<TD>{v}</TD>' for v in values) + '</TR>' for label, *values in rows)
        return f'<TABLE><THEAD><TR><TH>과 목</TH>{cells}</TR></THEAD><TBODY>{body}</TBODY></TABLE>'

    BALANCED = (('자산총계', '100', '90'), ('부채총계', '40', '40'), ('자 본 총 계', '60', '50'))

    def test_bracketed_totals_are_subtotals_when_total_assets_is_bracketed(self):
        from src.preprocessing.dart_extract import document_facts, select_filing
        head = '<P>재 무 상 태 표 제 54 기 2015.09.30 현재 제 53 기 2015.06.30 현재 (단위 : 백만원)</P>'
        rows = (('자산총계', '(100)', '(90)'), ('부채총계', '(40)', '(40)'), ('자 본 총 계', '(60)', '(50)'))
        facts, audit = document_facts(head + self.table(rows, ('제 54 (당) 기', '제 53 (전) 기')), '2015-09')
        self.assertEqual(select_filing(facts)['equity'], '60000000')
        self.assertTrue(all(f['brackets_as_subtotals'] for f in facts))
        self.assertEqual(audit['brackets_as_subtotals'], 1)
        # A triangle is a minus sign even in such a table: 100 - 110 = -10 still balances.
        rows = (('자산총계', '(100)', '(90)'), ('부채총계', '(110)', '(40)'), ('자 본 총 계', '△10', '(50)'))
        facts, _ = document_facts(head + self.table(rows, ('제 54 (당) 기', '제 53 (전) 기')), '2015-09')
        self.assertEqual(select_filing(facts)['equity'], '-10000000')

    def test_trust_account_and_superseded_statements_are_not_the_company(self):
        from src.preprocessing.dart_extract import document_facts
        dated = ' 제 54 기 2015.09.30 현재 제 53 기 2015.06.30 현재 (단위 : 백만원)</P>'
        heads = ('제 54 (당) 기', '제 53 (전) 기')
        for title in ('<P>[신탁계정] 재 무 상 태 표', '<P>재 무 상 태 표 - 정정전', '<P>대차대조표 (변경전)'):
            self.assertEqual(document_facts(title + dated + self.table(self.BALANCED, heads), '2015-09')[0], [], title)
        self.assertTrue(document_facts('<P>재 무 상 태 표' + dated + self.table(self.BALANCED, heads), '2015-09')[0])

    def test_cover_term_and_period_come_from_the_cover_not_a_correction_notice(self):
        from src.preprocessing.dart_extract import document_facts, select_filing
        cover = ('<COVER><P>정정신고: 제25기 재무제표의 내용 중 일부를 정정합니다.</P>' + '<P>표지</P>' * 1500
                 + '<P>제 26 기 (2003.01.01 부터 2003.12.31 까지)</P></COVER>')
        text = cover + '<P>요약 재무정보 (단위 : 백만원)</P>' + self.table(self.BALANCED)
        decision = select_filing(document_facts(text, '2003-12')[0])
        self.assertEqual((decision['equity'], decision['fiscal_period'], decision['fiscal_period_basis']),
                         ('60000000', '2003-12-31', 'cover_term_period'))

    def test_a_table_whose_latest_term_precedes_the_cover_term_is_a_prior_year_comparative(self):
        from src.preprocessing.dart_extract import document_facts
        cover = '<COVER><P>제 26 기 (2003.01.01 부터 2003.12.31 까지)</P></COVER>'
        prior = '<P>재 무 상 태 표 (단위 : 백만원)</P>' + self.table(self.BALANCED, ('제 25 기', '제 24 기'))
        facts, audit = document_facts(cover + prior, '2003-12')
        self.assertEqual(facts, [])
        self.assertEqual(audit['latest_term_is_prior_period'], 1)

    def test_a_day_the_month_lacks_is_that_month_end(self):
        from src.preprocessing.dart_extract import document_facts, select_filing
        head = '<P>재 무 상 태 표 제 26 기 2003.06.31 현재 제 25 기 2002.12.31 현재 (단위 : 백만원)</P>'
        decision = select_filing(document_facts(head + self.table(self.BALANCED, ('제 26 (당) 기', '제 25 (전) 기')), '2003-06')[0])
        self.assertEqual((decision['equity'], decision['fiscal_period'], decision['fiscal_period_basis']),
                         ('60000000', '2003-06-30', 'invalid_day_month_end'))
