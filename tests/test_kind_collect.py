# 3단계 KIND(kind_collect, 보류·보관용): 압축 저장·해시, 차단 판정, DART 접수번호 짝짓기 핵심만 둔다.

import unittest

import test_dart_extract as extract_tests

class KindTargetsTests(unittest.TestCase):
    """KIND bodies of a DART-paired list: gzip storage, list-only download, extraction. Mocked network, temp folders."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        from src.preprocessing.kind_collect import Collector, Journal, OUTPUT
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.folder = self.root / OUTPUT / '20000101_20211231'
        self.folder.mkdir(parents=True)
        self.journal = Journal(self.folder)
        self.addCleanup(self.journal.db.close)
        self.t = [1000.0]
        self.client = Collector(self.root, self.folder, self.journal, {}, 100, clock=lambda: self.t[0],
                                sleep=lambda s: self.t.__setitem__(0, self.t[0] + s))
        self.addCleanup(self.client.session.close)

    @staticmethod
    def filing(receipt='20151116000001', title='분기보고서'):
        # No period in the title: KIND's own scope says hold; the DART pairing decides instead.
        return {'receipt_no': receipt, 'receipt_date': f'{receipt[:4]}-{receipt[4:6]}-{receipt[6:8]}', 'report_kind': '분기보고서',
                'report_name': title, 'company_name': 'fixture', 'kind_company_code': '01234', 'market_label_observed': '유가증권',
                'revision_mark_observed': False, 'attachment_addition_observed': False}

    @staticmethod
    def response(body):
        from unittest.mock import Mock
        return Mock(status_code=200, headers={'Content-Type': 'text/html; charset=utf-8'},
                    iter_content=Mock(return_value=[body.encode()]), close=Mock())

    def download(self, item, roles, body=None):
        from unittest.mock import patch
        from src.preprocessing.kind_collect import BASE
        receipt, cid = item['receipt_no'], item['receipt_no'][:-2] + '99'
        prefix = f'{BASE}/external/{receipt[:4]}/{receipt[4:6]}/{receipt[6:8]}/{receipt[8:]}/{cid}/'
        body = body or '<html><body>분기보고서' + extract_tests.DartEquityExtractTests().body() + '</body></html>'
        responses = [self.response(f'<option value="{cid}|Y">current</option>'),
                     self.response(f"<script>parent.setPath('{prefix}report_toc.htm','{prefix}report.htm','');</script>"),
                     self.response(body)]
        with patch.object(self.client.session, 'request', side_effect=responses) as network:
            self.client.document(item, roles=roles)
        return network.call_count, body

    def detail(self, receipt):
        import json
        return json.loads(self.journal.db.execute('SELECT detail FROM filings WHERE receipt=?', (receipt,)).fetchone()[0])

    def test_listed_hold_is_downloaded_gzip_stored_and_both_hashes_checked(self):
        import gzip
        import hashlib
        item = self.filing()
        self.client.store_filing(dict(item), 'fixture-list')
        calls, _ = self.download(dict(item), None)
        self.assertEqual(calls, 0)  # unlisted: KIND's own scope still holds an undated title
        self.assertEqual(self.detail(item['receipt_no'])['state'], 'hold_scope')
        calls, body = self.download(dict(item), ['holds'])
        detail = self.detail(item['receipt_no'])
        self.assertEqual((calls, detail['state'], detail['download_basis']), (3, 'original_saved', 'dart_paired_targets'))
        record = detail['response']
        path = self.root / record['path']
        self.assertTrue(path.name.endswith('.html.gz'))
        self.assertEqual(gzip.decompress(path.read_bytes()), body.encode())
        self.assertEqual(record['sha256'], hashlib.sha256(body.encode()).hexdigest())
        self.assertLess(record['stored_bytes'], record['bytes'])
        self.client.verify_response(record)
        path.write_bytes(gzip.compress(b'changed', mtime=0))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.client.verify_response(record)

    def test_block_phrase_inside_a_report_body_is_not_a_block_screen(self):
        from src.preprocessing.kind_collect import StopRun
        item = self.filing()
        self.client.store_filing(dict(item), 'fixture-list')
        report = ('<html><body><h1 class="COVER-TITLE" id="toc_1"><P class=\'COVER-TITLE\'>분 기 보 고 서</h1></P>'
                  '(주)한국가스기기 : 가스누출 자동차단장치') + extract_tests.DartEquityExtractTests().body() + '</body></html>'
        self.download(dict(item), ['targets'], body=report)
        self.assertEqual(self.detail(item['receipt_no'])['state'], 'original_saved')
        other = self.filing('20151117000105')  # its own content id: no cache shared with the first body
        self.client.store_filing(dict(other), 'fixture-list')
        with self.assertRaisesRegex(StopRun, 'blocking_suspected'):
            self.download(dict(other), ['targets'], body='<html><body>자동 수집 방지를 위해 접근이 차단되었습니다</body></html>')

    def write_targets(self, folder):
        import pandas as pd
        folder.mkdir(parents=True)
        base = {'corp_code': '00000001', 'corp_name': 'fixture', 'stock_code': '012340', 'rcept_no': '20151116000500',
                'rcept_dt': '20151116', 'report_kind': '분기', 'report_period_label': '2015-09', 'report_nm': '분기보고서 (2015.09)',
                'dart_cause': 'dart_encoding_broken'}
        pd.DataFrame([{**base, 'kind_receipt_no': '20151116000001', 'outcome': 'target'}]).to_csv(folder / 'targets.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame([{**base, 'rcept_no': '20151116000600', 'kind_receipt_no': '20151116000002;20151116000003',
                       'outcome': 'hold:ambiguous', 'same_day_candidate': '', 'wide_candidate': '20151120K00004'}]).to_csv(
            folder / 'holds.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame([{**base, 'rcept_no': '20151116000700', 'kind_receipt_no': '20151116000001', 'pilot_role': 'dart_value_control'}]).to_csv(
            folder / 'pilot.csv', index=False, encoding='utf-8-sig')
        (folder / 'summary.json').write_text('{}', encoding='utf-8')

    def test_kind_extract_pairs_on_the_dart_receipt_and_compares_controls(self):
        import pandas as pd
        from src.preprocessing.kind_collect import kind_extract
        from src.preprocessing.kind_collect import OUTPUT
        item = self.filing()
        self.client.store_filing(dict(item), 'fixture-list')
        self.download(dict(item), ['targets'])
        targets = self.folder / 'targets' / '20261103T000000'
        self.write_targets(targets)
        dart = self.root / 'data/interim/dart_equity/history_fixture'
        dart.mkdir(parents=True)
        pd.DataFrame({'receipt_id': ['20151116000700'], 'equity': ['55000000']}).to_parquet(dart / 'equity_filings.parquet')
        out = self.root / 'outputs/collection/KIND_EQUITY_EXTRACT/kind_fixture'
        summary = kind_extract(self.root, OUTPUT / self.folder.name, targets, out, dart_extract=dart)
        table = pd.read_parquet(out / 'kind_filings.parquet')
        target = table[table.pairing.eq('target')].iloc[0]
        self.assertEqual((target.rcept_no, target.rcept_dt, target.equity, target.equity_tier),
                         ('20151116000500', '20151116', '55000000', 'T1_consolidated_owners'))
        self.assertEqual(summary['by_pairing']['hold'], {'pairs': 3, 'acquired': 0, 'value': 0, 'dated': 0})
        self.assertEqual(summary['dart_value_controls'], {'compared': 1, 'same_amount': 1})
        again = kind_extract(self.root, OUTPUT / self.folder.name, targets, out, dart_extract=dart)
        self.assertEqual(again['extracted_this_run'], 0)  # resumed: same body, same code

    @staticmethod
    def kind_row(receipt, code, name, date, market='KOSPI', decision='candidate'):
        return {'receipt_no': receipt, 'kind_company_code': code, 'company_name': name, 'report_name': name,
                'receipt_date_observed_or_inferred': date, 'market_group': market, 'decision': decision, 'reasons': 'r'}

    def test_kind_match_needs_the_same_company_kind_and_date_never_the_number_alone(self):
        import pandas as pd
        from src.preprocessing.kind_collect import match_kind
        dart = lambda receipt, stock, name, kind='분기', day='20001114': {
            'corp_code': 'c' + receipt[-2:], 'corp_name': name, 'stock_code': stock, 'corp_cls': 'Y', 'report_nm': kind,
            'rcept_no': receipt, 'rcept_dt': day, 'report_kind': kind, 'report_period_label': '2000-09'}
        receipts = pd.DataFrame([dart('20001114000001', '000010', 'A'), dart('20001114000002', '000020', 'B'),
                                 dart('20001114000003', '000030', 'C'), dart('20001114000004', '000040', 'D'),
                                 dart('20001114000005', '', 'E'), dart('20001114000006', '000060', 'F', '반기')])
        kind = pd.DataFrame([self.kind_row('20001114000001', '99999', 'Z', '2000-11-14'),  # same number, other company
                             {**self.kind_row('20001114K00011', '00001', 'A', '2000-11-15'), 'report_name': '분기보고서(일반법인)'},
                             {**self.kind_row('20001113K00021', '00002', 'B', '2000-11-13'), 'report_name': '분기보고서(일반법인)'},
                             {**self.kind_row('20001114K00022', '00002', 'B', '2000-11-14'), 'report_name': '분기보고서(일반법인)'},
                             {**self.kind_row('20001114K00031', '00003', 'C', '2000-11-14', market='KOSDAQ'), 'report_name': '분기보고서'},
                             {**self.kind_row('20000815K00041', '00004', 'D', '2000-08-15'), 'report_name': '반기보고서'},
                             {**self.kind_row('20001114K00051', '00005', 'E', '2000-11-14'), 'report_name': '분기보고서'},
                             {**self.kind_row('20001114K00061', '00006', 'F', '2000-11-14'), 'report_name': '분기보고서'}])
        out = match_kind(receipts, kind).set_index('rcept_no')
        self.assertEqual(out.match.to_dict(), {'20001114000001': 'target', '20001114000002': 'ambiguous',
                                               '20001114000003': 'kosdaq_at_the_time', '20001114000004': 'not_in_kind_list',
                                               '20001114000005': 'target', '20001114000006': 'kind_report_kind_differs'})
        self.assertEqual((out.loc['20001114000001'].match_basis, out.loc['20001114000001'].kind_receipt_no), ('code_date_kind', '20001114K00011'))
        self.assertEqual(out.loc['20001114000005'].match_basis, 'name_date_kind')
        twice = match_kind(receipts.iloc[[0, 0]].assign(rcept_no=['20001114000001', '20001114000009']), kind)
        self.assertEqual(twice.match.tolist(), ['ambiguous', 'ambiguous'])  # one KIND filing cannot stand for two DART versions
