# 1단계 DART 수집(dart_collect): 보관용 단계라 안전장치(요청 상한·즉시 중단·재개·해시)와 기간 판정 핵심만 둔다.

import unittest

import test_dart_extract as extract_tests

class Responses:
    """OpenDART response stand-ins shared by the collection tests (formerly on the pilot test class)."""

    @staticmethod
    def response(endpoint='document.xml', status=None):
        import io
        import zipfile
        from unittest.mock import Mock
        if status:
            body = f'<result><status>{status}</status></result>'.encode()
        else:
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                archive.writestr('instance.xbrl' if endpoint == 'fnlttXbrl.xml' else 'report.xml', '<root/>')
            body = stream.getvalue()
        return Mock(status_code=200, content=body)



class DartCacheAuditTests(unittest.TestCase):
    """Offline cache/manifest fixtures; no financial or PIT parsing."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / 'data/raw/opendart/filing_originals'
        self.raw.mkdir(parents=True)
        self.manifest = self.root / 'outputs/collection/DART003/manifest.jsonl'
        self.manifest.parent.mkdir(parents=True)
        self.records = []

    @staticmethod
    def row(receipt, name='사업보고서 (2015.12)', corp='00111704'):
        return {'corp_code': corp, 'rcept_no': receipt, 'rcept_dt': receipt[:8],
                'report_nm': name, 'report_kind': '사업', 'report_period_label': '2015-12'}

    def add(self, receipt, endpoint, *, status='ZIP_BODY', tag='a', bad_hash=False, missing=False):
        import io
        import zipfile
        import hashlib
        out = io.BytesIO()
        if status == 'ZIP_BODY':
            with zipfile.ZipFile(out, 'w') as archive:
                archive.writestr('instance.xbrl' if endpoint == 'fnlttXbrl.xml' else 'attachment_00760.xml', '<x>' + tag + '</x>')
            body = out.getvalue()
        else:
            body = b'<result><status>014</status></result>'
        path = self.raw / (receipt + '_' + endpoint + '_' + tag + '.zip')
        if not missing:
            path.write_bytes(body)
        rec = {'event': 'raw', 'endpoint': endpoint, 'params': {'rcept_no': receipt},
               'path': path.relative_to(self.root).as_posix(), 'http': 200, 'status': status,
               'sha256': '0' * 64 if bad_hash else hashlib.sha256(body).hexdigest(), 'bytes': len(body)}
        self.records.append(rec)
        return rec

    def cache(self):
        import json
        from src.preprocessing.dart_collect import Cache
        self.manifest.write_text(''.join(json.dumps(r) + '\n' for r in self.records), encoding='utf-8')
        return Cache(self.root, [self.manifest])

    def test_four_categories_and_failure_file_is_not_an_original(self):
        import pandas as pd
        from src.preprocessing.dart_collect import audit_frame, CATEGORIES
        ids = [f'20160330{i:06d}' for i in range(1, 6)]
        self.add(ids[0], 'fnlttXbrl.xml')
        self.add(ids[0], 'document.xml')
        self.add(ids[1], 'document.xml')
        self.add(ids[1], 'fnlttXbrl.xml', status='014')
        self.add(ids[2], 'fnlttSinglAcntAll.json', status='000')  # deliberately ignored
        self.add(ids[3], 'fnlttXbrl.xml', bad_hash=True)
        self.add(ids[3], 'document.xml', status='014')
        self.add(ids[4], 'fnlttXbrl.xml', missing=True)
        result, summary = audit_frame(pd.DataFrame([self.row(x) for x in ids]), self.cache(), {'00111704'})
        byid = result.set_index('rcept_no')
        self.assertEqual([byid.loc[x, 'cache_category'] for x in ids],
                         [CATEGORIES[0], CATEGORIES[1], CATEGORIES[2], CATEGORIES[3], CATEGORIES[2]])
        self.assertTrue(byid.loc[ids[1], 'xbrl_file_exists'])
        self.assertFalse(byid.loc[ids[1], 'xbrl_response_success'])
        self.assertEqual(summary['new_collection_required_count'], 2)
        self.assertEqual(summary['verification_required_receipts'], [ids[3]])


class DartOriginalCollectTests(unittest.TestCase):
    """Bulk collector tests are offline and only use temporary fixtures."""

    setUp = DartCacheAuditTests.setUp
    add = DartCacheAuditTests.add
    cache = DartCacheAuditTests.cache
    row = staticmethod(DartCacheAuditTests.row)
    response = staticmethod(Responses.response)

    def rows(self, specs=None):
        import pandas as pd
        from src.preprocessing.dart_collect import audit_frame
        rows = specs or [self.row('20160330000001')]
        return audit_frame(pd.DataFrame(rows), self.cache(), set())[0].astype(str).to_dict('records')

    def collector(self, cap=1000):
        from src.preprocessing.dart_collect import Cache
        from src.preprocessing.dart_collect import Collector
        folder = self.root / 'outputs/collection/DART_ORIGINAL_COLLECT/test'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'manifest.jsonl').touch()
        cache = Cache(self.root, [self.manifest, folder / 'manifest.jsonl'])
        collector = Collector(self.root, folder, cache, cap)
        self.addCleanup(collector.session.close)
        return collector

    def test_per_invocation_budget_and_resume_reuses_originals(self):
        from unittest.mock import patch
        rows = self.rows([self.row(f'20160330{i:06d}') for i in range(1, 3)])
        collector = self.collector(cap=1)
        with patch.object(collector.session, 'get', return_value=self.response()) as get, patch.object(collector, 'wait', return_value=True):
            collector.collect(rows, 'TESTKEY')
            self.assertEqual(get.call_count, 1)
        resumed = self.collector(cap=2)
        def response(url, **kwargs):
            self.assertFalse(kwargs['allow_redirects'])
            return self.response('fnlttXbrl.xml' if 'fnlttXbrl' in url else 'document.xml')
        with patch.object(resumed.session, 'get', side_effect=response) as get, patch.object(resumed, 'wait', return_value=True):
            resumed.collect(rows, 'TESTKEY')
            self.assertEqual(get.call_count, 2)
        self.assertEqual(resumed.totals['attempts_lifetime'], 3)
        self.assertEqual(sum(e['event'] == 'request_started' and e['invocation_id'] == resumed.invocation
                             for e in resumed.jobs[rows[0]['rcept_no'] + ':document.xml']), 0)

    def test_JSON_020_stops_immediately_without_retry_or_next_receipt(self):
        from unittest.mock import patch
        import json
        rows = self.rows([self.row(f'20160330{i:06d}') for i in range(1, 3)])
        collector = self.collector(cap=1)
        response = self.response(status='020')
        response.content = b'{"status":"020","message":"request limit exceeded"}'
        with patch.object(collector.session, 'get', return_value=response) as get, patch.object(collector, 'wait', return_value=True) as wait:
            collector.collect(rows, 'TESTKEY')
            self.assertEqual(get.call_count, 1)
            self.assertFalse(any(call.args[0] >= 1 for call in wait.call_args_list))
            self.assertEqual(collector.request(rows[1], 'document', {'rcept_no': rows[1]['rcept_no']}, 'TESTKEY')[2], 'stopped')
            self.assertEqual(get.call_count, 1)
        progress = json.loads((collector.folder / 'progress.json').read_text())
        self.assertEqual(progress['status'], 'stopped')
        self.assertEqual(progress['stop_reason'], 'opendart_request_limit_020')
        self.assertEqual(progress['calls_this_invocation'], 1)
        events = [json.loads(line) for line in collector.manifest.read_text(encoding='utf-8').splitlines()]
        raw = next(e for e in events if e['event'] == 'raw')
        self.assertEqual(raw['status'], '020')
        self.assertFalse(raw['original_success'])
        self.assertEqual(raw['disposition'], 'limited')
        self.assertEqual((self.root / raw['path']).read_bytes(), response.content)
        self.assertTrue(any(e['event'] == 'collection_stopped' and e['status'] == 'stopped'
                            and e['stop_reason'] == 'opendart_request_limit_020' for e in events))
        self.assertFalse(any(e['event'] == 'cooldown' for e in events))

    def test_http_429_is_distinct_and_explicit_resume_preserves_raw_and_history(self):
        from unittest.mock import patch
        import json
        rows = self.rows()
        collector = self.collector()
        response = self.response(status='014')  # misleading payload must not become absence
        response.status_code = 429
        with patch.object(collector.session, 'get', return_value=response) as get, patch.object(collector, 'wait', return_value=True):
            collector.collect(rows, 'TESTKEY')
            self.assertEqual(get.call_count, 1)
        progress = json.loads((collector.folder / 'progress.json').read_text())
        self.assertEqual(progress['status'], 'stopped')
        self.assertEqual(progress['stop_reason'], 'http_429')
        events = [json.loads(line) for line in collector.manifest.read_text(encoding='utf-8').splitlines()]
        raw = next(e for e in events if e['event'] == 'raw')
        self.assertEqual(raw['disposition'], 'limited')
        self.assertFalse(raw['original_success'])
        history_before = collector.manifest.read_bytes()
        original_before = (self.root / raw['path']).read_bytes()
        resumed = self.collector(cap=2)  # Represents an explicitly started new CLI invocation.
        def valid(url, **kwargs):
            return self.response('fnlttXbrl.xml' if 'fnlttXbrl' in url else 'document.xml')
        with patch.object(resumed.session, 'get', side_effect=valid) as get, patch.object(resumed, 'wait', return_value=True):
            resumed.collect(rows, 'TESTKEY')
            self.assertEqual(get.call_count, 2)
        self.assertTrue(resumed.manifest.read_bytes().startswith(history_before))
        self.assertEqual((self.root / raw['path']).read_bytes(), original_before)
        self.assertEqual(resumed.totals['attempts_lifetime'], 3)

    def test_missing_success_is_not_redownloaded_and_stop_is_safe(self):
        from unittest.mock import patch
        receipt = '20160330000001'
        self.add(receipt, 'document.xml')
        self.add(receipt, 'fnlttXbrl.xml')
        rows = self.rows()
        collector = self.collector()
        record = self.records[0]
        (self.root / record['path']).unlink()
        with patch.object(collector.session, 'get') as get:
            collector.collect(rows, 'TESTKEY')
            get.assert_not_called()
        self.assertEqual(collector.states[receipt + ':document.xml']['state'], 'source_integrity_failure')
        resumed = self.collector()
        (resumed.folder / 'STOP').touch()
        with patch.object(resumed.session, 'get') as get:
            resumed.collect(rows, 'TESTKEY')
            get.assert_not_called()

    def test_appended_manifest_still_matches_its_pinned_version_but_an_edit_does_not(self):
        import hashlib
        from src.preprocessing.dart_collect import appended_to
        path = self.root / 'manifest.jsonl'
        path.write_bytes(b'{"a":1}\n{"b":2}\n')
        pinned = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open('ab') as stream:
            stream.write(b'{"c":3}\n')
        self.assertTrue(appended_to(path, pinned))
        path.write_bytes(b'{"a":1}\n{"b":9}\n{"c":3}\n')
        self.assertFalse(appended_to(path, pinned))

    def test_files_of_later_runs_are_accounted_for_but_not_reused(self):
        import json
        from src.preprocessing.dart_collect import Cache
        later = self.root / 'outputs/collection/DART_QUARTER_COLLECT/q/manifest.jsonl'
        later.parent.mkdir(parents=True)
        record = self.add('20160330000009', 'fnlttXbrl.xml')
        self.records.clear()
        later.write_text(json.dumps(record) + '\n', encoding='utf-8')
        self.manifest.write_text('', encoding='utf-8')
        self.assertTrue(Cache(self.root, [self.manifest]).unlinked_original_files)
        cache = Cache(self.root, [self.manifest], known_only=[later])
        self.assertEqual(cache.unlinked_original_files, [])
        self.assertEqual(cache.original('20160330000009', '00111704', 'fnlttXbrl.xml')['state'], 'no_manifest_record')

    def test_refetch_stopped_by_limit_continues_on_the_next_run(self):
        import csv
        from unittest.mock import patch
        from src.preprocessing.dart_collect import DELETION_LEDGERS
        receipt = '20160330000001'
        records = [self.add(receipt, 'document.xml'), self.add(receipt, 'fnlttXbrl.xml')]
        rows = self.rows()
        ledger = self.root / DELETION_LEDGERS[0]
        ledger.parent.mkdir(parents=True)
        with ledger.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['path', 'bytes', 'sha256'])
            writer.writerows([r['path'], r['bytes'], r['sha256']] for r in records)
        for r in records:
            (self.root / r['path']).unlink()
        limit = self.response()
        limit.content = b'{"status":"020","message":"request limit exceeded"}'
        first = self.collector()
        with patch.object(first.session, 'get', side_effect=[self.response(), limit]) as get, \
                patch.object(first, 'wait', return_value=True):
            first.collect(rows, 'TESTKEY')
        self.assertEqual(get.call_count, 2)  # document received, then the limit on the XBRL request
        self.assertEqual(first.states[receipt + ':fnlttXbrl.xml']['state'], 'limited')
        resumed = self.collector()
        with patch.object(resumed.session, 'get', return_value=self.response('fnlttXbrl.xml')) as get, \
                patch.object(resumed, 'wait', return_value=True):
            resumed.collect(rows, 'TESTKEY')
        self.assertEqual(get.call_count, 1)  # only the XBRL; the document received yesterday is reused
        self.assertTrue(get.call_args.args[0].endswith('/fnlttXbrl.xml'))
        self.assertEqual({resumed.states[receipt + ':' + e]['state'] for e in ('document.xml', 'fnlttXbrl.xml')},
                         {'original_verified'})

    def test_original_deleted_by_cleanup_is_requested_again_into_the_same_run(self):
        import csv
        from unittest.mock import patch
        from src.preprocessing.dart_collect import DELETION_LEDGERS
        receipt = '20160330000001'
        document = self.add(receipt, 'document.xml')
        xbrl = self.add(receipt, 'fnlttXbrl.xml')
        rows = self.rows()  # audited while both files were present
        ledger = self.root / DELETION_LEDGERS[0]
        ledger.parent.mkdir(parents=True)
        with ledger.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['path', 'bytes', 'sha256'])
            writer.writerow([document['path'], document['bytes'], document['sha256']])  # xbrl is not in the ledger
        (self.root / document['path']).unlink()
        (self.root / xbrl['path']).unlink()
        collector = self.collector()
        with patch.object(collector.session, 'get', return_value=self.response()) as get, \
                patch.object(collector, 'wait', return_value=True):
            collector.collect(rows, 'TESTKEY')
        self.assertEqual(get.call_count, 1)
        self.assertTrue(get.call_args.args[0].endswith('/document.xml'))
        state = collector.states[receipt + ':document.xml']
        self.assertEqual(state['state'], 'original_verified')
        self.assertTrue(state['path'].startswith('data/raw/opendart/original_collect/test/'))
        # A vanished file that no approved cleanup lists is still an integrity incident, never re-requested.
        self.assertEqual(collector.states[receipt + ':fnlttXbrl.xml']['state'], 'source_integrity_failure')
        resumed = self.collector()  # same command again: the new file is reused, nothing requested
        with patch.object(resumed.session, 'get') as get:
            resumed.collect(rows, 'TESTKEY')
            get.assert_not_called()
        self.assertEqual(resumed.states[receipt + ':document.xml']['path'], state['path'])

    def test_keyboard_interrupt_and_exclusive_lock(self):
        from unittest.mock import patch
        from src.preprocessing.dart_collect import run_lock
        rows = self.rows()
        collector = self.collector()
        with patch.object(collector.session, 'get', side_effect=KeyboardInterrupt), patch.object(collector, 'wait', return_value=True):
            collector.collect(rows, 'TESTKEY')
        import json
        self.assertEqual(json.loads((collector.folder / 'progress.json').read_text())['status'], 'interrupted')
        resumed = self.collector()
        with patch.object(resumed.session, 'get') as get:
            self.assertEqual(resumed.request(rows[0], 'document', {'rcept_no': rows[0]['rcept_no']}, 'TESTKEY')[0], 'interrupted_request_unconfirmed')
            get.assert_not_called()
        with run_lock(collector.folder):
            with self.assertRaises(FileExistsError):
                with run_lock(collector.folder):
                    self.fail('second writer acquired existing lock')


class DartListOnlyTests(unittest.TestCase):
    """Only two preserved list responses and small synthetic pages; no network."""

    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        self.project = Path(__file__).resolve().parents[1]
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        folder = self.root / 'data/processed/opendart'
        folder.mkdir(parents=True)
        (folder / 'company_mapping.parquet').write_bytes(
            (self.project / 'data/processed/opendart/company_mapping.parquet').read_bytes())
        for name in ['filing_versions', 'equity_facts']:
            (folder / (name + '.parquet')).write_bytes(b'untouched legacy fixture')
        self.legacy = self.root / 'outputs/collection/DART003'
        self.legacy.mkdir(parents=True)
        (self.legacy / 'progress.json').write_text('{"legacy":true}', encoding='utf-8')
        from src.preprocessing.dart_collect import BASE
        from src.preprocessing.dart_extract import _sha, _request_key
        self.samples = []
        for corp in ['00111704', '00126380']:
            params = {'corp_code': corp, 'bgn_de': '20150101', 'end_de': '20261031', **BASE, 'page_no': '1'}
            data = {'status': '000', 'total_page': 1, 'total_count': 2, 'page_count': 100, 'page_no': 1,
                    'list': [{'corp_code': corp, 'rcept_no': f'20160330{corp[-6:]}', 'rcept_dt': '20160330',
                              'report_nm': '사업보고서 (2015.12)'},
                             {'corp_code': corp, 'rcept_no': f'20160401{corp[-6:]}', 'rcept_dt': '20160401',
                              'report_nm': '[기재정정]사업보고서 (2015.12)'}]}
            body = json.dumps(data).encode('utf-8')
            rec = {'event': 'raw', 'endpoint': 'list.json', 'params': params,
                   'key': _request_key('list.json', params), 'path': f'data/raw/opendart/filing_index/{corp}.json',
                   'sha256': _sha(body), 'bytes': len(body), 'http': 200, 'status': '000'}
            target = self.root / rec['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            self.samples.append(rec)
        (self.legacy / 'manifest.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in self.samples), encoding='utf-8')

    def collector(self, **kwargs):
        from src.preprocessing.dart_collect import ListCollector
        return ListCollector(self.root, offline=True, **kwargs)

    def cached_page(self, c, corp, data):
        import json
        from src.preprocessing.dart_collect import BASE
        from src.preprocessing.dart_extract import _sha, _request_key
        params = {'corp_code': corp, 'bgn_de': c.start, 'end_de': c.end, **BASE, 'page_no': '1'}
        body = json.dumps(data).encode()
        path = self.root / 'data/raw/opendart/synthetic_page1.json'
        path.write_bytes(body)
        record = {'event': 'raw', 'endpoint': 'list.json', 'params': params,
                  'key': _request_key('list.json', params), 'path': path.relative_to(self.root).as_posix(),
                  'sha256': _sha(body), 'bytes': len(body), 'http': 200, 'status': data['status']}
        c.cache = {record['key']: record}
        c.raw_records = [record]

    @staticmethod
    def response(data):
        import json
        from unittest.mock import Mock
        response = Mock(status_code=200, content=json.dumps(data).encode())
        response.json.return_value = data
        return response

    @staticmethod
    def paged(duplicate=False):
        rows = [{'corp_code': '00111704', 'rcept_no': f'20160330{i:06d}',
                 'rcept_dt': '20160330', 'report_nm': '사업보고서 (2015.12)'} for i in range(101)]
        common = {'status': '000', 'page_count': 100, 'total_count': 101, 'total_page': 2}
        return ({**common, 'page_no': 1, 'list': rows[:100]},
                {**common, 'page_no': 2, 'list': [rows[0] if duplicate else rows[-1]]})

    def test_preserved_original_and_corrections_and_separate_checkpoint(self):
        import json
        from unittest.mock import patch
        from src.preprocessing.dart_extract import CollectionStop, _report_group
        c = self.collector()
        before = (self.legacy / 'progress.json').read_bytes()
        c.corps = ['00111704', '00126380']  # bounded two-company offline fixture order
        with patch('src.preprocessing.dart_extract.requests.get', side_effect=AssertionError('network forbidden')) as network:
            for endpoint in ['fnlttSinglAcntAll.json', 'fnlttXbrl.xml', 'document.xml', 'corpCode.xml']:
                with self.assertRaises(CollectionStop):
                    c.get(endpoint, {})
            report = c.run_lists(2)
            network.assert_not_called()
        self.assertEqual(report['network_calls'], 0)
        self.assertEqual(report['completed_list_corps'], 2)
        expected = set()
        corrections = 0
        for rec in self.samples:
            data = json.loads((self.root / rec['path']).read_bytes())
            self.assertEqual(int(data['total_page']), 1)
            for row in data['list']:
                if _report_group(row['report_nm']):
                    expected.add((row['corp_code'], row['rcept_no']))
                    corrections += '정정' in row['report_nm']
        self.assertGreater(corrections, 0)
        self.assertEqual(set(c.filings), expected)
        self.assertEqual((self.legacy / 'progress.json').read_bytes(), before)
        for name in ['filing_versions', 'equity_facts']:
            self.assertEqual((self.root / f'data/processed/opendart/{name}.parquet').read_bytes(), b'untouched legacy fixture')
        resumed = self.collector()
        self.assertEqual(resumed.progress['completed_corps'], c.progress['completed_corps'])

    def test_adjacent_identical_repeat_counts_once_even_across_a_page_break(self):
        from src.preprocessing.dart_extract import CollectionStop
        from src.preprocessing.dart_collect import validate_pages
        row = lambda receipt, name='a': {'rcept_no': receipt, 'report_nm': name}
        def window(*parts):
            total = sum(map(len, parts))
            return {n: ({}, {'status': '000', 'total_count': str(total), 'total_page': str(len(parts)), 'page_no': str(n),
                             'page_count': '100', 'list': part}) for n, part in enumerate(parts, 1)}
        rows = validate_pages(window([row('1'), row('1'), row('2')]), require_complete=True)
        self.assertEqual([r['rcept_no'] for r in rows], ['1', '2'])
        head = [row(str(i)) for i in range(99)] + [row('x')]
        rows = validate_pages(window(head, [row('x'), row('x'), row('y')]), require_complete=True)
        self.assertEqual(len(rows), 101)
        for broken in (window([row('1'), row('1', 'b')]), window([row('1'), row('2'), row('1')])):
            with self.assertRaises(CollectionStop):
                validate_pages(broken, require_complete=True)

    def test_call_budget_and_stop_file(self):
        from unittest.mock import patch
        c = self.collector(max_calls=0)
        c.cache = {}
        c.offline = False
        c.corps = ['00111704']
        with patch('src.preprocessing.dart_extract.requests.get', side_effect=AssertionError('network forbidden')) as network:
            report = c.run_lists(1)
            network.assert_not_called()
        self.assertEqual(report['stop_reason'], 'request_budget_reached')
        self.assertEqual(report['completed_list_corps'], 0)
        stopped = self.collector()
        (stopped.run / 'STOP').touch()
        with patch('src.preprocessing.dart_extract.requests.get', side_effect=AssertionError('network forbidden')) as network:
            report = stopped.run_lists(1)
            network.assert_not_called()
        self.assertEqual(report['stop_reason'], 'list_stop_requested')


class DartMarketListTests(unittest.TestCase):
    """corp_code-free quarter windows over a mocked list.json; no key or network."""

    def collector(self, root):
        from src.preprocessing.dart_collect import MarketListCollector
        return MarketListCollector(root, start='20000101', end='20000630', max_calls=20, offline=True)

    @staticmethod
    def page(lo, count, number=1, name='사업보고서 (1999.12)'):
        rows = [{'corp_code': f'{i:08d}', 'corp_name': 'x', 'stock_code': '', 'corp_cls': 'E', 'report_nm': name,
                 'rcept_no': f'{lo}{i:06d}', 'flr_nm': 'x', 'rcept_dt': lo, 'rm': ''} for i in range(count)]
        return {'status': '000', 'total_count': count, 'total_page': 1, 'page_no': number, 'page_count': 100, 'list': rows}

    def scripted(self, collector, pages):
        requested = []

        def get(endpoint, params):
            requested.append(params)
            if 'corp_code' in params or params['end_de'] > '20000630':
                raise AssertionError('outside contract')
            return {'path': 'p', 'sha256': 's'}, b'', pages[params['bgn_de']]
        collector.get = get
        return requested

    def test_quarter_windows_clip_and_never_exceed_three_months(self):
        from src.preprocessing.dart_collect import quarter_windows
        windows = quarter_windows('20000101', '20211231')
        self.assertEqual(len(windows), 88)
        self.assertEqual(windows[0], ('20000101', '20000331'))
        self.assertEqual(windows[-1], ('20211001', '20211231'))
        self.assertEqual(quarter_windows('20000215', '20000520'), [('20000215', '20000331'), ('20000401', '20000520')])


class DartQuarterAuditTests(unittest.TestCase):
    """Read-only metadata audit fixtures; every file is temporary and HTTP is forbidden."""

    @staticmethod
    def cover(receipt='20151116000001', start='20150101', end='20150930', corp='00111111', cover=True):
        import io
        import zipfile
        text = ('<?xml version="1.0" encoding="utf-8"?><DOCUMENT><DOCUMENT-HEADER>'
                '<DOCUMENT-NAME ACODE="11013">분기보고서</DOCUMENT-NAME>'
                f'<COMPANY-NAME AREGCIK="{corp}">가상기업</COMPANY-NAME></DOCUMENT-HEADER>')
        if cover:
            text += ('<COVER><COVER-TITLE>분 기 보 고 서</COVER-TITLE><P>(제 10 기)</P>'
                     f'<P>사업연도</P><TU AUNIT="PERIODFROM" AUNITVALUE="{start}">시작</TU>'
                     f'<TU AUNIT="PERIODTO" AUNITVALUE="{end}">종료</TU></COVER>')
        text += '</DOCUMENT>'
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as archive:
            archive.writestr('/' + receipt + '.xml', text.encode('cp949'))
        return output.getvalue()

    def test_cover_period_selects_q3_even_when_form_acode_is_q1(self):
        from src.preprocessing.dart_collect import quarter_cover_metadata
        result = quarter_cover_metadata(self.cover(), '00111111', '20151116000001', '분기보고서 (2015.09)')
        self.assertEqual(result['code'], '11014')
        self.assertEqual(result['period_start'], '2015-01-01')
        self.assertEqual(result['period_end'], '2015-09-30')
        self.assertEqual(result['encoding'], 'cp949')
        self.assertEqual(result['header_acode_not_used_for_quarter'], '11013')


class DartQuarterSelectionTests(unittest.TestCase):
    """Exact-plan selective requests; originals and evidence stay in temporary fixtures."""

    setUp = DartCacheAuditTests.setUp
    add = DartCacheAuditTests.add

    @staticmethod
    def plan():
        import json
        import pandas as pd
        rows = []
        for index, code in enumerate(('11013', '11014', None)):
            receipt = f'20151116{index + 1:06d}'
            start, end = '2015-01-01', '2015-03-31' if code == '11013' else '2015-09-30'
            rows.append({'rcept_no': receipt, 'corp_code': '00111704', 'rcept_dt': '20151116',
                'report_nm': '분기보고서', 'report_kind': '분기', 'report_period_label': end[:7],
                'group_id': 'fixture', 'version_order': '1', 'missing_initial': 'False', 'audit_exception': 'False',
                'endpoint': 'fnlttXbrl.xml', 'job': receipt + ':fnlttXbrl.xml',
                'action': 'request_ready' if code else 'unresolved', 'source_listing_matched': 'True',
                'confirmed_reprt_code': code or '', 'period_start': start, 'period_end': end,
                'request_params_if_ready': json.dumps({'rcept_no': receipt, 'reprt_code': code}) if code else '',
                'metadata_evidence': json.dumps({'code': code, 'corporation_confirmed': True, 'cover_location': 'fixture/COVER',
                                                 'period_start': start, 'period_end': end}),
                'cache_evidence': json.dumps({'xbrl': {'reusable': False}}), 'source_files': '[]'})
        summary = {'by_action': {'offline_resolved': 0, 'request_ready': 2, 'unresolved': 1},
                   'confirmed_reprt_codes': {'11013': 1, '11014': 1}}
        return pd.DataFrame(rows), summary

    def selected(self):
        from src.preprocessing.dart_collect import validate_quarter_selection
        frame, summary = self.plan()
        return validate_quarter_selection(frame, summary, expected_ready=2, expected_total=3)

    def collector(self, cap=10, name='selected'):
        import json
        from src.preprocessing.dart_collect import Collector, QUARTER_OUTPUT, QUARTER_RAW
        from src.preprocessing.dart_collect import Cache
        folder = self.root / QUARTER_OUTPUT / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'manifest.jsonl').touch()
        self.manifest.write_text('\n'.join(json.dumps(r) for r in self.records), encoding='utf-8')
        cache = Cache(self.root, [self.manifest, *sorted((self.root / QUARTER_OUTPUT).glob('*/manifest.jsonl'))])
        collector = Collector(self.root, folder, cache, cap, raw_parent=QUARTER_RAW)
        self.addCleanup(collector.session.close)
        return collector

    @staticmethod
    def identity():
        return {'selection_csv': 'approved/candidates.csv', 'selection_csv_sha256': 'fixture-digest'}

    def test_020_and_429_halt_all_remaining_jobs_without_retry(self):
        import json
        from unittest.mock import Mock, patch
        for index, (http, body, reason) in enumerate([(200, b'<result><status>020</status></result>', 'opendart_request_limit_020'),
                (200, b'{"status":"020"}', 'opendart_request_limit_020'), (429, b'limited', 'http_429')]):
            collector = self.collector(name='limit' + str(index))
            with patch.object(collector.session, 'get', return_value=Mock(status_code=http, content=body)) as get, \
                 patch.object(collector, 'wait', return_value=True):
                collector.collect_quarter(self.selected(), 'TESTKEY', self.identity())
            self.assertEqual(get.call_count, 1)
            report = json.loads((collector.folder / 'progress.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'stopped')
            self.assertEqual(report['stop_reason'], reason)
            self.assertEqual(report['calls_this_invocation'], 1)
            self.assertEqual(report['unvisited_endpoints'], 1)
            self.assertEqual(report['http_status_counts'], {str(http): 1})
            self.assertNotIn('original_verified', report['endpoint_states'])
            self.assertNotIn('absence_014', report['endpoint_states'])


class DartHistorySelectionTests(unittest.TestCase):
    """Target selection and body-only traversal; responses are mocked."""

    setUp = DartCacheAuditTests.setUp

    @staticmethod
    def filing(receipt, label='2003-12', kind='사업', corp='00000001', stock='', cls='E', rm=''):
        return {'corp_code': corp, 'corp_name': 'x', 'stock_code': stock, 'corp_cls': cls, 'report_nm': 'x',
                'rcept_no': receipt, 'flr_nm': 'x', 'rcept_dt': receipt[:8], 'rm': rm,
                'report_kind': kind, 'report_period_label': label}

    def select(self, market, listed=(), kind=(), done=(), sample=1):
        import pandas as pd
        from src.preprocessing.dart_collect import select_history
        mapping = pd.DataFrame({'stock_id': ['000020', '000040'], 'corp_code': ['00000002', None]})
        columns = list(self.filing('20040101000001'))
        return select_history(pd.DataFrame(market, columns=columns), pd.DataFrame(list(listed), columns=columns),
                              mapping, set(kind), set(done), sample)

    def test_any_basis_qualifies_and_all_are_recorded(self):
        frame = self.select([self.filing('20040101000001', stock='000040'), self.filing('20040101000002', corp='00000002'),
                             self.filing('20040101000003', cls='Y', corp='00000003'), self.filing('20040101000004', corp='00000004'),
                             self.filing('20040101000005', corp='00000005')], kind=['20040101000004'])
        self.assertEqual(set(frame.rcept_no), {f'2004010100000{i}' for i in range(1, 5)})
        row = frame.set_index('rcept_no').loc['20040101000004']
        self.assertTrue(row.basis_kind_kospi)
        self.assertFalse(row.basis_panel_code)

    def collector(self, cap=10):
        from collections import defaultdict
        from src.preprocessing.dart_collect import Collector, HISTORY_OUTPUT, HISTORY_RAW
        from src.preprocessing.dart_collect import Cache
        folder = self.root / HISTORY_OUTPUT / 'fixture'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'manifest.jsonl').touch()
        cache = Cache.__new__(Cache)
        cache.root, cache.raw, cache.checked = self.root, (self.root / 'data/raw/opendart').resolve(), {}
        cache.index, cache.unlinked_original_files, cache.manifest_issues = defaultdict(list), [], []
        collector = Collector(self.root, folder, cache, cap, raw_parent=HISTORY_RAW)
        self.addCleanup(collector.session.close)
        return collector

    @staticmethod
    def rows(count=3):
        return [{'corp_code': '00000001', 'rcept_no': f'2004010100000{i}', 'rcept_dt': '20040101', 'report_nm': '사업보고서 (2003.12)',
                 'report_kind': '사업', 'report_period_label': '2003-12', 'group_id': 'g', 'version_order': '', 'missing_initial': 'False',
                 'exception': 'False', 'receipt_link_ambiguous': 'False', 'document_reusable': 'False'} for i in range(count)]

    identity = {'mode': 'history-collect', 'selection_csv': 's.csv', 'selection_csv_sha256': 'fixture'}

    def test_body_only_success_absence_and_resume(self):
        import json
        from unittest.mock import patch
        collector = self.collector()
        responses = [Responses.response('document.xml'), Responses.response(status='014'),
                     Responses.response('document.xml')]
        with patch.object(collector.session, 'get', side_effect=responses) as get, patch.object(collector, 'wait', return_value=True):
            collector.collect_history(self.rows(), 'TESTKEY', self.identity, 10**9)
        self.assertTrue(all(c.args[0].endswith('/document.xml') for c in get.call_args_list))
        report = json.loads((collector.folder / 'progress.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'history_traversal_finished_not_PIT_verified')
        self.assertEqual(report['endpoint_states'], {'original_verified': 2, 'absence_014': 1})
        self.assertEqual(report['mode'], 'history-collect')
        self.assertGreater(report['raw_bytes_this_run'], 0)

    def test_retry_failed_reruns_only_failures_and_only_when_asked(self):
        import json
        from unittest.mock import patch
        from src.preprocessing.dart_collect import Collector, HISTORY_RAW
        response = Responses.response
        first = self.collector()
        again = lambda: Collector(self.root, first.folder, first.cache, 10, raw_parent=HISTORY_RAW)  # one cache, as one run
        with patch.object(first.session, 'get', side_effect=[response(status='100'), response(status='014'), response()]), \
                patch.object(first, 'wait', return_value=True):
            first.collect_history(self.rows(), 'TESTKEY', self.identity, 10**9)
        self.assertEqual(first.counts['endpoint_failed'], 1)
        resumed = again()
        with patch.object(resumed.session, 'get') as get, patch.object(resumed, 'wait', return_value=True):
            resumed.collect_history(self.rows(), 'TESTKEY', self.identity, 10**9)
        self.assertEqual(get.call_count, 0)  # an ordinary resume never re-requests a recorded failure
        retry = again()
        with patch.object(retry.session, 'get', side_effect=[response()]) as get, patch.object(retry, 'wait', return_value=True):
            retry.collect_history(self.rows(), 'TESTKEY', self.identity, 10**9, retry_failed=True)
        self.assertEqual(get.call_count, 1)  # the 014 and the success are not requested again
        report = json.loads((retry.folder / 'progress.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'retry_failed_finished')
        self.assertEqual(report['endpoint_states'], {'original_verified': 2, 'absence_014': 1})

    def test_same_folder_rerun_requests_only_deleted_originals_even_after_old_retries(self):
        import json
        from unittest.mock import patch
        from src.preprocessing.dart_collect import Collector, HISTORY_RAW
        response = Responses.response
        first = self.collector()
        with patch.object(first.session, 'get', side_effect=[response(status='800')] * 3 + [response()]), \
                patch.object(first, 'wait', return_value=True):
            first.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9)
        self.assertEqual(first.counts['retry_exhausted'], 1)
        retry = Collector(self.root, first.folder, first.cache, 10, raw_parent=HISTORY_RAW)
        self.addCleanup(retry.session.close)
        with patch.object(retry.session, 'get', side_effect=[response()]), patch.object(retry, 'wait', return_value=True):
            retry.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9, retry_failed=True)
        self.assertEqual(retry.counts['original_verified'], 2)  # receipt 0 now has 3 old transient attempts on record
        events = [json.loads(line) for line in first.manifest.read_text(encoding='utf-8').splitlines()]
        kept, deleted = [e for e in events if e['event'] == 'raw' and e['original_success']]
        self.assertEqual(deleted['rcept_no'], self.rows(1)[0]['rcept_no'])
        cache = first.cache
        cache.deleted = {deleted['path']: deleted['sha256']}
        cache.checked.clear()
        (self.root / deleted['path']).unlink()
        again = Collector(self.root, first.folder, cache, 10, raw_parent=HISTORY_RAW)
        self.addCleanup(again.session.close)
        self.assertEqual(again.raw_bytes - again.purged_bytes,
                         sum(e['bytes'] for e in events if e['event'] == 'raw') - deleted['bytes'])
        with patch.object(again.session, 'get', side_effect=[response()]) as get, patch.object(again, 'wait', return_value=True):
            again.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(again.states[deleted['rcept_no'] + ':document.xml']['state'], 'original_verified')
        self.assertTrue((self.root / again.states[deleted['rcept_no'] + ':document.xml']['path']).exists())
        self.assertEqual(again.states[kept['rcept_no'] + ':document.xml']['path'], kept['path'])

    def test_refetch_stopped_by_020_resumes_with_the_same_command_ignoring_pre_cleanup_retries(self):
        import json
        from unittest.mock import patch
        from src.preprocessing.dart_collect import Collector, HISTORY_RAW
        response = Responses.response
        first = self.collector()
        with patch.object(first.session, 'get', side_effect=[response(status='800')] * 3 + [response()]), \
                patch.object(first, 'wait', return_value=True):
            first.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9)
        fix = Collector(self.root, first.folder, first.cache, 10, raw_parent=HISTORY_RAW)
        self.addCleanup(fix.session.close)
        with patch.object(fix.session, 'get', side_effect=[response()]), patch.object(fix, 'wait', return_value=True):
            fix.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9, retry_failed=True)
        raws = [e for e in (json.loads(line) for line in first.manifest.read_text(encoding='utf-8').splitlines())
                if e['event'] == 'raw' and e['original_success']]
        first.cache.deleted = {e['path']: e['sha256'] for e in raws}
        first.cache.checked.clear()
        for e in raws:
            (self.root / e['path']).unlink()
        limit = response()
        limit.content = b'{"status":"020","message":"request limit exceeded"}'
        stopped = Collector(self.root, first.folder, first.cache, 10, raw_parent=HISTORY_RAW)
        self.addCleanup(stopped.session.close)
        with patch.object(stopped.session, 'get', side_effect=[limit]) as get, patch.object(stopped, 'wait', return_value=True):
            stopped.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(stopped.last_status, 'stopped')
        nextday = Collector(self.root, first.folder, first.cache, 10, raw_parent=HISTORY_RAW)  # same command, next day
        self.addCleanup(nextday.session.close)
        with patch.object(nextday.session, 'get', side_effect=[response(), response()]) as get, \
                patch.object(nextday, 'wait', return_value=True):
            nextday.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9)
        self.assertEqual(get.call_count, 2)  # receipt 0's three pre-cleanup transient attempts do not count
        self.assertEqual(nextday.last_status, 'history_traversal_finished_not_PIT_verified')
        self.assertEqual(nextday.counts['original_verified'], 2)

    def test_deleted_original_that_dart_now_answers_differently_is_recorded_not_an_error(self):
        import json
        from unittest.mock import patch
        from src.preprocessing.dart_collect import Collector, HISTORY_RAW
        response = Responses.response
        first = self.collector()
        with patch.object(first.session, 'get', side_effect=[response(), response()]), patch.object(first, 'wait', return_value=True):
            first.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9)
        raws = [json.loads(line) for line in first.manifest.read_text(encoding='utf-8').splitlines()]
        raws = [e for e in raws if e['event'] == 'raw']
        first.cache.deleted = {e['path']: e['sha256'] for e in raws}
        first.cache.checked.clear()
        for e in raws:
            (self.root / e['path']).unlink()
        import io
        import zipfile
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('report.xml', '<root>reissued</root>')  # same receipt, different bytes
        changed = response()
        changed.content = stream.getvalue()
        again = Collector(self.root, first.folder, first.cache, 10, raw_parent=HISTORY_RAW)
        self.addCleanup(again.session.close)
        with patch.object(again.session, 'get', side_effect=[changed, response(status='014')]), \
                patch.object(again, 'wait', return_value=True):
            again.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9)
        report = json.loads((again.folder / 'progress.json').read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'history_traversal_finished_not_PIT_verified')
        self.assertEqual(report['endpoint_states'], {'original_verified': 1, 'absence_014': 1})
        self.assertNotEqual(again.states[raws[0]['rcept_no'] + ':document.xml']['sha256'], raws[0]['sha256'])
        once_more = Collector(self.root, first.folder, first.cache, 10, raw_parent=HISTORY_RAW)
        self.addCleanup(once_more.session.close)
        with patch.object(once_more.session, 'get') as get:
            once_more.collect_history(self.rows(2), 'TESTKEY', self.identity, 10**9)
            get.assert_not_called()  # the new answers are kept; nothing is requested a third time
        self.assertEqual(once_more.counts['absence_014'], 1)


class DartHistoryTests(unittest.TestCase):
    """Night chain and morning report on temporary fixtures; responses are mocked, no network."""

    setUp = DartCacheAuditTests.setUp
    identity = DartHistorySelectionTests.identity

    @staticmethod
    def zipped(equity='60'):
        import io
        import zipfile
        import types
        text = extract_tests.DartEquityExtractTests().body(equity=equity).replace('2015.09.30', '2003.12.31').replace('2014.09.30', '2002.12.31')
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as archive:
            archive.writestr('20040101000001.xml', text.encode('utf-8'))
        return types.SimpleNamespace(status_code=200, content=out.getvalue(), headers={})

    @staticmethod
    def rows(count=2, year=2003, sample=False):
        return [{'corp_code': '00000001', 'rcept_no': f'{year + 1}0101{i:06d}', 'rcept_dt': f'{year + 1}0101',
                 'report_nm': f'사업보고서 ({year}.12)', 'report_kind': '사업', 'report_period_label': f'{year}-12',
                 'group_id': 'g', 'version_order': '', 'missing_initial': 'False', 'exception': 'False',
                 'receipt_link_ambiguous': 'False', 'document_reusable': 'False', 'sample_first': str(sample)} for i in range(count)]

    def prepare(self):
        from collections import defaultdict
        from src.preprocessing.dart_collect import HISTORY_OUTPUT
        from src.preprocessing.dart_collect import Cache
        self.folder = self.root / HISTORY_OUTPUT / 'fixture'
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / 'manifest.jsonl').touch(exist_ok=True)
        self.extract = self.root / 'data/interim/dart_equity/history_rolling'
        self.extract.mkdir(parents=True, exist_ok=True)
        cache = Cache.__new__(Cache)
        cache.root, cache.raw, cache.checked = self.root, (self.root / 'data/raw/opendart').resolve(), {}
        cache.index, cache.unlinked_original_files, cache.manifest_issues = defaultdict(list), [], []
        return cache

    def events(self, name):
        from src.preprocessing.dart_collect import read_events
        return [e for e in read_events(self.folder / 'manifest.jsonl') if e['event'] == name]

    def test_night_chain_shares_one_call_cap_and_never_deletes(self):
        import json
        from unittest.mock import patch
        from src.preprocessing.dart_collect import night
        from src.preprocessing.dart_collect import now
        self.prepare()
        calls = []

        def step(root, log, name, args):
            calls.append((name, args))
            if name == 'retry_failed':  # the retry run spends two calls of the night's cap
                with (self.folder / 'manifest.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(''.join(json.dumps({'event': 'request_started', 'at': now()}) + '\n' for _ in range(2)))
            return 0
        with patch('src.preprocessing.dart_collect._step', side_effect=step):
            result = night(self.root, self.folder, self.extract, max_calls=10, max_raw_gb=1, workers=1, retry_calls=3, not_before=None,
                           to_period='2014-12')
        self.assertEqual([name for name, _ in calls], ['retry_failed', 'collect', 'extract', 'report'])
        self.assertEqual(calls[1][1][-2:], ['--max-calls', '8'])
        self.assertTrue(all(args[args.index('--to-period') + 1] == '2014-12' for name, args in calls[:2]))
        self.assertIn('--retry-failed', calls[0][1])
        self.assertFalse(any('purge' in ' '.join(args) for _, args in calls))
        self.assertEqual(result['status'], 'night_finished')
        self.assertTrue((self.root / result['log']).exists())

    def collected(self, rows, responses):
        """history-collect, then a dart_equity-style extraction into its own folder; nothing deleted yet."""
        import json
        import pandas as pd
        from unittest.mock import patch
        from src.preprocessing.dart_collect import Collector, HISTORY_RAW
        from src.preprocessing.dart_extract import CODE_FINGERPRINT, extract_filing, load_jobs, write_tables
        cache = self.prepare()
        pd.DataFrame(rows).to_csv(self.folder / 'selection.csv', index=False)
        collector = Collector(self.root, self.folder, cache, 50, raw_parent=HISTORY_RAW)
        self.addCleanup(collector.session.close)
        with patch.object(collector.session, 'get', side_effect=responses), patch.object(collector, 'wait', return_value=True):
            collector.collect_history(rows, 'TESTKEY', self.identity, 10**9)
        self.extract = self.root / 'data/interim/dart_equity/history_vX'
        self.extract.mkdir(parents=True)
        jobs = load_jobs(self.root, [self.folder], source_runs=())[0]
        with (self.extract / 'extractions.jsonl').open('w', encoding='utf-8') as stream:
            stream.writelines(json.dumps(extract_filing(job), ensure_ascii=False) + '\n' for job in jobs)
        write_tables(self.extract, jobs)
        (self.extract / 'progress.json').write_text(json.dumps({'status': 'extraction_finished_not_PIT_verified',
                                                                'code_sha256': CODE_FINGERPRINT}), encoding='utf-8')
        (self.extract / 'collection.json').write_text(json.dumps({'source_runs': [self.folder.relative_to(self.root).as_posix()]}), encoding='utf-8')
        return cache

    def purge_rows(self):
        return self.rows(2) + self.rows(1, year=2004, sample=True) + self.rows(1, year=2015)

    def test_morning_report_counts_states_failures_and_extraction(self):
        import pandas as pd
        from unittest.mock import patch
        from src.preprocessing import dart_extract
        from src.preprocessing.dart_collect import Collector, HISTORY_RAW, history_report, report_markdown
        rows = self.rows(3)
        cache = self.prepare()
        pd.DataFrame(rows).to_csv(self.folder / 'selection.csv', index=False)
        collector = Collector(self.root, self.folder, cache, 10, raw_parent=HISTORY_RAW)
        self.addCleanup(collector.session.close)
        responses = [self.zipped(), Responses.response(status='014'), Responses.response(status='100')]
        with patch.object(collector.session, 'get', side_effect=responses), patch.object(collector, 'wait', return_value=True):
            collector.collect_history(rows, 'TESTKEY', self.identity, 10**9)
        argv = ['dart_extract', '--source-run', str(self.folder.relative_to(self.root)), '--run-dir',
                str(self.extract.relative_to(self.root)), '--workers', '1', '--execute']
        with patch('sys.argv', argv), patch('pathlib.Path.cwd', return_value=self.root):
            dart_extract.main()
        report = history_report(self.root, self.folder, self.extract)
        collection, extraction = report['collection'], report['extraction']
        self.assertEqual(collection['states'], {'original_verified': 1, 'absence_014': 1, 'endpoint_failed': 1})
        self.assertEqual([r['state'] for r in collection['failed']], ['endpoint_failed'])
        self.assertEqual((extraction['originals'], extraction['not_extracted_with_current_code']), (1, 0))
        self.assertEqual(extraction['by_year']['2003']['value'], 1)
        self.assertIn('| 2003 | 1 | 1 | 1 | 0 |', report_markdown(report))


class CompareExtractsTests(unittest.TestCase):
    """Version-to-version change table of two extraction folders."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_compare_extracts_names_each_kind_of_change(self):
        import json
        from src.preprocessing.dart_collect import compare_extracts
        old, new = self.root / 'v1', self.root / 'v2'
        decisions = {'a': ({'equity': '-5', 'equity_tier': 'T3'}, {'equity': '5', 'equity_tier': 'T3'}),
                     'b': ({'equity': None}, {'equity': '7', 'fiscal_period': '2003-12-31'}),
                     'c': ({'equity': '1', 'equity_tier': 'T3'}, {'equity': '1', 'equity_tier': 'T3'})}
        for folder, side in ((old, 0), (new, 1)):
            folder.mkdir()
            (folder / 'extractions.jsonl').write_text(''.join(json.dumps({'receipt_id': r, 'decision': d[side]}) + '\n'
                                                              for r, d in decisions.items()), encoding='utf-8')
        summary = compare_extracts(old, new)
        self.assertEqual(summary['changes'], {'sign_flipped_to_positive': 1, 'recovered:dated': 1, 'same': 1})
        self.assertTrue((new / summary['csv']).exists())
