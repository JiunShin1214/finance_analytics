"""Stage 3, KIND collection and extraction (on hold, kept as code; not connected to the other stages).
Merged 2026-11-04 from kind_history_probe, kind_original_collect and the KIND part of dart_history_rolling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime
import hashlib
import html
import json
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urljoin, urlsplit
from bs4 import BeautifulSoup
import requests
import calendar
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import gzip
import os
import shutil
import sqlite3
from urllib.parse import urlsplit
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from decimal import Decimal
import io
import subprocess
import sys
import zipfile
import pandas as pd

from src.preprocessing.dart_collect import (
    HISTORY_OUTPUT, KIND_REVIEW, LAST_ROLLING_PERIOD, _append, _map, infer_headers, within)
from src.preprocessing.dart_extract import OUTPUT as EXTRACT_OUTPUT
from src.preprocessing.dart_extract import (
    CODE_FINGERPRINT, EXTRACTOR_VERSION, MARCAP_REL, _amount, _current_columns, _decode_document, document_facts, load_jobs, read_events, select_filing, source_key)


BASE = 'https://kind.krx.co.kr'


VIEWER = BASE + '/common/disclsviewer.do'


PROBE_OUTPUT = Path('outputs/collection/KIND_HISTORY_PROBE')


DATES = ['2001-03-30', '2005-03-31', '2008-03-31', '2012-03-30']


UPSTREAM = 'https://github.com/beaten-by-the-market/krx-kind-data-api'


SOURCES = [UPSTREAM + '/blob/main_probe/krx_kind_data_api/' + name + '.py'
           for name in ('endpoints', 'parsers', 'transport')]


KINDS = ('사업보고서', '반기보고서', '분기보고서')


def compact(text):
    return re.sub(r'\s+', '', text)


def decode(body, content_type=''):
    meta = re.search(br'charset\s*=\s*["\']?([a-zA-Z0-9_-]+)', body[:4096])
    header = re.search(r'charset=([\w-]+)', content_type, re.I)
    choices = [meta[1].decode('ascii') if meta else '', header[1] if header else '', 'utf-8-sig', 'cp949']
    for encoding in dict.fromkeys(choices):
        if not encoding:
            continue
        try:
            return body.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError):
            pass
    return body.decode('utf-8', errors='replace'), 'utf-8-replacement_unconfirmed'


def search_form(day, page):
    # Preserve the reviewed full blank-field contract; avoid invented report category codes.
    blank = ('searchCodeType repIsuSrtCd allRepIsuSrtCd oldSearchCorpName disclosureType '
             'disTypevalue reportCd searchCorpName business marketType kosdaqSegment '
             'settlementMonth securities submitOblgNm enterprise reportNmTemp reportNmPop')
    form = dict.fromkeys(blank.split(), '')
    for category in ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '13', '14', '20'):
        form['disclosureType' + category] = ''
        form['pDisclosureType' + category] = ''
    form.update(method='searchDetailsSub', forward='details_sub', currentPageSize='100',
                pageIndex=str(page), orderMode='1', orderStat='A', bfrDsclsType='on',
                fromDate=day, toDate=day, reportNm='보고서', lastReport='')
    return form


def viewer_url(receipt):
    return VIEWER + '?' + urlencode({'method': 'search', 'acptno': receipt,
                                    'docno': '', 'viewerhost': '', 'viewerport': ''})


def kind_receipt_date(receipt):
    """Preserve numeric and historical date + letter + five digit KIND IDs."""
    if not isinstance(receipt, str) or not re.fullmatch(r'\d{8}(?:\d{6}|[A-Za-z]\d{5})', receipt):
        return None
    try:
        return date(int(receipt[:4]), int(receipt[4:6]), int(receipt[6:8])).isoformat()
    except ValueError:
        return None


def report_title_metadata(title):
    """Classify the actual title; substring search terms are not exclusive report types."""
    normalized = re.sub(r'^(?:\[[^\]]+\])+', '', compact(title))
    candidates = [kind for kind in KINDS if kind in normalized]
    canonical = bool(re.match(r'^(?:영문|국문)?(?:사업|반기|분기)보고서(?=\(|$|[ㆍ·,/])', normalized))
    ancillary = bool(re.match(r'^(?:관리종목|투자유의종목|기타시장안내|주권매매거래)', normalized))
    classification = ('regular' if canonical and len(candidates) == 1 else
                      'ambiguous_regular' if canonical and candidates else
                      'regular_related_unconfirmed' if candidates and not ancillary else 'non_regular')
    return {'report_kind': candidates[0] if classification == 'regular' else None,
            'report_kind_candidates': candidates, 'report_class': classification,
            'classification_policy': 2}


def parse_search(text, day):
    soup = BeautifulSoup(text, 'html.parser')
    table = soup.select_one('table.list')
    if table is None or table.find('tbody') is None:
        raise ValueError('search_table_or_tbody_missing')
    rows, skipped, placeholder = [], 0, False
    for tr in table.find('tbody').find_all('tr', recursive=False):
        cells = tr.find_all('td', recursive=False)
        if len(cells) == 1 and re.search(r'(조회|검색).*(없|0\s*건)', cells[0].get_text(' ', strip=True)):
            placeholder = True
            continue
        if not cells:
            continue
        if len(cells) < 5:
            skipped += 1
            continue
        anchor = cells[3].find('a')
        title = (anchor.get('title') or anchor.get_text(' ', strip=True)) if anchor else cells[3].get_text(' ', strip=True)
        visible_title = anchor.get_text(' ', strip=True) if anchor else title
        # Historical rows put the period in a nested font, outside the tooltip.
        label = re.search(r'\(\s*\d{4}[./-]\d{2}\s*\)', visible_title)
        if label and not re.search(r'\(\s*\d{4}[./-]\d{2}\s*\)', title):
            title += ' ' + label[0]
        match = re.search(r'openDisclsViewer\s*\(\s*["\']([^"\']+)["\']', str(anchor or cells[3]))
        receipt = match[1] if match else None
        company = cells[2].find('a')
        market_icon = cells[2].find('img', class_='legend')
        company_id = re.search(r'(?:companysummary_open|openCompanyInfoNew)\s*\(\s*["\']([A-Za-z0-9]+)', str(company or ''))
        raw_time = cells[1].get_text(' ', strip=True)
        explicit = re.search(r'\b(\d{4})[./-](\d{2})[./-](\d{2})\b', raw_time)
        received = '-'.join(explicit.groups()) if explicit else day
        prefix_matches = kind_receipt_date(receipt) == day if kind_receipt_date(receipt) else None
        report = report_title_metadata(title)
        rows.append({'company_name': company.get_text(' ', strip=True) if company else cells[2].get_text(' ', strip=True),
                     'kind_company_code': company_id[1] if company_id else None,
                     'market_label_observed': market_icon.get('alt') if market_icon else None,
                     'report_name': title, 'report_name_visible': visible_title,
                     **report, 'receipt_no': receipt,
                     'receipt_date': received, 'receipt_date_basis': 'explicit_list_cell' if explicit else 'single_day_query_only',
                     'receipt_time_raw': raw_time, 'search_date': day, 'receipt_prefix_matches_query': prefix_matches,
                     'revision_mark_observed': '정정' in title, 'attachment_addition_observed': '첨부추가' in title,
                     'initial_version_confirmed': False,
                     'version_note': 'Title marks only; absence of a correction mark does not prove first version.',
                     'viewer_url': viewer_url(receipt) if receipt else None, 'actual_html_url': None})
    total_match = re.search(r'(?:총|전체)\s*[:：]?\s*([\d,]+)\s*건', soup.get_text(' ', strip=True))
    count_input = soup.find('input', attrs={'name': 'totalCount'})
    total = int(total_match[1].replace(',', '')) if total_match else None
    if total is None and count_input and re.fullmatch(r'\d+', str(count_input.get('value', ''))):
        total = int(count_input['value'])
    if not rows and not placeholder and total != 0:
        raise ValueError('empty_structure_not_a_confirmed_empty_result')
    paging = soup.select_one('.paging-group .info')
    page_match = re.search(r':\s*(\d+)\s*/\s*(\d+)', paging.get_text(' ', strip=True)) if paging else None
    size = soup.select_one('select[name="currentPageSize"] option[selected]')
    return rows, {'total_reported': total, 'unparsed_rows': skipped,
                  'explicit_empty_placeholder': placeholder,
                  'current_page': int(page_match[1]) if page_match else None,
                  'total_pages': int(page_match[2]) if page_match else None,
                  'page_size': int(size.get('value')) if size and str(size.get('value', '')).isdigit() else None}


def content_options(shell):
    soup = BeautifulSoup(shell, 'html.parser')
    result = []
    for option in soup.find_all('option'):
        match = re.fullmatch(r'([^|]+)\|([YN])', str(option.get('value', '')))
        if match and kind_receipt_date(match[1]) and (match[1], match[2]) not in [(v['content_id'], v['flag']) for v in result]:
            result.append({'content_id': match[1], 'flag': match[2], 'label': option.get_text(' ', strip=True)})
    return result


def html_urls(text):
    # URLs must actually appear in the server response; never guess docno or external paths.
    normalized = html.unescape(text.replace('\\/', '/'))
    values = re.findall(r'''(?:https?://[^\s"'<>]+|/external/[^\s"'<>]+)\.html?(?:\?[^\s"'<>]*)?''', normalized, re.I)
    for node in BeautifulSoup(normalized, 'html.parser').select('iframe[src],frame[src],a[href]'):
        value = node.get('src') or node.get('href')
        if re.search(r'\.html?(?:\?|$)', value or '', re.I):
            values.append(urljoin(VIEWER, value))
    return list(dict.fromkeys(urljoin(BASE, v) for v in values
                             if not re.search(r'/blank\.html?(?:\?|$)', v, re.I)))


def select_body_url(urls, receipt, content_id=None, response_text=''):
    """Select an observed body, never manufacture a URL by removing _toc."""
    prefix = f'/external/{receipt[:4]}/{receipt[4:6]}/{receipt[6:8]}/{receipt[8:]}/' if kind_receipt_date(receipt) else None
    evidence, bodies = [], []
    for url in urls:
        parsed = urlsplit(url)
        toc = bool(re.search(r'_toc\.html?$', parsed.path, re.I))
        receipt_match = bool(prefix and parsed.path.startswith(prefix))
        cid_match = parsed.path.startswith(prefix + content_id + '/') if prefix and content_id else None
        eligible = (not toc and parsed.hostname == 'kind.krx.co.kr' and receipt_match and cid_match is not False)
        evidence.append({'url': url, 'role': 'toc' if toc else 'body_candidate',
                         'receipt_path_matches': receipt_match, 'content_id_path_matches': cid_match,
                         'eligible_body': eligible})
        if eligible:
            bodies.append(url)
    # KIND explicitly passes (toc, body, ...) to setPath. Use the second argument
    # only when it is also an independently observed, correctly bound body candidate.
    paths = re.findall(r'''(?:parent\.)?setPath\s*\(\s*(['"])(.*?)\1\s*,\s*(['"])(.*?)\3''',
                       html.unescape(response_text.replace('\\/', '/')), re.S)
    declared = list(dict.fromkeys(urljoin(BASE, m[3]) for m in paths))
    preferred = [url for url in declared if url in bodies]
    if len(preferred) == 1:
        return preferred[0], evidence, 'setPath_body_argument_and_receipt_content_id_paths'
    if len(bodies) == 1:
        return bodies[0], evidence, 'unique_non_TOC_bound_body'
    reason = ('body_url_missing' if not urls else 'TOC_only_no_body_url' if all(e['role'] == 'toc' for e in evidence)
              else 'body_url_receipt_or_content_id_unconfirmed' if not bodies else 'multiple_bound_body_urls')
    return None, evidence, reason


def classify_html(text, row, url, content_id=None):
    soup = BeautifulSoup(text, 'html.parser')
    for node in soup(['script', 'style', 'noscript']):
        node.decompose()
    visible = (soup.body or soup).get_text(' ', strip=True)
    kinds = [row['report_kind']] if row.get('report_kind') else row.get('report_kind_candidates', [])
    heading = any(kind in compact(visible) for kind in kinds)
    shell = bool(soup.find(['iframe', 'frame', 'frameset']) or soup.find('form', attrs={'name': 'docpathfrm'}))
    error = bool(re.search(r'(접근.*(?:차단|제한)|오류가 발생|페이지를 찾을 수|잘못된 접근|요청하신.*없)', visible[:300]))
    markup_ok = bool(soup.find(['html', 'body', 'table']) and visible and not shell and not error)
    section = bool(re.search(r'(재무제표|재무상태표|대차대조표|사업의내용|회사의개요)', compact(visible)))
    report_material = (heading and (bool(soup.find('table')) or section)) or (section and bool(soup.find('table')))
    receipt = row['receipt_no'] or ''
    expected = f'/external/{receipt[:4]}/{receipt[4:6]}/{receipt[6:8]}/{receipt[8:]}/' if kind_receipt_date(receipt) else None
    path = urlsplit(url).path
    receipt_match = path.startswith(expected) if expected else None
    cid_match = path.startswith(expected + content_id + '/') if expected and content_id else None
    toc = bool(re.search(r'_toc\.html?$', path, re.I))
    html_ok = bool(markup_ok and report_material and not toc and receipt_match is not False and cid_match is not False)
    return {'html_original_accessible': html_ok,
            'html_response_received': markup_ok,
            'document_status': ('html_accessible' if html_ok else 'TOC_not_original' if toc else
                                'receipt_or_content_id_mismatch' if receipt_match is False or cid_match is False else 'viewer_shell' if shell else
                                'html_without_report_evidence' if markup_ok else 'error_or_non_document'),
            'report_heading_observed': heading,
            'company_name_observed': compact(row['company_name']) in compact(visible) if row['company_name'] else False,
            'receipt_path_matches': receipt_match, 'content_id_path_matches': cid_match,
            'version_identity_verified': False,
            'identity_note': 'HTTP/HTML access and URL path are evidence only; first/corrected version identity is not independently verified.'}


class ProbeStop(RuntimeError):
    pass


class Client:
    def __init__(self, folder, max_calls, *, session=None, interval=1.3):
        self.folder, self.max_calls, self.calls = folder, max_calls, 0
        self.session = session or requests.Session()
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=0))
        self.session.mount('http://', requests.adapters.HTTPAdapter(max_retries=0))
        self.interval, self.last_request = interval, 0.0
        self.records = []

    def log(self, record):
        record = {'at': datetime.now().astimezone().isoformat(), **record}
        with (self.folder / 'requests.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.records.append(record)

    def fetch(self, method, url, *, data=None, label='request'):
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or parsed.hostname != 'kind.krx.co.kr' or parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
            raise ValueError('external_or_invalid_url_not_requested')
        for retry in range(2):
            if self.calls >= self.max_calls:
                raise ProbeStop('call_cap_reached')
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.calls += 1
            number = self.calls
            self.last_request = time.monotonic()
            self.log({'event': 'request_started', 'number': number, 'method': method, 'url': url,
                      'params': data, 'retry': retry, 'label': label})
            try:
                response = self.session.request(method, url, data=data, timeout=(15, 40), stream=True,
                    allow_redirects=False, headers={'User-Agent': 'Mozilla/5.0', 'Referer': BASE + '/'})
            except requests.RequestException as error:
                self.log({'event': 'transport_error', 'number': number, 'error_type': type(error).__name__, 'retry': retry})
                if retry == 0:
                    time.sleep(2)
                    continue
                raise ValueError('transport_failed_twice') from None
            body = bytearray()
            try:
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body) > 8 * 1024**2:
                        break
            except requests.RequestException as error:
                self.log({'event': 'response_read_error', 'number': number, 'error_type': type(error).__name__})
                raise ValueError('incomplete_response') from None
            finally:
                response.close()
            raw = bytes(body)
            text, encoding = decode(raw, response.headers.get('Content-Type', ''))
            path = self.folder / f'{number:03d}_{label}.html'
            with path.open('xb') as stream:
                stream.write(raw)
            self.log({'event': 'response', 'number': number, 'http': response.status_code, 'retry': retry,
                      'path': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                      'encoding': encoding, 'oversized_truncated': len(raw) > 8 * 1024**2})
            if response.status_code in {403, 429}:
                raise ProbeStop('blocked_or_rate_limited_no_bypass')
            if len(raw) > 8 * 1024**2:
                raise ValueError('response_exceeds_8MiB_not_complete')
            if 500 <= response.status_code <= 599 and retry == 0:
                time.sleep(2)
                continue
            if response.status_code != 200:
                raise ValueError('http_' + str(response.status_code) + '_redirects_not_followed')
            if encoding == 'utf-8-replacement_unconfirmed':
                raise ValueError('response_encoding_unconfirmed')
            return text
        raise ValueError('request_failed')


def inspect_document(client, row):
    result = {**row, 'html_original_accessible': False, 'version_identity_verified': False,
              'document_status': 'not_attempted', 'content_options': [], 'resolver_attempts': [],
              'failure_stage': None, 'failure_reason': None}

    def failure(stage, reason):
        result.update(document_status=reason, failure_stage=stage, failure_reason=reason)
        client.log({'event': 'document_failure', 'receipt_no': row['receipt_no'],
                    'content_id': result.get('selected_content_id', {}).get('content_id'),
                    'stage': stage, 'request_phase': phase, 'reason': reason,
                    'actual_html_url': result.get('actual_html_url')})
        return result

    phase = 'viewer'
    if not row['receipt_no']:
        return failure('url_resolution', 'receipt_number_missing')
    if row['receipt_prefix_matches_query'] is False:
        return failure('url_resolution', 'search_date_scope_mismatch')
    try:
        shell = client.fetch('GET', row['viewer_url'], label='viewer_' + row['receipt_no'])
        options = content_options(shell)
        result['content_options'] = options
        current = [v for v in options if v['flag'] == 'Y']
        if len(current) != 1:
            # Legacy shells may already expose one actual iframe link, without options.
            urls = html_urls(shell)
            result['original_url_source'] = 'explicit_viewer_link_version_unconfirmed'
            resolved = shell
            if len(current) > 1:
                return failure('url_resolution', 'current_content_id_ambiguous')
        else:
            chosen = current[0]
            result['selected_content_id'] = chosen
            phase = 'resolver'
            resolved = client.fetch('POST', VIEWER, data={'method': 'searchContents', 'docNo': chosen['content_id']},
                                    label='resolve_' + row['receipt_no'])
            urls = html_urls(resolved)
            result['resolver_attempts'].append({'content_id': chosen['content_id'], 'returned_urls': urls})
            result['original_url_source'] = 'searchContents_response'
        result['resolved_html_urls'] = urls
        cid = result.get('selected_content_id', {}).get('content_id')
        selected, evidence, basis = select_body_url(urls, row['receipt_no'], cid, resolved)
        result.update(html_url_candidates=evidence, body_url_selection_basis=basis)
        if selected is None:
            return failure('url_resolution', basis)
        result['actual_html_url'] = selected
        client.log({'event': 'body_url_selected', 'receipt_no': row['receipt_no'], 'content_id': cid,
                    'url': selected, 'basis': basis, 'candidates': evidence})
        phase = 'original'
        page = client.fetch('GET', selected, label='original_' + row['receipt_no'])
        result.update(classify_html(page, row, selected, cid))
        if not result['html_original_accessible']:
            return failure('body_validation', result['document_status'])
        client.log({'event': 'body_validation', 'receipt_no': row['receipt_no'], 'content_id': cid,
                    'url': selected, 'html_original_accessible': True,
                    'version_identity_verified': False})
    except ProbeStop as error:
        failure('http_access', str(error))
        result['stop_requested'] = True
    except ValueError as error:
        failure('http_access', str(error))
    return result


def probe(client, dates, max_pages, max_documents):
    searches, candidates = [], []
    stopped = False
    for day in dates:
        entry = {'search_date': day, 'lastReport': '', 'pages': [], 'search_status': 'not_attempted',
                 'list_search_succeeded': False, 'pagination_complete': False, 'all_versions_verified': False}
        seen, previous, reported = set(), None, None
        partial_parse = False
        for page in range(1, max_pages + 1):
            try:
                text = client.fetch('POST', BASE + '/disclosure/details.do', data=search_form(day, page),
                                    label=f'list_{day}_{page}')
                rows, metadata = parse_search(text, day)
                fingerprint = tuple(r['receipt_no'] or r['report_name'] for r in rows)
                entry['pages'].append({'page': page, 'rows': len(rows), **metadata})
                if previous is not None and rows and fingerprint == previous:
                    raise ValueError('repeated_page_pagination_unconfirmed')
                previous = fingerprint
                if any(r['receipt_prefix_matches_query'] is False or r['receipt_date'] != day for r in rows):
                    raise ValueError('returned_receipt_date_outside_requested_day')
                entry['list_search_succeeded'] = True
                partial_parse |= bool(metadata['unparsed_rows'])
                if metadata['unparsed_rows']:
                    entry['search_status'] = 'partial_parse'
                else:
                    entry['search_status'] = 'list_parsed' if rows else 'empty_result_unconfirmed_not_evidence_of_absence'
                for row in rows:
                    key = row['receipt_no'] or (row['company_name'], row['report_name'], row['receipt_time_raw'])
                    if key not in seen:
                        seen.add(key)
                        if row['report_kind']:
                            candidates.append(row)
                total = metadata['total_reported']
                if reported is not None and total is not None and total != reported:
                    raise ValueError('total_count_changed_between_pages')
                reported = total if total is not None else reported
                if total is not None and len(seen) > total:
                    raise ValueError('total_count_inconsistent_with_rows')
                if total is not None and len(seen) == total and not partial_parse:
                    entry['pagination_complete'] = True
                    break
                if not rows:
                    break
            except ProbeStop as error:
                entry.update(search_status=str(error), stop_requested=True)
                stopped = True
                break
            except ValueError as error:
                entry.update(search_status='search_or_parse_failed', failure_reason=str(error))
                break
        entry['unique_list_rows'] = len(seen)
        entry['note'] = 'Zero results do not prove historical absence; only specified days/pages were tested.'
        searches.append(entry)
        if stopped:
            break
    # Round-robin dates, choosing a correction as well as an unmarked candidate where observed.
    selected = []
    for revision in (True, False):
        for day in dates:
            row = next((r for r in candidates if r['search_date'] == day and r['revision_mark_observed'] == revision
                        and not r['attachment_addition_observed']), None)
            if row and row not in selected and len(selected) < max_documents:
                selected.append(row)
    documents = []
    if not stopped:
        for row in selected:
            result = inspect_document(client, row)
            documents.append(result)
            if result.get('stop_requested'):
                break
    return {'searches': searches, 'filings_observed': candidates, 'documents': documents,
            'network_attempts': client.calls, 'retry_attempts': sum(r['event'] == 'request_started' and r['retry'] > 0 for r in client.records),
            'scope': 'Specified submission dates only, not fiscal years; not a 2001-2014 coverage estimate or PIT verification.',
            'source_review': {'urls': SOURCES, 'reviewed': '2026-11-01', 'revision': 'main_probe; immutable commit not verified'},
            'report_kinds_observed': sorted({r['report_kind'] for r in candidates}),
            'correction_titles_observed': sum(r['revision_mark_observed'] for r in candidates),
            'original_identity_verified_count': 0,
            'version_warning': 'lastReport empty requests all versions; observed correction marks do not establish version completeness or relation.'}


RESEARCH_OUTPUT = Path('outputs/collection/KIND_RESEARCH_AUDIT')


STRATA = {'2001-2005': (2001, 2005, 6), '2006-2010': (2006, 2010, 6), '2011-2014': (2011, 2014, 8)}


OWNERS = {'지배기업소유주지분', '지배기업의소유주지분',
          '지배기업의소유주에게귀속되는자본', '지배기업소유주에게귀속되는자본'}


DIAGNOSTIC = {'자본총계': 'equity_total', '자본합계': 'equity_total', '비지배지분': 'noncontrolling_interest',
              '비지배주주지분': 'noncontrolling_interest', '소수주주지분': 'minority_interest_legacy'}


DATE_PART = r'(\d{4})\s*[년./-]\s*(\d{1,2})\s*[월./-]\s*(\d{1,2})\s*일?'


def research_windows():
    """Fixed, declared submission windows, not adaptive expansion or a census."""
    return [{'from_date': f'{year}-{month}-{start}', 'to_date': f'{year}-{month}-{end}', 'kind': kind}
            for year in (2001, 2003, 2005, 2006, 2008, 2010, 2011, 2012, 2014)
            for month, start, end, kind in (('03', '20', '31', '사업보고서'),
                                          ('05', '10', '20', '분기보고서'),
                                          ('08', '10', '20', '반기보고서'))]


def stratum_of(row):
    year = int(row['receipt_date'][:4])
    return next((key for key, (start, end, _) in STRATA.items() if start <= year <= end), None)


def period_label(title):
    found = re.search(r'\((\d{4})[.-](\d{2})\)', title)
    return '-'.join(found.groups()) if found and 1 <= int(found[2]) <= 12 else None


def pair_key(row):
    # A company-name similarity is never an identifier or proof of version relation.
    label = period_label(row['report_name'])
    return (row['kind_company_code'], row['report_kind'], label) if row.get('kind_company_code') and label else None


def ordered_samples(rows, quota):
    """Prefer an observed candidate pair, then report-kind/company diversity; no synthetic rows."""
    pending = sorted(rows, key=lambda r: (r['receipt_date'], r['receipt_no'] or ''))
    selected = []
    for row in pending:
        key = pair_key(row)
        if key and not row['revision_mark_observed'] and not row['attachment_addition_observed']:
            revision = next((r for r in pending if pair_key(r) == key and r['revision_mark_observed']
                             and r['receipt_no'] != row['receipt_no']
                             and (r['receipt_date'], r['receipt_no']) > (row['receipt_date'], row['receipt_no'])), None)
            if revision:
                selected.extend([row, revision])
                break
    while len(selected) < min(len(pending), quota + 2):
        rest = [r for r in pending if r not in selected]
        if not rest:
            break
        kinds = Counter(r['report_kind'] for r in selected)
        companies = Counter(r.get('kind_company_code') or r['company_name'] for r in selected)
        row = min(rest, key=lambda r: (kinds[r['report_kind']], companies[r.get('kind_company_code') or r['company_name']],
                                      r['receipt_date'], r['receipt_no'] or ''))
        selected.append(row)
    return selected


def search_research(client, windows, max_pages):
    searches, candidates, known = [], [], set()
    for window in windows:
        record = {**window, 'pages': [], 'pagination_complete': False, 'status': 'not_attempted'}
        seen, previous, stopped, damaged, reported = set(), None, False, False, None
        for page in range(1, max_pages + 1):
            form = search_form(window['to_date'], page)
            form.update(fromDate=window['from_date'], reportNm=window['kind'], marketType='1')
            try:
                raw = client.fetch('POST', BASE + '/disclosure/details.do', data=form,
                                   label=f'research_list_{window["to_date"]}_{page}')
                rows, metadata = parse_search(raw, window['to_date'])
                record['pages'].append({'page': page, **metadata, 'rows': len(rows)})
                fingerprint = tuple(r['receipt_no'] or r['report_name'] for r in rows)
                if rows and previous == fingerprint:
                    raise ValueError('repeated_page')
                previous = fingerprint
                damaged |= bool(metadata['unparsed_rows'])
                for row in rows:
                    receipt = row['receipt_no']
                    if row['receipt_date_basis'] == 'explicit_list_cell' and receipt and len(receipt) == 14:
                        prefix = date(int(receipt[:4]), int(receipt[4:6]), int(receipt[6:8])).isoformat()
                        if prefix != row['receipt_date']:
                            damaged = True
                            continue
                    if row['receipt_date_basis'] != 'explicit_list_cell':
                        if not receipt or len(receipt) != 14:
                            damaged = True
                            continue
                        row['receipt_date'] = date(int(receipt[:4]), int(receipt[4:6]), int(receipt[6:8])).isoformat()
                        row['receipt_date_basis'] = 'receipt_identifier_prefix_not_independently_verified'
                    in_window = window['from_date'] <= row['receipt_date'] <= window['to_date']
                    if not in_window:
                        damaged = True
                        continue
                    row.update(receipt_prefix_matches_query=True, search_window=window,
                               sampling_stratum=stratum_of(row), stratum_basis='submission_year_not_fiscal_year')
                    seen.add(receipt or (row['company_name'], row['report_name']))
                    if receipt and receipt not in known and row['report_kind'] in KINDS:
                        known.add(receipt)
                        candidates.append(row)
                record['status'] = 'list_parsed' if rows else 'empty_unconfirmed_not_absence'
                total = metadata['total_reported']
                if total is not None and reported is not None and total != reported:
                    raise ValueError('total_count_changed_between_pages')
                reported = total if total is not None else reported
                if total is not None and len(seen) > total:
                    raise ValueError('total_count_inconsistent_with_rows')
                if total is not None and len(seen) == total and not damaged:
                    record['pagination_complete'] = True
                    break
                if not rows:
                    break
            except ProbeStop as error:
                record['status'], stopped = str(error), True
                break
            except ValueError as error:
                record.update(status='search_failed_or_scope_unconfirmed', reason=str(error))
                break
        record['unique_rows'] = len(seen)
        record['scope_or_parse_unconfirmed'] = damaged
        searches.append(record)
        if stopped:
            break
    return searches, candidates


def statement_context(table):
    pieces = []
    statement = r'(재무상태표|대차대조표|손익계산서|포괄손익계산서|현금흐름표|자본변동표|이익잉여금처분계산서)'
    caption = table.find('caption', recursive=False)
    if caption:
        pieces.append(caption.get_text(' ', strip=True))
        if re.search(statement, compact(pieces[0])):
            text = pieces[0]
            return text, ('CFS' if re.search(r'연결(?:재무상태표|대차대조표)', compact(text)) else
                          'OFS' if re.search(r'(별도|개별)(?:재무상태표|대차대조표)', compact(text)) else 'UNKNOWN')
    for node in table.previous_siblings:
        if not getattr(node, 'name', None):
            continue
        text = node.get_text(' ', strip=True)
        if node.name == 'table':
            # KIND places statement title/dates/units in a separate borderless table.
            # A previous numeric statement is a boundary, never heading context.
            monetary_cells = any(re.fullmatch(r'[()△▲+\-]?[\d,]+(?:\.\d+)?[)]?', c.get_text(strip=True))
                                 and ',' in c.get_text() for c in node.find_all(['td', 'th']))
            heading_metadata = re.search(statement, compact(text)) or re.search(r'\d{4}[년./-]\d{1,2}|단위[:：]', compact(text))
            if not (heading_metadata and len(text) <= 1200 and not monetary_cells):
                break
        elif node.find('table'):
            break
        if not text or text in pieces:
            continue
        pieces.append(text)
        # A nearer income/CF statement must not inherit an earlier balance-sheet title.
        if re.search(statement, compact(text)) or len(pieces) >= 12:
            break
    context = ' '.join(reversed(pieces))
    norm = compact(context)
    if re.search(r'연결(?:재무상태표|대차대조표)', norm):
        scope = 'CFS'
    elif re.search(r'(개별|별도)(?:재무상태표|대차대조표)', norm):
        scope = 'OFS'
    else:
        scope = 'UNKNOWN'  # absence of 연결 is not proof of OFS
    return context, scope


def cover_period(text, title):
    ranges = set()
    for match in re.finditer(r'사업\s*연도\s*[:：]?\s*' + DATE_PART + r'\s*(?:부터|~|∼)\s*' + DATE_PART, text):
        try:
            start = date(*map(int, match.groups()[:3])).isoformat()
            end = date(*map(int, match.groups()[3:])).isoformat()
            if start <= end and (not period_label(title) or end[:7] == period_label(title)):
                ranges.add((start, end))
        except ValueError:
            continue
    return next(iter(ranges)) if len(ranges) == 1 else (None, None)


def unit_evidence(context):
    found = re.findall(r'단위[:：]?(백만|천)?(원|USD|미달러|달러)', compact(context), re.I)
    values = {({'': 1, '천': 1000, '백만': 1000000}[scale], 'KRW' if currency == '원' else 'USD')
              for scale, currency in found}
    return next(iter(values)) if len(values) == 1 else (None, None)


def table_adapter(table):
    """Adapt legacy leading dated rows to DART's thead helper in memory only."""
    clone = BeautifulSoup(str(table), 'html.parser').find('table')
    if clone.find('thead', recursive=False):
        return clone, 'native_thead'
    head = BeautifulSoup('<thead></thead>', 'html.parser').thead
    for tr in list(clone.find_all('tr')):
        if tr.find_parent('table') is not clone:
            continue
        normalized = compact(tr.get_text(' ', strip=True))
        if not (re.search(r'\d{4}[년./-]\d{1,2}|제\d+기|당기|전기|과목|계정과목', normalized)
                or (tr.find('th') and not tr.find('td'))):
            break
        head.append(tr.extract())
    if head.find('tr'):
        clone.insert(0, head)
    return clone, 'legacy_header_promoted_in_memory'


def extract_financial_html(text, row, provenance):
    # Only pure helpers are imported. Never instantiate or alter the DART collector.
    from src.preprocessing.dart_extract import _amount, _current_columns
    soup = BeautifulSoup(text, 'html.parser')
    flat = soup.get_text(' ', strip=True)
    norm = compact(flat)
    ifrs = bool(re.search(r'한국채택국제회계기준|K-?IFRS', norm, re.I))
    kgaap = bool(re.search(r'일반기업회계기준|기업회계기준서|기업회계기준에따라', norm))
    basis = 'mixed_mentions_unconfirmed' if ifrs and kgaap else 'IFRS_label_observed' if ifrs else 'KGAAP_label_observed' if kgaap else 'basis_unconfirmed'
    company = soup.find('company-name')
    body_code = company.get('aregcik') if company else None
    start, end = cover_period(flat, row['report_name'])
    cover_source = 'explicit_cover' if end else 'unconfirmed'
    if not end and body_code:
        from src.preprocessing.dart_collect import infer_headers
        metadata = infer_headers(row['report_name'], [text], [], body_code)
        start, end = metadata.get('period_start'), metadata.get('period_end')
        if end:
            cover_source = 'DART_header_helper_explicit_body_metadata'
    receipt_ids = set(re.findall(r'접수\s*번호\s*[:：]?\s*(\d{14})', flat))
    for item in soup.find_all('input'):
        if (item.get('name') or '').lower() in {'acptno', 'rcept_no', 'rceptno'} and re.fullmatch(r'\d{14}', item.get('value', '')):
            receipt_ids.add(item['value'])
    body_dates = set()
    for match in re.finditer(r'(?:접수|제출)\s*일\s*[:：]?\s*' + DATE_PART, flat):
        try:
            body_dates.add(date(*map(int, match.groups())).isoformat())
        except ValueError:
            pass
    identity_conflict = bool(receipt_ids and receipt_ids != {row['receipt_no']})
    month = end[:7] if end else period_label(row['report_name'])
    common = {**provenance, 'sampling_stratum': row['sampling_stratum'], 'company_name': row['company_name'],
              'kind_company_code': row.get('kind_company_code'), 'corp_code_from_body': body_code,
              'body_company_name': company.get_text(' ', strip=True) if company else None,
              'company_identity_status': 'observed_identifiers_not_historical_mapping_verified',
              'body_receipt_ids_observed': json.dumps(sorted(receipt_ids)),
              'body_submission_dates_observed': json.dumps(sorted(body_dates)),
              'body_receipt_identity_conflict': identity_conflict,
              'receipt_no': row['receipt_no'], 'receipt_date': row['receipt_date'],
              'receipt_date_basis': row['receipt_date_basis'], 'report_name': row['report_name'],
              'report_kind': row['report_kind'], 'period_start': start, 'period_end': end,
              'period_basis': cover_source, 'accounting_basis': basis,
              'standard_account_id': None, 'manual_verified': False, 'research_usable': False}
    results = []
    for ti, table in enumerate(soup.find_all('table')):
        context, scope = statement_context(table)
        if not re.search(r'재무상태표|대차대조표', compact(context)):
            continue
        adapted, adaptation = table_adapter(table)
        head = adapted.find('thead', recursive=False)
        header_text = head.get_text(' ', strip=True) if head else ''
        local = context + ' ' + header_text
        # Units are often printed between caption/date heading and first data row.
        local += ' ' + ' '.join(t.get_text(' ', strip=True) for t in table.find_all(['p', 'caption'], recursive=False))
        scale, currency = unit_evidence(local)
        try:
            columns, table_end = _current_columns(adapted, local, month) if month else ([], None)
        except (ValueError, TypeError):
            columns, table_end = [], None
        end_conflict = bool(end and table_end and end != table_end)
        if end_conflict:
            columns = []
        for ri, tr in enumerate(table.find_all('tr')):
            if tr.find_parent('table') is not table:
                continue
            cells = tr.find_all(['td', 'th'], recursive=False)
            if len(cells) < 2:
                continue
            label = cells[0].get_text(' ', strip=True)
            key = re.sub(r'^[\d.ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ\s]+', '', compact(label))
            category = 'owners_equity' if key in OWNERS else DIAGNOSTIC.get(key)
            if category is None:
                continue
            record = {**common, 'statement_scope': scope, 'statement_heading': context,
                      'account_name_original': label, 'account_category': category,
                      'table_index': ti, 'row_index': ri, 'column_indexes': json.dumps(columns),
                      'header_text': header_text, 'row_text': tr.get_text(' | ', strip=True),
                      'unit_evidence': local, 'display_scale': scale, 'currency': currency,
                      'amount_raw': None, 'amount_normalized': None, 'header_adapter': adaptation,
                      'statement_period_end': table_end, 'target_automatic_candidate': False,
                      'reused_helpers': '_amount,_current_columns',
                      'extraction_status': 'cover_statement_end_conflict' if end_conflict else 'column_or_period_unconfirmed'}
            if not end and table_end:
                record.update(period_end=table_end, period_basis='statement_dated_header_only')
            eligible = category == 'owners_equity' and scope == 'CFS'
            period_year = (end or month or row['receipt_date'])[:4]
            pre_ifrs = int(period_year) < 2011 and basis != 'IFRS_label_observed'
            record['definition_status'] = ('pre2011_definition_unconfirmed' if eligible and pre_ifrs else
                                           'owners_label_candidate_definition_unverified' if eligible else 'diagnostic_not_target')
            expanded = []
            body_spans = any(int(c.get('rowspan', 1)) != 1 for tr0 in adapted.find_all('tr')
                             if tr0.find_parent('table') is adapted and not tr0.find_parent('thead')
                             for c in tr0.find_all(['td', 'th'], recursive=False))
            if body_spans:
                record['extraction_status'] = 'body_rowspan_unconfirmed'
                results.append(record)
                continue
            for ci, cell in enumerate(cells):
                width = int(cell.get('colspan', 1))
                if not 1 <= width <= 100:
                    raise ValueError('body_colspan_out_of_bounds')
                expanded.extend([(ci, cell)] * width)
            if columns:
                selected = {ci: cell for col in columns if col < len(expanded) for ci, cell in [expanded[col]]}
                nonempty = [(ci, c.get_text(' ', strip=True)) for ci, c in selected.items()
                            if c.get_text(' ', strip=True) not in {'', '\\', '₩', 'KRW', 'USD', '$'}]
                if not nonempty:
                    record['extraction_status'] = 'current_amount_blank'
                elif len(nonempty) != 1:
                    record['extraction_status'] = 'multiple_current_amount_cells_unconfirmed'
                else:
                    ci, raw = nonempty[0]
                    record.update(amount_raw=raw, physical_column_index=ci)
                    if scale is None:
                        record['extraction_status'] = 'unit_unconfirmed'
                    else:
                        amount = _amount(raw, scale)
                        record.update(amount_normalized=amount, extraction_status='amount_extracted' if amount is not None else 'numeric_parse_or_display_dash_unconfirmed')
            record['target_automatic_candidate'] = bool(eligible and basis == 'IFRS_label_observed' and
                not identity_conflict and record['amount_normalized'] is not None and record.get('period_start') and record.get('period_end'))
            results.append(record)
    if not results:
        observed = any(label in norm for label in OWNERS)
        results.append({**common, 'account_category': 'unconfirmed', 'target_automatic_candidate': False,
                        'extraction_status': 'target_label_observed_but_not_extracted' if observed else 'target_account_not_observed_in_downloaded_html',
                        'definition_status': 'pre2011_definition_unconfirmed' if int(row['receipt_date'][:4]) < 2011 else 'definition_unconfirmed'})
    return results


def financial_section_links(text, base_url):
    links = []
    for anchor in BeautifulSoup(text, 'html.parser').find_all('a', href=True):
        label = compact(anchor.get_text(' ', strip=True))
        url = urljoin(base_url, anchor['href'])
        if (re.search(r'재무제표|재무상태표|대차대조표', label) and re.search(r'\.html?(?:\?|$)', url)
                and urlsplit(url).hostname == 'kind.krx.co.kr'
                and urlsplit(url)._replace(fragment='') != urlsplit(base_url)._replace(fragment='')):
            links.append((0 if '연결' in label else 1, url))
    return list(dict.fromkeys(url for _, url in sorted(links)))


def research_audit(client, folder, max_pages):
    (folder / 'sources.jsonl').touch(exist_ok=False)
    searches, pool = search_research(client, research_windows(), max_pages)
    samples, facts, paired = [], [], []
    halted = any(s['status'] in {'call_cap_reached', 'blocked_or_rate_limited_no_bypass'} for s in searches)
    if not halted:
        for stratum, (_, _, quota) in STRATA.items():
            obtained = 0
            available = [r for r in pool if r['sampling_stratum'] == stratum]
            for row in ordered_samples(available, quota):
                result = inspect_document(client, row)
                original_record = next((r for r in reversed(client.records) if r['event'] == 'response'
                                        and any(e['event'] == 'request_started' and e['number'] == r['number']
                                                and e.get('label') == 'original_' + row['receipt_no'] for e in client.records)), None)
                source = {'source_url': result.get('actual_html_url'), 'source_path': original_record['path'] if original_record else None,
                          'source_sha256': original_record['sha256'] if original_record else None}
                result.update(source, manual_verified=False, research_usable=False)
                # URL/receipt mismatch remains a failure; it must not satisfy the 20-document target.
                acquired = bool(result['html_original_accessible'] and result.get('receipt_path_matches') is not False)
                result['sample_original_acquired'] = acquired
                if acquired and original_record:
                    obtained += 1
                    try:
                        raw = (folder / original_record['path']).read_bytes()
                        if hashlib.sha256(raw).hexdigest() != source['source_sha256']:
                            raise ValueError('source_hash_mismatch')
                        extracted = extract_financial_html(decode(raw)[0], row, source)
                        section_urls = financial_section_links(decode(raw)[0], source['source_url'])
                        result['financial_section_links_observed'] = section_urls
                        # One explicit financial-section link per receipt at most, never a guessed URL.
                        if section_urls and not any(r.get('target_automatic_candidate') for r in extracted):
                            section_url = section_urls[0]
                            try:
                                section = client.fetch('GET', section_url, label='financial_' + row['receipt_no'])
                                response = next(r for r in reversed(client.records) if r['event'] == 'response')
                                section_source = {'source_url': section_url, 'source_path': response['path'], 'source_sha256': response['sha256']}
                                binding = classify_html(section, row, section_url)
                                result['financial_section_binding'] = binding
                                if binding['html_original_accessible'] and binding['receipt_path_matches'] is not False:
                                    extracted.extend(extract_financial_html(section, row, section_source))
                            except ProbeStop as error:
                                result.update(stop_requested=True, financial_section_failure=str(error))
                            except ValueError as error:
                                result['financial_section_failure'] = str(error)
                        result['extraction_status'] = 'automatic_extraction_attempted_not_manual_verified'
                        owners = [r for r in extracted if r.get('account_category') == 'owners_equity'
                                  and r.get('statement_scope') == 'CFS' and r.get('amount_normalized') is not None]
                        amounts = {}
                        for fact in owners:
                            amounts.setdefault((fact.get('period_end'), fact.get('currency')), set()).add(fact['amount_normalized'])
                        result['within_receipt_owner_amount_conflict'] = any(len(v) > 1 for v in amounts.values())
                        if result['within_receipt_owner_amount_conflict']:
                            for fact in extracted:
                                fact['target_automatic_candidate'] = False
                                fact['within_receipt_owner_amount_conflict'] = True
                        kgaap_total = any(r.get('account_category') == 'equity_total' and r.get('statement_scope') == 'CFS'
                                         and r.get('accounting_basis') == 'KGAAP_label_observed' for r in extracted)
                        result['target_account_status'] = ('owners_amount_extracted_definition_unverified' if owners else
                            'owners_label_or_scope_unconfirmed' if any(label in compact(decode(raw)[0]) for label in OWNERS) else
                            'KGAAP_CFS_total_only_target_not_observed_no_substitution' if kgaap_total else
                            'owners_account_not_observed_in_accessed_HTML_not_proof_of_absence')
                        result['extraction_evidence'] = extracted
                        facts.extend(extracted)
                    except Exception as error:
                        result['extraction_status'] = 'parser_failed_' + type(error).__name__
                        result['parser_failure_reason'] = str(error)[:300]
                        facts.append({**source, 'sampling_stratum': stratum, 'receipt_no': row['receipt_no'],
                                      'extraction_status': result['extraction_status'], 'manual_verified': False, 'research_usable': False})
                else:
                    result['extraction_status'] = 'original_missing_or_receipt_binding_unconfirmed'
                    facts.append({**source, 'sampling_stratum': stratum, 'receipt_no': row['receipt_no'],
                                  'extraction_status': result['extraction_status'], 'manual_verified': False, 'research_usable': False})
                samples.append(result)
                with (folder / 'sources.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                if result.get('stop_requested'):
                    halted = True
                    break
                if obtained >= quota:
                    break
            if halted:
                break
    for initial in samples:
        if initial['revision_mark_observed'] or initial['attachment_addition_observed'] or not pair_key(initial):
            continue
        for corrected in samples:
            if (corrected['revision_mark_observed'] and pair_key(corrected) == pair_key(initial)
                    and (corrected['receipt_date'], corrected['receipt_no']) > (initial['receipt_date'], initial['receipt_no'])):
                paired.append({'unmarked_receipt': initial['receipt_no'], 'corrected_receipt': corrected['receipt_no'],
                               'both_originals_acquired': bool(initial['sample_original_acquired'] and corrected['sample_original_acquired']),
                               'relation_status': 'candidate_same_KIND_code_kind_period_not_final_identity_verified'})
    columns = sorted({key for row in facts for key in row}) or ['receipt_no', 'extraction_status']
    with (folder / 'extractions.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(facts)
    counts = {}
    for stratum, (_, _, quota) in STRATA.items():
        subset = [r for r in samples if r['sampling_stratum'] == stratum]
        subset_facts = [r for r in facts if r['sampling_stratum'] == stratum]
        acquired = sum(r['sample_original_acquired'] for r in subset)
        counts[stratum] = {'target': quota, 'attempted': len(subset), 'originals_acquired': acquired,
                           'shortfall': max(0, quota - acquired),
                           'report_kinds_acquired': dict(Counter(r['report_kind'] for r in subset if r['sample_original_acquired'])),
                           'owners_amount_receipts': len({r['receipt_no'] for r in subset_facts if r.get('account_category') == 'owners_equity'
                               and r.get('statement_scope') == 'CFS' and r.get('amount_normalized') is not None}),
                           'definition_compatible_automatic_candidates': len({r['receipt_no'] for r in subset_facts if r.get('target_automatic_candidate')}),
                           'manual_verified': 0}
    summary = {'strata': counts, 'network_attempts': client.calls, 'candidate_pairs': paired,
               'searches': searches, 'observed_candidate_count': len(pool),
               'extraction_status_rows': dict(Counter(r['extraction_status'] for r in facts)),
               'target_account_status_receipts': dict(Counter(r.get('target_account_status', r['extraction_status']) for r in samples)),
               'definition_status_rows': dict(Counter(r.get('definition_status', 'not_extracted') for r in facts)),
               'decision': '채택 판단 보류: 원문·회계기준·계정 정의·기업/버전 동일성의 수동 대조 필요',
               'scope': 'KOSPI marketType=1 requested, submission-year strata, fixed 27 search windows, at most 26 document attempts and one explicit financial section per receipt; not period-wide coverage, historical universe or PIT verification.',
               'dart_reuse': {'reused': ['_amount', '_current_columns (includes _header_grid)', 'infer_headers when body corp identifier is present'],
                             'not_used_as_is': '_document_evidence requires DART ZIP/XML corporation tags; its implicit OFS fallback is not applied to KIND HTML.'},
               'source_review': SOURCES}
    summary['selection_policy'] = '6/6/8 acquired HTML receipts by submission-year stratum; prefer observed candidate pairs, then report-kind/company diversity; at most quota+2 attempts per stratum. No selection on monetary extraction success.'
    with (folder / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    lines = ['# KIND 연구자료 적합성 검증', '',
             '자동 추출 결과이며 수동 검증/PIT 검증 완료가 아니다. 제출연도 구간과 실제 회계기간을 구분한다.', '',
             '| 제출연도 구간 | 목표 | 실제 원문 | 부족 | CFS 소유주지분 금액 | 정의 검토 후보 | 수동 검증 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, value in counts.items():
        lines.append(f'| {key} | {value["target"]} | {value["originals_acquired"]} | {value["shortfall"]} | {value["owners_amount_receipts"]} | {value["definition_compatible_automatic_candidates"]} | 0 |')
    lines += ['', summary['decision'], '',
              'K-GAAP의 자본총계·소수주주지분을 지배기업 소유주지분으로 대체하지 않는다. 계정 미관측은 원문 전체에 계정이 없다는 확정이 아니다.',
              'sources.jsonl과 원본 HTML의 표/행/열/단위를 대조해야 한다. 본문 레이블은 표준 계정코드 검증이 아니다.',
              '검색 0건·페이지 미완결·표본 부족·미확정 최초/정정 관계는 그대로 보존한다. 전체 2001~2014년 가용성을 일반화하지 않는다.',
              'DART 숫자/당기열 helper를 재사용했으며 기존 파서·수집기·자료·환경은 수정하지 않았다.']
    with (folder / 'SUMMARY.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return summary


def self_test_probe():
    import tempfile
    import unittest
    from unittest.mock import Mock, patch

    def fixture(receipt='20050331000001', title='사업보고서 (2004.12)'):
        return f'''<html><p>총 1 건</p><table class="list"><tbody><tr><td>1</td><td>10:20</td>
        <td><a>가상기업</a></td><td><a title="{title}" onclick="openDisclsViewer('{receipt}','')">보고서</a></td>
        <td>제출자</td></tr></tbody></table></html>'''

    def financial_fixture(amount='123', heading='연결재무상태표', year=2014, basis='한국채택국제회계기준'):
        return f'''<html><body><company-name aregcik="00111704">가상기업</company-name>
        <p>{basis}</p><p>사업연도 {year}년 1월 1일부터 {year}년 12월 31일까지</p>
        <h2>{heading}</h2><p>(단위: 백만원)</p><table><thead><tr><th rowspan="2">과목</th>
        <th colspan="2">{year}-12-31</th><th colspan="2">{year-1}-12-31</th></tr>
        <tr><th>통화</th><th>금액</th><th>통화</th><th>금액</th></tr></thead><tbody>
        <tr><td>지배기업 소유주지분</td><td></td><td>{amount}</td><td>₩</td><td>999</td></tr>
        <tr><td>자본총계</td><td></td><td>456</td><td></td><td>888</td></tr>
        <tr><td>비지배지분</td><td></td><td>333</td><td></td><td>111</td></tr>
        </tbody></table></body></html>'''

    def financial_row(year=2014):
        return {'sampling_stratum': '2011-2014' if year >= 2011 else '2001-2005',
                'company_name': '가상기업', 'kind_company_code': '01234',
                'receipt_no': f'{year}0331000001', 'receipt_date': f'{year}-03-31',
                'receipt_date_basis': 'fixture', 'report_name': f'사업보고서 ({year}.12)', 'report_kind': '사업보고서'}

    class Tests(unittest.TestCase):
        def test_financial_target_and_diagnostic_accounts_are_separate(self):
            rows = extract_financial_html(financial_fixture(), financial_row(), {'source_path': 'fixture.html'})
            owner = next(r for r in rows if r['account_category'] == 'owners_equity')
            self.assertEqual(owner['amount_normalized'], '123000000')
            self.assertEqual(owner['statement_scope'], 'CFS')
            self.assertEqual(owner['period_start'], '2014-01-01')
            self.assertEqual(owner['period_end'], '2014-12-31')
            self.assertTrue(owner['target_automatic_candidate'])
            self.assertFalse(owner['manual_verified'])
            self.assertFalse(owner['research_usable'])
            self.assertIsNone(owner['standard_account_id'])
            self.assertFalse(any(r['target_automatic_candidate'] for r in rows if r['account_category'] != 'owners_equity'))
            self.assertIn('column_indexes', owner)
            self.assertIn('unit_evidence', owner)

        def test_saved_OCI_balance_sheet_target_and_income_lookalike(self):
            saved = Path(__file__).resolve().parents[2] / 'outputs/collection/KIND_RESEARCH_AUDIT/20261101T233745597484'
            if not (saved / 'sources.jsonl').exists():
                self.skipTest('saved OCI original unavailable')
            row = next(json.loads(line) for line in (saved / 'sources.jsonl').read_text(encoding='utf-8').splitlines()
                       if json.loads(line)['receipt_no'] == '20110512000216')
            path = saved / row['source_path']
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, row['source_sha256'])
            text = decode(path.read_bytes())[0]
            records = extract_financial_html(text, row, {'source_path': path.name})
            owners = [r for r in records if r.get('account_category') == 'owners_equity']
            self.assertEqual(len(owners), 1)
            target = owners[0]
            self.assertEqual((target['table_index'], target['row_index'], target['physical_column_index']), (166, 34, 1))
            self.assertEqual(target['amount_raw'], '2,227,563,970')
            self.assertEqual(target['amount_normalized'], '2227563970000')
            self.assertEqual((target['period_start'], target['period_end']), ('2011-01-01', '2011-03-31'))
            self.assertEqual((target['statement_scope'], target['currency'], target['display_scale']), ('CFS', 'KRW', 1000))
            self.assertFalse(any(r.get('amount_raw') == '307,319,162' for r in owners))
            income = BeautifulSoup(text, 'html.parser').find_all('table')[83]
            self.assertNotIn('재무상태표', compact(statement_context(income)[0]))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

        def test_blank_zero_negative_and_prior_period_are_not_confused(self):
            for value, expected in [('', None), ('0', '0'), ('(123)', '-123000000')]:
                rows = extract_financial_html(financial_fixture(amount=value), financial_row(), {})
                owner = next(r for r in rows if r['account_category'] == 'owners_equity')
                self.assertEqual(owner['amount_normalized'], expected)
                if value == '':
                    self.assertEqual(owner['extraction_status'], 'current_amount_blank')
                    self.assertFalse(owner['target_automatic_candidate'])

        def test_legacy_headers_and_kgaap_definition_are_not_substitution(self):
            legacy = financial_fixture(year=2005, heading='연결대차대조표', basis='기업회계기준에 따라').replace('<thead>', '').replace('</thead>', '')
            rows = extract_financial_html(legacy, financial_row(2005), {})
            owner = next(r for r in rows if r['account_category'] == 'owners_equity')
            self.assertEqual(owner['amount_normalized'], '123000000')
            self.assertEqual(owner['definition_status'], 'pre2011_definition_unconfirmed')
            self.assertFalse(owner['target_automatic_candidate'])
            no_owner = legacy.replace('지배기업 소유주지분', '자본금')
            rows = extract_financial_html(no_owner, financial_row(2005), {})
            self.assertFalse(any(r.get('target_automatic_candidate') for r in rows))
            self.assertTrue(any(r['account_category'] == 'equity_total' for r in rows))

        def test_scope_and_unit_uncertainty_block_target(self):
            for heading in ('개별재무상태표', '재무상태표'):
                rows = extract_financial_html(financial_fixture(heading=heading), financial_row(), {})
                self.assertFalse(any(r['target_automatic_candidate'] for r in rows))
            rows = extract_financial_html(financial_fixture().replace('(단위: 백만원)', ''), financial_row(), {})
            self.assertFalse(any(r['target_automatic_candidate'] for r in rows))
            self.assertTrue(all(r['amount_normalized'] is None for r in rows))

        def test_sampling_quotas_and_pair_links_need_identifiers(self):
            rows = []
            for i in range(12):
                row = financial_row()
                row.update(receipt_no=f'20140331{i:06d}', revision_mark_observed=i % 2 == 1,
                           attachment_addition_observed=False, report_kind=KINDS[i % 3])
                rows.append(row)
            selected = ordered_samples(rows, 8)
            self.assertEqual(len(selected), 10)
            self.assertEqual({r['report_kind'] for r in selected}, set(KINDS))
            self.assertEqual(pair_key({**rows[0], 'kind_company_code': None}), None)
            self.assertEqual(sum(v[2] for v in STRATA.values()), 20)
            self.assertEqual(len(research_windows()), 27)

        def test_research_mode_dry_run_has_no_http_or_new_folder(self):
            import sys
            with patch.object(sys, 'argv', ['probe', '--research-audit']), patch(__name__ + '.Client') as client:
                main_probe()
                client.assert_not_called()

        def test_research_end_to_end_20_sources_and_no_manual_claim(self):
            pool = []
            for key, (year, _, quota) in STRATA.items():
                for number in range(quota):
                    row = financial_row(year)
                    row.update(receipt_no=f'{year}0331{number:06d}', sampling_stratum=key,
                               revision_mark_observed=number % 2 == 1, attachment_addition_observed=False,
                               receipt_prefix_matches_query=True, viewer_url='fixture', actual_html_url=None)
                    pool.append(row)
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                client = Mock(calls=0, records=[])
                def original(_client, row):
                    year = int(row['receipt_date'][:4])
                    raw = financial_fixture(year=year, basis='한국채택국제회계기준' if year >= 2011 else '기업회계기준에 따라').replace('₩', '\\').encode('cp949')
                    path = folder / (row['receipt_no'] + '.html')
                    path.write_bytes(raw)
                    number = len(client.records) + 1
                    client.records.extend([{'event': 'request_started', 'number': number, 'label': 'original_' + row['receipt_no']},
                                           {'event': 'response', 'number': number, 'path': path.name, 'sha256': hashlib.sha256(raw).hexdigest()}])
                    return {**row, 'html_original_accessible': True, 'receipt_path_matches': True,
                            'actual_html_url': BASE + '/external/fixture.htm'}
                with patch(__name__ + '.search_research', return_value=([], pool)), patch(__name__ + '.inspect_document', side_effect=original):
                    summary = research_audit(client, folder, 2)
                self.assertEqual([s['originals_acquired'] for s in summary['strata'].values()], [6, 6, 8])
                self.assertEqual(len((folder / 'sources.jsonl').read_text(encoding='utf-8').splitlines()), 20)
                with (folder / 'extractions.csv').open(encoding='utf-8-sig', newline='') as stream:
                    records = list(csv.DictReader(stream))
                self.assertTrue(records)
                self.assertTrue(all(r['manual_verified'] == 'False' and r['research_usable'] == 'False' for r in records))
                self.assertTrue((folder / 'SUMMARY.md').exists())
                client.fetch.assert_not_called()

        def test_empty_research_pool_reports_shortfall_without_filling(self):
            with tempfile.TemporaryDirectory() as temporary:
                client = Mock(calls=0, records=[])
                with patch(__name__ + '.search_research', return_value=([], [])):
                    summary = research_audit(client, Path(temporary), 2)
                self.assertEqual(sum(r['shortfall'] for r in summary['strata'].values()), 20)
                self.assertTrue(all(r['originals_acquired'] == 0 for r in summary['strata'].values()))
                client.fetch.assert_not_called()

        def test_range_search_preserves_original_date_and_company_identifier(self):
            client = Mock()
            sample = fixture(receipt='20050323000001').replace('<a>가상기업</a>', '<a onclick="companysummary_open(\'01234\')">가상기업</a>')
            client.fetch.return_value = sample
            searches, rows = search_research(client, [{'from_date': '2005-03-20', 'to_date': '2005-03-31', 'kind': '사업보고서'}], 2)
            self.assertEqual(rows[0]['receipt_date'], '2005-03-23')
            self.assertEqual(rows[0]['kind_company_code'], '01234')
            self.assertIn('not_independently_verified', rows[0]['receipt_date_basis'])
            self.assertTrue(searches[0]['pagination_complete'])
            self.assertEqual(client.fetch.call_args.kwargs['data']['lastReport'], '')

        def test_body_receipt_conflict_and_period_conflict_never_become_candidates(self):
            text = financial_fixture().replace('<body>', '<body>접수번호: 19990101000001')
            rows = extract_financial_html(text, financial_row(), {})
            self.assertTrue(all(r['body_receipt_identity_conflict'] for r in rows))
            self.assertFalse(any(r['target_automatic_candidate'] for r in rows))
            text = financial_fixture().replace('2014년 12월 31일', '2014년 12월 30일')
            rows = extract_financial_html(text, financial_row(), {})
            self.assertFalse(any(r['target_automatic_candidate'] for r in rows))

        def test_search_versions_and_empty_are_distinct(self):
            rows, metadata = parse_search(fixture(title='[기재정정] 사업보고서 (2004.12)'), '2005-03-31')
            self.assertTrue(rows[0]['revision_mark_observed'])
            self.assertFalse(rows[0]['initial_version_confirmed'])
            self.assertEqual(rows[0]['receipt_date_basis'], 'single_day_query_only')
            self.assertEqual(metadata['total_reported'], 1)
            empty = '<table class="list"><tbody><tr><td colspan="6">조회된 결과값이 없습니다.</td></tr></tbody></table>'
            self.assertEqual(parse_search(empty, '2005-03-31')[0], [])
            with self.assertRaises(ValueError):
                parse_search('<html>로그인</html>', '2005-03-31')
            self.assertEqual(search_form('2005-03-31', 2)['lastReport'], '')
            self.assertEqual(search_form('2005-03-31', 2)['pageIndex'], '2')

        def test_docno_is_content_id_and_no_old_version_fallback(self):
            row = parse_search(fixture(), '2005-03-31')[0][0]
            client = Mock()
            url = BASE + '/external/2005/03/31/000001/20050331000002/12345.htm'
            client.fetch.side_effect = ['<select><option value="20050331000001|N">old</option><option value="20050331000002|Y">new</option></select>',
                                        f'<script>location.href="{url}"</script>', '<html><body>가상기업 사업보고서<table><tr><td>본문</td></tr></table></body></html>']
            result = inspect_document(client, row)
            self.assertTrue(result['html_original_accessible'])
            self.assertTrue(result['receipt_path_matches'])
            self.assertFalse(result['version_identity_verified'])
            self.assertEqual(client.fetch.call_args_list[1].kwargs['data']['docNo'], '20050331000002')
            client.reset_mock()
            client.fetch.side_effect = ['<select><option value="20050331000001|N">old</option></select>']
            self.assertFalse(inspect_document(client, row)['html_original_accessible'])
            self.assertEqual(client.fetch.call_count, 1)

        def test_repeated_page_not_complete_and_zero_not_absence(self):
            client = Mock(calls=0, records=[])
            client.fetch.return_value = fixture().replace('총 1 건', '총 300 건')
            result = probe(client, ['2005-03-31'], 2, 0)
            self.assertFalse(result['searches'][0]['pagination_complete'])
            self.assertIn('repeated_page', result['searches'][0]['failure_reason'])
            client.fetch.return_value = '<p>총 0 건</p><table class="list"><tbody><tr><td>검색된 결과가 없습니다.</td></tr></tbody></table>'
            result = probe(client, ['2005-03-31'], 2, 0)
            self.assertIn('unconfirmed', result['searches'][0]['search_status'])

        def test_cap_includes_failures_and_retries(self):
            with tempfile.TemporaryDirectory() as temporary:
                session = Mock()
                session.request.side_effect = requests.Timeout('fixture failure')
                client = Client(Path(temporary), 1, session=session, interval=0)
                with patch.object(time, 'sleep'):
                    with self.assertRaises(ProbeStop):
                        client.fetch('GET', VIEWER)
                self.assertEqual(session.request.call_count, 1)
                self.assertEqual(client.calls, 1)
                self.assertFalse(session.request.call_args.kwargs['allow_redirects'])

        def test_encoding_urls_and_shell_are_not_originals(self):
            text = '<html><body>사업보고서</body></html>'
            self.assertEqual(decode(text.encode('cp949'))[1], 'cp949')
            self.assertEqual(html_urls('<iframe src="/blank.html"></iframe>'), [])
            self.assertEqual(html_urls('<iframe src="/external/a.htm"></iframe>'), [BASE + '/external/a.htm'])
            row = parse_search(fixture(), '2005-03-31')[0][0]
            self.assertFalse(classify_html('<html><body><iframe>사업보고서</iframe></body></html>', row, VIEWER)['html_original_accessible'])
            self.assertFalse(classify_html('<html><body>로그인 안내</body></html>', row, VIEWER)['html_original_accessible'])

        def test_pagination_and_search_success_do_not_imply_document_success(self):
            client = Mock(calls=0, records=[])
            client.fetch.side_effect = [fixture().replace('총 1 건', '총 2 건'),
                                       fixture(receipt='20050331000002').replace('총 1 건', '총 2 건')]
            result = probe(client, ['2005-03-31'], 2, 0)
            self.assertTrue(result['searches'][0]['pagination_complete'])
            self.assertEqual(len(result['filings_observed']), 2)
            client.fetch.side_effect = [fixture(), '<html><body>본문 옵션 없음</body></html>']
            result = probe(client, ['2005-03-31'], 1, 1)
            self.assertTrue(result['searches'][0]['list_search_succeeded'])
            self.assertFalse(result['documents'][0]['html_original_accessible'])

        def test_legacy_explicit_iframe_and_rate_limit(self):
            row = parse_search(fixture(), '2005-03-31')[0][0]
            client = Mock()
            client.fetch.side_effect = ['<iframe src="/external/2005/03/31/000001/id/a.htm"></iframe>',
                                        '<html><body>가상기업 사업보고서<table><tr><td>사업의 내용</td></tr></table></body></html>']
            result = inspect_document(client, row)
            self.assertTrue(result['html_original_accessible'])
            self.assertFalse(result['version_identity_verified'])
            with tempfile.TemporaryDirectory() as temporary:
                session = Mock()
                response = Mock(status_code=429, headers={'Content-Type': 'text/html; charset=utf-8'})
                response.iter_content.return_value = [b'<html>rate limited</html>']
                session.request.return_value = response
                bounded = Client(Path(temporary), 10, session=session, interval=0)
                with self.assertRaises(ProbeStop):
                    bounded.fetch('GET', VIEWER)
                self.assertEqual(bounded.calls, 1)
                self.assertEqual(session.request.call_count, 1)

        def test_saved_capro_resolution_reaches_body_get_with_synthetic_body(self):
            saved = Path(__file__).resolve().parents[2] / 'outputs/collection/KIND_RESEARCH_AUDIT/20261101T232209757575'
            if not (saved / 'sources.jsonl').exists():
                self.skipTest('saved KIND resolver fixture unavailable')
            row = next(json.loads(line) for line in (saved / 'sources.jsonl').read_text(encoding='utf-8').splitlines()
                       if json.loads(line)['receipt_no'] == '20010321000007')
            events = [json.loads(line) for line in (saved / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
            paths = [saved / name for name in ('055_viewer_20010321000007.html', '056_resolve_20010321000007.html')]
            before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            for path, digest in zip(paths, before):
                self.assertEqual(digest, next(e['sha256'] for e in events if e.get('path') == path.name))
            client = Mock()
            # Only shell/resolver are preserved real responses. The body below is explicitly synthetic.
            client.fetch.side_effect = [decode(paths[0].read_bytes())[0], decode(paths[1].read_bytes())[0],
                                        '<html><body>카프로 사업보고서<table><tr><td>회사의 개요</td></tr></table></body></html>']
            result = inspect_document(client, row)
            expected = BASE + '/external/2001/03/21/000007/20010321000017/2001032100001704.htm'
            self.assertEqual(result['actual_html_url'], expected)
            self.assertEqual(client.fetch.call_count, 3)
            self.assertEqual(client.fetch.call_args_list[2].args, ('GET', expected))
            self.assertEqual(result['selected_content_id']['content_id'], '20010321000017')
            self.assertTrue(result['content_id_path_matches'])
            self.assertFalse(result['version_identity_verified'])
            self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])

        def test_saved_26_pairs_resolve_or_keep_receipt_mismatch_unconfirmed(self):
            saved = Path(__file__).resolve().parents[2] / 'outputs/collection/KIND_RESEARCH_AUDIT/20261101T232209757575/sources.jsonl'
            if not saved.exists():
                self.skipTest('saved KIND resolver fixture unavailable')
            rows = [json.loads(line) for line in saved.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 26)
            selected_receipts, unresolved_receipts = [], []
            for row in rows:
                selected, evidence, _ = select_body_url(row['resolved_html_urls'], row['receipt_no'], row['selected_content_id']['content_id'])
                if selected:
                    selected_receipts.append(row['receipt_no'])
                    self.assertFalse(urlsplit(selected).path.endswith('_toc.htm'))
                else:
                    unresolved_receipts.append(row['receipt_no'])
                    self.assertTrue(all(not e['receipt_path_matches'] for e in evidence))
                self.assertEqual(sum(e['role'] == 'toc' for e in evidence), 1)
            self.assertEqual(len(selected_receipts), 25)
            # Its resolver points to sequence 000081, not requested receipt sequence 000029.
            self.assertEqual(unresolved_receipts, ['20060510000029'])

        def test_url_selection_preserves_receipt_and_content_id_binding(self):
            receipt, cid = '20050331000001', '20050331000002'
            base = BASE + '/external/2005/03/31/000001/' + cid + '/'
            self.assertIsNone(select_body_url([base + 'report_toc.htm'], receipt, cid)[0])
            self.assertIsNone(select_body_url([base.replace(cid, '20050331000003') + 'report.htm'], receipt, cid)[0])
            self.assertIsNone(select_body_url([base + 'a.htm', base + 'b.htm'], receipt, cid)[0])
            text = f"parent.setPath('{base}a_toc.htm','{base}b.htm','');"
            self.assertEqual(select_body_url([base + 'a.htm', base + 'b.htm'], receipt, cid, text)[0], base + 'b.htm')

        def test_http_and_body_failure_stages_are_separate(self):
            row = parse_search(fixture(), '2005-03-31')[0][0]
            base = BASE + '/external/2005/03/31/000001/20050331000002/report.htm'
            shell = '<option value="20050331000002|Y">current</option>'
            resolver = f'<script>location.href="{base}"</script>'
            client = Mock()
            client.fetch.side_effect = [shell, resolver, ValueError('http_404_redirects_not_followed')]
            result = inspect_document(client, row)
            self.assertEqual(result['failure_stage'], 'http_access')
            self.assertEqual(result['actual_html_url'], base)
            client.fetch.side_effect = [shell, resolver, '<html><body>로그인</body></html>']
            result = inspect_document(client, row)
            self.assertEqual(result['failure_stage'], 'body_validation')
            self.assertFalse(result['html_original_accessible'])
            client.fetch.side_effect = [shell, '<html><body>no URL</body></html>']
            result = inspect_document(client, row)
            self.assertEqual(result['failure_stage'], 'url_resolution')

        def test_TOC_and_title_only_HTTP_200_are_not_originals(self):
            row = parse_search(fixture(), '2005-03-31')[0][0]
            base = BASE + '/external/2005/03/31/000001/20050331000002/report'
            report = '<html><body>사업보고서<table><tr><td>회사의 개요</td></tr></table></body></html>'
            self.assertFalse(classify_html(report, row, base + '_toc.htm')['html_original_accessible'])
            self.assertFalse(classify_html('<html><body>사업보고서</body></html>', row, base + '.htm')['html_original_accessible'])
            self.assertFalse(classify_html('<html><head><title>사업보고서</title></head><body></body></html>', row, base + '.htm')['html_original_accessible'])

        def test_default_dry_run_never_constructs_http_client(self):
            import sys
            with patch.object(sys, 'argv', ['probe']), patch(__name__ + '.Client') as client:
                main_probe()
                client.assert_not_called()

        def test_external_link_and_date_mismatch_do_not_trigger_requests(self):
            with tempfile.TemporaryDirectory() as temporary:
                session = Mock()
                client = Client(Path(temporary), 10, session=session, interval=0)
                with self.assertRaises(ValueError):
                    client.fetch('GET', 'https://example.com/a.htm')
                session.request.assert_not_called()
            row = parse_search(fixture(), '2001-03-30')[0][0]
            client = Mock()
            self.assertEqual(inspect_document(client, row)['document_status'], 'search_date_scope_mismatch')
            client.fetch.assert_not_called()

    result = unittest.TextTestRunner(verbosity=2, buffer=True).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main_probe():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--research-audit', action='store_true', help='Integrated 6/6/8 original-and-financial-evidence audit')
    parser.add_argument('--dates', nargs='+')
    parser.add_argument('--max-calls', type=int, default=None)
    parser.add_argument('--max-pages', type=int, default=2)
    parser.add_argument('--max-documents', type=int, default=8)
    args = parser.parse_args()
    if args.self_test:
        self_test_probe()
        return
    args.max_calls = args.max_calls if args.max_calls is not None else (180 if args.research_audit else 40)
    if args.research_audit:
        if args.dates is not None:
            parser.error('research-audit uses its declared 27 fixed windows, not --dates')
        if not (1 <= args.max_calls <= 240 and 1 <= args.max_pages <= 3):
            parser.error('research caps: calls 1..240, pages 1..3 per window')
        print(json.dumps({'mode': 'research_execute' if args.execute else 'research_dry_run',
                          'targets_by_submission_year': {k: v[2] for k, v in STRATA.items()},
                          'search_windows': research_windows(), 'max_calls': args.max_calls,
                          'max_document_attempts': 26, 'manual_verification_automatic': False}, ensure_ascii=False, indent=2))
        if not args.execute:
            print('DRY-RUN: network=0, no files written; no KIND research usability conclusion yet.')
            return
        folder = Path.cwd() / RESEARCH_OUTPUT / datetime.now().strftime('%Y%m%dT%H%M%S%f')
        folder.mkdir(parents=True, exist_ok=False)
        client = Client(folder, args.max_calls)
        try:
            summary = research_audit(client, folder, args.max_pages)
            print(json.dumps({'strata': summary['strata'], 'decision': summary['decision'],
                              'network_attempts': client.calls, 'output': str(folder)}, ensure_ascii=False, indent=2))
        finally:
            client.session.close()
        return
    args.dates = args.dates or DATES
    if not (1 <= args.max_calls <= 60 and 1 <= args.max_pages <= 3 and 0 <= args.max_documents <= 8 and 1 <= len(args.dates) <= 8):
        parser.error('caps: calls 1..60, pages 1..3 per date, documents 0..8 total, dates 1..8')
    for day in args.dates:
        try:
            parsed = date.fromisoformat(day)
            if parsed.isoformat() != day or not 2001 <= parsed.year <= 2014:
                raise ValueError
        except ValueError:
            parser.error('dates must be YYYY-MM-DD within 2001..2014')
    days = list(dict.fromkeys(args.dates))
    print(json.dumps({'mode': 'execute' if args.execute else 'dry-run', 'submission_dates': days,
                      'max_calls': args.max_calls, 'max_pages_per_day': args.max_pages,
                      'max_documents_total': args.max_documents, 'lastReport': '',
                      'search_reportNm': '보고서', 'local_report_kinds': KINDS,
                      'note': 'No automatic date expansion; search and HTML access judged separately.'}, ensure_ascii=False, indent=2))
    if not args.execute:
        print('DRY-RUN: no network, no output files. Run --self-test for offline fixtures.')
        return
    folder = Path.cwd() / PROBE_OUTPUT / datetime.now().strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True, exist_ok=False)
    client = Client(folder, args.max_calls)
    try:
        result = probe(client, days, args.max_pages, args.max_documents)
    except KeyboardInterrupt:
        result = {'status': 'interrupted', 'network_attempts': client.calls,
                  'note': 'Raw response files and request records preserved; no automatic resume.'}
    finally:
        client.session.close()
    with (folder / 'result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'network_attempts': client.calls, 'retry_attempts': result.get('retry_attempts'),
                      'searches': result.get('searches'),
                      'html_accessible': sum(d['html_original_accessible'] for d in result.get('documents', [])),
                      'result': str(folder / 'result.json')}, ensure_ascii=False, indent=2))


OUTPUT = Path('outputs/collection/KIND_ORIGINAL_COLLECT')


RAW = Path('data/raw/kind')


DEFAULT_RUN = OUTPUT / '20000101_20211231'


KST = timezone(timedelta(hours=9))


SCHEMA = 1


HTTP_RATE_LIMIT = 40


LIST_POLICY = 3


RECEIPT_START = '2000-01-01'


RECEIPT_END = '2015-12-31'


SCOPE_RULE = {'receipt_start': RECEIPT_START, 'receipt_end': RECEIPT_END,
              'period_years': [2000, 2014], 'market': 'observed_KOSPI_only',
              'uncertain_period_market_identity': 'hold', 'warmup': '2000_receipt_1999-12',
              'bulk_gate': 'complete_lists_no_holds_reviewed_capacity_plan'}


LEGACY_SCOPE_DESCRIPTION = 'Keep all regular-report list versions across markets. Download target/unknown-period candidates; preserve unknown-period originals. Name links only prioritize; no company identity inference.'


SCOPE_DESCRIPTION = 'Keep list versions across markets; download only reviewed observed-KOSPI candidates in the narrowed receipt/period scope. Hold uncertain periods, markets and identities; preserve all existing originals.'


WARMUP_PRECEDING_RULE = {'report_period': '1999-12', 'receipt_year': 2000, 'report_kinds': list(KINDS)}


def compatible_run_config(previous, current):
    """Keep collection.json immutable; allow only explicitly approved policy additions."""
    rates = {f'{n} request attempts per rolling 60 seconds across all KIND bulk endpoints and retries'
             for n in (30, 40)}
    descriptions = {LEGACY_SCOPE_DESCRIPTION, SCOPE_DESCRIPTION}
    if previous.get('scope_policy') != current.get('scope_policy') and not all(c.get('scope_policy') in descriptions for c in (previous, current)):
        return False
    return (previous.get('rate_limit') in rates and current.get('rate_limit') in rates and
            all(c.get('warmup_preceding_rule') in (None, WARMUP_PRECEDING_RULE) and c.get('list_validation_policy') in (None, 2, LIST_POLICY)
                and c.get('receipt_end') in (None, '2021-12-31', RECEIPT_END) and c.get('download_scope') in (None, SCOPE_RULE)
                for c in (previous, current)) and
            {k: v for k, v in previous.items() if k not in {'rate_limit', 'warmup_preceding_rule', 'list_validation_policy', 'receipt_end', 'download_scope', 'scope_policy'}} ==
            {k: v for k, v in current.items() if k not in {'rate_limit', 'warmup_preceding_rule', 'list_validation_policy', 'receipt_end', 'download_scope', 'scope_policy'}})


def date_evidence(row):
    """List display and ID prefix are observations, never a verified public date."""
    inferred = kind_receipt_date(row.get('receipt_no'))
    displayed = row.get('list_display_date')
    if not displayed and row.get('receipt_date_basis') == 'explicit_list_cell':
        displayed = row.get('receipt_date')
    return {'list_display_date': displayed, 'receipt_id_date': inferred,
            'date_conflict': bool(row.get('date_conflict') or (displayed and inferred and displayed != inferred)),
            'date_status': 'date_unverified', 'publication_date': None,
            'publication_date_verified': False,
            # Compatibility/sorting field only; its source and lack of verification
            # are explicit. No PIT or public-date decision is made by this collector.
            'receipt_date': displayed or inferred,
            'receipt_date_basis': 'explicit_list_cell' if displayed else 'receipt_identifier_prefix_not_independently_verified'}


def warmup_preceding(row, period):
    observed = date_evidence(row)['receipt_date']
    return bool(period == WARMUP_PRECEDING_RULE['report_period'] and observed and
                observed.startswith(str(WARMUP_PRECEDING_RULE['receipt_year']) + '-') and
                (row.get('report_kind') in KINDS or bool(row.get('report_kind_candidates'))))


def market_label(row):
    value = row.get('market_label_observed')
    return 'KOSPI' if value in {'유가증권', 'KOSPI'} else 'KOSDAQ' if value in {'코스닥', 'KOSDAQ'} else 'unknown'


def scope_decision(row, market_history=None):
    report = report_title_metadata(row['report_name'])
    period = period_label(row['report_name'])
    dates = date_evidence(row)
    observed, market = dates['receipt_date'], market_label(row)
    company = row.get('kind_company_code')
    history = (market_history or {}).get(company, set())
    reasons = []
    if observed and not RECEIPT_START <= observed <= RECEIPT_END:
        return 'excluded', ['receipt_outside_scope']
    if report['report_class'] == 'non_regular':
        return 'excluded', ['non_regular']
    if period and not (2000 <= int(period[:4]) <= 2014 or warmup_preceding({**row, **report}, period)):
        return 'excluded', ['period_outside_scope']
    if market == 'KOSDAQ' and 'KOSPI' not in history:
        return 'excluded', ['observed_KOSDAQ']
    if not observed:
        reasons.append('receipt_date_unknown')
    if dates['receipt_id_date'] and not RECEIPT_START <= dates['receipt_id_date'] <= RECEIPT_END:
        reasons.append('receipt_date_boundary_conflict')
    if not period:
        reasons.append('period_unknown_2015' if observed and observed[:4] == '2015' else 'period_unknown')
    if market == 'unknown':
        reasons.append('market_unknown')
    if {'KOSPI', 'KOSDAQ'} <= history:
        reasons.append('historical_market_conflict')
    if not company or row.get('identity_conflict'):
        reasons.append('company_identity_unconfirmed')
    if report['report_class'] == 'regular_related_unconfirmed':
        reasons.append('report_kind_unconfirmed')
    return ('hold', reasons) if reasons else ('candidate', ['observed_KOSPI_regular_in_scope'])


def filing_policy(row):
    report = report_title_metadata(row['report_name'])
    period = period_label(row['report_name'])
    return report, period, scope_decision(row)[0] == 'candidate'


def make_scope_review(root, rows, list_complete=False, budget_gb=20):
    history = defaultdict(set)
    for row in rows:
        observed = date_evidence(row)['receipt_date']
        if observed and RECEIPT_START <= observed <= RECEIPT_END and row.get('kind_company_code'):
            history[row['kind_company_code']].add(market_label(row))
    decisions = []
    for row in rows:
        status, reasons = scope_decision(row, history)
        decisions.append({'receipt_no': row['receipt_no'], 'kind_company_code': row.get('kind_company_code'),
                          'company_name': row.get('company_name'), 'report_name': row['report_name'],
                          'receipt_date_observed_or_inferred': date_evidence(row)['receipt_date'],
                          'receipt_date_basis': date_evidence(row)['receipt_date_basis'],
                          'list_display_date': date_evidence(row)['list_display_date'],
                          'receipt_id_date': date_evidence(row)['receipt_id_date'], 'date_status': 'date_unverified',
                          'period_label': period_label(row['report_name']), 'market_observed': row.get('market_label_observed'),
                          'market_group': market_label(row), 'decision': status, 'reasons': reasons,
                          'date_conflict': date_evidence(row)['date_conflict'], 'publication_date_verified': False})
    samples = []
    parent = root / 'outputs/collection/KIND_RESEARCH_AUDIT/20261101T233745597484'
    source = parent / 'sources.jsonl'
    if source.exists():
        for line in source.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            if item.get('sample_original_acquired') and item.get('source_path'):
                path = (parent / item['source_path']).resolve()
                if not path.is_relative_to(parent.resolve()) or digest(path) != item['source_sha256']:
                    raise ValueError('KIND_sample_integrity_failure')
                samples.append(path.stat().st_size)
    sizes = sorted(samples)
    count = sum(d['decision'] == 'candidate' for d in decisions)
    held = sum(d['decision'] == 'hold' for d in decisions)
    mean = sum(sizes) / len(sizes) if sizes else None
    p90 = sizes[max(0, (len(sizes) * 9 + 9) // 10 - 1)] if sizes else None
    # Viewer/resolver/control/log allowance is explicit and conservative, not a measured guarantee.
    estimated = count * p90 * 2 + 2 * 1024**3 if p90 else None
    free = shutil.disk_usage(root).free
    budget = min(budget_gb * 1000**3, free)
    candidate_ids = sorted(d['receipt_no'] for d in decisions if d['decision'] == 'candidate')
    digest_value = hashlib.sha256(json.dumps({'scope': SCOPE_RULE, 'rows': sorted(decisions, key=lambda r:r['receipt_no'])}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    report = {'scope': SCOPE_RULE, 'list_complete': list_complete, 'scope_plan_sha256': digest_value,
              'observed_filings': len(decisions), 'candidate_count': count, 'hold_count': held,
              'excluded_count': sum(d['decision'] == 'excluded' for d in decisions),
              'candidate_unique_receipts': len(set(candidate_ids)),
              'by_market': dict(Counter(d['market_group'] for d in decisions)),
              'by_market_and_decision': dict(Counter(d['market_group'] + ':' + d['decision'] for d in decisions)),
              'excluded_by_reason': dict(Counter(reason for d in decisions if d['decision']=='excluded' for reason in d['reasons'])),
              'candidates_by_period': dict(Counter(d['period_label'] for d in decisions if d['decision']=='candidate')),
              'holds_by_reason': dict(Counter(reason for d in decisions if d['decision']=='hold' for reason in d['reasons'])),
              'sample_count': len(sizes), 'sample_bytes_mean': mean, 'sample_bytes_p90': p90,
              'sample_bytes_max': max(sizes) if sizes else None,
              'body_estimate_mean_bytes': count * mean if mean else None,
              'body_estimate_p90_bytes': count * p90 if p90 else None,
              'body_estimate_sample_max_bytes': count * max(sizes) if sizes else None,
              'planning_required_bytes': estimated, 'planning_formula': 'candidate_count * sample_body_p90 * 2 + 2 GiB; additionally reserve 4 GiB',
              'disk_free_bytes': free, 'user_budget_bytes': budget_gb * 1000**3,
              'capacity_with_reserve_ok': bool(estimated is not None and estimated + 4 * 1024**3 <= budget),
              'final_estimate': bool(list_complete), 'bulk_ready': bool(list_complete and held == 0 and sizes and estimated + 4 * 1024**3 <= budget),
              'network_calls': 0, 'PIT_verified': False,
              'notes': ['Market labels are KIND list observations, not an independently certified historical universe.',
                        'Unknowns are held, not deleted. No DART company mapping limits the population.',
                        'The 20-report convenience sample is not a statistical upper bound; later list discovery changes volume.']}
    return report, decisions, history


def save_scope_review(folder, report, decisions):
    target = folder / 'scope_reviews' / datetime.now(KST).strftime('%Y%m%dT%H%M%S%f')
    target.mkdir(parents=True, exist_ok=False)
    with (target / 'scope_review.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    with (target / 'scope_review.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]) if decisions else ['receipt_no','decision','reasons'])
        writer.writeheader()
        for row in decisions:
            writer.writerow({**row, 'reasons': '|'.join(row['reasons'])})
    with (target / 'holds.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(decisions[0]) if decisions else ['receipt_no','decision','reasons'])
        writer.writeheader()
        for row in decisions:
            if row['decision'] == 'hold':
                writer.writerow({**row, 'reasons': '|'.join(row['reasons'])})
    return target


def now():
    return datetime.now(KST).isoformat()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stored_bytes(path):
    """The response bytes as received; original bodies are kept gzip-compressed (lossless)."""
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def kind_source_observations(root):
    """Read preserved KIND source pages directly; query coverage is never inferred."""
    parent = root / RAW
    pages, originals = [], {}
    if not parent.is_dir(): return pages, originals
    folders = [parent] + [p for p in parent.iterdir() if p.is_dir()]
    for folder in folders:
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {'.html', '.htm', '.gz'}: continue
            raw = stored_bytes(path)
            text = decode(raw, '')[0]
            if BeautifulSoup(text, 'html.parser').select_one('table.list') is not None:
                rows, _ = parse_search(text, '')
                pages.extend({**r, **date_evidence(r)} for r in rows)
                continue
            if not re.search(r'class=["\']?COVER-TITLE|<DOCUMENT\b', text, re.I): continue
            receipts = set(re.findall(r'(?:acptno|rcpNo|receiptNo|receipt_no)\s*[=:]\s*["\']?(\d{14})', text, re.I))
            named = re.match(r'(\d{14})[_\.]', path.name)
            if named: receipts.add(named[1])
            if len(receipts) != 1:
                raise ValueError('KIND_original_receipt_unidentifiable:' + str(path.relative_to(root)))
            receipt = receipts.pop()
            if not kind_receipt_date(receipt): raise ValueError('KIND_original_receipt_invalid')
            record = {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'content_type': 'text/html', 'period_label': period_label(BeautifulSoup(text, 'html.parser').get_text(' ', strip=True)),
                'period_status': 'observed_in_original'}
            if path.suffix == '.gz':
                record.update(compression='gzip', stored_bytes=path.stat().st_size, stored_sha256=digest(path))
            if receipt in originals and originals[receipt]['sha256'] != record['sha256']:
                raise ValueError('multiple_KIND_original_payloads_without_selection:' + receipt)
            originals[receipt] = record
    return pages, originals


TARGET_FILES = ('targets', 'holds', 'pilot')


def load_targets(folder, names=TARGET_FILES):
    """KIND receipts named by a 'kind_collect pairs targets' folder, in file order.

    A hold row lists every candidate (ambiguous receipts, same-day and +-15-day candidates):
    downloading them resolves nothing, it only keeps the later manual look offline.
    """
    found, files = {}, {}
    for name in names:
        path = folder / (name + '.csv')
        if not path.exists():
            raise ValueError('targets_file_missing:' + name)
        rows = 0
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                rows += 1
                candidates = (row.get('kind_receipt_no') or '').split(';') + [row.get('same_day_candidate'), row.get('wide_candidate')]
                for receipt in candidates:
                    receipt = (receipt or '').strip()
                    if receipt:
                        if not re.fullmatch(r'\d{8}[0-9KM]\d{5}', receipt):  # e.g. 20020401K00463
                            raise ValueError('target_receipt_malformed:' + receipt)
                        found.setdefault(receipt, set()).add(name)
        files[name] = {'sha256': digest(path), 'rows': rows}
    return {r: sorted(n) for r, n in found.items()}, {'folder': folder.as_posix(), 'files': files, 'receipts': len(found)}


def atomic_json(path, value):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    replace_retry(temp, path)


def replace_retry(temp, path):
    for attempt in range(4):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(0.2 * (attempt + 1))


@contextmanager
def exclusive(path):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


class StopRun(RuntimeError):
    pass


def request_key(method, url, params):
    return hashlib.sha256(json.dumps([method, url, params or {}], sort_keys=True).encode()).hexdigest()


def block_screen(text):
    plain = BeautifulSoup(text[:100000], 'html.parser').get_text(' ', strip=True)
    return bool(re.search(r'captcha|access\s*denied|too\s*many\s*requests|비정상.{0,20}접근|'
                          r'접근.{0,15}(차단|제한)|자동.{0,10}(차단|방지)|보안문자|로봇.{0,10}확인', plain, re.I))


def window_id(start, end, kind):
    return kind + '|' + start + '|' + end


def months(start, end):
    current = start.replace(day=1)
    while current <= end:
        last = date(current.year, current.month, calendar.monthrange(current.year, current.month)[1])
        for kind in KINDS:
            yield {'id': window_id(max(start, current).isoformat(), min(end, last).isoformat(), kind),
                   'start': max(start, current).isoformat(), 'end': min(end, last).isoformat(),
                   'kind': kind, 'state': 'pending', 'page': 1, 'expected': None, 'seen': []}
        current = (last + timedelta(days=1)).replace(day=1)


def issuer_priority_sources(root):
    """Only approved development name CSVs and KRX event folders. No global data walk."""
    names, sources, codes = defaultdict(set), [], set()
    for period in ('warmup_2000', 'train_2001_2021'):
        path = root / 'data/raw' / (period + '__company_name.csv')
        if not path.exists():
            sources.append({'path': str(path.relative_to(root)), 'status': 'missing'})
            continue
        if path.resolve() != path:
            raise ValueError('redirected_panel_source')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if header[0] != 'month' or any(not re.fullmatch(r'[A-Za-z0-9]{6}', c) for c in header[1:]):
                raise ValueError('development_code_header_invalid')
            codes.update(header[1:])
            for row in reader:
                if len(row) != len(header):
                    raise ValueError('company_name_csv_shape_invalid')
                for code, name in zip(header[1:], row[1:]):
                    if name.strip():
                        names[name.strip()].add(code)
        sources.append({'path': str(path.relative_to(root)), 'sha256': digest(path)})
    for folder in ('상장', '상장폐지'):
        parent = root / 'data/raw' / folder
        for path in sorted(parent.glob('*.csv')):
            if not path.resolve().is_relative_to(parent.resolve()):
                raise ValueError('redirected_history_source')
            with path.open(encoding='cp949', newline='') as stream:
                for row in csv.DictReader(stream):
                    if row.get('시장구분') == 'KOSPI' and row.get('종목코드') and row.get('종목명'):
                        names[row['종목명'].strip()].add(row['종목코드'])
            sources.append({'path': str(path.relative_to(root)), 'sha256': digest(path)})
    return names, {'sources': sources, 'provided_codes': len(codes),
                   'identity_warning': 'Exact name matches are priority hints, not historical issuer/security mappings. No issuer is excluded by current membership.'}


class Journal:
    """Append-only manifest is authoritative; SQLite is a replayable local index."""
    def __init__(self, folder):
        self.folder, self.path = folder, folder / 'manifest.jsonl'
        self.path.touch(exist_ok=True)
        self.db = sqlite3.connect(folder / 'state.sqlite3', timeout=10)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, payload TEXT, valid INTEGER);
            CREATE TABLE IF NOT EXISTS windows (id TEXT PRIMARY KEY, start TEXT, state TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS filings (receipt TEXT PRIMARY KEY, priority INTEGER, doc_state TEXT, payload TEXT, detail TEXT, period_scope TEXT, period_status TEXT);
            CREATE TABLE IF NOT EXISTS exceptions (k TEXT PRIMARY KEY, payload TEXT);
            CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, invocation TEXT, finished INTEGER);
            CREATE TABLE IF NOT EXISTS raw_files (path TEXT PRIMARY KEY, payload TEXT);
            CREATE TABLE IF NOT EXISTS filing_policy (receipt TEXT PRIMARY KEY, classification TEXT, period TEXT, eligible INTEGER, unknown_period INTEGER);
        ''')
        self.db.commit()
        offset = self.db.execute("SELECT v FROM meta WHERE k='offset'").fetchone()
        offset = int(offset[0]) if offset else 0
        if offset > self.path.stat().st_size:
            raise ValueError('journal_truncated_do_not_reset')
        with self.path.open('rb') as stream:
            stream.seek(offset)
            for line in stream:
                if not line.endswith(b'\n'):
                    raise ValueError('torn_journal_tail_manual_review_required')
                self.apply(json.loads(line), stream.tell())

    def append(self, event):
        event = {'at': now(), **event}
        with self.path.open('ab') as stream:
            stream.write((json.dumps(event, ensure_ascii=False) + '\n').encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
            offset = stream.tell()
        self.apply(event, offset)

    def apply(self, e, offset):
        db = self.db
        with db:
            event = e['event']
            if event == 'request_started':
                db.execute('INSERT OR IGNORE INTO attempts VALUES (?,?,0)', (e['attempt_id'], e['invocation']))
            elif event in {'response', 'transport_error', 'response_rejected'}:
                db.execute('UPDATE attempts SET finished=1 WHERE id=?', (e['attempt_id'],))
                if event == 'response':
                    db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)', (e['key'], json.dumps(e, ensure_ascii=False), int(e['cacheable'])))
                    db.execute('INSERT OR IGNORE INTO raw_files VALUES (?,?)', (e['path'], json.dumps(e, ensure_ascii=False)))
            elif event == 'cache_seed':
                db.execute('INSERT OR IGNORE INTO cache VALUES (?,?,1)', (e['key'], json.dumps(e['response'], ensure_ascii=False)))
                db.execute('INSERT OR IGNORE INTO raw_files VALUES (?,?)', (e['response']['path'], json.dumps(e['response'], ensure_ascii=False)))
            elif event == 'cache_invalidated':
                db.execute('UPDATE cache SET valid=0 WHERE k=?', (e['key'],))
            elif event == 'cache_revalidated':
                db.execute('UPDATE cache SET valid=1 WHERE k=?', (e['key'],))
            elif event == 'warmup_preceding_requeued':
                db.execute("UPDATE filings SET doc_state='pending' WHERE receipt=? AND doc_state='excluded_title_period'",
                           (e['receipt_no'],))
            elif event == 'window':
                w = e['window']
                db.execute('INSERT OR REPLACE INTO windows VALUES (?,?,?,?)', (w['id'], w['start'], w['state'], json.dumps(w, ensure_ascii=False)))
            elif event == 'list_policy_started':
                for w in e['windows']:
                    db.execute('INSERT OR IGNORE INTO windows VALUES (?,?,?,?)',
                               (w['id'], w['start'], w['state'], json.dumps(w, ensure_ascii=False)))
                db.execute("INSERT OR REPLACE INTO meta VALUES ('list_policy',?)", (str(e['policy']),))
            elif event == 'filing_reclassification':
                for change in e['changes']:
                    record = db.execute('SELECT payload,detail FROM filings WHERE receipt=?', (change['receipt_no'],)).fetchone()
                    if not record:
                        raise ValueError('reclassified_receipt_missing')
                    row = {**json.loads(record[0]), **change['classification']}
                    db.execute('UPDATE filings SET payload=? WHERE receipt=?', (json.dumps(row, ensure_ascii=False), change['receipt_no']))
                    if row['report_class'] == 'non_regular' and not json.loads(record[1]).get('original_acquired'):
                        db.execute("UPDATE filings SET doc_state='excluded_non_regular' WHERE receipt=?", (change['receipt_no'],))
                    self.index_filing(row)
            elif event == 'filing':
                row = e['filing']
                old = db.execute('SELECT payload FROM filings WHERE receipt=?', (row['receipt_no'],)).fetchone()
                if old:
                    previous = json.loads(old[0])
                    if (previous.get('kind_company_code') and row.get('kind_company_code') and
                        previous['kind_company_code'] != row['kind_company_code']):
                        previous['identity_conflict'] = True
                        db.execute('UPDATE filings SET payload=?,doc_state=? WHERE receipt=?',
                                   (json.dumps(previous, ensure_ascii=False), 'identity_conflict', row['receipt_no']))
                        db.execute('INSERT OR REPLACE INTO exceptions VALUES (?,?)',
                                   ('identity:' + row['receipt_no'], json.dumps({'category': 'identity_conflict', 'receipt_no': row['receipt_no'], 'variants': [previous, row]}, ensure_ascii=False)))
                    else:
                        observations = previous.get('date_observations', [
                            {**date_evidence(previous), 'source_query_key': previous.get('source_query_key')}])
                        observation = {**date_evidence(row), 'source_query_key': row.get('source_query_key')}
                        if observation not in observations:
                            observations.append(observation)
                        merged = {**previous, **row, 'date_observations': observations}
                        # Different displayed dates are metadata uncertainty, not
                        # evidence of different securities or grounds to drop a receipt.
                        displayed = {o['list_display_date'] for o in observations if o.get('list_display_date')}
                        merged['date_conflict'] = any(o['date_conflict'] for o in observations) or len(displayed) > 1
                        merged['identity_conflict'] = previous.get('identity_conflict', False)
                        db.execute('UPDATE filings SET payload=? WHERE receipt=?',
                                   (json.dumps(merged, ensure_ascii=False), row['receipt_no']))
                else:
                    db.execute('INSERT INTO filings(receipt,priority,doc_state,payload,detail) VALUES (?,?,?,?,?)',
                               (row['receipt_no'], e['priority'], 'pending', json.dumps(row, ensure_ascii=False), '{}'))
                stored = json.loads(db.execute('SELECT payload FROM filings WHERE receipt=?', (row['receipt_no'],)).fetchone()[0])
                if stored.get('report_class') == 'non_regular':
                    detail = json.loads(db.execute('SELECT detail FROM filings WHERE receipt=?', (row['receipt_no'],)).fetchone()[0])
                    if not detail.get('original_acquired'):
                        db.execute("UPDATE filings SET doc_state='excluded_non_regular' WHERE receipt=?", (row['receipt_no'],))
                self.index_filing(stored)
            elif event == 'document':
                db.execute('UPDATE filings SET doc_state=?,detail=?,period_scope=?,period_status=? WHERE receipt=?',
                           (e['state'], json.dumps(e, ensure_ascii=False), e.get('research_period_scope'), e.get('period_status'), e['receipt_no']))
                if e.get('original_acquired'):
                    db.execute('DELETE FROM exceptions WHERE k=?', ('document:' + e['receipt_no'],))
            elif event == 'exception':
                db.execute('INSERT OR REPLACE INTO exceptions VALUES (?,?)', (e['key'], json.dumps(e, ensure_ascii=False)))
            elif event == 'retry_failed':
                query = "SELECT id,payload FROM windows WHERE state='failed'"
                args = ()
                if e.get('validation_policy'):
                    query += " AND json_extract(payload,'$.validation_policy')=?"
                    args = (e['validation_policy'],)
                for identity, payload in db.execute(query, args).fetchall():
                    w = json.loads(payload)
                    w.update(state='pending', page=1, expected=None, seen=[])
                    db.execute('UPDATE windows SET state=?,payload=? WHERE id=?', ('pending', json.dumps(w, ensure_ascii=False), identity))
                db.execute("UPDATE filings SET doc_state='pending' WHERE doc_state IN ('failed','version_unconfirmed')")
            db.execute("INSERT OR REPLACE INTO meta VALUES ('offset',?)", (str(offset),))

    def index_filing(self, row):
        report, period, eligible = filing_policy(row)
        self.db.execute('INSERT OR REPLACE INTO filing_policy VALUES (?,?,?,?,?)',
                        (row['receipt_no'], report['report_class'], period, int(eligible), int(period is None)))

    def rebuild_filing_policy(self):
        with self.db:
            for payload, in self.db.execute('SELECT payload FROM filings').fetchall():
                self.index_filing(json.loads(payload))

    def cached(self, key):
        row = self.db.execute('SELECT payload FROM cache WHERE k=? AND valid=1', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def exception(self, key, category, **fields):
        self.append({'event': 'exception', 'key': key, 'category': category, **fields})


class Collector:
    def __init__(self, root, folder, journal, names, max_calls, *, clock=time.time, sleep=time.sleep):
        self.root, self.folder, self.journal, self.names = root, folder, journal, names
        self.max_calls, self.calls, self.cache_hits = max_calls, 0, 0
        self.stage = 'initializing'
        self.calls_by_stage = Counter({'lists': 0, 'originals': 0, 'other': 0})
        self.invocation = uuid.uuid4().hex
        self.clock, self.sleep = clock, sleep
        self.session = requests.Session()
        for protocol in ('http://', 'https://'):
            self.session.mount(protocol, requests.adapters.HTTPAdapter(max_retries=0))
        self.rate_path = root / OUTPUT / 'HTTP_RATE.json'
        self.last_response, self.last_key = None, None
        self.verified_this_invocation = set()
        self.doc_traversal_finished = False
        self.last_exception_export = 0
        self.last_exception_count = 0
        self.min_free_bytes = 4 * 1024**3 + 64 * 1024**2
        self.targets = None
        policy = journal.db.execute("SELECT v FROM meta WHERE k='list_policy'").fetchone()
        self.validation_policy = int(policy[0]) if policy else None

    def log(self, event):
        self.journal.append({'invocation': self.invocation, **event})

    def stop_check(self):
        if (self.folder / 'STOP').exists():
            raise StopRun('stop_file')

    def wait(self, seconds):
        until = self.clock() + seconds
        while self.clock() < until:
            self.stop_check()
            self.sleep(min(0.25, max(0, until - self.clock())))
        self.stop_check()

    def reserve(self):
        self.stop_check()
        if self.calls >= self.max_calls:
            raise StopRun('max_calls')
        times = json.loads(self.rate_path.read_text(encoding='utf-8')) if self.rate_path.exists() else []
        while True:
            current = self.clock()
            recent = [t for t in times if current - t < 60.05]
            delay = max(0, recent[-1] + 1.51 - current) if recent else 0
            if len(recent) >= HTTP_RATE_LIMIT:
                delay = max(delay, recent[-HTTP_RATE_LIMIT] + 60.05 - current)
            if delay <= 0:
                break
            self.wait(delay)
        # Reserved slots survive restarts and different KIND bulk run folders.
        atomic_json(self.rate_path, recent + [self.clock()])

    def safe_source(self, value):
        path = (self.root / value).resolve()
        parents = [self.root / RAW, self.root / 'outputs/collection/KIND_RESEARCH_AUDIT', self.root / 'outputs/collection/KIND_HISTORY_PROBE']
        if not any(path.is_relative_to(p.resolve()) for p in parents):
            raise ValueError('source_outside_KIND_only_scope')
        return path

    def verify_response(self, record):
        path = self.safe_source(record['path'])
        if record.get('compression') == 'gzip':
            # Both the stored file and the bytes it restores must match what was received.
            if path.stat().st_size != record['stored_bytes'] or digest(path) != record['stored_sha256']:
                raise ValueError('cached_original_integrity_failure')
            raw = stored_bytes(path)
            if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
                raise ValueError('cached_original_integrity_failure')
            return path
        if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('cached_original_integrity_failure')
        return path

    def fetch(self, method, url, *, data=None, label='request'):
        self.stop_check()
        if label == 'list' and (not data or not RECEIPT_START <= data.get('fromDate', '') <= data.get('toDate', '') <= RECEIPT_END):
            raise ValueError('list_request_outside_narrowed_scope')
        parsed = urlsplit(url)
        if parsed.hostname != 'kind.krx.co.kr' or parsed.scheme not in {'http', 'https'} or parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
            raise ValueError('non_KIND_or_invalid_URL')
        key = request_key(method, url, data)
        self.last_key = key
        cached = self.journal.cached(key)
        if cached:
            path = self.verify_response(cached)  # mismatch never becomes a new network request
            self.cache_hits += 1
            self.last_response = cached
            return decode(stored_bytes(path), cached.get('content_type', ''))[0]
        for retry in range(3):
            if shutil.disk_usage(self.root).free < self.min_free_bytes:
                raise StopRun('disk_reserve_reached')
            self.reserve()
            attempt = self.invocation + '-' + str(self.calls + 1)
            self.log({'event': 'request_started', 'attempt_id': attempt, 'key': key, 'method': method,
                      'url': url, 'params': data or {}, 'retry': retry, 'label': label})
            self.calls += 1
            stage = 'lists' if label == 'list' else 'originals' if label.startswith(('viewer:', 'resolver:', 'original:')) else 'other'
            self.calls_by_stage[stage] += 1
            try:
                response = self.session.request(method, url, data=data, timeout=(15, 60), stream=True,
                    allow_redirects=False, headers={'User-Agent': 'Mozilla/5.0', 'Referer': BASE + '/'})
                body = bytearray()
                try:
                    for chunk in response.iter_content(65536):
                        body.extend(chunk)
                        if len(body) > 32 * 1024**2:
                            break
                finally:
                    response.close()
            except requests.RequestException as error:
                self.log({'event': 'transport_error', 'attempt_id': attempt, 'key': key, 'error_type': type(error).__name__, 'retry': retry})
                if retry < 2:
                    self.wait(2 ** (retry + 1))
                    continue
                raise StopRun('transport_failed_3_attempts') from None
            raw = bytes(body)
            content_type = response.headers.get('Content-Type', '')
            text, encoding = decode(raw, content_type)
            # A report body names products such as '가스누출 자동차단장치'; a block screen never carries the report cover title.
            report_body = label.startswith('original:') and bool(re.search(r'class=["\']?COVER-TITLE', text[:50000], re.I))
            blocked = (response.status_code in {401, 403, 429} or 300 <= response.status_code < 400
                       or (block_screen(text) and not report_body))
            valid = response.status_code == 200 and bool(raw) and not blocked and len(raw) <= 32 * 1024**2 and encoding != 'utf-8-replacement_unconfirmed'
            target = self.root / RAW / self.folder.name
            target.mkdir(parents=True, exist_ok=True)
            # Bodies average ~720KB and deflate about tenfold; list/viewer pages stay as received.
            compress = label.startswith('original:')
            stored = gzip.compress(raw, 9, mtime=0) if compress else raw
            if compress and gzip.decompress(stored) != raw:
                raise StopRun('compression_roundtrip_failed')
            path = target / (attempt + ('.html.gz' if compress else '.html'))
            with path.open('xb') as stream:
                stream.write(stored)
                stream.flush()
                os.fsync(stream.fileno())
            record = {'event': 'response', 'attempt_id': attempt, 'key': key, 'method': method, 'url': url,
                      'params': data or {}, 'http': response.status_code, 'retry': retry, 'label': label,
                      'path': path.relative_to(self.root).as_posix(), 'bytes': len(raw),
                      'sha256': hashlib.sha256(raw).hexdigest(), 'content_type': content_type,
                      'encoding': encoding, 'cacheable': valid, 'blocked': blocked,
                      'oversized_truncated': len(raw) > 32 * 1024**2,
                      **({'compression': 'gzip', 'stored_bytes': len(stored),
                          'stored_sha256': hashlib.sha256(stored).hexdigest()} if compress else {})}
            record['receipt_no'] = label.split(':', 1)[1] if ':' in label else None
            self.log(record)
            self.last_response = record
            if blocked:
                raise StopRun('http_429' if response.status_code == 429 else 'blocking_suspected')
            if 500 <= response.status_code < 600 and retry < 2:
                self.wait(2 ** (retry + 1))
                continue
            if 500 <= response.status_code < 600:
                raise StopRun('server_errors_3_attempts')
            if not valid:
                raise ValueError('HTTP_' + str(response.status_code) + '_empty_encoding_size_or_redirect_failure')
            return text
        raise ValueError('request_failed')

    def invalidate(self):
        if self.last_key:
            self.log({'event': 'cache_invalidated', 'key': self.last_key})

    def revalidate_legacy_list_cache(self):
        """Explicit retry only: revalidate stored list bytes under the current policy."""
        for key, payload in self.journal.db.execute('SELECT k,payload FROM cache WHERE valid=0').fetchall():
            self.stop_check()
            record = json.loads(payload)
            if record.get('label') != 'list' or not record.get('cacheable') or record.get('http') != 200:
                continue
            if (record.get('params', {}).get('fromDate', RECEIPT_START) > RECEIPT_END or
                not RECEIPT_START <= record.get('params', {}).get('toDate', RECEIPT_END) <= RECEIPT_END):
                continue
            try:
                path = self.verify_response(record)
                text = decode(path.read_bytes(), record.get('content_type', ''))[0]
                rows, meta = parse_search(text, record['params']['toDate'])
                if not rows or meta['unparsed_rows'] or meta['total_reported'] is None or meta['total_reported'] < len(rows):
                    continue
                if meta.get('current_page') is not None and meta['current_page'] != int(record['params']['pageIndex']):
                    continue
                for row in rows:
                    day = kind_receipt_date(row.get('receipt_no'))
                    observed_day = row['receipt_date'] if row['receipt_date_basis'] == 'explicit_list_cell' else day
                    if not day or not record['params']['fromDate'] <= observed_day <= record['params']['toDate']:
                        raise ValueError('legacy_list_not_revalidated')
                # Reusing a transport response is NOT completing a search interval.
                # collect_window records date conflicts and still rejects incomplete pages.
                self.log({'event': 'cache_revalidated', 'key': key, 'reason': 'list_policy_v2_response_revalidated',
                          'path': record['path'], 'sha256': record['sha256'], 'rows': len(rows)})
            except (ValueError, OSError, KeyError):
                continue  # Missing/changed/malformed files are never marked successful.

    def seed_cache(self):
        """Read existing KIND request records only. No DART directory access."""
        for parent in ('KIND_RESEARCH_AUDIT', 'KIND_HISTORY_PROBE'):
            for manifest in sorted((self.root / 'outputs/collection' / parent).glob('*/requests.jsonl')):
                starts = {}
                for line in manifest.read_text(encoding='utf-8').splitlines():
                    self.stop_check()
                    e = json.loads(line)
                    if e['event'] == 'request_started':
                        starts[e['number']] = e
                    elif e['event'] == 'response' and e.get('http') == 200 and not e.get('oversized_truncated'):
                        start = starts.get(e['number'])
                        if not start:
                            continue
                        key = request_key(start['method'], start['url'], start.get('params'))
                        if self.journal.cached(key):
                            continue
                        path = manifest.parent / e['path']
                        record = {**e, 'path': path.relative_to(self.root).as_posix(), 'key': key,
                                  'method': start['method'], 'url': start['url'], 'params': start.get('params') or {}, 'cacheable': True}
                        path = self.verify_response(record)
                        if not block_screen(decode(path.read_bytes())[0]):
                            self.log({'event': 'cache_seed', 'key': key, 'response': record, 'source_manifest': str(manifest.relative_to(self.root))})
                sources = manifest.parent / 'sources.jsonl'
                if sources.exists():
                    keys = ('company_name', 'kind_company_code', 'market_label_observed', 'report_name', 'report_kind',
                            'receipt_no', 'receipt_date', 'receipt_date_basis', 'revision_mark_observed',
                            'attachment_addition_observed', 'viewer_url')
                    for line in sources.read_text(encoding='utf-8').splitlines():
                        old = json.loads(line)
                        if not all(old.get(k) for k in ('receipt_no', 'receipt_date', 'report_kind')):
                            continue
                        if not RECEIPT_START <= old['receipt_date'] <= RECEIPT_END:
                            continue
                        row = {k: old.get(k) for k in keys}
                        self.store_filing(row, 'preserved_KIND_source:' + str(sources.relative_to(self.root)))

    def store_filing(self, row, source_key):
        row.update(report_title_metadata(row['report_name']))
        row.update(date_evidence(row))
        candidate_codes = sorted(self.names.get(row['company_name'].strip(), []))
        tags = re.findall(r'\[([^\]]+)\]', row['report_name'])
        version_type = ('attachment_addition' if '첨부추가' in row['report_name'] else
                        'attachment_correction' if '첨부정정' in row['report_name'] else
                        'content_correction' if '기재정정' in row['report_name'] else
                        'correction_unspecified' if '정정' in row['report_name'] else 'unmarked_initial_candidate')
        row.update(source_query_key=source_key, candidate_stock_codes=candidate_codes,
                   issuer_link_status='exact_name_candidate_only' if candidate_codes else 'historical_issuer_unresolved',
                   revision_types=tags, version_type=version_type,
                   initial_version_confirmed=False)
        self.log({'event': 'filing', 'filing': row, 'priority': 0 if candidate_codes else 1})
        if row['date_conflict']:
            self.journal.exception('date:' + row['receipt_no'], 'date_conflict', receipt_no=row['receipt_no'],
                                   list_display_date=row['list_display_date'], receipt_id_date=row['receipt_id_date'],
                                   date_status='date_unverified', publication_date_verified=False,
                                   viewer_url=row.get('viewer_url') or viewer_url(row['receipt_no']),
                                   source_query_key=source_key)

    def split_or_fail(self, w, reason):
        self.journal.exception('window:' + w['id'], 'list_integrity', window=w['id'], reason=reason)
        if w['start'] < w['end']:
            start, end = date.fromisoformat(w['start']), date.fromisoformat(w['end'])
            middle = start + (end - start) // 2
            for a, b in ((start, middle), (middle + timedelta(days=1), end)):
                child = {'id': window_id(a.isoformat(), b.isoformat(), w['kind']), 'parent': w['id'],
                         'start': a.isoformat(), 'end': b.isoformat(), 'kind': w['kind'],
                         'state': 'pending', 'page': 1, 'expected': None, 'seen': []}
                if w.get('validation_policy'):
                    child.update(id=child['id'] + '|list-v' + str(w['validation_policy']), validation_policy=w['validation_policy'])
                if not self.journal.db.execute('SELECT 1 FROM windows WHERE id=?', (child['id'],)).fetchone():
                    self.log({'event': 'window', 'window': child})
            w['state'] = 'split'
        else:
            w['state'] = 'failed'
        self.log({'event': 'window', 'window': w})

    def collect_window(self, w):
        while True:
            self.stop_check()
            form = search_form(w['end'], w['page'])
            form.update(fromDate=w['start'], reportNm=w['kind'], lastReport='', marketType='')
            try:
                text = self.fetch('POST', BASE + '/disclosure/details.do', data=form, label='list')
                rows, meta = parse_search(text, w['end'])
                if meta['unparsed_rows']:
                    raise ValueError('unparsed_list_rows')
                if meta.get('current_page') is not None and meta['current_page'] != w['page']:
                    raise ValueError('response_current_page_mismatch')
                if meta.get('page_size') is not None and meta['page_size'] != int(form['currentPageSize']):
                    raise ValueError('server_page_size_mismatch')
                seen = set(w['seen'])
                new_ids = set()
                for row in rows:
                    receipt = row['receipt_no']
                    day = kind_receipt_date(receipt)
                    if not day:
                        raise ValueError('unidentifiable_receipt')
                    evidence = date_evidence(row)
                    observed_day = evidence['list_display_date'] or day
                    if not w['start'] <= observed_day <= w['end']:
                        raise ValueError('server_ignored_date_window')
                    row.update(evidence, receipt_prefix_matches_query=w['start'] <= day <= w['end'])
                    # reportNm is a title substring, not an exclusive report type.
                    # Keep all returned IDs in pagination, including non-target notices.
                    if w['kind'] not in re.sub(r'\s+', '', row['report_name']):
                        self.journal.exception('filter:' + w['id'], 'server_title_filter_mismatch',
                                               window=w['id'], receipt_no=receipt, requested=w['kind'], report_name=row['report_name'])
                    self.store_filing(row, self.last_key)
                    new_ids.add(receipt)
                if rows and (new_ids & seen or len(new_ids) != len(rows)):
                    raise ValueError('duplicate_or_repeated_page')
                seen.update(new_ids)
                total = meta['total_reported']
                if total is None:
                    raise ValueError('total_count_missing')
                if not rows and total == 0 and w['expected'] and len(seen) == w['expected']:
                    total = w['expected']  # some end-of-list pages reset the displayed total
                if total == 0:
                    known = self.journal.db.execute("SELECT 1 FROM filings WHERE json_extract(payload,'$.receipt_date') BETWEEN ? AND ? AND instr(replace(json_extract(payload,'$.report_name'),' ',''),?)>0 LIMIT 1",
                                                    (w['start'], w['end'], w['kind'])).fetchone()
                    if known:
                        raise ValueError('empty_result_contradicts_preserved_filing')
                if w['expected'] is not None and total != w['expected']:
                    raise ValueError('total_changed_between_pages')
                if len(seen) > total or (not rows and len(seen) != total):
                    raise ValueError('page_count_inconsistent')
                if w['start'] != w['end'] and total > 2000:
                    raise ValueError('chunk_exceeds_20_pages_split_to_verify')
                if w['page'] > 1000:
                    raise ValueError('daily_page_safety_bound_not_complete')
                w.update(expected=total, seen=sorted(seen))
                if meta.get('total_pages') is not None and total > 0:
                    size = meta.get('page_size') or int(form['currentPageSize'])
                    if meta['total_pages'] != (total + size - 1) // size:
                        raise ValueError('reported_page_count_inconsistent')
                    if w['page'] == meta['total_pages']:
                        if len(seen) != total:
                            raise ValueError('last_page_does_not_cover_reported_total')
                        w['state'] = 'done'
                        self.log({'event': 'window', 'window': w})
                        return
                    if len(seen) == total:
                        raise ValueError('count_reached_before_reported_last_page')
                # One empty page beyond the advertised last row detects ignored size/count caps.
                if not rows and len(seen) == total:
                    w['state'] = 'empty_unconfirmed' if total == 0 else 'done'
                    if total == 0:
                        self.journal.exception('empty:' + w['id'], 'empty_search_not_absence_proof', window=w['id'])
                    self.log({'event': 'window', 'window': w})
                    return
                if rows and len(seen) == total and len(rows) < 100 and meta.get('total_pages') is None:
                    w['state'] = 'done'
                    self.log({'event': 'window', 'window': w})
                    return
                w['page'] += 1
                self.log({'event': 'window', 'window': w})
            except ValueError as error:
                self.invalidate()
                self.split_or_fail(w, str(error))
                return

    def collect(self, phase):
        if phase in {'all', 'list'}:
            self.collect_lists()
        if phase == 'targets':
            # The approved DART-paired list replaces the bulk scope gate, not list completeness.
            if self.validation_policy and self.journal.db.execute("SELECT 1 FROM windows WHERE json_extract(payload,'$.validation_policy')=? AND state IN ('pending','failed') LIMIT 1", (self.validation_policy,)).fetchone():
                raise StopRun('list_validation_incomplete')
            self.download_authorized = True
            self.collect_target_originals()
        if phase in {'all', 'originals'}:
            if self.validation_policy and self.journal.db.execute("SELECT 1 FROM windows WHERE json_extract(payload,'$.validation_policy')=? AND state IN ('pending','failed') LIMIT 1", (self.validation_policy,)).fetchone():
                raise StopRun('list_validation_incomplete')
            self.authorize_originals()
            self.collect_originals()

    def scope_review(self, save=False):
        rows = [json.loads(p) for p, in self.journal.db.execute('SELECT payload FROM filings')]
        incomplete = self.journal.db.execute("SELECT 1 FROM windows WHERE json_extract(payload,'$.validation_policy')=? AND state IN ('pending','failed') LIMIT 1", (LIST_POLICY,)).fetchone()
        complete = self.validation_policy == LIST_POLICY and incomplete is None
        report, decisions, self.market_history = make_scope_review(self.root, rows, complete)
        if save:
            report['review_output'] = str(save_scope_review(self.folder, report, decisions))
        return report

    def authorize_originals(self):
        report = self.scope_review(save=True)
        self.log({'event': 'scope_review', 'report': report})
        path = getattr(self, 'approved_plan', None)
        if not report['bulk_ready']:
            raise StopRun('scope_or_capacity_unresolved')
        if not path:
            raise StopRun('explicit_scope_capacity_review_required')
        approved = json.loads(path.read_text(encoding='utf-8'))
        if (approved.get('scope_plan_sha256') != report['scope_plan_sha256'] or not approved.get('bulk_ready') or
            approved.get('scope') != SCOPE_RULE):
            raise StopRun('reviewed_scope_plan_stale_or_unready')
        self.download_authorized = True

    def collect_lists(self):
        self.stage = 'lists'
        self.checkpoint('running_lists')
        while True:
            query = "SELECT payload FROM windows WHERE state='pending'"
            args = ()
            if self.validation_policy:
                query += " AND json_extract(payload,'$.validation_policy')=?"
                args = (self.validation_policy,)
            record = self.journal.db.execute(query + ' ORDER BY start,id LIMIT 1', args).fetchone()
            if not record:
                break
            self.collect_window(json.loads(record[0]))
            self.checkpoint('running_lists')

    def prepare_list_validation(self, allow_upgrade=False):
        """Append a new validation generation; never reset/delete legacy windows."""
        if self.validation_policy != LIST_POLICY:
            exists = self.journal.db.execute('SELECT 1 FROM windows LIMIT 1').fetchone()
            if exists and not allow_upgrade:
                raise StopRun('list_policy_upgrade_requires_--revalidate-lists')
            windows = []
            for w in months(date.fromisoformat(RECEIPT_START), date.fromisoformat(RECEIPT_END)):
                w.update(id=w['id'] + '|list-v' + str(LIST_POLICY), validation_policy=LIST_POLICY)
                windows.append(w)
            self.log({'event': 'list_policy_started', 'policy': LIST_POLICY, 'windows': windows,
                      'scope': SCOPE_RULE,
                      'reason': 'Recheck only months through 2015; preserve all legacy windows/responses, including 2016-2021.'})
            self.validation_policy = LIST_POLICY
        changes = []
        for receipt, payload in self.journal.db.execute('SELECT receipt,payload FROM filings').fetchall():
            row = json.loads(payload)
            report = report_title_metadata(row['report_name'])
            if ((report['report_class'] != 'regular' or report['report_kind'] != row.get('report_kind')) and
                any(row.get(k) != v for k, v in report.items())):
                changes.append({'receipt_no': receipt, 'classification': report})
        if changes:
            self.log({'event': 'filing_reclassification', 'policy': LIST_POLICY, 'changes': changes})
        self.journal.rebuild_filing_policy()

    def requeue_warmup_preceding(self):
        for payload, in self.journal.db.execute("SELECT payload FROM filings WHERE doc_state='excluded_title_period'").fetchall():
            self.stop_check()
            row = json.loads(payload)
            if warmup_preceding(row, period_label(row['report_name'])):
                self.log({'event': 'warmup_preceding_requeued', 'receipt_no': row['receipt_no'],
                          'rule': WARMUP_PRECEDING_RULE, 'reason': 'approved_warmup_preceding_original'})

    def document(self, row, roles=None):
        """roles: the --targets files naming this receipt. Their DART pairing (company, kind, date,
        marcap KOSPI) replaces KIND's own list-based scope and title-period gates."""
        receipt = row['receipt_no']
        existing = self.journal.db.execute('SELECT doc_state,detail FROM filings WHERE receipt=?', (receipt,)).fetchone()
        detail = json.loads(existing[1])
        if detail.get('original_acquired'):
            self.verify_response(detail['response'])
            self.verified_this_invocation.add(receipt)
            return
        if existing[0] not in ({'pending', 'hold_scope', 'excluded_scope', 'excluded_title_period'} if roles else {'pending'}):
            return
        report = report_title_metadata(row['report_name'])
        row.update(report)
        if report['report_class'] == 'non_regular':
            self.log({'event': 'document', 'receipt_no': receipt, 'state': 'excluded_non_regular', 'original_acquired': False})
            return
        decision, reasons = scope_decision(row, getattr(self, 'market_history', None))
        if decision != 'candidate' and not roles:
            self.log({'event': 'document', 'receipt_no': receipt, 'state': 'hold_scope' if decision == 'hold' else 'excluded_scope',
                      'original_acquired': False, 'scope_reasons': reasons, **date_evidence(row)})
            return
        if row.get('identity_conflict'):
            self.log({'event': 'document', 'receipt_no': receipt, 'state': 'identity_conflict', 'original_acquired': False})
            return
        title_period = period_label(row['report_name'])
        preceding = warmup_preceding(row, title_period)
        if title_period and not 2000 <= int(title_period[:4]) <= 2014 and not preceding and not roles:
            self.log({'event': 'document', 'receipt_no': receipt, 'state': 'excluded_title_period', 'original_acquired': False, 'title_period': title_period})
            return
        # Late reports without a period need the cover too: do not assume receipt year == fiscal year.
        result = {'event': 'document', 'receipt_no': receipt, 'original_acquired': False,
                  'title_period': title_period, 'version_identity_verified': False,
                  'download_period_intent': 'warmup_preceding' if preceding else 'target_title' if title_period else 'period_verification_original_always_preserved',
                  **date_evidence(row), 'original_url_binding_verified': False,
                  **({'download_basis': 'dart_paired_targets', 'target_roles': roles, 'kind_scope_decision': decision,
                      'kind_scope_reasons': reasons} if roles else {})}
        try:
            shell = self.fetch('GET', viewer_url(receipt), label='viewer:' + receipt)
            options = content_options(shell)
            result['content_options'] = options
            # Try observed IDs only. N is accepted solely with this receipt's own URL path,
            # never as an unchecked fallback to another version.
            options = sorted(options, key=lambda item: item['flag'] != 'Y')
            attempts = []
            chosen = None
            for option in options[:20] or [None]:
                if option:
                    resolved = self.fetch('POST', VIEWER, data={'method': 'searchContents', 'docNo': option['content_id']}, label='resolver:' + receipt)
                    cid = option['content_id']
                else:
                    resolved, cid = shell, None
                urls = html_urls(resolved)
                url, evidence, basis = select_body_url(urls, receipt, cid, resolved)
                attempts.append({'content_id': cid, 'flag': option['flag'] if option else None, 'urls': urls, 'binding_evidence': evidence, 'basis': basis})
                if url:
                    chosen = (url, cid, basis)
                    break
            result['resolver_attempts'] = attempts
            if not chosen:
                raise ValueError('version_or_body_URL_unconfirmed_no_substitution')
            url, cid, basis = chosen
            result['original_url_binding_verified'] = True
            body = self.fetch('GET', url, label='original:' + receipt)
            validation = classify_html(body, row, url, cid)
            result.update(validation, content_id=cid, source_url=url, url_binding_basis=basis)
            result['content_id_flag'] = next((v['flag'] for v in options if v['content_id'] == cid), None)
            if not validation['html_original_accessible']:
                self.invalidate()
                raise ValueError('body_validation:' + validation['document_status'])
            response = self.last_response
            # Only cover-period metadata; never call financial/equity parsers.
            flat = BeautifulSoup(body, 'html.parser').get_text(' ', strip=True)
            start, end = cover_period(flat, '')
            period = end[:7] if end else title_period
            period_status = 'explicit_cover' if end else 'title_label_only' if title_period else 'unconfirmed'
            scope = 'in_scope' if period and 2000 <= int(period[:4]) <= 2014 else 'out_of_scope' if period else 'unconfirmed'
            if warmup_preceding(row, period):
                scope = 'warmup_preceding'
            if end and title_period and end[:7] != title_period:
                period_status, scope = 'cover_title_conflict', 'unconfirmed'
            # Even unknown or out-of-scope verification originals stay physically preserved.
            result.update(state='original_saved', original_acquired=True, response=response,
                          period_start=start, period_end=end, period_label=period,
                          period_status=period_status, research_period_scope=scope,
                          source_path=response['path'], sha256=response['sha256'], http=response['http'])
            self.log(result)
            self.verified_this_invocation.add(receipt)
            if period_status != 'explicit_cover' or scope not in {'in_scope', 'warmup_preceding'}:
                self.journal.exception('period:' + receipt, 'period_pending_or_out_of_scope_original_preserved', receipt_no=receipt,
                                       period_status=period_status, research_period_scope=scope, source_path=response['path'])
        except ValueError as error:
            result.update(state='version_unconfirmed' if 'version_or_body' in str(error) else 'failed', reason=str(error))
            self.log(result)
            self.journal.exception('document:' + receipt, 'document_unconfirmed_or_failed', receipt_no=receipt, reason=str(error))

    def collect_originals(self):
        self.stage = 'originals'
        self.checkpoint('running_originals')
        # Stream rows, including saved ones for hash reconciliation. No full corpus rewrite.
        for payload, in self.journal.db.execute('SELECT payload FROM filings ORDER BY priority,receipt').fetchall():
            self.stop_check()
            row = json.loads(payload)
            try:
                self.document(row)
            except (OSError, ValueError) as error:
                self.log({'event': 'document', 'receipt_no': row['receipt_no'], 'state': 'integrity_error', 'original_acquired': False, 'reason': type(error).__name__})
                self.journal.exception('integrity:' + row['receipt_no'], 'original_integrity_unconfirmed_no_redownload', receipt_no=row['receipt_no'], reason=type(error).__name__)
            if len(self.verified_this_invocation) % 20 == 0:
                self.checkpoint('running_originals')
        self.doc_traversal_finished = True

    def collect_target_originals(self):
        """--targets: only the listed receipts; saved ones are hash-checked, never fetched again."""
        self.stage = 'originals'
        self.checkpoint('running_targets')
        for number, (receipt, roles) in enumerate(self.targets.items(), 1):
            self.stop_check()
            record = self.journal.db.execute('SELECT payload FROM filings WHERE receipt=?', (receipt,)).fetchone()
            if not record:
                self.journal.exception('target:' + receipt, 'target_receipt_not_in_kind_list', receipt_no=receipt, roles=roles)
                continue
            try:
                self.document(json.loads(record[0]), roles=roles)
            except (OSError, ValueError) as error:
                self.log({'event': 'document', 'receipt_no': receipt, 'state': 'integrity_error', 'original_acquired': False, 'reason': type(error).__name__})
                self.journal.exception('integrity:' + receipt, 'original_integrity_unconfirmed_no_redownload', receipt_no=receipt, reason=type(error).__name__)
            if number % 20 == 0:
                self.checkpoint('running_targets')

    def checkpoint(self, status, reason=None):
        db = self.journal.db
        where = " WHERE json_extract(payload,'$.validation_policy')=?" if self.validation_policy else ''
        args = (self.validation_policy,) if self.validation_policy else ()
        windows = dict(db.execute('SELECT state,count(*) FROM windows' + where + ' GROUP BY state', args).fetchall())
        docs = dict(db.execute('SELECT doc_state,count(*) FROM filings GROUP BY doc_state').fetchall())
        periods = {scope + ':' + state: n for scope, state, n in db.execute("SELECT period_scope,period_status,count(*) FROM filings WHERE doc_state='original_saved' GROUP BY period_scope,period_status")}
        incomplete = windows.get('pending', 0) + windows.get('failed', 0)
        unresolved = sum(n for key, n in docs.items() if key not in {'original_saved', 'excluded_title_period', 'excluded_non_regular', 'excluded_scope'})
        unfinished = db.execute('SELECT count(*) FROM attempts WHERE finished=0').fetchone()[0]
        exception_counts = dict(db.execute("SELECT json_extract(payload,'$.category'),count(*) FROM exceptions GROUP BY json_extract(payload,'$.category')"))
        failure_query = """SELECT COALESCE(json_extract(e.payload,'$.reason'),'unknown'),count(*)
            FROM windows w LEFT JOIN exceptions e ON e.k='window:'||w.id
            WHERE w.state='failed'"""
        if self.validation_policy:
            failure_query += " AND json_extract(w.payload,'$.validation_policy')=?"
        failure_reasons = dict(db.execute(failure_query + " GROUP BY COALESCE(json_extract(e.payload,'$.reason'),'unknown')", args))
        report = {'status': status, 'stop_reason': reason, 'updated_at': now(), 'invocation': self.invocation,
                  'calls_this_invocation': self.calls, 'max_calls_this_invocation': self.max_calls,
                  'current_stage': self.stage, 'calls_by_stage_this_invocation': dict(self.calls_by_stage),
                  'http_rate_limit_per_minute': HTTP_RATE_LIMIT,
                  'list_validation_policy': self.validation_policy,
                  'download_scope': SCOPE_RULE,
                  'scope_capacity_review_required': not getattr(self, 'download_authorized', False),
                  'legacy_list_windows': dict(db.execute("SELECT state,count(*) FROM windows WHERE COALESCE(json_extract(payload,'$.validation_policy'),0)<>? GROUP BY state", (self.validation_policy,))) if self.validation_policy else {},
                  'report_class_counts': dict(db.execute('SELECT classification,count(*) FROM filing_policy GROUP BY classification')),
                  'download_candidates': db.execute('SELECT count(*) FROM filing_policy WHERE eligible=1').fetchone()[0],
                  'candidate_period_unconfirmed': db.execute('SELECT count(*) FROM filing_policy WHERE eligible=1 AND unknown_period=1').fetchone()[0],
                  'candidate_policy_index_complete': db.execute('SELECT count(*) FROM filing_policy').fetchone()[0] == db.execute('SELECT count(*) FROM filings').fetchone()[0],
                  'warmup_preceding_rule': WARMUP_PRECEDING_RULE,
                  'list_collection_complete': bool(windows and not incomplete),
                  'originals_pending': docs.get('pending', 0),
                  'exception_counts': exception_counts, 'active_list_failure_reasons': failure_reasons,
                  'date_conflict_receipts': db.execute("SELECT count(*) FROM filings WHERE json_extract(payload,'$.date_conflict')=1").fetchone()[0],
                  'date_unverified_receipts': db.execute('SELECT count(*) FROM filings').fetchone()[0],
                  'date_review_required': True,
                  'calls_lifetime_reserved': db.execute('SELECT count(*) FROM attempts').fetchone()[0],
                  'unfinished_request_records': unfinished,
                  'cache_hits_this_invocation': self.cache_hits, 'list_windows': windows, 'document_states': docs,
                  'filings_total': db.execute('SELECT count(*) FROM filings').fetchone()[0],
                  'issuer_name_priority_hints': db.execute('SELECT count(*) FROM filings WHERE priority=0').fetchone()[0],
                  'historical_issuer_mappings_verified': 0,
                  'originals_saved': docs.get('original_saved', 0), 'period_states': dict(periods),
                  'verified_originals_this_invocation': len(self.verified_this_invocation),
                  'original_hash_traversal_finished': self.doc_traversal_finished,
                  'collection_complete': bool(self.doc_traversal_finished and not incomplete and not unresolved and not unfinished),
                  'research_ready': False,
                  'scope_note': 'Technical collection only. Unknown periods are retained, current historical-issuer identity is unresolved, empty responses do not prove absence, PIT completeness is not asserted.'}
        if self.targets:
            states = Counter()
            for receipt in self.targets:
                found = db.execute('SELECT doc_state FROM filings WHERE receipt=?', (receipt,)).fetchone()
                states[found[0] if found else 'not_in_kind_list'] += 1
            report['target_receipts'] = len(self.targets)
            report['target_states'] = dict(states)
        if status in {'finished', 'stopped'}:
            parent = self.root / RAW / self.folder.name
            known = {str((self.root / p).resolve()) for p, in db.execute('SELECT path FROM raw_files')}
            orphans = [p for p in parent.glob('*') if p.is_file() and str(p.resolve()) not in known]
            report['orphan_raw_files'] = len(orphans)
            if orphans:
                report['collection_complete'] = False
                self.journal.exception('raw_orphans', 'raw_files_without_manifest_not_completed', paths=[str(p.relative_to(self.root)) for p in orphans])
            if status == 'finished' and not report['collection_complete']:
                report['status'] = 'finished_with_unresolved'
        atomic_json(self.folder / 'progress.json', report)
        exception_count = sum(exception_counts.values())
        if (status not in {'finished', 'stopped'} and self.last_exception_count > 0 and
            self.clock() - self.last_exception_export < 30):
            return report
        temp = self.folder / ('exceptions.' + uuid.uuid4().hex + '.tmp')
        with temp.open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['key', 'category', 'receipt_no', 'window', 'reason', 'current_window_state', 'validation_policy', 'active_validation_window', 'active_failed', 'details'])
            for key, payload, window_state, policy in db.execute("""SELECT e.k,e.payload,w.state,json_extract(w.payload,'$.validation_policy') FROM exceptions e
                LEFT JOIN windows w ON w.id=json_extract(e.payload,'$.window') ORDER BY e.k"""):
                e = json.loads(payload)
                active = bool(window_state and (not self.validation_policy or policy == self.validation_policy))
                writer.writerow([key, e.get('category'), e.get('receipt_no'), e.get('window'), e.get('reason'),
                                 window_state, policy, active, active and window_state == 'failed', payload])
            stream.flush()
            os.fsync(stream.fileno())
        replace_retry(temp, self.folder / 'exceptions.csv')
        self.last_exception_export = self.clock()
        self.last_exception_count = exception_count
        return report


def main_collect():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--max-calls', type=int, default=18000)
    parser.add_argument('--phase', choices=('all', 'list', 'originals', 'review', 'targets'), default='list')
    parser.add_argument('--targets-dir', type=Path, help='--phase targets: a <run>/targets/<stamp> folder from kind_collect pairs targets')
    parser.add_argument('--targets-files', default=','.join(TARGET_FILES), help='Which lists of --targets-dir, e.g. pilot')
    parser.add_argument('--min-free-gb', type=float, default=4.0, help='Stop before a request when free disk space falls below this')
    parser.add_argument('--approved-plan', type=Path, help='Previously reviewed ready scope_review.json; never bypasses unresolved scope or capacity')
    parser.add_argument('--retry-failed', action='store_true',
                        help='Revalidate saved legacy lists and requeue failed intervals without resetting the run')
    parser.add_argument('--revalidate-lists', action='store_true', help='Append narrowed validation windows through 2015; preserve every legacy window and response')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test_collect()
        return
    if args.max_calls < 1:
        parser.error('max-calls must be positive')
    root = Path.cwd().resolve()
    folder = (root / args.run_dir).resolve()
    if not folder.is_relative_to((root / OUTPUT).resolve()) or folder.parent != (root / OUTPUT).resolve():
        parser.error('run-dir must be a direct child of outputs/collection/KIND_ORIGINAL_COLLECT')
    if args.phase == 'review':
        if (folder / 'RUNNING').exists() or (folder / 'state.sqlite3-wal').exists():
            parser.error('stop the KIND collector before read-only scope review')
        if not (folder / 'state.sqlite3').is_file():
            rows, _ = kind_source_observations(root)
            report, decisions, _ = make_scope_review(root, rows, False)
            if args.execute: report['review_output'] = str(save_scope_review(folder, report, decisions))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        db = sqlite3.connect((folder / 'state.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            rows = [json.loads(p) for p, in db.execute('SELECT payload FROM filings')]
            policy = db.execute("SELECT v FROM meta WHERE k='list_policy'").fetchone()
            incomplete = db.execute("SELECT 1 FROM windows WHERE json_extract(payload,'$.validation_policy')=? AND state IN ('pending','failed') LIMIT 1", (LIST_POLICY,)).fetchone()
            complete = bool(policy and int(policy[0]) == LIST_POLICY and incomplete is None)
            report, decisions, _ = make_scope_review(root, rows, complete)
        finally:
            db.close()
        if args.execute:
            report['review_output'] = str(save_scope_review(folder, report, decisions))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    targets = target_info = None
    if args.phase == 'targets':
        if not args.targets_dir:
            parser.error('--phase targets needs --targets-dir')
        chosen = (root / args.targets_dir).resolve()
        if chosen.parent != folder / 'targets' or not (chosen / 'summary.json').exists():
            parser.error('--targets-dir must be a saved <run-dir>/targets/<stamp> folder')
        names = tuple(n.strip() for n in args.targets_files.split(',') if n.strip())
        if not names or not set(names) <= set(TARGET_FILES):
            parser.error('--targets-files: comma list of ' + ','.join(TARGET_FILES))
        targets, target_info = load_targets(chosen, names)
        target_info['folder'] = chosen.relative_to(root).as_posix()
        print(json.dumps({'targets': target_info}, ensure_ascii=False, indent=2))
    config = {'schema': SCHEMA, 'receipt_start': RECEIPT_START, 'receipt_end': RECEIPT_END,
              'download_scope': SCOPE_RULE,
              'list_validation_policy': LIST_POLICY,
              'warmup_preceding_rule': WARMUP_PRECEDING_RULE,
              'raw_root': RAW.as_posix(),
              'target_period_end_years': [2000, 2014], 'market_filter': 'all', 'lastReport': '',
              'scope_policy': SCOPE_DESCRIPTION,
              'rate_limit': '40 request attempts per rolling 60 seconds across all KIND bulk endpoints and retries',
              'financial_parsing': False}
    print(json.dumps({'mode': 'execute' if args.execute else 'dry-run', 'run_dir': str(folder),
                      'initial_month_type_windows': 576, 'max_calls_this_invocation': args.max_calls,
                      **config}, ensure_ascii=False, indent=2))
    if not args.execute:
        print('No network or files written. Explicit execution resumes this same run directory; no automatic next batch.')
        return
    (root / OUTPUT).mkdir(parents=True, exist_ok=True)
    with exclusive(root / OUTPUT / 'ACTIVE'):
        folder.mkdir(exist_ok=True)
        with exclusive(folder / 'RUNNING'):
            run = folder / 'collection.json'
            if run.exists():
                previous = json.loads(run.read_text(encoding='utf-8'))
                if not compatible_run_config(previous['config'], config):
                    raise ValueError('immutable_run_configuration_changed')
            names, source_info = issuer_priority_sources(root)
            if run.exists() and previous['issuer_priority_sources'] != source_info:
                raise ValueError('issuer_priority_inputs_changed_create_a_separate_KIND_run')
            if not run.exists():
                with run.open('x', encoding='utf-8') as stream:
                    json.dump({'created_at': now(), 'config': config, 'issuer_priority_sources': source_info}, stream, ensure_ascii=False, indent=2)
            journal = Journal(folder)
            collector = Collector(root, folder, journal, names, args.max_calls)
            collector.min_free_bytes = int(args.min_free_gb * 1024**3)
            collector.targets = targets
            collector.approved_plan = (root / args.approved_plan).resolve() if args.approved_plan else None
            if collector.approved_plan and (not collector.approved_plan.is_relative_to(folder / 'scope_reviews') or collector.approved_plan.name != 'scope_review.json'):
                raise ValueError('approved_plan_outside_this_KIND_scope_reviews')
            status, reason = 'finished', None
            try:
                journal.append({'event': 'invocation_started', 'invocation': collector.invocation, 'phase': args.phase,
                                'max_calls': args.max_calls, 'effective_rate_limit': config['rate_limit'],
                                'warmup_preceding_rule': WARMUP_PRECEDING_RULE, 'min_free_bytes': collector.min_free_bytes,
                                **({'targets': target_info} if target_info else {})})
                collector.prepare_list_validation(allow_upgrade=args.revalidate_lists)
                if args.retry_failed:
                    collector.revalidate_legacy_list_cache()
                    journal.append({'event': 'retry_failed', 'validation_policy': LIST_POLICY})
                collector.checkpoint('initializing_cache')
                collector.seed_cache()
                collector.requeue_warmup_preceding()
                collector.collect(args.phase)
            except StopRun as error:
                status, reason = 'stopped', str(error)
            except KeyboardInterrupt:
                status, reason = 'stopped', 'ctrl_c'
            except Exception as error:
                status, reason = 'stopped', 'error_' + type(error).__name__
            finally:
                try:
                    journal.append({'event': 'invocation_finished', 'invocation': collector.invocation,
                                    'status': status, 'reason': reason, 'calls': collector.calls})
                    print(json.dumps(collector.checkpoint(status, reason), ensure_ascii=False, indent=2))
                finally:
                    collector.session.close()
                    journal.db.close()


def self_test_collect():
    import tempfile
    import unittest
    from unittest.mock import Mock, patch

    class Clock:
        def __init__(self):
            self.t = 1000.0

        def now(self):
            return self.t

        def sleep(self, seconds):
            self.t += seconds

    def row(receipt='20150331000001', title='사업보고서 (2014.12)'):
        return {'receipt_no': receipt, 'receipt_date': date(int(receipt[:4]), int(receipt[4:6]), int(receipt[6:8])).isoformat(),
                'report_kind': '사업보고서', 'report_name': title, 'company_name': 'fixture',
                'kind_company_code': '01234', 'market_label_observed': '유가증권', 'revision_mark_observed': '정정' in title,
                'attachment_addition_observed': '첨부추가' in title}

    def response(body, http=200):
        return Mock(status_code=http, headers={'Content-Type': 'text/html; charset=utf-8'},
                    iter_content=Mock(return_value=[body.encode()]), close=Mock())

    def listing(receipt, total=1):
        return f'''<p>총 {total} 건</p><table class="list"><tbody><tr><td>1</td><td>10:00</td>
        <td><img class="legend" alt="유가증권"><a onclick="companysummary_open('01234')">fixture</a></td>
        <td><a title="사업보고서 (2014.12)" onclick="openDisclsViewer('{receipt}','')">사업보고서</a></td><td>issuer</td></tr></tbody></table>'''

    class Tests(unittest.TestCase):
        def setUp(self):
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)
            self.folder = self.root / OUTPUT / 'fixture'
            self.folder.mkdir(parents=True)
            self.clock = Clock()
            self.journal = Journal(self.folder)
            self.addCleanup(self.journal.db.close)
            self.client = Collector(self.root, self.folder, self.journal, {}, 100,
                                    clock=self.clock.now, sleep=self.clock.sleep)
            self.addCleanup(self.client.session.close)

        def add_document(self, item, body=None):
            self.client.store_filing(item, 'fixture-list')
            receipt = item['receipt_no']
            cid = receipt[:-2] + '99'
            prefix = f'{BASE}/external/{receipt[:4]}/{receipt[4:6]}/{receipt[6:8]}/{receipt[8:]}/{cid}/'
            url = prefix + 'report.htm'
            responses = [response(f'<option value="{cid}|Y">current</option>'),
                         response(f"<script>parent.setPath('{prefix}report_toc.htm','{url}','');</script>"),
                         response(body or '<html><body>사업보고서<table><tr><td>회사의 개요</td></tr></table></body></html>')]
            with patch.object(self.client.session, 'request', side_effect=responses) as network:
                self.client.document(item)
                self.assertEqual(network.call_count, 3)
            return json.loads(self.journal.db.execute('SELECT detail FROM filings WHERE receipt=?', (receipt,)).fetchone()[0])

        def test_narrow_scope_period_market_identity_and_boundary_holds(self):
            self.assertEqual(scope_decision(row())[0], 'candidate')
            self.assertEqual(scope_decision(row('20151231000001', '사업보고서 (2015.12)'))[0], 'excluded')
            self.assertEqual(scope_decision(row('20160101000001'))[1], ['receipt_outside_scope'])
            self.assertEqual(scope_decision(row('20150331000001', '사업보고서'))[1], ['period_unknown_2015'])
            self.assertEqual(scope_decision(row('20050331000001', '사업보고서'))[0], 'hold')
            self.assertEqual(scope_decision(dict(row(), market_label_observed='코스닥'))[1], ['observed_KOSDAQ'])
            self.assertIn('market_unknown', scope_decision(dict(row(), market_label_observed=None))[1])
            self.assertIn('company_identity_unconfirmed', scope_decision(dict(row(), kind_company_code=None))[1])
            self.assertIn('historical_market_conflict', scope_decision(row(), {'01234': {'KOSPI', 'KOSDAQ'}})[1])
            self.assertEqual(scope_decision(row('20000331000001', '사업보고서 (1999.12)'))[0], 'candidate')
            self.assertEqual(scope_decision(row('20010331000001', '사업보고서 (1999.12)'))[0], 'excluded')
            migrated = row('19991231000001', '사업보고서 (1999.12)')
            migrated.update(receipt_date='2000-01-03', receipt_date_basis='explicit_list_cell')
            self.assertIn('receipt_date_boundary_conflict', scope_decision(migrated)[1])

        def test_scope_review_capacity_and_explicit_approval_cannot_bypass_holds(self):
            parent = self.root / 'outputs/collection/KIND_RESEARCH_AUDIT/20261101T233745597484'
            parent.mkdir(parents=True)
            records = []
            for i, size in enumerate((100, 200)):
                path = parent / f'{i}.html'
                path.write_bytes(b'x' * size)
                records.append({'sample_original_acquired': True, 'source_path': path.name, 'source_sha256': digest(path)})
            (parent / 'sources.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            report, decisions, _ = make_scope_review(self.root, [row()], list_complete=True)
            self.assertEqual(report['candidate_count'], 1)
            self.assertEqual(report['sample_count'], 2)
            self.assertEqual(report['sample_bytes_mean'], 150)
            self.assertEqual(report['sample_bytes_p90'], 200)
            target = save_scope_review(self.folder, report, decisions)
            with (target / 'holds.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            held_report, held_rows, _ = make_scope_review(self.root, [row('20150331000001', '사업보고서')])
            held_target = save_scope_review(self.folder, held_report, held_rows)
            with (held_target / 'holds.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(next(csv.DictReader(stream))['reasons'], 'period_unknown_2015')
            self.client.approved_plan = target / 'scope_review.json'
            with patch.object(self.client, 'scope_review', return_value=dict(report, bulk_ready=False)):
                with self.assertRaisesRegex(StopRun, 'scope_or_capacity_unresolved'):
                    self.client.authorize_originals()
            ready = dict(report, bulk_ready=True)
            self.client.approved_plan = None
            with patch.object(self.client, 'scope_review', return_value=ready):
                with self.assertRaisesRegex(StopRun, 'explicit_scope_capacity_review_required'):
                    self.client.authorize_originals()
            approved = self.folder / 'approved.json'
            approved.write_text(json.dumps(ready), encoding='utf-8')
            self.client.approved_plan = approved
            with patch.object(self.client, 'scope_review', return_value=ready):
                self.client.authorize_originals()
            self.assertFalse(decisions[0]['publication_date_verified'])

        def test_low_disk_reserve_prevents_any_http_attempt(self):
            with patch('shutil.disk_usage', return_value=Mock(free=1024**3)), patch.object(self.client.session, 'request') as network:
                with self.assertRaisesRegex(StopRun, 'disk_reserve_reached'):
                    self.client.fetch('GET', VIEWER)
                network.assert_not_called()
            self.assertEqual(self.client.calls, 0)

        def test_narrow_generation_never_schedules_2016_to_2021(self):
            previous = {'receipt_start': RECEIPT_START, 'receipt_end': '2021-12-31',
                        'scope_policy': LEGACY_SCOPE_DESCRIPTION,
                        'rate_limit': '40 request attempts per rolling 60 seconds across all KIND bulk endpoints and retries'}
            current = dict(previous, receipt_end=RECEIPT_END, scope_policy=SCOPE_DESCRIPTION,
                           download_scope=SCOPE_RULE, list_validation_policy=LIST_POLICY)
            self.assertTrue(compatible_run_config(previous, current))
            self.assertFalse(compatible_run_config(previous, dict(current, scope_policy='unapproved scope')))
            self.client.prepare_list_validation()
            active = [json.loads(p) for p, in self.journal.db.execute("SELECT payload FROM windows WHERE json_extract(payload,'$.validation_policy')=?", (LIST_POLICY,))]
            self.assertEqual(len(active), 576)
            self.assertEqual(max(w['end'] for w in active), '2015-12-31')
            self.assertTrue(all(w['start'] >= '2000-01-01' and w['end'] <= '2015-12-31' for w in active))
            with patch.object(self.client.session, 'request') as network:
                with self.assertRaisesRegex(ValueError, 'outside_narrowed_scope'):
                    self.client.fetch('POST', BASE + '/disclosure/details.do', data={'fromDate':'2016-01-01','toDate':'2016-01-31'}, label='list')
                network.assert_not_called()

        def test_combined_regular_titles_are_not_forced_to_first_kind(self):
            combined = report_title_metadata('[기재정정]사업보고서ㆍ반기보고서ㆍ분기보고서(외국기업의 경우)')
            self.assertIsNone(combined['report_kind'])
            self.assertEqual(combined['report_kind_candidates'], list(KINDS))
            self.assertEqual(combined['report_class'], 'ambiguous_regular')
            self.assertEqual(report_title_metadata('관리종목지정(사업보고서미제출)')['report_class'], 'non_regular')
            related = report_title_metadata('해외증권거래소등에신고한사업보고서등의국내신고')
            self.assertEqual(related['report_class'], 'regular_related_unconfirmed')
            self.assertIsNone(related['report_kind'])
            unknown = row(title='사업보고서ㆍ반기보고서ㆍ분기보고서(외국기업의 경우) (2014.12)')
            detail = self.add_document(unknown)
            self.assertTrue(detail['original_acquired'])
            self.assertEqual(detail['period_status'], 'title_label_only')

        def test_mixed_search_counts_all_rows_but_excludes_market_notices(self):
            def tr(receipt, title):
                return re.search(r'<tbody>(.*?)</tbody>', listing(receipt, 3), re.S)[1].replace('사업보고서', title)
            first = '<p>전체 3건</p><table class="list"><tbody>' + tr('20150315000001', '사업보고서ㆍ반기보고서ㆍ분기보고서(외국기업의 경우)') + tr('20150315000002', '관리종목지정(사업보고서미제출)') + '</tbody></table>'
            w = next(months(date(2015, 3, 15), date(2015, 3, 15)))
            with patch.object(self.client, 'fetch', side_effect=[first, listing('20150315000003', 3)]):
                self.client.collect_window(w)
            self.assertEqual(w['state'], 'done')
            self.assertEqual(len(w['seen']), 3)
            self.assertEqual(self.journal.db.execute('SELECT count(*) FROM filings').fetchone()[0], 3)
            self.assertEqual(self.journal.db.execute("SELECT count(*) FROM filings WHERE doc_state='excluded_non_regular'").fetchone()[0], 1)
            self.assertEqual(self.journal.db.execute('SELECT sum(eligible) FROM filing_policy').fetchone()[0], 2)
            with patch.object(self.client.session, 'request') as network:
                self.client.document(row('20150315000002', '관리종목지정(사업보고서미제출)'))
                network.assert_not_called()

        def test_explicit_last_page_does_not_request_clamped_page_beyond_end(self):
            def page(number):
                rows = ''.join(re.search(r'<tbody>(.*?)</tbody>', listing(f'20150315{i:06d}', 200), re.S)[1]
                               for i in range((number - 1) * 100 + 1, number * 100 + 1))
                return ('<p>전체 200건</p><table class="list"><tbody>' + rows + '</tbody></table>' +
                        f'<section class="paging-group"><div class="info">전체 200건 : <strong>{number}</strong>/2' +
                        '<select name="currentPageSize"><option selected="selected" value="100">100건</option></select></div></section>')
            w = next(months(date(2015, 3, 15), date(2015, 3, 15)))
            with patch.object(self.client, 'fetch', side_effect=[page(1), page(2)]) as fetch:
                self.client.collect_window(w)
                self.assertEqual(fetch.call_count, 2)
            self.assertEqual(w['state'], 'done')
            self.assertEqual(len(w['seen']), 200)
            wrong = dict(w, id='wrong-page', state='pending', page=3)
            with patch.object(self.client, 'fetch', return_value=page(2)):
                self.client.collect_window(wrong)
            self.assertEqual(wrong['state'], 'failed')

        def test_validation_generation_preserves_legacy_and_rechecks_done_windows(self):
            originals = {}
            for state in ('done', 'empty_unconfirmed', 'failed', 'split', 'pending'):
                w = dict(next(months(date(2015, 3, 1), date(2015, 3, 31))), id='legacy-' + state, state=state)
                self.journal.append({'event': 'window', 'window': w})
                originals[w['id']] = json.dumps(w, ensure_ascii=False)
            # Simulate a filing indexed by the old substring-only parser.
            item = row(title='기타시장안내(사업보고서미제출)')
            self.journal.append({'event': 'filing', 'filing': item, 'priority': 1})
            history = self.journal.path.read_bytes()
            with self.assertRaisesRegex(StopRun, 'requires_--revalidate-lists'):
                self.client.prepare_list_validation()
            self.client.prepare_list_validation(allow_upgrade=True)
            self.assertTrue(self.journal.path.read_bytes().startswith(history))
            for identity, payload in originals.items():
                self.assertEqual(self.journal.db.execute('SELECT payload FROM windows WHERE id=?', (identity,)).fetchone()[0], payload)
            self.assertEqual(self.journal.db.execute("SELECT count(*) FROM windows WHERE json_extract(payload,'$.validation_policy')=?", (LIST_POLICY,)).fetchone()[0], 576)
            self.assertEqual(self.journal.db.execute('SELECT doc_state FROM filings').fetchone()[0], 'excluded_non_regular')
            self.journal.append({'event': 'retry_failed', 'validation_policy': LIST_POLICY})
            self.assertEqual(self.journal.db.execute("SELECT state FROM windows WHERE id='legacy-failed'").fetchone()[0], 'failed')
            self.client.prepare_list_validation(allow_upgrade=True)
            self.assertEqual(self.journal.db.execute('SELECT count(*) FROM windows').fetchone()[0], 581)
            report = self.client.checkpoint('running_lists')
            self.assertEqual(report['list_windows'], {'pending': 576})
            self.assertEqual(report['legacy_list_windows']['done'], 1)
            self.assertFalse(report['list_collection_complete'])
            self.journal.exception('window:legacy-failed', 'list_integrity', window='legacy-failed', reason='old_failure')
            self.client.checkpoint('stopped', 'test')
            with (self.folder / 'exceptions.csv').open(encoding='utf-8-sig') as stream:
                old = next(csv.DictReader(stream))
            self.assertEqual(old['current_window_state'], 'failed')
            self.assertEqual(old['active_failed'], 'False')

        def test_empty_query_cannot_contradict_combined_title_with_unknown_subtype(self):
            item = row('20150315000001', '사업보고서ㆍ반기보고서ㆍ분기보고서(외국기업의 경우)')
            self.client.store_filing(item, 'preserved')
            w = list(months(date(2015, 3, 15), date(2015, 3, 15)))[1]
            empty = '<p>전체 0건</p><table class="list"><tbody><tr><td>조회된 결과가 없습니다.</td></tr></tbody></table>'
            with patch.object(self.client, 'fetch', return_value=empty):
                self.client.collect_window(w)
            self.assertEqual(w['state'], 'failed')
            reason = json.loads(self.journal.db.execute('SELECT payload FROM exceptions').fetchone()[0])['reason']
            self.assertEqual(reason, 'empty_result_contradicts_preserved_filing')

        def test_original_phase_is_blocked_until_active_validation_finishes(self):
            self.client.prepare_list_validation()
            with patch.object(self.client, 'collect_originals') as originals:
                with self.assertRaisesRegex(StopRun, 'list_validation_incomplete'):
                    self.client.collect('originals')
                originals.assert_not_called()

        def test_saved_kind_mismatch_cases_are_combined_titles_and_repeat_is_clamp(self):
            root = Path(__file__).resolve().parents[2]
            source = root / DEFAULT_RUN / 'manifest.jsonl'
            if not source.exists():
                self.skipTest('saved KIND run not available')
            windows, reasons, responses = {}, {}, {}
            with source.open(encoding='utf-8') as stream:
                for line in stream:
                    e = json.loads(line)
                    if e['event'] == 'window':
                        windows[e['window']['id']] = e['window']
                    elif e['event'] == 'exception' and e.get('category') == 'list_integrity':
                        reasons[e['window']] = e['reason']
                    elif e['event'] == 'response':
                        responses[e['key']] = e
                    elif e['event'] == 'cache_seed':
                        responses.setdefault(e['key'], e['response'])
            checked = clamped = 0
            for identity, w in windows.items():
                if w['state'] != 'failed' or reasons.get(identity) not in {'server_ignored_report_kind', 'duplicate_or_repeated_page'}:
                    continue
                form = search_form(w['end'], w['page'])
                form.update(fromDate=w['start'], reportNm=w['kind'], lastReport='', marketType='')
                record = responses[request_key('POST', BASE + '/disclosure/details.do', form)]
                path = root / record['path']
                self.assertEqual(digest(path), record['sha256'])
                rows, meta = parse_search(decode(path.read_bytes(), record.get('content_type', ''))[0], w['end'])
                if reasons[identity] == 'server_ignored_report_kind':
                    combined = [r for r in rows if r['report_class'] == 'ambiguous_regular']
                    self.assertTrue(combined)
                    self.assertTrue(all(w['kind'] in r['report_kind_candidates'] for r in combined))
                    self.assertTrue(all(w['kind'] in compact['report_name'].replace(' ', '') for compact in rows))
                    checked += 1
                else:
                    self.assertLess(meta['current_page'], w['page'])
                    self.assertEqual(meta['current_page'], meta['total_pages'])
                    self.assertEqual(len(set(w['seen'])), meta['total_reported'])
                    clamped += 1
            self.assertGreater(checked, 0)
            self.assertGreater(clamped, 0)

        def test_warmup_preceding_scope_is_narrow_and_dates_remain_unverified(self):
            item = row('20000331000001', '사업보고서 (1999.12)')
            for kind in KINDS:
                self.assertTrue(warmup_preceding(dict(item, report_kind=kind), '1999-12'))
            self.assertFalse(warmup_preceding(row('20010331000001'), '1999-12'))
            self.assertFalse(warmup_preceding(item, '1999-11'))
            self.assertFalse(warmup_preceding(dict(item, report_kind='other'), '1999-12'))
            item.update(receipt_date='2000-04-01', receipt_date_basis='explicit_list_cell')
            detail = self.add_document(item, '<html><body>사업보고서 사업연도 1999년 1월 1일부터 1999년 12월 31일까지<table><tr><td>회사의 개요</td></tr></table></body></html>')
            self.assertTrue(detail['original_acquired'])
            self.assertEqual(detail['research_period_scope'], 'warmup_preceding')
            self.assertEqual(detail['download_period_intent'], 'warmup_preceding')
            self.assertTrue(detail['date_conflict'])
            self.assertEqual(detail['date_status'], 'date_unverified')
            self.assertFalse(detail['publication_date_verified'])

        def test_previously_excluded_warmup_receipts_are_requeued_without_reset(self):
            included = row('20000331000001', '사업보고서 (1999.12)')
            excluded = row('20010331000001', '사업보고서 (1999.12)')
            for item in (included, excluded):
                self.client.store_filing(item, 'fixture')
                self.client.log({'event': 'document', 'receipt_no': item['receipt_no'],
                                 'state': 'excluded_title_period', 'original_acquired': False})
            before = self.journal.path.read_bytes()
            with patch.object(self.client.session, 'request') as network:
                self.client.requeue_warmup_preceding()
                self.client.requeue_warmup_preceding()
                network.assert_not_called()
            states = dict(self.journal.db.execute('SELECT receipt,doc_state FROM filings'))
            self.assertEqual(states[included['receipt_no']], 'pending')
            self.assertEqual(states[excluded['receipt_no']], 'excluded_title_period')
            self.assertTrue(self.journal.path.read_bytes().startswith(before))
            self.assertEqual(self.journal.path.read_text(encoding='utf-8').count('"event": "warmup_preceding_requeued"'), 1)

        def test_unknown_period_is_held_but_existing_original_is_preserved(self):
            item = row(title='사업보고서(일반법인)')
            self.client.store_filing(item, 'fixture')
            with patch.object(self.client.session, 'request') as network:
                self.client.document(item)
                network.assert_not_called()
            held = json.loads(self.journal.db.execute('SELECT detail FROM filings').fetchone()[0])
            self.assertEqual(held['state'], 'hold_scope')
            self.assertIn('period_unknown_2015', held['scope_reasons'])
            # A previously downloaded file stays preserved, even if now held/excluded.
            known = row('20150331000002')
            detail = self.add_document(known)
            path = self.root / detail['source_path']
            original = path.read_bytes()
            known['report_name'] = '사업보고서(일반법인)'
            self.client.store_filing(known, 'later_observation')
            with patch.object(self.client.session, 'request') as network:
                self.client.document(known)
                network.assert_not_called()
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(self.client.calls, 3)

        def test_mid_document_budget_stop_resumes_cached_steps_only(self):
            item = row()
            self.client.store_filing(item, 'fixture')
            receipt, cid = item['receipt_no'], '20150331000099'
            prefix = f'{BASE}/external/2015/03/31/000001/{cid}/'
            self.client.max_calls = 2
            with patch.object(self.client.session, 'request', side_effect=[
                    response(f'<option value="{cid}|Y">current</option>'),
                    response(f"<script>parent.setPath('{prefix}report_toc.htm','{prefix}report.htm','');</script>")]):
                with self.assertRaisesRegex(StopRun, 'max_calls'):
                    self.client.document(item)
            before = self.journal.path.read_bytes()
            resumed = Collector(self.root, self.folder, self.journal, {}, 1,
                                clock=self.clock.now, sleep=self.clock.sleep)
            self.addCleanup(resumed.session.close)
            with patch.object(resumed.session, 'request', return_value=response(
                    '<html><body>사업보고서<table><tr><td>회사의 개요</td></tr></table></body></html>')) as network:
                resumed.document(item)
                self.assertEqual(network.call_count, 1)
            self.assertEqual(resumed.calls, 1)
            self.assertEqual(resumed.cache_hits, 2)
            self.assertTrue(self.journal.path.read_bytes().startswith(before))
            self.assertEqual(self.journal.db.execute('SELECT doc_state FROM filings').fetchone()[0], 'original_saved')

        def test_observed_older_content_requires_matching_receipt_path(self):
            item = row()
            self.client.store_filing(item, 'fixture')
            cid, other = '20150331000099', '20160331000099'
            prefix = f'{BASE}/external/2015/03/31/000001/{cid}/'
            bad = f'{BASE}/external/2016/03/31/000001/{other}/wrong.htm'
            with patch.object(self.client.session, 'request', side_effect=[
                    response(f'<option value="{other}|Y">latest</option><option value="{cid}|N">older</option>'),
                    response(f"<script>parent.setPath('','{bad}','');</script>"),
                    response(f"<script>parent.setPath('{prefix}report_toc.htm','{prefix}report.htm','');</script>"),
                    response('<html><body>사업보고서<table><tr><td>회사의 개요</td></tr></table></body></html>')]) as network:
                self.client.document(item)
                self.assertEqual(network.call_count, 4)
            detail = json.loads(self.journal.db.execute('SELECT detail FROM filings').fetchone()[0])
            self.assertTrue(detail['original_acquired'])
            self.assertEqual(detail['content_id_flag'], 'N')
            self.assertEqual(detail['content_id'], cid)
            self.assertFalse(detail['version_identity_verified'])

        def test_late_2014_reports_and_old_period_corrections_are_not_excluded(self):
            body = '<html><body>사업보고서 사업연도 2014년 1월 1일부터 2014년 12월 31일까지<table><tr><td>회사의 개요</td></tr></table></body></html>'
            for receipt, title in [('20150331000001', '사업보고서 (2014.12)')]:
                detail = self.add_document(row(receipt, title), body)
                self.assertEqual(detail['research_period_scope'], 'in_scope')
                self.assertEqual(detail['period_end'], '2014-12-31')
            excluded = row('20160331000003', '사업보고서 (2015.12)')
            self.client.store_filing(excluded, 'list')
            with patch.object(self.client.session, 'request') as network:
                self.client.document(excluded)
                late = row('20190331000002', '[기재정정]사업보고서 (2014.12)')
                self.client.store_filing(late, 'late_outside_receipt_scope')
                self.client.document(late)
                network.assert_not_called()

        def test_period_conflict_does_not_discard_successful_original(self):
            body = '<html><body>사업보고서 사업연도 2015년 1월 1일부터 2015년 12월 31일까지<table><tr><td>회사의 개요</td></tr></table></body></html>'
            detail = self.add_document(row(), body)
            self.assertTrue(detail['original_acquired'])
            self.assertEqual(detail['period_status'], 'cover_title_conflict')
            self.assertTrue((self.root / detail['source_path']).is_file())

        def test_rate_40_rolling_seconds_and_restart_share_reservations(self):
            starts = []
            def reply(*args, **kwargs):
                starts.append(self.clock.now())
                self.assertFalse(kwargs['allow_redirects'])
                return response('<html><body>fixture</body></html>')
            with patch.object(self.client.session, 'request', side_effect=reply):
                for i in range(41):
                    self.client.fetch('GET', VIEWER + '?test=' + str(i))
            self.assertEqual(self.client.calls, 41)
            self.assertGreaterEqual(starts[-1] - starts[0], 60)
            self.assertTrue(all(sum(t - 60 < s <= t for s in starts) <= 40 for t in starts))
            other = Collector(self.root, self.folder, self.journal, {}, 1, clock=self.clock.now, sleep=self.clock.sleep)
            self.addCleanup(other.session.close)
            last = self.clock.now()
            with patch.object(other.session, 'request', return_value=response('<html>fixture</html>')):
                other.fetch('GET', VIEWER + '?test=after-restart')
            self.assertGreaterEqual(self.clock.now() - last, 1.5)

        def test_budget_includes_timeouts_and_STOP_prevents_requests(self):
            self.client.max_calls = 2
            with patch.object(self.client.session, 'request', side_effect=requests.Timeout('fixture')) as network:
                with self.assertRaisesRegex(StopRun, 'max_calls'):
                    self.client.fetch('GET', VIEWER)
                self.assertEqual(network.call_count, 2)
            (self.folder / 'STOP').touch()
            with patch.object(self.client.session, 'request') as network:
                with self.assertRaisesRegex(StopRun, 'stop_file'):
                    self.client.fetch('GET', VIEWER)
                network.assert_not_called()

        def test_ctrl_c_keeps_request_journal_and_stopped_checkpoint(self):
            with patch.object(self.client.session, 'request', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.client.fetch('GET', VIEWER)
            self.client.log({'event': 'invocation_finished', 'status': 'stopped', 'reason': 'ctrl_c'})
            report = self.client.checkpoint('stopped', 'ctrl_c')
            self.assertEqual(report['calls_this_invocation'], 1)
            self.assertEqual(report['unfinished_request_records'], 1)
            self.assertFalse(report['collection_complete'])
            self.assertEqual(json.loads((self.folder / 'progress.json').read_text(encoding='utf-8'))['stop_reason'], 'ctrl_c')

        def test_blocked_429_empty_redirect_and_invalid_body_are_not_success(self):
            for http, body in [(403, '<html>forbidden</html>'), (429, '<html>limited</html>'), (200, '<html>Access denied</html>'), (302, '')]:
                with patch.object(self.client.session, 'request', return_value=response(body, http)) as network:
                    with self.assertRaises(StopRun):
                        self.client.fetch('GET', VIEWER + '?error=' + str(http))
                    self.assertEqual(network.call_count, 1)
            with patch.object(self.client.session, 'request', return_value=response('')):
                with self.assertRaises(ValueError):
                    self.client.fetch('GET', VIEWER + '?empty')
            self.assertFalse(classify_html('<html>login</html>', row(), VIEWER)['html_original_accessible'])

        def test_rate_change_resumes_old_config_without_relaxing_other_contracts(self):
            old = {'raw_root': 'data/raw/kind', 'rate_limit': '30 request attempts per rolling 60 seconds across all KIND bulk endpoints and retries'}
            current = dict(old, rate_limit=old['rate_limit'].replace('30 ', '40 ', 1))
            self.assertTrue(compatible_run_config(old, current))
            self.assertFalse(compatible_run_config(old, dict(current, raw_root='elsewhere')))
            self.assertFalse(compatible_run_config(dict(old, rate_limit='unknown'), current))
            self.assertTrue(compatible_run_config(old, dict(current, warmup_preceding_rule=WARMUP_PRECEDING_RULE)))
            self.assertFalse(compatible_run_config(old, dict(current, warmup_preceding_rule={'report_period': '1998-12'})))

        def test_explicit_date_conflict_is_preserved_without_discarding_filing(self):
            body = listing('20150315000001').replace('<td>10:00</td>', '<td>2015-03-16 10:00</td>')
            rows, _ = parse_search(body, '2015-03-31')
            self.assertEqual(rows[0]['receipt_date'], '2015-03-16')
            w = next(months(date(2015, 3, 16), date(2015, 3, 16)))
            with patch.object(self.client, 'fetch', return_value=body):
                self.client.collect_window(w)
            self.assertEqual(w['state'], 'done')
            stored = json.loads(self.journal.db.execute('SELECT payload FROM filings').fetchone()[0])
            self.assertEqual(stored['list_display_date'], '2015-03-16')
            self.assertEqual(stored['receipt_id_date'], '2015-03-15')
            self.assertEqual(stored['receipt_no'], '20150315000001')
            self.assertTrue(stored['date_conflict'])
            self.assertFalse(stored['receipt_prefix_matches_query'])
            self.assertEqual(stored['date_status'], 'date_unverified')
            self.assertIsNone(stored['publication_date'])
            detail = self.add_document(stored)
            self.assertTrue(detail['original_url_binding_verified'])
            self.assertTrue(detail['original_acquired'])
            self.assertTrue(detail['date_conflict'])
            self.assertFalse(detail['publication_date_verified'])

        def test_date_conflict_does_not_skip_later_pages_or_fake_completion(self):
            first = listing('20150315000001', 2).replace('<td>10:00</td>', '<td>2015-03-16 10:00</td>')
            w = next(months(date(2015, 3, 16), date(2015, 3, 16)))
            self.journal.append({'event': 'window', 'window': w})
            with patch.object(self.client, 'fetch', side_effect=[first, StopRun('max_calls')]):
                with self.assertRaises(StopRun):
                    self.client.collect_window(w)
            self.assertEqual(w['state'], 'pending')
            self.assertEqual(w['page'], 2)
            self.assertEqual(w['seen'], ['20150315000001'])
            with patch.object(self.client, 'fetch', return_value=listing('20150316000002', 2)) as fetch:
                self.client.collect_window(w)
                self.assertEqual(fetch.call_args.kwargs['data']['pageIndex'], '2')
            self.assertEqual(w['state'], 'done')
            self.assertEqual(self.journal.db.execute('SELECT count(*) FROM filings').fetchone()[0], 2)

        def test_same_receipt_date_observations_are_not_security_identity_conflict(self):
            item = row()
            self.client.store_filing(dict(item), 'prefix-only')
            item.update(receipt_date='2015-04-01', receipt_date_basis='explicit_list_cell')
            self.client.store_filing(item, 'list-displayed')
            state, payload = self.journal.db.execute('SELECT doc_state,payload FROM filings').fetchone()
            stored = json.loads(payload)
            self.assertEqual(state, 'pending')
            self.assertFalse(stored['identity_conflict'])
            self.assertEqual(len(stored['date_observations']), 2)
            self.assertTrue(stored['date_conflict'])
            self.assertFalse(stored['publication_date_verified'])

        def test_legacy_receipts_titles_and_invalid_identifiers(self):
            legacy = '20150315K00001'
            body = listing('20150315000001').replace('20150315000001', legacy)
            body = body.replace('title="사업보고서 (2014.12)"', 'title="사업보고서"')
            body = body.replace('>사업보고서</a>', '>사업보고서<font>(2014.12)</font></a>')
            rows, meta = parse_search(body, '2015-03-15')
            self.assertEqual(rows[0]['receipt_no'], legacy)
            self.assertEqual(kind_receipt_date(legacy), '2015-03-15')
            self.assertEqual(meta['total_reported'], 1)
            self.assertIn('(2014.12)', rows[0]['report_name'])
            for value in ('20150315', legacy + '9', '20150230K00001', "20150315K00001');bad()"):
                self.assertIsNone(kind_receipt_date(value))
            options = content_options(f'<option value="{legacy}|Y">observed</option>')
            self.assertEqual(options[0]['content_id'], legacy)
            url = f'{BASE}/external/2015/03/15/K00001/{legacy}/report.htm'
            selected, _, _ = select_body_url([url], legacy, legacy)
            self.assertEqual(selected, url)

        def test_failed_legacy_window_retry_reuses_original_and_retains_history(self):
            w = next(months(date(2015, 3, 1), date(2015, 3, 31)))
            form = search_form(w['end'], 1)
            form.update(fromDate=w['start'], reportNm=w['kind'], lastReport='', marketType='')
            body = listing('20150315000001').replace('20150315000001', '20150315K00001')
            with patch.object(self.client.session, 'request', return_value=response(body)):
                self.client.fetch('POST', BASE + '/disclosure/details.do', data=form, label='list')
            source = self.root / self.client.last_response['path']
            before_hash = digest(source)
            self.client.invalidate()  # State left by the old numeric-only validator.
            w['state'] = 'failed'
            self.journal.append({'event': 'window', 'window': w})
            self.journal.exception('window:' + w['id'], 'list_integrity', window=w['id'], reason='unidentifiable_receipt')
            report = self.client.checkpoint('running_lists')
            self.assertEqual(report['active_list_failure_reasons'], {'unidentifiable_receipt': 1})
            with (self.folder / 'exceptions.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            # A plain resume does not turn failed intervals into completed intervals.
            with patch.object(self.client.session, 'request') as network:
                self.client.collect_lists()
                network.assert_not_called()
            self.assertFalse(self.client.checkpoint('running_lists')['list_collection_complete'])
            history = self.journal.path.read_bytes()
            self.client.revalidate_legacy_list_cache()
            self.journal.append({'event': 'retry_failed'})
            with patch.object(self.client.session, 'request') as network:
                self.client.collect_lists()
                network.assert_not_called()
            self.assertEqual(self.journal.db.execute('SELECT state FROM windows').fetchone()[0], 'done')
            self.assertEqual(self.journal.db.execute('SELECT receipt FROM filings').fetchone()[0], '20150315K00001')
            self.assertEqual(digest(source), before_hash)
            self.assertTrue(self.journal.path.read_bytes().startswith(history))
            report = self.client.checkpoint('finished')
            self.assertEqual(report['active_list_failure_reasons'], {})
            with (self.folder / 'exceptions.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(next(csv.DictReader(stream))['active_failed'], 'False')

        def test_empty_and_unreadable_receipts_are_not_equivalent(self):
            malformed = listing('20150315000001').replace('20150315000001', 'unknown-id')
            w = next(months(date(2015, 3, 15), date(2015, 3, 15)))
            self.journal.append({'event': 'window', 'window': w})
            with patch.object(self.client, 'fetch', return_value=malformed):
                self.client.collect_window(w)
            self.assertEqual(w['state'], 'failed')
            empty = '<p>전체 0건</p><table class="list"><tbody><tr><td>조회된 결과가 없습니다.</td></tr></tbody></table>'
            w = dict(w, id='empty-test', state='pending', seen=[], expected=None)
            with patch.object(self.client, 'fetch', return_value=empty):
                self.client.collect_window(w)
            self.assertEqual(w['state'], 'empty_unconfirmed')
            with self.assertRaises(ValueError):
                parse_search('<html>검색 오류</html>', '2015-03-15')

        def test_all_phase_transitions_from_lists_to_originals(self):
            w = next(months(date(2015, 3, 1), date(2015, 3, 31)))
            self.journal.append({'event': 'window', 'window': w})
            receipt, cid = '20150315000001', '20150315000099'
            prefix = f'{BASE}/external/2015/03/15/000001/{cid}/'
            self.client.max_calls = 4
            with patch.object(self.client.session, 'request', side_effect=[
                    response(listing(receipt)), response(f'<option value="{cid}|Y">current</option>'),
                    response(f"<script>parent.setPath('{prefix}report_toc.htm','{prefix}report.htm','');</script>"),
                    response('<html><body>사업보고서<table><tr><td>회사의 개요</td></tr></table></body></html>')]), patch.object(self.client, 'authorize_originals'):
                self.client.collect('all')
            report = self.client.checkpoint('finished')
            self.assertEqual(report['current_stage'], 'originals')
            self.assertTrue(report['list_collection_complete'])
            self.assertEqual(report['calls_by_stage_this_invocation'], {'lists': 1, 'originals': 3, 'other': 0})
            self.assertEqual(report['originals_saved'], 1)
            self.assertTrue(report['collection_complete'])
            stored = json.loads(self.journal.db.execute('SELECT detail FROM filings').fetchone()[0])
            self.assertTrue(stored['source_path'].startswith('data/raw/kind/'))

        def test_lists_can_exhaust_shared_budget_before_original_download(self):
            w = next(months(date(2015, 3, 1), date(2015, 3, 31)))
            self.journal.append({'event': 'window', 'window': w})
            self.client.max_calls = 18000
            self.client.calls = 17999  # Last remaining slot of the same invocation.
            self.client.calls_by_stage['lists'] = 17999
            with patch.object(self.client.session, 'request', return_value=response(listing('20150315000001'))) as network, patch.object(self.client, 'authorize_originals'):
                with self.assertRaisesRegex(StopRun, 'max_calls'):
                    self.client.collect('all')
                self.assertEqual(network.call_count, 1)
            report = self.client.checkpoint('stopped', 'max_calls')
            self.assertEqual(report['calls_this_invocation'], 18000)
            self.assertEqual(report['calls_by_stage_this_invocation']['originals'], 0)
            self.assertTrue(report['list_collection_complete'])
            self.assertEqual(report['originals_saved'], 0)
            self.assertEqual(report['originals_pending'], 1)

        def test_list_pages_resume_without_restarting_completed_page(self):
            w = next(months(date(2015, 3, 1), date(2015, 3, 31)))
            self.journal.append({'event': 'window', 'window': w})
            with patch.object(self.client, 'fetch', side_effect=[listing('20150315000001', 2), StopRun('max_calls')]):
                with self.assertRaises(StopRun):
                    self.client.collect_window(w)
            stored = json.loads(self.journal.db.execute('SELECT payload FROM windows WHERE id=?', (w['id'],)).fetchone()[0])
            self.assertEqual(stored['page'], 2)
            with patch.object(self.client, 'fetch', return_value=listing('20150316000002', 2)) as fetch:
                self.client.collect_window(stored)
                self.assertEqual(fetch.call_count, 1)
                self.assertEqual(fetch.call_args.kwargs['data']['pageIndex'], '2')
            self.assertEqual(self.journal.db.execute('SELECT count(*) FROM filings').fetchone()[0], 2)
            self.assertEqual(self.journal.db.execute('SELECT state FROM windows WHERE id=?', (w['id'],)).fetchone()[0], 'done')

        def test_repeated_page_splits_and_daily_failure_is_not_complete(self):
            w = next(months(date(2015, 3, 1), date(2015, 3, 31)))
            self.journal.append({'event': 'window', 'window': w})
            with patch.object(self.client, 'fetch', return_value=listing('20150315000001', 2)):
                self.client.collect_window(w)
            self.assertEqual(w['state'], 'split')
            self.assertEqual(self.journal.db.execute("SELECT count(*) FROM windows WHERE state='pending'").fetchone()[0], 2)
            daily = dict(w, id='daily', start='2015-03-15', end='2015-03-15', page=1, seen=[], expected=None, state='pending')
            with patch.object(self.client, 'fetch', return_value=listing('20150315000001', 2)):
                self.client.collect_window(daily)
            self.assertEqual(daily['state'], 'failed')

        def test_journal_replays_unindexed_tail_and_windows_lock_preserves_checkpoint(self):
            item = row()
            event = {'event': 'filing', 'filing': item, 'priority': 1}
            with self.journal.path.open('ab') as stream:
                stream.write((json.dumps(event) + '\n').encode())
            resumed = Journal(self.folder)
            self.addCleanup(resumed.db.close)
            self.assertEqual(resumed.db.execute('SELECT count(*) FROM filings').fetchone()[0], 1)
            checkpoint = self.folder / 'progress.json'
            atomic_json(checkpoint, {'status': 'old'})
            before = checkpoint.read_bytes()
            with patch.object(Path, 'replace', side_effect=PermissionError), patch('time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_json(checkpoint, {'status': 'new'})
            self.assertEqual(checkpoint.read_bytes(), before)

        def test_existing_cache_hash_failure_is_not_redownloaded(self):
            detail = self.add_document(row())
            (self.root / detail['source_path']).write_bytes(b'corrupt fixture')
            with patch.object(self.client.session, 'request') as network:
                self.client.collect_originals()
                network.assert_not_called()
            self.assertEqual(self.journal.db.execute('SELECT doc_state FROM filings').fetchone()[0], 'integrity_error')

        def test_original_versions_are_retained_and_unknown_issuers_not_excluded(self):
            for i, tag in enumerate(('', '[기재정정]', '[첨부정정]', '[첨부추가]'), 1):
                self.client.store_filing(row(f'20150331{i:06d}', tag + '사업보고서 (2014.12)'), 'fixture')
            self.assertEqual(self.journal.db.execute('SELECT count(*) FROM filings').fetchone()[0], 4)
            self.assertTrue(all(json.loads(p)['issuer_link_status']=='historical_issuer_unresolved' for p, in self.journal.db.execute('SELECT payload FROM filings')))

        def test_saved_bulk_list_responses_identify_all_observed_receipts(self):
            root = Path(__file__).resolve().parents[2]
            manifest = root / DEFAULT_RUN / 'manifest.jsonl'
            if not manifest.exists():
                self.skipTest('saved bulk lists not available')
            checked, legacy, empty = 0, 0, 0
            with manifest.open(encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line)
                    if record['event'] != 'response' or record.get('label') != 'list':
                        continue
                    path = (root / record['path']).resolve()
                    self.assertTrue(path.is_relative_to((root / RAW).resolve()))
                    self.assertEqual(digest(path), record['sha256'])
                    rows, meta = parse_search(decode(path.read_bytes(), record.get('content_type', ''))[0], record['params']['toDate'])
                    self.assertEqual(meta['unparsed_rows'], 0)
                    self.assertIsNotNone(meta['total_reported'])
                    self.assertGreaterEqual(meta['total_reported'], len(rows))
                    if not rows:
                        empty += 1
                        self.assertEqual(meta['total_reported'], 0)
                    for row in rows:
                        day = kind_receipt_date(row['receipt_no'])
                        self.assertIsNotNone(day)
                        # A real list date can disagree with the ID prefix. Preserve
                        # that evidence; collect_window must reject, never overwrite it.
                        self.assertRegex(row['receipt_date'], r'^\d{4}-\d{2}-\d{2}$')
                        observed = row['receipt_date'] if row['receipt_date_basis'] == 'explicit_list_cell' else day
                        self.assertTrue(record['params']['fromDate'] <= observed <= record['params']['toDate'])
                        legacy += bool(re.fullmatch(r'\d{8}[A-Za-z]\d{5}', row['receipt_no']))
                    checked += 1
            self.assertGreater(checked, 0)
            self.assertGreater(legacy, 0)
            self.assertGreater(empty, 0)

        def test_actual_20_preserved_KIND_originals_validate_offline(self):
            root = Path(__file__).resolve().parents[2]
            parent = root / 'outputs/collection/KIND_RESEARCH_AUDIT/20261101T233745597484'
            if not (parent / 'sources.jsonl').exists():
                self.skipTest('saved KIND originals not available')
            records = [json.loads(line) for line in (parent / 'sources.jsonl').read_text(encoding='utf-8').splitlines()]
            count = 0
            for item in records:
                if not item.get('sample_original_acquired'):
                    continue
                path = parent / item['source_path']
                before = digest(path)
                self.assertEqual(before, item['source_sha256'])
                html = decode(path.read_bytes())[0]
                self.assertFalse(block_screen(html))
                result = classify_html(html, item, item['source_url'], item['selected_content_id']['content_id'])
                self.assertTrue(result['html_original_accessible'])
                self.assertEqual(digest(path), before)
                count += 1
            self.assertEqual(count, 20)

    result = unittest.TextTestRunner(verbosity=2, buffer=True).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def _kind_calls(root: Path, kind_run: Path) -> int:
    progress = root / kind_run / "progress.json"
    return json.loads(progress.read_text(encoding="utf-8")).get("calls_this_invocation", 0) if progress.exists() else 0


def _kind_collect(kind_run: Path, targets: Path, calls: int, min_free_gb: float) -> list[str]:
    return ["src.preprocessing.kind_collect", "--phase", "targets", "--run-dir", kind_run.as_posix(),
            "--targets-dir", targets.as_posix(), "--max-calls", str(calls), "--min-free-gb", str(min_free_gb), "--execute"]


KIND_RUN = Path("outputs/collection/KIND_ORIGINAL_COLLECT/20000101_20211231")
PILOT_GAPS, PILOT_CONTROLS, PILOT_SEED = 50, 10, 42


KIND_KIND = re.compile(r"^(?:\[[^\]]+\])?\s*(사업|반기|분기)")


BROKEN_TEXT = re.compile("紐⑹감|諛섍린|蹂닿퀬|[가-힣]�")  # UTF-8 Korean once misread as cp949 by the source


MATCH_DAYS = 5


def _broken_encoding(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        name = sorted(n for n in archive.namelist() if n.lower().endswith(".xml"))[0]
        head = _decode_document(archive.read(name))[0][:3000]
    return head.count("�") > 3 or bool(BROKEN_TEXT.search(head))


def dart_receipt_status(root: Path, folder: Path, extract: Path) -> pd.DataFrame:
    """One row per visited DART receipt (labels <= 2014-12): does DART give a dated value, and if not, why."""
    if not (folder / 'selection.csv').is_file() or not (extract / 'extractions.jsonl').is_file():
        from src.preprocessing.dart_extract import load_common_dart, decisions_from_facts, filing_observations
        metadata, facts = load_common_dart(root)
        selected = decisions_from_facts(metadata, facts).set_index('receipt_id')
        observed = filing_observations(root)
        observed = observed[observed.disclosure_format.eq('document_only') &
                            observed.report_period_label.le(LAST_ROLLING_PERIOD)].copy()
        observed['dart_cause'] = [
            'value' if pd.notna(selected.at[r, 'equity']) and pd.notna(selected.at[r, 'fiscal_period'])
            else 'value_date_unconfirmed' if pd.notna(selected.at[r, 'equity']) else 'no_value_observed'
            for r in observed.rcept_no]
        observed['group'] = observed.corp_code + '|' + observed.report_period_label + '|' + observed.report_kind
        observed['group_filled'] = observed.group.map(observed.groupby('group').dart_cause.agg(lambda s: s.eq('value').any()))
        return observed
    states = {e["rcept_no"]: e["state"] for e in read_events(folder / "manifest.jsonl") if e["event"] == "endpoint_state"}
    keys = {j["rcept_no"]: j for j in load_jobs(root, [folder], source_runs=())[0]}
    journal = {}
    with (extract / "extractions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            journal[event["source_key"]] = (event["decision"].get("equity"), event["decision"].get("fiscal_period"),
                                            event["audit"].get("shape_candidate_tables", 0))
    rows = []
    selection = pd.read_csv(folder / "selection.csv", dtype=str, keep_default_na=False)
    for item in selection[selection.report_period_label <= LAST_ROLLING_PERIOD].to_dict("records"):
        state = states.get(item["rcept_no"])
        if state is None:
            continue  # not visited yet; a later run of this builder picks it up
        equity = fiscal = None
        if state.startswith("absence"):
            cause = "dart_" + state
        elif state != "original_verified":
            cause = "dart_" + state
        elif keys[item["rcept_no"]]["source_key"] not in journal:
            cause = "not_extracted"
        else:
            equity, fiscal, tables = journal[keys[item["rcept_no"]]["source_key"]]
            cause = ("value" if equity is not None and fiscal else "value_date_unconfirmed" if equity is not None
                     else "dart_encoding_broken" if _broken_encoding(root / keys[item["rcept_no"]]["document"]["path"])
                     else "no_statement_table" if tables == 0 else "no_value_with_table")
        rows.append({**{k: item[k] for k in ("corp_code", "corp_name", "stock_code", "corp_cls", "report_nm", "rcept_no", "rcept_dt",
                                             "report_kind", "report_period_label")},
                     "dart_cause": cause, "group": f"{item['corp_code']}|{item['report_period_label']}|{item['report_kind']}"})
    frame = pd.DataFrame(rows)
    frame["group_filled"] = frame.group.map(frame.groupby("group").dart_cause.agg(lambda s: (s == "value").any()))
    return frame


def match_kind(receipts: pd.DataFrame, kind: pd.DataFrame) -> pd.DataFrame:
    """KIND counterpart of each DART receipt: the same receipt number, else the only KIND row of the same company
    (first five stock-code digits), report kind and receipt date within MATCH_DAYS. Labels are not compared: KIND
    holds many rows without a period in the title. Ambiguous in either direction is never resolved here.

    A shared receipt number alone is not identity: KIND and DART number filings separately, so the same number can be
    another company's filing on the same day, or another date or report kind. It counts only with the same company
    (stock-code prefix, or the exact name when DART has no stock code), the same kind and a date within MATCH_DAYS.
    """
    kind = kind.assign(kind_kind=kind.report_name.str.extract(KIND_KIND)[0],
                       kind_date=pd.to_datetime(kind.receipt_date_observed_or_inferred, errors="coerce"))
    by_receipt = {r: g for r, g in kind.groupby("receipt_no")}
    by_code = {c: g for c, g in kind.groupby("kind_company_code")}
    # Delisted companies have no stock code in DART: an exact name that KIND uses for one code only stands in.
    names = kind.groupby("company_name").kind_company_code.agg(lambda c: c.iloc[0] if c.nunique() == 1 else None).dropna()
    out = []
    for row in receipts.to_dict("records"):
        found, basis = by_receipt.get(row["rcept_no"]), "same_receipt"
        if found is not None:
            same_company = (found.kind_company_code == row["stock_code"][:5]) if row["stock_code"] else (found.company_name == row["corp_name"])
            found = found[same_company & (found.kind_kind == row["report_kind"])
                          & ((found.kind_date - pd.Timestamp(row["rcept_dt"])).abs() <= pd.Timedelta(days=MATCH_DAYS))]
            found = found if len(found) else None
        if found is None:
            basis = "code_date_kind" if row["stock_code"] else "name_date_kind"
            code = row["stock_code"][:5] if row["stock_code"] else names.get(row["corp_name"])
            pool = by_code.get(code) if code else None
            if pool is None:
                out.append({**row, "match": "no_kind_company" if row["stock_code"] else "no_stock_code_or_unique_name"})
                continue
            near = pool[(pool.kind_date - pd.Timestamp(row["rcept_dt"])).abs() <= pd.Timedelta(days=MATCH_DAYS)]
            found = near[near.kind_kind == row["report_kind"]]
            if found.empty:
                out.append({**row, "match": "kind_report_kind_differs" if len(near) else "not_in_kind_list"})
                continue
        if len(found) > 1:
            out.append({**row, "match": "ambiguous", "kind_receipt_no": ";".join(found.receipt_no), "match_basis": basis})
            continue
        k = found.iloc[0]
        out.append({**row, "match": {"KOSPI": "target", "KOSDAQ": "kosdaq_at_the_time"}.get(k.market_group, "kind_market_unknown"),
                    "match_basis": basis, "kind_receipt_no": k.receipt_no, "kind_report_name": k.report_name,
                    "kind_date": k.receipt_date_observed_or_inferred, "kind_market": k.market_group, "kind_decision": k.decision,
                    "kind_reasons": k.reasons, "date_gap_days": abs((k.kind_date - pd.Timestamp(row["rcept_dt"])).days)})
    frame = pd.DataFrame(out)
    # One KIND filing can stand for one DART version only.
    shared = frame[frame.match.eq("target")].kind_receipt_no.duplicated(keep=False)
    frame.loc[shared[shared].index, "match"] = "ambiguous"
    return frame


def kind_targets(root: Path, folder: Path, extract: Path, kind_review: Path) -> dict:
    """KIND download list: versions of company x period x kind groups DART leaves without a dated value."""
    receipts = dart_receipt_status(root, folder, extract)
    # A group with a version not yet extracted is still open on the DART side: not a gap yet.
    pending = set(receipts[receipts.dart_cause.eq("not_extracted")].group)
    if kind_review.is_file():
        kind = pd.read_csv(kind_review, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    else:
        pages, _ = kind_source_observations(root)
        if not pages: raise FileNotFoundError('required_KIND_list_sources_missing:data/raw/kind')
        _, decisions, _ = make_scope_review(root, pages, False)
        kind = pd.DataFrame(decisions)
    gaps = match_kind(receipts[~receipts.group_filled & ~receipts.group.isin(pending)], kind)
    controls = match_kind(receipts[receipts.dart_cause.eq("value")], kind)
    controls = controls[controls.match.eq("target")]
    listed = kospi_listing(root, sorted({d[:4] for d in gaps.rcept_dt}))
    month = gaps.rcept_dt.str[:4] + "-" + gaps.rcept_dt.str[4:6]
    gaps["kospi_at_receipt"] = [("code" if s and (s, m) in listed else "name" if (n, m) in listed else "no")
                                for s, n, m in zip(gaps.stock_code, gaps.corp_name, month)]
    # Not a KOSPI security that month (marcap): outside the study, whatever KIND says or lacks.
    gaps["outcome"] = gaps.match.where(gaps.kospi_at_receipt.ne("no") | gaps.match.eq("target"), "not_kospi_at_receipt")
    gaps.loc[gaps.match.eq("kosdaq_at_the_time"), "outcome"] = "kosdaq_at_the_time"
    gaps.loc[gaps.match.eq("target") & gaps.kospi_at_receipt.eq("no"), "outcome"] = "hold:kind_kospi_but_not_in_marcap"
    gaps.loc[~gaps.outcome.isin(["target", "not_kospi_at_receipt", "kosdaq_at_the_time"]) & ~gaps.outcome.str.startswith("hold:"),
             "outcome"] = "hold:" + gaps.match
    holds = gaps[gaps.outcome.str.startswith("hold:")].copy()
    holds = holds.join(_hold_hints(holds, kind))
    targets = gaps[gaps.outcome.eq("target")].copy()
    era = targets.report_period_label.str[:4].map(lambda y: "1999-2000" if y <= "2000" else "2001" if y == "2001"
                                                  else "2002-2003" if y <= "2003" else "2004+")
    strata = targets.groupby([targets.dart_cause, era], group_keys=False)
    share = -(-PILOT_GAPS // max(1, strata.ngroups))
    probes = pd.concat([holds[holds.outcome.eq("hold:ambiguous")].sample(min(5, int(holds.outcome.eq("hold:ambiguous").sum())), random_state=PILOT_SEED),
                        holds[holds.wide_candidate.notna()].sample(min(5, int(holds.wide_candidate.notna().sum())), random_state=PILOT_SEED)
                        .assign(kind_receipt_no=lambda f: f.wide_candidate)])
    pilot = pd.concat([strata.apply(lambda g: g.sample(min(len(g), share), random_state=PILOT_SEED)).head(PILOT_GAPS).assign(pilot_role="gap"),
                       controls.sample(min(len(controls), PILOT_CONTROLS), random_state=PILOT_SEED).assign(pilot_role="dart_value_control"),
                       probes.assign(pilot_role="hold_probe")])
    # One row per KIND receipt to fetch: an ambiguous probe lists every candidate.
    pilot = pilot.assign(kind_receipt_no=pilot.kind_receipt_no.str.split(";")).explode("kind_receipt_no")
    outcome = gaps.groupby("group").outcome.agg(lambda s: "target" if s.eq("target").any() else s.iloc[0])
    summary = {"receipts_visited": len(receipts), "groups": int(receipts.group.nunique()),
               "groups_with_dart_value": int(receipts[receipts.group_filled].group.nunique()),
               "groups_pending_extraction": len(pending - set(receipts[receipts.group_filled].group)),
               "gap_receipts_by_outcome": gaps.outcome.value_counts().to_dict(),
               "gap_groups_by_outcome": outcome.value_counts().to_dict(),
               "target_receipts": len(targets), "target_kind_receipts": int(targets.kind_receipt_no.nunique()),
               "targets_by_basis": targets.match_basis.value_counts().to_dict(),
               "targets_by_kind_decision": targets.kind_decision.value_counts().to_dict(),
               "targets_by_dart_cause": targets.dart_cause.value_counts().to_dict(),
               "targets_by_year": targets.report_period_label.str[:4].value_counts().sort_index().to_dict(),
               "holds_by_dart_cause": holds.dart_cause.value_counts().to_dict(),
               "pilot_kind_receipts": pilot.pilot_role.value_counts().to_dict(), "dart_cause": receipts.dart_cause.value_counts().to_dict(),
               "extract_dir": extract.relative_to(root).as_posix(), "kind_review": kind_review.relative_to(root).as_posix(),
               "rule": ("same company (stock-code prefix or unique exact name), report kind, receipt date within 5 days, 1:1; "
                        "KIND market KOSPI and marcap KOSPI that month; no source other than regular reports"),
               "holds": "not resolved automatically: open some originals with the user first (ambiguous, no counterpart, kind differs, marcap/KIND conflict)",
               "public_date": "paired DART rcept_dt; KIND date is a cross-check only"}
    return {"summary": summary, "targets": targets, "pilot": pilot, "holds": holds,
            "excluded": gaps[gaps.outcome.isin(["not_kospi_at_receipt", "kosdaq_at_the_time"])]}


def kospi_listing(root: Path, years: list[str]) -> set[tuple[str, str]]:
    """(code, YYYY-MM) and (name, YYYY-MM) of every KOSPI security marcap shows in that month."""
    from src.preprocessing.dart_extract import MARCAP_REL
    seen = set()
    for year in years:
        frame = pd.read_parquet(root / MARCAP_REL / f"marcap-{year}.parquet", columns=["Date", "Code", "Name", "Market"])
        frame = frame[frame.Market == "KOSPI"].assign(month=lambda f: f.Date.dt.strftime("%Y-%m"))
        seen |= set(zip(frame.Code, frame.month)) | set(zip(frame.Name, frame.month))
    return seen


def _hold_hints(holds: pd.DataFrame, kind: pd.DataFrame) -> pd.DataFrame:
    """Facts that help the later manual look: a same-day candidate, a same-kind KIND row within 15 days, KIND coverage start."""
    kind = kind.assign(kind_kind=kind.report_name.str.extract(KIND_KIND)[0],
                       kind_date=pd.to_datetime(kind.receipt_date_observed_or_inferred, errors="coerce"))
    by_code = {c: g for c, g in kind.groupby("kind_company_code")}
    dates = kind.set_index("receipt_no").receipt_date_observed_or_inferred.to_dict()
    rows = {}
    for index, row in holds.iterrows():
        day = pd.Timestamp(row.rcept_dt)
        same_day = [r for r in str(row.get("kind_receipt_no") or "").split(";") if r and dates.get(r, "").replace("-", "") == row.rcept_dt]
        pool = by_code.get(row.stock_code[:5]) if row.stock_code else None
        wide = pool[(pool.kind_kind == row.report_kind) & ((pool.kind_date - day).abs() <= pd.Timedelta(days=15))] if pool is not None else []
        rows[index] = {"same_day_candidate": same_day[0] if len(same_day) == 1 else None,
                       "wide_candidate": wide.receipt_no.iloc[0] if len(wide) == 1 else None,
                       "kind_company_first_date": pool.kind_date.min().strftime("%Y-%m-%d") if pool is not None else None,
                       "before_kind_coverage": bool(pool is not None and day < pool.kind_date.min())}
    return pd.DataFrame.from_dict(rows, orient="index")


KIND_EXTRACT = Path("outputs/collection/KIND_EQUITY_EXTRACT")


PAIR_FIELDS = ("corp_code", "corp_name", "stock_code", "rcept_no", "rcept_dt", "report_kind", "report_period_label", "report_nm", "dart_cause")


DECISION_FIELDS = ("fiscal_period", "fiscal_period_basis", "equity", "equity_tier", "statement_type", "account_name", "section",
                   "column_basis", "identity_ok", "owners_identity_ok", "owners_resolution", "reason", "raw_path", "sha256")


def kind_pairs(targets_dir: Path, *, tables=None) -> pd.DataFrame:
    """(DART receipt, KIND receipt) rows of a --kind-targets folder; only 'target' pairs may later fill a DART gap."""
    rows = []
    for name in ("targets", "holds", "pilot"):
        frame = tables[name] if tables is not None else pd.read_csv(targets_dir / f"{name}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for row in frame.to_dict("records"):
            if name == "targets":
                pairing = "target"
            elif name == "holds":
                pairing = row.get("outcome") or "hold:"
            else:
                pairing = "pilot:" + row.get("pilot_role", "")
            candidates = row.get("kind_receipt_no", "").split(";") + [row.get("same_day_candidate", ""), row.get("wide_candidate", "")]
            for receipt in dict.fromkeys(c.strip() for c in candidates if c and c.strip()):
                rows.append({**{k: row.get(k, "") for k in PAIR_FIELDS}, "kind_receipt_no": receipt, "pairing": pairing})
    return pd.DataFrame(rows, columns=[*PAIR_FIELDS, "kind_receipt_no", "pairing"]).drop_duplicates()


def kind_acquired(root: Path, kind_run: Path) -> dict:
    """KIND receipt -> saved body record. Read-only, so it can run beside a collector."""
    if not (root / kind_run / 'state.sqlite3').is_file():
        return kind_source_observations(root)[1]
    db = sqlite3.connect((root / kind_run / "state.sqlite3").as_uri() + "?mode=ro", uri=True)
    try:
        out = {}
        for receipt, detail in db.execute("SELECT receipt,detail FROM filings WHERE doc_state='original_saved'"):
            detail = json.loads(detail)
            if detail.get("original_acquired"):
                out[receipt] = {**detail["response"], "period_label": detail.get("period_label"), "period_status": detail.get("period_status")}
        return out
    finally:
        db.close()


def kind_extract_filing(job: dict) -> dict:
    """Worker: restore the stored KIND body, check both hashes, parse it as the DART body parser does."""
    from src.preprocessing.kind_collect import decode
    from src.preprocessing.kind_collect import stored_bytes
    record, problems, facts, audit = job["record"], [], [], {}
    path = Path(job["root"]) / record["path"]
    try:
        if record.get("compression") == "gzip" and hashlib.sha256(path.read_bytes()).hexdigest() != record["stored_sha256"]:
            raise ValueError("stored_hash_mismatch")
        raw = stored_bytes(path)
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("source_hash_mismatch")
        facts, audit = document_facts(decode(raw, record.get("content_type", ""))[0], job["report_period_label"], member=path.name)
        facts = [{**f, "raw_path": record["path"], "sha256": record["sha256"]} for f in facts]
    except (OSError, ValueError) as error:
        problems.append("kind:" + str(error)[:80])
    decision = select_filing(facts)
    if decision["equity"] is None and problems:
        decision["reason"] = ";".join(problems)
    return {"event": "extraction", "source_key": job["source_key"], "kind_receipt_no": job["kind_receipt_no"],
            "report_period_label": job["report_period_label"], "facts": facts, "decision": decision, "audit": dict(audit),
            "problems": problems, "extractor_version": EXTRACTOR_VERSION, "code_sha256": CODE_FINGERPRINT}


def kind_extract(root: Path, kind_run: Path, targets_dir: Path, out: Path, *, dart_extract: Path | None = None, workers: int = 1) -> dict:
    """KIND bodies of a targets folder -> one row per (DART receipt, KIND receipt) pair. Resumable; no network.

    The paired DART receipt keeps the identity and the public date (DART rcept_dt); the KIND list date is
    not used. A new extractor fingerprint gives new keys, so a changed rule goes to a new folder.
    """
    tables = None
    if not all((targets_dir / f'{name}.csv').is_file() for name in ('targets', 'holds', 'pilot')):
        tables = kind_targets(root, root / HISTORY_OUTPUT / 'current', dart_extract or root / EXTRACT_OUTPUT,
                              root / KIND_REVIEW / 'scope_review.csv')
    pairs = kind_pairs(targets_dir, tables=tables)
    saved = kind_acquired(root, kind_run)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "collection.json").exists():
        atomic_json(out / "collection.json", {"created_at": now(), "kind_run": kind_run.as_posix(), "targets_dir": targets_dir.relative_to(root).as_posix(),
                                       "extractor_version": EXTRACTOR_VERSION, "code_sha256": CODE_FINGERPRINT})
    journal = out / "extractions.jsonl"
    seen = {e["source_key"] for e in read_events(journal)}
    jobs = {}
    for kind_receipt, label in pairs[pairs.kind_receipt_no.isin(saved)][["kind_receipt_no", "report_period_label"]].drop_duplicates().itertuples(index=False):
        record = saved[kind_receipt]
        key = source_key(kind_receipt + "|" + label, {"kind": {"path": record["path"], "sha256": record["sha256"]}})
        if key not in seen:
            jobs[key] = {"root": str(root), "source_key": key, "kind_receipt_no": kind_receipt, "report_period_label": label, "record": record}
    results = _map(kind_extract_filing, list(jobs.values()), workers)
    _append(journal, results)
    latest = {(e["kind_receipt_no"], e["report_period_label"]): e for e in read_events(journal) if e.get("code_sha256") == CODE_FINGERPRINT}
    rows = []
    for pair in pairs.to_dict("records"):
        record = saved.get(pair["kind_receipt_no"])
        event = latest.get((pair["kind_receipt_no"], pair["report_period_label"]))
        decision = event["decision"] if event else {}
        rows.append({**pair, **{k: decision.get(k) for k in DECISION_FIELDS},
                     "kind_acquired": record is not None, "kind_period_label": record.get("period_label") if record else None,
                     "kind_period_status": record.get("period_status") if record else None,
                     "kind_period_matches": bool(record and record.get("period_label") == pair["report_period_label"]),
                     "problems": ";".join(event["problems"]) if event else None})
    columns = [*pairs.columns, *DECISION_FIELDS, "kind_acquired", "kind_period_label", "kind_period_status", "kind_period_matches", "problems"]
    table = pd.DataFrame(rows, columns=columns).astype("string")
    if dart_extract and (dart_extract / "equity_filings.parquet").exists():
        dart = pd.read_parquet(dart_extract / "equity_filings.parquet", columns=["receipt_id", "equity"]).drop_duplicates("receipt_id")
        table["dart_equity"] = table.rcept_no.map(dart.set_index("receipt_id").equity).astype("string")
    table.to_parquet(out / "kind_filings.parquet", index=False)
    table.to_csv(out / "kind_filings.csv", index=False, encoding="utf-8-sig")
    value = table.equity.notna() & table.equity.ne("")
    group = table.pairing.str.split(":").str[0]
    summary = {"status": "kind_extraction_finished_not_PIT_verified", "updated_at": now(), "code_sha256": CODE_FINGERPRINT,
               "pairs": len(table), "kind_receipts": int(table.kind_receipt_no.nunique()), "extracted_this_run": len(results),
               "by_pairing": {p: {"pairs": len(g), "acquired": int(g.kind_acquired.eq("True").sum()), "value": int(value[g.index].sum()),
                                  "dated": int((value[g.index] & g.fiscal_period.notna()).sum())}
                              for p, g in table.groupby(group)},
               "targets_by_year": {y: {"pairs": len(g), "acquired": int(g.kind_acquired.eq("True").sum()), "value": int(value[g.index].sum())}
                                   for y, g in table[group.eq("target")].groupby(table.report_period_label.str[:4])},
               "kind_period_mismatch": int((table.kind_acquired.eq("True") & table.kind_period_matches.eq("False")).sum()),
               "problems": int(table.problems.fillna("").ne("").sum())}
    if "dart_equity" in table:
        controls = table[table.pairing.eq("pilot:dart_value_control") & value]
        summary["dart_value_controls"] = {"compared": len(controls), "same_amount": int((controls.equity == controls.dart_equity).sum())}
    atomic_json(out / "progress.json", summary)
    return summary


def main_pairs() -> None:
    """KIND list for DART groups without a value (targets) and offline extraction of saved KIND bodies (extract).
    Formerly the --kind-targets / --kind-extract options of the DART history CLI."""
    parser = argparse.ArgumentParser(description=main_pairs.__doc__)
    parser.add_argument("action", choices=("targets", "extract"))
    parser.add_argument("--run-dir", type=Path, required=True, help="DART_HISTORY_COLLECT folder")
    parser.add_argument("--extract-dir", type=Path, required=True, help="DART_EQUITY_EXTRACT folder")
    parser.add_argument("--kind-review", type=Path, default=KIND_REVIEW / "scope_review.csv", help="KIND list review CSV")
    parser.add_argument("--kind-targets-dir", type=Path, help="extract: a saved targets/<stamp> folder")
    parser.add_argument("--kind-extract-dir", type=Path, help="extract: KIND_EQUITY_EXTRACT folder (new folder after a rule change)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--execute", action="store_true", help="targets: save the list; default is a dry-run")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    folder = within(root, args.run_dir, HISTORY_OUTPUT)
    extract = within(root, args.extract_dir, EXTRACT_OUTPUT)
    if args.action == "extract":
        if not (args.kind_targets_dir and args.kind_extract_dir):
            parser.error("extract needs --kind-targets-dir and --kind-extract-dir")
        targets_dir = within(root, args.kind_targets_dir, KIND_RUN / "targets")
        print(json.dumps(kind_extract(root, KIND_RUN, targets_dir, within(root, args.kind_extract_dir, KIND_EXTRACT),
                                      dart_extract=extract, workers=args.workers), ensure_ascii=False, indent=2))
        return
    built = kind_targets(root, folder, extract, within(root, args.kind_review, KIND_RUN))
    print(json.dumps(built["summary"], ensure_ascii=False, indent=2))
    if not args.execute:
        print("DRY-RUN: no files written.")
        return
    target = root / KIND_RUN / "targets" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    target.mkdir(parents=True)
    for name in ("targets", "pilot", "holds", "excluded"):
        built[name].to_csv(target / f"{name}.csv", index=False, encoding="utf-8-sig")
    atomic_json(target / "summary.json", {**built["summary"], "files_sha256": {
        n: hashlib.sha256((target / f"{n}.csv").read_bytes()).hexdigest() for n in ("targets", "pilot")}})
    print("saved:", target.relative_to(root).as_posix())


SUBCOMMANDS = {"probe": main_probe, "collect": main_collect, "pairs": main_pairs}


def main() -> None:
    """First argument picks the former CLI; the rest are its options."""
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        raise SystemExit('usage: python -m src.preprocessing.kind_collect {' + '|'.join(SUBCOMMANDS) + '} [options]')
    command = sys.argv.pop(1)
    SUBCOMMANDS[command]()


if __name__ == "__main__":
    from src.preprocessing.dart_extract import record_cli_execution
    record_cli_execution(main,"KIND_COLLECTION")
