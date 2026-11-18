"""Sentence units and release-date evidence from preserved BOK text."""

def prepare(input_path=None, output=None):
    from src.h2.data_io import assert_unique, sha256
    source = Path(input_path) if input_path else ROOT/'data/h2/bok_text_clean_h2_full_ready.csv'
    output = Path(output) if output else PROCESSED/'policy'
    df = pd.read_csv(source)
    df['meeting_date'] = pd.to_datetime(df.meeting_date, errors='raise')
    df['year'] = df.meeting_date.dt.year
    full = df[df.year.between(2001, 2021)].sort_values(['meeting_date','title']).reset_index(drop=True)
    key = full.meeting_date.dt.strftime('%Y%m%d')
    full['document_id'] = key+'_'+full.groupby(key).cumcount().add(1).astype(str).str.zfill(2)
    assert_unique(full, ['document_id'])
    rows, releases = [], []
    for _, r in full.iterrows():
        content = r.get('h2_clean_text', r.get('clean_text',''))
        for i, unit in enumerate(split_units(content), 1):
            rows.append({'document_id':r.document_id,'meeting_date':r.meeting_date.date().isoformat(),
                         'release_date':r.get('release_date',''),'release_date_source':r.get('release_date_source',''),
                         'year_month':r.get('year_month',''),'title':r.get('title',''),'source_url':r.get('source_url',''),
                         'unit_id':f'{r.document_id}_U{i:02d}','unit_order':i,'text_unit':unit,'char_len':len(unit)})
        date, method = parse_header_release_date(content)
        inferred = pd.isna(date)
        releases.append({'document_id':r.document_id,'meeting_date':r.meeting_date,
                         'release_date_strict':date,'release_date_sensitivity':r.meeting_date if inferred else date,
                         'release_date_method':method,'release_date_status':'legacy_unverified' if inferred else ('verified_document_header' if date==r.meeting_date else 'date_conflict'),
                         'release_date_sensitivity_source':'legacy_same_day_inferred' if inferred else 'verified_document_header'})
    write_csv(full, output/'documents.csv')
    write_csv(pd.DataFrame(rows), output/'units.csv')
    write_csv(pd.DataFrame(releases), output/'release_dates.csv')
    write_json({'input':str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                'sha256':sha256(source),'documents':len(full),'units':len(rows),
                'boundary':'Preserved extracted/recovered text. Live collection and raw HWP/PDF conversion are separate, unverified acquisition steps.'},output/'provenance.json')
    return output
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd
from src.h2.data_io import ROOT, PROCESSED, VALIDATION, write_csv, write_json

BULLET_RE = re.compile('^\\s*(?:[□■○●◦•▪※]|[oOㅇ]\\s+|[-–—]\\s+)\\s*')

CUE_PATTERNS = ['금융통화위원회는\\s*다음\\s*통화정책방향', '금융통화위원회는\\s*앞으로', '최근\\s*국내\\s*경기', '국내경제는', '세계경제는', '소비자물가', '금융시장에서는', '앞으로의\\s*통화정책방향', '앞으로\\s*통화정책', '향후\\s*통화정책']

CUE_RE = re.compile('(?=(' + '|'.join(CUE_PATTERNS) + '))')

def normalize_line(line):
    line = str(line)
    line = line.replace('\u3000', ' ')
    line = line.replace('Œÿ', '')
    line = re.sub('[ \\t]+', ' ', line)
    return line.strip()

def prepare_lines(text):
    lines = [normalize_line(x) for x in str(text or '').splitlines()]
    lines = [x for x in lines if x]
    headings = [i for i, x in enumerate(lines) if re.sub('\\s+', '', x) == '통화정책방향']
    if headings:
        idx = headings[-1]
        if idx + 1 < len(lines):
            lines = lines[idx + 1:]
    cleaned = []
    for line in lines:
        compact = re.sub('\\s+', '', line)
        if compact.startswith('문의처'):
            continue
        if compact.startswith('공보실'):
            continue
        if '한국은행보도자료는인터넷' in compact:
            continue
        if re.fullmatch('-?\\d+-?', compact):
            continue
        cleaned.append(line)
    return cleaned

def split_units(text):
    lines = prepare_lines(text)
    units = []
    current = ''
    saw_bullet = False
    for line in lines:
        if BULLET_RE.match(line):
            saw_bullet = True
            if current:
                units.append(current.strip())
            current = BULLET_RE.sub('', line, count=1).strip()
        elif current:
            current = (current + ' ' + line).strip()
        else:
            current = line
    if current:
        units.append(current.strip())
    joined = ' '.join(lines)
    if (not saw_bullet or len(units) < 3) and len(joined) > 300:
        starts = sorted({m.start() for m in CUE_RE.finditer(joined)})
        cue_units = []
        if starts:
            for i, start in enumerate(starts):
                end = starts[i + 1] if i + 1 < len(starts) else len(joined)
                segment = joined[start:end].strip()
                if len(segment) >= 15:
                    cue_units.append(segment)
        if len(cue_units) >= 2:
            units = cue_units
    result = []
    seen = set()
    for unit in units:
        unit = re.sub('\\s+', ' ', unit).strip()
        if len(unit) < 15:
            continue
        if unit not in seen:
            result.append(unit)
            seen.add(unit)
    if not result and joined.strip():
        result = [joined.strip()]
    return result

def parse_header_release_date(text: str):
    """
    공식 보도자료 헤더에서 배포일 후보를 직접 추출한다.

    Returns
    -------
    (Timestamp|NaT, method)
    """
    text = '' if pd.isna(text) else str(text)
    head = text[:1800]
    m = re.search('((?:19|20)\\d{2})\\s*년\\s*(\\d{1,2})\\s*월\\s*(\\d{1,2})\\s*일', head)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return (pd.Timestamp(year=y, month=mo, day=d), 'document_header_korean_date')
        except ValueError:
            pass
    m = re.search('년\\s*((?:19|20)\\d{2})\\s*(\\d{1,2})\\s*월\\s*일\\s*(\\d{1,2})', head)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return (pd.Timestamp(year=y, month=mo, day=d), 'document_header_scrambled_korean')
        except ValueError:
            pass
    m = re.search('년\\s*월\\s*일\\s*공보\\s*((?:19|20)\\d{2})\\s+(\\d{1,2})\\s+(\\d{1,2})(?:\\s+(?:19|20)\\d{2})?', head)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return (pd.Timestamp(year=y, month=mo, day=d), 'document_header_scrambled_bulletin')
        except ValueError:
            pass
    m = re.search('공보\\s*((?:19|20)\\d{2})\\s+(\\d{1,2})\\s+(\\d{1,2})\\s+(?:19|20)\\d{2}', head)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return (pd.Timestamp(year=y, month=mo, day=d), 'document_header_scrambled_bulletin_alt')
        except ValueError:
            pass
    return (pd.NaT, 'legacy_unverified')

def has_distribution_notice(text: str) -> bool:
    """
    '이 자료는 배포시부터 취급...' 문구가 있는지 확인한다.
    """
    text = '' if pd.isna(text) else str(text)
    compact = re.sub('\\s+', '', text[:2200])
    candidates = ['이자료는배포시부터취급', '이자료는배포시부터', '배포시부터취급']
    return any((phrase in compact for phrase in candidates))
