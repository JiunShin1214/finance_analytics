"""Prepare PIT evidence queues and optionally fetch original DART filings.
No model fitting, no test data, no automatic source certification.
Python 3.10+; standard library only.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
import urllib.request
import urllib.parse
import urllib.error
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import xml.etree.ElementTree as ET


def sha(b):
    return hashlib.sha256(b).hexdigest()


def locate(folder, stem, suffix='.csv'):
    for name in (stem+suffix, stem+'(1)'+suffix):
        p=folder/name
        if p.is_file():
            return p
    raise FileNotFoundError(folder/(stem+suffix))


def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(p, rows, fields=None):
    if fields is None:
        fields=list(rows[0]) if rows else []
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def flag(x):
    return str(x).strip().lower() in ('true','1','yes')


def review_fields():
    return dict(evidence_status='REVIEW', evidence_ref='', evidence_sha256='',
                verified_publication_date='', accounting_scope='', verified_fiscal_date='',
                original_liabilities='', original_equity='', original_debt_ratio_pct='',
                reviewer='', review_note='')


def receipt(x):
    x=str(x).strip()
    if not re.fullmatch(r'\d{14}',x):
        raise ValueError('Invalid receipt number: '+x)
    return x


def make_queues(dart, guide, policy):
    dart=sorted(dart,key=lambda x:(not flag(x['unresolved']),
                                 -int(x['source_rows']), x['source_rcept_no'],x['stock_id']))
    dq=[]
    for x in dart:
        rc=receipt(x['source_rcept_no'])
        dq.append(dict(x, priority='P0' if flag(x['unresolved']) else 'P1',
                       receipt=rc, **review_fields()))
    gq=[]
    for x in sorted(guide,key=lambda x:(not flag(x['unresolved']),-int(x['source_rows']),x['stock_id'],x['source_fiscal_date'])):
        gq.append(dict(x,priority='P0' if flag(x['unresolved']) else 'P1',**review_fields()))
    # Include both source IDs, not just the sensitivity-oriented document_id.
    docs={}
    for x in policy:
        for mode in ('strict','sensitivity'):
            doc=x.get(mode+'_source_document_id','').strip()
            if not doc: continue
            date=x.get(mode+'_source_release_date','').strip()
            score=x.get('final_rule_v2_'+mode,'').strip()
            a=docs.setdefault(doc,dict(dates=set(), uses=set(), scores=set(), modes=set(), inferred=False, metadata=set()))
            if date: a['dates'].add(date)
            a['uses'].add(x['month']+':'+x['role']+':'+mode)
            if score: a['scores'].add(mode+':'+score)
            a['modes'].add(mode)
            # Release metadata in original queue refers to document_id, not necessarily strict ID.
            if doc==x.get('document_id',''):
                src=x.get('release_date_sensitivity_source','')
                if src: a['metadata'].add(src)
                if 'inferred' in src: a['inferred']=True
    pq=[]
    for doc,a in sorted(docs.items()):
        unknown=not a['metadata']; conflict=len(a['dates'])!=1
        pq.append(dict(document_id=doc, expected_release_dates=';'.join(sorted(a['dates'])),
                       modes=';'.join(sorted(a['modes'])), uses=';'.join(sorted(a['uses'])),
                       expected_scores=';'.join(sorted(a['scores'])),
                       metadata=';'.join(sorted(a['metadata'])),
                       priority='P0' if (unknown or conflict or a['inferred']) else 'P1',
                       evidence_status='REVIEW', official_url='', original_file_sha256='',
                       verified_release_date='', verified_rule_score='', reviewer='', review_note=''))
    pq.sort(key=lambda x:(x['priority'], x['document_id']))
    return dq,gq,pq


def inspect_zip(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if not z.namelist() or z.testzip() is not None:
            raise ValueError('Invalid ZIP')
        return len(z.namelist())


def fetch(rc, key):
    url='https://opendart.fss.or.kr/api/document.xml?'+urllib.parse.urlencode(dict(crtfc_key=key,rcept_no=rc))
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data=response.read()
    except urllib.error.HTTPError as e:
        # Do not include the URL, which contains the secret.
        return None, 'HTTP_'+str(e.code)
    except (OSError, TimeoutError):
        return None, 'NETWORK_ERROR'
    if zipfile.is_zipfile(io.BytesIO(data)):
        try: inspect_zip(data)
        except (ValueError, RuntimeError, OSError, zipfile.BadZipFile): return None,'BAD_ZIP'
        return data,'DOWNLOADED'
    try:
        status=ET.fromstring(data).findtext('.//status') or 'UNKNOWN'
        if not re.fullmatch(r'\d{3}',status): status='UNKNOWN'
    except ET.ParseError: status='UNKNOWN'
    return None,'DART_'+status


def download(queue, out, limit, delay):
    key=os.environ.get('DART_API_KEY') or os.environ.get('OPENDART_API_KEY') or os.environ.get('OPEN_DART_API_KEY')
    cache=out/'dart_original_zip'; cache.mkdir(exist_ok=True)
    manifest=[]; seen=set(); calls=0
    for row in queue:
        rc=row['receipt']
        if rc in seen: continue
        seen.add(rc)
        p=cache/(rc+'.zip'); data=None; status='NOT_REQUESTED'; members=0
        if p.exists():
            try:
                data=p.read_bytes(); members=inspect_zip(data); status='CACHED'
            except (ValueError, OSError, RuntimeError, zipfile.BadZipFile):
                data=None; status='INVALID_CACHE'
        if data is None and calls<limit:
            if not key: raise SystemExit('Set DART_API_KEY (or OPENDART_API_KEY) before --download.')
            data,status=fetch(rc,key); calls+=1
            if data is not None:
                members=inspect_zip(data)
                tmp=p.with_suffix('.tmp'); tmp.write_bytes(data); tmp.replace(p)
            time.sleep(delay)
        manifest.append(dict(receipt=rc, priority=row['priority'], download_status=status,
                             original_zip=str(p.relative_to(out)) if data else '',
                             sha256=sha(data) if data else '',zip_members=members,
                             source_certified=False))
        # Stop network calls on access/key/rate limit errors; preserve partial evidence.
        if status in ('DART_010','DART_011','DART_012','DART_020','DART_901','HTTP_429'):
            limit=calls
    write(out/'dart_download_manifest.csv',manifest)
    return manifest,calls


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scope-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--limit',type=int,default=30,help='Maximum new API calls per run; cached ZIPs do not count')
    ap.add_argument('--delay',type=float,default=.5)
    args=ap.parse_args()
    if args.limit<0 or args.delay<0: ap.error('limit and delay must be nonnegative')
    if args.scope_dir.resolve()==args.out.resolve(): ap.error('Use separate output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    paths={s:locate(args.scope_dir,s) for s in ('dart_report_queue','dataguide_report_queue','policy_document_queue','fit_scope','scope_by_month','rate_official_comparison')}
    paths['pit_scope_audit']=locate(args.scope_dir,'pit_scope_audit','.json')
    audit=json.loads(paths['pit_scope_audit'].read_text(encoding='utf-8-sig'))
    if audit.get('test_used') is not False or audit.get('model_fitting') is not False:
        raise ValueError('Unexpected scope audit status')
    dart,guide,policy=(read(paths[s]) for s in ('dart_report_queue','dataguide_report_queue','policy_document_queue'))
    dq,gq,pq=make_queues(dart,guide,policy)
    for name,rows in [('dart_original_review',dq),('dataguide_vintage_review',gq),('policy_original_review',pq)]:
        p=args.out/(name+'.csv')
        # Preserve human review work. Use a fresh out directory to rebuild templates.
        if p.exists():
            old=read(p)
            keys={'dart_original_review':('stock_id','source_fiscal_date','receipt'),
                  'dataguide_vintage_review':('stock_id','source_fiscal_date','source_type'),
                  'policy_original_review':('document_id',)}[name]
            if {tuple(r[k] for k in keys) for r in old}!={tuple(r[k] for k in keys) for r in rows}:
                raise ValueError('Review keys changed; choose a new --out directory: '+name)
        else: write(p,rows)
    manifest,calls=download(dq,args.out,args.limit,args.delay) if args.download else ([],0)
    summary=dict(test_used=False,model_fitting=False,submission_certified=False,
                 original_values_certified=False,
                 input_hashes={k:sha(v.read_bytes()) for k,v in paths.items()},
                 script_sha256=sha(Path(__file__).read_bytes()),
                 generated_utc=datetime.now(timezone.utc).isoformat(),
                 dart_candidates=len(dq),dart_flagged=sum(x['priority']=='P0' for x in dq),
                 dataguide_candidates=len(gq),dataguide_flagged=sum(x['priority']=='P0' for x in gq),
                 policy_documents=len(pq),policy_priority_p0=sum(x['priority']=='P0' for x in pq),
                 policy_missing_from_original_document_id=sorted({x['document_id'] for x in pq}-{x['document_id'] for x in policy}),
                 api_calls=calls,downloaded_or_cached=sum(x['download_status'] in ('DOWNLOADED','CACHED') for x in manifest),
                 source_status='REVIEW: downloading originals does not certify financial values or vintage')
    (args.out/'evidence_collection_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
