# -*- coding: utf-8 -*-
"""H2 Train universe -> OpenDART corp codes -> filings (metadata only).
No Test labels, no model fitting, no PIT financial values. Requires DART_API_KEY for live collection.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from src.h2.data_io import ROOT

BASE = 'https://opendart.fss.or.kr/api'
REPORT_PATTERN = re.compile(r'(사업보고서|분기보고서|반기보고서)\s*\(\s*(\d{4})\s*\.\s*(\d{2})\s*\)')

def norm_code(x):
    s = str(x).strip().upper()
    if s.startswith('A'): s = s[1:]
    if s.endswith('.0'): s = s[:-2]
    return s.zfill(6) if s.isdigit() else ''

def find_panel(project, override):
    if override:
        p=Path(override).expanduser().resolve()
    else:
        p=project/'data'/'h2'/'h2_final_rule_v2_panel_2001_2021.csv'
    if not p.is_file():
        raise FileNotFoundError(f'H2 패널 CSV를 찾을 수 없습니다: {p}. --panel 로 위치를 지정하세요.')
    return p

def read_universe(panel):
    head=pd.read_csv(panel,nrows=2)
    field = 'stock_id' if 'stock_id' in head.columns else 'stock_code' if 'stock_code' in head.columns else None
    if not field: raise ValueError(f'stock_id/stock_code가 없음. 컬럼: {list(head.columns)}')
    df=pd.read_csv(panel,usecols=[field],dtype={field:str},low_memory=False)
    df['stock_code']=df[field].map(norm_code)
    invalid=df.loc[df.stock_code.eq(''),field].dropna().unique().tolist()
    codes=sorted(set(df.loc[df.stock_code.str.fullmatch(r'\d{6}'),'stock_code']))
    if not codes:raise ValueError('정상 종목코드 0건')
    return pd.DataFrame({'stock_code':codes}),invalid

def save_csv(df, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    df.to_csv(temp,index=False,encoding='utf-8-sig')
    temp.replace(path)

def api_key_or_error():
    key=os.environ.get('DART_API_KEY','').strip()
    if not key:
        raise RuntimeError('DART_API_KEY 미등록. 같은 PowerShell에서 $env:DART_API_KEY="..." 를 설정하세요. 키를 코드에 넣지 마세요.')
    return key

class DartClient:
    def __init__(self,key,cache,delay=0.16):
        self.key=key;self.cache=cache;self.cache.mkdir(parents=True,exist_ok=True)
        self.s=requests.Session();self.delay=delay
    def get(self,api,params=None,binary=False):
        params=params or {}
        sanitized={k:str(v) for k,v in sorted(params.items())}
        digest=hashlib.sha256(json.dumps({'api':api,'params':sanitized},sort_keys=True).encode()).hexdigest()[:22]
        cachepath=self.cache/(api.replace('.','_')+'_'+digest+('.zip' if binary else '.json'))
        if cachepath.exists():
            if binary:return cachepath.read_bytes()
            return json.loads(cachepath.read_text(encoding='utf-8'))
        # Network / 5xx retry; 429 and API status errors are surfaced (not silently treated as empty)
        for attempt in range(4):
            try:
                r=self.s.get(f'{BASE}/{api}',params={'crtfc_key':self.key,**params},timeout=60)
                if r.status_code in (429,500,502,503,504):
                    raise requests.HTTPError(f'{r.status_code} HTTP response',response=r)
                r.raise_for_status()
                if binary:
                    if not zipfile.is_zipfile(io.BytesIO(r.content)):
                        raise RuntimeError(f'{api}: ZIP 아닌 응답. API 인증/호출량 점검: {r.text[:250]}')
                    cachepath.write_bytes(r.content);time.sleep(self.delay)
                    return r.content
                d=r.json();status=d.get('status')
                if status not in ('000','013'):
                    raise RuntimeError(f'OpenDART {api}: status={status} message={d.get("message")}')
                cachepath.write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8')
                time.sleep(self.delay)
                return d
            except (requests.ConnectionError,requests.Timeout,requests.HTTPError) as exc:
                if attempt==3: raise
                time.sleep(min(2**attempt+1,12))
        raise AssertionError('unreachable')
    def corp_codes(self):
        raw=self.get('corpCode.xml',binary=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.xml')]
            if not names:raise RuntimeError('corpCode ZIP에 XML 없음')
            root=ET.fromstring(z.read(names[0]))
        rows=[]
        for e in root.findall('.//list'):
            rows.append({'stock_code':norm_code(e.findtext('stock_code') or ''),
                         'corp_code':(e.findtext('corp_code') or '').strip(),
                         'corp_name':(e.findtext('corp_name') or '').strip(),
                         'modify_date':(e.findtext('modify_date') or '').strip()})
        df=pd.DataFrame(rows)
        return df[df.stock_code.str.fullmatch(r'\d{6}') & df.corp_code.ne('')].copy()
    def filings(self,corp,start,end):
        results=[];page=1
        while True:
            params={'corp_code':corp,'bgn_de':start,'end_de':end,
                    'pblntf_ty':'A','last_reprt_at':'N','page_no':page,'page_count':100}
            res=self.get('list.json',params)
            if res['status']=='013':break
            results.extend(res.get('list',[]))
            if page>=int(res.get('total_page',1)):break
            page+=1
        return results

def parse_filing(raw,stock,corp):
    name=str(raw.get('report_nm',''))
    match=REPORT_PATTERN.search(name)
    if not match:return None
    report_type,yy,mm=match.groups()
    mm=int(mm);year=int(yy)
    if report_type == "사업보고서":
        period_status = (
            "DEC_ANNUAL_TITLE_CANDIDATE"
            if mm == 12
            else "NON_DEC_ANNUAL_TITLE_CANDIDATE"
        )
    else:
        period_status = "INTERIM_REPORT_TITLE_MONTH_NOT_FISCAL_YEAR"
    rcept_date=str(raw.get('rcept_dt',''))
    if not re.fullmatch(r'\d{8}',rcept_date):return None
    date=pd.to_datetime(rcept_date,format='%Y%m%d',errors='coerce')
    if pd.isna(date):return None
    rcept_no=str(raw.get('rcept_no',''))
    return {'stock_code':stock,'corp_code':corp,'corp_name':raw.get('corp_name',''),
            'report_nm':name,'report_type':report_type,'fiscal_year_label':year,
            'fiscal_month_label':mm,'fiscal_period_label':f'{year:04d}-{mm:02d}',
            'filing_date':date.strftime('%Y-%m-%d'),'rcept_no':rcept_no,
            'is_amendment_name': bool(re.search(r"정정|첨부추가|철회", name)),
            'period_status':period_status,
            'rcept_url':'https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+rcept_no}

def main():
    p=argparse.ArgumentParser(description='H2 전체 종목 OpenDART 기업고유번호·정기공시 이력 수집 (학습/테스트 미실행)')
    p.add_argument('--project',default=None)
    p.add_argument('--panel',default=None)
    p.add_argument('--output',default=None)
    p.add_argument('--start',default='20010101')
    p.add_argument('--end',default='20221231',help='2021년 사업보고서는 2022년에 제출될 수 있음')
    p.add_argument('--only-map',action='store_true',help='API 키 없이 종목 universe만 저장하는 것은 --offline-universe 사용')
    p.add_argument('--offline-universe',action='store_true',help='DART API 없이 대상 종목 목록만 생성')
    p.add_argument('--limit',type=int,default=0,help='시험용 최대 종목 개수(0=전체)')
    p.add_argument('--delay',type=float,default=0.16)
    args=p.parse_args()
    project=Path(args.project).expanduser().resolve() if args.project else ROOT
    output=Path(args.output).expanduser() if args.output else project/'results'/'h2_disclosure_audit'/'all_stocks'
    if not output.is_absolute():output=project/output
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch(r'\d{8}',args.start) or not re.fullmatch(r'\d{8}',args.end) or args.end<args.start:
        raise ValueError('--start 및 --end는 순서대로 YYYYMMDD 여야 함')
    universe,invalid=read_universe(find_panel(project,args.panel))
    save_csv(universe,output/'h2_stock_universe.csv')
    print(f'H2 대상 종목 수: {len(universe)} / 무효 코드 종류 수: {len(invalid)}')
    if args.offline_universe:
        (output/'offline_summary.json').write_text(json.dumps({'universe':len(universe),'invalid_codes':list(map(str,invalid)), 'live_api_collected':False},ensure_ascii=False,indent=2),encoding='utf-8')
        print('오프라인 종목 목록 저장 완료:',output)
        return
    dart=DartClient(api_key_or_error(),output/'api_cache',args.delay)
    official=dart.corp_codes()
    # Do not pick arbitrary one of duplicate identifiers
    duplicate=official.loc[official.stock_code.duplicated(keep=False)]
    if not duplicate.empty:save_csv(duplicate.sort_values('stock_code'),output/'duplicate_corp_codes.csv')
    mapping=universe.merge(official,on='stock_code',how='left',validate='one_to_many')
    counts=mapping.groupby('stock_code')['corp_code'].transform('count')
    mapping['mapping_status']='MATCHED'
    mapping.loc[mapping.corp_code.isna(),'mapping_status']='UNMATCHED'
    mapping.loc[counts.gt(1),'mapping_status']='AMBIGUOUS'
    save_csv(mapping,output/'dart_corp_code_mapping.csv')
    todo=mapping[mapping.mapping_status.eq('MATCHED')].copy().sort_values('stock_code')
    if args.limit>0:todo=todo.head(args.limit)
    print('매칭:',sum(mapping.mapping_status.eq('MATCHED')),'누락:',sum(mapping.mapping_status.eq('UNMATCHED')),'중복:',sum(mapping.mapping_status.eq('AMBIGUOUS')))
    # Per-company cache: completed companies are not re-fetched, including those with no regular filings
    perdir=output/'per_stock';perdir.mkdir(exist_ok=True)
    done=[];failed=[]
    for idx,r in enumerate(todo.itertuples(index=False),1):
        code=r.stock_code;corp=r.corp_code
        perfile=perdir/f'{code}_{args.start}_{args.end}.json'
        if perfile.exists():
            try:
                obj=json.loads(perfile.read_text(encoding='utf-8'))
                if obj.get('complete') and obj.get('corp_code')==corp:
                    done.append((code,len(obj.get('filings',[]))))
                    if idx%100==0:print(f'{idx}/{len(todo)} (캐시 사용)')
                    continue
            except (ValueError,KeyError):pass
        try:
            records=dart.filings(corp,args.start,args.end)
            filings=[f for x in records if (f:=parse_filing(x,code,corp)) is not None]
            obj={'complete':True,'stock_code':code,'corp_code':corp,'search_start':args.start,
                 'search_end':args.end,'filings':filings}
            temp=perfile.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8');temp.replace(perfile)
            done.append((code,len(filings)))
        except Exception as e:
            failed.append({'stock_code':code,'corp_code':corp,'error_type':type(e).__name__,'message':str(e)[:500]})
            # Stop on API quota/authorization errors; retry next time after quota issue fixed
            if 'status=020' in str(e) or 'status=010' in str(e) or 'status=011' in str(e) or 'status=012' in str(e):
                print('API 인증/호출 제한 의심, 안전하게 수집 중단:',str(e)[:200]);break
        if idx%50==0:print(f'진행 {idx}/{len(todo)} / 수집완료 {len(done)} / 실패 {len(failed)}')
    all_filings=[]
    for perfile in perdir.glob(f'*_{args.start}_{args.end}.json'):
        try:
            obj=json.loads(perfile.read_text(encoding='utf-8'))
            if obj.get('complete'):all_filings.extend(obj.get('filings',[]))
        except Exception:pass
    cols=['stock_code','corp_code','corp_name','report_nm','report_type','fiscal_year_label','fiscal_month_label',
          'fiscal_period_label','filing_date','rcept_no','is_amendment_name','period_status','rcept_url']
    filingsdf=pd.DataFrame(all_filings,columns=cols)
    if not filingsdf.empty:
        filingsdf=filingsdf.drop_duplicates(['stock_code','rcept_no']).sort_values(['stock_code','filing_date','rcept_no'])
    save_csv(filingsdf,output/'dart_disclosure_history.csv')
    save_csv(pd.DataFrame(failed,columns=['stock_code','corp_code','error_type','message']),output/'collection_failures.csv')
    summary={'universe_stocks':int(universe.shape[0]),'matched_unique_stocks':int(mapping.mapping_status.eq('MATCHED').sum()),
             'unmatched_stocks':int(mapping.mapping_status.eq('UNMATCHED').sum()),
             'ambiguous_mapping_rows':int(mapping.mapping_status.eq('AMBIGUOUS').sum()),
             'attempt_scope_stocks':int(len(todo)), 'completed_stocks_this_run':int(len(done)),
             'failed_requests_this_run':int(len(failed)), 'filing_rows_collected_from_completed_cache':int(len(filingsdf)),
             'filing_stocks':int(filingsdf.stock_code.nunique()) if len(filingsdf) else 0,
             'start':args.start,'end':args.end,
             'disclaimer':'공시 이력 수집만 완료. 최초제출·정정 원문 수치 PIT 검증, 비상장/종목변경 누락, 특정 보고서 시점 검증은 별도 필요. Test 수익률 미사용.'}
    (output/'collection_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('결과 저장:',output)

if __name__=='__main__':main()
