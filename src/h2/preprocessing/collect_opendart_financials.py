"""OpenDART 재무항목 원천 수집 (소수 기업 파일럿부터).

필요: 환경변수 DART_API_KEY, 회사 매핑 CSV(stock_code,corp_code).
실행 전 반드시 --check-only, 소수 기업으로 --execute 테스트.
OpenDART 전체 재무제표 API는 2015년 이후만 제공하며 과거 PIT 재현을 보장하지 않음.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json'
REPORTS = {'q1':'11013','h1':'11012','q3':'11014','annual':'11011'}
ACCOUNT_KEYWORDS = (
    '유동부채','부채총계','자산총계','단기차입금','유동성장기',
    '이자비용','금융비용','영업이익','현금및현금성자산',
)

def get_companies(path, limit):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'stock_code','corp_code'}.issubset(reader.fieldnames or []):
            raise ValueError('회사 매핑 CSV에는 stock_code,corp_code 두 컬럼이 필요합니다.')
        rows=list(reader)
    output=[]
    seen=set()
    for row in rows:
        stock=str(row['stock_code']).strip().replace('A','',1) if str(row['stock_code']).strip().startswith('A') else str(row['stock_code']).strip()
        stock=stock.zfill(6)
        corp=str(row['corp_code']).strip().zfill(8)
        if len(stock)!=6 or not stock.isdigit() or len(corp)!=8 or not corp.isdigit():
            raise ValueError('잘못된 회사 매핑: '+str(row))
        if (stock,corp) not in seen:
            seen.add((stock,corp)); output.append((stock,corp))
    if not output: raise ValueError('기업 목록이 비어 있습니다')
    return output[:limit]

def api_request(key,corp,year,report,fs_div,timeout):
    q=urlencode({'crtfc_key':key,'corp_code':corp,'bsns_year':year,'reprt_code':report,'fs_div':fs_div})
    request=Request(BASE+'?'+q,headers={'User-Agent':'finance-analytics-research/1.0'})
    try:
        with urlopen(request,timeout=timeout) as response:
            return json.load(response)
    except Exception as exc:
        raise RuntimeError(f'DART 연결 오류 corp={corp}, year={year}, report={report}: {exc}') from exc

def main():
    p=argparse.ArgumentParser(description='OpenDART 기업 재무항목 원천 수집 (안전한 파일럿)')
    p.add_argument('--company-map',type=Path,required=True,help='stock_code,corp_code 컬럼이 포함된 CSV')
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--start-year',type=int,default=2019)
    p.add_argument('--end-year',type=int,default=2021)
    p.add_argument('--max-companies',type=int,default=5)
    p.add_argument('--fs-div',choices=['CFS','OFS'],default='CFS')
    p.add_argument('--sleep',type=float,default=0.2)
    p.add_argument('--timeout',type=int,default=20)
    p.add_argument('--out',type=Path,default=Path('data/raw/h2/opendart_financials_pilot.csv'))
    p.add_argument('--execute',action='store_true',help='실제 API 요청과 파일 저장')
    p.add_argument('--check-only',action='store_true',help='수집 계획만 표시')
    a=p.parse_args()
    if not (2015<=a.start_year<=a.end_year<=2021): raise ValueError('OpenDART API 제공범위(2015~) 및 학습구간(2021까지)을 확인하세요.')
    if a.max_companies<1 or a.sleep<0:raise ValueError('기업수·대기시간 설정 오류')
    companies=get_companies(a.root.resolve()/a.company_map,a.max_companies)
    output=a.root.resolve()/a.out
    count=len(companies)*(a.end_year-a.start_year+1)*len(REPORTS)
    print('기업 수:',len(companies),'연도:',a.start_year,'~',a.end_year,'요청 최대:',count,'재무제표:',a.fs_div)
    print('출력 경로:',output)
    if not a.execute:
        print('[점검 완료] 네트워크 요청 및 파일 저장 없음. --execute로 수행')
        return
    key=os.environ.get('DART_API_KEY','').strip()
    if not key: raise RuntimeError('DART_API_KEY 환경변수를 설정하세요. 키를 코드나 Git에 저장하지 마세요.')
    logpath=output.with_name(output.stem+'_audit.json')
    if output.exists() or logpath.exists(): raise FileExistsError(f'기존 파일 덮어쓰기 금지: {[str(x) for x in (output, logpath) if x.exists()]}')
    rows=[]; logs=[]
    collected_at=datetime.now(timezone.utc).isoformat()
    for stock,corp in companies:
        for year in range(a.start_year,a.end_year+1):
            for name,rc in REPORTS.items():
                payload=api_request(key,corp,year,rc,a.fs_div,a.timeout)
                status=str(payload.get('status',''))
                logs.append({'stock_code':stock,'corp_code':corp,'bsns_year':year,'reprt_code':rc,'status':status,'message':payload.get('message','')})
                if status=='000':
                    for item in payload.get('list',[]):
                        nm=str(item.get('account_nm',''))
                        if any(k in nm for k in ACCOUNT_KEYWORDS):
                            rows.append({
                                'stock_code':stock,'corp_code':corp,'bsns_year':year,'report_type':name,
                                'reprt_code':rc,'fs_div':a.fs_div,'rcept_no':item.get('rcept_no',''),
                                'sj_div':item.get('sj_div',''),'account_id':item.get('account_id',''),
                                'account_nm':nm,'account_detail':item.get('account_detail',''),
                                'thstrm_amount':item.get('thstrm_amount',''),
                                'thstrm_add_amount':item.get('thstrm_add_amount',''),
                                'currency':item.get('currency',''),'collected_at_utc':collected_at,
                            })
                elif status!='013':
                    raise RuntimeError(f'API 상태 {status}: {payload.get("message")} / {stock} {year} {name}; 파일 미저장')
                time.sleep(a.sleep)
    # 원천만 저장. 계정별 정의·비율·PIT 적용은 이후 별도 검증.
    fieldnames=['stock_code','corp_code','bsns_year','report_type','reprt_code','fs_div','rcept_no','sj_div','account_id','account_nm','account_detail','thstrm_amount','thstrm_add_amount','currency','collected_at_utc']
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
    logpath.write_text(json.dumps({'source':BASE,'request_log':logs,'rows':len(rows),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'limitations':['2015+ only','API response not historical PIT','interest expense account can be absent or differently named','stock-to-corp mapping must be validated']},ensure_ascii=False,indent=2),encoding='utf-8')
    print('수집 완료. 선택된 재무 계정 행',len(rows),'출력:',output)

if __name__=='__main__':main()
