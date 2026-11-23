"""한국은행 공식 기준금리 변경표(2008-02까지 콜금리 목표)에서 월별 금리 변화 생성.

프로젝트 루트에서: python -m src.h2.collection.build_bok_rate_history --check-only
실제 저장:       python -m src.h2.collection.build_bok_rate_history --execute
공식 출처: https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200761
변경일·변경후 금리를 공식 표에서 전사한 고정 스냅샷. 후속 변경 시 코드 리뷰 필요.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

SOURCE_URL = 'https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=200761'
# Official table, ascending chronological order. Event rate = rate in force after the stated effective date.
EVENTS = '''
1999-05-06,4.75
2000-02-10,5.00
2000-10-05,5.25
2001-02-08,5.00
2001-07-05,4.75
2001-08-09,4.50
2001-09-19,4.00
2002-05-07,4.25
2003-05-13,4.00
2003-07-10,3.75
2004-08-12,3.50
2004-11-11,3.25
2005-10-11,3.50
2005-12-08,3.75
2006-02-09,4.00
2006-06-08,4.25
2006-08-10,4.50
2007-07-12,4.75
2007-08-09,5.00
2008-08-07,5.25
2008-10-09,5.00
2008-10-27,4.25
2008-11-07,4.00
2008-12-11,3.00
2009-01-09,2.50
2009-02-12,2.00
2010-07-09,2.25
2010-11-16,2.50
2011-01-13,2.75
2011-03-10,3.00
2011-06-10,3.25
2012-07-12,3.00
2012-10-11,2.75
2013-05-09,2.50
2014-08-14,2.25
2014-10-15,2.00
2015-03-12,1.75
2015-06-11,1.50
2016-06-09,1.25
2017-11-30,1.50
2018-11-30,1.75
2019-07-18,1.50
2019-10-16,1.25
2020-03-17,0.75
2020-05-28,0.50
2021-08-26,0.75
2021-11-25,1.00
'''.strip()


def make():
    rows = [line.split(',') for line in EVENTS.splitlines()]
    ev = pd.DataFrame(rows, columns=['effective_date','rate_after_pct'])
    ev['effective_date'] = pd.to_datetime(ev.effective_date, format='%Y-%m-%d')
    ev['rate_after_pct'] = pd.to_numeric(ev.rate_after_pct)
    assert ev.effective_date.is_monotonic_increasing and not ev.effective_date.duplicated().any()
    ev['rate_before_pct'] = ev.rate_after_pct.shift()
    ev['event_change_pp'] = ev.rate_after_pct - ev.rate_before_pct
    ev['regime'] = ev.effective_date.map(lambda x: 'call_rate_target' if x < pd.Timestamp('2008-03-01') else 'bok_base_rate')
    ev['source_url'] = SOURCE_URL
    months = pd.period_range('2000-12','2021-12',freq='M')
    month_end = months.to_timestamp(how='end').normalize()
    asof = pd.merge_asof(pd.DataFrame({'date':month_end}), ev.rename(columns={'effective_date':'event_date'}), left_on='date', right_on='event_date', direction='backward')
    if asof.rate_after_pct.isna().any():
        raise ValueError('월말 금리 상태를 알 수 없는 달이 있습니다.')
    d = pd.DataFrame({'feature_month':months.astype(str),'rate_pct':asof.rate_after_pct.to_numpy()})
    d['rate_change_pp'] = d.rate_pct.diff()
    d['rate_change_bp'] = d.rate_change_pp * 100
    d['regime'] = pd.Series(months).map(lambda x: 'call_rate_target' if x < pd.Period('2008-03','M') else 'bok_base_rate').values
    d['source_url'] = SOURCE_URL
    d = d[d.feature_month.between('2001-01','2021-12')].reset_index(drop=True)
    if len(d) != 252 or d.rate_change_pp.isna().any():
        raise ValueError('2001~2021 252개월 완전성 검사 실패')
    return ev, d


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('data/raw/h2/bok_rate_monthly_2001_2021.csv'))
    p.add_argument('--events-out',type=Path,default=Path('data/raw/h2/bok_rate_events_through_2021.csv'))
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    ev,d=make()
    print('금리변경 사건:',len(ev),'| 월별:',len(d),'| 인상/인하 월:',int((d.rate_change_pp>0).sum()),int((d.rate_change_pp<0).sum()))
    print('2008-03 전후 정책금리 제도 구분:',d.groupby('regime').size().to_dict())
    print(d.tail(5).to_string(index=False))
    if not a.execute:
        print('[점검 완료] 실제 파일 미생성: 저장하려면 --execute')
        return
    root=a.root.resolve()
    paths=[root/a.out,root/a.events_out]
    if any(x.exists() for x in paths): raise FileExistsError('기존 파일 덮어쓰기 금지: '+str([str(x) for x in paths if x.exists()]))
    for x in paths: x.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(paths[0],index=False,encoding='utf-8-sig')
    ev.to_csv(paths[1],index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    for x in paths: print('saved',x,'sha256',hashlib.sha256(x.read_bytes()).hexdigest())

if __name__=='__main__': main()
