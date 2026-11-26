"""Read-only development H2-8 leakage audit. No model fitting; only development inputs are requested.
PASS is a narrowly scoped consistency check, never a PIT certificate.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

KEY=['stock_id','base_month']
EIGHT=['debt','policy','policy_change','policy_x_debt','debt_change','rate_change','policy_change_x_debt','rate_change_x_debt']
DART='DART_FIRST_FILED_CANDIDATE'; DG='DATAGUIDE_6M_PROXY'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def end(m): return pd.Series(pd.PeriodIndex(m.astype(str),freq='M').to_timestamp('M'),index=m.index)
def prev(m): return (pd.PeriodIndex(m.astype(str),freq='M')-1).astype(str)
def num(s): return pd.to_numeric(s,errors='coerce').astype(float)
def date(s): return pd.to_datetime(s,errors='coerce')
def truth(s): return s.astype('string').str.lower().isin(['true','1','1.0','yes'])
def eq(a,b): return np.isclose(num(a),num(b),rtol=1e-8,atol=1e-8,equal_nan=True)
def require(d,cols):
    if not set(cols)<=set(d): raise ValueError('Missing columns: '+str(sorted(set(cols)-set(d))))

def normalize(d,month):
    d=d.copy(); require(d,['stock_id',month]); d=d.rename(columns={month:'base_month'})
    s=d.stock_id.astype('string').str.strip().str.replace(r'^A(?=\d{6}$)','',regex=True).str.replace(r'\.0$','',regex=True)
    d['stock_id']=s.str.zfill(6); d['base_month']=d.base_month.astype('string')
    if not d.stock_id.str.fullmatch(r'[0-9A-Z]{6}',na=False).all() or not d.base_month.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])',na=False).all() or d.duplicated(KEY).any(): raise ValueError('Invalid or duplicate stock-month keys')
    return d

class Audit:
    def __init__(self): self.rows=[];self.examples=[];self.hashes={};self.details={}
    def add(self,name,status,n,detail): self.rows.append(dict(check=name,status=status,count=int(n),detail=detail))
    def check(self,name,bad,detail,frame=None):
        b=np.asarray(pd.Series(bad).fillna(True),dtype=bool); n=b.sum();self.add(name,'FAIL' if n else 'PASS',n,detail)
        if n and frame is not None:
            c=[x for x in ['stock_id','base_month','target_month','month','model','source_type','source_available_date','source_fiscal_date','dart_next_filing_date','document_id','effective_date'] if x in frame]
            if c:self.examples.append(frame.loc[b,c].head(100).assign(check=name))
    def review(self,name,n,detail): self.add(name,'REVIEW' if n else 'PASS',n,detail)
    def read(self,path,cols=None):
        self.hashes[str(path)]=sha(path)
        if path.suffix=='.parquet':return pd.read_parquet(path,columns=cols)
        return pd.read_csv(path,dtype={'stock_id':'string','source_rcept_no':'string','dart_rcept_no':'string','document_id':'string','sensitivity_source_document_id':'string'},encoding='utf-8-sig',low_memory=False,usecols=cols)
    def section(self,name,fn):
        try:return fn()
        except Exception as e:self.add(name,'BLOCKED',1,f'{type(e).__name__}: {e}');return None

def audit_archive(a,h):
    require(h,KEY+['eligible','debt_raw']+EIGHT)
    a.check('archive_development_only',~h.base_month.between('2001-01','2021-11'),'Base month Jan 2001–Nov 2021 only.',h)
    a.check('archive_eligible',~truth(h.eligible),'All H2 feature rows must be eligible at base month.',h)
    for c in EIGHT+['debt_raw']:
        a.check('finite_'+c,np.isinf(num(h[c]))|(h[c].notna()&num(h[c]).isna()),'No infinite or invalid numeric feature.',h)
    for c in ['policy','policy_change','rate_change']:
        a.check('monthly_constant_'+c,h.groupby('base_month')[c].transform('nunique').gt(1),'Macro feature constant across firms within month.',h)
    for c in ['policy','policy_change','rate_change']:
        a.check('interaction_'+c,~eq(h[c+'_x_debt'],num(h[c])*num(h.debt)),'Interaction uses same base-month components.',h)
    # Reproduce the shared H2 monthly normalization without targets.
    def z(s):
        if s.notna().sum()<3:return s*np.nan
        lo,hi=s.quantile([.01,.99]); v=s.clip(lo,hi); sd=v.std(ddof=0)
        return (v-v.mean())/sd if np.isfinite(sd) and sd>0 else s*np.nan
    expected=num(h.debt_raw).groupby(h.base_month).transform(z)
    a.check('debt_monthly_standardize',~eq(h.debt,expected),'1/99% winsorization + within-base-month ddof=0 z-score.',h)

def audit_debt(a,h,d):
    require(d,KEY+['debt_final_raw','source_type','source_available_date','source_fiscal_date','dart_next_filing_date'])
    j=h[KEY+['debt_raw','debt_change']].merge(d,on=KEY,how='left',validate='one_to_one',indicator=True)
    use=j.debt_raw.notna()|j.debt_change.notna(); cut=end(j.base_month); avail=date(j.source_available_date); fiscal=date(j.source_fiscal_date)
    a.check('hybrid_archive_match',~eq(j.debt_raw,j.debt_final_raw),'Archived debt equals the supplied Hybrid source, including missing values.',j)
    a.check('hybrid_source_exists',use&j._merge.ne('both'),'Every used debt/debt-change row has a source.',j)
    a.check('hybrid_available_date',use&(avail.isna()|avail.gt(cut)),'Declared availability no later than base-month end.',j)
    a.check('hybrid_fiscal_date',use&(fiscal.isna()|fiscal.gt(cut)|fiscal.gt(avail)),'Fiscal end no later than availability/cutoff.',j)
    a.check('hybrid_source_type',use&~j.source_type.isin([DART,DG]),'Recognized Hybrid source.',j)
    dart=use&j.source_type.eq(DART); fallback=use&j.source_type.eq(DG); nxt=date(j.dart_next_filing_date)
    a.check('dart_superseded',dart&(nxt.le(cut)|(j.dart_next_filing_date.notna()&nxt.isna())),'Selected filing not already superseded at cutoff; missing next date is not proof.',j)
    if 'source_rcept_no' in j:
        receipt=pd.to_datetime(j.source_rcept_no.astype('string').str[:8],format='%Y%m%d',errors='coerce')
        a.check('dart_receipt_timing',dart&(receipt.isna()|receipt.gt(cut)|avail.lt(receipt+pd.Timedelta(days=1))),'Receipt date and next-day availability convention.',j)
    else:a.add('dart_receipt_timing','BLOCKED',int(dart.sum()),'source_rcept_no absent.')
    if 'has_unresolved_followup' in j:a.review('dart_unresolved_followup',int((dart&truth(j.has_unresolved_followup)).sum()),'Flagged rows need original filing/revision evidence; flag alone does not prove leakage.')
    else:a.add('dart_unresolved_followup','BLOCKED',1,'Flag column absent.')
    a.review('dataguide_historical_vintage',int(fallback.sum()),'Six-month lag is a timing proxy, not proof of historical values free of later restatements.')
    a.review('dart_original_filing_vintage',int(dart.sum()),'Receipt ID/date alone cannot establish actual archived first-filed values; inspect source reports and correction history.')
    a.details['debt_source_counts']=j.loc[use,'source_type'].value_counts(dropna=False).to_dict()
    flagged=truth(j.has_unresolved_followup) if 'has_unresolved_followup' in j else pd.Series(False,index=j.index)
    a.details['debt_review_rows']=j.loc[use,KEY+['source_type']].assign(dart_selected=dart[use],dataguide_proxy=fallback[use],dart_unresolved=(dart&flagged)[use]).groupby('base_month')[['dart_selected','dataguide_proxy','dart_unresolved']].sum().reset_index()

def audit_debt_change(a,h,d,c):
    require(c,KEY+['debt_change_raw','debt_final_raw','source_available_date','source_fiscal_date','debt_change_status','latest_report_first_seen_month','previous_fiscal_date'])
    j=h[KEY+['debt_change','debt_raw']].merge(c,on=KEY,how='left',validate='one_to_one'); use=j.debt_change.notna();cut=end(j.base_month)
    a.check('debt_change_archive_match',~eq(j.debt_change,j.debt_change_raw),'Archived DebtChange equals raw source CSV.',j)
    a.check('debt_change_current_debt',use&~eq(j.debt_raw,j.debt_final_raw),'Current debt source matches archived Debt.',j)
    a.check('debt_change_dates',use&(date(j.source_available_date).isna()|date(j.source_available_date).gt(cut)|date(j.source_fiscal_date).isna()|date(j.source_fiscal_date).gt(cut)|j.latest_report_first_seen_month.isna()|j.latest_report_first_seen_month.astype('string').gt(j.base_month)),'No future dates/first-seen month.',j)
    a.check('debt_change_status',use&~j.debt_change_status.isin(['new_report','carried_forward']),'Only valid report changes/forward carry.',j)
    new=use&j.debt_change_status.eq('new_report'); old=date(j.previous_fiscal_date)
    a.check('debt_change_previous_period',new&(old.isna()|old.ge(date(j.source_fiscal_date))),'Previous fiscal period strictly older.',j)
    # Independent replay of distinct-report state machine on historical source rows.
    from src.h2.experiments.build_h2_debt_change import compute_debt_change
    x=d.rename(columns={'base_month':'feature_month'}).copy(); x['source_fiscal_date']=date(x.source_fiscal_date); x['source_available_date']=date(x.source_available_date);x['debt_final_raw']=num(x.debt_final_raw)
    x['prediction_date']=end(x.feature_month)
    replay=compute_debt_change(x).rename(columns={'feature_month':'base_month'})
    r=c[KEY+['debt_change_raw','debt_change_status']].merge(replay[KEY+['debt_change_raw','debt_change_status']],on=KEY,how='outer',suffixes=('_csv','_replay'),indicator=True,validate='one_to_one')
    a.check('debt_change_forward_replay',r._merge.ne('both')|~eq(r.debt_change_raw_csv,r.debt_change_raw_replay)|r.debt_change_status_csv.ne(r.debt_change_status_replay),'Chronological replay, including carry-forward. Does not prove vintage.',r)

def audit_policy(a,h,p,release=None):
    require(p,['month','final_rule_v2_sensitivity','sensitivity_source_document_id','sensitivity_source_release_date'])
    p=p.copy();p['month']=p.month.astype(str);p=p.sort_values('month')
    if p.month.duplicated().any():raise ValueError('duplicate policy months')
    idx=pd.period_range(p.month.min(),p.month.max(),freq='M').astype(str);s=p.set_index('month').final_rule_v2_sensitivity.reindex(idx)
    j=h[KEY+['policy','policy_change']].merge(p.rename(columns={'month':'base_month'}),on='base_month',how='left',validate='many_to_one')
    a.check('policy_archive_match',j.final_rule_v2_sensitivity.isna()|~eq(j.policy,j.final_rule_v2_sensitivity),'Policy equals the mapped source month.',j)
    dt=date(j.sensitivity_source_release_date);a.check('policy_release_cutoff',dt.isna()|dt.ge(end(j.base_month))|j.sensitivity_source_document_id.isna(),'Release strictly before month-end (next-day convention).',j)
    a.check('policy_change_previous_calendar_month',~eq(j.policy_change,j.base_month.map(s.diff())),'Policy(t)-Policy(t-1 calendar month).',j)
    used=p[p.month.isin(h.base_month.unique())].copy()
    a.review('policy_rule_and_text_vintage',1,'Score/release mapping consistency alone cannot establish original text vintage or rule selection independent of future outcomes.')
    if release is None:a.add('policy_release_evidence','BLOCKED',len(used),'release_dates.csv missing: cannot distinguish verified headers from inferred meeting dates.')
    else:
        require(release,['document_id','release_date_sensitivity','release_date_sensitivity_source'])
        q=used.merge(release,left_on='sensitivity_source_document_id',right_on='document_id',how='left',validate='many_to_one',indicator=True)
        a.check('policy_release_evidence_match',q._merge.ne('both')|date(q.sensitivity_source_release_date).ne(date(q.release_date_sensitivity)),'Monthly mapping agrees with release evidence.',q)
        unverified=~q.release_date_sensitivity_source.eq('verified_document_header');a.review('policy_inferred_release',int(unverified.sum()),'Inferred same-day release needs dated original BOK document or official release evidence.')
        a.details['policy_release_status_months']=q.release_date_sensitivity_source.value_counts(dropna=False).to_dict()
        a.details['policy_review_rows']=q.loc[unverified,['month','sensitivity_source_document_id','sensitivity_source_release_date','release_date_sensitivity_source']]

def audit_rates(a,h,r,e):
    require(r,['feature_month','rate_pct','rate_change_pp']);require(e,['effective_date','rate_after_pct'])
    r=r.sort_values('feature_month').copy();r['feature_month']=r.feature_month.astype(str);e=e.copy();e['effective_date']=date(e.effective_date);e=e.sort_values('effective_date')
    if r.feature_month.gt('2021-12').any() or e.effective_date.gt(pd.Timestamp('2021-12-31')).any():raise PermissionError('Rate sources outside development')
    if list(r.feature_month)!=list(pd.period_range(r.feature_month.min(),r.feature_month.max(),freq='M').astype(str)):raise ValueError('Rate months not consecutive')
    if e.effective_date.isna().any() or e.effective_date.duplicated().any() or r.feature_month.duplicated().any():raise ValueError('Invalid rate keys')
    months=pd.period_range(pd.Period(r.feature_month.min(),'M')-1,r.feature_month.max(),freq='M'); q=pd.DataFrame({'feature_month':months.astype(str),'cutoff':months.to_timestamp('M')})
    q=pd.merge_asof(q,e[['effective_date','rate_after_pct']],left_on='cutoff',right_on='effective_date',direction='backward');q['change']=num(q.rate_after_pct).diff()
    t=r.merge(q,on='feature_month',validate='one_to_one');a.check('rate_event_replay',~eq(t.rate_pct,t.rate_after_pct)|~eq(t.rate_change_pp,t.change)|t.effective_date.isna()|t.effective_date.gt(t.cutoff),'Backward as-of event replay; net monthly change.',t)
    expected=r.set_index('feature_month').rate_change_pp.astype(float).copy();expected.iloc[0]=np.nan
    a.check('rate_archive_match',~eq(h.rate_change,h.base_month.map(expected)),'Archived RateChange matches base-month rate; first source month missing by design.',h)
    a.review('rate_original_source_evidence',1,'Local dated event table consistency is not independent verification of official publication/effective dates.')
    return r.set_index('feature_month').rate_change_pp.astype(float)

def audit_fit(a,f,r):
    require(f,['features','profile','train_month_list','eval_month_list','valid_year','feature_names','train_start','train_end','train_months','eval_months'])
    f=f[(num(f.features)==8)&f.profile.eq('original')];bad=[];gate=[];names=[]
    if f.empty:raise ValueError('No original 8-feature fits')
    for row in f.itertuples(index=False):
        tr=str(row.train_month_list).split(';');va=str(row.eval_month_list).split(';');year=int(row.valid_year)
        bad.append(len(set(tr))!=int(row.train_months) or len(set(va))!=int(row.eval_months) or min(tr)!=row.train_start or max(tr)!=row.train_end or max(tr)>f'{year-2}-12' or any(not v.startswith(str(year)+'-') or v>'2021-12' for v in va))
        gate.append(not pd.Series(prev(pd.Series(tr+va))).map(r).gt(0).all());names.append(set(row.feature_names.split(','))!={'x_'+c for c in EIGHT})
    a.check('fit_chronology',bad,'Training targets end by Y-2; held-out year Y; no test years.')
    a.check('fit_gate_base_month',gate,'All trained/predicted expert months have prior-month net positive rate change.')
    a.check('fit_feature_whitelist',names,'Only eight H2 inputs; no targets or future columns.')

def audit_universe(a,h,panel,elig,pred,rate):
    require(panel,KEY+['return','eligible']);require(elig,KEY+['eligible']);require(pred,KEY+['target_month','model','features','profile','trees','without_h2','h2_score','final_score','y'])
    p=pred[(num(pred.features)==8)&pred.profile.eq('original')&(num(pred.trees)==500)].copy()
    if p.empty or p.duplicated(['model']+KEY).any():raise ValueError('Missing/duplicate original active predictions')
    a.check('prediction_target_alignment',np.asarray(prev(p.target_month))!=p.base_month.astype(str).to_numpy(),'Target is exactly next calendar month.',p)
    a.check('prediction_gate',~p.base_month.map(rate).gt(0),'Active H2 only on positive base-month net change.',p)
    y=panel[KEY+['return']].rename(columns={'base_month':'target_month','return':'actual'})
    v=p.merge(y,on=['stock_id','target_month'],how='left',validate='many_to_one');a.check('prediction_target_values',v.actual.isna()|~eq(v.y,v.actual),'Evaluation target matches next month actual return.',v)
    e=elig[KEY+['eligible']].rename(columns={'eligible':'exante_eligible'}); b=panel[KEY+['eligible']].merge(e,on=KEY,how='left',validate='one_to_one',indicator=True)
    a.check('panel_eligibility_alignment',b._merge.ne('both')|truth(b.eligible).ne(truth(b.exante_eligible)),'Panel eligibility agrees with stored base-month eligibility.',b)
    a.review('eligibility_unknown',int(b.exante_eligible.isna().sum()),'Unknown eligibility requires separate coverage review; not silently deemed eligible.')
    ae=e[truth(e.exante_eligible)]; actual_h=h[KEY]; overlap=ae[ae.base_month.isin(h.base_month.unique())].merge(actual_h,on=KEY,how='outer',indicator=True)
    a.check('archive_eligible_universe',overlap._merge.ne('both'),'Archive matches base-month eligible rows.',overlap)
    rows=[]
    for (model,m),g in p.groupby(['model','base_month']):
        universe=ae[ae.base_month.eq(m)][KEY]; q=universe.merge(g[KEY],on=KEY,how='outer',indicator=True); target=g.target_month.iloc[0]
        miss=q[q._merge.eq('left_only')][KEY].assign(target_month=target).merge(y,on=['stock_id','target_month'],how='left',validate='one_to_one')
        b=g.without_h2.rank();z=g.h2_score.rank(); expected=(95*b+5*z)/(100*(len(g)+1))
        a.check('blend_replay_'+str(model)+'_'+str(m),~eq(g.final_score,expected),'Reproduce old evaluation-only rank blend.',g)
        rows.append(dict(model=model,base_month=m,target_month=target,exante_eligible=len(universe),evaluation_scores=len(g),eligible_missing=int(q._merge.eq('left_only').sum()),missing_with_no_future_return=int(miss.actual.isna().sum()),evaluation_outside_eligible=int(q._merge.eq('right_only').sum())))
    tab=pd.DataFrame(rows);a.details['universe_rows']=tab
    a.review('evaluation_vs_exante_universe',int((tab.eligible_missing+tab.evaluation_outside_eligible).sum()),'Counts by model-month, not unique firms. Differences require scores generated before outcome filtering.')
    a.add('deployment_rank_order','BLOCKED',1,'Existing development path filters realized target availability before blending. Implement ex-ante prediction universe and rank before joining labels; do not submit these scores directly.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/leakage_audit_v1'))
    for n,v in [('archive','data/processed/archive/h2_8features.parquet'),('hybrid','data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv'),('debt-change','data/processed/final/debt_hybrid/h2_debt_change_2001_2021.csv'),('rates','data/raw/h2/bok_rate_monthly_2001_2021.csv'),('events','data/raw/h2/bok_rate_events_through_2021.csv'),('diagnostic','outputs/model_results/h2/overfit_diagnostic_v1')]:p.add_argument('--'+n,type=Path,default=Path(v))
    p.add_argument('--release-evidence',type=Path);args=p.parse_args();root=args.root.resolve();out=root/args.out
    if out.exists():raise FileExistsError(out)
    a=Audit();cfg=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8'));paths=cfg['PATHS'];diag=root/args.diagnostic
    load=lambda path,month:a.section('load_'+Path(path).name,lambda:normalize(a.read(root/path),month))
    h=load(args.archive,'base_month');d=load(args.hybrid,'feature_month');c=load(args.debt_change,'feature_month')
    # These are development-specific inputs only. Reject unexpected future rows before computation.
    for name,frame in [('archive',h),('hybrid',d),('debt_change',c)]:
        if frame is not None and (frame.base_month.gt('2021-12')|frame.base_month.lt('2000-01')).any():raise PermissionError(name+' outside development dates')
    ref=a.section('load_run_audit',lambda:json.loads((diag/'run_audit.json').read_text(encoding='utf-8-sig')))
    if ref is not None:
        a.hashes[str(diag/'run_audit.json')]=sha(diag/'run_audit.json');a.check('reference_test_unused',[ref.get('test_used') is not False],'Recorded run declares no test use; declaration only.')
        a.check('reference_archive_hash',[h is None or a.hashes.get(str(root/args.archive))!=ref.get('archive',{}).get('sha256')],'Same archive bytes as overfit run.')
        a.check('reference_model_settings',[ref.get('alpha')!=.05 or ref.get('rule')!='hike_1m' or ref.get('main_run',{}).get('trees')!=500],'Frozen 5% gate and 500 trees.')
        for raw_path,expected in ref.get('code_sha256',{}).items():
            rel=raw_path.replace(chr(92),'/').split('/src/',1)
            if len(rel)!=2:continue
            path=root/'src'/rel[1]
            if not path.is_file():a.add('code_hash_'+path.name,'BLOCKED',1,str(path));continue
            actual=sha(path);a.hashes[str(path)]=actual;a.check('code_hash_'+path.name,[actual!=expected],'Code SHA256 of prior diagnostic dependencies.')
        for model, rec in ref.get('baseline_runs',{}).items():
            for filename, expected in rec.get('sha256',{}).items():
                path=diag.parent/f'conditional_full_{model}_v1'/filename
                if not path.is_file():a.add('baseline_hash_'+model+'_'+filename,'BLOCKED',1,str(path));continue
                actual=sha(path);a.hashes[str(path)]=actual;a.check('baseline_hash_'+model+'_'+filename,[actual!=expected],'Frozen baseline artifact SHA256.')
    if h is not None:a.section('archive_checks',lambda:audit_archive(a,h))
    if h is not None and d is not None:a.section('debt_checks',lambda:audit_debt(a,h,d))
    if h is not None and d is not None and c is not None:a.section('debt_change_checks',lambda:audit_debt_change(a,h,d,c))
    policy=a.section('load_policy',lambda:a.read(root/paths['h2_monthly_csv']))
    candidates=[root/args.release_evidence] if args.release_evidence else [root/'data/processed/final/policy/release_dates.csv',root/'data/raw/h2/release_dates.csv',root/'data/processed/final/release_dates.csv']
    release_path=next((x for x in candidates if x.is_file()),None);release=a.section('load_release_evidence',lambda:a.read(release_path)) if release_path else None
    if policy is not None and 'month' in policy and policy.month.astype(str).gt('2021-12').any():raise PermissionError('Policy outside development')
    if policy is not None and h is not None:a.section('policy_checks',lambda:audit_policy(a,h,policy,release))
    r=a.section('load_rate_monthly',lambda:a.read(root/args.rates));e=a.section('load_rate_events',lambda:a.read(root/args.events))
    rate=None
    if r is not None and e is not None and h is not None:rate=a.section('rate_checks',lambda:audit_rates(a,h,r,e))
    f=a.section('load_fit_audit',lambda:a.read(diag/'fit_audit.csv'))
    if f is not None and rate is not None:a.section('fit_checks',lambda:audit_fit(a,f,rate))
    panel=load(Path(paths['panel']),'month');elig=load(Path(paths['eligibility']),'month')
    for name,frame in [('panel',panel),('eligibility',elig)]:
        if frame is not None and frame.base_month.gt('2021-12').any():raise PermissionError(name+' contains test months')
    pred=a.section('load_active_predictions',lambda:a.read(diag/'active_predictions.csv'))
    if pred is not None and pred.target_month.astype(str).gt('2021-12').any():raise PermissionError('test predictions')
    if all(x is not None for x in [h,panel,elig,pred,rate]):a.section('universe_checks',lambda:audit_universe(a,h,panel,elig,pred,rate))
    a.add('final_submission_not_certified','BLOCKED',1,'Development diagnostic only; no fixed production model, no final prediction file, no independent PIT proof audited.')
    tab=pd.DataFrame(a.rows);out.mkdir(parents=True,exist_ok=False);tab.to_csv(out/'leakage_checks.csv',index=False,encoding='utf-8-sig')
    pd.concat(a.examples,ignore_index=True).to_csv(out/'violation_examples.csv',index=False,encoding='utf-8-sig') if a.examples else pd.DataFrame(columns=['check']).to_csv(out/'violation_examples.csv',index=False,encoding='utf-8-sig')
    for key,name in [('universe_rows','universe_by_month.csv'),('debt_review_rows','debt_review_by_month.csv'),('policy_review_rows','policy_release_review.csv')]:
        if key in a.details:a.details.pop(key).to_csv(out/name,index=False,encoding='utf-8-sig')
    (out/'leakage_audit.json').write_text(json.dumps(dict(status_counts=tab.status.value_counts().to_dict(),inputs_sha256=a.hashes,details=a.details,script_sha256=sha(Path(__file__)),test_used=False,submission_certified=False),ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(tab.to_string(index=False));print('\nSaved:',out);print('PASS = scoped check only; REVIEW/BLOCKED must be resolved before submission.')
    if tab.status.eq('FAIL').any():raise SystemExit(2)

if __name__=='__main__':main()
