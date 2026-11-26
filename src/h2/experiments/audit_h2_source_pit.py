"""Read-only H2 vintage evidence triage (development only, no fitting).
Reconstruct exact H2 fit keys; include whole eligible cross-section and DebtChange ancestry.
Metadata matching is not original-value PIT certification.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np

KEY=['stock_id','base_month']
DART='DART_FIRST_FILED_CANDIDATE'; DG='DATAGUIDE_6M_PROXY'
# Independent transcription of BOK official historical rate table, through 2021 only.
RATE_URL='https://www.bok.or.kr/portal/singl/baseRate/list.do?dataSeCd=01&menuNo=20064'
RATE_TEXT='''1999-05-06 4.75
2000-02-10 5.00
2000-10-05 5.25
2001-02-08 5.00
2001-07-05 4.75
2001-08-09 4.50
2001-09-19 4.00
2002-05-07 4.25
2003-05-13 4.00
2003-07-10 3.75
2004-08-12 3.50
2004-11-11 3.25
2005-10-11 3.50
2005-12-08 3.75
2006-02-09 4.00
2006-06-08 4.25
2006-08-10 4.50
2007-07-12 4.75
2007-08-09 5.00
2008-08-07 5.25
2008-10-09 5.00
2008-10-27 4.25
2008-11-07 4.00
2008-12-11 3.00
2009-01-09 2.50
2009-02-12 2.00
2010-07-09 2.25
2010-11-16 2.50
2011-01-13 2.75
2011-03-10 3.00
2011-06-10 3.25
2012-07-12 3.00
2012-10-11 2.75
2013-05-09 2.50
2014-08-14 2.25
2014-10-15 2.00
2015-03-12 1.75
2015-06-11 1.50
2016-06-09 1.25
2017-11-30 1.50
2018-11-30 1.75
2019-07-18 1.50
2019-10-16 1.25
2020-03-17 0.75
2020-05-28 0.50
2021-08-26 0.75
2021-11-25 1.00'''
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def require(d,cols):
    if not set(cols)<=set(d):raise ValueError('Missing columns '+str(sorted(set(cols)-set(d))))
def read(p):
    if p.suffix=='.parquet':return pd.read_parquet(p)
    return pd.read_csv(p,dtype={k:'string' for k in ['stock_id','stock_code','source_rcept_no','rcept_no','dart_rcept_no','sensitivity_source_document_id','document_id']},encoding='utf-8-sig',low_memory=False)
def norm(d,month='base_month'):
    d=d.rename(columns={month:'base_month'}).copy(); require(d,KEY)
    d.stock_id=d.stock_id.astype('string').str.strip().str.replace(r'^A(?=\d{6}$)','',regex=True).str.replace(r'\.0$','',regex=True).str.zfill(6)
    d.base_month=d.base_month.astype('string')
    if not d.stock_id.str.fullmatch(r'[0-9A-Z]{6}',na=False).all() or not d.base_month.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])',na=False).all() or d.duplicated(KEY).any():raise ValueError('Invalid/duplicate keys')
    if not d.base_month.between('2000-01','2021-12').all():raise PermissionError('Development only')
    return d
def previous(s):return (pd.PeriodIndex(s.astype(str),freq='M')-1).astype(str)
def end(s):return pd.PeriodIndex(s.astype(str),freq='M').to_timestamp('M')
def same(a,b):return np.isclose(pd.to_numeric(a,errors='coerce').to_numpy(dtype=float,na_value=np.nan),pd.to_numeric(b,errors='coerce').to_numpy(dtype=float,na_value=np.nan),rtol=1e-9,atol=1e-9,equal_nan=True)
def flag(s):return s.astype('string').str.lower().isin(['true','1','1.0','yes'])
def rate_compare(local):
    ref=pd.DataFrame([x.split() for x in RATE_TEXT.splitlines()],columns=['effective_date','official_rate_pct']);ref.official_rate_pct=ref.official_rate_pct.astype(float)
    require(local,['effective_date','rate_after_pct']);local=local[['effective_date','rate_after_pct']].copy();local.effective_date=pd.to_datetime(local.effective_date,errors='raise').dt.strftime('%Y-%m-%d')
    if local.effective_date.duplicated().any() or local.effective_date.gt('2021-12-31').any():raise ValueError('Duplicate/future rate event')
    q=ref.merge(local,on='effective_date',how='outer',indicator=True,validate='one_to_one');q['status']=np.where(q._merge.eq('both')&same(q.official_rate_pct,q.rate_after_pct),'PASS','FAIL');q['official_source']=RATE_URL
    return q.drop(columns='_merge').sort_values('effective_date')

def trace_debt(h,d,c,scope):
    """Rows of all relevant cross-sections, plus historical source rows used by DebtChange."""
    require(h,KEY+['debt_raw','debt_change']);require(d,KEY+['debt_final_raw','source_type','source_fiscal_date','source_available_date','source_rcept_no','has_unresolved_followup']);require(c,KEY+['debt_change_raw','debt_change_status','latest_report_first_seen_month','previous_fiscal_date','previous_debt_raw'])
    if not scope[KEY].merge(h[KEY],on=KEY,how='left',indicator=True,validate='one_to_one')._merge.eq('both').all():raise ValueError('Direct fit/predict key missing from archive')
    a=h[h.base_month.isin(scope.base_month)].merge(scope,on=KEY,how='left',validate='one_to_one');a[['train','predict']]=a[['train','predict']].fillna(False).astype(bool)
    q=a.merge(d,on=KEY,how='left',validate='one_to_one').merge(c[KEY+['debt_change_raw','debt_change_status','latest_report_first_seen_month','previous_fiscal_date','previous_debt_raw']],on=KEY,how='left',validate='one_to_one')
    if not same(q.debt_raw,q.debt_final_raw).all() or not same(q.debt_change,q.debt_change_raw).all():raise ValueError('Archive/source values differ')
    # All monthly eligible peers matter for debt z-score and DebtChange rank transform.
    q['scope']=np.select([q.train&q.predict,q.train,q.predict],['train_and_predict','train','predict'],default='cross_section_peer')
    q['debt_available']=q.debt_raw.notna();q['change_available']=q.debt_change.notna()
    used=q[q.debt_available|q.change_available].copy()
    # DebtChange references the first-seen new report and previous distinct state report.
    origins=c[c.debt_change_status.eq('new_report')][KEY+['previous_fiscal_date','previous_debt_raw','debt_change_raw']].rename(columns={'base_month':'origin_month','debt_change_raw':'origin_change'})
    needs=q[q.change_available][KEY+['scope','debt_change','latest_report_first_seen_month']].rename(columns={'base_month':'use_month','latest_report_first_seen_month':'origin_month'})
    t=needs.merge(origins,on=['stock_id','origin_month'],how='left',validate='many_to_one')
    if t.previous_fiscal_date.isna().any() or not same(t.debt_change,t.origin_change).all() or t.origin_month.gt(t.use_month).any():raise ValueError('Unresolved DebtChange origin')
    state=c[c.debt_change_status.isin(['new_report','first_report'])][KEY].merge(d,on=KEY,validate='one_to_one')
    state['fiscal_key']=pd.to_datetime(state.source_fiscal_date,errors='raise').dt.strftime('%Y-%m-%d');t['fiscal_key']=pd.to_datetime(t.previous_fiscal_date,errors='raise').dt.strftime('%Y-%m-%d')
    if state.duplicated(['stock_id','fiscal_key']).any():raise ValueError('Ambiguous state report')
    pr=state[['stock_id','fiscal_key','base_month','debt_final_raw']].rename(columns={'base_month':'previous_seen_month','debt_final_raw':'previous_value'})
    t=t.merge(pr,on=['stock_id','fiscal_key'],how='left',validate='many_to_one')
    if t.previous_seen_month.isna().any() or t.previous_seen_month.ge(t.origin_month).any() or not same(t.previous_debt_raw,t.previous_value).all():raise ValueError('Unresolved previous state report')
    # Include source selected at both report origins; current row may have different provenance.
    dep=pd.concat([t[['stock_id','origin_month']].rename(columns={'origin_month':'base_month'}).assign(dependency='change_current_report'),t[['stock_id','previous_seen_month']].rename(columns={'previous_seen_month':'base_month'}).assign(dependency='change_previous_report')]).drop_duplicates()
    dep=dep.merge(d,on=KEY,how='left',validate='many_to_one')
    if dep.debt_final_raw.isna().any():raise ValueError('Missing ancestry source')
    cols=KEY+['source_type','source_fiscal_date','source_available_date','source_rcept_no','debt_final_raw','has_unresolved_followup']
    for x in ['dart_next_filing_date','source_confidence']:
        if x in d:cols.append(x)
    src=pd.concat([used[cols].assign(dependency='monthly_debt'),dep[cols+['dependency']]],ignore_index=True).drop_duplicates()
    src['source_rcept_no']=src.source_rcept_no.fillna('');src['unresolved']=flag(src.has_unresolved_followup)
    if not src.source_type.isin([DART,DG]).all():raise ValueError('Unexpected source type')
    if not (pd.to_datetime(src.source_available_date,errors='raise')<=end(src.base_month)).all():raise ValueError('Future source date')
    group=['stock_id','source_type','source_fiscal_date','source_rcept_no','debt_final_raw']
    reports=src.groupby(group,dropna=False,as_index=False).agg(first_use_month=('base_month','min'),last_use_month=('base_month','max'),source_rows=('base_month','nunique'),unresolved=('unresolved','max'),dependency_types=('dependency',lambda x:';'.join(sorted(set(x)))))
    reports['review_status']='REVIEW_ORIGINAL_VALUE'
    return q,t,src,reports

def policy_scope(p,release,months):
    require(p,['month','sensitivity_source_document_id','sensitivity_source_release_date','final_rule_v2_sensitivity']);p=p.copy();p.month=p.month.astype(str)
    if p.month.duplicated().any() or p.month.gt('2021-12').any():raise ValueError('Invalid policy months')
    m=pd.DataFrame({'base_month':sorted(set(months))});m['previous_month']=previous(m.base_month)
    s=pd.concat([m[['base_month']].rename(columns={'base_month':'month'}).assign(role='current_policy'),m[['previous_month']].rename(columns={'previous_month':'month'}).assign(role='previous_policy')]).drop_duplicates()
    s=s.merge(p,on='month',how='left',validate='many_to_one');require(release,['document_id','release_date_sensitivity','release_date_sensitivity_source'])
    r=release[['document_id','release_date_sensitivity','release_date_sensitivity_source']].copy();r.document_id=r.document_id.astype('string')
    if r.document_id.duplicated().any():raise ValueError('Duplicate release document')
    s=s.merge(r,left_on='sensitivity_source_document_id',right_on='document_id',how='left',validate='many_to_one')
    matched=s.document_id.notna(); dates=pd.to_datetime(s.release_date_sensitivity,errors='coerce'); mapped=pd.to_datetime(s.sensitivity_source_release_date,errors='coerce')
    if (matched & (dates.isna() | dates.ne(mapped) | (dates>=end(s.month)))).any():raise ValueError('Policy release mapping/cutoff mismatch')
    s['status']=np.where(s.document_id.isna(),'REVIEW_MISSING_DOCUMENT',np.where(s.release_date_sensitivity_source.eq('verified_document_header'),'REVIEW_ORIGINAL_TEXT','REVIEW_RELEASE_DATE_AND_TEXT'))
    return s

def main():
    import yaml
    from src.h2.experiments import evaluate_h2_conditional_full_model as prev, evaluate_h2_independent_robustness as rb, evaluate_h2_independent_regime_model as ind, evaluate_h2_regime_specialized_model as spec, evaluate_h2_training_period_and_features as old
    from src.h2.preprocessing.build_bok_rate_history import make
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/source_pit_v1'));p.add_argument('--exante',type=Path,default=Path('outputs/model_results/h2/exante_universe_v2'));p.add_argument('--dart-events',type=Path,default=Path('results/validation/dart_versions/dart_event_version_audit.csv'));a=p.parse_args();root=a.root.resolve();out=root/a.out
    if out.exists():raise FileExistsError(out)
    cfg=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8'));diag=root/'outputs/model_results/h2/overfit_diagnostic_v1';ex=root/a.exante;leak=root/'outputs/model_results/h2/leakage_audit_v1/leakage_audit.json'
    la=json.loads(leak.read_text(encoding='utf-8-sig'));rr=json.loads((diag/'run_audit.json').read_text(encoding='utf-8-sig'));er=json.loads((ex/'run_audit.json').read_text(encoding='utf-8-sig'))
    if any(x.get('test_used') is not False for x in [la,rr,er]) or er.get('alpha')!=.05 or er.get('rule')!='hike_1m' or er['settings']['trees']!=500:raise ValueError('Wrong development run')
    if er['leakage_audit_sha256']!=sha(leak) or er['reference_diagnostic_sha256']['run_audit.json']!=sha(diag/'run_audit.json'):raise ValueError('Reference audit changed')
    for mod in [rb,prev,spec,ind,old]:spec.hash_match(rr['code_sha256'],root,'src/h2/experiments/'+Path(mod.__file__).name)
    hashes={}
    def audited(rel):
        spec.hash_match(la['inputs_sha256'],root,str(rel).replace('\\','/'));path=root/rel;hashes[str(rel)]=sha(path);return read(path)
    arc=Path('data/processed/archive/h2_8features.parquet');h=norm(audited(arc));d=norm(audited(Path('data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv')),'feature_month');c=norm(audited(Path('data/processed/final/debt_hybrid/h2_debt_change_2001_2021.csv')),'feature_month')
    for key in ['panel','eligibility']:spec.hash_match(la['inputs_sha256'],root,cfg['PATHS'][key]);hashes[cfg['PATHS'][key]]=sha(root/cfg['PATHS'][key])
    r=rate_compare(audited(Path('data/raw/h2/bok_rate_events_through_2021.csv')))
    args=SimpleNamespace(period_policy='full',sample_policy='fill',min_stocks=20,smoke=False)
    base,_,_,_,folds,_=prev.prepare(root,cfg,args);_,rates=make();g=ind.strict_gate(base,rates);z=base.merge(g[['target_month',rb.G]],on='target_month',how='left',validate='many_to_one');z[rb.G]=z[rb.G].eq(True)
    z,archive=rb.archive_frame(root,cfg,z,arc);plan=rb.make_plan(z,folds,'archive_hike8',5,20);plan=plan[plan.run_specialist]
    ref=audited(Path('outputs/model_results/h2/overfit_diagnostic_v1/fit_audit.csv')).query("features==8 and profile=='original'")
    f=read(ex/'fit_audit.csv');hashes[str(a.exante/'fit_audit.csv')]=sha(ex/'fit_audit.csv');hashes[str(a.exante/'run_audit.json')]=sha(ex/'run_audit.json');active=set(er['active_target_months']);parts=[];fit_rows=[]
    for v in plan.itertuples(index=False):
        tr=z[z.fit_member & z[rb.G] & z.target_month.between(v.train_start,v.train_end)]
        q=ref[ref.valid_year.eq(v.valid_year)];q2=f[f.valid_year.eq(v.valid_year)]
        ml=';'.join(sorted(tr.target_month.unique()))
        if len(q)!=len(er['models']) or len(q2)!=len(er['models']) or not q.train_rows.eq(len(tr)).all() or not q.train_month_list.eq(ml).all() or not q2.h2_train_rows.eq(len(tr)).all() or not q2.h2_train_months.eq(tr.target_month.nunique()).all():raise ValueError('Original fit reconstruction mismatch')
        parts.append(tr[KEY].assign(valid_year=v.valid_year));fit_rows.append({'valid_year':v.valid_year,'rows':len(tr),'target_months':ml})
    if set(plan.valid_year)!=set(f.valid_year):raise ValueError('Fit years differ')
    fit=pd.concat(parts,ignore_index=True);train=norm(fit[KEY].drop_duplicates()).assign(train=True)
    ep=read(ex/'exante_active_predictions.csv');hashes[str(a.exante/'exante_active_predictions.csv')]=sha(ex/'exante_active_predictions.csv');require(ep,KEY+['model','target_month','final_score'])
    if set(ep.model)!=set(er['models']) or set(ep.target_month)!=active or ep.duplicated(['model']+KEY).any() or not np.isfinite(ep.final_score).all() or not ((pd.PeriodIndex(ep.base_month,freq='M')+1).astype(str)==ep.target_month.astype(str)).all():raise ValueError('Invalid exante predictions')
    predict=norm(ep[KEY].drop_duplicates()).assign(predict=True);expected=h[h.base_month.isin(previous(pd.Series(sorted(active))))][KEY]
    for model,v in ep.groupby('model'):
        k=norm(v[KEY]);m=expected.merge(k,on=KEY,how='outer',indicator=True)
        if not m._merge.eq('both').all():raise ValueError('Prediction universe differs '+model)
    scope=train.merge(predict,on=KEY,how='outer',validate='one_to_one');scope[['train','predict']]=scope[['train','predict']].fillna(False).astype(bool)
    q,t,src,reports=trace_debt(h,d,c,scope)
    policy=policy_scope(audited(Path(cfg['PATHS']['h2_monthly_csv'])),audited(Path('data/processed/final/policy/release_dates.csv')),scope.base_month)
    # Compare original candidate audit metadata, not independently extracted report values.
    dart=reports[reports.source_type.eq(DART)].copy();dart['dart_url']='https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+dart.source_rcept_no
    ev=root/a.dart_events
    if ev.is_file():
        e=read(ev);require(e,['stock_code','rcept_no','debt_ratio_pct','version_audit_status','initial_candidate','next_filing_date']);e.stock_code=e.stock_code.str.zfill(6)
        e=e.rename(columns={'stock_code':'stock_id','rcept_no':'source_rcept_no'});e=e[['stock_id','source_rcept_no','debt_ratio_pct','version_audit_status','initial_candidate','next_filing_date']]
        dart=dart.merge(e,on=['stock_id','source_rcept_no'],how='left',validate='many_to_one');ok=same(dart.debt_final_raw,dart.debt_ratio_pct)&flag(dart.initial_candidate)&dart.version_audit_status.eq('FIRST_FILING_METADATA_MATCH_CANDIDATE')
        dart['candidate_metadata_status']=np.where(ok,'MATCH_REVIEW_ORIGINAL','REVIEW_METADATA_MISMATCH_OR_ABSENT');hashes[str(a.dart_events)]=sha(ev)
    else:dart['candidate_metadata_status']='REVIEW_NO_LOCAL_EVENT_AUDIT'
    dg=reports[reports.source_type.eq(DG)].copy();dg['evidence_needed']='Historical snapshot or contemporaneous filing ratio; 6M lag not vintage proof'
    summary=q.groupby(['base_month','scope'],as_index=False).agg(rows=('stock_id','size'),debt_present=('debt_available','sum'),debt_change_present=('change_available','sum'),dataguide=('source_type',lambda s:s.eq(DG).sum()),dart=('source_type',lambda s:s.eq(DART).sum()),unresolved=('has_unresolved_followup',lambda s:flag(s).sum()))
    out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('rate_official_comparison',r),('fit_scope',pd.DataFrame(fit_rows)),('scope_by_month',summary),('dart_report_queue',dart),('dataguide_report_queue',dg),('policy_document_queue',policy),('debt_change_ancestry',t),('source_row_dependencies',src)]:frame.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    info=dict(test_used=False,submission_certified=False,model_fitting=False,unique_train_rows=len(train),prediction_rows=len(predict),scope_months=sorted(scope.base_month.unique()),cross_section_rows=len(q),dart_report_candidates=len(dart),dataguide_report_candidates=len(dg),dart_flagged_reports=int(dart.unresolved.sum()),policy_documents=int(policy.document_id.nunique()),policy_inferred_documents=int(policy.loc[policy.status.eq('REVIEW_RELEASE_DATE_AND_TEXT'),'document_id'].nunique()),official_rate_events=len(r),official_rate_mismatch=int(r.status.eq('FAIL').sum()),source_status='REVIEW: original value and publication evidence not independently certified',inputs_sha256=hashes,script_sha256=sha(Path(__file__)))
    (out/'pit_scope_audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str),encoding='utf-8');print(json.dumps(info,ensure_ascii=False,indent=2,default=str));print('Saved:',out)
    if r.status.eq('FAIL').any():raise SystemExit(2)
if __name__=='__main__':main()
