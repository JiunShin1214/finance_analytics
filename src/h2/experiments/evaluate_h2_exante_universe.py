"""Development-only repair of H2 active-month blending universe.
Keep the prior fit rows, H3 choice, eight inputs, 500 trees and 5% blend.
Refit identical historical models solely to recover scores for missing-label stocks.
No production prediction or final-test data access; do not use outputs for submission.
"""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.h2.experiments import evaluate_h2_independent_robustness as rb
from src.h2.experiments import evaluate_h2_conditional_full_model as prev
from src.h2.experiments import evaluate_h2_regime_specialized_model as spec
from src.h2.experiments import evaluate_h2_independent_regime_model as ind
from src.h2.experiments import evaluate_h2_training_period_and_features as old

KEY, G, EIGHT = rb.KEY, rb.G, rb.EIGHT
ALPHA = .05
EXPECTED = list(pd.period_range('2006-01','2021-12',freq='M').astype(str))
TOL=1e-8


def keys(frame):
    x=frame[KEY].copy()
    if x.isna().any().any(): raise ValueError('Null keys')
    x[KEY]=x[KEY].astype(str)
    if x.duplicated(KEY).any(): raise ValueError('Duplicate stock-month')
    if not x.base_month.between('2000-12','2021-11').all() or not x.target_month.between('2001-01','2021-12').all():
        raise PermissionError('Development months only')
    if not ((pd.PeriodIndex(x.base_month,freq='M')+1).astype(str)==x.target_month).all(): raise ValueError('Target alignment')
    return x


def same_keys(a,b,context):
    q=keys(a).merge(keys(b),on=KEY,how='outer',indicator=True,validate='one_to_one')
    if not q._merge.eq('both').all():
        raise ValueError(f'{context}: different universe: {q._merge.value_counts().to_dict()}')


def close(a,b,context,atol=TOL):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=atol,equal_nan=True):
        raise ValueError(f'{context}: reproduction mismatch; do not interpret result')
    finite=np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.


def raw_score_check(q, column, kind):
    """Compare XGB native float32 exactly, rather than CSV's decimal float64.

    Original conditional_full predictions.csv stores XGB float32 values with
    pandas' shortest decimal representation. Re-reading that CSV as float64
    can exceed a 1e-8 absolute tolerance even when the native float32 bits
    agree. The archived baseline and H2 reference are still hash-checked.
    Never accept a different native float32 value, even by one ULP. For H2
    also retain the old absolute check (archived expert was promoted to f64).
    """
    context=f'{kind} raw {column}'
    a=q[column+'_old'].to_numpy(dtype=np.float64)
    b=q[column+'_new'].to_numpy(dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'{context}: nonfinite scores')
    maxdiff=float(np.max(np.abs(a-b))) if len(a) else 0.
    if kind.startswith('xgb'):
        a32=a.astype(np.float32); b32=b.astype(np.float32)
        if not np.isfinite(a32).all() or not np.isfinite(b32).all() or not np.array_equal(b,b32.astype(np.float64)):
            raise ValueError(f'{context}: new scores are not native float32; do not interpret result')
        neq=a32.view(np.uint32)!=b32.view(np.uint32)
        n=int(neq.sum())
        if n:
            sample=q.loc[neq,KEY+[column+'_old',column+'_new']].head(5)
            print(f'{context}: {n}/{len(a)} native float32 mismatches; max float64 abs diff={maxdiff:.17g}',flush=True)
            print(sample.to_string(index=False,float_format=lambda x:f'{x:.17g}'),flush=True)
            raise ValueError(f'{context}: genuine native float32 reproduction mismatch; do not interpret result')
        if column!='without_h2':
            close(a,b,context)
        print(f'{context}: native float32 exact; CSV float64 max abs diff={maxdiff:.17g}',flush=True)
    else:
        close(a,b,context)
        print(f'{context}: float64 reproduction passed; max abs diff={maxdiff:.17g}',flush=True)
    return maxdiff


def score_only(frame,base_cols):
    """Explicit feature whitelist; no y, predict flag, target transforms or future eligibility."""
    k=keys(frame)
    cols=list(dict.fromkeys([*base_cols,*EIGHT]))
    k=k.join(frame[cols].reset_index(drop=True).set_axis(k.index))
    if np.isinf(k[cols].to_numpy(float)).any(): raise ValueError('Infinite prediction input')
    return k


def blend_scores(scores):
    """Rank within the supplied, already fixed ex-ante universe. Reject label-bearing inputs."""
    if set(scores.columns)!=set(KEY+['without_h2','h2_score']):
        raise ValueError('Blend accepts keys and raw scores only')
    keys(scores)
    if not np.isfinite(scores[['without_h2','h2_score']].to_numpy(float)).all(): raise ValueError('Nonfinite score')
    out=scores.copy()
    g=out.groupby('target_month',sort=False)
    n=g.stock_id.transform('size')
    b=g.without_h2.rank(method='average'); e=g.h2_score.rank(method='average')
    # Exactly the old formula, but the denominator/ranks are fixed before joining labels.
    out['final_score']=(95*b+5*e)/(100*(n+1))
    return out


def predict_fold(tr_base,tr_h2,va,base_cols,kind,run,args):
    if not len(va) or not len(tr_base) or not len(tr_h2): raise ValueError('Empty fold')
    va=score_only(va,base_cols)
    for tr in (tr_base,tr_h2):
        keys(tr)
        if tr.target_month.max()>=va.target_month.min() or not np.isfinite(tr.y).all(): raise ValueError('Fit chronology or missing label')
    if not tr_h2[G].all(): raise ValueError('H2 training gate')
    b,_,nb=old.fit_predict(tr_base,va[base_cols],base_cols,kind,run,args.threads,args.xgb_device)
    h,_,nh=old.fit_predict(tr_h2,va[EIGHT],EIGHT,kind,run,args.threads,args.xgb_device)
    if nb!=500 or nh!=500: raise ValueError('Fixed 500 trees required')
    return blend_scores(va[KEY].assign(without_h2=b,h2_score=h))


def prepare_exante(root,cfg,args,active,base,labels):
    """Reuse exact team's feature transforms, without the old final predict/y filter.
    Existing preparation internally attaches targets; only feature columns/eligible flag
    are extracted. This is not an independent audit of all upstream feature sources.
    """
    from src.evaluation import model_test as mt, base_model_runs as bmr
    panel=pd.read_parquet(root/cfg['PATHS']['panel'],columns=['stock_id','month','return','eligible'])
    if not panel.month.between('2000-01','2021-12').all(): raise PermissionError('Development panel only')
    elig=pd.read_parquet(root/cfg['PATHS']['eligibility'])
    if not elig.month.between('2000-01','2021-12').all() or elig.duplicated(['stock_id','month']).any(): raise ValueError('Invalid eligibility input')
    h3=mt.compute_h3_sigma_grid(panel,elig)
    specs=bmr.feature_specs(cfg,'boosting')
    p=mt.prepare_inputs(root,cfg,h3_grid=h3,extra_features=specs,input_transform='rank',fill_missing=True,
        exclude_train_target_months=cfg['BASE_FEATURES']['model_input']['exclude_train_target_months'])
    d=p['data']; keys(d)
    if set(mt.boosting_labels(p))!=set(labels): raise ValueError('H3 candidate mismatch')
    # Build expected eligible keys from the independent base-month eligibility file.
    e=elig.loc[elig.eligible.eq(True).fillna(False),['stock_id','month']].rename(columns={'month':'base_month'}).copy()
    e['target_month']=(pd.PeriodIndex(e.base_month,freq='M')+1).astype(str)
    e=e[e.target_month.isin(active)]
    cols=list(dict.fromkeys([c for lab in labels for c in spec.columns(base,lab)]))
    d=d.loc[d.eligible.eq(True).fillna(False)&d.target_month.isin(active),KEY+cols].copy()
    d[KEY]=d[KEY].astype(str)
    same_keys(d,e,'Unfiltered prepared input vs base-month eligibility')
    # rb.archive_frame only needs the common flag to merge archive features; flags
    # for evaluation are taken from the frozen prior, never used to select forecasts.
    d['common']=True
    d,archive=rb.archive_frame(root,cfg,d,args.h2_archive)
    d=d.drop(columns='common')
    same_keys(d,e,'Archive merge vs ex-ante eligibility')
    return d,archive


def evaluate(prior,reference,scores,kind):
    """Compare old/new on identical realized evaluation rows AFTER score freeze."""
    p=prior.copy();p[KEY]=p[KEY].astype(str)
    r=reference.copy();r[KEY]=r[KEY].astype(str)
    s=scores[KEY+['without_h2','h2_score','final_score']].copy();keys(s)
    active=set(s.target_month)
    same_keys(p[p.target_month.isin(active)],r,'Prior active/reference')
    if not p[p.target_month.isin(active)].eligible.all(): raise ValueError('Old active sample contains ineligible stocks')
    check=p.loc[p.target_month.isin(active),KEY+['y','common']].merge(r[KEY+['y','common']],on=KEY,suffixes=('_prior','_ref'),validate='one_to_one')
    close(check.y_prior,check.y_ref,'Reference labels')
    if not check.common_prior.eq(check.common_ref).all(): raise ValueError('Reference complete-case flags differ')
    q=r.merge(s,on=KEY,how='left',suffixes=('_old','_new'),validate='one_to_one',indicator=True)
    if not q._merge.eq('both').all(): raise ValueError('Corrected forecast missing an old evaluated stock')
    errors={c:raw_score_check(q,c,kind) for c in ['without_h2','h2_score']}
    # Check the old blend both from archived raw scores and from recreated model scores
    # restricted to the old universe. No silently changed model or tie ordering.
    for suffix in ['old','new']:
        check=blend_scores(q[KEY+['without_h2_'+suffix,'h2_score_'+suffix]].rename(columns={'without_h2_'+suffix:'without_h2','h2_score_'+suffix:'h2_score'}))
        close(check.final_score,q.final_score_old,f'{kind} legacy blend {suffix}')
    q=q.drop(columns='_merge')
    pp=p.merge(q[KEY+['final_score_old','final_score_new','without_h2_old','h2_score_old','h2_score_new']],on=KEY,how='left',validate='one_to_one')
    on=pp.target_month.isin(active)
    if not pp.final_score_new.notna().eq(on).all(): raise ValueError('Active coverage mismatch')
    close(pp.loc[on,'without_h2'],pp.loc[on,'without_h2_old'],f'{kind} baseline prior')
    pp['old_combined']=pp.final_score_old.where(on,pp.without_h2)
    pp['new_combined']=pp.final_score_new.where(on,pp.without_h2)
    if not np.array_equal(pp.loc[~on,'new_combined'],pp.loc[~on,'without_h2']): raise ValueError('Inactive score changed')
    rows=[];universe=[]
    for m,full in s.groupby('target_month',sort=True):
        ev=pp[pp.target_month.eq(m)]
        absent=full[KEY].merge(ev[KEY],on=KEY,how='left',indicator=True)
        universe.append(dict(model=kind,target_month=m,base_month=full.base_month.iloc[0],exante_eligible=len(full),scored=len(full),evaluated=len(ev),added=len(full)-len(ev),missing_exante_scores=0,old_score_changed=int((~np.isclose(ev.old_combined,ev.new_combined,rtol=0,atol=1e-12)).sum()),max_abs_combined_score_change=float((ev.old_combined-ev.new_combined).abs().max()),missing_from_old_keys=int(absent._merge.eq('left_only').sum())))
    for scope,d in [('all_stocks',pp),('complete_case',pp[pp.common])]:
        for m,h in d.groupby('target_month',sort=True):
            if len(h)<3 or h.y.nunique()<2: continue
            ic=lambda c:ind.rankic(h[c],h.y)
            b,o,n=[ic(c) for c in ['without_h2','old_combined','new_combined']]
            rows.append(dict(model=kind,scope=scope,target_month=m,active=m in active,n=len(h),baseline_ic=b,old_combined_ic=o,new_combined_ic=n,old_delta_ic=o-b,new_delta_ic=n-b,repair_delta_ic=n-o))
    return pd.DataFrame(rows),pd.DataFrame(universe),errors


def summarize(monthly):
    import statsmodels.api as sm
    rows=[]
    for (kind,scope),d in monthly.groupby(['model','scope']):
        for period,h in [('all',d),('active_ready',d[d.active])]:
            h=h.sort_values('target_month'); delta=h.new_delta_ic
            p=lo=hi=np.nan
            if list(h.target_month)==EXPECTED and delta.std()>0:
                fit=sm.OLS(delta.to_numpy(),np.ones((len(h),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6})
                p=float(fit.pvalues[0]);lo,hi=map(float,fit.conf_int()[0])
            rows.append(dict(model=kind,scope=scope,period=period,months=len(h),active_months=int(h.active.sum()),baseline_ic=h.baseline_ic.mean(),old_combined_ic=h.old_combined_ic.mean(),new_combined_ic=h.new_combined_ic.mean(),old_delta_ic=h.old_delta_ic.mean(),new_delta_ic=delta.mean(),repair_delta_ic=h.repair_delta_ic.mean(),new_active_win_months=int((h.active & delta.gt(0)).sum()),hac6_p_two_sided=p,hac6_ci_low=lo,hac6_ci_high=hi))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/exante_universe_v1'))
    p.add_argument('--h2-archive',type=Path,default=old.ARCHIVE_PATH)
    p.add_argument('--diagnostic',type=Path,default=Path('outputs/model_results/h2/overfit_diagnostic_v1'))
    p.add_argument('--leakage-audit',type=Path,default=Path('outputs/model_results/h2/leakage_audit_v1/leakage_audit.json'))
    p.add_argument('--models',default=','.join(prev.KINDS));p.add_argument('--threads',type=int,default=4);p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu');p.add_argument('--check-only',action='store_true')
    for k in prev.KINDS: p.add_argument('--prior-'+k.replace('_','-'),type=Path,default=Path(f'outputs/model_results/h2/conditional_full_{k}_v1'))
    a=p.parse_args();a.period_policy='full';a.sample_policy='fill';a.min_stocks=20;a.min_train_months=5;a.smoke=False
    root=a.root.resolve();out=root/a.out;diag=root/a.diagnostic;kinds=a.models.split(',')
    if not kinds or len(kinds)!=len(set(kinds)) or not set(kinds)<=set(prev.KINDS): raise ValueError('Invalid models')
    if out.exists() and not a.check_only: raise FileExistsError(out)
    cfg=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8'));run=cfg['BASE_FEATURES']['model_input']['main_run']
    if (run['trees'],run['depth'],run['learning_rate'],run['xgb_l2'])!=(500,3,.01,0): raise ValueError('Original settings required')
    ref=json.loads((diag/'run_audit.json').read_text(encoding='utf-8-sig'))
    leak=json.loads((root/a.leakage_audit).read_text(encoding='utf-8-sig'))
    if ref.get('test_used') is not False or leak.get('test_used') is not False or ref.get('alpha')!=ALPHA or ref.get('rule')!=G or ref.get('main_run')!=run or ref.get('threads')!=a.threads or ref.get('xgb_device')!=a.xgb_device or ref.get('features',{}).get('8')!=EIGHT or not set(kinds)<=set(ref['models']): raise ValueError('Prior diagnostic settings differ')
    for name in ['run_audit.json','fit_audit.csv','active_predictions.csv']:
        spec.hash_match(leak['inputs_sha256'],root,(a.diagnostic/ name).as_posix())
    spec.hash_match(leak['inputs_sha256'],root,a.h2_archive.as_posix())
    for mod in [rb,prev,spec,ind,old]:
        rel='src/h2/experiments/'+Path(mod.__file__).name
        spec.hash_match(ref['code_sha256'],root,rel)
    d,base,full,labels,folds,_=prev.prepare(root,cfg,a)
    if len(base)!=30 or len(full)!=34 or base!=ref['baseline_features']: raise ValueError('Baseline features changed')
    from src.h2.preprocessing.build_bok_rate_history import make
    _,rates=make();g=ind.strict_gate(d,rates)
    z=d.merge(g[['target_month',G]],on='target_month',how='left',validate='many_to_one');z[G]=z[G].eq(True)
    z,archive=rb.archive_frame(root,cfg,z,a.h2_archive)
    if prev.sha(archive)!=ref['archive']['sha256']: raise ValueError('Archive changed')
    plan=rb.make_plan(z,folds,'archive_hike8',5,20)
    # Reconstruct evaluation gates using calendar and known base-month rate only,
    # not whether any stock happens to have a realized target in that month.
    rate=rates.set_index('feature_month').rate_change_pp
    known={}
    for r in plan.itertuples(index=False):
        months=pd.period_range(f'{r.valid_year}-01',f'{r.valid_year}-12',freq='M')
        changes=rate.reindex((months-1).astype(str))
        if changes.isna().any(): raise ValueError('Missing base-month rate')
        known[r.valid_year]=';'.join(months[changes.to_numpy()>0].astype(str)) if r.ready else ''
    ready=plan.loc[plan.ready & plan.valid_year.map(known).ne('')].copy()
    if not ready.eval_month_list.eq(ready.valid_year.map(known)).all(): raise ValueError('Prior gate dates differ from calendar-derived dates')
    ready['eval_month_list']=ready.valid_year.map(known)
    active=sorted({m for s in ready.eval_month_list for m in s.split(';') if m})
    if not active: raise ValueError('No ready months')
    ex,arc=prepare_exante(root,cfg,a,active,base,labels)
    if arc!=archive: raise ValueError('Archive path mismatch')
    refpred=pd.read_csv(diag/'active_predictions.csv',dtype={k:str for k in KEY},encoding='utf-8-sig')
    refpred=refpred.query("features==8 and profile=='original' and trees==500").copy()
    reffit=pd.read_csv(diag/'fit_audit.csv',encoding='utf-8-sig').query("features==8 and profile=='original'")
    if not set(refpred.target_month)==set(active): raise ValueError('Active dates changed')
    # Feature transforms must reproduce on previously evaluated rows, for every H3 option.
    cols=list(dict.fromkeys([c for lab in labels for c in spec.columns(base,lab)]+EIGHT))
    old_active=z[z.target_month.isin(active)].copy();old_active[KEY]=old_active[KEY].astype(str)
    q=old_active[KEY+cols].merge(ex[KEY+cols],on=KEY,suffixes=('_old','_new'),how='left',indicator=True,validate='one_to_one')
    if not q._merge.eq('both').all(): raise ValueError('Old active row absent in ex-ante inputs')
    feature_diff=max(close(q[c+'_old'],q[c+'_new'],'Feature '+c,atol=0) for c in cols)
    print('Active target months:',active,'| ex-ante rows:',len(ex),'| old evaluated rows:',len(old_active),'| missing recovered:',len(ex)-len(old_active),flush=True)
    all_scores=[];all_monthly=[];all_universe=[];fits_out=[];errors={};sources={}
    for kind in kinds:
        prior,fits,sources[kind]=spec.load_prior(root,getattr(a,'prior_'+kind),kind,cfg,d,base,full,labels,folds,a)
        if sources[kind]['sha256']!=ref['baseline_runs'][kind]['sha256']: raise ValueError('Baseline reference differs')
        # Use exact common case flag from the 8-feature historical archive on old evaluation rows.
        prior=prior.drop(columns='common').merge(z[KEY+['common']],on=KEY,how='left',validate='one_to_one',sort=False)
        if prior.common.isna().any(): raise ValueError('Complete-case alignment')
        rpred=refpred[refpred.model.eq(kind)]
        same_keys(prior[prior.target_month.isin(active)],rpred,'Old active reference')
        print(kind,'reference verified',flush=True)
        if a.check_only: continue
        fdict={f['valid_year']:f for f in folds};pred=[]
        for r in ready.itertuples(index=False):
            f=fdict[r.valid_year];lo,hi=f['train_years'];lab=fits.loc[r.valid_year,'h3_label'];bc=spec.columns(base,lab)
            tr=d.loc[d.fit_member & d.year.between(lo,hi)]
            th=z.loc[z.fit_member & z[G] & z.target_month.between(r.train_start,r.train_end)].sort_values(['target_month','stock_id'])
            va=ex[ex.target_month.isin(r.eval_month_list.split(';'))].sort_values(['target_month','stock_id'])
            rf=reffit[(reffit.model.eq(kind))&reffit.valid_year.eq(r.valid_year)]
            if len(rf)!=1: raise ValueError('Missing prior H2 fit')
            rf=rf.iloc[0]
            if len(tr)!=int(fits.loc[r.valid_year,'fit_rows']) or len(th)!=r.train_rows or len(th)!=rf.train_rows or ';'.join(sorted(th.target_month.unique()))!=rf.train_month_list or ','.join(EIGHT)!=rf.feature_names or th.target_month.nunique()!=r.train_months or th.target_month.min()!=rf.train_start or th.target_month.max()!=rf.train_end or int(rf.depth)!=3 or float(rf.l2)!=0.: raise ValueError('Training rows/features changed')
            v=predict_fold(tr,th,va,bc,kind,run,a)
            pred.append(v.assign(model=kind,fold=r.fold))
            fits_out.append(dict(model=kind,fold=r.fold,valid_year=r.valid_year,h3_label=lab,baseline_train_rows=len(tr),baseline_train_start=tr.target_month.min(),baseline_train_end=tr.target_month.max(),h2_train_rows=len(th),h2_train_months=th.target_month.nunique(),h2_train_start=th.target_month.min(),h2_train_end=th.target_month.max(),eval_rows=len(va),eval_month_list=r.eval_month_list,trees=500,alpha=ALPHA))
            print(kind,r.valid_year,'ex-ante predicted:',len(v),flush=True)
        scores=pd.concat(pred,ignore_index=True);same_keys(scores,ex,'Corrected score coverage')
        # Final score table is frozen before evaluation joins any realized outcomes.
        all_scores.append(scores.copy())
        m,u,errors[kind]=evaluate(prior,rpred,scores,kind);all_monthly.append(m);all_universe.append(u)
    if a.check_only: print('CHECK-ONLY passed; model-score reproduction requires full run.');return
    monthly=pd.concat(all_monthly,ignore_index=True);summary=summarize(monthly)
    # Reference IC also must match the prior original-500 aggregate exactly.
    orig=pd.read_csv(diag/'overfit_summary.csv').query("features==8 and profile=='original' and trees==500")
    s=summary.merge(orig[['model','scope','period','baseline_ic','combined_ic','months']],on=['model','scope','period'],suffixes=('','_ref'),validate='one_to_one')
    if len(s)!=len(summary) or not s.months.eq(s.months_ref).all(): raise ValueError('Prior summary coverage mismatch')
    close(s.baseline_ic,s.baseline_ic_ref,'Prior baseline IC');close(s.old_combined_ic,s.combined_ic,'Prior combined IC')
    for kind in kinds:
        m=monthly[(monthly.model==kind)&(monthly.scope=='all_stocks')]
        if list(m.target_month)!=EXPECTED: raise ValueError('192-month coverage required')
    out.mkdir(parents=True,exist_ok=False)
    tables={'exante_active_predictions':pd.concat(all_scores,ignore_index=True),'universe_comparison':pd.concat(all_universe,ignore_index=True),'monthly_comparison':monthly,'repair_summary':summary,'fit_audit':pd.DataFrame(fits_out)}
    for name,t in tables.items():t.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    audit=dict(test_used=False,submission_certified=False,scope='active H2 blend only; inactive evaluation scores reused',models=kinds,settings=run,alpha=ALPHA,rule=G,active_target_months=active,exante_rows_per_model=len(ex),old_eval_active_rows_per_model=len(old_active),max_old_feature_abs_diff=feature_diff,raw_score_reproduction_max_abs_diff=errors,raw_score_reproduction_rule='XGB native float32 bit equality; H2 also absolute 1e-8; LGB absolute 1e-8; legacy blend and IC checks unchanged',nonactive_evaluation_score_equality=True,active_score_columns=list(tables['exante_active_predictions'].columns),archive_sha256=prev.sha(archive),baseline_runs=sources,reference_diagnostic_sha256={name:prev.sha(diag/name) for name in ['run_audit.json','fit_audit.csv','active_predictions.csv','overfit_summary.csv']},leakage_audit_sha256=prev.sha(root/a.leakage_audit),script_sha256=prev.sha(__file__),git_commit=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),python=platform.python_version(),versions={k:version(k) for k in ['numpy','pandas','lightgbm','xgboost','statsmodels']},notes='No parameter tuning; original fit rows/H3 choices reconstructed, not persisted boosters. Score generation uses keys/features only after shared input preparation. Corrects active blending universe only; inactive baseline reused for evaluation. Historical PIT, upstream eligibility provenance, fixed production model and 2026 inputs NOT certified.')
    (out/'run_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print('Saved:',out)

if __name__=='__main__':main()
