"""H2 overfitting diagnostic: matched 4/8 inputs, fixed 5% blend, development only.
Run from repository root. Does not choose a model, refit the baseline or use final test labels.
"""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import yaml
from src.h2.experiments import evaluate_h2_independent_robustness as rb
from src.h2.experiments import evaluate_h2_independent_regime_model as ind
from src.h2.experiments import evaluate_h2_conditional_full_model as prev
from src.h2.experiments import evaluate_h2_regime_specialized_model as spec
from src.h2.experiments import evaluate_h2_training_period_and_features as old

KEY, G = rb.KEY, rb.G
FEATURES = {4: rb.FOUR, 8: rb.EIGHT}
# Fixed diagnostic grid, not a hyperparameter search. Fit once per profile and use prefixes.
CHECKPOINTS = (50, 100, 200, 300, 500)
PROFILES = {'original': (3, 0.), 'shallow': (2, 0.), 'l2': (3, 1.)}
ALPHA = .05
SCOPES = ('all_stocks', 'complete_case')


def fit_curve(tr, va, features, kind, run, profile, args):
    """Exact old label/parameter definitions, with only depth or L2 changed."""
    from src.models import boosting
    depth, l2 = PROFILES[profile]
    n = CHECKPOINTS[-1]
    X = tr[features].fillna(0).to_numpy(float)
    Xv = va[features].fillna(0).to_numpy(float)
    y = old.rank_target(tr, kind)
    if not np.isfinite(X).all() or not np.isfinite(Xv).all():
        raise ValueError('Nonfinite features')
    loss = 'pairwise' if kind == 'xgb_pairwise' else 'mse'
    lr = float(run['learning_rate'])
    if kind == 'lgb_rankreg':
        import lightgbm as lgb
        params = boosting.lgb_params(kind, loss, depth, lr, device='cpu', n_threads=args.threads)
        if profile == 'l2': params['lambda_l2'] = l2
        model = lgb.train(params, lgb.Dataset(X, label=y, feature_name=features), num_boost_round=n)
        actual = model.num_trees()
        predict = lambda k: (model.predict(X, num_iteration=k), model.predict(Xv, num_iteration=k))
    else:
        import xgboost as xgb
        overrides = {'lambda': l2 if profile == 'l2' else float(run['xgb_l2'])}
        if kind == 'xgb_pairwise': overrides.update(run['pairwise_pairs'])
        params = boosting.xgb_params(kind, loss, depth, lr, device=args.xgb_device, n_threads=args.threads, overrides=overrides)
        qid = pd.factorize(tr.target_month, sort=True)[0] if kind == 'xgb_pairwise' else None
        dm = xgb.DMatrix(X, label=y, qid=qid, feature_names=features)
        dv = xgb.DMatrix(Xv, feature_names=features)
        model = xgb.train(params, dm, num_boost_round=n)
        actual = model.num_boosted_rounds()
        predict = lambda k: (model.predict(dm, iteration_range=(0,k)), model.predict(dv, iteration_range=(0,k)))
    if actual != n:
        raise RuntimeError(f'{kind}/{profile}: {actual} actual trees, expected {n}; no silent truncation')
    scores = {}
    for k in CHECKPOINTS:
        pt, pv = predict(k)
        if not np.isfinite(pt).all() or not np.isfinite(pv).all(): raise ValueError('Invalid predictions')
        scores[k] = (pt, pv)
    return scores, params


def check_frame(z, prior):
    if z.duplicated(KEY).any() or prior.duplicated(KEY).any(): raise ValueError('duplicate stock-month')
    if z.target_month.ge('2022-01').any() or prior.target_month.ge('2022-01').any(): raise PermissionError('development only')
    if not ((pd.PeriodIndex(z.base_month, freq='M')+1).astype(str)==z.target_month.astype(str)).all(): raise ValueError('target alignment')
    if not np.isfinite(prior[['without_h2','y']]).all().all(): raise ValueError('invalid baseline')


def ic_rows(frame, score, meta):
    f = frame.assign(score=np.asarray(score,float))
    rows=[]
    for scope in SCOPES:
        a=f if scope=='all_stocks' else f[f.common]
        for m,h in a.groupby('target_month',sort=True):
            if len(h)<3 or h.y.nunique()<2: continue
            rows.append(dict(**meta,scope=scope,target_month=m,n=len(h),ic=ind.rankic(h.score,h.y)))
    return rows


def run_model(z, prior, plan, kind, run, args):
    """Paired expanding folds; all profile/feature arms receive identical rows."""
    check_frame(z,prior)
    prior=prior.copy().reset_index(drop=True)
    prior[KEY]=prior[KEY].astype(str)
    prior=prior.drop(columns=['common']).merge(z[KEY+['common']].assign(**{k:z[k].astype(str) for k in KEY}),on=KEY,how='left',validate='one_to_one',sort=False)
    if prior.common.isna().any(): raise ValueError('common case alignment')
    prior['gate_on']=prior.target_month.isin(set(z.loc[z[G],'target_month']))
    ready=plan[plan.run_specialist]
    shared=set(m for s in ready.eval_month_list for m in s.split(';') if m)
    if not shared: raise ValueError('No common ready hikes')
    prior['active']=prior.target_month.isin(shared)
    if (prior.active & ~prior.gate_on).any(): raise ValueError('gate violation')
    train_rows=[]; audits=[]; expert_rows=[]; predictions=[]
    # Store only active forecasts; all nonactive forecasts are copied unchanged below.
    cache={(nf,profile,k):np.full(len(prior),np.nan) for nf in FEATURES for profile in PROFILES for k in CHECKPOINTS}
    for r in ready.itertuples(index=False):
        tr=z.loc[z.fit_member & z[G] & z.target_month.between(r.train_start,r.train_end)].sort_values(['target_month','stock_id'])
        va=z.loc[z.year.eq(r.valid_year)&z[G]].sort_values(['target_month','stock_id'])
        if len(tr)!=r.train_rows or tr.target_month.nunique()!=r.train_months or not len(va) or tr.target_month.max()>=va.target_month.min() or not tr[G].all() or not va[G].all(): raise ValueError('fit timing or sample differs')
        if not tr.target_month.max() <= r.train_end: raise ValueError('train end')
        mask=prior.fold.eq(r.fold)&prior.active
        mapping=prior.loc[mask,KEY].merge(va[KEY].astype(str).assign(pos=np.arange(len(va))),on=KEY,how='left',validate='one_to_one',sort=False)
        if len(mapping)!=len(va) or mapping.pos.isna().any(): raise ValueError('OOS alignment')
        if not np.allclose(prior.loc[mask,'y'],va.y.to_numpy()[mapping.pos.to_numpy(int)]): raise ValueError('targets differ')
        for nf,features in FEATURES.items():
            for profile,(depth,l2) in PROFILES.items():
                scores,params=fit_curve(tr,va,features,kind,run,profile,args)
                audits.append(dict(model=kind,features=nf,profile=profile,depth=depth,l2=l2,fold=r.fold,valid_year=r.valid_year,train_start=tr.target_month.min(),train_end=tr.target_month.max(),train_months=tr.target_month.nunique(),train_rows=len(tr),eval_months=va.target_month.nunique(),eval_rows=len(va),feature_names=','.join(features),train_month_list=';'.join(sorted(tr.target_month.unique())),eval_month_list=';'.join(sorted(va.target_month.unique())),params=json.dumps(params,sort_keys=True)))
                for k,(pt,pv) in scores.items():
                    meta=dict(model=kind,features=nf,profile=profile,depth=depth,l2=l2,trees=k,fold=r.fold,valid_year=r.valid_year)
                    train_rows.extend(ic_rows(tr,pt,meta))
                    expert_rows.extend(ic_rows(va,pv,meta))
                    cache[nf,profile,k][mask.to_numpy()]=pv[mapping.pos.to_numpy(int)]
        print(f'{kind} | {r.valid_year} | {len(tr):,} train rows / {r.train_months} hike months | {r.active_eval_months} OOS months | 4/8 x 3 profiles done',flush=True)
    # Equal weight per month, exact baseline preservation outside the active gate.
    monthly=[]
    baseline=prior.without_h2.to_numpy(float)
    for (nf,profile,k),expert in cache.items():
        active=prior.active.to_numpy(bool)
        if not np.array_equal(np.isfinite(expert),active): raise ValueError('expert availability mismatch')
        final=baseline.copy()
        for _,h in prior.loc[active].groupby('target_month',sort=True):
            idx=h.index.to_numpy(); b=h.without_h2.rank(method='average').to_numpy(); e=pd.Series(expert[idx]).rank(method='average').to_numpy()
            final[idx]=(95*b+5*e)/(100*(len(h)+1))
        if not np.array_equal(final[~active],baseline[~active]): raise ValueError('baseline modified outside active hike')
        meta=dict(model=kind,features=nf,profile=profile,depth=PROFILES[profile][0],l2=PROFILES[profile][1],trees=k)
        # Archive active stock-level predictions for verification; avoid writing millions of duplicate baseline rows.
        predictions.append(prior.loc[active,KEY+['fold','y','common','without_h2']].assign(**meta,h2_score=expert[active],final_score=final[active]))
        p=prior.assign(final_score=final)
        for scope in SCOPES:
            s=p if scope=='all_stocks' else p[p.common]
            for m,h in s.groupby('target_month',sort=True):
                if len(h)<3 or h.y.nunique()<2: continue
                b=ind.rankic(h.without_h2,h.y); v=ind.rankic(h.final_score,h.y)
                monthly.append(dict(**meta,scope=scope,target_month=m,fold=int(h.fold.iloc[0]),active=bool(h.active.iloc[0]),n=len(h),baseline_ic=b,combined_ic=v,delta_ic=v-b))
    # Explicitly audit all non-active stock scores (the assertion above is on the arrays, not just IC).
    off=prior.loc[~prior.active,KEY+['fold','without_h2','gate_on']].copy()
    off['model']=kind
    return dict(train_monthly=pd.DataFrame(train_rows),oos_expert_monthly=pd.DataFrame(expert_rows),combined_monthly=pd.DataFrame(monthly),fit_audit=pd.DataFrame(audits),active_predictions=pd.concat(predictions,ignore_index=True),inactive_baseline=off)


CONFIG=['model','features','profile','depth','l2','trees','scope']

def summarize(tables):
    import statsmodels.api as sm
    tr=tables['train_monthly']; va=tables['oos_expert_monthly']; c=tables['combined_monthly']
    a=tr.groupby(CONFIG+['fold'],as_index=False).agg(train_ic=('ic','mean'),train_months=('ic','size'))
    b=va.groupby(CONFIG+['fold'],as_index=False).agg(oos_ic=('ic','mean'),oos_months=('ic','size'))
    f=a.merge(b,on=CONFIG+['fold'],validate='one_to_one')
    f['gap']=f.train_ic-f.oos_ic
    summaries=[]
    for key,h in f.groupby(CONFIG,sort=False):
        w=h.oos_months
        meta=dict(zip(CONFIG,key)); match=np.logical_and.reduce([c[x].eq(v) for x,v in meta.items()]); q=c.loc[match].sort_values('target_month'); on=q[q.active]
        for period,s in [('all',q),('active_ready',on)]:
            delta=s.delta_ic; p=lo=hi=np.nan
            if period=='all' and len(s)>=24 and delta.std()>0 and set(s.target_month)==spec.month_range(s.target_month.min(),s.target_month.max()):
                fit=sm.OLS(delta.to_numpy(),np.ones((len(delta),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6});p=float(fit.pvalues[0]);lo,hi=map(float,fit.conf_int()[0])
            summaries.append(dict(**meta,period=period,months=len(s),active_months=len(on),folds=len(h),train_ic_oos_fold_weighted=np.average(h.train_ic,weights=w),oos_expert_ic=np.average(h.oos_ic,weights=w),generalization_gap=np.average(h.gap,weights=w),baseline_ic=s.baseline_ic.mean(),combined_ic=s.combined_ic.mean(),combined_delta_ic=delta.mean(),combined_win_active_share=on.delta_ic.gt(0).mean(),hac6_p_two_sided=p,hac6_ci_low=lo,hac6_ci_high=hi))
    s=pd.DataFrame(summaries)
    ix=[x for x in CONFIG if x!='features']+['period']
    compare=s[s.features.eq(4)].merge(s[s.features.eq(8)],on=ix,suffixes=('_4','_8'),validate='one_to_one')
    for name in ('train_ic_oos_fold_weighted','oos_expert_ic','generalization_gap','combined_ic','combined_delta_ic'):
        compare['delta8_minus4_'+name]=compare[name+'_8']-compare[name+'_4']
    if not compare.months_4.eq(compare.months_8).all() or not compare.active_months_4.eq(compare.active_months_8).all(): raise ValueError('unpaired summary')
    # Compare standalone H2 on the identical OOS stock-month rows.
    keys=[x for x in CONFIG if x!='features']+['fold','target_month']
    paired=va[va.features.eq(4)].merge(va[va.features.eq(8)],on=keys,suffixes=('_4','_8'),validate='one_to_one')
    if not paired.n_4.eq(paired.n_8).all(): raise ValueError('different paired expert rows')
    paired['delta_ic']=paired.ic_8-paired.ic_4
    tables.update(fold_gap=f,overfit_summary=s,feature_comparison=compare,expert_paired_monthly=paired)
    return tables


def verify_reference(tables, folder, archive, args):
    """Original 500-tree 4/8 results must reproduce if prior feature_count folder supplied."""
    a=json.loads((folder/'run_audit.json').read_text(encoding='utf-8-sig'))
    if a['groups']!={'feature_count':['2006-01','archive_hike4','archive_hike8']} or a['archive']['sha256']!=prev.sha(archive) or a['test_used'] is not False or a['min_train_months']!=args.min_train_months or a['min_stocks']!=args.min_stocks or a['threads']!=args.threads or a['xgb_device']!=args.xgb_device:
        raise ValueError('Reference settings/archive mismatch')
    old_m=pd.read_csv(folder/'h2_only_monthly.csv',dtype={'target_month':str})
    old_m=old_m[old_m.scope.isin(SCOPES)&old_m.model.isin(args.models.split(','))].copy(); old_m['features']=old_m.variant.map({'archive_hike4':4,'archive_hike8':8})
    v=tables['oos_expert_monthly'].query("profile == 'original' and trees == 500")
    keys=['model','features','scope','target_month']
    q=v.merge(old_m[keys+['ic','n']],on=keys,suffixes=('','_ref'),how='outer',indicator=True,validate='one_to_one')
    if not q._merge.eq('both').all() or not q.n.eq(q.n_ref).all() or not np.allclose(q.ic,q.ic_ref,atol=1e-8,rtol=0): raise ValueError('Original H2 OOS result does not reproduce')
    old_m=pd.read_csv(folder/'model_summary.csv')
    old_m=old_m[old_m.scope.isin(SCOPES)&old_m.model.isin(args.models.split(','))&old_m.period.isin(['all','active_ready'])&old_m.arm.eq('H2_fixed_05')].copy(); old_m['features']=old_m.variant.map({'archive_hike4':4,'archive_hike8':8})
    # summary column name is verified against prior code/results, not inferred from ICIR.
    keys=['model','features','scope','period']; s=tables['overfit_summary'].query("profile == 'original' and trees == 500")
    q2=s.merge(old_m[keys+['mean_rankic']],on=keys,how='outer',indicator=True,validate='one_to_one')
    if not q2._merge.eq('both').all() or not np.allclose(q2.combined_ic,q2.mean_rankic,atol=1e-8,rtol=0): raise ValueError('Original 5% combined IC does not reproduce')
    return {'folder':str(folder),'expert_month_rows':len(q),'combined_summary_rows':len(q2),'max_expert_abs_diff':float((q.ic-q.ic_ref).abs().max()),'max_combined_abs_diff':float((q2.combined_ic-q2.mean_rankic).abs().max()),'sha256':{k:prev.sha(folder/k) for k in ['run_audit.json','h2_only_monthly.csv','model_summary.csv']}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--models',default=','.join(prev.KINDS))
    p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/overfit_diagnostic_v1')); p.add_argument('--h2-archive',type=Path,default=old.ARCHIVE_PATH)
    p.add_argument('--reference-dir',type=Path,default=Path('outputs/model_results/h2/independent_features_v1'),help='Original result; verified if folder exists')
    for k in prev.KINDS: p.add_argument('--prior-'+k.replace('_','-'),type=Path,default=Path(f'outputs/model_results/h2/conditional_full_{k}_v1'))
    p.add_argument('--threads',type=int,default=4);p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu');p.add_argument('--check-only',action='store_true')
    a=p.parse_args();a.sample_policy='fill';a.period_policy='full';a.min_stocks=20;a.min_train_months=5;a.smoke=False
    kinds=a.models.split(',');root=a.root.resolve();out=root/a.out
    if not kinds or not set(kinds)<=set(prev.KINDS) or len(set(kinds))!=len(kinds) or a.threads<1: raise ValueError('Invalid settings')
    if out.exists() and not a.check_only: raise FileExistsError(out)
    cfg=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8'));run=cfg['BASE_FEATURES']['model_input']['main_run']
    if (run['trees'],run['depth'],run['learning_rate'],run['xgb_l2'])!=(500,3,.01,0): raise ValueError('Unexpected original configuration')
    d,base,full,labels,folds,coverage=prev.prepare(root,cfg,a)
    if len(base)!=30 or len(full)!=34: raise ValueError('Unexpected baseline features')
    from src.h2.preprocessing.build_bok_rate_history import make
    _,rates=make();g=ind.strict_gate(d,rates);z=d.merge(g[['target_month',G]],on='target_month',how='left',validate='many_to_one');z[G]=z[G].eq(True)
    z,archive=rb.archive_frame(root,cfg,z,a.h2_archive)
    plan=rb.make_plan(z,folds,'archive_hike8',a.min_train_months,a.min_stocks)
    print(plan[['valid_year','train_months','train_rows','active_eval_months','run_specialist']].to_string(index=False))
    print('Frozen baseline:',len(base),'| shared ready target months:',plan.loc[plan.run_specialist,'active_eval_months'].sum(),'| archive SHA256:',prev.sha(archive),flush=True)
    reference=root/a.reference_dir
    if not reference.is_dir(): print('WARNING: original feature_count result unavailable; reference reproduction will be skipped.',flush=True)
    runs={};sources={}
    for k in kinds:
        prior,_,sources[k]=spec.load_prior(root,getattr(a,'prior_'+k),k,cfg,d,base,full,labels,folds,a)
        print('Verified frozen baseline:',k,len(prior),flush=True)
        if not a.check_only:
            for name,t in run_model(z,prior,plan,k,run,a).items(): runs.setdefault(name,[]).append(t)
    if a.check_only: return
    tables=summarize({k:pd.concat(v,ignore_index=True) for k,v in runs.items()})
    reproduction=verify_reference(tables,reference,archive,a) if reference.is_dir() else None
    out.mkdir(parents=True,exist_ok=False)
    tables.update(fit_plan=plan,gate_months=g,monthly_coverage=coverage)
    for k,t in tables.items(): t.to_csv(out/f'{k}.csv',index=False,encoding='utf-8-sig')
    audit=dict(test_used=False,models=kinds,rule=G,features=FEATURES,baseline_features=base,baseline_runs=sources,alpha=ALPHA,profiles=PROFILES,checkpoints=CHECKPOINTS,main_run=run,archive={'path':str(archive),'sha256':prev.sha(archive)},reference_reproduction=reproduction,threads=a.threads,xgb_device=a.xgb_device,versions={k:version(k) for k in ['numpy','pandas','lightgbm','xgboost','statsmodels']},python=platform.python_version(),git_commit=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),code_sha256={str(Path(m.__file__).resolve()):prev.sha(m.__file__) for m in [rb,ind,prev,spec,old]},script_sha256=prev.sha(__file__),notes='Development diagnostic only; no checkpoint selection; repeated outer development is not a new holdout. Train gap weighted by number of outer months in each fold. Archive vintage PIT not proven; combined ranks use prior evaluation universe, not deployable ex-ante universe; nonactive score equality asserted exactly. 5% chosen in earlier development. HAC only on consecutive full months; no correction for multiple comparisons.')
    (out/'run_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    view=tables['overfit_summary'].query("scope == 'all_stocks' and period == 'active_ready' and trees in [200,500]")
    print(view[['model','features','profile','trees','train_ic_oos_fold_weighted','oos_expert_ic','generalization_gap','combined_delta_ic']].to_string(index=False));print('Saved:',out)

if __name__=='__main__': main()
