"""H2-only training comparisons; frozen Base/H1/H3/H4; strict base-month hike gate.
Run as module at repository root. Development data only, no shared file writes.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.h2.experiments import evaluate_h2_independent_regime_model as ind
from src.h2.experiments import evaluate_h2_conditional_full_model as prev
from src.h2.experiments import evaluate_h2_regime_specialized_model as spec
from src.h2.experiments import evaluate_h2_training_period_and_features as old

G = ind.RULE
KEY = ind.KEY
FOUR = list(ind.FEATURES)
EIGHT = ['x_'+v for v in old.EIGHT]
VARIANTS = {
    'exp_hike4': ('expanding', 'hike', FOUR),
    'exp_all4': ('expanding', 'all', FOUR),
    'fixed_hike4': ('fixed', 'hike', FOUR),
    'archive_hike4': ('expanding', 'hike', FOUR),
    'archive_hike8': ('expanding', 'hike', EIGHT),
}
GROUPS = {
    'train_regime': ('2006-01', 'exp_hike4', 'exp_all4'),
    'train_period': ('2013-01', 'exp_hike4', 'fixed_hike4'),
    'feature_count': ('2006-01', 'archive_hike4', 'archive_hike8'),
}


def make_plan(z, folds, variant, min_months, min_stocks):
    mode, regime, features = VARIANTS[variant]
    rows = []
    for f in folds:
        yy = f['valid_year']; lo, hi = f['train_years']
        start, end = ('2009-01','2012-12') if mode == 'fixed' else (f'{lo}-01',f'{hi}-12')
        active = sorted(z.loc[z.year.eq(yy)&z[G], 'target_month'].unique())
        tr = z.loc[z.fit_member & z.target_month.between(start,end) & (z[G] if regime=='hike' else True)]
        counts=tr.groupby('target_month').size()
        # Fixed model is unavailable until its final target outcome is known.
        available=end < f'{yy}-01'
        ready=available and len(counts)>=min_months and bool(counts.ge(min_stocks).all())
        rows.append(dict(variant=variant,fold=f['fold'],valid_year=yy,train_start=start,train_end=end,
            train_regime=regime,features=','.join(features),train_months=len(counts),train_rows=len(tr),
            min_month_stocks=int(counts.min()) if len(counts) else 0,train_month_list=';'.join(sorted(counts.index)),
            eval_month_list=';'.join(active),active_eval_months=len(active),ready=ready,run_specialist=ready and bool(active)))
    return pd.DataFrame(rows)


def fit_variant(z, prior, plan, variant, kind, run, args):
    mode, regime, features=VARIANTS[variant]
    p=prior.copy().reset_index(drop=True)
    p['model']=kind; p[ind.BASE]=p.without_h2; p[ind.RAW]=p.without_h2
    p['gate_on']=p.target_month.isin(set(z.loc[z[G],'target_month']))
    p['specialist_used']=False; p['h2_score']=np.nan
    records=[]; groups=[]
    for r in plan.itertuples(index=False):
        rec=r._asdict(); rec.update(model=kind,trees=0,status='no_active_month' if not r.active_eval_months else 'fallback_not_ready')
        records.append(rec)
        if not r.run_specialist: continue
        tr=z.loc[z.fit_member & z.target_month.between(r.train_start,r.train_end) & (z[G] if regime=='hike' else True)].sort_values(['target_month','stock_id'])
        va=z.loc[z.year.eq(r.valid_year)&z[G]].sort_values(['target_month','stock_id'])
        if len(tr)!=r.train_rows or tr.target_month.nunique()!=r.train_months or not len(va) or tr.target_month.max()>=va.target_month.min():
            raise ValueError('Training sample/time mismatch')
        if not va[G].all() or (regime=='hike' and not tr[G].all()): raise ValueError('Hike gate violation')
        groups.append((r,rec,tr,va))
    # A fixed expert has exactly one fit, never refitted on later data.
    if mode=='fixed' and groups:
        first=groups[0][2]
        if any(not tr.index.equals(first.index) for _,_,tr,_ in groups): raise ValueError('Fixed training drift')
        va=pd.concat([v for _,_,_,v in groups],ignore_index=True)
        score,_,n=old.fit_predict(first,va,features,kind,run,args.threads,args.xgb_device)
        if n!=500: raise ValueError('Unexpected trees')
        offset=0
        forecasts=[]
        for r,rec,tr,v in groups:
            forecasts.append((r,rec,tr,v,score[offset:offset+len(v)],n)); offset+=len(v)
    else:
        forecasts=[]
        for r,rec,tr,va in groups:
            score,_,n=old.fit_predict(tr,va,features,kind,run,args.threads,args.xgb_device)
            forecasts.append((r,rec,tr,va,score,n))
    for r,rec,tr,va,score,n in forecasts:
        if n!=500 or not np.isfinite(score).all(): raise ValueError('Invalid expert forecasts')
        new=va[KEY].astype(str).copy(); new['h2_score']=score
        select=p.fold.eq(r.fold)&p.gate_on
        q=p.loc[select,KEY].merge(new,on=KEY,how='left',validate='one_to_one',sort=False)
        if len(q)!=len(new) or q.h2_score.isna().any(): raise ValueError('Forecast key mismatch')
        p.loc[select,'h2_score']=q.h2_score.to_numpy(float)
        p.loc[select,'specialist_used']=True
        rec.update(status='fit',trees=n,actual_train_start=tr.target_month.min(),actual_train_end=tr.target_month.max(),eval_rows=len(va),fit_id=f'{variant}_fixed' if mode=='fixed' else f'{variant}_{r.valid_year}')
        print(f'{kind} | {variant} | {r.valid_year} | H2 ONLY {len(tr):,} rows / {r.train_months} months | {r.active_eval_months} eval months',flush=True)
    if p.specialist_used.ne(p.h2_score.notna()).any() or (p.specialist_used & ~p.gate_on).any(): raise ValueError('Invalid routing')
    return p,pd.DataFrame(records)


def archive_frame(root,cfg,d,path):
    """Use original archive for BOTH four/eight arms, not silently mix vintages."""
    from src.evaluation import model_test as mt
    archive=old.resolve_archive(root,path)
    rel=archive.relative_to(root).as_posix()
    specs={n:{'path':rel,'transform':'input' if n=='debt_change' else 'none'} for n in old.EIGHT}
    prep=mt.prepare_inputs(root,cfg,extra_features=specs,input_transform='rank',fill_missing=True)['data']
    cols=KEY+EIGHT+[n+'_available' for n in old.EIGHT]
    if prep.target_month.ge('2022-01').any(): raise PermissionError('test data')
    p=prep[cols].copy(); p[KEY]=p[KEY].astype(str)
    b=d.drop(columns=[c for c in cols if c not in KEY and c in d]).copy(); b[KEY]=b[KEY].astype(str)
    b=b.merge(p,on=KEY,how='left',validate='one_to_one',indicator=True,sort=False)
    if not b._merge.eq('both').all() or np.isinf(b[EIGHT].to_numpy(float)).any(): raise ValueError('Archive feature alignment')
    observed=b[[n+'_available' for n in old.EIGHT]].fillna(False).all(axis=1)
    b['common']=b.common & observed
    return b.drop(columns='_merge'),archive


def restrict_pair(p, shared, start, common):
    p=p.loc[p.target_month.ge(start)].copy()
    p['common']=common.reindex(pd.MultiIndex.from_frame(p[KEY])).to_numpy(bool)
    on=p.target_month.isin(shared)
    p['specialist_used']=on
    p.loc[~on,'h2_score']=np.nan
    p[ind.RAW]=p[ind.BASE]
    p.loc[on,ind.RAW]=p.loc[on,'with_h2']
    return p


def evaluate_pair(left,right,group,start,args):
    # Same held-out dates, stock rows and expert-ready dates for both alternatives.
    left,right=left.copy(),right.copy()
    if not left[KEY].equals(right[KEY]) or not np.array_equal(left.without_h2,right.without_h2): raise ValueError('Baseline or sample differs')
    shared=set(left.loc[left.specialist_used & left.target_month.ge(start),'target_month']) & set(right.loc[right.specialist_used & right.target_month.ge(start),'target_month'])
    if not shared: raise ValueError(f'{group}: no common ready hike months')
    common=pd.Series((left.common & right.common).to_numpy(),index=pd.MultiIndex.from_frame(left[KEY]))
    tables={}; monthly=[]
    def add(name,df): tables.setdefault(name,[]).append(df)
    for variant,p in zip(GROUPS[group][1:],(left,right)):
        p=restrict_pair(p,shared,start,common)
        p,a,c=ind.blend(p,args.min_calibration_months)
        m,s,pc,e=ind.evaluate(p)
        for name,df in [('predictions',p),('alpha_selection',a),('calibration_monthly_ic',c),('monthly_rankic',m),('model_summary',s),('paired_comparisons',pc),('h2_only_monthly',e)]:
            add(name,df.assign(comparison=group,variant=variant))
        monthly.append(m)
    # Direct paired difference between alternatives at identical fixed weights / adaptive policies.
    import statsmodels.api as sm
    a,b=monthly
    rows=[]
    for scope in a.scope.unique():
        for period in ('all','active_ready'):
            aa=a.loc[a.scope.eq(scope)]; bb=b.loc[b.scope.eq(scope)]
            if period=='active_ready': aa=aa[aa.specialist_used]; bb=bb[bb.specialist_used]
            for arm in [ind.arm(w) for w in ind.WEIGHTS]+[ind.ADAPTIVE]:
                l=aa[aa.arm.eq(arm)].set_index('target_month').sort_index(); r=bb[bb.arm.eq(arm)].set_index('target_month').sort_index()
                if not l.index.equals(r.index) or not l.n.equals(r.n) or not np.allclose(l.universe_return,r.universe_return): raise ValueError('Unpaired comparison')
                delta=r.ic-l.ic; ex=r.top20_minus_universe-l.top20_minus_universe
                pv=np.nan
                if period=='all' and len(delta)>=24 and delta.std()>0 and set(delta.index)==spec.month_range(delta.index.min(),delta.index.max()):
                    pv=float(sm.OLS(delta.to_numpy(),np.ones((len(delta),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6}).pvalues[0])
                rows.append(dict(comparison=group,model=l.model.iloc[0],scope=scope,period=period,arm=arm,left=GROUPS[group][1],right=GROUPS[group][2],months=len(l),expert_months=int(l.specialist_used.sum()),left_ic=l.ic.mean(),right_ic=r.ic.mean(),right_minus_left=delta.mean(),delta_top20_excess=ex.mean(),hac6_p_two_sided=pv))
    add('direct_comparison',pd.DataFrame(rows))
    add('shared_ready_months',pd.DataFrame({'comparison':group,'model':left.model.iloc[0],'target_month':sorted(shared)}))
    return {k:pd.concat(v,ignore_index=True) for k,v in tables.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.')); p.add_argument('--models',default=','.join(prev.KINDS))
    p.add_argument('--experiments',default='train_regime,train_period'); p.add_argument('--h2-archive',type=Path,default=old.ARCHIVE_PATH)
    p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/independent_robustness_v1'))
    for k in prev.KINDS: p.add_argument('--prior-'+k.replace('_','-'),type=Path,default=Path(f'outputs/model_results/h2/conditional_full_{k}_v1'))
    p.add_argument('--sample-policy',choices=['fill'],default='fill'); p.add_argument('--period-policy',choices=['full'],default='full')
    p.add_argument('--min-stocks',type=int,default=20);p.add_argument('--min-train-months',type=int,default=5);p.add_argument('--min-calibration-months',type=int,default=3)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu');p.add_argument('--check-only',action='store_true')
    a=p.parse_args(); a.smoke=False
    root=a.root.resolve(); out=root/a.out
    groups=a.experiments.split(','); kinds=a.models.split(',')
    if not set(groups)<=set(GROUPS) or len(set(groups))!=len(groups) or not set(kinds)<=set(prev.KINDS) or len(set(kinds))!=len(kinds) or min(a.min_train_months,a.min_calibration_months)<1 or a.min_stocks<3: raise ValueError('Invalid settings')
    if out.exists() and not a.check_only: raise FileExistsError(out)
    cfg=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8')); run=cfg['BASE_FEATURES']['model_input']['main_run']
    if (run['trees'],run['depth'],run['learning_rate'])!=(500,3,.01): raise ValueError('Unexpected settings')
    d,base,full,labels,folds,coverage=prev.prepare(root,cfg,a)
    if len(base)!=30 or len(full)!=34: raise ValueError('Unexpected baseline')
    from src.h2.preprocessing.build_bok_rate_history import make
    _,rates=make(); g=ind.strict_gate(d,rates)
    z=d.merge(g[['target_month',G]],on='target_month',how='left',validate='many_to_one'); z[G]=z[G].eq(True)
    frames={v:z for v in VARIANTS}; archive=None
    if 'feature_count' in groups:
        az,archive=archive_frame(root,cfg,z,a.h2_archive)
        frames['archive_hike4']=frames['archive_hike8']=az
    variants=list(dict.fromkeys(v for group in groups for v in GROUPS[group][1:]))
    plans={v:make_plan(frames[v],folds,v,a.min_train_months,a.min_stocks) for v in variants}
    for v,plan in plans.items():
        print(v, '| ready months:',int(plan.loc[plan.run_specialist,'active_eval_months'].sum()),flush=True)
        print(plan[['valid_year','train_months','train_rows','active_eval_months','run_specialist']].to_string(index=False))
    for group in groups:
        start,l,r=GROUPS[group]
        ready=lambda v: set(m for s in plans[v].loc[plans[v].run_specialist,'eval_month_list'] for m in s.split(';') if m>=start)
        shared=sorted(ready(l)&ready(r)); print(group,'common ready:',len(shared),shared)
        if not shared: raise ValueError('No common ready months')
    tables={}; sources={}
    def add(name,df): tables.setdefault(name,[]).append(df)
    for k in kinds:
        prior,_,sources[k]=spec.load_prior(root,getattr(a,'prior_'+k),k,cfg,d,base,full,labels,folds,a)
        print('Verified frozen 30-feature baseline:',k,len(prior),flush=True)
        if a.check_only: continue
        preds={}
        for v in variants:
            preds[v],fit=fit_variant(frames[v],prior,plans[v],v,k,run,a)
            if v.startswith('archive_'):
                flags=frames[v][KEY+['common']].copy(); flags[KEY]=flags[KEY].astype(str)
                preds[v]['common']=preds[v][KEY].merge(flags,on=KEY,validate='one_to_one',how='left').common.to_numpy(bool)
            add('fit_audit',fit)
        for group in groups:
            start,l,r=GROUPS[group]
            for name,df in evaluate_pair(preds[l],preds[r],group,start,a).items(): add(name,df)
    if a.check_only: return
    out.mkdir(parents=True,exist_ok=False)
    add('fit_plan',pd.concat(plans.values(),ignore_index=True));add('gate_months',g);add('monthly_coverage',coverage)
    for name,dfs in tables.items(): pd.concat(dfs,ignore_index=True).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    dc=pd.concat(tables['direct_comparison'],ignore_index=True); decision=dc[dc.scope.eq('all_stocks')].copy(); decision.to_csv(out/'h2_decision.csv',index=False,encoding='utf-8-sig')
    audit=dict(groups={k:GROUPS[k] for k in groups},variants={k:VARIANTS[k] for k in variants},baseline_features=base,models=kinds,main_run=run,prior_runs=sources,rule=G,test_used=False,min_train_months=a.min_train_months,min_calibration_months=a.min_calibration_months,min_stocks=a.min_stocks,threads=a.threads,xgb_device=a.xgb_device,weights=ind.WEIGHTS,archive={'path':str(archive),'sha256':prev.sha(archive)} if archive else None,code_sha256={str(Path(m.__file__).resolve()):prev.sha(m.__file__) for m in [prev,spec,ind,old]},script_sha256=prev.sha(__file__),git_commit=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),notes='Frozen baseline; all arms apply only on shared ready base-month hikes; adaptive calibrated separately within each common period using earlier OOS months only; fixed target months 2009-2012, first evaluation 2013; archive four vs eight both historical vintage; development exploration; sparse dates, PIT caveats; no causal test.')
    (out/'run_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(decision[['comparison','model','period','arm','months','expert_months','left_ic','right_ic','right_minus_left','hac6_p_two_sided']].to_string(index=False));print('Saved:',out)

if __name__=='__main__': main()
