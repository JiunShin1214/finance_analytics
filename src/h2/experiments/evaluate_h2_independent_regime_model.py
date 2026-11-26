"""Independent H2 four-feature hike-only expert + frozen full-history baseline.
Development only; execute from repository root. No shared files are modified.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.h2.experiments import evaluate_h2_conditional_full_model as prev
from src.h2.experiments import evaluate_h2_regime_specialized_model as spec

RULE = 'hike_1m'
KEY = spec.KEY
FEATURES = ['x_' + n for n in prev.FOUR]
WEIGHTS = (0., .05, .1, .2, .3)
BASE = 'A_baseline'
ADAPTIVE = 'H2_adaptive'
RAW = 'B_full_raw_h2_routed'

def arm(w):
    return BASE if w == 0 else f'H2_fixed_{int(round(w*100)):02d}'

ARMS = [BASE, RAW, *[arm(w) for w in WEIGHTS[1:]], ADAPTIVE]


def rankic(score, y):
    a, b = pd.Series(np.asarray(score,float)).rank(), pd.Series(np.asarray(y,float)).rank()
    if b.nunique() < 2: raise ValueError('Constant target')
    return 0. if a.nunique() < 2 else float(a.corr(b))


def strict_gate(d, rates):
    g = spec.make_gates(rates,d)
    s = rates.set_index('feature_month').rate_change_pp.astype(float)
    g['base_rate_change_pp'] = g.base_month.map(s)
    if g.base_rate_change_pp.isna().any() or not g[RULE].eq(g.base_rate_change_pp.gt(0)).all():
        raise ValueError('hike_1m does not match actual base-month rate hikes')
    if not ((pd.PeriodIndex(g.base_month,freq='M')+1).astype(str)==g.target_month).all():
        raise ValueError('gate timing mismatch')
    return g


def fit_expert(z, prior, plan, kind, run, args):
    from src.h2.experiments import evaluate_h2_training_period_and_features as old
    p=prior.copy().reset_index(drop=True)
    p[BASE]=p.without_h2
    p[RAW]=p.without_h2
    p['gate_on']=p.target_month.isin(set(z.loc[z[RULE],'target_month']))
    p['specialist_used']=False
    p['h2_score']=np.nan
    audit=[]
    for r in plan.itertuples(index=False):
        rec=dict(model=kind,fold=r.fold,valid_year=r.valid_year,train_rows=r.train_rows,train_months=r.train_months,
                 train_month_list=r.train_month_list,eval_month_list=r.eval_month_list,features=','.join(FEATURES),trees=0,
                 status='no_active_month' if not r.active_eval_months else 'fallback_insufficient_train')
        if r.run_specialist:
            tr=z.loc[z.fit_member & z.target_month.between(r.train_start,r.train_end) & z[RULE]].sort_values(['target_month','stock_id'])
            va=z.loc[z.year.eq(r.valid_year)&z[RULE]].sort_values(['target_month','stock_id'])
            if len(tr)!=r.train_rows or tr.target_month.nunique()!=r.train_months or not len(va) or tr.target_month.max()>=va.target_month.min():
                raise ValueError('Expert sample or time mismatch')
            # The only training columns are the four H2 features. No Base/H1/H3/H4 fitting.
            pr,_,n=old.fit_predict(tr,va,FEATURES,kind,run,args.threads,args.xgb_device)
            if n!=500 or not np.isfinite(pr).all(): raise ValueError('Expert fit invalid')
            new=va[KEY].astype(str).copy();new['h2_score']=pr
            select=p.fold.eq(r.fold)&p.gate_on
            q=p.loc[select,KEY].merge(new,on=KEY,how='left',validate='one_to_one',sort=False)
            if len(q)!=len(new) or q.h2_score.isna().any(): raise ValueError('Forecast key mismatch')
            p.loc[select,'h2_score']=q.h2_score.to_numpy(float)
            p.loc[select,RAW]=p.loc[select,'with_h2']
            p.loc[select,'specialist_used']=True
            rec.update(status='fit',trees=n,actual_train_start=tr.target_month.min(),actual_train_end=tr.target_month.max(),eval_rows=len(va))
            print(f'{kind} | {r.valid_year} | H2 ONLY {len(tr):,} rows / {r.train_months} months | {r.active_eval_months} eval months',flush=True)
        audit.append(rec)
    if (p.specialist_used & ~p.gate_on).any() or p.h2_score.notna().ne(p.specialist_used).any(): raise ValueError('Expert outside gate')
    p['model']=kind
    return p,pd.DataFrame(audit)


def blend(p, min_calibration_months):
    """Tune at start of each outer year using strictly earlier out-of-sample months.
    Frozen prior forecasts, including expert forecasts, never in-sample for calibration.
    Fixed weights are diagnostic only; adaptive is the predeclared primary strategy.
    """
    p=p.copy().sort_values(['target_month','stock_id']).reset_index(drop=True)
    for a in ARMS:
        if a not in p: p[a]=p[BASE]
    p['alpha']=0.
    hist=[];audit=[]
    for year, frame in p.groupby(p.target_month.str[:4],sort=True):
        # Current-year outcomes cannot affect this year's alpha, even within the year.
        past=pd.DataFrame(hist,columns=['target_month','alpha','ic'])
        if not past.empty and not past.target_month.lt(year+'-01').all(): raise ValueError('Calibration leakage')
        n=past.target_month.nunique()
        means=past.groupby('alpha').ic.mean().to_dict() if n else {}
        if n>=min_calibration_months:
            if set(means)!=set(WEIGHTS): raise ValueError('Incomplete calibration candidates')
            best=max(WEIGHTS,key=lambda w:(round(means[w],12),-w))
        else: best=0.
        audit.append(dict(model=frame.model.iloc[0],valid_year=int(year),calibration_months=n,
                          calibration_month_list=';'.join(sorted(past.target_month.unique())),selected_alpha=best,
                          status='selected' if n>=min_calibration_months else 'fallback_insufficient_calibration',
                          **{f'past_ic_{int(w*100):02d}':means.get(w,np.nan) for w in WEIGHTS}))
        new=[]
        for m,h in frame.groupby('target_month',sort=True):
            if not h.specialist_used.eq(h.specialist_used.iloc[0]).all(): raise ValueError('Mixed monthly routing')
            if not h.specialist_used.iloc[0]: continue
            if not h.gate_on.all() or not np.isfinite(h.h2_score).all(): raise ValueError('Invalid expert')
            b,e=h[BASE].rank(method='average'),h.h2_score.rank(method='average')
            for w in WEIGHTS:
                k=int(round(w*100))
                # Integer-percent weights on half-integer ranks preserve exact mathematical ties.
                s=h[BASE] if w==0 else ((100-k)*b+k*e)/(100*(len(h)+1))
                if w>0: p.loc[h.index,arm(w)]=s.to_numpy()
                new.append((m,w,rankic(s,h.y)))
            p.loc[h.index,'alpha']=best
            p.loc[h.index,ADAPTIVE]=p.loc[h.index,arm(best)].to_numpy()
        hist.extend(new)
    off=~p.specialist_used
    for a in ARMS:
        if not np.array_equal(p.loc[off,a].to_numpy(),p.loc[off,BASE].to_numpy()): raise ValueError('Changed baseline outside ready hike')
    if not np.array_equal(p.loc[p.alpha.eq(0),ADAPTIVE].to_numpy(),p.loc[p.alpha.eq(0),BASE].to_numpy()): raise ValueError('Zero alpha must preserve baseline exactly')
    if not np.isfinite(p[ARMS]).all().all(): raise ValueError('Nonfinite final forecast')
    return p,pd.DataFrame(audit),pd.DataFrame(hist,columns=['target_month','alpha','ic']).assign(model=p.model.iloc[0])


def evaluate(p):
    from src.h2.experiments.evaluate_h2_training_period_and_features import weighted_tail
    from src.h2.experiments.evaluate_h2_rate_hike_robustness import summarize
    import statsmodels.api as sm
    rows=[];expert=[];summary=[];pairs=[]
    for scope,s in [('all_stocks',p),('all_eligible',p[p.eligible]),('complete_case',p[p.common])]:
        for m,h in s.groupby('target_month',sort=True):
            if len(h)<3:
                if scope=='complete_case': continue
                raise ValueError('Insufficient evaluation rows')
            y=h.y.to_numpy(float);uni=y.mean()
            meta=dict(model=h.model.iloc[0],scope=scope,target_month=m,n=len(h),gate_on=bool(h.gate_on.iloc[0]),specialist_used=bool(h.specialist_used.iloc[0]),alpha=float(h.alpha.iloc[0]))
            for a in ARMS+(['H2_only'] if meta['specialist_used'] else []):
                s=h.h2_score if a=='H2_only' else h[a]
                top=weighted_tail(s,y);bottom=weighted_tail(s,y,False)
                row=dict(**meta,arm=a,ic=rankic(s,y),top20_return=top,bottom20_return=bottom,universe_return=uni,top20_minus_universe=top-uni,long_short=top-bottom)
                (expert if a=='H2_only' else rows).append(row)
    monthly=pd.DataFrame(rows)
    for scope,g in monthly.groupby('scope'):
        for period,sub in [('all',g),('active_ready',g[g.specialist_used]),('gate_on',g[g.gate_on]),('adaptive_on',g[g.alpha.gt(0)]),('2016_2021',g[g.target_month.ge('2016-01')])]:
            if sub.empty: continue
            ref=sub[sub.arm.eq(BASE)].set_index('target_month').sort_index()
            for a in ARMS:
                h=sub[sub.arm.eq(a)].set_index('target_month').sort_index()
                if not h.index.equals(ref.index) or not h.n.equals(ref.n): raise ValueError('Unpaired samples')
                summary.append(dict(model=p.model.iloc[0],scope=scope,period=period,arm=a,**summarize(h)))
                if a==BASE: continue
                delta=h.ic-ref.ic;ex=h.top20_minus_universe-ref.top20_minus_universe
                pv=low=high=np.nan
                if period=='all' and len(delta)>=24 and delta.std()>0 and set(delta.index)==spec.month_range(delta.index.min(),delta.index.max()):
                    fit=sm.OLS(delta.to_numpy(),np.ones((len(delta),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6})
                    pv=float(fit.pvalues[0]);low,high=map(float,fit.conf_int()[0])
                pairs.append(dict(model=p.model.iloc[0],scope=scope,period=period,arm=a,reference=BASE,months=len(h),expert_months=int(h.specialist_used.sum()),adaptive_on_months=int(h.alpha.gt(0).sum()),reference_ic=ref.ic.mean(),model_ic=h.ic.mean(),delta_ic=delta.mean(),ic_win_share=delta.gt(0).mean(),delta_top20_excess=ex.mean(),delta_down_top20_excess=ex[ref.universe_return.lt(0)].mean(),hac6_p_two_sided=pv,hac6_ci_low=low,hac6_ci_high=high))
    return monthly,pd.DataFrame(summary),pd.DataFrame(pairs),pd.DataFrame(expert)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--models',default=','.join(prev.KINDS))
    p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/independent_regime_v1'))
    for k in prev.KINDS: p.add_argument('--prior-'+k.replace('_','-'),type=Path,default=Path(f'outputs/model_results/h2/conditional_full_{k}_v1'))
    p.add_argument('--sample-policy',choices=['fill','complete_case'],default='fill');p.add_argument('--period-policy',choices=['full','common'],default='full')
    p.add_argument('--min-stocks',type=int,default=20);p.add_argument('--min-train-months',type=int,default=5);p.add_argument('--min-calibration-months',type=int,default=3)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--check-only',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();root=a.root.resolve();cfg=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8'));run=cfg['BASE_FEATURES']['model_input']['main_run']
    kinds=[k.strip() for k in a.models.split(',')]
    if not kinds or len(kinds)!=len(set(kinds)) or not set(kinds)<=set(prev.KINDS) or min(a.min_train_months,a.min_calibration_months)<1 or a.min_stocks<3: raise ValueError('Invalid settings')
    if (run['trees'],run['depth'],run['learning_rate'])!=(500,3,.01): raise ValueError('Unexpected training parameters')
    out=root/a.out
    if out.exists() and not a.check_only: raise FileExistsError(out)
    smoke=a.smoke;a.smoke=False
    d,base,full,labels,folds,coverage=prev.prepare(root,cfg,a)
    a.smoke=smoke
    from src.h2.preprocessing.build_bok_rate_history import make
    _,rates=make();g=strict_gate(d,rates);z,plan=spec.build_plan(d,g,folds,[RULE],a.min_train_months,a.min_stocks)
    if not plan.run_specialist.any(): raise ValueError('No feasible expert folds')
    if smoke:
        yy=int(plan.loc[plan.run_specialist,'valid_year'].min());folds=[f for f in folds if f['valid_year']==yy];plan=plan[plan.valid_year.eq(yy)]
    print('Frozen baseline:',len(base),'| independent H2 features:',FEATURES,'| gate:',RULE)
    print(plan.drop(columns=['train_month_list','eval_month_list']).to_string(index=False))
    print('Ready active target months:',int(plan.loc[plan.run_specialist,'active_eval_months'].sum()))
    sources={}; tables={}
    def add(name,df): tables.setdefault(name,[]).append(df)
    for k in kinds:
        prior,_,sources[k]=spec.load_prior(root,getattr(a,'prior_'+k),k,cfg,d,base,full,labels,folds,a)
        print('Verified existing baseline:',k,len(prior),flush=True)
        if a.check_only: continue
        pred,fit=fit_expert(z,prior,plan,k,run,a);pred,alpha,cal=blend(pred,a.min_calibration_months)
        m,s,c,e=evaluate(pred)
        for name,df in [('predictions',pred),('fit_audit',fit),('alpha_selection',alpha),('calibration_monthly_ic',cal),('monthly_rankic',m),('model_summary',s),('paired_comparisons',c),('h2_only_monthly',e)]: add(name,df)
    if a.check_only: return
    out.mkdir(parents=True,exist_ok=False)
    pairs=pd.concat(tables['paired_comparisons'],ignore_index=True)
    decision=pairs[pairs.scope.eq('all_stocks')&pairs.period.isin(['all','active_ready','adaptive_on'])].copy()
    add('h2_decision',decision);add('gate_fold_coverage',plan);add('gate_months',g);add('monthly_coverage',coverage)
    for name,dfs in tables.items(): pd.concat(dfs,ignore_index=True).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    audit=dict(models=kinds,primary=ADAPTIVE,diagnostic_fixed_weights=list(WEIGHTS),rule=RULE,features=FEATURES,baseline_features=base,folds=folds,smoke=smoke,test_used=False,main_run=run,threads=a.threads,xgb_device=a.xgb_device,min_train_months=a.min_train_months,min_calibration_months=a.min_calibration_months,min_stocks=a.min_stocks,sample_policy=a.sample_policy,period_policy=a.period_policy,prior_runs=sources,code_sha256={str(Path(m.__file__).resolve()):prev.sha(m.__file__) for m in [prev,spec]},script_sha256=prev.sha(__file__),git_commit=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),calibration='only strictly earlier outer-year OOS expert-ready months; maximize mean monthly RankIC, smaller alpha wins ties; fallback zero',gate='positive base-month net rate change; target next month; baseline unchanged outside ready gate',limitations='exploratory; sparse hike months; rank blend uses same observed-return evaluation universe as prior; production rank on prediction universe; hybrid H2 PIT and H4 timing not independently verified; pre-2008 call rate; no causal sign test')
    (out/'run_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(decision[['model','period','arm','months','expert_months','adaptive_on_months','reference_ic','model_ic','delta_ic','hac6_p_two_sided']].to_string(index=False));print('Saved:',out)

if __name__=='__main__': main()
