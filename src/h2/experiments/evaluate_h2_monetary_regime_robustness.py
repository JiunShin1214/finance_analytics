"""Reaggregate existing development predictions by realized monetary regime. No fitting.
Run at repository root. Returns in input/output CSVs are decimals, not percentages.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = {'lgb_rankreg', 'xgb_rankreg', 'xgb_pairwise'}
SCOPES = {'all_eligible', 'complete_case'}
ARMS = {'standalone': ('fixed_4','fixed_8','expanding_4','expanding_8'),
        'integrated': ('baseline_h1_h3','h2_4','h2_8','h2_score_0912')}
START = {'standalone':'2013-01','integrated':'2016-01'}
END = '2021-12'
KEY = ['model','arm','scope','target_month']
METRICS = ['n','ic','rate_change_pp_target','top20_return','bottom20_return',
           'universe_return','top20_minus_universe','long_short']
# Frozen from the existing BOK history used by the original experiment.
# 2012-07 and 2012-10 cuts precede the evaluation window.
CHANGES = {'2013-05':-.25,'2014-08':-.25,'2014-10':-.25,'2015-03':-.25,
           '2015-06':-.25,'2016-06':-.25,'2017-11':.25,'2018-11':.25,
           '2019-07':-.25,'2019-10':-.25,'2020-03':-.5,'2020-05':-.25,
           '2021-08':.25,'2021-11':.25}

def months(a,b):
    return set(pd.period_range(a,b,freq='M').astype(str))

def period_sets(kind):
    window=months(START[kind],END)
    h1=months('2017-11','2018-11') & window
    h2=months('2021-08',END) & window  # ongoing cycle truncated at development end
    c1=months('2012-07','2016-06') & window
    c2=months('2019-07','2020-05') & window
    hike={m for m,v in CHANGES.items() if v>0} & window
    cut={m for m,v in CHANGES.items() if v<0} & window
    hc=h1|h2; cc=c1|c2
    he=(months('2017-11','2019-06')|months('2021-08',END))&window
    ce=(months('2012-07','2017-10')|months('2019-07','2021-07'))&window
    assert not hc&cc and not he&ce and he|ce==window
    return {'all':window,'hike_month':hike,'cut_month':cut,'hold_month':window-hike-cut,
            'hike_cycle':hc,'cut_cycle':cc,'outside_primary_cycles':window-hc-cc,
            'hike_cycle_hold':hc-hike,'cut_cycle_hold':cc-cut,
            'hike_2017_2018':h1,'hike_2021':h2,'cut_2012_2016':c1,'cut_2019_2020':c2,
            'hike_until_next_cut':he,'cut_until_next_hike':ce}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def load_runs(kind, folders):
    frames=[]; provenance=[]; signature=None
    for folder in folders:
        p=folder/'monthly_rankic.csv'; ap=folder/'run_audit.json'
        audit=json.loads(ap.read_text(encoding='utf-8-sig'))
        if audit.get('smoke') is not False or audit.get('train_sample')!='common' or (audit.get('tree_count'),audit.get('depth'),audit.get('learning_rate'))!=(500,3,.01):
            raise ValueError(f'Unexpected smoke/train sample/model settings: {folder}')
        if kind=='standalone':
            if audit.get('test_2022_2025_used') is not False or audit.get('target_month_training_fixed')!=['2009-01','2012-12'] or audit.get('target_month_validation')!=[START[kind],END]:
                raise ValueError(f'Unexpected standalone date/test settings: {folder}')
            if set(audit.get('arms',{}))!=set(ARMS[kind]): raise ValueError('Unexpected standalone arms')
            sig={k:audit[k] for k in ('train_sample','arms','inputs_sha256','portfolio','evaluation')}
        else:
            if audit.get('test_used') is not False or audit.get('validation')!=[START[kind],END] or audit.get('base_controls')!=[] or set(audit.get('arms',[]))!=set(ARMS[kind]):
                raise ValueError(f'Unexpected integrated date/test/base/arm settings: {folder}')
            sig={k:audit[k] for k in ('train_sample','arms','input_sha256','base_controls','h3_selection')}
        if signature is not None and signature!=sig: raise ValueError(f'Input hashes or experiment settings differ: {folder}')
        signature=sig
        d=pd.read_csv(p,encoding='utf-8-sig',dtype={'target_month':str})
        if not set(KEY+METRICS+['regime']).issubset(d.columns): raise ValueError(f'Incorrect monthly_rankic format: {p}')
        if set(d.model)!=set(audit['models']): raise ValueError(f'Model/audit mismatch: {folder}')
        frames.append(d); provenance.append({'folder':str(folder),'monthly_sha256':sha(p),'audit_sha256':sha(ap),'git_commit':audit.get('git_commit')})
    d=pd.concat(frames,ignore_index=True)
    validate(kind,d)
    return d,provenance

def validate(kind,d):
    w=months(START[kind],END)
    if d[KEY].isna().any().any() or d.duplicated(KEY).any(): raise ValueError('Missing or duplicate monthly keys')
    if set(d.model)!=MODELS or set(d.scope)!=SCOPES or set(d.arm)!=set(ARMS[kind]): raise ValueError('Missing/unexpected model, arm or scope')
    if set(d.target_month)!=w: raise ValueError('Invalid month coverage, final test data forbidden')
    if not np.isfinite(d[METRICS].to_numpy(float)).all() or (d.n<3).any() or (d.ic.abs()>1+1e-10).any(): raise ValueError('Invalid metrics')
    r=d.target_month.map(CHANGES).fillna(0.)
    if not np.allclose(r,d.rate_change_pp_target,atol=1e-10,rtol=0) or not d.regime.eq(np.where(r>0,'hike',np.where(r<0,'cut','hold'))).all(): raise ValueError('Rate history or regime mismatch')
    if not np.allclose(d.top20_return-d.universe_return,d.top20_minus_universe,atol=1e-10,rtol=0) or not np.allclose(d.top20_return-d.bottom20_return,d.long_short,atol=1e-10,rtol=0): raise ValueError('Return arithmetic mismatch')
    if len(d)!=len(w)*len(MODELS)*len(SCOPES)*len(ARMS[kind]): raise ValueError('Missing combinations')
    for _,g in d.groupby(['model','arm','scope']):
        if set(g.target_month)!=w: raise ValueError('Incomplete group')
    for _,g in d.groupby(['scope','target_month']):
        if g.n.nunique()!=1 or not np.allclose(g.universe_return,g.universe_return.iloc[0],atol=1e-10,rtol=0): raise ValueError('Evaluation sample mismatch')

def summarize(g):
    down=g[g.universe_return<0]; sd=g.ic.std(ddof=1)
    return dict(months=len(g),mean_rankic=g.ic.mean(),icir=g.ic.mean()/sd if sd>0 else np.nan,
                positive_ic_share=g.ic.gt(0).mean(),mean_top20_return=g.top20_return.mean(),
                mean_universe_return=g.universe_return.mean(),mean_top20_excess=g.top20_minus_universe.mean(),
                mean_long_short=g.long_short.mean(),down_months=len(down),
                down_market_top20_excess=down.top20_minus_universe.mean(),worst_top20_month=g.top20_return.min())

def evaluate(kind,d):
    summaries=[];pairs=[]; membership=[]
    comparisons=( [('fixed_4',a) for a in ('fixed_8','expanding_4','expanding_8')] if kind=='standalone'
                  else [(a,'baseline_h1_h3') for a in ('h2_4','h2_8','h2_score_0912')] )
    for period,ms in period_sets(kind).items():
        membership.extend(dict(experiment=kind,period=period,target_month=m,rate_change_pp=CHANGES.get(m,0.)) for m in sorted(ms))
        if not ms: continue
        sub=d[d.target_month.isin(ms)]
        for (model,arm,scope),g in sub.groupby(['model','arm','scope']):
            summaries.append(dict(experiment=kind,model=model,arm=arm,scope=scope,period=period,**summarize(g)))
        for (model,scope),g in sub.groupby(['model','scope']):
            for arm,ref in comparisons:
                a=g[g.arm.eq(arm)].set_index('target_month').sort_index()
                b=g[g.arm.eq(ref)].set_index('target_month').sort_index()
                if not a.index.equals(b.index) or set(a.index)!=ms: raise ValueError('Unpaired months')
                diff=a.ic-b.ic; ex=a.top20_minus_universe-b.top20_minus_universe; down=a.universe_return<0
                pval=np.nan
                if period=='all' and len(diff)>=24 and diff.std()>0:
                    import statsmodels.api as sm
                    pval=float(sm.OLS(diff.to_numpy(),np.ones((len(diff),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6}).pvalues[0])
                pairs.append(dict(experiment=kind,model=model,scope=scope,period=period,arm=arm,reference=ref,months=len(diff),
                                  mean_rankic_reference=b.ic.mean(),mean_rankic_arm=a.ic.mean(),mean_delta_ic=diff.mean(),ic_win_share=diff.gt(0).mean(),
                                  delta_ic_hac6_p_two_sided=pval,mean_delta_top20_excess=ex.mean(),down_months=int(down.sum()),
                                  mean_delta_down_top20_excess=ex[down].mean(),worst_top20_return_reference=b.top20_return.min(),worst_top20_return_arm=a.top20_return.min()))
    return pd.DataFrame(summaries),pd.DataFrame(pairs),pd.DataFrame(membership)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path('outputs/model_results/h2')
    p.add_argument('--standalone-runs',nargs='+',type=Path,default=[base/'regime_lgb500',base/'regime_xgb500'])
    p.add_argument('--integrated-runs',nargs='+',type=Path,default=[base/'regime_integrated_lgb500',base/'regime_integrated_xgb500'])
    p.add_argument('--out',type=Path,default=base/'monetary_regime_eval_v1')
    p.add_argument('--check-only',action='store_true'); a=p.parse_args()
    results={}; audits={}
    for kind,folders in [('standalone',a.standalone_runs),('integrated',a.integrated_runs)]:
        d,provenance=load_runs(kind,folders); results[kind]=d; audits[kind]=provenance
        print(kind, 'period months:', {k:len(v) for k,v in period_sets(kind).items()})
    if a.check_only: print('PASS: input audit, monthly coverage and rate classification. No files written.'); return
    if a.out.exists(): raise FileExistsError(a.out)
    s=[];c=[];m=[]
    for kind,d in results.items():
        ss,cc,mm=evaluate(kind,d); s.append(ss);c.append(cc);m.append(mm)
    summary=pd.concat(s,ignore_index=True); paired=pd.concat(c,ignore_index=True); membership=pd.concat(m,ignore_index=True)
    primary=['all','hike_month','cut_month','hold_month','hike_cycle','cut_cycle','outside_primary_cycles','hike_until_next_cut','cut_until_next_hike']
    decision=paired.loc[paired.experiment.eq('integrated') & paired.arm.eq('h2_4') & paired.period.isin(primary)].copy()
    a.out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('regime_summary',summary),('regime_paired_comparisons',paired),('regime_months',membership),('h2_decision',decision)]:
        frame.to_csv(a.out/f'{name}.csv',index=False,encoding='utf-8-sig')
    audit={'inputs':audits,'code_sha256':sha(Path(__file__)),'retrained':False,'test_used':False,
           'rate_changes_pp':CHANGES,'primary_cycles':'first through last same-direction hike/cut; ongoing 2021 hike clipped to development end; 2012 cut clipped to evaluation start',
           'sensitivity':'first directional change until month before next opposite change',
           'note':'Realized target-month regimes are ex-post evaluation, not model inputs. HAC only on contiguous all months. Regime subsets descriptive. Gross decimal returns.'}
    (a.out/'regime_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    cols=['model','scope','period','months','mean_rankic_reference','mean_rankic_arm','mean_delta_ic','mean_delta_top20_excess','mean_delta_down_top20_excess']
    print('\nIntegrated H2(4) minus baseline [returns in decimals]\n',decision.loc[decision.scope.eq('all_eligible'),cols].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('Saved:',a.out)

if __name__=='__main__': main()
