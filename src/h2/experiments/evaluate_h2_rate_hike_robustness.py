"""Post-process evaluate_h2_training_period_and_features outputs; no fitting or test-period data.
Run from finance_analytics repository root. Returns are decimals in CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

START, END = '2013-01', '2021-12'
MONTHS = set(pd.period_range(START, END, freq='M').astype(str))
HIKES = {'2017-11', '2018-11', '2021-08', '2021-11'}
ARMS = ['fixed_4', 'fixed_8', 'expanding_4', 'expanding_8']
KEY = ['model', 'arm', 'scope', 'target_month']
METRICS = ['ic','top20_return','bottom20_return','universe_return','top20_minus_universe','long_short','rate_change_pp_target']


def monthset(start, end):
    return set(pd.period_range(start,end,freq='M').astype(str)) & MONTHS


def periods():
    first = monthset('2017-11','2018-11')
    second = monthset('2021-08','2021-12')
    # Sensitivity: from first hike until month before next cut (2019-07); second cycle truncated.
    until_cut = monthset('2017-11','2019-06') | second
    def after(n):
        return {str(pd.Period(m,freq='M')+i) for m in HIKES for i in range(n+1)} & MONTHS
    return {'all': MONTHS, 'hike_month': HIKES,
            'hike_cycle': first | second, 'cycle_2017_2018': first, 'cycle_2021': second,
            'outside_cycle': MONTHS - (first | second), 'cycle_hold_month': (first | second)-HIKES,
            'cycle_until_next_cut': until_cut, 'hike_plus_3m': after(3), 'hike_plus_6m': after(6)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runs(dirs):
    frames, provenance, reference = [], [], None
    for folder in dirs:
        path = folder/'monthly_rankic.csv'; audit_path = folder/'run_audit.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
        if (audit.get('smoke') is not False or audit.get('test_2022_2025_used') is not False
                or audit.get('tree_count') != 500 or audit.get('depth') != 3 or audit.get('learning_rate') != .01
                or audit.get('target_month_training_fixed') != ['2009-01','2012-12']
                or audit.get('target_month_validation') != [START,END]):
            raise ValueError(f'Unexpected experiment settings: {folder}')
        # Require same sample policy, definitions and input hashes across all model runs.
        signature = {k: audit[k] for k in ('train_sample','inputs_sha256','arms','evaluation','portfolio')}
        if reference is not None and signature != reference:
            raise ValueError('Input hashes, sample policy or experiment definitions differ between runs')
        reference = signature
        df = pd.read_csv(path, dtype={'target_month':str}, encoding='utf-8-sig')
        if not set(KEY+METRICS+['n','regime']).issubset(df.columns):
            raise ValueError(f'Not regime experiment monthly_rankic.csv: {path}')
        if set(df.model) != set(audit['models']):
            raise ValueError(f'Models inconsistent with audit: {folder}')
        frames.append(df); provenance.append({'folder':str(folder),'monthly_sha256':sha(path),'audit_sha256':sha(audit_path)})
    data = pd.concat(frames,ignore_index=True)
    validate(data)
    return data, provenance, reference


def validate(data):
    if data[KEY].isna().any().any() or data.duplicated(KEY).any():
        raise ValueError('Missing or duplicate model-arm-scope-month key')
    if set(data.target_month) != MONTHS:
        raise ValueError('Expected exactly 2013-01 to 2021-12; no test months permitted')
    if set(data.scope) != {'all_eligible','complete_case'}:
        raise ValueError('Expected both evaluation scopes')
    if not np.isfinite(data[METRICS+['n']].to_numpy(float)).all():
        raise ValueError('Nonfinite monthly metrics; investigate before averaging')
    if (data.n<3).any() or (data.ic.abs()>1+1e-10).any():
        raise ValueError('Invalid monthly sample count or IC')
    if not np.allclose(data.top20_return-data.universe_return,data.top20_minus_universe,atol=1e-10,rtol=0):
        raise ValueError('Excess return arithmetic mismatch')
    if not np.allclose(data.top20_return-data.bottom20_return,data.long_short,atol=1e-10,rtol=0):
        raise ValueError('Long-short arithmetic mismatch')
    for model,g in data.groupby('model'):
        if set(zip(g.arm,g.scope)) != {(a,s) for a in ARMS for s in ('all_eligible','complete_case')}:
            raise ValueError(f'Missing arms or scopes: {model}')
        for (arm,scope), h in g.groupby(['arm','scope']):
            if set(h.target_month)!=MONTHS: raise ValueError(f'Incomplete months: {model} {arm} {scope}')
    if set(data.loc[data.rate_change_pp_target>0,'target_month']) != HIKES:
        raise ValueError('Unexpected hike months')
    if not data.regime.eq(np.where(data.rate_change_pp_target>0,'hike',np.where(data.rate_change_pp_target<0,'cut','hold'))).all():
        raise ValueError('Rate regime mismatch')
    if (data.groupby('target_month').rate_change_pp_target.nunique()!=1).any():
        raise ValueError('Monthly rate change mismatch')
    for _,g in data.groupby(['scope','target_month']):
        if g.n.nunique()!=1 or not np.allclose(g.universe_return,g.universe_return.iloc[0],atol=1e-10,rtol=0):
            raise ValueError('Monthly evaluation sample counts or universe returns differ')


def summarize(g):
    down=g[g.universe_return<0]; sd=g.ic.std(ddof=1)
    return {'months':len(g),'mean_rankic':g.ic.mean(),'icir':g.ic.mean()/sd if sd>0 else np.nan,
            'positive_ic_share':g.ic.gt(0).mean(),'mean_top20_return':g.top20_return.mean(),
            'mean_universe_return':g.universe_return.mean(),'mean_top20_excess':g.top20_minus_universe.mean(),
            'mean_long_short':g.long_short.mean(),'top20_negative_month_share':g.top20_return.lt(0).mean(),
            'down_market_months':len(down),'down_market_top20_return':down.top20_return.mean(),
            'down_market_universe_return':down.universe_return.mean(), 'down_market_top20_excess':down.top20_minus_universe.mean(),
            'down_market_outperformance_share':down.top20_minus_universe.gt(0).mean(),
            'worst_top20_month':g.top20_return.min()}


def evaluate(data):
    summary, paired, details = [], [], []
    for name,months in periods().items():
        sub=data[data.target_month.isin(months)].copy()
        details.append(sub.assign(period=name))
        for (model,arm,scope),g in sub.groupby(['model','arm','scope']):
            summary.append(dict(model=model,arm=arm,scope=scope,period=name,**summarize(g)))
        for (model,scope),g in sub.groupby(['model','scope']):
            a=g[g.arm.eq('fixed_4')].set_index('target_month')
            for ref in ['fixed_8','expanding_4','expanding_8']:
                b=g[g.arm.eq(ref)].set_index('target_month').reindex(a.index)
                z=a.ic-b.ic; excess=a.top20_minus_universe-b.top20_minus_universe
                down=a.universe_return<0
                paired.append(dict(model=model,scope=scope,period=name,arm='fixed_4',reference=ref,months=len(z),
                    mean_delta_ic=z.mean(),ic_win_share=z.gt(0).mean(),mean_delta_top20_excess=excess.mean(),
                    down_market_months=int(down.sum()),mean_delta_top20_return_down_market=excess[down].mean()))
    return pd.DataFrame(summary),pd.DataFrame(paired),pd.concat(details,ignore_index=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,default=[Path('outputs/model_results/h2/regime_lgb500'),Path('outputs/model_results/h2/regime_xgb500')])
    p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/hike_cycle_eval_v1'))
    p.add_argument('--check-only',action='store_true'); a=p.parse_args()
    data,provenance,settings=load_runs(a.runs)
    for k,v in periods().items(): print(f'{k}: {len(v)} months')
    print('Models:',sorted(data.model.unique()),'| train_sample:',settings['train_sample'])
    if a.check_only: return
    s,c,d=evaluate(data)
    a.out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('cycle_summary',s),('cycle_paired_comparisons',c),('cycle_monthly_detail',d)]:
        frame.to_csv(a.out/f'{name}.csv',index=False,encoding='utf-8-sig')
    audit={'inputs':provenance,'settings':settings,'periods':{k:sorted(v) for k,v in periods().items()},
           'test_used':False,'retrained':False,'returns':'decimal, gross, equal-weight eligible universe benchmark',
           'note':'Ex-post regime analysis only. No causal inference or significance claim. Disjoint periods not compounded.'}
    (a.out/'cycle_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    cols=['model','arm','period','months','mean_rankic','mean_top20_excess','down_market_months','down_market_top20_excess']
    print(s.loc[s.scope.eq('all_eligible') & s.period.isin(['all','hike_month','hike_cycle']),cols].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('Saved:',a.out)

if __name__=='__main__': main()
