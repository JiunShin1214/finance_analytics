"""Regime-specialized H2 fit + prediction, using audited full-training predictions.
Run at repo root. Development only; no shared file changes. Gate is known at base-month end.
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

KINDS = prev.KINDS
RULES = ('hike_6m', 'hike_12m', 'hike_until_cut')
PRIMARY = 'hike_6m'
ARMS = ('A_all_without_h2', 'B_all_with_h2_routed', 'C_hike_without_h2', 'D_hike_with_h2')
KEY = ['stock_id', 'base_month', 'target_month']
FOLD_KEYS = ('fold', 'valid_year', 'inner_valid_year', 'train_years')


def month_range(a, b):
    return set(pd.period_range(a, b, freq='M').astype(str))


def norm(s):
    return str(s).replace('\\', '/').lower()


def hash_match(recorded, root, rel):
    """Match Windows absolute audit paths against current repository-relative paths."""
    suffix = norm(rel).strip('/')
    hits = [v for k, v in recorded.items() if norm(k) == suffix or norm(k).endswith('/' + suffix)]
    if len(hits) != 1 or hits[0] != prev.sha(root / rel):
        raise ValueError(f'Prior run hash differs or missing: {rel}. Re-run conditional_full on current data/code with a new folder.')


def fold_signature(f):
    return tuple(tuple(f[k]) if k == 'train_years' else f[k] for k in FOLD_KEYS)


def load_prior(root, folder, kind, cfg, d, base, full, labels, folds, args):
    folder = root / folder
    paths = {k: folder / k for k in ('run_audit.json', 'predictions.csv', 'fit_audit.csv', 'monthly_rankic.csv')}
    a = json.loads(paths['run_audit.json'].read_text(encoding='utf-8-sig'))
    if (a.get('smoke') is not False or a.get('test_used') is not False or kind not in a.get('models', [])
            or a.get('features_without_h2') != base or a.get('features_with_h2') != full
            or a.get('sample_policy') != args.sample_policy or a.get('period_policy') != args.period_policy
            or a.get('minimum_stocks') != args.min_stocks or a.get('main_run') != cfg['BASE_FEATURES']['model_input']['main_run']
            or a.get('h3_selection') != 'baseline inner IC at 500 trees, shared span, no refit'):
        raise ValueError(f'Unexpected prior settings: {folder}')
    if kind.startswith('xgb') and a.get('xgb_device') != args.xgb_device:
        raise ValueError('Prior XGB device differs; match --xgb-device for controlled comparison')
    if a.get('threads') != args.threads:
        raise ValueError('Prior threads differ; match --threads')
    expected = {fold_signature(f) for f in folds}
    saved = {fold_signature(f) for f in a['folds']}
    if not expected <= saved or (not args.smoke and expected != saved):
        raise ValueError('Prior folds differ')
    for rel in ['config/experiment.yaml', *[cfg['PATHS'][k] for k in ('panel','eligibility','h1_components','base_features','h2','h4')]]:
        hash_match(a['input_sha256'], root, rel)
    # Every code file tracked by previous script must match. Do not silently compare revised features/models.
    code = ['src/evaluation/model_test.py','src/evaluation/base_model_runs.py','src/models/boosting.py',
            'src/features/base.py','src/features/h3.py','src/features/h2.py','src/h2/preprocessing/build_bok_rate_history.py',
            'src/h2/experiments/evaluate_h2_training_period_and_features.py','src/h2/experiments/evaluate_h2_rate_hike_robustness.py',
            'src/h2/experiments/evaluate_h2_conditional_full_model.py']
    for rel in code:
        hash_match(a['code_sha256'], root, rel)
    years = [f['valid_year'] for f in folds]
    p = pd.read_csv(paths['predictions.csv'], dtype={k:str for k in KEY}, encoding='utf-8-sig')
    p = p.loc[p.model.eq(kind) & p.target_month.str[:4].astype(int).isin(years)].copy()
    if p[KEY].isna().any().any() or p.duplicated(KEY).any() or not np.isfinite(p[['without_h2','with_h2','y']]).all().all():
        raise ValueError('Prior predictions invalid')
    actual = d.loc[d.year.isin(years), KEY+['y','eligible','common']].copy()
    actual[KEY] = actual[KEY].astype(str)
    m = actual.merge(p[KEY+['y','eligible','common','fold','without_h2','with_h2']],on=KEY,how='outer',indicator=True,suffixes=('','_old'),validate='one_to_one')
    if not m._merge.eq('both').all() or not np.allclose(m.y,m.y_old,rtol=1e-10,atol=1e-12):
        raise ValueError('Prior stock-month keys or targets differ')
    for col in ('eligible','common'):
        if not m[col].eq(m[col+'_old']).all(): raise ValueError(f'Prior {col} flag differs')
    fits = pd.read_csv(paths['fit_audit.csv'], dtype={'fit_start':str,'fit_end':str}, encoding='utf-8-sig')
    fits = fits.loc[fits.model.eq(kind) & fits.valid_year.isin(years)].copy()
    if len(fits)!=len(folds) or fits.valid_year.duplicated().any() or not set(fits.h3_label)<=set(labels) or not fits.trees.eq(500).all():
        raise ValueError('Prior fit audit invalid')
    fits = fits.set_index('valid_year')
    for f in folds:
        yy = f['valid_year']; lo, hi = f['train_years']; row = fits.loc[yy]
        tr = d.loc[d.fit_member & d.year.between(lo, hi)]
        inn = d.loc[d.fit_member & d.year.eq(f['inner_valid_year'])]
        val = m.loc[m.target_month.str[:4].astype(int).eq(yy)]
        if (row.fit_start != tr.target_month.min() or row.fit_end != tr.target_month.max()
            or int(row.fit_rows)!=len(tr) or int(row.inner_rows)!=len(inn) or int(row.eval_rows)!=len(val)
            or int(row.inner_year)!=f['inner_valid_year'] or not val.fold.eq(f['fold']).all()):
            raise ValueError(f'Prior fit mismatch: {yy}')
        for name, features in [('baseline_features',base),('full_features',full)]:
            if row[name] != ','.join(columns(features,row.h3_label)):
                raise ValueError(f'Prior feature order mismatch: {yy} {name}')
    # Recompute prior monthly scores to verify stored forecasts reproduce previous tables.
    from src.evaluation import model_test as mt
    old_m = pd.read_csv(paths['monthly_rankic.csv'],dtype={'target_month':str},encoding='utf-8-sig')
    old_m = old_m.loc[old_m.model.eq(kind) & old_m.scope.eq('all_stocks') & old_m.arm.isin(['without_h2','with_h2']) & old_m.target_month.isin(m.target_month)]
    if old_m.duplicated(['arm','target_month']).any() or len(old_m)!=2*m.target_month.nunique():
        raise ValueError('Prior monthly IC coverage invalid')
    for arm in ('without_h2','with_h2'):
        ic = mt.monthly_ic(m.target_month.to_numpy(),m[arm].to_numpy(float),m.y.to_numpy(float))
        ic.loc[ic.reason.eq('constant_score'),'ic']=0.
        q = ic.merge(old_m.loc[old_m.arm.eq(arm),['target_month','ic']],on='target_month',suffixes=('','_old'),validate='one_to_one')
        if len(q)!=m.target_month.nunique() or not np.allclose(q.ic,q.ic_old,atol=1e-8,rtol=0):
            raise ValueError(f'Prior predictions do not reproduce IC: {arm}')
    m = m[KEY+['y','eligible','common','fold','without_h2','with_h2']].sort_values(['target_month','stock_id']).reset_index(drop=True)
    return m, fits, {'folder':str(folder), 'sha256':{k:prev.sha(v) for k,v in paths.items()}, 'prior_git_commit':a.get('git_commit')}


def columns(features, lab):
    return ['x_sigma_'+lab if n=='sigma' else 'x_'+n for n in features]


def make_gates(rates, d):
    # 2001-01 has base month 2000-12, absent from make() monthly table and excluded from fits.
    # Build the gate from the earliest target with an available prior month; never guess a regime.
    first = str(pd.Period(rates.feature_month.min(),freq='M')+1)
    targets = sorted(set(d.target_month[d.target_month.ge(first)]))
    g = prev.gates_from_rates(rates, targets)
    lookup = g.set_index('target_month')
    absent = d.fit_member & ~d.target_month.isin(lookup.index)
    if absent.any(): raise ValueError('Training months without gate history')
    if not (g.last_known_change_month.dropna() < g.loc[g.last_known_change_month.notna(),'target_month']).all():
        raise ValueError('Future gate information')
    return g


def build_plan(d, gates, folds, rules, min_train_months, min_stocks):
    z = d.merge(gates[['target_month',*rules]],on='target_month',how='left',validate='many_to_one')
    for rule in rules: z[rule]=z[rule].eq(True)
    plan=[]
    for rule in rules:
        for f in folds:
            yy=f['valid_year']; lo,hi=f['train_years']
            tr=z.loc[z.fit_member & z.year.between(lo,hi) & z[rule]]
            counts=tr.groupby('target_month').size()
            eval_months=sorted(z.loc[z.year.eq(yy)&z[rule],'target_month'].unique())
            fit_months=sorted(counts.index)
            # Do not silently discard thin months: refuse specialist for that fold, use A fallback.
            ready=len(counts)>=min_train_months and bool((counts>=min_stocks).all())
            plan.append(dict(rule=rule,fold=f['fold'],valid_year=yy,train_start=f'{lo}-01',train_end=f'{hi}-12',
                             train_months=len(counts),train_rows=len(tr),min_month_stocks=int(counts.min()) if len(counts) else 0,
                             train_month_list=';'.join(fit_months),active_eval_months=len(eval_months),eval_month_list=';'.join(eval_months),
                             ready=ready,run_specialist=ready and bool(eval_months)))
    return z,pd.DataFrame(plan)


def fit_specialists(z, prior, fits, plan, base, full, kind, rule, run, args):
    from src.h2.experiments import evaluate_h2_training_period_and_features as old
    p=prior.copy()
    p['A_all_without_h2']=p.without_h2
    for arm in ARMS[1:]: p[arm]=p.without_h2
    active_months=set(z.loc[z[rule],'target_month'])
    p['gate_on']=p.target_month.isin(active_months)
    p['specialist_used']=False
    records=[]
    for r in plan.loc[plan.rule.eq(rule)].itertuples(index=False):
        yy=r.valid_year
        record=dict(model=kind,rule=rule,fold=r.fold,valid_year=yy,train_months=r.train_months,train_rows=r.train_rows,
                    active_eval_months=r.active_eval_months,h3_label=fits.loc[yy,'h3_label'],status='no_active_month' if not r.active_eval_months else 'fallback_insufficient_train',trees=0)
        if not r.run_specialist:
            records.append(record); continue
        lab=record['h3_label']
        tr=z.loc[z.fit_member & z.target_month.between(r.train_start,r.train_end) & z[rule]].copy()
        val=z.loc[z.year.eq(yy) & z[rule]].copy()
        tr=tr.sort_values(['target_month','stock_id']); val=val.sort_values(['target_month','stock_id'])
        if not len(val) or len(tr)!=r.train_rows or not tr[rule].all() or not val[rule].all() or tr.target_month.max()>=val.target_month.min():
            raise ValueError('Specialist sample or time violation')
        pred0,_,n0=old.fit_predict(tr,val,columns(base,lab),kind,run,args.threads,args.xgb_device)
        pred1,_,n1=old.fit_predict(tr,val,columns(full,lab),kind,run,args.threads,args.xgb_device)
        if n0!=500 or n1!=500: raise ValueError('Unexpected tree count')
        new=val[KEY].copy();new[KEY]=new[KEY].astype(str)
        new['p0']=pred0;new['p1']=pred1
        select=p.fold.eq(r.fold)&p.gate_on
        q=p.loc[select,KEY].merge(new,on=KEY,how='left',validate='one_to_one',sort=False)
        if len(q)!=len(new) or not np.isfinite(q[['p0','p1']]).all().all(): raise ValueError('Specialist forecast key mismatch')
        p.loc[select,'B_all_with_h2_routed']=p.loc[select,'with_h2']
        p.loc[select,'C_hike_without_h2']=q.p0.to_numpy(float)
        p.loc[select,'D_hike_with_h2']=q.p1.to_numpy(float)
        p.loc[select,'specialist_used']=True
        records.append(dict(record,status='fit',trees=500,fit_actual_start=tr.target_month.min(),fit_actual_end=tr.target_month.max(),eval_rows=len(val)))
        print(f'{kind} | {rule} | {yy} | hike fit {len(tr):,} rows / {r.train_months} months | eval {r.active_eval_months} months | H3 {lab}',flush=True)
    if (p.specialist_used & ~p.gate_on).any(): raise ValueError('Routing outside gate')
    if not np.isfinite(p[list(ARMS)]).all().all(): raise ValueError('Invalid forecasts')
    p['model']=kind;p['rule']=rule
    return p,pd.DataFrame(records)


def evaluate(p):
    from src.h2.experiments import evaluate_h2_training_period_and_features as old
    from src.h2.experiments.evaluate_h2_rate_hike_robustness import summarize
    from src.evaluation import model_test as mt
    import statsmodels.api as sm
    rows=[]
    for scope,sub in [('all_stocks',p),('all_eligible',p[p.eligible]),('complete_case',p[p.common])]:
        for m,h in sub.groupby('target_month',sort=True):
            if len(h)<3:
                if scope=='complete_case': continue
                raise ValueError('Insufficient evaluation sample')
            if h.specialist_used.nunique()!=1 or h.gate_on.nunique()!=1: raise ValueError('Stock-specific gate')
            y=h.y.to_numpy(float); uni=y.mean()
            for arm in ARMS:
                score=h[arm].to_numpy(float)
                q=mt.monthly_ic(h.target_month.to_numpy(),score,y).iloc[0]
                ic=0. if q.reason=='constant_score' else float(q.ic)
                if not np.isfinite(ic): raise ValueError('Invalid IC')
                top=old.weighted_tail(score,y); bottom=old.weighted_tail(score,y,False)
                rows.append(dict(model=h.model.iloc[0],rule=h.rule.iloc[0],scope=scope,target_month=m,arm=arm,n=len(h),ic=ic,
                                 gate_on=bool(h.gate_on.iloc[0]),specialist_used=bool(h.specialist_used.iloc[0]),
                                 top20_return=top,bottom20_return=bottom,universe_return=uni,top20_minus_universe=top-uni,long_short=top-bottom))
    monthly=pd.DataFrame(rows); summary=[]; pairs=[]
    comparisons=[(ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[3],ARMS[0]),(ARMS[3],ARMS[2]),(ARMS[3],ARMS[1])]
    for (model,rule,scope),g in monthly.groupby(['model','rule','scope']):
        periods={'all':g,'active_ready':g[g.specialist_used], '2016_2021':g[g.target_month.ge('2016-01')],
                 'active_ready_2016_2021':g[g.specialist_used&g.target_month.ge('2016-01')],
                 'gate_on_fallback':g[g.gate_on&~g.specialist_used]}
        for period,sub in periods.items():
            if sub.empty: continue
            by={a:sub[sub.arm.eq(a)].set_index('target_month').sort_index() for a in ARMS}
            ref=by[ARMS[0]]
            for arm,b in by.items():
                if not b.index.equals(ref.index) or not b.n.equals(ref.n) or not np.allclose(b.universe_return,ref.universe_return):
                    raise ValueError('Unequal evaluation sample')
                summary.append(dict(model=model,rule=rule,scope=scope,period=period,arm=arm,**summarize(b)))
            for arm,reference in comparisons:
                a,b=by[reference],by[arm]; z=b.ic-a.ic; ex=b.top20_minus_universe-a.top20_minus_universe; down=a.universe_return<0
                pv=low=high=np.nan
                if period=='all' and len(z)>=24 and z.std()>0 and list(z.index)==sorted(month_range(z.index.min(),z.index.max())):
                    r=sm.OLS(z.to_numpy(),np.ones((len(z),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6})
                    pv=float(r.pvalues[0]);low,high=map(float,r.conf_int()[0])
                pairs.append(dict(model=model,rule=rule,scope=scope,period=period,arm=arm,reference=reference,
                                  months=len(z),gate_months=int(b.gate_on.sum()),specialist_months=int(b.specialist_used.sum()),
                                  reference_ic=a.ic.mean(),model_ic=b.ic.mean(),delta_ic=z.mean(),ic_win_share=z.gt(0).mean(),
                                  delta_top20_excess=ex.mean(),delta_down_top20_excess=ex[down].mean(),
                                  hac6_p_two_sided=pv,hac6_ci_low=low,hac6_ci_high=high))
    return monthly,pd.DataFrame(summary),pd.DataFrame(pairs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--models',default=','.join(KINDS))
    p.add_argument('--rules',default=','.join(RULES),help='comma-separated gate names; hike_1m/hike_3m also supported for sensitivity')
    p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/regime_specialized_v1'))
    for kind in KINDS:
        p.add_argument('--prior-'+kind.replace('_','-'),type=Path,default=Path(f'outputs/model_results/h2/conditional_full_{kind}_v1'))
    p.add_argument('--sample-policy',choices=['fill','complete_case'],default='fill');p.add_argument('--period-policy',choices=['full','common'],default='full')
    p.add_argument('--min-stocks',type=int,default=20);p.add_argument('--min-train-months',type=int,default=12)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--check-only',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();root=a.root.resolve();cfg_path=root/'config/experiment.yaml'
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'));run=cfg['BASE_FEATURES']['model_input']['main_run']
    if (run['trees'],run['depth'],run['learning_rate'])!=(500,3,.01): raise ValueError('Expected 500 trees, depth 3, lr .01')
    kinds=[x.strip() for x in a.models.split(',')];rules=[x.strip() for x in a.rules.split(',')]
    if len(kinds)!=len(set(kinds)) or not set(kinds)<=set(KINDS) or not kinds: raise ValueError('Invalid models')
    if len(rules)!=len(set(rules)) or not set(rules)<=set(prev.GATES) or not rules or a.min_train_months<1 or a.min_stocks<3: raise ValueError('Invalid gate settings')
    out=root/a.out
    if not a.check_only and out.exists(): raise FileExistsError(out)
    from src.h2.preprocessing.build_bok_rate_history import make
    # Smoke uses the full plan first to select the first evaluable fold (not necessarily 2006).
    requested_smoke=a.smoke;a.smoke=False
    d,base,full,labels,folds,coverage=prev.prepare(root,cfg,a)
    a.smoke=requested_smoke
    _,rates=make();gates=make_gates(rates,d);z,plan=build_plan(d,gates,folds,rules,a.min_train_months,a.min_stocks)
    if not plan.run_specialist.any(): raise ValueError('No feasible specialist folds')
    if a.smoke:
        yy=int(plan.loc[plan.run_specialist,'valid_year'].min());folds=[f for f in folds if f['valid_year']==yy];plan=plan.loc[plan.valid_year.eq(yy)].copy()
    print('Full / without H2:',len(full),'/',len(base),'| rules:',rules,'| min active fit months:',a.min_train_months)
    print(plan.drop(columns=['train_month_list','eval_month_list']).to_string(index=False))
    print('Ready active target months by rule:',plan.loc[plan.run_specialist].groupby('rule').active_eval_months.sum().to_dict())
    sources={};prior={};fits={}
    for kind in kinds:
        prior[kind],fits[kind],sources[kind]=load_prior(root,getattr(a,'prior_'+kind),kind,cfg,d,base,full,labels,folds,a)
        print('Verified previous full-training forecasts:',kind,len(prior[kind]),flush=True)
    if a.check_only:return
    preds=[];fit_rows=[];monthly=[];summary=[];paired=[]
    for kind in kinds:
        for rule in rules:
            pr,fr=fit_specialists(z,prior[kind],fits[kind],plan,base,full,kind,rule,run,a)
            m,s,c=evaluate(pr);preds.append(pr);fit_rows.append(fr);monthly.append(m);summary.append(s);paired.append(c)
    c=pd.concat(paired,ignore_index=True)
    decision=c[c.scope.eq('all_stocks') & c.period.isin(['all','active_ready']) & c.arm.eq(ARMS[3])].copy()
    out.mkdir(parents=True,exist_ok=False)
    for name,df in [('h2_decision',decision),('model_summary',pd.concat(summary,ignore_index=True)),('paired_comparisons',c),
                    ('monthly_rankic',pd.concat(monthly,ignore_index=True)),('predictions',pd.concat(preds,ignore_index=True)),
                    ('fit_audit',pd.concat(fit_rows,ignore_index=True)),('gate_fold_coverage',plan),('gate_months',gates),('monthly_coverage',coverage)]:
        df.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    audit=dict(models=kinds,rules=rules,primary_rule=PRIMARY,arms=ARMS,folds=folds,smoke=a.smoke,test_used=False,
               sample_policy=a.sample_policy,period_policy=a.period_policy,min_train_months=a.min_train_months,min_stocks=a.min_stocks,
               main_run=run,threads=a.threads,xgb_device=a.xgb_device,prior_runs=sources,
               input_sha256={str(x):prev.sha(root/x) for x in [Path('config/experiment.yaml'),*[Path(cfg['PATHS'][k]) for k in ('panel','eligibility','h1_components','base_features','h2','h4')]]},
               code_sha256={str(Path(__file__).resolve()):prev.sha(Path(__file__)),str(Path(prev.__file__).resolve()):prev.sha(Path(prev.__file__))},
               git_commit=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
               gate='as-of base-month end; last known move hike; age limit; subsequent cut disables. Training and specialist predictions restricted to same gate.',
               h3_selection='reuse full baseline prior fold span as shared external control; no specialist hyperparameter tuning; no refit on inner year',
               fallback='all arms use A outside gate or if < minimum active training months / minimum stocks in each active month; compare A/B/C/D on identical ready months',
               inference='HAC(6) on consecutive whole-period paired differences including zero inactive months; active-only statistics descriptive; multiple comparisons exploratory',
               limitations='few independent tightening cycles; pre-2008 call rate; H2 hybrid PIT and H4 EPU publication timing not independently verified')
    (out/'run_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(decision[['model','rule','period','reference','months','specialist_months','reference_ic','model_ic','delta_ic','hac6_p_two_sided']].to_string(index=False))
    print('Saved:',out)


if __name__=='__main__':main()
