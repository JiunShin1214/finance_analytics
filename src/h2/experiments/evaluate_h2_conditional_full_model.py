"""Full model H2 ablation + strictly lagged monetary-policy switching.
Run at repo root; development only. No changes to shared config, data or code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FOUR = ['debt', 'policy', 'policy_change', 'policy_x_debt']
KINDS = ['lgb_rankreg', 'xgb_rankreg', 'xgb_pairwise']
GATES = {'hike_1m': 1, 'hike_3m': 3, 'hike_6m': 6, 'hike_12m': 12, 'hike_until_cut': None}
PRIMARY = 'hike_6m'
SCOPES = ['all_stocks', 'all_eligible', 'complete_case']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def gates_from_rates(rates, targets):
    """Use only rate movements through base-month end, never target-month events."""
    s = rates.set_index('feature_month').rate_change_pp.astype(float).sort_index()
    idx = pd.period_range(s.index.min(), s.index.max(), freq='M').astype(str)
    if not s.index.is_unique or list(s.index) != list(idx) or not np.isfinite(s).all():
        raise ValueError('rate history must be finite, unique and consecutive')
    last_direction, last_change = 0, None
    known = {}
    for m, x in s.items():
        if x != 0:
            last_direction, last_change = int(np.sign(x)), m
        known[m] = (last_direction, last_change)
    rows = []
    for t in sorted(set(targets)):
        base = str(pd.Period(t, freq='M') - 1)
        if base not in known or t not in s.index:
            raise ValueError(f'missing rate history for {t}')
        direction, last = known[base]
        age = (pd.Period(t, freq='M').ordinal - pd.Period(last, freq='M').ordinal) if last else None
        row = dict(target_month=t, base_month=base, last_known_change_month=last,
                   last_known_direction=direction, months_since_known_change=age,
                   realized_target_change_pp=float(s[t]))
        for name, limit in GATES.items():
            row[name] = bool(direction > 0 and age is not None and age >= 1 and (limit is None or age <= limit))
        rows.append(row)
    return pd.DataFrame(rows)


def common_years(counts, minimum):
    """Longest contiguous run of full calendar years with sufficient observed rows."""
    good = counts[counts >= minimum].index.astype(str)
    years = sorted({int(x[:4]) for x in good if int(x[:4]) >= 2001})
    good_set = set(good)
    years = [y for y in years if set(pd.period_range(f'{y}-01', f'{y}-12', freq='M').astype(str)) <= good_set]
    runs = []
    for y in years:
        if not runs or y != runs[-1][-1] + 1:
            runs.append([y])
        else:
            runs[-1].append(y)
    if not runs:
        raise ValueError('no complete common-availability year')
    # Deterministic: earliest start on tie.
    run = max(runs, key=lambda x: (len(x), -x[0]))
    return run[0], run[-1]


def prepare(root, cfg, args):
    from src.evaluation import model_test as mt
    from src.evaluation import base_model_runs as bmr
    mt.check_no_test(cfg)
    if cfg['UNITS']['return'] != 'decimal':
        raise ValueError('decimal return required')
    specs = bmr.feature_specs(cfg, 'boosting')
    full = bmr.input_sets('boosting')['base+h1+h3+h2+h4']
    h2 = [n for n, s in specs.items() if s['path'] == cfg['PATHS']['h2']]
    if set(h2) != set(FOUR) or not set(FOUR) <= set(full):
        raise ValueError('requires the H2 four-feature branch, not the old eight-feature main')
    if not {'h4_interaction', 'epu_raw', 'size', 'bm', 'mom', 'sigma'} <= set(full):
        raise ValueError('unexpected full-model feature set')
    base = [n for n in full if n not in FOUR]
    excluded = cfg['BASE_FEATURES']['model_input']['exclude_train_target_months']
    panel = pd.read_parquet(root/cfg['PATHS']['panel'], columns=['stock_id','month','return','eligible'])
    if not panel.month.between('2000-01','2021-12').all():
        raise PermissionError('development panel only')
    eligibility = pd.read_parquet(root/cfg['PATHS']['eligibility'])
    h3 = mt.compute_h3_sigma_grid(panel, eligibility)
    p = mt.prepare_inputs(root, cfg, h3_grid=h3, extra_features=specs, input_transform='rank', fill_missing=True,
                          exclude_train_target_months=excluded)
    d = p['data'].copy()
    if d.target_month.ge('2022-01').any():
        raise PermissionError('test period forbidden')
    if not ((pd.PeriodIndex(d.base_month, freq='M') + 1).astype(str) == d.target_month.astype(str)).all():
        raise ValueError('base/target month alignment')
    if d[['stock_id','target_month']].isna().any().any() or d.duplicated(['stock_id','target_month']).any():
        raise ValueError('invalid stock-month keys')
    labels = mt.boosting_labels(p)
    xcols = ['x_' + n for n in full if n != 'sigma'] + ['x_sigma_' + n for n in labels]
    if np.isinf(d[xcols].to_numpy(float)).any():
        raise ValueError('infinite inputs')
    # Determine raw observation coverage using flags, never from the zero-filled values.
    h3 = h3[['stock_id','base_month','vol_available',*['sigma_'+n for n in labels]]].copy()
    h3['h3_observed'] = h3.vol_available.eq(True).fillna(False) & np.isfinite(h3[['sigma_'+n for n in labels]].to_numpy(float)).all(axis=1)
    flags = d[['stock_id','base_month']].merge(h3[['stock_id','base_month','h3_observed']],on=['stock_id','base_month'],how='left',validate='one_to_one',sort=False)
    d['eligible'] = d.eligible.eq(True).fillna(False).astype(bool)
    d['common'] = d.eligible & d.h1_available.eq(True).fillna(False) & flags.h3_observed.fillna(False).to_numpy(bool)
    for name in specs:
        d['common'] &= d[name+'_available'].eq(True).fillna(False)
    d = d.loc[d.predict & np.isfinite(d.y)].copy()
    d['fit_member'] = (d.common if args.sample_policy == 'complete_case' else d.eligible) & ~d.target_month.isin(excluded)
    d = d.sort_values(['target_month','stock_id']).reset_index(drop=True)
    coverage = d.groupby('target_month').agg(n_all=('y','size'), n_eligible=('eligible','sum'), n_complete=('common','sum'), n_train=('fit_member','sum'))
    grid = dict(cfg['MODEL_GRID'])
    if args.period_policy == 'common':
        start, end = common_years(coverage.n_complete, args.min_stocks)
        grid['train_start_target_year'] = max(int(grid.get('train_start_target_year',2001)), start)
        grid['first_validation_year'] = max(int(grid['first_validation_year']), grid['train_start_target_year'] + 5)
        grid['last_validation_year'] = min(int(grid['last_validation_year']), end)
    if grid['first_validation_year'] > grid['last_validation_year']:
        raise ValueError('insufficient common period for 4 fit years + 1 inner validation year')
    folds = mt.plan_folds(grid)
    if args.smoke:
        folds = folds[:1]
    for f in folds:
        lo, hi = f['train_years']; yy = f['valid_year']; inner = f['inner_valid_year']
        if not (lo <= hi < inner < yy <= 2021):
            raise ValueError('fold timing')
        fit_months = set(d.loc[d.fit_member & d.year.between(lo,hi),'target_month'])
        expected = set(pd.period_range(f'{lo}-01', f'{hi}-12',freq='M').astype(str)) - set(excluded)
        if not expected <= fit_months:
            raise ValueError(f'fit months missing: {sorted(expected-fit_months)}; use --period-policy common if appropriate')
        for y, mask in [(inner, d.fit_member), (yy, pd.Series(True,index=d.index))]:
            counts = d.loc[mask & d.year.eq(y)].groupby('target_month').size()
            expected = pd.period_range(f'{y}-01',f'{y}-12',freq='M').astype(str)
            if (counts.reindex(expected,fill_value=0) < args.min_stocks).any():
                raise ValueError(f'insufficient rows in {y}; check coverage')
    return d, base, full, labels, folds, coverage.reset_index()


def fit_two_models(d, base, full, labels, folds, kinds, run, args):
    from src.h2.experiments import evaluate_h2_training_period_and_features as old
    from src.evaluation import model_test as mt
    predictions, fit_records, inner_records = [], [], []
    def cols(names, lab):
        return ['x_sigma_' + lab if n == 'sigma' else 'x_' + n for n in names]
    def ic_mean(data, score):
        tab = mt.monthly_ic(data.target_month.to_numpy(),np.asarray(score,float),data.y.to_numpy(float))
        tab.loc[tab.reason.eq('constant_score'),'ic'] = 0.
        if len(tab) != data.target_month.nunique() or not np.isfinite(tab.ic).all():
            raise ValueError('invalid monthly IC')
        return float(tab.ic.mean())
    for kind in kinds:
        for f in folds:
            lo,hi = f['train_years']; yy=f['valid_year']; inner=f['inner_valid_year']
            tr = d.loc[d.fit_member & d.year.between(lo,hi)]
            inn = d.loc[d.fit_member & d.year.eq(inner)]
            val = d.loc[d.year.eq(yy)]
            ev = pd.concat([inn,val],ignore_index=True)
            options = []
            for lab in labels:
                pr, _, n = old.fit_predict(tr,ev,cols(base,lab),kind,run,args.threads,args.xgb_device)
                score = ic_mean(inn,pr[:len(inn)])
                options.append((score,int(lab.removeprefix('span')),lab,pr[len(inn):]))
                inner_records.append(dict(model=kind,valid_year=yy,h3_label=lab,baseline_inner_ic=score,trees=n))
            best = max(options,key=lambda v:(v[0],v[1]))
            _,_,lab,p0 = best
            p1,_,n = old.fit_predict(tr,val,cols(full,lab),kind,run,args.threads,args.xgb_device)
            b = val[['stock_id','base_month','target_month','y','eligible','common']].copy()
            b['model']=kind; b['fold']=f['fold']; b['without_h2']=p0; b['with_h2']=p1
            predictions.append(b)
            fit_records.append(dict(model=kind,valid_year=yy,fit_start=tr.target_month.min(),fit_end=tr.target_month.max(),
                                    inner_year=inner,fit_rows=len(tr),inner_rows=len(inn),eval_rows=len(val),h3_label=lab,
                                    baseline_inner_ic=best[0],trees=n,baseline_features=','.join(cols(base,lab)),
                                    full_features=','.join(cols(full,lab))))
            print(f'{kind} | {yy} | fit={len(tr):,} | H3={lab} | baseline/full finished',flush=True)
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(fit_records),pd.DataFrame(inner_records)


def evaluate(pred, gates):
    from src.h2.experiments import evaluate_h2_training_period_and_features as old
    from src.h2.experiments.evaluate_h2_rate_hike_robustness import summarize
    from src.evaluation import model_test as mt
    import statsmodels.api as sm
    p = pred.merge(gates,on=['base_month','target_month'],how='left',validate='many_to_one')
    if p[list(GATES)].isna().any().any() or p.duplicated(['model','stock_id','target_month']).any():
        raise ValueError('gate alignment or duplicate forecast')
    # Whole-month routing: no mixing scores of different scales within a cross-section.
    arms = ['without_h2','with_h2',*GATES]
    for name in GATES:
        p[name] = np.where(p[name],p.with_h2,p.without_h2)
    rows = []
    for kind,g in p.groupby('model',sort=True):
        for scope,sub in [('all_stocks',g),('all_eligible',g[g.eligible]),('complete_case',g[g.common])]:
            for m,h in sub.groupby('target_month',sort=True):
                if len(h)<3:
                    if scope=='complete_case': continue
                    raise ValueError(f'insufficient evaluation rows {scope} {m}')
                y=h.y.to_numpy(float); uni=float(y.mean())
                for arm in arms:
                    s=h[arm].to_numpy(float)
                    t=mt.monthly_ic(h.target_month.to_numpy(),s,y).iloc[0]
                    ic=0. if t.reason=='constant_score' else float(t.ic)
                    if not np.isfinite(ic): raise ValueError(f'nonfinite IC {kind} {scope} {m} {arm}')
                    top=old.weighted_tail(s,y); bottom=old.weighted_tail(s,y,False)
                    rows.append(dict(model=kind,scope=scope,target_month=m,arm=arm,n=len(h),ic=ic,
                                     top20_return=top,bottom20_return=bottom,universe_return=uni,
                                     top20_minus_universe=top-uni,long_short=top-bottom))
    m=pd.DataFrame(rows).merge(gates.drop(columns='base_month'),on='target_month',validate='many_to_one')
    summaries=[]; pairs=[]
    for (kind,scope),g in m.groupby(['model','scope']):
        periods={'all':g,'2016_2021':g[g.target_month.between('2016-01','2021-12')],
                 'primary_on':g[g[PRIMARY]],'primary_off':g[~g[PRIMARY]],
                 'actual_hike':g[g.realized_target_change_pp>0],'actual_cut':g[g.realized_target_change_pp<0],
                 'actual_hold':g[g.realized_target_change_pp==0]}
        for period,sub in periods.items():
            if sub.empty: continue
            a=sub[sub.arm.eq('without_h2')].set_index('target_month').sort_index()
            for arm in arms:
                b=sub[sub.arm.eq(arm)].set_index('target_month').sort_index()
                if not b.index.equals(a.index) or not b.n.equals(a.n) or not np.allclose(b.universe_return,a.universe_return):
                    raise ValueError('different sample across arms')
                summaries.append(dict(model=kind,scope=scope,period=period,arm=arm,**summarize(b)))
                if arm=='without_h2': continue
                z=b.ic-a.ic; ex=b.top20_minus_universe-a.top20_minus_universe
                down=a.universe_return<0
                consecutive=list(z.index)==list(pd.period_range(z.index.min(),z.index.max(),freq='M').astype(str))
                pval=np.nan; low=np.nan; high=np.nan
                if period=='all' and consecutive and len(z)>=24 and z.std()>0:
                    result=sm.OLS(z.to_numpy(),np.ones((len(z),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6})
                    pval=float(result.pvalues[0]);low,high=map(float,result.conf_int()[0])
                pairs.append(dict(model=kind,scope=scope,period=period,arm=arm,months=len(z),
                                  baseline_ic=a.ic.mean(),model_ic=b.ic.mean(),delta_ic=z.mean(),
                                  hac6_p_two_sided=pval,hac6_ci_low=low,hac6_ci_high=high,
                                  delta_top20_excess=ex.mean(),delta_down_top20_excess=ex[down].mean(),
                                  ic_win_share=z.gt(0).mean()))
    return m,pd.DataFrame(summaries),pd.DataFrame(pairs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--models',default=','.join(KINDS))
    p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/conditional_full_v1'))
    p.add_argument('--period-policy',choices=['full','common'],default='full')
    p.add_argument('--sample-policy',choices=['fill','complete_case'],default='fill')
    p.add_argument('--min-stocks',type=int,default=20);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--check-only',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();root=a.root.resolve();cfg_path=root/'config/experiment.yaml'
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8')); run=cfg['BASE_FEATURES']['model_input']['main_run']
    if (run['trees'],run['depth'],run['learning_rate'])!=(500,3,.01):
        raise ValueError('expected team settings 500 trees, depth 3, lr .01')
    kinds=[x.strip() for x in a.models.split(',') if x.strip()]
    if not kinds or len(kinds)!=len(set(kinds)) or not set(kinds)<=set(KINDS): raise ValueError('invalid models')
    paths=[root/cfg['PATHS'][n] for n in ['panel','eligibility','h1_components','base_features','h2','h4']]
    missing=[str(x) for x in paths if not x.is_file()]
    if missing: raise FileNotFoundError('Required data missing:\n'+'\n'.join(missing))
    out=root/a.out
    if not a.check_only and out.exists(): raise FileExistsError(f'Choose a new --out: {out}')
    from src.h2.preprocessing.build_bok_rate_history import make
    d,base,full,labels,folds,coverage=prepare(root,cfg,a)
    _,rates=make()
    years=[f['valid_year'] for f in folds]
    targets=pd.period_range(f'{years[0]}-01',f'{years[-1]}-12',freq='M').astype(str)
    gates=gates_from_rates(rates,targets)
    print('Full features:',len(full),full,'\nWithout H2:',len(base),base)
    print('Folds:',years,'| H3 candidates:',labels,'| fit policy:',a.sample_policy,'| period:',a.period_policy)
    print('All-stocks evaluation:',int(d.year.isin(years).sum()),'rows | eligible:',int((d.year.isin(years)&d.eligible).sum()),'| complete:',int((d.year.isin(years)&d.common).sum()))
    print('ON months:',{g:int(gates[g].sum()) for g in GATES},'/',len(gates),'| primary:',PRIMARY)
    print(gates.loc[gates[PRIMARY],['target_month','last_known_change_month','months_since_known_change']].to_string(index=False))
    if a.check_only: return
    pred,fit,inner=fit_two_models(d,base,full,labels,folds,kinds,run,a)
    m,s,c=evaluate(pred,gates)
    out.mkdir(parents=True,exist_ok=False)
    decision=c[c.period.eq('all')].copy()
    for name,frame in [('predictions',pred),('monthly_rankic',m),('model_summary',s),('paired_comparisons',c),('h2_decision',decision),('gate_months',gates),('fit_audit',fit),('h3_selection',inner),('monthly_coverage',coverage)]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    code=[Path(__file__).resolve(),*[root/p for p in ['src/evaluation/model_test.py','src/evaluation/base_model_runs.py','src/models/boosting.py','src/features/base.py','src/features/h3.py','src/features/h2.py','src/h2/preprocessing/build_bok_rate_history.py','src/h2/experiments/evaluate_h2_training_period_and_features.py','src/h2/experiments/evaluate_h2_rate_hike_robustness.py']]]
    audit=dict(models=kinds,features_without_h2=base,features_with_h2=full,primary_gate=PRIMARY,gate_rules=GATES,
               gate_timing='last known nonzero policy rate move as of base-month end; target move is descriptive only',
               period_policy=a.period_policy,sample_policy=a.sample_policy,minimum_stocks=a.min_stocks,folds=folds,
               test_used=False,smoke=a.smoke,main_run=run,threads=a.threads,xgb_device=a.xgb_device,
               h3_selection='baseline inner IC at 500 trees, shared span, no refit',
               input_sha256={str(x):sha(x) for x in [cfg_path,*paths]},code_sha256={str(x):sha(x) for x in code},
               git_commit=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
               limitations='exploratory gate selection on development data; pre-2008 call rate; H2 hybrid PIT and H4 EPU availability not independently verified')
    (out/'run_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(decision[decision.scope.eq('all_stocks')].to_string(index=False)); print('Saved:',out)

if __name__=='__main__': main()
