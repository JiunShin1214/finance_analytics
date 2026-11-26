"""H1+H3 vs +H2(8), +H2(4), +H2 score trained on 2009–2012.
Requires earlier run_h2_regime_fixed outputs and the repository's model_test.
All target dates are development only. No shared code/data is modified.
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

from src.h2.experiments import evaluate_h2_training_period_and_features as old

START, END = '2013-01', '2021-12'
FIRST_OUT = 2016
ARMS = ('baseline_h1_h3', 'h2_8', 'h2_4', 'h2_score_0912')
KEY = ['stock_id', 'target_month']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def month_set(a,b):
    return set(pd.period_range(a,b,freq='M').astype(str))


def period_sets(last):
    all_m = month_set(f'{FIRST_OUT}-01', last)
    hike = {'2017-11','2018-11','2021-08','2021-11'} & all_m
    first = month_set('2017-11','2018-11') & all_m
    second = month_set('2021-08','2021-12') & all_m
    cycle = first | second
    return {'all':all_m,'hike_month':hike,'hike_cycle':cycle,'cycle_2017_2018':first,'cycle_2021':second,
            'outside_cycle':all_m-cycle,'cycle_hold_month':cycle-hike,
            'cycle_until_next_cut':(month_set('2017-11','2019-06')|second)&all_m}


def assert_window(d, a,b, minimum=20):
    old.check_window(d,a,b,min_stocks=minimum)


def prepare(root,cfg,include_base=False):
    from src.evaluation import model_test as mt, base_model_runs as bmr
    full = bmr.feature_specs(cfg,'boosting')
    if not set(old.EIGHT) <= set(full):
        raise ValueError('Pull latest main: H2 eight-feature specs missing')
    base = list(bmr.input_sets('boosting')['base']) if include_base else []
    names = [*base,*old.EIGHT]
    specs = {n:full[n] for n in names}
    p = mt.prepare_inputs(root,cfg,extra_features=specs,input_transform='rank',fill_missing=False)
    labels = mt.boosting_labels(p)
    d = p['data'].copy()
    if d.target_month.ge('2022-01').any(): raise PermissionError('test data')
    d = d.loc[d.target_month.between(START,END) & d.eligible.eq(True) & np.isfinite(d.y)].copy()
    d['stock_id'] = old.stock(d.stock_id)
    d['target_month'] = d.target_month.astype(str)
    d = d.sort_values(['target_month','stock_id']).reset_index(drop=True)
    if d[KEY].isna().any().any() or d.duplicated(KEY).any(): raise ValueError('invalid prepared keys')
    xcols = ['x_'+n for n in [*base,'size','bm','mom',*old.EIGHT]]+['x_sigma_'+lab for lab in labels]
    if np.isinf(d[xcols].to_numpy(float)).any(): raise ValueError('infinite features')
    d['common'] = np.isfinite(d[xcols].to_numpy(float)).all(axis=1)
    assert_window(d,START,END)
    assert_window(d[d.common],START,END)
    return d,labels,base


def load_score(folder, kind, d, root, cfg):
    audit_path=folder/'run_audit.json'; fit_path=folder/'fit_audit.csv'; pred_path=folder/'predictions.csv'
    audit=json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if (audit.get('target_month_training_fixed')!=['2009-01','2012-12'] or
        audit.get('target_month_validation')!=[START,END] or audit.get('smoke') is not False or
        audit.get('test_2022_2025_used') is not False or audit.get('train_sample')!='common' or
        (audit.get('tree_count'),audit.get('depth'),audit.get('learning_rate'))!=(500,3,.01) or
        audit.get('arms',{}).get('fixed_4')!=['fixed',old.FOUR] or kind not in audit.get('models',[])):
        raise ValueError(f'Unexpected H2 prior experiment: {folder}')
    # Path separators differ on Windows; verify data bytes, not path spelling.
    saved={k.replace('\\','/'):v for k,v in audit['inputs_sha256'].items()}
    for key in ('panel','h2'):
        path=Path(cfg['PATHS'][key]); norm=path.as_posix().replace('\\','/')
        if saved.get(norm)!=digest(root/path):
            raise ValueError(f'Prior H2 model input hash differs: {path}; rerun H2 standalone before integration')
    f=pd.read_csv(fit_path,dtype=str)
    f=f.loc[f.model.eq(kind)&f.arm.eq('fixed_4')]
    if len(f)!=1: raise ValueError('Expected exactly one prior fixed_4 fit')
    r=f.iloc[0]
    if (r.train_actual_start,r.train_actual_end,r.eval_start,r.eval_end)!=('2009-01','2012-12',START,END) or int(r.trees)!=500 or int(r.train_months)!=48:
        raise ValueError('Prior H2 fit dates or tree count mismatch')
    pred=pd.read_csv(pred_path,dtype={'stock_id':str,'target_month':str},usecols=KEY+['model','arm','score','y'])
    pred=pred.loc[pred.model.eq(kind)&pred.arm.eq('fixed_4'),KEY+['score','y']].copy()
    pred['stock_id']=old.stock(pred.stock_id)
    if pred[KEY].isna().any().any() or pred.duplicated(KEY).any() or not np.isfinite(pred[['score','y']].to_numpy(float)).all():
        raise ValueError('Prior scores missing, duplicate or nonfinite')
    if set(pred.target_month)!=month_set(START,END): raise ValueError('Prior score months mismatch')
    # Do not silently drop any stock-month or fill absent H2 predictions.
    m=d[KEY+['y']].merge(pred,on=KEY,how='outer',suffixes=('','_prior'),indicator=True,validate='one_to_one',sort=False)
    if not m._merge.eq('both').all():
        raise ValueError(f'H1/H3 and H2 evaluation keys differ: {m._merge.value_counts().to_dict()}; inspect eligibility, do not silently intersect')
    if not np.allclose(m.y,m.y_prior,atol=1e-12,rtol=1e-10): raise ValueError('Prior score target mismatch')
    score=d[KEY].merge(pred[KEY+['score']],on=KEY,how='left',validate='one_to_one').score.to_numpy(float)
    return score, {'folder':str(folder),'prior_train_rows':int(r.train_rows),'prior_fit_end':r.train_actual_end,
                   'sha256':{x.name:digest(x) for x in (audit_path,fit_path,pred_path)}}


def cols_for(arm, span, base):
    common=['x_'+n for n in [*base,'size','bm','mom']]+['x_sigma_'+span]
    extras={'baseline_h1_h3':[], 'h2_8':['x_'+n for n in old.EIGHT],
            'h2_4':['x_'+n for n in old.FOUR], 'h2_score_0912':['h2_score']}
    return common+extras[arm]


def ic_table(d,p):
    from src.evaluation import model_test as mt
    ic=mt.monthly_ic(d.target_month.to_numpy(),np.asarray(p,float),d.y.to_numpy(float))
    ic.loc[ic.reason.eq('constant_score'),'ic']=0.0
    return ic


def run_kind(d,kind,labels,base,run,threads,device,policy,smoke):
    predictions=[]; audit=[]; candidates=[]
    last=FIRST_OUT if smoke else 2021
    for year in range(FIRST_OUT,last+1):
        # Same Y-2 fit / Y-1 inner / Y outer as team, no final refit.
        fit_end=f'{year-2}-12'; inner_a=f'{year-1}-01'; inner_b=f'{year-1}-12'
        out_a=f'{year}-01'; out_b=f'{year}-12'
        member=d.common if policy=='common' else pd.Series(True,index=d.index)
        tr=d.loc[member & d.target_month.between(START,fit_end)].copy()
        inn=d.loc[member & d.target_month.between(inner_a,inner_b)].copy()
        val=d.loc[d.target_month.between(out_a,out_b)].copy()
        assert_window(tr,START,fit_end); assert_window(inn,inner_a,inner_b); assert_window(val,out_a,out_b)
        if not tr.target_month.max()<inn.target_month.min() or not inn.target_month.max()<val.target_month.min(): raise ValueError('temporal leakage')
        # Baseline-only lambda selection; reuse chosen span in all four arms.
        # Predict inner+outer in one call; only inner outcomes determine selection.
        ev=pd.concat([inn,val],ignore_index=True)
        options=[]
        for span in labels:
            c=cols_for(ARMS[0],span,base)
            pr,pt,nt=old.fit_predict(tr,ev,c,kind,run,threads,device)
            ii=float(ic_table(inn,pr[:len(inn)]).ic.mean())
            if not np.isfinite(ii): raise ValueError('nonfinite inner IC')
            options.append((ii,int(span.removeprefix('span')),span,pr[len(inn):],pt,nt))
            candidates.append({'model':kind,'valid_year':year,'span':span,'inner_rankic':ii})
        best=max(options,key=lambda x:(x[0],x[1])); span=best[2]
        for arm in ARMS:
            c=cols_for(arm,span,base)
            if arm==ARMS[0]: pred,pt,nt=best[3:]
            else: pred,pt,nt=old.fit_predict(tr,val,c,kind,run,threads,device)
            if nt!=500: raise ValueError('not 500 trees')
            b=val[KEY+['y','common']].copy(); b['score']=pred; b['model']=kind;b['arm']=arm
            predictions.append(b)
            audit.append({'model':kind,'arm':arm,'valid_year':year,'train_start':START,'train_end':fit_end,
                          'inner_year':year-1,'train_rows':len(tr),'inner_rows':len(inn),'eval_rows':len(val),
                          'span':span,'baseline_inner_ic':best[0],'train_ic':ic_table(tr,pt).ic.mean(),
                          'trees':nt,'features':','.join(c)})
            print(f'{kind} | {year} | {arm} | fit {len(tr):,} | span {span} | 500 trees',flush=True)
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(audit),pd.DataFrame(candidates)


def evaluate(pred,rate,last):
    from src.h2.experiments.evaluate_h2_rate_hike_robustness import summarize
    monthly=[]
    for (kind,arm),p in pred.groupby(['model','arm']):
        monthly.append(old.monthly_metrics(p,rate).assign(model=kind,arm=arm))
    m=pd.concat(monthly,ignore_index=True)
    if m.duplicated(['model','arm','scope','target_month']).any(): raise ValueError('duplicate metric keys')
    periods=period_sets(last); summaries=[]; comparisons=[]
    for name,ms in periods.items():
        if not ms: continue
        for (kind,scope),g in m.loc[m.target_month.isin(ms)].groupby(['model','scope']):
            a=g[g.arm.eq(ARMS[0])].set_index('target_month').sort_index()
            if set(a.index)!=ms: raise ValueError(f'month coverage: {kind} {scope} {name}')
            for arm in ARMS:
                b=g[g.arm.eq(arm)].set_index('target_month').sort_index()
                if not b.index.equals(a.index) or not b.n.equals(a.n) or not np.allclose(a.universe_return,b.universe_return):
                    raise ValueError('unequal evaluation sample')
                if not np.isfinite(b.ic).all(): raise ValueError('nonfinite IC')
                summaries.append({'model':kind,'arm':arm,'scope':scope,'period':name,**summarize(b)})
                if arm==ARMS[0]: continue
                diff=b.ic-a.ic; ex=b.top20_minus_universe-a.top20_minus_universe; down=a.universe_return<0
                # HAC only on full consecutive sequence, descriptive subgroup differences.
                pval=np.nan
                if name=='all' and len(diff)>=24 and diff.std()>0:
                    import statsmodels.api as sm
                    pval=float(sm.OLS(diff.to_numpy(),np.ones((len(diff),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6}).pvalues[0])
                comparisons.append({'model':kind,'arm':arm,'reference':ARMS[0],'scope':scope,'period':name,'months':len(diff),
                                    'mean_delta_ic':diff.mean(),'ic_win_share':diff.gt(0).mean(),'delta_ic_hac6_p_two_sided':pval,
                                    'mean_delta_top20_excess':ex.mean(),'down_months':int(down.sum()),
                                    'mean_delta_down_top20_excess':ex[down].mean()})
    return m,pd.DataFrame(summaries),pd.DataFrame(comparisons)


def main():
    from src.evaluation import model_test as mt
    from src.h2.preprocessing.build_bok_rate_history import make
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--models',default='lgb_rankreg')
    p.add_argument('--lgb-run',type=Path,default=Path('outputs/model_results/h2/regime_lgb500'))
    p.add_argument('--xgb-run',type=Path,default=Path('outputs/model_results/h2/regime_xgb500'))
    p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/regime_integrated_v1'))
    p.add_argument('--train-sample',choices=['common','fill'],default='common')
    p.add_argument('--include-base',action='store_true');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu');p.add_argument('--check-only',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();root=a.root.resolve(); cfg_path=root/'config/experiment.yaml'
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'));mt.check_no_test(cfg)
    if cfg['UNITS']['return']!='decimal': raise ValueError('expected decimal returns')
    run=cfg['BASE_FEATURES']['model_input']['main_run']
    if (int(run['trees']),int(run['depth']),float(run['learning_rate']))!=(500,3,.01): raise ValueError('expected 500 trees, depth 3, lr .01')
    kinds=[s.strip() for s in a.models.split(',') if s.strip()]
    if not kinds or len(set(kinds))!=len(kinds) or not set(kinds)<=set(old.KINDS): raise ValueError('invalid model list')
    d,labels,base=prepare(root,cfg,a.include_base)
    sources={};score={}
    for kind in kinds:
        score[kind],sources[kind]=load_score(root/(a.lgb_run if kind=='lgb_rankreg' else a.xgb_run),kind,d,root,cfg)
    coverage=d.groupby('target_month').agg(eligible_rows=('y','size'),common_rows=('common','sum')).reset_index()
    print('H2 prior score provenance:',{k:v['prior_train_rows'] for k,v in sources.items()})
    print('Integrated 2013–2021 rows:',len(d),'common:',int(d.common.sum()),'minimum monthly common:',int(coverage.common_rows.min()))
    print('H3 candidates:',labels,'| base controls:',base,'| train sample:',a.train_sample)
    print('First outer 2016: fit 2013–2014, inner 2015; last outer 2021: fit 2013–2019, inner 2020')
    if a.check_only:return
    out=root/a.out
    if out.exists():raise FileExistsError(out)
    _,rates=make();rate=rates.set_index('feature_month').rate_change_pp
    pred=[];fit=[];cand=[]
    for kind in kinds:
        dd=d.assign(h2_score=score[kind]); pp,ff,cc=run_kind(dd,kind,labels,base,run,a.threads,a.xgb_device,a.train_sample,a.smoke)
        pred.append(pp);fit.append(ff);cand.append(cc)
    pred=pd.concat(pred,ignore_index=True);fit=pd.concat(fit,ignore_index=True);cand=pd.concat(cand,ignore_index=True)
    last='2016-12' if a.smoke else END
    m,s,c=evaluate(pred,rate,last)
    out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('predictions',pred),('monthly_rankic',m),('model_summary',s),('paired_comparisons',c),('fit_audit',fit),('h3_selection',cand),('monthly_coverage',coverage)]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    paths=[cfg_path,*[root/cfg['PATHS'][x] for x in ['panel','eligibility','h1_components','h2']+(['base_features'] if a.include_base else [])]]
    audit={'models':kinds,'arms':ARMS,'train_sample':a.train_sample,'base_controls':base,'validation':[f'{FIRST_OUT}-01',last],
           'h2_score_sources':sources,'tree_count':500,'depth':3,'learning_rate':.01,'smoke':a.smoke,'test_used':False,
           'h3_selection':'baseline inner IC at 500 trees, shared across arms, ties longer span; no refit',
           'periods':{k:sorted(v) for k,v in period_sets(last).items()},'input_sha256':{str(x):digest(x) for x in paths},
           'code_sha256':{str(x):digest(x) for x in [Path(__file__),Path(old.__file__),root/'src/evaluation/model_test.py',root/'src/models/boosting.py']},
           'git_commit':subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
           'return_note':'equal-weight eligible universe; gross returns; realized hike months ex-post only; not causal evidence'}
    (out/'run_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=['model','arm','period','months','mean_rankic','mean_top20_excess','down_market_top20_excess']
    print(s.loc[s.scope.eq('all_eligible')&s.period.isin(['all','hike_month','hike_cycle']),cols].to_string(index=False))
    print(c.loc[c.scope.eq('all_eligible')&c.period.isin(['all','hike_month','hike_cycle'])].to_string(index=False));print('Saved:',out)

if __name__=='__main__':main()
