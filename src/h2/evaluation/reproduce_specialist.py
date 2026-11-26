"""Verify canonical H2 + new modules against frozen exante_universe_v2.
Default is read-only check, no fitting. --execute refits H2 only, not baseline.
Development only; no submission scores. Run from repository root as a module.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.h2.models import specialist as s, regime as rg, blend as bl

KEY = bl.KEY
FEATURES = list(s.FEATURES)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Historical fingerprint mismatch: {path}')


def config_check(h, cfg):
    expected = dict(enabled_for_submission=False, features=list(s.RAW_FEATURES),
                    rank_transform=['debt_change'], input_transform='rank', fill_missing=True,
                    gate=rg.RULE, training='annual_expanding', min_train_months=rg.MIN_TRAIN_MONTHS,
                    min_stocks=rg.MIN_STOCKS, alpha=.05, preferred_candidate='xgb_pairwise',
                    models=list(s.KINDS), run=s.frozen_run(), development_target_start='2006-01',
                    development_target_end='2021-12')
    for k,v in expected.items():
        if h.get(k) != v: raise ValueError(f'Frozen H2 config drift: {k}')
    if h['feature_path'] != cfg['PATHS']['h2']:
        raise ValueError('Canonical H2 path differs from shared config')
    run=cfg['BASE_FEATURES']['model_input']['main_run']
    if any(run.get(k)!=v for k,v in s.frozen_run().items()):
        raise ValueError('Shared model settings differ from frozen candidate')
    return run


def keys(d):
    q=d[KEY].copy()
    if q.isna().any().any(): raise ValueError('Null key')
    q[KEY]=q[KEY].astype(str)
    bl._validate(q,KEY,'keys')
    if not q.target_month.between('2001-01','2021-12').all():
        raise PermissionError('Development data only')
    return q


def normalize(d):
    q=d.copy(); q[KEY]=keys(q); return q


def compare_scores(expected, actual, kind):
    bl._same_keys(expected, actual, 'Reproduction score universe')
    q=expected.merge(actual,on=KEY,suffixes=('_old','_new'),validate='one_to_one')
    a=q.h2_score_old.to_numpy(float); b=q.h2_score_new.to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite H2 scores')
    if kind.startswith('xgb'):
        if not np.array_equal(a.astype(np.float32).view(np.uint32),b.astype(np.float32).view(np.uint32)):
            raise ValueError('Native XGB score mismatch')
    elif not np.allclose(a,b,rtol=0,atol=1e-8): raise ValueError('H2 raw score mismatch')
    return float(np.max(np.abs(a-b)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default='xgb_pairwise');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu')
    mode=p.add_mutually_exclusive_group(); mode.add_argument('--check-only',action='store_true');mode.add_argument('--execute',action='store_true')
    p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/modular_reproduction_v1'))
    a=p.parse_args();root=Path.cwd();a.root=root
    if a.execute and (root/a.out).exists(): raise FileExistsError(a.out)
    cfg=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8-sig'))
    h=yaml.safe_load((root/'config/h2.yaml').read_text(encoding='utf-8-sig'))['h2_specialist']
    run=config_check(h,cfg); kinds=a.models.split(',')
    if len(kinds)!=len(set(kinds)) or not set(kinds)<=set(s.KINDS): raise ValueError('Invalid models')
    for path,digest in h['audited_inputs_sha256'].items(): verify(root/path,digest)
    ref=root/h['reference']
    for name,digest in h['reference_sha256'].items(): verify(ref/name,digest)
    r=json.loads((ref/'run_audit.json').read_text(encoding='utf-8-sig'))
    if r['test_used'] is not False or r['rule']!=rg.RULE or r['alpha']!=h['alpha'] or r['settings']!=run or not set(kinds)<=set(r['models']):
        raise ValueError('Reference settings mismatch')
    diag_path=root/'outputs/model_results/h2/overfit_diagnostic_v1/run_audit.json'
    verify(diag_path,r['reference_diagnostic_sha256']['run_audit.json'])
    diag=json.loads(diag_path.read_text(encoding='utf-8-sig'))
    if a.threads!=diag['threads'] or a.xgb_device!=diag['xgb_device']: raise ValueError('Use historical threads/device')
    from src.h2.experiments import evaluate_h2_exante_universe as exa
    from src.h2.preprocessing.build_bok_rate_history import make
    # Reuse historical shared preparation; canonical parquet replaces only the
    # old archive path. No call to load_prior, whose old H2 hash is intentionally stale.
    if list(exa.EIGHT)!=FEATURES: raise ValueError('Legacy feature order drift')
    a.h2_archive=Path(h['feature_path']);a.period_policy='full';a.sample_policy='fill';a.min_stocks=20;a.min_train_months=5;a.smoke=False
    for mod in [exa.rb,exa.prev,exa.spec,exa.ind,exa.old]:
        exa.spec.hash_match(diag['code_sha256'],root,'src/h2/experiments/'+Path(mod.__file__).name)
    d,base,full,labels,folds,_=exa.prev.prepare(root,cfg,a)
    d=normalize(d)
    if len(base)!=30 or len(full)!=34 or base!=diag['baseline_features']: raise ValueError('Common model feature drift')
    z,path=exa.rb.archive_frame(root,cfg,d,a.h2_archive)
    if path.resolve()!=(root/h['feature_path']).resolve(): raise ValueError('Incorrect H2 source')
    z=normalize(z)
    if z.fit_member.isna().any() or not z.fit_member.isin([True,False]).all(): raise ValueError('Invalid fit membership')
    z['fit_member']=z.fit_member.astype(bool)
    _,rates=make();rates=rates[['feature_month','rate_change_pp']].copy()
    months=list(pd.period_range(h['development_target_start'],h['development_target_end'],freq='M').astype(str))
    gate=rg.build_regime_gate(rates,sorted(z.target_month.unique()))
    hikes=set(gate.loc[gate[rg.RULE],'target_month'])
    plans=[]; histories={}
    for f in folds:
        year=int(f['valid_year']);lo,hi=f['train_years'];start,end=f'{lo}-01',f'{hi}-12'
        # Shared model_test.plan_folds reserves Y-1 for inner validation.
        # Keep the historical fit cutoff; do not refit through Y-1.
        expected_end=s.annual_train_end(year)
        if end!=expected_end or int(f.get('inner_valid_year',year-1))!=year-1:
            raise ValueError(f'Unexpected expanding fold: year={year}, fit_end={end}, expected={expected_end}, inner={f.get("inner_valid_year")}')
        tr=z.loc[z.target_month.between(start,end),KEY+FEATURES+['y','fit_member']].copy()
        chosen=tr.loc[tr.fit_member & tr.target_month.isin(hikes)]
        counts=chosen.groupby('target_month').size()
        plans.append(dict(valid_year=year,train_end=end,train_months=len(counts),min_month_stocks=int(counts.min()) if len(counts) else 0,ready=bool(len(counts)>=rg.MIN_TRAIN_MONTHS and counts.ge(rg.MIN_STOCKS).all())))
        histories[year]=(start,tr,chosen)
    routing=rg.route_months(rates,months,pd.DataFrame(plans))
    active=sorted(rg.active_target_months(routing))
    if active!=sorted(r['active_target_months']): raise ValueError(f'Active months drift: {active}')
    ex,path=exa.prepare_exante(root,cfg,a,active,base,labels);ex=normalize(ex)
    if path.resolve()!=(root/h['feature_path']).resolve(): raise ValueError('Incorrect forecast H2 source')
    elig=pd.read_parquet(root/cfg['PATHS']['eligibility'],columns=['stock_id','month','eligible'])
    if not elig.month.between('2000-01','2021-12').all(): raise PermissionError('Eligibility scope')
    e=elig.loc[elig.eligible.eq(True),['stock_id','month']].rename(columns={'month':'base_month'}).copy()
    e['target_month']=(pd.PeriodIndex(e.base_month,freq='M')+1).astype(str)
    e=keys(e.loc[e.target_month.isin(active)])
    bl._same_keys(ex,e,'Forecast vs independent eligibility')
    if len(ex)!=r['exante_rows_per_model']: raise ValueError('Reference row count drift')
    fit=pd.read_csv(ref/'fit_audit.csv',encoding='utf-8-sig');pred=pd.read_csv(ref/'exante_active_predictions.csv',dtype={k:str for k in KEY},encoding='utf-8-sig')
    records=[];output=[]
    print('Active months:',active,'| rows per model:',len(ex),flush=True)
    for kind in kinds:
        expected=pred.loc[pred.model.eq(kind),KEY+['without_h2','h2_score','final_score']].copy()
        bl._same_keys(expected,e,'Frozen full ex-ante universe')
        for year in sorted({int(m[:4]) for m in active}):
            start,tr,chosen=histories[year]
            am=[m for m in active if m.startswith(str(year)+'-')]
            va=ex.loc[ex.target_month.isin(am),KEY+FEATURES].copy();ek=e.loc[e.target_month.isin(am)]
            rf=fit.loc[fit.model.eq(kind)&fit.valid_year.eq(year)]
            if len(rf)!=1: raise ValueError('Missing historical fold')
            f=rf.iloc[0]
            if (len(chosen)!=f.h2_train_rows or chosen.target_month.nunique()!=f.h2_train_months or chosen.target_month.min()!=f.h2_train_start or chosen.target_month.max()!=f.h2_train_end or len(va)!=f.eval_rows or ';'.join(am)!=f.eval_month_list or f.trees!=500): raise ValueError(f'Fit scope drift: {kind} {year}')
            b=expected.loc[expected.target_month.isin(am)].copy()
            check=bl.blend_conditional(b[KEY+['without_h2']],b[KEY+['h2_score']],ek,am)
            q=b.merge(check[KEY+['final_score']],on=KEY,suffixes=('_old','_new'),validate='one_to_one')
            if not np.allclose(q.final_score_old,q.final_score_new,atol=1e-12,rtol=0): raise ValueError('Frozen blend mismatch')
            rec=dict(model=kind,valid_year=year,train_window_start=start,train_window_end=s.annual_train_end(year),inner_valid_year=year-1,train_rows=len(chosen),train_months=chosen.target_month.nunique(),forecast_rows=len(va),active_target_months=';'.join(am),model_fitted=False)
            if a.execute:
                scores,audit=s.predict_annual_fold(tr,va,ek,rates,valid_year=year,train_start=start,kind=kind,threads=a.threads,device='cpu' if kind=='lgb_rankreg' else a.xgb_device)
                if audit['active_target_months']!=am or audit['train_rows']!=len(chosen) or audit['trees']!=500 or audit['train_end']!=s.annual_train_end(year): raise ValueError('Modular fit audit drift')
                rec['max_raw_abs_diff']=compare_scores(b[KEY+['h2_score']],scores,kind)
                v=bl.blend_conditional(b[KEY+['without_h2']],scores,ek,am)
                q=b.merge(v[KEY+['final_score']],on=KEY,suffixes=('_old','_new'),validate='one_to_one')
                rec['max_blend_abs_diff']=float((q.final_score_old-q.final_score_new).abs().max())
                if rec['max_blend_abs_diff']>1e-12: raise ValueError('Recomputed blend mismatch')
                rec['model_fitted']=True;output.append(v.assign(model=kind,valid_year=year))
            records.append(rec);print(rec,flush=True)
    if not a.execute:
        print('CHECK-ONLY passed: fingerprints, routing, fit scope, universe, frozen blend; no fitting.');return
    out=root/a.out;out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(records).to_csv(out/'reproduction.csv',index=False,encoding='utf-8-sig')
    pd.concat(output,ignore_index=True).to_csv(out/'active_predictions.csv',index=False,encoding='utf-8-sig')
    routing.to_csv(out/'routing.csv',index=False,encoding='utf-8-sig')
    (out/'run_audit.json').write_text(json.dumps(dict(test_used=False,submission_certified=False,pit_certified=False,baseline_refit=False,models=kinds,active_months=active,reference=str(h['reference']),reference_sha256=h['reference_sha256'],input_sha256=h['audited_inputs_sha256'],code_sha256={str(Path(m.__file__).relative_to(root)):sha(m.__file__) for m in (s,rg,bl)},config_sha256=sha(root/'config/h2.yaml'),script_sha256=sha(__file__)),indent=2),encoding='utf-8')
    print('REPRODUCTION passed:',out)

if __name__=='__main__': main()
