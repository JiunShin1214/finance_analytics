"""Development-only H2 post-transform source restriction stress test.
Run at repository root. Reuses existing ex-ante universe, baseline scores, fit rows,
settings and 5% blend. Does NOT rebuild 9/12-month vintage datasets or certify PIT.
Original H2 expert must reproduce before any alternative is interpreted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['stock_id', 'base_month', 'target_month']
EIGHT = ['x_debt','x_policy','x_policy_change','x_policy_x_debt','x_debt_change',
         'x_rate_change','x_policy_change_x_debt','x_rate_change_x_debt']
LEVEL = ['x_debt','x_policy_x_debt','x_policy_change_x_debt','x_rate_change_x_debt']
DEBT = LEVEL + ['x_debt_change']
VARIANTS = ['dg_age9_mask','dg_age12_mask','unresolved_mask','no_debt']
MODELS = ['lgb_rankreg','xgb_rankreg','xgb_pairwise']
HYBRID = 'data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv'
ARCHIVE = 'data/processed/archive/h2_8features.parquet'
REFERENCE = 'outputs/model_results/h2/exante_universe_v2'
DG = 'DATAGUIDE_6M_PROXY'
DART = 'DART_FIRST_FILED_CANDIDATE'


def sha(p):
    with Path(p).open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def csv(p):
    return pd.read_csv(p,dtype={'stock_id':str,'base_month':str,'target_month':str},low_memory=False,encoding='utf-8-sig')


def need(d,cols,context):
    missing=set(cols)-set(d)
    if missing: raise ValueError(f'{context}: missing {sorted(missing)}; actual columns={list(d)}')


def stock(s):
    x=s.astype('string').str.strip().str.replace(r'^A(?=\d+$)','',regex=True).str.replace(r'\.0$','',regex=True).str.zfill(6)
    if not x.str.fullmatch(r'\d{6}',na=False).all(): raise ValueError('Invalid stock_id')
    return x


def flag(s):
    v=s.astype('string').str.strip().str.lower()
    if not v.isin(['true','false','1','0']).all(): raise ValueError('Invalid unresolved flag')
    return v.isin(['true','1'])


def field(d,explicit,options,context):
    if explicit:
        need(d,[explicit],context); return explicit
    hits=[x for x in options if x in d]
    if not hits: raise ValueError(f'{context}: cannot identify column; actual columns={list(d)}')
    # Prefer established names; do not silently accept conflicting aliases.
    if len(hits)>1 and any(not d[hits[0]].astype('string').fillna('').equals(d[x].astype('string').fillna('')) for x in hits[1:]):
        raise ValueError(f'{context}: ambiguous columns {hits}; use explicit CLI option')
    return hits[0]


def load_metadata(path,args):
    d=csv(path)
    m=field(d,args.month_col,['feature_month','base_month','month'],'hybrid month')
    f=field(d,args.fiscal_col,['source_fiscal_date','debt_source_fiscal_date','debt_final_fiscal_date'],'hybrid fiscal')
    if args.source_col:
        s=field(d,args.source_col,[],'hybrid source')
    else:
        # Exact source labels from the scope audit, not column-name guessing.
        hits=[c for c in d if d[c].astype('string').isin([DG,DART]).any()]
        if len(hits)!=1: raise ValueError(f'Cannot uniquely identify hybrid source column: {hits}; columns={list(d)}; use --source-col')
        s=hits[0]
    need(d,['stock_id','debt_final_raw'], 'hybrid')
    d=pd.DataFrame({'sid':stock(d.stock_id),'bm':d[m].astype('string').str.slice(0,7),
                    'source':d[s].astype('string'),'fiscal':pd.to_datetime(d[f],errors='coerce'),
                    'raw':pd.to_numeric(d.debt_final_raw,errors='coerce')})
    if not d.bm.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])',na=False).all() or not d.bm.between('2000-01','2021-12').all():
        raise PermissionError('Hybrid must contain only development months')
    if d.duplicated(['sid','bm']).any(): raise ValueError('Duplicate hybrid stock-month')
    valid=d.raw.notna()
    cutoff=pd.PeriodIndex(d.bm,freq='M').to_timestamp('M')
    if (valid & (~d.source.isin([DG,DART])|d.fiscal.isna()|d.fiscal.gt(cutoff))).any():
        raise ValueError('Observed raw debt has unsupported source or invalid fiscal date; inspect metadata')
    d=d.sort_values(['sid','bm']).reset_index(drop=True)
    # Most recent selected DataGuide fiscal month available up to this row.
    # Also guards DebtChange when a DART current report may depend on a DG predecessor.
    d['dg_seen']=d.fiscal.where(d.source.eq(DG)&d.raw.notna())
    d['dg_seen']=d.groupby('sid',sort=False).dg_seen.transform(lambda x:x.cummax().ffill())
    return d


def scope_file(folder,name):
    p=folder/name
    if p.is_file(): return p
    q=folder/(Path(name).stem+'(1)'+Path(name).suffix)
    if q.is_file(): return q
    raise FileNotFoundError(p)


def load_flags(folder,audit):
    rows=[]
    for name,label,count in [('dart',DART,audit['dart_report_candidates']),('dataguide',DG,audit['dataguide_report_candidates'])]:
        p=scope_file(folder,f'{name}_report_queue.csv');d=csv(p)
        need(d,['stock_id','source_type','source_fiscal_date','first_use_month','unresolved'],name)
        if len(d)!=count or not d.source_type.eq(label).all(): raise ValueError(f'{name} queue scope changed')
        v=d.loc[flag(d.unresolved)].copy()
        if name=='dart' and len(v)!=audit['dart_flagged_reports']: raise ValueError('DART flag count changed')
        rows.append(pd.DataFrame({'sid':stock(v.stock_id),'source':v.source_type,'fiscal':pd.to_datetime(v.source_fiscal_date,errors='raise'),'first':v.first_use_month.astype(str)}))
    d=pd.concat(rows,ignore_index=True)
    if not d['first'].str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all() or not d['first'].between('2000-01','2021-11').all(): raise ValueError('Flag month invalid')
    return d


def month_age(bm,date):
    # Month-end accounting convention: fiscal Dec + 9 months = next Sep month-end.
    b=pd.PeriodIndex(bm,freq='M')
    return pd.Series(b.year*12+b.month,index=date.index)-(date.dt.year*12+date.dt.month)


def masks(frame,meta,flags):
    x=pd.DataFrame({'sid':stock(frame.stock_id).to_numpy(),'bm':frame.base_month.astype(str).to_numpy()})
    x=x.merge(meta,on=['sid','bm'],how='left',validate='one_to_one',sort=False)
    if 'debt_available' not in frame: raise ValueError('Archive debt_available required for metadata check')
    observed=frame.debt_available.reset_index(drop=True).eq(True).fillna(False)
    if (observed & x.raw.isna()).any(): raise ValueError('Archive observed debt has no matching hybrid raw value')
    age=month_age(x.bm,x.fiscal); past_age=month_age(x.bm,x.dg_seen)
    known=x.source.isin([DG,DART])&x.raw.notna()
    if (known & age.lt(0)).any(): raise ValueError('Future fiscal date')
    # Report identity is conservatively stock + source type + fiscal date.
    # A flagged earlier report may be a carried DebtChange dependency.
    identities=flags[['sid','source','fiscal']].drop_duplicates().assign(flagged=True)
    cur=x.merge(identities,on=['sid','source','fiscal'],how='left',validate='many_to_one',sort=False).flagged.eq(True)
    first=flags.groupby('sid')['first'].min()
    ever=x.sid.map(first)
    historical=ever.notna()&x.bm.ge(ever.fillna('9999-12'))
    out={}
    for lag in [9,12]:
        level=known&x.source.eq(DG)&age.lt(lag)
        change=level|(past_age.notna()&past_age.lt(lag))
        out[f'dg_age{lag}_mask']=(level.to_numpy(bool),change.to_numpy(bool))
    out['unresolved_mask']=(cur.to_numpy(bool),(cur|historical).to_numpy(bool))
    out['no_debt']=(np.ones(len(x),dtype=bool),np.ones(len(x),dtype=bool))
    return out


def restrict(frame,pair,variant):
    out=frame.copy()
    if variant=='no_debt': return out,[c for c in EIGHT if c not in DEBT]
    lev,chg=pair
    out.loc[lev,LEVEL]=np.nan
    out.loc[chg,'x_debt_change']=np.nan
    return out,EIGHT.copy()


def exact_original(old,new,kind):
    a,b=np.asarray(old,float),np.asarray(new,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Original score invalid')
    if kind.startswith('xgb'):
        ok=np.array_equal(a.astype(np.float32).view(np.uint32),b.astype(np.float32).view(np.uint32))
    else: ok=np.allclose(a,b,atol=1e-8,rtol=0)
    if not ok: raise ValueError(f'{kind}: original H2 reproduction failed, max abs diff={np.max(np.abs(a-b))}; stop')
    return float(np.max(np.abs(a-b))) if len(a) else 0.


def fit_stress_variant(old, tx, vx, cols, kind, run, args, variant, year):
    """Original stays strict. Only alternative LGB fits may replay a native tree cap.

    The shared helper raises before returning the truncated model. Retry at its
    reported tree count with all other settings unchanged; this is a documented
    replay, not a silent relaxation or a claim of exact score reproduction.
    """
    requested = int(run['trees'])
    context = f'{kind} | {year} | {variant}'
    print(f'{context} | fitting, requested trees={requested}, '
          f'train_rows={len(tx)}, features={len(cols)}', flush=True)
    status = 'full_requested_trees'
    try:
        h, _, n = old.fit_predict(tx, vx[cols], cols, kind, run, args.threads, args.xgb_device)
    except RuntimeError as err:
        match = re.fullmatch(r'lgb_rankreg: actual trees=(\d+), requested=(\d+); no silent early termination', str(err))
        if (variant == 'original' or kind != 'lgb_rankreg' or match is None
                or int(match[2]) != requested or not 1 <= int(match[1]) < requested):
            raise RuntimeError(f'{context}: {err}') from err
        actual = int(match[1])
        retry = dict(run, trees=actual)
        print(f'{context} | native tree limit: {actual}/{requested}; '
              'replaying with reported tree cap, no other setting changed', flush=True)
        try:
            h, _, n = old.fit_predict(tx, vx[cols], cols, kind, retry, args.threads, args.xgb_device)
        except Exception as retry_err:
            raise RuntimeError(f'{context}: truncated-tree replay failed at {actual} trees: {retry_err}') from retry_err
        if n != actual:
            raise ValueError(f'{context}: replay count mismatch: {n} != {actual}')
        status = 'native_tree_limit_replay'
    if status == 'full_requested_trees' and n != requested:
        raise ValueError(f'{context}: unexpected tree count {n}/{requested}')
    h = np.asarray(h, dtype=float)
    if h.ndim != 1 or len(h) != len(vx) or not np.isfinite(h).all():
        raise ValueError(f'{context}: invalid expert forecasts')
    return h, n, status


def summaries(monthly):
    import statsmodels.api as sm
    rows=[]
    for (kind,variant,scope),d in monthly.groupby(['model','variant','scope'],sort=False):
        for period,h in [('all',d),('active_ready',d[d.active])]:
            h=h.sort_values('target_month');delta=h.variant_ic-h.baseline_ic; diff=h.variant_ic-h.original_ic
            p=lo=hi=np.nan
            if list(h.target_month)==list(pd.period_range('2006-01','2021-12',freq='M').astype(str)) and delta.std()>0:
                fit=sm.OLS(delta.to_numpy(float),np.ones((len(delta),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6})
                p=float(fit.pvalues[0]);lo,hi=map(float,fit.conf_int()[0])
            rows.append(dict(model=kind,variant=variant,scope=scope,period=period,months=len(h),active_months=int(h.active.sum()),baseline_ic=h.baseline_ic.mean(),original_ic=h.original_ic.mean(),variant_ic=h.variant_ic.mean(),delta_vs_baseline=delta.mean(),delta_vs_original=diff.mean(),active_win_months=int((h.active & delta.gt(0)).sum()),hac6_p_two_sided=p,hac6_ci_low=lo,hac6_ci_high=hi))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--scope-dir',type=Path)
    p.add_argument('--reference',type=Path,default=Path(REFERENCE));p.add_argument('--out',type=Path,default=Path('outputs/model_results/h2/pit_robustness_v1'))
    p.add_argument('--models',default=','.join(MODELS));p.add_argument('--variants',default=','.join(VARIANTS))
    p.add_argument('--threads',type=int,default=4);p.add_argument('--xgb-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--source-col');p.add_argument('--fiscal-col');p.add_argument('--month-col');p.add_argument('--check-only',action='store_true')
    a=p.parse_args();root=a.root.resolve();ref=root/a.reference;out=root/a.out
    if out.exists() and not a.check_only: raise FileExistsError(f'{out} exists; use another --out')
    if a.scope_dir is None:
        candidates=[x.parent for d in ['outputs','results'] for x in (root/d).rglob('pit_scope_audit.json')]
        if len(candidates)!=1: raise ValueError(f'Provide --scope-dir; found {candidates}')
        a.scope_dir=candidates[0]
    folder=root/a.scope_dir;scope=load_json(scope_file(folder,'pit_scope_audit.json'))
    if scope['test_used'] is not False or scope['model_fitting'] is not False: raise ValueError('Invalid PIT scope audit')
    # All audited inputs must remain identical; do not interpret a different vintage.
    hashes={k.replace('\\','/'):v for k,v in scope['inputs_sha256'].items()}
    for rel,h in hashes.items():
        path=root/rel
        if not path.is_file() or sha(path)!=h: raise ValueError(f'PIT audited input changed/missing: {rel}')
    for name in ['fit_audit.csv','run_audit.json','exante_active_predictions.csv']:
        if sha(ref/name)!=hashes.get(f'{REFERENCE}/{name}'): raise ValueError(f'Reference not PIT-audited: {name}')
    r=load_json(ref/'run_audit.json');kinds=a.models.split(',');variants=a.variants.split(',')
    if r.get('test_used') is not False or r.get('alpha')!=.05 or r.get('rule')!='hike_1m' or not set(kinds)<=set(MODELS)&set(r['models']) or len(set(kinds))!=len(kinds) or not set(variants)<=set(VARIANTS) or len(set(variants))!=len(variants): raise ValueError('Invalid reference/models/variants')
    from src.h2.experiments import evaluate_h2_exante_universe as exa
    import yaml
    rb,prev,spec,ind,old=exa.rb,exa.prev,exa.spec,exa.ind,exa.old
    if list(rb.KEY)!=KEY or list(rb.EIGHT)!=EIGHT: raise ValueError('Original feature/key definitions changed')
    a.h2_archive=Path(ARCHIVE);a.period_policy='full';a.sample_policy='fill';a.min_stocks=20;a.min_train_months=5;a.smoke=False
    cfg=yaml.safe_load((root/'config/experiment.yaml').read_text(encoding='utf-8'));run=cfg['BASE_FEATURES']['model_input']['main_run']
    if run!=r['settings'] or (run['trees'],run['depth'],run['learning_rate'],run['xgb_l2'])!=(500,3,.01,0): raise ValueError('Reference model settings changed')
    diag=load_json(root/'outputs/model_results/h2/overfit_diagnostic_v1/run_audit.json')
    if a.threads!=diag['threads'] or a.xgb_device!=diag['xgb_device']: raise ValueError('Use reference threads/device')
    for mod in [rb,prev,spec,ind,old]:
        spec.hash_match(diag['code_sha256'],root,'src/h2/experiments/'+Path(mod.__file__).name)
    d,base,full,labels,folds,_=prev.prepare(root,cfg,a)
    exa.keys(d)
    if len(base)!=30 or len(full)!=34 or base!=diag['baseline_features']: raise ValueError('Baseline feature drift')
    from src.h2.preprocessing.build_bok_rate_history import make
    _,rates=make();g=ind.strict_gate(d,rates)
    z=d.merge(g[['target_month',exa.G]],on='target_month',how='left',validate='many_to_one');z[exa.G]=z[exa.G].eq(True)
    z,archive=rb.archive_frame(root,cfg,z,a.h2_archive)
    if sha(archive)!=r['archive_sha256']: raise ValueError('Archive changed')
    active=sorted(r['active_target_months']);ex,arc=exa.prepare_exante(root,cfg,a,active,base,labels)
    if arc!=archive or len(ex)!=scope['prediction_rows']: raise ValueError('Reference prediction scope changed')
    fit=csv(ref/'fit_audit.csv');pred=csv(ref/'exante_active_predictions.csv');monthly_ref=csv(ref/'monthly_comparison.csv')
    need(fit,['model','valid_year','h2_train_rows','h2_train_months','h2_train_start','h2_train_end','eval_rows','eval_month_list'],'fit reference')
    need(pred,KEY+['model','h2_score','without_h2','final_score'],'score reference')
    need(monthly_ref,['model','scope','target_month','active','n','baseline_ic','new_combined_ic'],'monthly reference')
    meta=load_metadata(root/HYBRID,a);flags=load_flags(folder,scope)
    coverage=[];fits=[];scores=[];allmonthly=[];reproduction=[]
    print('Active target months:',active,'| rows:',len(ex),'| flagged reports:',len(flags),flush=True)
    for kind in kinds:
        prior,priorfit,source=spec.load_prior(root,Path(f'outputs/model_results/h2/conditional_full_{kind}_v1'),kind,cfg,d,base,full,labels,folds,a)
        if source['sha256']!=r['baseline_runs'][kind]['sha256']: raise ValueError('Baseline source mismatch')
        prior=prior.drop(columns='common').merge(z[KEY+['common']],on=KEY,how='left',validate='one_to_one',sort=False)
        if prior.common.isna().any(): raise ValueError('Missing reference common flag')
        pr=pred[pred.model.eq(kind)].copy();exa.same_keys(pr,ex,'Reference ex-ante scores')
        alltr=[];parts={v:[] for v in ['original',*variants]}
        for f in fit[fit.model.eq(kind)].sort_values('valid_year').itertuples(index=False):
            tr=z.loc[z.fit_member & z[exa.G] & z.target_month.between(f.h2_train_start,f.h2_train_end)].sort_values(['target_month','stock_id']).reset_index(drop=True)
            va=ex[ex.target_month.isin(f.eval_month_list.split(';'))].sort_values(['target_month','stock_id']).reset_index(drop=True)
            if len(tr)!=f.h2_train_rows or tr.target_month.nunique()!=f.h2_train_months or len(va)!=f.eval_rows or tr.target_month.max()>=va.target_month.min() or not np.isfinite(tr.y).all(): raise ValueError('Fit reference rows/time mismatch')
            alltr.append(tr[KEY]);tm,vm=masks(tr,meta,flags),masks(va,meta,flags)
            frozen=va[KEY].merge(pr[KEY+['without_h2','h2_score','final_score']],on=KEY,how='left',validate='one_to_one',sort=False)
            if not np.isfinite(frozen[['without_h2','h2_score','final_score']].to_numpy(float)).all(): raise ValueError('Missing frozen score')
            for variant in ['original',*variants]:
                if variant=='original':
                    tx,vx,cols=tr,va,EIGHT
                else:
                    tx,cols=restrict(tr,tm[variant],variant)
                    vx,vcols=restrict(va,vm[variant],variant)
                    if cols!=vcols: raise ValueError('Train/predict feature mismatch')
                # Keep exactly the old fit members. Missing values are passed to native tree handling.
                if variant!='original':
                    for name,fr,ms in [('train',tr,tm),('predict',va,vm)]:
                        lev,chg=ms[variant]
                        coverage.append(dict(model=kind,valid_year=f.valid_year,variant=variant,scope=name,rows=len(fr),level_mask_rows=int(lev.sum()),change_mask_rows=int(chg.sum()),level_observed_mask_rows=int((lev&fr.debt_available.eq(True).fillna(False).to_numpy(bool)).sum()),change_observed_mask_rows=int((chg&fr.debt_change_available.eq(True).fillna(False).to_numpy(bool)).sum())))
                if a.check_only: continue
                h,n,fit_status=fit_stress_variant(old,tx,vx,cols,kind,run,a,variant,f.valid_year)
                tied_months=int(pd.DataFrame({'target_month':vx.target_month.to_numpy(),'score':h}).groupby('target_month').score.nunique().eq(1).sum())
                if variant=='original':
                    diff=exact_original(frozen.h2_score,h,kind);reproduction.append(dict(model=kind,valid_year=f.valid_year,max_abs_diff=diff))
                    # Use the actual frozen reference after reproduction succeeds.
                    h=frozen.h2_score.to_numpy(float)
                blended=exa.blend_scores(frozen[KEY+['without_h2']].assign(h2_score=h))
                if variant=='original': exa.close(blended.final_score,frozen.final_score,'Frozen blend')
                parts[variant].append(blended.assign(model=kind,variant=variant))
                fits.append(dict(model=kind,variant=variant,valid_year=f.valid_year,train_rows=len(tx),train_months=tx.target_month.nunique(),eval_rows=len(vx),trees=n,requested_trees=int(run['trees']),fit_status=fit_status,constant_score_months=tied_months,features=','.join(cols)))
                print(kind,f.valid_year,variant,'done | trees:',n,'| status:',fit_status,'| tied months:',tied_months,flush=True)
        if len(pd.concat(alltr).drop_duplicates())!=scope['unique_train_rows']: raise ValueError('Unique fit scope mismatch')
        if a.check_only: continue
        # Freeze all alternative scores before realized outcome joins for evaluation.
        mr=monthly_ref[monthly_ref.model.eq(kind)].copy();mr['active']=flag(mr.active)
        if set(mr.loc[mr.active,'target_month'])!=set(active): raise ValueError('Active month mismatch')
        for variant in ['without_h2','original',*variants]:
            m=mr[['model','scope','target_month','active','n','baseline_ic','new_combined_ic']].rename(columns={'new_combined_ic':'original_ic'}).copy()
            m['variant']=variant;m['variant_ic']=m.baseline_ic
            if variant!='without_h2':
                sc=pd.concat(parts[variant],ignore_index=True);exa.same_keys(sc,ex,'Variant forecast coverage');scores.append(sc)
                pp=prior[prior.target_month.isin(active)].copy();pp[KEY]=pp[KEY].astype(str)
                pp=pp.merge(sc[KEY+['final_score']],on=KEY,how='left',validate='one_to_one',sort=False)
                if pp.final_score.isna().any(): raise ValueError('Missing variant evaluation score')
                for i,row in m.loc[m.active].iterrows():
                    q=pp[pp.target_month.eq(row.target_month)]
                    if row.scope=='complete_case': q=q[q.common]
                    elif row.scope!='all_stocks': raise ValueError('Unknown reference scope')
                    if len(q)!=row.n: raise ValueError('Evaluation sample changed')
                    b=ind.rankic(q.without_h2,q.y);v=ind.rankic(q.final_score,q.y)
                    exa.close([b],[row.baseline_ic],'Baseline IC')
                    if variant=='original': exa.close([v],[row.original_ic],'Original IC')
                    m.loc[i,'variant_ic']=v
            if not np.isfinite(m[['baseline_ic','original_ic','variant_ic']].to_numpy(float)).all(): raise ValueError('Nonfinite IC')
            allmonthly.append(m)
    cov=pd.DataFrame(coverage)
    print(cov.to_string(index=False),flush=True)
    if a.check_only:
        print('CHECK-ONLY passed: source schema, hashes, scope, masks; no model fitting.');return
    monthly=pd.concat(allmonthly,ignore_index=True);summary=summaries(monthly)
    out.mkdir(parents=True,exist_ok=False)
    for name,t in {'mask_coverage':cov,'fit_audit':pd.DataFrame(fits),'original_reproduction':pd.DataFrame(reproduction),'monthly_rankic':monthly,'robustness_summary':summary,'active_predictions':pd.concat(scores,ignore_index=True)}.items():t.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    audit=dict(native_tree_limit_replays=sum(x['fit_status']=='native_tree_limit_replay' for x in fits),native_tree_limit_policy='Original and XGB strict; alternative LGB only: replay at reported actual tree cap, record requested/actual counts and constant-score months; no other settings changed.',test_used=False,submission_certified=False,pit_certified=False,model_fitting=True,baseline_refit=False,models=kinds,variants=variants,alpha=.05,settings=run,active_months=active,reference=str(ref.relative_to(root)),scope_audit_sha256=sha(scope_file(folder,'pit_scope_audit.json')),queue_sha256={n:sha(scope_file(folder,n)) for n in ['dart_report_queue.csv','dataguide_report_queue.csv']},monthly_reference_sha256=sha(ref/'monthly_comparison.csv'),input_sha256=hashes,script_sha256=sha(__file__),notes='Retrospective post-transform masking; not a delayed-source reconstruction, not reranked peer inputs, not a source vintage audit. Unresolved source fiscal identity ignores receipt and change mask conservatively retains any earlier flagged dependency. No tuning; alternative LGB native tree-limit replay is a documented exception to the 500-tree target. No production scoring.')
    (out/'run_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.to_string(index=False));print('Saved:',out)

if __name__=='__main__':main()
