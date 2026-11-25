"""H2-only fixed 2009-2012 training vs expanding, 4 vs 8 features.
Target-month split; no 2022-2025 access. Run at repository root.
Uses team's boosting parameters / RankIC / BOK history functions; no global changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FOUR = ['debt', 'policy', 'policy_change', 'policy_x_debt']
EIGHT = FOUR + ['debt_change', 'rate_change', 'policy_change_x_debt', 'rate_change_x_debt']
ARMS = {'fixed_4': ('fixed', FOUR), 'fixed_8': ('fixed', EIGHT),
        'expanding_4': ('expanding', FOUR), 'expanding_8': ('expanding', EIGHT)}
FIXED_START, FIXED_END = '2009-01', '2012-12'
EVAL_START, EVAL_END = '2013-01', '2021-12'
KINDS = ('lgb_rankreg', 'xgb_rankreg', 'xgb_pairwise')
ARCHIVE_PATH = Path('data/processed/archive/h2_8features.parquet')


def resolve_archive(root, path=ARCHIVE_PATH):
    # Never fall back to the production 4-feature file.
    root = Path(root).resolve()
    actual = (root / path).resolve()
    if not actual.is_relative_to(root):
        raise ValueError('H2 archive must be inside repository root')
    if not actual.is_file():
        raise FileNotFoundError(f'Missing historical 8-feature H2 archive: {actual}; restore original bytes, not regenerated data')
    return actual



def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stock(s):
    return s.astype('string').str.strip().str.replace(r'^A(?=\d{6}$)', '', regex=True).str.replace(r'\.0$', '', regex=True).str.zfill(6)


def validate_keys(df, key, name):
    if df[key].isna().any().any() or df.duplicated(key).any():
        raise ValueError(f'{name}: missing or duplicate keys')
    month = key[-1]
    if not df[month].str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', na=False).all():
        raise ValueError(f'{name}: invalid month')


def prepare_frames(panel, h2):
    p, h = panel.copy(), h2.copy()
    p['stock_id'], h['stock_id'] = stock(p.stock_id), stock(h.stock_id)
    p['month'], h['base_month'] = p.month.astype('string'), h.base_month.astype('string')
    validate_keys(p, ['stock_id', 'month'], 'panel')
    validate_keys(h, ['stock_id', 'base_month'], 'h2')
    if not p.month.between('2000-01', '2021-12').all() or not h.base_month.between('2001-01', '2021-11').all():
        raise PermissionError('development months only')
    h[EIGHT] = h[EIGHT].apply(pd.to_numeric, errors='raise').astype(float)
    if np.isinf(h[EIGHT].to_numpy()).any():
        raise ValueError('infinite H2 input')
    # Use eligibility at t, never target-month eligibility to decide universe.
    base = p.loc[p.eligible.eq(True).fillna(False) & p.month.between('2000-12','2021-11'), ['stock_id','month']].rename(columns={'month':'base_month'})
    base['target_month'] = (pd.PeriodIndex(base.base_month, freq='M') + 1).astype(str)
    d = base.merge(h[['stock_id','base_month'] + EIGHT], on=['stock_id','base_month'], how='left', validate='one_to_one')
    # Cross-sectional transformation before using target observations, as in team feature_specs.
    v = d.debt_change
    g = v.groupby(d.base_month)
    d['debt_change'] = 2 * g.rank(method='average') / (g.transform('count') + 1) - 1
    y = p[['stock_id','month','return']].rename(columns={'month':'target_month','return':'y'})
    d = d.merge(y, on=['stock_id','target_month'], how='left', validate='one_to_one')
    d['y'] = pd.to_numeric(d.y, errors='raise').astype(float)
    d = d[np.isfinite(d.y)].sort_values(['target_month','stock_id']).reset_index(drop=True)
    d['common'] = d[EIGHT].notna().all(axis=1)
    return d


def train_rows(d, start, end, policy):
    m = d.target_month.between(start, end)
    return d.loc[m & d.common if policy == 'common' else m].copy()


def rank_target(df, kind):
    g = df.groupby('target_month').y
    r = g.rank(method='average')
    n = g.transform('size')
    if kind == 'xgb_pairwise':
        return np.floor(10 * (r - 1) / n).astype(int).to_numpy()
    return (r / (n + 1)).to_numpy(float)


def check_window(df, start, end, min_stocks=20):
    counts = df.groupby('target_month').size()
    expected = pd.period_range(start, end, freq='M').astype(str)
    bad = counts.reindex(expected, fill_value=0)
    bad = bad[bad < min_stocks]
    if len(bad):
        raise ValueError(f'Insufficient monthly rows ({start}~{end}): {bad.to_dict()}')


def fit_predict(train, val, features, kind, run, threads, device):
    from src.models import boosting
    n = int(run['trees'])
    if n != 500:
        raise ValueError('fixed 500 trees required')
    X = train[features].fillna(0).to_numpy(float)
    Xv = val[features].fillna(0).to_numpy(float)
    y = rank_target(train, kind)
    loss = 'pairwise' if kind == 'xgb_pairwise' else 'mse'
    depth, lr = int(run['depth']), float(run['learning_rate'])
    if kind == 'lgb_rankreg':
        import lightgbm as lgb
        params = boosting.lgb_params(kind, loss, depth, lr, device='cpu', n_threads=threads)
        booster = lgb.train(params, lgb.Dataset(X, label=y, feature_name=features), num_boost_round=n)
        actual = booster.num_trees()
        pred = booster.predict(Xv, num_iteration=n)
        fitted = booster.predict(X, num_iteration=n)
    else:
        import xgboost as xgb
        ov = {'lambda': float(run['xgb_l2'])}
        if kind == 'xgb_pairwise':
            ov.update(run['pairwise_pairs'])
        params = boosting.xgb_params(kind, loss, depth, lr, device=device, n_threads=threads, overrides=ov)
        qid = pd.factorize(train.target_month, sort=True)[0] if kind == 'xgb_pairwise' else None
        dm = xgb.DMatrix(X, label=y, qid=qid, feature_names=features)
        booster = xgb.train(params, dm, num_boost_round=n)
        actual = booster.num_boosted_rounds()
        pred = booster.predict(xgb.DMatrix(Xv, feature_names=features), iteration_range=(0,n))
        fitted = booster.predict(dm, iteration_range=(0,n))
    if actual != n:
        raise RuntimeError(f'{kind}: actual trees={actual}, requested={n}; no silent early termination')
    if not np.isfinite(pred).all() or not np.isfinite(fitted).all():
        raise ValueError('nonfinite model predictions')
    return pred, fitted, actual


def weighted_tail(score, y, top=True, fraction=.2):
    # Fractional allocation at boundary ties: no arbitrary stock-code tiebreak.
    score = pd.Series(np.asarray(score,float))
    rank = score.rank(method='min', ascending=not top).to_numpy(float) - 1
    size = score.map(score.value_counts()).to_numpy(float)
    w = np.clip((len(score)*fraction - rank) / size, 0, 1)
    return float(np.dot(w, y) / w.sum())


def monthly_metrics(pred, rate):
    from src.evaluation import model_test as mt
    rows = []
    for scope, subset in [('all_eligible',pred), ('complete_case',pred[pred.common])]:
        for m,g in subset.groupby('target_month', sort=True):
            if len(g) < 3: continue
            y, p = g.y.to_numpy(float), g.score.to_numpy(float)
            t = mt.monthly_ic(g.target_month.to_numpy(), p, y).iloc[0]
            ic = 0.0 if t.reason == 'constant_score' else float(t.ic)
            top, bottom = weighted_tail(p,y), weighted_tail(p,y,False)
            rate_change = rate.loc[m]
            rows.append({'scope':scope,'target_month':m,'n':len(g),'ic':ic,
                         'rate_change_pp_target':float(rate_change),'regime':'hike' if rate_change>0 else ('cut' if rate_change<0 else 'hold'),
                         'top20_return':top,'bottom20_return':bottom,'universe_return':float(y.mean()),
                         'top20_minus_universe':top-y.mean(),'long_short':top-bottom,
                         'ic_reason':str(t.reason)})
    return pd.DataFrame(rows)


def run_experiments(d, rate, kinds, run, policy, threads, device, smoke):
    from src.evaluation import model_test as mt
    pred_list, results, fits = [], [], []
    val = d[d.target_month.between(EVAL_START, '2013-12' if smoke else EVAL_END)].copy()
    for kind in kinds:
        for arm,(mode,features) in ARMS.items():
            groups = [(None,val)] if mode=='fixed' else list(val.groupby(val.target_month.str[:4],sort=True))
            blocks=[]
            for year,g in groups:
                start = FIXED_START if mode=='fixed' else '2001-01'
                end = FIXED_END if mode=='fixed' else f'{int(year)-1}-12'
                train = train_rows(d,start,end,policy)
                if mode=='fixed': check_window(train,start,end)
                if len(train)<100 or train.target_month.nunique()<12:
                    raise ValueError(f'{arm}: insufficient train rows')
                if train.target_month.max() >= g.target_month.min(): raise ValueError('look-ahead in training')
                p,pt,n = fit_predict(train,g,features,kind,run,threads,device)
                ti = mt.monthly_ic(train.target_month.to_numpy(),pt,train.y.to_numpy(float))
                ti.loc[ti.reason.eq('constant_score'),'ic']=0.
                fits.append({'model':kind,'arm':arm,'train_requested_start':start,'train_requested_end':end,
                             'train_actual_start':train.target_month.min(),'train_actual_end':train.target_month.max(),
                             'train_rows':len(train),'train_months':train.target_month.nunique(),'train_ic':ti.ic.mean(),
                             'eval_start':g.target_month.min(),'eval_end':g.target_month.max(),'trees':n})
                b=g[['stock_id','target_month','y','common']].copy();b['score']=p;blocks.append(b)
                print(f'{kind} | {arm} | train {start}~{end}: {len(train):,} | eval {g.target_month.min()}~{g.target_month.max()} | {n} trees',flush=True)
            pred=pd.concat(blocks,ignore_index=True).sort_values(['target_month','stock_id'])
            ref=val[['stock_id','target_month']].reset_index(drop=True)
            if not pred[['stock_id','target_month']].reset_index(drop=True).equals(ref): raise ValueError('evaluation keys differ')
            results.append(monthly_metrics(pred,rate).assign(model=kind,arm=arm))
            pred_list.append(pred.assign(model=kind,arm=arm))
    return pd.concat(results,ignore_index=True), pd.concat(pred_list,ignore_index=True), pd.DataFrame(fits)


def summarize(monthly):
    rows=[]
    for (model,arm,scope),g in monthly.groupby(['model','arm','scope'],sort=True):
        for regime,h in [('all',g), *list(g.groupby('regime'))]:
            v=h.ic; sd=v.std(ddof=1)
            rows.append({'model':model,'arm':arm,'scope':scope,'regime':regime,'months':len(h),'valid_ic_months':v.notna().sum(),
                         'mean_rankic':v.mean(),'icir':v.mean()/sd if sd>0 else np.nan,
                         'positive_ic_share':(v.dropna()>0).mean(),'mean_top20_return':h.top20_return.mean(),
                         'mean_universe_return':h.universe_return.mean(),'mean_top20_excess':h.top20_minus_universe.mean(),
                         'mean_long_short':h.long_short.mean(),'top20_negative_month_share':(h.top20_return<0).mean()})
    return pd.DataFrame(rows)


def paired(monthly):
    pairs=[('fixed_4','fixed_8'),('fixed_4','expanding_4'),('fixed_8','expanding_8'),('expanding_4','expanding_8')]
    rows=[]
    for (model,scope),g in monthly.groupby(['model','scope']):
        for a,b in pairs:
            x=g[g.arm.eq(a)].merge(g[g.arm.eq(b)],on=['target_month','regime'],suffixes=('_a','_b'),validate='one_to_one')
            for regime,h in [('all',x),*list(x.groupby('regime'))]:
                z=h.ic_a-h.ic_b
                rows.append({'model':model,'scope':scope,'regime':regime,'arm':a,'reference':b,'months':len(h),
                             'paired_valid_ic_months':z.notna().sum(),'mean_delta_ic':z.mean(),
                             'mean_delta_top20_excess':(h.top20_minus_universe_a-h.top20_minus_universe_b).mean()})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--models',default='lgb_rankreg')
    p.add_argument('--h2-archive',type=Path,default=ARCHIVE_PATH,help='Historical 8-feature parquet, relative to repository root')
    p.add_argument('--train-sample',choices=['common','fill'],default='common',help='common: same 8-feature complete rows for all arms; fill: team zero fill all eligible')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--xgb-device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--check-only',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--out',type=Path)
    a=p.parse_args(); root=a.root.resolve()
    from src.evaluation import model_test as mt
    from src.h2.preprocessing.build_bok_rate_history import make
    cfg_path=root/'config/experiment.yaml';cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    mt.check_no_test(cfg)
    if cfg['UNITS']['return']!='decimal': raise ValueError('expected decimal returns')
    run=cfg['BASE_FEATURES']['model_input']['main_run']
    if (int(run['trees']),int(run['depth']),float(run['learning_rate']))!=(500,3,.01): raise ValueError('expected team main_run: 500 trees, depth 3, lr .01')
    kinds=[x.strip() for x in a.models.split(',') if x.strip()]
    if not kinds or len(kinds)!=len(set(kinds)) or not set(kinds)<=set(KINDS): raise ValueError('invalid model list')
    panel_path=root/cfg['PATHS']['panel'];h2_path=resolve_archive(root,a.h2_archive)
    print('Historical H2 archive:',h2_path,'| SHA256:',sha(h2_path))
    d=prepare_frames(pd.read_parquet(panel_path,columns=['stock_id','month','return','eligible']),pd.read_parquet(h2_path,columns=['stock_id','base_month']+EIGHT))
    _,rates=make();rate=rates.set_index('feature_month').rate_change_pp
    fixed=train_rows(d,FIXED_START,FIXED_END,a.train_sample)
    check_window(fixed,FIXED_START,FIXED_END)
    check_window(d[d.target_month.between(EVAL_START,EVAL_END)],EVAL_START,EVAL_END)
    coverage=d.groupby('target_month').agg(eligible_with_return=('y','size'),complete_8=('common','sum')).join(d.groupby('target_month')[EIGHT].count()).reset_index()
    print('Fixed train:',len(fixed),'rows;',fixed.target_month.nunique(),'months; validation:',EVAL_START,'~',EVAL_END)
    print('Validation rate hike target months:',rate.loc[EVAL_START:EVAL_END].loc[lambda s:s>0].to_dict())
    print(coverage[coverage.target_month.between(FIXED_START,FIXED_END)].to_string(index=False))
    if a.check_only: return
    out=(root/a.out if a.out else root/'outputs/model_results/h2'/('regime_fixed_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    if out.exists(): raise FileExistsError(out)
    monthly,preds,fits=run_experiments(d,rate,kinds,run,a.train_sample,a.threads,a.xgb_device,a.smoke)
    summary=summarize(monthly);comparison=paired(monthly)
    out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('model_summary',summary),('monthly_rankic',monthly),('paired_comparisons',comparison),('fit_audit',fits),('monthly_coverage',coverage),('predictions',preds)]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    audit={'target_month_training_fixed':[FIXED_START,FIXED_END],'target_month_validation':[EVAL_START,'2013-12' if a.smoke else EVAL_END],
           'models':kinds,'arms':ARMS,'train_sample':a.train_sample,'tree_count':500,'depth':3,'learning_rate':.01,
           'test_2022_2025_used':False,'smoke':a.smoke,'rate_hike_grouping':'realized target-month rate change; ex-post analysis only',
           'portfolio':'equal-weight top/bottom quintile; fractional boundary ties; gross returns; no costs',
           'evaluation':'same eligible t-month stocks with next-month realized return; 8-feature complete-case secondary',
           'inputs_sha256':{str(x.relative_to(root)):sha(x) for x in [panel_path,h2_path,cfg_path]},
           'h2_archive_path':h2_path.relative_to(root).as_posix(),
           'code_sha256':{str(x.relative_to(root)):sha(x) for x in [Path(__file__).resolve(),root/'src/models/boosting.py',root/'src/evaluation/model_test.py',root/'src/h2/preprocessing/build_bok_rate_history.py']},
           'git_commit':subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()}
    (out/'run_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.loc[summary.scope.eq('all_eligible') & summary.regime.isin(['all','hike'])].to_string(index=False));print('Saved:',out)

if __name__=='__main__': main()
