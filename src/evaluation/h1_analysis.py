"""승인 범위의 H1 특징·주성분 진단과 기술적 순위 상관 비교. 평균 검정은 별도."""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from src.contracts import H1_MONTHLY_IC_COLUMNS
from src.features.h1 import calculate_h1_score, winsorize_1_99
from src.preprocessing.panel import attach_target_month, join_training_targets
from src.evaluation.rank_ic import rank_ic_for_month

FEATURES = ('size', 'bm', 'mom')
LABELS = {'size':'-규모', 'bm':'가치', 'mom':'모멘텀', 'equal':'기존 동일가중', 'winsor':'극단값 제한 동일가중'}
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']


def check_pc1_ic_approval(config):
    s=config['H1_PCA']; h=config['H1_TEST']
    rule='first_largest_abs_positive_then_previous_same_rank'
    if (s.get('pc1_ic_approved') is not True or s.get('pc1_ic_variants') not in (['raw'],['raw','winsor'])
        or h.get('pca_components')!=['PC1'] or h.get('pca_sign_rule')!=rule
        or s.get('display_sign_rule')!=rule):
        raise PermissionError('pc1_existing_sign_ic_approval_required')
    if s['run_category']!='h1_pca반영지수':
        raise ValueError('user_run_category_required')


def append_report(path, marker, text):
    """보고서는 출력이다. 이전 파일이나 노션 연결 없이 새로 생성할 수 있다."""
    path.parent.mkdir(parents=True,exist_ok=True)
    previous=path.read_text(encoding='utf-8') if path.is_file() else ''
    path.write_text(previous.split(marker)[0]+text,encoding='utf-8')


def run_pca_pc1_ic(config, *, root=Path('.'), preconditions, persist=True):
    """공통 특징에서 계수를 계산·적용한 뒤 목표를 결합해 같은 표본으로 비교한다."""
    from src.contracts import H1_PCA_SCORE_COLUMNS
    from src.evaluation.rank_ic import mean_rank_ic_on_common_months
    check_pc1_ic_approval(config)
    started=datetime.now(); paths=config['PATHS']; settings=config['H1_PCA']
    c=pd.read_parquet(root/paths['h1_components'])
    h=pd.read_parquet(root/paths['h1'])
    weights=pca_structure_analysis(c,h)[0]['pca_structure_monthly']
    variants=settings['pc1_ic_variants']
    weights=weights[weights.variant.isin(variants) & (weights.component=='PC1')]
    h=pd.read_parquet(root/paths['h1'])
    sources={'raw':h}
    if 'winsor' in variants:
        c=pd.read_parquet(root/paths['h1_components'])
        sources['winsor']=calculate_h1_score(winsorize_1_99(c))
        pd.testing.assert_frame_equal(h[['stock_id','base_month','eligible','h1_available']],
            sources['winsor'][['stock_id','base_month','eligible','h1_available']])
    rows=[]; score_parts=[]
    names=[method+'_pc1'+('_winsor' if variant=='winsor' else '')
           for variant in variants for method in ('pca_a','pca_b')]
    for variant,source in sources.items():
        cumulative_sum=np.zeros(3); cumulative_n=0
        for month,part in source[source.base_month.between('2000-12','2021-11')].groupby('base_month',sort=True):
            x=part[['z_size','z_bm','z_mom']].to_numpy(float,na_value=np.nan).copy(); x[:,0]*=-1
            member=np.isfinite(x).all(axis=1)
            cumulative_sum+=x[member].sum(axis=0); cumulative_n+=int(member.sum())
            for method in ('pca_a','pca_b'):
                w=weights[(weights.base_month==month) & (weights.method==method) & (weights.variant==variant)]
                if len(w)!=1:raise ValueError('pc1_weight_keys')
                w=w.iloc[0]; score=np.full(len(part),np.nan)
                if w.available:
                    if w.fit_last_base_month>month:raise ValueError('future_pc1_fit')
                    if int(w.current_n)!=int(member.sum()):raise ValueError('pc1_members_changed')
                    vector=w[['weight_neg_size','weight_bm','weight_mom']].to_numpy(float)
                    np.testing.assert_allclose(vector@vector,1,atol=1e-12)
                    center=x[member].mean(axis=0) if method=='pca_a' else cumulative_sum/cumulative_n
                    score[member]=(x[member]-center)@vector
                    if method=='pca_a':
                        np.testing.assert_allclose(np.var(score[member],ddof=1),w.eigenvalue,atol=1e-11)
                f=part[['stock_id','base_month','eligible']].copy()
                f['candidate']=method+'_pc1'+('_winsor' if variant=='winsor' else ''); f['score']=score
                f['score_available']=np.isfinite(score)
                f['score_exclusion_reason']=np.where(f.score_available,'',part.h1_exclusion_reason)
                if (f.loc[~f.score_available,'score_exclusion_reason']=='').any():
                    raise ValueError('missing_pc1_exclusion_reason')
                score_parts.append(f)
    scores=pd.concat(score_parts,ignore_index=True)
    # Warmup 행도 유지하되 점수는 계산하지 않는다.
    for name in names:
        f=h[h.base_month<'2000-12'][['stock_id','base_month','eligible']].copy()
        f['candidate']=name;f['score']=np.nan;f['score_available']=False
        f['score_exclusion_reason']='warmup_history_only';score_parts.append(f)
    saved_scores=pd.concat(score_parts,ignore_index=True).sort_values(['candidate','stock_id','base_month'])
    saved_scores=saved_scores[list(H1_PCA_SCORE_COLUMNS)].reset_index(drop=True)
    if saved_scores.duplicated(['candidate','stock_id','base_month']).any():raise ValueError('duplicate_pc1_scores')
    panel=pd.read_parquet(root/paths['panel']);panel.attrs['units']=config['UNITS']
    base=h[h.base_month.between('2000-12','2021-11')].copy()
    wide=base[['stock_id','base_month','eligible','h1']].rename(columns={'h1':'equal'})
    ids=['equal']
    if 'winsor' in sources:
        w=sources['winsor'][['stock_id','base_month','h1']].rename(columns={'h1':'equal_winsor'})
        wide=wide.merge(w,on=['stock_id','base_month'],validate='one_to_one');ids.append('equal_winsor')
    for name in names:
        f=scores[scores.candidate==name]
        wide=wide.merge(f[['stock_id','base_month','score']].rename(columns={'score':name}),
                        on=['stock_id','base_month'],validate='one_to_one')
    ids+=names
    availability=np.isfinite(wide[ids].to_numpy(float))
    if not (availability==availability[:,[0]]).all():raise ValueError('pc1_samples_differ')
    # 점수를 완성한 뒤 정확한 다음 달력월 수익률을 결합한다.
    joined=join_training_targets(wide,panel)
    for month,part in joined.groupby('base_month',sort=True):
        mask=np.isfinite(part[ids].to_numpy(float)).all(axis=1)
        for name in ids:
            f=part[['stock_id','target_month','y']].copy()
            f['score']=part[name].where(mask);f['candidate']=name
            f['model']='feature_score';f['fold']='development'
            row=rank_ic_for_month(f,score_column='score').iloc[0].to_dict()
            rows.append({**row,'base_month':month,'sample_scope':'common_h1'})
    monthly=pd.DataFrame(rows,columns=H1_MONTHLY_IC_COLUMNS)
    if monthly.groupby('target_month').n.nunique().max()!=1:raise ValueError('ic_pair_samples_differ')
    summary=mean_rank_ic_on_common_months(monthly,comparison_ids=ids)
    eq=summary.loc[summary.candidate=='equal','mean_ic'].iloc[0]
    summary['mean_delta_vs_equal']=summary.mean_ic-eq
    summary['mean_delta_vs_same_method_raw']=np.nan
    if 'winsor' in variants:
        for name,raw_name in [('equal_winsor','equal'),('pca_a_pc1_winsor','pca_a_pc1'),('pca_b_pc1_winsor','pca_b_pc1')]:
            summary.loc[summary.candidate==name,'mean_delta_vs_same_method_raw']=(
                summary.loc[summary.candidate==name,'mean_ic'].iloc[0]-summary.loc[summary.candidate==raw_name,'mean_ic'].iloc[0])
    yearly=pd.concat([mean_rank_ic_on_common_months(f,comparison_ids=ids).assign(target_year=year)
        for year,f in monthly.groupby(monthly.target_month.str[:4])],ignore_index=True)
    checks={'same_feature_sample':True,'same_ic_pairs':True,'targets_attached_after_scores':True,
            'future_fit_excluded':True,'pca_a_projected_variance_matches_eigenvalue':True,
            'shared_input_contract_checked':True,'all_feature_rows_retained':len(saved_scores)==len(names)*len(h)}
    if not all(checks.values()):raise ValueError('pc1_ic_checks_failed')
    if not persist:return {'monthly':monthly,'scores':saved_scores,'weights':weights}
    folder=root/paths['run_records']/settings['run_category']/started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    tables={'monthly_ic':monthly,'candidate_compare':summary,'ic_summary_by_target_year':yearly}
    for name,f in tables.items():f.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    saved_scores.to_parquet(folder/'pc1_scores.parquet',index=False)
    inputs=[paths['h1'],paths['h1_components'],paths['panel']]
    code=['src/evaluation/h1_analysis.py','src/evaluation/rank_ic.py','src/preprocessing/panel.py',
          'src/features/h1.py','src/contracts.py','src/pipeline.py','config/experiment.yaml']
    record={'status':'EXECUTED','stage':'pc1_monthly_ic','command':' '.join(sys.argv),
        'approved_decisions':{'components':['PC1'],'variants':variants,'sign':config['H1_TEST']['pca_sign_rule'],
            'aggregation':'equal_weight_common_valid_months','hac':False,'candidate_selection':None,'figures':False},
        'preconditions':preconditions,'input_sha256':{p:sha256(root/p) for p in inputs},
        'code_sha256':{p:sha256(root/p) for p in code},'checks':checks,
        'table_rows':{k:len(v) for k,v in tables.items()},'score_rows':len(saved_scores),
        'common_score_cells':int(availability[:,0].sum()),
        'common_return_pairs':int(monthly[monthly.candidate=='equal'].n.sum()),
        'method':'raw or monthly common raw features clipped at linear q01/q99 then restandardized ddof=1; center signed monthly Z on current mean (A) or expanding row mean (B), project saved unit PC1; calendar t+1; average tied-rank Spearman; unavailable months NaN; target return unchanged',
        'figures':[],'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir()},
        'deferred_checks':{'related_and_full_tests':'large category end','temporary_reexecution':'large category end'}}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    note=root/config['H1_TEST']['notion_tables_path'];marker='\n\n## 원값 PC1 월별 순위 상관 비교'
    text=marker+' · 원값/극단값 제한 확장\n\n상태 EXECUTED. 사용자 승인: PC1만, 원값 및 월별 공통 집합 원특징의 1·99 분위수 제한 후 재표준화, 기존 계수 방향 규칙 유지. 원값·제한값 각각의 PCA 계수를 적용한다. 기존 H1과 같은 종목·목표월에서 비교하며 유효 월을 동일가중한다. 목표수익률은 제한하지 않는다.\n\n'
    text+=markdown_table(summary)+'\n\n실행: `'+folder.relative_to(root).as_posix()+'`. 월별 표 `monthly_ic.csv`, 연도별 표 `ic_summary_by_target_year.csv`, 점수 `pc1_scores.parquet`, 근거 `run.json`.\n\n'
    text+='그림 없음: 단순 수치 비교는 표로 제공. 기존 후속 그림 목록도 자동 생성 약속이 아니며 변동·구조 파악이 필요한 경우에만 작성한다. 평균 검정·후보 선정은 미실행. 관련/전체 테스트와 임시 재실행은 큰 파트 종료 시 수행한다.\n'
    append_report(note,marker,text)
    return {'run_dir':str(folder),'comparison':summary.to_dict(orient='records'),'checks':checks}


def check_constrained_approval(config):
    s=config['H1_PCA']
    if s.get('constrained_exploration_approved') is not True or s.get('constrained_variants')!=['raw','winsor']:
        raise PermissionError('constrained_exploration_approval_required')
    if s['run_category']!='h1_pca반영지수':raise ValueError('user_run_category_required')


def run_constrained_pca(config, *, root=Path('.'), preconditions, persist=True):
    """기존 결과를 고정한 별도 사후 탐색. 비음수 최대분산 점수 4개만 추가."""
    from src.features.h1 import constrained_pca
    from src.contracts import H1_PCA_SCORE_COLUMNS
    check_constrained_approval(config)
    started=datetime.now();s=config['H1_PCA'];paths=config['PATHS']
    ordinary_data=run_pca_pc1_ic(config,root=root,preconditions=preconditions,persist=False)
    old_monthly=ordinary_data['monthly']
    ordinary=ordinary_data['weights']
    c=pd.read_parquet(root/paths['h1_components']);h=pd.read_parquet(root/paths['h1'])
    sources={'raw':h,'winsor':calculate_h1_score(winsorize_1_99(c))}
    fit_parts=[];score_parts=[]
    for variant,source in sources.items():
        pd.testing.assert_frame_equal(h[['stock_id','base_month','eligible','h1_available']],
                                      source[['stock_id','base_month','eligible','h1_available']])
        analysis=source[source.base_month.between('2000-12','2021-11')]
        for method in ('pca_a','pca_b'):
            name='constrained '+method.replace('_','-')+(' (winsor)' if variant=='winsor' else '')
            fit=constrained_pca(analysis,method=method)
            if fit.loc[fit.available,'optimal_solutions'].gt(1).any():
                raise RuntimeError('multiple_constrained_optima_require_user_decision:'+name)
            ref=ordinary[(ordinary.variant==variant)&(ordinary.method==method)]
            comp=fit.merge(ref[['base_month','explained_variance_ratio','current_n','fit_n']],
                           on='base_month',validate='one_to_one',suffixes=('','_ref'))
            if len(comp)!=252 or not comp.current_n.eq(comp.current_n_ref).all() or not comp.fit_n.eq(comp.fit_n_ref).all():
                raise ValueError('constrained_fit_sample_differs_from_ordinary')
            np.testing.assert_allclose(comp.ordinary_pc1_variance_ratio,comp.explained_variance_ratio_ref,
                                       atol=1e-12,rtol=1e-12,equal_nan=True)
            fit['variant']=variant;fit['candidate']=name;fit_parts.append(fit)
            indexed=fit.set_index('base_month')
            for month,part in source.groupby('base_month',sort=True):
                f=part[['stock_id','base_month','eligible']].copy();f['candidate']=name
                f['score']=np.nan;f['score_available']=False
                f['score_exclusion_reason']='warmup_history_only' if month<'2000-12' else part.h1_exclusion_reason
                if month>='2000-12':
                    row=indexed.loc[month]
                    if row.available:
                        x=part[['z_size','z_bm','z_mom']].to_numpy(float,na_value=np.nan).copy();x[:,0]*=-1
                        valid=np.isfinite(x).all(axis=1)
                        center=row[['center_neg_size','center_bm','center_mom']].to_numpy(float)
                        w=row[['weight_neg_size','weight_bm','weight_mom']].to_numpy(float)
                        f.loc[valid,'score']=(x[valid]-center)@w
                        f['score_available']=np.isfinite(f.score)
                        f.loc[f.score_available,'score_exclusion_reason']=''
                        if method=='pca_a':
                            np.testing.assert_allclose(f.loc[valid,'score'].var(ddof=1),row.projected_variance,atol=1e-11)
                if (f.loc[~f.score_available,'score_exclusion_reason']=='').any():raise ValueError('constrained_missing_reason')
                score_parts.append(f)
    fits=pd.concat(fit_parts,ignore_index=True)
    scores=pd.concat(score_parts,ignore_index=True)[list(H1_PCA_SCORE_COLUMNS)]
    if len(scores)!=4*len(h) or scores.duplicated(['candidate','stock_id','base_month']).any():
        raise ValueError('constrained_score_rows')
    # 목표 관측과 무관하게 특징 표본을 먼저 대조한다.
    for name,f in scores.groupby('candidate',sort=False):
        merged=f.merge(h[['stock_id','base_month','h1_available']],on=['stock_id','base_month'],validate='one_to_one')
        if not merged.score_available.eq(merged.h1_available & merged.base_month.ge('2000-12')).all():
            raise ValueError('constrained_feature_sample_changed')
    panel=pd.read_parquet(root/paths['panel']);panel.attrs['units']=config['UNITS']
    rows=[]
    for name,f in scores[scores.base_month.ge('2000-12')].groupby('candidate',sort=False):
        joined=join_training_targets(f,panel)
        for month,part in joined.groupby('base_month',sort=True):
            part=part.assign(model='feature_score',fold='development')
            row=rank_ic_for_month(part,score_column='score').iloc[0].to_dict()
            rows.append({**row,'base_month':month,'sample_scope':'common_h1'})
    monthly=pd.concat([old_monthly,pd.DataFrame(rows,columns=H1_MONTHLY_IC_COLUMNS)],ignore_index=True)
    if monthly.groupby('target_month').n.nunique().max()!=1:raise ValueError('constrained_return_pair_sample_changed')
    wide=monthly.pivot(index='target_month',columns='candidate',values='ic')
    common=np.isfinite(wide).all(axis=1)
    if int(common.sum())!=251:raise ValueError('constrained_common_month_count_changed')
    baseline=monthly[monthly.candidate.eq('equal')][['target_month','ic']].rename(columns={'ic':'ic_equal'})
    delta=monthly.merge(baseline,on='target_month',validate='many_to_one')
    delta['delta_ic']=delta.ic-delta.ic_equal
    summaries=[]
    for name,f in monthly.groupby('candidate',sort=False):
        values=wide.loc[common,name]
        summaries.append({'candidate':name,**metrics(values),'excluded_months':int((~common).sum()),
                         'mean_delta_ic':float((values-wide.loc[common,'equal']).mean()),
                         'exploratory':name.startswith('constrained ')})
    summary=pd.DataFrame(summaries)
    var_summary=[];weight_summary=[]
    for (variant,method,name),f in fits.groupby(['variant','method','candidate'],sort=False):
        valid=f[f.available]
        var_summary.append({'variant':variant,'method':method,'candidate':name,'valid_months':len(valid),
            'ordinary_pc1_mean_ratio':valid.ordinary_pc1_variance_ratio.mean(),
            'constrained_mean_ratio':valid.explained_variance_ratio.mean(),
            'mean_ratio_delta':valid.variance_ratio_delta.mean(),
            'constrained_min_ratio':valid.explained_variance_ratio.min(),
            'constrained_max_ratio':valid.explained_variance_ratio.max()})
        for feature in ('neg_size','bm','mom'):
            w=valid['weight_'+feature]
            weight_summary.append({'candidate':name,'feature':feature,'valid_months':len(w),
                'weight_mean':w.mean(),'weight_sd':w.std(ddof=1),'weight_min':w.min(),'weight_max':w.max(),
                'zero_months':int(w.eq(0).sum()),'positive_months':int(w.gt(0).sum())})
    checks={'ordinary_results_from_same_inputs':True,'nonnegative_unit_weights_and_optimality':True,
        'all_7_supports_considered':True,'no_ambiguous_global_optima':True,'same_feature_and_target_pairs':True,
        'ordinary_fit_variance_matches_current_inputs':True,'no_future_fit_rows':bool(
            fits.loc[fits.available,'fit_last_base_month'].le(fits.loc[fits.available,'base_month']).all()),
        'valid_months_251':True,'target_pairs_165331':int(monthly[monthly.candidate.eq('equal')].n.sum())==165331}
    if not all(checks.values()):raise ValueError('constrained_checks_failed')
    tables={'monthly_ic':monthly,'candidate_compare':summary,'monthly_delta_ic':delta,
            'constrained_weights_monthly':fits,'variance_compare':pd.DataFrame(var_summary),
            'weight_summary':pd.DataFrame(weight_summary)}
    if not persist:return {'tables':tables,'scores':scores,'ordinary_scores':ordinary_data['scores']}
    folder=root/paths['run_records']/s['run_category']/started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,f in tables.items():f.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    scores.to_parquet(folder/'constrained_scores.parquet',index=False)
    inputs=[paths['h1_components'],paths['h1'],paths['panel']]
    code=['src/features/h1.py','src/evaluation/h1_analysis.py','src/evaluation/rank_ic.py',
          'src/preprocessing/panel.py','src/pipeline.py','src/contracts.py','config/experiment.yaml']
    record={'status':'EXECUTED','stage':'constrained_pca_exploration','command':' '.join(sys.argv),
        'started':started.isoformat(),'preconditions':preconditions,'checks':checks,
        'approved_decisions':{'exploratory_after_prior_results':True,'variants':['raw','winsor'],
            'constraints':'w>=0, L2 norm=1','objective':'maximize w.T @ centered_sample_covariance @ w',
            'methods':'A current cross-section; B expanding rows through current month',
            'hac':False,'final_selection':None,'figures':False},
        'solver':'enumerate all 7 nonempty supports and their strictly positive signed eigenvectors; compare actual Rayleigh objectives; certify active stationarity and inactive one-sided derivatives; reject multiple distinct optima before IC',
        'variance_ratio_denominator':'trace of the same centered fit covariance; all weights unit L2',
        'zero_definition':'exact zero for coordinates outside chosen support; no posthoc clipping of ordinary PC1',
        'icir_definition':'mean monthly IC / sample SD ddof=1; not annualized',
        'input_sha256':{p:sha256(root/p) for p in inputs},'code_sha256':{p:sha256(root/p) for p in code},
        'table_rows':{k:len(v) for k,v in tables.items()},'score_rows':len(scores),
        'common_valid_months':251,'common_return_pairs':165331,'figures':[],
        'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir()},
        'deferred_checks':{'related_and_full_tests':'large part end','temporary_reexecution':'large part end'}}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    note=root/config['H1_TEST']['notion_tables_path'];marker='\n\n## 부호 제약 PCA · 사후 탐색 실험'
    text=marker+'\n\n기존 결과를 확인한 뒤 추가한 탐색 실험이며 사전 주검정·최종 지수 선정이 아니다. 원값·월별 1·99 분위수 제한 후 재표준화를 모두 비교. 특징 방향 [-Z(Size), Z(B/M), Z(Mom)] 유지, 목표수익률 제한 없음.\n\n'
    text+='방법: 매월 A는 당월, B는 해당 월까지의 누적 표본 공분산에서 w≥0, ‖w‖₂=1 제약의 최대분산을 7개 지지집합 열거로 직접 계산했다. 기존 PC1의 절댓값·0 절단을 사용하지 않았다. 분산비율은 투영 분산/추정 공분산의 대각합. 제외된 좌표의 가중치만 정확히 0이며 합계 1 제약은 없다.\n\n'
    text+='### 성능 비교\n\n'+markdown_table(summary)+'\n\n### 설명분산 비교\n\n'+markdown_table(tables['variance_compare'])
    text+='\n\n### 가중치와 0인 달의 수\n\n'+markdown_table(tables['weight_summary'])
    text+='\n\n실행 `'+folder.relative_to(root).as_posix()+'`. 월별 가중치 constrained_weights_monthly.csv, 월별 순위 상관 monthly_ic.csv, 월별 ΔIC monthly_delta_ic.csv, 점수 constrained_scores.parquet, 실행 근거 run.json. 같은 251개월·165,331쌍, 첫 목표월은 미산출. ICIR은 월평균/월별 표본표준편차이며 연율화하지 않는다.\n\n'
    text+='그림 없음. 필수 내장 검사 통과, 별도 테스트·임시 재실행은 큰 파트 종료 시 보류. 후보 선정·HAC 검정 미실행. 같은 자료로 결과를 본 뒤 추가했으므로 비교 결과는 탐색적으로만 해석한다.\n'
    append_report(note,marker,text)
    return {'run_dir':str(folder),'comparison':summary.to_dict(orient='records'),
            'variance':tables['variance_compare'].to_dict(orient='records'),
            'weights':tables['weight_summary'].to_dict(orient='records'),'checks':checks}


HYPOTHESIS_CANDIDATES = (
    'equal','equal_winsor','pca_a_pc1','pca_b_pc1','pca_a_pc1_winsor','pca_b_pc1_winsor',
    'constrained pca-a','constrained pca-b','constrained pca-a (winsor)','constrained pca-b (winsor)',
)
HYPOTHESIS_LABELS = dict(zip(HYPOTHESIS_CANDIDATES,(
    '기존 동일가중 H1','동일가중 H1 · 극단값 제한','일반 PCA-A PC1 · 원값','일반 PCA-B PC1 · 원값',
    '일반 PCA-A PC1 · 극단값 제한','일반 PCA-B PC1 · 극단값 제한',
    'constrained pca-a · 원값','constrained pca-b · 원값',
    'constrained pca-a · 극단값 제한','constrained pca-b · 극단값 제한')))


def check_hypothesis_inputs(config, *, root=Path('.')):
    settings=config.get('H1_HYPOTHESIS',{})
    expected={'approved':True,'run_category':'h1가설검정','alpha':.05,'reference_distribution':'t',
        'df':'T-1','lag_rule':'floor(4*(T/100)^(2/9))','kernel':'Bartlett',
        'small_sample_correction':False,'holm_family':'nine_paired_differences_only','selected_candidate':None}
    if any(settings.get(k)!=v for k,v in expected.items()) or not config['H1_TEST'].get('hac_approved'):
        raise PermissionError('fixed_hypothesis_scope_approval_required')
    if not config['H1_PCA'].get('development_closed'):raise PermissionError('development_must_be_closed')
    from src.pipeline import check_features_current
    return check_features_current(config,root=root)


def hypothesis_tables(monthly):
    """확정된 10개 월별 시계열에만 평균/대응차이 검정을 적용한다."""
    from src.evaluation.rank_ic import newey_west_mean_test,holm_bonferroni
    f=monthly.copy()
    if set(f.candidate)!=set(HYPOTHESIS_CANDIDATES):raise ValueError('fixed_ten_candidates_required')
    if f.duplicated(['candidate','target_month']).any():raise ValueError('duplicate_hypothesis_months')
    if not f.target_month.between('2001-01','2021-12').all():raise PermissionError('hypothesis_development_only')
    if not ((pd.PeriodIndex(f.base_month,freq='M')+1).astype(str)==f.target_month.to_numpy()).all():
        raise ValueError('hypothesis_calendar_alignment')
    excluded=f[f.target_month.eq('2001-01')].copy()
    if len(excluded)!=10 or excluded.ic.notna().any() or not excluded.n.eq(0).all():
        raise ValueError('first_target_month_exclusion_changed')
    f=f[~f.target_month.eq('2001-01')].sort_values(['candidate','target_month']).reset_index(drop=True)
    if len(f)!=2510 or not np.isfinite(f.ic).all() or not f.reason.eq('').all():
        raise ValueError('hypothesis_missing_or_invalid_month')
    wide=f.pivot(index='target_month',columns='candidate',values='ic').reindex(columns=HYPOTHESIS_CANDIDATES)
    expected=pd.period_range('2001-02','2021-12',freq='M').astype(str)
    if wide.index.tolist()!=expected.tolist():raise ValueError('hypothesis_nonconsecutive_months')
    ns=f.pivot(index='target_month',columns='candidate',values='n')
    if ns.nunique(axis=1).max()!=1 or not ns.sum().eq(165331).all():raise ValueError('hypothesis_pair_counts')
    performance=pd.DataFrame([{'candidate':name,'label':HYPOTHESIS_LABELS[name],**metrics(wide[name]),
        'analysis_role':'primary' if name=='equal' else 'alternative_comparison',
        'added_after_prior_results':name.startswith('constrained ')} for name in HYPOTHESIS_CANDIDATES])
    main=pd.DataFrame([{'candidate':'equal',**newey_west_mean_test(wide['equal']),
        'holm_family':'not_in_candidate_difference_family','cross_hypothesis_holm':'pending_H1_H2_H3_H4'}])
    differences=[]
    for name in HYPOTHESIS_CANDIDATES[1:]:
        result=newey_west_mean_test(wide[name]-wide['equal'])
        differences.append({'candidate':name,'label':HYPOTHESIS_LABELS[name],**result})
    delta=pd.DataFrame(differences).rename(columns={'mean':'mean_delta_ic','sample_sd':'delta_ic_sd'})
    delta['holm_p']=holm_bonferroni(delta.p_two_sided)
    delta['significant_holm_05']=delta.holm_p.lt(.05)
    perf=performance[['candidate','mean_ic','ic_sd','icir','positive_ic_fraction']]
    delta=delta.merge(perf,on='candidate',validate='one_to_one')
    baseline=f[f.candidate.eq('equal')][['target_month','ic']].rename(columns={'ic':'ic_equal'})
    f=f.merge(baseline,on='target_month',validate='many_to_one');f['delta_ic']=f.ic-f.ic_equal
    return {'primary_hac_test':main,'performance_summary':performance,'delta_hac_tests':delta,
            'monthly_ic_delta':f,'excluded_months':excluded}


def audit_hypothesis_sample(config, monthly, *, root=Path('.'), candidate_data=None):
    """저장 점수의 실제 종목 키·목표 쌍과 기존 IC를 독립적으로 대조한다.

    저장되지 않았던 극단값 제한 동일가중 점수만 확정 산식으로 감사용 재현.
    새 지수 추정·선택이나 기존 결과 대체를 하지 않는다.
    """
    from scipy.stats import spearmanr
    paths=config['PATHS'];s=config['H1_PCA']
    h=pd.read_parquet(root/paths['h1']);c=pd.read_parquet(root/paths['h1_components'])
    w=calculate_h1_score(winsorize_1_99(c))
    keys=['stock_id','base_month']
    wide=h[keys+['eligible','h1']].rename(columns={'h1':'equal'})
    wide=wide.merge(w[keys+['h1']].rename(columns={'h1':'equal_winsor'}),on=keys,validate='one_to_one')
    if candidate_data is None:candidate_data=run_constrained_pca(config,root=root,preconditions={},persist=False)
    for scores in (candidate_data['ordinary_scores'],candidate_data['scores']):
        if scores.duplicated(['candidate',*keys]).any():raise ValueError('duplicate_frozen_score_keys')
        if not scores.score_available.eq(np.isfinite(scores.score)).all():raise ValueError('frozen_score_availability')
        for name,part in scores.groupby('candidate',sort=False):
            if len(part)!=len(h):raise ValueError('frozen_score_row_scope')
            wide=wide.merge(part[keys+['score']].rename(columns={'score':name}),on=keys,validate='one_to_one')
    wide=wide[wide.base_month.between('2000-12','2021-11')].sort_values(keys).reset_index(drop=True)
    present=np.isfinite(wide[list(HYPOTHESIS_CANDIDATES)].to_numpy(float))
    if not (present==present[:,[0]]).all():raise ValueError('frozen_score_membership_differs')
    if not wide.loc[present[:,0],'eligible'].eq(True).all():raise ValueError('frozen_score_ineligible')
    panel=pd.read_parquet(root/paths['panel']);panel.attrs['units']=config['UNITS']
    wide=join_training_targets(wide,panel)
    stored=monthly.set_index(['candidate','target_month'])
    audits=[];pairs=[];max_error=0.
    for target,part in wide.groupby('target_month',sort=True):
        valid=np.isfinite(part['equal']) & np.isfinite(part.y)
        sample=part.loc[valid,['stock_id','base_month','target_month','y']].sort_values('stock_id')
        digest=hashlib.sha256(sample[['stock_id','target_month']].to_csv(index=False).encode('utf-8')).hexdigest()
        for name in HYPOTHESIS_CANDIDATES:
            pair=np.isfinite(part[name]) & np.isfinite(part.y)
            if not np.array_equal(pair,valid):raise ValueError('hypothesis_actual_stock_pairs_differ')
            saved=stored.loc[(name,target)]
            if int(saved.n)!=len(sample):raise ValueError('hypothesis_stored_n_disagrees')
            if target!='2001-01':
                value=float(spearmanr(part.loc[pair,name],part.loc[pair,'y']).statistic)
                error=abs(value-float(saved.ic));max_error=max(max_error,error)
                if error>1e-12:raise ValueError('hypothesis_stored_ic_disagrees')
        audits.append({'target_month':target,'n':len(sample),'candidate_count':10,'pair_keys_sha256':digest,
                       'same_pairs_all_candidates':True})
        if len(sample):pairs.append(sample)
    common=pd.concat(pairs,ignore_index=True).sort_values(['target_month','stock_id']).reset_index(drop=True)
    if len(common)!=165331 or common.duplicated(['stock_id','target_month']).any():raise ValueError('hypothesis_common_sample_count')
    return pd.DataFrame(audits),common,max_error


def hypothesis_interpretation(tables):
    main=tables['primary_hac_test'].iloc[0];delta=tables['delta_hac_tests']
    if main.significant_raw_05:
        primary=('평균 IC 양수: 예상한 방향의 순위 예측 관계에 대한 근거.' if main['mean']>0
                 else '평균 IC 음수: 예상과 반대 방향의 순위 예측 관계에 대한 근거.')
    else:primary='평균 IC가 0과 다르다는 충분한 근거를 확보하지 못했다. 예측 관계가 없다는 결론은 아니다.'
    comparisons=[]
    for _,row in delta.iterrows():
        if row.significant_holm_05:
            result='기존보다 평균 IC가 높다' if row.mean_delta_ic>0 else '기존보다 평균 IC가 낮다'
            phrase='9개 비교 Holm 보정 후 유의: '+result
        else:phrase='9개 비교 Holm 보정 후 유의하지 않음. 두 방식이 동등하다는 증거는 아니다'
        comparisons.append({'candidate':row.candidate,'interpretation':phrase})
    return {'primary':primary,'alternatives':comparisons,
        'limits':['주가설은 원값 동일가중 H1의 양측검정이며 예상 방향은 양수였다.',
          '원 p값의 5% 판정이며 H1~H4 네 주가설 전체의 Holm 판정은 다른 결과가 모일 때까지 보류한다.',
          '9개 대응 차이 검정의 Holm 집합에는 주가설 검정을 포함하지 않았다.',
          '차이 신뢰구간은 각각의 95% 구간이며 Holm 동시 신뢰구간이 아니다.',
          'HAC는 월별 시계열 의존성을 보정하지만 사후 탐색·후보 선택 편향을 없애지 않는다.',
          '부호 제약 PCA는 앞선 결과를 확인한 뒤 추가한 탐색 실험이다.',
          '같은 개발기간에서 반복 비교했으므로 독립적인 표본외 검증으로 간주하지 않는다.',
          '통계적 유의성은 효과 크기·실거래 수익성·비용 차감 성능을 입증하지 않는다.',
          '기존 원천 자기자본의 시점 불확실성과 사후 고정 종목 집합의 한계는 유지된다.',
          '최종 지수 선정은 보류했다.']}


def run_hypothesis(config, *, root=Path('.'), preconditions, persist=True):
    settings=config['H1_HYPOTHESIS'];started=datetime.now()
    candidate_data=run_constrained_pca(config,root=root,preconditions=preconditions,persist=False)
    monthly=candidate_data['tables']['monthly_ic']
    tables=hypothesis_tables(monthly)
    audit,common,max_error=audit_hypothesis_sample(config,monthly,root=root,candidate_data=candidate_data)
    tables['sample_audit']=audit
    interpretation=hypothesis_interpretation(tables)
    if not persist:return {'tables':tables,'common_sample':common,'interpretation':interpretation}
    folder=root/config['PATHS']['run_records']/settings['run_category']/started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():table.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    common.to_parquet(folder/'common_sample.parquet',index=False)
    (folder/'interpretation_summary.json').write_text(json.dumps(interpretation,ensure_ascii=False,indent=2),encoding='utf-8')
    code=['src/evaluation/rank_ic.py','src/evaluation/h1_analysis.py','src/preprocessing/panel.py',
          'src/features/h1.py','src/pipeline.py','src/contracts.py','config/experiment.yaml']
    record={'status':'EXECUTED','stage':'h1_hypothesis','command':' '.join(sys.argv),'started':started.isoformat(),
        'approved_decisions':dict(settings),'config':config,'candidate_order':list(HYPOTHESIS_CANDIDATES),
        'input_sha256':preconditions['input_sha256'],'code_sha256':{p:sha256(root/p) for p in code},
        'checks':{'source_hashes_unchanged':True,'duplicate_free_calendar_keys':True,'same_actual_pairs_all_ten':True,
                  'independent_scipy_ic_max_error':max_error,'valid_months':251,'common_pairs':len(common),
                  'excluded_target_month':'2001-01','t_p_ci_consistent':True,'holm_family_size':9},
        'method':{'kernel':'Bartlett w_l=1-l/(L+1)','hac_variance':'(sum u_t^2 + 2 sum_l w_l sum_t u_t*u_(t-l))/T^2',
                  'finite_sample_multiplier':1,'p_and_ci_reference':'t(T-1), two-sided',
                  'ordinary_t':'reference only','candidate_difference_ci':'pointwise 95%, not Holm simultaneous CI',
                  'winsor_equal_score_reconstruction':'audit only, original frozen recipe; no replacement of prior index',
                  'statistical_reference':'https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html'},
        'table_rows':{k:len(v) for k,v in tables.items()},'figures':[],
        'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir()},'verification':'pending final combined checks'}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    note=root/config['H1_TEST']['notion_tables_path'];marker='\n\n## H1 개발 종료 · 가설검정'
    def display(table):
        f=table.copy()
        for col in f.columns:
            if pd.api.types.is_float_dtype(f[col]):
                f[col]=f[col].map(lambda v:'—' if pd.isna(v) else ('<0.000001' if col in ('p_two_sided','holm_p') and 0<v<1e-6 else f'{v:.6f}'))
        return markdown_table(f)
    text=marker+'\n\n지수 개발·탐색 종료. 확정된 10개 지수·가중치·PC1 부호를 유지하며 PC2 실험·새 후보·최종 선정 없음. 기존 2001-01 목표월만 제외하고 2001-02~2021-12의 251개월·165,331개 실제 쌍을 대조했다. 2022~2025 자료 미사용.\n\n'
    text+='방법: 월별 Spearman 동일가중. Bartlett Newey–West HAC, L=floor(4(T/100)^(2/9))=4, 추가 소표본 배율 없음. 통계량·양측 p·95% 구간 모두 자유도 250의 t분포 기준. 9개 대응차이에만 Holm 보정, 각 95% 구간은 보정 전 개별 구간.\n\n'
    text+='### A. 기존 H1 주가설\n\n'+display(tables['primary_hac_test'])+'\n\n'+interpretation['primary']
    text+='\n\n### 10개 지수 성능\n\n'+display(tables['performance_summary'])
    text+='\n\n### B. 기존 H1 대비 9개 대응 차이\n\n'+display(tables['delta_hac_tests'])
    text+='\n\n'+'\n'.join('- '+HYPOTHESIS_LABELS[r['candidate']]+': '+r['interpretation'] for r in interpretation['alternatives'])
    text+='\n\n### 한계\n\n'+'\n'.join('- '+p for p in interpretation['limits'])
    text+='\n\n실행 `'+folder.relative_to(root).as_posix()+'`. CSV 원정밀도 보존, 표시 소수 6자리. monthly_ic_delta.csv는 검정에 사용한 251개월 전 지수의 월별 IC·ΔIC, excluded_months.csv는 제외 월 사유, sample_audit.csv/common_sample.parquet는 표본 대조 근거, interpretation_summary.json은 해석 요약, run.json은 입력·코드 해시와 설정. 그림 없음.\n\n[HAC 구현 기준](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)의 Bartlett 커널을 사용하고 소표본 배율은 사용하지 않음을 명시했다.\n'
    append_report(note,marker,text)
    return {'run_dir':str(folder),'primary':tables['primary_hac_test'].to_dict(orient='records'),
            'differences':tables['delta_hac_tests'].to_dict(orient='records'),'checks':record['checks']}


def check_mean_extension_inputs(config, *, root=Path('.')):
    s=config['H1_HYPOTHESIS']
    if (s.get('mean_zero_extension_approved') is not True or s.get('mean_zero_holm_family')!='nine_alternative_means_only'
        or s.get('run_category')!='h1가설검정' or s.get('reference_distribution')!='t'):
        raise PermissionError('alternative_mean_zero_tests_approval_required')
    return check_hypothesis_inputs(config,root=root)


def mean_zero_tests(monthly):
    """주검정은 고정하고 대안 9개의 평균 검정만 별도의 Holm 집합으로 묶는다."""
    from src.evaluation.rank_ic import newey_west_mean_test,holm_bonferroni
    f=monthly.copy()
    if set(f.candidate)!=set(HYPOTHESIS_CANDIDATES) or f.duplicated(['candidate','target_month']).any():
        raise ValueError('fixed_ten_mean_series_required')
    wide=f.pivot(index='target_month',columns='candidate',values='ic').reindex(columns=HYPOTHESIS_CANDIDATES)
    if wide.index.tolist()!=pd.period_range('2001-02','2021-12',freq='M').astype(str).tolist() or not np.isfinite(wide).all().all():
        raise ValueError('mean_tests_require_frozen_251_months')
    result=pd.DataFrame([{'candidate':name,'label':HYPOTHESIS_LABELS[name],
        **newey_west_mean_test(wide[name]),'analysis_role':'primary' if name=='equal' else 'exploratory_alternative_mean',
        'holm_family':'excluded_primary' if name=='equal' else 'nine_alternative_means_only'} for name in HYPOTHESIS_CANDIDATES])
    result['holm_p']=np.nan
    alternatives=result.candidate.ne('equal')
    result.loc[alternatives,'holm_p']=holm_bonferroni(result.loc[alternatives,'p_two_sided'])
    result['significant_holm_05']=pd.Series(pd.NA,index=result.index,dtype='boolean')
    result.loc[alternatives,'significant_holm_05']=result.loc[alternatives,'holm_p'].lt(.05)
    return result


def run_mean_extension(config, *, root=Path('.'), preconditions):
    s=config['H1_HYPOTHESIS']
    previous=run_hypothesis(config,root=root,preconditions=preconditions,persist=False)
    monthly=previous['tables']['monthly_ic_delta']
    table=mean_zero_tests(monthly)
    primary=previous['tables']['primary_hac_test']
    for col in ('mean','hac_se','statistic','p_two_sided','ci95_low','ci95_high'):
        np.testing.assert_allclose(table.loc[table.candidate.eq('equal'),col],primary[col],rtol=1e-12,atol=1e-16)
    folder=root/config['PATHS']['run_records']/s['run_category']/datetime.now().strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,t in previous['tables'].items():t.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    previous['common_sample'].to_parquet(folder/'common_sample.parquet',index=False)
    (folder/'interpretation_summary.json').write_text(json.dumps(previous['interpretation'],ensure_ascii=False,indent=2),encoding='utf-8')
    table.to_csv(folder/'mean_ic_hac_tests.csv',index=False,encoding='utf-8-sig')
    interpretation={'scope':'각 지수 평균 IC=0 검정. 앞선 기존 H1 대비 평균 차이 검정과 별개.',
        'family':'기존 동일가중 주검정은 제외하고 대안 9개의 평균 검정에만 별도 Holm 보정.',
        'result':'9개 대안의 평균 IC는 모두 양수이며 보정 전·Holm 보정 후 모두 5%에서 유의.' if table.iloc[1:].significant_holm_05.all() else '일부 대안의 평균 검정이 유의하지 않으므로 표를 개별 확인.',
        'limits':'각각 0과 다른 것과 기존 H1보다 좋은 것은 별개다. 이전 9개 차이 검정은 비유의 상태 그대로이며 동등성 증거도 아니다. 추가 평균 검정은 탐색적 분석이고 다중검정 보정은 사후 선택 편향·표본외 검증을 대체하지 않는다. 최종 지수 선정 보류.'}
    (folder/'mean_zero_interpretation.json').write_text(json.dumps(interpretation,ensure_ascii=False,indent=2),encoding='utf-8')
    code=['src/evaluation/rank_ic.py','src/evaluation/h1_analysis.py','src/pipeline.py','config/experiment.yaml']
    record={'status':'EXECUTED','stage':'alternative_mean_zero_extension','command':' '.join(sys.argv),
        'approved_decisions':dict(s),'input_sha256':preconditions['input_sha256'],
        'code_sha256':{p:sha256(root/p) for p in code},'checks':{'previous_primary_identical':True,
        'original_recipes_recomputed_from_shared_data':True,'same_251_months':True,'index_recipe_unchanged':True},
        'method':{'reference':'t(T-1)','kernel':'Bartlett','inputs':'shared data only'},'new_holm_family':'nine_alternative_means_only',
        'previous_holm_family_unchanged':'nine_paired_differences_only','figures':[],
        'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir()},'verification':'pending'}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    note=root/config['H1_TEST']['notion_tables_path'];marker='\n\n## 추가 평균 IC=0 검정'
    shown=table[['label','mean','hac_se','statistic','p_two_sided','holm_p','ci95_low','ci95_high']].copy()
    for column in shown.columns[1:]:
        shown[column]=shown[column].map(lambda v:'—' if pd.isna(v) else (f'{v:.6e}' if 0<abs(v)<.000001 else f'{v:.6f}'))
    text=marker+'\n\n각 지수의 평균 IC가 0과 다른지 추가 HAC 양측검정. 251개월, Bartlett L=4, t(250), 소표본 배율 없음. 기존 주검정은 유지하고 대안 9개 평균 검정에만 별도 Holm 보정. 앞선 9개 대응 차이 검정의 보정 집합과 섞지 않는다.\n\n'
    text+=markdown_table(shown)+'\n\n'+interpretation['result']+'\n\n'+interpretation['limits']
    text+='\n\n실행 `'+folder.relative_to(root).as_posix()+'`, mean_ic_hac_tests.csv 및 mean_zero_interpretation.json 추가. 이전 표·월별 자료 8개는 그대로 복사하여 함께 제공. 그림 없음.\n'
    append_report(note,marker,text)
    return {'run_dir':str(folder),'mean_tests':table.to_dict(orient='records')}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def metrics(values):
    s=pd.Series(values).dropna()
    sd=float(s.std(ddof=1)) if len(s)>1 else np.nan
    return {'valid_months':len(s), 'mean_ic':float(s.mean()), 'ic_sd':sd,
            'icir':float(s.mean()/sd) if np.isfinite(sd) and sd>0 else np.nan,
            'positive_ic_fraction':float(s.gt(0).mean()) if len(s) else np.nan}


def analyze(components, equal_scores, panel, *, representative_year=2021):
    """표준화 완료 표를 받은 뒤에만 목표 결합. 반환 표는 파일 없이 검증 가능하다."""
    if not 2001 <= representative_year <= 2021: raise ValueError('representative_year_outside_train')
    c=components.loc[components.base_month.between('2000-12','2021-11')].reset_index(drop=True)
    equal=equal_scores.loc[equal_scores.base_month.between('2000-12','2021-11')].reset_index(drop=True)
    pd.testing.assert_frame_equal(c[['stock_id','base_month','eligible']],equal[['stock_id','base_month','eligible']])
    limited=winsorize_1_99(c)
    winsor=calculate_h1_score(limited)
    # 경계 추정과 표준화를 마친 뒤에만 목표를 결합한다.
    raw=join_training_targets(c,panel)
    scores=join_training_targets(equal,panel)
    w=scores[['stock_id','base_month','target_month','y']].merge(
        winsor[['stock_id','base_month','h1']],on=['stock_id','base_month'],how='left',validate='one_to_one')
    all_months=sorted(raw.base_month.unique())
    ic_rows=[]; distribution=[]; extreme=[]; correlations=[]; contributions=[]; changes=[]
    group={'model':'feature_score','fold':'development'}
    max_sample_mismatch=0
    for month in all_months:
        f=raw[raw.base_month.eq(month)].copy()
        eq=scores[scores.base_month.eq(month)].copy()
        ws=w[w.base_month.eq(month)].copy()
        wc=limited[limited.base_month.eq(month)].copy()
        wz=winsor[winsor.base_month.eq(month)].copy()
        target=str(pd.Period(month,freq='M')+1)
        member=f.eligible.eq(True).fillna(False) & f[[x+'_available' for x in FEATURES]].all(axis=1)
        common=f.loc[member]
        z=eq.loc[member,['z_size','z_bm','z_mom']].copy()
        z.columns=list(FEATURES); z['size']=-z['size']
        for feature in FEATURES:
            for scope,mask in [('common',member),('own',f.eligible.eq(True).fillna(False) & f[feature+'_available'])]:
                t=f[['stock_id','target_month','y']].copy()
                t['score']=f[feature].where(mask)*(-1 if feature=='size' else 1)
                for key,value in {**group,'candidate':feature}.items(): t[key]=value
                row=rank_ic_for_month(t,score_column='score').iloc[0].to_dict()
                ic_rows.append({**row,'base_month':month,'sample_scope':scope})
            for scope,part in [('own',f.loc[f.eligible.eq(True).fillna(False) & f[feature+'_available']]),('common',common)]:
                v=part[feature].dropna()
                for variant,vals in [('raw',v), *([('winsor',wc.loc[part.index,feature].dropna())] if scope=='common' else [])]:
                    q=vals.quantile([.01,.05,.25,.5,.75,.95,.99],interpolation='linear')
                    distribution.append({'base_month':month,'target_month':target,'feature':feature,
                        'sample_scope':scope,'variant':variant,'n':len(vals), 'min':vals.min(),'max':vals.max(),
                        **{f'q{int(k*100):02d}':v for k,v in q.items()},'skewness':vals.skew(),
                        'excess_kurtosis':vals.kurt(), 'shape_reason':'' if len(vals)>=4 and vals.nunique()>1 else 'too_few_or_constant'})
                if len(v):
                    lo,hi=v.quantile([.01,.99],interpolation='linear')
                    for idx,value in v[(v<lo)|(v>hi)].items():
                        extreme.append({'stock_id':f.loc[idx,'stock_id'],'base_month':month,'target_month':target,
                            'feature':feature,'sample_scope':scope,'value':value,'q01':lo,'q99':hi,
                            'side':'below_q01' if value<lo else 'above_q99'})
        for method in ('pearson','spearman'):
            matrix=z.corr(method=method)
            for a in FEATURES:
                for b in FEATURES:
                    correlations.append({'level':'month','base_month':month,'target_month':target,'method':method,
                                         'feature_a':a,'feature_b':b,'n':len(z),'correlation':matrix.loc[a,b]})
        hv=eq.loc[member,'h1']
        variance=float(hv.var(ddof=1)) if len(hv)>1 else np.nan
        for feature in FEATURES:
            valid=len(z)>=2 and z[feature].nunique()>1 and hv.nunique()>1
            contributions.append({'level':'month','base_month':month,'target_month':target,'feature':feature,'n':len(z),
                'corr_z_h1':float(z[feature].corr(hv)) if valid else np.nan,
                'variance_share':float((z[feature]/3).cov(hv)/variance) if valid and variance>0 else np.nan,
                'reason':'' if valid else 'unavailable_or_constant'})
        # 비교 점수의 관측 집합을 일치시킨다. 목표는 경계·표준화에 쓰지 않는다.
        pair_scores=np.isfinite(eq.h1) & np.isfinite(ws.h1)
        ns=[]
        for candidate,source in [('equal',eq),('winsor',ws)]:
            t=source[['stock_id','target_month','y']].copy()
            t['score']=source.h1.where(pair_scores)
            for key,value in {**group,'candidate':candidate}.items(): t[key]=value
            row=rank_ic_for_month(t,score_column='score').iloc[0].to_dict()
            ic_rows.append({**row,'base_month':month,'sample_scope':'paired_common'})
            ns.append(row['n'])
        max_sample_mismatch=max(max_sample_mismatch,abs(ns[0]-ns[1]))
        ranks_valid=pair_scores.sum()>=2 and eq.loc[pair_scores,'h1'].nunique()>1 and ws.loc[pair_scores,'h1'].nunique()>1
        before=eq.loc[pair_scores,'h1']; after=ws.loc[pair_scores,'h1']
        rank_before=before.rank(method='average'); rank_after=after.rank(method='average')
        change={'level':'month','base_month':month,'target_month':target,'n_scores':int(pair_scores.sum()),
                'rank_spearman':float(before.corr(after,method='spearman')) if ranks_valid else np.nan,
                'mean_abs_rank_change':float((rank_before-rank_after).abs().mean()) if ranks_valid else np.nan,
                'max_abs_rank_change':float((rank_before-rank_after).abs().max()) if ranks_valid else np.nan,
                'rank_reason':'' if ranks_valid else 'unavailable_or_constant'}
        for feature in FEATURES:
            change['max_abs_z_before_'+feature]=eq.loc[member,'z_'+feature].abs().max()
            change['max_abs_z_after_'+feature]=wz.loc[member,'z_'+feature].abs().max()
        changes.append(change)
    ic=pd.DataFrame(ic_rows,columns=H1_MONTHLY_IC_COLUMNS)
    summary=[]
    for (scope,candidate),part in ic.groupby(['sample_scope','candidate'],sort=True):
        for year,t in [('2001~2021',part),*part.groupby(part.target_month.str[:4],sort=True)]:
            summary.append({'target_year':year,'candidate':candidate,'sample_scope':scope,**metrics(t.ic),
                            'unable_months':int(t.ic.isna().sum())})
    corr=pd.DataFrame(correlations)
    avg=corr.groupby(['method','feature_a','feature_b'],as_index=False).agg(correlation=('correlation','mean'),valid_months=('correlation','count'))
    corr=pd.concat([corr,avg.assign(level='overall')],ignore_index=True)
    contribution=pd.DataFrame(contributions)
    annual=contribution.groupby([contribution.target_month.str[:4].rename('target_year'),'feature'],as_index=False).agg(
        corr_z_h1=('corr_z_h1','mean'),variance_share=('variance_share','mean'),valid_months=('variance_share','count'))
    overall=contribution.groupby('feature',as_index=False).agg(corr_z_h1=('corr_z_h1','mean'),variance_share=('variance_share','mean'),valid_months=('variance_share','count')).assign(target_year='2001~2021')
    contribution=pd.concat([contribution,annual.assign(level='year'),overall.assign(level='overall')],ignore_index=True)
    compare=pd.DataFrame(changes)
    paired=ic[ic.sample_scope.eq('paired_common')].pivot(index='target_month',columns='candidate',values='ic')
    delta=paired.winsor-paired.equal
    compare['ic_equal']=compare.target_month.map(paired.equal)
    compare['ic_winsor']=compare.target_month.map(paired.winsor)
    compare['delta_ic']=compare.target_month.map(delta)
    comparison_rows=[]
    valid_months=paired.dropna().index
    for candidate in ('equal','winsor'):
        comparison_rows.append({'candidate':candidate,**metrics(paired.loc[valid_months,candidate]),
            'unable_months':len(paired)-len(valid_months),
            'mean_delta_ic':float(delta.loc[valid_months].mean()) if candidate=='winsor' else 0.,
            'mean_rank_spearman':float(compare.rank_spearman.mean()) if candidate=='winsor' else 1.,
            'mean_abs_rank_change':float(compare.mean_abs_rank_change.mean()) if candidate=='winsor' else 0.,
            'selection':'pending_user_decision'})
    candidates=pd.DataFrame(comparison_rows)
    sums=contribution[contribution.level.eq('month')].groupby('target_month').variance_share.sum(min_count=1).dropna()
    equal_n=ic[ic.candidate.eq('equal')].set_index('target_month').n
    common_n=ic[ic.sample_scope.eq('common') & ic.candidate.eq('size')].set_index('target_month').n
    checks={'paired_score_and_target_samples_equal':max_sample_mismatch==0,
            'feature_common_pair_counts_equal':all(ic[ic.sample_scope.eq('common') & ic.candidate.eq(c)].set_index('target_month').n.equals(common_n) for c in FEATURES),
            'equal_score_pair_count_matches_common':equal_n.equals(common_n),
            'variance_contribution_sums_to_one':bool(np.allclose(sums,1.,atol=1e-10,rtol=1e-10)),
            'monthly_rows_preserved':len(ic)==len(all_months)*8,
            'no_test_targets':bool(ic.target_month.between('2001-01','2021-12').all()),
            'representative_selected_before_results':representative_year==2021}
    if not all(checks.values()):raise ValueError('h1_analysis_checks_failed:'+str(checks))
    representative=ic[ic.sample_scope.eq('common') & ic.target_month.str.startswith(str(representative_year))].pivot(
        index=['target_month','base_month','n'],columns='candidate',values='ic').reset_index()
    return {'monthly_ic':ic,'ic_summary_by_target_year':pd.DataFrame(summary),'feature_distribution':pd.DataFrame(distribution),
            'extreme_values':pd.DataFrame(extreme),'feature_correlation':corr,'contribution':contribution,
            'winsor_compare':compare,'candidate_compare':candidates,'representative_year':representative},checks


def draw_figures(tables, folder):
    from src.features.observe import setup_matplotlib
    plt=setup_matplotlib()
    files=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/name);plt.close(fig);files.append(name)
    dist=tables['feature_distribution']; monthly=tables['monthly_ic']
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for feature,ax in zip(FEATURES,axes):
        f=dist[dist.feature.eq(feature)&dist.sample_scope.eq('own')&dist.variant.eq('raw')]
        x=pd.to_datetime(f.target_month)
        ax.fill_between(x,f.q01,f.q99,alpha=.12,label='1~99 분위수')
        ax.fill_between(x,f.q25,f.q75,alpha=.3,label='25~75 분위수')
        ax.plot(x,f.q50,label='중앙값');ax.plot(x,f['min'],lw=.7,alpha=.6,label='최소·최대');ax.plot(x,f['max'],lw=.7,alpha=.6)
        ax.set_ylabel(LABELS[feature] if feature!='size' else '규모 원값');ax.legend(fontsize=8,ncol=3)
    axes[0].set_title('특징별 월별 분포와 극단값 · 특징 자체 유효 집합');axes[-1].set_xlabel('목표월')
    save(fig,'01_특징분포_극단값.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for method,ax in zip(('pearson','spearman'),axes):
        f=tables['feature_correlation'];f=f[f.level.eq('overall')&f.method.eq(method)]
        matrix=f.pivot(index='feature_a',columns='feature_b',values='correlation').reindex(index=FEATURES,columns=FEATURES)
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(3),[LABELS[c] for c in FEATURES]);ax.set_yticks(range(3),[LABELS[c] for c in FEATURES])
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center')
        ax.set_title(method+' · 월별 상관의 평균');ax.grid(False)
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.03)
    # colorbar를 포함한 레이아웃은 직접 조정한다.
    fig.subplots_adjust(left=.09,right=.88,bottom=.16,top=.88,wspace=.35)
    fig.savefig(folder/'02_특징상관행렬.png');plt.close(fig);files.append('02_특징상관행렬.png')
    fig,ax=plt.subplots(figsize=(12,4.8))
    for c,color in zip(FEATURES,COLORS):
        f=monthly[monthly.sample_scope.eq('common')&monthly.candidate.eq(c)]
        ax.plot(pd.to_datetime(f.target_month),f.ic,label=LABELS[c],color=color,lw=1)
    ax.axhline(0,color='gray',lw=.8);ax.set(title='특징별 월별 순위 상관 · H1 공통 집합',xlabel='목표월',ylabel='Spearman 순위 상관');ax.legend()
    save(fig,'03_특징별_월별순위상관.png')
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for c,ax in zip(FEATURES,axes):
        for variant,label in [('raw','기존'),('winsor','극단값 제한')]:
            f=dist[dist.sample_scope.eq('common')&dist.feature.eq(c)&dist.variant.eq(variant)]
            ax.plot(pd.to_datetime(f.target_month),f['max'],label=label+' 최댓값')
            ax.plot(pd.to_datetime(f.target_month),f.q99,label=label+' 99 분위수',ls='--',alpha=.7)
        ax.set_ylabel('규모' if c=='size' else LABELS[c]);ax.legend(ncol=2,fontsize=8)
    axes[0].set_title('같은 공통 집합의 극단값 제한 전후 원특징 분포');axes[-1].set_xlabel('목표월')
    save(fig,'04_극단값제한_전후분포.png')
    f=tables['winsor_compare'];x=pd.to_datetime(f.target_month)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    axes[0].plot(x,f.rank_spearman);axes[0].set_ylabel('전후 순위 상관')
    axes[1].plot(x,f.mean_abs_rank_change);axes[1].set_ylabel('평균 순위 이동 폭')
    axes[2].plot(x,f.delta_ic);axes[2].axhline(0,color='gray',lw=.8);axes[2].set_ylabel('차월 순위 상관 차이');axes[2].set_xlabel('목표월')
    axes[0].set_title('극단값 제한 전후 순위 변화와 월별 성능 차이')
    save(fig,'05_순위변화_월별차이.png')
    fig,axes=plt.subplots(3,1,figsize=(12,8.5),sharex=True)
    for c,ax in zip(FEATURES,axes):
        ax.plot(x,f['max_abs_z_before_'+c],label='기존');ax.plot(x,f['max_abs_z_after_'+c],label='극단값 제한')
        ax.set_ylabel(('규모' if c=='size' else LABELS[c])+' 최대 절댓값');ax.legend()
    axes[0].set_title('월별 표준화 점수 최대 절댓값 전후 비교');axes[-1].set_xlabel('목표월')
    save(fig,'06_최대표준점수_전후.png')
    fig,ax=plt.subplots(figsize=(12,4.8))
    for c in ('equal','winsor'):
        f=monthly[monthly.candidate.eq(c)]
        ax.plot(pd.to_datetime(f.target_month),f.ic,label=LABELS[c],lw=1)
    ax.axhline(0,color='gray',lw=.8);ax.set(title='동일 표본의 기존·극단값 제한 점수 월별 순위 상관',xlabel='목표월',ylabel='Spearman 순위 상관');ax.legend()
    save(fig,'07_동일가중_월별순위상관.png')
    return files


def markdown_table(frame, columns=None):
    f=frame if columns is None else frame[columns]
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in f.itertuples(index=False,name=None))


def report_markdown(tables, run_dir, figures):
    """기존 단일 붙여넣기 파일의 본문. 실행 폴더에 복제하지 않는다."""
    text='\n\n## 10주차 2단계 · 대표 연도 2021년\n\n'
    text+='실행: `'+run_dir.as_posix()+'`\n\n'
    text+='규모는 시가총액의 자연로그, 가치는 자기자본÷시가총액, 모멘텀은 과거 11개월 복리 수익률이다. 규모는 비교 시 부호를 반전했다. 분포·극단값·월별 상관·기여도·1·99 분위수 제한 비교를 실행했다. 원특징 경계는 당월 H1 공통 집합에서 선형 보간 분위수로 계산한 후 월별 표준화를 다시 수행했다. 목표는 그 뒤에 붙였다.\n\n'
    text+='### 3.1 대표 연도 월별 결과\n\n'
    r=tables['representative_year'].rename(columns={'target_month':'목표월','base_month':'기준월','n':'계산 종목 수','size':'−규모 순위 상관','bm':'가치 순위 상관','mom':'모멘텀 순위 상관'})
    text+=markdown_table(r)+'\n'
    summary=tables['ic_summary_by_target_year']
    common=summary[summary.sample_scope.eq('common')]
    wide=common.pivot(index='target_year',columns='candidate',values='mean_ic').reset_index()
    counts=common.groupby('target_year').valid_months.min()
    wide['valid_months']=wide.target_year.map(counts)
    text+='### 3.1 목표월 연도별 평균 순위 상관 · 공통 집합\n\n'+markdown_table(wide.rename(columns={'target_year':'목표 수익률 연도','valid_months':'유효 월 수','size':'−규모 평균','bm':'가치 평균','mom':'모멘텀 평균'}))+'\n'
    own=summary[summary.sample_scope.eq('own') & summary.target_year.eq('2001~2021')].copy(); own['candidate']=own.candidate.map(LABELS)
    text+='### 보조: 특징 자체 유효 집합 · 전체 평균\n\n'+markdown_table(own[['candidate','valid_months','mean_ic','ic_sd','icir','positive_ic_fraction']].rename(columns={'candidate':'특징','valid_months':'유효 월 수','mean_ic':'평균 순위 상관','ic_sd':'월별 표준편차','icir':'평균÷표준편차','positive_ic_fraction':'양수 월 비율'}))+'\n'
    corr=tables['feature_correlation'];corr=corr[corr.level.eq('overall')]
    text+='### 특징 간 방향 정렬 상관 · 월별 상관의 평균\n\n'+markdown_table(corr[['method','feature_a','feature_b','correlation','valid_months']].replace({'feature_a':LABELS,'feature_b':LABELS}).rename(columns={'method':'계산 방법','feature_a':'특징 1','feature_b':'특징 2','correlation':'월별 상관 평균','valid_months':'유효 월 수'}))+'\n'
    con=tables['contribution'];con=con[con.level.eq('year')|con.level.eq('overall')]
    text+='### 구성요소 기여도 · 목표월 연도별 월평균\n\n'+markdown_table(con[['target_year','feature','corr_z_h1','variance_share','valid_months']].replace({'feature':LABELS}).rename(columns={'target_year':'목표 수익률 연도','feature':'구성요소','corr_z_h1':'구성요소와 점수의 상관','variance_share':'분산 기여 비율','valid_months':'유효 월 수'}))+'\n'
    text+='### 3.2.1 같은 표본의 극단값 제한 전후\n\n'
    c=tables['candidate_compare'].copy();c['candidate']=c.candidate.map(LABELS)
    text+=markdown_table(c.drop(columns='selection').rename(columns={'candidate':'방식','valid_months':'유효 월 수','mean_ic':'평균 순위 상관','ic_sd':'월별 표준편차','icir':'평균÷표준편차','positive_ic_fraction':'양수 월 비율','mean_delta_ic':'기존 대비 평균 차이','unable_months':'불능 월 수','mean_rank_spearman':'전후 순위 상관 평균','mean_abs_rank_change':'평균 순위 이동 폭'}))+'\n'
    comp=tables['winsor_compare']
    effect=pd.DataFrame([{'특징':LABELS[c], '기존 최대 절댓값':comp['max_abs_z_before_'+c].max(),
                         '제한 후 최대 절댓값':comp['max_abs_z_after_'+c].max(),
                         '기존 월평균 최대 절댓값':comp['max_abs_z_before_'+c].mean(),
                         '제한 후 월평균 최대 절댓값':comp['max_abs_z_after_'+c].mean()} for c in FEATURES])
    text+='### 극단값 영향 · 표준화 점수 최대 절댓값\n\n'+markdown_table(effect)+'\n'
    text+='### 분포·극단값 목록\n\n각 월의 최소·최대, 1·5·25·50·75·95·99 분위수, 표본 왜도와 초과첨도는 `feature_distribution.csv`, 1·99 분위수 밖 종목별 목록은 `extreme_values.csv`에 전부 기록했다. 자체 유효 집합과 공통 집합을 구분했다. 초과첨도는 정규분포 기준 0, 표본 편향 보정 방식이다.\n\n'
    text+='### 산출 그림 목록\n\n'+''.join('- `'+(run_dir/name).as_posix()+'`\n' for name in figures)
    text+='\n### 후속 그림 목록 · 승인 후\n\n- 주성분 설명분산·누적 설명분산\n- 월별 적재계수\n- 주성분 점수와 기존 점수의 월별 순위 상관\n- 기존·선정 점수 및 월별 성능 차이\n- 평균 순위 상관·차이의 신뢰구간\n'
    text+='\n### 한계·다음 결정\n\n성능 차이는 개발기간의 기술적 비교다. 가설 지지·유의성·실거래 수익성을 판단하지 않는다. 같은 자료로 후보를 탐색하므로 선택의 영향을 남긴다. 자기자본 원천·과거 공개 자료·사후 고정 종목의 기존 한계는 유지된다. 주성분 입력에 극단값 제한을 적용할지는 사용자 결정 대기. 주성분 성분·결합식·부호 규칙 승인과 평균 검정 승인은 별도다.\n'
    return text


def run_analysis(config, *, root=Path('.')):
    from src.pipeline import check_features_current
    pre=check_features_current(config,root=root)
    settings=config['H1_TEST']
    if settings['approved_stage']!=2 or settings['representative_year']!=2021:
        raise PermissionError('only_approved_stage2_representative_2021')
    paths=config['PATHS']; started=datetime.now()
    c=pd.read_parquet(root/paths['h1_components']);h=pd.read_parquet(root/paths['h1'])
    panel=pd.read_parquet(root/paths['panel']);panel.attrs['units']=config['UNITS']
    tables,checks=analyze(c,h,panel,representative_year=settings['representative_year'])
    folder=root/paths['run_records']/'H1_TEST'/started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():table.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    figures=draw_figures(tables,folder)
    inputs=[paths[n] for n in ('panel','h1_components','h1','mom')]
    code=['src/evaluation/h1_analysis.py','src/evaluation/rank_ic.py','src/preprocessing/panel.py',
          'src/features/h1.py','src/features/observe.py','src/contracts.py','src/pipeline.py']
    record={'status':'EXECUTED','started':started.isoformat(),'command':' '.join(sys.argv),
        'preconditions':pre,'approved_decisions':dict(settings),'h1_config':dict(config['H1']),
        'input_sha256':{p:sha256(root/p) for p in inputs},'code_sha256':{p:sha256(root/p) for p in code},
        'checks':checks,'sample_counts':{'common_feature_cells':int(h.loc[h.base_month.between('2000-12','2021-11'),'h1_available'].sum()),'common_ic_pairs':int(tables['monthly_ic'].loc[lambda x:x.candidate.eq('equal'),'n'].sum()),'comparison_valid_months':int(tables['candidate_compare'].valid_months.min())},'table_rows':{n:len(t) for n,t in tables.items()},'figures':figures,
        'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()},
        'method':{'year_axis':'target_month','quantile_interpolation':'linear','distribution_skew':'unbiased sample',
                  'distribution_kurtosis':'unbiased Fisher excess','main_sample':'eligible common Size/BM/Mom',
                  'correlation':'direction-aligned Z, monthly then equal-weight mean',
                  'extreme_list':'outside own/common monthly q01 and q99; diagnostic only',
                  'ic_sd_ddof':1,'rank_change_sample':'common scores independent of target availability',
                  'pca':'not implemented/executed','hac':'not executed, requires separate approval'}}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=1,default=str),encoding='utf-8')
    # 사용자 지정 단일 파일만 갱신한다. 과거 실행의 다른 산출물은 변경하지 않는다.
    note=root/settings['notion_tables_path']
    old=note.read_text(encoding='utf-8') if note.is_file() else ''
    for a,b in {'stocks':'종목 수','base_months':'저장 기준월','rows_per_table':'표별 행 수','analysis_common_cells':'세 값 공동 유효 칸','h1_valid_months':'점수 산출 가능 월 수','h1_available_cells':'점수 산출 칸'}.items():
        old=old.replace('| '+a+' |','| '+b+' |')
    append_report(note,'\n\n## 10주차 2단계',report_markdown(tables,folder.relative_to(root),figures))
    return {'run_dir':folder.as_posix(),'checks':checks,'candidate_compare':tables['candidate_compare'].to_dict(orient='records'),
            'representative_year':2021,'notion_tables_path':note.as_posix(),'figures':figures}


def pca_variance_analysis(components, equal_scores):
    """설명분산만 생성. 목표 결합·점수 선택·적재계수·순위 상관 계산 없음."""
    from src.features.h1 import pca_a, pca_b
    c=components[components.base_month.between('2000-12','2021-11')].reset_index(drop=True)
    raw=equal_scores[equal_scores.base_month.between('2000-12','2021-11')].reset_index(drop=True)
    pd.testing.assert_frame_equal(c[['stock_id','base_month','eligible']],raw[['stock_id','base_month','eligible']])
    inputs={'raw':raw,'winsor':calculate_h1_score(winsorize_1_99(c))}
    tables=[]
    for variant,source in inputs.items():
        for function in (pca_a,pca_b):
            f=function(source).assign(variant=variant)
            tables.append(f[['variant',*f.columns.drop('variant')]])
    monthly=pd.concat(tables,ignore_index=True)
    valid=monthly[monthly.available]
    keys=['variant','method','component']
    def summary(part):
        return part.groupby(keys,as_index=False).agg(
            valid_months=('explained_variance_ratio','count'),
            mean_ratio=('explained_variance_ratio','mean'),median_ratio=('explained_variance_ratio','median'),
            min_ratio=('explained_variance_ratio','min'),max_ratio=('explained_variance_ratio','max'),
            mean_cumulative=('cumulative_explained_variance_ratio','mean'),
            min_cumulative=('cumulative_explained_variance_ratio','min'),max_cumulative=('cumulative_explained_variance_ratio','max'),
            months_cumulative_ge_80=('cumulative_explained_variance_ratio',lambda s:int(s.ge(.8).sum())))
    overall=summary(valid)
    final=valid[valid.base_month.eq(valid.base_month.max())][keys+['explained_variance_ratio','cumulative_explained_variance_ratio','fit_n']]
    final=final.rename(columns={'explained_variance_ratio':'final_ratio','cumulative_explained_variance_ratio':'final_cumulative','fit_n':'final_fit_n'})
    overall=overall.merge(final,on=keys,validate='one_to_one')
    yearly=pd.concat([summary(t).assign(target_year=year) for year,t in valid.groupby(valid.target_month.str[:4])],ignore_index=True)
    # 미래 목표를 참조하지 않고 네 경우의 기준월·현재 종목 수를 대조한다.
    counts=monthly[monthly.component.eq('PC1')].pivot(index='base_month',columns=['variant','method'],values='current_n')
    sums=valid.groupby(['variant','method','base_month']).explained_variance_ratio.sum()
    inc=valid.sort_values(['variant','method','base_month','component']).groupby(['variant','method','base_month']).cumulative_explained_variance_ratio.diff().dropna()
    b=monthly[monthly.method.eq('pca_b') & monthly.component.eq('PC1')].copy()
    expected=b.groupby('variant').current_n.cumsum()
    checks={'ratios_sum_to_one':bool(np.allclose(sums,1.,atol=1e-12)),
            'ratios_between_zero_and_one':bool(valid.explained_variance_ratio.between(0,1).all()),
            'cumulative_nondecreasing':bool(inc.ge(-1e-12).all()),
            'same_current_stocks_all_four':bool(counts.eq(counts.iloc[:,0],axis=0).all().all()),
            'cumulative_fit_counts_exact':bool(b.fit_n.eq(expected).all()),
            'no_future_fit_months':bool(valid.fit_last_base_month.le(valid.base_month).all()),
            'calendar_months_and_components_retained':len(monthly)==len(c.base_month.unique())*4*3,
            'no_ic_or_scores_in_outputs':not any(x in monthly for x in ('y','ic','h1','score','selected_component'))}
    if not all(checks.values()):raise ValueError('pca_variance_checks_failed:'+str(checks))
    return {'pca_variance_monthly':monthly,'pca_variance_summary':overall,'pca_variance_by_target_year':yearly},checks


PCA_METHOD_LABEL={'pca_a':'월별 방식','pca_b':'과거 누적 방식'}
PCA_VARIANT_LABEL={'raw':'원값 유지','winsor':'1·99 분위수 제한'}


def pca_variance_comparison(summary):
    rows=[]
    for (variant,method),part in summary.groupby(['variant','method'],sort=True):
        f=part.set_index('component');second=f.loc['PC2']
        rows.append({'방식':PCA_METHOD_LABEL[method],'입력':PCA_VARIANT_LABEL[variant],
            '유효 월 수':int(f.valid_months.min()),'첫째 성분(%)':100*f.loc['PC1','mean_ratio'],
            '둘째 성분(%)':100*f.loc['PC2','mean_ratio'],'셋째 성분(%)':100*f.loc['PC3','mean_ratio'],
            '첫째+둘째 누적(%)':100*second.mean_cumulative,
            '첫째+둘째 누적 최솟값(%)':100*second.min_cumulative,
            '첫째+둘째 누적 최댓값(%)':100*second.max_cumulative,
            '첫째+둘째 누적 80% 이상 월 수':int(second.months_cumulative_ge_80)})
    return pd.DataFrame(rows)


def draw_pca_variance(tables,folder):
    from src.features.observe import setup_matplotlib
    plt=setup_matplotlib();summary=tables['pca_variance_summary'];monthly=tables['pca_variance_monthly']
    order=[('raw','pca_a'),('winsor','pca_a'),('raw','pca_b'),('winsor','pca_b')]
    fig,ax=plt.subplots(figsize=(11,5.7));bottom=np.zeros(4)
    for k,color in enumerate(COLORS,1):
        vals=np.array([summary.loc[summary.variant.eq(v)&summary.method.eq(m)&summary.component.eq(f'PC{k}'),'mean_ratio'].iloc[0] for v,m in order])*100
        ax.bar(range(4),vals,bottom=bottom,color=color,label=f'{k}번째 성분',width=.6)
        for i,value in enumerate(vals):ax.text(i,bottom[i]+value/2,f'{value:.2f}%',ha='center',va='center',color='white',fontsize=12)
        bottom+=vals
    ax.set_xticks(range(4),[PCA_METHOD_LABEL[m]+'\n'+PCA_VARIANT_LABEL[v] for v,m in order])
    ax.set_ylim(0,105);ax.set_ylabel('설명분산 비율(%)');ax.set_title('주성분별 설명분산 · 유효 월별 비율의 평균')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False)
    fig.tight_layout();fig.savefig(folder/'01_주성분별_설명분산비율.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),sharex=True,sharey=True)
    for (variant,method),ax in zip(order,axes.flat):
        for k,color in enumerate(COLORS,1):
            f=monthly[monthly.variant.eq(variant)&monthly.method.eq(method)&monthly.component.eq(f'PC{k}')]
            ax.plot(pd.to_datetime(f.target_month),f.cumulative_explained_variance_ratio*100,color=color,label=f'1~{k}번째 누적')
        ax.axhline(80,color='#555555',ls='--',lw=.8,label='80% 참고선')
        ax.set_title(PCA_METHOD_LABEL[method]+' · '+PCA_VARIANT_LABEL[variant]);ax.set_ylim(0,105)
        ax.set_xlabel('목표월(기준월의 다음 달, 표시용)');ax.set_ylabel('누적 설명분산(%)');ax.legend(fontsize=8,ncol=2,loc='lower right')
    fig.suptitle('주성분 누적 설명분산의 월별 변화 · 성분 선택은 미실행')
    fig.tight_layout();fig.savefig(folder/'02_월별_누적설명분산비율.png');plt.close(fig)
    return ['01_주성분별_설명분산비율.png','02_월별_누적설명분산비율.png']


def pca_variance_report(tables,folder,figures):
    text='\n\n## 주성분 설명분산 · 원값 유지와 1·99 분위수 제한 (2026-11-06)\n\n'
    text+='실행: `'+folder.as_posix()+'`\n\n'
    text+='승인 범위는 설명분산 확인까지다. 사용할 성분·부호·결합식·후보는 정하지 않았고, 목표 수익률 결합·순위 상관·평균 검정은 실행하지 않았다. 비율이 높다는 것은 세 특징의 변동을 많이 담는다는 뜻이며 수익률 예측력이 높다는 뜻은 아니다.\n\n'
    text+='### 전체기간 월평균 비율\n\n'+markdown_table(pca_variance_comparison(tables['pca_variance_summary']))+'\n'
    text+='표의 평균은 유효한 월별 비율의 동일가중 평균이다. 하나의 전기간 주성분 분석에서 나온 비율이 아니다. 80%는 참고선이며 자동 선택 기준으로 적용하지 않았다.\n\n'
    yearly=tables['pca_variance_by_target_year'];year=yearly[yearly.target_year.eq('2021')]
    text+='### 대표 연도 2021년 · 월평균 비율\n\n'+markdown_table(pca_variance_comparison(year))+'\n'
    final=tables['pca_variance_summary'].copy()
    for c in ('mean_ratio','mean_cumulative'):final[c]=final['final_ratio' if c=='mean_ratio' else 'final_cumulative']
    finalcomp=pca_variance_comparison(final)[['방식','입력','첫째 성분(%)','둘째 성분(%)','셋째 성분(%)','첫째+둘째 누적(%)']]
    text+='### 마지막 기준월 2021-11 · 목표월 표시 2021-12\n\n'+markdown_table(finalcomp)+'\n'
    text+='### 왜·어떻게 계산했는가\n\n매월 적격·세 값 공통 유효 집합에서 표본표준편차로 표준화하고 규모 부호를 반전한 3열을 사용했다. 제한안은 원특징을 그달 공통 집합의 1·99 분위수로 제한한 뒤 다시 표준화했다. 월별 방식은 그달 행만, 과거 누적 방식은 그달까지의 월별 표준점수 행을 누적해 다시 추정했다. 누적 방식은 행별 동일가중이므로 종목이 많은 달이 더 큰 비중을 갖는다. 누적 자료를 추가로 재표준화하지 않았으며, 중심화한 표본 공분산의 고유값을 큰 순서로 정렬하고 총합으로 나눴다.\n\n'
    text+='각 행의 사용 종목 수·누적 행 수·누적 첫달과 마지막 달·미산출 사유는 `pca_variance_monthly.csv`, 전체기간 요약은 `pca_variance_summary.csv`, 연도별 요약은 `pca_variance_by_target_year.csv`에 저장했다. 첫 목표월 2001-01은 유효한 모멘텀이 없어 미산출로 남겼다.\n\n'
    text+='### 생성 그림 목록\n\n'+''.join('- `'+(folder/name).as_posix()+'`\n' for name in figures)
    text+='\n### 후속 그림 목록 · 별도 승인 필요\n\n- 선택한 성분의 월별 적재계수·안정성\n- 원값·제한값 주성분 점수와 기존 점수의 월별 순위 상관 비교\n- 선정 지수 및 차이의 평균 검정·신뢰구간\n\n작업 묶음은 앞으로 `h1_pca반영지수`, 그다음 지수 선정은 `h1_지수선정`처럼 큰 범주를 사용한다. 기존 실행은 옮기거나 이름을 바꾸지 않는다. 이 붙여넣기 문서 외 같은 용도 문서는 추가하지 않는다.\n'
    return text


def run_pca_variance(config,*,root=Path('.'),preconditions):
    settings=config['H1_PCA'];paths=config['PATHS'];started=datetime.now()
    if settings['approved_scope'] not in ('variance_only','structure_diagnostics') or settings['variants']!=['raw','winsor']:
        raise PermissionError('only_approved_pca_variance_scope')
    c=pd.read_parquet(root/paths['h1_components']);h=pd.read_parquet(root/paths['h1'])
    tables,checks=pca_variance_analysis(c,h)
    folder=root/paths['run_records']/settings['run_category']/started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():table.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    figures=draw_pca_variance(tables,folder)
    code=['src/features/h1.py','src/evaluation/h1_analysis.py','src/pipeline.py','src/contracts.py',
          'src/preprocessing/panel.py','src/features/observe.py']
    inputs=[paths[n] for n in ('h1_components','h1','mom','panel','eligibility','asof')]
    record={'status':'EXECUTED','stage':'pca_variance_only','started':started.isoformat(),
       'command':' '.join(sys.argv),'python':sys.version,'preconditions':preconditions,'approved_decisions':dict(settings),
       'input_sha256':{p:sha256(root/p) for p in inputs},'code_sha256':{p:sha256(root/p) for p in code},
       'config_sha256':hashlib.sha256(json.dumps(dict(config),ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
       'checks':checks,'table_rows':{k:len(t) for k,t in tables.items()},'figures':figures,
       'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()},
       'method':{'input':'monthly common eligible signed Z, ddof=1','variants':settings['variants'],
          'pca_a':'current month centered sample covariance','pca_b':'expanding rows through current base month, row equal weight; no extra standardization',
          'eigenvalues':'descending, eigvalsh; tiny negatives within 3*machine_epsilon*covariance_spectral_norm clipped to numerical zero',
          'summary':'equal-weight mean of valid monthly variance ratios','target_month':'label only; no target returns attached',
          'component_selection':None,'sign_rule':None,'score_combination':None,'ic_or_hac':'not executed'}}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=1,default=str),encoding='utf-8')
    note=root/config['H1_TEST']['notion_tables_path']
    old=note.read_text(encoding='utf-8') if note.is_file() else '';marker='\n\n## 주성분 설명분산 · 원값 유지와 1·99 분위수 제한'
    append_report(note,marker,pca_variance_report(tables,folder.relative_to(root),figures))
    return {'run_dir':folder.as_posix(),'checks':checks,'comparison':pca_variance_comparison(tables['pca_variance_summary']).to_dict(orient='records'),
            'figures':figures,'notion_tables_path':str(note),'scope':'variance_only; component selection pending'}


def pca_structure_analysis(components,equal_scores):
    """성분의 내용 진단. 가중치와 추정/당월 집합 상관을 별도 열로 유지한다."""
    from src.features.h1 import pca_a,pca_b,cross_sectional_zscore
    c=components[components.base_month.between('2000-12','2021-11')].reset_index(drop=True)
    raw=equal_scores[equal_scores.base_month.between('2000-12','2021-11')].reset_index(drop=True)
    pd.testing.assert_frame_equal(c[['stock_id','base_month','eligible']],raw[['stock_id','base_month','eligible']])
    inputs={'raw':raw,'winsor':cross_sectional_zscore(winsorize_1_99(c),columns=FEATURES)}
    frames=[]
    for variant,source in inputs.items():
        for fn in (pca_a,pca_b):
            f=fn(source,include_structure=True)
            f.insert(0,'variant',variant);frames.append(f)
    monthly=pd.concat(frames,ignore_index=True)
    valid=monthly[monthly.available].copy()
    summary_rows=[];stability=[]
    for scope,part in [('overall',valid),('2021',valid[valid.target_month.str.startswith('2021')])]:
        for (variant,method,pc),group in part.groupby(['variant','method','component'],sort=True):
            for feature in ('neg_size','bm','mom'):
                weight=group['weight_'+feature];fitcorr=group['fit_corr_'+feature];currentcorr=group['current_corr_'+feature]
                summary_rows.append({'scope':scope,'variant':variant,'method':method,'component':pc,'feature':feature,
                   'valid_months':len(group),'weight_mean':weight.mean(),'weight_sd':weight.std(ddof=1),
                   'weight_min':weight.min(),'weight_max':weight.max(),'positive_weight_fraction':weight.gt(0).mean(),
                   'fit_corr_mean':fitcorr.mean(),'fit_corr_min':fitcorr.min(),'fit_corr_max':fitcorr.max(),
                   'current_corr_mean':currentcorr.mean(),'current_corr_min':currentcorr.min(),'current_corr_max':currentcorr.max()})
            stability.append({'scope':scope,'variant':variant,'method':method,'component':pc,'valid_months':len(group),
                'mean_previous_axis_cosine':group.previous_axis_cosine.mean(),
                'min_previous_axis_cosine':group.previous_axis_cosine.min(),
                'min_eigenvalue_gap_ratio':group.nearest_eigenvalue_gap_ratio.min(),
                'unidentifiable_months':int((~group.eigenvector_identifiable).sum())})
    latest=valid[valid.base_month.eq(valid.base_month.max())].copy()
    weights=valid[['weight_neg_size','weight_bm','weight_mom']].to_numpy()
    ref=valid.display_reference_base_month
    correlations=valid[[c for c in valid if c.startswith('fit_corr_') and c!='fit_corr_reason']+
                       [c for c in valid if c.startswith('current_corr_') and c!='current_corr_reason']].to_numpy()
    sums=valid.groupby(['variant','method','base_month']).explained_variance_ratio.sum()
    checks={'unit_length_weights':bool(np.allclose(np.sum(weights**2,axis=1),1.,atol=1e-10)),
            'correlations_bounded':bool(np.nanmax(np.abs(correlations))<=1+1e-10),
            'variance_ratios_sum_to_one':bool(np.allclose(sums,1.,atol=1e-12)),
            'display_sign_uses_past_only':bool(ref[ref.notna()].lt(valid.loc[ref.notna(),'base_month']).all()),
            'display_axis_cosine_nonnegative':bool(valid.previous_axis_cosine.dropna().ge(-1e-10).all()),
            'fit_last_month_not_future':bool(valid.fit_last_base_month.le(valid.base_month).all()),
            'all_calendar_months_and_components_kept':len(monthly)==len(c.base_month.unique())*4*3,
            'no_targets_or_prediction_scores':not any(k in monthly for k in ('y','h1','score','ic','selected_component')),
            'orthogonality_eigen_equation_covariance_reconstruction':'checked inside each decomposition'}
    if not all(checks.values()):raise ValueError('pca_structure_checks_failed:'+str(checks))
    return {'pca_structure_monthly':monthly,'pca_feature_weights_summary':pd.DataFrame(summary_rows),
            'pca_axis_stability':pd.DataFrame(stability),'pca_structure_last_month':latest,
            'pca_structure_2021_monthly':monthly[monthly.target_month.str.startswith('2021')].copy()},checks


def draw_pca_structure(tables,folder):
    from src.features.observe import setup_matplotlib
    plt=setup_matplotlib();monthly=tables['pca_structure_monthly'];last=tables['pca_structure_last_month']
    order=[('raw','pca_a'),('winsor','pca_a'),('raw','pca_b'),('winsor','pca_b')]
    feature_labels=['-규모','가치','모멘텀'];files=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/name);plt.close(fig);files.append(name)
    for prefix,title,name in [('weight_','주성분 계산 가중치 · 마지막 기준월 2021-11','03_성분별_가중치.png'),
                               ('fit_corr_','특징과 성분의 상관 · 추정에 사용한 집합','04_특징과_성분상관.png')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for (variant,method),ax in zip(order,axes.flat):
            f=last[last.variant.eq(variant)&last.method.eq(method)].set_index('component').reindex(['PC1','PC2','PC3'])
            matrix=f[[prefix+'neg_size',prefix+'bm',prefix+'mom']].to_numpy()
            ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=12,color='white' if abs(matrix[i,j])>.65 else 'black')
            ax.set_xticks(range(3),feature_labels);ax.set_yticks(range(3),['첫째','둘째','셋째'])
            ax.set_title(PCA_METHOD_LABEL[method]+' · '+PCA_VARIANT_LABEL[variant]);ax.grid(False)
        fig.suptitle(title+'\n빨강: 양수 · 파랑: 음수 / 부호는 진단 표시용 정렬')
        save(fig,name)
    for k in (1,2,3):
        fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
        for (variant,method),ax in zip(order,axes.flat):
            f=monthly[monthly.variant.eq(variant)&monthly.method.eq(method)&monthly.component.eq(f'PC{k}')]
            for feature,label,color in zip(('neg_size','bm','mom'),feature_labels,COLORS):
                ax.plot(pd.to_datetime(f.target_month),f['weight_'+feature],color=color,label=label,lw=1.3)
            ax.axhline(0,color='#555555',lw=.8);ax.set_ylim(-1.05,1.05)
            ax.set_title(PCA_METHOD_LABEL[method]+' · '+PCA_VARIANT_LABEL[variant]);ax.set_xlabel('목표월(표시용)')
            ax.set_ylabel('계산 가중치');ax.legend(ncol=3,fontsize=8,loc='lower right')
        fig.suptitle(f'{k}번째 성분의 월별 가중치 · 직전 유효월과 표시 부호 정렬')
        save(fig,f'0{k+4}_성분{k}_월별가중치.png')
    return files


def pca_structure_report(tables,folder,figures):
    text='\n\n## 주성분 가중치·특징 상관·월별 변화 (2026-11-06)\n\n'
    text+='실행: `'+folder.as_posix()+'`\n\n'
    text+='승인된 진단만 실행했다. 사용할 성분·결합 가중치·최종 점수 부호는 미선정이며 수익률 성능 비교·가설검정은 미실행. 별도 테스트·임시 재실행은 큰 파트 종료 시까지 보류했다.\n\n'
    text+='### 읽는 기준\n\n계산 가중치는 세 표준특징을 어떻게 섞어 성분을 만드는지 나타내며, 제곱합이 1이고 합이 1인 비중은 아니다. 특징과 성분의 상관은 그 성분이 어떤 특징과 함께 움직이는지 나타낸다. 누적 방식의 추정 집합 상관은 그달까지의 누적 자료 기준이며, 당월 종목만의 상관도 별도로 저장했다. 설명분산은 세 특징의 변동을 담는 비율이며 수익률 설명력은 아니다.\n\n'
    text+='### 표시 부호와 축의 안정성\n\n첫 유효월은 가장 큰 절댓값 가중치를 양수로 표시하고, 이후 같은 분산 순서 성분을 직전 유효월 축과 내적이 양수가 되도록 표시한다. 미래 월·목표 수익률은 쓰지 않았다. 이 표시는 최종 H1 부호 규칙이 아니다. 성분 순서는 매월 설명분산 순서이며, 순서를 바꿔 추적하지 않는다. 고유값이 비슷하면 같은 순서의 축도 크게 변할 수 있다. 방향 유사도는 정렬 후 직전 축과의 내적으로, 1에 가까우면 같은 방향, 0에 가까우면 큰 방향 변화다.\n\n'
    last=tables['pca_structure_last_month']
    for (variant,method),f in last.groupby(['variant','method'],sort=True):
        text+='### 마지막 기준월 2021-11 · '+PCA_METHOD_LABEL[method]+' · '+PCA_VARIANT_LABEL[variant]+'\n\n'
        f=f.copy();f['분산(%)']=100*f.explained_variance_ratio
        columns={'component':'성분','weight_neg_size':'-규모 가중치','weight_bm':'가치 가중치','weight_mom':'모멘텀 가중치',
                 'fit_corr_neg_size':'-규모 상관(추정 집합)','fit_corr_bm':'가치 상관(추정 집합)','fit_corr_mom':'모멘텀 상관(추정 집합)'}
        text+=markdown_table(f[['component','weight_neg_size','weight_bm','weight_mom','fit_corr_neg_size','fit_corr_bm','fit_corr_mom','분산(%)']].rename(columns=columns))+'\n'
        if method=='pca_b':
            text+='당월 종목만의 상관:\n\n'+markdown_table(f[['component','current_corr_neg_size','current_corr_bm','current_corr_mom']].rename(columns={
              'component':'성분','current_corr_neg_size':'-규모 상관(당월)','current_corr_bm':'가치 상관(당월)','current_corr_mom':'모멘텀 상관(당월)'}))+'\n'
    text+='### 대표 연도 2021년 · 가중치와 상관의 월평균 및 범위\n\n'
    summary=tables['pca_feature_weights_summary'];f=summary[summary.scope.eq('2021')].copy()
    f['method']=f.method.map(PCA_METHOD_LABEL);f['variant']=f.variant.map(PCA_VARIANT_LABEL)
    f['feature']=f.feature.map({'neg_size':'-규모','bm':'가치','mom':'모멘텀'})
    text+=markdown_table(f[['method','variant','component','feature','weight_mean','weight_min','weight_max','fit_corr_mean','current_corr_mean']].rename(columns={
        'method':'방식','variant':'입력','component':'성분','feature':'특징','weight_mean':'가중치 월평균','weight_min':'가중치 최소','weight_max':'가중치 최대',
        'fit_corr_mean':'추정 집합 상관 월평균','current_corr_mean':'당월 상관 월평균'}))+'\n'
    text+='평균 가중치 자체는 특정 월의 주성분 가중치가 아니며 새 점수 산식으로 채택하지 않는다.\n\n'
    f=tables['pca_axis_stability'];f=f[f.scope.eq('overall')].copy()
    f['method']=f.method.map(PCA_METHOD_LABEL);f['variant']=f.variant.map(PCA_VARIANT_LABEL)
    text+='### 전체기간 축 방향 안정성\n\n'+markdown_table(f[['method','variant','component','valid_months','mean_previous_axis_cosine','min_previous_axis_cosine','min_eigenvalue_gap_ratio','unidentifiable_months']].rename(columns={
        'method':'방식','variant':'입력','component':'성분','valid_months':'유효 월 수','mean_previous_axis_cosine':'직전 축 유사도 평균',
        'min_previous_axis_cosine':'직전 축 유사도 최소','min_eigenvalue_gap_ratio':'최소 고유값 간격÷총분산','unidentifiable_months':'수치적으로 축 미식별 월 수'}))+'\n'
    text+='### 생성 그림 목록\n\n'+''.join('- `'+(folder/name).as_posix()+'`\n' for name in figures)
    text+='\n### 상세 표와 다음 단계\n\n월별 전체 계수·상관·분산은 `pca_structure_monthly.csv`, 특징별 요약은 `pca_feature_weights_summary.csv`, 축 안정성은 `pca_axis_stability.csv`, 마지막 달은 `pca_structure_last_month.csv`, 대표 2021년 월별 전체는 `pca_structure_2021_monthly.csv`에 기록했다. 모두 같은 `h1_pca반영지수` 범주다. 성분 선택·결합식·최종 부호 규칙을 정한 뒤 점수와 수익률 성능 비교를 진행해야 한다.\n'
    return text


def run_pca_structure(config,*,root=Path('.'),preconditions):
    settings=config['H1_PCA'];paths=config['PATHS'];started=datetime.now()
    if settings['approved_scope']!='structure_diagnostics':raise PermissionError('structure_diagnostics_only')
    c=pd.read_parquet(root/paths['h1_components']);h=pd.read_parquet(root/paths['h1'])
    tables,checks=pca_structure_analysis(c,h)
    checks['variance_recomputed_from_current_inputs']=True
    folder=root/paths['run_records']/settings['run_category']/started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():table.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    figures=draw_pca_structure(tables,folder)
    code=['src/features/h1.py','src/evaluation/h1_analysis.py','src/pipeline.py','src/contracts.py',
          'src/preprocessing/panel.py','src/features/observe.py']
    inputs=[paths[n] for n in ('h1_components','h1','mom','panel','eligibility','asof')]
    record={'status':'EXECUTED','stage':'pca_structure_diagnostics','started':started.isoformat(),'command':' '.join(sys.argv),
       'approved_decisions':dict(settings),'preconditions':preconditions,
       'input_sha256':{p:sha256(root/p) for p in inputs},'code_sha256':{p:sha256(root/p) for p in code},
       'checks':checks,'table_rows':{k:len(v) for k,v in tables.items()},'figures':figures,
       'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()},
       'method':{'input':'signed monthly common-eligible Z, ddof=1','weight_normalization':'unit Euclidean norm; sum is not constrained to one',
        'fit_correlation':'C_fit*w/sqrt(diag(C_fit)*(w^T*C_fit*w))',
        'current_correlation':'same covariance formula on current-month rows only; no targets',
        'display_sign':'first largest-absolute coefficient positive; then dot with previous same-ranked axis nonnegative; no future',
        'component_tracking':'variance rank only; no permutation; near eigenvalue ties can change the economic axis',
        'axis_identifiable':'nearest eigenvalue gap greater than numerical PSD tolerance, not a substantive stability threshold',
        'selection_or_scores_or_ic':'not executed'},
       'deferred_checks':{'related_tests':'deferred until h1_pca반영지수 part ends','full_tests':'deferred until part ends','temporary_reexecution':'deferred until part ends'}}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=1,default=str),encoding='utf-8')
    note=root/config['H1_TEST']['notion_tables_path']
    marker='\n\n## 주성분 가중치·특징 상관·월별 변화'
    append_report(note,marker,pca_structure_report(tables,folder.relative_to(root),figures))
    return {'run_dir':str(folder),'checks':checks,'last_month':tables['pca_structure_last_month'][[
        'variant','method','component','weight_neg_size','weight_bm','weight_mom','fit_corr_neg_size','fit_corr_bm','fit_corr_mom',
        'current_corr_neg_size','current_corr_bm','current_corr_mom','explained_variance_ratio']].to_dict(orient='records'),
        'figures':figures,'tests':'deferred until large part ends','notion_tables_path':str(note)}
