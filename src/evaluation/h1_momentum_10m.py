"""H1의 t-11~t-2 10개월 모멘텀 실험. 가용성·값과 분포를 단계별 관측.

입력은 기존 개발 패널과 확정 특징표다. 원본·기존 산출물을 보존하고
outputs/run_id/h1_모멘텀10개월/<시각>/에만 저장한다.
월별 가용성은 기준월 적격 집합, 연도는 다음 달력월 연도로 집계한다.
mom 표는 기존 MOM_COLUMNS 계약을 사용하며 표준화·목표 결합 전에 저장한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.contracts import MOM_COLUMNS
from src.features.h1 import calculate_momentum_11m
from src.preprocessing.panel import enforce_unique_stock_month, normalize_month, normalize_stock_id


RUN_CATEGORY = 'h1_모멘텀10개월'
ANALYSIS_START, ANALYSIS_END = '2000-12', '2021-11'


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()


def calculate_momentum_10m(returns, keys):
    """정확한 달력 t-11~t-2의 10개 원수익률 복리. 결측·오류는 사유와 보존."""
    enforce_unique_stock_month(returns,month_column='month')
    enforce_unique_stock_month(keys,month_column='base_month')
    normalize_stock_id(returns.stock_id.tolist());normalize_stock_id(keys.stock_id.tolist())
    normalize_month(returns.month.tolist());normalize_month(keys.base_month.tolist())
    if (not returns.month.between('2000-01','2021-12').all()
        or not keys.base_month.between('2000-01','2021-11').all()):
        raise PermissionError('momentum_10m_development_only')
    if returns.attrs.get('units',{}).get('return')!='decimal':raise ValueError('return_unit_must_be_decimal')
    if not pd.api.types.is_numeric_dtype(returns['return']):raise ValueError('numeric_returns_required')
    wide=returns.pivot(index='stock_id',columns='month',values='return')
    out=keys[['stock_id','base_month','eligible']].copy()
    out['mom']=np.nan;out['mom_available']=False;out['mom_valid_months']=0
    out['mom_exclusion_reason']=''
    for month,part in keys.groupby('base_month',sort=True):
        period=pd.Period(month,freq='M')
        window=[str(period+offset) for offset in range(-11,-1)]
        values=wide.reindex(index=part.stock_id,columns=window).to_numpy(dtype=float,na_value=np.nan)
        valid=np.isfinite(values)&(values>=-1)
        count=valid.sum(axis=1)
        invalid=((~np.isnan(values))&(~np.isfinite(values)|(values < -1))).any(axis=1)
        value=np.full(len(part),np.nan);full=count==10
        with np.errstate(over='ignore',invalid='ignore'):
            value[full]=np.prod(1+values[full],axis=1)-1
        overflow=full&~np.isfinite(value)
        reason=np.select([invalid,overflow,np.full(len(part),month<'2000-12'),~full],
            ['invalid_return_in_window','nonfinite_momentum','window_before_2000','incomplete_10_calendar_months'],'')
        available=reason==''
        out.loc[part.index,'mom']=np.where(available,value,np.nan)
        out.loc[part.index,'mom_available']=available
        out.loc[part.index,'mom_valid_months']=count
        out.loc[part.index,'mom_exclusion_reason']=reason
    return out[list(MOM_COLUMNS)]


def validate_calendar_calculation(returns, mom):
    """달력월 rolling10/shift2의 별도 계산으로 모든 행을 대조한다."""
    values=returns.pivot(index='month',columns='stock_id',values='return')
    months=pd.period_range('1999-02','2021-11',freq='M').astype(str)
    values=values.reindex(months)
    valid=values.where(np.isfinite(values)&values.ge(-1))
    rolling=(1+valid).rolling(10,min_periods=10).apply(np.prod,raw=True).shift(2)-1
    counts=valid.notna().astype(int).rolling(10,min_periods=1).sum().shift(2).fillna(0)
    idx=pd.MultiIndex.from_frame(mom[['base_month','stock_id']])
    expected=rolling.stack(future_stack=True).reindex(idx).to_numpy(float).copy()
    expected[mom.base_month.lt('2000-12').to_numpy()]=np.nan
    expected[~np.isfinite(expected)]=np.nan
    np.testing.assert_allclose(mom.mom.to_numpy(),expected,equal_nan=True,atol=1e-12,rtol=1e-12)
    expected_counts=counts.stack(future_stack=True).reindex(idx).fillna(0).to_numpy(int)
    np.testing.assert_array_equal(mom.mom_valid_months,expected_counts)


def availability_tables(components, old_mom, new_mom):
    keys=['stock_id','base_month']
    frame=components.copy()
    pd.testing.assert_frame_equal(frame[list(MOM_COLUMNS)],old_mom[list(MOM_COLUMNS)])
    merged=frame.merge(new_mom,on=keys,validate='one_to_one',suffixes=('_11m','_10m'))
    if not merged.eligible_11m.equals(merged.eligible_10m):raise ValueError('eligible_changed')
    analysis=merged[merged.base_month.between(ANALYSIS_START,ANALYSIS_END)].copy()
    analysis['target_month']=(pd.PeriodIndex(analysis.base_month,freq='M')+1).astype(str)
    analysis['target_year']=analysis.target_month.str[:4]
    eligible=analysis.eligible_11m.eq(True).fillna(False)
    other=analysis.size_available & analysis.bm_available
    analysis['eligible_count']=eligible.astype(int)
    for variant in ('11m','10m'):
        analysis['mom_count_'+variant]=(eligible&analysis['mom_available_'+variant]).astype(int)
        analysis['joint_count_'+variant]=(eligible&other&analysis['mom_available_'+variant]).astype(int)
    analysis['mom_gained']=((analysis.mom_count_10m==1)&(analysis.mom_count_11m==0)).astype(int)
    analysis['mom_lost']=((analysis.mom_count_10m==0)&(analysis.mom_count_11m==1)).astype(int)
    analysis['joint_gained']=((analysis.joint_count_10m==1)&(analysis.joint_count_11m==0)).astype(int)
    analysis['joint_lost']=((analysis.joint_count_10m==0)&(analysis.joint_count_11m==1)).astype(int)
    counts=['eligible_count','mom_count_11m','mom_count_10m','joint_count_11m','joint_count_10m',
            'mom_gained','mom_lost','joint_gained','joint_lost']
    monthly=analysis.groupby(['base_month','target_month','target_year'],as_index=False)[counts].sum()
    for variant in ('11m','10m'):
        valid_months={}
        for month,part in analysis.groupby('base_month',sort=True):
            member=part['joint_count_'+variant].eq(1)
            x=part.loc[member,['size','bm','mom_'+variant]].to_numpy(float)
            sd=x.std(axis=0,ddof=1) if len(x)>=2 else np.full(3,np.nan)
            valid_months[month]=bool(len(x)>=2 and np.isfinite(sd).all() and (sd>0).all())
        monthly['h1_month_available_'+variant]=monthly.base_month.map(valid_months)
        monthly['mom_available_fraction_'+variant]=monthly['mom_count_'+variant]/monthly.eligible_count
        monthly['joint_available_fraction_'+variant]=monthly['joint_count_'+variant]/monthly.eligible_count
    yearly=monthly.groupby('target_year',as_index=False)[counts].sum()
    sizes=monthly.groupby('target_year').size()
    yearly['calendar_months']=yearly.target_year.map(sizes)
    for variant in ('11m','10m'):
        yearly['h1_available_months_'+variant]=yearly.target_year.map(
            monthly.groupby('target_year')['h1_month_available_'+variant].sum())
        yearly['mom_available_fraction_'+variant]=yearly['mom_count_'+variant]/yearly.eligible_count
        yearly['joint_available_fraction_'+variant]=yearly['joint_count_'+variant]/yearly.eligible_count
    reason_parts=[]
    for variant in ('11m','10m'):
        f=analysis.loc[eligible,['target_year','mom_exclusion_reason_'+variant]].copy()
        f.columns=['target_year','reason'];f['reason']=f.reason.replace('','available')
        reason_parts.append(f.groupby(['target_year','reason'],as_index=False).size().rename(columns={'size':'rows'}).assign(variant=variant))
    reasons=pd.concat(reason_parts,ignore_index=True)
    changed=analysis[(analysis.mom_gained+analysis.mom_lost+analysis.joint_gained+analysis.joint_lost)>0].copy()
    changed=changed[['stock_id','base_month','target_month','target_year','eligible_11m',
        'mom_available_11m','mom_available_10m','mom_valid_months_11m','mom_valid_months_10m',
        'mom_exclusion_reason_11m','mom_exclusion_reason_10m','mom_gained','mom_lost','joint_gained','joint_lost']]
    changed=changed.rename(columns={'eligible_11m':'eligible'})
    totals={key:int(monthly[key].sum()) for key in counts}
    totals.update({f'h1_available_months_{v}':int(monthly['h1_month_available_'+v].sum()) for v in ('11m','10m')})
    if totals['mom_lost'] or totals['joint_lost']:raise ValueError('unexpected_availability_loss_requires_review')
    checks={'same_keys_and_eligibility':True,'old_mom_matches_components':True,
        'analysis_calendar_months_252':len(monthly)==252,
        'old_common_feature_cells_165573':totals['joint_count_11m']==165573,
        'no_old_available_rows_lost':True,
        'target_month_is_calendar_label_only':True}
    if not all(checks.values()):raise ValueError('availability_checks_failed')
    return {'availability_monthly':monthly,'availability_by_target_year':yearly,
            'exclusion_reasons_by_target_year':reasons},changed,totals,checks


def run(root, config, *, execute=False):
    allowed={'panel':'data/interim/panel.parquet','h1_components':'data/processed/h1_components.parquet',
             'mom':'data/processed/mom.parquet'}
    for key,expected in allowed.items():
        if Path(config['PATHS'][key]).as_posix()!=expected:raise PermissionError('fixed_development_inputs_required')
    from src.pipeline import check_features_current
    record=check_features_current(config,root=root)
    inputs={str(Path(config['PATHS'][name])):sha256(root/config['PATHS'][name]) for name in ('panel','h1_components','mom')}
    panel=pd.read_parquet(root/allowed['panel'],columns=['stock_id','month','return'])
    panel.attrs['units']=config['UNITS']
    components=pd.read_parquet(root/allowed['h1_components'])
    old=pd.read_parquet(root/allowed['mom'])
    # 원자료 전체를 받기 전에 고정 개발 경로를 제한하고, 읽은 기간도 계약으로 검사한다.
    keys=components[['stock_id','base_month','eligible']]
    baseline=calculate_momentum_11m(panel,keys[['stock_id','base_month']])
    baseline=baseline.merge(keys,on=['stock_id','base_month'],validate='one_to_one')[list(MOM_COLUMNS)]
    pd.testing.assert_frame_equal(old,baseline,check_exact=False,rtol=1e-12,atol=1e-12)
    mom=calculate_momentum_10m(panel,keys)
    validate_calendar_calculation(panel,mom)
    tables,changed,totals,checks=availability_tables(components,old,mom)
    checks.update(independent_calendar_product_matches_rolling=True,baseline_11m_reproduced=True,
                  full_feature_grid_preserved=len(mom)==len(old),no_y_attached='y' not in mom)
    if not all(checks.values()):raise ValueError('momentum_availability_checks_failed')
    result={'execute':execute,'stage':'availability','totals':totals,'checks':checks}
    if not execute:return result
    folder=root/config['PATHS']['run_records']/RUN_CATEGORY/datetime.now().strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,table in tables.items():table.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    mom.to_parquet(folder/'mom_10m.parquet',index=False)
    changed.to_parquet(folder/'availability_changes.parquet',index=False)
    code=['src/evaluation/h1_momentum_10m.py','src/features/h1.py','src/contracts.py',
          'src/preprocessing/panel.py','config/experiment.yaml']
    for name,value in inputs.items():
        if sha256(root/name)!=value:raise RuntimeError('source_changed_during_availability:'+name)
    run_record={'status':'EXECUTED','stage':'momentum_10m_availability','command':' '.join(sys.argv),
        'experiment':{'window_start_offset':-11,'window_end_offset':-2,'required_months':10,
          'baseline_offsets':[-11,-1],'baseline_months':11,'analysis_base_months':[ANALYSIS_START,ANALYSIS_END],
          'year_aggregation':'next calendar target month year','denominator':'base-month eligible=True rows',
          'available_momentum':'all required returns finite and >=-1, finite compounded result',
          'common_features':'eligible and size_available and bm_available and mom_available',
          'h1_available_month':'common n>=2 and all three sample standard deviations finite and positive',
          'future_candidates':['equal','equal_winsor','pca_a_pc1','pca_b_pc1','pca_a_pc1_winsor','pca_b_pc1_winsor'],
          'return_relationship_analysis':'pending next approval','p_values':'not calculated'},
        'source_features':'data/processed/h1_components.parquet',
        'ingest':record.get('preconditions',{}),
        'input_sha256':inputs,'code_sha256':{p:sha256(root/p) for p in code},
        'counts':{'rows':len(mom),'stocks':mom.stock_id.nunique(),'base_months':mom.base_month.nunique(),
                  'changed_rows':len(changed),**totals},'checks':checks,
        'table_rows':{name:len(table) for name,table in tables.items()},'figures':[],
        'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir()},
        'deferred_checks':{'related_and_full_tests':'end of the momentum-10-month experiment',
                           'temporary_reexecution':'end of the experiment'}}
    (folder/'run.json').write_text(json.dumps(run_record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return {**result,'run_dir':str(folder)}


def load_availability_inputs(root, config, source_run=None):
    """가용성 실행의 입력·산출물 해시를 확인하고 저장 모멘텀을 재사용한다."""
    from src.pipeline import check_features_current
    if source_run:raise ValueError('source_run_removed_use_common_data_inputs')
    record=check_features_current(config,root=root)
    hashes={config['PATHS'][name]:sha256(root/config['PATHS'][name]) for name in ('panel','h1_components','mom')}
    source=root/'data/processed'
    components=pd.read_parquet(root/config['PATHS']['h1_components'])
    old=pd.read_parquet(root/config['PATHS']['mom'])
    panel=pd.read_parquet(root/config['PATHS']['panel'],columns=['stock_id','month','return'])
    panel.attrs['units']=config['UNITS']
    new=calculate_momentum_10m(panel,components[['stock_id','base_month','eligible']])
    for f in (components,old,new):enforce_unique_stock_month(f,month_column='base_month')
    pd.testing.assert_frame_equal(components[list(MOM_COLUMNS)],old[list(MOM_COLUMNS)])
    pd.testing.assert_frame_equal(old[['stock_id','base_month','eligible']],new[['stock_id','base_month','eligible']])
    frame=components.merge(new,on=['stock_id','base_month'],validate='one_to_one',suffixes=('_11m','_10m'))
    frame=frame[frame.base_month.between(ANALYSIS_START,ANALYSIS_END)].copy()
    frame['target_month']=(pd.PeriodIndex(frame.base_month,freq='M')+1).astype(str)
    frame['target_year']=frame.target_month.str[:4]
    frame['eligible']=frame.eligible_11m.eq(True).fillna(False)
    return source,record,hashes,frame


def value_statistics(values):
    """관측 행 기준 부호·분위수. 0은 정확히 0인 값으로 구분한다."""
    s=pd.Series(values,dtype=float)
    if not np.isfinite(s).all():raise ValueError('distribution_requires_finite_values')
    n=len(s)
    result={'n':n,'positive':int(s.gt(0).sum()),'negative':int(s.lt(0).sum()),'zero':int(s.eq(0).sum())}
    for sign in ('positive','negative','zero'):result[sign+'_fraction']=result[sign]/n if n else np.nan
    result.update(mean=float(s.mean()),sample_sd=float(s.std(ddof=1)),min=float(s.min()),max=float(s.max()),
                  skewness=float(s.skew()) if n>=3 else np.nan,
                  excess_kurtosis=float(s.kurt()) if n>=4 else np.nan)
    for q in (.01,.05,.25,.5,.75,.95,.99):result['q'+str(round(q*100)).zfill(2)]=float(s.quantile(q)) if n else np.nan
    result['reason']='' if n else 'empty_scope'
    if result['positive']+result['negative']+result['zero']!=n:raise ValueError('sign_count_mismatch')
    return result


def distribution_tables(frame):
    """월별·연도별·전체 값 통계와 공통 집합 내 순위 변화. 수익률 목표 미결합."""
    eligible=frame.eligible
    both=eligible&frame.mom_available_11m&frame.mom_available_10m
    common=both&frame.size_available&frame.bm_available
    gained=eligible&~frame.mom_available_11m&frame.mom_available_10m
    scope_masks={'own':None,'paired_momentum':both,'paired_h1_common':common,'newly_available':gained}
    monthly=[];annual=[];overall=[];rank_rows=[];transition_rows=[];changed_rows=[];difference_rows=[]
    for scope,mask in scope_masks.items():
        for variant in ('11m','10m'):
            member=(eligible&frame['mom_available_'+variant]) if mask is None else (mask&frame['mom_available_'+variant])
            values=frame.loc[member]
            overall.append({'scope':scope,'variant':variant,**value_statistics(values['mom_'+variant])})
            for (month,target),part in frame.groupby(['base_month','target_month'],sort=True):
                selected=part.loc[member.loc[part.index],'mom_'+variant]
                monthly.append({'scope':scope,'variant':variant,'base_month':month,'target_month':target,
                                **value_statistics(selected)})
            for year,part in frame.groupby('target_year',sort=True):
                selected=part.loc[member.loc[part.index],'mom_'+variant]
                annual.append({'scope':scope,'variant':variant,'target_year':year,**value_statistics(selected)})
    for scope,mask in (('paired_momentum',both),('paired_h1_common',common)):
        values=frame.loc[mask].copy()
        difference=values.mom_10m-values.mom_11m
        difference_rows.append({'scope':scope,**value_statistics(difference)})
        values['sign_11m']=np.sign(values.mom_11m).astype(int)
        values['sign_10m']=np.sign(values.mom_10m).astype(int)
        for before in (-1,0,1):
            for after in (-1,0,1):
                count=int((values.sign_11m.eq(before)&values.sign_10m.eq(after)).sum())
                transition_rows.append({'scope':scope,'sign_11m':before,'sign_10m':after,'n':count,
                                        'fraction':count/len(values) if len(values) else np.nan})
        selected=values[values.sign_11m.ne(values.sign_10m)][
            ['stock_id','base_month','target_month','mom_11m','mom_10m','sign_11m','sign_10m']].copy()
        selected['scope']=scope;changed_rows.append(selected)
        for (month,target),part in frame.groupby(['base_month','target_month'],sort=True):
            part=part.loc[mask.loc[part.index]].copy();n=len(part)
            reason='too_few_pairs' if n<2 else ''
            if not reason and (part.mom_11m.nunique()<2 or part.mom_10m.nunique()<2):reason='constant_momentum'
            corr=np.nan;mean_shift=max_shift=np.nan
            if not reason:
                ranks=part[['mom_11m','mom_10m']].rank(method='average')
                corr=float(np.corrcoef(ranks.to_numpy(),rowvar=False)[0,1])
                shift=(ranks.mom_10m-ranks.mom_11m).abs()
                mean_shift=float(shift.mean());max_shift=float(shift.max())
            oldsign=np.sign(part.mom_11m);newsign=np.sign(part.mom_10m)
            rank_rows.append({'scope':scope,'base_month':month,'target_month':target,'n':n,
                'rank_spearman':corr,'mean_abs_rank_change':mean_shift,'max_abs_rank_change':max_shift,
                'positive_to_negative':int(((oldsign>0)&(newsign<0)).sum()),
                'negative_to_positive':int(((oldsign<0)&(newsign>0)).sum()),
                'sign_changed':int((oldsign!=newsign).sum()),
                'sign_changed_fraction':float((oldsign!=newsign).mean()) if n else np.nan,'reason':reason})
    ranks=pd.DataFrame(rank_rows)
    rank_summary=ranks.groupby('scope',as_index=False).agg(
        valid_months=('rank_spearman','count'),mean_monthly_rank_spearman=('rank_spearman','mean'),
        min_monthly_rank_spearman=('rank_spearman','min'),max_monthly_rank_spearman=('rank_spearman','max'),
        mean_monthly_abs_rank_change=('mean_abs_rank_change','mean'),
        mean_monthly_sign_changed_fraction=('sign_changed_fraction','mean'))
    tables={'distribution_overall':pd.DataFrame(overall),'distribution_monthly':pd.DataFrame(monthly),
        'distribution_by_target_year':pd.DataFrame(annual),'sign_transitions':pd.DataFrame(transition_rows),
        'rank_changes_monthly':ranks,'rank_changes_summary':rank_summary,
        'momentum_difference_summary':pd.DataFrame(difference_rows)}
    if int(both.sum())!=167576 or int(common.sum())!=165573 or int(gained.sum())!=181:
        raise ValueError('stage1_memberships_changed')
    for scope,count in (('paired_momentum',int(both.sum())),('paired_h1_common',int(common.sum()))):
        if sum(r['n'] for r in transition_rows if r['scope']==scope)!=count:raise ValueError('transition_counts_mismatch')
    return tables,pd.concat(changed_rows,ignore_index=True)


def draw_distribution(tables, folder):
    """월별 부호·분포·순위 변화를 한 그림으로 확인한다."""
    from src.features.observe import setup_matplotlib
    plt=setup_matplotlib();fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    colors={'11m':'#2a78d6','10m':'#eb6834'}
    for variant,label in (('11m','기존 11개월'),('10m','신규 10개월')):
        part=tables['distribution_monthly'].loc[lambda x:x.scope.eq('paired_momentum')&x.variant.eq(variant)]
        dates=pd.to_datetime(part.target_month)
        axes[0,0].plot(dates,part.positive_fraction*100,label=label,color=colors[variant])
        axes[1,0].plot(dates,part.q50,label=label,color=colors[variant])
        axes[1,0].fill_between(dates,part.q05,part.q95,color=colors[variant],alpha=.12)
    for scope,label,color in (('paired_momentum','모멘텀 공통 집합','#2a78d6'),('paired_h1_common','H1 공통 집합','#1baf7a')):
        part=tables['rank_changes_monthly'].loc[lambda x:x.scope.eq(scope)]
        axes[0,1].plot(pd.to_datetime(part.target_month),part.rank_spearman,label=label,color=color)
    part=tables['rank_changes_monthly'].loc[lambda x:x.scope.eq('paired_momentum')]
    axes[1,1].plot(pd.to_datetime(part.target_month),part.sign_changed_fraction*100,color='#7656a5')
    for ax,title,ylabel in zip(axes.flat,
            ['양수 모멘텀 관측 비율','11개월·10개월 점수 순위 상관','모멘텀 중앙값과 5~95 분위수','모멘텀 부호가 바뀐 관측 비율'],
            ['양수 관측 비율(%)','Spearman 상관','누적수익률(소수 단위)','부호 변경 비율(%)']):
        ax.set_title(title);ax.set_ylabel(ylabel);ax.set_xlabel('목표월');ax.tick_params(axis='x',rotation=30)
        handles,_=ax.get_legend_handles_labels()
        if handles:ax.legend(fontsize=8)
    fig.suptitle('10개월 모멘텀: 같은 종목·기준월에서 부호·분포·순위 비교')
    fig.tight_layout();name='01_모멘텀부호_분포_순위변화.png'
    fig.savefig(folder/name,dpi=160);plt.close(fig)
    return [name]


def run_distribution(root, config, *, execute=False, source_run=None):
    source,prior,hashes,frame=load_availability_inputs(root,config,source_run)
    tables,changed=distribution_tables(frame)
    # 제거한 t-1 수익률을 다시 곱해 기존 11개월 값과 대조한다.
    panel=pd.read_parquet(root/'data/interim/panel.parquet',columns=['stock_id','month','return'])
    if not panel.month.between('2000-01','2021-12').all():raise PermissionError('development_returns_only')
    paired=frame[frame.eligible&frame.mom_available_11m&frame.mom_available_10m].copy()
    paired['excluded_month']=(pd.PeriodIndex(paired.base_month,freq='M')-1).astype(str)
    audit=paired.merge(panel.rename(columns={'month':'excluded_month'}),on=['stock_id','excluded_month'],validate='many_to_one')
    expected=(1+audit.mom_10m)*(1+audit['return'])-1
    np.testing.assert_allclose(audit.mom_11m,expected,rtol=1e-12,atol=1e-12)
    checks={'previous_availability_inputs_and_outputs_unchanged':True,'same_keys_eligibility_and_scopes':True,
        'sign_counts_exhaust_rows':True,'sign_transitions_exhaust_paired_rows':True,
        'excluded_month_return_reconstructs_11m':True,
        'no_target_return_attached':all('y' not in t for t in tables.values()),
        'no_ic_or_pvalue_calculation':all(not any(k in t for k in ('ic','pvalue','p_two_sided')) for t in tables.values())}
    result={'execute':execute,'stage':'distribution','overall':tables['distribution_overall'].to_dict(orient='records'),
            'rank_summary':tables['rank_changes_summary'].to_dict(orient='records'),
            'transitions':tables['sign_transitions'].to_dict(orient='records'),'checks':checks}
    if not execute:return result
    folder=root/config['PATHS']['run_records']/RUN_CATEGORY/datetime.now().strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,t in tables.items():t.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    changed.to_parquet(folder/'sign_changed_rows.parquet',index=False)
    figures=draw_distribution(tables,folder)
    for name,value in hashes.items():
        if sha256(root/name)!=value:raise RuntimeError('source_changed_during_distribution:'+name)
    code=['src/evaluation/h1_momentum_10m.py','src/features/observe.py','src/contracts.py','config/experiment.yaml']
    record={'status':'EXECUTED','stage':'momentum_10m_distribution','command':' '.join(sys.argv),
        'source_availability_run':str(source.relative_to(root)),
        'input_sha256':hashes,'code_sha256':{p:sha256(root/p) for p in code},'checks':checks,
        'scopes':{'own':'eligible and available separately for each window',
            'paired_momentum':'eligible and both momentum windows available: 167576 rows',
            'paired_h1_common':'paired momentum and Size/BM available: 165573 rows',
            'newly_available':'eligible and new momentum only: 181 rows; old distribution empty'},
        'method':{'signs':'exact >0, <0, ==0; counts are stock-month observations',
            'statistics':'pooled stock-month values within each month, target year or overall; ddof=1 SD; linear quantiles',
            'rank_spearman':'average tied ranks then Pearson, within same month and scope; no pvalue routine',
            'rank_summary':'valid monthly correlations and mean absolute rank movements equally averaged across months',
            'pair_identity_max_error':float(np.max(np.abs(audit.mom_11m-expected))),
            'target_month':'calendar label only; return relationship and IC pending'},
        'table_rows':{name:len(t) for name,t in tables.items()},'sign_changed_rows':len(changed),
        'figures':figures,'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir()},
        'deferred_checks':{'related_and_full_tests':'large experiment end','temporary_reexecution':'large experiment end'}}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return {**result,'run_dir':str(folder),'figures':figures}


def run_structure(root, config, *, execute=False, source_run=None, persist=True):
    """확정 10개월 모멘텀으로 월별 표준화와 일반 PC1 구조만 계산한다."""
    from src.features.h1 import calculate_h1_score, winsorize_1_99, pca_a, pca_b
    source,prior,hashes,_=load_availability_inputs(root,config,source_run)
    components=pd.read_parquet(root/'data/processed/h1_components.parquet')
    panel=pd.read_parquet(root/config['PATHS']['panel'],columns=['stock_id','month','return'])
    panel.attrs['units']=config['UNITS']
    new=calculate_momentum_10m(panel,components[['stock_id','base_month','eligible']])
    for col in ('mom','mom_available','mom_valid_months','mom_exclusion_reason'):
        components[col]=new[col].to_numpy(copy=True)
    sources={'raw':calculate_h1_score(components),
             'winsor':calculate_h1_score(winsorize_1_99(components))}
    weights=[];standardized=[];score_parts=[];audits=[]
    zcols=['z_size','z_bm','z_mom'];wcols=['weight_neg_size','weight_bm','weight_mom']
    for variant,h in sources.items():
        analysis=h[h.base_month.between(ANALYSIS_START,ANALYSIS_END)]
        valid=analysis[analysis.h1_available]
        for month,part in valid.groupby('base_month',sort=True):
            x=part[zcols].to_numpy(float)
            np.testing.assert_allclose(x.mean(axis=0),0,atol=1e-11)
            np.testing.assert_allclose(x.std(axis=0,ddof=1),1,atol=1e-11)
            np.testing.assert_allclose(part.h1,(-x[:,0]+x[:,1]+x[:,2])/3,atol=1e-12)
            audits.append({'variant':variant,'base_month':month,'n':len(part),
                'max_abs_z_mean':float(np.max(np.abs(x.mean(axis=0)))),
                'max_abs_z_sd_error':float(np.max(np.abs(x.std(axis=0,ddof=1)-1)))})
        structure=pd.concat([pca_a(analysis,include_structure=True),pca_b(analysis,include_structure=True)],ignore_index=True)
        structure=structure[structure.component.eq('PC1')].copy()
        structure['variant']=variant;weights.append(structure)
        standardized.append(h.assign(variant=variant))
        equal=h[['stock_id','base_month','eligible','h1','h1_available','h1_exclusion_reason']].rename(
            columns={'h1':'score','h1_available':'score_available','h1_exclusion_reason':'score_exclusion_reason'})
        score_parts.append(equal.assign(candidate='equal'+('_winsor' if variant=='winsor' else '')))
        cumulative_sum=np.zeros(3);cumulative_n=0
        for month,part in analysis.groupby('base_month',sort=True):
            x=part[zcols].to_numpy(float).copy();x[:,0]*=-1
            member=np.isfinite(x).all(axis=1);m=int(member.sum())
            cumulative_sum+=x[member].sum(axis=0);cumulative_n+=m
            for method in ('pca_a','pca_b'):
                w=structure[structure.method.eq(method)&structure.base_month.eq(month)].iloc[0]
                score=np.full(len(part),np.nan)
                if w.available:
                    if w.fit_last_base_month>month:raise ValueError('future_pc1_fit')
                    expected_n=m if method=='pca_a' else cumulative_n
                    if int(w.fit_n)!=expected_n:raise ValueError('pc1_fit_count')
                    vector=w[wcols].to_numpy(float)
                    np.testing.assert_allclose(vector@vector,1,atol=1e-12)
                    center=x[member].mean(axis=0) if method=='pca_a' else cumulative_sum/cumulative_n
                    score[member]=(x[member]-center)@vector
                    if method=='pca_a':np.testing.assert_allclose(np.var(score[member],ddof=1),w.eigenvalue,atol=1e-10)
                f=part[['stock_id','base_month','eligible']].copy()
                f['score']=score;f['score_available']=np.isfinite(score)
                f['score_exclusion_reason']=np.where(f.score_available,'',part.h1_exclusion_reason)
                f['candidate']=method+'_pc1'+('_winsor' if variant=='winsor' else '')
                score_parts.append(f)
        for method in ('pca_a','pca_b'):
            f=h[h.base_month<ANALYSIS_START][['stock_id','base_month','eligible']].copy()
            f['candidate']=method+'_pc1'+('_winsor' if variant=='winsor' else '')
            f['score']=np.nan;f['score_available']=False;f['score_exclusion_reason']='warmup_history_only'
            score_parts.append(f)
    weights=pd.concat(weights,ignore_index=True)
    summary=[];sign_rows=[]
    for (variant,method),part in weights[weights.available].groupby(['variant','method'],sort=True):
        row={'variant':variant,'method':method,'valid_months':len(part),
             'mean_explained_variance_ratio':part.explained_variance_ratio.mean(),
             'min_explained_variance_ratio':part.explained_variance_ratio.min(),
             'max_explained_variance_ratio':part.explained_variance_ratio.max()}
        for col in wcols:
            row['mean_'+col]=part[col].mean()
            sign_rows.append({'variant':variant,'method':method,'feature':col,'valid_months':len(part),
                'positive_months':int(part[col].gt(0).sum()),'negative_months':int(part[col].lt(0).sum()),
                'zero_months':int(part[col].eq(0).sum()),
                'sign_switches':int((np.sign(part[col].to_numpy())[1:]!=np.sign(part[col].to_numpy())[:-1]).sum()),
                'sample_sd':part[col].std(ddof=1),'min':part[col].min(),'max':part[col].max()})
        summary.append(row)
    scores=pd.concat(score_parts,ignore_index=True).sort_values(['candidate','stock_id','base_month']).reset_index(drop=True)
    if scores.duplicated(['candidate','stock_id','base_month']).any():raise ValueError('duplicate_scores')
    for name,part in scores.groupby('candidate',sort=True):
        if len(part)!=len(components):raise ValueError('full_grid_not_preserved')
        if int(part.score_available.sum())!=int(sources['raw'].h1_available.sum()):raise ValueError('candidate_members_differ')
        if not part.loc[~part.score_available,'score_exclusion_reason'].ne('').all():raise ValueError('missing_score_reason')
    pd.testing.assert_series_equal(sources['raw'].h1_available,sources['winsor'].h1_available)
    tables={'pc1_structure_monthly':weights,'pc1_structure_summary':pd.DataFrame(summary),
        'pc1_weight_signs':pd.DataFrame(sign_rows),'standardization_audit':pd.DataFrame(audits)}
    checks={'prior_inputs_outputs_unchanged':True,'z_mean_zero_sample_sd_one':True,'equal_formula':True,
        'raw_winsor_same_members':True,'six_candidates_full_grid':True,'unit_weights_eigen_equation':True,
        'pca_a_projected_variance':True,'pca_b_fit_through_current_month_only':True,'no_target_ic_pvalue':True}
    result={'stage':'structure','execute':execute,'summary':summary,'weight_signs':sign_rows,'checks':checks}
    if not persist:return {'scores':scores,'standardized':pd.concat(standardized,ignore_index=True)}
    if not execute:return result
    folder=root/config['PATHS']['run_records']/RUN_CATEGORY/datetime.now().strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True,exist_ok=False)
    for name,t in tables.items():t.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    pd.concat(standardized,ignore_index=True).to_parquet(folder/'standardized_10m.parquet',index=False)
    scores.to_parquet(folder/'index_scores_10m.parquet',index=False)
    for name,value in hashes.items():
        if sha256(root/name)!=value:raise RuntimeError('source_changed_during_structure:'+name)
    code=['src/evaluation/h1_momentum_10m.py','src/features/h1.py','src/contracts.py','config/experiment.yaml']
    record={'status':'EXECUTED','stage':'momentum_10m_structure','command':' '.join(sys.argv),
        'source_availability_run':str(source.relative_to(root)),'input_sha256':hashes,
        'code_sha256':{p:sha256(root/p) for p in code},'checks':checks,
        'method':{'standardization':'eligible finite Size/BM/10m momentum common population each month; ddof=1 before targets',
            'winsor':'monthly common raw features clipped at linear q01/q99, then restandardized',
            'pca_a':'current monthly signed Z sample covariance, PC1 centered at current fit mean',
            'pca_b':'expanding signed monthly Z rows through current month, row equally weighted, centered at expanding fit mean',
            'sign':'first valid month largest absolute coefficient positive; later align dot product with previous PC1',
            'features':['-Z(Size)','Z(B/M)','Z(Mom10m)'],'summary':'monthly statistics equally averaged; PC1 only',
            'targets':'not attached; IC and pvalues pending'},
        'table_rows':{name:len(t) for name,t in tables.items()},'score_rows':len(scores),
        'summary':summary,'figures':[], 'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir()},
        'deferred_checks':{'related_and_full_tests':'large experiment end','temporary_reexecution':'large experiment end'}}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return {**result,'run_dir':str(folder)}


def run_ic(root, config, *, execute=False, source_run=None):
    """저장 점수를 같은 종목·월의 목표와 비교한다. p값은 계산하지 않는다."""
    from src.preprocessing.panel import join_training_targets
    from src.features.h1 import calculate_h1_score, winsorize_1_99
    _,_,hashes,_=load_availability_inputs(root,config,source_run)
    from src.evaluation.h1_analysis import run_pca_pc1_ic
    category=root/config['PATHS']['run_records']/RUN_CATEGORY
    new=run_structure(root,config,persist=False)['scores']
    ordinary_data=run_pca_pc1_ic(config,root=root,preconditions={},persist=False)
    old=ordinary_data['scores']
    h=pd.read_parquet(root/'data/processed/h1.parquet')
    c=pd.read_parquet(root/'data/processed/h1_components.parquet')
    for name,part in [('equal',h),('equal_winsor',calculate_h1_score(winsorize_1_99(c)))]:
        f=part[['stock_id','base_month','eligible','h1','h1_available','h1_exclusion_reason']].rename(
            columns={'h1':'score','h1_available':'score_available','h1_exclusion_reason':'score_exclusion_reason'})
        old=pd.concat([old,f.assign(candidate=name)],ignore_index=True)
    candidates=sorted(new.candidate.unique())
    if sorted(old.candidate.unique())!=candidates or len(candidates)!=6:raise ValueError('six_candidate_names_required')
    key=['stock_id','base_month']
    def wide_scores(scores):
        if scores.duplicated(['candidate',*key]).any():raise ValueError('duplicate_score_keys')
        if not scores.score_available.eq(np.isfinite(scores.score)).all():raise ValueError('score_availability_mismatch')
        table=scores.pivot(index=key,columns='candidate',values='score').reset_index()
        return table[table.base_month.between(ANALYSIS_START,ANALYSIS_END)].copy()
    a=wide_scores(new);b=wide_scores(old)
    panel=pd.read_parquet(root/config['PATHS']['panel'],columns=['stock_id','month','return'])
    panel.attrs['units']=config['UNITS']
    joined=join_training_targets(a,panel)
    joined=joined.merge(b.rename(columns={k:k+'_11m' for k in candidates}),on=key,validate='one_to_one')
    own=np.isfinite(joined[candidates]).all(axis=1)&np.isfinite(joined.y)
    paired=own&np.isfinite(joined[[k+'_11m' for k in candidates]]).all(axis=1)
    months=sorted(joined.target_month.unique());monthly=[];audit=[]
    for scope,mask in [('new_10m_common',own),('paired_10m_11m',paired)]:
        for month,part in joined.groupby('target_month',sort=True):
            selected=part[mask.loc[part.index]]
            audit.append({'sample_scope':scope,'target_month':month,'base_month':part.base_month.iloc[0],
                'total_rows':len(part),'n':len(selected),'excluded_rows':len(part)-len(selected)})
            for window in (['10m'] if scope=='new_10m_common' else ['10m','11m']):
                for candidate in candidates:
                    col=candidate+('_11m' if window=='11m' else '')
                    reason='too_few_pairs' if len(selected)<3 else ''
                    if not reason and selected[col].nunique()<2:reason='constant_score'
                    if not reason and selected.y.nunique()<2:reason='constant_target'
                    ic=np.nan
                    if not reason:
                        ranks=selected[[col,'y']].rank(method='average').to_numpy(float)
                        ic=float(np.corrcoef(ranks,rowvar=False)[0,1])
                        if not np.isfinite(ic) or abs(ic)>1+1e-12:raise ValueError('invalid_rank_ic')
                    monthly.append({'model':'feature_score','candidate':candidate,'fold':'development',
                        'sample_scope':scope,'momentum_window':window,'base_month':part.base_month.iloc[0],
                        'target_month':month,'n':len(selected),'ic':ic,'reason':reason})
    monthly=pd.DataFrame(monthly)
    reference_ic=ordinary_data['monthly']
    old_ic=monthly[monthly.momentum_window.eq('11m')].merge(reference_ic[['candidate','target_month','n','ic']],
        on=['candidate','target_month'],suffixes=('','_reference'),validate='one_to_one')
    np.testing.assert_allclose(old_ic.ic,old_ic.ic_reference,atol=1e-12,equal_nan=True)
    np.testing.assert_array_equal(old_ic.n,old_ic.n_reference)
    summary=[]
    for keys,part in monthly.groupby(['sample_scope','momentum_window','candidate'],sort=True):
        values=part.ic.dropna();sd=values.std(ddof=1)
        positive=int(values.gt(0).sum());negative=int(values.lt(0).sum());zero=int(values.eq(0).sum())
        if positive+negative+zero!=len(values):raise ValueError('ic_sign_count')
        summary.append(dict(zip(['sample_scope','momentum_window','candidate'],keys))|
            {'valid_months':len(values),'unavailable_months':int(part.ic.isna().sum()),'pair_count':int(part.n.sum()),
             'mean_ic':values.mean(),'sample_sd':sd,'icir':values.mean()/sd if sd>0 else np.nan,
             'positive_months':positive,'negative_months':negative,'zero_months':zero,'positive_fraction':positive/len(values)})
    summary=pd.DataFrame(summary)
    paired_ic=monthly[monthly.sample_scope.eq('paired_10m_11m')]
    delta=paired_ic.pivot(index=['candidate','base_month','target_month'],columns='momentum_window',values='ic').reset_index()
    delta['delta_ic_10m_minus_11m']=delta['10m']-delta['11m']
    delta_summary=delta.groupby('candidate',sort=True).delta_ic_10m_minus_11m.agg(['count','mean','std']).reset_index()
    delta_summary=delta_summary.rename(columns={'count':'valid_months','mean':'mean_delta_ic','std':'delta_sample_sd'})
    yearly=monthly.assign(target_year=monthly.target_month.str[:4]).groupby(
        ['sample_scope','momentum_window','candidate','target_year'],sort=True).agg(valid_months=('ic','count'),mean_ic=('ic','mean')).reset_index()
    checks={'saved_scores_and_inputs_unchanged':True,'calendar_target_only_development':True,
        'same_pairs_per_candidate_within_scope':True,'original_11m_monthly_ic_reproduced':True,
        'no_duplicate_keys':True,'rank_average_ties_no_pvalue':True,'sign_month_counts_exhaust_valid_months':True}
    result={'execute':execute,'stage':'ic','own_pairs':int(own.sum()),'paired_pairs':int(paired.sum()),
        'summary':summary.to_dict(orient='records'),'delta_summary':delta_summary.to_dict(orient='records'),'checks':checks}
    if not execute:return result
    folder=category/datetime.now().strftime('%Y%m%dT%H%M%S%f');folder.mkdir(parents=True,exist_ok=False)
    tables={'monthly_ic':monthly,'performance_summary':summary,'monthly_delta_ic_10m_vs_11m':delta,
        'delta_summary_10m_vs_11m':delta_summary,'ic_by_target_year':yearly,'sample_audit':pd.DataFrame(audit),
        'excluded_months':monthly[monthly.ic.isna()]}
    for name,t in tables.items():t.to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    joined[own].to_parquet(folder/'sample_with_scores.parquet',index=False)
    for name,value in hashes.items():
        if sha256(root/name)!=value:raise RuntimeError('source_changed_during_ic:'+name)
    code=['src/evaluation/h1_momentum_10m.py','src/features/h1.py','src/preprocessing/panel.py','config/experiment.yaml']
    record={'status':'EXECUTED','stage':'momentum_10m_ic','command':' '.join(sys.argv),
        'score_source':'common data; same original recipes recomputed in memory',
        'input_sha256':hashes,'code_sha256':{p:sha256(root/p) for p in code},'checks':checks,
        'samples':{'new_10m_common_pairs':int(own.sum()),'paired_10m_11m_pairs':int(paired.sum()),
            'calendar_months':len(months),'first_target':'2001-01 unavailable'},
        'method':'saved scores only; calendar t+1 raw target; average tied ranks Pearson correlation; equal monthly mean; ddof=1 SD; nonannualized ICIR; exact IC signs; paired delta=10m minus same-method 11m',
        'table_rows':{name:len(t) for name,t in tables.items()},'figures':[],
        'outputs_sha256':{p.name:sha256(p) for p in folder.iterdir()},
        'deferred_checks':{'related_and_full_tests':'large experiment end','temporary_reexecution':'large experiment end'},
        'hypothesis_tests':'not performed; next approval required'}
    (folder/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return {**result,'run_dir':str(folder)}


def main(argv=None):
    parser=argparse.ArgumentParser(description='10개월 모멘텀 가용성 관측')
    parser.add_argument('--config',default='config/experiment.yaml')
    parser.add_argument('--stage',choices=['availability','distribution','structure','ic'],default='availability')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    root=Path.cwd()
    config=yaml.safe_load((root/args.config).read_text(encoding='utf-8'))
    if args.stage=='availability':result=run(root,config,execute=args.execute)
    elif args.stage=='distribution':result=run_distribution(root,config,execute=args.execute)
    elif args.stage=='structure':result=run_structure(root,config,execute=args.execute)
    else:result=run_ic(root,config,execute=args.execute)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=='__main__':main()
