"""H1 원특징과 동일가중 점수. 파일·목표수익률에 접근하지 않는 계산 함수."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd
from pandas import DataFrame
from src.contracts import FEATURE_KEY, H1_COLUMNS
from src.preprocessing.panel import enforce_unique_stock_month, normalize_month, normalize_stock_id


def _keys(frame: DataFrame, month: str) -> None:
    enforce_unique_stock_month(frame, month_column=month)
    normalize_stock_id(frame.stock_id.tolist())
    normalize_month(frame[month].tolist())
    if not frame[month].between('2000-01', '2021-12').all():
        raise ValueError('development_period_only')


def _numeric(series: pd.Series, name: str) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise ValueError(f'{name}: numeric values required; parsing is upstream')
    return series.astype(float)


def _unit(frame: DataFrame, column: str, unit: str) -> None:
    if frame.attrs.get('units', {}).get(column) != unit:
        raise ValueError(f'{column}: unit must be {unit}')


def calculate_momentum_11m(returns: DataFrame, base_keys: DataFrame) -> DataFrame:
    """정확한 달력 t-11~t-1 복리. 누락·범위 밖·비유한 수익률을 채우지 않는다.

    입력 오류값은 그 창만 격리한다. -100%는 유효하며 곱이 0인 Mom=-1이다.
    mom_valid_months는 창 안의 실제 유효 달 수(자료 시작 전 달은 0개로 센다).
    """
    _keys(returns, 'month')
    _keys(base_keys, 'base_month')
    _unit(returns, 'return', 'decimal')
    r = _numeric(returns['return'], 'return')
    invalid = r.notna() & (~np.isfinite(r) | r.lt(-1))
    src = returns[['stock_id', 'month']].copy()
    src['value'] = r.where(~invalid)
    src['invalid'] = invalid.astype(int)
    months = pd.period_range('1999-02', max(base_keys.base_month), freq='M').astype(str)
    wide = src.pivot(index='month', columns='stock_id', values='value').reindex(months)
    bad = src.pivot(index='month', columns='stock_id', values='invalid').reindex(months).fillna(0)
    counts = wide.notna().astype(int).rolling(11, min_periods=1).sum().shift(1).fillna(0)
    errors = bad.rolling(11, min_periods=1).sum().shift(1).fillna(0)
    # 전체 달력월로 재배열한 뒤에만 위치 이동을 사용한다.
    product = (1 + wide).rolling(11, min_periods=11).apply(np.prod, raw=True).shift(1) - 1
    idx = pd.MultiIndex.from_frame(base_keys[['base_month', 'stock_id']])
    out = base_keys[list(FEATURE_KEY)].copy()
    out['mom_valid_months'] = counts.stack(future_stack=True).reindex(idx).fillna(0).to_numpy(dtype=int)
    out['mom'] = product.stack(future_stack=True).reindex(idx).to_numpy(dtype=float)
    bad_window = errors.stack(future_stack=True).reindex(idx).fillna(0).to_numpy() > 0
    full = out.mom_valid_months.eq(11)
    overflow = full & ~np.isfinite(out.mom)
    early = out.base_month.lt('2000-12')
    out['mom_exclusion_reason'] = np.select(
        [bad_window, overflow, early, ~full],
        ['invalid_return_in_window', 'nonfinite_momentum', 'window_before_2000', 'incomplete_11_calendar_months'], '')
    out['mom_available'] = out.mom_exclusion_reason.eq('')
    out.loc[~out.mom_available, 'mom'] = np.nan
    return out


def calculate_size_bm(panel: DataFrame, equity_asof: DataFrame) -> DataFrame:
    """기존 as-of의 usable을 적용하고 공개 전·만료 후 사용은 중단한다."""
    _keys(panel, 'month')
    _unit(panel, 'market_cap', 'KRW')
    _unit(equity_asof, 'equity', 'KRW')
    a = equity_asof.rename(columns={'stock_code': 'stock_id'}).copy()
    _keys(a, 'base_month')
    usable = a.usable.eq(True).fillna(False)
    for column in ('available_month', 'expiry_date'):
        if column not in a:
            raise ValueError(f'asof missing {column}')
    if (a.loc[usable, 'available_month'].isna().any()
            or a.loc[usable, 'available_month'].gt(a.loc[usable, 'base_month']).any()
            or a.loc[usable, 'expiry_date'].isna().any()
            or a.loc[usable, 'expiry_date'].str[:7].lt(a.loc[usable, 'base_month']).any()):
        raise ValueError('asof_future_or_expired')
    f = panel[['stock_id', 'month', 'eligible', 'market_cap']].rename(columns={'month': 'base_month'})
    f = f.merge(a[['stock_id', 'base_month', 'equity', 'usable', 'reason']], on=list(FEATURE_KEY), how='left', validate='one_to_one')
    mc, eq = _numeric(f.market_cap, 'market_cap'), _numeric(f.equity, 'equity')
    size_ok = np.isfinite(mc) & mc.gt(0)
    f['size'] = np.log(mc.where(size_ok))
    f['size_available'] = size_ok
    f['size_exclusion_reason'] = np.select([mc.isna(), ~np.isfinite(mc), mc.le(0)],
                                         ['market_cap_missing', 'market_cap_nonfinite', 'market_cap_nonpositive'], '')
    bm_ok = size_ok & f.usable.eq(True).fillna(False) & np.isfinite(eq) & eq.gt(0)
    f['bm'] = (eq / mc).where(bm_ok)
    bm_ok &= np.isfinite(f.bm)
    f['bm_available'] = bm_ok
    f['bm_exclusion_reason'] = np.select(
        [~size_ok, eq.isna(), ~np.isfinite(eq), eq.le(0), ~f.usable.eq(True).fillna(False), ~np.isfinite(f.bm)],
        [f.size_exclusion_reason, f.reason.fillna('no_asof_value'), 'equity_nonfinite', 'nonpositive_equity',
         f.reason.fillna('').replace('', 'asof_unusable'), 'bm_nonfinite'], '')
    f.loc[~bm_ok, 'bm'] = np.nan
    return f.drop(columns=['market_cap', 'equity', 'usable', 'reason'])


def cross_sectional_zscore(common_valid: DataFrame, *, columns: Sequence[str], ddof: int = 1) -> DataFrame:
    """당월 적격·공통 유효 집합의 표본표준편차로 계산. y는 참조하지 않는다."""
    if ddof != 1 or tuple(columns) != ('size', 'bm', 'mom'):
        raise ValueError('H1 requires size/bm/mom and ddof=1')
    _keys(common_valid, 'base_month')
    f = common_valid
    for c in columns:
        _numeric(f[c], c)
        if not f[f'{c}_available'].eq(np.isfinite(f[c])).all():
            raise ValueError(f'{c}: availability/value mismatch')
    if not f.eligible.dropna().isin([True, False]).all():
        raise ValueError('eligible must be nullable boolean')
    member = f.eligible.eq(True).fillna(False) & np.isfinite(f[list(columns)]).all(axis=1)
    values = f[list(columns)].where(member, np.nan)
    groups = values.groupby(f.base_month)
    n = member.astype(int).groupby(f.base_month).transform('sum')
    means, sds = groups.transform('mean'), groups.transform('std', ddof=1)
    valid_sd = (np.isfinite(sds) & sds.gt(0)).all(axis=1)
    z = (values - means) / sds
    reason = pd.Series(np.select(
        [f.eligible.isna(), ~f.eligible.eq(True).fillna(False), ~member, n.lt(2), ~valid_sd, ~np.isfinite(z).all(axis=1)],
        ['eligibility_unknown', 'ineligible', 'components_unavailable', 'too_few_common_stocks',
         'zero_or_nonfinite_sd', 'nonfinite_zscore'], ''), index=f.index)
    out = f[[*FEATURE_KEY, 'eligible']].copy()
    for c in columns:
        out[f'z_{c}'] = z[c].where(reason.eq(''))
    out['standardization_n'] = n
    out['h1_exclusion_reason'] = reason
    return out


def calculate_h1_score(components: DataFrame) -> DataFrame:
    """[-Z(Size)+Z(B/M)+Z(Mom)]/3. 표준화 뒤까지 목표를 결합하지 않는다."""
    out = cross_sectional_zscore(components, columns=('size', 'bm', 'mom'))
    out['h1'] = (-out.z_size + out.z_bm + out.z_mom) / 3
    out['h1_available'] = out.h1_exclusion_reason.eq('')
    return out[list(H1_COLUMNS)]


def validate_h1_contract(features: DataFrame, components: DataFrame, provenance: DataFrame) -> None:
    """월별 독립 재계산으로 집합·수식·상태를 검사한다.

    provenance는 사용한 as-of 표. 모멘텀 창은 단위 시험과
    파이프라인의 원수익률 독립 곱셈으로 검산한다.
    """
    _keys(features, 'base_month')
    _keys(components, 'base_month')
    if tuple(features.columns) != H1_COLUMNS:
        raise ValueError('h1_columns_mismatch')
    a = provenance.rename(columns={'stock_code': 'stock_id'})
    _keys(a, 'base_month')
    usable = a.usable.eq(True).fillna(False)
    if (a.loc[usable, 'available_month'].isna().any()
            or a.loc[usable, 'available_month'].gt(a.loc[usable, 'base_month']).any()
            or a.loc[usable, 'expiry_date'].isna().any()
            or a.loc[usable, 'expiry_date'].str[:7].lt(a.loc[usable, 'base_month']).any()):
        raise ValueError('asof_future_or_expired')
    matched = components.merge(a[[*FEATURE_KEY, 'equity', 'usable']], on=list(FEATURE_KEY), how='left', validate='one_to_one')
    bm_member = matched.bm_available
    if not (matched.loc[bm_member, 'usable'].eq(True).all() and matched.loc[bm_member, 'equity'].gt(0).all()):
        raise ValueError('bm_asof_provenance_mismatch')
    np.testing.assert_allclose(matched.loc[bm_member, 'bm'],
                               matched.loc[bm_member, 'equity'] / np.exp(matched.loc[bm_member, 'size']), rtol=1e-12)
    for c in ('size', 'bm', 'mom'):
        ok = components[f'{c}_available']
        if not ok.eq(np.isfinite(components[c])).all():
            raise ValueError('component_availability_mismatch')
        if not components[f'{c}_exclusion_reason'].eq('').eq(ok).all():
            raise ValueError('component_reason_mismatch')
    if (components.mom_available & components.mom_valid_months.ne(11)).any():
        raise ValueError('momentum_count_mismatch')
    if not components[list(FEATURE_KEY)].reset_index(drop=True).equals(features[list(FEATURE_KEY)].reset_index(drop=True)):
        raise ValueError('h1_keys_mismatch')
    pd.testing.assert_series_equal(features.eligible, components.eligible, check_names=False)
    for month, part in components.groupby('base_month', sort=True):
        member = part.eligible.eq(True).fillna(False) & np.isfinite(part[['size', 'bm', 'mom']]).all(axis=1)
        v = part.loc[member, ['size', 'bm', 'mom']].to_numpy(float)
        got = features.loc[part.index]
        if not got.standardization_n.eq(len(v)).all():
            raise ValueError('standardization_n_mismatch')
        sd = np.std(v, axis=0, ddof=1) if len(v) >= 2 else np.full(3, np.nan)
        can = len(v) >= 2 and np.isfinite(sd).all() and (sd > 0).all()
        expected = member & can
        if not got.h1_available.eq(expected).all():
            raise ValueError('h1_availability_mismatch')
        if can:
            z = (v - np.mean(v, axis=0)) / sd
            np.testing.assert_allclose(got.loc[member, ['z_size', 'z_bm', 'z_mom']], z, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(got.loc[member, 'h1'], (-z[:, 0]+z[:, 1]+z[:, 2])/3, rtol=1e-10, atol=1e-10)
        if not got.loc[~expected, ['z_size', 'z_bm', 'z_mom', 'h1']].isna().all().all():
            raise ValueError('unavailable_score_present')
        if not got.h1_exclusion_reason.eq('').eq(got.h1_available).all():
            raise ValueError('h1_reason_mismatch')


def winsorize_1_99(components: DataFrame) -> DataFrame:
    """당월 H1 공통 적격 집합의 원특징 1·99 분위수(선형 보간)로 제한한다.

    경계는 목표값을 결합하기 전에 추정하며 원본은 수정하지 않는다.
    집합 밖 값은 그대로 유지하되 H1 계산에는 원래처럼 포함하지 않는다.
    """
    _keys(components, 'base_month')
    out = components.copy()
    member = components.eligible.eq(True).fillna(False) & components[['size_available','bm_available','mom_available']].all(axis=1)
    for _, part in components[member].groupby('base_month', sort=True):
        for c in ('size','bm','mom'):
            lo, hi = part[c].quantile([.01,.99], interpolation='linear')
            out.loc[part.index,c] = part[c].clip(lo,hi)
    return out


def _pca_months(standardized: DataFrame):
    """목표 없이 방향 정렬한 월별 공통 유효 Z 행렬을 순서대로 반환한다."""
    _keys(standardized, 'base_month')
    if not standardized.base_month.between('2000-12','2021-11').all():
        raise PermissionError('pca_development_base_months_only')
    if 'y' in standardized or 'return' in standardized:
        raise ValueError('pca_input_must_not_include_targets_or_returns')
    cols=['z_size','z_bm','z_mom']
    for c in cols: _numeric(standardized[c],c)
    v=standardized[cols].to_numpy(float,na_value=np.nan)
    if np.isinf(v).any() or ((np.isfinite(v).sum(axis=1)!=0)&(np.isfinite(v).sum(axis=1)!=3)).any():
        raise ValueError('partial_or_nonfinite_pca_input')
    member=np.isfinite(v).all(axis=1)
    if not standardized.loc[member,'eligible'].eq(True).fillna(False).all():
        raise ValueError('pca_input_not_eligible')
    for month,part in standardized.groupby('base_month',sort=True):
        x=part[cols].to_numpy(float,na_value=np.nan)
        x=x[np.isfinite(x).all(axis=1)].copy()
        x[:,0]*=-1
        yield month,x


def _pca_variance_rows(method, month, current_n, n, fit_months, first, last, scatter,
                       *, include_structure=False, previous_axes=None, current_scatter=None):
    """표본 공분산의 설명분산. 선택적으로 가중치·상관 진단을 함께 반환한다."""
    reason='too_few_fit_rows' if n<2 else ''
    eigen=np.full(3,np.nan);ratio=np.full(3,np.nan);clipped=False
    vectors=None;covariance=None;tolerance=np.nan
    if not reason:
        covariance=scatter/(n-1)
        if not np.isfinite(covariance).all(): raise ValueError('nonfinite_pca_covariance')
        covariance=(covariance+covariance.T)/2
        if include_structure:
            eigen,vectors=np.linalg.eigh(covariance)
            eigen=eigen[::-1];vectors=vectors[:,::-1]
        else:eigen=np.linalg.eigvalsh(covariance)[::-1]
        tolerance=3*np.finfo(float).eps*np.linalg.norm(covariance,ord=2)
        if eigen.min() < -tolerance: raise ValueError('pca_covariance_not_positive_semidefinite')
        clipped=bool((eigen<0).any());eigen=np.maximum(eigen,0.)
        total=float(eigen.sum())
        if not np.isfinite(total) or total<=0:reason='zero_or_nonfinite_total_variance'
        else:ratio=eigen/total
    cumulative=np.cumsum(ratio)
    rows=[{'method':method,'base_month':month,'target_month':str(pd.Period(month,freq='M')+1),
           'current_n':current_n,'fit_n':n,'fit_months':fit_months,
           'fit_first_base_month':first,'fit_last_base_month':last,
           'component':f'PC{k+1}','eigenvalue':float(eigen[k]),
           'explained_variance_ratio':float(ratio[k]),'cumulative_explained_variance_ratio':float(cumulative[k]),
           'available':not reason,'exclusion_reason':reason,'tiny_negative_eigenvalue_clipped':clipped} for k in range(3)]
    if not include_structure:return rows
    from src.contracts import H1_PCA_STRUCTURE_COLUMNS
    if reason:
        for row in rows:
            row.update({c:np.nan for c in H1_PCA_STRUCTURE_COLUMNS})
            row.update(fit_corr_reason=reason,current_corr_reason=reason,
                       display_reference_base_month=None,display_sign_flipped=False,eigenvector_identifiable=False)
        return rows
    def correlations(cov,weight,count):
        if count<2 or cov is None:return np.full(3,np.nan),'too_few_current_rows'
        component_var=float(weight@cov@weight)
        if component_var<=tolerance:return np.full(3,np.nan),'constant_component'
        denom=np.sqrt(np.diag(cov)*component_var)
        if (denom<=0).any():return np.full(3,np.nan),'constant_feature'
        corr=(cov@weight)/denom
        if not np.isfinite(corr).all() or (np.abs(corr)>1+1e-10).any():
            raise ValueError('invalid_feature_component_correlation')
        return corr,''
    if previous_axes is None:previous_axes={}
    current_cov=current_scatter/(current_n-1) if current_n>=2 and current_scatter is not None else None
    for k,row in enumerate(rows):
        weight=vectors[:,k].copy();previous=previous_axes.get(k)
        if previous is None:
            flip=bool(weight[np.argmax(np.abs(weight))]<0);reference=None;similarity=np.nan
        else:
            reference,prior=previous
            if reference>=month:raise ValueError('future_display_sign_reference')
            dot=float(prior@weight)
            flip=bool(dot<0) if dot!=0 else bool(weight[np.argmax(np.abs(weight))]<0)
            similarity=abs(dot)
        if flip:weight*=-1
        vectors[:,k]=weight;previous_axes[k]=(month,weight.copy())
        fit_corr,fit_reason=correlations(covariance,weight,n)
        current_corr,current_reason=correlations(current_cov,weight,current_n)
        gap=float(np.min(np.abs(eigen[k]-np.delete(eigen,k))))
        row.update({**dict(zip(('weight_neg_size','weight_bm','weight_mom'),weight)),
                    **dict(zip(('fit_corr_neg_size','fit_corr_bm','fit_corr_mom'),fit_corr)),
                    **dict(zip(('current_corr_neg_size','current_corr_bm','current_corr_mom'),current_corr)),
                    'fit_corr_reason':fit_reason,'current_corr_reason':current_reason,
                    'display_reference_base_month':reference,'display_sign_flipped':flip,
                    'previous_axis_cosine':similarity,'nearest_eigenvalue_gap_ratio':gap/float(eigen.sum()),
                    'eigenvector_identifiable':bool(gap>tolerance)})
    np.testing.assert_allclose(vectors.T@vectors,np.eye(3),atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(covariance@vectors,vectors*eigen,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose((vectors*eigen)@vectors.T,covariance,atol=1e-10,rtol=1e-10)
    return rows


def nonnegative_variance_weights(covariance):
    """3차원 비음수 단위구면의 전역 최대. 7개 비어 있지 않은 지지집합을 열거.

    최적해의 양수 좌표에서는 주부분행렬의 고유벡터 조건이 성립한다.
    모든 지지집합의 양수 고유벡터를 비교한다. 중복 고유값의 내부 해는
    같은 목적값의 경계 해를 가지므로 더 작은 지지집합에서 포함된다.
    일반 PC1을 절댓값/절단하여 후보로 사용하는 방식이 아니다.
    """
    c=np.asarray(covariance,dtype=float)
    if c.shape!=(3,3) or not np.isfinite(c).all():raise ValueError('invalid_constrained_covariance')
    np.testing.assert_allclose(c,c.T,atol=1e-12)
    if np.linalg.eigvalsh(c).min() < -1e-10:raise ValueError('covariance_not_psd')
    candidates=[]
    for mask in range(1,8):
        active=[j for j in range(3) if mask & (1<<j)]
        _,vectors=np.linalg.eigh(c[np.ix_(active,active)])
        for vector in vectors.T:
            for direction in (1.,-1.):
                v=direction*vector
                if not (v>0).all():continue
                w=np.zeros(3);w[active]=v/np.linalg.norm(v)
                candidates.append((float(w@c@w),w))
    value,w=max(candidates,key=lambda item:item[0])
    tolerance=1e-11*max(1.,np.linalg.norm(c,ord=2))
    optima=[]
    for objective,vector in candidates:
        if abs(objective-value)<=tolerance and not any(np.linalg.norm(vector-v)<1e-8 for v in optima):
            optima.append(vector)
    active=w>0; gradient=c@w
    np.testing.assert_allclose(np.linalg.norm(w),1.,atol=1e-12)
    np.testing.assert_allclose(gradient[active],value*w[active],atol=tolerance,rtol=1e-10)
    if (gradient[~active]>tolerance).any():raise ValueError('constrained_boundary_optimality_failed')
    if value>np.linalg.eigvalsh(c)[-1]+tolerance:raise ValueError('constrained_exceeds_unconstrained')
    return w,value,len(optima)


def constrained_pca(standardized: DataFrame, *, method: str) -> DataFrame:
    """기존 A/B 추정 집합에서 비음수 단위 가중치 분산을 직접 최대화한다."""
    from src.contracts import H1_CONSTRAINED_PCA_COLUMNS
    if method not in ('pca_a','pca_b'):raise ValueError('unknown_constrained_pca_method')
    rows=[];n=0;mean=np.zeros(3);scatter=np.zeros((3,3));first=None;last=None;months=0
    for month,x in _pca_months(standardized):
        m=len(x);batch_mean=x.mean(axis=0) if m else np.zeros(3);centered=x-batch_mean
        if method=='pca_a':
            n=m;mean=batch_mean;scatter=centered.T@centered
            first=last=month if m else None;months=int(m>0)
        elif m:
            delta=batch_mean-mean
            scatter+=centered.T@centered+np.outer(delta,delta)*(n*m/(n+m))
            mean+=delta*m/(n+m);n+=m;months+=1;first=first or month;last=month
        reason='too_few_fit_rows' if n<2 else ''
        w=np.full(3,np.nan);value=ratio=ordinary=np.nan;ties=0
        if not reason:
            cov=scatter/(n-1);total=float(np.trace(cov))
            if total<=0:reason='zero_or_nonfinite_total_variance'
            else:
                w,value,ties=nonnegative_variance_weights(cov)
                ratio=value/total;ordinary=float(np.linalg.eigvalsh(cov)[-1]/total)
        row={'method':method,'base_month':month,'target_month':str(pd.Period(month,freq='M')+1),
            'current_n':m,'fit_n':n,'fit_months':months,'fit_first_base_month':first,'fit_last_base_month':last,
            'available':not reason,'exclusion_reason':reason,'projected_variance':value,
            'explained_variance_ratio':ratio,'ordinary_pc1_variance_ratio':ordinary,
            'variance_ratio_delta':ratio-ordinary,'optimal_solutions':ties,
            **dict(zip(('weight_neg_size','weight_bm','weight_mom'),w)),
            **dict(zip(('center_neg_size','center_bm','center_mom'),mean if n else [np.nan]*3))}
        rows.append(row)
    return pd.DataFrame(rows,columns=H1_CONSTRAINED_PCA_COLUMNS)


def pca_a(standardized: DataFrame, *, include_structure: bool = False) -> DataFrame:
    """당월 3열 방향 정렬 Z의 설명분산 3개만 반환한다. 성분·부호·점수 선택 없음."""
    from src.contracts import H1_PCA_VARIANCE_COLUMNS
    from src.contracts import H1_PCA_STRUCTURE_COLUMNS
    rows=[];previous_axes={}
    for month,x in _pca_months(standardized):
        n=len(x);centered=x-x.mean(axis=0) if n else x
        rows.extend(_pca_variance_rows('pca_a',month,n,n,int(n>0),month if n else None,
                                      month if n else None,centered.T@centered, include_structure=include_structure,
                                      previous_axes=previous_axes,current_scatter=centered.T@centered))
    return pd.DataFrame(rows,columns=(*H1_PCA_VARIANCE_COLUMNS,*(H1_PCA_STRUCTURE_COLUMNS if include_structure else ())))


def pca_b(standardized: DataFrame, *, include_structure: bool = False) -> DataFrame:
    """그 기준월까지의 월별 Z 행을 누적한 표본 공분산. 미래 행·수익률 미사용.

    월별 Z를 다시 전기간 표준화하지 않는다. 행별 동일가중으로 누적하고,
    배치 평균·중심 제곱합 결합식으로 전체 누적 행을 중심화한 PCA와 같게 계산한다.
    """
    from src.contracts import H1_PCA_VARIANCE_COLUMNS
    from src.contracts import H1_PCA_STRUCTURE_COLUMNS
    rows=[];n=0;mean=np.zeros(3);scatter=np.zeros((3,3));months=0;first=None;last=None;previous_axes={}
    for month,x in _pca_months(standardized):
        m=len(x)
        if m:
            batch_mean=x.mean(axis=0);centered=x-batch_mean
            delta=batch_mean-mean
            scatter+=centered.T@centered+np.outer(delta,delta)*(n*m/(n+m))
            mean+=delta*(m/(n+m));n+=m;months+=1;first=first or month;last=month
        centered_current=x-x.mean(axis=0) if m else x
        rows.extend(_pca_variance_rows('pca_b',month,m,n,months,first,last,scatter,
                    include_structure=include_structure,previous_axes=previous_axes,
                    current_scatter=centered_current.T@centered_current))
    return pd.DataFrame(rows,columns=(*H1_PCA_VARIANCE_COLUMNS,*(H1_PCA_STRUCTURE_COLUMNS if include_structure else ())))
