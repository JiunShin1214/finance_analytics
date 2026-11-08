"""모형 예측과 특징 점수에 공통인 월별 Spearman 순위 상관 계약."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from pandas import DataFrame
from src.contracts import IC_COLUMNS, PREDICTION_KEY
from src.preprocessing.panel import enforce_unique_stock_month, normalize_month, normalize_stock_id

GROUP = ['model', 'candidate', 'fold', 'target_month']


def newey_west_mean_test(values) -> dict:
    """동일 간격·결측 없는 시계열 평균의 Bartlett HAC 양측 t 검정.

    L=floor(4*(T/100)^(2/9)), gamma_l=sum(u[t]*u[t-l])/T.
    Var(mean)=(gamma_0+2*sum((1-l/(L+1))*gamma_l))/T.
    추가 T/(T-k) 소표본 배율 없음. p·CI는 모두 t(T-1).
    결측 제거·시간 정렬은 하지 않고 호출부에서 검증한 순서를 사용한다.
    """
    from scipy.stats import t as student_t
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('hac_requires_finite_ordered_series_at_least_two')
    n=len(x);lag=int(np.floor(4*(n/100)**(2/9)))
    if lag>=n:raise ValueError('hac_lag_exceeds_series')
    mean=float(x.mean());u=x-mean;sd=float(x.std(ddof=1))
    meat=float(u@u)
    for ell in range(1,lag+1):meat+=2*(1-ell/(lag+1))*float(u[ell:]@u[:-ell])
    if meat < -1e-12*max(1.,float(u@u)):raise ValueError('negative_hac_variance')
    se=float(np.sqrt(max(0.,meat))/n)
    statistic=mean/se if se>0 else np.nan
    p=float(2*student_t.sf(abs(statistic),df=n-1)) if se>0 else np.nan
    critical=float(student_t.ppf(.975,df=n-1))
    low=mean-critical*se if se>0 else np.nan;high=mean+critical*se if se>0 else np.nan
    usual_se=sd/np.sqrt(n)
    usual_t=mean/usual_se if usual_se>0 else np.nan
    usual_p=float(2*student_t.sf(abs(usual_t),df=n-1)) if usual_se>0 else np.nan
    if se>0 and bool(p<.05)!=bool(low>0 or high<0):raise ValueError('hac_p_ci_disagree')
    return {'valid_months':n,'mean':mean,'sample_sd':sd,'hac_se':se,'statistic':statistic,
        'p_two_sided':p,'ci95_low':low,'ci95_high':high,'lag':lag,'df':n-1,
        'reference_distribution':'t','kernel':'Bartlett','small_sample_correction':False,
        'significant_raw_05':bool(p<.05) if np.isfinite(p) else False,
        'usual_se':usual_se,'usual_t':usual_t,'usual_p_two_sided':usual_p,
        'reason':'' if se>0 else 'zero_hac_standard_error'}


def holm_bonferroni(pvalues):
    """명시한 고정 검정 집합의 양측 p값을 단계 하강 방식으로 보정한다."""
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('holm_requires_all_family_pvalues')
    order=np.argsort(p,kind='stable');m=len(p)
    adjusted=np.minimum(1.,np.maximum.accumulate((m-np.arange(m))*p[order]))
    out=np.empty(m);out[order]=adjusted
    return out


def validate_prediction_keys(predictions: DataFrame, actuals: DataFrame) -> None:
    if not set(PREDICTION_KEY).issubset(predictions) or predictions[list(PREDICTION_KEY)].isna().any().any():
        raise ValueError('prediction_keys_missing')
    if predictions.duplicated(list(PREDICTION_KEY)).any():
        raise ValueError('duplicate_prediction_keys')
    enforce_unique_stock_month(actuals, month_column='target_month')
    for f in (predictions, actuals):
        normalize_stock_id(f.stock_id.tolist())
        normalize_month(f.target_month.tolist())
        if not f.target_month.between('2001-01', '2021-12').all():
            raise PermissionError('training_target_month_only')


def join_validation_actuals(predictions: DataFrame, actuals: DataFrame) -> DataFrame:
    validate_prediction_keys(predictions, actuals)
    if 'y' in predictions:
        raise ValueError('target_already_attached')
    return predictions.merge(actuals[['stock_id', 'target_month', 'y']],
                             on=['stock_id', 'target_month'], how='left', validate='many_to_one')


def rank_ic_for_month(month_predictions: DataFrame, *, min_valid_stocks: int = 3,
                      tie_method: str = 'average', score_column: str = 'yhat') -> DataFrame:
    """단일 그룹·목표월 -> IC_COLUMNS 한 행. score_column으로 특징도 평가한다.

    NaN 쌍은 제외한다. 무한값은 입력 오류로 그 월을 미산출하고 이유를 남긴다.
    최소 3쌍, 양쪽 비상수. 평균 동순위의 Pearson 상관으로 Spearman을 계산한다.
    """
    if min_valid_stocks != 3 or tie_method != 'average':
        raise ValueError('rank_ic_rule_change')
    f = month_predictions
    if not len(f) or not set([*GROUP, 'stock_id', score_column, 'y']).issubset(f):
        raise ValueError('rank_ic_columns_or_rows_missing')
    if f[GROUP].isna().any().any() or len(f[GROUP].drop_duplicates()) != 1:
        raise ValueError('mixed_or_null_ic_group')
    enforce_unique_stock_month(f, month_column='target_month')
    normalize_stock_id(f.stock_id.tolist()); normalize_month(f.target_month.tolist())
    if not f.target_month.between('2001-01', '2021-12').all():
        raise PermissionError('training_target_month_only')
    if not all(pd.api.types.is_numeric_dtype(f[c]) for c in (score_column, 'y')):
        raise ValueError('rank_ic_requires_numeric')
    v = f[[score_column, 'y']].to_numpy(dtype=float, na_value=np.nan)
    paired = v[np.isfinite(v).all(axis=1)]
    n = len(paired)
    if np.isinf(v).any(): reason = 'nonfinite_input'
    elif n < 3: reason = 'too_few_pairs'
    elif np.unique(paired[:, 0]).size < 2: reason = 'constant_score'
    elif np.unique(paired[:, 1]).size < 2: reason = 'constant_target'
    else: reason = ''
    ic = np.nan
    if not reason:
        ranks = pd.DataFrame(paired).rank(method='average').to_numpy()
        ic = float(np.corrcoef(ranks[:, 0], ranks[:, 1])[0, 1])
    return pd.DataFrame([{**f.iloc[0][GROUP].to_dict(), 'n':n, 'ic':ic, 'reason':reason}], columns=IC_COLUMNS)


def mean_rank_ic_on_common_months(monthly_ic: DataFrame, *, comparison_ids: Sequence[str]) -> DataFrame:
    """명시한 candidate의 공통 유효 월만 동일가중. 입력 종목 표본 일치는 호출부 책임."""
    ids = list(comparison_ids)
    if not ids or len(set(ids)) != len(ids): raise ValueError('comparison_ids_invalid')
    f = monthly_ic[monthly_ic.candidate.isin(ids)]
    if f.duplicated(['candidate', 'fold', 'target_month']).any(): raise ValueError('duplicate_monthly_ic')
    wide = f.pivot(index=['fold', 'target_month'], columns='candidate', values='ic').reindex(columns=ids)
    valid = np.isfinite(wide).all(axis=1)
    return pd.DataFrame([{'candidate':c, 'valid_months':int(valid.sum()), 'excluded_months':int((~valid).sum()),
                          'mean_ic':float(wide.loc[valid,c].mean()),
                          'reason':'' if valid.any() else 'no_common_valid_months'} for c in ids])
