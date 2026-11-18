"""H3(EWMA 총변동성) 분석: 노션 10주차 H1 흐름을 단일 특징에 맞춘 것(2026-11-08 사용자 지시).

순서
1) 특징: data/interim의 panel·eligibility만으로 H3를 메모리에서 계산한다(src.features.h3, 4개월 KOSPI 비상장 공백 초기화).
2) 분석(run_analysis): 변동성 원값 sigma=sqrt(v)의 월별 분포·극단값, 커버리지, 원값 H3의 월별 Rank IC,
   1·99 분위수 제한(당월 H3 표준화 집합에서 선형 보간 경계 -> 제한 -> ddof=1 재표준화) 전후 비교.
3) 가설검정(run_hypothesis, 별도 승인): 원값 H3 평균 IC=0 주검정, 제한 H3 평균 IC=0, 대응 차이 ΔIC=0.
   Newey–West HAC(Bartlett), L=floor(4(T/100)^(2/9)), t(T-1). H1~H4 Holm은 네 결과가 모일 때까지 보류.

특징 계산·표준화·경계 추정을 모두 끝낸 뒤 차월(달력월 t+1) 제공 수익률을 결합한다. 예상 방향은 IC<0이다.
이전 실행 기록은 읽지 않는다. 출력은 outputs/run_id/<run_category>/<시각>/ 와 사용자 지정 붙여넣기 파일 하나.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import H1_MONTHLY_IC_COLUMNS
from src.evaluation.rank_ic import newey_west_mean_test, rank_ic_for_month
from src.preprocessing.panel import join_training_targets

BASE_FIRST, BASE_LAST = '2000-12', '2021-11'
CANDIDATES = ('h3', 'h3_winsor')
LABELS = {'h3': 'H3 원값', 'h3_winsor': 'H3 · 1·99 분위수 제한'}
QUANTILES = (.01, .05, .25, .5, .75, .95, .99)
H3_EXPECTED = {'lambda': .97, 'init_n_valid': 10, 'init_ddof': 1, 'recent_window': 12, 'recent_min_valid': 10,
               'ddof': 1, 'includes_base_month': True, 'reinitialize_every_12_months': False}


# ---------- 공통 ----------

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def check_h3_config(config):
    if any(config['H3'].get(k) != v for k, v in H3_EXPECTED.items()):
        raise ValueError('unapproved_H3_config')
    if config['UNITS'].get('return') != 'decimal':
        raise ValueError('H3_return_unit_mismatch')


def metrics(values):
    s = pd.Series(values, dtype=float).dropna()
    sd = float(s.std(ddof=1)) if len(s) > 1 else np.nan
    return {'valid_months': len(s), 'mean_ic': float(s.mean()) if len(s) else np.nan, 'ic_sd': sd,
            'icir': float(s.mean() / sd) if np.isfinite(sd) and sd > 0 else np.nan,
            'positive_ic_fraction': float(s.gt(0).mean()) if len(s) else np.nan,
            'negative_ic_fraction': float(s.lt(0).mean()) if len(s) else np.nan}


def markdown_table(frame):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return '—' if not np.isfinite(v) else (f'{v:.6e}' if v != 0 and abs(v) < 1e-4 else f'{v:.6f}')
        return '—' if pd.isna(v) else str(v).replace('|', '/')
    head = '| ' + ' | '.join(map(str, frame.columns)) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n'
    return head + ''.join('| ' + ' | '.join(cell(v) for v in row) + ' |\n' for row in frame.itertuples(index=False, name=None))


def append_report(path, marker, text):
    old = path.read_text(encoding='utf-8') if path.is_file() else '# H3 노션 붙여넣기 표\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(old.rstrip() + '\n' + marker + text.split(marker, 1)[-1], encoding='utf-8')


# ---------- 1) 특징 ----------

def compute_h3(panel, eligibility):
    """src.features.h3.compute_h3_from_panel과 같다(features 단계와 같은 계산)."""
    from src.features.h3 import compute_h3_from_panel
    return compute_h3_from_panel(panel, eligibility, last_base_month=BASE_LAST)


def winsorize_h3(features):
    """당월 H3 표준화 집합(h3_available)의 sigma를 1·99 분위수(선형 보간)로 제한하고 ddof=1로 다시 표준화한다.

    집합 밖 행은 결측으로 둔다. 목표값은 쓰지 않는다. 원본은 바꾸지 않는다.
    """
    if 'y' in features:
        raise ValueError('winsor_input_must_not_include_targets')
    out = features[['stock_id', 'base_month', 'eligible', 'sqrt_v', 'h3', 'h3_available']].copy()
    sigma_w = np.full(len(out), np.nan)
    z_w = np.full(len(out), np.nan)
    q01 = np.full(len(out), np.nan)
    q99 = np.full(len(out), np.nan)
    member = out.h3_available.to_numpy(bool)
    for _, pos in out.groupby('base_month', sort=True).indices.items():
        pos = pos[member[pos]]
        if len(pos) < 2:
            continue
        s = out.sqrt_v.iloc[pos]
        lo, hi = s.quantile([.01, .99], interpolation='linear')
        clipped = s.clip(lo, hi)
        sd = clipped.std(ddof=1)
        sigma_w[pos], q01[pos], q99[pos] = clipped.to_numpy(), lo, hi
        if np.isfinite(sd) and sd > 0:
            z_w[pos] = ((clipped - clipped.mean()) / sd).to_numpy()
    out['sigma_winsor'], out['h3_winsor'], out['winsor_q01'], out['winsor_q99'] = sigma_w, z_w, q01, q99
    out['clipped'] = member & (out.sqrt_v.lt(out.winsor_q01) | out.sqrt_v.gt(out.winsor_q99)).to_numpy()
    return out


# ---------- 2) 분석 ----------

def _shape(values):
    v = pd.Series(values, dtype=float).dropna()
    q = v.quantile(list(QUANTILES), interpolation='linear') if len(v) else pd.Series(np.nan, index=list(QUANTILES))
    return {'n': len(v), 'min': v.min(), **{f'q{int(round(k * 100)):02d}': x for k, x in q.items()}, 'max': v.max(),
            'mean': v.mean(), 'sd': v.std(ddof=1), 'skewness': v.skew(), 'excess_kurtosis': v.kurt(),
            'shape_reason': '' if len(v) >= 4 and v.nunique() > 1 else 'too_few_or_constant'}


def analyze(features, winsor, panel, *, representative_year=2021):
    """표준화·경계 추정이 끝난 표만 받아 목표를 결합하고 표를 만든다. 파일 입출력 없음."""
    if representative_year != 2021:
        raise ValueError('representative_year_fixed_2021')
    f = features.loc[features.base_month.between(BASE_FIRST, BASE_LAST)].reset_index(drop=True)
    w = winsor.loc[winsor.base_month.between(BASE_FIRST, BASE_LAST)].reset_index(drop=True)
    pd.testing.assert_frame_equal(f[['stock_id', 'base_month']], w[['stock_id', 'base_month']])
    returns = panel[['stock_id', 'month', 'return']].copy()
    returns.attrs['units'] = {'return': 'decimal'}
    scored = f[['stock_id', 'base_month', 'eligible', 'vol_available', 'sqrt_v', 'h3', 'h3_available',
                'h3_exclusion_reason', 'standardization_n']].assign(h3_winsor=w.h3_winsor, sigma_winsor=w.sigma_winsor,
                                                                    clipped=w.clipped)
    # 경계 추정과 표준화를 마친 뒤에만 목표를 결합한다.
    t = join_training_targets(scored, returns)
    group = {'model': 'feature_score', 'fold': 'development'}
    ic_rows, dist, extreme, changes, coverage, z_checks = [], [], [], [], [], []
    for month, part in t.groupby('base_month', sort=True):
        target = str(pd.Period(month, freq='M') + 1)
        member = part.h3_available.to_numpy(bool)
        paired = member & np.isfinite(part.h3_winsor.to_numpy(float))
        for cand in CANDIDATES:
            x = part[['stock_id', 'target_month', 'y']].copy()
            x['score'] = part[cand].where(paired)
            for k, v in {**group, 'candidate': cand}.items():
                x[k] = v
            row = rank_ic_for_month(x, score_column='score').iloc[0].to_dict()
            ic_rows.append({**row, 'base_month': month, 'sample_scope': 'h3_universe'})
        m = part[member]
        for variant, col in (('raw', 'sqrt_v'), ('winsor', 'sigma_winsor')):
            dist.append({'base_month': month, 'target_month': target, 'variant': variant, **_shape(m[col])})
        if len(m):
            lo, hi = m.sqrt_v.quantile([.01, .99], interpolation='linear')
            out = m[(m.sqrt_v < lo) | (m.sqrt_v > hi)]
            extreme += [{'stock_id': s, 'base_month': month, 'target_month': target, 'sigma': v, 'h3': z,
                         'q01': lo, 'q99': hi, 'side': 'below_q01' if v < lo else 'above_q99'}
                        for s, v, z in zip(out.stock_id, out.sqrt_v, out.h3)]
        before, after = m.h3, m.h3_winsor
        ok = len(m) >= 2 and before.nunique() > 1 and after.nunique() > 1
        rb, ra = before.rank(method='average'), after.rank(method='average')
        changes.append({'base_month': month, 'target_month': target, 'n_scores': len(m), 'n_clipped': int(m.clipped.sum()),
                        'rank_spearman': float(before.corr(after, method='spearman')) if ok else np.nan,
                        'mean_abs_rank_change': float((rb - ra).abs().mean()) if ok else np.nan,
                        'max_abs_rank_change': float((rb - ra).abs().max()) if ok else np.nan,
                        'max_abs_z_before': before.abs().max(), 'max_abs_z_after': after.abs().max(),
                        'rank_reason': '' if ok else 'unavailable_or_constant'})
        elig = part.eligible.eq(True).fillna(False).to_numpy(bool)
        sd = float(m.sqrt_v.std(ddof=1)) if len(m) >= 2 else np.nan
        coverage.append({'base_month': month, 'target_month': target, 'eligible': int(elig.sum()),
                         'vol_available_eligible': int((elig & part.vol_available.to_numpy(bool)).sum()),
                         'h3': int(member.sum()), 'pairs': int((member & part.y.notna().to_numpy()).sum()),
                         'score_month_valid': bool(len(m) >= 2 and np.isfinite(sd) and sd > 0)})
        if len(m) >= 2:
            # 독립 재계산: numpy 표본표준편차로 같은 집합의 Z를 다시 만든다.
            s = m.sqrt_v.to_numpy(float)
            ref = (s - s.mean()) / s.std(ddof=1)
            z_checks.append({'base_month': month, 'max_abs_diff': float(np.max(np.abs(ref - m.h3.to_numpy(float)))),
                             'z_mean': float(m.h3.mean()), 'z_sd': float(m.h3.std(ddof=1)),
                             'zw_mean': float(after.mean()), 'zw_sd': float(after.std(ddof=1))})
    ic = pd.DataFrame(ic_rows, columns=H1_MONTHLY_IC_COLUMNS)
    cov = pd.DataFrame(coverage)
    ic_valid = ic[ic.candidate.eq('h3')].set_index('base_month').ic.notna()
    cov['ic_month_valid'] = cov.base_month.map(ic_valid)
    summary = []
    for cand, part in ic.groupby('candidate', sort=False):
        for year, x in [('2001~2021', part), *part.groupby(part.target_month.str[:4], sort=True)]:
            summary.append({'target_year': year, 'candidate': cand, **metrics(x.ic), 'unable_months': int(x.ic.isna().sum())})
    compare = pd.DataFrame(changes)
    wide = ic.pivot(index='target_month', columns='candidate', values='ic')
    delta = wide.h3_winsor - wide.h3
    compare['ic_raw'] = compare.target_month.map(wide.h3)
    compare['ic_winsor'] = compare.target_month.map(wide.h3_winsor)
    compare['delta_ic'] = compare.target_month.map(delta)
    valid = wide.dropna().index
    candidates = pd.DataFrame([{'candidate': c, **metrics(wide.loc[valid, c]), 'unable_months': len(wide) - len(valid),
                                'mean_delta_ic': float(delta.loc[valid].mean()) if c == 'h3_winsor' else 0.,
                                'mean_rank_spearman': float(compare.rank_spearman.mean()) if c == 'h3_winsor' else 1.,
                                'mean_abs_rank_change': float(compare.mean_abs_rank_change.mean()) if c == 'h3_winsor' else 0.,
                                'mean_clipped_per_month': float(compare.n_clipped.mean()) if c == 'h3_winsor' else 0.,
                                'mean_max_abs_z': float(compare['max_abs_z_after' if c == 'h3_winsor' else 'max_abs_z_before'].mean())}
                               for c in CANDIDATES])
    m = cov.assign(year=cov.target_month.str[:4])
    agg = dict(score_months=('score_month_valid', 'sum'), ic_months=('ic_month_valid', 'sum'), months=('base_month', 'size'),
               eligible=('eligible', 'mean'), h3=('h3', 'mean'), pairs=('pairs', 'mean'))
    by_year = pd.concat([m.groupby('year').agg(**agg).reset_index(), m.assign(year='2001~2021').groupby('year').agg(**agg).reset_index()],
                        ignore_index=True)
    by_year['h3_share_pct'] = 100 * by_year.h3 / by_year.eligible
    zc = pd.DataFrame(z_checks)
    n_raw = ic[ic.candidate.eq('h3')].set_index('target_month').n
    n_w = ic[ic.candidate.eq('h3_winsor')].set_index('target_month').n
    # 독립 Spearman 대조(scipy)
    from scipy.stats import spearmanr
    ind = []
    for month, part in t.groupby('base_month', sort=True):
        x = part[part.h3_available & part.y.notna()]
        if len(x) >= 3:
            ind.append(abs(spearmanr(x.h3, x.y).statistic - ic.loc[ic.candidate.eq('h3') & ic.base_month.eq(month), 'ic'].iloc[0]))
    checks = {'paired_sample_equal': bool(n_raw.equals(n_w)),
              'h3_count_equals_eligible_vol_available': bool(cov.h3.eq(cov.vol_available_eligible).all()),
              'z_independent_recompute': bool(zc.max_abs_diff.max() < 1e-12),
              'z_mean0_sd1': bool(zc.z_mean.abs().max() < 1e-12 and (zc.z_sd - 1).abs().max() < 1e-12
                                  and zc.zw_mean.abs().max() < 1e-12 and (zc.zw_sd - 1).abs().max() < 1e-12),
              'spearman_independent_scipy': bool(max(ind) < 1e-12),
              'monthly_rows_preserved': len(ic) == 2 * t.base_month.nunique() == 2 * 252,
              'no_test_targets': bool(ic.target_month.between('2001-01', '2021-12').all()),
              'winsor_preserves_member_order': bool(compare.rank_spearman.dropna().ge(.99).all())}
    if not all(checks.values()):
        raise ValueError('h3_analysis_checks_failed:' + str(checks))
    rep = ic[ic.target_month.str.startswith(str(representative_year))].pivot(
        index=['target_month', 'base_month', 'n'], columns='candidate', values='ic').reset_index()
    tables = {'monthly_ic': ic, 'ic_summary_by_target_year': pd.DataFrame(summary), 'sigma_distribution': pd.DataFrame(dist),
              'extreme_values': pd.DataFrame(extreme), 'coverage_monthly': cov, 'coverage_by_target_year': by_year,
              'winsor_compare': compare, 'candidate_compare': candidates, 'representative_year': rep}
    stats = {'max_z_recompute_diff': float(zc.max_abs_diff.max()), 'max_scipy_ic_diff': float(max(ind)),
             'h3_cells': int(cov.h3.sum()), 'ic_pairs': int(n_raw.sum()), 'ic_valid_months': int(wide.h3.notna().sum())}
    return tables, checks, stats


def draw_figures(tables, folder):
    from src.features.observe import setup_matplotlib
    plt = setup_matplotlib()
    files = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name)
        plt.close(fig)
        files.append(name)
    d = tables['sigma_distribution']
    raw = d[d.variant.eq('raw')]
    x = pd.to_datetime(raw.target_month)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.fill_between(x, raw.q01, raw.q99, alpha=.12, label='1~99 분위수')
    ax.fill_between(x, raw.q25, raw.q75, alpha=.3, label='25~75 분위수')
    ax.plot(x, raw.q50, label='중앙값')
    ax.plot(x, raw['min'], lw=.7, alpha=.6, label='최소·최대')
    ax.plot(x, raw['max'], lw=.7, alpha=.6)
    ax.set(title='월별 변동성 원값 sqrt(v) 분포 · H3 표준화 집합', xlabel='목표월', ylabel='월 변동성(소수)')
    ax.legend(fontsize=8, ncol=4)
    save(fig, '01_변동성분포_극단값.png')
    ic = tables['monthly_ic']
    r = ic[ic.candidate.eq('h3')]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(pd.to_datetime(r.target_month), r.ic, lw=1, label='월별')
    ax.plot(pd.to_datetime(r.target_month), r.ic.rolling(12, min_periods=12).mean(), lw=2, label='최근 12개월 평균')
    ax.axhline(0, color='gray', lw=.8)
    ax.set(title='H3 원값의 월별 순위 상관(예상 방향: 음수)', xlabel='목표월', ylabel='Spearman 순위 상관')
    ax.legend()
    save(fig, '02_H3_월별순위상관.png')
    w = d[d.variant.eq('winsor')]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x, raw['max'], label='제한 전 최댓값')
    ax.plot(x, raw.q99, ls='--', alpha=.7, label='제한 전 99 분위수')
    ax.plot(pd.to_datetime(w.target_month), w['max'], label='제한 후 최댓값')
    ax.plot(x, raw['min'], lw=.8, label='제한 전 최솟값')
    ax.plot(pd.to_datetime(w.target_month), w['min'], lw=.8, label='제한 후 최솟값')
    ax.set(title='극단값 제한 전후 변동성 원값 분포', xlabel='목표월', ylabel='월 변동성(소수)')
    ax.legend(ncol=3, fontsize=8)
    save(fig, '03_극단값제한_전후분포.png')
    c = tables['winsor_compare']
    xc = pd.to_datetime(c.target_month)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(xc, c.rank_spearman)
    axes[0].set_ylabel('전후 순위 상관')
    axes[1].plot(xc, c.mean_abs_rank_change)
    axes[1].set_ylabel('평균 순위 이동 폭')
    axes[2].plot(xc, c.delta_ic)
    axes[2].axhline(0, color='gray', lw=.8)
    axes[2].set_ylabel('순위 상관 차이(제한-원값)')
    axes[2].set_xlabel('목표월')
    axes[0].set_title('극단값 제한 전후 순위 변화와 월별 성능 차이')
    save(fig, '04_순위변화_월별차이.png')
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(xc, c.max_abs_z_before, label='원값')
    ax.plot(xc, c.max_abs_z_after, label='극단값 제한')
    ax.set(title='월별 H3 표준화 점수 최대 절댓값 전후', xlabel='목표월', ylabel='최대 절댓값')
    ax.legend()
    save(fig, '05_최대표준점수_전후.png')
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for cand in CANDIDATES:
        s = ic[ic.candidate.eq(cand)]
        ax.plot(pd.to_datetime(s.target_month), s.ic, lw=1, label=LABELS[cand])
    ax.axhline(0, color='gray', lw=.8)
    ax.set(title='동일 표본의 원값·극단값 제한 H3 월별 순위 상관', xlabel='목표월', ylabel='Spearman 순위 상관')
    ax.legend()
    save(fig, '06_원값_제한_월별순위상관.png')
    return files


def report_markdown(tables, stats, run_dir, figures):
    text = '\n\n## H3 분석 · 분포·순위 상관·극단값 제한 비교\n\n실행: `' + run_dir.as_posix() + '`\n\n'
    text += ('H3 = 당월 적격·산출 종목의 sqrt(v) 월별 Z(ddof=1). v는 λ=0.97 EWMA(최초 유효 수익률 10개 표본분산 초기화, 기준월 t 포함), '
             '최근 12개월 유효 10개 이상일 때만 산출. KOSPI 비상장(월말 비적격이고 그 달 수익률 없음) 4번째 달에 상태 초기화. '
             '극단값 제한은 당월 H3 표준화 집합의 sqrt(v)를 1·99 분위수(선형 보간)로 자른 뒤 다시 표준화. 목표(차월 제공 수익률)는 그 뒤에 결합. 예상 방향은 음수.\n\n')
    cov = tables['coverage_by_target_year'].rename(columns={
        'year': '연도(목표월)', 'score_months': 'H3 산출 가능 월 수', 'ic_months': 'IC 계산 가능 월 수', 'months': '월 수',
        'eligible': '월평균 적격 종목', 'h3': 'H3 월평균', 'h3_share_pct': 'H3/적격(%)', 'pairs': 'IC 계산 쌍 월평균'})
    text += '### 커버리지 · 목표월 연도\n\n' + markdown_table(cov[['연도(목표월)', 'H3 산출 가능 월 수', 'IC 계산 가능 월 수', '월평균 적격 종목',
                                                         'H3 월평균', 'H3/적격(%)', 'IC 계산 쌍 월평균']].map(
        lambda v: f'{v:.1f}' if isinstance(v, (float, np.floating)) else v)) + '\n'
    text += f"H3 산출 칸 {stats['h3_cells']:,}, IC 계산 쌍 {stats['ic_pairs']:,}, IC 유효 월 {stats['ic_valid_months']}.\n\n"
    rep = tables['representative_year'].rename(columns={'target_month': '목표월', 'base_month': '기준월', 'n': 'IC 계산 종목 수',
                                                       'h3': 'H3 원값 IC', 'h3_winsor': '제한 H3 IC'})
    text += '### 대표 연도 월별 결과 — 2021년\n\n' + markdown_table(rep) + '\n'
    s = tables['ic_summary_by_target_year']
    wide = s.pivot(index='target_year', columns='candidate', values='mean_ic').reset_index()
    wide['valid'] = wide.target_year.map(s[s.candidate.eq('h3')].set_index('target_year').valid_months)
    wide = wide.sort_values('target_year', key=lambda y: y.eq('2001~2021') * 10000 + y.str[:4].astype(int))
    text += '### 목표월 연도별 평균 Rank IC\n\n' + markdown_table(wide[['target_year', 'valid', 'h3', 'h3_winsor']].rename(columns={
        'target_year': '목표 수익률 연도', 'valid': 'IC 유효 월 수', 'h3': 'H3 원값 평균 IC', 'h3_winsor': '제한 H3 평균 IC'})) + '\n'
    d = tables['sigma_distribution']
    last = d[d.base_month.eq(BASE_LAST)].set_index('variant')
    cols = ['n', 'min', 'q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99', 'max', 'skewness', 'excess_kurtosis']
    text += '### 변동성 원값 분포 — 기준월 2021-11\n\n' + markdown_table(
        last[cols].reset_index().replace({'variant': {'raw': '제한 전', 'winsor': '제한 후'}}).rename(columns={'variant': '구분'})) + '\n'
    full = d.groupby('variant')[['skewness', 'excess_kurtosis']].mean().reset_index().replace(
        {'variant': {'raw': '제한 전', 'winsor': '제한 후'}})
    text += '### 왜도·초과첨도 월평균\n\n' + markdown_table(full.rename(columns={'variant': '구분', 'skewness': '왜도', 'excess_kurtosis': '초과첨도'})) + '\n'
    c = tables['candidate_compare'].copy()
    c['candidate'] = c.candidate.map(LABELS)
    text += '### 같은 표본의 극단값 제한 전후\n\n' + markdown_table(c.rename(columns={
        'candidate': '방식', 'valid_months': '유효 월 수', 'mean_ic': '평균 Rank IC', 'ic_sd': 'IC 표본표준편차', 'icir': 'ICIR',
        'positive_ic_fraction': 'IC 양수 월 비율', 'negative_ic_fraction': 'IC 음수 월 비율', 'unable_months': '불능 월 수',
        'mean_delta_ic': '원값 대비 평균 ΔIC', 'mean_rank_spearman': '처리 전후 점수 순위 상관', 'mean_abs_rank_change': '평균 절대 순위 이동',
        'mean_clipped_per_month': '월평균 제한 종목 수', 'mean_max_abs_z': '월별 최대 |Z| 평균'})) + '\n'
    text += ('1·99 분위수 제한은 순서를 바꾸지 않는 변환이라 제한 구간 안 종목들이 같은 값(동순위)이 되는 것 말고는 순위가 그대로다. '
             '따라서 Rank IC 차이는 양 끝 약 2% 종목의 동순위 처리에서만 생긴다.\n\n')
    text += '### 산출 그림\n\n' + ''.join('- `' + (run_dir / n).as_posix() + '`\n' for n in figures)
    text += ('\n### 한계\n\n개발기간의 기술적 비교이며 가설 지지 여부를 판정하지 않는다. 평균 IC=0 검정(HAC)은 별도 승인 후 실행한다. '
             '4개월 공백 판단은 KOSPI 상장만 보며 다른 시장 상장은 보지 않는다. 1,086개 사후 고정 종목의 표본선택 한계는 유지된다.\n')
    return text


def _settings(config, key):
    s = config.get(key) or {}
    if not s.get('approved'):
        raise PermissionError(key.lower() + '_approval_required')
    return s


def check_h3_inputs(config, *, root=Path('.')):
    from src.pipeline import check_ingest_current
    check_h3_config(config)
    return check_ingest_current(config, root=root)


def _load(config, root):
    paths = config['PATHS']
    panel = pd.read_parquet(root / paths['panel'])
    elig = pd.read_parquet(root / paths['eligibility'])
    return panel, elig


def run_analysis(config, *, root=Path('.'), preconditions):
    settings = _settings(config, 'H3_TEST')
    started = datetime.now()
    panel, elig = _load(config, root)
    features, states, history, _ = compute_h3(panel, elig)
    winsor = winsorize_h3(features)
    tables, checks, stats = analyze(features, winsor, panel, representative_year=settings['representative_year'])
    folder = root / config['PATHS']['run_records'] / settings['run_category'] / started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(folder / (name + '.csv'), index=False, encoding='utf-8-sig')
    resets = states.loc[states.reset, ['stock_id', 'base_month', 'reset_basis']]
    resets.to_csv(folder / 'resets.csv', index=False, encoding='utf-8-sig')
    figures = draw_figures(tables, folder)
    code = ['src/evaluation/h3_analysis.py', 'src/evaluation/rank_ic.py', 'src/features/h3.py', 'src/preprocessing/listing.py',
            'src/preprocessing/panel.py', 'src/features/observe.py', 'src/contracts.py', 'src/pipeline.py']
    record = {'status': 'EXECUTED', 'started': started.isoformat(), 'command': ' '.join(sys.argv), 'preconditions': preconditions,
              'approved_decisions': dict(settings), 'h3_config': dict(config['H3']),
              'input_sha256': {config['PATHS'][n]: sha256(root / config['PATHS'][n]) for n in ('panel', 'eligibility')},
              'code_sha256': {p: sha256(root / p) for p in code}, 'checks': checks, 'stats': stats,
              'resets_total': len(resets), 'table_rows': {n: len(t) for n, t in tables.items()}, 'figures': figures,
              'method': {'year_axis': 'target_month', 'quantile_interpolation': 'linear', 'winsor_sample': 'monthly H3 standardization set',
                         'skew': 'unbiased sample', 'kurtosis': 'unbiased Fisher excess', 'expected_sign': 'negative',
                         'listing_reset': 'KOSPI only, 4th unlisted month (month-end ineligible and no return)', 'hac': 'not executed'},
              'outputs_sha256': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}
    (folder / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    note = root / settings['notion_tables_path']
    append_report(note, '\n\n## H3 분석 · 분포·순위 상관·극단값 제한 비교', report_markdown(tables, stats, folder.relative_to(root), figures))
    return {'run_dir': folder.as_posix(), 'checks': checks, 'stats': stats,
            'candidate_compare': tables['candidate_compare'].to_dict(orient='records'), 'notion_tables_path': note.as_posix()}


# ---------- 4) 추가 분석: H3와 H1의 겹침(2026-11-08, 가설검정 뒤) ----------

OVERLAP_PAIRS = {'h1': 'H1', 'neg_z_size': '−Z(Size)', 'z_bm': 'Z(B/M)', 'z_mom': 'Z(Mom)'}


def overlap_analysis(h1, h3, panel):
    """공유 특징표(h1.parquet·h3.parquet)만 받아 월별 상관과 같은 표본의 IC를 계산한다.

    공통 집합 = 당월 H1·H3 모두 산출된 종목. 상관은 목표 없이 계산하고, IC는 그 뒤 차월 수익률을 결합한다.
    """
    keys = ['stock_id', 'base_month']
    a = h1.loc[h1.base_month.between(BASE_FIRST, BASE_LAST), [*keys, 'h1', 'z_size', 'z_bm', 'z_mom', 'h1_available']]
    b = h3.loc[h3.base_month.between(BASE_FIRST, BASE_LAST), [*keys, 'h3', 'h3_available']]
    f = a.merge(b, on=keys, how='inner', validate='one_to_one').assign(neg_z_size=lambda d: -d.z_size)
    f['common'] = f.h1_available & f.h3_available
    returns = panel[['stock_id', 'month', 'return']].copy()
    returns.attrs['units'] = {'return': 'decimal'}
    corr_rows = []
    for month, part in f[f.common].groupby('base_month', sort=True):
        for col in OVERLAP_PAIRS:
            ok = len(part) >= 3 and part[col].nunique() > 1
            corr_rows.append({'base_month': month, 'target_month': str(pd.Period(month, freq='M') + 1), 'feature': col, 'n': len(part),
                              'pearson': float(part.h3.corr(part[col])) if ok else np.nan,
                              'spearman': float(part.h3.corr(part[col], method='spearman')) if ok else np.nan})
    corr = pd.DataFrame(corr_rows)
    # 상관 계산을 마친 뒤에만 목표를 결합한다.
    t = join_training_targets(f[[*keys, 'h1', 'h3', 'h1_available', 'h3_available', 'common']], returns)
    group = {'model': 'feature_score', 'fold': 'development'}
    ic_rows = []
    for month, part in t.groupby('base_month', sort=True):
        for cand, scope, mask in (('h3', 'h3_own', part.h3_available), ('h3', 'h1_h3_common', part.common),
                                  ('h1', 'h1_own', part.h1_available), ('h1', 'h1_h3_common', part.common)):
            x = part[['stock_id', 'target_month', 'y']].copy()
            x['score'] = part[cand].where(mask)
            for k, v in {**group, 'candidate': cand}.items():
                x[k] = v
            ic_rows.append({**rank_ic_for_month(x, score_column='score').iloc[0].to_dict(), 'base_month': month, 'sample_scope': scope})
    ic = pd.DataFrame(ic_rows, columns=H1_MONTHLY_IC_COLUMNS)
    year = corr.target_month.str[:4]
    corr_summary = pd.concat([
        corr.groupby('feature', sort=False)[['pearson', 'spearman']].agg(['mean', 'std', 'count']).pipe(
            lambda d: d.set_axis(['_'.join(c) for c in d.columns], axis=1)).reset_index().assign(target_year='2001~2021'),
        corr.groupby([year.rename('target_year'), 'feature'], sort=True)[['pearson', 'spearman']].mean().reset_index()],
        ignore_index=True)
    ic_summary = []
    for (cand, scope), part in ic.groupby(['candidate', 'sample_scope'], sort=False):
        common_months = ic[ic.sample_scope.eq('h1_h3_common')].pivot(index='target_month', columns='candidate', values='ic').dropna().index
        p = part[part.target_month.isin(common_months)] if scope == 'h1_h3_common' else part
        ic_summary.append({'candidate': cand, 'sample_scope': scope, **metrics(p.ic), 'pairs': int(p.n.sum()),
                           'unable_months': int(part.ic.isna().sum())})
    checks = {'common_pairs_equal': bool(ic[ic.sample_scope.eq('h1_h3_common') & ic.candidate.eq('h1')].n.to_numpy().tolist()
                                         == ic[ic.sample_scope.eq('h1_h3_common') & ic.candidate.eq('h3')].n.to_numpy().tolist()),
              'no_test_targets': bool(ic.target_month.between('2001-01', '2021-12').all()),
              'common_subset_of_both': bool((f.common <= f.h1_available).all() and (f.common <= f.h3_available).all())}
    if not all(checks.values()):
        raise ValueError('h3_overlap_checks_failed:' + str(checks))
    stats = {'common_cells': int(f.common.sum()), 'h1_cells': int(f.h1_available.sum()), 'h3_cells': int(f.h3_available.sum()),
             'corr_months': int(corr.base_month.nunique())}
    return {'overlap_corr_monthly': corr, 'overlap_corr_summary': corr_summary, 'overlap_monthly_ic': ic,
            'overlap_ic_summary': pd.DataFrame(ic_summary)}, checks, stats


def draw_overlap(tables, folder):
    from src.features.observe import COLORS, setup_matplotlib
    plt = setup_matplotlib()
    c = tables['overlap_corr_monthly']
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for (col, label), color in zip(OVERLAP_PAIRS.items(), [*COLORS, '#7a5cc4']):
        s = c[c.feature.eq(col)]
        ax.plot(pd.to_datetime(s.target_month), s.spearman, lw=1.2, label='H3 · ' + label.replace('−', '-'), color=color)
    ax.axhline(0, color='gray', lw=.8)
    ax.set(title='H3와 H1·구성요소의 월별 Spearman 상관 · H1·H3 공통 집합', xlabel='목표월', ylabel='Spearman 상관')
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / '07_H3_H1_월별상관.png')
    plt.close(fig)
    return ['07_H3_H1_월별상관.png']


def run_overlap(config, *, root=Path('.'), preconditions):
    settings = _settings(config, 'H3_TEST')
    started = datetime.now()
    paths = config['PATHS']
    h1 = pd.read_parquet(root / paths['h1'])
    h3 = pd.read_parquet(root / paths['h3'])
    panel = pd.read_parquet(root / paths['panel'])
    tables, checks, stats = overlap_analysis(h1, h3, panel)
    folder = root / paths['run_records'] / 'h3추가분석' / started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(folder / (name + '.csv'), index=False, encoding='utf-8-sig')
    figures = draw_overlap(tables, folder)
    record = {'status': 'EXECUTED', 'started': started.isoformat(), 'command': ' '.join(sys.argv), 'preconditions': preconditions,
              'checks': checks, 'stats': stats, 'figures': figures,
              'input_sha256': {paths[n]: sha256(root / paths[n]) for n in ('h1', 'h3', 'panel')},
              'code_sha256': {p: sha256(root / p) for p in ('src/evaluation/h3_analysis.py', 'src/evaluation/rank_ic.py')},
              'outputs_sha256': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}
    (folder / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    s = tables['overlap_corr_summary']
    overall = s[s.target_year.eq('2001~2021')].assign(feature=lambda d: d.feature.map(OVERLAP_PAIRS))
    text = ('\n\n## H3 추가 분석 · H1과의 겹침\n\n실행: `' + folder.relative_to(root).as_posix() + '`\n\n'
            f"공통 집합 = 당월 H1·H3 모두 산출된 적격 종목({stats['common_cells']:,}칸, {stats['corr_months']}개월). "
            '월별 상관을 계산해 단순평균했다. 상관은 목표 없이 계산했다.\n\n### H3와의 월별 상관 · 251개월 평균\n\n'
            + markdown_table(overall[['feature', 'pearson_mean', 'spearman_mean', 'spearman_std', 'spearman_count']].rename(columns={
                'feature': 'H3와 비교', 'pearson_mean': 'Pearson 평균', 'spearman_mean': 'Spearman 평균',
                'spearman_std': 'Spearman 월별 표준편차', 'spearman_count': '월 수'})))
    ics = tables['overlap_ic_summary'].replace({'candidate': {'h1': 'H1', 'h3': 'H3'}, 'sample_scope': {
        'h3_own': 'H3 자체 집합', 'h1_own': 'H1 자체 집합', 'h1_h3_common': 'H1·H3 공통 집합(공통 유효 월)'}})
    text += '\n### 같은 표본의 평균 Rank IC\n\n' + markdown_table(ics[['candidate', 'sample_scope', 'valid_months', 'mean_ic', 'ic_sd', 'icir', 'pairs']].rename(
        columns={'candidate': '지수', 'sample_scope': '표본', 'valid_months': '유효 월 수', 'mean_ic': '평균 Rank IC', 'ic_sd': 'IC 표본표준편차',
                 'icir': 'ICIR', 'pairs': 'IC 계산 쌍'}))
    text += '\n그림: ' + ''.join('`' + (folder.relative_to(root) / n).as_posix() + '`' for n in figures) + '\n'
    append_report(root / settings['notion_tables_path'], '\n\n## H3 추가 분석 · H1과의 겹침', text)
    return {'run_dir': folder.as_posix(), 'checks': checks, 'stats': stats,
            'corr': overall.to_dict(orient='records'), 'ic': tables['overlap_ic_summary'].to_dict(orient='records')}


# ---------- 3) 가설검정(별도 승인) ----------

def hypothesis_tables(monthly_ic):
    """원값 주검정, 제한값 평균 검정, 대응 차이 검정. 공통 유효 월만 시간순으로 쓴다."""
    wide = monthly_ic.pivot(index='target_month', columns='candidate', values='ic').sort_index()
    valid = wide[list(CANDIDATES)].dropna()
    excluded = wide.index.difference(valid.index)
    rows = []
    for name, series in (('h3', valid.h3), ('h3_winsor', valid.h3_winsor), ('delta_winsor_minus_raw', valid.h3_winsor - valid.h3)):
        rows.append({'test': name, **newey_west_mean_test(series.to_numpy())})
    tests = pd.DataFrame(rows)
    tests['holm'] = np.where(tests.test.eq('h3'), 'pending_H1_H4_family', 'single_test_no_family')
    return tests, list(excluded)


def run_hypothesis(config, *, root=Path('.'), preconditions):
    settings = _settings(config, 'H3_HYPOTHESIS')
    test_settings = _settings(config, 'H3_TEST')
    started = datetime.now()
    panel, elig = _load(config, root)
    features, _, _, _ = compute_h3(panel, elig)
    tables, checks, stats = analyze(features, winsorize_h3(features), panel, representative_year=test_settings['representative_year'])
    tests, excluded = hypothesis_tables(tables['monthly_ic'])
    folder = root / config['PATHS']['run_records'] / settings['run_category'] / started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True, exist_ok=False)
    tests.to_csv(folder / 'hac_tests.csv', index=False, encoding='utf-8-sig')
    tables['monthly_ic'].to_csv(folder / 'monthly_ic.csv', index=False, encoding='utf-8-sig')
    ex = tables['monthly_ic'][tables['monthly_ic'].target_month.isin(excluded)]
    ex.to_csv(folder / 'excluded_months.csv', index=False, encoding='utf-8-sig')
    record = {'status': 'EXECUTED', 'started': started.isoformat(), 'command': ' '.join(sys.argv), 'preconditions': preconditions,
              'approved_decisions': dict(settings), 'checks': checks, 'stats': stats, 'excluded_target_months': excluded,
              'input_sha256': {config['PATHS'][n]: sha256(root / config['PATHS'][n]) for n in ('panel', 'eligibility')},
              'code_sha256': {p: sha256(root / p) for p in ('src/evaluation/h3_analysis.py', 'src/evaluation/rank_ic.py', 'src/features/h3.py')},
              'outputs_sha256': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}
    (folder / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    label = {'h3': 'H3 원값 평균 IC=0 (주검정)', 'h3_winsor': 'H3 제한 평균 IC=0', 'delta_winsor_minus_raw': '제한−원값 ΔIC=0'}
    show = tests.assign(test=tests.test.map(label))[['test', 'valid_months', 'mean', 'sample_sd', 'hac_se', 'statistic', 'p_two_sided',
                                                      'ci95_low', 'ci95_high', 'lag', 'df', 'usual_t', 'usual_p_two_sided']]
    text = ('\n\n## H3 가설검정\n\n실행: `' + folder.relative_to(root).as_posix() + '`\n\n'
            'H0 평균=0, 양측 5%, Newey–West HAC(Bartlett), L=floor(4(T/100)^(2/9)), t(T−1), 소표본 배율 없음. 예상 방향 음수. '
            'H1~H4 Holm 보정은 네 결과가 모일 때까지 보류하고 원 p값만 보고한다. 일반 t는 참고값.\n\n' + markdown_table(show.rename(columns={
                'test': '검정', 'valid_months': '유효 월 수', 'mean': '평균', 'sample_sd': '표본표준편차', 'hac_se': 'HAC 표준오차',
                'statistic': '검정통계량', 'p_two_sided': '양측 p값', 'ci95_low': '95% 하한', 'ci95_high': '95% 상한', 'lag': 'L', 'df': '자유도',
                'usual_t': '일반 t', 'usual_p_two_sided': '일반 t p값'})))
    append_report(root / test_settings['notion_tables_path'], '\n\n## H3 가설검정', text)
    return {'run_dir': folder.as_posix(), 'tests': tests.to_dict(orient='records'), 'excluded_target_months': excluded}
