"""H1 Size 구성요소 사후 진단: 기존 H1(A)·Size 제외(B)·Size 부호 반전(C) 비교.

비교용 단독 실행이다. 기존 H1·특징표·설정을 바꾸지 않고, 결과로 후보를 추가하거나 고르지 않는다.
- 입력: data/processed/h1.parquet의 월별 Z(ddof=1)와 data/interim/panel.parquet의 개발 수익률.
- 세 지수는 같은 Z값에서 부호·분모만 바꿔 메모리에서 계산한다. 새로 표준화하지 않고 종목 집합도 넓히지 않는다.
- 목표값은 지수 계산을 마친 뒤 달력월 t+1로 결합한다.
- 검정: 대응 차이 IC_B-IC_A, IC_C-IC_A 두 개만 Bartlett HAC t(T-1) 양측, Holm 집합 크기 2.
  각 후보의 평균=0 검정은 하지 않는다(2026-11-09 결정).
- 연도는 목표월의 달력 연도.
- 저장: --execute일 때만 outputs/run_id/h1_size_test/<시각>/.
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

from src.evaluation.rank_ic import holm_bonferroni, newey_west_mean_test, rank_ic_for_month
from src.preprocessing.panel import join_training_targets

CANDIDATES = ('A', 'B', 'C')
LABELS = {'A': '기존 H1 (−Size+B/M+Mom)/3', 'B': 'Size 제외 (B/M+Mom)/2', 'C': 'Size 부호 반전 (+Size+B/M+Mom)/3'}
COEFFICIENTS = {'A': (-1, 1, 1, 3), 'B': (0, 1, 1, 2), 'C': (1, 1, 1, 3)}
DIFFERENCES = (('B', 'necessity', 'Size 필요성 (B−A)'), ('C', 'direction', 'Size 방향성 (C−A)'))
BASE_START, BASE_END = '2000-12', '2021-11'
FIRST_TARGET = '2001-01'
KEYS = ['stock_id', 'base_month']
EXPECTED_SETTINGS = {
    'approved': True, 'run_category': 'h1_size_test', 'candidates': list(CANDIDATES),
    'coefficients': {k: list(v) for k, v in COEFFICIENTS.items()}, 'baseline': 'A', 'alpha': .05,
    'reference_distribution': 't', 'df': 'T-1', 'lag_rule': 'floor(4*(T/100)^(2/9))', 'kernel': 'Bartlett',
    'small_sample_correction': False, 'holm_family': 'two_paired_differences_only', 'mean_zero_tests': False,
    'selected_candidate': None,
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def check_settings(config):
    """고정 범위 설정이 정확히 같을 때만 진행한다."""
    settings = config.get('H1_SIZE_TEST') or {}
    if any(settings.get(k) != v for k, v in EXPECTED_SETTINGS.items()):
        raise PermissionError('fixed_size_test_scope_approval_required')
    h1 = config.get('H1', {})
    if h1.get('coefficients') != {'size': -1, 'bm': 1, 'mom': 1} or h1.get('divisor') != 3:
        raise ValueError('baseline_h1_definition_changed')
    return settings


def candidate_scores(h1):
    """h1 표의 Z값으로 A·B·C를 계산한다. 목표값이 붙은 표는 받지 않는다."""
    if 'y' in h1 or 'target_month' in h1:
        raise ValueError('scores_before_target_join_required')
    if h1.duplicated(KEYS).any():
        raise ValueError('duplicate_h1_keys')
    f = h1[h1.base_month.between(BASE_START, BASE_END)].sort_values(KEYS).reset_index(drop=True)
    z = f[['z_size', 'z_bm', 'z_mom']].to_numpy(float)
    member = f.h1_available.eq(True).to_numpy()
    if not np.array_equal(np.isfinite(z).all(axis=1), member) or not np.array_equal(np.isfinite(f.h1), member):
        raise ValueError('h1_membership_mismatch')
    if not f.loc[member, 'eligible'].eq(True).all():
        raise ValueError('ineligible_scored_row')
    out = f[[*KEYS, 'eligible']].copy()
    for name, (a, b, c, d) in COEFFICIENTS.items():
        out[name] = (a * z[:, 0] + b * z[:, 1] + c * z[:, 2]) / d
    np.testing.assert_allclose(out.loc[member, 'A'], f.loc[member, 'h1'], rtol=1e-12, atol=1e-12)
    present = np.isfinite(out[list(CANDIDATES)].to_numpy(float))
    if not (present == member[:, None]).all():
        raise ValueError('candidate_membership_differs')
    diff = np.abs(out.loc[member, 'A'].to_numpy(float) - f.loc[member, 'h1'].to_numpy(float))
    return out, float(diff.max(initial=0.))


def monthly_ic(scored):
    """목표 결합된 표 -> 후보별 월별 IC와 월별 공통 쌍 감사. 세 후보의 쌍이 다르면 중단."""
    from scipy.stats import spearmanr
    rows, audits, pairs = [], [], []
    max_error = 0.
    for target, part in scored.groupby('target_month', sort=True):
        valid = np.isfinite(part['A'].to_numpy(float)) & np.isfinite(part.y.to_numpy(float))
        for name in CANDIDATES:
            if not np.array_equal(np.isfinite(part[name].to_numpy(float)) & np.isfinite(part.y.to_numpy(float)), valid):
                raise ValueError('candidate_actual_pairs_differ')
            t = part[['stock_id', 'target_month', 'y']].copy()
            t['score'] = part[name]
            t['model'], t['candidate'], t['fold'] = 'feature_score', name, 'development'
            row = rank_ic_for_month(t, score_column='score').iloc[0].to_dict()
            row['base_month'] = part.base_month.iloc[0]
            if not row['reason']:
                check = float(spearmanr(part.loc[valid, name], part.loc[valid, 'y']).statistic)
                max_error = max(max_error, abs(check - row['ic']))
            rows.append(row)
        sample = part.loc[valid, ['stock_id', 'target_month']].sort_values('stock_id')
        audits.append({'target_month': target, 'n': len(sample), 'candidate_count': len(CANDIDATES),
                       'pair_keys_sha256': hashlib.sha256(sample.to_csv(index=False).encode('utf-8')).hexdigest(),
                       'same_pairs_all_candidates': True})
        if len(sample):
            pairs.append(sample)
    if max_error > 1e-12:
        raise ValueError('independent_spearman_disagrees')
    common = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=['stock_id', 'target_month'])
    return pd.DataFrame(rows), pd.DataFrame(audits), common, max_error


def performance(series):
    s = pd.Series(series, dtype=float)
    sd = float(s.std(ddof=1))
    return {'valid_months': len(s), 'mean_ic': float(s.mean()), 'ic_sd': sd,
            'icir': float(s.mean() / sd) if sd > 0 else np.nan, 'positive_ic_fraction': float(s.gt(0).mean())}


def comparison_tables(monthly, *, expected_months, expected_pairs, first_target=FIRST_TARGET):
    """월별 IC -> 비교표·대응 차이 검정·월별/연도별 ΔIC·기간 안정성."""
    f = monthly.copy()
    if set(f.candidate) != set(CANDIDATES) or f.duplicated(['candidate', 'target_month']).any():
        raise ValueError('fixed_three_candidates_required')
    if not f.target_month.between('2001-01', '2021-12').all():
        raise PermissionError('development_target_months_only')
    excluded = f[f.target_month.eq(first_target)]
    if len(excluded) and (excluded.ic.notna().any() or not excluded.n.eq(0).all()):
        raise ValueError('first_target_month_exclusion_changed')
    f = f[~f.target_month.eq(first_target)]
    if f.ic.isna().any() or not f.reason.eq('').all():
        raise ValueError('missing_or_invalid_month')
    wide = f.pivot(index='target_month', columns='candidate', values='ic').reindex(columns=list(CANDIDATES)).sort_index()
    ns = f.pivot(index='target_month', columns='candidate', values='n').reindex(columns=list(CANDIDATES))
    expected_index = pd.period_range(wide.index[0], wide.index[-1], freq='M').astype(str).tolist()
    if wide.index.tolist() != expected_index or len(wide) != expected_months:
        raise ValueError('valid_months_mismatch')
    if ns.nunique(axis=1).max() != 1 or not ns.sum().eq(expected_pairs).all():
        raise ValueError('pair_counts_mismatch')

    perf = pd.DataFrame([{'candidate': c, 'label': LABELS[c], **performance(wide[c])} for c in CANDIDATES])
    tests = []
    for name, question, label in DIFFERENCES:
        result = newey_west_mean_test((wide[name] - wide['A']).to_numpy())
        tests.append({'candidate': name, 'question': question, 'label': label, **result})
    delta = pd.DataFrame(tests).rename(columns={'mean': 'mean_delta_ic', 'sample_sd': 'delta_ic_sd'})
    delta['holm_p'] = holm_bonferroni(delta.p_two_sided)
    delta['significant_holm_05'] = delta.holm_p.lt(.05)
    delta['holm_family_size'] = len(delta)

    month = wide.rename(columns=lambda c: 'ic_' + c).reset_index()
    month['n'] = ns['A'].to_numpy()
    month['delta_b'] = month.ic_B - month.ic_A
    month['delta_c'] = month.ic_C - month.ic_A
    month['cum_delta_b'] = month.delta_b.cumsum()
    month['cum_delta_c'] = month.delta_c.cumsum()
    month.insert(1, 'year', month.target_month.str[:4].astype(int))

    year = month.groupby('year').agg(months=('target_month', 'size'), n=('n', 'sum'),
        ic_A=('ic_A', 'mean'), ic_B=('ic_B', 'mean'), ic_C=('ic_C', 'mean'),
        delta_b=('delta_b', 'mean'), delta_c=('delta_c', 'mean'),
        positive_delta_b_months=('delta_b', lambda s: int(s.gt(0).sum())),
        positive_delta_c_months=('delta_c', lambda s: int(s.gt(0).sum()))).reset_index()

    stability = []
    for column, (name, question, label) in zip(('delta_b', 'delta_c'), DIFFERENCES):
        total = month[column].sum()
        by_year = month.groupby('year')[column].sum()
        loo = {y: (total - by_year[y]) / (len(month) - int(month.year.eq(y).sum())) for y in by_year.index}
        top = by_year.abs().idxmax()
        first_half = month.year.le(2010)
        stability.append({'candidate': name, 'label': label, 'mean_delta_ic': month[column].mean(),
            'positive_months': int(month[column].gt(0).sum()), 'months': len(month),
            'positive_years': int(year[column].gt(0).sum()), 'years': len(year),
            'mean_delta_2001_2010': month.loc[first_half, column].mean(),
            'mean_delta_2011_2021': month.loc[~first_half, column].mean(),
            'largest_abs_year': int(top), 'largest_abs_year_share_of_sum': float(by_year[top] / total) if total else np.nan,
            'leave_one_year_out_min': min(loo.values()), 'leave_one_year_out_max': max(loo.values()),
            'leave_one_year_out_sign_changes': bool(min(loo.values()) < 0 < max(loo.values()))})
    return {'performance_summary': perf, 'delta_hac_tests': delta, 'monthly_ic_delta': month,
            'yearly_ic_delta': year, 'period_stability': pd.DataFrame(stability)}


VERDICT_RULE = {
    'Holm 유의 & ΔIC<0 (A가 높음)': '근거 있음 — 기존 −Size 처리가 대안보다 높은 IC',
    'Holm 유의 & ΔIC>0 (대안이 높음)': '추가 검증 필요 — 대안이 높지만 개발 표본 사후 진단이므로 H1을 바꾸지 않음',
    'Holm 비유의': '근거 부족 — 기존 처리의 우위가 확인되지 않음(두 지수가 같다는 뜻은 아님)',
}


def interpretation(tables):
    """필요성(B−A)·방향성(C−A)을 각각 하나의 판정으로. 판정 규칙은 VERDICT_RULE."""
    out = {'rule': VERDICT_RULE, 'conclusions': {}, 'h1_change': '없음 — 기존 H1 잠정 유지, 자동 변경 없음',
           'limitations': ['2001~2021 개발 표본 사후 진단. 결과로 가중치·후보를 추가 탐색하지 않음.',
                           '2022~2025 보존 테스트 기간 미사용.',
                           'C는 사전 경제 가설(소형주 효과)과 반대 방향이므로 개발 표본 결과만으로 방향을 바꾸지 않음.',
                           '각 후보의 평균=0 검정은 범위 밖(대응 차이 2개만 Holm 보정).']}
    perf = tables['performance_summary'].set_index('candidate')
    for row in tables['delta_hac_tests'].itertuples(index=False):
        if row.significant_holm_05 and row.mean_delta_ic < 0:
            verdict = '근거 있음'
        elif row.significant_holm_05:
            verdict = '추가 검증 필요'
        else:
            verdict = '근거 부족'
        out['conclusions'][row.question] = {
            'comparison': row.label, 'verdict': verdict,
            'mean_ic_baseline': float(perf.loc['A', 'mean_ic']), 'mean_ic_alternative': float(perf.loc[row.candidate, 'mean_ic']),
            'mean_delta_ic': float(row.mean_delta_ic), 'ci95': [float(row.ci95_low), float(row.ci95_high)],
            'p_two_sided': float(row.p_two_sided), 'holm_p': float(row.holm_p),
            'text': (f'{row.label}: 평균 ΔIC {row.mean_delta_ic:+.6f}, 95% CI [{row.ci95_low:+.6f}, {row.ci95_high:+.6f}], '
                     f'p={row.p_two_sided:.4f}, Holm p={row.holm_p:.4f} → {verdict}.')}
    return out


def markdown_table(frame):
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if pd.isna(v):
            return '—'
        if isinstance(v, (float, np.floating)):
            return f'{v:.6f}'
        return str(v).replace('|', '/')
    head = '| ' + ' | '.join(map(str, frame.columns)) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n'
    return head + ''.join('| ' + ' | '.join(cell(v) for v in r) + ' |\n' for r in frame.itertuples(index=False, name=None))


def report_markdown(tables, interp, checks):
    perf = tables['performance_summary'][['candidate', 'label', 'valid_months', 'mean_ic', 'ic_sd', 'icir', 'positive_ic_fraction']]
    delta = tables['delta_hac_tests'][['label', 'mean_delta_ic', 'delta_ic_sd', 'hac_se', 'statistic', 'p_two_sided',
                                       'holm_p', 'ci95_low', 'ci95_high', 'lag', 'significant_holm_05']]
    stab = tables['period_stability'].drop(columns=['candidate'])
    year = tables['yearly_ic_delta'][['year', 'months', 'ic_A', 'ic_B', 'ic_C', 'delta_b', 'delta_c']]
    lines = ['# H1 Size 구성요소 진단 (A·B·C)', '',
             f"표본: 첫 목표월 2001-01 제외 후 {checks['valid_months']}개월, 종목·수익률 쌍 {checks['common_pairs']:,}개(세 후보 동일).",
             '연도는 목표월 연도. 검정은 Bartlett HAC t(T−1) 양측, 대응 차이 2개만 Holm 보정.', '',
             '## 성능 비교', '', markdown_table(perf), '## 기존 H1 대비 대응 차이 검정', '', markdown_table(delta),
             '## 결론', '']
    for key, title in (('necessity', '필요성'), ('direction', '방향성')):
        lines.append(f"- **{title}**: {interp['conclusions'][key]['text']}")
    lines += [f"- 기존 H1: {interp['h1_change']}", '', '판정 규칙:', '']
    lines += [f'- {k}: {v}' for k, v in interp['rule'].items()]
    lines += ['', '## 기간 안정성', '', markdown_table(stab), '## 연도별 평균 IC·ΔIC', '', markdown_table(year), '## 한계', '']
    lines += [f'- {x}' for x in interp['limitations']]
    return '\n'.join(lines) + '\n'


def run(root, config, *, execute=False):
    settings = check_settings(config)
    paths = config['PATHS']
    from src.pipeline import check_features_current
    pre = check_features_current(config, root=root)
    names = ('panel', 'h1')
    inputs = {str(Path(paths[n]).as_posix()): sha256(root / paths[n]) for n in names}
    h1 = pd.read_parquet(root / paths['h1'])
    scores, a_max_error = candidate_scores(h1)
    panel = pd.read_parquet(root / paths['panel'], columns=['stock_id', 'month', 'return'])
    panel.attrs['units'] = config['UNITS']
    scored = join_training_targets(scores, panel)  # 지수 계산을 끝낸 뒤 달력 t+1 결합
    monthly, audit, common, ic_error = monthly_ic(scored)
    tables = comparison_tables(monthly, expected_months=settings['expected_valid_months'],
                               expected_pairs=settings['expected_pairs'])
    if len(common) != settings['expected_pairs'] or common.duplicated(['stock_id', 'target_month']).any():
        raise ValueError('common_sample_count')
    tables['sample_audit'] = audit
    tables['monthly_ic_long'] = monthly
    interp = interpretation(tables)
    checks = {'valid_months': int(tables['performance_summary'].valid_months.iloc[0]), 'common_pairs': len(common),
              'same_pairs_all_candidates': True, 'excluded_target_month': FIRST_TARGET,
              'a_equals_stored_h1_max_abs_error': a_max_error, 'independent_scipy_ic_max_error': ic_error,
              'holm_family_size': 2, 'test_period_used': False}
    result = {'execute': execute, 'stage': 'h1_size_test', 'checks': checks,
              'performance': tables['performance_summary'][['candidate', 'mean_ic', 'icir']].to_dict('records'),
              'conclusions': {k: v['verdict'] for k, v in interp['conclusions'].items()}}
    if not execute:
        return result
    started = datetime.now()
    folder = root / paths['run_records'] / settings['run_category'] / started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(folder / (name + '.csv'), index=False, encoding='utf-8-sig')
    common.to_parquet(folder / 'common_sample.parquet', index=False)
    (folder / 'interpretation_summary.json').write_text(json.dumps(interp, ensure_ascii=False, indent=2), encoding='utf-8')
    (folder / 'report.md').write_text(report_markdown(tables, interp, checks), encoding='utf-8')
    for name, value in inputs.items():
        if sha256(root / name) != value:
            raise RuntimeError('source_changed_during_run:' + name)
    code = ['src/evaluation/h1_size_test.py', 'src/evaluation/rank_ic.py', 'src/preprocessing/panel.py',
            'src/features/h1.py', 'src/contracts.py', 'config/experiment.yaml']
    record = {'status': 'EXECUTED', 'stage': 'h1_size_test', 'command': ' '.join(sys.argv), 'started': started.isoformat(),
              'approved_decisions': dict(settings), 'candidates': {c: {'label': LABELS[c], 'coefficients': COEFFICIENTS[c]} for c in CANDIDATES},
              'input_sha256': inputs, 'ingest': pre.get('preconditions', {}) if isinstance(pre, dict) else {},
              'code_sha256': {p: sha256(root / p) for p in code}, 'checks': checks,
              'method': {'scores': 'h1.parquet z_size/z_bm/z_mom 재사용, 새 표준화 없음',
                         'target': 'base_month t -> calendar t+1', 'ic': 'Spearman average ties, min 3 pairs',
                         'hac': 'Bartlett, L=floor(4*(T/100)^(2/9)), t(T-1), two-sided', 'year': 'target_month year'},
              'outputs_sha256': {p.name: sha256(p) for p in folder.iterdir()}}
    (folder / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return {**result, 'run_dir': str(folder)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='H1 Size 구성요소 사후 진단(A·B·C 비교)')
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / args.config).read_text(encoding='utf-8'))
    print(json.dumps(run(root, config, execute=args.execute), ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
