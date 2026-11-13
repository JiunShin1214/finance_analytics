"""OLS·Ridge 비교군: 확장 창 16개 Fold에서 원값·제한 판 × 주·참고 입력 비교(노션 11주차 5.4).

비교군이라 최종 선택 대상이 아니며 통계 검정은 하지 않는다. 단독 실행이다.
- 입력: data/processed/h1_components·h1·h3, data/interim/panel(개발 수익률). 과거 실행 기록은 읽지 않는다.
- 목표값: 차월 원 수정수익률. 평가는 예측값 순위와 원수익률의 월별 Spearman Rank IC.
- 학습: 세 특징·H3·차월 수익률이 모두 있는 행. 예측·평가: 그 달 수익률이 있는 모든 종목.
- 결측: 있는 원값은 그 달 공통 집합(H1 산출 집합)의 평균·표준편차로 Z 변환, 빠진 특징은 0. 예측에만 쓴다.
- 제한 판: 입력 원값·H3 σ를 그 달 표준화 집합의 1·99 분위수로 제한 후 다시 표준화, 학습 목표값도 그 달 학습 행의
  1·99 분위수로 제한. 평가는 원수익률. 경계는 모두 그 달 단면에서만 계산하므로 다른 달·Fold 정보가 섞이지 않는다.
- Ridge: 입력을 Fold 학습 행 평균·표준편차로 표준화한 뒤 벌점(glmnet standardize=TRUE). OLS는 크기와 무관해 표준화하지 않는다.
  Fold마다 학습 기간 마지막 내부 검증으로 λ 선택(평가 목표값은 원수익률) 후 전체 학습 기간으로 재학습. 계수는 원래 크기로 기록.
- 저장: --execute일 때만 outputs/run_id/linear_reg_test/<시각>/.
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

from src.evaluation.expanding_window import plan_annual_expanding_folds, verify_no_test_access
from src.evaluation.rank_ic import rank_ic_for_month
from src.models import ols, ridge
from src.models.interface import fit_fold_transformer, validate_fold_inputs
from src.preprocessing.panel import join_training_targets

BASE_FIRST, BASE_LAST = '2000-12', '2021-11'
RAW = ('size', 'bm', 'mom')
Z = ('z_size', 'z_bm', 'z_mom')
INPUT_SETS = {'main': ['z_size', 'z_bm', 'z_mom', 'h3'], 'reference': ['h1', 'h3']}
VARIANTS = ('raw', 'winsor')
MODELS = ('ols', 'ridge')
Q = (.01, .99)
RIDGE_SCALER = {'kind': 'standardize'}
PERIODS = {'2006~2010': (2006, 2010), '2011~2021': (2011, 2021), '2018~2021': (2018, 2021)}
EXPECTED_SETTINGS = {
    'approved': True, 'run_category': 'linear_reg_test', 'models': list(MODELS), 'target': 'raw_adjusted_monthly_return',
    'input_sets': INPUT_SETS, 'variants': list(VARIANTS), 'winsor_quantiles': list(Q),
    'winsor_scope': 'monthly_cross_section_train_target_and_inputs', 'training_rows': 'all_inputs_and_target_present',
    'prediction_rows': 'all_stocks_with_target_return', 'missing_feature_fill': 'common_set_zscore_then_zero_prediction_only',
    'validation_years': [2006, 2021], 'hypothesis_tests': False, 'selected_candidate': None,
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def check_settings(config):
    s = config.get('LINEAR_REG_TEST') or {}
    if any(s.get(k) != v for k, v in EXPECTED_SETTINGS.items()):
        raise PermissionError('fixed_linear_reg_scope_approval_required')
    ridge.enumerate_ridge_candidates(config=config)
    return s


# ---------- 1) 입력 조립(목표값 결합 전) ----------

def monthly_clip(frame, column, member, month='base_month'):
    """그 달 member 행의 1·99 분위수(선형 보간)로 같은 달 모든 행의 값을 제한한다. 경계가 없는 달은 그대로."""
    bounds = frame.loc[member].groupby(month)[column].quantile(list(Q), interpolation='linear').unstack()
    lo = frame[month].map(bounds[Q[0]]) if len(bounds) else pd.Series(np.nan, index=frame.index)
    hi = frame[month].map(bounds[Q[1]]) if len(bounds) else pd.Series(np.nan, index=frame.index)
    values = frame[column]
    return values.where(lo.isna(), values.clip(lower=lo, upper=hi))


def monthly_z(frame, column, member, month='base_month'):
    """그 달 member 행의 평균·표준편차(ddof=1)로 모든 행을 표준화한다. 값이 없거나 기준이 없으면 NaN."""
    stats = frame.loc[member].groupby(month)[column].agg(['mean', 'std'])
    mean, sd = frame[month].map(stats['mean']), frame[month].map(stats['std'])
    z = (frame[column] - mean) / sd
    return z.where(np.isfinite(z))


def build_inputs(components, h1, h3, *, variant):
    """기준월 2000-12~2021-11의 종목×월 입력. 반환 열: 키, eligible, z_*, h1, h3, member, filled.

    member = H1·H3가 모두 산출된 행(학습 후보). 그 밖의 행은 있는 값만 변환하고 빠진 특징은 0.
    """
    if variant not in VARIANTS:
        raise ValueError('unknown_variant')
    keys = ['stock_id', 'base_month']
    c = components[components.base_month.between(BASE_FIRST, BASE_LAST)].reset_index(drop=True)
    f = c[keys + ['eligible', *RAW, *[r + '_available' for r in RAW]]].merge(
        h1[keys + [*Z, 'h1', 'h1_available']], on=keys, how='left', validate='one_to_one').merge(
        h3[keys + ['sigma', 'h3', 'h3_available']], on=keys, how='left', validate='one_to_one')
    if len(f) != len(c):
        raise ValueError('input_rows_changed')
    h1m = f.h1_available.eq(True).to_numpy()
    h3m = f.h3_available.eq(True).to_numpy()
    out = f[keys + ['eligible']].copy()
    for r, zc in zip(RAW, Z):
        source = r
        if variant == 'winsor':
            f[r + '_w'] = monthly_clip(f, r, h1m)
            source = r + '_w'
        z = monthly_z(f, source, h1m)
        out[zc] = z.where(f[r + '_available'].eq(True)).fillna(0.0)
    sig = 'sigma'
    if variant == 'winsor':
        f['sigma_w'] = monthly_clip(f, 'sigma', h3m)
        sig = 'sigma_w'
    h3z = monthly_z(f, sig, h3m)
    out['h3'] = h3z.where(h3m).fillna(0.0)
    out['h1'] = (-out.z_size + out.z_bm + out.z_mom) / 3
    out['member'] = h1m & h3m
    filled = np.zeros(len(f), dtype=int)
    for r in RAW:
        filled += (~f[r + '_available'].eq(True)).to_numpy().astype(int)
    filled += (~h3m).astype(int)
    out['filled_features'] = filled
    checks = {}
    if variant == 'raw':
        err = np.abs(out.loc[h1m, list(Z) + ['h1']].to_numpy() - f.loc[h1m, list(Z) + ['h1']].to_numpy()).max()
        err3 = np.abs(out.loc[h3m, 'h3'].to_numpy() - f.loc[h3m, 'h3'].to_numpy()).max()
        if not (err < 1e-9 and err3 < 1e-9):
            raise ValueError('recomputed_z_differs_from_stored')
        checks['stored_z_max_abs_error'] = float(max(err, err3))
    else:
        from src.features.h1 import calculate_h1_score, winsorize_1_99
        ref = calculate_h1_score(winsorize_1_99(c)).set_index(keys).loc[pd.MultiIndex.from_frame(out[keys])]
        err = np.abs(out.loc[h1m, list(Z) + ['h1']].to_numpy() - ref.loc[h1m, list(Z) + ['h1']].to_numpy()).max()
        if not err < 1e-9:
            raise ValueError('winsor_z_differs_from_h1_winsorize')
        checks['h1_winsorize_max_abs_error'] = float(err)
    if (out.loc[out.member, 'filled_features'] != 0).any():
        raise ValueError('member_rows_must_not_be_filled')
    return out, checks


def attach_targets(inputs, panel, *, variant):
    """입력을 다 만든 뒤 달력 t+1 수익률을 붙이고 학습 목표값(y_fit)을 만든다.

    train = member이고 y가 있는 행, predict = y가 있는 모든 행.
    제한 판의 y_fit은 그 목표월 학습 행의 1·99 분위수로 제한, 평가 y는 원값.
    """
    out = join_training_targets(inputs, panel)
    out['train'] = out.member & out.y.notna()
    out['predict'] = out.y.notna()
    out['y_fit'] = out.y
    if variant == 'winsor':
        out['y_fit'] = monthly_clip(out, 'y', out.train.to_numpy(), month='target_month')
    out.loc[~out.train, 'y_fit'] = np.nan
    return out


# ---------- 2) Fold 실행 ----------

def run_folds(data, folds, candidates, *, variant, input_set):
    cols = INPUT_SETS[input_set]
    preds, coefs, inner = [], [], []
    for fold in folds:
        tr = data[data.train & data.target_month.between(fold['train_start'], fold['train_end'])]
        va = data[data.predict & data.target_month.between(fold['valid_start'], fold['valid_end'])]
        trX, vaX = tr[cols], va[cols]
        trk, vak = tr[['stock_id', 'target_month']], va[['stock_id', 'target_month']]
        validate_fold_inputs(trX, tr.y_fit, vaX, trk, vak, fold=fold)
        transformer = fit_fold_transformer(trX, transformer_spec=None)
        cand_ols = ols.plan_ols_config()
        m_ols = ols.fit_ols_fold(trX, tr.y_fit, trk, fold=fold, candidate=cand_ols, transformer=transformer)
        p_ols = ols.predict_ols_fold(m_ols, vaX, vak, fold=fold, candidate=cand_ols, transformer=transformer)
        scaler = fit_fold_transformer(trX, transformer_spec=RIDGE_SCALER)
        chosen, inner_ic = ridge.select_ridge_lambda(trX, tr.y_fit, trk, fold=fold, candidates=candidates,
                                                     transformer_spec=RIDGE_SCALER, eval_y=tr.y)
        m_r = ridge.fit_ridge_fold(trX, tr.y_fit, trk, fold=fold, candidate=chosen, transformer=scaler)
        p_r = ridge.predict_ridge_fold(m_r, vaX, vak, fold=fold, candidate=chosen, transformer=scaler)
        for model, pred in (('ols', p_ols), ('ridge', p_r)):
            pred = pred.assign(model=model, y=va.y.to_numpy(), common=va.member.to_numpy(),
                               filled=va.filled_features.to_numpy() > 0)
            preds.append(pred)
        for model, m in (('ols', m_ols), ('ridge', m_r)):
            coef = m.get('coef_original', m['coef'])
            intercept = m.get('intercept_original', m['intercept'])
            coefs.append({'model': model, 'fold': fold['fold'], 'n_train': m['n_train'], 'intercept': intercept,
                          'lambda': m.get('lambda', np.nan), 'alpha': m.get('alpha', np.nan),
                          **{'coef_' + c: v for c, v in zip(cols, coef)}})
        summary = inner_ic.groupby('lambda').ic.agg(inner_mean_ic='mean', inner_months='count').reset_index()
        summary['chosen'] = summary['lambda'].eq(chosen['parameters']['lambda'])
        inner.append(summary.assign(fold=fold['fold']))
    tag = {'variant': variant, 'input_set': input_set}
    return (pd.concat(preds, ignore_index=True).assign(**tag), pd.DataFrame(coefs).assign(**tag),
            pd.concat(inner, ignore_index=True).assign(**tag))


def monthly_ic(predictions, score='yhat', group=('variant', 'input_set', 'model')):
    """조합·목표월별 전체 종목 IC와 공통 표본 IC. scipy 독립 계산과 대조한다."""
    from scipy.stats import spearmanr
    rows, max_err = [], 0.
    for key, part in predictions.groupby([*group, 'target_month'], sort=True):
        for scope, rows_ in (('full', part), ('common', part[part.common])):
            t = rows_[['stock_id', 'target_month', 'y', score]].assign(model='linear', candidate='|'.join(key[:-1]), fold=scope)
            r = rank_ic_for_month(t, score_column=score).iloc[0].to_dict()
            if not r['reason']:
                ok = np.isfinite(rows_[score]) & np.isfinite(rows_.y)
                max_err = max(max_err, abs(float(spearmanr(rows_.loc[ok, score], rows_.loc[ok, 'y']).statistic) - r['ic']))
            rows.append({**dict(zip(group, key[:-1])), 'target_month': key[-1], 'scope': scope, 'n': r['n'], 'ic': r['ic'],
                         'reason': r['reason']})
    if max_err > 1e-12:
        raise ValueError('independent_spearman_disagrees')
    return pd.DataFrame(rows), max_err


# ---------- 3) 보고표 ----------

def metrics(series):
    s = pd.Series(series, dtype=float)
    sd = float(s.std(ddof=1))
    return {'valid_months': len(s), 'mean_ic': float(s.mean()), 'ic_sd': sd,
            'icir': float(s.mean() / sd) if sd > 0 else np.nan, 'positive_ic_fraction': float(s.gt(0).mean())}


def stability(series_by_month):
    s = series_by_month.sort_index()
    year = pd.Series(s.index.str[:4].astype(int), index=s.index)
    by_year_sum, total, n = s.groupby(year).sum(), s.sum(), len(s)
    loo = [(total - by_year_sum[y]) / (n - int(year.eq(y).sum())) for y in by_year_sum.index]
    drop5 = s.loc[s.abs().sort_values().index[:-5]].mean()
    row = {f'mean_{k}': float(s[year.between(*v)].mean()) for k, v in PERIODS.items()}
    row.update(positive_years=int((s.groupby(year).mean() > 0).sum()), years=int(year.nunique()),
               leave_one_year_out_min=float(min(loo)), leave_one_year_out_max=float(max(loo)),
               drop_top5_abs_months_mean=float(drop5))
    return row


def report_tables(ic, index_ic, coefs, inner, coverage):
    keys = ['variant', 'input_set', 'model']
    perf, stab, yearly = [], [], []
    for key, part in ic.groupby(keys + ['scope'], sort=True):
        s = part.set_index('target_month').ic
        perf.append({**dict(zip(keys + ['scope'], key)), **metrics(s)})
        if key[-1] == 'full':
            stab.append({**dict(zip(keys, key[:-1])), **stability(s)})
            y = s.groupby(s.index.str[:4]).mean()
            yearly.append(pd.DataFrame({**dict(zip(keys, key[:-1])), 'year': y.index.astype(int), 'mean_ic': y.to_numpy()}))
    base = []
    for key, part in index_ic.groupby(['score_index', 'scope'], sort=True):
        base.append({'score_index': key[0], 'scope': key[1], **metrics(part.ic)})
    lam = inner[inner.chosen][['variant', 'input_set', 'fold', 'lambda', 'inner_mean_ic']].rename(columns={'lambda': 'chosen_lambda'})
    return {'performance_summary': pd.DataFrame(perf), 'period_stability': pd.DataFrame(stab),
            'yearly_ic': pd.concat(yearly, ignore_index=True), 'index_baseline': pd.DataFrame(base),
            'fold_coefficients': coefs, 'ridge_lambda_selected': lam, 'ridge_inner_validation': inner,
            'prediction_coverage': coverage}


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


def report_markdown(tables, checks):
    perf = tables['performance_summary']
    lines = ['# OLS·Ridge 비교군 (linear_reg_test)', '',
             f"검증 {checks['validation_months']}개월(2006-01~2021-12), 학습 공통 표본 {checks['training_pairs_total']:,}쌍, "
             f"예측 {checks['prediction_rows_validation']:,}행(결측 채움 {checks['filled_rows_validation']:,}행).",
             '평가는 원수익률과의 월별 Spearman Rank IC. 검정 없음(비교군).', '',
             '## 성능 — 전체 종목', '', markdown_table(perf[perf.scope.eq('full')].drop(columns='scope')),
             '## 성능 — 공통 표본', '', markdown_table(perf[perf.scope.eq('common')].drop(columns='scope')),
             '## 기간 안정성 — 전체 종목', '', markdown_table(tables['period_stability']),
             '## 지수 그대로의 IC (같은 192개월)', '', markdown_table(tables['index_baseline']),
             '## Ridge 선택 λ', '', markdown_table(tables['ridge_lambda_selected'])]
    return '\n'.join(lines) + '\n'


# ---------- 4) 실행 ----------

def run(root, config, *, execute=False):
    settings = check_settings(config)
    paths = config['PATHS']
    verify_no_test_access(stage='validate', declared_target_years=list(range(2001, 2022)), source_scope_verified=True)
    from src.pipeline import check_features_current
    pre = check_features_current(config, root=root)
    names = ('panel', 'h1_components', 'h1', 'h3')
    inputs = {Path(paths[n]).as_posix(): sha256(root / paths[n]) for n in names}
    components = pd.read_parquet(root / paths['h1_components'])
    h1 = pd.read_parquet(root / paths['h1'])
    h3 = pd.read_parquet(root / paths['h3'])
    panel = pd.read_parquet(root / paths['panel'], columns=['stock_id', 'month', 'return'])
    panel.attrs['units'] = config['UNITS']
    folds = plan_annual_expanding_folds(config=config)
    candidates = ridge.enumerate_ridge_candidates(config=config)
    preds, coefs, inner, checks, coverage = [], [], [], {}, None
    for variant in VARIANTS:
        x, c = build_inputs(components, h1, h3, variant=variant)
        checks.update({f'{variant}_{k}': v for k, v in c.items()})
        data = attach_targets(x, panel, variant=variant)
        if data.target_month.ge('2022-01').any():
            raise PermissionError('test_period_in_samples')
        if variant == 'raw':
            checks['training_pairs_total'] = int(data.train.sum())
            if checks['training_pairs_total'] != settings['expected_training_pairs']:
                raise ValueError('training_pairs_changed')
            v = data[data.predict & data.target_month.between('2006-01', '2021-12')]
            coverage = v.groupby('target_month').agg(predicted=('stock_id', 'size'), common=('member', 'sum'),
                                                     filled=('filled_features', lambda s: int((s > 0).sum()))).reset_index()
            checks['prediction_rows_validation'] = int(len(v))
            checks['filled_rows_validation'] = int((v.filled_features > 0).sum())
            index_rows = v.assign(common=v.member)
        for input_set in INPUT_SETS:
            p, cf, inn = run_folds(data, folds, candidates, variant=variant, input_set=input_set)
            preds.append(p); coefs.append(cf); inner.append(inn)
    predictions = pd.concat(preds, ignore_index=True)
    ic, err = monthly_ic(predictions)
    checks['independent_scipy_ic_max_error'] = err
    months = ic.groupby(['variant', 'input_set', 'model', 'scope']).target_month.nunique()
    if not months.eq(settings['expected_validation_months']).all() or ic.ic.isna().any():
        raise ValueError('validation_months_incomplete')
    checks['validation_months'] = int(months.iloc[0])
    per_month = predictions.groupby(['variant', 'input_set', 'model', 'target_month']).size().unstack(['variant', 'input_set', 'model'])
    expected = coverage.set_index('target_month').predicted
    if not per_month.apply(lambda col: col.equals(expected.reindex(col.index))).all():
        raise ValueError('prediction_counts_differ_from_stocks_with_returns')
    checks['all_stocks_with_returns_predicted'] = True
    idx = pd.concat([index_rows.assign(score_index=name, yhat_index=index_rows[name]) for name in ('h1', 'h3')], ignore_index=True)
    index_ic, _ = monthly_ic(idx, score='yhat_index', group=('score_index',))
    tables = report_tables(ic, index_ic, pd.concat(coefs, ignore_index=True), pd.concat(inner, ignore_index=True), coverage)
    tables['monthly_ic'] = ic
    checks.update(test_period_used=False, folds=len(folds), combinations=len(VARIANTS) * len(INPUT_SETS) * len(MODELS))
    perf = tables['performance_summary']
    result = {'execute': execute, 'stage': 'linear_reg_test', 'checks': checks,
              'mean_ic': {f"{r.variant}|{r.input_set}|{r.model}|{r.scope}": round(r.mean_ic, 6) for r in perf.itertuples()},
              'index_baseline': {f'{r.score_index}|{r.scope}': round(r.mean_ic, 6) for r in tables['index_baseline'].itertuples()}}
    if not execute:
        return result
    started = datetime.now()
    folder = root / paths['run_records'] / settings['run_category'] / started.strftime('%Y%m%dT%H%M%S%f')
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(folder / (name + '.csv'), index=False, encoding='utf-8-sig')
    predictions.to_parquet(folder / 'predictions.parquet', index=False)
    (folder / 'report.md').write_text(report_markdown(tables, checks), encoding='utf-8')
    for name, value in inputs.items():
        if sha256(root / name) != value:
            raise RuntimeError('source_changed_during_run:' + name)
    code = ['src/evaluation/linear_reg_test.py', 'src/evaluation/expanding_window.py', 'src/evaluation/rank_ic.py',
            'src/models/interface.py', 'src/models/ols.py', 'src/models/ridge.py', 'src/preprocessing/panel.py',
            'src/features/h1.py', 'config/experiment.yaml']
    record = {'status': 'EXECUTED', 'stage': 'linear_reg_test', 'command': ' '.join(sys.argv), 'started': started.isoformat(),
              'approved_decisions': dict(settings), 'ridge': dict(config['RIDGE']), 'folds': folds,
              'input_sha256': inputs, 'ingest': pre.get('preconditions', {}) if isinstance(pre, dict) else {},
              'code_sha256': {p: sha256(root / p) for p in code}, 'checks': checks,
              'outputs_sha256': {p.name: sha256(p) for p in folder.iterdir()}}
    (folder / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return {**result, 'run_dir': str(folder)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='OLS·Ridge 비교군(확장 창 16 Fold)')
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / args.config).read_text(encoding='utf-8'))
    print(json.dumps(run(root, config, execute=args.execute), ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
