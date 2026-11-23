"""기본 피처 검사: 가용률, 학습 행, 단독 Rank IC, 상관, 배당 계산 방식 검증.

한 명령으로 피처 계산(src/features/base.py)과 검사를 함께 한다. 산출물은
data/processed/base_features.parquet 하나와 outputs/run_id/BASE_ANALYSIS/<시각>/의 report.md·run.json 둘뿐이다.
월별 IC 같은 세부 값은 파일로 남기지 않는다(피처 표에서 다시 계산할 수 있음).

단독 Rank IC: 기준월 t 피처와 달력월 t+1 수익률(교수님 원값)의 Spearman, 적격 종목, 쌍 3개 이상·양쪽 비상수
(rank_ic.rank_ic_for_month와 같은 규칙). 목표월 2001-01~2021-12. t값은 월별 IC 평균 / (표준편차/√T), 자유도 T−1.

실행: python -m src.evaluation.base_analysis             (계산·검사, 쓰지 않음)
      python -m src.evaluation.base_analysis --execute   (피처 표·report.md·run.json 쓰기)
      --reuse : 이미 있는 base_features.parquet로 검사만(피처 계산 검사 값은 빠짐)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.features.base import EXTERNAL, FEATURES, MARKET_FEATURES, STOCK_FEATURES, build_base, guard_development
from src.preprocessing.dataguide_extract import read_sheet, sha256

H_VARS = ('size', 'bm', 'mom', 'sigma')
TARGET_MONTHS = ('2001-01', '2021-12')
PERIODS = {'2001-2010': ('2001', '2010'), '2011-2021': ('2011', '2021')}


def load_frame(root: Path, config: dict, base: pd.DataFrame) -> pd.DataFrame:
    """피처 + H1·H3 + 목표값(t+1 수익률)."""
    paths = config['PATHS']
    h1 = pd.read_parquet(root / paths['h1_components'], columns=['stock_id', 'base_month', 'size', 'bm', 'mom'])
    h3 = pd.read_parquet(root / paths['h3'], columns=['stock_id', 'base_month', 'sigma'])
    panel = pd.read_parquet(root / paths['panel'], columns=['stock_id', 'month', 'return'])
    y = panel.assign(base_month=(pd.PeriodIndex(panel.month, freq='M') - 1).astype(str)).rename(columns={'month': 'target_month', 'return': 'y'})
    d = base.merge(h1, on=['stock_id', 'base_month'], how='left').merge(h3, on=['stock_id', 'base_month'], how='left')
    d = d.merge(y, on=['stock_id', 'base_month'], how='left')
    d['target_month'] = (pd.PeriodIndex(d.base_month, freq='M') + 1).astype(str)
    return d[d.eligible.eq(True)]


def availability(d: pd.DataFrame) -> pd.DataFrame:
    """피처 × 목표월 연도: 적격 종목·월 중 값이 있는 비율."""
    t = d[d.target_month.between(*TARGET_MONTHS)]
    return t.groupby(t.target_month.str[:4])[list(STOCK_FEATURES) + list(H_VARS)].apply(lambda g: g.notna().mean()).round(3)


def training_rows(d: pd.DataFrame) -> pd.DataFrame:
    """D-1 규칙(모든 입력·목표 관측 행만 학습): H1·H3·목표에서 시작해 피처를 하나씩 더할 때 남는 행."""
    t = d[d.target_month.between(*TARGET_MONTHS)]
    mask = t[list(H_VARS) + ['y']].notna().all(axis=1)
    rows = [('H1+H3+목표', int(mask.sum()), 0)]
    for f in STOCK_FEATURES:
        new = mask & t[f].notna()
        rows.append(('+' + f, int(new.sum()), int((mask & ~new).sum())))
        mask = new
    out = pd.DataFrame(rows, columns=['단계', '남는 행', '이번에 빠진 행'])
    out['남은 비율'] = (out['남는 행'] / out['남는 행'].iloc[0]).round(3)
    return out


def monthly_ic(d: pd.DataFrame, col: str) -> pd.Series:
    """목표월별 Spearman(평균 동순위), 쌍 3개 이상·양쪽 비상수."""
    t = d[d.target_month.between(*TARGET_MONTHS) & d[col].notna() & d.y.notna() & np.isfinite(d[col]) & np.isfinite(d.y)][['target_month', col, 'y']]
    g = t.groupby('target_month')
    ok = (g.size() >= 3) & (g[col].nunique() >= 2) & (g.y.nunique() >= 2)
    t = t[t.target_month.isin(ok[ok].index)]
    r = t.assign(a=t.groupby('target_month')[col].rank(method='average'), b=t.groupby('target_month').y.rank(method='average'))
    return r.groupby('target_month')[['a', 'b']].apply(lambda x: x.a.corr(x.b)).rename(col)


def ic_summary(d: pd.DataFrame, signs: dict) -> pd.DataFrame:
    rows = []
    for f in list(STOCK_FEATURES) + list(H_VARS):
        ic = monthly_ic(d, f)
        row = {'피처': f, '기대 부호': signs.get(f, ''), '월 수': len(ic)}
        for name, (a, b) in [('전체', ('2001', '2021'))] + list(PERIODS.items()):
            y = pd.Series(ic.index.str[:4], index=ic.index)
            s = ic[y.between(a, b).to_numpy()]
            tstat = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
            row[f'{name} 평균'] = round(float(s.mean()), 4)
            row[f'{name} t'] = round(float(tstat), 2)
            if name == '전체':
                row['p(양측, 자유도 T−1)'] = round(float(2 * stats.t.sf(abs(tstat), len(s) - 1)), 4)
                row['양수 비율'] = round(float((s > 0).mean()), 3)
        e = signs.get(f, '')
        row['부호 일치'] = '' if e not in ('+', '-') else ('예' if np.sign(row['전체 평균']) == (1 if e == '+' else -1) else '아니요')
        rows.append(row)
    return pd.DataFrame(rows)


def correlations(d: pd.DataFrame) -> pd.DataFrame:
    """월별 Spearman 상관행렬의 평균(피처끼리 + H 변수). 한 피처가 그달 전부 비어 계산이 안 되는 달은 그 칸 평균에서 뺀다."""
    cols = list(STOCK_FEATURES) + list(H_VARS)
    t = d[d.target_month.between(*TARGET_MONTHS)]
    mats = np.stack([g[cols].rank().corr().to_numpy() for _, g in t.groupby('base_month') if len(g) >= 3])
    return pd.DataFrame(np.nanmean(mats, axis=0), index=cols, columns=cols).round(2)


def dividend_check(root: Path, d_all: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """C-4 계산 방식 재현: 12월 dy × 12월 말 시가총액(= 그 해 배당액 추정) vs DataGuide 현금배당액(보통주), 12월 결산, 2011~2020."""
    path = root / EXTERNAL / '2009_2021_개별_feature.xlsx'
    fmp = root / EXTERNAL / '2009_2021_연결.xlsx'
    guard_development(path)
    dg, _ = read_sheet(path, {'3000705002': 'div'})
    fm, _ = read_sheet(fmp, {'CP10002200': 'fm'})
    dg = dg[dg.quarter.eq('결산')].merge(fm[fm.quarter.eq('결산')][['code', 'fy', 'fm']], on=['code', 'fy'], how='left')
    dg = dg[dg.fm.astype(str).str.replace('.0', '', regex=False).eq('12') & dg['div'].notna()]
    dg = dg.assign(stock_id=dg.code.str[1:], base_month=dg.fy.astype(str) + '-12')
    cap = panel.rename(columns={'month': 'base_month'})[['stock_id', 'base_month', 'market_cap']]
    x = d_all[d_all.base_month.str.endswith('-12') & d_all.eligible.eq(True)][['stock_id', 'base_month', 'dy']]
    x = x.merge(cap, on=['stock_id', 'base_month']).merge(dg[['stock_id', 'base_month', 'div']], on=['stock_id', 'base_month'])
    x = x[x.base_month.between('2011-12', '2020-12') & x.dy.notna()]
    x['impl'] = x.dy * x.market_cap
    pos = x[x['div'] > 0]
    ratio = pos.impl / pos['div']
    return {'firm_years': int(len(x)), 'dividend_payers': int(len(pos)),
            'spearman_amount_payers': round(float(pos.impl.corr(pos['div'], method='spearman')), 4),
            'ratio_median': round(float(ratio.median()), 4), 'within_20pct': round(float(((ratio - 1).abs() <= .2).mean()), 4),
            'spearman_yield_all': round(float(x.dy.corr(x['div'] / x.market_cap, method='spearman')), 4)}


def _md(df: pd.DataFrame, index: bool = False) -> str:
    t = df.reset_index() if index else df
    cols = [str(c) for c in t.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join('' if pd.isna(v) else str(v) for v in row) + ' |' for row in t.itertuples(index=False)]
    return '\n'.join(lines)


def run(root: Path, config: dict, *, execute: bool = False, reuse: bool = False, log=print) -> dict:
    root = Path(root)
    target = root / config['PATHS']['base_features']
    if reuse:
        base, build_record = pd.read_parquet(target), {'status': 'REUSED_EXISTING', 'output_sha256': sha256(target)}
    else:
        base, build_record = build_base(root, config)
        if execute:
            target.parent.mkdir(parents=True, exist_ok=True)
            base.to_parquet(target, index=False)
            build_record['status'], build_record['output_sha256'] = 'WRITTEN', sha256(target)
        else:
            build_record['status'] = 'DRY_RUN_NOT_WRITTEN'
    signs = {k: v[1] for k, v in config['BASE_FEATURES']['features'].items()}
    d = load_frame(root, config, base)
    panel = pd.read_parquet(root / config['PATHS']['panel'], columns=['stock_id', 'month', 'market_cap'])
    tables = {'avail': availability(d), 'train': training_rows(d), 'ic': ic_summary(d, signs), 'corr': correlations(d)}
    div = dividend_check(root, base, panel)
    corr = tables['corr']
    high = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr.index) for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) >= 0.7]
    record = {'created': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'build': build_record, 'dividend_check': div,
              'corr_abs_ge_0.7': high, 'training_rows': tables['train'].to_dict('records')}
    report = '\n\n'.join([
        '# 기본 피처 검사', f"피처 표: `{config['PATHS']['base_features']}` ({build_record['status']})",
        '## 단독 Rank IC (목표월 2001-01~2021-12, 적격 종목)', _md(tables['ic']),
        '## 학습 행 (H1·H3·목표 관측에서 피처를 하나씩 더할 때)', _md(tables['train']),
        '## 가용률 (목표월 연도 × 피처, 적격 종목·월 중 값이 있는 비율)', _md(tables['avail'], index=True),
        '## 월별 순위 상관 평균', _md(corr, index=True),
        '## |상관| ≥ 0.7', '\n'.join(f'- {a} – {b}: {v:.2f}' for a, b, v in high) or '- 없음',
        '## 배당수익률 계산 방식 재현 (2011~2020, 12월 결산)', '\n'.join(f'- {k}: {v}' for k, v in div.items()),
        '## 피처 계산 검사 (T2~T4)', '```json\n' + json.dumps(build_record.get('checks', {}), ensure_ascii=False, indent=1) + '\n```'])
    if execute:
        folder = root / config['PATHS']['base_analysis'] / datetime.now().strftime('%Y%m%dT%H%M%S')
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'report.md').write_text(report, encoding='utf-8')
        (folder / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        record['run_dir'] = folder.as_posix()
    log(report if not execute else f"기본 피처 검사 저장: {record['run_dir']}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description='기본 피처 계산 + 검사')
    ap.add_argument('--config', default='config/experiment.yaml')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--reuse', action='store_true')
    a = ap.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / a.config).read_text(encoding='utf-8'))
    run(root, config, execute=a.execute, reuse=a.reuse)


if __name__ == '__main__':
    main()
