"""H2 특징표: 팀원(src/h2/)이 만든 CSV 2개 → data/processed/h2.parquet (종목 × 기준월).

정의는 팀원 구현 그대로다(2026-11-15 결정, 팀원 코드는 수정하지 않고 함수만 가져다 쓴다).
- debt          = proxy_masked(DataGuide 연간 부채비율, 결산+6개월 공개 가정, 12월 결산 기업만)를
                  기준월마다 적격 종목 안에서 1%/99% 윈저 → z(ddof=0). src/h2/features/transforms.monthly_standardize.
- policy        = monthly.csv final_rule_v2_sensitivity(한국은행 통화정책방향 문장 규칙 점수, 월 공통, −1~1).
- policy_x_debt = policy × debt.
표준화 집합 = 우리 패널의 적격 행(기준월 2001-01~2021-11). 팀원 training_frame과 같은 집합이다(적격 행 전체에서
표준화한 뒤 목표값 유무로 행을 고름). 목표값은 이 표에 없다.

model_test에서 debt·policy_x_debt는 우리 월별 순위 변환의 예외다(순위로 바꾸면 정책 세기가 사라짐). config H2 칸.
입력은 data/ 파일과 config뿐이며 이전 실행 기록을 읽지 않는다. 기준월 2021-12 행(목표월 2022-01, 테스트 기간)은 버린다.

실행: python -m src.features.h2            (검사만, 쓰지 않음)
      python -m src.features.h2 --execute  (h2.parquet 쓰기, 있으면 같은 내용인지 비교만)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.h2.features.transforms import monthly_standardize   # 팀원 정의 원문(수정하지 않음)

CODE_ROOT = Path(__file__).resolve().parents[2]
BASE_FIRST, BASE_LAST = '2001-01', '2021-11'     # 정책 점수 2001-01 시작, 목표월 2021-12까지
DEBT_COLUMNS = ['stock_id', 'feature_month', 'target_month', 'proxy_masked', 'fiscal_date', 'proxy_available_date',
                'fiscal_end_leak', 'proxy_date_leak', 'lag_months']
POLICY_COLUMNS = ['month', 'final_rule_v2_sensitivity', 'sensitivity_source_document_id', 'sensitivity_source_release_date']
OUTPUT_COLUMNS = ['stock_id', 'base_month', 'eligible', 'debt_raw', 'debt', 'policy', 'policy_x_debt']


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _month_end(months: pd.Series) -> pd.Series:
    return pd.Series(pd.PeriodIndex(months, freq='M').to_timestamp('M'), index=months.index)


def read_debt(path) -> pd.DataFrame:
    """부채 CSV 읽기·검사: 열, (종목, 기준월) 중복, 6개월 지연, 공개일 ≤ 기준월 말, 12월 결산, 누수 표시 없음."""
    d = pd.read_csv(path, dtype={'stock_id': 'string', 'feature_month': 'string', 'target_month': 'string'},
                    encoding='utf-8-sig')
    missing = set(DEBT_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError('h2_debt_columns_missing:' + ','.join(sorted(missing)))
    d = d[DEBT_COLUMNS].copy()
    d['stock_id'] = d.stock_id.str.zfill(6)
    if not d.stock_id.str.fullmatch(r'\d{6}').all() or not d.feature_month.str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('h2_debt_key_format')
    if d.duplicated(['stock_id', 'feature_month']).any():
        raise ValueError('h2_debt_duplicate_keys')
    if not (pd.PeriodIndex(d.feature_month, freq='M') + 1).astype(str).equals(pd.Index(d.target_month.astype(str))):
        raise ValueError('h2_debt_target_month_not_next_month')
    if not d.lag_months.eq(6).all():
        raise ValueError('h2_debt_lag_not_6_months')
    if d.fiscal_end_leak.astype(bool).any() or d.proxy_date_leak.astype(bool).any():
        raise ValueError('h2_debt_leak_flag_set')
    d['proxy_masked'] = pd.to_numeric(d.proxy_masked, errors='raise').astype(float)
    if np.isinf(d.proxy_masked).any():
        raise ValueError('h2_debt_infinite_values')
    x = d[d.proxy_masked.notna()]
    avail = pd.to_datetime(x.proxy_available_date, errors='coerce')
    fiscal = pd.to_datetime(x.fiscal_date, errors='coerce')
    if avail.isna().any() or fiscal.isna().any():
        raise ValueError('h2_debt_dates_missing')
    if (avail > _month_end(x.feature_month)).any():
        raise ValueError('h2_debt_available_after_base_month_end')
    if not ((fiscal.dt.month == 12) & (fiscal.dt.day == 31)).all():
        raise ValueError('h2_debt_not_december_fiscal_year')
    return d


def read_policy(path) -> pd.DataFrame:
    """정책 CSV 읽기·검사: 월당 한 줄, 점수 −1~1, 발표일 < 기준월 말일(말일 발표는 다음 달부터, 팀원 규칙)."""
    m = pd.read_csv(path, dtype={'month': 'string'}, encoding='utf-8-sig')
    missing = set(POLICY_COLUMNS) - set(m.columns)
    if missing:
        raise ValueError('h2_policy_columns_missing:' + ','.join(sorted(missing)))
    m = m[POLICY_COLUMNS].copy()
    if m.month.duplicated().any():
        raise ValueError('h2_policy_more_than_one_score_per_month')
    m['final_rule_v2_sensitivity'] = pd.to_numeric(m.final_rule_v2_sensitivity, errors='raise').astype(float)
    s = m.final_rule_v2_sensitivity
    if s.isna().any() or not s.between(-1, 1).all():
        raise ValueError('h2_policy_score_missing_or_out_of_range')
    release = pd.to_datetime(m.sensitivity_source_release_date, errors='coerce')
    if release.isna().any() or (release >= _month_end(m.month)).any():
        raise ValueError('h2_policy_release_not_before_month_end')
    return m


def build_h2(root: Path, config) -> tuple[pd.DataFrame, dict]:
    """CSV 2개 + 우리 패널(적격) → H2 특징표(메모리)와 검사 기록."""
    paths = config['PATHS']
    debt_path, policy_path, panel_path = (root / paths[k] for k in ('h2_debt_csv', 'h2_monthly_csv', 'panel'))
    debt, policy = read_debt(debt_path), read_policy(policy_path)
    panel = pd.read_parquet(panel_path, columns=['stock_id', 'month', 'eligible'])
    if not panel.month.between('2000-01', '2021-12').all():
        raise PermissionError('development_months_only')
    p = panel[panel.month.between(BASE_FIRST, BASE_LAST)].rename(columns={'month': 'base_month'})
    p = p.assign(stock_id=p.stock_id.astype(str), eligible=p.eligible.eq(True).fillna(False).astype(bool))
    elig = p[p.eligible].copy()
    dropped_test = int(debt.feature_month.gt(BASE_LAST).sum())         # 기준월 2021-12(목표월 2022-01) 행은 버림
    debt = debt[debt.feature_month.between(BASE_FIRST, BASE_LAST)]
    # 팀원 training_frame과 같은 순서: 적격 행에 왼쪽 결합 → feature_month별 monthly_standardize → policy × debt
    f = elig.merge(debt[['stock_id', 'feature_month', 'proxy_masked']].rename(columns={'feature_month': 'base_month'}),
                   on=['stock_id', 'base_month'], how='left', validate='one_to_one')
    f['feature_month'] = f.base_month
    f['debt'] = monthly_standardize(f, 'proxy_masked')
    f = f.merge(policy[['month', 'final_rule_v2_sensitivity']].rename(columns={'month': 'base_month'}),
                on='base_month', how='left', validate='many_to_one')
    f['policy'] = f.final_rule_v2_sensitivity
    f['policy_x_debt'] = f.policy * f.debt
    out = (f.rename(columns={'proxy_masked': 'debt_raw'})[OUTPUT_COLUMNS]
           .sort_values(['stock_id', 'base_month']).reset_index(drop=True))
    if out.policy.isna().any():
        raise ValueError('h2_policy_missing_for_base_month')
    if out.debt_raw.notna().ne(out.debt.notna()).any():
        raise ValueError('h2_debt_standardization_dropped_rows')
    # 붙는 비율: 적격 행 중 부채가 있는 비율, 부채 CSV 값 중 적격 행에 붙은 비율
    have = debt[debt.proxy_masked.notna()]
    keys = set(zip(elig.stock_id, elig.base_month))
    in_panel = np.fromiter(((s, m) in keys for s, m in zip(have.stock_id, have.feature_month)), bool, len(have))
    by_year = out.assign(year=out.base_month.str[:4]).groupby('year').debt.apply(lambda s: round(float(s.notna().mean()), 4))
    checks = {
        'rows': int(len(out)), 'base_months': [out.base_month.min(), out.base_month.max()],
        'debt_csv_rows': int(len(debt) + dropped_test), 'debt_csv_rows_dropped_base_2021_12': dropped_test,
        'eligible_rows_with_debt': int(out.debt.notna().sum()),
        'eligible_rows_with_debt_share': float(out.debt.notna().mean()),
        'debt_values_matched_to_eligible_rows': int(in_panel.sum()), 'debt_values_total': int(len(have)),
        'debt_values_matched_share': float(in_panel.mean()) if len(have) else np.nan,
        'debt_share_by_base_year': by_year.to_dict(),
        'policy_months': int(out.base_month.nunique()), 'policy_one_score_per_month': True,
        'policy_value_counts': {str(k): int(v) for k, v in policy.final_rule_v2_sensitivity.value_counts().sort_index().items()},
        'debt_monthly_mean_abs_max': float(out.groupby('base_month').debt.mean().abs().max()),
        'debt_monthly_std_ddof0_range': [float(x) for x in out.groupby('base_month').debt.std(ddof=0).agg(['min', 'max'])],
    }
    record = {'inputs_sha256': {paths[k]: sha256(root / paths[k]) for k in ('h2_debt_csv', 'h2_monthly_csv', 'panel')},
              'definition': 'teammate src/h2: debt=monthly_standardize(proxy_masked) (1/99 winsor, z ddof=0, eligible rows), '
                            'policy=final_rule_v2_sensitivity, policy_x_debt=policy*debt',
              'code_sha256': {p: sha256(CODE_ROOT / p) for p in ('src/features/h2.py', 'src/h2/features/transforms.py')},
              'checks': checks}
    return out, record


def same_table(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """키·결측 위치는 정확히, 값은 정확히 같은지(같은 코드·같은 입력이면 같은 값)."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.kind == 'f' or y.dtype.kind == 'f':
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            if not np.array_equal(np.isnan(xv), np.isnan(yv)) or not np.array_equal(xv[~np.isnan(xv)], yv[~np.isnan(yv)]):
                return False
        elif not x.astype(str).equals(y.astype(str)):
            return False
    return True


def ensure_h2(root: Path, config, *, execute: bool = False, log=print) -> dict:
    """h2.parquet를 만든다(없을 때, execute=True) 또는 있는 파일이 다시 계산한 값과 같은지 확인한다(다르면 멈춤)."""
    root = Path(root)
    out, record = build_h2(root, config)
    target = root / config['PATHS']['h2']
    if target.exists():
        if not same_table(pd.read_parquet(target), out):
            raise ValueError(f'h2_parquet_differs_from_recomputed:{target}')
        record['status'] = 'EXISTS_SAME'
    elif execute:
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)
        record['status'] = 'WRITTEN'
    else:
        record['status'] = 'DRY_RUN_NOT_WRITTEN'
    record['output'] = config['PATHS']['h2']
    if target.exists():
        record['output_sha256'] = sha256(target)
    c = record['checks']
    log(f"H2 {record['status']}: {c['rows']:,}행(적격, 기준월 {c['base_months'][0]}~{c['base_months'][1]}), "
        f"부채 있는 행 {c['eligible_rows_with_debt']:,} ({c['eligible_rows_with_debt_share']:.1%}), "
        f"부채 값 중 적격 행에 붙은 비율 {c['debt_values_matched_share']:.1%}, 정책 {c['policy_months']}개월")
    return record


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description='H2 특징표(data/processed/h2.parquet)')
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / args.config).read_text(encoding='utf-8'))
    record = ensure_h2(root, config, execute=args.execute)
    print(json.dumps(record, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
