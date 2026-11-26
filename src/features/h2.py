"""H2 특징표: Hybrid(DART 공시일 우선 + DataGuide 6개월 지연 보완) + 정책 CSV + 공통 패널 → h2.parquet.

정의는 팀원 구현 그대로다(2026-11-15 결정, 팀원 코드는 수정하지 않고 함수만 가져다 쓴다).
- debt          = 선택한 원천의 부채비율(기본: DART Hybrid, 선택: DataGuide 6개월 지연)을
                  기준월마다 적격 종목 안에서 1%/99% 윈저 → z(ddof=0). src/h2/features/transforms.monthly_standardize.
- policy        = monthly.csv final_rule_v2_sensitivity(한국은행 통화정책방향 문장 규칙 점수, 월 공통, −1~1).
- policy_x_debt = policy × debt.
- policy_change = policy(t) − policy(t−1), 직전 달력월 기준(첫 달 결측).
H2 공식 특징표는 8개 피처를 생성한다. 기존 4개 열의 정의는 유지한다.
- debt_change = 직전 다른 결산기 보고서 대비 부채비율 변화(원값).
- rate_change = 기준월 말 정책금리 − 직전월 말 정책금리(%p), 첫 파일 월 결측.
- policy_change_x_debt = policy_change × debt.
- rate_change_x_debt = rate_change × debt.
표준화 집합 = 우리 패널의 적격 행(기준월 2001-01~2021-11). 팀원 training_frame과 같은 집합이다(적격 행 전체에서
표준화한 뒤 목표값 유무로 행을 고름). 목표값은 이 표에 없다.

model_test에서 debt·policy_x_debt는 우리 월별 순위 변환의 예외다(순위로 바꾸면 정책 세기가 사라짐). config H2 칸.
입력은 data/ 파일과 config뿐이며 이전 실행 기록을 읽지 않는다. 기준월 2021-12 행(목표월 2022-01, 테스트 기간)은 버린다.

실행: python -m src.features.h2            (검사만, 쓰지 않음)
      python -m src.features.h2 --execute  (Hybrid 생성, 기존 결과가 다르면 보호 오류)
      python -m src.features.h2 --execute --replace  (기존 결과 백업 후 Hybrid 갱신)
      python -m src.features.h2 --debt-source lag_6m  (이전 6개월 지연 방식 검사)

주의: Hybrid는 완전히 검증된 historical PIT 자료가 아닌 후보 데이터다.
출력은 관리용 4개 열과 모델 피처 8개 열이다. 원천이 다르면 교체를 중단한다.
기존 4피처 파일 및 보존된 8피처 참조의 값·결측 위치를 검증한다.
lag_6m 모드는 기존 4개 피처 점검만 허용하며 공식 출력은 Hybrid 8개 피처로 유지한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.h2.features.transforms import monthly_standardize   # 팀원 정의 원문(수정하지 않음)

CODE_ROOT = Path(__file__).resolve().parents[2]
BASE_FIRST, BASE_LAST = '2001-01', '2021-11'     # 정책 점수 2001-01 시작, 목표월 2021-12까지
DEBT_COLUMNS = ['stock_id', 'feature_month', 'target_month', 'proxy_masked', 'fiscal_date', 'proxy_available_date',
                'fiscal_end_leak', 'proxy_date_leak', 'lag_months']
POLICY_COLUMNS = ['month', 'final_rule_v2_sensitivity', 'sensitivity_source_document_id', 'sensitivity_source_release_date']
# H2_EIGHT_FEATURE_UPGRADE_V1
FOUR_FEATURES = ['debt', 'policy', 'policy_x_debt', 'policy_change']
EXTRA_FEATURES = ['debt_change', 'rate_change', 'policy_change_x_debt', 'rate_change_x_debt']
MODEL_FEATURES = FOUR_FEATURES + EXTRA_FEATURES
FOUR_OUTPUT_COLUMNS = ['stock_id', 'base_month', 'eligible', 'debt_raw', *FOUR_FEATURES]
OUTPUT_COLUMNS = FOUR_OUTPUT_COLUMNS + EXTRA_FEATURES
REFERENCE_ARCHIVE = Path('data/processed/archive/h2_8features.parquet')
REFERENCE_SHA256 = '8a504b7d02f5e961fe07302cb33ae30db17e67fefd92273311cc08019a598df1'
DEFAULT_HYBRID_CSV = Path('data/processed/final/debt_hybrid/h2_debt_hybrid_candidate_2001_2021.csv')
HYBRID_COLUMNS = ('stock_id', 'feature_month', 'debt_final_raw', 'source_type',
                  'source_available_date', 'source_fiscal_date')
HYBRID_DART = 'DART_FIRST_FILED_CANDIDATE'
HYBRID_FALLBACK = 'DATAGUIDE_6M_PROXY'


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



def read_hybrid_debt(path) -> pd.DataFrame:
    """Hybrid 결과를 검증하고 기존 H2 부채 입력과 동일한 열 구조로 변환한다.

    DART 후보/6개월 지연 대체값 모두 공개일·결산일이 예측 기준월 말보다 늦으면 거부한다.
    재무수치 자체의 역사적 vintage(PIT) 검증을 보장하지는 않는다.
    """
    d = pd.read_csv(path, dtype={'stock_id': 'string', 'feature_month': 'string'},
                    encoding='utf-8-sig', low_memory=False)
    missing = set(HYBRID_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError('h2_hybrid_columns_missing:' + ','.join(sorted(missing)))
    d['stock_id'] = (d.stock_id.str.strip()
                     .str.replace(r'^A(?=\d{6}$)', '', regex=True)
                     .str.replace(r'\.0$', '', regex=True))
    if not d.stock_id.str.fullmatch(r'\d{1,6}', na=False).all():
        raise ValueError('h2_hybrid_stock_id_invalid')
    d['stock_id'] = d.stock_id.str.zfill(6)
    if not d.feature_month.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', na=False).all():
        raise ValueError('h2_hybrid_feature_month_invalid')
    if d.duplicated(['stock_id', 'feature_month']).any():
        raise ValueError('h2_hybrid_duplicate_keys')
    if 'target_month' in d.columns:
        expected = (pd.PeriodIndex(d.feature_month, freq='M') + 1).astype(str)
        if not expected.equals(pd.Index(d.target_month.astype(str))):
            raise ValueError('h2_hybrid_target_not_next_month')

    raw = pd.to_numeric(d.debt_final_raw, errors='raise').astype(float)
    if np.isinf(raw.to_numpy()).any():
        raise ValueError('h2_hybrid_infinite_values')
    active = raw.notna()
    source = d.source_type.astype('string')
    if (active & ~source.isin([HYBRID_DART, HYBRID_FALLBACK]).fillna(False)).any():
        raise ValueError('h2_hybrid_unknown_source_type')

    end = _month_end(d.feature_month)
    if 'prediction_date' in d.columns:
        prediction = pd.to_datetime(d.prediction_date, errors='coerce')
        if prediction.isna().any() or not prediction.eq(end).all():
            raise ValueError('h2_hybrid_prediction_date_not_month_end')
    for col in ('source_available_date', 'source_fiscal_date'):
        when = pd.to_datetime(d[col], errors='coerce')
        if when[active].isna().any() or when[active].gt(end[active]).any():
            raise ValueError('h2_hybrid_future_or_missing_' + col)
        if col == 'source_fiscal_date' and not (
            (when[active].dt.month == 12) & (when[active].dt.day == 31)
        ).all():
            raise ValueError('h2_hybrid_non_december_fiscal_year')

    # 후속 공시 플래그는 기업·월에 후보가 존재했다는 감사 정보이며,
    # 실제 선택된 DART 후보가 사용 시점에 유효하지 않다는 뜻은 아니다.
    dart_selected = active & source.eq(HYBRID_DART).fillna(False)
    if 'dart_next_filing_date' not in d.columns:
        raise ValueError('h2_hybrid_missing_dart_next_filing_date')
    next_raw = d['dart_next_filing_date'].astype('string').str.strip()
    next_date = pd.to_datetime(next_raw, errors='coerce')
    invalid_next_date = (dart_selected & next_raw.notna()
                         & next_raw.ne('') & next_date.isna())
    if invalid_next_date.any():
        raise ValueError('h2_hybrid_invalid_dart_next_filing_date')
    cutoff = pd.to_datetime(d['prediction_date'], errors='coerce') if 'prediction_date' in d.columns else end
    invalid_selected = dart_selected & next_date.notna() & next_date.le(cutoff)
    if invalid_selected.any():
        raise ValueError(f'h2_hybrid_selected_dart_superseded:{int(invalid_selected.sum())}')

    if 'has_unresolved_followup' in d.columns:
        unresolved = (d['has_unresolved_followup'].astype('string')
                      .str.lower().isin(['true', '1', 'yes']))
        flagged = dart_selected & unresolved
        if flagged.any():
            print(f'[H2 Hybrid] 미해결 후속공시 플래그 DART 사용 행: '
                  f'{int(flagged.sum()):,}건 (선택 공시의 후속 접수일 조건은 통과)')

    # downstream 월별 표준화에 입력하는 이름을 기존과 동일하게 유지
    return pd.DataFrame({
        'stock_id': d.stock_id,
        'feature_month': d.feature_month,
        'proxy_masked': raw,
    })

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


def monthly_changes(policy: pd.DataFrame) -> pd.DataFrame:
    """월별 정책 점수 변화량: 직전 달력월과 비교하며 첫 달은 결측으로 유지."""
    months = pd.PeriodIndex(policy.month, freq='M')
    full = pd.period_range(months.min(), months.max(), freq='M')
    score = pd.Series(policy.final_rule_v2_sensitivity.to_numpy(float), index=months).reindex(full)
    return pd.DataFrame({'base_month': full.astype(str),
                         'policy_change': (score - score.shift(1)).to_numpy()})


def extend_h2_eight(frame, debt_path):
    """Add four raw model features on the same base-month eligible universe.

    DebtChange is rebuilt from the exact Hybrid source used for debt. No targets,
    future months, archived feature values, ranking, or zero imputation are used.
    The first rate CSV month remains missing, as in the historical eight-feature run.
    """
    from src.h2.experiments.build_h2_debt_change import read_hybrid, compute_debt_change
    from src.h2.preprocessing.build_bok_rate_history import make

    source = read_hybrid(debt_path)
    # Never allow rows after the historical development input horizon to enter computation.
    source = source.loc[source.feature_month.le('2021-12')].copy()
    changes = compute_debt_change(source)
    changes = changes[['stock_id', 'feature_month', 'debt_final_raw', 'debt_change_raw']].rename(
        columns={'feature_month': 'base_month', 'debt_change_raw': 'debt_change'})
    out = frame.merge(changes, on=['stock_id', 'base_month'], how='left', validate='one_to_one')
    raw = out.proxy_masked.to_numpy(float)
    other = out.debt_final_raw.to_numpy(float)
    if not np.array_equal(np.isnan(raw), np.isnan(other)) or not np.allclose(raw, other, rtol=1e-12, atol=1e-10, equal_nan=True):
        raise ValueError('h2_debt_change_source_mismatch')
    out = out.drop(columns='debt_final_raw')
    _, rates = make()
    rates = rates[['feature_month', 'rate_change_pp']].sort_values('feature_month').copy()
    if rates.feature_month.duplicated().any() or rates.empty:
        raise ValueError('h2_invalid_rate_months')
    # Match run_h2_ratechange_ablation.add_rate_change: first file month forced NaN.
    rates.loc[rates.feature_month.eq(rates.feature_month.iloc[0]), 'rate_change_pp'] = np.nan
    rates = rates.rename(columns={'feature_month': 'base_month', 'rate_change_pp': 'rate_change'})
    if not out.base_month.isin(rates.base_month).all():
        raise ValueError('h2_rate_month_missing')
    out = out.merge(rates, on='base_month', how='left', validate='many_to_one')
    out['policy_change_x_debt'] = out.policy_change * out.debt
    out['rate_change_x_debt'] = out.rate_change * out.debt
    if np.isinf(out[MODEL_FEATURES].to_numpy(float)).any():
        raise ValueError('h2_infinite_feature')
    return out


def compare_feature_tables(candidate, reference, columns, *, atol=0.0, rtol=0.0):
    """Compare by keys, not storage row order. Report missingness and numeric drift."""
    key = ['stock_id', 'base_month']
    cols = key + [c for c in columns if c not in key]
    left, right = candidate[cols].copy(), reference[cols].copy()
    for table in (left, right):
        if table[key].isna().any().any():
            raise ValueError('h2_comparison_missing_keys')
        for c in key:
            table[c] = table[c].astype(str)
        if table.duplicated(key).any():
            raise ValueError('h2_comparison_duplicate_keys')
    left, right = (t.set_index(key).sort_index() for t in (left, right))
    same_keys = left.index.equals(right.index)
    result = {'keys_identical': bool(same_keys), 'candidate_rows': len(left),
              'reference_rows': len(right), 'candidate_only_keys': len(left.index.difference(right.index)),
              'reference_only_keys': len(right.index.difference(left.index)),
              'atol': atol, 'rtol': rtol, 'columns': {}}
    if same_keys:
        for c in cols[len(key):]:
            x, y = left[c], right[c]
            if c == 'eligible':
                bad = int((x.astype('string').fillna('<NA>') != y.astype('string').fillna('<NA>')).sum())
                result['columns'][c] = {'mismatches': bad}
                continue
            a, b = x.to_numpy(float), y.to_numpy(float)
            ma, mb = np.isnan(a), np.isnan(b)
            valid = ~ma & ~mb
            bad = int((~np.isclose(a[valid], b[valid], atol=atol, rtol=rtol)).sum())
            result['columns'][c] = {'missing_mismatches': int((ma ^ mb).sum()),
                                    'value_mismatches': bad,
                                    'max_abs_difference': float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else None}
    result['match'] = bool(same_keys and all(
        not any(v.get(k, 0) for k in ('mismatches', 'missing_mismatches', 'value_mismatches'))
        for v in result['columns'].values()))
    return result


def check_eight_archive(root, out):
    """The frozen development archive is a reference only, never a build input."""
    path = root / REFERENCE_ARCHIVE
    if not path.is_file():
        raise FileNotFoundError(f'H2 historical reference missing: {path}')
    actual = sha256(path)
    if actual != REFERENCE_SHA256:
        raise ValueError(f'H2 historical reference SHA mismatch: {actual}')
    reference = pd.read_parquet(path)
    report = compare_feature_tables(out, reference, MODEL_FEATURES, atol=1e-10, rtol=1e-12)
    report['reference_sha256'] = actual
    report['reference_path'] = str(REFERENCE_ARCHIVE)
    return report


def build_h2(root: Path, config, *, debt_source: str = 'hybrid',
             hybrid_csv: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """공통 패널 적격 표본에서 H2 특징표를 계산한다. Hybrid가 기본값."""
    root = Path(root)
    paths = config['PATHS']
    policy_path, panel_path = (root / paths[k] for k in ('h2_monthly_csv', 'panel'))
    if debt_source == 'lag_6m':
        debt_path = root / paths['h2_debt_csv']
        debt = read_debt(debt_path)
    elif debt_source == 'hybrid':
        configured = hybrid_csv if hybrid_csv is not None else paths.get('h2_hybrid_csv', DEFAULT_HYBRID_CSV)
        debt_path = Path(configured)
        if not debt_path.is_absolute():
            debt_path = root / debt_path
        if not debt_path.is_file():
            raise FileNotFoundError(f'H2 Hybrid 입력 CSV 없음: {debt_path}')
        debt = read_hybrid_debt(debt_path)
    else:
        raise ValueError('invalid_h2_debt_source:' + str(debt_source))
    policy = read_policy(policy_path)
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
    f = f.merge(monthly_changes(policy), on='base_month', how='left', validate='many_to_one')
    if debt_source == 'hybrid':
        f = extend_h2_eight(f, debt_path)
    output_columns = OUTPUT_COLUMNS if debt_source == 'hybrid' else FOUR_OUTPUT_COLUMNS
    model_features = MODEL_FEATURES if debt_source == 'hybrid' else FOUR_FEATURES
    out = (f.rename(columns={'proxy_masked': 'debt_raw'})[output_columns]
           .sort_values(['stock_id', 'base_month']).reset_index(drop=True))
    if out.duplicated(['stock_id', 'base_month']).any():
        raise ValueError('h2_output_duplicate_keys')
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
        'policy_change_nonzero_months': int(out.groupby('base_month').policy_change.first().fillna(0).ne(0).sum()),
        'first_month_change_missing': out.base_month.min(),
        'output_columns': list(output_columns),
        'model_features': list(model_features),
        'missing_counts': {c: int(n) for c, n in out.isna().sum().items()},
        'duplicate_keys': int(out.duplicated(['stock_id', 'base_month']).sum()),
    }
    if debt_source == 'lag_6m':
        # 기존 기록 형식도 유지하여 과거 6개월 지연 실험과 비교 가능하게 한다.
        record = {'inputs_sha256': {paths[k]: sha256(root / paths[k]) for k in ('h2_debt_csv', 'h2_monthly_csv', 'panel')},
                  'definition': 'teammate src/h2: debt=monthly_standardize(proxy_masked) (1/99 winsor, z ddof=0, eligible rows), '
                                'policy=final_rule_v2_sensitivity, policy_x_debt=policy*debt, '
                                'policy_change=policy(t)-policy(t-1)',
                  'code_sha256': {p: sha256(CODE_ROOT / p) for p in ('src/features/h2.py', 'src/h2/features/transforms.py')},
                  'checks': checks}
    else:
        try:
            displayed_path = str(debt_path.relative_to(root))
        except ValueError:
            displayed_path = str(debt_path)
        record = {'inputs_sha256': {displayed_path: sha256(debt_path),
                                    paths['h2_monthly_csv']: sha256(policy_path),
                                    paths['panel']: sha256(panel_path)},
                  'debt_source': 'hybrid',
                  'full_historical_pit_verified': False,
                  'definition': 'debt=monthly_standardize(debt_final_raw) (1/99 winsor, z ddof=0, eligible rows), '
                                'policy=final_rule_v2_sensitivity, policy_x_debt=policy*debt, '
                                'policy_change=policy(t)-policy(t-1)',
                  'code_sha256': {p: sha256(CODE_ROOT / p) for p in ('src/features/h2.py', 'src/h2/features/transforms.py')},
                  'checks': checks}
    if debt_source == 'hybrid':
        record['definition'] += '; debt_change=distinct available fiscal report delta (raw); rate_change=base-month rate delta pp (first month missing); policy_change_x_debt=policy_change*debt; rate_change_x_debt=rate_change*debt'
        record['code_sha256'].update({p: sha256(CODE_ROOT / p) for p in ('src/h2/experiments/build_h2_debt_change.py', 'src/h2/preprocessing/build_bok_rate_history.py')})
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


def ensure_h2(root: Path, config, *, execute: bool = False, log=print,
              debt_source: str = 'hybrid', hybrid_csv: Path | str | None = None,
              replace: bool = False) -> dict:
    """출력은 언제나 PATHS.h2. 차이가 나면 자동 덮어쓰기 금지.

    기존 lag_6m은 원래의 결과 동일성 검사 정책을 유지한다.
    Hybrid는 --execute --replace 조합일 때만 다른 기존 parquet을 백업하고 교체한다.
    """
    root = Path(root)
    if execute and debt_source != 'hybrid':
        raise ValueError('legacy_lag_6m_is_read_only; official h2.parquet requires hybrid eight features')
    if replace and (debt_source != 'hybrid' or not execute):
        raise ValueError('replace_requires_hybrid_and_execute')
    out, record = build_h2(root, config, debt_source=debt_source, hybrid_csv=hybrid_csv)
    target = root / config['PATHS']['h2']
    if debt_source == 'hybrid':
        report = check_eight_archive(root, out)
        record['archive_8feature_comparison'] = report
        if execute and not report['match']:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            raise ValueError('h2_eight_archive_mismatch; replacement blocked')
    if target.exists():
        existing = pd.read_parquet(target)
        if debt_source == 'hybrid':
            if set(existing.columns) not in (set(FOUR_OUTPUT_COLUMNS), set(OUTPUT_COLUMNS)):
                raise ValueError('h2_existing_unexpected_schema; replacement blocked')
            retained = compare_feature_tables(out, existing, FOUR_OUTPUT_COLUMNS)
            record['retained_4feature_comparison'] = retained
            record['old_sha256'] = sha256(target)
            if execute and not retained['match']:
                print(json.dumps(retained, ensure_ascii=False, indent=2))
                raise ValueError('h2_retained_four_features_differ; replacement blocked')
        match = same_table(existing, out)
        if match:
            record['status'] = 'EXISTS_SAME'
        elif not execute:
            record['status'] = 'DIFFERS_DRY_RUN'
        elif debt_source == 'hybrid' and replace:
            record['status'] = 'REPLACED_AFTER_BACKUP'
        else:
            raise ValueError(f'h2_parquet_differs_from_recomputed:{target} (입력 원천이 다른 경우 자동 덮어쓰기 금지)')
    elif execute:
        record['status'] = 'WRITTEN'
    else:
        record['status'] = 'DRY_RUN_NOT_WRITTEN'

    if execute and record['status'] in ('WRITTEN', 'REPLACED_AFTER_BACKUP'):
        target.parent.mkdir(parents=True, exist_ok=True)
        # 이미 존재하는 h2.parquet을 수정할 때는 기존 버전을 백업한다.
        if target.exists():
            backup_dir = target.parent / '_h2_backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup = backup_dir / f'h2_{timestamp}.parquet'
            shutil.copy2(target, backup)
            if sha256(backup) != sha256(target):
                raise ValueError('h2_backup_sha256_mismatch')
            record['backup'] = str(backup)
            old_manifest = target.with_suffix('.manifest.json')
            if old_manifest.exists():
                backup_manifest = backup.with_suffix('.manifest.json')
                shutil.copy2(old_manifest, backup_manifest)
                record['backup_manifest'] = str(backup_manifest)

        # 출력 검증을 완료한 뒤 원자적으로 교체해 중간 실패 시 파일 손상 방지.
        with tempfile.NamedTemporaryFile(suffix='.parquet', dir=target.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            out.to_parquet(temporary, index=False)
            if not same_table(pd.read_parquet(temporary), out):
                raise ValueError('h2_parquet_roundtrip_mismatch')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        if debt_source == 'hybrid':
            # 생성 원천이 무엇인지 알 수 있도록 같은 위치에 실행 기록 남김.
            manifest = target.with_suffix('.manifest.json')
            record['manifest'] = str(manifest)
            record['output_sha256'] = sha256(target)
            manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    record['output'] = config['PATHS']['h2']
    if target.exists():
        record['output_sha256'] = sha256(target)
    c = record['checks']
    log(f"H2 {record['status']} ({debt_source}): {c['rows']:,}행(적격, 기준월 {c['base_months'][0]}~{c['base_months'][1]}), "
        f"부채 있는 행 {c['eligible_rows_with_debt']:,} ({c['eligible_rows_with_debt_share']:.1%}), "
        f"부채 값 중 적격 행에 붙은 비율 {c['debt_values_matched_share']:.1%}, 정책 {c['policy_months']}개월")
    return record


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description='H2 특징표(data/processed/h2.parquet)')
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--debt-source', choices=['lag_6m', 'hybrid'], default='hybrid')
    parser.add_argument('--hybrid-csv', type=Path, default=None)
    parser.add_argument('--replace', action='store_true', help='Hybrid 실행 시 다른 기존 h2.parquet 백업 후 교체')
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / args.config).read_text(encoding='utf-8'))
    record = ensure_h2(root, config, execute=args.execute, debt_source=args.debt_source,
                       hybrid_csv=args.hybrid_csv, replace=args.replace)
    print(json.dumps(record, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
