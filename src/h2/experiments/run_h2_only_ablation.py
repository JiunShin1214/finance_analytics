# -*- coding: utf-8 -*-
"""H2 전용 변수 조합(Ablation): 기존 6종 + DebtChange 확장 3종 (공통 표본).

프로젝트 루트에서 실행:
    python -m src.h2.experiments.run_h2_only_ablation --check-only
    python -m src.h2.experiments.run_h2_only_ablation --years 2006-2007 --models Ridge,RandomForest
    python -m src.h2.experiments.run_h2_only_ablation

DebtChange: build_h2_debt_change.py로 별도 생성한 Hybrid PIT-aware 결과를 사용.
원래 H2-0~H2-5 결과와 직접 수치 비교하지 말 것 (평가 표본 달라짐).

기존 src.h2.experiments.run_h2_paper_models.prepare/_build_model/_ic를 재사용.
H1/H3 특징은 prepare()가 읽더라도 어떤 실험 조합에도 투입하지 않음.
2022~2025 최종 테스트 자료는 읽지 않음. 원본 파일을 변경하지 않음.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .run_h2_paper_models import prepare, _build_model, _ic

ALLOWED_MODELS = ('Ridge', 'RandomForest', 'XGBoost', 'LightGBM')
DEFAULT_DEBT_CHANGE = Path('data/processed/final/debt_hybrid/h2_debt_change_2001_2021.csv')
KEY = ['stock_id', 'feature_month']


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def add_debt_change(data: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict]:
    """종목-기준월 1:1 결합; 원천의 공시 시점·결산일 검증.

    기존 DebtChange 생성기는 보고서 변경시 변화량을 갱신하고 다음 보고서
    공시 전까지 그 값을 유지한다. 새로 계산하거나 0으로 채우지 않는다.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f'DebtChange 파일 없음: {path}\n'
            '먼저 python -m src.h2.experiments.build_h2_debt_change --execute 실행'
        )
    t = pd.read_csv(path, dtype={'stock_id': 'string', 'feature_month': 'string'}, low_memory=False)
    need = set(KEY + ['debt_change_raw', 'debt_final_raw', 'debt_change_status',
                      'source_fiscal_date', 'source_available_date'])
    if not need.issubset(t):
        raise ValueError(f'DebtChange 필수 컬럼 누락: {sorted(need - set(t.columns))}')
    t['stock_id'] = (t['stock_id'].str.strip().str.replace(r'^A(?=\d{6}$)', '', regex=True)
                     .str.replace(r'\.0$', '', regex=True).str.zfill(6))
    t['feature_month'] = t['feature_month'].astype('string').str.slice(0, 7)
    if t[KEY].isna().any().any() or t.duplicated(KEY).any():
        raise ValueError('DebtChange 종목-기준월 키에 결측/중복이 있습니다')
    if not t['feature_month'].str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', na=False).all():
        raise ValueError('DebtChange 기준월 형식 오류')
    t['debt_final_raw'] = pd.to_numeric(t['debt_final_raw'], errors='coerce')
    t['debt_change_raw'] = pd.to_numeric(t['debt_change_raw'], errors='coerce')
    t['debt_change_raw'] = t['debt_change_raw'].replace([np.inf, -np.inf], np.nan)
    if t['debt_change_status'].isna().any():
        raise ValueError('DebtChange 상태값 결측')
    cutoff = pd.PeriodIndex(t['feature_month'].astype(str), freq='M').to_timestamp('M')
    active = t['debt_change_raw'].notna()
    for date_col in ('source_fiscal_date', 'source_available_date'):
        dates = pd.to_datetime(t[date_col], errors='coerce')
        if (active & (dates.isna() | dates.gt(cutoff))).any():
            raise ValueError(f'DebtChange 미래/누락 출처일: {date_col}')
    invalid_status = {'missing_source', 'first_report', 'same_fiscal_value_changed', 'fiscal_regression'}
    if (active & t['debt_change_status'].isin(invalid_status)).any():
        raise ValueError('비정상 상태의 DebtChange에 숫자값이 있습니다')
    d = data.copy()
    if d[KEY].isna().any().any() or d.duplicated(KEY).any():
        raise ValueError('기본 H2 패널 종목-기준월 키에 결측/중복이 있습니다')
    d = d.merge(t[KEY + ['debt_final_raw', 'debt_change_raw', 'debt_change_status']],
                on=KEY, how='left', validate='one_to_one')
    # 이전에 생성한 DebtChange와 현재 prepare()의 Hybrid 원천이 다른 버전이면 중단.
    both = d['debt_raw'].notna() & d['debt_final_raw'].notna()
    mismatch = both & ~np.isclose(d['debt_raw'].fillna(0),
                                  d['debt_final_raw'].fillna(0),
                                  rtol=1e-8, atol=1e-8)
    if mismatch.any():
        raise ValueError(f'Hybrid 부채 원천 불일치: {int(mismatch.sum()):,}행. DebtChange 재생성 필요')
    one_missing = d['debt_raw'].notna() ^ d['debt_final_raw'].notna()
    # 기본 패널에 없는 종목-월은 left join에서 미매칭될 수 있으므로
    # 원천 일치 여부를 검증할 수 있는 매칭 행에 한정해 검사한다.
    matched = d['debt_change_status'].notna()
    if (one_missing & matched).any():
        raise ValueError('Hybrid 부채 원천의 결측 위치가 DebtChange CSV와 다릅니다')
    d = d.drop(columns=['debt_final_raw'])
    # DebtChange는 부채비율 자체의 차이(원천 단위가 %이면 %p)로 사용한다.
    d['debt_change'] = d['debt_change_raw']
    audit = {'file': str(path), 'sha256': _sha256(path),
             'rows': int(len(t)),
             'matched_rows': int(d['debt_change_status'].notna().sum()),
             'available_rows': int(d['debt_change'].notna().sum()),
             'missing_or_invalid_rows': int(d['debt_change'].isna().sum()),
             'status_counts': {str(k): int(v) for k, v in
                               d['debt_change_status'].value_counts(dropna=False).items()},
             'definition': 'Latest distinct fiscal report debt ratio minus previous distinct report; carry forward',
             'pit_limitation': 'Derived from Hybrid vintage, not verified full historical PIT'}
    return d, audit


def add_policy_change(data: pd.DataFrame) -> pd.DataFrame:
    """기준월 t의 정책 기조 변화 = PolicyScore(t) - PolicyScore(t-1).

    같은 기준월의 PolicyScore는 모든 종목에서 동일해야 한다.
    이전 '관측월'이 아닌 정확한 직전 '달력월'의 값과 비교한다.
    직전 달 점수가 없는 월은 NaN (임의 0/선형보간 없음).
    """
    d = data.copy()
    required = {'feature_month', 'policy', 'debt', 'policy_x_debt'}
    if not required.issubset(d.columns):
        raise ValueError(f'H2 필수 열 누락: {sorted(required - set(d.columns))}')
    if d['feature_month'].isna().any() or d['policy'].isna().any():
        raise ValueError('기준월 또는 정책점수에 결측이 있습니다.')
    monthly = d.groupby('feature_month', sort=True)['policy'].agg(['min', 'max', 'first'])
    if not np.allclose(monthly['min'], monthly['max'], atol=1e-12, rtol=0):
        raise ValueError('동일 기준월에 다른 PolicyScore가 존재합니다.')
    try:
        months = pd.PeriodIndex(monthly.index.astype(str), freq='M')
    except (TypeError, ValueError) as exc:
        raise ValueError('feature_month가 YYYY-MM 형식이 아닙니다.') from exc
    if months.has_duplicates:
        raise ValueError('정책 기준월 중복 발생')
    # 월 누락이 있는 경우 정책변화량을 서로 떨어진 관측월 차이로 계산하지 않는다.
    full = pd.period_range(months.min(), months.max(), freq='M')
    policy = pd.Series(monthly['first'].to_numpy(dtype=float), index=months).reindex(full)
    changes = policy - policy.shift(1)
    mapping = pd.Series(changes.to_numpy(dtype=float), index=full.astype(str))
    d['policy_change'] = d['feature_month'].astype(str).map(mapping).astype(float)
    d['policy_change_x_debt'] = d['policy_change'] * d['debt']
    return d


def feature_sets() -> OrderedDict[str, list[str]]:
    """H1/H3 제외: 기존 6종 + DebtChange 관련 3종."""
    return OrderedDict([
        ('H2_0_debt', ['debt']),
        ('H2_1_debt_policy', ['debt', 'policy']),
        ('H2_2_level_interaction', ['debt', 'policy', 'policy_x_debt']),
        ('H2_3_policy_change', ['debt', 'policy', 'policy_change']),
        ('H2_4_change_interaction', ['debt', 'policy', 'policy_change', 'policy_change_x_debt']),
        ('H2_5_both_interactions', ['debt', 'policy', 'policy_change',
                                    'policy_change_x_debt', 'policy_x_debt']),
        ('H2_6_debt_change', ['debt', 'debt_change']),
        ('H2_7_debt_policy_change', ['debt', 'policy', 'policy_change', 'debt_change']),
        ('H2_8_debt_change_interaction', ['debt', 'policy', 'policy_change',
                                         'debt_change', 'policy_change_x_debt']),
    ])


COMPARISONS = (
    ('H2_1_debt_policy', 'H2_0_debt', 'Policy_given_Debt'),
    ('H2_2_level_interaction', 'H2_1_debt_policy', 'LevelInteraction_given_PolicyDebt'),
    ('H2_3_policy_change', 'H2_1_debt_policy', 'PolicyChange_given_PolicyDebt'),
    ('H2_4_change_interaction', 'H2_3_policy_change', 'ChangeInteraction_given_Change'),
    ('H2_5_both_interactions', 'H2_4_change_interaction', 'LevelInteraction_given_Change'),
    ('H2_4_change_interaction', 'H2_2_level_interaction', 'ChangeSpecification_vs_LevelInteraction'),
    ('H2_5_both_interactions', 'H2_0_debt', 'FullH2_vs_DebtOnly'),
    ('H2_6_debt_change', 'H2_0_debt', 'DebtChange_given_Debt'),
    ('H2_7_debt_policy_change', 'H2_3_policy_change', 'DebtChange_given_PolicyChange'),
    ('H2_8_debt_change_interaction', 'H2_7_debt_policy_change', 'ChangeInteraction_given_DebtChange'),
    ('H2_8_debt_change_interaction', 'H2_4_change_interaction', 'DebtChange_given_ChangeInteraction'),
)


def run_experiment(data, groups, models, start_year, end_year):
    """동일 학습 표본, 동일 평가 대상, 기존 모델/RANKIC 계산 규칙."""
    required = list(dict.fromkeys(c for cols in groups.values() for c in cols))
    training_ok = data['eligible'] & data[required + ['y']].notna().all(axis=1)
    # 기존 6종도 신규 DebtChange 유효 표본에서만 평가: 공정한 paired 비교.
    evaluation_ok = training_ok
    rows, fold_rows = [], []
    for year in range(start_year, end_year + 1):
        train_mask = training_ok & data.target_month.lt(f'{year}-01')
        valid_mask = evaluation_ok & data.target_month.between(f'{year}-01', f'{year}-12')
        n_train, n_valid = int(train_mask.sum()), int(valid_mask.sum())
        if n_train < 100 or n_valid < 50:
            raise RuntimeError(f'{year}: 학습/검증 표본 부족: {n_train}/{n_valid}')
        target = data.loc[valid_mask, ['stock_id', 'target_month', 'y']].copy()
        y_train = data.loc[train_mask, 'y'].to_numpy(dtype=float)
        fold_rows.append({'year': year, 'train_rows': n_train, 'eval_rows': n_valid,
                          'months': int(target.target_month.nunique())})
        for set_name, cols in groups.items():
            X_train = data.loc[train_mask, cols].to_numpy(dtype=float)
            X_valid = data.loc[valid_mask, cols].to_numpy(dtype=float)
            for model_name in models:
                model = _build_model(model_name)
                model.fit(X_train, y_train)
                predictions = target.copy()
                predictions['pred'] = model.predict(X_valid)
                for month, part in predictions.groupby('target_month', sort=True):
                    rows.append({'year': year, 'model': model_name, 'feature_set': set_name,
                                 'target_month': month, 'n': len(part),
                                 'ic': _ic(part.pred.to_numpy(dtype=float),
                                           part.y.to_numpy(dtype=float))})
            print(f'{year} {set_name} | train={n_train:,}, valid={n_valid:,}', flush=True)

    monthly = pd.DataFrame(rows)
    summary = (monthly.groupby(['model', 'feature_set'], sort=True)
               .agg(months=('ic', 'count'), mean_rankic=('ic', 'mean'),
                    std_ic=('ic', 'std'), median_rankic=('ic', 'median'),
                    positive_ic_share=('ic', lambda s: float((s > 0).mean())),
                    eval_rows=('n', 'sum')).reset_index())
    summary['icir_monthly'] = summary.mean_rankic / summary.std_ic.replace(0, np.nan)

    wide = monthly.pivot(index=['model', 'target_month'], columns='feature_set', values='ic')
    contrast_rows, paired_rows = [], []
    for added, reference, label in COMPARISONS:
        delta = (wide[added] - wide[reference]).dropna()
        if delta.empty:
            raise RuntimeError(f'RankIC 비교 가능한 월이 없습니다: {label}')
        for (model_name, month), value in delta.items():
            paired_rows.append({'comparison': label, 'model': model_name,
                                'target_month': month, 'delta_ic': float(value)})
        for model_name, part in delta.groupby(level='model'):
            values = part.to_numpy(dtype=float)
            contrast_rows.append({'model': model_name, 'comparison': label,
                                  'added_set': added, 'reference_set': reference,
                                  'n_months': len(values),
                                  'mean_delta_ic': float(values.mean()),
                                  'median_delta_ic': float(np.median(values)),
                                  'positive_month_share': float((values > 0).mean())})
    return (monthly, summary, pd.DataFrame(contrast_rows),
            pd.DataFrame(paired_rows), pd.DataFrame(fold_rows))


def main():
    p = argparse.ArgumentParser(description='H2 독립 변수 9종 비교: DebtChange 확장 및 동일 표본')
    p.add_argument('--root', default='.', help='finance_analytics 프로젝트 최상위 경로')
    p.add_argument('--years', default='2006-2021', help='연도 범위: 2006-2007 또는 2006-2021')
    p.add_argument('--models', default=','.join(ALLOWED_MODELS))
    p.add_argument('--debt-source', choices=['hybrid'], default='hybrid',
                   help='DebtChange는 Hybrid 원천에서 생성하므로 hybrid만 지원')
    p.add_argument('--debt-change', type=Path, default=DEFAULT_DEBT_CHANGE,
                   help='build_h2_debt_change --execute로 생성된 CSV')
    p.add_argument('--out', default=None, help='신규 결과 폴더. 기존 폴더가 있으면 중단')
    p.add_argument('--check-only', action='store_true', help='입력 및 Feature 구성 검사, 결과 저장 없음')
    args = p.parse_args()

    begin, end = map(int, args.years.split('-'))
    if not (2006 <= begin <= end <= 2021):
        raise ValueError('--years는 2006~2021 범위여야 합니다.')
    models = [name.strip() for name in args.models.split(',') if name.strip()]
    if not models or any(name not in ALLOWED_MODELS for name in models):
        raise ValueError('사용 가능한 모델: ' + ', '.join(ALLOWED_MODELS))
    root = Path(args.root).resolve()
    d, base, h2, checks = prepare(root, debt_source=args.debt_source)
    d = add_policy_change(d)
    change_path = args.debt_change if args.debt_change.is_absolute() else root / args.debt_change
    d, debt_audit = add_debt_change(d, change_path)
    sets = feature_sets()
    needed = list(dict.fromkeys(c for cols in sets.values() for c in cols))
    common = d.eligible & d[needed + ['y']].notna().all(axis=1)
    valid = common  # 전 조합에 동일한 평가 종목/월 적용
    details = {
        **checks,
        'experiment': 'H2-only 9 sets with DebtChange; shared complete-case samples',
        'debt_change_audit': debt_audit,
        'same_train_and_validation_sample_for_all_sets': True,
        'feature_sets': sets,
        'requested_models': models,
        'validation_years': args.years,
        'debt_source': args.debt_source,
        'h1_h3_used_as_model_inputs': False,
        'h1_h3_feature_columns_loaded_but_unused': base,
        'common_complete_train_rows': int(common.sum()),
        'eval_eligible_target_rows': int(valid.sum()),
        'policy_change_missing_rows': int(d['policy_change'].isna().sum()),
        'policy_change_definition': 'PolicyScore(t) - PolicyScore(t-1 calendar month)',
        'missing_policy_month_handling': 'no forward-fill, missing if calendar predecessor unavailable',
        'limitations': [
            'PolicyChange is a month-to-month text-score change, not an identified monetary surprise.',
            'Policy level and change constant across stocks within each month.',
            'All configurations use eligible complete-case observations including DebtChange; compared only within this run.',
            'Old H2-only runs have a different sample and must not be compared numerically without refitting.',
            'Feature selection must precede evaluation on 2022-2025 holdout.',
        ],
    }
    print('H2-only 변수 실험:', json.dumps(sets, indent=2, ensure_ascii=False))
    print('Hybrid 부채 사용:', args.debt_source == 'hybrid')
    print('공통 학습 후보:', int(common.sum()), '행')
    print('평가 대상 후보:', int(valid.sum()), '행')
    print('PolicyChange 결측 행:', int(d.policy_change.isna().sum()))
    print('DebtChange 결측 행:', int(d.debt_change.isna().sum()))
    print('DebtChange 상태별 건수:', debt_audit['status_counts'])
    print('모델:', ', '.join(models), '| 검증 연도:', args.years)
    if args.check_only:
        print('[검사 완료] 학습/출력 없음')
        return

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(args.out).resolve() if args.out else root / 'outputs' / 'h2_only_debtchange_ablation' / stamp
    if out.exists():
        raise FileExistsError(f'기존 출력 보호: {out}')
    monthly, summary, comparisons, paired, folds = run_experiment(d, sets, models, begin, end)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'data_checks.json').write_text(json.dumps(details, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    frames = {
        'feature_sets.csv': pd.DataFrame([{'feature_set': name, 'features': ','.join(cols),
                                            'n_features': len(cols)} for name, cols in sets.items()]),
        'monthly_rankic.csv': monthly,
        'model_summary.csv': summary,
        'comparisons.csv': comparisons,
        'monthly_paired_delta.csv': paired,
        'fold_coverage.csv': folds,
    }
    for filename, frame in frames.items():
        frame.to_csv(out / filename, index=False, encoding='utf-8-sig')
    print('\n모델별 평균 RankIC\n', summary[['model', 'feature_set', 'mean_rankic']].to_string(index=False))
    print('\n비교별 ΔRankIC\n', comparisons[['model', 'comparison', 'mean_delta_ic']].to_string(index=False))
    print('\n결과 폴더:', out)


if __name__ == '__main__':
    main()
