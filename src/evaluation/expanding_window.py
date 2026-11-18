# 목표월을 기준으로 학습·검증 구간을 나누는 연별 확장창을 정의한다.
# Fold 시간순서, 내부 검증 분할 및 Test 접근 범위를 검사한다.
# 산출물: 기간 구분이 추가된 표본표와 연도별 학습·검증 Fold 목록(메모리).

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from src.contracts import Config
from src.models.interface import FoldSpec
if TYPE_CHECKING:
    from pandas import DataFrame


def _years(config: Config, name: str) -> tuple[int, int]:
    start, end = config['PERIODS'][name]
    return int(start), int(end)


def assign_target_month_split(samples: DataFrame, *, config: Config) -> DataFrame:
    """target_month로 warmup/train/test 구분 열 `period`를 붙인 사본을 반환한다.

    준비기간은 시차와 EWMA 초기화에만 쓰고 학습·평가에서 제외한다.
    범위 밖 월과 잘못된 월 형식은 오류로 처리한다.
    """
    import pandas as pd
    from src.preprocessing.panel import normalize_month
    normalize_month(samples.target_month.tolist())
    year = samples.target_month.str[:4].astype(int)
    out = samples.copy()
    out['period'] = None
    for name in ('warmup', 'train', 'test'):
        lo, hi = _years(config, name)
        out.loc[year.between(lo, hi), 'period'] = name
    if out.period.isna().any():
        raise ValueError('target_month_outside_periods')
    return out


def plan_annual_expanding_folds(*, config: Config, available_target_months: Sequence[str] | None = None) -> Sequence[FoldSpec]:
    """기간 설정 -> 연별 확장창 Fold 목록.

    학습 목표월 train_start_year-01 ~ (Y-1)-12, 검증 Y-01 ~ Y-12,
    Y = first_validation_year ~ last_validation_year (기본 2006~2021, 16개).
    available_target_months를 주면 모든 검증월이 있어야 한다(비연속·누락이면 오류).
    """
    f = config['FOLDS']
    if f.get('split_key') != 'target_month':
        raise ValueError('folds_must_split_on_target_month')
    start, first, last = int(f['train_start_year']), int(f['first_validation_year']), int(f['last_validation_year'])
    if int(f['initial_train_end_year']) != first - 1 or not start < first <= last:
        raise ValueError('fold_boundaries_inconsistent')
    train_lo, train_hi = _years(config, 'train')
    test_lo, _ = _years(config, 'test')
    if start < train_lo or last > train_hi or last >= test_lo:
        raise PermissionError('folds_must_stay_inside_train_period')
    folds = [FoldSpec(fold=str(y), train_start=f'{start}-01', train_end=f'{y-1}-12',
                      valid_start=f'{y}-01', valid_end=f'{y}-12') for y in range(first, last + 1)]
    if available_target_months is not None:
        months = set(available_target_months)
        for fold in folds:
            y = int(fold['fold'])
            missing = [f'{y}-{m:02d}' for m in range(1, 13) if f'{y}-{m:02d}' not in months]
            if missing:
                raise ValueError('validation_months_missing:' + ','.join(missing))
    return folds


def inner_validation_split(fold: FoldSpec, *, years: int = 1) -> FoldSpec:
    """Fold의 학습 구간 마지막 `years`년을 내부 검증으로 뗀 Fold.

    바깥 검증 구간은 포함하지 않는다. 내부 학습이 1년 미만이면 오류.
    """
    start, end = int(fold['train_start'][:4]), int(fold['train_end'][:4])
    if years < 1 or end - years < start:
        raise ValueError('inner_validation_leaves_no_training_year')
    return FoldSpec(fold=fold['fold'] + '_inner', train_start=fold['train_start'], train_end=f'{end-years}-12',
                    valid_start=f'{end-years+1}-01', valid_end=fold['train_end'])


def validate_fold_chronology(train_keys: DataFrame, valid_keys: DataFrame, *, fold: FoldSpec) -> None:
    """학습/검증 (stock_id, target_month) 키가 Fold 경계 안에 있고 시간순인지 검사한다.

    max(train target) < min(valid target), 키 교집합 없음, 테스트 기간(2022~) 혼입 금지.
    """
    for name, keys, lo, hi in (('train', train_keys, fold['train_start'], fold['train_end']),
                               ('valid', valid_keys, fold['valid_start'], fold['valid_end'])):
        if not len(keys):
            raise ValueError(f'empty_{name}_keys')
        if keys.target_month.ge('2022-01').any():
            raise PermissionError('test_period_in_fold')
        if not keys.target_month.between(lo, hi).all():
            raise ValueError(f'{name}_keys_outside_fold')
        if keys.duplicated(['stock_id', 'target_month']).any():
            raise ValueError(f'duplicate_{name}_keys')
    if not train_keys.target_month.max() < valid_keys.target_month.min():
        raise ValueError('train_not_before_valid')
    overlap = train_keys[['stock_id', 'target_month']].merge(valid_keys[['stock_id', 'target_month']])
    if len(overlap):
        raise ValueError('train_valid_keys_overlap')


def verify_no_test_access(*, stage: str, declared_target_years: Sequence[int], source_scope_verified: bool) -> None:
    """파일을 열기 전에 단계·목표연도·범위 확인 여부로 접근을 검사한다.

    final-test 단계, 2022년 이후 목표연도, 범위가 확인되지 않은 자료는 차단한다.
    읽은 뒤 기간을 거르는 방식으로 이 검사를 대체하지 않는다.
    """
    if stage == 'final-test' or not source_scope_verified:
        raise PermissionError('Test 데이터 접근이 차단되었습니다.')
    if not declared_target_years or any(int(y) >= 2022 or int(y) < 2001 for y in declared_target_years):
        raise PermissionError('Test 데이터 접근이 차단되었습니다.')
