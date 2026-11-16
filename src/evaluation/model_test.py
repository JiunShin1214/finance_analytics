"""v2 모델 비교(model_test): 공통 입력 준비 + linear_model(OLS·Ridge) 실행. 설계는 config MODEL_GRID와 노션 「11주차 v2 모델」.

입력은 data/interim(panel, eligibility)·data/processed(h1_components, h3)와 config뿐이다. outputs/의 이전 결과를
읽지 않고 이어서 돌리지 않는다(같은 세션 메모리로 표를 만든다). 저장은 --execute일 때만, 쓰기 전용.

흐름
1. prepare_inputs: λ 후보 10개의 σ(한 번에), 매달 변환 입력(순위 [−1, 1] 또는 z점수), 목표값 3종(그 달 학습 행 기준), 학습·예측 행.
2. plan_folds: 학습 2003년 시작, 바깥 검증 2006~2021(16개), 내부 검증 = 학습 기간 마지막 1년, 재학습 없음.
3. h1_index: 지수형 ① — 각 해 Y의 H1 지수는 2001~Y−2로 학습·Y−1로 조정값을 고른 ①(②와 같은 모델·목표값·손실)의
   예측. 매달 순위 [−1, 1]로 바꿔 ②에 넣는다.
4. run_linear: Fold × (모델·목표값·손실·입력 조합)마다 조정값(H3 λ × Ridge λ)을 내부 검증 평균 IC 최고 한 쌍으로 고르고
   (정확히 같으면 Ridge λ 큰 쪽 → H3 긴 기간 쪽), 내부 학습 모델 그대로 바깥 검증 연도를 예측.
5. 보고 표 저장: outputs/model_results/<작성자>/<시각>_<실험>_<모델·단계>/ (표 + result.zip, 예측값 parquet는 PC에만).

검사(2026-11-15 결정 가·마): 테스트 기간 차단만 남긴다. 설정값·승인 범위·결과 형식 검사는 하지 않는다 — 무엇을 돌렸는지는
결과 표의 열(모델·목표값·손실·hparams·입력 조합·입력 변환)과 폴더 이름, 코드는 Git 이력으로 본다.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.expanding_window import verify_no_test_access
from src.features.h3 import compute_h3_sigma_grid, h3_lambda_candidates
from src.models import boosting, linear

COMPONENTS = ('size', 'bm', 'mom', 'sigma')
H1_PARTS = ('size', 'bm', 'mom')
TARGETS = ('raw', 'winsor_1_99', 'monthly_zscore')
RANK_TARGETS = ('decile', 'percentile')                # 순위 모델 정답: 10분위 floor(10(r−1)/N), 백분위 r/(N+1) [2026-11-14 결정]
BOOSTING_SETS = ('ind:sigma', 'ind:size+bm+mom+sigma', 'idx:h1', 'idx:h1+sigma')   # 부스팅은 주 결과 입력만(노션 §5.1)
# 실행 단계(2026-11-14 결정): 개별형을 먼저, 지수형은 ①(H1 지수) → ②로 따로. [σ] 단독은 지수형 3조합의 H3 칸이라 ②에 둔다.
BOOSTING_PHASES = {'individual': ('ind:size+bm+mom+sigma',), 'index': ('ind:sigma', 'idx:h1', 'idx:h1+sigma')}
WINSOR_Q = (0.01, 0.99)
BASE_FIRST, BASE_LAST = '2000-12', '2021-11'          # 목표월 2001-01~2021-12
TRAIN_START, INDEX_TRAIN_START = 2003, 2001
PERIODS = {'2006~2010': (2006, 2010), '2011~2021': (2011, 2021), '2018~2021': (2018, 2021)}
INNER_SCOPES = ('full', 'common')


# ---------- 설정 ----------

def check_no_test(config) -> None:
    """남기는 유일한 설정 검사: 최종 테스트는 꺼져 있어야 한다(테스트 기간 차단)."""
    if config.get('FINAL_TEST_ENABLED') is not False:
        raise PermissionError('final_test_must_stay_disabled')


def input_sets() -> dict:
    """서로 다른 입력 조합 17개: 개별형 15개 + 지수형 [H1 지수], [H1 지수, σ]. 지수형 [σ]는 개별형 [σ]와 같아 한 번만."""
    sets = {}
    for k in range(1, 5):
        for combo in itertools.combinations(COMPONENTS, k):
            sets['ind:' + '+'.join(combo)] = list(combo)
    sets['idx:h1'] = ['h1_index']
    sets['idx:h1+sigma'] = ['h1_index', 'sigma']
    return sets


# ---------- 1) 입력 준비 ----------

def rank_pm1(values: pd.Series, member: np.ndarray, months: pd.Series) -> np.ndarray:
    """매달 member 행 안 평균 동순위 r → 2r/(N+1) − 1. member가 아니면 NaN."""
    v = values.where(member)
    r = v.groupby(months.to_numpy()).rank(method='average')
    n = member.astype(int)
    count = pd.Series(n).groupby(months.to_numpy()).transform('sum').to_numpy()
    out = 2 * r.to_numpy() / (count + 1) - 1
    return np.where(member, out, np.nan)


INPUT_TRANSFORMS = ('rank', 'zscore', 'winsor_zscore')


def zscore_by_month(values: pd.Series, member: np.ndarray, months: pd.Series, *, winsor: bool = False) -> np.ndarray:
    """매달 member 행 안 (값 − 평균) ÷ 표준편차(ddof=1) [프로젝트 명세 §6.4의 H1 정의]. member가 아니면 NaN.
    winsor=True면 먼저 그 달 member 행의 1%·99% 분위수에서 자른다(관례값, 튜닝 7번 비교용 2026-11-15)."""
    m = months.to_numpy()
    v = values.where(member).astype(float)
    if winsor:
        g = v.groupby(m)
        v = v.clip(lower=g.transform(lambda s: s.quantile(WINSOR_Q[0])), upper=g.transform(lambda s: s.quantile(WINSOR_Q[1])))
    g = v.groupby(m)
    z = ((v - g.transform('mean')) / g.transform('std')).to_numpy(float)
    return np.where(member & np.isfinite(z), z, np.nan)


def transform_input(values: pd.Series, member: np.ndarray, months: pd.Series, how: str) -> np.ndarray:
    if how == 'rank':
        return rank_pm1(values, member, months)
    if how in ('zscore', 'winsor_zscore'):
        return zscore_by_month(values, member, months, winsor=how == 'winsor_zscore')
    raise ValueError('input_transform_unknown:' + str(how))


def monthly_target_transforms(y: pd.Series, train: np.ndarray, months: pd.Series) -> dict:
    """목표값 3종 + 순위 정답 2종. 경계·평균·표준편차·순위는 그 목표월 학습 행에서만 계산하고 학습 행에만 값을 둔다.

    순위 정답(2026-11-14 결정): 그 달 학습 행 안 평균 동순위 r(1~N)으로 10분위 = floor(10(r−1)/N) ∈ {0..9},
    백분위 = r/(N+1) ∈ (0, 1)(입력 변환 2r/(N+1)−1과 같은 식, GKX 부록 식 A.1).
    """
    m = months.to_numpy()
    yt = y.where(train)
    g = yt.groupby(m)
    lo, hi = g.transform(lambda s: s.quantile(WINSOR_Q[0])), g.transform(lambda s: s.quantile(WINSOR_Q[1]))
    mean, sd = g.transform('mean'), g.transform(lambda s: s.std(ddof=1))
    rank, count = g.rank(method='average'), g.transform('count')
    out = {'raw': yt.to_numpy(float),
           'winsor_1_99': yt.clip(lower=lo, upper=hi).to_numpy(float),
           'monthly_zscore': ((yt - mean) / sd).to_numpy(float),
           'decile': np.floor(10 * (rank - 1) / count).to_numpy(float),
           'percentile': (rank / (count + 1)).to_numpy(float)}
    for name, arr in out.items():
        if not np.isfinite(arr[train]).all() or np.isfinite(arr[~train]).any():
            raise ValueError('target_transform_invalid:' + name)
    return out


def build_model_data(comp: pd.DataFrame, sigma_grid: pd.DataFrame, returns: pd.DataFrame, labels: list[str], *, units,
                     input_transform: str = 'rank') -> pd.DataFrame:
    """특징(기준월 2000-12~2021-11)·λ별 σ·개발 수익률 → 종목×목표월 모델 입력(메모리).

    순위 입력은 매달 적격·산출 종목 안에서, 목표값 3종은 그 목표월 학습 행 안에서 만든다(목표값은 입력을 다 만든 뒤 결합).
    train = 적격이고 H1 구성요소 3개·σ·목표값이 모두 있는 행(조합 간 공통), predict = 목표값(다음 달 수익률)이 있는 행.
    """
    from src.preprocessing.panel import join_training_targets
    keys = ['stock_id', 'base_month']
    c = comp[comp.base_month.between(BASE_FIRST, BASE_LAST)]
    d = c[keys + ['eligible', *H1_PARTS, *[p + '_available' for p in H1_PARTS]]].merge(
        sigma_grid[keys + ['eligible', 'vol_available', *['sigma_' + x for x in labels]]].rename(columns={'eligible': 'eligible_h3'}),
        on=keys, how='left', validate='one_to_one')
    if len(d) != len(c) or not d.eligible.equals(d.eligible_h3):
        raise ValueError('component_h3_rows_or_eligibility_differ')
    r = returns[['stock_id', 'month', 'return']].copy()
    r.attrs['units'] = units
    d = join_training_targets(d.drop(columns='eligible_h3'), r)
    d = d.sort_values(['target_month', 'stock_id']).reset_index(drop=True)
    if d.target_month.ge('2022-01').any():
        raise PermissionError('test_period_in_samples')
    elig = d.eligible.eq(True).fillna(False).to_numpy(bool)
    avail = {p: d[p + '_available'].eq(True).to_numpy(bool) & elig for p in H1_PARTS}
    avail['sigma'] = d.vol_available.eq(True).to_numpy(bool) & elig
    months = d.base_month
    data = d[['stock_id', 'base_month', 'target_month', 'eligible', 'y']].copy()
    data['year'] = d.target_month.str[:4].astype(int).to_numpy()
    for p in H1_PARTS:
        data['x_' + p] = transform_input(d[p], avail[p], months, input_transform)
    for x in labels:
        data['x_sigma_' + x] = transform_input(d['sigma_' + x], avail['sigma'], months, input_transform)
    data['h1_available'] = avail['size'] & avail['bm'] & avail['mom']
    data['train'] = data.h1_available.to_numpy() & avail['sigma'] & d.y.notna().to_numpy()
    data['predict'] = d.y.notna().to_numpy()
    for name, arr in monthly_target_transforms(d.y, data.train.to_numpy(), data.target_month).items():
        data['t_' + name] = arr
    return data


# 특징 변환: rank = 매달 순위 [−1, 1](기본), none = 값 그대로(순위 변환 예외, H2 debt·policy_x_debt)
TARGET_COLUMNS = TARGETS + RANK_TARGETS


def feature_spec(spec) -> dict:
    """특징 지정 → {'path', 'column', 'transform'}. 문자열이면 경로(열 이름 = 특징 이름, 순위 변환)."""
    spec = {'path': spec} if isinstance(spec, (str, Path)) else dict(spec)
    spec['path'] = Path(spec['path']).as_posix()
    spec.setdefault('transform', 'rank')
    return spec


def _data_path(path) -> Path:
    rel = Path(path)
    if rel.is_absolute() or not rel.as_posix().startswith('data/'):
        raise PermissionError(f'feature_file_must_be_under_data:{rel.as_posix()}')
    return rel


def load_extra_feature(root: Path, name: str, path, *, column: str | None = None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """특징 파일(data/ 아래 parquet) 읽기. 열: stock_id, base_month(YYYY-MM), <column>(기본 = name, 없으면 NaN).
    종목 × 기준월 한 줄, 기준월 2000-12~2021-11만(그 뒤는 테스트 기간이라 거부). t월 말에 알 수 있던 값인지는 만든 사람 책임.
    한 파일에서 여러 열을 받을 때는 frame(이미 읽은 표)을 넘겨 파일을 한 번만 읽는다. 반환 열 이름은 name."""
    column = column or name
    f = pd.read_parquet(root / _data_path(path)) if frame is None else frame
    f = f[['stock_id', 'base_month', column]].rename(columns={column: name}).copy()
    f['stock_id'] = f.stock_id.astype(str)
    f['base_month'] = f.base_month.astype(str)
    if f.base_month.gt(BASE_LAST).any():
        raise PermissionError(f'feature_has_test_period_months:{name}')
    f[name] = pd.to_numeric(f[name], errors='raise').astype(float)
    return f[f.base_month.ge(BASE_FIRST)]


def add_extra_features(data: pd.DataFrame, features: dict, *, transforms: dict | None = None,
                       train_start_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """특징 표들을 붙인다: 매달 적격·관측 종목 안 순위 [−1, 1](x_<이름>). transforms[이름] = 'none'이면 값 그대로(순위 변환 예외,
    config H2.model_test). 학습 행은 새 특징까지 모두 관측된 행만(2026-11-15 결정 A, 교수님 지침 "학습은 관측값만"), 예측 행은
    그대로(빈 특징은 예측 때 0). 학습 행이 바뀌므로 목표값 변환(윈저·월별 표준화·10분위·백분위)을 관측 행 안에서 다시 만든다.
    observed = 새 특징까지 관측된 모든 해의 행(붙인 지수 ①의 학습용). train_start_year를 주면 ②의 학습 행(train)은 그 목표연도부터
    (H2: 2004, 2026-11-15 결정). 기존 학습 행·목표값은 base_train·base_t_<목표값>으로 남긴다(H1 지수 ①은 기존 행 그대로).
    반환: (새 표, 연도별 관측 비율 표)."""
    d = data.copy()
    transforms = transforms or {}
    if 'base_train' not in d:
        d['base_train'] = d.train
        for name in TARGET_COLUMNS:
            d['base_t_' + name] = d['t_' + name]
    elig = d.eligible.eq(True).fillna(False).to_numpy(bool)
    coverage = []
    for name, f in features.items():
        merged = d[['stock_id', 'base_month']].merge(f, on=['stock_id', 'base_month'], how='left', validate='one_to_one')
        avail = merged[name].notna().to_numpy() & elig
        if transforms.get(name, 'rank') == 'none':
            d['x_' + name] = np.where(avail, merged[name].to_numpy(float), np.nan)
        else:
            d['x_' + name] = rank_pm1(merged[name], avail, d.base_month)
        d[name + '_available'] = avail
        before = d.train.to_numpy(bool)
        d['train'] = before & avail
        for y_, g in d.groupby('year'):
            coverage.append({'feature': name, 'target_year': int(y_), 'predict_rows': int(g.predict.sum()),
                             'available_share_of_predict': float((g.predict & g[name + '_available']).sum() / max(1, g.predict.sum())),
                             'train_rows_before': int(before[g.index].sum()), 'train_rows_after': int(d.train[g.index].sum())})
    d['observed'] = d.train
    for name, arr in monthly_target_transforms(d.y, d.observed.to_numpy(), d.target_month).items():
        d['t_' + name] = arr
    if train_start_year is not None:
        d['train'] = d.observed & d.year.ge(int(train_start_year))
    return d, pd.DataFrame(coverage)


def prepare_inputs(root: Path, config, *, h3_grid: pd.DataFrame | None = None, extra_features: dict | None = None,
                   indexes: dict | None = None, train_start_year: int | None = None, input_transform: str = 'rank') -> dict:
    """공유 표 3개(+ 특징 파일)만 읽어 메모리 입력을 만든다. 반환 dict: data(종목×월 표), sigma_labels, checks(행 수 등 정보).

    extra_features = {이름: 'data/processed/<파일>.parquet' 또는 {'path', 'column', 'transform'}}. 주면 학습 행이 새 특징까지
    관측된 행으로 줄어든다(예측 행은 그대로). 한 파일에서 여러 열을 받을 수 있다(파일은 한 번만 읽고 해시도 한 번).
    indexes = {'<이름>_index': {'parts': [특징…], 'transform': 'rank'|'centered_raw'}}: 붙인 특징으로 만드는 지수형 지수(①).
    train_start_year: ② 학습 시작 목표연도. H2는 h2_inputs(config)가 세 인자를 config에서 만든다.
    """
    check_no_test(config)
    verify_no_test_access(stage='validate', declared_target_years=list(range(2001, 2022)), source_scope_verified=True)
    paths = config['PATHS']
    panel = pd.read_parquet(root / paths['panel'], columns=['stock_id', 'month', 'return', 'eligible'])
    if not panel.month.between('2000-01', '2021-12').all():
        raise PermissionError('development_months_only')
    eligibility = pd.read_parquet(root / paths['eligibility'])
    comp = pd.read_parquet(root / paths['h1_components'])
    if h3_grid is None:
        h3_grid = compute_h3_sigma_grid(panel, eligibility)
    cand = h3_grid.attrs.get('h3_lambda_candidates', h3_lambda_candidates())
    labels = list(cand['label'])
    g = h3_grid.sort_values(['stock_id', 'base_month']).reset_index(drop=True)
    data = build_model_data(comp, g, panel[['stock_id', 'month', 'return']], labels, units=config['UNITS'],
                            input_transform=input_transform)
    checks = {'input_transform': input_transform}
    coverage = pd.DataFrame()
    specs = {n: feature_spec(s) for n, s in (extra_features or {}).items()}
    indexes = {n: {'parts': list(s['parts']), 'transform': s.get('transform', 'rank')} for n, s in (indexes or {}).items()}
    if specs:
        files = {s['path']: pd.read_parquet(root / _data_path(s['path'])) for s in specs.values()}
        feats = {n: load_extra_feature(root, n, s['path'], column=s.get('column'), frame=files[s['path']]) for n, s in specs.items()}
        data, coverage = add_extra_features(data, feats, transforms={n: s['transform'] for n, s in specs.items()},
                                            train_start_year=train_start_year)
    val = data[data.predict & data.year.between(2006, 2021)]
    checks.update(validation_prediction_rows=int(len(val)),
                  training_rows_2003_2021=int(data.train[data.year >= TRAIN_START].sum()),
                  training_rows_2001_2021=int(data.train.sum()))
    if train_start_year is not None:
        checks['train_start_target_year'] = int(train_start_year)
    out = {'data': data, 'sigma_labels': labels, 'lambda_candidates': cand, 'checks': checks,
           'extra_features': {n: (s['path'] if s['transform'] == 'rank' and not s.get('column') else s) for n, s in specs.items()},
           'feature_coverage': coverage}
    if indexes:
        out['indexes'] = indexes
    if train_start_year is not None:
        out['train_start_year'] = int(train_start_year)
    return out


def h2_inputs(config) -> dict:
    """config H2 칸 → prepare_inputs 인자(extra_features·indexes·train_start_year). 정의는 팀원 구현 그대로(2026-11-15 결정):
    debt·policy_x_debt는 순위 변환 예외(z 그대로), H2 지수는 ① 예측 − ①(입력 0) 예측, ② 학습은 2004년부터."""
    h2 = config['H2']['model_test']
    path = config['PATHS']['h2']
    return {'extra_features': {n: {'path': path, 'transform': h2['input_transform_exception']} for n in h2['features']},
            'indexes': {'h2_index': {'parts': list(h2['index_parts']), 'transform': h2['index_transform']}},
            'train_start_year': int(h2['train_start_target_year'])}


def h2_input_sets(config) -> dict:
    """config H2 입력 조합 → {'individual': {조합 이름: 열}, 'index': {...}}. 각각 H2를 넣은 조합 + 같은 행의 비교용 조합."""
    s = config['H2']['model_test']['input_sets']
    comp = [list(c) for c in s['comparison']]
    pick = lambda idx: [c for c in comp if any(x.endswith('_index') for x in c) == idx]   # noqa: E731
    return {'individual': dict(make_input_set(c) for c in [s['individual'], *pick(False)]),
            'index': dict(make_input_set(c) for c in [s['index'], *pick(True)])}


def feature_names(prepared: dict) -> list[str]:
    """입력 조합에 쓸 수 있는 특징 이름: 기본 4개 + 붙인 특징."""
    return list(COMPONENTS) + list(prepared.get('extra_features', {}))


def make_input_set(columns) -> tuple[str, list[str]]:
    """특징 목록 → (조합 이름, 목록). 예: ['size', 'bm', 'mom', 'sigma', 'h2'] → ('ind:size+bm+mom+sigma+h2', ...),
    ['h1_index', 'sigma', 'h2_index'] → ('idx:h1+sigma+h2', ...)."""
    cols = list(columns)
    if any(c.endswith('_index') for c in cols):
        name = 'idx:' + '+'.join(c[:-len('_index')] if c.endswith('_index') else c for c in cols)
    else:
        name = 'ind:' + '+'.join(cols)
    return name, cols


def index_specs(prepared: dict) -> dict:
    """쓸 수 있는 지수: H1 지수(항상, 매달 순위) + 붙인 지수(H2 등)."""
    return {'h1_index': {'parts': list(H1_PARTS), 'transform': 'rank'}, **prepared.get('indexes', {})}


# ---------- 2) Fold ----------

def plan_folds(grid) -> list[dict]:
    """바깥 검증 Y마다 내부 학습 2003~Y−2, 내부 검증 Y−1. 재학습이 없으므로 바깥 검증 예측도 내부 학습 모델로 한다."""
    folds = []
    for k, y in enumerate(range(grid['first_validation_year'], grid['last_validation_year'] + 1), start=1):
        inner_years = (y - 2) - TRAIN_START + 1
        folds.append({'fold': k, 'valid_year': y, 'train_years': (TRAIN_START, y - 2), 'inner_valid_year': y - 1,
                      'inner_train_years': inner_years, 'min_years_exception': inner_years < grid['min_inner_train_years']})
    if folds[-1]['valid_year'] >= 2022:
        raise PermissionError('folds_must_stay_inside_validation')
    return folds


# ---------- 월별 Rank IC(빠른 계산) ----------

def _average_ranks(codes: np.ndarray, s: np.ndarray) -> np.ndarray:
    """월 코드별 평균 동순위(1부터). 정렬 한 번으로 모든 달을 처리한다."""
    n = len(s)
    if not n:
        return np.empty(0)
    order = np.lexsort((s, codes))
    cs, ss = codes[order], s[order]
    new_month = np.r_[True, cs[1:] != cs[:-1]]
    month_start = np.maximum.accumulate(np.where(new_month, np.arange(n), 0))
    pos = np.arange(n) - month_start + 1.0
    new_tie = new_month | np.r_[True, ss[1:] != ss[:-1]]
    starts = np.flatnonzero(new_tie)
    ends = np.r_[starts[1:] - 1, n - 1]
    avg = (pos[starts] + pos[ends]) / 2
    ranks = np.empty(n)
    ranks[order] = avg[np.cumsum(new_tie) - 1]
    return ranks


def _ic_from_parts(codes, n_m, ds, dy):
    """월별 Σds·dy / √(Σds²·Σdy²). 순위 편차는 0.5 단위라 합이 정확히 계산된다(합 순서와 무관)."""
    k = len(n_m)
    sy = np.bincount(codes, ds * dy, minlength=k)
    ss = np.bincount(codes, ds * ds, minlength=k)
    yy = np.bincount(codes, dy * dy, minlength=k)
    denom = np.sqrt(ss * yy)
    with np.errstate(divide='ignore', invalid='ignore'):
        ic = np.where((n_m >= 3) & (denom > 0), sy / np.where(denom > 0, denom, 1.0), np.nan)
    reason = np.select([n_m < 3, ss <= 0, yy <= 0], ['too_few_pairs', 'constant_score', 'constant_target'], '')
    return ic, reason


def monthly_ic(months: np.ndarray, score: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """목표월별 Spearman(평균 동순위의 Pearson). 비유한 쌍 제외, 3쌍 미만·상수는 NaN과 이유.
    rank_ic.rank_ic_for_month와 같은 규칙."""
    score, y = np.asarray(score, float), np.asarray(y, float)
    ok = np.isfinite(score) & np.isfinite(y)
    uniq, codes = np.unique(np.asarray(months)[ok], return_inverse=True)
    n_m = np.bincount(codes, minlength=len(uniq))
    half = (n_m[codes] + 1) / 2
    ds = _average_ranks(codes, score[ok]) - half
    dy = _average_ranks(codes, y[ok]) - half
    ic, reason = _ic_from_parts(codes, n_m, ds, dy)
    return pd.DataFrame({'n': n_m, 'ic': ic, 'reason': reason}, index=pd.Index(uniq, name='target_month'))


def mean_ic(table: pd.DataFrame) -> float:
    return float(table.ic.mean()) if table.ic.notna().any() else np.nan


class InnerScorer:
    """고정된 내부 검증 행의 월별 IC를 빠르게: 원수익률 순위는 한 번만 계산한다(결과는 monthly_ic와 같음)."""

    def __init__(self, months: np.ndarray, y: np.ndarray):
        self.months, self.y = np.asarray(months), np.asarray(y, float)
        if not np.isfinite(self.y).all():
            raise ValueError('inner_validation_target_missing')
        uniq, self.codes = np.unique(self.months, return_inverse=True)
        self.n_m = np.bincount(self.codes, minlength=len(uniq))
        self.half = (self.n_m[self.codes] + 1) / 2
        self.dy = _average_ranks(self.codes, self.y) - self.half

    def score(self, yhat: np.ndarray) -> tuple[float, int]:
        yhat = np.asarray(yhat, float)
        if not np.isfinite(yhat).all():
            t = monthly_ic(self.months, yhat, self.y)
            return mean_ic(t), int(t.ic.notna().sum())
        ic, _ = _ic_from_parts(self.codes, self.n_m, _average_ranks(self.codes, yhat) - self.half, self.dy)
        s = pd.Series(ic)
        return (float(s.mean()) if s.notna().any() else np.nan), int(s.notna().sum())


# ---------- 3·4) 선형 모델 적합·선택 ----------

def _candidates(model: str, sigma_labels: list[str] | None, decay_of: dict) -> list[dict]:
    """H3 λ 바깥, Ridge λ 안쪽(작은 것부터) 순서. 같은 H3 λ 안에서 앞 λ의 Huber 해를 다음 λ의 시작점으로 쓴다."""
    lams = linear.RIDGE_LAMBDAS if model == 'ridge' else (None,)
    labels = sigma_labels if sigma_labels else [None]
    return [{'ridge_lambda': lam, 'h3_label': lab, 'decay': decay_of.get(lab, np.nan)} for lab in labels for lam in lams]


def choose(results: list[dict]) -> tuple[dict, int]:
    """내부 검증 평균 IC 최고 한 쌍. 정확히 같으면 Ridge λ 큰 쪽 → H3 λ 큰 쪽(긴 기간). 반환: (선택, 최고값 동률 수)."""
    alive = [r for r in results if np.isfinite(r['inner_ic'])]
    if not alive:
        raise ValueError('all_candidates_failed')
    best = max(r['inner_ic'] for r in alive)
    tied = [r for r in alive if r['inner_ic'] == best]
    key = lambda r: (r['ridge_lambda'] if r['ridge_lambda'] is not None else -1.0,
                     r['decay'] if np.isfinite(r['decay']) else -1.0)
    return max(tied, key=key), len(tied)


def _design(data: pd.DataFrame, columns: list[str], rows: np.ndarray, *, fill: bool) -> np.ndarray:
    X = data[columns].to_numpy(float)[rows]
    if fill:
        X = np.where(np.isfinite(X), X, 0.0)
    elif not np.isfinite(X).all():
        raise ValueError('training_rows_with_missing_inputs')
    return X


def _columns(input_cols: list[str], label: str | None, index_cols: dict | None) -> list[str]:
    out = []
    for c in input_cols:
        if c == 'sigma':
            out.append('x_sigma_' + label)
        elif c.endswith('_index'):
            out.append((index_cols or {})[c])
        else:
            out.append('x_' + c)
    return out


class FoldContext:
    """한 Fold(또는 지수형 ①의 한 해)의 고정 행. 입력 행렬·목표값·Huber② σ를 한 번만 만들어 모든 설정이 함께 쓴다.

    fit_rows: 학습 행(공통 표본), inner_rows: 내부 검증 평가 행, out_rows: 예측 행(채움 허용).
    """

    def __init__(self, data: pd.DataFrame, fit_rows: np.ndarray, inner_rows: np.ndarray, out_rows: np.ndarray | None = None):
        self.data, self.fit_rows, self.inner_rows, self.out_rows = data, fit_rows, inner_rows, out_rows
        months = data.target_month.to_numpy()
        self.m_fit = months[fit_rows]
        self.scorer = InnerScorer(months[inner_rows], data.y.to_numpy(float)[inner_rows])
        self.n_train = int(fit_rows.sum())
        self._X, self._y, self._sigma, self._q999 = {}, {}, {}, {}
        self._train_scorer = None

    @property
    def train_scorer(self) -> 'InnerScorer':
        """학습 행의 월별 IC(원수익률 기준). 부스팅의 트리별 학습 IC용."""
        if self._train_scorer is None:
            self._train_scorer = InnerScorer(self.m_fit, self.data.y.to_numpy(float)[self.fit_rows])
        return self._train_scorer

    def X(self, cols: list[str], which: str) -> np.ndarray:
        key = (tuple(cols), which)
        if key not in self._X:
            rows = {'fit': self.fit_rows, 'inner': self.inner_rows, 'out': self.out_rows}[which]
            self._X[key] = _design(self.data, cols, rows, fill=which != 'fit')
        return self._X[key]

    def y(self, target: str) -> np.ndarray:
        if target not in self._y:
            self._y[target] = self.data['t_' + target].to_numpy(float)[self.fit_rows]
        return self._y[target]

    def q999(self, target: str) -> float:
        if target not in self._q999:
            self._q999[target] = linear.huber_delta_q999(self.y(target), self.m_fit)
        return self._q999[target]

    def sigma2(self, cols: list[str], target: str) -> dict:
        """Huber② σ: 같은 학습 행·입력·목표값의 HuberRegressor(OLS). OLS·Ridge가 함께 쓴다."""
        key = (tuple(cols), target)
        if key not in self._sigma:
            self._sigma[key] = linear.huber_sigma_sklearn(self.X(cols, 'fit'), self.y(target))
        return self._sigma[key]


def select_and_fit(ctx: FoldContext, *, model, target, loss, input_cols, sigma_labels, decay_of, index_cols=None):
    """한 Fold·한 설정의 조정값 선택. 모든 후보가 같은 학습 행(ctx.fit_rows)으로 학습하고 내부 검증은 원수익률로 평가.
    반환: (선택 결과 dict, 후보별 내부 IC 목록)."""
    has_sigma = 'sigma' in input_cols
    y_fit = ctx.y(target)
    results, warm = [], {}
    for cand in _candidates(model, sigma_labels if has_sigma else None, decay_of):
        cols = _columns(input_cols, cand['h3_label'], index_cols)
        X = ctx.X(cols, 'fit')
        delta, sig = None, None
        if loss == 'huber_q999':
            delta = ctx.q999(target)
        elif loss == 'huber_sigma135':
            sig = ctx.sigma2(cols, target)
            delta = sig['delta']
        try:
            fitted = linear.fit_linear(X, y_fit, model=model, loss=loss, ridge_lambda=cand['ridge_lambda'], delta=delta,
                                       start=warm.get(cand['h3_label']) if loss != 'mse' else None)
        except ValueError as error:
            results.append({**cand, 'inner_ic': np.nan, 'inner_months': 0, 'failure': str(error)})
            continue
        if loss != 'mse':
            warm[cand['h3_label']] = (fitted['intercept'], fitted['coef'])
        ic, months = ctx.scorer.score(linear.predict_linear(fitted, ctx.X(cols, 'inner')))
        results.append({**cand, 'inner_ic': ic, 'inner_months': months, 'failure': '', 'fitted': fitted, 'columns': cols,
                        'huber_sigma': sig['sigma'] if sig else np.nan,
                        'huber_sigma_converged': sig['converged'] if sig else None})
    chosen, ties = choose(results)
    table = [{k: v for k, v in r.items() if k not in ('fitted', 'columns')} for r in results]
    return {**chosen, 'ties': ties, 'n_train': ctx.n_train}, table


def _index_frame(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """지수 ①이 학습할 표. H1 지수: 기존 학습 행·목표값(base_*, 특징을 붙였을 때만 있음) → 기존 실험과 같은 지수(2026-11-15 결정).
    붙인 지수(H2 등): 새 특징까지 관측된 모든 해의 행(observed) — ② 학습 시작 연도 이전 해도 ① 학습에는 쓴다."""
    if name == 'h1_index':
        if 'base_train' not in data:
            return data
        return data.assign(train=data.base_train, **{'t_' + t: data['base_t_' + t] for t in TARGET_COLUMNS})
    return data.assign(train=data.observed)


def _index_member(data: pd.DataFrame, name: str, parts: list[str]) -> np.ndarray:
    """지수를 만드는 행: 구성요소가 모두 있는 적격 행."""
    if name == 'h1_index':
        return data.h1_available.to_numpy(bool)
    return np.logical_and.reduce([data[p + '_available'].to_numpy(bool) for p in parts])


def index_years(data: pd.DataFrame) -> range:
    """① 지수를 만들 해: ② 학습 첫해(기본 2003, H2 실험 2004)~2021."""
    first = int(data.year[data.train].min()) if data.train.any() else TRAIN_START
    return range(max(TRAIN_START, first), 2022)


def _finish_index(raw: np.ndarray, data: pd.DataFrame, transform: str) -> np.ndarray:
    if transform == 'rank':
        member = np.isfinite(raw)
        return rank_pm1(pd.Series(raw), member, data.base_month.reset_index(drop=True))
    return raw                         # centered_raw: 빈 행은 NaN → 예측 때 0(= ①에 입력 0을 넣은 예측)


def build_index(data, name: str, spec: dict, *, model, target, loss, inner_scope, decay_of, years=None):
    """지수형 ① 일반형. 해 Y의 지수 = 2001~Y−2 학습 행으로 맞추고 Y−1 내부 검증으로 조정값을 고른 ①(②와 같은 모델·목표값·손실)의
    예측(재학습 없음). 구성요소(spec['parts'])가 모두 있는 적격 행에만 만든다.
    transform: rank = 매달 그 행 안 순위 [−1, 1](H1), centered_raw = ① 예측 − ①(입력 0) 예측(선형은 절편을 뺀 값, H2 — 순위로
    바꾸면 정책 세기가 사라져서, 2026-11-15 결정). 반환: (지수 배열, 선택 표)."""
    d = _index_frame(data, name)
    parts = list(spec['parts'])
    years = index_years(data) if years is None else years
    raw_index = np.full(len(d), np.nan)
    year = d.year.to_numpy()
    train = d.train.to_numpy(bool)
    scope = d.predict.to_numpy(bool) if inner_scope == 'full' else train
    member = _index_member(d, name, parts)
    rows = []
    for Y in years:
        target_rows = member & (year == Y)
        ctx = FoldContext(d, train & (year >= INDEX_TRAIN_START) & (year <= Y - 2), scope & (year == Y - 1), target_rows)
        chosen, _ = select_and_fit(ctx, model=model, target=target, loss=loss, input_cols=parts,
                                   sigma_labels=None, decay_of=decay_of)
        pred = linear.predict_linear(chosen['fitted'], _design(d, chosen['columns'], target_rows, fill=False))
        center = 0.0
        if spec['transform'] == 'centered_raw':
            center = float(linear.predict_linear(chosen['fitted'], np.zeros((1, len(parts))))[0])
            pred = pred - center
        raw_index[target_rows] = pred
        row = {'index_year': Y, 'model': model, 'target': target, 'loss': loss, 'stage1_train_years': f'{INDEX_TRAIN_START}~{Y-2}',
               'stage1_inner_valid_year': Y - 1, 'ridge_lambda': chosen['ridge_lambda'], 'inner_ic': chosen['inner_ic'],
               'ties': chosen['ties'], 'n_train': chosen['n_train'], 'intercept': chosen['fitted']['intercept'],
               **{'w_' + p: w for p, w in zip(parts, chosen['fitted']['coef'])}}
        if name != 'h1_index':
            row.update(index=name, center=center)
        rows.append(row)
    return _finish_index(raw_index, d, spec['transform']), pd.DataFrame(rows)


def build_h1_index(data, *, model, target, loss, inner_scope, decay_of, years=range(TRAIN_START, 2022)):
    """지수형 ① H1 지수: Size·B/M·Mom → 매달 순위 [−1, 1]. 반환: (순위 지수 배열, 선택 표)."""
    return build_index(data, 'h1_index', {'parts': list(H1_PARTS), 'transform': 'rank'}, model=model, target=target, loss=loss,
                       inner_scope=inner_scope, decay_of=decay_of, years=years)


def index_column(name: str, setting) -> str:
    """지수 열 이름. H1은 예전 이름 그대로(idx_<설정>), 붙인 지수는 <이름>_<설정>."""
    return ('idx_' if name == 'h1_index' else name + '_') + '_'.join(map(str, setting))


def linear_settings():
    return [(m, t, l) for m in linear.LINEAR_MODELS for t in TARGETS for l in linear.LOSSES]


LINEAR_PARTS = [(m, l) for m in linear.LINEAR_MODELS for l in linear.LOSSES]   # 노트북 셀 단위: 모델 × 손실 6개


def part_settings(model: str, loss: str):
    if (model, loss) not in LINEAR_PARTS:
        raise ValueError('linear_part_unknown')
    return [(model, t, loss) for t in TARGETS]


_WORKER = {}


def _init_worker(payload):
    _WORKER.clear()
    _WORKER.update(payload)


def _progress(message: str) -> None:
    """진행 줄. 순차 실행은 바로 출력, 병렬 실행은 큐로 본 프로세스에 보낸다(작업 프로세스 출력은 노트북에 안 보임)."""
    if _WORKER.get('queue') is not None:
        _WORKER['queue'].put(message)
    elif _WORKER.get('log') is not None:
        _WORKER['log'](message)


def _run_fold(fold: dict) -> dict:
    """한 Fold의 모든 (입력 조합 × 설정). 병렬 실행 단위. 파일을 읽거나 쓰지 않는다."""
    p = _WORKER
    data = p['data']
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    predict = data.predict.to_numpy(bool)
    scope = predict if p['inner_scope'] == 'full' else train
    t_fold = time.perf_counter()
    lo, hi = fold['train_years']
    out_rows = predict & (year == fold['valid_year'])
    ctx = FoldContext(data, train & (year >= lo) & (year <= hi), scope & (year == fold['inner_valid_year']), out_rows)
    y_all = data.y.to_numpy(float)
    selections, inner, timings, preds = [], [], [], {}
    n_sets = len(p['sets'])
    for i_set, (set_name, cols) in enumerate(p['sets'].items(), start=1):
        for model, target, loss in p['settings']:
            t0 = time.perf_counter()
            chosen, table = select_and_fit(ctx, model=model, target=target, loss=loss, input_cols=cols,
                                           sigma_labels=p['labels'], decay_of=p['decay_of'],
                                           index_cols=p['index_cols'].get((model, target, loss)))
            tag = {'family': 'linear', 'model': model, 'target': target, 'loss': loss, 'input_set': set_name, 'fold': fold['fold']}
            f = chosen['fitted']
            preds[config_id(tag)] = linear.predict_linear(f, ctx.X(chosen['columns'], 'out'))
            train_ic = monthly_ic(ctx.m_fit, linear.predict_linear(f, ctx.X(chosen['columns'], 'fit')), y_all[ctx.fit_rows])
            selections.append({**tag, 'valid_year': fold['valid_year'], 'inner_valid_year': fold['inner_valid_year'],
                               'inner_train_years': fold['inner_train_years'], 'min_years_exception': fold['min_years_exception'],
                               'h3_label': chosen['h3_label'], 'h3_decay': chosen['decay'],
                               'h3_span': float(2 / (1 - chosen['decay']) - 1) if np.isfinite(chosen['decay']) else np.nan,
                               'ridge_lambda': np.nan if chosen['ridge_lambda'] is None else chosen['ridge_lambda'],
                               'ties': chosen['ties'], 'inner_ic': chosen['inner_ic'],
                               'train_ic': mean_ic(train_ic), 'n_train': chosen['n_train'],
                               'delta': np.nan if f['delta'] is None else f['delta'],
                               'huber_sigma': chosen.get('huber_sigma', np.nan),
                               'huber_sigma_converged': chosen.get('huber_sigma_converged'),
                               'outlier_share': f.get('outlier_share', np.nan), 'huber_iterations': f.get('iterations', np.nan),
                               'intercept': f['intercept'], **{'w_' + c: w for c, w in zip(cols, f['coef'])}})
            inner.append(pd.DataFrame(table).assign(**tag))
            timings.append({'part': 'fold', 'fold': fold['fold'], 'model': model, 'target': target, 'loss': loss,
                            'input_set': set_name, 'seconds': time.perf_counter() - t0})
        elapsed = time.perf_counter() - t_fold
        _progress(f"  [Fold {fold['fold']:>2}] 조합 {i_set:>2}/{n_sets} ({set_name}) 완료, 경과 {elapsed:.0f}초, "
                  f"남은 예상 {elapsed / i_set * (n_sets - i_set):.0f}초")
    return {'fold': fold, 'selections': selections, 'inner': inner, 'timings': timings, 'preds': preds,
            'seconds': time.perf_counter() - t_fold}


def run_linear(prepared: dict, config, *, inner_scope: str = 'common', folds: list[dict] | None = None,
               settings=None, sets: dict | None = None, workers: int = 1, log=print) -> dict:
    """linear_model 전체(또는 일부 설정)를 실행한다. 반환: val_keys, predictions(설정별 배열), selections, inner_ic, stage1, timings.

    workers>1이면 Fold를 프로세스로 나눠 계산한다(난수 없음, 결과는 Fold 순서로 합쳐 같다).
    """
    if inner_scope not in INNER_SCOPES:
        raise ValueError('inner_scope_unknown')
    data = prepared['data'].copy()
    labels = prepared['sigma_labels']
    decay_of = _decay_of(prepared)
    folds = plan_folds(config['MODEL_GRID']) if folds is None else folds
    settings = linear_settings() if settings is None else list(settings)
    sets = input_sets() if sets is None else sets
    val_keys, offsets = validation_keys(data, folds)
    stage1, timings, index_cols = [], [], {}
    specs = index_specs(prepared)
    used = [n for n in specs if any(n in cols for cols in sets.values())]
    if used:
        t_all = time.perf_counter()
        years = index_years(data)
        for model, target, loss in settings:
            for name in used:
                t0 = time.perf_counter()
                col = index_column(name, (model, target, loss))
                data[col], table = build_index(data, name, specs[name], model=model, target=target, loss=loss,
                                               inner_scope=inner_scope, decay_of=decay_of, years=years)
                index_cols.setdefault((model, target, loss), {})[name] = col
                stage1.append(table)
                timing = {'part': 'stage1', 'fold': None, 'model': model, 'target': target, 'loss': loss,
                          'seconds': time.perf_counter() - t0}
                timings.append(timing if name == 'h1_index' else {**timing, 'index': name})
        log(f'지수형 ① 완료: {len(settings)}개 설정, {time.perf_counter() - t_all:.1f}초')
    payload = {'data': data, 'labels': labels, 'decay_of': decay_of, 'settings': settings, 'sets': sets,
               'index_cols': index_cols, 'inner_scope': inner_scope}
    done = execute_folds(_run_fold, payload, folds, workers=workers, log=log)
    return {**assemble_folds(done, folds, val_keys, offsets, timings), 'stage1': _concat(stage1), 'inner_scope': inner_scope,
            'sets': dict(sets), 'settings': list(settings)}


# ---------- 선형·부스팅 공통 실행 뼈대 ----------

def _decay_of(prepared: dict) -> dict:
    return dict(zip(prepared['lambda_candidates'].label, prepared['lambda_candidates'].decay))


def _concat(frames: list) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validation_keys(data: pd.DataFrame, folds: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    """검증 행(바깥 검증 연도들의 수익률 있는 전 종목)은 설정과 무관하게 같다 → 키는 한 번, 예측은 설정별 배열."""
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    predict = data.predict.to_numpy(bool)
    fold_rows = {f['fold']: np.flatnonzero(predict & (year == f['valid_year'])) for f in folds}
    val_pos = np.concatenate([fold_rows[f['fold']] for f in folds])
    offsets = np.cumsum([0] + [len(fold_rows[f['fold']]) for f in folds])
    val_keys = pd.DataFrame({'stock_id': data.stock_id.to_numpy()[val_pos], 'target_month': data.target_month.to_numpy()[val_pos],
                             'fold': np.repeat([f['fold'] for f in folds], np.diff(offsets)),
                             'y': data.y.to_numpy(float)[val_pos], 'common': train[val_pos], 'row': val_pos})
    return val_keys, offsets


def execute_folds(fold_fn, payload: dict, folds: list[dict], *, workers: int = 1, log=print) -> dict:
    """Fold마다 fold_fn을 실행한다. workers>1이면 프로세스로 나눈다(작업 프로세스의 진행 줄은 큐로 전달). 반환: {fold 번호: 결과}."""
    done = {}
    report = lambda r: log(f"Fold {r['fold']['fold']:>2} (검증 {r['fold']['valid_year']}) 완료: {r['seconds']:.1f}초")
    if workers > 1 and len(folds) > 1:
        import multiprocessing
        import queue as queue_mod
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        order = sorted(folds, key=lambda f: -f['valid_year'])          # 큰 Fold부터 넣어 마지막 대기를 줄임
        with multiprocessing.Manager() as manager:
            messages = manager.Queue()

            def drain():
                while True:
                    try:
                        log(messages.get_nowait())
                    except queue_mod.Empty:
                        return

            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                     initargs=({**payload, 'queue': messages},)) as pool:
                pending = {pool.submit(fold_fn, f) for f in order}
                while pending:
                    finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    drain()
                    for fut in finished:
                        r = fut.result()
                        done[r['fold']['fold']] = r
                        report(r)
            drain()
    else:
        _init_worker({**payload, 'log': log})
        try:
            for f in folds:
                r = fold_fn(f)
                done[f['fold']] = r
                report(r)
        finally:
            _WORKER.clear()
    return done


def assemble_folds(done: dict, folds: list[dict], val_keys: pd.DataFrame, offsets: np.ndarray, timings: list) -> dict:
    """Fold 결과를 Fold 순서로 합친다: 설정별 예측 배열, 선택 표, 내부 검증 표, 시간(부스팅은 곡선도)."""
    preds, selections, inner, curves = {}, [], [], []
    timings = list(timings)
    for k, f in enumerate(folds):
        r = done[f['fold']]
        for cid, yhat in r['preds'].items():
            preds.setdefault(cid, np.full(len(val_keys), np.nan))[offsets[k]:offsets[k + 1]] = yhat
        selections += r['selections']
        inner += r['inner']
        timings += r['timings']
        curves += r.get('curves', [])
    out = {'val_keys': val_keys, 'predictions': preds, 'selections': pd.DataFrame(selections),
           'inner_ic': pd.concat(inner, ignore_index=True), 'timings': pd.DataFrame(timings)}
    if curves:
        out['curves'] = curves
    return out


def merge_results(parts: list[dict]) -> dict:
    """셀마다 따로 계산한 결과(같은 세션 메모리)를 하나로 합친다. 예측 배열이 같은 행 순서여야 하므로 검증 행만 같은지 본다
    (다르면 예측이 어긋나 IC가 틀림). 종류·단계 등 부스팅 정보는 첫 셀 것을 쓴다."""
    first = parts[0]
    preds = {}
    for p in parts:
        if not p['val_keys'].equals(first['val_keys']):
            raise ValueError('parts_differ_in_validation_rows')
        preds.update(p['predictions'])
    cat = lambda key: _concat([p[key] for p in parts])
    out = {'val_keys': first['val_keys'], 'predictions': preds, 'selections': cat('selections'), 'inner_ic': cat('inner_ic'),
           'stage1': cat('stage1'), 'timings': cat('timings'), 'inner_scope': first['inner_scope']}
    if any('curves' in p for p in parts):
        out['curves'] = [c for p in parts for c in p.get('curves', [])]
    if any('params' in p for p in parts):
        out['params'] = {k: v for p in parts for k, v in p.get('params', {}).items()}
    for key in ('kind', 'phase', 'device', 'followup'):
        if key in first:
            out[key] = first[key]
    return out


# ---------- 부스팅(boosting_model) ----------

def boosting_labels(prepared: dict) -> list[str]:
    """부스팅의 H3 λ 후보 이름(span 12·36·60개월). 입력 준비에 셋 다 있어야 한다."""
    labels = [f'span{n}' for n in boosting.H3_SPANS]
    missing = [x for x in labels if x not in prepared['sigma_labels']]
    if missing:
        raise ValueError('boosting_h3_labels_missing:' + ','.join(missing))
    return labels


def boosting_input_sets() -> dict:
    """부스팅 입력 4개(주 결과): [σ], [Size, B/M, Mom, σ], [H1 지수], [H1 지수, σ]. 부분집합 비교군은 선형에서만."""
    sets = input_sets()
    return {k: sets[k] for k in BOOSTING_SETS}


def _boost_tag(setting, input_set: str | None = None, overrides=None) -> dict:
    kind, target, loss, depth, lr = setting
    tag = {'family': 'boosting', 'model': kind, 'target': target, 'loss': loss, 'hparams': boosting.hparam_tag(depth, lr, overrides)}
    if input_set is not None:
        tag['input_set'] = input_set
    return tag


def choose_boosting(fits: list[dict]) -> tuple[dict, int, int]:
    """(H3 λ × 트리 수) 모든 쌍 중 내부 검증 IC 최고 한 쌍. 정확히 같으면 트리 적은 쪽 → H3 긴 기간(λ 큰 쪽) [tie_rules].
    반환: (고른 학습, 고른 트리 수, 최고값 동률 수)."""
    alive = [f for f in fits if np.isfinite(f['curve']).any()]
    if not alive:
        raise ValueError('all_candidates_failed')
    best = max(float(np.nanmax(f['curve'])) for f in alive)
    cands, ties = [], 0
    for f in alive:
        hit = np.flatnonzero(f['curve'] == best)
        ties += len(hit)
        if len(hit):
            cands.append((int(hit[0]) + 1, f))
    trees, chosen = max(cands, key=lambda c: (-c[0], c[1]['decay'] if np.isfinite(c[1]['decay']) else -1.0))
    return chosen, trees, ties


def _curve_record(tag: dict, fitted: dict, *, part: str, fold=None, index_year=None, h3_label=None) -> dict:
    return {**tag, 'part': part, 'fold': fold, 'index_year': index_year, 'h3_label': h3_label,
            'best_trees': fitted['best_trees'], 'seconds': fitted['seconds'],
            'inner': fitted['inner_curve'].astype(np.float32), 'train': fitted['train_curve'].astype(np.float32)}


def select_and_fit_boosting(ctx: FoldContext, setting, *, input_cols, sigma_labels, decay_of, index_cols=None,
                            device='cpu', n_threads=None, verbose_every=None, say=None, prefix='', overrides=None):
    """한 Fold·한 부스팅 설정. σ가 있으면 H3 λ 후보마다 1000트리를 학습하고, (λ × 트리 수) 중 내부 검증 IC 최고 한 쌍을 고른다.
    모든 학습이 같은 학습 행(ctx.fit_rows)을 쓰고 내부 검증은 원수익률로 평가한다. Huber② σ는 같은 행·입력·목표값의
    OLS HuberRegressor(ctx.sigma2, 선형과 같은 값). 반환: (선택 dict, 후보별 표, 학습 목록)."""
    kind, target, loss, depth, lr = setting
    labels = sigma_labels if 'sigma' in input_cols else [None]
    y_fit = ctx.y(target)
    m_inner = ctx.data.target_month.to_numpy()[ctx.inner_rows]
    fits, table = [], []
    for lab in labels:
        cols = _columns(input_cols, lab, index_cols)
        sig = None
        if loss == 'huber_q999':
            delta = ctx.q999(target)
        elif loss == 'huber_sigma135':
            sig = ctx.sigma2(cols, target)
            delta = sig['delta']
        else:
            delta = None
        decay = decay_of.get(lab, np.nan)
        try:
            f = boosting.fit_boosting(kind, ctx.X(cols, 'fit'), y_fit, ctx.m_fit, target=target, loss=loss, depth=depth,
                                      learning_rate=lr, delta=delta, inner_X=ctx.X(cols, 'inner'), inner_months=m_inner,
                                      inner_score=lambda p: ctx.scorer.score(p)[0],
                                      train_score=lambda p: ctx.train_scorer.score(p)[0],
                                      device=device, n_threads=n_threads, verbose_every=verbose_every, log=say,
                                      overrides=overrides)
        except ValueError as error:
            table.append({'h3_label': lab, 'decay': decay, 'best_trees': np.nan, 'inner_ic': np.nan, 'inner_ic_tree1000': np.nan,
                          'failure': str(error), 'seconds': np.nan, 'huber_sigma': sig['sigma'] if sig else np.nan})
            continue
        fits.append({'h3_label': lab, 'decay': decay, 'fitted': f, 'columns': cols, 'curve': f['inner_curve'],
                     'huber_sigma': sig['sigma'] if sig else np.nan, 'huber_sigma_converged': sig['converged'] if sig else None})
        table.append({'h3_label': lab, 'decay': decay, 'best_trees': f['best_trees'], 'inner_ic': f['best_inner_ic'],
                      'inner_ic_tree1000': float(f['inner_curve'][-1]), 'failure': '', 'seconds': f['seconds'],
                      'huber_sigma': sig['sigma'] if sig else np.nan})
        if say:
            k = -(-f['best_trees'] // f['train_every'])                   # 고른 트리 수 이상 가장 가까운 기록 지점
            say(f"    {prefix}{' ' + lab if lab else ''}: 트리 {f['best_trees']:>4}/{boosting.MAX_TREES}, "
                f"학습 IC {f['train_curve'][k - 1]:.4f}(트리 {k * f['train_every']}), 내부 검증 IC {f['best_inner_ic']:.4f}, "
                f"{f['seconds']:.1f}초")
    chosen, trees, ties = choose_boosting(fits)
    return ({**chosen, 'trees': trees, 'inner_ic': float(chosen['curve'][trees - 1]), 'ties': ties,
             'n_train': ctx.n_train}, table, fits)


def build_index_boosting(data, name: str, spec: dict, setting, *, inner_scope, decay_of, device='cpu', n_threads=None, log=None,
                         years=None, overrides=None):
    """지수형 ① 일반형(부스팅). build_index와 같은 행·연도·변환 규칙에 ②와 같은 종류·목표값·손실·깊이·학습률, 트리 수는 Y−1
    내부 검증. centered_raw의 중심값 = 고른 트리 수로 입력 0을 예측한 값. 반환: (지수 배열, 선택 표, 곡선 목록)."""
    d = _index_frame(data, name)
    parts = list(spec['parts'])
    years = index_years(data) if years is None else years
    raw_index = np.full(len(d), np.nan)
    year = d.year.to_numpy()
    train = d.train.to_numpy(bool)
    scope = d.predict.to_numpy(bool) if inner_scope == 'full' else train
    member = _index_member(d, name, parts)
    tag = _boost_tag(setting, overrides=overrides)
    ctag = tag if name == 'h1_index' else {**tag, 'index': name}
    label = '①' if name == 'h1_index' else f'① {name}'
    rows, curves = [], []
    for Y in years:
        target_rows = member & (year == Y)
        ctx = FoldContext(d, train & (year >= INDEX_TRAIN_START) & (year <= Y - 2), scope & (year == Y - 1), target_rows)
        chosen, _, fits = select_and_fit_boosting(ctx, setting, input_cols=parts, sigma_labels=None, decay_of=decay_of,
                                                  device=device, n_threads=n_threads, say=log, prefix=f'[{label} {Y}] ',
                                                  overrides=overrides)
        f = chosen['fitted']
        pred = boosting.predict_boosting(f, _design(d, chosen['columns'], target_rows, fill=False), chosen['trees'])
        center = 0.0
        if spec['transform'] == 'centered_raw':
            center = float(boosting.predict_boosting(f, np.zeros((1, len(parts))), chosen['trees'])[0])
            pred = pred - center
        raw_index[target_rows] = pred
        gain = boosting.feature_gain(f, len(parts), chosen['trees'])
        share = gain / gain.sum() if gain.sum() > 0 else gain
        row = {'index_year': Y, **tag, 'stage1_train_years': f'{INDEX_TRAIN_START}~{Y-2}', 'stage1_inner_valid_year': Y - 1,
               'trees': chosen['trees'], 'inner_ic': chosen['inner_ic'], 'ties': chosen['ties'], 'n_train': chosen['n_train'],
               'delta': np.nan if f['delta'] is None else f['delta'], 'huber_sigma': chosen['huber_sigma'],
               **{'gain_' + p: s for p, s in zip(parts, share)}}
        if name != 'h1_index':
            row.update(index=name, center=center)
        rows.append(row)
        curves += [_curve_record(ctag, x['fitted'], part='stage1', index_year=Y) for x in fits]
    return _finish_index(raw_index, d, spec['transform']), pd.DataFrame(rows), curves


def build_h1_index_boosting(data, setting, *, inner_scope, decay_of, device='cpu', n_threads=None, log=None,
                            years=range(TRAIN_START, 2022), overrides=None):
    """지수형 ① H1 지수(부스팅). 반환: (순위 지수 배열, 선택 표, 곡선 목록)."""
    return build_index_boosting(data, 'h1_index', {'parts': list(H1_PARTS), 'transform': 'rank'}, setting, inner_scope=inner_scope,
                                decay_of=decay_of, device=device, n_threads=n_threads, log=log, years=years,
                                overrides=overrides)


def _run_fold_boosting(fold: dict) -> dict:
    """한 Fold의 모든 (입력 조합 × 부스팅 설정). 파일을 읽거나 쓰지 않는다."""
    p = _WORKER
    data = p['data']
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    predict = data.predict.to_numpy(bool)
    scope = predict if p['inner_scope'] == 'full' else train
    t_fold = time.perf_counter()
    lo, hi = fold['train_years']
    ctx = FoldContext(data, train & (year >= lo) & (year <= hi), scope & (year == fold['inner_valid_year']),
                      predict & (year == fold['valid_year']))
    y_all = data.y.to_numpy(float)
    selections, inner, timings, preds, curves, params = [], [], [], {}, [], {}
    n_jobs, i_job = len(p['sets']) * len(p['settings']), 0
    for set_name, cols in p['sets'].items():
        for setting in p['settings']:
            i_job += 1
            t0 = time.perf_counter()
            tag = {**_boost_tag(setting, set_name, p.get('overrides')), 'fold': fold['fold']}
            prefix = f"[Fold {fold['fold']:>2}] {setting[0]} {setting[1]} {setting[2]} {tag['hparams']} {set_name}"
            chosen, table, fits = select_and_fit_boosting(
                ctx, setting, input_cols=cols, sigma_labels=p['labels'], decay_of=p['decay_of'],
                index_cols=p['index_cols'].get(setting), device=p['device'], n_threads=p['n_threads'],
                say=_progress, prefix=prefix, overrides=p.get('overrides'))
            f, trees = chosen['fitted'], chosen['trees']
            preds[config_id(tag)] = boosting.predict_boosting(f, ctx.X(chosen['columns'], 'out'), trees)
            train_ic = monthly_ic(ctx.m_fit, boosting.predict_boosting(f, ctx.X(chosen['columns'], 'fit'), trees), y_all[ctx.fit_rows])
            gain = boosting.feature_gain(f, len(cols), trees)
            share = gain / gain.sum() if gain.sum() > 0 else gain
            selections.append({**tag, 'valid_year': fold['valid_year'], 'inner_valid_year': fold['inner_valid_year'],
                               'inner_train_years': fold['inner_train_years'], 'min_years_exception': fold['min_years_exception'],
                               'h3_label': chosen['h3_label'], 'h3_decay': chosen['decay'],
                               'h3_span': float(2 / (1 - chosen['decay']) - 1) if np.isfinite(chosen['decay']) else np.nan,
                               'trees': trees, 'ties': chosen['ties'], 'inner_ic': chosen['inner_ic'],
                               'train_ic': mean_ic(train_ic), 'n_train': chosen['n_train'],
                               'delta': np.nan if f['delta'] is None else f['delta'], 'huber_sigma': chosen['huber_sigma'],
                               'huber_sigma_converged': chosen['huber_sigma_converged'],
                               **{'gain_' + c: s for c, s in zip(cols, share)}})
            inner.append(pd.DataFrame(table).assign(**tag))
            curves += [_curve_record(tag, x['fitted'], part='fold', fold=fold['fold'], h3_label=x['h3_label']) for x in fits]
            params.setdefault('|'.join(map(str, setting)), f['params'])
            seconds = time.perf_counter() - t0
            timings.append({'part': 'fold', 'fold': fold['fold'], 'model': setting[0], 'target': setting[1], 'loss': setting[2],
                            'hparams': tag['hparams'], 'input_set': set_name, 'seconds': seconds})
            elapsed = time.perf_counter() - t_fold
            _progress(f"  {prefix} 선택: {chosen['h3_label'] or '-'} 트리 {trees}, 내부 IC {chosen['inner_ic']:.4f}, "
                      f"학습 IC {mean_ic(train_ic):.4f}, {seconds:.0f}초 ({i_job}/{n_jobs}, 남은 예상 {elapsed / i_job * (n_jobs - i_job):.0f}초)")
    return {'fold': fold, 'selections': selections, 'inner': inner, 'timings': timings, 'preds': preds, 'curves': curves,
            'params': params, 'seconds': time.perf_counter() - t_fold}


def _kind_settings(kind: str, settings, followup: bool = False) -> list[tuple]:
    """설정 목록 (종류, 목표값, 손실, 깊이, 학습률). 안 주면 그 종류의 기본 목록, 주면 그대로(값 범위 검사 없음, 2026-11-15 결정 마)."""
    return boosting.settings(kind, followup=followup) if settings is None else [tuple(s) for s in settings]


def _followup_suffix(kind: str) -> str:
    """후속 튜닝 결과 폴더 꼬리표: 깊이 4면 '_d4'."""
    return '_d' + ''.join(str(d) for d in boosting.FOLLOWUP_DEPTHS[kind])


def run_boosting_stage1(prepared: dict, config, kind: str, *, settings=None, inner_scope: str = 'common', device: str | None = None,
                        n_threads: int | None = None, followup: bool = False, overrides: dict | None = None,
                        indexes=None, log=print) -> dict:
    """지수형 ①: 설정마다 ② 학습 첫해~2021년 지수를 만든다(H1 지수 = 순위 [−1, 1], 붙인 지수는 그 규칙). indexes = 만들 지수 이름
    목록(기본: 쓸 수 있는 모든 지수 — 특징을 안 붙였으면 H1만). 결과는 같은 세션 메모리로 run_boosting(phase='index')에 넘긴다
    (파일로 저장했다가 다시 읽지 않음). 반환: kind, settings, inner_scope, n_rows, index({설정: {지수: 배열}}), stage1 표, curves, timings."""
    if inner_scope not in INNER_SCOPES:
        raise ValueError('inner_scope_unknown')
    data = prepared['data']
    settings = _kind_settings(kind, settings, followup)
    device = device or boosting.default_device(kind)
    decay_of = _decay_of(prepared)
    specs = index_specs(prepared)
    names = list(specs) if indexes is None else list(indexes)
    years = index_years(data)
    index, tables, curves, timings = {}, [], [], []
    for setting in settings:
        index[setting] = {}
        for name in names:
            t0 = time.perf_counter()
            index[setting][name], table, c = build_index_boosting(data, name, specs[name], setting, inner_scope=inner_scope,
                                                                  decay_of=decay_of, device=device, n_threads=n_threads,
                                                                  years=years, overrides=overrides)
            tables.append(table)
            curves += c
            seconds = time.perf_counter() - t0
            timing = {'part': 'stage1', 'fold': None, 'model': kind, 'target': setting[1], 'loss': setting[2],
                      'hparams': boosting.hparam_tag(*setting[3:], overrides), 'seconds': seconds}
            timings.append(timing if name == 'h1_index' else {**timing, 'index': name})
            log(f"지수형 ①{'' if name == 'h1_index' else ' ' + name} {setting[1]} {setting[2]} "
                f"{boosting.hparam_tag(*setting[3:], overrides)}: {seconds:.0f}초, 고른 트리 {table.trees.min()}~{table.trees.max()}")
    return {'kind': kind, 'settings': settings, 'inner_scope': inner_scope, 'device': device, 'followup': followup,
            'overrides': dict(overrides or {}), 'indexes': names,
            'n_rows': len(data), 'index': index,
            'stage1': _concat(tables), 'curves': curves, 'timings': timings}


def run_boosting(prepared: dict, config, kind: str, *, phase: str, stage1: dict | None = None, settings=None,
                 sets: dict | None = None, folds: list[dict] | None = None, inner_scope: str = 'common', device: str | None = None,
                 n_threads: int | None = None, workers: int = 1, followup: bool = False,
                 overrides: dict | None = None, log=print) -> dict:
    """한 부스팅 종류·한 단계(전체 또는 일부 설정)를 실행한다. 반환 형식은 run_linear와 같고 curves·params가 더 있다.

    phase='individual': [Size, B/M, Mom, σ]. phase='index': [σ]·[H1 지수]·[H1 지수, σ], 같은 세션의 run_boosting_stage1 결과가
    필요하다(같은 설정의 지수). followup=True면 후속 튜닝 설정(boosting.FOLLOWUP_DEPTHS)이 기본 설정이 되고 폴더 끝에 _d4가 붙는다.
    settings·sets·overrides(라이브러리 설정 덮어쓰기)는 자유롭게 준다(검사 없음). workers>1이면 Fold를 프로세스로 나눈다
    (n_threads를 함께 줄일 것). 난수 없음.
    """
    if inner_scope not in INNER_SCOPES:
        raise ValueError('inner_scope_unknown')
    data = prepared['data'].copy()
    decay_of = _decay_of(prepared)
    folds = plan_folds(config['MODEL_GRID']) if folds is None else folds
    settings = _kind_settings(kind, settings, followup)
    device = device or boosting.default_device(kind)
    sets = {k: v for k, v in boosting_input_sets().items() if k in BOOSTING_PHASES[phase]} if sets is None else sets
    specs = index_specs(prepared)
    val_keys, offsets = validation_keys(data, folds)
    index_cols, stage1_rows, curves, timings = {}, pd.DataFrame(), [], []
    used = [n for n in specs if any(n in cols for cols in sets.values())]
    if used:
        if stage1 is None:
            raise ValueError('stage1_result_required')
        for setting in settings:
            for name in used:
                col = index_column(name, setting)
                data[col] = stage1['index'][setting][name]
                index_cols.setdefault(setting, {})[name] = col
        hp = {boosting.hparam_tag(*s[3:], overrides) + '|' + s[1] + '|' + s[2] for s in settings}
        mine = lambda r: f"{r['hparams']}|{r['target']}|{r['loss']}" in hp
        st = stage1['stage1']
        stage1_rows = st[[mine(r) for r in st.to_dict('records')]].reset_index(drop=True)
        curves = [c for c in stage1['curves'] if mine(c)]
        timings = [t for t in stage1['timings'] if mine(t)]
    payload = {'data': data, 'labels': boosting_labels(prepared), 'decay_of': decay_of, 'settings': settings, 'sets': sets,
               'index_cols': index_cols, 'inner_scope': inner_scope, 'device': device, 'n_threads': n_threads,
               'overrides': dict(overrides or {})}
    done = execute_folds(_run_fold_boosting, payload, folds, workers=workers, log=log)
    out = assemble_folds(done, folds, val_keys, offsets, timings)
    out['curves'] = curves + out.get('curves', [])
    out['params'] = {k: v for f in folds for k, v in done[f['fold']]['params'].items()}
    return {**out, 'stage1': stage1_rows, 'inner_scope': inner_scope, 'kind': kind, 'phase': phase, 'device': device,
            'followup': followup, 'overrides': dict(overrides or {}), 'settings': list(settings), 'sets': dict(sets)}


def curves_frame(curves: list[dict]) -> pd.DataFrame:
    """학습별 곡선 → 넓은 표: 키 열 + inner_0001~inner_1000(트리별 내부 검증 IC) + train_0010~train_1000(10트리마다 학습 IC)."""
    if not curves:
        return pd.DataFrame()
    keys = pd.DataFrame([{k: v for k, v in c.items() if k not in ('inner', 'train')} for c in curves])
    inner = np.vstack([c['inner'] for c in curves])
    train = np.vstack([c['train'] for c in curves])
    every = boosting.MAX_TREES // train.shape[1]
    wide = pd.DataFrame(np.hstack([inner, train]), columns=[f'inner_{i:04d}' for i in range(1, inner.shape[1] + 1)]
                        + [f'train_{every * i:04d}' for i in range(1, train.shape[1] + 1)])
    return pd.concat([keys, wide], axis=1)


def boosting_part_summary(result: dict) -> pd.DataFrame:
    """셀 하나의 진행 확인용 요약: 입력 × 목표값 × 손실 × 깊이·학습률별 바깥 검증 평균 IC와 Fold 평균 학습 IC·고른 트리 수."""
    ic = outer_monthly_ic(result['val_keys'], result['predictions'])
    keys = ['input_set', 'target', 'loss', 'hparams']
    valid = ic.groupby(keys + ['scope']).ic.mean().unstack('scope')
    sel = result['selections'].groupby(keys)
    out = pd.DataFrame({'valid_ic_full': valid['full'], 'valid_ic_common': valid['common'],
                        'train_ic': sel.train_ic.mean(), 'trees_median': sel.trees.median()})
    out['train_minus_valid'] = out.train_ic - out.valid_ic_full
    return out.reset_index()


def finalize_boosting(root: Path, config, prepared: dict, result: dict, *, started: datetime, out_root: Path | None = None,
                      log=print, author: str, experiment: str) -> dict:
    """부스팅 한 종류·한 단계의 저장: 보고 표 + 곡선(parquet). 폴더 <시각>_<실험>_<종류>_<단계>(후속 튜닝은 끝에 _d4)."""
    suffix = f"{result['kind']}_{result['phase']}" + (_followup_suffix(result['kind']) if result.get('followup') else '')
    return save_run(root, config, prepared, result, folder_suffix=suffix, started=started, out_root=out_root, author=author,
                    experiment=experiment, extra_tables={'curves': (curves_frame(result['curves']), 'parquet')}, log=log)


def measure_boosting_fit(prepared: dict, config, setting, *, fold_no: int = 16, input_set: str = 'ind:size+bm+mom+sigma',
                         h3_label: str = 'span36', device: str | None = None, n_threads: int | None = None,
                         inner_scope: str = 'common', verbose_every: int = 100, log=print) -> dict:
    """시간 측정용 학습 1회(1000트리, 100트리마다 자세한 줄). 지수형 입력은 쓰지 않는다(①이 필요해서).
    반환: 시간, 고른 트리 수, 내부 검증 IC, 학습 행 수, 내부 검증 예측(장치 간 비교용)."""
    fold = next(f for f in plan_folds(config['MODEL_GRID']) if f['fold'] == fold_no)
    cols_in = input_sets()[input_set]
    if 'h1_index' in cols_in:
        raise ValueError('measure_without_index_inputs')
    data = prepared['data']
    year, train = data.year.to_numpy(), data.train.to_numpy(bool)
    scope = data.predict.to_numpy(bool) if inner_scope == 'full' else train
    lo, hi = fold['train_years']
    ctx = FoldContext(data, train & (year >= lo) & (year <= hi), scope & (year == fold['inner_valid_year']))
    kind, target, loss, depth, lr = setting
    device = device or boosting.default_device(kind)
    cols = _columns(cols_in, h3_label, None)
    delta = ctx.q999(target) if loss == 'huber_q999' else ctx.sigma2(cols, target)['delta'] if loss == 'huber_sigma135' else None
    log(f"[측정] {kind} {target} {loss} {boosting.hparam_tag(depth, lr)} {input_set} {h3_label}, Fold {fold_no}, "
        f"학습 {ctx.n_train:,}행, 장치 {device}")
    f = boosting.fit_boosting(kind, ctx.X(cols, 'fit'), ctx.y(target), ctx.m_fit, target=target, loss=loss, depth=depth,
                              learning_rate=lr, delta=delta, inner_X=ctx.X(cols, 'inner'),
                              inner_months=data.target_month.to_numpy()[ctx.inner_rows],
                              inner_score=lambda p: ctx.scorer.score(p)[0], train_score=lambda p: ctx.train_scorer.score(p)[0],
                              device=device, n_threads=n_threads, verbose_every=verbose_every, log=log)
    log(f"[측정] 끝: {f['seconds']:.1f}초, 트리 {f['best_trees']}, 내부 검증 IC {f['best_inner_ic']:.4f}")
    return {'seconds': f['seconds'], 'best_trees': f['best_trees'], 'best_inner_ic': f['best_inner_ic'], 'n_train': ctx.n_train,
            'fold': fold_no, 'device': device, 'inner_curve': f['inner_curve'],
            'inner_pred': boosting.predict_boosting(f, ctx.X(cols, 'inner'), boosting.MAX_TREES)}


def estimate_boosting_minutes(prepared: dict, kind: str, seconds_by_fold: dict, *, sets=BOOSTING_SETS, n_settings=None) -> dict:
    """한 종류 전체 예상 시간. seconds_by_fold = {Fold 번호: 학습 1회 초}(Fold 1·16 실측)를 학습 행 수로 보간한다.
    학습 횟수: Fold·설정마다 입력 조합별(σ 있으면 λ 10개, 없으면 1개), 지수형 ①은 설정마다 2003~2021년 19회."""
    data = prepared['data']
    train = data.train.to_numpy(bool)
    year = data.year.to_numpy()
    folds = plan_folds({'first_validation_year': 2006, 'last_validation_year': 2021, 'min_inner_train_years': 5,
                        'min_inner_train_exceptions': [1, 2, 3]})
    (f_a, s_a), (f_b, s_b) = sorted(seconds_by_fold.items())
    n_of = lambda lo, hi: int((train & (year >= lo) & (year <= hi)).sum())
    n_a, n_b = (n_of(*next(f for f in folds if f['fold'] == k)['train_years']) for k in (f_a, f_b))
    per_fit = lambda n: s_a + (n - n_a) / (n_b - n_a) * (s_b - s_a)
    all_sets = input_sets()
    fits_per_fold = sum(len(boosting.H3_SPANS) if 'sigma' in all_sets[s] else 1 for s in sets)
    n_settings = len(boosting.settings(kind)) if n_settings is None else n_settings
    folds_sec = sum(per_fit(n_of(*f['train_years'])) * fits_per_fold for f in folds) * n_settings
    stage1_sec = (sum(per_fit(n_of(INDEX_TRAIN_START, Y - 2)) for Y in range(TRAIN_START, 2022)) * n_settings
                  if any('h1_index' in all_sets[s] for s in sets) else 0.0)
    return {'kind': kind, 'settings': n_settings, 'fits_per_fold_setting': fits_per_fold,
            'folds_hours': folds_sec / 3600, 'stage1_hours': stage1_sec / 3600, 'total_hours': (folds_sec + stage1_sec) / 3600}


# ---------- 5) 평가·보고·종료 조건 ----------

CONFIG_KEYS = ['family', 'model', 'target', 'loss', 'input_set']


def config_id(tag: dict) -> str:
    """선형: family|model|target|loss|input_set. 부스팅은 끝에 |d<깊이>_lr<학습률>을 붙인다(선형 ID는 그대로)."""
    base = '|'.join(str(tag[k]) for k in CONFIG_KEYS)
    hp = tag.get('hparams')
    return base + '|' + hp if isinstance(hp, str) and hp else base


def split_config_id(cid: str) -> dict:
    return dict(zip(CONFIG_KEYS + ['hparams'], cid.split('|')))


def _selection_ids(sel: pd.DataFrame) -> list[str]:
    cols = CONFIG_KEYS + (['hparams'] if 'hparams' in sel else [])
    return [config_id(r) for r in sel[cols].to_dict('records')]


def _scopes(val_keys: pd.DataFrame):
    full = np.ones(len(val_keys), dtype=bool)
    return (('full', full), ('common', val_keys.common.to_numpy(bool)))


def outer_monthly_ic(val_keys: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """설정 × 범위(전체 종목·공통 표본) × 목표월 IC. 평가는 원수익률.
    한 달 예측이 모두 같으면(reason constant_score) 순위를 전혀 구분하지 못한 것이므로 IC = 0으로 센다
    (2026-11-15 결정 A: 그 달을 빼면 실패한 달이 평균에서 사라져 부풀려지고 모델마다 평가 달이 달라짐. 관례적 선택).
    내부 검증 IC(선택)에는 적용하지 않는다."""
    rows = []
    months, y = val_keys.target_month.to_numpy(), val_keys.y.to_numpy(float)
    for cid, yhat in predictions.items():
        for scope, mask in _scopes(val_keys):
            t = monthly_ic(months[mask], yhat[mask], y[mask]).reset_index()
            t.loc[t.reason.eq('constant_score'), 'ic'] = 0.0
            rows.append(t.assign(config=cid, scope=scope))
    ic = pd.concat(rows, ignore_index=True)
    parts = pd.DataFrame([split_config_id(c) for c in ic.config], index=ic.index)
    return pd.concat([parts, ic], axis=1)


def metrics(s: pd.Series) -> dict:
    s = s.astype(float)
    sd = float(s.std(ddof=1))
    return {'months': int(s.notna().sum()), 'mean_ic': float(s.mean()), 'ic_sd': sd,
            'icir': float(s.mean() / sd) if sd > 0 else np.nan, 'ic_positive_share': float(s.gt(0).mean())}


def r2_pct(y: np.ndarray, yhat: np.ndarray) -> float:
    """GKX 식 19: 1 − Σ(r − r̂)²/Σr²(평균 차감 없음), 수익률 단위 예측(원값·윈저 목표)만."""
    return float(100 * (1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)))


def report_tables(result: dict, ic: pd.DataFrame, *, input_transform: str = 'rank') -> dict:
    """보고 표. performance_summary에는 IC 개월 수(months)·채점 행 수(scored_rows)·입력 변환·내부 검증 범위를 숫자·글자로만 둔다
    (192개월·133,876행 고정 검사는 하지 않음, 2026-11-15 결정 가)."""
    sel, preds, val_keys = result['selections'], result['predictions'], result['val_keys']
    sel = sel.assign(config=_selection_ids(sel))
    train_by_config = sel.groupby('config').train_ic.mean()
    masks = dict(_scopes(val_keys))
    y = val_keys.y.to_numpy(float)
    perf = []
    for (cid, scope), part in ic.groupby(['config', 'scope'], sort=True):
        s = part.set_index('target_month').ic
        cfg = split_config_id(cid)
        m = masks[scope]
        row = {**cfg, 'input_transform': input_transform, 'inner_scope': result['inner_scope'], 'scope': scope, **metrics(s),
               'scored_rows': int(part.n.sum()), 'train_ic_fold_mean': float(train_by_config.loc[cid]),
               'r2_pct': r2_pct(y[m], preds[cid][m]) if cfg['target'] in ('raw', 'winsor_1_99') else np.nan}
        row['train_minus_valid'] = row['train_ic_fold_mean'] - row['mean_ic']
        year = s.index.str[:4].astype(int)
        row.update({f'mean_ic_{k}': float(s[(year >= a) & (year <= b)].mean()) for k, (a, b) in PERIODS.items()})
        perf.append(row)
    fold_ic = ic[ic.scope.eq('full')].assign(valid_year=lambda f: f.target_month.str[:4].astype(int))
    fold_ic = fold_ic.groupby(['config', 'valid_year']).ic.mean().rename('valid_ic').reset_index()
    fold_table = sel.merge(fold_ic, on=['config', 'valid_year'], how='left', validate='one_to_one')
    fold_table['train_minus_valid'] = fold_table.train_ic - fold_table.valid_ic
    tuning = sel.groupby(['model', 'input_set']).agg(
        fold_settings=('fold', 'size'), ties_over_1=('ties', lambda s: int((s > 1).sum()))).reset_index()
    return {'performance_summary': pd.DataFrame(perf), 'fold_table': fold_table, 'tuning_ties': tuning,
            'monthly_ic': ic, 'inner_validation_ic': result['inner_ic'], 'stage1_h1_index': result['stage1'],
            'timings': result['timings']}


def part_summary(result: dict) -> pd.DataFrame:
    """셀(모델 × 손실) 하나의 요약: 입력 조합 × 목표값별 바깥 검증 평균 IC(전체 종목·공통 표본)와 Fold 평균 학습 IC.
    선택용이 아니라 진행 확인용(최종 표는 finalize_linear)."""
    ic = outer_monthly_ic(result['val_keys'], result['predictions'])
    valid = ic.groupby(['input_set', 'target', 'scope']).ic.mean().unstack('scope')
    train = result['selections'].groupby(['input_set', 'target']).train_ic.mean()
    out = pd.DataFrame({'valid_ic_full': valid['full'], 'valid_ic_common': valid['common'], 'train_ic': train})
    out['train_minus_valid'] = out.train_ic - out.valid_ic_full
    order = list(result.get('sets') or input_sets())
    order += [s for s in out.index.get_level_values('input_set').unique() if s not in order]
    return (out.reset_index().assign(_o=lambda f: f.input_set.map(order.index))
            .sort_values(['_o', 'target']).drop(columns='_o').reset_index(drop=True))


def prediction_frame(result: dict) -> pd.DataFrame:
    """저장용 넓은 표: 검증 행 키 + 설정별 예측 열."""
    keys = result['val_keys'].drop(columns='row')
    return pd.concat([keys, pd.DataFrame(result['predictions'], index=keys.index)], axis=1)


# ---------- 실행 ----------

def _makespan(seconds: list[float], workers: int) -> float:
    """큰 작업부터 가장 한가한 작업자에게 배정했을 때 끝나는 시각(병렬 예상 시간)."""
    load = [0.0] * max(1, workers)
    for s in sorted(seconds, reverse=True):
        i = load.index(min(load))
        load[i] += s
    return max(load)


def estimate_part_seconds(prepared, config, *, model: str, loss: str, inner_scope='common', workers: int = 1, log=print) -> dict:
    """한 셀(모델 × 손실)의 예상 시간. Fold 1·16에서 σ 있는 조합·없는 조합 하나씩 실측해 학습 행 수로 보간하고,
    지수형 ①은 목표값 하나를 실측해 ×3. 병렬이면 Fold 시간을 작업자에 배정해 계산한다."""
    folds = plan_folds(config['MODEL_GRID'])
    sets = input_sets()
    settings = part_settings(model, loss)
    sample = {'ind:size+bm+mom+sigma': sets['ind:size+bm+mom+sigma'], 'ind:size+bm+mom': sets['ind:size+bm+mom']}
    measured = {}
    for fold in (folds[0], folds[-1]):
        r = run_linear(prepared, config, inner_scope=inner_scope, folds=[fold], settings=settings, sets=sample, log=lambda *_: None)
        t = r['timings']
        measured[fold['fold']] = {name: float(t[t.input_set.eq(name)].seconds.sum()) for name in sample}
    data = prepared['data']
    n = {f['fold']: int((data.train & data.year.between(*f['train_years'])).sum()) for f in folds}
    n_sigma = sum('sigma' in c for c in sets.values())
    n_plain = len(sets) - n_sigma
    per_fold = []
    for f in folds:
        w = (n[f['fold']] - n[1]) / (n[16] - n[1])
        per = {k: measured[1][k] + w * (measured[16][k] - measured[1][k]) for k in sample}
        per_fold.append(n_sigma * per['ind:size+bm+mom+sigma'] + n_plain * per['ind:size+bm+mom'])
    decay_of = dict(zip(prepared['lambda_candidates'].label, prepared['lambda_candidates'].decay))
    t0 = time.perf_counter()
    build_h1_index(data, model=model, target='raw', loss=loss, inner_scope=inner_scope, decay_of=decay_of)
    stage1 = (time.perf_counter() - t0) * len(settings)
    folds_wall = _makespan(per_fold, workers)
    est = {'model': model, 'loss': loss, 'workers': workers, 'folds_seconds_serial': sum(per_fold),
           'folds_seconds_wall': folds_wall, 'stage1_seconds': stage1, 'total_minutes': (folds_wall + stage1) / 60,
           'measured_fold1': measured[1], 'measured_fold16': measured[16]}
    log(f"[{model} × {loss}] 예상 약 {est['total_minutes']:.1f}분 (Fold {folds_wall/60:.1f}분, 작업자 {workers}개 + 지수형 ① {stage1/60:.1f}분)")
    return est


def finalize_linear(root: Path, config, prepared: dict, result: dict, *, started: datetime, out_root: Path | None = None,
                    log=print, author: str, experiment: str) -> dict:
    """선형 결과 저장(합친 결과 또는 셀 하나). 폴더 <시각>_<실험>_linear."""
    return save_run(root, config, prepared, result, folder_suffix='linear', started=started, out_root=out_root, author=author,
                    experiment=experiment, log=log)


RESULT_ZIP = 'result.zip'
NOT_SHARED = {'validation_predictions.parquet', RESULT_ZIP}    # 예측값은 커서(선형 356MB) 각자 PC에만


def save_run(root: Path, config, prepared: dict, result: dict, *, folder_suffix: str, started: datetime, author: str,
             experiment: str, out_root: Path | None = None, extra_tables: dict | None = None, log=print) -> dict:
    """선형·부스팅 공통 저장(2026-11-15 결정 가~다): 검사 없이 보고 표·예측값을 새 폴더에 쓰고 공유용 result.zip을 만든다.

    폴더: <저장 위치>/<작성자>/<시각>_<실험>_<folder_suffix>/. 저장 위치는 config PATHS.model_results, Colab처럼 다른 곳은 out_root.
    작성자는 영문·숫자·_·-만(폴더 이름·Git 경로). 실험 이름은 자유(한글이면 Mac에서 깨져 보일 수 있음).
    무엇을 돌렸는지는 표의 열(모델·목표값·손실·hparams·입력 조합·입력 변환)과 폴더 이름으로 본다(run.json 없음).
    extra_tables: {이름: (표, 'csv'|'parquet')}. 붙인 특징이 있으면 feature_coverage.csv도 저장."""
    t0 = time.perf_counter()
    ic = outer_monthly_ic(result['val_keys'], result['predictions'])
    tables = report_tables(result, ic, input_transform=prepared['checks'].get('input_transform', 'rank'))
    folder = result_folder(root, config, author=author, experiment=experiment, started=started, suffix=folder_suffix,
                           out_root=out_root)
    for name, table in tables.items():
        table.to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
    prediction_frame(result).to_parquet(folder / 'validation_predictions.parquet', index=False)
    extra = dict(extra_tables or {})
    if len(prepared.get('feature_coverage', [])):
        extra['feature_coverage'] = (prepared['feature_coverage'], 'csv')
    for name, (table, fmt) in extra.items():
        if fmt == 'parquet':
            table.to_parquet(folder / f'{name}.parquet', index=False)
        else:
            table.to_csv(folder / f'{name}.csv', index=False, encoding='utf-8-sig')
    pack_result(folder)
    log(f"저장 ({time.perf_counter() - t0:.0f}초): {folder}")
    return {'run_dir': str(folder), 'tables': tables}


def pack_result(folder: Path) -> Path:
    """결과 폴더의 공유용 파일(표·곡선 등, 예측값 제외)을 result.zip 하나로 묶는다(Git에는 이것만 올림, 3.6MB vs 풀린 표 28MB).
    같은 내용이면 같은 파일이 되도록 파일 순서와 시각을 고정한다."""
    import zipfile
    folder = Path(folder)
    files = sorted(f for f in folder.iterdir() if f.is_file() and f.name not in NOT_SHARED)
    out = folder / RESULT_ZIP
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            info = zipfile.ZipInfo(f.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())
    return out


def check_author(author) -> None:
    """작성자 폴더 이름: 영문·숫자·_·-만. 한글은 Mac이 다른 방식(NFD)으로 저장해 Git에서 같은 폴더가 둘로 보일 수 있다."""
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', '' if author is None else str(author)):
        raise ValueError(f'author_must_be_ascii_letters_digits_underscore_hyphen:{author!r}')


def result_folder(root: Path, config, *, author: str, experiment: str, started: datetime, suffix: str,
                  out_root: Path | None = None) -> Path:
    """결과 폴더 경로: <model_results>/<작성자>/<시각>_<실험>_<모델·단계>. 만들기만 하고 이미 있으면 멈춘다(덮어쓰기 없음)."""
    check_author(author)
    if not str(experiment) or any(c in str(experiment) for c in '/\\'):
        raise ValueError(f'experiment_name_empty_or_has_path_separator:{experiment!r}')
    base = Path(out_root) if out_root is not None else root / config['PATHS']['model_results']
    folder = base / author / f"{started.strftime('%Y%m%dT%H%M%S')}_{experiment}_{suffix}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder


# ---------- 노트북에서 가져다 쓰는 도우미 ----------

def validation_rows(prepared: dict, config) -> pd.DataFrame:
    """채점할 행(2006~2021, 수익률 있는 전 종목)과 순서.
    열: stock_id, target_month, fold, y(원수익률), common(학습 행 여부), row(prepared['data']의 행 번호)."""
    return validation_keys(prepared['data'], plan_folds(config['MODEL_GRID']))[0]


def fold_rows(prepared: dict, fold: dict, *, inner_scope: str = 'common') -> dict:
    """한 Fold의 행 마스크(prepared['data'] 기준): fit(학습), inner(내부 검증, 설정 고르기), out(바깥 검증=채점).
    선형·부스팅과 같은 규칙: 학습 = 학습 연도의 학습 행, 내부 검증 = 직전 1년, 고른 뒤 다시 학습하지 않음."""
    data = prepared['data']
    year = data.year.to_numpy()
    train = data.train.to_numpy(bool)
    predict = data.predict.to_numpy(bool)
    scope = predict if inner_scope == 'full' else train
    lo, hi = fold['train_years']
    return {'fit': train & (year >= lo) & (year <= hi), 'inner': scope & (year == fold['inner_valid_year']),
            'out': predict & (year == fold['valid_year'])}


def run(root: Path, config, *, execute: bool = False, inner_scope: str = 'common', out_root: Path | None = None,
        workers: int = 1, prepared: dict | None = None, log=print, author: str | None = None, experiment: str = 'base',
        input_transform: str = 'rank') -> dict:
    """execute=False: 입력 준비와 셀(모델 × 손실)별 예상 시간만. execute=True: 6개 셀을 차례로 계산 → 합침 → 저장."""
    started = datetime.now()
    t0 = time.perf_counter()
    prepared = prepare_inputs(root, config, input_transform=input_transform) if prepared is None else prepared
    log(f"입력 준비 {time.perf_counter() - t0:.0f}초: 검증 예측 행 {prepared['checks']['validation_prediction_rows']:,}, "
        f"학습 행(2003~) {prepared['checks']['training_rows_2003_2021']:,}")
    summary = {'stage': 'model_test', 'family': 'linear', 'execute': execute, 'inner_scope': inner_scope, 'workers': workers,
               'input_checks': prepared['checks'],
               'settings': len(linear_settings()), 'input_sets': len(input_sets()), 'folds': 16}
    if not execute:
        ests = [estimate_part_seconds(prepared, config, model=m, loss=l, inner_scope=inner_scope, workers=workers, log=log)
                for m, l in LINEAR_PARTS]
        summary['estimate_minutes'] = {f"{e['model']}|{e['loss']}": round(e['total_minutes'], 1) for e in ests}
        summary['estimate_total_minutes'] = round(sum(e['total_minutes'] for e in ests), 1)
        log(f"전체 예상 약 {summary['estimate_total_minutes']:.0f}분 + 저장 약 1분")
        return summary
    parts = []
    for m, l in LINEAR_PARTS:
        t_part = time.perf_counter()
        parts.append(run_linear(prepared, config, inner_scope=inner_scope, settings=part_settings(m, l), workers=workers, log=log))
        log(f'[{m} × {l}] 완료 {time.perf_counter() - t_part:.0f}초')
    out = finalize_linear(root, config, prepared, merge_results(parts), started=started, out_root=out_root, log=log,
                          author=author, experiment=experiment)
    return {**summary, **out, 'seconds': time.perf_counter() - t0}


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description='v2 model_test: linear_model(OLS·Ridge)')
    parser.add_argument('--config', default='config/experiment.yaml')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--inner-scope', choices=INNER_SCOPES, default='common')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--out-root', default=None)
    parser.add_argument('--author', default=None, help='결과 폴더의 작성자(--execute에 필수)')
    parser.add_argument('--experiment', default='base')
    parser.add_argument('--input-transform', choices=INPUT_TRANSFORMS, default='rank')
    args = parser.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / args.config).read_text(encoding='utf-8'))
    if args.execute:
        check_author(args.author)
    out = run(root, config, execute=args.execute, inner_scope=args.inner_scope, workers=args.workers,
              out_root=Path(args.out_root) if args.out_root else None, author=args.author, experiment=args.experiment,
              input_transform=args.input_transform)
    out.pop('tables', None)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
