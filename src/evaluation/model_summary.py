"""model_test 결과 모으기(보고용). 결과 폴더의 저장 표만 읽어 비교 표를 만든다. 재계산·재학습 없음.

- 읽는 것: 각 폴더의 performance_summary.csv, monthly_ic.csv, fold_table.csv(폴더에 없으면 result.zip 안에서).
- 검사 없음(2026-11-15 결정 라): 같은 설정이 두 폴더에 있으면 둘 다 보여 주고(run = 출처 폴더), IC 개월 수는 months 칸으로 본다.
- 작성자·실험은 폴더 이름에서 읽는다: <base>/<작성자>/<시각>_<실험>_<모델·단계>/ (실험 없는 예전 폴더는 'base').
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MAIN_SETS = ('ind:size+bm+mom+sigma', 'ind:sigma', 'idx:h1', 'idx:h1+sigma')   # 주 결과 입력(부분집합 15개는 선형만)
SET_LABELS = {'ind:size+bm+mom+sigma': '개별형 [Size, B/M, Mom, σ]', 'ind:sigma': '[σ]',
              'idx:h1': '지수형 [H1 지수]', 'idx:h1+sigma': '지수형 [H1 지수, σ]'}
MODEL_ORDER = ('ols', 'ridge', 'xgb_reg', 'lgb_reg', 'xgb_pairwise', 'xgb_rankreg', 'lgb_rankreg')
TABLES = ('performance_summary', 'monthly_ic', 'fold_table')
RESULT_ZIP = 'result.zip'
# 폴더 이름 끝의 모델·단계 꼬리표(model_test.finalize_linear·finalize_boosting이 붙임). 그 앞이 실험 이름.
_SUFFIX = re.compile(r'(?:^|_)(linear|(?:xgb_reg|lgb_reg|xgb_pairwise|xgb_rankreg|lgb_rankreg)_(?:individual|index)'
                     r'(?P<followup>_d\d+)?)(?:_resaved)?$')


def read_member(folder, name: str) -> bytes:
    """결과 폴더의 파일 하나. 폴더에 풀린 파일이 있으면 그것을, 없으면 result.zip 안의 것을 읽는다
    (Git으로 받은 다른 사람 결과는 result.zip만 있다)."""
    import zipfile
    folder = Path(folder)
    if (folder / name).is_file():
        return (folder / name).read_bytes()
    with zipfile.ZipFile(folder / RESULT_ZIP) as z:
        return z.read(name)


def has_member(folder, name: str) -> bool:
    import zipfile
    folder = Path(folder)
    if (folder / name).is_file():
        return True
    if (folder / RESULT_ZIP).is_file():
        with zipfile.ZipFile(folder / RESULT_ZIP) as z:
            return name in z.namelist()
    return False


def _csv(folder, name: str) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(read_member(folder, name)))


def read_parquet(folder, name: str) -> pd.DataFrame:
    import io
    return pd.read_parquet(io.BytesIO(read_member(folder, name)))


def find_run_folders(base_dir) -> list[Path]:
    """결과 폴더 모으기(예: outputs/model_results): <base>/<작성자>/<결과 폴더>와 바로 아래 예전 폴더 중 performance_summary가
    있는 것(풀린 파일 또는 result.zip 안)."""
    base = Path(base_dir)
    candidates = sorted({p for p in [*base.glob('*'), *base.glob('*/*')] if p.is_dir()})
    return [f for f in candidates if has_member(f, 'performance_summary.csv')] if base.is_dir() else []


def parse_folder(folder) -> dict:
    """폴더 이름 → 작성자·실험·후속 튜닝 여부. 예: jiun/20261115T153602_h2_lgb_reg_index → jiun, h2, False."""
    folder = Path(folder)
    rest = re.sub(r'^\d{8}T\d{6}_?', '', folder.name)
    m = _SUFFIX.search(rest)
    experiment = rest[:m.start()] if m else rest
    parent = folder.parent.name
    return {'author': parent if parent not in ('', 'model_results', 'model_test', 'runs') else '미기재',
            'experiment': experiment or 'base', 'followup': bool(m and m.group('followup'))}


def load_runs(folders) -> dict:
    """폴더 목록 → {'runs': 폴더 표, 'performance': ..., 'monthly_ic': ..., 'fold_table': ..., 'folders': {이름: 경로}}."""
    folders = [Path(f) for f in folders]
    if not folders:
        raise ValueError('no_run_folders')
    tables = {t: [] for t in TABLES}
    meta = {}
    for folder in folders:
        meta[folder.name] = parse_folder(folder)
        for t in TABLES:
            frame = _csv(folder, f'{t}.csv')
            if 'hparams' not in frame:
                frame.insert(frame.columns.get_loc('input_set') + 1, 'hparams', '')      # 선형은 초모수 열이 없다
            tables[t].append(frame.assign(hparams=frame.hparams.fillna(''), run=folder.name))
    m = pd.DataFrame.from_dict(meta, orient='index')
    out = {t: pd.concat(v, ignore_index=True).join(m, on='run') for t, v in tables.items()}
    perf = out.pop('performance_summary')
    # 이 칸이 생기기 전 결과는 '미기록'(실험 이름으로 구분: base = 순위, base_zscore = z점수)
    perf['input_transform'] = perf['input_transform'].fillna('미기록') if 'input_transform' in perf else '미기록'
    out['performance'] = perf
    out['folders'] = {f.name: f for f in folders}               # 그림(curves.parquet)용
    out['runs'] = m.rename_axis('run').reset_index()
    return out


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return frame.assign(_o=frame.model.map(rank)).sort_values(['_o']).drop(columns='_o')


COLUMNS = ['model', 'target', 'loss', 'hparams', 'input_set', 'input_transform', 'months', 'mean_ic', 'icir', 'ic_positive_share', 'train_ic_fold_mean',
           'train_minus_valid', 'mean_ic_2006~2010', 'mean_ic_2011~2021', 'mean_ic_2018~2021', 'run']


def _pick(loaded: dict, experiment) -> pd.DataFrame:
    p = loaded['performance']
    return p if experiment is None else p[p.experiment.eq(experiment)]


def best_per_model(loaded: dict, *, scope: str = 'full', sets=MAIN_SETS, include_followup: bool = True,
                   experiment: str | None = 'base') -> pd.DataFrame:
    """모델 종류별 바깥 검증 평균 IC 1등 설정(주 결과 입력 안에서, 기본은 base 실험만). 동률이면 표에 먼저 나온 것."""
    p = _pick(loaded, experiment)
    p = p[p.scope.eq(scope) & p.input_set.isin(sets) & (include_followup | ~p.followup)]
    best = p.loc[p.groupby('model').mean_ic.idxmax()]
    return _order(best)[COLUMNS + ['followup']].reset_index(drop=True)


def best_per_model_and_set(loaded: dict, *, scope: str = 'full', value: str = 'mean_ic', sets=MAIN_SETS,
                           experiment: str | None = 'base') -> pd.DataFrame:
    """모델 × 주 입력별 1등 설정의 값(기본 평균 IC) 넓은 표. 후속 튜닝(깊이 4) 포함."""
    p = _pick(loaded, experiment)
    p = p[p.scope.eq(scope) & p.input_set.isin(sets)]
    best = p.loc[p.groupby(['model', 'input_set']).mean_ic.idxmax()]
    wide = best.pivot(index='model', columns='input_set', values=value)
    wide = wide.reindex([m for m in MODEL_ORDER if m in wide.index])[[s for s in sets if s in wide.columns]]
    return wide.rename(columns=SET_LABELS)


PRESENT_SETS = {'idx:h1': 'H1 지수', 'ind:sigma': 'σ', 'idx:h1+sigma': 'H1 지수 + σ', 'ind:size+bm+mom+sigma': 'Size + B/M + Mom + σ'}
PRESENT_MODELS = {'ols': ('선형', 'OLS'), 'ridge': ('선형', 'Ridge'), 'xgb_reg': ('부스팅 회귀', 'XGBoost'),
                  'lgb_reg': ('부스팅 회귀', 'LightGBM'), 'xgb_pairwise': ('부스팅 순위', 'XGBoost pairwise'),
                  'xgb_rankreg': ('부스팅 순위', 'XGBoost 순위 회귀'), 'lgb_rankreg': ('부스팅 순위', 'LightGBM 순위 회귀')}


def presentation_table(loaded: dict, value: str = 'mean_ic', digits: int = 3, experiment: str | None = 'base') -> pd.DataFrame:
    """발표용: 모든 모델이 공통으로 돌린 입력 4가지 × 모델, 칸 = 그 모델·입력 1등 설정(평균 IC 기준)의 값.
    깊이 4 후속 시험은 넣지 않는다(모든 모델이 같은 후보 범위가 되도록)."""
    p = _pick(loaded, experiment)
    p = p[p.scope.eq('full') & p.input_set.isin(PRESENT_SETS) & ~p.followup]
    best = p.loc[p.groupby(['model', 'input_set']).mean_ic.idxmax()]
    wide = best.pivot(index='model', columns='input_set', values=value)
    wide = wide.reindex([m for m in PRESENT_MODELS if m in wide.index])[list(PRESENT_SETS)].round(digits)
    idx = pd.MultiIndex.from_tuples([PRESENT_MODELS[m] for m in wide.index], names=['계열', '모델'])
    return pd.DataFrame(wide.to_numpy(), index=idx, columns=list(PRESENT_SETS.values()))


def all_settings(loaded: dict, *, scope: str = 'full', sets=MAIN_SETS, experiment: str | None = None) -> pd.DataFrame:
    """모든 설정(평균 IC 내림차순). 기본은 모든 실험, sets=None이면 모든 입력 조합."""
    p = _pick(loaded, experiment)
    p = p[p.scope.eq(scope) & (p.input_set.isin(sets) if sets is not None else True)]
    return p.sort_values('mean_ic', ascending=False)[['author', 'experiment'] + COLUMNS + ['followup']] \
        .reset_index(drop=True)


def depth_table(loaded: dict, *, scope: str = 'full', input_set: str = 'ind:size+bm+mom+sigma',
                experiment: str | None = 'base') -> pd.DataFrame:
    """부스팅: 모델 × 목표·손실 × 깊이별 평균 IC(같은 목표·손실 안에서 깊이 비교)."""
    p = _pick(loaded, experiment)
    p = p[p.scope.eq(scope) & p.input_set.eq(input_set) & p.family.eq('boosting')]
    p = p.assign(depth=p.hparams.str.extract(r'^d(\d+)_')[0].astype(int))
    wide = p.pivot_table(index=['model', 'target', 'loss'], columns='depth', values='mean_ic')
    return wide.add_prefix('depth_')


def save_tables(loaded: dict, out_dir) -> dict:
    """비교 표를 out_dir(예: outputs/모델_v1)에 csv로 쓴다. 다시 실행하면 같은 이름의 표를 덮어쓴다(저장 표에서 다시 만들 수 있음).
    sources.csv = 읽은 결과 폴더 목록(어떤 결과로 만든 표인지)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {'sources': loaded['runs'], 'best_per_model': best_per_model(loaded),
              'by_input_mean_ic': best_per_model_and_set(loaded).reset_index(),
              'by_input_icir': best_per_model_and_set(loaded, value='icir').reset_index(),
              'boosting_depth': depth_table(loaded).reset_index(), 'all_settings': all_settings(loaded),
              'presentation_mean_ic': presentation_table(loaded).reset_index(),
              'presentation_icir': presentation_table(loaded, 'icir', 2).reset_index()}
    for name, table in tables.items():
        table.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    return {name: str(out / f'{name}.csv') for name in tables}
