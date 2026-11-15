"""model_test 결과 모으기(보고용). 노트북에 적은 실행 폴더들의 저장 표만 읽어 비교 표를 만든다.

- 계산 입력이 아니다: 재계산·재학습 없음. 폴더 경로는 호출하는 쪽이 넘긴다(outputs/run_id를 찾거나 전제하지 않음).
- 읽는 것: 각 폴더의 run.json, performance_summary.csv, monthly_ic.csv, fold_table.csv(폴더에 없으면 result.zip 안에서).
- 합치기 전에 검사: 모두 PASSED, 같은 공통 입력 해시·채점 행·Fold 계획·내부 검증 범위·평가 계약, 설정 ID 중복 없음, 192개월.
- 팀 실험(2026-11-15): 결과 규격 outputs/model_results/<작성자>/<시각>_<실험>_<모델·단계>/. run.json의 author·experiment로 구분(없으면
  '미기재'·'base'). 새 특징 파일은 같은 이름이면 같은 해시여야 함께 합친다. 학습 행 수는 실험마다 다를 수 있어 열로 보여 준다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# 결과를 서로 비교해도 되는지 판단하는 평가 계약(MODEL_GRID 일부). 모델 후보(학습률·깊이 등)는 실행마다 달라도 된다.
CONTRACT_KEYS = ('train_start_target_year', 'first_validation_year', 'last_validation_year', 'inner_validation_years',
                 'min_inner_train_years', 'min_inner_train_exceptions', 'training_rows', 'prediction_rows',
                 'target_returns_unchanged', 'components', 'input_transform', 'h3_fixed', 'missing_feature_fill')
MAIN_SETS = ('ind:size+bm+mom+sigma', 'ind:sigma', 'idx:h1', 'idx:h1+sigma')   # 주 결과 입력(부분집합 15개는 선형만)
SET_LABELS = {'ind:size+bm+mom+sigma': '개별형 [Size, B/M, Mom, σ]', 'ind:sigma': '[σ]',
              'idx:h1': '지수형 [H1 지수]', 'idx:h1+sigma': '지수형 [H1 지수, σ]'}
MODEL_ORDER = ('ols', 'ridge', 'xgb_reg', 'lgb_reg', 'xgb_pairwise', 'xgb_rankreg', 'lgb_rankreg')
MONTHS = 192
TABLES = ('performance_summary', 'monthly_ic', 'fold_table')
RESULT_ZIP = 'result.zip'


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


BASE_INPUTS = ('data/interim/panel.parquet', 'data/interim/eligibility.parquet', 'data/processed/h1_components.parquet',
               'data/processed/h3.parquet')


def find_run_folders(base_dir, *, log=print) -> list[Path]:
    """결과 폴더 모으기(예: outputs/model_results). 규격 <base>/<작성자>/<결과 폴더>/run.json과, 작성자 폴더 없이 바로 아래에 둔
    예전 결과 폴더를 함께 찾는다. PASSED만 돌려주고 나머지는 이유를 알리고 뺀다."""
    base = Path(base_dir)
    marks = [*base.glob('*/run.json'), *base.glob('*/*/run.json'), *base.glob(f'*/{RESULT_ZIP}'), *base.glob(f'*/*/{RESULT_ZIP}')]
    candidates = sorted({r.parent for r in marks})
    out = []
    for folder in candidates:
        status = json.loads(read_member(folder, 'run.json').decode('utf-8')).get('status')
        if status == 'PASSED':
            out.append(folder)
        else:
            log(f'제외: {folder.name} ({status})')
    return out


def load_runs(folders) -> dict:
    """폴더 목록 → {'runs': 실행 요약 표, 'performance': ..., 'monthly_ic': ..., 'fold_table': ...}. 검사 실패 시 ValueError."""
    folders = [Path(f) for f in folders]
    if not folders:
        raise ValueError('no_run_folders')
    records, tables = [], {t: [] for t in TABLES}
    for folder in folders:
        record = json.loads(read_member(folder, 'run.json').decode('utf-8'))
        if record.get('status') != 'PASSED':
            raise ValueError(f"run_not_passed:{folder.name}:{record.get('status')}")
        records.append((folder, record))
        for t in TABLES:
            frame = _csv(folder, f'{t}.csv')
            if 'hparams' not in frame:
                frame.insert(frame.columns.get_loc('input_set') + 1, 'hparams', '')      # 선형은 초모수 열이 없다
            tables[t].append(frame.assign(hparams=frame.hparams.fillna(''), run=folder.name))
    _check_same_basis(records)
    meta = {f.name: {'author': r.get('author') or _author_from_folder(f), 'experiment': r.get('experiment') or 'base',
                     'train_rows': r['input_checks'].get('training_rows_2003_2021'), 'followup': bool(r.get('followup', False))}
            for f, r in records}
    out = {t: pd.concat(v, ignore_index=True) for t, v in tables.items()}
    for t in TABLES:
        m = pd.DataFrame.from_dict(meta, orient='index')
        out[t] = out[t].join(m, on='run')
    perf = out['performance_summary']
    ids = perf[perf.scope.eq('full')].assign(
        cid=lambda d: d.author + '|' + d.experiment + '|' + d.model + '|' + d.target + '|' + d.loss + '|' + d.input_set + '|' + d.hparams)
    dup = ids.cid[ids.cid.duplicated()]
    if len(dup):
        raise ValueError('config_in_two_runs:' + dup.iloc[0])
    if not perf.months.eq(MONTHS).all():
        raise ValueError('not_all_192_months')
    out['performance'] = perf
    del out['performance_summary']
    out['folders'] = {f.name: f for f, _ in records}            # 그림(curves.parquet)용
    out['runs'] = pd.DataFrame([{
        'run': f.name, 'author': meta[f.name]['author'], 'experiment': meta[f.name]['experiment'],
        'train_rows_2003_2021': meta[f.name]['train_rows'], 'extra_features': ','.join(r.get('extra_features', {})),
        'family': r['family'], 'kind': r.get('kind') or r['family'], 'phase': r.get('phase', ''),
        'followup': bool(r.get('followup', False)), 'device': (r['device'] or {}).get('boosting', 'cpu') if isinstance(r['device'], dict) else 'cpu',
        'gpu': (r['device'] or {}).get('gpu') if isinstance(r['device'], dict) else None, 'configs': r['configs'],
        'commit': (r.get('commit') or '없음')[:7], 'computed_commit': (r.get('computed_commit') or r.get('commit') or '없음')[:7],
        'constant_score_months': len(r.get('constant_score_months', [])), 'started': r['started'][:16], 'finished': r['finished'][:16]}
        for f, r in records])
    return out


def _author_from_folder(folder: Path) -> str:
    """run.json에 작성자가 없는 예전 결과: 규격 위치(<base>/<작성자>/<결과 폴더>)에 두었으면 상위 폴더 이름, 아니면 '미기재'."""
    parent = folder.parent.name
    return parent if parent and parent not in ('model_results', 'model_test', 'runs') else '미기재'


def _check_same_basis(records) -> None:
    base_folder, base = records[0]
    seen = dict(base['input_sha256'])
    for folder, r in records[1:]:
        for key in BASE_INPUTS:                                  # 공통 입력 4개는 모두 같아야 한다
            if r['input_sha256'].get(key) != base['input_sha256'].get(key):
                raise ValueError(f'input_sha256_differs:{key}:{folder.name}')
        for key, value in r['input_sha256'].items():            # 새 특징 파일은 같은 이름이면 같은 해시
            if seen.setdefault(key, value) != value:
                raise ValueError(f'feature_file_version_differs:{key}:{folder.name}')
        for key in ('folds_plan', 'inner_scope'):
            if r[key] != base[key]:
                raise ValueError(f'{key}_differs:{folder.name}')
        for key in ('validation_prediction_rows', 'sigma_decay097_equals_h3_parquet'):
            if r['input_checks'].get(key) != base['input_checks'].get(key):
                raise ValueError(f'input_check_{key}_differs:{folder.name}')
        for key in CONTRACT_KEYS:
            if r['model_grid'].get(key) != base['model_grid'].get(key):
                raise ValueError(f'model_grid_{key}_differs:{folder.name}')


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return frame.assign(_o=frame.model.map(rank)).sort_values(['_o']).drop(columns='_o')


COLUMNS = ['model', 'target', 'loss', 'hparams', 'input_set', 'mean_ic', 'icir', 'ic_positive_share', 'train_ic_fold_mean',
           'train_minus_valid', 'mean_ic_2006~2010', 'mean_ic_2011~2021', 'mean_ic_2018~2021']


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
    return p.sort_values('mean_ic', ascending=False)[['author', 'experiment', 'train_rows'] + COLUMNS + ['followup']] \
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
    sources.csv = 읽은 실행 폴더·커밋·장치 목록(어떤 결과로 만든 표인지)."""
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
