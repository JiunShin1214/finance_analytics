"""기본 피처 확장 모델 실행(2026-11-21 사용자 결정, config BASE_FEATURES.model_input): 입력 세트 6개 × 선형·LGB·XGB.

- 입력: 기본 피처(data/processed/base_features.parquet) + H1·H3(model_test 기본) + H2(h2.parquet) + H4(h4_epu_current.parquet).
- 결측: 관측값으로 매달 순위·z를 만든 뒤 빈 칸을 0(그 달 평균)으로 채워 학습·예측 모두에 쓴다(교수님 지침의 예외, 확인 전).
  목표값은 채우지 않는다. 시장 변동성 12개월 창이 안 찬 목표월 2001-01·02는 학습에서 뺀다.
- 중간 저장·재시작: 계산을 단위(선형 = 변환 × 모델 × 손실, 부스팅 = 세트 × 종류 × 학습률 × L2)로 나눠 끝날 때마다
  <결과 위치>/<작성자>/<RUN_NAME>_parts/<단위>.pkl로 저장한다. 다시 실행하면 저장된 단위는 건너뛴다. 단, 저장 때의
  설정·코드·입력 지문(fingerprint)이 지금과 다르면 섞이지 않게 멈춘다. 모든 단위가 끝나면 model_test 저장 규격으로 합친다.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.evaluation import model_test as mt
from src.features.base import MARKET_FEATURES, STOCK_FEATURES

H1 = ['size', 'bm', 'mom']
CODE_FILES = ['src/evaluation/model_test.py', 'src/evaluation/base_model_runs.py', 'src/models/linear.py', 'src/models/boosting.py',
              'src/evaluation/shap_values.py']
XGB_L2_KEY = 'lambda'


def feature_specs(config, family: str) -> dict:
    """prepare_inputs(extra_features=)에 넣을 명세. family: 'linear' | 'boosting'(시장 공통·월 공통 변수 포함)."""
    paths = config['PATHS']
    base = paths['base_features']
    specs = {n: {'path': base, 'transform': 'input'} for n in STOCK_FEATURES}
    specs.update({n: {'path': paths['h2'], 'transform': 'none'} for n in ('debt', 'policy_x_debt')})   # 팀원 z 그대로(H2 규칙)
    # 2026-11-23 팀원 H2 확장: debt_change는 종목별 원값(%p)이라 다른 기본 피처처럼 변환, 상호작용은 policy_x_debt처럼 값 그대로
    specs['debt_change'] = {'path': paths['h2'], 'transform': 'input'}
    specs.update({n: {'path': paths['h2'], 'transform': 'none'} for n in ('policy_change_x_debt', 'rate_change_x_debt')})
    # Mom × EPU는 값 그대로(팀원 H4 규칙). EPU가 그 달 모든 종목에 같아서 매달 순위·z로 바꾸면 Mom의 순위·z와 똑같아져
    # 상호작용 정보가 사라지고 mom과 완전히 같은 열이 된다(2026-11-21 첫 실행에서 Huber 행렬 특이로 발견).
    specs['h4_interaction'] = {'path': paths['h4'], 'transform': 'none'}
    if family == 'boosting':
        specs.update({n: {'path': base, 'transform': 'none'} for n in MARKET_FEATURES})   # 월 공통: 값 그대로
        specs.update({n: {'path': paths['h2'], 'transform': 'none'} for n in ('policy', 'policy_change', 'rate_change')})   # 월 공통
        specs['epu_raw'] = {'path': paths['h4'], 'transform': 'none'}
    return specs


def input_sets(family: str) -> dict:
    """입력 세트 6개(사용자 확인 2026-11-21): 기본 / +H1 / +H3 / +H1+H3 / +H2 / +H2+H4. 부스팅만 월 공통 변수를 넣는다."""
    base = list(STOCK_FEATURES) + (list(MARKET_FEATURES) if family == 'boosting' else [])
    h2 = (['debt', 'policy_x_debt', 'debt_change', 'policy_change_x_debt', 'rate_change_x_debt']
          + (['policy', 'policy_change', 'rate_change'] if family == 'boosting' else []))   # 팀원 H2 변수 8개(2026-11-23)
    h4 = ['h4_interaction'] + (['epu_raw'] if family == 'boosting' else [])
    return {'base': base, 'base+h1': base + H1, 'base+h3': base + ['sigma'], 'base+h1+h3': base + H1 + ['sigma'],
            'base+h1+h3+h2': base + H1 + ['sigma'] + h2, 'base+h1+h3+h2+h4': base + H1 + ['sigma'] + h2 + h4}


def prepare(root: Path, config, family: str, *, input_transform: str, h3_grid=None, log=print) -> dict:
    """전처리 셀: 피처 붙이기·변환·결측 0 채움·학습 제외 월. 반환은 model_test.prepare_inputs와 같은 dict."""
    t0 = time.perf_counter()
    mi = config['BASE_FEATURES']['model_input']
    prepared = mt.prepare_inputs(root, config, h3_grid=h3_grid, extra_features=feature_specs(config, family),
                                 input_transform=input_transform, fill_missing=True,
                                 exclude_train_target_months=mi['exclude_train_target_months'])
    prepared['family'] = family
    check_duplicate_columns(prepared, family)
    c = prepared['checks']
    log(f"전처리 {time.perf_counter() - t0:.0f}초 ({family}, 변환 {input_transform}): 학습 행 {c['training_rows']:,}, "
        f"검증 예측 행 {c['validation_prediction_rows']:,}, 제외 목표월 {c.get('excluded_train_target_months')}")
    return prepared


def check_duplicate_columns(prepared: dict, family: str) -> None:
    """세트 안에 학습 행에서 완전히 같은 입력 열이 있으면 멈춘다(변환 때문에 정보가 사라진 열을 잡기 위해)."""
    import numpy as np
    d = prepared['data']
    rows = d.train.to_numpy(bool)
    label = prepared['sigma_labels'][0]
    for name, cols in input_sets(family).items():
        xs = {c: d['x_sigma_' + label if c == 'sigma' else 'x_' + c].to_numpy(float)[rows] for c in cols}
        keys = list(xs)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if np.array_equal(xs[a], xs[b]):
                    raise ValueError(f'duplicate_input_columns:{name}:{a}={b}')


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root: Path, config, family: str, input_transform: str) -> dict:
    """단위 재사용 조건: config 내용, 계산 코드, 입력 파일이 모두 같아야 한다(커밋은 기록만)."""
    paths = config['PATHS']
    inputs = [paths[k] for k in ('panel', 'eligibility', 'h1_components', 'base_features', 'h2', 'h4')]
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ''
    return {'family': family, 'input_transform': input_transform,
            'config_sha256': hashlib.sha256(yaml.safe_dump(config, sort_keys=True, allow_unicode=True).encode()).hexdigest(),
            'code_sha256': {f: _sha(root / f) for f in CODE_FILES},
            'inputs_sha256': {f: _sha(root / f) for f in inputs}, 'commit_recorded': commit}


def parts_dir(root: Path, config, *, author: str, run_name: str, out_root: Path | None = None) -> Path:
    mt.check_author(author)
    base = Path(out_root) if out_root is not None else root / config['PATHS']['model_results']
    d = base / author / f'{run_name}_parts'
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_units(units: list[tuple[str, dict, callable]], folder: Path, fp: dict, *, log=print) -> dict:
    """units = [(이름, 단위 설명, 계산 함수)]. 저장된 단위(같은 지문)는 읽고, 없으면 계산해 바로 저장. 반환: {이름: 결과}."""
    out = {}
    for i, (name, spec, fn) in enumerate(units, start=1):
        path = folder / f'{name}.pkl'
        want = {**fp, 'unit': spec}
        if path.exists():
            with path.open('rb') as f:
                saved = pickle.load(f)
            if saved['fingerprint'] != want:
                raise ValueError(f'part_fingerprint_differs:{name} — 설정·코드·입력이 저장 때와 다르다. 새 RUN_NAME으로 다시 시작할 것')
            out[name] = saved['result']
            log(f'[{i}/{len(units)}] {name}: 저장된 결과 사용(건너뜀)')
            continue
        t0 = time.perf_counter()
        log(f'[{i}/{len(units)}] {name}: 시작')
        result = fn()
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            pickle.dump({'fingerprint': want, 'result': result, 'saved': datetime.now().isoformat()}, f)
        tmp.replace(path)
        out[name] = result
        log(f'[{i}/{len(units)}] {name}: 완료·저장 {time.perf_counter() - t0:.0f}초')
    return out


# ---------------------------------------------------------------- 선형

def linear_units(prepared: dict, config, *, workers: int = 1, log=print) -> list:
    """단위 = 모델 × 손실(세트 6개 × 목표값 2개를 함께). 변환마다 prepared가 따로 있다."""
    lin = config['BASE_FEATURES']['model_input']['linear']
    sets = input_sets('linear')
    t = prepared['checks']['input_transform']
    units = []
    for model in lin['models']:
        for loss in lin['losses']:
            settings = [(model, target, loss) for target in lin['targets']]
            spec = {'transform': t, 'model': model, 'loss': loss, 'targets': lin['targets'], 'sets': list(sets)}
            fn = (lambda st=settings: mt.run_linear(prepared, config, settings=st, sets=sets, workers=workers, log=log))
            units.append((f'linear__{t}__{model}__{loss}', spec, fn))
    return units


def _once(folder: Path, key: str, make, log) -> dict:
    """저장은 한 번만: 이미 저장했으면(표시 파일) 다시 Run All 해도 결과 폴더를 또 만들지 않는다."""
    mark = folder / f'finalized_{key}.json'
    if mark.exists():
        done = json.loads(mark.read_text(encoding='utf-8'))
        log(f"{key}: 이미 저장됨 → {done['run_dir']} (건너뜀)")
        return done
    out = make()
    done = {'run_dir': out['run_dir'], 'saved': datetime.now().isoformat()}
    mark.write_text(json.dumps(done, ensure_ascii=False), encoding='utf-8')
    return done


def finalize_linear(root: Path, config, prepared: dict, results: dict, *, author: str, run_name: str, started: datetime,
                    parts: Path, out_root: Path | None = None, log=print) -> dict:
    t = prepared['checks']['input_transform']
    merged = lambda: mt.merge_results([r for n, r in results.items() if n.startswith(f'linear__{t}__')])
    return _once(parts, f'linear_{t}', lambda: mt.finalize_linear(root, config, prepared, merged(), started=started,
                                                                  out_root=out_root, log=log, author=author,
                                                                  experiment=f'{run_name}_{t}'), log)


# ---------------------------------------------------------------- 부스팅

def boosting_units(prepared: dict, config, library: str, *, n_threads: int | None = None, workers: int = 1, log=print) -> list:
    """단위 = 세트 × 종류 × 학습률 × L2. 종류 안의 목표값·손실(회귀 4개)은 한 단위에서 함께."""
    spec_cfg = config['BASE_FEATURES']['model_input'][library]
    sets = input_sets('boosting')
    prefix = 'xgb' if library == 'xgboost' else 'lgb'
    kinds = {'regression': f'{prefix}_reg', 'pairwise': 'xgb_pairwise', 'rank_regression': f'{prefix}_rankreg'}
    depth = spec_cfg['depths'][0] if len(spec_cfg['depths']) == 1 else None
    if depth is None:
        raise ValueError('one_depth_expected')
    fixed = [500] if 'fixed_500' in spec_cfg['tree_count'] else []
    units = []
    for set_name, cols in sets.items():
        for k in spec_cfg['kinds']:
            kind = kinds[k]
            if k == 'regression':
                combos = [(t, l) for t in spec_cfg['regression_targets'] for l in spec_cfg['regression_losses']]
            else:
                combos = [(mt.boosting.KINDS[kind]['targets'][0], mt.boosting.KINDS[kind]['losses'][0])]
            l2s = (spec_cfg.get('l2') or {}).get(k, [0])
            for lr in spec_cfg['learning_rates']:
                for l2 in l2s:
                    settings = [(kind, t, l, depth, lr) for t, l in combos]
                    overrides = {XGB_L2_KEY: l2} if (library == 'xgboost' and l2) else None
                    spec = {'set': set_name, 'cols': cols, 'kind': kind, 'settings': settings, 'overrides': overrides,
                            'fixed_trees': fixed}
                    fn = (lambda st=settings, ov=overrides, sn=set_name, c=cols, kd=kind:
                          mt.run_boosting(prepared, config, kd, phase='basefeat', settings=st, sets={sn: c}, overrides=ov,
                                          fixed_trees=fixed, n_threads=n_threads, workers=workers, log=log))
                    units.append((f'{kind}__{set_name}__lr{lr:g}__l2{l2:g}', spec, fn))
    return units


def finalize_boosting(root: Path, config, prepared: dict, results: dict, *, author: str, run_name: str, started: datetime,
                      parts: Path, out_root: Path | None = None, log=print) -> list:
    """종류마다 결과 폴더 하나(<시각>_<RUN_NAME>_<종류>_basefeat)."""
    out = []
    for kind in sorted({n.split('__')[0] for n in results}):
        merged = lambda k=kind: mt.merge_results([r for n, r in results.items() if n.startswith(k + '__')])
        out.append(_once(parts, kind, lambda m=merged: mt.finalize_boosting(root, config, prepared, m(), started=started,
                                                                            out_root=out_root, log=log, author=author,
                                                                            experiment=run_name), log))
    return out


# ---------------------------------------------------------------- 본 실행(5단계, 2026-11-22)

def main_run_spec(config) -> dict:
    return config['BASE_FEATURES']['model_input']['main_run']


def only_fixed(result: dict, n_trees: int) -> dict:
    """트리 n개 고정 결과만 남긴다(같은 학습에서 나오는 해마다 선택 예측·선택 행은 버림). 곡선·내부 검증 표는 그대로."""
    tail = f'_t{n_trees}'
    out = dict(result)
    out['predictions'] = {k: v for k, v in result['predictions'].items() if k.endswith(tail)}
    out['selections'] = result['selections'][result['selections'].hparams.str.endswith(tail)].reset_index(drop=True)
    if not out['predictions']:
        raise ValueError('no_fixed_tree_predictions')
    return out


def main_units(prepared: dict, config, kinds, *, n_threads: int | None = None, workers: int = 1, log=print) -> list:
    """단위 = 종류 × 세트(6개). 설정은 config main_run 하나(깊이·학습률·트리 고정·L2·pairwise 쌍). 모델을 결과에 담는다.
    트리 상한(mt.boosting.MAX_TREES)을 고정 트리 수와 같게 맞춰 두어야 한다(같은 학습의 첫 n개 트리라 결과는 같고 시간만 준다)."""
    run = main_run_spec(config)
    n = int(run['trees'])
    if mt.boosting.MAX_TREES != n:
        raise ValueError(f'set_mt.boosting.MAX_TREES_to_{n}')
    units = []
    for kind in kinds:
        target, loss = run['kinds'][kind]
        ov = {}
        if kind == 'xgb_pairwise':
            ov.update(run['pairwise_pairs'])
        if mt.boosting.KINDS[kind]['library'] == 'xgboost' and run['xgb_l2']:
            ov[XGB_L2_KEY] = run['xgb_l2']
        setting = [(kind, target, loss, int(run['depth']), float(run['learning_rate']))]
        for set_name, cols in input_sets('boosting').items():
            spec = {'set': set_name, 'cols': cols, 'kind': kind, 'settings': setting, 'overrides': ov or None, 'trees': n,
                    'max_trees': mt.boosting.MAX_TREES, 'keep_models': bool(run['save_models'])}
            fn = (lambda st=setting, ov=ov or None, sn=set_name, c=cols, kd=kind:
                  only_fixed(mt.run_boosting(prepared, config, kd, phase='basefeat', settings=st, sets={sn: c}, overrides=ov,
                                             fixed_trees=[n], keep_models=bool(run['save_models']), n_threads=n_threads,
                                             workers=workers, log=log), n))
            units.append((f'{kind}__{set_name}', spec, fn))
    return units


def shap_units(prepared: dict, config, results: dict, *, device: str = 'cpu', log=print) -> list:
    """단위 = 종류 × SHAP 세트. results는 main_units 결과({단위 이름: 결과})."""
    from src.evaluation import shap_values
    sh = main_run_spec(config)['shap']
    units = []
    for name, r in results.items():
        kind, set_name = name.split('__')
        if set_name not in sh['sets']:
            continue
        spec = {'kind': kind, 'set': set_name, 'interactions': set_name in sh['interaction_sets'], 'source_unit': name}
        fn = (lambda r=r, sn=set_name:
              shap_values.compute(prepared, config, r, sets=[sn], interaction_sets=[s for s in sh['interaction_sets'] if s == sn],
                                  device=device, log=log))
        units.append((f'shap__{name}', spec, fn))
    return units


def finalize_main(root: Path, config, prepared: dict, results: dict, shap_results: dict, *, author: str, run_name: str,
                  started: datetime, parts: Path, out_root: Path | None = None, log=print) -> list:
    """종류마다 결과 폴더 하나(<시각>_<RUN_NAME>_<종류>_basefeat): 보고 표 + shap_importance·shap_interactions(공유),
    models.pkl·shap_values.parquet(각자 PC, result.zip 제외)."""
    import pandas as pd
    out = []
    for kind in sorted({n.split('__')[0] for n in results}):
        def make(kind=kind):
            merged = mt.merge_results([r for n, r in results.items() if n.startswith(kind + '__')])
            sh = [r for n, r in shap_results.items() if n.startswith(f'shap__{kind}__')]
            cat = lambda key: pd.concat([r[key] for r in sh if len(r[key])], ignore_index=True) if sh else pd.DataFrame()
            extra = {'shap_importance': (cat('importance'), 'csv'), 'shap_interactions': (cat('interactions'), 'csv')}
            done = mt.finalize_boosting(root, config, prepared, merged, started=started, out_root=out_root, log=log,
                                        author=author, experiment=run_name, extra_tables=extra)
            folder = Path(done['run_dir'])
            with (folder / 'models.pkl').open('wb') as f:
                pickle.dump(merged.get('models', []), f)
            if sh:
                cat('values').to_parquet(folder / 'shap_values.parquet', index=False)
            log(f"{kind}: 모델 {len(merged.get('models', []))}개·SHAP 저장 → {folder}")
            return done
        out.append(_once(parts, f'main_{kind}', make, log))
    return out
