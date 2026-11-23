"""본 실행 모델의 SHAP(2026-11-22 사용자 결정, config BASE_FEATURES.model_input.main_run.shap).

- 대상: run_boosting(keep_models=True)이 저장한 모델(Fold × 세트, 예측에 쓴 λ·트리 500개). 행은 그 Fold의 바깥 검증 행
  (2006~2021을 Fold마다 이어 붙임). 입력 행렬은 예측 때와 같다(_design, 빈 칸 0).
- SHAP 값: XGBoost는 predict(pred_contribs), LightGBM은 predict(pred_contrib). 모델 출력(원점수, margin) 단위이고
  한 행의 SHAP 합 + 기본값 = 그 행 예측값이다. 계산마다 (1) 저장 모델 예측 = 실행 때 예측, (2) 합 = 예측을 확인하고 어긋나면 멈춘다.
- 상호작용: XGBoost predict(pred_interactions), LightGBM shap.TreeExplainer.shap_interaction_values. 원값은 너무 커서
  (행 × 피처 × 피처) 저장하지 않고 Fold별 평균만 남긴다. 두 피처 쌍의 값은 I[a,b] + I[b,a](대칭 두 칸 합).
- 출력: values(행 × 피처 SHAP, 저장은 shap_values.parquet), importance(피처별 평균 |SHAP|·평균·입력값과의 상관),
  interactions(피처 쌍 평균 |상호작용|·평균). 파일을 읽거나 쓰지 않는다.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.evaluation import model_test as mt
from src.models import boosting

CHUNK = 4000          # 상호작용 계산을 나누는 행 수(행 × 피처² 메모리)
PRED_TOL = 1e-4       # 저장 모델 예측과 실행 때 예측의 최대 차이(float32)
SUM_TOL = 1e-3        # SHAP 합 + 기본값과 예측의 최대 차이


def _contribs(library: str, booster, X: np.ndarray) -> np.ndarray:
    """(행, 피처 + 1). 마지막 열 = 기본값."""
    if library == 'xgboost':
        import xgboost as xgb
        return booster.predict(xgb.DMatrix(X), pred_contribs=True).astype(float)
    return np.asarray(booster.predict(X, pred_contrib=True), float)


def _predict(library: str, booster, X: np.ndarray) -> np.ndarray:
    if library == 'xgboost':
        import xgboost as xgb
        return booster.predict(xgb.DMatrix(X)).astype(float)
    return np.asarray(booster.predict(X), float)


def _interactions(library: str, booster, X: np.ndarray) -> np.ndarray:
    """(행, 피처, 피처). 기본값 칸은 뺀다."""
    if library == 'xgboost':
        import xgboost as xgb
        return booster.predict(xgb.DMatrix(X), pred_interactions=True)[:, :-1, :-1].astype(float)
    import shap
    return np.asarray(shap.TreeExplainer(booster).shap_interaction_values(X), float)


def _corr(x: np.ndarray, s: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(s) == 0:
        return np.nan
    return float(np.corrcoef(x, s)[0, 1])


def compute(prepared: dict, config, result: dict, *, sets, interaction_sets=(), device: str = 'cpu', log=print) -> dict:
    """result['models'] 중 sets에 든 세트의 SHAP. interaction_sets 세트는 상호작용 요약도. 반환: values·importance·interactions 표."""
    models = [m for m in result.get('models', []) if m['input_set'] in set(sets)]
    if not models:
        raise ValueError('no_saved_models_for_shap')
    data = prepared['data']
    folds = {f['fold']: f for f in mt.plan_folds(config['MODEL_GRID'])}
    vk = result['val_keys']
    values, imp, inter = [], [], []
    pooled = {}                                   # (모델, 세트) → [(X, SHAP)] 전체 기간 상관용
    for i, m in enumerate(sorted(models, key=lambda z: (z['model'], z['input_set'], z['fold'])), start=1):
        t0 = time.perf_counter()
        rows = mt.fold_rows(prepared, folds[m['fold']])['out']
        keys = vk[vk.fold.eq(m['fold'])]
        if not np.array_equal(keys.row.to_numpy(), np.flatnonzero(rows)):
            raise ValueError(f"shap_rows_differ_from_validation_rows:{m['config_id']}:fold{m['fold']}")
        X = mt._design(data, m['columns'], rows, fill=True)
        booster = boosting.load_model(m['library'], m['raw'], device=device)
        pred = _predict(m['library'], booster, X)
        stored = result['predictions'][m['config_id']][keys.index.to_numpy()]
        gap = float(np.max(np.abs(pred - stored)))
        if gap > PRED_TOL:
            raise ValueError(f"saved_model_prediction_differs:{m['config_id']}:fold{m['fold']}:{gap:.2e}")
        c = _contribs(m['library'], booster, X)
        gap_sum = float(np.max(np.abs(c.sum(axis=1) - pred)))
        if gap_sum > SUM_TOL:
            raise ValueError(f"shap_sum_differs_from_prediction:{m['config_id']}:fold{m['fold']}:{gap_sum:.2e}")
        feats, S = m['features'], c[:, :-1]
        tag = {'model': m['model'], 'input_set': m['input_set'], 'fold': m['fold'], 'valid_year': m['valid_year']}
        frame = pd.DataFrame(S.astype(np.float32), columns=feats)
        frame.insert(0, 'base_value', c[:, -1].astype(np.float32))
        for k, v in reversed(list({**tag, 'stock_id': keys.stock_id.to_numpy(),
                                    'target_month': keys.target_month.to_numpy()}.items())):
            frame.insert(0, k, v)
        values.append(frame)
        for j, f in enumerate(feats):
            imp.append({**tag, 'feature': f, 'rows': len(X), 'mean_abs': float(np.abs(S[:, j]).mean()),
                        'mean': float(S[:, j].mean()), 'corr_input': _corr(X[:, j], S[:, j])})
        pooled.setdefault((m['model'], m['input_set']), []).append((X, S, feats))
        msg = f"[{i}/{len(models)}] {m['model']} {m['input_set']} Fold {m['fold']:>2}: SHAP {len(X):,}행, 예측 차이 {gap:.1e}, 합 차이 {gap_sum:.1e}"
        if m['input_set'] in set(interaction_sets):
            p = len(feats)
            tot_abs, tot = np.zeros((p, p)), np.zeros((p, p))
            gap_int = 0.0
            for a in range(0, len(X), CHUNK):
                I = _interactions(m['library'], booster, X[a:a + CHUNK])
                gap_int = max(gap_int, float(np.max(np.abs(I.sum(axis=2) - S[a:a + CHUNK]))))
                pair = I + I.transpose(0, 2, 1)
                idx = np.arange(p)
                pair[:, idx, idx] = I[:, idx, idx]               # 대각 = 주효과(한 칸)
                tot_abs += np.abs(pair).sum(axis=0)
                tot += pair.sum(axis=0)
            if gap_int > SUM_TOL:
                raise ValueError(f"shap_interaction_sum_differs:{m['config_id']}:fold{m['fold']}:{gap_int:.2e}")
            for a in range(p):
                for b in range(a, p):
                    inter.append({**tag, 'feature_a': feats[a], 'feature_b': feats[b], 'rows': len(X),
                                  'mean_abs': tot_abs[a, b] / len(X), 'mean': tot[a, b] / len(X)})
            msg += f', 상호작용 합 차이 {gap_int:.1e}'
        log(f'{msg}, {time.perf_counter() - t0:.1f}초')
    imp = pd.DataFrame(imp)
    # 전체 기간(fold='all'): 평균은 행 수 가중, 상관은 이어 붙인 행으로
    allrows = []
    for (model, s), parts in pooled.items():
        feats = parts[0][2]
        X = np.vstack([x for x, _, _ in parts])
        S = np.vstack([v for _, v, _ in parts])
        for j, f in enumerate(feats):
            allrows.append({'model': model, 'input_set': s, 'fold': 'all', 'valid_year': 'all', 'feature': f, 'rows': len(X),
                            'mean_abs': float(np.abs(S[:, j]).mean()), 'mean': float(S[:, j].mean()),
                            'corr_input': _corr(X[:, j], S[:, j])})
    imp = pd.concat([imp.astype({'fold': object, 'valid_year': object}), pd.DataFrame(allrows)], ignore_index=True)
    imp['share'] = imp.mean_abs / imp.groupby(['model', 'input_set', 'fold']).mean_abs.transform('sum')
    inter = pd.DataFrame(inter)
    if len(inter):
        g = inter.groupby(['model', 'input_set', 'feature_a', 'feature_b'], sort=False)
        w = lambda col: g.apply(lambda d: float(np.average(d[col], weights=d.rows)), include_groups=False)
        allp = pd.DataFrame({'rows': g.rows.sum(), 'mean_abs': w('mean_abs'), 'mean': w('mean')}).reset_index()
        inter = pd.concat([inter.astype({'fold': object, 'valid_year': object}),
                           allp.assign(fold='all', valid_year='all')], ignore_index=True)
    return {'values': pd.concat(values, ignore_index=True), 'importance': imp, 'interactions': inter}
