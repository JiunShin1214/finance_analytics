"""기본 피처 확장 실행(2026-11-21): 결측 0 채움·학습 제외 월·새 피처 변환·트리 500 고정·단위 저장 재시작. 합성 자료만 사용."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import base_model_runs as bmr
from src.evaluation import model_test as mt
from tests.test_model_test import CONFIG, LABELS, synthetic


def _data(fill: bool):
    comp, grid, r = synthetic(n_stocks=20)
    return mt.build_model_data(comp, grid, r, LABELS, units={'return': 'decimal'}, fill_missing=fill)


class FillTests(unittest.TestCase):
    def test_fill_off_is_unchanged_and_fill_on_zero_after_transform(self):
        off, on = _data(False), _data(True)
        x = [c for c in off.columns if c.startswith('x_')]
        obs = np.isfinite(off[x].to_numpy())
        np.testing.assert_array_equal(on[x].to_numpy()[obs], off[x].to_numpy()[obs])   # 관측값의 변환은 그대로
        self.assertTrue((on[x].to_numpy()[~obs] == 0).all())                             # 빈 칸만 0
        np.testing.assert_array_equal(on.train.to_numpy(bool),
                                      on.eligible.eq(True).fillna(False).to_numpy(bool) & on.y.notna().to_numpy())
        self.assertTrue(on.train.sum() > off.train.sum())

    def test_extra_feature_transform_and_fill(self):
        d = _data(True)
        f = d[['stock_id', 'base_month']].copy()
        rng = np.random.default_rng(1)
        f['feat'] = rng.normal(size=len(f))
        f.loc[rng.random(len(f)) < .2, 'feat'] = np.nan
        before = d.train.copy()
        out, _ = mt.add_extra_features(d, {'feat': f}, transforms={'feat': 'input'}, input_transform='zscore', fill_missing=True)
        self.assertTrue(out.train.equals(before))                                       # 채우면 학습 행이 줄지 않음
        avail = out.feat_available
        z = out[avail].groupby('base_month').x_feat.agg(['mean', 'std'])
        np.testing.assert_allclose(z['mean'], 0, atol=1e-12)                             # 관측값 안 z → 평균 0
        self.assertTrue((out.loc[~avail, 'x_feat'] == 0).all())
        out2, _ = mt.add_extra_features(_data(False), {'feat': f}, transforms={'feat': 'rank'})
        self.assertTrue(np.isnan(out2.loc[~out2.feat_available, 'x_feat']).all())        # 기본(채움 없음)은 그대로 결측


class InputSetTests(unittest.TestCase):
    def test_six_sets_market_only_for_boosting(self):
        lin, boo = bmr.input_sets('linear'), bmr.input_sets('boosting')
        self.assertEqual(list(lin), ['base', 'base+h1', 'base+h3', 'base+h1+h3', 'base+h1+h3+h2', 'base+h1+h3+h2+h4'])
        for m in ('mkt_ret', 'mkt_ret3', 'mkt_vol12', 'breadth', 'policy', 'policy_change', 'rate_change', 'epu_raw'):
            self.assertFalse(any(m in c for c in lin.values()))
            self.assertIn(m, boo['base+h1+h3+h2+h4'])
        # 선형 주 세트: 종목 기본 20 + H1 3 + σ + H2 5(debt, policy_x_debt, debt_change, 변화×debt 2) + H4 1 = 30
        self.assertEqual(len(lin['base+h1+h3+h2+h4']), 30)
        self.assertEqual(len(boo['base+h1+h3+h2+h4']), 30 + 4 + 3 + 1)   # + 시장 공통 4, 월 공통 H2 3, EPU


class UnitTests(unittest.TestCase):
    def test_saved_units_skipped_and_mismatch_stops(self):
        calls = []
        units = [('a', {'n': 1}, lambda: calls.append('a') or {'v': 1}), ('b', {'n': 2}, lambda: calls.append('b') or {'v': 2})]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fp = {'config_sha256': 'x'}
            first = bmr.run_units(units, folder, fp, log=lambda *_: None)
            again = bmr.run_units(units, folder, fp, log=lambda *_: None)
            self.assertEqual(calls, ['a', 'b'])                                          # 두 번째는 저장분 사용
            self.assertEqual(first, again)
            with self.assertRaises(ValueError):
                bmr.run_units(units, folder, {'config_sha256': 'y'}, log=lambda *_: None)


class FixedTreesTests(unittest.TestCase):
    def test_fixed_500_added_and_yearly_unchanged(self):
        comp, grid, r = synthetic(n_stocks=20)
        data = mt.build_model_data(comp, grid, r, LABELS, units={'return': 'decimal'})
        from src.features.h3 import h3_lambda_candidates
        prepared = {'data': data, 'sigma_labels': LABELS, 'lambda_candidates': h3_lambda_candidates(),
                    'checks': {'input_transform': 'rank'}}
        folds = [f for f in mt.plan_folds(CONFIG['MODEL_GRID']) if f['fold'] == 1]
        kw = dict(settings=[('lgb_rankreg', 'percentile', 'mse', 1, 0.05)], sets={'ind:bm': ['bm']}, folds=folds,
                  log=lambda *_: None)
        plain = mt.run_boosting(prepared, CONFIG, 'lgb_rankreg', phase='individual', **kw)
        both = mt.run_boosting(prepared, CONFIG, 'lgb_rankreg', phase='individual', fixed_trees=[500], **kw)
        self.assertEqual(len(both['predictions']), 2 * len(plain['predictions']))
        for k, v in plain['predictions'].items():
            np.testing.assert_array_equal(both['predictions'][k], v)                    # 해마다 선택 결과는 그대로
        fixed = both['selections'][both['selections'].hparams.str.endswith('_t500')]
        self.assertTrue((fixed.trees == 500).all() and len(fixed) == 1)


if __name__ == '__main__':
    unittest.main()
