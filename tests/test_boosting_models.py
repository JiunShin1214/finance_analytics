"""v2 부스팅 모듈·실행부 계약 — 합성 자료만 사용(실제 data/ 파일을 읽지 않음)."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model_test import CONFIG, prepared_from, synthetic  # noqa: E402

from src.evaluation import model_test as mt  # noqa: E402
from src.models import boosting  # noqa: E402


def toy(n_months=6, per=40, seed=0):
    rng = np.random.default_rng(seed)
    months = np.repeat([f'2010-{m:02d}' for m in range(1, n_months + 1)], per)
    X = rng.uniform(-1, 1, size=(len(months), 2))
    y = 0.3 * X[:, 0] + rng.normal(0, 1, len(months))
    return months, X, y


def deciles(months, y):
    s = pd.Series(y)
    r = s.groupby(months).rank(method='average')
    n = s.groupby(months).transform('count')
    return np.floor(10 * (r - 1) / n).to_numpy(float)


def fit(kind, loss='mse', target=None, delta=None, depth=2, lr=0.01, seed=0):
    months, X, y = toy(seed=seed)
    target = target or boosting.KINDS[kind]['targets'][0]
    label = deciles(months, y) if boosting.KINDS[kind]['task'] == 'rank' else y
    scorer = mt.InnerScorer(months, y)
    out = boosting.fit_boosting(kind, X, label, months, target=target, loss=loss, depth=depth, learning_rate=lr, delta=delta,
                                inner_X=X, inner_months=months, inner_score=lambda p: scorer.score(p)[0],
                                train_score=lambda p: scorer.score(p)[0], n_threads=1)
    return out, X, months, y, scorer


class BoostingModuleTests(unittest.TestCase):
    def test_grid_matches_config_and_setting_counts(self):
        boosting.check_boosting_grid(CONFIG['MODEL_GRID'])
        counts = {k: len(boosting.settings(k)) for k in boosting.KINDS}
        self.assertEqual(counts, {'xgb_reg': 12, 'lgb_reg': 12,
                                  'xgb_pairwise': 3, 'xgb_rankreg': 3, 'lgb_rankreg': 3})
        bad = copy.deepcopy(CONFIG['MODEL_GRID'])
        bad['depths'] = [1, 2]
        with self.assertRaises(ValueError):
            boosting.check_boosting_grid(bad)

    def test_library_parameters(self):
        p = boosting.xgb_params('xgb_pairwise', 'pairwise', 3, 0.01)
        self.assertEqual((p['objective'], p['lambdarank_pair_method'], p['lambdarank_num_pair_per_sample']),
                         ('rank:pairwise', 'topk', 1000))
        self.assertEqual((p['min_child_weight'], p['lambda'], p['alpha'], p['subsample'], p['max_bin']), (0.001, 0.0, 0.0, 1.0, 256))
        self.assertEqual(boosting.xgb_params('xgb_reg', 'huber_sigma135', 1, 0.1, 0.7)['huber_slope'], 0.7)
        q = boosting.lgb_params('lgb_rankreg', 'mse', 3, 0.01)
        self.assertEqual((q['objective'], q['num_leaves'], q['min_data_in_leaf'], q['min_sum_hessian_in_leaf']),
                         ('regression', 8, 1, 0.001))
        self.assertNotIn('xgb_lambdarank', boosting.KINDS)
        self.assertNotIn('lgb_lambdarank', boosting.KINDS)
        self.assertEqual(boosting.lgb_params('lgb_reg', 'huber_sigma135', 2, 0.1, 0.3)['alpha'], 0.3)

    def test_unapproved_settings_rejected(self):
        months, X, y = toy()
        s = lambda p: 0.0
        for kw in ({'depth': 4, 'learning_rate': 0.01, 'loss': 'mse'}, {'depth': 1, 'learning_rate': 0.1, 'loss': 'mse'},
                   {'depth': 1, 'learning_rate': 0.01, 'loss': 'huber_q999'}):
            with self.assertRaises(ValueError):
                boosting.fit_boosting('xgb_reg', X, y, months, target='winsor_1_99', inner_X=X, inner_months=months,
                                      inner_score=s, train_score=s, **kw)
        with self.assertRaises(ValueError):          # 순위 정답은 0~9 정수 등급만
            boosting.fit_boosting('xgb_pairwise', X, y, months, target='decile', loss='pairwise', depth=1, learning_rate=0.01,
                                  inner_X=X, inner_months=months, inner_score=s, train_score=s)

    def test_rank_group_over_all_pairs_limit_rejected(self):
        with self.assertRaises(ValueError):
            boosting._groups(np.array(['2010-01'] * 1001))
        with self.assertRaises(ValueError):
            boosting._groups(np.array(['2010-01', '2010-02', '2010-01']))

    def test_curve_equals_staged_prediction_and_tree_choice(self):
        for kind, loss, delta in (('xgb_reg', 'huber_sigma135', 0.5), ('lgb_reg', 'mse', None),
                                  ('xgb_pairwise', 'pairwise', None), ('lgb_rankreg', 'mse', None)):
            with self.subTest(kind=kind):
                out, X, months, y, scorer = fit(kind, loss=loss, delta=delta)
                for n in (1, 10, 437, 1000):
                    self.assertAlmostEqual(out['inner_curve'][n - 1], scorer.score(boosting.predict_boosting(out, X, n))[0], places=12)
                # 학습 IC는 10트리 묶음 예측 누적(XGBoost는 float32 묶음이라 끝자리만 다를 수 있음)
                for j in (0, 49, 99):
                    self.assertAlmostEqual(out['train_curve'][j], scorer.score(boosting.predict_boosting(out, X, 10 * (j + 1)))[0],
                                           delta=1e-6)
                best = out['inner_curve'].max()
                self.assertEqual(out['best_trees'], int(np.flatnonzero(out['inner_curve'] == best)[0]) + 1)
                self.assertTrue(np.isnan(out['train_curve']).sum() == 0 and len(out['train_curve']) == 100)

    def test_deterministic_repeat(self):
        for kind in ('xgb_pairwise', 'lgb_reg', 'lgb_rankreg'):
            with self.subTest(kind=kind):
                loss = boosting.KINDS[kind]['losses'][0]
                a, X, *_ = fit(kind, loss=loss)
                b, *_ = fit(kind, loss=loss)
                np.testing.assert_array_equal(a['inner_curve'], b['inner_curve'])
                np.testing.assert_array_equal(boosting.predict_boosting(a, X, 1000), boosting.predict_boosting(b, X, 1000))


class BoostingRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prep = prepared_from(*synthetic())
        cand = prep['lambda_candidates']
        cls.prep = {**prep, 'sigma_labels': ['span12', 'span36', 'span60'],
                    'lambda_candidates': cand[cand.label.isin(['span12', 'span36', 'span60'])].reset_index(drop=True)}
        cls.folds = mt.plan_folds(CONFIG['MODEL_GRID'])

    def test_rank_targets_within_training_rows(self):
        d = self.prep['data']
        t = d[d.train]
        self.assertTrue(d.loc[~d.train, ['t_decile', 't_percentile']].isna().all().all())
        for month, g in list(t.groupby('target_month'))[:30]:
            r = g.y.rank(method='average')
            np.testing.assert_array_equal(g.t_decile, np.floor(10 * (r - 1) / len(g)))
            np.testing.assert_allclose(g.t_percentile, r / (len(g) + 1))
        self.assertEqual((t.t_decile.min(), t.t_decile.max()), (0, 9))

    def test_choose_tie_rules_fewer_trees_then_longer_span(self):
        c = lambda decay, curve: {'decay': decay, 'curve': np.array(curve, float)}
        chosen, trees, ties = mt.choose_boosting([c(0.85, [0.1, 0.3, 0.3]), c(0.97, [0.3, 0.2, 0.1])])
        self.assertEqual((chosen['decay'], trees, ties), (0.97, 1, 3))
        chosen, trees, _ = mt.choose_boosting([c(0.85, [0.1, 0.3]), c(0.97, [0.2, 0.3])])
        self.assertEqual((chosen['decay'], trees), (0.97, 2))

    def test_run_contract_one_fold(self):
        setting = ('xgb_reg', 'monthly_zscore', 'huber_sigma135', 1, 0.01)
        quiet = lambda *_: None
        ind = mt.run_boosting(self.prep, CONFIG, 'xgb_reg', phase='individual', settings=[setting], folds=[self.folds[0]], device='cpu',
                              n_threads=1, log=quiet)
        with self.assertRaises(ValueError):          # 지수형 ②는 같은 세션의 ① 결과가 있어야 한다
            mt.run_boosting(self.prep, CONFIG, 'xgb_reg', phase='index', settings=[setting], folds=[self.folds[0]], device='cpu', log=quiet)
        st1 = mt.run_boosting_stage1(self.prep, CONFIG, 'xgb_reg', settings=[setting], device='cpu', n_threads=1, log=quiet)
        idx = mt.run_boosting(self.prep, CONFIG, 'xgb_reg', phase='index', stage1=st1, settings=[setting], folds=[self.folds[0]], device='cpu',
                              n_threads=1, log=quiet)
        with self.assertRaises(ValueError):          # ①과 ②의 장치가 다르면 거부
            mt.run_boosting(self.prep, CONFIG, 'xgb_reg', phase='index', stage1=st1, settings=[setting], folds=[self.folds[0]],
                            device='cuda', log=quiet)
        self.assertEqual((boosting.default_device('xgb_pairwise'), boosting.default_device('lgb_reg')), ('cuda', 'cpu'))
        with self.assertRaises(ValueError):          # 단계가 다르면 합치지 않는다
            mt.merge_results([ind, idx])
        self.assertEqual(set(ind['predictions']), {mt.config_id(mt._boost_tag(setting, 'ind:size+bm+mom+sigma'))})
        self.assertEqual(set(idx['predictions']), {mt.config_id(mt._boost_tag(setting, s)) for s in mt.BOOSTING_PHASES['index']})
        self.assertEqual(len(ind['curves']), 3)      # λ 3개(span 12·36·60)
        self.assertEqual(len(idx['curves']), 19 + 3 + 1 + 3)   # ① 19 + [σ] λ 3 + [H1 지수] 1 + [H1 지수, σ] λ 3
        self.assertEqual(len(idx['stage1']), 19)
        self.assertTrue(idx['stage1'].trees.between(1, 1000).all())
        self.assertTrue(ind['selections'].train_rows_sha256.iloc[0] == idx['selections'].train_rows_sha256.iloc[0])
        r = idx
        sel, vk = ind['selections'], r['val_keys']
        self.assertTrue(all(len(v) == len(vk) and np.isfinite(v).all() for v in r['predictions'].values()))
        self.assertEqual(r['selections'].train_rows_sha256.nunique(), 1)
        # Huber② σ는 같은 행·입력·목표값의 선형 OLS HuberRegressor와 같은 값
        d = self.prep['data']
        f = self.folds[0]
        ctx = mt.FoldContext(d, d.train.to_numpy() & d.year.between(*f['train_years']).to_numpy(),
                             d.train.to_numpy() & d.year.eq(f['inner_valid_year']).to_numpy())
        row = sel[sel.input_set.eq('ind:size+bm+mom+sigma')].iloc[0]
        cols = mt._columns(mt.input_sets()['ind:size+bm+mom+sigma'], row.h3_label, None)
        self.assertEqual(row.huber_sigma, ctx.sigma2(cols, 'monthly_zscore')['sigma'])
        cf = mt.curves_frame(r['curves'])
        self.assertEqual(cf.filter(like='inner_').shape[1] + cf.filter(like='train_').shape[1], 1100)
        ic = mt.outer_monthly_ic(vk, r['predictions'])
        checks = mt.end_checks(self.prep, r, ic, full_run=False)
        self.assertTrue(all(v for k, v in checks.items() if k != 'scipy_independent_max_error'))
        summary = mt.boosting_part_summary(r)
        self.assertEqual(len(summary), 3)
        # 저장: 단계별 폴더, 곡선 parquet, run.json(단계·장치)
        import json
        import tempfile
        from datetime import datetime
        with tempfile.TemporaryDirectory() as tmp:
            out = mt.finalize_boosting(Path(__file__).resolve().parents[1], CONFIG, self.prep, idx, started=datetime(2026, 1, 1),
                                       out_root=Path(tmp), require_complete=False, log=quiet)
            folder = Path(out['run_dir'])
            self.assertEqual((out['status'], folder.name), ('PASSED', '20260101T000000_xgb_reg_index'))
            self.assertTrue((folder / 'curves.parquet').is_file() and (folder / 'stage1_h1_index.csv').is_file())
            record = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual((record['phase'], record['device']['boosting'], record['device']['device_matches_config']),
                             ('index', 'cpu', False))

    def test_linear_ids_unchanged_and_boosting_ids(self):
        tag = {'family': 'linear', 'model': 'ols', 'target': 'raw', 'loss': 'mse', 'input_set': 'ind:bm'}
        self.assertEqual(mt.config_id(tag), 'linear|ols|raw|mse|ind:bm')
        bid = mt.config_id(mt._boost_tag(('lgb_reg', 'raw', 'mse', 2, 0.01), 'idx:h1'))
        self.assertEqual(bid, 'boosting|lgb_reg|raw|mse|idx:h1|d2_lr0.01')
        self.assertEqual(mt.split_config_id(bid)['hparams'], 'd2_lr0.01')
        self.assertEqual(len(mt.expected_boosting_ids('xgb_reg')), 12 * 4)

    def test_scipy_crosscheck_missing_ic(self):
        vk = pd.DataFrame({'target_month': ['2010-01'] * 4 + ['2010-02'] * 4, 'y': [.1, .2, .3, .4, .4, .3, .2, .1],
                           'common': [True] * 8})
        preds = {'c|boosting|m|t|l|s': np.array([1., 1., 1., 1., 1., 2., 3., 4.])}   # 1월은 상수 예측 → IC 결측
        ic = mt.outer_monthly_ic(vk.rename(columns={'c': 'c'}), {'boosting|m|t|l|s': preds['c|boosting|m|t|l|s']})
        self.assertEqual(mt.scipy_crosscheck(vk, {'boosting|m|t|l|s': preds['c|boosting|m|t|l|s']}, ic), 0.0)
        bad = ic.copy()
        bad.loc[bad.target_month.eq('2010-01'), 'ic'] = 0.5                         # 한쪽만 결측이면 불일치
        self.assertEqual(mt.scipy_crosscheck(vk, {'boosting|m|t|l|s': preds['c|boosting|m|t|l|s']}, bad), float('inf'))

    def test_sigma_h3_comparison_tolerance(self):
        keys = pd.DataFrame({'stock_id': ['a', 'a', 'b'], 'base_month': ['2010-01', '2010-02', '2010-01']})
        h3 = keys.assign(sigma=[0.1, np.nan, 0.2])
        grid = keys.assign(sigma_decay097=[0.1 + 5.551e-17, np.nan, 0.2])
        c = mt.compare_sigma_to_h3(grid, h3, atol=1e-10)
        self.assertEqual((c['sigma_decay097_values_differ'], c['sigma_decay097_max_abs_diff'] > 0), (1, True))
        with self.assertRaises(ValueError):
            mt.compare_sigma_to_h3(grid.assign(sigma_decay097=[0.1 + 1e-9, np.nan, 0.2]), h3, atol=1e-10)
        with self.assertRaises(ValueError):
            mt.compare_sigma_to_h3(grid.assign(sigma_decay097=[0.1, 0.3, 0.2]), h3, atol=1e-10)

    def test_unapproved_input_set_rejected(self):
        with self.assertRaises(ValueError):
            mt.run_boosting(self.prep, CONFIG, 'xgb_reg', phase='individual', sets={'ind:bm': ['bm']}, folds=[self.folds[0]],
                            log=lambda *_: None)
        with self.assertRaises(ValueError):
            mt.run_boosting(self.prep, CONFIG, 'xgb_reg', phase='individual', sets={'ind:sigma': ['sigma']},
                            folds=[self.folds[0]], log=lambda *_: None)


if __name__ == '__main__':
    unittest.main()
