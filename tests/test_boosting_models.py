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


def fit(kind, loss='mse', target=None, delta=None, depth=2, lr=0.1, seed=0):
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
        self.assertEqual(counts, {'xgb_reg': 54, 'lgb_reg': 54, 'xgb_lambdarank': 6, 'lgb_lambdarank': 6,
                                  'xgb_pairwise': 6, 'xgb_rankreg': 6, 'lgb_rankreg': 6})
        bad = copy.deepcopy(CONFIG['MODEL_GRID'])
        bad['depths'] = [1, 2]
        with self.assertRaises(ValueError):
            boosting.check_boosting_grid(bad)

    def test_library_parameters(self):
        p = boosting.xgb_params('xgb_lambdarank', 'lambdarank', 3, 0.01)
        self.assertEqual((p['objective'], p['lambdarank_pair_method'], p['lambdarank_num_pair_per_sample'], p['ndcg_exp_gain']),
                         ('rank:ndcg', 'topk', 1000, True))
        self.assertEqual((p['min_child_weight'], p['lambda'], p['alpha'], p['subsample'], p['max_bin']), (0.001, 0.0, 0.0, 1.0, 256))
        self.assertEqual(boosting.xgb_params('xgb_reg', 'huber_q999', 1, 0.1, 0.7)['huber_slope'], 0.7)
        q = boosting.lgb_params('lgb_lambdarank', 'lambdarank', 3, 0.1)
        self.assertEqual((q['num_leaves'], q['lambdarank_truncation_level'], q['label_gain'][9], q['min_data_in_leaf']),
                         (8, 1000, 511.0, 1))
        self.assertEqual(boosting.lgb_params('lgb_reg', 'huber_sigma135', 2, 0.1, 0.3)['alpha'], 0.3)

    def test_unapproved_settings_rejected(self):
        months, X, y = toy()
        s = lambda p: 0.0
        for kw in ({'depth': 4, 'learning_rate': 0.1, 'loss': 'mse'}, {'depth': 1, 'learning_rate': 0.05, 'loss': 'mse'},
                   {'depth': 1, 'learning_rate': 0.1, 'loss': 'huber_q999'}):
            with self.assertRaises(ValueError):
                boosting.fit_boosting('xgb_reg', X, y, months, target='raw', inner_X=X, inner_months=months,
                                      inner_score=s, train_score=s, **kw)
        with self.assertRaises(ValueError):          # 순위 정답은 0~9 정수 등급만
            boosting.fit_boosting('xgb_pairwise', X, y, months, target='decile', loss='pairwise', depth=1, learning_rate=0.1,
                                  inner_X=X, inner_months=months, inner_score=s, train_score=s)

    def test_rank_group_over_all_pairs_limit_rejected(self):
        with self.assertRaises(ValueError):
            boosting._groups(np.array(['2010-01'] * 1001))
        with self.assertRaises(ValueError):
            boosting._groups(np.array(['2010-01', '2010-02', '2010-01']))

    def test_curve_equals_staged_prediction_and_tree_choice(self):
        for kind, loss, delta in (('xgb_reg', 'huber_sigma135', 0.5), ('lgb_reg', 'mse', None), ('xgb_lambdarank', 'lambdarank', None),
                                  ('lgb_lambdarank', 'lambdarank', None), ('xgb_pairwise', 'pairwise', None)):
            with self.subTest(kind=kind):
                out, X, months, y, scorer = fit(kind, loss=loss, delta=delta)
                for n in (1, 10, 437, 1000):
                    self.assertAlmostEqual(out['inner_curve'][n - 1], scorer.score(boosting.predict_boosting(out, X, n))[0], places=12)
                self.assertAlmostEqual(out['train_curve'][49], scorer.score(boosting.predict_boosting(out, X, 500))[0], places=12)
                best = out['inner_curve'].max()
                self.assertEqual(out['best_trees'], int(np.flatnonzero(out['inner_curve'] == best)[0]) + 1)
                self.assertTrue(np.isnan(out['train_curve']).sum() == 0 and len(out['train_curve']) == 100)

    def test_deterministic_repeat(self):
        for kind in ('xgb_pairwise', 'lgb_lambdarank', 'lgb_rankreg'):
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
        cls.prep = {**prep, 'sigma_labels': ['span12', 'decay097'],
                    'lambda_candidates': cand[cand.label.isin(['span12', 'decay097'])].reset_index(drop=True)}
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
        setting = ('xgb_reg', 'monthly_zscore', 'huber_sigma135', 1, 0.1)
        r = mt.run_boosting(self.prep, CONFIG, 'xgb_reg', settings=[setting], folds=[self.folds[0]], n_threads=1, log=lambda *_: None)
        sel, vk = r['selections'], r['val_keys']
        self.assertEqual(set(r['predictions']), {mt.config_id(mt._boost_tag(setting, s)) for s in mt.BOOSTING_SETS})
        self.assertTrue(all(len(v) == len(vk) and np.isfinite(v).all() for v in r['predictions'].values()))
        self.assertEqual(sel.train_rows_sha256.nunique(), 1)
        # Huber② σ는 같은 행·입력·목표값의 선형 OLS HuberRegressor와 같은 값
        d = self.prep['data']
        f = self.folds[0]
        ctx = mt.FoldContext(d, d.train.to_numpy() & d.year.between(*f['train_years']).to_numpy(),
                             d.train.to_numpy() & d.year.eq(f['inner_valid_year']).to_numpy())
        row = sel[sel.input_set.eq('ind:size+bm+mom+sigma')].iloc[0]
        cols = mt._columns(mt.input_sets()['ind:size+bm+mom+sigma'], row.h3_label, None)
        self.assertEqual(row.huber_sigma, ctx.sigma2(cols, 'monthly_zscore')['sigma'])
        # 지수형 ①은 2003~2021년 19개, 곡선 = ① 19 + Fold(σ 조합 3개 × λ 2 + [H1 지수] 1)
        self.assertEqual(len(r['stage1']), 19)
        self.assertEqual(len(r['curves']), 19 + 7)
        self.assertTrue(r['stage1'].trees.between(1, 1000).all())
        cf = mt.curves_frame(r['curves'])
        self.assertEqual(cf.filter(like='inner_').shape[1] + cf.filter(like='train_').shape[1], 1100)
        ic = mt.outer_monthly_ic(vk, r['predictions'])
        checks = mt.end_checks(self.prep, r, ic, full_run=False)
        self.assertTrue(all(v for k, v in checks.items() if k != 'scipy_independent_max_error'))
        summary = mt.boosting_part_summary(r)
        self.assertEqual(len(summary), 4)

    def test_linear_ids_unchanged_and_boosting_ids(self):
        tag = {'family': 'linear', 'model': 'ols', 'target': 'raw', 'loss': 'mse', 'input_set': 'ind:bm'}
        self.assertEqual(mt.config_id(tag), 'linear|ols|raw|mse|ind:bm')
        bid = mt.config_id(mt._boost_tag(('lgb_reg', 'raw', 'mse', 2, 0.01), 'idx:h1'))
        self.assertEqual(bid, 'boosting|lgb_reg|raw|mse|idx:h1|d2_lr0.01')
        self.assertEqual(mt.split_config_id(bid)['hparams'], 'd2_lr0.01')
        self.assertEqual(len(mt.expected_boosting_ids('xgb_reg')), 54 * 4)

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
            mt.run_boosting(self.prep, CONFIG, 'xgb_reg', sets={'ind:bm': ['bm']}, folds=[self.folds[0]], log=lambda *_: None)


if __name__ == '__main__':
    unittest.main()
