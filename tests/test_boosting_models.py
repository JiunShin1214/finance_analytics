"""v2 부스팅 모듈·실행부 계약 — 합성 자료만 사용(실제 data/ 파일을 읽지 않음)."""
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
    def test_setting_counts(self):
        counts = {k: len(boosting.settings(k)) for k in boosting.KINDS}
        self.assertEqual(counts, {'xgb_reg': 12, 'lgb_reg': 12,
                                  'xgb_pairwise': 3, 'xgb_rankreg': 3, 'lgb_rankreg': 3})

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

    def test_settings_needed_for_computation_rejected(self):
        months, X, y = toy()
        s = lambda p: 0.0
        for kw in ({'depth': 1, 'learning_rate': 0.01, 'loss': 'huber_q999'},):     # Huber인데 δ 없음
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
        self.assertEqual((boosting.default_device('xgb_pairwise'), boosting.default_device('lgb_reg')), ('cuda', 'cpu'))
        self.assertEqual(set(ind['predictions']), {mt.config_id(mt._boost_tag(setting, 'ind:size+bm+mom+sigma'))})
        self.assertEqual(set(idx['predictions']), {mt.config_id(mt._boost_tag(setting, s)) for s in mt.BOOSTING_PHASES['index']})
        self.assertEqual(len(ind['curves']), 3)      # λ 3개(span 12·36·60)
        self.assertEqual(len(idx['curves']), 19 + 3 + 1 + 3)   # ① 19 + [σ] λ 3 + [H1 지수] 1 + [H1 지수, σ] λ 3
        self.assertEqual(len(idx['stage1']), 19)
        self.assertTrue(idx['stage1'].trees.between(1, 1000).all())
        self.assertEqual(ind['selections'].n_train.iloc[0], idx['selections'].n_train.iloc[0])
        r = idx
        sel, vk = ind['selections'], r['val_keys']
        self.assertTrue(all(len(v) == len(vk) and np.isfinite(v).all() for v in r['predictions'].values()))
        self.assertEqual(r['selections'].n_train.nunique(), 1)
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
        summary = mt.boosting_part_summary(r)
        self.assertEqual(len(summary), 3)
        # 저장: 단계별 폴더, 곡선 parquet, result.zip
        import tempfile
        from datetime import datetime
        with tempfile.TemporaryDirectory() as tmp:
            out = mt.finalize_boosting(Path(__file__).resolve().parents[1], CONFIG, self.prep, idx, started=datetime(2026, 1, 1),
                                       out_root=Path(tmp), log=quiet, author='tester', experiment='base')
            folder = Path(out['run_dir'])
            self.assertEqual((folder.name, folder.parent.name), ('20260101T000000_base_xgb_reg_index', 'tester'))
            self.assertTrue((folder / 'curves.parquet').is_file() and (folder / 'stage1_h1_index.csv').is_file())
            self.assertTrue((folder / 'result.zip').is_file())

    def test_followup_depth4_separate_run(self):
        self.assertEqual(boosting.settings('lgb_rankreg', followup=True), [('lgb_rankreg', 'percentile', 'mse', 4, 0.01)])
        self.assertEqual(boosting.settings('xgb_reg', followup=True), [])
        self.assertEqual(boosting.lgb_params('lgb_rankreg', 'mse', 4, 0.01)['num_leaves'], 16)
        quiet = lambda *_: None
        r = mt.run_boosting(self.prep, CONFIG, 'lgb_rankreg', phase='individual', followup=True, folds=[self.folds[0]],
                            n_threads=1, log=quiet)
        self.assertEqual(set(r['selections'].hparams), {'d4_lr0.01'})
        import tempfile
        from datetime import datetime
        with tempfile.TemporaryDirectory() as tmp:
            out = mt.finalize_boosting(Path(__file__).resolve().parents[1], CONFIG, self.prep, r, started=datetime(2026, 1, 1),
                                       out_root=Path(tmp), log=quiet, author='tester', experiment='base')
            self.assertEqual(Path(out['run_dir']).name, '20260101T000000_base_lgb_rankreg_individual_d4')

    def test_linear_ids_unchanged_and_boosting_ids(self):
        tag = {'family': 'linear', 'model': 'ols', 'target': 'raw', 'loss': 'mse', 'input_set': 'ind:bm'}
        self.assertEqual(mt.config_id(tag), 'linear|ols|raw|mse|ind:bm')
        bid = mt.config_id(mt._boost_tag(('lgb_reg', 'raw', 'mse', 2, 0.01), 'idx:h1'))
        self.assertEqual(bid, 'boosting|lgb_reg|raw|mse|idx:h1|d2_lr0.01')
        self.assertEqual(mt.split_config_id(bid)['hparams'], 'd2_lr0.01')

    def test_constant_month_counts_zero(self):
        vk = pd.DataFrame({'target_month': ['2010-01'] * 4 + ['2010-02'] * 4, 'y': [.1, .2, .3, .4, .4, .3, .2, .1],
                           'common': [True] * 8})
        preds = {'boosting|m|t|l|s': np.array([1., 1., 1., 1., 1., 2., 3., 4.])}   # 1월은 상수 예측 → IC 0(결정 A)
        ic = mt.outer_monthly_ic(vk, preds)
        jan = ic[ic.target_month.eq('2010-01')]
        self.assertTrue(jan.ic.eq(0.0).all() and jan.reason.eq('constant_score').all())
        self.assertTrue(ic.ic.notna().all())


if __name__ == '__main__':
    unittest.main()
