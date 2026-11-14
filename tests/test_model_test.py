"""model_test(v2) 공통부·linear_model 계약 — 합성 자료만 사용(실제 data/ 파일을 읽지 않음).

누수 없음, 조합 간 같은 학습 행, 테스트 기간 차단, 동률 규칙, 예측 행 = 수익률 있는 모든 종목,
그리고 선형 모델 구현(HuberRegressor·Ridge 일치)을 확인한다.
"""
from pathlib import Path
import copy
import unittest

import numpy as np
import pandas as pd

from src.evaluation import model_test as mt
from src.features.h3 import h3_lambda_candidates
from src.models import linear
from src.pipeline import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / 'config/experiment.yaml')
LABELS = list(h3_lambda_candidates().label)
DECAY_OF = dict(zip(h3_lambda_candidates().label, h3_lambda_candidates().decay))


def synthetic(n_stocks=24, seed=0, returns_override=None):
    """기준월 2000-12~2021-11 합성 특징·σ 격자·수익률. 일부 행은 특징이 없거나 비적격, 일부 수익률 결측."""
    rng = np.random.default_rng(seed)
    base = pd.period_range('2000-12', '2021-11', freq='M').astype(str)
    stocks = [f'{i:06d}' for i in range(10, 10 + n_stocks)]
    keys = pd.MultiIndex.from_product([stocks, base], names=['stock_id', 'base_month']).to_frame(index=False)
    n = len(keys)
    comp = keys.copy()
    comp['eligible'] = pd.array(rng.random(n) > .03, dtype='boolean')
    for p in ('size', 'bm', 'mom'):
        comp[p] = rng.normal(size=n)
        comp[p + '_available'] = rng.random(n) > .05
        comp.loc[~comp[p + '_available'], p] = np.nan
    grid = keys.copy()
    grid['eligible'] = comp['eligible']
    grid['vol_available'] = rng.random(n) > .05
    base_sigma = rng.lognormal(-2, .3, n)
    for j, lab in enumerate(LABELS):
        grid['sigma_' + lab] = np.where(grid.vol_available, base_sigma * (1 + .02 * j * rng.random(n)), np.nan)
    months = pd.period_range('2000-01', '2021-12', freq='M').astype(str)
    r = pd.MultiIndex.from_product([stocks, months], names=['stock_id', 'month']).to_frame(index=False)
    signal = comp.set_index(['stock_id', 'base_month'])
    r['return'] = rng.normal(0, .1, len(r))
    nxt = (pd.PeriodIndex(signal.index.get_level_values(1), freq='M') + 1).astype(str)
    effect = pd.Series(.02 * signal['bm'].fillna(0).to_numpy(), index=pd.MultiIndex.from_arrays(
        [signal.index.get_level_values(0), nxt]))
    r['return'] += r.set_index(['stock_id', 'month']).index.map(effect).fillna(0).to_numpy()
    r.loc[rng.random(len(r)) < .03, 'return'] = np.nan
    r['return'] = r['return'].clip(lower=-.95)
    if returns_override is not None:
        r = returns_override(r)
    return comp, grid, r


def prepared_from(comp, grid, r):
    data = mt.build_model_data(comp, grid, r, LABELS, units={'return': 'decimal'})
    return {'data': data, 'sigma_labels': LABELS, 'lambda_candidates': h3_lambda_candidates(),
            'checks': {'sigma_decay097_equals_h3_parquet': True}, 'inputs': {}}


SETTINGS = [('ols', 'raw', 'mse'), ('ridge', 'winsor_1_99', 'huber_q999'), ('ridge', 'monthly_zscore', 'huber_sigma135')]
SETS = {k: v for k, v in mt.input_sets().items() if k in ('ind:bm', 'ind:size+sigma', 'idx:h1+sigma')}


class GridAndRuleTests(unittest.TestCase):
    def test_config_approved_and_17_sets(self):
        mt.check_grid(CONFIG)
        sets = mt.input_sets()
        self.assertEqual(len(sets), 17)
        self.assertEqual(sum(k.startswith('ind:') for k in sets), 15)
        folds = mt.plan_folds(CONFIG['MODEL_GRID'])
        self.assertEqual([f['valid_year'] for f in folds], list(range(2006, 2022)))
        self.assertEqual(folds[0]['train_years'], (2003, 2004))
        self.assertEqual(folds[-1]['train_years'], (2003, 2019))
        self.assertEqual([f['fold'] for f in folds if f['min_years_exception']], [1, 2, 3])

    def test_changed_grid_rejected(self):
        bad = copy.deepcopy(CONFIG)
        bad['MODEL_GRID']['ridge_lambdas'] = [0.1, 1]
        with self.assertRaises(ValueError):
            mt.check_grid(bad)
        bad = copy.deepcopy(CONFIG)
        bad['MODEL_GRID']['train_start_target_year'] = 2001
        with self.assertRaises(PermissionError):
            mt.check_grid(bad)

    def test_tie_rule_ridge_larger_then_longer_span(self):
        r = lambda ic, lam, decay: {'inner_ic': ic, 'ridge_lambda': lam, 'decay': decay}
        chosen, ties = mt.choose([r(.05, .01, .97), r(.05, 1.0, .85), r(.05, 1.0, .92), r(.04, 10.0, .97), r(np.nan, 10.0, .97)])
        self.assertEqual((chosen['ridge_lambda'], chosen['decay'], ties), (1.0, .92, 3))
        chosen, ties = mt.choose([r(.05, None, .85), r(.05, None, .97), r(.01, None, .96)])
        self.assertEqual((chosen['decay'], ties), (.97, 2))
        chosen, ties = mt.choose([r(.05, .1, .97), r(.0500000001, .0001, .85)])
        self.assertEqual((chosen['ridge_lambda'], ties), (.0001, 1))   # 허용 오차로 동률을 만들지 않음

    def test_rank_pm1_average_ties_open_interval(self):
        v = pd.Series([3., 1., 1., np.nan, 5., 2.])
        member = np.array([True, True, True, False, True, False])
        months = pd.Series(['a', 'a', 'a', 'a', 'b', 'b'])
        out = mt.rank_pm1(v, member, months)
        np.testing.assert_allclose(out[:3], [2 * 3 / 4 - 1, 2 * 1.5 / 4 - 1, 2 * 1.5 / 4 - 1])
        self.assertTrue(np.isnan(out[3]) and np.isnan(out[5]))
        self.assertEqual(out[4], 0.0)

    def test_monthly_ic_matches_scipy(self):
        from scipy.stats import spearmanr
        rng = np.random.default_rng(1)
        m = np.repeat(['2006-01', '2006-02'], 50)
        s, y = rng.normal(size=100), rng.normal(size=100)
        s[3], y[60] = np.nan, np.nan
        s[10:14] = 0.5
        ic = mt.monthly_ic(m, s, y)
        for month, row in ic.iterrows():
            k = (m == month) & np.isfinite(s) & np.isfinite(y)
            self.assertAlmostEqual(row.ic, spearmanr(s[k], y[k]).statistic, places=13)
        self.assertEqual(mt.monthly_ic(np.array(['x'] * 4), np.ones(4), np.arange(4.)).reason.iloc[0], 'constant_score')


class LinearModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.X = rng.uniform(-1, 1, (800, 3))
        self.y = self.X @ [.3, -.2, .1] + rng.standard_t(2, 800) * .1

    def test_ridge_mse_equals_sklearn_alpha_lambda_n(self):
        from sklearn.linear_model import Ridge
        f = linear.fit_linear(self.X, self.y, model='ridge', loss='mse', ridge_lambda=.1)
        ref = Ridge(alpha=.1 * 800).fit(self.X, self.y)
        np.testing.assert_allclose(f['coef'], ref.coef_, atol=1e-12)

    def test_huber_large_delta_equals_mse(self):
        for model, lam in (('ols', None), ('ridge', .01)):
            a = linear.fit_linear(self.X, self.y, model=model, loss='mse', ridge_lambda=lam)
            b = linear.fit_linear(self.X, self.y, model=model, loss='huber_q999', ridge_lambda=lam, delta=1e6)
            np.testing.assert_allclose(a['coef'], b['coef'], atol=1e-10)

    def test_ols_huber2_equals_huber_regressor(self):
        self.assertLess(linear.check_huber_regressor_agreement(self.X, self.y), 1e-6)

    def test_huber_q999_delta_is_monthly_deviation_quantile(self):
        y = np.array([1., 3., 10., 20.])
        months = np.array(['a', 'a', 'b', 'b'])
        self.assertAlmostEqual(linear.huber_delta_q999(y, months), 5.0)

    def test_unapproved_values_rejected(self):
        with self.assertRaises(ValueError):
            linear.fit_linear(self.X, self.y, model='ridge', loss='mse', ridge_lambda=.5)
        with self.assertRaises(ValueError):
            linear.fit_linear(self.X, self.y, model='ols', loss='huber_q999')


class PipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folds = [f for f in mt.plan_folds(CONFIG['MODEL_GRID']) if f['fold'] in (1, 9)]
        cls.prepared = prepared_from(*synthetic())
        cls.result = mt.run_linear(cls.prepared, CONFIG, folds=cls.folds, settings=SETTINGS, sets=SETS, log=lambda *_: None)

    def test_test_period_blocked(self):
        comp, grid, r = synthetic()
        extra = r[r.month.eq('2021-12')].assign(month='2022-01')
        with self.assertRaises(PermissionError):
            prepared_from(comp, grid, pd.concat([r, extra], ignore_index=True))
        late = comp[comp.base_month.eq('2021-11')].assign(base_month='2021-12')
        data = mt.build_model_data(pd.concat([comp, late]), grid, r, LABELS, units={'return': 'decimal'})
        self.assertTrue(data.target_month.le('2021-12').all())          # 기준월 2021-12 행은 입력에서 잘려 들어오지 않음
        with self.assertRaises(PermissionError):
            mt.verify_no_test_access(stage='validate', declared_target_years=[2021, 2022], source_scope_verified=True)

    def test_training_rows_and_targets(self):
        d = self.prepared['data']
        expect = d.h1_available & d.y.notna() & np.isfinite(d.filter(like='x_sigma_').iloc[:, 0])
        self.assertTrue(d.train.equals(expect))
        for t in mt.TARGETS:
            self.assertTrue(np.isfinite(d.loc[d.train, 't_' + t]).all() and d.loc[~d.train, 't_' + t].isna().all())
        z = d[d.train].groupby('target_month').t_monthly_zscore.agg(['mean', 'std'])
        np.testing.assert_allclose(z['mean'], 0, atol=1e-12)
        np.testing.assert_allclose(z['std'], 1, atol=1e-12)

    def test_same_training_rows_and_all_stocks_predicted(self):
        ic = mt.outer_monthly_ic(self.result['val_keys'], self.result['predictions'])
        checks = mt.end_checks(self.prepared, self.result, ic, full_run=False)
        self.assertTrue(all(v for k, v in checks.items() if k != 'scipy_independent_max_error'), checks)
        sel = self.result['selections']
        self.assertEqual(sel.groupby('fold').n_train.nunique().max(), 1)
        d = self.prepared['data']
        filled = d.predict & ~d.train & d.year.isin([2006, 2014])
        self.assertGreater(int(filled.sum()), 0)                        # 특징이 빠졌어도 수익률이 있으면 예측
        self.assertEqual(len(self.result['val_keys']), int((d.predict & d.year.isin([2006, 2014])).sum()))

    def test_no_leakage_from_validation_or_later_years(self):
        """바깥 검증 연도와 그 뒤의 수익률을 바꿔도 Fold의 선택·예측·지수가 같다."""
        def scramble(r):
            r = r.copy()
            later = r.month.ge('2006-01') & r['return'].notna()        # Fold 1: 검증 2006 이후(결측 위치는 유지)
            r.loc[later, 'return'] = np.random.default_rng(9).normal(0, .3, int(later.sum())).clip(-.95)
            return r
        fold1 = [self.folds[0]]
        base = mt.run_linear(self.prepared, CONFIG, folds=fold1, settings=SETTINGS, sets=SETS, log=lambda *_: None)
        changed = prepared_from(*synthetic(returns_override=scramble))
        other = mt.run_linear(changed, CONFIG, folds=fold1, settings=SETTINGS, sets=SETS, log=lambda *_: None)
        cols = ['config', 'h3_label', 'ridge_lambda', 'inner_ic', 'train_ic', 'intercept']
        a = base['selections'].assign(config=lambda f: f[mt.CONFIG_KEYS].astype(str).agg('|'.join, axis=1))[cols]
        b = other['selections'].assign(config=lambda f: f[mt.CONFIG_KEYS].astype(str).agg('|'.join, axis=1))[cols]
        pd.testing.assert_frame_equal(a, b)
        for cid in base['predictions']:
            np.testing.assert_array_equal(base['predictions'][cid], other['predictions'][cid])
        s1a, s1b = base['stage1'], other['stage1']
        early = s1a.index_year.le(2006)
        pd.testing.assert_frame_equal(s1a[early].reset_index(drop=True), s1b[early].reset_index(drop=True))

    def test_parallel_and_parts_equal_single_run(self):
        """Fold 병렬 계산·(모델 × 손실) 셀별 계산을 합친 결과가 한 번에 돌린 결과와 같다."""
        par = mt.run_linear(self.prepared, CONFIG, folds=self.folds, settings=SETTINGS, sets=SETS, workers=2, log=lambda *_: None)
        parts = [mt.run_linear(self.prepared, CONFIG, folds=self.folds, settings=[s], sets=SETS, log=lambda *_: None) for s in SETTINGS]
        merged = mt.merge_results(parts)
        cols = [*mt.CONFIG_KEYS, 'fold', 'h3_label', 'ridge_lambda', 'inner_ic', 'train_ic', 'intercept']
        key = [*mt.CONFIG_KEYS, 'fold']
        ref = self.result['selections'][cols].sort_values(key).reset_index(drop=True)
        for other in (par, merged):
            pd.testing.assert_frame_equal(ref, other['selections'][cols].sort_values(key).reset_index(drop=True))
            self.assertEqual(set(other['predictions']), set(self.result['predictions']))
            for cid, v in self.result['predictions'].items():
                np.testing.assert_array_equal(v, other['predictions'][cid])
        with self.assertRaises(ValueError):
            mt.merge_results([parts[0], parts[0]])

    def test_part_summary_progress_and_finalize(self):
        """셀 요약 표, 병렬 진행 메시지(큐 경유), 마무리 저장(부분 실행은 거부, 허용 시 새 폴더에 표·run.json)."""
        import tempfile
        from datetime import datetime
        lines = []
        part = mt.run_linear(self.prepared, CONFIG, folds=self.folds, settings=SETTINGS, sets=SETS, workers=2, log=lines.append)
        self.assertTrue(any('조합' in s and '남은 예상' in s for s in lines))
        self.assertEqual(sum(s.startswith('Fold') for s in lines), len(self.folds))
        summary = mt.part_summary(part)
        self.assertEqual(len(summary), len(SETS) * len({t for _, t, _ in SETTINGS}))
        self.assertTrue({'valid_ic_full', 'valid_ic_common', 'train_ic', 'train_minus_valid'} <= set(summary))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                mt.finalize_linear(ROOT, CONFIG, self.prepared, part, started=datetime.now(), out_root=Path(tmp), log=lambda *_: None)
            out = mt.finalize_linear(ROOT, CONFIG, self.prepared, part, started=datetime.now(), out_root=Path(tmp),
                                     require_complete=False, log=lambda *_: None)
            self.assertEqual(out['status'], 'PASSED')
            files = {p.name for p in Path(out['run_dir']).iterdir()}
            self.assertTrue({'run.json', 'performance_summary.csv', 'validation_predictions.parquet'} <= files)

    def test_inner_scorer_equals_monthly_ic(self):
        rng = np.random.default_rng(5)
        m = np.repeat([f'2006-{i:02d}' for i in range(1, 13)], 40)
        y = rng.normal(size=len(m))
        y[::7] = 0.0
        sc = mt.InnerScorer(m, y)
        for _ in range(5):
            s = np.round(rng.normal(size=len(m)), 1)
            self.assertEqual(sc.score(s)[0], mt.mean_ic(mt.monthly_ic(m, s, y)))

    def test_index_stage1_uses_prior_years_only(self):
        s1 = self.result['stage1']
        self.assertEqual(sorted(s1.index_year.unique()), list(range(2003, 2022)))
        row = s1[s1.index_year.eq(2003)].iloc[0]
        self.assertEqual((row.stage1_train_years, row.stage1_inner_valid_year), ('2001~2001', 2002))


if __name__ == '__main__':
    unittest.main()
