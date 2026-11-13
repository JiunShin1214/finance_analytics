"""확장 창 Fold·공통 인터페이스·OLS·Ridge 계약(묶음 1). 합성 자료만 사용한다."""
import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation import expanding_window as ew
from src.models import interface, ols, ridge

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config/experiment.yaml').read_text(encoding='utf-8'))


def panel(months, stocks=30, seed=0, cols=('a', 'b', 'c')):
    rng = np.random.default_rng(seed)
    rows = [{'stock_id': f'{i:06d}', 'target_month': m} for m in months for i in range(stocks)]
    keys = pd.DataFrame(rows)
    X = pd.DataFrame(rng.normal(size=(len(keys), len(cols))), columns=list(cols))
    y = X.to_numpy() @ np.array([0.3, -0.2, 0.1][:len(cols)]) + rng.normal(scale=0.5, size=len(keys))
    return keys, X, pd.Series(y)


def months(start, end):
    return pd.period_range(start, end, freq='M').astype(str).tolist()


FOLD = interface.FoldSpec(fold='2006', train_start='2001-01', train_end='2005-12', valid_start='2006-01', valid_end='2006-12')
IDENTITY = {'kind': 'identity', 'columns': ['a', 'b', 'c']}


class FoldTests(unittest.TestCase):
    def test_sixteen_expanding_folds(self):
        folds = ew.plan_annual_expanding_folds(config=CONFIG)
        self.assertEqual(len(folds), 16)
        self.assertEqual(folds[0], FOLD)
        self.assertEqual(folds[-1], dict(fold='2021', train_start='2001-01', train_end='2020-12',
                                         valid_start='2021-01', valid_end='2021-12'))
        for prev, cur in zip(folds, folds[1:]):
            self.assertEqual(int(cur['train_end'][:4]), int(prev['train_end'][:4]) + 1)
            self.assertLess(cur['train_end'], cur['valid_start'])

    def test_missing_validation_month_rejected(self):
        available = [m for m in months('2001-01', '2021-12') if m != '2013-07']
        with self.assertRaisesRegex(ValueError, '2013-07'):
            ew.plan_annual_expanding_folds(config=CONFIG, available_target_months=available)

    def test_folds_cannot_reach_test_period(self):
        bad = copy.deepcopy(CONFIG)
        bad['FOLDS']['last_validation_year'] = 2022
        with self.assertRaises(PermissionError):
            ew.plan_annual_expanding_folds(config=bad)

    def test_inner_split_uses_last_training_year_only(self):
        inner = ew.inner_validation_split(FOLD, years=1)
        self.assertEqual((inner['train_start'], inner['train_end'], inner['valid_start'], inner['valid_end']),
                         ('2001-01', '2004-12', '2005-01', '2005-12'))
        self.assertLess(inner['valid_end'], FOLD['valid_start'])

    def test_period_labels(self):
        f = pd.DataFrame({'target_month': ['2000-05', '2001-01', '2021-12', '2022-01']})
        self.assertEqual(ew.assign_target_month_split(f, config=CONFIG).period.tolist(), ['warmup', 'train', 'train', 'test'])
        with self.assertRaises(ValueError):
            ew.assign_target_month_split(pd.DataFrame({'target_month': ['2026-01']}), config=CONFIG)


class ChronologyTests(unittest.TestCase):
    def setUp(self):
        self.train, _, _ = panel(months('2001-01', '2005-12'), stocks=3)
        self.valid, _, _ = panel(months('2006-01', '2006-12'), stocks=3)

    def test_valid_passes(self):
        ew.validate_fold_chronology(self.train, self.valid, fold=FOLD)

    def test_valid_rows_in_training_rejected(self):
        with self.assertRaises(ValueError):
            ew.validate_fold_chronology(pd.concat([self.train, self.valid.iloc[:1]]), self.valid, fold=FOLD)

    def test_test_period_rejected(self):
        late = self.valid.assign(target_month='2022-01').iloc[:1]
        with self.assertRaises(PermissionError):
            ew.validate_fold_chronology(self.train, late, fold=FOLD)

    def test_access_guard(self):
        ew.verify_no_test_access(stage='validate', declared_target_years=[2006, 2021], source_scope_verified=True)
        for kwargs in ({'stage': 'final-test', 'declared_target_years': [2021], 'source_scope_verified': True},
                       {'stage': 'validate', 'declared_target_years': [2022], 'source_scope_verified': True},
                       {'stage': 'validate', 'declared_target_years': [2010], 'source_scope_verified': False}):
            with self.assertRaises(PermissionError):
                ew.verify_no_test_access(**kwargs)


class InterfaceTests(unittest.TestCase):
    def test_inputs_checked(self):
        tk, tX, ty = panel(months('2001-01', '2005-12'), stocks=4)
        vk, vX, _ = panel(months('2006-01', '2006-12'), stocks=4)
        interface.validate_fold_inputs(tX, ty, vX, tk, vk, fold=FOLD)
        bad = vX.copy(); bad.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            interface.validate_fold_inputs(tX, ty, bad, tk, vk, fold=FOLD)
        with self.assertRaises(ValueError):
            interface.validate_fold_inputs(tX, ty, vX[['b', 'a', 'c']], tk, vk, fold=FOLD)

    def test_prediction_frame(self):
        vk, _, _ = panel(['2006-01'], stocks=3)
        out = interface.format_prediction_frame([1, 2, 3], vk, model='ols', candidate='ols', fold='2006')
        self.assertEqual(list(out.columns), ['model', 'candidate', 'fold', 'stock_id', 'target_month', 'yhat'])
        with self.assertRaises(ValueError):
            interface.format_prediction_frame([1, 2, np.inf], vk, model='ols', candidate='ols', fold='2006')
        with self.assertRaises(ValueError):
            interface.format_prediction_frame([1, 2, 3], pd.concat([vk.iloc[:2], vk.iloc[:1]]), model='ols', candidate='ols', fold='2006')

    def test_transformer_identity_and_train_only_standardize(self):
        _, X, _ = panel(['2001-01'], stocks=50)
        t = interface.fit_fold_transformer(X, transformer_spec=None)
        self.assertIs(interface.apply_fold_transformer(t, X), X)
        s = interface.fit_fold_transformer(X, transformer_spec={'kind': 'standardize'})
        z = interface.apply_fold_transformer(s, X)
        np.testing.assert_allclose(z.mean(), 0, atol=1e-12)
        np.testing.assert_allclose(z.std(ddof=0), 1, atol=1e-12)
        _, V, _ = panel(['2006-01'], stocks=50, seed=9)
        np.testing.assert_allclose(interface.apply_fold_transformer(s, V), (V - X.mean()) / X.std(ddof=0))  # 학습 통계만 사용
        with self.assertRaises(ValueError):
            interface.fit_fold_transformer(X, transformer_spec={'kind': 'minmax'})


class OlsTests(unittest.TestCase):
    def test_matches_least_squares_and_predicts(self):
        tk, tX, ty = panel(months('2001-01', '2005-12'))
        cand = ols.plan_ols_config()
        model = ols.fit_ols_fold(tX, ty, tk, fold=FOLD, candidate=cand, transformer=IDENTITY)
        design = np.column_stack([np.ones(len(tX)), tX.to_numpy()])
        ref = np.linalg.lstsq(design, ty.to_numpy(), rcond=None)[0]
        np.testing.assert_allclose([model['intercept'], *model['coef']], ref, atol=1e-12)
        vk, vX, _ = panel(months('2006-01', '2006-12'), seed=1)
        pred = ols.predict_ols_fold(model, vX, vk, fold=FOLD, candidate=cand, transformer=IDENTITY)
        np.testing.assert_allclose(pred.yhat, ref[0] + vX.to_numpy() @ ref[1:], atol=1e-12)

    def test_requires_intercept_and_fold_rows(self):
        with self.assertRaises(ValueError):
            ols.plan_ols_config(fit_intercept=False)
        tk, tX, ty = panel(months('2006-01', '2006-12'))
        with self.assertRaises(ValueError):
            ols.fit_ols_fold(tX, ty, tk, fold=FOLD, candidate=ols.plan_ols_config(), transformer=IDENTITY)

    def test_perfect_collinearity_rejected(self):
        tk, tX, ty = panel(months('2001-01', '2002-12'))
        tX = tX.assign(c=(tX.a + tX.b) / 2)  # 세 특징 + 그 합성을 함께 넣는 경우
        with self.assertRaises(ValueError):
            ols.fit_ols_fold(tX, ty, tk, fold=FOLD, candidate=ols.plan_ols_config(), transformer=IDENTITY)


class RidgeTests(unittest.TestCase):
    def setUp(self):
        self.cands = ridge.enumerate_ridge_candidates(config=CONFIG)

    def test_candidates_from_config(self):
        self.assertEqual([c['parameters']['lambda'] for c in self.cands], [0.001, 0.01, 0.1, 1.0, 10.0])
        bad = copy.deepcopy(CONFIG); bad['RIDGE']['lambdas'] = [0.01, 0.1, 1, 10, 100]
        with self.assertRaises(ValueError):
            ridge.enumerate_ridge_candidates(config=bad)

    def test_same_lambda_same_strength_regardless_of_rows(self):
        # 행을 두 배로 늘려도(같은 자료 반복) 같은 λ면 계수가 같아야 한다: alpha = λ × 행 수.
        tk, tX, ty = panel(months('2001-01', '2002-12'))
        cand = self.cands[3]  # λ = 1, 효과가 큰 값
        one = ridge.fit_ridge_fold(tX, ty, tk, fold=FOLD, candidate=cand, transformer=IDENTITY)
        tk2 = pd.concat([tk, tk.assign(stock_id=tk.stock_id.str.replace('0000', '9900'))], ignore_index=True)
        two = ridge.fit_ridge_fold(pd.concat([tX, tX], ignore_index=True), pd.concat([ty, ty], ignore_index=True), tk2,
                                   fold=FOLD, candidate=cand, transformer=IDENTITY)
        np.testing.assert_allclose(one['coef'], two['coef'], atol=1e-10)
        self.assertAlmostEqual(two['alpha'], 2 * one['alpha'])
        ref = ols.fit_ols_fold(tX, ty, tk, fold=FOLD, candidate=ols.plan_ols_config(), transformer=IDENTITY)
        self.assertLess(np.abs(one['coef']).sum(), np.abs(ref['coef']).sum())  # 실제로 줄어듦

    def test_tiny_lambda_close_to_ols(self):
        tk, tX, ty = panel(months('2001-01', '2002-12'))
        r = ridge.fit_ridge_fold(tX, ty, tk, fold=FOLD, candidate=self.cands[0], transformer=IDENTITY)
        o = ols.fit_ols_fold(tX, ty, tk, fold=FOLD, candidate=ols.plan_ols_config(), transformer=IDENTITY)
        np.testing.assert_allclose(r['coef'], o['coef'], rtol=5e-3)

    def test_choose_exact_tie_picks_larger_lambda_and_skips_failures(self):
        rows = [{'candidate': c['candidate'], 'target_month': m, 'ic': 0.05}
                for c in self.cands for m in ('2005-01', '2005-02')]
        ic = pd.DataFrame(rows)
        self.assertEqual(ridge.choose_ridge_candidate(ic, self.cands, pd.DataFrame())['parameters']['lambda'], 10.0)
        failures = pd.DataFrame({'candidate': ['ridge_lambda_10']})
        self.assertEqual(ridge.choose_ridge_candidate(ic, self.cands, failures)['parameters']['lambda'], 1.0)
        ic.loc[ic.candidate.eq('ridge_lambda_0.01'), 'ic'] = 0.051
        self.assertEqual(ridge.choose_ridge_candidate(ic, self.cands, pd.DataFrame())['parameters']['lambda'], 0.01)
        with self.assertRaises(ValueError):
            ridge.choose_ridge_candidate(ic, self.cands, pd.DataFrame({'candidate': [c['candidate'] for c in self.cands]}))

    def test_inner_selection_uses_only_last_training_year(self):
        tk, tX, ty = panel(months('2001-01', '2005-12'))
        chosen, monthly = ridge.select_ridge_lambda(tX, ty, tk, fold=FOLD, candidates=self.cands)
        self.assertIn(chosen, self.cands)
        self.assertTrue(monthly.target_month.between('2005-01', '2005-12').all())
        self.assertEqual(monthly.target_month.nunique(), 12)
        self.assertEqual(set(monthly['lambda']), {0.001, 0.01, 0.1, 1.0, 10.0})
        with self.assertRaises(ValueError):  # 바깥 검증 행이 섞이면 거부
            vk, vX, vy = panel(['2006-01'])
            ridge.select_ridge_lambda(pd.concat([tX, vX], ignore_index=True), pd.concat([ty, vy], ignore_index=True),
                                      pd.concat([tk, vk], ignore_index=True), fold=FOLD, candidates=self.cands)

    def test_inner_evaluation_uses_eval_target(self):
        tk, tX, ty = panel(months('2001-01', '2005-12'))
        _, a = ridge.select_ridge_lambda(tX, ty, tk, fold=FOLD, candidates=self.cands)
        _, b = ridge.select_ridge_lambda(tX, ty, tk, fold=FOLD, candidates=self.cands, eval_y=-ty)
        np.testing.assert_allclose(a.ic.to_numpy(), -b.ic.to_numpy(), atol=1e-12)

    def test_standardized_ridge_ignores_input_scale(self):
        # 표준화 후 벌점이면 한 입력의 단위를 10배로 바꿔도 예측이 같다(ESL 3.4.1). 표준화 없이는 달라진다.
        tk, tX, ty = panel(months('2001-01', '2002-12'))
        vk, vX, _ = panel(months('2006-01', '2006-12'), seed=3)
        cand = self.cands[4]  # λ = 10, 벌점이 큰 값
        def fit_predict(X, V, spec):
            t = interface.fit_fold_transformer(X, transformer_spec=spec)
            m = ridge.fit_ridge_fold(X, ty, tk, fold=FOLD, candidate=cand, transformer=t)
            return m, ridge.predict_ridge_fold(m, V, vk, fold=FOLD, candidate=cand, transformer=t).yhat.to_numpy()
        spec = {'kind': 'standardize'}
        m1, a = fit_predict(tX, vX, spec)
        _, b = fit_predict(tX.assign(a=tX.a * 10), vX.assign(a=vX.a * 10), spec)
        np.testing.assert_allclose(a, b, atol=1e-12)
        _, c = fit_predict(tX, vX, None)
        _, d = fit_predict(tX.assign(a=tX.a * 10), vX.assign(a=vX.a * 10), None)
        self.assertGreater(np.abs(c - d).max(), 1e-3)
        # 원래 크기 계수로 다시 계산한 예측이 같다
        np.testing.assert_allclose(m1['intercept_original'] + vX.to_numpy() @ m1['coef_original'], a, atol=1e-12)

    def test_inner_selection_refits_scaler_on_inner_rows(self):
        tk, tX, ty = panel(months('2001-01', '2005-12'))
        shifted = tX.copy()
        shifted.loc[tk.target_month.ge('2005-01').to_numpy(), 'a'] += 100  # 내부 검증 연도 입력만 크게 이동
        _, a = ridge.select_ridge_lambda(tX, ty, tk, fold=FOLD, candidates=self.cands, transformer_spec={'kind': 'standardize'})
        _, b = ridge.select_ridge_lambda(shifted, ty, tk, fold=FOLD, candidates=self.cands, transformer_spec={'kind': 'standardize'})
        # 표준화를 내부 학습 행으로만 맞췄다면 적합은 같고 내부 검증 예측은 모든 종목이 같은 폭으로 이동하므로 IC가 그대로다.
        # 내부 검증 행까지 써서 표준화했다면 표준편차가 바뀌어 벌점과 계수가 달라진다.
        np.testing.assert_allclose(a.ic.to_numpy(), b.ic.to_numpy(), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
