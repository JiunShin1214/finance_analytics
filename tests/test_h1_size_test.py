"""H1 Size 사후 진단(A·B·C) 계약. 합성 데이터만 사용한다."""
import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluation import h1_size_test as st
from src.features.h1 import calculate_h1_score
from src.preprocessing.panel import join_training_targets

ROOT = Path(__file__).resolve().parents[1]
BASE = pd.period_range('2000-12', '2002-11', freq='M').astype(str)  # 첫 기준월은 점수 없음 -> 23개 유효 목표월


def synthetic(seed=0, stocks=8):
    rng = np.random.default_rng(seed)
    rows = []
    for month in BASE:
        for i in range(stocks + 1):
            ok = month != '2000-12' and i < stocks
            v = rng.normal(size=3) if ok else [np.nan] * 3
            rows.append({'stock_id': f'{i:06d}', 'base_month': month, 'eligible': i < stocks,
                         'size': v[0], 'bm': v[1], 'mom': v[2]})
    c = pd.DataFrame(rows)
    for name in ('size', 'bm', 'mom'):
        c[name + '_available'] = np.isfinite(c[name])
    h1 = calculate_h1_score(c)
    months = pd.period_range('2001-01', '2002-12', freq='M').astype(str)
    panel = pd.DataFrame([{'stock_id': f'{i:06d}', 'month': m, 'return': rng.normal(0, .1)}
                          for m in months for i in range(stocks + 1)])
    panel.attrs['units'] = {'return': 'decimal'}
    return h1, panel


def run_tables(seed=0):
    h1, panel = synthetic(seed)
    scores, _ = st.candidate_scores(h1)
    monthly, audit, common, err = st.monthly_ic(join_training_targets(scores, panel))
    tables = st.comparison_tables(monthly, expected_months=23, expected_pairs=len(common))
    return h1, scores, monthly, common, tables


class CandidateScoreTests(unittest.TestCase):
    def test_formulas_and_same_membership(self):
        h1, _ = synthetic()
        scores, err = st.candidate_scores(h1)
        m = h1.set_index(['stock_id', 'base_month']).loc[scores.set_index(['stock_id', 'base_month']).index]
        np.testing.assert_allclose(scores.A, m.h1.to_numpy(), equal_nan=True)
        np.testing.assert_allclose(scores.B, ((m.z_bm + m.z_mom) / 2).to_numpy(), equal_nan=True)
        np.testing.assert_allclose(scores.C, ((m.z_size + m.z_bm + m.z_mom) / 3).to_numpy(), equal_nan=True)
        present = np.isfinite(scores[['A', 'B', 'C']].to_numpy())
        self.assertTrue((present == present[:, [0]]).all())
        self.assertLess(err, 1e-12)

    def test_rejects_target_before_scores(self):
        h1, _ = synthetic()
        with self.assertRaises(ValueError):
            st.candidate_scores(h1.assign(y=0.))

    def test_drops_base_months_outside_development(self):
        h1, _ = synthetic()
        extra = h1[h1.base_month.eq('2002-11')].assign(base_month='2021-12')
        scores, _ = st.candidate_scores(pd.concat([h1, extra], ignore_index=True))
        self.assertFalse(scores.base_month.eq('2021-12').any())


class ComparisonTests(unittest.TestCase):
    def test_tables_and_two_test_holm_family(self):
        _, _, monthly, common, t = run_tables()
        self.assertEqual(t['performance_summary'].candidate.tolist(), ['A', 'B', 'C'])
        self.assertTrue(t['performance_summary'].valid_months.eq(23).all())
        d = t['delta_hac_tests']
        self.assertEqual(d.candidate.tolist(), ['B', 'C'])
        self.assertTrue(d.holm_family_size.eq(2).all())
        self.assertTrue((d.holm_p >= d.p_two_sided - 1e-15).all())
        mon = t['monthly_ic_delta']
        np.testing.assert_allclose(mon.delta_b, mon.ic_B - mon.ic_A)
        np.testing.assert_allclose(d.mean_delta_ic, [mon.delta_b.mean(), mon.delta_c.mean()])
        self.assertEqual(mon.n.sum(), len(common))
        self.assertFalse(mon.target_month.eq('2001-01').any())
        self.assertEqual(t['yearly_ic_delta'].year.tolist(), [2001, 2002])
        self.assertEqual(t['yearly_ic_delta'].months.tolist(), [11, 12])

    def test_size_sign_flip_only_changes_size_term(self):
        # Size 항을 0으로 만들면 A=B·C 순위가 같아 ΔIC가 0이다.
        h1, panel = synthetic()
        h1 = h1.assign(z_size=h1.z_size * 0)
        h1['h1'] = (-h1.z_size + h1.z_bm + h1.z_mom) / 3
        scores, _ = st.candidate_scores(h1)
        monthly, *_ = st.monthly_ic(join_training_targets(scores, panel))
        wide = monthly.dropna(subset=['ic']).pivot(index='target_month', columns='candidate', values='ic')
        np.testing.assert_allclose(wide.B, wide.A, atol=1e-12)
        np.testing.assert_allclose(wide.C, wide.A, atol=1e-12)

    def test_guards(self):
        _, _, monthly, common, _ = run_tables()
        with self.assertRaises(ValueError):
            st.comparison_tables(monthly, expected_months=23, expected_pairs=len(common) + 1)
        with self.assertRaises(ValueError):
            st.comparison_tables(monthly, expected_months=22, expected_pairs=len(common))
        extra = monthly[monthly.candidate.eq('C')].assign(candidate='D')
        with self.assertRaises(ValueError):
            st.comparison_tables(pd.concat([monthly, extra]), expected_months=23, expected_pairs=len(common))
        late = monthly.assign(target_month=monthly.target_month.replace('2002-12', '2022-01'))
        with self.assertRaises(PermissionError):
            st.comparison_tables(late, expected_months=23, expected_pairs=len(common))


class InterpretationTests(unittest.TestCase):
    def tables(self, sig_b, delta_b, sig_c, delta_c):
        perf = pd.DataFrame({'candidate': ['A', 'B', 'C'], 'mean_ic': [.01, .01 + delta_b, .01 + delta_c]})
        d = pd.DataFrame({'candidate': ['B', 'C'], 'question': ['necessity', 'direction'], 'label': ['B−A', 'C−A'],
                          'mean_delta_ic': [delta_b, delta_c], 'ci95_low': [-1, -1], 'ci95_high': [1, 1],
                          'p_two_sided': [.01, .5], 'holm_p': [.02, .5], 'significant_holm_05': [sig_b, sig_c]})
        return {'performance_summary': perf, 'delta_hac_tests': d}

    def test_verdict_mapping(self):
        out = st.interpretation(self.tables(True, -.01, False, .02))['conclusions']
        self.assertEqual(out['necessity']['verdict'], '근거 있음')
        self.assertEqual(out['direction']['verdict'], '근거 부족')
        out = st.interpretation(self.tables(True, .01, True, .01))['conclusions']
        self.assertEqual(out['necessity']['verdict'], '추가 검증 필요')
        self.assertEqual(out['direction']['verdict'], '추가 검증 필요')

    def test_never_changes_h1(self):
        out = st.interpretation(self.tables(True, .05, True, .05))
        self.assertIn('자동 변경 없음', out['h1_change'])


class SettingsTests(unittest.TestCase):
    def test_project_config_passes_and_changes_fail(self):
        config = yaml.safe_load((ROOT / 'config/experiment.yaml').read_text(encoding='utf-8'))
        st.check_settings(config)
        for key, value in (('holm_family', 'nine'), ('mean_zero_tests', True), ('candidates', ['A', 'B', 'C', 'D']),
                           ('selected_candidate', 'B')):
            changed = copy.deepcopy(config)
            changed['H1_SIZE_TEST'][key] = value
            with self.assertRaises(PermissionError):
                st.check_settings(changed)


if __name__ == '__main__':
    unittest.main()
