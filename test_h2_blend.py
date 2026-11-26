"""Synthetic unit tests; no project data or models are read."""
import unittest
import numpy as np
import pandas as pd
from src.h2.models.blend import KEY, blend_conditional


class TestBlend(unittest.TestCase):
    def setUp(self):
        self.b = pd.DataFrame([
            ('000001','2021-07','2021-08',2.), ('000002','2021-07','2021-08',1.),
            ('000003','2021-07','2021-08',1.), ('000001','2021-08','2021-09',-3.),
            ('000002','2021-08','2021-09',8.)], columns=KEY+['without_h2'])
        self.k = self.b[KEY].copy()
        self.e = self.k.iloc[:3].assign(h2_score=[0.,2.,1.])
        self.active = ['2021-08']

    def run_blend(self, b=None, e=None, k=None, active=None):
        return blend_conditional(self.b if b is None else b, self.e if e is None else e,
                                 self.k if k is None else k, self.active if active is None else active)

    def test_exact_reference_formula_and_inactive(self):
        o = self.run_blend()
        expected = (95*np.array([3,1.5,1.5])+5*np.array([1,3,2]))/(100*4)
        np.testing.assert_array_equal(o.final_score.iloc[:3], expected)
        np.testing.assert_array_equal(o.final_score.iloc[3:], self.b.without_h2.iloc[3:])
        self.assertEqual(o.h2_active.tolist(), [True,True,True,False,False])

    def test_constant_specialist_preserves_order(self):
        o = self.run_blend(e=self.e.assign(h2_score=1.))
        np.testing.assert_array_equal(o.final_score.iloc[:3].rank(), self.b.without_h2.iloc[:3].rank())

    def test_no_active(self):
        o = self.run_blend(e=self.e.iloc[:0],active=[])
        np.testing.assert_array_equal(o.final_score,self.b.without_h2)

    def test_reject_incomplete_universe(self):
        for b,e in [(self.b.iloc[:-1], self.e), (self.b,self.e.iloc[:-1])]:
            with self.assertRaises(ValueError): self.run_blend(b=b,e=e)

    def test_reject_labels(self):
        with self.assertRaises(ValueError): self.run_blend(b=self.b.assign(y=0))

    def test_reject_duplicate_and_nonfinite(self):
        with self.assertRaises(ValueError): self.run_blend(e=pd.concat([self.e,self.e.iloc[:1]]))
        with self.assertRaises(ValueError): self.run_blend(e=self.e.assign(h2_score=np.nan))

    def test_reject_extra_month_and_bad_chronology(self):
        with self.assertRaises(ValueError): self.run_blend(active=['2021-10'])
        with self.assertRaises(ValueError): self.run_blend(b=self.b.assign(base_month='2021-01'))

    def test_shuffled_specialist(self):
        np.testing.assert_array_equal(self.run_blend().final_score,self.run_blend(e=self.e.iloc[::-1]).final_score)


if __name__ == '__main__':
    unittest.main()
