import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.h2.models import specialist as s
from src.h2.models.blend import blend_conditional


def make_frame(months, n=20):
    rows = []
    for m in months:
        base = str(pd.Period(m, freq='M') - 1)
        for i in range(n):
            rows.append([f'{i:06d}', base, m, *[float(i) for _ in s.FEATURES]])
    return pd.DataFrame(rows, columns=s.KEY+list(s.FEATURES))


class TestSpecialist(unittest.TestCase):
    def setUp(self):
        self.train = make_frame(['2006-02','2006-03','2006-04','2006-05','2006-06','2006-07'])
        self.train['y'] = np.arange(len(self.train), dtype=float)
        self.train['fit_member'] = True
        self.forecast = make_frame(['2008-02','2008-03'])
        self.keys = self.forecast[s.KEY].copy()
        self.rates = pd.DataFrame({'feature_month':['2006-01','2006-02','2006-03','2006-04','2006-05','2006-06','2008-01','2008-02'],
                                   'rate_change_pp':[.25,.25,.25,.25,.25,0,.25,0]})

    def call(self, **changes):
        args = dict(train=self.train, forecast=self.forecast, eligible_keys=self.keys, rates=self.rates, valid_year=2008)
        args.update(changes)
        return s.predict_annual_fold(**args)

    def test_feature_specs(self):
        p = s.feature_specs({'PATHS':{'h2':'data/processed/h2.parquet'}})
        self.assertEqual(list(p), list(s.RAW_FEATURES))
        self.assertEqual([k for k,v in p.items() if v['transform']=='input'], ['debt_change'])
        self.assertTrue(all(v['path']=='data/processed/h2.parquet' for v in p.values()))

    @patch.object(s, '_legacy_fit')
    def test_fit_contract_and_blend(self, fit):
        fit.return_value = (np.arange(20,dtype=np.float32), None,500)
        scores, audit = self.call()
        tr, va, features, kind, run, threads, device = fit.call_args.args
        self.assertEqual(len(tr),100)
        self.assertEqual(list(va),list(s.FEATURES))
        self.assertEqual(features,list(s.FEATURES))
        self.assertEqual(kind,'xgb_pairwise')
        self.assertEqual(run,s.frozen_run())
        self.assertEqual(audit['active_target_months'],['2008-02'])
        self.assertEqual(audit['train_months'],5)
        self.assertEqual(audit['train_end'],'2006-12')
        b = self.keys.assign(without_h2=np.arange(40,dtype=float))
        out=blend_conditional(b,scores,self.keys,audit['active_target_months'])
        np.testing.assert_array_equal(out.loc[~out.h2_active,'final_score'],b.loc[b.target_month.eq('2008-03'),'without_h2'])

    @patch.object(s, '_legacy_fit')
    def test_unready(self, fit):
        tr=self.train.copy(); tr.loc[tr.target_month.eq('2006-02'),'fit_member']=False
        scores,audit=self.call(train=tr)
        self.assertTrue(scores.empty); self.assertFalse(audit['ready']); fit.assert_not_called()

    @patch.object(s, '_legacy_fit')
    def test_small_month(self, fit):
        scores,audit=self.call(train=self.train.iloc[1:])
        self.assertFalse(audit['ready']); self.assertTrue(scores.empty);fit.assert_not_called()

    @patch.object(s, '_legacy_fit')
    def test_no_hike_or_unknown(self, fit):
        rates=self.rates[self.rates.feature_month.lt('2008-01')]
        scores,audit=self.call(rates=rates)
        self.assertTrue(audit['ready']); self.assertTrue(scores.empty);fit.assert_not_called()

    def test_future_training_and_test_period(self):
        with self.assertRaises(ValueError): self.call(train=pd.concat([self.train,self.forecast.assign(y=0.,fit_member=True)]))
        with self.assertRaises(ValueError): self.call(valid_year=2022)

    def test_historical_annual_cutoff(self):
        for year in range(2006,2022):
            self.assertEqual(s.annual_train_end(year),f'{year-2}-12')
        for year in (2002,2022,True):
            with self.assertRaises(ValueError): s.annual_train_end(year)

    @patch.object(s, '_legacy_fit')
    def test_inner_validation_year_cannot_enter_fit(self, fit):
        inner=make_frame(['2007-02']).assign(y=0.,fit_member=True)
        with self.assertRaisesRegex(ValueError,'Y-1 reserved'):
            self.call(train=pd.concat([self.train,inner],ignore_index=True))
        fit.assert_not_called()

    def test_labels_and_universe_rejected(self):
        with self.assertRaises(ValueError): self.call(forecast=self.forecast.assign(y=0.))
        with self.assertRaises(ValueError): self.call(forecast=self.forecast.iloc[1:])
        with self.assertRaises(ValueError): self.call(train=pd.concat([self.train,self.train.iloc[:1]]))

    @patch.object(s, '_legacy_fit')
    def test_invalid_fit(self, fit):
        fit.return_value=(np.zeros(20),None,499)
        with self.assertRaises(ValueError): self.call()
        fit.return_value=(np.full(20,np.nan),None,500)
        with self.assertRaises(ValueError): self.call()

if __name__ == '__main__': unittest.main()
