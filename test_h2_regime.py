import unittest
import numpy as np
import pandas as pd
from src.h2.models.regime import build_regime_gate, route_months, active_target_months
from src.h2.models.blend import blend_conditional


class TestRegime(unittest.TestCase):
    def setUp(self):
        self.rates = pd.DataFrame({'feature_month':['2020-12','2021-01','2021-02','2021-03','2021-05'], 'rate_change_pp':[.25,0,-.25,.25,np.nan]})
        self.targets = ['2021-01','2021-02','2021-03','2021-04','2021-05','2021-06']
        self.plan = pd.DataFrame({'valid_year':[2021], 'train_end':['2020-12'], 'train_months':[5], 'min_month_stocks':[20], 'ready':[True]})

    def test_base_month_alignment(self):
        g = build_regime_gate(self.rates,self.targets)
        self.assertEqual(g.base_month.tolist(),['2020-12','2021-01','2021-02','2021-03','2021-04','2021-05'])
        self.assertEqual(g.rate_status.tolist(),['hike','hold','cut','hike','unknown','unknown'])
        self.assertEqual(g.hike_1m.tolist(),[True,False,False,True,False,False])

    def test_future_rate_cannot_change_current_gate(self):
        r=self.rates.copy(); r.loc[r.feature_month.eq('2021-01'),'rate_change_pp']=99
        a=build_regime_gate(self.rates,['2021-01']); b=build_regime_gate(r,['2021-01'])
        pd.testing.assert_frame_equal(a,b)

    def test_ready_routing(self):
        r=route_months(self.rates,self.targets,self.plan)
        self.assertEqual(active_target_months(r),['2021-01','2021-04'])
        self.assertEqual(r.route_reason.tolist(),['active','not_hike','not_hike','active','rate_unknown','rate_unknown'])

    def test_unready_and_missing_plan(self):
        for p in [self.plan.assign(ready=False),self.plan.iloc[:0]]:
            r=route_months(self.rates,self.targets,p)
            self.assertEqual(active_target_months(r),[])
            self.assertEqual(r.route_reason.iloc[0],'model_not_ready')

    def test_reject_invalid_ready_plan(self):
        for p in [self.plan.assign(train_end='2021-01'),self.plan.assign(train_months=4),self.plan.assign(min_month_stocks=19),self.plan.assign(ready='False'),pd.concat([self.plan,self.plan])]:
            with self.assertRaises(ValueError): route_months(self.rates,self.targets,p)

    def test_reject_invalid_rates(self):
        for rates in [pd.concat([self.rates,self.rates]),self.rates.assign(rate_change_pp=np.inf),self.rates.assign(rate_change_pp=True),self.rates.assign(y=1),self.rates.assign(feature_month='2021-13')]:
            with self.assertRaises(ValueError): build_regime_gate(rates,self.targets)
        with self.assertRaises(ValueError): build_regime_gate(self.rates,['2021-01','2021-01'])

    def test_shuffled_rates(self):
        pd.testing.assert_frame_equal(build_regime_gate(self.rates,self.targets),build_regime_gate(self.rates.sample(frac=1,random_state=1),self.targets))

    def test_blend_integration(self):
        routing=route_months(self.rates,self.targets,self.plan)
        active=active_target_months(routing)
        keys=pd.DataFrame([(s,b,t) for b,t in zip(routing.base_month,routing.target_month) for s in ['000001','000002']],columns=['stock_id','base_month','target_month'])
        baseline=keys.assign(without_h2=[1.,2.]*len(self.targets))
        specialist=keys[keys.target_month.isin(active)].assign(h2_score=[2.,1.]*len(active))
        out=blend_conditional(baseline,specialist,keys,active)
        self.assertTrue(out.h2_active.eq(out.target_month.isin(active)).all())
        np.testing.assert_array_equal(out.loc[~out.h2_active,'final_score'],out.loc[~out.h2_active,'without_h2'])
        np.testing.assert_allclose(out.loc[out.h2_active,'final_score'],[105/300,195/300]*len(active))


if __name__=='__main__': unittest.main()
