"""순위 상관·달력 목표 결합·2단계 공통 표본 검증. 합성 자료만 사용한다."""
import unittest
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.evaluation.rank_ic import rank_ic_for_month, join_validation_actuals, mean_rank_ic_on_common_months
from src.preprocessing.panel import attach_target_month, join_training_targets


class RankICContractTests(unittest.TestCase):
    def frame(self, scores=(1.,2.,3.,4.), targets=(1.,2.,3.,4.)):
        n=len(scores)
        return pd.DataFrame({'stock_id':[f'{i:06d}' for i in range(n)],'target_month':['2021-01']*n,
                             'model':['feature_score']*n,'candidate':['equal']*n,'fold':['development']*n,
                             'score':scores,'y':targets})

    def test_same_reverse_and_tied_ranks(self):
        for a,b,expected in [([1.,2.,3.,4.],[1.,2.,3.,4.],1.),([1.,2.,3.,4.],[4.,3.,2.,1.],-1.),
                              ([1.,1.,2.,3.],[2.,1.,2.,3.],spearmanr([1,1,2,3],[2,1,2,3]).statistic)]:
            row=rank_ic_for_month(self.frame(a,b),score_column='score').iloc[0]
            self.assertAlmostEqual(row.ic,expected);self.assertEqual(row.reason,'')

    def test_unavailable_months_not_zero(self):
        for a,b,reason in [([1.,2.],[1.,2.],'too_few_pairs'),([1.,1.,1.],[1.,2.,3.],'constant_score'),
                            ([1.,2.,3.],[1.,1.,1.],'constant_target'),([1.,2.,np.inf],[1.,2.,3.],'nonfinite_input')]:
            row=rank_ic_for_month(self.frame(a,b),score_column='score').iloc[0]
            self.assertTrue(pd.isna(row.ic));self.assertEqual(row.reason,reason)

    def test_nan_pairs_and_actual_zero_ic(self):
        row=rank_ic_for_month(self.frame([1.,2.,3.,4.,np.nan],[2.,4.,1.,3.,1.]),score_column='score').iloc[0]
        self.assertEqual(row.n,4);self.assertAlmostEqual(row.ic,0.);self.assertEqual(row.reason,'')

    def test_group_duplicate_test_period_and_rules_rejected(self):
        f=self.frame()
        with self.assertRaises(ValueError):rank_ic_for_month(pd.concat([f,f.iloc[:1]]),score_column='score')
        for col,value in [('candidate','other'),('target_month','2021-02')]:
            t=f.copy();t.loc[0,col]=value
            with self.assertRaises(ValueError):rank_ic_for_month(t,score_column='score')
        f['target_month']='2022-01'
        with self.assertRaises(PermissionError):rank_ic_for_month(f,score_column='score')
        with self.assertRaises(ValueError):rank_ic_for_month(self.frame(),min_valid_stocks=2,score_column='score')

    def test_ic_columns_and_model_prediction_default(self):
        from src.contracts import IC_COLUMNS
        t=self.frame().rename(columns={'score':'yhat'})
        out=rank_ic_for_month(t)
        self.assertEqual(tuple(out.columns),IC_COLUMNS)
        self.assertAlmostEqual(out.iloc[0].ic,1.)

    def test_prediction_join_is_grouped_and_missing_kept(self):
        t=self.frame().rename(columns={'score':'yhat'});actual=t[['stock_id','target_month','y']].iloc[:3]
        p=t.drop(columns='y');other=p.copy();other['candidate']='second'
        joined=join_validation_actuals(pd.concat([p,other]),actual)
        self.assertEqual(len(joined),8);self.assertEqual(joined.y.isna().sum(),2)
        with self.assertRaises(ValueError):join_validation_actuals(p,pd.concat([actual,actual.iloc[:1]]))

    def test_common_months_equal_weight_and_no_common(self):
        f=pd.DataFrame({'candidate':['a','a','a','b','b','b'],'fold':['dev']*6,
                        'target_month':['2021-01','2021-02','2021-03']*2,
                        'ic':[.1,.3,.9,.2,.4,np.nan],'n':[3,100,200,3,100,200]})
        out=mean_rank_ic_on_common_months(f,comparison_ids=['a','b']).set_index('candidate')
        self.assertAlmostEqual(out.loc['a','mean_ic'],.2);self.assertEqual(out.loc['a','valid_months'],2)
        f['ic']=np.nan
        self.assertTrue(mean_rank_ic_on_common_months(f,comparison_ids=['a','b']).reason.eq('no_common_valid_months').all())

    def test_calendar_target_join_gaps_and_missing(self):
        f=pd.DataFrame({'stock_id':['000001']*2,'base_month':['2020-12','2021-02'],'h1':[1.,2.]})
        r=pd.DataFrame({'stock_id':['000001']*2,'month':['2021-01','2021-02'],'return':[.1,.9]})
        r.attrs['units']={'return':'decimal'}
        out=join_training_targets(f,r)
        self.assertEqual(out.target_month.tolist(),['2021-01','2021-03'])
        self.assertAlmostEqual(out.loc[0,'y'],.1);self.assertTrue(pd.isna(out.loc[1,'y']))
        self.assertEqual(out.loc[1,'target_exclusion_reason'],'target_missing')
        pd.testing.assert_series_equal(out.h1,f.h1)

    def test_warmup_and_test_targets_blocked(self):
        for month in ('2000-11','2021-12','2022-01'):
            with self.assertRaises(PermissionError):
                attach_target_month(pd.DataFrame({'stock_id':['000001'],'base_month':[month]}))
        f=pd.DataFrame({'stock_id':['000001'],'base_month':['2020-12']})
        r=pd.DataFrame({'stock_id':['000001'],'month':['2022-01'],'return':[.1]})
        r.attrs['units']={'return':'decimal'}
        with self.assertRaises(PermissionError):join_training_targets(f,r)


class HACMeanTests(unittest.TestCase):
    def test_hac_matches_independent_bartlett_matrix_and_t_reference(self):
        from src.evaluation.rank_ic import newey_west_mean_test
        from scipy.stats import t
        rng=np.random.default_rng(42);x=rng.normal(size=251)
        for i in range(1,len(x)):x[i]+=.6*x[i-1]
        result=newey_west_mean_test(x)
        u=x-x.mean();distance=np.abs(np.arange(251)[:,None]-np.arange(251)[None,:])
        kernel=np.where(distance<=4,1-distance/5,0.)
        expected=float(np.sqrt(u@kernel@u)/251)
        self.assertEqual(result['lag'],4);self.assertEqual(result['df'],250)
        self.assertAlmostEqual(result['hac_se'],expected)
        self.assertAlmostEqual(result['statistic'],x.mean()/expected)
        self.assertAlmostEqual(result['p_two_sided'],2*t.sf(abs(x.mean()/expected),250))
        self.assertAlmostEqual(result['ci95_high'],x.mean()+t.ppf(.975,250)*expected)
        self.assertNotAlmostEqual(result['hac_se'],result['usual_se'])

    def test_missing_and_degenerate_series_not_silently_filled(self):
        from src.evaluation.rank_ic import newey_west_mean_test
        for x in ([1.],[1.,np.nan,2.],[[1.,2.],[3.,4.]]):
            with self.assertRaises(ValueError):newey_west_mean_test(x)
        result=newey_west_mean_test(np.ones(251))
        self.assertTrue(np.isnan(result['p_two_sided']))
        self.assertEqual(result['reason'],'zero_hac_standard_error')

    def test_holm_monotonicity_original_order_and_fixed_family(self):
        from src.evaluation.rank_ic import holm_bonferroni
        p=np.array([.04,.001,.02,.5,.03,.8,.6,.4,.01])
        result=holm_bonferroni(p)
        np.testing.assert_allclose(result,[.20,.009,.14,1.,.18,1.,1.,1.,.08])
        order=np.argsort(p)
        self.assertTrue((np.diff(result[order])>=0).all())
        for bad in ([.1,np.nan],[-.1,.4],[1.1],[]):
            with self.assertRaises(ValueError):holm_bonferroni(bad)

    def monthly(self):
        from src.evaluation.h1_analysis import HYPOTHESIS_CANDIDATES
        rng=np.random.default_rng(21);rows=[]
        months=pd.period_range('2001-01','2021-12',freq='M')
        for k,name in enumerate(HYPOTHESIS_CANDIDATES):
            values=.03+.1*rng.normal(size=251)+k*.001
            for i,target in enumerate(months):
                rows.append({'candidate':name,'target_month':str(target),'base_month':str(target-1),
                    'n':0 if i==0 else 658+int(i<=173),'ic':np.nan if i==0 else values[i-1],
                    'reason':'too_few_pairs' if i==0 else ''})
        return pd.DataFrame(rows)

    def test_monthly_alignment_fixed_nine_family_and_shuffled_rows(self):
        from src.evaluation.h1_analysis import hypothesis_tables
        f=self.monthly();tables=hypothesis_tables(f)
        self.assertEqual(len(tables['primary_hac_test']),1)
        self.assertEqual(len(tables['delta_hac_tests']),9)
        self.assertEqual(len(tables['monthly_ic_delta']),2510)
        other=hypothesis_tables(f.sample(frac=1,random_state=42))
        for key in tables:
            if key=='excluded_months':continue
            pd.testing.assert_frame_equal(tables[key],other[key])
        for bad in (pd.concat([f,f.iloc[:1]]),f.iloc[1:],f[f.candidate!='equal_winsor']):
            with self.assertRaises(ValueError):hypothesis_tables(bad)

    def test_mean_zero_tests_use_separate_nine_family_excluding_primary(self):
        from src.evaluation.h1_analysis import mean_zero_tests
        from src.evaluation.rank_ic import holm_bonferroni,newey_west_mean_test
        f=self.monthly();f=f[f.target_month.ne('2001-01')]
        result=mean_zero_tests(f)
        primary=result[result.candidate.eq('equal')].iloc[0]
        self.assertTrue(pd.isna(primary.holm_p))
        self.assertAlmostEqual(primary.p_two_sided,newey_west_mean_test(f[f.candidate.eq('equal')].ic)['p_two_sided'])
        alternatives=result[result.candidate.ne('equal')]
        self.assertEqual(len(alternatives),9)
        np.testing.assert_allclose(alternatives.holm_p,holm_bonferroni(alternatives.p_two_sided))
        with self.assertRaises(ValueError):mean_zero_tests(f.iloc[1:])


if __name__=='__main__':unittest.main()
