"""H1 기간·입력·공통 집합·핵심 수식 계약 검증(합성 자료만 사용)."""
import unittest
import numpy as np
import pandas as pd
from src.features.h1 import (calculate_momentum_11m, calculate_size_bm, calculate_h1_score,
                             cross_sectional_zscore, validate_h1_contract)


class H1ContractTests(unittest.TestCase):
    def returns(self):
        r = pd.DataFrame({'stock_id': ['000001'] * 14,
                          'month': pd.period_range('2000-01', periods=14, freq='M').astype(str),
                          'return': [.01] * 14})
        r.attrs['units'] = {'return': 'decimal'}
        return r

    def keys(self, month='2000-12'):
        return pd.DataFrame({'stock_id': ['000001'], 'base_month': [month]})

    def components(self):
        f = pd.DataFrame({'stock_id': ['000001', '000002', '000003', '000004'],
                          'base_month': ['2000-12'] * 4, 'eligible': [True, True, True, True],
                          'size': [1., 2., 5., 100.], 'bm': [4., 2., 1., 200.],
                          'mom': [.1, .2, .5, np.nan], 'mom_valid_months': [11, 11, 11, 10]})
        for c in ('size', 'bm', 'mom'):
            f[f'{c}_available'] = f[c].notna()
            f[f'{c}_exclusion_reason'] = np.where(f[c].notna(), '', 'missing')
        return f

    def test_momentum_product_excludes_base_and_future(self):
        r = self.returns()
        r.loc[r.month.ge('2000-12'), 'return'] = 100.
        result = calculate_momentum_11m(r, self.keys()).iloc[0]
        self.assertAlmostEqual(result.mom, 1.01**11-1)
        self.assertEqual(result.mom_valid_months, 11)

    def test_missing_calendar_month_not_compressed(self):
        r = self.returns().loc[lambda x: x.month.ne('2000-06')]
        result = calculate_momentum_11m(r, self.keys('2001-01')).iloc[0]
        self.assertFalse(result.mom_available)
        self.assertEqual(result.mom_valid_months, 10)
        self.assertTrue(pd.isna(result.mom))

    def test_zero_and_minus_one_are_real_returns(self):
        for value, expected in [(0., 0.), (-1., -1.)]:
            r = self.returns(); r['return'] = value
            result = calculate_momentum_11m(r, self.keys()).iloc[0]
            self.assertTrue(result.mom_available)
            self.assertEqual(result.mom, expected)

    def test_invalid_returns_quarantined_not_replaced(self):
        for value in [-1.1, np.inf]:
            r = self.returns(); r.loc[0, 'return'] = value
            result = calculate_momentum_11m(r, self.keys()).iloc[0]
            self.assertFalse(result.mom_available)
            self.assertEqual(result.mom_exclusion_reason, 'invalid_return_in_window')
            self.assertTrue(pd.isna(result.mom))

    def test_first_target_and_early_window(self):
        r = self.returns(); r.loc[0, 'return'] = np.nan
        result = calculate_momentum_11m(r, self.keys()).iloc[0]
        self.assertFalse(result.mom_available)
        early = calculate_momentum_11m(r, self.keys('2000-05')).iloc[0]
        self.assertEqual(early.mom_exclusion_reason, 'window_before_2000')
        self.assertEqual(early.mom_valid_months, 3)

    def test_duplicate_units_and_test_period_rejected(self):
        r = self.returns()
        with self.assertRaises(ValueError):
            calculate_momentum_11m(pd.concat([r, r.iloc[:1]]), self.keys())
        r.attrs.clear()
        with self.assertRaisesRegex(ValueError, 'unit'):
            calculate_momentum_11m(r, self.keys())
        with self.assertRaisesRegex(ValueError, 'development_period'):
            calculate_momentum_11m(self.returns(), self.keys('2022-01'))

    def size_inputs(self):
        p = pd.DataFrame({'stock_id': ['000001']*4, 'month': ['2000-01','2000-02','2000-03','2000-04'],
                          'eligible': [True]*4, 'market_cap': [10., 0., 20., np.nan]})
        a = pd.DataFrame({'stock_code': ['000001']*4, 'base_month': p.month,
                          'equity': [20., 10., -10., 20.], 'usable': [True, True, False, True],
                          'reason': ['', '', 'nonpositive_equity', ''],
                          'available_month': ['1999-12']*4, 'expiry_date': ['2001-06-30']*4})
        p.attrs['units']={'market_cap':'KRW'}; a.attrs['units']={'equity':'KRW'}
        return p,a

    def test_size_bm_positivity_usable_and_units(self):
        p,a=self.size_inputs(); f=calculate_size_bm(p,a)
        self.assertAlmostEqual(f.loc[0,'size'], np.log(10))
        self.assertEqual(f.loc[0,'bm'], 2.)
        self.assertTrue(f.loc[1:,'bm'].isna().all())
        a.loc[0,'usable']=False
        self.assertFalse(calculate_size_bm(p,a).loc[0,'bm_available'])
        a.attrs.clear()
        with self.assertRaisesRegex(ValueError,'unit'): calculate_size_bm(p,a)

    def test_future_or_expired_asof_rejected(self):
        for column,value in [('available_month','2000-02'),('expiry_date','1999-12-31')]:
            p,a=self.size_inputs(); a.loc[0,column]=value
            with self.assertRaisesRegex(ValueError,'asof_future_or_expired'): calculate_size_bm(p,a)

    def test_common_set_ddof_sign_and_divisor(self):
        f=self.components(); result=calculate_h1_score(f)
        v=f.loc[:2,['size','bm','mom']].to_numpy()
        z=(v-v.mean(axis=0))/v.std(axis=0,ddof=1)
        np.testing.assert_allclose(result.loc[:2,['z_size','z_bm','z_mom']],z)
        np.testing.assert_allclose(result.loc[:2,'h1'],(-z[:,0]+z[:,1]+z[:,2])/3)
        self.assertTrue(result.standardization_n.eq(3).all())
        self.assertTrue(pd.isna(result.loc[3,'h1']))

    def test_y_independence_and_ineligible_outlier(self):
        f=self.components(); original=calculate_h1_score(f)
        f['y']=[np.nan, 100., -100., 1.]
        pd.testing.assert_frame_equal(original,calculate_h1_score(f))
        f.loc[3,'eligible']=False
        pd.testing.assert_frame_equal(original.loc[:2],calculate_h1_score(f).loc[:2])

    def test_two_stocks_valid_one_stock_and_constant_invalid(self):
        f=self.components().iloc[:2].copy()
        self.assertTrue(calculate_h1_score(f).h1_available.all())
        self.assertFalse(calculate_h1_score(f.iloc[:1]).h1_available.any())
        f['bm']=1.
        out=calculate_h1_score(f)
        self.assertFalse(out.h1_available.any())
        self.assertTrue(out.h1_exclusion_reason.eq('zero_or_nonfinite_sd').all())
        with self.assertRaises(ValueError): cross_sectional_zscore(f,columns=('size','bm','mom'),ddof=0)

    def test_unknown_eligibility_preserved(self):
        f=self.components(); f['eligible']=pd.Series([True,True,pd.NA,False],dtype='boolean')
        out=calculate_h1_score(f)
        self.assertTrue(pd.isna(out.loc[2,'eligible']))
        self.assertEqual(out.loc[2,'h1_exclusion_reason'],'eligibility_unknown')
        self.assertTrue(out.standardization_n.eq(2).all())

    def test_validator_detects_modified_score(self):
        f=self.components(); out=calculate_h1_score(f)
        a=pd.DataFrame({'stock_code':f.stock_id,'base_month':f.base_month, 'equity':f.bm*np.exp(f['size']),
                        'usable':[True]*4,'available_month':['2000-01']*4,'expiry_date':['2001-12-31']*4})
        validate_h1_contract(out,f,a)
        out.loc[0,'h1'] += .01
        with self.assertRaises(AssertionError): validate_h1_contract(out,f,a)


class H1AnalysisTests(unittest.TestCase):
    def components(self):
        f=pd.concat([pd.DataFrame({'stock_id':[f'{i:06d}' for i in range(1,11)],
            'base_month':[m]*10,'eligible':[True]*9+[False], 'size':np.arange(1,11,dtype=float),
            'bm':np.arange(1,11,dtype=float)**2,'mom':[.1,.3,-.2,.4,.7,1.,.5,.8,100.,10000.],
            'mom_valid_months':[11]*10}) for m in ('2020-12','2021-01','2021-02')],ignore_index=True)
        for c in ('size','bm','mom'):
            f[c+'_available']=True;f[c+'_exclusion_reason']=''
        return f

    def test_winsor_boundaries_common_only_and_no_future(self):
        from src.features.h1 import winsorize_1_99
        f=self.components();original=f.copy(deep=True);out=winsorize_1_99(f)
        v=f.iloc[:9].mom
        self.assertAlmostEqual(out.loc[8,'mom'],v.quantile(.99))
        self.assertEqual(out.loc[9,'mom'],10000.)
        f.loc[f.base_month.eq('2021-02'),'mom']=1e8
        pd.testing.assert_frame_equal(out.iloc[:10],winsorize_1_99(f).iloc[:10])
        pd.testing.assert_frame_equal(original,self.components())

    def test_analysis_counts_contribution_delta_and_year(self):
        from src.evaluation.h1_analysis import analyze
        f=self.components();h=calculate_h1_score(f)
        panel=pd.DataFrame({'stock_id':f.stock_id,'month':(pd.PeriodIndex(f.base_month,freq='M')+1).astype(str),
                            'return':np.tile(np.arange(10,dtype=float)/100,3)})
        panel.loc[0,'return']=np.nan;panel.attrs['units']={'return':'decimal'}
        tables,checks=analyze(f,h,panel,representative_year=2021)
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(tables['monthly_ic']),24)
        self.assertEqual(len(tables['representative_year']),3)
        compare=tables['winsor_compare']
        np.testing.assert_allclose(compare.delta_ic,compare.ic_winsor-compare.ic_equal)
        before=tables['feature_distribution'].copy()
        panel['return']=panel['return'].iloc[::-1].to_numpy()
        other,_=analyze(f,h,panel,representative_year=2021)
        pd.testing.assert_frame_equal(before,other['feature_distribution'])
        pd.testing.assert_frame_equal(tables['feature_correlation'],other['feature_correlation'])
        pd.testing.assert_frame_equal(tables['contribution'],other['contribution'])


class PCAVarianceTests(unittest.TestCase):
    def scores(self):
        rng=np.random.default_rng(2021);frames=[]
        for month,n in [('2020-12',8),('2021-01',7),('2021-02',6)]:
            x=rng.normal(size=(n,3));x=(x-x.mean(axis=0))/x.std(axis=0,ddof=1)
            frames.append(pd.DataFrame({'stock_id':[f'{i:06d}' for i in range(1,n+1)],
                'base_month':[month]*n,'eligible':[True]*n,'z_size':x[:,0],'z_bm':x[:,1],'z_mom':x[:,2]}))
        return pd.concat(frames,ignore_index=True)

    def test_monthly_and_expanding_match_full_svd(self):
        from src.features.h1 import pca_a,pca_b
        from sklearn.decomposition import PCA
        f=self.scores()
        for fn in (pca_a,pca_b):
            out=fn(f)
            for month in f.base_month.unique():
                part=f[f.base_month.eq(month) if fn==pca_a else f.base_month.le(month)]
                x=part[['z_size','z_bm','z_mom']].to_numpy(copy=True);x[:,0]*=-1
                ref=PCA(n_components=3,svd_solver='full').fit(x)
                got=out[out.base_month.eq(month)]
                np.testing.assert_allclose(got.explained_variance_ratio,ref.explained_variance_ratio_,atol=1e-12)
                np.testing.assert_allclose(got.eigenvalue,ref.explained_variance_,atol=1e-12)
                self.assertTrue(got.fit_n.eq(len(part)).all())
                self.assertTrue(got.fit_last_base_month.eq(month).all())
                self.assertAlmostEqual(got.explained_variance_ratio.sum(),1.)

    def test_expanding_has_no_future_rows(self):
        from src.features.h1 import pca_b
        f=self.scores();past=pca_b(f[f.base_month.le('2021-01')])
        f.loc[f.base_month.eq('2021-02'),['z_size','z_bm','z_mom']]=1e6
        out=pca_b(f)
        pd.testing.assert_frame_equal(past,out[out.base_month.le('2021-01')].reset_index(drop=True),check_exact=True)

    def test_first_unavailable_month_preserved_and_no_zero_fill(self):
        from src.features.h1 import pca_a,pca_b
        f=self.scores();f.loc[f.base_month.eq('2020-12'),['z_size','z_bm','z_mom']]=np.nan
        for fn in (pca_a,pca_b):
            out=fn(f);first=out[out.base_month.eq('2020-12')]
            self.assertEqual(len(first),3);self.assertTrue(first.explained_variance_ratio.isna().all())
            self.assertTrue(first.exclusion_reason.eq('too_few_fit_rows').all())
            self.assertTrue(first.fit_n.eq(0).all())
        later=pca_b(f);self.assertTrue(later[later.base_month.eq('2021-01')].fit_first_base_month.eq('2021-01').all())

    def test_two_rows_rank_one_and_constant_reason(self):
        from src.features.h1 import pca_a
        f=self.scores().iloc[:2].copy()
        f[['z_size','z_bm','z_mom']]=np.array([[-1.,-1.,-1.],[1.,1.,1.]])
        out=pca_a(f);np.testing.assert_allclose(out.explained_variance_ratio,[1.,0.,0.],atol=1e-12)
        f[['z_size','z_bm','z_mom']]=0.
        out=pca_a(f);self.assertTrue(out.explained_variance_ratio.isna().all())
        self.assertTrue(out.exclusion_reason.eq('zero_or_nonfinite_total_variance').all())

    def test_pca_rejects_future_targets_partial_and_ineligible(self):
        from src.features.h1 import pca_a
        f=self.scores()
        for kind in ('future','target','partial','ineligible'):
            t=f.copy()
            if kind=='future':t['base_month']='2022-01'
            elif kind=='target':t['y']=0.
            elif kind=='partial':t.loc[0,'z_bm']=np.nan
            else:t.loc[0,'eligible']=False
            with self.assertRaises((ValueError,PermissionError)):pca_a(t)

    def test_four_variants_keep_same_sample_and_no_performance_outputs(self):
        from src.evaluation.h1_analysis import pca_variance_analysis
        f=H1AnalysisTests().components();h=calculate_h1_score(f)
        tables,checks=pca_variance_analysis(f,h)
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(tables['pca_variance_monthly']),3*4*3)
        self.assertEqual(set(tables),{'pca_variance_monthly','pca_variance_summary','pca_variance_by_target_year'})
        self.assertNotIn('ic',tables['pca_variance_monthly'])
        self.assertNotIn('selected_component',tables['pca_variance_monthly'])


class ConstrainedPCATests(unittest.TestCase):
    def test_boundary_optimum_is_not_clipped_ordinary_pc1(self):
        from src.features.h1 import nonnegative_variance_weights
        cov=np.array([[1.,.6,-.2],[.6,1.,-.3],[-.2,-.3,1.]])
        w,value,ties=nonnegative_variance_weights(cov)
        np.testing.assert_allclose(w,[2**-.5,2**-.5,0.],atol=1e-12)
        self.assertEqual(w[2],0.);self.assertAlmostEqual(value,1.6);self.assertEqual(ties,1)
        _,vectors=np.linalg.eigh(cov);ordinary=vectors[:,-1]
        if ordinary[0]<0:ordinary=-ordinary
        clipped=np.maximum(ordinary,0);clipped/=np.linalg.norm(clipped)
        self.assertGreater(value,float(clipped@cov@clipped)+1e-6)

    def test_interior_solution_and_multiple_optima_are_reported(self):
        from src.features.h1 import nonnegative_variance_weights
        cov=np.full((3,3),.4);np.fill_diagonal(cov,1.)
        w,value,ties=nonnegative_variance_weights(cov)
        np.testing.assert_allclose(w,np.ones(3)/np.sqrt(3),atol=1e-12)
        self.assertAlmostEqual(value,1.8);self.assertEqual(ties,1)
        _,value,ties=nonnegative_variance_weights(np.eye(3))
        self.assertEqual(value,1.);self.assertGreater(ties,1)

    def test_expanding_uses_exact_covariance_and_no_future_rows(self):
        from src.features.h1 import constrained_pca
        f=PCAVarianceTests().scores()
        out=constrained_pca(f,method='pca_b')
        for _,row in out.iterrows():
            part=f[f.base_month.le(row.base_month)]
            x=part[['z_size','z_bm','z_mom']].to_numpy(copy=True);x[:,0]*=-1
            cov=np.cov(x,rowvar=False,ddof=1)
            w=row[['weight_neg_size','weight_bm','weight_mom']].to_numpy(float)
            self.assertTrue((w>=0).all());self.assertAlmostEqual(np.linalg.norm(w),1.)
            self.assertAlmostEqual(row.projected_variance,float(w@cov@w))
            self.assertAlmostEqual(row.explained_variance_ratio,float(w@cov@w/np.trace(cov)))
            self.assertEqual(row.fit_n,len(part))
        past=out[out.base_month.le('2021-01')].reset_index(drop=True)
        f.loc[f.base_month.eq('2021-02'),['z_size','z_bm','z_mom']]=1e6
        changed=constrained_pca(f,method='pca_b')
        pd.testing.assert_frame_equal(past,changed[changed.base_month.le('2021-01')].reset_index(drop=True),check_exact=True)


if __name__ == '__main__':
    unittest.main()
