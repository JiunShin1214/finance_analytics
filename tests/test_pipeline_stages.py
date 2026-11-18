"""실행 기록 없이 현재 입력 계약을 검사하는 합성 자료 테스트."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.pipeline import STAGES,check_ingest_current,check_observe_current,check_features_current,plan_h1_test,plan_h1_pca_variance,plan_h1_constrained_pca,plan_h1_hypothesis
from src.contracts import WIDE_PANEL_COLUMNS,ELIGIBILITY_COLUMNS,EQUITY_ASOF_COLUMNS,MOM_COLUMNS
from src.features.h1 import calculate_h1_score

class PipelineStageTests(unittest.TestCase):
    def inputs(self,root):
        names=('panel','eligibility','asof','h1_components','mom','h1')
        config={'PATHS':{n:'data/'+n+'.parquet' for n in names},'H1':{},'UNITS':{},'H1_TEST':{'approved_stage':2,'representative_year':2021,'notion_tables_path':'outputs/report.md'}}
        (root/'data').mkdir()
        ids=['000001','000002','000003'];month=['2001-01']*3
        panel=pd.DataFrame({c:[None]*3 for c in WIDE_PANEL_COLUMNS})
        panel['stock_id']=ids;panel['month']=month;panel['eligible']=True
        panel['market_cap']=[1.,2.,4.];panel['return']=[.1,.2,.3]
        eligibility=pd.DataFrame({c:[None]*3 for c in ELIGIBILITY_COLUMNS})
        eligibility['stock_id']=ids;eligibility['month']=month;eligibility['eligible']=True
        asof=pd.DataFrame({c:[None]*3 for c in EQUITY_ASOF_COLUMNS})
        asof['stock_code']=ids;asof['base_month']=month;asof['equity']=[1.,4.,12.]
        asof['usable']=True;asof['available_month']='2000-12';asof['expiry_date']='2002-06-30'
        components=pd.DataFrame({'stock_id':ids,'base_month':month,'eligible':[True]*3,'size':np.log([1.,2.,4.]),'bm':[1.,2.,3.],'mom':[.1,.4,.2],'mom_valid_months':[11]*3})
        for col in ('size','bm','mom'):
            components[col+'_available']=True;components[col+'_exclusion_reason']=''
        h=calculate_h1_score(components)
        for name,table in [('panel',panel),('eligibility',eligibility),('asof',asof),('h1_components',components),('mom',components[list(MOM_COLUMNS)]),('h1',h)]:
            table.to_parquet(root/config['PATHS'][name],index=False)
        return config

    def test_steps_and_final_test_guard(self):
        from src.pipeline import final_test
        self.assertLess(STAGES.index('ingest'),STAGES.index('features'))
        self.assertLess(STAGES.index('features'),STAGES.index('h1-test'))
        with self.assertRaises(PermissionError):final_test({})

    def test_valid_data_without_any_outputs_or_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.inputs(root)
            self.assertTrue(check_ingest_current(config,root=root)['data_inputs_checked'])
            self.assertTrue(check_observe_current(config,root=root)['data_inputs_checked'])
            self.assertTrue(check_features_current(config,root=root)['feature_contract_checked'])
            self.assertFalse((root/'outputs').exists())

    def test_missing_input_names_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.inputs(root);(root/config['PATHS']['asof']).unlink()
            with self.assertRaisesRegex(FileNotFoundError,'asof.parquet'):check_ingest_current(config,root=root)

    def test_eligibility_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.inputs(root);path=root/config['PATHS']['eligibility']
            table=pd.read_parquet(path);table.loc[0,'eligible']=False;table.to_parquet(path,index=False)
            with self.assertRaises(AssertionError):check_ingest_current(config,root=root)

    def test_duplicate_and_test_period_inputs_stop(self):
        for kind in ('duplicate','test'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);config=self.inputs(root);path=root/config['PATHS']['panel'];table=pd.read_parquet(path)
                if kind=='duplicate':table=pd.concat([table,table.iloc[:1]],ignore_index=True)
                else:table.loc[0,'month']='2022-01'
                table.to_parquet(path,index=False)
                with self.assertRaises((ValueError,PermissionError)):check_ingest_current(config,root=root)

    def test_changed_feature_value_stops_without_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.inputs(root);path=root/config['PATHS']['h1'];table=pd.read_parquet(path)
            table.loc[0,'h1']+=1;table.to_parquet(path,index=False)
            with self.assertRaises(AssertionError):check_features_current(config,root=root)

    def test_h1_dry_run_never_writes_or_runs_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.inputs(root)
            with patch('src.evaluation.h1_analysis.run_analysis') as analysis:
                self.assertFalse(plan_h1_test(config,root=root)['execute']);analysis.assert_not_called()
            self.assertFalse((root/'outputs').exists())

    def test_approved_scope_still_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.inputs(root);config['H1_TEST']['representative_year']=2020
            with self.assertRaises(PermissionError):plan_h1_test(config,root=root)

    def test_pca_variance_accepts_current_structure_scope(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.inputs(root)
            config['H1_PCA']={'approved_scope':'structure_diagnostics','variants':['raw','winsor'],'run_category':'h1_pca반영지수'}
            self.assertFalse(plan_h1_pca_variance(config,root=root)['execute'])
            config['H1_PCA']['approved_scope']='scores'
            with self.assertRaises(PermissionError):plan_h1_pca_variance(config,root=root)

    def test_constrained_scope_still_required(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=self.inputs(root)
            config['H1_PCA']={'constrained_exploration_approved':True,'constrained_variants':['raw','winsor'],'run_category':'h1_pca반영지수'}
            self.assertFalse(plan_h1_constrained_pca(config,root=root)['execute'])
            config['H1_PCA']['constrained_exploration_approved']=False
            with self.assertRaises(PermissionError):plan_h1_constrained_pca(config,root=root)

    def test_hypothesis_dry_run_never_executes_analysis(self):
        with patch('src.evaluation.h1_analysis.check_hypothesis_inputs',return_value={'checked':True}),patch('src.evaluation.h1_analysis.run_hypothesis') as analysis:
            self.assertFalse(plan_h1_hypothesis({},execute=False)['execute']);analysis.assert_not_called()

    def test_report_created_without_previous_file_or_notion(self):
        from src.evaluation.h1_analysis import append_report
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'outputs/report.md';append_report(path,'## result','## result\nfirst')
            append_report(path,'## result','## result\nsecond')
            self.assertEqual(path.read_text(encoding='utf-8'),'## result\nsecond')

if __name__=='__main__':unittest.main()
