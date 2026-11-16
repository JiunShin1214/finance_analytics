"""model_summary: 결과 폴더 표만 읽어 합치는 규칙(가짜 폴더로 확인). 검사 없음 — 중복·짧은 기간도 그대로 보여 준다."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import model_summary as ms  # noqa: E402


def make_run(root: Path, name: str, *, author='jiun', family='boosting', model='xgb_reg', hparams='d1_lr0.01',
             input_set='ind:sigma', mean_ic=0.08, months=192):
    """<root>/<작성자>/<name> 결과 폴더(표 3개, run.json 없음)."""
    d = root / author / name
    d.mkdir(parents=True)
    key = {'family': family, 'model': model, 'target': 'monthly_zscore', 'loss': 'mse', 'input_set': input_set}
    if family == 'boosting':
        key['hparams'] = hparams
    perf = pd.DataFrame([{**key, 'scope': s, 'months': months, 'mean_ic': mean_ic, 'ic_sd': 0.1, 'icir': mean_ic / 0.1,
                          'ic_positive_share': 0.7, 'train_ic_fold_mean': 0.1, 'r2_pct': None, 'train_minus_valid': 0.1 - mean_ic,
                          'mean_ic_2006~2010': mean_ic, 'mean_ic_2011~2021': mean_ic, 'mean_ic_2018~2021': mean_ic}
                         for s in ('full', 'common')])
    perf.to_csv(d / 'performance_summary.csv', index=False)
    pd.DataFrame([{**key, 'target_month': '2006-01', 'n': 700, 'ic': mean_ic, 'reason': '', 'scope': 'full'}]).to_csv(
        d / 'monthly_ic.csv', index=False)
    pd.DataFrame([{**key, 'fold': 1, 'valid_year': 2006}]).to_csv(d / 'fold_table.csv', index=False)
    return d


class ModelSummaryTests(unittest.TestCase):
    def test_load_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [make_run(root, '20261113T000000_linear', family='linear', model='ridge', mean_ic=0.087),
                    make_run(root, '20261114T000000_xgb_reg_individual', mean_ic=0.080),
                    make_run(root, '20261114T010000_xgb_reg_individual', hparams='d2_lr0.01', mean_ic=0.082),
                    make_run(root, '20261114T020000_xgb_rankreg_individual_d4', hparams='d4_lr0.01', model='xgb_rankreg',
                             mean_ic=0.085)]
            self.assertEqual(ms.find_run_folders(root), sorted(runs))
            loaded = ms.load_runs(runs)
            best = ms.best_per_model(loaded)
            self.assertEqual(dict(zip(best.model, best.mean_ic)), {'ridge': 0.087, 'xgb_reg': 0.082, 'xgb_rankreg': 0.085})
            self.assertEqual(best.loc[best.model.eq('xgb_rankreg'), 'followup'].item(), True)
            self.assertEqual(list(ms.best_per_model(loaded, include_followup=False).model), ['ridge', 'xgb_reg'])
            self.assertEqual(set(loaded['runs'].author), {'jiun'})
            self.assertEqual(set(loaded['runs'].experiment), {'base'})
            self.assertEqual(set(loaded['performance'].input_transform), {'미기록'})   # 칸이 없던 예전 결과

    def test_duplicates_and_short_runs_shown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [make_run(root, '20261114T000000_xgb_reg_individual', mean_ic=0.08),
                    make_run(root, '20261115T000000_xgb_reg_individual', mean_ic=0.07, months=185)]
            table = ms.all_settings(ms.load_runs(runs))
            self.assertEqual(sorted(table.run), sorted(r.name for r in runs))
            self.assertEqual(sorted(table.months), [185, 192])


if __name__ == '__main__':
    unittest.main()
