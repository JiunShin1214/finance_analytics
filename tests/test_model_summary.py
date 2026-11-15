"""model_summary: 실행 폴더 표만 읽어 합치는 규칙(가짜 폴더로 확인)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import model_summary as ms  # noqa: E402


def make_run(root: Path, name: str, *, family='boosting', model='xgb_reg', hparams='d1_lr0.01', input_set='ind:sigma',
             mean_ic=0.08, status='PASSED', sha='a', months=192):
    d = root / name
    d.mkdir()
    grid = {k: 1 for k in ms.CONTRACT_KEYS}
    record = {'status': status, 'family': family, 'kind': model, 'phase': 'individual', 'device': {'boosting': 'cpu', 'gpu': None},
              'configs': 1, 'commit': 'abcdef0123', 'followup': hparams.startswith('d4'), 'started': '2026-11-14T10:00:00', 'finished': '2026-11-14T11:00:00',
              'input_sha256': {'h3': sha}, 'folds_plan': [1], 'inner_scope': 'common', 'model_grid': grid,
              'input_checks': {'validation_prediction_rows': 133876, 'training_rows_2003_2021': 151761,
                               'training_rows_2001_2021': 165331, 'sigma_decay097_equals_h3_parquet': True}}
    (d / 'run.json').write_text(json.dumps(record), encoding='utf-8')
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
            runs = [make_run(root, 'lin', family='linear', model='ridge', mean_ic=0.087),
                    make_run(root, 'a', mean_ic=0.080), make_run(root, 'b', hparams='d2_lr0.01', mean_ic=0.082),
                    make_run(root, 'c', hparams='d4_lr0.01', model='xgb_rankreg', mean_ic=0.085)]
            loaded = ms.load_runs(runs)
            best = ms.best_per_model(loaded)
            self.assertEqual(dict(zip(best.model, best.mean_ic)), {'ridge': 0.087, 'xgb_reg': 0.082, 'xgb_rankreg': 0.085})
            self.assertEqual(best.loc[best.model.eq('xgb_rankreg'), 'followup'].item(), True)
            self.assertEqual(list(ms.best_per_model(loaded, include_followup=False).model), ['ridge', 'xgb_reg'])
            self.assertEqual(loaded['runs'].run.tolist(), ['lin', 'a', 'b', 'c'])

    def test_rejects_failed_mismatch_duplicate_and_short(self):
        cases = [dict(status='FAILED_END_CHECKS'), dict(sha='other'), dict(), dict(hparams='d2_lr0.01', months=185)]
        for extra in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runs = [make_run(root, 'a'), make_run(root, 'b', **extra)]     # 빈 extra = 같은 설정 ID가 두 폴더에
                with self.assertRaises(ValueError):
                    ms.load_runs(runs)


if __name__ == '__main__':
    unittest.main()
