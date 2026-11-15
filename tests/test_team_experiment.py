"""팀 실험 통로(2026-11-15): 새 특징 파일, 실험용 부스팅 설정, 공통 저장(save_result), 작성자·실험 폴더 — 합성 자료만."""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_model_test import CONFIG, prepared_from, synthetic  # noqa: E402

from src.evaluation import model_summary as ms  # noqa: E402
from src.evaluation import model_test as mt  # noqa: E402
from src.models import boosting  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUIET = lambda *_: None  # noqa: E731


def feature_file(root: Path, data: pd.DataFrame, *, name='h2', seed=0, missing=0.2, extra_rows=None) -> str:
    rng = np.random.default_rng(seed)
    keys = data[['stock_id', 'base_month']].drop_duplicates()
    f = keys.assign(**{name: rng.normal(size=len(keys))})
    f.loc[rng.random(len(f)) < missing, name] = np.nan
    if extra_rows is not None:
        f = pd.concat([f, extra_rows], ignore_index=True)
    rel = f'data/processed/{name}.parquet'
    (root / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    f.to_parquet(root / rel, index=False)
    return rel


def with_feature(prepared: dict, root: Path, rel: str, name='h2') -> dict:
    data, cov = mt.add_extra_features(prepared['data'], {name: mt.load_extra_feature(root, name, rel)})
    return {**prepared, 'data': data, 'extra_features': {name: rel}, 'feature_coverage': cov}


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = prepared_from(*synthetic())

    def test_feature_rows_and_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel = feature_file(root, self.prep['data'])
            p = with_feature(self.prep, root, rel)
        d0, d = self.prep['data'], p['data']
        self.assertTrue(d.predict.equals(d0.predict))                       # 예측·채점 행은 그대로
        self.assertTrue((d.train <= d0.train).all() and d.train.sum() < d0.train.sum())   # 학습은 새 특징까지 관측된 행만
        self.assertTrue(d.loc[d.train, 'x_h2'].notna().all())
        x = d.x_h2.dropna()
        self.assertTrue(x.abs().lt(1).all())
        # 목표값 변환은 새 학습 행 안에서 다시 만든다
        t = d[d.train]
        for _, g in list(t.groupby('target_month'))[:5]:
            r = g.y.rank(method='average')
            np.testing.assert_allclose(g.t_percentile, r / (len(g) + 1))
        self.assertTrue(d.loc[~d.train, 't_percentile'].isna().all())
        self.assertEqual(set(p['feature_coverage'].feature), {'h2'})

    def test_feature_file_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.prep['data']
            dup = data[['stock_id', 'base_month']].head(1).assign(h2=1.0)
            test = pd.DataFrame({'stock_id': ['000010'], 'base_month': ['2021-12'], 'h2': [1.0]})
            for extra, err in ((dup, ValueError), (test, PermissionError)):
                rel = feature_file(root, data, extra_rows=extra)
                with self.subTest(err=err), self.assertRaises(err):
                    mt.load_extra_feature(root, 'h2', rel)
            rel = feature_file(root, data)
            for name, path, err in (('sigma', rel, ValueError), ('h2', '../h2.parquet', PermissionError),
                                    ('x_bad', rel, ValueError)):
                with self.subTest(name=name), self.assertRaises(err):
                    mt.load_extra_feature(root, name, path)


class CustomBoostingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prep = prepared_from(*synthetic())
        cand = prep['lambda_candidates']
        cls.prep = {**prep, 'sigma_labels': ['span12', 'span36', 'span60'],
                    'lambda_candidates': cand[cand.label.isin(['span12', 'span36', 'span60'])].reset_index(drop=True)}
        cls.folds = mt.plan_folds(CONFIG['MODEL_GRID'])

    def test_custom_settings_overrides_and_feature(self):
        setting = ('lgb_rankreg', 'percentile', 'mse', 5, 0.05)            # 승인 밖 깊이·학습률
        with self.assertRaises(ValueError):
            mt.run_boosting(self.prep, CONFIG, 'lgb_rankreg', phase='individual', settings=[setting], folds=[self.folds[0]], log=QUIET)
        with self.assertRaises(ValueError):                                 # 목적함수는 덮어쓸 수 없음
            boosting.lgb_params('lgb_rankreg', 'mse', 3, 0.01, overrides={'objective': 'huber'})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel = feature_file(root, self.prep['data'])
            prep = with_feature(self.prep, root, rel)
            name, cols = mt.make_input_set(['size', 'bm', 'mom', 'sigma', 'h2'])
            r = mt.run_boosting(prep, CONFIG, 'lgb_rankreg', phase='individual', settings=[setting], sets={name: cols},
                                folds=[self.folds[0]], custom=True, overrides={'lambda_l2': 5.0}, n_threads=1, log=QUIET)
            self.assertEqual(set(r['selections'].hparams), {'d5_lr0.05_lambda_l25'})
            self.assertIn('gain_h2', r['selections'])
            p = next(iter(r['params'].values()))
            self.assertEqual((p['lambda_l2'], p['max_depth'], p['learning_rate']), (5.0, 5, 0.05))
            out = mt.finalize_boosting(ROOT, CONFIG, prep, r, started=datetime(2026, 1, 1), out_root=root / 'runs',
                                       require_complete=True, log=QUIET, author='hong', experiment='h2_d5',
                                       description='test', changes={'depths': [5]})
            folder = Path(out['run_dir'])
            self.assertEqual((folder.parent.name, folder.name), ('hong', '20260101T000000_h2_d5_lgb_rankreg_individual'))
            rec = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual((rec['author'], rec['experiment'], rec['custom'], rec['overrides']),
                             ('hong', 'h2_d5', True, {'lambda_l2': 5.0}))
            self.assertEqual(rec['extra_features'], {'h2': rel})
            self.assertEqual(rec['input_sets'], {name: cols})
            # Fold 1개라 행 수·192개월 검사만 실패하고 나머지는 통과해야 한다
            self.assertEqual({k for k, v in out['checks'].items() if v is False}, {'validation_prediction_rows', 'all_192_months_ic'})

    def test_default_run_unchanged_by_new_options(self):
        setting = ('xgb_reg', 'monthly_zscore', 'mse', 1, 0.01)
        r = mt.run_boosting(self.prep, CONFIG, 'xgb_reg', phase='individual', settings=[setting], folds=[self.folds[0]],
                            device='cpu', n_threads=1, log=QUIET)
        self.assertEqual(set(r['selections'].hparams), {'d1_lr0.01'})
        self.assertEqual((r['custom'], r['overrides']), (False, {}))
        with self.assertRaises(ValueError):                                 # 기본 단계에 새 조합은 custom 없이 거부
            mt.run_boosting(self.prep, CONFIG, 'xgb_reg', phase='individual', settings=[setting],
                            sets=dict([mt.make_input_set(['size', 'sigma'])]), folds=[self.folds[0]], device='cpu', log=QUIET)


class SaveResultTests(unittest.TestCase):
    def test_team_model_saved_in_same_format(self):
        prep = prepared_from(*synthetic())
        val = mt.validation_rows(prep, CONFIG)
        data = prep['data']
        score = data.x_bm.to_numpy(float)[val.row.to_numpy()]
        score = np.where(np.isfinite(score), score, 0.0)                    # 빈 특징은 0(결측 규칙)
        folds = mt.plan_folds(CONFIG['MODEL_GRID'])
        sel = pd.DataFrame([{'target': 'raw', 'loss': 'none', 'input_set': 'ind:bm', 'hparams': 'sign', 'fold': f['fold'],
                             'valid_year': f['valid_year'], 'train_ic': 0.0, 'chosen_note': 'no tuning'} for f in folds])
        with tempfile.TemporaryDirectory() as tmp:
            out = mt.save_result(ROOT, CONFIG, prep, model='bm_sign', predictions={('raw', 'none', 'ind:bm', 'sign'): score},
                                 selections=sel, started=datetime(2026, 1, 2), author='kim', experiment='bm_only',
                                 out_root=Path(tmp), log=QUIET)
            folder = Path(out['run_dir'])
            self.assertEqual((folder.parent.name, folder.name), ('kim', '20260102T000000_bm_only_bm_sign'))
            self.assertEqual({k for k, v in out['checks'].items() if v is False}, {'validation_prediction_rows'})   # 합성 자료 행 수
            rec = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual((rec['family'], rec['author'], rec['experiment']), ('custom', 'kim', 'bm_only'))
        with self.assertRaises(ValueError):                                 # 예측 길이가 채점 행과 다르면 거부
            mt.save_result(ROOT, CONFIG, prep, model='bad', predictions={('raw', 'none', 'ind:bm', 'sign'): score[:-1]},
                           selections=sel, started=datetime(2026, 1, 2), author='kim', experiment='x', log=QUIET)
        for author, experiment in (('a/b', 'x'), ('', 'x'), ('kim', None), ('kim', 'a b'), ('kim', '../x'), ('홍길동', 'x')):
            with self.subTest(author=author, experiment=experiment), self.assertRaises(ValueError):   # 폴더 이름 규칙
                mt.save_result(ROOT, CONFIG, prep, model='bm_sign', predictions={('raw', 'none', 'ind:bm', 'sign'): score},
                               selections=sel, started=datetime(2026, 1, 2), author=author, experiment=experiment, log=QUIET)
        with tempfile.TemporaryDirectory() as tmp:                          # 규격 폴더 찾기: <base>/<작성자>/<결과>
            out = mt.save_result(ROOT, CONFIG, prep, model='bm_sign', predictions={('raw', 'none', 'ind:bm', 'sign'): score},
                                 selections=sel, started=datetime(2026, 1, 3), author='kim', experiment='bm_only',
                                 out_root=Path(tmp), log=QUIET)
            self.assertEqual(ms.find_run_folders(Path(tmp), log=QUIET), [])          # 합성 자료라 FAILED → 제외


class ResultZipTests(unittest.TestCase):
    def test_zip_only_folder_reads_like_full_folder(self):
        import zipfile
        from test_model_summary import make_run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = make_run(root, 'a', mean_ic=0.08)
            (a / 'validation_predictions.parquet').write_bytes(b'big')           # 예측값은 zip에 넣지 않는다
            z = mt.pack_result(a)
            names = zipfile.ZipFile(z).namelist()
            self.assertIn('run.json', names)
            self.assertNotIn('validation_predictions.parquet', names)
            first = z.read_bytes()
            self.assertEqual(mt.pack_result(a).read_bytes(), first)               # 같은 내용이면 같은 파일
            full = ms.load_runs([a])
            only = root / 'only' / 'a'                                            # Git으로 받은 상태: zip만
            only.mkdir(parents=True)
            (only / 'result.zip').write_bytes(first)
            self.assertEqual(ms.find_run_folders(root / 'only', log=QUIET), [only])
            got = ms.load_runs([only])
            pd.testing.assert_frame_equal(full['performance'].drop(columns=['run', 'author']), got['performance'].drop(columns=['run', 'author']))   # 작성자는 run.json에 없으면 상위 폴더 이름


class SummaryExperimentTests(unittest.TestCase):
    def test_same_config_in_two_experiments_allowed(self):
        from test_model_summary import make_run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = make_run(root, 'a', mean_ic=0.08)
            b = make_run(root, 'b', mean_ic=0.09)
            rec = json.loads((b / 'run.json').read_text(encoding='utf-8'))
            rec.update(author='hong', experiment='h2')
            (b / 'run.json').write_text(json.dumps(rec), encoding='utf-8')
            loaded = ms.load_runs([a, b])
            self.assertEqual(sorted(loaded['runs'].experiment), ['base', 'h2'])
            self.assertEqual(ms.best_per_model(loaded).mean_ic.tolist(), [0.08])   # 기본 표는 base 실험만
            self.assertEqual(len(ms.all_settings(loaded)), 2)


if __name__ == '__main__':
    unittest.main()
