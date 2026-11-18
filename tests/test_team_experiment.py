"""붙인 특징 파일, 자유 부스팅 설정·라이브러리 덮어쓰기, 저장 폴더 이름, result.zip — 합성 자료만."""
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
            test = pd.DataFrame({'stock_id': ['000010'], 'base_month': ['2021-12'], 'h2': [1.0]})
            rel = feature_file(root, data, extra_rows=test)
            with self.assertRaises(PermissionError):                        # 테스트 기간 값
                mt.load_extra_feature(root, 'h2', rel)
            with self.assertRaises(PermissionError):                        # data/ 밖 파일
                mt.load_extra_feature(root, 'h2', '../h2.parquet')


class BoostingOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prep = prepared_from(*synthetic())
        cand = prep['lambda_candidates']
        cls.prep = {**prep, 'sigma_labels': ['span12', 'span36', 'span60'],
                    'lambda_candidates': cand[cand.label.isin(['span12', 'span36', 'span60'])].reset_index(drop=True)}
        cls.folds = mt.plan_folds(CONFIG['MODEL_GRID'])

    def test_free_settings_overrides_feature_and_save(self):
        setting = ('lgb_rankreg', 'percentile', 'mse', 5, 0.05)            # 기본 목록 밖 깊이·학습률도 그대로 받는다
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel = feature_file(root, self.prep['data'])
            prep = with_feature(self.prep, root, rel)
            name, cols = mt.make_input_set(['size', 'bm', 'mom', 'sigma', 'h2'])
            r = mt.run_boosting(prep, CONFIG, 'lgb_rankreg', phase='individual', settings=[setting], sets={name: cols},
                                folds=[self.folds[0]], overrides={'lambda_l2': 5.0}, n_threads=1, log=QUIET)
            self.assertEqual(set(r['selections'].hparams), {'d5_lr0.05_lambda_l25'})
            self.assertIn('gain_h2', r['selections'])
            p = next(iter(r['params'].values()))
            self.assertEqual((p['lambda_l2'], p['max_depth'], p['learning_rate']), (5.0, 5, 0.05))
            out = mt.finalize_boosting(ROOT, CONFIG, prep, r, started=datetime(2026, 1, 1), out_root=root / 'runs',
                                       log=QUIET, author='hong', experiment='h2_d5')
            folder = Path(out['run_dir'])
            self.assertEqual((folder.parent.name, folder.name), ('hong', '20260101T000000_h2_d5_lgb_rankreg_individual'))
            files = {f.name for f in folder.iterdir()}
            self.assertTrue({'performance_summary.csv', 'curves.parquet', 'feature_coverage.csv', 'result.zip'} <= files)
            self.assertEqual(ms.parse_folder(folder), {'author': 'hong', 'experiment': 'h2_d5', 'followup': False})

    def test_folder_name_rules(self):
        prep = prepared_from(*synthetic())
        with tempfile.TemporaryDirectory() as tmp:
            for author, experiment in (('a/b', 'x'), ('', 'x'), ('홍길동', 'x'), ('kim', ''), ('kim', '../x')):
                with self.subTest(author=author, experiment=experiment), self.assertRaises(ValueError):
                    mt.result_folder(ROOT, CONFIG, author=author, experiment=experiment, started=datetime(2026, 1, 2),
                                     suffix='linear', out_root=Path(tmp))
            folder = mt.result_folder(ROOT, CONFIG, author='kim', experiment='부채 실험', started=datetime(2026, 1, 2),
                                      suffix='linear', out_root=Path(tmp))           # 실험 이름은 자유
            self.assertTrue(folder.is_dir())
        self.assertEqual(len(mt.validation_rows(prep, CONFIG)), int((prep['data'].predict & prep['data'].year.between(2006, 2021)).sum()))


class ResultZipTests(unittest.TestCase):
    def test_zip_only_folder_reads_like_full_folder(self):
        import zipfile
        from test_model_summary import make_run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = make_run(root, '20261114T000000_xgb_reg_individual', mean_ic=0.08)
            (a / 'validation_predictions.parquet').write_bytes(b'big')           # 예측값은 zip에 넣지 않는다
            z = mt.pack_result(a)
            names = zipfile.ZipFile(z).namelist()
            self.assertIn('performance_summary.csv', names)
            self.assertNotIn('validation_predictions.parquet', names)
            first = z.read_bytes()
            self.assertEqual(mt.pack_result(a).read_bytes(), first)               # 같은 내용이면 같은 파일
            full = ms.load_runs([a])
            only = root / 'only' / 'jiun' / a.name                                # Git으로 받은 상태: zip만
            only.mkdir(parents=True)
            (only / 'result.zip').write_bytes(first)
            self.assertEqual(ms.find_run_folders(root / 'only'), [only])
            got = ms.load_runs([only])
            pd.testing.assert_frame_equal(full['performance'], got['performance'])


class SummaryExperimentTests(unittest.TestCase):
    def test_same_config_in_two_experiments_allowed(self):
        from test_model_summary import make_run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = make_run(root, '20261114T000000_xgb_reg_individual', mean_ic=0.08)
            b = make_run(root, '20261115T000000_h2_xgb_reg_individual', author='hong', mean_ic=0.09)
            loaded = ms.load_runs([a, b])
            self.assertEqual(sorted(loaded['runs'].experiment), ['base', 'h2'])
            self.assertEqual(ms.best_per_model(loaded).mean_ic.tolist(), [0.08])   # 기본 표는 base 실험만
            self.assertEqual(len(ms.all_settings(loaded)), 2)


if __name__ == '__main__':
    unittest.main()
