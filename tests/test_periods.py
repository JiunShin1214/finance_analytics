"""src/periods.py와 교수님 CSV 기간 잠금(7-4 ③④): 개발은 테스트 묶음을 열지 않고, 최종(final)만 연다."""
import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from src import periods
from src.data_io import _approved_sources

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config/experiment.yaml').read_text(encoding='utf-8'))


def fake_root(directory: str, last_test_month: str = '2025-12') -> Path:
    """month 열만 있는 가짜 수익률 CSV 3묶음."""
    root = Path(directory)
    (root / 'data/raw').mkdir(parents=True)
    ends = {'warmup_2000': '2000-12', 'train_2001_2021': '2021-12', 'test_2022_2025': last_test_month}
    for group, end in ends.items():
        path = root / CONFIG['PATHS']['wide_csv'][group]['adjusted_monthly_return']
        path.write_text('month,000010\n' + f'{end},0.1\n', encoding='utf-8')
    return root


class PeriodTests(unittest.TestCase):
    def test_development_ignores_test_group(self):
        p = periods.resolve(CONFIG)
        self.assertEqual((p.final, p.end, p.last_base_month), (False, '2021-12', '2021-11'))
        self.assertEqual(p.groups, ('warmup_2000', 'train_2001_2021'))

    def test_final_end_comes_from_data(self):
        with tempfile.TemporaryDirectory() as d:
            p = periods.resolve(CONFIG, fake_root(d, '2026-06'), final=True)
        self.assertEqual((p.end, p.last_base_month, p.end_year), ('2026-06', '2026-05', 2026))
        self.assertEqual(p.groups[-1], 'test_2022_2025')

    def test_table_path(self):
        self.assertEqual(periods.table_path(CONFIG, 'panel').as_posix(), 'data/interim/panel.parquet')
        self.assertEqual(periods.table_path(CONFIG, 'panel', final=True).as_posix(), 'data/final/panel.parquet')

    def test_development_end_must_match_config_groups(self):
        bad = copy.deepcopy(CONFIG)
        bad['DATA_PERIOD']['development_end'] = '2025-12'
        with self.assertRaises(ValueError):
            periods.resolve(bad)

    def test_marcap_missing_year_stops(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, 'marcap_missing'):
                periods.marcap_paths(Path(d), periods.resolve(CONFIG))


class CsvLockTests(unittest.TestCase):
    def test_test_group_blocked_without_final(self):
        with self.assertRaises(PermissionError):
            _approved_sources(ROOT, CONFIG, ('warmup_2000', 'train_2001_2021', 'test_2022_2025'))

    def test_test_group_allowed_with_final(self):
        sources = _approved_sources(ROOT, CONFIG, ('warmup_2000', 'train_2001_2021', 'test_2022_2025'), final=True)
        self.assertEqual({s[0] for s in sources}, {'warmup_2000', 'train_2001_2021', 'test_2022_2025'})

    def test_development_default_reads_only_development_groups(self):
        sources = _approved_sources(ROOT, CONFIG, ('warmup_2000', 'train_2001_2021'))
        self.assertFalse(any('test_2022' in str(s[2]) for s in sources))


if __name__ == '__main__':
    unittest.main()
