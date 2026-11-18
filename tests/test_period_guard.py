# 목표월 기준 기간 구분과 연별 확장창의 시간순서를 검사한다.
# 파일 읽기 전에 Test·범위 미확인 자료를 차단하는지 확인한다.

import unittest
from unittest.mock import patch
from pathlib import Path
import copy
import tempfile

import yaml

from src.data_io import load_panel_contract

ROOT = Path(__file__).resolve().parents[1]


class LoaderAccessTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))

    def assert_blocked_before_open(self, config, periods, root=ROOT):
        # 실제 Test 파일을 만들거나 읽지 않는다. 어느 파일도 open되면 테스트 실패.
        with patch.object(Path, "open", side_effect=AssertionError("content open forbidden")) as opened:
            with self.assertRaises(PermissionError):
                load_panel_contract(root, column_mapping=config["COLUMN_MAPPING"]["panel"],
                                    config=config, periods=periods)
            opened.assert_not_called()

    def test_test_unknown_mixed_and_empty_requests_are_blocked(self):
        for periods in [("test_2022_2025",), ("unknown",), (),
                        ("warmup_2000", "test_2022_2025"),
                        ("warmup_2000", "warmup_2000")]:
            with self.subTest(periods=periods):
                self.assert_blocked_before_open(self.config, periods)

    def test_manifest_injection_is_blocked_before_any_development_read(self):
        for path in ["data/raw/test_2022_2025__isin.csv", "data/raw/unknown.csv",
                     "data/raw/../test_2022_2025__isin.csv", str(ROOT / "data/raw/unknown.csv")]:
            config = copy.deepcopy(self.config)
            # 마지막 변수의 경로 오류도 첫 허용 파일을 읽기 전에 차단해야 한다.
            config["PATHS"]["wide_csv"]["train_2001_2021"]["isin"] = path
            with self.subTest(path=path):
                self.assert_blocked_before_open(config, ("warmup_2000", "train_2001_2021"))

    def test_extra_test_manifest_and_missing_scope_are_blocked(self):
        config = copy.deepcopy(self.config)
        config["PATHS"]["wide_csv"]["test_2022_2025"] = {}
        self.assert_blocked_before_open(config, ("warmup_2000",))
        config = copy.deepcopy(self.config)
        config["PATHS"].pop("wide_csv")
        self.assert_blocked_before_open(config, ("warmup_2000",))

    def test_final_test_flag_cannot_bypass_development_loader(self):
        config = copy.deepcopy(self.config)
        config["FINAL_TEST_ENABLED"] = True
        config["TEST_ACCESS_REQUIRES_FROZEN_CONFIG"] = False
        self.assert_blocked_before_open(config, ("test_2022_2025",))

    def test_arbitrary_csv_path_is_not_a_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "test_2022_2025__isin.csv"
            self.assert_blocked_before_open(self.config, ("warmup_2000",), root=fake)


@unittest.skip("목표월 분할·16 Fold·학습기간 규칙은 아직 미구현")
def test_period_guard() -> None:
    """후속 목표월 분할·학습/평가 시간 분리의 검증 골격이다.

    TODO: 2000 준비기간 fit/IC 제외, 2000-12 기준의 2001-01 목표 Train 구분.
    2001~2005 -> 2006부터 2001~2020 -> 2021까지 16개 Fold를 검사한다.
    외부자료 최소 5년 학습, 부족 시 보류를 검사한다.
    wide 로더의 읽기 전 차단은 위 LoaderAccessTests에서 실제 검증한다.
    """
    raise NotImplementedError
