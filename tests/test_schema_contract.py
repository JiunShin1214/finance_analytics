# 공통·확장 스키마, 유일키 및 공개시점 결합의 무결성을 검사한다.
# H2/H4 상태 구분과 비활성 확장이 기준 표본에 미치는 영향을 확인한다.

import copy
import csv
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from src.contracts import DEVELOPMENT_MONTHS, NUMERIC_PANEL_UNITS, WIDE_VARIABLE_COLUMNS
from src.data_io import load_panel_contract, normalize_columns_contract, validate_input_schema
from src.preprocessing.panel import normalize_month, normalize_stock_id

ROOT = Path(__file__).resolve().parents[1]
REAL_INPUT_SUMMARY = None


def read_config():
    return yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))


@contextmanager
def fixture():
    """개발기간 이름의 임시 CSV만 생성. 실제 원자료에는 쓰지 않는다."""
    config = read_config()
    with tempfile.TemporaryDirectory(prefix="finance-input-") as directory:
        root = Path(directory)
        (root / "data/raw").mkdir(parents=True)
        for period, (start, end) in DEVELOPMENT_MONTHS.items():
            for variable, field in WIDE_VARIABLE_COLUMNS.items():
                path = root / config["PATHS"]["wide_csv"][period][variable]
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["month", "000020", "0120G0", "0126Z0"])
                    for month in pd.period_range(start, end, freq="M").astype(str):
                        values = ["0", "", "0.03"] if field in NUMERIC_PANEL_UNITS else ["NA", "", "NaN"]
                        writer.writerow([month, *values])
        yield root, config


def edit_csv(path, edit):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    edit(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def load_fixture(root, config, periods=("warmup_2000",)):
    return load_panel_contract(root, column_mapping=config["COLUMN_MAPPING"]["panel"],
                               config=config, periods=periods)


class InputSchemaTests(unittest.TestCase):
    def test_real_warmup_train_contract(self):
        global REAL_INPUT_SUMMARY
        config = read_config()
        panel = load_fixture(ROOT, config, tuple(DEVELOPMENT_MONTHS))
        self.assertEqual(panel.shape, (264 * 1086, 9))
        self.assertEqual(panel["stock_id"].nunique(), 1086)
        self.assertEqual(panel["month"].nunique(), 264)
        self.assertEqual((panel["month"].min(), panel["month"].max()), ("2000-01", "2021-12"))
        self.assertTrue({"000020", "0120G0", "0126Z0"}.issubset(panel["stock_id"].unique()))
        self.assertTrue(panel["eligible"].isna().all())
        self.assertFalse(panel.duplicated(["stock_id", "month"]).any())
        self.assertFalse({"y", "target_month", "h1", "h3", "sample_id"} & set(panel.columns))
        audits = panel.attrs["input_audit"]
        self.assertEqual(len(audits), 12)
        for audit in audits:
            field = WIDE_VARIABLE_COLUMNS[audit["variable"]]
            mask = panel["month"].str.startswith("2000-")
            values = panel.loc[mask if audit["period"] == "warmup_2000" else ~mask, field]
            self.assertEqual(int(values.isna().sum()), audit["empty_cells"])
            self.assertEqual(audit["columns"], 1087)
            self.assertEqual(audit["rows"], 12 if audit["period"] == "warmup_2000" else 252)
            if field in NUMERIC_PANEL_UNITS:
                self.assertEqual(int(values.eq(0).sum()), audit["zero_cells"])
        self.assertEqual(panel.attrs["units"], NUMERIC_PANEL_UNITS)
        REAL_INPUT_SUMMARY = {"shape": list(panel.shape), "columns": panel.columns.tolist(),
                              "dtypes": {k: str(v) for k, v in panel.dtypes.items()}, "inputs": audits}

    def test_zero_blank_and_literal_text_are_distinct(self):
        with fixture() as (root, config):
            panel = load_fixture(root, config)
            first = panel.iloc[:3]
            for field in NUMERIC_PANEL_UNITS:
                self.assertEqual(first[field].iloc[0], 0.0)
                self.assertTrue(np.isnan(first[field].iloc[1]))
                self.assertEqual(first[field].iloc[2], 0.03)
            for field in ("company_name", "isin"):
                self.assertEqual(first[field].iloc[0], "NA")
                self.assertTrue(pd.isna(first[field].iloc[1]))
                self.assertEqual(first[field].iloc[2], "NaN")

    def test_identifier_and_month_validation(self):
        identifiers = ["000020", "0120G0", "0126Z0"]
        self.assertEqual(normalize_stock_id(identifiers), identifiers)
        for invalid in [20, "20", "00000020", " 120G0", None]:
            with self.subTest(identifier=invalid), self.assertRaises(ValueError):
                normalize_stock_id([invalid])
        self.assertEqual(normalize_month(["2000-12", "2001-01"]), ["2000-12", "2001-01"])
        for invalid in ["2000-1", "2000-00", "2000-13", "0000-01", "2000-01-01", None]:
            with self.subTest(month=invalid), self.assertRaises(ValueError):
                normalize_month([invalid])

    def test_nonempty_numeric_parse_failures(self):
        for token in ["bad", "NA", "NaN", " "]:
            with self.subTest(token=token), fixture() as (root, config):
                path = root / config["PATHS"]["wide_csv"]["warmup_2000"]["adjusted_monthly_return"]
                edit_csv(path, lambda rows: rows[1].__setitem__(1, token))
                with self.assertRaisesRegex(ValueError, "numeric_parse_failure.*not an empty cell"):
                    load_fixture(root, config)

    def test_nonfinite_numeric_values(self):
        for token in ["inf", "-inf"]:
            with self.subTest(token=token), fixture() as (root, config):
                path = root / config["PATHS"]["wide_csv"]["warmup_2000"]["raw_month_end_market_cap"]
                edit_csv(path, lambda rows: rows[1].__setitem__(1, token))
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    load_fixture(root, config)

    def test_month_integrity(self):
        edits = {
            "duplicate": lambda rows: rows[2].__setitem__(0, "2000-01"),
            "missing": lambda rows: rows.pop(2),
            "invalid": lambda rows: rows[1].__setitem__(0, "2000-13"),
            "out_of_scope": lambda rows: rows[1].__setitem__(0, "2022-01"),
            "unordered": lambda rows: rows.__setitem__(slice(1, 3), rows[1:3][::-1]),
        }
        for name, edit in edits.items():
            with self.subTest(case=name), fixture() as (root, config):
                path = root / config["PATHS"]["wide_csv"]["warmup_2000"]["isin"]
                edit_csv(path, edit)
                with self.assertRaises(ValueError):
                    load_fixture(root, config)

    def test_stock_header_and_field_count(self):
        edits = [lambda rows: rows[0].__setitem__(2, "000020"),
                 lambda rows: rows[0].__setitem__(0, "date"),
                 lambda rows: rows[0].__setitem__(1, "20"),
                 lambda rows: rows[1].pop()]
        for i, edit in enumerate(edits):
            with self.subTest(case=i), fixture() as (root, config):
                path = root / config["PATHS"]["wide_csv"]["warmup_2000"]["isin"]
                edit_csv(path, edit)
                with self.assertRaises(ValueError):
                    load_fixture(root, config)

    def test_column_order_and_set_across_variables_and_periods(self):
        for period in DEVELOPMENT_MONTHS:
            for change in [lambda rows: rows[0].__setitem__(slice(1, 3), rows[0][1:3][::-1]),
                           lambda rows: rows[0].__setitem__(1, "999999")]:
                with self.subTest(period=period), fixture() as (root, config):
                    path = root / config["PATHS"]["wide_csv"][period]["isin"]
                    edit_csv(path, change)
                    with self.assertRaisesRegex(ValueError, "stock columns/order mismatch"):
                        load_fixture(root, config, tuple(DEVELOPMENT_MONTHS))

    def test_units_mapping_and_panel_key_integrity(self):
        with fixture() as (root, config):
            for field in NUMERIC_PANEL_UNITS:
                bad = copy.deepcopy(config)
                bad["UNITS"][field] = "unknown"
                with self.subTest(unit=field), self.assertRaises(ValueError):
                    load_fixture(root, bad)
            panel = load_fixture(root, config)
            for bad in [pd.concat([panel, panel.iloc[:1]], ignore_index=True),
                        panel.drop(columns="market_cap")]:
                with self.assertRaises(ValueError):
                    validate_input_schema(bad, contract_name="panel", config=config)
            bad = panel.copy()
            bad.loc[0, "stock_id"] = None
            with self.assertRaises(ValueError):
                validate_input_schema(bad, contract_name="panel", config=config)

    def test_column_mapping_preserves_values_and_dtypes(self):
        source = pd.DataFrame({"source_label": [0.0, np.nan]})
        result = normalize_columns_contract(source, column_mapping={"value": "source_label"})
        pd.testing.assert_series_equal(result["value"], source["source_label"], check_names=False)
        self.assertEqual(source.columns.tolist(), ["source_label"])


@unittest.skip("입력 외 공시 PIT·H2/H4·확장 스키마는 아직 미구현")
def test_schema_contract() -> None:
    """후속 공시·확장 특징의 PIT·상태 계약은 구현 후 검증한다.

    TODO: 기업번호 연결 유효기간 중복, 미래 공시·정정 소급,
    월말 시각불명 다음 기준월 반영을 확인한다.
    H2 점수 식·분모0, 무발표와 수집/추출 실패, 기업규모 표준화·결합 변수를 확인한다.
    H4 기사 중복·검색조건·분자/분모, 기사0건과 실패·유효비율0, 원 모멘텀·곱을 확인한다.
    활성 확장만 필수열·null 명세를 검사하고 특징/목표/확장별 제외사유를 보존한다.
    """
    raise NotImplementedError
