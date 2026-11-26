"""dart_filings: 시장 전체 목록 수집 결과를 공통 DART 표에 덧붙이는 단계(2026-11-26). 가짜 자료만 쓴다."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.contracts import DART_FILING_INPUT_COLUMNS
from src.preprocessing import dart_filings as df

RUN = df.MARKET_RUN / "market_20220101_20221231"


def _existing() -> pd.DataFrame:
    rows = [
        {"corp_code": "00000001", "corp_name": "가", "stock_code": "000010", "receipt_id": "20210315000001",
         "published_at": "20210315", "report_name": "사업보고서 (2020.12)", "report_kind": "사업", "report_period_label": "2020.12",
         "revision": "최초표시", "listed_in_public_index": True, "disclosure_format": "structured_supported",
         "source_path": "data/raw/opendart/list_only/a.json", "source_sha256": "x"},
        {"corp_code": "00000002", "corp_name": "나", "stock_code": "000020", "receipt_id": "20220315000002",
         "published_at": "20220315", "report_name": "사업보고서 (2021.12)", "report_kind": "사업", "report_period_label": "2021.12",
         "revision": "최초표시", "listed_in_public_index": False, "disclosure_format": "document_only",
         "source_path": None, "source_sha256": None},
    ]
    t = pd.DataFrame(rows)[list(DART_FILING_INPUT_COLUMNS)]
    for c in DART_FILING_INPUT_COLUMNS:
        if c != "listed_in_public_index":
            t[c] = t[c].astype("string")
    return t


def _collected(rows, status="complete", start="20220101", end="20221231"):
    base = {"corp_name": "", "stock_code": "", "corp_cls": "Y", "flr_nm": "", "rm": "", "source_path": "data/raw/opendart/list_only/n.json",
            "source_sha256": "y", "report_kind": "사업", "report_period_label": "2021.12", "bsns_year_status": "unconfirmed"}
    frame = pd.DataFrame([{**base, **r} for r in rows]).astype("string")
    progress = {"criteria": {"receipt_start": start, "receipt_end": end}, "completed_windows": [[start, end]], "status": status}
    return frame, progress


class DartFilingsMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data/interim").mkdir(parents=True)
        _existing().to_parquet(self.root / df.DART_FILINGS, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_run(self, rows, **kw):
        frame, progress = _collected(rows, **kw)
        (self.root / RUN).mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.root / RUN / "filings.parquet", index=False)
        (self.root / RUN / "progress.json").write_text(json.dumps(progress), encoding="utf-8")

    def test_dry_run_does_not_write(self):
        self._write_run([{"corp_code": "00000003", "rcept_no": "20220330000003", "rcept_dt": "20220330",
                          "report_nm": "[기재정정]사업보고서 (2021.12)"}])
        before = (self.root / df.DART_FILINGS).read_bytes()
        out = df.run(self.root, execute=False)                       # 기본 = market_list 아래 폴더 전부
        self.assertEqual(out["summary"]["added"], 1)
        self.assertEqual((self.root / df.DART_FILINGS).read_bytes(), before)
        self.assertFalse((self.root / df.BACKUPS).exists())
        self.assertFalse((self.root / "outputs").exists())

    def test_execute_appends_keeps_order_and_promotes(self):
        self._write_run([{"corp_code": "00000003", "rcept_no": "20220330000003", "rcept_dt": "20220330",
                          "report_nm": "[기재정정]사업보고서 (2021.12)"},
                         {"corp_code": "00000002", "rcept_no": "20220315000002", "rcept_dt": "20220315",
                          "report_nm": "사업보고서 (2021.12)"}])
        out = df.run(self.root, [RUN], execute=True)
        t = pd.read_parquet(self.root / df.DART_FILINGS)
        self.assertEqual(t.receipt_id.tolist(), ["20210315000001", "20220315000002", "20220330000003"])   # 기존 순서 + 새 공시
        self.assertTrue(bool(t.listed_in_public_index.iloc[1]))                     # 추출로만 알던 공시가 공개 목록에 나옴
        self.assertEqual(t.disclosure_format.iloc[1], "document_only")              # 기존 값은 그대로
        self.assertEqual(t.revision.iloc[2], "기재정정")
        self.assertTrue(pd.isna(t.disclosure_format.iloc[2]))
        self.assertEqual(list(t.columns), list(DART_FILING_INPUT_COLUMNS))
        self.assertEqual(out["summary"]["promoted_to_public"], 1)
        backup = self.root / out["backup"]
        self.assertTrue(backup.exists() and backup.with_suffix(".json").exists())
        pd.testing.assert_frame_equal(pd.read_parquet(backup), _existing(), check_dtype=False)
        self.assertFalse((self.root / "outputs").exists())
        again = df.run(self.root, execute=True)                       # 같은 명령을 다시: 추가 0, 쓰지 않음
        self.assertTrue(again.get("unchanged"))
        self.assertEqual(len(list((self.root / df.BACKUPS).glob("*.parquet"))), 1)
        pd.testing.assert_frame_equal(t.iloc[:1].reset_index(drop=True),
                                      _existing().iloc[:1].reset_index(drop=True), check_dtype=False)

    def test_identity_conflict_stops(self):
        self._write_run([{"corp_code": "99999999", "rcept_no": "20220315000002", "rcept_dt": "20220315",
                          "report_nm": "사업보고서 (2021.12)"}])
        with self.assertRaisesRegex(ValueError, "dart_filing_identity_conflict"):
            df.run(self.root, [RUN])

    def test_incomplete_run_stops(self):
        self._write_run([{"corp_code": "00000003", "rcept_no": "20220330000003", "rcept_dt": "20220330",
                          "report_nm": "사업보고서 (2021.12)"}], status="stopped")
        with self.assertRaisesRegex(ValueError, "dart_list_run_not_complete"):
            df.run(self.root, [RUN])

    def test_receipt_outside_window_stops(self):
        self._write_run([{"corp_code": "00000003", "rcept_no": "20230105000003", "rcept_dt": "20230105",
                          "report_nm": "사업보고서 (2021.12)"}])
        with self.assertRaisesRegex(ValueError, "dart_list_receipt_outside_window"):
            df.run(self.root, [RUN])


if __name__ == "__main__":
    unittest.main()
