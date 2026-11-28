"""dataguide_files: 파일 이름 대신 항목 코드로 DataGuide 자료 종류를 정하고 잘못 받은 파일을 알려 준다(2026-11-26)."""
import tempfile
import unittest
from pathlib import Path

import yaml

from src.preprocessing import dataguide_files as dg

CONFIG = yaml.safe_load(Path("config/experiment.yaml").read_text(encoding="utf-8"))
TABLE = dg.roles(CONFIG)


def workbook(path, rows):
    """DataGuide 모양의 작은 xlsx(test_dataguide_extract와 같은 방식)."""
    import zipfile
    from xml.sax.saxutils import escape

    def col(i):
        name, i = "", i + 1
        while i:
            i, r = divmod(i - 1, 26)
            name = chr(65 + r) + name
        return name
    body = []
    for r, cells in enumerate(rows, 1):
        out = []
        for i, v in enumerate(cells):
            if v is None:
                continue
            ref = f"{col(i)}{r}"
            out.append(f'<c r="{ref}"><v>{v}</v></c>' if isinstance(v, (int, float))
                       else f'<c r="{ref}" t="inlineStr"><is><t>{escape(v)}</t></is></c>')
        body.append(f'<row r="{r}">{"".join(out)}</row>')
    xml = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
           + "".join(body) + "</sheetData></worksheet>")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", xml)


def sheet(codes, rows):
    """codes: 항목 코드 목록, rows: [(코드, 회계연도, 분기, 값들...)]"""
    head = [[None] * 5 + list(codes), [None] * 5 + ["이름"] * len(codes), ["코드", "코드명", "결산월", "회계연도", "분기"]]
    return head + [[c, "x", "12", fy, q, *vals] for c, fy, q, *vals in rows]


class ClassifyTests(unittest.TestCase):
    def test_codes_decide_role_not_name(self):
        for role in ("kifrs_consolidated", "kifrs_separate", "flows_KIFRS_cons", "industry_hist", "total_return"):
            self.assertEqual(dg.classify(set(TABLE[role]), TABLE), (role, []))

    def test_fiscal_month_only_file_is_fiscal_month(self):
        self.assertEqual(dg.classify({"CP10002200"}, TABLE)[0], "kgaap_fiscal_month")

    def test_partial_file_reports_missing_items(self):
        codes = set(TABLE["kifrs_consolidated"]) - {"6000902001"}
        self.assertEqual(dg.classify(codes, TABLE), ("kifrs_consolidated", ["6000902001"]))

    def test_unknown_codes(self):
        self.assertEqual(dg.classify({"9999999999"}, TABLE), (None, []))

    def test_one_file_with_balance_sheet_and_flows_serves_both(self):
        # 테스트 기간 연결 파일: 재무상태표 6개 + 손익·현금흐름 5개 항목이 한 파일에(2026-11-27)
        codes = set(TABLE["kifrs_consolidated"]) | set(TABLE["flows_KIFRS_cons"])
        self.assertEqual(sorted(dg.complete_roles(codes, TABLE)), ["flows_KIFRS_cons", "kifrs_consolidated"])

    def test_fiscal_month_not_attached_to_kifrs_file(self):
        self.assertEqual(dg.complete_roles(set(TABLE["kifrs_consolidated"]), TABLE), ["kifrs_consolidated"])


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ext = self.root / dg.EXTERNAL

    def tearDown(self):
        self.tmp.cleanup()

    def test_any_name_subfolder_sealed_and_problems(self):
        codes = list(TABLE["kifrs_separate"])
        workbook(self.ext / "아무 이름.xlsx", sheet(codes, [("A000010", "2020", "결산", 1, 2, 3)]))
        workbook(self.ext / "2026" / "재무.xlsx", sheet(codes, [("A000010", "2025", "결산", 1, 2, 3)]))
        workbook(self.ext / "test_sealed" / "t.xlsx", sheet(codes, [("A000010", "2023", "결산", 1, 2, 3)]))
        workbook(self.ext / "partial.xlsx", sheet(["6000901001", "6000903001"], [("A000010", "2020", "결산", 1, 2)]))
        workbook(self.ext / "memo.xlsx", [["메모"], ["아무것도 아님"]])
        s = dg.scan(self.root, CONFIG)
        self.assertEqual([p.name for p in s["files"]["kifrs_separate"]], ["재무.xlsx", "아무 이름.xlsx"])  # test_sealed 제외
        self.assertEqual(s["unknown"], ["data/raw/external_data/memo.xlsx"])
        partial = [p for p in s["problems"] if p["kind"] == "missing_items"]
        self.assertEqual((partial[0]["role"], [c for c, _ in partial[0]["missing"]]),
                         ("kifrs_consolidated", ["6000902001", "6000903003", "6000903004", "CP10002200"]))
        missing_roles = {p["role"] for p in s["problems"] if p["kind"] == "missing_role"}
        self.assertIn("total_return", missing_roles)
        self.assertNotIn("kifrs_consolidated", missing_roles)          # 부분 파일이 있으면 '없음'이 아니라 '빠진 항목'
        self.assertEqual(len(dg.scan(self.root, CONFIG, include_test=True)["files"]["kifrs_separate"]), 3)
        text = dg.report(s, s["problems"])
        self.assertIn("6000902001 부채총계", text)
        self.assertIn("memo.xlsx", text)

    def _values(self, rows_a, rows_b, need_fy=None):
        codes = list(TABLE["kifrs_separate"])
        workbook(self.ext / "a.xlsx", sheet(codes, rows_a))
        workbook(self.ext / "b.xlsx", sheet(codes, rows_b))
        return dg.check_values(self.root, dg.scan(self.root, CONFIG), need_fy=need_fy)

    def test_overlap_same_values_ok_and_years(self):
        p = self._values([("A000010", "2020", "결산", 100, 40, 60)], [("A000010", "2020", "결산", 100, 40, 60),
                                                                       ("A000010", "2021", "결산", 110, 40, 70)], need_fy=2021)
        self.assertEqual(p, [])
        p = self._values([("A000010", "2020", "결산", 100, 40, 60)], [("A000010", "2021", "결산", 1, 1, 1)], need_fy=2025)
        self.assertEqual([(x["kind"], x["have"], x["need"]) for x in p], [("years_short", [2020, 2021], 2025)])

    def test_conflict_unit_and_duplicates(self):
        p = self._values([("A000010", "2020", "결산", 100000, 40000, 60000)],
                         [("A000010", "2020", "결산", 100, 40, 60)])
        self.assertEqual({x["kind"] for x in p}, {"unit_mismatch"})
        p = self._values([("A000010", "2020", "결산", 100, 40, 60)], [("A000010", "2020", "결산", 100, 45, 55)])
        self.assertEqual({(x["kind"], x["item"]) for x in p}, {("value_conflict", "liabilities"), ("value_conflict", "sep")})
        p = self._values([("A000010", "2020", "결산", 100, 40, 60), ("A000010", "2020", "결산", 100, 40, 60)],
                         [("A000020", "2020", "결산", 1, 1, 1)])
        self.assertEqual([x["kind"] for x in p], ["duplicate_rows"])


if __name__ == "__main__":
    unittest.main()
