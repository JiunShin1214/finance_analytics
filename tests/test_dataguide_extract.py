# 4단계 DataGuide 추출(dataguide_extract): 엑셀 읽기, 결산월 → 재무상태표 달, 회계기준·단계 선택.

import unittest


class DataGuideEquityTests(unittest.TestCase):
    """DataGuide equity rows dated by DART receipts; tiny synthetic workbooks, no network."""

    @staticmethod
    def workbook(path, rows):
        """Minimal xlsx: one worksheet of inline strings and numbers, enough for the streaming reader."""
        import zipfile
        from xml.sax.saxutils import escape

        def col(i):
            name = ''
            i += 1
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
                ref = f'{col(i)}{r}'
                if isinstance(v, (int, float)):
                    out.append(f'<c r="{ref}"><v>{v}</v></c>')
                else:
                    out.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(v)}</t></is></c>')
            body.append(f'<row r="{r}">{"".join(out)}</row>')
        xml = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
               + ''.join(body) + '</sheetData></worksheet>')
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('xl/worksheets/sheet1.xml', xml)

    def test_reader_keeps_one_copy_of_period_repeated_items_and_refuses_disagreeing_copies(self):
        import tempfile
        from pathlib import Path
        from src.preprocessing.dataguide_extract import read_sheet
        head = [[None, None, None, None, None, '1000930000', '1000930000'],
                [None, None, None, None, None, '자본총계(원)', '자본총계(원)'],
                ['코드', '코드명', '결산월', '회계연도', '분기']]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'a.xlsx'
            self.workbook(path, head + [['A000001', 'x', '12', '2003', '결산', 5000, 5000]])
            frame, info = read_sheet(path, {'1000930000': 'sep'})
            self.assertEqual((frame.loc[0, 'sep'], info['columns_per_item']), (5000, {'sep': 2}))
            self.workbook(path, head + [['A000001', 'x', '12', '2003', '결산', 5000, 6000]])
            with self.assertRaisesRegex(ValueError, 'disagree'):
                read_sheet(path, {'1000930000': 'sep'})

    def test_statement_month_uses_that_years_fiscal_month(self):
        from src.preprocessing.dataguide_extract import statement_month
        self.assertEqual(statement_month('2003', '12', '결산'), '2003-12')
        self.assertEqual(statement_month('2003', '12', '1st-Quarter'), '2003-03')
        self.assertEqual(statement_month('2004', '3', '1st-Quarter'), '2004-06')  # April-March year 2004 ends 2005-03
        self.assertEqual(statement_month('2004', '6', '결산'), '2004-06')

    def frames(self, **extra):
        import pandas as pd
        base = {'code': 'A000001', 'name': '가나', 'fy': '2005', 'quarter': '결산'}
        f = lambda **v: pd.DataFrame([{**base, **v}])
        return {'kgaap_separate': f(sep=100.0), 'kgaap_consolidated': f(cons=150.0, nci=20.0, owners=None),
                'kgaap_fiscal_month': f(fm='12'), 'kifrs_consolidated': f(cons=999.0, owners=900.0, nci=None, fm='12'),
                'kifrs_separate': f(sep=888.0), **extra}

    def test_kgaap_first_through_2010_then_tier_order(self):
        from src.preprocessing.dataguide_extract import combine
        row = combine(self.frames()).iloc[0]
        self.assertEqual((row['std'], row.tier, row.equity), ('KGAAP', 'T1D_consolidated_total_less_nci', 130.0))
