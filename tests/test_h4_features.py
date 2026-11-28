"""src/features/h4.py: 가짜 mom·EPU로 기준월 범위·결합·결측 사유를 확인한다(실제 자료는 읽지 않음)."""
import unittest

import numpy as np
import pandas as pd

from src.features import h4

RANGE = ('2000-12', '2025-11')


def fake_mom(first, last):
    months = pd.period_range(first, last, freq='M').astype(str)
    rows = [(s, m) for m in months for s in ('000010', '000020', '000030')]
    mom = pd.DataFrame(rows, columns=['stock_id', 'base_month'])
    mom['eligible'] = mom.stock_id.ne('000030')
    mom['mom'] = np.linspace(-0.5, 0.5, len(mom))
    mom['mom_available'] = ~(mom.stock_id.eq('000020') & mom.base_month.eq('2021-12'))
    return mom


def fake_epu(first, last):
    months = pd.period_range(first, last, freq='M')
    return pd.DataFrame({'base_month': months.astype(str), 'epu_month': months.astype(str),
                         'epu_raw': np.arange(len(months), dtype=float) + 100})


class BuildH4Tests(unittest.TestCase):
    def test_rows_and_values(self):
        out = h4.build_h4(fake_mom('2000-01', '2025-12'), fake_epu('1990-01', '2025-11'), RANGE)
        self.assertEqual((out.base_month.min(), out.base_month.max()), RANGE)
        self.assertEqual(len(out), 300 * 2)                   # 적격 2종목 × 300개월, 범위 밖 달은 버림
        self.assertTrue(np.allclose(out.h4_interaction.dropna(), (out.epu_raw * out.mom).dropna()))
        miss = out[out.h4_exclusion_reason.eq('missing_momentum')]
        self.assertEqual(miss[['stock_id', 'base_month']].values.tolist(), [['000020', '2021-12']])
        self.assertTrue(out.epu_month.eq(out.base_month).all())   # 기준월 t에 EPU(t)

    def test_epu_gap_stops(self):
        with self.assertRaisesRegex(ValueError, 'epu_months_missing'):
            h4.build_h4(fake_mom(*RANGE), fake_epu('1990-01', '2025-10'), RANGE)

    def test_mom_gap_stops(self):
        with self.assertRaisesRegex(ValueError, 'mom_base_months_incomplete'):
            h4.build_h4(fake_mom('2000-12', '2025-10'), fake_epu('1990-01', '2025-11'), RANGE)


if __name__ == '__main__':
    unittest.main()
