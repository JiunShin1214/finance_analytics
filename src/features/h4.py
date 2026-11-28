"""H4 특징표: Cho–Kim(2023) 한국 EPU(월 공통) × 팀 11개월 모멘텀.

정의(config H4, 팀원 h4_epu_current.parquet와 같음. 2026-11-26 새 EPU 파일로 174,952행 값·결측 위치 동일 확인).
- epu_raw        = 「Korean New EPU Index」 News-based EPU Index. 기준월 t에 EPU(t − epu_lag_months), 기본 0.
- mom            = mom 표(H1과 같은 t−11~t−1 정확히 11개 달력월 누적 원수익률). mom_available이 아니면 결측.
- h4_interaction = epu_raw × mom. 두 열 모두 값 그대로 쓴다(순위·z 변환 안 함).
- 행 = mom 표의 적격(eligible) 종목×기준월, 기준월 2000-12부터. 목표값은 이 표에 없다.

개발(기본): 기준월 ~2021-11, mom = PATHS.mom. 결과를 PATHS.h4(팀원 파일)와 대조만 한다(쓰지 않음).
최종(--final): 기준월 ~자료 끝의 전달(src.periods), mom = data/final/mom.parquet, 결과 = data/final/<PATHS.h4 이름>.
기준월 범위의 EPU 달이 하나라도 없으면 멈춘다(2026년분 예측 전에 EPU 파일을 새로 받아야 함).

실행: python -m src.features.h4                     (개발 기간 재계산 → PATHS.h4와 대조)
      python -m src.features.h4 --final             (전체 기간 계산·검사만)
      python -m src.features.h4 --final --execute   (data/final에 저장)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import periods
from src.h4.features.epu import load_epu_index

COLUMNS = ['stock_id', 'base_month', 'mom', 'epu_month', 'epu_raw', 'h4_interaction', 'h4_available', 'h4_exclusion_reason']
MOM_COLUMNS = ['stock_id', 'base_month', 'eligible', 'mom', 'mom_available']


def load_epu(root: Path, h4: dict) -> pd.DataFrame:
    """EPU 엑셀 → base_month, epu_month, epu_raw(월 유일, 결측 없음)."""
    epu = load_epu_index(Path(root) / h4['epu_xlsx'])
    lag = int(h4['epu_lag_months'])
    if lag < 0:
        raise ValueError('epu_lag_months_negative')
    months = pd.PeriodIndex(epu.epu_month, freq='M')
    return pd.DataFrame({'base_month': (months + lag).astype(str), 'epu_month': months.astype(str),
                         'epu_raw': epu.epu_raw.to_numpy(float)})


def build_h4(mom: pd.DataFrame, epu: pd.DataFrame, base_months: tuple[str, str]) -> pd.DataFrame:
    """적격 종목×기준월에 mom·EPU를 붙이고 곱을 만든다. 기준월 범위의 EPU가 하나라도 없으면 중단한다."""
    first, last = base_months
    missing = [c for c in MOM_COLUMNS if c not in mom.columns]
    if missing:
        raise ValueError(f'mom_columns_missing:{missing}')
    m = mom[MOM_COLUMNS].copy()
    m['base_month'] = pd.PeriodIndex(m.base_month, freq='M').astype(str)
    m = m[m.base_month.between(first, last)]
    want = set(pd.period_range(first, last, freq='M').astype(str))
    if set(m.base_month) != want:
        raise ValueError(f'mom_base_months_incomplete:{sorted(want - set(m.base_month))[:5]}')
    if m.duplicated(['stock_id', 'base_month']).any():
        raise ValueError('mom_duplicate_keys')
    if not want <= set(epu.base_month):
        raise ValueError(f'epu_months_missing:{sorted(want - set(epu.base_month))[:5]}')
    m['mom'] = pd.to_numeric(m['mom']).where(m.mom_available.eq(True))
    out = m.loc[m.eligible.eq(True), ['stock_id', 'base_month', 'mom']].merge(
        epu, on='base_month', how='left', validate='many_to_one')
    out['h4_interaction'] = out.epu_raw * out.mom
    if np.isinf(out[['mom', 'epu_raw', 'h4_interaction']].to_numpy(float)).any():
        raise ValueError('h4_nonfinite')
    out['h4_available'] = out.epu_raw.notna() & out.mom.notna()
    reason = pd.Series(pd.NA, index=out.index, dtype='string')
    reason[out.mom.isna()] = 'missing_momentum'
    reason[out.epu_raw.isna()] = 'missing_epu'
    out['h4_exclusion_reason'] = reason
    for c in ('stock_id', 'base_month', 'epu_month'):
        out[c] = out[c].astype('string')
    return out[COLUMNS].sort_values(['base_month', 'stock_id']).reset_index(drop=True)


def same_table(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a = a[COLUMNS].sort_values(['base_month', 'stock_id']).reset_index(drop=True)
    b = b[COLUMNS].sort_values(['base_month', 'stock_id']).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    except AssertionError:
        return False
    return True


def ensure_h4(root: Path, config: dict, *, final: bool = False, execute: bool = False) -> dict:
    root = Path(root)
    h4 = config['H4']
    period = periods.resolve(config, root, final=final)
    first, last = h4['base_months']
    if final:
        last = period.last_base_month
    elif last != period.last_base_month:
        raise ValueError('h4_base_months_end_must_match_DATA_PERIOD')
    mom_path, out_path = (periods.table_path(config, k, final=final) for k in ('mom', 'h4'))
    out = build_h4(pd.read_parquet(root / mom_path), load_epu(root, h4), (first, last))
    record = {'final': final, 'base_months': [first, last], 'epu_lag_months': int(h4['epu_lag_months']),
              'mom': mom_path.as_posix(), 'rows': len(out), 'available_rows': int(out.h4_available.sum()),
              'exclusion': out.h4_exclusion_reason.value_counts(dropna=True).to_dict(), 'output': out_path.as_posix()}
    if not final:
        if not same_table(out, pd.read_parquet(root / out_path)):
            raise ValueError(f'development_h4_differs_from_existing:{out_path}')
        record['existing_same'] = True
        return record
    if execute:
        (root / out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(root / out_path, index=False)
        record['written'] = True
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--final', action='store_true', help='전체 기간(src.periods) → DATA_PERIOD.final_dir')
    ap.add_argument('--execute', action='store_true', help='최종 표 저장')
    a = ap.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / 'config/experiment.yaml').read_text(encoding='utf-8'))
    print(json.dumps(ensure_h4(root, config, final=a.final, execute=a.execute), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
