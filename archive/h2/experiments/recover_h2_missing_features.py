# -*- coding: utf-8 -*-
"""H2에 연결하지 못한 기존 H1(B/M·Momentum)·H3(EWMA) Feature 복원.
프로젝트 루트에서 실행 (python -m 없이 실행 가능):
    python -m src.h2.experiments.recover_h2_missing_features               # 파일 존재/스키마 확인만
    python -m src.h2.experiments.recover_h2_missing_features --execute     # 누락된 결과 파일만 생성
기존 산출물은 덮어쓰지 않음. 기존 프로젝트의 원래 Feature 함수를 사용함.
ROA·유동비율·RateChange는 이 스크립트의 생성 범위가 아님.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

FILES = {
    'panel': Path('data/interim/panel.parquet'),
    'equity_asof': Path('data/interim/asof.parquet'),
    'eligibility': Path('data/interim/eligibility.parquet'),
    'h1_components': Path('data/processed/h1_components.parquet'),
    'h3': Path('data/processed/h3.parquet'),
    'rate_change': Path('data/h2/h2_rate_change_panel_2001_2021.csv'),
}


def report(root):
    print('\n[입력/출력 파일 존재 확인]')
    for name, rel in FILES.items():
        p = root / rel
        print(f'{name:15} {"있음" if p.is_file() else "없음"}  {rel}')
    for key in ('h1_components', 'h3', 'equity_asof', 'eligibility'):
        p=root/FILES[key]
        if p.is_file():
            df=pd.read_parquet(p)
            print(f'[{key}] rows={len(df):,}; columns={list(df.columns)}')
            if {'stock_id', 'base_month'}.issubset(df):
                print(f'   중복 종목-기준월: {int(df.duplicated(["stock_id","base_month"]).sum()):,}')
            for col in ('bm', 'mom', 'sigma'):
                if col in df:
                    print(f'   {col} 결측률: {df[col].isna().mean():.2%}')
    print('')


def build_h1(root):
    import pandas as pd
    from src.features.h1 import calculate_size_bm, calculate_momentum_11m
    panel = pd.read_parquet(root / FILES['panel'])
    asof = pd.read_parquet(root / FILES['equity_asof'])
    # 기존 프로젝트 함수의 필수 단위 메타데이터
    panel.attrs['units'] = {'market_cap': 'KRW'}
    asof.attrs['units'] = {'equity': 'KRW'}
    size_bm = calculate_size_bm(panel, asof)
    ret = panel[['stock_id', 'month', 'return']].copy()
    ret.attrs['units'] = {'return': 'decimal'}
    mom = calculate_momentum_11m(ret, size_bm[['stock_id', 'base_month']])
    out = size_bm.merge(mom, on=['stock_id', 'base_month'], how='left', validate='one_to_one')
    if out.duplicated(['stock_id', 'base_month']).any():
        raise ValueError('H1 중복 키')
    return out


def build_h3(root):
    from src.features.h3 import compute_h3_from_panel, h3_output_table
    panel = pd.read_parquet(root / FILES['panel'])
    eligibility = pd.read_parquet(root / FILES['eligibility'])
    features, states, _, _ = compute_h3_from_panel(panel, eligibility, last_base_month='2021-11')
    out = h3_output_table(features, states)
    if out.duplicated(['stock_id', 'base_month']).any():
        raise ValueError('H3 중복 키')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='누락 H1/H3 Feature 생성, 기존 파일 덮어쓰기 없음')
    args = parser.parse_args()
    root = Path.cwd()
    if not (root/'src').is_dir():
        raise SystemExit('finance_analytics 프로젝트 루트에서 실행하세요 (src/ 폴더를 찾을 수 없음).')
    report(root)
    if not args.execute:
        print('검사만 완료했습니다. 실제 생성하려면 --execute를 붙여 실행하세요.')
        return
    if not (root/FILES['panel']).is_file():
        raise SystemExit('data/interim/panel.parquet 없음: 공유 월별 패널 생성이 먼저 필요합니다.')
    for name, reqs, creator in (
        ('h1_components', ('equity_asof',), build_h1),
        ('h3', ('eligibility',), build_h3),
    ):
        dst = root/FILES[name]
        if dst.is_file():
            print(f'[유지] {dst.relative_to(root)} (기존 파일 덮어쓰지 않음)')
            continue
        absent = [r for r in reqs if not (root/FILES[r]).is_file()]
        if absent:
            print(f'[미생성] {name}: 필요한 입력 파일 없음: {absent}')
            continue
        try:
            df=creator(root)
        except Exception as e:
            print(f'[실패] {name}: {type(e).__name__}: {e}')
            print('기존 입력 규격(공시일/적격성 포함)을 확인하세요. 임의 값으로 대체하지 않습니다.')
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f'[유지] 실행 도중 기존 파일 발견: {dst}')
            continue
        df.to_parquet(dst, index=False)
        print(f'[생성] {dst.relative_to(root)} : {len(df):,}행 / {list(df.columns)}')
    print('\n이후 python run_h2_factorial_models.py 를 다시 실행할 수 있습니다.')

if __name__=='__main__':
    main()
