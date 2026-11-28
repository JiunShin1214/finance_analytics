"""기본(통제) 피처: 종목×기준월 표 하나(data/processed/base_features.parquet).

정의·근거·결정은 config BASE_FEATURES(2026-11-19~21 사용자 결정, 노션 「12주차 실행」 2-2)를 따른다.
중간 집계(marcap 월 집계, 시장 일수익률, 연간 재무)는 파일로 남기지 않고 메모리에서 계산한다.
입력은 data/ 파일과 config뿐이며 이전 실행 기록을 읽지 않는다. 테스트 기간(2022~) 파일은 열지 않는다.
예외: --final(최종 표, src.periods)은 교수님 CSV 마지막 달까지 모든 자료(테스트 봉인 폴더 포함)를 읽고 DATA_PERIOD.final_dir에 쓴다.

열: stock_id, base_month, eligible, financial(금융업 판정), 피처 24개(종목 20 + 시장 공통 4), 피처별 <이름>_reason(값이 없을 때 이유).
시장 공통 4개(mkt_ret, mkt_ret3, mkt_vol12, breadth)는 같은 달 모든 종목이 같은 값이다.

실행: python -m src.features.base             (계산·검사만, 쓰지 않음)
      python -m src.features.base --execute   (base_features.parquet 쓰기)
      python -m src.features.base --final --execute   (전체 기간 → data/final/base_features.parquet)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.preprocessing import dataguide_files
from src.preprocessing.dataguide_extract import iter_rows, read_sheet, sha256

STOCK_FEATURES = ('str', 'ltr', 'high52', 'turnover', 'ln_dvol', 'illiq', 'ivol', 'max_ret', 'beta', 'ln_price', 'roa', 'ep', 'dy', 'rmw',
                  'asset_growth', 'gpa', 'cfp', 'sp', 'pct_accruals', 'sales_growth')
MARKET_FEATURES = ('mkt_ret', 'mkt_ret3', 'mkt_vol12', 'breadth')
FEATURES = STOCK_FEATURES + MARKET_FEATURES
DAILY_FEATURES = ('ln_dvol', 'illiq', 'ivol', 'max_ret')
ACCOUNTING_FEATURES = ('roa', 'ep', 'rmw', 'asset_growth', 'gpa', 'cfp', 'sp', 'pct_accruals', 'sales_growth')
FINANCIAL_EXCLUDED = ('asset_growth', 'gpa', 'cfp', 'sp', 'pct_accruals', 'sales_growth')   # 금융업 결측(2026-11-20~21 결정)
NUMERATOR = {'roa': 'ni_owner', 'ep': 'ni_owner', 'rmw': 'op_income', 'asset_growth': 'assets_cur', 'gpa': 'sales', 'cfp': 'cfo',
             'sp': 'sales', 'pct_accruals': 'ni_owner', 'sales_growth': 'sales'}   # 값이 없을 때 이유 분류용 대표 입력
FLOW_ITEMS = ('ni_owner', 'op_income', 'sales', 'cogs', 'cfo')
ACC_VALUES = ('ni_owner', 'op_income', 'sales', 'cogs', 'cfo', 'assets_cur', 'assets_prev', 'sales_prev')
MARCAP_COLUMNS = ['Code', 'Date', 'Close', 'ChangesRatio', 'Volume', 'Amount', 'High', 'Marcap', 'Market']
CONSOLIDATED_TIERS = ('T1_consolidated_owners', 'T1D_consolidated_total_less_nci', 'T2_consolidated_total')
EXCEL_EPOCH = pd.Timestamp('1899-12-30')
EXTERNAL = Path('data/raw/external_data')


def guard_development(path) -> None:
    """테스트 기간 파일은 경로 단계에서 막는다."""
    s = Path(path).as_posix()
    if 'test_sealed' in s or 'test_2022_2025' in s:
        raise PermissionError('test_period_input_blocked:' + s)


def _wide(s: pd.Series, months) -> pd.DataFrame:
    """(stock_id, month) 색인 → 월 × 종목 표(모든 달 행)."""
    return s.unstack('stock_id').reindex(pd.Index(months, name='month'))


def _long(w: pd.DataFrame, name: str) -> pd.Series:
    s = w.stack(future_stack=True).rename(name)
    s.index = s.index.set_names(['month', 'stock_id'])
    return s.swaplevel().sort_index()


# ---------------------------------------------------------------- 일별 시장 자료(marcap)

def load_marcap(root: Path, cfg: dict, stock_ids: set[str], last_year: int = 2021) -> tuple[pd.DataFrame, pd.DataFrame]:
    """marcap 2000~last_year(개발 2021, 최종은 자료 끝 연도). (패널 종목의 일별 행, 시장수익률용 KOSPI 전 종목 행).
    종목코드 6자리, 일수익률은 소수."""
    keep, kospi = [], []
    for f in sorted(glob.glob(str(root / cfg['inputs']['marcap_dir'] / 'marcap-*.parquet'))):
        if int(Path(f).stem.split('-')[1]) > last_year:
            continue
        if last_year <= 2021:
            guard_development(f)
        m = pd.read_parquet(f, columns=MARCAP_COLUMNS)
        m['stock_id'] = m.Code.astype(str).str.strip().str.zfill(6)   # 2000-01~05 파일은 앞 0이 빠져 있음
        m['date'] = pd.to_datetime(m.Date)
        m['r'] = m.ChangesRatio.astype(float) / 100
        kospi.append(m.loc[m.Market.eq('KOSPI'), ['stock_id', 'date', 'r', 'Marcap']])
        keep.append(m.loc[m.stock_id.isin(stock_ids), ['stock_id', 'date', 'r', 'Close', 'High', 'Volume', 'Amount']])
    def tidy(x):
        return x.sort_values(['stock_id', 'date']).drop_duplicates(['stock_id', 'date'], keep='last').reset_index(drop=True)
    daily = tidy(pd.concat(keep, ignore_index=True))
    daily['month'] = daily.date.dt.strftime('%Y-%m')
    return daily, tidy(pd.concat(kospi, ignore_index=True))


def market_daily_return(kospi: pd.DataFrame) -> pd.Series:
    """B-3: KOSPI 전 종목 일수익률을 전일 시가총액으로 가중 평균한 시장 일수익률(날짜 색인)."""
    k = kospi.copy()
    days = np.sort(k.date.unique())
    prev_day = pd.Series(days[:-1], index=days[1:])
    g = k.groupby('stock_id')
    k['w'] = g.Marcap.shift(1)
    # 바로 앞 거래일의 행일 때만 가중치로 쓴다(종목코드 재사용: 000030은 2002년 폐지 회사 → 2014-11-19 우리은행 신규 상장)
    k = k[g.date.shift(1).eq(k.date.map(prev_day)) & k.w.gt(0) & k.r.notna()]
    k['wr'] = k.w * k.r
    g = k.groupby('date')
    return (g.wr.sum() / g.w.sum()).rename('mkt_r')


def daily_monthly(daily: pd.DataFrame, mkt: pd.Series, ivol_min_days: int) -> pd.DataFrame:
    """종목×월 집계: 거래일수, 거래량·거래대금, MAX, Amihud, 고유 변동성, 주가수익률(월 누적), 52주 고가용 값."""
    d = daily.copy()
    d['lr'] = np.log1p(d.r.fillna(0))
    d['idx'] = np.exp(d.groupby('stock_id').lr.cumsum())          # 수정주가 지수(분할 보정 일수익률 누적)
    d['traded'] = d.Volume.gt(0)                                   # 거래정지일(거래량 0)은 일별 통계에서 뺌
    ok_hi = d.traded & d.High.gt(0) & d.Close.gt(0)
    d['adj_high'] = (d.High * d.idx / d.Close).where(ok_hi)
    d['mkt_r'] = d.date.map(mkt)
    key = ['stock_id', 'month']
    out = d.groupby(key).agg(days=('date', 'size'), last_idx=('idx', 'last'), lr=('lr', 'sum'), hi_m=('adj_high', 'max'))
    out['r_px'] = np.expm1(out.pop('lr'))
    t = d[d.traded]
    out = out.join(t.groupby(key).agg(trade_days=('date', 'size'), vol=('Volume', 'sum'), amt=('Amount', 'sum'), max_ret=('r', 'max')))
    am = t[t.Amount.gt(0) & t.r.notna()]
    out = out.join((am.r.abs() / am.Amount).groupby([am.stock_id, am.month]).mean().rename('illiq'))
    # 고유 변동성: t월 거래일 일수익률을 시장 일수익률에 회귀한 잔차 표준편차(자유도 n-2)
    x = t[t.mkt_r.notna() & t.r.notna()]
    s = x.assign(xy=x.r * x.mkt_r, xx=x.mkt_r ** 2, yy=x.r ** 2).groupby(key).agg(
        n=('r', 'size'), sx=('mkt_r', 'sum'), sy=('r', 'sum'), sxy=('xy', 'sum'), sxx=('xx', 'sum'), syy=('yy', 'sum'))
    cxx, cxy, cyy = s.sxx - s.sx ** 2 / s.n, s.sxy - s.sx * s.sy / s.n, s.syy - s.sy ** 2 / s.n
    ssr = (cyy - (cxy ** 2 / cxx).where(cxx > 0)).clip(lower=0)
    s['ivol'] = np.sqrt(ssr / (s.n - 2)).where((s.n >= ivol_min_days) & (cxx > 0))
    out = out.join(s[['ivol']]).reset_index()
    out['trade_days'] = out.trade_days.fillna(0).astype(int)
    out[['vol', 'amt']] = out[['vol', 'amt']].fillna(0)            # 한 달 내내 거래정지면 거래량·거래대금 0
    return out


def weekly_beta(daily: pd.DataFrame, mkt: pd.Series, months: list[str], window: int = 156, min_weeks: int = 52) -> pd.DataFrame:
    """B-4: 기준월 말까지 직전 156주 주별 수익률 회귀 기울기. 거래가 있던 주가 52주 이상일 때만."""
    d = daily[['stock_id', 'date', 'r', 'Volume']].copy()
    d['week'] = d.date.dt.to_period('W-FRI')
    d['lr'] = np.log1p(d.r.fillna(0))
    d['traded'] = d.Volume.gt(0)
    w = d.groupby(['stock_id', 'week']).agg(lr=('lr', 'sum'), traded=('traded', 'any')).reset_index()
    w = w[w.traded]
    mw = np.expm1(np.log1p(mkt).groupby(mkt.index.to_period('W-FRI')).sum())
    # 창은 달력 기준 156주(정확히 3년): 휴장으로 시장 자료가 없는 주도 칸을 두어 창이 과거로 늘어나지 않게 한다
    weeks = pd.period_range(mw.index.min(), mw.index.max(), freq='W-FRI')
    y = np.expm1(w.pivot(index='week', columns='stock_id', values='lr')).reindex(weeks)
    Y = y.to_numpy()
    X = np.where(np.isnan(Y), np.nan, mw.reindex(weeks).to_numpy()[:, None])
    ends = weeks.end_time.normalize()
    rows = {}
    for m in months:   # 기준월 t: 주 끝(금요일)이 t월 말일 이하인 마지막 주에서 끝나는 156주 창을 직접 잘라 계산
        k = int(np.searchsorted(ends, pd.Period(m, 'M').to_timestamp(how='end').normalize(), side='right'))
        if k == 0:
            continue
        yy, xx = Y[max(0, k - window):k], X[max(0, k - window):k]
        n = np.sum(~np.isnan(yy), axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            xc, yc = xx - np.nanmean(xx, axis=0), yy - np.nanmean(yy, axis=0)   # 평균을 뺀 뒤 곱해 합(누적 오차 방지)
            b_ = np.nansum(xc * yc, axis=0) / np.nansum(xc * xc, axis=0)
        rows[m] = pd.Series(np.where(n >= min_weeks, b_, np.nan), index=y.columns)
    b = pd.DataFrame(rows).T
    b.index.name = 'month'
    return _long(b, 'beta')


# ---------------------------------------------------------------- 월 자료(KOSPI 지수, 배당 포함 수익률)

def kospi_index_file(root: Path, cfg: dict) -> Path:
    """KOSIS 코스피 지수 CSV. config 값은 파일 이름 규칙(예: 코스피_지수_*.csv)이고, 여럿이면 이름(받은 시각)이 가장 늦은 것."""
    files = sorted((root / cfg['inputs']['kospi_index_csv']).parent.glob(Path(cfg['inputs']['kospi_index_csv']).name))
    if not files:
        raise FileNotFoundError('kospi_index_csv_missing:' + cfg['inputs']['kospi_index_csv'])
    return files[-1]


def kospi_monthly(root: Path, cfg: dict) -> pd.Series:
    """월말 지수(연월말 행). KOSIS 받는 방식에 따라 인코딩(UTF-8/cp949)과 구분 열 수(2~3개)가 달라서
    'YYYY.MM' 모양인 열만 월로 읽는다(2026-11-26)."""
    path = kospi_index_file(root, cfg)
    for enc in ('utf-8-sig', 'cp949'):
        try:
            k = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    row = k[k.iloc[:, 0].astype(str).str.contains('연월말')].iloc[0]
    months = [c for c in k.columns if re.fullmatch(r'\d{4}\.\d{2}', str(c))]
    level = pd.to_numeric(row[months], errors='coerce').astype(float)
    level.index = [c.replace('.', '-') for c in months]
    return level


def total_return_monthly(root: Path, cfg: dict, *, include_test: bool = False) -> pd.DataFrame:
    """DataGuide 수정주가수익률(현금배당포함), 종목×월, 소수. 파일은 항목 코드로 찾는다(dataguide_files)."""
    rows = []
    for path in dataguide_files.paths(root, 'total_return', include_test=include_test):
        codes = {}
        for i, r in iter_rows(path):
            keys = sorted(r)
            if i == 9:
                codes = {k: r[k] for k in keys[1:]}
            if i <= 14:
                continue
            m = (EXCEL_EPOCH + pd.Timedelta(days=float(r[keys[0]]))).strftime('%Y-%m')
            rows += [(codes[k][1:], m, float(r[k]) / 100) for k in keys[1:] if k in codes and r.get(k) not in (None, '')]
    return pd.DataFrame(rows, columns=['stock_id', 'month', 'r_tot']).drop_duplicates(['stock_id', 'month'], keep='last')


# ---------------------------------------------------------------- 연간 재무(ROA, E/P)

def annual_statements(root: Path, *, include_test: bool = False) -> pd.DataFrame:
    """결정 ①③④: H1 자기자본과 같은 공시일·재무제표 선택(equity_dating.build)의 사업보고서(연간) 행."""
    from src.preprocessing.equity_dating import build
    panel, _, _, _ = build(root, include_test=include_test)   # 약 8~9분(2026-11-21 측정 506초)
    return select_annual(panel)


def select_annual(panel: pd.DataFrame) -> pd.DataFrame:
    if 'panel' in panel:
        panel = panel[panel.panel]
    # H1 자기자본이 쓰지 않는 행(공시일 없음, 연결이 보고서에 없어 값 없음, 항등식 검사 실패 제외)은 재무 피처에서도 쓰지 않는다
    a = panel[panel.kind.eq('사업') & panel.rcept_no.notna() & panel.available_month.notna()
              & panel.tier.isin(CONSOLIDATED_TIERS + ('T3_separate_total',)) & panel.exclude_reason.isna()].copy()
    a['side'] = np.where(a.tier.isin(CONSOLIDATED_TIERS), 'cons', 'sep')
    return a[['code', 'stock_code', 'fy', 'month', 'std', 'tier', 'side', 'available_month', 'rcept_dt', 'cons_assets', 'sep_assets']]


def annual_flows(root: Path, cfg: dict, *, include_test: bool = False) -> pd.DataFrame:
    """(code, fy, std, side) → 연간 지배주주 순이익, 영업이익(분기 누적의 결산 행). 파일은 항목 코드로 찾는다."""
    parts = []
    for key, spec in cfg['inputs']['flows_xlsx'].items():
        std, side = key.split('_')
        f, _ = dataguide_files.read_role(root, f'flows_{key}', {str(k): v for k, v in spec['items'].items()},
                                         include_test=include_test)
        f = f[f.quarter.eq('결산')]
        f = f.assign(**{i: np.nan for i in FLOW_ITEMS if i not in f})
        parts.append(f[['code', 'fy', *FLOW_ITEMS]].assign(std=std, side=side))
    return pd.concat(parts, ignore_index=True).drop_duplicates(['code', 'fy', 'std', 'side'])


# ---------------------------------------------------------------- 금융업 판정

def read_hist_industry(path: Path, item: str) -> pd.DataFrame:
    """시장 대표 업종(Hist.): 연도마다 같은 아이템 열이 반복되고, 한 행(회사×결산)에는 그 연도 칸에만 값이 있다."""
    rows, base, cols, codes, prev = [], None, [], {}, {}
    for _, c in iter_rows(path):
        if base is None:
            if {'코드', '분기'} <= set(c.values()):
                base = next(i for i, v in c.items() if v == '코드')
                cols = [i for i, v in codes.items() if i >= base + 5 and v == item]
                if not cols:
                    raise ValueError('dataguide_items_missing:' + path.name)
            else:
                codes, prev = prev, c
            continue
        code = str(c.get(base, ''))
        if not code.startswith('A'):
            continue
        vals = {c[i] for i in cols if c.get(i) not in (None, '')}
        if len(vals) > 1:
            raise ValueError('hist_values_disagree:' + code)
        rows.append({'code': code, 'fm': c.get(base + 2), 'fy': c.get(base + 3), 'industry': next(iter(vals)) if vals else None})
    return pd.DataFrame(rows)


def financial_flags(root: Path, cfg: dict, *, include_test: bool = False) -> tuple[dict, dict]:
    """금융업 판정(2026-11-23 사용자 B): KSIC 11차 대분류 K(Static, 현재 분류)이고 그 사업보고서 결산월의 시장 대표 업종(Hist.)도
    금융일 때만 금융. 시점은 재무 피처와 같은 사업보고서 공시월(결정 A). Hist. 값이 없는 결산월은 그 회사의 가장 가까운
    이전 결산월 값을 이어 쓰고, 이전 값도 없으면 Static K를 따른다."""
    from src.preprocessing.dataguide_extract import statement_month
    ind = cfg['inputs']['industry']
    s, _ = dataguide_files.read_role(root, 'industry_static', {ind['ksic_section_item']: 'ksic'}, text=('ksic',),
                                     include_test=include_test)
    static_k = {c[1:] for c in s.code[s.ksic.eq('K')]}
    hs = [read_hist_industry(p, ind['hist_item']) for p in dataguide_files.paths(root, 'industry_hist', include_test=include_test)]
    h = hs[0] if len(hs) == 1 else pd.concat(hs, ignore_index=True).drop_duplicates(['code', 'fm', 'fy'])
    h = h[h.industry.notna()]
    no_fm = int(h.fm.isna().sum())   # 결산월 빈칸(2026-11-23 확인: 9행, 일반서비스 2개사)은 결산월을 정할 수 없어 뺀다
    h = h[h.fm.notna()]
    h = h.assign(month=[statement_month(y, m, '결산') for y, m in zip(h.fy, h.fm)], fin=h.industry.str.contains('금융'),
                 stock_id=h.code.str[1:])
    h = h.groupby(['stock_id', 'month']).fin.any().reset_index()
    hist = {k: sorted(zip(g.month, g.fin)) for k, g in h[h.stock_id.isin(static_k)].groupby('stock_id')}
    return {'static_k': static_k, 'hist': hist}, {'static_k_companies': len(static_k), 'hist_company_statements': int(len(h)),
                                                 'hist_rows_without_fiscal_month': no_fm}


def is_financial(fin: dict | None, stock_id: str, statement_month: str) -> bool:
    """financial_flags 규칙으로 한 종목·사업보고서(결산월)의 금융 여부."""
    if fin is None or stock_id not in fin['static_k']:
        return False
    best = None
    for m, f in fin['hist'].get(stock_id, []):   # 결산월 오름차순
        if m > statement_month:
            break
        best = f
    return True if best is None else bool(best)


# ---------------------------------------------------------------- 연간 재무 as-of

def accounting_asof(stmts: pd.DataFrame, flows: pd.DataFrame, months: list[str], last_days: pd.Series,
                    fin: dict | None = None) -> pd.DataFrame:
    """종목×기준월: 기준월까지 공개된 가장 최근 사업보고서의 연간 값(지배순이익, 영업이익, 매출, 매출원가, 영업현금흐름),
    같은 재무제표(연결/개별)의 당해·전년 말 총자산과 전년 매출, 그 보고서 결산월 기준 금융업 여부."""
    flows = flows.assign(**{i: np.nan for i in FLOW_ITEMS if i not in flows})
    s = stmts.merge(flows, on=['code', 'fy', 'std', 'side'], how='left')
    # 전년 값은 전년 사업보고서 행에서 올해와 같은 재무제표(연결/개별) 쪽을 쓴다: 두 쪽 매출을 모두 붙여 둔다
    for side in ('cons', 'sep'):
        f = flows[flows.side.eq(side)][['code', 'fy', 'std', 'sales']].rename(columns={'sales': side + '_sales'})
        s = s.merge(f.drop_duplicates(['code', 'fy', 'std']), on=['code', 'fy', 'std'], how='left')
    cols = {'cons_assets': 'cons_aprev', 'sep_assets': 'sep_aprev', 'cons_sales': 'cons_sprev', 'sep_sales': 'sep_sprev'}
    prev = s[['stock_code', 'month', *cols]].drop_duplicates(['stock_code', 'month'])
    prev = prev.assign(month=(pd.PeriodIndex(prev.month, freq='M') + 12).astype(str)).rename(columns=cols)
    s = s.merge(prev, on=['stock_code', 'month'], how='left')
    cons = s.side.eq('cons')
    s['assets_cur'] = np.where(cons, s.cons_assets, s.sep_assets)
    s['assets_prev'] = np.where(cons, s.cons_aprev, s.sep_aprev)
    s['sales_prev'] = np.where(cons, s.cons_sprev, s.sep_sprev)
    s['financial'] = [is_financial(fin, c, m) for c, m in zip(s.stock_code, s.month)]
    last = {str(k): v for k, v in last_days.items()}
    rows = []
    for code, g in s.groupby('stock_code'):
        recs = g.sort_values(['available_month', 'month']).to_dict('records')
        i, best = 0, None
        for t in months:
            while i < len(recs) and recs[i]['available_month'] <= t:
                if best is None or recs[i]['month'] >= best['month']:
                    best = recs[i]
                i += 1
            if best is None:
                continue
            expiry = (pd.Period(best['month'], 'M') + 18).to_timestamp(how='end').normalize()   # P3
            if last.get(t, pd.Period(t, 'M').to_timestamp(how='end')) > expiry:
                rows.append((code, t, *[np.nan] * len(ACC_VALUES), best['month'], best['side'], best['financial'], 'expired_18m'))
            else:
                rows.append((code, t, *[best[v] for v in ACC_VALUES], best['month'], best['side'], best['financial'], None))
    return pd.DataFrame(rows, columns=['stock_id', 'base_month', *ACC_VALUES, 'statement_month', 'side', 'financial', 'acc_reason'])


def industry_features(acc: pd.DataFrame, cap: pd.Series) -> dict:
    """금융업 결측 피처 6개(2026-11-20~21 결정 B-5, C-1~C-3, GP/A). cap은 acc 색인 순서의 기준월 t 말 시가총액."""
    pos = lambda x: x.where(x.gt(0))
    capv = pos(pd.Series(np.asarray(cap, dtype=float), index=acc.index))
    f = {
        'asset_growth': acc.assets_cur / pos(acc.assets_prev) - 1,                     # Cooper·Gulen·Schill 2008
        'gpa': (acc.sales - pos(acc.cogs)) / pos(acc.assets_cur),                      # Novy-Marx 2013, 매출원가 0·빈칸 결측
        'cfp': acc.cfo / capv,                                                          # Desai 등 2004 CFO/P, 음수 그대로
        'sp': acc.sales / capv,                                                         # Barbee 등 1996
        'pct_accruals': (acc.ni_owner - acc.cfo) / acc.ni_owner.abs().where(acc.ni_owner.ne(0)),   # Hafzalla 등 2011
        'sales_growth': acc.sales / pos(acc.sales_prev) - 1,                            # Lakonishok 등 1994, GHZ sgr
    }
    nonfin = ~acc.financial.fillna(False).astype(bool)
    return {k: v.where(nonfin) for k, v in f.items()}


def equity_asof(root: Path, config: dict, *, final: bool = False) -> pd.Series:
    """B/M과 같은 자기자본(H1 as-of의 usable 값), (stock_id, base_month) 색인.
    as-of는 0 이하(자본잠식)를 usable=False, reason=nonpositive_equity로 두므로 그 값은 남겨 이유 표시에 쓰고, RMW에서는 결측."""
    from src import periods
    a = pd.read_parquet(root / periods.table_path(config, 'asof', final=final),
                        columns=['stock_code', 'base_month', 'equity', 'usable', 'reason'])
    a = a[a.usable.eq(True) | a.reason.eq('nonpositive_equity')]
    return a.set_index(['stock_code', 'base_month']).equity.rename_axis(['stock_id', 'base_month']).astype(float)


# ---------------------------------------------------------------- 조립

def build_base(root: Path, config: dict, *, statements: pd.DataFrame | None = None,
               include_test: bool = False) -> tuple[pd.DataFrame, dict]:
    """include_test=True: 최종 표(src.periods final). 기준월 끝 = 자료 끝의 전달, 입력은 data/final 표."""
    from src import periods
    cfg = config['BASE_FEATURES']
    root = Path(root)
    period = periods.resolve(config, root, final=include_test)
    first, last = cfg['base_months']
    if include_test:
        last = period.last_base_month
    elif last != period.last_base_month:
        raise ValueError('base_months_end_must_match_DATA_PERIOD')
    months = pd.period_range(first, last, freq='M').astype(str).tolist()
    panel = pd.read_parquet(root / periods.table_path(config, 'panel', final=include_test),
                            columns=['stock_id', 'month', 'return', 'market_cap', 'price', 'shares_outstanding', 'eligible'])
    if panel.month.max() > period.end:
        raise PermissionError('development_months_only')
    all_months = sorted(panel.month.unique())
    td = pd.read_parquet(root / periods.table_path(config, 'trading_days', final=include_test))
    last_days = pd.to_datetime(td.set_index('month').last_trading_day)
    p = panel.set_index(['stock_id', 'month']).sort_index()
    feats = {}

    # 교수님 CSV: STR, 장기 반전(t-35~t-12 24개월 모두 필요), ln 주가
    ret = _wide(p['return'], all_months)
    feats['str'] = p['return']
    feats['ltr'] = _long(np.expm1(np.log1p(ret).rolling(24, min_periods=24).sum().shift(12)), 'ltr')
    feats['ln_price'] = np.log(p.price.where(p.price.gt(0)))

    # marcap 일별
    daily, kospi = load_marcap(root, cfg, set(panel.stock_id), last_year=period.end_year)
    mkt = market_daily_return(kospi)
    del kospi
    dm = daily_monthly(daily, mkt, int(cfg['daily_rules']['ivol_min_days']))
    dmi = dm.set_index(['stock_id', 'month'])
    feats['ln_dvol'] = np.log(dmi.amt.where(dmi.amt.gt(0)))
    feats['illiq'] = dmi.illiq
    feats['max_ret'] = dmi.max_ret
    feats['ivol'] = dmi.ivol
    sh = p.shares_outstanding.where(p.shares_outstanding.gt(0)).reindex(dmi.index)
    feats['turnover'] = _long(_wide(dmi.vol / sh, all_months).rolling(3, min_periods=3).mean(), 'turnover')
    hi = _wide(dmi.hi_m, all_months).rolling(12, min_periods=12).max()     # 12개월 모두 거래가 있어야 함
    li = _wide(dmi.last_idx.where(dmi.trade_days.gt(0)), all_months)
    feats['high52'] = _long(li / hi, 'high52')
    feats['beta'] = weekly_beta(daily, mkt, months)
    del daily

    # 배당수익률(C-4): 월 배당수익 = (1+배당 포함 수익률)/(1+주가수익률) − 1, × 전월 말 시가총액, 최근 12개월 합 / t월 말 시가총액
    tot = total_return_monthly(root, cfg, include_test=include_test).set_index(['stock_id', 'month']).r_tot
    div_ret = (1 + tot.reindex(dmi.index)) / (1 + dmi.r_px) - 1
    capw = _wide(p.market_cap.where(p.market_cap.gt(0)), all_months)
    amt12 = (_wide(div_ret, all_months) * capw.shift(1)).rolling(12, min_periods=12).sum()
    feats['dy'] = _long(amt12 / capw, 'dy')

    # 재무(ROA, E/P)
    stmts = statements if statements is not None else annual_statements(root, include_test=include_test)
    fin, fin_record = financial_flags(root, cfg, include_test=include_test)
    acc = accounting_asof(stmts, annual_flows(root, cfg, include_test=include_test), months, last_days, fin).set_index(['stock_id', 'base_month'])
    feats['roa'] = acc.ni_owner / acc.assets_prev.where(acc.assets_prev.gt(0))
    cap_at = p.market_cap.where(p.market_cap.gt(0)).reindex(pd.MultiIndex.from_tuples(acc.index, names=['stock_id', 'month']))
    feats['ep'] = pd.Series(acc.ni_owner.to_numpy() / cap_at.to_numpy(), index=acc.index)
    # RMW: 영업이익 / B/M과 같은 자기자본(t월 as-of). 자본 0 이하(자본잠식)는 결측(팀원 결정, Ken French OP 포함 조건)
    equity = equity_asof(root, config, final=include_test)
    eq = equity.reindex(acc.index)
    feats['rmw'] = acc.op_income / eq.where(eq.gt(0))
    feats.update(industry_features(acc, cap_at))

    out = panel[panel.month.between(first, last)][['stock_id', 'month', 'eligible']].rename(columns={'month': 'base_month'})
    keys = pd.MultiIndex.from_frame(out[['stock_id', 'base_month']])
    out['financial'] = acc.financial.astype('boolean').reindex(keys).to_numpy()   # 공개된 사업보고서가 없는 행은 결측
    for name in STOCK_FEATURES:
        s = feats[name]
        s.index = s.index.set_names(['stock_id', 'base_month'])
        out[name] = s.replace([np.inf, -np.inf], np.nan).reindex(keys).to_numpy()

    # 시장 공통
    lvl = kospi_monthly(root, cfg)
    r_m = lvl / lvl.shift(1) - 1
    m = pd.DataFrame({'mkt_ret': r_m, 'mkt_ret3': lvl / lvl.shift(3) - 1, 'mkt_vol12': r_m.rolling(12, min_periods=12).std(ddof=1)}).reindex(months)
    e = panel[panel.eligible.eq(True) & panel['return'].notna()]
    m['breadth'] = e['return'].gt(0).groupby(e.month).mean().reindex(months)
    out = out.merge(m.rename_axis('base_month').reset_index(), on='base_month', how='left')

    out = add_reasons(out, dm, acc, equity)
    record = {'rows': int(len(out)), 'base_months': [first, last], 'stocks': int(out.stock_id.nunique()),
              'checks': checks(out, dm, mkt, lvl, ret, acc, stmts), 'industry': fin_record,
              'inputs_sha256': input_hashes(root, config, include_test=include_test)}
    return out.sort_values(['stock_id', 'base_month']).reset_index(drop=True), record


def add_reasons(out: pd.DataFrame, dm: pd.DataFrame, acc: pd.DataFrame, equity: pd.Series | None = None) -> pd.DataFrame:
    """값이 없을 때 이유(간단 분류). 연도별 가용률은 base_analysis에서 본다."""
    keys = pd.MultiIndex.from_frame(out[['stock_id', 'base_month']])
    eq = (equity.reindex(keys) if equity is not None else pd.Series(np.nan, index=keys)).to_numpy()
    td = dm.set_index(['stock_id', 'month']).trade_days
    td.index = td.index.set_names(['stock_id', 'base_month'])
    td = td.reindex(keys).to_numpy()
    accr = acc.acc_reason.reindex(keys).to_numpy()
    has = {c: acc[c].reindex(keys).notna().to_numpy() for c in set(NUMERATOR.values())}
    finv = acc.financial.astype('boolean').reindex(keys).fillna(False).to_numpy(dtype=bool)
    seen = keys.isin(acc.index)
    for name in FEATURES:
        r = np.full(len(out), None, dtype=object)
        miss = out[name].isna().to_numpy()
        if name in DAILY_FEATURES:
            r[miss & pd.isna(td)] = 'no_marcap_rows'
            r[miss & (td == 0)] = 'no_trading_day'
        elif name in ACCOUNTING_FEATURES:
            r[miss & (accr == 'expired_18m')] = 'expired_18m'
            r[miss & ~seen] = 'no_public_annual_statement'
            if name in FINANCIAL_EXCLUDED:
                r[miss & seen & finv & (accr != 'expired_18m')] = 'financial_industry'
            value = has[NUMERATOR[name]]
            r[miss & seen & pd.isna(r) & ~value & (accr != 'expired_18m')] = 'value_missing'
            if name == 'rmw':   # 분자가 있는데 빠졌으면 분모(B/M 자기자본) 이유
                r[miss & pd.isna(r) & np.isnan(eq)] = 'no_equity_asof'
                r[miss & pd.isna(r) & (eq <= 0)] = 'nonpositive_equity'
        r[miss & pd.isna(r)] = 'window_or_input_missing'
        out[name + '_reason'] = r
    return out


def checks(out, dm, mkt, lvl, ret, acc, stmts) -> dict:
    """종료 조건 T2·T3·T4 검사 값."""
    c = {}
    el = out[out.eligible.eq(True)]
    have = pd.MultiIndex.from_frame(dm[['stock_id', 'month']])
    h = pd.MultiIndex.from_frame(el[['stock_id', 'base_month']]).isin(have)
    c['T2_marcap_coverage_eligible_by_year'] = pd.Series(h).groupby(el.base_month.str[:4].to_numpy()).mean().round(4).to_dict()
    prof = _long(ret, 'r_prof').reset_index()
    j = dm.merge(prof, on=['stock_id', 'month'])
    j = j[j.r_prof.notna() & j.trade_days.gt(0)].merge(el[['stock_id', 'base_month']].rename(columns={'base_month': 'month'}), on=['stock_id', 'month'])
    hit = (j.r_px - j.r_prof).abs() <= 0.005
    c['T2_monthly_return_match_0.5pp_by_year'] = hit.groupby(j.month.str[:4]).mean().round(4).to_dict()
    c['T2_samsung_split_month'] = j[(j.stock_id == '005930') & j.month.isin(['2018-04', '2018-05'])][['month', 'r_px', 'r_prof']].round(6).to_dict('records')
    mm = np.expm1(np.log1p(mkt).groupby(mkt.index.strftime('%Y-%m')).sum())
    both = pd.concat([mm.rename('ours'), (lvl / lvl.shift(1) - 1).rename('kospi')], axis=1).dropna()
    diff = (both.ours - both.kospi).abs()
    c['T3_market_vs_kospi'] = {'months': int(len(both)), 'corr': round(float(both.ours.corr(both.kospi)), 5),
                               'median_abs_diff': round(float(diff.median()), 5), 'max_abs_diff': round(float(diff.max()), 5)}
    c['T4_annual_statements'] = {'rows': int(len(stmts)), 'by_side': stmts.side.value_counts().to_dict(),
                                 'by_std': stmts['std'].value_counts().to_dict()}
    c['T4_ni_owner_present_share_of_asof_rows'] = round(float(acc.ni_owner.notna().mean()), 4)
    c['nonnull_share_eligible'] = {n: round(float(el[n].notna().mean()), 4) for n in FEATURES}
    return c


def input_hashes(root: Path, config: dict, *, include_test: bool = False) -> dict:
    cfg = config['BASE_FEATURES']['inputs']
    from src import periods
    files = [periods.table_path(config, k, final=include_test).as_posix() for k in ('panel', 'trading_days', 'asof')]
    files += [kospi_index_file(root, config["BASE_FEATURES"]).relative_to(root).as_posix()]
    roles = ['total_return', 'industry_static', 'industry_hist', *[f'flows_{k}' for k in cfg['flows_xlsx']]]
    files += [p.relative_to(root).as_posix() for r in roles for p in dataguide_files.paths(root, r, include_test=include_test)]
    return {f: sha256(root / f) for f in files}


def ensure_base(root: Path, config: dict, *, execute: bool = False, statements: pd.DataFrame | None = None, log=print,
                include_test: bool = False) -> dict:
    root = Path(root)
    from src import periods
    out, record = build_base(root, config, statements=statements, include_test=include_test)
    output = periods.table_path(config, 'base_features', final=include_test)
    target = root / output
    if execute:
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)
        record['status'] = 'WRITTEN'
        record['output_sha256'] = sha256(target)
    else:
        record['status'] = 'DRY_RUN_NOT_WRITTEN'
    record['output'] = output.as_posix()
    log(f"base features {record['status']}: {record['rows']:,}행, 종목 {record['stocks']:,}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description='기본 피처(data/processed/base_features.parquet)')
    ap.add_argument('--config', default='config/experiment.yaml')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--final', action='store_true', help='전체 기간(src.periods) → DATA_PERIOD.final_dir/base_features.parquet')
    a = ap.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / a.config).read_text(encoding='utf-8'))
    print(json.dumps(ensure_base(root, config, execute=a.execute, include_test=a.final), ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
