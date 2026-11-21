"""기본(통제) 피처: 종목×기준월 표 하나(data/processed/base_features.parquet).

정의·근거·결정은 config BASE_FEATURES(2026-11-19~21 사용자 결정, 노션 「12주차 실행」 2-2)를 따른다.
중간 집계(marcap 월 집계, 시장 일수익률, 연간 재무)는 파일로 남기지 않고 메모리에서 계산한다.
입력은 data/ 파일과 config뿐이며 이전 실행 기록을 읽지 않는다. 테스트 기간(2022~) 파일은 열지 않는다.

열: stock_id, base_month, eligible, 피처 18개(종목 14 + 시장 공통 4), 피처별 <이름>_reason(값이 없을 때 이유).
시장 공통 4개(mkt_ret, mkt_ret3, mkt_vol12, breadth)는 같은 달 모든 종목이 같은 값이다.

실행: python -m src.features.base             (계산·검사만, 쓰지 않음)
      python -m src.features.base --execute   (base_features.parquet 쓰기)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.preprocessing.dataguide_extract import iter_rows, read_sheet, sha256

STOCK_FEATURES = ('str', 'ltr', 'high52', 'turnover', 'ln_dvol', 'illiq', 'ivol', 'max_ret', 'beta', 'ln_price', 'roa', 'ep', 'dy', 'rmw')
MARKET_FEATURES = ('mkt_ret', 'mkt_ret3', 'mkt_vol12', 'breadth')
FEATURES = STOCK_FEATURES + MARKET_FEATURES
DAILY_FEATURES = ('ln_dvol', 'illiq', 'ivol', 'max_ret')
ACCOUNTING_FEATURES = ('roa', 'ep', 'rmw')
FLOW_ITEMS = ('ni_owner', 'op_income')
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

def load_marcap(root: Path, cfg: dict, stock_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """marcap 2000~2021. (패널 종목의 일별 행, 시장수익률용 KOSPI 전 종목 행). 종목코드 6자리, 일수익률은 소수."""
    keep, kospi = [], []
    for f in sorted(glob.glob(str(root / cfg['inputs']['marcap_dir'] / 'marcap-*.parquet'))):
        if int(Path(f).stem.split('-')[1]) > 2021:
            continue
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

def kospi_monthly(root: Path, cfg: dict) -> pd.Series:
    path = root / cfg['inputs']['kospi_index_csv']
    guard_development(path)
    k = pd.read_csv(path, encoding='utf-8-sig')
    row = k[k.iloc[:, 0].astype(str).str.contains('연월말')].iloc[0]
    level = pd.to_numeric(row.iloc[2:], errors='coerce').astype(float)
    level.index = [c.replace('.', '-') for c in level.index]
    return level


def total_return_monthly(root: Path, cfg: dict) -> pd.DataFrame:
    """DataGuide 수정주가수익률(현금배당포함), 종목×월, 소수."""
    rows = []
    for p in cfg['inputs']['total_return_xlsx']:
        path = root / p
        guard_development(path)
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

def annual_statements(root: Path) -> pd.DataFrame:
    """결정 ①③④: H1 자기자본과 같은 공시일·재무제표 선택(equity_dating.build)의 사업보고서(연간) 행."""
    from src.preprocessing.equity_dating import build
    panel, _, _, _ = build(root)   # 약 8~9분(2026-11-21 측정 506초)
    return select_annual(panel)


def select_annual(panel: pd.DataFrame) -> pd.DataFrame:
    if 'panel' in panel:
        panel = panel[panel.panel]
    # H1 자기자본이 쓰지 않는 행(공시일 없음, 연결이 보고서에 없어 값 없음, 항등식 검사 실패 제외)은 재무 피처에서도 쓰지 않는다
    a = panel[panel.kind.eq('사업') & panel.rcept_no.notna() & panel.available_month.notna()
              & panel.tier.isin(CONSOLIDATED_TIERS + ('T3_separate_total',)) & panel.exclude_reason.isna()].copy()
    a['side'] = np.where(a.tier.isin(CONSOLIDATED_TIERS), 'cons', 'sep')
    return a[['code', 'stock_code', 'fy', 'month', 'std', 'tier', 'side', 'available_month', 'rcept_dt', 'cons_assets', 'sep_assets']]


def annual_flows(root: Path, cfg: dict) -> pd.DataFrame:
    """(code, fy, std, side) → 연간 지배주주 순이익, 영업이익(분기 누적의 결산 행)."""
    parts = []
    for key, spec in cfg['inputs']['flows_xlsx'].items():
        std, side = key.split('_')
        path = root / EXTERNAL / spec['file']
        guard_development(path)
        f, _ = read_sheet(path, {str(k): v for k, v in spec['items'].items()})
        f = f[f.quarter.eq('결산')]
        parts.append(f[['code', 'fy', *FLOW_ITEMS]].assign(std=std, side=side))
    return pd.concat(parts, ignore_index=True).drop_duplicates(['code', 'fy', 'std', 'side'])


def accounting_asof(stmts: pd.DataFrame, flows: pd.DataFrame, months: list[str], last_days: pd.Series) -> pd.DataFrame:
    """종목×기준월: 기준월까지 공개된 가장 최근 사업보고서의 지배순이익·영업이익, 같은 재무제표(연결/개별)의 전년 말 총자산."""
    if 'op_income' not in flows:
        flows = flows.assign(op_income=np.nan)
    s = stmts.merge(flows, on=['code', 'fy', 'std', 'side'], how='left')
    prev = s[['stock_code', 'month', 'cons_assets', 'sep_assets']].drop_duplicates(['stock_code', 'month'])
    prev = prev.assign(month=(pd.PeriodIndex(prev.month, freq='M') + 12).astype(str))
    s = s.merge(prev.rename(columns={'cons_assets': 'cons_prev', 'sep_assets': 'sep_prev'}), on=['stock_code', 'month'], how='left')
    s['assets_prev'] = np.where(s.side.eq('cons'), s.cons_prev, s.sep_prev)
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
                rows.append((code, t, np.nan, np.nan, np.nan, best['month'], best['side'], 'expired_18m'))
            else:
                rows.append((code, t, best['ni_owner'], best['op_income'], best['assets_prev'], best['month'], best['side'], None))
    return pd.DataFrame(rows, columns=['stock_id', 'base_month', 'ni_owner', 'op_income', 'assets_prev', 'statement_month', 'side',
                                       'acc_reason'])


def equity_asof(root: Path, config: dict) -> pd.Series:
    """B/M과 같은 자기자본(H1 as-of의 usable 값), (stock_id, base_month) 색인.
    as-of는 0 이하(자본잠식)를 usable=False, reason=nonpositive_equity로 두므로 그 값은 남겨 이유 표시에 쓰고, RMW에서는 결측."""
    a = pd.read_parquet(root / config['PATHS']['asof'], columns=['stock_code', 'base_month', 'equity', 'usable', 'reason'])
    a = a[a.usable.eq(True) | a.reason.eq('nonpositive_equity')]
    return a.set_index(['stock_code', 'base_month']).equity.rename_axis(['stock_id', 'base_month']).astype(float)


# ---------------------------------------------------------------- 조립

def build_base(root: Path, config: dict, *, statements: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = config['BASE_FEATURES']
    root = Path(root)
    first, last = cfg['base_months']
    months = pd.period_range(first, last, freq='M').astype(str).tolist()
    panel = pd.read_parquet(root / config['PATHS']['panel'],
                            columns=['stock_id', 'month', 'return', 'market_cap', 'price', 'shares_outstanding', 'eligible'])
    if panel.month.max() > '2021-12':
        raise PermissionError('development_months_only')
    all_months = sorted(panel.month.unique())
    td = pd.read_parquet(root / config['PATHS']['trading_days'])
    last_days = pd.to_datetime(td.set_index('month').last_trading_day)
    p = panel.set_index(['stock_id', 'month']).sort_index()
    feats = {}

    # 교수님 CSV: STR, 장기 반전(t-35~t-12 24개월 모두 필요), ln 주가
    ret = _wide(p['return'], all_months)
    feats['str'] = p['return']
    feats['ltr'] = _long(np.expm1(np.log1p(ret).rolling(24, min_periods=24).sum().shift(12)), 'ltr')
    feats['ln_price'] = np.log(p.price.where(p.price.gt(0)))

    # marcap 일별
    daily, kospi = load_marcap(root, cfg, set(panel.stock_id))
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
    tot = total_return_monthly(root, cfg).set_index(['stock_id', 'month']).r_tot
    div_ret = (1 + tot.reindex(dmi.index)) / (1 + dmi.r_px) - 1
    capw = _wide(p.market_cap.where(p.market_cap.gt(0)), all_months)
    amt12 = (_wide(div_ret, all_months) * capw.shift(1)).rolling(12, min_periods=12).sum()
    feats['dy'] = _long(amt12 / capw, 'dy')

    # 재무(ROA, E/P)
    stmts = statements if statements is not None else annual_statements(root)
    acc = accounting_asof(stmts, annual_flows(root, cfg), months, last_days).set_index(['stock_id', 'base_month'])
    feats['roa'] = acc.ni_owner / acc.assets_prev.where(acc.assets_prev.gt(0))
    cap_at = p.market_cap.where(p.market_cap.gt(0)).reindex(pd.MultiIndex.from_tuples(acc.index, names=['stock_id', 'month']))
    feats['ep'] = pd.Series(acc.ni_owner.to_numpy() / cap_at.to_numpy(), index=acc.index)
    # RMW: 영업이익 / B/M과 같은 자기자본(t월 as-of). 자본 0 이하(자본잠식)는 결측(팀원 결정, Ken French OP 포함 조건)
    equity = equity_asof(root, config)
    eq = equity.reindex(acc.index)
    feats['rmw'] = acc.op_income / eq.where(eq.gt(0))

    out = panel[panel.month.between(first, last)][['stock_id', 'month', 'eligible']].rename(columns={'month': 'base_month'})
    keys = pd.MultiIndex.from_frame(out[['stock_id', 'base_month']])
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
              'checks': checks(out, dm, mkt, lvl, ret, acc, stmts), 'inputs_sha256': input_hashes(root, config)}
    return out.sort_values(['stock_id', 'base_month']).reset_index(drop=True), record


def add_reasons(out: pd.DataFrame, dm: pd.DataFrame, acc: pd.DataFrame, equity: pd.Series | None = None) -> pd.DataFrame:
    """값이 없을 때 이유(간단 분류). 연도별 가용률은 base_analysis에서 본다."""
    keys = pd.MultiIndex.from_frame(out[['stock_id', 'base_month']])
    eq = (equity.reindex(keys) if equity is not None else pd.Series(np.nan, index=keys)).to_numpy()
    td = dm.set_index(['stock_id', 'month']).trade_days
    td.index = td.index.set_names(['stock_id', 'base_month'])
    td = td.reindex(keys).to_numpy()
    accr = acc.acc_reason.reindex(keys).to_numpy()
    has = {'ni_owner': acc.ni_owner.reindex(keys).notna().to_numpy(),
           'op_income': acc.op_income.reindex(keys).notna().to_numpy()}
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
            value = has['op_income' if name == 'rmw' else 'ni_owner']
            r[miss & seen & ~value & (accr != 'expired_18m')] = 'value_missing'
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


def input_hashes(root: Path, config: dict) -> dict:
    cfg = config['BASE_FEATURES']['inputs']
    files = [config['PATHS']['panel'], config['PATHS']['trading_days'], config['PATHS']['asof'], cfg['kospi_index_csv'],
             *cfg['total_return_xlsx'],
             *[(EXTERNAL / v['file']).as_posix() for v in cfg['flows_xlsx'].values()]]
    return {f: sha256(root / f) for f in files}


def ensure_base(root: Path, config: dict, *, execute: bool = False, statements: pd.DataFrame | None = None, log=print) -> dict:
    root = Path(root)
    out, record = build_base(root, config, statements=statements)
    target = root / config['PATHS']['base_features']
    if execute:
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)
        record['status'] = 'WRITTEN'
        record['output_sha256'] = sha256(target)
    else:
        record['status'] = 'DRY_RUN_NOT_WRITTEN'
    record['output'] = config['PATHS']['base_features']
    log(f"base features {record['status']}: {record['rows']:,}행, 종목 {record['stocks']:,}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description='기본 피처(data/processed/base_features.parquet)')
    ap.add_argument('--config', default='config/experiment.yaml')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args(argv)
    root = Path.cwd()
    config = yaml.safe_load((root / a.config).read_text(encoding='utf-8'))
    print(json.dumps(ensure_base(root, config, execute=a.execute), ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
