# 자료 정리부터 공시 결합, 특징 생성, 내부 검증까지 처리 순서를 연결한다.
# YAML 설정을 읽고 각 단계를 선택하는 진입점과 최종평가 접근 차단을 제공한다.
# 예상 산출물: panel/asof/features/samples.parquet 및 내부 검증 predictions.csv·monthly_ic.csv.

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import yaml
from src.contracts import Config

# observe는 ingest 표와 collect(as-of)를 읽는 가용성 점검 단계라 그 둘 뒤, features 앞에 둔다.
STAGES = ("ingest", "collect", "observe", "features", "h1-test", "h1-pca-variance", "h1-pca-structure", "h1-pca-ic", "h1-constrained-pca", "h1-hypothesis", "h1-mean-tests", "h3-test", "h3-hypothesis", "h3-overlap", "validate", "final-test")


def load_config(config_path: Path) -> Config:
    """YAML 설정 경로를 받아 실험 설정 매핑을 반환한다.

    최상위 값은 매핑이어야 한다. 읽기 오류와 YAML 문법 오류는 호출자에게 전달한다.
    TODO: 열 대응, 단위, 특징 정의, 기간과 후보의 의미적 정합성을 검증한다.
    H2/H4를 활성화한 경우 각 절의 필수 명세가 null이면 오류로 처리한다.
    비활성 확장 명세는 기준모형 실행의 필수 조건으로 요구하지 않는다.
    """
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("YAML 설정의 최상위 값은 매핑이어야 합니다.")
    return config


INGEST_RUNS = Path("outputs/run_id/INGEST")


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_ingest(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """개발 원자료(2000-01~2021-12)를 정리해 interim 표 3개를 만든다.

    - panel: 제공 wide CSV 12개를 종목×월로 펼친 WIDE_PANEL_COLUMNS. 원본 읽기 계약(eligible 비어 있음)을
      검사한 뒤, eligible만 아래 적격성 판정 값으로 채운다. 다른 값은 바꾸지 않는다.
    - eligibility: ELIGIBILITY_COLUMNS. KRX KOSPI 상장·폐지 목록 + marcap 월말 KOSPI 거래(이전 상장 기간).
    - trading_days: TRADING_DAY_COLUMNS. marcap 거래 기록의 월별 마지막 거래일.
    execute=True면 outputs/run_id/INGEST/<시각>/에 표와 run.json을 남기고 최신본을 PATHS 고정 경로에 복사한다.
    내장 검사(키 유일, 행 수, 적격 근거 일관성, 모든 달의 거래일)가 하나라도 실패하면 저장하지 않는다.
    """
    import json
    import shutil
    import sys
    from datetime import datetime

    import pandas as pd

    from src.contracts import ELIGIBILITY_COLUMNS, TRADING_DAY_COLUMNS, WIDE_PANEL_COLUMNS
    from src.data_io import load_panel_contract
    from src.preprocessing.equity_dating import last_trading_days
    from src.preprocessing.listing import (
        DELISTING_DIR, LISTING_DIR, kospi_month_end_observations, listing_intervals, monthly_eligibility, read_events)

    started = datetime.now()
    panel = load_panel_contract(root, column_mapping=config["COLUMN_MAPPING"]["panel"], config=config)
    stocks = list(panel.attrs["stock_columns"])
    months = sorted(panel.month.unique())
    years = range(int(months[0][:4]), int(months[-1][:4]) + 1)
    marcap_files = [Path(f"data/raw/marcap/marcap-{y}.parquet") for y in years]  # 개발 기간 파일만 연다
    marcap = pd.concat(pd.read_parquet(root / p, columns=["Code", "Market", "Date"]) for p in marcap_files)
    ltd = {str(k): pd.Timestamp(v) for k, v in last_trading_days(marcap.Date).items()}
    trading = pd.DataFrame({"month": months, "last_trading_day": [ltd[m].strftime("%Y-%m-%d") if m in ltd else None
                                                                    for m in months]})[list(TRADING_DAY_COLUMNS)]

    listing, delisting, event_info = read_events(root)
    kospi_codes = set(marcap.loc[marcap.Market.eq("KOSPI"), "Code"].astype(str))
    intervals = listing_intervals(listing, delisting, stocks, kospi_codes)
    observed = kospi_month_end_observations(marcap, ltd)
    elig = monthly_eligibility(intervals, months, ltd, observed).rename(columns={"base_month": "month"})
    elig = elig[list(ELIGIBILITY_COLUMNS)]
    out = panel.drop(columns="eligible").merge(elig[["stock_id", "month", "eligible"]], on=["stock_id", "month"],
                                               how="left", validate="one_to_one")[list(WIDE_PANEL_COLUMNS)]

    state = elig.eligibility_basis.str.split(":").str[1].fillna("")
    true_states = {"listed", "prior_listing_kospi_observed", ""}
    checks = {
        "panel_rows": len(out) == len(stocks) * len(months),
        "eligibility_rows": len(elig) == len(out),
        "unique_keys": not out.duplicated(["stock_id", "month"]).any() and not elig.duplicated(["stock_id", "month"]).any(),
        "eligible_true_only_when_listed": bool(state[elig.eligible.eq(True).fillna(False)].isin(true_states).all()),
        "unknown_only_without_events": bool(elig.eligibility_basis[elig.eligible.isna()].eq("no_event_not_kospi_observed").all()),
        "every_month_has_trading_day": bool(trading.last_trading_day.notna().all()),
        "panel_eligible_matches_table": bool(
            out[["stock_id", "month", "eligible"]].merge(elig[["stock_id", "month", "eligible"]], on=["stock_id", "month"])
            .pipe(lambda m: (m.eligible_x.astype("object").fillna("NA") == m.eligible_y.astype("object").fillna("NA")).all())),
    }
    mc = out.market_cap.gt(0)
    summary = {
        "months": [months[0], months[-1], len(months)], "stocks": len(stocks), "rows": len(out),
        "eligible": {"true": int(elig.eligible.eq(True).sum()), "false": int(elig.eligible.eq(False).sum()),
                     "unknown": int(elig.eligible.isna().sum())},
        "prior_listing_kospi_observed_cells": int(state.eq("prior_listing_kospi_observed").sum()),
        "market_cap_cells": int(mc.sum()), "market_cap_ineligible": int((mc & out.eligible.eq(False)).sum()),
        "market_cap_unknown": int((mc & out.eligible.isna()).sum()),
        "basis_counts": elig.eligibility_basis.value_counts().to_dict(),
    }
    result = {"checks": checks, "summary": summary, "events": event_info, "execute": execute}
    if not all(checks.values()):
        raise ValueError("ingest_checks_failed:" + ",".join(k for k, v in checks.items() if not v))
    if not execute:
        return result

    stamp = started.strftime("%Y%m%dT%H%M%S")
    run_dir = root / INGEST_RUNS / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    tables = {"panel": out, "eligibility": elig, "trading_days": trading}
    for name, table in tables.items():
        table.to_parquet(run_dir / f"{name}.parquet", index=False)
    inputs = [Path(p) for files in config["PATHS"]["wide_csv"].values() for p in files.values()]
    inputs += marcap_files + sorted(d / p.name for d in (LISTING_DIR, DELISTING_DIR) for p in (root / d).glob("*.csv"))
    code = [Path("src/pipeline.py"), Path("src/preprocessing/listing.py"), Path("src/data_io.py"), Path("src/contracts.py")]
    run = {**result, "started": started.isoformat(timespec="seconds"), "command": " ".join(sys.argv),
           "input_sha256": {p.as_posix(): _sha256(root / p) for p in inputs},
           "code_sha256": {p.as_posix(): _sha256(root / p) for p in code},
           "outputs": {name: {"run_copy": (INGEST_RUNS / stamp / f"{name}.parquet").as_posix(),
                              "latest": config["PATHS"][name], "sha256": _sha256(run_dir / f"{name}.parquet")}
                       for name in tables}}
    (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for name in tables:
        target = root / config["PATHS"][name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / f"{name}.parquet", target)
    result["run_dir"] = run_dir.as_posix()
    return result


def check_ingest_current(config: Config, *, root: Path = Path(".")) -> dict:
    """현재 공통 입력의 계약을 검사한다. 과거 실행 기록은 읽지 않는다."""
    import pandas as pd
    from src.contracts import WIDE_PANEL_COLUMNS, ELIGIBILITY_COLUMNS, EQUITY_ASOF_COLUMNS
    from src.preprocessing.panel import enforce_unique_stock_month, normalize_month, normalize_stock_id
    tables={}
    for name,columns in (("panel",WIDE_PANEL_COLUMNS),("eligibility",ELIGIBILITY_COLUMNS),("asof",EQUITY_ASOF_COLUMNS)):
        path=root/config['PATHS'][name]
        if not path.is_file():raise FileNotFoundError('required_data_missing:'+str(path))
        tables[name]=pd.read_parquet(path)
        if not set(columns).issubset(tables[name]):raise ValueError('input_columns:'+name)
    panel,elig,asof=tables['panel'],tables['eligibility'],tables['asof']
    for name,f in tables.items():
        month='month' if name!='asof' else 'base_month'
        keyed=f.rename(columns={'stock_code':'stock_id'})
        enforce_unique_stock_month(keyed,month_column=month)
        normalize_stock_id(keyed.stock_id.tolist());normalize_month(keyed[month].tolist())
        last='2021-12' if name!='asof' else '2021-11'
        if not keyed[month].between('2000-01',last).all():raise PermissionError('development_input_only:'+name)
    pd.testing.assert_frame_equal(panel[['stock_id','month','eligible']].sort_values(['stock_id','month']).reset_index(drop=True),
        elig[['stock_id','month','eligible']].sort_values(['stock_id','month']).reset_index(drop=True),check_dtype=False)
    return {'data_inputs_checked':True,'input_sha256':{config['PATHS'][n]:_sha256(root/config['PATHS'][n]) for n in tables}}


def plan_observe(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """H1·H3 가용성 관측. 현재 공통 입력의 열·키·기간·적격성을 먼저 검사한다.

    execute=False면 선행 조건만 확인한다. execute=True면 outputs/데이터관측/{H1,H3}/<시각>/에 표·그림을 남긴다.
    """
    result = {"preconditions": check_ingest_current(config, root=root), "execute": execute}
    if execute:
        from src.features.observe import main as observe_main
        result.update({k: str(v) for k, v in observe_main(root, config).items()})
    return result


def plan_collect(config: Config) -> None:
    """공시와 활성화된 정책·뉴스 자료를 확보하고 공개시점 기준으로 정리한다.

    TODO: 원문·추출문 보존 -> 종목 연결 -> 공시 계정 선택 -> PIT 검사.
    중복 수집을 방지하고 결합 결과는 data/interim/asof.parquet에 저장한다.
    H2.enabled이면 한국은행 발표문·공개시점·수집/추출 상태를 보존하고
    POLICY_TEXT_COLUMNS 형식으로 정리한다. H2.lexicon/context_rules를 표현 집계에 전달한다. 공개시점은
    H2.publication_date_rule에 따라 실제 공개일을 우선하고 proxy 사용 여부를 보존한다.
    기업규모는 월간 패널의 직전 월말 시가총액을 H2.sensitivity_spec에 따라 준비한다.
    H4.enabled이면 H4.search_spec의 동일 검색조건으로 뉴스 API를 우선 확인한다.
    부족 구간은 이용조건을 확인한 웹 수집으로 보완하고 NEWS_ARTICLE_COLUMNS와
    월별 coverage를 구성한다. 요청 기간·검색 위치 제한과 실제 누락범위를 기록한다.
    확장을 비활성화하면 해당 자료 수집을 요구하지 않는다.
    수집기와 분석기를 분리하고 저장된 원문·추출문을 재사용한다.
    재실행 시 이미 확보된 원문을 중복 수집하지 않고 실패·재수집 범위를 구분한다.
    """
    raise NotImplementedError


FEATURES_H1_RUNS = Path("outputs/run_id/FEATURES_H1")


def check_observe_current(config: Config, *, root: Path = Path(".")) -> dict:
    """호환 진입점: 관측 출력 대신 현재 공통 입력을 검사한다."""
    return check_ingest_current(config,root=root)


def plan_features(config: Config, *, root: Path = Path("."), execute: bool = False, verify_existing: bool = False) -> dict:
    """H1 표 3개와 H3 표(data/processed/h3.parquet)를 계산·검사·저장. samples·목표 결합은 TODO.

    H3 계약 검산(validate_h3_contract)을 전체 자료에 돌려 약 6~7분 걸린다. 실행 기록 폴더 이름(FEATURES_H1)은 유지한다.
    """
    import json
    import shutil
    import sys
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from src.contracts import H1_COMPONENT_COLUMNS, MOM_COLUMNS, H1_COLUMNS
    from src.features.h1 import calculate_momentum_11m, calculate_size_bm, calculate_h1_score, validate_h1_contract

    if execute and verify_existing:
        raise ValueError("verification_must_not_overwrite_existing_tables")
    started = datetime.now()
    pre = check_observe_current(config, root=root)
    expected = {"momentum_months": 11, "momentum_start_offset": -11, "momentum_end_offset": -1,
                "size_transform": "natural_log", "ddof": 1, "min_stocks": 2,
                "coefficients": {"size": -1, "bm": 1, "mom": 1}, "divisor": 3}
    if any(config["H1"].get(k) != v for k, v in expected.items()):
        raise ValueError("unapproved_H1_config")
    if any(config["UNITS"].get(k) != v for k, v in {"return": "decimal", "market_cap": "KRW", "equity": "KRW"}.items()):
        raise ValueError("H1_units_mismatch")
    panel = pd.read_parquet(root / config["PATHS"]["panel"])
    asof = pd.read_parquet(root / config["PATHS"]["asof"])
    # parquet에는 단위 attrs가 없어 승인된 설정의 단위 계약을 전달한다.
    panel.attrs["units"] = config["UNITS"]
    asof.attrs["units"] = config["UNITS"]
    base = panel.loc[panel.month.le("2021-11")].copy()
    keys = base[["stock_id", "month"]].rename(columns={"month": "base_month"})
    components = calculate_size_bm(base, asof).merge(calculate_momentum_11m(panel, keys),
                      on=["stock_id", "base_month"], how="left", validate="one_to_one")
    components = components[list(H1_COMPONENT_COLUMNS)].sort_values(["stock_id", "base_month"]).reset_index(drop=True)
    scores = calculate_h1_score(components)
    validate_h1_contract(scores, components, asof)
    mom = components[list(MOM_COLUMNS)].copy()
    months = pd.period_range("2000-01", "2021-11", freq="M").astype(str)
    stocks = sorted(panel.stock_id.unique())
    expected_keys = pd.MultiIndex.from_product([stocks, months], names=["stock_id", "base_month"])
    grid_ok = pd.MultiIndex.from_frame(components[["stock_id", "base_month"]]).equals(expected_keys)
    # 독립 검산: 달력월로 재배열한 뒤 명시적인 11개월 곱(rolling 미사용).
    r = panel.pivot(index="stock_id", columns="month", values="return")
    m = components.pivot(index="stock_id", columns="base_month", values="mom").reindex(index=r.index)
    window_ok = True
    for month in months:
        t = pd.Period(month, freq="M")
        window = [str(t + offset) for offset in range(-11, 0)]
        raw = r.reindex(columns=window).to_numpy(float)
        valid = np.isfinite(raw).all(axis=1) & (raw >= -1).all(axis=1)
        ref = np.full(len(raw), np.nan)
        ref[valid] = np.prod(1 + raw[valid], axis=1) - 1
        window_ok &= bool(np.allclose(m[month], ref, equal_nan=True, rtol=1e-12, atol=1e-12))
    member = components.eligible.eq(True).fillna(False) & components[["size_available", "bm_available", "mom_available"]].all(axis=1)
    actual = member.groupby(components.base_month).sum().loc["2000-12":"2021-11"]
    monthly=actual.rename('actual_joint').to_frame()
    monthly['target_month']=(pd.PeriodIndex(monthly.index,freq='M')+1).astype(str)
    monthly['h1_n']=scores.groupby('base_month').h1_available.sum().reindex(monthly.index)
    monthly['actual_h1_month_valid']=monthly.h1_n.ge(2)
    checks = {"full_grid_1086_x_263": grid_ok and len(stocks) == 1086,
              "contract_independent_monthly_recalculation": True,
              "momentum_independent_calendar_product": window_ok,
              "first_target_momentum_and_h1_unavailable": bool(components.loc[components.base_month.eq("2000-12"), "mom"].isna().all()
                   and scores.loc[scores.base_month.eq("2000-12"), "h1"].isna().all()),
              "common_members_match_scores": bool(monthly.actual_joint.eq(monthly.h1_n).all()),
              "eligible_copied": components.eligible.equals(scores.eligible) and components.eligible.equals(mom.eligible),
              "no_target_columns": all("y" not in t and "target_month" not in t for t in (components, mom, scores))}
    summary = {"stocks": len(stocks), "base_months": [months[0], months[-1], len(months)], "rows_per_table": len(components),
               "analysis_common_cells": int(actual.sum()), "h1_valid_months": int(monthly.actual_h1_month_valid.sum()),
               "h1_available_cells": int(scores.h1_available.sum())}
    # H3(2026-11-08 사용자 결정: KOSPI만 보는 4개월 공백 초기화, 원값 유지). 목표 결합 없음.
    from src.contracts import H3_COLUMNS
    from src.features.h3 import compute_h3_from_panel, h3_output_table, validate_h3_contract
    expected_h3 = {"lambda": 0.97, "init_n_valid": 10, "init_ddof": 1, "recent_window": 12, "recent_min_valid": 10, "ddof": 1,
                   "includes_base_month": True, "reinitialize_every_12_months": False}
    if any(config["H3"].get(k) != v for k, v in expected_h3.items()):
        raise ValueError("unapproved_H3_config")
    elig_table = pd.read_parquet(root / config["PATHS"]["eligibility"])
    h3_features, h3_states, h3_history, h3_returns = compute_h3_from_panel(panel, elig_table)
    validate_h3_contract(h3_features, h3_states, h3_returns, listing_history=h3_history)
    h3 = h3_output_table(h3_features, h3_states)
    h3_member = h3.eligible.eq(True).fillna(False) & h3_features.sort_values(["stock_id", "base_month"]).vol_available.to_numpy()
    h3_monthly = h3.groupby("base_month").agg(h3_n=("h3_available", "sum"))
    h3_monthly["member_n"] = h3_member.groupby(h3.base_month).sum()
    checks.update({
        "h3_full_grid_1086_x_263": pd.MultiIndex.from_frame(h3[["stock_id", "base_month"]]).equals(expected_keys),
        "h3_contract_independent_validation": True,
        "h3_columns": list(h3.columns) == list(H3_COLUMNS),
        "h3_count_equals_eligible_with_volatility": bool(h3_monthly.h3_n.eq(h3_monthly.member_n).all()),
        "h3_first_target_month_available": bool(h3.loc[h3.base_month.eq("2000-12"), "h3_available"].sum() >= 2),
        "h3_eligible_copied": h3.eligible.equals(components.eligible),
        "h3_no_target_columns": "y" not in h3 and "target_month" not in h3})
    target_months = h3_monthly.loc["2000-12":"2021-11"]
    summary.update({"h3_available_cells_2000_12_2021_11": int(target_months.h3_n.sum()),
                    "h3_valid_months": int(target_months.h3_n.ge(2).sum()),
                    "h3_resets": int(h3_states.reset.sum())})
    result = {"execute": execute, "preconditions": pre, "checks": checks, "summary": summary}
    if not all(checks.values()):
        raise ValueError("features_checks_failed:" + ",".join(k for k,v in checks.items() if not v))
    if verify_existing:
        for name, table in {"h1_components": components, "mom": mom, "h1": scores, "h3": h3}.items():
            pd.testing.assert_frame_equal(table, pd.read_parquet(root / config["PATHS"][name]), check_exact=True)
        result["checks"]["existing_feature_tables_exact_equal"] = True
    if not execute:
        return result
    stamp = started.strftime("%Y%m%dT%H%M%S%f")
    run_dir = root / FEATURES_H1_RUNS / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    tables = {"h1_components": components, "mom": mom, "h1": scores, "h3": h3}
    for name, table in tables.items():
        table.to_parquet(run_dir / f"{name}.parquet", index=False)
    monthly.to_csv(run_dir / "availability_reconciliation.csv", encoding="utf-8-sig")
    h3_states.loc[h3_states.reset, ["stock_id", "base_month", "reset_basis"]].to_csv(run_dir / "h3_resets.csv", index=False, encoding="utf-8-sig")
    code = ["src/features/h1.py", "src/features/h3.py", "src/preprocessing/listing.py", "src/pipeline.py", "src/contracts.py",
            "src/preprocessing/panel.py"]
    input_paths = [Path(config["PATHS"][n]) for n in ("panel", "eligibility", "asof")]
    run = {**result, "status": "EXECUTED", "started": started.isoformat(), "command": " ".join(sys.argv),
           "python": sys.version, "approved_decisions": "2026-11-05 user instruction (H1 stage 1); 2026-11-08 H3 KOSPI-only 4-month reset, raw H3 kept; no target join",
           "h1_config": dict(config["H1"]), "h3_config": dict(config["H3"]), "units": dict(config["UNITS"]),
           "input_sha256": {p.as_posix(): _sha256(root / p) for p in input_paths},
           "code_sha256": {p: _sha256(root / p) for p in code},
           "outputs": {n: {"run_copy": (FEATURES_H1_RUNS / stamp / f"{n}.parquet").as_posix(),
                              "latest": config["PATHS"][n], "sha256": _sha256(run_dir / f"{n}.parquet")} for n in tables}}
    (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for name in tables:
        target = root / config["PATHS"][name]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / f"{name}.parquet", target)
    return {**result, "run_dir": run_dir.as_posix()}


def check_features_current(config: Config, *, root: Path = Path("."), revalidate: bool = False) -> dict:
    """고정 경로의 특징·입력 자체를 검사한다. 실행 이력은 선택 조건이 아니다."""
    import pandas as pd
    from src.features.h1 import validate_h1_contract
    pre=check_ingest_current(config,root=root)
    tables={}
    for name in ('h1_components','mom','h1'):
        path=root/config['PATHS'][name]
        if not path.is_file():raise FileNotFoundError('required_data_missing:'+str(path))
        tables[name]=pd.read_parquet(path)
    asof=pd.read_parquet(root/config['PATHS']['asof'])
    validate_h1_contract(tables['h1'],tables['h1_components'],asof)
    from src.contracts import MOM_COLUMNS
    pd.testing.assert_frame_equal(tables['h1_components'][list(MOM_COLUMNS)],tables['mom'])
    pre['input_sha256'].update({config['PATHS'][n]:_sha256(root/config['PATHS'][n]) for n in tables})
    return {**pre,'feature_contract_checked':True}


def plan_h1_test(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """2단계 진단만 실행. execute 없으면 선행 자료와 승인 설정만 확인한다."""
    pre = check_features_current(config, root=root)
    settings = config.get("H1_TEST", {})
    if settings.get("approved_stage") != 2 or settings.get("representative_year") != 2021:
        raise PermissionError("stage2_and_representative_2021_approval_required")
    if not execute:
        return {"execute":False, "preconditions":pre, "approved_decisions":dict(settings)}
    from src.evaluation.h1_analysis import run_analysis
    return run_analysis(config, root=root)


def plan_h1_pca_variance(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """승인된 원값/제한값 설명분산만 계산. 기존 특징은 재저장 없이 검산한다."""
    settings=config.get("H1_PCA", {})
    if settings.get("approved_scope") not in ("variance_only","structure_diagnostics") or settings.get("variants") != ["raw", "winsor"]:
        raise PermissionError("pca_variance_only_both_variants_approval_required")
    if settings.get("run_category") != "h1_pca반영지수":
        raise ValueError("user_pca_run_category_required")
    pre=check_features_current(config,root=root,revalidate=True)
    if not execute:return {"execute":False,"preconditions":pre,"approved_decisions":dict(settings)}
    from src.evaluation.h1_analysis import run_pca_variance
    return run_pca_variance(config,root=root,preconditions=pre)


def plan_h1_pca_structure(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """성분별 가중치·특징 상관·월별 안정성 진단. 성분 선택·수익률 평가 없음."""
    settings=config.get("H1_PCA", {})
    if settings.get("approved_scope")!="structure_diagnostics" or settings.get("variants")!=["raw","winsor"]:
        raise PermissionError("pca_structure_diagnostics_approval_required")
    if settings.get("run_category")!="h1_pca반영지수":raise ValueError("user_pca_run_category_required")
    if settings.get("display_sign_rule")!="first_largest_abs_positive_then_previous_same_rank":
        raise ValueError("diagnostic_display_sign_rule_required")
    pre=check_features_current(config,root=root,revalidate=True)
    if not execute:return {"execute":False,"preconditions":pre,"approved_decisions":dict(settings)}
    from src.evaluation.h1_analysis import run_pca_structure
    return run_pca_structure(config,root=root,preconditions=pre)


def plan_h1_pca_ic(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """승인된 원값 PC1의 월별 순위 상관과 단순평균만 비교한다."""
    from src.evaluation.h1_analysis import check_pc1_ic_approval, run_pca_pc1_ic
    check_pc1_ic_approval(config)
    pre=check_features_current(config,root=root,revalidate=True)
    if not execute:return {"execute":False,"preconditions":pre}
    return run_pca_pc1_ic(config,root=root,preconditions=pre)


def plan_h1_constrained_pca(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    from src.evaluation.h1_analysis import check_constrained_approval, run_constrained_pca
    check_constrained_approval(config)
    pre=check_features_current(config,root=root,revalidate=True)
    if not execute:return {"execute":False,"preconditions":pre}
    return run_constrained_pca(config,root=root,preconditions=pre)


def plan_h1_hypothesis(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    from src.evaluation.h1_analysis import check_hypothesis_inputs, run_hypothesis
    pre=check_hypothesis_inputs(config,root=root)
    if not execute:return {"execute":False,"preconditions":pre}
    return run_hypothesis(config,root=root,preconditions=pre)


def plan_h1_mean_tests(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    from src.evaluation.h1_analysis import check_mean_extension_inputs, run_mean_extension
    pre=check_mean_extension_inputs(config,root=root)
    if not execute:return {"execute":False,"preconditions":pre}
    return run_mean_extension(config,root=root,preconditions=pre)


def plan_h3_test(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """H3 분포·원값 Rank IC·1·99 분위수 제한 비교. data/interim panel·eligibility만 읽고 H3를 메모리에서 계산한다."""
    from src.evaluation.h3_analysis import check_h3_inputs, run_analysis
    settings = config.get("H3_TEST") or {}
    if not settings.get("approved") or settings.get("representative_year") != 2021 or settings.get("winsorization") != [0.01, 0.99]:
        raise PermissionError("h3_test_approval_required")
    pre = check_h3_inputs(config, root=root)
    if not execute:
        return {"execute": False, "preconditions": pre, "approved_decisions": dict(settings)}
    return run_analysis(config, root=root, preconditions=pre)


def plan_h3_hypothesis(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """H3 평균 IC=0 HAC 검정(원값 주검정, 제한값, 대응 차이). 승인 설정이 있어야 실행한다."""
    from src.evaluation.h3_analysis import check_h3_inputs, run_hypothesis
    if not (config.get("H3_HYPOTHESIS") or {}).get("approved"):
        raise PermissionError("h3_hypothesis_approval_required")
    pre = check_h3_inputs(config, root=root)
    if not execute:
        return {"execute": False, "preconditions": pre}
    return run_hypothesis(config, root=root, preconditions=pre)


def plan_h3_overlap(config: Config, *, root: Path = Path("."), execute: bool = False) -> dict:
    """추가 분석: 공유 특징표 h1·h3의 월별 상관과 같은 표본 IC. data/processed 표만 읽는다."""
    from src.evaluation.h3_analysis import check_h3_inputs, run_overlap
    if not (config.get("H3_TEST") or {}).get("approved"):
        raise PermissionError("h3_test_approval_required")
    pre = check_h3_inputs(config, root=root)
    for name in ("h1", "h3"):
        if not (root / config["PATHS"][name]).is_file():
            raise FileNotFoundError("required_data_missing:" + config["PATHS"][name])
    if not execute:
        return {"execute": False, "preconditions": pre}
    return run_overlap(config, root=root, preconditions=pre)


def plan_validate(config: Config) -> None:
    """공통 표본과 연별 Fold로 세 모형의 후보를 비교하고 설정을 선택한다.

    TODO: 학습행 전처리 적합 -> 후보별 적합·예측 -> 실제값 1:1 결합 ->
    월별 IC -> 공통 유효월 평균 -> 모형별 동률 규칙 적용.
    수렴 실패와 평가 불가 사유, 예측값 및 월별 IC를 기록한다.
    활성 확장마다 H1/H3와 해당 주효과·상호작용을 입력으로 구성한다.
    H2는 H2_VALUE_COLUMNS, H4는 mom과 H4_VALUE_COLUMNS를 보존한다.
    기준·확장모형 모두 같은 가용 종목·목표월로 다시 학습·평가한다.
    외부자료 학습 이력이 5년 미만이면 해당 확장 검증을 보류한다.
    """
    raise NotImplementedError


def final_test(config: Config) -> None:
    """동결한 설정으로 최종평가를 수행하는 진입점.

    TODO: Train 내부 설정 확정 -> 2001~2021 전체 학습행으로 모형·입력 변환기
    재적합 -> Test 최초 열람·적재 -> 고정 절차로 한 번 예측·평가·보고.
    Warmup은 특징·시차·상태 초기화에만 사용하고 모형 학습·평가에서 제외한다.
    모형·변환기는 models/, 최종 예측·월별 IC·요약표는 outputs/final_test/에 저장한다.
    Test 데이터 접근을 차단한다.
    """
    raise PermissionError("Test 데이터 접근이 차단되었습니다.")


def run_stage(stage: str, *, config: Config) -> None:
    """단계명과 실험 설정을 받아 해당 처리 단계를 선택한다.

    TODO: 단계명과 선행 품질 검사 결과를 확인하고 입력·출력을 연결한다.
    오류가 발생하면 후속 처리를 중단하고 영향 범위를 기록한다.
    """
    if stage == "final-test":
        raise PermissionError("Test 데이터 접근이 차단되었습니다.")
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: --config(기본 config/experiment.yaml) --stage <단계> [--execute].

    현재 구현된 단계는 ingest·observe·features(H1만)·h1-test(2단계)다. --execute 없이 실행하면 ingest는 계산·검사만, observe는 선행 조건만 확인하고
    저장하지 않는다. features는 공통 data 입력으로 계산·검사한다. collect(as-of)는 equity_dating·equity_asof
    CLI로 따로 만든다. h1-test는 execute 없으면 선행 조건·승인 설정만 검사한다. 다른 단계는 run_stage에서 그대로 NotImplementedError/PermissionError로 멈춘다.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m src.pipeline")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--execute", action="store_true", help="표와 실행 기록을 저장")
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    runners = {"ingest": plan_ingest, "observe": plan_observe, "features": plan_features, "h1-test": plan_h1_test,
               "h1-pca-variance": plan_h1_pca_variance, "h1-pca-structure": plan_h1_pca_structure, "h1-pca-ic": plan_h1_pca_ic,
               "h1-constrained-pca": plan_h1_constrained_pca, "h1-hypothesis": plan_h1_hypothesis,
               "h1-mean-tests": plan_h1_mean_tests, "h3-test": plan_h3_test, "h3-hypothesis": plan_h3_hypothesis,
               "h3-overlap": plan_h3_overlap}
    if args.stage in runners:
        runner = runners[args.stage]
        result = runner(config, execute=args.execute)
        print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
        return
    run_stage(args.stage, config=config)


if __name__ == "__main__":
    main()
