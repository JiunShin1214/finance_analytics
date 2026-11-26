"""DataGuide 엑셀 찾기·점검(2026-11-26): 파일 이름이 아니라 파일 안의 항목 코드로 자료 종류를 정한다.

DataGuide 자료는 사람이 화면에서 항목을 골라 받는다. 저장 이름은 사람마다 다르므로(예: '2009_2021_연결.xlsx',
'test_2022_2025__연결.xlsx', 교수님의 2026년 파일) 이름을 보지 않는다. 엑셀 첫 줄들(머리)에 DataGuide 항목 코드가
들어 있으므로 그 코드 묶음으로 종류를 정하고, 같은 종류의 여러 파일(기간별로 나눠 받은 것)은 함께 쓴다.

종류와 필요한 항목 코드는 코드·설정에 이미 있는 것을 그대로 쓴다(가이드 문서의 '받을 항목'과 같은 목록).
- 재무상태표 5종: dataguide_extract.SOURCES (K-GAAP 개별·연결, K-GAAP 결산월, K-IFRS 연결·개별)
- 손익·현금흐름 4종: config BASE_FEATURES.inputs.flows_xlsx
- 업종 2종: config BASE_FEATURES.inputs.industry (KSIC 대분류, 시장 대표 업종)
- 수정주가수익률: S410007900

점검(멈춤 = 오류, 계산을 시작하지 않음)
  1 어떤 자료인지 모르는 파일            -> 건너뜀(목록만)
  2 필요한 항목 코드가 일부만 있는 파일  -> 멈춤(빠진 항목)
  3 필요한 종류의 파일이 없음            -> 멈춤(받을 항목 코드)
  4 회계연도가 모자람(--need-fy)         -> 멈춤
  5 같은 회사·회계연도·분기 값이 파일끼리 다름 -> 멈춤(1000배 차이면 단위 옵션 안내)
  6 같은 파일 안에 같은 회사·회계연도·분기가 두 번 -> 멈춤
  4~6은 파일 내용을 다 읽어야 해서(파일당 약 10초) --values 일 때만 한다.

개발 기간(기본)에는 test_sealed 폴더를 열지 않는다. --include-test 는 모델을 고정한 뒤 테스트 입력을 만들 때만 쓴다.

실행: python -m src.preprocessing.dataguide_files                  (종류 분류·빠진 항목만, 빠름)
      python -m src.preprocessing.dataguide_files --values --need-fy 2025 --include-test
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.preprocessing.dataguide_extract import SOURCES, iter_rows, read_sheet

EXTERNAL = Path("data/raw/external_data")
SEALED = "test_sealed"
HEAD_ROWS = 20
TOTAL_RETURN_ITEM = "S410007900"
CODE = re.compile(r"\d{10}|[A-Z]{1,2}\d{7,9}")

LABEL = {   # 자료 종류 이름(사람이 읽는 말)
    "kgaap_separate": "K-GAAP 개별 재무상태표", "kgaap_consolidated": "K-GAAP 연결 재무상태표",
    "kgaap_fiscal_month": "K-GAAP 결산월", "kifrs_consolidated": "K-IFRS 연결 재무상태표",
    "kifrs_separate": "K-IFRS 개별 재무상태표",
    "flows_KGAAP_cons": "K-GAAP 연결 손익·현금흐름", "flows_KGAAP_sep": "K-GAAP 개별 손익·현금흐름",
    "flows_KIFRS_cons": "K-IFRS 연결 손익·현금흐름", "flows_KIFRS_sep": "K-IFRS 개별 손익·현금흐름",
    "industry_static": "업종(KSIC 11차, 현재 분류)", "industry_hist": "시장 대표 업종(결산 연도별)",
    "total_return": "수정주가수익률(현금배당 포함, 월간)",
}
CONCEPT = {   # 항목 이름
    "assets": "자산총계", "liabilities": "부채총계", "sep": "자본총계(개별)", "cons": "자본총계(연결)",
    "owners": "지배주주지분", "nci": "비지배주주지분", "fm": "결산월", "ni_owner": "(지배주주)순이익",
    "op_income": "영업이익", "sales": "매출액", "cogs": "매출원가", "cfo": "영업활동현금흐름",
    "ksic": "한국표준산업분류 11차 대분류", "hist": "시장 대표 업종", "tr": "수정주가수익률",
}
KIFRS_ROLES = ("kifrs_consolidated", "kifrs_separate", "flows_KIFRS_cons", "flows_KIFRS_sep")
FINANCIAL = {"kgaap_separate", "kgaap_consolidated", "kifrs_consolidated", "kifrs_separate",
             "flows_KGAAP_cons", "flows_KGAAP_sep", "flows_KIFRS_cons", "flows_KIFRS_sep"}


def roles(config: dict) -> dict[str, dict[str, str]]:
    """종류 -> {항목 코드: 개념}."""
    out = {role: dict(items) for role, (_, _, items) in SOURCES.items()}
    inputs = config["BASE_FEATURES"]["inputs"]
    for key, spec in inputs["flows_xlsx"].items():
        out[f"flows_{key}"] = {str(k): v for k, v in spec["items"].items()}
    ind = inputs["industry"]
    out["industry_static"] = {ind["ksic_section_item"]: "ksic"}
    out["industry_hist"] = {ind["hist_item"]: "hist"}
    out["total_return"] = {TOTAL_RETURN_ITEM: "tr"}
    return out


def head_codes(path: Path) -> set[str]:
    """머리 줄들에 있는 DataGuide 항목 코드."""
    codes = set()
    for n, cells in iter_rows(path):
        codes |= {str(v) for v in cells.values() if CODE.fullmatch(str(v))}
        if n >= HEAD_ROWS:
            break
    return codes


def classify(codes: set[str], table: dict[str, dict[str, str]]) -> tuple[str | None, list[str]]:
    """가장 많은 필요 코드를 가진 종류(같으면 필요 코드 중 갖춘 비율이 높은 종류). 반환: (종류, 빠진 코드).
    예: 결산월 코드 하나만 있는 파일은 'K-GAAP 결산월'(1/1)이지 'K-IFRS 연결'(1/6)이 아니다."""
    best = max(table, key=lambda r: (len(codes & set(table[r])), len(codes & set(table[r])) / len(table[r])))
    hit = codes & set(table[best])
    if not hit:
        return None, []
    return best, sorted(set(table[best]) - codes)


def scan(root: Path, config: dict, *, include_test: bool = False) -> dict:
    """external_data 아래 엑셀 전부(하위 폴더 포함) -> 종류별 파일과 문제."""
    table = roles(config)
    files, unknown, problems = {r: [] for r in table}, [], []
    for path in sorted((root / EXTERNAL).rglob("*.xlsx")):
        rel = path.relative_to(root)
        if path.name.startswith("~$") or (SEALED in rel.parts and not include_test):
            continue
        try:
            codes = head_codes(path)
        except Exception as error:          # 엑셀이 아니거나 깨진 파일
            problems.append({"kind": "unreadable", "file": rel.as_posix(), "detail": type(error).__name__})
            continue
        role, missing = classify(codes, table)
        if role is None:
            unknown.append(rel.as_posix())
        elif missing:
            problems.append({"kind": "missing_items", "file": rel.as_posix(), "role": role,
                             "missing": [(c, table[role][c]) for c in missing]})
        else:
            files[role].append(rel)
    for role, items in table.items():
        if not files[role] and not any(p.get("role") == role for p in problems):
            problems.append({"kind": "missing_role", "role": role, "missing": sorted(items.items())})
    return {"files": files, "unknown": unknown, "problems": problems, "roles": table}


_CACHE: dict = {}


def paths(root: Path, role: str, *, include_test: bool = False) -> list[Path]:
    """계산 코드가 쓰는 입구: 그 종류의 파일들(root 기준 절대 경로, 이름 순). 빠진 항목·없는 종류가 있으면 멈춘다.
    한 실행 안에서는 분류를 한 번만 한다(파일 머리만 읽어 약 2초)."""
    key = (Path(root).resolve(), include_test)
    if key not in _CACHE:
        config = yaml.safe_load((Path(root) / "config/experiment.yaml").read_text(encoding="utf-8"))
        scanned = scan(Path(root), config, include_test=include_test)
        if scanned["problems"]:
            raise ValueError("dataguide_files_problems:\n" + report(scanned, scanned["problems"]))
        _CACHE[key] = scanned["files"]
    return [Path(root) / p for p in _CACHE[key][role]]


def read_role(root: Path, role: str, items: dict[str, str], *, text: tuple[str, ...] = ("fm",),
              include_test: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """그 종류의 파일을 모두 read_sheet로 읽어 합친다. 파일이 하나면 read_sheet 결과 그대로(행 순서까지).
    여럿이면 (code, fy, quarter)가 겹치는 행의 값이 같아야 하고(차이 1 이하, 한쪽 빈칸은 다른 쪽으로 채움) 다르면 멈춘다."""
    from src.preprocessing.dataguide_extract import sha256
    frames, info = [], []
    for p in paths(root, role, include_test=include_test):
        f, i = read_sheet(p, items, text=text)
        frames.append(f)
        info.append({"file": p.relative_to(root).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size, **i})
    if len(frames) == 1:
        return frames[0], info
    key = ["code", "fy", "quarter"]
    both = pd.concat(frames, ignore_index=True)
    for c in items.values():
        if c in text or c not in both:
            continue
        g = both[both[c].notna()].groupby(key)[c]
        spread = g.max() - g.min()
        if (spread > 1).any():
            raise ValueError(f"dataguide_overlap_conflict:{role}:{c}:{spread[spread > 1].index[:3].tolist()}")
    return both.groupby(key, as_index=False, sort=False).first(), info


def check_values(root: Path, scanned: dict, *, need_fy: int | None = None) -> list[dict]:
    """재무 자료: 파일 안 중복, 파일끼리 값 충돌(단위), 회계연도 범위."""
    problems, key = [], ["code", "fy", "quarter"]
    for role in sorted(FINANCIAL):
        paths = scanned["files"].get(role, [])
        if not paths:
            continue
        items = scanned["roles"][role]
        values = [c for c in items.values() if c != "fm"]
        frames = []
        for p in paths:
            f, _ = read_sheet(root / p, items)
            f = f[f.fy.astype(str).str.fullmatch(r"\d{4}")]
            dup = f.duplicated(key)
            if dup.any():
                problems.append({"kind": "duplicate_rows", "role": role, "file": p.as_posix(), "rows": int(dup.sum()),
                                 "example": f.loc[dup, key].head(3).to_dict("records")})
            frames.append(f.drop_duplicates(key).assign(_file=p.as_posix()))
        both = pd.concat(frames, ignore_index=True)
        years = both.fy.astype(int)
        if need_fy and role in KIFRS_ROLES and years.max() < need_fy:
            problems.append({"kind": "years_short", "role": role, "have": [int(years.min()), int(years.max())],
                             "need": need_fy, "files": [p.as_posix() for p in paths]})
        if len(frames) > 1:
            g = both[both.duplicated(key, keep=False)].sort_values(key + ["_file"])
            first, other = g.drop_duplicates(key, keep="first").set_index(key), g.drop_duplicates(key, keep="last").set_index(key)
            for c in values:
                a, b = first[c], other[c]
                ok = a.isna() & b.isna() | ((a - b).abs() <= 1)
                bad = ~ok & a.notna() & b.notna()
                if bad.any():
                    ratio = (b[bad] / a[bad]).abs()
                    unit = bool(((ratio - 1000).abs() < 1).mean() > 0.9 or ((ratio - 0.001).abs() < 1e-6).mean() > 0.9)
                    ex = pd.DataFrame({"값1": a[bad], "파일1": first.loc[bad, "_file"], "값2": b[bad],
                                       "파일2": other.loc[bad, "_file"]}).head(3).reset_index().to_dict("records")
                    problems.append({"kind": "unit_mismatch" if unit else "value_conflict", "role": role, "item": c,
                                     "rows": int(bad.sum()), "example": ex})
    return problems


def report(scanned: dict, problems: list[dict]) -> str:
    """사람이 읽는 점검 결과(그대로 Claude에 붙여 넣어도 무엇을 다시 받을지 알 수 있게)."""
    t = scanned["roles"]
    lines = ["[DataGuide 점검] " + EXTERNAL.as_posix(), "", "인식한 파일"]
    for role, paths in scanned["files"].items():
        if paths:
            lines.append(f"  {LABEL[role]}: " + ", ".join(p.name for p in paths))
    if scanned["unknown"]:
        lines += ["", "건너뛴 파일(어떤 DataGuide 자료인지 알 수 없음): " + ", ".join(scanned["unknown"])]
    if not problems:
        return "\n".join(lines + ["", "문제 없음"])
    lines += ["", f"문제 {len(problems)}건 - 계산을 시작하지 않습니다"]
    for p in problems:
        name = LABEL.get(p.get("role"), p.get("role"))
        items = lambda m: ", ".join(f"{c} {CONCEPT.get(k, k)}" for c, k in m)
        if p["kind"] == "missing_items":
            lines.append(f"  - {p['file']} ({name}로 보임): 항목이 빠졌습니다 -> {items(p['missing'])}")
        elif p["kind"] == "missing_role":
            lines.append(f"  - {name} 파일이 없습니다. 받을 항목: {items(p['missing'])}")
        elif p["kind"] == "years_short":
            lines.append(f"  - {name}: 회계연도가 {p['have'][0]}~{p['have'][1]}까지만 있습니다. {p['need']} 회계연도까지 필요합니다"
                         f" (받을 항목: {items(sorted(t[p['role']].items()))}).")
        elif p["kind"] == "duplicate_rows":
            lines.append(f"  - {p['file']} ({name}): 같은 회사·회계연도·분기 행이 {p['rows']}건 중복. 예: {p['example']}")
        elif p["kind"] == "unit_mismatch":
            lines.append(f"  - {name} {CONCEPT.get(p['item'], p['item'])}: 파일끼리 값이 약 1000배 다릅니다({p['rows']}건)."
                         f" 금액 단위(원/천원) 옵션을 확인하세요. 예: {p['example']}")
        elif p["kind"] == "value_conflict":
            lines.append(f"  - {name} {CONCEPT.get(p['item'], p['item'])}: 같은 회사·기간 값이 파일끼리 다릅니다({p['rows']}건). 예: {p['example']}")
        elif p["kind"] == "unreadable":
            lines.append(f"  - {p['file']}: 엑셀로 읽을 수 없습니다({p['detail']}).")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--values", action="store_true", help="파일 내용까지 읽어 중복·충돌·기간 점검(파일당 약 10초)")
    parser.add_argument("--need-fy", type=int, help="K-IFRS 자료에 있어야 하는 마지막 회계연도(예: 2025)")
    parser.add_argument("--include-test", action="store_true", help="test_sealed 폴더도 포함(모델 고정 뒤에만)")
    args = parser.parse_args(argv)
    config = yaml.safe_load((args.root / "config/experiment.yaml").read_text(encoding="utf-8"))
    scanned = scan(args.root, config, include_test=args.include_test)
    problems = list(scanned["problems"])
    if args.values:
        problems += check_values(args.root, scanned, need_fy=args.need_fy)
    print(report(scanned, problems))
    if problems:
        sys.exit(2)


if __name__ == "__main__":
    main()
