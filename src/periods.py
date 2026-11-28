"""기간 끝·저장 위치를 한 곳에서 정한다(7-4 ③④, 2026-11-26 사용자 결정, config DATA_PERIOD).

- 개발(final=False, 기본): 자료 끝 = DATA_PERIOD.development_end(2021-12). 교수님 CSV는 개발 묶음(warmup·train)만,
  표는 PATHS 경로(data/interim·processed). 학습·검증·튜닝은 이 표만 읽는다(테스트 잠금은 그대로).
- 최종(final=True): 교수님 CSV 묶음 전체(테스트·2026 포함). 자료 끝 = 그 CSV들의 마지막 달(자료에서 자동,
  2026년분을 넣으면 코드 수정 없이 늘어남). 표는 DATA_PERIOD.final_dir 아래 PATHS와 같은 파일 이름.
피처 기준월의 끝은 자료 끝의 전달이다(목표월 = 기준월 + 1이 자료 안에 있어야 함).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.contracts import DEVELOPMENT_MONTHS

FIRST_MONTH = "2000-01"
MARCAP_DIR = Path("data/raw/marcap")


@dataclass(frozen=True)
class Period:
    final: bool
    end: str                    # 자료 마지막 달(YYYY-MM)
    groups: tuple[str, ...]     # 읽을 교수님 CSV 묶음(PATHS.wide_csv 키)

    @property
    def last_base_month(self) -> str:
        return str(pd.Period(self.end, "M") - 1)

    @property
    def end_year(self) -> int:
        return int(self.end[:4])

    @property
    def end_day(self) -> pd.Timestamp:
        return pd.Period(self.end, "M").end_time.normalize()

    def months(self, last: str | None = None) -> list[str]:
        return pd.period_range(FIRST_MONTH, last or self.end, freq="M").astype(str).tolist()


def _csv_months(path: Path) -> list[str]:
    """wide CSV의 첫 열(month)만 읽는다. 값 열은 읽지 않는다."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        next(rows)
        return [str(pd.Period(r[0], "M")) for r in rows if r]


def resolve(config: dict, root: Path = Path("."), *, final: bool = False) -> Period:
    if not final and ("DATA_PERIOD" not in config or "wide_csv" not in config.get("PATHS", {})):
        # 최소 설정(합성 자료 테스트 등): 개발 기간은 계약(DEVELOPMENT_MONTHS)으로 정해져 있다
        groups = tuple(DEVELOPMENT_MONTHS)
        return Period(False, DEVELOPMENT_MONTHS[groups[-1]][1], groups)
    manifest = config["PATHS"]["wide_csv"]
    if not final:
        groups = tuple(g for g in manifest if g in DEVELOPMENT_MONTHS)
        if groups != tuple(DEVELOPMENT_MONTHS):
            raise ValueError(f"development_groups:{groups}")
        end = str(config["DATA_PERIOD"]["development_end"])
        if end != DEVELOPMENT_MONTHS[groups[-1]][1]:
            raise ValueError("development_end_must_match_last_development_group")
        return Period(False, end, groups)
    groups = tuple(manifest)
    last = max(max(_csv_months(Path(root) / manifest[g]["adjusted_monthly_return"])) for g in groups)
    return Period(True, last, groups)


def table_path(config: dict, key: str, *, final: bool = False) -> Path:
    """PATHS[key](개발) 또는 final_dir/같은 파일 이름(최종). root 기준 상대 경로."""
    path = Path(config["PATHS"][key])
    return Path(config["DATA_PERIOD"]["final_dir"]) / path.name if final else path


def marcap_paths(root: Path, period: Period) -> list[Path]:
    """marcap-2000 ~ 자료 끝 연도 파일(상대 경로). 하나라도 없으면 멈춘다."""
    paths = [MARCAP_DIR / f"marcap-{y}.parquet" for y in range(int(FIRST_MONTH[:4]), period.end_year + 1)]
    missing = [p.as_posix() for p in paths if not (Path(root) / p).is_file()]
    if missing:
        raise FileNotFoundError(f"marcap_missing:{missing}")
    return paths


def panel_csv(config: dict, period: Period, variable: str) -> list[Path]:
    """읽을 묶음의 교수님 CSV 경로(변수 하나)."""
    return [Path(config["PATHS"]["wide_csv"][g][variable]) for g in period.groups]
