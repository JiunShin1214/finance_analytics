from pathlib import Path
from urllib.parse import urljoin
import re
import subprocess
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


from src.h2.data_io import ROOT as BASE_DIR

BASE_URL = "https://www.bok.or.kr"

YEAR_PAGE_URL = (
    "https://www.bok.or.kr/portal/singl/"
    "crncyPolicyDrcMtg/listYear.do"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120 Safari/537.36"
    )
}

from src.h2.data_io import office_executable
SOFFICE = office_executable()

# 일단 경계연도만 테스트
# YEARS = [2001, 2021]

# 경계연도 테스트가 성공하면 아래로 교체
YEARS = range(2001, 2022)

RAW_BASE = (
    BASE_DIR
    / "data"
    / "raw"
    / "bok"
)

CONVERTED_BASE = (
    BASE_DIR
    / "data"
    / "converted"
    / "bok"
)

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bok_metadata_full.csv"
)


def clean_text(text):
    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def get_year_page(session, year):

    params = {
        "menuNo": "200755",
        "mtgSe": "A",
        "pYear": year,
    }

    response = session.get(
        YEAR_PAGE_URL,
        params=params,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    return (
        BeautifulSoup(
            response.text,
            "html.parser"
        ),
        response.url
    )


def parse_meeting_date(
    row_text,
    year
):

    match = re.search(
        r"(\d{1,2})월\s*"
        r"(\d{1,2})일",
        row_text
    )

    if not match:
        return None

    month = int(
        match.group(1)
    )

    day = int(
        match.group(2)
    )

    return (
        f"{year:04d}-"
        f"{month:02d}-"
        f"{day:02d}"
    )


def find_policy_file(tr):
    """
    통화정책방향 결정문만 찾는다.

    우선순위:
    1. PDF
    2. HWP/HWPX

    기자간담회 / 의사록은 제외
    """

    candidates = []

    for a in tr.find_all(
        "a",
        href=True
    ):

        href = a.get(
            "href",
            ""
        )

        if "fileDown.do" not in href:
            continue

        text = clean_text(
            a.get_text(
                " ",
                strip=True
            )
        )

        title = clean_text(
            a.get(
                "title",
                ""
            )
        )

        file_name = (
            title
            or text
        )

        lower = (
            file_name.lower()
        )

        # 필요 없는 자료 제외
        exclude_words = [
            "기자간담회",
            "의사록",
        ]

        if any(
            word in file_name
            for word in exclude_words
        ):
            continue

        if lower.endswith(".pdf"):
            priority = 1
            file_type = "pdf"

        elif lower.endswith(".hwp"):
            priority = 2
            file_type = "hwp"

        elif lower.endswith(".hwpx"):
            priority = 3
            file_type = "hwpx"

        else:
            continue

        candidates.append({
            "priority": priority,
            "file_name": file_name,
            "href": href,
            "file_type": file_type,
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x:
        x["priority"]
    )

    return candidates[0]


def download_file(
    session,
    url,
    output_path
):

    response = session.get(
        url,
        headers=HEADERS,
        timeout=60,
        allow_redirects=True
    )

    response.raise_for_status()

    output_path.write_bytes(
        response.content
    )


def extract_pdf_text(
    path
):

    reader = PdfReader(
        path
    )

    texts = []

    for page in reader.pages:

        text = (
            page.extract_text()
            or ""
        )

        if text.strip():
            texts.append(
                text
            )

    return "\n".join(
        texts
    )


def read_txt(path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]

    for encoding in encodings:

        try:
            return path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    return ""


def convert_hwp_to_txt(
    input_path,
    output_dir
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not SOFFICE.exists():

        raise FileNotFoundError(
            f"LibreOffice 없음: {SOFFICE}"
        )

    output_path = (
        output_dir
        / (
            input_path.stem
            + ".txt"
        )
    )

    if output_path.exists():
        return output_path

    result = subprocess.run(
        [
            str(SOFFICE),
            "--headless",
            "--convert-to",
            "txt",
            "--outdir",
            str(output_dir),
            str(input_path),
        ],
        capture_output=True,
        text=True
    )

    if (
        result.returncode != 0
        or not output_path.exists()
    ):

        print(
            "[HWP 변환 실패]"
        )

        print(
            result.stdout
        )

        print(
            result.stderr
        )

        return None

    return output_path


def process_year(
    session,
    year
):

    print()
    print(
        "=" * 70
    )

    print(
        f"{year}년 수집 시작"
    )

    soup, source_url = (
        get_year_page(
            session,
            year
        )
    )

    raw_dir = (
        RAW_BASE
        / str(year)
    )

    converted_dir = (
        CONVERTED_BASE
        / str(year)
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    meeting_count = 0

    for table in soup.find_all(
        "table"
    ):

        for tr in table.find_all(
            "tr"
        ):

            row_text = clean_text(
                tr.get_text(
                    " ",
                    strip=True
                )
            )

            meeting_date = (
                parse_meeting_date(
                    row_text,
                    year
                )
            )

            if not meeting_date:
                continue

            meeting_count += 1

            policy_file = (
                find_policy_file(
                    tr
                )
            )

            if not policy_file:

                print(
                    "[파일 없음]",
                    meeting_date
                )

                records.append({
                    "meeting_date":
                        meeting_date,

                    "release_date":
                        meeting_date,

                    "release_date_source":
                        "meeting_date_proxy",

                    "year_month":
                        meeting_date[:7],

                    "title":
                        f"통화정책방향({meeting_date})",

                    "source_url":
                        source_url,

                    "attachment_url":
                        None,

                    "original_file_name":
                        None,

                    "original_format":
                        None,

                    "original_path":
                        None,

                    "processed_format":
                        None,

                    "processed_path":
                        None,

                    "text_length":
                        0,

                    "status":
                        "file_not_found",
                })

                continue

            date_tag = (
                meeting_date
                .replace("-", "_")
            )

            file_type = (
                policy_file[
                    "file_type"
                ]
            )

            file_name = (
                f"{date_tag}_policy."
                f"{file_type}"
            )

            raw_path = (
                raw_dir
                / file_name
            )

            attachment_url = (
                urljoin(
                    BASE_URL,
                    policy_file[
                        "href"
                    ]
                )
            )

            print(
                "[수집]",
                meeting_date,
                policy_file[
                    "file_name"
                ]
            )

            try:

                if not raw_path.exists():

                    download_file(
                        session,
                        attachment_url,
                        raw_path
                    )

                    time.sleep(0.25)

                processed_path = None
                processed_format = None
                text = ""

                # PDF
                if file_type == "pdf":

                    processed_path = (
                        raw_path
                    )

                    processed_format = (
                        "pdf"
                    )

                    text = (
                        extract_pdf_text(
                            raw_path
                        )
                    )

                # HWP / HWPX
                elif file_type in [
                    "hwp",
                    "hwpx",
                ]:

                    txt_path = (
                        convert_hwp_to_txt(
                            raw_path,
                            converted_dir
                        )
                    )

                    if txt_path:

                        processed_path = (
                            txt_path
                        )

                        processed_format = (
                            "txt"
                        )

                        text = (
                            read_txt(
                                txt_path
                            )
                        )

                status = (
                    "ok"
                    if len(text) > 0
                    else "text_empty"
                )

            except Exception as e:

                print(
                    "[처리 실패]",
                    meeting_date,
                    e
                )

                processed_path = None
                processed_format = None
                text = ""
                status = (
                    f"error:{type(e).__name__}"
                )

            def relative(path):

                if path is None:
                    return None

                return str(
                    path.relative_to(
                        BASE_DIR
                    )
                ).replace(
                    "\\",
                    "/"
                )

            records.append({

                "meeting_date":
                    meeting_date,

                # 아직 실제 게시시각을
                # 검증하지 않았으므로 proxy
                "release_date":
                    meeting_date,

                "release_date_source":
                    "meeting_date_proxy",

                "year_month":
                    meeting_date[:7],

                "title":
                    f"통화정책방향({meeting_date})",

                "source_url":
                    source_url,

                "attachment_url":
                    attachment_url,

                "original_file_name":
                    policy_file[
                        "file_name"
                    ],

                "original_format":
                    file_type,

                "original_path":
                    relative(
                        raw_path
                    ),

                "processed_format":
                    processed_format,

                "processed_path":
                    relative(
                        processed_path
                    ),

                "text_length":
                    len(text),

                "status":
                    status,
            })

    print(
        f"{year}년 회의 행:",
        meeting_count
    )

    return records


def main():

    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    session = requests.Session()

    all_records = []

    for year in YEARS:

        try:

            records = (
                process_year(
                    session,
                    year
                )
            )

            all_records.extend(
                records
            )

        except Exception as e:

            print(
                f"[{year}년 실패]",
                type(e).__name__,
                e
            )

    df = pd.DataFrame(
        all_records
    )

    if df.empty:

        print(
            "수집 결과가 없습니다."
        )

        return

    df = (
        df
        .sort_values(
            [
                "meeting_date",
                "original_file_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    df.to_csv(
        METADATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "===== 전체 수집 결과 ====="
    )

    print(
        df[
            [
                "meeting_date",
                "original_format",
                "processed_format",
                "text_length",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "총 회의 수:",
        len(df)
    )

    print(
        "정상 처리:",
        (df["status"] == "ok").sum()
    )

    print(
        "텍스트 비어 있음:",
        (
            df["status"]
            == "text_empty"
        ).sum()
    )

    print(
        "metadata:",
        METADATA_PATH
    )


if __name__ == "__main__":
    main()