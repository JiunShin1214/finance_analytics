from pathlib import Path
from urllib.parse import urljoin
import re
import shutil
import subprocess
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document


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

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bok_metadata_full.csv"
)

BACKUP_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bok_metadata_full_before_recovery.csv"
)

RAW_BASE = (
    BASE_DIR
    / "data"
    / "raw"
    / "bok"
)

RECOVERED_BASE = (
    BASE_DIR
    / "data"
    / "recovered"
    / "bok"
)


def clean_text(text):
    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


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


def extract_pdf_text(path):
    try:
        reader = PdfReader(path)

        parts = []

        for page in reader.pages:
            text = page.extract_text() or ""

            if text.strip():
                parts.append(text)

        return "\n".join(parts)

    except Exception:
        return ""


def extract_docx_text(path):
    try:
        doc = Document(path)

        texts = [
            p.text.strip()
            for p in doc.paragraphs
            if p.text.strip()
        ]

        return "\n".join(texts)

    except Exception:
        return ""


def detect_hwp_version(path):
    data = path.read_bytes()[:32]

    if data.startswith(
        b"HWP Document File V3.00"
    ):
        return "hwp3"

    if data.startswith(
        bytes.fromhex(
            "D0 CF 11 E0 A1 B1 1A E1"
        )
    ):
        return "hwp5"

    return "unknown"


def libreoffice_convert(
    input_path,
    output_dir,
    target_format
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result = subprocess.run(
        [
            str(SOFFICE),
            "--headless",
            "--convert-to",
            target_format,
            "--outdir",
            str(output_dir),
            str(input_path),
        ],
        capture_output=True,
        text=True
    )

    return result


def try_hwp5txt(
    input_path,
    output_path
):
    """
    HWP5(OLE2)인 경우 hwp5txt 사용.
    """

    try:
        result = subprocess.run(
            [
                str(Path(sys.executable).with_name("hwp5txt.exe")),
                str(input_path),
            ],
            capture_output=True
        )

        if result.returncode != 0:
            return ""

        # stdout 인코딩 후보
        for encoding in [
            "utf-8",
            "cp949",
            "euc-kr",
        ]:
            try:
                text = result.stdout.decode(
                    encoding
                )

                if text.strip():
                    output_path.write_text(
                        text,
                        encoding="utf-8"
                    )

                    return text

            except UnicodeDecodeError:
                continue

    except FileNotFoundError:
        pass

    return ""


def recover_hwp(
    hwp_path,
    output_dir
):
    version = detect_hwp_version(
        hwp_path
    )

    print(
        "   HWP 형식:",
        version
    )

    # 1. HWP5라면 hwp5txt 우선
    if version == "hwp5":

        txt_path = (
            output_dir
            / f"{hwp_path.stem}_hwp5.txt"
        )

        text = try_hwp5txt(
            hwp_path,
            txt_path
        )

        if len(text.strip()) >= 100:

            return {
                "text": text,
                "path": txt_path,
                "format": "txt",
                "method": "hwp5txt",
            }

    # 2. LibreOffice → TXT
    libreoffice_convert(
        hwp_path,
        output_dir,
        "txt"
    )

    txt_path = (
        output_dir
        / f"{hwp_path.stem}.txt"
    )

    if txt_path.exists():

        text = read_txt(
            txt_path
        )

        if len(text.strip()) >= 100:

            return {
                "text": text,
                "path": txt_path,
                "format": "txt",
                "method": "libreoffice_txt",
            }

    # 3. LibreOffice → DOCX
    libreoffice_convert(
        hwp_path,
        output_dir,
        'docx:"Office Open XML Text"'
    )

    docx_path = (
        output_dir
        / f"{hwp_path.stem}.docx"
    )

    if docx_path.exists():

        text = extract_docx_text(
            docx_path
        )

        if len(text.strip()) >= 100:

            return {
                "text": text,
                "path": docx_path,
                "format": "docx",
                "method": "libreoffice_docx",
            }

    return None


def get_year_soup(
    session,
    year
):
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

    return BeautifulSoup(
        response.text,
        "html.parser"
    )


def find_meeting_row(
    soup,
    meeting_date
):
    date = pd.to_datetime(
        meeting_date
    )

    target_month = date.month
    target_day = date.day

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

            match = re.search(
                r"(\d{1,2})월\s*"
                r"(\d{1,2})일",
                row_text
            )

            if not match:
                continue

            month = int(
                match.group(1)
            )

            day = int(
                match.group(2)
            )

            if (
                month == target_month
                and day == target_day
            ):
                return tr

    return None


def get_policy_candidates(tr):
    """
    정책문 PDF/HWP 후보를 모두 가져옴.
    기자간담회/의사록 제외.
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

        file_name = (
            a.get("title")
            or a.get_text(
                " ",
                strip=True
            )
        ).strip()

        if (
            "기자간담회" in file_name
            or "의사록" in file_name
        ):
            continue

        lower = (
            file_name.lower()
        )

        if lower.endswith(".pdf"):
            file_type = "pdf"

        elif lower.endswith(".hwp"):
            file_type = "hwp"

        elif lower.endswith(".hwpx"):
            file_type = "hwpx"

        else:
            continue

        candidates.append({
            "file_name": file_name,
            "file_type": file_type,
            "url": urljoin(
                BASE_URL,
                href
            ),
        })

    return candidates


def download_candidate(
    session,
    candidate,
    output_dir
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    suffix = (
        "." + candidate["file_type"]
    )

    safe_name = (
        candidate["file_name"]
        .replace("/", "_")
        .replace("\\", "_")
    )

    if not safe_name.lower().endswith(
        suffix
    ):
        safe_name += suffix

    output_path = (
        output_dir
        / safe_name
    )

    response = session.get(
        candidate["url"],
        headers=HEADERS,
        timeout=60
    )

    response.raise_for_status()

    output_path.write_bytes(
        response.content
    )

    return output_path


def relative(path):
    return str(
        path.relative_to(
            BASE_DIR
        )
    ).replace(
        "\\",
        "/"
    )


def main():

    df = pd.read_csv(
        METADATA_PATH
    )

    # 원본 metadata 백업
    if not BACKUP_PATH.exists():
        shutil.copy2(
            METADATA_PATH,
            BACKUP_PATH
        )

    targets = df[
        (
            df["status"] != "ok"
        )
        |
        (
            df["text_length"]
            < 100
        )
    ].copy()

    print(
        "복구 대상:",
        len(targets),
        "건"
    )

    session = requests.Session()

    year_cache = {}

    for idx, row in targets.iterrows():

        meeting_date = (
            row["meeting_date"]
        )

        year = int(
            meeting_date[:4]
        )

        print()
        print("=" * 70)
        print(
            "[복구 대상]",
            meeting_date
        )

        if year not in year_cache:

            year_cache[year] = (
                get_year_soup(
                    session,
                    year
                )
            )

        soup = year_cache[year]

        tr = find_meeting_row(
            soup,
            meeting_date
        )

        if tr is None:

            print(
                " 회의 행을 찾지 못함"
            )

            continue

        candidates = (
            get_policy_candidates(
                tr
            )
        )

        print(
            " 후보:",
            [
                x["file_name"]
                for x in candidates
            ]
        )

        # 기존 형식이 PDF면 HWP를 먼저,
        # 기존 형식이 HWP면 PDF를 먼저 시도
        original_format = (
            str(
                row[
                    "original_format"
                ]
            )
            .lower()
        )

        if original_format == "pdf":

            candidates.sort(
                key=lambda x:
                0 if x["file_type"] in [
                    "hwp",
                    "hwpx"
                ] else 1
            )

        else:

            candidates.sort(
                key=lambda x:
                0 if x["file_type"]
                == "pdf" else 1
            )

        recovered = None

        recovery_dir = (
            RECOVERED_BASE
            / str(year)
            / meeting_date
        )

        for candidate in candidates:

            print(
                " 시도:",
                candidate[
                    "file_name"
                ]
            )

            try:

                downloaded = (
                    download_candidate(
                        session,
                        candidate,
                        recovery_dir
                    )
                )

                file_type = (
                    candidate[
                        "file_type"
                    ]
                )

                if file_type == "pdf":

                    text = (
                        extract_pdf_text(
                            downloaded
                        )
                    )

                    if len(
                        text.strip()
                    ) >= 100:

                        recovered = {
                            "text":
                                text,

                            "processed_path":
                                downloaded,

                            "processed_format":
                                "pdf",

                            "method":
                                "alternate_pdf",

                            "original_path":
                                downloaded,

                            "original_format":
                                "pdf",

                            "original_file_name":
                                candidate[
                                    "file_name"
                                ],
                        }

                        break

                elif file_type in [
                    "hwp",
                    "hwpx",
                ]:

                    result = recover_hwp(
                        downloaded,
                        recovery_dir
                    )

                    if result:

                        recovered = {
                            "text":
                                result[
                                    "text"
                                ],

                            "processed_path":
                                result[
                                    "path"
                                ],

                            "processed_format":
                                result[
                                    "format"
                                ],

                            "method":
                                result[
                                    "method"
                                ],

                            "original_path":
                                downloaded,

                            "original_format":
                                file_type,

                            "original_file_name":
                                candidate[
                                    "file_name"
                                ],
                        }

                        break

            except Exception as e:

                print(
                    "  실패:",
                    type(e).__name__,
                    e
                )

            time.sleep(0.2)

        if recovered:

            print(
                " 복구 성공!"
            )

            print(
                " 방법:",
                recovered[
                    "method"
                ]
            )

            print(
                " 글자 수:",
                len(
                    recovered[
                        "text"
                    ]
                )
            )

            df.loc[
                idx,
                "original_file_name"
            ] = recovered[
                "original_file_name"
            ]

            df.loc[
                idx,
                "original_format"
            ] = recovered[
                "original_format"
            ]

            df.loc[
                idx,
                "original_path"
            ] = relative(
                recovered[
                    "original_path"
                ]
            )

            df.loc[
                idx,
                "processed_format"
            ] = recovered[
                "processed_format"
            ]

            df.loc[
                idx,
                "processed_path"
            ] = relative(
                recovered[
                    "processed_path"
                ]
            )

            df.loc[
                idx,
                "text_length"
            ] = len(
                recovered[
                    "text"
                ]
            )

            df.loc[
                idx,
                "status"
            ] = "ok"

            df.loc[
                idx,
                "recovery_method"
            ] = recovered[
                "method"
            ]

        else:

            print(
                " 복구 실패"
            )

            df.loc[
                idx,
                "recovery_method"
            ] = "failed"

    df.to_csv(
        METADATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("복구 후 상태")
    print("=" * 70)

    remaining = df[
        df["status"] != "ok"
    ]

    print(
        "남은 실패:",
        len(remaining)
    )

    if not remaining.empty:

        print(
            remaining[
                [
                    "meeting_date",
                    "original_format",
                    "text_length",
                    "status",
                    "recovery_method",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
