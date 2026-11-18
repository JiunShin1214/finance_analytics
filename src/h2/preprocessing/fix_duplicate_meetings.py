from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import requests
from pypdf import PdfReader


from src.h2.data_io import ROOT as BASE_DIR

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
    / "bok_metadata_full_before_duplicate_fix.csv"
)

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

from src.h2.data_io import office_executable
SOFFICE = office_executable()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64)"
    )
}

TARGET_MONTHS = [
    "2001-09",
    "2008-10",
]


def extract_pdf_text(path):

    reader = PdfReader(path)

    texts = []

    for page in reader.pages:

        text = page.extract_text() or ""

        if text.strip():
            texts.append(text)

    return "\n".join(texts)


def read_txt(path):

    for encoding in [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]:

        try:
            return path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError:
            continue

    return ""


def detect_hwp_version(path):

    header = path.read_bytes()[:32]

    if header.startswith(
        b"HWP Document File V3.00"
    ):
        return "hwp3"

    if header.startswith(
        bytes.fromhex(
            "D0 CF 11 E0 A1 B1 1A E1"
        )
    ):
        return "hwp5"

    return "unknown"


def convert_hwp5(
    input_path,
    output_path
):

    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("hwp5txt.exe")),
            str(input_path),
        ],
        capture_output=True
    )

    if result.returncode != 0:
        return ""

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

    return ""


def convert_hwp3(
    input_path,
    output_dir
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    subprocess.run(
        [
            str(SOFFICE),
            "--headless",
            "--convert-to",
            "txt",
            "--outdir",
            str(output_dir),
            str(input_path),
        ],
        capture_output=True
    )

    output_path = (
        output_dir
        / f"{input_path.stem}.txt"
    )

    if not output_path.exists():
        return "", None

    return (
        read_txt(output_path),
        output_path
    )


def relative(path):

    return str(
        path.relative_to(BASE_DIR)
    ).replace(
        "\\",
        "/"
    )


def main():

    df = pd.read_csv(
        METADATA_PATH
    )

    if not BACKUP_PATH.exists():

        shutil.copy2(
            METADATA_PATH,
            BACKUP_PATH
        )

    session = requests.Session()

    targets = df[
        df["year_month"]
        .isin(TARGET_MONTHS)
    ].copy()

    print(
        "재수집 대상:",
        len(targets),
        "건"
    )

    for idx, row in targets.iterrows():

        meeting_date = str(
            row["meeting_date"]
        )

        year = meeting_date[:4]

        file_type = str(
            row["original_format"]
        ).lower()

        attachment_url = (
            row["attachment_url"]
        )

        print()
        print("=" * 70)
        print(
            "재수집:",
            meeting_date
        )

        date_tag = (
            meeting_date
            .replace("-", "_")
        )

        raw_dir = (
            RAW_BASE
            / year
        )

        raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        raw_path = (
            raw_dir
            / (
                f"{date_tag}"
                f"_policy."
                f"{file_type}"
            )
        )

        response = session.get(
            attachment_url,
            headers=HEADERS,
            timeout=60
        )

        response.raise_for_status()

        raw_path.write_bytes(
            response.content
        )

        print(
            "다운로드:",
            raw_path.name
        )

        processed_path = None
        processed_format = None
        text = ""

        if file_type == "pdf":

            text = extract_pdf_text(
                raw_path
            )

            processed_path = raw_path
            processed_format = "pdf"

        elif file_type == "hwp":

            version = detect_hwp_version(
                raw_path
            )

            print(
                "HWP 버전:",
                version
            )

            converted_dir = (
                CONVERTED_BASE
                / year
            )

            converted_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            if version == "hwp5":

                txt_path = (
                    converted_dir
                    / (
                        f"{date_tag}"
                        "_policy.txt"
                    )
                )

                text = convert_hwp5(
                    raw_path,
                    txt_path
                )

                processed_path = txt_path

            else:

                text, processed_path = (
                    convert_hwp3(
                        raw_path,
                        converted_dir
                    )
                )

            processed_format = "txt"

        print(
            "텍스트 길이:",
            len(text)
        )

        df.loc[
            idx,
            "original_path"
        ] = relative(
            raw_path
        )

        df.loc[
            idx,
            "processed_path"
        ] = relative(
            processed_path
        )

        df.loc[
            idx,
            "processed_format"
        ] = processed_format

        df.loc[
            idx,
            "text_length"
        ] = len(text)

        df.loc[
            idx,
            "status"
        ] = (
            "ok"
            if len(text) > 0
            else "text_empty"
        )

    df.to_csv(
        METADATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("수정 완료")
    print("=" * 70)

    print(
        df[
            df["year_month"]
            .isin(TARGET_MONTHS)
        ][
            [
                "meeting_date",
                "original_path",
                "text_length",
                "status",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
