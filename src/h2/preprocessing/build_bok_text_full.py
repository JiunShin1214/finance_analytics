from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from docx import Document


from src.h2.data_io import ROOT as BASE_DIR

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bok_metadata_full.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bok_text_full.csv"
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

    raise ValueError(
        f"텍스트 인코딩 확인 필요: {path}"
    )


def read_pdf(path):
    reader = PdfReader(path)

    texts = []

    for page in reader.pages:
        text = page.extract_text() or ""

        if text.strip():
            texts.append(text)

    return "\n".join(texts)


def read_docx(path):
    doc = Document(path)

    texts = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            texts.append(text)

    return "\n".join(texts)


def load_text(row):

    processed_path = row["processed_path"]

    if pd.isna(processed_path):
        return ""

    path = (
        BASE_DIR
        / processed_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"파일 없음: {path}"
        )

    file_type = str(
        row["processed_format"]
    ).lower()

    if file_type == "txt":
        return read_txt(path)

    elif file_type == "pdf":
        return read_pdf(path)

    elif file_type == "docx":
        return read_docx(path)

    else:
        raise ValueError(
            f"지원하지 않는 형식: {file_type}"
        )


def main():

    metadata = pd.read_csv(
        METADATA_PATH
    )

    records = []

    for i, row in metadata.iterrows():

        meeting_date = row[
            "meeting_date"
        ]

        try:

            text = load_text(
                row
            )

            status = (
                "ok"
                if len(text.strip()) > 0
                else "empty"
            )

        except Exception as e:

            print(
                "[오류]",
                meeting_date,
                type(e).__name__,
                e
            )

            text = ""
            status = "error"

        records.append({

            "meeting_date":
                row["meeting_date"],

            "release_date":
                row["release_date"],

            "release_date_source":
                row.get(
                    "release_date_source",
                    None
                ),

            "year_month":
                row["year_month"],

            "title":
                row["title"],

            "source_url":
                row["source_url"],

            "original_file_name":
                row.get(
                    "original_file_name",
                    None
                ),

            "original_format":
                row["original_format"],

            "processed_format":
                row["processed_format"],

            "text":
                text,

            "text_length":
                len(text),

            "load_status":
                status,
        })

        if (i + 1) % 25 == 0:
            print(
                f"{i + 1}개 처리 완료"
            )

    df = pd.DataFrame(
        records
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("===== BOK 전체 텍스트 구축 결과 =====")
    print("=" * 70)

    print(
        "전체 문서:",
        len(df)
    )

    print(
        "정상:",
        (df["load_status"] == "ok").sum()
    )

    print(
        "빈 텍스트:",
        (df["load_status"] == "empty").sum()
    )

    print(
        "오류:",
        (df["load_status"] == "error").sum()
    )

    print()
    print(
        "텍스트 길이 요약:"
    )

    print(
        df["text_length"]
        .describe()
    )

    print()
    print(
        "저장 완료:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()