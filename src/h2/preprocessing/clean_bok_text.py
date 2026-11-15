from pathlib import Path
import re

import pandas as pd


from src.h2.data_io import ROOT as BASE_DIR

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bok_text_full.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bok_text_clean.csv"
)


def repair_known_errors(text, meeting_date):
    """
    근거가 확인된 추출 오류만 명시적으로 교정.
    원본 text 컬럼은 수정하지 않음.
    """

    # 2002-04-04:
    # 동시기 언론의 금통위 발표문 인용을 통해 원문 확인
    if meeting_date == "2002-04-04":

        text = text.replace(
            "경기회복의 충분한 ??이 확인되지 않고 있으며",
            "경기회복의 충분한 부력(浮力)이 확인되지 않고 있으며"
        )

    return text


def is_broken_old_title(line):
    """
    2001~2002 구형 HWP 변환 시 깨진 제목 행만 판별.

    예:
    2001㈃ 1?? ??????
    2002㈃ 10?? ??????

    본문의 ??는 삭제하지 않도록
    '연도로 시작하는 짧은 행'에만 적용.
    """

    stripped = line.strip()

    if len(stripped) > 80:
        return False

    if "??" not in stripped:
        return False

    # 2001, 2002 등 4자리 연도로 시작하는 제목
    if re.match(r"^20\d{2}", stripped):
        return True

    return False


def clean_document(text, title, meeting_date):

    if pd.isna(text):
        return ""

    text = str(text)

    # -----------------------------------------
    # 1. 알려진 추출 오류 복구
    # -----------------------------------------
    text = repair_known_errors(
        text,
        meeting_date
    )

    # -----------------------------------------
    # 2. 줄 단위 정리
    # -----------------------------------------
    lines = text.splitlines()

    clean_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # 첨부 표시 제거
        if line in {
            "(붙 임)",
            "(붙임)",
            "붙 임",
            "붙임",
        }:
            continue

        # 구형 HWP에서 깨진 제목만 제거
        if is_broken_old_title(line):
            continue

        clean_lines.append(line)

    body = "\n".join(
        clean_lines
    )

    # -----------------------------------------
    # 3. 공백 정리
    # -----------------------------------------

    # 탭/여러 칸 공백
    body = re.sub(
        r"[ \t]+",
        " ",
        body
    )

    # 과도한 빈 줄
    body = re.sub(
        r"\n{3,}",
        "\n\n",
        body
    )

    body = body.strip()

    # -----------------------------------------
    # 4. metadata의 정상 제목을 앞에 삽입
    # -----------------------------------------

    clean_text = (
        f"{str(title).strip()}\n"
        f"{body}"
    )

    return clean_text.strip()


def main():

    df = pd.read_csv(
        INPUT_PATH
    )

    df["clean_text"] = df.apply(
        lambda row: clean_document(
            row["text"],
            row["title"],
            str(row["meeting_date"])
        ),
        axis=1
    )

    df["clean_text_length"] = (
        df["clean_text"]
        .str.len()
    )

    # 원문 대비 너무 많이 사라진 문서 확인용
    df["length_ratio"] = (
        df["clean_text_length"]
        / df["text_length"]
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 70)
    print("===== BOK 텍스트 정제 결과 =====")
    print("=" * 70)

    print(
        "전체 문서:",
        len(df)
    )

    print(
        "빈 clean_text:",
        (df["clean_text_length"] == 0).sum()
    )

    print(
        "평균 clean_text 길이:",
        round(
            df["clean_text_length"].mean(),
            1
        )
    )

    # -----------------------------------------
    # ?? 잔존 여부
    # -----------------------------------------

    remaining = df[
        df["clean_text"]
        .fillna("")
        .str.contains(
            "??",
            regex=False
        )
    ]

    print(
        "정제 후 ?? 포함 문서:",
        len(remaining)
    )

    if not remaining.empty:

        print(
            remaining[
                [
                    "meeting_date",
                    "clean_text_length",
                ]
            ]
            .to_string(index=False)
        )

    # -----------------------------------------
    # 2002-04-04 복구 확인
    # -----------------------------------------

    target = df[
        df["meeting_date"]
        == "2002-04-04"
    ]

    if not target.empty:

        text = target.iloc[0][
            "clean_text"
        ]

        keyword = "경기회복의"

        pos = text.find(keyword)

        print()
        print("=" * 70)
        print("2002-04-04 복구 확인")
        print("=" * 70)

        if pos != -1:

            print(
                text[
                    max(0, pos - 50):
                    pos + 150
                ]
            )

    print()
    print(
        "저장 완료:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()