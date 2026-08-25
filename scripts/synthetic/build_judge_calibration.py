from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "synthetic_v01_audit_100.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "judge_calibration_v01.jsonl"
)


# ============================================================
# IO
# ============================================================

def read_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(
    path,
    rows,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# corruption functions
# ============================================================

def corrupt_clock(
    row,
):

    row = copy.deepcopy(
        row
    )

    clock = (
        row
        .get("computed", {})
        .get("clock")
    )

    if not clock:
        return None


    match = re.match(
        r"(\d{2}):(\d{2})",
        clock,
    )

    if not match:
        return None


    hour = int(
        match.group(1)
    )

    minute = match.group(2)


    wrong_hour = (
        hour + 5
    ) % 24


    wrong_clock = (
        f"{wrong_hour:02d}:"
        f"{minute}"
    )


    # 只改英文，
    # 其他三种语言保持正确
    en = row["texts"]["en"]

    if clock not in en:
        return None


    row["texts"]["en"] = (
        en.replace(
            clock,
            wrong_clock,
            1,
        )
    )


    row[
        "calibration_expected"
    ] = "REJECT"

    row[
        "calibration_error_type"
    ] = "time_error"

    return row


def corrupt_negation(
    row,
):

    row = copy.deepcopy(
        row
    )

    en = row["texts"]["en"]


    # future positive -> negative
    if " will " in en and " will not " not in en:

        row["texts"]["en"] = (
            en.replace(
                " will ",
                " will not ",
                1,
            )
        )

    # future negative -> positive
    elif " will not " in en:

        row["texts"]["en"] = (
            en.replace(
                " will not ",
                " will ",
                1,
            )
        )

    elif " does not " in en:

        row["texts"]["en"] = (
            en.replace(
                " does not ",
                " ",
                1,
            )
        )

    elif " do not " in en:

        row["texts"]["en"] = (
            en.replace(
                " do not ",
                " ",
                1,
            )
        )

    else:

        # 通用方式：
        # 强行改变Chinese polarity
        zh = row["texts"]["zh"]

        # 避免问句等不适合的样本
        if len(zh) < 3:
            return None

        row["texts"]["zh"] = (
            "不" + zh
        )


    row[
        "calibration_expected"
    ] = "REJECT"

    row[
        "calibration_error_type"
    ] = "negation_error"

    return row


ENTITY_REPLACEMENTS = [
    ("the airport", "the hotel"),
    ("the hotel", "the airport"),
    ("the station", "the hospital"),
    ("the hospital", "the station"),
    ("Tashkent", "Moscow"),
    ("Moscow", "Beijing"),
    ("Beijing", "Tashkent"),
    ("a phone", "a passport"),
    ("a passport", "a phone"),
    ("a ticket", "a book"),
    ("a book", "a ticket"),
    ("food", "water"),
    ("water", "coffee"),
]


def corrupt_entity(
    row,
):

    row = copy.deepcopy(
        row
    )

    en = row["texts"]["en"]


    for old, new in (
        ENTITY_REPLACEMENTS
    ):

        if old in en:

            row[
                "texts"
            ]["en"] = (
                en.replace(
                    old,
                    new,
                    1,
                )
            )

            row[
                "calibration_expected"
            ] = "REJECT"

            row[
                "calibration_error_type"
            ] = "entity_error"

            return row


    return None


def corrupt_meaning(
    row,
    donor,
):

    row = copy.deepcopy(
        row
    )

    # 用完全不同样本的 Uzbek
    # 替换当前 Uzbek
    row["texts"]["uz"] = (
        donor[
            "texts"
        ]["uz"]
    )


    row[
        "calibration_expected"
    ] = "REJECT"

    row[
        "calibration_error_type"
    ] = "meaning_error"

    return row


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()


    rng = random.Random(
        args.seed
    )


    rows = read_jsonl(
        INPUT_FILE
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    rng.shuffle(
        rows
    )


    calibration = []


    # ========================================================
    # 1. CLEAN CONTROL - 20
    # ========================================================

    for row in rows[:20]:

        clean = copy.deepcopy(
            row
        )

        clean[
            "calibration_expected"
        ] = "ACCEPT"

        clean[
            "calibration_error_type"
        ] = "none"

        calibration.append(
            clean
        )


    # ========================================================
    # 2. TIME ERRORS - 20
    # ========================================================

    count = 0

    for row in rows:

        corrupted = corrupt_clock(
            row
        )

        if corrupted is None:
            continue

        calibration.append(
            corrupted
        )

        count += 1

        if count >= 20:
            break


    # ========================================================
    # 3. NEGATION ERRORS - 20
    # ========================================================

    count = 0

    for row in rows:

        corrupted = corrupt_negation(
            row
        )

        if corrupted is None:
            continue

        calibration.append(
            corrupted
        )

        count += 1

        if count >= 20:
            break


    # ========================================================
    # 4. ENTITY / OBJECT ERRORS - 20
    # ========================================================

    count = 0

    for row in rows:

        corrupted = corrupt_entity(
            row
        )

        if corrupted is None:
            continue

        calibration.append(
            corrupted
        )

        count += 1

        if count >= 20:
            break


    # ========================================================
    # 5. GENERAL MEANING ERRORS - 20
    # ========================================================

    for i in range(20):

        source = rows[
            i
        ]

        donor = rows[
            -(i + 1)
        ]

        corrupted = (
            corrupt_meaning(
                source,
                donor,
            )
        )

        calibration.append(
            corrupted
        )


    rng.shuffle(
        calibration
    )


    for i, row in enumerate(
        calibration,
        start=1,
    ):

        row[
            "calibration_id"
        ] = (
            f"cal_{i:04d}"
        )


    write_jsonl(
        OUTPUT_FILE,
        calibration,
    )


    print("=" * 80)
    print("Judge Calibration Dataset")
    print("=" * 80)

    print(
        "Total:",
        len(calibration),
    )


    from collections import Counter

    labels = Counter(
        row[
            "calibration_expected"
        ]
        for row in calibration
    )

    error_types = Counter(
        row[
            "calibration_error_type"
        ]
        for row in calibration
    )


    print(
        "\nExpected labels:"
    )

    for k, v in (
        labels.items()
    ):

        print(
            f"{k:<15}{v}"
        )


    print(
        "\nError types:"
    )

    for k, v in sorted(
        error_types.items()
    ):

        print(
            f"{k:<20}{v}"
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()