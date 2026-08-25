from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "semantic_v01_raw.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
)

VALID_FILE = (
    OUTPUT_DIR
    / "semantic_v01_valid.jsonl"
)

REJECT_FILE = (
    OUTPUT_DIR
    / "semantic_v01_rejected.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "validation_summary.json"
)


LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]


HAN_RE = re.compile(
    r"[\u4e00-\u9fff]"
)

CYRILLIC_RE = re.compile(
    r"[\u0400-\u04FF]"
)

CLOCK_RE = re.compile(
    r"(?<!\d)(?:[01]\d|2[0-3]):[0-5]\d(?!\d)"
)


def normalize(
    text,
):

    return (
        str(text)
        .replace("’", "'")
        .replace("ʻ", "'")
        .replace("`", "'")
        .casefold()
        .strip()
    )


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


def validate_scripts(
    texts,
):

    errors = []

    zh = texts["zh"]
    en = texts["en"]
    ru = texts["ru"]
    uz = texts["uz"]


    if not HAN_RE.search(zh):
        errors.append(
            "zh_missing_han"
        )


    if CYRILLIC_RE.search(en):
        errors.append(
            "en_contains_cyrillic"
        )


    if HAN_RE.search(en):
        errors.append(
            "en_contains_han"
        )


    if not CYRILLIC_RE.search(ru):
        errors.append(
            "ru_missing_cyrillic"
        )


    if CYRILLIC_RE.search(uz):
        errors.append(
            "uz_contains_cyrillic"
        )


    if HAN_RE.search(uz):
        errors.append(
            "uz_contains_han"
        )


    return errors


def validate_trace(
    row,
):

    errors = []

    texts = row["texts"]
    trace = row["trace"]


    for lang in LANGUAGES:

        text_norm = normalize(
            texts[lang]
        )


        for (
            slot,
            surface,
        ) in trace[
            lang
        ].items():

            surface_norm = normalize(
                surface
            )


            if (
                surface_norm
                not in text_norm
            ):

                errors.append(
                    f"trace_missing:"
                    f"{lang}:"
                    f"{slot}"
                )


    return errors

def validate_clock(
    row,
):

    errors = []

    expected_clock = (
        row
        .get("computed", {})
        .get("clock")
    )

    # 这个 Semantic Sample 本来就没有 clock，
    # 不需要检查
    if not expected_clock:
        return errors

    texts = row.get(
        "texts",
        {}
    )

    for lang in LANGUAGES:

        text = str(
            texts.get(
                lang,
                ""
            )
        )

        if expected_clock not in text:

            errors.append(
                f"clock_missing:{lang}"
            )

    return errors
def validate_row(
    row,
):

    errors = []


    if "texts" not in row:

        return [
            "missing_texts"
        ]


    texts = row[
        "texts"
    ]


    for lang in LANGUAGES:

        text = texts.get(
            lang,
            "",
        )


        if not str(
            text
        ).strip():

            errors.append(
                f"empty_{lang}"
            )


        if (
            "{"
            in str(text)
            or
            "}"
            in str(text)
        ):

            errors.append(
                f"placeholder_{lang}"
            )


        if len(
            str(text)
        ) > 300:

            errors.append(
                f"too_long_{lang}"
            )


    errors.extend(
        validate_scripts(
            texts
        )
    )


    errors.extend(
        validate_trace(
            row
        )
    )


    errors.extend(
        validate_clock(
            row
        )
    )


    return sorted(
        set(errors)
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    rows = read_jsonl(
        INPUT_FILE
    )


    accepted = []

    rejected = []

    reason_counter = Counter()

    semantic_text_keys = set()


    for row in rows:

        errors = validate_row(
            row
        )


        text_key = tuple(
            normalize(
                row["texts"][
                    lang
                ]
            )
            for lang in LANGUAGES
        )


        if text_key in (
            semantic_text_keys
        ):

            errors.append(
                "duplicate_semantic_text"
            )

        else:

            semantic_text_keys.add(
                text_key
            )


        if errors:

            row[
                "validation_errors"
            ] = sorted(
                set(errors)
            )

            rejected.append(
                row
            )

            reason_counter.update(
                row[
                    "validation_errors"
                ]
            )

        else:

            row[
                "rule_validation"
            ] = "PASS"

            accepted.append(
                row
            )


    with open(
        VALID_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for row in accepted:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


    with open(
        REJECT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rejected:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


    summary = {
        "total":
            len(rows),

        "accepted":
            len(accepted),

        "rejected":
            len(rejected),

        "accept_rate":
            (
                len(accepted)
                / len(rows)
                if rows
                else 0
            ),

        "reject_reasons":
            dict(
                reason_counter
            ),
    }


    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print("=" * 80)
    print("Synthetic validation")
    print("=" * 80)

    print(
        "Total:",
        len(rows),
    )

    print(
        "Accepted:",
        len(accepted),
    )

    print(
        "Rejected:",
        len(rejected),
    )

    print(
        "Accept rate:",
        f"{summary['accept_rate']:.2%}",
    )

    print(
        "\nReject reasons:"
    )

    for key, value in (
        reason_counter.items()
    ):

        print(
            f"{key:<30} "
            f"{value}"
        )


if __name__ == "__main__":
    main()