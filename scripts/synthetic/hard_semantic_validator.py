from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "semantic_v01_valid.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "hard_v2"
)

LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]


# ============================================================
# Regex
# ============================================================

HAN_RE = re.compile(
    r"[\u4e00-\u9fff]"
)

CYRILLIC_RE = re.compile(
    r"[\u0400-\u04FF]"
)

CLOCK_RE = re.compile(
    r"(?<!\d)"
    r"(?:[01]\d|2[0-3]):[0-5]\d"
    r"(?!\d)"
)


# ------------------------------------------------------------
# Unexpected negative markers
# ------------------------------------------------------------

EN_NEG_RE = re.compile(
    r"\b("
    r"not|never|cannot|can't|"
    r"don't|doesn't|didn't|"
    r"won't|wouldn't|"
    r"shouldn't|couldn't"
    r")\b",
    flags=re.IGNORECASE,
)

RU_NEG_RE = re.compile(
    r"\b(?:не|нет|никогда)\b",
    flags=re.IGNORECASE,
)

UZ_NEG_RE = re.compile(
    r"("
    r"\bemas\b|"
    r"\byo['‘’ʻ`]q\b|"
    r"\bhech\b|"
    r"\b\w*may\w*\b|"
    r"\b\w*mas\w*\b"
    r")",
    flags=re.IGNORECASE,
)


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def write_jsonl(
    path: Path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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


def get_record_id(row):

    return (
        row.get("calibration_id")
        or
        row.get("semantic_id")
        or
        row.get("id")
    )


# ============================================================
# Normalization
# ============================================================

def normalize_text(text):

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("ʻ", "'")
        .replace("`", "'")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return (
        text.strip().casefold()
    )


# ============================================================
# 1. Required fields
# ============================================================

def validate_required_fields(row):

    errors = []

    if not get_record_id(row):

        errors.append(
            "missing_record_id"
        )

    texts = row.get(
        "texts"
    )

    if not isinstance(
        texts,
        dict,
    ):

        return [
            "missing_texts"
        ]


    for lang in LANGUAGES:

        text = texts.get(
            lang
        )

        if (
            text is None
            or
            not str(text).strip()
        ):

            errors.append(
                f"empty_text:{lang}"
            )


    return errors


# ============================================================
# 2. Script checks
# ============================================================

def validate_scripts(row):

    errors = []

    texts = row.get(
        "texts",
        {},
    )


    zh = str(
        texts.get(
            "zh",
            "",
        )
    )

    en = str(
        texts.get(
            "en",
            "",
        )
    )

    ru = str(
        texts.get(
            "ru",
            "",
        )
    )

    uz = str(
        texts.get(
            "uz",
            "",
        )
    )


    if not HAN_RE.search(zh):

        errors.append(
            "zh_missing_han"
        )


    if HAN_RE.search(en):

        errors.append(
            "en_contains_han"
        )

    if CYRILLIC_RE.search(en):

        errors.append(
            "en_contains_cyrillic"
        )


    if not CYRILLIC_RE.search(ru):

        errors.append(
            "ru_missing_cyrillic"
        )


    if HAN_RE.search(uz):

        errors.append(
            "uz_contains_han"
        )

    if CYRILLIC_RE.search(uz):

        errors.append(
            "uz_contains_cyrillic"
        )


    return errors


# ============================================================
# 3. Placeholder check
# ============================================================

def validate_placeholders(row):

    errors = []

    texts = row.get(
        "texts",
        {},
    )


    for lang in LANGUAGES:

        text = str(
            texts.get(
                lang,
                "",
            )
        )

        if (
            "{"
            in text
            or
            "}"
            in text
        ):

            errors.append(
                f"placeholder:{lang}"
            )


    return errors


# ============================================================
# 4. Trace / semantic-slot consistency
#
# trace is generated by our renderer and represents the
# canonical expected surface forms.
#
# Example:
#
# trace["en"]["object"] = "food"
#
# but corrupted English:
# "I eat water."
#
# -> expected "food" is missing
# -> hard reject
# ============================================================

def validate_trace(row):

    errors = []

    texts = row.get(
        "texts",
        {},
    )

    trace = row.get(
        "trace",
        {},
    )


    if not isinstance(
        trace,
        dict,
    ):

        return [
            "missing_trace"
        ]


    for lang in LANGUAGES:

        text_norm = normalize_text(
            texts.get(
                lang,
                "",
            )
        )

        lang_trace = trace.get(
            lang,
            {},
        )


        if not isinstance(
            lang_trace,
            dict,
        ):

            errors.append(
                f"missing_trace_lang:{lang}"
            )

            continue


        for (
            slot_name,
            expected_surface,
        ) in lang_trace.items():

            expected_surface = str(
                expected_surface
            ).strip()

            if not expected_surface:
                continue


            expected_norm = normalize_text(
                expected_surface
            )


            if (
                expected_norm
                not in text_norm
            ):

                errors.append(
                    "slot_surface_mismatch:"
                    f"{lang}:"
                    f"{slot_name}:"
                    f"expected={expected_surface}"
                )


    return errors


# ============================================================
# 5. Clock consistency
# ============================================================

def validate_clock(row):

    errors = []

    expected_clock = (
        row
        .get(
            "computed",
            {},
        )
        .get(
            "clock"
        )
    )


    if not expected_clock:

        return errors


    expected_clock = str(
        expected_clock
    ).strip()


    texts = row.get(
        "texts",
        {},
    )


    for lang in LANGUAGES:

        text = str(
            texts.get(
                lang,
                "",
            )
        )

        found = CLOCK_RE.findall(
            text
        )


        if not found:

            errors.append(
                f"clock_missing:{lang}"
            )

            continue


        if expected_clock not in found:

            errors.append(
                f"clock_mismatch:{lang}:"
                f"expected={expected_clock}:"
                f"found={','.join(found)}"
            )


    return errors


# ============================================================
# 6. Polarity consistency
#
# Negative samples are already protected by the exact
# expected verb surface stored in trace.
#
# Here we especially detect unexpected added negation in
# canonical POSITIVE samples.
# ============================================================

def validate_polarity(row):

    errors = []

    polarity = (
        row
        .get(
            "features",
            {},
        )
        .get(
            "polarity"
        )
    )


    if polarity not in {
        "pos",
        "neg",
    }:

        return errors


    texts = row.get(
        "texts",
        {},
    )


    # --------------------------------------------------------
    # Negative sample:
    #
    # trace already contains:
    #
    # zh: 不吃
    # en: do not eat
    # ru: не ...
    # uz: yemayman
    #
    # If negation is removed, validate_trace() catches it.
    # --------------------------------------------------------

    if polarity == "neg":

        return errors


    # --------------------------------------------------------
    # Positive sample:
    # unexpected negation must not appear.
    # --------------------------------------------------------

    zh = str(
        texts.get(
            "zh",
            "",
        )
    )

    en = str(
        texts.get(
            "en",
            "",
        )
    )

    ru = str(
        texts.get(
            "ru",
            "",
        )
    )

    uz = str(
        texts.get(
            "uz",
            "",
        )
    )


    # Controlled synthetic vocabulary:
    # these markers are safe as hard rules in V0.1.

    if any(
        marker in zh
        for marker in [
            "不",
            "没",
            "没有",
            "别",
        ]
    ):

        errors.append(
            "unexpected_negation:zh"
        )


    if EN_NEG_RE.search(en):

        errors.append(
            "unexpected_negation:en"
        )


    if RU_NEG_RE.search(ru):

        errors.append(
            "unexpected_negation:ru"
        )


    if UZ_NEG_RE.search(uz):

        errors.append(
            "unexpected_negation:uz"
        )


    return errors


# ============================================================
# 7. Basic structural consistency
# ============================================================

def validate_basic_structure(row):

    errors = []

    texts = row.get(
        "texts",
        {},
    )


    for lang in LANGUAGES:

        text = str(
            texts.get(
                lang,
                "",
            )
        ).strip()


        if len(text) > 400:

            errors.append(
                f"text_too_long:{lang}"
            )


        if len(text) < 2:

            errors.append(
                f"text_too_short:{lang}"
            )


    return errors


# ============================================================
# Full validation
# ============================================================

def validate_row(row):

    errors = []

    errors.extend(
        validate_required_fields(
            row
        )
    )

    # If texts itself is missing,
    # do not continue deeper checks.

    if "missing_texts" in errors:

        return sorted(
            set(errors)
        )


    errors.extend(
        validate_scripts(
            row
        )
    )

    errors.extend(
        validate_placeholders(
            row
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

    errors.extend(
        validate_polarity(
            row
        )
    )

    errors.extend(
        validate_basic_structure(
            row
        )
    )


    return sorted(
        set(errors)
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )


    args = parser.parse_args()


    input_file = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{input_file}"
        )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    all_file = (
        output_dir
        / "hard_judged.jsonl"
    )

    accepted_file = (
        output_dir
        / "hard_accepted.jsonl"
    )

    rejected_file = (
        output_dir
        / "hard_rejected.jsonl"
    )

    summary_file = (
        output_dir
        / "hard_summary.json"
    )


    rows = read_jsonl(
        input_file
    )


    accepted = []
    rejected = []
    judged = []

    reason_counter = Counter()


    for row in rows:

        errors = validate_row(
            row
        )

        row = dict(
            row
        )


        row[
            "hard_validation"
        ] = {
            "pass":
                len(errors) == 0,

            "errors":
                errors,
        }


        row[
            "hard_accept"
        ] = (
            len(errors) == 0
        )


        judged.append(
            row
        )


        if errors:

            rejected.append(
                row
            )

            reason_counter.update(
                errors
            )

        else:

            accepted.append(
                row
            )


    write_jsonl(
        all_file,
        judged,
    )

    write_jsonl(
        accepted_file,
        accepted,
    )

    write_jsonl(
        rejected_file,
        rejected,
    )


    summary = {
        "input_file":
            str(input_file),

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
                reason_counter.most_common()
            ),
    }


    with summary_file.open(
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
    print("HARD SEMANTIC VALIDATOR V2")
    print("=" * 80)

    print(
        "Input:",
        input_file,
    )

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
        (
            f"{len(accepted)/len(rows):.2%}"
            if rows
            else "0%"
        ),
    )


    print(
        "\nReject reasons:"
    )

    if reason_counter:

        for (
            reason,
            count,
        ) in reason_counter.most_common():

            print(
                f"{reason:<65}"
                f"{count}"
            )

    else:

        print(
            "None"
        )


    print(
        "\nFiles:"
    )

    print(
        all_file
    )

    print(
        accepted_file
    )

    print(
        rejected_file
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    main()