from __future__ import annotations

import argparse
import copy
import json
import random
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

CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "concepts.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v1"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "linguistic_calibration_v01.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "linguistic_calibration_v01_summary.json"
)


# ============================================================
# Constants
# ============================================================

LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]

TARGET_PER_GROUP = 20

PERSONS = [
    "1sg",
    "2sg",
    "3sg",
    "1pl",
    "3pl",
]


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


# ============================================================
# Resources
# ============================================================

def load_concepts():

    concepts = read_jsonl(
        CONCEPT_FILE
    )

    return {
        row["id"]: row
        for row in concepts
    }


# ============================================================
# Helpers
# ============================================================

def get_subject_person(
    row,
    concepts,
):

    subject_id = (
        row
        .get(
            "slots",
            {}
        )
        .get(
            "subject"
        )
    )

    if not subject_id:
        return None


    concept = concepts.get(
        subject_id
    )

    if not concept:
        return None


    return (
        concept
        .get(
            "meta",
            {}
        )
        .get(
            "person"
        )
    )


def replace_once(
    text,
    old,
    new,
):

    text = str(text)
    old = str(old)
    new = str(new)

    if not old:
        return None

    if old not in text:
        return None

    return text.replace(
        old,
        new,
        1,
    )


def choose_wrong_person(
    current_person,
    rng,
):

    candidates = [
        p
        for p in PERSONS
        if p != current_person
    ]

    if not candidates:
        return None

    return rng.choice(
        candidates
    )


# ============================================================
# Russian linguistic corruption
#
# Goal:
#   semantic slots unchanged
#   text and trace remain consistent
#   but Russian verb agreement becomes wrong
#
# Restrict to:
#   present
#   positive
#
# Example:
#
# Я покупаю билет.
# ->
# Я покупает билет.
#
# Hard validator:
# PASS
#
# Qwen:
# should reject grammar/agreement.
# ============================================================

def corrupt_russian_agreement(
    row,
    concepts,
    rng,
):

    features = row.get(
        "features",
        {}
    )

    if features.get(
        "tense"
    ) != "present":

        return None


    if features.get(
        "polarity"
    ) != "pos":

        return None


    slots = row.get(
        "slots",
        {}
    )


    verb_id = slots.get(
        "verb"
    )

    subject_id = slots.get(
        "subject"
    )


    if (
        not verb_id
        or
        not subject_id
    ):

        return None


    verb_concept = concepts.get(
        verb_id
    )

    if not verb_concept:

        return None


    current_person = get_subject_person(
        row,
        concepts,
    )

    if not current_person:

        return None


    wrong_person = choose_wrong_person(
        current_person,
        rng,
    )

    if not wrong_person:
        return None


    ru_forms = (
        verb_concept
        .get(
            "forms",
            {}
        )
        .get(
            "ru",
            {}
        )
    )


    wrong_form = ru_forms.get(
        f"present_{wrong_person}"
    )


    trace = (
        row
        .get(
            "trace",
            {}
        )
        .get(
            "ru",
            {}
        )
    )


    correct_form = trace.get(
        "verb"
    )


    if (
        not correct_form
        or
        not wrong_form
        or
        correct_form == wrong_form
    ):

        return None


    new_text = replace_once(
        row["texts"]["ru"],
        correct_form,
        wrong_form,
    )


    if new_text is None:
        return None


    corrupted = copy.deepcopy(
        row
    )


    corrupted[
        "texts"
    ]["ru"] = new_text


    # Important:
    # update trace intentionally so Hard Validator
    # cannot detect this linguistic corruption.

    corrupted[
        "trace"
    ]["ru"]["verb"] = wrong_form


    corrupted[
        "calibration_expected"
    ] = "REJECT"

    corrupted[
        "calibration_error_type"
    ] = "ru_agreement_error"

    corrupted[
        "calibration_language"
    ] = "ru"

    corrupted[
        "linguistic_corruption"
    ] = {
        "type":
            "verb_person_agreement",

        "language":
            "ru",

        "subject_person":
            current_person,

        "wrong_verb_person":
            wrong_person,

        "original":
            correct_form,

        "corrupted":
            wrong_form,
    }


    return corrupted


# ============================================================
# Uzbek linguistic corruption
#
# Example:
#
# Men ovqatni yeyman.
# ->
# Men ovqatni yeydi.
#
# Trace is also changed to "yeydi".
#
# Hard Validator:
# PASS
#
# Qwen:
# should reject person agreement.
# ============================================================

def corrupt_uzbek_agreement(
    row,
    concepts,
    rng,
):

    slots = row.get(
        "slots",
        {}
    )


    verb_id = slots.get(
        "verb"
    )

    subject_id = slots.get(
        "subject"
    )


    if (
        not verb_id
        or
        not subject_id
    ):

        return None


    verb_concept = concepts.get(
        verb_id
    )

    if not verb_concept:

        return None


    current_person = get_subject_person(
        row,
        concepts,
    )

    if not current_person:

        return None


    polarity = (
        row
        .get(
            "features",
            {}
        )
        .get(
            "polarity",
            "pos",
        )
    )


    if polarity not in {
        "pos",
        "neg",
    }:

        return None


    wrong_person = choose_wrong_person(
        current_person,
        rng,
    )


    if not wrong_person:
        return None


    uz_forms = (
        verb_concept
        .get(
            "forms",
            {}
        )
        .get(
            "uz",
            {}
        )
    )


    wrong_form = uz_forms.get(
        f"finite_{polarity}_{wrong_person}"
    )


    trace = (
        row
        .get(
            "trace",
            {}
        )
        .get(
            "uz",
            {}
        )
    )


    correct_form = trace.get(
        "verb"
    )


    if (
        not correct_form
        or
        not wrong_form
        or
        correct_form == wrong_form
    ):

        return None


    new_text = replace_once(
        row["texts"]["uz"],
        correct_form,
        wrong_form,
    )


    if new_text is None:
        return None


    corrupted = copy.deepcopy(
        row
    )


    corrupted[
        "texts"
    ]["uz"] = new_text


    corrupted[
        "trace"
    ]["uz"]["verb"] = wrong_form


    corrupted[
        "calibration_expected"
    ] = "REJECT"

    corrupted[
        "calibration_error_type"
    ] = "uz_agreement_error"

    corrupted[
        "calibration_language"
    ] = "uz"

    corrupted[
        "linguistic_corruption"
    ] = {
        "type":
            "verb_person_agreement",

        "language":
            "uz",

        "subject_person":
            current_person,

        "wrong_verb_person":
            wrong_person,

        "original":
            correct_form,

        "corrupted":
            wrong_form,
    }


    return corrupted


# ============================================================
# English linguistic corruption
#
# Present positive only.
#
# He buys ...
# ->
# He buy ...
#
# or:
#
# We buy ...
# ->
# We buys ...
#
# Trace is updated too.
# ============================================================

def corrupt_english_agreement(
    row,
    concepts,
    rng,
):

    features = row.get(
        "features",
        {}
    )


    if features.get(
        "tense"
    ) != "present":

        return None


    if features.get(
        "polarity"
    ) != "pos":

        return None


    slots = row.get(
        "slots",
        {}
    )


    verb_id = slots.get(
        "verb"
    )

    subject_id = slots.get(
        "subject"
    )


    if (
        not verb_id
        or
        not subject_id
    ):

        return None


    verb_concept = concepts.get(
        verb_id
    )


    if not verb_concept:
        return None


    current_person = get_subject_person(
        row,
        concepts,
    )


    if not current_person:
        return None


    en_forms = (
        verb_concept
        .get(
            "forms",
            {}
        )
        .get(
            "en",
            {}
        )
    )


    base_form = en_forms.get(
        "base"
    )

    third_form = en_forms.get(
        "present_3sg"
    )


    if (
        not base_form
        or
        not third_form
    ):

        return None


    trace = (
        row
        .get(
            "trace",
            {}
        )
        .get(
            "en",
            {}
        )
    )


    correct_form = trace.get(
        "verb"
    )


    if not correct_form:
        return None


    if current_person == "3sg":

        wrong_form = base_form

    else:

        wrong_form = third_form


    if wrong_form == correct_form:
        return None


    new_text = replace_once(
        row["texts"]["en"],
        correct_form,
        wrong_form,
    )


    if new_text is None:
        return None


    corrupted = copy.deepcopy(
        row
    )


    corrupted[
        "texts"
    ]["en"] = new_text


    corrupted[
        "trace"
    ]["en"]["verb"] = wrong_form


    corrupted[
        "calibration_expected"
    ] = "REJECT"

    corrupted[
        "calibration_error_type"
    ] = "en_agreement_error"

    corrupted[
        "calibration_language"
    ] = "en"

    corrupted[
        "linguistic_corruption"
    ] = {
        "type":
            "subject_verb_agreement",

        "language":
            "en",

        "subject_person":
            current_person,

        "original":
            correct_form,

        "corrupted":
            wrong_form,
    }


    return corrupted


# ============================================================
# Chinese unnatural word order
#
# We keep all semantic surfaces unchanged.
#
# Example:
#
# 我今天买票。
# ->
# 我买票今天。
#
# or
#
# 我明天08:30去机场。
# ->
# 我去机场明天08:30。
#
# Hard Validator:
# PASS because all trace surfaces are still present.
#
# Qwen:
# should flag unnatural Chinese word order.
# ============================================================

def corrupt_chinese_word_order(
    row,
):

    trace = (
        row
        .get(
            "trace",
            {}
        )
        .get(
            "zh",
            {}
        )
    )


    subject = trace.get(
        "subject"
    )

    verb = trace.get(
        "verb"
    )

    obj = trace.get(
        "object"
    )

    destination = trace.get(
        "destination"
    )

    time_surface = (
        trace.get("time")
        or
        trace.get("day")
    )

    clock = trace.get(
        "clock"
    )


    # Need subject + verb + temporal element.
    if (
        not subject
        or
        not verb
        or
        not time_surface
    ):

        return None


    complement = (
        obj
        or
        destination
    )


    if not complement:
        return None


    # Deliberately awkward:
    #
    # subject + verb + complement + time
    #
    # If clock exists:
    # subject + verb + complement + day + clock

    if clock:

        bad_text = (
            f"{subject}"
            f"{verb}"
            f"{complement}"
            f"{time_surface}"
            f"{clock}"
            f"。"
        )

    else:

        bad_text = (
            f"{subject}"
            f"{verb}"
            f"{complement}"
            f"{time_surface}"
            f"。"
        )


    if (
        bad_text
        ==
        row[
            "texts"
        ]["zh"]
    ):

        return None


    corrupted = copy.deepcopy(
        row
    )


    corrupted[
        "texts"
    ]["zh"] = bad_text


    # Do NOT alter semantic slots.
    #
    # Trace remains valid because every expected
    # surface is still present.

    corrupted[
        "calibration_expected"
    ] = "REJECT"

    corrupted[
        "calibration_error_type"
    ] = "zh_word_order_error"

    corrupted[
        "calibration_language"
    ] = "zh"

    corrupted[
        "linguistic_corruption"
    ] = {
        "type":
            "unnatural_word_order",

        "language":
            "zh",

        "original":
            row[
                "texts"
            ]["zh"],

        "corrupted":
            bad_text,
    }


    return corrupted


# ============================================================
# Candidate collection
# ============================================================

def collect_corruptions(
    rows,
    corrupt_fn,
    target_n,
    used_ids,
    rng,
    *extra_args,
):

    candidates = rows[:]

    rng.shuffle(
        candidates
    )


    results = []


    for row in candidates:

        semantic_id = row.get(
            "semantic_id"
        )


        if semantic_id in used_ids:
            continue


        corrupted = corrupt_fn(
            row,
            *extra_args,
        )


        if corrupted is None:
            continue


        results.append(
            corrupted
        )

        used_ids.add(
            semantic_id
        )


        if len(results) >= target_n:
            break


    return results


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
        "--seed",
        type=int,
        default=2026,
    )


    parser.add_argument(
        "--per-group",
        type=int,
        default=TARGET_PER_GROUP,
    )


    args = parser.parse_args()


    input_file = Path(
        args.input
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Input not found:\n"
            f"{input_file}"
        )


    if not CONCEPT_FILE.exists():

        raise FileNotFoundError(
            f"Concept file not found:\n"
            f"{CONCEPT_FILE}"
        )


    rows = read_jsonl(
        input_file
    )


    concepts = load_concepts()


    rng = random.Random(
        args.seed
    )


    used_ids = set()

    calibration = []


    # ========================================================
    # 1. Clean controls
    # ========================================================

    clean_candidates = rows[:]

    rng.shuffle(
        clean_candidates
    )


    for row in clean_candidates:

        semantic_id = row.get(
            "semantic_id"
        )


        if semantic_id in used_ids:
            continue


        clean = copy.deepcopy(
            row
        )


        clean[
            "calibration_expected"
        ] = "ACCEPT"

        clean[
            "calibration_error_type"
        ] = "none"

        clean[
            "calibration_language"
        ] = "none"

        clean[
            "linguistic_corruption"
        ] = None


        calibration.append(
            clean
        )

        used_ids.add(
            semantic_id
        )


        if (
            sum(
                1
                for x in calibration
                if x[
                    "calibration_error_type"
                ]
                == "none"
            )
            >= args.per_group
        ):

            break


    # ========================================================
    # 2. Russian
    # ========================================================

    ru_rows = collect_corruptions(
        rows,
        corrupt_russian_agreement,
        args.per_group,
        used_ids,
        rng,
        concepts,
        rng,
    )

    calibration.extend(
        ru_rows
    )


    # ========================================================
    # 3. Uzbek
    # ========================================================

    uz_rows = collect_corruptions(
        rows,
        corrupt_uzbek_agreement,
        args.per_group,
        used_ids,
        rng,
        concepts,
        rng,
    )

    calibration.extend(
        uz_rows
    )


    # ========================================================
    # 4. English
    # ========================================================

    en_rows = collect_corruptions(
        rows,
        corrupt_english_agreement,
        args.per_group,
        used_ids,
        rng,
        concepts,
        rng,
    )

    calibration.extend(
        en_rows
    )


    # ========================================================
    # 5. Chinese
    # ========================================================

    zh_rows = collect_corruptions(
        rows,
        corrupt_chinese_word_order,
        args.per_group,
        used_ids,
        rng,
    )

    calibration.extend(
        zh_rows
    )


    # ========================================================
    # Verify group counts
    # ========================================================

    counts = Counter(
        row[
            "calibration_error_type"
        ]
        for row in calibration
    )


    expected_groups = [
        "none",
        "ru_agreement_error",
        "uz_agreement_error",
        "en_agreement_error",
        "zh_word_order_error",
    ]


    missing = []

    for group in expected_groups:

        actual = counts.get(
            group,
            0,
        )

        if actual < args.per_group:

            missing.append(
                (
                    group,
                    actual,
                )
            )


    if missing:

        print(
            "\n[ERROR] Could not build enough "
            "calibration samples:"
        )

        for group, actual in missing:

            print(
                f"{group:<24}"
                f"{actual}/"
                f"{args.per_group}"
            )

        raise RuntimeError(
            "Linguistic calibration "
            "dataset incomplete."
        )


    # ========================================================
    # Shuffle and assign IDs
    # ========================================================

    rng.shuffle(
        calibration
    )


    for index, row in enumerate(
        calibration,
        start=1,
    ):

        row[
            "calibration_id"
        ] = (
            f"ling_cal_{index:04d}"
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    write_jsonl(
        OUTPUT_FILE,
        calibration,
    )


    expected_counter = Counter(
        row[
            "calibration_expected"
        ]
        for row in calibration
    )


    language_counter = Counter(
        row[
            "calibration_language"
        ]
        for row in calibration
    )


    summary = {
        "source_file":
            str(input_file),

        "seed":
            args.seed,

        "per_group":
            args.per_group,

        "total":
            len(calibration),

        "expected_labels":
            dict(
                expected_counter
            ),

        "error_types":
            dict(
                counts
            ),

        "languages":
            dict(
                language_counter
            ),
    }


    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # Console
    # ========================================================

    print("=" * 80)
    print("LINGUISTIC CALIBRATION V1")
    print("=" * 80)


    print(
        "Source rows:",
        len(rows),
    )


    print(
        "Calibration rows:",
        len(calibration),
    )


    print(
        "\nExpected labels:"
    )

    for key, value in sorted(
        expected_counter.items()
    ):

        print(
            f"{key:<15}"
            f"{value}"
        )


    print(
        "\nError types:"
    )

    for key, value in sorted(
        counts.items()
    ):

        print(
            f"{key:<25}"
            f"{value}"
        )


    print(
        "\nLanguages:"
    )

    for key, value in sorted(
        language_counter.items()
    ):

        print(
            f"{key:<10}"
            f"{value}"
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()