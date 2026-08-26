from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v1"
    / "linguistic_calibration_v01.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "grammar_hard_v1"
)

CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "concepts.jsonl"
)

LANGUAGES = [
    "en",
    "ru",
    "uz",
]


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path) -> list[dict]:

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
    rows: list[dict],
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


def load_concepts() -> dict[str, dict]:

    if not CONCEPT_FILE.exists():

        raise FileNotFoundError(
            f"Concept file not found:\n"
            f"{CONCEPT_FILE}"
        )

    rows = read_jsonl(
        CONCEPT_FILE
    )

    return {
        row["id"]: row
        for row in rows
        if "id" in row
    }


def get_record_id(row: dict) -> str:

    return str(
        row.get("calibration_id")
        or row.get("semantic_id")
        or row.get("id")
        or "UNKNOWN"
    )


# ============================================================
# Helpers
# ============================================================

def normalize_text(value: Any) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def contains_surface(
    text: str,
    surface: str,
) -> bool:

    text = normalize_text(
        text
    )

    surface = normalize_text(
        surface
    )

    if not surface:
        return False

    # --------------------------------------------------------
    # Exact token / phrase matching
    #
    # Prevent:
    #
    # "go"   matching "goes"
    # "eat"  matching "eats"
    # "find" matching "finds"
    #
    # while still allowing multi-word forms such as:
    #
    # "do not eat"
    # "will go"
    # "sotib oladi"
    # --------------------------------------------------------

    pattern = (
        r"(?<![\w'’ʻ])"
        +
        re.escape(surface)
        +
        r"(?![\w'’ʻ])"
    )

    return (
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )
def get_subject_id(
    row: dict,
) -> str | None:

    return (
        row
        .get("slots", {})
        .get("subject")
    )


def get_verb_id(
    row: dict,
) -> str | None:

    return (
        row
        .get("slots", {})
        .get("verb")
    )


def get_subject_person(
    row: dict,
    concepts: dict[str, dict],
) -> str | None:

    subject_id = get_subject_id(
        row
    )

    if not subject_id:
        return None

    subject = concepts.get(
        subject_id
    )

    if not subject:
        return None

    return (
        subject
        .get("meta", {})
        .get("person")
    )


def get_verb_concept(
    row: dict,
    concepts: dict[str, dict],
) -> dict | None:

    verb_id = get_verb_id(
        row
    )

    if not verb_id:
        return None

    return concepts.get(
        verb_id
    )


# ============================================================
# ENGLISH
# ============================================================

def validate_english(
    row: dict,
    concepts: dict[str, dict],
) -> list[str]:

    errors = []

    sentence = (
        row
        .get("texts", {})
        .get("en", "")
    )

    features = row.get(
        "features",
        {},
    )

    tense = features.get(
        "tense"
    )

    polarity = features.get(
        "polarity"
    )

    person = get_subject_person(
        row,
        concepts,
    )

    verb = get_verb_concept(
        row,
        concepts,
    )

    if (
        not sentence
        or
        not person
        or
        not verb
    ):
        return errors

    forms = (
        verb
        .get("forms", {})
        .get("en", {})
    )

    base = forms.get(
        "base"
    )

    third = forms.get(
        "present_3sg"
    )

    # --------------------------------------------------------
    # Deterministic rule:
    # present + positive
    #
    # 3sg -> present_3sg
    # others -> base
    # --------------------------------------------------------

    if (
        tense == "present"
        and
        polarity == "pos"
        and
        base
        and
        third
    ):

        if person == "3sg":

            expected = third

        else:

            expected = base

        if not contains_surface(
            sentence,
            expected,
        ):

            errors.append(
                "en_subject_verb_agreement:"
                f"person={person}:"
                f"expected={expected}"
            )

    return errors


# ============================================================
# RUSSIAN
# ============================================================

def validate_russian(
    row: dict,
    concepts: dict[str, dict],
) -> list[str]:

    errors = []

    sentence = (
        row
        .get("texts", {})
        .get("ru", "")
    )

    features = row.get(
        "features",
        {},
    )

    tense = features.get(
        "tense"
    )

    polarity = features.get(
        "polarity"
    )

    person = get_subject_person(
        row,
        concepts,
    )

    verb = get_verb_concept(
        row,
        concepts,
    )

    if (
        not sentence
        or
        not person
        or
        not verb
    ):
        return errors

    forms = (
        verb
        .get("forms", {})
        .get("ru", {})
    )

    # --------------------------------------------------------
    # 当前 V1 只硬判：
    # present + positive
    #
    # Future aspect 暂时不在 Hard Validator 判断。
    # --------------------------------------------------------

    if (
        tense == "present"
        and
        polarity == "pos"
    ):

        key = (
            f"present_{person}"
        )

        expected = forms.get(
            key
        )

        if (
            expected
            and
            not contains_surface(
                sentence,
                expected,
            )
        ):

            errors.append(
                "ru_subject_verb_agreement:"
                f"person={person}:"
                f"expected={expected}"
            )

    return errors


# ============================================================
# UZBEK
# ============================================================

def get_uz_allowed_forms(
    forms: dict,
    polarity: str,
    person: str,
) -> list[str]:

    """
    Uzbek 第三人称复数需要特殊处理。

    对显式 Ular 主语：

        Ular keladi.
        Ular keladilar.

    均可能成立。

    因此 3pl 不应机械要求只有 3pl morphology。
    """

    allowed = []

    exact = forms.get(
        f"finite_{polarity}_{person}"
    )

    if exact:

        allowed.append(
            exact
        )

    if person == "3pl":

        third_singular = forms.get(
            f"finite_{polarity}_3sg"
        )

        if (
            third_singular
            and
            third_singular
            not in allowed
        ):

            allowed.append(
                third_singular
            )

    return allowed


def validate_uzbek(
    row: dict,
    concepts: dict[str, dict],
) -> list[str]:

    errors = []

    sentence = (
        row
        .get("texts", {})
        .get("uz", "")
    )

    features = row.get(
        "features",
        {},
    )

    polarity = features.get(
        "polarity"
    )

    person = get_subject_person(
        row,
        concepts,
    )

    verb = get_verb_concept(
        row,
        concepts,
    )

    if (
        not sentence
        or
        not person
        or
        not verb
    ):
        return errors

    if polarity not in {
        "pos",
        "neg",
    }:
        return errors

    forms = (
        verb
        .get("forms", {})
        .get("uz", {})
    )

    allowed_forms = (
        get_uz_allowed_forms(
            forms,
            polarity,
            person,
        )
    )

    if not allowed_forms:
        return errors

    matched = any(
        contains_surface(
            sentence,
            surface,
        )
        for surface
        in allowed_forms
    )

    if not matched:

        errors.append(
            "uz_subject_verb_agreement:"
            f"person={person}:"
            f"allowed="
            f"{'|'.join(allowed_forms)}"
        )

    return errors


# ============================================================
# Full row validation
# ============================================================

def validate_row(
    row: dict,
    concepts: dict[str, dict],
) -> dict:

    errors_by_language = {
        "en":
            validate_english(
                row,
                concepts,
            ),

        "ru":
            validate_russian(
                row,
                concepts,
            ),

        "uz":
            validate_uzbek(
                row,
                concepts,
            ),
    }

    all_errors = []

    for lang, lang_errors in (
        errors_by_language.items()
    ):

        for error in lang_errors:

            all_errors.append(
                f"{lang}:{error}"
            )

    return {
        "pass":
            len(all_errors) == 0,

        "errors":
            all_errors,

        "errors_by_language":
            errors_by_language,
    }


# ============================================================
# MAIN
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

    concepts = load_concepts()

    rows = read_jsonl(
        input_file
    )

    judged = []
    accepted = []
    rejected = []

    reason_counter = Counter()
    language_counter = Counter()

    for source_row in rows:

        row = dict(
            source_row
        )

        result = validate_row(
            row,
            concepts,
        )

        row[
            "grammar_hard_validation"
        ] = result

        row[
            "grammar_hard_accept"
        ] = result[
            "pass"
        ]

        judged.append(
            row
        )

        if result["pass"]:

            accepted.append(
                row
            )

        else:

            rejected.append(
                row
            )

            for error in result[
                "errors"
            ]:

                reason_counter[
                    error
                ] += 1

            for lang, errors in (
                result[
                    "errors_by_language"
                ].items()
            ):

                if errors:

                    language_counter[
                        lang
                    ] += 1

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    judged_file = (
        output_dir
        / "grammar_judged.jsonl"
    )

    accepted_file = (
        output_dir
        / "grammar_accepted.jsonl"
    )

    rejected_file = (
        output_dir
        / "grammar_rejected.jsonl"
    )

    summary_file = (
        output_dir
        / "grammar_summary.json"
    )

    write_jsonl(
        judged_file,
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
                reason_counter
                .most_common()
            ),

        "rejected_by_language":
            dict(
                language_counter
                .most_common()
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

    print("=" * 90)
    print("GRAMMAR HARD VALIDATOR V1")
    print("=" * 90)

    print(
        "Input:",
        input_file
    )

    print(
        "Total:",
        len(rows)
    )

    print(
        "Accepted:",
        len(accepted)
    )

    print(
        "Rejected:",
        len(rejected)
    )

    print(
        "Accept rate:",
        (
            f"{len(accepted)/len(rows):.2%}"
            if rows
            else "0%"
        )
    )

    print(
        "\nRejected by language:"
    )

    if language_counter:

        for lang, count in (
            language_counter
            .most_common()
        ):

            print(
                f"{lang:<8}{count}"
            )

    else:

        print(
            "None"
        )

    print(
        "\nReject reasons:"
    )

    if reason_counter:

        for reason, count in (
            reason_counter
            .most_common()
        ):

            print(
                f"{reason:<80}"
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
        judged_file
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