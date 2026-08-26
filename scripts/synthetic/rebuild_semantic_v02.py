from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from scripts.synthetic.renderer_v2 import (
    load_verb_policies,
    render_zh_verb_v2,
    render_ru_verb_v2,
    replace_once,
    is_find_concept,
)


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
    / "renderer_v2"
)


DEFAULT_OUTPUT = (
    DEFAULT_OUTPUT_DIR
    / "semantic_v02.jsonl"
)


DEFAULT_SUMMARY = (
    DEFAULT_OUTPUT_DIR
    / "semantic_v02_summary.json"
)


CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "concepts.jsonl"
)


# ============================================================
# IO
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:

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
) -> None:

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


    concepts = {}


    for row in rows:

        concept_id = row.get(
            "id"
        )

        if not concept_id:
            continue


        concepts[
            str(concept_id)
        ] = row


    return concepts


# ============================================================
# Semantic helpers
# ============================================================

def get_subject_person(
    row: dict,
    concepts: dict[str, dict],
) -> str | None:

    subject_id = (
        row
        .get(
            "slots",
            {},
        )
        .get(
            "subject"
        )
    )


    if not subject_id:

        return None


    concept = concepts.get(
        str(subject_id)
    )


    if not concept:

        return None


    return (
        concept
        .get(
            "meta",
            {},
        )
        .get(
            "person"
        )
    )


def get_verb_id(
    row: dict,
) -> str | None:

    value = (
        row
        .get(
            "slots",
            {},
        )
        .get(
            "verb"
        )
    )


    if value is None:
        return None


    return str(
        value
    )


# ============================================================
# Chinese V2 patch
# ============================================================

def rebuild_chinese(
    row: dict,
    policies: dict,
) -> tuple[bool, str | None]:

    verb_id = get_verb_id(
        row
    )


    if not verb_id:

        return False, None


    if not is_find_concept(
        verb_id
    ):

        return False, None


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


    # Only known problem:
    # FIND + negative

    if polarity != "neg":

        return False, None


    texts = row.get(
        "texts",
        {},
    )

    trace = row.get(
        "trace",
        {},
    )


    zh_trace = trace.get(
        "zh",
        {},
    )


    old_surface = zh_trace.get(
        "verb"
    )


    if not old_surface:

        return (
            False,
            "zh_trace_missing_verb",
        )


    new_surface = render_zh_verb_v2(
        verb_id=verb_id,
        original_surface=str(
            old_surface
        ),
        tense=tense,
        polarity=polarity,
        policies=policies,
    )


    if new_surface == old_surface:

        return False, None


    old_text = texts.get(
        "zh",
        "",
    )


    new_text = replace_once(
        old_text,
        str(old_surface),
        str(new_surface),
    )


    if new_text is None:

        return (
            False,
            (
                "zh_old_surface_not_found:"
                f"{old_surface}"
            ),
        )


    row[
        "texts"
    ]["zh"] = new_text


    row[
        "trace"
    ]["zh"]["verb"] = (
        new_surface
    )


    return True, None


# ============================================================
# Russian V2 patch
# ============================================================

def rebuild_russian(
    row: dict,
    concepts: dict[str, dict],
    policies: dict,
) -> tuple[bool, str | None]:

    verb_id = get_verb_id(
        row
    )


    if not verb_id:

        return False, None


    if not is_find_concept(
        verb_id
    ):

        return False, None


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


    # Current V2 known Russian fix:
    # FIND + future

    if tense != "future":

        return False, None


    person = get_subject_person(
        row,
        concepts,
    )


    if not person:

        return (
            False,
            "ru_subject_person_missing",
        )


    texts = row.get(
        "texts",
        {},
    )

    trace = row.get(
        "trace",
        {},
    )


    ru_trace = trace.get(
        "ru",
        {},
    )


    old_surface = ru_trace.get(
        "verb"
    )


    if not old_surface:

        return (
            False,
            "ru_trace_missing_verb",
        )


    try:

        new_surface = render_ru_verb_v2(
            verb_id=verb_id,
            original_surface=str(
                old_surface
            ),
            person=person,
            tense=tense,
            polarity=polarity,
            policies=policies,
        )

    except Exception as exc:

        return (
            False,
            "ru_render_error:"
            + repr(exc),
        )


    if new_surface == old_surface:

        return False, None


    old_text = texts.get(
        "ru",
        "",
    )


    new_text = replace_once(
        old_text,
        str(old_surface),
        str(new_surface),
    )


    if new_text is None:

        return (
            False,
            (
                "ru_old_surface_not_found:"
                f"{old_surface}"
            ),
        )


    row[
        "texts"
    ]["ru"] = new_text


    row[
        "trace"
    ]["ru"]["verb"] = (
        new_surface
    )


    return True, None


# ============================================================
# Rebuild one sample
# ============================================================

def rebuild_row(
    source_row: dict,
    concepts: dict[str, dict],
    policies: dict,
) -> tuple[dict, list[str], list[str]]:

    row = copy.deepcopy(
        source_row
    )


    fixes = []
    errors = []


    # ========================================================
    # Chinese
    # ========================================================

    zh_changed, zh_error = (
        rebuild_chinese(
            row,
            policies,
        )
    )


    if zh_changed:

        fixes.append(
            "zh_find_negative"
        )


    if zh_error:

        errors.append(
            zh_error
        )


    # ========================================================
    # Russian
    # ========================================================

    ru_changed, ru_error = (
        rebuild_russian(
            row,
            concepts,
            policies,
        )
    )


    if ru_changed:

        fixes.append(
            "ru_find_future_aspect"
        )


    if ru_error:

        errors.append(
            ru_error
        )


    # ========================================================
    # Provenance
    # ========================================================

    row[
        "renderer_version"
    ] = "v2"


    row[
        "renderer_v2"
    ] = {
        "source_version":
            source_row.get(
                "renderer_version",
                "v1",
            ),

        "fixes":
            fixes,

        "errors":
            errors,
    }


    return (
        row,
        fixes,
        errors,
    )


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
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT
        ),
    )


    args = parser.parse_args()


    input_file = Path(
        args.input
    )

    output_file = Path(
        args.output
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{input_file}"
        )


    rows = read_jsonl(
        input_file
    )


    concepts = load_concepts()


    policies = load_verb_policies()


    output_rows = []

    fix_counter = Counter()

    error_counter = Counter()


    for source_row in rows:

        (
            row,
            fixes,
            errors,
        ) = rebuild_row(
            source_row,
            concepts,
            policies,
        )


        output_rows.append(
            row
        )


        fix_counter.update(
            fixes
        )


        error_counter.update(
            errors
        )


    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    write_jsonl(
        output_file,
        output_rows,
    )


    summary_file = (
        output_file.parent
        / "semantic_v02_summary.json"
    )


    summary = {
        "source_file":
            str(input_file),

        "output_file":
            str(output_file),

        "total_rows":
            len(rows),

        "renderer_version":
            "v2",

        "fix_counts":
            dict(
                fix_counter
            ),

        "error_counts":
            dict(
                error_counter
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
    print("REBUILD SEMANTIC V02")
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
        "\nApplied fixes:"
    )


    if fix_counter:

        for key, value in (
            fix_counter
            .most_common()
        ):

            print(
                f"{key:<35}"
                f"{value}"
            )

    else:

        print(
            "None"
        )


    print(
        "\nErrors:"
    )


    if error_counter:

        for key, value in (
            error_counter
            .most_common()
        ):

            print(
                f"{key:<70}"
                f"{value}"
            )

    else:

        print(
            "None"
        )


    print(
        "\nSaved:"
    )

    print(
        output_file
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    main()