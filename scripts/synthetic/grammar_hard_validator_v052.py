from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.2"
FROZEN_PARENT_VERSION = "0.5.1"

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v052_smoke_100"
    / "02_hard_semantic"
    / "hard_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v052_smoke_100"
    / "03_grammar_hard"
)


NEW_FRAMES = {
    "CALL_PERSON",
    "WAIT_PERSON",
    "WAIT_AT_PLACE",
}

NEW_VERBS = {
    "CALL",
    "WAIT",
}


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSONL: "
                    f"{path}:{line_no}"
                ) from exc

            rows.append(row)

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


def write_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")


def add_violation(
    violations: list[dict],
    violation_type: str,
    *,
    language: str | None = None,
    **kwargs,
) -> None:

    item = {
        "type": violation_type,
    }

    if language is not None:
        item[
            "language"
        ] = language

    item.update(kwargs)

    violations.append(
        item
    )


# ============================================================
# Frozen V0.5.1 Grammar Validator
# ============================================================

def run_frozen_v051_validator(
    *,
    rows: list[dict],
    output_dir: Path,
) -> list[dict]:

    if not rows:
        return []

    input_file = (
        output_dir
        / "_frozen_v051_input.jsonl"
    )

    validator_dir = (
        output_dir
        / "_frozen_v051_validator"
    )

    write_jsonl(
        input_file,
        rows,
    )

    validator_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.synthetic.grammar_hard_validator_v051",
        "--input",
        str(input_file),
        "--output-dir",
        str(validator_dir),
    ]

    print()
    print("=" * 100)
    print(
        "RUNNING FROZEN V0.5.1 GRAMMAR HARD"
    )
    print("=" * 100)

    print(
        "Rows:",
        len(rows),
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    # ========================================================
    # Locate final V0.5.1 outputs
    #
    # IMPORTANT:
    # We must use the FINAL V0.5.1 accepted/rejected files as
    # the source of truth.
    #
    # Some frozen V0.4.4.1 rows may carry legacy validation
    # field names. Merely reading row["grammar_validation"]
    # therefore loses their final acceptance status.
    # ========================================================

    judged_file = (
        validator_dir
        / "grammar_judged.jsonl"
    )

    accepted_file = (
        validator_dir
        / "grammar_accepted.jsonl"
    )

    rejected_file = (
        validator_dir
        / "grammar_rejected.jsonl"
    )

    if not judged_file.exists():

        candidates = sorted(
            validator_dir.rglob(
                "grammar_judged.jsonl"
            )
        )

        if candidates:
            judged_file = candidates[0]

    if not accepted_file.exists():

        candidates = sorted(
            validator_dir.rglob(
                "grammar_accepted.jsonl"
            )
        )

        if candidates:
            accepted_file = candidates[0]

    if not rejected_file.exists():

        candidates = sorted(
            validator_dir.rglob(
                "grammar_rejected.jsonl"
            )
        )

        if candidates:
            rejected_file = candidates[0]

    if not judged_file.exists():

        raise RuntimeError(
            "Frozen V0.5.1 grammar validator "
            "did not produce grammar_judged.jsonl. "
            f"Return code={result.returncode}"
        )

    if not accepted_file.exists():

        raise RuntimeError(
            "Frozen V0.5.1 grammar validator "
            "did not produce grammar_accepted.jsonl."
        )

    if not rejected_file.exists():

        raise RuntimeError(
            "Frozen V0.5.1 grammar validator "
            "did not produce grammar_rejected.jsonl."
        )

    judged_rows = read_jsonl(
        judged_file
    )

    accepted_rows = read_jsonl(
        accepted_file
    )

    rejected_rows = read_jsonl(
        rejected_file
    )

    if len(judged_rows) != len(rows):

        raise RuntimeError(
            "Frozen grammar validator row "
            "count mismatch: "
            f"input={len(rows)}, "
            f"output={len(judged_rows)}"
        )

    # ========================================================
    # Canonical acceptance IDs
    # ========================================================

    accepted_ids = {
        row.get(
            "semantic_id"
        )
        for row in accepted_rows
    }

    rejected_ids = {
        row.get(
            "semantic_id"
        )
        for row in rejected_rows
    }

    overlap = (
        accepted_ids
        & rejected_ids
    )

    if overlap:

        raise RuntimeError(
            "Frozen validator produced semantic_ids "
            "in both accepted and rejected sets: "
            f"{sorted(overlap)[:10]}"
        )

    # ========================================================
    # Normalize legacy validation fields
    #
    # Final V0.5.1 accepted/rejected output is authoritative.
    #
    # This deliberately prevents:
    #
    # V0.4 diagnostic reject
    #        ↓
    # V0.4.4.1 corrected accept
    #        ↓
    # V0.5.2 accidentally reading stale/legacy field
    # ========================================================

    normalized = []

    for row in judged_rows:

        output_row = copy.deepcopy(
            row
        )

        semantic_id = output_row.get(
            "semantic_id"
        )

        # Preserve any old validator information for debugging.
        legacy_validation = None

        for key in (
            "grammar_validation",
            "grammar_hard_validation",
        ):

            value = output_row.get(
                key
            )

            if (
                isinstance(
                    value,
                    dict,
                )
                and "accept" in value
            ):

                legacy_validation = copy.deepcopy(
                    value
                )

                break

        if legacy_validation is not None:

            output_row[
                "frozen_grammar_diagnostic"
            ] = legacy_validation

        # ----------------------------------------------------
        # Accepted by final frozen V0.5.1 validator
        # ----------------------------------------------------

        if semantic_id in accepted_ids:

            output_row[
                "grammar_validation"
            ] = {
                "validator_version":
                    FROZEN_PARENT_VERSION,

                "accept":
                    True,

                "violations":
                    [],
            }

        # ----------------------------------------------------
        # Rejected by final frozen V0.5.1 validator
        # ----------------------------------------------------

        elif semantic_id in rejected_ids:

            violations = []

            if legacy_validation:

                violations = copy.deepcopy(
                    legacy_validation.get(
                        "violations",
                        [],
                    )
                )

            if not violations:

                violations = [
                    {
                        "type":
                            "FROZEN_V051_GRAMMAR_REJECT"
                    }
                ]

            output_row[
                "grammar_validation"
            ] = {
                "validator_version":
                    FROZEN_PARENT_VERSION,

                "accept":
                    False,

                "violations":
                    violations,
            }

        else:

            raise RuntimeError(
                "Frozen grammar row was neither "
                "accepted nor rejected by V0.5.1: "
                f"{semantic_id}"
            )

        normalized.append(
            output_row
        )

    # ========================================================
    # Sanity checks
    # ========================================================

    normalized_accept = sum(
        1
        for row in normalized
        if row.get(
            "grammar_validation",
            {},
        ).get(
            "accept"
        )
    )

    normalized_reject = (
        len(normalized)
        - normalized_accept
    )

    print()
    print(
        "Frozen V0.5.1 normalized accepted:",
        normalized_accept,
    )

    print(
        "Frozen V0.5.1 normalized rejected:",
        normalized_reject,
    )

    if (
        normalized_accept
        != len(
            accepted_rows
        )
    ):

        raise RuntimeError(
            "Frozen accepted normalization "
            "count mismatch."
        )

    if (
        normalized_reject
        != len(
            rejected_rows
        )
    ):

        raise RuntimeError(
            "Frozen rejected normalization "
            "count mismatch."
        )

    return normalized
# ============================================================
# Utility
# ============================================================

def contains_phrase(
    text: str,
    phrase: str,
) -> bool:

    if not text or not phrase:
        return False

    return phrase in text


# ============================================================
# V0.5.2 Grammar checks
# ============================================================

def validate_new_row(
    row: dict,
    *,
    renderer,
) -> dict:

    violations = []

    frame_id = row.get(
        "frame_id"
    )

    slots = row.get(
        "slots",
        {},
    )

    features = row.get(
        "features",
        {},
    )

    texts = row.get(
        "texts",
        {},
    )

    if not isinstance(
        slots,
        dict,
    ):

        add_violation(
            violations,
            "INVALID_SLOTS",
        )

        return {
            "validator_version": VERSION,
            "accept": False,
            "violations": violations,
        }

    if not isinstance(
        texts,
        dict,
    ):

        add_violation(
            violations,
            "INVALID_TEXTS",
        )

        return {
            "validator_version": VERSION,
            "accept": False,
            "violations": violations,
        }

    subject_id = slots.get(
        "subject"
    )

    verb_id = slots.get(
        "verb"
    )

    tense = features.get(
        "tense"
    )

    polarity = features.get(
        "polarity"
    )

    # ========================================================
    # Subject
    # ========================================================

    subject = renderer.subjects.get(
        subject_id,
        {},
    )

    person_code = subject.get(
        "person_code"
    )

    if not person_code:

        add_violation(
            violations,
            "UNKNOWN_SUBJECT_PERSON",
        )

        return {
            "validator_version": VERSION,
            "accept": False,
            "violations": violations,
        }

    # ========================================================
    # Expected verbal morphology
    # ========================================================

    try:

        en_expected = (
            renderer._en_verb(
                verb_id=verb_id,
                person_code=person_code,
                tense=tense,
                polarity=polarity,
            )
        )

        ru_expected = (
            renderer._ru_verb(
                verb_id=verb_id,
                person_code=person_code,
                tense=tense,
                polarity=polarity,
            )
        )

        uz_expected = (
            renderer._uz_verb(
                verb_id=verb_id,
                person_code=person_code,
                polarity=polarity,
            )
        )

    except Exception as exc:

        add_violation(
            violations,
            "VERB_MORPHOLOGY_LOOKUP_FAILED",
            reason=str(exc),
        )

        return {
            "validator_version": VERSION,
            "accept": False,
            "violations": violations,
        }

    # ========================================================
    # English agreement / tense
    # ========================================================

    en_text = texts.get(
        "en",
        "",
    )

    if not contains_phrase(
        en_text,
        en_expected,
    ):

        add_violation(
            violations,
            "EN_AGREEMENT_OR_TENSE",
            language="en",
            expected=en_expected,
            actual=en_text,
        )

    # ========================================================
    # Russian agreement / tense
    # ========================================================

    ru_text = texts.get(
        "ru",
        "",
    )

    if not contains_phrase(
        ru_text,
        ru_expected,
    ):

        add_violation(
            violations,
            "RU_AGREEMENT_OR_TENSE",
            language="ru",
            expected=ru_expected,
            actual=ru_text,
        )

    # ========================================================
    # Uzbek agreement / polarity
    #
    # Uzbek present-future morphology is intentionally treated
    # as capable of carrying future semantics.
    #
    # Do NOT require an English-style explicit "will".
    # ========================================================

    uz_text = texts.get(
        "uz",
        "",
    )

    if not contains_phrase(
        uz_text,
        uz_expected,
    ):

        add_violation(
            violations,
            "UZ_AGREEMENT_OR_POLARITY",
            language="uz",
            expected=uz_expected,
            actual=uz_text,
        )

    # ========================================================
    # Argument case / grammatical construction
    # ========================================================

    if frame_id == "CALL_PERSON":

        person_id = slots.get(
            "person"
        )

        target = renderer.person_targets.get(
            person_id,
            {},
        )

        en_object = target.get(
            "en_object"
        )

        ru_dat = target.get(
            "ru_dat"
        )

        uz_dat = target.get(
            "uz_dat"
        )

        if (
            en_object
            and en_object not in en_text
        ):

            add_violation(
                violations,
                "EN_OBJECT_CASE",
                language="en",
                expected=en_object,
            )

        if (
            ru_dat
            and ru_dat not in ru_text
        ):

            add_violation(
                violations,
                "RU_DATIVE_CASE",
                language="ru",
                expected=ru_dat,
            )

        if (
            uz_dat
            and uz_dat not in uz_text
        ):

            add_violation(
                violations,
                "UZ_DATIVE_CASE",
                language="uz",
                expected=uz_dat,
            )

        # Chinese CALL construction
        zh_text = texts.get(
            "zh",
            "",
        )

        zh_target = target.get(
            "zh"
        )

        if zh_target:

            required = (
                f"给{zh_target}打电话"
            )

            if required not in zh_text:

                add_violation(
                    violations,
                    "ZH_CALL_CONSTRUCTION",
                    language="zh",
                    expected=required,
                )

    # ========================================================
    # WAIT_PERSON
    # ========================================================

    elif frame_id == "WAIT_PERSON":

        person_id = slots.get(
            "person"
        )

        target = renderer.person_targets.get(
            person_id,
            {},
        )

        en_object = target.get(
            "en_object"
        )

        ru_acc = target.get(
            "ru_acc"
        )

        uz_acc = target.get(
            "uz_acc"
        )

        if (
            en_object
            and f"for {en_object}" not in en_text
        ):

            add_violation(
                violations,
                "EN_WAIT_FOR_CONSTRUCTION",
                language="en",
                expected=f"for {en_object}",
            )

        if (
            ru_acc
            and ru_acc not in ru_text
        ):

            add_violation(
                violations,
                "RU_ACCUSATIVE_CASE",
                language="ru",
                expected=ru_acc,
            )

        if (
            uz_acc
            and uz_acc not in uz_text
        ):

            add_violation(
                violations,
                "UZ_ACCUSATIVE_CASE",
                language="uz",
                expected=uz_acc,
            )

    # ========================================================
    # WAIT_AT_PLACE
    # ========================================================

    elif frame_id == "WAIT_AT_PLACE":

        place_id = slots.get(
            "place"
        )

        place = renderer.place_targets.get(
            place_id,
            {},
        )

        expected_places = {
            "zh":
                place.get(
                    "zh_loc"
                ),

            "en":
                place.get(
                    "en_loc"
                ),

            "ru":
                place.get(
                    "ru_loc"
                ),

            "uz":
                place.get(
                    "uz_loc"
                ),
        }

        for language, expected in (
            expected_places.items()
        ):

            if not expected:
                continue

            actual = texts.get(
                language,
                "",
            )

            if expected not in actual:

                add_violation(
                    violations,
                    "LOCATIVE_GRAMMAR",
                    language=language,
                    expected=expected,
                    actual=actual,
                )

    else:

        add_violation(
            violations,
            "UNSUPPORTED_V052_FRAME",
        )

    # ========================================================
    # Upstream integrity
    # ========================================================

    hard_semantic = row.get(
        "hard_semantic_validation",
        {},
    )

    if (
        isinstance(
            hard_semantic,
            dict,
        )
        and hard_semantic.get(
            "accept"
        ) is False
    ):

        add_violation(
            violations,
            "UPSTREAM_HARD_SEMANTIC_REJECT",
        )

    return {
        "validator_version":
            VERSION,

        "accept":
            len(violations) == 0,

        "violations":
            violations,
    }


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Grammar Hard Validator V0.5.2"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    rows = read_jsonl(
        args.input
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_rows = []
    new_rows = []

    for row in rows:

        if row.get(
            "frame_id"
        ) in NEW_FRAMES:

            new_rows.append(
                row
            )

        else:

            frozen_rows.append(
                row
            )

    print("=" * 100)
    print("GRAMMAR HARD VALIDATOR V0.5.2")
    print("=" * 100)

    print(
        "Input:",
        args.input,
    )

    print(
        "Total:",
        len(rows),
    )

    print(
        "Frozen V0.5.1 rows:",
        len(frozen_rows),
    )

    print(
        "V0.5.2 new rows:",
        len(new_rows),
    )

    # ========================================================
    # Frozen
    # ========================================================

    frozen_judged = (
        run_frozen_v051_validator(
            rows=frozen_rows,
            output_dir=args.output_dir,
        )
    )

    # ========================================================
    # New
    # ========================================================

    from scripts.synthetic.renderer_v052 import (
        V052Renderer,
    )

    renderer = V052Renderer()

    new_judged = []

    for row in new_rows:

        output_row = copy.deepcopy(
            row
        )

        output_row[
            "grammar_validation"
        ] = validate_new_row(
            row,
            renderer=renderer,
        )

        new_judged.append(
            output_row
        )

    # ========================================================
    # Merge preserving order
    # ========================================================

    by_id = {}

    for row in (
        frozen_judged
        + new_judged
    ):

        semantic_id = row.get(
            "semantic_id"
        )

        if semantic_id in by_id:

            raise RuntimeError(
                f"Duplicate semantic_id: "
                f"{semantic_id}"
            )

        by_id[
            semantic_id
        ] = row

    judged = []

    for original in rows:

        semantic_id = original.get(
            "semantic_id"
        )

        row = by_id.get(
            semantic_id
        )

        if row is None:

            raise RuntimeError(
                "Missing validated semantic_id: "
                f"{semantic_id}"
            )

        judged.append(
            row
        )

    # ========================================================
    # Statistics
    # ========================================================

    accepted = []
    rejected = []

    reject_counter = Counter()
    language_counter = Counter()

    new_accepted = 0
    new_rejected = 0

    new_frame_counter = Counter()

    for row in judged:

        validation = row.get(
            "grammar_validation",
            {},
        )

        accept = bool(
            validation.get(
                "accept",
                False,
            )
        )

        frame_id = row.get(
            "frame_id"
        )

        if accept:

            accepted.append(
                row
            )

        else:

            rejected.append(
                row
            )

        if frame_id in NEW_FRAMES:

            new_frame_counter[
                frame_id
            ] += 1

            if accept:
                new_accepted += 1
            else:
                new_rejected += 1

        for violation in validation.get(
            "violations",
            [],
        ):

            violation_type = (
                violation.get(
                    "type",
                    "UNKNOWN",
                )
            )

            reject_counter[
                violation_type
            ] += 1

            language = violation.get(
                "language"
            )

            if language:

                language_counter[
                    language
                ] += 1

    # ========================================================
    # Output
    # ========================================================

    judged_file = (
        args.output_dir
        / "grammar_judged.jsonl"
    )

    accepted_file = (
        args.output_dir
        / "grammar_accepted.jsonl"
    )

    rejected_file = (
        args.output_dir
        / "grammar_rejected.jsonl"
    )

    summary_file = (
        args.output_dir
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

    total = len(
        judged
    )

    summary = {
        "validator_version":
            VERSION,

        "frozen_parent_validator":
            FROZEN_PARENT_VERSION,

        "total":
            total,

        "frozen_rows":
            len(frozen_rows),

        "new_rows":
            len(new_rows),

        "accepted":
            len(accepted),

        "rejected":
            len(rejected),

        "accept_rate":
            (
                len(accepted) / total
                if total
                else 0.0
            ),

        "new_accepted":
            new_accepted,

        "new_rejected":
            new_rejected,

        "new_frame_distribution":
            dict(
                new_frame_counter
            ),

        "reject_reasons":
            dict(
                reject_counter
            ),

        "rejected_by_language":
            dict(
                language_counter
            ),
    }

    write_json(
        summary_file,
        summary,
    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print("=" * 100)
    print(
        "V0.5.2 GRAMMAR HARD VALIDATION COMPLETE"
    )
    print("=" * 100)

    print(
        "Total:",
        total,
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
            f"{len(accepted) / total:.2%}"
            if total
            else "0.00%"
        ),
    )

    print()

    print("New V0.5.2:")

    print(
        "Accepted:",
        new_accepted,
    )

    print(
        "Rejected:",
        new_rejected,
    )

    print()

    print("Rejected by language:")
    print("-" * 80)

    if language_counter:

        for language, count in (
            language_counter.most_common()
        ):
            print(
                f"{language:<10}{count}"
            )

    else:

        print("None")

    print()

    print("Reject reasons:")
    print("-" * 80)

    if reject_counter:

        for name, count in (
            reject_counter.most_common()
        ):

            print(
                f"{name:<40}{count}"
            )

    else:

        print("None")

    print()

    print("Files:")
    print(judged_file)
    print(accepted_file)
    print(rejected_file)
    print(summary_file)

    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()