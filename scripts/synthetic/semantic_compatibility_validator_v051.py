from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATOR_VERSION = "0.5.1"

# ============================================================
# Resources
# ============================================================

V04_RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

V05_RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

BASE_CONCEPTS = (
    V04_RESOURCE_DIR
    / "concepts_v044.jsonl"
)

V05_COMPATIBILITY = (
    V05_RESOURCE_DIR
    / "semantic_compatibility_v05.json"
)

V05_ARGUMENTS = (
    V05_RESOURCE_DIR
    / "argument_realization_v051.json"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
    / "semantic_v051_raw.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
    / "01_compatibility"
)


# ============================================================
# V0.5.1 activation contract
# ============================================================

ACTIVE_NEW_FRAMES = {
    "SEE_OBJECT": "SEE",
    "TAKE_OBJECT": "TAKE",
}

ACTIVE_NEW_VERBS = {
    "SEE",
    "TAKE",
}

BLOCKED_NEW_FRAMES = {
    "LOSE_OBJECT",
    "NEED_OBJECT",
    "CALL_PERSON",
    "WAIT_PERSON",
    "WAIT_AT_PLACE",
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
    "LEAVE_PLACE",
    "RETURN_PLACE",
    "WHERE_OBJECT",
    "WHERE_PERSON",
}

BLOCKED_NEW_VERBS = {
    "LOSE",
    "NEED",
    "CALL",
    "WAIT",
    "GIVE",
    "BRING",
    "LEAVE",
    "RETURN",
}


# ============================================================
# IO
# ============================================================

def read_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return data


def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
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
                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL: "
                    f"{path}:{line_no}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise RuntimeError(
                    f"{path}:{line_no} "
                    f"is not a JSON object."
                )

            rows.append(
                row
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


# ============================================================
# Semantic-class index
# ============================================================

def build_class_indexes(
    compatibility: dict,
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
]:

    raw_classes = (
        compatibility.get(
            "semantic_classes",
            {}
        )
    )

    class_to_concepts: dict[
        str,
        set[str],
    ] = {}

    concept_to_classes: dict[
        str,
        set[str],
    ] = {}

    for class_id, members in (
        raw_classes.items()
    ):

        if not isinstance(
            members,
            list,
        ):
            raise RuntimeError(
                f"Semantic class {class_id} "
                f"must be a list."
            )

        member_set = set(
            members
        )

        class_to_concepts[
            class_id
        ] = member_set

        for concept_id in member_set:

            concept_to_classes.setdefault(
                concept_id,
                set(),
            ).add(
                class_id
            )

    return (
        class_to_concepts,
        concept_to_classes,
    )


# ============================================================
# New V0.5.1 validation
# ============================================================

def validate_new_row(
    row: dict,
    compatibility: dict,
    argument_resource: dict,
) -> tuple[
    bool,
    list[dict],
    list[dict],
]:

    violations: list[dict] = []
    warnings: list[dict] = []

    frame_id = row.get(
        "frame_id"
    )

    slots = row.get(
        "slots",
        {},
    )

    if not isinstance(
        slots,
        dict,
    ):
        return (
            False,
            [{
                "type":
                    "INVALID_SLOTS",

                "detail":
                    "slots must be an object",
            }],
            [],
        )

    verb_id = slots.get(
        "verb"
    )

    subject_id = slots.get(
        "subject"
    )

    object_id = slots.get(
        "object"
    )

    # ========================================================
    # Blocked capability leakage
    # ========================================================

    if frame_id in BLOCKED_NEW_FRAMES:

        violations.append({
            "type":
                "BLOCKED_FRAME",

            "frame_id":
                frame_id,
        })

    if verb_id in BLOCKED_NEW_VERBS:

        violations.append({
            "type":
                "BLOCKED_VERB",

            "verb":
                verb_id,
        })

    # ========================================================
    # Frame → fixed verb
    # ========================================================

    expected_verb = (
        ACTIVE_NEW_FRAMES.get(
            frame_id
        )
    )

    if expected_verb is None:

        violations.append({
            "type":
                "NEW_FRAME_NOT_ENABLED",

            "frame_id":
                frame_id,
        })

        return (
            False,
            violations,
            warnings,
        )

    if verb_id != expected_verb:

        violations.append({
            "type":
                "FRAME_VERB_MISMATCH",

            "frame_id":
                frame_id,

            "expected_verb":
                expected_verb,

            "actual_verb":
                verb_id,
        })

    # ========================================================
    # Subject
    # ========================================================

    subjects = (
        argument_resource.get(
            "subjects",
            {}
        )
    )

    if subject_id not in subjects:

        violations.append({
            "type":
                "UNKNOWN_SUBJECT",

            "subject":
                subject_id,
        })

    # ========================================================
    # Object
    # ========================================================

    objects = (
        argument_resource.get(
            "objects",
            {}
        )
    )

    if object_id not in objects:

        violations.append({
            "type":
                "UNKNOWN_OBJECT",

            "object":
                object_id,
        })

    # ========================================================
    # Verb semantic compatibility
    # ========================================================

    verb_rules = (
        compatibility.get(
            "verb_rules",
            {}
        )
    )

    verb_rule = verb_rules.get(
        verb_id
    )

    if not isinstance(
        verb_rule,
        dict,
    ):

        violations.append({
            "type":
                "UNKNOWN_VERB_RULE",

            "verb":
                verb_id,
        })

        return (
            len(violations) == 0,
            violations,
            warnings,
        )

    allowed_classes = set(
        verb_rule.get(
            "object_classes",
            []
        )
    )

    _, concept_to_classes = (
        build_class_indexes(
            compatibility
        )
    )

    object_classes = (
        concept_to_classes.get(
            object_id,
            set(),
        )
    )

    # --------------------------------------------------------
    # Object must belong to at least one class known by V0.5.
    # --------------------------------------------------------

    if not object_classes:

        violations.append({
            "type":
                "OBJECT_WITHOUT_SEMANTIC_CLASS",

            "verb":
                verb_id,

            "object":
                object_id,
        })

    # --------------------------------------------------------
    # At least one object class must be allowed by the verb.
    # --------------------------------------------------------

    elif not (
        object_classes
        & allowed_classes
    ):

        violations.append({
            "type":
                "OBJECT_CLASS_NOT_ALLOWED",

            "verb":
                verb_id,

            "object":
                object_id,

            "object_classes":
                sorted(
                    object_classes
                ),

            "allowed_classes":
                sorted(
                    allowed_classes
                ),
        })

    # ========================================================
    # Basic semantic features
    # ========================================================

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

    if tense not in {
        "present",
        "future",
    }:

        violations.append({
            "type":
                "UNSUPPORTED_TENSE",

            "tense":
                tense,
        })

    if polarity not in {
        "pos",
        "neg",
    }:

        violations.append({
            "type":
                "UNSUPPORTED_POLARITY",

            "polarity":
                polarity,
        })

    return (
        len(violations) == 0,
        violations,
        warnings,
    )


# ============================================================
# Frozen V0.4 validation
# ============================================================

def run_frozen_validator(
    rows: list[dict],
    work_dir: Path,
) -> list[dict]:

    if not rows:
        return []

    input_file = (
        work_dir
        / "_frozen_core_input.jsonl"
    )

    validator_output = (
        work_dir
        / "_frozen_v04_validator"
    )

    if validator_output.exists():

        shutil.rmtree(
            validator_output
        )

    write_jsonl(
        input_file,
        rows,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.synthetic.semantic_compatibility_validator_v04",

        "--input",
        str(input_file),

        "--output-dir",
        str(validator_output),

        "--concepts",
        str(BASE_CONCEPTS),
    ]

    print()
    print("=" * 100)
    print("RUNNING FROZEN V0.4 COMPATIBILITY VALIDATOR")
    print("=" * 100)

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Frozen V0.4 compatibility validator "
            f"failed with return code "
            f"{result.returncode}"
        )

    judged_file = (
        validator_output
        / "compatibility_judged.jsonl"
    )

    judged_rows = read_jsonl(
        judged_file
    )

    if len(
        judged_rows
    ) != len(
        rows
    ):

        raise RuntimeError(
            "Frozen validator output count mismatch: "
            f"input={len(rows)}, "
            f"judged={len(judged_rows)}"
        )

    return judged_rows


# ============================================================
# Helpers
# ============================================================

def is_v051_new_row(
    row: dict,
) -> bool:

    frame_id = row.get(
        "frame_id"
    )

    metadata = row.get(
        "metadata",
        {},
    )

    if frame_id in ACTIVE_NEW_FRAMES:
        return True

    if frame_id in BLOCKED_NEW_FRAMES:
        return True

    if isinstance(
        metadata,
        dict,
    ) and metadata.get(
        "generation_source"
    ) == "v051_new_frame":
        return True

    return False


def extract_accept(
    row: dict,
) -> bool:

    validation = row.get(
        "compatibility_validation",
        {}
    )

    if isinstance(
        validation,
        dict,
    ):

        if "accept" in validation:
            return bool(
                validation.get(
                    "accept"
                )
            )

    # Fallback for possible old validator naming.
    for key, value in row.items():

        if (
            "compat" in key.lower()
            and isinstance(
                value,
                dict,
            )
            and "accept" in value
        ):
            return bool(
                value.get(
                    "accept"
                )
            )

    raise RuntimeError(
        "Cannot locate compatibility accept field "
        f"for semantic_id={row.get('semantic_id')}"
    )


def extract_violations(
    row: dict,
) -> list[dict]:

    validation = row.get(
        "compatibility_validation",
        {}
    )

    if isinstance(
        validation,
        dict,
    ):

        violations = validation.get(
            "violations",
            []
        )

        if isinstance(
            violations,
            list,
        ):
            return violations

    for key, value in row.items():

        if (
            "compat" in key.lower()
            and isinstance(
                value,
                dict,
            )
        ):

            violations = value.get(
                "violations",
                []
            )

            if isinstance(
                violations,
                list,
            ):
                return violations

    return []


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Semantic Compatibility Validator "
            "V0.5.1"
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--compatibility",
        default=str(
            V05_COMPATIBILITY
        ),
    )

    parser.add_argument(
        "--arguments",
        default=str(
            V05_ARGUMENTS
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    compatibility = read_json(
        Path(
            args.compatibility
        )
    )

    arguments = read_json(
        Path(
            args.arguments
        )
    )

    rows = read_jsonl(
        input_path
    )

    # ========================================================
    # Split old / new
    # ========================================================

    frozen_rows = []
    new_rows = []

    for row in rows:

        if is_v051_new_row(
            row
        ):
            new_rows.append(
                row
            )

        else:
            frozen_rows.append(
                row
            )

    print("=" * 100)
    print("SEMANTIC COMPATIBILITY VALIDATOR V0.5.1")
    print("=" * 100)

    print(
        "Input:",
        input_path,
    )

    print(
        "Total:",
        len(rows),
    )

    print(
        "Frozen V0.4.4.1 rows:",
        len(frozen_rows),
    )

    print(
        "V0.5.1 new rows:",
        len(new_rows),
    )

    # ========================================================
    # Frozen validation
    # ========================================================

    frozen_judged = run_frozen_validator(
        frozen_rows,
        output_dir,
    )

    frozen_map = {
        row.get(
            "semantic_id"
        ): row
        for row in frozen_judged
    }

    # ========================================================
    # New validation
    # ========================================================

    new_map: dict[
        str,
        dict,
    ] = {}

    for row in new_rows:

        output_row = dict(
            row
        )

        accept, violations, warnings = (
            validate_new_row(
                row,
                compatibility,
                arguments,
            )
        )

        output_row[
            "compatibility_validation"
        ] = {
            "validator_version":
                VALIDATOR_VERSION,

            "accept":
                accept,

            "violations":
                violations,

            "warnings":
                warnings,
        }

        new_map[
            row.get(
                "semantic_id"
            )
        ] = output_row

    # ========================================================
    # Restore original ordering
    # ========================================================

    judged_rows = []

    for original_row in rows:

        semantic_id = (
            original_row.get(
                "semantic_id"
            )
        )

        if semantic_id in new_map:

            judged_rows.append(
                new_map[
                    semantic_id
                ]
            )

        elif semantic_id in frozen_map:

            judged_rows.append(
                frozen_map[
                    semantic_id
                ]
            )

        else:

            raise RuntimeError(
                "Missing judged result for "
                f"{semantic_id}"
            )

    # ========================================================
    # Accepted / rejected
    # ========================================================

    accepted_rows = []
    rejected_rows = []

    reject_counter = Counter()
    warning_rows = 0

    for row in judged_rows:

        accepted = extract_accept(
            row
        )

        violations = (
            extract_violations(
                row
            )
        )

        if accepted:
            accepted_rows.append(
                row
            )

        else:
            rejected_rows.append(
                row
            )

            for violation in violations:

                violation_type = (
                    violation.get(
                        "type",
                        "UNKNOWN",
                    )
                    if isinstance(
                        violation,
                        dict,
                    )
                    else "UNKNOWN"
                )

                reject_counter[
                    violation_type
                ] += 1

        validation = row.get(
            "compatibility_validation",
            {}
        )

        if isinstance(
            validation,
            dict,
        ) and validation.get(
            "warnings"
        ):
            warning_rows += 1

    # ========================================================
    # New distributions
    # ========================================================

    new_frame_counter = Counter(
        row.get(
            "frame_id"
        )
        for row in new_rows
    )

    new_verb_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "verb"
        )
        for row in new_rows
    )

    new_object_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "object"
        )
        for row in new_rows
    )

    # ========================================================
    # Save
    # ========================================================

    judged_file = (
        output_dir
        / "compatibility_judged.jsonl"
    )

    accepted_file = (
        output_dir
        / "compatibility_accepted.jsonl"
    )

    rejected_file = (
        output_dir
        / "compatibility_rejected.jsonl"
    )

    summary_file = (
        output_dir
        / "compatibility_summary.json"
    )

    write_jsonl(
        judged_file,
        judged_rows,
    )

    write_jsonl(
        accepted_file,
        accepted_rows,
    )

    write_jsonl(
        rejected_file,
        rejected_rows,
    )

    summary = {
        "validator_version":
            VALIDATOR_VERSION,

        "input":
            str(
                input_path
            ),

        "total":
            len(rows),

        "frozen_core_rows":
            len(frozen_rows),

        "new_rows":
            len(new_rows),

        "accepted":
            len(accepted_rows),

        "rejected":
            len(rejected_rows),

        "accept_rate":
            (
                len(accepted_rows)
                / len(rows)
                if rows
                else 0.0
            ),

        "warning_rows":
            warning_rows,

        "reject_reasons":
            dict(
                reject_counter
            ),

        "new_frame_distribution":
            dict(
                new_frame_counter
            ),

        "new_verb_distribution":
            dict(
                new_verb_counter
            ),

        "new_object_distribution":
            dict(
                new_object_counter
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
    print("V0.5.1 COMPATIBILITY VALIDATION COMPLETE")
    print("=" * 100)

    print(
        "Total:",
        len(rows),
    )

    print(
        "Accepted:",
        len(accepted_rows),
    )

    print(
        "Rejected:",
        len(rejected_rows),
    )

    print(
        "Accept rate:",
        (
            f"{len(accepted_rows) / len(rows):.2%}"
            if rows
            else "0.00%"
        ),
    )

    print(
        "Rows with warnings:",
        warning_rows,
    )

    print()

    print(
        "New frame distribution:"
    )

    for key, value in (
        new_frame_counter.items()
    ):
        print(
            f"{str(key):<25}{value}"
        )

    print()

    print(
        "New verb distribution:"
    )

    for key, value in (
        new_verb_counter.items()
    ):
        print(
            f"{str(key):<20}{value}"
        )

    print()

    print(
        "Reject reasons:"
    )

    if reject_counter:

        for key, value in (
            reject_counter.most_common()
        ):
            print(
                f"{key:<35}{value}"
            )

    else:
        print(
            "None"
        )

    print()

    print(
        "Files:"
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