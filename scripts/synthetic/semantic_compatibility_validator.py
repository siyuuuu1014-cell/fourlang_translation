from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "renderer_v2"
    / "semantic_v02.jsonl"
)


DEFAULT_RESOURCE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "semantic_compatibility.json"
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "renderer_v2"
    / "semantic_compatibility"
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


def load_resource(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"Compatibility resource not found:\n"
            f"{path}"
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

        raise ValueError(
            "semantic_compatibility.json "
            "must contain a JSON object."
        )

    return data


# ============================================================
# Normalization
# ============================================================

def normalize_concept_id(
    value: Any,
) -> str:

    if value is None:
        return ""

    value = str(
        value
    ).strip().upper()

    # Normalize common separators.
    value = re.sub(
        r"[\s\-]+",
        "_",
        value,
    )

    return value


def concept_matches_pattern(
    concept_id: str,
    pattern: str,
) -> bool:

    """
    Supports concept IDs such as:

        FOOD
        OBJECT_FOOD
        ITEM_FOOD

    against the configured pattern:

        FOOD

    We intentionally avoid unrestricted substring matching
    because e.g. CAR could accidentally match CARD.
    """

    concept_id = normalize_concept_id(
        concept_id
    )

    pattern = normalize_concept_id(
        pattern
    )

    if not concept_id or not pattern:
        return False

    if concept_id == pattern:
        return True

    concept_tokens = [
        token
        for token in concept_id.split("_")
        if token
    ]

    return pattern in concept_tokens


# ============================================================
# Concept classification
# ============================================================

def classify_concept(
    concept_id: str | None,
    resource: dict,
) -> list[str]:

    """
    A concept can belong to multiple semantic classes.
    """

    if not concept_id:
        return []

    classes = []

    class_map = resource.get(
        "concept_classes",
        {},
    )

    for class_name, patterns in (
        class_map.items()
    ):

        if not isinstance(
            patterns,
            list,
        ):
            continue

        for pattern in patterns:

            if concept_matches_pattern(
                concept_id,
                str(pattern),
            ):

                classes.append(
                    class_name
                )

                break

    return sorted(
        set(classes)
    )


# ============================================================
# Verb rule matching
# ============================================================

def find_verb_rule(
    verb_id: str | None,
    resource: dict,
) -> tuple[str | None, dict | None]:

    if not verb_id:
        return None, None

    rules = resource.get(
        "verb_rules",
        {},
    )

    for rule_verb, rule in (
        rules.items()
    ):

        if concept_matches_pattern(
            verb_id,
            rule_verb,
        ):

            return (
                rule_verb,
                rule,
            )

    return None, None


# ============================================================
# Explicit forbidden pairs
# ============================================================

def find_explicit_forbidden_pair(
    verb_id: str | None,
    object_id: str | None,
    resource: dict,
) -> dict | None:

    if not verb_id or not object_id:
        return None

    pairs = resource.get(
        "explicit_forbidden_pairs",
        [],
    )

    for pair in pairs:

        expected_verb = pair.get(
            "verb"
        )

        expected_object = pair.get(
            "object"
        )

        if (
            expected_verb
            and
            expected_object
            and
            concept_matches_pattern(
                verb_id,
                expected_verb,
            )
            and
            concept_matches_pattern(
                object_id,
                expected_object,
            )
        ):

            return pair

    return None


# ============================================================
# Slot helpers
# ============================================================

def get_slots(
    row: dict,
) -> dict:

    slots = row.get(
        "slots",
        {},
    )

    if not isinstance(
        slots,
        dict,
    ):

        return {}

    return slots


def get_record_id(
    row: dict,
) -> str:

    return str(
        row.get("semantic_id")
        or row.get("calibration_id")
        or row.get("id")
        or "UNKNOWN"
    )


# ============================================================
# Compatibility check
# ============================================================

def validate_row(
    row: dict,
    resource: dict,
) -> dict:

    slots = get_slots(
        row
    )

    verb_id = slots.get(
        "verb"
    )

    object_id = slots.get(
        "object"
    )

    destination_id = slots.get(
        "destination"
    )


    normalized_verb = (
        normalize_concept_id(
            verb_id
        )
    )

    normalized_object = (
        normalize_concept_id(
            object_id
        )
    )

    normalized_destination = (
        normalize_concept_id(
            destination_id
        )
    )


    object_classes = (
        classify_concept(
            object_id,
            resource,
        )
    )

    destination_classes = (
        classify_concept(
            destination_id,
            resource,
        )
    )


    violations = []

    warnings = []


    # ========================================================
    # Explicit forbidden pair
    # ========================================================

    forbidden = (
        find_explicit_forbidden_pair(
            verb_id,
            object_id,
            resource,
        )
    )


    if forbidden:

        violations.append({
            "type":
                "EXPLICIT_FORBIDDEN_PAIR",

            "verb":
                normalized_verb,

            "object":
                normalized_object,

            "reason":
                forbidden.get(
                    "reason",
                    "Explicitly forbidden pair.",
                ),
        })


    # ========================================================
    # Verb rule
    # ========================================================

    (
        rule_verb,
        rule,
    ) = find_verb_rule(
        verb_id,
        resource,
    )

    # ========================================================
    # Frames without a verb slot
    # ========================================================

    if not normalized_verb:

        # WHERE_PLACE and similar frames may legitimately
        # contain no verb concept.
        #
        # This is not an UNKNOWN_VERB_RULE.
        rule_verb = None
        rule = None


    elif rule is None:

        warnings.append({
            "type":
                "UNKNOWN_VERB_RULE",

            "verb":
                normalized_verb,
        })

    else:

        required_slot = rule.get(
            "required_slot"
        )


        # ----------------------------------------------------
        # Required object
        # ----------------------------------------------------

        if (
            required_slot == "object"
            and
            not object_id
        ):

            violations.append({
                "type":
                    "MISSING_REQUIRED_OBJECT",

                "verb":
                    normalized_verb,
            })


        # ----------------------------------------------------
        # Required destination
        # ----------------------------------------------------

        if (
            required_slot == "destination"
            and
            not destination_id
        ):

            violations.append({
                "type":
                    "MISSING_REQUIRED_DESTINATION",

                "verb":
                    normalized_verb,
            })


        # ----------------------------------------------------
        # Object compatibility
        # ----------------------------------------------------

        allowed_object_classes = (
            rule.get(
                "allowed_object_classes"
            )
        )


        if (
            object_id
            and
            allowed_object_classes
        ):

            if not object_classes:

                warnings.append({
                    "type":
                        "UNKNOWN_OBJECT_CLASS",

                    "verb":
                        normalized_verb,

                    "object":
                        normalized_object,
                })

            else:

                compatible = any(
                    cls
                    in allowed_object_classes

                    for cls
                    in object_classes
                )


                if not compatible:

                    violations.append({
                        "type":
                            "OBJECT_CLASS_MISMATCH",

                        "verb":
                            normalized_verb,

                        "object":
                            normalized_object,

                        "object_classes":
                            object_classes,

                        "allowed_classes":
                            allowed_object_classes,
                    })


        # ----------------------------------------------------
        # Destination compatibility
        # ----------------------------------------------------

        allowed_destination_classes = (
            rule.get(
                "allowed_destination_classes"
            )
        )


        if (
            destination_id
            and
            allowed_destination_classes
        ):

            if not destination_classes:

                warnings.append({
                    "type":
                        "UNKNOWN_DESTINATION_CLASS",

                    "verb":
                        normalized_verb,

                    "destination":
                        normalized_destination,
                })

            else:

                compatible = any(
                    cls
                    in allowed_destination_classes

                    for cls
                    in destination_classes
                )


                if not compatible:

                    violations.append({
                        "type":
                            "DESTINATION_CLASS_MISMATCH",

                        "verb":
                            normalized_verb,

                        "destination":
                            normalized_destination,

                        "destination_classes":
                            destination_classes,

                        "allowed_classes":
                            allowed_destination_classes,
                    })


    # ========================================================
    # Unknown policy
    # ========================================================

    unknown_policy = resource.get(
        "unknown_concept_policy",
        "warn",
    )


    if unknown_policy == "reject":

        for warning in warnings:

            if warning.get(
                "type"
            ) in {
                "UNKNOWN_VERB_RULE",
                "UNKNOWN_OBJECT_CLASS",
                "UNKNOWN_DESTINATION_CLASS",
            }:

                violations.append({
                    "type":
                        "UNKNOWN_CONCEPT_REJECTED",

                    "source_warning":
                        warning,
                })


    accept = (
        len(violations)
        == 0
    )


    return {
        "accept":
            accept,

        "verb":
            normalized_verb,

        "object":
            normalized_object
            or None,

        "destination":
            normalized_destination
            or None,

        "object_classes":
            object_classes,

        "destination_classes":
            destination_classes,

        "matched_rule":
            rule_verb,

        "violations":
            violations,

        "warnings":
            warnings,
    }


# ============================================================
# Summary helpers
# ============================================================

def violation_key(
    violation: dict,
) -> str:

    violation_type = violation.get(
        "type",
        "UNKNOWN",
    )

    verb = violation.get(
        "verb",
        "",
    )

    obj = violation.get(
        "object",
        "",
    )

    destination = violation.get(
        "destination",
        "",
    )

    return (
        f"{violation_type}:"
        f"{verb}:"
        f"{obj}:"
        f"{destination}"
    )


def warning_key(
    warning: dict,
) -> str:

    warning_type = warning.get(
        "type",
        "UNKNOWN",
    )

    verb = warning.get(
        "verb",
        "",
    )

    obj = warning.get(
        "object",
        "",
    )

    destination = warning.get(
        "destination",
        "",
    )

    return (
        f"{warning_type}:"
        f"{verb}:"
        f"{obj}:"
        f"{destination}"
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
        "--resource",
        type=str,
        default=str(
            DEFAULT_RESOURCE
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

    resource_file = Path(
        args.resource
    )

    output_dir = Path(
        args.output_dir
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{input_file}"
        )


    resource = load_resource(
        resource_file
    )


    rows = read_jsonl(
        input_file
    )


    judged_rows = []

    accepted_rows = []

    rejected_rows = []

    warning_rows = []


    violation_counter = Counter()

    warning_counter = Counter()

    verb_counter = Counter()

    rejected_pair_counter = Counter()

    unknown_object_counter = Counter()

    unknown_destination_counter = Counter()

    unknown_verb_counter = Counter()


    # ========================================================
    # Validate
    # ========================================================

    for source_row in rows:

        row = dict(
            source_row
        )


        result = validate_row(
            row,
            resource,
        )


        row[
            "semantic_compatibility"
        ] = result


        row[
            "semantic_compatibility_accept"
        ] = result[
            "accept"
        ]


        judged_rows.append(
            row
        )


        verb_counter[
            result.get(
                "verb",
                "UNKNOWN",
            )
        ] += 1


        for violation in result[
            "violations"
        ]:

            violation_counter[
                violation_key(
                    violation
                )
            ] += 1


        for warning in result[
            "warnings"
        ]:

            warning_counter[
                warning_key(
                    warning
                )
            ] += 1


            warning_type = warning.get(
                "type"
            )


            if (
                warning_type
                == "UNKNOWN_OBJECT_CLASS"
            ):

                unknown_object_counter[
                    warning.get(
                        "object",
                        "UNKNOWN",
                    )
                ] += 1


            elif (
                warning_type
                ==
                "UNKNOWN_DESTINATION_CLASS"
            ):

                unknown_destination_counter[
                    warning.get(
                        "destination",
                        "UNKNOWN",
                    )
                ] += 1


            elif (
                warning_type
                ==
                "UNKNOWN_VERB_RULE"
            ):

                unknown_verb_counter[
                    warning.get(
                        "verb",
                        "UNKNOWN",
                    )
                ] += 1


        if result[
            "warnings"
        ]:

            warning_rows.append(
                row
            )


        if result[
            "accept"
        ]:

            accepted_rows.append(
                row
            )

        else:

            rejected_rows.append(
                row
            )


            pair = (
                result.get(
                    "verb"
                ),
                result.get(
                    "object"
                ),
                result.get(
                    "destination"
                ),
            )


            rejected_pair_counter[
                str(pair)
            ] += 1


    # ========================================================
    # Save
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


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


    warnings_file = (
        output_dir
        / "compatibility_warnings.jsonl"
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


    write_jsonl(
        warnings_file,
        warning_rows,
    )


    total = len(
        rows
    )


    summary = {
        "version":
            "semantic_compatibility_v1",

        "input_file":
            str(input_file),

        "resource_file":
            str(resource_file),

        "total":
            total,

        "accepted":
            len(accepted_rows),

        "rejected":
            len(rejected_rows),

        "accept_rate":
            (
                len(accepted_rows)
                / total
                if total
                else 0
            ),

        "rows_with_warnings":
            len(warning_rows),

        "warning_rate":
            (
                len(warning_rows)
                / total
                if total
                else 0
            ),

        "verb_distribution":
            dict(
                verb_counter
                .most_common()
            ),

        "violation_counts":
            dict(
                violation_counter
                .most_common()
            ),

        "warning_counts":
            dict(
                warning_counter
                .most_common()
            ),

        "rejected_pairs":
            dict(
                rejected_pair_counter
                .most_common()
            ),

        "unknown_objects":
            dict(
                unknown_object_counter
                .most_common()
            ),

        "unknown_destinations":
            dict(
                unknown_destination_counter
                .most_common()
            ),

        "unknown_verbs":
            dict(
                unknown_verb_counter
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


    # ========================================================
    # Console
    # ========================================================

    print("=" * 100)
    print("SEMANTIC COMPATIBILITY VALIDATOR V1")
    print("=" * 100)


    print(
        "Input:",
        input_file
    )


    print(
        "Resource:",
        resource_file
    )


    print(
        "Total:",
        total
    )


    print(
        "Accepted:",
        len(accepted_rows)
    )


    print(
        "Rejected:",
        len(rejected_rows)
    )


    print(
        "Accept rate:",
        (
            f"{len(accepted_rows)/total:.2%}"
            if total
            else "0%"
        )
    )


    print(
        "Rows with warnings:",
        len(warning_rows)
    )


    print(
        "Warning rate:",
        (
            f"{len(warning_rows)/total:.2%}"
            if total
            else "0%"
        )
    )


    # --------------------------------------------------------
    # Verb distribution
    # --------------------------------------------------------

    print(
        "\nVerb distribution:"
    )

    print("-" * 100)


    for verb, count in (
        verb_counter
        .most_common()
    ):

        print(
            f"{verb:<40}"
            f"{count}"
        )


    # --------------------------------------------------------
    # Violations
    # --------------------------------------------------------

    print(
        "\nViolations:"
    )

    print("-" * 100)


    if violation_counter:

        for key, count in (
            violation_counter
            .most_common()
        ):

            print(
                f"{key:<85}"
                f"{count}"
            )

    else:

        print(
            "None"
        )


    # --------------------------------------------------------
    # Rejected combinations
    # --------------------------------------------------------

    print(
        "\nRejected combinations:"
    )

    print("-" * 100)


    if rejected_pair_counter:

        for pair, count in (
            rejected_pair_counter
            .most_common()
        ):

            print(
                f"{pair:<85}"
                f"{count}"
            )

    else:

        print(
            "None"
        )


    # --------------------------------------------------------
    # Unknown resources
    # --------------------------------------------------------

    print(
        "\nUnknown object concepts:"
    )

    print("-" * 100)


    if unknown_object_counter:

        for concept, count in (
            unknown_object_counter
            .most_common()
        ):

            print(
                f"{concept:<50}"
                f"{count}"
            )

    else:

        print(
            "None"
        )


    print(
        "\nUnknown destination concepts:"
    )

    print("-" * 100)


    if unknown_destination_counter:

        for concept, count in (
            unknown_destination_counter
            .most_common()
        ):

            print(
                f"{concept:<50}"
                f"{count}"
            )

    else:

        print(
            "None"
        )


    print(
        "\nUnknown verb rules:"
    )

    print("-" * 100)


    if unknown_verb_counter:

        for concept, count in (
            unknown_verb_counter
            .most_common()
        ):

            print(
                f"{concept:<50}"
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
        warnings_file
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    main()