from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_CONCEPTS = (
    DEFAULT_RESOURCE_DIR
    / "concepts_v04.jsonl"
)

DEFAULT_COMPATIBILITY = (
    DEFAULT_RESOURCE_DIR
    / "semantic_compatibility_v04.json"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v04_smoke_100"
    / "semantic_v04_raw.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v04_smoke_100"
    / "01_compatibility"
)


# ============================================================
# IO
# ============================================================

def read_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
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
            f"{path} root must be object."
        )

    return data


def read_jsonl(
    path: Path,
) -> list[dict]:

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

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSONL "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):

                raise RuntimeError(
                    f"{path}:{line_no} "
                    "must be JSON object."
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
# Resource helpers
# ============================================================

def semantic_classes_of(
    concept: dict,
) -> set[str]:

    classes = concept.get(
        "semantic_classes",
        [],
    )

    if isinstance(
        classes,
        str,
    ):

        classes = [
            classes
        ]

    if not isinstance(
        classes,
        list,
    ):

        return set()

    return {
        str(value)
        for value in classes
    }


def is_enabled(
    concept: dict,
) -> bool:

    meta = concept.get(
        "meta",
        {},
    )

    if not isinstance(
        meta,
        dict,
    ):

        return True

    return bool(
        meta.get(
            "enabled",
            True,
        )
    )


# ============================================================
# Validator
# ============================================================

class CompatibilityValidatorV04:

    def __init__(
        self,
        *,
        concepts_path: Path,
        compatibility_path: Path,
    ) -> None:

        concept_rows = read_jsonl(
            concepts_path
        )

        self.concepts = {
            row["id"]: row
            for row in concept_rows
        }

        self.compatibility = read_json(
            compatibility_path
        )

        semantic_classes = (
            self.compatibility.get(
                "semantic_classes",
                {},
            )
        )

        self.class_parent: dict[
            str,
            str | None,
        ] = {}

        if isinstance(
            semantic_classes,
            dict,
        ):

            for class_id, spec in (
                semantic_classes.items()
            ):

                parent = None

                if isinstance(
                    spec,
                    dict,
                ):

                    parent = spec.get(
                        "parent"
                    )

                self.class_parent[
                    class_id
                ] = parent

    # ========================================================
    # Semantic inheritance
    # ========================================================

    def expanded_classes(
        self,
        concept: dict,
    ) -> set[str]:

        direct = semantic_classes_of(
            concept
        )

        result = set(
            direct
        )

        for class_id in direct:

            current = class_id

            seen = set()

            while current:

                if current in seen:
                    break

                seen.add(
                    current
                )

                parent = (
                    self.class_parent.get(
                        current
                    )
                )

                if not parent:
                    break

                result.add(
                    parent
                )

                current = parent

        return result

    # ========================================================
    # Explicit forbidden
    # ========================================================

    def explicitly_forbidden(
        self,
        *,
        verb_id: str,
        role: str,
        concept_id: str,
    ) -> dict | None:

        rules = (
            self.compatibility.get(
                "explicit_forbidden",
                [],
            )
        )

        if not isinstance(
            rules,
            list,
        ):

            return None

        for rule in rules:

            if not isinstance(
                rule,
                dict,
            ):
                continue

            if rule.get(
                "verb"
            ) != verb_id:

                continue

            expected = rule.get(
                role
            )

            if (
                expected is None
                and role == "object"
            ):
                expected = rule.get(
                    "object"
                )

            if expected == concept_id:

                return rule

        return None

    # ========================================================
    # Role validation
    # ========================================================

    def validate_role(
        self,
        *,
        verb_id: str,
        role: str,
        concept_id: str,
    ) -> list[dict]:

        violations = []

        concept = self.concepts.get(
            concept_id
        )

        if concept is None:

            violations.append({
                "type":
                    "UNKNOWN_CONCEPT",

                "verb":
                    verb_id,

                "role":
                    role,

                "concept":
                    concept_id,
            })

            return violations

        if not is_enabled(
            concept
        ):

            violations.append({
                "type":
                    "DISABLED_CONCEPT",

                "verb":
                    verb_id,

                "role":
                    role,

                "concept":
                    concept_id,
            })

        forbidden_rule = (
            self.explicitly_forbidden(
                verb_id=verb_id,
                role=role,
                concept_id=concept_id,
            )
        )

        if forbidden_rule:

            violations.append({
                "type":
                    "EXPLICIT_FORBIDDEN",

                "verb":
                    verb_id,

                "role":
                    role,

                "concept":
                    concept_id,

                "reason":
                    forbidden_rule.get(
                        "reason",
                        "",
                    ),
            })

        verb_rules = (
            self.compatibility
            .get(
                "verb_rules",
                {},
            )
        )

        rule = (
            verb_rules
            .get(
                verb_id,
                {},
            )
            .get(
                role
            )
        )

        if rule is None:

            unknown_policy = (
                self.compatibility.get(
                    "unknown_policy",
                    "reject",
                )
            )

            if unknown_policy == "reject":

                violations.append({
                    "type":
                        "UNKNOWN_ROLE_RULE",

                    "verb":
                        verb_id,

                    "role":
                        role,

                    "concept":
                        concept_id,
                })

            return violations

        if not isinstance(
            rule,
            dict,
        ):

            violations.append({
                "type":
                    "INVALID_ROLE_RULE",

                "verb":
                    verb_id,

                "role":
                    role,
            })

            return violations

        allowed_classes = (
            rule.get(
                "allowed_classes",
                [],
            )
        )

        if allowed_classes:

            concept_classes = (
                self.expanded_classes(
                    concept
                )
            )

            overlap = (
                concept_classes
                & set(
                    allowed_classes
                )
            )

            if not overlap:

                violations.append({
                    "type":
                        "SEMANTIC_CLASS_MISMATCH",

                    "verb":
                        verb_id,

                    "role":
                        role,

                    "concept":
                        concept_id,

                    "concept_classes":
                        sorted(
                            concept_classes
                        ),

                    "allowed_classes":
                        sorted(
                            allowed_classes
                        ),
                })

        return violations

    # ========================================================
    # Whole sample
    # ========================================================

    def validate_sample(
        self,
        row: dict,
    ) -> dict:

        slots = row.get(
            "slots",
            {},
        )

        semantic_id = row.get(
            "semantic_id"
        )

        verb_id = slots.get(
            "verb"
        )

        violations = []

        # ----------------------------------------------------
        # Verbless frames such as WHERE_PLACE are valid.
        # ----------------------------------------------------

        if not verb_id:

            return {
                "semantic_id":
                    semantic_id,

                "accept":
                    True,

                "violations":
                    [],
            }

        verb = self.concepts.get(
            verb_id
        )

        if verb is None:

            return {
                "semantic_id":
                    semantic_id,

                "accept":
                    False,

                "violations": [
                    {
                        "type":
                            "UNKNOWN_VERB",

                        "verb":
                            verb_id,
                    }
                ],
            }

        if not is_enabled(
            verb
        ):

            violations.append({
                "type":
                    "DISABLED_VERB",

                "verb":
                    verb_id,
            })

        # ----------------------------------------------------
        # Semantic roles that compatibility rules may govern
        # ----------------------------------------------------

        for role in (
            "object",
            "destination",
            "source",
            "location",
            "recipient",
        ):

            concept_id = (
                slots.get(
                    role
                )
            )

            if not concept_id:
                continue

            violations.extend(
                self.validate_role(
                    verb_id=verb_id,
                    role=role,
                    concept_id=concept_id,
                )
            )

        # ----------------------------------------------------
        # Required compatibility roles
        # ----------------------------------------------------

        verb_rules = (
            self.compatibility
            .get(
                "verb_rules",
                {},
            )
            .get(
                verb_id,
                {},
            )
        )

        if isinstance(
            verb_rules,
            dict,
        ):

            for role, rule in (
                verb_rules.items()
            ):

                if not isinstance(
                    rule,
                    dict,
                ):
                    continue

                if (
                    rule.get(
                        "required",
                        False,
                    )
                    and not slots.get(
                        role
                    )
                ):

                    violations.append({
                        "type":
                            "MISSING_REQUIRED_ROLE",

                        "verb":
                            verb_id,

                        "role":
                            role,
                    })

        return {
            "semantic_id":
                semantic_id,

            "accept":
                not violations,

            "violations":
                violations,
        }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Semantic Compatibility "
            "Validator V0.4"
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
        "--concepts",
        default=str(
            DEFAULT_CONCEPTS
        ),
    )

    parser.add_argument(
        "--compatibility",
        default=str(
            DEFAULT_COMPATIBILITY
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

    rows = read_jsonl(
        input_path
    )

    validator = (
        CompatibilityValidatorV04(
            concepts_path=Path(
                args.concepts
            ),
            compatibility_path=Path(
                args.compatibility
            ),
        )
    )

    judged_rows = []

    accepted_rows = []

    rejected_rows = []

    violation_counter = Counter()

    verb_counter = Counter()

    warning_count = 0

    for row in rows:

        result = (
            validator.validate_sample(
                row
            )
        )

        output_row = dict(
            row
        )

        output_row[
            "compatibility_validation"
        ] = result

        judged_rows.append(
            output_row
        )

        verb_id = (
            row.get(
                "slots",
                {},
            ).get(
                "verb"
            )
        )

        if verb_id:

            verb_counter[
                verb_id
            ] += 1

        if result[
            "accept"
        ]:

            accepted_rows.append(
                output_row
            )

        else:

            rejected_rows.append(
                output_row
            )

            for violation in (
                result[
                    "violations"
                ]
            ):

                violation_counter[
                    violation.get(
                        "type",
                        "UNKNOWN",
                    )
                ] += 1

    total = len(
        rows
    )

    accepted = len(
        accepted_rows
    )

    rejected = len(
        rejected_rows
    )

    summary = {
        "validator_version":
            "0.4",

        "input":
            str(
                input_path
            ),

        "total":
            total,

        "accepted":
            accepted,

        "rejected":
            rejected,

        "accept_rate":
            (
                accepted / total
                if total
                else 0.0
            ),

        "rows_with_warnings":
            warning_count,

        "warning_rate":
            (
                warning_count / total
                if total
                else 0.0
            ),

        "verb_distribution":
            dict(
                verb_counter.most_common()
            ),

        "violations":
            dict(
                violation_counter.most_common()
            ),
    }

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

    write_json(
        summary_file,
        summary,
    )

    print(
        "=" * 90
    )

    print(
        "SEMANTIC COMPATIBILITY VALIDATOR V0.4"
    )

    print(
        "=" * 90
    )

    print(
        "Input:",
        input_path,
    )

    print(
        "Total:",
        total,
    )

    print(
        "Accepted:",
        accepted,
    )

    print(
        "Rejected:",
        rejected,
    )

    print(
        "Accept rate:",
        f"{summary['accept_rate']:.2%}",
    )

    print(
        "Rows with warnings:",
        warning_count,
    )

    print(
        "Warning rate:",
        f"{summary['warning_rate']:.2%}",
    )

    print()

    print(
        "Verb distribution:"
    )

    print(
        "-" * 70
    )

    if verb_counter:

        for verb_id, count in (
            verb_counter.most_common()
        ):

            print(
                f"{verb_id:<20}"
                f"{count}"
            )

    else:

        print(
            "None"
        )

    print()

    print(
        "Violations:"
    )

    print(
        "-" * 70
    )

    if violation_counter:

        for name, count in (
            violation_counter.most_common()
        ):

            print(
                f"{name:<35}"
                f"{count}"
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