from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.2"
FROZEN_PARENT_VERSION = "0.5.1"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

ACTIVATION_FILE = (
    RESOURCE_DIR
    / "activation_v052.json"
)

ARGUMENT_RESOURCE = (
    RESOURCE_DIR
    / "argument_realization_v052.json"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v052_overlap_fix_200"
    / "semantic_v052_raw.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v052_overlap_fix_200"
    / "01_compatibility"
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

BLOCKED_VERBS = {
    "GIVE",
    "BRING",
    "LOSE",
    "RETURN",
    "NEED",
    "LEAVE",
}

BLOCKED_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
    "LOSE_OBJECT",
    "RETURN_PLACE",
    "NEED_OBJECT",
    "LEAVE_PLACE",
}


# =============================================================================
# Participant compatibility
#
# Exact reflexive is disabled.
#
# Also disabled:
#
#   PERSON_I  -> PERSON_WE
#   PERSON_WE -> PERSON_I
#
# because the speaker is a member of the WE/US participant set.
# =============================================================================

PARTICIPANT_OVERLAP_PAIRS = {
    ("PERSON_I", "PERSON_WE"),
    ("PERSON_WE", "PERSON_I"),
}


def person_target_allowed(
    subject_id: str | None,
    person_id: str | None,
) -> bool:

    if not subject_id or not person_id:
        return False

    if subject_id == person_id:
        return False

    if (
        subject_id,
        person_id,
    ) in PARTICIPANT_OVERLAP_PAIRS:
        return False

    return True


# =============================================================================
# IO
# =============================================================================

def read_json(path: Path) -> dict:

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
            f"Expected JSON object: {path}"
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

        for line_number, line in enumerate(
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
                    f"Invalid JSONL at "
                    f"{path}:{line_number}"
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


# =============================================================================
# Resources
# =============================================================================

def load_resources():

    activation = read_json(
        ACTIVATION_FILE
    )

    arguments = read_json(
        ARGUMENT_RESOURCE
    )

    if (
        activation.get(
            "activation_version"
        )
        != VERSION
    ):

        raise RuntimeError(
            "Activation version mismatch."
        )

    if (
        activation.get(
            "status"
        )
        != "validated"
    ):

        raise RuntimeError(
            "Activation has not passed validation."
        )

    enabled_frames = set(
        activation.get(
            "enabled_frames",
            [],
        )
    )

    enabled_verbs = set(
        activation.get(
            "enabled_verbs",
            [],
        )
    )

    if enabled_frames != NEW_FRAMES:

        raise RuntimeError(
            "Enabled frame whitelist mismatch. "
            f"expected={sorted(NEW_FRAMES)}, "
            f"actual={sorted(enabled_frames)}"
        )

    if enabled_verbs != NEW_VERBS:

        raise RuntimeError(
            "Enabled verb whitelist mismatch. "
            f"expected={sorted(NEW_VERBS)}, "
            f"actual={sorted(enabled_verbs)}"
        )

    person_targets = arguments.get(
        "person_targets",
        {},
    )

    place_targets = arguments.get(
        "place_targets",
        {},
    )

    frame_contracts = arguments.get(
        "frame_contracts_v052",
        {},
    )

    if not isinstance(
        person_targets,
        dict,
    ) or not person_targets:

        raise RuntimeError(
            "person_targets resource is empty."
        )

    if not isinstance(
        place_targets,
        dict,
    ) or not place_targets:

        raise RuntimeError(
            "place_targets resource is empty."
        )

    return (
        activation,
        person_targets,
        place_targets,
        frame_contracts,
    )


# =============================================================================
# Helpers
# =============================================================================

def add_violation(
    violations: list[dict],
    violation_type: str,
    **kwargs,
) -> None:

    item = {
        "type":
            violation_type,
    }

    item.update(
        kwargs
    )

    violations.append(
        item
    )


def validate_texts(
    row: dict,
    violations: list[dict],
) -> None:

    texts = row.get(
        "texts"
    )

    if not isinstance(
        texts,
        dict,
    ):

        add_violation(
            violations,
            "INVALID_TEXT_CONTAINER",
        )

        return

    for language in (
        "zh",
        "en",
        "ru",
        "uz",
    ):

        text = texts.get(
            language
        )

        if (
            not isinstance(
                text,
                str,
            )
            or not text.strip()
        ):

            add_violation(
                violations,
                "MISSING_LANGUAGE_TEXT",
                language=language,
            )


def validate_person_relation(
    *,
    subject_id: str | None,
    person_id: str | None,
    violations: list[dict],
) -> None:

    if not subject_id:
        return

    if not person_id:
        return

    # Exact reflexive.
    if subject_id == person_id:

        add_violation(
            violations,
            "REFLEXIVE_SEMANTICS_NOT_ENABLED",
            subject=subject_id,
            person=person_id,
        )

        return

    # Speaker / group participant overlap.
    if (
        subject_id,
        person_id,
    ) in PARTICIPANT_OVERLAP_PAIRS:

        add_violation(
            violations,
            "PARTICIPANT_OVERLAP_NOT_ENABLED",
            subject=subject_id,
            person=person_id,
        )


# =============================================================================
# Frozen parent compatibility
#
# V0.5.2 does NOT redefine the semantic behavior of V0.5.1.
# =============================================================================

def validate_frozen_row(
    row: dict,
) -> tuple[
    list[dict],
    list[dict],
]:

    violations = []
    warnings = []

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

        add_violation(
            violations,
            "INVALID_SLOTS",
        )

        return (
            violations,
            warnings,
        )

    verb_id = slots.get(
        "verb"
    )

    if verb_id in BLOCKED_VERBS:

        add_violation(
            violations,
            "BLOCKED_VERB_LEAKAGE",
            verb=verb_id,
        )

    if frame_id in BLOCKED_FRAMES:

        add_violation(
            violations,
            "BLOCKED_FRAME_LEAKAGE",
            frame=frame_id,
        )

    if (
        verb_id in NEW_VERBS
        or frame_id in NEW_FRAMES
    ):

        add_violation(
            violations,
            "V052_CAPABILITY_IN_FROZEN_PARENT",
            frame=frame_id,
            verb=verb_id,
        )

    validate_texts(
        row,
        violations,
    )

    return (
        violations,
        warnings,
    )


# =============================================================================
# CALL_PERSON
# =============================================================================

def validate_call_person(
    row: dict,
    *,
    person_targets: dict,
) -> tuple[
    list[dict],
    list[dict],
]:

    violations = []
    warnings = []

    slots = row.get(
        "slots",
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

        return (
            violations,
            warnings,
        )

    if slots.get(
        "verb"
    ) != "CALL":

        add_violation(
            violations,
            "FRAME_VERB_MISMATCH",
            expected="CALL",
            actual=slots.get("verb"),
        )

    subject_id = slots.get(
        "subject"
    )

    person_id = slots.get(
        "person"
    )

    if not subject_id:

        add_violation(
            violations,
            "MISSING_SUBJECT",
        )

    if not person_id:

        add_violation(
            violations,
            "MISSING_PERSON",
        )

    elif person_id not in person_targets:

        add_violation(
            violations,
            "UNKNOWN_PERSON_TARGET",
            person=person_id,
        )

    validate_person_relation(
        subject_id=subject_id,
        person_id=person_id,
        violations=violations,
    )

    forbidden = {
        "object",
        "destination",
        "place",
        "time",
        "day",
    }

    for slot_name in forbidden:

        if slots.get(
            slot_name
        ) is not None:

            add_violation(
                violations,
                "FORBIDDEN_SLOT",
                slot=slot_name,
            )

    validate_texts(
        row,
        violations,
    )

    return (
        violations,
        warnings,
    )


# =============================================================================
# WAIT_PERSON
# =============================================================================

def validate_wait_person(
    row: dict,
    *,
    person_targets: dict,
) -> tuple[
    list[dict],
    list[dict],
]:

    violations = []
    warnings = []

    slots = row.get(
        "slots",
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

        return (
            violations,
            warnings,
        )

    if slots.get(
        "verb"
    ) != "WAIT":

        add_violation(
            violations,
            "FRAME_VERB_MISMATCH",
            expected="WAIT",
            actual=slots.get("verb"),
        )

    subject_id = slots.get(
        "subject"
    )

    person_id = slots.get(
        "person"
    )

    if not subject_id:

        add_violation(
            violations,
            "MISSING_SUBJECT",
        )

    if not person_id:

        add_violation(
            violations,
            "MISSING_PERSON",
        )

    elif person_id not in person_targets:

        add_violation(
            violations,
            "UNKNOWN_PERSON_TARGET",
            person=person_id,
        )

    validate_person_relation(
        subject_id=subject_id,
        person_id=person_id,
        violations=violations,
    )

    forbidden = {
        "object",
        "destination",
        "place",
        "time",
        "day",
    }

    for slot_name in forbidden:

        if slots.get(
            slot_name
        ) is not None:

            add_violation(
                violations,
                "FORBIDDEN_SLOT",
                slot=slot_name,
            )

    validate_texts(
        row,
        violations,
    )

    return (
        violations,
        warnings,
    )


# =============================================================================
# WAIT_AT_PLACE
# =============================================================================

def validate_wait_at_place(
    row: dict,
    *,
    place_targets: dict,
) -> tuple[
    list[dict],
    list[dict],
]:

    violations = []
    warnings = []

    slots = row.get(
        "slots",
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

        return (
            violations,
            warnings,
        )

    if slots.get(
        "verb"
    ) != "WAIT":

        add_violation(
            violations,
            "FRAME_VERB_MISMATCH",
            expected="WAIT",
            actual=slots.get("verb"),
        )

    subject_id = slots.get(
        "subject"
    )

    place_id = slots.get(
        "place"
    )

    if not subject_id:

        add_violation(
            violations,
            "MISSING_SUBJECT",
        )

    if not place_id:

        add_violation(
            violations,
            "MISSING_PLACE",
        )

    elif place_id not in place_targets:

        add_violation(
            violations,
            "UNKNOWN_PLACE_TARGET",
            place=place_id,
        )

    forbidden = {
        "object",
        "person",
        "destination",
        "time",
        "day",
    }

    for slot_name in forbidden:

        if slots.get(
            slot_name
        ) is not None:

            add_violation(
                violations,
                "FORBIDDEN_SLOT",
                slot=slot_name,
            )

    validate_texts(
        row,
        violations,
    )

    return (
        violations,
        warnings,
    )


# =============================================================================
# Single row
# =============================================================================

def validate_row(
    row: dict,
    *,
    person_targets: dict,
    place_targets: dict,
) -> tuple[
    list[dict],
    list[dict],
]:

    frame_id = row.get(
        "frame_id"
    )

    metadata = row.get(
        "metadata",
        {},
    )

    generation_source = (
        metadata.get(
            "generation_source"
        )
        if isinstance(
            metadata,
            dict,
        )
        else None
    )

    slots = row.get(
        "slots",
        {},
    )

    verb_id = (
        slots.get("verb")
        if isinstance(
            slots,
            dict,
        )
        else None
    )

    # -------------------------------------------------------------------------
    # Block future capabilities everywhere
    # -------------------------------------------------------------------------

    if (
        verb_id in BLOCKED_VERBS
        or frame_id in BLOCKED_FRAMES
    ):

        violations = []

        if verb_id in BLOCKED_VERBS:

            add_violation(
                violations,
                "BLOCKED_VERB_LEAKAGE",
                verb=verb_id,
            )

        if frame_id in BLOCKED_FRAMES:

            add_violation(
                violations,
                "BLOCKED_FRAME_LEAKAGE",
                frame=frame_id,
            )

        return (
            violations,
            [],
        )

    # -------------------------------------------------------------------------
    # V0.5.2 new frames
    # -------------------------------------------------------------------------

    if frame_id == "CALL_PERSON":

        return validate_call_person(
            row,
            person_targets=person_targets,
        )

    if frame_id == "WAIT_PERSON":

        return validate_wait_person(
            row,
            person_targets=person_targets,
        )

    if frame_id == "WAIT_AT_PLACE":

        return validate_wait_at_place(
            row,
            place_targets=place_targets,
        )

    # -------------------------------------------------------------------------
    # Frozen parent
    # -------------------------------------------------------------------------

    if (
        generation_source
        == "frozen_parent_v051"
    ):

        return validate_frozen_row(
            row
        )

    return validate_frozen_row(
        row
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Semantic Compatibility "
            "Validator V0.5.2"
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


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_args()

    (
        activation,
        person_targets,
        place_targets,
        frame_contracts,
    ) = load_resources()

    rows = read_jsonl(
        args.input
    )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    judged = []
    accepted = []
    rejected = []

    violation_counter = Counter()
    warning_counter = Counter()

    new_frame_counter = Counter()
    new_verb_counter = Counter()

    rows_with_warnings = 0

    for row in rows:

        output_row = copy.deepcopy(
            row
        )

        (
            violations,
            warnings,
        ) = validate_row(
            row,
            person_targets=person_targets,
            place_targets=place_targets,
        )

        accept = (
            len(violations) == 0
        )

        validation = {
            "validator_version":
                VERSION,

            "accept":
                accept,

            "violations":
                violations,

            "warnings":
                warnings,
        }

        output_row[
            "compatibility_validation"
        ] = validation

        judged.append(
            output_row
        )

        if accept:
            accepted.append(
                output_row
            )

        else:
            rejected.append(
                output_row
            )

        if warnings:
            rows_with_warnings += 1

        for violation in violations:

            violation_counter[
                violation.get(
                    "type",
                    "UNKNOWN",
                )
            ] += 1

        for warning in warnings:

            warning_counter[
                warning.get(
                    "type",
                    "UNKNOWN",
                )
            ] += 1

        frame_id = row.get(
            "frame_id"
        )

        if frame_id in NEW_FRAMES:

            new_frame_counter[
                frame_id
            ] += 1

            new_verb_counter[
                row.get(
                    "slots",
                    {},
                ).get(
                    "verb"
                )
            ] += 1

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
        rows
    )

    summary = {
        "validator_version":
            VERSION,

        "input":
            str(
                args.input
            ),

        "total":
            total,

        "accepted":
            len(
                accepted
            ),

        "rejected":
            len(
                rejected
            ),

        "accept_rate":
            (
                len(accepted)
                / total
                if total
                else 0.0
            ),

        "rows_with_warnings":
            rows_with_warnings,

        "warning_rate":
            (
                rows_with_warnings
                / total
                if total
                else 0.0
            ),

        "new_frame_distribution":
            dict(
                new_frame_counter
            ),

        "new_verb_distribution":
            dict(
                new_verb_counter
            ),

        "violation_types":
            dict(
                violation_counter
            ),

        "warning_types":
            dict(
                warning_counter
            ),

        "participant_overlap_policy": {
            "exact_reflexive_allowed":
                False,

            "speaker_group_overlap_allowed":
                False,

            "blocked_pairs": [
                list(pair)
                for pair in sorted(
                    PARTICIPANT_OVERLAP_PAIRS
                )
            ],
        },

        "activation_version":
            activation.get(
                "activation_version"
            ),
    }

    write_json(
        summary_file,
        summary,
    )

    print("=" * 100)
    print(
        "SEMANTIC COMPATIBILITY VALIDATOR V0.5.2"
    )
    print("=" * 100)

    print(
        "Input:",
        args.input,
    )

    print(
        "Total:",
        total,
    )

    print(
        "Accepted:",
        len(
            accepted
        ),
    )

    print(
        "Rejected:",
        len(
            rejected
        ),
    )

    print(
        "Accept rate:",
        (
            f"{len(accepted) / total:.2%}"
            if total
            else "0.00%"
        ),
    )

    print(
        "Rows with warnings:",
        rows_with_warnings,
    )

    print(
        "Warning rate:",
        (
            f"{rows_with_warnings / total:.2%}"
            if total
            else "0.00%"
        ),
    )

    print()

    print(
        "Participant overlap policy:"
    )

    print(
        "Blocked:",
        sorted(
            PARTICIPANT_OVERLAP_PAIRS
        ),
    )

    print()

    print(
        "New frame distribution:"
    )

    for frame_id, count in (
        new_frame_counter.most_common()
    ):

        print(
            f"{frame_id:<25}"
            f"{count}"
        )

    print()

    print(
        "New verb distribution:"
    )

    for verb_id, count in (
        new_verb_counter.most_common()
    ):

        print(
            f"{str(verb_id):<20}"
            f"{count}"
        )

    print()

    print("Violations:")
    print("-" * 80)

    if violation_counter:

        for name, count in (
            violation_counter.most_common()
        ):

            print(
                f"{name:<40}"
                f"{count}"
            )

    else:
        print("None")

    print()

    print("Warnings:")
    print("-" * 80)

    if warning_counter:

        for name, count in (
            warning_counter.most_common()
        ):

            print(
                f"{name:<40}"
                f"{count}"
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