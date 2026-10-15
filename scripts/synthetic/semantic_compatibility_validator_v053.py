from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.3"
FROZEN_PARENT = "0.5.2"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v053.json"
)

ACTIVATION_FILE = (
    RESOURCE_DIR
    / "activation_v053.json"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v053_smoke_100"
    / "semantic_v053_raw.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v053_smoke_100"
    / "01_compatibility"
)


NEW_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
}

NEW_VERBS = {
    "GIVE",
    "BRING",
}

BLOCKED_VERBS = {
    "LEAVE",
    "LOSE",
    "NEED",
    "RETURN",
}

BLOCKED_FRAMES = {
    "LEAVE_PLACE",
    "LOSE_OBJECT",
    "NEED_OBJECT",
    "RETURN_PLACE",
}

PARTICIPANT_OVERLAP_PAIRS = {
    ("PERSON_I", "PERSON_WE"),
    ("PERSON_WE", "PERSON_I"),
}


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
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return data


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL: "
                    f"{path}:{line_number}"
                ) from exc

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
# Helpers
# =============================================================================

def add_violation(
    violations: list[dict],
    violation_type: str,
    **kwargs,
) -> None:
    item = {
        "type": violation_type,
    }

    item.update(kwargs)

    violations.append(item)


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
# Resource load
# =============================================================================

def load_resources():
    activation = read_json(
        ACTIVATION_FILE
    )

    arguments = read_json(
        ARGUMENT_FILE
    )

    if (
        activation.get(
            "activation_version"
        )
        != VERSION
    ):
        raise RuntimeError(
            "activation version mismatch"
        )

    if (
        activation.get("status")
        != "validated"
    ):
        raise RuntimeError(
            "V0.5.3 activation is not validated"
        )

    objects = arguments.get(
        "objects",
        {},
    )

    persons = arguments.get(
        "person_targets",
        {},
    )

    places = arguments.get(
        "place_targets",
        {},
    )

    if not objects:
        raise RuntimeError(
            "objects resource is empty"
        )

    if not persons:
        raise RuntimeError(
            "person_targets resource is empty"
        )

    if not places:
        raise RuntimeError(
            "place_targets resource is empty"
        )

    return (
        activation,
        objects,
        persons,
        places,
    )


# =============================================================================
# Frozen parent
# =============================================================================

def run_frozen_v052(
    rows: list[dict],
    output_dir: Path,
) -> list[dict]:
    if not rows:
        return []

    input_file = (
        output_dir
        / "_frozen_v052_input.jsonl"
    )

    validator_dir = (
        output_dir
        / "_frozen_v052_validator"
    )

    write_jsonl(
        input_file,
        rows,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.synthetic.semantic_compatibility_validator_v052",
        "--input",
        str(input_file),
        "--output-dir",
        str(validator_dir),
    ]

    print()
    print("=" * 100)
    print(
        "RUNNING FROZEN V0.5.2 "
        "COMPATIBILITY"
    )
    print("=" * 100)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    judged_file = (
        validator_dir
        / "compatibility_judged.jsonl"
    )

    if not judged_file.exists():
        candidates = sorted(
            validator_dir.rglob(
                "compatibility_judged.jsonl"
            )
        )

        if candidates:
            judged_file = candidates[0]

    if not judged_file.exists():
        raise RuntimeError(
            "Frozen V0.5.2 compatibility "
            "did not produce judged JSONL. "
            f"returncode={result.returncode}"
        )

    judged = read_jsonl(
        judged_file
    )

    if len(judged) != len(rows):
        raise RuntimeError(
            "Frozen compatibility count mismatch"
        )

    return judged


# =============================================================================
# New row validation
# =============================================================================

def validate_new_row(
    row: dict,
    *,
    objects: dict,
    persons: dict,
    places: dict,
) -> dict:
    violations = []
    warnings = []

    sid = row.get(
        "semantic_id",
        "UNKNOWN",
    )

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
            "warnings": warnings,
        }

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

    # =========================================================================
    # GIVE
    # =========================================================================

    if frame_id == "GIVE_OBJECT_PERSON":
        if verb_id != "GIVE":
            add_violation(
                violations,
                "FRAME_VERB_MISMATCH",
                expected="GIVE",
                actual=verb_id,
            )

        subject_id = slots.get(
            "subject"
        )

        object_id = slots.get(
            "object"
        )

        person_id = slots.get(
            "person"
        )

        if not subject_id:
            add_violation(
                violations,
                "MISSING_SUBJECT",
            )

        if not object_id:
            add_violation(
                violations,
                "MISSING_OBJECT",
            )

        elif object_id not in objects:
            add_violation(
                violations,
                "UNKNOWN_OBJECT",
                object=object_id,
            )

        if not person_id:
            add_violation(
                violations,
                "MISSING_PERSON",
            )

        elif person_id not in persons:
            add_violation(
                violations,
                "UNKNOWN_PERSON",
                person=person_id,
            )

        if (
            subject_id
            and person_id
            and not person_target_allowed(
                subject_id,
                person_id,
            )
        ):
            if subject_id == person_id:
                violation_type = (
                    "REFLEXIVE_SEMANTICS_NOT_ENABLED"
                )
            else:
                violation_type = (
                    "PARTICIPANT_OVERLAP_NOT_ENABLED"
                )

            add_violation(
                violations,
                violation_type,
                subject=subject_id,
                person=person_id,
            )

        # Critical role-swap defense.
        if object_id in persons:
            add_violation(
                violations,
                "OBJECT_PERSON_ROLE_SWAP",
                object=object_id,
            )

        if person_id in objects:
            add_violation(
                violations,
                "PERSON_OBJECT_ROLE_SWAP",
                person=person_id,
            )

        for forbidden in (
            "destination",
            "place",
            "time",
            "day",
        ):
            if slots.get(
                forbidden
            ) is not None:
                add_violation(
                    violations,
                    "FORBIDDEN_SLOT",
                    slot=forbidden,
                )

    # =========================================================================
    # BRING
    # =========================================================================

    elif frame_id == (
        "BRING_OBJECT_DESTINATION"
    ):
        if verb_id != "BRING":
            add_violation(
                violations,
                "FRAME_VERB_MISMATCH",
                expected="BRING",
                actual=verb_id,
            )

        subject_id = slots.get(
            "subject"
        )

        object_id = slots.get(
            "object"
        )

        destination_id = slots.get(
            "destination"
        )

        if not subject_id:
            add_violation(
                violations,
                "MISSING_SUBJECT",
            )

        if not object_id:
            add_violation(
                violations,
                "MISSING_OBJECT",
            )

        elif object_id not in objects:
            add_violation(
                violations,
                "UNKNOWN_OBJECT",
                object=object_id,
            )

        if not destination_id:
            add_violation(
                violations,
                "MISSING_DESTINATION",
            )

        elif destination_id not in places:
            add_violation(
                violations,
                "UNKNOWN_DESTINATION",
                destination=destination_id,
            )

        # Critical role-swap defense.
        if object_id in places:
            add_violation(
                violations,
                "OBJECT_DESTINATION_ROLE_SWAP",
                object=object_id,
            )

        if destination_id in objects:
            add_violation(
                violations,
                "DESTINATION_OBJECT_ROLE_SWAP",
                destination=destination_id,
            )

        for forbidden in (
            "person",
            "place",
            "time",
            "day",
        ):
            if slots.get(
                forbidden
            ) is not None:
                add_violation(
                    violations,
                    "FORBIDDEN_SLOT",
                    slot=forbidden,
                )

    else:
        add_violation(
            violations,
            "UNSUPPORTED_V053_NEW_FRAME",
            frame=frame_id,
        )

    # =========================================================================
    # Features
    # =========================================================================

    if features.get(
        "tense"
    ) not in {
        "present",
        "future",
    }:
        add_violation(
            violations,
            "INVALID_TENSE",
            tense=features.get("tense"),
        )

    if features.get(
        "polarity"
    ) not in {
        "pos",
        "neg",
    }:
        add_violation(
            violations,
            "INVALID_POLARITY",
            polarity=features.get(
                "polarity"
            ),
        )

    # =========================================================================
    # Text completeness
    # =========================================================================

    if not isinstance(
        texts,
        dict,
    ):
        add_violation(
            violations,
            "INVALID_TEXT_CONTAINER",
        )

    else:
        for language in (
            "zh",
            "en",
            "ru",
            "uz",
        ):
            value = texts.get(
                language
            )

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                add_violation(
                    violations,
                    "MISSING_LANGUAGE_TEXT",
                    language=language,
                )

    return {
        "validator_version":
            VERSION,

        "accept":
            len(violations) == 0,

        "violations":
            violations,

        "warnings":
            warnings,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

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

def main():
    args = parse_args()

    (
        activation,
        objects,
        persons,
        places,
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

    frozen_rows = []
    new_rows = []

    for row in rows:
        if (
            row.get("frame_id")
            in NEW_FRAMES
        ):
            new_rows.append(row)
        else:
            frozen_rows.append(row)

    frozen_judged = run_frozen_v052(
        frozen_rows,
        output_dir,
    )

    new_judged = []

    for row in new_rows:
        output = copy.deepcopy(
            row
        )

        output[
            "compatibility_validation"
        ] = validate_new_row(
            row,
            objects=objects,
            persons=persons,
            places=places,
        )

        new_judged.append(
            output
        )

    by_id = {}

    for row in (
        frozen_judged
        + new_judged
    ):
        by_id[
            row["semantic_id"]
        ] = row

    judged = [
        by_id[
            row["semantic_id"]
        ]
        for row in rows
    ]

    accepted = []
    rejected = []

    reasons = Counter()
    new_frame_counter = Counter()

    for row in judged:
        validation = row.get(
            "compatibility_validation",
            {},
        )

        if validation.get(
            "accept"
        ):
            accepted.append(row)
        else:
            rejected.append(row)

        if (
            row.get("frame_id")
            in NEW_FRAMES
        ):
            new_frame_counter[
                row["frame_id"]
            ] += 1

        for violation in validation.get(
            "violations",
            [],
        ):
            reasons[
                violation.get(
                    "type",
                    "UNKNOWN",
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

    total = len(judged)

    summary = {
        "validator_version":
            VERSION,

        "total":
            total,

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

        "new_frame_distribution":
            dict(
                new_frame_counter
            ),

        "reject_reasons":
            dict(reasons),
    }

    write_json(
        summary_file,
        summary,
    )

    print("=" * 100)
    print(
        "SEMANTIC COMPATIBILITY "
        "VALIDATOR V0.5.3"
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
        "Frozen V0.5.2 rows:",
        len(frozen_rows),
    )

    print(
        "V0.5.3 new rows:",
        len(new_rows),
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
    print(
        "New frame distribution:"
    )

    for frame_id, count in (
        new_frame_counter.most_common()
    ):
        print(
            f"{frame_id:<32}{count}"
        )

    print()
    print("Violations:")
    print("-" * 80)

    if reasons:
        for reason, count in (
            reasons.most_common()
        ):
            print(
                f"{reason:<45}{count}"
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