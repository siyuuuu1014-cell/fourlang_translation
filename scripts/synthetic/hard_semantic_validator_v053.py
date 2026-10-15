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

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v053_smoke_100"
    / "01_compatibility"
    / "compatibility_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v053_smoke_100"
    / "02_hard_semantic"
)


NEW_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
}

PARTICIPANT_OVERLAP_PAIRS = {
    ("PERSON_I", "PERSON_WE"),
    ("PERSON_WE", "PERSON_I"),
}


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
        encoding="utf-8-sig",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
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


def write_json(
    path: Path,
    data: dict,
) -> None:
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
    **kwargs,
):
    item = {
        "type": violation_type,
    }

    item.update(kwargs)

    violations.append(item)


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
        "scripts.synthetic.hard_semantic_validator_v052",
        "--input",
        str(input_file),
        "--output-dir",
        str(validator_dir),
    ]

    print()
    print("=" * 100)
    print(
        "RUNNING FROZEN V0.5.2 "
        "HARD SEMANTIC"
    )
    print("=" * 100)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    judged_file = (
        validator_dir
        / "hard_judged.jsonl"
    )

    if not judged_file.exists():
        candidates = sorted(
            validator_dir.rglob(
                "hard_judged.jsonl"
            )
        )

        if candidates:
            judged_file = candidates[0]

    if not judged_file.exists():
        raise RuntimeError(
            "Frozen V0.5.2 hard semantic "
            "did not produce hard_judged.jsonl. "
            f"returncode={result.returncode}"
        )

    judged = read_jsonl(
        judged_file
    )

    if len(judged) != len(rows):
        raise RuntimeError(
            "Frozen hard semantic count mismatch"
        )

    return judged


# =============================================================================
# V0.5.3
# =============================================================================

def validate_new_row(
    row: dict,
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

    compatibility = row.get(
        "compatibility_validation",
        {},
    )

    if (
        isinstance(
            compatibility,
            dict,
        )
        and compatibility.get(
            "accept"
        ) is False
    ):
        add_violation(
            violations,
            "UPSTREAM_COMPATIBILITY_REJECT",
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
            "validator_version":
                VERSION,

            "accept":
                False,

            "violations":
                violations,
        }

    # =========================================================================
    # GIVE semantic structure
    # =========================================================================

    if frame_id == (
        "GIVE_OBJECT_PERSON"
    ):
        expected_slots = {
            "subject",
            "verb",
            "object",
            "person",
        }

        if set(
            slots.keys()
        ) != expected_slots:
            add_violation(
                violations,
                "GIVE_SLOT_STRUCTURE_MISMATCH",
                actual=sorted(
                    slots.keys()
                ),
            )

        if slots.get(
            "verb"
        ) != "GIVE":
            add_violation(
                violations,
                "FRAME_VERB_MISMATCH",
            )

        subject_id = slots.get(
            "subject"
        )

        person_id = slots.get(
            "person"
        )

        if subject_id == person_id:
            add_violation(
                violations,
                "REFLEXIVE_SEMANTICS_NOT_ENABLED",
            )

        if (
            subject_id,
            person_id,
        ) in PARTICIPANT_OVERLAP_PAIRS:
            add_violation(
                violations,
                "PARTICIPANT_OVERLAP_NOT_ENABLED",
            )

    # =========================================================================
    # BRING semantic structure
    # =========================================================================

    elif frame_id == (
        "BRING_OBJECT_DESTINATION"
    ):
        expected_slots = {
            "subject",
            "verb",
            "object",
            "destination",
        }

        if set(
            slots.keys()
        ) != expected_slots:
            add_violation(
                violations,
                "BRING_SLOT_STRUCTURE_MISMATCH",
                actual=sorted(
                    slots.keys()
                ),
            )

        if slots.get(
            "verb"
        ) != "BRING":
            add_violation(
                violations,
                "FRAME_VERB_MISMATCH",
            )

    else:
        add_violation(
            violations,
            "UNSUPPORTED_V053_FRAME",
            frame=frame_id,
        )

    # =========================================================================
    # Deterministic re-render
    #
    # This is the strongest role-swap detector:
    # semantic frame -> expected four-language realization
    # must equal stored text exactly.
    # =========================================================================

    if not violations:
        try:
            expected = renderer.render(
                frame_id=frame_id,
                slots=slots,
                features=features,
            )

        except Exception as exc:
            add_violation(
                violations,
                "EXPECTED_RENDER_FAILED",
                reason=str(exc),
            )

        else:
            for language in (
                "zh",
                "en",
                "ru",
                "uz",
            ):
                actual_text = (
                    texts.get(
                        language
                    )
                    if isinstance(
                        texts,
                        dict,
                    )
                    else None
                )

                expected_text = (
                    expected.get(
                        language
                    )
                )

                if (
                    actual_text
                    != expected_text
                ):
                    add_violation(
                        violations,
                        "TEXT_RENDER_MISMATCH",
                        language=language,
                        expected=expected_text,
                        actual=actual_text,
                    )

    return {
        "validator_version":
            VERSION,

        "accept":
            len(violations) == 0,

        "violations":
            violations,
    }


# =============================================================================
# Main
# =============================================================================

def main():
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

    args = parser.parse_args()

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

    frozen_rows = [
        row
        for row in rows
        if row.get(
            "frame_id"
        ) not in NEW_FRAMES
    ]

    new_rows = [
        row
        for row in rows
        if row.get(
            "frame_id"
        ) in NEW_FRAMES
    ]

    frozen_judged = run_frozen_v052(
        frozen_rows,
        output_dir,
    )

    from scripts.synthetic.renderer_v053 import (
        V053Renderer,
    )

    renderer = V053Renderer()

    new_judged = []

    for row in new_rows:
        output = copy.deepcopy(
            row
        )

        output[
            "hard_semantic_validation"
        ] = validate_new_row(
            row,
            renderer,
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

    new_accepted = 0
    new_rejected = 0

    for row in judged:
        validation = row.get(
            "hard_semantic_validation",
            {},
        )

        accept = bool(
            validation.get(
                "accept",
                False,
            )
        )

        if accept:
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

            if accept:
                new_accepted += 1
            else:
                new_rejected += 1

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

    write_json(
        summary_file,
        {
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

            "new_accepted":
                new_accepted,

            "new_rejected":
                new_rejected,

            "new_frame_distribution":
                dict(
                    new_frame_counter
                ),

            "reject_reasons":
                dict(reasons),
        },
    )

    print("=" * 100)
    print(
        "V0.5.3 HARD SEMANTIC "
        "VALIDATION COMPLETE"
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
    print("New V0.5.3:")
    print(
        "Accepted:",
        new_accepted,
    )
    print(
        "Rejected:",
        new_rejected,
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
    print("Reject reasons:")
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