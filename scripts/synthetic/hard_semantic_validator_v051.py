from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


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

DEFAULT_CONCEPTS = (
    V04_RESOURCE_DIR
    / "concepts_v044.jsonl"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
    / "01_compatibility"
    / "compatibility_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
    / "02_hard_semantic"
)


# ============================================================
# New V0.5.1 contract
# ============================================================

NEW_FRAME_TO_VERB = {
    "SEE_OBJECT": "SEE",
    "TAKE_OBJECT": "TAKE",
}

NEW_FRAMES = set(
    NEW_FRAME_TO_VERB
)

NEW_VERBS = set(
    NEW_FRAME_TO_VERB.values()
)

BLOCKED_FRAMES = {
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

BLOCKED_VERBS = {
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

def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}"
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

            if not isinstance(
                row,
                dict,
            ):
                raise RuntimeError(
                    f"{path}:{line_no} "
                    f"is not a JSON object."
                )

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


# ============================================================
# Row classification
# ============================================================

def is_v051_new_row(
    row: dict,
) -> bool:

    frame_id = row.get(
        "frame_id"
    )

    if frame_id in NEW_FRAMES:
        return True

    if frame_id in BLOCKED_FRAMES:
        return True

    metadata = row.get(
        "metadata",
        {},
    )

    if (
        isinstance(metadata, dict)
        and metadata.get(
            "generation_source"
        ) == "v051_new_frame"
    ):
        return True

    return False


# ============================================================
# Compatibility guard
# ============================================================

def get_compatibility_accept(
    row: dict,
) -> bool | None:

    value = row.get(
        "compatibility_validation"
    )

    if isinstance(
        value,
        dict,
    ) and "accept" in value:

        return bool(
            value["accept"]
        )

    # Frozen validator may use a related field name.
    for key, value in row.items():

        if (
            "compat" in key.lower()
            and isinstance(value, dict)
            and "accept" in value
        ):

            return bool(
                value["accept"]
            )

    return None


# ============================================================
# Renderer
# ============================================================

def build_renderer():

    from scripts.synthetic.renderer_v051 import (
        V051Renderer,
    )

    return V051Renderer()


# ============================================================
# V0.5.1 hard semantic validation
# ============================================================

def validate_new_row(
    row: dict,
    renderer,
) -> tuple[
    bool,
    list[dict],
]:

    violations: list[dict] = []

    semantic_id = row.get(
        "semantic_id"
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

    # ========================================================
    # Basic object structure
    # ========================================================

    if not semantic_id:

        violations.append({
            "type":
                "MISSING_SEMANTIC_ID",
        })

    if not isinstance(
        slots,
        dict,
    ):

        violations.append({
            "type":
                "INVALID_SLOTS",
        })

        return (
            False,
            violations,
        )

    if not isinstance(
        features,
        dict,
    ):

        violations.append({
            "type":
                "INVALID_FEATURES",
        })

        return (
            False,
            violations,
        )

    if not isinstance(
        texts,
        dict,
    ):

        violations.append({
            "type":
                "INVALID_TEXTS",
        })

        return (
            False,
            violations,
        )

    # ========================================================
    # Compatibility layer must already have accepted the row.
    # ========================================================

    compatibility_accept = (
        get_compatibility_accept(
            row
        )
    )

    if compatibility_accept is False:

        violations.append({
            "type":
                "COMPATIBILITY_NOT_ACCEPTED",
        })

    # ========================================================
    # Block disabled capabilities
    # ========================================================

    verb_id = slots.get(
        "verb"
    )

    if frame_id in BLOCKED_FRAMES:

        violations.append({
            "type":
                "BLOCKED_FRAME",

            "frame_id":
                frame_id,
        })

    if verb_id in BLOCKED_VERBS:

        violations.append({
            "type":
                "BLOCKED_VERB",

            "verb":
                verb_id,
        })

    # ========================================================
    # Frame
    # ========================================================

    expected_verb = (
        NEW_FRAME_TO_VERB.get(
            frame_id
        )
    )

    if expected_verb is None:

        violations.append({
            "type":
                "UNSUPPORTED_V051_FRAME",

            "frame_id":
                frame_id,
        })

        return (
            False,
            violations,
        )

    # ========================================================
    # Required slots
    # ========================================================

    subject_id = slots.get(
        "subject"
    )

    object_id = slots.get(
        "object"
    )

    if not subject_id:

        violations.append({
            "type":
                "MISSING_REQUIRED_SLOT",

            "slot":
                "subject",
        })

    if not verb_id:

        violations.append({
            "type":
                "MISSING_REQUIRED_SLOT",

            "slot":
                "verb",
        })

    if not object_id:

        violations.append({
            "type":
                "MISSING_REQUIRED_SLOT",

            "slot":
                "object",
        })

    # ========================================================
    # Frame → verb semantic identity
    # ========================================================

    if (
        verb_id
        and verb_id != expected_verb
    ):

        violations.append({
            "type":
                "FRAME_VERB_MISMATCH",

            "frame_id":
                frame_id,

            "expected":
                expected_verb,

            "actual":
                verb_id,
        })

    # ========================================================
    # Unexpected slots
    #
    # SEE_OBJECT / TAKE_OBJECT must stay simple transitive.
    # ========================================================

    allowed_slot_names = {
        "subject",
        "verb",
        "object",
    }

    unexpected_slots = (
        set(slots)
        - allowed_slot_names
    )

    if unexpected_slots:

        violations.append({
            "type":
                "UNEXPECTED_SLOT",

            "slots":
                sorted(
                    unexpected_slots
                ),
        })

    # ========================================================
    # Tense / polarity
    # ========================================================

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

    # ========================================================
    # Prevent accidental event semantics leakage.
    #
    # V0.5.1 SEE / TAKE do not yet model:
    # habitual / planned / completed / resultative.
    # ========================================================

    if "event_type" in features:

        violations.append({
            "type":
                "UNEXPECTED_EVENT_TYPE",

            "event_type":
                features.get(
                    "event_type"
                ),
        })

    # ========================================================
    # Language presence
    # ========================================================

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

            violations.append({
                "type":
                    "MISSING_LANGUAGE_TEXT",

                "language":
                    language,
            })

    # Stop before rendering if structure itself is invalid.
    if violations:

        return (
            False,
            violations,
        )

    # ========================================================
    # Re-render from semantic representation.
    #
    # This is the central hard semantic check:
    #
    # slots + features
    #       ↓
    # deterministic renderer
    #       ↓
    # expected multilingual texts
    #
    # Stored texts must be identical.
    # ========================================================

    try:

        expected_texts = renderer.render(
            frame_id=frame_id,
            slots=slots,
            features=features,
        )

    except Exception as exc:

        violations.append({
            "type":
                "RENDER_EXCEPTION",

            "detail":
                str(exc),
        })

        return (
            False,
            violations,
        )

    for language in (
        "zh",
        "en",
        "ru",
        "uz",
    ):

        expected = expected_texts.get(
            language
        )

        actual = texts.get(
            language
        )

        if actual != expected:

            violations.append({
                "type":
                    "TEXT_RENDER_MISMATCH",

                "language":
                    language,

                "expected":
                    expected,

                "actual":
                    actual,
            })

    return (
        len(violations) == 0,
        violations,
    )


# ============================================================
# Frozen V0.4.4.1 validator
# ============================================================

def run_frozen_validator(
    rows: list[dict],
    work_dir: Path,
    concepts_path: Path,
) -> list[dict]:

    if not rows:
        return []

    frozen_input = (
        work_dir
        / "_frozen_core_input.jsonl"
    )

    frozen_output = (
        work_dir
        / "_frozen_v0441_validator"
    )

    if frozen_output.exists():

        shutil.rmtree(
            frozen_output
        )

    write_jsonl(
        frozen_input,
        rows,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.synthetic.hard_semantic_validator_v0441",

        "--input",
        str(
            frozen_input
        ),

        "--output-dir",
        str(
            frozen_output
        ),

        "--concepts",
        str(
            concepts_path
        ),
    ]

    print()
    print("=" * 100)
    print("RUNNING FROZEN V0.4.4.1 HARD SEMANTIC")
    print("=" * 100)

    print(
        "Rows:",
        len(rows),
    )

    result = subprocess.run(
        command,
        cwd=str(
            PROJECT_ROOT
        ),
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Frozen hard semantic validator "
            f"failed with return code "
            f"{result.returncode}"
        )

    judged_file = (
        frozen_output
        / "hard_judged.jsonl"
    )

    judged_rows = read_jsonl(
        judged_file
    )

    if len(judged_rows) != len(rows):

        raise RuntimeError(
            "Frozen hard validator row count "
            f"mismatch: "
            f"input={len(rows)}, "
            f"output={len(judged_rows)}"
        )

    return judged_rows


# ============================================================
# Frozen result helpers
# ============================================================

def extract_hard_accept(
    row: dict,
) -> bool:

    preferred = row.get(
        "hard_semantic_validation"
    )

    if (
        isinstance(
            preferred,
            dict,
        )
        and "accept" in preferred
    ):

        return bool(
            preferred["accept"]
        )

    for key, value in row.items():

        if (
            "hard" in key.lower()
            and "semantic" in key.lower()
            and isinstance(
                value,
                dict,
            )
            and "accept" in value
        ):

            return bool(
                value["accept"]
            )

    raise RuntimeError(
        "Unable to locate hard semantic result "
        f"for {row.get('semantic_id')}"
    )


def extract_hard_violations(
    row: dict,
) -> list[dict]:

    preferred = row.get(
        "hard_semantic_validation"
    )

    if isinstance(
        preferred,
        dict,
    ):

        violations = preferred.get(
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
            "hard" in key.lower()
            and "semantic" in key.lower()
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
            "Hard Semantic Validator V0.5.1"
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

    concepts_path = Path(
        args.concepts
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_jsonl(
        input_path
    )

    # ========================================================
    # Split
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
    print("HARD SEMANTIC VALIDATOR V0.5.1")
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
    # Frozen rows
    # ========================================================

    frozen_judged = run_frozen_validator(
        rows=frozen_rows,
        work_dir=output_dir,
        concepts_path=concepts_path,
    )

    frozen_map = {
        row.get(
            "semantic_id"
        ): row
        for row in frozen_judged
    }

    # ========================================================
    # New rows
    # ========================================================

    renderer = build_renderer()

    new_map = {}

    for row in new_rows:

        output_row = dict(
            row
        )

        accept, violations = (
            validate_new_row(
                row,
                renderer,
            )
        )

        output_row[
            "hard_semantic_validation"
        ] = {
            "validator_version":
                VALIDATOR_VERSION,

            "accept":
                accept,

            "violations":
                violations,
        }

        new_map[
            row.get(
                "semantic_id"
            )
        ] = output_row

    # ========================================================
    # Restore original order
    # ========================================================

    judged_rows = []

    for original in rows:

        semantic_id = original.get(
            "semantic_id"
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
                "Missing hard semantic result: "
                f"{semantic_id}"
            )

    # ========================================================
    # Results
    # ========================================================

    accepted_rows = []
    rejected_rows = []

    reject_counter = Counter()

    new_accept = 0
    new_reject = 0

    for row in judged_rows:

        accept = extract_hard_accept(
            row
        )

        violations = (
            extract_hard_violations(
                row
            )
        )

        if accept:

            accepted_rows.append(
                row
            )

        else:

            rejected_rows.append(
                row
            )

            for violation in violations:

                if isinstance(
                    violation,
                    dict,
                ):

                    violation_type = (
                        violation.get(
                            "type",
                            "UNKNOWN",
                        )
                    )

                else:

                    violation_type = (
                        "UNKNOWN"
                    )

                reject_counter[
                    violation_type
                ] += 1

        if row.get(
            "frame_id"
        ) in NEW_FRAMES:

            if accept:
                new_accept += 1
            else:
                new_reject += 1

    # ========================================================
    # Files
    # ========================================================

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

        "new_accepted":
            new_accept,

        "new_rejected":
            new_reject,

        "reject_reasons":
            dict(
                reject_counter.most_common()
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
    print("V0.5.1 HARD SEMANTIC VALIDATION COMPLETE")
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

    print()

    print(
        "New V0.5.1:"
    )

    print(
        "Accepted:",
        new_accept,
    )

    print(
        "Rejected:",
        new_reject,
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

    print(judged_file)
    print(accepted_file)
    print(rejected_file)
    print(summary_file)


if __name__ == "__main__":
    main()