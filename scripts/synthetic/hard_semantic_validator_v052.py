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
    / "v052_overlap_fix_200"
    / "01_compatibility"
    / "compatibility_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v052_overlap_fix_200"
    / "02_hard_semantic"
)


NEW_FRAMES = {
    "CALL_PERSON",
    "WAIT_PERSON",
    "WAIT_AT_PLACE",
}

FRAME_TO_VERB = {
    "CALL_PERSON": "CALL",
    "WAIT_PERSON": "WAIT",
    "WAIT_AT_PLACE": "WAIT",
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
# Participant overlap policy
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
                    f"Invalid JSONL: "
                    f"{path}:{line_number}"
                ) from exc

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


# =============================================================================
# Frozen V0.5.1 Validator
# =============================================================================

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
        "scripts.synthetic.hard_semantic_validator_v051",
        "--input",
        str(input_file),
        "--output-dir",
        str(validator_dir),
    ]

    print()
    print("=" * 100)
    print(
        "RUNNING FROZEN V0.5.1 HARD SEMANTIC"
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
            "Frozen V0.5.1 hard semantic "
            "validator did not produce "
            "hard_judged.jsonl. "
            f"Return code={result.returncode}"
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
            "Frozen validator row count mismatch: "
            f"input={len(rows)}, "
            f"output={len(judged_rows)}"
        )

    return judged_rows


# =============================================================================
# New frame structure
# =============================================================================

def validate_structure(
    row: dict,
) -> list[dict]:

    violations = []

    frame_id = row.get(
        "frame_id"
    )

    slots = row.get(
        "slots"
    )

    features = row.get(
        "features"
    )

    if not isinstance(
        slots,
        dict,
    ):

        add_violation(
            violations,
            "INVALID_SLOTS",
        )

        return violations

    if not isinstance(
        features,
        dict,
    ):

        add_violation(
            violations,
            "INVALID_FEATURES",
        )

        return violations

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

    expected_verb = FRAME_TO_VERB.get(
        frame_id
    )

    if expected_verb is None:

        add_violation(
            violations,
            "UNSUPPORTED_V052_FRAME",
            frame=frame_id,
        )

        return violations

    if verb_id != expected_verb:

        add_violation(
            violations,
            "FRAME_VERB_MISMATCH",
            expected=expected_verb,
            actual=verb_id,
        )

    subject_id = slots.get(
        "subject"
    )

    if not subject_id:

        add_violation(
            violations,
            "MISSING_SUBJECT",
        )

    # -------------------------------------------------------------------------
    # CALL_PERSON / WAIT_PERSON
    # -------------------------------------------------------------------------

    if frame_id in {
        "CALL_PERSON",
        "WAIT_PERSON",
    }:

        person_id = slots.get(
            "person"
        )

        if not person_id:

            add_violation(
                violations,
                "MISSING_PERSON",
            )

        elif subject_id:

            if subject_id == person_id:

                add_violation(
                    violations,
                    "REFLEXIVE_SEMANTICS_NOT_ENABLED",
                    subject=subject_id,
                    person=person_id,
                )

            elif (
                subject_id,
                person_id,
            ) in PARTICIPANT_OVERLAP_PAIRS:

                add_violation(
                    violations,
                    "PARTICIPANT_OVERLAP_NOT_ENABLED",
                    subject=subject_id,
                    person=person_id,
                )

    # -------------------------------------------------------------------------
    # WAIT_AT_PLACE
    # -------------------------------------------------------------------------

    if frame_id == "WAIT_AT_PLACE":

        if not slots.get(
            "place"
        ):

            add_violation(
                violations,
                "MISSING_PLACE",
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

        add_violation(
            violations,
            "INVALID_TENSE",
            tense=tense,
        )

    if polarity not in {
        "pos",
        "neg",
    }:

        add_violation(
            violations,
            "INVALID_POLARITY",
            polarity=polarity,
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

    return violations


# =============================================================================
# Deterministic semantic realization check
# =============================================================================

def validate_rendered_semantics(
    row: dict,
    *,
    renderer,
) -> list[dict]:

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

    actual_texts = row.get(
        "texts",
        {},
    )

    if not isinstance(
        actual_texts,
        dict,
    ):

        add_violation(
            violations,
            "INVALID_TEXT_CONTAINER",
        )

        return violations

    try:

        expected_texts = renderer.render(
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

        return violations

    for language in (
        "zh",
        "en",
        "ru",
        "uz",
    ):

        expected = expected_texts.get(
            language
        )

        actual = actual_texts.get(
            language
        )

        if actual != expected:

            add_violation(
                violations,
                "TEXT_RENDER_MISMATCH",
                language=language,
                expected=expected,
                actual=actual,
            )

    return violations


# =============================================================================
# Explicit semantic-role realization
# =============================================================================

def validate_argument_roles(
    row: dict,
    *,
    renderer,
) -> list[dict]:

    violations = []

    frame_id = row.get(
        "frame_id"
    )

    slots = row.get(
        "slots",
        {},
    )

    texts = row.get(
        "texts",
        {},
    )

    if not isinstance(
        texts,
        dict,
    ):
        return violations

    # -------------------------------------------------------------------------
    # CALL_PERSON
    # -------------------------------------------------------------------------

    if frame_id == "CALL_PERSON":

        person_id = slots.get(
            "person"
        )

        target = (
            renderer.person_targets.get(
                person_id,
                {},
            )
        )

        en_target = target.get(
            "en_object"
        )

        ru_target = target.get(
            "ru_dat"
        )

        uz_target = target.get(
            "uz_dat"
        )

        zh_target = target.get(
            "zh"
        )

        if (
            en_target
            and en_target not in texts.get(
                "en",
                "",
            )
        ):

            add_violation(
                violations,
                "EN_PERSON_ROLE_MISMATCH",
                expected=en_target,
            )

        if (
            ru_target
            and ru_target not in texts.get(
                "ru",
                "",
            )
        ):

            add_violation(
                violations,
                "RU_DATIVE_ROLE_MISMATCH",
                expected=ru_target,
            )

        if (
            uz_target
            and uz_target not in texts.get(
                "uz",
                "",
            )
        ):

            add_violation(
                violations,
                "UZ_DATIVE_ROLE_MISMATCH",
                expected=uz_target,
            )

        if zh_target:

            expected_zh = (
                f"给{zh_target}打电话"
            )

            if expected_zh not in texts.get(
                "zh",
                "",
            ):

                add_violation(
                    violations,
                    "ZH_CALL_ROLE_MISMATCH",
                    expected=expected_zh,
                )

    # -------------------------------------------------------------------------
    # WAIT_PERSON
    # -------------------------------------------------------------------------

    elif frame_id == "WAIT_PERSON":

        person_id = slots.get(
            "person"
        )

        target = (
            renderer.person_targets.get(
                person_id,
                {},
            )
        )

        en_target = target.get(
            "en_object"
        )

        ru_target = target.get(
            "ru_acc"
        )

        uz_target = target.get(
            "uz_acc"
        )

        zh_target = target.get(
            "zh"
        )

        en_text = texts.get(
            "en",
            "",
        )

        if (
            en_target
            and f"for {en_target}" not in en_text
        ):

            add_violation(
                violations,
                "EN_WAIT_FOR_ROLE_MISMATCH",
                expected=f"for {en_target}",
            )

        if (
            ru_target
            and ru_target not in texts.get(
                "ru",
                "",
            )
        ):

            add_violation(
                violations,
                "RU_ACCUSATIVE_ROLE_MISMATCH",
                expected=ru_target,
            )

        if (
            uz_target
            and uz_target not in texts.get(
                "uz",
                "",
            )
        ):

            add_violation(
                violations,
                "UZ_ACCUSATIVE_ROLE_MISMATCH",
                expected=uz_target,
            )

        if (
            zh_target
            and zh_target not in texts.get(
                "zh",
                "",
            )
        ):

            add_violation(
                violations,
                "ZH_WAIT_PERSON_ROLE_MISMATCH",
                expected=zh_target,
            )

    # -------------------------------------------------------------------------
    # WAIT_AT_PLACE
    # -------------------------------------------------------------------------

    elif frame_id == "WAIT_AT_PLACE":

        place_id = slots.get(
            "place"
        )

        place = (
            renderer.place_targets.get(
                place_id,
                {},
            )
        )

        expected_by_language = {
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
            expected_by_language.items()
        ):

            if (
                expected
                and expected not in texts.get(
                    language,
                    "",
                )
            ):

                add_violation(
                    violations,
                    "LOCATIVE_ROLE_MISMATCH",
                    language=language,
                    expected=expected,
                )

    return violations


# =============================================================================
# Validate new row
# =============================================================================

def validate_new_row(
    row: dict,
    *,
    renderer,
) -> dict:

    violations = []

    violations.extend(
        validate_structure(
            row
        )
    )

    if not violations:

        violations.extend(
            validate_rendered_semantics(
                row,
                renderer=renderer,
            )
        )

        violations.extend(
            validate_argument_roles(
                row,
                renderer=renderer,
            )
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
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Hard Semantic Validator V0.5.2"
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

    rows = read_jsonl(
        args.input
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_rows = []
    new_rows = []

    for row in rows:

        frame_id = row.get(
            "frame_id"
        )

        metadata = row.get(
            "metadata",
            {},
        )

        source = (
            metadata.get(
                "generation_source"
            )
            if isinstance(
                metadata,
                dict,
            )
            else None
        )

        if frame_id in NEW_FRAMES:

            new_rows.append(
                row
            )

        elif source == "v052_new_frame":

            bad = copy.deepcopy(
                row
            )

            bad[
                "hard_semantic_validation"
            ] = {
                "validator_version":
                    VERSION,

                "accept":
                    False,

                "violations": [
                    {
                        "type":
                            "INVALID_V052_NEW_FRAME",

                        "frame":
                            frame_id,
                    }
                ],
            }

            new_rows.append(
                bad
            )

        else:

            frozen_rows.append(
                row
            )

    print("=" * 100)
    print(
        "HARD SEMANTIC VALIDATOR V0.5.2"
    )
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

    print(
        "Participant overlap blocked:",
        sorted(
            PARTICIPANT_OVERLAP_PAIRS
        ),
    )

    # -------------------------------------------------------------------------
    # Frozen
    # -------------------------------------------------------------------------

    frozen_judged = (
        run_frozen_v051_validator(
            rows=frozen_rows,
            output_dir=output_dir,
        )
    )

    # -------------------------------------------------------------------------
    # New
    # -------------------------------------------------------------------------

    from scripts.synthetic.renderer_v052 import (
        V052Renderer,
    )

    renderer = V052Renderer()

    new_judged = []

    for row in new_rows:

        existing = row.get(
            "hard_semantic_validation"
        )

        if (
            isinstance(
                existing,
                dict,
            )
            and existing.get(
                "accept"
            ) is False
            and existing.get(
                "validator_version"
            ) == VERSION
        ):

            new_judged.append(
                row
            )

            continue

        output_row = copy.deepcopy(
            row
        )

        output_row[
            "hard_semantic_validation"
        ] = validate_new_row(
            row,
            renderer=renderer,
        )

        new_judged.append(
            output_row
        )

    # -------------------------------------------------------------------------
    # Merge preserving source order
    # -------------------------------------------------------------------------

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

        validated = by_id.get(
            semantic_id
        )

        if validated is None:

            raise RuntimeError(
                "Missing validated row: "
                f"{semantic_id}"
            )

        judged.append(
            validated
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    accepted = []
    rejected = []

    reject_counter = Counter()
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
            accepted.append(
                row
            )

        else:
            rejected.append(
                row
            )

        frame_id = row.get(
            "frame_id"
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

            reject_counter[
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

    total = len(
        judged
    )

    summary = {
        "validator_version":
            VERSION,

        "frozen_parent_validator":
            FROZEN_PARENT_VERSION,

        "input":
            str(
                args.input
            ),

        "total":
            total,

        "frozen_rows":
            len(
                frozen_rows
            ),

        "new_rows":
            len(
                new_rows
            ),

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
    }

    write_json(
        summary_file,
        summary,
    )

    print()
    print("=" * 100)
    print(
        "V0.5.2 HARD SEMANTIC VALIDATION COMPLETE"
    )
    print("=" * 100)

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

    print()

    print(
        "New V0.5.2:"
    )

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
            f"{frame_id:<25}"
            f"{count}"
        )

    print()

    print(
        "Reject reasons:"
    )

    print("-" * 80)

    if reject_counter:

        for name, count in (
            reject_counter.most_common()
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