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
    / "02_hard_semantic"
    / "hard_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v053_smoke_100"
    / "03_grammar_hard"
)


NEW_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
}


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


# =============================================================================
# Frozen V0.5.2
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
        "scripts.synthetic.grammar_hard_validator_v052",
        "--input",
        str(input_file),
        "--output-dir",
        str(validator_dir),
    ]

    print()
    print("=" * 100)
    print(
        "RUNNING FROZEN V0.5.2 "
        "GRAMMAR HARD"
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
        / "grammar_judged.jsonl"
    )

    if not judged_file.exists():

        candidates = sorted(
            validator_dir.rglob(
                "grammar_judged.jsonl"
            )
        )

        if candidates:
            judged_file = candidates[0]

    if not judged_file.exists():
        raise RuntimeError(
            "Frozen V0.5.2 grammar validator "
            "did not produce grammar_judged.jsonl. "
            f"returncode={result.returncode}"
        )

    judged = read_jsonl(
        judged_file
    )

    if len(judged) != len(rows):
        raise RuntimeError(
            "Frozen V0.5.2 grammar "
            "row count mismatch: "
            f"{len(judged)} != {len(rows)}"
        )

    return judged


# =============================================================================
# New V0.5.3 validation
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

    # -------------------------------------------------------------------------
    # Upstream validation must have passed.
    # -------------------------------------------------------------------------

    compatibility = row.get(
        "compatibility_validation",
        {},
    )

    if (
        isinstance(
            compatibility,
            dict,
        )
        and compatibility.get("accept") is False
    ):
        add_violation(
            violations,
            "UPSTREAM_COMPATIBILITY_REJECT",
        )

    hard_semantic = row.get(
        "hard_semantic_validation",
        {},
    )

    if (
        isinstance(
            hard_semantic,
            dict,
        )
        and hard_semantic.get("accept") is False
    ):
        add_violation(
            violations,
            "UPSTREAM_HARD_SEMANTIC_REJECT",
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
        features,
        dict,
    ):
        add_violation(
            violations,
            "INVALID_FEATURES",
        )

        return {
            "validator_version": VERSION,
            "accept": False,
            "violations": violations,
        }

    if frame_id not in NEW_FRAMES:
        add_violation(
            violations,
            "UNSUPPORTED_V053_FRAME",
            frame=frame_id,
        )

        return {
            "validator_version": VERSION,
            "accept": False,
            "violations": violations,
        }

    subject_id = slots.get(
        "subject"
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

    if not subject_id:
        add_violation(
            violations,
            "MISSING_SUBJECT",
        )

    if violations:
        return {
            "validator_version": VERSION,
            "accept": False,
            "violations": violations,
        }

    # =========================================================================
    # Deterministic grammatical realization
    #
    # At this point semantic structure has already passed Hard Semantic.
    # Therefore a mismatch here is treated as a language realization /
    # agreement / tense problem.
    # =========================================================================

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

        return {
            "validator_version": VERSION,
            "accept": False,
            "violations": violations,
        }

    language_error_type = {
        "zh": "ZH_TENSE_OR_WORD_ORDER",
        "en": "EN_AGREEMENT_OR_TENSE",
        "ru": "RU_AGREEMENT_OR_TENSE",
        "uz": "UZ_AGREEMENT_OR_TENSE",
    }

    for language in (
        "zh",
        "en",
        "ru",
        "uz",
    ):

        actual = (
            texts.get(language)
            if isinstance(texts, dict)
            else None
        )

        expected_text = expected.get(
            language
        )

        if actual != expected_text:

            add_violation(
                violations,
                language_error_type[
                    language
                ],
                language=language,
                expected=expected_text,
                actual=actual,
            )

    # =========================================================================
    # Additional verb-form checks
    #
    # These checks make the reason more explicit if someone later changes only
    # morphology while keeping the rest of the sentence superficially intact.
    # =========================================================================

    person_code = renderer._person_code(
        subject_id
    )

    verb_id = slots.get(
        "verb"
    )

    if verb_id in {
        "GIVE",
        "BRING",
    }:

        resource = renderer.v053_verbs[
            verb_id
        ]

        # English
        expected_en_verb = (
            renderer._form(
                resource,
                "en",
                person_code=person_code,
                tense=tense,
                polarity=polarity,
            )
        )

        en_text = texts.get(
            "en",
            "",
        )

        if (
            expected_en_verb
            and expected_en_verb not in en_text
        ):
            if not any(
                x["type"]
                == "EN_AGREEMENT_OR_TENSE"
                for x in violations
            ):
                add_violation(
                    violations,
                    "EN_AGREEMENT_OR_TENSE",
                    expected_verb=expected_en_verb,
                    actual=en_text,
                )

        # Russian
        expected_ru_verb = (
            renderer._form(
                resource,
                "ru",
                person_code=person_code,
                tense=tense,
                polarity=polarity,
            )
        )

        ru_text = texts.get(
            "ru",
            "",
        )

        if (
            expected_ru_verb
            and expected_ru_verb not in ru_text
        ):
            if not any(
                x["type"]
                == "RU_AGREEMENT_OR_TENSE"
                for x in violations
            ):
                add_violation(
                    violations,
                    "RU_AGREEMENT_OR_TENSE",
                    expected_verb=expected_ru_verb,
                    actual=ru_text,
                )

        # Uzbek
        expected_uz_verb = (
            renderer._form(
                resource,
                "uz",
                person_code=person_code,
                tense=tense,
                polarity=polarity,
            )
        )

        uz_text = texts.get(
            "uz",
            "",
        )

        if (
            expected_uz_verb
            and expected_uz_verb not in uz_text
        ):
            if not any(
                x["type"]
                == "UZ_AGREEMENT_OR_TENSE"
                for x in violations
            ):
                add_violation(
                    violations,
                    "UZ_AGREEMENT_OR_TENSE",
                    expected_verb=expected_uz_verb,
                    actual=uz_text,
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
            "Grammar Hard Validator V0.5.3"
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

def main():

    args = parse_args()

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
        if row.get("frame_id")
        not in NEW_FRAMES
    ]

    new_rows = [
        row
        for row in rows
        if row.get("frame_id")
        in NEW_FRAMES
    ]

    print("=" * 100)
    print(
        "GRAMMAR HARD VALIDATOR V0.5.3"
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
        "Frozen V0.5.2 rows:",
        len(frozen_rows),
    )

    print(
        "V0.5.3 new rows:",
        len(new_rows),
    )

    # =========================================================================
    # Frozen parent
    # =========================================================================

    frozen_judged = run_frozen_v052(
        frozen_rows,
        output_dir,
    )

    # =========================================================================
    # V0.5.3
    # =========================================================================

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
            "grammar_validation"
        ] = validate_new_row(
            row,
            renderer,
        )

        new_judged.append(
            output
        )

    # =========================================================================
    # Merge preserving order
    # =========================================================================

    by_id = {}

    for row in (
        frozen_judged
        + new_judged
    ):

        sid = row.get(
            "semantic_id"
        )

        if sid in by_id:
            raise RuntimeError(
                f"Duplicate semantic_id: {sid}"
            )

        by_id[sid] = row

    judged = []

    for original in rows:

        sid = original.get(
            "semantic_id"
        )

        if sid not in by_id:
            raise RuntimeError(
                f"Missing validated row: {sid}"
            )

        judged.append(
            by_id[sid]
        )

    accepted = []
    rejected = []

    reason_counter = Counter()
    language_counter = Counter()
    frame_counter = Counter()

    new_accepted = 0
    new_rejected = 0

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

        if accept:
            accepted.append(row)
        else:
            rejected.append(row)

        if (
            row.get("frame_id")
            in NEW_FRAMES
        ):
            frame_counter[
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

            violation_type = violation.get(
                "type",
                "UNKNOWN",
            )

            reason_counter[
                violation_type
            ] += 1

            language = violation.get(
                "language"
            )

            if language:
                language_counter[
                    language
                ] += 1

    # =========================================================================
    # Write
    # =========================================================================

    judged_file = (
        output_dir
        / "grammar_judged.jsonl"
    )

    accepted_file = (
        output_dir
        / "grammar_accepted.jsonl"
    )

    rejected_file = (
        output_dir
        / "grammar_rejected.jsonl"
    )

    summary_file = (
        output_dir
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

        "frozen_parent":
            FROZEN_PARENT,

        "input":
            str(args.input),

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
            dict(frame_counter),

        "rejected_by_language":
            dict(language_counter),

        "reject_reasons":
            dict(reason_counter),
    }

    write_json(
        summary_file,
        summary,
    )

    # =========================================================================
    # Report
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "V0.5.3 GRAMMAR HARD "
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
        frame_counter.most_common()
    ):
        print(
            f"{frame_id:<32}{count}"
        )

    print()
    print(
        "Rejected by language:"
    )
    print("-" * 80)

    if language_counter:

        for language, count in (
            language_counter.most_common()
        ):
            print(
                f"{language:<12}{count}"
            )

    else:
        print("None")

    print()
    print("Reject reasons:")
    print("-" * 80)

    if reason_counter:

        for reason, count in (
            reason_counter.most_common()
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