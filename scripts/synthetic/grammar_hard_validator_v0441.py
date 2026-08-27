from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


from scripts.synthetic.generate_synthetic_v04 import (
    PROJECT_ROOT,
    DEFAULT_FRAMES,
)

from scripts.synthetic.renderer_v0441 import (
    V0441Renderer,
)


# ============================================================
# Version
# ============================================================

VALIDATOR_VERSION = "0.4.4.1"


RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v044.jsonl"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v0441_regression_200_fix2"
    / "02_hard_semantic_v0441"
    / "hard_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v0441_regression_200_fix2"
    / "03_grammar_hard_v0441"
)


# ============================================================
# IO
# ============================================================

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
                    f"Invalid JSONL at "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):

                raise RuntimeError(
                    f"{path}:{line_no} "
                    "must contain JSON object."
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
# Base validator
# ============================================================

def run_base_v04_validator(
    *,
    input_path: Path,
    output_dir: Path,
    concepts_path: Path,
) -> None:

    """
    Run the existing verified Grammar Hard V0.4 validator.

    We do not rewrite all of its EN/RU/UZ checks.

    V0.4.4.1 only repairs the part that became outdated:
    Russian habitual motion lexicalization.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.synthetic.grammar_hard_validator_v04",

        "--input",
        str(
            input_path
        ),

        "--output-dir",
        str(
            output_dir
        ),

        "--concepts",
        str(
            concepts_path
        ),
    ]

    print(
        "=" * 90
    )

    print(
        "RUNNING BASE GRAMMAR HARD V0.4"
    )

    print(
        "=" * 90
    )

    print(
        " ".join(
            command
        )
    )

    print()

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Base Grammar Hard V0.4 "
            f"failed with return code "
            f"{result.returncode}"
        )


# ============================================================
# Locate old validation result
# ============================================================

def find_base_validation(
    row: dict,
) -> dict:

    preferred_keys = (
        "grammar_hard_validation",
        "grammar_validation",
        "grammar_judgment",
        "grammar_judge",
    )

    for key in preferred_keys:

        value = row.get(
            key
        )

        if (
            isinstance(
                value,
                dict,
            )
            and "accept" in value
        ):

            return value

    # Defensive fallback:
    # search any grammar-looking field.

    for key, value in row.items():

        if "grammar" not in str(
            key
        ).lower():

            continue

        if not isinstance(
            value,
            dict,
        ):

            continue

        if "accept" in value:

            return value

    raise RuntimeError(
        "Could not locate base grammar "
        f"validation result for "
        f"{row.get('semantic_id')}"
    )


# ============================================================
# Helpers
# ============================================================

def is_habitual_motion(
    row: dict,
) -> bool:

    slots = row.get(
        "slots",
        {},
    )

    features = row.get(
        "features",
        {},
    )

    return (
        slots.get(
            "verb"
        )
        in {
            "GO",
            "COME",
        }

        and

        features.get(
            "event_type"
        )
        == "habitual"
    )


def violation_language(
    violation: dict,
) -> str | None:

    for key in (
        "language",
        "lang",
    ):

        value = violation.get(
            key
        )

        if value:

            return str(
                value
            ).lower()

    violation_type = str(
        violation.get(
            "type",
            ""
        )
    ).upper()

    # Common forms such as:
    #
    # RU_VERB_AGREEMENT
    # VERB_AGREEMENT_RU
    # RU_PERSON_MISMATCH

    if (
        violation_type.startswith(
            "RU_"
        )
        or "_RU_" in violation_type
        or violation_type.endswith(
            "_RU"
        )
    ):

        return "ru"

    return None


def is_russian_violation(
    violation: dict,
) -> bool:

    return (
        violation_language(
            violation
        )
        == "ru"
    )


# ============================================================
# V0.4.4.1 structural checks
# ============================================================

def validate_habitual_structure(
    row: dict,
) -> list[dict]:

    violations: list[dict] = []

    if not is_habitual_motion(
        row
    ):

        return violations

    slots = row.get(
        "slots",
        {},
    )

    features = row.get(
        "features",
        {},
    )

    verb_id = slots.get(
        "verb"
    )

    destination = slots.get(
        "destination"
    )

    time_id = (
        slots.get(
            "time"
        )
        or slots.get(
            "day"
        )
    )

    if features.get(
        "tense"
    ) != "present":

        violations.append({
            "type":
                "HABITUAL_TENSE_MISMATCH",

            "expected":
                "present",

            "actual":
                features.get(
                    "tense"
                ),
        })

    if features.get(
        "polarity"
    ) != "pos":

        violations.append({
            "type":
                "HABITUAL_POLARITY_MISMATCH",

            "expected":
                "pos",

            "actual":
                features.get(
                    "polarity"
                ),
        })

    if time_id != "TIME_EVERY_DAY":

        violations.append({
            "type":
                "HABITUAL_TIME_MISMATCH",

            "expected":
                "TIME_EVERY_DAY",

            "actual":
                time_id,
        })

    if (
        verb_id == "GO"
        and destination is None
    ):

        violations.append({
            "type":
                "HABITUAL_GO_MISSING_DESTINATION",
        })

    return violations


# ============================================================
# V0441 renderer verification
# ============================================================

def rerender_v0441(
    *,
    renderer: V0441Renderer,
    row: dict,
) -> dict[str, str]:

    return renderer.render(
        frame_id=row.get(
            "frame_id"
        ),

        slots=row.get(
            "slots",
            {},
        ),

        features=row.get(
            "features",
            {},
        ),

        computed=row.get(
            "computed",
            {},
        ),
    )


def validate_rendered_text(
    *,
    renderer: V0441Renderer,
    row: dict,
) -> list[dict]:

    violations = []

    actual_texts = row.get(
        "texts",
        {},
    )

    try:

        expected = rerender_v0441(
            renderer=renderer,
            row=row,
        )

    except Exception as exc:

        return [{
            "type":
                "V0441_RERENDER_ERROR",

            "detail":
                str(exc),
        }]

    for language in (
        "zh",
        "en",
        "ru",
        "uz",
    ):

        actual = actual_texts.get(
            language
        )

        wanted = expected.get(
            language
        )

        if actual != wanted:

            violations.append({
                "type":
                    "TEXT_RENDER_MISMATCH",

                "language":
                    language,

                "expected":
                    wanted,

                "actual":
                    actual,
            })

    return violations


# ============================================================
# Row validation
# ============================================================

def validate_row(
    *,
    renderer: V0441Renderer,
    row: dict,
) -> dict:

    semantic_id = row.get(
        "semantic_id"
    )

    base_validation = (
        find_base_validation(
            row
        )
    )

    base_violations = base_validation.get(
        "violations",
        []
    )

    if not isinstance(
        base_violations,
        list,
    ):

        base_violations = []

    final_violations: list[dict] = []

    habitual = is_habitual_motion(
        row
    )

    # ========================================================
    # Non-habitual rows
    #
    # Keep old Grammar Hard V0.4 result exactly.
    # ========================================================

    if not habitual:

        final_violations.extend(
            copy.deepcopy(
                base_violations
            )
        )

        return {
            "semantic_id":
                semantic_id,

            "accept":
                len(
                    final_violations
                ) == 0,

            "violations":
                final_violations,
        }

    # ========================================================
    # Habitual GO / COME
    #
    # V0.4 validator has outdated Russian morphology.
    #
    # Keep every non-Russian violation.
    # Russian violations will be replaced by V0441 checks.
    # ========================================================

    for violation in (
        base_violations
    ):

        if not isinstance(
            violation,
            dict,
        ):

            continue

        if is_russian_violation(
            violation
        ):

            continue

        final_violations.append(
            copy.deepcopy(
                violation
            )
        )

    # ========================================================
    # Structural habitual checks
    # ========================================================

    final_violations.extend(
        validate_habitual_structure(
            row
        )
    )

    # ========================================================
    # Exact V0441 render verification
    #
    # This verifies:
    #
    # local GO     -> ходить
    # travel GO    -> ездить
    # local COME   -> приходить
    # travel COME  -> приезжать
    #
    # together with correct person morphology.
    # ========================================================

    final_violations.extend(
        validate_rendered_text(
            renderer=renderer,
            row=row,
        )
    )

    return {
        "semantic_id":
            semantic_id,

        "accept":
            len(
                final_violations
            ) == 0,

        "violations":
            final_violations,
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Grammar Hard Validator V0.4.4.1"
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
        "--frames",
        default=str(
            DEFAULT_FRAMES
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

    frames_path = Path(
        args.frames
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path}"
        )

    if not concepts_path.exists():

        raise FileNotFoundError(
            f"Concepts file not found: "
            f"{concepts_path}"
        )

    if not frames_path.exists():

        raise FileNotFoundError(
            f"Frames file not found: "
            f"{frames_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Run old validator into a diagnostic subdirectory.
    # --------------------------------------------------------

    base_output_dir = (
        output_dir
        / "_base_v04"
    )

    run_base_v04_validator(
        input_path=input_path,
        output_dir=base_output_dir,
        concepts_path=concepts_path,
    )

    base_judged_file = (
        base_output_dir
        / "grammar_judged.jsonl"
    )

    if not base_judged_file.exists():

        raise FileNotFoundError(
            "Base validator did not produce "
            f"{base_judged_file}"
        )

    rows = read_jsonl(
        base_judged_file
    )

    # --------------------------------------------------------
    # V0441 renderer
    # --------------------------------------------------------

    renderer = V0441Renderer(
        concepts_path=concepts_path,
        frames_path=frames_path,
    )

    # --------------------------------------------------------
    # Final files
    # --------------------------------------------------------

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

    judged_rows = []
    accepted_rows = []
    rejected_rows = []

    reason_counter = Counter()
    language_counter = Counter()

    habitual_count = 0
    habitual_accepted = 0

    for row in rows:

        result = validate_row(
            renderer=renderer,
            row=row,
        )

        output_row = copy.deepcopy(
            row
        )

        # Remove possible old final field ambiguity.
        output_row[
            "grammar_hard_validation_v0441"
        ] = result

        judged_rows.append(
            output_row
        )

        if is_habitual_motion(
            row
        ):

            habitual_count += 1

        if result[
            "accept"
        ]:

            accepted_rows.append(
                output_row
            )

            if is_habitual_motion(
                row
            ):

                habitual_accepted += 1

        else:

            rejected_rows.append(
                output_row
            )

            for violation in (
                result[
                    "violations"
                ]
            ):

                violation_type = (
                    violation.get(
                        "type",
                        "UNKNOWN",
                    )
                )

                reason_counter[
                    violation_type
                ] += 1

                language = (
                    violation_language(
                        violation
                    )
                )

                if language:

                    language_counter[
                        language
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

    accept_rate = (
        accepted / total
        if total
        else 0.0
    )

    # --------------------------------------------------------
    # Base statistics
    # --------------------------------------------------------

    base_rejected_file = (
        base_output_dir
        / "grammar_rejected.jsonl"
    )

    if base_rejected_file.exists():

        base_rejected = len(
            read_jsonl(
                base_rejected_file
            )
        )

    else:

        base_rejected = None

    summary = {
        "validator_version":
            VALIDATOR_VERSION,

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
            accept_rate,

        "base_v04_rejected":
            base_rejected,

        "habitual_motion_samples":
            habitual_count,

        "habitual_motion_accepted":
            habitual_accepted,

        "habitual_motion_accept_rate":
            (
                habitual_accepted
                / habitual_count
                if habitual_count
                else 0.0
            ),

        "reject_reasons":
            dict(
                reason_counter.most_common()
            ),

        "rejected_by_language":
            dict(
                language_counter.most_common()
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

    # ========================================================
    # Console
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        "GRAMMAR HARD VALIDATOR V0.4.4.1"
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
        f"{accept_rate:.2%}",
    )

    print()

    print(
        "Base V0.4 rejected:",
        base_rejected,
    )

    print(
        "Habitual motion samples:",
        habitual_count,
    )

    print(
        "Habitual motion accepted:",
        habitual_accepted,
    )

    print()

    print(
        "Reject reasons:"
    )

    print(
        "-" * 70
    )

    if reason_counter:

        for key, value in (
            reason_counter.most_common()
        ):

            print(
                f"{key:<45}"
                f"{value}"
            )

    else:

        print(
            "None"
        )

    print()

    print(
        "Rejected by language:"
    )

    print(
        "-" * 70
    )

    if language_counter:

        for key, value in (
            language_counter.most_common()
        ):

            print(
                f"{key:<20}"
                f"{value}"
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

    print()

    print(
        "Base V0.4 diagnostic:"
    )

    print(
        base_output_dir
    )


if __name__ == "__main__":
    main()