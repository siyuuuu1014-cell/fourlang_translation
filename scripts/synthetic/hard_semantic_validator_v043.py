from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts.synthetic.hard_semantic_validator_v04 import (
    HardSemanticValidatorV04,
    read_jsonl,
    write_json,
    write_jsonl,
    DEFAULT_FRAMES,
)

from scripts.synthetic.generate_synthetic_v04 import (
    PROJECT_ROOT,
)


RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v043.jsonl"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v043_regression_200"
    / "01_compatibility"
    / "compatibility_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v043_regression_200"
    / "02_hard_semantic"
)


VALID_TENSES = {
    "present",
    "future",
}

VALID_POLARITIES = {
    "pos",
    "neg",
}


class HardSemanticValidatorV043(
    HardSemanticValidatorV04
):

    def validate_features(
        self,
        row: dict,
    ) -> list[dict]:

        violations = []

        features = row.get(
            "features"
        )

        if not isinstance(
            features,
            dict,
        ):

            return [{
                "type":
                    "INVALID_FEATURES"
            }]

        slots = row.get(
            "slots",
            {},
        )

        if not isinstance(
            slots,
            dict,
        ):
            slots = {}

        has_verb = bool(
            slots.get(
                "verb"
            )
        )

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity"
        )

        # ----------------------------------------------------
        # Verbal frames:
        # tense remains mandatory.
        # ----------------------------------------------------

        if has_verb:

            if tense not in VALID_TENSES:

                violations.append({
                    "type":
                        "INVALID_TENSE",

                    "value":
                        tense,
                })

        # ----------------------------------------------------
        # Verbless frame:
        # tense MUST NOT exist.
        # ----------------------------------------------------

        else:

            if tense is not None:

                violations.append({
                    "type":
                        "VERBLESS_FRAME_HAS_TENSE",

                    "value":
                        tense,
                })

        # ----------------------------------------------------
        # Polarity
        # ----------------------------------------------------

        if polarity is not None:

            if polarity not in VALID_POLARITIES:

                violations.append({
                    "type":
                        "INVALID_POLARITY",

                    "value":
                        polarity,
                })

        elif has_verb:

            violations.append({
                "type":
                    "MISSING_POLARITY"
            })

        return violations


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Hard Semantic Validator V0.4.3"
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

    rows = read_jsonl(
        input_path
    )

    validator = (
        HardSemanticValidatorV043(
            concepts_path=Path(
                args.concepts
            ),
            frames_path=Path(
                args.frames
            ),
        )
    )

    judged_rows = []
    accepted_rows = []
    rejected_rows = []

    violations = Counter()

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
            "hard_semantic_validation"
        ] = result

        judged_rows.append(
            output_row
        )

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

                violations[
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

    accept_rate = (
        accepted / total
        if total
        else 0.0
    )

    summary = {
        "validator_version":
            "0.4.3",

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

        "reject_reasons":
            dict(
                violations.most_common()
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
        "HARD SEMANTIC VALIDATOR V0.4.3"
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
        "Reject reasons:"
    )

    print(
        "-" * 70
    )

    if violations:

        for key, value in (
            violations.most_common()
        ):

            print(
                f"{key:<40}{value}"
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