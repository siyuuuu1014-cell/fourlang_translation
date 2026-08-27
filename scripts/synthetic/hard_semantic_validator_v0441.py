from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts.synthetic.generate_synthetic_v04 import (
    PROJECT_ROOT,
)

from scripts.synthetic.hard_semantic_validator_v04 import (
    DEFAULT_FRAMES,
    read_jsonl,
    write_json,
    write_jsonl,
)

from scripts.synthetic.hard_semantic_validator_v043 import (
    HardSemanticValidatorV043,
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
    / "01_compatibility"
    / "compatibility_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v0441_regression_200_fix2"
    / "02_hard_semantic_v0441"
)


# ============================================================
# Validator
# ============================================================

class HardSemanticValidatorV0441(
    HardSemanticValidatorV043
):

    def __init__(
        self,
        *,
        concepts_path: Path,
        frames_path: Path,
    ) -> None:

        super().__init__(
            concepts_path=concepts_path,
            frames_path=frames_path,
        )

        # ----------------------------------------------------
        # Important:
        #
        # Parent V0.4.3 validator may still reconstruct text
        # using its old renderer.
        #
        # Therefore we keep an explicit V0.4.4.1 renderer for
        # final text-equivalence checking.
        # ----------------------------------------------------

        self.v0441_renderer = V0441Renderer(
            concepts_path=concepts_path,
            frames_path=frames_path,
        )

    # ========================================================
    # V0.4.4.1 event semantics
    # ========================================================

    def validate_event_type(
        self,
        row: dict,
    ) -> list[dict]:

        violations: list[dict] = []

        slots = row.get(
            "slots",
            {},
        )

        if not isinstance(
            slots,
            dict,
        ):
            slots = {}

        features = row.get(
            "features",
            {},
        )

        if not isinstance(
            features,
            dict,
        ):
            features = {}

        verb_id = slots.get(
            "verb"
        )

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity"
        )

        event_type = features.get(
            "event_type"
        )

        time_id = (
            slots.get(
                "time"
            )
            or slots.get(
                "day"
            )
        )

        destination = slots.get(
            "destination"
        )

        # ----------------------------------------------------
        # No event_type:
        #
        # perfectly legal for all old/non-motion frames.
        # ----------------------------------------------------

        if event_type is None:
            return violations

        # ====================================================
        # Habitual
        # ====================================================

        if event_type == "habitual":

            if verb_id not in {
                "GO",
                "COME",
            }:

                violations.append({
                    "type":
                        "HABITUAL_INVALID_VERB",

                    "verb":
                        verb_id,
                })

            if tense != "present":

                violations.append({
                    "type":
                        "HABITUAL_TENSE_MISMATCH",

                    "expected":
                        "present",

                    "actual":
                        tense,
                })

            if polarity != "pos":

                violations.append({
                    "type":
                        "HABITUAL_POLARITY_MISMATCH",

                    "expected":
                        "pos",

                    "actual":
                        polarity,
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

            # ------------------------------------------------
            # GO habitual without destination:
            #
            # "I go every day"
            #
            # deliberately excluded in V0.4.4.1.
            # ------------------------------------------------

            if (
                verb_id == "GO"
                and destination is None
            ):

                violations.append({
                    "type":
                        "HABITUAL_GO_MISSING_DESTINATION",
                })

            # ------------------------------------------------
            # If there is a destination, it must have a valid
            # Russian lexicalization class:
            #
            # local / travel
            # ------------------------------------------------

            if (
                verb_id in {
                    "GO",
                    "COME",
                }
                and destination is not None
            ):

                try:

                    self.v0441_renderer.get_destination_class(
                        destination
                    )

                except Exception as exc:

                    violations.append({
                        "type":
                            "UNKNOWN_MOTION_DESTINATION_CLASS",

                        "destination":
                            destination,

                        "detail":
                            str(exc),
                    })

            return violations

        # ====================================================
        # Planned
        # ====================================================

        if event_type == "planned":

            if verb_id not in {
                "GO",
                "COME",
            }:

                violations.append({
                    "type":
                        "PLANNED_INVALID_VERB",

                    "verb":
                        verb_id,
                })

            if tense != "future":

                violations.append({
                    "type":
                        "PLANNED_TENSE_MISMATCH",

                    "expected":
                        "future",

                    "actual":
                        tense,
                })

            # TIME_EVERY_DAY is reserved for habitual in
            # V0.4.4.1.

            if time_id == "TIME_EVERY_DAY":

                violations.append({
                    "type":
                        "PLANNED_EVERY_DAY_FORBIDDEN",
                })

            return violations

        # ====================================================
        # Unknown event_type
        # ====================================================

        violations.append({
            "type":
                "UNKNOWN_EVENT_TYPE",

            "event_type":
                event_type,
        })

        return violations

    # ========================================================
    # V0.4.4.1 text render verification
    # ========================================================

    def rerender_v0441(
        self,
        row: dict,
    ) -> dict[str, str]:

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

        computed = row.get(
            "computed",
            {},
        )

        if not isinstance(
            slots,
            dict,
        ):
            slots = {}

        if not isinstance(
            features,
            dict,
        ):
            features = {}

        if not isinstance(
            computed,
            dict,
        ):
            computed = {}

        return self.v0441_renderer.render(
            frame_id=frame_id,
            slots=slots,
            features=features,
            computed=computed,
        )

    # ========================================================
    # Main sample validation
    # ========================================================

    def validate_sample(
        self,
        row: dict,
    ) -> dict:

        # ----------------------------------------------------
        # 1. Run all V0.4.3 checks first.
        # ----------------------------------------------------

        parent_result = (
            super().validate_sample(
                row
            )
        )

        semantic_id = row.get(
            "semantic_id"
        )

        parent_violations = (
            parent_result.get(
                "violations",
                []
            )
        )

        if not isinstance(
            parent_violations,
            list,
        ):
            parent_violations = []

        final_violations: list[dict] = []

        # ----------------------------------------------------
        # 2. Re-render using V0.4.4.1.
        #
        # Parent TEXT_RENDER_MISMATCH may simply mean that the
        # old validator expected:
        #
        # едет / приходит
        #
        # while V0.4.4.1 correctly generated:
        #
        # ходит / ездит / приезжает
        # ----------------------------------------------------

        rerendered = None

        needs_rerender = any(
            v.get("type")
            == "TEXT_RENDER_MISMATCH"
            for v in parent_violations
            if isinstance(
                v,
                dict,
            )
        )

        if needs_rerender:

            try:

                rerendered = (
                    self.rerender_v0441(
                        row
                    )
                )

            except Exception as exc:

                final_violations.append({
                    "type":
                        "V0441_RERENDER_ERROR",

                    "detail":
                        str(exc),
                })

        actual_texts = row.get(
            "texts",
            {},
        )

        if not isinstance(
            actual_texts,
            dict,
        ):
            actual_texts = {}

        # ----------------------------------------------------
        # Process parent violations.
        # ----------------------------------------------------

        for violation in (
            parent_violations
        ):

            if not isinstance(
                violation,
                dict,
            ):
                continue

            violation_type = (
                violation.get(
                    "type"
                )
            )

            # -----------------------------------------------
            # Non-render violation:
            #
            # keep exactly as parent found it.
            # -----------------------------------------------

            if (
                violation_type
                != "TEXT_RENDER_MISMATCH"
            ):

                final_violations.append(
                    violation
                )

                continue

            # -----------------------------------------------
            # Parent text mismatch:
            #
            # validate again with V0441Renderer.
            # -----------------------------------------------

            language = violation.get(
                "language"
            )

            if (
                rerendered is None
                or language is None
            ):

                final_violations.append(
                    violation
                )

                continue

            expected_v0441 = (
                rerendered.get(
                    language
                )
            )

            actual = (
                actual_texts.get(
                    language
                )
            )

            # -----------------------------------------------
            # V0.4.4.1 agrees with generated corpus:
            #
            # old mismatch was just validator drift.
            # -----------------------------------------------

            if expected_v0441 == actual:
                continue

            # -----------------------------------------------
            # Real mismatch remains.
            # -----------------------------------------------

            final_violations.append({
                "type":
                    "TEXT_RENDER_MISMATCH",

                "language":
                    language,

                "expected":
                    expected_v0441,

                "actual":
                    actual,
            })

        # ----------------------------------------------------
        # 3. Event-type semantics
        # ----------------------------------------------------

        final_violations.extend(
            self.validate_event_type(
                row
            )
        )

        # ----------------------------------------------------
        # 4. Return
        # ----------------------------------------------------

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
            "Hard Semantic Validator V0.4.4.1"
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
            f"Concept file not found: "
            f"{concepts_path}"
        )

    if not frames_path.exists():

        raise FileNotFoundError(
            f"Frame file not found: "
            f"{frames_path}"
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
        HardSemanticValidatorV0441(
            concepts_path=concepts_path,
            frames_path=frames_path,
        )
    )

    judged_rows = []
    accepted_rows = []
    rejected_rows = []

    reason_counter = Counter()

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

                reason_counter[
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

        "reject_reasons":
            dict(
                reason_counter.most_common()
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
        "HARD SEMANTIC VALIDATOR V0.4.4.1"
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