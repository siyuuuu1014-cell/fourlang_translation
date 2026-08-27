from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.synthetic.renderer_v04 import (
    RenderError,
    V04Renderer,
)


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v04.jsonl"
)

DEFAULT_FRAMES = (
    RESOURCE_DIR
    / "frames_v04.json"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v04_smoke_100"
    / "01_compatibility"
    / "compatibility_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v04_smoke_100"
    / "02_hard_semantic"
)


LANGUAGES = (
    "zh",
    "en",
    "ru",
    "uz",
)


VALID_TENSES = {
    "present",
    "future",
}


VALID_POLARITIES = {
    "pos",
    "neg",
}


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
            f"{path}: root must be JSON object."
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
                    f"Invalid JSONL at "
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
# Helpers
# ============================================================

def is_enabled(
    item: dict,
) -> bool:

    meta = item.get(
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


def normalize_text(
    text: Any,
) -> str:

    if text is None:
        return ""

    return str(
        text
    ).strip()


# ============================================================
# Validator
# ============================================================

class HardSemanticValidatorV04:

    def __init__(
        self,
        *,
        concepts_path: Path,
        frames_path: Path,
    ) -> None:

        concepts = read_jsonl(
            concepts_path
        )

        frames_data = read_json(
            frames_path
        )

        self.concepts = {
            row["id"]: row
            for row in concepts
        }

        frames = frames_data.get(
            "frames",
            [],
        )

        if not isinstance(
            frames,
            list,
        ):

            raise RuntimeError(
                "frames_v04.json: "
                "'frames' must be list."
            )

        self.frames = {
            frame["id"]: frame
            for frame in frames
            if (
                isinstance(
                    frame,
                    dict,
                )
                and frame.get(
                    "id"
                )
            )
        }

        self.renderer = V04Renderer(
            concepts_path=concepts_path,
            frames_path=frames_path,
        )

    # ========================================================
    # Concept check
    # ========================================================

    def validate_concept_reference(
        self,
        *,
        slot_name: str,
        concept_id: str,
        expected_types: list[str] | None,
    ) -> list[dict]:

        violations = []

        concept = self.concepts.get(
            concept_id
        )

        if concept is None:

            violations.append({
                "type":
                    "UNKNOWN_CONCEPT",

                "slot":
                    slot_name,

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

                "slot":
                    slot_name,

                "concept":
                    concept_id,
            })

        concept_type = concept.get(
            "concept_type"
        )

        if (
            expected_types
            and concept_type
            not in expected_types
        ):

            violations.append({
                "type":
                    "CONCEPT_TYPE_MISMATCH",

                "slot":
                    slot_name,

                "concept":
                    concept_id,

                "actual_type":
                    concept_type,

                "expected_types":
                    expected_types,
            })

        return violations

    # ========================================================
    # Frame structure
    # ========================================================

    def validate_frame_structure(
        self,
        row: dict,
    ) -> list[dict]:

        violations = []

        frame_id = row.get(
            "frame_id"
        )

        if not frame_id:

            return [{
                "type":
                    "MISSING_FRAME_ID"
            }]

        frame = self.frames.get(
            frame_id
        )

        if frame is None:

            return [{
                "type":
                    "UNKNOWN_FRAME",

                "frame":
                    frame_id,
            }]

        if not is_enabled(
            frame
        ):

            violations.append({
                "type":
                    "DISABLED_FRAME",

                "frame":
                    frame_id,
            })

            return violations

        slots = row.get(
            "slots"
        )

        if not isinstance(
            slots,
            dict,
        ):

            return violations + [{
                "type":
                    "INVALID_SLOTS"
            }]

        slot_specs = frame.get(
            "slots",
            {},
        )

        if not isinstance(
            slot_specs,
            dict,
        ):

            return violations + [{
                "type":
                    "INVALID_FRAME_SLOT_SCHEMA",

                "frame":
                    frame_id,
            }]

        # ----------------------------------------------------
        # Required and fixed slots
        # ----------------------------------------------------

        for slot_name, spec in (
            slot_specs.items()
        ):

            if not isinstance(
                spec,
                dict,
            ):

                continue

            required = bool(
                spec.get(
                    "required",
                    False,
                )
            )

            fixed_id = spec.get(
                "fixed_concept_id"
            )

            value = slots.get(
                slot_name
            )

            if (
                required
                and not value
                and not fixed_id
            ):

                violations.append({
                    "type":
                        "MISSING_REQUIRED_SLOT",

                    "frame":
                        frame_id,

                    "slot":
                        slot_name,
                })

                continue

            if fixed_id:

                # Generator may explicitly store the
                # fixed value, or Renderer may resolve it.
                if (
                    value is not None
                    and value != fixed_id
                ):

                    violations.append({
                        "type":
                            "FIXED_CONCEPT_MISMATCH",

                        "frame":
                            frame_id,

                        "slot":
                            slot_name,

                        "expected":
                            fixed_id,

                        "actual":
                            value,
                    })

                value = (
                    value
                    or fixed_id
                )

            if not value:
                continue

            expected_types = spec.get(
                "concept_types"
            )

            if not isinstance(
                expected_types,
                list,
            ):

                expected_types = None

            violations.extend(
                self.validate_concept_reference(
                    slot_name=slot_name,
                    concept_id=value,
                    expected_types=expected_types,
                )
            )

        # ----------------------------------------------------
        # Unexpected semantic slots
        # ----------------------------------------------------

        allowed_slot_names = set(
            slot_specs.keys()
        )

        for slot_name in slots:

            if slot_name not in (
                allowed_slot_names
            ):

                violations.append({
                    "type":
                        "UNEXPECTED_SLOT",

                    "frame":
                        frame_id,

                    "slot":
                        slot_name,
                })

        return violations

    # ========================================================
    # Feature validation
    # ========================================================

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

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity"
        )

        if tense not in VALID_TENSES:

            violations.append({
                "type":
                    "INVALID_TENSE",

                "value":
                    tense,
            })

        if polarity not in VALID_POLARITIES:

            violations.append({
                "type":
                    "INVALID_POLARITY",

                "value":
                    polarity,
            })

        return violations

    # ========================================================
    # Time → tense consistency
    # ========================================================

    def validate_time_tense(
        self,
        row: dict,
    ) -> list[dict]:

        violations = []

        slots = row.get(
            "slots",
            {},
        )

        features = row.get(
            "features",
            {},
        )

        if not isinstance(
            slots,
            dict,
        ):

            return violations

        if not isinstance(
            features,
            dict,
        ):

            return violations

        actual_tense = (
            features.get(
                "tense"
            )
        )

        for slot_name in (
            "time",
            "day",
        ):

            concept_id = slots.get(
                slot_name
            )

            if not concept_id:
                continue

            concept = self.concepts.get(
                concept_id
            )

            if concept is None:
                continue

            hint = (
                concept.get(
                    "time_features",
                    {},
                ).get(
                    "tense_hint"
                )
            )

            if (
                hint
                and actual_tense
                and hint != actual_tense
            ):

                violations.append({
                    "type":
                        "TIME_TENSE_MISMATCH",

                    "slot":
                        slot_name,

                    "concept":
                        concept_id,

                    "expected_tense":
                        hint,

                    "actual_tense":
                        actual_tense,
                })

        return violations

    # ========================================================
    # Language texts
    # ========================================================

    def validate_language_texts(
        self,
        row: dict,
    ) -> list[dict]:

        violations = []

        texts = row.get(
            "texts"
        )

        if not isinstance(
            texts,
            dict,
        ):

            return [{
                "type":
                    "INVALID_TEXTS"
            }]

        for lang in LANGUAGES:

            if lang not in texts:

                violations.append({
                    "type":
                        "MISSING_LANGUAGE",

                    "language":
                        lang,
                })

                continue

            text = normalize_text(
                texts.get(
                    lang
                )
            )

            if not text:

                violations.append({
                    "type":
                        "EMPTY_LANGUAGE_TEXT",

                    "language":
                        lang,
                })

        return violations

    # ========================================================
    # Deterministic rerender check
    # ========================================================

    def validate_rerender(
        self,
        row: dict,
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

        computed = row.get(
            "computed",
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

            return violations

        if not isinstance(
            features,
            dict,
        ):

            return violations

        if not isinstance(
            computed,
            dict,
        ):

            computed = {}

        if not isinstance(
            texts,
            dict,
        ):

            return violations

        try:

            rebuilt = self.renderer.render(
                frame_id=frame_id,
                slots=slots,
                features=features,
                computed=computed,
            )

        except RenderError as exc:

            violations.append({
                "type":
                    "RERENDER_ERROR",

                "message":
                    str(exc),
            })

            return violations

        for lang in LANGUAGES:

            expected = normalize_text(
                rebuilt.get(
                    lang
                )
            )

            actual = normalize_text(
                texts.get(
                    lang
                )
            )

            if actual != expected:

                violations.append({
                    "type":
                        "TEXT_RENDER_MISMATCH",

                    "language":
                        lang,

                    "expected":
                        expected,

                    "actual":
                        actual,
                })

        return violations

    # ========================================================
    # Verb-specific resource constraints
    # ========================================================

    def validate_verb_constraints(
        self,
        row: dict,
    ) -> list[dict]:

        violations = []

        slots = row.get(
            "slots",
            {},
        )

        features = row.get(
            "features",
            {},
        )

        if not isinstance(
            slots,
            dict,
        ):

            return violations

        verb_id = slots.get(
            "verb"
        )

        if not verb_id:

            return violations

        verb = self.concepts.get(
            verb_id
        )

        if verb is None:

            return violations

        if not is_enabled(
            verb
        ):

            violations.append({
                "type":
                    "DISABLED_VERB",

                "verb":
                    verb_id,
            })

            return violations

        allowed_tenses = (
            verb.get(
                "features",
                {},
            ).get(
                "allowed_tenses"
            )
        )

        actual_tense = (
            features.get(
                "tense"
            )
            if isinstance(
                features,
                dict,
            )
            else None
        )

        if (
            isinstance(
                allowed_tenses,
                list,
            )
            and actual_tense
            not in allowed_tenses
        ):

            violations.append({
                "type":
                    "VERB_TENSE_NOT_ALLOWED",

                "verb":
                    verb_id,

                "tense":
                    actual_tense,

                "allowed_tenses":
                    allowed_tenses,
            })

        # Current V0.4 policy:
        # ARRIVE requires destination.
        if (
            verb_id == "ARRIVE"
            and not slots.get(
                "destination"
            )
        ):

            violations.append({
                "type":
                    "ARRIVE_WITHOUT_DESTINATION"
            })

        return violations

    # ========================================================
    # Whole row
    # ========================================================

    def validate_sample(
        self,
        row: dict,
    ) -> dict:

        violations = []

        semantic_id = row.get(
            "semantic_id"
        )

        if not isinstance(
            semantic_id,
            str,
        ) or not semantic_id:

            violations.append({
                "type":
                    "INVALID_SEMANTIC_ID"
            })

        violations.extend(
            self.validate_frame_structure(
                row
            )
        )

        violations.extend(
            self.validate_features(
                row
            )
        )

        violations.extend(
            self.validate_time_tense(
                row
            )
        )

        violations.extend(
            self.validate_language_texts(
                row
            )
        )

        violations.extend(
            self.validate_verb_constraints(
                row
            )
        )

        # Only rerender if the basic structure did
        # not already prove the row unusable.
        blocking_types = {
            "UNKNOWN_FRAME",
            "DISABLED_FRAME",
            "INVALID_SLOTS",
            "INVALID_FEATURES",
            "INVALID_TEXTS",
        }

        existing_types = {
            item.get(
                "type"
            )
            for item in violations
        }

        if not (
            existing_types
            & blocking_types
        ):

            violations.extend(
                self.validate_rerender(
                    row
                )
            )

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
            "Hard Semantic Validator V0.4"
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
        HardSemanticValidatorV04(
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

    violation_counter = Counter()

    frame_counter = Counter()

    verb_counter = Counter()

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

        frame_id = row.get(
            "frame_id"
        )

        if frame_id:

            frame_counter[
                frame_id
            ] += 1

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

    accept_rate = (
        accepted / total
        if total
        else 0.0
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
            accept_rate,

        "violation_counts":
            dict(
                violation_counter.most_common()
            ),

        "frame_distribution":
            dict(
                frame_counter.most_common()
            ),

        "verb_distribution":
            dict(
                verb_counter.most_common()
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
        "HARD SEMANTIC VALIDATOR V0.4"
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

    if violation_counter:

        for name, count in (
            violation_counter.most_common()
        ):

            print(
                f"{name:<40}"
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