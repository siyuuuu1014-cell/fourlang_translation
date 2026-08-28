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
# Paths
# ============================================================

V04_RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

V05_RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

DEFAULT_CONCEPTS = (
    V04_RESOURCE_DIR
    / "concepts_v044.jsonl"
)

DEFAULT_VERB_REALIZATION = (
    V05_RESOURCE_DIR
    / "verb_realization_v051.json"
)

DEFAULT_ARGUMENT_REALIZATION = (
    V05_RESOURCE_DIR
    / "argument_realization_v051.json"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
    / "02_hard_semantic"
    / "hard_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
    / "03_grammar_hard"
)


# ============================================================
# V0.5.1 activation
# ============================================================

NEW_FRAME_TO_VERB = {
    "SEE_OBJECT": "SEE",
    "TAKE_OBJECT": "TAKE",
}

NEW_FRAMES = set(
    NEW_FRAME_TO_VERB
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

def read_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing JSON: {path}"
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
            f"Missing JSONL: {path}"
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
                    f"Invalid JSONL "
                    f"{path}:{line_no}"
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


# ============================================================
# Classification
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
# Frozen V0.4.4.1 validator
# ============================================================

def run_frozen_validator(
    *,
    rows: list[dict],
    output_dir: Path,
    concepts_path: Path,
) -> list[dict]:

    if not rows:
        return []

    frozen_input = (
        output_dir
        / "_frozen_core_input.jsonl"
    )

    frozen_output = (
        output_dir
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
        "scripts.synthetic.grammar_hard_validator_v0441",

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
    print("RUNNING FROZEN V0.4.4.1 GRAMMAR HARD")
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
            "Frozen grammar validator failed "
            f"with return code "
            f"{result.returncode}"
        )

    judged_file = (
        frozen_output
        / "grammar_judged.jsonl"
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
            "Frozen grammar validator row "
            f"count mismatch: "
            f"input={len(rows)}, "
            f"output={len(judged_rows)}"
        )

    return judged_rows


# ============================================================
# Frozen result helpers
# ============================================================
def _find_final_grammar_validation(
    row: dict,
) -> dict | None:
    """
    Locate the final grammar decision.

    Important:
    Frozen V0.4.4.1 rows can retain an older V0.4
    grammar_validation result while also carrying a later
    V0.4.4.1 override/final decision.

    We must always prefer the newest/final validation layer.
    """

    # --------------------------------------------------------
    # 1. Prefer explicitly versioned / final V0.4.4.1 fields.
    # --------------------------------------------------------

    preferred_keys = (
        "grammar_validation_v0441",
        "grammar_hard_validation_v0441",
        "grammar_v0441_validation",
        "grammar_validation_final",
        "final_grammar_validation",
    )

    for key in preferred_keys:

        value = row.get(key)

        if (
            isinstance(value, dict)
            and "accept" in value
        ):
            return value

    # --------------------------------------------------------
    # 2. Search grammar-related dictionaries.
    #
    # Prefer keys containing:
    # v0441 / 0441 / final / override
    # --------------------------------------------------------

    candidates = []

    for key, value in row.items():

        if not isinstance(value, dict):
            continue

        key_lower = key.lower()

        if (
            "grammar" not in key_lower
            or "accept" not in value
        ):
            continue

        score = 0

        if "v0441" in key_lower:
            score += 100

        if "0441" in key_lower:
            score += 80

        if "final" in key_lower:
            score += 60

        if "override" in key_lower:
            score += 50

        if "adjust" in key_lower:
            score += 40

        if key == "grammar_validation":
            score += 1

        candidates.append(
            (
                score,
                key,
                value,
            )
        )

    if candidates:

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates[0][2]

    return None


def extract_grammar_accept(
    row: dict,
) -> bool:

    validation = (
        _find_final_grammar_validation(
            row
        )
    )

    if validation is None:

        raise RuntimeError(
            "Cannot locate final grammar validation "
            f"for {row.get('semantic_id')}"
        )

    return bool(
        validation.get(
            "accept"
        )
    )


def extract_grammar_violations(
    row: dict,
) -> list[dict]:

    validation = (
        _find_final_grammar_validation(
            row
        )
    )

    if validation is None:
        return []

    violations = validation.get(
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
# New V0.5.1 grammar
# ============================================================

class V051GrammarChecker:

    def __init__(
        self,
        *,
        verb_resource: dict,
        argument_resource: dict,
    ) -> None:

        self.verbs = verb_resource.get(
            "verbs",
            {},
        )

        self.subjects = argument_resource.get(
            "subjects",
            {},
        )

        self.objects = argument_resource.get(
            "objects",
            {},
        )

    # --------------------------------------------------------
    # Verb forms
    # --------------------------------------------------------

    def english_verb(
        self,
        *,
        verb_id: str,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        policy = self.verbs[
            verb_id
        ][
            "en"
        ]

        if tense == "future":

            if polarity == "pos":
                return policy[
                    "future"
                ]

            return policy[
                "future_negative"
            ]

        if polarity == "pos":

            return policy[
                "present"
            ][
                person_code
            ]

        return policy[
            "present_negative"
        ][
            person_code
        ]

    def russian_verb(
        self,
        *,
        verb_id: str,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        policy = self.verbs[
            verb_id
        ][
            "ru"
        ]

        surface = policy[
            tense
        ][
            person_code
        ]

        if polarity == "neg":

            return (
                "не "
                + surface
            )

        return surface

    def uzbek_verb(
        self,
        *,
        verb_id: str,
        person_code: str,
        polarity: str,
    ) -> str:

        policy = self.verbs[
            verb_id
        ][
            "uz"
        ]

        group = (
            "present_future"
            if polarity == "pos"
            else "negative"
        )

        return policy[
            group
        ][
            person_code
        ]

    def chinese_verb(
        self,
        *,
        verb_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        policy = self.verbs[
            verb_id
        ][
            "zh"
        ]

        key = (
            f"{tense}_"
            f"{'positive' if polarity == 'pos' else 'negative'}"
        )

        return policy[
            key
        ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate(
        self,
        row: dict,
    ) -> tuple[
        bool,
        list[dict],
    ]:

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

        verb_id = slots.get(
            "verb"
        )

        subject_id = slots.get(
            "subject"
        )

        object_id = slots.get(
            "object"
        )

        # ====================================================
        # Activation
        # ====================================================

        if frame_id in BLOCKED_FRAMES:

            violations.append({
                "type":
                    "BLOCKED_FRAME",
            })

        if verb_id in BLOCKED_VERBS:

            violations.append({
                "type":
                    "BLOCKED_VERB",
            })

        expected_verb = (
            NEW_FRAME_TO_VERB.get(
                frame_id
            )
        )

        if expected_verb is None:

            violations.append({
                "type":
                    "UNSUPPORTED_FRAME",
            })

            return (
                False,
                violations,
            )

        if verb_id != expected_verb:

            violations.append({
                "type":
                    "FRAME_VERB_MISMATCH",

                "expected":
                    expected_verb,

                "actual":
                    verb_id,
            })

        # ====================================================
        # Resources
        # ====================================================

        subject = self.subjects.get(
            subject_id
        )

        if subject is None:

            violations.append({
                "type":
                    "UNKNOWN_SUBJECT",

                "subject":
                    subject_id,
            })

        obj = self.objects.get(
            object_id
        )

        if obj is None:

            violations.append({
                "type":
                    "UNKNOWN_OBJECT",

                "object":
                    object_id,
            })

        if verb_id not in self.verbs:

            violations.append({
                "type":
                    "UNKNOWN_VERB",

                "verb":
                    verb_id,
            })

        if violations:

            return (
                False,
                violations,
            )

        # ====================================================
        # Features
        # ====================================================

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

        if violations:

            return (
                False,
                violations,
            )

        person_code = subject[
            "person_code"
        ]

        # ====================================================
        # Build independently expected grammatical forms
        # ====================================================

        zh_verb = self.chinese_verb(
            verb_id=verb_id,
            tense=tense,
            polarity=polarity,
        )

        en_verb = self.english_verb(
            verb_id=verb_id,
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        ru_verb = self.russian_verb(
            verb_id=verb_id,
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        uz_verb = self.uzbek_verb(
            verb_id=verb_id,
            person_code=person_code,
            polarity=polarity,
        )

        expected = {
            "zh":
                (
                    f"{subject['zh']}"
                    f"{zh_verb}"
                    f"{obj['zh']}。"
                ),

            "en":
                (
                    f"{subject['en']} "
                    f"{en_verb} "
                    f"{obj['en']}."
                ),

            "ru":
                (
                    f"{subject['ru']} "
                    f"{ru_verb} "
                    f"{obj['ru_acc']}."
                ),

            "uz":
                (
                    f"{subject['uz']} "
                    f"{obj['uz_acc']} "
                    f"{uz_verb}."
                ),
        }

        # ====================================================
        # Independent grammatical checks
        # ====================================================

        for language in (
            "zh",
            "en",
            "ru",
            "uz",
        ):

            actual = texts.get(
                language
            )

            expected_text = expected[
                language
            ]

            if actual != expected_text:

                violations.append({
                    "type":
                        "GRAMMAR_FORM_MISMATCH",

                    "language":
                        language,

                    "expected":
                        expected_text,

                    "actual":
                        actual,
                })

        return (
            len(violations) == 0,
            violations,
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Grammar Hard Validator V0.5.1"
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
        "--verbs",
        default=str(
            DEFAULT_VERB_REALIZATION
        ),
    )

    parser.add_argument(
        "--arguments",
        default=str(
            DEFAULT_ARGUMENT_REALIZATION
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

    rows = read_jsonl(
        input_path
    )

    # ========================================================
    # Split old/new
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
    print("GRAMMAR HARD VALIDATOR V0.5.1")
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
    # Frozen
    # ========================================================

    frozen_judged = run_frozen_validator(
        rows=frozen_rows,
        output_dir=output_dir,
        concepts_path=Path(
            args.concepts
        ),
    )
    frozen_wrapper_rejected = [
        row
        for row in frozen_judged
        if not extract_grammar_accept(row)
    ]

    if frozen_wrapper_rejected:

        print()
        print(
            "WARNING: final frozen grammar rows "
            "still rejected:",
            len(frozen_wrapper_rejected),
        )

    frozen_map = {
        row.get(
            "semantic_id"
        ): row
        for row in frozen_judged
    }

    # ========================================================
    # New
    # ========================================================

    checker = V051GrammarChecker(
        verb_resource=read_json(
            Path(
                args.verbs
            )
        ),

        argument_resource=read_json(
            Path(
                args.arguments
            )
        ),
    )

    new_map = {}

    for row in new_rows:

        output_row = dict(
            row
        )

        accept, violations = (
            checker.validate(
                row
            )
        )

        output_row[
            "grammar_validation"
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
                "Missing grammar result for "
                f"{semantic_id}"
            )

    # ========================================================
    # Stats
    # ========================================================

    accepted_rows = []
    rejected_rows = []

    reject_reasons = Counter()
    rejected_by_language = Counter()

    new_accepted = 0
    new_rejected = 0

    for row in judged_rows:

        accept = extract_grammar_accept(
            row
        )

        violations = (
            extract_grammar_violations(
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

                if not isinstance(
                    violation,
                    dict,
                ):
                    continue

                violation_type = (
                    violation.get(
                        "type",
                        "UNKNOWN",
                    )
                )

                reject_reasons[
                    violation_type
                ] += 1

                language = violation.get(
                    "language"
                )

                if language:

                    rejected_by_language[
                        language
                    ] += 1

        if row.get(
            "frame_id"
        ) in NEW_FRAMES:

            if accept:
                new_accepted += 1
            else:
                new_rejected += 1

    # ========================================================
    # Save
    # ========================================================

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
            new_accepted,

        "new_rejected":
            new_rejected,

        "reject_reasons":
            dict(
                reject_reasons
            ),

        "rejected_by_language":
            dict(
                rejected_by_language
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
    print("V0.5.1 GRAMMAR HARD VALIDATION COMPLETE")
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
        new_accepted,
    )

    print(
        "Rejected:",
        new_rejected,
    )

    print()

    print(
        "Rejected by language:"
    )

    if rejected_by_language:

        for language, count in (
            rejected_by_language.items()
        ):

            print(
                f"{language:<10}{count}"
            )

    else:

        print(
            "None"
        )

    print()

    print(
        "Reject reasons:"
    )

    if reject_reasons:

        for reason, count in (
            reject_reasons.most_common()
        ):

            print(
                f"{reason:<35}{count}"
            )

    else:

        print(
            "None"
        )

    print()

    print("Files:")
    print(judged_file)
    print(accepted_file)
    print(rejected_file)
    print(summary_file)


if __name__ == "__main__":
    main()