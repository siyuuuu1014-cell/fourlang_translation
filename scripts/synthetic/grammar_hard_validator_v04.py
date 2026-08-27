from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


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

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v04_smoke_100"
    / "02_hard_semantic"
    / "hard_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v04_smoke_100"
    / "03_grammar_hard"
)


LANGUAGES = (
    "zh",
    "en",
    "ru",
    "uz",
)


RU_FUTURE_AUX = {
    "1sg": "буду",
    "2sg": "будешь",
    "3sg": "будет",
    "1pl": "будем",
    "2pl": "будете",
    "3pl": "будут",
}


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
                    f"Invalid JSONL "
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
# Text helpers
# ============================================================

def normalize_text(
    text: Any,
) -> str:

    if text is None:
        return ""

    text = str(
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_surface(
    text: str,
    surface: str,
) -> bool:
    """
    Boundary-aware surface matching.

    Important:
    "go" must NOT match "goes".
    """

    text = normalize_text(
        text
    )

    surface = normalize_text(
        surface
    )

    if not surface:
        return False

    pattern = (
        r"(?<![\w'’ʻ])"
        + re.escape(
            surface
        )
        + r"(?![\w'’ʻ])"
    )

    return (
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


def contains_zh_surface(
    text: str,
    surface: str,
) -> bool:

    return (
        normalize_text(
            surface
        )
        in normalize_text(
            text
        )
    )


# ============================================================
# Validator
# ============================================================

class GrammarHardValidatorV04:

    def __init__(
        self,
        *,
        concepts_path: Path,
    ) -> None:

        rows = read_jsonl(
            concepts_path
        )

        self.concepts = {
            row["id"]: row
            for row in rows
        }

    # ========================================================
    # Resource helpers
    # ========================================================

    def concept(
        self,
        concept_id: str,
    ) -> dict | None:

        return self.concepts.get(
            concept_id
        )

    def person_code(
        self,
        subject_id: str,
        lang: str,
    ) -> str | None:

        subject = self.concept(
            subject_id
        )

        if subject is None:
            return None

        forms = subject.get(
            "forms",
            {},
        )

        lang_forms = forms.get(
            lang,
            {},
        )

        if isinstance(
            lang_forms,
            dict,
        ):

            code = lang_forms.get(
                "person_code"
            )

            if code:
                return str(
                    code
                )

        features = subject.get(
            "person_features",
            {},
        )

        person = features.get(
            "person"
        )

        number = features.get(
            "number"
        )

        if person not in {
            1,
            2,
            3,
        }:
            return None

        if number == "singular":

            return (
                f"{person}sg"
            )

        if number == "plural":

            return (
                f"{person}pl"
            )

        return None

    # ========================================================
    # Expected Chinese surface
    # ========================================================

    def expected_zh_verb(
        self,
        *,
        verb: dict,
        tense: str,
        polarity: str,
    ) -> str | None:

        forms = (
            verb.get(
                "forms",
                {}
            ).get(
                "zh",
                {}
            )
        )

        if not isinstance(
            forms,
            dict,
        ):
            return None

        base = forms.get(
            "base"
        )

        if not base:
            return None

        if tense == "future":

            if polarity == "neg":

                return str(
                    forms.get(
                        "future_negative"
                    )
                    or (
                        "不会"
                        + str(base)
                    )
                )

            return str(
                forms.get(
                    "future_positive"
                )
                or (
                    "会"
                    + str(base)
                )
            )

        if polarity == "neg":

            return str(
                forms.get(
                    "present_negative"
                )
                or forms.get(
                    "negative"
                )
                or (
                    "不"
                    + str(base)
                )
            )

        return str(
            forms.get(
                "present_positive"
            )
            or forms.get(
                "positive"
            )
            or base
        )

    # ========================================================
    # Expected English surface
    # ========================================================

    def expected_en_verb(
        self,
        *,
        verb: dict,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str | None:

        forms = (
            verb.get(
                "forms",
                {}
            ).get(
                "en",
                {}
            )
        )

        if not isinstance(
            forms,
            dict,
        ):
            return None

        base = forms.get(
            "base"
        )

        if not base:
            return None

        base = str(
            base
        )

        if tense == "future":

            if polarity == "neg":

                return (
                    f"will not {base}"
                )

            return (
                f"will {base}"
            )

        third_singular = (
            person_code
            == "3sg"
        )

        if polarity == "neg":

            if third_singular:

                return (
                    f"does not {base}"
                )

            return (
                f"do not {base}"
            )

        if third_singular:

            return str(
                forms.get(
                    "present_3sg"
                )
                or (
                    base + "s"
                )
            )

        return base

    # ========================================================
    # Russian
    # ========================================================

    def ru_present(
        self,
        forms: dict,
        person_code: str,
    ) -> str | None:

        present = forms.get(
            "present",
            {},
        )

        if isinstance(
            present,
            dict,
        ):

            value = present.get(
                person_code
            )

            if value:

                return str(
                    value
                )

        imperfective = forms.get(
            "imperfective",
            {},
        )

        if isinstance(
            imperfective,
            dict,
        ):

            value = imperfective.get(
                f"present_{person_code}"
            )

            if value:

                return str(
                    value
                )

        value = forms.get(
            f"present_{person_code}"
        )

        if value:

            return str(
                value
            )

        return None

    def ru_future_perfective(
        self,
        forms: dict,
        person_code: str,
    ) -> str | None:

        perfective = forms.get(
            "perfective",
            {},
        )

        if not isinstance(
            perfective,
            dict,
        ):
            return None

        value = perfective.get(
            f"future_{person_code}"
        )

        if value:

            return str(
                value
            )

        future = perfective.get(
            "future",
            {},
        )

        if isinstance(
            future,
            dict,
        ):

            value = future.get(
                person_code
            )

            if value:

                return str(
                    value
                )

        return None

    def expected_ru_verb(
        self,
        *,
        verb: dict,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str | None:

        forms = (
            verb.get(
                "forms",
                {}
            ).get(
                "ru",
                {}
            )
        )

        if not isinstance(
            forms,
            dict,
        ):
            return None

        infinitive = forms.get(
            "infinitive"
        )

        if tense == "present":

            surface = self.ru_present(
                forms,
                person_code,
            )

            if not surface:
                return None

            if polarity == "neg":

                return (
                    f"не {surface}"
                )

            return surface

        strategy = forms.get(
            "future_strategy",
            "analytic",
        )

        if strategy == "perfective":

            surface = (
                self.ru_future_perfective(
                    forms,
                    person_code,
                )
            )

            if not surface:
                return None

            if polarity == "neg":

                return (
                    f"не {surface}"
                )

            return surface

        if strategy == "analytic":

            auxiliary = (
                RU_FUTURE_AUX.get(
                    person_code
                )
            )

            if (
                not auxiliary
                or not infinitive
            ):

                return None

            if polarity == "neg":

                return (
                    f"не {auxiliary} "
                    f"{infinitive}"
                )

            return (
                f"{auxiliary} "
                f"{infinitive}"
            )

        return None

    # ========================================================
    # Uzbek
    # ========================================================

    def expected_uz_verb(
        self,
        *,
        verb: dict,
        person_code: str,
        polarity: str,
    ) -> str | None:

        forms = (
            verb.get(
                "forms",
                {}
            ).get(
                "uz",
                {}
            )
        )

        if not isinstance(
            forms,
            dict,
        ):
            return None

        table_name = (
            "negative_present_future"
            if polarity == "neg"
            else "present_future"
        )

        table = forms.get(
            table_name,
            {},
        )

        if not isinstance(
            table,
            dict,
        ):
            return None

        value = table.get(
            person_code
        )

        # Deliberate grammar policy:
        #
        # Uzbek plural subject may use explicit
        # plural verb form or, in some contexts,
        # a 3sg finite surface.
        #
        # However the generated resource currently
        # contains explicit plural forms, so prefer
        # exact form first.
        if (
            not value
            and person_code == "3pl"
        ):

            value = table.get(
                "3sg"
            )

        if not value:
            return None

        return str(
            value
        )

    # ========================================================
    # Language checks
    # ========================================================

    def validate_zh(
        self,
        *,
        row: dict,
        verb: dict,
        tense: str,
        polarity: str,
    ) -> list[dict]:

        text = (
            row.get(
                "texts",
                {}
            ).get(
                "zh",
                "",
            )
        )

        expected = (
            self.expected_zh_verb(
                verb=verb,
                tense=tense,
                polarity=polarity,
            )
        )

        if not expected:

            return [{
                "type":
                    "ZH_EXPECTED_FORM_MISSING",

                "verb":
                    verb.get(
                        "id"
                    ),
            }]

        if not contains_zh_surface(
            text,
            expected,
        ):

            return [{
                "type":
                    "ZH_VERB_FORM_MISMATCH",

                "verb":
                    verb.get(
                        "id"
                    ),

                "expected":
                    expected,

                "text":
                    text,
            }]

        return []

    def validate_en(
        self,
        *,
        row: dict,
        verb: dict,
        subject_id: str,
        tense: str,
        polarity: str,
    ) -> list[dict]:

        violations = []

        person_code = self.person_code(
            subject_id,
            "en",
        )

        if not person_code:

            return [{
                "type":
                    "EN_PERSON_CODE_MISSING",

                "subject":
                    subject_id,
            }]

        expected = (
            self.expected_en_verb(
                verb=verb,
                person_code=person_code,
                tense=tense,
                polarity=polarity,
            )
        )

        if not expected:

            return [{
                "type":
                    "EN_EXPECTED_FORM_MISSING",

                "verb":
                    verb.get(
                        "id"
                    ),
            }]

        text = (
            row.get(
                "texts",
                {}
            ).get(
                "en",
                "",
            )
        )

        if not contains_surface(
            text,
            expected,
        ):

            violations.append({
                "type":
                    "EN_AGREEMENT_OR_TENSE",

                "subject":
                    subject_id,

                "person_code":
                    person_code,

                "verb":
                    verb.get(
                        "id"
                    ),

                "tense":
                    tense,

                "polarity":
                    polarity,

                "expected":
                    expected,

                "text":
                    text,
            })

        return violations

    def validate_ru(
        self,
        *,
        row: dict,
        verb: dict,
        subject_id: str,
        tense: str,
        polarity: str,
    ) -> list[dict]:

        person_code = self.person_code(
            subject_id,
            "ru",
        )

        if not person_code:

            return [{
                "type":
                    "RU_PERSON_CODE_MISSING",

                "subject":
                    subject_id,
            }]

        expected = (
            self.expected_ru_verb(
                verb=verb,
                person_code=person_code,
                tense=tense,
                polarity=polarity,
            )
        )

        if not expected:

            return [{
                "type":
                    "RU_EXPECTED_FORM_MISSING",

                "verb":
                    verb.get(
                        "id"
                    ),

                "person_code":
                    person_code,

                "tense":
                    tense,
            }]

        text = (
            row.get(
                "texts",
                {}
            ).get(
                "ru",
                "",
            )
        )

        if not contains_surface(
            text,
            expected,
        ):

            return [{
                "type":
                    "RU_AGREEMENT_OR_TENSE",

                "subject":
                    subject_id,

                "person_code":
                    person_code,

                "verb":
                    verb.get(
                        "id"
                    ),

                "tense":
                    tense,

                "polarity":
                    polarity,

                "expected":
                    expected,

                "text":
                    text,
            }]

        return []

    def validate_uz(
        self,
        *,
        row: dict,
        verb: dict,
        subject_id: str,
        polarity: str,
    ) -> list[dict]:

        person_code = self.person_code(
            subject_id,
            "uz",
        )

        if not person_code:

            return [{
                "type":
                    "UZ_PERSON_CODE_MISSING",

                "subject":
                    subject_id,
            }]

        expected = (
            self.expected_uz_verb(
                verb=verb,
                person_code=person_code,
                polarity=polarity,
            )
        )

        if not expected:

            return [{
                "type":
                    "UZ_EXPECTED_FORM_MISSING",

                "verb":
                    verb.get(
                        "id"
                    ),

                "person_code":
                    person_code,
            }]

        text = (
            row.get(
                "texts",
                {}
            ).get(
                "uz",
                "",
            )
        )

        if not contains_surface(
            text,
            expected,
        ):

            return [{
                "type":
                    "UZ_AGREEMENT",

                "subject":
                    subject_id,

                "person_code":
                    person_code,

                "verb":
                    verb.get(
                        "id"
                    ),

                "polarity":
                    polarity,

                "expected":
                    expected,

                "text":
                    text,
            }]

        return []

    # ========================================================
    # Whole sample
    # ========================================================

    def validate_sample(
        self,
        row: dict,
    ) -> dict:

        semantic_id = row.get(
            "semantic_id"
        )

        slots = row.get(
            "slots",
            {},
        )

        features = row.get(
            "features",
            {},
        )

        violations = []

        # ----------------------------------------------------
        # WHERE_PLACE and other verbless frames
        # ----------------------------------------------------

        verb_id = slots.get(
            "verb"
        )

        if not verb_id:

            return {
                "semantic_id":
                    semantic_id,

                "accept":
                    True,

                "violations":
                    [],
            }

        subject_id = slots.get(
            "subject"
        )

        if not subject_id:

            return {
                "semantic_id":
                    semantic_id,

                "accept":
                    False,

                "violations": [
                    {
                        "type":
                            "SUBJECT_MISSING"
                    }
                ],
            }

        verb = self.concept(
            verb_id
        )

        if verb is None:

            return {
                "semantic_id":
                    semantic_id,

                "accept":
                    False,

                "violations": [
                    {
                        "type":
                            "UNKNOWN_VERB",

                        "verb":
                            verb_id,
                    }
                ],
            }

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
                    "INVALID_TENSE",

                "value":
                    tense,
            })

        if polarity not in {
            "pos",
            "neg",
        }:

            violations.append({
                "type":
                    "INVALID_POLARITY",

                "value":
                    polarity,
            })

        if violations:

            return {
                "semantic_id":
                    semantic_id,

                "accept":
                    False,

                "violations":
                    violations,
            }

        # ----------------------------------------------------
        # Four deterministic grammar checks
        # ----------------------------------------------------

        violations.extend(
            self.validate_zh(
                row=row,
                verb=verb,
                tense=tense,
                polarity=polarity,
            )
        )

        violations.extend(
            self.validate_en(
                row=row,
                verb=verb,
                subject_id=subject_id,
                tense=tense,
                polarity=polarity,
            )
        )

        violations.extend(
            self.validate_ru(
                row=row,
                verb=verb,
                subject_id=subject_id,
                tense=tense,
                polarity=polarity,
            )
        )

        violations.extend(
            self.validate_uz(
                row=row,
                verb=verb,
                subject_id=subject_id,
                polarity=polarity,
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
            "Grammar Hard Validator V0.4"
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    rows = read_jsonl(
        input_path
    )

    validator = (
        GrammarHardValidatorV04(
            concepts_path=Path(
                args.concepts
            ),
        )
    )

    judged_rows = []
    accepted_rows = []
    rejected_rows = []

    violation_counter = Counter()
    language_counter = Counter()
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
            "grammar_hard_validation"
        ] = result

        judged_rows.append(
            output_row
        )

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

                violation_type = (
                    violation.get(
                        "type",
                        "UNKNOWN",
                    )
                )

                violation_counter[
                    violation_type
                ] += 1

                if violation_type.startswith(
                    "ZH_"
                ):

                    language_counter[
                        "zh"
                    ] += 1

                elif violation_type.startswith(
                    "EN_"
                ):

                    language_counter[
                        "en"
                    ] += 1

                elif violation_type.startswith(
                    "RU_"
                ):

                    language_counter[
                        "ru"
                    ] += 1

                elif violation_type.startswith(
                    "UZ_"
                ):

                    language_counter[
                        "uz"
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

        "rejected_by_language":
            dict(
                language_counter
                .most_common()
            ),

        "reject_reasons":
            dict(
                violation_counter
                .most_common()
            ),

        "verb_distribution":
            dict(
                verb_counter
                .most_common()
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
        "GRAMMAR HARD VALIDATOR V0.4"
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
        "Rejected by language:"
    )

    print(
        "-" * 70
    )

    if language_counter:

        for lang, count in (
            language_counter.most_common()
        ):

            print(
                f"{lang:<10}{count}"
            )

    else:

        print(
            "None"
        )

    print()

    print(
        "Reject reasons:"
    )

    print(
        "-" * 70
    )

    if violation_counter:

        for reason, count in (
            violation_counter.most_common()
        ):

            print(
                f"{reason:<40}{count}"
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