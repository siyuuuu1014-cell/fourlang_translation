from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "data/synthetic/audit/"
    "v042_1k_qwen_150/"
    "linguistic_rejected.jsonl"
)


def read_jsonl(path: Path) -> list[dict]:
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
                    f"Invalid JSON at "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            rows.append(row)

    return rows


def first_value(
    obj: Any,
    keys: tuple[str, ...],
) -> Any:

    if not isinstance(obj, dict):
        return None

    for key in keys:

        if key in obj:
            return obj[key]

    return None


def get_texts(row: dict) -> dict:
    texts = row.get(
        "texts",
        {},
    )

    if isinstance(texts, dict):
        return texts

    return {}


def get_slots(row: dict) -> dict:
    slots = row.get(
        "slots",
        {},
    )

    if isinstance(slots, dict):
        return slots

    return {}


def get_features(row: dict) -> dict:
    features = row.get(
        "features",
        {},
    )

    if isinstance(features, dict):
        return features

    return {}


def find_language_results(
    row: dict,
) -> list[tuple[str, dict]]:

    """
    Try several possible schemas used by
    linguistic judge output.
    """

    candidates = []

    possible_containers = (
        "linguistic_judgments",
        "language_judgments",
        "judgments",
        "linguistic_validation",
        "linguistic_judge",
        "judge_results",
        "language_results",
    )

    for key in possible_containers:

        container = row.get(key)

        if not container:
            continue

        # ---------------------------------------------
        # Schema:
        # {
        #   "zh": {...},
        #   "en": {...}
        # }
        # ---------------------------------------------

        if isinstance(container, dict):

            for lang in (
                "zh",
                "en",
                "ru",
                "uz",
            ):

                result = container.get(lang)

                if isinstance(result, dict):
                    candidates.append(
                        (
                            lang,
                            result,
                        )
                    )

        # ---------------------------------------------
        # Schema:
        # [
        #   {"language": "ru", ...}
        # ]
        # ---------------------------------------------

        elif isinstance(container, list):

            for item in container:

                if not isinstance(item, dict):
                    continue

                lang = first_value(
                    item,
                    (
                        "language",
                        "lang",
                    ),
                )

                if lang in {
                    "zh",
                    "en",
                    "ru",
                    "uz",
                }:

                    candidates.append(
                        (
                            lang,
                            item,
                        )
                    )

    # -------------------------------------------------
    # Sometimes judge fields may be stored directly
    # under row["judge"].
    # -------------------------------------------------

    judge = row.get(
        "judge"
    )

    if isinstance(judge, dict):

        lang = first_value(
            judge,
            (
                "language",
                "lang",
            ),
        )

        if lang in {
            "zh",
            "en",
            "ru",
            "uz",
        }:

            candidates.append(
                (
                    lang,
                    judge,
                )
            )

    # Remove duplicates
    unique = []

    seen = set()

    for lang, result in candidates:

        marker = (
            lang,
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

        if marker in seen:
            continue

        seen.add(marker)

        unique.append(
            (
                lang,
                result,
            )
        )

    return unique


def is_rejected(result: dict) -> bool:

    accepted = first_value(
        result,
        (
            "accept",
            "accepted",
            "is_accepted",
            "pass",
        ),
    )

    if isinstance(
        accepted,
        bool,
    ):
        return not accepted

    verdict = first_value(
        result,
        (
            "verdict",
            "decision",
            "label",
            "status",
        ),
    )

    if verdict is not None:

        verdict = str(
            verdict
        ).strip().lower()

        if verdict in {
            "reject",
            "rejected",
            "fail",
            "failed",
            "false",
        }:
            return True

        if verdict in {
            "accept",
            "accepted",
            "pass",
            "passed",
            "true",
        }:
            return False

    error_type = first_value(
        result,
        (
            "error_type",
            "type",
        ),
    )

    if error_type:

        value = str(
            error_type
        ).upper()

        if value not in {
            "NONE",
            "NO_ERROR",
            "OK",
            "PASS",
        }:
            return True

    return False


def extract_error_type(
    result: dict,
) -> str:

    value = first_value(
        result,
        (
            "error_type",
            "type",
            "category",
        ),
    )

    if value is None:
        return "UNKNOWN"

    return str(value)


def extract_reason(
    result: dict,
) -> str:

    value = first_value(
        result,
        (
            "reason",
            "explanation",
            "message",
            "detail",
            "feedback",
        ),
    )

    if value is None:
        return ""

    return str(value)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
    )

    args = parser.parse_args()

    path = Path(
        args.input
    )

    rows = read_jsonl(
        path
    )

    language_counter = Counter()
    error_counter = Counter()
    verb_counter = Counter()

    verb_language_counter = Counter()

    print(
        "=" * 100
    )
    print(
        "LINGUISTIC REJECT REVIEW"
    )
    print(
        "=" * 100
    )

    print(
        "Rejected samples:",
        len(rows),
    )

    for index, row in enumerate(
        rows,
        start=1,
    ):

        semantic_id = row.get(
            "semantic_id",
            "UNKNOWN",
        )

        frame_id = row.get(
            "frame_id",
            "UNKNOWN",
        )

        slots = get_slots(
            row
        )

        features = get_features(
            row
        )

        texts = get_texts(
            row
        )

        verb = slots.get(
            "verb",
            "NONE",
        )

        rejected_results = []

        for lang, result in (
            find_language_results(
                row
            )
        ):

            if is_rejected(
                result
            ):

                rejected_results.append(
                    (
                        lang,
                        result,
                    )
                )

                language_counter[
                    lang
                ] += 1

                error_type = (
                    extract_error_type(
                        result
                    )
                )

                error_counter[
                    error_type
                ] += 1

                verb_language_counter[
                    (
                        verb,
                        lang,
                    )
                ] += 1

        if rejected_results:

            verb_counter[
                verb
            ] += 1

        print()
        print(
            "=" * 100
        )

        print(
            f"[{index}/{len(rows)}] "
            f"{semantic_id}"
        )

        print(
            "-" * 100
        )

        print(
            "frame:",
            frame_id,
        )

        print(
            "scenario:",
            row.get(
                "scenario"
            ),
        )

        print(
            "verb:",
            verb,
        )

        print(
            "subject:",
            slots.get(
                "subject"
            ),
        )

        print(
            "object:",
            slots.get(
                "object"
            ),
        )

        print(
            "destination:",
            slots.get(
                "destination"
            ),
        )

        print(
            "time:",
            slots.get(
                "time"
            )
            or slots.get(
                "day"
            ),
        )

        print(
            "clock:",
            row.get(
                "computed",
                {},
            ).get(
                "clock"
            ),
        )

        print(
            "tense:",
            features.get(
                "tense"
            ),
        )

        print(
            "polarity:",
            features.get(
                "polarity"
            ),
        )

        print()
        print(
            "ZH:",
            texts.get(
                "zh",
                "",
            ),
        )

        print(
            "EN:",
            texts.get(
                "en",
                "",
            ),
        )

        print(
            "RU:",
            texts.get(
                "ru",
                "",
            ),
        )

        print(
            "UZ:",
            texts.get(
                "uz",
                "",
            ),
        )

        print()

        if not rejected_results:

            print(
                "WARNING: "
                "could not locate rejected "
                "language judgment schema."
            )

            print(
                "Available top-level keys:",
                list(
                    row.keys()
                ),
            )

        else:

            for lang, result in (
                rejected_results
            ):

                print(
                    f"REJECTED LANGUAGE: "
                    f"{lang}"
                )

                print(
                    "ERROR TYPE:",
                    extract_error_type(
                        result
                    ),
                )

                print(
                    "REASON:",
                    extract_reason(
                        result
                    ),
                )

                print(
                    "RAW JUDGMENT:",
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2,
                    ),
                )

                print()

    print()
    print(
        "=" * 100
    )
    print(
        "SUMMARY"
    )
    print(
        "=" * 100
    )

    print(
        "Rejected by language:",
        dict(
            language_counter
        ),
    )

    print(
        "Error types:",
        dict(
            error_counter
        ),
    )

    print(
        "Rejected samples by verb:",
        dict(
            verb_counter.most_common()
        ),
    )

    print()

    print(
        "Verb x Language:"
    )

    for (
        verb,
        lang,
    ), count in (
        verb_language_counter
        .most_common()
    ):

        print(
            f"{verb:<15} "
            f"{lang:<5} "
            f"{count}"
        )


if __name__ == "__main__":
    main()