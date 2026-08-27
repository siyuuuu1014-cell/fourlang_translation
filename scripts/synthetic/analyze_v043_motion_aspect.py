from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v043_regression_200"
    / "03_grammar_hard"
    / "grammar_accepted.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v043_motion_present_analysis.jsonl"
)

DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v043_motion_present_analysis_summary.json"
)


TARGET_VERBS = {
    "GO",
    "COME",
}

MOTION_FRAMES = {
    "INTRANSITIVE",
    "INTRANSITIVE_TIME",
    "MOTION_DESTINATION",
    "MOTION_TIME",
    "MOTION_CLOCK",
}


def read_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
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
                    f"Invalid JSONL at "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise RuntimeError(
                    f"{path}:{line_no} "
                    "must be JSON object."
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


def get_time_id(
    row: dict,
) -> str | None:

    slots = row.get(
        "slots",
        {},
    )

    return (
        slots.get("time")
        or slots.get("day")
    )


def classify_present_motion(
    row: dict,
) -> dict:

    frame_id = row.get(
        "frame_id"
    )

    slots = row.get(
        "slots",
        {},
    )

    time_id = get_time_id(
        row
    )

    destination = slots.get(
        "destination"
    )

    # --------------------------------------------------------
    # TIME_NOW
    #
    # Example:
    #
    # I come to the hospital now.
    #
    # English strongly prefers:
    #
    # I'm coming to the hospital now.
    #
    # Therefore this is a clear aspect mismatch candidate.
    # --------------------------------------------------------

    if time_id == "TIME_NOW":

        return {
            "risk_level": "HIGH",

            "risk_type":
                "PRESENT_NOW_SIMPLE",

            "reason":
                (
                    "Present motion with TIME_NOW "
                    "is likely to require an "
                    "ongoing/progressive interpretation."
                ),

            "suggested_aspect":
                "progressive",
        }

    # --------------------------------------------------------
    # No temporal cue
    #
    # Examples:
    #
    # He goes.
    # We come to the bank.
    #
    # Grammatically possible, but weak / underspecified
    # as standalone translation training sentences.
    # --------------------------------------------------------

    if time_id is None:

        if frame_id == "INTRANSITIVE":

            return {
                "risk_level": "HIGH",

                "risk_type":
                    "BARE_MOTION_SIMPLE_PRESENT",

                "reason":
                    (
                        "Bare motion simple present "
                        "without a habitual or current "
                        "event cue is context-poor."
                    ),

                "suggested_aspect":
                    "habitual_or_progressive",
            }

        if (
            frame_id
            == "MOTION_DESTINATION"
            and destination
        ):

            return {
                "risk_level": "HIGH",

                "risk_type":
                    "DESTINATION_SIMPLE_PRESENT_NO_CUE",

                "reason":
                    (
                        "Motion-to-destination simple "
                        "present without a temporal or "
                        "habitual cue is often unnatural "
                        "as a standalone utterance."
                    ),

                "suggested_aspect":
                    "habitual_or_progressive",
            }

        return {
            "risk_level": "MEDIUM",

            "risk_type":
                "PRESENT_WITHOUT_ASPECT_CUE",

            "reason":
                (
                    "Present motion lacks an explicit "
                    "aspect/event interpretation."
                ),

            "suggested_aspect":
                "habitual_or_progressive",
        }

    # --------------------------------------------------------
    # TODAY
    #
    # "I go to the bank today" is possible in some contexts,
    # but for isolated synthetic training data it is ambiguous:
    #
    # scheduled?
    # current activity?
    # one-off event?
    # --------------------------------------------------------

    if time_id == "TIME_TODAY":

        return {
            "risk_level": "MEDIUM",

            "risk_type":
                "TODAY_PRESENT_AMBIGUOUS",

            "reason":
                (
                    "TIME_TODAY with simple-present "
                    "motion is ambiguous between "
                    "scheduled, progressive and "
                    "one-off event interpretations."
                ),

            "suggested_aspect":
                "scheduled_or_progressive",
        }

    # --------------------------------------------------------
    # Other present temporal concepts
    # --------------------------------------------------------

    return {
        "risk_level": "MEDIUM",

        "risk_type":
            "OTHER_PRESENT_MOTION",

        "reason":
            (
                "Present motion requires explicit "
                "aspect interpretation before scaling."
            ),

        "suggested_aspect":
            "review",
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze V0.4.3 GO/COME present "
            "motion aspect risks."
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--summary",
        default=str(
            DEFAULT_SUMMARY
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    summary_path = Path(
        args.summary
    )

    rows = read_jsonl(
        input_path
    )

    analyzed = []

    verb_counter = Counter()
    frame_counter = Counter()
    time_counter = Counter()
    polarity_counter = Counter()
    subject_counter = Counter()

    risk_counter = Counter()
    risk_type_counter = Counter()

    verb_frame_counter = Counter()
    verb_time_counter = Counter()
    verb_risk_counter = Counter()

    for row in rows:

        frame_id = row.get(
            "frame_id"
        )

        if frame_id not in MOTION_FRAMES:
            continue

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

        if verb_id not in TARGET_VERBS:
            continue

        tense = features.get(
            "tense"
        )

        if tense != "present":
            continue

        polarity = features.get(
            "polarity"
        )

        subject = slots.get(
            "subject"
        )

        time_id = get_time_id(
            row
        )

        classification = (
            classify_present_motion(
                row
            )
        )

        output_row = {
            "semantic_id":
                row.get(
                    "semantic_id"
                ),

            "scenario":
                row.get(
                    "scenario"
                ),

            "frame_id":
                frame_id,

            "slots":
                slots,

            "features":
                features,

            "computed":
                row.get(
                    "computed",
                    {},
                ),

            "texts":
                row.get(
                    "texts",
                    {},
                ),

            "motion_aspect_analysis":
                classification,
        }

        analyzed.append(
            output_row
        )

        verb_counter[
            verb_id
        ] += 1

        frame_counter[
            frame_id
        ] += 1

        time_counter[
            time_id or "NO_TIME"
        ] += 1

        polarity_counter[
            polarity or "NONE"
        ] += 1

        subject_counter[
            subject or "NONE"
        ] += 1

        risk = classification[
            "risk_level"
        ]

        risk_type = classification[
            "risk_type"
        ]

        risk_counter[
            risk
        ] += 1

        risk_type_counter[
            risk_type
        ] += 1

        verb_frame_counter[
            (
                verb_id,
                frame_id,
            )
        ] += 1

        verb_time_counter[
            (
                verb_id,
                time_id or "NO_TIME",
            )
        ] += 1

        verb_risk_counter[
            (
                verb_id,
                risk,
            )
        ] += 1

    summary = {
        "source_samples":
            len(
                rows
            ),

        "present_motion_samples":
            len(
                analyzed
            ),

        "target_verbs":
            sorted(
                TARGET_VERBS
            ),

        "verb_distribution":
            dict(
                verb_counter.most_common()
            ),

        "frame_distribution":
            dict(
                frame_counter.most_common()
            ),

        "time_distribution":
            dict(
                time_counter.most_common()
            ),

        "polarity_distribution":
            dict(
                polarity_counter.most_common()
            ),

        "subject_distribution":
            dict(
                subject_counter.most_common()
            ),

        "risk_distribution":
            dict(
                risk_counter.most_common()
            ),

        "risk_type_distribution":
            dict(
                risk_type_counter.most_common()
            ),

        "verb_frame_distribution": {
            f"{verb}|{frame}":
                count
            for (
                verb,
                frame,
            ), count in (
                verb_frame_counter
                .most_common()
            )
        },

        "verb_time_distribution": {
            f"{verb}|{time_id}":
                count
            for (
                verb,
                time_id,
            ), count in (
                verb_time_counter
                .most_common()
            )
        },

        "verb_risk_distribution": {
            f"{verb}|{risk}":
                count
            for (
                verb,
                risk,
            ), count in (
                verb_risk_counter
                .most_common()
            )
        },
    }

    write_jsonl(
        output_path,
        analyzed,
    )

    write_json(
        summary_path,
        summary,
    )

    print(
        "=" * 90
    )

    print(
        "V0.4.3 PRESENT MOTION ASPECT ANALYSIS"
    )

    print(
        "=" * 90
    )

    print(
        "Source samples:",
        len(
            rows
        ),
    )

    print(
        "GO/COME present samples:",
        len(
            analyzed
        ),
    )

    print()

    print(
        "Verb distribution:"
    )

    print(
        "-" * 60
    )

    for key, value in (
        verb_counter.most_common()
    ):

        print(
            f"{key:<20}"
            f"{value}"
        )

    print()

    print(
        "Frame distribution:"
    )

    print(
        "-" * 60
    )

    for key, value in (
        frame_counter.most_common()
    ):

        print(
            f"{key:<30}"
            f"{value}"
        )

    print()

    print(
        "Time distribution:"
    )

    print(
        "-" * 60
    )

    for key, value in (
        time_counter.most_common()
    ):

        print(
            f"{key:<30}"
            f"{value}"
        )

    print()

    print(
        "Risk distribution:"
    )

    print(
        "-" * 60
    )

    for key, value in (
        risk_counter.most_common()
    ):

        print(
            f"{key:<20}"
            f"{value}"
        )

    print()

    print(
        "Risk types:"
    )

    print(
        "-" * 60
    )

    for key, value in (
        risk_type_counter.most_common()
    ):

        print(
            f"{key:<40}"
            f"{value}"
        )

    print()

    print(
        "Files:"
    )

    print(
        output_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()