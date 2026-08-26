from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CALIBRATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v1"
    / "linguistic_calibration_v01.jsonl"
)


DEFAULT_JUDGED_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v1"
    / "language_specific_v1"
    / "linguistic_judged.jsonl"
)


DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v1"
    / "language_specific_v1"
    / "evaluation.json"
)


LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]


# ============================================================
# IO
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def get_record_id(
    row: dict,
) -> str:

    return str(
        row.get("calibration_id")
        or row.get("semantic_id")
        or row.get("id")
        or "UNKNOWN"
    )


# ============================================================
# Main evaluation
# ============================================================

def evaluate(
    calibration_rows: list[dict],
    judged_rows: list[dict],
) -> dict:

    judged_map = {
        get_record_id(row):
            row

        for row in judged_rows
    }


    confusion = Counter()

    error_stats = defaultdict(
        lambda: {
            "total": 0,
            "final_rejected": 0,
            "final_accepted": 0,
            "target_judge_rejected": 0,
            "target_judge_accepted": 0,
            "target_parse_error": 0,
            "correction_provided": 0,
        }
    )


    language_stats = defaultdict(
        lambda: {
            "corrupted_total": 0,
            "target_rejected": 0,
            "target_accepted": 0,
        }
    )


    clean_language_rejections = Counter()

    false_accepts = []

    false_rejects = []

    missing_results = []

    parse_error_calls = 0

    total_language_calls = 0


    for original in calibration_rows:

        record_id = get_record_id(
            original
        )


        judged = judged_map.get(
            record_id
        )


        if judged is None:

            missing_results.append(
                record_id
            )

            continue


        expected = original.get(
            "calibration_expected",
            "UNKNOWN",
        )


        error_type = original.get(
            "calibration_error_type",
            "unknown",
        )


        corruption_language = (
            original.get(
                "calibration_language",
                "none",
            )
        )


        final_accept = bool(
            judged.get(
                "final_accept",
                False,
            )
        )


        predicted = (
            "ACCEPT"
            if final_accept
            else
            "REJECT"
        )


        confusion[
            (
                expected,
                predicted,
            )
        ] += 1


        language_judges = (
            judged.get(
                "language_judges",
                {}
            )
        )


        # ----------------------------------------------------
        # Count language calls and parse errors
        # ----------------------------------------------------

        for lang in LANGUAGES:

            result = (
                language_judges
                .get(
                    lang,
                    {},
                )
            )


            total_language_calls += 1


            error_types = result.get(
                "error_types",
                [],
            )


            if (
                "PARSE_ERROR"
                in error_types
            ):

                parse_error_calls += 1


        # ----------------------------------------------------
        # Clean sample
        # ----------------------------------------------------

        if expected == "ACCEPT":

            if final_accept:

                pass

            else:

                false_rejects.append(
                    record_id
                )


            for lang in LANGUAGES:

                lang_result = (
                    language_judges
                    .get(
                        lang,
                        {},
                    )
                )


                if not lang_result.get(
                    "accept",
                    False,
                ):

                    clean_language_rejections[
                        lang
                    ] += 1


            error_stats[
                "none"
            ][
                "total"
            ] += 1


            if final_accept:

                error_stats[
                    "none"
                ][
                    "final_accepted"
                ] += 1

            else:

                error_stats[
                    "none"
                ][
                    "final_rejected"
                ] += 1


            continue


        # ----------------------------------------------------
        # Corrupted sample
        # ----------------------------------------------------

        stat = error_stats[
            error_type
        ]

        stat[
            "total"
        ] += 1


        if final_accept:

            stat[
                "final_accepted"
            ] += 1

            false_accepts.append({
                "id":
                    record_id,

                "error_type":
                    error_type,

                "language":
                    corruption_language,
            })

        else:

            stat[
                "final_rejected"
            ] += 1


        # ----------------------------------------------------
        # Target-language recall
        #
        # 不是只看“整个 sample 被拒绝”，
        # 而是检查真正被污染的那种语言 Judge
        # 有没有识别出来。
        # ----------------------------------------------------

        if (
            corruption_language
            in LANGUAGES
        ):

            target_result = (
                language_judges
                .get(
                    corruption_language,
                    {},
                )
            )


            target_accept = bool(
                target_result.get(
                    "accept",
                    False,
                )
            )


            language_stats[
                corruption_language
            ][
                "corrupted_total"
            ] += 1


            if target_accept:

                stat[
                    "target_judge_accepted"
                ] += 1

                language_stats[
                    corruption_language
                ][
                    "target_accepted"
                ] += 1

            else:

                stat[
                    "target_judge_rejected"
                ] += 1

                language_stats[
                    corruption_language
                ][
                    "target_rejected"
                ] += 1


            target_errors = (
                target_result.get(
                    "error_types",
                    [],
                )
            )


            if (
                "PARSE_ERROR"
                in target_errors
            ):

                stat[
                    "target_parse_error"
                ] += 1


            corrected = str(
                target_result.get(
                    "corrected_sentence",
                    "",
                )
            ).strip()


            original_bad_sentence = str(
                original
                .get(
                    "texts",
                    {},
                )
                .get(
                    corruption_language,
                    "",
                )
            ).strip()


            if (
                not target_accept
                and
                corrected
                and
                corrected
                != original_bad_sentence
            ):

                stat[
                    "correction_provided"
                ] += 1


    # ========================================================
    # Confusion Matrix
    # ========================================================

    clean_accept = confusion[
        (
            "ACCEPT",
            "ACCEPT",
        )
    ]

    clean_reject = confusion[
        (
            "ACCEPT",
            "REJECT",
        )
    ]

    bad_accept = confusion[
        (
            "REJECT",
            "ACCEPT",
        )
    ]

    bad_reject = confusion[
        (
            "REJECT",
            "REJECT",
        )
    ]


    clean_total = (
        clean_accept
        +
        clean_reject
    )


    bad_total = (
        bad_accept
        +
        bad_reject
    )


    evaluated_total = (
        clean_total
        +
        bad_total
    )


    clean_accept_rate = (
        clean_accept
        / clean_total
        if clean_total
        else 0
    )


    corrupted_reject_rate = (
        bad_reject
        / bad_total
        if bad_total
        else 0
    )


    overall_accuracy = (
        (
            clean_accept
            +
            bad_reject
        )
        /
        evaluated_total
        if evaluated_total
        else 0
    )


    return {
        "evaluated_samples":
            evaluated_total,

        "missing_results":
            missing_results,

        "confusion_matrix": {
            "clean_accept":
                clean_accept,

            "clean_reject":
                clean_reject,

            "bad_accept":
                bad_accept,

            "bad_reject":
                bad_reject,
        },

        "core_metrics": {
            "clean_acceptance_rate":
                clean_accept_rate,

            "corrupted_rejection_rate":
                corrupted_reject_rate,

            "overall_accuracy":
                overall_accuracy,
        },

        "error_stats":
            dict(
                error_stats
            ),

        "language_stats":
            dict(
                language_stats
            ),

        "clean_language_rejections":
            dict(
                clean_language_rejections
            ),

        "operational": {
            "total_language_calls":
                total_language_calls,

            "parse_error_calls":
                parse_error_calls,

            "parse_error_rate":
                (
                    parse_error_calls
                    / total_language_calls
                    if total_language_calls
                    else 0
                ),

            "false_accepts":
                false_accepts,

            "false_rejects":
                false_rejects,
        },
    }


# ============================================================
# Console report
# ============================================================

def print_report(
    report: dict,
) -> None:

    confusion = report[
        "confusion_matrix"
    ]

    metrics = report[
        "core_metrics"
    ]


    print("=" * 90)
    print(
        "LANGUAGE-SPECIFIC LINGUISTIC "
        "CALIBRATION EVALUATION"
    )
    print("=" * 90)


    print(
        "\nFinal Confusion Matrix"
    )

    print("-" * 90)


    print(
        "Clean -> ACCEPT :",
        confusion[
            "clean_accept"
        ]
    )

    print(
        "Clean -> REJECT :",
        confusion[
            "clean_reject"
        ]
    )

    print(
        "Bad   -> ACCEPT :",
        confusion[
            "bad_accept"
        ]
    )

    print(
        "Bad   -> REJECT :",
        confusion[
            "bad_reject"
        ]
    )


    print(
        "\nCore Metrics"
    )

    print("-" * 90)


    print(
        "Clean acceptance rate     : "
        f"{metrics['clean_acceptance_rate']:.2%}"
    )

    print(
        "Corrupted rejection rate : "
        f"{metrics['corrupted_rejection_rate']:.2%}"
    )

    print(
        "Overall accuracy          : "
        f"{metrics['overall_accuracy']:.2%}"
    )


    print(
        "\nPer-Error Targeted Recall"
    )

    print("-" * 90)


    for error_type in sorted(
        report[
            "error_stats"
        ].keys()
    ):

        if error_type == "none":
            continue


        stat = report[
            "error_stats"
        ][
            error_type
        ]


        total = stat[
            "total"
        ]


        rejected = stat[
            "target_judge_rejected"
        ]


        recall = (
            rejected / total
            if total
            else 0
        )


        correction_rate = (
            stat[
                "correction_provided"
            ]
            / rejected
            if rejected
            else 0
        )


        print(
            f"{error_type:<28}"
            f" detected="
            f"{rejected:>2}/{total:<2}"
            f" recall="
            f"{recall:>7.2%}"
            f" correction="
            f"{correction_rate:>7.2%}"
        )


    print(
        "\nPer-Language Targeted Recall"
    )

    print("-" * 90)


    for lang in LANGUAGES:

        stat = (
            report[
                "language_stats"
            ]
            .get(
                lang,
                {},
            )
        )


        total = stat.get(
            "corrupted_total",
            0,
        )

        rejected = stat.get(
            "target_rejected",
            0,
        )


        recall = (
            rejected / total
            if total
            else 0
        )


        print(
            f"{lang:<5}"
            f" detected="
            f"{rejected:>2}/{total:<2}"
            f" recall="
            f"{recall:.2%}"
        )


    print(
        "\nClean False Rejections By Language"
    )

    print("-" * 90)


    clean_rejections = (
        report[
            "clean_language_rejections"
        ]
    )


    for lang in LANGUAGES:

        print(
            f"{lang:<5}"
            f"{clean_rejections.get(lang, 0)}"
        )


    operational = report[
        "operational"
    ]


    print(
        "\nOperational Metrics"
    )

    print("-" * 90)


    print(
        "Language calls:",
        operational[
            "total_language_calls"
        ]
    )

    print(
        "Parse error calls:",
        operational[
            "parse_error_calls"
        ]
    )

    print(
        "Parse error rate:",
        f"{operational['parse_error_rate']:.2%}"
    )

    print(
        "False accepts:",
        len(
            operational[
                "false_accepts"
            ]
        )
    )

    print(
        "False rejects:",
        len(
            operational[
                "false_rejects"
            ]
        )
    )


    if operational[
        "false_accepts"
    ]:

        print(
            "\nFalse Accept Details"
        )

        print("-" * 90)


        for item in operational[
            "false_accepts"
        ]:

            print(
                f"{item['id']:<18}"
                f"{item['language']:<5}"
                f"{item['error_type']}"
            )


    if operational[
        "false_rejects"
    ]:

        print(
            "\nFalse Reject Details"
        )

        print("-" * 90)


        for record_id in operational[
            "false_rejects"
        ]:

            print(
                record_id
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--calibration",
        type=str,
        default=str(
            DEFAULT_CALIBRATION_FILE
        ),
    )


    parser.add_argument(
        "--judged",
        type=str,
        default=str(
            DEFAULT_JUDGED_FILE
        ),
    )


    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT_FILE
        ),
    )


    args = parser.parse_args()


    calibration_file = Path(
        args.calibration
    )

    judged_file = Path(
        args.judged
    )

    output_file = Path(
        args.output
    )


    if not calibration_file.exists():

        raise FileNotFoundError(
            f"Calibration file not found:\n"
            f"{calibration_file}"
        )


    if not judged_file.exists():

        raise FileNotFoundError(
            f"Judged file not found:\n"
            f"{judged_file}"
        )


    calibration_rows = read_jsonl(
        calibration_file
    )

    judged_rows = read_jsonl(
        judged_file
    )


    report = evaluate(
        calibration_rows,
        judged_rows,
    )


    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with output_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print_report(
        report
    )


    print(
        "\nSaved evaluation:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()