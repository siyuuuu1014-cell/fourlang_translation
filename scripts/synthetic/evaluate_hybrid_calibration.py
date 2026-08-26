from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CALIBRATION = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "judge_calibration_v01.jsonl"
)


def read_jsonl(path):

    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(line)
                )

    return rows


def get_record_id(row):

    return (
        row.get(
            "calibration_id"
        )
        or
        row.get(
            "semantic_id"
        )
        or
        row.get(
            "id"
        )
    )


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--calibration",
        type=str,
        default=str(
            DEFAULT_CALIBRATION
        ),
    )


    parser.add_argument(
        "--hard",
        type=str,
        required=True,
    )


    parser.add_argument(
        "--qwen",
        type=str,
        required=True,
    )


    args = parser.parse_args()


    original_rows = read_jsonl(
        args.calibration
    )

    hard_rows = read_jsonl(
        args.hard
    )

    qwen_rows = read_jsonl(
        args.qwen
    )


    hard_map = {
        get_record_id(row):
            row

        for row in hard_rows
    }


    qwen_map = {
        get_record_id(row):
            row

        for row in qwen_rows
    }


    confusion = Counter()

    hard_confusion = Counter()

    error_stats = defaultdict(
        lambda: {
            "total": 0,
            "hard_reject": 0,
            "final_reject": 0,
            "final_accept": 0,
        }
    )


    parse_errors = 0

    missing_qwen = 0


    false_accept_rows = []

    false_reject_rows = []


    for original in original_rows:

        record_id = get_record_id(
            original
        )


        expected = original.get(
            "calibration_expected",
            "UNKNOWN",
        )


        error_type = original.get(
            "calibration_error_type",
            "unknown",
        )


        hard = hard_map.get(
            record_id
        )


        if hard is None:

            hard_accept = False

        else:

            hard_accept = bool(
                hard.get(
                    "hard_accept",
                    False,
                )
            )


        # ----------------------------------------------------
        # Hard-only decision
        # ----------------------------------------------------

        hard_prediction = (
            "ACCEPT"
            if hard_accept
            else "REJECT"
        )


        hard_confusion[
            (
                expected,
                hard_prediction,
            )
        ] += 1


        # ----------------------------------------------------
        # Hybrid:
        #
        # HARD reject => final reject
        #
        # HARD pass => must also pass Qwen
        # ----------------------------------------------------

        if not hard_accept:

            final_accept = False

        else:

            qwen = qwen_map.get(
                record_id
            )


            if qwen is None:

                missing_qwen += 1

                final_accept = False

            else:

                grade = (
                    qwen
                    .get(
                        "qwen_judge",
                        {},
                    )
                    .get(
                        "grade"
                    )
                )


                if (
                    grade
                    ==
                    "PARSE_ERROR"
                ):

                    parse_errors += 1


                final_accept = bool(
                    qwen.get(
                        "qwen_accept",
                        False,
                    )
                )


        prediction = (
            "ACCEPT"
            if final_accept
            else "REJECT"
        )


        confusion[
            (
                expected,
                prediction,
            )
        ] += 1


        stat = error_stats[
            error_type
        ]

        stat[
            "total"
        ] += 1


        if not hard_accept:

            stat[
                "hard_reject"
            ] += 1


        if final_accept:

            stat[
                "final_accept"
            ] += 1

        else:

            stat[
                "final_reject"
            ] += 1


        if (
            expected == "REJECT"
            and
            final_accept
        ):

            false_accept_rows.append(
                original
            )


        if (
            expected == "ACCEPT"
            and
            not final_accept
        ):

            false_reject_rows.append(
                original
            )


    # ========================================================
    # Metrics
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


    print("=" * 90)
    print("HYBRID QUALITY VALIDATOR V2")
    print("=" * 90)


    print(
        "\nFinal Confusion Matrix"
    )

    print("-" * 90)

    print(
        f"Clean -> ACCEPT : "
        f"{clean_accept}"
    )

    print(
        f"Clean -> REJECT : "
        f"{clean_reject}"
    )

    print(
        f"Bad   -> ACCEPT : "
        f"{bad_accept}"
    )

    print(
        f"Bad   -> REJECT : "
        f"{bad_reject}"
    )


    print(
        "\nCore Metrics"
    )

    print("-" * 90)


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


    total = len(
        original_rows
    )


    accuracy = (
        (
            clean_accept
            +
            bad_reject
        )
        /
        total
        if total
        else 0
    )


    print(
        "Clean acceptance rate     : "
        f"{clean_accept_rate:.2%}"
    )

    print(
        "Corrupted rejection rate : "
        f"{corrupted_reject_rate:.2%}"
    )

    print(
        "Overall accuracy          : "
        f"{accuracy:.2%}"
    )


    print(
        "\nError Detection"
    )

    print("-" * 90)


    for error_type in sorted(
        error_stats
    ):

        stat = error_stats[
            error_type
        ]

        total_type = stat[
            "total"
        ]


        if error_type == "none":

            rate = (
                stat[
                    "final_accept"
                ]
                /
                total_type
                if total_type
                else 0
            )


            print(
                f"{error_type:<20}"
                f"accepted="
                f"{stat['final_accept']:>3}"
                f"/{total_type:<3}"
                f" rate="
                f"{rate:.2%}"
            )

        else:

            recall = (
                stat[
                    "final_reject"
                ]
                /
                total_type
                if total_type
                else 0
            )


            print(
                f"{error_type:<20}"
                f"hard="
                f"{stat['hard_reject']:>3}"
                f"/{total_type:<3}"
                f" final="
                f"{stat['final_reject']:>3}"
                f"/{total_type:<3}"
                f" recall="
                f"{recall:.2%}"
            )


    print(
        "\nOperational Metrics"
    )

    print("-" * 90)

    print(
        "Qwen parse errors:",
        parse_errors,
    )

    print(
        "Missing Qwen results:",
        missing_qwen,
    )

    print(
        "False accepts:",
        len(
            false_accept_rows
        ),
    )

    print(
        "False rejects:",
        len(
            false_reject_rows
        ),
    )


    if false_accept_rows:

        print(
            "\nFalse Accept Details"
        )

        print("-" * 90)


        for row in false_accept_rows:

            print(
                get_record_id(
                    row
                ),
                "|",
                row.get(
                    "calibration_error_type"
                ),
            )


    if false_reject_rows:

        print(
            "\nFalse Reject Details"
        )

        print("-" * 90)


        for row in false_reject_rows:

            print(
                get_record_id(
                    row
                ),
            )


if __name__ == "__main__":
    main()