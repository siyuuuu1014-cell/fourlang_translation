from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.synthetic.judge_linguistic_with_qwen import (
    load_qwen,
    load_concepts,
    read_jsonl,
    write_jsonl,
    judge_language_with_retry,
    make_parse_error_result,
    get_record_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_MODEL_PATH = Path(
    "/root/autodl-tmp/models/Qwen3-8B"
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v2"
    / "grammar_hard"
    / "grammar_accepted.jsonl"
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v2"
    / "residual_qwen_v1"
)


# 当前 Calibration V2 剩余任务只有中文。
DEFAULT_LANGUAGES = [
    "zh",
]


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        type=str,
        default=str(
            DEFAULT_INPUT
        ),
    )


    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )


    parser.add_argument(
        "--model",
        type=str,
        default=str(
            DEFAULT_MODEL_PATH
        ),
    )


    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        choices=[
            "zh",
            "en",
            "ru",
            "uz",
        ],
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
    )


    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
    )


    parser.add_argument(
        "--save-raw",
        action="store_true",
    )


    args = parser.parse_args()


    input_file = Path(
        args.input
    )


    output_dir = Path(
        args.output_dir
    )


    model_path = Path(
        args.model
    )


    # ========================================================
    # Validate
    # ========================================================

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{input_file}"
        )


    if not model_path.exists():

        raise FileNotFoundError(
            f"Model path not found:\n"
            f"{model_path}"
        )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    judged_file = (
        output_dir
        / "residual_judged.jsonl"
    )


    accepted_file = (
        output_dir
        / "residual_accepted.jsonl"
    )


    rejected_file = (
        output_dir
        / "residual_rejected.jsonl"
    )


    summary_file = (
        output_dir
        / "residual_summary.json"
    )


    # ========================================================
    # Load data
    # ========================================================

    rows = read_jsonl(
        input_file
    )


    if (
        args.limit is not None
        and
        args.limit > 0
    ):

        rows = rows[
            :args.limit
        ]


    concepts = load_concepts()


    print("=" * 90)
    print("RESIDUAL LINGUISTIC JUDGE V1")
    print("=" * 90)


    print(
        "Input:",
        input_file
    )


    print(
        "Samples:",
        len(rows)
    )


    print(
        "Languages:",
        ", ".join(
            args.languages
        )
    )


    print(
        "Qwen calls:",
        len(rows)
        * len(args.languages)
    )


    print(
        "Output:",
        output_dir
    )


    # ========================================================
    # Load model once
    # ========================================================

    tokenizer, model = load_qwen(
        model_path
    )


    judged_rows = []

    accepted_rows = []

    rejected_rows = []


    parse_error_calls = 0


    language_accept_counter = Counter()

    language_reject_counter = Counter()


    total = len(
        rows
    )


    # ========================================================
    # Judge
    # ========================================================

    for index, source_row in enumerate(
        rows,
        start=1,
    ):

        row = dict(
            source_row
        )


        residual_judges = {}


        for language in args.languages:

            try:

                (
                    result,
                    raw_outputs,
                ) = judge_language_with_retry(
                    row=row,
                    language=language,
                    tokenizer=tokenizer,
                    model=model,
                    concepts=concepts,
                    max_retries=
                        args.max_retries,
                )


                if args.save_raw:

                    result[
                        "raw_outputs"
                    ] = raw_outputs


            except Exception as exc:

                parse_error_calls += 1


                result = (
                    make_parse_error_result(
                        exc
                    )
                )


            residual_judges[
                language
            ] = result


            if result.get(
                "accept",
                False,
            ):

                language_accept_counter[
                    language
                ] += 1

            else:

                language_reject_counter[
                    language
                ] += 1


        # ====================================================
        # Final residual decision
        #
        # 所有 residual language checks 都必须 PASS。
        # ====================================================

        residual_accept = all(

            residual_judges[
                language
            ].get(
                "accept",
                False,
            )

            for language
            in args.languages
        )


        row[
            "residual_judges"
        ] = residual_judges


        row[
            "residual_languages"
        ] = args.languages


        row[
            "residual_accept"
        ] = residual_accept


        row[
            "residual_failed_languages"
        ] = [

            language

            for language
            in args.languages

            if not residual_judges[
                language
            ].get(
                "accept",
                False,
            )
        ]


        judged_rows.append(
            row
        )


        if residual_accept:

            accepted_rows.append(
                row
            )

        else:

            rejected_rows.append(
                row
            )


        # ====================================================
        # Checkpoint
        # ====================================================

        if (
            index
            % args.checkpoint_every
            == 0
            or
            index == total
        ):

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


            print(
                f"{index}/{total}"
                f" | accepted="
                f"{len(accepted_rows)}"
                f" | rejected="
                f"{len(rejected_rows)}"
                f" | parse_errors="
                f"{parse_error_calls}"
            )


    # ========================================================
    # Summary
    # ========================================================

    total_calls = (
        len(rows)
        *
        len(args.languages)
    )


    summary = {
        "version":
            "residual_linguistic_judge_v1",

        "input_file":
            str(input_file),

        "total_samples":
            len(rows),

        "languages":
            list(
                args.languages
            ),

        "total_qwen_calls":
            total_calls,

        "accepted":
            len(accepted_rows),

        "rejected":
            len(rejected_rows),

        "accept_rate":
            (
                len(accepted_rows)
                / len(rows)
                if rows
                else 0
            ),

        "parse_error_calls":
            parse_error_calls,

        "parse_error_rate":
            (
                parse_error_calls
                / total_calls
                if total_calls
                else 0
            ),

        "per_language": {

            language: {
                "accepted":
                    language_accept_counter[
                        language
                    ],

                "rejected":
                    language_reject_counter[
                        language
                    ],
            }

            for language
            in args.languages
        },
    }


    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 90)
    print("RESIDUAL LINGUISTIC JUDGE V1 COMPLETE")
    print("=" * 90)


    print(
        "Samples:",
        len(rows)
    )


    print(
        "Qwen calls:",
        total_calls
    )


    print(
        "Accepted:",
        len(accepted_rows)
    )


    print(
        "Rejected:",
        len(rejected_rows)
    )


    print(
        "Accept rate:",
        (
            f"{len(accepted_rows)/len(rows):.2%}"
            if rows
            else "0%"
        )
    )


    print(
        "Parse errors:",
        parse_error_calls
    )


    print(
        "Parse error rate:",
        (
            f"{parse_error_calls/total_calls:.2%}"
            if total_calls
            else "0%"
        )
    )


    print(
        "\nPer-language:"
    )


    for language in args.languages:

        print(
            f"{language:<5}"
            f" accepted="
            f"{language_accept_counter[language]:<4}"
            f" rejected="
            f"{language_reject_counter[language]}"
        )


    print(
        "\nFiles:"
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