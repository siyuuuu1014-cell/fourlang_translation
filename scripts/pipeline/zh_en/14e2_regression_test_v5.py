from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    "/root/autodl-tmp/fourlang_translation"
)

MAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "pipeline"
    / "zh_en"
    / "14e2_review_zh_en_with_qwen3.py"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "zh_en"
    / "14e_qwen_review"
    / "qwen_review_input_v1.parquet"
)


# ============================================================
# Load judge module
# ============================================================

spec = importlib.util.spec_from_file_location(
    "zh_en_judge",
    MAIN_SCRIPT,
)

judge = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    judge
)


# ============================================================
# Regression cases
# ============================================================

TARGETS = [

    "zh_en_review_00439",
    # Tom tried -> 汤姆累了
    # expected FAIL

    "zh_en_review_00401",
    # thought crossed my mind -> 应证
    # expected FAIL

    "zh_en_review_00848",
    # based out of NYC -> NYC以外
    # expected FAIL

    "zh_en_review_03951",
    # Download QQ -> 下个QQ
    # expected PASS/MINOR

    "zh_en_review_00659",
    # 160 million -> 1.6亿
    # expected PASS

    "zh_en_review_00638",
    # 9pm -> 晚上9点
    # expected PASS
]


# ============================================================
# Main
# ============================================================

def main():

    df = pd.read_parquet(
        INPUT_FILE
    )

    part = (
        df[
            df[
                "review_id"
            ]
            .isin(
                TARGETS
            )
        ]
        .copy()
    )

    # Preserve requested order
    order = {
        rid: i
        for i, rid
        in enumerate(
            TARGETS
        )
    }

    part[
        "_order"
    ] = (
        part[
            "review_id"
        ]
        .map(
            order
        )
    )

    part = (
        part
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "=" * 120
    )

    print(
        "ZH-EN JUDGE REGRESSION TEST"
    )

    print(
        "=" * 120
    )

    print(
        "Prompt version:",
        judge.PROMPT_VERSION
    )

    print(
        "Rows:",
        len(
            part
        )
    )

    tokenizer, model = (
        judge.load_judge_model(
            Path(
                "/root/autodl-tmp/models/Qwen3-8B"
            )
        )
    )

    records = judge.judge_batch(

        batch_df=
            part,

        model=
            model,

        tokenizer=
            tokenizer,

        max_input_tokens=
            1536,

        max_new_tokens=
            260,

        parse_retries=
            2,
    )

    result = (
        part[
            [
                "review_id",
                "en",
                "zh",
            ]
        ]
        .merge(
            pd.DataFrame(
                records
            ),
            on="review_id",
            how="left",
        )
    )

    for _, row in result.iterrows():

        print(
            "\n"
            +
            "=" * 120
        )

        print(
            "REVIEW:",
            row[
                "review_id"
            ]
        )

        print(
            "\nEN:"
        )

        print(
            row[
                "en"
            ]
        )

        print(
            "\nZH:"
        )

        print(
            row[
                "zh"
            ]
        )

        print(
            "\nLABEL:",
            row[
                "judge_label"
            ]
        )

        print(
            "FAILED:",
            row[
                "judge_failed_dimensions"
            ]
        )

        print(
            "REASON:"
        )

        print(
            row[
                "judge_reason"
            ]
        )

        print(
            "\ncore_meaning:",
            row[
                "judge_core_meaning_preserved"
            ]
        )

        print(
            "action_state:",
            row[
                "judge_main_action_state_match"
            ]
        )

        print(
            "location_direction:",
            row[
                "judge_location_direction_match"
            ]
        )

        print(
            "quantity:",
            row[
                "judge_quantity_match"
            ]
        )


if __name__ == "__main__":

    main()