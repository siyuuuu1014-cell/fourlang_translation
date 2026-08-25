from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


PROJECT_ROOT = Path(
    "/root/autodl-tmp/fourlang_translation"
)

MODEL_PATH = Path(
    "/root/autodl-tmp/models/Qwen3-8B"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "semantic_v01_valid.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
)

JUDGED_FILE = (
    OUTPUT_DIR
    / "semantic_v01_qwen_judged.jsonl"
)

ACCEPT_FILE = (
    OUTPUT_DIR
    / "semantic_v01_qwen_accepted.jsonl"
)

REJECT_FILE = (
    OUTPUT_DIR
    / "semantic_v01_qwen_rejected.jsonl"
)


def read_jsonl(
    path,
):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def save_jsonl(
    path,
    rows,
):

    with open(
        path,
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


def extract_json(
    text,
):

    text = (
        text
        .replace(
            "```json",
            ""
        )
        .replace(
            "```",
            ""
        )
        .strip()
    )


    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )


    if not match:

        raise ValueError(
            "No JSON found"
        )


    return json.loads(
        match.group(0)
    )


def build_prompt(
    row,
):

    texts = row[
        "texts"
    ]

    return f"""
You are a multilingual translation quality auditor.

The following four sentences were generated from ONE semantic representation.
They MUST express the same meaning.

Chinese:
{texts["zh"]}

English:
{texts["en"]}

Russian:
{texts["ru"]}

Uzbek:
{texts["uz"]}

Semantic frame:
{row["frame_id"]}

Features:
{json.dumps(row["features"], ensure_ascii=False)}

Check semantic equivalence and grammar.

Return JSON ONLY:

{{
  "grade": "A",
  "semantic_consistent": true,
  "grammar_ok": {{
    "zh": true,
    "en": true,
    "ru": true,
    "uz": true
  }},
  "errors": {{
    "number_error": false,
    "time_error": false,
    "negation_error": false,
    "entity_error": false,
    "meaning_error": false
  }},
  "reason": ""
}}

Grade rules:
A = fully correct
B = correct meaning with harmless stylistic variation
C = minor language or grammar issue
D = important translation error
F = unusable

Do not explain outside JSON.
""".strip()


@torch.inference_mode()
def judge_one(
    row,
    tokenizer,
    model,
):

    prompt = build_prompt(
        row
    )


    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]


    text = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


    inputs = tokenizer(
        text,
        return_tensors="pt",
    )


    device = (
        model
        .get_input_embeddings()
        .weight
        .device
    )


    inputs = {
        key: value.to(
            device
        )
        for key, value
        in inputs.items()
    }


    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )


    generated = output[
        0,
        inputs["input_ids"].shape[1]:
    ]


    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


    return extract_json(
        answer
    )


def judge_accept(
    result,
):

    if result.get(
        "grade"
    ) not in {
        "A",
        "B",
    }:
        return False


    if not result.get(
        "semantic_consistent",
        False,
    ):
        return False


    grammar = result.get(
        "grammar_ok",
        {}
    )


    if not all(
        grammar.get(
            lang,
            False,
        )
        for lang in [
            "zh",
            "en",
            "ru",
            "uz",
        ]
    ):
        return False


    errors = result.get(
        "errors",
        {}
    )


    if any(
        errors.get(
            key,
            False,
        )
        for key in [
            "number_error",
            "time_error",
            "negation_error",
            "entity_error",
            "meaning_error",
        ]
    ):
        return False


    return True


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_FILE),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of samples to judge",
    )
    args = parser.parse_args()

    input_file = Path(
        args.input
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{input_file}"
        )

    rows = read_jsonl(
        input_file
    )


    if args.limit:

        rows = rows[
            :args.limit
        ]


    print(
        "Loading Qwen3-8B..."
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )


    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    )


    model.eval()


    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


    judged = []

    accepted = []

    rejected = []


    total = len(
        rows
    )


    for i, row in enumerate(
        rows,
        start=1,
    ):

        try:

            result = judge_one(
                row,
                tokenizer,
                model,
            )

            row[
                "qwen_judge"
            ] = result


            accepted_flag = (
                judge_accept(
                    result
                )
            )


            row[
                "qwen_accept"
            ] = accepted_flag


        except Exception as exc:

            row[
                "qwen_accept"
            ] = False

            row[
                "qwen_judge"
            ] = {
                "grade":
                    "PARSE_ERROR",

                "reason":
                    repr(exc),
            }

            accepted_flag = False


        judged.append(
            row
        )


        if accepted_flag:

            accepted.append(
                row
            )

        else:

            rejected.append(
                row
            )


        if (
            i % 10 == 0
            or
            i == total
        ):

            save_jsonl(
                JUDGED_FILE,
                judged,
            )

            save_jsonl(
                ACCEPT_FILE,
                accepted,
            )

            save_jsonl(
                REJECT_FILE,
                rejected,
            )

            print(
                f"{i}/{total} | "
                f"accepted="
                f"{len(accepted)} | "
                f"rejected="
                f"{len(rejected)}"
            )


    print("=" * 80)
    print("Qwen validation complete")
    print("=" * 80)

    print(
        "Total:",
        total,
    )

    print(
        "Accepted:",
        len(accepted),
    )

    print(
        "Rejected:",
        len(rejected),
    )

    print(
        "Accept rate:",
        f"{len(accepted)/total:.2%}"
        if total
        else "0%",
    )


if __name__ == "__main__":
    main()