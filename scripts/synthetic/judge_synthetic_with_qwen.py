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


def build_prompt(row):

    texts = row["texts"]

    slots = row.get(
        "slots",
        {}
    )

    features = row.get(
        "features",
        {}
    )

    computed = row.get(
        "computed",
        {}
    )

    return f"""
You are a strict multilingual translation quality auditor for a training-data pipeline.

Your job is NOT to approve the sample.
Your job is to actively search for semantic, grammatical, morphological,
temporal, polarity, entity, number, and naturalness errors.

The four sentences below are intended to express exactly ONE semantic structure.

============================================================
CANONICAL SEMANTIC SPECIFICATION
============================================================

Frame:
{row["frame_id"]}

Slots:
{json.dumps(slots, ensure_ascii=False)}

Features:
{json.dumps(features, ensure_ascii=False)}

Computed values:
{json.dumps(computed, ensure_ascii=False)}

============================================================
GENERATED SENTENCES
============================================================

Chinese (zh):
{texts["zh"]}

English (en):
{texts["en"]}

Russian (ru):
{texts["ru"]}

Uzbek (uz):
{texts["uz"]}

============================================================
AUDIT REQUIREMENTS
============================================================

Evaluate EACH language independently against the canonical semantic
specification.

Then compare the four languages with each other.

Check especially:

1. Meaning equivalence
2. Subject/person
3. Object
4. Destination/location
5. Tense/aspect
6. Positive vs negative polarity
7. Time/date/clock
8. Numbers
9. Named entities
10. Missing information
11. Added information
12. Russian case/conjugation/aspect
13. Uzbek case suffixes/person/tense/morphology
14. English tense/agreement/articles
15. Chinese grammatical and semantic naturalness
16. Whether the sentence sounds natural for a native speaker

IMPORTANT:

Do NOT give grade A merely because the four sentences look similar.

A sentence that is understandable but unnatural or grammatically questionable
must NOT automatically receive grade A.

Grade definitions:

A:
Fully correct, semantically precise, grammatically correct, and natural
in all four languages.

B:
Semantically correct in all four languages, with only harmless stylistic
or very minor naturalness differences.

C:
Meaning is mostly preserved, but at least one language contains a real
minor grammar, morphology, tense, aspect, or naturalness problem.

D:
Important semantic error, omission, addition, incorrect polarity,
incorrect time/number/entity, or significant grammar problem.

F:
Unusable or seriously incorrect.

Return JSON only.

Use exactly this schema:

{{
  "grade": "",
  "semantic_consistent": false,

  "grammar_ok": {{
    "zh": false,
    "en": false,
    "ru": false,
    "uz": false
  }},

  "natural_ok": {{
    "zh": false,
    "en": false,
    "ru": false,
    "uz": false
  }},

  "errors": {{
    "number_error": false,
    "time_error": false,
    "negation_error": false,
    "entity_error": false,
    "meaning_error": false,
    "tense_aspect_error": false,
    "omission_error": false,
    "addition_error": false,
    "morphology_error": false
  }},

  "problem_languages": [],

  "reason": ""
}}

The reason MUST NOT be empty.

If there is no error, briefly state why the meanings and grammatical forms
are consistent.

Do not output anything outside JSON.
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


def judge_accept(result):

    # --------------------------------------------------------
    # Grade
    # --------------------------------------------------------

    if result.get("grade") not in {
        "A",
        "B",
    }:
        return False


    # --------------------------------------------------------
    # Semantic consistency
    # --------------------------------------------------------

    if not result.get(
        "semantic_consistent",
        False,
    ):
        return False


    # --------------------------------------------------------
    # Grammar
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Naturalness
    # --------------------------------------------------------

    natural = result.get(
        "natural_ok",
        {}
    )

    if not all(
        natural.get(
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


    # --------------------------------------------------------
    # Error flags
    # --------------------------------------------------------

    errors = result.get(
        "errors",
        {}
    )

    high_risk_errors = [
        "number_error",
        "time_error",
        "negation_error",
        "entity_error",
        "meaning_error",
        "tense_aspect_error",
        "omission_error",
        "addition_error",
        "morphology_error",
    ]

    if any(
        errors.get(
            key,
            False,
        )
        for key in high_risk_errors
    ):
        return False


    # --------------------------------------------------------
    # Reason必须存在
    # --------------------------------------------------------

    reason = str(
        result.get(
            "reason",
            ""
        )
    ).strip()

    if not reason:
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