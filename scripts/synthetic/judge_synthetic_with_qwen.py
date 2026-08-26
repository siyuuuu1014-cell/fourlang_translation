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


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = Path(
    "/root/autodl-tmp/models/Qwen3-8B"
)

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "semantic_v01_valid.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "qwen_v2"
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
):

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


def write_jsonl(
    path: Path,
    rows,
):

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


# ============================================================
# Strict prompt
# ============================================================

def build_prompt(
    row,
    retry=False,
):

    texts = row[
        "texts"
    ]

    slots = row.get(
        "slots",
        {},
    )

    features = row.get(
        "features",
        {},
    )

    computed = row.get(
        "computed",
        {},
    )


    retry_instruction = ""

    if retry:

        retry_instruction = """
IMPORTANT RETRY:
Your previous response could not be parsed.

Return ONE valid compact JSON object only.
Do not use Markdown.
Do not use code fences.
Do not output text before or after JSON.
""".strip()


    return f"""
You are a STRICT multilingual translation quality auditor.

This is training data for a multilingual translation model.

Your goal is NOT to approve the sample.

Your goal is to actively search for ANY:

- semantic mismatch
- object mismatch
- destination mismatch
- entity mismatch
- time mismatch
- number mismatch
- polarity mismatch
- subject mismatch
- action mismatch
- omission
- addition
- tense/aspect error
- grammar error
- morphology error
- unnatural wording

The canonical semantic specification below is the GROUND TRUTH.

============================================================
CANONICAL SEMANTIC SPECIFICATION
============================================================

Frame:
{row.get("frame_id")}

Slots:
{json.dumps(slots, ensure_ascii=False)}

Features:
{json.dumps(features, ensure_ascii=False)}

Computed:
{json.dumps(computed, ensure_ascii=False)}

============================================================
GENERATED SENTENCES
============================================================

Chinese:
{texts["zh"]}

English:
{texts["en"]}

Russian:
{texts["ru"]}

Uzbek:
{texts["uz"]}

============================================================
MANDATORY SEMANTIC RULES
============================================================

The following are REAL semantic errors.

They are NEVER stylistic differences.

FOOD vs WATER
=> object mismatch
=> REJECT

AIRPORT vs HOTEL
=> destination mismatch
=> REJECT

TASHKENT vs MOSCOW
=> entity mismatch
=> REJECT

20:30 vs 01:30
=> time mismatch
=> REJECT

positive vs negative
=> polarity mismatch
=> REJECT

different subject
=> subject mismatch
=> REJECT

different action
=> action mismatch
=> REJECT

missing semantic information
=> omission
=> REJECT

added semantic information
=> addition
=> REJECT


If you detect ANY mismatch in:

- subject
- action
- object
- destination
- entity
- number
- time
- polarity

the grade MUST be D or F.

Never describe such a mismatch as:

- stylistic
- harmless
- minor wording variation
- acceptable variation


Grade B is allowed ONLY when:

ALL semantic information is identical

AND

the difference is purely wording/style.


============================================================
LANGUAGE-SPECIFIC CHECKS
============================================================

Chinese:
- grammar
- word order
- naturalness
- temporal expression

English:
- tense
- agreement
- article usage
- naturalness

Russian:
- case
- conjugation
- tense/aspect
- morphology
- naturalness

Uzbek:
- case suffix
- person agreement
- tense
- morphology
- naturalness


============================================================
GRADING
============================================================

A:
Fully correct, precise, grammatical and natural in all 4 languages.

B:
All semantic slots are identical.
Only harmless wording/style variation exists.

C:
Meaning is mostly correct but there is a genuine minor
grammar, morphology, tense/aspect, or naturalness issue.

D:
Important semantic mismatch, omission, addition,
wrong time, entity, object, polarity, number,
subject or action.

F:
Seriously wrong or unusable.


============================================================
OUTPUT
============================================================

Return ONE JSON object only.

Do not use Markdown.

Do not use code fences.

The reason MUST NOT be empty.

Keep reason under 40 words.

Schema:

{{
  "grade": "A",

  "semantic_consistent": true,

  "slot_checks": {{
    "subject": true,
    "action": true,
    "object": true,
    "destination": true,
    "time": true,
    "number": true,
    "polarity": true
  }},

  "grammar_ok": {{
    "zh": true,
    "en": true,
    "ru": true,
    "uz": true
  }},

  "natural_ok": {{
    "zh": true,
    "en": true,
    "ru": true,
    "uz": true
  }},

  "problem_languages": [],

  "error_types": [],

  "reason": "All four sentences preserve the canonical semantics."
}}

Allowed error_types:

[
  "SUBJECT_ERROR",
  "ACTION_ERROR",
  "OBJECT_ERROR",
  "DESTINATION_ERROR",
  "ENTITY_ERROR",
  "NUMBER_ERROR",
  "TIME_ERROR",
  "NEGATION_ERROR",
  "TENSE_ASPECT_ERROR",
  "OMISSION_ERROR",
  "ADDITION_ERROR",
  "MORPHOLOGY_ERROR",
  "GRAMMAR_ERROR",
  "NATURALNESS_ERROR",
  "MEANING_ERROR"
]

{retry_instruction}
""".strip()


# ============================================================
# Robust JSON extraction
# ============================================================

def clean_model_output(
    text,
):

    text = str(text)

    # Remove Qwen thinking tags if any.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
        | re.IGNORECASE,
    )


    text = (
        text
        .replace(
            "```json",
            ""
        )
        .replace(
            "```JSON",
            ""
        )
        .replace(
            "```",
            ""
        )
        .strip()
    )


    return text


def extract_json(
    text,
):

    text = clean_model_output(
        text
    )


    decoder = json.JSONDecoder()


    # Try every '{' until one complete
    # valid JSON object is found.

    for index, char in enumerate(
        text
    ):

        if char != "{":
            continue

        try:

            obj, _ = decoder.raw_decode(
                text[
                    index:
                ]
            )

        except json.JSONDecodeError:

            continue


        if isinstance(
            obj,
            dict,
        ):

            return obj


    raise ValueError(
        "No valid JSON object found "
        f"in model output: {text[:300]!r}"
    )


# ============================================================
# Result normalization
# ============================================================

def normalize_result(
    result,
):

    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "Judge result is not a dict"
        )


    grade = str(
        result.get(
            "grade",
            "",
        )
    ).strip().upper()


    result[
        "grade"
    ] = grade


    # --------------------------------------------------------
    # Handle older "errors" dict format if returned.
    # --------------------------------------------------------

    if (
        "error_types"
        not in result
        and
        isinstance(
            result.get("errors"),
            dict,
        )
    ):

        converted = []

        mapping = {
            "number_error":
                "NUMBER_ERROR",

            "time_error":
                "TIME_ERROR",

            "negation_error":
                "NEGATION_ERROR",

            "entity_error":
                "ENTITY_ERROR",

            "meaning_error":
                "MEANING_ERROR",

            "tense_aspect_error":
                "TENSE_ASPECT_ERROR",

            "omission_error":
                "OMISSION_ERROR",

            "addition_error":
                "ADDITION_ERROR",

            "morphology_error":
                "MORPHOLOGY_ERROR",
        }


        for (
            old_key,
            new_key,
        ) in mapping.items():

            if result[
                "errors"
            ].get(
                old_key
            ) is True:

                converted.append(
                    new_key
                )


        result[
            "error_types"
        ] = converted


    if not isinstance(
        result.get(
            "error_types"
        ),
        list,
    ):

        result[
            "error_types"
        ] = [
            "INVALID_ERROR_TYPES"
        ]


    if not isinstance(
        result.get(
            "problem_languages"
        ),
        list,
    ):

        result[
            "problem_languages"
        ] = []


    return result


# ============================================================
# Determine which semantic slots actually apply
# ============================================================

def applicable_slot_checks(
    row,
):

    slots = row.get(
        "slots",
        {},
    )

    computed = row.get(
        "computed",
        {},
    )

    features = row.get(
        "features",
        {},
    )


    checks = []


    if "subject" in slots:
        checks.append(
            "subject"
        )


    if "verb" in slots:
        checks.append(
            "action"
        )


    if "object" in slots:
        checks.append(
            "object"
        )


    if (
        "destination"
        in slots
        or
        "place"
        in slots
    ):

        checks.append(
            "destination"
        )


    if (
        "time" in slots
        or
        "day" in slots
        or
        "clock" in computed
    ):

        checks.append(
            "time"
        )


    if (
        "number" in slots
        or
        "number" in computed
    ):

        checks.append(
            "number"
        )


    if "polarity" in features:

        checks.append(
            "polarity"
        )


    return checks


# ============================================================
# Final Qwen acceptance
# ============================================================

def judge_accept(
    result,
    row,
):

    # --------------------------------------------------------
    # Grade
    # --------------------------------------------------------

    if result.get(
        "grade"
    ) not in {
        "A",
        "B",
    }:

        return False


    # --------------------------------------------------------
    # Overall semantic equivalence
    # --------------------------------------------------------

    if result.get(
        "semantic_consistent"
    ) is not True:

        return False


    # --------------------------------------------------------
    # Applicable semantic slots
    # --------------------------------------------------------

    slot_checks = result.get(
        "slot_checks",
        {},
    )


    if not isinstance(
        slot_checks,
        dict,
    ):

        return False


    for slot in (
        applicable_slot_checks(
            row
        )
    ):

        if slot_checks.get(
            slot
        ) is not True:

            return False


    # --------------------------------------------------------
    # Grammar
    # --------------------------------------------------------

    grammar = result.get(
        "grammar_ok",
        {},
    )


    if not isinstance(
        grammar,
        dict,
    ):

        return False


    for lang in LANGUAGES:

        if grammar.get(
            lang
        ) is not True:

            return False


    # --------------------------------------------------------
    # Naturalness
    # --------------------------------------------------------

    natural = result.get(
        "natural_ok",
        {},
    )


    if not isinstance(
        natural,
        dict,
    ):

        return False


    for lang in LANGUAGES:

        if natural.get(
            lang
        ) is not True:

            return False


    # --------------------------------------------------------
    # Any declared error => reject
    # --------------------------------------------------------

    error_types = result.get(
        "error_types",
        [],
    )


    if len(
        error_types
    ) > 0:

        return False


    # --------------------------------------------------------
    # Reason must exist
    # --------------------------------------------------------

    reason = str(
        result.get(
            "reason",
            "",
        )
    ).strip()


    if not reason:

        return False


    return True


# ============================================================
# Single Qwen generation
# ============================================================

@torch.inference_mode()
def generate_once(
    row,
    tokenizer,
    model,
    retry=False,
):

    prompt = build_prompt(
        row,
        retry=retry,
    )


    messages = [
        {
            "role":
                "user",

            "content":
                prompt,
        }
    ]


    chat_text = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )


    device = (
        model
        .get_input_embeddings()
        .weight
        .device
    )


    inputs = {
        key:
            value.to(device)

        for (
            key,
            value,
        ) in inputs.items()
    }


    outputs = model.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=False,
        pad_token_id=
            tokenizer.eos_token_id,
    )


    generated_tokens = outputs[
        0,
        inputs[
            "input_ids"
        ].shape[1]:
    ]


    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


    return answer


# ============================================================
# Judge with parse retry
# ============================================================

def judge_with_retry(
    row,
    tokenizer,
    model,
    max_retries=2,
):

    last_error = None

    raw_outputs = []


    for attempt in range(
        max_retries + 1
    ):

        try:

            raw = generate_once(
                row,
                tokenizer,
                model,
                retry=(
                    attempt > 0
                ),
            )


            raw_outputs.append(
                raw
            )


            parsed = extract_json(
                raw
            )


            parsed = normalize_result(
                parsed
            )


            parsed[
                "_parse_attempt"
            ] = (
                attempt + 1
            )


            return (
                parsed,
                raw_outputs,
            )


        except Exception as exc:

            last_error = exc


            print(
                "[PARSE RETRY] "
                f"record="
                f"{get_record_id(row)} "
                f"attempt="
                f"{attempt + 1}/"
                f"{max_retries + 1} "
                f"error="
                f"{repr(exc)}"
            )


    raise RuntimeError(
        "Judge parse failed "
        f"after "
        f"{max_retries + 1} "
        f"attempts: "
        f"{repr(last_error)}"
    )


# ============================================================
# Load model
# ============================================================

def load_qwen(
    model_path: Path,
):

    print(
        "Loading Qwen3-8B..."
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_path,
            local_files_only=True,
        )
    )


    if (
        tokenizer.pad_token_id
        is None
    ):

        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
        )


    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    )


    model.eval()


    model.generation_config.do_sample = (
        False
    )

    model.generation_config.temperature = (
        None
    )

    model.generation_config.top_p = (
        None
    )

    model.generation_config.top_k = (
        None
    )


    print(
        "Qwen loaded."
    )


    return (
        tokenizer,
        model,
    )


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
        "--limit",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
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


    if not input_file.exists():

        raise FileNotFoundError(
            f"Input not found:\n"
            f"{input_file}"
        )


    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{model_path}"
        )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    judged_file = (
        output_dir
        / "qwen_judged.jsonl"
    )

    accepted_file = (
        output_dir
        / "qwen_accepted.jsonl"
    )

    rejected_file = (
        output_dir
        / "qwen_rejected.jsonl"
    )

    summary_file = (
        output_dir
        / "qwen_summary.json"
    )


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


    print(
        "Input:",
        input_file
    )

    print(
        "Samples:",
        len(rows)
    )

    print(
        "Output dir:",
        output_dir
    )


    tokenizer, model = load_qwen(
        model_path
    )


    judged = []
    accepted = []
    rejected = []

    parse_errors = 0


    total = len(
        rows
    )


    for (
        index,
        row,
    ) in enumerate(
        rows,
        start=1,
    ):

        row = dict(
            row
        )


        try:

            result, raw_outputs = (
                judge_with_retry(
                    row,
                    tokenizer,
                    model,
                    max_retries=
                        args.max_retries,
                )
            )


            qwen_accept = judge_accept(
                result,
                row,
            )


            row[
                "qwen_judge"
            ] = result

            row[
                "qwen_accept"
            ] = qwen_accept

            row[
                "qwen_raw_outputs"
            ] = raw_outputs


        except Exception as exc:

            parse_errors += 1


            qwen_accept = False


            row[
                "qwen_accept"
            ] = False

            row[
                "qwen_judge"
            ] = {
                "grade":
                    "PARSE_ERROR",

                "semantic_consistent":
                    False,

                "slot_checks":
                    {},

                "grammar_ok":
                    {},

                "natural_ok":
                    {},

                "problem_languages":
                    [],

                "error_types":
                    [
                        "PARSE_ERROR"
                    ],

                "reason":
                    repr(exc),
            }


        judged.append(
            row
        )


        if qwen_accept:

            accepted.append(
                row
            )

        else:

            rejected.append(
                row
            )


        # ----------------------------------------------------
        # Save every 10 rows
        # ----------------------------------------------------

        if (
            index % 10 == 0
            or
            index == total
        ):

            write_jsonl(
                judged_file,
                judged,
            )

            write_jsonl(
                accepted_file,
                accepted,
            )

            write_jsonl(
                rejected_file,
                rejected,
            )


            print(
                f"{index}/{total}"
                f" | accepted="
                f"{len(accepted)}"
                f" | rejected="
                f"{len(rejected)}"
                f" | parse_errors="
                f"{parse_errors}"
            )


    summary = {
        "input_file":
            str(input_file),

        "total":
            total,

        "accepted":
            len(accepted),

        "rejected":
            len(rejected),

        "accept_rate":
            (
                len(accepted)
                / total
                if total
                else 0
            ),

        "parse_errors":
            parse_errors,

        "parse_error_rate":
            (
                parse_errors
                / total
                if total
                else 0
            ),
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


    print()
    print("=" * 80)
    print("QWEN STRICT JUDGE V2 COMPLETE")
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
        (
            f"{len(accepted)/total:.2%}"
            if total
            else "0%"
        ),
    )

    print(
        "Parse errors:",
        parse_errors,
    )

    print(
        "Parse error rate:",
        (
            f"{parse_errors/total:.2%}"
            if total
            else "0%"
        ),
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