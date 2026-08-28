from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen3-8B"

DEFAULT_INPUT = (
    "data/synthetic/audit/"
    "v051_targeted_40_qwen/"
    "linguistic_rejected.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    "data/synthetic/audit/"
    "v051_uz_future_reaudit"
)


SYSTEM_PROMPT = """
You are a linguistic auditor specializing in modern standard Uzbek.

You are re-auditing machine-generated Uzbek sentences that were previously
flagged as possible TENSE_ERROR cases.

Important Uzbek grammar rule:

Uzbek forms built with -a/-y and personal endings are commonly used as
present-future forms. Their negative forms with -ma/-may can also express
present, habitual, or future meaning depending on semantic context.

Examples of such forms include:
- ko'radi / ko'rmaydi
- oladi / olmaydi
- boradi / bormaydi
- keladi / kelmaydi

Do NOT assume Uzbek needs an English-style explicit auxiliary equivalent
to "will".

Do NOT reject a sentence merely because there is no time adverb such as
"ertaga".

A future-time adverb must not be required unless the semantic representation
itself contains that time information.

Your task is to decide whether the Uzbek sentence is grammatically capable
of expressing the supplied semantic meaning.

Also check:
1. subject-person agreement
2. negation
3. object case
4. verb morphology
5. semantic consistency
6. whether the sentence is acceptable in modern Uzbek

Return STRICT JSON only:

{
  "accept": true,
  "error_types": [],
  "corrected_sentence": "...",
  "reason": "..."
}

If the sentence is valid, accept=true and error_types=[].

If there is a real grammatical or semantic error, accept=false and use
specific error types such as:
TENSE_ERROR
PERSON_AGREEMENT_ERROR
CASE_ERROR
NEGATION_ERROR
SEMANTIC_ERROR
NATURALNESS_ERROR

Do not reject merely because another wording might also be possible.
""".strip()


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
                    f"{path}:{line_no}"
                ) from exc

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


def extract_json_object(
    text: str,
) -> dict[str, Any]:

    text = text.strip()

    # Remove markdown code fences if Qwen adds them.
    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

    # First try direct JSON.
    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

    except json.JSONDecodeError:
        pass

    # Fallback: find first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        candidate = text[
            start:end + 1
        ]

        obj = json.loads(candidate)

        if isinstance(obj, dict):
            return obj

    raise ValueError(
        f"Could not parse JSON from response: {text}"
    )


def build_user_prompt(
    row: dict,
) -> str:

    slots = row.get(
        "slots",
        {},
    )

    features = row.get(
        "features",
        {},
    )

    texts = row.get(
        "texts",
        {},
    )

    return f"""
Evaluate the following Uzbek translation.

Semantic ID:
{row.get("semantic_id")}

Frame:
{row.get("frame_id")}

Semantic slots:
{json.dumps(slots, ensure_ascii=False)}

Semantic features:
{json.dumps(features, ensure_ascii=False)}

Chinese reference:
{texts.get("zh")}

English reference:
{texts.get("en")}

Russian reference:
{texts.get("ru")}

Uzbek sentence:
{texts.get("uz")}

Important:
The semantic feature "future" represents future-event semantics.
It does NOT require an English-style explicit future auxiliary.

Determine whether the Uzbek sentence itself can grammatically express
that future meaning.

Return JSON only.
""".strip()


def load_model(
    model_path: str,
):

    print(
        f"Loading tokenizer: {model_path}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    )

    print("Tokenizer loaded.")

    print(
        f"Loading model: {model_path}"
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    )

    model.eval()

    print("Qwen loaded.")

    return (
        tokenizer,
        model,
    )


@torch.inference_mode()
def judge_one(
    *,
    tokenizer,
    model,
    row: dict,
) -> tuple[dict, str]:

    messages = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },
        {
            "role":
                "user",

            "content":
                build_user_prompt(row),
        },
    ]

    try:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # V100 / device_map=auto
    input_device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model_inputs = {
        key:
            value.to(input_device)
        for key, value in model_inputs.items()
    }

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
    )

    generated = outputs[
        0,
        model_inputs[
            "input_ids"
        ].shape[1]:
    ]

    raw_response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    parsed = extract_json_object(
        raw_response
    )

    # Normalize required fields.
    parsed[
        "accept"
    ] = bool(
        parsed.get(
            "accept",
            False,
        )
    )

    if not isinstance(
        parsed.get(
            "error_types"
        ),
        list,
    ):
        parsed[
            "error_types"
        ] = []

    parsed.setdefault(
        "corrected_sentence",
        row.get(
            "texts",
            {},
        ).get(
            "uz"
        ),
    )

    parsed.setdefault(
        "reason",
        "",
    )

    return (
        parsed,
        raw_response,
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "V0.5.1 Uzbek future targeted re-audit"
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    return parser.parse_args()


def main():

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

    rows = read_jsonl(
        input_path
    )

    # --------------------------------------------------------
    # Only re-audit Uzbek TENSE_ERROR candidates.
    # --------------------------------------------------------

    selected = []

    for row in rows:

        uz_judge = (
            row.get(
                "language_judges",
                {},
            ).get(
                "uz",
                {},
            )
        )

        error_types = uz_judge.get(
            "error_types",
            [],
        )

        if (
            uz_judge.get("accept") is False
            and "TENSE_ERROR" in error_types
        ):
            selected.append(row)

    print("=" * 100)
    print("V0.5.1 UZBEK FUTURE RE-AUDIT")
    print("=" * 100)

    print(
        "Input:",
        input_path,
    )

    print(
        "Input rejected rows:",
        len(rows),
    )

    print(
        "Selected Uzbek tense cases:",
        len(selected),
    )

    print(
        "Model:",
        args.model,
    )

    if not selected:

        print(
            "No Uzbek TENSE_ERROR cases found."
        )
        return

    # --------------------------------------------------------
    # CUDA cleanup before model load
    # --------------------------------------------------------

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer, model = load_model(
        args.model
    )

    judged = []
    accepted = []
    rejected = []

    error_counter = Counter()

    parse_errors = 0

    for index, row in enumerate(
        selected,
        start=1,
    ):

        output_row = dict(row)

        try:

            result, raw_response = (
                judge_one(
                    tokenizer=tokenizer,
                    model=model,
                    row=row,
                )
            )

            result[
                "parse_error"
            ] = False

        except Exception as exc:

            parse_errors += 1

            result = {
                "accept":
                    False,

                "error_types": [
                    "PARSE_ERROR"
                ],

                "corrected_sentence":
                    None,

                "reason":
                    str(exc),

                "parse_error":
                    True,
            }

            raw_response = None

        output_row[
            "uzbek_future_reaudit"
        ] = {
            "version":
                "0.5.1",

            **result,

            "raw_response":
                raw_response,
        }

        judged.append(
            output_row
        )

        if result[
            "accept"
        ]:

            accepted.append(
                output_row
            )

        else:

            rejected.append(
                output_row
            )

            for error_type in result.get(
                "error_types",
                [],
            ):
                error_counter[
                    error_type
                ] += 1

        print(
            f"{index}/{len(selected)}"
            f" | {row.get('semantic_id')}"
            f" | accept={result['accept']}"
            f" | errors={result.get('error_types', [])}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    judged_file = (
        output_dir
        / "uz_future_rejudged.jsonl"
    )

    accepted_file = (
        output_dir
        / "uz_future_reaccepted.jsonl"
    )

    rejected_file = (
        output_dir
        / "uz_future_rerejected.jsonl"
    )

    summary_file = (
        output_dir
        / "uz_future_resummary.json"
    )

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

    summary = {
        "version":
            "0.5.1",

        "input":
            str(input_path),

        "model":
            args.model,

        "total":
            len(selected),

        "accepted":
            len(accepted),

        "rejected":
            len(rejected),

        "accept_rate":
            (
                len(accepted)
                / len(selected)
                if selected
                else 0.0
            ),

        "parse_errors":
            parse_errors,

        "error_types":
            dict(
                error_counter
            ),
    }

    write_json(
        summary_file,
        summary,
    )

    print()
    print("=" * 100)
    print("UZBEK FUTURE RE-AUDIT COMPLETE")
    print("=" * 100)

    print(
        "Total:",
        len(selected),
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
            f"{len(accepted) / len(selected):.2%}"
            if selected
            else "0.00%"
        ),
    )

    print(
        "Parse errors:",
        parse_errors,
    )

    print()

    print("Error types:")

    if error_counter:

        for name, count in (
            error_counter.most_common()
        ):
            print(
                f"{name:<30}{count}"
            )

    else:

        print("None")

    print()

    print("Files:")
    print(judged_file)
    print(accepted_file)
    print(rejected_file)
    print(summary_file)


if __name__ == "__main__":
    main()