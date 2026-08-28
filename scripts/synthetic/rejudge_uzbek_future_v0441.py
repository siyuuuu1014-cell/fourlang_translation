from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v0441_targeted_80_qwen"
    / "linguistic_rejected.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v0441_uz_future_reaudit"
)

DEFAULT_MODEL = Path(
    "/root/autodl-tmp/models/Qwen3-8B"
)

LANG = "uz"


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}"
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
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL: {path}:{line_no}"
                ) from exc

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


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a strict but linguistically informed Uzbek language auditor.

Your task is to judge whether the Uzbek sentence is grammatically valid,
natural enough for modern standard Uzbek, and semantically compatible
with the supplied semantic structure.

IMPORTANT RULES FOR UZBEK TENSE:

1. Uzbek verb forms with -adi / -ydi and negative -maydi can function
   as PRESENT-FUTURE forms.

2. Forms such as:
   boradi
   keladi
   bormaydi
   kelmaydi
   yeydi
   sotib oladi
   sotib olmaydi
   may validly express a future event depending on context.

3. Do NOT reject a sentence merely because the semantic feature says
   "future" while the Uzbek sentence uses one of these present-future
   forms.

4. Future interpretation can be licensed by:
   - explicit temporal adverbs such as "ertaga";
   - clock/date expressions;
   - the intended event interpretation;
   - the Uzbek present-future paradigm itself.

5. An explicit equivalent of English "will" is NOT required.

6. A sentence such as:
      U maktabga keladi.
   may legitimately mean:
      She will come to the school.
   depending on intended context.

7. A negative form such as:
      U Pekinga bormaydi.
   can legitimately mean:
      She will not go to Beijing.

8. Do NOT change polarity.
   If the input is negative, the correction must remain negative.

9. Ambiguity between present and future is NOT itself a grammatical
   error when the Uzbek form legitimately allows the intended future
   reading.

10. If you reject a sentence, the corrected sentence MUST contain a
    concrete linguistic correction. Do not reject and then return an
    identical sentence as the correction.

Judge Uzbek according to Uzbek grammar, not by forcing a one-to-one
English tense morphology mapping.

Return ONLY one JSON object in this exact schema:

{
  "accept": true,
  "error_types": [],
  "corrected_sentence": "...",
  "reason": "..."
}

Allowed error_types:
[
  "TENSE_ERROR",
  "POLARITY_ERROR",
  "AGREEMENT_ERROR",
  "CASE_ERROR",
  "WORD_ORDER_ERROR",
  "LEXICAL_ERROR",
  "NATURALNESS_ERROR",
  "SEMANTIC_ERROR",
  "OTHER"
]
""".strip()


def build_user_prompt(row: dict) -> str:

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

    texts = row.get(
        "texts",
        {},
    )

    audit_group = (
        row.get(
            "audit_metadata",
            {},
        )
        .get(
            "audit_group"
        )
    )

    return f"""
Audit this Uzbek sentence.

semantic_id:
{row.get("semantic_id")}

audit_group:
{audit_group}

frame:
{row.get("frame_id")}

slots:
{json.dumps(slots, ensure_ascii=False)}

features:
{json.dumps(features, ensure_ascii=False)}

computed:
{json.dumps(computed, ensure_ascii=False)}

Cross-language semantic references:

ZH:
{texts.get("zh")}

EN:
{texts.get("en")}

RU:
{texts.get("ru")}

UZBEK SENTENCE TO JUDGE:
{texts.get("uz")}

Important:
The other languages are semantic references only.
Do not require Uzbek to copy their tense morphology literally.

Return JSON only.
""".strip()


# ============================================================
# JSON parsing
# ============================================================

def parse_json_object(text: str) -> dict:

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip().startswith(
            "```"
        ):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        obj = json.loads(text)

        if not isinstance(obj, dict):
            raise ValueError(
                "Result is not JSON object"
            )

        return obj

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            f"No JSON object found in response: {text!r}"
        )

    obj = json.loads(
        text[start : end + 1]
    )

    if not isinstance(
        obj,
        dict,
    ):
        raise ValueError(
            "Parsed result is not an object"
        )

    return obj


def normalize_result(
    result: dict,
    original_sentence: str,
) -> dict:

    accept = bool(
        result.get(
            "accept",
            False,
        )
    )

    error_types = result.get(
        "error_types",
        [],
    )

    if not isinstance(
        error_types,
        list,
    ):
        error_types = [
            str(error_types)
        ]

    corrected = str(
        result.get(
            "corrected_sentence",
            original_sentence,
        )
    ).strip()

    reason = str(
        result.get(
            "reason",
            "",
        )
    ).strip()

    original_norm = (
        original_sentence
        .strip()
    )

    identical_correction = (
        corrected
        == original_norm
    )

    judge_inconsistent = (
        (not accept)
        and identical_correction
    )

    return {
        "accept":
            accept,

        "error_types":
            error_types,

        "corrected_sentence":
            corrected,

        "reason":
            reason,

        "identical_correction":
            identical_correction,

        "judge_inconsistent":
            judge_inconsistent,
    }


# ============================================================
# Judge
# ============================================================

class UzbekFutureRejudge:

    def __init__(
        self,
        model_path: Path,
    ) -> None:

        print(
            f"Loading tokenizer: {model_path}"
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(model_path),
                trust_remote_code=True,
            )
        )

        print(
            f"Loading model: {model_path}"
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        )

        self.model.eval()

        # Avoid generation warnings from model defaults.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        print(
            "Qwen loaded."
        )

    @torch.inference_mode()
    def judge_once(
        self,
        row: dict,
    ) -> tuple[dict, str]:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(
                    row
                ),
            },
        ]

        rendered = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
        )

        model_device = (
            next(
                self.model.parameters()
            ).device
        )

        inputs = {
            k: v.to(
                model_device
            )
            for k, v in inputs.items()
        }

        generated = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

        input_length = (
            inputs[
                "input_ids"
            ].shape[1]
        )

        output_ids = generated[
            0,
            input_length:,
        ]

        raw = (
            self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
            .strip()
        )

        parsed = parse_json_object(
            raw
        )

        normalized = normalize_result(
            parsed,
            row.get(
                "texts",
                {},
            ).get(
                "uz",
                "",
            ),
        )

        return normalized, raw


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "V0.4.4.1 Uzbek future re-audit"
        )
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    model_path = Path(
        args.model
    )

    rows = read_jsonl(
        input_path
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    judged_file = (
        output_dir
        / "uz_future_rejudged.jsonl"
    )

    accepted_file = (
        output_dir
        / "uz_future_accepted.jsonl"
    )

    rejected_file = (
        output_dir
        / "uz_future_rejected.jsonl"
    )

    inconsistent_file = (
        output_dir
        / "uz_future_inconsistent.jsonl"
    )

    summary_file = (
        output_dir
        / "uz_future_summary.json"
    )

    print(
        "=" * 100
    )

    print(
        "V0.4.4.1 UZBEK FUTURE RE-AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "Input:",
        input_path,
    )

    print(
        "Samples:",
        len(rows),
    )

    print(
        "Model:",
        model_path,
    )

    print(
        "=" * 100
    )

    judge = UzbekFutureRejudge(
        model_path
    )

    judged_rows = []
    accepted_rows = []
    rejected_rows = []
    inconsistent_rows = []

    error_counter = Counter()

    parse_errors = 0

    for index, row in enumerate(
        rows,
        start=1,
    ):

        final_result = None
        raw_response = None
        last_error = None

        for attempt in range(
            1,
            args.max_retries + 2,
        ):

            try:
                final_result, raw_response = (
                    judge.judge_once(
                        row
                    )
                )

                final_result[
                    "parse_attempt"
                ] = attempt

                break

            except Exception as exc:

                last_error = str(
                    exc
                )

        output_row = dict(
            row
        )

        if final_result is None:

            parse_errors += 1

            final_result = {
                "accept":
                    False,

                "error_types": [
                    "PARSE_ERROR"
                ],

                "corrected_sentence":
                    row.get(
                        "texts",
                        {},
                    ).get(
                        "uz",
                        "",
                    ),

                "reason":
                    last_error
                    or "Unknown parsing error",

                "identical_correction":
                    True,

                "judge_inconsistent":
                    False,

                "parse_attempt":
                    args.max_retries + 1,
            }

        final_result[
            "raw_response"
        ] = raw_response

        output_row[
            "uz_future_rejudge"
        ] = final_result

        judged_rows.append(
            output_row
        )

        if final_result[
            "judge_inconsistent"
        ]:

            inconsistent_rows.append(
                output_row
            )

        if final_result[
            "accept"
        ]:

            accepted_rows.append(
                output_row
            )

        else:

            rejected_rows.append(
                output_row
            )

            for error_type in (
                final_result.get(
                    "error_types",
                    [],
                )
            ):

                error_counter[
                    error_type
                ] += 1

        print(
            f"{index}/{len(rows)}"
            f" | "
            f"{row.get('semantic_id')}"
            f" | "
            f"accept={final_result['accept']}"
            f" | "
            f"inconsistent="
            f"{final_result['judge_inconsistent']}"
        )

    total = len(
        rows
    )

    accepted = len(
        accepted_rows
    )

    rejected = len(
        rejected_rows
    )

    inconsistent = len(
        inconsistent_rows
    )

    summary = {
        "audit_version":
            "v0441_uz_future_v2",

        "input":
            str(
                input_path
            ),

        "total":
            total,

        "accepted":
            accepted,

        "rejected":
            rejected,

        "accept_rate":
            (
                accepted / total
                if total
                else 0.0
            ),

        "judge_inconsistent":
            inconsistent,

        "parse_errors":
            parse_errors,

        "error_types":
            dict(
                error_counter.most_common()
            ),
    }

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

    write_jsonl(
        inconsistent_file,
        inconsistent_rows,
    )

    write_json(
        summary_file,
        summary,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "UZBEK FUTURE RE-AUDIT COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "Total:",
        total,
    )

    print(
        "Accepted:",
        accepted,
    )

    print(
        "Rejected:",
        rejected,
    )

    print(
        "Accept rate:",
        (
            f"{accepted / total:.2%}"
            if total
            else "0.00%"
        ),
    )

    print(
        "Judge inconsistent:",
        inconsistent,
    )

    print(
        "Parse errors:",
        parse_errors,
    )

    print()

    print(
        "Error types:"
    )

    if error_counter:

        for key, value in (
            error_counter.most_common()
        ):
            print(
                f"{key:<30}{value}"
            )

    else:
        print(
            "None"
        )

    print()

    print(
        "Files:"
    )

    print(judged_file)
    print(accepted_file)
    print(rejected_file)
    print(inconsistent_file)
    print(summary_file)


if __name__ == "__main__":
    main()