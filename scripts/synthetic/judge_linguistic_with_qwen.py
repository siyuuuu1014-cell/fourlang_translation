from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Default model
# ============================================================

DEFAULT_MODEL = (
    "/root/autodl-tmp/models/Qwen3-8B"
)


# ============================================================
# Supported languages
# ============================================================

ALL_LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]


LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ru": "Russian",
    "uz": "Uzbek",
}


# ============================================================
# Language-specific linguistic responsibilities
# ============================================================

LANGUAGE_FOCUS = {

    "zh": """
Focus ONLY on linguistic quality of the Chinese sentence.

Check carefully:
- Chinese word order
- grammar
- temporal expression placement
- aspect and resultative expression
- verb complements
- collocation
- particles such as 了 / 过 / 着 when relevant
- modal words such as 会 / 能 / 要 when required by the intended tense
- naturalness in modern Mandarin Chinese

Do NOT reject a sentence merely because another valid wording
would sound slightly more elegant.

Do NOT invent mandatory grammar rules that Chinese does not have.

Important examples:
- "她现在找到了护照。" is natural.
- "她明天会找到护照。" is natural.
- "她现在没找到护照。" is natural.
- "她明天不会找到护照。" is natural.

Reject only when there is a real grammatical, aspectual,
word-order, collocation, or clear naturalness problem.
""",

    "en": """
Focus ONLY on linguistic quality of the English sentence.

Check carefully:
- subject-verb agreement
- tense
- auxiliary verbs
- articles
- singular/plural number
- prepositions
- word order
- grammar
- collocation
- naturalness

Do NOT reject a sentence just because another stylistic
alternative is also possible.
""",

    "ru": """
Focus ONLY on linguistic quality of the Russian sentence.

Check carefully:
- subject-verb agreement
- person and number
- conjugation
- grammatical case
- gender
- tense
- aspect
- morphology
- prepositions
- word order
- naturalness

Be conservative when rejecting.

Important:
- "Мы пьём воду." is grammatically correct.
- "Мы сегодня пьём воду." is grammatically correct.
- "Мы едим еду." is grammatically valid even if somewhat generic.
- imperfective analytic future such as
  "Я буду есть" / "Я не буду есть"
  is grammatically valid.

Do NOT claim a correct first-person plural form is an
agreement error.
""",

    "uz": """
Focus ONLY on linguistic quality of the Uzbek sentence.

The text uses Uzbek Latin script.

Check carefully:
- subject-verb person agreement
- person suffixes
- tense
- case suffixes
- possessive suffixes
- morphology
- word order
- collocation
- grammar
- naturalness

Be conservative when rejecting.

Important:
- "Sen ... olasan" is valid 2nd person singular agreement.
- an explicit future-time adverb such as "ertaga"
  can make a present-form verb refer naturally to the future
  in appropriate Uzbek contexts.
- plural subject "ular" does not always require an overt
  plural verbal ending in every context.

Do NOT reject merely because another synonymous form exists.
""",
}


# ============================================================
# Allowed error types
# ============================================================

ALLOWED_ERROR_TYPES = {
    "WORD_ORDER_ERROR",
    "GRAMMAR_ERROR",
    "SUBJECT_VERB_AGREEMENT",
    "CONJUGATION_ERROR",
    "TENSE_ERROR",
    "ASPECT_ERROR",
    "CASE_ERROR",
    "GENDER_ERROR",
    "NUMBER_ERROR",
    "ARTICLE_ERROR",
    "PREPOSITION_ERROR",
    "MORPHOLOGY_ERROR",
    "COLLOCATION_ERROR",
    "NATURALNESS_ERROR",
    "PARTICLE_ERROR",
    "AUXILIARY_ERROR",
    "OTHER_LINGUISTIC_ERROR",
}


# ============================================================
# IO
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{path}"
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

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSONL at line "
                    f"{line_no}:\n{exc}"
                ) from exc

            rows.append(
                row
            )

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


# ============================================================
# JSON parser
# ============================================================

def strip_markdown_fence(
    text: str,
) -> str:

    text = text.strip()

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

    return text.strip()


def extract_json_object(
    text: str,
) -> dict | None:

    """
    Robust JSON extraction.

    Supports:
    - plain JSON
    - ```json ... ```
    - extra text before / after JSON
    """

    if not text:
        return None

    cleaned = strip_markdown_fence(
        text
    )

    # --------------------------------------------------------
    # First try normal json.loads
    # --------------------------------------------------------

    try:

        obj = json.loads(
            cleaned
        )

        if isinstance(
            obj,
            dict,
        ):

            return obj

    except Exception:
        pass

    # --------------------------------------------------------
    # Then use JSONDecoder.raw_decode from each "{"
    # --------------------------------------------------------

    decoder = json.JSONDecoder()

    for match in re.finditer(
        r"\{",
        cleaned,
    ):

        start = match.start()

        try:

            obj, _ = decoder.raw_decode(
                cleaned[start:]
            )

            if isinstance(
                obj,
                dict,
            ):

                return obj

        except Exception:
            continue

    return None


# ============================================================
# Normalization of Judge result
# ============================================================

def normalize_error_types(
    value: Any,
) -> list[str]:

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):

        value = [
            value
        ]

    if not isinstance(
        value,
        list,
    ):

        return [
            "OTHER_LINGUISTIC_ERROR"
        ]

    output = []

    for item in value:

        item = (
            str(item)
            .strip()
            .upper()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if not item:
            continue

        if item not in ALLOWED_ERROR_TYPES:

            item = (
                "OTHER_LINGUISTIC_ERROR"
            )

        if item not in output:

            output.append(
                item
            )

    return output


def normalize_accept(
    value: Any,
) -> bool | None:

    if isinstance(
        value,
        bool,
    ):

        return value

    if isinstance(
        value,
        str,
    ):

        v = (
            value
            .strip()
            .lower()
        )

        if v in {
            "true",
            "yes",
            "accept",
            "accepted",
            "pass",
        }:

            return True

        if v in {
            "false",
            "no",
            "reject",
            "rejected",
            "fail",
        }:

            return False

    return None


def normalize_judge_result(
    parsed: dict,
) -> dict | None:

    accept = normalize_accept(
        parsed.get(
            "accept"
        )
    )

    if accept is None:
        return None

    error_types = (
        normalize_error_types(
            parsed.get(
                "error_types",
                parsed.get(
                    "error_type"
                ),
            )
        )
    )

    corrected_sentence = (
        parsed.get(
            "corrected_sentence"
        )
    )

    reason = (
        parsed.get(
            "reason"
        )
    )

    if corrected_sentence is None:
        corrected_sentence = ""

    if reason is None:
        reason = ""

    corrected_sentence = str(
        corrected_sentence
    ).strip()

    reason = str(
        reason
    ).strip()

    # --------------------------------------------------------
    # ACCEPT must not carry errors
    # --------------------------------------------------------

    if accept:

        error_types = []

    # --------------------------------------------------------
    # REJECT must contain at least one error type
    # --------------------------------------------------------

    if (
        not accept
        and
        not error_types
    ):

        error_types = [
            "OTHER_LINGUISTIC_ERROR"
        ]

    return {
        "accept":
            accept,

        "error_types":
            error_types,

        "corrected_sentence":
            corrected_sentence,

        "reason":
            reason,
    }


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    *,
    row: dict,
    lang: str,
) -> str:

    texts = row.get(
        "texts",
        {},
    )

    sentence = texts.get(
        lang,
        "",
    )

    frame_id = row.get(
        "frame_id",
        "",
    )

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

    language_name = (
        LANGUAGE_NAMES[
            lang
        ]
    )

    focus = (
        LANGUAGE_FOCUS[
            lang
        ]
    )

    prompt = f"""
You are a strict but conservative {language_name}
linguistic quality reviewer.

Your task is NOT translation evaluation.

Your task is ONLY to determine whether the provided
{language_name} sentence is grammatically and linguistically
acceptable for the intended semantic frame.

The semantic information is provided only as context.
Do NOT reject the sentence because another language uses
a different wording.

Do NOT check whether all four languages match each other.
Do NOT evaluate translation quality between languages.

The deterministic validation pipeline has already checked:
- semantic slots
- entity identity
- object identity
- destination identity
- clock consistency
- polarity consistency
- other explicit structural constraints

You should focus only on residual linguistic quality.

{focus}

SEMANTIC FRAME:
{frame_id}

SLOTS:
{json.dumps(slots, ensure_ascii=False)}

FEATURES:
{json.dumps(features, ensure_ascii=False)}

COMPUTED:
{json.dumps(computed, ensure_ascii=False)}

LANGUAGE:
{lang} ({language_name})

SENTENCE:
{sentence}

Decision rules:

1. ACCEPT if the sentence is grammatically valid and reasonably
   natural in normal modern usage.

2. Do NOT reject merely because:
   - you prefer a different style,
   - another synonymous wording exists,
   - the sentence is simple,
   - the sentence is slightly generic,
   - a more elegant expression is possible.

3. REJECT only for a genuine linguistic problem such as:
   grammar, morphology, agreement, tense/aspect misuse,
   word order, case, article, particle, conjugation,
   clear collocation error, or clearly unnatural construction.

4. If REJECT:
   corrected_sentence MUST contain a genuinely corrected
   version of the sentence.
   Do not output exactly the same sentence while claiming
   it is wrong unless punctuation alone is the problem.

5. If ACCEPT:
   corrected_sentence should be exactly the original sentence.

Return exactly ONE JSON object.

Required schema:

{{
  "accept": true,
  "error_types": [],
  "corrected_sentence": "...",
  "reason": "..."
}}

or

{{
  "accept": false,
  "error_types": [
    "TENSE_ERROR"
  ],
  "corrected_sentence": "...",
  "reason": "..."
}}

Allowed error_types:

{sorted(ALLOWED_ERROR_TYPES)}

Do not output Markdown.
Do not output explanation outside JSON.
""".strip()

    return prompt


# ============================================================
# Model loading
# ============================================================

def load_model(
    model_path: str,
):

    print(
        f"Loading {Path(model_path).name}..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    print(
        "Qwen loaded."
    )

    return (
        tokenizer,
        model,
    )


# ============================================================
# Generate one Judge response
# ============================================================

@torch.inference_mode()
def generate_once(
    *,
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 384,
) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are a linguistic quality "
                "evaluation system. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # --------------------------------------------------------
    # Qwen chat template
    # --------------------------------------------------------

    try:

        text = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    except TypeError:

        # Compatibility fallback
        text = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # Put inputs on model input device
    # --------------------------------------------------------

    try:

        device = model.device

    except Exception:

        device = next(
            model.parameters()
        ).device

    inputs = {
        key:
            value.to(device)

        for key, value
        in inputs.items()
    }

    input_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    output_ids = model.generate(
        **inputs,

        max_new_tokens=max_new_tokens,

        do_sample=False,

        use_cache=True,

        eos_token_id=(
            tokenizer.eos_token_id
        ),

        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )

    generated_ids = output_ids[
        0,
        input_length:,
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# Judge one language
# ============================================================

def judge_one_language(
    *,
    tokenizer,
    model,
    row: dict,
    lang: str,
    max_retries: int,
    save_raw: bool,
) -> tuple[
    dict,
    int,
]:

    prompt = build_prompt(
        row=row,
        lang=lang,
    )

    raw_attempts = []

    attempts_total = (
        max_retries
        + 1
    )

    for attempt in range(
        1,
        attempts_total + 1,
    ):

        raw = generate_once(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
        )

        if save_raw:

            raw_attempts.append(
                raw
            )

        parsed = (
            extract_json_object(
                raw
            )
        )

        if parsed is None:

            continue

        normalized = (
            normalize_judge_result(
                parsed
            )
        )

        if normalized is None:

            continue

        normalized[
            "parse_attempt"
        ] = attempt

        if save_raw:

            normalized[
                "raw_outputs"
            ] = raw_attempts

        return (
            normalized,
            0,
        )

    # ========================================================
    # Complete parse failure
    # ========================================================

    result = {
        "accept":
            False,

        "error_types": [
            "OTHER_LINGUISTIC_ERROR"
        ],

        "corrected_sentence":
            "",

        "reason":
            "PARSE_ERROR",

        "parse_attempt":
            attempts_total,
    }

    if save_raw:

        result[
            "raw_outputs"
        ] = raw_attempts

    return (
        result,
        1,
    )


# ============================================================
# Rebuild summary from rows
# ============================================================

def build_summary(
    *,
    rows: list[dict],
    languages: list[str],
    parse_error_calls: int,
    model_path: str,
) -> dict:

    per_language = {}

    error_types = Counter()

    accepted_samples = 0
    rejected_samples = 0

    for lang in languages:

        accepted = 0
        rejected = 0

        for row in rows:

            judges = row.get(
                "language_judges",
                {},
            )

            result = judges.get(
                lang
            )

            if not result:
                continue

            if result.get(
                "accept",
                False,
            ):

                accepted += 1

            else:

                rejected += 1

                for error_type in (
                    result.get(
                        "error_types",
                        [],
                    )
                ):

                    error_types[
                        error_type
                    ] += 1

        per_language[
            lang
        ] = {
            "accepted":
                accepted,

            "rejected":
                rejected,

            "accept_rate":
                (
                    accepted
                    / len(rows)
                    if rows
                    else 0.0
                ),
        }

    for row in rows:

        if row.get(
            "linguistic_accept",
            False,
        ):

            accepted_samples += 1

        else:

            rejected_samples += 1

    language_calls = (
        len(rows)
        * len(languages)
    )

    return {

        "judge_version":
            "language_specific_v1",

        "model":
            model_path,

        "languages":
            languages,

        "samples":
            len(rows),

        "language_calls":
            language_calls,

        "accepted":
            accepted_samples,

        "rejected":
            rejected_samples,

        "final_accept_rate":
            (
                accepted_samples
                / len(rows)
                if rows
                else 0.0
            ),

        "parse_error_calls":
            parse_error_calls,

        "parse_error_rate":
            (
                parse_error_calls
                / language_calls
                if language_calls
                else 0.0
            ),

        "per_language":
            per_language,

        "error_types":
            dict(
                error_types.most_common()
            ),
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    *,
    output_dir: Path,
    judged_rows: list[dict],
    languages: list[str],
    parse_error_calls: int,
    model_path: str,
) -> None:

    checkpoint_file = (
        output_dir
        / "linguistic_checkpoint.jsonl"
    )

    checkpoint_summary = (
        output_dir
        / "linguistic_checkpoint_summary.json"
    )

    write_jsonl(
        checkpoint_file,
        judged_rows,
    )

    summary = build_summary(
        rows=judged_rows,
        languages=languages,
        parse_error_calls=parse_error_calls,
        model_path=model_path,
    )

    write_json(
        checkpoint_summary,
        summary,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Language-Specific Linguistic "
            "Judge using Qwen."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Qwen model path. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number "
            "of samples."
        ),
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        choices=ALL_LANGUAGES,
        default=ALL_LANGUAGES,
        help=(
            "Languages to judge. "
            "Examples: "
            "--languages zh "
            "or --languages zh en. "
            "Default: zh en ru uz."
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help=(
            "Number of parse retries "
            "after the first attempt."
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help=(
            "Save checkpoint every N samples. "
            "Set 0 to disable."
        ),
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
        help=(
            "Save raw Qwen responses "
            "inside results."
        ),
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Remove duplicate language args while preserving order
    # --------------------------------------------------------

    languages = []

    for lang in args.languages:

        if lang not in languages:

            languages.append(
                lang
            )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    rows = read_jsonl(
        input_path
    )

    if args.limit is not None:

        if args.limit <= 0:

            raise ValueError(
                "--limit must be > 0"
            )

        rows = rows[
            :args.limit
        ]

    # --------------------------------------------------------
    # Validate selected text fields
    # --------------------------------------------------------

    for i, row in enumerate(
        rows,
        start=1,
    ):

        texts = row.get(
            "texts",
            {},
        )

        for lang in languages:

            text = texts.get(
                lang
            )

            if not text:

                raise RuntimeError(
                    f"Sample #{i} "
                    f"{row.get('semantic_id')} "
                    f"missing texts[{lang!r}]"
                )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        "=" * 90
    )

    print(
        "LANGUAGE-SPECIFIC LINGUISTIC JUDGE V1"
    )

    print(
        "=" * 90
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
        "Languages:",
        ", ".join(
            languages
        ),
    )

    print(
        "Language checks:",
        (
            len(rows)
            * len(languages)
        ),
    )

    print(
        "Output:",
        output_dir,
    )

    print(
        "Model:",
        args.model,
    )

    # --------------------------------------------------------
    # Load Qwen
    # --------------------------------------------------------

    tokenizer, model = load_model(
        args.model
    )

    judged_rows = []

    parse_error_calls = 0

    sample_accept_count = 0
    sample_reject_count = 0

    start_time = time.time()

    # ========================================================
    # Judge
    # ========================================================

    for sample_index, source_row in enumerate(
        rows,
        start=1,
    ):

        row = dict(
            source_row
        )

        language_judges = {}

        sample_accept = True

        # ----------------------------------------------------
        # Only selected languages
        # ----------------------------------------------------

        for lang in languages:

            result, parse_error = (
                judge_one_language(
                    tokenizer=tokenizer,
                    model=model,
                    row=row,
                    lang=lang,
                    max_retries=(
                        args.max_retries
                    ),
                    save_raw=(
                        args.save_raw
                    ),
                )
            )

            language_judges[
                lang
            ] = result

            parse_error_calls += (
                parse_error
            )

            if not result.get(
                "accept",
                False,
            ):

                sample_accept = False

        row[
            "language_judges"
        ] = language_judges

        row[
            "judged_languages"
        ] = languages

        row[
            "linguistic_accept"
        ] = sample_accept

        if sample_accept:

            sample_accept_count += 1

        else:

            sample_reject_count += 1

        judged_rows.append(
            row
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            sample_index % 5 == 0
            or
            sample_index == len(rows)
        ):

            print(
                f"{sample_index}/{len(rows)}"
                f" | accepted="
                f"{sample_accept_count}"
                f" | rejected="
                f"{sample_reject_count}"
                f" | parse_error_calls="
                f"{parse_error_calls}"
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            args.checkpoint_every
            and
            sample_index
            % args.checkpoint_every
            == 0
        ):

            save_checkpoint(
                output_dir=output_dir,
                judged_rows=judged_rows,
                languages=languages,
                parse_error_calls=(
                    parse_error_calls
                ),
                model_path=args.model,
            )

    # ========================================================
    # Split
    # ========================================================

    accepted_rows = [
        row
        for row in judged_rows
        if row.get(
            "linguistic_accept",
            False,
        )
    ]

    rejected_rows = [
        row
        for row in judged_rows
        if not row.get(
            "linguistic_accept",
            False,
        )
    ]

    # ========================================================
    # Output files
    # ========================================================

    judged_file = (
        output_dir
        / "linguistic_judged.jsonl"
    )

    accepted_file = (
        output_dir
        / "linguistic_accepted.jsonl"
    )

    rejected_file = (
        output_dir
        / "linguistic_rejected.jsonl"
    )

    summary_file = (
        output_dir
        / "linguistic_summary.json"
    )

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

    summary = build_summary(
        rows=judged_rows,
        languages=languages,
        parse_error_calls=(
            parse_error_calls
        ),
        model_path=args.model,
    )

    summary[
        "input"
    ] = str(
        input_path
    )

    summary[
        "output_dir"
    ] = str(
        output_dir
    )

    summary[
        "elapsed_seconds"
    ] = (
        time.time()
        - start_time
    )

    write_json(
        summary_file,
        summary,
    )

    # ========================================================
    # Console result
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "LANGUAGE-SPECIFIC JUDGE V1 COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "Samples:",
        summary[
            "samples"
        ],
    )

    print(
        "Languages:",
        ", ".join(
            languages
        ),
    )

    print(
        "Language calls:",
        summary[
            "language_calls"
        ],
    )

    print(
        "Accepted:",
        summary[
            "accepted"
        ],
    )

    print(
        "Rejected:",
        summary[
            "rejected"
        ],
    )

    print(
        "Final accept rate:",
        (
            f"{summary['final_accept_rate']:.2%}"
        ),
    )

    print(
        "Parse error calls:",
        summary[
            "parse_error_calls"
        ],
    )

    print(
        "Parse error rate:",
        (
            f"{summary['parse_error_rate']:.2%}"
        ),
    )

    print()

    print(
        "Per-language:"
    )

    for lang in languages:

        item = (
            summary[
                "per_language"
            ][lang]
        )

        print(
            f"{lang:<4}"
            f" accepted="
            f"{item['accepted']}"
            f" rejected="
            f"{item['rejected']}"
            f" accept_rate="
            f"{item['accept_rate']:.2%}"
        )

    print()

    print(
        "Error types:"
    )

    if summary[
        "error_types"
    ]:

        for (
            error_type,
            count,
        ) in summary[
            "error_types"
        ].items():

            print(
                f"{error_type:<30}"
                f"{count}"
            )

    else:

        print(
            "None"
        )

    print()

    print(
        "Files:"
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

    # ========================================================
    # Cleanup
    # ========================================================

    try:

        del model
        del tokenizer

    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


if __name__ == "__main__":

    main()