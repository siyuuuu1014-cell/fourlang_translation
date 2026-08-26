from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = Path(
    "/root/autodl-tmp/models/Qwen3-8B"
)

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v1"
    / "hard"
    / "hard_accepted.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v1"
    / "language_specific_v1"
)

CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "concepts.jsonl"
)


LANGUAGES = [
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
# IO
# ============================================================

def read_jsonl(path: Path) -> list[dict]:

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


def load_concepts() -> dict[str, dict]:

    if not CONCEPT_FILE.exists():

        print(
            "[WARN] Concept file not found:"
        )
        print(
            CONCEPT_FILE
        )

        return {}

    rows = read_jsonl(
        CONCEPT_FILE
    )

    return {
        row["id"]: row
        for row in rows
        if "id" in row
    }


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
# Semantic metadata
# ============================================================

def get_subject_person(
    row: dict,
    concepts: dict[str, dict],
) -> str | None:

    subject_id = (
        row
        .get("slots", {})
        .get("subject")
    )

    if not subject_id:
        return None

    concept = concepts.get(
        subject_id
    )

    if not concept:
        return None

    return (
        concept
        .get("meta", {})
        .get("person")
    )


def get_target_lexical_anchors(
    row: dict,
    language: str,
    concepts: dict[str, dict],
) -> dict[str, Any]:

    """
    给 Judge 提供 concept 层面的词义提示。

    注意：
    不直接把 trace 里的变位形式告诉 Qwen，
    因为 Linguistic Calibration 中 trace 被故意同步修改过。

    我们尽量只提供：
    - concept id
    - lemma/base/inf 等相对基础形式
    """

    anchors: dict[str, Any] = {}

    slots = row.get(
        "slots",
        {},
    )

    for slot_name, concept_id in slots.items():

        concept = concepts.get(
            concept_id
        )

        if not concept:
            anchors[slot_name] = {
                "concept_id":
                    concept_id,
            }
            continue

        forms = (
            concept
            .get("forms", {})
            .get(language, {})
        )

        anchor = None

        # 尽量只提供 lemma/base，
        # 不给具体人称变位答案。
        for key in [
            "base",
            "inf",
            "lemma",
        ]:

            if key in forms:

                anchor = forms[
                    key
                ]

                break

        anchors[slot_name] = {
            "concept_id":
                concept_id,

            "lexical_anchor":
                anchor,
        }

    return anchors


def build_semantic_context(
    row: dict,
    language: str,
    concepts: dict[str, dict],
) -> dict[str, Any]:

    return {
        "frame_id":
            row.get(
                "frame_id"
            ),

        "slots":
            row.get(
                "slots",
                {},
            ),

        "features":
            row.get(
                "features",
                {},
            ),

        "computed":
            row.get(
                "computed",
                {},
            ),

        "subject_person":
            get_subject_person(
                row,
                concepts,
            ),

        "lexical_anchors":
            get_target_lexical_anchors(
                row,
                language,
                concepts,
            ),
    }


# ============================================================
# Language-specific instructions
# ============================================================

def get_language_instructions(
    language: str,
) -> str:

    if language == "zh":

        return """
You are checking ONLY the Chinese sentence.

Check strictly:

1. Chinese word order.
2. Placement of time expressions.
3. Placement of location and destination phrases.
4. Verb-object collocation.
5. Grammar.
6. Naturalness for a native Mandarin speaker.
7. Whether the sentence sounds like normal modern spoken/written Chinese.

IMPORTANT:

A sentence can preserve the correct meaning but still be REJECTED
because its Chinese word order or expression is unnatural.

Example:

Natural:
我明天去机场。

Unnatural:
我去机场明天。

The second sentence MUST be rejected for WORD_ORDER_ERROR.

Do not accept an unnatural sentence merely because it is understandable.
""".strip()


    if language == "en":

        return """
You are checking ONLY the English sentence.

Check strictly:

1. Subject-verb agreement.
2. Person and number agreement.
3. Tense.
4. Auxiliary verbs.
5. Article usage.
6. Singular/plural forms.
7. Word order.
8. Grammar.
9. Naturalness.

IMPORTANT:

Semantic meaning being understandable is NOT enough.

Example:

Correct:
She buys a ticket.

Incorrect:
She buy a ticket.

The incorrect sentence MUST be rejected for SUBJECT_VERB_AGREEMENT.

Any real grammatical agreement error must cause rejection.
""".strip()


    if language == "ru":

        return """
You are checking ONLY the Russian sentence.

Check strictly:

1. Subject-verb person agreement.
2. Singular/plural agreement.
3. Verb conjugation.
4. Russian case.
5. Government of prepositions.
6. Gender agreement where applicable.
7. Tense.
8. Aspect where applicable.
9. Morphology.
10. Word order and naturalness.

IMPORTANT:

Do NOT accept a sentence merely because the intended meaning is understandable.

Example:

Correct:
Мы идём домой.

Incorrect:
Мы идёт домой.

The incorrect sentence MUST be rejected because the subject and verb
do not agree in person/number.

Similarly, a third-person singular subject must not use a plural verb form.

Any genuine Russian morphology, case, conjugation, or agreement error
must cause rejection.
""".strip()


    if language == "uz":

        return """
You are checking ONLY the Uzbek sentence.

The Uzbek text uses LATIN SCRIPT.

Check strictly:

1. Subject-verb person agreement.
2. Singular/plural agreement.
3. Verb person suffixes.
4. Tense suffixes.
5. Case suffixes.
6. Direction/location suffixes.
7. Morphology.
8. Word order.
9. Grammar.
10. Naturalness for standard Uzbek.

IMPORTANT:

Do NOT accept a sentence merely because its meaning is understandable.

Example:

Correct:
Biz Toshkentga boramiz.

Incorrect:
Biz Toshkentga boradi.

The incorrect sentence MUST be rejected because "Biz" requires
first-person plural verb agreement.

Any genuine Uzbek suffix, morphology, or agreement error must cause rejection.
""".strip()


    raise ValueError(
        f"Unsupported language: "
        f"{language}"
    )


# ============================================================
# Prompt
# ============================================================

def build_prompt(
    row: dict,
    language: str,
    concepts: dict[str, dict],
    retry: bool = False,
) -> str:

    sentence = (
        row
        .get("texts", {})
        .get(
            language,
            "",
        )
    )

    semantic_context = (
        build_semantic_context(
            row,
            language,
            concepts,
        )
    )

    language_instruction = (
        get_language_instructions(
            language
        )
    )


    retry_instruction = ""

    if retry:

        retry_instruction = """
RETRY REQUIREMENT:

Your previous response could not be parsed.

Return exactly ONE valid JSON object.
No Markdown.
No code fences.
No text before JSON.
No text after JSON.
""".strip()


    return f"""
You are a strict {LANGUAGE_NAMES[language]} linguistic quality auditor.

This sentence is intended for TRAINING DATA of a translation model.

You must judge LANGUAGE QUALITY, not merely whether the meaning is understandable.

============================================================
SEMANTIC CONTEXT
============================================================

{json.dumps(
    semantic_context,
    ensure_ascii=False,
    indent=2,
)}

============================================================
TARGET LANGUAGE
============================================================

Language:
{LANGUAGE_NAMES[language]} ({language})

Sentence:
{sentence}

============================================================
LANGUAGE-SPECIFIC AUDIT
============================================================

{language_instruction}

============================================================
DECISION POLICY
============================================================

ACCEPT only if the sentence is:

- grammatically correct,
- morphologically correct,
- internally consistent,
- natural enough for native-language training data.

REJECT if there is ANY genuine:

- agreement error,
- conjugation error,
- case error,
- suffix error,
- tense error,
- morphology error,
- word-order error,
- grammar error,
- clearly unnatural construction.

Do NOT excuse a grammatical error as:

- stylistic variation,
- harmless difference,
- understandable wording,
- minor semantic variation.

If you detect a real linguistic error, accept MUST be false.

If accept is false:

1. error_types MUST contain at least one error.
2. corrected_sentence MUST contain a corrected version.
3. reason MUST clearly identify the linguistic problem.

If accept is true:

1. error_types MUST be [].
2. corrected_sentence should normally equal the original sentence.
3. reason must briefly explain why it is linguistically valid.

============================================================
OUTPUT
============================================================

Return ONE compact JSON object only.

Do not use Markdown.
Do not use code fences.

Schema:

{{
  "accept": true,
  "error_types": [],
  "corrected_sentence": "{sentence}",
  "reason": "The sentence is grammatically and morphologically correct."
}}

Allowed error_types:

[
  "SUBJECT_VERB_AGREEMENT",
  "PERSON_AGREEMENT",
  "NUMBER_AGREEMENT",
  "CONJUGATION_ERROR",
  "CASE_ERROR",
  "SUFFIX_ERROR",
  "TENSE_ERROR",
  "ASPECT_ERROR",
  "MORPHOLOGY_ERROR",
  "ARTICLE_ERROR",
  "WORD_ORDER_ERROR",
  "COLLOCATION_ERROR",
  "GRAMMAR_ERROR",
  "NATURALNESS_ERROR",
  "OTHER_LINGUISTIC_ERROR"
]

Keep reason under 50 words.

{retry_instruction}
""".strip()


# ============================================================
# JSON parsing
# ============================================================

def clean_model_output(
    text: str,
) -> str:

    text = str(text)

    # Remove thinking blocks if model emits them.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
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
    text: str,
) -> dict:

    text = clean_model_output(
        text
    )

    decoder = json.JSONDecoder()

    # 尝试从每一个 { 开始解析，
    # 找到第一个完整 JSON object。
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
        "No valid JSON object found. "
        f"Raw output: {text[:500]!r}"
    )


# ============================================================
# Result normalization
# ============================================================

def normalize_result(
    result: dict,
    original_sentence: str,
) -> dict:

    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "Judge result is not a dict."
        )


    raw_accept = result.get(
        "accept"
    )


    if isinstance(
        raw_accept,
        str,
    ):

        raw_accept_lower = (
            raw_accept
            .strip()
            .lower()
        )

        if raw_accept_lower == "true":
            raw_accept = True

        elif raw_accept_lower == "false":
            raw_accept = False


    if not isinstance(
        raw_accept,
        bool,
    ):

        raise ValueError(
            "Field 'accept' is not boolean."
        )


    error_types = result.get(
        "error_types",
        [],
    )


    if isinstance(
        error_types,
        str,
    ):

        if not error_types.strip():

            error_types = []

        else:

            error_types = [
                error_types.strip()
            ]


    if not isinstance(
        error_types,
        list,
    ):

        raise ValueError(
            "Field 'error_types' is not a list."
        )


    error_types = [
        str(x).strip().upper()
        for x in error_types
        if str(x).strip()
    ]


    corrected_sentence = str(
        result.get(
            "corrected_sentence",
            "",
        )
    ).strip()


    reason = str(
        result.get(
            "reason",
            "",
        )
    ).strip()


    # --------------------------------------------------------
    # Internal consistency enforcement
    # --------------------------------------------------------

    accept = raw_accept


    # 模型说有错误，却又 accept=true：
    # 强制判 REJECT。
    if (
        accept
        and
        error_types
    ):

        accept = False


    # 模型说 reject，但是忘记 error type：
    # 补通用错误类型。
    if (
        not accept
        and
        not error_types
    ):

        error_types = [
            "OTHER_LINGUISTIC_ERROR"
        ]


    # 缺 corrected sentence。
    if not corrected_sentence:

        if accept:

            corrected_sentence = (
                original_sentence
            )

        else:

            corrected_sentence = ""


    # 缺 reason 不允许通过。
    if not reason:

        accept = False

        if (
            "OTHER_LINGUISTIC_ERROR"
            not in error_types
        ):

            error_types.append(
                "OTHER_LINGUISTIC_ERROR"
            )

        reason = (
            "Judge did not provide "
            "a valid explanation."
        )


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
# Model generation
# ============================================================

@torch.inference_mode()
def generate_once(
    row: dict,
    language: str,
    tokenizer,
    model,
    concepts: dict[str, dict],
    retry: bool = False,
) -> str:

    prompt = build_prompt(
        row=row,
        language=language,
        concepts=concepts,
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
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=
            tokenizer.eos_token_id,
    )


    generated = outputs[
        0,
        inputs[
            "input_ids"
        ].shape[1]:
    ]


    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


    return answer


# ============================================================
# Judge one language with retry
# ============================================================

def judge_language_with_retry(
    row: dict,
    language: str,
    tokenizer,
    model,
    concepts: dict[str, dict],
    max_retries: int,
) -> tuple[dict, list[str]]:

    last_error = None

    raw_outputs = []


    original_sentence = (
        row
        .get("texts", {})
        .get(
            language,
            "",
        )
    )


    for attempt in range(
        max_retries + 1
    ):

        try:

            raw = generate_once(
                row=row,
                language=language,
                tokenizer=tokenizer,
                model=model,
                concepts=concepts,
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


            result = normalize_result(
                parsed,
                original_sentence,
            )


            result[
                "parse_attempt"
            ] = (
                attempt + 1
            )


            return (
                result,
                raw_outputs,
            )


        except Exception as exc:

            last_error = exc


            print(
                "[PARSE RETRY]"
                f" id="
                f"{get_record_id(row)}"
                f" lang={language}"
                f" attempt="
                f"{attempt + 1}/"
                f"{max_retries + 1}"
                f" error="
                f"{repr(exc)}"
            )


    raise RuntimeError(
        "Language judge failed after "
        f"{max_retries + 1} attempts. "
        f"Last error: "
        f"{repr(last_error)}"
    )


# ============================================================
# Parse-error result
# ============================================================

def make_parse_error_result(
    exc: Exception,
) -> dict:

    return {
        "accept":
            False,

        "error_types": [
            "PARSE_ERROR"
        ],

        "corrected_sentence":
            "",

        "reason":
            repr(exc),

        "parse_attempt":
            None,
    }


# ============================================================
# Load Qwen
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
# Statistics
# ============================================================

def build_summary(
    rows: list[dict],
    parse_error_calls: int,
) -> dict:

    language_accept = {
        lang:
            Counter()

        for lang in LANGUAGES
    }


    final_counter = Counter()


    for row in rows:

        final_counter[
            "accepted"
            if row.get(
                "final_accept",
                False,
            )
            else
            "rejected"
        ] += 1


        language_results = (
            row.get(
                "language_judges",
                {}
            )
        )


        for lang in LANGUAGES:

            result = (
                language_results
                .get(
                    lang,
                    {},
                )
            )


            if result.get(
                "accept",
                False,
            ):

                language_accept[
                    lang
                ][
                    "accepted"
                ] += 1

            else:

                language_accept[
                    lang
                ][
                    "rejected"
                ] += 1


    total_samples = len(
        rows
    )

    total_calls = (
        total_samples
        * len(LANGUAGES)
    )


    return {
        "total_samples":
            total_samples,

        "total_language_calls":
            total_calls,

        "final_accepted":
            final_counter[
                "accepted"
            ],

        "final_rejected":
            final_counter[
                "rejected"
            ],

        "final_accept_rate":
            (
                final_counter[
                    "accepted"
                ]
                / total_samples
                if total_samples
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

        "language_results": {
            lang: {
                "accepted":
                    language_accept[
                        lang
                    ][
                        "accepted"
                    ],

                "rejected":
                    language_accept[
                        lang
                    ][
                        "rejected"
                    ],
            }

            for lang in LANGUAGES
        },
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        type=str,
        default=str(
            DEFAULT_INPUT_FILE
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


    print("=" * 80)
    print("LANGUAGE-SPECIFIC LINGUISTIC JUDGE V1")
    print("=" * 80)

    print(
        "Input:",
        input_file
    )

    print(
        "Samples:",
        len(rows)
    )

    print(
        "Language checks:",
        len(rows)
        * len(LANGUAGES)
    )

    print(
        "Output:",
        output_dir
    )


    tokenizer, model = load_qwen(
        model_path
    )


    judged_rows = []

    accepted_rows = []

    rejected_rows = []

    parse_error_calls = 0


    total = len(
        rows
    )


    for sample_index, source_row in enumerate(
        rows,
        start=1,
    ):

        row = dict(
            source_row
        )


        language_judges = {}


        for lang in LANGUAGES:

            try:

                (
                    result,
                    raw_outputs,
                ) = (
                    judge_language_with_retry(
                        row=row,
                        language=lang,
                        tokenizer=tokenizer,
                        model=model,
                        concepts=concepts,
                        max_retries=
                            args.max_retries,
                    )
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


            language_judges[
                lang
            ] = result


        # ----------------------------------------------------
        # Final decision:
        # all four languages must pass
        # ----------------------------------------------------

        final_accept = all(
            language_judges[
                lang
            ].get(
                "accept",
                False,
            )

            for lang in LANGUAGES
        )


        row[
            "language_judges"
        ] = language_judges

        row[
            "final_accept"
        ] = final_accept


        failed_languages = [
            lang

            for lang in LANGUAGES

            if not language_judges[
                lang
            ].get(
                "accept",
                False,
            )
        ]


        row[
            "failed_languages"
        ] = failed_languages


        judged_rows.append(
            row
        )


        if final_accept:

            accepted_rows.append(
                row
            )

        else:

            rejected_rows.append(
                row
            )


        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            sample_index
            % args.checkpoint_every
            == 0
            or
            sample_index == total
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
                f"{sample_index}/{total}"
                f" | accepted="
                f"{len(accepted_rows)}"
                f" | rejected="
                f"{len(rejected_rows)}"
                f" | parse_error_calls="
                f"{parse_error_calls}"
            )


    summary = build_summary(
        judged_rows,
        parse_error_calls,
    )


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
    print("LANGUAGE-SPECIFIC JUDGE V1 COMPLETE")
    print("=" * 80)

    print(
        "Samples:",
        summary[
            "total_samples"
        ]
    )

    print(
        "Language calls:",
        summary[
            "total_language_calls"
        ]
    )

    print(
        "Accepted:",
        summary[
            "final_accepted"
        ]
    )

    print(
        "Rejected:",
        summary[
            "final_rejected"
        ]
    )

    print(
        "Final accept rate:",
        f"{summary['final_accept_rate']:.2%}"
    )

    print(
        "Parse error calls:",
        summary[
            "parse_error_calls"
        ]
    )

    print(
        "Parse error rate:",
        f"{summary['parse_error_rate']:.2%}"
    )


    print(
        "\nPer-language:"
    )


    for lang in LANGUAGES:

        stats = summary[
            "language_results"
        ][lang]

        print(
            f"{lang:<5}"
            f" accepted="
            f"{stats['accepted']:<4}"
            f" rejected="
            f"{stats['rejected']}"
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