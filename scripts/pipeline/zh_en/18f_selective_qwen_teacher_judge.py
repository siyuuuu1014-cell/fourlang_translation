from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_VERSION = "18F_V1"

DEFAULT_QWEN_PATH = "/root/autodl-tmp/models/Qwen3-8B"

VALID_AB_WINNERS = {
    "A",
    "B",
    "TIE",
    "BOTH_BAD",
}

VALID_MODEL_WINNERS = {
    "OPUS",
    "MADLAD",
    "TIE",
    "BOTH_BAD",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 18F - selective production Qwen Teacher judge for only "
            "the Step18E3 QWEN_REQUIRED rows. Reuses matching Step18E2 "
            "calibration decisions and judges only the remaining rows."
        )
    )

    p.add_argument("--project_root", default=None)

    p.add_argument(
        "--projection",
        default=(
            "data/distillation/zh_en/v1/"
            "18e_qwen_calibration/18e3_gate_policy/"
            "teacher_gate_projection_20k_v1.parquet"
        ),
    )

    p.add_argument(
        "--calibration_results",
        default=(
            "data/distillation/zh_en/v1/"
            "18e_qwen_calibration/18e2_pairwise/"
            "qwen_teacher_routing_calibration_results_v1.parquet"
        ),
    )

    p.add_argument(
        "--model_path",
        default=DEFAULT_QWEN_PATH,
    )

    p.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18f_selective_qwen"
        ),
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--max_input_tokens",
        type=int,
        default=1536,
    )

    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=192,
    )

    p.add_argument(
        "--max_retries",
        type=int,
        default=2,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def stable_int(text: str) -> int:
    return int(
        hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()[:16],
        16,
    )


def clean_text(value) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def save_json(obj: dict, path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def append_jsonl(rows: list[dict], path: Path):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
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


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

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
                json.loads(
                    line
                )
            )

    return pd.DataFrame(
        rows
    )


def gpu_name():
    if not torch.cuda.is_available():
        return "CPU"

    return torch.cuda.get_device_name(
        0
    )


def cleanup_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_qwen(model_path: Path):
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    if (
        tokenizer.pad_token_id
        is None
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print("Tokenizer loaded.")

    print("Loading Qwen3-8B...")

    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        kwargs[
            "torch_dtype"
        ] = torch.float16

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(
                model_path
            ),
            **kwargs,
        )
    )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    try:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    except Exception:
        pass

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Model loaded.")
    print(
        "Parameters:",
        f"{params:,}",
    )

    return (
        tokenizer,
        model,
        params,
    )


def render_chat_prompt(
    tokenizer,
    prompt: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:
        return (
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    except TypeError:
        return (
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


def build_prompt(
    *,
    row: pd.Series,
    a_model: str,
    a_text: str,
    b_model: str,
    b_text: str,
    retry_note: str = "",
) -> str:
    source = clean_text(
        row[
            "source_text"
        ]
    )

    human = clean_text(
        row[
            "human_reference"
        ]
    )

    prompt = f"""You are selecting the safer machine-translation Teacher target.

Judge semantic faithfulness to SOURCE first.
The HUMAN REFERENCE is auxiliary only. It may be noisy or only one valid wording.
Do not prefer a candidate merely because it resembles the reference.

SOURCE LANGUAGE: {row["source_lang"]}
TARGET LANGUAGE: {row["target_lang"]}

SOURCE:
{source}

HUMAN REFERENCE (auxiliary only):
{human}

CANDIDATE A ({a_model}):
{a_text}

CANDIDATE B ({b_model}):
{b_text}

A MAJOR ERROR means a material error in core meaning, action/state, participant roles,
entity identity, quantity, negation/event status, time/event order, location/direction,
important omission, or unsupported factual addition.

Choose:
A         = A is meaningfully safer/better.
B         = B is meaningfully safer/better.
TIE       = both are acceptable and neither has a meaningful semantic advantage.
BOTH_BAD  = both contain substantive semantic errors and should not be KD targets.

Consistency:
- major_error=true implies acceptable=false.
- If only A has a major error, winner cannot be A.
- If only B has a major error, winner cannot be B.
- BOTH_BAD should only be used when both are substantively unsafe.

Return exactly one JSON object, nothing else:
{{
  "winner": "A|B|TIE|BOTH_BAD",
  "confidence": "HIGH|MEDIUM|LOW",
  "A": {{"acceptable": true, "major_error": false}},
  "B": {{"acceptable": true, "major_error": false}},
  "reason": "brief evidence-based reason"
}}"""

    if retry_note:
        prompt += (
            "\n\nRETRY NOTE:\n"
            + retry_note
        )

    return prompt


def extract_json_object(
    text: str,
):
    text = str(
        text
    ).strip()

    if not text:
        return None

    try:
        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            dict,
        ):
            return obj

    except Exception:
        pass

    first = text.find(
        "{"
    )

    last = text.rfind(
        "}"
    )

    if (
        first >= 0
        and
        last > first
    ):
        candidate = text[
            first:
            last + 1
        ]

        try:
            obj = json.loads(
                candidate
            )

            if isinstance(
                obj,
                dict,
            ):
                return obj

        except Exception:
            pass

    return None


def normalize_and_validate(
    obj,
):
    if not isinstance(
        obj,
        dict,
    ):
        return (
            False,
            None,
            [
                "not_object"
            ],
        )

    winner = str(
        obj.get(
            "winner",
            "",
        )
    ).strip().upper()

    if winner not in VALID_AB_WINNERS:
        return (
            False,
            obj,
            [
                "invalid_winner"
            ],
        )

    a = obj.get(
        "A"
    )

    b = obj.get(
        "B"
    )

    if not isinstance(
        a,
        dict,
    ):
        return (
            False,
            obj,
            [
                "invalid_A"
            ],
        )

    if not isinstance(
        b,
        dict,
    ):
        return (
            False,
            obj,
            [
                "invalid_B"
            ],
        )

    required = [
        "acceptable",
        "major_error",
    ]

    for key in required:
        if not isinstance(
            a.get(
                key
            ),
            bool,
        ):
            return (
                False,
                obj,
                [
                    f"invalid_A_{key}"
                ],
            )

        if not isinstance(
            b.get(
                key
            ),
            bool,
        ):
            return (
                False,
                obj,
                [
                    f"invalid_B_{key}"
                ],
            )

    # Deterministic schema normalization learned from Step18E2:
    # major_error=True always dominates acceptable=True.
    if a[
        "major_error"
    ]:
        a[
            "acceptable"
        ] = False

    if b[
        "major_error"
    ]:
        b[
            "acceptable"
        ] = False

    errors = []

    if (
        a[
            "major_error"
        ]
        and
        not b[
            "major_error"
        ]
        and
        winner
        ==
        "A"
    ):
        errors.append(
            "winner_A_unique_major"
        )

    if (
        b[
            "major_error"
        ]
        and
        not a[
            "major_error"
        ]
        and
        winner
        ==
        "B"
    ):
        errors.append(
            "winner_B_unique_major"
        )

    if (
        winner
        ==
        "BOTH_BAD"
        and
        not (
            a[
                "major_error"
            ]
            and
            b[
                "major_error"
            ]
        )
    ):
        errors.append(
            "both_bad_without_two_major_errors"
        )

    normalized = {
        "winner": winner,
        "confidence": str(
            obj.get(
                "confidence",
                "",
            )
        ).strip().upper(),
        "A": {
            "acceptable": bool(
                a[
                    "acceptable"
                ]
            ),
            "major_error": bool(
                a[
                    "major_error"
                ]
            ),
        },
        "B": {
            "acceptable": bool(
                b[
                    "acceptable"
                ]
            ),
            "major_error": bool(
                b[
                    "major_error"
                ]
            ),
        },
        "reason": str(
            obj.get(
                "reason",
                "",
            )
        ).strip(),
    }

    return (
        len(errors)
        ==
        0,
        normalized,
        errors,
    )


def map_ab_to_model(
    winner_ab: str,
    a_model: str,
    b_model: str,
) -> str:
    if winner_ab == "A":
        return a_model

    if winner_ab == "B":
        return b_model

    if winner_ab in {
        "TIE",
        "BOTH_BAD",
    }:
        return winner_ab

    return "PARSE_ERROR"


def model_quality_from_ab(
    normalized: dict,
    a_model: str,
    b_model: str,
):
    if (
        a_model
        ==
        "OPUS"
    ):
        opus = normalized[
            "A"
        ]
        madlad = normalized[
            "B"
        ]

    else:
        opus = normalized[
            "B"
        ]
        madlad = normalized[
            "A"
        ]

    return {
        "opus_acceptable": bool(
            opus[
                "acceptable"
            ]
        ),
        "opus_major_error": bool(
            opus[
                "major_error"
            ]
        ),
        "madlad_acceptable": bool(
            madlad[
                "acceptable"
            ]
        ),
        "madlad_major_error": bool(
            madlad[
                "major_error"
            ]
        ),
    }


def generate_batch(
    *,
    prompts,
    tokenizer,
    model,
    max_input_tokens,
    max_new_tokens,
):
    rendered = [
        render_chat_prompt(
            tokenizer,
            p,
        )
        for p in prompts
    ]

    encoded = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    )

    input_lengths = (
        encoded[
            "attention_mask"
        ]
        .sum(
            dim=1
        )
        .tolist()
    )

    if torch.cuda.is_available():
        encoded = {
            k: v.cuda(
                non_blocking=True
            )
            for k, v
            in encoded.items()
        }

        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=(
                tokenizer.eos_token_id
            ),
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    prompt_width = encoded[
        "input_ids"
    ].shape[
        1
    ]

    texts = []

    for sequence in generated:
        output_tokens = sequence[
            prompt_width:
        ]

        texts.append(
            tokenizer.decode(
                output_tokens,
                skip_special_tokens=True,
            ).strip()
        )

    return (
        texts,
        input_lengths,
        elapsed,
    )


def make_metadata(
    row: pd.Series,
):
    candidate_id = str(
        row[
            "kd_candidate_id"
        ]
    )

    swap = (
        stable_int(
            candidate_id
        )
        %
        2
        ==
        1
    )

    if not swap:
        a_model = "OPUS"
        a_text = str(
            row[
                "opus_prediction"
            ]
        )
        b_model = "MADLAD"
        b_text = str(
            row[
                "madlad_prediction"
            ]
        )

    else:
        a_model = "MADLAD"
        a_text = str(
            row[
                "madlad_prediction"
            ]
        )
        b_model = "OPUS"
        b_text = str(
            row[
                "opus_prediction"
            ]
        )

    return {
        "candidate_id": candidate_id,
        "a_model": a_model,
        "a_text": a_text,
        "b_model": b_model,
        "b_text": b_text,
        "swapped": bool(
            swap
        ),
    }


def build_valid_record(
    *,
    row,
    meta,
    normalized,
    raw_output,
    input_tokens,
    retry_count,
    elapsed_per_row,
):
    winner_ab = normalized[
        "winner"
    ]

    winner_model = map_ab_to_model(
        winner_ab,
        meta[
            "a_model"
        ],
        meta[
            "b_model"
        ],
    )

    quality = model_quality_from_ab(
        normalized,
        meta[
            "a_model"
        ],
        meta[
            "b_model"
        ],
    )

    return {
        "kd_candidate_id": str(
            row[
                "kd_candidate_id"
            ]
        ),
        "decision_source": (
            "QWEN_18F_PRODUCTION"
        ),
        "candidate_a_model": (
            meta[
                "a_model"
            ]
        ),
        "candidate_b_model": (
            meta[
                "b_model"
            ]
        ),
        "candidate_order_swapped": (
            meta[
                "swapped"
            ]
        ),
        "winner_ab": winner_ab,
        "winner_model": (
            winner_model
        ),
        "confidence": (
            normalized[
                "confidence"
            ]
        ),
        "opus_acceptable": (
            quality[
                "opus_acceptable"
            ]
        ),
        "opus_major_error": (
            quality[
                "opus_major_error"
            ]
        ),
        "madlad_acceptable": (
            quality[
                "madlad_acceptable"
            ]
        ),
        "madlad_major_error": (
            quality[
                "madlad_major_error"
            ]
        ),
        "reason": (
            normalized[
                "reason"
            ]
        ),
        "parse_success": True,
        "logical_consistency": True,
        "input_tokens": int(
            input_tokens
        ),
        "retry_count": int(
            retry_count
        ),
        "generation_seconds_per_row": float(
            elapsed_per_row
        ),
        "raw_qwen_output": (
            raw_output
        ),
    }


def judge_new_rows(
    *,
    rows,
    tokenizer,
    model,
    batch_size,
    max_input_tokens,
    max_new_tokens,
    max_retries,
    checkpoint_path,
    unresolved_path,
):
    existing = load_jsonl(
        checkpoint_path
    )

    if len(
        existing
    ):
        if (
            existing[
                "kd_candidate_id"
            ]
            .duplicated()
            .any()
        ):
            raise RuntimeError(
                "Duplicate kd_candidate_id in valid checkpoint."
            )

        completed = set(
            existing[
                "kd_candidate_id"
            ]
            .astype(str)
            .tolist()
        )

    else:
        completed = set()

    pending = rows.loc[
        ~rows[
            "kd_candidate_id"
        ]
        .astype(str)
        .isin(
            completed
        )
    ].copy()

    print(
        "\n18F production judge:"
    )

    print(
        "Existing valid checkpoint rows:",
        len(
            completed
        ),
    )

    print(
        "Pending new Qwen rows:",
        len(
            pending
        ),
    )

    total = len(
        pending
    )

    all_new_valid = []

    unresolved_final = []

    processed = 0

    for start in range(
        0,
        total,
        batch_size,
    ):
        batch = pending.iloc[
            start:
            min(
                start
                +
                batch_size,
                total,
            )
        ].copy()

        work_items = []

        for _, row in batch.iterrows():
            meta = make_metadata(
                row
            )

            work_items.append(
                {
                    "row": row,
                    "meta": meta,
                    "retry_count": 0,
                    "last_raw": "",
                    "last_errors": [],
                    "last_input_tokens": 0,
                }
            )

        valid_batch_records = []

        remaining = work_items

        for attempt in range(
            0,
            max_retries + 1,
        ):
            if not remaining:
                break

            prompts = []

            for item in remaining:
                retry_note = ""

                if attempt > 0:
                    retry_note = (
                        "The previous response was invalid or logically inconsistent. "
                        "Re-check winner versus major_error fields. "
                        "Return only the compact JSON schema."
                    )

                prompts.append(
                    build_prompt(
                        row=item[
                            "row"
                        ],
                        a_model=item[
                            "meta"
                        ][
                            "a_model"
                        ],
                        a_text=item[
                            "meta"
                        ][
                            "a_text"
                        ],
                        b_model=item[
                            "meta"
                        ][
                            "b_model"
                        ],
                        b_text=item[
                            "meta"
                        ][
                            "b_text"
                        ],
                        retry_note=retry_note,
                    )
                )

            outputs, input_lengths, elapsed = generate_batch(
                prompts=prompts,
                tokenizer=tokenizer,
                model=model,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )

            elapsed_per_row = (
                elapsed
                /
                len(
                    remaining
                )
                if remaining
                else 0.0
            )

            next_remaining = []

            for (
                item,
                raw,
                input_tokens,
            ) in zip(
                remaining,
                outputs,
                input_lengths,
            ):
                obj = extract_json_object(
                    raw
                )

                valid, normalized, errors = normalize_and_validate(
                    obj
                )

                item[
                    "retry_count"
                ] = attempt

                item[
                    "last_raw"
                ] = raw

                item[
                    "last_errors"
                ] = errors

                item[
                    "last_input_tokens"
                ] = int(
                    input_tokens
                )

                if valid:
                    record = build_valid_record(
                        row=item[
                            "row"
                        ],
                        meta=item[
                            "meta"
                        ],
                        normalized=normalized,
                        raw_output=raw,
                        input_tokens=input_tokens,
                        retry_count=attempt,
                        elapsed_per_row=elapsed_per_row,
                    )

                    valid_batch_records.append(
                        record
                    )

                else:
                    next_remaining.append(
                        item
                    )

            remaining = (
                next_remaining
            )

        if remaining:
            for item in remaining:
                unresolved_final.append(
                    {
                        "kd_candidate_id": str(
                            item[
                                "row"
                            ][
                                "kd_candidate_id"
                            ]
                        ),
                        "errors": item[
                            "last_errors"
                        ],
                        "retry_count": item[
                            "retry_count"
                        ],
                        "input_tokens": item[
                            "last_input_tokens"
                        ],
                        "raw_qwen_output": item[
                            "last_raw"
                        ],
                    }
                )

        # IMPORTANT:
        # Only logically valid decisions are written to the resume checkpoint.
        append_jsonl(
            valid_batch_records,
            checkpoint_path,
        )

        all_new_valid.extend(
            valid_batch_records
        )

        processed += len(
            batch
        )

        counts = Counter(
            r[
                "winner_model"
            ]
            for r in valid_batch_records
        )

        print(
            f"{processed}/{total}"
            f" | valid {len(valid_batch_records)}/{len(batch)}"
            f" | {dict(counts)}"
        )

    if unresolved_final:
        # Overwrite unresolved diagnostic with the current run's unresolved set.
        with unresolved_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            for row in unresolved_final:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    +
                    "\n"
                )

    elif unresolved_path.exists():
        unresolved_path.unlink()

    final_checkpoint = load_jsonl(
        checkpoint_path
    )

    return (
        final_checkpoint,
        unresolved_final,
    )


def build_reused_calibration(
    *,
    qwen_rows,
    calibration,
):
    reusable = calibration.loc[
        calibration[
            "kd_candidate_id"
        ]
        .astype(str)
        .isin(
            qwen_rows[
                "kd_candidate_id"
            ]
            .astype(str)
        )
    ].copy()

    if not len(
        reusable
    ):
        return pd.DataFrame()

    rows = []

    for _, r in reusable.iterrows():
        rows.append(
            {
                "kd_candidate_id": str(
                    r[
                        "kd_candidate_id"
                    ]
                ),
                "decision_source": (
                    "REUSED_18E2_DUAL_ORDER"
                ),
                "candidate_a_model": "",
                "candidate_b_model": "",
                "candidate_order_swapped": False,
                "winner_ab": "",
                "winner_model": str(
                    r[
                        "final_winner"
                    ]
                ),
                "confidence": str(
                    r.get(
                        "adjudication_confidence",
                        "",
                    )
                    if (
                        str(
                            r.get(
                                "final_resolution",
                                "",
                            )
                        )
                        ==
                        "INDEPENDENT_ADJUDICATION"
                    )
                    else r.get(
                        "pass1_confidence",
                        "",
                    )
                ),
                "opus_acceptable": bool(
                    r[
                        "final_opus_acceptable"
                    ]
                ),
                "opus_major_error": bool(
                    r[
                        "final_opus_major_error"
                    ]
                ),
                "madlad_acceptable": bool(
                    r[
                        "final_madlad_acceptable"
                    ]
                ),
                "madlad_major_error": bool(
                    r[
                        "final_madlad_major_error"
                    ]
                ),
                "reason": str(
                    r.get(
                        "adjudication_reason",
                        "",
                    )
                    if (
                        str(
                            r.get(
                                "final_resolution",
                                "",
                            )
                        )
                        ==
                        "INDEPENDENT_ADJUDICATION"
                    )
                    else r.get(
                        "pass1_reason",
                        "",
                    )
                ),
                "parse_success": True,
                "logical_consistency": True,
                "input_tokens": 0,
                "retry_count": 0,
                "generation_seconds_per_row": 0.0,
                "raw_qwen_output": "",
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    args = parse_args()

    root = (
        Path(
            args.project_root
        ).resolve()
        if args.project_root
        else infer_project_root()
    )

    projection_path = resolve_path(
        root,
        args.projection,
    )

    calibration_path = resolve_path(
        root,
        args.calibration_results,
    )

    model_path = resolve_path(
        root,
        args.model_path,
    )

    output_dir = resolve_path(
        root,
        args.output_dir,
    )

    if not projection_path.exists():
        raise FileNotFoundError(
            projection_path
        )

    if not calibration_path.exists():
        raise FileNotFoundError(
            calibration_path
        )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        /
        "checkpoint_qwen_valid_v1.jsonl"
    )

    unresolved_path = (
        output_dir
        /
        "unresolved_qwen_attempts_v1.jsonl"
    )

    final_output = (
        output_dir
        /
        "selective_qwen_results_v1.parquet"
    )

    summary_output = (
        output_dir
        /
        "selective_qwen_summary_v1.csv"
    )

    report_output = (
        output_dir
        /
        "selective_qwen_report_v1.json"
    )

    if args.overwrite:
        for p in [
            checkpoint_path,
            unresolved_path,
            final_output,
            summary_output,
            report_output,
        ]:
            if p.exists():
                p.unlink()

    elif final_output.exists():
        raise RuntimeError(
            "Final Step18F output already exists. "
            "Use --overwrite only if you intentionally want to rebuild it."
        )

    print(
        "=" * 115
    )

    print(
        "ZH-EN DISTILLATION PIPELINE"
    )

    print(
        "STEP 18F - SELECTIVE QWEN PRODUCTION TEACHER JUDGE"
    )

    print(
        "=" * 115
    )

    print(
        "\nProjection:"
    )

    print(
        projection_path
    )

    print(
        "\n18E2 calibration results:"
    )

    print(
        calibration_path
    )

    projection = pd.read_parquet(
        projection_path
    ).copy()

    calibration = pd.read_parquet(
        calibration_path
    ).copy()

    required_projection = {
        "kd_candidate_id",
        "direction",
        "source_dataset",
        "calibration_band",
        "qwen_required",
        "source_lang",
        "target_lang",
        "source_text",
        "human_reference",
        "opus_prediction",
        "madlad_prediction",
    }

    missing = (
        required_projection
        -
        set(
            projection.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Projection missing columns: {sorted(missing)}"
        )

    required_calibration = {
        "kd_candidate_id",
        "final_winner",
        "final_opus_acceptable",
        "final_opus_major_error",
        "final_madlad_acceptable",
        "final_madlad_major_error",
        "final_resolution",
    }

    missing_cal = (
        required_calibration
        -
        set(
            calibration.columns
        )
    )

    if missing_cal:
        raise RuntimeError(
            f"Calibration missing columns: {sorted(missing_cal)}"
        )

    qwen_required_mask = (
        projection[
            "qwen_required"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    qwen_rows = projection.loc[
        qwen_required_mask
    ].copy()

    qwen_rows[
        "kd_candidate_id"
    ] = qwen_rows[
        "kd_candidate_id"
    ].astype(
        str
    )

    calibration[
        "kd_candidate_id"
    ] = calibration[
        "kd_candidate_id"
    ].astype(
        str
    )

    reused = build_reused_calibration(
        qwen_rows=qwen_rows,
        calibration=calibration,
    )

    reused_ids = (
        set(
            reused[
                "kd_candidate_id"
            ]
            .astype(str)
            .tolist()
        )
        if len(
            reused
        )
        else set()
    )

    new_rows = qwen_rows.loc[
        ~qwen_rows[
            "kd_candidate_id"
        ]
        .isin(
            reused_ids
        )
    ].copy()

    print(
        "\n20K rows:",
        len(
            projection
        ),
    )

    print(
        "Qwen required:",
        len(
            qwen_rows
        ),
    )

    print(
        "Reusable from 18E2:",
        len(
            reused
        ),
    )

    print(
        "New Qwen judgments required:",
        len(
            new_rows
        ),
    )

    if (
        len(
            reused
        )
        >
        0
    ):
        invalid_reused = set(
            reused[
                "winner_model"
            ]
            .astype(str)
            .unique()
        ) - VALID_MODEL_WINNERS

        if invalid_reused:
            raise RuntimeError(
                f"Invalid reused winners: {sorted(invalid_reused)}"
            )

    tokenizer = None
    model = None
    params = 0

    if len(
        new_rows
    ):
        print(
            "\nGPU:",
            gpu_name(),
        )

        tokenizer, model, params = load_qwen(
            model_path
        )

        checkpoint, unresolved = judge_new_rows(
            rows=new_rows,
            tokenizer=tokenizer,
            model=model,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            max_retries=args.max_retries,
            checkpoint_path=checkpoint_path,
            unresolved_path=unresolved_path,
        )

    else:
        checkpoint = load_jsonl(
            checkpoint_path
        )
        unresolved = []

    if len(
        checkpoint
    ):
        checkpoint[
            "kd_candidate_id"
        ] = checkpoint[
            "kd_candidate_id"
        ].astype(
            str
        )

    combined = pd.concat(
        [
            reused,
            checkpoint,
        ],
        ignore_index=True,
    )

    if len(
        combined
    ):
        combined = (
            qwen_rows[
                [
                    "kd_candidate_id",
                    "direction",
                    "source_dataset",
                    "calibration_band",
                    "teacher_disagreement_score",
                ]
            ]
            .merge(
                combined,
                on="kd_candidate_id",
                how="inner",
                validate="one_to_one",
            )
        )

    unresolved_ids = (
        set(
            qwen_rows[
                "kd_candidate_id"
            ]
            .astype(str)
        )
        -
        set(
            combined[
                "kd_candidate_id"
            ]
            .astype(str)
        )
    )

    assertions = {
        "projection_rows_20000": (
            len(
                projection
            )
            ==
            20000
        ),
        "qwen_required_nonempty": (
            len(
                qwen_rows
            )
            >
            0
        ),
        "qwen_candidate_id_unique": (
            qwen_rows[
                "kd_candidate_id"
            ]
            .is_unique
        ),
        "reused_candidate_id_unique": (
            reused[
                "kd_candidate_id"
            ]
            .is_unique
            if len(
                reused
            )
            else True
        ),
        "checkpoint_candidate_id_unique": (
            checkpoint[
                "kd_candidate_id"
            ]
            .is_unique
            if len(
                checkpoint
            )
            else True
        ),
        "all_qwen_rows_resolved": (
            len(
                unresolved_ids
            )
            ==
            0
        ),
        "final_result_count_matches_qwen_required": (
            len(
                combined
            )
            ==
            len(
                qwen_rows
            )
        ),
        "final_winners_valid": (
            combined[
                "winner_model"
            ]
            .astype(str)
            .isin(
                VALID_MODEL_WINNERS
            )
            .all()
            if len(
                combined
            )
            else False
        ),
    }

    # If unresolved remain, preserve partial checkpoint/results for resume,
    # but do not publish a misleading "complete" final artifact.
    failed = [
        k
        for k, v in assertions.items()
        if not bool(
            v
        )
    ]

    if failed:
        print(
            "\n"
            +
            "=" * 115
        )

        print(
            "STEP 18F INCOMPLETE"
        )

        print(
            "=" * 115
        )

        print(
            "\nUnresolved rows:",
            len(
                unresolved_ids
            ),
        )

        print(
            "Valid new checkpoint rows:",
            len(
                checkpoint
            ),
        )

        print(
            "Reusable rows:",
            len(
                reused
            ),
        )

        print(
            "\nFailed assertions:"
        )

        for key in failed:
            print(
                key
            )

        print(
            "\nResume WITHOUT --overwrite after inspecting unresolved rows."
        )

        raise RuntimeError(
            "STEP18F incomplete; valid checkpoint is preserved."
        )

    combined.to_parquet(
        final_output,
        index=False,
    )

    summary = (
        combined.groupby(
            [
                "direction",
                "source_dataset",
                "calibration_band",
                "decision_source",
                "winner_model",
            ],
            dropna=False,
        )
        .agg(
            rows=(
                "kd_candidate_id",
                "size",
            ),
            mean_disagreement_score=(
                "teacher_disagreement_score",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "direction",
                "source_dataset",
                "calibration_band",
                "decision_source",
                "winner_model",
            ]
        )
    )

    summary.to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    winner_counts = {
        str(k): int(v)
        for k, v in combined[
            "winner_model"
        ]
        .value_counts()
        .to_dict()
        .items()
    }

    source_counts = {
        str(k): int(v)
        for k, v in combined[
            "decision_source"
        ]
        .value_counts()
        .to_dict()
        .items()
    }

    report = {
        "step": "18F",
        "step_version": STEP_VERSION,
        "inputs": {
            "projection": str(
                projection_path
            ),
            "calibration_results": str(
                calibration_path
            ),
        },
        "model": {
            "path": str(
                model_path
            ),
            "parameters": int(
                params
            ),
        },
        "generation": {
            "batch_size": int(
                args.batch_size
            ),
            "max_input_tokens": int(
                args.max_input_tokens
            ),
            "max_new_tokens": int(
                args.max_new_tokens
            ),
            "max_retries": int(
                args.max_retries
            ),
            "dual_order": False,
            "balanced_candidate_position": True,
            "schema": (
                "compact production pairwise"
            ),
            "invalid_results_written_to_resume_checkpoint": False,
        },
        "counts": {
            "projection_rows": int(
                len(
                    projection
                )
            ),
            "qwen_required_rows": int(
                len(
                    qwen_rows
                )
            ),
            "reused_18e2_rows": int(
                len(
                    reused
                )
            ),
            "new_qwen_rows": int(
                len(
                    new_rows
                )
            ),
            "final_resolved_rows": int(
                len(
                    combined
                )
            ),
            "winner": winner_counts,
            "decision_source": source_counts,
        },
        "assertions": {
            k: bool(v)
            for k, v in assertions.items()
        },
        "outputs": {
            "valid_checkpoint": str(
                checkpoint_path
            ),
            "results": str(
                final_output
            ),
            "summary": str(
                summary_output
            ),
        },
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": (
            "READY_FOR_STEP_18G_BUILD_FINAL_KD_DATASET"
        ),
    }

    save_json(
        report,
        report_output,
    )

    print(
        "\n"
        +
        "=" * 115
    )

    print(
        "STEP 18F RESULT"
    )

    print(
        "=" * 115
    )

    print(
        "\nQwen required rows:",
        len(
            qwen_rows
        ),
    )

    print(
        "Reused Step18E2 rows:",
        len(
            reused
        ),
    )

    print(
        "New Qwen rows:",
        len(
            new_rows
        ),
    )

    print(
        "Final resolved:",
        len(
            combined
        ),
        "/",
        len(
            qwen_rows
        ),
    )

    print(
        "\nWinner distribution:"
    )

    print(
        combined[
            "winner_model"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDecision source:"
    )

    print(
        combined[
            "decision_source"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDirection × source × band × winner:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nAssertions:"
    )

    for k, v in assertions.items():
        print(
            f"{k}: {bool(v)}"
        )

    print(
        "\nResults:"
    )

    print(
        final_output
    )

    print(
        "\nReport:"
    )

    print(
        report_output
    )

    print(
        "\nSTATUS:"
    )

    print(
        "READY_FOR_STEP_18G_BUILD_FINAL_KD_DATASET"
    )

    if model is not None:
        del model

    if tokenizer is not None:
        del tokenizer

    cleanup_cuda()


if __name__ == "__main__":
    main()
