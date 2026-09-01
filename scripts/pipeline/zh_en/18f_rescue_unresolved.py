from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Rescue only unresolved Step18F rows with a stricter compact-JSON prompt."
    )
    p.add_argument(
        "--project_root",
        default="/root/autodl-tmp/fourlang_translation",
    )
    p.add_argument(
        "--model_path",
        default="/root/autodl-tmp/models/Qwen3-8B",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    p.add_argument(
        "--max_input_tokens",
        type=int,
        default=1536,
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=320,
    )
    p.add_argument(
        "--max_retries",
        type=int,
        default=2,
    )
    return p.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def try_parse_with_brace_repairs(step18f, raw: str):
    raw = str(raw).strip()

    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    elif raw.startswith("```"):
        raw = raw[len("```"):].strip()

    if raw.endswith("```"):
        raw = raw[:-3].strip()

    candidates = [raw]

    if raw.startswith('"winner"'):
        candidates.append("{" + raw)
        candidates.append("{" + raw + "}")

    for candidate in candidates:
        obj = step18f.extract_json_object(candidate)
        if obj is None:
            continue

        valid, normalized, errors = step18f.normalize_and_validate(obj)

        if valid:
            return True, normalized, candidate, []

    return False, None, raw, ["not_object_or_invalid_logic"]


def build_rescue_prompt(row, meta, attempt: int):
    source = str(row["source_text"]).strip()
    human = str(row["human_reference"]).strip()

    a_model = meta["a_model"]
    a_text = str(meta["a_text"]).strip()

    b_model = meta["b_model"]
    b_text = str(meta["b_text"]).strip()

    retry = ""
    if attempt > 0:
        retry = """
IMPORTANT RETRY:
Your previous answer did not follow the output format.
Do not analyze.
Do not write prose before the JSON.
Do not use Markdown.
Return one compact JSON object only.
"""

    return f"""You are a machine-translation safety judge.

Judge ONLY semantic faithfulness to SOURCE.
HUMAN REFERENCE is auxiliary and may be noisy.

SOURCE LANGUAGE: {row["source_lang"]}
TARGET LANGUAGE: {row["target_lang"]}

SOURCE:
{source}

HUMAN REFERENCE:
{human}

A ({a_model}):
{a_text}

B ({b_model}):
{b_text}

Decision rules:
- A = A is meaningfully safer/better.
- B = B is meaningfully safer/better.
- TIE = both are acceptable and neither has a meaningful semantic advantage.
- BOTH_BAD = both have substantive semantic errors.
- major_error=true => acceptable=false.
- If only A has a major error, winner cannot be A.
- If only B has a major error, winner cannot be B.

OUTPUT RULES — STRICT:
1. Your FIRST character must be {{
2. Your LAST character must be }}
3. Output JSON only.
4. No analysis.
5. No Markdown.
6. No explanation before or after JSON.
7. Keep reason under 25 words.

Return exactly:
{{
  "winner":"A|B|TIE|BOTH_BAD",
  "confidence":"HIGH|MEDIUM|LOW",
  "A":{{"acceptable":true,"major_error":false}},
  "B":{{"acceptable":true,"major_error":false}},
  "reason":"brief reason"
}}
{retry}"""


def main():
    args = parse_args()

    root = Path(args.project_root).resolve()

    original_script = (
        root
        / "scripts/pipeline/zh_en/"
          "18f_selective_qwen_teacher_judge.py"
    )

    projection_path = (
        root
        / "data/distillation/zh_en/v1/"
          "18e_qwen_calibration/18e3_gate_policy/"
          "teacher_gate_projection_20k_v1.parquet"
    )

    outdir = (
        root
        / "data/distillation/zh_en/v1/"
          "18f_selective_qwen"
    )

    unresolved_path = (
        outdir
        / "unresolved_qwen_attempts_v1.jsonl"
    )

    checkpoint_path = (
        outdir
        / "checkpoint_qwen_valid_v1.jsonl"
    )

    unresolved_backup = (
        outdir
        / "unresolved_qwen_attempts_v1.before_rescue.jsonl"
    )

    rescue_diag = (
        outdir
        / "unresolved_qwen_attempts_v1.after_rescue.jsonl"
    )

    if not original_script.exists():
        raise FileNotFoundError(original_script)

    if not projection_path.exists():
        raise FileNotFoundError(projection_path)

    if not unresolved_path.exists():
        raise FileNotFoundError(unresolved_path)

    # Load original Step18F utilities.
    spec = importlib.util.spec_from_file_location(
        "step18f",
        original_script,
    )

    step18f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step18f)

    projection = pd.read_parquet(projection_path).copy()
    projection["kd_candidate_id"] = projection["kd_candidate_id"].astype(str)
    projection = projection.set_index("kd_candidate_id", drop=False)

    unresolved = load_rows(unresolved_path)

    unresolved_backup.write_text(
        unresolved_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    existing = step18f.load_jsonl(checkpoint_path)

    if len(existing):
        existing_ids = set(
            existing["kd_candidate_id"].astype(str).tolist()
        )
    else:
        existing_ids = set()

    work = []

    for item in unresolved:
        candidate_id = str(item["kd_candidate_id"])

        if candidate_id in existing_ids:
            continue

        if candidate_id not in projection.index:
            raise RuntimeError(
                f"Candidate missing from projection: {candidate_id}"
            )

        row = projection.loc[candidate_id]
        meta = step18f.make_metadata(row)

        work.append(
            {
                "row": row,
                "meta": meta,
                "candidate_id": candidate_id,
                "last_raw": "",
                "last_errors": [],
                "last_input_tokens": 0,
                "retry_count": 0,
            }
        )

    print("=" * 110)
    print("STEP 18F RESCUE - STRICT JSON MODE")
    print("=" * 110)

    print("Original unresolved:", len(unresolved))
    print("Already in valid checkpoint:", len(unresolved) - len(work))
    print("Pending rescue:", len(work))
    print("Batch size:", args.batch_size)
    print("Max new tokens:", args.max_new_tokens)

    if not work:
        print("\nNothing to rescue.")
        return

    tokenizer, model, _ = step18f.load_qwen(
        Path(args.model_path)
    )

    successful = []
    remaining = work

    for attempt in range(args.max_retries + 1):
        if not remaining:
            break

        print(
            f"\nRESCUE ATTEMPT {attempt + 1}/{args.max_retries + 1}"
        )

        next_remaining = []
        processed = 0

        for start in range(
            0,
            len(remaining),
            args.batch_size,
        ):
            batch_items = remaining[
                start:
                min(
                    start + args.batch_size,
                    len(remaining),
                )
            ]

            prompts = [
                build_rescue_prompt(
                    item["row"],
                    item["meta"],
                    attempt,
                )
                for item in batch_items
            ]

            outputs, input_lengths, elapsed = step18f.generate_batch(
                prompts=prompts,
                tokenizer=tokenizer,
                model=model,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
            )

            elapsed_per_row = (
                elapsed / len(batch_items)
                if batch_items
                else 0.0
            )

            valid_records = []

            for item, raw, input_tokens in zip(
                batch_items,
                outputs,
                input_lengths,
            ):
                valid, normalized, repaired_raw, errors = (
                    try_parse_with_brace_repairs(
                        step18f,
                        raw,
                    )
                )

                item["last_raw"] = raw
                item["last_errors"] = errors
                item["last_input_tokens"] = int(input_tokens)
                item["retry_count"] = attempt

                if valid:
                    record = step18f.build_valid_record(
                        row=item["row"],
                        meta=item["meta"],
                        normalized=normalized,
                        raw_output=repaired_raw,
                        input_tokens=input_tokens,
                        retry_count=attempt,
                        elapsed_per_row=elapsed_per_row,
                    )

                    record["decision_source"] = (
                        "QWEN_18F_RESCUE_STRICT_JSON"
                    )

                    valid_records.append(record)
                    successful.append(record)

                else:
                    next_remaining.append(item)

            # Only valid records enter checkpoint.
            step18f.append_jsonl(
                valid_records,
                checkpoint_path,
            )

            processed += len(batch_items)

            counts = Counter(
                r["winner_model"]
                for r in valid_records
            )

            print(
                f"{processed}/{len(remaining)}"
                f" | valid {len(valid_records)}/{len(batch_items)}"
                f" | {dict(counts)}"
            )

        remaining = next_remaining

        print(
            f"Attempt {attempt + 1} remaining:",
            len(remaining),
        )

    # Rewrite unresolved with only still-failed rows.
    with unresolved_path.open("w", encoding="utf-8") as f:
        for item in remaining:
            row = {
                "kd_candidate_id": item["candidate_id"],
                "errors": item["last_errors"],
                "retry_count": item["retry_count"],
                "input_tokens": item["last_input_tokens"],
                "raw_qwen_output": item["last_raw"],
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Separate diagnostic snapshot too.
    with rescue_diag.open("w", encoding="utf-8") as f:
        for item in remaining:
            f.write(
                json.dumps(
                    {
                        "kd_candidate_id": item["candidate_id"],
                        "errors": item["last_errors"],
                        "retry_count": item["retry_count"],
                        "input_tokens": item["last_input_tokens"],
                        "raw_qwen_output": item["last_raw"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    final_checkpoint = step18f.load_jsonl(checkpoint_path)

    print("\n" + "=" * 110)
    print("STEP 18F RESCUE RESULT")
    print("=" * 110)

    print("Rescue input:", len(work))
    print("Rescued:", len(successful))
    print("Still unresolved:", len(remaining))
    print("Valid checkpoint total:", len(final_checkpoint))

    if len(successful):
        print("\nRescue winner distribution:")
        print(
            pd.Series(
                [r["winner_model"] for r in successful]
            )
            .value_counts()
            .to_string()
        )

    print("\nUnresolved file:")
    print(unresolved_path)

    if len(remaining) == 0:
        print("\nSTATUS:")
        print("RESCUE_COMPLETE_RUN_STEP18F_FINALIZER")
    else:
        print("\nSTATUS:")
        print("RESCUE_PARTIAL_INSPECT_REMAINING")


if __name__ == "__main__":
    main()
