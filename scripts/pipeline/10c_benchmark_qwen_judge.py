from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """
You are a strict bilingual translation quality judge.

Your job is NOT to judge whether two translations are literally identical.
Different wording is allowed if the meaning is faithfully preserved.

You are evaluating a MADLAD teacher translation that may later be used
to train a smaller translation model.

Judge the TEACHER TRANSLATION against:
1. the SOURCE sentence;
2. the HUMAN/REAL REFERENCE.

Important rules:
- Semantic faithfulness matters more than wording similarity.
- A teacher translation can be PASS even when it differs from the reference.
- Do not punish valid paraphrases or synonyms.
- Numbers, dates, times, entities and negation must be preserved accurately.
- Do not treat the human reference as the only possible correct translation.
- For English-to-Uzbek, the teacher output is normalized to Latin Uzbek.
- Be conservative with FAIL.
- MINOR means essentially correct with a small non-critical issue.
- UNCERTAIN means you genuinely cannot judge reliably.

Labels:
PASS
MINOR
FAIL
UNCERTAIN

Teacher usefulness:
HIGH
MEDIUM
LOW
REJECT

Return ONLY one JSON object.

Required schema:

{
  "label": "PASS|MINOR|FAIL|UNCERTAIN",
  "confidence": 0.0,
  "semantic_equivalent": true,
  "teacher_usefulness": "HIGH|MEDIUM|LOW|REJECT",
  "errors": {
    "omission": false,
    "addition": false,
    "mistranslation": false,
    "number_error": false,
    "time_error": false,
    "entity_error": false,
    "negation_error": false
  },
  "reason": "brief reason"
}
""".strip()


VALID_LABELS = {
    "PASS",
    "MINOR",
    "FAIL",
    "UNCERTAIN",
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qwen_path",
        type=str,
        default="/root/autodl-tmp/models/Qwen3-8B",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="2,3,4,6",
    )

    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=192,
    )

    return parser.parse_args()


def build_user_prompt(row):

    if row["direction"] == "en_uz":
        direction = "English -> Uzbek (Latin)"
    else:
        direction = "Uzbek (Latin) -> English"

    return f"""
DIRECTION:
{direction}

SOURCE:
{row["source_text"]}

HUMAN REFERENCE:
{row["real_reference"]}

TEACHER TRANSLATION:
{row["teacher_prediction"]}

Evaluate whether the TEACHER TRANSLATION is a good additional
training target for this SOURCE.
""".strip()


def render_prompt(
    tokenizer,
    user_prompt,
):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    try:

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def extract_json(text):

    text = str(text).strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass

    start = text.find("{")

    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):

        c = text[i]

        if in_string:

            if escaped:
                escaped = False

            elif c == "\\":
                escaped = True

            elif c == '"':
                in_string = False

            continue

        if c == '"':
            in_string = True

        elif c == "{":
            depth += 1

        elif c == "}":

            depth -= 1

            if depth == 0:

                candidate = text[start:i + 1]

                try:

                    obj = json.loads(candidate)

                    if isinstance(obj, dict):
                        return obj

                except Exception:
                    return None

    return None


def valid_json_result(obj):

    if not isinstance(obj, dict):
        return False

    label = str(
        obj.get("label", "")
    ).strip().upper()

    return label in VALID_LABELS


def prepare_benchmark_samples(
    df,
    samples,
):

    if samples % 2 != 0:
        raise ValueError(
            "--samples must be even."
        )

    each = samples // 2

    parts = []

    for direction in [
        "en_uz",
        "uz_en",
    ]:

        sub = (
            df[
                df["direction"] == direction
            ]
            .copy()
        )

        # deterministic pseudo-random order
        sub["_benchmark_key"] = (
            sub["candidate_id"]
            .astype(str)
            .map(
                lambda x:
                    hash(
                        "benchmark_v1|" + x
                    )
            )
        )

        sub = (
            sub
            .sort_values(
                "_benchmark_key"
            )
            .head(each)
            .drop(
                columns=["_benchmark_key"]
            )
        )

        parts.append(sub)

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    # Same exact ordering for every batch test
    result = (
        result
        .sort_values(
            [
                "direction",
                "source_word_count",
                "candidate_id",
            ]
        )
        .reset_index(drop=True)
    )

    return result


def run_benchmark(
    model,
    tokenizer,
    prompts,
    batch_size,
    max_input_length,
    max_new_tokens,
):

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_samples = 0
    valid_json = 0

    start_all = time.perf_counter()

    for start in range(
        0,
        len(prompts),
        batch_size,
    ):

        batch_prompts = prompts[
            start:start + batch_size
        ]

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        encoded = {
            k: v.cuda()
            for k, v in encoded.items()
        }

        input_length = (
            encoded["input_ids"]
            .shape[1]
        )

        with torch.inference_mode():

            generated = model.generate(
                **encoded,

                max_new_tokens=
                    max_new_tokens,

                do_sample=False,
                num_beams=1,

                use_cache=True,

                pad_token_id=
                    tokenizer.pad_token_id,

                eos_token_id=
                    tokenizer.eos_token_id,
            )

        outputs = (
            tokenizer.batch_decode(
                generated[:, input_length:],
                skip_special_tokens=True,
            )
        )

        for output in outputs:

            obj = extract_json(
                output
            )

            if valid_json_result(obj):
                valid_json += 1

        total_samples += len(
            batch_prompts
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start_all
    )

    max_memory_gb = (
        torch.cuda.max_memory_allocated()
        /
        1024 ** 3
    )

    return {
        "batch_size":
            batch_size,

        "samples":
            total_samples,

        "total_seconds":
            elapsed,

        "seconds_per_sample":
            (
                elapsed
                /
                total_samples
            ),

        "samples_per_second":
            (
                total_samples
                /
                elapsed
            ),

        "max_gpu_memory_gb":
            max_memory_gb,

        "parse_success_percent":
            (
                valid_json
                /
                total_samples
                *
                100
            ),
    }


def main():

    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    calibration_file = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "10c_qwen_quality_gate"
        / "calibration_500"
        / "selected_candidates.parquet"
    )

    output_dir = (
        project_root
        / "results"
        / "benchmarks"
        / "qwen_10c"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        output_dir
        / "qwen_10c_batch_benchmark.csv"
    )

    qwen_path = Path(
        args.qwen_path
    )

    batch_sizes = [
        int(x.strip())
        for x in args.batch_sizes.split(",")
        if x.strip()
    ]

    print("=" * 100)
    print("STEP 10C PERFORMANCE BENCHMARK")
    print("=" * 100)

    print(
        "\nModel:",
        qwen_path
    )

    print(
        "Samples:",
        args.samples
    )

    print(
        "Batch sizes:",
        batch_sizes
    )

    if not calibration_file.exists():

        raise FileNotFoundError(
            calibration_file
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0)
    )

    # ========================================================
    # Data
    # ========================================================

    df = pd.read_parquet(
        calibration_file
    )

    benchmark_df = (
        prepare_benchmark_samples(
            df,
            args.samples,
        )
    )

    print(
        "\nDirection:"
    )

    print(
        benchmark_df[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(qwen_path),
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    # ========================================================
    # Model
    # ========================================================

    print(
        "Loading Qwen3-8B FP16..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(qwen_path),

            torch_dtype=
                torch.float16,

            local_files_only=True,
            trust_remote_code=True,
        )
        .cuda()
    )

    model.eval()
    model.config.use_cache = True

    # Important:
    # deterministic Judge
    model.generation_config.do_sample = False

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Base GPU memory:",
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )

    # ========================================================
    # Render prompts ONCE
    # ========================================================

    prompts = []

    for _, row in benchmark_df.iterrows():

        user_prompt = (
            build_user_prompt(row)
        )

        prompts.append(
            render_prompt(
                tokenizer,
                user_prompt,
            )
        )

    # ========================================================
    # Warm-up
    # ========================================================

    print(
        "\nWarm-up..."
    )

    warmup_prompts = prompts[:2]

    encoded = tokenizer(
        warmup_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
    )

    encoded = {
        k: v.cuda()
        for k, v in encoded.items()
    }

    with torch.inference_mode():

        _ = model.generate(
            **encoded,

            max_new_tokens=
                args.max_new_tokens,

            do_sample=False,
            num_beams=1,

            use_cache=True,

            pad_token_id=
                tokenizer.pad_token_id,

            eos_token_id=
                tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()

    print(
        "Warm-up complete."
    )

    # ========================================================
    # Benchmark
    # ========================================================

    results = []

    print("\n")
    print("=" * 100)
    print("BENCHMARK")
    print("=" * 100)

    for batch_size in batch_sizes:

        print(
            f"\nTesting batch={batch_size}..."
        )

        try:

            result = run_benchmark(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                batch_size=batch_size,
                max_input_length=
                    args.max_input_length,
                max_new_tokens=
                    args.max_new_tokens,
            )

            result[
                "status"
            ] = "OK"

            results.append(
                result
            )

            print(
                f"batch={batch_size}"
            )

            print(
                "  seconds/sample :",
                f"{result['seconds_per_sample']:.3f}"
            )

            print(
                "  samples/sec    :",
                f"{result['samples_per_second']:.3f}"
            )

            print(
                "  max GPU memory :",
                f"{result['max_gpu_memory_gb']:.2f} GB"
            )

            print(
                "  parse success  :",
                f"{result['parse_success_percent']:.2f}%"
            )

        except torch.OutOfMemoryError:

            print(
                f"batch={batch_size}: CUDA OOM"
            )

            torch.cuda.empty_cache()

            results.append(
                {
                    "batch_size":
                        batch_size,

                    "samples":
                        0,

                    "total_seconds":
                        None,

                    "seconds_per_sample":
                        None,

                    "samples_per_second":
                        None,

                    "max_gpu_memory_gb":
                        None,

                    "parse_success_percent":
                        None,

                    "status":
                        "OOM",
                }
            )

        except RuntimeError as exc:

            if (
                "out of memory"
                in str(exc).lower()
            ):

                print(
                    f"batch={batch_size}: CUDA OOM"
                )

                torch.cuda.empty_cache()

                results.append(
                    {
                        "batch_size":
                            batch_size,

                        "samples":
                            0,

                        "total_seconds":
                            None,

                        "seconds_per_sample":
                            None,

                        "samples_per_second":
                            None,

                        "max_gpu_memory_gb":
                            None,

                        "parse_success_percent":
                            None,

                        "status":
                            "OOM",
                    }
                )

            else:

                raise

    # ========================================================
    # Result
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    good = result_df[
        result_df["status"] == "OK"
    ].copy()

    if not good.empty:

        baseline = good[
            good["batch_size"] == 2
        ]

        if not baseline.empty:

            base_speed = float(
                baseline.iloc[0][
                    "samples_per_second"
                ]
            )

            result_df[
                "speedup_vs_batch2"
            ] = (
                result_df[
                    "samples_per_second"
                ]
                /
                base_speed
            )

        best = (
            good
            .sort_values(
                "samples_per_second",
                ascending=False,
            )
            .iloc[0]
        )

        best_batch = int(
            best["batch_size"]
        )

    else:

        best_batch = None

    result_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        result_df.to_string(
            index=False
        )
    )

    if best_batch is not None:

        print(
            "\nBEST BATCH SIZE:",
            best_batch
        )

    print(
        "\nSaved:"
    )

    print(
        output_csv
    )

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":

    main()