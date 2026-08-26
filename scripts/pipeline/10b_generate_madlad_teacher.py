from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# ============================================================
# Constants
# ============================================================

EXPECTED_TOTAL = 20_000
EXPECTED_PER_DIRECTION = 10_000

VALID_DIRECTIONS = {
    "en_uz",
    "uz_en",
}

CYRILLIC_RE = re.compile(
    r"[\u0400-\u04FF]"
)


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "STEP 10B - Generate MADLAD teacher "
            "targets for EN-UZ distillation pilot."
        )
    )

    parser.add_argument(
        "--madlad_path",
        type=str,
        default=None,
        help=(
            "MADLAD local snapshot path. "
            "If omitted, auto-detect from HF cache."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous checkpoint and regenerate all.",
    )

    return parser.parse_args()


# ============================================================
# JSONL helpers
# ============================================================

def append_jsonl(
    path: Path,
    rows: list[dict],
):

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                )
            )

            f.write("\n")

        # Make checkpoint safer for long runs
        f.flush()
        os.fsync(
            f.fileno()
        )


def load_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        return []

    rows = []

    with open(
        path,
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
                    json.loads(
                        line
                    )
                )

            except Exception:

                print(
                    f"[WARNING] Broken checkpoint "
                    f"line skipped: {line_no}"
                )

    return rows


# ============================================================
# Stable config hash
# ============================================================

def config_hash(
    config: dict,
) -> str:

    payload = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ============================================================
# MADLAD auto detection
# ============================================================

def find_madlad_snapshot() -> Path:

    roots = [

        Path(
            "/root/autodl-tmp/"
            "huggingface/hub/"
            "models--google--madlad400-3b-mt/"
            "snapshots"
        ),

        Path(
            "/root/.cache/"
            "huggingface/hub/"
            "models--google--madlad400-3b-mt/"
            "snapshots"
        ),
    ]

    candidates = []

    for root in roots:

        if not root.exists():
            continue

        for child in root.iterdir():

            if (
                child.is_dir()
                and
                (
                    child
                    /
                    "config.json"
                ).exists()
            ):

                candidates.append(
                    child
                )

    if not candidates:

        raise FileNotFoundError(
            "Could not auto-detect MADLAD snapshot."
        )

    # newest snapshot first
    candidates = sorted(
        candidates,
        key=lambda p:
            p.stat().st_mtime,
        reverse=True,
    )

    return candidates[0]


# ============================================================
# Cyrillic detection
# ============================================================

def contains_cyrillic(
    text: str,
) -> bool:

    return bool(
        CYRILLIC_RE.search(
            str(text or "")
        )
    )


# ============================================================
# Uzbek Cyrillic -> Latin
# ============================================================

SIMPLE_MAP = {

    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",

    "ё": "yo",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",

    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",

    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",

    "ф": "f",
    "х": "x",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",

    "ъ": "'",
    "ь": "",

    "э": "e",
    "ю": "yu",
    "я": "ya",

    "ў": "o'",
    "қ": "q",
    "ғ": "g'",
    "ҳ": "h",
}


CYRILLIC_VOWELS = set(
    "аеёиоуўэюяАЕЁИОУЎЭЮЯ"
)


APOSTROPHE_LIKE = {
    "'",
    "’",
    "‘",
    "`",
    "ʻ",
    "ʼ",
    "ʹ",
    "՚",
    "ъ",
    "Ъ",
}


def preserve_case(
    source_char: str,
    replacement: str,
) -> str:

    if not replacement:
        return replacement

    if source_char.isupper():

        if len(replacement) == 1:
            return replacement.upper()

        return (
            replacement[0].upper()
            +
            replacement[1:]
        )

    return replacement


def transliterate_uzbek_cyrillic(
    text: str,
) -> str:

    text = str(
        text or ""
    )

    result = []

    for i, char in enumerate(text):

        lower = char.lower()

        # ----------------------------------------------------
        # Е / е
        # ----------------------------------------------------

        if lower == "е":

            previous = (
                text[i - 1]
                if i > 0
                else ""
            )

            at_word_start = (
                i == 0
                or
                not previous.isalpha()
            )

            after_vowel = (
                previous
                in CYRILLIC_VOWELS
            )

            after_apostrophe = (
                previous
                in APOSTROPHE_LIKE
            )

            if (
                at_word_start
                or
                after_vowel
                or
                after_apostrophe
            ):

                replacement = "ye"

            else:

                replacement = "e"

            result.append(
                preserve_case(
                    char,
                    replacement,
                )
            )

            continue

        # ----------------------------------------------------
        # Standard mapping
        # ----------------------------------------------------

        if lower in SIMPLE_MAP:

            result.append(
                preserve_case(
                    char,
                    SIMPLE_MAP[
                        lower
                    ],
                )
            )

            continue

        result.append(
            char
        )

    return "".join(
        result
    )


# ============================================================
# Latin Uzbek normalization
# ============================================================

def normalize_uzbek_latin(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    )

    for old in [
        "’",
        "‘",
        "`",
        "ʻ",
        "ʼ",
        "ʹ",
        "՚",
        "´",
    ]:

        text = text.replace(
            old,
            "'",
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text,
    )

    text = re.sub(
        r"'{2,}",
        "'",
        text,
    )

    return text.strip()


# ============================================================
# Normalize teacher output
# ============================================================

def normalize_teacher_prediction(
    raw_prediction: str,
    direction: str,
):

    raw_prediction = str(
        raw_prediction or ""
    ).strip()

    if not raw_prediction:

        return (
            "",
            "EMPTY",
            False,
        )

    # --------------------------------------------------------
    # EN -> UZ
    # --------------------------------------------------------

    if direction == "en_uz":

        was_cyrillic = (
            contains_cyrillic(
                raw_prediction
            )
        )

        if was_cyrillic:

            normalized = (
                transliterate_uzbek_cyrillic(
                    raw_prediction
                )
            )

            normalized = (
                normalize_uzbek_latin(
                    normalized
                )
            )

            script = (
                "CYRILLIC_TO_LATIN"
            )

        else:

            normalized = (
                normalize_uzbek_latin(
                    raw_prediction
                )
            )

            script = "LATIN"

        return (
            normalized,
            script,
            was_cyrillic,
        )

    # --------------------------------------------------------
    # UZ -> EN
    # --------------------------------------------------------

    normalized = re.sub(
        r"\s+",
        " ",
        raw_prediction,
    ).strip()

    return (
        normalized,
        "ENGLISH",
        False,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # Project paths
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_file = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "10a_candidates"
        / "distillation_pilot_20k_v1.parquet"
    )

    output_dir = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "10b_teacher_generation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = (
        output_dir
        / "generation_checkpoint.jsonl"
    )

    config_file = (
        output_dir
        / "generation_config_v1.json"
    )

    output_parquet = (
        output_dir
        / "teacher_predictions_20k_v1.parquet"
    )

    output_csv = (
        output_dir
        / "teacher_predictions_20k_v1.csv"
    )

    report_file = (
        output_dir
        / "10b_report_v1.json"
    )

    # ========================================================
    # MADLAD path
    # ========================================================

    if args.madlad_path:

        madlad_path = Path(
            args.madlad_path
        )

    else:

        madlad_path = (
            find_madlad_snapshot()
        )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 110)
    print("EN-UZ STUDENT PIPELINE")
    print(
        "STEP 10B - "
        "MADLAD TEACHER GENERATION V1"
    )
    print("=" * 110)

    print(
        "\nInput:"
    )

    print(
        input_file
    )

    print(
        "\nMADLAD:"
    )

    print(
        madlad_path
    )

    print(
        "\nBatch size:",
        args.batch_size
    )

    print(
        "Beam size :",
        args.num_beams
    )

    print(
        "Max source:",
        args.max_source_length
    )

    print(
        "Max output:",
        args.max_new_tokens
    )

    # ========================================================
    # Checks
    # ========================================================

    if not input_file.exists():

        raise FileNotFoundError(
            f"Step10A pilot not found:\n"
            f"{input_file}"
        )

    if not madlad_path.exists():

        raise FileNotFoundError(
            f"MADLAD path not found:\n"
            f"{madlad_path}"
        )

    if not (
        madlad_path
        /
        "config.json"
    ).exists():

        raise FileNotFoundError(
            "MADLAD config.json not found:\n"
            f"{madlad_path}"
        )

    # ========================================================
    # CUDA
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nCUDA:",
        True
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "PyTorch CUDA:",
        torch.version.cuda
    )

    # ========================================================
    # Load candidate pilot
    # ========================================================

    print(
        "\nLoading Step10A pilot..."
    )

    df = pd.read_parquet(
        input_file
    )

    required_columns = [
        "candidate_id",
        "normalized_pair_id",
        "direction",
        "src_lang",
        "tgt_lang",
        "source_text",
        "real_reference",
        "teacher_input",
        "quality_tier",
        "data_source",
        "length_bucket",
        "candidate_status",
        "leak_validation",
        "leak_benchmark",
        "leak_challenge",
        "cyrillic_uzbek",
    ]

    missing = [
        column
        for column
        in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing candidate columns: "
            f"{missing}"
        )

    print(
        "Rows:",
        len(df)
    )

    print(
        "\nDirections:"
    )

    print(
        df[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Strict 10A integrity verification
    # ========================================================

    if len(df) != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} candidates, "
            f"got {len(df)}."
        )

    direction_counts = (
        df[
            "direction"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        direction_counts.get(
            "en_uz",
            0,
        )
        !=
        EXPECTED_PER_DIRECTION
        or
        direction_counts.get(
            "uz_en",
            0,
        )
        !=
        EXPECTED_PER_DIRECTION
    ):

        raise RuntimeError(
            "Pilot direction distribution "
            "is not 10K / 10K."
        )

    if (
        df[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate candidate_id found."
        )

    unexpected_directions = (
        set(
            df[
                "direction"
            ]
            .astype(str)
        )
        -
        VALID_DIRECTIONS
    )

    if unexpected_directions:

        raise RuntimeError(
            f"Unexpected directions: "
            f"{unexpected_directions}"
        )

    if (
        df[
            "candidate_status"
        ]
        .astype(str)
        .ne("READY")
        .any()
    ):

        raise RuntimeError(
            "Non-READY candidate detected."
        )

    for col in [
        "leak_validation",
        "leak_benchmark",
        "leak_challenge",
        "cyrillic_uzbek",
    ]:

        if (
            df[
                col
            ]
            .astype(bool)
            .any()
        ):

            raise RuntimeError(
                f"Unsafe candidate flag detected: "
                f"{col}"
            )

    if (
        df[
            "source_text"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        raise RuntimeError(
            "Empty source found."
        )

    if (
        df[
            "real_reference"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        raise RuntimeError(
            "Empty real reference found."
        )

    if (
        df[
            "teacher_input"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        raise RuntimeError(
            "Empty teacher input found."
        )

    print(
        "\nStep10A integrity: PASS"
    )

    # ========================================================
    # Run configuration
    # ========================================================

    run_config = {

        "step":
            "10B",

        "version":
            "v1",

        "model_path":
            str(
                madlad_path.resolve()
            ),

        "input_file":
            str(
                input_file.resolve()
            ),

        "total_candidates":
            len(df),

        "batch_size":
            args.batch_size,

        "num_beams":
            args.num_beams,

        "max_source_length":
            args.max_source_length,

        "max_new_tokens":
            args.max_new_tokens,

        "precision":
            "float16",

        "do_sample":
            False,
    }

    current_config_hash = (
        config_hash(
            run_config
        )
    )

    run_config[
        "config_hash"
    ] = current_config_hash

    # ========================================================
    # Overwrite / resume safety
    # ========================================================

    if args.overwrite:

        print(
            "\n[OVERWRITE] "
            "Removing previous generation files..."
        )

        for path in [
            checkpoint_file,
            config_file,
            output_parquet,
            output_csv,
            report_file,
        ]:

            if path.exists():

                path.unlink()

    elif (
        checkpoint_file.exists()
        and
        config_file.exists()
    ):

        with open(
            config_file,
            "r",
            encoding="utf-8",
        ) as f:

            previous_config = (
                json.load(
                    f
                )
            )

        previous_hash = (
            previous_config.get(
                "config_hash"
            )
        )

        if (
            previous_hash
            !=
            current_config_hash
        ):

            raise RuntimeError(
                "\nExisting checkpoint was generated "
                "with a DIFFERENT configuration.\n\n"
                "Do not mix generation settings.\n\n"
                "If you intentionally want to rerun, use:\n"
                "--overwrite"
            )

    elif (
        checkpoint_file.exists()
        and
        not config_file.exists()
    ):

        raise RuntimeError(
            "Checkpoint exists but generation "
            "config is missing."
        )

    # Save / refresh config
    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Resume existing results
    # ========================================================

    existing_rows = (
        load_jsonl(
            checkpoint_file
        )
    )

    result_map = {}

    for row in existing_rows:

        candidate_id = str(
            row.get(
                "candidate_id",
                "",
            )
        )

        if not candidate_id:
            continue

        result_map[
            candidate_id
        ] = row

    completed_ids = set(
        result_map.keys()
    )

    valid_candidate_ids = set(
        df[
            "candidate_id"
        ]
        .astype(str)
    )

    unknown_completed = (
        completed_ids
        -
        valid_candidate_ids
    )

    if unknown_completed:

        raise RuntimeError(
            "Checkpoint contains candidate IDs "
            "not present in current 10A pilot."
        )

    print(
        "\nAlready completed:",
        len(
            completed_ids
        )
    )

    print(
        "Pending:",
        len(df)
        -
        len(
            completed_ids
        )
    )

    # ========================================================
    # Load MADLAD
    # ========================================================

    print(
        "\nLoading MADLAD tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(
                madlad_path
            ),
            local_files_only=True,
        )
    )

    print(
        "Tokenizer:",
        type(
            tokenizer
        ).__name__
    )

    print(
        "\nLoading MADLAD model..."
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            str(
                madlad_path
            ),
            dtype=torch.float16,
            local_files_only=True,
        )
        .to(
            device
        )
    )

    model.eval()

    model.config.use_cache = True

    print(
        "Model:",
        type(
            model
        ).__name__
    )

    print(
        "GPU allocated:",
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )

    # ========================================================
    # Pending dataframe
    # ========================================================

    pending_df = df[
        ~df[
            "candidate_id"
        ]
        .astype(str)
        .isin(
            completed_ids
        )
    ].copy()

    # deterministic order
    pending_df = (
        pending_df
        .sort_values(
            [
                "direction",
                "candidate_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Generation
    # ========================================================

    run_start = (
        time.perf_counter()
    )

    processed_this_run = 0

    print("\n")
    print("=" * 110)
    print("MADLAD GENERATION")
    print("=" * 110)

    for start in range(
        0,
        len(
            pending_df
        ),
        args.batch_size,
    ):

        end = min(
            start
            +
            args.batch_size,
            len(
                pending_df
            ),
        )

        batch = (
            pending_df
            .iloc[
                start:end
            ]
            .copy()
        )

        teacher_inputs = (
            batch[
                "teacher_input"
            ]
            .astype(str)
            .tolist()
        )

        # ====================================================
        # Tokenize
        # ====================================================

        encoded = tokenizer(
            teacher_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=
                args.max_source_length,
        )

        encoded = {
            key:
                value.to(
                    device
                )

            for key, value
            in encoded.items()
        }

        # ====================================================
        # Generate
        # ====================================================

        torch.cuda.synchronize()

        batch_start = (
            time.perf_counter()
        )

        with torch.inference_mode():

            generated = (
                model.generate(
                    **encoded,

                    num_beams=
                        args.num_beams,

                    do_sample=False,

                    max_new_tokens=
                        args.max_new_tokens,

                    use_cache=True,
                )
            )

        torch.cuda.synchronize()

        batch_seconds = (
            time.perf_counter()
            -
            batch_start
        )

        # ====================================================
        # Decode
        # ====================================================

        raw_predictions = (
            tokenizer
            .batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

        generated_token_counts = (
            (
                generated
                !=
                tokenizer.pad_token_id
            )
            .sum(
                dim=1
            )
            .detach()
            .cpu()
            .tolist()
        )

        latency_per_sample = (
            batch_seconds
            /
            len(batch)
        )

        batch_rows = []

        for (
            (_, row),
            raw_prediction,
            generated_tokens,
        ) in zip(
            batch.iterrows(),
            raw_predictions,
            generated_token_counts,
        ):

            direction = str(
                row[
                    "direction"
                ]
            )

            (
                normalized_prediction,
                output_script,
                was_cyrillic,
            ) = (
                normalize_teacher_prediction(
                    raw_prediction=
                        raw_prediction,

                    direction=
                        direction,
                )
            )

            generation_status = (
                "OK"
                if
                normalized_prediction.strip()
                else
                "EMPTY_OUTPUT"
            )

            result = {

                # --------------------------------------------
                # Candidate identity
                # --------------------------------------------

                "candidate_id":
                    str(
                        row[
                            "candidate_id"
                        ]
                    ),

                "source_sample_id":
                    str(
                        row.get(
                            "source_sample_id",
                            "",
                        )
                    ),

                "normalized_pair_id":
                    str(
                        row[
                            "normalized_pair_id"
                        ]
                    ),

                "split_group_id":
                    str(
                        row.get(
                            "split_group_id",
                            "",
                        )
                    ),

                "pair_fingerprint":
                    str(
                        row.get(
                            "pair_fingerprint",
                            "",
                        )
                    ),

                # --------------------------------------------
                # Direction
                # --------------------------------------------

                "direction":
                    direction,

                "src_lang":
                    str(
                        row[
                            "src_lang"
                        ]
                    ),

                "tgt_lang":
                    str(
                        row[
                            "tgt_lang"
                        ]
                    ),

                # --------------------------------------------
                # Original data
                # --------------------------------------------

                "source_text":
                    str(
                        row[
                            "source_text"
                        ]
                    ),

                "real_reference":
                    str(
                        row[
                            "real_reference"
                        ]
                    ),

                "teacher_input":
                    str(
                        row[
                            "teacher_input"
                        ]
                    ),

                # --------------------------------------------
                # Teacher output
                # --------------------------------------------

                "teacher_prediction_raw":
                    str(
                        raw_prediction
                    ),

                "teacher_prediction":
                    str(
                        normalized_prediction
                    ),

                "teacher_output_script":
                    output_script,

                "teacher_raw_has_cyrillic":
                    bool(
                        was_cyrillic
                    ),

                "generation_status":
                    generation_status,

                # --------------------------------------------
                # Generation metadata
                # --------------------------------------------

                "generated_token_count":
                    int(
                        generated_tokens
                    ),

                "generation_seconds":
                    float(
                        latency_per_sample
                    ),

                "batch_generation_seconds":
                    float(
                        batch_seconds
                    ),

                "num_beams":
                    int(
                        args.num_beams
                    ),

                # --------------------------------------------
                # Candidate metadata
                # --------------------------------------------

                "quality_tier":
                    str(
                        row[
                            "quality_tier"
                        ]
                    ),

                "training_weight":
                    float(
                        row.get(
                            "training_weight",
                            1.0,
                        )
                    ),

                "data_source":
                    str(
                        row[
                            "data_source"
                        ]
                    ),

                "length_bucket":
                    str(
                        row[
                            "length_bucket"
                        ]
                    ),

                "source_word_count":
                    int(
                        row.get(
                            "source_word_count",
                            0,
                        )
                    ),

                "target_word_count":
                    int(
                        row.get(
                            "target_word_count",
                            0,
                        )
                    ),
            }

            batch_rows.append(
                result
            )

            result_map[
                result[
                    "candidate_id"
                ]
            ] = result

        # ====================================================
        # Save checkpoint IMMEDIATELY
        # ====================================================

        append_jsonl(
            checkpoint_file,
            batch_rows,
        )

        processed_this_run += (
            len(
                batch_rows
            )
        )

        # ====================================================
        # Progress
        # ====================================================

        completed_total = (
            len(
                result_map
            )
        )

        elapsed = (
            time.perf_counter()
            -
            run_start
        )

        throughput = (
            processed_this_run
            /
            elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            EXPECTED_TOTAL
            -
            completed_total
        )

        eta_seconds = (
            remaining
            /
            throughput
            if throughput > 0
            else 0
        )

        # print every ~100 samples
        if (
            processed_this_run
            %
            100
            <
            args.batch_size
            or
            completed_total
            ==
            EXPECTED_TOTAL
        ):

            gpu_memory = (
                torch.cuda.memory_allocated()
                /
                1024**3
            )

            print(
                f"Completed: "
                f"{completed_total}/"
                f"{EXPECTED_TOTAL} "
                f"| Batch: {batch_seconds:.2f}s "
                f"| Speed: {throughput:.2f} samples/s "
                f"| ETA: {eta_seconds/60:.1f} min "
                f"| GPU: {gpu_memory:.2f}GB"
            )

    # ========================================================
    # Release model before final aggregation
    # ========================================================

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()

    # ========================================================
    # Reload checkpoint as source of truth
    # ========================================================

    print(
        "\nReloading checkpoint..."
    )

    checkpoint_rows = (
        load_jsonl(
            checkpoint_file
        )
    )

    final_map = {}

    for row in checkpoint_rows:

        candidate_id = str(
            row.get(
                "candidate_id",
                "",
            )
        )

        if not candidate_id:
            continue

        final_map[
            candidate_id
        ] = row

    result_df = pd.DataFrame(
        list(
            final_map.values()
        )
    )

    # ========================================================
    # Completeness
    # ========================================================

    if len(
        result_df
    ) != EXPECTED_TOTAL:

        raise RuntimeError(
            "\nGeneration incomplete.\n"
            f"Expected: {EXPECTED_TOTAL}\n"
            f"Found: {len(result_df)}\n\n"
            "Run the script again to resume."
        )

    result_ids = set(
        result_df[
            "candidate_id"
        ]
        .astype(str)
    )

    missing_ids = (
        valid_candidate_ids
        -
        result_ids
    )

    extra_ids = (
        result_ids
        -
        valid_candidate_ids
    )

    if missing_ids:

        raise RuntimeError(
            f"Missing candidate IDs: "
            f"{len(missing_ids)}"
        )

    if extra_ids:

        raise RuntimeError(
            f"Unexpected candidate IDs: "
            f"{len(extra_ids)}"
        )

    # Restore original 10A ordering
    order_map = {

        str(
            candidate_id
        ):
            index

        for index, candidate_id
        in enumerate(
            df[
                "candidate_id"
            ]
            .astype(str)
        )
    }

    result_df[
        "_order"
    ] = (
        result_df[
            "candidate_id"
        ]
        .astype(str)
        .map(
            order_map
        )
    )

    result_df = (
        result_df
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    empty_count = int(
        result_df[
            "generation_status"
        ]
        .eq(
            "EMPTY_OUTPUT"
        )
        .sum()
    )

    cyrillic_count = int(
        result_df[
            "teacher_raw_has_cyrillic"
        ]
        .astype(bool)
        .sum()
    )

    en_uz_df = (
        result_df[
            result_df[
                "direction"
            ]
            ==
            "en_uz"
        ]
    )

    uz_en_df = (
        result_df[
            result_df[
                "direction"
            ]
            ==
            "uz_en"
        ]
    )

    en_uz_cyrillic_count = int(
        en_uz_df[
            "teacher_raw_has_cyrillic"
        ]
        .astype(bool)
        .sum()
    )

    en_uz_cyrillic_rate = (
        en_uz_cyrillic_count
        /
        max(
            len(
                en_uz_df
            ),
            1,
        )
        *
        100
    )

    avg_latency = float(
        result_df[
            "generation_seconds"
        ]
        .mean()
    )

    avg_tokens = float(
        result_df[
            "generated_token_count"
        ]
        .mean()
    )

    # ========================================================
    # Save final
    # ========================================================

    print(
        "\nSaving final Teacher dataset..."
    )

    result_df.to_parquet(
        output_parquet,
        index=False,
    )

    result_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Report
    # ========================================================

    report = {

        "step":
            "10B",

        "version":
            "v1",

        "status":
            (
                "READY_FOR_QWEN"
                if
                empty_count == 0
                else
                "READY_WITH_EMPTY_OUTPUTS"
            ),

        "input": {

            "file":
                str(
                    input_file
                ),

            "samples":
                len(df),

            "directions": {

                "en_uz":
                    len(
                        en_uz_df
                    ),

                "uz_en":
                    len(
                        uz_en_df
                    ),
            },
        },

        "teacher": {

            "model":
                "google/madlad400-3b-mt",

            "model_path":
                str(
                    madlad_path
                ),

            "precision":
                "float16",

            "num_beams":
                args.num_beams,

            "batch_size":
                args.batch_size,

            "max_source_length":
                args.max_source_length,

            "max_new_tokens":
                args.max_new_tokens,

            "do_sample":
                False,
        },

        "generation": {

            "total":
                len(
                    result_df
                ),

            "empty_outputs":
                empty_count,

            "avg_generation_seconds_per_sample":
                avg_latency,

            "avg_generated_tokens":
                avg_tokens,
        },

        "script": {

            "raw_cyrillic_total":
                cyrillic_count,

            "en_uz_raw_cyrillic":
                en_uz_cyrillic_count,

            "en_uz_raw_cyrillic_rate_percent":
                en_uz_cyrillic_rate,

            "cyrillic_to_latin_normalization":
                True,
        },

        "config_hash":
            current_config_hash,
    }

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Preview
    # ========================================================

    print("\n")
    print("=" * 110)
    print("TEACHER OUTPUT PREVIEW")
    print("=" * 110)

    preview = (
        result_df
        .groupby(
            "direction",
            group_keys=False,
        )
        .head(
            5
        )
    )

    for row in preview.itertuples(
        index=False
    ):

        print()

        print(
            "ID       :",
            row.candidate_id
        )

        print(
            "Direction:",
            row.direction
        )

        print(
            "Source   :",
            row.source_text
        )

        print(
            "Real ref :",
            row.real_reference
        )

        print(
            "Teacher raw:"
        )

        print(
            row.teacher_prediction_raw
        )

        print(
            "Teacher normalized:"
        )

        print(
            row.teacher_prediction
        )

        print(
            "Script   :",
            row.teacher_output_script
        )

    # ========================================================
    # Complete
    # ========================================================

    print("\n")
    print("=" * 110)
    print("STEP 10B COMPLETE")
    print("=" * 110)

    print(
        "\nTotal:",
        len(
            result_df
        )
    )

    print(
        "EN -> UZ:",
        len(
            en_uz_df
        )
    )

    print(
        "UZ -> EN:",
        len(
            uz_en_df
        )
    )

    print(
        "\nEmpty outputs:",
        empty_count
    )

    print(
        "\nEN->UZ raw Cyrillic:",
        en_uz_cyrillic_count
    )

    print(
        "EN->UZ Cyrillic rate:",
        f"{en_uz_cyrillic_rate:.2f}%"
    )

    print(
        "\nAvg generation latency:",
        f"{avg_latency:.4f}s/sample"
    )

    print(
        "Avg generated tokens:",
        f"{avg_tokens:.2f}"
    )

    print(
        "\nOutput:"
    )

    print(
        output_parquet
    )

    print(
        "\nCheckpoint:"
    )

    print(
        checkpoint_file
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )

    if empty_count == 0:

        print(
            "\nSTATUS: READY_FOR_QWEN"
        )

    else:

        print(
            "\nSTATUS: READY_WITH_EMPTY_OUTPUTS"
        )

        print(
            "Empty Teacher outputs should be "
            "handled in Step10C."
        )


if __name__ == "__main__":

    main()