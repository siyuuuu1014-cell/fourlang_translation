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
            "STEP 10B - High-throughput MADLAD "
            "teacher generation for EN-UZ pilot."
        )
    )

    parser.add_argument(
        "--madlad_path",
        type=str,
        default=None,
    )

    # --------------------------------------------------------
    # Runtime parameters
    #
    # These CAN change when resuming.
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=128,
        help=(
            "Flush checkpoint after approximately "
            "this many newly generated samples."
        ),
    )

    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
    )

    # --------------------------------------------------------
    # Generation identity
    #
    # These MUST NOT change when resuming.
    # --------------------------------------------------------

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
    )

    return parser.parse_args()


# ============================================================
# JSONL
# ============================================================

def append_jsonl(
    path: Path,
    rows: list[dict],
    fsync: bool = True,
):

    if not rows:
        return

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

        f.flush()

        if fsync:
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
                    "[WARNING] Broken checkpoint "
                    f"line skipped: {line_no}"
                )

    return rows


# ============================================================
# Hash
# ============================================================

def stable_hash(
    payload: dict,
) -> str:

    text = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# MADLAD snapshot
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
            "MADLAD snapshot not found."
        )

    candidates.sort(
        key=lambda path:
            path.stat().st_mtime,
        reverse=True,
    )

    return candidates[0]


# ============================================================
# Uzbek script
# ============================================================

def contains_cyrillic(
    text: str,
) -> bool:

    return bool(
        CYRILLIC_RE.search(
            str(text or "")
        )
    )


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

    for index, char in enumerate(text):

        lower = char.lower()

        # ----------------------------------------------------
        # Е / е
        # ----------------------------------------------------

        if lower == "е":

            previous = (
                text[index - 1]
                if index > 0
                else ""
            )

            at_word_start = (
                index == 0
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

            replacement = (
                "ye"
                if (
                    at_word_start
                    or
                    after_vowel
                    or
                    after_apostrophe
                )
                else
                "e"
            )

            result.append(
                preserve_case(
                    char,
                    replacement,
                )
            )

            continue

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

    if direction == "en_uz":

        was_cyrillic = (
            contains_cyrillic(
                raw_prediction
            )
        )

        if was_cyrillic:

            prediction = (
                transliterate_uzbek_cyrillic(
                    raw_prediction
                )
            )

            prediction = (
                normalize_uzbek_latin(
                    prediction
                )
            )

            script = (
                "CYRILLIC_TO_LATIN"
            )

        else:

            prediction = (
                normalize_uzbek_latin(
                    raw_prediction
                )
            )

            script = "LATIN"

        return (
            prediction,
            script,
            was_cyrillic,
        )

    prediction = re.sub(
        r"\s+",
        " ",
        raw_prediction,
    ).strip()

    return (
        prediction,
        "ENGLISH",
        False,
    )


# ============================================================
# Generation identity compatibility
# ============================================================

def build_generation_identity(
    madlad_path: Path,
    input_file: Path,
    args,
):

    # IMPORTANT:
    #
    # batch_size / checkpoint_every are intentionally excluded.
    #
    # Changing these does NOT change model outputs.
    #
    return {

        "model_path":
            str(
                madlad_path.resolve()
            ),

        "input_file":
            str(
                input_file.resolve()
            ),

        "total_candidates":
            EXPECTED_TOTAL,

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


def verify_old_or_new_config(
    existing_config: dict,
    current_identity: dict,
):

    """
    Supports BOTH:

    Previous 10B config:
        batch_size was included in config_hash

    New optimized config:
        runtime parameters are separated.

    We compare only generation-relevant fields.
    """

    keys = [

        "model_path",
        "input_file",
        "total_candidates",
        "num_beams",
        "max_source_length",
        "max_new_tokens",
        "precision",
        "do_sample",
    ]

    mismatches = []

    # New schema
    if (
        "generation_identity"
        in existing_config
    ):

        old_identity = (
            existing_config[
                "generation_identity"
            ]
        )

    # Old schema
    else:

        old_identity = (
            existing_config
        )

    for key in keys:

        if (
            old_identity.get(
                key
            )
            !=
            current_identity.get(
                key
            )
        ):

            mismatches.append(
                (
                    key,
                    old_identity.get(
                        key
                    ),
                    current_identity.get(
                        key
                    ),
                )
            )

    if mismatches:

        print("\nGeneration identity mismatch:")

        for key, old, new in mismatches:

            print(
                f"{key}:"
            )

            print(
                f"  existing = {old}"
            )

            print(
                f"  current  = {new}"
            )

        raise RuntimeError(
            "\nGeneration settings changed.\n"
            "Existing checkpoint cannot be mixed.\n"
            "Use --overwrite only if you intentionally "
            "want to regenerate everything."
        )


# ============================================================
# Checkpoint buffer
# ============================================================

class CheckpointBuffer:

    def __init__(
        self,
        path: Path,
        flush_every: int,
    ):

        self.path = path

        self.flush_every = max(
            1,
            flush_every,
        )

        self.rows = []

        self.total_flushed = 0

    def add(
        self,
        rows: list[dict],
    ):

        self.rows.extend(
            rows
        )

        if (
            len(
                self.rows
            )
            >=
            self.flush_every
        ):

            self.flush()

    def flush(self):

        if not self.rows:
            return

        append_jsonl(
            self.path,
            self.rows,
            fsync=True,
        )

        self.total_flushed += (
            len(
                self.rows
            )
        )

        self.rows.clear()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # Paths
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
    # MADLAD
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
        "HIGH-THROUGHPUT MADLAD GENERATION"
    )
    print("=" * 110)

    print(
        "\nMADLAD:"
    )

    print(
        madlad_path
    )

    print(
        "\nRuntime:"
    )

    print(
        "Batch size       :",
        args.batch_size
    )

    print(
        "Checkpoint every :",
        args.checkpoint_every
    )

    print(
        "Progress every   :",
        args.progress_every
    )

    print(
        "\nGeneration:"
    )

    print(
        "Beam             :",
        args.num_beams
    )

    print(
        "Max source       :",
        args.max_source_length
    )

    print(
        "Max new tokens   :",
        args.max_new_tokens
    )

    # ========================================================
    # Files
    # ========================================================

    if not input_file.exists():

        raise FileNotFoundError(
            input_file
        )

    if not madlad_path.exists():

        raise FileNotFoundError(
            madlad_path
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
        "\nGPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    # ========================================================
    # Load pilot
    # ========================================================

    print(
        "\nLoading Step10A pilot..."
    )

    df = pd.read_parquet(
        input_file
    )

    print(
        "Rows:",
        len(df)
    )

    if len(df) != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL}, "
            f"got {len(df)}."
        )

    counts = (
        df[
            "direction"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        counts.get(
            "en_uz",
            0,
        )
        !=
        EXPECTED_PER_DIRECTION
        or
        counts.get(
            "uz_en",
            0,
        )
        !=
        EXPECTED_PER_DIRECTION
    ):

        raise RuntimeError(
            "Direction distribution invalid."
        )

    if (
        df[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate candidate_id."
        )

    # ========================================================
    # Generation identity
    # ========================================================

    generation_identity = (
        build_generation_identity(
            madlad_path,
            input_file,
            args,
        )
    )

    identity_hash = stable_hash(
        generation_identity
    )

    # ========================================================
    # Overwrite
    # ========================================================

    if args.overwrite:

        print(
            "\n[OVERWRITE] "
            "Removing old Step10B files."
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

    # ========================================================
    # Existing config compatibility
    # ========================================================

    if (
        config_file.exists()
        and
        not args.overwrite
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

        verify_old_or_new_config(
            previous_config,
            generation_identity,
        )

        print(
            "\nExisting generation "
            "configuration: COMPATIBLE"
        )

    elif (
        checkpoint_file.exists()
        and
        not config_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "Checkpoint exists but generation "
            "config does not exist."
        )

    # ========================================================
    # Save NEW config schema
    # ========================================================

    current_config = {

        "step":
            "10B",

        "version":
            "v1_high_throughput",

        "generation_identity":
            generation_identity,

        "generation_identity_hash":
            identity_hash,

        "runtime": {

            "batch_size":
                args.batch_size,

            "checkpoint_every":
                args.checkpoint_every,

            "progress_every":
                args.progress_every,
        },
    }

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            current_config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Resume
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
                ""
            )
        )

        if candidate_id:

            result_map[
                candidate_id
            ] = row

    completed_ids = set(
        result_map.keys()
    )

    candidate_ids = set(
        df[
            "candidate_id"
        ]
        .astype(str)
    )

    unknown_ids = (
        completed_ids
        -
        candidate_ids
    )

    if unknown_ids:

        raise RuntimeError(
            "Checkpoint contains unknown "
            f"candidate IDs: {len(unknown_ids)}"
        )

    print(
        "\nAlready completed:",
        len(
            completed_ids
        )
    )

    print(
        "Pending:",
        EXPECTED_TOTAL
        -
        len(
            completed_ids
        )
    )

    # ========================================================
    # IMPORTANT OPTIMIZATION:
    #
    # Sort by:
    #
    # direction
    # source length
    # candidate ID
    #
    # Similar lengths enter same batch -> less padding.
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

    if (
        "source_word_count"
        not in pending_df.columns
    ):

        pending_df[
            "source_word_count"
        ] = (
            pending_df[
                "source_text"
            ]
            .astype(str)
            .str.split()
            .str.len()
        )

    pending_df = (
        pending_df
        .sort_values(
            [
                "direction",
                "source_word_count",
                "candidate_id",
            ],
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Load model
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
        "Loading MADLAD model..."
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
        "Initial GPU memory:",
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )

    # ========================================================
    # Buffer
    # ========================================================

    checkpoint_buffer = (
        CheckpointBuffer(
            checkpoint_file,
            args.checkpoint_every,
        )
    )

    # ========================================================
    # Generation
    # ========================================================

    print("\n")
    print("=" * 110)
    print("HIGH-THROUGHPUT GENERATION")
    print("=" * 110)

    start_time = (
        time.perf_counter()
    )

    processed_this_run = 0

    last_progress = 0

    try:

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
            )

            teacher_inputs = (
                batch[
                    "teacher_input"
                ]
                .astype(str)
                .tolist()
            )

            # =================================================
            # Tokenize
            # =================================================

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
                        device,
                        non_blocking=True,
                    )

                for key, value
                in encoded.items()
            }

            # =================================================
            # Generate
            # =================================================

            torch.cuda.synchronize()

            batch_start = (
                time.perf_counter()
            )

            with torch.inference_mode():

                generated = model.generate(

                    **encoded,

                    num_beams=
                        args.num_beams,

                    do_sample=False,

                    max_new_tokens=
                        args.max_new_tokens,

                    use_cache=True,
                )

            torch.cuda.synchronize()

            batch_seconds = (
                time.perf_counter()
                -
                batch_start
            )

            # =================================================
            # Decode
            # =================================================

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

            sample_latency = (
                batch_seconds
                /
                len(batch)
            )

            batch_rows = []

            for (
                (_, row),
                raw_prediction,
                token_count,
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
                    teacher_prediction,
                    output_script,
                    was_cyrillic,
                ) = normalize_teacher_prediction(
                    raw_prediction,
                    direction,
                )

                status = (
                    "OK"
                    if
                    teacher_prediction.strip()
                    else
                    "EMPTY_OUTPUT"
                )

                result = {

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

                    "teacher_prediction_raw":
                        str(
                            raw_prediction
                        ),

                    "teacher_prediction":
                        teacher_prediction,

                    "teacher_output_script":
                        output_script,

                    "teacher_raw_has_cyrillic":
                        bool(
                            was_cyrillic
                        ),

                    "generation_status":
                        status,

                    "generated_token_count":
                        int(
                            token_count
                        ),

                    "generation_seconds":
                        float(
                            sample_latency
                        ),

                    "batch_generation_seconds":
                        float(
                            batch_seconds
                        ),

                    "num_beams":
                        int(
                            args.num_beams
                        ),

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

            # =================================================
            # Buffered checkpoint
            # =================================================

            checkpoint_buffer.add(
                batch_rows
            )

            processed_this_run += (
                len(
                    batch_rows
                )
            )

            # =================================================
            # Progress
            # =================================================

            completed = len(
                result_map
            )

            if (
                completed
                -
                last_progress
                >=
                args.progress_every
                or
                completed
                ==
                EXPECTED_TOTAL
            ):

                elapsed = (
                    time.perf_counter()
                    -
                    start_time
                )

                speed = (
                    processed_this_run
                    /
                    elapsed
                    if elapsed > 0
                    else 0.0
                )

                remaining = (
                    EXPECTED_TOTAL
                    -
                    completed
                )

                eta = (
                    remaining
                    /
                    speed
                    if speed > 0
                    else 0.0
                )

                memory_allocated = (
                    torch.cuda
                    .memory_allocated()
                    /
                    1024**3
                )

                memory_reserved = (
                    torch.cuda
                    .memory_reserved()
                    /
                    1024**3
                )

                print(
                    f"Completed "
                    f"{completed}/{EXPECTED_TOTAL} "
                    f"| Batch={len(batch)} "
                    f"| BatchTime={batch_seconds:.2f}s "
                    f"| Speed={speed:.2f}/s "
                    f"| ETA={eta/60:.1f}min "
                    f"| Alloc={memory_allocated:.2f}GB "
                    f"| Reserved={memory_reserved:.2f}GB"
                )

                last_progress = (
                    completed
                )

    # ========================================================
    # Ctrl+C protection
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\n[INTERRUPTED] "
            "Flushing checkpoint buffer..."
        )

        checkpoint_buffer.flush()

        print(
            "Checkpoint saved."
        )

        raise

    # ========================================================
    # Other errors
    # ========================================================

    except Exception:

        print(
            "\n[ERROR] "
            "Flushing checkpoint buffer..."
        )

        checkpoint_buffer.flush()

        raise

    finally:

        # Always save remaining rows
        checkpoint_buffer.flush()

    # ========================================================
    # Release model
    # ========================================================

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()

    # ========================================================
    # Reload checkpoint
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

        if candidate_id:

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
            f"Found   : {len(result_df)}\n\n"
            "Run the SAME script again. "
            "It will resume automatically."
        )

    result_ids = set(
        result_df[
            "candidate_id"
        ]
        .astype(str)
    )

    if (
        result_ids
        !=
        candidate_ids
    ):

        raise RuntimeError(
            "Candidate integrity mismatch."
        )

    # ========================================================
    # Restore Step10A order
    # ========================================================

    order_map = {

        str(candidate_id):
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
    # Statistics
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

    en_uz = result_df[
        result_df[
            "direction"
        ]
        ==
        "en_uz"
    ]

    uz_en = result_df[
        result_df[
            "direction"
        ]
        ==
        "uz_en"
    ]

    cyrillic_count = int(
        en_uz[
            "teacher_raw_has_cyrillic"
        ]
        .astype(bool)
        .sum()
    )

    cyrillic_rate = (
        cyrillic_count
        /
        len(en_uz)
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
    # Save
    # ========================================================

    print(
        "\nSaving final teacher dataset..."
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

    report = {

        "step":
            "10B",

        "version":
            "v1_high_throughput",

        "status":
            (
                "READY_FOR_QWEN"
                if
                empty_count == 0
                else
                "READY_WITH_EMPTY_OUTPUTS"
            ),

        "samples":
            len(
                result_df
            ),

        "direction": {

            "en_uz":
                len(en_uz),

            "uz_en":
                len(uz_en),
        },

        "generation_identity":
            generation_identity,

        "runtime_last_run": {

            "batch_size":
                args.batch_size,

            "checkpoint_every":
                args.checkpoint_every,
        },

        "generation": {

            "empty_outputs":
                empty_count,

            "avg_seconds_per_sample":
                avg_latency,

            "avg_generated_tokens":
                avg_tokens,
        },

        "uzbek_script": {

            "en_uz_raw_cyrillic":
                cyrillic_count,

            "en_uz_raw_cyrillic_rate":
                cyrillic_rate,

            "normalized_to_latin":
                True,
        },
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
            en_uz
        )
    )

    print(
        "UZ -> EN:",
        len(
            uz_en
        )
    )

    print(
        "\nEmpty outputs:",
        empty_count
    )

    print(
        "EN->UZ Cyrillic raw:",
        cyrillic_count
    )

    print(
        "Cyrillic rate:",
        f"{cyrillic_rate:.2f}%"
    )

    print(
        "\nAverage latency:",
        f"{avg_latency:.4f}s/sample"
    )

    print(
        "Average generated tokens:",
        f"{avg_tokens:.2f}"
    )

    print(
        "\nOutput:"
    )

    print(
        output_parquet
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


if __name__ == "__main__":
    main()