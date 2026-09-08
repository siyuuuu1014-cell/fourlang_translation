from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from filelock import FileLock
from opencc import OpenCC

from .common import (
    DIRECTIONS,
    LANGUAGES,
    atomic_json,
    digest,
    file_hash,
    read_json,
    source_path,
    suite_root,
    verify_files,
    versions,
    writable,
)

CYRILLIC = dict(
    zip(
        "абвгдёжзийклмнопрстуфхцчшщъыьэюяўқғҳ",
        (
            "a",
            "b",
            "v",
            "g",
            "d",
            "yo",
            "j",
            "z",
            "i",
            "y",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "r",
            "s",
            "t",
            "u",
            "f",
            "x",
            "ts",
            "ch",
            "sh",
            "shch",
            "'",
            "i",
            "",
            "e",
            "yu",
            "ya",
            "o'",
            "q",
            "g'",
            "h",
        ),
        strict=True,
    )
)
APOSTROPHES = "’‘`ʻʼʹ՚´"
KEYS = ["src_lang", "tgt_lang", "src_text", "tgt_text"]


@lru_cache(maxsize=1)
def chinese_converter():
    return OpenCC("t2s")


@lru_cache(maxsize=100000)
def normalize_text(lang: str, text: str) -> str:
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip()
    if lang == "zh":
        for _ in range(8):
            converted = chinese_converter().convert(text)
            if converted == text:
                return text
            text = converted
        raise ValueError("Chinese script normalization did not converge.")
    if lang == "uz":
        result = []
        for i, ch in enumerate(text):
            low = ch.lower()
            if low == "е":
                previous = text[i - 1].lower() if i else ""
                replacement = (
                    "ye"
                    if not previous.isalpha()
                    or previous in "аеёиоуўэюяъь'" + APOSTROPHES
                    else "e"
                )
            else:
                replacement = CYRILLIC.get(low, ch)
            if ch.isupper() and low in CYRILLIC.keys() | {"е"}:
                replacement = replacement[:1].upper() + replacement[1:]
            result.append(replacement)
        text = "".join(result)
        for apostrophe in APOSTROPHES:
            text = text.replace(apostrophe, "'")
        text = re.sub(r"'{2,}", "'", re.sub(r"\s+([,.!?;:])", r"\1", text))
        if re.search(r"[\u0400-\u052f]", text):
            raise ValueError("Unsupported Cyrillic character in Uzbek text.")
    return text


def read_table(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported data file: {path}")


def normalize_rows(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Adapt existing schemas locally; KD provenance must be explicit."""
    frame = frame.copy()
    aliases = {
        "src_lang": ["source_lang"],
        "tgt_lang": ["target_lang"],
        "src_text": ["source_text"],
        "tgt_text": ["target_text"],
        "training_source": ["training_origin", "sample_origin"],
    }
    for canonical, names in aliases.items():
        if canonical not in frame:
            present = [name for name in names if name in frame]
            if present:
                frame[canonical] = frame[present[0]]
    if "direction" in frame and not {"src_lang", "tgt_lang"}.issubset(frame.columns):
        parts = (
            frame.direction.astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", "-")
            .str.extract(r"^([a-z]{2})-([a-z]{2})$")
        )
        if parts.isna().any().any():
            raise ValueError("Malformed legacy direction column.")
        frame["src_lang"], frame["tgt_lang"] = parts[0], parts[1]
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing directed translation columns: {sorted(missing)}")
    if "training_source" not in frame:
        if kind == "kd":
            raise ValueError(
                "KD input must identify Teacher rows explicitly; refusing to guess."
            )
        frame["training_source"] = "human_parallel"
    origin = frame.training_source.fillna("").astype(str).str.lower()
    if kind == "kd":
        frame = frame[origin.str.contains("teacher")].copy()
    elif (origin.str.contains("teacher|pseudo|synthetic", regex=True)).any():
        raise ValueError(
            "Teacher/pseudo/synthetic row found in HUMAN data or validation."
        )
    for side in ("src", "tgt"):
        frame[f"{side}_lang"] = (
            frame[f"{side}_lang"].astype(str).str.lower().str.strip()
        )
        frame[f"{side}_text"] = [
            normalize_text(lang, text)
            for lang, text in zip(
                frame[f"{side}_lang"], frame[f"{side}_text"].fillna(""), strict=True
            )
        ]
    frame = frame[(frame.src_text != "") & (frame.tgt_text != "")].copy()
    frame["direction"] = frame.src_lang + "-" + frame.tgt_lang
    if not set(frame.direction).issubset(DIRECTIONS):
        raise ValueError("Input contains unsupported languages/directions.")
    frame["kind"] = "teacher" if kind == "kd" else "human"
    frame = (
        frame[KEYS + ["direction", "kind"]].drop_duplicates(KEYS).reset_index(drop=True)
    )
    frame["row_id"] = [
        digest(list(row)) for row in frame[KEYS].itertuples(index=False, name=None)
    ]
    return frame


def text_keys(frame):
    return {
        (lang, text.casefold())
        for row in frame.itertuples(index=False)
        for lang, text in ((row.src_lang, row.src_text), (row.tgt_lang, row.tgt_text))
    }


def overlap(frame, protected):
    return pd.Series(
        [
            (row.src_lang, row.src_text.casefold()) in protected
            or (row.tgt_lang, row.tgt_text.casefold()) in protected
            for row in frame.itertuples(index=False)
        ],
        index=frame.index,
        dtype=bool,
    )


class Vocabulary:
    def __init__(self, path, max_length):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.max_length = max_length
        self.pad, self.unk, self.bos, self.eos = (
            self.sp.pad_id(),
            self.sp.unk_id(),
            self.sp.bos_id(),
            self.sp.eos_id(),
        )
        self.languages = {
            lang: self.sp.piece_to_id(f"<lang:{lang}>") for lang in LANGUAGES
        }
        if (
            min(self.languages.values()) <= self.unk
            or len(set(self.languages.values())) != 4
        ):
            raise ValueError("Language symbols are missing from the shared vocabulary.")

    def encode(self, row):
        source = self.sp.encode(row["src_text"], out_type=int)
        target = self.sp.encode(row["tgt_text"], out_type=int)
        return {
            **row,
            "src_ids": [
                self.languages[row["src_lang"]],
                self.languages[row["tgt_lang"]],
            ]
            + source[: self.max_length - 3]
            + [self.eos],
            "tgt_ids": [self.bos] + target[: self.max_length - 2] + [self.eos],
            "source_truncated": len(source) > self.max_length - 3,
            "target_truncated": len(target) > self.max_length - 2,
        }

    def decode(self, ids):
        result = []
        for token in ids:
            if int(token) == self.eos:
                break
            if int(token) not in (self.pad, self.bos, *self.languages.values()):
                result.append(int(token))
        return self.sp.decode(result)


def collate(rows, pad=0, device="cpu"):
    src = torch.full(
        (len(rows), max(len(row["src_ids"]) for row in rows)), pad, dtype=torch.long
    )
    tgt = torch.full(
        (len(rows), max(len(row["tgt_ids"]) for row in rows)), pad, dtype=torch.long
    )
    for index, row in enumerate(rows):
        src[index, : len(row["src_ids"])] = torch.tensor(
            row["src_ids"], dtype=torch.long
        )
        tgt[index, : len(row["tgt_ids"])] = torch.tensor(
            row["tgt_ids"], dtype=torch.long
        )
    return src.to(device), tgt[:, :-1].to(device), tgt[:, 1:].to(device)


def prepare_signature(config, inputs):
    return {
        "schema": 1,
        "inputs": inputs,
        "tokenizer": config["tokenizer"],
        "length": config["model"]["max_length"],
        "pair_data": config["pair_data"],
        "benchmarks": config["benchmarks"],
        "validation_samples": config["evaluation"]["validation_per_direction"],
        "seed": config["experiment"]["seed"],
        "code": file_hash(Path(__file__)),
        "sentencepiece_version": spm.__version__,
        "preparation_versions": {
            key: value
            for key, value in versions().items()
            if key
            in (
                "numpy",
                "pandas",
                "pyarrow",
                "sentencepiece",
                "opencc-python-reimplemented",
            )
        },
    }


def prepare(config):
    root = suite_root(config) / "prepared"
    writable(root).mkdir(parents=True, exist_ok=True)
    paths = sorted(
        {
            item[key]
            for item in config["pair_data"]
            for key in ("human", "kd", "validation")
        }
        | set(config["benchmarks"].values())
    )
    print("Checking shared source files (read-only)...", flush=True)
    inputs = {name: file_hash(source_path(name)) for name in paths}
    signature = prepare_signature(config, inputs)
    with FileLock(str(writable(root / ".prepare.lock")), timeout=0):
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["signature"] != signature:
                raise RuntimeError(
                    "Prepared data/config changed. Choose --suite with a new name."
                )
            verify_files(root, manifest["files"])
            print("Prepared data already complete and verified.", flush=True)
            return manifest
        human_parts, teacher_parts, validations = [], [], []
        for number, item in enumerate(config["pair_data"], 1):
            print(f"[{number}/6] Reading and normalizing {item['pair']}...", flush=True)
            for key, kind, destination in (
                ("human", "human", human_parts),
                ("kd", "kd", teacher_parts),
                ("validation", "human", validations),
            ):
                part = normalize_rows(read_table(source_path(item[key])), kind)
                expected = set(item["pair"].split("_"))
                if any(
                    {a, b} != expected for a, b in zip(part.src_lang, part.tgt_lang)
                ):
                    raise ValueError(f"Wrong pair in {item[key]}")
                destination.append(part)
                print(f"  {key}: {len(part):,} unique directed rows", flush=True)
        human = pd.concat(human_parts, ignore_index=True).drop_duplicates(KEYS)
        teacher = pd.concat(teacher_parts, ignore_index=True).drop_duplicates(KEYS)
        validation = pd.concat(validations, ignore_index=True).drop_duplicates(KEYS)
        protected = set()
        benchmarks = {}
        for name, path in config["benchmarks"].items():
            frame = pd.read_parquet(source_path(path))
            for lang in LANGUAGES:
                if lang not in frame or frame[lang].isna().any():
                    raise ValueError(
                        f"Invalid benchmark language column: {path} / {lang}"
                    )
                frame[lang] = [normalize_text(lang, text) for text in frame[lang]]
                if frame[lang].eq("").any():
                    raise ValueError(f"Empty benchmark text: {path} / {lang}")
                protected.update((lang, text.casefold()) for text in frame[lang])
            benchmarks[name] = frame
        print(
            "Checking both-side overlap across all pairs and benchmarks...", flush=True
        )
        val_bad = overlap(validation, protected)
        protected |= text_keys(validation)
        validation = validation[~val_bad].copy()
        human_bad, teacher_bad = overlap(human, protected), overlap(teacher, protected)
        human, teacher = human[~human_bad].copy(), teacher[~teacher_bad].copy()
        exact_reference = teacher.row_id.isin(set(human.row_id))
        teacher = teacher[~exact_reference].copy()
        for name, frame in (
            ("human", human),
            ("teacher", teacher),
            ("validation", validation),
        ):
            if set(frame.direction) != set(DIRECTIONS):
                raise ValueError(f"Clean {name} pool must contain all 12 directions.")
        # Fail before tokenizer/training work if either arm's quotas are infeasible.
        for group in ("human_only", "human_kd"):
            sampling_plan(
                config,
                group,
                human[["direction"]].to_dict("records"),
                teacher[["direction"]].to_dict("records"),
                block=0,
            )
        audit = {
            "human_removed": int(human_bad.sum()),
            "teacher_overlap_removed": int(teacher_bad.sum()),
            "teacher_reference_copies_removed": int(exact_reference.sum()),
            "validation_benchmark_removed": int(val_bad.sum()),
            "remaining_protected_overlap": int(
                overlap(human, protected).sum() + overlap(teacher, protected).sum()
            ),
        }
        # One vocabulary shared by the NEW arms; no KD, validation or test text enters it.
        print(
            "Training a new SentencePiece vocabulary on human TRAINING text only...",
            flush=True,
        )
        corpus = writable(root / "tokenizer_corpus.txt")
        human_text = {lang: set() for lang in LANGUAGES}
        for row in human.itertuples(index=False):
            human_text[row.src_lang].add(row.src_text)
            human_text[row.tgt_lang].add(row.tgt_text)
        limit = min(
            config["tokenizer"]["max_sentences_per_language"],
            *(len(texts) for texts in human_text.values()),
        )
        with corpus.open("w", encoding="utf-8") as stream:
            for lang, texts in human_text.items():
                for text in sorted(
                    texts,
                    key=lambda text: digest([config["experiment"]["seed"], lang, text]),
                )[:limit]:
                    stream.write(text + "\n")
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(writable(root / "vocabulary")),
            model_type="unigram",
            vocab_size=config["tokenizer"]["vocab_size"],
            character_coverage=config["tokenizer"]["character_coverage"],
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=[f"<lang:{lang}>" for lang in LANGUAGES],
            input_sentence_size=0,
            shuffle_input_sentence=False,
            num_threads=1,
            hard_vocab_limit=False,
            max_sentence_length=65536,
        )
        vocab = Vocabulary(root / "vocabulary.model", config["model"]["max_length"])
        count = min(
            config["evaluation"]["validation_per_direction"],
            int(validation.groupby("direction").size().min()),
        )
        val_parts = []
        for direction in DIRECTIONS:
            part = validation[validation.direction == direction].copy()
            part["order"] = part.row_id.map(
                lambda key: digest([config["experiment"]["seed"], key])
            )
            val_parts.append(
                part.sort_values("order").head(count).drop(columns="order")
            )
        selected_validation = pd.concat(val_parts, ignore_index=True)
        test_rows = [
            dict(
                src_lang=a,
                tgt_lang=b,
                src_text=str(row[a]),
                tgt_text=str(row[b]),
                direction=f"{a}-{b}",
                kind="benchmark",
            )
            for row in benchmarks["test"].to_dict("records")
            for a in LANGUAGES
            for b in LANGUAGES
            if a != b
        ]
        reports = {}
        for name, frame in (
            ("human", human),
            ("teacher", teacher),
            ("validation", selected_validation),
            ("test", pd.DataFrame(test_rows)),
        ):
            print(f"Encoding {name}: {len(frame):,} rows...", flush=True)
            encoded = pd.DataFrame(
                [vocab.encode(row) for row in frame.to_dict("records")]
            )
            encoded.to_parquet(writable(root / f"{name}.parquet"), index=False)
            reports[name] = {
                "rows": len(frame),
                "directions": frame.direction.value_counts().sort_index().to_dict(),
                "source_truncated": int(encoded.source_truncated.sum()),
                "target_truncated": int(encoded.target_truncated.sum()),
            }
        # Do not mark complete if a shared source changed during preparation.
        if inputs != {name: file_hash(source_path(name)) for name in paths}:
            raise RuntimeError(
                "A shared source changed during preparation. No complete manifest was written."
            )
        outputs = [
            "human.parquet",
            "teacher.parquet",
            "validation.parquet",
            "test.parquet",
            "vocabulary.model",
            "vocabulary.vocab",
            "tokenizer_corpus.txt",
        ]
        manifest = {
            "signature": signature,
            "audit": audit,
            "pools": reports,
            "vocabulary_size": vocab.sp.get_piece_size(),
            "tokenizer_sentences_per_language": limit,
            "files": {name: file_hash(root / name) for name in outputs},
        }
        atomic_json(manifest_path, manifest)
        print(f"Preparation complete: {manifest_path}", flush=True)
        return manifest


def load_prepared(config):
    root = suite_root(config) / "prepared"
    manifest = read_json(root / "manifest.json")
    if manifest["signature"] != prepare_signature(
        config, manifest["signature"]["inputs"]
    ):
        raise RuntimeError(
            "Prepared configuration/code mismatch. Choose a new suite and prepare again."
        )
    verify_files(root, manifest["files"])
    return root, manifest


def cycle_indices(size: int, offset: int, count: int, seed: int) -> list[int]:
    """A shuffled coverage ring; cursor persists across blocks, repeats differ by <=1.

    The ring stays fixed so crossing its boundary cannot select the same row twice
    before all other rows were visited. The final mixed block is shuffled separately.
    """
    if size < 1 or offset < 0 or count < 0:
        raise ValueError("Invalid coverage sampling request.")
    order = np.random.default_rng(seed).permutation(size)
    return order[(np.arange(count) + offset) % size].tolist()


def sampling_plan(
    config, group: str, human: list[dict], teacher: list[dict], block: int
):
    if group not in ("human_only", "human_kd"):
        raise ValueError(group)
    quota = config["sampling"]["rows_per_direction"]
    cap = config["sampling"]["max_teacher_repeats_per_block"]
    teacher_quota = (
        round(quota * config["sampling"]["kd_ratio"]) if group == "human_kd" else 0
    )
    seed = config["experiment"]["seed"]
    result, report = [], {}
    for index, direction in enumerate(DIRECTIONS):
        pools = {
            "human": [
                i for i, row in enumerate(human) if row["direction"] == direction
            ],
            "teacher": [
                i for i, row in enumerate(teacher) if row["direction"] == direction
            ],
        }
        if teacher_quota and (
            not pools["teacher"]
            or math.ceil(teacher_quota / len(pools["teacher"])) > cap
        ):
            raise ValueError(
                f"{direction}: KD quota exceeds repetition cap. Adjust BOTH arms' common config/new suite."
            )
        summary = {}
        for kind, count in (
            ("human", quota - teacher_quota),
            ("teacher", teacher_quota),
        ):
            if not count:
                continue
            pool = pools[kind]
            chosen = cycle_indices(
                len(pool), block * count, count, seed + index * 2 + (kind == "teacher")
            )
            result.extend((kind, pool[i]) for i in chosen)
            multiplicities = Counter(chosen)
            summary[kind] = {
                "draws": count,
                "unique": len(multiplicities),
                "max_repeats": max(multiplicities.values()),
            }
            if kind == "teacher" and max(multiplicities.values()) > cap:
                raise ValueError(
                    f"{direction}: coverage-cycle boundary exceeded Teacher repetition cap."
                )
        report[direction] = summary
    rng = np.random.default_rng(np.random.SeedSequence([seed, block, 999]))
    rng.shuffle(result)
    return result, report
