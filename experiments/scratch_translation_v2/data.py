from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch

from .common import DIRECTIONS, LANGUAGES, frozen_prepared


class Vocabulary:
    def __init__(self, path, max_length):
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.max_length = max_length
        self.pad, self.unk = self.sp.pad_id(), self.sp.unk_id()
        self.bos, self.eos = self.sp.bos_id(), self.sp.eos_id()
        self.languages = {lang: self.sp.piece_to_id(f"<lang:{lang}>") for lang in LANGUAGES}
        if min(self.languages.values()) <= self.unk or len(set(self.languages.values())) != 4:
            raise ValueError("Frozen vocabulary has no valid language symbols.")

    def decode(self, ids):
        tokens = []
        for token in ids:
            token = int(token)
            if token == self.eos:
                break
            if token not in (self.pad, self.bos, *self.languages.values()):
                tokens.append(token)
        return self.sp.decode(tokens)


def collate(rows, pad=0, device="cpu"):
    source = torch.full((len(rows), max(len(row["src_ids"]) for row in rows)), pad, dtype=torch.long)
    target = torch.full((len(rows), max(len(row["tgt_ids"]) for row in rows)), pad, dtype=torch.long)
    for index, row in enumerate(rows):
        source[index, :len(row["src_ids"])] = torch.tensor(row["src_ids"], dtype=torch.long)
        target[index, :len(row["tgt_ids"])] = torch.tensor(row["tgt_ids"], dtype=torch.long)
    return source.to(device), target[:, :-1].to(device), target[:, 1:].to(device)


def load_data(config):
    root, manifest = frozen_prepared(config)
    frames = {name: pd.read_parquet(root / f"{name}.parquet")
              for name in ("human", "teacher", "validation", "test")}
    for name, frame in frames.items():
        if set(frame.direction) != set(DIRECTIONS):
            raise RuntimeError(f"Frozen {name} data no longer covers all 12 directions.")
    return root, manifest, frames


def cycle_indices(size, offset, count, seed):
    if size < 1 or offset < 0 or count < 0:
        raise ValueError("Invalid coverage sampling request.")
    order = np.random.default_rng(seed).permutation(size)
    return order[(np.arange(count) + offset) % size].tolist()


def sampling_plan(config, phase, human, teacher, block):
    if phase not in ("mixed", "human"):
        raise ValueError(phase)
    quota = config["sampling"]["rows_per_direction"]
    cap = config["sampling"]["max_teacher_repeats_per_block"]
    seed = config["experiment"]["seed"]
    result, report = [], {}
    for number, direction in enumerate(DIRECTIONS):
        human_pool = [i for i, row in enumerate(human) if row["direction"] == direction]
        teacher_pool = [i for i, row in enumerate(teacher) if row["direction"] == direction]
        teacher_count = round(quota * config["sampling"]["kd_ratio"][direction]) if phase == "mixed" else 0
        counts = (("human", quota - teacher_count, human_pool),
                  ("teacher", teacher_count, teacher_pool))
        summary = {}
        for kind, count, pool in counts:
            if not count:
                continue
            if not pool:
                raise ValueError(f"{direction} has no {kind} rows.")
            chosen = cycle_indices(len(pool), block * count, count,
                                   seed + number * 2 + (kind == "teacher"))
            repeats = Counter(chosen)
            if kind == "teacher" and math.ceil(count / len(pool)) > cap:
                raise ValueError(f"{direction}: KD quota exceeds repetition cap {cap}.")
            result.extend((kind, pool[index]) for index in chosen)
            summary[kind] = {"draws": count, "unique": len(repeats),
                             "max_repeats": max(repeats.values())}
        report[direction] = summary
    np.random.default_rng(np.random.SeedSequence([seed, 200 if phase == "mixed" else 201,
                                                   block])).shuffle(result)
    return result, report
