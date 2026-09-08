"""Diagnostic-only Exp1/Exp2 pilot: overlap checks, resumable inference, blind packet."""

from __future__ import annotations

import argparse
import gc
import html
import secrets
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from filelock import FileLock  # noqa: E402
from scripts.pipeline_v3 import diagnose_zh_uz as diag  # noqa: E402

LANGS = ("en", "zh", "uz", "ru")
DEFAULT_CORPORA = [
    f"data/multilingual/fourlang/{exp}/{split}.jsonl"
    for exp in ("exp1", "exp2")
    for split in ("train", "validation")
]


def normalized(text):
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def load_tasks(directory):
    manifest = diag.read_json(directory / "manifest.json")
    for name in ("sources.jsonl", "tasks.jsonl"):
        if diag.file_sha256(directory / name) != manifest["files"][name]:
            raise ValueError(
                "Pilot changed: create a new reviewed version and manifest"
            )
    sources = diag.read_rows(directory / "sources.jsonl")
    tasks = diag.read_rows(directory / "tasks.jsonl")
    by_id = {s["source_id"]: s for s in sources}
    if len(by_id) != len(sources) or len({r["task_id"] for r in tasks}) != len(tasks):
        raise ValueError("Duplicate IDs")
    expected = {
        (s["source_id"], lang)
        for s in sources
        for lang in LANGS
        if lang != s["src_lang"]
    }
    if {(r["source_id"], r["tgt_lang"]) for r in tasks} != expected or len(
        tasks
    ) != len(expected):
        raise ValueError("Incomplete/duplicate direction coverage")
    for r in tasks:
        s = by_id[r["source_id"]]
        if (
            r["src_lang"] not in LANGS
            or r["src_text"] != s["src_text"]
            or r["src_lang"] != s["src_lang"]
            or not normalized(r["src_text"])
        ):
            raise ValueError("Invalid source mapping")
        if r["usage"] != "ACCEPTANCE_ONLY_NOT_FOR_TRAINING":
            raise ValueError("Pilot must be acceptance-only")
    return sources, sorted(tasks, key=lambda r: r["task_id"])


def overlap(sources, paths):
    keys = defaultdict(list)
    for s in sources:
        keys[(s["src_lang"], normalized(s["src_text"]))].append(s["source_id"])
    hits, files = [], {}
    for path in paths:
        diag.log(f"Checking overlap: {path}")
        digest = diag.file_sha256(path)
        frame = diag.normalize_rows(diag._read_table(path), origin=str(path))
        matches = Counter()
        for row in frame.to_dict("records"):
            for side in ("src", "tgt"):
                for sid in keys.get(
                    (row[f"{side}_lang"], normalized(row[f"{side}_text"])), []
                ):
                    matches[(sid, side)] += 1
        if diag.file_sha256(path) != digest:
            raise RuntimeError("Corpus changed during scan")
        files[str(path)] = digest
        hits.extend(
            {"source_id": sid, "side": side, "occurrences": n, "file": str(path)}
            for (sid, side), n in sorted(matches.items())
        )
    return {
        "method": "NFKC + casefold + alphanumeric only; both source and target",
        "semantic_near_duplicate_check": "NOT_PERFORMED",
        "coverage": "Specified final datasets only; not all raw/KD pools or pretraining",
        "files": files,
        "hits": hits,
    }


def blind_packet(tasks, predictions):
    packet, key = [], []
    for r in tasks:
        roles = ["exp1", "exp2"]
        if secrets.randbelow(2):
            roles.reverse()
        bid = "case-" + secrets.token_hex(8)
        packet.append(
            {
                "blind_id": bid,
                "src_lang": r["src_lang"],
                "tgt_lang": r["tgt_lang"],
                "scenario": r["scenario"],
                "src_text": r["src_text"],
                "A": predictions[roles[0]][r["task_id"]],
                "B": predictions[roles[1]][r["task_id"]],
                "grade_A": "",
                "grade_B": "",
                "errors_A": [],
                "errors_B": [],
                "reviewer": "",
                "notes": "",
            }
        )
        key.append(
            {"blind_id": bid, "task_id": r["task_id"], "A": roles[0], "B": roles[1]}
        )
    secrets.SystemRandom().shuffle(packet)
    return {"review": packet, "private_key": key}


def render(packet):
    # Read-only printable document, not an app. Escape all model output as text.
    esc = html.escape
    sections = []
    for r in packet:
        sections.append(
            f"<section><h2>{esc(r['blind_id'])} · {esc(r['src_lang'])} → {esc(r['tgt_lang'])}</h2><p>原文：{esc(r['src_text'])}</p><p>A：{esc(r['A'])}</p><p>B：{esc(r['B'])}</p><p>A 等级：________　B 等级：________</p><p>错误及审核人：________________________________</p></section>"
        )
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>匿名翻译诊断审核</title><style>body{max-width:960px;margin:30px auto;font:16px/1.7 sans-serif}section{border-top:1px solid #aaa;padding:12px 0;break-inside:avoid}h2{font-size:17px}p{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>匿名翻译诊断审核</h1><p>AI 场景草稿，非正式上线验收。等级：DIRECT / EDIT / FAIL / UNJUDGEABLE。未填不算通过。</p>'
        + "".join(sections)
        + "</html>"
    )


def infer(tasks, output, signature, config, models, batch_size):
    predictions = {}
    groups = defaultdict(list)
    for r in tasks:
        groups[(r["src_lang"], r["tgt_lang"])].append({**r, "tgt_text": ""})
    for role, path in models.items():
        predictions[role] = {}
        tokenizer = model = predict = None
        try:
            for (src, tgt), rows in sorted(groups.items()):
                folder = output / "chunks" / role / f"{src}-{tgt}"
                folder.mkdir(parents=True, exist_ok=True)
                result = diag.chunk_predictions(
                    folder, rows, signature, batch_size, None
                )
                if result is None:
                    if model is None:
                        diag.log(f"Loading {role}: {path}")
                        diag.flow.set_seed(2026)
                        tokenizer, model = diag.flow.load_model(
                            {"family": "nllb", "path": str(path)},
                            src,
                            tgt,
                            training=True,
                        )
                        model.eval()
                        model.requires_grad_(False)
                    tokenizer.src_lang = config["language_codes"]["nllb"][src]
                    for r in rows:
                        if (
                            len(tokenizer.encode(r["src_text"]))
                            > config["training"]["max_source_length"]
                        ):
                            raise ValueError(
                                f"Source too long; refusing truncation: {r['task_id']}"
                            )

                    def predict(texts, tok=tokenizer, mdl=model):
                        with diag.flow.torch.autocast(
                            device_type="cuda",
                            dtype=diag.flow.torch.float16,
                            enabled=diag.flow.torch.cuda.is_available(),
                        ):
                            return diag.flow.translate(
                                tok, mdl, "nllb", src, tgt, texts, config
                            )

                    result = diag.chunk_predictions(
                        folder, rows, signature, batch_size, predict
                    )
                predictions[role].update(
                    {r["task_id"]: text for r, text in zip(rows, result, strict=True)}
                )
        finally:
            del predict, tokenizer, model
            gc.collect()
            if diag.flow.torch.cuda.is_available():
                diag.flow.torch.cuda.empty_cache()
    return predictions


def run(args):
    if args.batch_size <= 0:
        raise ValueError("Positive batch size required")
    directory = diag.project_path(args.pilot)
    sources, tasks = load_tasks(directory)
    output = diag.checked_output(args.output)
    config = diag.load_config(args.config)
    paths = [diag.project_path(p) for p in DEFAULT_CORPORA + args.extra_corpus]
    scan = overlap(sources, paths)
    models = {role: diag.project_path(getattr(args, role)) for role in ("exp1", "exp2")}
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".lock"), timeout=0):
        if not (output / "manifest.json").exists() and set(
            p.name for p in output.iterdir()
        ) - {".lock"}:
            raise ValueError("Unowned output directory")
        signature = diag.bind_manifest(
            output / "manifest.json",
            {
                "tasks": diag.fingerprint(tasks),
                "corpora": scan["files"],
                "config": config,
                "batch_size": args.batch_size,
                "models": {k: str(v) for k, v in models.items()},
                "code": {
                    str(p): diag.file_sha256(p)
                    for p in (
                        Path(__file__),
                        Path(diag.__file__),
                        Path(diag.flow.__file__),
                    )
                },
                "versions": diag.versions(),
            },
        )
        diag.preserve_text(output / ".gitignore", "*\n")
        diag.save_json(output / "overlap_report.json", scan)
        if scan["hits"]:
            raise ValueError(
                "Overlap found; inspect report. No automatic deletion/inference."
            )
        if args.check_only:
            diag.log(
                "Overlap check complete; no models loaded. Near-duplicate check not performed."
            )
            return
        diag.log("Hashing local model files for safe resume...")
        model_hash = diag.bind_manifest(
            output / "model_manifest.json",
            {role: diag.model_signature(p) for role, p in models.items()},
        )
        started = time.monotonic()
        predictions = infer(
            tasks,
            output,
            diag.fingerprint([signature, model_hash]),
            config,
            models,
            args.batch_size,
        )
        diag.save_json(output / "organizer_predictions.json", predictions)
        private = output / "organizer_only"
        private.mkdir(exist_ok=True)
        assignment = private / "assignment.json"
        if not assignment.exists():
            diag.save_json(assignment, blind_packet(tasks, predictions))
        bundle = diag.read_json(assignment)
        # Private assignment is itself fingerprint-bound before publishing.
        diag.bind_manifest(
            private / "assignment_manifest.json",
            {"sha256": diag.file_sha256(assignment), "run": signature},
        )
        diag.save_jsonl(output / "blind_review.jsonl", bundle["review"])
        diag.preserve_text(output / "blind_review.html", render(bundle["review"]))
        diag.save_json(
            output / "summary.json",
            {
                "status": "DIAGNOSTIC_TRANSLATIONS_READY_NOT_ACCEPTANCE_APPROVED",
                "source_rows": len(sources),
                "tasks_per_model": len(tasks),
                "translations": 2 * len(tasks),
                "human_scores": "PENDING",
                "training_started": False,
                "source_language_review": "NOT_CERTIFIED",
                "near_duplicate_check": "NOT_PERFORMED",
                "direction_counts": dict(
                    Counter(f"{r['src_lang']}-{r['tgt_lang']}" for r in tasks)
                ),
            },
        )
        diag.log(
            f"Done in {time.monotonic() - started:.1f}s this invocation. Send only blind_review.html/jsonl to reviewers, NOT organizer files."
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", default="acceptance/fourlang_pilot_v1")
    p.add_argument("--config", default="configs/multilingual/fourlang.toml")
    p.add_argument(
        "--output", default="reports/diagnostics/fourlang/acceptance_pilot_v1"
    )
    for role in ("exp1", "exp2"):
        p.add_argument(
            f"--{role}", default=f"results/student/fourlang/{role}/best_model/shared"
        )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--extra-corpus", action="append", default=[])
    p.add_argument("--check-only", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
