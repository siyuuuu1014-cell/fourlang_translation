from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    pair_info,
    project_path,
    write_json,
)
from scripts.pipeline_v2.data_flow import normalized_key, pair_hash, paths  # noqa: E402
from scripts.pipeline_v3.language_normalization import (  # noqa: E402
    normalize_language_text,
)


def _safe_member(archive: zipfile.ZipFile, language: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if name.endswith(f".{language}")
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one safe *.{language} member, found {sorted(matches)}"
        )
    return matches[0]

def _decode_moses_lines(payload: bytes) -> list[str]:
    lines = payload.decode("utf-8", errors="replace").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.removesuffix("\r") for line in lines]

def read_parallel_archive(
    path: Path, archive_source: str, archive_target: str
) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member in {path}: {bad}")
        source = _decode_moses_lines(
            archive.read(_safe_member(archive, archive_source))
        )

        target = _decode_moses_lines(
            archive.read(_safe_member(archive, archive_target))
        )
    if len(source) != len(target):
        raise RuntimeError(
            f"Unaligned OPUS archive {path}: {len(source)} != {len(target)}"
        )
    return source, target


def _download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(
        url, headers={"User-Agent": "fourlang-translation/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"Downloaded ZIP has a corrupt member: {bad}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest.hexdigest()


def download_corpora(config: dict[str, Any]) -> None:
    pair, _, _, _ = pair_info(config)
    inventory = []
    for corpus in config["data"]["corpora"]:
        destination = project_path(corpus["local_archive"])
        if destination.is_file():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            status = "reused"
        else:
            digest = _download(str(corpus["url"]), destination)
            status = "downloaded"
        source_lines, _ = read_parallel_archive(
            destination,
            str(corpus["archive_source_lang"]),
            str(corpus["archive_target_lang"]),
        )
        expected = int(corpus.get("expected_pairs", 0))
        if expected and len(source_lines) != expected:
            raise RuntimeError(
                f"OPUS row-count drift for {corpus['name']}@{corpus['version']}: "
                f"expected {expected}, found {len(source_lines)}"
            )
        inventory.append(
            {
                "corpus": corpus["name"],
                "version": corpus["version"],
                "url": corpus["url"],
                "path": str(destination.relative_to(PROJECT_ROOT)),
                "sha256": digest,
                "rows": len(source_lines),
                "status": status,
            }
        )
    write_json(
        PROJECT_ROOT / "reports/pipeline" / pair / "opus_downloads.json",
        {"schema_version": 1, "pair": pair, "corpora": inventory},
    )


def build_candidates(config: dict[str, Any]) -> None:
    pair, source, target, _ = pair_info(config)
    rows: list[dict[str, Any]] = []
    corpus_report = []
    conversions: Counter[str] = Counter()
    for corpus in config["data"]["corpora"]:
        archive_source = str(corpus["archive_source_lang"])
        archive_target = str(corpus["archive_target_lang"])
        archive_path = project_path(corpus["local_archive"])
        left, right = read_parallel_archive(
            archive_path, archive_source, archive_target
        )
        before = len(rows)
        for row_number, (left_text, right_text) in enumerate(
            zip(left, right, strict=True), start=1
        ):
            by_language = {archive_source: left_text, archive_target: right_text}
            source_raw, target_raw = by_language[source], by_language[target]
            source_text = normalize_language_text(source, source_raw)
            target_text = normalize_language_text(target, target_raw)
            conversions[f"{source}_cells"] += 1
            conversions[f"{target}_cells"] += 1
            conversions[f"{source}_converted"] += source_raw.strip() != source_text
            conversions[f"{target}_converted"] += target_raw.strip() != target_text
            rows.append(
                {
                    "pair_id": pair_hash(source_text, target_text),
                    source: source_text,
                    target: target_text,
                    "source_text": source_text,
                    "target_text": target_text,
                    "source_lang": source,
                    "target_lang": target,
                    "source_corpus": f"{corpus['name']}@{corpus['version']}",
                    "source_url": corpus["url"],
                    "source_row": row_number,
                }
            )
        corpus_report.append(
            {
                "corpus": corpus["name"],
                "version": corpus["version"],
                "input_rows": len(rows) - before,
            }
        )
    raw_rows = len(rows)
    frame = pd.DataFrame(rows)
    empty = (frame["source_text"] == "") | (frame["target_text"] == "")
    frame = frame[~empty].drop_duplicates("pair_id", keep="first").copy()
    unique_pairs = len(frame)
    source_counts = frame.groupby(frame["source_text"].map(normalized_key))[
        "pair_id"
    ].transform("count")
    target_counts = frame.groupby(frame["target_text"].map(normalized_key))[
        "pair_id"
    ].transform("count")
    ambiguous = (source_counts != 1) | (target_counts != 1)
    ambiguous_rows = int(ambiguous.sum())
    frame = frame[~ambiguous].reset_index(drop=True)
    output = paths(config)["candidates"]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_json(
        paths(config)["reports"] / "candidates.json",
        {
            "schema_version": 2,
            "pair": pair,
            "status": "CANDIDATE_ONLY",
            "raw_rows": raw_rows,
            "empty_rows": int(empty.sum()),
            "exact_duplicate_rows": raw_rows - int(empty.sum()) - unique_pairs,
            "ambiguous_one_to_many_rows": ambiguous_rows,
            "rows": len(frame),
            "corpora": corpus_report,
            "script_normalization": dict(sorted(conversions.items())),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize pinned OPUS parallel corpora."
    )
    parser.add_argument("action", choices=("download", "build"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "download":
        download_corpora(config)
    else:
        build_candidates(config)


if __name__ == "__main__":
    main()
