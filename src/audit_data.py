from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
HAN_RE = re.compile(r"[\u3400-\u9FFF]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit translation JSONL files without loading them into RAM")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", default="reports/data_audit.json")
    return parser.parse_args()


def script_name(text: str) -> str:
    counts = {
        "cyrillic": len(CYRILLIC_RE.findall(text)),
        "latin": len(LATIN_RE.findall(text)),
        "han": len(HAN_RE.findall(text)),
    }
    name, count = max(counts.items(), key=lambda item: item[1])
    return name if count else "other"


def audit_file(path: Path) -> dict[str, object]:
    directions: Counter[str] = Counter()
    source_scripts: Counter[str] = Counter()
    target_scripts: Counter[str] = Counter()
    seen: set[str] = set()
    total = empty = identical = duplicates = invalid_json = 0
    source_chars = target_chars = 0

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            total += 1
            source = str(row.get("src_text", "")).strip()
            target = str(row.get("tgt_text", "")).strip()
            src_lang = str(row.get("src_lang", ""))
            tgt_lang = str(row.get("tgt_lang", ""))
            directions[f"{src_lang}-{tgt_lang}"] += 1
            source_scripts[script_name(source)] += 1
            target_scripts[script_name(target)] += 1
            source_chars += len(source)
            target_chars += len(target)

            if not source or not target:
                empty += 1
            if source.casefold() == target.casefold() and source:
                identical += 1
            digest = hashlib.sha256(f"{src_lang}\0{tgt_lang}\0{source}\0{target}".encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates += 1
            else:
                seen.add(digest)

    return {
        "path": str(path),
        "samples": total,
        "directions": dict(sorted(directions.items())),
        "source_scripts": dict(sorted(source_scripts.items())),
        "target_scripts": dict(sorted(target_scripts.items())),
        "empty_pairs": empty,
        "identical_pairs": identical,
        "exact_duplicate_pairs": duplicates,
        "invalid_json_lines": invalid_json,
        "average_source_chars": source_chars / total if total else 0,
        "average_target_chars": target_chars / total if total else 0,
    }


def main() -> None:
    args = parse_args()
    report = {"files": [audit_file(Path(value).resolve()) for value in args.files]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
