"""Interactive translator for the M2M100 four-language model.

Loads one shared M2M100 model once and switches among all 12 directions
(zh/en/ru/uz) interactively, without reloading weights.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.engine import TranslationEngine, parse_direction  # noqa: E402
from inference.loader import load_translation_model  # noqa: E402

DEFAULT_MODEL = (
    "results/student/fourlang_m2m100/m2m100_targeted_v1/best_model/shared"
)

LANGUAGES = ("zh", "en", "ru", "uz")

DIRECTIONS = tuple(
    f"{source}-{target}"
    for source in LANGUAGES
    for target in LANGUAGES
    if source != target
)

INLINE_DIRECTION_RE = re.compile(
    r"^\s*([a-z]{2})\s*(?:->|-|_)\s*([a-z]{2})\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)
BARE_DIRECTION_RE = re.compile(
    r"^\s*([a-z]{2})\s*(?:->|-|_)\s*([a-z]{2})\s*$",
    re.IGNORECASE,
)
WATERMARK_RE = re.compile(
    r"^\s*(translated\s+(?:with|by|via)\s+[^\s]*\.[^\s]*)\s*",
    re.IGNORECASE,
)


def normalize_direction(value: str) -> str:
    source, target = parse_direction(value)
    return f"{source}-{target}"


def parse_input(value: str, current_direction: str) -> tuple[str, str, str | None]:
    """Return (action, direction, text) for the shared-model interactive shell."""
    text = value.strip()
    lowered = text.lower()
    if lowered in {"/quit", "/exit", "quit", "exit", "q"}:
        return "quit", current_direction, None
    if lowered in {"/help", "help"}:
        return "help", current_direction, None
    if lowered in {"/directions", "/list"}:
        return "directions", current_direction, None
    if lowered == "/direction" or lowered == "/direction ":
        raise ValueError("usage: /direction zh-uz")
    if lowered.startswith("/direction "):
        parts = text.split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            raise ValueError("usage: /direction zh-uz")
        return "switch", normalize_direction(parts[1]), None
    match = INLINE_DIRECTION_RE.match(text)
    if match:
        direction = normalize_direction(f"{match.group(1)}-{match.group(2)}")
        return "translate", direction, match.group(3).strip()
    bare = BARE_DIRECTION_RE.match(text)
    if bare:
        direction = normalize_direction(f"{bare.group(1)}-{bare.group(2)}")
        return "switch", direction, None
    return "translate", current_direction, text


def strip_watermark(text: str) -> tuple[str, str | None]:
    """Strip a leading memorized 'Translated with/by/via <url>' watermark."""
    match = WATERMARK_RE.match(text)
    if match:
        return text[match.end():], match.group(1)
    return text, None


def print_help() -> None:
    print("Commands:")
    print("  zh-uz              switch the default direction")
    print("  /direction zh-uz   same as above")
    print("  zh-uz: text        translate one sentence in an explicit direction")
    print("  /directions        show all 12 supported directions")
    print("  /quit              exit")


def print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["translation"])
        print(f"[model] {result['model_name']}")
        print(f"[path] {result['model_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model path; defaults to the M2M100 four-language model.",
    )
    parser.add_argument(
        "--direction",
        help="Direction for --text, e.g. zh-uz or ru_en.",
    )
    parser.add_argument("--text", help="One sentence; omit for interactive input.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Model directory has no config.json: {model_path}")

    loaded = load_translation_model(
        model_path,
        device=args.device,
        dtype=args.dtype,
    )
    direction = normalize_direction(args.direction or "zh-en")
    engine = TranslationEngine(
        loaded,
        direction=direction,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    model_name = "m2m100_fourlang"

    def translate(text: str, requested_direction: str | None = None) -> None:
        result = engine.translate(text, direction=requested_direction)
        result["model_name"] = model_name
        cleaned, stripped = strip_watermark(result["translation"])
        result["translation"] = cleaned
        if stripped:
            result["stripped_prefix"] = stripped
        print_result(result, args.json)

    if args.text is not None:
        translate(args.text)
        return 0

    print(f"Loaded M2M100 four-language model: {model_path}")
    print(
        "Default direction is zh-en. Switch with a bare direction like 'zh-uz', "
        "or translate one line as 'zh-uz: text'. Enter /help for commands."
    )
    current_direction = direction
    while True:
        try:
            raw = input(f"[{current_direction}]> ")
        except EOFError:
            break
        if not raw.strip():
            continue
        try:
            action, requested_direction, sentence = parse_input(
                raw, current_direction
            )
        except ValueError as error:
            print(f"[error] {error}")
            continue
        if action == "quit":
            break
        if action == "help":
            print_help()
            continue
        if action == "directions":
            print(" ".join(DIRECTIONS))
            continue
        if action == "switch":
            current_direction = requested_direction
            engine.set_direction(current_direction)
            print(f"[direction] {current_direction}")
            continue
        if sentence:
            translate(sentence, requested_direction)
            current_direction = requested_direction
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
