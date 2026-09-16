"""Interactive translator for the six bidirectional SMaLL-100 pair specialists.

Each pair model covers two directions. Models are loaded lazily on first use and
cached in memory, so switching between the two directions of a pair reuses the
same weights without reloading.
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

DEFAULT_MANIFEST = PROJECT_ROOT / "configs/specialists/current_pair_models.json"

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


def load_routes(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, dict[str, str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    routes: dict[str, dict[str, str]] = {}
    for item in payload["pairs"]:
        model_name = item["model_name"]
        model_path = item["model_path"]
        for direction in item["directions"]:
            routes[direction] = {"model_name": model_name, "path": model_path}
    return routes


ROUTES = load_routes()


def normalize_direction(value: str) -> str:
    source, target = parse_direction(value)
    return f"{source}-{target}"


def resolve_path(direction: str) -> tuple[dict[str, str], Path]:
    route = ROUTES[direction]
    path = Path(route["path"]).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return route, path.resolve()


def parse_input(value: str, current_direction: str) -> tuple[str, str, str | None]:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"/quit", "/exit", "quit", "exit", "q"}:
        return "quit", current_direction, None
    if lowered in {"/help", "help"}:
        return "help", current_direction, None
    if lowered in {"/directions", "/list"}:
        return "directions", current_direction, None
    if lowered in {"/loaded"}:
        return "loaded", current_direction, None
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
    match = WATERMARK_RE.match(text)
    if match:
        return text[match.end():], match.group(1)
    return text, None


def print_help() -> None:
    print("Commands:")
    print("  zh-uz              switch the default direction")
    print("  /direction zh-uz   same as above")
    print("  zh-uz: text        translate one sentence in an explicit direction")
    print("  /directions        show all 12 directions")
    print("  /loaded            show which pair models are loaded")
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
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all 12 routes without loading model weights",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.list_models:
        rows = []
        for direction in DIRECTIONS:
            route, path = resolve_path(direction)
            rows.append(
                {
                    "direction": direction,
                    "model_name": route["model_name"],
                    "model_path": str(path),
                    "available": path.is_dir(),
                }
            )
        print(json.dumps({"schema_version": 1, "models": rows}, ensure_ascii=False, indent=2))
        return 0

    cache: dict[str, tuple[str, TranslationEngine]] = {}

    def engine_for(direction: str) -> tuple[str, TranslationEngine]:
        route, path = resolve_path(direction)
        key = str(path)
        entry = cache.get(key)
        if entry is None:
            if not path.is_dir():
                raise FileNotFoundError(f"Model does not exist: {path}")
            if not (path / "config.json").is_file():
                raise FileNotFoundError(f"Model has no config.json: {path}")
            loaded = load_translation_model(
                path,
                device=args.device,
                dtype=args.dtype,
            )
            engine = TranslationEngine(
                loaded,
                direction=direction,
                max_source_length=args.max_source_length,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
            entry = (route["model_name"], engine)
            cache[key] = entry
            print(f"[loading {route['model_name']}] {path}")
        return entry

    def translate(text: str, requested_direction: str | None = None) -> None:
        direction = normalize_direction(requested_direction or current_direction)
        model_name, engine = engine_for(direction)
        result = engine.translate(text, direction=direction)
        result["model_name"] = model_name
        cleaned, stripped = strip_watermark(result["translation"])
        result["translation"] = cleaned
        if stripped:
            result["stripped_prefix"] = stripped
        print_result(result, args.json)

    if args.text is not None:
        current_direction = normalize_direction(args.direction or "zh-en")
        translate(args.text)
        return 0

    current_direction = normalize_direction("zh-en")
    print(
        "Six-pair specialist translator. Models are loaded lazily and cached. "
        "Switch with a bare direction like 'zh-uz', or translate one line as "
        "'zh-uz: text'. Enter /help for commands."
    )
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
        if action == "loaded":
            if cache:
                for name, _ in cache.values():
                    print(f"  loaded: {name}")
            else:
                print("  (no models loaded yet)")
            continue
        if action == "switch":
            current_direction = requested_direction
            print(f"[direction] {current_direction}")
            continue
        if sentence:
            translate(sentence, requested_direction)
            current_direction = requested_direction
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
