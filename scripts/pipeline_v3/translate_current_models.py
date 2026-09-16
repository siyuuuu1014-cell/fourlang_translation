"""Translate with the currently selected specialist for any FourLang direction."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/specialists/current_pair_models.json"
FOURLANG_STAGES = {
    "exp1": "results/student/fourlang/exp1/best_model/shared",
    "exp2": "results/student/fourlang/exp2/best_model/shared",
    "exp3_v2": "results/student/fourlang/exp3_v2/best_model/shared",
}
INLINE_DIRECTION_RE = re.compile(
    r"^\s*([a-z]{2})\s*(?:->|-|_)\s*([a-z]{2})\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)
BARE_DIRECTION_RE = re.compile(
    r"^\s*([a-z]{2})\s*(?:->|-|_)\s*([a-z]{2})\s*$",
    re.IGNORECASE,
)

from inference.engine import TranslationEngine, parse_direction  # noqa: E402
from inference.loader import load_translation_model  # noqa: E402


def load_pair_models(
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs: dict[str, dict[str, Any]] = {}
    for item in payload["pairs"]:
        source, target = item["directions"][0].split("-")
        pairs[item["id"]] = {
            "languages": (source, target),
            "name": item["model_name"],
            "path": item["model_path"],
        }
    return pairs


PAIR_MODELS = load_pair_models()


def direction_catalog() -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for pair_id, spec in PAIR_MODELS.items():
        first, second = spec["languages"]
        for source, target in ((first, second), (second, first)):
            direction = f"{source}-{target}"
            catalog[direction] = {
                "pair": pair_id,
                "model_name": str(spec["name"]),
                "relative_path": str(spec["path"]),
            }
    return catalog


def normalize_direction(value: str) -> str:
    source, target = parse_direction(value)
    return f"{source}-{target}"


def resolve_model(
    direction: str,
    *,
    project_root: Path = PROJECT_ROOT,
    override: str | None = None,
) -> tuple[dict[str, str], Path]:
    normalized = normalize_direction(direction)
    route = direction_catalog()[normalized]
    env_name = f"FOURLANG_{normalized.upper().replace('-', '_')}_MODEL_PATH"
    selected = override or os.environ.get(env_name) or route["relative_path"]
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return route, path.resolve()


def model_manifest(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    rows = []
    for direction in sorted(direction_catalog()):
        route, path = resolve_model(direction, project_root=project_root)
        rows.append(
            {
                "direction": direction,
                "pair": route["pair"],
                "model_name": route["model_name"],
                "model_path": str(path),
                "available": path.is_dir(),
                "config_available": (path / "config.json").is_file(),
            }
        )
    return rows


def require_model(path: Path, direction: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Model for {direction} does not exist: {path}\n"
            "Run --list-models to inspect all configured paths."
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Model directory for {direction} has no config.json: {path}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", help="Direction such as zh-uz or ru_en")
    parser.add_argument(
        "--all-directions",
        action="store_true",
        help=(
            "Load one explicitly supplied shared model once, then switch among "
            "all 12 directions interactively"
        ),
    )
    parser.add_argument(
        "--fourlang",
        nargs="?",
        const="exp2",
        choices=tuple(FOURLANG_STAGES),
        metavar="STAGE",
        help=(
            "Interactive shared four-language model; optional stage is one of "
            + ", ".join(FOURLANG_STAGES)
            + " (default exp2). Implies --all-directions."
        ),
    )
    parser.add_argument("--text", help="One sentence; omit for interactive input")
    parser.add_argument(
        "--model",
        help="Optional model-path override for this run",
    )
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


def parse_multidirection_input(
    value: str, current_direction: str
) -> tuple[str, str, str | None]:
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
        direction = normalize_direction(parts[1])
        return "switch", direction, None
    match = INLINE_DIRECTION_RE.match(text)
    if match:
        direction = normalize_direction(f"{match.group(1)}-{match.group(2)}")
        return "translate", direction, match.group(3).strip()
    bare = BARE_DIRECTION_RE.match(text)
    if bare:
        direction = normalize_direction(f"{bare.group(1)}-{bare.group(2)}")
        return "switch", direction, None
    return "translate", current_direction, text


def print_multidirection_help() -> None:
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


def main() -> int:
    args = build_parser().parse_args()
    if args.list_models:
        print(
            json.dumps(
                {"schema_version": 1, "models": model_manifest()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.fourlang is not None:
        if args.direction or args.text is not None:
            raise SystemExit(
                "--fourlang is interactive; do not combine it with --direction or --text"
            )
        if args.model:
            raise SystemExit("--fourlang and --model are mutually exclusive")
        args.all_directions = True
        args.model = FOURLANG_STAGES[args.fourlang]
    if args.all_directions and args.direction:
        raise SystemExit("--all-directions cannot be combined with --direction")
    if args.all_directions and args.text is not None:
        raise SystemExit("--all-directions is interactive; omit --text")
    if args.all_directions and not args.model:
        raise SystemExit(
            "--all-directions requires --model pointing to one shared four-language model"
        )
    if not args.direction and not args.all_directions:
        raise SystemExit("--direction is required unless --list-models is used")

    direction = normalize_direction(args.direction or "zh-en")
    route, model_path = resolve_model(
        direction,
        override=args.model,
    )
    require_model(model_path, direction)
    loaded = load_translation_model(
        model_path,
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

    model_name = (
        "explicit_shared_fourlang_model"
        if args.all_directions
        else route["model_name"]
    )

    def translate(text: str, requested_direction: str | None = None) -> None:
        result = engine.translate(text, direction=requested_direction)
        result["model_name"] = model_name
        print_result(result, args.json)

    if args.text is not None:
        translate(args.text)
        return 0

    if args.all_directions:
        print(
            "Loaded one shared four-language model. Default direction is zh-en. "
            "Switch with a bare direction like 'zh-uz', or translate one line as "
            "'zh-uz: text'. Enter /help for commands."
        )
        current_direction = direction
        while True:
            try:
                text = input(f"[{current_direction}]> ")
            except EOFError:
                break
            if not text.strip():
                continue
            try:
                action, requested_direction, sentence = parse_multidirection_input(
                    text, current_direction
                )
            except ValueError as error:
                print(f"[error] {error}")
                continue
            if action == "quit":
                break
            if action == "help":
                print_multidirection_help()
                continue
            if action == "directions":
                print(" ".join(sorted(direction_catalog())))
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

    print(f"Loaded {route['model_name']} for {direction}. Enter /quit to exit.")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if text.lower() in {"/quit", "/exit", "quit", "exit", "q"}:
            break
        if text:
            translate(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
