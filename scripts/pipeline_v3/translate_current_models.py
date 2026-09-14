"""Translate with the currently selected specialist for any FourLang direction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from inference.engine import TranslationEngine, parse_direction  # noqa: E402
from inference.loader import load_translation_model  # noqa: E402


# One shared bidirectional model per unordered language pair. These are the
# latest selected/validated artifacts, not training checkpoints chosen ad hoc.
PAIR_MODELS = {
    "en_zh": {
        "languages": ("en", "zh"),
        "name": "en_zh_pair_exp2",
        "path": "results/student/pair_specialists/en_zh/exp2/best_model/shared",
    },
    "en_uz": {
        "languages": ("en", "uz"),
        "name": "en_uz_small100_exp2_v1",
        "path": "models/final_specialists/en_uz_small100_v1",
    },
    "en_ru": {
        "languages": ("en", "ru"),
        "name": "en_ru_pair_exp2",
        "path": "results/student/pair_specialists/en_ru/exp2/best_model/shared",
    },
    "zh_uz": {
        "languages": ("zh", "uz"),
        "name": "zh_uz_flores_relaxed_8k_ep3",
        "path": (
            "results/experiments/weak_pair_ablation/zh_uz/"
            "flores_relaxed_8k_ep3/best_model/shared"
        ),
    },
    "zh_ru": {
        "languages": ("zh", "ru"),
        "name": "zh_ru_pair_exp2",
        "path": "results/student/pair_specialists/zh_ru/exp2/best_model/shared",
    },
    "uz_ru": {
        "languages": ("uz", "ru"),
        "name": "uz_ru_pair_exp2",
        "path": "results/student/pair_specialists/uz_ru/exp2/best_model/shared",
    },
}


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
    if not args.direction:
        raise SystemExit("--direction is required unless --list-models is used")

    direction = normalize_direction(args.direction)
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

    def translate(text: str) -> None:
        result = engine.translate(text)
        result["model_name"] = route["model_name"]
        print_result(result, args.json)

    if args.text is not None:
        translate(args.text)
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
