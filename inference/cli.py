"""JSON CLI and interactive shell for four-language translation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .engine import SUPPORTED_DIRECTIONS, TranslationEngine, parse_direction
from .loader import DEFAULT_MODEL_PATH, load_translation_model


def _direction(value: str) -> str:
    try:
        source, target = parse_direction(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return f"{source}-{target}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate between Chinese, English, Russian, and Uzbek"
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Full model path or Hugging Face model ID",
    )
    parser.add_argument("--adapter-path", help="Optional PEFT/LoRA adapter path")
    parser.add_argument("--direction", type=_direction, default="zh-en")
    parser.add_argument("--text", help="Translate once; omit for interactive mode")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--list-directions",
        action="store_true",
        help="Print supported directions as JSON without loading a model",
    )
    return parser


def emit(payload: dict[str, Any], pretty: bool = False) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if pretty else None,
        ),
        flush=True,
    )


def _interactive(engine: TranslationEngine, pretty: bool) -> int:
    emit(
        {
            "ok": True,
            "event": "ready",
            "direction": engine.direction,
            "commands": ["/direction zh-en", "/directions", "/help", "/quit"],
        },
        pretty,
    )
    while True:
        print(f"[{engine.direction}]> ", end="", file=sys.stderr, flush=True)
        line = sys.stdin.readline()
        if line == "":
            return 0
        value = line.strip()
        if not value:
            continue
        if value in {"/quit", "/exit"}:
            emit({"ok": True, "event": "bye"}, pretty)
            return 0
        if value == "/directions":
            emit({"ok": True, "directions": list(SUPPORTED_DIRECTIONS)}, pretty)
            continue
        if value == "/help":
            emit(
                {
                    "ok": True,
                    "commands": [
                        "/direction <src>-<tgt>",
                        "/directions",
                        "/help",
                        "/quit",
                    ],
                },
                pretty,
            )
            continue
        if value.startswith("/direction "):
            try:
                selected = engine.set_direction(value.split(maxsplit=1)[1])
                emit(
                    {"ok": True, "event": "direction_changed", "direction": selected},
                    pretty,
                )
            except ValueError as exc:
                emit({"ok": False, "error": str(exc)}, pretty)
            continue
        try:
            emit(engine.translate(value), pretty)
        except Exception as exc:  # Keep an interactive session alive after one bad input.
            emit({"ok": False, "error": str(exc)}, pretty)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_directions:
        emit({"ok": True, "directions": list(SUPPORTED_DIRECTIONS)}, args.pretty)
        return 0

    try:
        loaded = load_translation_model(
            args.model_path,
            adapter_path=args.adapter_path,
            device=args.device,
            dtype=args.dtype,
        )
        engine = TranslationEngine(
            loaded=loaded,
            direction=args.direction,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        if args.text is not None:
            emit(engine.translate(args.text), args.pretty)
            return 0
        return _interactive(engine, args.pretty)
    except Exception as exc:
        emit({"ok": False, "error": str(exc)}, args.pretty)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
