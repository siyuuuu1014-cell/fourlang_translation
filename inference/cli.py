from __future__ import annotations

import argparse
import json

from .registry import ModelRegistry


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "FourLang unified translation CLI."
        )
    )

    parser.add_argument(
        "--direction",
        default=None,
        help=(
            "Translation direction, e.g. en_zh."
        ),
    )

    parser.add_argument(
        "--text",
        default=None,
        help=(
            "One-shot text. Omit for interactive mode."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        choices=[
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Print one-shot result as JSON."
        ),
    )

    parser.add_argument(
        "--warmup",
        action="store_true",
        help=(
            "Warm up the selected direction "
            "before the real translation."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help=(
            "List ready translation directions."
        ),
    )

    parser.add_argument(
        "--list-all",
        action="store_true",
        help=(
            "List ready/staged/disabled registry entries."
        ),
    )

    return parser


def print_result(
    result: dict,
    *,
    as_json: bool = False,
):
    if as_json:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("\nTranslation:")
    print(
        result[
            "translation"
        ]
    )

    print(
        "\nModel:",
        result[
            "model"
        ],
    )

    print(
        "Direction:",
        result[
            "direction"
        ],
    )

    print(
        "Latency:",
        (
            f"{result['generation_latency_seconds']:.4f}s"
        ),
    )


def interactive(
    engine,
    initial_direction: str | None,
):
    ready = (
        engine
        .available_directions(
            ready_only=True
        )
    )

    if not ready:
        raise RuntimeError(
            "No ready models are registered."
        )

    direction = initial_direction

    if direction is None:
        print(
            "\nReady directions:"
        )

        for index, item in enumerate(
            ready,
            start=1,
        ):
            print(
                f"  {index}. {item}"
            )

        while direction is None:
            choice = input(
                "\nSelect direction: "
            ).strip()

            if choice in ready:
                direction = choice
                break

            if choice.isdigit():
                index = int(
                    choice
                ) - 1

                if (
                    0
                    <=
                    index
                    <
                    len(
                        ready
                    )
                ):
                    direction = ready[
                        index
                    ]

                    break

            print(
                "Invalid selection."
            )

    if direction not in ready:
        raise RuntimeError(
            f"Direction {direction!r} is not ready."
        )

    print(
        "\nCurrent direction:",
        direction
    )

    print(
        "Commands:"
        "\n  /quit"
        "\n  /directions"
        "\n  /use <direction>"
        "\n  /warmup"
        "\n  /unload"
    )

    while True:
        text = input(
            "\n> "
        ).strip()

        if not text:
            continue

        if text in {
            "/quit",
            "/exit",
            "quit",
            "exit",
            "q",
        }:
            return

        if text == "/directions":
            print(
                ", ".join(
                    engine.available_directions(
                        ready_only=True
                    )
                )
            )
            continue

        if text.startswith(
            "/use "
        ):
            candidate = (
                text[5:]
                .strip()
                .lower()
            )

            if (
                candidate
                not in
                engine.available_directions(
                    ready_only=True
                )
            ):
                print(
                    "Direction is not ready."
                )
                continue

            direction = candidate

            print(
                "Current direction:",
                direction
            )

            continue

        if text == "/warmup":
            result = engine.warmup(
                direction
            )

            print(
                "Warm-up complete:",
                (
                    f"{result['generation_latency_seconds']:.4f}s"
                ),
            )

            continue

        if text == "/unload":
            engine.unload(
                direction
            )

            print(
                "Model unloaded from cache."
            )

            continue

        result = engine.translate(
            direction,
            text,
        )

        print_result(
            result
        )


def main():
    args = (
        build_parser()
        .parse_args()
    )

    # Registry-only commands stay lightweight and do not import
    # torch/transformers or load any model.
    if args.list_all:
        registry = ModelRegistry()

        print(
            json.dumps(
                registry.describe(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.list:
        registry = ModelRegistry()

        for direction in (
            registry.directions(
                ready_only=True
            )
        ):
            print(
                direction
            )
        return

    from .engine import TranslatorEngine

    engine = TranslatorEngine(
        device=args.device
    )

    if (
        args.text is not None
        and
        args.direction is None
    ):
        raise SystemExit(
            "--text requires --direction."
        )

    if args.warmup:
        if args.direction is None:
            raise SystemExit(
                "--warmup requires --direction."
            )

        print(
            f"Warming up {args.direction}..."
        )

        warmup_result = engine.warmup(
            args.direction
        )

        print(
            "Warm-up complete:",
            (
                f"{warmup_result['generation_latency_seconds']:.4f}s"
            ),
        )

    if args.text is not None:
        result = engine.translate(
            args.direction,
            args.text,
        )

        print_result(
            result,
            as_json=args.json,
        )

        return

    interactive(
        engine,
        args.direction,
    )


if __name__ == "__main__":
    main()
