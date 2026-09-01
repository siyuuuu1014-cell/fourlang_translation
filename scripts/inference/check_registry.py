from __future__ import annotations

import json

from inference.registry import ModelRegistry


def main():
    registry = ModelRegistry()

    print("=" * 100)
    print("FOURLANG MODEL REGISTRY CHECK")
    print("=" * 100)

    rows = registry.describe()

    print(
        json.dumps(
            rows,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\nReady:")
    print(
        registry.directions(
            ready_only=True
        )
    )

    print("\nAll:")
    print(
        registry.directions(
            ready_only=False
        )
    )


if __name__ == "__main__":
    main()
