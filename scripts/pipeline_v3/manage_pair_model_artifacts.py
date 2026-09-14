"""Inventory and safely remove explicitly abandoned pair-model experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/specialists/current_pair_models.json"
CONFIRMATION = "DELETE_DISCARDED_PAIR_EXPERIMENTS"


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_inside(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"Manifest path must be project-relative: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes project root: {relative}") from exc
    return resolved


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def cleanup_plan(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    protected = [resolve_inside(root, item) for item in manifest["protected_prefixes"]]
    protected.extend(
        resolve_inside(root, pair["model_path"]) for pair in manifest["pairs"]
    )
    plan = []
    for relative in manifest["safe_delete"]:
        target = resolve_inside(root, relative)
        if any(overlaps(target, item) for item in protected):
            raise RuntimeError(f"Cleanup target overlaps protected path: {relative}")
        plan.append(
            {
                "path": relative,
                "absolute_path": str(target),
                "exists": target.exists(),
                "bytes": size_bytes(target),
            }
        )
    return plan


def model_inventory(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    def artifact(relative: str) -> dict[str, Any]:
        path = resolve_inside(root, relative)
        return {
            "path": relative,
            "available": path.exists(),
            "bytes": size_bytes(path),
        }

    rows = []
    for pair in manifest["pairs"]:
        path = resolve_inside(root, pair["model_path"])
        weight_names = (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        config_available = (path / "config.json").is_file()
        weights_available = any((path / name).is_file() for name in weight_names)
        rows.append(
            {
                "pair": pair["id"],
                "directions": pair["directions"],
                "status": pair["status"],
                "model_path": pair["model_path"],
                "available": path.is_dir(),
                "config_available": config_available,
                "weights_available": weights_available,
                "model_complete": path.is_dir()
                and config_available
                and weights_available,
                "bytes": size_bytes(path),
                "training_entrypoint": pair["training_entrypoint"],
                "training_entrypoint_available": resolve_inside(
                    root, pair["training_entrypoint"]
                ).is_file(),
                "training_config": pair["training_config"],
                "training_config_available": resolve_inside(
                    root, pair["training_config"]
                ).is_file(),
                "training_data": [artifact(item) for item in pair["training_data"]],
                "source_data_to_preserve": [
                    artifact(item) for item in pair.get("source_data_to_preserve", [])
                ],
                "validation_data": artifact(pair["validation_data"]),
                "evaluation_evidence": [
                    artifact(item) for item in pair["evaluation_evidence"]
                ],
            }
        )
    return rows


def remove_target(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "cleanup"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    inventory = model_inventory(manifest, PROJECT_ROOT)
    if args.action == "inventory":
        result = {
            "schema_version": 1,
            "suite": manifest["suite"],
            "protected_shared_model": True,
            "models": inventory,
            "active_follow_up": manifest["active_follow_up"],
        }
    else:
        plan = cleanup_plan(manifest, PROJECT_ROOT)
        missing_models = [
            item["pair"] for item in inventory if not item["model_complete"]
        ]
        if args.apply and missing_models:
            raise RuntimeError(
                "Refusing cleanup because selected models are incomplete: "
                + ", ".join(missing_models)
            )
        if args.apply and args.confirm != CONFIRMATION:
            raise RuntimeError(f"--apply requires --confirm {CONFIRMATION}")
        removed = []
        if args.apply:
            for item in plan:
                target = Path(item["absolute_path"])
                if target.exists():
                    remove_target(target)
                    removed.append(item["path"])
        result = {
            "schema_version": 1,
            "mode": "applied" if args.apply else "dry_run",
            "protected_shared_model": True,
            "selected_models_complete": not missing_models,
            "missing_selected_models": missing_models,
            "targets": plan,
            "reclaimable_bytes": sum(item["bytes"] for item in plan),
            "removed": removed,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
