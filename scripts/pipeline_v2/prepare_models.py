from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

try:
    from .common import (
        PROJECT_ROOT,
        commercial_candidates,
        load_config,
        pair_info,
        write_json,
    )
except ImportError:
    from common import (
        PROJECT_ROOT,
        commercial_candidates,
        load_config,
        pair_info,
        write_json,
    )


def materialize(reference: str, revision: str, destination: str) -> str:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=reference,
        revision=revision,
        local_dir=path,
    )


def references(candidate: dict) -> list[tuple[str, str, str, str]]:
    if candidate["family"] == "marian_pair":
        return [
            (
                value,
                candidate[key.removesuffix("repo_id") + "revision"],
                candidate[key.removesuffix("repo_id") + "license"],
                candidate[key.removesuffix("repo_id") + "path"],
            )
            for key, value in candidate.items()
            if key.endswith("_repo_id")
        ]
    return [
        (
            candidate["repo_id"],
            candidate["revision"],
            candidate["license"],
            candidate["path"],
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only commercially eligible bake-off candidates."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pair, _, _, _ = pair_info(config)
    inventory = []
    seen: set[str] = set()
    api = HfApi()
    for role in ("student", "teacher"):
        for candidate in commercial_candidates(config, role):
            for repo_id, revision, declared_license, destination in references(candidate):
                identity = f"{repo_id}@{revision}"
                if identity in seen:
                    continue
                seen.add(identity)
                info = api.model_info(repo_id=repo_id, revision=revision)
                actual_license = str(
                    getattr(info.card_data, "license", "") or ""
                ).lower()
                if actual_license != str(declared_license).lower():
                    raise RuntimeError(
                        f"License mismatch for {repo_id}@{revision}: configured={declared_license}, hub={actual_license or 'missing'}"
                    )
                resolved = materialize(repo_id, revision, destination)
                inventory.append(
                    {
                        "role": role,
                        "candidate_id": candidate["id"],
                        "repo_id": repo_id,
                        "revision": revision,
                        "resolved_path": resolved,
                        "declared_license": declared_license,
                        "verified_hub_license": actual_license,
                        "commercial_allowed": True,
                    }
                )
    write_json(
        PROJECT_ROOT / "reports" / "pipeline" / pair / "model_inventory.json",
        {
            "schema_version": 2,
            "commercial_use": True,
            "models": inventory,
            "artifacts": [str(Path(item["resolved_path"]) / "config.json") for item in inventory],
        },
    )


if __name__ == "__main__":
    main()
