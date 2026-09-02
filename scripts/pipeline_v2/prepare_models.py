from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    from .common import PROJECT_ROOT, commercial_candidates, load_config, pair_info, write_json
except ImportError:
    from common import PROJECT_ROOT, commercial_candidates, load_config, pair_info, write_json


def available(reference: str) -> tuple[bool, str]:
    path = Path(reference)
    if path.exists():
        return True, str(path.resolve())
    try:
        cached = snapshot_download(repo_id=reference, local_files_only=True)
        return True, cached
    except Exception:
        return False, ""


def materialize(reference: str) -> str:
    exists, resolved = available(reference)
    if exists:
        return resolved
    return snapshot_download(repo_id=reference)


def references(candidate: dict) -> list[str]:
    if candidate["family"] == "marian_pair":
        return [value for key, value in candidate.items() if key.endswith("_repo_id")]
    return [candidate["repo_id"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only commercially eligible bake-off candidates.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pair, _, _, _ = pair_info(config)
    inventory = []
    seen: set[str] = set()
    for role in ("student", "teacher"):
        for candidate in commercial_candidates(config, role):
            for repo_id in references(candidate):
                if repo_id in seen:
                    continue
                seen.add(repo_id)
                resolved = materialize(repo_id)
                inventory.append({"role": role, "candidate_id": candidate["id"], "repo_id": repo_id, "resolved_path": resolved, "license": candidate["license"], "commercial_allowed": True})
    write_json(PROJECT_ROOT / "reports" / "pipeline" / pair / "model_inventory.json", {"schema_version": 1, "commercial_use": True, "models": inventory})


if __name__ == "__main__":
    main()
