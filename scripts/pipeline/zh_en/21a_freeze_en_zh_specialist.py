from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="Freeze the selected EN->ZH specialist into the production model registry."
    )
    p.add_argument(
        "--project-root",
        "--project_root",
        dest="project_root",
        default=None,
        help=(
            "Repository root. Defaults to the root containing this script, so the "
            "same command works locally and under /root/autodl-tmp on the server."
        ),
    )
    p.add_argument(
        "--version",
        default="v1",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[3]
    )

    # Step20 showed epoch1 has the strongest held-out aggregate:
    # mean ΔBLEU +0.083432, mean ΔchrF++ +0.027692.
    source_model = (
        root
        / "results/specialists/en_zh/"
          "opus_mt_en_zh/exp2_kd_v2/"
          "epoch_models/epoch_1"
    )

    step20_report = (
        root
        / "results/specialists/en_zh/"
          "opus_mt_en_zh/exp2_kd_v2/"
          "step20_generation_eval/"
          "step20_report_v1.json"
    )

    target_model = (
        root
        / "models/final_specialists"
        / f"en_zh_{args.version}"
    )

    if not source_model.exists():
        raise FileNotFoundError(
            f"Selected source model does not exist: {source_model}"
        )

    if target_model.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Target already exists: {target_model}\n"
                "Use --overwrite only if you intentionally want to rebuild it."
            )
        shutil.rmtree(target_model)

    target_model.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("STEP 21A - FREEZE EN->ZH SPECIALIST")
    print("=" * 100)
    print("Source:", source_model)
    print("Target:", target_model)

    shutil.copytree(source_model, target_model)

    if step20_report.exists():
        report = json.loads(
            step20_report.read_text(encoding="utf-8")
        )
    else:
        report = {}

    # Extract the selected epoch1 rows from Step20 if available.
    selected_metrics = []
    for row in report.get("metrics", []):
        if row.get("model_name") == "EXP2_V2_EPOCH1":
            selected_metrics.append(row)

    weight_files = sorted(
        list(target_model.glob("*.safetensors"))
        + list(target_model.glob("pytorch_model*.bin"))
    )

    hashes = {
        str(p.name): sha256_file(p)
        for p in weight_files
    }

    manifest = {
        "direction": "en_zh",
        "source_lang": "en",
        "target_lang": "zh",
        "version": args.version,
        "model_role": "PAIR_SPECIFIC_SPECIALIST",
        "architecture_family": "OPUS_MT_MARIAN",
        "selected_from": str(source_model.relative_to(root)),
        "frozen_model_path": str(target_model.relative_to(root)),
        "selection_reason": (
            "Exp2 V2 epoch1 selected after Step20 held-out arbitration. "
            "It had the stronger aggregate held-out improvement versus Exp1 "
            "(mean delta BLEU +0.083432, mean delta chrF++ +0.027692) "
            "while passing the no-large-regression gate."
        ),
        "training_recipe": {
            "exp2_version": "exp2_kd_v2",
            "human_weight": 1.0,
            "teacher_kd_weight": 0.5,
            "human_replay_rows": 62990,
            "teacher_kd_rows": 8126,
        },
        "step20_metrics": selected_metrics,
        "weights_sha256": hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN",
    }

    manifest_path = target_model / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nManifest:")
    print(manifest_path)
    print("\nRegistry:")
    print(root / "models/model_registry.json")
    print("No tracked registry file was modified; availability is path-based.")

    print("\nSTATUS:")
    print("EN_ZH_SPECIALIST_V1_FROZEN")


if __name__ == "__main__":
    main()
