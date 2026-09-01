from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/root/autodl-tmp/fourlang_translation").resolve()


def human_size(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def safe_count_rows(path: Path) -> int | None:
    try:
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows)

        if suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                count = sum(1 for _ in f)
            return max(0, count - 1)

        if suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for line in f if line.strip())

    except Exception:
        return None

    return None


def file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
    }

    if not path.exists():
        return info

    if path.is_file():
        info.update(
            {
                "type": "file",
                "size_bytes": path.stat().st_size,
                "size": human_size(path.stat().st_size),
                "rows": safe_count_rows(path),
            }
        )
    elif path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        info.update(
            {
                "type": "directory",
                "files": len(files),
                "total_size_bytes": total,
                "total_size": human_size(total),
            }
        )
    return info


def collect_glob(pattern: str) -> list[Path]:
    return sorted([p for p in PROJECT_ROOT.glob(pattern) if p.exists()])


def show_section(title: str, paths: list[Path]):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    if not paths:
        print("NONE")
        return

    for path in paths:
        relative = path.relative_to(PROJECT_ROOT)

        if path.is_file():
            rows = safe_count_rows(path)
            row_text = f" | rows={rows}" if rows is not None else ""
            print(
                f"[FILE] {relative} | "
                f"{human_size(path.stat().st_size)}"
                f"{row_text}"
            )
        else:
            print(f"[DIR ] {relative}")


def find_model_dirs(root: Path) -> list[Path]:
    found: list[Path] = []

    if not root.exists():
        return found

    for config in root.rglob("config.json"):
        directory = config.parent
        if (
            (directory / "model.safetensors").exists()
            or
            (directory / "pytorch_model.bin").exists()
        ):
            found.append(directory)

    return sorted(set(found))


def main():
    print("=" * 120)
    print("FOURLANG TRANSLATION PIPELINE")
    print("STEP 26A - ZH->EN ASSET INVENTORY")
    print("=" * 120)

    print("\nProject root:")
    print(PROJECT_ROOT)

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(PROJECT_ROOT)

    split_paths = [
        PROJECT_ROOT / "data/splits/zh_en/v1/train_pairs_v1.parquet",
        PROJECT_ROOT / "data/splits/zh_en/v1/validation_pairs_v1.parquet",
    ]
    split_existing = [p for p in split_paths if p.exists()]
    show_section("1. PAIR-LEVEL TRAIN / VALIDATION SPLITS", split_existing)

    benchmark_paths = collect_glob("data/benchmark/zh_en/*")
    show_section("2. FROZEN BENCHMARKS", benchmark_paths)

    exp1_candidates: list[Path] = []
    known_exp1_roots = [
        PROJECT_ROOT / "results/specialists/zh_en",
        PROJECT_ROOT / "results/specialists/en_zh",
    ]
    for candidate_root in known_exp1_roots:
        if candidate_root.exists():
            exp1_candidates.extend(
                [
                    p
                    for p in candidate_root.rglob("*")
                    if p.is_dir()
                    and (
                        "exp1" in p.name.lower()
                        or "human" in p.name.lower()
                    )
                ]
            )
    exp1_candidates = sorted(set(exp1_candidates))
    show_section("3. POSSIBLE EXP1 / HUMAN-REPLAY OUTPUTS", exp1_candidates)

    teacher_selection = [
        p
        for p in collect_glob("data/teacher_selection/zh_en/**/*")
        if p.is_file()
    ]
    show_section(
        "4. TEACHER SELECTION / QWEN ROUTING ARTIFACTS",
        teacher_selection,
    )

    distillation_files = [
        p
        for p in collect_glob("data/distillation/zh_en/**/*")
        if p.is_file()
    ]
    show_section("5. DISTILLATION ARTIFACTS", distillation_files)

    candidate_keywords = (
        "opus",
        "madlad",
        "teacher",
        "qwen",
        "routing",
        "calibration",
        "gate",
        "kd",
    )

    candidate_teacher_files = [
        p
        for p in distillation_files
        if any(keyword in str(p).lower() for keyword in candidate_keywords)
    ]
    show_section("6. TEACHER / QWEN / KD CANDIDATE FILES", candidate_teacher_files)

    exp2_dirs: list[Path] = []
    results_root = PROJECT_ROOT / "results"
    if results_root.exists():
        for p in results_root.rglob("*"):
            if p.is_dir() and "exp2" in p.name.lower():
                text = str(p).lower()
                if "zh_en" in text or "en_zh" in text:
                    exp2_dirs.append(p)

    exp2_dirs = sorted(set(exp2_dirs))
    show_section("7. POSSIBLE EXP2 OUTPUTS", exp2_dirs)

    final_root = PROJECT_ROOT / "models/final_specialists"
    final_models = find_model_dirs(final_root)
    show_section("8. FROZEN DEPLOYMENT MODELS", final_models)

    script_files = collect_glob("scripts/pipeline/zh_en/*.py")
    show_section("9. EXISTING ZH_EN PIPELINE SCRIPTS", script_files)

    assessment = {
        "train_split": (
            PROJECT_ROOT
            / "data/splits/zh_en/v1/train_pairs_v1.parquet"
        ).exists(),
        "validation_split": (
            PROJECT_ROOT
            / "data/splits/zh_en/v1/validation_pairs_v1.parquet"
        ).exists(),
        "flores_benchmark": any(
            "flores" in p.name.lower()
            for p in benchmark_paths
        ),
        "tatoeba_benchmark": any(
            "tatoeba" in p.name.lower()
            for p in benchmark_paths
        ),
        "teacher_selection_artifacts": bool(teacher_selection),
        "distillation_artifacts": bool(distillation_files),
        "possible_exp1_outputs": bool(exp1_candidates),
        "possible_exp2_outputs": bool(exp2_dirs),
        "frozen_final_models_found": [
            str(p.relative_to(PROJECT_ROOT))
            for p in final_models
        ],
    }

    print("\n" + "=" * 120)
    print("10. ASSESSMENT")
    print("=" * 120)
    print(json.dumps(assessment, ensure_ascii=False, indent=2))

    print("\n" + "=" * 120)
    print("11. NEXT-STEP RECOMMENDATION")
    print("=" * 120)

    required_found = (
        assessment["train_split"]
        and assessment["validation_split"]
        and assessment["flores_benchmark"]
        and assessment["tatoeba_benchmark"]
    )

    if not required_found:
        print("STOP: foundational split/benchmark assets are incomplete.")
        print(
            "Do not start ZH->EN training until the missing frozen assets "
            "are resolved."
        )
    elif assessment["possible_exp2_outputs"]:
        print("Existing Exp2-like outputs were found.")
        print(
            "NEXT: inspect the exact ZH->EN Exp1/Exp2 training reports and "
            "generation metrics before rerunning anything."
        )
    elif assessment["possible_exp1_outputs"]:
        print(
            "Existing Exp1/Human outputs were found, but no clear Exp2 "
            "output was detected."
        )
        print(
            "NEXT: verify whether the Exp1 model is truly ZH->EN and, if "
            "valid, resume from teacher/KD stages instead of retraining Exp1."
        )
    else:
        print("No clear ZH->EN Exp1/Exp2 model output was detected.")
        print("NEXT: begin the ZH->EN specialist pipeline from Exp1 Human Replay.")

    output_dir = PROJECT_ROOT / "reports/pipeline_inventory"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "zh_en_inventory_v1.json"

    report = {
        "project_root": str(PROJECT_ROOT),
        "assessment": assessment,
        "split_files": [file_info(p) for p in split_existing],
        "benchmark_files": [file_info(p) for p in benchmark_paths],
        "possible_exp1_dirs": [
            str(p.relative_to(PROJECT_ROOT))
            for p in exp1_candidates
        ],
        "possible_exp2_dirs": [
            str(p.relative_to(PROJECT_ROOT))
            for p in exp2_dirs
        ],
        "final_models": [
            str(p.relative_to(PROJECT_ROOT))
            for p in final_models
        ],
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nReport:")
    print(report_path)

    print("\nSTATUS:")
    print("ZH_EN_INVENTORY_COMPLETE")


if __name__ == "__main__":
    main()
