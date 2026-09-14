"""Create an immutable, verified experiment evidence snapshot; never delete sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT = {".json", ".jsonl", ".toml", ".yaml", ".yml", ".md", ".py", ".txt", ".log", ".csv"}
SCOPES = [
    "scripts/pipeline", "scripts/pipeline_v2", "scripts/pipeline_v3",
    "scripts/evaluation", "scripts/model_management", "inference", "configs",
    "docs", "tests", "models/model_registry.json", "models/final_specialists",
    "models/final_pair_specialists", "results/student/pair_specialists",
    "results/student/en_ru", "results/student/small100",
    "results/experiments/weak_pair_ablation", "results/evaluation/pair_specialists",
    "results/evaluation/weak_pair_ablation", "results/evaluation/en_ru",
    "reports/experiments", "reports/pipeline", "data/specialists",
    "data/experiments/weak_pair_ablation", "data/splits", "data/distillation",
    "data/pipeline_v2", "reports/diagnostics", "logs",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "reports/experiment_archive" / stamp
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    records, reports, skipped = [], [], []
    files = set()
    for scope in SCOPES:
        path = root / scope
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file())
    for path in sorted(files):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            skipped.append({"path": rel, "reason": "symlink"})
            continue
        stat = path.stat()
        row = {"path": rel, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        # Copy full small metadata and experiment data, never weight binaries.
        if path.suffix.lower() in TEXT and stat.st_size <= 32 * 1024 * 1024:
            raw = path.read_bytes()
            # Refuse to archive common exposed API credential formats.
            import re
            if re.search(rb"sk-[A-Za-z0-9_-]{20,}", raw):
                row["snapshot"] = "omitted_potential_secret"
                skipped.append({"path": rel, "reason": "potential_secret"})
            else:
                destination = evidence / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                row["sha256"] = hashlib.sha256(raw).hexdigest()
                if digest(destination) != row["sha256"]:
                    raise RuntimeError(f"Snapshot verification failed: {rel}")
                row["snapshot"] = "copied_verified"
                if path.suffix == ".json" and (
                    rel.startswith("results/evaluation/")
                    or path.name == "train_report.json"
                    or rel.startswith("reports/experiments/")
                ):
                    try:
                        reports.append((rel, json.loads(raw)))
                    except (ValueError, UnicodeDecodeError):
                        pass
        else:
            row["snapshot"] = "inventory_only_original_preserved"
            # Hash data files to bind versions, but avoid rereading all weights.
            if path.suffix in {".parquet", ".jsonl"}:
                row["sha256"] = digest(path)
        after = path.stat()
        row["changed_during_read"] = (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns)
        records.append(row)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    save("file_inventory.json", records)
    save("omissions.json", skipped)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    save("provenance.json", {"utc": stamp, "project_root": str(root), "git_head": git.stdout.strip(), "source_mutations": False, "files": len(records), "copied": sum(r.get("snapshot") == "copied_verified" for r in records)})
    lines = ["# 六语言对专用模型实验归档", "", f"归档时间（UTC）：{stamp}。服务器：{root}。", "",
             "本归档保留实验原始证据、实际代码快照、数据版本与存放位置。未删除或改写原文件；四语集合模型不在清理范围。", "",
             "## 阅读与复现", "", "evidence/ 保存代码、配置、训练报告、指标和小型数据文件的原样副本；file_inventory.json 记录所有扫描文件的位置、大小和数据 SHA256。权重及大型文件仍在原位置，清单不等于权重备份。", "",
             "omissions.json 列出疑似密钥等未复制内容；changed_during_read=true 的文件表示采集期间仍在写入，不能作为冻结数据。DeepSeek 全量任务属于后续实验。", "",
             "验证集用于训练选模；FLORES dev 用于消融比较；FLORES devtest 属最终评测。不得把这些指标混在同一排名。不同 tokenizer 的 BLEU、chrF2 不直接互相换算。", "",
             "## 实验经过与选型依据", "",
             "1. 六个语言对分别建立 SMaLL-100 双向专用模型。Exp1 使用人工平行数据，Exp2 在 Exp1 上引入教师蒸馏数据及人工数据回放。EN-UZ 复用既有冻结模型，EN-RU 复用原 Exp1 后训练 Exp2。", "",
             "2. ZH-UZ 进一步比较双向全量、两个单向模型、60/40 权重、较低学习率与 FLORES-like 增量数据。以下原始报告保留各变体实际指标。", "",
             "3. FLORES-like 数据经历源文本审核、Wikipedia 补充、教师翻译与复审，再组成每方向 8000 条增量数据和旧数据回放。2 epoch 与 3 epoch 分别训练；ep3 经 dev 选择后进行 devtest 比较。", "",
             "4. DeepSeek 先完成 400 条 pilot，再规划每方向 9000 条全量生成。它尚不替代当前已训练模型。规则校验不代表完整语义正确率。", "",
             "历史操作中出现源语西里尔字母归一化失败、源配额不足、MINOR 复审容量不足、模型路径错误、进程 Killed、CUDA 版本不兼容与 API 连接中断。经过来自会话记录；不能仅凭 Killed 推断已证实的 OOM 根因。以留存报告/日志为证据，缺失历史日志无法补造。", "",
             "## 训练与评估报告索引", "", "| 报告 | 角色 |", "|---|---|"]
    for rel, payload in reports:
        role = payload.get("benchmark", "训练报告" if rel.endswith("train_report.json") else "原始报告") if isinstance(payload, dict) else "原始报告"
        lines.append(f"| [{rel}](evidence/{rel}) | {role} |")
    lines += ["", "## 原始数值与参数（完整摘录）", ""]
    for rel, payload in reports:
        lines += [f"### {rel}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2), "```", ""]
    (output / "EXPERIMENT_RECORD.md").write_text("\n".join(lines), encoding="utf-8")
    with tarfile.open(output / "evidence.tar.gz", "w:gz") as archive:
        archive.add(evidence, arcname="evidence", recursive=True)
    save("archive_checksums.json", {name: digest(output / name) for name in ["EXPERIMENT_RECORD.md", "file_inventory.json", "provenance.json", "omissions.json", "evidence.tar.gz"]})
    print(json.dumps({"archive": str(output), "files": len(records), "reports": len(reports), "omissions": len(skipped), "deleted": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
