"""Build an additive, descriptive navigation view; never move source artifacts.

Only the new --output directory is written. Symlinks are navigation shortcuts,
not copies or read-only protection. No training, downloads, deletion, or routing.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tomllib


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_child(root, relative):
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"Not a relative child: {relative}")
    return root / part


def describe(path, data=False):
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        result["status"] = "MISSING"
        return result
    if path.is_dir():
        result.update(status="PRESENT_DIRECTORY", kind="directory")
        return result
    before = path.stat()
    result.update(status="PRESENT_FILE", kind="file", bytes=before.st_size,
                  mtime_ns=before.st_mtime_ns, sha256=digest(path))
    if data:
        if path.suffix == ".jsonl":
            count, columns, directions = 0, set(), Counter()
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    count += 1
                    columns.update(row)
                    if "src_lang" in row and "tgt_lang" in row:
                        directions[f'{row["src_lang"]}-{row["tgt_lang"]}'] += 1
            result.update(rows=count, columns=sorted(columns), directions=dict(directions))
        elif path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                metadata = pq.ParquetFile(path)
                result.update(rows=metadata.metadata.num_rows, columns=metadata.schema_arrow.names)
            except ImportError:
                result["schema_note"] = "pyarrow unavailable; content hash recorded only"
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed during scan: {path}")
    return result


class Navigation:
    def __init__(self, project, output, links):
        self.project, self.output, self.links = project, output, links
        self.cache, self.items = {}, []

    def add(self, group, alias, source, role, data=False):
        source = Path(source)
        if not source.is_absolute():
            source = safe_child(self.project, source)
        key = (str(source), data)
        if key not in self.cache:
            self.cache[key] = describe(source, data)
        item = dict(self.cache[key], group=group, alias=alias, role=role)
        dest = safe_child(self.output, group + "/" + alias)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.links and source.exists():
            dest.symlink_to(source, target_is_directory=source.is_dir())
            item["navigation_link"] = str(dest)
        self.items.append(item)
        return item

    def card(self, group, title, explanation):
        lines = [f"# {title}", "", explanation, "",
                 "这些是原文件的导航入口，不是副本。不要通过链接编辑或删除原文件；不要在导航目录运行训练脚本。", "",
                 "| 清晰名称 | 用途 | 实测状态 / 行数 | 原绝对路径 |", "|---|---|---|---|"]
        for x in self.items:
            if x["group"] == group:
                state = x["status"] + (f' / {x["rows"]}行' if "rows" in x else "")
                lines.append(f'| `{x["alias"]}` | {x["role"]} | {state} | `{x["path"]}` |')
        path = safe_child(self.output, group + "/README.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(project, output, links=False, backup_root=None):
    project, output = project.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output must be a NEW directory; existing views are not overwritten")
    if output.parent.resolve() != output.parent:
        raise ValueError("Output parent may not resolve through a symlink")
    manifest = read_json(project / "configs/specialists/current_pair_models.json")
    pair_cfg = tomllib.loads((project / "configs/specialists/six_pair.toml").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    nav = Navigation(project, output, links)
    configs = {x["id"]: x for x in pair_cfg["pairs"]}
    for pair in manifest["pairs"]:
        pid = pair["id"]
        group = "01_pair_models/" + pid
        conf = configs[pid]
        nav.add(group, "01_training_entry.py", pair["training_entrypoint"], "训练编排入口；从项目原位置运行")
        if pid != "en_uz":
            nav.add(group, "02_training_config.toml", pair["training_config"], "当前配置；历史参数以原训练报告为准")
        else:
            nav.add(group, "02_reuse_registry_config.toml", pair["training_config"], "仅记录复用关系；旧英乌训练器不读取此配置")
        for field, name, role in (("human_train", "human_source", "人工源数据池"),
                                  ("kd_train", "kd_source", "教师/人工混合输入池；不是自动等同实际训练文件"),
                                  ("validation", "human_validation_source", "原始人工验证集")):
            source = conf[field]
            nav.add(group, f"03_data/{pid}__{name}{Path(source).suffix}", source, role, True)
        if pid == "en_uz":
            actual = [("exp2", conf["kd_train"], conf["validation"])]
            run_root = "results/student/small100/exp2_distillation_v1"
            report = run_root + "/training_report.json"
            explanation = "SMaLL-100双向英乌。复用旧Exp2；11b脚本直接读取Parquet，不能套用通用prepare命令重训。"
        elif pid == "zh_uz":
            stage = "flores_relaxed_8k_ep3"
            prefix = f"data/experiments/weak_pair_ablation/{pid}/{stage}"
            actual = [(stage, prefix + "/train.jsonl", prefix + "/validation.jsonl")]
            run_root = f"results/experiments/weak_pair_ablation/{pid}/{stage}"
            report = run_root + "/train_report.json"
            nav.add(group, "03_data/zh_uz__relaxed_8k__input_pool.jsonl", "data/distillation/zh_uz/flores_relaxed_8k/train.jsonl", "组装后的输入池，prepare再去重/防泄漏后形成实际训练文件", True)
            explanation = "SMaLL-100双向中乌。人工Exp1 → relaxed_8k输入池 → prepare实际数据 → 从Exp1重新训练3轮 → dev选型 → 冻结候选后devtest。不是从2轮模型再续1轮。"
        else:
            actual = [(stage, f"data/specialists/{pid}/{stage}/train.jsonl", f"data/specialists/{pid}/{stage}/validation.jsonl") for stage in ("exp1", "exp2")]
            run_root = f"results/student/pair_specialists/{pid}/exp2"
            report = run_root + "/train_report.json"
            explanation = "SMaLL-100双向专用模型。人工/KD输入池 → prepare归一化、防泄漏、采样 → 实际JSONL → train → 双向evaluate/gate。"
            if pid == "en_ru":
                explanation += " 英俄Exp1复用旧模型；导航中的新exp1准备数据不是旧Exp1实际训练数据的证明。"
        for stage, train, validation in actual:
            nav.add(group, f"03_data/{pid}__{stage}__trainer_input.jsonl" if train.endswith("jsonl") else f"03_data/{pid}__{stage}__trainer_input.parquet", train, "训练器当前代码读取路径；与历史该次运行绑定需核对报告/指纹", True)
            nav.add(group, f"03_data/{pid}__{stage}__trainer_validation{Path(validation).suffix}", validation, "训练器验证输入；不等同FLORES", True)
        nav.add(group, "04_selected_model", pair["model_path"], "当前选用模型导出；不是检查点目录")
        nav.add(group, "05_run_artifacts", run_root, "训练产物原目录；历史文件保留")
        nav.add(group, "06_training_report.json", report, "历史实际训练参数/行数/起始模型证据")
        for index, source in enumerate(pair["evaluation_evidence"], 1):
            nav.add(group, f"07_evaluation/{index:02d}_{Path(source).name}", source, "原评测/选型证据；仅收录，不重新评测")
        nav.card(group, pid + " 独立双向模型", explanation)
        print("Indexed pair " + pid, flush=True)
    for stage in ("exp1", "exp2", "exp3", "exp3_v2"):
        group = "02_shared_fourlang/" + stage
        nav.add(group, "01_training_entry.py", "scripts/pipeline_v3/fourlang_flow.py", "四语独立实验线训练编排")
        nav.add(group, "02_training_config.toml", "configs/multilingual/fourlang.toml", "四语配置；配置值不替代历史报告")
        for split in ("train", "validation"):
            relative = f"data/multilingual/fourlang/{stage}/{split}.jsonl"
            nav.add(group, f"03_data/fourlang__{stage}__{split}.jsonl", relative, "训练器原路径；若缺失，不创建悬空链接或自动恢复", True)
            if backup_root is not None:
                nav.add(group, f"03_data/fourlang__{stage}__{split}__backup_only.jsonl", backup_root / relative, "独立备份；不是已恢复原路径；不代表所有历史训练指纹已匹配", True)
        for alias, source, role in (
            ("04_model", f"results/student/fourlang/{stage}/best_model/shared", "集合模型导出；存在不等于发布通过"),
            ("05_train_report.json", f"results/student/fourlang/{stage}/train_report.json", "该阶段训练完成证据"),
            ("06_evaluation.json", f"results/evaluation/fourlang/{stage}/metrics.json", "集合模型评测")):
            nav.add(group, alias, source, role)
        if stage == "exp3_v2":
            nav.add(group, "07_retained_checkpoint_1438", "results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1438", "保留的完整最终断点")
        nav.card(group, "四语 " + stage, "NLLB-600M集合模型线：基座→Exp1→Exp2→Exp3_v2。Exp3仅有历史数据版本，训练完成仍待核实。不改变另一任务的代码/路由/数据；备份和原路径分别记录。")
    for name in ("flores_dev", "flores_devtest"):
        nav.add("03_common", name + ".parquet", "data/benchmark/fourlang/" + name + ".parquet", "保护基准，不可混入训练", True)
    for alias, source in (("shared_trainer.py", "scripts/pipeline_v2/seq2seq_flow.py"),
                          ("data_normalization.py", "scripts/pipeline_v3/fourlang_flow.py"),
                          ("translation_entry.py", "scripts/pipeline_v3/translate_current_models.py")):
        nav.add("03_common", alias, source, "共同依赖，保留")
    nav.card("03_common", "共同代码和保护基准", "共同文件由多个模型使用，不能按单一模型决定删除。")
    for old in manifest.get("legacy_retained", []):
        nav.add("04_history", old["id"], old["model_path"], "历史Marian单向模型，旧路由仍引用，保留")
    for source in ("results/student/pair_specialists", "results/experiments/weak_pair_ablation", "results/student/small100", "results/student/en_ru"):
        nav.add("04_history", source.replace("/", "__"), source, "完整历史实验目录，非当前模型列表；不移动/删除")
    nav.card("04_history", "历史对照与完整实验目录", "保留原实验名称和报告，不把落选等同可删除；只有上一轮批准的四个检查点目录已删除。")
    for item in manifest.get("active_follow_up", []):
        nav.add("05_follow_up", item["id"] + "__data", item["data_root"], "后续教师数据实验；不是当前已训练模型；进度需单独核实")
        nav.add("05_follow_up", item["id"] + "__code.py", item["code"], "后续流程代码；本次不调用API")
    nav.card("05_follow_up", "后续实验", "不能将教师生成完成当作学生模型训练完成。")
    overview = ["# 项目统一导航", "", "这是新增的分类导航，不是移动后的新训练根目录。原文件名/权重/数据均未改动。",
                "", "链接访问的是原文件，编辑会影响原件。训练脚本仍从项目根目录的原路径运行。",
                "", "- [训练流程](TRAINING_WORKFLOWS.md)", "- [数据与命名规则](DATA_AND_NAMING.md)",
                "- [实时数据目录](DATASETS.md)", "- [缺失项与后续工作](OPEN_ITEMS.md)", ""]
    groups = sorted({x["group"] for x in nav.items})
    overview += [f"- [{group}]({group}/README.md)" for group in groups]
    (output / "README.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    datasets = [x for x in nav.items if "rows" in x or "/03_data/" in x["group"] + "/" + x["alias"] or x["alias"].endswith(".parquet")]
    lines = ["# 数据文件用途与实测状态", "", "别名≠重命名。行数/列名/哈希只证明当前文件内容，不单独证明它属于某次历史训练。详细字段见catalog.json。", "", "| 所属 | 名称 | 角色 | 状态 | 行数 |", "|---|---|---|---|---|"]
    for x in datasets:
        lines.append(f'| {x["group"]} | `{x["alias"]}` | {x["role"]} | {x["status"]} | {x.get("rows", "未读取/缺失")} |')
    (output / "DATASETS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [x for x in nav.items if not x["exists"]]
    lines = ["# 未完成事项", "", "1. 下列缺失项不得用自动重生成代替历史数据；先查备份与训练指纹。", "2. 四语原路径恢复由共同依赖协调后执行；本导航只引用备份。", "3. 历史代码的完整可重跑验证未完成，尤其缺失preview_manual_zh_uz_isolation.py的旧构建链。", "4. 当前参数与历史参数可能不同，实际运行以train_report和指纹为准。", "5. 没有进行语音ASR项目重构、路由迁移、权重搬迁或全仓库改名。", ""]
    lines += [f'- {x["group"]} / {x["alias"]}：`{x["path"]}`' for x in missing]
    (output / "OPEN_ITEMS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in ("TRAINING_WORKFLOWS.md", "DATA_AND_NAMING.md"):
        source = project / "docs/project_structure" / name
        if source.is_file():
            (output / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    catalog = {"schema_version": 1, "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
               "project": str(project), "navigation_root": str(output), "source_mutated": False,
               "links_enabled": links, "records": nav.items,
               "summary": {"references": len(nav.items), "missing_references": len(missing), "data_references": len(datasets)}}
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(catalog["summary"]), flush=True)
    return catalog


def check(output):
    catalog = read_json(output / "catalog.json")
    changes = []
    for item in catalog["records"]:
        path = Path(item["path"])
        if path.exists() != item["exists"]:
            changes.append({"path": str(path), "reason": "existence_changed"})
        elif path.is_file() and "sha256" in item and digest(path) != item["sha256"]:
            changes.append({"path": str(path), "reason": "content_changed"})
        if "navigation_link" in item:
            link = Path(item["navigation_link"])
            if not link.is_symlink() or link.resolve() != path.resolve():
                changes.append({"path": str(link), "reason": "link_changed"})
    print(json.dumps({"changed": changes, "references": len(catalog["records"]), "note": "Unchanged missing entries remain missing; this is not a training readiness check."}))
    return bool(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--links", action="store_true", help="Create Linux navigation symlinks to existing files only")
    parser.add_argument("--backup-root", type=Path)
    args = parser.parse_args()
    if args.action == "check":
        raise SystemExit(check(args.output))
    build(args.project_root, args.output, args.links, args.backup_root)


if __name__ == "__main__":
    main()
