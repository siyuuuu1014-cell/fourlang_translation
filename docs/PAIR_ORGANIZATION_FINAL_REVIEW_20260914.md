# 独立模型整理：恢复记录与清理边界

## 结果

原实验记录中的 62 份报告全部在旧 evidence.tar.gz 中找到，JSON 内容与本地留存原记录全部一致。当前仍在原路径的 47 份也与历史一致，缺失的 15 份已恢复。之前“62 对 47”的差异已解释，不再列为报告证据缺口。

此外，从归档对照得到原路径缺失的 559 个文件：

- 487 个位于 `reports/diagnostics/fourlang/`，归四语任务，完全未处理。
- 其余 72 个已保存到项目外独立恢复目录，逐文件记录大小、SHA256 和处理状态。
- 其中 63 个仅在原路径不存在时补回；全部再次读取核对 SHA256 通过，总计 201225984 字节。没有覆盖现有文件，没有删除文件。
- 9 个仅备份，未放回项目：EN-RU 历史配置、人工审核决策配置、3 份旧审查日志、LoRA 冻结脚本及3份涉及共享流程的辅助脚本。其状态见恢复日志。

## 代码验证与共同依赖

前一轮恢复的 `pair_specialist_flow.py` 及对应测试再次运行，5 项测试通过。它们与本地留存副本哈希一致。

本轮补回的 `build_clean_zh_uz_v4.py` 与本地留存文件哈希一致。但本地对应测试尝试在服务器运行时，导入缺失的 `preview_manual_zh_uz_isolation.py` 失败。检查发现该依赖的默认目录指向四语诊断，故停止扩大恢复范围：此脚本目前不宣称可运行，不触发数据构建；交由双方共同依赖核对后处理。

这说明“路径已经恢复”不等于“所有历史构建流程已经可重跑”。独立模型主要训练入口已经验证，涉及共同流程的旧数据构建仍有依赖待核对。

## 资料位置

服务器项目：`/root/autodl-tmp/fourlang_translation`。

服务器独立整理根目录：`/root/autodl-tmp/fourlang_diagnostics_recovery_20260914`。

| 内容 | 独立整理根目录下的位置 |
|---|---|
| 模型—代码—配置—数据—评测绝对路径索引 | `pair_organization/20260914T062318Z/PAIR_ORGANIZATION_INDEX.md` |
| 历史62份报告逐项比对 | `historical_report_recovery_v1/reconciliation.json` |
| 报告缺失情况（恢复前快照） | `historical_report_recovery_v1/HISTORICAL_REPORT_RECONCILIATION.md` |
| 本轮72份归档证据副本 | `pair_evidence_recovery_v1/evidence/` |
| 逐文件恢复/仅备份日志 | `pair_evidence_recovery_v1/recovery_log.json` |
| 前一轮重建的当前快照 | `experiment_archive/20260914T062427Z/` |

本地逐文件恢复日志及15份报告副本在 `reports/diagnostics/historical_report_recovery_v1/`；当前快照的清单及校验文件也已备份本地。前一轮快照反映恢复前状态，本轮 recovery_log 是后续恢复增量，不覆盖修改旧快照。

旧 `file_inventory.json` 和 `archive_checksums.json` 不在原位置，也不包含在 evidence 源文件压缩包中，不能宣称找回。现有新快照与恢复增量提供当前追溯依据，不能证明旧权重备份完整。

## 清理清单结论

| 类别 | 本轮决定 | 理由 |
|---|---|---|
| 六组当前独立模型及旧基线 | 保留 | 当前调用、实验对照需要 |
| 实际训练/验证数据、原始数据、评测与恢复报告 | 保留 | 实验复现与追溯需要 |
| 历史落选模型与专用代码 | 保留 | 未完成使用方核对，不把未选中等同可删 |
| EN-RU 重复源模型 | 保留 | 仍可能通过动态路径访问 |
| 中乌2 epoch、3 epoch的 checkpoints | 暂列候选，不执行 | 共约22.35 GiB；删除会失去优化器/RNG精确续训及检查点对照能力 |
| 四语诊断487项、四语模型与共同依赖 | 本任务不处理 | 另一任务负责，只读核对后统一决策 |

两处检查点候选的精确路径：

`/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints`

`/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints`

之前核对的逻辑大小分别为11998013319、11998018300字节；执行删除前必须重新测量并确认无活跃写入。它们不是已授权删除清单。

本阶段完成独立模型主要代码恢复、实验报告和实际数据恢复及路径索引；未进行全仓库清理。源码仍有未跟踪文件，不执行 git clean 或带删除的同步。后续共同依赖和检查点处置须由一个任务统一执行，避免再次冲突。
