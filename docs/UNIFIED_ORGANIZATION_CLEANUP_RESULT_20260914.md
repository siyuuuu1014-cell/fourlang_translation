# 统一整理：已批准检查点清理结果

2026-09-14。状态：**COMPLETE**。本次仅执行用户明确批准的 P1、P2、S1、S2，不代表授权继续删除其他实验内容。

## 结果与空间

删除前逐文件核对四个目录与统一审批清单一致，无符号链接；检查未发现训练、复制或清理任务并行运行。先完成独立备份及校验，再执行删除，最后复核保留文件。

| 编号 | 已删除的服务器绝对路径 | 文件数 | 删除前逻辑字节 |
|---|---|---:|---:|
| P1 | `/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints` | 51 | 11998013319 |
| P2 | `/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints` | 52 | 11998018300 |
| S1 | `/root/autodl-tmp/fourlang_translation/results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-500` | 15 | 7418548049 |
| S2 | `/root/autodl-tmp/fourlang_translation/results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1000` | 15 | 7418548224 |

共133个文件，逻辑大小38833127892字节。删除阶段观测可用空间增加 **38833512448字节，约36.17 GiB**。完成后磁盘可用87340400640字节，约81.34 GiB，使用率66%。这是执行时观测值，后续任务会改变磁盘用量。

## 保留与校验

- 六组当前独立模型的导出、四语Exp1/Exp2/Exp3_v2的导出、四语最终断点1438，合计101个文件，删除前后SHA256全部一致。
- Git stash 6条提交前后完全一致；未恢复、弹出或删除stash。
- 不删除最佳模型、旧基线和其他模型权重，不修改训练数据、源码、配置、路由、环境、基础模型或SSH授权。
- 本次没有执行Git清理、重置、拉取或推送，也没有重新训练、推理或评估。文件一致性检查不等于新的模型质量认证。

最终断点保留：`/root/autodl-tmp/fourlang_translation/results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1438`。

## 备份位置与可恢复性

服务器独立留存目录：

`/root/autodl-tmp/fourlang_unified_organization/approved_cleanup_20260914T071530Z`

- `checkpoint_metadata/`：117个检查点元数据及辅助文件；逐个校验源文件与副本SHA256一致。
- `stash_dataset_backup/`：8份四语实际训练/验证数据，从已记录的stash对象只读导出；大小和SHA256与原盘点证据一致。未恢复到项目工作目录，原stash仍保留。
- `receipt.json`：审批来源、精确目标及逐文件清单、备份哈希、保留文件哈希、删除结果、空间变化和stash核对记录。

**没有为删除的检查点权重和优化器状态制作完整备份。不能仅凭本次元数据备份原样恢复这些检查点或精确续训。** 保留的best_model可继续用于原有模型调用；四语1438完整断点未删除。不得把元数据备份称为完整模型备份。

本地回执：[执行前](APPROVED_CLEANUP_PREPARE_RECEIPT_20260914.json)、[执行后](APPROVED_CLEANUP_FINAL_RECEIPT_20260914.json)。历史审批文件保留原始“待审批”状态，本结果及执行回执记录其后的明确授权与操作，不篡改历史快照。

## 整理资料入口与未完成事项

- [统一审批范围和历史证据](UNIFIED_ORGANIZATION_APPROVAL_20260914.md)
- [独立模型代码、数据与历史报告恢复记录](PAIR_ORGANIZATION_FINAL_REVIEW_20260914.md)
- [当前模型使用说明](CURRENT_MODEL_USAGE.md)
- 服务器独立模型路径总索引：`/root/autodl-tmp/fourlang_diagnostics_recovery_20260914/pair_organization/20260914T062318Z/PAIR_ORGANIZATION_INDEX.md`。
- 四语新交接：`/root/autodl-tmp/fourlang_shared_organization/handoff_20260914_145040`；旧盘点：`/root/autodl-tmp/fourlang_shared_organization/inventory_20260914_143314`。旧缺失状态仍须按最新交接解释。

本轮批准清理已结束，不扩大范围。代码尚无已确认可无条件删除项。共同依赖`preview_manual_zh_uz_isolation.py`的原路径恢复、历史构建流程可重跑性、四语工作目录的数据恢复及发布验收仍是独立待处理事项；不宣称整个仓库已完成所有修复。
