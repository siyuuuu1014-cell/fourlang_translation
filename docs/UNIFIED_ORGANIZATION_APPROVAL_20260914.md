# 四语集合模型 + 六组独立模型：统一待审批清单

**状态：仅供审批，没有删除授权。所有候选 approved=false，execution_permitted=false。**

实时核对时间：2026-09-14 14:57:33（北京时间）。服务器项目：`/root/autodl-tmp/fourlang_translation`。

机器可读清单及证据：[UNIFIED_ORGANIZATION_REVIEW_20260914.json](UNIFIED_ORGANIZATION_REVIEW_20260914.json)。包含逐文件大小、mtime、适用的小文件哈希、代码依赖边、当前路径存在状态、来源文件哈希及需保留的 stash 提交。

## 1. 合并后的四项候选

| 编号 | 所属 | 精确绝对路径 | 逻辑字节 | 约 GiB | 本次状态 |
|---|---|---|---:|---:|---|
| P1 | 独立中乌2 epoch | `/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints` | 11998013319 | 11.174 | 待审批，不执行 |
| P2 | 独立中乌3 epoch | `/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints` | 11998018300 | 11.174 | 待审批，不执行 |
| S1 | 四语 Exp3_v2 中间步500 | `/root/autodl-tmp/fourlang_translation/results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-500` | 7418548049 | 6.909 | 待审批，不执行 |
| S2 | 四语 Exp3_v2 中间步1000 | `/root/autodl-tmp/fourlang_translation/results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1000` | 7418548224 | 6.909 | 待审批，不执行 |

独立组共23996031619字节，四语组共14837096273字节，总计 **38833127892字节，约36.17 GiB**。这是逻辑大小，不承诺释放同等物理磁盘空间；硬链接、文件系统及实际文件变化可能影响结果。

**影响：** 删除会失去这些步骤的权重、优化器、调度器、随机数及缩放器状态，无法原样从对应步骤续训或重新比较该检查点。P1/P2是整个 checkpoints 目录，S1/S2仅是两个叶子目录，严禁把 S1/S2 扩大为四语 checkpoints 父目录。

独立 best_model 导出不在上述路径内。删除前仍须将候选中的训练状态、清单等元数据逐文件备份并校验；现有元数据归档不等于优化器和检查点权重的完整备份。如果希望保留精确续训能力，不应批准删除对应目录。

## 2. 统一保留名单

| 类别 | 决定 |
|---|---|
| 六组独立模型的当前权重、Exp1旧基线、历史对照 | 全部保留；不因本次整理改变默认路由 |
| 四语 Exp1、Exp2、Exp3_v2 的 best_model/shared | 全部保留；本次不重新选型或发布 |
| 四语最终断点1438 | 明确保留，不列入候选总量 |
| 原始数据、双语人工数据、KD数据、已恢复实际训练/验证数据、FLORES基准 | 全部保留；共同依赖不由单个任务清理 |
| 基础模型 `/root/autodl-tmp/models`、环境 `/root/autodl-tmp/venvs` | 保留 |
| 通用训练/推理、语言归一化、安全检查、配置、测试、实验代码与文档 | 全部保留；静态无调用不能证明可删除 |
| 评测结果、原始日志、归档、两个任务的工作目录和本次合并目录 | 保留 |
| Git stash | 本轮只读列出6条提交，全部保留，不pop/apply/drop/clear |
| 临时SSH公钥授权 | 不改动；是否撤销由用户另行决定，不展示密钥内容 |

最终断点保留路径：

`/root/autodl-tmp/fourlang_translation/results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1438`

六组独立模型路径以项目 `configs/specialists/current_pair_models.json` 及本次JSON中的 preserve_explicit_paths 为准。四语是NLLB-600M实验线，六组独立模型是SMaLL-100实验线；两者不可因同一Python环境名或相同语言方向而混为同一模型。

## 3. 时间与缺失项协调

先读四语最新 current_changes.md/current_existence_recheck.json，再读旧盘点。实测与最新交接记录列出的存在状态一致，未将旧missing_files直接当成当前事实。

- `build_clean_zh_uz_v4.py` 旧盘点缺失、本次存在；这是独立任务已补回的文件，不再列为当前缺失。
- 该构建脚本依赖的 `preview_manual_zh_uz_isolation.py` 仍缺失于项目。副本位于独立恢复目录，涉及四语默认诊断路径；本次不恢复、不运行。
- 四语 Exp1/Exp2/Exp3/Exp3_v2 的8个实际训练/验证JSONL仍不在工作目录。四语任务已记录stash对象，Exp3_v2训练行指纹与记录一致；本次仅承接该证据，未再次读取完整数据或恢复stash。
- 部分manifest是旧流程可选探测项；不存在不能直接推断曾经丢失。四语Exp3训练完成情况仍待核实。
- acceptance目录当前缺失，但交接补充确认stash含3个子文件；不能沿用旧清单对目录精确匹配为null的结论。
- 独立侧62份报告已核对，15份缺失报告及其他专用产物已按恢复日志补回。旧归档file_inventory/checksum缺失由新快照与恢复增量补充追溯，不声称旧权重备份完整。
- 四语发布门槛未通过与已训练权重存在是两件事；本次不执行发布、不修调用示例、不作质量排名。

## 4. 审批前与执行前的条件

1. 用户明确选择批准哪些编号及其精确路径，可以只批准某一组；本文件本身不是授权。
2. 确认无需对应步骤的精确续训，候选中元数据已在删除范围外备份并校验。
3. 四语训练数据的stash保留/独立备份方案须先确定；不通过批量stash恢复来解决。继续缺少可用恢复方案时，S1/S2保持待审批。
4. 执行前重新测量文件清单，检查路径、符号链接、活跃写入及保留模型的完整性；本次路径/元数据盘点不等于执行前检查已完成。
5. 只交一个任务执行，禁止并发清理、移动、训练写入或同步删除。禁止git clean、reset --hard、stash恢复和路由变更。
6. 执行后逐项记录处理路径、实际结果、空间变化与可恢复性。当前没有删除工具执行、没有删除命令清单。

## 5. 来源与交付

四语旧快照：`/root/autodl-tmp/fourlang_shared_organization/inventory_20260914_143314`。

四语新补充：`/root/autodl-tmp/fourlang_shared_organization/handoff_20260914_145040`。

独立整理与恢复：`/root/autodl-tmp/fourlang_diagnostics_recovery_20260914`；对应代码已推送GitHub提交 `1f51712`。本次只读查询时服务器HEAD仍为 `4f5dfa0`，不自动pull或覆盖工作区。

本次统一包：`/root/autodl-tmp/fourlang_unified_organization/20260914T065733Z`。

本次仅写入独立合并目录及本地交付文件，没有修改源项目、两个原交接包、Git状态、模型、数据、stash或SSH授权。来源哈希与当前存在状态记录于配套JSON。对权重没有重新做完整哈希或GPU加载；四语原先的哈希核对结论仍注明来自交接包。
