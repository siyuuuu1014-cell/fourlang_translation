# 独立模型代码整理交接

## 本轮已完成

- 恢复服务器缺失的 `scripts/pipeline_v3/pair_specialist_flow.py` 和 `tests/test_pair_specialist_flow.py`。来源为原实验归档压缩包，两个文件分别与本地副本 SHA256 完全一致；采用不覆盖已有文件的复制方式。
- 服务器执行 `python -m unittest tests.test_pair_specialist_flow`，5 项测试通过。测试涉及语言对配置、采样、质量门槛和实验阶段约束；没有训练模型。
- 恢复后重新检查模型清单引用的模型目录、代码、配置、训练数据、验证数据和评测报告，缺失引用为 0。该结论仅覆盖清单列出的路径，不代表全仓库所有历史文件完整。
- 使用原路径加总索引组织代码，未大规模移动或重命名文件，避免破坏现有调用。
- 移除使用文档中过时的删除命令示例。旧清理入口继续禁用，不执行删除。

## 从哪里查

- 完整路径总索引：服务器 `/root/autodl-tmp/fourlang_diagnostics_recovery_20260914/pair_organization/20260914T062318Z/PAIR_ORGANIZATION_INDEX.md`。
- 同目录 `inventory.json` 为机器可读清单；本地副本在 `reports/diagnostics/20260914T062318Z/`。
- 六组独立模型运行入口：项目中的 `scripts/pipeline_v3/translate_current_models.py`。
- 模型到代码、数据、评测的映射：`configs/specialists/current_pair_models.json`。
- 实际使用方法：`docs/CURRENT_MODEL_USAGE.md`。
- 历史实验经过和数值：本地保留的 `docs/PAIR_EXPERIMENT_RECORD_20260914.md` 与 `docs/PAIR_EXPERIMENT_ARCHIVE_GUIDE.md`。早期记录反映归档当时状态，并不证明文件现在仍在原位置。

## 代码职责

| 用途 | 代码 / 配置 |
|---|---|
| 六组独立双向模型实验编排 | `scripts/pipeline_v3/pair_specialist_flow.py`；`configs/specialists/six_pair.toml` |
| 中乌消融与 ep3 实验 | `scripts/pipeline_v3/weak_pair_ablation.py`；`configs/specialists/weak_pair_ablation.toml` |
| 英乌既有 Exp2 训练 | `scripts/pipeline/11b_train_student_exp2.py` |
| 源文本筛选、教师数据组装 | `scripts/pipeline_v3/flores_like_data.py` |
| DeepSeek 教师数据流程 | `scripts/pipeline_v3/deepseek_teacher.py` |
| 独立模型推理入口 | `scripts/pipeline_v3/translate_current_models.py` |
| 通用模型加载 / 推理 | `inference/loader.py`、`inference/engine.py`、`src/model_utils.py`，共同依赖，不清理 |
| 通用训练 / 数据辅助 | `scripts/pipeline_v2/seq2seq_flow.py`、`scripts/pipeline_v3/fourlang_flow.py`，共同依赖，不清理 |
| 归档与目录复核 | `scripts/pipeline_v3/archive_pair_experiments.py`、`scripts/pipeline_v3/reconcile_pair_inventory.py` |

上表相对服务器项目根目录 `/root/autodl-tmp/fourlang_translation`。总索引提供各组代码与数据的绝对路径。

## 重建归档及历史缺口

旧归档的 `file_inventory.json` 和校验清单已不在原位置；`evidence.tar.gz` 仍在，且本轮从中成功恢复了两个代码文件。因此未删除旧归档，也不声称旧归档全部丢失或全部完整。

新快照位于项目外：

`/root/autodl-tmp/fourlang_diagnostics_recovery_20260914/experiment_archive/20260914T062427Z`

记录当前 3056 个文件、47 份报告，没有疑似密钥遗漏。早期记录包含 62 份报告，新快照不能替代早期历史记录；保留两者，禁止将减少的报告数解释成全部历史已经恢复。快照只复制代码、元数据和适合复制的小数据，**不是完整模型权重备份**。

`archive_pair_experiments.py` 新增 `--output-root`，后续归档可放到项目外。脚本本轮从独立目录运行，源项目不执行清理。

## 边界和未完成项

- 四语言集合模型及其数据、共享依赖不改动，由另一任务只读整理。两边完成依赖核对前不清理共同文件。
- 当前阶段完成的是独立模型主要代码恢复、路径索引和当前证据快照；尚未执行全仓库废弃代码判定或删除。
- 两个中乌 checkpoint 目录仍保留。不能只因已导出 best_model 就自动删除 optimizer/RNG 断点。
- 不以文件是否受 Git 跟踪判断其是否可删除；本次丢失的训练入口此前即为未跟踪文件。
