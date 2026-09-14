# 专用模型实验归档导读

2026-09-14 已重新连接服务器，读取实际实验产物完成第一份删除前快照。没有删除任何模型、数据或代码。四语集合模型及其依赖保持原位。

## 归档与完整性

服务器项目根目录：`/root/autodl-tmp/fourlang_translation`。

快照目录：`reports/experiment_archive/20260914T053818Z/`。

- `EXPERIMENT_RECORD.md`：实验经过、报告索引、62 份训练和评估报告的完整 JSON 数值摘录。
- `evidence/`：2651 份原始代码、配置、日志、报告和小型数据文件副本。
- `file_inventory.json`：3607 个文件的路径、大小、时间、复制状态及适用的数据 SHA256。
- `evidence.tar.gz`：证据目录压缩包。
- `archive_checksums.json`：主文档、文件清单、来源记录、遗漏记录和压缩包的校验值；已在服务器逐项复验通过。
- 本次检测到的读取期间变化文件为 0，疑似密钥遗漏记录为 0；这不表示运行中的实验已经结束。

本地可阅读 [完整记录](PAIR_EXPERIMENT_RECORD_20260914.md)。其中 evidence 相对链接对应服务器快照目录，本地未复制整个约 1.4GB 的快照。

这份快照不是完整权重备份。超过复制阈值的数据、二进制权重仍在服务器原路径；大型 parquet/jsonl 已在清单中记录哈希。所有训练数据暂时保留。实验流程是根据代码、报告及会话记录重建的，不是完整的逐次终端录屏；缺失历史日志不能补造。

## 当前六组模型

以下均相对于服务器项目根目录，覆盖 12 个方向。

| 方向 | 当前模型 | 原始评估依据 |
|---|---|---|
| en↔zh | `results/student/pair_specialists/en_zh/exp2/best_model/shared` | pair Exp2 门槛 PASS |
| en↔uz | `models/final_specialists/en_uz_small100_v1` | 复用冻结 Exp2，门槛 PASS |
| en↔ru | `models/final_pair_specialists/en_ru_v1` | 已冻结，pair Exp2 门槛 PASS |
| zh↔ru | `results/student/pair_specialists/zh_ru/exp2/best_model/shared` | pair Exp2 门槛 PASS |
| uz↔ru | `results/student/pair_specialists/uz_ru/exp2/best_model/shared` | pair Exp2 门槛 PASS |
| zh↔uz | `results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared` | 后续 dev 胜出、final_devtest PASS；旧 pair Exp2 门槛 FAIL |

PASS 表示超过实验设定的比较门槛，不是人工验证的应用准确率。中英两套旧 Marian 仍被服务器 registry 引用，继续保留。

## 代码与训练数据链

- 专用模型编排：`scripts/pipeline_v3/pair_specialist_flow.py`，配置 `configs/specialists/six_pair.toml`。实际采样后的训练/验证文件在 `data/specialists/<pair>/<exp>/`；原始人工和 KD 路径见配置及文件清单。
- 通用训练、解码及评测：`scripts/pipeline_v2/seq2seq_flow.py`；依赖的 `fourlang_flow.py` 为共享工具，不应因整理专用模型而移除。
- EN-UZ 原始 Exp2：`scripts/pipeline/11b_train_student_exp2.py`；人工数据在 `data/splits/en_uz/v1/`，混合 KD 在 `data/distillation/en_uz/v1/11a_exp2_training/`。
- EN-ZH 人工数据在 `data/splits/zh_en/v1/`，KD 在 `data/distillation/zh_en/v1/18h_exp2_training/`。
- EN-RU、ZH-RU、UZ-RU：人工数据分别在 `data/splits/<pair>/v1/`，KD 分别在 `data/distillation/<pair>/v1/`。
- 中乌消融：`scripts/pipeline_v3/weak_pair_ablation.py`，配置 `configs/specialists/weak_pair_ablation.toml`，采样数据在 `data/experiments/weak_pair_ablation/zh_uz/<variant>/`。
- 中乌当前增量数据：`data/distillation/zh_uz/flores_relaxed_8k/`；源数据含 `data/splits/zh_uz/v2/` 和 `data/distillation/zh_uz/v4/`。这些目录也可能被集合模型读取，继续保护。
- 源筛选与数据组装：`scripts/pipeline_v3/flores_like_data.py`；审查/容量/组装报告在 `reports/experiments/zh_uz_flores_*`。
- 新 DeepSeek 数据：`scripts/pipeline_v3/deepseek_teacher.py`，配置 `configs/directions/zh_uz_deepseek_teacher_v1.toml`，输出 `data/pipeline_v2/zh_uz_deepseek_teacher_v1/`；它是后续数据工作，未替代当前模型。
- 推理入口：`scripts/pipeline_v3/translate_current_models.py`；路径清单 `configs/specialists/current_pair_models.json`。

模型族的训练参数以完整记录里的 train_report 为准。配置是当前文件快照，不能独自证明历史运行时使用了完全相同的值。

## 中乌 FLORES dev 消融结果

每方向 997 条；每格为 BLEU / chrF2，四舍五入到四位小数。

| 变体 | zh→uz | uz→zh |
|---|---:|---:|
| baseline_exp1 | 4.4481 / 31.5975 | 19.2183 / 14.0049 |
| baseline_exp2 | 4.5936 / 31.7329 | 19.4816 / 14.2778 |
| bidir_full | 4.6826 / 31.9993 | 19.4283 / 14.3473 |
| directional_full | 4.6114 / 31.8712 | 19.0815 / 13.9124 |
| full_weighted_60_40 | 4.7790 / 32.1060 | 19.5005 / 14.4050 |
| full_native_lr2e6 | 4.5235 / 31.2532 | 19.2043 / 13.9369 |
| flores_relaxed_8k（2 epoch） | 4.9843 / 32.7875 | 19.7218 / 14.4084 |
| flores_relaxed_8k_ep3 | 5.2018 / 33.3753 | 20.0736 / 14.6009 |

原始数值来自 `results/evaluation/weak_pair_ablation/zh_uz/*.json`，完整记录保留了对应报告。两个 directional_full 是各自独立的单向运行。

ep3 在最终 devtest 上为 zh→uz 5.4873 / 33.6935、uz→zh 20.2646 / 14.5486，每方向 1012 条。旧 pair Exp2 的 FAIL 与后续消融 dev 数值略升并不矛盾，它们不是同一轮评测与门槛。

## 待清理状态

当前删除入口已禁用。旧的 safe_delete 列表只是待复核候选，不代表已经满足删除条件。

需要进一步审查：旧模型是否被其他配置或服务引用；EN-RU 冻结副本与源权重是否一致；checkpoint 是否仍需恢复训练；待删目录中的所有独有数据是否已有完整可用副本。当前目录存在 config/index 文件不足以单独证明模型全部权重可加载。

旧消融报告和 checkpoint 中 trainer_state 等元数据已进入本快照；如后续继续训练，应重新归档发生变化的产物，再重新评估删除范围。历史代码占用较小，保留有助于复现实验，不按“未选中模型”推断其代码可以删除。
