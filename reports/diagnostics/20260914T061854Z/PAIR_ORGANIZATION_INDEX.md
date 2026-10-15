# 独立模型整理总索引（清理后复核）

项目：`/root/autodl-tmp/fourlang_translation`。

本次只读盘点，无移动、删除或模型质量测试。四语言集合模型不纳入独立模型的清理范围。

## 六组模型及复现依赖

### en_zh

| 用途 | 绝对路径 | 存在 |
|---|---|---|
| model_path | `/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/en_zh/exp2/best_model/shared` | 是 |
| training_entrypoint | `/root/autodl-tmp/fourlang_translation/scripts/pipeline_v3/pair_specialist_flow.py` | 否，需核查 |
| training_config | `/root/autodl-tmp/fourlang_translation/configs/specialists/six_pair.toml` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/splits/zh_en/v1/train_bidirectional_v1.parquet` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/distillation/zh_en/v1/18h_exp2_training/exp2_train_combined_v1.parquet` | 是 |
| validation_data | `/root/autodl-tmp/fourlang_translation/data/splits/zh_en/v1/validation_bidirectional_v1.parquet` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/en_zh/exp2.json` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/en_zh/promotion_gate.json` | 是 |

### en_uz

| 用途 | 绝对路径 | 存在 |
|---|---|---|
| model_path | `/root/autodl-tmp/fourlang_translation/models/final_specialists/en_uz_small100_v1` | 是 |
| training_entrypoint | `/root/autodl-tmp/fourlang_translation/scripts/pipeline/11b_train_student_exp2.py` | 是 |
| training_config | `/root/autodl-tmp/fourlang_translation/configs/specialists/six_pair.toml` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/splits/en_uz/v1/train_exp1_bidirectional_v1.parquet` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/distillation/en_uz/v1/11a_exp2_training/exp2_train_combined_v1.parquet` | 是 |
| validation_data | `/root/autodl-tmp/fourlang_translation/data/splits/en_uz/v1/validation_exp1_bidirectional_v1.parquet` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/en_uz/exp2.json` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/en_uz/promotion_gate.json` | 是 |

### en_ru

| 用途 | 绝对路径 | 存在 |
|---|---|---|
| model_path | `/root/autodl-tmp/fourlang_translation/models/final_pair_specialists/en_ru_v1` | 是 |
| training_entrypoint | `/root/autodl-tmp/fourlang_translation/scripts/pipeline_v3/pair_specialist_flow.py` | 否，需核查 |
| training_config | `/root/autodl-tmp/fourlang_translation/configs/specialists/six_pair.toml` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/splits/en_ru/v1/train.jsonl` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/distillation/en_ru/v1/train.jsonl` | 是 |
| validation_data | `/root/autodl-tmp/fourlang_translation/data/splits/en_ru/v1/validation.jsonl` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/en_ru/exp2.json` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/en_ru/promotion_gate.json` | 是 |

### zh_uz

| 用途 | 绝对路径 | 存在 |
|---|---|---|
| model_path | `/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared` | 是 |
| training_entrypoint | `/root/autodl-tmp/fourlang_translation/scripts/pipeline_v3/weak_pair_ablation.py` | 是 |
| training_config | `/root/autodl-tmp/fourlang_translation/configs/specialists/weak_pair_ablation.toml` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/distillation/zh_uz/flores_relaxed_8k/train.jsonl` | 是 |
| validation_data | `/root/autodl-tmp/fourlang_translation/data/distillation/zh_uz/flores_relaxed_8k/validation.jsonl` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/weak_pair_ablation/zh_uz/comparison.json` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/weak_pair_ablation/zh_uz/final_devtest.json` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/train_report.json` | 是 |
| source_data_to_preserve | `/root/autodl-tmp/fourlang_translation/data/splits/zh_uz/v2/train.jsonl` | 是 |
| source_data_to_preserve | `/root/autodl-tmp/fourlang_translation/data/distillation/zh_uz/v4/train.jsonl` | 是 |

### zh_ru

| 用途 | 绝对路径 | 存在 |
|---|---|---|
| model_path | `/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_ru/exp2/best_model/shared` | 是 |
| training_entrypoint | `/root/autodl-tmp/fourlang_translation/scripts/pipeline_v3/pair_specialist_flow.py` | 否，需核查 |
| training_config | `/root/autodl-tmp/fourlang_translation/configs/specialists/six_pair.toml` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/splits/zh_ru/v1/train.jsonl` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/distillation/zh_ru/v1/train.jsonl` | 是 |
| validation_data | `/root/autodl-tmp/fourlang_translation/data/splits/zh_ru/v1/validation.jsonl` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/zh_ru/exp2.json` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/zh_ru/promotion_gate.json` | 是 |

### uz_ru

| 用途 | 绝对路径 | 存在 |
|---|---|---|
| model_path | `/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/uz_ru/exp2/best_model/shared` | 是 |
| training_entrypoint | `/root/autodl-tmp/fourlang_translation/scripts/pipeline_v3/pair_specialist_flow.py` | 否，需核查 |
| training_config | `/root/autodl-tmp/fourlang_translation/configs/specialists/six_pair.toml` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/splits/uz_ru/v1/train.jsonl` | 是 |
| training_data | `/root/autodl-tmp/fourlang_translation/data/distillation/uz_ru/v1/train.jsonl` | 是 |
| validation_data | `/root/autodl-tmp/fourlang_translation/data/splits/uz_ru/v1/validation.jsonl` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/uz_ru/exp2.json` | 是 |
| evaluation_evidence | `/root/autodl-tmp/fourlang_translation/results/evaluation/pair_specialists/uz_ru/promotion_gate.json` | 是 |

## 历史与协调

历史证据快照：`/root/autodl-tmp/fourlang_translation/reports/experiment_archive/20260914T053818Z`。
实验经过见该目录 EXPERIMENT_RECORD.md；归档是元数据/小文件证据，不是完整模型权重备份。
警告：历史 file_inventory.json 已不在原位置；无法计算清理前后差异。缺失数组为空不代表没有文件丢失。旧归档不能继续认定完整；删除暂停。
共享代码、原始双语数据和基础模型均按共同依赖保留，不由任一任务单方面清理。
四语任务仅写自己的独立整理目录；两个任务均不得执行全项目清理、同步删除或覆盖原归档。

## 原检查点候选当前状态

- `/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints`：exists=True，逻辑字节=11998013319。
- `/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints`：exists=True，逻辑字节=11998018300。

## 使用入口

统一入口：`/root/autodl-tmp/fourlang_translation/scripts/pipeline_v3/translate_current_models.py`。
调用说明：`/root/autodl-tmp/fourlang_translation/docs/CURRENT_MODEL_USAGE.md`。
保持原路径，使用索引整理逻辑关系，避免移动目录破坏脚本。待双方依赖盘点完成，再合并一份删除清单执行。