# 四语 M2M100 Exp4：代码、流程与训练思路总览

## 1. 文档范围与最终结论

本文整理当前四语整体模型（中文 `zh`、英文 `en`、俄文 `ru`、乌兹别克文 `uz`）从已有数据到最终 Exp4 模型的完整代码链路。

当前确定的整体模型是：

- 基座：`facebook/m2m100_418M`，固定 revision `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`，配置记录许可证为 MIT；
- 主模型：`m2m100_error_replay_v1`（简称 Exp4）；
- 服务器模型绝对路径：`/root/autodl-tmp/fourlang_translation/results/student/fourlang_m2m100/m2m100_error_replay_v1/best_model/shared`；
- 覆盖 4 种语言的 12 个有向翻译方向；
- 训练方式：共享模型、全参数微调，不是 LoRA；
- 可选路由增强：仅 `ru→uz` 加载独立 LoRA，其余 11 个方向仍使用字节不变的 Exp4 主模型；
- NLLB 只作为历史指标参考，不进入最终可商用模型路线。

必须区分：Exp4 主模型、后续 LoRA 候选、Soup 模型融合、解码参数搜索是不同实验。Soup、`zh→uz` LoRA、目标乌兹别克语共享 LoRA、`zh→uz` 解码搜索均未取代 Exp4。

## 2. 总体流程

```text
原始双语/人工平行数据 + 已生成 Teacher 蒸馏数据
                │
                ▼
       Exp1 人工平衡四语数据
                │
                ▼
M2M100-418M ──全参训练 3 epochs──> m2m100_human_v1
                │
                ▼
       Exp2 KD + 人工回放数据
                │
                ▼
     全参训练，最多 3 epochs ──> m2m100_kd_from_human_v1
                │
                ├─────────────── Exp3_v2 定向数据（作为 Exp4 数据来源）
                │
                ▼
用 KD 模型给 Exp1 + Exp2 + Exp3_v2 共 324,000 个物理样本计算 NLL 难度
                │
                ▼
按方向内难度、质量标记、精确重复组和方向总权重重新加权
                │
                ▼
  Exp4 正式数据（保留全部物理行，不删除最简单/最困难样本）
                │
                ▼
从 m2m100_kd_from_human_v1 继续全参训练 1 epoch，LR=2e-6
                │
                ▼
     m2m100_error_replay_v1（最终四语整体主模型）
                │
                ├── FLORES devtest：12 方向，每方向 1,012 条
                └── 可选：ru→uz 专用 LoRA 路由
```

关键设计：Exp4 **没有**从 Exp3_v2 续训模型权重开始。它从 `m2m100_kd_from_human_v1` 最佳模型开始，但把 Exp3_v2 数据与 Exp1、Exp2 一并纳入错误回放数据池。这样吸收定向样本，而不直接继承此前定向续训可能造成的方向回退。

## 3. 目录和输入输出

服务器项目根目录：

```text
/root/autodl-tmp/fourlang_translation
```

基础模型：

```text
/root/autodl-tmp/models/m2m100_418M
```

冻结输入数据：

```text
/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp1/train.jsonl
/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp1/validation.jsonl
/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp2/train.jsonl
/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp2/validation.jsonl
/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp3_v2/train.jsonl
/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp3_v2/validation.jsonl
/root/autodl-tmp/fourlang_translation/data/benchmark/fourlang/flores_devtest.parquet
```

Exp4 正式数据和清单：

```text
/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp4_error_replay_v1/train.jsonl
/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp4_error_replay_v1/manifest.json
```

Exp4 诊断与预览：

```text
/root/autodl-tmp/fourlang_translation/reports/diagnostics/fourlang/m2m100_error_replay_v1/
```

训练和评测产物：

```text
/root/autodl-tmp/fourlang_translation/results/student/fourlang_m2m100/m2m100_human_v1/
/root/autodl-tmp/fourlang_translation/results/student/fourlang_m2m100/m2m100_kd_from_human_v1/
/root/autodl-tmp/fourlang_translation/results/student/fourlang_m2m100/m2m100_error_replay_v1/
/root/autodl-tmp/fourlang_translation/results/evaluation/fourlang_m2m100/m2m100_error_replay_v1/metrics.json
/root/autodl-tmp/fourlang_translation/results/evaluation/fourlang_m2m100/error_replay_v1_comparison.json
```

## 4. 直接参与最终路线的代码

### 4.1 入口与配置

| 文件 | 作用 |
|---|---|
| `configs/multilingual/fourlang_m2m100_v1.toml` | 人工基础、KD、早期定向实验的模型、数据及训练参数 |
| `scripts/pipeline_v3/run_fourlang_m2m100_v1.sh` | 人工基础路线的服务器 Shell 入口 |
| `scripts/pipeline_v3/fourlang_m2m100_student.py` | M2M100 阶段编排、预检、训练、FLORES 评测和阶段对比 |
| `configs/multilingual/fourlang_m2m100_exp4_preview_v1.toml` | Exp4 难度分段、质量标记和候选权重配置 |
| `scripts/pipeline_v3/run_m2m100_error_replay_preview_v1.sh` | Exp4 只读预览入口 |
| `scripts/pipeline_v3/preview_m2m100_error_replay.py` | 数据盘点、逐样本 NLL 打分、难度分段、质量标记、去重权重和方向归一化 |
| `configs/multilingual/fourlang_m2m100_error_replay_v1.toml` | 经用户批准的 Exp4 正式构建与训练参数 |
| `scripts/pipeline_v3/build_m2m100_error_replay_v1.py` | 从已冻结打分分片构建不可静默覆盖的正式 Exp4 数据 |
| `scripts/pipeline_v3/run_m2m100_error_replay_v1.sh` | Exp4 构建、预检、训练、评测、对比入口 |

### 4.2 共用训练、推理和安全代码

| 文件 | 作用 |
|---|---|
| `scripts/pipeline_v2/seq2seq_flow.py` | 模型加载、分词、加权损失、训练器、12 方向验证、生成和 BLEU/chrF2 计算 |
| `scripts/pipeline_v2/training_safety.py` | 文件哈希、模型签名、运行指纹、安全断点和原子 JSON 写入 |
| `scripts/pipeline_v2/common.py` | 项目根目录、TOML 配置读取、JSON 写入等公共能力 |
| `scripts/pipeline_v3/fourlang_flow.py` | 固定四种语言和 12 个有向方向；同时承担上游数据聚合流程 |
| `scripts/pipeline_v3/language_normalization.py` | 训练运行指纹记录的语言规范化实现依赖 |
| `inference/loader.py` | 加载 M2M100 模型及 tokenizer |
| `inference/engine.py` | 设置源/目标语言并执行翻译 |
| `scripts/pipeline_v3/translate_m2m100_fourlang.py` | 交互式四语推理入口；使用时必须显式传入 Exp4 路径，当前文件默认值仍指向较早模型 |

### 4.3 与最终路线相关的测试

```text
tests/test_fourlang_m2m100_student.py
tests/test_preview_m2m100_error_replay.py
tests/test_m2m100_error_replay_v1.py
tests/test_fourlang_pipeline.py
tests/test_training_optimizations.py
```

这些测试覆盖模型路线锁定、配置一致性、权重计算、方向配额、正式构建审批、防覆盖、源模型选择、加权损失和安全恢复等逻辑。服务器上的真实 418M 训练与 FLORES 评测仍是最终验证，单元测试不能替代它。

## 5. 上游数据代码

M2M100 训练脚本消费的是已经存在的 Exp1、Exp2、Exp3_v2 JSONL，不会在开训时重新生成 Teacher 数据。需要追溯数据形成过程时，查看：

| 文件 | 作用 |
|---|---|
| `configs/multilingual/fourlang.toml` | 原始语言对数据源、Exp1/Exp2/Exp3_v2 配额及历史训练设置 |
| `scripts/pipeline_v3/fourlang_flow.py` | 统一字段、语言规范化、数据隔离、Exp1/Exp2 聚合与平衡 |
| `scripts/pipeline_v3/data_safety.py` | 训练/验证/FLORES 文本重叠保护 |
| `scripts/pipeline_v3/preview_exp3_data.py` | Exp3/Exp3_v2 只读预览和配额检查 |
| `scripts/pipeline_v3/build_exp3_data.py` | 经确认后构建版本化 Exp3/Exp3_v2 数据 |

Exp1 是每方向平衡的人工平行数据基础；Exp2 是 Teacher KD 与人工回放混合数据，并加强低资源方向；Exp3_v2 是基于阶段诊断构建的定向数据。Exp4 不假定旧记录仍正确，而是读取现存文件、计算哈希并重新逐样本打分。

## 6. 基础训练路线

### 6.1 阶段一：人工数据基础模型

输入：

```text
模型：/root/autodl-tmp/models/m2m100_418M
数据：data/multilingual/fourlang/exp1/train.jsonl
验证：data/multilingual/fourlang/exp1/validation.jsonl
```

参数：

```text
epochs=3
learning_rate=3e-5
batch_size=8
gradient_accumulation_steps=4
effective_batch_size=32
weight_decay=0.01
warmup_ratio=0.05
max_source_length=256
max_target_length=256
optimizer=adamw_torch_fused
```

输出：

```text
results/student/fourlang_m2m100/m2m100_human_v1/best_model/shared
```

目的：先让 M2M100 建立四语、12 方向的人工语料基础，避免直接用蒸馏数据塑造全部行为。

### 6.2 阶段二：从人工基础继续 KD

输入模型：

```text
results/student/fourlang_m2m100/m2m100_human_v1/best_model/shared
```

训练数据：

```text
data/multilingual/fourlang/exp2/train.jsonl
```

参数：

```text
最多 epochs=3
learning_rate=1e-5
early_stopping_patience=2
batch_size=8
gradient_accumulation_steps=4
effective_batch_size=32
```

每轮在固定人工验证子集上对 12 个方向分别生成评测，每方向最多 200 条。选模指标是 12 方向等权 `eval_macro_chrf2`，而不是总 loss，也不使用 FLORES devtest 挑模型。连续两轮没有改善时早停。

输出：

```text
results/student/fourlang_m2m100/m2m100_kd_from_human_v1/best_model/shared
```

这也是 Exp4 的源模型。

### 6.3 Exp3_v2 的位置

仓库曾做过：

```text
m2m100_kd_from_human_v1 → Exp3_v2 → m2m100_targeted_from_human_v1/v2
```

这些定向续训是对照实验，不是最终 Exp4 的权重起点。Exp4 只把 `exp3_v2/train.jsonl` 作为重要的数据来源，与 Exp1、Exp2 统一重新打分和加权。

## 7. Exp4 数据重构逻辑

### 7.1 只读盘点和打分

`preview_m2m100_error_replay.py` 读取三个训练集：

```text
exp1/train.jsonl
exp2/train.jsonl
exp3_v2/train.jsonl
```

共保留 324,000 个物理记录。每条记录生成：

- 含来源文件和行号的稳定 `record_id`；
- 精确规范化重复组 `duplicate_group_id`；
- 当前 KD 模型下的 teacher-forced 平均 token NLL，即 `difficulty_nll`；
- 原始样本、方向和来源信息。

打分以分片形式写入，分片有 SHA-256 完整性记录，可以恢复和复用。预览脚本不能写正式训练集，也不能开始训练。

### 7.2 每方向独立难度排序

每个翻译方向内部按 NLL 从低到高排序：

| 分位区间 | 名称 | 倍率 | 含义 |
|---|---:|---:|---|
| 0–30% | easy | 0.65 | 当前模型已经较会，降低但不删除 |
| 30–80% | medium | 1.00 | 主体训练数据 |
| 80–95% | hard | 1.35 | 较难且值得重点学习 |
| 95–100% | extreme | 0.85 | 可能非常难，也可能含噪，保留但谨慎降权 |

难度只表示“模型当前觉得难”，不直接等于质量好坏。

### 7.3 独立质量风险标记

自动检查四类可解释风险：

- `source_target_identical`：源文和目标文相同；
- `length_ratio_outlier`：方向内稳健长度比异常；
- `target_script_mismatch`：目标文字脚本与目标语言不符；
- `high_cross_language_token_overlap`：跨语言异常高词项重叠。

权重：

```text
无标记：1.00
1 个标记：0.65
多个标记：0.40
```

自动标记只降权和提供抽查线索，不删除记录。

### 7.4 重复样本处理

源文、目标文、语言方向在 NFKC、空白归一和 casefold 后完全相同的记录归为一个精确重复组。每条物理记录仍然保留，但单条乘以：

```text
duplicate_multiplier = 1 / duplicate_group_size
```

因此重复组总体贡献约等于一条独立记录，避免重复数据通过出现次数放大影响。

### 7.5 方向权重归一化

先计算：

```text
preliminary_weight = original_weight
                   × difficulty_band_multiplier
                   × quality_multiplier
                   × duplicate_multiplier
```

再按方向乘以归一化系数，使 11 个普通方向的总有效权重完全相等，`zh→uz` 总有效权重恰好为普通方向的 1.10 倍：

```text
final_weight = preliminary_weight × direction_mass_normalization_multiplier
```

这实现了“`zh→uz` 稍微优先，但其他方向同样重要”。所有最终权重都必须为有限正数，所以全部物理样本在一轮训练中都会被访问。

### 7.6 审批与防误操作

正式构建前必须满足：

- Exp4 配置明确标记 `approval.status=USER_APPROVED`；
- 批准参数与训练参数完全一致；
- 预览状态和打分分片完整；
- 预览参数在批准后没有变化；
- Exp2 验证集与训练池没有精确重复泄漏；
- 源模型签名、配置和文件哈希一致；
- 至少 16 GiB 可用磁盘。

正式数据采用原子写入和内容寻址。若目标文件已存在，只有签名和哈希完全一致时才复用，否则拒绝覆盖。

## 8. Exp4 训练实现

训练起点：

```text
results/student/fourlang_m2m100/m2m100_kd_from_human_v1/best_model/shared
```

参数：

```text
epochs=1
learning_rate=2e-6
batch_size=8
gradient_accumulation_steps=4
effective_batch_size=32
weight_decay=0.01
warmup_ratio=0.05
max_source_length=256
max_target_length=256
optimizer=adamw_torch_fused
save_total_limit=1
direction_validation_samples=200
minimum_free_disk_gb=16
```

这是共享模型的全参数微调。`WeightedTrainer.compute_loss()` 先计算每条样本的平均 token 交叉熵，再按数据中的 `weight` 做加权平均：

```text
sample_loss_i = 有效目标 token 的平均交叉熵
batch_loss = Σ(sample_loss_i × weight_i) / Σ(weight_i)
```

训练器会：

- 在开训前保存固定的 12 方向验证子集并评测源模型；
- 每轮评测各方向 BLEU、chrF2 和 loss；
- 按宏平均 chrF2 选择最佳模型；
- 保存模型、优化器、调度器、随机状态、AMP scaler 和 Trainer 状态；
- 用数据、配置、代码、依赖版本和源模型共同生成运行指纹；
- 仅恢复与运行指纹一致的完整断点；
- 完成后重复运行时复用已导出的模型，不重复训练；
- 当前安全训练明确限制单进程、单 GPU。

## 9. 最终评测

最终评测使用固定 FLORES devtest，每个方向 1,012 条，不能用于训练选模。解码默认 `num_beams=5`、`max_new_tokens=256`。

服务器实际得到的 Exp4 指标如下：

| 方向 | BLEU | chrF2 |
|---|---:|---:|
| en→zh | 33.9437 | 22.5157 |
| en→uz | 13.1531 | 44.7403 |
| en→ru | 22.4906 | 48.2810 |
| zh→en | 20.5383 | 48.5209 |
| zh→uz | 8.0494 | 38.2328 |
| zh→ru | 12.9141 | 39.0081 |
| uz→en | 23.2504 | 49.7243 |
| uz→zh | 24.4642 | 17.0882 |
| uz→ru | 14.7809 | 40.2492 |
| ru→en | 27.6602 | 53.2960 |
| ru→zh | 28.0837 | 19.3771 |
| ru→uz | 9.5103 | 39.5615 |

汇总：

```text
Macro BLEU  = 19.9032
Macro chrF2 = 38.3829
Worst chrF2 = 17.0882（uz→zh）
```

历史 NLLB Exp3_v2 参考值为 Macro BLEU `24.2029`、Macro chrF2 `42.5825`，但 NLLB 路线因许可证/商用约束不进入最终方案。M2M100 基座的 MIT 许可不自动解决训练数据与 Teacher 输出的许可问题，正式商用仍需单独审查数据来源和 Teacher 条款。

## 10. 后续实验与最终采用情况

### 10.1 采用：`ru→uz` 独立 LoRA 路由

代码：

```text
configs/multilingual/fourlang_m2m100_ru_uz_lora_v1.toml
scripts/pipeline_v3/fourlang_m2m100_ru_uz_lora.py
scripts/pipeline_v3/run_fourlang_m2m100_ru_uz_lora_v1.sh
tests/test_fourlang_m2m100_ru_uz_lora.py
```

设置：`r=16`、`alpha=32`、`dropout=0.05`、目标模块 `q_proj/v_proj`、1 epoch、LR `5e-5`，约 236 万个可训练参数，adapter 约 9 MB。

路由后 `ru→uz` 为 BLEU `9.6637`、chrF2 `40.3510`；组合系统 Macro BLEU `19.9160`、Macro chrF2 `38.4487`，通过既定门槛。它不修改 Exp4，只在运行时针对 `ru→uz` 加载 adapter。

### 10.2 未采用的候选

- `zh→uz` 单方向 LoRA：提升未达到门槛；
- `en→uz + zh→uz` 共享目标-Uz LoRA：两个方向提升均不足；
- `zh→uz` beam/length-penalty 搜索：验证集选择后未带来固定测试集改进；
- Exp4 与其他模型的 25%/50%/75% Soup：没有候选通过门槛。

这些代码用于保留实验可复现性，不属于 Exp4 主训练必需代码：

```text
scripts/pipeline_v3/fourlang_m2m100_soup.py
scripts/pipeline_v3/fourlang_m2m100_ru_uz_lora.py
scripts/pipeline_v3/fourlang_m2m100_target_uz_lora.py
scripts/pipeline_v3/run_m2m100_zh_uz_decode_calibration_v1.sh
scripts/pipeline_v3/run_fourlang_m2m100_zh_uz_lora_v1.sh
scripts/pipeline_v3/run_fourlang_m2m100_target_uz_lora_v1.sh
scripts/pipeline_v3/run_fourlang_m2m100_soup_v1.sh
```

## 11. 服务器复现顺序

### 11.1 已有冻结数据时，从人工基础路线开始

```bash
cd /root/autodl-tmp/fourlang_translation
export FOURLANG_MODEL_ROOT=/root/autodl-tmp/models

bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh preflight-human-route

nohup bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh run-human-route \
  > fourlang_m2m100_human_route.log 2>&1 &
```

注意：`run-human-route` 为保留完整历史对照，还会继续训练和评测
`m2m100_targeted_from_human_v1`。Exp4 本身只依赖其中已经完成的
`m2m100_kd_from_human_v1`。如果仅复现最终 Exp4 且不需要重做该历史对照，
可依次运行 `train-human`、`eval-human`、`train-kd-from-human`、
`eval-kd-from-human`，然后进入 Exp4 预览。

### 11.2 Exp4 只读打分与预览

```bash
cd /root/autodl-tmp/fourlang_translation

nohup bash scripts/pipeline_v3/run_m2m100_error_replay_preview_v1.sh run-preview \
  > fourlang_m2m100_error_replay_preview_v1.log 2>&1 &
```

检查：

```text
reports/diagnostics/fourlang/m2m100_error_replay_v1/preview.json
reports/diagnostics/fourlang/m2m100_error_replay_v1/review_samples.jsonl
reports/diagnostics/fourlang/m2m100_error_replay_v1/score_manifest.json
```

### 11.3 用户确认参数后构建和训练 Exp4

```bash
cd /root/autodl-tmp/fourlang_translation

bash scripts/pipeline_v3/run_m2m100_error_replay_v1.sh build
bash scripts/pipeline_v3/run_m2m100_error_replay_v1.sh preflight

nohup bash scripts/pipeline_v3/run_m2m100_error_replay_v1.sh run-all \
  > fourlang_m2m100_error_replay_v1.log 2>&1 &
```

`run-all` 会再次校验并复用相同正式数据，然后依次完成预检、训练、评测和对比。不要同时启动第二个相同训练进程。

## 12. 推理调用

当前交互脚本的默认模型仍是旧的 `m2m100_targeted_v1`，因此调用最终模型时必须显式指定：

```bash
cd /root/autodl-tmp/fourlang_translation
PY=/root/autodl-tmp/venvs/small100_student/bin/python

"$PY" scripts/pipeline_v3/translate_m2m100_fourlang.py \
  --model /root/autodl-tmp/fourlang_translation/results/student/fourlang_m2m100/m2m100_error_replay_v1/best_model/shared \
  --direction zh-uz \
  --text "今天天气很好。"
```

交互运行时，模型只加载一次，可以用 `zh-uz`、`ru-en` 等命令切换 12 个方向。

如要使用最终推荐的路由系统，`ru→uz` 应通过 `run_fourlang_m2m100_ru_uz_lora_v1.sh translate` 或后续统一路由器加载 LoRA；其余方向直接使用 Exp4。

## 13. 尚未完成或必须注意的事项

1. 现有移动端 ONNX 包来自较早的 `m2m100_targeted_from_human_v1`，不是最终 Exp4，不能当作 Exp4 发布包。
2. 需要重新从 Exp4 导出 ONNX/量化模型，并决定手机端如何条件加载 `ru→uz` LoRA。
3. `translate_m2m100_fourlang.py` 的默认模型路径尚未切换到 Exp4；在修改并验证前必须显式传 `--model`。
4. 自动比较报告只提供实验决策证据，不会自动覆盖、删除或晋级旧模型。
5. 训练数据、基础模型、模型权重和大型评测产物通常不适合通过普通 Git 传输；Git 主要同步代码、配置、测试和文档。
6. 不要删除 Exp1、Exp2、Exp3_v2、KD-from-human、Exp4 数据清单或模型签名文件；它们是复现与审计链的一部分。

## 14. 一句话概括训练思路

先用人工平衡数据建立可靠四语基础，再用 Teacher 蒸馏数据扩大能力；随后不用盲目追加定向 epoch，而是让当前模型对所有历史训练样本重新判定难度，结合质量风险、重复度和方向公平性生成连续样本权重，最后从稳定的 KD 模型以极低学习率完成一轮全参数错误回放，并只对确有收益的单一弱方向使用隔离 LoRA 路由。
