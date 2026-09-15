# M2M100 四语 Student 训练与评测

该流程直接读取已经存在的四语 Exp2 蒸馏数据和 Exp3_v2 定向数据，不重建、修改或覆盖数据。
新产物与历史 NLLB 实验完全分开，写入：

- `results/student/fourlang_m2m100/`
- `results/evaluation/fourlang_m2m100/`

## 固定输入

- 基础模型：`/root/autodl-tmp/models/m2m100_418M`
- KD训练集：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp2/train.jsonl`
- KD验证集：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp2/validation.jsonl`
- 定向训练集：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp3_v2/train.jsonl`
- 定向验证集：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp3_v2/validation.jsonl`
- 最终评测：`/root/autodl-tmp/fourlang_translation/data/benchmark/fourlang/flores_devtest.parquet`

## 服务器运行

在Blackwell服务器上使用新环境时，通过`FOURLANG_STUDENT_PYTHON`指定Python；不要复用旧V100服务器中基于CUDA 12.1安装的虚拟环境。

先仅检查模型与数据：

```bash
cd /root/autodl-tmp/fourlang_translation
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh preflight
```

Blackwell服务器示例：

```bash
export FOURLANG_STUDENT_PYTHON=/root/autodl-tmp/venvs/m2m100_blackwell/bin/python
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh preflight
```

确认显示 `M2M100_PREFLIGHT_READY` 后，在后台执行完整流程：

```bash
cd /root/autodl-tmp/fourlang_translation
nohup bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh run-all \
  > fourlang_m2m100_v1.log 2>&1 &
echo $!
tail -n 50 -f fourlang_m2m100_v1.log
```

流程顺序为：KD训练、KD评测、定向续训、定向评测、两阶段对比。中断后重新执行同一条命令，训练器会从匹配的完整断点恢复。

也可以逐阶段运行：

```bash
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh train-kd
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh eval-kd
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh train-targeted
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh eval-targeted
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh compare
```

## 结果查看

```bash
python -m json.tool results/evaluation/fourlang_m2m100/m2m100_kd_v1/metrics.json
python -m json.tool results/evaluation/fourlang_m2m100/m2m100_targeted_v1/metrics.json
python -m json.tool results/evaluation/fourlang_m2m100/comparison.json
```

## KD v2：追加一轮通用蒸馏训练

该实验不会从原始M2M100重新训练，也不会覆盖KD v1或定向模型。它从：

`/root/autodl-tmp/fourlang_translation/results/student/fourlang_m2m100/m2m100_kd_v1/best_model/shared`

加载已经完成两轮KD的最佳权重，重新初始化优化器和学习率调度器，用Exp2的158,000条训练数据以`5e-6`学习率再训练一轮。新日志中的epoch会从0重新计数，但模型权重不会重置。

完整执行训练、FLORES devtest评测，并与KD v1和定向v1比较：

```bash
cd /root/autodl-tmp/fourlang_translation
nohup bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh run-kd-v2 \
  > fourlang_m2m100_kd_v2.log 2>&1 &
echo $!
tail -n 50 -f fourlang_m2m100_kd_v2.log
```

也可以分阶段执行：

```bash
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh train-kd-v2
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh eval-kd-v2
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh compare-kd-v2
```

新产物独立写入：

```text
results/student/fourlang_m2m100/m2m100_kd_v2/
results/evaluation/fourlang_m2m100/m2m100_kd_v2/metrics.json
results/evaluation/fourlang_m2m100/kd_v2_comparison.json
```

全部完成标志为：

```text
M2M100_KD_V2_COMPARISON_READY
```

## 完整人工基础路线（推荐）

该路线用于公平复刻历史 NLLB 的训练顺序，不覆盖现有 M2M100 实验：

1. 原始 M2M100-418M → Exp1 人工平衡数据，3 epochs，学习率 `3e-5`。
2. 人工基础模型 → Exp2 Teacher KD + 人工回放混合数据，2 epochs，学习率 `5e-6`。
3. KD 模型 → Exp3_v2 定向数据，1 epoch，学习率 `1e-6`。

运行前先执行完整路线专用检查，它会逐项验证 Exp1、Exp2、Exp3_v2、FLORES devtest 和基础模型；任何输入缺失都会在训练前停止：

```bash
cd /root/autodl-tmp/fourlang_translation
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh preflight-human-route
```

完整后台运行：

```bash
cd /root/autodl-tmp/fourlang_translation
nohup bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh run-human-route \
  > fourlang_m2m100_human_route.log 2>&1 &
echo $!
tail -n 50 -f fourlang_m2m100_human_route.log
```

可分阶段执行：

```bash
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh train-human
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh eval-human
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh train-kd-from-human
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh eval-kd-from-human
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh train-targeted-from-human
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh eval-targeted-from-human
bash scripts/pipeline_v3/run_fourlang_m2m100_v1.sh compare-human-route
```

新产物：

```text
results/student/fourlang_m2m100/m2m100_human_v1/
results/student/fourlang_m2m100/m2m100_kd_from_human_v1/
results/student/fourlang_m2m100/m2m100_targeted_from_human_v1/
results/evaluation/fourlang_m2m100/m2m100_human_v1/metrics.json
results/evaluation/fourlang_m2m100/m2m100_kd_from_human_v1/metrics.json
results/evaluation/fourlang_m2m100/m2m100_targeted_from_human_v1/metrics.json
results/evaluation/fourlang_m2m100/human_route_comparison.json
```

全部完成标志：

```text
M2M100_HUMAN_ROUTE_COMPARISON_READY
```
