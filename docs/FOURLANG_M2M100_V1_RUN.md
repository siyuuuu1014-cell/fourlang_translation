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

先仅检查模型与数据：

```bash
cd /root/autodl-tmp/fourlang_translation
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

