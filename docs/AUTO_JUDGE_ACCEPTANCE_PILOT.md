# 自动盲评 Exp1 / Exp2

该流程使用本地 Qwen3-8B 对 `acceptance_pilot_v2` 的 240 个案例做两遍盲评。
第二遍交换 A/B 位置；只有两遍评分一致的候选才计入比较。运行可断点续跑。

`v2` 评测协议以语义为先，并只在送入裁判模型的副本中统一中文标点；原始
候选译文不会被修改。标点、空格或等价措辞不得单独决定胜负，漏译分句应按
实质错误处理。必须使用新的 `v2` 输出目录，不能复用旧 `v1` 缓存。

它不会训练、写训练数据或修改模型。结果属于自动诊断，不等同母语人工验收。

```bash
cd /root/autodl-tmp/fourlang_translation

nohup env \
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/root/autodl-tmp/venvs/qwen3_judge/bin/python \
scripts/pipeline_v3/auto_judge_acceptance_pilot.py \
--source reports/diagnostics/fourlang/acceptance_pilot_v2 \
--output reports/diagnostics/fourlang/acceptance_pilot_v2_auto_judge_v2 \
--model /root/autodl-tmp/models/Qwen3-8B \
--batch-size 32 \
> fourlang_acceptance_auto_judge_v2.log 2>&1 &

echo $!
tail -n 30 -f fourlang_acceptance_auto_judge_v2.log
```

完成后查看：

```bash
python -m json.tool \
reports/diagnostics/fourlang/acceptance_pilot_v2_auto_judge_v2/summary.json
```

需要人工抽查的失败、不一致及无法判断案例位于：

```text
reports/diagnostics/fourlang/acceptance_pilot_v2_auto_judge_v2/manual_review_packet.json
```

## 第三轮候选独立裁决

如果两遍盲评的一致案例过少，使用下面的流程。它只处理第一阶段中
`winner=UNRESOLVED` 的案例，将 A、B 分开评分，再与前两遍做多数决。

```bash
nohup env \
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/root/autodl-tmp/venvs/qwen3_judge/bin/python \
scripts/pipeline_v3/adjudicate_acceptance_auto_judge.py \
--source reports/diagnostics/fourlang/acceptance_pilot_v2_auto_judge_v2 \
--output reports/diagnostics/fourlang/acceptance_pilot_v2_auto_adjudication_v2 \
--model /root/autodl-tmp/models/Qwen3-8B \
--batch-size 32 \
> fourlang_acceptance_auto_adjudication_v2.log 2>&1 &

echo $!
tail -n 30 -f fourlang_acceptance_auto_adjudication_v2.log
```

最终汇总位于：

```text
reports/diagnostics/fourlang/acceptance_pilot_v2_auto_adjudication_v2/summary.json
```

## Exp3 数据预览（不训练、不写训练集）

语义评测确认后，先检查 Exp3 的 46,000 行定向混合是否能全部不放回采样。
该命令只写一份报告，不生成 `data/multilingual/fourlang/exp3`，也不会加载模型。

```bash
env FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
/root/autodl-tmp/venvs/small100_student/bin/python \
scripts/pipeline_v3/preview_exp3_data.py \
--config configs/multilingual/fourlang.toml

python -m json.tool reports/pipeline/fourlang/exp3_data_preview.json
```

只有状态为 `EXP3_DATA_PREVIEW_READY_NOT_WRITTEN` 且
`sampled_with_replacement` 为空时，才进入单独的版本化数据构建步骤。

## 扩展到每方向 100 条：先做源文预览

下面的步骤只从仓库中已有、未进入四语言训练/KD/验证/FLORES 的人工语料
测试切分里筛选源文。它同时排除旧试点源文，输出每种语言 100 条候选预览；
不会生成正式验收集，也不会运行模型。语料属于公开平行语料，并非真实用户消息。

```bash
env FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
/root/autodl-tmp/venvs/small100_student/bin/python \
scripts/pipeline_v3/preview_acceptance_v3_sources.py \
--config configs/multilingual/fourlang.toml \
--per-language 100

python -m json.tool \
reports/diagnostics/fourlang/acceptance_v3_source_preview/summary.json
```

状态为 `ACCEPTANCE_V3_SOURCE_PREVIEW_READY_NOT_FROZEN` 只表示精确隔离和数量
满足要求。仍需完成近重复检查后，才能冻结为 1,200 个方向任务。
