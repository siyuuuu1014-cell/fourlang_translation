# Exp2 中乌退步诊断：数据追溯与三模型逐句对比

本流程只做诊断，不训练、不改采样比例、不重新聚合数据、不重跑 Qwen，
也不覆盖 Exp1 / Exp2 模型、评测或选模结果。先收集证据，再决定是否需要新的对照实验。

## 要回答的问题

1. 原始 v3 新增中→乌 Teacher 池的 3,771 条 MINOR，在实际 Exp2 采样后占多少条、多少个不同文本对、多少权重？不能直接把原始池比例当成训练比例。
2. 在训练时的固定中乌验证样本上，选定 Teacher 是否优于 Exp1？Exp2 具体改变了哪些译文？
3. 本次重测是否复现原训练日志中的 Exp1 / Exp2 指标？若不能，先排查评测环境与口径。

## 同步代码

新增文件仅有以下三个；现有训练入口、配置、数据及模型不变：

- `scripts/pipeline_v3/diagnose_zh_uz.py`
- `tests/test_zh_uz_diagnostics.py`
- `docs/ZH_UZ_EXP2_DIAGNOSTICS.md`

本地 PowerShell，在项目目录执行（仅提交这三个文件，避免夹带其他改动）：

```powershell
cd D:\dev\projects\fourlang_translation
git add -- scripts/pipeline_v3/diagnose_zh_uz.py tests/test_zh_uz_diagnostics.py docs/ZH_UZ_EXP2_DIAGNOSTICS.md
git diff --cached --name-only
```

确认暂存区没有其他任务的文件，再执行：

```powershell
git commit --only -m "Add resumable zh-uz Exp2 diagnostics" -- scripts/pipeline_v3/diagnose_zh_uz.py tests/test_zh_uz_diagnostics.py docs/ZH_UZ_EXP2_DIAGNOSTICS.md
git push origin main
```

服务器端：

```bash
cd /root/autodl-tmp/fourlang_translation
git pull --ff-only origin main
```

若拉取提示本地修改或分支分歧，先保留现场，不要强制重置。
无需新增依赖、下载模型或重新运行 `aggregate` / `train`。

## 一条命令执行完整诊断

使用原训练环境，默认先统计数据，然后依次加载本地 Teacher、Exp1、Exp2。
每个模型都对相同的两个方向验证子集生成译文；当前应为各 200 条，共 1,200 次翻译。
同一时刻只加载一个模型；复用已有 3.3B Teacher 权重时，采用 FP32 权重 + CUDA FP16 autocast，
与训练时方向验证的精度方式一致，而不是继续训练。

```bash
cd /root/autodl-tmp/fourlang_translation
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
  /root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline_v3/diagnose_zh_uz.py \
  --config configs/multilingual/fourlang.toml \
  >> fourlang_zh_uz_diagnose.log 2>&1 &
echo $!
tail -n 40 -f fourlang_zh_uz_diagnose.log
```

这条命令自动执行整个诊断，无需每个模型手动接续。
每 32 条生成结果原子保存，日志会显示开始和已保存进度。
按 Ctrl+C 退出这里的 `tail` 只停止看日志；后台任务是否仍在运行可用下面的只读检查确认：

```bash
ps -eo pid,etime,pcpu,pmem,args | grep '[d]iagnose_zh_uz.py'
```

如诊断进程确实已退出，重新执行同一条启动命令即可续跑。
已经保存的分段和完整模型方向结果都会复用；未保存的当前分段会重算。
目录锁阻止两个进程同时写同一套结果，不能用它代替检查旧进程是否仍在工作。

模型文件校验阶段会读取并校验全部权重，不占 GPU 计算但可能受磁盘速度影响。
不要因为暂时没有生成进度就再次启动。同一个模型加载或一段翻译期间也可能暂时没有新日志。

## 可选：先仅统计数据，不加载模型

```bash
env FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
  /root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline_v3/diagnose_zh_uz.py --audit-only
python -m json.tool reports/diagnostics/fourlang/exp2_zh_uz/training_audit.json
```

之后直接运行完整诊断命令即可，不需要更换目录。

## 查看结果

```bash
cat reports/diagnostics/fourlang/exp2_zh_uz/summary.md
python -m json.tool reports/diagnostics/fourlang/exp2_zh_uz/metrics.json
python -m json.tool reports/diagnostics/fourlang/exp2_zh_uz/training_audit.json
python -m json.tool reports/diagnostics/fourlang/exp2_zh_uz/done.json
```

|文件|用途|
|---|---|
|`summary.md`|核心指标、已确认 MINOR 比例、解释边界|
|`metrics.json`|三模型各方向 corpus BLEU / chrF2，Exp2−Exp1、Teacher−Exp1，与训练日志之差|
|`comparisons.jsonl`|全部固定验证样本逐句原文、参考、三个译文、句级 chrF2；顺序严格对应|
|`review_samples.jsonl`|每方向固定随机最多 20 条及实际下降最多 20 条，不足则取全部，重叠合并；人工意见字段为空|
|`training_audit.json`|实际训练条数、去重文本对数、重复次数、审核标签、权重及 MINOR 比例上下界|
|`training_sample_trace.jsonl`|实际训练文本对追溯至当前 KD 文件的标签、Teacher ID、来源及权重|
|`teacher_coverage.json`|本次评测的 Teacher ID 与训练中已有 Teacher ID 分布|
|`audit_manifest.json` / `comparison_manifest.json`|输入、源代码、依赖版本、模型文件、精度、解码和分段大小的指纹|
|`chunks/`|各模型各方向的推理断点及实际 tokenizer / generation_config 信息|
|`done.json`|诊断完整结束的标志与输出校验值；PASS 仅表示诊断完成，不表示模型晋级|

输出目录自带忽略规则，避免把包含原文的诊断产物误提交到 Git。

人工标注前请另存副本，避免与可复现生成产物混用：

```bash
cp -n reports/diagnostics/fourlang/exp2_zh_uz/review_samples.jsonl \
      reports/diagnostics/fourlang/exp2_zh_uz/human_review.jsonl
```

重点看译错、漏译、增译、语言/文字混杂、专名与数字错误，不能只看句级分数。
指标下降最多的样本有选择偏差；其错误率不代表所有样本。固定随机部分可作辅助检查，
这 200 条/方向也不能替代全量质量保证。

## 安全与解释边界

- 固定子集从 Exp2 `validation_subset.json` 读取，并与原训练数据、种子和选择规则核对；不使用 FLORES devtest 挑样本、改参数或选模型。
- 核对 `run_manifest.json`、`initial_validation.json`、`finished.json` 的运行指纹；训练数据必须与真实 Exp2 的行内容指纹一致。
- Exp1 文件必须等于 Exp2 记录的起始模型，Exp2 必须等于已完成导出签名。缺文件或签名变化立即停止，不回退预训练模型。
- 当前选定 Teacher 必须与 v3 数据中的 Teacher ID 相符。混有其他历史 Teacher 的数据会在 coverage 报告中列出，不能将其也归因于本次评测的 Teacher。
- 旧版 KD 若未保留审核标签，记为 UNKNOWN；文本相同但匹配记录标签冲突，记为 AMBIGUOUS。不用低权重猜 MINOR，也不把未知标签视为 PASS。
- 元数据来自当前原始 KD 文件；不能声称对缺失的历史标签完成了恢复。已确认 MINOR 比例和含未知标签的可能范围分别给出。
- 权重占比是数据集中权重和的比例，不是精确梯度贡献；训练损失在 batch 内归一化，句长及梯度也会影响实际学习。
- 指标仍沿用项目现有 `CHRF(word_order=2)` 定义。整体 corpus 指标不是逐句指标的算术平均。
- 解码设置来自已完成训练的运行记录，不随当前配置中的新参数改变；记录本次依赖和精度，与旧训练不同时明确警告。
- 默认 32 条保存一次。若要改 `--checkpoint-rows`（必须为 8 的倍数）或改代码/环境/数据/模型，应使用全新的 `--output reports/diagnostics/...`，不能混用旧断点。
- 已有报告或人工修改不会被覆盖。若检测到损坏文件，应保留现场并换输出目录重做，不强行跳过完整性检查。
- Teacher 更高分、MINOR 更高占比，都不能单独证明导致退步的原因。需结合逐句人工核对，必要时另开受控对照实验；本脚本不会自动开训。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_zh_uz_diagnostics.py -q
.\.venv\Scripts\python.exe -m ruff check scripts/pipeline_v3/diagnose_zh_uz.py tests/test_zh_uz_diagnostics.py
```

测试使用合成数据和本地极小模型，不需要服务器权重、真实语料或联网。
