# Exp1 / Exp2 场景小样本诊断

80 条 AI 草稿源文，240 个任务，两个模型共 480 条译文。本流程不是正式质量认证。
源文尚未经母语确认；需另做语言确认及语义近重复检查才能作为正式验收候选。
不会训练、下载模型、覆盖训练集或修改模型文件。

## 服务器

```bash
cd /root/autodl-tmp/fourlang_translation
git pull --ff-only origin main
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline_v3/run_acceptance_pilot.py \
  >> fourlang_acceptance_pilot_v1.log 2>&1 &
echo $!
tail -n 30 -f fourlang_acceptance_pilot_v1.log
```

默认检查 Exp1/Exp2 聚合后的 train.jsonl 和 validation.jsonl，原文、译文两侧均检查。
标准化为 NFKC、大小写折叠、仅保留字母数字；发现重复停止，不删题、不运行推理。
缺失文件也停止。未检查所有原始/KD候选池、FLORES及预训练语料，不承诺零泄漏。
可用重复的 `--extra-corpus 路径` 添加支持标准方向字段的 JSONL/Parquet；不能直接传四列 FLORES 表。
`--check-only` 只查重，完成后去掉参数继续。首次及续跑会核对本地模型权重哈希。

两版统一使用 fourlang.toml 的长度与 beam 参数以及 NLLB 语言代码；复用项目生成后文字标准化。
遇到超过输入长度限制的题目会报错而不是静默截断。生成长度上限仍可能影响译文，审核时需检查结尾完整性。
每批默认 8 条完成即保存。中断后重跑相同命令跳过已完成批次，不是自动重启。
更改题目、数据、模型、代码或参数需要新输出目录：`--output reports/diagnostics/fourlang/acceptance_pilot_v2`。
不要与 Qwen 审核或训练同时占用同一 GPU；先确认其他任务结束。

## 输出与保密

目录：`reports/diagnostics/fourlang/acceptance_pilot_v1/`

- `summary.json`：完成状态及数量，不是质量指标。
- `overlap_report.json`：查重范围、文件哈希与命中。
- **交给审核者**：`blind_review.html`（浏览器查看/打印）、`blind_review.jsonl`（填写等级）。
- **仅组织者保留**：`organizer_predictions.json`、`organizer_only/`、`chunks/`、其他运行元数据。不要把整个目录发给盲评者。

A/B 对每道题随机交换、任务顺序随机，映射保存在 organizer_only/assignment.json。
同一运行重跑保持同一映射。表格采用 DIRECT / EDIT / FAIL / UNJUDGEABLE，不预填分数。
HTML 是只读打印表，不会自动保存输入；电子填写请使用 blind_review.jsonl 的 grade_A/grade_B、errors_A/errors_B、reviewer、notes。
修改人工评分请另存副本，不编辑原始生成文件。评分完成后再做揭盲汇总。
没有人工参考译文，因此不计算 BLEU/chrF，不自动决定上线，不自动发起后续训练。
