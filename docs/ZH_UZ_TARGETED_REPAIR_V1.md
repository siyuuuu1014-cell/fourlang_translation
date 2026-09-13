# zh→uz 定向修复数据 v1

目标：从 Exp2 的真实场景错误类型出发，构建与110条诊断题不重复的新训练候选数据。该流程不会修改现有训练集，也不会自动开始训练。

## 1. 生成并检查6,000条中文源句

```bash
cd /root/autodl-tmp/fourlang_translation

python scripts/pipeline_v3/build_zh_uz_targeted_sources.py

python -m json.tool \
  data/targeted/zh_uz/v1/source_manifest.json
```

预期为六类各1,000条，总计6,000条，`exact_protected_overlap` 必须为0，`missing_protected_files` 必须为空。服务器拥有完整训练文件，因此应以服务器生成的 manifest 为准；缺少任一受保护数据文件时程序会拒绝继续。

## 2. 使用 MADLAD 生成第一路候选

```bash
nohup env \
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
/root/autodl-tmp/venvs/small100_student/bin/python \
scripts/pipeline_v3/generate_zh_uz_targeted_teachers.py \
  --teacher-id madlad400_3b_mt \
  --family madlad \
  --model /root/autodl-tmp/models/madlad400-3b-mt \
  --output reports/diagnostics/fourlang/zh_uz_targeted_v1_madlad \
> zh_uz_targeted_v1_madlad.log 2>&1 &
```

## 3. 第一条结束后，使用 M2M100 生成第二路候选

```bash
nohup env \
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
/root/autodl-tmp/venvs/small100_student/bin/python \
scripts/pipeline_v3/generate_zh_uz_targeted_teachers.py \
  --teacher-id m2m100_1_2b \
  --family m2m100 \
  --model /root/autodl-tmp/models/m2m100_1.2B \
  --output reports/diagnostics/fourlang/zh_uz_targeted_v1_m2m100 \
> zh_uz_targeted_v1_m2m100.log 2>&1 &
```

同一张GPU不要同时运行两个教师。两个命令都支持按分片安全续跑；使用完全相同的命令即可恢复。

## 4. 双路盲评与保守筛选

先验证输入，不加载Qwen：

```bash
/root/autodl-tmp/venvs/qwen3_judge/bin/python \
scripts/pipeline_v3/judge_zh_uz_targeted_teachers.py \
  --teacher reports/diagnostics/fourlang/zh_uz_targeted_v1_madlad/teacher_candidates.jsonl \
  --teacher reports/diagnostics/fourlang/zh_uz_targeted_v1_m2m100/teacher_candidates.jsonl \
  --preferred-teacher madlad400_3b_mt \
  --prepare-only
```

验证成功后运行两遍顺序反转的Qwen盲评：

```bash
nohup env \
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/root/autodl-tmp/venvs/qwen3_judge/bin/python \
scripts/pipeline_v3/judge_zh_uz_targeted_teachers.py \
  --teacher reports/diagnostics/fourlang/zh_uz_targeted_v1_madlad/teacher_candidates.jsonl \
  --teacher reports/diagnostics/fourlang/zh_uz_targeted_v1_m2m100/teacher_candidates.jsonl \
  --preferred-teacher madlad400_3b_mt \
  --batch-size 32 \
> zh_uz_targeted_v1_judge.log 2>&1 &
```

只保留两遍判断一致、源文可用且至少一个教师达到 `DIRECT` 的样本。MADLAD 与另一个教师同时达到 `DIRECT` 时优先选 MADLAD。

## 5. 查看候选结果

```bash
python -m json.tool \
  reports/diagnostics/fourlang/zh_uz_targeted_v1_judge_v1/summary.json
```

筛选后的候选：

```text
reports/diagnostics/fourlang/zh_uz_targeted_v1_judge_v1/selected_candidates.jsonl
```

这些样本仍标记为 `TRAINING_CANDIDATE_PENDING_HUMAN_SPOTCHECK`。需要按六个类别分别抽查后，才能冻结为训练数据。Qwen两遍判断不是两个独立审核者，也不能替代乌兹别克语母语认证。
