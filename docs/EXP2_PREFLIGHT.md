# Exp2 开训前优化与运行说明

本次只改训练代码、配置与测试，不改已完成的 Exp1 权重，也不重跑 Teacher/Qwen。

## 保持不变

- Student 仍从已选定的 Exp1 模型继续训练；本项目当前选择是 NLLB-600M。
- Exp2 目标总量 158,000 条；12 个方向的配额不变。
- 目标混合比例 Teacher KD 60%、人工回放 40%；Teacher 每条在生成的数据集中最多出现 3 次。若 Teacher 不足，按原规则用人工回放补足。
- 最多训练 2 轮，学习率 5e-6，物理 batch 8、梯度累积 4。
- 最终晋级仍要求每个方向 BLEU 与 chrF2 均不低于 Exp1，不自动放宽。

## 已调整

1. **覆盖优先采样**：先完整覆盖不足量的样本池，再均匀重复。重复次数相差最多 1；超过池大小的配额不等于新增独立语料。
2. **全局数据隔离**：在文字规范化后，按语言检查源端、目标端，跨所有语言对排除训练数据与验证/FLORES 的文字重叠。Exp2 验证集还排除已经在 Exp1 训练集中出现过的文字。只修改聚合产物，不修改原始语料。该检查针对文字相等，不证明不存在语义近重复，也不能撤销 Exp1 已经学习过的内容。
3. **等方向选模**：完整验证集仍计算 loss；另从人工验证集每方向固定抽取最多 200 条，所有方向数量相同。开训前测 Exp1 基线，每轮测 BLEU、chrF2、loss 与相对基线变化。按 12 方向等权平均 chrF2 选最佳轮次，不用 FLORES devtest 挑轮次。这里沿用项目现有 `CHRF(word_order=2)` 定义。
4. **安全断点**：每 1,000 个优化器步额外保存，轮末评估后也保存；保留最多 3 个断点。保存模型、优化器、学习率调度器、随机状态、混合精度 scaler 和 Trainer 状态。恢复前核对数据、配置、源模型、代码及依赖版本指纹；只接受完整断点。已完成的相同训练重复运行不会额外训练。
5. **严格加载 Exp1**：缺少权重、分片或 tokenizer 时立即报错，不回退到原始预训练模型。
6. **减少重复损失计算**：保留原先的逐样本加权损失，去掉重复交叉熵计算；已对比数值与梯度。兼容本地 Transformers 4.46.3 的 NLLB 解码输入和新版 PyTorch 的 RNG 安全加载。

方向验证会增加一次开训前评估及每轮的生成评估时间，不能据此承诺整体训练变快或指标必然提高。当前安全恢复仅支持单进程/单 GPU。

## 同步范围

需要一起同步以下文件，不能只同步配置：

- `scripts/pipeline_v2/seq2seq_flow.py`
- `scripts/pipeline_v2/training_safety.py`（新增）
- `scripts/pipeline_v3/fourlang_flow.py`
- `scripts/pipeline_v3/data_safety.py`（新增）
- `configs/multilingual/fourlang.toml`
- `configs/pipelines/fourlang.toml`
- `tests/test_fourlang_pipeline.py`
- `tests/test_training_optimizations.py`（新增）
- 本说明文件

不需要重新下载模型、重新生成 Teacher 或重新做 Qwen 审核。

## 服务器操作顺序

同步上述代码后，在 `/root/autodl-tmp/fourlang_translation` 执行：

```bash
env FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
  /root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline_v3/fourlang_flow.py aggregate \
  --experiment exp2 --config configs/multilingual/fourlang.toml

python -m json.tool reports/pipeline/fourlang/exp2_data.json
wc -l data/multilingual/fourlang/exp2/train.jsonl \
      data/multilingual/fourlang/exp2/validation.jsonl
```

检查 `sampling_strategy=coverage_first_cycles_v1`、`output_rows=158000` 和
`leakage_audit.protected_overlap_after=0`。验证集可能因保护检查而少于旧版的 27,070 条；查看删除计数和各方向可用量。若删除异常多或缺少方向，先查数据，不强行开训。

原来生成的 158,000 条旧文件不能直接沿用，新训练入口会拒绝未经新版审查或已变更的聚合文件。

确认报告后，启动 **仅训练 Exp2**：

```bash
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
  /root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline_v3/fourlang_flow.py train --experiment exp2 \
  --config configs/multilingual/fourlang.toml \
  >> fourlang_exp2_train.log 2>&1 &
tail -f fourlang_exp2_train.log
```

训练任务退出后，重跑同一训练命令会校验并恢复最近的完整断点；不要在旧进程仍运行时再启动一个。断开 `tail` 不等于中断后台训练。数据、配置或代码有变化时会拒绝恢复，需明确保留旧运行并使用独立目录，不能删除指纹文件来绕过检查。

训练结束后另行执行 `evaluate --experiment exp2`；`train` 不会自动完成最终评测、晋级或发布。

## 看哪些报告

- `reports/pipeline/fourlang/exp2_data.json`：采样覆盖、重复上限、混合比例及隔离结果。
- `results/student/fourlang/exp2/checkpoints/shared/initial_validation.json`：同一固定验证子集的 Exp1 初始成绩。
- 同目录 `validation_subset.json`：实际选模样本；`direction_metrics_step_*.json`：各轮各方向成绩与相对基线变化。
- `results/student/fourlang/exp2/train_report.json`：最佳断点、选模指标、初始成绩与宏平均变化。续训时 `train_loss` 沿用 Trainer 的报告口径，不作为跨运行质量比较依据。

## 本地验证

```powershell
./.venv/Scripts/python.exe -m pytest tests -q --disable-warnings
```

覆盖配额/重复率、跨语言对数据重叠、过期产物拒绝、12 方向语言标签、加权损失与梯度、混合精度状态读写、小模型真实训练及中断恢复。CPU 小模型恢复后的最佳权重与不中断参考运行完全一致，并测试了最终导出中断后的恢复。

本地测试不替代服务器 NLLB-600M/V100 的实际训练和最终指标验证。
