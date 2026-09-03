# FourLang 主翻译流水线

本文只覆盖主翻译系统；`notebooks/`、`scripts/synthetic/`、`data/synthetic/` 均不属于这条流程。

## 环境与存储

本地和服务器使用相同的项目内相对路径。环境差异只写在
`configs/runtime_profiles.toml`：服务器项目位于
`/root/autodl-tmp/fourlang_translation`，Student 与 Judge 使用各自虚拟环境，模型与
Hugging Face 缓存位于 `/root/autodl-tmp`。权重、缓存、候选数据、训练结果和报告不会上传 Git；
只有代码、方向配置和 `models/model_registry.json` 会进入仓库。

## EN-RU 的真实流程

`configs/pipelines/en_ru.toml` 是唯一执行清单，顺序为：

1. 先冻结基准：FLORES dev 只用于选模，FLORES devtest 与 Tatoeba 只用于最终验收；
2. News Commentary 规范化、成对及单边去重，不做未经确认的行数截断；
3. 规则把数据分为 `HARD_REJECT`、`NEEDS_QWEN`、`AUTO_ACCEPT`；
4. Qwen 全审 `NEEDS_QWEN`，并对 `AUTO_ACCEPT` 做固定种子的分层 500 条抽审；
5. 首审 `FAIL/UNCERTAIN` 进入无首审结论提示的独立二审；
6. 依据明确结论构建 GOLD/SILVER/BRONZE approved 数据；
7. 从方向配置指定的质量层中按固定种子取数，再按 pair、英语文本、俄语文本及所有保护基准构建互斥切分；EN-RU Exp1 明确使用 61,216 对 GOLD+SILVER，另留 3,000 对验证和 3,000 对内部测试；
8. 核验 Hub 模型卡许可证和固定 revision，然后在 FLORES dev 上海选；
9. EN→RU 与 RU→EN 分别完成 Student 和 Teacher 实测；EN-RU 根据速度约束显式采用实测过的 SMaLL-100 Student，Teacher 仍按两个方向独立选择；同一多语言 Student 用于双向时训练一个共享双向模型；
10. 对 Student 做全参数 Exp1 人工数据微调，再仅在最终基准上评估；
11. Teacher 双向生成；Qwen 先审计固定 500 条，再全审 Teacher 输出；
12. 只接纳 `PASS + HIGH/MEDIUM usefulness`，按 usefulness 加权并混入保留原质量权重的 human replay；
13. 从 Exp1 继续做全参数 Exp2 序列级蒸馏训练；
14. Exp2 必须在两个方向、每一个最终基准及合并指标上均不低于 Exp1，才能安全冻结并注册为 `ready`。

流程不调用 `train_lora.py`，也不生成 adapter。

EN-RU 在完整 bake-off 后根据用户确认的训练时长约束采用 SMaLL-100 全参数训练。物理 batch 为 16、
梯度累积为 2，有效 batch 为 32；Exp1 为 3 个 epoch、学习率 3e-5，Exp2 为 2 个 epoch、学习率 5e-6。
训练启用动态 padding、按长度分组、4 个数据加载进程及 CUDA fused AdamW。训练开始时会打印真实样本数、
有效 batch 和预计优化步数，避免再次因隐式使用全部 approved 数据而产生数十小时的意外训练。

Qwen Judge 使用方向配置中的批量大小。EN-RU 从 32 开始批量推理；V100 显存不足时按
32→16→8→4→2→1 自动减半。每约 100 条原子写入 Parquet，重新运行时按 `judge_id`
跳过已成功审核的数据。

## 商业许可证门禁

EN-RU 当前候选只包含声明允许商业使用的模型：MADLAD（Apache-2.0）、M2M100（MIT）、
SMaLL-100（MIT）和 OPUS 双向专用模型（Apache-2.0 / CC-BY-4.0）。NLLB 的
CC-BY-NC 许可证不满足商业用途，因此没有进入候选配置。最终冻结的 `model_card.json`
保留获胜模型及许可证信息；带 attribution 要求的模型必须在产品声明中保留归属。

## 运行

先查看阶段：

```bash
/root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline/run_direction.py en_ru --profile server --list
```

确认后从项目根目录运行：

```bash
/root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline/run_direction.py en_ru --profile server
```

状态写入 `.fourlang/pipeline_state/en_ru.json`。只有命令、配置、输入签名、流程代码和有效产物均未变化时才自动跳过；
可使用 `--from`、`--until`、`--only` 和 `--force` 精确控制续跑。

## 新语言方向

保留 `scripts/pipeline_v2/` 不变，新增一份 `configs/directions/<pair>.toml` 和对应 pipeline
清单即可。方向配置必须显式填写：原始语料、FLORES 语言代码、Tatoeba 地址与后缀、
商业合规的 Student/Teacher 候选、数据规模和部署位置。未产生模型比选结果、Judge 策略或
promotion gate 时，后续阶段会停止，不会猜测模型或提前注册。
