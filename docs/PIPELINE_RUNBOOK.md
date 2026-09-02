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

1. News Commentary 规范化、去重、规则过滤，产物仍叫 candidate；
2. Qwen 对 500 条确定性样本打分并冻结该方向阈值；
3. Qwen 审核全部 70,000 条候选，只有达到阈值的数据才能进入 approved；
4. 按 pair_id 构建互斥的 train/validation/test；
5. 下载并冻结 FLORES devtest 与 Tatoeba 测试集，严禁进入训练和 KD；
6. 在相同测试集上比较全部商业合规 Student，按“双向最差 chrF++”优先自动选择；
7. 比较全部商业合规 Teacher，使用相同规则自动选择；
8. 对选中的 Student 做全参数 Exp1 人工数据微调；
9. Teacher 双向生成，Qwen 再校准并审核 Teacher 输出；
10. approved Teacher 数据与 human replay 组成带权 KD 数据；
11. 从 Exp1 继续做全参数 Exp2 蒸馏训练；
12. Exp2 必须在两个方向同时不低于 Exp1，才能冻结并把注册状态改为 `ready`。

流程不调用 `train_lora.py`，也不生成 adapter。

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

状态写入 `.fourlang/pipeline_state/en_ru.json`。相同命令、配置和产物均存在时自动跳过；
可使用 `--from`、`--until`、`--only` 和 `--force` 精确控制续跑。

## 新语言方向

保留 `scripts/pipeline_v2/` 不变，新增一份 `configs/directions/<pair>.toml` 和对应 pipeline
清单即可。方向配置必须显式填写：原始语料、FLORES 语言代码、Tatoeba 地址与后缀、
商业合规的 Student/Teacher 候选、数据规模和部署位置。未产生模型比选结果、Judge 策略或
promotion gate 时，后续阶段会停止，不会猜测模型或提前注册。
