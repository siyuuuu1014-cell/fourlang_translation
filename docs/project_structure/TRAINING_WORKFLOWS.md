# 模型训练流程

服务器项目根目录：`/root/autodl-tmp/fourlang_translation`。本页是流程说明，不会启动训练；重跑之前必须选择新的实验版本和输出目录，不能覆盖当前结果。

## 1. 六组独立模型

每组使用一个SMaLL-100模型支持两个方向。en_zh与历史数据目录zh_en表示同一个语言对，但不能因此批量改名。

```text
人工输入池 / 教师与人工混合池
    ↓ prepare：归一化、验证集与FLORES防泄漏、配额采样
实际 train.jsonl + validation.jsonl + 数据指纹报告
    ↓ train：加载起始模型、分词、加权损失、验证选最佳模型
best_model/shared + train_report.json
    ↓ evaluate / gate
逐方向评测、选型证据 → 显式冻结/路由（本次不执行）
```

### 通用分支：英中、英俄、中俄、乌俄

- 编排：`scripts/pipeline_v3/pair_specialist_flow.py`。
- 配置：`configs/specialists/six_pair.toml`。
- `prepare()`读取配置的human_train或kd_train，不把整个源池直接交给训练器。产物是`data/specialists/<pair>/<exp1或exp2>/train.jsonl`与`validation.jsonl`。
- 数据审计：`reports/pipeline/pair_specialists/<pair>/<stage>_data.json`，记录配置指纹、源文件与实际数据哈希、防泄漏结果。
- Exp1从基座开始；Exp2从对应Exp1开始。英俄复用`results/student/en_ru/exp1/best_model/shared`，不能把后来prepare得到的Exp1数据当作旧训练输入证明。
- 训练结果：`results/student/pair_specialists/<pair>/<stage>/`；评测：`results/evaluation/pair_specialists/<pair>/`。
- 当前配置：Exp1三轮、学习率3e-5；Exp2两轮、5e-6；batch16×累积2。实际完成参数以各次训练报告为准，配置可能后来修改。

### 英乌：旧训练器复用分支

- 入口：`scripts/pipeline/11b_train_student_exp2.py`。它不读取six_pair.toml；该配置只是登记复用关系。
- 起点：`results/student/small100/exp1_finetune/best_model`。
- 直接读取`data/distillation/en_uz/v1/11a_exp2_training/exp2_train_combined_v1.parquet`。
- 验证：`data/splits/en_uz/v1/validation_exp1_bidirectional_v1.parquet`。
- 原产物：`results/student/small100/exp2_distillation_v1/`，报告名是`training_report.json`，不是`train_report.json`。
- 当前导出：`models/final_specialists/en_uz_small100_v1`。历史参数必须结合原报告和命令行覆盖值判断。

### 中乌：relaxed_8k_ep3选用分支

- 入口：`scripts/pipeline_v3/weak_pair_ablation.py`；配置：`configs/specialists/weak_pair_ablation.toml`。
- 输入池：`data/distillation/zh_uz/flores_relaxed_8k/train.jsonl`，包含人工、既有KD、新增教师数据；不是“总共8000行”。
- prepare实际读取的原验证源：`data/splits/zh_uz/v2/validation.jsonl`。
- 实际训练输入：`data/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/{train,validation}.jsonl`。
- 起点：`results/student/pair_specialists/zh_uz/exp1/best_model/shared`。
- 三轮实验从Exp1重新训练，不是接着两轮模型再跑一轮。历史报告：45521训练行、1000验证行、3轮、5e-6、batch16×累积2；选型验证子集与全部验证行数分开理解。
- 输出：`results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/`。
- 在FLORES dev比较，冻结候选后做devtest；不要不断以devtest选择新模型。
- DeepSeek教师数据属于后续实验，不是上述已选模型的训练来源。不能把教师输出生成完成等同于学生训练完成。

## 2. 通用训练核心

`scripts/pipeline_v2/seq2seq_flow.py`中的`train_model()`负责加载、分词、训练、验证和导出；`WeightedTrainer.compute_loss()`在有权重时按有效目标token计算每样本损失，再按样本权重归一化。语言方向验证启用时用宏chrF2选最佳模型，否则可能按eval_loss；以报告的metric_for_best_model为准。

`scripts/pipeline_v3/fourlang_flow.py`还提供共同数据归一化等辅助能力，因此并非“只被四语使用”的可随意移动代码。

## 3. 四语集合模型（独立实验线）

现存选型证据指向NLLB-600M，一个模型覆盖12方向；small100_student只是环境名。

```text
六组双语源数据 → 12方向汇总、防泄漏、平衡
基座 → Exp1人工训练 → Exp2教师蒸馏与人工回放 → Exp3_v2定向修复
```

- 入口：`scripts/pipeline_v3/fourlang_flow.py`。
- 配置：`configs/multilingual/fourlang.toml`；阶段编排：`configs/pipelines/fourlang.toml`。
- 实际输入：`data/multilingual/fourlang/<stage>/{train,validation}.jsonl`。
- 原结果：`results/student/fourlang/<stage>/`；评测：`results/evaluation/fourlang/<stage>/metrics.json`。
- 报告记录Exp1/Exp2/Exp3_v2训练行数120000/158000/46000，轮数3/2/1；Exp3另一个数据版本，训练完成尚未核实。
- 数据备份与原路径分别列在导航中；原路径缺失时不允许自动用重新聚合的数据冒充历史文件。
- 存在权重不等于通过发布门槛。本次不修改另一任务的训练、路由或恢复状态。

## 4. 复现检查顺序

1. 找到该实验训练报告、起始模型、实际输入文件和数据指纹，不只看当前配置。
2. 核对数据哈希、环境版本、seed、采样方式、权重、超参数和选型指标。
3. 若需精确续训，还需对应优化器/RNG等完整断点；已删除的四个目录不能仅靠元数据恢复。
4. 使用新实验输出目录。先小规模验证，再按新的明确训练请求执行。

本整理没有重新训练或保证所有历史流程可一键复现；缺失项见OPEN_ITEMS.md。
