# 数据分层与文件命名

## 浏览目录

```text
organized/<version>/
  README.md                    总入口
  TRAINING_WORKFLOWS.md         训练逻辑
  DATA_AND_NAMING.md            本说明
  DATASETS.md                  数据角色、行数、存在状态
  catalog.json                 原路径、别名、大小、列名、哈希
  OPEN_ITEMS.md                缺失和待核实项
  01_pair_models/<pair>/       六组独立模型
    01_training_entry.py        原训练脚本链接
    02_training_config.toml     原配置链接（英乌仅复用登记）
    03_data/                   数据角色清楚的入口
    04_selected_model/         选用权重原目录链接
    05_run_artifacts/           原实验产物链接
    06_training_report.json     原训练报告链接
    07_evaluation/              原评测证据链接
  02_shared_fourlang/<stage>/  四语分阶段资料
  03_common/                   共同训练代码、保护基准
  04_history/                  历史对照与完整旧实验目录
  05_follow_up/                未作为当前模型的后续实验
```

这是新的导航层，**不是移动后的训练根目录**。入口是符号链接，改入口所指文件会修改原文件，链接不是备份或只读权限。不要在导航目录运行训练脚本，仍从项目根目录使用原脚本路径。

## 原目录职责

| 原目录 | 作用 |
|---|---|
| data/splits | 人工划分数据，供prepare或旧训练器读取 |
| data/distillation | 教师/人工混合池、版本化蒸馏数据 |
| data/pipeline_v2 | 采集、筛选、教师生成和审核中间产物；并非全是训练输入 |
| data/specialists | 通用独立模型prepare生成的实际训练/验证文件 |
| data/experiments/weak_pair_ablation | 中乌等消融prepare后的实际输入 |
| data/multilingual/fourlang | 四语汇总后的实际输入，按阶段分开 |
| data/benchmark | 独立保护的FLORES基准，不混入训练 |
| scripts / configs | 原代码与配置；共同依赖不能随意重命名 |
| results/student、results/experiments | 训练结果与历史实验 |
| models/final_* | 部分冻结导出；不是所有当前模型都在这里 |
| results/evaluation、reports | 评测、数据审计及过程证据 |

## 统一可读别名

使用`<语言对>__<实验或数据版本>__<用途>.<原格式>`，例如：

- `zh_uz__flores_relaxed_8k_ep3__trainer_input.jsonl`：训练器读取的实际数据。
- `en_zh__human_source.parquet`：人工源池，不冒称实际训练子集。
- `fourlang__exp2__train__backup_only.jsonl`：独立备份，不冒称已经恢复原路径。

用途词：human_source人工源池；kd_source混合输入池；trainer_input实际训练输入；trainer_validation实际验证输入；backup_only备份。

不改变原扩展名、不改变内容、不将jsonl伪装为parquet。zh_en旧数据目录与en_zh模型ID保留映射，不直接改名。原模型内部config.json、tokenizer.json、model.safetensors等格式约定名称保持不变。

## 数据证据强度

存在路径 < 可解析并有行数/列名 < 有内容SHA256 < 与该次训练报告/manifest指纹一致。

catalog记录当前可解析数据的行数、列名和SHA256；不能从这些信息单独推断许可证合规、语义质量或历史训练归属。缺失文件不建悬空入口；原路径和备份单独记录。

未来物理搬迁应另有逐项原路径→新路径映射、所有调用方修改、哈希对照、回滚方案及测试；本轮不批量搬迁权重/数据，也不删除历史实验。
