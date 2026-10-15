# FourLang项目完整交接与上手手册

更新日期：2026-09-14。适用范围：当前会话整理的四语言翻译项目、六组独立模型，以及另一任务交接的四语集合模型证据。语音ASR、服务部署和其他并行实验仅记录边界，未经本任务全面盘点；本文件不是“全仓库都已验证”的保证。

## 0. 给接手对话的第一条说明

先完整阅读本文件，再根据用户最新请求行动。本文是项目资料，不自行授予删除、重训、收费API、发布、模型切换、stash恢复或远程登录权限。以用户当前授权为准。不要重复已经完成的清理，不要把历史待审批清单当新授权。

当前用户最关心：项目结构清晰、全部模型训练入口与流程可追溯、数据角色与存放路径明确、文件名称易理解、其他对话能快速接手。当前整理方式是新增导航，不是批量迁移原文件。

## 1. 两分钟现状

| 项目 | 当前结论 |
|---|---|
| 六组独立模型 | EN-ZH、EN-UZ、EN-RU、ZH-UZ、ZH-RU、UZ-RU各一个双向SMaLL-100，合计12方向 |
| 四语集合模型 | 单一NLLB-600M覆盖12方向；Exp1、Exp2、Exp3_v2权重保留，不等于通过发布门槛 |
| 模型调用 | 独立模型统一测试入口translate_current_models.py；不可把它的默认选型误当四语集合模型 |
| 当前质量 | 中乌ep3正式指标优于Exp1；中乌、乌俄日常诊断仍有严重语义错误，不能宣称无人复核可用 |
| 数据整理 | 当前输入池与实际训练文件已分层编目，内容哈希/行数/列名见catalog；不等于所有历史训练归属已验证 |
| 四语数据 | 8份JSONL已独立备份并校验，原训练目录仍缺失，未批量恢复stash |
| 清理 | 明确批准的P1/P2/S1/S2已删除，约释放36.17 GiB；其他模型、源码、数据、stash保留 |
| 新导航 | organized/v1_20260914已生成，132个路径引用、57个数据引用，复核changed=[] |
| 未完成 | 原路径数据恢复协调、部分旧构建依赖、完整历史复现验证、物理目录迁移、后续质量改进 |

## 2. 工作环境与安全

- 服务器项目：`/root/autodl-tmp/fourlang_translation`。
- 本地项目：`D:/dev/projects/fourlang_translation`，PowerShell。
- 用户先前提供的服务器地址：`region-46.seetacloud.com`，SSH端口28893，用户名root。连接是否仍授权/可用需按当前对话确认；本文件不含凭据。
- 历史主要解释器：`/root/autodl-tmp/venvs/small100_student/bin/python`。
- Qwen审查解释器：`/root/autodl-tmp/venvs/qwen3_judge/bin/python`；历史环境发生过包混用，不盲目安装或升级。
- 基础模型：`/root/autodl-tmp/models`；通常设`FOURLANG_MODEL_ROOT`指向此目录。
- 历史GPU为Tesla V100 32GB；当前硬件/进程/可用空间需重新读取。
- 最新清理结束时磁盘可用87340400640字节（约81.34 GiB），66%使用率；不是永久值。
- 密码与API密钥曾在旧会话中出现，不复制进文档/代码/日志，建议用户另行轮换。不要打印环境变量全集或SSH私钥。临时SSH公钥授权是否撤销由用户决定，本轮未更改。
- 本地与服务器都有历史未跟踪/未提交文件；禁止git clean、reset --hard、覆盖式同步和盲目stash操作。未跟踪不等于无价值。
- 整理恢复代码曾推送提交`1f51712`；此前服务器HEAD为`4f5dfa0`，之后新增导航是直接上传。不要推断两端同步；先读git status与HEAD，勿自动pull覆盖。

## 3. 从哪里开始浏览

服务器总入口：`/root/autodl-tmp/fourlang_translation/PROJECT_GUIDE.md`。

```text
fourlang_translation/
  PROJECT_HANDOFF.md             本完整交接手册
  PROJECT_GUIDE.md               简短入口
  organized/v1_20260914/
    README.md                   分类导航
    TRAINING_WORKFLOWS.md        训练逻辑
    DATA_AND_NAMING.md           数据分层、命名与原目录职责
    DATASETS.md                 数据实测目录
    catalog.json                每项原路径、别名、状态、哈希、适用行数/列名
    OPEN_ITEMS.md               缺失项
    01_pair_models/<pair>/      六组独立模型
    02_shared_fourlang/<stage>/ 四语阶段
    03_common/                 共同训练代码、FLORES
    04_history/                历史实验与旧模型
    05_follow_up/              DeepSeek等后续实验
  scripts/                     原训练/数据/诊断代码
  configs/                     原配置、选型映射
  data/                        原数据路径（下面详述）
  results/                     原训练、评测与实验结果
  models/                      部分冻结导出、旧路由模型
  reports/                     审核、诊断、构建及归档证据
```

organized里的现存文件入口是符号链接，不是复制品、备份或只读权限。通过链接修改会影响原件；不要在导航路径执行训练脚本，仍从项目根目录使用原入口。缺失原文件不建悬空链接。原模型内部约定文件名保持不变。

## 4. 六组当前独立模型

以下相对路径均相对于服务器项目根目录；全部模型支持所列两个方向。

| 模型ID | 方向 | 当前权重目录 |
|---|---|---|
| en_zh | en-zh / zh-en | results/student/pair_specialists/en_zh/exp2/best_model/shared |
| en_uz | en-uz / uz-en | models/final_specialists/en_uz_small100_v1 |
| en_ru | en-ru / ru-en | models/final_pair_specialists/en_ru_v1 |
| zh_uz | zh-uz / uz-zh | results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared |
| zh_ru | zh-ru / ru-zh | results/student/pair_specialists/zh_ru/exp2/best_model/shared |
| uz_ru | uz-ru / ru-uz | results/student/pair_specialists/uz_ru/exp2/best_model/shared |

映射文件：`configs/specialists/current_pair_models.json`。其中历史safe_delete字段不是当前授权，不执行其旧清单。当前测试入口和服务旧registry可能不同：`models/model_registry.json`仍引用旧Marian单向`models/final_specialists/en_zh_v1`、`zh_en_v1`，因此两者保留，不能静默删掉或切路由。

基本调用（由用户要求测试时才实际运行）：

```bash
cd /root/autodl-tmp/fourlang_translation
PY=/root/autodl-tmp/venvs/small100_student/bin/python
"$PY" scripts/pipeline_v3/translate_current_models.py --list-models
"$PY" scripts/pipeline_v3/translate_current_models.py --direction zh-en --text "你好。" --json
```

不带--text可交互测试；/quit退出。显式--model覆盖本次选型，优先于每方向环境变量。默认FP16、beam5。载入前确认真实model_path与family，不因环境名猜架构。

单次加载四语模型并在同一会话切换12方向：

```bash
"$PY" scripts/pipeline_v3/translate_current_models.py --all-directions --model results/student/fourlang/exp2/best_model/shared
```

会话内用`/direction zh-uz`切换默认方向，或直接输入`ru-en: Текст`；`/directions`列出方向，`/quit`退出。该模式强制显式指定共享模型，避免误载六个独立模型，模型只加载一次。

## 5. 训练代码逻辑与实际数据

### 5.1 通用独立训练：英中、英俄、中俄、乌俄

`scripts/pipeline_v3/pair_specialist_flow.py` + `configs/specialists/six_pair.toml`。

流程：人工源池/KD混合池 → prepare归一化、防泄漏、方向配额采样 → 实际JSONL与指纹报告 → train → evaluate → gate → 经单独批准才promote。

- Exp1从SMaLL-100基座开始；Exp2从该语言对Exp1开始。
- 英俄复用旧Exp1：`results/student/en_ru/exp1/best_model/shared`，不要重新跑通用Exp1替代它。
- 实际读取：`data/specialists/<pair>/<exp1或exp2>/train.jsonl`与`validation.jsonl`。
- prepare证据：`reports/pipeline/pair_specialists/<pair>/<stage>_data.json`；训练前验证配置指纹和文件哈希。
- 训练输出：`results/student/pair_specialists/<pair>/<stage>/`；`train_report.json`记录实际参数、起点、行数、指标与检查点。
- 评测/门槛：`results/evaluation/pair_specialists/<pair>/{exp1,exp2,promotion_gate}.json`。
- 当前默认配置Exp1三轮3e-5、Exp2两轮5e-6、batch16×累积2；不要用当前配置冒充每次历史实际参数。

### 5.2 英乌旧训练分支

- `scripts/pipeline/11b_train_student_exp2.py`直接读取Parquet，不读取six_pair.toml；后者只登记复用。
- 起点：`results/student/small100/exp1_finetune/best_model`。
- 实际训练：`data/distillation/en_uz/v1/11a_exp2_training/exp2_train_combined_v1.parquet`。
- 实际验证：`data/splits/en_uz/v1/validation_exp1_bidirectional_v1.parquet`。
- 原产物：`results/student/small100/exp2_distillation_v1/`，报告名`training_report.json`，另有training_history.csv。
- 选用冻结导出见第4节。不要套用通用prepare命令声称已复现英乌旧训练。

### 5.3 中乌当前三轮分支

- 编排：`scripts/pipeline_v3/weak_pair_ablation.py`，配置`configs/specialists/weak_pair_ablation.toml`。
- 输入池：`data/distillation/zh_uz/flores_relaxed_8k/train.jsonl`。
- prepare读取原验证源`data/splits/zh_uz/v2/validation.jsonl`，去重、防泄漏后产生实际输入。
- 实际训练：`data/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/train.jsonl`，验证为同目录validation.jsonl。
- 起点：`results/student/pair_specialists/zh_uz/exp1/best_model/shared`。三轮实验从Exp1重新训练，不是两轮模型继续一轮。
- 历史报告45521训练行、1000验证行、3轮、5e-6、batch16×累积2、4269计划优化器步、约1008.93秒。
- 初始宏chrF2 41.9301，最佳42.4710，增量0.5409；这是训练验证指标，不是FLORES。
- 选型FLORES dev；候选冻结后final_evaluate执行一次devtest。不要对最终测试集不断调参。

### 5.4 共同核心

`scripts/pipeline_v2/seq2seq_flow.py::train_model()`负责加载、分词、训练、验证和导出。WeightedTrainer有权重时计算有效目标token的每样本损失，再按样本权重归一化。启用逐方向验证时按宏chrF2选模，否则可能按eval_loss；以报告为准。

`scripts/pipeline_v3/fourlang_flow.py`同时提供共同数据辅助，不是只被集合模型使用。`inference/loader.py`、`inference/engine.py`、`src/model_utils.py`也属于共同依赖。

## 6. 六组源数据池位置

以下是输入来源，不可自动等同训练器最终读取文件；最终输入见第5节与catalog。

| pair | 人工池 | KD/混合池 | 原验证源 |
|---|---|---|---|
| en_zh | data/splits/zh_en/v1/train_bidirectional_v1.parquet | data/distillation/zh_en/v1/18h_exp2_training/exp2_train_combined_v1.parquet | data/splits/zh_en/v1/validation_bidirectional_v1.parquet |
| en_uz | data/splits/en_uz/v1/train_exp1_bidirectional_v1.parquet | data/distillation/en_uz/v1/11a_exp2_training/exp2_train_combined_v1.parquet | data/splits/en_uz/v1/validation_exp1_bidirectional_v1.parquet |
| en_ru | data/splits/en_ru/v1/train.jsonl | data/distillation/en_ru/v1/train.jsonl | data/splits/en_ru/v1/validation.jsonl |
| zh_uz | data/splits/zh_uz/v2/train.jsonl | data/distillation/zh_uz/v4/train.jsonl；当前增量模型另用flores_relaxed_8k/train.jsonl | data/splits/zh_uz/v2/validation.jsonl |
| zh_ru | data/splits/zh_ru/v1/train.jsonl | data/distillation/zh_ru/v1/train.jsonl | data/splits/zh_ru/v1/validation.jsonl |
| uz_ru | data/splits/uz_ru/v1/train.jsonl | data/distillation/uz_ru/v1/train.jsonl | data/splits/uz_ru/v1/validation.jsonl |

保护基准：`data/benchmark/fourlang/flores_dev.parquet`与`flores_devtest.parquet`。不将测试集/诊断句直接加进训练后再用同一份声称提升。

新别名格式`<pair>__<stage或version>__<role>.<原格式>`。role区分human_source、kd_source、trainer_input、trainer_validation、backup_only。原目录zh_en与模型ID en_zh映射保留；不修改模型内部config.json、tokenizer.json、model.safetensors等名称。

## 7. 四语集合模型与数据（另一任务的交接范围）

已训练导出分别在`results/student/fourlang/{exp1,exp2,exp3_v2}/best_model/shared`。选型证据`results/model_selection/fourlang/selected_student.json`指向NLLB-600M。

- 入口：`scripts/pipeline_v3/fourlang_flow.py`。
- 配置：`configs/multilingual/fourlang.toml`；流程阶段`configs/pipelines/fourlang.toml`。
- 基座→Exp1人工平衡→Exp2教师与人工回放→Exp3_v2定向修复。
- 实际输入`data/multilingual/fourlang/<stage>/{train,validation}.jsonl`。
- 报告`results/student/fourlang/<stage>/train_report.json`；评测`results/evaluation/fourlang/<stage>/metrics.json`。

| stage | 训练行数 | 验证行数 | 轮数 | 学习率 | 历史FLORES宏BLEU / chrF2 |
|---|---:|---:|---:|---:|---|
| exp1 | 120000 | 27070 | 3 | 3e-5 | 23.5956 / 42.1831 |
| exp2 | 158000 | 26910 | 2 | 5e-6 | 24.2378 / 42.6084 |
| exp3_v2 | 46000 | 27070 | 1 | 1e-6 | 24.2029 / 42.5825 |

上述来自四语交接报告，宏平均不等于所有方向都提升。四语发布门槛未通过，不能自动切生产路由。Exp3是另一数据版本，有46000条训练数据但训练完成证据仍待核实，不能混成Exp3_v2。

8份Exp1/Exp2/Exp3/Exp3_v2训练与验证JSONL原位置仍缺失。已从stash对象只读导出到：

`/root/autodl-tmp/fourlang_unified_organization/approved_cleanup_20260914T071530Z/stash_dataset_backup/data/multilingual/fourlang/`

大小/字节SHA与原盘点一致；Exp3_v2行指纹与训练manifest匹配，其他阶段缺可比较基准，不声称完成历史绑定。不要重新聚合冒充原训练数据，不批量stash恢复。原位置恢复需与四语任务协调；本轮只建备份导航。

## 8. 实验经过与质量结论

1. 六组独立模型建立人工Exp1、蒸馏Exp2，英乌复用既有模型，英俄复用旧Exp1。
2. 中乌严格源审查容量不足，补充Wikipedia中文15000、乌文8000，源配额达到目标。
3. 教师审查与MINOR二审通过量仍不足；用户同意先用放宽筛选的每方向8000条新教师数据试验，和旧数据/人工数据合并45521行。报告训练中FLORES行数0。
4. 两轮验证未超过起点（宏chrF2 -0.4094）；三轮实验提升并成为dev双方向胜者，冻结后devtest胜过Exp1。

| devtest方向（各1012条） | Exp1 BLEU / chrF2 | ep3 BLEU / chrF2 | 增量BLEU / chrF2 |
|---|---|---|---|
| zh-uz | 4.6314 / 31.7887 | 5.4873 / 33.6935 | +0.8559 / +1.9048 |
| uz-zh | 19.0720 / 13.9811 | 20.2646 / 14.5486 | +1.1926 / +0.5675 |

证据：`results/evaluation/weak_pair_ablation/zh_uz/final_devtest.json`。不同指标、方向、分词器、测试集不可直接换算；这些数值不是生产准确率。

5. 日常诊断发现错义，恢复比较了中乌/乌俄当前与Exp1旧版：240/240输入编码一致，237/240译文逐字一致；未发现系统性语言设置错误，新旧模型均有严重词义错误。不能把一致率当正确率，也不应立即回退旧模型。
6. 用户偏好“意思正确即可”，允许措辞差异，但数字、否定、实体、动作错误不能因此放行。
7. DeepSeek pilot完成400条（各200）。初始38条数字规则拒绝；改规则重审后组合可用zh-uz197/200、uz-zh172/200。规则通过不等于语义全对。
8. DeepSeek全量计划收敛至每方向9000，共18000；配置`configs/directions/zh_uz_deepseek_teacher_v1.toml`，代码`scripts/pipeline_v3/deepseek_teacher.py`，数据根`data/pipeline_v2/zh_uz_deepseek_teacher_v1`。此轮未核实最终生成进度，不能据旧in_progress标签推断当前仍运行；更没有证据它已替代当前学生模型。检查full/report.json、检查点与活跃任务后再定下一步。

历史问题包括归一化遇到不支持西里尔字符、路径拼错、CUDA环境不兼容、Killed、API响应中断。仅凭Killed不能确诊OOM。Qwen int8与fp16二审359条判定约89.69%一致，不应将精度切换当完全等价；环境已能运行过，不默认重装。

## 9. 已完成的整理、恢复和清理

### 恢复与索引

- 缺失的pair_specialist_flow.py及测试从原归档恢复，与本地哈希相同；服务器5项测试通过，非训练全链路验证。
- 旧实验62份报告全部在原evidence.tar.gz找到；原位置47份匹配，缺失15份恢复。
- 原路径缺失559份归档文件中487份四语诊断归另一任务未处理；其余72份独立备份，其中63份仅缺失时补回（39数据、22报告、1文档、1脚本），9份仅备份。
- build_clean_zh_uz_v4.py已补回，但preview_manual_zh_uz_isolation.py共同依赖仍待核对，未宣称旧构建链可运行。
- 旧file_inventory.json和archive_checksums.json没有找回，用新快照和恢复增量补充追溯，不篡改旧快照。

### 已授权删除（已完成，不可再次执行）

下列相对路径为2026-09-14用户明确批准的仅四项：

| ID | 删除目录 |
|---|---|
| P1 | results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints |
| P2 | results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints |
| S1 | results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-500 |
| S2 | results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1000 |

删除前备份117个检查点辅助/元数据文件和8份四语数据。执行前后6组独立选用模型、四语3组导出与最终1438共101个文件SHA一致；6条stash提交未改变。实际增加可用空间38833512448字节。

**没有完整备份被删除检查点权重和优化器状态，不能恢复对应精确续训。** 四语checkpoint-1438完整保留，最佳模型、旧基线、其他代码/数据/日志/环境/基础模型与SSH授权未清理。

### 新导航

新增`build_project_navigation.py`及docs/project_structure文档；服务器`organized/v1_20260914`生成132个引用、57个数据引用，复核changed=[]。

13个缺失引用分成：英俄通用Exp1准备文件2个（复用分支，不能认定旧训练丢失）；四语原数据8个（已备份未原位恢复）；Exp3模型/训练报告/评测3个（完成情况未知）。不是“全部数据齐全”。源码编译与3项小检查通过；未做全部模型训练重跑测试。

## 10. 证据与归档绝对路径

| 内容 | 服务器位置 |
|---|---|
| 新导航和逐项catalog | /root/autodl-tmp/fourlang_translation/organized/v1_20260914 |
| 独立模型早期全路径索引 | /root/autodl-tmp/fourlang_diagnostics_recovery_20260914/pair_organization/20260914T062318Z |
| 独立诊断对照结果 | /root/autodl-tmp/fourlang_diagnostics_recovery_20260914/20260914T060725Z |
| 历史报告核对 | /root/autodl-tmp/fourlang_diagnostics_recovery_20260914/historical_report_recovery_v1 |
| 72份证据及63份恢复日志 | /root/autodl-tmp/fourlang_diagnostics_recovery_20260914/pair_evidence_recovery_v1 |
| 重建实验快照 | /root/autodl-tmp/fourlang_diagnostics_recovery_20260914/experiment_archive/20260914T062427Z |
| 四语历史盘点 | /root/autodl-tmp/fourlang_shared_organization/inventory_20260914_143314 |
| 四语最新交接 | /root/autodl-tmp/fourlang_shared_organization/handoff_20260914_145040 |
| 统一审批原快照 | /root/autodl-tmp/fourlang_unified_organization/20260914T065733Z |
| 清理回执与备份 | /root/autodl-tmp/fourlang_unified_organization/approved_cleanup_20260914T071530Z |

四语先读current_changes.md、current_existence_recheck.json，再看旧HANDOFF/experiment_history/data_lineage等。旧missing_files是历史快照，不是当前事实。

项目内补充文档：docs/PAIR_EXPERIMENT_RECORD_20260914.md（原报告完整数值摘录）、PAIR_DIAGNOSTIC_RECOVERY_20260914.md、PAIR_ORGANIZATION_FINAL_REVIEW_20260914.md、UNIFIED_ORGANIZATION_CLEANUP_RESULT_20260914.md、docs/project_structure目录。日期旧文档中的“待删除/仍保留”以最新清理结果校正，不改写历史回执。

## 11. 接手后优先事项

1. 先确认用户要的是继续整理、模型测试、数据修复还是新训练，不把“继续”扩展为全仓库重构。
2. 读PROJECT_GUIDE及organized的catalog/OPEN_ITEMS，核对代码和模型映射当前版本。确认其他任务是否正在写四语或共同数据。
3. 当前可安全继续完善：逐模型数据来源链和构建报告绑定、旧英俄Exp1真实输入追溯、英乌旧报告与当前导出关系、文档链接一致性。
4. 四语原数据恢复：有独立副本但需协调；只按确定哈希、缺失目标逐项恢复，不批量stash恢复、不重生成替代。
5. 历史构建缺依赖先只读比对备份和调用方；不因为一个导入失败就恢复所有共享旧脚本。
6. 若用户要真实物理改名搬迁：另立旧→新映射、更新所有调用方/配置、逐项哈希核对、兼容与回滚测试；先处理小型入口，不直接搬权重和共同数据。
7. 若用户要提升质量：先核对已生成DeepSeek数据、有限语义抽查和独立保留测试集，避免立即重复全量付费生成或用已见devtest调参。
8. Git同步需先核对两端脏工作区；本轮新增导航和手册不代表已推送GitHub。只提交明确范围，不夹带ASR、synthetic或其他任务改动。

## 12. 快速只读检查

在用户授权的服务器环境中，可用：

```bash
cd /root/autodl-tmp/fourlang_translation
git status --short
git rev-parse --short HEAD
/root/autodl-tmp/venvs/small100_student/bin/python scripts/pipeline_v3/build_project_navigation.py check --output /root/autodl-tmp/fourlang_translation/organized/v1_20260914
```

check只复核目录所录存在状态、文件内容哈希和导航链接；changed=[]表示相对扫描未变，**并不表示缺失项消失或训练已就绪**。目录型模型在本轮导航仅查存在，完整101文件权重哈希校验来自此前清理回执，两种核验不要混说。

## 13. 维护此手册

每完成一次实质操作更新：日期、已完成项、准确路径、验证证据、失败/待办、授权边界和回滚能力。新的已训练模型必须有起始模型、实际数据指纹、参数、报告和选型依据；没有证据就写待核实。

不要把本手册当永久实时数据库。它是接手所需的统一上下文，原始报告和新鲜只读检查才是后续事实依据。无需把密码、密钥、整份训练数据或模型权重塞入Markdown。
