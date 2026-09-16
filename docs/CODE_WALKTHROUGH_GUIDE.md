# 训练代码走读提纲

> 目标：给技术领导 30 分钟内讲清"训练思路 → 核心实现 → 严谨性"。
> 顺序按"先讲流程，再看实现，最后看证据"。

---

## 路线总览（开场 1 分钟）

1. `docs/project_structure/TRAINING_WORKFLOWS.md` —— 训练逻辑总览（一张图讲清链路）
2. `scripts/pipeline_v2/seq2seq_flow.py` —— 训练核心（加载 / 分词 / 训练 / 验证 / 导出）
3. `scripts/pipeline_v3/pair_specialist_flow.py` 等 —— 各条训练线编排
4. `scripts/pipeline/10*` + `qwen_judge.py` —— 教师蒸馏链
5. `data_safety.py` + `training_safety.py` —— 防泄漏与证据链

---

## 第 1 层：训练核心（最重要，重点讲）

文件 `scripts/pipeline_v2/seq2seq_flow.py`

| 行号 | 看点 |
|---|---|
| L508 `train_model()` | 完整训练流程：加载基座 → 分词 → 训练循环 → 逐方向验证选最优 → 导出 |
| L319 `WeightedTrainer` / L348 `compute_loss()` | **加权损失**：有权重时按有效目标 token 算每样本损失，再按样本权重归一化（人工回放 + 教师蒸馏能混训的关键） |
| L400 `DirectionAwareTrainer` / L405 `evaluate_directions()` | 逐方向验证、用宏 chrF2 选最优 |
| L96 `load_model()` | 按 family（small100 / m2m100 / nllb）区分基座加载 |
| L124 `prepare_inputs()` | 各 family 的方向语言码 / forced_bos 处理 |
| L205 `metrics()` | BLEU / chrF2 打分 |

**讲法**：先讲 L508 主流程，再停下来细讲 L348 的加权损失——这是"人工数据 + 教师数据"能一起训的核心机制。

---

## 第 2 层：各条训练线编排

| 文件 | 行号 | 看点 |
|---|---|---|
| `scripts/pipeline_v3/pair_specialist_flow.py` | L212 `prepare()` / L367 `train()` / L414 `evaluate()` / L450 `gate()` / L477 `promote()` | 通用独立模型的 `prepare→train→evaluate→gate→promote` 全链路 |
| `scripts/pipeline_v3/weak_pair_ablation.py` | L290 `prepare()` / L452 `train()` / L580 `final_evaluate()` / L691 `compare()` | 中乌三轮消融（体现"弱对特殊处理"） |
| `scripts/pipeline_v3/fourlang_flow.py` | L520 `bakeoff()` / L562 `select_student()` / L683 `source_model_for_experiment()` / L693 `train()` / L728 `evaluate()` | 四语线：多个基座打分 → 选最优 → 训练 |

**讲法**：以 `pair_specialist_flow.py` 为主线讲一遍，其他两条线只说"差异点"即可。

---

## 第 3 层：教师蒸馏链（技术亮点）

| 文件 | 行号 | 看点 |
|---|---|---|
| `scripts/pipeline/10b_generate_madlad_teacher.py` | L799 `main()` / L381 `transliterate_uzbek_cyrillic()` / L589 `build_generation_identity()` / L737 `CheckpointBuffer` | 教师生成：断点续跑 + 生成身份哈希校验 + 乌文西里尔转拉丁 |
| `scripts/pipeline_v2/qwen_judge.py` | L86 `prompt()` / L336 `generate_batch()` / L368 `main()` | 教师输出评审（PASS/FAIL + usefulness + 错误标志） |
| `scripts/pipeline/10d_build_distillation_dataset.py` | L130 `collect_frozen_texts()` / L668-740 拒绝逻辑 / L400 `main()` | 构建干净 KD 集：先到先得剔除（非 PASS / 错误标志 / 评估集泄漏 / 照抄源文），HIGH 1.0 / MEDIUM 0.8 加权 |
| `scripts/pipeline_v3/deepseek_teacher.py` | L672 `generate()` / L877 `main()` | DeepSeek 教师（后续实验线，含续跑） |

**讲法**：讲一条完整链"候选 → MadLAD 生成 → Qwen 评审 → 10d 过滤 → 学生训练"，重点落在"防泄漏 + 权重"。

---

## 第 4 层：严谨性 / 证据链（最该主动展示）

| 文件 | 行号 | 看点 |
|---|---|---|
| `scripts/pipeline_v3/data_safety.py` | L35 `protect_splits()` | 验证集 + FLORES 防泄漏 |
| `scripts/pipeline_v2/training_safety.py` | L12 `file_sha256()` / L20 `fingerprint()` / L38 `model_files()` / L92 `bind_run()` | 训练前校验配置指纹与文件哈希、把训练报告绑定到指纹 |

**讲法**：强调"每个实验都有 `train_report.json` + 指纹 + 评测证据，可复现、可追溯"，以及"删除/重训/选型都要单独审批"的纪律。

---

## 第 5 层：推理 / 服务（可选）

| 文件 | 行号 | 看点 |
|---|---|---|
| `inference/loader.py` | `load_translation_model()` | 模型加载（small100 / m2m100 / nllb 自动识别） |
| `inference/engine.py` | `TranslationEngine` | 方向切换 + 生成 |
| `scripts/pipeline_v3/translate_current_models.py` | `main()` | 统一测试入口 |

---

## 给领导讲时的 3 个要点

1. **先讲数据分层**：human_source / kd_source / trainer_input / 保护基准——这是他最容易理解"你这套为什么严谨"的入口。
2. **再讲蒸馏 + 加权损失**：这是你的技术亮点（`WeightedTrainer` + 教师链）。
3. **最后主动讲证据链**：指纹 + `train_report` + 评测门槛 + 审批纪律，比代码本身更能建立信任。
