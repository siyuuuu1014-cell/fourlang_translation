下面就是可以直接保存为：

```text
docs/EN_UZ_Pipeline_V1_Experiment_Summary.md
```

的完整 Markdown 正文。

````markdown
# EN↔UZ Translation Pipeline V1 实验总结与最终模型记录

> 项目：FourLang Translation  
> 语言对：English ↔ Uzbek（`en_uz` / `uz_en`）  
> 项目根目录：`/root/autodl-tmp/fourlang_translation`  
> 当前阶段：EN↔UZ Pair-specific Model V1 已完成  
> 最终模型：SMaLL-100 Student Exp2  
> 最终判定：`EXP2_ACCEPT`  
> 整理日期：2026-08-27

---

# 1. 项目目标

FourLang Translation 项目的最终目标是构建一个支持以下四种语言的轻量翻译模型：

```text
ZH 中文
EN 英语
RU 俄语
UZ 乌兹别克语
````

最终需要支持 12 个有向翻译方向：

```text
ZH → EN    EN → ZH
ZH → RU    RU → ZH
ZH → UZ    UZ → ZH

EN → RU    RU → EN
EN → UZ    UZ → EN

RU → UZ    UZ → RU
```

当前首先完成的语言对是：

```text
EN → UZ
UZ → EN
```

本阶段目标不是直接进行移动端部署，而是先完成：

```text
数据构建
    ↓
Baseline
    ↓
Human Fine-tuning
    ↓
Held-out Evaluation
    ↓
Teacher Distillation
    ↓
Final Evaluation
    ↓
冻结最终 EN↔UZ 模型
```

---

# 2. EN↔UZ Pipeline 最终状态

当前 EN↔UZ 已经完成：

```text
数据汇总                     ✅
数据标准化                   ✅
去重和规则过滤               ✅
Risk Routing                 ✅
Qwen 数据质量审核            ✅
Approved Dataset             ✅
Train / Validation Split     ✅
Original SMaLL-100 Baseline  ✅
Student Exp1                 ✅
Held-out Evaluation          ✅
MADLAD Teacher               ✅
Qwen3-8B Quality Gate        ✅
Clean Distillation Dataset   ✅
Human Replay + KD Dataset    ✅
Student Exp2                 ✅
Exp1 vs Exp2 Validation      ✅
Final Held-out Evaluation    ✅
Final Model Selection        ✅

Final Decision:
EXP2_ACCEPT
```

因此：

```text
EN↔UZ Pair-specific Model V1
已经完成
```

当前可以冻结 Exp2，不需要继续反复微调 EN↔UZ。

---

# 3. 整体 Pipeline

```text
原始 EN-UZ 平行语料
        │
        ├── public_5k_v1
        ├── public_5k_v2
        ├── HPLT
        └── Tatoeba
        │
        ▼
Step 01
多数据源汇总
        │
        ▼
Step 02
文本标准化
        │
        ▼
Step 03
去重 + 规则过滤
        │
        ▼
Step 04
Risk Routing
        │
        ├── AUTO_ACCEPT
        │
        └── NEEDS_QWEN
                │
                ▼
Step 05A
Qwen Review 数据准备
                │
                ▼
Step 05B
Qwen 一审
                │
                ▼
Step 05C
Qwen 二审
                │
                ▼
Step 06
Approved Dataset
        │
        ▼
Step 07
Train / Validation Split
        │
        ▼
Step 08A
Original SMaLL-100 Baseline
        │
        ▼
Step 08B
Student Exp1 Fine-tuning
        │
        ▼
Step 08C
Exp1 Validation Evaluation
        │
        ▼
Step 09
Tatoeba + Challenge Held-out
        │
        ▼
Step 09A
Uzbek Script Normalization
        │
        ▼
发现 EN→UZ 跨域泛化仍有优化空间
        │
        ▼
Step 10A
Distillation Candidates
        │
        ▼
Step 10B
MADLAD Teacher Generation
        │
        ▼
Step 10C
Qwen3-8B Teacher Quality Gate
        │
        ▼
Step 10D
Clean Distillation Dataset
        │
        ▼
Step 11A
Human Replay + Teacher KD
        │
        ▼
Step 11B
Student Exp2
        │
        ▼
Step 11C
Exp1 vs Exp2 Validation
        │
        ▼
Step 12
Final Held-out Evaluation
        │
        ▼
EXP2_ACCEPT
```

---

# 4. 项目目录与运行环境

## 4.1 服务器项目目录

```text
/root/autodl-tmp/fourlang_translation
```

本地 Windows 项目目录：

```text
D:\dev\projects\fourlang_translation
```

---

## 4.2 SMaLL-100 Student 环境

Python 环境：

```text
/root/autodl-tmp/venvs/small100_student
```

激活命令：

```bash
source /root/autodl-tmp/venvs/small100_student/bin/activate
```

主要用于：

```text
08A Baseline
08B Exp1
08C Exp1 Evaluation
09 Held-out Evaluation
11A Exp2 Dataset
11B Exp2 Training
11C Exp2 Evaluation
12 Final Evaluation
```

---

## 4.3 Qwen Judge 环境

```text
/root/autodl-tmp/venvs/qwen3_judge
```

激活：

```bash
source /root/autodl-tmp/venvs/qwen3_judge/bin/activate
```

主要用于：

```text
10C Qwen3-8B Quality Gate
```

Qwen Judge 与 SMaLL-100 单独建立环境，是为了避免：

```text
transformers
tokenizer
torch
```

版本冲突。

---

# 5. 使用的模型

## 5.1 Student Base Model：SMaLL-100

模型：

```text
alirezamsh/small100
```

服务器本地路径：

```text
/root/autodl-tmp/models/small100
```

主要模型文件：

```text
model.safetensors
config.json
generation_config.json
sentencepiece.bpe.model
vocab.json
tokenizer_config.json
special_tokens_map.json
tokenization_small100.py
```

模型权重约：

```text
1.3 GB
```

SMaLL-100 是最终需要部署的小模型。

---

## 5.2 Teacher：MADLAD-400-3B-MT

模型：

```text
google/madlad400-3b-mt
```

Hugging Face Cache：

```text
/root/autodl-tmp/huggingface/hub/
models--google--madlad400-3b-mt
```

MADLAD 的作用：

```text
Source Sentence
      ↓
MADLAD
      ↓
生成 Teacher Translation
      ↓
Qwen3-8B 审核
      ↓
Clean Teacher Target
```

MADLAD **不是最终部署模型**。

它只负责产生更丰富的翻译目标。

---

## 5.3 Judge：Qwen3-8B

服务器路径：

```text
/root/autodl-tmp/models/Qwen3-8B
```

模型约：

```text
16 GB
```

作用：

```text
Source
+
Human Reference
+
MADLAD Teacher Translation
        ↓
Qwen3-8B
        ↓
PASS / MINOR / FAIL / UNCERTAIN
+
Teacher usefulness
+
错误类型
```

Qwen3-8B 也不会进入移动端。

---

# 6. 服务器硬件

主要实验服务器 GPU：

```text
Tesla V100-PCIE-32GB
```

主要阶段均在该 GPU 上完成：

```text
SMaLL-100 Exp1
MADLAD Teacher Generation
Qwen3-8B Judge
SMaLL-100 Exp2
Final Evaluation
```

---

# 7. Step 01：原始 EN-UZ 数据汇总

## 作用

将多个来源的 English-Uzbek 平行语料合并。

主要数据源：

```text
public_5k_v1
public_5k_v2
HPLT
Tatoeba
```

Step01 汇总结果：

```text
Total rows             211388
Unique pair_id          66803
Duplicate rows         144585
```

各数据源：

```text
public_5k_v1     129926
public_5k_v2      70823
hplt              10050
tatoeba              589
```

## 脚本

Step01～06 属于项目早期数据 Pipeline。

由于当前最终阶段没有重新从服务器核验这几个早期脚本的实际文件名，因此本文不杜撰文件名。

其功能可以确定为：

```text
Step01:
Combine EN-UZ Parallel Sources
```

---

# 8. Step 02：文本标准化

主要进行：

```text
Unicode Normalization
Whitespace Normalization
Empty Text Check
English / Uzbek Length Statistics
Uzbek Cyrillic Detection
Normalized Pair Construction
```

结果：

```text
Rows                     211388
Normalized empty              0

Uzbek Cyrillic              160
Cyrillic rate            0.0757%

Average English words     17.31
Average Uzbek words       14.03
```

功能：

```text
Step02:
Normalize EN-UZ Parallel Corpus
```

---

# 9. Step 03：去重与规则过滤

主要进行：

```text
Normalized Pair Deduplication
Word Length Filtering
Length Ratio Filtering
Uzbek Cyrillic Filtering
Benchmark Leakage Filtering
Multi-source Consensus Analysis
```

结果：

```text
Input                        211388
Unique normalized pairs       66791

Duplicates removed           144597
```

Rejected：

```text
invalid_word_length               5
invalid_length_ratio              1
cyrillic_uz                     160
benchmark_leak                  729
```

最终：

```text
Accepted                     65897
Rejected                       894

Accepted rate               98.66%
```

Source consensus：

```text
1 source      9934
2 sources    55954
3 sources        9
```

功能：

```text
Step03:
Deduplicate + Rule Filter
```

---

# 10. Step 04：Risk Routing

不是直接把所有 65K 数据交给 Qwen。

先通过规则进行风险分层：

```text
AUTO_ACCEPT
NEEDS_QWEN
```

结果：

```text
Input          65897

AUTO_ACCEPT    62318
94.57%

NEEDS_QWEN      3579
5.43%
```

主要风险：

```text
NEGATION_MISMATCH     2792
NUMBER_MISMATCH        564
LOW_LETTER_RATIO       183
TIME_MISMATCH           75
DATE_MISMATCH           21
URL_MISMATCH             7
REPEATED_PUNCT           3
SOURCE_TARGET_SAME       2
EMAIL_MISMATCH           1
```

平均 quality score：

```text
79.19
```

这一步的核心思想是：

```text
规则可以明确判断的数据
        ↓
AUTO_ACCEPT

存在高风险的数据
        ↓
Qwen Review
```

从而避免对全部数据调用大模型。

---

# 11. Step 05A：Qwen Review 数据准备

Risk Review：

```text
3579
```

同时从 AUTO_ACCEPT 中随机抽查：

```text
500
```

因此总 Review：

```text
4079
```

目的：

```text
高风险数据 → 全审

低风险数据 → Audit Sample
```

---

# 12. Step 05B：Qwen 一审

总数据：

```text
4079
```

结果：

```text
PASS          2610
MINOR          974
FAIL           481
UNCERTAIN       14
```

其中 Risk Review：

```text
3579

PASS          2147
MINOR          949
FAIL           470
UNCERTAIN       13
```

AUTO_ACCEPT Audit：

```text
500

PASS           463
MINOR           25
FAIL            11
UNCERTAIN        1
```

---

# 13. Step 05C：Qwen 二审

对第一次 Judge 中的问题数据进一步复核。

二审：

```text
495
```

最终：

```text
CONFIRMED_FAIL      414
QUARANTINE           42
DOWNGRADED_MINOR     29
RESOLVED_PASS         5
JUDGE_ERROR           3
RESOLVED_MINOR        2
```

确认错误：

```text
omission          408
addition          345
mistranslation    408
number_error      114
time_error        123
entity_error      337
negation_error     25
```

---

# 14. Step 06：Approved Dataset

综合：

```text
Rule Quality Gate
+
Qwen Review
+
Qwen Second Review
```

构建 Approved Dataset。

结果：

```text
Input        65897

Approved     65441
Approved %   99.31%
```

Quality Tier：

```text
SILVER       61826    weight 0.8
GOLD          2610    weight 1.0
BRONZE        1005    weight 0.5

REJECT         414
QUARANTINE      42
```

Exp1 实际采用：

```text
GOLD
+
SILVER
```

---

# 15. Step 07：Train / Validation Split

代码：

```text
scripts/pipeline/07_build_train_validation_split.py
```

主要作用：

```text
Approved Pairs
       ↓
严格 Pair-level Split
       ↓
Train
Validation
       ↓
展开为
EN→UZ + UZ→EN
```

并进行：

```text
Pair Leakage Check
Group Leakage Check
English Sentence Leakage
Uzbek Sentence Leakage
```

结果：

```text
Approved pairs       65441

Train pairs          62169
Validation pairs      3272
```

Exp1 只使用 GOLD + SILVER：

```text
Train pairs          61216
Validation pairs      3220
```

双向展开：

```text
Train directed samples      122432
Validation directed samples   6440
```

训练质量：

```text
SILVER      58738
GOLD         2478
```

Validation：

```text
SILVER       3088
GOLD          132
```

Leakage：

```text
Pair leakage       0
Group leakage      0
English leakage    0
Uzbek leakage      0
```

关键文件目录：

```text
data/splits/en_uz/v1/
```

Train：

```text
data/splits/en_uz/v1/
train_exp1_bidirectional_v1.parquet
```

Validation：

```text
data/splits/en_uz/v1/
validation_exp1_bidirectional_v1.parquet
```

---

# 16. Step 08A：Original SMaLL-100 Baseline

代码：

```text
scripts/pipeline/08a_eval_small100_baseline.py
```

模型：

```text
/root/autodl-tmp/models/small100
```

测试数据：

```text
data/splits/en_uz/v1/
validation_exp1_bidirectional_v1.parquet
```

结果：

## EN→UZ

```text
BLEU      6.8625
chrF++   21.3215
Exact     0.00%
Latency   0.1010 s/sample
```

## UZ→EN

```text
BLEU      6.7995
chrF++   27.9452
Exact     0.28%
Latency   0.0997 s/sample
```

结论：

```text
Original SMaLL-100 对当前 EN↔UZ 数据域能力明显不足。
```

因此进入：

```text
Human Fine-tuning
```

---

# 17. Step 08B：Student Exp1

代码：

```text
scripts/pipeline/08b_train_small100_exp1.py
```

初始化：

```text
/root/autodl-tmp/models/small100
```

Train：

```text
data/splits/en_uz/v1/
train_exp1_bidirectional_v1.parquet
```

Validation：

```text
data/splits/en_uz/v1/
validation_exp1_bidirectional_v1.parquet
```

训练核心参数：

```text
epochs                   3
batch_size               16
gradient_accumulation    2

effective batch          32

learning_rate            3e-5
max_length               256

FP16                     True
```

输出：

```text
results/student/small100/exp1_finetune/
```

最终 Exp1 模型：

```text
results/student/small100/
exp1_finetune/best_model
```

训练记录：

```text
results/student/small100/
exp1_finetune/training_history.csv
```

```text
results/student/small100/
exp1_finetune/training_report.json
```

最佳 Validation Loss：

```text
1.1960
```

---

# 18. Step 08C：Exp1 Validation Evaluation

代码：

```text
scripts/pipeline/08c_eval_small100_exp1.py
```

评估：

```text
BLEU
chrF++
Exact Match
Latency
```

同时与 Original Baseline 比较。

Exp1：

## EN→UZ

```text
BLEU      42.6848
chrF++    64.1272
Exact      9.19%
Latency    0.0345 s/sample
```

## UZ→EN

```text
BLEU      45.1219
chrF++    63.9964
Exact     10.09%
Latency    0.0282 s/sample
```

相较 Original：

```text
Mean ΔBLEU     ≈ +37.07
Mean ΔchrF++   ≈ +39.43
```

说明 Human Fine-tuning 极其有效。

---

# 19. Step 09：Held-out Evaluation

代码：

```text
scripts/pipeline/09_eval_heldout_en_uz.py
```

目的：

不再只看 Validation。

使用真正 Held-out：

```text
Tatoeba
Challenge
```

比较：

```text
Original SMaLL-100
Student Exp1
MADLAD-400-3B
```

---

# 20. Tatoeba Benchmark

文件：

```text
data/benchmark/en_uz/
tatoeba_en_uz_500.csv
```

实际有效数据：

```text
429 pairs
```

双向：

```text
858 samples
```

作用：

```text
真实外部 Held-out
跨域泛化测试
```

简单理解：

```text
Tatoeba =
模型遇到训练数据以外的普通真实句子时
整体翻译能力怎么样
```

---

# 21. Challenge Dataset

文件：

```text
data/benchmark/en_uz/
challenge_v1.csv
```

数量：

```text
300 pairs
```

双向：

```text
600 samples
```

包含：

```text
normal       50
entity       50
number       50
negation     50
time_date    50
long         50
```

作用：

```text
专项压力测试
```

简单理解：

```text
Tatoeba
→ 平时会不会翻

Challenge
→ 遇到特殊难点会不会翻车
```

---

# 22. Step 09A：Uzbek Script Normalization

代码：

```text
scripts/pipeline/09a_normalize_madlad_uzbek.py
```

实验发现：

MADLAD EN→UZ 很多结果输出为：

```text
Uzbek Cyrillic
```

但 Reference 多数使用：

```text
Uzbek Latin
```

如果直接算 BLEU：

```text
语义可能正确
但 script 不一致
导致 BLEU 被严重低估
```

因此 Step09A：

```text
Cyrillic
    ↓
Latin Transliteration
    ↓
Unicode NFKC
    ↓
Apostrophe Normalization
    ↓
重新计算 BLEU / chrF++
```

MADLAD EN→UZ：

```text
729 samples

Cyrillic:
710

Rate:
97.39%
```

输出目录：

```text
results/student/small100/
heldout_eval_v1/script_normalized/
```

核心文件：

```text
all_predictions_script_normalized.parquet
```

```text
heldout_summary_script_normalized.csv
```

```text
challenge_summary_script_normalized.csv
```

```text
madlad_raw_vs_latin.csv
```

```text
script_normalization_report.json
```

---

# 23. Exp1 Corrected Held-out

## Tatoeba EN→UZ

```text
BLEU      20.1993
chrF++    53.6659
Exact     10.02%
```

## Tatoeba UZ→EN

```text
BLEU      44.5986
chrF++    58.7955
Exact     21.45%
```

## Challenge EN→UZ

```text
BLEU      48.8614
chrF++    65.9015
Exact     10.33%
```

## Challenge UZ→EN

```text
BLEU      48.1108
chrF++    64.7384
Exact      8.33%
```

这里发现一个重要问题：

```text
Validation EN→UZ    ≈ BLEU 42.68

Challenge EN→UZ     ≈ BLEU 48.86

但是

Tatoeba EN→UZ       ≈ BLEU 20.20
```

说明：

```text
Exp1 对当前训练域已经很好
但 EN→UZ 跨域泛化仍有明显优化空间
```

因此启动：

```text
MADLAD + Qwen + Distillation
```

---

# 24. Step 10A：Distillation Candidate Preparation

代码：

```text
scripts/pipeline/
10a_prepare_distillation_candidates.py
```

输入：

```text
data/splits/en_uz/v1/
train_exp1_bidirectional_v1.parquet
```

原始：

```text
122432
```

再次严格检查：

```text
Validation Leakage
Tatoeba Leakage
Challenge Leakage

Pair Leakage
English Leakage
Uzbek Leakage

Empty
Cyrillic
Duplicate
```

最终 Full Pool：

```text
122390
```

方向完全平衡：

```text
EN→UZ     61195
UZ→EN     61195
```

从中构建 Teacher Pilot：

```text
20000
```

方向：

```text
EN→UZ     10000
UZ→EN     10000
```

质量：

```text
SILVER    19190
GOLD        810
```

输出：

```text
data/distillation/en_uz/v1/
10a_candidates/
```

主要文件：

```text
distillation_candidates_full_v1.parquet
```

```text
distillation_pilot_20k_v1.parquet
```

---

# 25. Step 10B：MADLAD Teacher Generation

代码：

```text
scripts/pipeline/
10b_generate_madlad_teacher.py
```

Teacher：

```text
google/madlad400-3b-mt
```

输入：

```text
data/distillation/en_uz/v1/
10a_candidates/
distillation_pilot_20k_v1.parquet
```

输出：

```text
data/distillation/en_uz/v1/
10b_teacher_generation/
```

主要：

```text
teacher_predictions_20k_v1.parquet
```

```text
teacher_predictions_20k_v1.csv
```

```text
generation_checkpoint.jsonl
```

```text
generation_config_v1.json
```

```text
10b_report_v1.json
```

结果：

```text
Total        20000

EN→UZ        10000
UZ→EN        10000

Empty            0
```

EN→UZ Raw Cyrillic：

```text
9792 / 10000
97.92%
```

平均：

```text
Latency        0.2306 s/sample
Generated      33.87 tokens
```

其中：

```text
teacher_prediction_raw
```

保存 MADLAD 原始结果。

```text
teacher_prediction
```

保存规范化后的结果。

---

# 26. Step 10C：Qwen3-8B Teacher Quality Gate

代码：

```text
scripts/pipeline/
10c_qwen_teacher_quality_gate.py
```

Qwen：

```text
/root/autodl-tmp/models/Qwen3-8B
```

Judge 输入：

```text
SOURCE
+
HUMAN REFERENCE
+
TEACHER TRANSLATION
```

输出：

```text
label:
PASS
MINOR
FAIL
UNCERTAIN
```

以及：

```text
confidence

semantic_equivalent

teacher_usefulness:
HIGH
MEDIUM
LOW
REJECT
```

Error Flags：

```text
omission
addition
mistranslation
number_error
time_error
entity_error
negation_error
```

---

# 27. Step 10C Calibration 500

先审核：

```text
500
```

其中：

```text
EN→UZ 250
UZ→EN 250
```

结果：

```text
PASS       294
MINOR      109
FAIL        97
```

Usefulness：

```text
HIGH       280
MEDIUM     122
REJECT      97
LOW          1
```

Parse：

```text
100%
```

状态：

```text
CALIBRATION_PASS
```

输出：

```text
data/distillation/en_uz/v1/
10c_qwen_quality_gate/calibration_500/
```

---

# 28. Qwen Batch Benchmark

代码：

```text
scripts/pipeline/
10c_benchmark_qwen_judge.py
```

用于测试：

```text
batch 2
batch 3
batch 4
...
batch 24
```

V100 上最终选择：

```text
batch_size = 24
```

测试约：

```text
0.989 sec/sample
1.01 samples/sec
GPU ≈ 20.19 GB
```

---

# 29. Step 10C Full 20K

输出：

```text
data/distillation/en_uz/v1/
10c_qwen_quality_gate/full_20k/
```

关键：

```text
qwen_judge_results.parquet
```

```text
judge_report.json
```

最终结果：

```text
Total        20000

PASS         11611
58.06%

MINOR         4278
21.39%

FAIL          4036
20.18%

UNCERTAIN       75
0.38%
```

方向：

## EN→UZ

```text
PASS        5171
MINOR       1970
FAIL        2792
UNCERTAIN     67
```

## UZ→EN

```text
PASS        6440
MINOR       2308
FAIL        1244
UNCERTAIN      8
```

Teacher usefulness：

```text
HIGH      11179
MEDIUM     4684
REJECT     4111
LOW          26
```

错误：

```text
omission          3399
addition          2996
mistranslation    5434
number_error       618
time_error         619
entity_error      3965
negation_error      92
```

平均 confidence：

```text
0.956
```

Parse Success：

```text
99.62%
```

---

# 30. Step 10D：Clean Distillation Dataset

代码：

```text
scripts/pipeline/
10d_build_distillation_dataset.py
```

最终 V1 Teacher Target 准入规则：

```text
teacher_label == PASS

AND

teacher_usefulness in:
HIGH
MEDIUM

AND

omission == False
addition == False
mistranslation == False
number_error == False
time_error == False
entity_error == False
negation_error == False

AND

teacher != source

AND

No Evaluation Leakage

AND

No Duplicate

AND

EN→UZ Teacher 不含 Cyrillic
```

也就是说：

```text
宁愿少一些 Teacher 数据
也不把有争议的 Teacher Target
加入第一次 KD 实验
```

输入：

```text
data/distillation/en_uz/v1/
10c_qwen_quality_gate/full_20k/
qwen_judge_results.parquet
```

输出：

```text
data/distillation/en_uz/v1/
10d_distillation_dataset/
```

核心：

```text
distillation_teacher_targets_v1.parquet
```

```text
distillation_teacher_targets_v1.csv
```

```text
distillation_rejected_v1.parquet
```

```text
10d_report_v1.json
```

最终：

```text
Input          20000

Clean KD       11251
```

方向：

```text
EN→UZ          4974
UZ→EN          6277
```

Usefulness：

```text
HIGH          10714
MEDIUM          537
```

Teacher == Human Reference：

```text
530
```

Duplicates Removed：

```text
43
```

最终：

```text
READY_FOR_STUDENT_EXP2
```

---

# 31. Step 11A：Exp2 Training Dataset

代码：

```text
scripts/pipeline/
11a_prepare_student_exp2_dataset.py
```

Human Replay：

```text
data/distillation/en_uz/v1/
10a_candidates/
distillation_candidates_full_v1.parquet
```

数量：

```text
122390
```

Teacher：

```text
data/distillation/en_uz/v1/
10d_distillation_dataset/
distillation_teacher_targets_v1.parquet
```

数量：

```text
11251
```

为了让 Teacher 真正提供：

```text
Alternate Translation Target
```

Exp2 V1 排除：

```text
teacher_prediction
==
human_reference
```

共：

```text
530
```

最终 Teacher：

```text
10721
```

组合：

```text
Human Replay      122390

Teacher KD         10721
-------------------------
Combined           133111
```

KD 占比：

```text
8.05%
```

方向：

```text
EN→UZ     66089
UZ→EN     67022
```

Teacher：

```text
HIGH       10184
MEDIUM       537
```

输出：

```text
data/distillation/en_uz/v1/
11a_exp2_training/
```

核心：

```text
exp2_human_replay_v1.parquet
```

```text
exp2_teacher_kd_v1.parquet
```

```text
exp2_train_combined_v1.parquet
```

```text
exp2_train_combined_v1.csv
```

```text
11a_report_v1.json
```

最终：

```text
READY_FOR_EXP2_TRAINING
```

---

# 32. Step 11B：Student Exp2

代码：

```text
scripts/pipeline/
11b_train_student_exp2.py
```

注意：

Exp2 **不是从 Original SMaLL-100 开始**。

而是：

```text
Original SMaLL-100
        ↓
Exp1
        ↓
Exp1 best_model
        ↓
Exp2
```

初始化：

```text
results/student/small100/
exp1_finetune/best_model
```

数据：

```text
data/distillation/en_uz/v1/
11a_exp2_training/
exp2_train_combined_v1.parquet
```

Loss Weight：

```text
HUMAN_REPLAY      1.0
TEACHER HIGH      1.0
TEACHER MEDIUM    0.8
```

参数：

```text
epochs                   2

batch_size               16

gradient_accumulation    2

effective_batch          32

learning_rate            5e-6

warmup_ratio             0.05

max_length               256

FP16                     True
```

学习率比 Exp1 小很多：

```text
Exp1:
3e-5

Exp2:
5e-6
```

因为 Exp2 的目标是：

```text
Refinement
```

而不是重新学习 EN-UZ。

---

# 33. Exp2 Training Result

Exp1 baseline validation loss：

```text
1.1960
```

Exp2：

```text
1.1860
```

Best Epoch：

```text
2
```

Best Source：

```text
EXP2_KD
```

Improvement：

```text
0.0100
```

相对：

```text
≈ 0.84%
```

输出：

```text
results/student/small100/
exp2_distillation_v1/
```

核心：

```text
best_model/
```

```text
training_history.csv
```

```text
training_report.json
```

```text
checkpoints/latest/
```

Exp2 checkpoint 只保留：

```text
latest
```

避免每 epoch 保存完整 optimizer state 占满数据盘。

---

# 34. Step 11C：Exp1 vs Exp2 Validation

代码：

```text
scripts/pipeline/
11c_eval_small100_exp2.py
```

Validation：

```text
data/splits/en_uz/v1/
validation_exp1_bidirectional_v1.parquet
```

与 Exp1 使用完全同一个冻结 Validation。

输出：

```text
results/student/small100/
exp2_distillation_v1/evaluation/
```

主要：

```text
validation_predictions.parquet
```

```text
validation_predictions.csv
```

```text
validation_metrics.json
```

```text
exp1_vs_exp2_validation.json
```

```text
exp1_vs_exp2_validation.csv
```

---

# 35. Validation 最终结果

## EN→UZ

```text
Exp1 BLEU       42.6848
Exp2 BLEU       42.9079

ΔBLEU           +0.2231
```

```text
Exp1 chrF++     64.1272
Exp2 chrF++     64.4336

ΔchrF++         +0.3064
```

Exact：

```text
9.19%
→
9.25%
```

## UZ→EN

```text
Exp1 BLEU       45.1219
Exp2 BLEU       45.1618

ΔBLEU           +0.0399
```

```text
Exp1 chrF++     63.9964
Exp2 chrF++     64.0309

ΔchrF++         +0.0346
```

Exact：

```text
10.09%
→
10.22%
```

平均：

```text
Mean ΔBLEU      +0.1315

Mean ΔchrF++    +0.1705
```

最终：

```text
VALIDATION_PASS_BOTH_DIRECTIONS
```

---

# 36. Step 12：Final Held-out Evaluation

代码：

```text
scripts/pipeline/
12_final_exp1_vs_exp2.py
```

最终只比较：

```text
Exp1
vs
Exp2
```

不再重新跑：

```text
Original
MADLAD
```

Exp1 使用 Step09A 已保存的 corrected prediction：

```text
results/student/small100/
heldout_eval_v1/script_normalized/
all_predictions_script_normalized.parquet
```

Exp2：

```text
results/student/small100/
exp2_distillation_v1/best_model
```

输出：

```text
results/student/small100/
exp2_distillation_v1/
final_heldout_eval/
```

主要：

```text
exp2_predictions.parquet
```

```text
exp2_predictions_normalized.parquet
```

```text
exp1_vs_exp2_heldout.csv
```

```text
exp1_vs_exp2_challenge_categories.csv
```

```text
final_decision.json
```

---

# 37. Final Tatoeba Result

## EN→UZ

Exp1：

```text
BLEU      20.1993
chrF++    53.6659
Exact     10.02%
```

Exp2：

```text
BLEU      20.9903
chrF++    54.3290
Exact     10.02%
```

变化：

```text
ΔBLEU     +0.7909

ΔchrF++   +0.6631
```

这是本次 Distillation 最关键的结果之一。

因为 EN→UZ Tatoeba 原本正是 Exp1 的明显弱项。

---

## UZ→EN

Exp1：

```text
BLEU      44.5986
chrF++    58.7955
Exact     21.45%
```

Exp2：

```text
BLEU      45.0245
chrF++    59.4748
Exact     21.21%
```

变化：

```text
ΔBLEU     +0.4259

ΔchrF++   +0.6793
```

说明：

```text
两个 Tatoeba 方向都提升
```

---

# 38. Final Challenge Result

## EN→UZ

Exp1：

```text
BLEU      48.8614
chrF++    65.9015
Exact     10.33%
```

Exp2：

```text
BLEU      48.4659
chrF++    65.6617
Exact     10.67%
```

变化：

```text
ΔBLEU     -0.3955

ΔchrF++   -0.2398
```

---

## UZ→EN

Exp1：

```text
BLEU      48.1108
chrF++    64.7384
Exact      8.33%
```

Exp2：

```text
BLEU      47.8408
chrF++    64.5439
Exact      9.00%
```

变化：

```text
ΔBLEU     -0.2700

ΔchrF++   -0.1946
```

Challenge 有轻微下降。

但是仍低于预先设定的：

```text
0.5
```

主单元 Regression 容忍阈值。

---

# 39. Challenge Category 分析

Exp2 并不是所有 category 都提升。

表现下降较明显：

```text
entity / EN→UZ

ΔBLEU      -0.8945
ΔchrF++    -0.9309
```

```text
long / UZ→EN

ΔBLEU      -1.1690
ΔchrF++    -0.7771
```

提升较明显：

```text
normal / EN→UZ

ΔBLEU      +1.6215
```

```text
time_date / EN→UZ

ΔBLEU      +0.8953
```

```text
number / UZ→EN

ΔBLEU      +0.6494
ΔchrF++    +0.5833
```

因此更准确的结论是：

```text
Exp2 提高了整体和跨域泛化能力，

但在少数专项能力上出现了轻微能力重分布。
```

---

# 40. Final Decision

Step11C：

```text
VALIDATION_PASS_BOTH_DIRECTIONS
```

Step12：

```text
Tatoeba EN→UZ

ΔBLEU      +0.7909
ΔchrF++    +0.6631
```

四个 Held-out 主单元平均：

```text
Mean ΔBLEU      +0.1378

Mean ΔchrF++    +0.2270
```

最差主单元：

```text
Minimum ΔBLEU      -0.3955

Minimum ΔchrF++    -0.2398
```

均没有超过：

```text
-0.5
```

因此：

```text
No large regression:
True
```

最终：

```text
DECISION:
EXP2_ACCEPT
```

---

# 41. 当前最终模型

当前 EN↔UZ V1 最终模型：

```text
/root/autodl-tmp/fourlang_translation/
results/student/small100/
exp2_distillation_v1/best_model
```

这是当前推荐冻结的：

```text
EN↔UZ Pair-specific Student V1
```

---

# 42. Exp1 保留位置

Exp1 仍建议保留，用于以后 Regression Comparison：

```text
/root/autodl-tmp/fourlang_translation/
results/student/small100/
exp1_finetune/best_model
```

不要删除。

---

# 43. 当前推荐保留的核心数据文件

## Train / Validation

```text
data/splits/en_uz/v1/
train_exp1_bidirectional_v1.parquet
```

```text
data/splits/en_uz/v1/
validation_exp1_bidirectional_v1.parquet
```

---

## Held-out

```text
data/benchmark/en_uz/
tatoeba_en_uz_500.csv
```

```text
data/benchmark/en_uz/
challenge_v1.csv
```

---

## Distillation Candidates

```text
data/distillation/en_uz/v1/
10a_candidates/
distillation_candidates_full_v1.parquet
```

```text
data/distillation/en_uz/v1/
10a_candidates/
distillation_pilot_20k_v1.parquet
```

---

## MADLAD Teacher

```text
data/distillation/en_uz/v1/
10b_teacher_generation/
teacher_predictions_20k_v1.parquet
```

---

## Qwen Quality Gate

```text
data/distillation/en_uz/v1/
10c_qwen_quality_gate/full_20k/
qwen_judge_results.parquet
```

---

## Clean Teacher Targets

```text
data/distillation/en_uz/v1/
10d_distillation_dataset/
distillation_teacher_targets_v1.parquet
```

---

## Exp2 Training Dataset

```text
data/distillation/en_uz/v1/
11a_exp2_training/
exp2_train_combined_v1.parquet
```

---

# 44. 当前完整核心脚本

```text
scripts/pipeline/
```

核心：

```text
07_build_train_validation_split.py

08a_eval_small100_baseline.py
08b_train_small100_exp1.py
08c_eval_small100_exp1.py

09_eval_heldout_en_uz.py
09a_normalize_madlad_uzbek.py

10a_prepare_distillation_candidates.py
10b_generate_madlad_teacher.py

10c_benchmark_qwen_judge.py
10c_qwen_teacher_quality_gate.py

10d_build_distillation_dataset.py

11a_prepare_student_exp2_dataset.py
11b_train_student_exp2.py
11c_eval_small100_exp2.py

12_final_exp1_vs_exp2.py
```

Step01～06 的早期 Python 文件名当前没有重新通过服务器目录核验，因此本实验记录不人为猜测文件名。

对应功能分别为：

```text
01 数据源合并
02 文本标准化
03 去重和规则过滤
04 Risk Routing
05A Qwen Review Preparation
05B Qwen First Review
05C Qwen Second Review
06 Approved Dataset Builder
```

---

# 45. 各脚本作用速查

| Step      | Script                                   | 作用                          |
| --------- | ---------------------------------------- | --------------------------- |
| 01        | 早期脚本名待核验                                 | 汇总多源 EN-UZ 数据               |
| 02        | 早期脚本名待核验                                 | 文本和 Unicode 标准化             |
| 03        | 早期脚本名待核验                                 | 去重、规则过滤、leak 检查             |
| 04        | 早期脚本名待核验                                 | AUTO_ACCEPT / NEEDS_QWEN    |
| 05A       | 早期脚本名待核验                                 | Qwen 审核数据准备                 |
| 05B       | 早期脚本名待核验                                 | Qwen 一审                     |
| 05C       | 早期脚本名待核验                                 | Qwen 二审                     |
| 06        | 早期脚本名待核验                                 | Approved Dataset            |
| 07        | `07_build_train_validation_split.py`     | Train / Validation 严格切分     |
| 08A       | `08a_eval_small100_baseline.py`          | Original SMaLL-100 Baseline |
| 08B       | `08b_train_small100_exp1.py`             | Human Fine-tuning           |
| 08C       | `08c_eval_small100_exp1.py`              | Exp1 Validation             |
| 09        | `09_eval_heldout_en_uz.py`               | Tatoeba + Challenge         |
| 09A       | `09a_normalize_madlad_uzbek.py`          | Uzbek Script Normalization  |
| 10A       | `10a_prepare_distillation_candidates.py` | Distillation Candidate Pool |
| 10B       | `10b_generate_madlad_teacher.py`         | MADLAD Teacher              |
| 10C-Bench | `10c_benchmark_qwen_judge.py`            | Qwen batch 压测               |
| 10C       | `10c_qwen_teacher_quality_gate.py`       | Teacher Quality Gate        |
| 10D       | `10d_build_distillation_dataset.py`      | Clean KD Dataset            |
| 11A       | `11a_prepare_student_exp2_dataset.py`    | Human Replay + KD           |
| 11B       | `11b_train_student_exp2.py`              | Student Exp2                |
| 11C       | `11c_eval_small100_exp2.py`              | Exp1 vs Exp2 Validation     |
| 12        | `12_final_exp1_vs_exp2.py`               | Final Held-out Decision     |

---

# 46. 可删除的数据

如果确认：

```text
Exp2 不再继续恢复训练
```

则可以删除：

```text
results/student/small100/
exp2_distillation_v1/checkpoints/latest/
```

因为：

```text
best_model/
```

已经存在。

但是必须保留：

```text
exp2_distillation_v1/best_model/
```

以及：

```text
final_heldout_eval/
```

---

# 47. EN↔UZ 当前是否还需要继续训练？

当前建议：

```text
NO
```

冻结：

```text
EN-UZ Exp2 V1
```

不要为了继续提高：

```text
0.x BLEU
```

反复针对当前 benchmark 调参数。

否则容易出现：

```text
Test-set overfitting
```

---

# 48. 什么情况下重新开启 EN↔UZ V2？

建议只有以下情况才重新优化：

```text
真实 App 用户反馈
出现稳定系统性翻译问题
```

或者：

```text
entity EN→UZ
long UZ→EN
```

被真实业务验证为重要弱点。

或者：

```text
ONNX / INT8
量化后 BLEU / chrF++
明显下降
```

或者未来获得：

```text
新的高质量 EN-UZ 数据
```

再启动：

```text
EN-UZ V2
```

---

# 49. 下一阶段语言对

EN↔UZ 冻结后，开始：

```text
ZH ↔ EN
```

然后：

```text
RU ↔ EN
```

然后：

```text
ZH ↔ RU
```

然后：

```text
ZH ↔ UZ
```

最后：

```text
RU ↔ UZ
```

---

# 50. 后续语言对不默认跑完整 Distillation

EN-UZ 实验已经证明了一件非常重要的事情：

```text
MADLAD
+
Qwen
+
Distillation
```

是有效的。

但是它不应该成为所有语言对的默认流程。

以后应该：

```text
Original SMaLL-100
        ↓
Baseline
        ↓
质量足够？
   ┌────┴─────┐
   │          │
  YES        NO
   │          │
冻结       Human Exp1
              ↓
          再评估
              ↓
         仍存在明显问题？
          ┌───┴───┐
          │       │
         NO      YES
          │       │
        Freeze    ↓
             MADLAD
                +
              Qwen
                +
               KD
```

因此：

```text
Teacher + Judge + KD
```

属于：

```text
Weak Direction Enhancement Pipeline
```

而不是所有语言方向无条件执行。

---

# 51. 最终四语模型路线

当前 EN↔UZ Exp2：

```text
不是最终整个 App 唯一模型
```

它现在是：

```text
Pair-specific Best Model
```

后续所有语言方向完成后，再考虑：

```text
EN↔UZ Best Knowledge
ZH↔EN Best Knowledge
RU↔EN Best Knowledge
ZH↔RU Best Knowledge
ZH↔UZ Best Knowledge
RU↔UZ Best Knowledge
        ↓
Multilingual Consolidation
        ↓
FourLang Student V1
        ↓
12 Direction Regression Test
        ↓
ONNX
        ↓
INT8
        ↓
Mobile Deployment
```

---

# 52. 最终实验结论

当前 EN↔UZ Pipeline 已完整跑通：

```text
Original SMaLL-100
        ↓
Human Parallel Data
        ↓
Student Exp1
        ↓
Held-out Test
        ↓
发现 EN→UZ 泛化弱点
        ↓
MADLAD Teacher
        ↓
Qwen3-8B Quality Gate
        ↓
Clean Teacher Targets
        ↓
Human Replay
+
Sequence-level KD
        ↓
Student Exp2
        ↓
Frozen Validation
        ↓
Tatoeba
+
Challenge
        ↓
EXP2_ACCEPT
```

最终模型：

```text
/root/autodl-tmp/fourlang_translation/
results/student/small100/
exp2_distillation_v1/best_model
```

最终核心结果：

```text
Validation:
两个方向均提升
```

```text
Tatoeba:
两个方向均提升
```

尤其：

```text
Tatoeba EN→UZ

BLEU:
20.1993
→
20.9903

+0.7909
```

```text
chrF++:
53.6659
→
54.3290

+0.6631
```

Challenge：

```text
轻微下降
但整体处于可接受 Regression 范围
```

四个 Held-out 主单元平均：

```text
Mean ΔBLEU      +0.1378
Mean ΔchrF++    +0.2270
```

最终：

```text
EXP2_ACCEPT
```

因此：

> EN↔UZ Pair-specific Translation Model V1 已完成，可以冻结 Exp2，并开始其他语言对的 Baseline Pipeline。

````

你现在直接把上面代码块内的内容保存到：

```text
D:\dev\projects\fourlang_translation\docs\EN_UZ_Pipeline_V1_Experiment_Summary.md
````

然后建议提交 Git：

```powershell
git add docs/EN_UZ_Pipeline_V1_Experiment_Summary.md
git commit -m "docs: add en-uz pipeline v1 experiment summary"
git push
```

这样 EN↔UZ 这一阶段的实验记录就正式归档完成了。下一步可以直接开始 **ZH↔EN Baseline Pipeline**。
