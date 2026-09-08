# 中乌训练数据质量审查与小规模对照草案

## 当前做什么

先检查实际 Exp2 训练数据，不训练、不重新推理、不重新调用 Qwen。
本地没有服务器上的真实训练数据，因此本地测试通过不等于真实数据已审核。
审查入口只写入独立的 reports/diagnostics 子目录，不修改训练配置、模型、训练集或已有评测。
当前版本为 v2，默认输出 zh_uz_training_quality_v2；已有 v1 报告保留，不在原目录重跑新代码。

本次需同步的八个文件（含新增和修改）：

- scripts/pipeline_v3/audit_zh_uz_training_quality.py
- scripts/pipeline_v3/quality_checks.py
- scripts/pipeline_v3/judge_metadata_recovery.py
- scripts/supplemental/import_pair_v2.py
- tests/test_zh_uz_training_quality.py
- tests/test_quality_checks.py
- tests/test_judge_metadata_recovery.py
- 本文档

已有 diagnose_zh_uz.py 及其训练数据校验函数被复用，但没有修改，旧推理断点不失效。

## 同步到服务器

在本地 PowerShell 执行。以下提交只包含本次八个文件，不使用 git add .；只有上一步成功才执行下一步：

```powershell
cd D:\dev\projects\fourlang_translation
$qualityFiles = @(
  "scripts/pipeline_v3/audit_zh_uz_training_quality.py",
  "scripts/pipeline_v3/quality_checks.py",
  "scripts/pipeline_v3/judge_metadata_recovery.py",
  "scripts/supplemental/import_pair_v2.py",
  "tests/test_zh_uz_training_quality.py",
  "tests/test_quality_checks.py",
  "tests/test_judge_metadata_recovery.py",
  "docs/ZH_UZ_TRAINING_QUALITY_REVIEW.md"
)
git add -- $qualityFiles
if ($LASTEXITCODE -ne 0) { throw "暂存失败，请停止并检查错误" }
git diff --cached --name-only -- $qualityFiles
git commit --only -m "Harden quality hints and trace saved Teacher judgments" -- $qualityFiles
if ($LASTEXITCODE -ne 0) { throw "提交失败，请停止并检查错误" }
git push origin main
```

先确认检查输出为本次路径。若已有其他暂存内容，保留它们；带 --only 的提交不纳入其他路径。
仅在提交和推送成功后继续服务器操作。如果 TLS 或网络报错，不关闭证书验证、不强制重置；保留错误信息排查。

服务器终端：

```bash
cd /root/autodl-tmp/fourlang_translation
git pull --ff-only origin main
```

只有拉取成功才运行下一段。发生本地改动冲突或分支分歧时停下，不覆盖服务器文件。

## 一次运行只读审查

```bash
cd /root/autodl-tmp/fourlang_translation
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \
  FOURLANG_MODEL_ROOT=/root/autodl-tmp/models \
  /root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline_v3/audit_zh_uz_training_quality.py \
  --config configs/multilingual/fourlang.toml \
  --output reports/diagnostics/fourlang/zh_uz_training_quality_v2 \
  >> fourlang_zh_uz_training_quality_v2.log 2>&1 &
echo $!
tail -n 30 -f fourlang_zh_uz_training_quality_v2.log
```

数据读取和规范化使用 CPU。不会加载 Teacher、Student 或 Qwen 权重，不需要新增依赖。
看到 Review packet ready 表示审查包生成结束，不代表通过质量审核。
Ctrl+C 只退出这里的日志查看。可用下面的只读命令检查审查进程是否存在：

```bash
ps -eo pid,etime,pcpu,pmem,args | grep '[a]udit_zh_uz_training_quality.py'
```

若进程确已退出且没有完成，重新运行同一命令会重新计算 CPU 审查，并校验/复用一致的报告。
没有逐条推理断点，因为没有推理；中途中断的 CPU 计算会重做。
不要同时启动两个审查；目录锁会阻止重复写入。

## 交回结果

简短完成状态查看命令：

```bash
python -m json.tool reports/diagnostics/fourlang/zh_uz_training_quality_v2/done.json
```

完整内容请用 JupyterLab 文件管理器下载，再上传到对话，不要用 cat 粘贴长文件。
需要下载的文件位置（不是可执行命令）：

```text
/root/autodl-tmp/fourlang_translation/reports/diagnostics/fourlang/zh_uz_training_quality_v2/review_packet.json
```

只需上传 review_packet.json；它包含统计、抽样来源及抽到的训练原文/译文。
若需要逐条核查恢复证据，再附上同目录的 metadata_recovery.json。该文件保存当前 KD 池的 Teacher 记录恢复前后标签、证据文件及从 0 开始的行号；统计不是实际重复采样次数。
不要把整个训练集或模型上传。

## v2 的检查与元数据恢复

- 共用粤语检查规则，补上带上下文的“而家”；“然而家人”“而家长”这类普通话反例不会因新增规则被标记。
- 审查入口对两边中文都给出 non_mandarin_review 提示，不自动删数据。import_pair_v2 的既有普通话导入门禁使用同一规则；本次不运行重新导入。
- 数字检查支持阿拉伯数字、千位分隔、万/亿/ming/million 等显式倍率，并保留重复次数；“2万/20000”“5000万/50 million”不会仅因写法不同被标记。
- 对紧邻数字的已知长度、面积、质量、时间单位提供 number_unit_review。单位先折算为基准量再比较，容许 0.5% 的舍入差异；这是疑点检查，不是工程或计量验收精度。
- 中文、乌语数字全拼、日期、不同地区的小数分隔符、未列出的单位等尚不能完整解析。七到八与 7–8 仍可能进入人工复核；未标记不能证明译文正确，也不会自动更正单位或数字。
- 每条记录的 numeric_evidence 保存提取到的数值、单位、比较状态及限制；盲审文件不展示提示。

默认尝试以下已保存的全量 Teacher 审核文件（只读取，不重新调用 Qwen）：

```text
data/pipeline_v2/zh_uz/teacher_judged.parquet
data/pipeline_v2/zh_uz_v2/teacher_judged.parquet
data/pipeline_v2/zh_uz_v3/teacher_judged.parquet
```

缺失的默认文件会写入日志和 input_inventory，不中止整个审查，也不凭配置猜标签。
若真实文件放在其他位置，可重复传入 --judge-metadata 指定全部要检查的文件；这会替代默认列表。显式指定却缺失的文件会报错。
输入必须包含方向、原文、teacher_text、judge_label、judge_parse_ok；不能拿人工参考文本冒充 Teacher 输出。

追溯使用现有语言规范化后的“方向 + 完整原文 + 完整 Teacher 译文”，不模糊匹配、不只靠 pair_id 或权重。
所有匹配证据必须可解析且已知字段一致，才在审计副本中补缺失字段；重复一致证据保留全部位置，冲突不任选其一。
不覆盖已有字段；发现已有标签与证据冲突时记录 CONFLICT，并移出 known_pass_candidate 候选统计。
某一字段缺失就保持 UNKNOWN，不能从候选模型配置猜 teacher_id，也不能将低权重当 MINOR。
恢复到 PASS 只说明存在一致的历史自动审核记录，不等于此次质量验收通过。

恢复状态包括 RECOVERED、CONSISTENT_NO_FILL、NO_MATCH、CONFLICT、PARSE_UNVERIFIED、LABEL_UNVERIFIED、USEFULNESS_UNVERIFIED。
所有存在的证据文件纳入输入哈希；缺失文件后来出现、文件被改动或参数变化，都需要新的输出目录，不能覆盖旧审查。

## 报告的边界

- 验证 train/validation 内容指纹仍与已完成 Exp2 一致，核对保存的验证子集与运行记录；数据变化会报错，不会重新 aggregate。
- 检查读取前后输入文件哈希；输入、代码或参数变化需换新输出目录，不覆盖旧结果。
- 按文本、方向、training_source、weight 回溯当前 KD 元数据。标签缺失是 UNKNOWN，冲突是 AMBIGUOUS，不凭低权重推断 MINOR。
- 分开统计去重记录数与训练实际重复条数，避免重复采样放大导致误判。
- 疑点包括长串重复、数字集合差异、文字混杂、原文照抄、与同源人工作为标签的数据相比异常的长度。它们都不是自动错误判定：数字可用文字表达，外文专名可合法，短译文不必然漏译。
- 参考候选仅来自当前 KD 文件中同方向、同原文、training_source 以 human 开头的训练行。它们不保证是正确参考；不使用验证集作为训练数据参考池。
- 单语蒸馏来源往往没有人工参考，这是正常情况。不能拿 Teacher 输出自证正确，也不会据此自动剔除。
- 不做自动术语/语义审核；仅在已保存、可解析、精确匹配且无冲突的审核产物中追溯标签，不重建缺失的 Qwen 输出。
- known_pass_candidate 仅表示审计元数据为 MATCHED、PASS、HIGH/MEDIUM、Teacher ID 已知，且没有追溯冲突或解析异常；不是“质量合格”或“胜过 Exp1”。
- 每方向每个标签层固定抽最多 20 条不同记录，另补充最多 20 条带规则疑点的记录。是分层审查，不是整体按重复次数加权的随机样本，不能直接把审查错误率当成总体错误率。
- review_blind.jsonl 隐藏旧审核标签、Teacher/Human 来源及规则疑点，供先行盲审；对应元数据在 review_packet.json 中。若已经打开元数据，不应再声称盲审。
- 不覆盖人工意见；如需人工填写，另存 review_blind.jsonl 的副本。原生成文件被修改后，同目录重跑会拒绝覆盖。
- 输出 .gitignore 防止语料误提交到 Git；下载的审查包仍含文本，分享前注意授权范围。

## 下一次小规模对照：草案，尚未执行/冻结

先审阅真实数据包。若发现确切的来源错误、标签串行或训练数据错误，先处理该问题，不能把错误数据直接用来验证学习率猜想。

若没有发现必须先修复的构建错误，可优先验证一个因素：中乌 Teacher 监督的相对权重。
不同时更改样本量、筛选标签、学习率和训练轮次。

|项目|A 对照组|B 试验组|
|---|---|---|
|起点|同一个已保存的 Exp1 模型|完全相同|
|训练行及顺序|同一份冻结的 158,000 条、同一抽样/打乱种子|完全相同|
|中→乌、乌→中 Teacher 行权重|原权重 × 1.0|原权重 × 0.5|
|人工数据和其他 10 个方向的权重|原值|完全相同|
|优化器/学习率等|同一设置；建议沿用原 Exp2 的 5e-6 起始设定|完全相同|
|先行预算|固定 1,000 个 optimizer steps|固定 1,000 个 optimizer steps|
|评估|step 0、500、1000；固定相同 12 方向验证样本|完全相同|
|比较终点|固定 step 1000|固定 step 1000|

这个方案只改变一个“监督相对权重”参数。原损失在 batch 内归一化，×0.5 不等于实际梯度贡献恰好减半。
它测试的是权重策略，不是单独的 MINOR 因果效果，不证明更少 KD 必然更好。
学习率调度/预热按相同 1,000 步预算设置，两组一致；不能把短跑结果直接与历史 9,876 步的 Exp2 当作同预算对照。

执行前还需实现并核对两组独立输出目录、模型/数据指纹、优化器断点和除目标权重外的完全一致性。
本次没有提供训练启动入口或改写现有训练配置，防止未经审查就启动试验。

评测约定应在开训前冻结：

1. 原 BLEU/chrF2++ 保留为主报告，旁路同时记录纯字符 chrF2 和统一标点后的结果；不能事后挑有利指标。
2. 同样检查其他 10 个方向，不仅看中乌均值；先行小样本不构成正式晋级依据。
3. 若差异落在抽样噪声内，不宣称获益；单一种子的短跑仅用来筛查方向。
4. 不用 FLORES devtest 选参数、挑 checkpoint、筛训练样本，也不改变既有正式晋级门槛。
5. 若前期结果值得继续，另行确认完整训练预算和成功标准；否则保留两份原模型及所有诊断记录。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_quality_checks.py tests/test_judge_metadata_recovery.py tests/test_zh_uz_training_quality.py tests/test_zh_uz_diagnostics.py tests/test_fourlang_pipeline.py tests/test_training_optimizations.py tests/test_supplemental_cleaning.py -q
.\.venv\Scripts\python.exe -m ruff check scripts/pipeline_v3/audit_zh_uz_training_quality.py scripts/pipeline_v3/quality_checks.py scripts/pipeline_v3/judge_metadata_recovery.py scripts/supplemental/import_pair_v2.py tests/test_zh_uz_training_quality.py tests/test_quality_checks.py tests/test_judge_metadata_recovery.py
```

测试使用合成数据，覆盖计数、未知标签/冲突、同源参考连接、确定性抽样、禁止审查入口加载模型、输入变更拒绝和人工文件保护；新增普通话正反例、数值/单位转换、审核解析状态、证据文件哈希与恢复不改原数据的测试。
