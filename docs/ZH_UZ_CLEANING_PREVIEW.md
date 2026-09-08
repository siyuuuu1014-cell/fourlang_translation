# 中乌清洗预览（不生成正式训练集）

## 本次范围

复用已经完成的 zh_uz_training_quality_v2，不重做 Teacher、Qwen、训练或模型评测。
先验证 v2 的清单、完成标记、产物哈希及原始输入文件未改变，再重新计算规则提示。
修正 g'alaba 的 g 被当成克的误报，支持 foiz、foizdan、foizi 等明确的百分比形式。
不覆盖 v1/v2 报告，不删除、改写、重采样或重新加权任何训练样本。

|分类|含义|是否自动删除|
|---|---|---|
|ISOLATE_CANDIDATE|此前助手逐句初审发现明确问题、登记了完整 audit_id 的建议隔离候选|否|
|REVIEW_REQUIRED|其他规则疑点、MINOR、无法确认的 Teacher 元数据或未知来源角色|否|
|KEEP_CANDIDATE|当前有限规则与已登记清单没有发现问题，暂时保留|否|

三类互斥且覆盖全部审查记录；分别统计不同记录数与训练出现次数。
当前数据每方向三类出现次数应合计 20,000。脚本按 v2 记录的数量校验，不凭固定数值补齐。
原样本的文本、权重、重复次数、历史标签保留；新增分类字段不构成训练许可。
KEEP_CANDIDATE 不是“翻译正确”，NO_SUPPORTED_MISMATCH_FOUND 也不是质量通过。
数字文字全拼、语义漏译和复杂单位仍需要人工复核，不能把所有提示等同于错误。

登记的 13 条候选来自先前 180 条审查包的助手初审，不是母语审校员认证，也不是总体错误数量。
配置只保存完整 audit_id 和理由，不复制完整原文译文。
无精确匹配的登记项会列入 unmatched_case_ids，不能模糊匹配或按相近文本扩大隔离范围。
只处理这 13 条也不能保证模型改善；需要结合待复核队列及后续受控验证。

## 本地同步

PowerShell，提交仅限本次六个文件。现有其他改动不纳入提交。

```powershell
cd D:\dev\projects\fourlang_translation
$previewFiles = @(
  "scripts/pipeline_v3/quality_checks.py",
  "scripts/pipeline_v3/preview_zh_uz_cleaning.py",
  "configs/review/zh_uz_isolation_candidates_v1.json",
  "tests/test_quality_checks.py",
  "tests/test_cleaning_preview.py",
  "docs/ZH_UZ_CLEANING_PREVIEW.md"
)
git add -- $previewFiles
if ($LASTEXITCODE -ne 0) { throw "暂存失败，请停止" }
git diff --cached --name-only -- $previewFiles
git commit --only -m "Fix unit hints and add non-destructive cleaning preview" -- $previewFiles
if ($LASTEXITCODE -ne 0) { throw "提交失败，请停止" }
git push origin main
```

推送成功后才继续。遇到 TLS 错误不要关闭证书验证，遇到分支或文件冲突不要强制重置。

## 服务器运行

```bash
cd /root/autodl-tmp/fourlang_translation
git pull --ff-only origin main
```

拉取成功后，执行下段。这里运行的是新的 preview 脚本，不要在旧 v2 目录重跑修改后的 audit 脚本。

```bash
nohup env PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="" \
  /root/autodl-tmp/venvs/small100_student/bin/python \
  scripts/pipeline_v3/preview_zh_uz_cleaning.py \
  --audit reports/diagnostics/fourlang/zh_uz_training_quality_v2 \
  --output reports/diagnostics/fourlang/zh_uz_cleaning_preview_v1 \
  >> fourlang_zh_uz_cleaning_preview_v1.log 2>&1 &
echo $!
tail -n 30 -f fourlang_zh_uz_cleaning_preview_v1.log
```

只使用 CPU，不加载模型；看到 Cleaning preview ready 表示预览生成完成，不代表数据验收或训练完成。
Ctrl+C 退出日志查看，不会停止上面的后台进程。若执行失败，先查看日志，不要重新 aggregate 或删除原文件。
相同输入与代码可重复运行，已生成结果内容一致才复用；代码、输入或候选清单变化须换新的输出目录。
目录锁阻止重复写入；生成文件被手动改动后会拒绝覆盖。

## 交回结果

完成状态和分类数量可用以下命令查看：

```bash
python -m json.tool reports/diagnostics/fourlang/zh_uz_cleaning_preview_v1/done.json
python -m json.tool reports/diagnostics/fourlang/zh_uz_cleaning_preview_v1/cleaning_preview.json
```

下载并上传下面这个文件即可。它包含统计、所有命中的隔离候选，以及每方向另外两类各最多 20 条固定抽样。
抽样不用于估计总体错误率。建议下载后保持文件名 cleaning_preview_packet.json，避免与旧 review_packet.json 混淆。

```text
/root/autodl-tmp/fourlang_translation/reports/diagnostics/fourlang/zh_uz_cleaning_preview_v1/cleaning_preview_packet.json
```

上面是文件位置，不是终端命令。同目录还保留三份完整清单：

- isolate_candidates.jsonl
- review_required.jsonl
- keep_candidates.jsonl

这些是审查清单，不能当作正式 train.jsonl 使用。它们按不同审查记录存储，重复次数只是 occurrences 字段。
所有 human_confirmation 字段均为空，不把助手建议伪装成人工确认。
若需填写审查意见，另存副本，不编辑原生成文件。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cleaning_preview.py tests/test_quality_checks.py tests/test_judge_metadata_recovery.py tests/test_zh_uz_training_quality.py tests/test_zh_uz_diagnostics.py tests/test_fourlang_pipeline.py tests/test_training_optimizations.py tests/test_supplemental_cleaning.py -q
.\.venv\Scripts\python.exe -m ruff check scripts/pipeline_v3/quality_checks.py scripts/pipeline_v3/preview_zh_uz_cleaning.py tests/test_quality_checks.py tests/test_cleaning_preview.py
```

验证包含误报正反例、三类数量守恒、重复数与权重校验、候选精确匹配、输入变更阻止发布、人工意见保护及审查入口禁止加载模型。
本地只有上传的抽样包，没有服务器的完整 v2 目录和训练文件。完整数量需在服务器运行后确认。
