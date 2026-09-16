# 当前状态与下一步交接（2026-09-16 v2）

> 给下一个会话的快速上手。顺序：先读本文，再看 `PROJECT_HANDOFF.md`（历史背景）、
> `docs/CODE_WALKTHROUGH_GUIDE.md`（训练代码走读）。

## 一句话现状

- **六组独立 SMaLL-100 双向模型（MIT）** 覆盖 12 方向，已用 FastAPI 对外服务（supervisord 常驻）。
- **四语模型 = m2m100_418M（MIT）**，已导出 **int8 ONNX 移动端包（734MB，2 文件，带 KV cache）**，推理已验证正确。
- 目标：六组模型走**服务器 API**；四语模型**量化包塞进手机 App**（App 在另一个项目）。

## 当前模型（权威路径）

### 六组独立模型（12 方向）

机器映射：`configs/specialists/current_pair_models.json`（权威，别用旧的 `models/model_registry.json`）。

| pair | 模型 |
|---|---|
| en_zh | results/student/pair_specialists/en_zh/exp2/best_model/shared |
| en_uz | models/final_specialists/en_uz_small100_v1 |
| en_ru | models/final_pair_specialists/en_ru_v1 |
| zh_uz | results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared |
| zh_ru | results/student/pair_specialists/zh_ru/exp2/best_model/shared |
| uz_ru | results/student/pair_specialists/uz_ru/exp2/best_model/shared |

### 四语模型（当前 = m2m100）

- 权重：`results/student/fourlang_m2m100/m2m100_targeted_from_human_v1/best_model/shared`（MIT，m2m100_418M，484M 参数）
- 移动端包：`onnx_export/m2m100_fourlang_mobile/`（**734MB，int8，2 文件，KV cache 保留**）
  - `encoder_model.onnx` 287MB + `decoder_model_merged.onnx` 474MB + 分词器 ~8MB
  - 内含 `README.md`（集成指南 + 语言 token id 表 + 解码伪代码）+ `MANIFEST.json`
- 旧 NLLB 四语：`results/student/fourlang/`（exp1/exp2/exp3_v2，CC-BY-NC，已弃用）

## 关键代码

- `inference/pair_service.py` —— 六组模型懒加载服务类（FastAPI 和 CLI 共用）
- `scripts/pipeline_v3/translate_pair_specialists.py` —— 六组 CLI
- `scripts/pipeline_v3/translate_m2m100_fourlang.py` —— 四语 CLI
- `service/app.py` / `dependencies.py` / `schemas.py` —— FastAPI（六组）
- `service/supervisord_api.conf` + `scripts/service/manage_api.sh` —— 常驻服务
- `docs/deployment/ONNX_MOBILE_PACKAGE.md` + `MANIFEST_m2m100_mobile.json` —— 移动端包文档

## 部署状态

- FastAPI：`http://<server>:8000`，supervisord 管理。
  - `POST /translate` body `{source_lang, target_lang, text}`；管理 `bash scripts/service/manage_api.sh {start|stop|status}`
  - **实例重启后需手动 `start`**
- 移动端包：`onnx_export/m2m100_fourlang_mobile/`（scp 可拉到 App 端）

## 对 App 端 5 个关键点的结论（已定）

1. **ONNX 在服务器**，不在 App 项目里。下载：
   `scp -P 28893 -r root@region-46.seetacloud.com:/root/autodl-tmp/fourlang_translation/onnx_export/m2m100_fourlang_mobile/ ./`
2. **ORT 冲突**：复用 sherpa 已有的 onnxruntime 1.27.1 的 Java 桥 + libonnxruntime4j_jni.so。导出用标准算子，1.27.1 能跑。
3. **体积/内存**：最终包压到 **734MB（磁盘）**，运行时内存 ~1GB（单实例）；**必须限制单 session 并发**（多实例会压垮手机）。磁盘 734MB 仍偏大，建议"首次下载"而非打进 APK；再小只能 int4（掉点）或蒸馏。
4. **分词器**：SentencePiece BPE（128000 pieces，无 byte fallback）。用 `sentencepiece.bpe.model`（已含合并规则），别用 vocab.json；`vocab.txt`（piece+score）已提取供 Kotlin BPE 参考。
5. **六组模型**：现状只有 FastAPI 在线（SMaLL-100 自定义 Python 分词器，ONNX 导出更麻烦），短期先调 `POST :8000/translate`，离线包二期。

## 下一步（按优先级）

1. **App 端接入**：六组走 API；四语走 `m2m100_fourlang_mobile/` 包（`onnxruntime-mobile` + 自己写解码循环，见 README）。
2. 出 **Python 版 merged 解码参考实现**（便于 App 端转 Swift/Kotlin）。
3. **git 提交**（本会话改动均未提交）。
4. 训练数据许可核对（商用前，见 `docs/PIPELINE_RUNBOOK.md`）。

## 隐患 / 注意

- **onnxruntime 图优化必须用 `ORT_ENABLE_BASIC`**，不能 EXTENDED（1.30 的 `TransposeDQWeightsForMatMulNBits` 对跨子图 int8 权重有 bug，README 已写清）。
- protobuf 4.25.3 → 7.36.1（onnx 依赖），核心依赖 import 正常。
- 磁盘 205G/235G，可用 ~31G。
- 服务器 SSH：`region-46.seetacloud.com:28893`，凭据在用户处，不写入文档。
- 未经用户确认，不删除/重训/改模型路由。
