# 四语翻译 + ASR 对接交付包

本目录包含「语音识别（ASR）+ 翻译」接入一个 App 所需的全部东西：翻译模型、对接文档、可复用参考代码、ASR 引擎模型。OCR 尚未确定，暂不包含。

---

## 一、目录结构

```
delivery/
├── m2m100_fourlang_mobile.tar      # 四语翻译模型（int8 量化，12 方向，734MB）
├── APP_INTEGRATION_GUIDE.md        # 翻译模型对接文档（token id / 解码契约 / 接口）
├── README_交付说明.md              # 本文件
├── reference_code/                 # 可复用 Kotlin 参考代码（从测试 App 抽取）
│   ├── translation/                # ★ 翻译模块（本地 ONNX 引擎 + 下载管理）
│   ├── asr/                        # ★ ASR 模块（sherpa / vosk 封装）
│   ├── pipeline/                   # ★ 管线（Speaker → LID → ASR → 回填）
│   ├── model/                      # 共享数据类（AppLanguage / AsrResult）
│   ├── lid/                        # 语言识别
│   ├── speaker/                    # 说话人识别
│   ├── vad/                        # 端点检测
│   ├── audio/                      # 录音
│   └── util/                       # 资源加载 / 内存统计
└── asr_models/                     # ASR 模型文件（第三方，394MB）
    ├── models/en|uz|zh|ru          # 四语 ASR 模型
    ├── silero_vad.onnx             # VAD
    ├── speaker_embedding.onnx       # 说话人
    └── voxlingua107_float32.tflite  # 语言识别（LID）
```

---

## 二、各部分说明

### 1. 翻译模型包（我们训练侧产出，核心交付物）

`m2m100_fourlang_mobile.tar`，解压后是 `m2m100_fourlang_mobile/` 目录：

```
m2m100_fourlang_mobile/
├── MANIFEST.json       # 模型元数据（词表大小/层数/语言 token id/方向机制）
├── encoder_model.onnx  # encoder
├── decoder_model_merged.onnx  # merged decoder（含 KV cache）
├── sentencepiece.bpe.model    # BPE 分词器
└── README.md
```

- 覆盖 4 语 12 方向（zh/en/ru/uz 任意两两）。
- 已 int8 量化，双文件结构（encoder + merged decoder），移动端可直接跑。

### 2. 对接文档

`APP_INTEGRATION_GUIDE.md` 里有完整契约，关键几点：

| 项 | 值 |
|---|---|
| onnxruntime 图优化 | 必须 `BASIC`（EXTENDED 有 int8 权重 bug） |
| 语言 token id | en=128022 / zh=128102 / uz=128096 / ru=128077 |
| 特殊 token | bos=0 / eos=2 / pad=1 |
| 方向机制 | M2M100 式：`[src_lang] + X + [eos]`，首步 `[eos, tgt_lang]` |
| 解码 | greedy + KV cache |

### 3. 参考代码（可直接复用）

`reference_code/` 是从测试 App 抽出来的、已经跑通的 Kotlin 代码。**同事复制后改包名即可用**，核心是三个模块：

- **`translation/`** —— 翻译的完整实现：
  - `engine/KitTranslationEngine.kt`：本地 ONNX 解码（已按上面的契约实现，greedy + KV cache + 循环复读检测）
  - `tokenizer/M2m100Tokenizer.kt`：Kotlin 版 BPE 分词器
  - `TranslationService.kt`：路由（本地专项 → 本地总模型 → 服务器 API 兜底）+ 单实例缓存 + 版本检测
  - `download/` + `model/`：模型下载、解压、清单同步、SHA 校验
- **`asr/`** —— ASR 的完整实现：
  - `AsrEngine.kt`：统一接口 `transcribe(samples, sampleRate)`
  - `ChineseSherpaEngine.kt` / `EnglishSherpaEngine.kt` / `UzbekSherpaEngine.kt`：sherpa-onnx 封装
  - `RussianVoskEngine.kt`：Vosk 封装
  - `AsrManager.kt`：引擎路由
- **`pipeline/ConversationPipeline.kt`** —— 完整管线样板（Speaker → LID → 语种路由 → ASR → code-switch 回填）

---

## 三、ASR → 翻译 的对接（核心一步）

两边已经天然对齐，**语言枚举是同一个 `AppLanguage`（zh/en/ru/uz）**，所以对接就是一行调用：

```kotlin
// ASR 管线输出
val pipelineResult: PipelineResult = conversationPipeline.processSegment(...)
//   pipelineResult.displayText      -> 识别出的文字
//   pipelineResult.routedLanguage   -> 识别出的源语言 (AppLanguage)

// 翻译（目标语言由用户选择）
val translated: TranslationResult = translationService.translate(
    text   = pipelineResult.displayText,
    source = pipelineResult.routedLanguage,
    target = targetLanguage,   // AppLanguage，用户选
)
```

`TranslationResult` 里拿 `outputText` 就是译文。

---

## 四、依赖（同事需在 build.gradle 加）

| 库 | 版本 | 用途 |
|---|---|---|
| `ai.onnxruntime:onnxruntime-android` | 1.27.1 | 翻译 + sherpa 共用（sherpa AAR 自带 .so） |
| `com.k2fsa.sherpa.onnx:sherpa-onnx` | 1.13.7 | 中/英/乌 ASR |
| `com.alphacephei:vosk-android` | 0.3.75 | 俄语 ASR |
| sentencepiece（可选） | — | 由 `M2m100Tokenizer` 自实现，无需额外引入 |

---

## 五、待办 / 未确定

- **OCR**：尚未确定引擎（PaddleOCR / ML Kit 待选），确定后补进交付包。
- 六组小模型（`zh_uz` / `en_zh` 等专项方向）若需要，单独发对应 tar。
