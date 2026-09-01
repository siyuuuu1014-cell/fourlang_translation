# FourLang Inference

该目录是独立的推理工程模块，不修改或依赖任何训练流程。统一加载器支持完整的
Seq2Seq 模型以及可选的 PEFT/LoRA adapter；CLI 支持中文（`zh`）、英文（`en`）、
俄文（`ru`）和乌兹别克文（`uz`）之间的 12 个翻译方向。

## 模型路径

默认继续使用项目中现有的模型目录：

```text
models/small100
```

模型不会被移动、合并或覆盖。也可以通过 `--model-path` 传入项目相对路径、绝对路径
或 Hugging Face model ID。加载 LoRA 时，用 `--adapter-path` 指定现有 adapter，并用
`--model-path` 明确指定它对应的基础模型。

完整模型推理不要求安装 PEFT；只有传入 `--adapter-path` 加载 LoRA 时才需要
`peft==0.13.2`。

## 单次翻译

在项目根目录运行：

```powershell
python -m inference --model-path models/small100 --direction zh-en --text "你好，世界"
```

切换模型、adapter 或设备：

```powershell
python -m inference `
  --model-path models/small100 `
  --adapter-path models/lora/exp1_small100_uz `
  --direction en-uz `
  --device auto `
  --text "Good morning"
```

标准输出为 JSON：

```json
{"ok":true,"direction":"zh-en","source_language":"zh","target_language":"en","input":"你好，世界","translation":"Hello, world","model_path":"models/small100","adapter_path":null,"device":"cuda","latency_ms":123.456}
```

需要缩进格式时增加 `--pretty`。

## 交互模式与 direction 切换

省略 `--text` 即进入交互模式：

```powershell
python -m inference --direction zh-en
```

交互命令：

- `/direction ru-zh`：切换翻译方向，不重新加载模型
- `/directions`：以 JSON 列出全部方向
- `/help`：显示命令
- `/quit`：退出

普通输入行会被当作待翻译文本。事件、结果和错误都写为 JSON；交互提示写到 stderr，
因此 stdout 可以直接交给 JSONL 消费程序。

方向同时接受 `zh-en`、`zh_en`、`zh:en` 和 `zh->en` 写法。查看所有方向且不加载模型：

```powershell
python -m inference --list-directions --pretty
```

## Python API

```python
from inference import TranslationEngine, load_translation_model

loaded = load_translation_model("models/small100", device="auto")
engine = TranslationEngine(loaded, direction="zh-en")

result = engine.translate("你好")
engine.set_direction("en-ru")
other_result = engine.translate("Good morning")
```

`load_translation_model()` 是统一模型加载入口；一次加载后可重复切换 direction。
