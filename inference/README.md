# FourLang Inference

该目录是独立的推理工程模块，不修改或依赖任何训练流程。统一加载器支持完整的
Seq2Seq 模型以及可选的 PEFT/LoRA adapter；CLI 支持中文（`zh`）、英文（`en`）、
俄文（`ru`）和乌兹别克文（`uz`）之间的 12 个翻译方向。

## 模型路径

CLI 默认启用自动路由，不再要求传入 `--model-path`。加载器依次探测：

```text
基础模型：
  <项目>/models/small100
  <项目父目录>/models/small100

最终 specialist：
  <项目>/models/final_specialists/en_uz_small100_v1
  <项目>/models/final_specialists/en_zh_v1

ZH→EN 实验 specialist（尚无冻结版本时）：
  <项目>/results/specialists/zh_en/opus_mt_zh_en/exp2_kd_v1/best_model
```

模型不会被移动、合并或覆盖。也可以通过 `--model-path` 传入项目相对路径、绝对路径
或 Hugging Face model ID，此时关闭自动路由并让所有 direction 使用该模型。加载
LoRA 时，用 `--adapter-path` 指定现有 adapter，并必须用 `--model-path` 明确指定基础模型。

完整模型推理不要求安装 PEFT；只有传入 `--adapter-path` 加载 LoRA 时才需要
`peft==0.13.2`。

## 单次翻译

在项目根目录运行：

```powershell
python -m inference --direction zh-en --text "你好，世界"
```

服务器会自动找到 `/root/autodl-tmp/models/small100` 及项目内的 specialist；本地没有
对应 specialist 时会自动回退至 `models/small100`。

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
{"ok":true,"direction":"zh-en","source_language":"zh","target_language":"en","input":"你好，世界","translation":"Hello, world","model_path":"/root/autodl-tmp/fourlang_translation/results/specialists/zh_en/opus_mt_zh_en/exp2_kd_v1/best_model","adapter_path":null,"device":"cuda","latency_ms":123.456,"model_name":"zh_en_exp2_kd_v1","routing_mode":"auto","specialized_model":true,"loaded_model_count":1}
```

需要缩进格式时增加 `--pretty`。

## 交互模式与 direction 切换

省略 `--text` 即进入交互模式：

```powershell
python -m inference --direction zh-en
```

交互命令：

- `/direction ru-zh`：切换翻译方向；新路由首次使用时加载一次，之后复用缓存
- `/directions`：以 JSON 列出全部方向
- `/routes`：显示每个方向当前解析到的模型路径
- `/help`：显示命令
- `/quit`：退出

普通输入行会被当作待翻译文本。事件、结果和错误都写为 JSON；交互提示写到 stderr，
因此 stdout 可以直接交给 JSONL 消费程序。

方向同时接受 `zh-en`、`zh_en`、`zh:en` 和 `zh->en` 写法。查看所有方向且不加载模型：

```powershell
python -m inference --list-directions --pretty
```

查看自动探测结果且不加载模型权重：

```powershell
python -m inference --list-routes --pretty
```

自动路由规则：

- `en-uz`、`uz-en`：优先 `en_uz_small100_v1`
- `en-zh`：优先 `en_zh_v1`（Marian/OPUS）
- `zh-en`：优先冻结的 `zh_en_v1`，否则使用 `exp2_kd_v1/best_model`
- 其余八个方向：通用 SMaLL-100

已加载的模型会在当前进程内缓存，来回切换 direction 不会重复加载同一模型。也可以用
`--base-model-path` 或 `--specialists-root` 覆盖自动探测根目录；部署环境还可设置
`FOURLANG_BASE_MODEL_PATH`、`FOURLANG_EN_UZ_MODEL_PATH`、
`FOURLANG_EN_ZH_MODEL_PATH`、`FOURLANG_ZH_EN_MODEL_PATH`。

## Python API

```python
from inference import ModelRouter, RoutedTranslationEngine

router = ModelRouter(device="auto")
engine = RoutedTranslationEngine(router, direction="zh-en")

result = engine.translate("你好")
engine.set_direction("en-ru")
other_result = engine.translate("Good morning")
```

`ModelRouter` 负责路径探测、模型选择和缓存；`load_translation_model()` 仍可作为加载单个
模型的底层入口。

加载器会自动识别 `models/small100/tokenization_small100.py` 并使用 SMaLL-100 的
目标语言前缀机制；M2M100 使用 `forced_bos_token_id`；Marian/OPUS 使用模型固有的
固定方向。三种架构不需要手动切换推理方式。
