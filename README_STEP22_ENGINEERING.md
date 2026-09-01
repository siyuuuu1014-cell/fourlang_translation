# FourLang Step22 - Unified Inference Engineering V1

## 1. Current frozen / accepted models

### EN -> ZH
Experimental source:
`results/specialists/en_zh/opus_mt_en_zh/exp2_kd_v2/epoch_models/epoch_1`

Frozen deployment path:
`models/final_specialists/en_zh_v1`

Registry status:
`ready`

Architecture:
`Marian`

### EN <-> UZ
Accepted experimental model:
`results/student/small100/exp2_distillation_v1/best_model`

Deployment target:
`models/final_specialists/en_uz_small100_v1`

Registry status in this package:
`ready`

The model is already accepted scientifically. Runtime `--list` shows only entries whose frozen deployment directory actually exists on the current machine. Run the freeze script on the server to materialize the deployment copy; no tracked config is edited on the server.

---

## 2. Files to place in the repository

Copy the package contents into the repository root:

```text
fourlang_translation/
├── inference/
│   ├── __init__.py
│   ├── paths.py
│   ├── registry.py
│   ├── loaders.py
│   ├── engine.py
│   └── cli.py
│
├── models/
│   └── model_registry.json
│
└── scripts/
    ├── inference/
    │   └── check_registry.py
    │
    └── model_management/
        └── freeze_en_uz_exp2.py
```

Do NOT place model weights in Git.

---

## 3. Path policy

Local repository root:

`D:\dev\projects\fourlang_translation`

Server repository root:

`/root/autodl-tmp/fourlang_translation`

Runtime code contains no hard-coded repository absolute paths.

Model paths inside `models/model_registry.json` are repository-relative, for example:

`models/final_specialists/en_zh_v1`

The runtime resolves them against the actual repository root.

---

## 4. Local checks

From:

`D:\dev\projects\fourlang_translation`

Run:

```powershell
python -m py_compile inference\__init__.py
python -m py_compile inference\paths.py
python -m py_compile inference\registry.py
python -m py_compile inference\loaders.py
python -m py_compile inference\engine.py
python -m py_compile inference\cli.py
python -m py_compile scripts\inference\check_registry.py
python -m py_compile scripts\model_management\freeze_en_uz_exp2.py

python -m inference.cli --list
python -m inference.cli --list-all
```

It is expected that large server-only model paths may not exist locally.

Do not copy large model weights to the local repository just to satisfy static tests.

---

## 5. Git workflow

```powershell
git status
git diff

git add inference
git add models\model_registry.json
git add scripts\inference
git add scripts\model_management

git commit -m "feat: add unified translation inference engine"
git push
```

---

## 6. Server synchronization

```bash
cd /root/autodl-tmp/fourlang_translation
git pull --ff-only

source /root/autodl-tmp/venvs/small100_student/bin/activate
```

Check:

```bash
python -m inference.cli --list
python -m inference.cli --list-all
python scripts/inference/check_registry.py
```

Before freezing EN-UZ, `--list-all` shows all approved registry entries plus `path_exists`. `--list` shows only approved entries whose frozen model directory exists on that machine.

---

## 7. Test EN -> ZH

```bash
python -m inference.cli \
  --direction en_zh \
  --warmup \
  --text "Where is the nearest hospital?"
```

JSON:

```bash
python -m inference.cli \
  --direction en_zh \
  --text "I love Beijing." \
  --json
```

Interactive:

```bash
python -m inference.cli --direction en_zh
```

---

## 8. Freeze and activate EN <-> UZ

First confirm the accepted Exp2 exists:

```bash
ls -lah \
results/student/small100/exp2_distillation_v1/best_model
```

Then:

```bash
python scripts/model_management/freeze_en_uz_exp2.py
```

This copies:

```text
results/student/small100/exp2_distillation_v1/best_model
```

to:

```text
models/final_specialists/en_uz_small100_v1
```

The script does not modify `models/model_registry.json`; this keeps Git/local code as the single source of truth. The registry already points at the frozen destination, so once the directory exists the directions become runtime-available automatically.

Then verify:

```bash
python -m inference.cli --list
```

Expected:

```text
en_uz
en_zh
uz_en
```

---

## 9. Test EN -> UZ

```bash
python -m inference.cli \
  --direction en_uz \
  --warmup \
  --text "Where is the nearest hospital?"
```

## 10. Test UZ -> EN

```bash
python -m inference.cli \
  --direction uz_en \
  --warmup \
  --text "Eng yaqin shifoxona qayerda?"
```

---

## 11. Engineering boundaries

`results/`
- experiment outputs
- checkpoints
- evaluation artifacts

`models/final_specialists/`
- frozen deployment candidates
- not training workspace

`inference/`
- production-oriented PyTorch inference layer
- does not train
- does not modify model weights

`models/model_registry.json`
- maps logical translation directions to frozen model locations

The next stage after Step22 is an API layer (e.g. FastAPI), not additional model training.


## 12. Lightweight registry inspection

`python -m inference.cli --list` and `--list-all` use only the registry layer.
They do not import model loaders and do not load any model weights. This is
intentional so local development can validate paths/configuration even when
server-only model weights are absent.
