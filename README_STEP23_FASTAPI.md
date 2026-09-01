# FourLang Step23 - FastAPI Service V1

## Purpose

Wrap the completed Step22 `TranslatorEngine` with a stable HTTP API.

This step does not train, quantize, convert, or modify model weights.

## Files

```text
fourlang_translation/
├── service/
│   ├── __init__.py
│   ├── app.py
│   ├── dependencies.py
│   └── schemas.py
└── scripts/
    └── service/
        └── run_api.py
```

## Dependencies

Use the existing server environment:

```bash
source /root/autodl-tmp/venvs/small100_student/bin/activate
python -m pip install fastapi "uvicorn[standard]"
```

## Local syntax checks

```powershell
python -m py_compile service\__init__.py
python -m py_compile service\schemas.py
python -m py_compile service\dependencies.py
python -m py_compile service\app.py
python -m py_compile scripts\service\run_api.py
```

## Server run

```bash
cd /root/autodl-tmp/fourlang_translation
source /root/autodl-tmp/venvs/small100_student/bin/activate

python scripts/service/run_api.py --host 0.0.0.0 --port 8000
```

Use one worker for now. Multiple workers would create independent PyTorch caches and duplicate GPU model memory.

## Endpoints

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Models

```bash
curl http://127.0.0.1:8000/models
```

### EN -> ZH

```bash
curl -X POST http://127.0.0.1:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"source_lang":"en","target_lang":"zh","text":"Where is the nearest hospital?"}'
```

### EN -> UZ

```bash
curl -X POST http://127.0.0.1:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"source_lang":"en","target_lang":"uz","text":"Where is the nearest hospital?"}'
```

### UZ -> EN

```bash
curl -X POST http://127.0.0.1:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"source_lang":"uz","target_lang":"en","text":"Eng yaqin shifoxona qayerda?"}'
```

Swagger UI:

```text
http://SERVER_IP:8000/docs
```

## Completion criteria

Step23 is complete when `/health`, `/models`, and `/translate` work for `en_zh`, `en_uz`, and `uz_en`, and repeated calls reuse the cached model instead of reloading it.
