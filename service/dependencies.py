from __future__ import annotations
from functools import lru_cache
from inference.engine import TranslatorEngine

@lru_cache(maxsize=1)
def get_translator_engine() -> TranslatorEngine:
    return TranslatorEngine()
