from __future__ import annotations
from functools import lru_cache

from inference.pair_service import PairModelService


@lru_cache(maxsize=1)
def get_pair_service() -> PairModelService:
    return PairModelService()
