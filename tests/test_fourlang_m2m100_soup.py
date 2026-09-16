import tomllib
from pathlib import Path

import pytest

from scripts.pipeline_v3 import fourlang_m2m100_soup as soup


def metrics(value, *, ru_uz=None, zh_uz=None, worst=None):
    rows = {
        direction: {"bleu": value, "chrf2": value, "samples": 1}
        for direction in soup.directions()
    }
    if ru_uz is not None:
        rows["ru-uz"]["chrf2"] = ru_uz
    if zh_uz is not None:
        rows["zh-uz"]["chrf2"] = zh_uz
    summary = soup.summarize_metrics(rows)
    if worst is not None:
        summary["worst_chrf2"] = worst
    return {"metrics": rows, "summary": summary}


def selection_settings():
    return {
        "regression_tolerance": 0.1,
        "regression_sum_penalty": 2.0,
        "maximum_regression_penalty": 3.0,
        "hard_max_direction_regression": 0.2,
        "minimum_ru_uz_recovery_from_exp4": 0.3,
        "maximum_zh_uz_regression": 0.1,
        "minimum_macro_chrf2": 38.35,
        "minimum_worst_chrf2": 17.07,
    }


def test_alpha_ids_are_stable():
    assert soup.alpha_id(0.25) == "exp4_025"
    assert soup.alpha_id(0.5) == "exp4_050"
    assert soup.alpha_id(0.75) == "exp4_075"


def test_config_locks_the_approved_three_candidate_sweep():
    with Path("configs/multilingual/fourlang_m2m100_soup_v1.toml").open("rb") as stream:
        config = tomllib.load(stream)
    assert config["experiment"]["alphas_exp4"] == [0.25, 0.5, 0.75]
    assert config["selection"]["hard_max_direction_regression"] == pytest.approx(0.2)
    assert config["selection"]["minimum_ru_uz_recovery_from_exp4"] == pytest.approx(0.3)


def test_penalty_rejects_large_single_direction_regression():
    exp4 = metrics(38.4, ru_uz=39.5, worst=17.1)
    targeted = metrics(38.4, ru_uz=40.0, worst=17.1)
    candidate = metrics(38.4, ru_uz=39.7, worst=17.1)
    result = soup.score_candidate(candidate, exp4, targeted, selection_settings())
    assert result["maximum_direction_regression"] == pytest.approx(0.3)
    assert result["passed"] is False
    assert result["penalty"] > 0


def test_candidate_passes_only_when_recovery_and_all_limits_hold():
    exp4 = metrics(38.4, ru_uz=39.5, zh_uz=38.2, worst=17.1)
    targeted = metrics(38.4, ru_uz=39.9, zh_uz=38.2, worst=17.1)
    candidate = metrics(38.4, ru_uz=39.82, zh_uz=38.2, worst=17.1)
    result = soup.score_candidate(candidate, exp4, targeted, selection_settings())
    assert result["constraints"]["ru_uz_recovery_from_exp4_at_least_minimum"] is True
    assert result["passed"] is True


def test_no_candidate_is_promoted_implicitly():
    # The comparison report records selection only; production promotion is intentionally absent.
    assert "promotion" not in soup.run.__name__
