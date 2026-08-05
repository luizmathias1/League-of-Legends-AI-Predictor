from __future__ import annotations

import pytest

from lol_ai.modeling.series import series_probabilities


def test_bo3_com_p_meio_a_meio():
    result = series_probabilities([0.5], best_of=3)
    assert result["a_series_win"] == pytest.approx(0.5)
    assert sum(result["score_probabilities"].values()) == pytest.approx(1.0)


def test_bo5_bate_com_formula_fechada():
    p, q = 0.6, 0.4
    result = series_probabilities([p], best_of=5)
    assert result["score_probabilities"]["3x0"] == pytest.approx(p**3)
    assert result["score_probabilities"]["3x1"] == pytest.approx(3 * p**3 * q)
    assert result["score_probabilities"]["3x2"] == pytest.approx(6 * p**3 * q**2)
    assert result["a_series_win"] == pytest.approx(p**3 + 3 * p**3 * q + 6 * p**3 * q**2)


def test_probabilidades_diferentes_por_jogo_sao_usadas():
    uniform = series_probabilities([0.5, 0.5, 0.5], best_of=3)
    strong_late = series_probabilities([0.5, 0.9, 0.9], best_of=3)
    assert strong_late["a_series_win"] > uniform["a_series_win"]


def test_placar_mais_provavel():
    result = series_probabilities([0.9], best_of=3)
    assert result["most_likely_score"] == "2x0"


def test_best_of_invalido():
    with pytest.raises(ValueError):
        series_probabilities([0.5], best_of=4)
