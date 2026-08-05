from __future__ import annotations

import pytest

from lol_ai.modeling.rating import EloConfig, elo_delta, expected_score


def test_expectativa_igual_para_ratings_iguais():
    assert expected_score(1500.0, 1500.0) == pytest.approx(0.5)


def test_expectativa_soma_um():
    assert expected_score(1600.0, 1450.0) + expected_score(1450.0, 1600.0) == pytest.approx(1.0)


def test_favorito_tem_expectativa_maior():
    assert expected_score(1700.0, 1300.0) > 0.9


def test_vantagem_de_lado_aumenta_expectativa():
    assert expected_score(1500.0, 1500.0, advantage_a=30.0) > 0.5


def test_vitoria_esperada_rende_pouco_e_derrota_inesperada_custa_caro():
    expected = expected_score(1700.0, 1300.0)  # ~0.91
    gain_expected_win = elo_delta(32.0, 1.0, expected)
    loss_unexpected = elo_delta(32.0, 0.0, expected)
    assert 0 < gain_expected_win < 5
    assert loss_unexpected < -25


def test_config_padrao():
    config = EloConfig()
    assert config.initial_rating == 1500.0
    assert config.k > 0
