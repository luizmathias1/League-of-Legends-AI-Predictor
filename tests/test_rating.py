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


from lol_ai.modeling.rating import RatingEngine  # noqa: E402

LINEUP_A = {"top": "a1", "jng": "a2", "mid": "a3", "bot": "a4", "sup": "a5"}
LINEUP_B = {"top": "b1", "jng": "b2", "mid": "b3", "bot": "b4", "sup": "b5"}


def _play(engine, blue, red, blue_win, year=2026, blue_lineup=None, red_lineup=None):
    return engine.process_game(
        date="2026-01-01",
        league="CBLOL",
        year=year,
        blue_team=blue,
        red_team=red,
        blue_lineup=blue_lineup or LINEUP_A,
        red_lineup=red_lineup or LINEUP_B,
        blue_win=blue_win,
    )


def test_probabilidade_retornada_e_pre_jogo():
    engine = RatingEngine(EloConfig())
    prob = _play(engine, "FURIA", "LEV", blue_win=True)
    assert prob == pytest.approx(0.5)  # ambos 1500 antes do jogo


def test_vitorias_sobem_o_rating_do_vencedor():
    engine = RatingEngine(EloConfig())
    for _ in range(5):
        _play(engine, "FURIA", "LEV", blue_win=True)
    assert engine.rating("FURIA") > 1500.0 > engine.rating("LEV")


def test_derrota_inesperada_penaliza_mais_que_vitoria_esperada_premia():
    engine = RatingEngine(EloConfig())
    for _ in range(10):
        _play(engine, "FURIA", "LEV", blue_win=True)
    rating_before = engine.rating("FURIA")
    _play(engine, "FURIA", "LEV", blue_win=True)
    small_gain = engine.rating("FURIA") - rating_before
    rating_before = engine.rating("FURIA")
    _play(engine, "FURIA", "LEV", blue_win=False)
    big_loss = rating_before - engine.rating("FURIA")
    assert big_loss > small_gain


def test_virada_de_temporada_regride_a_media():
    engine = RatingEngine(EloConfig(season_carry=0.5))
    for _ in range(10):
        _play(engine, "FURIA", "LEV", blue_win=True, year=2025)
    rating_2025 = engine.rating("FURIA")
    _play(engine, "FURIA", "LEV", blue_win=True, year=2026)
    entry = [h for h in engine.history if h["team"] == "FURIA"][-1]
    expected_after_carry = 1500.0 + (rating_2025 - 1500.0) * 0.5
    assert entry["season_adjustment"] == pytest.approx(expected_after_carry - rating_2025, abs=0.01)


def test_troca_de_jogador_regride_e_ajusta_por_impacto():
    lookup = {("a2", "jng"): 40.0, ("novo", "jng"): 70.0}
    engine = RatingEngine(
        EloConfig(roster_regression_per_player=0.10, impact_scale=1.0),
        impact_lookup=lookup,
    )
    for _ in range(10):
        _play(engine, "FURIA", "LEV", blue_win=True)
    rating_before = engine.rating("FURIA")
    new_lineup = dict(LINEUP_A, jng="novo")
    _play(engine, "FURIA", "LEV", blue_win=True, blue_lineup=new_lineup)
    entry = [h for h in engine.history if h["team"] == "FURIA"][-1]
    assert entry["roster_changes"] == 1
    regressed = 1500.0 + (rating_before - 1500.0) * 0.9
    expected_adjustment = (regressed - rating_before) + 1.0 * (70.0 - 40.0)
    assert entry["roster_adjustment"] == pytest.approx(expected_adjustment, abs=0.01)


def test_mov_desligado_por_padrao():
    config = EloConfig()
    assert config.mov_weight == 0.0
    engine_a = RatingEngine(EloConfig())
    engine_b = RatingEngine(EloConfig())
    _play(engine_a, "FURIA", "LEV", blue_win=True)
    engine_b.process_game(
        date="2026-01-01", league="CBLOL", year=2026,
        blue_team="FURIA", red_team="LEV",
        blue_lineup=LINEUP_A, red_lineup=LINEUP_B,
        blue_win=True, margin_gpm=500.0,
    )
    assert engine_a.rating("FURIA") == pytest.approx(engine_b.rating("FURIA"))


def _play_mov(margin_gpm):
    engine = RatingEngine(EloConfig(mov_weight=1.0, mov_reference=150.0))
    engine.process_game(
        date="2026-01-01", league="CBLOL", year=2026,
        blue_team="FURIA", red_team="LEV",
        blue_lineup=LINEUP_A, red_lineup=LINEUP_B,
        blue_win=True, margin_gpm=margin_gpm,
    )
    return engine


def test_vitoria_dominante_rende_mais_que_apertada():
    dominante = _play_mov(400.0)
    apertada = _play_mov(40.0)
    assert dominante.rating("FURIA") > apertada.rating("FURIA") > 1500.0


def test_margem_na_referencia_e_neutra():
    referencia = _play_mov(150.0)
    sem_margem = _play_mov(None)
    assert referencia.rating("FURIA") == pytest.approx(sem_margem.rating("FURIA"))


def test_margem_extrema_e_limitada():
    extrema = _play_mov(100000.0)
    base_gain = 0.5 * 24.0  # k padrão 24? não: usa EloConfig().k
    config = EloConfig()
    max_gain = config.k * 0.5 * 2.0  # multiplicador máximo 2x
    assert extrema.rating("FURIA") - 1500.0 <= max_gain + 1e-9


def test_mov_preserva_soma_zero():
    engine = _play_mov(400.0)
    assert engine.rating("FURIA") - 1500.0 == pytest.approx(1500.0 - engine.rating("LEV"))


def test_primeiro_jogo_nao_conta_troca():
    engine = RatingEngine(EloConfig())
    _play(engine, "FURIA", "LEV", blue_win=True)
    entry = [h for h in engine.history if h["team"] == "FURIA"][-1]
    assert entry["roster_changes"] == 0
    assert entry["roster_adjustment"] == 0.0
