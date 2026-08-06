from __future__ import annotations

import pytest

from lol_ai.modeling.player_elo import (
    LivePlayerRatingLookup,
    PlayerEloConfig,
    PlayerEloEngine,
)

LINEUP = {"top": "T1", "jng": "J1", "mid": "M1", "bot": "B1", "sup": "S1"}


def _side(engine, *, win, expected=0.5, game_id="g1", team="FURIA", opponent="LEV", lineup=None):
    engine.process_side(
        date="2026-01-01",
        game_id=game_id,
        team=team,
        opponent=opponent,
        lineup=lineup or LINEUP,
        expected=expected,
        win=win,
    )


def test_vitoria_sobe_derrota_desce():
    engine = PlayerEloEngine(PlayerEloConfig(), performance_lookup={})
    _side(engine, win=True)
    assert engine.rating("M1", "mid") > 1500.0
    engine2 = PlayerEloEngine(PlayerEloConfig(), performance_lookup={})
    _side(engine2, win=False)
    assert engine2.rating("M1", "mid") < 1500.0


def test_quem_carrega_ganha_mais_na_vitoria():
    lookup = {("g1", "m1", "mid"): 90.0, ("g1", "t1", "top"): 10.0}
    engine = PlayerEloEngine(PlayerEloConfig(), performance_lookup=lookup)
    _side(engine, win=True)
    carry_gain = engine.rating("M1", "mid") - 1500.0
    passenger_gain = engine.rating("T1", "top") - 1500.0
    assert carry_gain > passenger_gain > 0


def test_quem_joga_mal_perde_mais_na_derrota():
    lookup = {("g1", "m1", "mid"): 90.0, ("g1", "t1", "top"): 10.0}
    engine = PlayerEloEngine(PlayerEloConfig(), performance_lookup=lookup)
    _side(engine, win=False)
    good_loss = 1500.0 - engine.rating("M1", "mid")
    bad_loss = 1500.0 - engine.rating("T1", "top")
    assert bad_loss > good_loss > 0


def test_ganhar_de_time_fraco_rende_pouco():
    engine_favorito = PlayerEloEngine(PlayerEloConfig(), performance_lookup={})
    _side(engine_favorito, win=True, expected=0.9)
    engine_parelho = PlayerEloEngine(PlayerEloConfig(), performance_lookup={})
    _side(engine_parelho, win=True, expected=0.5)
    assert (engine_favorito.rating("M1", "mid") - 1500.0) < (engine_parelho.rating("M1", "mid") - 1500.0)


def test_perder_de_time_fraco_pune_muito():
    engine = PlayerEloEngine(PlayerEloConfig(), performance_lookup={})
    _side(engine, win=False, expected=0.9)
    big_loss = 1500.0 - engine.rating("M1", "mid")
    engine2 = PlayerEloEngine(PlayerEloConfig(), performance_lookup={})
    _side(engine2, win=False, expected=0.5)
    normal_loss = 1500.0 - engine2.rating("M1", "mid")
    assert big_loss > normal_loss


def test_transferencia_mantem_rating():
    engine = PlayerEloEngine(PlayerEloConfig(), performance_lookup={})
    _side(engine, win=True, team="FURIA")
    rating_after_furia = engine.rating("M1", "mid")
    _side(engine, win=True, game_id="g2", team="LOUD")
    assert engine.rating("M1", "mid") > rating_after_furia
    entry = [h for h in engine.history if h["player"] == "M1"][-1]
    assert entry["team"] == "LOUD"


def test_sem_performance_usa_neutro():
    engine = PlayerEloEngine(PlayerEloConfig(k=16.0), performance_lookup={})
    _side(engine, win=True, expected=0.5)
    # base = 16 * 0.5 = 8, multiplicador neutro = 1.0
    assert engine.rating("M1", "mid") == pytest.approx(1508.0)


def test_lookup_ao_vivo_devolve_1500_para_desconhecido():
    engine = PlayerEloEngine(PlayerEloConfig(), performance_lookup={})
    lookup = LivePlayerRatingLookup(engine)
    assert lookup.get(("desconhecido", "mid"), 50.0) == 1500.0
    _side(engine, win=True)
    assert lookup.get(("m1", "mid"), 50.0) > 1500.0
