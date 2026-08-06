from __future__ import annotations

import pandas as pd
import pytest

from lol_ai.modeling.rating import EloConfig
from lol_ai.modeling.rating_backtest import (
    calibrate_config,
    estimate_side_advantage,
    run_walk_forward,
    series_level_accuracy,
)


def _synthetic_frame(n_games: int = 40) -> pd.DataFrame:
    lineups = {
        "FORTE": {p: f"f_{p}" for p in ("top", "jng", "mid", "bot", "sup")},
        "FRACO": {p: f"w_{p}" for p in ("top", "jng", "mid", "bot", "sup")},
    }
    rows = []
    for i in range(n_games):
        strong_is_blue = i % 2 == 0
        blue_team = "FORTE" if strong_is_blue else "FRACO"
        red_team = "FRACO" if strong_is_blue else "FORTE"
        rows.append(
            {
                "series_id": f"s{i // 2}",
                "game_number": (i % 2) + 1,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=i),
                "league": "CBLOL",
                "year": 2026,
                "blue_team": blue_team,
                "red_team": red_team,
                "blue_win": 1 if strong_is_blue else 0,
                "winner_team": "FORTE",
                "game_id": f"g{i}",
                "gamelength": 1800,
                "blue_gold_diff": 9000 if strong_is_blue else -9000,
                **{f"blue_{p}_player": lineups[blue_team][p] for p in ("top", "jng", "mid", "bot", "sup")},
                **{f"red_{p}_player": lineups[red_team][p] for p in ("top", "jng", "mid", "bot", "sup")},
            }
        )
    return pd.DataFrame(rows)


def test_vantagem_de_lado_neutra_e_positiva():
    assert estimate_side_advantage(0.5) == pytest.approx(0.0)
    assert estimate_side_advantage(0.55) > 0.0
    assert estimate_side_advantage(0.45) < 0.0


def test_walk_forward_cobre_todos_os_jogos_e_aprende():
    frame = _synthetic_frame()
    engine, probs = run_walk_forward(frame, EloConfig(), impact_lookup={})
    assert len(probs) == len(frame)
    assert probs.index.equals(frame.index)
    assert engine.rating("FORTE") > engine.rating("FRACO")
    # último jogo com FORTE no azul: o modelo já deve favorecer o FORTE
    last_strong_blue = frame[frame["blue_team"] == "FORTE"].index[-1]
    assert probs.loc[last_strong_blue] > 0.6


def test_probabilidade_e_pre_jogo():
    frame = _synthetic_frame(2)
    _, probs = run_walk_forward(frame, EloConfig(), impact_lookup={})
    assert probs.iloc[0] == pytest.approx(0.5)


def test_calibracao_retorna_config_com_vantagem():
    frame = _synthetic_frame()
    validation_index = frame.index[-10:]
    config = calibrate_config(frame, validation_index, impact_lookup={}, side_advantage=10.0)
    assert isinstance(config, EloConfig)
    assert config.side_advantage == pytest.approx(10.0)


import numpy as np

from lol_ai.modeling.rating_backtest import blend_probabilities, fit_draft_weight


def test_blend_peso_zero_devolve_rating():
    assert blend_probabilities(0.7, 0.2, 0.0) == pytest.approx(0.7)


def test_blend_draft_neutro_nao_muda():
    assert blend_probabilities(0.7, 0.5, 1.0) == pytest.approx(0.7)


def test_blend_draft_favoravel_aumenta():
    assert blend_probabilities(0.6, 0.8, 1.0) > 0.6


def test_fit_draft_weight_ignora_draft_ruidoso():
    rng = np.random.default_rng(42)
    y = pd.Series(rng.integers(0, 2, size=200))
    p_rating = pd.Series([0.8 if value == 1 else 0.2 for value in y], index=y.index)
    p_draft = pd.Series(rng.uniform(0.05, 0.95, size=200), index=y.index)
    assert fit_draft_weight(p_rating, p_draft, y) == pytest.approx(0.0)


def test_fit_draft_weight_usa_draft_informativo():
    rng = np.random.default_rng(42)
    y = pd.Series(rng.integers(0, 2, size=200))
    p_rating = pd.Series(0.5, index=y.index)
    p_draft = pd.Series([0.85 if value == 1 else 0.15 for value in y], index=y.index)
    assert fit_draft_weight(p_rating, p_draft, y) > 0.5


from lol_ai.modeling.player_elo import PlayerEloConfig
from lol_ai.modeling.rating_backtest import run_walk_forward_with_players


def test_walk_forward_com_jogadores_mantem_probabilidades():
    frame = _synthetic_frame()
    team_engine, player_engine, probs = run_walk_forward_with_players(
        frame, EloConfig(), PlayerEloConfig(), performance_lookup={}
    )
    assert len(probs) == len(frame)
    assert team_engine.rating("FORTE") > team_engine.rating("FRACO")
    # jogadores do time vencedor sobem, do perdedor descem
    assert player_engine.rating("f_mid", "mid") > 1500.0 > player_engine.rating("w_mid", "mid")


def test_troca_por_desconhecido_penaliza_time_vencedor():
    frame = _synthetic_frame()
    # último jogo: FORTE troca o mid por um novato sem histórico (Elo 1500)
    last = frame.index[-2]  # FORTE é blue nos índices pares
    frame.loc[last, "blue_mid_player"] = "novato"
    config = EloConfig(roster_regression_per_player=0.0, impact_scale=0.5)
    team_engine, _, _ = run_walk_forward_with_players(
        frame, config, PlayerEloConfig(), performance_lookup={}
    )
    entries = [h for h in team_engine.history if h["team"] == "FORTE" and h["roster_changes"] > 0]
    assert entries, "a troca deveria ter sido registrada"
    # primeira troca = titular (Elo > 1500) sai, novato (1500) entra -> penalidade;
    # no jogo seguinte o titular volta, o que conta como nova troca positiva
    assert entries[0]["roster_adjustment"] < 0


def test_mov_acelera_convergencia_com_vitorias_dominantes():
    frame = _synthetic_frame()
    # margem sintética: 9000 de ouro em 30min = 300 gpm; referência 150 -> dominante
    _, probs_sem_mov = run_walk_forward(frame, EloConfig(), impact_lookup={})
    engine_mov, probs_mov = run_walk_forward(
        frame, EloConfig(mov_weight=1.0, mov_reference=150.0), impact_lookup={}
    )
    assert engine_mov.rating("FORTE") > 1500.0
    ultimo = frame[frame["blue_team"] == "FORTE"].index[-1]
    assert probs_mov.loc[ultimo] > probs_sem_mov.loc[ultimo]


def test_margem_ausente_nao_quebra():
    frame = _synthetic_frame().drop(columns=["gamelength", "blue_gold_diff"])
    _, probs = run_walk_forward(frame, EloConfig(mov_weight=1.0, mov_reference=150.0), impact_lookup={})
    assert len(probs) == len(frame)


def test_acuracia_por_serie():
    frame = _synthetic_frame(8)
    probs = pd.Series([0.9 if row.blue_team == "FORTE" else 0.1 for row in frame.itertuples()], index=frame.index)
    accuracy = series_level_accuracy(frame, probs, frame.index)
    assert accuracy == pytest.approx(1.0)
