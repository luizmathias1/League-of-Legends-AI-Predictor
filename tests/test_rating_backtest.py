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


def test_acuracia_por_serie():
    frame = _synthetic_frame(8)
    probs = pd.Series([0.9 if row.blue_team == "FORTE" else 0.1 for row in frame.itertuples()], index=frame.index)
    accuracy = series_level_accuracy(frame, probs, frame.index)
    assert accuracy == pytest.approx(1.0)
